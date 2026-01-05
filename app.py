import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.data_loader import get_market_data, fetch_portfolio_data, calculate_returns
from src.portfolio import Portfolio
from src.risk_metrics import RiskCalculator
from src.backtesting import VaRBacktester
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Portfolio Risk Management", 
                   page_icon="📊", 
                   layout="wide")

st.title("📊 Portfolio Risk Management System")
st.markdown("**Value-at-Risk (VaR) and Conditional VaR (CVaR) Analysis**")

# Sidebar - Portfolio Configuration
st.sidebar.header("Portfolio Configuration")

# Default tickers
default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC', 
                   'GS', 'XOM', 'JNJ', 'PG', 'KO']

tickers_input = st.sidebar.text_area(
    "Enter Tickers (comma-separated)",
    value=', '.join(default_tickers)
)
tickers = [t.strip().upper() for t in tickers_input.split(',')]

# Time period
years = st.sidebar.slider("Historical Data (Years)", 1, 5, 3)

# Portfolio value
portfolio_value = st.sidebar.number_input(
    "Portfolio Value ($)",
    min_value=10000,
    value=1000000,
    step=10000
)

# Confidence level
confidence_level = st.sidebar.slider(
    "Confidence Level (%)",
    90, 99, 95
) / 100

# Monte Carlo simulations
n_simulations = st.sidebar.number_input(
    "Monte Carlo Simulations",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000
)

# Load data button
if st.sidebar.button("Calculate Risk Metrics"):
    with st.spinner("Loading market data..."):
        try:
            # Fetch data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            prices = fetch_portfolio_data(tickers, start_date, end_date)
            returns = calculate_returns(prices)
            
            # Create portfolio
            portfolio = Portfolio(returns)
            port_returns = portfolio.portfolio_returns()
            
            # Store in session state
            st.session_state['portfolio'] = portfolio
            st.session_state['port_returns'] = port_returns
            st.session_state['prices'] = prices
            st.session_state['returns'] = returns
            st.session_state['confidence_level'] = confidence_level
            st.session_state['portfolio_value'] = portfolio_value
            st.session_state['n_simulations'] = n_simulations
            
            st.sidebar.success("Data loaded successfully!")
            
        except Exception as e:
            st.sidebar.error(f"Error loading data: {str(e)}")

# Main content
if 'portfolio' in st.session_state:
    portfolio = st.session_state['portfolio']
    port_returns = st.session_state['port_returns']
    prices = st.session_state['prices']
    returns = st.session_state['returns']
    
    # Portfolio Overview
    st.header("Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats = portfolio.get_statistics()
    
    with col1:
        st.metric("Expected Return (Annual)", 
                  f"{stats['Expected Return (Annual)']*100:.2f}%")
    with col2:
        st.metric("Volatility (Annual)", 
                  f"{stats['Volatility (Annual)']*100:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.3f}")
    with col4:
        st.metric("Skewness", f"{stats['Skewness']:.3f}")
    
    # Portfolio value chart
    st.subheader("Portfolio Value Over Time")
    portfolio_val = portfolio.portfolio_value(st.session_state['portfolio_value'])
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=portfolio_val.index,
        y=portfolio_val.values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='royalblue', width=2)
    ))
    fig1.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig1, width='stretch')
    
    # VaR Metrics
    st.header("Value-at-Risk (VaR) Analysis")
    
    calc = RiskCalculator(port_returns, st.session_state['confidence_level'])
    metrics, dollar_metrics = calc.get_all_metrics(
        st.session_state['portfolio_value'],
        st.session_state['n_simulations']
    )
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Historical Method")
        st.metric("VaR", f"${dollar_metrics['Historical VaR']:,.0f}")
        st.metric("CVaR", f"${dollar_metrics['Historical CVaR']:,.0f}")
    
    with col2:
        st.subheader("Parametric Method")
        st.metric("VaR", f"${dollar_metrics['Parametric VaR']:,.0f}")
        st.metric("CVaR", f"${dollar_metrics['Parametric CVaR']:,.0f}")
    
    with col3:
        st.subheader("Monte Carlo Method")
        st.metric("VaR", f"${dollar_metrics['Monte Carlo VaR']:,.0f}")
        st.metric("CVaR", f"${dollar_metrics['Monte Carlo CVaR']:,.0f}")
    
    st.info(f"""
    **Interpretation**: At {st.session_state['confidence_level']*100:.0f}% confidence level:
    - **VaR**: Maximum expected loss over 1 day under normal market conditions
    - **CVaR**: Expected loss given that VaR threshold is exceeded (tail risk)
    """)
    
    # VaR Comparison Chart
    st.subheader("VaR Method Comparison")
    
    var_df = pd.DataFrame({
        'Method': ['Historical', 'Parametric', 'Monte Carlo'],
        'VaR': [dollar_metrics['Historical VaR'],
                dollar_metrics['Parametric VaR'],
                dollar_metrics['Monte Carlo VaR']],
        'CVaR': [dollar_metrics['Historical CVaR'],
                 dollar_metrics['Parametric CVaR'],
                 dollar_metrics['Monte Carlo CVaR']]
    })
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=var_df['Method'],
        y=var_df['VaR'],
        name='VaR',
        marker_color='indianred'
    ))
    fig2.add_trace(go.Bar(
        x=var_df['Method'],
        y=var_df['CVaR'],
        name='CVaR',
        marker_color='darkred'
    ))
    fig2.update_layout(
        barmode='group',
        xaxis_title="Method",
        yaxis_title="Risk Measure ($)",
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig2, width='stretch')
    
    # VaR Confidence Intervals (Monte Carlo)
    st.subheader("VaR Confidence Intervals (Monte Carlo)")
    
    confidence_levels = [0.90, 0.95, 0.99]
    mc_vars = []
    
    for cl in confidence_levels:
        calc_temp = RiskCalculator(port_returns, cl)
        var_val = calc_temp.monte_carlo_var(n_simulations=st.session_state['n_simulations'])
        mc_vars.append(var_val * st.session_state['portfolio_value'])
    
    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(
        x=[f"{cl*100:.0f}%" for cl in confidence_levels],
        y=mc_vars,
        mode='lines+markers',
        name='VaR',
        line=dict(color='darkred', width=3),
        marker=dict(size=10)
    ))
    fig_ci.update_layout(
        xaxis_title="Confidence Level",
        yaxis_title="VaR ($)",
        height=350,
        template='plotly_white'
    )
    st.plotly_chart(fig_ci, width='stretch')
    
    # Returns Distribution
    st.header("Returns Distribution Analysis")
    
    fig3 = go.Figure()
    
    # Histogram
    fig3.add_trace(go.Histogram(
        x=port_returns*100,
        nbinsx=50,
        name='Returns Distribution',
        marker_color='lightblue'
    ))
    
    # VaR line
    fig3.add_vline(
        x=-metrics['Historical VaR']*100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR ({st.session_state['confidence_level']*100:.0f}%)"
    )
    
    fig3.update_layout(
        xaxis_title="Daily Returns (%)",
        yaxis_title="Frequency",
        height=400,
        template='plotly_white',
        showlegend=True
    )
    st.plotly_chart(fig3, width='stretch')
    
    # Backtesting
    st.header("VaR Model Backtesting")
    
    backtester = VaRBacktester(
        port_returns,
        pd.Series(metrics['Historical VaR'], index=port_returns.index),
        st.session_state['confidence_level']
    )
    
    kupiec = backtester.kupiec_test()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Violations", kupiec['n_violations'])
        st.metric("Violation Rate", f"{kupiec['violation_rate']*100:.2f}%")
    
    with col2:
        st.metric("Expected Rate", f"{kupiec['expected_rate']*100:.2f}%")
        st.metric("P-Value", f"{kupiec['p_value']:.4f}")
    
    with col3:
        status = "❌ Rejected" if kupiec['reject_null'] else "✅ Accepted"
        st.metric("Model Accuracy", status)
    
    st.info("""
    **Kupiec POF Test**: Tests if the number of VaR violations matches the expected rate.
    - **Null Hypothesis**: VaR model is accurate
    - **P-value > 0.05**: Model is statistically accurate
    - **P-value < 0.05**: Model underestimates or overestimates risk
    """)
    
    # Rolling VaR
    st.subheader("Rolling VaR Analysis")
    
    window = st.slider("Rolling Window (Days)", 60, 500, 252)
    
    var_estimates, actual_returns = backtester.rolling_var_backtest(
        window=window,
        method='historical'
    )
    
    fig4 = go.Figure()
    
    fig4.add_trace(go.Scatter(
        x=actual_returns.index,
        y=-actual_returns*100,
        mode='lines',
        name='Actual Losses',
        line=dict(color='lightgray', width=1)
    ))
    
    fig4.add_trace(go.Scatter(
        x=var_estimates.index,
        y=var_estimates*100,
        mode='lines',
        name='VaR Estimate',
        line=dict(color='red', width=2)
    ))
    
    # Highlight violations
    violations = -actual_returns > var_estimates
    if violations.sum() > 0:
        fig4.add_trace(go.Scatter(
            x=actual_returns[violations].index,
            y=-actual_returns[violations]*100,
            mode='markers',
            name='VaR Violations',
            marker=dict(color='red', size=8, symbol='x')
        ))
    
    fig4.update_layout(
        xaxis_title="Date",
        yaxis_title="Loss (%)",
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    st.plotly_chart(fig4, width='stretch')
    
    # Stress Testing
    st.header("Stress Testing & Scenario Analysis")
    
    st.subheader("Market Crash Scenarios")
    
    scenarios = {
        '2008 Financial Crisis': -0.05,
        'Black Monday 1987': -0.22,
        'COVID-19 Crash': -0.12,
        'Moderate Downturn': -0.03
    }
    
    scenario_results = []
    for scenario_name, shock in scenarios.items():
        loss = st.session_state['portfolio_value'] * abs(shock)
        scenario_results.append({
            'Scenario': scenario_name,
            'Shock': f"{shock*100:.0f}%",
            'Portfolio Loss': loss
        })
    
    scenario_df = pd.DataFrame(scenario_results)
    
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        x=scenario_df['Scenario'],
        y=scenario_df['Portfolio Loss'],
        marker_color='crimson',
        text=[f"${val:,.0f}" for val in scenario_df['Portfolio Loss']],
        textposition='outside'
    ))
    fig6.update_layout(
        xaxis_title="Scenario",
        yaxis_title="Expected Loss ($)",
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig6, width='stretch')
    
    st.info("""
    **Historical Context:**
    - **Black Monday (Oct 1987)**: Dow fell 22% in one day
    - **2008 Financial Crisis**: Lehman collapse triggered -5% to -10% daily moves
    - **COVID-19 (Mar 2020)**: Market fell 12% in single day
    - **Moderate Downturn**: Typical bad day in volatile market
    """)
    
    # Correlation Matrix
    st.header("Portfolio Diversification")
    
    corr_matrix = portfolio.get_correlation_matrix()
    
    fig5 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10}
    ))
    
    fig5.update_layout(
        title="Asset Correlation Matrix",
        height=600,
        template='plotly_white'
    )
    st.plotly_chart(fig5, width='stretch')
    
    # About section
    with st.expander("ℹ️ About Risk Metrics"):
        st.write("""
        ### Value-at-Risk (VaR)
        VaR measures the maximum expected loss over a given time period at a specified confidence level.
        
        **Three Methods**:
        1. **Historical VaR**: Uses historical return distribution (non-parametric)
        2. **Parametric VaR**: Assumes normal distribution of returns
        3. **Monte Carlo VaR**: Simulates thousands of potential return scenarios
        
        ### Conditional VaR (CVaR / Expected Shortfall)
        CVaR measures the expected loss given that the VaR threshold has been exceeded.
        It captures "tail risk" better than VaR and is more conservative.
        
        ### Kupiec POF Test
        Statistical test to validate VaR model accuracy by comparing actual violations
        to expected violations. A good model should have violations close to (1-confidence level).
        """)

else:
    st.info("👈 Configure your portfolio in the sidebar and click 'Calculate Risk Metrics' to begin analysis")

# Footer
st.markdown("---")
st.markdown("""
**Developed by Dylan Kenny** | MSc Financial & Computational Mathematics, UCC  
[GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)
""")
