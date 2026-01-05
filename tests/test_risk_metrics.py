import pandas as pd
from src.data_loader import get_market_data
from src.portfolio import Portfolio
from src.risk_metrics import RiskCalculator
from src.backtesting import VaRBacktester

# Load data
prices, returns = get_market_data()
portfolio = Portfolio(returns)
port_returns = portfolio.portfolio_returns()

# Calculate VaR metrics
calc = RiskCalculator(port_returns, confidence_level=0.95)
metrics, dollar_metrics = calc.get_all_metrics(portfolio_value=1000000)

print("VaR Metrics (%):")
for key, value in metrics.items():
    print(f"{key}: {value*100:.4f}%")

print("\nVaR Metrics ($):")
for key, value in dollar_metrics.items():
    print(f"{key}: ${value:,.2f}")

# Backtest
var_series = pd.Series(metrics['Historical VaR'], index=port_returns.index)
backtester = VaRBacktester(port_returns, var_series, 0.95)
kupiec_results = backtester.kupiec_test()

print("\nBacktest Results:")
for key, value in kupiec_results.items():
    print(f"{key}: {value}")
