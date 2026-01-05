# 📊 Portfolio Risk Management System

> Comprehensive VaR and CVaR analytics platform with backtesting and interactive visualization

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)](https://streamlit.io/)

## 🎯 Overview

Interactive risk analytics platform calculating portfolio **Value-at-Risk (VaR)** and **Conditional VaR (CVaR)** using three industry-standard methodologies with statistical validation via Kupiec backtesting.

Built for quantitative finance and risk analyst applications, demonstrating:
- Derivative pricing theory
- Monte Carlo simulation
- Statistical hypothesis testing
- Financial data visualization

## ✨ Features

### Risk Metrics
- ✅ **Historical VaR** - Non-parametric, actual distribution
- ✅ **Parametric VaR** - Closed-form normal approximation
- ✅ **Monte Carlo VaR** - 10,000+ scenario simulation
- ✅ **CVaR/Expected Shortfall** - Tail risk measurement

### Model Validation
- ✅ Kupiec Proportion of Failures (POF) test
- ✅ Rolling window backtesting
- ✅ Violation tracking and visualization
- ✅ Statistical significance testing (p-values)

### Portfolio Analytics
- Multi-asset construction with custom/equal weighting
- Performance metrics (Sharpe ratio, volatility, skewness, kurtosis)
- Correlation matrix analysis
- Time series value tracking
- Stress testing scenarios

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/portfolio-risk-var.git

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
