import numpy as np
import pandas as pd

class Portfolio:
    def __init__(self, returns, weights=None):
        """
        Initialize portfolio
        
        Parameters:
        returns: DataFrame of asset returns
        weights: Array of portfolio weights (sum to 1)
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        
        if weights is None:
            # Equal weights if not specified
            self.weights = np.ones(self.n_assets) / self.n_assets
        else:
            self.weights = np.array(weights)
            assert np.isclose(self.weights.sum(), 1.0), "Weights must sum to 1"
    
    def portfolio_returns(self):
        """Calculate portfolio returns time series"""
        return (self.returns * self.weights).sum(axis=1)
    
    def portfolio_value(self, initial_value=1000000):
        """Calculate portfolio value over time"""
        portfolio_rets = self.portfolio_returns()
        portfolio_value = initial_value * (1 + portfolio_rets).cumprod()
        return portfolio_value
    
    def get_statistics(self):
        """Calculate portfolio statistics"""
        port_returns = self.portfolio_returns()
        
        stats = {
            'Expected Return (Annual)': port_returns.mean() * 252,
            'Volatility (Annual)': port_returns.std() * np.sqrt(252),
            'Sharpe Ratio': (port_returns.mean() * 252) / (port_returns.std() * np.sqrt(252)),
            'Skewness': port_returns.skew(),
            'Kurtosis': port_returns.kurtosis(),
            'Max Daily Return': port_returns.max(),
            'Min Daily Return': port_returns.min()
        }
        
        return stats
    
    def get_correlation_matrix(self):
        """Get correlation matrix of assets"""
        return self.returns.corr()
