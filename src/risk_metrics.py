import numpy as np
import pandas as pd
from scipy import stats

class RiskCalculator:
    def __init__(self, returns, confidence_level=0.95):
        """
        Initialize risk calculator
        
        Parameters:
        returns: Series or array of returns
        confidence_level: Confidence level for VaR (default 95%)
        """
        self.returns = returns
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def historical_var(self):
        """Calculate Historical VaR"""
        var = np.percentile(self.returns, self.alpha * 100)
        return -var  # Return as positive loss
    
    def historical_cvar(self):
        """Calculate Historical CVaR (Expected Shortfall)"""
        var = -self.historical_var()  # Get as negative return
        # Average of returns below VaR threshold
        cvar = self.returns[self.returns <= var].mean()
        return -cvar  # Return as positive loss

    def parametric_var(self):
        """Calculate Parametric VaR (assumes normal distribution)"""
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(self.alpha)
        
        var = -(mu + sigma * z_score)
        return var
    
    def parametric_cvar(self):
        """Calculate Parametric CVaR"""
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        z_score = stats.norm.ppf(self.alpha)
        
        # CVaR formula for normal distribution
        cvar = -(mu - sigma * stats.norm.pdf(z_score) / self.alpha)
        return cvar
    def monte_carlo_var(self, n_simulations=10000):
        """Calculate Monte Carlo VaR"""
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        # Simulate returns
        simulated_returns = np.random.normal(mu, sigma, n_simulations)
        
        var = np.percentile(simulated_returns, self.alpha * 100)
        return -var
    
    def monte_carlo_cvar(self, n_simulations=10000):
        """Calculate Monte Carlo CVaR"""
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        # Simulate returns
        simulated_returns = np.random.normal(mu, sigma, n_simulations)
        
        var = np.percentile(simulated_returns, self.alpha * 100)
        cvar = simulated_returns[simulated_returns <= var].mean()
        return -cvar
    
    def get_all_metrics(self, portfolio_value=1000000, n_simulations=10000):
        """Calculate all VaR and CVaR metrics"""
        metrics = {
            'Historical VaR': self.historical_var(),
            'Historical CVaR': self.historical_cvar(),
            'Parametric VaR': self.parametric_var(),
            'Parametric CVaR': self.parametric_cvar(),
            'Monte Carlo VaR': self.monte_carlo_var(n_simulations),
            'Monte Carlo CVaR': self.monte_carlo_cvar(n_simulations)
        }
        
        # Convert to dollar amounts
        dollar_metrics = {k: v * portfolio_value for k, v in metrics.items()}
        
        return metrics, dollar_metrics
