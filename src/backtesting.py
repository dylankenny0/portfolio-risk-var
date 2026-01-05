import numpy as np
import pandas as pd
from scipy import stats

class VaRBacktester:
    def __init__(self, returns, var_estimates, confidence_level=0.95):
        """
        Initialize backtester
        
        Parameters:
        returns: Actual returns
        var_estimates: VaR estimates (as positive losses)
        confidence_level: Confidence level used for VaR
        """
        self.returns = returns
        self.var_estimates = var_estimates
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def get_violations(self):
        """Identify VaR violations (exceedances)"""
        # VaR violation occurs when loss exceeds VaR
        losses = -self.returns
        violations = losses > self.var_estimates
        return violations
    
    def kupiec_test(self):
        """
        Kupiec POF (Proportion of Failures) test
        Tests if violation rate matches expected rate
        """
        violations = self.get_violations()
        n_violations = violations.sum()
        n_obs = len(violations)
        
        violation_rate = n_violations / n_obs
        expected_rate = self.alpha
        
        # Likelihood ratio test statistic
        if n_violations == 0 or n_violations == n_obs:
            lr_stat = 0
            p_value = 1.0
        else:
            lr_stat = -2 * (
                n_violations * np.log(expected_rate) +
                (n_obs - n_violations) * np.log(1 - expected_rate) -
                n_violations * np.log(violation_rate) -
                (n_obs - n_violations) * np.log(1 - violation_rate)
            )
            # Chi-squared distribution with 1 df
            p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            'n_violations': n_violations,
            'n_observations': n_obs,
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'reject_null': p_value < 0.05
        }
    
    def calculate_avg_exceedance(self):
        """Calculate average size of VaR exceedances"""
        violations = self.get_violations()
        losses = -self.returns
        
        if violations.sum() == 0:
            return 0
        
        exceedances = losses[violations] - self.var_estimates[violations]
        return exceedances.mean()
    
    def rolling_var_backtest(self, window=252, method='historical'):
        """
        Rolling window VaR backtest
        
        Parameters:
        window: Rolling window size
        method: 'historical', 'parametric', or 'monte_carlo'
        """
        from src.risk_metrics import RiskCalculator
        
        var_estimates = []
        actual_returns = []
        
        for i in range(window, len(self.returns)):
            # Use window of historical returns
            window_returns = self.returns[i-window:i]
            
            # Calculate VaR
            calc = RiskCalculator(window_returns, self.confidence_level)
            
            if method == 'historical':
                var = calc.historical_var()
            elif method == 'parametric':
                var = calc.parametric_var()
            elif method == 'monte_carlo':
                var = calc.monte_carlo_var()
            
            var_estimates.append(var)
            actual_returns.append(self.returns.iloc[i])
        
        var_estimates = pd.Series(var_estimates, index=self.returns.index[window:])
        actual_returns = pd.Series(actual_returns, index=self.returns.index[window:])
        
        return var_estimates, actual_returns
