import numpy as np
from scipy.stats import norm
from typing import Dict

class OptionCalculator:
    """Black-Scholes option pricing and Greeks calculator"""
    
    def __init__(self):
        pass
    
    def black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """Calculate Black-Scholes option price"""
        if T <= 0 or sigma <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0)
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """Calculate option Greeks (delta, gamma, vega)"""
        # Handle edge cases
        if T <= 1e-6 or sigma <= 1e-8 or S <= 1e-6:
            if option_type == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            return {'delta': delta, 'gamma': 0.0, 'vega': 0.0}
        
        # Standard Black-Scholes Greeks calculation
        K_checked = max(K, 1e-6)
        d1 = (np.log(S / K_checked) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        # Delta calculation
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1.0
        
        # Gamma calculation (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega calculation (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega
        }
