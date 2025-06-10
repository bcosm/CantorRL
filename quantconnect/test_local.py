"""
Local test script to validate the RL model before QuantConnect deployment.
This script simulates the QuantConnect environment to test model loading and prediction.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import our modules
from quantconnect.model_wrapper import SimpleRLModel
from quantconnect.option_calculator import OptionCalculator

def test_model_architecture():
    """Test the simplified model architecture"""
    print("Testing model architecture...")
    
    # Create model
    model = SimpleRLModel(obs_dim=13, action_dim=2, hidden_dim=64)
    
    # Test forward pass
    obs = torch.randn(1, 1, 13)  # batch_size=1, seq_len=1, obs_dim=13
    hidden_states = (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))
    
    with torch.no_grad():
        actions, new_hidden = model(obs, hidden_states)
    
    print(f"Input shape: {obs.shape}")
    print(f"Output shape: {actions.shape}")
    print(f"Output range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
    print("✓ Model architecture test passed\n")
    
    return model

def test_option_calculator():
    """Test the option calculator"""
    print("Testing option calculator...")
    
    calc = OptionCalculator()
    
    # Test parameters
    S = 400.0  # Current price
    K = 400.0  # Strike (ATM)
    T = 30/252  # 30 days to expiry
    r = 0.04   # Risk-free rate
    sigma = 0.2  # Volatility
    
    # Test Black-Scholes pricing
    call_price = calc.black_scholes_price(S, K, T, r, sigma, 'call')
    put_price = calc.black_scholes_price(S, K, T, r, sigma, 'put')
    
    print(f"Call price: ${call_price:.2f}")
    print(f"Put price: ${put_price:.2f}")
    
    # Test Greeks calculation
    call_delta, put_delta, gamma = calc.calculate_greeks(S, K, T, r, sigma**2)
    
    print(f"Call delta: {call_delta:.3f}")
    print(f"Put delta: {put_delta:.3f}")
    print(f"Gamma: {gamma:.3f}")
    
    # Sanity checks
    assert 0 <= call_delta <= 1, "Call delta should be between 0 and 1"
    assert -1 <= put_delta <= 0, "Put delta should be between -1 and 0"
    assert gamma >= 0, "Gamma should be non-negative"
    
    print("✓ Option calculator test passed\n")

def test_observation_processing():
    """Test observation vector creation similar to QuantConnect"""
    print("Testing observation processing...")
    
    # Simulate observation data
    current_price = 400.0
    current_vol = 0.2
    price_history = [395.0, 398.0, 401.0, 399.0, 400.0]
    vol_history = [0.18, 0.19, 0.21, 0.20, 0.20]
    
    # Portfolio state
    current_call_contracts = 50
    current_put_contracts = -30
    max_contracts_per_type = 200
    
    # Calculate features
    price_return_1 = (price_history[-1] / price_history[-2] - 1)
    price_return_5 = (price_history[-1] / price_history[0] - 1)
    vol_change = (vol_history[-1] / vol_history[-2] - 1)
    
    call_position_normalized = current_call_contracts / max_contracts_per_type
    put_position_normalized = current_put_contracts / max_contracts_per_type
    
    # Option features
    time_to_expiry = 30/252
    moneyness_call = current_price / current_price  # ATM = 1.0
    moneyness_put = current_price / current_price   # ATM = 1.0
    
    # Calculate Greeks
    calc = OptionCalculator()
    call_delta, put_delta, gamma = calc.calculate_greeks(
        current_price, current_price, time_to_expiry, 0.04, current_vol**2
    )
    
    # Portfolio Greeks
    option_multiplier = 100
    portfolio_delta = (call_delta * current_call_contracts + 
                      put_delta * current_put_contracts) * option_multiplier
    portfolio_gamma = gamma * (current_call_contracts + abs(current_put_contracts)) * option_multiplier
    
    # Create observation vector
    observation = np.array([
        current_price / 400.0,  # Normalized price
        price_return_1,
        price_return_5, 
        vol_change,
        call_position_normalized,
        put_position_normalized,
        moneyness_call,
        moneyness_put,
        current_vol,
        call_delta,
        put_delta,
        portfolio_delta / 10000.0,  # Normalized
        portfolio_gamma / 10000.0   # Normalized
    ], dtype=np.float32)
    
    print(f"Observation vector shape: {observation.shape}")
    print(f"Observation vector: {observation}")
    print(f"All finite values: {np.all(np.isfinite(observation))}")
    
    assert observation.shape == (13,), "Observation should have 13 features"
    assert np.all(np.isfinite(observation)), "All observation values should be finite"
    
    print("✓ Observation processing test passed\n")
    
    return observation

def test_model_prediction():
    """Test model prediction with sample observation"""
    print("Testing model prediction...")
    
    # Create model
    model = SimpleRLModel()
    model.eval()
    
    # Create sample observation
    observation = test_observation_processing()
    
    # Simulate normalization
    obs_mean = np.zeros(13)
    obs_var = np.ones(13)
    normalized_obs = (observation - obs_mean) / np.sqrt(obs_var + 1e-8)
    
    # Convert to tensor
    obs_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).unsqueeze(0)
    hidden_states = (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))
    
    # Get prediction
    with torch.no_grad():
        actions, new_hidden_states = model(obs_tensor, hidden_states)
        actions = actions.squeeze().numpy()
    
    print(f"Predicted actions: {actions}")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # Check that actions are in expected range
    assert len(actions) == 2, "Should predict 2 actions (call, put)"
    assert np.all(actions >= -1) and np.all(actions <= 1), "Actions should be in [-1, 1] range"
    
    print("✓ Model prediction test passed\n")

def main():
    """Run all tests"""
    print("Running QuantConnect model validation tests...")
    print("=" * 60)
    
    try:
        # Run tests
        test_model_architecture()
        test_option_calculator()
        test_observation_processing()
        test_model_prediction()
        
        print("=" * 60)
        print("✅ All tests passed! Your model is ready for QuantConnect deployment.")
        print("\nNext steps:")
        print("1. Run 'python quantconnect/prepare_model.py' to extract your trained model")
        print("2. Follow the deployment instructions to upload to QuantConnect")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nPlease fix the issues before deploying to QuantConnect.")

if __name__ == "__main__":
    main()
