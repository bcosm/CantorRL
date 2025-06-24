"""
QuantConnect RL Hedging Algorithm - Local Debug Runner
This script allows you to test your algorithm components locally before running on QuantConnect
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def mock_quantconnect_environment():
    """Create mock QuantConnect environment for local testing"""
    
    class MockAlgorithm:
        def __init__(self):
            self.Time = datetime.now()
            self.Portfolio = {}
            self.IsWarmingUp = False
            
        def Log(self, message):
            print(f"[{self.Time}] {message}")
            
        def ObjectStore(self):
            # Mock object store for testing
            pass
    
    return MockAlgorithm()

def test_model_wrapper():
    """Test the ModelWrapper class locally"""
    print("=== Testing ModelWrapper ===")
    
    try:
        # Mock algorithm for testing
        mock_algo = mock_quantconnect_environment()
        
        # Test basic import
        from model_wrapper import ModelWrapper
        print("✓ ModelWrapper import successful")
        
        # Note: Full initialization requires model files
        print("⚠ Full model test requires model files in ObjectStore")
        
    except Exception as e:
        print(f"✗ ModelWrapper test failed: {e}")

def test_option_calculator():
    """Test the OptionCalculator class locally"""
    print("\n=== Testing OptionCalculator ===")
    
    try:
        from option_calculator import OptionCalculator
        calc = OptionCalculator()
        print("✓ OptionCalculator import and initialization successful")
        
        # Test basic Black-Scholes calculation if method exists
        # Add your specific tests here
        
    except Exception as e:
        print(f"✗ OptionCalculator test failed: {e}")

def test_observation_generation():
    """Test observation vector generation"""
    print("\n=== Testing Observation Generation ===")
    
    # Mock some market data
    price_history = [100 + np.sin(i/10) * 5 for i in range(50)]
    vol_history = [0.2 + np.sin(i/20) * 0.05 for i in range(50)]
    
    # Test if we can create a proper observation vector
    try:
        # This would test the get_observation method from your algorithm
        observation_length = 13  # Expected length based on your model
        mock_observation = np.random.randn(observation_length)
        
        print(f"✓ Mock observation vector created: shape {mock_observation.shape}")
        print(f"  Sample values: {mock_observation[:5]}")
        
    except Exception as e:
        print(f"✗ Observation generation test failed: {e}")

def main():
    """Run all local tests"""
    print("QuantConnect RL Hedging Algorithm - Local Debug Tests")
    print("=" * 60)
    
    test_model_wrapper()
    test_option_calculator()
    test_observation_generation()
    
    print("\n" + "=" * 60)
    print("Local testing complete!")
    print("\nNext steps:")
    print("1. Set breakpoints in your code")
    print("2. Use F5 to start debugging in VS Code")
    print("3. Use Ctrl+Shift+P -> 'Tasks: Run Task' for LEAN commands")

if __name__ == "__main__":
    main()
