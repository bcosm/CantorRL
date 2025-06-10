"""Test the fixed model architecture locally"""

import numpy as np
import torch
import pickle
import sys
import os

# Mock AlgorithmImports for testing
class MockAlgorithm:
    def Log(self, message):
        print(f"[LOG] {message}")

# Add path and import our modules
sys.path.append('.')
from model_wrapper_fixed import ModelWrapper, RecurrentPPOModel

def test_model_architecture():
    """Test if the model architecture matches the trained weights"""
    
    print("Testing RecurrentPPOModel architecture...")
    
    # Load the actual model weights
    state_dict = torch.load('model_files/policy_weights.pth', map_location='cpu')
    
    print("\nState dict keys and shapes:")
    for key, tensor in state_dict.items():
        print(f"  {key}: {tensor.shape}")
    
    # Create our model with correct architecture
    model = RecurrentPPOModel(obs_dim=13, action_dim=2, lstm_hidden_size=128)
    
    print(f"\nOur model parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # Test the weight mapping manually
    print(f"\nTesting weight mapping...")
    
    try:
        # Create mapping
        our_state_dict = {}
        
        # LSTM weights
        our_state_dict['lstm_actor.weight_ih_l0'] = state_dict['lstm_actor.weight_ih_l0']
        our_state_dict['lstm_actor.weight_hh_l0'] = state_dict['lstm_actor.weight_hh_l0'] 
        our_state_dict['lstm_actor.bias_ih_l0'] = state_dict['lstm_actor.bias_ih_l0']
        our_state_dict['lstm_actor.bias_hh_l0'] = state_dict['lstm_actor.bias_hh_l0']
        
        # Feature extractor weights  
        our_state_dict['mlp_extractor_policy_net.0.weight'] = state_dict['mlp_extractor.policy_net.0.weight']
        our_state_dict['mlp_extractor_policy_net.0.bias'] = state_dict['mlp_extractor.policy_net.0.bias']
        our_state_dict['mlp_extractor_policy_net.2.weight'] = state_dict['mlp_extractor.policy_net.2.weight']
        our_state_dict['mlp_extractor_policy_net.2.bias'] = state_dict['mlp_extractor.policy_net.2.bias']
        
        # Action network weights
        our_state_dict['action_net.weight'] = state_dict['action_net.weight']
        our_state_dict['action_net.bias'] = state_dict['action_net.bias']
        
        # Log std parameter
        our_state_dict['log_std'] = state_dict['log_std']
        
        # Load mapped weights
        model.load_state_dict(our_state_dict)
        model.eval()
        
        print("✓ Weight mapping successful!")
        
        # Test forward pass
        print(f"\nTesting forward pass...")
        
        # Create test input
        batch_size = 1
        seq_len = 1
        obs_dim = 13
        hidden_size = 128
        
        test_obs = torch.randn(batch_size, seq_len, obs_dim)
        hidden_states = (
            torch.zeros(1, batch_size, hidden_size),
            torch.zeros(1, batch_size, hidden_size)
        )
        
        with torch.no_grad():
            actions, new_hidden = model(test_obs, hidden_states)
        
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {test_obs.shape}")
        print(f"  Output shape: {actions.shape}")
        print(f"  Output values: {actions.squeeze().numpy()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def test_full_wrapper():
    """Test the complete ModelWrapper"""
    
    print(f"\n" + "="*50)
    print("Testing complete ModelWrapper...")
    
    # Mock algorithm
    mock_algo = MockAlgorithm()
    
    # Mock ObjectStore reads
    class MockObjectStore:
        def ReadBytes(self, filename):
            with open(f'model_files/{filename}', 'rb') as f:
                return f.read()
    
    mock_algo.ObjectStore = MockObjectStore()
    
    try:
        # Create wrapper
        wrapper = ModelWrapper(mock_algo)
        
        if wrapper.loaded:
            print("✓ Model wrapper loaded successfully!")
            
            # Test prediction
            test_obs = np.random.randn(13).astype(np.float32)
            action = wrapper.predict(test_obs)
            
            if action is not None:
                print(f"✓ Prediction successful!")
                print(f"  Input: {test_obs}")
                print(f"  Output: {action}")
                print(f"  Actions in range [-1,1]: {np.all(action >= -1) and np.all(action <= 1)}")
                return True
            else:
                print("✗ Prediction failed")
                return False
        else:
            print("✗ Model wrapper failed to load")
            return False
            
    except Exception as e:
        print(f"✗ Error in wrapper test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Fixed Model Architecture")
    print("=" * 50)
    
    # Test 1: Architecture compatibility
    success1 = test_model_architecture()
    
    # Test 2: Full wrapper
    success2 = test_full_wrapper()
    
    print(f"\n" + "="*50)
    print("SUMMARY")
    print(f"Architecture test: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Wrapper test: {'✓ PASS' if success2 else '✗ FAIL'}")
    print(f"Overall: {'✓ READY FOR DEPLOYMENT' if success1 and success2 else '✗ NEEDS FIXES'}")
