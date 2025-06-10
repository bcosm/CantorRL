#!/usr/bin/env python3
"""
Test script to verify model loading works correctly before uploading to QuantConnect
"""
import numpy as np
import torch
import pickle
import os
import sys
from pathlib import Path

# Mock QuantConnect imports for testing
class MockAlgorithm:
    def Log(self, message):
        print(f"[LOG] {message}")

# Add the quantconnect directory to path
sys.path.append(str(Path(__file__).parent))

# Import our model wrapper (with mocked AlgorithmImports)
sys.modules['AlgorithmImports'] = type(sys)('mocked_module')
from model_wrapper import RecurrentPPOModel

def test_model_loading():
    """Test loading the extracted model weights"""
    print("Testing model loading...")
    
    # Load the extracted files
    model_files_dir = Path("model_files")
    
    # Load model weights
    policy_weights = torch.load(model_files_dir / "policy_weights.pth", map_location='cpu')
    print("✓ Policy weights loaded")
    
    # Load normalization stats
    with open(model_files_dir / "normalization_stats.pkl", 'rb') as f:
        norm_stats = pickle.load(f)
    print("✓ Normalization stats loaded")
    
    # Load architecture info
    with open(model_files_dir / "architecture_info.pkl", 'rb') as f:
        arch_info = pickle.load(f)
    print("✓ Architecture info loaded")
    
    # Create model
    model = RecurrentPPOModel(
        obs_dim=arch_info['observation_dim'],
        action_dim=arch_info['action_dim'],
        lstm_hidden_size=arch_info['hidden_dim']
    )
    print("✓ Model created")
    
    # Map and load weights
    our_state_dict = {}
    
    # LSTM weights
    our_state_dict['lstm_actor.weight_ih_l0'] = policy_weights['lstm_actor.weight_ih_l0']
    our_state_dict['lstm_actor.weight_hh_l0'] = policy_weights['lstm_actor.weight_hh_l0'] 
    our_state_dict['lstm_actor.bias_ih_l0'] = policy_weights['lstm_actor.bias_ih_l0']
    our_state_dict['lstm_actor.bias_hh_l0'] = policy_weights['lstm_actor.bias_hh_l0']
    
    # Feature extractor weights  
    our_state_dict['mlp_extractor_policy_net.0.weight'] = policy_weights['mlp_extractor.policy_net.0.weight']
    our_state_dict['mlp_extractor_policy_net.0.bias'] = policy_weights['mlp_extractor.policy_net.0.bias']
    our_state_dict['mlp_extractor_policy_net.2.weight'] = policy_weights['mlp_extractor.policy_net.2.weight']
    our_state_dict['mlp_extractor_policy_net.2.bias'] = policy_weights['mlp_extractor.policy_net.2.bias']
    
    # Action network weights
    our_state_dict['action_net.weight'] = policy_weights['action_net.weight']
    our_state_dict['action_net.bias'] = policy_weights['action_net.bias']
    
    # Log std parameter
    our_state_dict['log_std'] = policy_weights['log_std']
    
    # Load weights
    model.load_state_dict(our_state_dict)
    model.eval()
    print("✓ Weights loaded successfully")
    
    # Test prediction
    obs_mean = norm_stats['obs_mean']
    obs_var = norm_stats['obs_var']
    
    # Create test observation
    test_obs = np.random.randn(13).astype(np.float32)
    print(f"Test observation shape: {test_obs.shape}")
    
    # Normalize
    normalized_obs = (test_obs - obs_mean) / np.sqrt(obs_var + 1e-8)
    
    # Convert to tensor
    obs_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).unsqueeze(0)
    print(f"Tensor shape: {obs_tensor.shape}")
    
    # Initialize hidden states
    hidden_dim = arch_info['hidden_dim']
    hidden_states = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
    
    # Test prediction
    with torch.no_grad():
        actions, new_hidden_states = model(obs_tensor, hidden_states)
        actions = actions.squeeze().numpy()
    
    print(f"✓ Prediction successful: {actions}")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # Test multiple predictions to ensure consistency
    print("\nTesting multiple predictions:")
    for i in range(5):
        test_obs = np.random.randn(13).astype(np.float32)
        normalized_obs = (test_obs - obs_mean) / np.sqrt(obs_var + 1e-8)
        obs_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            actions, hidden_states = model(obs_tensor, hidden_states)
            actions = actions.squeeze().numpy()
        
        print(f"  Prediction {i+1}: call={actions[0]:.3f}, put={actions[1]:.3f}")
    
    print("\n✅ All tests passed! Model is ready for QuantConnect.")

if __name__ == "__main__":
    test_model_loading()
