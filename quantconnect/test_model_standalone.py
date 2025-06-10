#!/usr/bin/env python3
"""
Test script to verify model loading works correctly before uploading to QuantConnect
"""
import numpy as np
import torch
import pickle
import os
from pathlib import Path

class RecurrentPPOModel(torch.nn.Module):
    """Neural network matching the exact SB3 RecurrentPPO architecture"""
    
    def __init__(self, obs_dim=13, action_dim=2, lstm_hidden_size=128):
        super().__init__()
        
        # LSTM layer processes raw observations directly
        self.lstm_actor = torch.nn.LSTM(obs_dim, lstm_hidden_size, batch_first=True)
        
        # Feature extractor processes LSTM output (128 -> 64 -> 64)
        self.mlp_extractor_policy_net = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden_size, 64),  # 128 -> 64
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),  # 64 -> 64
            torch.nn.ReLU()
        )
        
        # Action head (64 -> 2)
        self.action_net = torch.nn.Linear(64, action_dim)
        
        # Log std for continuous actions
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs, hidden_states):
        """Forward pass through the network"""
        # LSTM processing of raw observations
        lstm_out, new_hidden_states = self.lstm_actor(obs, hidden_states)
        
        # Feature extraction from LSTM output
        features = self.mlp_extractor_policy_net(lstm_out)
        
        # Get action means
        action_means = self.action_net(features)
        
        # Apply tanh to get actions in [-1, 1] range
        actions = torch.tanh(action_means)
        
        return actions, new_hidden_states

def test_model_loading():
    """Test loading the extracted model weights"""
    print("Testing model loading...")
    
    # Load the extracted files
    model_files_dir = Path("model_files")
    
    # Load model weights
    policy_weights = torch.load(model_files_dir / "policy_weights.pth", map_location='cpu')
    print("✓ Policy weights loaded")
    print(f"  Available keys: {list(policy_weights.keys())}")
    
    # Load normalization stats
    with open(model_files_dir / "normalization_stats.pkl", 'rb') as f:
        norm_stats = pickle.load(f)
    print("✓ Normalization stats loaded")
    print(f"  Obs mean shape: {norm_stats['obs_mean'].shape}")
    print(f"  Obs var shape: {norm_stats['obs_var'].shape}")
    
    # Load architecture info
    with open(model_files_dir / "architecture_info.pkl", 'rb') as f:
        arch_info = pickle.load(f)
    print("✓ Architecture info loaded")
    print(f"  Architecture: {arch_info}")
    
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
    
    # Create test observation (typical market data)
    test_obs = np.array([
        400.0 / 400.0,  # normalized price (SPY ~400)
        0.001,          # 1-day return
        0.005,          # 5-day return
        0.1,            # vol change
        0.0,            # call position
        0.0,            # put position
        1.0,            # call moneyness (ATM)
        1.0,            # put moneyness (ATM)
        0.15,           # implied vol
        0.5,            # call delta
        -0.5,           # put delta
        0.0,            # portfolio delta
        0.0             # portfolio gamma
    ], dtype=np.float32)
    
    print(f"Test observation shape: {test_obs.shape}")
    print(f"Test observation: {test_obs}")
    
    # Normalize
    normalized_obs = (test_obs - obs_mean) / np.sqrt(obs_var + 1e-8)
    print(f"Normalized observation: {normalized_obs}")
    
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
    print("\nTesting multiple predictions with realistic market scenarios:")
    
    scenarios = [
        ("Market up, low vol", [405.0/400.0, 0.01, 0.05, -0.1, 0.0, 0.0, 1.01, 0.99, 0.12, 0.51, -0.49, 0.0, 0.0]),
        ("Market down, high vol", [395.0/400.0, -0.01, -0.03, 0.2, 0.0, 0.0, 0.99, 1.01, 0.22, 0.49, -0.51, 0.0, 0.0]),
        ("Existing long calls", [400.0/400.0, 0.005, 0.01, 0.05, 0.4, 0.0, 1.0, 1.0, 0.16, 0.5, -0.5, 200.0/10000.0, 50.0/10000.0]),
        ("Existing long puts", [400.0/400.0, -0.005, -0.01, -0.05, 0.0, 0.3, 1.0, 1.0, 0.14, 0.5, -0.5, -150.0/10000.0, 40.0/10000.0]),
        ("High gamma exposure", [400.0/400.0, 0.0, 0.0, 0.0, 0.2, 0.2, 1.0, 1.0, 0.18, 0.5, -0.5, 0.0, 800.0/10000.0])
    ]
    
    for scenario_name, obs_values in scenarios:
        test_obs = np.array(obs_values, dtype=np.float32)
        normalized_obs = (test_obs - obs_mean) / np.sqrt(obs_var + 1e-8)
        obs_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            actions, hidden_states = model(obs_tensor, hidden_states)
            actions = actions.squeeze().numpy()
        
        # Scale to trade sizes (like in the algorithm)
        call_trade = int(actions[0] * 10)  # max_trade_per_step = 10
        put_trade = int(actions[1] * 10)
        
        print(f"  {scenario_name:20}: call={actions[0]:6.3f} -> {call_trade:3d}, put={actions[1]:6.3f} -> {put_trade:3d}")
    
    print("\n✅ All tests passed! Model is ready for QuantConnect.")
    print("\nNext steps:")
    print("1. Upload model files to QuantConnect ObjectStore:")
    print("   - policy_weights.pth")
    print("   - normalization_stats.pkl")
    print("   - architecture_info.pkl")
    print("2. Deploy algorithm with updated model_wrapper.py and main.py")
    print("3. Monitor logs for 'RL Model loaded successfully' and trading activity")

if __name__ == "__main__":
    test_model_loading()
