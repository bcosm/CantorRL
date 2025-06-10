"""Simple test for the model architecture without QuantConnect dependencies"""

import numpy as np
import torch
import pickle

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

def test_architecture():
    """Test the model architecture against actual weights"""
    
    print("Testing RecurrentPPOModel Architecture")
    print("=" * 50)
    
    # Load the actual model weights
    try:
        state_dict = torch.load('model_files/policy_weights.pth', map_location='cpu')
        print("✓ Loaded policy weights")
    except Exception as e:
        print(f"✗ Failed to load weights: {e}")
        return False
    
    print(f"\nOriginal state dict structure:")
    for key, tensor in state_dict.items():
        print(f"  {key}: {tensor.shape}")
    
    # Create our model
    model = RecurrentPPOModel(obs_dim=13, action_dim=2, lstm_hidden_size=128)
    
    print(f"\nOur model structure:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # Create the mapping
    print(f"\nMapping weights...")
    try:
        our_state_dict = {}
        
        # LSTM weights - direct mapping
        our_state_dict['lstm_actor.weight_ih_l0'] = state_dict['lstm_actor.weight_ih_l0']
        our_state_dict['lstm_actor.weight_hh_l0'] = state_dict['lstm_actor.weight_hh_l0'] 
        our_state_dict['lstm_actor.bias_ih_l0'] = state_dict['lstm_actor.bias_ih_l0']
        our_state_dict['lstm_actor.bias_hh_l0'] = state_dict['lstm_actor.bias_hh_l0']
        
        # Feature extractor weights - map from mlp_extractor to our naming
        our_state_dict['mlp_extractor_policy_net.0.weight'] = state_dict['mlp_extractor.policy_net.0.weight']
        our_state_dict['mlp_extractor_policy_net.0.bias'] = state_dict['mlp_extractor.policy_net.0.bias']
        our_state_dict['mlp_extractor_policy_net.2.weight'] = state_dict['mlp_extractor.policy_net.2.weight']
        our_state_dict['mlp_extractor_policy_net.2.bias'] = state_dict['mlp_extractor.policy_net.2.bias']
        
        # Action network weights - direct mapping
        our_state_dict['action_net.weight'] = state_dict['action_net.weight']
        our_state_dict['action_net.bias'] = state_dict['action_net.bias']
        
        # Log std parameter - direct mapping
        our_state_dict['log_std'] = state_dict['log_std']
        
        print("✓ Weight mapping created")
        
        # Verify all weights are mapped
        model_params = set(name for name, _ in model.named_parameters())
        mapped_params = set(our_state_dict.keys())
        
        missing = model_params - mapped_params
        extra = mapped_params - model_params
        
        if missing:
            print(f"✗ Missing parameters: {missing}")
            return False
        if extra:
            print(f"✗ Extra parameters: {extra}")
            return False
            
        print("✓ All parameters mapped correctly")
        
        # Load the weights
        model.load_state_dict(our_state_dict)
        model.eval()
        print("✓ Weights loaded successfully")
        
    except Exception as e:
        print(f"✗ Error in weight mapping: {e}")
        return False
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    try:
        batch_size = 1
        seq_len = 1
        obs_dim = 13
        hidden_size = 128
        
        # Create test input
        test_obs = torch.randn(batch_size, seq_len, obs_dim)
        hidden_states = (
            torch.zeros(1, batch_size, hidden_size),
            torch.zeros(1, batch_size, hidden_size)
        )
        
        print(f"  Input shape: {test_obs.shape}")
        print(f"  Hidden state shapes: {hidden_states[0].shape}, {hidden_states[1].shape}")
        
        # Forward pass
        with torch.no_grad():
            actions, new_hidden = model(test_obs, hidden_states)
        
        print(f"  Output shape: {actions.shape}")
        print(f"  Output values: {actions.squeeze().numpy()}")
        print(f"  Actions in [-1,1]: {torch.all(actions >= -1) and torch.all(actions <= 1)}")
        
        print("✓ Forward pass successful!")
        return True
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False

def test_prediction_pipeline():
    """Test the complete prediction pipeline"""
    
    print(f"\n" + "="*50)
    print("Testing Prediction Pipeline")
    
    try:
        # Load normalization stats
        with open('model_files/normalization_stats.pkl', 'rb') as f:
            norm_stats = pickle.load(f)
        
        obs_mean = norm_stats['obs_mean']
        obs_var = norm_stats['obs_var']
        
        print("✓ Loaded normalization stats")
        print(f"  obs_mean shape: {obs_mean.shape}")
        print(f"  obs_var shape: {obs_var.shape}")
        
        # Create model and load weights (reuse from previous test)
        model = RecurrentPPOModel(obs_dim=13, action_dim=2, lstm_hidden_size=128)
        state_dict = torch.load('model_files/policy_weights.pth', map_location='cpu')
        
        # Map weights
        our_state_dict = {
            'lstm_actor.weight_ih_l0': state_dict['lstm_actor.weight_ih_l0'],
            'lstm_actor.weight_hh_l0': state_dict['lstm_actor.weight_hh_l0'],
            'lstm_actor.bias_ih_l0': state_dict['lstm_actor.bias_ih_l0'],
            'lstm_actor.bias_hh_l0': state_dict['lstm_actor.bias_hh_l0'],
            'mlp_extractor_policy_net.0.weight': state_dict['mlp_extractor.policy_net.0.weight'],
            'mlp_extractor_policy_net.0.bias': state_dict['mlp_extractor.policy_net.0.bias'],
            'mlp_extractor_policy_net.2.weight': state_dict['mlp_extractor.policy_net.2.weight'],
            'mlp_extractor_policy_net.2.bias': state_dict['mlp_extractor.policy_net.2.bias'],
            'action_net.weight': state_dict['action_net.weight'],
            'action_net.bias': state_dict['action_net.bias'],
            'log_std': state_dict['log_std']
        }
        
        model.load_state_dict(our_state_dict)
        model.eval()
        
        # Initialize hidden states
        hidden_size = 128
        hidden_states = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
        
        # Test multiple predictions
        print(f"\nTesting multiple predictions...")
        
        for i in range(3):
            # Create realistic observation
            observation = np.array([
                4.0,    # spy_normalized (around 400/100)
                0.001,  # spy_return
                0.5,    # call_delta  
                -0.5,   # put_delta
                0.01,   # gamma
                0.02,   # call_price_norm
                0.02,   # put_price_norm
                0.0,    # portfolio_delta
                0.2,    # volatility
                0.0,    # portfolio_gamma
                0.1,    # time_to_expiration
                0.1,    # call_vega
                0.1     # put_vega
            ], dtype=np.float32)
            
            # Normalize observation
            normalized_obs = (observation - obs_mean) / np.sqrt(obs_var + 1e-8)
            
            # Convert to tensor
            obs_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                action, hidden_states = model(obs_tensor, hidden_states)
                action = action.squeeze().numpy()
            
            # Clip actions
            action = np.clip(action, -1, 1)
            
            print(f"  Prediction {i+1}: call_action={action[0]:.3f}, put_action={action[1]:.3f}")
        
        print("✓ Prediction pipeline working!")
        return True
        
    except Exception as e:
        print(f"✗ Error in prediction pipeline: {e}")
        return False

if __name__ == "__main__":
    success1 = test_architecture()
    success2 = test_prediction_pipeline()
    
    print(f"\n" + "="*50)
    print("FINAL RESULTS")
    print(f"Architecture Test: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Pipeline Test: {'✓ PASS' if success2 else '✗ FAIL'}")
    print(f"Overall Status: {'✓ READY FOR DEPLOYMENT' if success1 and success2 else '✗ NEEDS FIXES'}")
