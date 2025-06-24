import numpy as np
import torch
import os
import pickle
import io
from typing import Optional, Union
from AlgorithmImports import *

class ModelWrapper:
    """Wrapper class for loading and using the trained RL model in QuantConnect"""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.model = None
        self.device = torch.device('cpu')  # Use CPU for QuantConnect
        self.hidden_states = None
        self.loaded = False
          # Normalization parameters
        self.obs_mean = None
        self.obs_var = None
        
        # Debug counters
        self.prediction_count = 0
        self.logged_normalization = False
        
        # Load model from ObjectStore
        self.LoadModel()
    
    def LoadModel(self):
        """Load the trained model and normalization from QuantConnect ObjectStore"""
        try:
            # Load model weights from ObjectStore
            weights_bytes = self.algorithm.ObjectStore.ReadBytes("policy_weights.pth")
            weights_buffer = io.BytesIO(weights_bytes)
            policy_weights = torch.load(weights_buffer, map_location='cpu')
            
            # Load normalization stats from ObjectStore
            norm_bytes = self.algorithm.ObjectStore.ReadBytes("normalization_stats.pkl")
            norm_buffer = io.BytesIO(norm_bytes)
            norm_stats = pickle.load(norm_buffer)
            
            # Load architecture info from ObjectStore
            arch_bytes = self.algorithm.ObjectStore.ReadBytes("architecture_info.pkl")
            arch_buffer = io.BytesIO(arch_bytes)
            arch_info = pickle.load(arch_buffer)
            
            # Initialize model with loaded architecture
            self.model = RecurrentPPOModel(
                obs_dim=arch_info['observation_dim'],
                action_dim=arch_info['action_dim'],
                lstm_hidden_size=arch_info['hidden_dim']
            )
            
            # Load the trained weights with custom mapping
            self.LoadSB3Weights(policy_weights)
            
            # Load normalization statistics (FIXED: use correct keys)
            self.obs_mean = norm_stats['obs_mean']
            self.obs_var = norm_stats['obs_var']
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Initialize hidden states
            self.reset_hidden_states()
            
            self.loaded = True
            self.algorithm.Log(f"Model loaded successfully. Obs dim: {arch_info['observation_dim']}, Action dim: {arch_info['action_dim']}")
            self.algorithm.Log(f"Normalization - Mean shape: {self.obs_mean.shape}, Var shape: {self.obs_var.shape}")
            self.algorithm.Log(f"Mean range: [{np.min(self.obs_mean):.4f}, {np.max(self.obs_mean):.4f}]")
            self.algorithm.Log(f"Var range: [{np.min(self.obs_var):.4f}, {np.max(self.obs_var):.4f}]")
            
        except Exception as e:
            self.algorithm.Log(f"Failed to load model: {str(e)}")
            self.loaded = False
    
    def LoadSB3Weights(self, state_dict):
        """Load SB3 RecurrentPPO weights into our custom model"""
        try:
            # Map SB3 state dict to our model
            our_state_dict = {}
            
            # LSTM weights (actor only for inference)
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
            self.model.load_state_dict(our_state_dict)
            self.model.eval()
            
            self.algorithm.Log("SB3 weights mapped and loaded successfully")
            
        except Exception as e:
            self.algorithm.Log(f"Error mapping SB3 weights: {str(e)}")
            raise
    def predict(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """Make prediction using the loaded model - FIXED CRITICAL ISSUE"""
        if not self.loaded or self.model is None:
            self.algorithm.Log("Model not loaded, cannot make prediction")
            return None
            
        try:
            # Log first few predictions for debugging
            self.prediction_count += 1
            if self.prediction_count <= 5:
                self.algorithm.Log(f"=== PREDICTION #{self.prediction_count} ===")
                self.algorithm.Log(f"Input observation shape: {observation.shape}")
                self.algorithm.Log(f"Obs mean shape: {self.obs_mean.shape if self.obs_mean is not None else 'None'}")
                self.algorithm.Log(f"Obs var shape: {self.obs_var.shape if self.obs_var is not None else 'None'}")
            
            # Normalize observation using saved statistics
            if self.obs_mean is None or self.obs_var is None:
                self.algorithm.Log("ERROR: Normalization statistics not loaded!")
                return None
                
            normalized_obs = (observation - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
            
            if self.prediction_count <= 5:
                self.algorithm.Log(f"Normalized obs first 6 values: {normalized_obs[:6]}")
            
            # Convert to tensor with proper shape for LSTM: (batch_size, seq_len, features)
            obs_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).unsqueeze(0)
            
            # Get prediction from model
            with torch.no_grad():
                if self.hidden_states is None:
                    self.reset_hidden_states()
                    
                action, self.hidden_states = self.model(obs_tensor, self.hidden_states)
                action = action.squeeze().numpy()
            
            # Clip actions to [-1, 1] range
            action = np.clip(action, -1, 1)
            
            self.algorithm.Log(f"Model prediction: {action}")
            return action
            
        except Exception as e:
            self.algorithm.Log(f"Error in model prediction: {str(e)}")
            import traceback
            self.algorithm.Log(f"Traceback: {traceback.format_exc()}")
            return None
    
    def reset_hidden_states(self):
        """Reset LSTM hidden states (call at start of new episode/day)"""
        if self.loaded and self.model is not None:
            hidden_dim = 128  # Default LSTM hidden size
            self.hidden_states = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
            self.algorithm.Log("Reset LSTM hidden states")


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
