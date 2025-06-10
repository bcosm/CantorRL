#!/usr/bin/env python3
"""
Extract trained model weights and create files for QuantConnect
"""
import torch
import pickle
import zipfile
import os
import tempfile
import shutil
from pathlib import Path

def extract_model_weights():
    """Extract and convert the SB3 model for QuantConnect"""
    
    # Paths
    model_files_dir = Path("model_files")
    final_model_path = model_files_dir / "final_model.zip"
    vecnorm_path = model_files_dir / "final_vecnormalize.pkl"
    
    print(f"Loading model from: {final_model_path}")
    print(f"Loading vecnormalize from: {vecnorm_path}")
    
    # Extract SB3 model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the zip file
        with zipfile.ZipFile(final_model_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Load the SB3 model
        model_path = os.path.join(temp_dir, "policy.pth")
        print(f"Loading policy from: {model_path}")
        
        policy_state_dict = torch.load(model_path, map_location='cpu')
        print("Policy state dict keys:", list(policy_state_dict.keys()))
        
        # Extract weights
        policy_weights = {}
        
        # Get the policy network weights
        if 'state_dict' in policy_state_dict:
            state_dict = policy_state_dict['state_dict']
        else:
            state_dict = policy_state_dict
            
        print("Available keys in state dict:")
        for key in state_dict.keys():
            print(f"  {key}: {state_dict[key].shape}")
        
        # Save the policy weights for QuantConnect
        torch.save(state_dict, model_files_dir / "policy_weights.pth")
        print("Saved policy_weights.pth")
        
        # Load and process VecNormalize
        print(f"\nLoading VecNormalize from: {vecnorm_path}")
        with open(vecnorm_path, 'rb') as f:
            vecnorm = pickle.load(f)
        
        print("VecNormalize attributes:", dir(vecnorm))
        
        # Extract normalization stats
        if hasattr(vecnorm, 'obs_rms'):
            obs_mean = vecnorm.obs_rms.mean
            obs_var = vecnorm.obs_rms.var
        elif hasattr(vecnorm, 'running_mean'):
            obs_mean = vecnorm.running_mean
            obs_var = vecnorm.running_var
        else:
            print("Warning: Could not find normalization stats, using defaults")
            obs_mean = torch.zeros(13)
            obs_var = torch.ones(13)
        
        print(f"Observation mean shape: {obs_mean.shape}")
        print(f"Observation var shape: {obs_var.shape}")
        
        normalization_stats = {
            'obs_mean': obs_mean.numpy() if torch.is_tensor(obs_mean) else obs_mean,
            'obs_var': obs_var.numpy() if torch.is_tensor(obs_var) else obs_var
        }
        
        # Save normalization stats
        with open(model_files_dir / "normalization_stats.pkl", 'wb') as f:
            pickle.dump(normalization_stats, f)
        print("Saved normalization_stats.pkl")
        
        # Create architecture info
        architecture_info = {
            'observation_dim': 13,  # Based on the training environment
            'action_dim': 2,       # Based on the action space
            'hidden_dim': 128      # LSTM hidden size from RecurrentPPO
        }
        
        # Save architecture info
        with open(model_files_dir / "architecture_info.pkl", 'wb') as f:
            pickle.dump(architecture_info, f)
        print("Saved architecture_info.pkl")
        
        print("\nModel extraction completed successfully!")
        print("Files created:")
        print(f"  - {model_files_dir / 'policy_weights.pth'}")
        print(f"  - {model_files_dir / 'normalization_stats.pkl'}")
        print(f"  - {model_files_dir / 'architecture_info.pkl'}")

if __name__ == "__main__":
    extract_model_weights()
