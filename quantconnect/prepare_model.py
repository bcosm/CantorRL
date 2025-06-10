"""
Utility script to prepare your trained RL model for QuantConnect deployment.

This script will:
1. Load your trained model
2. Extract the necessary components 
3. Create simplified model files for QuantConnect
4. Generate instructions for upload

Run this script before deploying to QuantConnect.
"""

import os
import numpy as np
import torch
import pickle
import joblib
from pathlib import Path

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "src/results/models/lossabs_w0.0001_l0.01"
model_path = MODEL_DIR / "final_model.zip"
vecnorm_path = MODEL_DIR / "final_vecnormalize.pkl"
output_dir = PROJECT_ROOT / "quantconnect" / "model_files"

def extract_model_weights():
    """Extract model weights and architecture for QuantConnect deployment"""
    print("Loading trained model...")
    
    try:
        # Import required libraries
        from stable_baselines3.common.vec_env import VecNormalize
        from sb3_contrib import RecurrentPPO
        
        # Load the trained model
        model = RecurrentPPO.load(str(model_path), device="cpu")
        
        # Load normalization statistics
        with open(str(vecnorm_path), 'rb') as f:
            vec_normalize = pickle.load(f)
        
        print("Model loaded successfully!")
        
        # Extract policy network weights
        policy_state_dict = model.policy.state_dict()
        
        # Extract normalization statistics
        obs_rms = vec_normalize.obs_rms
        ret_rms = vec_normalize.ret_rms
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(policy_state_dict, output_dir / "policy_weights.pth")
        
        # Save normalization statistics
        normalization_stats = {
            'obs_mean': obs_rms.mean,
            'obs_var': obs_rms.var,
            'obs_count': obs_rms.count,
            'ret_mean': ret_rms.mean if ret_rms else None,
            'ret_var': ret_rms.var if ret_rms else None,
            'ret_count': ret_rms.count if ret_rms else None,
            'norm_obs': vec_normalize.norm_obs,
            'norm_reward': vec_normalize.norm_reward,
            'epsilon': vec_normalize.epsilon
        }
        
        with open(output_dir / "normalization_stats.pkl", 'wb') as f:
            pickle.dump(normalization_stats, f)
        
        # Save model architecture info
        architecture_info = {
            'observation_dim': 13,
            'action_dim': 2,
            'hidden_dim': 64,  # You may need to adjust this based on your actual model
            'lstm_layers': 1,
            'model_type': 'RecurrentPPO'
        }
        
        with open(output_dir / "architecture_info.pkl", 'wb') as f:
            pickle.dump(architecture_info, f)
        
        print(f"Model components saved to: {output_dir}")
        print("\nFiles created:")
        print("- policy_weights.pth (PyTorch model weights)")
        print("- normalization_stats.pkl (VecNormalize statistics)")
        print("- architecture_info.pkl (Model architecture info)")
        
        return True
        
    except Exception as e:
        print(f"Error extracting model: {e}")
        return False

def create_deployment_instructions():
    """Create deployment instructions for QuantConnect"""
    instructions = """
# QuantConnect Deployment Instructions

## 1. Upload Model Files to QuantConnect ObjectStore

Upload the following files to your QuantConnect ObjectStore:
- policy_weights.pth
- normalization_stats.pkl  
- architecture_info.pkl

In QuantConnect IDE:
1. Go to Object Store tab
2. Click "Upload File"
3. Upload each of the model files

## 2. Update Model Loading Code

In model_wrapper.py, replace the simulated loading with actual ObjectStore loading:

```python
def LoadModel(self):
    try:
        # Load model weights
        weights_bytes = self.algorithm.ObjectStore.ReadBytes("policy_weights.pth")
        weights_buffer = io.BytesIO(weights_bytes)
        policy_weights = torch.load(weights_buffer, map_location='cpu')
        
        # Load normalization stats
        norm_bytes = self.algorithm.ObjectStore.ReadBytes("normalization_stats.pkl")
        norm_buffer = io.BytesIO(norm_bytes)
        norm_stats = pickle.load(norm_buffer)
        
        # Load architecture info
        arch_bytes = self.algorithm.ObjectStore.ReadBytes("architecture_info.pkl")
        arch_buffer = io.BytesIO(arch_bytes)
        arch_info = pickle.load(arch_buffer)
        
        # Initialize model with loaded weights
        self.model = SimpleRLModel(
            obs_dim=arch_info['observation_dim'],
            action_dim=arch_info['action_dim'],
            hidden_dim=arch_info['hidden_dim']
        )
        self.model.load_state_dict(policy_weights)
        self.model.eval()
        
        # Set normalization stats
        self.obs_mean = norm_stats['obs_mean']
        self.obs_var = norm_stats['obs_var']
        
        self.loaded = True
        self.algorithm.Log("RL Model loaded successfully")
        
    except Exception as e:
        self.algorithm.Log(f"Error loading model: {str(e)}")
        self.loaded = False
```

## 3. Required Libraries

Make sure these libraries are available in your QuantConnect environment:
- torch
- numpy
- scipy
- pickle

## 4. Testing

1. Start with a small date range for testing
2. Monitor logs for any errors
3. Check that option trades are being executed
4. Verify position tracking is working correctly

## 5. Configuration

Adjust these parameters in main.py as needed:
- Start/end dates
- Initial cash
- shares_to_hedge
- max_contracts_per_type
- max_trade_per_step
- transaction_cost_per_contract

## 6. Monitoring

Key metrics to monitor:
- Model prediction success rate
- Option trade execution
- Portfolio PnL
- Position tracking accuracy
- Greek calculations

"""
    
    with open(output_dir / "DEPLOYMENT_INSTRUCTIONS.md", 'w') as f:
        f.write(instructions)
    
    print("Deployment instructions saved to DEPLOYMENT_INSTRUCTIONS.md")

def main():
    """Main function to prepare model for QuantConnect"""
    print("Preparing RL model for QuantConnect deployment...")
    print("=" * 50)
    
    # Check if model files exist
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not vecnorm_path.exists():
        print(f"Error: VecNormalize file not found at {vecnorm_path}")
        return
    
    # Extract model components
    if extract_model_weights():
        print("\n" + "=" * 50)
        print("Model extraction completed successfully!")
        
        # Create deployment instructions
        create_deployment_instructions()
        
        print("\n" + "=" * 50)
        print("Next steps:")
        print("1. Review the files in quantconnect/model_files/")
        print("2. Upload the model files to QuantConnect ObjectStore")
        print("3. Update the model loading code in model_wrapper.py")
        print("4. Deploy and test your algorithm in QuantConnect")
        print("\nSee DEPLOYMENT_INSTRUCTIONS.md for detailed instructions.")
    else:
        print("Model extraction failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
