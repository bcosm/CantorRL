
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

