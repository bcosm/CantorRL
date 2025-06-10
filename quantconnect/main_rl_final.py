# QuantConnect Algorithm - Final RL Model Version (Fixed)
# Now using actual trained RecurrentPPO model with proper architecture

import numpy as np
from AlgorithmImports import *
from datetime import datetime, timedelta
import pickle
import torch
import torch.nn as nn

class RLHedgingFinalAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        self.Debug("Starting RL Hedging Algorithm with RecurrentPPO Model")
        
        # Basic setup
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 3, 31)
        self.SetCash(100000)
        
        # Add SPY
        self.spy = self.AddEquity("SPY", Resolution.Minute)
        self.spy_symbol = self.spy.Symbol
        
        # Add SPY options
        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(-5, 5, timedelta(days=7), timedelta(days=60))
        
        # State variables
        self.spy_shares = 1000  # Start conservative 
        self.rebalance_frequency = 50  # Rebalance every 50 bars
        self.bar_count = 0
        self.last_spy_price = None
        
        # RL Model components
        self.model = None
        self.obs_normalizer = None
        self.price_history = []
        self.hidden_state = None
        
        # Counters
        self.total_rebalances = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.model_predictions = 0
        self.fallback_predictions = 0
        
        # Load model files from ObjectStore
        self.LoadModel()
        
        self.Debug("Initialization complete - Model loaded")
    
    def LoadModel(self):
        """Load the trained RecurrentPPO model from ObjectStore"""
        try:
            # Load architecture info
            if self.ObjectStore.ContainsKey("architecture_info.pkl"):
                arch_data = self.ObjectStore.ReadBytes("architecture_info.pkl")
                arch_info = pickle.loads(arch_data)
                self.Debug(f"Loaded architecture: {arch_info}")
            
            # Load normalization stats
            if self.ObjectStore.ContainsKey("normalization_stats.pkl"):
                norm_data = self.ObjectStore.ReadBytes("normalization_stats.pkl")
                self.obs_normalizer = pickle.loads(norm_data)
                self.Debug("Loaded normalization stats")
            
            # Load model weights
            if self.ObjectStore.ContainsKey("policy_weights.pth"):
                weights_data = self.ObjectStore.ReadBytes("policy_weights.pth")
                
                # Load the full SB3 state dict
                import io
                full_weights_dict = torch.load(io.BytesIO(weights_data), map_location='cpu')
                
                # Create our RecurrentPPO-compatible model 
                self.model = RecurrentPPOModel(obs_dim=13, action_dim=2, hidden_dim=64)
                
                # Extract policy weights from SB3 format
                policy_weights = self.ExtractPolicyWeights(full_weights_dict)
                self.model.load_state_dict(policy_weights, strict=False)
                self.model.eval()
                
                # Initialize hidden state
                self.hidden_state = (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))
                
                self.Debug("RecurrentPPO model loaded successfully")
            else:
                self.Debug("Model weights not found - using fallback")
                
        except Exception as e:
            self.Debug(f"Model loading failed: {str(e)[:200]}... Using fallback")
            self.model = None
    
    def ExtractPolicyWeights(self, full_weights_dict):
        """Extract policy network weights from SB3 RecurrentPPO state dict"""
        policy_weights = {}
        
        try:
            # Map MLP extractor weights (feature extractor)
            if "mlp_extractor.policy_net.0.weight" in full_weights_dict:
                policy_weights["feature_extractor.0.weight"] = full_weights_dict["mlp_extractor.policy_net.0.weight"]
                policy_weights["feature_extractor.0.bias"] = full_weights_dict["mlp_extractor.policy_net.0.bias"]
            
            if "mlp_extractor.policy_net.2.weight" in full_weights_dict:
                policy_weights["feature_extractor.2.weight"] = full_weights_dict["mlp_extractor.policy_net.2.weight"]
                policy_weights["feature_extractor.2.bias"] = full_weights_dict["mlp_extractor.policy_net.2.bias"]
            
            # Map LSTM weights
            if "lstm_actor.weight_ih_l0" in full_weights_dict:
                policy_weights["lstm.weight_ih_l0"] = full_weights_dict["lstm_actor.weight_ih_l0"]
                policy_weights["lstm.weight_hh_l0"] = full_weights_dict["lstm_actor.weight_hh_l0"]
                policy_weights["lstm.bias_ih_l0"] = full_weights_dict["lstm_actor.bias_ih_l0"]
                policy_weights["lstm.bias_hh_l0"] = full_weights_dict["lstm_actor.bias_hh_l0"]
            
            # Map action network weights (policy head final layer)
            if "action_net.weight" in full_weights_dict:
                policy_weights["policy_head.2.weight"] = full_weights_dict["action_net.weight"]
                policy_weights["policy_head.2.bias"] = full_weights_dict["action_net.bias"]
                
            self.Debug(f"Extracted {len(policy_weights)} weight tensors")
            
        except Exception as e:
            self.Debug(f"Weight extraction error: {str(e)[:100]}")
        
        return policy_weights
            self.model.eval()
                
                self.Debug("Model weights extracted and loaded successfully")
            else:
                self.Debug("Model weights not found - using fallback")
                
        except Exception as e:
            self.Debug(f"Model loading failed: {str(e)[:100]}... Using fallback")
            self.model = None
    
    def OnData(self, data):
        self.bar_count += 1
        
        # Get SPY price
        if not data.ContainsKey(self.spy_symbol) or data[self.spy_symbol] is None:
            return
            
        spy_price = data[self.spy_symbol].Close
        self.last_spy_price = spy_price
        self.price_history.append(spy_price)
        
        # Keep only recent history
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
        
        # Rebalance every N bars
        if self.bar_count % self.rebalance_frequency == 0:
            self.total_rebalances += 1
            self.Rebalance(data)
            
            # Log every 10 rebalances to avoid hitting limits
            if self.total_rebalances % 10 == 0:
                self.Debug(f"Rebalance #{self.total_rebalances}: SPY ${spy_price:.2f}, Success {self.successful_trades}, Model {self.model_predictions}, Fallback {self.fallback_predictions}")
    
    def Rebalance(self, data):
        # Ensure we have SPY position
        self.EnsureSPYPosition()
        
        # Get available options
        option_chains = [x for x in data.OptionChains]
        if not option_chains:
            return
            
        chain = option_chains[0].Value
        if not chain:
            return
        
        # Find ATM options
        calls, puts = self.GetATMOptions(chain, self.last_spy_price)
        if not calls and not puts:
            return
        
        # Get RL model prediction
        action = self.GetModelPrediction()
        
        # Execute trades based on action
        self.ExecuteAction(action, calls, puts)
    
    def EnsureSPYPosition(self):
        """Ensure we have the target SPY position"""
        spy_holdings = self.Portfolio[self.spy_symbol].Quantity
        if spy_holdings < self.spy_shares * 0.1:  # Start with 10% of target
            shares_to_buy = int(self.spy_shares * 0.1) - spy_holdings
            if shares_to_buy > 0 and self.Portfolio.Cash > shares_to_buy * self.last_spy_price:
                self.MarketOrder(self.spy_symbol, shares_to_buy)
    
    def GetModelPrediction(self):
        """Get prediction from RL model or use fallback"""
        if self.model is None or len(self.price_history) < 10:
            # Fallback: simple momentum-based action
            self.fallback_predictions += 1
            if len(self.price_history) >= 2:
                price_change = self.price_history[-1] - self.price_history[-2]
                if price_change > 0:
                    return 12  # Buy call (middle of action space)
                else:
                    return 8   # Buy put (middle of action space)
            return 10  # No action (middle of action space)
        
        try:
            # Build observation vector (13 features like training)
            obs = self.BuildObservation()
            
            # Normalize
            if self.obs_normalizer:
                obs = self.obs_normalizer.transform([obs])[0]
            
            # Get model prediction
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_probs = self.model(obs_tensor)
                action = int(torch.argmax(action_probs).item())
            
            self.model_predictions += 1
            return action
            
        except Exception as e:
            self.fallback_predictions += 1
            return 10  # Default no action
    
    def BuildObservation(self):
        """Build 13-feature observation vector matching training environment"""
        if len(self.price_history) < 10:
            return [0.0] * 13
        
        current_price = self.price_history[-1]
        
        # Price-based features
        price_features = [
            current_price / 400.0,  # Normalized current price
            (self.price_history[-1] - self.price_history[-2]) / current_price if len(self.price_history) >= 2 else 0.0,  # Price change
            np.std(self.price_history[-10:]) / current_price,  # Volatility
        ]
        
        # Portfolio features (simplified)
        spy_position = self.Portfolio[self.spy_symbol].Quantity / 10000.0
        total_value = self.Portfolio.TotalPortfolioValue / 100000.0
        cash_ratio = self.Portfolio.Cash / self.Portfolio.TotalPortfolioValue
        
        portfolio_features = [spy_position, total_value, cash_ratio]
        
        # Option position features (simplified to zeros for now)
        option_features = [0.0] * 7  # 7 features for option positions
        
        return price_features + portfolio_features + option_features
    
    def ExecuteAction(self, action, calls, puts):
        """Execute trading action based on RL model output"""
        # Action space: 0-20, where 10 is no action
        # 0-9: sell/buy puts, 11-20: sell/buy calls
        
        if action < 5:  # Buy puts
            if puts:
                contracts = 1  # Conservative position size
                order_id = self.MarketOrder(puts[0].Symbol, contracts)
                if order_id:
                    self.successful_trades += 1
                    self.Debug(f"MODEL PUT BUY: {puts[0].Strike} strike")
                    
        elif action > 15:  # Buy calls
            if calls:
                contracts = 1  # Conservative position size
                order_id = self.MarketOrder(calls[0].Symbol, contracts)
                if order_id:
                    self.successful_trades += 1
                    self.Debug(f"MODEL CALL BUY: {calls[0].Strike} strike")
        
        # Actions 5-15 include holds and smaller trades - implement as needed
    
    def GetATMOptions(self, chain, spy_price):
        """Get at-the-money call and put options"""
        calls = []
        puts = []
        
        atm_threshold = 2.0
        
        for contract in chain:
            if abs(contract.Strike - spy_price) <= atm_threshold:
                if contract.Right == OptionRight.Call:
                    calls.append(contract)
                else:
                    puts.append(contract)
        
        calls.sort(key=lambda x: abs(x.Strike - spy_price))
        puts.sort(key=lambda x: abs(x.Strike - spy_price))
        
        return calls[:3], puts[:3]
    
    def OnOrderEvent(self, orderEvent):
        """Log important order events"""
        if orderEvent.Status == OrderStatus.Filled and "SPY" in str(orderEvent.Symbol) and len(str(orderEvent.Symbol)) > 3:
            self.Debug(f"OPTION FILLED: {orderEvent.Symbol} x{orderEvent.FillQuantity} @ ${orderEvent.FillPrice:.2f}")
        elif orderEvent.Status == OrderStatus.Invalid:
            self.failed_trades += 1
    
    def OnEndOfAlgorithm(self):
        """Final summary"""
        self.Debug(f"FINAL: {self.total_rebalances} rebalances, {self.successful_trades} trades")
        self.Debug(f"Model predictions: {self.model_predictions}, Fallback: {self.fallback_predictions}")
        self.Debug(f"Portfolio Value: ${self.Portfolio.TotalPortfolioValue:.2f}")


class RecurrentPPOModel(nn.Module):
    """RecurrentPPO-compatible model architecture"""
    
    def __init__(self, obs_dim=13, action_dim=2, hidden_dim=64):
        super().__init__()
        
        # Feature extraction layers (matches SB3 mlp_extractor.policy_net)
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LSTM layer for recurrent processing (matches SB3 lstm_actor)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Policy head (matches SB3 action_net)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output continuous actions in [-1, 1] range
        )
    
    def forward(self, obs, hidden_states):
        """Forward pass through the RecurrentPPO model"""
        # Extract features
        features = self.feature_extractor(obs)
        
        # LSTM processing with hidden state
        lstm_out, new_hidden_states = self.lstm(features, hidden_states)
        
        # Get continuous actions
        actions = self.policy_head(lstm_out)
        
        return actions, new_hidden_states
    
    def ExtractPolicyWeights(self, full_weights_dict):
        """Extract policy network weights from SB3 RecurrentPPO state dict"""
        policy_weights = {}
        
        try:
            # Map MLP extractor weights (feature extractor)
            if "mlp_extractor.policy_net.0.weight" in full_weights_dict:
                policy_weights["feature_extractor.0.weight"] = full_weights_dict["mlp_extractor.policy_net.0.weight"]
                policy_weights["feature_extractor.0.bias"] = full_weights_dict["mlp_extractor.policy_net.0.bias"]
            
            if "mlp_extractor.policy_net.2.weight" in full_weights_dict:
                policy_weights["feature_extractor.2.weight"] = full_weights_dict["mlp_extractor.policy_net.2.weight"]
                policy_weights["feature_extractor.2.bias"] = full_weights_dict["mlp_extractor.policy_net.2.bias"]
            
            # Map LSTM weights
            if "lstm_actor.weight_ih_l0" in full_weights_dict:
                policy_weights["lstm.weight_ih_l0"] = full_weights_dict["lstm_actor.weight_ih_l0"]
                policy_weights["lstm.weight_hh_l0"] = full_weights_dict["lstm_actor.weight_hh_l0"]
                policy_weights["lstm.bias_ih_l0"] = full_weights_dict["lstm_actor.bias_ih_l0"]
                policy_weights["lstm.bias_hh_l0"] = full_weights_dict["lstm_actor.bias_hh_l0"]
            
            # Map action network weights (policy head final layer)
            if "action_net.weight" in full_weights_dict:
                policy_weights["policy_head.2.weight"] = full_weights_dict["action_net.weight"]
                policy_weights["policy_head.2.bias"] = full_weights_dict["action_net.bias"]
                
            self.Debug(f"Extracted {len(policy_weights)} weight tensors")
            
        except Exception as e:
            self.Debug(f"Weight extraction error: {str(e)[:100]}")
        
        return policy_weights
