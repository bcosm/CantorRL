# region imports
from AlgorithmImports import *
import numpy as np
import torch
import pickle
import io
from typing import Optional
# endregion

class RLHedgingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        # Set basic algorithm parameters
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 2, 1)    # 1 month for testing
        self.SetCash(1000000)
        
        # Add SPY 
        self.spy = self.AddEquity("SPY")
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
        # Add options
        option = self.AddOption("SPY")
        option.SetFilter(self.OptionFilter)
        
        # CONSERVATIVE position sizing
        self.target_spy_allocation = 0.6  # 60% in SPY
        self.max_contracts_per_type = 10  # Start very small
        self.max_trade_per_step = 2       # Very conservative
        
        # State tracking
        self.last_price = None
        self.last_vol = 0.2
        self.price_history = []
        self.vol_history = [0.2, 0.2, 0.2, 0.2, 0.2]  # Initialize with some values
        self.current_call_contracts = 0
        self.current_put_contracts = 0
        self.current_call_symbol = None
        self.current_put_symbol = None
        
        # Model components
        self.model = None
        self.obs_mean = None
        self.obs_var = None
        self.hidden_states = None
        self.model_loaded = False
        
        # Load the actual RL model
        self.LoadRLModel()
        
        # Debug counters
        self.rebalance_count = 0
        self.prediction_count = 0
        self.trade_attempt_count = 0
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 60),
            self.Rebalance
        )
        
        self.Log("RL Hedging Algorithm with Real Model Initialized")
    
    def LoadRLModel(self):
        """Load the actual trained RL model from ObjectStore"""
        try:
            # Load model weights
            weights_bytes = self.ObjectStore.ReadBytes("policy_weights.pth")
            weights_buffer = io.BytesIO(weights_bytes)
            policy_weights = torch.load(weights_buffer, map_location='cpu')
            
            # Load normalization stats
            norm_bytes = self.ObjectStore.ReadBytes("normalization_stats.pkl")
            norm_buffer = io.BytesIO(norm_bytes)
            norm_stats = pickle.load(norm_buffer)
            
            # Load architecture info
            arch_bytes = self.ObjectStore.ReadBytes("architecture_info.pkl")
            arch_buffer = io.BytesIO(arch_bytes)
            arch_info = pickle.load(arch_buffer)
            
            # Create model
            self.model = SimpleRLModel(
                obs_dim=arch_info['observation_dim'],
                action_dim=arch_info['action_dim'],
                hidden_dim=arch_info['hidden_dim']
            )
            self.model.load_state_dict(policy_weights)
            self.model.eval()
            
            # Set normalization
            self.obs_mean = norm_stats['obs_mean']
            self.obs_var = norm_stats['obs_var']
            
            # Initialize hidden states
            hidden_dim = arch_info['hidden_dim']
            self.hidden_states = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
            
            self.model_loaded = True
            self.Log("✅ RL Model loaded successfully from ObjectStore!")
            
        except Exception as e:
            self.Log(f"❌ Error loading model: {str(e)}")
            self.Log("Will use fallback prediction logic")
            self.model_loaded = False
    
    def OptionFilter(self, universe):
        return (universe
                .Strikes(-3, 3)
                .Expiration(15, 60)
                .OnlyApplyFilterAtMarketOpen())
    
    def OnData(self, data):
        """Update state and find options"""
        if self.spy.Symbol in data.Bars:
            current_price = data.Bars[self.spy.Symbol].Close
            if current_price > 0:
                if self.last_price is not None:
                    self.price_history.append(current_price)
                self.last_price = current_price
        
        # Process options
        if data.OptionChains.Count > 0:
            for chain in data.OptionChains.Values:
                self.UpdateOptions(chain)
        
        # Keep history manageable
        if len(self.price_history) > 20:
            self.price_history = self.price_history[-10:]
    
    def UpdateOptions(self, chain):
        """Find best ATM options"""
        if self.last_price is None:
            return
            
        best_call = None
        best_put = None
        min_strike_diff = float('inf')
        
        for contract in chain:
            strike_diff = abs(contract.Strike - self.last_price)
            
            if strike_diff < min_strike_diff:
                min_strike_diff = strike_diff
                if contract.Right == 0:  # Call
                    best_call = contract.Symbol
                elif contract.Right == 1:  # Put
                    best_put = contract.Symbol
        
        # Update current options
        if best_call is not None:
            self.current_call_symbol = best_call
        if best_put is not None:
            self.current_put_symbol = best_put
        
        self.Debug(f"Found options: Call={best_call is not None}, Put={best_put is not None}")
    
    def Rebalance(self):
        """Main rebalancing with extensive logging"""
        self.rebalance_count += 1
        self.Log(f"🔄 Rebalance #{self.rebalance_count}")
        
        if self.last_price is None:
            self.Log("❌ No price data")
            return
        
        # Initialize SPY position
        if not self.Portfolio[self.spy.Symbol].Invested:
            self.SetHoldings(self.spy.Symbol, self.target_spy_allocation)
            self.Log(f"✅ Initialized SPY position at ${self.last_price:.2f}")
        
        # Check if we have enough data
        if len(self.price_history) < 5:
            self.Log(f"❌ Insufficient price history: {len(self.price_history)}")
            return
        
        # Create observation
        observation = self.CreateObservation()
        if observation is None:
            self.Log("❌ Could not create observation")
            return
        
        self.Log(f"✅ Created observation: {observation[:3]}...")
        
        # Get model prediction
        actions = self.GetModelPrediction(observation)
        if actions is None:
            self.Log("❌ Model prediction failed")
            return
        
        self.prediction_count += 1
        self.Log(f"✅ Model prediction #{self.prediction_count}: [{actions[0]:.3f}, {actions[1]:.3f}]")
        
        # Execute trades
        self.ExecuteTrades(actions)
    
    def CreateObservation(self):
        """Create 13-feature observation vector"""
        try:
            # Price features
            current_price = self.last_price
            price_return_1 = (self.price_history[-1] / self.price_history[-2] - 1) if len(self.price_history) >= 2 else 0
            price_return_5 = (self.price_history[-1] / self.price_history[-5] - 1) if len(self.price_history) >= 5 else 0
            
            # Vol features
            current_vol = self.last_vol
            vol_change = 0 if len(self.vol_history) < 2 else (self.vol_history[-1] / self.vol_history[-2] - 1)
            
            # Position features
            call_pos_norm = self.current_call_contracts / self.max_contracts_per_type
            put_pos_norm = self.current_put_contracts / self.max_contracts_per_type
            
            # Simple Greeks calculation
            time_to_expiry = 30/252
            d1 = 0.5  # Simplified for ATM
            call_delta = 0.5
            put_delta = -0.5
            
            # Portfolio Greeks
            portfolio_delta = (call_delta * self.current_call_contracts + put_delta * self.current_put_contracts) * 100
            portfolio_gamma = 0.1 * (abs(self.current_call_contracts) + abs(self.current_put_contracts))
            
            observation = np.array([
                current_price / 400.0,  # Normalized price
                price_return_1,
                price_return_5,
                vol_change,
                call_pos_norm,
                put_pos_norm,
                1.0,  # moneyness call (ATM)
                1.0,  # moneyness put (ATM)
                current_vol,
                call_delta,
                put_delta,
                portfolio_delta / 10000.0,
                portfolio_gamma / 10000.0
            ], dtype=np.float32)
            
            return observation
            
        except Exception as e:
            self.Log(f"Error creating observation: {e}")
            return None
    
    def GetModelPrediction(self, observation):
        """Get prediction from trained model or fallback"""
        if self.model_loaded and self.model is not None:
            try:
                # Normalize observation
                normalized_obs = (observation - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
                
                # Convert to tensor
                obs_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).unsqueeze(0)
                
                # Get prediction
                with torch.no_grad():
                    actions, self.hidden_states = self.model(obs_tensor, self.hidden_states)
                    actions = actions.squeeze().numpy()
                
                actions = np.clip(actions, -1, 1)
                return actions
                
            except Exception as e:
                self.Log(f"Model prediction error: {e}")
                return self.GetFallbackPrediction(observation)
        else:
            return self.GetFallbackPrediction(observation)
    
    def GetFallbackPrediction(self, observation):
        """Fallback prediction logic"""
        # Use price momentum for simple trading
        price_return_5 = observation[2]  # 5-period return
        
        # Simple momentum strategy
        call_signal = np.clip(price_return_5 * 3, -1, 1)  # Buy calls if price rising
        put_signal = np.clip(-price_return_5 * 3, -1, 1)  # Buy puts if price falling
        
        return np.array([call_signal, put_signal])
    
    def ExecuteTrades(self, actions):
        """Execute option trades with detailed logging"""
        if self.current_call_symbol is None or self.current_put_symbol is None:
            self.Log("❌ No options available for trading")
            return
        
        self.trade_attempt_count += 1
        self.Log(f"🔄 Trade attempt #{self.trade_attempt_count}")
        
        # Scale to trade sizes
        call_trade = int(actions[0] * self.max_trade_per_step)
        put_trade = int(actions[1] * self.max_trade_per_step)
        
        self.Log(f"Raw actions: [{actions[0]:.3f}, {actions[1]:.3f}]")
        self.Log(f"Scaled trades: Call={call_trade}, Put={put_trade}")
        
        # Execute call trade
        if call_trade != 0:
            new_call_pos = max(-self.max_contracts_per_type, 
                              min(self.max_contracts_per_type, 
                                  self.current_call_contracts + call_trade))
            trade_qty = new_call_pos - self.current_call_contracts
            
            if trade_qty != 0:
                self.Log(f"📞 Attempting call trade: {trade_qty} contracts")
                try:
                    self.MarketOrder(self.current_call_symbol, trade_qty)
                    self.current_call_contracts = new_call_pos
                    self.Log(f"✅ Call trade executed: {trade_qty}")
                except Exception as e:
                    self.Log(f"❌ Call trade failed: {e}")
        
        # Execute put trade
        if put_trade != 0:
            new_put_pos = max(-self.max_contracts_per_type,
                             min(self.max_contracts_per_type,
                                 self.current_put_contracts + put_trade))
            trade_qty = new_put_pos - self.current_put_contracts
            
            if trade_qty != 0:
                self.Log(f"📞 Attempting put trade: {trade_qty} contracts")
                try:
                    self.MarketOrder(self.current_put_symbol, trade_qty)
                    self.current_put_contracts = new_put_pos
                    self.Log(f"✅ Put trade executed: {trade_qty}")
                except Exception as e:
                    self.Log(f"❌ Put trade failed: {e}")
        
        if call_trade == 0 and put_trade == 0:
            self.Log("➡️ No trades needed (signals too small)")
    
    def OnOrderEvent(self, orderEvent):
        """Log all order events"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"🎯 FILL: {orderEvent.Symbol} qty: {orderEvent.FillQuantity} @ ${orderEvent.FillPrice:.2f}")
        elif orderEvent.Status == OrderStatus.Invalid:
            self.Log(f"❌ REJECTED: {orderEvent.Symbol} - {orderEvent.Message}")

# Simple RL Model class (same as before)
class SimpleRLModel(torch.nn.Module):
    def __init__(self, obs_dim=13, action_dim=2, hidden_dim=64):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Tanh()
        )
    
    def forward(self, obs, hidden_states):
        features = self.feature_extractor(obs)
        lstm_out, new_hidden_states = self.lstm(features, hidden_states)
        actions = self.policy_head(lstm_out)
        return actions, new_hidden_states
