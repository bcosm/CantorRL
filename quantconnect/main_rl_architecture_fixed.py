# QuantConnect Algorithm: RL Hedging Strategy with Fixed Architecture
# Load a trained recurrent PPO model for hedging SPY shares using options

from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the model wrapper to the path
sys.path.append('/tmp')

class RLHedgingAlgorithmFixed(QCAlgorithm):

    def Initialize(self):
        # Set start and end dates for backtesting
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 3, 31)
        
        # Set initial cash
        self.SetCash(500000)
        
        # Add SPY as our underlying asset
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
        # Add SPY options
        option = self.AddOption("SPY", Resolution.Daily)
        option.SetFilter(-5, 5, timedelta(days=7), timedelta(days=45))
        self.spy_option = option.Symbol
        
        # Load model files to ObjectStore if not already there
        self._upload_model_files()
        
        # Initialize the model wrapper
        self.model_wrapper = None
        try:
            from model_wrapper_fixed import ModelWrapper
            self.model_wrapper = ModelWrapper(self)
            self.Log("Model wrapper initialized successfully")
        except Exception as e:
            self.Log(f"Failed to initialize model wrapper: {str(e)}")
        
        # Initialize option calculator
        from option_calculator import OptionCalculator
        self.option_calc = OptionCalculator()
        
        # Trading state
        self.shares_to_hedge = 1000  # Conservative position size
        self.spy_shares = 0
        self.call_contracts = {}  # {symbol: quantity}
        self.put_contracts = {}   # {symbol: quantity}
        
        # Previous market data for observation
        self.prev_spy_price = None
        self.prev_vol = 0.2  # Initial volatility estimate
        
        # Market data storage
        self.spy_prices = []
        self.max_history = 20
        
        self.Log(f"Initialized RL Hedging Algorithm with {self.shares_to_hedge} shares to hedge")

    def _upload_model_files(self):
        """Upload model files to ObjectStore if they don't exist"""
        model_files = ["policy_weights.pth", "normalization_stats.pkl", "architecture_info.pkl"]
        
        for filename in model_files:
            try:
                # Check if file exists in ObjectStore
                self.ObjectStore.ReadBytes(filename)
                self.Log(f"Found {filename} in ObjectStore")
            except:
                # File doesn't exist, try to upload from local path
                try:
                    local_path = f"/tmp/model_files/{filename}"
                    with open(local_path, 'rb') as f:
                        file_data = f.read()
                    self.ObjectStore.SaveBytes(filename, file_data)
                    self.Log(f"Uploaded {filename} to ObjectStore")
                except Exception as e:
                    self.Log(f"Failed to upload {filename}: {str(e)}")

    def OnData(self, data):
        """Main trading logic"""
        if not data.HasData:
            return
            
        # Get current SPY price
        if self.spy.Symbol not in data.Bars:
            return
            
        current_spy_price = data.Bars[self.spy.Symbol].Close
        self.spy_prices.append(current_spy_price)
        if len(self.spy_prices) > self.max_history:
            self.spy_prices.pop(0)
            
        # Skip first few bars to build history
        if len(self.spy_prices) < 5:
            self.prev_spy_price = current_spy_price
            return
            
        # Calculate volatility from price history
        if len(self.spy_prices) >= 2:
            returns = []
            for i in range(1, len(self.spy_prices)):
                ret = np.log(self.spy_prices[i] / self.spy_prices[i-1])
                returns.append(ret)
            if returns:
                vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
                self.prev_vol = max(vol, 0.05)  # Minimum volatility floor
        
        # Get available options
        option_chain = data.OptionChains.get(self.spy_option, None)
        if not option_chain:
            self.prev_spy_price = current_spy_price
            return
            
        # Find ATM options
        atm_calls, atm_puts = self._find_atm_options(option_chain, current_spy_price)
        if not atm_calls or not atm_puts:
            self.prev_spy_price = current_spy_price
            return
        
        # Get best ATM options
        call_contract = atm_calls[0]
        put_contract = atm_puts[0]
        
        call_price = call_contract.LastPrice if call_contract.LastPrice > 0 else call_contract.BidPrice
        put_price = put_contract.LastPrice if put_contract.LastPrice > 0 else put_contract.BidPrice
        
        if call_price <= 0 or put_price <= 0:
            self.prev_spy_price = current_spy_price
            return
        
        # Create observation for the model
        observation = self._create_observation(
            current_spy_price, call_price, put_price, call_contract, put_contract
        )
        
        if observation is None:
            self.prev_spy_price = current_spy_price
            return
        
        # Get RL model prediction
        action = None
        if self.model_wrapper and self.model_wrapper.loaded:
            try:
                action = self.model_wrapper.predict(observation)
                if action is not None:
                    self.Log(f"RL Model prediction: call_action={action[0]:.3f}, put_action={action[1]:.3f}")
            except Exception as e:
                self.Log(f"Error getting model prediction: {str(e)}")
        
        # If no model prediction, use simple hedge
        if action is None:
            action = np.array([0.1, -0.1])  # Small conservative hedge
            self.Log("Using fallback hedge action")
        
        # Execute trades based on RL action
        self._execute_trades(action, call_contract, put_contract, current_spy_price)
        
        # Update state
        self.prev_spy_price = current_spy_price
        
        # Log portfolio status periodically
        if self.Time.day % 5 == 0:
            self._log_portfolio_status()

    def _create_observation(self, spy_price, call_price, put_price, call_contract, put_contract):
        """Create observation vector for the RL model"""
        try:
            if self.prev_spy_price is None:
                return None
            
            # Calculate returns and normalized prices
            spy_return = np.log(spy_price / self.prev_spy_price) if self.prev_spy_price > 0 else 0.0
            spy_normalized = spy_price / 100.0  # Normalize around typical SPY price
            
            # Time to expiration (approximate)
            dte = 30.0 / 252.0  # About 30 days, normalized to years
            
            # Risk-free rate
            risk_free_rate = 0.04            # Calculate Greeks using Black-Scholes
            call_delta, put_delta, gamma = self.option_calc.calculate_greeks(
                spy_price, call_contract.Strike, dte, risk_free_rate, self.prev_vol
            )
            
            # Current portfolio Greeks (simplified)
            portfolio_delta = 0.0
            portfolio_gamma = 0.0
            
            # Normalize prices (like original environment)
            s0_safe = max(self.initial_spy_price, 25.0) if hasattr(self, 'initial_spy_price') else 400.0
            norm_spy = spy_price / s0_safe
            norm_call = call_price / s0_safe
            norm_put = put_price / s0_safe
            
            # Normalize positions
            max_contracts = 100.0  # Reasonable max for normalization
            norm_call_held = call_positions / max_contracts
            norm_put_held = put_positions / max_contracts
            
            # Lagged returns (simplified - just use current values for now)
            lagged_spy_return = spy_return
            lagged_vol_change = 0.0  # Simplified
            
            # Create observation vector (13 features matching training environment)
            observation = np.array([
                norm_spy,               # 0: Normalized SPY price
                norm_call,              # 1: Normalized call price
                norm_put,               # 2: Normalized put price
                norm_call_held,         # 3: Normalized call contracts held
                norm_put_held,          # 4: Normalized put contracts held
                self.prev_vol,          # 5: Volatility
                dte,                    # 6: Time to expiration (normalized)
                call_delta,             # 7: Call delta
                gamma,                  # 8: Call gamma
                put_delta,              # 9: Put delta  
                gamma,                  # 10: Put gamma (same as call gamma)
                lagged_spy_return,      # 11: Lagged SPY return
                lagged_vol_change       # 12: Lagged volatility change
            ], dtype=np.float32)
            
            # Basic bounds checking
            observation = np.clip(observation, -10, 10)
            
            return observation
            
        except Exception as e:
            self.Log(f"Error creating observation: {str(e)}")
            return None    def _find_atm_options(self, option_chain, spy_price):
        """Find at-the-money call and put options"""
        calls = []
        puts = []
        
        for option in option_chain:
            # Filter for reasonable DTE (7-45 days)
            dte = (option.Expiry.date() - self.Time.date()).days
            if dte < 7 or dte > 45:
                continue
            
            # Check if we have price data
            if option.LastPrice <= 0 and option.BidPrice <= 0:
                continue
            
            # Separate calls and puts
            if option.Right == OptionRight.Call:
                calls.append(option)
            elif option.Right == OptionRight.Put:
                puts.append(option)
        
        # Sort by proximity to ATM
        calls.sort(key=lambda x: abs(x.Strike - spy_price))
        puts.sort(key=lambda x: abs(x.Strike - spy_price))
        
        return calls[:3], puts[:3]  # Return top 3 closest to ATM

    def _execute_trades(self, action, call_contract, put_contract, spy_price):
        """Execute trades based on RL model output"""
        try:
            # Scale actions to reasonable contract quantities
            max_contracts = min(10, self.shares_to_hedge // 100)  # Max 10 contracts or based on hedge size
            
            call_action = int(action[0] * max_contracts)
            put_action = int(action[1] * max_contracts)
            
            # Current positions
            current_call_qty = self.Portfolio[call_contract.Symbol].Quantity
            current_put_qty = self.Portfolio[put_contract.Symbol].Quantity
            
            # Calculate trade quantities
            call_trade_qty = call_action - current_call_qty
            put_trade_qty = put_action - current_put_qty
            
            # Execute call trade
            if abs(call_trade_qty) >= 1:
                if self._check_buying_power(call_contract.Symbol, call_trade_qty):
                    self.MarketOrder(call_contract.Symbol, call_trade_qty)
                    self.Log(f"Call trade: {call_trade_qty} contracts of {call_contract.Strike}C")
            
            # Execute put trade  
            if abs(put_trade_qty) >= 1:
                if self._check_buying_power(put_contract.Symbol, put_trade_qty):
                    self.MarketOrder(put_contract.Symbol, put_trade_qty)
                    self.Log(f"Put trade: {put_trade_qty} contracts of {put_contract.Strike}P")
            
            # Maintain underlying position if we don't have it
            if self.Portfolio[self.spy.Symbol].Quantity != self.shares_to_hedge:
                shares_to_buy = self.shares_to_hedge - self.Portfolio[self.spy.Symbol].Quantity
                if self._check_buying_power(self.spy.Symbol, shares_to_buy):
                    self.MarketOrder(self.spy.Symbol, shares_to_buy)
                    self.Log(f"SPY position adjustment: {shares_to_buy} shares")
                    
        except Exception as e:
            self.Log(f"Error executing trades: {str(e)}")

    def _check_buying_power(self, symbol, quantity):
        """Check if we have enough buying power for the trade"""
        try:
            if quantity == 0:
                return True
                
            # Conservative buying power check - use 50% of available
            available_buying_power = self.Portfolio.TotalPortfolioValue * 0.4
            current_buying_power = self.Portfolio.Cash
            
            return current_buying_power > available_buying_power
            
        except:
            return True  # If check fails, allow trade

    def _log_portfolio_status(self):
        """Log current portfolio status"""
        total_value = self.Portfolio.TotalPortfolioValue
        cash = self.Portfolio.Cash
        spy_position = self.Portfolio[self.spy.Symbol].Quantity
        
        # Count option positions
        call_positions = 0
        put_positions = 0
        
        for holding in self.Portfolio.Values:
            if holding.Symbol.SecurityType == SecurityType.Option:
                if holding.Symbol.ID.OptionRight == OptionRight.Call:
                    call_positions += holding.Quantity
                elif holding.Symbol.ID.OptionRight == OptionRight.Put:
                    put_positions += holding.Quantity
        
        self.Log(f"Portfolio: Total=${total_value:,.0f}, Cash=${cash:,.0f}, SPY={spy_position}, Calls={call_positions}, Puts={put_positions}")
