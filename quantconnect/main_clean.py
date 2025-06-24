# region imports
from AlgorithmImports import *
from datetime import timedelta
import numpy as np
import torch
import joblib
from typing import List, Dict, Any
import os
import sys

# Import our custom classes
from model_wrapper import ModelWrapper
from option_calculator import OptionCalculator
# endregion

class RLHedgingAlgorithm(QCAlgorithm):
    
    def initialize(self):        
        # Set basic algorithm parameters - FIXED API METHODS
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000000)  # $100M for 200 contracts per type
          # Add SPY as our underlying asset - FIXED API METHODS
        self.spy = self.AddEquity("SPY", Resolution.MINUTE)
        self.spy.SetDataNormalizationMode(DataNormalizationMode.RAW)
          # Add options universe for SPY - FIXED API METHODS
        option = self.AddOption("SPY", Resolution.MINUTE)
        option.SetFilter(self.option_filter)
        
        # Set warmup period - FIXED API METHOD
        self.SetWarmUp(timedelta(days=5))
        
        # Initialize our RL model
        self.model_wrapper = ModelWrapper(self)
        self.option_calculator = OptionCalculator()
        
        # Environment state tracking - MATCH TRAINING EXACTLY
        self.shares_to_hedge = 2000
        self.current_call_contracts = 0
        self.current_put_contracts = 0
        self.max_contracts_per_type = 200
        self.max_trade_per_step = 15
        self.option_multiplier = 100
        self.transaction_cost_per_contract = 0.05
        
        # State tracking for observations
        self.last_price = None
        self.last_vol = None
        self.price_history = []
        self.vol_history = []
        
        # Add data validation tracking
        self.warmup_complete = False
        self.available_options = {}  # Track available option symbols with valid data
        self.prediction_count = 0  # Track number of predictions for debugging
        
        # Schedule rebalancing to start at market open, then every hour
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 0),
            self.rebalance
        )
        
        # Additional rebalancing every hour after market open
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.Every(timedelta(minutes=60)),
            self.rebalance
        )
        self.Log("*** RL Hedging Algorithm Initialized ***")
        
    def option_filter(self, universe):
        """Filter options to match training: 30-day options exactly as used in training environment"""
        return (universe
            .Strikes(-4, 4)  # Widened for better coverage
            .Expiration(timedelta(25), timedelta(35))  # 30±5 days to match training
            )
    
    def OnData(self, data):
        """Main data handler - FIXED CRITICAL ISSUE WITH API METHODS"""
        # Skip processing during warmup period
        if self.IsWarmingUp:
            return
            
        if not self.warmup_complete:
            self.warmup_complete = True
            self.Log("*** WARMUP PERIOD COMPLETE - STARTING TRADING ***")
        
        if not self.spy.Symbol in data.Bars:
            return
            
        # Update current price
        current_price = data.Bars[self.spy.Symbol].Close
        if current_price > 0:  # Ensure valid price
            if self.last_price is not None:
                self.price_history.append(current_price)
                
            self.last_price = current_price
        
        # Update available options with valid data - CRITICAL FIX
        self.update_available_options(data)
        
        # Calculate implied volatility from options if available - CRITICAL FIX
        if hasattr(data, 'OptionChains') and data.OptionChains.Count > 0:
            for chain in data.OptionChains.Values:
                self.update_implied_volatility(chain, current_price)
        
        # Keep history manageable
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-50:]
        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-50:]

    def update_available_options(self, data=None):
        """Track which options have valid pricing data - FIXED CRITICAL ISSUE"""
        self.available_options.clear()
        if data and hasattr(data, 'OptionChains') and data.OptionChains.Count > 0:
            for chain in data.OptionChains.Values:
                for contract in chain:
                    # Check if option has valid pricing data and is reasonably liquid
                    if (contract.LastPrice > 0 or 
                        (contract.BidPrice > 0 and contract.AskPrice > 0 and 
                         contract.AskPrice > contract.BidPrice)):                        
                        # Additional filter for reasonable strikes and DTE
                        if self.last_price is not None:
                            strike_diff_pct = abs(contract.Strike - self.last_price) / self.last_price
                            time_to_expiry = (contract.Expiry.date() - self.Time.date()).days
                            # TRAINING MATCH: Use exactly 30 days to match T_OPTION_TENOR = 30/252 from training
                            if strike_diff_pct <= 0.10 and 25 <= time_to_expiry <= 35:
                                self.available_options[contract.Symbol] = contract
                                
        # CRITICAL DEBUG: Log how many options we found
        self.Log(f"*** FOUND {len(self.available_options)} VALID OPTIONS ***")

    def update_implied_volatility(self, chain, current_price):
        """Extract implied volatility from option chain - FIXED CRITICAL ISSUE"""
        atm_options = []
        
        for contract in chain:
            if abs(contract.Strike - current_price) < current_price * 0.02:  # Within 2%
                atm_options.append(contract)
        
        if atm_options:
            # Use average implied volatility of ATM options
            valid_ivs = [opt.ImpliedVolatility for opt in atm_options if opt.ImpliedVolatility > 0]
            if valid_ivs:                
                avg_iv = np.mean(valid_ivs)
                self.vol_history.append(avg_iv)
                self.last_vol = avg_iv
                self.Log(f"*** IMPLIED VOL UPDATED: {avg_iv:.4f} ***")

    def get_total_options_value(self) -> float:
        """Calculate total market value of current option positions"""
        total_value = 0.0
        for symbol in self.Portfolio.keys():
            if hasattr(symbol, 'SecurityType') and str(symbol.SecurityType) == "SecurityType.Option":
                holding = self.Portfolio[symbol]
                if holding.Quantity != 0 and holding.Price > 0:
                    total_value += holding.Quantity * holding.Price * 100  # Options are per 100 shares
        return total_value

    def update_market_data(self) -> bool:
        """Update market data and return success status"""
        if self.last_price is None:
            return False
        # Simple validation - we already update price in OnData
        return len(self.price_history) > 0

    def rebalance(self):
        """Main rebalancing logic using RL model predictions with enhanced debugging"""
        # Don't trade during warmup
        if self.IsWarmingUp:
            self.Log("Skipping rebalance - still in warmup period")
            return
            
        # Reset LSTM hidden states at start of new trading day
        if self.Time.hour == 9 and self.Time.minute == 30:  # Market open
            self.model_wrapper.reset_hidden_states()
            
        self.Log("=== REBALANCE METHOD CALLED ===")  # Debug: Verify rebalance is being called
            
        # Log current positions before rebalancing
        spy_qty = self.Portfolio[self.spy.Symbol].Quantity if self.spy.Symbol in self.Portfolio else 0
        self.Log(f"=== REBALANCE START === SPY: {spy_qty} shares, Call: {self.current_call_contracts}, Put: {self.current_put_contracts}")
        self.Log(f"Portfolio value: ${self.Portfolio.TotalPortfolioValue:,.0f}, Cash: ${self.Portfolio.Cash:,.0f}")
        
        # Check if we have valid price data first
        if self.last_price is None:
            self.Log("No price data available yet, skipping rebalance")
            return
            
        # FIXED: Initialize SPY position FIRST, before checking for options data
        if not self.Portfolio[self.spy.Symbol].Invested:
            # Initialize hedge position
            target_shares = self.shares_to_hedge
            self.Log(f"*** INITIALIZING SPY POSITION ***: {target_shares} shares at ${self.last_price:.2f}")
            order_ticket = self.MarketOrder(self.spy.Symbol, target_shares)
            if order_ticket:
                self.Log(f"*** SPY ORDER PLACED SUCCESSFULLY ***: {target_shares} shares")
            else:
                self.Log("*** CRITICAL ERROR: FAILED TO PLACE SPY ORDER ***")
        else:
            spy_qty = self.Portfolio[self.spy.Symbol].Quantity
            self.Log(f"SPY position already exists: {spy_qty} shares")
        
        # Update price and volatility data
        if not self.update_market_data():
            self.Log("Failed to update market data, skipping option trades")
            return
            
        # Check if we have any available options (only for options trading, not SPY)
        if not self.available_options:
            self.Log("No options with valid pricing data available, skipping option trades only")
            return
        
        # Get current observation
        observation = self.get_observation()
        if observation is None:
            self.Log("Could not create observation, skipping rebalance")
            return
            
        # Log observation details
        self.Log(f"Observation vector length: {len(observation)}, first 6 values: {observation[:6]}")
        
        # Get RL model prediction
        actions = self.model_wrapper.predict(observation)
        if actions is None:
            self.Log("Model prediction failed, using fallback action")
            # Simple fallback: no action
            actions = np.array([0.0, 0.0])

        self.Log(f"Model predicted actions: call={actions[0]:.6f}, put={actions[1]:.6f}")
        
        # Add detailed logging for debugging flat exposure
        self.Log(f"Raw model output: {actions}")
        self.Log(f"Model output after clipping: {np.clip(actions, -1, 1)}")
        self.Log(f"Max trade per step: {self.max_trade_per_step}")
        self.Log(f"Action scaling before rounding: call={actions[0] * self.max_trade_per_step:.3f}, put={actions[1] * self.max_trade_per_step:.3f}")

        # Check if actions are meaningful
        if abs(actions[0]) < 0.001 and abs(actions[1]) < 0.001:
            self.Log("WARNING: Model predicted nearly zero actions - may indicate prediction issue")
            
        # Execute trades based on RL actions
        self.execute_option_trades(actions)
        
        # Log positions after trades
        self.Log(f"=== REBALANCE END === Call: {self.current_call_contracts}, Put: {self.current_put_contracts}")
        self.Log(f"Option exposure - Total options value: ${self.get_total_options_value():,.0f}")
    
    def get_observation(self) -> np.ndarray:
        """Create observation vector EXACTLY matching training environment format"""
        self.prediction_count += 1
        
        if self.last_price is None or self.last_vol is None:
            self.Log("Missing price or volatility data for observation")
            return None
            
        if len(self.price_history) < 2:
            self.Log(f"Insufficient price history: {len(self.price_history)}")
            return None
        
        # Match training environment exactly
        S_t = self.last_price
        v_t = self.last_vol
        
        # Get current option prices from available options (approximation)
        C_t = 0.0
        P_t = 0.0
        if self.available_options:
            # Use first available call/put as proxy for ATM prices
            for symbol, contract in self.available_options.items():
                if hasattr(symbol.ID, 'OptionRight') and str(symbol.ID.OptionRight) == "OptionRight.Call" and C_t == 0.0:
                    C_t = contract.LastPrice if contract.LastPrice > 0 else contract.AskPrice
                elif hasattr(symbol.ID, 'OptionRight') and str(symbol.ID.OptionRight) == "OptionRight.Put" and P_t == 0.0:
                    P_t = contract.LastPrice if contract.LastPrice > 0 else contract.AskPrice
        
        # Use fallback Black-Scholes if no market prices
        if C_t <= 0 or P_t <= 0:
            time_to_expiry = 30/252  # 30 days
            call_greeks = self.option_calculator.calculate_greeks(S_t, S_t, time_to_expiry, 0.04, v_t, 'call')
            put_greeks = self.option_calculator.calculate_greeks(S_t, S_t, time_to_expiry, 0.04, v_t, 'put')
            if C_t <= 0:
                C_t = call_greeks.get('price', S_t * 0.05)  # fallback 5% of stock price
            if P_t <= 0:
                P_t = put_greeks.get('price', S_t * 0.05)
        
        # EXACTLY match training environment normalization
        s0_safe_obs = max(self.price_history[0] if self.price_history else S_t, 25.0)  # Use first price as S0
        norm_S_t = S_t / s0_safe_obs
        norm_C_t = C_t / s0_safe_obs  
        norm_P_t = P_t / s0_safe_obs
        
        # Position normalization (same as training)
        norm_call_held = self.current_call_contracts / self.max_contracts_per_type
        norm_put_held = self.current_put_contracts / self.max_contracts_per_type
        
        # Time feature (approximate episode progress)
        days_since_start = (self.Time.date() - self.StartDate.date()).days
        trading_days_elapsed = days_since_start * (5/7)  # Approximate trading days vs calendar days
        episode_progress = min(trading_days_elapsed / 252.0, 1.0)  # Cap at 100%
        norm_time_to_end = max(0.0, 1.0 - episode_progress)  # Remaining time in episode
        
        # Calculate Greeks for ATM options (same as training)
        K_atm_t = round(S_t)  # Round to nearest dollar
        call_greeks = self.option_calculator.calculate_greeks(S_t, K_atm_t, 30/252, 0.04, v_t, 'call')
        put_greeks = self.option_calculator.calculate_greeks(S_t, K_atm_t, 30/252, 0.04, v_t, 'put')
        
        call_delta = call_greeks['delta']
        call_gamma = call_greeks['gamma']
        put_delta = put_greeks['delta']
        put_gamma = call_gamma  # Same for calls and puts at same strike
        
        # Single-step returns (same as training)
        if len(self.price_history) >= 2:
            lagged_S_return = (S_t - self.price_history[-1]) / self.price_history[-1]
        else:
            lagged_S_return = 0.0
            
        # Volatility change (raw difference, same as training)  
        if len(self.vol_history) >= 2:
            lagged_v_change = v_t - self.vol_history[-1]
        else:
            lagged_v_change = 0.0
        
        # Clip to training environment bounds
        lagged_S_return = np.clip(lagged_S_return, -1.0, 1.0)
        lagged_v_change = np.clip(lagged_v_change, -1.0, 1.0)
        
        # Create observation EXACTLY matching training format
        obs = np.array([
            norm_S_t,           # 0: Normalized stock price
            norm_C_t,           # 1: Normalized call price  
            norm_P_t,           # 2: Normalized put price
            norm_call_held,     # 3: Call position normalized
            norm_put_held,      # 4: Put position normalized
            v_t,                # 5: Volatility (raw)
            norm_time_to_end,   # 6: Time remaining
            call_delta,         # 7: Call delta
            call_gamma,         # 8: Call gamma 
            put_delta,          # 9: Put delta
            put_gamma,          # 10: Put gamma
            lagged_S_return,    # 11: Stock return
            lagged_v_change     # 12: Vol change
        ], dtype=np.float32)
        
        # Add observation format validation (for debugging)
        if self.prediction_count <= 3:
            self.Log(f"=== OBSERVATION VALIDATION #{self.prediction_count} ===")
            expected_obs_labels = [
                "norm_S_t", "norm_C_t", "norm_P_t", "norm_call_held", "norm_put_held",
                "v_t", "norm_time_to_end", "call_delta", "call_gamma", 
                "put_delta", "put_gamma", "lagged_S_return", "lagged_v_change"
            ]
            for i, (label, value) in enumerate(zip(expected_obs_labels, obs)):
                self.Log(f"{i:2d}: {label:15s} = {value:.6f}")
        
        # Check for any NaN or infinite values
        if np.any(~np.isfinite(obs)):
            self.Log(f"WARNING: Observation contains NaN or infinite values: {obs}")
            
        self.Log(f"Training-format observation: S={norm_S_t:.3f}, C={norm_C_t:.3f}, P={norm_P_t:.3f}, call_pos={norm_call_held:.3f}, put_pos={norm_put_held:.3f}")
        return obs
    
    def execute_option_trades(self, actions: np.ndarray):
        """Execute option trades based on RL model actions"""
        if len(actions) != 2:
            self.Log(f"Invalid action length: {len(actions)}")
            return
              
        # Scale actions to actual trade sizes - FIXED ROUNDING
        call_trade = int(round(actions[0] * self.max_trade_per_step))
        put_trade = int(round(actions[1] * self.max_trade_per_step))
        
        self.Log(f"Scaled trades: call={call_trade}, put={put_trade} (from actions {actions[0]:.6f}, {actions[1]:.6f})")
        
        # Find ATM options to trade
        call_symbol, put_symbol = self.find_atm_options_with_data()
        
        if call_symbol is None or put_symbol is None:
            self.Log("No valid ATM options found with pricing data")
            return
        
        self.Log(f"Found options: call={call_symbol}, put={put_symbol}")
        
        # Execute call trades
        if call_trade != 0:
            # Use Price check instead of HasData
            if not self.Securities.ContainsKey(call_symbol) or self.Securities[call_symbol].Price <= 0:
                self.Log(f"Call option {call_symbol} does not have valid price, skipping trade")
                return
                
            new_call_position = max(-self.max_contracts_per_type, 
                                   min(self.max_contracts_per_type, 
                                       self.current_call_contracts + call_trade))
            trade_quantity = new_call_position - self.current_call_contracts
            
            if trade_quantity != 0:
                try:
                    # Check buying power before large trades
                    if abs(trade_quantity) > 10:  # For large trades
                        available_buying_power = self.Portfolio.BuyingPower
                        estimated_cost = abs(trade_quantity) * self.Securities[call_symbol].Price * 100
                        if estimated_cost > available_buying_power * 0.5:  # Use max 50% of buying power
                            self.Log(f"WARNING: Large call trade {trade_quantity} may exceed buying power. Cost: ${estimated_cost:,.0f}, Available: ${available_buying_power:,.0f}")
                            
                    self.Log(f"Attempting call trade: {trade_quantity} contracts")
                    order_ticket = self.MarketOrder(call_symbol, trade_quantity)
                    if order_ticket and order_ticket.OrderId > 0:
                        self.current_call_contracts = new_call_position
                        # Log transaction cost
                        cost = abs(trade_quantity) * self.transaction_cost_per_contract
                        self.Log(f"Call trade executed: {trade_quantity} contracts, cost: ${cost:.2f}")                  
                    else:
                        self.Log(f"Failed to place call order for {trade_quantity} contracts")
                except Exception as e:
                    self.Log(f"Error placing call order: {str(e)}")
        
        # Execute put trades  
        if put_trade != 0:
            # Use Price check instead of HasData
            if not self.Securities.ContainsKey(put_symbol) or self.Securities[put_symbol].Price <= 0:
                self.Log(f"Put option {put_symbol} does not have valid price, skipping trade")
                return
                
            new_put_position = max(-self.max_contracts_per_type,
                                  min(self.max_contracts_per_type,
                                      self.current_put_contracts + put_trade))
            trade_quantity = new_put_position - self.current_put_contracts
            
            if trade_quantity != 0:
                try:
                    # Check buying power before large trades
                    if abs(trade_quantity) > 10:  # For large trades
                        available_buying_power = self.Portfolio.BuyingPower
                        estimated_cost = abs(trade_quantity) * self.Securities[put_symbol].Price * 100
                        if estimated_cost > available_buying_power * 0.5:  # Use max 50% of buying power
                            self.Log(f"WARNING: Large put trade {trade_quantity} may exceed buying power. Cost: ${estimated_cost:,.0f}, Available: ${available_buying_power:,.0f}")
                    
                    self.Log(f"Attempting put trade: {trade_quantity} contracts")
                    order_ticket = self.MarketOrder(put_symbol, trade_quantity)
                    if order_ticket and order_ticket.OrderId > 0:
                        self.current_put_contracts = new_put_position                        
                        # Log transaction cost
                        cost = abs(trade_quantity) * self.transaction_cost_per_contract
                        self.Log(f"Put trade executed: {trade_quantity} contracts, cost: ${cost:.2f}")
                    else:
                        self.Log(f"Failed to place put order for {trade_quantity} contracts")
                except Exception as e:
                    self.Log(f"Error placing put order: {str(e)}")
    
    def find_atm_options_with_data(self):
        """Find ATM call and put options that have valid pricing data"""
        if self.last_price is None or not self.available_options:
            return None, None
        call_symbol = None
        put_symbol = None
        min_call_diff = float('inf')  # Separate tracking for calls
        min_put_diff = float('inf')   # Separate tracking for puts
        
        # Look through available options with valid data
        for symbol, contract in self.available_options.items():
            if hasattr(symbol, 'SecurityType') and str(symbol.SecurityType) == "SecurityType.Option" and symbol.Underlying == self.spy.Symbol:
                strike_price = symbol.ID.StrikePrice
                strike_diff = abs(strike_price - self.last_price)
                
                # Separate tracking for call and put strikes
                if hasattr(symbol.ID, 'OptionRight') and str(symbol.ID.OptionRight) == "OptionRight.Call" and strike_diff < min_call_diff:
                    call_symbol = symbol
                    min_call_diff = strike_diff
                elif hasattr(symbol.ID, 'OptionRight') and str(symbol.ID.OptionRight) == "OptionRight.Put" and strike_diff < min_put_diff:
                    put_symbol = symbol
                    min_put_diff = strike_diff
        
        return call_symbol, put_symbol
    
    def OnOrderEvent(self, order_event):
        """Handle order events and update position tracking"""
        if hasattr(order_event, 'Status') and str(order_event.Status) == "OrderStatus.Filled":
            symbol = order_event.Symbol
            if hasattr(symbol, 'SecurityType') and str(symbol.SecurityType) == "SecurityType.Option":
                self.Log(f"Option order filled: {symbol} quantity: {order_event.FillQuantity}")
