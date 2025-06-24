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
        self.spy = self.AddEquity("SPY", Resolution.Minute)
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
        # Add options universe for SPY - FIXED API METHODS
        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(self.option_filter)
        
        # Set warmup period - FIXED API METHOD
        self.SetWarmUp(timedelta(days=5))
        
        # Initialize our RL model
        self.model_wrapper = ModelWrapper(self)
        self.option_calculator = OptionCalculator()
          # Environment state tracking - FIXED: MATCH TRAINING EXACTLY
        self.shares_to_hedge = 10000  # CRITICAL FIX: Must match training (was 2000)
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
        self.vol_history = []        # Add data validation tracking
        self.warmup_complete = False
        self.available_options = {}  # Track available option symbols with valid data
        self.prediction_count = 0  # Track number of predictions for debugging
        self.spy_order_placed = False  # Track if initial SPY order has been placed
        
        # CRITICAL FIX: Add step tracking to match training environment
        self.current_step = 0  # Track steps for consistent time calculation
        self.episode_length = 252  # Approximate trading days in a year
        self.S_t_minus_1 = None  # Track previous price for consistent lagged returns
        self.v_t_minus_1 = None  # Track previous volatility
        self.initial_S0_for_episode = None  # CRITICAL: Store episode start price like training# Schedule rebalancing - FIXED: Stagger to prevent overlapping triggers
        # First rebalance at market open
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 0),
            self.rebalance
        )
        # Then hourly starting at 10:30 ET (60 minutes after open)
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 60),
            self.rebalance_hourly
        )
        self.Log("*** RL Hedging Algorithm Initialized ***")
        
        # Store canonical symbol for option chains
        self.spy_canonical = self.spy.Symbol
        
    def option_filter(self, universe):
        """Filter options to get a wider range for debugging"""
        return (universe
            .Strikes(-10, 10)  # Much wider range for debugging
            .Expiration(timedelta(5), timedelta(90))  # Much wider expiration range for debugging
            )
            
    def OnData(self, data):
        """Main data handler - CRITICAL FIX FOR OPTION CHAIN ACCESS"""
        # Skip processing during warmup period
        if self.IsWarmingUp:
            return
            
        if not self.warmup_complete:
            self.warmup_complete = True
            self.Log("*** WARMUP PERIOD COMPLETE - STARTING TRADING ***")
          # Update SPY price if available
        current_price = None
        if self.spy.Symbol in data.Bars:
            current_price = data.Bars[self.spy.Symbol].Close
            if current_price > 0:  # Ensure valid price
                if self.last_price is not None:
                    self.price_history.append(current_price)
                self.last_price = current_price
                self.Log(f"*** OnData: Updated SPY price to ${current_price:.2f} ***")
        elif self.spy.Symbol in data and hasattr(data[self.spy.Symbol], 'Price'):
            # Try alternative price source
            current_price = data[self.spy.Symbol].Price
            if current_price > 0:
                if self.last_price is not None:
                    self.price_history.append(current_price)
                self.last_price = current_price
                self.Log(f"*** OnData: Updated SPY price from alternative source to ${current_price:.2f} ***")
        else:
            # Try to get price from Securities as fallback
            if self.Securities[self.spy.Symbol].Price > 0:
                current_price = self.Securities[self.spy.Symbol].Price
                if self.last_price != current_price:  # Only update if different
                    if self.last_price is not None:
                        self.price_history.append(current_price)
                    self.last_price = current_price
                    self.Log(f"*** OnData: Updated SPY price from Securities to ${current_price:.2f} ***")
        
        # CRITICAL FIX: Process option chains using proper QuantConnect pattern
        if hasattr(data, 'OptionChains') and data.OptionChains.Count > 0:
            self.Log(f"*** OnData: Processing {data.OptionChains.Count} option chains ***")
            
            # Access option chain for SPY using canonical symbol
            if self.spy_canonical in data.OptionChains:
                spy_chain = data.OptionChains[self.spy_canonical]
                self.Log(f"*** Found SPY option chain with {len(spy_chain)} contracts ***")
                
                # Update available options from chain data
                self.update_available_options_from_chain(spy_chain)
                
                # Calculate implied volatility if we have current price
                if current_price is not None:
                    self.update_implied_volatility(spy_chain, current_price)
            else:
                self.Log(f"*** SPY canonical symbol {self.spy_canonical} not found in option chains ***")
                # Clear available options if no chain data
                self.available_options.clear()
        else:
            # No option chain data available
            if hasattr(data, 'OptionChains'):
                self.Log(f"*** OnData: No option chains available (Count: {data.OptionChains.Count}) ***")
            else:
                self.Log("*** OnData: No OptionChains attribute in data ***")
            self.available_options.clear()
              # Keep history manageable
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-50:]
        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-50:]

    def update_available_options_from_chain(self, chain):
        """Process option chain data and update available options - CRITICAL NEW METHOD"""
        self.available_options.clear()
        total_contracts = len(chain)
        filtered_contracts = 0
        
        self.Log(f"*** Processing option chain with {total_contracts} contracts ***")
        
        for contract in chain:
            # Check if option has valid pricing data
            has_price = contract.LastPrice > 0 or (contract.BidPrice > 0 and contract.AskPrice > 0 and contract.AskPrice > contract.BidPrice)
            
            if has_price and self.last_price is not None:
                strike_diff_pct = abs(contract.Strike - self.last_price) / self.last_price
                time_to_expiry = (contract.Expiry.date() - self.Time.date()).days
                
                # WIDENED filtering for better option availability
                if strike_diff_pct <= 0.25 and 5 <= time_to_expiry <= 90:  # 25% strike range, 5-90 days
                    self.available_options[contract.Symbol] = contract
                    filtered_contracts += 1
                      # Log first few contracts for debugging
                    if filtered_contracts <= 3:
                        self.Log(f"Added option: {contract.Symbol}, Strike: ${contract.Strike}, DTE: {time_to_expiry}, Price: ${contract.LastPrice:.2f}")
        
        self.Log(f"*** Option filtering complete: {total_contracts} total, {filtered_contracts} passed filters ***")
        if self.last_price:
            self.Log(f"*** Current SPY price: ${self.last_price:.2f} ***")

    def update_available_options(self, data=None):
        """Legacy method - now redirects to chain-based processing"""
        # This method is kept for compatibility but main processing is now in update_available_options_from_chain
        if data and hasattr(data, 'OptionChains') and data.OptionChains.Count > 0:
            self.Log("*** Using legacy update_available_options - redirecting to chain processing ***")
            for chain in data.OptionChains.Values:
                self.update_available_options_from_chain(chain)
        else:
            self.available_options.clear()
            self.Log("*** No option chain data in legacy method ***")

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
                if holding.Quantity != 0 and holding.Price > 0:                    total_value += holding.Quantity * holding.Price * 100  # Options are per 100 shares
        return total_value

    def update_market_data(self) -> bool:
        """Update market data and return success status"""
        if self.last_price is None:
            self.Log("update_market_data: No last_price available")
            return False
        
        # Initialize price history if empty but we have current price
        if len(self.price_history) == 0 and self.last_price is not None:
            self.price_history.append(self.last_price)
            self.Log(f"update_market_data: Initialized price history with current price ${self.last_price:.2f}")
        
        # We need at least some price data
        success = len(self.price_history) > 0
        self.Log(f"update_market_data: Success={success}, Price history length={len(self.price_history)}")
        return success

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
            # CRITICAL FIX: Try to get price directly from securities if OnData isn't working
            if self.Securities[self.spy.Symbol].Price > 0:
                self.last_price = self.Securities[self.spy.Symbol].Price
                self.Log(f"*** GOT PRICE FROM SECURITIES: ${self.last_price:.2f} ***")
            else:
                self.Log("No price data available yet, skipping rebalance")
                return        # FIXED: Initialize SPY position with order-in-flight guard
        if not self.Portfolio[self.spy.Symbol].Invested:
            # CRITICAL FIX: Set initial S0 if not already set (for consistency)
            if self.initial_S0_for_episode is None:
                self.initial_S0_for_episode = self.last_price
                if self.initial_S0_for_episode < 1e-6:
                    self.initial_S0_for_episode = 1.0
                self.Log(f"*** Set initial S0 for episode in rebalance: ${self.initial_S0_for_episode:.2f} ***")
                
            # Check for existing SPY orders to prevent duplicates
            open_orders = [t for t in self.Transactions.GetOpenOrders(self.spy.Symbol) 
                          if t.OrderType == OrderType.Market and t.Status == OrderStatus.Submitted]
            
            if open_orders:
                self.Log("*** Existing SPY order in flight - skipping duplicate order ***")
                return  # Wait for existing order to fill
            elif not self.spy_order_placed:
                # Initialize hedge position
                target_shares = self.shares_to_hedge
                self.Log(f"*** INITIALIZING SPY POSITION ***: {target_shares} shares at ${self.last_price:.2f}")
                order_ticket = self.MarketOrder(self.spy.Symbol, target_shares)
                if order_ticket:
                    self.Log(f"*** SPY ORDER PLACED SUCCESSFULLY ***: {target_shares} shares")
                    self.spy_order_placed = True  # Mark that we've placed the order
                else:
                    self.Log("*** CRITICAL ERROR: FAILED TO PLACE SPY ORDER ***")
                    return  # Don't continue if we can't place the SPY order
            else:
                self.Log("*** SPY order already placed but not filled yet - waiting ***")
                return
        else:
            spy_qty = self.Portfolio[self.spy.Symbol].Quantity
            self.Log(f"SPY position exists: {spy_qty} shares")

        # Update price and volatility data - improved logic
        if not self.update_market_data():
            self.Log("Failed to update market data, skipping option trades")
            # Don't return here - we should still allow SPY trades to complete
            # Only skip option trades
            return        # Check if we have any available options - use chain provider if needed
        if not self.available_options:
            self.Log("No options in cache - fetching from chain provider")
            self.refresh_option_chain_from_provider()
            
        if not self.available_options:
            self.Log("No options with valid pricing data available, skipping option trades")
            return
            
        # Get current observation
        observation = self.get_observation()
        if observation is None:
            # CRITICAL FIX: Use fallback volatility if no option data
            if self.last_vol is None:
                self.last_vol = 0.20  # Use 20% as fallback volatility
                self.Log("*** USING FALLBACK VOLATILITY: 20% ***")
            observation = self.get_observation()
            
        if observation is None:
            self.Log("Could not create observation even with fallbacks, skipping rebalance")
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
        
        # CRITICAL FIX: Update step tracking and previous values to match training environment
        self.S_t_minus_1 = self.last_price
        self.v_t_minus_1 = self.last_vol
        self.current_step += 1
    
    def get_observation(self) -> np.ndarray:
        """Create observation vector EXACTLY matching training environment format"""
        self.prediction_count += 1
        if self.last_price is None or self.last_vol is None:
            self.Log(f"Missing data - Price: {self.last_price}, Vol: {self.last_vol}")
            # Try to set fallback volatility
            if self.last_vol is None:
                self.last_vol = 0.20  # 20% fallback
                self.Log("*** SET FALLBACK VOLATILITY: 20% ***")            
            if self.last_price is None:
                return None
                
        if len(self.price_history) < 2:
            self.Log(f"Insufficient price history: {len(self.price_history)}")
            # Add current price to history if we have it
            if self.last_price is not None and len(self.price_history) == 0:
                self.price_history.append(self.last_price)
                self.Log(f"*** INITIALIZED PRICE HISTORY WITH CURRENT PRICE: ${self.last_price:.2f} ***")
            if len(self.price_history) < 2:
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
          # CRITICAL FIX: EXACTLY match training environment normalization
        # Training uses: s0_safe_obs = np.maximum(self.initial_S0_for_episode, 25.0)
        if self.initial_S0_for_episode is None:
            # Emergency fallback if S0 not set
            self.initial_S0_for_episode = self.last_price if self.last_price else 100.0
            self.Log(f"*** Emergency S0 fallback: ${self.initial_S0_for_episode:.2f} ***")
            
        s0_safe_obs = max(self.initial_S0_for_episode, 25.0)
        norm_S_t = S_t / s0_safe_obs
        norm_C_t = C_t / s0_safe_obs  
        norm_P_t = P_t / s0_safe_obs
          # Position normalization (same as training)
        norm_call_held = self.current_call_contracts / self.max_contracts_per_type
        norm_put_held = self.current_put_contracts / self.max_contracts_per_type
        
        # CRITICAL FIX: Time feature - match training environment exactly  
        # Training uses: (episode_length - current_step) / episode_length
        norm_time_to_end = (self.episode_length - self.current_step) / self.episode_length if self.episode_length != 0 else 0.0
        norm_time_to_end = max(0.0, min(1.0, norm_time_to_end))  # Ensure bounds [0,1]
        
        # Calculate Greeks for ATM options (same as training)
        K_atm_t = round(S_t)  # Round to nearest dollar
        call_greeks = self.option_calculator.calculate_greeks(S_t, K_atm_t, 30/252, 0.04, v_t, 'call')
        put_greeks = self.option_calculator.calculate_greeks(S_t, K_atm_t, 30/252, 0.04, v_t, 'put')
        
        call_delta = call_greeks['delta']
        call_gamma = call_greeks['gamma']
        put_delta = put_greeks['delta']
        put_gamma = call_gamma  # Same for calls and puts at same strike
          # CRITICAL FIX: Single-step returns - match training environment exactly
        # Training uses: (S_t - S_t_minus_1) / S_t_minus_1 if current_step > 0 else 0.0
        if self.current_step == 0 or self.S_t_minus_1 is None or self.S_t_minus_1 == 0:
            lagged_S_return = 0.0
        else:
            lagged_S_return = (S_t - self.S_t_minus_1) / self.S_t_minus_1
            
        # CRITICAL FIX: Volatility change - match training environment exactly  
        # Training uses: v_t - v_t_minus_1 if current_step > 0 else 0.0
        if self.current_step == 0 or self.v_t_minus_1 is None:
            lagged_v_change = 0.0
        else:
            lagged_v_change = v_t - self.v_t_minus_1
        
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
            
            # Track SPY order fills
            if symbol == self.spy.Symbol:
                self.Log(f"*** SPY ORDER FILLED ***: {symbol} quantity: {order_event.FillQuantity}")
                self.Log(f"*** SPY Position now: {self.Portfolio[symbol].Quantity} shares ***")
            
            # Track option order fills  
            elif hasattr(symbol, 'SecurityType') and str(symbol.SecurityType) == "SecurityType.Option":
                self.Log(f"Option order filled: {symbol} quantity: {order_event.FillQuantity}")
    
    def OnWarmupFinished(self):
        """Seed price and volatility buffers after warmup to prevent market data failures"""
        self.Log("*** WARMUP FINISHED - SEEDING DATA BUFFERS ***")
          # Grab the last 2 closing prices so price_history is primed
        try:
            history = self.History(self.spy.Symbol, 2, Resolution.Minute)
            if not history.empty:
                self.price_history = list(history['close'])
                self.last_price = self.price_history[-1]
                # CRITICAL FIX: Set initial S0 to match training environment
                self.initial_S0_for_episode = self.price_history[0] if len(self.price_history) > 0 else self.last_price
                if self.initial_S0_for_episode < 1e-6:
                    self.initial_S0_for_episode = 1.0
                self.Log(f"*** Seeded price history with {len(self.price_history)} points, last price: ${self.last_price:.2f} ***")
                self.Log(f"*** Set initial S0 for episode: ${self.initial_S0_for_episode:.2f} ***")
            else:
                # Fallback: use current security price
                if self.Securities[self.spy.Symbol].Price > 0:
                    self.last_price = self.Securities[self.spy.Symbol].Price
                    self.price_history = [self.last_price]
                    self.initial_S0_for_episode = self.last_price
                    self.Log(f"*** Fallback: Seeded with current price: ${self.last_price:.2f} ***")
        except Exception as e:
            self.Log(f"Failed to get price history: {e}")
            # Emergency fallback
            if self.Securities[self.spy.Symbol].Price > 0:
                self.last_price = self.Securities[self.spy.Symbol].Price
                self.price_history = [self.last_price]
        
        # Grab yesterday's IV for a starting point using option chain provider
        try:
            chain = self.OptionChainProvider.GetOptionContractList(self.spy.Symbol, self.Time)
            iv_samples = []
            for contract_symbol in chain:
                if self.Securities.ContainsKey(contract_symbol):
                    iv = self.Securities[contract_symbol].VolatilityModel.Volatility
                    if iv > 0:
                        iv_samples.append(float(iv))
            
            if iv_samples:
                self.last_vol = float(np.nanmean(iv_samples))
                self.vol_history = [self.last_vol]
                self.Log(f"*** Seeded volatility with {len(iv_samples)} IV samples, avg: {self.last_vol:.4f} ***")
            else:
                # Use 20-day historical volatility as fallback instead of hardcoded 0.20
                self.last_vol = self.calculate_historical_volatility()
                self.vol_history = [self.last_vol]
                self.Log(f"*** Used historical volatility fallback: {self.last_vol:.4f} ***")
        except Exception as e:
            self.Log(f"Failed to get IV data: {e}")
            self.last_vol = self.calculate_historical_volatility()
            self.vol_history = [self.last_vol]
    
    def calculate_historical_volatility(self) -> float:
        """Calculate 20-day historical volatility as fallback"""
        try:
            history = self.History(self.spy.Symbol, 21, Resolution.Daily)
            if len(history) >= 20:
                returns = np.log(history['close'] / history['close'].shift(1)).dropna()
                vol = float(returns.std() * np.sqrt(252))  # Annualized
                return max(vol, 0.10)  # Floor at 10%
            else:
                return 0.20  # Final fallback
        except:
            return 0.20  # Final fallback
    
    def rebalance_hourly(self):
        """Hourly rebalancing - options only (SPY position already established)"""
        if self.IsWarmingUp:
            return
            
        self.Log("=== HOURLY REBALANCE - OPTIONS ONLY ===")
        
        # Only proceed if SPY position is established
        if not self.Portfolio[self.spy.Symbol].Invested:
            self.Log("No SPY position yet - skipping hourly rebalance")
            return
            
        # Call main rebalance logic but skip SPY initialization
        self.rebalance_options_only()
    
    def rebalance_options_only(self):
        """Rebalance options without touching SPY position"""
        # Ensure we have market data
        if not self.update_market_data():
            self.Log("Failed to update market data for hourly rebalance")
            return
        
        # Get options using chain provider if needed
        if not self.available_options:
            self.refresh_option_chain_from_provider()
        
        if not self.available_options:
            self.Log("No options available for hourly rebalance")
            return
        
        # Get observation and execute trades (same as main rebalance)
        observation = self.get_observation()
        if observation is None:
            self.Log("Could not create observation for hourly rebalance")
            return
            
        # Get RL model prediction and execute trades
        actions = self.model_wrapper.predict(observation)
        if actions is None:
            actions = np.array([0.0, 0.0])
            
        self.Log(f"Hourly rebalance actions: call={actions[0]:.6f}, put={actions[1]:.6f}")
        self.execute_option_trades(actions)
    
    def refresh_option_chain_from_provider(self):
        """Fetch option chain on-demand using provider to avoid timing races"""
        try:
            chain = self.OptionChainProvider.GetOptionContractList(self.spy.Symbol, self.Time)
            filtered_contracts = 0
            self.available_options.clear()
            
            for contract_symbol in chain:
                if self.Securities.ContainsKey(contract_symbol):
                    contract = self.Securities[contract_symbol]
                    
                    # Check if option has valid pricing data
                    has_price = contract.Price > 0 or (contract.BidPrice > 0 and contract.AskPrice > 0)
                    
                    if has_price and self.last_price is not None:
                        strike_price = contract_symbol.ID.StrikePrice
                        strike_diff_pct = abs(strike_price - self.last_price) / self.last_price
                        time_to_expiry = (contract_symbol.ID.Date.date() - self.Time.date()).days
                        
                        # Same filtering as before
                        if strike_diff_pct <= 0.25 and 5 <= time_to_expiry <= 90:
                            self.available_options[contract_symbol] = contract
                            filtered_contracts += 1
            
            self.Log(f"*** Chain provider found {len(chain)} contracts, {filtered_contracts} passed filter ***")
            
        except Exception as e:
            self.Log(f"Failed to refresh option chain from provider: {e}")
        
        # Set option margin model for proper buying power management
        self.SetBuyingPowerModel(OptionMarginModel())
