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
        # Set basic algorithm parameters - CORRECT API METHODS
        self.set_start_date(2023, 1, 1)  # Fixed: SetStartDate -> set_start_date
        self.set_end_date(2024, 1, 1)    # Fixed: SetEndDate -> set_end_date  
        self.set_cash(100000000)          # FIXED: Increased to $100M for 200 contracts per type (was $30M for 20 contracts)
        
        # Add SPY as our underlying asset - CORRECT API METHODS
        self.spy = self.add_equity("SPY", Resolution.MINUTE)  # Fixed: AddEquity -> add_equity
        self.spy.set_data_normalization_mode(DataNormalizationMode.RAW)  # Fixed: SetDataNormalizationMode -> set_data_normalization_mode
        
        # Add options universe for SPY - CORRECT API METHODS
        option = self.add_option("SPY", Resolution.MINUTE)  # Fixed: AddOption -> add_option
        option.set_filter(self.option_filter)  # Fixed: SetFilter -> set_filter
        
        # Set warmup period - CORRECT API METHOD
        self.set_warm_up(timedelta(days=5))  # Fixed: SetWarmup -> set_warm_up
        
        # Initialize our RL model
        self.model_wrapper = ModelWrapper(self)
        self.option_calculator = OptionCalculator()
          # Environment state tracking - MATCH TRAINING EXACTLY
        self.shares_to_hedge = 2000   # Reduced from 5000 for better margin management
        self.current_call_contracts = 0
        self.current_put_contracts = 0
        self.max_contracts_per_type = 200   # FIXED: Match training max_contracts_held_per_type=200
        self.max_trade_per_step = 15        # FIXED: Match training (was 5, but training used higher limits)
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
        
        # Schedule rebalancing to start at market open, then every hour - FIXED TIMING
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 0),  # Start right at market open, not from midnight
            self.rebalance
        )
        
        # Additional rebalancing every hour after market open
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.every(timedelta(minutes=60)),  # Will fire during market hours
            self.rebalance
        )
        self.log("RL Hedging Algorithm Initialized")  # Fixed: Log -> log
        
    def option_filter(self, universe):
        """Filter options to match training: 30-day options exactly as used in training environment"""
        return (universe
            .strikes(-4, 4)  # Widened from -2,2 to -4,4 for better coverage
            .expiration(timedelta(25), timedelta(35))  # TRAINING MATCH: 30±5 days to match training T_OPTION_TENOR = 30/252
            # Removed: .only_apply_filter_at_market_open() to allow continuous updates
            )
    
    def on_data(self, data):  # Fixed: OnData -> on_data
        """Main data handler - updates price and volatility tracking"""
        # Skip processing during warmup period
        if self.is_warming_up:  # Fixed: IsWarmingUp -> is_warming_up
            return
            
        if not self.warmup_complete:
            self.warmup_complete = True
            self.log("Warmup period complete - starting trading")  # Fixed: Log -> log
        
        if not self.spy.symbol in data.bars:  # Fixed: Symbol -> symbol, Bars -> bars
            return
            
        # Update current price
        current_price = data.bars[self.spy.symbol].close  # Fixed: Close -> close
        if current_price > 0:  # Ensure valid price
            if self.last_price is not None:
                self.price_history.append(current_price)
                
            self.last_price = current_price
        
        # Update available options with valid data
        self.update_available_options(data)
        
        # Calculate implied volatility from options if available
        if data.option_chains.count > 0:  # Fixed: OptionChains.Count -> option_chains.count
            for chain in data.option_chains.values():  # Fixed: Values -> values() with parentheses
                self.update_implied_volatility(chain, current_price)
        
        # Keep history manageable
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-50:]
        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-50:]
    
    def update_available_options(self, data=None):
        """Track which options have valid pricing data"""
        self.available_options.clear()
        if data and data.option_chains.count > 0:  # Fixed: OptionChains.Count -> option_chains.count
            for chain in data.option_chains.values():  # Fixed: Values -> values() with parentheses
                for contract in chain:
                    # Check if option has valid pricing data and is reasonably liquid
                    if (contract.last_price > 0 or   # Fixed: LastPrice -> last_price
                        (contract.bid_price > 0 and contract.ask_price > 0 and   # Fixed: BidPrice/AskPrice -> bid_price/ask_price
                         contract.ask_price > contract.bid_price)):                        
                        # Additional filter for reasonable strikes and DTE
                        if self.last_price is not None:
                            strike_diff_pct = abs(contract.strike - self.last_price) / self.last_price  # Fixed: Strike -> strike
                            time_to_expiry = (contract.expiry.date() - self.time.date()).days  # Fixed: Expiry -> expiry, Time -> time
                            # TRAINING MATCH: Use exactly 30 days to match T_OPTION_TENOR = 30/252 from training
                            if strike_diff_pct <= 0.10 and 25 <= time_to_expiry <= 35:
                                self.available_options[contract.symbol] = contract  # Fixed: Symbol -> symbol
    
    def update_implied_volatility(self, chain, current_price):
        """Extract implied volatility from option chain"""
        atm_options = []
        
        for contract in chain:
            if abs(contract.strike - current_price) < current_price * 0.02:  # Within 2%  # Fixed: Strike -> strike
                atm_options.append(contract)
        
        if atm_options:
            # Use average implied volatility of ATM options
            valid_ivs = [opt.implied_volatility for opt in atm_options if opt.implied_volatility > 0]  # Fixed: ImpliedVolatility -> implied_volatility
            if valid_ivs:                
                avg_iv = np.mean(valid_ivs)
                self.vol_history.append(avg_iv)
                self.last_vol = avg_iv

    def get_total_options_value(self) -> float:
        """Calculate total market value of current option positions"""
        total_value = 0.0
        for symbol in self.portfolio.keys():
            if hasattr(symbol, 'security_type') and str(symbol.security_type) == "SecurityType.Option":
                holding = self.portfolio[symbol]
                if holding.quantity != 0 and holding.price > 0:
                    total_value += holding.quantity * holding.price * 100  # Options are per 100 shares
        return total_value

    def update_market_data(self) -> bool:
        """Update market data and return success status"""
        if self.last_price is None:
            return False
        # Simple validation - we already update price in on_data
        return len(self.price_history) > 0

    def rebalance(self):
        """Main rebalancing logic using RL model predictions with enhanced debugging"""
        # Don't trade during warmup
        if self.is_warming_up:  # Fixed: IsWarmingUp -> is_warming_up
            self.log("Skipping rebalance - still in warmup period")
            return
            
        # Reset LSTM hidden states at start of new trading day
        if self.time.hour == 9 and self.time.minute == 30:  # Market open  # Fixed: Time -> time
            self.model_wrapper.reset_hidden_states()
            
        self.log("=== REBALANCE METHOD CALLED ===")  # Debug: Verify rebalance is being called
            
        # Log current positions before rebalancing
        spy_qty = self.portfolio[self.spy.symbol].quantity if self.spy.symbol in self.portfolio else 0
        self.log(f"=== REBALANCE START === SPY: {spy_qty} shares, Call: {self.current_call_contracts}, Put: {self.current_put_contracts}")
        self.log(f"Portfolio value: ${self.portfolio.total_portfolio_value:,.0f}, Cash: ${self.portfolio.cash:,.0f}")
        
        # Check if we have valid price data first
        if self.last_price is None:
            self.log("No price data available yet, skipping rebalance")  # Fixed: Log -> log
            return
              # FIXED: Initialize SPY position FIRST, before checking for options data
        if not self.portfolio[self.spy.symbol].invested:  # Fixed: Portfolio -> portfolio, Symbol -> symbol, Invested -> invested
            # Initialize hedge position - use direct market order instead of set_holdings
            # FIXED: Use market_order to avoid buying power weight > 1.0 issues
            target_shares = self.shares_to_hedge
            self.log(f"*** INITIALIZING SPY POSITION ***: {target_shares} shares at ${self.last_price:.2f}")  # Fixed: Log -> log
            order_ticket = self.market_order(self.spy.symbol, target_shares)  # Fixed: Use market_order instead of set_holdings
            if order_ticket:
                self.log(f"*** SPY ORDER PLACED SUCCESSFULLY ***: {target_shares} shares")
            else:
                self.log("*** CRITICAL ERROR: FAILED TO PLACE SPY ORDER ***")
        else:
            spy_qty = self.portfolio[self.spy.symbol].quantity
            self.log(f"SPY position already exists: {spy_qty} shares")# Continue rebalancing even if no options data yet - SPY position is most important
        
        # Note: Available options are updated in on_data() method, no need to update here
        
        # Update price and volatility data
        if not self.update_market_data():
            self.log("Failed to update market data, skipping option trades")  # Fixed: Log -> log
            return
              # Check if we have any available options (only for options trading, not SPY)
        if not self.available_options:
            self.log("No options with valid pricing data available, skipping option trades only")  # Fixed: Log -> log
            return
        
        # Get current observation
        observation = self.get_observation()
        if observation is None:
            self.log("Could not create observation, skipping rebalance")  # Fixed: Log -> log
            return
            
        # Log observation details
        self.log(f"Observation vector length: {len(observation)}, first 6 values: {observation[:6]}")
        
        # Get RL model prediction
        actions = self.model_wrapper.predict(observation)
        if actions is None:
            self.log("Model prediction failed, using fallback action")  # Fixed: Log -> log
            # Simple fallback: no action
            actions = np.array([0.0, 0.0])

        self.log(f"Model predicted actions: call={actions[0]:.6f}, put={actions[1]:.6f}")  # Fixed: Log -> log
        
        # Add detailed logging for debugging flat exposure
        self.log(f"Raw model output: {actions}")
        self.log(f"Model output after clipping: {np.clip(actions, -1, 1)}")
        self.log(f"Max trade per step: {self.max_trade_per_step}")
        self.log(f"Action scaling before rounding: call={actions[0] * self.max_trade_per_step:.3f}, put={actions[1] * self.max_trade_per_step:.3f}")

        # Check if actions are meaningful
        if abs(actions[0]) < 0.001 and abs(actions[1]) < 0.001:
            self.log("WARNING: Model predicted nearly zero actions - may indicate prediction issue")
            
        # Execute trades based on RL actions
        self.execute_option_trades(actions)
        
        # Log positions after trades
        self.log(f"=== REBALANCE END === Call: {self.current_call_contracts}, Put: {self.current_put_contracts}")
        self.log(f"Option exposure - Total options value: ${self.get_total_options_value():,.0f}")
    
    def get_observation(self) -> np.ndarray:
        """Create observation vector EXACTLY matching training environment format"""
        self.prediction_count += 1
        
        if self.last_price is None or self.last_vol is None:
            self.log("Missing price or volatility data for observation")  # Fixed: Log -> log
            return None
            
        if len(self.price_history) < 2:
            self.log(f"Insufficient price history: {len(self.price_history)}")  # Fixed: Log -> log
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
                if hasattr(symbol.id, 'option_right') and str(symbol.id.option_right) == "OptionRight.Call" and C_t == 0.0:
                    C_t = contract.last_price if contract.last_price > 0 else contract.ask_price  # Fixed: LastPrice/AskPrice -> last_price/ask_price
                elif hasattr(symbol.id, 'option_right') and str(symbol.id.option_right) == "OptionRight.Put" and P_t == 0.0:
                    P_t = contract.last_price if contract.last_price > 0 else contract.ask_price  # Fixed: LastPrice/AskPrice -> last_price/ask_price
        
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
        # TRAINING MATCH: Episodes were 252 steps (trading days), we need to simulate this
        # Since we rebalance hourly, estimate our progress through a 252-step episode
        # Assume each "episode" represents roughly 252 trading hours = 252/6.5 ≈ 39 trading days
        
        days_since_start = (self.time.date() - self.get_start_date().date()).days
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
            self.log(f"=== OBSERVATION VALIDATION #{self.prediction_count} ===")
            expected_obs_labels = [
                "norm_S_t", "norm_C_t", "norm_P_t", "norm_call_held", "norm_put_held",
                "v_t", "norm_time_to_end", "call_delta", "call_gamma", 
                "put_delta", "put_gamma", "lagged_S_return", "lagged_v_change"
            ]
            for i, (label, value) in enumerate(zip(expected_obs_labels, obs)):
                self.log(f"{i:2d}: {label:15s} = {value:.6f}")
        
        # Check for any NaN or infinite values
        if np.any(~np.isfinite(obs)):
            self.log(f"WARNING: Observation contains NaN or infinite values: {obs}")
            
        self.log(f"Training-format observation: S={norm_S_t:.3f}, C={norm_C_t:.3f}, P={norm_P_t:.3f}, call_pos={norm_call_held:.3f}, put_pos={norm_put_held:.3f}")  # Fixed: Log -> log
        return obs
    
    def execute_option_trades(self, actions: np.ndarray):
        """Execute option trades based on RL model actions"""
        if len(actions) != 2:
            self.log(f"Invalid action length: {len(actions)}")  # Fixed: Log -> log
            return
              
        # Scale actions to actual trade sizes - FIXED ROUNDING
        call_trade = int(round(actions[0] * self.max_trade_per_step))  # Fixed: Use round() to avoid truncating small values
        put_trade = int(round(actions[1] * self.max_trade_per_step))   # Fixed: Use round() to avoid truncating small values
        
        self.log(f"Scaled trades: call={call_trade}, put={put_trade} (from actions {actions[0]:.6f}, {actions[1]:.6f})")  # Fixed: Log -> log
        
        # Find ATM options to trade
        call_symbol, put_symbol = self.find_atm_options_with_data()
        
        if call_symbol is None or put_symbol is None:
            self.log("No valid ATM options found with pricing data")  # Fixed: Log -> log
            return
        
        self.log(f"Found options: call={call_symbol}, put={put_symbol}")  # Fixed: Log -> log
        
        # Execute call trades
        if call_trade != 0:
            # FIXED: Use Price check instead of HasData
            if not self.securities.contains_key(call_symbol) or self.securities[call_symbol].price <= 0:
                self.log(f"Call option {call_symbol} does not have valid price, skipping trade")
                return
                
            new_call_position = max(-self.max_contracts_per_type, 
                                   min(self.max_contracts_per_type, 
                                       self.current_call_contracts + call_trade))
            trade_quantity = new_call_position - self.current_call_contracts
            
            if trade_quantity != 0:
                try:
                    # Check buying power before large trades
                    if abs(trade_quantity) > 10:  # For large trades
                        available_buying_power = self.portfolio.buying_power
                        estimated_cost = abs(trade_quantity) * self.securities[call_symbol].price * 100
                        if estimated_cost > available_buying_power * 0.5:  # Use max 50% of buying power
                            self.log(f"WARNING: Large call trade {trade_quantity} may exceed buying power. Cost: ${estimated_cost:,.0f}, Available: ${available_buying_power:,.0f}")
                            
                    self.log(f"Attempting call trade: {trade_quantity} contracts")  # Fixed: Log -> log
                    order_ticket = self.market_order(call_symbol, trade_quantity)  # Fixed: MarketOrder -> market_order
                    if order_ticket and order_ticket.order_id > 0:  # Fixed: OrderId -> order_id
                        self.current_call_contracts = new_call_position
                        # Log transaction cost
                        cost = abs(trade_quantity) * self.transaction_cost_per_contract
                        self.log(f"Call trade executed: {trade_quantity} contracts, cost: ${cost:.2f}")  # Fixed: Log -> log                    
                    else:
                        self.log(f"Failed to place call order for {trade_quantity} contracts")  # Fixed: Log -> log
                except Exception as e:
                    self.log(f"Error placing call order: {str(e)}")  # Fixed: Log -> log
        
        # Execute put trades  
        if put_trade != 0:
            # FIXED: Use Price check instead of HasData
            if not self.securities.contains_key(put_symbol) or self.securities[put_symbol].price <= 0:
                self.log(f"Put option {put_symbol} does not have valid price, skipping trade")
                return
                
            new_put_position = max(-self.max_contracts_per_type,
                                  min(self.max_contracts_per_type,
                                      self.current_put_contracts + put_trade))
            trade_quantity = new_put_position - self.current_put_contracts
            
            if trade_quantity != 0:
                try:
                    # Check buying power before large trades
                    if abs(trade_quantity) > 10:  # For large trades
                        available_buying_power = self.portfolio.buying_power
                        estimated_cost = abs(trade_quantity) * self.securities[put_symbol].price * 100
                        if estimated_cost > available_buying_power * 0.5:  # Use max 50% of buying power
                            self.log(f"WARNING: Large put trade {trade_quantity} may exceed buying power. Cost: ${estimated_cost:,.0f}, Available: ${available_buying_power:,.0f}")
                    
                    self.log(f"Attempting put trade: {trade_quantity} contracts")  # Fixed: Log -> log
                    order_ticket = self.market_order(put_symbol, trade_quantity)  # Fixed: MarketOrder -> market_order
                    if order_ticket and order_ticket.order_id > 0:  # Fixed: OrderId -> order_id
                        self.current_put_contracts = new_put_position                        
                        # Log transaction cost
                        cost = abs(trade_quantity) * self.transaction_cost_per_contract
                        self.log(f"Put trade executed: {trade_quantity} contracts, cost: ${cost:.2f}")  # Fixed: Log -> log
                    else:
                        self.log(f"Failed to place put order for {trade_quantity} contracts")  # Fixed: Log -> log
                except Exception as e:
                    self.log(f"Error placing put order: {str(e)}")  # Fixed: Log -> log
    
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
            if hasattr(symbol, 'security_type') and str(symbol.security_type) == "SecurityType.Option" and symbol.underlying == self.spy.symbol:
                strike_price = symbol.id.strike_price  # Fixed: ID.StrikePrice -> id.strike_price
                strike_diff = abs(strike_price - self.last_price)
                
                # Fixed: Separate tracking for call and put strikes
                if hasattr(symbol.id, 'option_right') and str(symbol.id.option_right) == "OptionRight.Call" and strike_diff < min_call_diff:
                    call_symbol = symbol
                    min_call_diff = strike_diff
                elif hasattr(symbol.id, 'option_right') and str(symbol.id.option_right) == "OptionRight.Put" and strike_diff < min_put_diff:
                    put_symbol = symbol
                    min_put_diff = strike_diff
        
        return call_symbol, put_symbol
    
    def on_order_event(self, order_event):  # Fixed: OnOrderEvent -> on_order_event, orderEvent -> order_event
        """Handle order events and update position tracking"""
        if hasattr(order_event, 'status') and str(order_event.status) == "OrderStatus.Filled":
            symbol = order_event.symbol  # Fixed: Symbol -> symbol
            if hasattr(symbol, 'security_type') and str(symbol.security_type) == "SecurityType.Option":
                self.log(f"Option order filled: {symbol} quantity: {order_event.fill_quantity}")  # Fixed: Log -> log, FillQuantity -> fill_quantity
