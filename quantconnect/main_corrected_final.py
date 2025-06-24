from AlgorithmImports import *
from datetime import timedelta
import numpy as np
import torch
from typing import List, Dict, Any
import os
import sys

from model_wrapper import ModelWrapper
from option_calculator import OptionCalculator

class RLHedgingAlgorithm(QCAlgorithm):
    """
    RL Hedging Algorithm - CORRECTED VERSION
    
    This implementation matches the training environment EXACTLY:
    - 252 trading days per episode (1 year)
    - Daily trading frequency (one trade per day) 
    - Exact observation vector construction
    - Proper ATM option handling
    - Correct state tracking
    """
    
    def initialize(self):
        # Use 2024 data with sufficient length for a full episode (252 trading days)
        self.SetStartDate(2024, 1, 2)
        self.SetEndDate(2024, 12, 31)  # Full year for complete episode
        self.SetCash(100000000)  # $100M to avoid capital constraints
        
        # Daily resolution to match training environment (DT = 1/252 years)
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
        # Daily option resolution to match training
        option = self.AddOption("SPY", Resolution.Daily)
        option.SetFilter(self.option_filter)
        
        # 5-day warmup to ensure data availability
        self.SetWarmUp(timedelta(days=5))
        
        # Initialize model and option calculator
        self.model_wrapper = ModelWrapper(self)
        self.option_calculator = OptionCalculator()
        
        # Training environment parameters - EXACT MATCH
        self.shares_to_hedge = 10000  # Fixed SPY position
        self.max_contracts_per_type = 200  # Max contracts per option type
        self.max_trade_per_step = 15  # Max trade per step (daily)
        self.option_multiplier = 100  # Standard option contract multiplier
        self.transaction_cost_per_contract = 0.05  # Transaction cost
        self.risk_free_rate = 0.04  # Risk-free rate
        self.option_tenor_years = 30 / 252  # 30-day tenor in years
        
        # Episode tracking - CRITICAL FOR TRAINING ENVIRONMENT MATCH
        self.episode_length = 252  # One trading year
        self.current_step = 0  # Current step in episode
        
        # State tracking variables
        self.current_call_contracts = 0  # Current call position
        self.current_put_contracts = 0   # Current put position
        self.initial_S0_for_episode = None  # Initial stock price for episode
        self.S_t_minus_1 = None  # Previous step stock price
        self.v_t_minus_1 = None  # Previous step volatility
        
        # Market data storage
        self.last_price = None
        self.last_vol = None
        self.price_history = []
        self.vol_history = []
        self.available_options = {}
        
        # Control flags
        self.warmup_complete = False
        self.spy_position_initialized = False
        self.daily_trade_executed = False
        
        # CRITICAL: Only schedule DAILY rebalancing to match training environment
        # Training environment: 1 action per day, daily frequency
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 0),  # Market open
            self.daily_rebalance
        )
        
        self.Log("*** RL Hedging Algorithm Initialized - TRAINING ENVIRONMENT MATCH ***")
        self.Log(f"Episode length: {self.episode_length} days")
        self.Log(f"Shares to hedge: {self.shares_to_hedge}")
        self.Log(f"Max trade per step: {self.max_trade_per_step}")
        
    def option_filter(self, universe):
        """Filter options to find ATM contracts with 30-day tenor"""
        return (universe
                .Strikes(-5, 5)  # Near ATM strikes
                .Expiration(timedelta(20), timedelta(40))  # ~30-day tenor
                )
    
    def OnData(self, data):
        """Process market data - daily frequency only"""
        if self.IsWarmingUp:
            return
            
        if not self.warmup_complete:
            self.warmup_complete = True
            self.Log("*** WARMUP COMPLETE - STARTING DAILY TRADING ***")
        
        # Update SPY price data
        if self.spy.Symbol in data.Bars:
            current_price = data.Bars[self.spy.Symbol].Close
            if current_price > 0:
                self.price_history.append(current_price)
                self.last_price = current_price
                
                # Set initial S0 for episode normalization
                if self.initial_S0_for_episode is None:
                    self.initial_S0_for_episode = current_price
                    self.Log(f"*** Set initial S0 for episode: ${self.initial_S0_for_episode:.2f} ***")
        
        # Update option chain data
        if hasattr(data, 'OptionChains') and data.OptionChains.Count > 0:
            spy_canonical = self.spy.Symbol
            if spy_canonical in data.OptionChains:
                self.update_available_options(data.OptionChains[spy_canonical])
        
        # Keep history manageable
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-50:]
        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-50:]
    
    def update_available_options(self, chain):
        """Update available options from chain data"""
        self.available_options.clear()
        
        if self.last_price is None:
            return
            
        for contract in chain:
            # Check if option has valid pricing
            has_price = (contract.LastPrice > 0 or 
                        (contract.BidPrice > 0 and contract.AskPrice > 0 and 
                         contract.AskPrice > contract.BidPrice))
            
            if has_price:
                # Filter for ATM options with appropriate tenor
                strike_diff_pct = abs(contract.Strike - self.last_price) / self.last_price
                days_to_expiry = (contract.Expiry.date() - self.Time.date()).days
                
                # Allow some flexibility but prefer ATM options with ~30-day tenor
                if strike_diff_pct <= 0.15 and 20 <= days_to_expiry <= 40:
                    self.available_options[contract.Symbol] = contract
        
        # Update implied volatility from ATM options
        self.update_implied_volatility()
    
    def update_implied_volatility(self):
        """Calculate average implied volatility from ATM options"""
        if not self.available_options or self.last_price is None:
            return
            
        atm_ivs = []
        for symbol, contract in self.available_options.items():
            if (abs(contract.Strike - self.last_price) < self.last_price * 0.05 and 
                contract.ImpliedVolatility > 0):
                atm_ivs.append(contract.ImpliedVolatility)
        
        if atm_ivs:
            self.last_vol = np.mean(atm_ivs)
            self.vol_history.append(self.last_vol)
    
    def daily_rebalance(self):
        """
        Main daily rebalancing function - matches training environment exactly
        This is called once per day to match the training frequency
        """
        if self.IsWarmingUp or self.daily_trade_executed:
            return
            
        self.Log(f"=== DAILY REBALANCE - Day {self.current_step + 1}/{self.episode_length} ===")
        
        # Reset LSTM hidden states daily (like training environment)
        self.model_wrapper.reset_hidden_states()
        
        # Initialize SPY position if not done
        if not self.spy_position_initialized:
            if not self.initialize_spy_position():
                return
        
        # Ensure we have required market data
        if not self.validate_market_data():
            self.Log("Insufficient market data for trading")
            self.advance_step()
            return
        
        # Get observation vector (exact match to training environment)
        observation = self.get_observation()
        if observation is None:
            self.Log("Failed to construct observation vector")
            self.advance_step() 
            return
        
        # Get model prediction
        actions = self.model_wrapper.predict(observation)
        if actions is None:
            self.Log("Model prediction failed, using no-trade action")
            actions = np.array([0.0, 0.0])
        
        self.Log(f"Model actions: call={actions[0]:.4f}, put={actions[1]:.4f}")
        
        # Execute option trades (exact match to training environment)
        self.execute_option_trades(actions)
        
        # Update state for next step (like training environment)
        self.advance_step()
        
        # Mark daily trade as executed
        self.daily_trade_executed = True
        
        # Reset for next day
        self.Schedule.On(
            self.DateRules.Tomorrow,
            self.TimeRules.At(0, 0),  # Midnight
            self.reset_daily_flag
        )
    
    def reset_daily_flag(self):
        """Reset the daily trading flag"""
        self.daily_trade_executed = False
    
    def initialize_spy_position(self):
        """Initialize the fixed SPY position"""
        if self.last_price is None:
            if self.Securities[self.spy.Symbol].Price > 0:
                self.last_price = self.Securities[self.spy.Symbol].Price
            else:
                return False
        
        # Check if SPY position already exists
        current_spy_qty = self.Portfolio[self.spy.Symbol].Quantity
        target_spy_qty = self.shares_to_hedge
        
        if abs(current_spy_qty - target_spy_qty) > 1:  # Allow for rounding
            trade_qty = target_spy_qty - current_spy_qty
            self.Log(f"*** Initializing SPY position: {trade_qty} shares ***")
            order_ticket = self.MarketOrder(self.spy.Symbol, trade_qty)
            if order_ticket:
                self.spy_position_initialized = True
                return True
        else:
            self.spy_position_initialized = True
            return True
        
        return False
    
    def validate_market_data(self):
        """Validate we have sufficient market data"""
        if self.last_price is None:
            return False
        if self.last_vol is None:
            self.last_vol = 0.20  # Fallback volatility
        return True
    
    def get_observation(self):
        """
        Construct observation vector EXACTLY matching training environment
        
        Returns 13-dimensional observation vector:
        [norm_S_t, norm_C_t, norm_P_t, norm_call_held, norm_put_held,
         v_t, norm_time_to_end, call_delta, call_gamma, 
         put_delta, put_gamma, lagged_S_return, lagged_v_change]
        """
        if self.last_price is None or self.initial_S0_for_episode is None:
            return None
        
        # Current market data
        S_t = self.last_price
        v_t = self.last_vol if self.last_vol is not None else 0.20
        
        # Get ATM option prices (or calculate using BS)
        C_t, P_t = self.get_atm_option_prices(S_t, v_t)
        
        # Normalization factor (training environment uses max(S0, 25.0))
        s0_safe = max(self.initial_S0_for_episode, 25.0)
        
        # Normalized prices
        norm_S_t = S_t / s0_safe
        norm_C_t = C_t / s0_safe
        norm_P_t = P_t / s0_safe
        
        # Normalized positions
        norm_call_held = self.current_call_contracts / self.max_contracts_per_type
        norm_put_held = self.current_put_contracts / self.max_contracts_per_type
        
        # Time to end of episode
        norm_time_to_end = max(0.0, (self.episode_length - self.current_step) / self.episode_length)
        
        # Calculate Greeks for ATM options
        K_atm = round(S_t)  # ATM strike
        call_greeks = self.option_calculator.calculate_greeks(
            S_t, K_atm, self.option_tenor_years, self.risk_free_rate, v_t, 'call'
        )
        put_greeks = self.option_calculator.calculate_greeks(
            S_t, K_atm, self.option_tenor_years, self.risk_free_rate, v_t, 'put'
        )
        
        call_delta = call_greeks.get('delta', 0.5)
        call_gamma = call_greeks.get('gamma', 0.0)
        put_delta = put_greeks.get('delta', -0.5)
        put_gamma = call_gamma  # Same gamma for call and put
        
        # Lagged returns (training environment logic)
        if self.current_step == 0 or self.S_t_minus_1 is None or self.S_t_minus_1 == 0:
            lagged_S_return = 0.0
        else:
            lagged_S_return = (S_t - self.S_t_minus_1) / self.S_t_minus_1
            lagged_S_return = np.clip(lagged_S_return, -1.0, 1.0)  # Clip to [-1, 1]
        
        # Lagged volatility change
        if self.current_step == 0 or self.v_t_minus_1 is None:
            lagged_v_change = 0.0
        else:
            lagged_v_change = v_t - self.v_t_minus_1
            lagged_v_change = np.clip(lagged_v_change, -1.0, 1.0)  # Clip to [-1, 1]
        
        # Construct observation vector (EXACT training environment format)
        observation = np.array([
            norm_S_t,          # 0: Normalized stock price
            norm_C_t,          # 1: Normalized call price
            norm_P_t,          # 2: Normalized put price
            norm_call_held,    # 3: Normalized call position
            norm_put_held,     # 4: Normalized put position
            v_t,               # 5: Current volatility (raw)
            norm_time_to_end,  # 6: Normalized time to episode end
            call_delta,        # 7: Call delta
            call_gamma,        # 8: Call gamma
            put_delta,         # 9: Put delta
            put_gamma,         # 10: Put gamma
            lagged_S_return,   # 11: Lagged stock return
            lagged_v_change    # 12: Lagged volatility change
        ], dtype=np.float32)
        
        # Validate observation
        if np.any(~np.isfinite(observation)):
            self.Log(f"WARNING: Invalid observation values: {observation}")
            return None
        
        self.Log(f"Obs: S={norm_S_t:.3f}, C={norm_C_t:.3f}, P={norm_P_t:.3f}, " +
                f"call_pos={norm_call_held:.3f}, put_pos={norm_put_held:.3f}, vol={v_t:.3f}")
        
        return observation
    
    def get_atm_option_prices(self, S_t, v_t):
        """Get ATM call and put option prices"""
        C_t = 0.0
        P_t = 0.0
        
        # Try to get prices from available options
        if self.available_options:
            atm_strike = round(S_t)
            best_call_price = None
            best_put_price = None
            min_strike_diff = float('inf')
            
            for symbol, contract in self.available_options.items():
                strike_diff = abs(contract.Strike - atm_strike)
                if strike_diff < min_strike_diff:
                    if hasattr(symbol.ID, 'OptionRight'):
                        if str(symbol.ID.OptionRight) == "OptionRight.Call":
                            price = contract.LastPrice if contract.LastPrice > 0 else contract.AskPrice
                            if price > 0:
                                best_call_price = price
                        elif str(symbol.ID.OptionRight) == "OptionRight.Put":
                            price = contract.LastPrice if contract.LastPrice > 0 else contract.AskPrice
                            if price > 0:
                                best_put_price = price
                    min_strike_diff = strike_diff
            
            if best_call_price is not None:
                C_t = best_call_price
            if best_put_price is not None:
                P_t = best_put_price
        
        # Fallback to Black-Scholes calculation if no market prices
        if C_t <= 0 or P_t <= 0:
            K_atm = round(S_t)
            call_greeks = self.option_calculator.calculate_greeks(
                S_t, K_atm, self.option_tenor_years, self.risk_free_rate, v_t, 'call'
            )
            put_greeks = self.option_calculator.calculate_greeks(
                S_t, K_atm, self.option_tenor_years, self.risk_free_rate, v_t, 'put'
            )
            
            if C_t <= 0:
                C_t = call_greeks.get('price', S_t * 0.05)
            if P_t <= 0:
                P_t = put_greeks.get('price', S_t * 0.05)
        
        return C_t, P_t
    
    def execute_option_trades(self, actions):
        """
        Execute option trades based on model actions
        EXACTLY matches training environment scaling and logic
        """
        if len(actions) != 2:
            self.Log(f"Invalid action vector length: {len(actions)}")
            return
        
        # Scale actions to trade quantities (training environment logic)
        raw_call_action = np.clip(actions[0], -1.0, 1.0)
        raw_put_action = np.clip(actions[1], -1.0, 1.0)
        
        # Scale by max_trade_per_step (training environment scaling)
        call_trade_float = raw_call_action * self.max_trade_per_step
        put_trade_float = raw_put_action * self.max_trade_per_step
        
        # Round to integer contracts (training environment logic)
        call_trade = int(round(call_trade_float))
        put_trade = int(round(put_trade_float))
        
        self.Log(f"Trade calc: call_action={raw_call_action:.4f} -> {call_trade} contracts")
        self.Log(f"Trade calc: put_action={raw_put_action:.4f} -> {put_trade} contracts")
        
        # Calculate new positions with clipping (training environment logic)
        new_call_position = np.clip(
            self.current_call_contracts + call_trade,
            -self.max_contracts_per_type,
            self.max_contracts_per_type
        )
        new_put_position = np.clip(
            self.current_put_contracts + put_trade,
            -self.max_contracts_per_type,
            self.max_contracts_per_type
        )
        
        # Calculate actual trades needed
        actual_call_trade = new_call_position - self.current_call_contracts
        actual_put_trade = new_put_position - self.current_put_contracts
        
        # Find ATM option symbols
        call_symbol, put_symbol = self.find_atm_option_symbols()
        
        # Execute call trade
        if actual_call_trade != 0 and call_symbol is not None:
            self.execute_single_option_trade(call_symbol, actual_call_trade, "CALL")
            self.current_call_contracts = new_call_position
        
        # Execute put trade  
        if actual_put_trade != 0 and put_symbol is not None:
            self.execute_single_option_trade(put_symbol, actual_put_trade, "PUT")
            self.current_put_contracts = new_put_position
    
    def find_atm_option_symbols(self):
        """Find ATM call and put option symbols"""
        if not self.available_options or self.last_price is None:
            return None, None
        
        atm_strike = round(self.last_price)
        call_symbol = None
        put_symbol = None
        min_strike_diff = float('inf')
        
        for symbol, contract in self.available_options.items():
            strike_diff = abs(contract.Strike - atm_strike)
            if strike_diff <= min_strike_diff:
                if hasattr(symbol.ID, 'OptionRight'):
                    if str(symbol.ID.OptionRight) == "OptionRight.Call":
                        call_symbol = symbol
                    elif str(symbol.ID.OptionRight) == "OptionRight.Put":
                        put_symbol = symbol
                    min_strike_diff = strike_diff
        
        return call_symbol, put_symbol
    
    def execute_single_option_trade(self, symbol, quantity, option_type):
        """Execute a single option trade"""
        try:
            if not self.Securities.ContainsKey(symbol):
                self.Log(f"Option {symbol} not in securities")
                return False
            
            if self.Securities[symbol].Price <= 0:
                self.Log(f"Option {symbol} has invalid price")
                return False
            
            self.Log(f"Executing {option_type} trade: {quantity} contracts of {symbol}")
            order_ticket = self.MarketOrder(symbol, quantity)
            
            if order_ticket and order_ticket.OrderId > 0:
                cost = abs(quantity) * self.transaction_cost_per_contract
                self.Log(f"{option_type} trade executed: {quantity} contracts, cost: ${cost:.2f}")
                return True
            else:
                self.Log(f"Failed to place {option_type} order")
                return False
                
        except Exception as e:
            self.Log(f"Error executing {option_type} trade: {str(e)}")
            return False
    
    def advance_step(self):
        """Advance to next step in episode (training environment logic)"""
        # Store current values for lagged calculations
        self.S_t_minus_1 = self.last_price
        self.v_t_minus_1 = self.last_vol
        
        # Increment step
        self.current_step += 1
        
        self.Log(f"Advanced to step {self.current_step}/{self.episode_length}")
        
        # Check if episode is complete
        if self.current_step >= self.episode_length:
            self.Log("*** EPISODE COMPLETE - RESETTING FOR NEW EPISODE ***")
            self.reset_episode()
    
    def reset_episode(self):
        """Reset for new episode (training environment logic)"""
        self.current_step = 0
        self.initial_S0_for_episode = self.last_price
        self.S_t_minus_1 = None
        self.v_t_minus_1 = None
        
        # Note: In training, positions are also reset, but in live trading
        # we might want to maintain positions across episodes
        self.Log("Episode reset complete")
    
    def OnOrderEvent(self, orderEvent):
        """Handle order events"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"Order filled: {orderEvent.Symbol} - {orderEvent.FillQuantity} @ ${orderEvent.FillPrice}")
