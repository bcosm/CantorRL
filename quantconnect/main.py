# region imports
from AlgorithmImports import *
import numpy as np
import torch
import joblib
from typing import List, Tuple, Dict, Any
import os
import sys

# Import our custom classes
from model_wrapper import ModelWrapper
from option_calculator import OptionCalculator
# endregion

class RLHedgingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        # Set basic algorithm parameters
        self.SetStartDate(2023, 1, 1)  # Adjust start date as needed
        self.SetEndDate(2024, 1, 1)    # Adjust end date as needed
        self.SetCash(1000000)          # Starting cash
        
        # Add SPY as our underlying asset
        self.spy = self.AddEquity("SPY", Resolution.Minute)
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
        # Add options universe for SPY
        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(self.OptionFilter)
        
        # Set warmup period to ensure all securities have data before trading
        self.SetWarmup(timedelta(days=5))  # 5 days of warmup data
        
        # Initialize our RL model
        self.model_wrapper = ModelWrapper(self)
        self.option_calculator = OptionCalculator()
          # Environment state tracking - INCREASED SIZING FOR ACTUAL TRADING
        self.shares_to_hedge = 5000   # Increased from 1000 for more realistic hedging
        self.current_call_contracts = 0
        self.current_put_contracts = 0
        self.max_contracts_per_type = 50   # Increased from 20 for more trading capacity
        self.max_trade_per_step = 10       # Increased from 3 for more aggressive trading
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
        
        # Schedule rebalancing every hour during market hours
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.Every(TimeSpan.FromMinutes(60)),
            self.Rebalance
        )
        
        self.Log("RL Hedging Algorithm Initialized")
    
    def OptionFilter(self, universe):
        """Filter options to only ATM calls and puts with ~30 day expiry"""
        return (universe
                .Strikes(-2, 2)  # Near the money strikes
                .Expiration(timedelta(20), timedelta(45))  # 20-45 days to expiry
                .OnlyApplyFilterAtMarketOpen())
    
    def OnData(self, data):
        """Main data handler - updates price and volatility tracking"""
        # Skip processing during warmup period
        if self.IsWarmingUp:
            return
            
        if not self.warmup_complete:
            self.warmup_complete = True
            self.Log("Warmup period complete - starting trading")
        
        if not self.spy.Symbol in data.Bars:
            return
            
        # Update current price
        current_price = data.Bars[self.spy.Symbol].Close
        if current_price > 0:  # Ensure valid price
            if self.last_price is not None:
                self.price_history.append(current_price)
                
            self.last_price = current_price
        
        # Update available options with valid data
        self.UpdateAvailableOptions(data)
        
        # Calculate implied volatility from options if available
        if data.OptionChains.Count > 0:
            for chain in data.OptionChains.Values:
                self.UpdateImpliedVolatility(chain, current_price)
        
        # Keep history manageable
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-50:]
        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-50:]
    def UpdateAvailableOptions(self, data):
        """Track which options have valid pricing data"""
        self.available_options.clear()
        
        if data.OptionChains.Count > 0:
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
                            
                            # Only include options within 5% of current price and 7-60 days to expiry
                            if strike_diff_pct <= 0.05 and 7 <= time_to_expiry <= 60:
                                self.available_options[contract.Symbol] = contract
    
    def UpdateImpliedVolatility(self, chain, current_price):
        """Extract implied volatility from option chain"""
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

    def Rebalance(self):
        """Main rebalancing logic using RL model predictions"""
        # Don't trade during warmup
        if self.IsWarmingUp:
            return
            
        # Reset LSTM hidden states at start of new trading day
        if self.Time.hour == 9 and self.Time.minute == 30:  # Market open
            self.model_wrapper.reset_hidden_states()
            
        # Check if we have valid price data
        if self.last_price is None:
            self.Log("No price data available yet, skipping rebalance")
            return
            
        # Check if we have any available options
        if not self.available_options:
            self.Log("No options with valid pricing data available, skipping rebalance")
            return
            
        if not self.Portfolio[self.spy.Symbol].Invested:
            # Initialize hedge position - calculate target dollar amount
            target_dollar_amount = self.shares_to_hedge * self.last_price
            target_weight = target_dollar_amount / self.Portfolio.TotalPortfolioValue
            self.SetHoldings(self.spy.Symbol, target_weight)
            self.Log(f"Initialized SPY position: {self.shares_to_hedge} shares at ${self.last_price:.2f}")
        
        # Get current observation
        observation = self.GetObservation()
        if observation is None:
            self.Log("Could not create observation, skipping rebalance")
            return
            
        # Get RL model prediction
        actions = self.model_wrapper.predict(observation)
        if actions is None:
            self.Log("Model prediction failed, using fallback action")
            # Simple fallback: no action
            actions = np.array([0.0, 0.0])
            self.Log(f"Model predicted actions: call={actions[0]:.3f}, put={actions[1]:.3f}")
            
        # Execute trades based on RL actions
        self.ExecuteOptionTrades(actions)
    
    def GetObservation(self) -> np.ndarray:
        """Create observation vector EXACTLY matching training environment format"""
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
                if symbol.ID.OptionRight == OptionRight.Call and C_t == 0.0:
                    C_t = contract.LastPrice if contract.LastPrice > 0 else contract.AskPrice
                elif symbol.ID.OptionRight == OptionRight.Put and P_t == 0.0:
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
        # Assume daily rebalancing over ~30 day "episode"
        trading_hours_elapsed = (self.Time.hour - 9) + (self.Time.minute / 60.0)  # Hours since 9 AM
        days_elapsed = trading_hours_elapsed / 6.5  # 6.5 hour trading day
        norm_time_to_end = max(0.0, 1.0 - (days_elapsed / 30.0))  # 30-day episode approximation
        
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
        
        self.Log(f"Training-format observation: S={norm_S_t:.3f}, C={norm_C_t:.3f}, P={norm_P_t:.3f}, call_pos={norm_call_held:.3f}, put_pos={norm_put_held:.3f}")
        
        return obs
    
    def ExecuteOptionTrades(self, actions: np.ndarray):
        """Execute option trades based on RL model actions"""
        if len(actions) != 2:
            self.Log(f"Invalid action length: {len(actions)}")
            return
            
        # Scale actions to actual trade sizes
        call_trade = int(actions[0] * self.max_trade_per_step)
        put_trade = int(actions[1] * self.max_trade_per_step)
        
        self.Log(f"Scaled trades: call={call_trade}, put={put_trade} (from actions {actions[0]:.3f}, {actions[1]:.3f})")
        
        # Find ATM options to trade
        call_symbol, put_symbol = self.FindATMOptionsWithData()
        
        if call_symbol is None or put_symbol is None:
            self.Log("No valid ATM options found with pricing data")
            return
        
        self.Log(f"Found options: call={call_symbol}, put={put_symbol}")
        
        # Execute call trades
        if call_trade != 0:
            # Validate that we have current price data for the call option
            if not self.Securities.ContainsKey(call_symbol) or not self.Securities[call_symbol].HasData:
                self.Log(f"Call option {call_symbol} does not have valid data, skipping trade")
                return
                
            new_call_position = max(-self.max_contracts_per_type, 
                                   min(self.max_contracts_per_type, 
                                       self.current_call_contracts + call_trade))
            trade_quantity = new_call_position - self.current_call_contracts
            
            if trade_quantity != 0:
                try:
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
            # Validate that we have current price data for the put option
            if not self.Securities.ContainsKey(put_symbol) or not self.Securities[put_symbol].HasData:
                self.Log(f"Put option {put_symbol} does not have valid data, skipping trade")
                return
                
            new_put_position = max(-self.max_contracts_per_type,
                                  min(self.max_contracts_per_type,
                                      self.current_put_contracts + put_trade))
            trade_quantity = new_put_position - self.current_put_contracts
            
            if trade_quantity != 0:
                try:
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
    def FindATMOptionsWithData(self):
        """Find ATM call and put options that have valid pricing data"""
        if self.last_price is None or not self.available_options:
            return None, None
            
        call_symbol = None
        put_symbol = None
        min_strike_diff = float('inf')
        
        # Look through available options with valid data
        for symbol, contract in self.available_options.items():
            if symbol.SecurityType == SecurityType.Option and symbol.Underlying == self.spy.Symbol:
                strike_price = symbol.ID.StrikePrice
                strike_diff = abs(strike_price - self.last_price)
                
                if strike_diff < min_strike_diff:
                    min_strike_diff = strike_diff
                    if symbol.ID.OptionRight == OptionRight.Call:
                        call_symbol = symbol
                    else:
                        put_symbol = symbol
        
        return call_symbol, put_symbol
    
    def OnOrderEvent(self, orderEvent):
        """Handle order events and update position tracking"""
        if orderEvent.Status == OrderStatus.Filled:
            symbol = orderEvent.Symbol
            if symbol.SecurityType == SecurityType.Option:
                self.Log(f"Option order filled: {symbol} quantity: {orderEvent.FillQuantity}")
