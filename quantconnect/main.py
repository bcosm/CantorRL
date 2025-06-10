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
        
        # Environment state tracking - CONSERVATIVE SIZING FOR TESTING
        self.shares_to_hedge = 1000   # Reduced from 10000 to avoid margin issues
        self.current_call_contracts = 0
        self.current_put_contracts = 0
        self.max_contracts_per_type = 20   # Reduced from 200
        self.max_trade_per_step = 3        # Reduced from 15
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
        """Create observation vector matching training environment"""
        if self.last_price is None or self.last_vol is None:
            self.Log("Missing price or volatility data for observation")
            return None
            
        if len(self.price_history) < 2 or len(self.vol_history) < 2:
            self.Log(f"Insufficient history: prices={len(self.price_history)}, vol={len(self.vol_history)}")
            return None
        
        # Calculate features similar to training environment
        current_price = self.last_price
        current_vol = self.last_vol
        
        # Price momentum features
        price_return_1 = (self.price_history[-1] / self.price_history[-2] - 1) if len(self.price_history) >= 2 else 0
        price_return_5 = (self.price_history[-1] / self.price_history[-5] - 1) if len(self.price_history) >= 5 else 0
        
        # Volatility features  
        vol_change = (self.vol_history[-1] / self.vol_history[-2] - 1) if len(self.vol_history) >= 2 else 0
        
        # Portfolio features
        call_position_normalized = self.current_call_contracts / self.max_contracts_per_type
        put_position_normalized = self.current_put_contracts / self.max_contracts_per_type
        
        # Option pricing features (simplified)
        time_to_expiry = 30/252  # Assume 30 days
        moneyness_call = current_price / current_price  # ATM = 1.0
        moneyness_put = current_price / current_price   # ATM = 1.0
        
        # Calculate Greeks for calls and puts
        call_greeks = self.option_calculator.calculate_greeks(
            current_price, current_price, time_to_expiry, 0.04, current_vol, 'call'
        )
        put_greeks = self.option_calculator.calculate_greeks(
            current_price, current_price, time_to_expiry, 0.04, current_vol, 'put'
        )
        
        call_delta = call_greeks['delta']
        put_delta = put_greeks['delta'] 
        gamma = call_greeks['gamma']  # Same for calls and puts
        
        # Portfolio Greeks
        portfolio_delta = (call_delta * self.current_call_contracts + 
                          put_delta * self.current_put_contracts) * self.option_multiplier
        portfolio_gamma = gamma * (self.current_call_contracts + self.current_put_contracts) * self.option_multiplier
        
        # Create observation vector matching training format
        observation = np.array([
            current_price / 400.0,  # Normalized price
            price_return_1,
            price_return_5, 
            vol_change,
            call_position_normalized,
            put_position_normalized,
            moneyness_call,
            moneyness_put,
            current_vol,
            call_delta,
            put_delta,
            portfolio_delta / 10000.0,  # Normalized
            portfolio_gamma / 10000.0   # Normalized
        ], dtype=np.float32)
        
        self.Log(f"Created observation: price={current_price:.2f}, vol={current_vol:.3f}, call_pos={call_position_normalized:.3f}, put_pos={put_position_normalized:.3f}")
        
        return observation    def ExecuteOptionTrades(self, actions: np.ndarray):
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
