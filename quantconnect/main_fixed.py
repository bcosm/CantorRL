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
        if not self.spy.Symbol in data.Bars:
            return
            
        # Update current price
        current_price = data.Bars[self.spy.Symbol].Close
        if current_price > 0:  # Ensure valid price
            if self.last_price is not None:
                self.price_history.append(current_price)
                
            self.last_price = current_price
        
        # Calculate implied volatility from options if available
        if data.OptionChains.Count > 0:
            for chain in data.OptionChains.Values:
                self.UpdateImpliedVolatility(chain, current_price)
        
        # Keep history manageable
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-50:]
        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-50:]
    
    def UpdateImpliedVolatility(self, chain, current_price):
        """Extract implied volatility from option chain"""
        atm_options = []
        
        for contract in chain:
            if abs(contract.Strike - current_price) < current_price * 0.02:  # Within 2%
                atm_options.append(contract)
        
        if atm_options:
            # Use average implied volatility of ATM options
            avg_iv = np.mean([opt.ImpliedVolatility for opt in atm_options if opt.ImpliedVolatility > 0])
            if avg_iv > 0:
                self.vol_history.append(avg_iv)
                self.last_vol = avg_iv
    
    def Rebalance(self):
        """Main rebalancing logic using RL model predictions"""
        # Check if we have valid price data
        if self.last_price is None:
            self.Log("No price data available yet, skipping rebalance")
            return
            
        if not self.Portfolio[self.spy.Symbol].Invested:
            # Initialize hedge position - calculate target dollar amount with conservative sizing
            target_dollar_amount = self.shares_to_hedge * self.last_price
            # Use much smaller weight to avoid margin issues
            max_position_value = self.Portfolio.TotalPortfolioValue * 0.3  # Max 30% of portfolio
            target_dollar_amount = min(target_dollar_amount, max_position_value)
            target_weight = target_dollar_amount / self.Portfolio.TotalPortfolioValue
            
            self.SetHoldings(self.spy.Symbol, target_weight)
            self.Log(f"Initialized SPY position: target ${target_dollar_amount:.0f} at ${self.last_price:.2f}")
        
        # Get current observation
        observation = self.GetObservation()
        if observation is None:
            self.Log("Could not create observation, skipping rebalance")
            return
            
        # Get RL model prediction
        actions = self.model_wrapper.predict(observation)
        if actions is None:
            return
            
        # Execute trades based on RL actions
        self.ExecuteOptionTrades(actions)
    
    def GetObservation(self) -> np.ndarray:
        """Create observation vector matching training environment"""
        if self.last_price is None or self.last_vol is None:
            return None
            
        if len(self.price_history) < 2 or len(self.vol_history) < 2:
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
        
        return observation
    
    def ExecuteOptionTrades(self, actions: np.ndarray):
        """Execute option trades based on RL model actions"""
        if len(actions) != 2:
            return
            
        # Scale actions to actual trade sizes
        call_trade = int(actions[0] * self.max_trade_per_step)
        put_trade = int(actions[1] * self.max_trade_per_step)
        
        # Find ATM options to trade
        call_symbol, put_symbol = self.FindATMOptions()
        
        if call_symbol is None or put_symbol is None:
            return
        
        # Execute call trades
        if call_trade != 0:
            new_call_position = max(-self.max_contracts_per_type, 
                                   min(self.max_contracts_per_type, 
                                       self.current_call_contracts + call_trade))
            trade_quantity = new_call_position - self.current_call_contracts
            
            if trade_quantity != 0:
                self.MarketOrder(call_symbol, trade_quantity)
                self.current_call_contracts = new_call_position
                
                # Log transaction cost
                cost = abs(trade_quantity) * self.transaction_cost_per_contract
                self.Log(f"Call trade: {trade_quantity} contracts, cost: ${cost:.2f}")
        
        # Execute put trades  
        if put_trade != 0:
            new_put_position = max(-self.max_contracts_per_type,
                                  min(self.max_contracts_per_type,
                                      self.current_put_contracts + put_trade))
            trade_quantity = new_put_position - self.current_put_positions
            
            if trade_quantity != 0:
                self.MarketOrder(put_symbol, trade_quantity)
                self.current_put_contracts = new_put_position
                
                # Log transaction cost
                cost = abs(trade_quantity) * self.transaction_cost_per_contract
                self.Log(f"Put trade: {trade_quantity} contracts, cost: ${cost:.2f}")
    
    def FindATMOptions(self) -> Tuple[Symbol, Symbol]:
        """Find the most liquid ATM call and put options"""
        if self.last_price is None:
            return None, None
            
        call_symbol = None
        put_symbol = None
        min_strike_diff = float('inf')
        
        # Look through current option chain
        for symbol in self.Securities.Keys:
            if symbol.SecurityType == SecurityType.Option and symbol.Underlying == self.spy.Symbol:
                # Access strike price through symbol.ID.StrikePrice, not security.Strike
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
