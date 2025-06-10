# region imports
from AlgorithmImports import *
import numpy as np
import torch
from typing import Optional
# endregion

class RLHedgingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        # Set basic algorithm parameters - CONSERVATIVE SETTINGS
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 1, 8)    # 1 week test
        self.SetCash(1000000)          # $1M starting cash
        
        # Add SPY as our underlying asset
        self.spy = self.AddEquity("SPY")
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
        # Add options universe for SPY
        option = self.AddOption("SPY")
        option.SetFilter(self.OptionFilter)
        
        # Conservative position sizing - MUCH SMALLER
        self.target_spy_allocation = 0.6  # Use 60% of portfolio for SPY hedge
        self.current_call_contracts = 0
        self.current_put_contracts = 0
        self.max_contracts_per_type = 20   # REDUCED from 200
        self.max_trade_per_step = 3        # REDUCED from 15
        self.option_multiplier = 100
        self.transaction_cost_per_contract = 0.05
        
        # State tracking
        self.last_price = None
        self.last_vol = 0.2  # Default volatility
        self.price_history = []
        self.vol_history = []
        self.current_call_symbol = None
        self.current_put_symbol = None
        
        # Simple RL model simulation (for now)
        self.model_loaded = False
        
        # Schedule rebalancing less frequently for testing
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 60),  # 1 hour after open
            self.Rebalance
        )
        
        self.Log("Conservative RL Hedging Algorithm Initialized")
    
    def OptionFilter(self, universe):
        """Filter options to only ATM calls and puts"""
        return (universe
                .Strikes(-2, 2)
                .Expiration(20, 45)
                .OnlyApplyFilterAtMarketOpen())
    
    def OnData(self, data):
        """Update price tracking"""
        if not self.spy.Symbol in data.Bars:
            return
            
        current_price = data.Bars[self.spy.Symbol].Close
        if current_price > 0:
            if self.last_price is not None:
                self.price_history.append(current_price)
            self.last_price = current_price
        
        # Process option data
        if data.OptionChains.Count > 0:
            for chain in data.OptionChains.Values:
                self.UpdateOptions(chain, current_price)
        
        # Keep history manageable
        if len(self.price_history) > 50:
            self.price_history = self.price_history[-25:]
    
    def UpdateOptions(self, chain, current_price):
        """Find ATM options and update volatility"""
        best_call = None
        best_put = None
        min_strike_diff = float('inf')
        
        for contract in chain:
            strike_diff = abs(contract.Strike - current_price)
            
            if strike_diff < min_strike_diff:
                min_strike_diff = strike_diff
                if contract.Right == 0:  # Call
                    best_call = contract.Symbol
                elif contract.Right == 1:  # Put
                    best_put = contract.Symbol
                
                # Update volatility estimate
                if contract.ImpliedVolatility > 0:
                    self.last_vol = contract.ImpliedVolatility
        
        if best_call is not None:
            self.current_call_symbol = best_call
        if best_put is not None:
            self.current_put_symbol = best_put
    
    def Rebalance(self):
        """Main rebalancing logic"""
        if self.last_price is None:
            self.Log("No price data available")
            return
        
        # Initialize SPY position conservatively
        if not self.Portfolio[self.spy.Symbol].Invested:
            self.SetHoldings(self.spy.Symbol, self.target_spy_allocation)
            spy_shares = int(self.Portfolio[self.spy.Symbol].Quantity)
            self.Log(f"Initialized SPY position: {spy_shares} shares at ${self.last_price:.2f}")
        
        # Get simple model prediction (simulate for now)
        actions = self.GetModelPrediction()
        if actions is not None:
            self.ExecuteOptionTrades(actions)
    
    def GetModelPrediction(self):
        """Simulate model prediction - replace with actual model later"""
        if len(self.price_history) < 5:
            return np.array([0.0, 0.0])  # No action if insufficient data
        
        # Simple momentum-based simulation
        recent_return = (self.price_history[-1] / self.price_history[-5] - 1)
        
        # Simple logic: buy calls if price going up, puts if going down
        call_signal = np.clip(recent_return * 2, -1, 1)
        put_signal = np.clip(-recent_return * 2, -1, 1)
        
        return np.array([call_signal, put_signal])
    
    def ExecuteOptionTrades(self, actions):
        """Execute conservative option trades"""
        if self.current_call_symbol is None or self.current_put_symbol is None:
            self.Log("No options available for trading")
            return
        
        call_trade = int(actions[0] * self.max_trade_per_step)
        put_trade = int(actions[1] * self.max_trade_per_step)
        
        # Execute call trades
        if call_trade != 0:
            new_position = max(-self.max_contracts_per_type, 
                              min(self.max_contracts_per_type, 
                                  self.current_call_contracts + call_trade))
            trade_quantity = new_position - self.current_call_contracts
            
            if trade_quantity != 0:
                try:
                    self.MarketOrder(self.current_call_symbol, trade_quantity)
                    self.current_call_contracts = new_position
                    self.Log(f"Call trade: {trade_quantity} contracts")
                except Exception as e:
                    self.Log(f"Call trade failed: {e}")
        
        # Execute put trades
        if put_trade != 0:
            new_position = max(-self.max_contracts_per_type,
                              min(self.max_contracts_per_type,
                                  self.current_put_contracts + put_trade))
            trade_quantity = new_position - self.current_put_contracts
            
            if trade_quantity != 0:
                try:
                    self.MarketOrder(self.current_put_symbol, trade_quantity)
                    self.current_put_contracts = new_position
                    self.Log(f"Put trade: {trade_quantity} contracts")
                except Exception as e:
                    self.Log(f"Put trade failed: {e}")
    
    def OnOrderEvent(self, orderEvent):
        """Log order events"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"Order filled: {orderEvent.Symbol} qty: {orderEvent.FillQuantity} @ ${orderEvent.FillPrice:.2f}")
        elif orderEvent.Status == OrderStatus.Invalid:
            self.Log(f"Order rejected: {orderEvent.Symbol} - {orderEvent.Message}")
