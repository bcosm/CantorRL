# QuantConnect Algorithm - Quiet Diagnostic Version
# Minimal logging to avoid rate limits

import numpy as np
from AlgorithmImports import *
from datetime import datetime, timedelta

class RLHedgingQuietAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        self.Debug("Starting Quiet Diagnostic Algorithm")
        
        # Basic setup
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 3, 31)
        self.SetCash(100000)
        
        # Add SPY
        self.spy = self.AddEquity("SPY", Resolution.Minute)
        self.spy_symbol = self.spy.Symbol
        
        # Add SPY options
        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(-5, 5, timedelta(days=7), timedelta(days=60))
        
        # State variables
        self.spy_shares = 1000
        self.rebalance_frequency = 50  # Less frequent rebalancing
        self.bar_count = 0
        self.last_spy_price = None
        
        # Counters (only log summary)
        self.total_rebalances = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.last_log_time = None
        
        self.Debug("Initialization complete")
    
    def OnData(self, data):
        self.bar_count += 1
        
        # Get SPY price
        if not data.ContainsKey(self.spy_symbol) or data[self.spy_symbol] is None:
            return
            
        spy_price = data[self.spy_symbol].Close
        self.last_spy_price = spy_price
        
        # Rebalance less frequently and log only key events
        if self.bar_count % self.rebalance_frequency == 0:
            self.total_rebalances += 1
            self.Rebalance(data)
            
            # Log summary only every 10 rebalances or significant events
            if self.total_rebalances % 10 == 0 or self.successful_trades > 0:
                self.Debug(f"Rebalance #{self.total_rebalances}: SPY ${spy_price:.2f}, Trades {self.successful_trades}/{self.failed_trades}")
    
    def Rebalance(self, data):
        # Buy SPY if needed (quietly)
        spy_holdings = self.Portfolio[self.spy_symbol].Quantity
        if spy_holdings < self.spy_shares:
            shares_to_buy = self.spy_shares - spy_holdings
            if self.Portfolio.Cash > shares_to_buy * self.last_spy_price:
                order_id = self.MarketOrder(self.spy_symbol, shares_to_buy)
                if order_id:
                    self.successful_trades += 1
        
        # Get options
        option_chains = [x for x in data.OptionChains]
        if not option_chains:
            return
            
        chain = option_chains[0].Value
        if not chain:
            return
        
        # Find ATM options
        calls, puts = self.GetATMOptions(chain, self.last_spy_price)
        
        if not calls and not puts:
            return
        
        # Simple alternating trade strategy
        trade_type = "call" if self.total_rebalances % 2 == 1 else "put"
        
        if trade_type == "call" and calls:
            option_to_trade = calls[0]
            order_id = self.MarketOrder(option_to_trade.Symbol, 1)
            if order_id:
                self.successful_trades += 1
                # Only log successful option trades
                self.Debug(f"CALL TRADE: {option_to_trade.Strike} strike, order {order_id}")
                
        elif trade_type == "put" and puts:
            option_to_trade = puts[0]
            order_id = self.MarketOrder(option_to_trade.Symbol, 1)
            if order_id:
                self.successful_trades += 1
                # Only log successful option trades
                self.Debug(f"PUT TRADE: {option_to_trade.Strike} strike, order {order_id}")
    
    def GetATMOptions(self, chain, spy_price):
        """Get at-the-money call and put options"""
        calls = []
        puts = []
        
        # Filter for options close to ATM
        atm_threshold = 2.0
        
        for contract in chain:
            if abs(contract.Strike - spy_price) <= atm_threshold:
                if contract.Right == OptionRight.Call:
                    calls.append(contract)
                else:
                    puts.append(contract)
        
        # Sort by how close to ATM
        calls.sort(key=lambda x: abs(x.Strike - spy_price))
        puts.sort(key=lambda x: abs(x.Strike - spy_price))
        
        return calls[:3], puts[:3]
    
    def OnOrderEvent(self, orderEvent):
        """Only log important order events"""
        if orderEvent.Status == OrderStatus.Filled and "SPY" in str(orderEvent.Symbol) and len(str(orderEvent.Symbol)) > 3:
            # This is an option fill (SPY options have longer symbol names)
            self.Debug(f"OPTION FILLED: {orderEvent.Symbol} x{orderEvent.FillQuantity} @ ${orderEvent.FillPrice:.2f}")
        elif orderEvent.Status == OrderStatus.Invalid:
            self.failed_trades += 1
            self.Debug(f"ORDER INVALID: {orderEvent.Symbol}")
    
    def OnEndOfAlgorithm(self):
        """Final summary"""
        self.Debug(f"FINAL: {self.total_rebalances} rebalances, {self.successful_trades} successful trades, {self.failed_trades} failed")
        self.Debug(f"Portfolio Value: ${self.Portfolio.TotalPortfolioValue:.2f}")
