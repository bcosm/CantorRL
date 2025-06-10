# QuantConnect Algorithm - Diagnostic Version
# This version forces trades to happen and provides extensive logging

import numpy as np
from AlgorithmImports import *
from datetime import datetime, timedelta

class RLHedgingDiagnosticAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        self.Debug("=== INITIALIZING DIAGNOSTIC ALGORITHM ===")
        
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
        self.spy_shares = 1000  # Start small for testing
        self.option_contracts = {}
        self.rebalance_frequency = 5  # Rebalance every 5 bars for testing
        self.bar_count = 0
        self.last_spy_price = None
        self.force_trade_counter = 0
        
        # Diagnostic counters
        self.total_rebalances = 0
        self.successful_trades = 0
        self.failed_trades = 0
        
        self.Debug(f"Target SPY shares: {self.spy_shares}")
        self.Debug("=== INITIALIZATION COMPLETE ===")
    
    def OnData(self, data):
        self.bar_count += 1
        
        # Get SPY price
        if not data.ContainsKey(self.spy_symbol) or data[self.spy_symbol] is None:
            return
            
        spy_price = data[self.spy_symbol].Close
        self.last_spy_price = spy_price
        
        # Force rebalancing every N bars for testing
        if self.bar_count % self.rebalance_frequency == 0:
            self.Debug(f"\n=== FORCED REBALANCE #{self.total_rebalances + 1} at bar {self.bar_count} ===")
            self.Debug(f"SPY Price: ${spy_price:.2f}")
            self.Rebalance(data)
    
    def Rebalance(self, data):
        self.total_rebalances += 1
        
        # Step 1: Check SPY position
        spy_holdings = self.Portfolio[self.spy_symbol].Quantity
        self.Debug(f"Current SPY holdings: {spy_holdings}")
        
        # Step 2: Buy SPY if we don't have enough
        if spy_holdings < self.spy_shares:
            shares_to_buy = self.spy_shares - spy_holdings
            if self.Portfolio.Cash > shares_to_buy * self.last_spy_price:
                self.Debug(f"Buying {shares_to_buy} SPY shares at ${self.last_spy_price:.2f}")
                order_id = self.MarketOrder(self.spy_symbol, shares_to_buy)
                if order_id:
                    self.successful_trades += 1
                    self.Debug(f"✓ SPY buy order placed: {order_id}")
                else:
                    self.failed_trades += 1
                    self.Debug("✗ SPY buy order failed")
        
        # Step 3: Get available options
        option_chains = [x for x in data.OptionChains]
        if not option_chains:
            self.Debug("✗ No option chains available")
            return
            
        chain = option_chains[0].Value
        if not chain:
            self.Debug("✗ Option chain is empty")
            return
            
        self.Debug(f"Option chain has {len(chain)} contracts")
        
        # Step 4: Find ATM options
        calls, puts = self.GetATMOptions(chain, self.last_spy_price)
        
        if not calls and not puts:
            self.Debug("✗ No ATM options found")
            return
            
        self.Debug(f"Found {len(calls)} ATM calls, {len(puts)} ATM puts")
        
        # Step 5: FORCE A SIMPLE TRADE for diagnostic purposes
        self.force_trade_counter += 1
        trade_type = "call" if self.force_trade_counter % 2 == 1 else "put"
        
        if trade_type == "call" and calls:
            option_to_trade = calls[0]
            contracts_to_trade = 1  # Just 1 contract for testing
            self.Debug(f"FORCING CALL TRADE: {option_to_trade.Symbol} x {contracts_to_trade}")
            
            order_id = self.MarketOrder(option_to_trade.Symbol, contracts_to_trade)
            if order_id:
                self.successful_trades += 1
                self.Debug(f"✓ Call option order placed: {order_id}")
            else:
                self.failed_trades += 1
                self.Debug("✗ Call option order failed")
                
        elif trade_type == "put" and puts:
            option_to_trade = puts[0]
            contracts_to_trade = 1  # Just 1 contract for testing
            self.Debug(f"FORCING PUT TRADE: {option_to_trade.Symbol} x {contracts_to_trade}")
            
            order_id = self.MarketOrder(option_to_trade.Symbol, contracts_to_trade)
            if order_id:
                self.successful_trades += 1
                self.Debug(f"✓ Put option order placed: {order_id}")
            else:
                self.failed_trades += 1
                self.Debug("✗ Put option order failed")
        
        # Step 6: Portfolio summary
        total_value = self.Portfolio.TotalPortfolioValue
        cash = self.Portfolio.Cash
        self.Debug(f"Portfolio Value: ${total_value:.2f}, Cash: ${cash:.2f}")
        self.Debug(f"Trade Stats - Success: {self.successful_trades}, Failed: {self.failed_trades}")
    
    def GetATMOptions(self, chain, spy_price):
        """Get at-the-money call and put options"""
        calls = []
        puts = []
        
        # Filter for options close to ATM
        atm_threshold = 2.0  # Within $2 of ATM
        
        for contract in chain:
            if abs(contract.Strike - spy_price) <= atm_threshold:
                if contract.Right == OptionRight.Call:
                    calls.append(contract)
                else:
                    puts.append(contract)
        
        # Sort by how close to ATM
        calls.sort(key=lambda x: abs(x.Strike - spy_price))
        puts.sort(key=lambda x: abs(x.Strike - spy_price))
        
        return calls[:3], puts[:3]  # Return top 3 of each
    
    def OnOrderEvent(self, orderEvent):
        """Log all order events for debugging"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"ORDER FILLED: {orderEvent.Symbol} | Qty: {orderEvent.FillQuantity} | Price: ${orderEvent.FillPrice:.2f}")
        elif orderEvent.Status == OrderStatus.Canceled:
            self.Debug(f"ORDER CANCELED: {orderEvent.Symbol}")
        elif orderEvent.Status == OrderStatus.Invalid:
            self.Debug(f"ORDER INVALID: {orderEvent.Symbol} | Message: {orderEvent.Message}")
    
    def OnEndOfAlgorithm(self):
        """Summary statistics"""
        self.Debug("\n=== ALGORITHM COMPLETE ===")
        self.Debug(f"Total Rebalances: {self.total_rebalances}")
        self.Debug(f"Successful Trades: {self.successful_trades}")
        self.Debug(f"Failed Trades: {self.failed_trades}")
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug("=== END SUMMARY ===")
