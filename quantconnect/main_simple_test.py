# QuantConnect Algorithm - Super Simple Test
# Just buy SPY and trade 1 option contract to test basic functionality

from AlgorithmImports import *
from datetime import timedelta

class SimpleOptionTestAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        self.Debug("=== SIMPLE OPTION TEST STARTING ===")
        
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 1, 31)  # Just one month
        self.SetCash(50000)
        
        # Add SPY
        spy = self.AddEquity("SPY", Resolution.Daily)
        self.spy_symbol = spy.Symbol
        
        # Add SPY options
        option = self.AddOption("SPY", Resolution.Daily)
        option.SetFilter(-2, 2, timedelta(days=30), timedelta(days=60))
        
        self.traded = False
        
        self.Debug("=== INITIALIZATION DONE ===")
    
    def OnData(self, data):
        if self.traded:
            return
            
        # Buy some SPY first
        if not self.Portfolio[self.spy_symbol].Invested:
            self.Debug("Buying 100 SPY shares")
            self.MarketOrder(self.spy_symbol, 100)
            return
        
        # Try to trade an option
        for kvp in data.OptionChains:
            chain = kvp.Value
            if len(chain) == 0:
                continue
                
            self.Debug(f"Found option chain with {len(chain)} contracts")
            
            # Just pick the first available contract
            contract = list(chain)[0]
            self.Debug(f"Trading option: {contract.Symbol}")
            self.Debug(f"Strike: {contract.Strike}, Expiry: {contract.Expiry}")
            
            # Buy 1 contract
            order_id = self.MarketOrder(contract.Symbol, 1)
            self.Debug(f"Option order placed: {order_id}")
            
            self.traded = True
            break
    
    def OnOrderEvent(self, orderEvent):
        self.Debug(f"ORDER EVENT: {orderEvent.Symbol} | Status: {orderEvent.Status} | Qty: {orderEvent.FillQuantity}")
    
    def OnEndOfAlgorithm(self):
        self.Debug(f"=== FINAL VALUE: ${self.Portfolio.TotalPortfolioValue:.2f} ===")
