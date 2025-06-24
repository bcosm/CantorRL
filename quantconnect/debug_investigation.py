# Debug Investigation Script for QuantConnect Trading Issues
# This script identifies and logs detailed debugging information

from AlgorithmImports import *
from datetime import timedelta
from typing import List, Dict, Any
import numpy as np

class DebugInvestigation(QCAlgorithm):
    """Debugging version to identify why trading stops after SPY initialization"""
    
    def initialize(self):
        self.set_start_date(2023, 1, 1)
        self.set_end_date(2023, 1, 15)  # Shorter period for debugging
        self.set_cash(100000000)
        
        # Add SPY
        self.spy = self.add_equity("SPY", Resolution.MINUTE)
        self.spy.set_data_normalization_mode(DataNormalizationMode.RAW)
        
        # Add options
        option = self.add_option("SPY", Resolution.MINUTE)
        option.set_filter(self.option_filter)
        
        self.set_warm_up(timedelta(days=2))  # Shorter warmup for debugging
        
        # Debugging counters
        self.rebalance_calls = 0
        self.on_data_calls = 0
        self.option_chain_received = 0
        self.available_options_count = 0
        self.model_predictions_made = 0
        
        # State tracking
        self.last_price = None
        self.last_vol = None
        self.price_history = []
        self.vol_history = []
        self.available_options = {}
        self.shares_to_hedge = 2000
        self.current_call_contracts = 0
        self.current_put_contracts = 0
        self.max_contracts_per_type = 200
        self.max_trade_per_step = 15
        
        # Schedule debugging
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 0),
            self.debug_rebalance
        )
        
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.every(timedelta(minutes=60)),
            self.debug_rebalance
        )
        
        self.log("=== DEBUG INVESTIGATION INITIALIZED ===")
    
    def option_filter(self, universe):
        return (universe
            .strikes(-4, 4)
            .expiration(timedelta(25), timedelta(35)))
    
    def on_data(self, data):
        self.on_data_calls += 1
        
        # Log first few on_data calls
        if self.on_data_calls <= 5:
            self.log(f"=== ON_DATA CALL #{self.on_data_calls} ===")
            self.log(f"Is warming up: {self.is_warming_up}")
            self.log(f"SPY in data.bars: {self.spy.symbol in data.bars}")
            self.log(f"Option chains count: {data.option_chains.count}")
        
        if self.is_warming_up:
            return
            
        if not self.spy.symbol in data.bars:
            if self.on_data_calls <= 10:
                self.log("SPY data missing in on_data")
            return
        
        # Update price
        current_price = data.bars[self.spy.symbol].close
        if current_price > 0:
            if self.last_price is not None:
                self.price_history.append(current_price)
            self.last_price = current_price
            
            if self.on_data_calls <= 10:
                self.log(f"Updated price: ${current_price:.2f}")
        
        # Check option chains
        if data.option_chains.count > 0:
            self.option_chain_received += 1
            if self.option_chain_received <= 5:
                self.log(f"=== OPTION CHAIN #{self.option_chain_received} RECEIVED ===")
                
            self.update_available_options(data)
            
            for chain in data.option_chains.values():
                self.update_implied_volatility(chain, current_price)
                if self.option_chain_received <= 2:
                    self.log(f"Chain has {len(list(chain))} contracts")
        else:
            if self.on_data_calls <= 10:
                self.log("No option chains in data")
        
        # Keep histories manageable
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-50:]
        if len(self.vol_history) > 100:
            self.vol_history = self.vol_history[-50:]
    
    def update_available_options(self, data):
        """Debug version of update_available_options"""
        self.available_options.clear()
        valid_contracts = 0
        
        if data and data.option_chains.count > 0:
            for chain in data.option_chains.values():
                for contract in chain:
                    # Check pricing data
                    has_last = contract.last_price > 0
                    has_bid_ask = (contract.bid_price > 0 and contract.ask_price > 0 and 
                                  contract.ask_price > contract.bid_price)
                    
                    if has_last or has_bid_ask:
                        if self.last_price is not None:
                            strike_diff_pct = abs(contract.strike - self.last_price) / self.last_price
                            time_to_expiry = (contract.expiry.date() - self.time.date()).days
                            
                            if strike_diff_pct <= 0.10 and 25 <= time_to_expiry <= 35:
                                self.available_options[contract.symbol] = contract
                                valid_contracts += 1
        
        self.available_options_count = len(self.available_options)
        
        if self.option_chain_received <= 5:
            self.log(f"Available options updated: {self.available_options_count} valid contracts")
            if self.available_options_count > 0:
                sample_symbols = list(self.available_options.keys())[:3]
                for sym in sample_symbols:
                    contract = self.available_options[sym]
                    self.log(f"  Sample: {sym} Strike: {contract.strike} Expiry: {contract.expiry.date()} Last: {contract.last_price}")
    
    def update_implied_volatility(self, chain, current_price):
        """Debug version of IV update"""
        atm_options = []
        
        for contract in chain:
            if abs(contract.strike - current_price) < current_price * 0.02:
                atm_options.append(contract)
        
        if atm_options:
            valid_ivs = [opt.implied_volatility for opt in atm_options if opt.implied_volatility > 0]
            if valid_ivs:
                avg_iv = np.mean(valid_ivs)
                self.vol_history.append(avg_iv)
                self.last_vol = avg_iv
                
                if self.option_chain_received <= 3:
                    self.log(f"IV updated: {avg_iv:.4f} from {len(valid_ivs)} ATM options")
    
    def debug_rebalance(self):
        """Debug version of rebalance with extensive logging"""
        self.rebalance_calls += 1
        
        self.log(f"=== REBALANCE CALL #{self.rebalance_calls} ===")
        self.log(f"Time: {self.time}")
        self.log(f"Is warming up: {self.is_warming_up}")
        
        if self.is_warming_up:
            self.log("Skipping - still warming up")
            return
        
        # Check SPY position
        spy_qty = self.portfolio[self.spy.symbol].quantity if self.spy.symbol in self.portfolio else 0
        spy_invested = self.portfolio[self.spy.symbol].invested
        self.log(f"SPY position: {spy_qty} shares, invested: {spy_invested}")
        
        # Initialize SPY if needed
        if not spy_invested:
            self.log(f"*** INITIALIZING SPY POSITION: {self.shares_to_hedge} shares ***")
            order_ticket = self.market_order(self.spy.symbol, self.shares_to_hedge)
            if order_ticket:
                self.log(f"SPY order placed successfully: {self.shares_to_hedge} shares")
            else:
                self.log("*** CRITICAL: SPY ORDER FAILED ***")
            return  # Exit early on first rebalance to focus on SPY initialization
        
        # Check data availability
        self.log(f"Last price: {self.last_price}")
        self.log(f"Last vol: {self.last_vol}")
        self.log(f"Price history length: {len(self.price_history)}")
        self.log(f"Vol history length: {len(self.vol_history)}")
        self.log(f"Available options count: {self.available_options_count}")
        
        # Check observation creation
        if self.last_price is None:
            self.log("ISSUE: No price data available")
            return
            
        if self.last_vol is None:
            self.log("ISSUE: No volatility data available")
            return
            
        if len(self.price_history) < 2:
            self.log(f"ISSUE: Insufficient price history: {len(self.price_history)}")
            return
        
        if not self.available_options:
            self.log("ISSUE: No available options with valid data")
            return
        
        # Try to create observation
        observation = self.create_debug_observation()
        if observation is None:
            self.log("ISSUE: Failed to create observation")
            return
        
        self.log(f"Observation created successfully: length {len(observation)}")
        self.log(f"Observation sample: {observation[:6]}")
        
        # Simulate model prediction (since we don't have the model in debug mode)
        mock_actions = np.array([0.1, -0.05])  # Small test actions
        self.log(f"Mock actions: {mock_actions}")
        
        # Test option trading execution
        self.debug_execute_trades(mock_actions)
        
        self.model_predictions_made += 1
        self.log(f"=== END REBALANCE #{self.rebalance_calls} ===")
    
    def create_debug_observation(self):
        """Debug version of observation creation"""
        try:
            S_t = self.last_price
            v_t = self.last_vol
            
            # Get option prices
            C_t = 0.0
            P_t = 0.0
            
            if self.available_options:
                for symbol, contract in self.available_options.items():
                    if hasattr(symbol.id, 'option_right'):
                        if str(symbol.id.option_right) == "OptionRight.Call" and C_t == 0.0:
                            C_t = contract.last_price if contract.last_price > 0 else contract.ask_price
                        elif str(symbol.id.option_right) == "OptionRight.Put" and P_t == 0.0:
                            P_t = contract.last_price if contract.last_price > 0 else contract.ask_price
            
            # Use Black-Scholes fallback
            if C_t <= 0 or P_t <= 0:
                # Simple BS approximation
                if C_t <= 0:
                    C_t = S_t * 0.05  # 5% of stock price
                if P_t <= 0:
                    P_t = S_t * 0.05
            
            # Normalization
            s0_safe = max(self.price_history[0] if self.price_history else S_t, 25.0)
            norm_S_t = S_t / s0_safe
            norm_C_t = C_t / s0_safe
            norm_P_t = P_t / s0_safe
            
            # Positions
            norm_call_held = self.current_call_contracts / self.max_contracts_per_type
            norm_put_held = self.current_put_contracts / self.max_contracts_per_type
            
            # Time feature
            norm_time_to_end = 0.5  # Mock value
            
            # Greeks (mock values for debugging)
            call_delta = 0.5
            call_gamma = 0.01
            put_delta = -0.5
            put_gamma = 0.01
            
            # Returns
            if len(self.price_history) >= 2:
                lagged_S_return = (S_t - self.price_history[-1]) / self.price_history[-1]
            else:
                lagged_S_return = 0.0
            
            if len(self.vol_history) >= 2:
                lagged_v_change = v_t - self.vol_history[-1]
            else:
                lagged_v_change = 0.0
            
            lagged_S_return = np.clip(lagged_S_return, -1.0, 1.0)
            lagged_v_change = np.clip(lagged_v_change, -1.0, 1.0)
            
            obs = np.array([
                norm_S_t, norm_C_t, norm_P_t, norm_call_held, norm_put_held,
                v_t, norm_time_to_end, call_delta, call_gamma,
                put_delta, put_gamma, lagged_S_return, lagged_v_change
            ], dtype=np.float32)
            
            self.log(f"Debug observation: S={norm_S_t:.3f}, C={norm_C_t:.3f}, P={norm_P_t:.3f}")
            return obs
            
        except Exception as e:
            self.log(f"Error creating observation: {str(e)}")
            return None
    
    def debug_execute_trades(self, actions):
        """Debug version of trade execution"""
        call_trade = int(round(actions[0] * self.max_trade_per_step))
        put_trade = int(round(actions[1] * self.max_trade_per_step))
        
        self.log(f"Scaled trades: call={call_trade}, put={put_trade}")
        
        # Find options
        call_symbol, put_symbol = self.find_debug_atm_options()
        
        if call_symbol is None or put_symbol is None:
            self.log("ISSUE: No ATM options found")
            return
        
        self.log(f"Found ATM options: call={call_symbol}, put={put_symbol}")
        
        # Check if options are in securities
        call_in_securities = self.securities.contains_key(call_symbol)
        put_in_securities = self.securities.contains_key(put_symbol)
        
        self.log(f"Call in securities: {call_in_securities}")
        self.log(f"Put in securities: {put_in_securities}")
        
        if call_in_securities:
            call_price = self.securities[call_symbol].price
            self.log(f"Call price: {call_price}")
        
        if put_in_securities:
            put_price = self.securities[put_symbol].price
            self.log(f"Put price: {put_price}")
        
        # For debugging, don't actually place trades, just log what would happen
        self.log(f"Would place call trade: {call_trade} contracts")
        self.log(f"Would place put trade: {put_trade} contracts")
    
    def find_debug_atm_options(self):
        """Debug version of ATM option finder"""
        if self.last_price is None or not self.available_options:
            return None, None
        
        call_symbol = None
        put_symbol = None
        min_call_diff = float('inf')
        min_put_diff = float('inf')
        
        call_candidates = []
        put_candidates = []
        
        for symbol, contract in self.available_options.items():
            if (hasattr(symbol, 'security_type') and 
                str(symbol.security_type) == "SecurityType.Option" and 
                symbol.underlying == self.spy.symbol):
                
                strike_price = symbol.id.strike_price
                strike_diff = abs(strike_price - self.last_price)
                
                if hasattr(symbol.id, 'option_right'):
                    if str(symbol.id.option_right) == "OptionRight.Call":
                        call_candidates.append((symbol, strike_diff))
                        if strike_diff < min_call_diff:
                            call_symbol = symbol
                            min_call_diff = strike_diff
                    elif str(symbol.id.option_right) == "OptionRight.Put":
                        put_candidates.append((symbol, strike_diff))
                        if strike_diff < min_put_diff:
                            put_symbol = symbol
                            min_put_diff = strike_diff
        
        self.log(f"Call candidates: {len(call_candidates)}, Put candidates: {len(put_candidates)}")
        if call_symbol:
            self.log(f"Best call strike diff: {min_call_diff:.2f}")
        if put_symbol:
            self.log(f"Best put strike diff: {min_put_diff:.2f}")
        
        return call_symbol, put_symbol
