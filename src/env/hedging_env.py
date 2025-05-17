import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.stats import norm

class HedgingEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 1}

    def __init__(self, data_file_path, 
                 transaction_cost_per_contract=0.05, 
                 lambda_cost=1.0, 
                 initial_cash=0.0,
                 shares_to_hedge=10000,
                 max_contracts_held_per_type=200): # Max long or short for 10k shares

        super().__init__()

        try:
            data = np.load(data_file_path)
            self.stock_paths = data['paths'].astype(np.float32)
            self.vol_paths = data['volatilities'].astype(np.float32)
            self.call_prices_paths = data['call_prices_atm'].astype(np.float32)
            self.put_prices_paths = data['put_prices_atm'].astype(np.float32)
        except Exception as e:
            raise FileNotFoundError(f"Could not load or parse data from {data_file_path}. Error: {e}")

        if not (self.stock_paths.shape == self.vol_paths.shape and \
                self.stock_paths.shape[0] == self.call_prices_paths.shape[0] == self.put_prices_paths.shape[0] and \
                self.stock_paths.shape[1] == self.call_prices_paths.shape[1] + 1 == self.put_prices_paths.shape[1] + 1):
            raise ValueError("Data shapes are inconsistent.")

        self.num_episodes = self.stock_paths.shape[0]
        self.episode_length = self.stock_paths.shape[1] -1 

        self.transaction_cost_per_contract = transaction_cost_per_contract
        self.lambda_cost = lambda_cost
        self.initial_cash = initial_cash
        self.max_contracts_held = max_contracts_held_per_type
        
        self.shares_held_fixed = shares_to_hedge
        self.option_contract_multiplier = 100 # Standard US equity options

        self.risk_free_rate = 0.04 
        self.option_tenor_years = 30/252 
        self.action_tiers = [1, 5, 15] # Number of contracts for each tier

        # Action Space (13 actions):
        # 0: Hold
        # 1-3: Buy Calls (1, 5, 15 contracts)
        # 4-6: Sell Calls (1, 5, 15 contracts)
        # 7-9: Buy Puts (1, 5, 15 contracts)
        # 10-12: Sell Puts (1, 5, 15 contracts)
        self.action_space = spaces.Discrete(1 + 2 * 2 * len(self.action_tiers)) # 1 + Call/Put * Buy/Sell * NumTiers

        low_bounds = np.array([
            0.1, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0
        ], dtype=np.float32)
        high_bounds = np.array([
            10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0, 1.0, 50.0, 1.0, 1.0
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, shape=(13,), dtype=np.float32)

        self.current_episode_idx = -1
        self.current_step = 0
        self.initial_S0_for_episode = 1.0

    def _calculate_greeks(self, S, K, T, r, v_spot):
        sigma = np.sqrt(np.maximum(v_spot, 1e-8)) 
        if T <= 1e-6 or sigma <= 1e-6: 
            call_delta = 1.0 if S > K else (0.5 if S == K else 0.0)
            put_delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
            gamma = 0.0
            return call_delta, gamma, put_delta, gamma

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_delta = norm.cdf(d1)
        put_delta = norm.cdf(d1) - 1.0 
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        return call_delta, gamma, put_delta, gamma

    def _get_observation(self):
        S_t = self.current_stock_price
        C_t = self.current_call_price
        P_t = self.current_put_price
        v_t = self.current_volatility

        norm_S_t = S_t / self.initial_S0_for_episode
        norm_C_t = C_t / self.initial_S0_for_episode 
        norm_P_t = P_t / self.initial_S0_for_episode
        
        norm_call_held = self.call_contracts_held / self.max_contracts_held
        norm_put_held = self.put_contracts_held / self.max_contracts_held
        
        norm_time_to_end = (self.episode_length - self.current_step) / self.episode_length

        K_atm_t = np.round(S_t) # ATM strike for tradable options
        call_delta, call_gamma, put_delta, put_gamma = self._calculate_greeks(
            S_t, K_atm_t, self.option_tenor_years, self.risk_free_rate, v_t
        )

        if self.current_step == 0 or self.S_t_minus_1 == 0:
            lagged_S_return = 0.0
            lagged_v_change = 0.0
        else:
            lagged_S_return = (S_t - self.S_t_minus_1) / self.S_t_minus_1
            lagged_v_change = v_t - self.v_t_minus_1
        
        lagged_S_return = np.clip(lagged_S_return, -1.0, 1.0)
        lagged_v_change = np.clip(lagged_v_change, -1.0, 1.0)

        # Normalize the observation
        obs = np.array([
            norm_S_t, norm_C_t, norm_P_t,
            norm_call_held, norm_put_held,
            v_t, norm_time_to_end,
            call_delta, call_gamma,
            put_delta, put_gamma,
            lagged_S_return, lagged_v_change
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_episode_idx = self.np_random.integers(self.num_episodes)
        self.current_path_S = self.stock_paths[self.current_episode_idx]
        self.current_path_v = self.vol_paths[self.current_episode_idx]
        self.current_path_C = self.call_prices_paths[self.current_episode_idx]
        self.current_path_P = self.put_prices_paths[self.current_episode_idx]
        
        self.current_step = 0
        
        self.initial_S0_for_episode = self.current_path_S[0]
        if self.initial_S0_for_episode == 0: self.initial_S0_for_episode = 1.0 

        self.current_stock_price = self.current_path_S[self.current_step]
        self.current_volatility = self.current_path_v[self.current_step]
        self.current_call_price = self.current_path_C[self.current_step] 
        self.current_put_price = self.current_path_P[self.current_step]

        self.call_contracts_held = 0
        self.put_contracts_held = 0
        self.cash_balance = self.initial_cash
        
        initial_options_value = 0 
        self.portfolio_value_t_minus_1 = (self.shares_held_fixed * self.current_stock_price) + \
                                         initial_options_value + \
                                         self.cash_balance
        
        self.S_t_minus_1 = self.current_stock_price 
        self.v_t_minus_1 = self.current_volatility 

        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action):
        prev_call_contracts = self.call_contracts_held
        prev_put_contracts = self.put_contracts_held
        
        contracts_traded_val = 0
        
        if action == 0: # Hold
            pass
        else:
            action_idx_offset = action -1 
            
            option_type_idx = action_idx_offset // (2 * len(self.action_tiers)) # 0 for Call, 1 for Put
            action_type_offset = action_idx_offset % (2 * len(self.action_tiers))
            
            direction_idx = action_type_offset // len(self.action_tiers) # 0 for Buy, 1 for Sell
            tier_idx = action_type_offset % len(self.action_tiers)
            
            contracts_traded_val = self.action_tiers[tier_idx]
            if direction_idx == 1: # Sell
                contracts_traded_val *= -1
            
            if option_type_idx == 0: # Call
                self.call_contracts_held += contracts_traded_val
            else: # Put
                self.put_contracts_held += contracts_traded_val
        
        self.call_contracts_held = np.clip(self.call_contracts_held, -self.max_contracts_held, self.max_contracts_held)
        self.put_contracts_held = np.clip(self.put_contracts_held, -self.max_contracts_held, self.max_contracts_held)

        actual_calls_changed = self.call_contracts_held - prev_call_contracts
        actual_puts_changed = self.put_contracts_held - prev_put_contracts

        transaction_costs_this_step = (abs(actual_calls_changed) + abs(actual_puts_changed)) * self.transaction_cost_per_contract
        self.cash_balance -= transaction_costs_this_step
        
        self.S_t_minus_1 = self.current_stock_price
        self.v_t_minus_1 = self.current_volatility
        
        self.current_step += 1
        terminated = (self.current_step >= self.episode_length)
        truncated = False 

        self.current_stock_price = self.current_path_S[self.current_step]
        self.current_volatility = self.current_path_v[self.current_step]

        if not terminated:
            self.current_call_price = self.current_path_C[self.current_step]
            self.current_put_price = self.current_path_P[self.current_step]
        else:
            self.current_call_price = self.current_path_C[self.current_step-1] 
            self.current_put_price = self.current_path_P[self.current_step-1]


        current_options_value = (self.call_contracts_held * self.current_call_price * self.option_contract_multiplier) + \
                                (self.put_contracts_held * self.current_put_price * self.option_contract_multiplier)
        
        portfolio_value_t = (self.shares_held_fixed * self.current_stock_price) + \
                            current_options_value + \
                            self.cash_balance
        
        step_pnl = portfolio_value_t - self.portfolio_value_t_minus_1
        
        reward = - (step_pnl**2) - (self.lambda_cost * transaction_costs_this_step)
        
        self.portfolio_value_t_minus_1 = portfolio_value_t
        
        observation = self._get_observation()
        info = {'step_pnl': step_pnl, 'transaction_costs': transaction_costs_this_step, 'portfolio_value': portfolio_value_t,
                'call_contracts': self.call_contracts_held, 'put_contracts': self.put_contracts_held, 'cash': self.cash_balance}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass