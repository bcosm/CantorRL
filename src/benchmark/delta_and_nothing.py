#!/usr/bin/env python3
import os
import sys
import numpy as np
import time # Good practice, though not strictly for benchmark logic

# --- Environment Import ---
# Adjust this import based on your project structure.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
try:
    from src.env.hedging_env import HedgingEnv # Assumes HedgingEnv.py is in src/env/
except ImportError:
    try:
        from hedging_env import HedgingEnv # Fallback if in same directory
    except ImportError:
        print("ERROR: Could not import HedgingEnv. Make sure it's in your PYTHONPATH,")
        print("or adjust the import statement in this benchmark script.")
        sys.exit(1)
# --- Configuration (should match your main training script's HedgingEnv params) ---
DATA_FILE = './data/paths_rbergomi_options_100k.npz' # Ensure this path is correct
TRANSACTION_COST = 0.05
LAMBDA_COST = 1.0
SHARES_TO_HEDGE = 10000
MAX_CONTRACTS_HELD_PER_TYPE = 200 # Max contracts held (position limit)
MAX_TRADE_PER_STEP = 15         # Max contracts to trade in one step (action magnitude limit)
OPTION_CONTRACT_MULTIPLIER = 100

NUM_EVAL_EPISODES = 100
PROFILE_INTERVAL_BENCHMARK = 10_000_000 # Effectively disable HedgingEnv's profiling prints

# --- Helper Function to Run a Strategy ---
def run_benchmark_strategy(strategy_name, action_selection_function, num_episodes=NUM_EVAL_EPISODES):
    """
    Runs a given strategy on the HedgingEnv and reports results.
    """
    print(f"\n--- Running Benchmark: {strategy_name} ---")
    
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found at {os.path.abspath(DATA_FILE)}")
        return float('nan'), float('nan')

    env = HedgingEnv(
        data_file_path=DATA_FILE,
        transaction_cost_per_contract=TRANSACTION_COST,
        lambda_cost=LAMBDA_COST,
        shares_to_hedge=SHARES_TO_HEDGE,
        max_contracts_held_per_type=MAX_CONTRACTS_HELD_PER_TYPE,
        max_trade_per_step=MAX_TRADE_PER_STEP, # Crucial for continuous action space
        profile_print_interval=PROFILE_INTERVAL_BENCHMARK
    )

    total_rewards = []
    total_raw_pnls = []
    total_pnl_variance_component = []
    total_tx_cost_penalty_component = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        episode_cumulative_reward = 0.0
        episode_cumulative_raw_pnl = 0.0
        episode_pnl_variance_term = 0.0
        episode_tx_cost_penalty_term = 0.0

        while not (terminated or truncated):
            state_info = {
                "S_t": env.current_stock_price,
                "v_t": env.current_volatility,
                "call_delta_atm": obs[7],
                "put_delta_atm": obs[9],
                "current_call_contracts": env.call_contracts_held,
                "current_put_contracts": env.put_contracts_held,
                "shares_to_hedge": env.shares_held_fixed,
                "option_contract_multiplier": env.option_contract_multiplier,
                "max_trade_per_step": env.max_trade_per_step # Pass this for clipping action
            }

            action = action_selection_function(state_info)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_cumulative_reward += reward
            episode_cumulative_raw_pnl += info.get('step_pnl_total', 0)
            episode_pnl_variance_term += info.get('reward_pnl_component', 0)
            episode_tx_cost_penalty_term += info.get('transaction_cost_penalty', 0)
        
        total_rewards.append(episode_cumulative_reward)
        total_raw_pnls.append(episode_cumulative_raw_pnl)
        total_pnl_variance_component.append(episode_pnl_variance_term)
        total_tx_cost_penalty_component.append(episode_tx_cost_penalty_term)

        if (episode + 1) % (num_episodes // 10 if num_episodes >= 10 else 1) == 0:
            print(f"  Episode {episode + 1}/{num_episodes} completed. "
                  f"Avg Reward so far: {np.mean(total_rewards):.2f}")

    env.close()

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_raw_pnl = np.mean(total_raw_pnls)
    std_raw_pnl = np.std(total_raw_pnls)
    avg_pnl_var_term = np.mean(total_pnl_variance_component)
    avg_tx_cost_term = np.mean(total_tx_cost_penalty_component)

    print(f"\n  --- Results for: {strategy_name} ({num_episodes} Episodes) ---")
    print(f"  Average Episodic Reward (VarianceMin Objective): {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"    Avg PnL Variance Component of Reward: {avg_pnl_var_term:.2f}")
    print(f"    Avg Tx Cost Penalty Component of Reward: {avg_tx_cost_term:.2f}") # This will be negative
    print(f"  Average Episodic Raw PnL (for insight):    {avg_raw_pnl:.2f} +/- {std_raw_pnl:.2f}")
    print(f"  ----------------------------------------------------")
    return avg_reward, avg_raw_pnl

# --- Strategy Definitions ---

def no_hedging_action_selector(state_info):
    """Always outputs an action to trade zero contracts for calls and puts."""
    return np.array([0.0, 0.0], dtype=np.float32)

def delta_hedging_action_selector(state_info):
    """
    Attempts to neutralize delta by requesting trades for calls or puts.
    Outputs a continuous action [trade_calls, trade_puts], clipped by max_trade_per_step.
    """
    call_delta_atm_pershare = state_info["call_delta_atm"]  # Typically ~0.5
    put_delta_atm_pershare = state_info["put_delta_atm"]    # Typically ~-0.5

    N_call_current = state_info["current_call_contracts"]
    N_put_current = state_info["current_put_contracts"]
    shares_to_hedge = state_info["shares_to_hedge"]
    multiplier = state_info["option_contract_multiplier"]
    max_trade = state_info["max_trade_per_step"]

    current_options_delta_shares = (N_call_current * call_delta_atm_pershare + \
                                    N_put_current * put_delta_atm_pershare) * multiplier
    stock_position_delta_shares = shares_to_hedge
    current_total_portfolio_delta_shares = stock_position_delta_shares + current_options_delta_shares
    
    target_options_delta_shares = -stock_position_delta_shares 
    delta_change_needed_from_options_shares = target_options_delta_shares - current_options_delta_shares

    # Threshold to avoid tiny trades (e.g., half the delta of one ATM call contract)
    min_delta_threshold = 0.5 * abs(call_delta_atm_pershare) * multiplier 
    if abs(delta_change_needed_from_options_shares) < min_delta_threshold:
        return np.array([0.0, 0.0], dtype=np.float32)

    requested_call_trade = 0.0
    requested_put_trade = 0.0

    # Simple heuristic: use calls for positive delta, puts for negative delta needed
    if delta_change_needed_from_options_shares > 0: # Need to add positive delta
        if abs(call_delta_atm_pershare) > 1e-6: # Avoid division by zero
            num_contracts = delta_change_needed_from_options_shares / (call_delta_atm_pershare * multiplier)
            requested_call_trade = np.clip(num_contracts, -max_trade, max_trade)
    elif delta_change_needed_from_options_shares < 0: # Need to add negative delta
        if abs(put_delta_atm_pershare) > 1e-6: # Avoid division by zero
            num_contracts = delta_change_needed_from_options_shares / (put_delta_atm_pershare * multiplier)
            requested_put_trade = np.clip(num_contracts, -max_trade, max_trade)
            
    # The environment's step function will round these continuous values.
    return np.array([requested_call_trade, requested_put_trade], dtype=np.float32)

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting benchmarks with {NUM_EVAL_EPISODES} episodes each.")
    print(f"Target HedgingEnv with CONTINUOUS action space.")
    print(f"Data file: {os.path.abspath(DATA_FILE)}")
    print(f"Reward function objective: Minimize PnL Variance (-(PnL_per_share)^2) and Tx Costs.")
    print(f"A reward closer to zero is better for this objective.")

    no_hedge_avg_reward, no_hedge_avg_pnl = run_benchmark_strategy(
        "No Hedging (Continuous Action [0,0])", 
        no_hedging_action_selector
    )

    delta_hedge_avg_reward, delta_hedge_avg_pnl = run_benchmark_strategy(
        "Delta Hedging (Continuous Action)", 
        delta_hedging_action_selector
    )

    print("\n--- Overall Benchmark Summary (Continuous Action Space Env) ---")
    print(f"No Hedging Avg Reward: {no_hedge_avg_reward:.2f}\t(Avg Raw PnL: {no_hedge_avg_pnl:.2f})")
    print(f"Delta Hedging Avg Reward: {delta_hedge_avg_reward:.2f}\t(Avg Raw PnL: {delta_hedge_avg_pnl:.2f})")
    print("Reminder: For this environment's reward, values closer to 0 are better.")