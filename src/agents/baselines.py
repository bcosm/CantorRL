import numpy as np
import pandas as pd
import os
import sys
import time
import logging
import argparse

_project_root_baselines = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root_baselines not in sys.path:
    sys.path.insert(0, _project_root_baselines)

import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]   # <project>/CantorRL
sys.path.append(str(PROJECT_ROOT))
from src.env.hedging_env import HedgingEnv # Assuming it's in src.env relative to a project rootDATA_FILE_BASELINES = os.path.join(_project_root_baselines, 'data', 'paths_rbergomi_options_100k.npz')
DATA_FILE_BASELINES = "./data/paths_rbergomi_options_100k.npz"

def setup_logging_baselines(algo_name_arg):
    logger = logging.getLogger() 
    for hdlr in logger.handlers[:]: 
        logger.removeHandler(hdlr)
        hdlr.close()
    logging.basicConfig( 
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] BASELINE_RUN({algo_name_arg}): %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def evaluate_baseline_policy(env_instance, policy_fn, num_episodes=100, algo_name="baseline"):
    all_episode_mean_abs_pnl = []
    all_episode_mean_costs = []

    for episode in range(num_episodes):
        obs, info = env_instance.reset() 
        terminated = False
        truncated = False
        
        current_ep_abs_pnl_sum = 0.0
        current_ep_cost_sum = 0.0
        current_ep_steps = 0

        while not (terminated or truncated):
            action = policy_fn(obs, env_instance) 
            obs, reward, terminated, truncated, info = env_instance.step(action)
            
            current_ep_abs_pnl_sum += info.get('raw_pnl_deviation_abs', 0.0)
            current_ep_cost_sum += info.get('transaction_costs_total', 0.0)
            current_ep_steps +=1
        
        if current_ep_steps > 0:
            all_episode_mean_abs_pnl.append(current_ep_abs_pnl_sum / current_ep_steps)
            all_episode_mean_costs.append(current_ep_cost_sum / current_ep_steps)

        if (episode + 1) % max(1, num_episodes // 10) == 0 :
             logging.info(f"Baseline {algo_name}: Episode {episode+1}/{num_episodes} completed.")
    
    final_results_pnl = all_episode_mean_abs_pnl[:num_episodes]
    final_results_cost = all_episode_mean_costs[:num_episodes]

    mean_abs_pnl = np.mean(final_results_pnl) if len(final_results_pnl) > 0 else 0
    mean_cost = np.mean(final_results_cost) if len(final_results_cost) > 0 else 0
    std_abs_pnl = np.std(final_results_pnl) if len(final_results_pnl) > 0 else 0
    
    logging.info(f"Baseline {algo_name} Results (avg per step over {len(final_results_pnl)} episodes):")
    logging.info(f"  Mean |ΔPnL| per share: {mean_abs_pnl:.4f}")
    logging.info(f"  Mean Transaction Cost: {mean_cost:.4f}")
    logging.info(f"  Std |ΔPnL| per share: {std_abs_pnl:.4f}")

    return mean_abs_pnl, mean_cost, std_abs_pnl

def policy_no_hedge(obs, env_instance):
    return np.array([0.0, 0.0], dtype=env_instance.action_space.dtype)

def policy_delta_every_step(obs, env_instance):
    call_delta_obs = obs[7] 
    put_delta_obs = obs[9]  

    current_call_pos = obs[3] * env_instance.max_contracts_held 
    current_put_pos = obs[4] * env_instance.max_contracts_held  

    trade_calls = 0.0
    trade_puts = 0.0
    
    current_portfolio_delta_from_options = (current_call_pos * call_delta_obs + current_put_pos * put_delta_obs) * env_instance.option_contract_multiplier
    current_portfolio_delta_from_shares = env_instance.shares_held_fixed
    total_current_delta = current_portfolio_delta_from_shares + current_portfolio_delta_from_options
    
    target_delta_offset = -total_current_delta 

    if abs(call_delta_obs * env_instance.option_contract_multiplier) > 1e-1: 
        needed_call_trades_for_delta = target_delta_offset / (call_delta_obs * env_instance.option_contract_multiplier)
        trade_calls = needed_call_trades_for_delta
    elif abs(put_delta_obs * env_instance.option_contract_multiplier) > 1e-1: 
        needed_put_trades_for_delta = target_delta_offset / (put_delta_obs * env_instance.option_contract_multiplier)
        trade_puts = needed_put_trades_for_delta
    
    trade_calls = np.clip(trade_calls, -env_instance.max_trade_per_step, env_instance.max_trade_per_step)
    trade_puts = np.clip(trade_puts, -env_instance.max_trade_per_step, env_instance.max_trade_per_step)
    
    return np.array([trade_calls, trade_puts], dtype=env_instance.action_space.dtype)


def main():
    parser = argparse.ArgumentParser(description="Run a specific baseline evaluation and save to its own CSV.")
    parser.add_argument("--algo_name", type=str, required=True, choices=["no_hedge", "delta_every_step"], help="Name of the baseline algorithm to run.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save/append the output CSV for this baseline.")
    parser.add_argument("--w_log", type=float, required=True, help="W_RL value to log for this run.")
    parser.add_argument("--lam_log", type=float, required=True, help="Lam_RL value to log for this run.")
    parser.add_argument("--seed", type=int, default=int(time.time() % 100000), help="Random seed for the evaluation.")
    args = parser.parse_args()

    setup_logging_baselines(f"{args.algo_name}_w{args.w_log:g}_l{args.lam_log:g}")
    logging.info(f"Running Baseline Evaluation for: {args.algo_name}, Logging w_rl={args.w_log}, lam_rl={args.lam_log}, Seed: {args.seed}")
    
    policy_map = {
        "no_hedge": policy_no_hedge,
        "delta_every_step": policy_delta_every_step
    }

    if args.algo_name not in policy_map:
        logging.error(f"Unknown algorithm name: {args.algo_name}")
        sys.exit(1)

    policy_fn_to_eval = policy_map[args.algo_name]
    
    num_eval_episodes_baselines = 100 
    record_greeks_for_this_baseline = "delta" in args.algo_name 

    eval_env_instance = HedgingEnv(
        data_file_path=DATA_FILE_BASELINES,
        pnl_penalty_weight=0.0, 
        lambda_cost=0.0,
        record_metrics=record_greeks_for_this_baseline,
        loss_type="abs" 
    )
    eval_env_instance.reset(seed=args.seed) 

    mean_pnl, mean_cost, std_pnl = evaluate_baseline_policy(eval_env_instance, policy_fn_to_eval, num_eval_episodes_baselines, args.algo_name)
    eval_env_instance.close()
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    df_new_row = pd.DataFrame([{
        "algo": args.algo_name, 
        "w": args.w_log, 
        "lam": args.lam_log, 
        "mean_abs_pnl": mean_pnl, "mean_cost": mean_cost, 
        "std_abs_pnl": std_pnl, "seed": args.seed,
        "status": "eval_done", "timestamp": timestamp
    }])

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    header_exists = os.path.isfile(args.output_csv)
    df_new_row.to_csv(args.output_csv, mode='a', header=not header_exists, index=False)
    logging.info(f"Baseline {args.algo_name} (w_rl={args.w_log}, lam_rl={args.lam_log}) results appended to {args.output_csv}")

    logging.info(f"Baseline evaluation for {args.algo_name} (w_rl={args.w_log}, lam_rl={args.lam_log}) finished.")

if __name__ == "__main__":
    main()