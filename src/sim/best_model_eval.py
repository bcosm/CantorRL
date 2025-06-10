#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np # For evaluate_policy if it returns numpy arrays
import torch

# --- PyTorch/CUDA/cuDNN Diagnostics (Optional, but good for context) ---
print(f"--- PyTorch/CUDA/cuDNN Diagnostics ---")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA available: True, Device: {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA available: False")
print(f"--------------------------------------")

# --- Main Imports ---
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed


# --- Environment Import ---
# Adjust this import based on your project structure.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
try:
    from src.env.hedging_env import HedgingEnv
except ImportError:
    try:
        from hedging_env import HedgingEnv # Fallback if in same directory
    except ImportError as e_imp:
        print(f"ERROR: Failed to import HedgingEnv: {e_imp}. Ensure hedging_env.py (continuous action version) is accessible.")
        sys.exit(1)

# --- Configuration - Ensure these match the training environment ---
DATA_FILE = './data/paths_rbergomi_options_100k.npz'
TRANSACTION_COST = 0.05
LAMBDA_COST = 1.0
SHARES = 10_000
MAX_CONTRACTS_HELD_PER_TYPE = 200
MAX_TRADE_PER_STEP = 15         # From your continuous action HedgingEnv
N_PARALLEL_ENVS_FOR_EVAL = 4    # Can be different from training, e.g., N_PARALLEL_ENVS // 2
SEED = 420 # Use a different seed for evaluation environment instances if desired

N_EVAL_EPISODES_FOR_SCRIPT = 100 # Number of episodes to run for this evaluation
EVAL_ENV_PROFILE_INTERVAL = 10_000_000 # Effectively disable env profiling prints

# !!! IMPORTANT: UPDATE THESE PATHS !!!
# Path to the 'best_model.zip' saved by EvalCallback during the long final training run
MODEL_PATH = "./ppo_hedging_logs/best_model_epic_continuous_final_trial38/best_model.zip" # EXAMPLE PATH - CHANGE THIS
# Path to the VecNormalize statistics saved at the end of that same final training run
VECNORM_PATH = "./ppo_hedging_vecnormalize_epic_continuous_final_trial38.pkl" # EXAMPLE PATH - CHANGE THIS


def create_eval_env(data_file_path, seed=0, profile_interval=EVAL_ENV_PROFILE_INTERVAL):
    """Factory for creating the evaluation environment."""
    env = HedgingEnv(
        data_file_path=data_file_path,
        transaction_cost_per_contract=TRANSACTION_COST,
        lambda_cost=LAMBDA_COST,
        shares_to_hedge=SHARES,
        max_contracts_held_per_type=MAX_CONTRACTS_HELD_PER_TYPE,
        max_trade_per_step=MAX_TRADE_PER_STEP,
        profile_print_interval=profile_interval
    )
    return Monitor(env, allow_early_resets=True)

if __name__ == "__main__":
    set_random_seed(SEED) # Seed for reproducibility of this evaluation script

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(VECNORM_PATH):
        logging.error(f"VecNormalize file not found: {VECNORM_PATH}")
        sys.exit(1)
    if not os.path.exists(DATA_FILE):
        logging.error(f"Data file for environment not found: {DATA_FILE}")
        sys.exit(1)

    logging.info(f"Loading model from: {MODEL_PATH}")
    logging.info(f"Loading VecNormalize stats from: {VECNORM_PATH}")

    # 1. Create the vectorized evaluation environment
    eval_env_seed_start = SEED + 100 # Use a different base seed for eval envs
    # Create the non-normalized VecEnv first
    eval_vec_env = SubprocVecEnv([
        lambda i=idx: create_eval_env(DATA_FILE, seed=eval_env_seed_start + i) 
        for idx in range(N_PARALLEL_ENVS_FOR_EVAL)
    ])

    # 2. Load the VecNormalize statistics and apply them to the new VecEnv
    #    It's important that this eval_vec_env has the same observation and action spaces
    #    as the environment used during training whose stats were saved.
    logging.info("Applying VecNormalize statistics to the evaluation environment...")
    eval_env_normalized = VecNormalize.load(VECNORM_PATH, eval_vec_env)
    
    # Set VecNormalize to evaluation mode (don't update stats, don't normalize rewards)
    eval_env_normalized.training = False
    eval_env_normalized.norm_reward = False
    logging.info("VecNormalize set to evaluation mode.")

    # 3. Load the trained PPO model
    #    Provide the now correctly wrapped and normalized environment to the load function.
    #    This helps SB3 verify spaces and set up the model correctly.
    try:
        model = RecurrentPPO.load(MODEL_PATH, env=eval_env_normalized, device="cuda")
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading the model: {e}", exc_info=True)
        if 'eval_vec_env' in locals(): eval_vec_env.close() # Clean up SubprocVecEnv
        sys.exit(1)


    # 4. Evaluate the loaded model
    logging.info(f"Starting evaluation for {N_EVAL_EPISODES_FOR_SCRIPT} episodes...")
    
    all_episode_rewards = []
    all_episode_lengths = []
    all_episode_raw_pnls = [] # To store raw PnL from info dict
    
    # Use a custom evaluation loop to access the info dictionary for raw PnL
    # Note: evaluate_policy also returns rewards and lengths, but not custom info easily.
    for episode_num in range(N_EVAL_EPISODES_FOR_SCRIPT):
        obs = eval_env_normalized.reset()
        # For RecurrentPPO, need to manage LSTM states if not using evaluate_policy's handling
        # However, model.predict() with SubprocVecEnv should handle this.
        # If using manual LSTM state handling, it's more complex.
        # For simplicity with evaluate_policy, we'll use it.
        # If we stick to a manual loop for info dict:
        lstm_states = None 
        episode_reward = 0
        episode_length = 0
        episode_raw_pnl = 0
        terminated = [False] * eval_env_normalized.num_envs
        truncated = [False] * eval_env_normalized.num_envs
        
        # Since N_PARALLEL_ENVS_FOR_EVAL can be > 1, we need to run until all envs complete one episode
        # This manual loop is more for single env eval. evaluate_policy is better for VecEnv.
        # Let's use evaluate_policy and then potentially a separate loop if info needed & not available.
        # For now, just use evaluate_policy for simplicity and standard metrics.

    # Simpler: Use evaluate_policy (but it won't give custom info like raw PnL easily)
    # To get raw PnL easily, a custom loop is better. Let's stick with evaluate_policy for consistency with training.
    # For raw PnL, one would typically add a callback during evaluate_policy or run a custom loop.
    # For this script, let's focus on the main reward metric.

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env_normalized,
        n_eval_episodes=N_EVAL_EPISODES_FOR_SCRIPT,
        deterministic=True,
        render=False,
        warn=True,
        return_episode_rewards=False # If True, returns list of rewards per episode
    )

    logging.info(f"--- Evaluation Results for {MODEL_PATH} ---")
    logging.info(f"Number of evaluation episodes: {N_EVAL_EPISODES_FOR_SCRIPT}")
    logging.info(f"Mean reward: {mean_reward:.2f}")
    logging.info(f"Std reward: {std_reward:.2f}")
    logging.info("Reminder: For this environment's reward (variance minimization), values closer to 0 are better.")

    # Clean up
    eval_env_normalized.close() # This will also close the underlying eval_vec_env

    logging.info("Evaluation script finished.")