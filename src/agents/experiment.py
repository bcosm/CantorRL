#!/usr/bin/env python3
import os
import sys
import time
import logging

import gymnasium as gym
import numpy as np
import torch
import optuna # Still imported in case HPO is re-enabled
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned

# --- PyTorch/CUDA/cuDNN Diagnostics ---
print(f"--- PyTorch/CUDA/cuDNN Diagnostics ---")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA available: True")
    print(f"CUDA version PyTorch built with: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    try:
        current_gpu_idx = torch.cuda.current_device()
        print(f"Current GPU index: {current_gpu_idx}")
        print(f"Current GPU name: {torch.cuda.get_device_name(current_gpu_idx)}")
    except Exception as e:
        print(f"Could not get current GPU info: {e}")
else:
    print(f"CUDA available: False")
print(f"Is cuDNN enabled? {torch.backends.cudnn.enabled}")
print(f"Is cuDNN benchmark mode (initial)? {torch.backends.cudnn.benchmark}")
print(f"--------------------------------------")

torch.backends.cudnn.benchmark = True
print(f"Is cuDNN benchmark mode (after setting)? {torch.backends.cudnn.benchmark}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
try:
    from src.env.hedging_env import HedgingEnv 
except ImportError:
    try:
        from hedging_env import HedgingEnv 
    except ImportError as e_imp:
        logging.error(f"Failed to import HedgingEnv: {e_imp}. Ensure hedging_env.py (continuous action version) is accessible.")
        sys.exit(1)

# --- Configuration ---
DATA_FILE = './data/paths_rbergomi_options_100k.npz'
LOG_DIR = "./ppo_hedging_logs/"
# HPO DB and Study Name (not used if RUN_HPO is False, but kept for completeness)
OPTUNA_DB_URL = "sqlite:///ppo_hedging_epic_continuous.db" 
STUDY_NAME = "ppo_hedging_epic_recurrent_continuous"     
N_OPTUNA_TRIALS = 50
N_TIMESTEPS_PER_TRIAL = 150_000 
N_EVAL_EPISODES_HPO = 10 # Eval episodes during HPO
EVAL_FREQ_HPO = N_TIMESTEPS_PER_TRIAL // 5 

# Final training params for this experiment
N_FINAL_TIMESTEPS_EXPERIMENT = 1_000_000 # Reduced for faster experimental feedback
MODEL_SAVE_PATH_BASE = "./ppo_hedging_model_epic_exp1" # Specific to this experiment
VECNORM_SAVE_PATH_BASE = "./ppo_hedging_vecnormalize_epic_exp1"

TRANSACTION_COST = 0.05 # Consider the previous discussion on this value
LAMBDA_COST = 1.0
SHARES = 10_000
MAX_CONTRACTS_HELD_PER_TYPE = 200
MAX_TRADE_PER_STEP = 15 
SEED = 42 # Use a consistent seed for this experiment
N_PARALLEL_ENVS = 8

# Profiling interval for HedgingEnv instances
# Set high for HPO to reduce log spam, lower for final detailed run if needed
HPO_ENV_PROFILE_INTERVAL = N_TIMESTEPS_PER_TRIAL * N_PARALLEL_ENVS * 2 
FINAL_ENV_PROFILE_INTERVAL = 10000 # More frequent for observing env step times

def create_env(data_file_path, seed=0, profile_interval=FINAL_ENV_PROFILE_INTERVAL):
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

def linear_schedule(initial_value: float, final_value: float = 0.0):
    if initial_value < final_value: raise ValueError("Initial must be >= final.")
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule

class ETAEstimator(BaseCallback):
    def __init__(self, total_timesteps: int, log_interval: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps_for_run = total_timesteps 
        self.log_interval = log_interval
        self.start_time = None
    def init_callback(self, model):
        super().init_callback(model)
        self.start_time = time.time()
    def _on_step(self) -> bool:
        if self.num_timesteps and self.num_timesteps % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            if self.num_timesteps > 0 and elapsed > 0:
                eta = elapsed * (self.total_timesteps_for_run - self.num_timesteps) / self.num_timesteps
                logging.info(f"Progress: {self.num_timesteps}/{self.total_timesteps_for_run} timesteps. ETA: {eta:.1f}s ({eta/3600:.2f}h)")
            else:
                logging.info(f"Progress: {self.num_timesteps}/{self.total_timesteps_for_run} timesteps. ETA: N/A")
        return True

class OptunaPruneCallback(BaseCallback): # Kept for potential HPO re-enabling
    def __init__(self, trial: optuna.Trial, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
    def _on_step(self) -> bool:
        if self.parent is None or not hasattr(self.parent, 'last_mean_reward'): return True
        mean_reward = self.parent.last_mean_reward
        self.trial.report(mean_reward, self.model.num_timesteps) 
        if self.trial.should_prune(): raise TrialPruned()
        return True

# HPO Objective function (kept for completeness, but RUN_HPO will be False)
def objective(trial: optuna.Trial) -> float:
    t0 = time.time()
    # ... (Full objective function as in previous script, for brevity not repeated here)
    # ... (It should use HPO_ENV_PROFILE_INTERVAL for create_env)
    # ... (And its search space should match the one discussed for EPIC models)
    # This function would only be called if RUN_HPO is True.
    # For this experimental script, we will skip calling it.
    # If you need it, copy from the script where it was fully defined with correct HPO params.
    logging.warning("Objective function called, but HPO is meant to be skipped for this experimental run.")
    return -float('inf') # Should not be reached if RUN_HPO is False

# --- Main Execution ---
if __name__ == "__main__":
    set_random_seed(SEED)
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.info(f"--- EPIC Continuous Action Space Agent - EXPERIMENT 1 ---")
    
    RUN_HPO = False # Set to False to go straight to the experimental training
    
    if RUN_HPO:
        # ... (Full HPO logic as in previous script - for brevity not repeated)
        # ... (This part would run Optuna and find best_hpo_params)
        logging.info("RUN_HPO was True, but this experimental script focuses on predefined settings.")
        logging.info("Please set RUN_HPO=False to run the experiment, or integrate HPO results manually.")
        # For this script, we will define experimental_params directly
        # If HPO was run, you'd get best_hpo_params = study.best_trial.params.copy()
        # and study_best_trial_num = study.best_trial.number
        sys.exit("RUN_HPO is True in experimental script. Please set to False or adapt.")

    else: 
        logging.info("RUN_HPO is False. Using predefined EXPERIMENTAL parameters.")
        # Base parameters (similar to generic/good defaults, to be partially overridden)
        experimental_params = { 
            "lr_final_factor": 0.1, 
            "n_steps": 512,         # Sequence length for LSTM
            "batch_size": 64,       
            "gamma": 0.99, 
            "gae_lambda": 0.95, 
            "clip_range": 0.2, 
            "clip_range_final_factor": 0.5,
            "ent_coef": 0.005,      # Typically smaller for continuous actions
            "vf_coef": 0.5, 
            "max_grad_norm": 0.8,
            # --- Parameters FOR THIS EXPERIMENT ---
            "lr": 2e-5,             # EXPERIMENTAL: Reduced Learning Rate
            "n_epochs": 4,          # EXPERIMENTAL: Reduced PPO Epochs
            # --- EPIC Architecture ---
            "lstm_hidden_size": 512, 
            "n_lstm_layers": 4 
        }
        study_best_trial_num = "exp1" # Identifier for this experiment
        logging.info(f"--- Using Experimental Parameters for EPIC Model (Trial: {study_best_trial_num}) ---")

    logging.info("Hyperparameters chosen for this experimental training run:")
    for k, v in experimental_params.items(): logging.info(f"  {k}: {v}")
    
    params = experimental_params 

    final_policy_kwargs = dict(
        lstm_hidden_size=params["lstm_hidden_size"],
        n_lstm_layers=params["n_lstm_layers"]
    )
    
    final_batch_size = params["batch_size"]
    if params["batch_size"] > params["n_steps"]:
        logging.warning(f"Selected batch_size ({params['batch_size']}) > n_steps ({params['n_steps']})."
                        f"Using batch_size = n_steps = {params['n_steps']} instead.")
        final_batch_size = params["n_steps"]
    
    rollout_buffer_size_final = params["n_steps"] * N_PARALLEL_ENVS
    if rollout_buffer_size_final % final_batch_size != 0:
         logging.warning(f"Rollout buffer size ({rollout_buffer_size_final}) not perfectly divisible by batch_size ({final_batch_size}).")

    final_env_seed_start = SEED + 7000 
    final_env = SubprocVecEnv([lambda i=idx: create_env(DATA_FILE, seed=final_env_seed_start + i, profile_interval=FINAL_ENV_PROFILE_INTERVAL) for idx in range(N_PARALLEL_ENVS)])
    final_env = VecNormalize(final_env, norm_obs=True, norm_reward=False, gamma=params["gamma"])

    final_model_start_time = time.time()
    final_model = RecurrentPPO(
        "MlpLstmPolicy", final_env,
        learning_rate=linear_schedule(params["lr"], params["lr"] * params["lr_final_factor"]),
        n_steps=params["n_steps"],
        batch_size=final_batch_size,
        n_epochs=params["n_epochs"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=linear_schedule(params["clip_range"], params["clip_range"] * params["clip_range_final_factor"]),
        ent_coef=params["ent_coef"],
        vf_coef=params["vf_coef"],
        max_grad_norm=params["max_grad_norm"],
        policy_kwargs=final_policy_kwargs,
        device="cuda",
        tensorboard_log=os.path.join(LOG_DIR, f"tb_epic_exp1_trial{study_best_trial_num}"),
        verbose=1, seed=SEED
    )
    final_model_init_duration = time.time() - final_model_start_time
    logging.info(f"Final EPIC model (Experiment 1) initialized in {final_model_init_duration:.2f} seconds.")

    final_eval_env_seed_start = SEED + 8000
    final_eval_env = SubprocVecEnv([lambda i=idx: create_env(DATA_FILE, seed=final_eval_env_seed_start + i, profile_interval=FINAL_ENV_PROFILE_INTERVAL) for idx in range(max(1, N_PARALLEL_ENVS//2))])
    final_eval_env = VecNormalize(final_eval_env, norm_obs=True, norm_reward=False, gamma=params["gamma"])
    final_eval_env.training = False; final_eval_env.norm_reward = False
    
    # Use N_FINAL_TIMESTEPS_EXPERIMENT for this run
    min_eval_freq = params["n_steps"] * N_PARALLEL_ENVS
    final_eval_freq = max(min_eval_freq, N_FINAL_TIMESTEPS_EXPERIMENT // 20, 5000) # Eval ~20 times
    final_checkpoint_save_freq = max(min_eval_freq * 5, N_FINAL_TIMESTEPS_EXPERIMENT // 10, 20000) # Save ~10 checkpoints

    final_eta_cb = ETAEstimator(N_FINAL_TIMESTEPS_EXPERIMENT, log_interval=max(min_eval_freq, N_FINAL_TIMESTEPS_EXPERIMENT//100))
    
    model_save_path_final = f"{MODEL_SAVE_PATH_BASE}_trial{study_best_trial_num}.zip"
    vecnorm_save_path_final = f"{VECNORM_SAVE_PATH_BASE}_trial{study_best_trial_num}.pkl"

    final_eval_cb = EvalCallback(
        final_eval_env, 
        best_model_save_path=os.path.join(LOG_DIR, f"best_model_epic_exp1_trial{study_best_trial_num}"),
        log_path=os.path.join(LOG_DIR, f"eval_logs_epic_exp1_trial{study_best_trial_num}"), 
        eval_freq=final_eval_freq,
        n_eval_episodes=N_EVAL_EPISODES_HPO * 2, # Use N_EVAL_EPISODES_HPO here (was 10)
        deterministic=True, render=False
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=final_checkpoint_save_freq,
        save_path=os.path.join(LOG_DIR, f"checkpoints_epic_exp1_trial{study_best_trial_num}"),
        name_prefix="ppo_hedging_epic_exp1"
    )
    
    logging.info(f"Starting EXPERIMENTAL final EPIC model training for {N_FINAL_TIMESTEPS_EXPERIMENT} timesteps.")
    logging.info(f"Architecture: LSTM {params['n_lstm_layers']} layers, {params['lstm_hidden_size']} units.")
    logging.info(f"Training HParams: LR={params['lr']:.2e}, n_steps={params['n_steps']}, batch_size={final_batch_size}, n_epochs={params['n_epochs']}")
    logging.info(f"Eval freq: {final_eval_freq}, Checkpoint freq: {final_checkpoint_save_freq}")

    learn_start_time = time.time()
    try:
        final_model.learn(total_timesteps=N_FINAL_TIMESTEPS_EXPERIMENT, callback=[final_eval_cb, checkpoint_cb, final_eta_cb], progress_bar=True)
    except Exception as e:
        logging.error(f"Error during EPIC model experimental training: {e}", exc_info=True)
    finally:
        learn_duration = time.time() - learn_start_time
        logging.info(f"EPIC model experimental training ({N_FINAL_TIMESTEPS_EXPERIMENT} timesteps) took {learn_duration:.2f} seconds ({learn_duration/3600:.2f} hours).")
        
        final_model.save(model_save_path_final)
        logging.info(f"Final EPIC experimental model saved to {model_save_path_final}")
        if hasattr(final_env, 'save'):
            final_env.save(vecnorm_save_path_final)
            logging.info(f"VecNormalize stats for final EPIC experimental model saved to {vecnorm_save_path_final}")

    logging.info(f"--- Evaluating the final EPIC experimental model (from {model_save_path_final}) ---")
    try:
        eval_load_env_seed = SEED + 9000
        loaded_eval_env_vec = SubprocVecEnv([lambda i=idx: create_env(DATA_FILE, seed=eval_load_env_seed + i, profile_interval=FINAL_ENV_PROFILE_INTERVAL) for idx in range(max(1, N_PARALLEL_ENVS//2))])
        loaded_eval_env_norm = VecNormalize.load(vecnorm_save_path_final, loaded_eval_env_vec)
        loaded_eval_env_norm.training = False 
        loaded_eval_env_norm.norm_reward = False 

        loaded_model = RecurrentPPO.load(model_save_path_final, env=loaded_eval_env_norm, device="cuda")
        
        mean_reward, std_reward = evaluate_policy(
            loaded_model, loaded_eval_env_norm, 
            n_eval_episodes=N_EVAL_EPISODES_HPO * 5, # More thorough eval
            deterministic=True, warn=True 
        )
        logging.info(f"Evaluation of final EPIC experimental model: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        if 'loaded_eval_env_vec' in locals(): loaded_eval_env_vec.close()
    except FileNotFoundError:
        logging.error(f"Could not find {model_save_path_final} or {vecnorm_save_path_final} for final evaluation.")
    except Exception as e:
        logging.error(f"Error during final evaluation of saved EPIC model: {e}", exc_info=True)

    if 'final_eval_env' in locals() and hasattr(final_eval_env, 'close'): final_eval_env.close()
    if 'final_env' in locals() and hasattr(final_env, 'close'): final_env.close()
    logging.info("Script finished.")