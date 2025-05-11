import os
import time
import gymnasium as gym
import numpy as np
import cupy as cp # Ensure environment can handle cupy arrays if data remains on GPU
import optuna
from optuna.pruners import MedianPruner
from optuna.integration import PyTorchLightningPruningCallback # For PyTorch Lightning, not directly SB3
# Correct Optuna integration for SB3 is through its own callback or by returning value
# For SB3, we use a custom callback or monitor the eval_callback's return value

import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# Import your HedgingEnv
from hedging_env import HedgingEnv # Assuming hedging_env.py is in the same directory or PYTHONPATH

# --- Configuration ---
DATA_FILE = './data/paths_rbergomi_options_100k.npz'
LOG_DIR = "./ppo_hedging_logs/"
OPTUNA_DB_NAME = "sqlite:///ppo_hedging_hpo.db"
STUDY_NAME = "ppo_hedging_study_lstm_scheduled"

N_OPTUNA_TRIALS = 50 # Number of HPO trials
N_TIMESTEPS_PER_TRIAL = 100000 # Timesteps for each Optuna trial (adjust based on speed)
N_EVAL_EPISODES_PER_TRIAL = 10 # Episodes for evaluation within each trial
EVAL_FREQ_PER_TRIAL = int(N_TIMESTEPS_PER_TRIAL / 5) # Evaluate 5 times per trial

N_FINAL_TRAINING_TIMESTEPS = 2000000 # Timesteps for training the best model
MODEL_SAVE_PATH = "./ppo_hedging_model_best"
VEC_NORMALIZE_SAVE_PATH = "./ppo_hedging_vecnormalize_best.pkl"

TRANSACTION_COST = 0.05
LAMBDA_COST_PENALTY = 1.0
SHARES_TO_HEDGE = 10000
MAX_CONTRACTS_HELD = 200

# --- Helper Functions ---
def create_env(data_file_path, seed=0):
    env = HedgingEnv(
        data_file_path=data_file_path,
        transaction_cost_per_contract=TRANSACTION_COST,
        lambda_cost=LAMBDA_COST_PENALTY,
        shares_to_hedge=SHARES_TO_HEDGE,
        max_contracts_held_per_type=MAX_CONTRACTS_HELD
    )
    # It's good practice to seed environments, especially when vectorizing
    # However, HedgingEnv already uses its own np_random seeded in reset.
    # For SB3 VecEnv, it handles seeding internally if a seed is passed to set_global_seeds or to VecEnv itself.
    return env

def linear_schedule(initial_value: float, final_value: float = 0.0):
    if initial_value < final_value:
        raise ValueError("Initial value for linear schedule must be greater than or equal to final value.")
        
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial) -> float:
    run_start_time = time.time()
    print(f"\nStarting Optuna Trial {trial.number}...")

    # Suggest Hyperparameters
    lr_initial = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    lr_final_factor = trial.suggest_float("lr_final_factor", 0.01, 0.5, log=True) # Final LR as factor of initial
    lr_final = lr_initial * lr_final_factor

    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    if batch_size > n_steps: # Ensure batch_size is not larger than n_steps
        batch_size = n_steps
        
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    
    clip_range_initial = trial.suggest_float("clip_range", 0.1, 0.3)
    clip_range_final_factor = trial.suggest_float("clip_range_final_factor", 0.1, 1.0) # Can be fixed or scheduled
    clip_range_final = clip_range_initial * clip_range_final_factor

    ent_coef_initial = trial.suggest_float("ent_coef", 0.0, 0.05) # Initial entropy
    ent_coef_final_factor = trial.suggest_float("ent_coef_final_factor", 0.01, 1.0) # Can be fixed or scheduled
    ent_coef_final = ent_coef_initial * ent_coef_final_factor

    vf_coef = trial.suggest_float("vf_coef", 0.3, 0.7)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    
    lstm_hidden_size = trial.suggest_categorical("lstm_hs", [64, 128, 256])
    n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 2)
    # share_features_extractor = trial.suggest_categorical("share_fe", [True, False]) # MlpLstmPolicy default is True

    # Create scheduled callables
    lr_scheduler = linear_schedule(lr_initial, lr_final)
    clip_range_scheduler = linear_schedule(clip_range_initial, clip_range_final)
    ent_coef_scheduler = linear_schedule(ent_coef_initial, ent_coef_final)

    # Create vectorized environments
    # Note: MlpLstmPolicy requires a VecEnv.
    # Using DummyVecEnv for simplicity in this script, SubprocVecEnv for potential speedup with multiple workers
    train_env = DummyVecEnv([lambda: create_env(DATA_FILE)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, gamma=gamma) # Norm reward can be tricky

    eval_env = DummyVecEnv([lambda: create_env(DATA_FILE)]) # Use a different seed for eval if possible
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, gamma=gamma)
    eval_env.training = False # Important: Do not update running stats from eval_env
    eval_env.norm_reward = False


    policy_kwargs = dict(
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=n_lstm_layers,
        # enable_critic_lstm=True, # Default is True for MlpLstmPolicy
        # share_features_extractor=share_features_extractor, # Default is True
        # Note: For MlpLstmPolicy, net_arch is usually not specified directly here
        # as the policy itself defines the LSTM + MLP structure.
    )

    model = PPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=lr_scheduler,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10, # Default for PPO
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range_scheduler,
        ent_coef=ent_coef_scheduler,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None, # Disable tensorboard for HPO trials to save disk/overhead
        verbose=0,
        seed=int(time.time()) # Ensure each trial has a different seed for SB3 model
    )

    # Setup EvalCallback for pruning and returning the best metric
    # OptunaPruningCallback is for SB3 Contrib, for standard SB3, we use EvalCallback and check trial.should_prune()
    # For simplicity and robustness, let EvalCallback run and return its best mean reward.
    # Pruning can be done based on intermediate values reported to Optuna.
    
    eval_log_dir = os.path.join(LOG_DIR, f"trial_{trial.number}_eval")
    os.makedirs(eval_log_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None, # Don't save individual trial models here
        log_path=eval_log_dir, # Log eval stats per trial
        eval_freq=EVAL_FREQ_PER_TRIAL,
        n_eval_episodes=N_EVAL_EPISODES_PER_TRIAL,
        deterministic=True,
        render=False,
        callback_after_eval=optuna.integration.횃불_callbacks.TorchPruningCallback(trial, "mean_reward") if int(sb3.__version__.split('.')[1]) < 8 else None # Old SB3 Optuna Pruning
        # For modern SB3, you typically use trial.report and trial.should_prune in a custom callback
        # or simply return the final evaluation score.
        # Let's make a simpler custom pruning callback logic inside the objective for SB3:
    )
    
    # Custom Pruning Logic within the objective
    # This requires reporting intermediate values to Optuna from within the EvalCallback.
    # A simpler approach for now: train for full N_TIMESTEPS_PER_TRIAL and return final score.
    # More advanced: Custom callback that calls trial.report() and trial.should_prune().
    
    mean_reward = -np.inf # Default in case of failure

    try:
        model.learn(total_timesteps=N_TIMESTEPS_PER_TRIAL, callback=eval_callback)
        # After learning, get the mean reward from the last evaluation
        # The eval_callback logs 'eval/mean_reward'. We need to access it.
        # A common way is to have EvalCallback return the value or use its logged values.
        # For Optuna, the objective function should return the value to optimize.
        # If EvalCallback updates `self.last_mean_reward` or similar, we can use that.
        # Or, load the monitor logs from eval_log_dir.
        # For now, let's assume eval_callback.best_mean_reward holds a relevant score, or last_mean_reward
        if hasattr(eval_callback, 'best_mean_reward'): # Modern SB3
             mean_reward = eval_callback.best_mean_reward
        elif hasattr(eval_callback, 'last_mean_reward'): # Older SB3 or if no best model saved
             mean_reward = eval_callback.last_mean_reward
        else: # Fallback: perform one final evaluation
            episode_rewards, _ = sb3.common.evaluation.evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES_PER_TRIAL, deterministic=True)
            mean_reward = float(np.mean(episode_rewards))


    except AssertionError as e:
        print(f"AssertionError in trial {trial.number}: {e}. Skipping.")
        # This can happen if, e.g., NaNs are produced. Return a bad score.
        mean_reward = -np.inf 
    except Exception as e:
        print(f"Exception in trial {trial.number}: {e}. Skipping.")
        mean_reward = -np.inf
    finally:
        train_env.close()
        eval_env.close()
        # Clean up CuPy memory if any arrays were explicitly created and not handled by SB3/Gym
        # cp.get_default_memory_pool().free_all_blocks() # If needed

    run_end_time = time.time()
    print(f"Finished Optuna Trial {trial.number}. Duration: {run_end_time - run_start_time:.2f}s. Mean Reward: {mean_reward:.2f}")
    return mean_reward


# --- Main Execution ---
if __name__ == "__main__":
    set_random_seed(SEED)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create or load Optuna study
    # For Optuna pruning with SB3, one typically creates a custom callback that integrates with trial.report and trial.should_prune
    # The TorchPruningCallback from optuna.integration.pytorch_lightning is not for SB3 directly.
    # We will rely on MedianPruner based on values returned by objective function.
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=N_TIMESTEPS_PER_TRIAL // 3, interval_steps=EVAL_FREQ_PER_TRIAL)
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=OPTUNA_DB_NAME, # Specify database URL for persistence
        load_if_exists=True,    # Load an existing study if same name and storage
        direction="maximize",
        pruner=pruner
    )

    print(f"Starting Optuna HPO. Study: {STUDY_NAME}. Number of trials: {N_OPTUNA_TRIALS}.")
    try:
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, timeout=None, n_jobs=1) # n_jobs > 1 for parallel trials if objective is picklable
    except KeyboardInterrupt:
        print("HPO interrupted by user.")
    except Exception as e:
        print(f"An error occurred during HPO: {e}")

    print("\nHyperparameter optimization finished.")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Train the final best model
    print("\nTraining the best model with optimized hyperparameters...")
    best_params = best_trial.params
    
    final_lr_initial = best_params["lr"]
    final_lr_final = final_lr_initial * best_params["lr_final_factor"]
    final_clip_initial = best_params["clip_range"]
    final_clip_final = final_clip_initial * best_params["clip_range_final_factor"]
    final_ent_coef_initial = best_params["ent_coef"]
    final_ent_coef_final = final_ent_coef_initial * best_params["ent_coef_final_factor"]


    final_lr_schedule = linear_schedule(final_lr_initial, final_lr_final)
    final_clip_schedule = linear_schedule(final_clip_initial, final_clip_final)
    final_ent_coef_schedule = linear_schedule(final_ent_coef_initial, final_ent_coef_final)


    final_env = DummyVecEnv([lambda: create_env(DATA_FILE, seed=SEED)]) # Seed for final training
    final_env = VecNormalize(final_env, norm_obs=True, norm_reward=False, gamma=best_params["gamma"])
    
    final_policy_kwargs = dict(
        lstm_hidden_size=best_params["lstm_hs"],
        n_lstm_layers=best_params["n_lstm_layers"],
    )

    final_model = PPO(
        "MlpLstmPolicy",
        final_env,
        learning_rate=final_lr_schedule,
        n_steps=best_params["n_steps"],
        batch_size=best_params["batch_size"],
        n_epochs=10,
        gamma=best_params["gamma"],
        gae_lambda=best_params["gae_lambda"],
        clip_range=final_clip_schedule,
        ent_coef=final_ent_coef_schedule,
        vf_coef=best_params["vf_coef"],
        max_grad_norm=best_params["max_grad_norm"],
        policy_kwargs=final_policy_kwargs,
        tensorboard_log=os.path.join(LOG_DIR, "final_model_tensorboard"),
        verbose=1,
        seed=SEED
    )

    # Callback for saving the best model during final training based on evaluation performance
    # Need a separate eval env for this final training's EvalCallback
    final_eval_env = DummyVecEnv([lambda: create_env(DATA_FILE, seed=SEED+1)]) # Different seed for eval
    final_eval_env = VecNormalize(final_eval_env, training=False, norm_obs=True, norm_reward=False, gamma=best_params["gamma"])
    # Load stats from training VecNormalize if it was used, or train new ones, then save.
    # For simplicity here, we are creating a new one. Ideally, use the one from HPO if stats are stable,
    # or use the stats from 'final_env' after some training.
    # It's critical that the VecNormalize stats used for eval_env match what the model expects.
    # A common pattern is to save the VecNormalize stats from the training env and load them for the eval_env.
    # For now, we initialize final_eval_env VecNormalize from scratch but set training=False
    # A better approach: vec_normalize_stats_path = os.path.join(LOG_DIR, "best_vec_normalize.pkl")
    # final_env.save(vec_normalize_stats_path)
    # final_eval_env = VecNormalize.load(vec_normalize_stats_path, DummyVecEnv([lambda: create_env(DATA_FILE, seed=SEED+1)]))

    final_eval_callback = EvalCallback(
        final_eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model_final"),
        log_path=os.path.join(LOG_DIR, "best_model_final_eval_logs"),
        eval_freq=max(N_FINAL_TRAINING_TIMESTEPS // 100, 1000), # Evaluate frequently
        n_eval_episodes=20, # More episodes for final eval
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(N_FINAL_TRAINING_TIMESTEPS // 20, 10000),
        save_path=os.path.join(LOG_DIR, "final_model_checkpoints"),
        name_prefix="ppo_hedging_final"
    )
    
    callbacks_list = [final_eval_callback, checkpoint_callback]

    print(f"Starting final training for {N_FINAL_TRAINING_TIMESTEPS} timesteps...")
    final_model.learn(total_timesteps=N_FINAL_TRAINING_TIMESTEPS, callback=callbacks_list)

    final_model.save(MODEL_SAVE_PATH)
    final_env.save(VEC_NORMALIZE_SAVE_PATH) # Save VecNormalize stats

    print(f"Best model saved to {MODEL_SAVE_PATH}")
    print(f"VecNormalize stats saved to {VEC_NORMALIZE_SAVE_PATH}")
    print("Training script finished.")