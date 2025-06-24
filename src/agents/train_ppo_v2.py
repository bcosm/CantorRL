import logging
import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
import optuna.visualization
import glob
import re
import json

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed, configure_logger
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement, BaseCallback
import multiprocessing as mp

_project_root_train_rl = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root_train_rl not in sys.path:
    sys.path.insert(0, _project_root_train_rl)
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # <project>/CantorRL
sys.path.append(str(PROJECT_ROOT))
from src.env.hedging_env_v2 import HedgingEnv


RESULTS_DIR = os.path.join(_project_root_train_rl, "results")
PARETO_RAW_CSV = os.path.join(RESULTS_DIR, "pareto_raw.csv")
OPTUNA_DB_DIR = os.path.join(RESULTS_DIR, "optuna_dbs")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
EVAL_ARTIFACTS_DIR = os.path.join(RESULTS_DIR, "eval_artifacts")


DATA_FILE = "./data/paths_rbergomi_options_100k.npz"

# -----------------------------------------------------------------------------
#   STATIC TRAINING CONSTANTS (unchanged)
# -----------------------------------------------------------------------------
N_ENVS = 2
LSTM_SIZE = 128
N_LSTM_LAYERS = 1
N_STEPS_AGENT = 256
BATCH_SIZE_AGENT = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED_OFFSET = 1000

# Windows multiprocessing fix
USE_SUBPROCESS_VECENV = True  # Set to False on Windows to avoid BrokenPipeError

# You can also set this to True if you want to try SubprocVecEnv on Windows
# USE_SUBPROCESS_VECENV = True

N_TIMESTEPS_PER_TRIAL_HPO = 30_000
N_OPTUNA_TRIALS = 6
N_EVAL_EPISODES_HPO = 5
EVAL_FREQ_HPO_ROLLOUTS = 2

N_TIMESTEPS_FINAL_TRAIN = 5_000_000
CHECKPOINT_FREQ_ROLLOUTS_FINAL = 25
EVAL_FREQ_ROLLOUTS_FINAL = 10
EARLY_STOP_PATIENCE_FINAL = 15

N_EVAL_EPISODES_FINAL = 100

# ─────────────────────────────────────────────────────────────────────────────
#  ↘  R U N - T I M E   C O N F I G  (was: CLI flags)
# ─────────────────────────────────────────────────────────────────────────────
LOSS_TYPE = "abs"       # one of {"mse", "abs", "cvar"}
W         = 0.001       # pnl penalty weight
LAM       = 0.0001      # transaction-cost penalty weight
MODE      = "hpo"       # {"hpo", "final", "eval"}
SEED      = 12345       # base RNG seed
THETA     = 0.0002      # reward weight on θ (time-decay) term
SLIP      = 1.0         # slippage in bps (1 bp = 0.01 %)
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging(log_file_path_full, loss_type_arg, w_arg, lam_arg, mode_arg):
    logger = logging.getLogger()
    logger.handlers.clear()

    formatter = logging.Formatter(f"%(asctime)s [%(levelname)s] TRAIN_RL(loss={loss_type_arg},w={w_arg:g},l={lam_arg:g},m={mode_arg}): %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_path_full, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def linear_schedule(initial_value: float, final_value: float = 0.0):
    if initial_value < final_value:
        raise ValueError("For linear_schedule, initial_value must be >= final_value.")
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule

def create_env_fn(loss_type_env, pnl_w, lam_c, theta_w, slip_bps, seed, record_metrics=True):
    def _init():
        try:
            env = HedgingEnv(
                data_file_path=DATA_FILE,
                loss_type=loss_type_env,
                pnl_penalty_weight=pnl_w,
                lambda_cost=lam_c,
                theta_weight=theta_w,
                slippage_bps=slip_bps,
                record_metrics=record_metrics
            )
            env = Monitor(env, info_keywords=('per_share_step_pnl', 'raw_pnl_deviation_abs', 'transaction_costs_total'))
            env.reset(seed=seed)
            return env
        except Exception as e:
            logging.error(f"Failed to create environment: {e}")
            raise
    return _init

def create_vec_env(loss_type_env, pnl_w, lam_c, theta_w, slip_bps, seed_base, n_envs=N_ENVS, record_metrics=True):
    """Create vectorized environment with Windows compatibility"""
    env_fns = [create_env_fn(loss_type_env, pnl_w, lam_c, theta_w, slip_bps, seed=seed_base + i, record_metrics=record_metrics) for i in range(n_envs)]
    
    if USE_SUBPROCESS_VECENV and os.name != 'nt':  # Only use SubprocVecEnv on non-Windows systems
        try:
            logging.info(f"Creating SubprocVecEnv with {n_envs} environments")
            return SubprocVecEnv(env_fns, start_method='spawn' if os.name == 'nt' else 'fork')
        except Exception as e:
            logging.warning(f"SubprocVecEnv failed, falling back to DummyVecEnv: {e}")
            return DummyVecEnv(env_fns)
    else:
        # Use DummyVecEnv on Windows or when USE_SUBPROCESS_VECENV is False
        logging.info(f"Creating DummyVecEnv with {n_envs} environments (Windows-compatible mode)")
        return DummyVecEnv(env_fns)

class OptunaPruningCallbackForEval(BaseCallback):
    def __init__(self, trial: optuna.Trial, loss_type_objective: str, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.is_pruned = False
        self.loss_type_objective = loss_type_objective
        self.all_episode_returns_for_cvar = []

    def _on_step(self) -> bool:
        if self.is_pruned:
            return False

        if self.parent is None or not hasattr(self.parent, 'last_mean_reward') or self.parent.last_mean_reward is None:
            return True

        current_reward = self.parent.last_mean_reward
        current_step_for_report = self.parent.num_timesteps

        metric_to_report = current_reward

        self.trial.report(metric_to_report, current_step_for_report)
        if self.trial.should_prune():
            self.is_pruned = True
            message = f"Trial {self.trial.number} pruned at step {current_step_for_report} with value {metric_to_report:.4f}."
            logging.info(message)
            raise TrialPruned(message)
        return True

def run_hpo(loss_type, pnl_w, lam_c, run_seed):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = f"hpo_loss{loss_type}_w{pnl_w:g}_l{lam_c:g}"
    optuna_db_path = f"sqlite:///{os.path.join(OPTUNA_DB_DIR, study_name + '.db')}"
    log_path_hpo_study = os.path.join(LOGS_DIR, "hpo", study_name)
    os.makedirs(log_path_hpo_study, exist_ok=True)

    logging.info(f"Starting HPO for loss={loss_type}, w={pnl_w}, lam={lam_c}. Study: {study_name}")

    def objective(trial: optuna.Trial):
        trial_seed = run_seed + trial.number + SEED_OFFSET * 2
        set_random_seed(trial_seed)

        trial_log_path = os.path.join(log_path_hpo_study, f"trial_{trial.number}")
        os.makedirs(trial_log_path, exist_ok=True)

        lr_init = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.93, 0.99, log=True)
        clip_range_init = trial.suggest_float("clip_range", 0.1, 0.4)
        ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-3, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
        vf_coef = trial.suggest_float("vf_coef", 0.3, 0.8)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 1.5)
        n_epochs = trial.suggest_int("n_epochs", 5, 20)
        log_std_init_hpo = trial.suggest_float("log_std_init", -0.5, 2.0)

        train_env = None
        eval_env_hpo = None
        model = None
        mean_reward = -np.inf

        try:
            train_env = create_vec_env(loss_type, pnl_w, lam_c, THETA, SLIP, seed_base=trial_seed)
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, gamma=gamma)

            eval_env_hpo = create_vec_env(loss_type, pnl_w, lam_c, THETA, SLIP, seed_base=trial_seed + N_ENVS + SEED_OFFSET)
            eval_env_hpo = VecNormalize(eval_env_hpo, norm_obs=True, norm_reward=False, training=False, gamma=gamma)
            if hasattr(train_env, 'obs_rms'): eval_env_hpo.obs_rms = train_env.obs_rms

            one_rollout_steps = N_STEPS_AGENT * N_ENVS
            eval_freq_this_trial = max(one_rollout_steps * EVAL_FREQ_HPO_ROLLOUTS, N_TIMESTEPS_PER_TRIAL_HPO // 3)

            optuna_pruning_cb = OptunaPruningCallbackForEval(trial, loss_type)
            eval_callback = EvalCallback(
                eval_env_hpo, best_model_save_path=None, log_path=os.path.join(trial_log_path, "eval"),
                eval_freq=eval_freq_this_trial, n_eval_episodes=N_EVAL_EPISODES_HPO, deterministic=True,
                warn=False, callback_on_new_best=None, callback_after_eval=optuna_pruning_cb
            )

            policy_kwargs_hpo=dict(lstm_hidden_size=LSTM_SIZE, n_lstm_layers=N_LSTM_LAYERS, log_std_init=log_std_init_hpo)

            model = RecurrentPPO(
                "MlpLstmPolicy", train_env, learning_rate=lr_init, n_steps=N_STEPS_AGENT, batch_size=BATCH_SIZE_AGENT,
                n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range_init, ent_coef=ent_coef,
                vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                policy_kwargs=policy_kwargs_hpo,
                device=DEVICE, seed=trial_seed, verbose=0
            )

            model.learn(total_timesteps=N_TIMESTEPS_PER_TRIAL_HPO, callback=eval_callback, progress_bar=False)

            if hasattr(eval_callback, 'last_mean_reward') and eval_callback.last_mean_reward is not None:
                mean_reward = eval_callback.last_mean_reward
            elif trial.last_step is not None and len(trial.intermediate_values) > 0:
                mean_reward = trial.intermediate_values[trial.last_step]

        except TrialPruned:
            raise
        except (AssertionError, ValueError, RuntimeError, OSError, EOFError, BrokenPipeError) as e:
            logging.warning(f"Trial {trial.number} failed with error: {e}")
            mean_reward = -np.inf
        finally:
            if model is not None: del model
            if train_env is not None: train_env.close(); del train_env
            if eval_env_hpo is not None: eval_env_hpo.close(); del eval_env_hpo
        return mean_reward

    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2, interval_steps=1)
    study = optuna.create_study(study_name=study_name, storage=optuna_db_path, load_if_exists=True, direction="maximize", pruner=pruner)

    try:
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, n_jobs=1)
    except Exception as e:
        logging.error(f"Optuna study optimization failed: {e}", exc_info=True)

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    best_params_to_return = None
    try:
        best_trial_obj = study.best_trial       # ← may throw ValueError
    except ValueError:
        best_trial_obj = None

    if best_trial_obj is not None:
        logging.info(
            f"HPO for loss={loss_type}, w={pnl_w}, lam={lam_c} completed. "
            f"Best trial: {best_trial_obj.number} with value {best_trial_obj.value:.4f}"
        )
        best_params_to_return = best_trial_obj.params

        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig_path = os.path.join(log_path_hpo_study, "hpo_importances.html")
            fig.write_html(fig_path)
            logging.info(f"Optuna param importances plot saved to {fig_path}")
        except Exception as e:
            logging.warning(f"Failed to generate or save Optuna importance plot: {e}")
    else:
        logging.warning(f"HPO for loss={loss_type}, w={pnl_w}, lam={lam_c} completed but no best trial found.")

    return best_params_to_return


def run_final_training(loss_type, pnl_w, lam_c, best_hpo_params, run_seed):
    final_train_seed = run_seed + SEED_OFFSET * 3
    set_random_seed(final_train_seed)

    run_identifier_base = f"loss{loss_type}_w{pnl_w:g}_l{lam_c:g}_v2"
    run_identifier = run_identifier_base
    if loss_type == "cvar":
        run_identifier = f"{run_identifier_base}_CVARPlaceholder"


    model_save_dir = os.path.join(MODELS_DIR, run_identifier)
    log_path_final = os.path.join(LOGS_DIR, "final", run_identifier)
    tb_log_path_final = os.path.join(LOGS_DIR, "tb_final", run_identifier)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_path_final, exist_ok=True)
    os.makedirs(tb_log_path_final, exist_ok=True)

    logging.info(f"Starting final training for {run_identifier} with seed {final_train_seed}")
    logging.info(f"Using HPO params: {best_hpo_params}")

    train_env_final_venv_initial = create_vec_env(loss_type, pnl_w, lam_c, THETA, SLIP, seed_base=final_train_seed)
    train_env = VecNormalize(train_env_final_venv_initial, norm_obs=True, norm_reward=True, gamma=best_hpo_params["gamma"])

    eval_env_final_venv = create_vec_env(loss_type, pnl_w, lam_c, THETA, SLIP, seed_base=final_train_seed + N_ENVS + SEED_OFFSET)
    eval_env_final = VecNormalize(eval_env_final_venv, norm_obs=True, norm_reward=False, training=False, gamma=best_hpo_params["gamma"])
    if hasattr(train_env, 'obs_rms'): eval_env_final.obs_rms = train_env.obs_rms

    one_rollout_steps = N_STEPS_AGENT * N_ENVS
    eval_freq_this_final = max(one_rollout_steps, one_rollout_steps * EVAL_FREQ_ROLLOUTS_FINAL)
    checkpoint_save_freq_val = max(one_rollout_steps, one_rollout_steps * CHECKPOINT_FREQ_ROLLOUTS_FINAL)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_save_freq_val, save_path=model_save_dir, name_prefix="rl_model", save_vecnormalize=True
    )
    early_stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=EARLY_STOP_PATIENCE_FINAL, min_evals=EARLY_STOP_PATIENCE_FINAL + 2, verbose=1
    )
    eval_callback_final = EvalCallback(
        eval_env_final, best_model_save_path=os.path.join(model_save_dir, "best_model"),
        log_path=log_path_final, eval_freq=eval_freq_this_final, n_eval_episodes=N_EVAL_EPISODES_HPO * 2,
        deterministic=True, warn=False, callback_on_new_best=None, callback_after_eval=early_stop_callback
    )

    model = None
    model_loaded_from_checkpoint_successfully = False
    steps_done = 0

    checkpoints = glob.glob(os.path.join(model_save_dir, "rl_model_*_steps.*zip"))
    latest_checkpoint_model_path = None
    if checkpoints:
        checkpoint_steps_list = []
        for cp_path in checkpoints:
            match = re.search(r"rl_model_(\d+)_steps.*\.zip", os.path.basename(cp_path))
            if match: checkpoint_steps_list.append((int(match.group(1)), cp_path))
        if checkpoint_steps_list:
            checkpoint_steps_list.sort(key=lambda x: x[0], reverse=True)
            steps_done, latest_checkpoint_model_path = checkpoint_steps_list[0]

    if latest_checkpoint_model_path and steps_done < N_TIMESTEPS_FINAL_TRAIN:
        expected_vecnorm_path = os.path.join(model_save_dir, f"rl_model_vecnormalize_{steps_done}_steps.pkl")
        if os.path.exists(expected_vecnorm_path):
            logging.info(f"Attempting to resume final training from checkpoint: {latest_checkpoint_model_path} at {steps_done} steps.")

            train_env.close()

            resuming_venv = create_vec_env(loss_type, pnl_w, lam_c, THETA, SLIP, seed_base=final_train_seed + steps_done)
            train_env = VecNormalize.load(expected_vecnorm_path, resuming_venv)
            train_env.training = True
            if hasattr(train_env, 'gamma') and train_env.gamma != best_hpo_params["gamma"]:
                setattr(train_env, 'gamma', best_hpo_params["gamma"])

            model = RecurrentPPO.load(latest_checkpoint_model_path, env=train_env, device=DEVICE)
            model.set_logger(configure_logger(verbose=1, tensorboard_log=tb_log_path_final, tb_log_name="RecurrentPPO"))

            model_loaded_from_checkpoint_successfully = True
            logging.info(f"Resumed model and VecNormalize from {steps_done} steps. Current model timesteps: {model.num_timesteps}")
        else:
            logging.warning(f"VecNormalize for checkpoint {expected_vecnorm_path} not found! Starting training from scratch.")

    if not model_loaded_from_checkpoint_successfully:
        steps_done = 0
        logging.info("No suitable checkpoint found or VecNormalize missing. Starting final training from scratch.")
        lr_val = best_hpo_params["lr"]
        lr_schedule_fn = linear_schedule(lr_val, lr_val * 0.1)

        clip_range_val = best_hpo_params.get("clip_range", 0.3)
        log_std_init_val = best_hpo_params.get("log_std_init", 1.5)

        policy_kwargs_final = dict(lstm_hidden_size=LSTM_SIZE, n_lstm_layers=N_LSTM_LAYERS, log_std_init=log_std_init_val)

        model = RecurrentPPO(
            "MlpLstmPolicy", train_env, learning_rate=lr_schedule_fn, n_steps=N_STEPS_AGENT, batch_size=BATCH_SIZE_AGENT,
            n_epochs=best_hpo_params["n_epochs"], gamma=best_hpo_params["gamma"], gae_lambda=best_hpo_params["gae_lambda"],
            clip_range=clip_range_val, ent_coef=best_hpo_params["ent_coef"], vf_coef=best_hpo_params["vf_coef"],
            max_grad_norm=best_hpo_params["max_grad_norm"],
            policy_kwargs=policy_kwargs_final,
            device=DEVICE, seed=final_train_seed, verbose=1, tensorboard_log=tb_log_path_final
        )

    try:
        current_model_timesteps = model.num_timesteps if model_loaded_from_checkpoint_successfully else 0

        if current_model_timesteps < N_TIMESTEPS_FINAL_TRAIN:
            reset_num_timesteps_flag = not model_loaded_from_checkpoint_successfully
            if model_loaded_from_checkpoint_successfully:
                logging.info(f"Resuming training. Model has {current_model_timesteps} steps. Will train until {N_TIMESTEPS_FINAL_TRAIN} total steps.")

            model.learn(total_timesteps=N_TIMESTEPS_FINAL_TRAIN, callback=[checkpoint_callback, eval_callback_final], reset_num_timesteps=reset_num_timesteps_flag, progress_bar=False)
        else:
            logging.info(f"Training target of {N_TIMESTEPS_FINAL_TRAIN} steps already met or exceeded by checkpoint ({current_model_timesteps} steps). Skipping .learn().")

    except Exception as e:
        logging.error(f"Final training for {run_identifier} failed: {e}", exc_info=True)
    finally:
        if model is not None:
            model_path = os.path.join(model_save_dir, "final_model.zip")
            vecnorm_path = os.path.join(model_save_dir, "final_vecnormalize.pkl")
            model.save(model_path)
            if hasattr(train_env, 'save'): train_env.save(vecnorm_path)
            logging.info(f"Final model saved to {model_path}, VecNormalize stats to {vecnorm_path}")

        if hasattr(train_env, 'close'): train_env.close()
        if hasattr(eval_env_final, 'close'): eval_env_final.close()

        if model is not None: del model
        if 'train_env' in locals() and train_env is not None: del train_env
        if 'eval_env_final' in locals() and eval_env_final is not None: del eval_env_final

        if torch.cuda.is_available(): torch.cuda.empty_cache()


def run_evaluation(loss_type, pnl_w, lam_c, run_seed):
    eval_seed_env = run_seed + SEED_OFFSET * 4
    set_random_seed(eval_seed_env)

    model_train_seed = run_seed + SEED_OFFSET * 3

    run_identifier_base = f"loss{loss_type}_w{pnl_w:g}_l{lam_c:g}"
    run_identifier = run_identifier_base
    if loss_type == "cvar":
        run_identifier = f"{run_identifier_base}_CVARPlaceholder"

    model_dir = os.path.join(MODELS_DIR, run_identifier)
    artifact_dir = os.path.join(EVAL_ARTIFACTS_DIR, run_identifier)
    os.makedirs(artifact_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.zip")
        if not os.path.exists(model_path):
            logging.error(f"No model found for {run_identifier} at {model_dir}")
            return

    vecnorm_path = os.path.join(model_dir, "final_vecnormalize.pkl")
    if not os.path.exists(vecnorm_path):
        logging.error(f"No VecNormalize stats found for {run_identifier} at {vecnorm_path}")
        return

    logging.info(f"Starting evaluation for {run_identifier} with env seed base {eval_seed_env} using model {model_path}")

    eval_env_vec = None
    eval_env = None
    model_eval = None

    try:
        eval_env_vec = create_vec_env(loss_type, pnl_w, lam_c, THETA, SLIP, seed_base=eval_seed_env, record_metrics=True)
        eval_env = VecNormalize.load(vecnorm_path, eval_env_vec)
        eval_env.training = False
        eval_env.norm_reward = False
        eval_env.norm_obs = True

        model_eval = RecurrentPPO.load(model_path, env=eval_env, device=DEVICE)

        all_episode_total_per_share_pnls = []
        all_episode_total_costs = []
        all_episode_total_rewards = []
        action_log_all_steps = []

        num_episodes_collected = 0
        env_episode_lengths = eval_env.get_attr("episode_length")[0]

        while num_episodes_collected < N_EVAL_EPISODES_FINAL:
            obs = eval_env.reset()
            states = None
            episode_starts = np.ones((N_ENVS,), dtype=bool)

            current_rollout_ep_pnl_sum = np.zeros(N_ENVS)
            current_rollout_ep_cost_sum = np.zeros(N_ENVS)
            current_rollout_ep_reward_sum = np.zeros(N_ENVS)
            current_rollout_ep_steps = np.zeros(N_ENVS, dtype=int)

            for _step_num in range(env_episode_lengths + 5):
                if num_episodes_collected >= N_EVAL_EPISODES_FINAL: break

                action_pred, states = model_eval.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
                new_obs, rewards, dones, infos = eval_env.step(action_pred)

                for i in range(N_ENVS):
                    current_rollout_ep_pnl_sum[i] += infos[i].get('per_share_step_pnl', 0.0)
                    current_rollout_ep_cost_sum[i] += infos[i].get('transaction_costs_total', 0.0)
                    current_rollout_ep_reward_sum[i] += rewards[i]
                    current_rollout_ep_steps[i] += 1

                    action_details = {
                        'episode_idx_global_approx': num_episodes_collected + i,
                        'step_in_episode': current_rollout_ep_steps[i],
                        'raw_action_call': float(action_pred[i,0]) if action_pred.ndim == 2 else float(action_pred[0]),
                        'raw_action_put': float(action_pred[i,1]) if action_pred.ndim == 2 else float(action_pred[1]),
                        'scaled_float_call': float(infos[i].get('scaled_float_call', 0.0)),
                        'scaled_float_put': float(infos[i].get('scaled_float_put', 0.0)),
                        'rounded_clipped_call': int(infos[i].get('requested_calls_rounded_clipped', 0)),
                        'rounded_clipped_put': int(infos[i].get('requested_puts_rounded_clipped', 0)),
                        'realized_call_trade': int(infos[i].get('actual_calls_traded', 0)),
                        'realized_put_trade': int(infos[i].get('actual_puts_traded', 0)),
                        'pnl_dev_step_abs': float(infos[i].get('raw_pnl_deviation_abs', 0.0)),
                        'cost_step': float(infos[i].get('transaction_costs_total', 0.0)),
                        'reward_step': float(rewards[i])
                    }
                    action_log_all_steps.append(action_details)

                    if dones[i]:
                        if num_episodes_collected < N_EVAL_EPISODES_FINAL:
                            all_episode_total_per_share_pnls.append(current_rollout_ep_pnl_sum[i])
                            all_episode_total_costs.append(current_rollout_ep_cost_sum[i])
                            all_episode_total_rewards.append(current_rollout_ep_reward_sum[i])
                            num_episodes_collected += 1

                        current_rollout_ep_pnl_sum[i] = 0.0
                        current_rollout_ep_cost_sum[i] = 0.0
                        current_rollout_ep_reward_sum[i] = 0.0
                        current_rollout_ep_steps[i] = 0

                obs = new_obs
                episode_starts = dones
            if num_episodes_collected >= N_EVAL_EPISODES_FINAL: break

        ep_mean_abs_pnl_per_step = [np.abs(pnl_sum)/env_episode_lengths for pnl_sum in all_episode_total_per_share_pnls[:N_EVAL_EPISODES_FINAL]]
        ep_mean_costs_per_step = [cost_sum/env_episode_lengths for cost_sum in all_episode_total_costs[:N_EVAL_EPISODES_FINAL]]

        mean_abs_pnl_val = np.mean(ep_mean_abs_pnl_per_step) if len(ep_mean_abs_pnl_per_step) > 0 else 0
        mean_cost_val = np.mean(ep_mean_costs_per_step) if len(ep_mean_costs_per_step) > 0 else 0
        std_abs_pnl_val = np.std(ep_mean_abs_pnl_per_step) if len(ep_mean_abs_pnl_per_step) > 0 else 0

        cvar95_abs_pnl = 0.0
        if len(ep_mean_abs_pnl_per_step) > 0: # CVaR of episode-average absolute per-step PnL deviations
            sorted_abs_pnls = sorted(ep_mean_abs_pnl_per_step)
            cvar95_abs_pnl = np.mean(sorted_abs_pnls[int(0.95 * len(sorted_abs_pnls)):])


        logging.info(f"Evaluation results for {run_identifier} over {len(ep_mean_abs_pnl_per_step)} episodes:")
        logging.info(f"  Mean |ΔPnL| per share per step: {mean_abs_pnl_val:.4f}")
        logging.info(f"  Mean Transaction Cost per step: {mean_cost_val:.4f}")
        logging.info(f"  Std |ΔPnL| per share per step: {std_abs_pnl_val:.4f}")
        logging.info(f"  CVaR95 of Mean |ΔPnL|: {cvar95_abs_pnl:.4f}")

        algo_name_log = f"rl_{run_identifier}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        df_new_row = pd.DataFrame([{
            "algo": algo_name_log, "loss_type": loss_type, "w": pnl_w, "lam": lam_c,
            "mean_abs_pnl": mean_abs_pnl_val, "mean_cost": mean_cost_val,
            "std_abs_pnl": std_abs_pnl_val, "seed": model_train_seed,
            "timesteps": N_TIMESTEPS_FINAL_TRAIN, "episodes_eval": len(ep_mean_abs_pnl_per_step),
            "cvar95_abs_pnl": cvar95_abs_pnl,
            "status": "eval_done", "timestamp": timestamp
        }])
        header_exists = os.path.isfile(PARETO_RAW_CSV)
        df_new_row.to_csv(PARETO_RAW_CSV, mode='a', header=not header_exists, index=False)
        logging.info(f"Results appended to {PARETO_RAW_CSV}")

        np.savez(os.path.join(artifact_dir, f"episode_stats_seed{eval_seed_env}.npz"),
                 episode_total_signed_pnls=np.array(all_episode_total_per_share_pnls[:N_EVAL_EPISODES_FINAL]),
                 episode_total_costs=np.array(all_episode_total_costs[:N_EVAL_EPISODES_FINAL]),
                 episode_total_rewards=np.array(all_episode_total_rewards[:N_EVAL_EPISODES_FINAL]))
        actions_df = pd.DataFrame(action_log_all_steps)
        actions_df.to_parquet(os.path.join(artifact_dir, f"actions_seed{eval_seed_env}.parquet"))

        frontier_dict = {"loss_type":loss_type, "w":pnl_w, "lam":lam_c,
                         "mean_abs_pnl":mean_abs_pnl_val, "cvar95_abs_pnl":cvar95_abs_pnl,
                         "mean_cost":mean_cost_val, "std_abs_pnl":std_abs_pnl_val,
                         "seed":model_train_seed, "timesteps":N_TIMESTEPS_FINAL_TRAIN}
        with open(os.path.join(artifact_dir, "frontier_point.json"), "w") as fp:
            json.dump(frontier_dict, fp, indent=4)
        logging.info(f"Evaluation artifacts saved to {artifact_dir}")

    except Exception as e:
        logging.error(f"Evaluation for {run_identifier} failed: {e}", exc_info=True)
    finally:
        if model_eval is not None: del model_eval
        if eval_env is not None: eval_env.close(); del eval_env
        if eval_env_vec is not None: eval_env_vec.close(); del eval_env_vec
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def main():
    # Windows multiprocessing fix
    if os.name == 'nt':  # Windows
        mp.set_start_method('spawn', force=True)
    
    os.makedirs(OPTUNA_DB_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(EVAL_ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOGS_DIR, "hpo"), exist_ok=True)
    os.makedirs(os.path.join(LOGS_DIR, "final"), exist_ok=True)
    os.makedirs(os.path.join(LOGS_DIR, "tb_final"), exist_ok=True)

    log_file_path_full = os.path.join(LOGS_DIR, f"train_rl_loss{LOSS_TYPE}_w{W:g}_l{LAM:g}_{MODE}.log")
    setup_logging(log_file_path_full, LOSS_TYPE, W, LAM, MODE)

    logging.info(f"RUN-CONFIG ⇒ loss_type={LOSS_TYPE}, w={W}, lam={LAM}, "
                 f"mode={MODE}, seed={SEED}, theta={THETA}, slip={SLIP}")
    logging.info(f"Using PyTorch Device: {DEVICE}")
    logging.info(f"Vectorized Environment: {'DummyVecEnv (Windows-compatible)' if not USE_SUBPROCESS_VECENV or os.name == 'nt' else 'SubprocVecEnv'}")
    logging.info(f"Operating System: {os.name} ({'Windows' if os.name == 'nt' else 'Unix-like'})")

    default_hpo_params = {
        "lr": 1e-4, "gamma": 0.99, "clip_range": 0.3, "ent_coef": 1e-5,
        "gae_lambda": 0.95, "vf_coef": 0.5, "max_grad_norm": 0.5, "n_epochs": 10,
        "log_std_init": 1.5
    }

    if MODE == "hpo":
        hpo_results = run_hpo(LOSS_TYPE, W, LAM, SEED)
        if hpo_results:
            logging.info(f"HPO successful, best params for loss={LOSS_TYPE}, w={W}, lam={LAM}: {hpo_results}")
    elif MODE == "final":
        study_name = f"hpo_loss{LOSS_TYPE}_w{W:g}_l{LAM:g}"
        optuna_db_path = f"sqlite:///{os.path.join(OPTUNA_DB_DIR, study_name + '.db')}"
        final_hpo_params = default_hpo_params.copy()
        try:
            study = optuna.load_study(study_name=study_name, storage=optuna_db_path)
            try:
                best_trial_obj = study.best_trial     # ← can raise ValueError
            except ValueError:
                best_trial_obj = None

            if best_trial_obj is not None:
                final_hpo_params.update(best_trial_obj.params)
                logging.info(
                    f"Loaded best HPO params for loss={LOSS_TYPE}, w={W}, lam={LAM}: "
                    f"{best_trial_obj.params}"
                )
            else:
                logging.warning(f"HPO study {study_name} loaded but no best trial found. Using default params for final training.")
        except Exception as e:
            logging.error(f"Could not load HPO study {study_name}. Using default params for final training. Error: {e}")

        run_final_training(LOSS_TYPE, W, LAM, final_hpo_params, SEED)
    elif MODE == "eval":
        run_evaluation(LOSS_TYPE, W, LAM, SEED)

if __name__ == "__main__":
    main()