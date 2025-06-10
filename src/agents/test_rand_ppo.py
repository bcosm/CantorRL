# test_rand_ppo.py
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from src.env.hedging_env import HedgingEnv    # Your custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
# test_norm_ppo.py
import numpy as np
from src.env.hedging_env import HedgingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Constants (same paths & params as before)
DATA_FILE        = "data/paths_rbergomi_options_100k.npz"
TRANSACTION_COST = 0.05
LAMBDA_COST      = 1.0
SHARES           = 10_000
MAX_CONTRACTS    = 200

# 1) Build raw env and wrap it exactly like in HPO
raw_env = HedgingEnv(DATA_FILE, TRANSACTION_COST,
                     LAMBDA_COST, 0.0, SHARES, MAX_CONTRACTS)
vec_env = DummyVecEnv([lambda: raw_env])
norm_env = VecNormalize(vec_env, norm_obs=True,
                        norm_reward=False, gamma=0.99)
norm_env.training   = False
norm_env.norm_reward= False

# 2) Instantiate the same untrained PPO
model = PPO("MlpPolicy", vec_env, verbose=0)

# 3) Evaluate on the normalized env
mean_norm, std_norm = evaluate_policy(
    model,
    norm_env,
    n_eval_episodes=10,
    deterministic=True,
    warn=False
)
print(f"With VecNormalize → mean_reward = {mean_norm:.3f} ± {std_norm:.3f}")

