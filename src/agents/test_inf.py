# test_inf.py
import numpy as np
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from src.env.hedging_env import HedgingEnv    # Your custom environment

# Re-use the same constants you just defined in test_env.py:
DATA_FILE           = "data/paths_rbergomi_options_100k.npz"
TRANSACTION_COST    = 0.05
LAMBDA_COST         = 1.0
SHARES              = 10_000
MAX_CONTRACTS       = 200

# Create the env
env = HedgingEnv(
    data_file_path                = DATA_FILE,
    transaction_cost_per_contract = TRANSACTION_COST,
    lambda_cost                   = LAMBDA_COST,
    shares_to_hedge               = SHARES,
    max_contracts_held_per_type   = MAX_CONTRACTS
)

obs, _ = env.reset()
for step in range(10_000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    # Spot non-finite rewards immediately:
    if not np.isfinite(reward):
        print(f"🚨 Step {step}: got reward = {reward!r}")
        print("Info:", info)
        break
    if done:
        obs, _ = env.reset()
else:
    print("✅  No infinities in 10k random steps.")
