import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from src.env.hedging_env import HedgingEnv    # Your custom environment

# 1) Define your constants here so Python knows about them:
DATA_FILE           = os.path.join(os.path.dirname(__file__),
                                  "..", "..", "data",
                                  "paths_rbergomi_options_100k.npz")
TRANSACTION_COST    = 0.05
LAMBDA_COST         = 1.0
SHARES              = 10_000
MAX_CONTRACTS       = 200

# 2) Now you can safely instantiate the environment:
env = HedgingEnv(
    data_file_path                = DATA_FILE,
    transaction_cost_per_contract = TRANSACTION_COST,
    lambda_cost                   = LAMBDA_COST,
    initial_cash                  = 0.0,
    shares_to_hedge               = SHARES,
    max_contracts_held_per_type   = MAX_CONTRACTS
)

# 3) Test it:
obs, info = env.reset()
print("Initial obs:", obs)     # Should no longer NameError
print("Info dict:", info)
