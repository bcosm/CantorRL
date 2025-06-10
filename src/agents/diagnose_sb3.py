# diagnose_sb3.py
import sys
import os
import importlib

print("--- SB3 Import Diagnostics ---")

# 1. Python Environment Details
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version.splitlines()[0]}")
print("\nCurrent sys.path:")
for i, p in enumerate(sys.path):
    print(f"  {i}: {p}")

# 2. Check for local 'stable_baselines3' or 'common' directories that might cause shadowing
project_root_dir_to_check = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# If diagnose_sb3.py is in src/agents/, then project_root_dir_to_check needs to be adjusted.
# Assuming diagnose_sb3.py is at C:\Users\bcosm\PycharmProjects\CantorRL\CantorRL\src\agents\diagnose_sb3.py
# then project_root_dir for CantorRL project is:
project_root_actual = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # This should be C:\Users\bcosm\PycharmProjects\CantorRL
print(f"\nChecking for conflicting directories in project root: {project_root_actual}")

conflicting_paths = []
potential_conflicts = ["stable_baselines3", "common"]
for pc in potential_conflicts:
    local_path = os.path.join(project_root_actual, pc)
    if os.path.exists(local_path):
        conflicting_paths.append(f"WARNING: Found local directory/file that might conflict: {local_path}")

if not conflicting_paths:
    print("No obvious conflicting local directories ('stable_baselines3', 'common') found in project root.")
else:
    for cp in conflicting_paths:
        print(cp)

# 3. Attempt to import stable_baselines3 and check its path
print("\nAttempting to import 'stable_baselines3'...")
try:
    import stable_baselines3 as sb3
    print(f"Successfully imported 'stable_baselines3'. Version: {sb3.__version__}")
    sb3_path = os.path.dirname(sb3.__file__)
    print(f"  'stable_baselines3' package is loaded from: {sb3_path}")
    if ".venv\\lib\\site-packages\\stable_baselines3" not in sb3_path and ".venv/lib/site-packages/stable_baselines3" not in sb3_path :
        print("  WARNING: 'stable_baselines3' is NOT being loaded from the expected venv site-packages path!")
except ImportError as e:
    print(f"FAILED to import 'stable_baselines3': {e}")
    print("--- Diagnostics Aborted ---")
    sys.exit(1)
except Exception as e:
    print(f"An UNEXPECTED error occurred during 'stable_baselines3' import: {e}")
    print("--- Diagnostics Aborted ---")
    sys.exit(1)

# 4. Attempt to import MlpLstmPolicy and check policies module path
print("\nAttempting to import 'MlpLstmPolicy' from 'stable_baselines3.common.policies'...")
try:
    from stable_baselines3.common import policies
    policies_module_path = os.path.dirname(policies.__file__)
    print(f"  'stable_basaines3.common.policies' module is loaded from: {policies_module_path}")
    if ".venv\\lib\\site-packages\\stable_baselines3\\common" not in policies_module_path and ".venv/lib/site-packages/stable_baselines3/common" not in policies_module_path:
        print("  WARNING: 'stable_baselines3.common.policies' is NOT being loaded from the expected venv site-packages path!")

    from stable_baselines3.common.policies import MlpLstmPolicy
    print(f"Successfully imported 'MlpLstmPolicy': {MlpLstmPolicy}")
    print("  Direct import of 'MlpLstmPolicy' SUCCEEDED.")
except ImportError as e:
    print(f"FAILED to import 'MlpLstmPolicy' or 'stable_baselines3.common.policies': {e}")
    print("  This ImportError is the most likely direct cause of the 'Policy unknown' error in PPO.")
except Exception as e:
    print(f"An UNEXPECTED error occurred during 'MlpLstmPolicy' import: {e}")

# 5. Check how SB3's PPO resolves the policy string (simulated)
# This part is a bit more involved as it mimics internal SB3 logic.
if 'sb3' in locals(): # Only if SB3 was imported
    print("\nSimulating SB3 policy name resolution for 'MlpLstmPolicy':")
    policy_name_to_check = "MlpLstmPolicy"
    resolved_class = None
    try:
        # Attempt 1: Check policy_aliases (would be empty in BaseAlgorithm's direct context)
        # We can't easily get PPO's policy_aliases without instantiating it.
        # Attempt 2: Check SB3_REGISTRY (usually for full paths)
        # from stable_baselines3.common.utils import SB3_REGISTRY
        # if policy_name_to_check in SB3_REGISTRY:
        #    print(f"Found '{policy_name_to_check}' in SB3_REGISTRY (this is unusual for short names).")
        # else:
        # Attempt 3: Load from stable_baselines3.common.policies by name (most likely path for short aliases)
        common_policies_module = importlib.import_module("stable_baselines3.common.policies")
        if hasattr(common_policies_module, policy_name_to_check):
            resolved_class = getattr(common_policies_module, policy_name_to_check)
            print(f"  Successfully resolved '{policy_name_to_check}' using importlib from 'stable_baselines3.common.policies' to: {resolved_class}")
        else:
            print(f"  FAILED to find '{policy_name_to_check}' as an attribute in 'stable_baselines3.common.policies' module.")
    except ImportError:
        print(f"  FAILED to import 'stable_baselines3.common.policies' module via importlib.")
    except Exception as e:
        print(f"  An UNEXPECTED error occurred during simulated resolution: {e}")

print("\n--- SB3 Import Diagnostics Finished ---")