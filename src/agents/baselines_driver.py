import subprocess
import os
import pandas as pd
import logging
import sys
import time
import yaml
from tqdm import tqdm

_project_root_baselines_driver = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINES_SCRIPT_PATH = os.path.join(_project_root_baselines_driver, "agents", "baselines.py") 
GRID_YAML_PATH_FOR_BASELINES = os.path.join(_project_root_baselines_driver, "agents", "grid.yaml") 
RESULTS_DIR_BASELINES_DRIVER = os.path.join(_project_root_baselines_driver, "results")
BASELINES_DRIVER_LOG_FILE = os.path.join(RESULTS_DIR_BASELINES_DRIVER, "baselines_driver.log")

BASELINE_ALGORITHMS = ["no_hedge", "delta_every_step"] 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] BASELINES_DRIVER: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASELINES_DRIVER_LOG_FILE, mode='a')
    ]
)

def load_grid(yaml_path):
    with open(yaml_path, 'r') as f:
        grid_config = yaml.safe_load(f)
    return grid_config.get('w', []), grid_config.get('lam', [])

def get_completed_baseline_wl_pairs(csv_path, target_algo_name):
    completed_wl_pairs = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and 'algo' in df.columns and 'w' in df.columns and 'lam' in df.columns and 'status' in df.columns:
                for _, row in df.iterrows():
                    if row['algo'] == target_algo_name and row['status'] == 'eval_done':
                        completed_wl_pairs.add((float(row['w']), float(row['lam'])))
        except pd.errors.EmptyDataError:
            logging.info(f"Baseline CSV {csv_path} for {target_algo_name} is empty.")
        except Exception as e:
            logging.error(f"Error reading baseline CSV {csv_path} for {target_algo_name}: {e}")
    return completed_wl_pairs

def run_command(command_list):
    logging.info(f"Executing: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            logging.info(f"  BASELINES_SUBPROCESS: {line.decode().strip()}")
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            logging.error(f"Command failed with exit code {return_code}: {' '.join(command_list)}")
            return False
    except Exception as e:
        logging.error(f"Exception during command execution: {e}")
        return False
    return True

def main():
    os.makedirs(RESULTS_DIR_BASELINES_DRIVER, exist_ok=True)
    logging.info("Starting Baselines Driver (iterating w, lam grid for each baseline).")

    if not os.path.exists(BASELINES_SCRIPT_PATH):
        logging.error(f"Baselines script not found: {BASELINES_SCRIPT_PATH}")
        return
    if not os.path.exists(GRID_YAML_PATH_FOR_BASELINES):
        logging.error(f"Grid configuration file not found: {GRID_YAML_PATH_FOR_BASELINES}")
        return

    weights_pnl, weights_lam = load_grid(GRID_YAML_PATH_FOR_BASELINES)
    if not weights_pnl or not weights_lam:
        logging.error("Grid weights for 'w' or 'lam' are empty in grid.yaml.")
        return

    driver_base_seed = int(time.time()) % 10000 

    for algo_name in tqdm(BASELINE_ALGORITHMS, desc="Baseline Algos", ncols=100, position=0):
        output_csv_name = f"baseline_{algo_name}_results.csv"
        output_csv_path = os.path.join(RESULTS_DIR_BASELINES_DRIVER, output_csv_name)
        logging.info(f"Processing baseline: {algo_name}. Output will be in: {output_csv_path}")
        
        completed_wl_for_this_baseline = get_completed_baseline_wl_pairs(output_csv_path, algo_name)
        logging.info(f"Found {len(completed_wl_for_this_baseline)} (w,lam) pairs already evaluated for {algo_name} in {output_csv_path}.")

        for i_w, w_val_in in enumerate(tqdm(weights_pnl, desc=f"w-grid ({algo_name})", ncols=100, position=1, leave=False)):
            for i_l, lam_val_in in enumerate(tqdm(weights_lam, desc=f"lam-grid (w={w_val_in}, {algo_name})", ncols=100, position=2, leave=False)):
                w_val = float(w_val_in)
                lam_val = float(lam_val_in)
                
                current_wl_pair_tuple = (w_val, lam_val)
                if current_wl_pair_tuple in completed_wl_for_this_baseline:
                    logging.info(f"Skipping (w={w_val}, lam={lam_val}) for baseline {algo_name} as it's already in {output_csv_path}.")
                    continue

                baseline_run_seed = driver_base_seed + sum(ord(c) for c in algo_name) + (i_w * len(weights_lam) + i_l) * 10

                logging.info(f"Running baseline {algo_name} for w_rl={w_val}, lam_rl={lam_val} with seed {baseline_run_seed}")

                cmd_baseline = [
                    sys.executable, BASELINES_SCRIPT_PATH,
                    "--algo_name", algo_name,
                    "--output_csv", output_csv_path,
                    "--w_log", str(w_val),      
                    "--lam_log", str(lam_val),    
                    "--seed", str(baseline_run_seed)
                ]
                
                if run_command(cmd_baseline):
                    logging.info(f"Baseline {algo_name} (w_rl={w_val}, lam_rl={lam_val}) executed. Results in {output_csv_path}")
                else:
                    logging.error(f"Baseline {algo_name} (w_rl={w_val}, lam_rl={lam_val}) execution failed.")
                
                time.sleep(1) 
        logging.info(f"Finished all (w,lam) pairs for baseline: {algo_name}")

    logging.info("Baselines Driver finished.")

if __name__ == "__main__":
    main()