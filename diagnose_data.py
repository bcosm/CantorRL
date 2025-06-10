import numpy as np

data_file_path = './data/paths_rbergomi_options_100k.npz'
data = np.load(data_file_path)
print("Checking data magnitudes:")
magnitude_threshold_for_concern = 1e10 # Example

for key in data.keys():
    arr = data[key].astype(np.float64) # Use float64 to see full precision of stored values

    print(f"\n--- Array: {key} (dtype: {data[key].dtype}, shape: {arr.shape}) ---")
    abs_arr = np.abs(arr)
    finite_abs_arr = abs_arr[np.isfinite(abs_arr)] # Consider only finite values for min/max magnitude

    if finite_abs_arr.size > 0:
        print(f"  Min Abs Value (finite): {np.min(finite_abs_arr)}")
        print(f"  Max Abs Value (finite): {np.max(finite_abs_arr)}")
        if np.max(finite_abs_arr) > magnitude_threshold_for_concern:
            print(f"  WARNING: Max magnitude {np.max(finite_abs_arr)} exceeds threshold {magnitude_threshold_for_concern}")
    else:
        print("  No finite values to compute min/max absolute magnitude.")

    # Re-check for inf/nan just in case
    if np.any(np.isinf(data[key])): # Check original array before astype
        print(f"  STRICT WARNING: Original array '{key}' contains inf values!")
    if np.any(np.isnan(data[key])):
        print(f"  STRICT WARNING: Original array '{key}' contains NaN values!")