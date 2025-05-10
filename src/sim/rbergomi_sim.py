import time
import random
import cupy as cp
import numpy as np
from tqdm import tqdm

INPUT_FILE = './data/historical_prices.csv'
OUTPUT_FILE = './data/paths_rbergomi_options.npz'
R = 0.04
DT = 1/252
N_PATHS = 500
N_STEPS = 252
SEED = 42

T_OPTION_TENOR = 30/252
N_PATHS_OPTION_MC = 2000

XI_DEFAULT = 0.04
H_DEFAULT = 0.1
ETA_DEFAULT = 1.0
RHO_DEFAULT = -0.7
S0_DEFAULT = 100.0

def np_mean(v):
    return np.mean(v) if len(v) > 0 else 0.0

def np_variance(v):
    if len(v) < 2:
        return 0.0
    return np.var(v, ddof=1)

def np_covariance(x, y):
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return np.cov(x, y, ddof=1)[0, 1]

def log_returns(prices_np):
    prices_np = np.array(prices_np, dtype=np.float64)
    if prices_np.size < 2:
        return np.array([])
    return np.log(prices_np[1:] / prices_np[:-1])

def estimate_xi(logrets_np, dt_yr_np):
    var_r_np = np_variance(logrets_np)
    return var_r_np / dt_yr_np

def detrend_segment(segment_np):
    n_np = len(segment_np)
    if n_np < 2:
        return segment_np
    t_np = np.arange(1, n_np + 1, dtype=np.float64)
    tm_np = np_mean(t_np)
    ym_np = np_mean(segment_np)
    num_np = np.sum((t_np - tm_np) * (segment_np - ym_np))
    den_np = np.sum((t_np - tm_np) ** 2)
    if abs(den_np) < 1e-14:
        return segment_np
    slope_np = num_np / den_np
    intercept_np = ym_np - slope_np * tm_np
    return segment_np - (slope_np * t_np + intercept_np)

def hurst_exponent_DFA(data_in_np):
    data_np = np.array(data_in_np, dtype=np.float64)
    if len(data_np) < 20:
        return H_DEFAULT
    data_np = data_np - np_mean(data_np)
    data_np = np.cumsum(data_np)
    log_window_size_np = []
    log_fluctuation_np = []
    min_window_size_np = 10
    max_window_size_np = len(data_np) // 4
    if max_window_size_np < min_window_size_np:
        return H_DEFAULT
        
    w_np = min_window_size_np
    while w_np <= max_window_size_np:
        fluctuations_np = []
        for start_np in range(0, len(data_np) - w_np + 1, w_np):
            segment_np = data_np[start_np:start_np+w_np]
            detrended_np = detrend_segment(segment_np)
            rms_np = np.sqrt(np.mean(detrended_np**2))
            fluctuations_np.append(rms_np)
        mf_np = np_mean(fluctuations_np)
        if mf_np > 1e-8:
            log_window_size_np.append(np.log(w_np))
            log_fluctuation_np.append(np.log(mf_np))
        
        if w_np * 2 > max_window_size_np and w_np < max_window_size_np :
             w_np = max_window_size_np
        elif w_np * 2 > max_window_size_np and w_np == max_window_size_np:
            break
        else:
            w_np *= 2
            
    n_np_hurst = len(log_window_size_np)
    if n_np_hurst < 2:
        return H_DEFAULT
    sumX_np = np.sum(log_window_size_np)
    sumY_np = np.sum(log_fluctuation_np)
    sumXX_np = np.sum(np.array(log_window_size_np) ** 2)
    sumXY_np = np.sum(np.array(log_window_size_np) * np.array(log_fluctuation_np))
    if (n_np_hurst * sumXX_np - sumX_np ** 2) == 0:
        return H_DEFAULT
    slope_np = (n_np_hurst * sumXY_np - sumX_np * sumY_np) / (n_np_hurst * sumXX_np - sumX_np ** 2)
    return np.clip(slope_np, 0.01, 0.49)

def estimate_H(logrets_np):
    return hurst_exponent_DFA(logrets_np)

def estimate_eta(logrets_np, H_np, window_np=20):
    if len(logrets_np) < window_np +1 :
        return ETA_DEFAULT
    realized_var_np = []
    for i_np in range(window_np - 1, len(logrets_np)):
        window_returns_np = logrets_np[i_np - window_np + 1:i_np + 1]
        rv_np = np.mean(np.square(window_returns_np))
        realized_var_np.append(rv_np)
    if not realized_var_np:
        return ETA_DEFAULT
    log_rv_np = np.log(np.array(realized_var_np))
    if len(log_rv_np) < 2:
        return ETA_DEFAULT
    log_diff_np = np.diff(log_rv_np)
    if len(log_diff_np) < 2:
        return ETA_DEFAULT
    daily_eta_np = np.std(log_diff_np, ddof=1)
    return daily_eta_np * cp.sqrt(252.0)

def estimate_rho(logrets_np):
    if len(logrets_np) < 2:
        return RHO_DEFAULT
    logrets_np_arr = np.array(logrets_np, dtype=np.float64)
    sq_np = logrets_np_arr ** 2
    c_np = np_covariance(logrets_np_arr, sq_np)
    var_logrets = np_variance(logrets_np_arr)
    var_sq = np_variance(sq_np)
    if var_logrets == 0 or var_sq == 0:
        return RHO_DEFAULT
    denom_np = cp.sqrt(var_logrets * var_sq)
    rho_np = c_np / denom_np if denom_np != 0.0 else 0.0
    if rho_np > 0.0: # Empirically rho is negative for equities
        rho_np = -0.3 
    return np.clip(rho_np, -0.99, -0.01)


def estimate_params(historical_prices_np, dt_yr_np=1/252):
    if len(historical_prices_np) < 21: # Minimum data points for reliable estimation
        print("Historical data too short, using default rBergomi parameters.")
        return S0_DEFAULT if len(historical_prices_np) == 0 else historical_prices_np[-1], XI_DEFAULT, H_DEFAULT, ETA_DEFAULT, RHO_DEFAULT

    prices_np = np.array(historical_prices_np, dtype=np.float64)
    S0_val = prices_np[-1]
    rets_np = log_returns(prices_np)
    if len(rets_np) == 0:
        return S0_val, XI_DEFAULT, H_DEFAULT, ETA_DEFAULT, RHO_DEFAULT

    xi_val = estimate_xi(rets_np, dt_yr_np)
    H_val = estimate_H(rets_np)
    eta_val = estimate_eta(rets_np, H_val)
    rho_val = estimate_rho(rets_np)
    
    if not all(map(np.isfinite, [xi_val, H_val, eta_val, rho_val])):
        print("Parameter estimation resulted in non-finite values, using defaults.")
        return S0_val, XI_DEFAULT, H_DEFAULT, ETA_DEFAULT, RHO_DEFAULT
        
    return S0_val, xi_val, H_val, eta_val, rho_val

def seed_everything(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    cp.random.seed(seed_val)

def next_power_of_two(n_val):
    p_val = 1
    while p_val < n_val:
        p_val <<= 1
    return p_val

def rbergomi_lambda_gpu(time_grid_gpu, H_param):
    return 0.5 * (time_grid_gpu ** (2 * H_param))

def rbergomi_phi_gpu(lam_gpu):
    N_phi = lam_gpu.shape[0]
    M_phi = next_power_of_two(N_phi)
    lam_padded_gpu = cp.zeros(M_phi, dtype=cp.float64)
    lam_padded_gpu[:N_phi] = lam_gpu
    return cp.fft.fft(lam_padded_gpu)

def fractional_gaussian_gpu(phi_val_gpu, Z_val_gpu, H_param, eta_param, out_len_val):
    if Z_val_gpu.ndim == 2: 
        A_gpu = phi_val_gpu[None, :] * Z_val_gpu
        A_ifft_gpu = cp.fft.ifft(A_gpu, axis=1).real
    elif Z_val_gpu.ndim == 3: 
        A_gpu = phi_val_gpu[None, None, :] * Z_val_gpu
        A_ifft_gpu = cp.fft.ifft(A_gpu, axis=2).real
    else:
        raise ValueError("Z_val_gpu must be 2D or 3D")
    
    scale_val = cp.sqrt(2 * H_param) * eta_param 
    return scale_val * A_ifft_gpu[..., :out_len_val]


def forward_variance_gpu(X_gpu, t_grid_gpu, xi_param_b, H_param, eta_param):
    v_gpu = cp.zeros_like(X_gpu)
    
    if xi_param_b.ndim < X_gpu.ndim:
         xi_param_b = xi_param_b.reshape(xi_param_b.shape + (1,) * (X_gpu.ndim - xi_param_b.ndim -1 ) + (X_gpu.shape[-1],)) #This is not right
         xi_param_b = xi_param_b[...,0] #Keep only first element from last dummy dimension
         xi_param_b = xi_param_b.reshape(xi_param_b.shape + (1,))


    for i_val in range(X_gpu.shape[-1]):
        t_val = t_grid_gpu[i_val]
        ma_gpu = -0.5 * eta_param * eta_param * (t_val ** (2 * H_param))
        if X_gpu.ndim == 2: # (n_paths, n_steps+1)
            v_gpu[:, i_val] = xi_param_b * cp.exp(X_gpu[:, i_val] + ma_gpu)
        elif X_gpu.ndim == 3: # (batch_size, n_mc_paths, n_steps+1)
             v_gpu[..., i_val] = xi_param_b[..., None] * cp.exp(X_gpu[..., i_val] + ma_gpu)


    return v_gpu


def price_rbergomi_option_gpu(S0_batch_gpu, K_batch_gpu, T_opt_val, r_opt_val, xi_batch_gpu, H_opt_val, eta_opt_val, rho_opt_val, option_type_opt, n_mc_paths_per_option, dt_opt_val):
    batch_size = S0_batch_gpu.shape[0]
    n_steps_opt = int(T_opt_val / dt_opt_val)

    if n_steps_opt <= 0:
        payoffs_gpu = cp.zeros_like(S0_batch_gpu)
        if option_type_opt == 'call':
            payoffs_gpu = cp.maximum(S0_batch_gpu - K_batch_gpu, 0.0)
        elif option_type_opt == 'put':
            payoffs_gpu = cp.maximum(K_batch_gpu - S0_batch_gpu, 0.0)
        return payoffs_gpu * cp.exp(-r_opt_val * T_opt_val)

    time_grid_opt_gpu = cp.linspace(0, n_steps_opt * dt_opt_val, n_steps_opt + 1, dtype=cp.float64)
    lam_opt_gpu = rbergomi_lambda_gpu(time_grid_opt_gpu, H_opt_val)
    phi_opt_gpu = rbergomi_phi_gpu(lam_opt_gpu)
    M_opt_val = phi_opt_gpu.shape[0]

    Z_opt_gpu = cp.random.normal(size=(batch_size, n_mc_paths_per_option, M_opt_val), dtype=cp.float64) + \
                1j * cp.random.normal(size=(batch_size, n_mc_paths_per_option, M_opt_val), dtype=cp.float64)

    X_paths_opt_gpu = fractional_gaussian_gpu(phi_opt_gpu, Z_opt_gpu, H_opt_val, eta_opt_val, n_steps_opt + 1)
    
    xi_batch_expanded_gpu = xi_batch_gpu
    v_paths_opt_gpu = forward_variance_gpu(X_paths_opt_gpu, time_grid_opt_gpu, xi_batch_expanded_gpu, H_opt_val, eta_opt_val)
    
    w_complex_opt_gpu = cp.fft.ifft(Z_opt_gpu, axis=2, n=M_opt_val)
    dW1_unscaled_opt_gpu = w_complex_opt_gpu.real * cp.sqrt(float(M_opt_val))
    dW2_unscaled_opt_gpu = w_complex_opt_gpu.imag * cp.sqrt(float(M_opt_val))

    current_prices_opt_gpu = cp.full((batch_size, n_mc_paths_per_option), S0_batch_gpu[:, None], dtype=cp.float64)
    sqrt_dt_opt = cp.sqrt(dt_opt_val)

    for j_opt_val in range(1, n_steps_opt + 1):
        dw1_opt_gpu = sqrt_dt_opt * dW1_unscaled_opt_gpu[..., j_opt_val - 1]
        dw2_opt_gpu = sqrt_dt_opt * dW2_unscaled_opt_gpu[..., j_opt_val - 1]
        
        dW_opt_gpu = rho_opt_val * dw1_opt_gpu + cp.sqrt(cp.maximum(0.0, 1.0 - rho_opt_val * rho_opt_val)) * dw2_opt_gpu
        
        vt_opt_gpu = v_paths_opt_gpu[..., j_opt_val - 1]
        drift_opt_gpu = (r_opt_val - 0.5 * vt_opt_gpu) * dt_opt_val
        diff_opt_gpu = cp.sqrt(cp.maximum(0.0, vt_opt_gpu)) * dW_opt_gpu
        current_prices_opt_gpu *= cp.exp(drift_opt_gpu + diff_opt_gpu)
        current_prices_opt_gpu = cp.maximum(current_prices_opt_gpu, 1e-8)

    terminal_prices_opt_gpu = current_prices_opt_gpu

    payoffs_final_gpu = cp.zeros_like(terminal_prices_opt_gpu)
    if option_type_opt == 'call':
        payoffs_final_gpu = cp.maximum(terminal_prices_opt_gpu - K_batch_gpu[:, None], 0.0)
    elif option_type_opt == 'put':
        payoffs_final_gpu = cp.maximum(K_batch_gpu[:, None] - terminal_prices_opt_gpu, 0.0)
        
    option_prices_batch_gpu = cp.mean(payoffs_final_gpu, axis=1) * cp.exp(-r_opt_val * T_opt_val)
    return option_prices_batch_gpu


def generate_paths_and_options(historical_prices_np_main, forward_steps_main, num_paths_main, r_main, dt_main, seed_main=None):
    if seed_main is not None:
        seed_everything(seed_main)

    S0_main, xi_main, H_main, eta_main, rho_main = estimate_params(historical_prices_np_main, dt_main)
    print(f"Estimated rBergomi Params: S0={S0_main:.2f}, xi={xi_main:.4f}, H={H_main:.4f}, eta={eta_main:.4f}, rho={rho_main:.4f}")


    time_grid_main_gpu = cp.linspace(0, forward_steps_main * dt_main, forward_steps_main + 1, dtype=cp.float64)
    lam_main_gpu = rbergomi_lambda_gpu(time_grid_main_gpu, H_main)
    phi_main_gpu = rbergomi_phi_gpu(lam_main_gpu)
    M_main_val = phi_main_gpu.shape[0]

    Z_main_gpu = cp.random.normal(size=(num_paths_main, M_main_val), dtype=cp.float64) + \
                 1j * cp.random.normal(size=(num_paths_main, M_main_val), dtype=cp.float64)
    
    X_main_gpu = fractional_gaussian_gpu(phi_main_gpu, Z_main_gpu, H_main, eta_main, forward_steps_main + 1)
    
    v_main_gpu = forward_variance_gpu(X_main_gpu, time_grid_main_gpu, cp.array([xi_main]), H_main, eta_main)


    w_complex_main_gpu = cp.fft.ifft(Z_main_gpu, axis=1, n=M_main_val)
    dW1_unscaled_main_gpu = w_complex_main_gpu.real * cp.sqrt(float(M_main_val))
    dW2_unscaled_main_gpu = w_complex_main_gpu.imag * cp.sqrt(float(M_main_val))

    paths_main_gpu = cp.zeros((num_paths_main, forward_steps_main + 1), dtype=cp.float64)
    paths_main_gpu[:, 0] = S0_main
    
    call_prices_atm_gpu = cp.zeros((num_paths_main, forward_steps_main), dtype=cp.float64)
    put_prices_atm_gpu = cp.zeros((num_paths_main, forward_steps_main), dtype=cp.float64)
    
    sqrt_dt_main = cp.sqrt(dt_main)
    
    total_option_pricing_time_tracker = 0.0

    print("Starting main path generation and option pricing...")
    for j_main_val in tqdm(range(1, forward_steps_main + 1), desc="Main Simulation Progress"):
        
        current_S_for_opt_gpu = paths_main_gpu[:, j_main_val - 1].copy()
        current_v_for_opt_gpu = v_main_gpu[:, j_main_val - 1].copy()
        K_atm_step_gpu = cp.round(current_S_for_opt_gpu)
        
        opt_pricing_step_start_time = time.time()
        if T_OPTION_TENOR > 1e-6 :
            call_prices_atm_gpu[:, j_main_val - 1] = price_rbergomi_option_gpu(
                current_S_for_opt_gpu, K_atm_step_gpu, T_OPTION_TENOR, r_main, 
                current_v_for_opt_gpu, H_main, eta_main, rho_main, 
                'call', N_PATHS_OPTION_MC, dt_main
            )
            put_prices_atm_gpu[:, j_main_val - 1] = price_rbergomi_option_gpu(
                current_S_for_opt_gpu, K_atm_step_gpu, T_OPTION_TENOR, r_main, 
                current_v_for_opt_gpu, H_main, eta_main, rho_main, 
                'put', N_PATHS_OPTION_MC, dt_main
            )
        else: 
             call_prices_atm_gpu[:, j_main_val - 1] = cp.maximum(current_S_for_opt_gpu - K_atm_step_gpu, 0.0)
             put_prices_atm_gpu[:, j_main_val - 1] = cp.maximum(K_atm_step_gpu - current_S_for_opt_gpu, 0.0)
        
        opt_pricing_step_end_time = time.time()
        total_option_pricing_time_tracker += (opt_pricing_step_end_time - opt_pricing_step_start_time)
        
        dw1_main_gpu = sqrt_dt_main * dW1_unscaled_main_gpu[:, j_main_val - 1]
        dw2_main_gpu = sqrt_dt_main * dW2_unscaled_main_gpu[:, j_main_val - 1]
        
        dW_main_gpu = rho_main * dw1_main_gpu + cp.sqrt(cp.maximum(0.0, 1.0 - rho_main * rho_main)) * dw2_main_gpu
        
        vt_main_gpu = v_main_gpu[:, j_main_val - 1]
        drift_main_gpu = (r_main - 0.5 * vt_main_gpu) * dt_main
        diff_main_gpu = cp.sqrt(cp.maximum(0.0, vt_main_gpu)) * dW_main_gpu
        
        paths_main_gpu[:, j_main_val] = paths_main_gpu[:, j_main_val - 1] * cp.exp(drift_main_gpu + diff_main_gpu)
        paths_main_gpu[:, j_main_val] = cp.maximum(paths_main_gpu[:, j_main_val], 1e-8)
    
    print(f"Total time spent in batched option pricing calls: {total_option_pricing_time_tracker:.2f} seconds.")
    return paths_main_gpu, v_main_gpu, call_prices_atm_gpu, put_prices_atm_gpu

def main():
    start_total_time = time.time()
    
    try:
        prices_np_hist = np.loadtxt(INPUT_FILE, dtype=np.float64, delimiter=',')
    except FileNotFoundError:
        print(f"Historical prices file {INPUT_FILE} not found. Using default S0 and rBergomi parameters.")
        prices_np_hist = np.array([]) 
    except Exception as e:
        print(f"Error loading historical prices: {e}. Using default S0 and rBergomi parameters.")
        prices_np_hist = np.array([])

    if prices_np_hist.ndim == 0 and prices_np_hist.size == 1:
        prices_np_hist = np.array([float(prices_np_hist)])
    elif prices_np_hist.ndim > 1:
        print("Historical prices file has more than one column, using the first column.")
        prices_np_hist = prices_np_hist[:,0]
    
    paths_gpu, vol_gpu, calls_gpu, puts_gpu = generate_paths_and_options(
        prices_np_hist, N_STEPS, N_PATHS, R, DT, SEED
    )
    
    paths_np = cp.asnumpy(paths_gpu)
    vol_np = cp.asnumpy(vol_gpu)
    calls_np = cp.asnumpy(calls_gpu)
    puts_np = cp.asnumpy(puts_gpu)
    
    np.savez_compressed(OUTPUT_FILE, paths=paths_np, volatilities=vol_np, call_prices_atm=calls_np, put_prices_atm=puts_np)
    print(f"Saved {N_PATHS} paths with {N_STEPS} steps, along with volatilities and ATM option prices to {OUTPUT_FILE}")
    
    end_total_time = time.time()
    print(f"Total script execution time: {(end_total_time - start_total_time):.2f} seconds.")

if __name__ == '__main__':
    main()