import numpy as np
import math
from rbergomi_params import estimate_params
from utils import seed_everything

# Configuration variables
INPUT_FILE = './data/historical_prices.csv'
OUTPUT_FILE = './data/paths.npy'
R = 0.04
DT = 1/252
N_PATHS = 500
N_STEPS = 252
SEED = 42

def next_power_of_two(n):
    p = 1
    while p < n:
        p <<= 1
    return p

def rbergomi_lambda(time_grid, H):
    return 0.5 * (time_grid ** (2 * H))

def rbergomi_phi(lam, H):
    N = lam.shape[0]
    M = next_power_of_two(N)
    lam_padded = np.zeros(M, dtype=np.float64)
    lam_padded[:N] = lam
    return np.fft.fft(lam_padded)

def fractional_gaussian(phi, Z, H, eta, out_len):
    A = phi[None, :] * Z
    A_ifft = np.fft.ifft(A, axis=1).real
    scale = math.sqrt(2 * H) * eta
    return scale * A_ifft[:, :out_len]

def forward_variance(X, t_grid, xi, H, eta):
    v = np.zeros_like(X)
    t_grid = np.array(t_grid, dtype=np.float64)
    for i in range(X.shape[1]):
        t = t_grid[i]
        ma = -0.5 * eta * eta * (t ** (2 * H))
        v[:, i] = xi * np.exp(X[:, i] + ma)
    return v

def generate_price_paths(historical_prices, forward_steps, num_paths, r, dt_yr, seed=None):
    if seed is not None:
        seed_everything(seed)

    prices = np.array(historical_prices, dtype=np.float64)
    S0 = prices[-1]
    xi, H, eta, rho = estimate_params(prices, dt_yr)

    dt = dt_yr
    time_grid = np.linspace(0, forward_steps * dt, forward_steps + 1)
    lam = rbergomi_lambda(time_grid, H)
    phi = rbergomi_phi(lam, H)

    M = phi.shape[0]
    Z = np.random.normal(size=(num_paths, M)) + 1j * np.random.normal(size=(num_paths, M))
    W1 = np.random.normal(size=(num_paths, forward_steps))
    W2 = np.random.normal(size=(num_paths, forward_steps))

    X = fractional_gaussian(phi, Z, H, eta, forward_steps + 1)
    v = forward_variance(X, time_grid, xi, H, eta)

    paths = np.zeros((num_paths, forward_steps + 1), dtype=np.float64)
    paths[:, 0] = S0
    for j in range(1, forward_steps + 1):
        dw1 = math.sqrt(dt) * W1[:, j - 1]
        dw2 = math.sqrt(dt) * W2[:, j - 1]
        dW = rho * dw1 + np.sqrt(np.maximum(0.0, 1 - rho * rho)) * dw2
        vt = v[:, j - 1]
        drift = (r - 0.5 * vt) * dt
        diff = np.sqrt(np.maximum(0.0, vt)) * dW
        paths[:, j] = paths[:, j - 1] * np.exp(drift + diff)

    return paths

def main():
    prices = np.loadtxt(INPUT_FILE, dtype=np.float64, delimiter=',')
    paths = generate_price_paths(prices, N_STEPS, N_PATHS, R, DT, SEED)
    np.save(OUTPUT_FILE, paths)

if __name__ == '__main__':
    main()
