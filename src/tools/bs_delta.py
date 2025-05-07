import numpy as np
import math
from scipy.stats import norm

# Configuration variables
INPUT_PATHS_FILE = 'data/paths.npy'
OUTPUT_PNL_FILE = 'results/bs_pnl.npy'
RISK_FREE_RATE = 0.04
DT = 1/252

def black_scholes_price(S, K, T, r, sigma, epsilon=1e-8):
    if sigma < epsilon or T <= 0:
        call = max(S - K * math.exp(-r * T), 0)
        return call
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call

def black_scholes_delta(S, K, T, r, sigma, epsilon=1e-8):
    if sigma < epsilon or T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)

def calculate_annualized_vol(prices):
    if len(prices) < 2:
        return 0.0
    log_returns = np.log(prices[1:] / prices[:-1])
    if len(log_returns) < 2:
        sigma_daily = 0.0
    else:
        sigma_daily = np.std(log_returns, ddof=1)
    return sigma_daily * math.sqrt(252)

def bs_delta_hedge(paths):
    n_sims, n_steps1 = paths.shape
    T_total = n_steps1 * DT
    pnls = np.zeros((n_sims, n_steps1))
    for i in range(n_sims):
        prices = paths[i]
        K = prices[0]
        cash = 0.0
        prev_delta = 0.0
        for t in range(n_steps1):
            S = prices[t]
            T_remain = max(T_total - t * DT, 0.0)
            sigma = calculate_annualized_vol(prices[:t+1])
            delta = black_scholes_delta(S, K, T_remain, RISK_FREE_RATE, sigma)
            d_delta = delta - prev_delta
            cash -= d_delta * S
            prev_delta = delta
            call_price = black_scholes_price(S, K, T_remain, RISK_FREE_RATE, sigma)
            pnls[i, t] = cash + prev_delta * S - call_price
    return pnls

def main():
    paths = np.load(INPUT_PATHS_FILE)
    pnls = bs_delta_hedge(paths)
    np.save(OUTPUT_PNL_FILE, pnls)
    print(f"Saved Black‑Scholes delta‑hedge PnL to {OUTPUT_PNL_FILE}")

if __name__ == '__main__':
    main()
