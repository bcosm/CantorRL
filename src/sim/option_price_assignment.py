import pandas as pd
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from tqdm import tqdm  # Import tqdm for the progress bar

# Define the risk-free rate
RISK_FREE_RATE = 0.05

def black_scholes_price(S, K, T, r, sigma, epsilon=1e-8):
    """
    Compute European call and put option prices using the Black-Scholes formula.
    
    Parameters:
    S : float - current underlying price.
    K : float - strike price.
    T : float - time to maturity (years).
    r : float - risk-free rate.
    sigma : float - annualized volatility.
    epsilon : float - threshold to detect near-zero volatility.
    
    Returns:
    call : float - call option price.
    put : float - put option price.
    """
    # Handle near-zero volatility to avoid division by zero.
    if sigma < epsilon:
        call = max(S - K * exp(-r * T), 0)
        put = max(K * exp(-r * T) - S, 0)
        return call, put

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    put = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put

def calculate_annualized_vol(prices):
    """
    Compute the annualized volatility using historical log returns from the start of the path
    up to the current time step.
    
    Parameters:
    prices : numpy.ndarray - array of prices from time 0 to current time step.
    
    Returns:
    sigma : float - estimated annualized volatility.
    """
    if len(prices) < 2:
        return 0.0
    log_returns = np.log(prices[1:] / prices[:-1])
    if len(log_returns) == 1:
        sigma_daily = 0.0
    else:
        sigma_daily = np.std(log_returns, ddof=1)
    sigma = sigma_daily * np.sqrt(252)
    return sigma

def process_price_paths(input_csv, output_csv):
    """
    Processes simulated price paths to compute call and put prices at every time step,
    displaying a progress bar with tqdm.
    
    The input CSV should have a header row with time step labels (0,1,...,N) and
    each subsequent row is a price path.
    
    The output CSV will be in long format with columns: sim_id, time_step, price, call, put.
    
    Parameters:
    input_csv : str - path to the input CSV file.
    output_csv : str - path to the output CSV file.
    """
    df_prices = pd.read_csv(input_csv, header=0)
    records = []

    # Wrap the simulation iteration with tqdm for a progress bar.
    for sim_id, row in tqdm(df_prices.iterrows(), total=df_prices.shape[0], desc="Processing simulations"):
        prices = row.values.astype(float)
        strike = round(prices[0])
        
        # Process every time step in the current simulation path.
        for t in range(len(prices)):
            S = prices[t]
            T = 1 - (t / 252)
            T = max(T, 0.0)
            sigma = calculate_annualized_vol(prices[:t+1])
            call_price, put_price = black_scholes_price(S, strike, T, RISK_FREE_RATE, sigma)

            records.append({
                "sim_id": sim_id,
                "time_step": t,
                "price": S,
                "call": call_price,
                "put": put_price
            })

    df_results = pd.DataFrame(records)
    df_results.to_csv(output_csv, index=False)
    print(f"Processed {df_results['sim_id'].nunique()} simulation(s) and saved option prices to {output_csv}")

# Main execution: adjust file paths as needed.
if __name__ == "__main__":
    input_csv = "scripts/simulation/price_paths.csv"   # your input CSV file containing the price paths
    output_csv = "scripts/simulation/price_paths_options.csv"      # desired output CSV file name
    process_price_paths(input_csv, output_csv)
