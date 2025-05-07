import pandas as pd
import numpy as np
from tqdm import tqdm

# *********************
# Technical Indicator Functions
# *********************

def compute_indicators(series):
    """
    Given a pandas Series of prices, compute 20 technical indicators.
    Returns a DataFrame with the following columns:
      SMA10, SMA50, EMA10, EMA50, RSI14,
      Bollinger_Upper20, Bollinger_Lower20, Bollinger_Bandwidth20,
      MACD, MACD_Signal, MACD_Histogram,
      Momentum10, ROC10, HistoricalVol20,
      Stochastic_K14, Stochastic_D3, Williams_R14,
      CCI20, ATR14, ROC5.
    """
    df = pd.DataFrame(index=series.index)
    
    # 1. Simple Moving Averages (SMA)
    df['SMA10'] = series.rolling(window=10, min_periods=1).mean()
    df['SMA50'] = series.rolling(window=50, min_periods=1).mean()
    
    # 2. Exponential Moving Averages (EMA)
    df['EMA10'] = series.ewm(span=10, adjust=False).mean()
    df['EMA50'] = series.ewm(span=50, adjust=False).mean()
    
    # 3. RSI14
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use a 14-period window (results will be NaN until 14 values are available)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # 4. Bollinger Bands (20)
    sma20 = series.rolling(window=20, min_periods=20).mean()
    std20 = series.rolling(window=20, min_periods=20).std()
    df['Bollinger_Upper20'] = sma20 + 2 * std20
    df['Bollinger_Lower20'] = sma20 - 2 * std20
    df['Bollinger_Bandwidth20'] = df['Bollinger_Upper20'] - df['Bollinger_Lower20']
    
    # 5. MACD and its Signal/Histrogram
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # 6. Momentum and Rate of Change (ROC)
    df['Momentum10'] = series / series.shift(10) - 1
    df['ROC10'] = 100 * (series - series.shift(10)) / series.shift(10)
    
    # 7. Historical Volatility (20): rolling standard deviation of log returns annualized
    log_returns = np.log(series / series.shift(1))
    df['HistoricalVol20'] = log_returns.rolling(window=20, min_periods=20).std() * np.sqrt(252)
    
    # 8. Stochastic Oscillator %K (14) and %D (3)
    lowest14 = series.rolling(window=14, min_periods=14).min()
    highest14 = series.rolling(window=14, min_periods=14).max()
    df['Stochastic_K14'] = 100 * (series - lowest14) / (highest14 - lowest14)
    df['Stochastic_D3'] = df['Stochastic_K14'].rolling(window=3, min_periods=3).mean()
    
    # 9. Williams %R (14)
    df['Williams_R14'] = -100 * (highest14 - series) / (highest14 - lowest14)
    
    # 10. Commodity Channel Index (CCI20)
    mean_dev = series.rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['CCI20'] = (series - sma20) / (0.015 * mean_dev)
    
    # 11. Average True Range (ATR14)
    true_range = (series - series.shift(1)).abs()
    df['ATR14'] = true_range.rolling(window=14, min_periods=14).mean()
    
    # 12. ROC5 – 5–period rate of change (in %)
    df['ROC5'] = 100 * (series - series.shift(5)) / series.shift(5)
    
    return df

# *********************
# Processing Option Prices and Merging with Indicators
# *********************

def process_option_prices(input_csv, hist_csv, output_csv):
    """
    Loads option_prices.csv (which contains simulation data in long format) and for each simulation:
      - Extends the simulation’s price series with historical SPY prices (from historical_prices.csv)
      - Computes 20 technical indicators using the extended series
      - Merges these indicators with the existing data (sim_id, time_step, price, call, put)
    The result is saved to output_csv.
    """
    # Load historical prices.
    # historical_prices.csv is assumed to have one float per line,
    # with the most recent price at the bottom.
    hist_prices = pd.read_csv(hist_csv, header=None).squeeze()
    
    # Determine maximum lookback needed. The largest window we use is 50 (e.g., SMA50),
    # so we need 49 historical data points.
    required_history_length = 50 - 1  
    if len(hist_prices) < required_history_length:
        print("Warning: Not enough historical prices; available:", len(hist_prices))
        hist_tail = hist_prices.values.astype(float)
    else:
        # Take the last 'required_history_length' prices from the historical file.
        hist_tail = hist_prices.iloc[-required_history_length:].values.astype(float)
    
    # Load option_prices.csv into a DataFrame.
    df_options = pd.read_csv(input_csv)
    # Assumes long format with columns: sim_id, time_step, price, call, put.
    
    processed_records = []
    groups = df_options.groupby('sim_id')
    
    # Iterate over simulation groups with a tqdm progress bar.
    for sim_id, group in tqdm(groups, total=groups.ngroups, desc="Processing simulations"):
        group = group.sort_values('time_step')
        sim_prices = group['price'].values.astype(float)
        
        # Extend the simulation's price series with historical tail.
        # Concatenate the historical tail (assumed ordered oldest-to-recent)
        # and then the simulation price path.
        extended_prices = np.concatenate([hist_tail, sim_prices])
        ext_series = pd.Series(extended_prices)
        
        # Compute technical indicators on the extended series.
        indicators = compute_indicators(ext_series)
        
        # Extract only the simulation part. The simulation data starts at index len(hist_tail).
        sim_indicators = indicators.iloc[len(hist_tail):].reset_index(drop=True)
        
        # Merge indicator data with the corresponding option price data.
        sim_indicators['sim_id'] = sim_id
        sim_indicators['time_step'] = group['time_step'].values
        sim_indicators['price'] = group['price'].values
        sim_indicators['call'] = group['call'].values
        sim_indicators['put'] = group['put'].values
        
        processed_records.append(sim_indicators)
    
    # Combine all simulations into one DataFrame.
    result_df = pd.concat(processed_records, ignore_index=True)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved preprocessed data with technical indicators to {output_csv}")

# *********************
# Main Execution
# *********************

if __name__ == "__main__":
    input_csv = "scripts/simulation/price_paths_options.csv"         # Input CSV from simulation runs
    hist_csv = "scripts/simulation/historical_prices.csv"        # Historical SPY prices (one float per line)
    output_csv = "dataset.csv"      # Output file with added technical indicators
    process_option_prices(input_csv, hist_csv, output_csv)
