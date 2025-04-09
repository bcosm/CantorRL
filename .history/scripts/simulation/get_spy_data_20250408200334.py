import yfinance as yf
import pandas as pd
import argparse
import sys

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Download historical SPY price data')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--days', type=int, default=1000, help='Number of days to download (default: 1000)')
    parser.add_argument('--output', type=str, default='historical_prices.csv', help='Output filename (default: historical_prices.csv)')
    args = parser.parse_args()
    
    ticker_symbol = args.ticker
    num_days = args.days
    output_filename = args.output
    
    # Download data from Yahoo Finance
    # Use a larger period to ensure we get enough trading days
    buffer_factor = 1.5  # Add 50% more calendar days to account for weekends and holidays
    period_str = f"{int(num_days * buffer_factor)}d"
    
    print(f"Downloading {ticker_symbol} data...")
    try:
        data = yf.download(ticker_symbol, period=period_str)
    except Exception as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)
    
    if data.empty:
        print(f"Error: No data downloaded for {ticker_symbol}")
        sys.exit(1)
    
    # Extract closing prices
    closing_prices = data['Close']
    
    # Trim to the desired number of days
    if len(closing_prices) > num_days:
        closing_prices = closing_prices[-num_days:]
    
    # Save to CSV in the format expected by rough-volatility.py
    pd.DataFrame(closing_prices.values).to_csv(output_filename, header=False, index=False)
    
    print(f"Successfully saved {len(closing_prices)} days of {ticker_symbol} closing prices to {output_filename}")
    print(f"Date range: {closing_prices.index[0].date()} to {closing_prices.index[-1].date()}")

if __name__ == "__main__":
    main()
