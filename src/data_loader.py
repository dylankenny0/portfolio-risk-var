import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def fetch_portfolio_data(tickers, start_date, end_date):
    """Fetch historical price data for portfolio tickers with retry logic"""
    
    # Try downloading all at once first
    try:
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date,
            progress=False,  # Disable progress bar
            auto_adjust=True  # Use adjusted close automatically
        )
        
        # If multiple tickers, extract Close prices
        if len(tickers) > 1:
            if 'Close' in data.columns:
                data = data['Close']
        
        # Handle single ticker
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers
        
        # Check if we got data
        if data.empty:
            raise ValueError("No data downloaded")
        
        # Handle missing data
        data = data.ffill().bfill()
        
        return data
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Trying individual downloads...")
        
        # Fallback: download each ticker individually
        all_data = {}
        for ticker in tickers:
            try:
                print(f"Downloading {ticker}...")
                time.sleep(0.5)  # Small delay to avoid rate limiting
                
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    all_data[ticker] = hist['Close']
                else:
                    print(f"Warning: No data for {ticker}")
                    
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue
        
        if not all_data:
            raise ValueError("Failed to download any data")
        
        # Combine into DataFrame
        data = pd.DataFrame(all_data)
        data = data.ffill().bfill()
        
        return data

def calculate_returns(prices):
    """Calculate log returns from prices"""
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()

def get_market_data(tickers=['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC', 
                              'GS', 'XOM', 'JNJ', 'PG', 'KO'],
                    years=3):
    """Get default portfolio data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    print(f"Fetching data from {start_date.date()} to {end_date.date()}")
    
    prices = fetch_portfolio_data(tickers, start_date, end_date)
    returns = calculate_returns(prices)
    
    print(f"Successfully loaded {len(prices)} days of data for {len(prices.columns)} tickers")
    
    return prices, returns
