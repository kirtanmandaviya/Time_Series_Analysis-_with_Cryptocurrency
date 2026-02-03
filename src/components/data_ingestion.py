"""
Data Fetching Module
Fetches cryptocurrency data from CoinGecko API
"""

import requests
import pandas as pd
import os
import time
import sys
from datetime import datetime
from dotenv import load_dotenv
from src.exception.exception import CustomException

load_dotenv()

# Import from our pipeline configuration
from src.constants.prediction_pipeline import (
    COINGECKO_BASE_URL,
    RAW_DATA_DIR,
    CRYPTOCURRENCIES,
    HISTORY_DAYS
)
from src.utils.utils import get_coingecko_url, ensure_dir_exists


class CryptoDataFetcher:
    def __init__(self):
        """Initialize the data fetcher with configuration from pipeline"""
        self.base_url = COINGECKO_BASE_URL
        self.raw_data_path = RAW_DATA_DIR
        self.api_key: str = os.getenv("COINGECKO_API_KEY")
        
        if not self.api_key:
            print("Warning: API key not found. Set COINGECKO_API_KEY environment variable.")
            print("Requests might fail due to rate limits or authorization errors.")
            
        # Create directory if it doesn't exist
        ensure_dir_exists(self.raw_data_path)

    def fetch_historical_data(self, crypto_id='bitcoin', vs_currency='usd', days=None):
        """
        Fetch historical price data from CoinGecko
        
        Args:
            crypto_id: CoinGecko crypto ID (e.g., 'bitcoin', 'ethereum')
            vs_currency: Currency to compare against (default: 'usd')
            days: Number of days of historical data (default: from config)
        
        Returns:
            DataFrame with historical data or None if error
        """
        if days is None:
            days = HISTORY_DAYS
        
        endpoint = f"coins/{crypto_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily'
        }
        
        # Use utility function to build URL with API key
        url = get_coingecko_url(endpoint, params)
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Extract data
            prices = data['prices']
            market_caps = data['market_caps']
            volumes = data['total_volumes']
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': [p[0] for p in prices],
                'price': [p[1] for p in prices],
                'market_cap': [m[1] for m in market_caps],
                'volume': [v[1] for v in volumes]
            })
            
            # Convert timestamp to date
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop('timestamp', axis=1)
            df = df[['date', 'price', 'market_cap', 'volume']]
            
            print(f"Successfully fetched {len(df)} days of {crypto_id} data")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def fetch_current_price(self, crypto_id='bitcoin', vs_currency='usd'):
        """
        Fetch current price and market data
        
        Args:
            crypto_id: CoinGecko crypto ID
            vs_currency: Currency to compare against
        
        Returns:
            Dictionary with current price data or None if error
        """
        endpoint = 'simple/price'
        params = {
            'ids': crypto_id,
            'vs_currencies': vs_currency,
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true'
        }

        # Use utility function to build URL with API key
        url = get_coingecko_url(endpoint, params)
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            return {
                'price': data[crypto_id][vs_currency],
                'market_cap': data[crypto_id][f'{vs_currency}_market_cap'],
                'volume_24h': data[crypto_id][f'{vs_currency}_24h_vol'],
                'change_24h': data[crypto_id][f'{vs_currency}_24h_change']
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_data(self, df, filename):
        """
        Save DataFrame to CSV
        
        Args:
            df: DataFrame to save
            filename: Name of the file
        """
        filepath = os.path.join(self.raw_data_path, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename):
        """
        Load DataFrame from CSV
        
        Args:
            filename: Name of the file to load
        
        Returns:
            DataFrame or None if file not found
        """
        filepath = os.path.join(self.raw_data_path, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            print(f"Data loaded from {filepath}")
            return df
        else:
            print(f"File {filepath} not found")
            return None
    
    def fetch_multiple_cryptos(self, crypto_list=None, days=None):
        """
        Fetch data for multiple cryptocurrencies
        
        Args:
            crypto_list: List of crypto symbols (default: from config)
            days: Number of days of data (default: from config)
        """
        if crypto_list is None:
            # Use cryptocurrencies from config and convert to CoinGecko IDs
            crypto_list = [get_crypto_id(symbol) for symbol in CRYPTOCURRENCIES]
        
        if days is None:
            days = HISTORY_DAYS
        
        for crypto in crypto_list:
            print(f"\nFetching data for {crypto}...")
            df = self.fetch_historical_data(crypto, days=days)
            
            if df is not None:
                filename = f"{crypto}_raw.csv"
                self.save_data(df, filename)
            
            # Be respectful to API rate limits
            time.sleep(1.5)
        
        print(f"\nCompleted fetching data for {len(crypto_list)} cryptocurrencies")


# Crypto ID mapping
CRYPTO_MAPPING = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'BNB': 'binancecoin'
}


def get_crypto_id(symbol):
    """
    Convert crypto symbol to CoinGecko ID
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        CoinGecko ID
    """
    return CRYPTO_MAPPING.get(symbol.upper(), symbol.lower())


if __name__ == "__main__":
    # Example usage
    print("=== Crypto Data Fetcher ===\n")
    
    # Initialize fetcher
    fetcher = CryptoDataFetcher()
    
    # Fetch Bitcoin data
    print(f"Fetching {HISTORY_DAYS} days of Bitcoin data...")
    btc_data = fetcher.fetch_historical_data('bitcoin')
    if btc_data is not None:
        fetcher.save_data(btc_data, 'btc_raw.csv')
        print(f"\nData shape: {btc_data.shape}")
        print(f"Date range: {btc_data['date'].min()} to {btc_data['date'].max()}")
    
    # Fetch current price
    print("\nFetching current Bitcoin price...")
    current = fetcher.fetch_current_price('bitcoin')
    if current:
        print(f"Current BTC Price: ${current['price']:,.2f}")
        print(f"Market Cap: ${current['market_cap']:,.0f}")
        print(f"24h Volume: ${current['volume_24h']:,.0f}")
        print(f"24h Change: {current['change_24h']:.2f}%")
    
    # Fetch multiple cryptocurrencies from config
    print("\n" + "="*50)
    print(f"Fetching data for all configured cryptocurrencies: {CRYPTOCURRENCIES}")
    print("="*50)
    # to fetch all:
    # fetcher.fetch_multiple_cryptos()