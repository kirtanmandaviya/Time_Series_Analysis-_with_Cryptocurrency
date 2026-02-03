"""
Feature Engineering 
Creates technical indicators and advanced features for crypto analysis
"""

import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator

class FeatureEngineer:
    def __init__(self):
        pass
    
    def add_moving_averages(self, df, column='price', windows=[7, 30, 90, 200]):
        """
        Add Simple Moving Averages (SMA)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        column : str
            Price column
        windows : list
            List of window sizes
        
        Returns:
        --------
        pd.DataFrame : Data with moving averages
        """
        df_ma = df.copy()
        
        for window in windows:
            df_ma[f'SMA_{window}'] = df_ma[column].rolling(window=window).mean()
        
        return df_ma
    
    def add_exponential_moving_averages(self, df, column='price', spans=[12, 26]):
        """
        Add Exponential Moving Averages (EMA)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        column : str
            Price column
        spans : list
            List of span values
        
        Returns:
        --------
        pd.DataFrame : Data with EMAs
        """
        df_ema = df.copy()
        
        for span in spans:
            df_ema[f'EMA_{span}'] = df_ema[column].ewm(span=span, adjust=False).mean()
        
        return df_ema
    
    def add_bollinger_bands(self, df, column='price', window=20, window_dev=2):
        """
        Add Bollinger Bands
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        column : str
            Price column
        window : int
            Rolling window
        window_dev : int
            Number of standard deviations
        
        Returns:
        --------
        pd.DataFrame : Data with Bollinger Bands
        """
        df_bb = df.copy()
        
        indicator_bb = BollingerBands(
            close=df_bb[column],
            window=window,
            window_dev=window_dev
        )
        
        df_bb['BB_High'] = indicator_bb.bollinger_hband()
        df_bb['BB_Mid'] = indicator_bb.bollinger_mavg()
        df_bb['BB_Low'] = indicator_bb.bollinger_lband()
        df_bb['BB_Width'] = indicator_bb.bollinger_wband()
        df_bb['BB_PctB'] = indicator_bb.bollinger_pband()
        
        return df_bb
    
    def add_rsi(self, df, column='price', window=14):
        """
        Add Relative Strength Index (RSI)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        column : str
            Price column
        window : int
            RSI period
        
        Returns:
        --------
        pd.DataFrame : Data with RSI
        """
        df_rsi = df.copy()
        
        indicator_rsi = RSIIndicator(close=df_rsi[column], window=window)
        df_rsi['RSI'] = indicator_rsi.rsi()
        
        return df_rsi
    
    def add_macd(self, df, column='price', window_slow=26, window_fast=12, window_sign=9):
        """
        Add MACD (Moving Average Convergence Divergence)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        column : str
            Price column
        
        Returns:
        --------
        pd.DataFrame : Data with MACD
        """
        df_macd = df.copy()
        
        indicator_macd = MACD(
            close=df_macd[column],
            window_slow=window_slow,
            window_fast=window_fast,
            window_sign=window_sign
        )
        
        df_macd['MACD'] = indicator_macd.macd()
        df_macd['MACD_Signal'] = indicator_macd.macd_signal()
        df_macd['MACD_Diff'] = indicator_macd.macd_diff()
        
        return df_macd
    
    def add_volatility_features(self, df, column='price', windows=[7, 30]):
        """
        Add volatility features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        column : str
            Price column
        windows : list
            List of rolling windows
        
        Returns:
        --------
        pd.DataFrame : Data with volatility features
        """
        df_vol = df.copy()
        
        # Rolling standard deviation
        for window in windows:
            df_vol[f'Volatility_{window}'] = df_vol['daily_return'].rolling(window=window).std()
        
        # Historical volatility (annualized)
        df_vol['Historical_Volatility'] = df_vol['daily_return'].rolling(window=30).std() * np.sqrt(365)
        
        return df_vol
    
    def add_price_momentum(self, df, column='price', periods=[7, 14, 30]):
        """
        Add price momentum features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        column : str
            Price column
        periods : list
            List of periods for momentum calculation
        
        Returns:
        --------
        pd.DataFrame : Data with momentum features
        """
        df_mom = df.copy()
        
        for period in periods:
            df_mom[f'Momentum_{period}'] = df_mom[column].pct_change(periods=period)
        
        return df_mom
    
    def add_lagged_features(self, df, column='price', lags=[1, 2, 3, 7, 14]):
        """
        Add lagged price features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        column : str
            Column to lag
        lags : list
            List of lag periods
        
        Returns:
        --------
        pd.DataFrame : Data with lagged features
        """
        df_lag = df.copy()
        
        for lag in lags:
            df_lag[f'{column}_lag_{lag}'] = df_lag[column].shift(lag)
        
        return df_lag
    
    def add_rolling_statistics(self, df, column='price', window=30):
        """
        Add rolling statistical features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        column : str
            Column to calculate statistics on
        window : int
            Rolling window size
        
        Returns:
        --------
        pd.DataFrame : Data with rolling statistics
        """
        df_stats = df.copy()
        
        df_stats[f'{column}_rolling_mean'] = df_stats[column].rolling(window=window).mean()
        df_stats[f'{column}_rolling_std'] = df_stats[column].rolling(window=window).std()
        df_stats[f'{column}_rolling_min'] = df_stats[column].rolling(window=window).min()
        df_stats[f'{column}_rolling_max'] = df_stats[column].rolling(window=window).max()
        df_stats[f'{column}_rolling_median'] = df_stats[column].rolling(window=window).median()
        
        return df_stats
    
    def add_volume_features(self, df, volume_col='volume', price_col='price'):
        """
        Add volume-based features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        volume_col : str
            Volume column name
        price_col : str
            Price column name
        
        Returns:
        --------
        pd.DataFrame : Data with volume features
        """
        df_vol = df.copy()
        
        # Volume moving averages
        df_vol['Volume_MA_7'] = df_vol[volume_col].rolling(window=7).mean()
        df_vol['Volume_MA_30'] = df_vol[volume_col].rolling(window=30).mean()
        
        # Volume change
        df_vol['Volume_Change'] = df_vol[volume_col].pct_change()
        
        # Price-volume ratio
        df_vol['Price_Volume_Ratio'] = df_vol[price_col] / (df_vol[volume_col] + 1)
        
        return df_vol
    
    def feature_engineering_pipeline(self, df):
        """
        Complete feature engineering pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed data
        
        Returns:
        --------
        pd.DataFrame : Data with all engineered features
        """
        print("Starting feature engineering pipeline...")
        
        df_features = df.copy()
        
        # Moving averages
        print("Adding moving averages...")
        df_features = self.add_moving_averages(df_features)
        df_features = self.add_exponential_moving_averages(df_features)
        
        # Bollinger Bands
        print("Adding Bollinger Bands...")
        df_features = self.add_bollinger_bands(df_features)
        
        # RSI
        print("Adding RSI...")
        df_features = self.add_rsi(df_features)
        
        # MACD
        print("Adding MACD...")
        df_features = self.add_macd(df_features)
        
        # Volatility features
        print("Adding volatility features...")
        df_features = self.add_volatility_features(df_features)
        
        # Momentum features
        print("Adding momentum features...")
        df_features = self.add_price_momentum(df_features)
        
        # Volume features
        print("Adding volume features...")
        if 'volume' in df_features.columns:
            df_features = self.add_volume_features(df_features)
        
        # Rolling statistics
        print("Adding rolling statistics...")
        df_features = self.add_rolling_statistics(df_features)
        
        # Remove rows with NaN created by feature engineering
        initial_rows = len(df_features)
        df_features = df_features.dropna()
        rows_removed = initial_rows - len(df_features)
        
        print(f"\nFeature engineering complete!")
        print(f"Total features: {len(df_features.columns)}")
        print(f"Rows removed due to NaN: {rows_removed}")
        print(f"Final shape: {df_features.shape}")
        
        return df_features

if __name__ == "__main__":
    from src.components.data_preprocessing import DataPreprocessor
    
    # Load preprocessed data
    preprocessor = DataPreprocessor()
    btc_data = preprocessor.load_processed_data('btc_cleaned.csv')
    
    if btc_data is not None:
        # Engineer features
        engineer = FeatureEngineer()
        btc_features = engineer.feature_engineering_pipeline(btc_data)
        
        # Save
        preprocessor.save_processed_data(btc_features, 'btc_features.csv')
        
        print("\nSample of data with features:")
        print(btc_features.head())
        print("\nColumns:")
        print(btc_features.columns.tolist())