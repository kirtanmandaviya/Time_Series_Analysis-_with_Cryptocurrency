"""
Data Preprocessing
Prepares cryptocurrency data for prediction using pre-trained models
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from dotenv import load_dotenv

load_dotenv()

# Import from our pipeline configuration
from src.constants.prediction_pipeline import (
    TARGET_COLUMN,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAVED_MODEL_DIR
)
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.utils import ensure_dir_exists


class DataPreprocessor:
    """
    Handles data preprocessing for inference with pre-trained models
    
    Key responsibilities:
    - Clean incoming data
    - Apply same transformations as training
    - Prepare data in format expected by models
    """
    
    def __init__(self):
        """Initialize preprocessor with configuration"""
        self.raw_data_path = RAW_DATA_DIR
        self.processed_data_path = PROCESSED_DATA_DIR
        self.target_column = TARGET_COLUMN
        
        # Scaler will be loaded from saved file
        self.scaler = None
        
        # Create directories if they don't exist
        ensure_dir_exists(self.raw_data_path)
        ensure_dir_exists(self.processed_data_path)
        ensure_dir_exists(SAVED_MODEL_DIR)
        
        logging.info("DataPreprocessor initialized for inference")
    
    def clean_data(self, df):
        """
        Clean raw cryptocurrency data
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        try:
            logging.info("Starting data cleaning process")
            
            df_clean = df.copy()
            
            # Remove duplicates based on date
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=['date'], keep='last')
            duplicates_removed = initial_rows - len(df_clean)
            
            if duplicates_removed > 0:
                logging.info(f"Removed {duplicates_removed} duplicate rows")
            
            # Sort by date
            df_clean = df_clean.sort_values('date').reset_index(drop=True)
            
            # Handle missing values
            missing_counts = df_clean.isnull().sum()
            if missing_counts.any():
                logging.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
                df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
                logging.info("Missing values filled using forward/backward fill")
            
            # Remove any remaining NaN rows
            df_clean = df_clean.dropna()
            
            # Validate data types
            df_clean['date'] = pd.to_datetime(df_clean['date'])
            
            # Ensure numeric columns are float
            numeric_cols = ['price', 'market_cap', 'volume']
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Remove invalid values (negative prices)
            df_clean = df_clean[df_clean['price'] > 0]
            
            logging.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
            
            return df_clean
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def handle_outliers(self, df, column='price', method='iqr', threshold=3):
        """
        Handle outliers in the data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        column : str
            Column to check for outliers
        method : str
            'iqr' or 'zscore'
        threshold : float
            Threshold for outlier detection
        
        Returns:
        --------
        pd.DataFrame : Data with outliers handled
        """
        df_out = df.copy()
        
        if method == 'iqr':
            Q1 = df_out[column].quantile(0.25)
            Q3 = df_out[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df_out[column] < lower_bound) | (df_out[column] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df_out[column] - df_out[column].mean()) / df_out[column].std())
            outliers = z_scores > threshold
        
        print(f"Number of outliers detected in {column}: {outliers.sum()}")
        
        # Replace outliers with interpolated values
        df_out.loc[outliers, column] = np.nan
        df_out[column] = df_out[column].interpolate(method='linear')
        
        return df_out
    
    
    def add_time_features(self, df):
        """
        Extract time-based features from date column
        Essential for model predictions
        
        Args:
            df: Input DataFrame with 'date' column
        
        Returns:
            DataFrame with time features
        """
        try:
            logging.info("Adding time-based features")
            
            df_time = df.copy()
            df_time['date'] = pd.to_datetime(df_time['date'])
            
            # Basic time components
            df_time['year'] = df_time['date'].dt.year
            df_time['month'] = df_time['date'].dt.month
            df_time['day'] = df_time['date'].dt.day
            df_time['day_of_week'] = df_time['date'].dt.dayofweek
            df_time['day_of_year'] = df_time['date'].dt.dayofyear
            df_time['week_of_year'] = df_time['date'].dt.isocalendar().week
            df_time['quarter'] = df_time['date'].dt.quarter
            
            # Weekend flag
            df_time['is_weekend'] = df_time['day_of_week'].isin([5, 6]).astype(int)
            
            logging.info("Time features added successfully")
            
            return df_time
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def calculate_returns(self, df, column='price'):
        """
        Calculate returns (needed by models)
        
        Args:
            df: Input DataFrame
            column: Price column name
        
        Returns:
            DataFrame with return columns
        """
        try:
            logging.info("Calculating returns")
            
            df_returns = df.copy()
            
            # Daily returns
            df_returns['daily_return'] = df_returns[column].pct_change()
            
            # Log returns
            df_returns['log_return'] = np.log(df_returns[column] / df_returns[column].shift(1))
            
            return df_returns
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def normalize_data(self, df, columns=['price', 'volume'], method='minmax'):
        """
        Normalize numerical columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        columns : list
            Columns to normalize
        method : str
            'minmax' or 'standard'
        
        Returns:
        --------
        pd.DataFrame : Data with normalized columns
        tuple : Scalers used for normalization
        """
        df_norm = df.copy()
        scalers = {}
        
        for col in columns:
            if col in df_norm.columns:
                if method == 'minmax':
                    min_val = df_norm[col].min()
                    max_val = df_norm[col].max()
                    df_norm[f'{col}_normalized'] = (df_norm[col] - min_val) / (max_val - min_val)
                    scalers[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
                    
                elif method == 'standard':
                    mean_val = df_norm[col].mean()
                    std_val = df_norm[col].std()
                    df_norm[f'{col}_normalized'] = (df_norm[col] - mean_val) / std_val
                    scalers[col] = {'mean': mean_val, 'std': std_val, 'method': 'standard'}
        
        return df_norm, scalers
    
    def preprocess_pipeline(self, df, outlier_threshold=3):
        """
        Complete preprocessing pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw data
        outlier_threshold : float
            Threshold for outlier detection
        
        Returns:
        --------
        pd.DataFrame : Fully preprocessed data
        """
        print("Starting preprocessing pipeline...")
        
        # Step 1: Clean data
        print("\n1. Cleaning data...")
        df_clean = self.clean_data(df)
        
        # Step 2: Handle outliers
        print("\n2. Handling outliers...")
        df_clean = self.handle_outliers(df_clean, column='price', threshold=outlier_threshold)
        
        # Step 3: Add time features
        print("\n3. Adding time features...")
        df_clean = self.add_time_features(df_clean)
        
        # Step 4: Calculate returns
        print("\n4. Calculating returns...")
        df_clean = self.calculate_returns(df_clean)
        
        print("\nPreprocessing complete!")
        print(f"Final data shape: {df_clean.shape}")
        
        return df_clean
    
    def save_processed_data(self, df, filename):
        """Save processed data to CSV"""
        filepath = os.path.join(self.processed_data_path, filename)
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filename):
        """Load processed data from CSV"""
        filepath = os.path.join(self.processed_data_path, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            print(f"File {filepath} not found")
            return None

if __name__ == "__main__":
    # Example usage
    from src.components.data_ingestion import CryptoDataFetcher
    
    # Fetch data
    fetcher = CryptoDataFetcher()
    btc_data = fetcher.load_data('btc_raw.csv')
    
    if btc_data is not None:
        # Preprocess
        preprocessor = DataPreprocessor()
        btc_cleaned = preprocessor.preprocess_pipeline(btc_data)
        
        # Save
        preprocessor.save_processed_data(btc_cleaned, 'btc_cleaned.csv')
        
        print("\nSample of cleaned data:")
        print(btc_cleaned.head())