"""
Cryptocurrency Prediction Pipeline
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import all components
from src.components.data_ingestion import CryptoDataFetcher
from src.components.data_preprocessing import DataPreprocessor
from src.components.feature_engineering import FeatureEngineer
from src.components.model_evaluate import ModelEvaluator
from src.components.prophet_model import CryptoProphetModel

# Import configuration
from src.constants.prediction_pipeline import (
    SAVED_MODEL_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from src.utils.utils import ensure_dir_exists
from src.exception.exception import CustomException
from src.logging.logger import logging


class CompletePredictionPipeline:
    """
    FINAL Prediction Pipeline
    """
    
    def __init__(self):
       
        self.fetcher = CryptoDataFetcher()
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        
        # Model placeholders
        self.prophet_model = None
        self.arima_model = None
        self.lstm_model = None
        self.lstm_scaler = None
        
        # Data placeholders
        self.df_raw = None
        self.df_processed = None
        self.df_features = None
        self.predictions = {}
        
        # Create directories
        ensure_dir_exists(SAVED_MODEL_DIR)
        ensure_dir_exists(RAW_DATA_DIR)
        ensure_dir_exists(PROCESSED_DATA_DIR)
    
    def step1_fetch_data(self, crypto_id='bitcoin', days=365):
        """Step 1: Fetch data"""
        logging.info("Initiate Data Fetching.")
        self.df_raw = self.fetcher.fetch_historical_data(crypto_id, days=days)
        if self.df_raw is not None:
            self.fetcher.save_data(self.df_raw, f'{crypto_id}_raw.csv')
            print(f"Data fetched: {self.df_raw.shape}")
        logging.info("Data Fetching Completed.")
        return self.df_raw
    def step2_preprocess_data(self):
        """Step 2: Preprocess"""
        
        logging.info("Initiate Data Preprocessing.")
        self.df_processed = self.preprocessor.preprocess_pipeline(self.df_raw)
        print(f"Preprocessing complete: {self.df_processed.shape}")
        logging.info("Data Preprocessing Completed.")
        return self.df_processed
    
    def step3_engineer_features(self):
        """Step 3: Feature engineering"""
    
        logging.info("Initiate Feature Engineering.")
        self.df_features = self.engineer.feature_engineering_pipeline(self.df_processed)
        
        # Add LSTM-specific features
        print("Adding LSTM-specific features...")
        self.df_features = self._add_lstm_features(self.df_features)
        
        self.preprocessor.save_processed_data(self.df_features, 'crypto_features.csv')
        print(f"Feature engineering complete: {self.df_features.shape}")
        logging.info("Feature Engineering Completed.")
        return self.df_features
    
    def _add_lstm_features(self, df):
        """Add LSTM required features"""
        df_lstm = df.copy()
        
        if 'price_lag_1' not in df_lstm.columns:
            df_lstm['price_lag_1'] = df_lstm['price'].shift(1)
        if 'price_ma_7' not in df_lstm.columns:
            df_lstm['price_ma_7'] = df_lstm['price'].rolling(window=7).mean()
        if 'price_ma_24' not in df_lstm.columns:
            df_lstm['price_ma_24'] = df_lstm['price'].rolling(window=24).mean()
        if 'volatility_24' not in df_lstm.columns:
            df_lstm['volatility_24'] = df_lstm['price'].rolling(window=24).std()
        
        df_lstm = df_lstm.dropna()
        print(f"LSTM features ready: {len(df_lstm)} rows")
        return df_lstm
    
    def step4_load_models(self):
        """Step 4: Load models with error handling"""
        
        loaded_models = {}
        
        logging.info("Initiate Models Loading")
        print("\n[1/3] Loading Prophet Model")
        try:
            logging.info("Loading Prophet Model")
            prophet_paths = [
                os.path.join(SAVED_MODEL_DIR, 'crypto_prophet_model.pkl'),
                'final_model1/crypto_prophet_model.pkl',
                'final_model/crypto_prophet_model.pkl'
            ]
            
            for path in prophet_paths:
                if os.path.exists(path):
                    self.prophet_model = CryptoProphetModel.load(path)
                    loaded_models['Prophet'] = self.prophet_model
                    print(f"Prophet loaded from: {path}")
                    break
            else:
                print("Prophet model not found")
        except Exception as e:
            raise CustomException(e, sys)
        
        logging.info("Prophet Model Loaded Successfully.")
        
        print("\n[2/3] Loading ARIMA Model")
        try:
            logging.info("Loading Arima Model")
            arima_paths = [
                os.path.join(SAVED_MODEL_DIR, 'arima.pkl'),
                'final_model1/arima.pkl',
                'final_model/arima.pkl'
            ]
            
            for path in arima_paths:
                if os.path.exists(path):
                    print(f"Attempting to load from: {path}")
                    
                    # Load with error handling
                    try:
                        self.arima_model = joblib.load(path)
                        
                        # TEST: Try a small forecast to verify it works
                        test_forecast = self.arima_model.forecast(steps=1)
                        
                        loaded_models['ARIMA'] = self.arima_model
                        print(f"ARIMA loaded and verified from: {path}")
                        break
                        
                    except Exception as load_error:
                        print(f"  Error loading ARIMA: {str(load_error)}")
                        print(f"  This ARIMA model may have compatibility issues.")
                        print(f"  Skipping ARIMA predictions...")
                        self.arima_model = None
                        break
            
        except Exception as e:
            raise CustomException(e, sys)
        logging.info("Arima Model Loaded Successfully.")
        
        print("\n[3/3] Loading LSTM Model")
        try:
            logging.info("Loading LSTM Model")
            lstm_paths = [
                (os.path.join(SAVED_MODEL_DIR, 'lstm_model.h5'), 
                 os.path.join(SAVED_MODEL_DIR, 'lstm_scaler.pkl')),
                ('final_model1/lstm_model.h5', 'final_model1/lstm_scaler.pkl'),
                ('final_model/lstm_model.h5', 'final_model/lstm_scaler.pkl')
            ]
            
            for model_path, scaler_path in lstm_paths:
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    print(f"Attempting to load from: {model_path}")
                    
                    try:
                        import tensorflow as tf
                        
                        self.lstm_model = tf.keras.models.load_model(
                            model_path,
                            compile=False  # Skip compilation to avoid parameter errors
                        )
                        
                        self.lstm_scaler = joblib.load(scaler_path)
                        loaded_models['LSTM'] = (self.lstm_model, self.lstm_scaler)
                        
                        print(f"LSTM model loaded from: {model_path}")
                        print(f"LSTM scaler loaded from: {scaler_path}")
                        logging.info("LSTM Model Loaded Successfully.")
                        break
                        
                    except Exception as load_error:
                        print(f"Error loading LSTM: {str(load_error)}")
                        
                        # Try alternative: Load 
                        print(f"Attempting alternative loading method...")
                        try:
                            # If you know the architecture, you can rebuild and load weights
                            print(f" LSTM model incompatible with current Keras version")
                            print(f" Skipping LSTM predictions...")
                            self.lstm_model = None
                            break
                        except:
                            pass
            
        except Exception as e:
            raise CustomException(e, sys)
       
        
        print(f"\n{'='*80}")
        print(f"MODELS LOADED: {len(loaded_models)}/3")
        for model_name in loaded_models.keys():
            print(f"{model_name}")
        print(f"{'='*80}")
        
        return loaded_models
    
    def step5_generate_predictions(self, forecast_steps=30):
        """Step 5: Generate predictions"""
        print("STEP 5: GENERATING PREDICTIONS")
        print(f"Forecast horizon: {forecast_steps} steps\n")
        
        logging.info("Initializing predictions...")
        self.predictions = {}
        
        # Prophet
        if self.prophet_model is not None:
            print("[1/3] Prophet Prediction")
            try:
                prophet_data = pd.DataFrame({
                    'ds': pd.to_datetime(self.df_processed['date']),
                    'y': pd.to_numeric(self.df_processed['price'], errors='coerce')
                }).dropna()
                
                print(f"  Data shape: {prophet_data.shape}")
                
                future = self.prophet_model.make_future_dataframe(periods=forecast_steps, freq='D')
                forecast = self.prophet_model.predict(future)
                prophet_predictions = forecast['yhat'].tail(forecast_steps).values
                
                self.predictions['Prophet'] = prophet_predictions
                print(f"Prophet: {len(prophet_predictions)} predictions")
                print(f"Mean: ${np.mean(prophet_predictions):,.2f}\n")
                
            except Exception as e:
                raise CustomException(e,sys)
        else:
            print("[1/3] Prophet - Not loaded, skipping\n")
        
        # ARIMA
        if self.arima_model is not None:
            print("[2/3] ARIMA Prediction")
            try:
                # Clean price data
                price_series = pd.Series(
                    pd.to_numeric(self.df_processed['price'], errors='coerce')
                ).dropna().reset_index(drop=True)
                
                print(f"  Data: {len(price_series)} points, dtype: {price_series.dtype}")
                
                arima_predictions = self.arima_model.forecast(steps=forecast_steps)
                
                if isinstance(arima_predictions, pd.Series):
                    arima_predictions = arima_predictions.values
                
                self.predictions['ARIMA'] = arima_predictions
                print(f"ARIMA: {len(arima_predictions)} predictions")
                print(f"Mean: ${np.mean(arima_predictions):,.2f}\n")
                
            except Exception as e:
                raise CustomException(e, sys)
        else:
            print("[2/3] ARIMA - Not loaded, skipping\n")
        
        # LSTM
        if self.lstm_model is not None and self.lstm_scaler is not None:
            print("[3/3] LSTM Prediction")
            try:
                feature_columns = [
                    'price',
                    'SMA_7', 'SMA_30', 'SMA_90', 'SMA_200',
                    'EMA_12', 'EMA_26',
                    'BB_High', 'BB_Mid', 'BB_Low', 'BB_Width', 'BB_PctB',
                    'RSI',
                    'MACD', 'MACD_Signal', 'MACD_Diff',
                    'Momentum_7', 'Momentum_14', 'Momentum_30',
                    'Volatility_7', 'Volatility_30',
                    'Historical_Volatility'
                ]
                
                features = self.df_features[feature_columns].values
                features_scaled = self.lstm_scaler.transform(features)
                
                print(f"Features: {features_scaled.shape}")
                
                lookback = 24
                if len(features_scaled) < lookback:
                    raise ValueError(f"Need {lookback} points, have {len(features_scaled)}")
                
                lstm_predictions = []
                current_sequence = features_scaled[-lookback:].reshape(1, lookback, len(feature_columns))
                
                for i in range(forecast_steps):
                    next_pred = self.lstm_model.predict(current_sequence, verbose=0)
                    lstm_predictions.append(next_pred[0, 0])
                    
                    new_features = current_sequence[0, -1, :].copy()
                    new_features[0] = next_pred[0, 0]
                    
                    current_sequence = np.append(
                        current_sequence[:, 1:, :],
                        new_features.reshape(1, 1, len(feature_columns)),
                        axis=1
                    )
                
                lstm_predictions = np.array(lstm_predictions)
                
                # Inverse transform
                dummy = np.zeros((len(lstm_predictions), len(feature_columns)))
                dummy[:, 0] = lstm_predictions
                lstm_predictions_unscaled = self.lstm_scaler.inverse_transform(dummy)[:, 0]
                
                self.predictions['LSTM'] = lstm_predictions_unscaled
                print(f"LSTM: {len(lstm_predictions_unscaled)} predictions")
                print(f"Mean: ${np.mean(lstm_predictions_unscaled):,.2f}\n")
                
            except Exception as e:
                raise CustomException(e, sys)
        else:
            print("[3/3] LSTM - Not loaded, skipping\n")
        
        print(f"{'='*80}")
        print(f"PREDICTIONS COMPLETE: {len(self.predictions)}/3 models succeeded")
        print(f"{'='*80}")
        
        logging.info("Models Prediction Completed.")
        return self.predictions
    
    def step6_save_predictions(self, crypto_id='bitcoin'):
        """Step 6: Save predictions"""
        print("STEP 6: SAVING PREDICTIONS")
        
        if not self.predictions:
            print("No predictions to save")
            return None
        
        df_pred = pd.DataFrame(self.predictions)
        
        last_date = self.df_processed['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(next(iter(self.predictions.values()))),
            freq='D'
        )
        df_pred.insert(0, 'date', future_dates)
        
        # Ensemble
        model_cols = [col for col in df_pred.columns if col != 'date']
        if len(model_cols) > 0:
            df_pred['Ensemble'] = df_pred[model_cols].mean(axis=1)
        
        # Save
        output_file = os.path.join(PROCESSED_DATA_DIR, f'{crypto_id}_predictions.csv')
        df_pred.to_csv(output_file, index=False)
        
        print(f"Saved to: {output_file}")
        print(f"Shape: {df_pred.shape}")
        print(f"\nSample predictions (first 5 days):")
        print(df_pred.head().to_string(index=False))
        
        return df_pred
    
    def run_complete_pipeline(self, crypto_id='bitcoin', days=365, forecast_steps=30):
        """Execute complete pipeline"""
        print("\n" + "="*80)
        print("STARTING COMPLETE PREDICTION PIPELINE")
        print("="*80)

        try:
            self.step1_fetch_data(crypto_id, days)
            self.step2_preprocess_data()
            self.step3_engineer_features()
            self.step4_load_models()
            self.step5_generate_predictions(forecast_steps)
            df_predictions = self.step6_save_predictions(crypto_id)
            
            # Summary
            print("\n" + "="*80)
            print("PIPELINE COMPLETED!")
            print("="*80)
            
            print(f"\nResults:")
            print(f"Models succeeded: {list(self.predictions.keys())}")
            print(f"Total models: {len(self.predictions)}/3")
            
            if len(self.predictions) == 0:
                print("\n WARNING: No models produced predictions")
                print("  Possible issues:")
                print("  1. ARIMA: Model has dtype compatibility issues - may need retraining")
                print("  2. LSTM: Keras version mismatch - model trained with older Keras")
                print("\n  Recommendations:")
                print("  - Retrain models with current library versions")
                print("  - Or use Prophet predictions only")
            
            return self.df_features, self.predictions
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CRYPTOCURRENCY PRICE PREDICTION PIPELINE")
    print("="*80)
    
    pipeline = CompletePredictionPipeline()
    df_features, predictions = pipeline.run_complete_pipeline(
        crypto_id='bitcoin',
        days=365,
        forecast_steps=30
    )
    
    if predictions:
        print("\n" + "="*80)
        print("FINAL PREDICTIONS")
        print("="*80)
        
        for model_name, preds in predictions.items():
            print(f"\n{model_name}:")
            print(f"  First 5: {preds[:5]}")
            print(f"  Mean: ${np.mean(preds):,.2f}")
            print(f"  Range: ${np.min(preds):,.2f} - ${np.max(preds):,.2f}")