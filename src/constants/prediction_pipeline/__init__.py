import os
import yaml
from pathlib import Path
import sys


"""
Crypto Prediction Pipeline - Constants and Configuration
"""

# LOAD CONFIGURATION FROM YAML

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / 'config.yaml'
    
    if not config_path.exists():
        config_path = Path('config.yaml')
    
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()


# API CONFIGURATION 

COINGECKO_BASE_URL: str = config['api']['coingecko_base']
API_KEY: str = os.getenv("COINGECKO_API_KEY", "")

# DATA CONFIGURATION 

TARGET_COLUMN: str = config['data']['defaults']['target_column']
CRYPTOCURRENCIES: list = config['data']['cryptocurrencies']
DEFAULT_SYMBOL: str = config['data']['defaults']['symbol']
HISTORY_DAYS: int = config['data']['defaults']['history_days']

# MODEL HYPERPARAMETERS 

# ARIMA
ARIMA_ORDER: tuple = tuple(config['models']['arima']['order'])
ARIMA_SEASONAL_ORDER: tuple = tuple(config['models']['arima']['seasonal_order'])

# LSTM
LSTM_SEQUENCE_LENGTH: int = config['models']['lstm']['sequence_length']
LSTM_EPOCHS: int = config['models']['lstm']['epochs']
LSTM_BATCH_SIZE: int = config['models']['lstm']['batch_size']
LSTM_HIDDEN_UNITS: list = config['models']['lstm']['hidden_units']
LSTM_DROPOUT_RATE: float = config['models']['lstm']['dropout_rate']
LSTM_LEARNING_RATE: float = config['models']['lstm']['learning_rate']

# Prophet
PROPHET_CHANGEPOINT_PRIOR_SCALE: float = config['models']['prophet']['changepoint_prior_scale']
PROPHET_SEASONALITY_PRIOR_SCALE: float = config['models']['prophet']['seasonality_prior_scale']
PROPHET_FEATURES: dict = config['models']['prophet']['features']

# DASHBOARD CONFIGURATION 

DASHBOARD_THEME: str = config['dashboard']['theme']
DASHBOARD_REFRESH_INTERVAL: int = config['dashboard']['refresh_interval']
DASHBOARD_FORECAST_HORIZON: int = config['dashboard']['forecast_horizon']
DASHBOARD_MOVING_AVERAGES: list = config['dashboard']['indicators']['moving_averages']
STREAMLIT_PAGE_TITLE: str = config['dashboard']['streamlit']['page_title']
STREAMLIT_PAGE_ICON: str = config['dashboard']['streamlit']['page_icon']
STREAMLIT_LAYOUT: str = config['dashboard']['streamlit']['layout']

# FILE PATHS AND NAMES 

# Pipeline Settings
PIPELINE_NAME: str = "CryptoForecast"
ARTIFACT_DIR: str = "Artifacts"

# File Names
FILE_NAME: str = "btc_live_data.csv"
MODEL_FILE_NAME: str = "model.pkl"
LSTM_MODEL_FILE_NAME: str = "lstm_model.h5"
PREPROCESSING_OBJECT_FILE_NAME: str = "scaler.pkl"
PREDICTION_FILE_NAME: str = "prediction.csv"

# Directory Paths
SAVED_MODEL_DIR: str = "final_model"
RAW_DATA_DIR: str = "data/raw/"
PROCESSED_DATA_DIR: str = "data/processed/"

# Data Ingestion
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.0

# Data Validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# Data Transformation
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Model Prediction
MODEL_PREDICTION_DIR_NAME: str = "model_prediction"