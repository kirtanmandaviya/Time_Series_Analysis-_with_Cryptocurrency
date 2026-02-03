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

# Directory Paths
SAVED_MODEL_DIR: str = "final_model"
RAW_DATA_DIR: str = "data/raw/"
PROCESSED_DATA_DIR: str = "data/processed/"

