import os
from urllib.parse import urlencode


def get_coingecko_url(endpoint: str, params: dict = None) -> str:
    """
    Construct CoinGecko API URL with API key
    
    Args:
        endpoint: API endpoint (e.g., 'coins/markets')
        params: Query parameters
    
    Returns:
        Complete URL with API key

    """
    from src.constants.prediction_pipeline import COINGECKO_BASE_URL, API_KEY
    
    base_params = {}
    if API_KEY:
        base_params['x_cg_demo_api_key'] = API_KEY
    
    if params:
        base_params.update(params)
    
    url = f"{COINGECKO_BASE_URL}{endpoint}"
    if base_params:
        url += f"?{urlencode(base_params)}"
    
    return url


def ensure_dir_exists(directory: str) -> str:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to directory
    
    Returns:
        Path to the directory
    """
    os.makedirs(directory, exist_ok=True)
    return directory