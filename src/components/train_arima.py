import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')


class BitcoinARIMAModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.train_data = None
        self.test_data = None
        self.predictions = None
        
    def load_data(self):
        """Load and preprocess Bitcoin dataset"""
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.set_index('date').sort_index()
        
        for col in ['price', 'market_cap', 'volume']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"Data loaded: {len(self.df)} rows")
        return self.df
    
    # SARIMAX Features
    # def engineer_features(self):
    #     """Create lag and rolling features"""
    #     self.df['price_lag_1'] = self.df['price'].shift(1)
    #     self.df['price_lag_7'] = self.df['price'].shift(7)
    #     self.df['ma_7'] = self.df['price'].rolling(window=7).mean()
    #     self.df['ma_30'] = self.df['price'].rolling(window=30).mean()
    #     print("Features engineered")
        
    def split_data(self, train_ratio=0.8):
        """Split data into train and test sets"""
        price_series = self.df['price'].dropna()
        split_idx = int(len(price_series) * train_ratio)
        self.train_data = price_series[:split_idx]
        self.test_data = price_series[split_idx:]
        print(f"Train: {len(self.train_data)}, Test: {len(self.test_data)}")
        
    def train(self):
        """Train ARIMA model with auto parameter selection"""
        print("Finding optimal parameters...")
        auto_model = auto_arima(self.train_data, 
                                start_p=0, start_q=0,
                                max_p=5, max_q=5,
                                seasonal=False,
                                stepwise=True,
                                suppress_warnings=True,
                                trace=False)
        
        optimal_order = auto_model.order
        print(f"Optimal order: {optimal_order}")
        
        arima_model = ARIMA(self.train_data, order=optimal_order)
        self.model = arima_model.fit()
        print("Model trained successfully")
        
    def evaluate(self):
        """Evaluate model performance"""
        self.predictions = self.model.forecast(steps=len(self.test_data))
        mae = mean_absolute_error(self.test_data, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.test_data, self.predictions))
        
        print(f"\nModel Performance:")
        print(f"MAE: ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        
        return {'mae': mae, 'rmse': rmse}
    
    def save_model(self, filename='arima.pkl'):
        """Save trained model to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filename}")
        
    def plot_results(self, save_path='results/predictions.png'):
        """Plot predictions vs actual"""
        plt.figure(figsize=(15, 6))
        plt.plot(self.train_data.index, self.train_data, label='Training', alpha=0.7)
        plt.plot(self.test_data.index, self.test_data, label='Actual', linewidth=2)
        plt.plot(self.test_data.index, self.predictions, 
                label='Predicted', linewidth=2, linestyle='--')
        plt.title('ARIMA - Bitcoin Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")


def main():
    # Initialize model
    btc_model = BitcoinARIMAModel('data/training_data/btc_extended.csv')
    
    # Load and prepare data
    btc_model.load_data()
    # btc_model.engineer_features()
    btc_model.split_data(train_ratio=0.8)
    
    # Train and evaluate
    btc_model.train()
    metrics = btc_model.evaluate()
    
    # Save model
    btc_model.save_model('arima.pkl')
    
    # Visualize
    # btc_model.plot_results()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
