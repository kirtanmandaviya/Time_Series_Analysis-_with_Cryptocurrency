"""
LSTM Bitcoin Price Prediction - Complete Training Pipeline
This script contains all necessary functions and classes for training and saving the LSTM model
Run this file in VS Code to train the model and generate all deliverables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math


class BitcoinLSTMPipeline:
    """
    Complete pipeline for Bitcoin price prediction using LSTM
    """
    
    def __init__(self, data_path='btc_features.csv'):
        """Initialize the pipeline with data path"""
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.feature_columns = [
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

        self.lookback = 24
        
        # For storing train/test data
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_data = None
        self.test_data = None
        
        print("="*70)
        print("LSTM Bitcoin Price Prediction Pipeline Initialized")
        print("="*70)
    
    def load_and_explore_data(self):
        """Step 1: Load and explore the dataset"""
        print("\n[STEP 1] Loading and Exploring Data...")
        
        # Load dataset
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nColumns: {self.df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nData Info:")
        self.df.info()
        print(f"\nMissing values: {self.df.isnull().sum().sum()}")
        
        return self.df
    
    def clean_data(self):
        """Step 2: Clean and preprocess data"""
        print("\n[STEP 2] Cleaning Data...")
        
        # Convert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Sort by date
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Check duplicates
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Date range
        print(f"Date Range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Total records: {len(self.df)}")
        
        # return self.df
        self.df_clean = self.df.copy()

        return self.df_clean
    
    def feature_engineering(self):
        """Step 3: Create features for LSTM"""
        print("\n[STEP 3] Feature Engineering...")
        
        # Time-based features
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['day'] = self.df['date'].dt.day
        self.df['month'] = self.df['date'].dt.month
        
        # Lag features
        self.df['price_lag_1'] = self.df['price'].shift(1)
        self.df['price_lag_24'] = self.df['price'].shift(24)
        
        # Rolling statistics (moving averages)
        self.df['price_ma_7'] = self.df['price'].rolling(window=7).mean()
        self.df['price_ma_24'] = self.df['price'].rolling(window=24).mean()
        
        # Price change features
        self.df['price_change'] = self.df['price'].diff()
        self.df['price_change_pct'] = self.df['price'].pct_change() * 100
        
        # Volatility
        self.df['volatility_24'] = self.df['price'].rolling(window=24).std()
        
        print(f"Features created. New shape: {self.df.shape}")
        print(f"Feature columns: {self.df.columns.tolist()}")
        
        # Drop NaN values
        self.df_clean = self.df.dropna().reset_index(drop=True)
        print(f"\nAfter dropping NaN: {self.df_clean.shape}")
        print(f"Rows removed: {len(self.df) - len(self.df_clean)}")
        
        return self.df_clean
    
    def visualize_data(self):
        """Step 4: Visualize the data"""
        print("\n[STEP 4] Visualizing Data...")
        
        # Price trend
        plt.figure(figsize=(14, 5))
        plt.plot(self.df_clean['date'], self.df_clean['price'], linewidth=1)
        plt.title('Bitcoin Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('price_trend.png', dpi=300, bbox_inches='tight')
        print("Saved: price_trend.png")
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation = self.df_clean[['price', 'price_lag_1', 'price_lag_24', 
                                      'price_ma_7', 'price_ma_24', 'volatility_24']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved: correlation_heatmap.png")
        plt.close()
    
    def prepare_train_test_split(self, train_size=0.8):
        """Step 5: Split data into train and test sets"""
        print("\n[STEP 5] Preparing Train/Test Split...")
        
        # Select features
        df_model = self.df_clean[['date'] + self.feature_columns].copy()
        
        # Split
        train_idx = int(len(df_model) * train_size)
        self.train_data = df_model[:train_idx]
        self.test_data = df_model[train_idx:]
        
        print(f"Total data: {len(df_model)} records")
        print(f"Training set: {len(self.train_data)} records ({train_size*100:.0f}%)")
        print(f"Test set: {len(self.test_data)} records ({(1-train_size)*100:.0f}%)")
        print(f"Train date range: {self.train_data['date'].min()} to {self.train_data['date'].max()}")
        print(f"Test date range: {self.test_data['date'].min()} to {self.test_data['date'].max()}")
        
        return self.train_data, self.test_data
    
    def scale_data(self):
        """Step 6: Scale the features"""
        print("\n[STEP 6] Scaling Data...")
        
        # Extract features (without date)
        train_features = self.train_data[self.feature_columns].values
        test_features = self.test_data[self.feature_columns].values
        
        # Fit scaler on training data only
        self.scaler.fit(train_features)
        
        # Transform both
        train_scaled = self.scaler.transform(train_features)
        test_scaled = self.scaler.transform(test_features)
        
        print(f"Original price range: ${train_features[:, 0].min():.2f} - ${train_features[:, 0].max():.2f}")
        print(f"Scaled price range: {train_scaled[:, 0].min():.4f} - {train_scaled[:, 0].max():.4f}")
        
        # Save scaler
        joblib.dump(self.scaler, 'lstm_scaler.pkl')
        print("✓ Scaler saved as: lstm_scaler.pkl")
        
        return train_scaled, test_scaled
    
    def create_sequences(self, data, lookback=24):
        """Create sequences for LSTM input"""
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])  # Predict price (first column)
        
        return np.array(X), np.array(y)
    
    def prepare_sequences(self, train_scaled, test_scaled):
        """Step 7: Create LSTM sequences"""
        print("\n[STEP 7] Creating LSTM Sequences...")
        
        self.X_train, self.y_train = self.create_sequences(train_scaled, self.lookback)
        self.X_test, self.y_test = self.create_sequences(test_scaled, self.lookback)
        
        print(f"X_train shape: {self.X_train.shape} (samples, timesteps, features)")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        
        print(f"\nTraining samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def build_model(self):
        """Step 8: Build LSTM model architecture"""
        print("\n[STEP 8] Building LSTM Model...")
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        self.model = Sequential([
            # First LSTM layer
            LSTM(50, return_sequences=True, input_shape=(self.lookback, len(self.feature_columns))),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(25, activation='relu'),
            Dense(1)  # Output: predicted price
        ])
        
        # Compile
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        # Display architecture
        print("\nModel Architecture:")
        self.model.summary()
        print(f"\nTotal parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.2):
        """Step 9: Train the LSTM model"""
        print("\n[STEP 9] Training Model...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Validation split: {validation_split}")
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train
        print("\nTraining started...")
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("\nTraining completed!")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE During Training')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: training_history.png")
        plt.close()
    
    def save_model(self, filename='lstm_model.h5'):
        """Step 10: Save the trained model"""
        print(f"\n[STEP 10] Saving Model...")
        
        try:
            self.model.save(filename)
            print(f"Model saved as: {filename}")
        except Exception as e:
            print(f"Error saving .h5 file: {e}")
            print("Attempting to save as .keras format...")
            keras_filename = filename.replace('.h5', '.keras')
            self.model.save(keras_filename)
            print(f"Model saved as: {keras_filename}")
    
    def evaluate_model(self):
        """Step 11: Evaluate model performance"""
        print("\n[STEP 11] Evaluating Model...")
        
        # Make predictions
        y_pred_scaled = self.model.predict(self.X_test, verbose=0)
        
        # Inverse transform predictions
        dummy = np.zeros((len(y_pred_scaled), len(self.feature_columns)))
        dummy[:, 0] = y_pred_scaled.flatten()
        y_pred = self.scaler.inverse_transform(dummy)[:, 0]
        
        # Inverse transform actual values
        dummy_test = np.zeros((len(self.y_test), len(self.feature_columns)))
        dummy_test[:, 0] = self.y_test
        y_test_actual = self.scaler.inverse_transform(dummy_test)[:, 0]
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_actual, y_pred)
        mse = mean_squared_error(y_test_actual, y_pred)
        rmse = math.sqrt(mse)
        mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
        
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        print(f"MAE (Mean Absolute Error):        ${mae:,.2f}")
        print(f"MSE (Mean Squared Error):         ${mse:,.2f}")
        print(f"RMSE (Root Mean Squared Error):   ${rmse:,.2f}")
        print(f"MAPE (Mean Absolute % Error):     {mape:.2f}%")
        print("="*50)
        
        # Save metrics
        with open('lstm_metrics.txt', 'w') as f:
            f.write("LSTM Model Performance Metrics\n")
            f.write("="*50 + "\n")
            f.write(f"MAE:  ${mae:,.2f}\n")
            f.write(f"MSE:  ${mse:,.2f}\n")
            f.write(f"RMSE: ${rmse:,.2f}\n")
            f.write(f"MAPE: {mape:.2f}%\n")
            f.write("="*50 + "\n")
        
        print("\n Metrics saved to: lstm_metrics.txt")
        
        # Visualize predictions
        self.visualize_predictions(y_test_actual, y_pred)
        
        return mae, mse, rmse, mape
    
    def visualize_predictions(self, y_actual, y_pred):
        """Visualize actual vs predicted prices"""
        # Get test dates
        test_dates = self.test_data['date'].iloc[self.lookback:].reset_index(drop=True)
        
        plt.figure(figsize=(14, 6))
        plt.plot(test_dates, y_actual, label='Actual Price', linewidth=2, alpha=0.7)
        plt.plot(test_dates, y_pred, label='Predicted Price', linewidth=2, alpha=0.7)
        plt.title('LSTM Model: Actual vs Predicted Bitcoin Prices')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: predictions_comparison.png")
        plt.close()
        
        # Prediction errors
        errors = y_actual - y_pred
        plt.figure(figsize=(14, 5))
        plt.plot(test_dates, errors, linewidth=1, color='red', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.title('Prediction Errors (Actual - Predicted)')
        plt.xlabel('Date')
        plt.ylabel('Error (USD)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('prediction_errors.png', dpi=300, bbox_inches='tight')
        print(" Saved: prediction_errors.png")
        plt.close()
    
    def run_complete_pipeline(self):
        """Execute the complete pipeline from start to finish"""
        print("\n" + "="*70)
        print("STARTING COMPLETE LSTM TRAINING PIPELINE")
        print("="*70)
        
        # Step by step execution
        self.load_and_explore_data()
        self.clean_data()
        # self.feature_engineering()
        # self.visualize_data()
        self.prepare_train_test_split(train_size=0.8)
        train_scaled, test_scaled = self.scale_data()
        self.prepare_sequences(train_scaled, test_scaled)
        self.build_model()
        self.train_model(epochs=100, batch_size=32)
        # self.save_model('lstm_model.h5')
        self.save_model('final_model1/lstm_model.h5')
        self.evaluate_model()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nDeliverables created:")
        print("  1. lstm_model.h5 (or lstm_model.keras) - Trained model")
        print("  2. lstm_scaler.pkl - Fitted scaler")
        print("  3. lstm_metrics.txt - Performance metrics")
        print("  4. Visualization plots (PNG files)")
        print("="*70)


# Predictor class for inference
class BitcoinPricePredictor:
    """
    Load trained model and make predictions
    """
    
    def __init__(self, model_path='lstm_model.h5', scaler_path='lstm_scaler.pkl'):
        """Initialize predictor with saved model and scaler"""
        print("Loading trained model...")
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from: {model_path}")
        except:
            # Try .keras format
            keras_path = model_path.replace('.h5', '.keras')
            self.model = tf.keras.models.load_model(keras_path)
            print(f" Model loaded from: {keras_path}")
        
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from: {scaler_path}")
        
        self.lookback = 24
        self.feature_columns = ['price', 'price_lag_1', 'price_ma_7', 'price_ma_24', 'volatility_24']
    
    def preprocess_data(self, data):
        """Preprocess input data"""
        df = data.copy()
        
        # Create features
        df['price_lag_1'] = df['price'].shift(1)
        df['price_ma_7'] = df['price'].rolling(window=7).mean()
        df['price_ma_24'] = df['price'].rolling(window=24).mean()
        df['volatility_24'] = df['price'].rolling(window=24).std()
        
        # Drop NaN
        df = df.dropna().reset_index(drop=True)
        
        # Extract and scale features
        features = df[self.feature_columns].values
        scaled_features = self.scaler.transform(features)
        
        return scaled_features, df['date']
    
    def predict_next_hours(self, data, steps=24):
        """
        Predict next N hours
        
        Parameters:
        - data: DataFrame with 'date' and 'price' columns (minimum 24+ rows)
        - steps: Number of hours to predict
        
        Returns:
        - DataFrame with predictions
        """
        scaled_data, dates = self.preprocess_data(data)
        
        if len(scaled_data) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} hours of data")
        
        # Use last sequence
        last_sequence = scaled_data[-self.lookback:].reshape(1, self.lookback, len(self.feature_columns))
        
        # Predict
        predictions_scaled = []
        for _ in range(steps):
            pred = self.model.predict(last_sequence, verbose=0)
            predictions_scaled.append(pred[0, 0])
        
        # Inverse transform
        dummy = np.zeros((len(predictions_scaled), len(self.feature_columns)))
        dummy[:, 0] = predictions_scaled
        predictions = self.scaler.inverse_transform(dummy)[:, 0]
        
        # Create output
        last_date = dates.iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=steps, freq='H')
        
        result_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': predictions,
            'model_name': 'LSTM'
        })
        
        return result_df


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("BITCOIN PRICE PREDICTION - LSTM MODEL TRAINING")
    print("="*70)
    
    # Create pipeline
    pipeline = BitcoinLSTMPipeline(data_path='btc_features.csv')
    
    # Run complete pipeline
    pipeline.run_complete_pipeline()
    
    print("\n" + "="*70)
    print("Testing Predictor...")
    print("="*70)
    
    # Test the predictor
    try:
        predictor = BitcoinPricePredictor(model_path='lstm_model.h5', scaler_path='lstm_scaler.pkl')
        
        # Use last 50 hours for prediction test
        test_data = pipeline.df_clean[['date', 'price']].tail(50)
        predictions = predictor.predict_next_hours(test_data, steps=24)
        
        print("\nSample Predictions (Next 24 hours):")
        print(predictions.head(10))
        
        print("\nPredictor working successfully!")
    except Exception as e:
        print(f"\nPredictor test error: {e}")
    
    print("\n" + "="*70)
    print("ALL DONE! Check your directory for output files.")
    print("="*70)