"""
Model Evaluation 
Evaluates and compares forecasting models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        pass
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        
        Returns:
        --------
        dict : Dictionary of metrics
        """
        # Handle pandas Series
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        # Remove any NaN values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        # Mean Forecast Error (Bias)
        mfe = np.mean(y_pred - y_true)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'MFE': mfe
        }
        
        return metrics
    
    def evaluate_model(self, y_true, y_pred, model_name='Model'):
        """
        Evaluate a single model
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        
        Returns:
        --------
        pd.DataFrame : Metrics dataframe
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        
        print(f"\n{model_name} Evaluation Metrics:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric:10s}: {value:12.4f}")
        
        df = pd.DataFrame([metrics])
        df.insert(0, 'Model', model_name)
        
        return df
    
    def compare_models(self, results_dict):
        """
        Compare multiple models
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with model names as keys and (y_true, y_pred) tuples as values
        
        Returns:
        --------
        pd.DataFrame : Comparison dataframe
        """
        comparison_list = []
        
        for model_name, (y_true, y_pred) in results_dict.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics['Model'] = model_name
            comparison_list.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_list)
        comparison_df = comparison_df[['Model', 'RMSE', 'MAE', 'MAPE', 'R2', 'MSE', 'MFE']]
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)
        
        # Find best model for each metric
        print("\nBest Models by Metric:")
        print("-" * 50)
        
        # For RMSE, MAE, MAPE, MSE - lower is better
        for metric in ['RMSE', 'MAE', 'MAPE', 'MSE']:
            best_idx = comparison_df[metric].idxmin()
            best_model = comparison_df.loc[best_idx, 'Model']
            best_value = comparison_df.loc[best_idx, metric]
            print(f"{metric:10s}: {best_model:15s} ({best_value:.4f})")
        
        # For R2 - higher is better
        best_idx = comparison_df['R2'].idxmax()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_value = comparison_df.loc[best_idx, 'R2']
        print(f"{'R2':10s}: {best_model:15s} ({best_value:.4f})")
        
        return comparison_df
    
    def plot_predictions(self, y_true, y_pred, model_name='Model', dates=None):
        """
        Plot actual vs predicted values
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        dates : array-like
            Date values for x-axis
        
        Returns:
        --------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Time series comparison
        if dates is not None:
            axes[0].plot(dates, y_true, label='Actual', linewidth=2, alpha=0.7)
            axes[0].plot(dates, y_pred, label='Predicted', linewidth=2, alpha=0.7)
        else:
            axes[0].plot(y_true, label='Actual', linewidth=2, alpha=0.7)
            axes[0].plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
        
        axes[0].set_title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axes[1].set_title(f'{model_name}: Actual vs Predicted Scatter', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Actual Price')
        axes[1].set_ylabel('Predicted Price')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, y_true, y_pred, model_name='Model'):
        """
        Plot residual analysis
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        
        Returns:
        --------
        matplotlib.figure.Figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Residuals over time
        axes[0, 0].plot(residuals, linewidth=1.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_title(f'{model_name}: Residuals Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residual histogram
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title(f'{model_name}: Residual Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals vs predicted
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_title(f'{model_name}: Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{model_name}: Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, comparison_df):
        """
        Plot model comparison
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            Comparison dataframe from compare_models()
        
        Returns:
        --------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
        
        for idx, metric in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            
            ax = axes[row, col]
            
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], alpha=0.7, edgecolor='black')
            
            # Color the best model
            if metric != 'R2':
                best_idx = comparison_df[metric].idxmin()
            else:
                best_idx = comparison_df[metric].idxmax()
            
            bars[best_idx].set_color('green')
            
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def forecast_error_analysis(self, y_true, y_pred, dates=None):
        """
        Analyze forecast errors
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        dates : array-like
            Date values
        
        Returns:
        --------
        pd.DataFrame : Error analysis dataframe
        """
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        pct_errors = (errors / y_true) * 100
        
        error_df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'error': errors,
            'abs_error': abs_errors,
            'pct_error': pct_errors
        })
        
        if dates is not None:
            error_df['date'] = dates
            error_df = error_df[['date', 'actual', 'predicted', 'error', 'abs_error', 'pct_error']]
        
        print("\nForecast Error Analysis:")
        print("-" * 50)
        print(f"Mean Error (Bias): {errors.mean():.4f}")
        print(f"Mean Absolute Error: {abs_errors.mean():.4f}")
        print(f"Mean Percentage Error: {pct_errors.mean():.4f}%")
        print(f"Std of Errors: {errors.std():.4f}")
        
        return error_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    y_true = np.random.randn(100) * 10 + 100
    y_pred1 = y_true + np.random.randn(100) * 2
    y_pred2 = y_true + np.random.randn(100) * 3
    y_pred3 = y_true + np.random.randn(100) * 1.5
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Compare models
    results_dict = {
        'ARIMA': (y_true, y_pred1),
        'Prophet': (y_true, y_pred2),
        'LSTM': (y_true, y_pred3)
    }
    
    comparison_df = evaluator.compare_models(results_dict)
    
    print("\nComparison complete!")