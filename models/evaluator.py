"""
Model Evaluation Module
Metrics and evaluation utilities for forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    """
    Evaluate forecasting model performance with multiple metrics.
    """
    
    def __init__(self):
        self.results: Dict[str, Dict[str, float]] = {}
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error.
        Measures average percentage deviation from actual values.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error.
        Penalizes large errors more heavily.
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error.
        More balanced than MAPE for values close to zero.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        mask = denominator != 0
        if not mask.any():
            return np.inf
        
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        """
        Mean Absolute Scaled Error.
        Scaled by in-sample naive forecast error.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Naive forecast error (1-step seasonal naive)
        naive_error = np.mean(np.abs(np.diff(y_train)))
        
        if naive_error == 0:
            return np.inf
        
        return np.mean(np.abs(y_true - y_pred)) / naive_error
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        y_train: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name for storing results
            y_train: Training data (for MASE calculation)
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'RMSE': self.rmse(y_true, y_pred),
            'MAE': self.mae(y_true, y_pred),
            'MAPE': self.mape(y_true, y_pred),
            'SMAPE': self.smape(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        
        if y_train is not None:
            metrics['MASE'] = self.mase(y_true, y_pred, y_train)
        
        self.results[model_name] = metrics
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models.
        
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        if not self.results:
            raise ValueError("No models evaluated yet. Call evaluate() first.")
        
        df = pd.DataFrame(self.results).T
        df.index.name = 'Model'
        
        # Add ranking
        for col in df.columns:
            # Lower is better for error metrics
            ascending = col != 'R2'
            df[f'{col}_rank'] = df[col].rank(ascending=ascending)
        
        return df
    
    def get_best_model(self, metric: str = 'RMSE') -> str:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison
        
        Returns:
            Name of the best model
        """
        if not self.results:
            raise ValueError("No models evaluated yet.")
        
        # Lower is better for all metrics except R2
        if metric == 'R2':
            best = max(self.results.items(), key=lambda x: x[1].get(metric, -np.inf))
        else:
            best = min(self.results.items(), key=lambda x: x[1].get(metric, np.inf))
        
        return best[0]
    
    def generate_report(self) -> str:
        """Generate a text report of model performance."""
        if not self.results:
            return "No models evaluated yet."
        
        report = ["=" * 60]
        report.append("MODEL PERFORMANCE REPORT")
        report.append("=" * 60 + "\n")
        
        for model_name, metrics in self.results.items():
            report.append(f"\n{model_name.upper()}")
            report.append("-" * 30)
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"  {metric}: {value:.4f}")
                else:
                    report.append(f"  {metric}: {value}")
        
        report.append("\n" + "=" * 60)
        
        # Best model summary
        best_rmse = self.get_best_model('RMSE')
        best_mape = self.get_best_model('MAPE')
        
        report.append(f"\nBest by RMSE: {best_rmse}")
        report.append(f"Best by MAPE: {best_mape}")
        
        return "\n".join(report)


class ForecastAnalyzer:
    """
    Analyze forecast results for patterns and insights.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, timestamps: pd.DatetimeIndex = None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.timestamps = timestamps
        self.errors = y_pred - y_true
    
    def error_by_hour(self) -> pd.DataFrame:
        """Analyze errors by hour of day."""
        if self.timestamps is None:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'hour': self.timestamps.hour,
            'error': self.errors,
            'abs_error': np.abs(self.errors)
        })
        
        return df.groupby('hour').agg({
            'error': ['mean', 'std'],
            'abs_error': 'mean'
        }).round(2)
    
    def error_by_day(self) -> pd.DataFrame:
        """Analyze errors by day of week."""
        if self.timestamps is None:
            return pd.DataFrame()
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        df = pd.DataFrame({
            'day': [days[d] for d in self.timestamps.dayofweek],
            'error': self.errors,
            'abs_error': np.abs(self.errors)
        })
        
        return df.groupby('day').agg({
            'error': ['mean', 'std'],
            'abs_error': 'mean'
        }).round(2)
    
    def peak_error_analysis(self, threshold_percentile: float = 90) -> Dict[str, Any]:
        """Analyze errors during peak demand periods."""
        threshold = np.percentile(self.y_true, threshold_percentile)
        peak_mask = self.y_true >= threshold
        
        return {
            'peak_threshold': threshold,
            'n_peak_periods': peak_mask.sum(),
            'peak_mae': np.mean(np.abs(self.errors[peak_mask])),
            'peak_mape': np.mean(np.abs(self.errors[peak_mask] / self.y_true[peak_mask])) * 100,
            'off_peak_mae': np.mean(np.abs(self.errors[~peak_mask])),
            'off_peak_mape': np.mean(np.abs(self.errors[~peak_mask] / self.y_true[~peak_mask])) * 100
        }
    
    def forecast_bias(self) -> Dict[str, Any]:
        """Analyze systematic forecast bias."""
        mean_error = np.mean(self.errors)
        over_predictions = np.sum(self.errors > 0)
        under_predictions = np.sum(self.errors < 0)
        
        return {
            'mean_bias': mean_error,
            'bias_direction': 'over' if mean_error > 0 else 'under',
            'over_prediction_ratio': over_predictions / len(self.errors),
            'under_prediction_ratio': under_predictions / len(self.errors)
        }


