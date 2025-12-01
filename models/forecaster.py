"""
Load Forecasting Models
Implements LSTM, Prophet, and XGBoost time-series forecasters.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
import warnings
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')


class LoadForecaster(ABC):
    """Abstract base class for load forecasting models."""
    
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_history: Dict[str, Any] = {}
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train the forecasting model."""
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs) -> np.ndarray:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def predict_future(self, steps: int, **kwargs) -> np.ndarray:
        """Predict future values."""
        pass
    
    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'LoadForecaster':
        """Load model from disk."""
        return joblib.load(path)


class LSTMForecaster(LoadForecaster):
    """
    LSTM-based load forecaster for capturing long-term dependencies.
    Uses TensorFlow/Keras backend.
    """
    
    def __init__(
        self,
        sequence_length: int = 24,
        n_features: int = 10,
        lstm_units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        super().__init__(model_name="LSTM")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.scaler = None
    
    def _build_model(self):
        """Build LSTM architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential([
                LSTM(self.lstm_units, return_sequences=True, 
                     input_shape=(self.sequence_length, self.n_features)),
                Dropout(self.dropout),
                BatchNormalization(),
                
                LSTM(self.lstm_units // 2, return_sequences=False),
                Dropout(self.dropout),
                BatchNormalization(),
                
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except ImportError:
            raise ImportError("TensorFlow required for LSTM. Install with: pip install tensorflow")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences (samples, timesteps, features)
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        
        Returns:
            Training history dictionary
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        self.n_features = X_train.shape[2] if len(X_train.shape) > 2 else 1
        self.sequence_length = X_train.shape[1]
        
        self.model = self._build_model()
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        self.training_history = history.history
        
        return self.training_history
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Generate predictions for input sequences."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def predict_future(
        self,
        last_sequence: np.ndarray,
        steps: int = 24,
        **kwargs
    ) -> np.ndarray:
        """
        Predict future values using recursive prediction.
        
        Args:
            last_sequence: Last known sequence (timesteps, features)
            steps: Number of future steps to predict
        
        Returns:
            Array of future predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Predict next value
            pred = self.model.predict(
                current_sequence.reshape(1, self.sequence_length, -1),
                verbose=0
            )[0, 0]
            predictions.append(pred)
            
            # Shift sequence and add prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred  # Assuming first feature is load
        
        return np.array(predictions)


class ProphetForecaster(LoadForecaster):
    """
    Facebook Prophet-based forecaster.
    Excellent for capturing seasonality and holiday effects.
    """
    
    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05
    ):
        super().__init__(model_name="Prophet")
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.last_date = None
    
    def train(
        self,
        df: pd.DataFrame,
        y_train=None,  # Unused, kept for interface compatibility
        X_val=None,
        y_val=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Prophet model.
        
        Args:
            df: DataFrame with 'ds' (datetime) and 'y' (target) columns
        
        Returns:
            Training info dictionary
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet required. Install with: pip install prophet")
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        # Add temperature regressor if available
        if 'temperature' in df.columns:
            self.model.add_regressor('temperature')
        
        self.model.fit(df)
        self.is_trained = True
        self.last_date = df['ds'].max()
        
        self.training_history = {
            'n_samples': len(df),
            'date_range': (df['ds'].min(), df['ds'].max())
        }
        
        return self.training_history
    
    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate predictions for given dates."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        forecast = self.model.predict(df)
        return forecast['yhat'].values
    
    def predict_future(
        self,
        steps: int = 24,
        freq: str = 'H',
        include_history: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Predict future values.
        
        Args:
            steps: Number of future periods to predict
            freq: Frequency of predictions ('H' for hourly, 'D' for daily)
            include_history: Whether to include historical fitted values
        
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        future = self.model.make_future_dataframe(
            periods=steps,
            freq=freq,
            include_history=include_history
        )
        
        # Add temperature forecast if needed (simple forward fill for demo)
        if 'temperature' in self.model.extra_regressors:
            future['temperature'] = 20  # Default temperature
        
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


class XGBoostForecaster(LoadForecaster):
    """
    XGBoost-based forecaster.
    Fast training, excellent for feature-rich tabular data.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
    ):
        super().__init__(model_name="XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.feature_names = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Names of features
        
        Returns:
            Training history with evaluation metrics
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost required. Install with: pip install xgboost")
        
        self.feature_names = feature_names
        
        # Flatten sequences if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], -1)
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            n_jobs=-1
        )
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_trained = True
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        self.training_history = {
            'feature_importance': importance,
            'n_features': X_train.shape[1],
            'n_samples': len(X_train)
        }
        
        return self.training_history
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Generate predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Flatten sequences if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def predict_future(
        self,
        last_features: np.ndarray,
        steps: int = 24,
        **kwargs
    ) -> np.ndarray:
        """
        Predict future values (simplified - uses last known features).
        For production, would need feature engineering for future timestamps.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = []
        current_features = last_features.copy()
        
        # Flatten if needed
        if len(current_features.shape) > 1:
            current_features = current_features.flatten()
        
        for _ in range(steps):
            pred = self.model.predict(current_features.reshape(1, -1))[0]
            predictions.append(pred)
            
            # Simple feature update (shift and add prediction)
            current_features = np.roll(current_features, -1)
            current_features[-1] = pred
        
        return np.array(predictions)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        importance = self.model.feature_importances_
        
        if self.feature_names and len(self.feature_names) == len(importance):
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df


class EnsembleForecaster(LoadForecaster):
    """
    Ensemble of multiple forecasters for improved accuracy.
    Combines LSTM, Prophet, and XGBoost predictions.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__(model_name="Ensemble")
        self.weights = weights or {'lstm': 0.4, 'prophet': 0.3, 'xgboost': 0.3}
        self.forecasters: Dict[str, LoadForecaster] = {}
    
    def add_forecaster(self, name: str, forecaster: LoadForecaster, weight: float = None):
        """Add a forecaster to the ensemble."""
        self.forecasters[name] = forecaster
        if weight is not None:
            self.weights[name] = weight
        
        # Normalize weights
        total = sum(self.weights.get(n, 1.0) for n in self.forecasters)
        self.weights = {n: self.weights.get(n, 1.0) / total for n in self.forecasters}
    
    def train(self, *args, **kwargs):
        """Train is handled by individual forecasters."""
        self.is_trained = all(f.is_trained for f in self.forecasters.values())
        return {'n_models': len(self.forecasters)}
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Generate weighted ensemble predictions."""
        predictions = []
        
        for name, forecaster in self.forecasters.items():
            pred = forecaster.predict(X, **kwargs)
            weight = self.weights.get(name, 1.0 / len(self.forecasters))
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def predict_future(self, steps: int = 24, **kwargs) -> np.ndarray:
        """Generate weighted ensemble future predictions."""
        predictions = []
        
        for name, forecaster in self.forecasters.items():
            pred = forecaster.predict_future(steps=steps, **kwargs)
            if isinstance(pred, pd.DataFrame):
                pred = pred['yhat'].values
            weight = self.weights.get(name, 1.0 / len(self.forecasters))
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)


