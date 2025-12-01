"""
Data Preprocessor Module
Handles normalization, resampling, and feature engineering for load data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime


class DataPreprocessor:
    """
    Preprocess electrical load data for forecasting models.
    Handles missing values, normalization, resampling, and feature engineering.
    """
    
    def __init__(self):
        self.scaler: Optional[MinMaxScaler] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.original_columns: List[str] = []
        self.is_fitted: bool = False
    
    def preprocess(
        self,
        df: pd.DataFrame,
        resample_freq: str = 'H',
        normalize: bool = True,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        
        Args:
            df: Input DataFrame with timestamp and load_mw columns
            resample_freq: Resampling frequency ('H' for hourly, 'D' for daily)
            normalize: Whether to normalize load values
            add_features: Whether to add time-based features
        
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        self.original_columns = list(df.columns)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # Handle missing values
        df = self._handle_missing(df)
        
        # Resample to specified frequency
        df = self._resample(df, resample_freq)
        
        # Add time-based features
        if add_features:
            df = self._add_time_features(df)
        
        # Normalize
        if normalize:
            df = self._normalize(df)
        
        self.is_fitted = True
        return df.reset_index()
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using interpolation."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isna().any():
                # Use linear interpolation for time series
                df[col] = df[col].interpolate(method='linear')
                # Fill any remaining NaN at edges
                df[col] = df[col].ffill().bfill()
        
        return df
    
    def _resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample data to specified frequency."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Aggregate by mean for most columns
        agg_dict = {col: 'mean' for col in numeric_cols}
        
        # Use max for peak indicators
        peak_cols = [c for c in numeric_cols if 'peak' in c.lower() or 'max' in c.lower()]
        for col in peak_cols:
            agg_dict[col] = 'max'
        
        df_resampled = df[numeric_cols].resample(freq).agg(agg_dict)
        
        return df_resampled
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features for forecasting."""
        idx = df.index
        
        # Hour of day (cyclical encoding)
        df['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24)
        
        # Day of week (cyclical encoding)
        df['dow_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7)
        
        # Month (cyclical encoding for seasonality)
        df['month_sin'] = np.sin(2 * np.pi * idx.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * idx.month / 12)
        
        # Day of year (cyclical encoding)
        df['doy_sin'] = np.sin(2 * np.pi * idx.dayofyear / 365)
        df['doy_cos'] = np.cos(2 * np.pi * idx.dayofyear / 365)
        
        # Binary features
        df['is_weekend'] = (idx.dayofweek >= 5).astype(int)
        df['is_business_hour'] = ((idx.hour >= 9) & (idx.hour <= 17)).astype(int)
        
        # Lag features
        if 'load_mw' in df.columns:
            df['load_lag_1h'] = df['load_mw'].shift(1)
            df['load_lag_24h'] = df['load_mw'].shift(24)
            df['load_lag_168h'] = df['load_mw'].shift(168)  # 1 week
            
            # Rolling statistics
            df['load_rolling_mean_24h'] = df['load_mw'].rolling(window=24, min_periods=1).mean()
            df['load_rolling_std_24h'] = df['load_mw'].rolling(window=24, min_periods=1).std()
            df['load_rolling_max_24h'] = df['load_mw'].rolling(window=24, min_periods=1).max()
        
        # Fill NaN from lag features
        df = df.ffill().bfill()
        
        return df
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric columns using MinMax scaling."""
        if self.scaler is None:
            self.scaler = MinMaxScaler()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if 'load_mw' in numeric_cols:
            # Separate scaler for target variable
            load_values = df[['load_mw']].values
            df['load_mw_normalized'] = self.scaler.fit_transform(load_values)
        
        # Scale features
        feature_cols = [c for c in numeric_cols if c != 'load_mw' and c != 'load_mw_normalized']
        if feature_cols and self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            df[feature_cols] = self.feature_scaler.fit_transform(df[feature_cols])
        
        return df
    
    def inverse_transform_load(self, normalized_values: np.ndarray) -> np.ndarray:
        """Convert normalized load values back to original scale."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Run preprocess first.")
        
        return self.scaler.inverse_transform(normalized_values.reshape(-1, 1)).flatten()
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 24,
        target_col: str = 'load_mw',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM model.
        
        Args:
            df: Preprocessed DataFrame
            sequence_length: Number of time steps for input sequence
            target_col: Target column name
            feature_cols: List of feature columns (None = use all numeric)
        
        Returns:
            Tuple of (X, y) arrays for training
        """
        if feature_cols is None:
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                          if c != target_col]
        
        # Use normalized load if available
        target = target_col + '_normalized' if target_col + '_normalized' in df.columns else target_col
        
        X, y = [], []
        data = df[feature_cols + [target]].values
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, :-1])  # Features
            y.append(data[i, -1])  # Target
        
        return np.array(X), np.array(y)
    
    def prepare_prophet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet model (requires 'ds' and 'y' columns)."""
        prophet_df = pd.DataFrame()
        
        if 'timestamp' in df.columns:
            prophet_df['ds'] = pd.to_datetime(df['timestamp'])
        elif isinstance(df.index, pd.DatetimeIndex):
            prophet_df['ds'] = df.index
        
        prophet_df['y'] = df['load_mw'].values if 'load_mw' in df.columns else df.iloc[:, 0].values
        
        # Add regressors if available
        if 'temperature_c' in df.columns:
            prophet_df['temperature'] = df['temperature_c'].values
        
        return prophet_df
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-series aware train/validation/test split.
        Maintains temporal order (no shuffling).
        """
        n = len(df)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))
        
        train = df.iloc[:val_idx]
        val = df.iloc[val_idx:test_idx]
        test = df.iloc[test_idx:]
        
        return train, val, test


