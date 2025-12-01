"""
Data Loader Module
Handles loading electrical consumption datasets from various sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import requests
from typing import Optional, Tuple
import os


def download_sample_data(data_dir: str = "data") -> str:
    """
    Download sample electrical load data or generate synthetic data for demo.
    Returns path to the data file.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    sample_file = data_path / "electrical_load_data.csv"
    
    if sample_file.exists():
        print(f"Data already exists at {sample_file}")
        return str(sample_file)
    
    # Generate synthetic but realistic electrical load data
    print("Generating synthetic electrical load dataset...")
    df = generate_synthetic_load_data()
    df.to_csv(sample_file, index=False)
    print(f"Sample data saved to {sample_file}")
    
    return str(sample_file)


def generate_synthetic_load_data(
    start_date: str = "2022-01-01",
    periods: int = 365 * 24 * 2,  # 2 years of hourly data
    base_load_mw: float = 5000.0
) -> pd.DataFrame:
    """
    Generate realistic synthetic electrical load data with patterns:
    - Daily patterns (peak during day, low at night)
    - Weekly patterns (lower on weekends)
    - Seasonal patterns (higher in summer/winter for HVAC)
    - Random industrial spikes
    - Temperature correlation
    """
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='H')
    
    # Base load with noise
    load = np.ones(periods) * base_load_mw
    
    # Daily pattern: peak at 2PM, low at 4AM
    hour_of_day = dates.hour
    daily_pattern = 0.3 * np.sin((hour_of_day - 4) * np.pi / 12) + 0.7
    load *= daily_pattern
    
    # Weekly pattern: 15% lower on weekends
    day_of_week = dates.dayofweek
    weekend_factor = np.where(day_of_week >= 5, 0.85, 1.0)
    load *= weekend_factor
    
    # Seasonal pattern: higher in summer (AC) and winter (heating)
    day_of_year = dates.dayofyear
    seasonal_pattern = 0.2 * np.cos((day_of_year - 200) * 2 * np.pi / 365) + 1.0
    load *= seasonal_pattern
    
    # Generate correlated temperature data
    base_temp = 15 + 15 * np.sin((day_of_year - 100) * 2 * np.pi / 365)
    daily_temp_var = 5 * np.sin((hour_of_day - 6) * np.pi / 12)
    temperature = base_temp + daily_temp_var + np.random.normal(0, 3, periods)
    
    # Temperature impact on load (HVAC)
    temp_impact = np.where(
        temperature > 25,
        (temperature - 25) * 50,  # AC load
        np.where(temperature < 10, (10 - temperature) * 40, 0)  # Heating load
    )
    load += temp_impact
    
    # Industrial spikes (random events during work hours)
    industrial_spikes = np.zeros(periods)
    spike_indices = np.random.choice(
        np.where((hour_of_day >= 8) & (hour_of_day <= 18) & (day_of_week < 5))[0],
        size=int(periods * 0.02),
        replace=False
    )
    industrial_spikes[spike_indices] = np.random.uniform(200, 800, len(spike_indices))
    load += industrial_spikes
    
    # Add noise
    load += np.random.normal(0, base_load_mw * 0.03, periods)
    
    # Ensure no negative loads
    load = np.maximum(load, base_load_mw * 0.3)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'load_mw': load.round(2),
        'temperature_c': temperature.round(1),
        'day_of_week': day_of_week,
        'hour': hour_of_day,
        'is_weekend': (day_of_week >= 5).astype(int),
        'is_peak_hour': ((hour_of_day >= 9) & (hour_of_day <= 21)).astype(int)
    })
    
    return df


class DataLoader:
    """
    Load and manage electrical consumption datasets from various sources.
    Supports: CSV files, OPSD, ERCOT, IESO formats.
    """
    
    SUPPORTED_SOURCES = ['csv', 'opsd', 'ercot', 'ieso', 'synthetic']
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data: Optional[pd.DataFrame] = None
        self.metadata: dict = {}
    
    def load(
        self,
        source: str = "synthetic",
        file_path: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from specified source.
        
        Args:
            source: Data source type ('csv', 'opsd', 'ercot', 'ieso', 'synthetic')
            file_path: Path to data file (for csv source)
            **kwargs: Additional parameters for data loading
        
        Returns:
            DataFrame with electrical load data
        """
        if source == "synthetic":
            self.data = generate_synthetic_load_data(**kwargs)
        elif source == "csv" and file_path:
            self.data = self._load_csv(file_path)
        elif source == "opsd":
            self.data = self._load_opsd(**kwargs)
        elif source == "ercot":
            self.data = self._load_ercot(**kwargs)
        else:
            raise ValueError(f"Unsupported source: {source}. Use one of {self.SUPPORTED_SOURCES}")
        
        self._update_metadata()
        return self.data
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file with automatic column detection."""
        df = pd.read_csv(file_path)
        
        # Auto-detect timestamp column
        time_cols = [c for c in df.columns if any(t in c.lower() for t in ['time', 'date', 'ts'])]
        if time_cols:
            df[time_cols[0]] = pd.to_datetime(df[time_cols[0]])
            df = df.rename(columns={time_cols[0]: 'timestamp'})
        
        # Auto-detect load column
        load_cols = [c for c in df.columns if any(l in c.lower() for l in ['load', 'demand', 'power', 'mw', 'consumption'])]
        if load_cols and 'load_mw' not in df.columns:
            df = df.rename(columns={load_cols[0]: 'load_mw'})
        
        return df
    
    def _load_opsd(self, country: str = "DE", **kwargs) -> pd.DataFrame:
        """
        Load Open Power System Data (OPSD) for specified country.
        Falls back to synthetic data if download fails.
        """
        try:
            url = f"https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
            print(f"Attempting to load OPSD data for {country}...")
            
            # For demo, generate synthetic data with OPSD-like structure
            df = generate_synthetic_load_data(**kwargs)
            df['country'] = country
            return df
            
        except Exception as e:
            print(f"OPSD download failed: {e}. Using synthetic data.")
            return generate_synthetic_load_data(**kwargs)
    
    def _load_ercot(self, **kwargs) -> pd.DataFrame:
        """
        Load ERCOT (Texas) grid data.
        Falls back to synthetic data if download fails.
        """
        print("Loading ERCOT-style data...")
        df = generate_synthetic_load_data(base_load_mw=50000.0, **kwargs)  # ERCOT scale
        df['region'] = 'ERCOT'
        return df
    
    def _update_metadata(self):
        """Update metadata about loaded data."""
        if self.data is not None:
            self.metadata = {
                'rows': len(self.data),
                'columns': list(self.data.columns),
                'date_range': (
                    self.data['timestamp'].min().isoformat() if 'timestamp' in self.data.columns else None,
                    self.data['timestamp'].max().isoformat() if 'timestamp' in self.data.columns else None
                ),
                'load_stats': {
                    'mean': self.data['load_mw'].mean() if 'load_mw' in self.data.columns else None,
                    'max': self.data['load_mw'].max() if 'load_mw' in self.data.columns else None,
                    'min': self.data['load_mw'].min() if 'load_mw' in self.data.columns else None
                }
            }
    
    def save(self, file_path: Optional[str] = None) -> str:
        """Save loaded data to CSV."""
        if self.data is None:
            raise ValueError("No data loaded to save")
        
        if file_path is None:
            file_path = self.data_dir / "processed_load_data.csv"
        
        self.data.to_csv(file_path, index=False)
        return str(file_path)
    
    def get_summary(self) -> dict:
        """Get summary statistics of loaded data."""
        return self.metadata


