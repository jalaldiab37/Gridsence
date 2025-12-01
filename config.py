"""
GridSense Configuration
Centralized configuration for the application.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class GridConfig:
    """Grid simulation configuration."""
    capacity_mw: float = 10000.0
    reserve_margin: float = 0.15
    n_generators: int = 20
    breaker_response_ms: float = 50.0
    
    # Risk thresholds (as percentage of capacity)
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'green': 0.70,
        'yellow': 0.80,
        'orange': 0.90,
        'red': 0.95
    })


@dataclass
class ModelConfig:
    """Model training configuration."""
    # LSTM settings
    lstm_units: int = 64
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    
    # XGBoost settings
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # Prophet settings
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_daily_seasonality: bool = True
    
    # Sequence settings
    sequence_length: int = 24  # Hours of history for LSTM


@dataclass
class DataConfig:
    """Data processing configuration."""
    resample_freq: str = 'H'  # Hourly
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Synthetic data defaults
    synthetic_base_load_mw: float = 5000.0
    synthetic_periods: int = 24 * 365  # 1 year of hourly data


@dataclass
class DashboardConfig:
    """Dashboard UI configuration."""
    page_title: str = "GridSense - Smart Grid Dashboard"
    page_icon: str = "⚡"
    layout: str = "wide"
    theme: str = "dark"
    
    # Chart colors
    primary_color: str = "#00D4FF"
    secondary_color: str = "#FFC107"
    danger_color: str = "#FF4444"
    success_color: str = "#00CC66"
    
    # Risk colors
    risk_colors: Dict[str, str] = field(default_factory=lambda: {
        'green': '#00CC66',
        'yellow': '#FFCC00',
        'orange': '#FF9933',
        'red': '#FF3333'
    })


class Config:
    """Main configuration container."""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"
    
    # Sub-configurations
    grid = GridConfig()
    model = ModelConfig()
    data = DataConfig()
    dashboard = DashboardConfig()
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.REPORTS_DIR.mkdir(exist_ok=True)


# Global config instance
config = Config()


