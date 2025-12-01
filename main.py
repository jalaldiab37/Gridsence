"""
GridSense - Smart Grid Load Forecasting & Outage Risk Simulation
Main entry point for running the application.
"""

import argparse
import sys
from pathlib import Path


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


def run_demo():
    """Run a quick demo of the GridSense system."""
    print("=" * 60)
    print("   GridSense - Smart Grid Load Forecasting Demo")
    print("=" * 60)
    print()
    
    # Import modules
    from data.data_loader import DataLoader
    from data.preprocessor import DataPreprocessor
    from models.forecaster import XGBoostForecaster
    from models.evaluator import ModelEvaluator
    from sim.grid_simulator import GridSimulator
    from sim.scenarios import ScenarioManager
    
    # Step 1: Load Data
    print("📊 Step 1: Loading synthetic electrical load data...")
    loader = DataLoader()
    df = loader.load(source="synthetic", periods=24*60)  # 60 days
    print(f"   Loaded {len(df):,} hourly records")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    # Step 2: Preprocess
    print("⚙️ Step 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df, resample_freq='H', normalize=False)
    print(f"   Added {len(df_processed.columns) - len(df.columns)} engineered features")
    print()
    
    # Step 3: Train Forecasting Model
    print("🤖 Step 3: Training XGBoost forecasting model...")
    
    # Prepare features
    feature_cols = [c for c in df_processed.columns 
                   if c not in ['timestamp', 'load_mw', 'load_mw_normalized']]
    
    X = df_processed[feature_cols].values[:-24]  # Leave last 24h for testing
    y = df_processed['load_mw'].values[24:]  # Shift by 24h for prediction
    
    # Align lengths
    min_len = min(len(X), len(y))
    X, y = X[:min_len], y[:min_len]
    
    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = XGBoostForecaster(n_estimators=100, max_depth=5)
    model.train(X_train, y_train)
    print("   ✓ Model trained successfully")
    print()
    
    # Step 4: Evaluate Model
    print("📏 Step 4: Evaluating model performance...")
    predictions = model.predict(X_test)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, predictions, "XGBoost")
    
    print(f"   RMSE: {metrics['RMSE']:.2f} MW")
    print(f"   MAE:  {metrics['MAE']:.2f} MW")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print()
    
    # Step 5: Grid Simulation
    print("🔌 Step 5: Running grid simulation...")
    simulator = GridSimulator(capacity_mw=8000, reserve_margin=0.15)
    scenario_mgr = ScenarioManager()
    
    # Use heat wave scenario
    scenario = scenario_mgr.get_scenario('summer_heat_wave')
    print(f"   Scenario: {scenario.name}")
    
    # Get load forecast for simulation
    forecast_loads = df['load_mw'].values[-72:]  # Last 72 hours
    
    from datetime import datetime
    result = simulator.simulate_scenario(
        forecast_loads,
        datetime.now(),
        scenario.to_dict()
    )
    
    print(f"   Peak Load: {result.peak_load:,.0f} MW")
    print(f"   Minimum Reserve Margin: {result.min_reserve_margin*100:.1f}%")
    print(f"   Outage Events: {len(result.outage_events)}")
    print(f"   Critical Hours: {sum(1 for r in result.risk_levels if r.value == 'red')}")
    print()
    
    # Step 6: Recommendations
    print("💡 Step 6: Mitigation Recommendations:")
    for i, action in enumerate(result.mitigation_actions[:5], 1):
        print(f"   {i}. {action}")
    
    print()
    print("=" * 60)
    print("   Demo complete! Run 'python main.py dashboard' for full UI")
    print("=" * 60)


def run_train_model(model_type: str = "xgboost"):
    """Train and save a forecasting model."""
    print(f"Training {model_type} model...")
    
    from data.data_loader import DataLoader
    from data.preprocessor import DataPreprocessor
    from models.forecaster import XGBoostForecaster, LSTMForecaster
    from models.evaluator import ModelEvaluator
    
    # Load data
    loader = DataLoader()
    df = loader.load(source="synthetic", periods=24*365)  # 1 year
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df, resample_freq='H', normalize=True)
    
    # Prepare data
    if model_type.lower() == "lstm":
        X, y = preprocessor.prepare_sequences(df_processed, sequence_length=24)
        split = int(len(X) * 0.8)
        model = LSTMForecaster(sequence_length=24, n_features=X.shape[2])
        model.train(X[:split], y[:split], X[split:], y[split:], epochs=30)
    else:
        feature_cols = [c for c in df_processed.columns 
                       if c not in ['timestamp', 'load_mw', 'load_mw_normalized']]
        X = df_processed[feature_cols].values[:-24]
        y = df_processed['load_mw'].values[24:]
        min_len = min(len(X), len(y))
        X, y = X[:min_len], y[:min_len]
        
        split = int(len(X) * 0.8)
        model = XGBoostForecaster(n_estimators=200)
        model.train(X[:split], y[:split], X[split:], y[split:])
    
    # Evaluate
    evaluator = ModelEvaluator()
    if model_type.lower() == "lstm":
        predictions = model.predict(X[split:])
        metrics = evaluator.evaluate(y[split:], predictions, model_type)
    else:
        predictions = model.predict(X[split:])
        metrics = evaluator.evaluate(y[split:], predictions, model_type)
    
    print("\nModel Performance:")
    print(evaluator.generate_report())
    
    # Save model
    model_path = Path("models") / f"{model_type.lower()}_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GridSense - Smart Grid Load Forecasting & Outage Simulation"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Dashboard command
    subparsers.add_parser("dashboard", help="Launch the interactive Streamlit dashboard")
    
    # Demo command
    subparsers.add_parser("demo", help="Run a quick demonstration of the system")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a forecasting model")
    train_parser.add_argument(
        "--model", 
        choices=["xgboost", "lstm"], 
        default="xgboost",
        help="Model type to train"
    )
    
    args = parser.parse_args()
    
    if args.command == "dashboard":
        run_dashboard()
    elif args.command == "demo":
        run_demo()
    elif args.command == "train":
        run_train_model(args.model)
    else:
        # Default: show help
        parser.print_help()
        print("\n💡 Quick start: python main.py demo")


if __name__ == "__main__":
    main()


