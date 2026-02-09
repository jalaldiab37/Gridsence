# GridSense

**Smart Grid Load Forecasting and Outage Risk Simulation System**

GridSense is a software-based simulation platform for electrical power grid analysis. It integrates machine learning forecasting models with grid stability simulation to support educational exploration of demand prediction, risk assessment, and outage scenario modeling. This project demonstrates applied techniques in time-series forecasting, power systems engineering, and interactive data visualization.

---

## Table of Contents

1. [Motivation and Scope](#motivation and scope)
2. [System Architecture](#system architecture)
3. [Data and Assumptions](#data and assumptions)
4. [Forecasting Models](#forecasting models)
5. [Grid Simulation Engine](#grid simulation engine)
6. [Dashboard Features](#dashboard features)
7. [Limitations](#limitations)
8. [Quick Start](#quick-start)
9. [Project Structure](#project structure)
10. [Roadmap](#roadmap)
11. [License](#license)

---

## Motivation and Scope

Modern electrical grids face increasing operational complexity due to variable renewable generation, demand fluctuations driven by weather and economic activity, and aging infrastructure. Accurate load forecasting and proactive risk assessment are essential for maintaining grid reliability and preventing cascading failures.

GridSense addresses these challenges in an educational context by providing:

- **Load Forecasting**: Predict electricity demand over horizons ranging from 24 hours to 7 days using established machine learning techniques.
- **Risk Assessment**: Evaluate grid stability through reserve margin analysis and load factor monitoring.
- **Scenario Simulation**: Model stress conditions including extreme weather events, equipment failures, and demand surges.
- **Visualization**: Present forecasts, risk indicators, and simulation outcomes through an interactive dashboard.

**Important**: GridSense is an educational and research tool designed for learning and experimentation. It is not intended to replace production-grade energy management systems or serve as operational decision support software for grid operators.

---

## System Architecture

GridSense follows a layered architecture that separates data handling, modeling, simulation, and presentation concerns.

### Data Ingestion Layer

The data module supports multiple input sources for electrical load data:

| Source | Description | Resolution |
|--------|-------------|------------|
| Synthetic | Built-in generator with realistic load patterns | Hourly |
| OPSD | Open Power System Data (European grids) | 15-min / Hourly |
| ERCOT | Electric Reliability Council of Texas | Hourly |
| IESO | Independent Electricity System Operator (Ontario) | Hourly |
| CSV Upload | User-provided datasets | Flexible |

Data preprocessing includes timestamp parsing, missing value interpolation, resampling to uniform intervals, and feature engineering for cyclical time patterns.

### Forecasting Layer

Three complementary forecasting models generate demand predictions:

- Long Short-Term Memory (LSTM) neural networks for sequence modeling
- XGBoost gradient boosting for tabular feature-based prediction
- Prophet for trend and seasonality decomposition

Models can be trained independently or combined through ensemble averaging.

### Simulation Layer

The grid simulation engine consumes forecast outputs and evaluates system stability under configurable scenarios. It computes risk classifications, simulates breaker response dynamics, and determines load shedding requirements when demand exceeds capacity thresholds.

### Visualization Layer

A Streamlit-based dashboard provides interactive access to forecasts, real-time risk gauges, scenario comparison tools, and report generation capabilities.

---

## Data and Assumptions

### Synthetic Data Generation

The built in synthetic generator produces realistic load profiles by combining:

- **Daily patterns**: Sinusoidal variation with peaks during afternoon hours and troughs overnight.
- **Weekly patterns**: Reduced demand on weekends (approximately 15% lower than weekdays).
- **Seasonal patterns**: Higher loads in summer (cooling) and winter (heating) months.
- **Temperature correlation**: Load increases when temperature deviates significantly from moderate ranges.
- **Industrial noise**: Random spikes during business hours to simulate manufacturing activity.

### Real Dataset Compatibility

When using external datasets (OPSD, ERCOT, IESO), users should verify:

- Timestamp alignment and timezone consistency
- Load values are in megawatts (MW) or converted appropriately
- Missing data periods are handled before training

### Modeling Assumptions

GridSense employs several simplifications for tractability:

- The grid is modeled as a single-bus system without explicit transmission network topology.
- Generator dispatch follows merit order based on capacity, without unit commitment optimization.
- Renewable generation profiles use simplified statistical curves rather than physics-based models.
- Demand response is modeled as instantaneous load reduction without behavioral dynamics.

---

## Forecasting Models

### LSTM Neural Network

The LSTM implementation uses a two-layer architecture with dropout regularization to capture long-term temporal dependencies in load sequences. Input sequences of 24 hourly observations are used to predict the next time step, with recursive application for multi-step horizons.

Typical performance on well structured data:
- MAPE: 4-7% depending on forecast horizon and data quality
- RMSE: Varies with load magnitude; generally proportional to mean load

### XGBoost

XGBoost provides gradient boosted decision tree ensembles trained on engineered features including:

- Cyclical encodings of hour, day-of-week, and month
- Lag features (1-hour, 24-hour, 168-hour lookbacks)
- Rolling statistics (mean, standard deviation, maximum over 24-hour windows)

This approach offers fast training times and competitive accuracy, particularly when feature engineering captures relevant patterns.

### Prophet

Facebook Prophet handles seasonality decomposition and trend detection with minimal hyperparameter tuning. It is particularly effective for datasets with strong weekly and yearly patterns and can incorporate external regressors such as temperature.

### Evaluation Metrics

Models are evaluated using:

- **MAPE** (Mean Absolute Percentage Error): Interpretable percentage deviation from actual values.
- **RMSE** (Root Mean Square Error): Penalizes large errors more heavily; useful for risk-sensitive applications.
- **MAE** (Mean Absolute Error): Robust measure of average prediction error magnitude.

Performance varies based on data quality, forecast horizon, and the presence of anomalous events. Reported metrics should be interpreted as indicative rather than guaranteed.

---

## Grid Simulation Engine

### Risk Classification

The simulation engine classifies grid risk into four levels based on the ratio of current load to available capacity:

| Risk Level | Load Factor | Interpretation |
|------------|-------------|----------------|
| Green | Below 70% | Normal operations; adequate reserve margin |
| Yellow | 70-80% | Elevated risk; increased monitoring recommended |
| Orange | 80-90% | High risk; prepare mitigation measures |
| Red | Above 90% | Critical; immediate intervention required |

Thresholds are configurable to reflect different grid operating philosophies.

### Breaker Response Simulation

Circuit breaker behavior is modeled with configurable response delays (default 50ms). Fault isolation success probability depends on fault current magnitude, with higher currents increasing cascading failure risk.

### Load Shedding

When forecasted or simulated load exceeds capacity, the engine calculates required load reduction and estimates affected service zones. Shedding is allocated proportionally across zones based on configurable priority schemes.

### Scenario Library

Pre-built scenarios enable exploration of grid behavior under stress:

- **Summer Heat Wave**: 25-35% load increase during peak hours with temperature derating of generation capacity.
- **Winter Cold Snap**: Elevated heating demand with potential fuel supply constraints.
- **Industrial Surge**: Sustained demand increase from manufacturing activity.
- **Generator Trip**: Sudden loss of a large generating unit.
- **Transmission Constraint**: Reduced import capacity due to line outages.
- **High Renewable Variability**: Fluctuating solar and wind output requiring reserve response.

Custom scenarios can be defined by specifying demand growth rates, weather conditions, and equipment availability.

---

## Dashboard Features

The Streamlit dashboard provides four primary views:

- **Live Monitoring**: Time series visualization of current load with capacity threshold overlays and a color coded risk gauge indicating system stress level.
- **Forecast**: Interactive forecast generation with selectable horizons (24 hours, 7 days, 30 days) and model types. Displays prediction intervals and summary statistics.
- **Simulation**: Scenario selection and parameter adjustment interface. Outputs include load profiles, risk timelines, and recommended mitigation actions.
- **Reports**: PDF export functionality for generating analysis summaries suitable for documentation or presentation.

---

## Limitations

GridSense is designed for educational purposes and incorporates several simplifications that limit its applicability to real-world grid operations:

1. **No AC Power Flow**: The simulation does not perform full AC power flow analysis or optimal power flow (OPF) calculations. Voltage, reactive power, and line losses are not modeled.

2. **Simplified Transmission**: The single bus model ignores network topology, congestion, and locational pricing effects.

3. **Statistical Weather Effects**: Temperature and weather impacts use statistical correlations rather than physics based building thermal models or meteorological forecasts.

4. **No Market Dynamics**: Electricity market behavior, price formation, and economic dispatch optimization are not represented.

5. **Deterministic Scenarios**: Scenarios apply fixed parameter modifications without probabilistic uncertainty quantification.

6. **No Real-Time Data Integration**: The system operates on historical or synthetic data; live grid telemetry integration is not implemented.

Results from GridSense simulations should be interpreted as illustrative of general patterns and sensitivities rather than precise operational predictions. Human judgment and domain expertise remain essential for drawing conclusions.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jalaldiab37/Gridsence.git
cd Gridsence

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Run demonstration of forecasting and simulation pipeline
python main.py demo

# Launch interactive dashboard (opens in browser)
python main.py dashboard

# Train a forecasting model on synthetic data
python main.py train --model xgboost
```

### Dependencies

Core requirements include:
- Python 3.9+
- pandas, numpy, scipy
- scikit-learn, xgboost
- TensorFlow/Keras (for LSTM)
- Prophet
- Streamlit, Plotly
- fpdf2 (for report generation)

See `requirements.txt` for complete dependency specifications.

---

## Project Structure

```
gridsense/
├── data/
│   ├── data_loader.py       # Dataset loading and source adapters
│   ├── preprocessor.py      # Normalization, resampling, feature engineering
│   └── sample_datasets.py   # Public dataset documentation
├── models/
│   ├── forecaster.py        # LSTM, XGBoost, Prophet implementations
│   └── evaluator.py         # Metrics and model comparison utilities
├── sim/
│   ├── grid_simulator.py    # Core simulation engine
│   └── scenarios.py         # Scenario definitions and parameter management
├── dashboard/
│   ├── app.py               # Streamlit application entry point
│   ├── components.py        # Visualization components (charts, gauges)
│   └── report_generator.py  # PDF export functionality
├── main.py                  # Command-line interface
├── config.py                # Configuration parameters
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Roadmap

Planned enhancements for future development:

1. **AC Power Flow Integration**: Implement DC or AC power flow solvers to model transmission constraints and voltage profiles.

2. **Probabilistic Forecasting**: Add uncertainty quantification to predictions using quantile regression or ensemble spread.

3. **Weather Data Coupling**: Integrate real time or forecast weather data from public APIs for improved load correlation.

4. **Containerization**: Provide Docker configuration for simplified deployment and reproducibility.

5. **Continuous Integration**: Add automated testing and linting workflows via GitHub Actions.

6. **Regional Calibration**: Support geographic customization of load patterns and grid parameters for different utility service territories.

7. **Demand Response Modeling**: Implement behavioral models for price responsive and incentive-based demand reduction programs.

---

## License

MIT License - See LICENSE file for details.

---

**Author**: Jalal Diab
