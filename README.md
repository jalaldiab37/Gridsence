# GridSense

**Smart Grid Load Forecasting & Outage Risk Simulation System**

GridSense is a comprehensive software-based electrical engineering simulation platform designed to predict power grid loads and assess outage risks. It combines machine learning forecasting with grid stability simulation to help grid operators make informed decisions about demand management and risk mitigation.

## Purpose

Modern power grids face increasing challenges from variable renewable energy sources, extreme weather events, and growing demand. GridSense addresses these challenges by providing:

- **Accurate Load Forecasting**: Predict electricity demand 24 hours to 7 days ahead
- **Risk Assessment**: Real-time evaluation of grid stability and outage probability
- **Scenario Simulation**: Model extreme events like heat waves, equipment failures, and demand surges
- **Actionable Insights**: Generate mitigation recommendations before critical situations occur

## Dataset Details

GridSense supports multiple data sources for electrical load analysis:

| Source | Description | Time Resolution |
|--------|-------------|-----------------|
| **Synthetic** | Built-in realistic load generator with seasonal, daily, and weekly patterns | Hourly |
| **OPSD** | Open Power System Data for European grids | 15-min / Hourly |
| **ERCOT** | Texas grid historical load data | Hourly |
| **IESO** | Ontario (Canada) electricity demand | Hourly |
| **CSV Upload** | Custom datasets in CSV format | Flexible |

The synthetic data generator models realistic load patterns including temperature correlation, peak/off-peak cycles, weekend effects, industrial spikes, and seasonal variations (HVAC loads).

## Forecasting Models

GridSense implements three complementary forecasting approaches:

- **LSTM Neural Network**: Deep learning model capturing long-term temporal dependencies using TensorFlow/Keras. Optimal for complex, non-linear patterns.

- **XGBoost**: Gradient boosting with engineered time features. Fast training, excellent for feature-rich tabular data with strong performance on structured patterns.

- **Prophet**: Facebook's forecasting library excelling at capturing seasonality, holiday effects, and trend changes with minimal tuning.

All models are evaluated using **MAPE** (Mean Absolute Percentage Error) and **RMSE** (Root Mean Square Error) metrics, with typical accuracy of 3-5% MAPE on well-structured data.

## Grid Simulation Engine

The simulation engine models grid behavior under stress conditions:

- **Load Factor Monitoring**: Continuous tracking of capacity utilization
- **Risk Classification**: Four-tier system (Green, Yellow, Orange, Red)
- **Breaker Response**: Simulated circuit breaker delays and cascading failure risks
- **Load Shedding**: Automatic calculation of affected zones during overload
- **Scenario Library**: Pre-built scenarios including heat waves, cold snaps, industrial surges, and equipment failures

Adjustable parameters include demand growth rate, renewable penetration, generator outages, and transmission constraints.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python main.py demo

# Launch dashboard
python main.py dashboard

# Train custom model
python main.py train --model xgboost
```

## Dashboard Features

The Streamlit-powered dashboard provides:

- **Live Load Graph**: Real-time visualization with capacity thresholds
- **Risk Gauge**: Color-coded meter showing current grid stress level
- **7-Day Forecast**: Interactive forecast curves with confidence intervals
- **Scenario Toggle**: Switch between simulation scenarios on-the-fly
- **PDF Export**: Generate detailed reports for stakeholder communication

## Project Structure

```
gridsense/
├── data/           # Data loading and preprocessing
├── models/         # Forecasting models (LSTM, XGBoost, Prophet)
├── sim/            # Grid simulation engine
├── dashboard/      # Streamlit visualization app
├── main.py         # CLI entry point
└── requirements.txt
```

## License

MIT License - See LICENSE file for details.

---

**Made by Jalal Diab**
