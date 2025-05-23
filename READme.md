# S&P 500 Forecasting System

A machine learning system for forecasting S&P 500 index movements using historical data, technical indicators, and time-series analysis.

## Overview

This project implements an end-to-end machine learning pipeline for predicting future S&P 500 index movements. It combines technical analysis, financial indicators, and advanced machine learning techniques to generate forecasts with configurable time horizons.

## Features

- Historical data acquisition from Yahoo Finance API (S&P 500 index and VIX volatility data)
- Comprehensive feature engineering with 70+ technical and temporal indicators:
  - Moving averages (5, 10, 20, 50, 200-day)
  - Momentum indicators (RSI, MACD, Stochastic Oscillator)
  - Volatility measures
  - Bollinger Bands
  - Temporal features (day of week, month, quarter, etc.)
  - Lag features for time-series analysis
  - VIX (Volatility Index) integration
- Time-series forecasting with multiple model options:
  - Gradient Boosting Regression
  - Random Forest Regression
  - Ridge Regression
  - ElasticNet
- Multiple forecast horizon options (1, 3, 5, 10, 30, 90 days)
- Automated data pipeline from acquisition to prediction
- Interactive visualizations and detailed forecast reports
- Validation system for data integrity and model performance
- SQLite database storage for efficient data management

## How It Works

The system operates through a modular pipeline with these key components:

1. **Data Acquisition** (`01_data_fetch.py`):
   - Fetches historical S&P 500 data from Yahoo Finance
   - Collects VIX (Volatility Index) data for enhanced predictions
   - Validates and stores data in both CSV and SQLite formats

2. **Data Preprocessing** (`02_data_preprocess.py`):
   - Calculates technical indicators (RSI, MACD, moving averages, etc.)
   - Adds temporal features to capture seasonality patterns
   - Creates lag features for time-series forecasting
   - Handles missing values and normalizes data
   - Generates feature importance visualizations

3. **Model Training** (`03_model_train.py`):
   - Implements time-series cross-validation to prevent data leakage
   - Performs hyperparameter optimization for multiple model types
   - Evaluates models using RMSE, MAE, and R² metrics
   - Creates performance visualizations
   - Saves the best performing model and its configuration

4. **Forecasting** (`05_forecast.py`):
   - Generates predictions for specified time horizons
   - Creates visualizations of predicted price movements
   - Calculates confidence intervals for predictions
   - Produces detailed forecast reports with key insights
   - Saves results to the reports directory

5. **Pipeline Orchestration** (`run_pipeline.py`):
   - Provides a unified interface to execute the entire process
   - Supports different operation modes (full, update, forecast-only)
   - Manages dependencies and execution flow
   - Handles errors and provides detailed logging

## Operation Modes

The system can be run in three different modes:

1. **Full Mode**: Complete pipeline from data acquisition to forecast
2. **Update Mode**: Updates data and regenerates forecast without retraining
3. **Forecast Mode**: Generates new forecasts using existing model and recent data

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/sp500_forecasting_project.git
cd sp500_forecasting_project
```

2. Create a virtual environment and activate it
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the complete pipeline

The complete pipeline includes data fetching, preprocessing, model training, and forecast generation:

```bash
python src/run_pipeline.py
```

### Specifying operation mode and forecast horizon

```bash
# Full pipeline with 30-day forecast
python src/run_pipeline.py --mode full --days 30

# Update data and generate 90-day forecast without retraining
python src/run_pipeline.py --mode update --days 90

# Generate 10-day forecast with existing model
python src/run_pipeline.py --mode forecast --days 10
```

### Running individual components

You can run each component separately:

1. Data fetching:
```bash
python src/01_data_fetch.py
```

2. Data preprocessing:
```bash
python src/02_data_preprocess.py
```

3. Model training:
```bash
python src/03_model_train.py
```

4. Generate forecast:
```bash
python src/05_forecast.py --days 30
```

### Output

After running the pipeline, you'll find:
- Forecast visualizations in the `reports/` directory
- Detailed forecast summary in `reports/forecast_report.txt`
- Performance metrics and visualizations
- Trained model and configuration in the `models/` directory
- Processed data in the `data/` directory

## Project Structure

- `src/`: Source code files
  - `01_data_fetch.py`: Historical data acquisition
  - `02_data_preprocess.py`: Feature engineering and data preparation
  - `03_model_train.py`: Model training and evaluation
  - `04_results_export.py`: Result export utilities
  - `05_forecast.py`: Forecast generation
  - `run_pipeline.py`: Pipeline orchestration
  - `data_validation.py`: Data validation utilities
  - `import_csv_to_sqlite.py`: Database utilities
  - `models/`: Model implementation modules
- `data/`: Data storage (created by scripts)
  - Raw and processed data in CSV and SQLite formats
- `models/`: Model storage (created by scripts)
  - Trained models, scalers, and configuration
- `reports/`: Output reports and visualizations
  - Forecast visualizations, performance metrics, feature importance

## Performance Considerations

- The forecast accuracy generally decreases as the forecast horizon increases
- Short-term forecasts (1-5 days) typically achieve higher accuracy
- Feature importance varies based on market conditions and time periods
- Model retraining is recommended periodically to capture changing market dynamics

## License

This project is licensed under the MIT License - see the LICENSE file for details.