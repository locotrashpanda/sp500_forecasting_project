# S&P 500 Forecasting System

A machine learning system for forecasting S&P 500 index movements using historical data, technical indicators, and time-series analysis.

## Features

- Historical data acquisition from Yahoo Finance
- Comprehensive feature engineering with 70+ technical and temporal indicators
- Machine learning time-series forecasting
- Automated data pipeline from acquisition to prediction
- Interactive visualizations and detailed reports
- Validation system for data integrity and model performance

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
python src/05_forecast.py
```

### Output

After running the pipeline, you'll find:
- Forecast visualizations in the `reports/` directory
- Detailed forecast summary in `reports/forecast_report.txt`
- Trained model and configuration in the `models/` directory
- Processed data in the `data/` directory

## Project Structure

- `src/`: Source code files
- `data/`: Data storage (created by scripts)
- `models/`: Model storage (created by scripts)
- `reports/`: Output reports and visualizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.