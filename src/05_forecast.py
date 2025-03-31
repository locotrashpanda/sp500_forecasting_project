import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('forecasting.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def load_model_and_config():
    """Load trained model and configuration"""
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            raise FileNotFoundError("Models directory not found")

        # Load training configuration
        try:
            with open('models/training_config.json', 'r') as f:
                training_config = json.load(f)

            logger.info(f"Loaded training configuration: {training_config}")
        except FileNotFoundError:
            logger.warning("Training config not found. Using default configuration.")
            training_config = {
                'target_type': 'next_price',
                'forecast_horizon': 1,
                'features': None
            }

        # Load model information
        try:
            with open('models/model_info.json', 'r') as f:
                model_info = json.load(f)

            logger.info(f"Model type: {model_info.get('model_type', 'Unknown')}")
            logger.info(f"Training date: {model_info.get('training_date', 'Unknown')}")
        except FileNotFoundError:
            logger.warning("Model info not found")
            model_info = {}

        # Load model metrics
        try:
            with open('models/model_metrics.json', 'r') as f:
                model_metrics = json.load(f)

            logger.info(
                f"Model performance: R² = {model_metrics.get('r2', 'Unknown')}, RMSE = {model_metrics.get('rmse', 'Unknown')}")
        except FileNotFoundError:
            logger.warning("Model metrics not found")
            model_metrics = {}

        # Load target variable information
        try:
            with open('models/target_info.json', 'r') as f:
                target_info = json.load(f)

            logger.info(f"Target variable info loaded: Mean = {target_info.get('mean', 'Unknown')}")
        except FileNotFoundError:
            logger.warning("Target info not found")
            target_info = {}

        # Determine path to model based on target variable
        target_type = training_config.get('target_type', 'next_price')
        model_path = f'models/sp500_{target_type}_model.joblib'

        if not os.path.exists(model_path):
            # Try to find any other model
            model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
            if model_files:
                model_path = os.path.join('models', model_files[0])
                logger.warning(f"Specified model not found, using {model_path} instead")
            else:
                raise FileNotFoundError(f"No model file found in models/ directory")

        # Load model
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load scaler if it exists
        scaler_path = 'models/feature_scaler.joblib'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Feature scaler loaded")
        else:
            scaler = None
            logger.warning("Feature scaler not found")

        return model, training_config, model_info, model_metrics, target_info, scaler

    except Exception as e:
        logger.error(f"Error loading model and configuration: {str(e)}", exc_info=True)
        raise


def load_recent_data():
    """Load recent data for forecasting"""
    try:
        with sqlite3.connect("data/sp500.db") as conn:
            # Check if table with recent data exists
            cursor = conn.cursor()
            recent_exists = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sp500_recent'"
            ).fetchone()

            if recent_exists:
                # Load recent data from sp500_recent table
                recent_data = pd.read_sql(
                    "SELECT * FROM sp500_recent ORDER BY Date DESC LIMIT 60",
                    conn, parse_dates=['Date']
                )
                logger.info(f"Loaded {len(recent_data)} recent records from sp500_recent")
            else:
                # Load from main table
                recent_data = pd.read_sql(
                    "SELECT * FROM sp500_processed ORDER BY Date DESC LIMIT 60",
                    conn, parse_dates=['Date']
                )
                logger.info(f"Loaded {len(recent_data)} recent records from sp500_processed")

            # Also load raw data to get latest prices
            raw_data = pd.read_sql(
                "SELECT * FROM sp500 ORDER BY Date DESC LIMIT 60",
                conn, parse_dates=['Date']
            )

            # Sort data by date (from old to new)
            recent_data = recent_data.sort_values('Date')
            raw_data = raw_data.sort_values('Date')

            return recent_data, raw_data
    except Exception as e:
        logger.error(f"Error loading recent data: {str(e)}", exc_info=True)
        raise


def generate_forecast_dates(last_date, forecast_days=90, trading_days_only=True):
    """Generate dates for forecast"""
    future_dates = []
    current_date = last_date + timedelta(days=1)

    # Ensure first forecast day is next business day
    while current_date.weekday() >= 5 and trading_days_only:  # 5=Saturday, 6=Sunday
        current_date += timedelta(days=1)

    future_dates.append(current_date)

    while len(future_dates) < forecast_days:
        current_date += timedelta(days=1)
        if not trading_days_only or current_date.weekday() < 5:  # Only business days
            future_dates.append(current_date)

    logger.info(f"Generated forecast dates from {future_dates[0]} to {future_dates[-1]}")
    return future_dates


def create_future_features(recent_data, forecast_dates, features, target_info=None):
    """Create features for future dates"""
    try:
        # Create DataFrame with future dates
        future_df = pd.DataFrame({'Date': forecast_dates})

        # Add temporal features
        future_df['Day_of_week'] = future_df['Date'].dt.dayofweek
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Quarter'] = future_df['Date'].dt.quarter
        future_df['Year'] = future_df['Date'].dt.year
        future_df['Is_month_start'] = future_df['Date'].dt.is_month_start.astype(int)
        future_df['Is_month_end'] = future_df['Date'].dt.is_month_end.astype(int)
        future_df['Day_of_month'] = future_df['Date'].dt.day
        future_df['Week_of_year'] = future_df['Date'].dt.isocalendar().week
        future_df['Is_quarter_start'] = future_df['Date'].dt.is_quarter_start.astype(int)
        future_df['Is_quarter_end'] = future_df['Date'].dt.is_quarter_end.astype(int)

        # Add cyclical features
        future_df['Month_sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
        future_df['Month_cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)
        future_df['Day_of_week_sin'] = np.sin(2 * np.pi * future_df['Day_of_week'] / 7)
        future_df['Day_of_week_cos'] = np.cos(2 * np.pi * future_df['Day_of_week'] / 7)

        # Get latest technical indicator values from source data
        last_row = recent_data.iloc[-1].copy()

        # Fill future features with latest available values
        for feature in features:
            if feature in future_df.columns:
                continue  # Temporal features already added

            if feature in recent_data.columns:
                # For lag features, fill with shifted values
                if '_Lag_' in feature:
                    base_feature, lag = feature.split('_Lag_')
                    lag = int(lag)

                    if base_feature in recent_data.columns and lag <= len(recent_data):
                        lag_value = recent_data[base_feature].iloc[-lag]
                        future_df[feature] = lag_value
                    else:
                        future_df[feature] = last_row.get(feature, 0)
                else:
                    future_df[feature] = last_row.get(feature, 0)
            else:
                # If feature doesn't exist in data, fill with zeros
                future_df[feature] = 0

        # Check if all necessary features are present
        missing_features = set(features) - set(future_df.columns)
        if missing_features:
            logger.warning(f"Missing features in forecast data: {missing_features}")
            for feature in missing_features:
                future_df[feature] = 0

        # Return only necessary features in correct order
        X_future = future_df[features].copy()

        return X_future, future_df

    except Exception as e:
        logger.error(f"Error creating future features: {str(e)}", exc_info=True)
        raise


def apply_model_to_forecast(model, X_future, training_config, target_info, raw_data):
    """Apply model for forecasting and transform predictions"""
    try:
        # Get model predictions
        predictions = model.predict(X_future)

        # Transform predictions depending on target type
        target_type = training_config.get('target_type', 'next_price')

        if target_type == 'next_return':
            # If forecasting returns, convert to prices
            last_price = raw_data['Close'].iloc[-1]
            prices = [last_price]

            for pred in predictions:
                # Convert return forecast to price
                next_price = last_price * (1 + pred)
                prices.append(next_price)
                last_price = next_price

            # Remove first value (original price)
            forecasted_prices = prices[1:]

        elif target_type == 'log_price':
            # If forecasting log price, apply exp
            forecasted_prices = np.exp(predictions)

        else:  # next_price
            # If forecasting price directly, use as is
            forecasted_prices = predictions

        # If forecasts differ too much from real prices - possible scaling issue
        last_real_price = raw_data['Close'].iloc[-1]
        if abs(forecasted_prices[0] / last_real_price - 1) > 0.2:  # Difference more than 20%
            logger.warning(f"Warning: First forecast price ({forecasted_prices[0]:.2f}) differs significantly "
                           f"from last real price ({last_real_price:.2f})")

            # If gap is too large, apply scale factor to forecasts
            if 'mean' in target_info and abs(forecasted_prices[0] / last_real_price - 1) > 0.2:
                scale_factor = last_real_price / forecasted_prices[0]
                logger.info(f"Applying scaling factor: {scale_factor}")
                forecasted_prices = forecasted_prices * scale_factor

        return forecasted_prices

    except Exception as e:
        logger.error(f"Error applying model to forecast: {str(e)}", exc_info=True)
        raise


def save_forecast_results(forecast_dates, forecasted_prices, raw_data):
    """Save forecast results"""
    try:
        # Create DataFrame with results
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted_Price': forecasted_prices
        })

        # Calculate changes
        last_real_price = raw_data['Close'].iloc[-1]
        forecast_df['Change_from_Last'] = (forecast_df['Predicted_Price'] / last_real_price - 1) * 100
        forecast_df['Daily_Change'] = forecast_df['Predicted_Price'].pct_change() * 100
        forecast_df['Daily_Change'].iloc[0] = (forecast_df['Predicted_Price'].iloc[0] / last_real_price - 1) * 100

        # Calculate forecast moving average
        forecast_df['Forecast_MA5'] = forecast_df['Predicted_Price'].rolling(5, min_periods=1).mean()

        # Save to database and CSV
        with sqlite3.connect('data/sp500.db') as conn:
            forecast_df.to_sql('future_predictions', conn, if_exists='replace', index=False)

        # Create directory for data if it doesn't exist
        os.makedirs('data', exist_ok=True)
        forecast_df.to_csv('data/sp500_forecast.csv', index=False)

        logger.info(f"Forecast saved to database and CSV")

        # Output general forecast information
        last_date = raw_data['Date'].iloc[-1]
        last_price = last_real_price
        final_price = forecast_df['Predicted_Price'].iloc[-1]
        total_change = (final_price / last_price - 1) * 100

        logger.info(f"Last real date: {last_date}, price: {last_price:.2f}")
        logger.info(f"Forecast end date: {forecast_dates[-1]}, price: {final_price:.2f}")
        logger.info(f"Forecasted total change: {total_change:.2f}%")

        # Create monthly statistics
        forecast_df['Month'] = forecast_df['Date'].dt.strftime('%Y-%m')
        monthly_stats = forecast_df.groupby('Month').agg({
            'Predicted_Price': ['first', 'last', 'mean', 'min', 'max']
        })
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
        monthly_stats['Monthly_Change'] = monthly_stats['Predicted_Price_last'] / monthly_stats[
            'Predicted_Price_first'] - 1

        # Add monthly statistics to log
        logger.info("\nMonthly forecast statistics:")
        for month, stats in monthly_stats.iterrows():
            logger.info(
                f"{month}: {stats['Monthly_Change'] * 100:.2f}% change (Start: {stats['Predicted_Price_first']:.2f}, "
                f"End: {stats['Predicted_Price_last']:.2f}, Min: {stats['Predicted_Price_min']:.2f}, "
                f"Max: {stats['Predicted_Price_max']:.2f})")

        return forecast_df

    except Exception as e:
        logger.error(f"Error saving forecast results: {str(e)}", exc_info=True)
        raise


def visualize_forecast(forecast_df, raw_data, model_metrics, target_info):
    """Visualize forecast"""
    try:
        # Create directory for reports if it doesn't exist
        os.makedirs('reports', exist_ok=True)

        # Prepare data for visualization
        historical_dates = raw_data['Date'].values[-60:]  # Last 60 days
        historical_prices = raw_data['Close'].values[-60:]

        # 1. Forecast chart with historical data
        plt.figure(figsize=(15, 8))

        # Draw historical data
        plt.plot(historical_dates, historical_prices, label='Historical Data', color='blue', linewidth=2)

        # Draw forecast
        plt.plot(forecast_df['Date'], forecast_df['Predicted_Price'], label='Forecast',
                 color='purple', linestyle='--', linewidth=2, marker='o', markersize=4)

        # Add uncertainty interval if RMSE metric is available
        if 'rmse' in model_metrics:
            rmse = model_metrics['rmse']
            # For long-term forecast, increase uncertainty over time
            days_from_start = [(d - forecast_df['Date'].iloc[0]).days for d in forecast_df['Date']]
            confidence_multiplier = [1 + day * 0.01 for day in days_from_start]  # 1% uncertainty increase per day

            upper_bound = forecast_df['Predicted_Price'] + rmse * np.array(confidence_multiplier)
            lower_bound = forecast_df['Predicted_Price'] - rmse * np.array(confidence_multiplier)

            # Don't allow forecast to be negative
            lower_bound = np.maximum(lower_bound, 0)

            plt.fill_between(forecast_df['Date'], lower_bound, upper_bound,
                             color='purple', alpha=0.2, label='Forecast Uncertainty')

        # Chart styling
        plt.title('S&P 500: Historical Data and Forecast', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()

        # Save chart
        plt.savefig('reports/forecast_with_history.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Detailed forecast chart
        plt.figure(figsize=(15, 8))

        # Main forecast
        plt.plot(forecast_df['Date'], forecast_df['Predicted_Price'],
                 label='Daily Forecast', color='purple', marker='o', linestyle='-', linewidth=2, markersize=4)

        # Forecast moving average
        plt.plot(forecast_df['Date'], forecast_df['Forecast_MA5'],
                 label='5-Day Moving Average', color='orange', linestyle='-', linewidth=2)

        # Mark maximums and minimums
        max_price_idx = forecast_df['Predicted_Price'].idxmax()
        min_price_idx = forecast_df['Predicted_Price'].idxmin()

        plt.scatter(forecast_df.loc[max_price_idx, 'Date'], forecast_df.loc[max_price_idx, 'Predicted_Price'],
                    color='green', s=100, zorder=5,
                    label=f'Max: {forecast_df.loc[max_price_idx, "Predicted_Price"]:.2f}')

        plt.scatter(forecast_df.loc[min_price_idx, 'Date'], forecast_df.loc[min_price_idx, 'Predicted_Price'],
                    color='red', s=100, zorder=5, label=f'Min: {forecast_df.loc[min_price_idx, "Predicted_Price"]:.2f}')

        # Annotations for maximum and minimum
        plt.annotate(f'{forecast_df.loc[max_price_idx, "Predicted_Price"]:.2f}',
                     xy=(forecast_df.loc[max_price_idx, 'Date'], forecast_df.loc[max_price_idx, 'Predicted_Price']),
                     xytext=(0, 15), textcoords='offset points', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                     fontweight='bold')

        plt.annotate(f'{forecast_df.loc[min_price_idx, "Predicted_Price"]:.2f}',
                     xy=(forecast_df.loc[min_price_idx, 'Date'], forecast_df.loc[min_price_idx, 'Predicted_Price']),
                     xytext=(0, -15), textcoords='offset points', ha='center', va='top',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                     fontweight='bold')

        # Calculate trend
        x = np.arange(len(forecast_df))
        y = forecast_df['Predicted_Price'].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        plt.plot(forecast_df['Date'], p(x), linestyle='--', color='red',
                 label=f'Trend: {"↑" if z[0] > 0 else "↓"} {abs(z[0] * 100):.2f}% per day')

        # Chart styling
        plt.title(
            f'S&P 500: 3-Month Forecast\n{forecast_df["Date"].min().strftime("%d %b %Y")} - {forecast_df["Date"].max().strftime("%d %b %Y")}',
            fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Predicted Price (USD)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()

        # Save chart
        plt.savefig('reports/detailed_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Visualize monthly changes
        forecast_df['Month'] = forecast_df['Date'].dt.strftime('%Y-%m')
        monthly_stats = forecast_df.groupby('Month').agg({
            'Predicted_Price': ['first', 'last', 'mean', 'min', 'max']
        })
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
        monthly_stats['Monthly_Change'] = monthly_stats['Predicted_Price_last'] / monthly_stats[
            'Predicted_Price_first'] - 1

        plt.figure(figsize=(15, 10))

        # Monthly changes chart
        plt.subplot(2, 1, 1)
        colors = ['green' if chg > 0 else 'red' for chg in monthly_stats['Monthly_Change']]

        bars = plt.bar(monthly_stats.index, monthly_stats['Monthly_Change'] * 100, color=colors)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height * 1.01 if height > 0 else height * 0.9,
                     f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top',
                     fontweight='bold')

        plt.title('Predicted Monthly Changes', fontsize=14)
        plt.ylabel('Percentage Change (%)', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)

        # Price range chart by month
        plt.subplot(2, 1, 2)

        for i, (month, row) in enumerate(monthly_stats.iterrows()):
            min_price = row['Predicted_Price_min']
            max_price = row['Predicted_Price_max']
            mean_price = row['Predicted_Price_mean']

            plt.plot([i, i], [min_price, max_price], 'k-', alpha=0.7)
            plt.plot([i - 0.1, i + 0.1], [min_price, min_price], 'k-', alpha=0.7)
            plt.plot([i - 0.1, i + 0.1], [max_price, max_price], 'k-', alpha=0.7)
            plt.scatter([i], [mean_price], color='blue', s=50, zorder=3)

            # Price labels
            plt.text(i, max_price, f'{max_price:.2f}', ha='center', va='bottom', fontsize=9)
            plt.text(i, min_price, f'{min_price:.2f}', ha='center', va='top', fontsize=9)

        plt.title('Predicted Price Range by Month', fontsize=14)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.xticks(range(len(monthly_stats)), monthly_stats.index)

        plt.tight_layout()
        plt.savefig('reports/monthly_forecast_stats.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Create report with forecast details
        create_forecast_report(forecast_df, historical_prices[-1], raw_data, model_metrics)

        logger.info("Forecast visualizations saved to reports directory")

    except Exception as e:
        logger.error(f"Error visualizing forecast: {str(e)}", exc_info=True)


def create_forecast_report(forecast_df, last_price, raw_data, model_metrics):
    """Create text report with forecast"""
    try:
        with open("reports/forecast_report.txt", "w") as f:
            f.write("========== S&P 500 FORECAST SUMMARY ==========\n\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Model information and metrics
            f.write("MODEL PERFORMANCE\n")
            f.write("-----------------\n")
            if model_metrics:
                f.write(f"RMSE: {model_metrics.get('rmse', 'N/A'):.4f}\n")
                f.write(f"MAE: {model_metrics.get('mae', 'N/A'):.4f}\n")
                f.write(f"R²: {model_metrics.get('r2', 'N/A'):.4f}\n")
            else:
                f.write("Model metrics not available\n")
            f.write("\n")

            # Forecast information
            f.write("FORECAST INFORMATION\n")
            f.write("--------------------\n")
            f.write(
                f"Forecast period: {forecast_df['Date'].min().strftime('%Y-%m-%d')} to {forecast_df['Date'].max().strftime('%Y-%m-%d')}\n")
            f.write(f"Number of days: {len(forecast_df)}\n")
            f.write(f"Starting price: {forecast_df['Predicted_Price'].iloc[0]:.2f}\n")
            f.write(f"Ending price: {forecast_df['Predicted_Price'].iloc[-1]:.2f}\n")

            # Calculate percentage change
            total_change = (forecast_df['Predicted_Price'].iloc[-1] / forecast_df['Predicted_Price'].iloc[0] - 1) * 100
            base_change = (forecast_df['Predicted_Price'].iloc[-1] / last_price - 1) * 100
            f.write(f"Total forecast change: {total_change:.2f}%\n")
            f.write(f"Change from last real price ({last_price:.2f}): {base_change:.2f}%\n\n")

            # Extremes
            max_idx = forecast_df['Predicted_Price'].idxmax()
            min_idx = forecast_df['Predicted_Price'].idxmin()
            f.write(
                f"Maximum price: {forecast_df['Predicted_Price'].iloc[max_idx]:.2f} on {forecast_df['Date'].iloc[max_idx].strftime('%Y-%m-%d')}\n")
            f.write(
                f"Minimum price: {forecast_df['Predicted_Price'].iloc[min_idx]:.2f} on {forecast_df['Date'].iloc[min_idx].strftime('%Y-%m-%d')}\n\n")

            # Statistics
            f.write(f"Average price: {forecast_df['Predicted_Price'].mean():.2f}\n")
            f.write(f"Median price: {forecast_df['Predicted_Price'].median():.2f}\n")
            f.write(f"Standard deviation: {forecast_df['Predicted_Price'].std():.2f}\n\n")

            # Monthly statistics
            forecast_df['Month'] = forecast_df['Date'].dt.strftime('%Y-%m')
            monthly_stats = forecast_df.groupby('Month').agg({
                'Predicted_Price': ['first', 'last', 'mean', 'min', 'max']
            })
            monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
            monthly_stats['Monthly_Change'] = monthly_stats['Predicted_Price_last'] / monthly_stats[
                'Predicted_Price_first'] - 1

            f.write("MONTHLY BREAKDOWN\n")
            f.write("-----------------\n")
            for month, row in monthly_stats.iterrows():
                f.write(
                    f"{month}: {row['Monthly_Change'] * 100:.2f}% change (Start: {row['Predicted_Price_first']:.2f}, End: {row['Predicted_Price_last']:.2f})\n")
                f.write(
                    f"    Min: {row['Predicted_Price_min']:.2f}, Max: {row['Predicted_Price_max']:.2f}, Avg: {row['Predicted_Price_mean']:.2f}\n")
            f.write("\n")

            # Visualization information
            f.write("VISUALIZATION NOTES\n")
            f.write("-------------------\n")
            f.write("The following visualization files have been created in the reports/ directory:\n")
            f.write("- forecast_with_history.png: Historical data with forecast\n")
            f.write("- detailed_forecast.png: Detailed forecast with trend line\n")
            f.write("- monthly_forecast_stats.png: Monthly statistics of the forecast\n\n")

            f.write("=================================================\n")

        logger.info("Forecast report created at reports/forecast_report.txt")

    except Exception as e:
        logger.error(f"Error creating forecast report: {str(e)}", exc_info=True)


def main(forecast_days=90):
    """Main forecasting function"""
    try:
        logger.info("=== Starting S&P 500 forecast process ===")
        logger.info(f"Forecast period: {forecast_days} days")

        # Create necessary directories
        os.makedirs('reports', exist_ok=True)
        os.makedirs('data', exist_ok=True)

        # Load model and configuration
        model, training_config, model_info, model_metrics, target_info, scaler = load_model_and_config()

        # Load recent data
        recent_data, raw_data = load_recent_data()

        # Determine last date
        last_date = raw_data['Date'].max()
        logger.info(f"Last available date: {last_date}")

        # Generate dates for forecast
        forecast_dates = generate_forecast_dates(last_date, forecast_days)

        # Get feature list for model
        if training_config and 'features' in training_config and training_config['features']:
            features = training_config['features']
        else:
            # If feature list not defined, try to get it from model
            if hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_.tolist()
            else:
                # Use all available columns except Date and targets
                features = [col for col in recent_data.columns if col not in ['Date'] and not col.startswith('Next_')]

        logger.info(f"Using {len(features)} features for forecast")

        # Create features for future dates
        X_future, future_df = create_future_features(recent_data, forecast_dates, features, target_info)

        # If scaler exists, apply it
        if scaler:
            # Get list of scaled features
            try:
                with open('models/scaled_features.json', 'r') as f:
                    scaled_features = json.load(f)

                # Apply scaling only to necessary features
                cols_to_scale = [col for col in scaled_features if col in X_future.columns]
                X_future[cols_to_scale] = scaler.transform(X_future[cols_to_scale])
                logger.info(f"Applied feature scaling to {len(cols_to_scale)} features")
            except FileNotFoundError:
                # If no list of scaled features, apply to all
                X_future = pd.DataFrame(scaler.transform(X_future), columns=X_future.columns)
                logger.info("Applied feature scaling to all features")

        # Apply model for forecasting
        forecasted_prices = apply_model_to_forecast(model, X_future, training_config, target_info, raw_data)

        # Save forecast results
        forecast_df = save_forecast_results(forecast_dates, forecasted_prices, raw_data)

        # Visualize forecast
        visualize_forecast(forecast_df, raw_data, model_metrics, target_info)

        logger.info("=== Forecast process completed successfully ===")

        # Output final forecast information
        last_real_price = raw_data['Close'].iloc[-1]
        final_price = forecast_df['Predicted_Price'].iloc[-1]
        total_change = (final_price / last_real_price - 1) * 100

        print(f"\nS&P 500 Forecast Summary:")
        print(f"  Last real price ({last_date.strftime('%Y-%m-%d')}): {last_real_price:.2f}")
        print(f"  Forecast end date: {forecast_dates[-1].strftime('%Y-%m-%d')}")
        print(f"  Forecasted price: {final_price:.2f} ({total_change:+.2f}%)")
        print(f"  See full forecast details in reports/forecast_report.txt")

        return forecast_df

    except Exception as e:
        logger.error(f"Error in forecast process: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Check command line arguments for number of forecast days
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            days = 90
    else:
        days = 90

    # Also check environment variable
    if "FORECAST_DAYS" in os.environ:
        try:
            days = int(os.environ["FORECAST_DAYS"])
        except ValueError:
            pass

    main(forecast_days=days)