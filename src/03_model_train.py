import os
import warnings
import json
import joblib
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime, timedelta

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Settings
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load processed data from database"""
    try:
        with sqlite3.connect("data/sp500.db") as conn:
            # Check if processed data exists
            cursor = conn.cursor()
            table_exists = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sp500_processed'"
            ).fetchone()

            if not table_exists:
                raise ValueError("Processed data not found. Run data preprocessing first.")

            df = pd.read_sql("SELECT * FROM sp500_processed", conn, parse_dates=['Date'])

            if df.empty:
                raise ValueError("No processed data available.")

        logger.info(f"Loaded processed data with shape: {df.shape}")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def prepare_features_targets(df, target_type='next_price', forecast_horizon=1):
    """
    Prepare features and target variables depending on the chosen strategy

    Args:
        df: DataFrame with processed data
        target_type: Target variable type ('next_return', 'next_price', 'log_price')
        forecast_horizon: Forecast horizon in days (1, 3, 5, 10)
    """
    try:
        # Exclude date columns and price columns that may cause "data leakage"
        exclude_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        # Exclude all target columns
        target_columns = [col for col in df.columns if col.startswith('Next_')
                          or col.startswith('Return_') or col.endswith('d')]

        exclude_columns.extend(target_columns)
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        # Define target variable depending on the chosen strategy
        if target_type == 'next_return':
            if forecast_horizon == 1:
                target_column = 'Next_Day_Return'
            else:
                target_column = f'Return_{forecast_horizon}d'
        elif target_type == 'next_price':
            target_column = f'Next_Close_{forecast_horizon}d'
        elif target_type == 'log_price':
            target_column = 'Next_Log_Close'
        else:
            raise ValueError(f"Invalid target_type: {target_type}")

        # Check if target column exists in the data
        if target_column not in df.columns:
            logger.warning(f"Target column {target_column} not found in data.")
            logger.warning(
                f"Available targets: {[col for col in df.columns if col.startswith('Next_') or col.startswith('Return_')]}")
            raise ValueError(f"Target column {target_column} not found in data")

        logger.info(f"Using target variable: {target_column}")
        logger.info(f"Number of features: {len(feature_columns)}")

        # Save feature list and target variable for later use
        training_config = {
            'features': feature_columns,
            'target_column': target_column,
            'target_type': target_type,
            'forecast_horizon': forecast_horizon
        }

        with open('models/training_config.json', 'w') as f:
            json.dump(training_config, f)

        # Check for NaN in target variable
        if df[target_column].isna().any():
            logger.warning(f"NaN values found in target column '{target_column}'. Filling forward.")
            df[target_column] = df[target_column].fillna(method='ffill')

            # If NaN still remain, fill with mean value
            if df[target_column].isna().any():
                logger.warning("Still have NaN values after forward fill. Using mean value.")
                df[target_column] = df[target_column].fillna(df[target_column].mean())

        # Check for infinite values in target variable
        if np.isinf(df[target_column]).any():
            logger.warning(f"Infinite values found in target column '{target_column}'. Replacing with NaN and filling.")
            df[target_column] = df[target_column].replace([np.inf, -np.inf], np.nan)
            df[target_column] = df[target_column].fillna(df[target_column].mean())

        return df[feature_columns], df[target_column], feature_columns, target_column

    except Exception as e:
        logger.error(f"Error preparing features and targets: {str(e)}")
        raise


def split_time_series_data(X, y, df, test_size=0.2, validation_size=0.1):
    """Split time series data into training, validation and test sets"""
    try:
        total_size = len(X)
        test_idx = int(total_size * (1 - test_size))
        val_idx = int(total_size * (1 - test_size - validation_size))

        # Data segments
        X_train = X.iloc[:val_idx]
        y_train = y.iloc[:val_idx]
        dates_train = df['Date'].iloc[:val_idx]

        X_val = X.iloc[val_idx:test_idx]
        y_val = y.iloc[val_idx:test_idx]
        dates_val = df['Date'].iloc[val_idx:test_idx]

        X_test = X.iloc[test_idx:]
        y_test = y.iloc[test_idx:]
        dates_test = df['Date'].iloc[test_idx:]

        logger.info(f"Train set: {X_train.shape[0]} samples ({dates_train.min()} to {dates_train.max()})")
        logger.info(f"Validation set: {X_val.shape[0]} samples ({dates_val.min()} to {dates_val.max()})")
        logger.info(f"Test set: {X_test.shape[0]} samples ({dates_test.min()} to {dates_test.max()})")

        return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test

    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise


def train_evaluate_model(X_train, y_train, X_val, y_val, target_type, feature_names):
    """Train and evaluate multiple models with hyperparameter optimization"""
    try:
        # Create TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Define target metric depending on the target type
        if target_type == 'next_return':
            scoring = 'neg_mean_squared_error'  # MSE is better for returns
        else:
            scoring = 'neg_mean_absolute_error'  # MAE is better for prices

        # Hyperparameter optimization for Gradient Boosting
        gb_param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }

        # Ridge regression for comparison
        ridge_param_grid = {
            'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
        }

        # RandomForest for comparison
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }

        models = {
            'GradientBoosting': {
                'estimator': GradientBoostingRegressor(random_state=42),
                'param_grid': gb_param_grid
            },
            'Ridge': {
                'estimator': Ridge(random_state=42),
                'param_grid': ridge_param_grid
            },
            'RandomForest': {
                'estimator': RandomForestRegressor(random_state=42),
                'param_grid': rf_param_grid
            }
        }

        best_models = {}
        best_val_score = float('-inf')
        best_model_name = None

        # Training and optimization of models
        for name, config in models.items():
            logger.info(f"Training {name} model...")

            # RandomizedSearchCV for hyperparameter optimization
            search = RandomizedSearchCV(
                estimator=config['estimator'],
                param_distributions=config['param_grid'],
                n_iter=20,
                cv=tscv,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            # Evaluation on validation set
            val_score = -mean_squared_error(y_val, best_model.predict(X_val))

            logger.info(f"{name} validation score: {val_score:.6f}")
            logger.info(f"Best parameters: {search.best_params_}")

            best_models[name] = {
                'model': best_model,
                'validation_score': val_score,
                'best_params': search.best_params_
            }

            # Track the best model
            if val_score > best_val_score:
                best_val_score = val_score
                best_model_name = name

        logger.info(f"Best model: {best_model_name} with validation score: {best_val_score:.6f}")

        # Return the best model
        best_model = best_models[best_model_name]['model']

        # If this is Gradient Boosting or RandomForest, output feature importances
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            # Save feature importances to DB
            with sqlite3.connect("data/sp500.db") as conn:
                feature_importance.to_sql('feature_importance', conn, if_exists='replace', index=False)

            logger.info("Top 10 important features:")
            for _, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")

        return best_model, best_model_name, best_models

    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise


def evaluate_model_performance(model, X_test, y_test, dates_test, target_type):
    """Evaluate model performance on test set"""
    try:
        # Check for missing values in test data
        if X_test.isna().any().any():
            logger.warning("NaN values found in X_test. Filling with zeros.")
            X_test = X_test.fillna(0)

        if y_test.isna().any():
            logger.warning("NaN values found in y_test. Removing NaN rows.")
            # Create mask for valid values
            valid_mask = ~y_test.isna()
            y_test = y_test[valid_mask]
            X_test = X_test.loc[valid_mask.index[valid_mask]]
            dates_test = dates_test[valid_mask]

        # Generate predictions
        predictions = model.predict(X_test)

        # Check for NaN in predictions
        if np.isnan(predictions).any():
            logger.warning("NaN values found in predictions. Removing NaN values.")
            valid_pred_mask = ~np.isnan(predictions)
            predictions = predictions[valid_pred_mask]
            actual_values = y_test.iloc[valid_pred_mask].values
            actual_dates = dates_test.iloc[valid_pred_mask]
        else:
            actual_values = y_test.values
            actual_dates = dates_test

        # If sizes don't match, log error and trim to minimum size
        if len(predictions) != len(actual_values):
            logger.warning(f"Mismatch in sizes: predictions={len(predictions)}, actual={len(actual_values)}")
            min_len = min(len(predictions), len(actual_values))
            predictions = predictions[:min_len]
            actual_values = actual_values[:min_len]
            actual_dates = actual_dates[:min_len]

        # Calculate metrics
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)

        # Create DataFrame with results
        results_df = pd.DataFrame({
            'Date': actual_dates,
            'Actual': actual_values,
            'Predicted': predictions
        })

        # If target variable is return, add cumulative return
        if target_type == 'next_return':
            results_df['Cumulative_Actual'] = (1 + results_df['Actual']).cumprod() - 1
            results_df['Cumulative_Predicted'] = (1 + results_df['Predicted']).cumprod() - 1

        # Save metrics
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'test_period_start': actual_dates.min().strftime('%Y-%m-%d'),
            'test_period_end': actual_dates.max().strftime('%Y-%m-%d')
        }

        # Save metrics to JSON
        with open('models/model_metrics.json', 'w') as f:
            json.dump(metrics, f)

        # Save prediction results
        with sqlite3.connect("data/sp500.db") as conn:
            results_df.to_sql('predictions', conn, if_exists='replace', index=False)

        logger.info("\n=== Model Performance ===")
        logger.info(f"MSE: {mse:.6f}")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"R²: {r2:.6f}")

        return results_df, metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def visualize_model_performance(results_df, metrics, target_type):
    """Visualize model results"""
    try:
        os.makedirs('reports', exist_ok=True)

        # 1. Plot of predictions vs actual values
        plt.figure(figsize=(14, 7))
        plt.plot(results_df['Date'], results_df['Actual'], label='Actual', color='blue', alpha=0.7)
        plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted', color='red', alpha=0.7)

        if target_type == 'next_return':
            plt.title('Actual vs Predicted Returns')
            plt.ylabel('Return')
        else:
            plt.title('Actual vs Predicted Prices')
            plt.ylabel('Price')

        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('reports/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Prediction errors plot
        plt.figure(figsize=(14, 7))

        errors = results_df['Actual'] - results_df['Predicted']
        plt.plot(results_df['Date'], errors, color='purple', alpha=0.8)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

        plt.title('Prediction Errors Over Time')
        plt.xlabel('Date')
        plt.ylabel('Error')
        plt.grid(True, alpha=0.3)
        plt.savefig('reports/prediction_errors.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. If this is return, show cumulative return
        if target_type == 'next_return':
            plt.figure(figsize=(14, 7))
            plt.plot(results_df['Date'], results_df['Cumulative_Actual'], label='Actual Cumulative Return',
                     color='blue', alpha=0.7)
            plt.plot(results_df['Date'], results_df['Cumulative_Predicted'], label='Predicted Cumulative Return',
                     color='red', alpha=0.7)

            plt.title('Cumulative Returns: Actual vs Predicted')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('reports/cumulative_returns.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Error distribution histogram
        plt.figure(figsize=(14, 7))
        sns.histplot(errors, kde=True, color='purple')
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)

        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig('reports/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Model performance visualizations saved to reports/ directory")

    except Exception as e:
        logger.error(f"Error visualizing model performance: {str(e)}")


def save_model(model, model_name, target_type, feature_names):
    """Save model and related information"""
    try:
        # Create directory for models if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Save model
        model_path = f'models/sp500_{target_type}_model.joblib'
        joblib.dump(model, model_path)

        # Save additional information
        model_info = {
            'model_type': model_name,
            'target_type': target_type,
            'feature_count': len(feature_names),
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'feature_names': feature_names,
        }

        with open('models/model_info.json', 'w') as f:
            json.dump(model_info, f)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Model info saved to models/model_info.json")

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def train_model(target_type='next_price', forecast_horizon=1):
    """Main model training pipeline"""
    try:
        logger.info("=== Starting model training ===")
        logger.info(f"Target type: {target_type}, Forecast horizon: {forecast_horizon} days")

        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)

        # Load data
        df = load_data()

        # Prepare features and target variables
        X, y, feature_names, target_column = prepare_features_targets(df, target_type, forecast_horizon)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test = \
            split_time_series_data(X, y, df)

        # Train and evaluate models
        best_model, best_model_name, all_models = train_evaluate_model(
            X_train, y_train, X_val, y_val, target_type, feature_names
        )

        # Evaluate on test set
        results_df, metrics = evaluate_model_performance(best_model, X_test, y_test, dates_test, target_type)

        # Visualize results
        visualize_model_performance(results_df, metrics, target_type)

        # Save model
        save_model(best_model, best_model_name, target_type, feature_names)

        logger.info("=== Model training completed successfully ===")

        return best_model, metrics

    except Exception as e:
        logger.error(f"Error in model training pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # You can choose target variable type and forecast horizon
    train_model(target_type='next_price', forecast_horizon=1)