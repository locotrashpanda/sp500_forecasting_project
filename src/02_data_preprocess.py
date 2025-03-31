import pandas as pd
import numpy as np
import sqlite3
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import sys
import json

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_preprocessing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def calculate_rsi(series, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()

    rs = avg_gain / avg_loss.replace(0, 1)
    return (100 - (100 / (1 + rs))).clip(0, 100)


def add_technical_indicators(df):
    """Add technical indicators"""
    logger.info("Adding technical indicators...")

    # Basic financial metrics
    df['Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()
    df['Volatility'] = df['Close'].rolling(20).std()
    df['Daily_Change'] = df['Close'] - df['Open']
    df['Daily_Range'] = df['High'] - df['Low']
    df['RSI'] = calculate_rsi(df['Close'])
    df['Log_Volume'] = np.log1p(df['Volume'])

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    # Bollinger Bands
    df['Upper_Band'] = df['MA_20'] + (2 * df['Volatility'])
    df['Lower_Band'] = df['MA_20'] - (2 * df['Volatility'])
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA_20']

    # Normalized price indicators (relative values instead of absolute)
    df['Price_to_MA_5'] = df['Close'] / df['MA_5']
    df['Price_to_MA_10'] = df['Close'] / df['MA_10']
    df['Price_to_MA_20'] = df['Close'] / df['MA_20']
    df['Price_to_MA_50'] = df['Close'] / df['MA_50']
    df['MA_5_to_MA_20'] = df['MA_5'] / df['MA_20']

    # Momentum indicators
    df['ROC_5'] = df['Close'].pct_change(5) * 100
    df['ROC_10'] = df['Close'].pct_change(10) * 100
    df['ROC_20'] = df['Close'].pct_change(20) * 100

    return df


def add_temporal_features(df):
    """Add temporal features"""
    logger.info("Adding temporal features...")

    # Check and convert Date column if necessary
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        logger.info("Converting Date column to datetime")
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    elif df['Date'].dt.tz is not None:
        logger.info("Removing timezone information from Date column")
        df['Date'] = df['Date'].dt.tz_localize(None)

    # Now add temporal features
    df['Day_of_week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    df['Is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['Is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['Day_of_month'] = df['Date'].dt.day
    df['Week_of_year'] = df['Date'].dt.isocalendar().week
    df['Is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
    df['Is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)

    # Add cyclical features for seasonality
    # Convert month and day of the week to sine and cosine to preserve cyclicality
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_of_week_sin'] = np.sin(2 * np.pi * df['Day_of_week'] / 7)
    df['Day_of_week_cos'] = np.cos(2 * np.pi * df['Day_of_week'] / 7)

    return df


def add_lag_features(df):
    """Add lag features"""
    logger.info("Adding lag features...")

    # Price lags
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f'Price_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)

    # Moving average lags
    for ma in [5, 10, 20, 50]:
        df[f'MA_{ma}_Lag_1'] = df[f'MA_{ma}'].shift(1)
        df[f'MA_{ma}_Lag_5'] = df[f'MA_{ma}'].shift(5)

    # Technical indicator lags
    df['RSI_Lag_1'] = df['RSI'].shift(1)
    df['MACD_Lag_1'] = df['MACD'].shift(1)
    df['BB_Width_Lag_1'] = df['BB_Width'].shift(1)
    df['Volatility_Lag_1'] = df['Volatility'].shift(1)

    # Log volume and its lags
    for lag in [1, 5, 10]:
        df[f'Log_Volume_Lag_{lag}'] = df['Log_Volume'].shift(lag)

    return df


def add_interaction_features(df):
    """Add feature interactions"""
    logger.info("Adding feature interactions...")

    # Interactions between indicators
    df['RSI_MACD'] = df['RSI'] * df['MACD']
    df['Volatility_Volume'] = df['Volatility'] * df['Log_Volume']
    df['ROC_RSI'] = df['ROC_5'] * df['RSI'] / 100

    # Trend indicators
    df['Trend_Strength'] = abs(df['MA_5'] - df['MA_50']) / df['MA_50']
    df['Above_200MA'] = (df['Close'] > df['MA_200']).astype(int)
    df['Above_50MA'] = (df['Close'] > df['MA_50']).astype(int)

    # Volatility compared to history
    df['Vol_Relative'] = df['Volatility'] / df['Volatility'].rolling(50).mean()

    # Volume indicators
    df['Volume_Delta'] = df['Log_Volume'] - df['Log_Volume'].shift(1)
    df['Volume_Trend'] = df['Log_Volume'] / df['Log_Volume'].rolling(20).mean()

    return df


def add_vix_features(df, conn):
    """Adding volatility data (VIX)"""
    try:
        logger.info("Adding VIX features...")

        # Check if VIX data table exists
        cursor = conn.cursor()
        vix_exists = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vix_data'").fetchone()

        if vix_exists:
            # Load VIX data
            vix_data = pd.read_sql("SELECT * FROM vix_data", conn, parse_dates=['Date'])

            # Join data
            df = pd.merge(df, vix_data, on='Date', how='left')

            # Fill missing values
            df['VIX'] = df['VIX'].ffill().bfill()

            # Add derivative features from VIX
            df['VIX_Change'] = df['VIX'].pct_change()
            df['VIX_MA_10'] = df['VIX'].rolling(10).mean()
            df['VIX_Ratio'] = df['VIX'] / df['VIX_MA_10']
            df['VIX_RSI'] = calculate_rsi(df['VIX'])

            logger.info("VIX features added successfully")
        else:
            logger.warning("VIX data table not found. VIX features will not be added.")
            # Add zero values for compatibility
            df['VIX'] = np.nan
            df['VIX_Change'] = np.nan
    except Exception as e:
        logger.error(f"Error adding VIX features: {str(e)}")
        df['VIX'] = np.nan
        df['VIX_Change'] = np.nan

    return df


def handle_missing_values(df):
    """Handle missing values"""
    logger.info("Handling missing values...")

    initial_rows = len(df)
    initial_missing = df.isna().sum().sum()

    logger.info(f"Initial missing values: {initial_missing}")
    logger.info(f"Columns with most missing values: \n{df.isna().sum().sort_values(ascending=False).head(10)}")

    # Fill gaps for main columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

    # Check if gaps remain and remove rows that still have NaN
    if df.isna().any().any():
        df = df.dropna()
        logger.info(f"Rows after removing remaining NaN: {len(df)}")

    # Replace remaining infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')

    final_missing = df.isna().sum().sum()
    logger.info(f"Final missing values: {final_missing}")

    return df


def transform_target(df, target_column='Close'):
    """Transform target variable for better forecasting"""
    logger.info(f"Transforming target variable: {target_column}")

    # Check for NaN and infinity in target variable
    if df[target_column].isna().any():
        logger.warning(f"NaN values found in {target_column}. Filling with forward fill method.")
        df[target_column] = df[target_column].fillna(method='ffill').fillna(method='bfill')

    if np.isinf(df[target_column]).any().any():
        logger.warning(f"Infinite values found in {target_column}. Replacing with NaN and filling.")
        df[target_column] = df[target_column].replace([np.inf, -np.inf], np.nan)
        df[target_column] = df[target_column].fillna(method='ffill').fillna(method='bfill')

    # Create target variable: percentage change in closing price for the next day
    df['Next_Day_Return'] = df[target_column].pct_change(periods=-1)

    # Also add logarithmic price and target variable as future log price
    df['Log_Close'] = np.log(df[target_column])
    df['Next_Log_Close'] = df['Log_Close'].shift(-1)

    # Shift values for 1, 3, 5, 10 days ahead
    for days in [1, 3, 5, 10]:
        df[f'Return_{days}d'] = df[target_column].pct_change(periods=-days)
        df[f'Next_Close_{days}d'] = df[target_column].shift(-days)

    # Check and fill NaN in target variables
    target_columns = ['Next_Day_Return', 'Next_Log_Close'] + \
                     [f'Return_{days}d' for days in [1, 3, 5, 10]] + \
                     [f'Next_Close_{days}d' for days in [1, 3, 5, 10]]

    for col in target_columns:
        if df[col].isna().any():
            # Use last available value for forecasting future prices
            logger.warning(f"Filling NaN values in {col}")
            df[col] = df[col].fillna(method='ffill')

            # If still NaN (for example, at the beginning of time series), use backfill
            if df[col].isna().any():
                df[col] = df[col].fillna(method='bfill')

    return df


def save_target_info(df, target_column='Close'):
    """Save target variable information for reverse transformation"""
    target_info = {
        'mean': float(df[target_column].mean()),
        'std': float(df[target_column].std()),
        'min': float(df[target_column].min()),
        'max': float(df[target_column].max()),
        'last_value': float(df[target_column].iloc[-1])
    }

    # Save information to JSON
    import json
    with open('models/target_info.json', 'w') as f:
        json.dump(target_info, f)

    logger.info(f"Target variable information saved: Mean={target_info['mean']:.2f}, Std={target_info['std']:.2f}")
    return target_info


def plot_feature_distributions(df, report_dir='reports'):
    """Plot key feature distributions"""
    try:
        os.makedirs(report_dir, exist_ok=True)

        # Select main features for visualization
        key_features = ['Close', 'Volume', 'Return', 'MA_5', 'MA_50',
                        'Volatility', 'RSI', 'MACD', 'BB_Width', 'VIX']

        # Select only features that exist in the data
        features_to_plot = [f for f in key_features if f in df.columns]

        # Define grid size
        rows = (len(features_to_plot) + 3) // 4
        cols = min(4, len(features_to_plot))

        plt.figure(figsize=(15, 4 * rows))

        for i, col in enumerate(features_to_plot, 1):
            plt.subplot(rows, cols, i)
            plt.hist(df[col].dropna(), bins=50, alpha=0.7)
            plt.title(col)
            plt.grid(True, linestyle='--', alpha=0.5)

            # Add distribution statistics
            mean = df[col].mean()
            median = df[col].median()
            plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
            plt.axvline(median, color='g', linestyle='-.', label=f'Median: {median:.2f}')

            if i <= 4:  # Add legend only in the top row
                plt.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{report_dir}/feature_distributions.png', dpi=300)
        plt.close()

        # Correlation analysis
        corr_cols = features_to_plot + ['Next_Day_Return']
        corr_cols = [c for c in corr_cols if c in df.columns]

        corr = df[corr_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig(f'{report_dir}/feature_correlations.png', dpi=300)
        plt.close()

        logger.info(f"Feature visualizations saved to {report_dir}/")
    except Exception as e:
        logger.error(f"Error plotting feature distributions: {str(e)}")


def save_processed_data(df, conn):
    """Save processed data to database"""
    try:
        # Save to DB
        df.to_sql("sp500_processed", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS date_index ON sp500_processed(Date)")
        conn.commit()

        # Also save the last data separately for forecasting
        last_90_days = df.sort_values('Date').tail(90)
        last_90_days.to_sql("sp500_recent", conn, if_exists="replace", index=False)

        logger.info(f"Data successfully saved to database, shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error saving data to database: {str(e)}")
        raise


def scale_features(df):
    """Scale numerical features using StandardScaler"""
    logger.info("Scaling features...")

    # Create a copy of the dataframe
    df_scaled = df.copy()

    # Identify columns to scale (exclude Date and target variables)
    exclude_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    target_columns = [col for col in df.columns if col.startswith('Next_')
                      or col.startswith('Return_')
                      or col.endswith('d')]

    exclude_columns.extend(target_columns)
    columns_to_scale = [col for col in df.columns if
                        col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])]

    # Apply StandardScaler to selected columns
    scaler = StandardScaler()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # Save the scaler for later use
    joblib.dump(scaler, 'models/feature_scaler.joblib')

    # Save the list of scaled features for reference
    with open('models/scaled_features.json', 'w') as f:
        json.dump(columns_to_scale, f)

    logger.info(f"Features scaled: {len(columns_to_scale)} columns")

    return df_scaled, scaler

def preprocess_data():
    """Main preprocessing pipeline"""
    try:
        logger.info("Starting data processing...")

        # Create directories for results
        os.makedirs("data", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        with sqlite3.connect("data/sp500.db") as conn:
            # Check existence of sp500 table
            cursor = conn.cursor()
            table_exists = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sp500'").fetchone()

            if not table_exists:
                logger.error("Table 'sp500' not found in database. Make sure to run data_fetch.py first.")
                return None

            # Important change: explicitly convert dates and remove time zones
            df = pd.read_sql("SELECT * FROM sp500", conn)
            # Convert 'Date' to datetime and remove timezone information
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

            if df.empty:
                logger.error("No data found in 'sp500' table.")
                return None

            logger.info(f"Loaded data with shape: {df.shape}")
            logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            logger.info(f"Initial S&P 500 price range: {df['Close'].min():.2f} to {df['Close'].max():.2f}")

        # Data processing pipeline
        df = add_technical_indicators(df)
        df = add_temporal_features(df)
        df = add_lag_features(df)
        df = add_interaction_features(df)

        with sqlite3.connect("data/sp500.db") as conn:
            df = add_vix_features(df, conn)

        df = handle_missing_values(df)
        df = transform_target(df)

        # Visualize features before scaling
        plot_feature_distributions(df, report_dir='reports/before_scaling')

        # Save target variable information
        target_info = save_target_info(df)

        # Scale features
        df_scaled, scaler = scale_features(df)

        # Visualize after scaling
        plot_feature_distributions(df_scaled, report_dir='reports/after_scaling')

        # Save feature list for model
        feature_cols = df_scaled.columns.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                                               'Next_Day_Return', 'Next_Log_Close',
                                               'Return_1d', 'Return_3d', 'Return_5d', 'Return_10d',
                                               'Next_Close_1d', 'Next_Close_3d', 'Next_Close_5d', 'Next_Close_10d'])

        # Save feature list
        pd.Series(feature_cols.tolist()).to_json('models/feature_names.json')

        # Save results
        with sqlite3.connect("data/sp500.db") as conn:
            save_processed_data(df_scaled, conn)

        logger.info(f"Processing complete. Final data shape: {df_scaled.shape}")
        logger.info(f"Feature count: {len(feature_cols)}")
        return df_scaled

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    preprocess_data()