import pandas as pd
import sqlite3
import os
import logging
import sys

# Setting up logging (fixed for Windows)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_import.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def import_csv_to_sqlite(csv_path, db_path, table_name='sp500'):
    """
    Imports data from CSV file to SQLite table

    Args:
        csv_path: Path to CSV file
        db_path: Path to SQLite database
        table_name: Name of table for import
    """
    try:
        logger.info(f"Importing data from {csv_path} to database {db_path}")

        # Check if CSV file exists
        if not os.path.exists(csv_path):
            logger.error(f"File {csv_path} not found")
            print(f"Error: File {csv_path} not found")
            return False

        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Load data from CSV
        df = pd.read_csv(csv_path, parse_dates=['Date'])

        # Data validation
        if df.empty:
            logger.error("CSV file contains no data")
            print("Error: CSV file contains no data")
            return False

        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"CSV is missing columns: {', '.join(missing_columns)}")
            print(f"Warning: CSV is missing columns: {', '.join(missing_columns)}")

        # Save to database
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)

            # Create index to speed up queries
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(Date)")

        logger.info(f"Imported {len(df)} rows into table {table_name}")
        print(f"Successfully imported {len(df)} rows into table {table_name}")
        return True

    except Exception as e:
        logger.error(f"Error importing data: {str(e)}")
        print(f"Error importing data: {str(e)}")
        return False


def import_vix_data(csv_path, db_path):
    """
    Imports VIX data from CSV file into SQLite database

    Args:
        csv_path: Path to CSV file with VIX data
        db_path: Path to SQLite database
    """
    try:
        if not os.path.exists(csv_path):
            logger.warning(f"VIX data file {csv_path} not found")
            print(f"Warning: VIX data file {csv_path} not found")
            return False

        # Load VIX data
        df = pd.read_csv(csv_path, parse_dates=['Date'])

        if df.empty or 'VIX' not in df.columns:
            logger.warning("Invalid VIX data")
            print("Warning: Invalid VIX data")
            return False

        # Save to database
        with sqlite3.connect(db_path) as conn:
            df.to_sql('vix_data', conn, if_exists='replace', index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vix_date ON vix_data(Date)")

        logger.info(f"Imported {len(df)} rows of VIX data")
        print(f"Successfully imported {len(df)} rows of VIX data")
        return True

    except Exception as e:
        logger.error(f"Error importing VIX data: {str(e)}")
        print(f"Error importing VIX data: {str(e)}")
        return False


if __name__ == "__main__":
    try:
        # Create directory for data if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Path to CSV file created in 01_data_fetch.py
        csv_path = 'data/sp500_1951_present.csv'
        vix_path = 'data/vix_data.csv'

        # Path to SQLite database
        db_path = 'data/sp500.db'

        # If arguments are passed via command line
        if len(sys.argv) > 1:
            csv_path = sys.argv[1]

        if len(sys.argv) > 2:
            db_path = sys.argv[2]

        # Import S&P 500 data
        sp500_success = import_csv_to_sqlite(csv_path, db_path)

        # Import VIX data if available
        vix_success = import_vix_data(vix_path, db_path)

        if sp500_success:
            print(f"S&P 500 data successfully imported to {db_path}")
        else:
            print("Error importing S&P 500 data")

        if vix_success:
            print(f"VIX data successfully imported to {db_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)