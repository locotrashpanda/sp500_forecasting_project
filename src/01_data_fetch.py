import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
import os
import sqlite3

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_fetch.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def get_sp500_history(start_year=1951, end_date=None):
    """
    Gets historical S&P 500 data starting from specified year to end_date

    Args:
        start_year (int): Year to start data collection from (default 1951)
        end_date (datetime, optional): End date for data collection (default is current date)

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
    """
    try:
        # Устанавливаем даты
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        start_date = datetime(start_year, 1, 1)

        logger.info(
            f"Loading S&P 500 data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

        # Загружаем данные
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            auto_adjust=False
        )

        # Проверяем и обрабатываем данные
        if data.empty:
            raise ValueError("No data returned from Yahoo Finance")

        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])

        # Проверяем Volume
        zero_volume = data[data['Volume'] == 0]
        if not zero_volume.empty:
            logger.warning(f"Found {len(zero_volume)} rows with Volume=0")
            # Заменяем нули медианным значением
            median_vol = data[data['Volume'] > 0]['Volume'].median()
            data['Volume'] = data['Volume'].replace(0, median_vol)

        # Выбираем нужные колонки
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_columns = [col for col in required_columns if col in data.columns]

        missing_cols = set(required_columns) - set(available_columns)
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")

        return data[available_columns]

    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}", exc_info=True)
        return None


def get_vix_history(start_year=1990, end_date=None):
    """
    Gets historical VIX (Volatility Index) data

    Args:
        start_year (int): Year to start data collection from (default 1990)
        end_date (datetime, optional): End date for data collection

    Returns:
        DataFrame with Date and VIX columns
    """
    try:
        # Устанавливаем даты
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        start_date = datetime(start_year, 1, 1)

        logger.info(f"Loading VIX data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

        # Загружаем данные VIX
        vix = yf.Ticker("^VIX")
        data = vix.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )

        if data.empty:
            logger.warning("No VIX data returned from Yahoo Finance")
            return None

        data = data.reset_index()
        data = data.rename(columns={'Close': 'VIX'})
        data['Date'] = pd.to_datetime(data['Date'])

        # Оставляем только нужные колонки
        return data[['Date', 'VIX']]

    except Exception as e:
        logger.warning(f"Error retrieving VIX data: {str(e)}")
        return None


def save_data(data, filename):
    """Сохраняет данные в CSV и SQLite"""
    try:
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Убираем timezone из данных перед сохранением
        if 'Date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Date']) and data[
            'Date'].dt.tz is not None:
            data = data.copy()
            data['Date'] = data['Date'].dt.tz_localize(None)

        # Сохраняем в CSV
        data.to_csv(filename, index=False, date_format='%Y-%m-%d')
        logger.info(f"Data successfully saved to {filename}")

        # Сохраняем в SQLite
        db_path = 'data/sp500.db'
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        with sqlite3.connect(db_path) as conn:
            if 'VIX' in data.columns:
                # Это VIX данные
                data.to_sql('vix_data', conn, if_exists='replace', index=False)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_vix_date ON vix_data(Date)")
                logger.info(f"VIX data saved to SQLite database")
            else:
                # Это S&P 500 данные
                data.to_sql('sp500', conn, if_exists='replace', index=False)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sp500_date ON sp500(Date)")
                logger.info(f"S&P 500 data saved to SQLite database")

        # Логируем информацию о данных
        logger.info("\nData summary:")
        logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        logger.info(f"Rows: {len(data)}")
        logger.info(f"Columns: {list(data.columns)}")

    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")


if __name__ == "__main__":
    # Загружаем данные с 1951 года
    sp500_data = get_sp500_history(start_year=1951)

    if sp500_data is not None:
        # Создаем директорию для данных, если её нет
        os.makedirs("data", exist_ok=True)

        # Сохраняем данные
        save_data(sp500_data, 'data/sp500_1951_present.csv')

        # Дополнительная диагностика
        logger.info("\nFirst 5 rows:")
        logger.info(sp500_data.head())

        logger.info("\nLast 5 rows:")
        logger.info(sp500_data.tail())

        logger.info("\nVolume statistics:")
        logger.info(sp500_data['Volume'].describe())

        # Проверяем нулевые объемы
        zero_vol_check = sp500_data[sp500_data['Volume'] == 0]
        if not zero_vol_check.empty:
            logger.warning(f"WARNING: Found {len(zero_vol_check)} rows with Volume=0 after processing")

        # Загрузка данных VIX (если доступны)
        vix_data = get_vix_history()
        if vix_data is not None:
            save_data(vix_data, 'data/vix_data.csv')
            logger.info(f"VIX data saved with {len(vix_data)} rows")
    else:
        logger.error("Failed to load S&P 500 data")