import pandas as pd
import sqlite3
import os
import logging
import sys

# Настройка логирования (исправлено для Windows)
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
    Импортирует данные из CSV-файла в таблицу SQLite

    Args:
        csv_path: Путь к CSV-файлу
        db_path: Путь к базе данных SQLite
        table_name: Имя таблицы для импорта
    """
    try:
        logger.info(f"Импорт данных из {csv_path} в базу данных {db_path}")

        # Проверяем наличие файла CSV
        if not os.path.exists(csv_path):
            logger.error(f"Файл {csv_path} не найден")
            print(f"Ошибка: Файл {csv_path} не найден")
            return False

        # Создаем директорию базы данных, если она не существует
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Загружаем данные из CSV
        df = pd.read_csv(csv_path, parse_dates=['Date'])

        # Проверка данных
        if df.empty:
            logger.error("CSV-файл не содержит данных")
            print("Ошибка: CSV-файл не содержит данных")
            return False

        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"В CSV отсутствуют столбцы: {', '.join(missing_columns)}")
            print(f"Предупреждение: В CSV отсутствуют столбцы: {', '.join(missing_columns)}")

        # Сохраняем в базу данных
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)

            # Создаем индекс для ускорения запросов
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(Date)")

        logger.info(f"Импортировано {len(df)} строк в таблицу {table_name}")
        print(f"Успешно импортировано {len(df)} строк в таблицу {table_name}")
        return True

    except Exception as e:
        logger.error(f"Ошибка при импорте данных: {str(e)}")
        print(f"Ошибка при импорте данных: {str(e)}")
        return False


def import_vix_data(csv_path, db_path):
    """
    Импортирует данные VIX из CSV-файла в базу SQLite

    Args:
        csv_path: Путь к CSV-файлу с данными VIX
        db_path: Путь к базе данных SQLite
    """
    try:
        if not os.path.exists(csv_path):
            logger.warning(f"Файл VIX данных {csv_path} не найден")
            print(f"Предупреждение: Файл VIX данных {csv_path} не найден")
            return False

        # Загружаем данные VIX
        df = pd.read_csv(csv_path, parse_dates=['Date'])

        if df.empty or 'VIX' not in df.columns:
            logger.warning("Некорректные данные VIX")
            print("Предупреждение: Некорректные данные VIX")
            return False

        # Сохраняем в базу данных
        with sqlite3.connect(db_path) as conn:
            df.to_sql('vix_data', conn, if_exists='replace', index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vix_date ON vix_data(Date)")

        logger.info(f"Импортировано {len(df)} строк VIX данных")
        print(f"Успешно импортировано {len(df)} строк VIX данных")
        return True

    except Exception as e:
        logger.error(f"Ошибка при импорте данных VIX: {str(e)}")
        print(f"Ошибка при импорте данных VIX: {str(e)}")
        return False


if __name__ == "__main__":
    try:
        # Создаем директорию для данных, если её нет
        os.makedirs("data", exist_ok=True)

        # Путь к CSV-файлу, созданному в 01_data_fetch.py
        csv_path = 'data/sp500_1951_present.csv'
        vix_path = 'data/vix_data.csv'

        # Путь к базе данных SQLite
        db_path = 'data/sp500.db'

        # Если аргументы переданы через командную строку
        if len(sys.argv) > 1:
            csv_path = sys.argv[1]

        if len(sys.argv) > 2:
            db_path = sys.argv[2]

        # Импортируем данные S&P 500
        sp500_success = import_csv_to_sqlite(csv_path, db_path)

        # Импортируем данные VIX, если они есть
        vix_success = import_vix_data(vix_path, db_path)

        if sp500_success:
            print(f"Данные S&P 500 успешно импортированы в {db_path}")
        else:
            print("Ошибка при импорте данных S&P 500")

        if vix_success:
            print(f"Данные VIX успешно импортированы в {db_path}")

    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {str(e)}")
        sys.exit(1)