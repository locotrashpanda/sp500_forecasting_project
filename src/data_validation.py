import pandas as pd
import numpy as np
import sqlite3
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
import argparse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_validation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def check_database_structure():
    """Проверка структуры базы данных"""
    try:
        # Проверяем существование файла базы данных
        db_path = 'data/sp500.db'
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            print(f"[!] Database file not found: {db_path}")
            return False, {'database_exists': False}

        results = {'database_exists': True}

        with sqlite3.connect(db_path) as conn:
            # Получаем список таблиц
            cursor = conn.cursor()
            tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            tables = [t[0] for t in tables]

            results['tables'] = tables

            # Проверяем наличие ключевых таблиц
            required_tables = ['sp500', 'sp500_processed', 'predictions', 'future_predictions']
            missing_tables = [t for t in required_tables if t not in tables]

            if missing_tables:
                logger.warning(f"Missing required tables: {missing_tables}")
                results['missing_tables'] = missing_tables
            else:
                results['missing_tables'] = []

            # Проверяем структуру каждой таблицы
            table_info = {}
            for table in tables:
                try:
                    # Получаем схему таблицы
                    schema = cursor.execute(f"PRAGMA table_info({table})").fetchall()
                    columns = [col[1] for col in schema]

                    # Получаем количество строк
                    row_count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

                    table_info[table] = {
                        'columns': columns,
                        'rows': row_count
                    }

                    # Проверяем наличие ключевых столбцов
                    if table == 'sp500' or table == 'sp500_processed':
                        if 'Date' not in columns:
                            logger.warning(f"Table {table} is missing Date column")
                            table_info[table]['missing_date'] = True

                    if table == 'future_predictions':
                        if 'Date' not in columns or 'Predicted_Price' not in columns and 'Predicted' not in columns:
                            logger.warning(f"Table {table} has improper structure")
                            table_info[table]['improper_structure'] = True

                except Exception as e:
                    logger.error(f"Error inspecting table {table}: {str(e)}")
                    table_info[table] = {'error': str(e)}

            results['table_info'] = table_info

        return True, results

    except Exception as e:
        logger.error(f"Error checking database structure: {str(e)}")
        return False, {'error': str(e)}


def check_stock_data():
    """Проверка данных S&P 500"""
    try:
        with sqlite3.connect('data/sp500.db') as conn:
            # Проверяем существование таблицы sp500
            cursor = conn.cursor()
            table_exists = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sp500'"
            ).fetchone()

            if not table_exists:
                logger.error("Table 'sp500' not found in database")
                return False, {'error': "Table 'sp500' not found"}

            # Загружаем данные
            df = pd.read_sql("SELECT * FROM sp500", conn, parse_dates=['Date'])

            if df.empty:
                logger.error("sp500 table is empty")
                return False, {'error': "sp500 table is empty"}

            # Базовая проверка данных
            results = {
                'row_count': len(df),
                'date_range': {
                    'start': df['Date'].min().strftime('%Y-%m-%d'),
                    'end': df['Date'].max().strftime('%Y-%m-%d')
                },
                'price_range': {
                    'min': float(df['Close'].min()),
                    'max': float(df['Close'].max()),
                    'latest': float(df['Close'].iloc[-1])
                }
            }

            # Проверка пропущенных значений
            na_counts = df.isna().sum()
            results['missing_values'] = {col: int(count) for col, count in na_counts.items() if count > 0}

            # Проверка дубликатов дат
            duplicate_dates = df[df.duplicated('Date', keep=False)]
            results['duplicate_dates'] = duplicate_dates.shape[0]

            # Проверка последовательности дат
            df_sorted = df.sort_values('Date')
            date_diffs = df_sorted['Date'].diff().dt.days
            large_gaps = date_diffs[date_diffs > 3]  # Пропуски больше 3 дней

            if not large_gaps.empty:
                results['date_gaps'] = {
                    'count': int(large_gaps.count()),
                    'max_gap': int(large_gaps.max()),
                    'example': df_sorted['Date'].iloc[large_gaps.index[0]].strftime('%Y-%m-%d')
                }

            # Проверка выбросов цен
            price_z_scores = (df['Close'] - df['Close'].mean()) / df['Close'].std()
            outliers = df[abs(price_z_scores) > 3]

            if not outliers.empty:
                results['price_outliers'] = {
                    'count': int(outliers.shape[0]),
                    'examples': [
                                    {
                                        'date': date.strftime('%Y-%m-%d'),
                                        'price': float(price),
                                        'z_score': float(z_score)
                                    }
                                    for date, price, z_score in zip(
                            outliers['Date'].values,
                            outliers['Close'].values,
                            price_z_scores[outliers.index].values
                        )
                                ][:5]  # Показываем только первые 5 выбросов
                }

            # Создаем визуализацию
            os.makedirs('reports/validation', exist_ok=True)

            plt.figure(figsize=(14, 7))
            plt.plot(df['Date'], df['Close'])
            plt.title('S&P 500 Historical Data')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.savefig('reports/validation/sp500_history.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Гистограмма объемов
            plt.figure(figsize=(14, 7))
            plt.hist(df['Volume'], bins=50)
            plt.title('S&P 500 Volume Distribution')
            plt.xlabel('Volume')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig('reports/validation/volume_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

            return True, results

    except Exception as e:
        logger.error(f"Error checking stock data: {str(e)}")
        return False, {'error': str(e)}


def check_model_files():
    """Проверка файлов модели"""
    try:
        # Проверяем существование директории моделей
        if not os.path.exists('models'):
            logger.error("Models directory not found")
            return False, {'models_dir_exists': False}

        results = {'models_dir_exists': True}

        # Проверяем наличие файлов модели
        model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
        results['model_files'] = model_files

        if not model_files:
            logger.warning("No model files found")
            results['model_found'] = False
            return True, results

        results['model_found'] = True

        # Проверяем конфигурационные файлы
        config_files = {
            'training_config.json': False,
            'model_info.json': False,
            'model_metrics.json': False,
            'feature_names.json': False
        }

        for config_file in config_files.keys():
            if os.path.exists(os.path.join('models', config_file)):
                config_files[config_file] = True

                # Читаем информацию из файлов
                with open(os.path.join('models', config_file), 'r') as f:
                    try:
                        data = json.load(f)
                        results[config_file] = data
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing {config_file}")
                        results[config_file] = {'error': 'Invalid JSON'}

        results['config_files'] = config_files

        return True, results

    except Exception as e:
        logger.error(f"Error checking model files: {str(e)}")
        return False, {'error': str(e)}


def check_forecast_results():
    """Проверка результатов прогнозирования"""
    try:
        with sqlite3.connect('data/sp500.db') as conn:
            # Проверяем существование таблицы future_predictions
            cursor = conn.cursor()
            table_exists = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='future_predictions'"
            ).fetchone()

            if not table_exists:
                logger.warning("No forecast results found")
                return True, {'forecast_exists': False}

            # Загружаем прогнозы
            forecast_df = pd.read_sql("SELECT * FROM future_predictions", conn, parse_dates=['Date'])

            if forecast_df.empty:
                logger.warning("future_predictions table is empty")
                return True, {'forecast_exists': True, 'forecast_empty': True}

            # Определяем колонку с прогнозом
            if 'Predicted_Price' in forecast_df.columns:
                pred_col = 'Predicted_Price'
            elif 'Predicted' in forecast_df.columns:
                pred_col = 'Predicted'
            else:
                logger.warning("No prediction column found in forecast results")
                return True, {
                    'forecast_exists': True,
                    'forecast_empty': False,
                    'error': 'No prediction column found',
                    'columns': list(forecast_df.columns)
                }

            # Базовая проверка прогноза
            results = {
                'forecast_exists': True,
                'forecast_empty': False,
                'row_count': len(forecast_df),
                'date_range': {
                    'start': forecast_df['Date'].min().strftime('%Y-%m-%d'),
                    'end': forecast_df['Date'].max().strftime('%Y-%m-%d')
                },
                'price_range': {
                    'min': float(forecast_df[pred_col].min()),
                    'max': float(forecast_df[pred_col].max()),
                    'start': float(forecast_df[pred_col].iloc[0]),
                    'end': float(forecast_df[pred_col].iloc[-1])
                }
            }

            # Проверка реалистичности прогноза
            try:
                historical = pd.read_sql("SELECT * FROM sp500 ORDER BY Date DESC LIMIT 1", conn)
                last_real_price = float(historical['Close'].iloc[0])

                first_forecast = float(forecast_df[pred_col].iloc[0])
                price_ratio = first_forecast / last_real_price

                results['reality_check'] = {
                    'last_real_price': last_real_price,
                    'first_forecast': first_forecast,
                    'price_ratio': price_ratio,
                    'seems_realistic': 0.8 < price_ratio < 1.2  # В пределах 20% от реальной цены
                }

                if not 0.8 < price_ratio < 1.2:
                    logger.warning(f"Forecast does not seem realistic (ratio: {price_ratio:.4f})")

            except Exception as e:
                logger.warning(f"Could not perform reality check: {str(e)}")

            # Визуализация прогноза
            os.makedirs('reports/validation', exist_ok=True)

            plt.figure(figsize=(14, 7))
            plt.plot(forecast_df['Date'], forecast_df[pred_col])
            plt.title('S&P 500 Forecast')
            plt.xlabel('Date')
            plt.ylabel('Predicted Price')
            plt.grid(True, alpha=0.3)
            plt.savefig('reports/validation/forecast_validation.png', dpi=300, bbox_inches='tight')
            plt.close()

            return True, results

    except Exception as e:
        logger.error(f"Error checking forecast results: {str(e)}")
        return False, {'error': str(e)}


def save_validation_report(results):
    """Создание отчета о валидации"""
    try:
        os.makedirs('reports/validation', exist_ok=True)

        with open('reports/validation/validation_report.json', 'w') as f:
            json.dump(results, f, indent=2)

        with open('reports/validation/validation_summary.txt', 'w') as f:
            f.write("=== S&P 500 FORECASTING SYSTEM VALIDATION ===\n\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Проверка структуры БД
            f.write("DATABASE STRUCTURE\n")
            f.write("-----------------\n")
            if results.get('database', {}).get('database_exists', False):
                f.write("Database exists: Yes\n")
                tables = results.get('database', {}).get('tables', [])
                f.write(f"Tables found: {', '.join(tables)}\n")

                missing = results.get('database', {}).get('missing_tables', [])
                if missing:
                    f.write(f"Missing tables: {', '.join(missing)}\n")
                else:
                    f.write("All required tables present: Yes\n")
            else:
                f.write("Database exists: No\n")
            f.write("\n")

            # Данные S&P 500
            f.write("S&P 500 DATA\n")
            f.write("-----------\n")
            stock_data = results.get('stock_data', {})
            if 'error' not in stock_data:
                f.write(f"Row count: {stock_data.get('row_count', 'N/A')}\n")
                date_range = stock_data.get('date_range', {})
                f.write(f"Date range: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}\n")

                price_range = stock_data.get('price_range', {})
                f.write(f"Price range: {price_range.get('min', 'N/A')} to {price_range.get('max', 'N/A')}\n")
                f.write(f"Latest price: {price_range.get('latest', 'N/A')}\n")

                missing_values = stock_data.get('missing_values', {})
                if missing_values:
                    f.write("Missing values detected in columns: ")
                    f.write(", ".join([f"{col} ({count})" for col, count in missing_values.items()]))
                    f.write("\n")
                else:
                    f.write("Missing values: None\n")

                if stock_data.get('duplicate_dates', 0) > 0:
                    f.write(f"Duplicate dates found: {stock_data.get('duplicate_dates')}\n")

                if 'date_gaps' in stock_data:
                    gaps = stock_data['date_gaps']
                    f.write(f"Date gaps found: {gaps.get('count')} (max: {gaps.get('max_gap')} days)\n")
            else:
                f.write(f"Error: {stock_data.get('error', 'Unknown error')}\n")
            f.write("\n")

            # Модель
            f.write("MODEL STATUS\n")
            f.write("------------\n")
            model_info = results.get('model', {})
            if model_info.get('model_found', False):
                f.write(f"Model files found: {', '.join(model_info.get('model_files', []))}\n")

                # Метрики модели, если доступны
                if 'model_metrics.json' in model_info and isinstance(model_info['model_metrics.json'], dict):
                    metrics = model_info['model_metrics.json']
                    f.write("Model metrics:\n")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric}: {value:.4f}\n")
                        else:
                            f.write(f"  {metric}: {value}\n")
            else:
                f.write("Model files found: No\n")
            f.write("\n")

            # Прогноз
            f.write("FORECAST STATUS\n")
            f.write("---------------\n")
            forecast_info = results.get('forecast', {})
            if forecast_info.get('forecast_exists', False):
                if forecast_info.get('forecast_empty', True):
                    f.write("Forecast exists but is empty\n")
                else:
                    f.write(f"Forecast rows: {forecast_info.get('row_count', 'N/A')}\n")
                    date_range = forecast_info.get('date_range', {})
                    f.write(f"Forecast period: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}\n")

                    price_range = forecast_info.get('price_range', {})
                    f.write(
                        f"Forecast price range: {price_range.get('min', 'N/A')} to {price_range.get('max', 'N/A')}\n")
                    f.write(f"Starting forecast: {price_range.get('start', 'N/A')}\n")
                    f.write(f"Ending forecast: {price_range.get('end', 'N/A')}\n")

                    # Проверка реалистичности
                    reality = forecast_info.get('reality_check', {})
                    if reality:
                        f.write(f"Last real price: {reality.get('last_real_price', 'N/A')}\n")
                        f.write(f"First forecast: {reality.get('first_forecast', 'N/A')}\n")
                        f.write(f"Forecast-to-actual ratio: {reality.get('price_ratio', 'N/A'):.4f}\n")
                        f.write(
                            f"Forecast seems realistic: {'Yes' if reality.get('seems_realistic', False) else 'No'}\n")
            else:
                f.write("Forecast exists: No\n")
            f.write("\n")

            # Итоговое заключение
            f.write("VALIDATION CONCLUSION\n")
            f.write("--------------------\n")

            # Оцениваем валидность системы
            db_valid = results.get('database', {}).get('database_exists', False) and not results.get('database',
                                                                                                     {}).get(
                'missing_tables', [])
            data_valid = 'error' not in results.get('stock_data', {})
            model_valid = results.get('model', {}).get('model_found', False)
            forecast_realistic = results.get('forecast', {}).get('reality_check', {}).get('seems_realistic', False)

            if db_valid and data_valid and model_valid:
                if forecast_realistic:
                    f.write("The forecasting system appears to be functioning correctly.\n")
                    f.write("The database structure, historical data, and model are all valid.\n")
                    f.write("The generated forecasts seem realistic compared to historical prices.\n")
                else:
                    f.write("WARNING: The system appears to be functional but the forecast values\n")
                    f.write("do not seem realistic compared to historical prices.\n")
                    f.write("This may indicate issues with scaling or model input/output processing.\n")
            else:
                f.write("ISSUES DETECTED: The forecasting system has the following issues:\n")
                if not db_valid:
                    f.write("- Database structure appears invalid or incomplete\n")
                if not data_valid:
                    f.write("- Historical data is missing or contains errors\n")
                if not model_valid:
                    f.write("- Model files are missing or incomplete\n")

            f.write("\nReport complete\n")

        logger.info("Validation report saved to reports/validation/")

        return True
    except Exception as e:
        logger.error(f"Error creating validation report: {str(e)}")
        return False


def run_validation():
    """Запуск всех проверок и создание отчета"""
    try:
        print("=== Starting S&P 500 Forecasting System Validation ===")

        results = {}

        # Проверка структуры базы данных
        print("[*] Checking database structure...")
        success, db_results = check_database_structure()
        results['database'] = db_results
        print(f"[{'✓' if success else '✗'}] Database structure check")

        # Проверка данных S&P 500
        print("[*] Checking S&P 500 data...")
        success, stock_results = check_stock_data()
        results['stock_data'] = stock_results
        print(f"[{'✓' if success else '✗'}] S&P 500 data check")

        # Проверка файлов модели
        print("[*] Checking model files...")
        success, model_results = check_model_files()
        results['model'] = model_results
        print(f"[{'✓' if success else '✗'}] Model files check")

        # Проверка результатов прогнозирования
        print("[*] Checking forecast results...")
        success, forecast_results = check_forecast_results()
        results['forecast'] = forecast_results
        print(f"[{'✓' if success else '✗'}] Forecast results check")

        # Создание отчета
        print("[*] Generating validation report...")
        success = save_validation_report(results)
        print(f"[{'✓' if success else '✗'}] Report generation")

        # Завершение
        print("\n=== Validation Complete ===")
        print("Validation report saved to reports/validation/")
        print("- validation_summary.txt: Text summary")
        print("- validation_report.json: Detailed report")
        print("- *.png: Validation visualizations")

        return results

    except Exception as e:
        logger.error(f"Error running validation: {str(e)}")
        print(f"[✗] Validation failed: {str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='S&P 500 Forecasting System Validation')
    parser.add_argument('--report-only', action='store_true', help='Generate report only without visualizations')

    args = parser.parse_args()

    run_validation()