import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.dates import MonthLocator, DateFormatter, WeekdayLocator
import logging
from statsmodels.graphics.tsaplots import plot_acf
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visualization.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def check_tables_exist(conn):
    """Проверяет существование необходимых таблиц"""
    cursor = conn.cursor()
    tables = ['sp500_processed', 'predictions', 'metrics', 'future_predictions']
    existing_tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    existing_tables = [t[0] for t in existing_tables]

    missing = [t for t in tables if t not in existing_tables]
    if missing:
        logger.warning(f"Отсутствующие таблицы: {missing}")
    return all(table in existing_tables for table in tables[:3])  # future_predictions может отсутствовать


def load_data():
    """Загрузка данных из базы"""
    try:
        with sqlite3.connect("data/sp500.db") as conn:
            if not check_tables_exist(conn):
                raise ValueError("Не все необходимые таблицы существуют в базе данных")

            # Загружаем только последние данные для наглядности (1 год)
            df = pd.read_sql(
                "SELECT * FROM sp500_processed ORDER BY Date DESC LIMIT 252",  # ~252 торговых дня в году
                conn, parse_dates=['Date']
            ).sort_values('Date')

            preds = pd.read_sql("SELECT * FROM predictions", conn, parse_dates=['Date'])
            metrics = pd.read_sql("SELECT * FROM metrics", conn)

            # Проверяем наличие future_predictions
            future = pd.DataFrame()
            if 'future_predictions' in pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)[
                'name'].values:
                future = pd.read_sql("SELECT * FROM future_predictions", conn, parse_dates=['Date'])

        return df, preds, metrics, future
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def plot_predictions(df, preds, metrics, future=None):
    """Визуализация предсказаний"""
    try:
        plt.figure(figsize=(16, 9))

        # Исторические данные
        plt.plot(df['Date'], df['Close'],
                 label='Historical Data',
                 color='#1f77b4',
                 alpha=0.7,
                 linewidth=1.5)

        # Тестовые данные и предсказания
        plt.plot(preds['Date'], preds['Actual'],
                 label='Actual Values (Test)',
                 color='#2ca02c',
                 linewidth=2.5)

        plt.plot(preds['Date'], preds['Predicted'],
                 label='Model Predictions',
                 color='#ff7f0e',
                 linewidth=2)

        # Будущие прогнозы (если есть)
        if not future.empty and 'Date' in future.columns and 'Predicted' in future.columns:
            plt.plot(future['Date'], future['Predicted'],
                     label=f'3-Month Forecast ({len(future)} days)',
                     color='#9467bd',
                     linestyle='--',
                     linewidth=2)

        # Доверительный интервал
        rmse = metrics.loc[0, 'Test_RMSE']

        # Для исторических предсказаний
        plt.fill_between(preds['Date'],
                         preds['Predicted'] - rmse,
                         preds['Predicted'] + rmse,
                         color='gray',
                         alpha=0.2,
                         label=f'Confidence Interval (±{rmse:.2f})')

        # Для прогнозов (с расширением интервала)
        if not future.empty and 'Date' in future.columns and 'Predicted' in future.columns:
            # Коэффициент расширения со временем для 3-месячного прогноза
            days = (future['Date'] - future['Date'].min()).dt.days
            confidence_multiplier = 1 + days * 0.02  # Постепенное увеличение неопределенности

            plt.fill_between(future['Date'],
                             future['Predicted'] - rmse * confidence_multiplier,
                             future['Predicted'] + rmse * confidence_multiplier,
                             color='#9467bd',
                             alpha=0.15,
                             label='Forecast Uncertainty')

        # Настройки графика
        title = 'S&P 500: Actual vs Predicted Prices'
        if not future.empty:
            last_historical = max(df['Date'].max(), preds['Date'].max())
            forecast_end = future['Date'].max()
            title += f"\nForecast: {last_historical.strftime('%d %b %Y')} - {forecast_end.strftime('%d %b %Y')}"

        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price (USD)', fontsize=14)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Форматирование дат (для 3-месячного прогноза показываем недели)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.xaxis.set_minor_locator(WeekdayLocator(byweekday=0))  # Отмечаем понедельники

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Сохранение
        os.makedirs("reports", exist_ok=True)
        plt.savefig("reports/predictions_with_ci.png",
                    dpi=300,
                    bbox_inches='tight')
        plt.close()
        logger.info("Prediction plot saved successfully")

    except Exception as e:
        logger.error(f"Error generating prediction plot: {str(e)}")
        raise


def plot_forecast_only(future):
    """Отдельный график только для прогноза"""
    if future.empty or 'Date' not in future.columns or 'Predicted' not in future.columns:
        logger.warning("No future predictions data available")
        return

    try:
        plt.figure(figsize=(14, 8))

        # Прогнозы
        plt.plot(future['Date'], future['Predicted'],
                 label='3-Month Forecast',
                 color='#9467bd',
                 marker='o',
                 linestyle='-',
                 linewidth=2.5)

        # Добавляем линию тренда
        z = np.polyfit(range(len(future)), future['Predicted'], 1)
        p = np.poly1d(z)
        plt.plot(future['Date'], p(range(len(future))),
                 linestyle='--',
                 color='red',
                 linewidth=2,
                 label=f'Trend Line (Slope: {z[0]:.2f})')

        # Отмечаем максимумы и минимумы
        max_idx = future['Predicted'].idxmax()
        min_idx = future['Predicted'].idxmin()

        plt.scatter(future.loc[max_idx, 'Date'], future.loc[max_idx, 'Predicted'],
                    color='green', s=100, zorder=5,
                    label=f'Max: {future.loc[max_idx, "Predicted"]:.2f} on {future.loc[max_idx, "Date"].strftime("%d %b")}')

        plt.scatter(future.loc[min_idx, 'Date'], future.loc[min_idx, 'Predicted'],
                    color='red', s=100, zorder=5,
                    label=f'Min: {future.loc[min_idx, "Predicted"]:.2f} on {future.loc[min_idx, "Date"].strftime("%d %b")}')

        # Добавляем аннотации для максимума и минимума
        plt.annotate(f'{future.loc[max_idx, "Predicted"]:.2f}',
                     (future.loc[max_idx, 'Date'], future.loc[max_idx, 'Predicted']),
                     xytext=(0, 15), textcoords='offset points',
                     ha='center', va='bottom',
                     fontweight='bold')

        plt.annotate(f'{future.loc[min_idx, "Predicted"]:.2f}',
                     (future.loc[min_idx, 'Date'], future.loc[min_idx, 'Predicted']),
                     xytext=(0, -15), textcoords='offset points',
                     ha='center', va='top',
                     fontweight='bold')

        # Настройки графика
        start_date = future['Date'].min()
        end_date = future['Date'].max()
        plt.title(f'3-Month S&P 500 Forecast\n{start_date.strftime("%d %b %Y")} - {end_date.strftime("%d %b %Y")}',
                  fontsize=16, pad=20)

        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Predicted Price (USD)', fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Форматирование дат для недель
        ax = plt.gca()

        # Для 3-месячного прогноза:
        # - Major ticks для начала каждого месяца
        # - Minor ticks для каждого понедельника
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.xaxis.set_minor_locator(WeekdayLocator(byweekday=0))

        # Удобное отображение диапазона цен - немного выше максимума и ниже минимума
        y_min = future['Predicted'].min() * 0.99
        y_max = future['Predicted'].max() * 1.01
        plt.ylim(y_min, y_max)

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Сохранение
        os.makedirs("reports", exist_ok=True)
        plt.savefig("reports/three_month_forecast.png",
                    dpi=300,
                    bbox_inches='tight')
        plt.close()
        logger.info("3-month forecast plot saved successfully")

    except Exception as e:
        logger.error(f"Error generating 3-month forecast plot: {str(e)}")
        raise


def plot_forecast_statistics(future):
    """Визуализация статистики прогноза"""
    if future.empty or 'Date' not in future.columns or 'Predicted' not in future.columns:
        logger.warning("No future predictions data available")
        return

    try:
        # Добавляем полезные метрики для месячных периодов
        future['Month'] = future['Date'].dt.to_period('M')
        monthly_stats = future.groupby('Month').agg({
            'Predicted': ['first', 'last', 'mean', 'min', 'max', 'std']
        })
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
        monthly_stats['Monthly_Change'] = monthly_stats['Predicted_last'] / monthly_stats['Predicted_first'] - 1
        monthly_stats['Volatility'] = monthly_stats['Predicted_std'] / monthly_stats['Predicted_mean']

        # Создаем DataFrame для графика
        months = [str(m) for m in monthly_stats.index.values]

        # График месячных изменений
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 1, 1)
        bars = plt.bar(months, monthly_stats['Monthly_Change'] * 100, color=[
            'green' if x > 0 else 'red' for x in monthly_stats['Monthly_Change']
        ])

        plt.title('Predicted Monthly Changes', fontsize=14)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Percentage Change (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')

        # Добавляем подписи со значениями
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.,
                     0.1 + height if height >= 0 else height - 0.5,
                     f'{height:.1f}%',
                     ha='center', va='bottom' if height >= 0 else 'top',
                     fontweight='bold')

        # График ценовых диапазонов по месяцам
        plt.subplot(2, 1, 2)

        for i, month in enumerate(months):
            min_price = monthly_stats.loc[monthly_stats.index[i], 'Predicted_min']
            max_price = monthly_stats.loc[monthly_stats.index[i], 'Predicted_max']
            mean_price = monthly_stats.loc[monthly_stats.index[i], 'Predicted_mean']

            plt.plot([i, i], [min_price, max_price], 'k-', alpha=0.7)
            plt.plot([i - 0.1, i + 0.1], [min_price, min_price], 'k-', alpha=0.7)
            plt.plot([i - 0.1, i + 0.1], [max_price, max_price], 'k-', alpha=0.7)
            plt.scatter([i], [mean_price], color='blue', s=50, zorder=3)

            # Подписи для мин/макс значений
            plt.text(i, max_price, f'{max_price:.1f}', ha='center', va='bottom')
            plt.text(i, min_price, f'{min_price:.1f}', ha='center', va='top')

        plt.title('Predicted Price Range by Month', fontsize=14)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.xticks(range(len(months)), months)

        plt.tight_layout()
        plt.savefig("reports/monthly_forecast_stats.png",
                    dpi=300,
                    bbox_inches='tight')
        plt.close()
        logger.info("Monthly forecast statistics plot saved successfully")

    except Exception as e:
        logger.error(f"Error generating monthly forecast statistics: {str(e)}")
        raise


def plot_error_analysis(preds):
    """Расширенный анализ ошибок"""
    try:
        preds['Error'] = preds['Actual'] - preds['Predicted']
        preds['Absolute_Error'] = preds['Error'].abs()
        preds['Percent_Error'] = (preds['Error'] / preds['Actual']) * 100

        # 1. Распределение ошибок
        plt.figure(figsize=(16, 12))

        plt.subplot(2, 2, 1)
        sns.histplot(preds['Error'], bins=20, kde=True, color='#d62728')
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.3)

        # 2. Ошибки по времени
        plt.subplot(2, 2, 2)
        plt.plot(preds['Date'], preds['Error'], color='#9467bd')
        plt.title('Prediction Errors Over Time')
        plt.xlabel('Date')
        plt.ylabel('Error')
        plt.grid(True, linestyle='--', alpha=0.3)

        # 3. Абсолютные ошибки по времени
        plt.subplot(2, 2, 3)
        plt.plot(preds['Date'], preds['Absolute_Error'], color='#e377c2')
        plt.title('Absolute Prediction Errors Over Time')
        plt.xlabel('Date')
        plt.ylabel('Absolute Error')
        plt.grid(True, linestyle='--', alpha=0.3)

        # 4. Автокорреляция ошибок
        plt.subplot(2, 2, 4)
        plot_acf(preds['Error'], lags=30, ax=plt.gca())
        plt.title('Autocorrelation of Prediction Errors')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig("reports/error_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Error analysis plots saved successfully")

    except Exception as e:
        logger.error(f"Error generating error analysis: {str(e)}")
        raise


def plot_feature_importance():
    """Визуализация важности признаков"""
    try:
        with sqlite3.connect("data/sp500.db") as conn:
            # Проверяем, существует ли таблица feature_importance
            cursor = conn.cursor()
            table_exists = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feature_importance'"
            ).fetchone()

            if not table_exists:
                logger.warning("Feature importance table not found in database")
                return

            fi = pd.read_sql("SELECT * FROM feature_importance", conn)

            if fi.empty:
                logger.warning("Feature importance table is empty")
                return

        plt.figure(figsize=(12, 8))
        fi = fi.sort_values('Importance', ascending=False).head(15)

        sns.barplot(x='Importance', y='Feature', data=fi, palette='viridis')
        plt.title('Top 15 Most Important Features', fontsize=16)
        plt.xlabel('Feature Importance', fontsize=14)
        plt.ylabel('Feature Name', fontsize=14)
        plt.grid(True, axis='x', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig("reports/feature_importance.png",
                    dpi=300,
                    bbox_inches='tight')
        plt.close()
        logger.info("Feature importance plot saved successfully")

    except Exception as e:
        logger.error(f"Error generating feature importance plot: {str(e)}")


def create_summary_report(df, preds, metrics, future):
    """Создание текстового отчета с результатами"""
    try:
        with open("reports/summary_report.txt", "w") as f:
            f.write("========== S&P 500 FORECAST SUMMARY ==========\n\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Параметры модели
            f.write("MODEL PERFORMANCE\n")
            f.write("-----------------\n")
            f.write(f"RMSE: {metrics.loc[0, 'Test_RMSE']:.4f}\n")
            f.write(f"MAE: {metrics.loc[0, 'Test_MAE']:.4f}\n")
            f.write(f"R²: {metrics.loc[0, 'Test_R2']:.4f}\n\n")

            # Информация о прогнозе
            if not future.empty:
                f.write("FORECAST INFORMATION\n")
                f.write("--------------------\n")
                f.write(
                    f"Forecast period: {future['Date'].min().strftime('%Y-%m-%d')} to {future['Date'].max().strftime('%Y-%m-%d')}\n")
                f.write(f"Number of days: {len(future)}\n")
                f.write(f"Starting price: {future['Predicted'].iloc[0]:.2f}\n")
                f.write(f"Ending price: {future['Predicted'].iloc[-1]:.2f}\n")

                # Расчет изменения в процентах
                total_change_pct = (future['Predicted'].iloc[-1] / future['Predicted'].iloc[0] - 1) * 100
                f.write(f"Total change: {total_change_pct:.2f}%\n\n")

                # Максимум и минимум
                max_idx = future['Predicted'].idxmax()
                min_idx = future['Predicted'].idxmin()
                f.write(
                    f"Maximum price: {future.loc[max_idx, 'Predicted']:.2f} on {future.loc[max_idx, 'Date'].strftime('%Y-%m-%d')}\n")
                f.write(
                    f"Minimum price: {future.loc[min_idx, 'Predicted']:.2f} on {future.loc[min_idx, 'Date'].strftime('%Y-%m-%d')}\n\n")

                # Среднее, медиана, и стандартное отклонение
                f.write(f"Average price: {future['Predicted'].mean():.2f}\n")
                f.write(f"Median price: {future['Predicted'].median():.2f}\n")
                f.write(f"Standard deviation: {future['Predicted'].std():.2f}\n\n")

                # Месячная статистика
                f.write("MONTHLY BREAKDOWN\n")
                f.write("-----------------\n")

                future['Month'] = future['Date'].dt.to_period('M')
                monthly_stats = future.groupby('Month').agg({
                    'Predicted': ['first', 'last', 'mean', 'min', 'max']
                })
                monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
                monthly_stats['Monthly_Change'] = monthly_stats['Predicted_last'] / monthly_stats['Predicted_first'] - 1

                for month, row in monthly_stats.iterrows():
                    f.write(
                        f"{month}: {row['Monthly_Change'] * 100:.2f}% change (Start: {row['Predicted_first']:.2f}, End: {row['Predicted_last']:.2f})\n")
                    f.write(
                        f"    Min: {row['Predicted_min']:.2f}, Max: {row['Predicted_max']:.2f}, Avg: {row['Predicted_mean']:.2f}\n")

                f.write("\n")

            # Данные для визуализаций
            f.write("VISUALIZATION NOTES\n")
            f.write("-------------------\n")
            f.write("The following visualization files have been created in the reports/ directory:\n")
            f.write("- predictions_with_ci.png: Historical data with predictions and forecast\n")
            f.write("- three_month_forecast.png: Detailed 3-month forecast with trend line\n")
            f.write("- monthly_forecast_stats.png: Monthly statistics of the forecast\n")
            f.write("- error_analysis.png: Analysis of prediction errors\n")
            f.write("- feature_importance.png: Importance of different features in the model\n\n")

            f.write("=================================================\n")

        logger.info("Summary report saved to reports/summary_report.txt")

    except Exception as e:
        logger.error(f"Error creating summary report: {str(e)}")


if __name__ == "__main__":
    try:
        logger.info("Starting visualization process...")

        # Создаем директорию для отчетов, если она не существует
        os.makedirs("reports", exist_ok=True)

        # Загрузка данных
        df, preds, metrics, future = load_data()

        # Создание визуализаций
        plot_predictions(df, preds, metrics, future)

        if not future.empty:
            plot_forecast_only(future)
            plot_forecast_statistics(future)

        plot_error_analysis(preds)

        try:
            plot_feature_importance()
        except Exception as e:
            logger.warning(f"Could not create feature importance plot: {str(e)}")

        # Создание текстового отчета
        create_summary_report(df, preds, metrics, future)

        logger.info("All visualizations saved to reports/ folder")
        print("Visualizations and reports saved to reports/ folder")

    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}", exc_info=True)
        raise