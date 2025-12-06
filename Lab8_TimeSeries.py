"""
Лабораторна робота №8
ПРОЕКТНИЙ ПРАКТИКУМ З ПРОГНОЗУВАННЯ ЕКОНОМІЧНИХ ПОКАЗНИКІВ
Аналіз Data_Set_11.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

matplotlib.use('Agg')
plt.rcParams['figure.figsize'] = (18, 10)
plt.rcParams['font.size'] = 10

GRAPHS_DIR = 'Graphs'

os.makedirs(GRAPHS_DIR, exist_ok=True)

class ProcessClassifier:
    """Клас для автоматичного визначення типу часового процесу"""

    @staticmethod
    def determine_process_type(data, adf_result=None, seasonality_result=None):
        """
        Автоматичне визначення типу часового ряду на основі статистичних характеристик

        Parameters:
        -----------
        data : pd.Series
            Часовий ряд
        adf_result : dict, optional
            Результат ADF-тесту стаціонарності
        seasonality_result : dict, optional
            Результат аналізу сезонності

        Returns:
        --------
        dict : Характеристики процесу та його тип
        """
        characteristics = {}

        values = data.values
        mean_val = np.mean(values)
        std_val = np.std(values)

        cv = (std_val / abs(mean_val) * 100) if mean_val != 0 else np.inf
        characteristics['cv'] = cv
        characteristics['mean'] = mean_val
        characteristics['std'] = std_val

        if isinstance(data.index, pd.DatetimeIndex):
            time_diffs = np.diff(data.index.to_numpy().astype('datetime64[s]')).astype(float)
            avg_interval_seconds = np.mean(time_diffs)
            avg_interval_days = avg_interval_seconds / (24 * 3600)
            characteristics['interval_days'] = avg_interval_days

            if avg_interval_days < 1:
                frequency = "Високочастотний"
            elif avg_interval_days < 7:
                frequency = "Щоденний"
            elif avg_interval_days < 31:
                frequency = "Тижневий"
            else:
                frequency = "Низькочастотний"
        else:
            frequency = "Дискретний"
            characteristics['interval_days'] = None

        if cv > 100:
            volatility = "високостохастичний"
        elif cv > 50:
            volatility = "стохастичний"
        elif cv > 20:
            volatility = "коливальний"
        elif cv > 10:
            volatility = "слабкоколивальний"
        else:
            volatility = "стабільний"

        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        characteristics['slope'] = slope

        normalized_slope = abs(slope) / (std_val + 1e-10)
        characteristics['normalized_slope'] = normalized_slope

        if normalized_slope > 0.01:
            if slope > 0:
                trend = "зростаючий тренд"
                trend_short = "зростаючий"
            else:
                trend = "спадаючий тренд"
                trend_short = "спадаючий"
        else:
            trend = "без тренду"
            trend_short = "стабільний"

        diffs = np.diff(values)
        is_monotonic_increasing = np.all(diffs >= -1e-10)
        is_monotonic_decreasing = np.all(diffs <= 1e-10)

        if is_monotonic_increasing and normalized_slope > 0.01:
            monotonicity = "монотонно зростаючий"
        elif is_monotonic_decreasing and normalized_slope > 0.01:
            monotonicity = "монотонно спадаючий"
        else:
            monotonicity = None

        has_negative = np.any(values < 0)
        min_val = np.min(values)
        characteristics['has_negative'] = has_negative
        characteristics['min_value'] = min_val

        unique_ratio = len(np.unique(values)) / len(values)
        is_integer = np.all(np.abs(values - np.round(values)) < 1e-10)
        is_positive = not has_negative
        is_counting = unique_ratio < 0.15 and is_integer and is_positive

        characteristics['unique_ratio'] = unique_ratio
        characteristics['is_integer'] = is_integer
        characteristics['is_counting'] = is_counting

        if adf_result:
            is_stationary = adf_result['is_stationary']
            stationarity = "стаціонарний" if is_stationary else "нестаціонарний"
            characteristics['adf_pvalue'] = adf_result['p_value']
        else:
            stationarity = None
            is_stationary = None

        if seasonality_result and seasonality_result.get('success'):
            has_seasonality = seasonality_result['has_seasonality']
            seasonal_strength = seasonality_result['seasonal_strength']
            characteristics['seasonal_strength'] = seasonal_strength
        else:
            has_seasonality = False
            seasonal_strength = 0

        type_components = []

        type_components.append(frequency)

        if monotonicity:
            type_components.append(monotonicity)
        else:
            type_components.append(volatility)

        if is_counting:
            type_components.append("лічильний")

        if has_negative:
            type_components.append("з негативними значеннями")

        if not monotonicity and normalized_slope > 0.01:
            type_components.append(trend)

        if has_seasonality:
            type_components.append(f"сезонний ({seasonal_strength:.1f}%)")

        if stationarity:
            type_components.append(f"({stationarity})")

        process_type = " ".join(type_components)

        return {
            'type': process_type,
            'frequency': frequency,
            'volatility': volatility,
            'trend': trend_short,
            'monotonic': monotonicity,
            'has_negative': has_negative,
            'is_counting': is_counting,
            'stationarity': stationarity,
            'has_seasonality': has_seasonality,
            'characteristics': characteristics
        }


class MultiProcessAnalyzer:
    """Клас для багатофакторного аналізу часових рядів з автокласифікацією"""

    def __init__(self, filename):
        self.filename = filename
        self.orders = None
        self.returns = None
        self.users = None
        self.df_merged = None
        self.time_series = {}
        self.models = {}
        self.adf_results = {}
        self.seasonality_results = {}
        self.classifier = ProcessClassifier()

    def load_data(self):
        """Завантаження та підготовка даних"""
        print("ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ")
        try:
            self.orders = pd.read_excel(self.filename, sheet_name='Orders')
            self.returns = pd.read_excel(self.filename, sheet_name='Returns')
            self.users = pd.read_excel(self.filename, sheet_name='Users')

            print(f"\nЗавантажено Orders: {len(self.orders)} записів")
            print(f"Завантажено Returns: {len(self.returns)} записів")
            print(f"Завантажено Users: {len(self.users)} записів")

            self.orders['Order Date'] = pd.to_datetime(self.orders['Order Date'])
            self.orders['Ship Date'] = pd.to_datetime(self.orders['Ship Date'])

            self.orders['Shipping_Days'] = (self.orders['Ship Date'] - self.orders['Order Date']).dt.days
            self.orders['Profit_Margin'] = (self.orders['Profit'] / self.orders['Sales'] * 100)

            self.df_merged = self.orders.copy()
            self.df_merged['Returned'] = self.df_merged['Order ID'].isin(self.returns['Order ID'])

            self.df_merged['Year'] = self.df_merged['Order Date'].dt.year
            self.df_merged['Month'] = self.df_merged['Order Date'].dt.month
            self.df_merged['Quarter'] = self.df_merged['Order Date'].dt.quarter
            self.df_merged['YearMonth'] = self.df_merged['Order Date'].dt.to_period('M')
            self.df_merged['DayOfWeek'] = self.df_merged['Order Date'].dt.dayofweek

        except Exception as e:
            print(f"Помилка: {e}")
            return False

        return True

    def create_time_series(self):
        """Створення множини часових рядів з БАЗОВОЮ класифікацією (без ADF/сезонності)"""
        print("\n" + "-" * 80)
        print("СТВОРЕННЯ МНОЖИНИ ЧАСОВИХ РЯДІВ")

        daily_sales = self.df_merged.groupby('Order Date')['Sales'].sum()
        sales_classification = self.classifier.determine_process_type(daily_sales)

        self.time_series['Daily_Sales'] = {
            'data': daily_sales,
            'name': 'Щоденні продажі',
            'type': sales_classification['type'],
            'unit': '$',
            'classification': sales_classification
        }

        daily_profit = self.df_merged.groupby('Order Date')['Profit'].sum()
        profit_classification = self.classifier.determine_process_type(daily_profit)

        self.time_series['Daily_Profit'] = {
            'data': daily_profit,
            'name': 'Щоденний прибуток',
            'type': profit_classification['type'],
            'unit': '$',
            'classification': profit_classification
        }

        daily_orders = self.df_merged.groupby('Order Date')['Order ID'].count()
        orders_classification = self.classifier.determine_process_type(daily_orders)

        self.time_series['Daily_Orders'] = {
            'data': daily_orders,
            'name': 'Кількість замовлень на день',
            'type': orders_classification['type'],
            'unit': 'шт',
            'classification': orders_classification
        }

        daily_margin = self.df_merged.groupby('Order Date')['Profit_Margin'].mean()
        margin_classification = self.classifier.determine_process_type(daily_margin)

        self.time_series['Daily_Margin'] = {
            'data': daily_margin,
            'name': 'Середня маржа прибутку',
            'type': margin_classification['type'],
            'unit': '%',
            'classification': margin_classification
        }

        daily_shipping = self.df_merged.groupby('Order Date')['Shipping_Days'].mean()
        shipping_classification = self.classifier.determine_process_type(daily_shipping)

        self.time_series['Daily_Shipping'] = {
            'data': daily_shipping,
            'name': 'Середній час доставки',
            'type': shipping_classification['type'],
            'unit': 'днів',
            'classification': shipping_classification
        }

        daily_returns = self.df_merged.groupby('Order Date')['Returned'].mean() * 100
        returns_classification = self.classifier.determine_process_type(daily_returns)

        self.time_series['Daily_Returns'] = {
            'data': daily_returns,
            'name': 'Коефіцієнт повернень',
            'type': returns_classification['type'],
            'unit': '%',
            'classification': returns_classification
        }

        daily_avg_check = self.df_merged.groupby('Order Date')['Sales'].mean()
        avgcheck_classification = self.classifier.determine_process_type(daily_avg_check)

        self.time_series['Daily_AvgCheck'] = {
            'data': daily_avg_check,
            'name': 'Середній чек',
            'type': avgcheck_classification['type'],
            'unit': '$',
            'classification': avgcheck_classification
        }

        cumulative_sales = daily_sales.cumsum()
        cumulative_classification = self.classifier.determine_process_type(cumulative_sales)

        self.time_series['Cumulative_Sales'] = {
            'data': cumulative_sales,
            'name': 'Кумулятивні продажі',
            'type': cumulative_classification['type'],
            'unit': '$',
            'classification': cumulative_classification
        }

        monthly_data = self.df_merged.groupby('YearMonth').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'count',
            'Profit_Margin': 'mean',
            'Shipping_Days': 'mean',
            'Returned': lambda x: (x.sum() / len(x) * 100)
        })
        monthly_data.columns = ['Sales', 'Profit', 'Orders_Count', 'Avg_Margin_%', 'Avg_Shipping_Days', 'Return_Rate_%']
        monthly_data['Avg_Check'] = monthly_data['Sales'] / monthly_data['Orders_Count']
        unique_dates_count = self.df_merged['Order Date'].nunique()
        self.monthly_data = monthly_data

        print(f"\nСтворено {len(self.time_series)} часових рядів:")
        print(f"Кількість унікальних дат - {unique_dates_count}")
        print(f"1 унікальна дата - 1 точка")
        for key, ts in self.time_series.items():
            print(f"\n{ts['name']}: {len(ts['data'])} точок")
            print(f"Тип: {ts['type']}")
            print(f"CV: {ts['classification']['characteristics']['cv']:.2f}%")
            print(f"Тренд: {ts['classification']['trend']}")

        return self.time_series

    def perform_adf_test(self, data, process_name):
        """Виконання розширеного тесту Дікі-Фуллера (ADF)"""
        try:
            result = adfuller(data, autolag='AIC')

            adf_stat = result[0]
            p_value = result[1]
            critical_values = result[4]

            is_stationary = p_value < 0.05

            interpretation = {
                'adf_statistic': adf_stat,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'conclusion': 'СТАЦІОНАРНИЙ' if is_stationary else 'НЕСТАЦІОНАРНИЙ'
            }

            return interpretation

        except Exception as e:
            print(f"Помилка ADF-тесту для {process_name}: {e}")
            return None

    def analyze_seasonality(self, data, process_name, period=7):
        """Декомпозиція часового ряду на компоненти"""
        try:
            if len(data) < period * 2:
                return {
                    'success': False,
                    'reason': f'Недостатньо даних для аналізу (потрібно мінімум {period * 2} точок)'
                }

            data_clean = data.dropna()

            if len(data_clean) < period * 2:
                return {
                    'success': False,
                    'reason': 'Занадто багато пропусків у даних'
                }

            decomposition = seasonal_decompose(data_clean, model='additive', period=period, extrapolate_trend='freq')

            seasonal_strength = np.var(decomposition.seasonal) / np.var(data_clean) * 100
            trend_strength = np.var(decomposition.trend.dropna()) / np.var(data_clean) * 100
            residual_strength = np.var(decomposition.resid.dropna()) / np.var(data_clean) * 100

            return {
                'success': True,
                'decomposition': decomposition,
                'seasonal_strength': seasonal_strength,
                'trend_strength': trend_strength,
                'residual_strength': residual_strength,
                'has_seasonality': seasonal_strength > 10
            }

        except Exception as e:
            print(f"Помилка аналізу сезонності для {process_name}: {e}")
            return {'success': False, 'reason': str(e)}

    def test_stationarity_and_seasonality(self):
        """Тестування стаціонарності та сезонності"""
        print("\n" + "-" * 80)
        print("Аналіз СТАЦІОНАРНОСТІ (ADF-ТЕСТ)")

        for key, ts in self.time_series.items():
            data = ts['data'].values
            name = ts['name']

            print(f"\n{'-' * 40}")
            print(f"ПРОЦЕС: {name}")
            print(f"{'-' * 40}")

            adf_result = self.perform_adf_test(data, name)

            if adf_result:
                self.adf_results[key] = adf_result

                print(f"\n ADF-статистика: {adf_result['adf_statistic']:.4f}")
                print(f"P-value: {adf_result['p_value']:.6f}")

                if adf_result['is_stationary']:
                    print(f"\nРяд: {adf_result['conclusion']}")
                    print(f"P-value ({adf_result['p_value']:.6f}) < 0.05")
                else:
                    print(f"\nВИСНОВОК: {adf_result['conclusion']}")
                    print(f"P-value ({adf_result['p_value']:.6f}) ≥ 0.05")

        print("\n" + "-" * 80)
        print("АНАЛІЗ СЕЗОННОСТІ:")

        for key, ts in self.time_series.items():
            name = ts['name']
            data = ts['data']

            print(f"\n{'-' * 40}")
            print(f"ПРОЦЕС: {name}")
            print(f"{'-' * 40}")

            period = 7
            seasonality_result = self.analyze_seasonality(data, name, period)
            self.seasonality_results[key] = seasonality_result

            if seasonality_result['success']:
                print(f"\nВнесок компонентів у загальну варіацію:")
                print(f"Тренд:      {seasonality_result['trend_strength']:.2f}%")
                print(f"Сезонність: {seasonality_result['seasonal_strength']:.2f}%")
                print(f"Залишки:    {seasonality_result['residual_strength']:.2f}%")

                if seasonality_result['has_seasonality']:
                    print(f"\nВиявлено СЕЗОННІСТЬ")
                    print(f"Сила сезонності: {seasonality_result['seasonal_strength']:.2f}% (> 10%)")
                else:
                    print(f"\nСезонність НЕ виявлено або слабка")
                    print(f"Сила сезонності: {seasonality_result['seasonal_strength']:.2f}% (< 10%)")
            else:
                print(f"\nДекомпозицію не виконано: {seasonality_result['reason']}")

        for key, ts in self.time_series.items():
            adf_result = self.adf_results.get(key)
            seasonality_result = self.seasonality_results.get(key)

            updated_classification = self.classifier.determine_process_type(
                ts['data'],
                adf_result=adf_result,
                seasonality_result=seasonality_result
            )

            old_type = ts['type']
            ts['type'] = updated_classification['type']
            ts['classification'] = updated_classification

    def visualize_stationarity_tests(self):
        """Візуалізація результатів тестів стаціонарності"""
        process_names = []
        p_values = []
        conclusions = []

        for key, result in self.adf_results.items():
            ts = self.time_series[key]
            process_names.append(ts['name'].replace('Щоденні ', '').replace('Щоденний ', '').replace('Кількість ', ''))
            p_values.append(result['p_value'])
            conclusions.append(result['is_stationary'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('РЕЗУЛЬТАТИ ТЕСТУВАННЯ СТАЦІОНАРНОСТІ (ADF-ТЕСТ)', fontsize=16, fontweight='bold')

        colors = ['green' if c else 'red' for c in conclusions]
        bars = ax1.barh(process_names, p_values, color=colors, alpha=0.7)
        ax1.axvline(x=0.05, color='blue', linestyle='--', linewidth=2, label='Поріг значущості (α=0.05)')
        ax1.set_xlabel('P-value')
        ax1.set_title('P-values ADF-тесту\n(зелений = стаціонарний, червоний = нестаціонарний)')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_xlim(0, max(p_values) * 1.1)

        stationary_count = sum(conclusions)
        non_stationary_count = len(conclusions) - stationary_count

        ax2.pie([stationary_count, non_stationary_count],
                labels=['Стаціонарні', 'Нестаціонарні'],
                autopct='%1.1f%%',
                colors=['green', 'red'],
                startangle=90,
                explode=(0.05, 0))
        ax2.set_title(f'Розподіл процесів за стаціонарністю\n(всього: {len(conclusions)} процесів)')

        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, '2_1_stationarity_tests.png'), dpi=300, bbox_inches='tight')
        print(f"\nГрафік збережено: {GRAPHS_DIR}/2_1_stationarity_tests.png")
        plt.close()

    def visualize_seasonality(self):
        """Візуалізація декомпозиції для 4 ключових процесів"""
        key_processes = ['Daily_Sales', 'Daily_Profit', 'Daily_Orders', 'Daily_AvgCheck']

        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle('АНАЛІЗ СЕЗОННОСТІ КЛЮЧОВИХ ПРОЦЕСІВ', fontsize=18, fontweight='bold')

        for i, key in enumerate(key_processes):
            if key in self.seasonality_results and self.seasonality_results[key]['success']:
                ts = self.time_series[key]
                result = self.seasonality_results[key]
                decomp = result['decomposition']

                axes[i, 0].plot(decomp.observed, linewidth=1, color='blue')
                axes[i, 0].set_ylabel(ts['name'], fontweight='bold')
                axes[i, 0].set_title('Оригінальний ряд')
                axes[i, 0].grid(True, alpha=0.3)

                axes[i, 1].plot(decomp.trend, linewidth=1.5, color='red')
                axes[i, 1].set_title(f'Тренд ({result["trend_strength"]:.1f}%)')
                axes[i, 1].grid(True, alpha=0.3)

                axes[i, 2].plot(decomp.seasonal, linewidth=1, color='green')
                axes[i, 2].set_title(f'Сезонність ({result["seasonal_strength"]:.1f}%)')
                axes[i, 2].grid(True, alpha=0.3)

                axes[i, 3].plot(decomp.resid, linewidth=0.5, color='gray', alpha=0.7)
                axes[i, 3].set_title(f'Залишки ({result["residual_strength"]:.1f}%)')
                axes[i, 3].grid(True, alpha=0.3)

                if i < 3:
                    for ax in axes[i]:
                        ax.set_xticklabels([])

        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, '2_2_seasonality_decomposition.png'), dpi=300, bbox_inches='tight')
        print(f"\nГрафік збережено: {GRAPHS_DIR}/2_2_seasonality_decomposition.png")
        plt.close()

    def analyze_properties(self):
        """Детальний аналіз властивостей часових рядів"""
        print("\n" + "-" * 80)
        print("АНАЛІЗ ВЛАСТИВОСТЕЙ ЧАСОВИХ РЯДІВ")

        for key, ts in self.time_series.items():
            data = ts['data'].values
            name = ts['name']
            classification = ts['classification']

            print(f"\n{'-' * 40}")
            print(f"ПРОЦЕС: {name}")
            print(f"{'-' * 40}")

            print(f"\nТип процесу : {ts['type']}")
            print(f"Основні характеристики:")
            print(f"Кількість спостережень: {len(data)}")
            print(f"Середнє значення: {np.mean(data):.2f} {ts['unit']}")
            print(f"Медіана: {np.median(data):.2f} {ts['unit']}")
            print(f"СКВ: {np.std(data):.2f} {ts['unit']}")
            print(f"Мін: {np.min(data):.2f} {ts['unit']}")
            print(f"Макс: {np.max(data):.2f} {ts['unit']}")
            print(f"Розмах: {np.max(data) - np.min(data):.2f} {ts['unit']}")
            print(f"Коефіцієнт варіації: {classification['characteristics']['cv']:.2f}%")

            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            outliers = np.sum((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR))
            print(f"\nАномалії:")
            print(f"Кількість викидів: {outliers} ({outliers / len(data) * 100:.2f}%)")

            ts['statistics'] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'cv': classification['characteristics']['cv'],
                'outliers_count': outliers
            }

    def visualize_all_processes(self):
        """Візуалізація всіх часових процесів"""

        key_processes = ['Daily_Sales', 'Daily_Profit', 'Daily_Orders',
                         'Daily_Margin', 'Daily_Shipping', 'Daily_Returns',
                         'Daily_AvgCheck', 'Cumulative_Sales']

        fig, axes = plt.subplots(4, 2, figsize=(20, 16))
        fig.suptitle('МНОЖИНА ЧАСОВИХ РЯДІВ',
                     fontsize=18, fontweight='bold')

        axes = axes.flatten()

        for i, key in enumerate(key_processes):
            if key in self.time_series:
                ts = self.time_series[key]
                data = ts['data']

                axes[i].plot(data.index, data.values, linewidth=1, alpha=0.7, color=f'C{i}')

                x = np.arange(len(data))
                coeffs = np.polyfit(x, data.values, 1)
                trend_line = coeffs[0] * x + coeffs[1]
                axes[i].plot(data.index, trend_line, 'r--', linewidth=2, label='Тренд')

                mean_val = data.values.mean()
                axes[i].axhline(y=mean_val, color='g', linestyle=':', linewidth=2, label=f'Середнє: {mean_val:.2f}')

                axes[i].set_title(f'{ts["name"]}',fontsize=10, fontweight='bold')
                axes[i].set_ylabel(f'{ts["unit"]}')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(fontsize=8)
                axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, '1_all_processes_visualization.png'), dpi=300, bbox_inches='tight')
        print(f"\nГрафіки збережено: {GRAPHS_DIR}/1_all_processes_visualization.png")
        plt.close()

    def statistical_analysis(self):
        """Статистичний аналіз та порівняння процесів"""
        print("\n" + "-" * 80)
        print("СТАТИСТИЧНИЙ АНАЛІЗ ТА КОРЕЛЯЦІЇ")

        combined_df = pd.DataFrame()

        for key, ts in self.time_series.items():
            if 'Daily' in key:
                monthly = ts['data'].resample('M').mean()
                combined_df[key] = monthly

        print("\nКореляційний аналіз (місячна агрегація):")
        corr_matrix = combined_df.corr()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(corr_matrix.round(3))

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('СТАТИСТИЧНИЙ АНАЛІЗ МНОЖИНИ ПРОЦЕСІВ', fontsize=16, fontweight='bold')

        im = axes[0].imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        axes[0].set_xticks(range(len(corr_matrix.columns)))
        axes[0].set_yticks(range(len(corr_matrix.columns)))
        axes[0].set_xticklabels([col.replace('Daily_', '') for col in corr_matrix.columns], rotation=45, ha='right')
        axes[0].set_yticklabels([col.replace('Daily_', '') for col in corr_matrix.columns])
        axes[0].set_title('Матриця кореляцій між процесами')

        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                text = axes[0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=axes[0])

        cv_data = []
        process_names = []
        for key, ts in self.time_series.items():
            if 'Daily' in key:
                cv = ts['classification']['characteristics']['cv']
                cv_data.append(cv)
                process_names.append(key.replace('Daily_', ''))

        axes[1].barh(process_names, cv_data, color='steelblue')
        axes[1].set_xlabel('Коефіцієнт варіації (%)')
        axes[1].set_title('Мінливість процесів (коефіцієнт варіації)')
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, '3_statistical_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"\nГрафіки збережено: {GRAPHS_DIR}/3_statistical_analysis.png")
        plt.close()

        print("\nНайсильніші кореляції між процесами:")
        corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr_pairs.append({
                    'Process_1': corr_matrix.columns[i].replace('Daily_', ''),
                    'Process_2': corr_matrix.columns[j].replace('Daily_', ''),
                    'Correlation': corr_matrix.iloc[i, j]
                })

        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
        print(corr_df.head(10))

    def build_forecast_models(self):
        """Побудова прогнозних моделей для ключових процесів"""
        print("\n" + "-" * 80)
        print("ПОБУДОВА МНК МОДЕЛЕЙ ДЛЯ ПРОГНОЗУВАННЯ")

        forecast_processes = {
            'Sales': self.monthly_data['Sales'].values,
            'Profit': self.monthly_data['Profit'].values,
            'Orders': self.monthly_data['Orders_Count'].values,
            'Avg_Margin': self.monthly_data['Avg_Margin_%'].values,
            'Avg_Check': self.monthly_data['Avg_Check'].values
        }

        n = len(self.monthly_data)
        x = np.arange(n)

        for name, data in forecast_processes.items():
            print(f"\n{'-' * 40}")
            print(f"МОДЕЛЬ: {name}")
            print(f"{'-' * 40}")

            poly_degree = 2
            coeffs = np.polyfit(x, data, poly_degree)
            poly = np.poly1d(coeffs)

            print("Математична модель (рівняння тренду):")

            print(f"Формула: y = ({coeffs[0]:.4f}) * x^2 + ({coeffs[1]:.4f}) * x + ({coeffs[2]:.4f})")
            print(f"Де: a={coeffs[0]:.4f}, b={coeffs[1]:.4f}, c={coeffs[2]:.4f}")
            fitted = poly(x)

            ss_res = np.sum((data - fitted) ** 2)
            ss_tot = np.sum((data - data.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean((data - fitted) ** 2))
            mape = np.mean(np.abs((data - fitted) / (data + 1e-10))) * 100

            print(f"R² (коефіцієнт детермінації): {r2:.4f}")
            print(f"RMSE (середньоквадратична помилка): {rmse:.2f}")
            print(f"MAPE (середня абсолютна відсоткова помилка): {mape:.2f}%")

            self.models[name] = {
                'poly': poly,
                'data': data,
                'fitted': fitted,
                'r2': r2,
                'rmse': rmse,
                'mape': mape
            }

    def forecast_and_visualize(self, periods=6):
        """Прогнозування та візуалізація"""
        print("\n" + "-" * 80)
        print(f"ПРОГНОЗУВАННЯ НА {periods} МІСЯЦІВ")

        n = len(self.monthly_data)
        x_future = np.arange(n, n + periods)

        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        fig.suptitle('МНК МОДЕЛЮВАННЯ ТА ПРОГНОЗУВАННЯ МНОЖИНИ ПРОЦЕСІВ',
                     fontsize=16, fontweight='bold')
        axes = axes.flatten()

        forecast_results = {}

        for i, (name, model) in enumerate(self.models.items()):
            if i >= len(axes):
                break

            forecast = model['poly'](x_future)
            forecast_results[name] = forecast

            print(f"\n{name}:")
            print(f"Прогноз на наступні {periods} місяців:")
            for j, val in enumerate(forecast, 1):
                print(f"Місяць +{j}: {val:.2f}")

            x_data = np.arange(len(model['data']))
            axes[i].plot(x_data, model['data'], 'b-', alpha=0.6, linewidth=1.5, label='Реальні дані')
            axes[i].plot(x_data, model['fitted'], 'r-', linewidth=2, label=f'МНК модель (R²={model["r2"]:.3f})')
            axes[i].plot(x_future, forecast, 'g--', linewidth=2, label='Прогноз')

            axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Місяць')
            axes[i].set_ylabel('Значення')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            textstr = f'RMSE: {model["rmse"]:.2f}\nMAPE: {model["mape"]:.2f}%'
            axes[i].text(0.02, 0.98, textstr, transform=axes[i].transAxes,
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        axes[5].remove()
        axes[5] = fig.add_subplot(3, 2, 6)

        for name, model in self.models.items():
            x_all = np.arange(len(model['data']) + periods)
            y_all = model['poly'](x_all)
            y_norm = (y_all - y_all.min()) / (y_all.max() - y_all.min())
            axes[5].plot(x_all, y_norm, linewidth=2, label=name)

        axes[5].set_title('Порівняння нормалізованих трендів', fontsize=12, fontweight='bold')
        axes[5].set_xlabel('Місяць')
        axes[5].set_ylabel('Нормалізоване значення (0-1)')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        axes[5].axvline(x=n, color='r', linestyle='--', linewidth=2, label='Початок прогнозу')

        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, '4_forecast_all_processes.png'), dpi=300, bbox_inches='tight')
        print(f"\nГрафіки збережено: {GRAPHS_DIR}/4_forecast_all_processes.png")
        plt.close()

        self.forecast_results = forecast_results
        self.forecast_by_region(periods)
        self.visualize_regional_forecast()
        self.forecast_by_manager(periods)
        self.visualize_manager_forecast()

    def visualize_regional_forecast(self):
        """Візуалізація прогнозів по регіонам"""

        regions = self.df_merged['Region'].unique()

        fig1, axes1 = plt.subplots(2, 4, figsize=(20, 10))
        fig1.suptitle('ПРОГНОЗУВАННЯ ПРОДАЖІВ ПО РЕГІОНАМ',
                      fontsize=16, fontweight='bold')
        axes1 = axes1.flatten()

        for idx, region in enumerate(regions):
            region_data = self.df_merged[self.df_merged['Region'] == region]
            monthly_region = region_data.groupby('YearMonth').agg({'Sales': 'sum'})

            n = len(monthly_region)
            x = np.arange(n)
            x_future = np.arange(n, n + 6)

            sales_data = monthly_region['Sales'].values
            sales_coeffs = np.polyfit(x, sales_data, 2)
            sales_poly = np.poly1d(sales_coeffs)
            sales_fitted = sales_poly(x)
            sales_forecast = sales_poly(x_future)

            axes1[idx].plot(x, sales_data, 'b-', alpha=0.6, linewidth=1.5, label='Реальні дані')
            axes1[idx].plot(x, sales_fitted, 'r-', linewidth=2, label='МНК модель')
            axes1[idx].plot(x_future, sales_forecast, 'g--', linewidth=2, label='Прогноз')

            axes1[idx].axvline(x=n - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axes1[idx].set_title(f'{region}', fontsize=11, fontweight='bold')
            axes1[idx].set_xlabel('Місяць')
            axes1[idx].set_ylabel('Sales')
            axes1[idx].legend(fontsize=8)
            axes1[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, '5_1_regional_sales_forecast.png'),
                    dpi=300, bbox_inches='tight')
        print(f"\nГрафік збережено: {GRAPHS_DIR}/5_1_regional_sales_forecast.png")
        plt.close()

        fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
        fig2.suptitle('ПРОГНОЗУВАННЯ ПРИБУТКУ ПО РЕГІОНАМ',
                      fontsize=16, fontweight='bold')
        axes2 = axes2.flatten()

        for idx, region in enumerate(regions):
            region_data = self.df_merged[self.df_merged['Region'] == region]
            monthly_region = region_data.groupby('YearMonth').agg({'Profit': 'sum'})

            n = len(monthly_region)
            x = np.arange(n)
            x_future = np.arange(n, n + 6)

            profit_data = monthly_region['Profit'].values
            profit_coeffs = np.polyfit(x, profit_data, 2)
            profit_poly = np.poly1d(profit_coeffs)
            profit_fitted = profit_poly(x)
            profit_forecast = profit_poly(x_future)

            axes2[idx].plot(x, profit_data, 'b-', alpha=0.6, linewidth=1.5, label='Реальні дані')
            axes2[idx].plot(x, profit_fitted, 'r-', linewidth=2, label='МНК модель')
            axes2[idx].plot(x_future, profit_forecast, 'g--', linewidth=2, label='Прогноз')

            axes2[idx].axvline(x=n - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axes2[idx].set_title(f'{region}', fontsize=11, fontweight='bold')
            axes2[idx].set_xlabel('Місяць')
            axes2[idx].set_ylabel('Profit')
            axes2[idx].legend(fontsize=8)
            axes2[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, '5_2_regional_profit_forecast.png'),
                    dpi=300, bbox_inches='tight')
        print(f"\nГрафік збережено: {GRAPHS_DIR}/5_2_regional_profit_forecast.png")
        plt.close()

    def visualize_manager_forecast(self):
        """Візуалізація прогнозів у менеджерів"""

        managers = self.users['Manager'].unique()
        valid_managers = []
        manager_sales_dict = {}
        manager_profit_dict = {}

        for manager in managers:
            manager_regions = self.users[self.users['Manager'] == manager]['Region'].tolist()

            region_manager_counts = {}
            for region in manager_regions:
                count = len(self.users[self.users['Region'] == region])
                region_manager_counts[region] = count

            manager_data = self.df_merged[self.df_merged['Region'].isin(manager_regions)].copy()

            manager_data['Manager_Count'] = manager_data['Region'].map(region_manager_counts)
            manager_data['Sales'] = manager_data['Sales'] / manager_data['Manager_Count']
            manager_data['Profit'] = manager_data['Profit'] / manager_data['Manager_Count']

            monthly_manager = manager_data.groupby('YearMonth').agg({
                'Sales': 'sum',
                'Profit': 'sum'
            })

            if len(monthly_manager) >= 3:
                valid_managers.append(manager)
                manager_sales_dict[manager] = monthly_manager['Sales']
                manager_profit_dict[manager] = monthly_manager['Profit']

        if not valid_managers:
            print("Немає менеджерів з достатньою кількістю даних для візуалізації")
            return

        n_managers = len(valid_managers)
        n_cols = 2
        n_rows = (n_managers + 1) // 2

        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        fig1.suptitle('ПРОГНОЗУВАННЯ ПРОДАЖІВ У МЕНЕДЖЕРІВ',
                      fontsize=16, fontweight='bold')

        if n_managers == 1:
            axes1 = np.array([axes1])
        axes1 = axes1.flatten()

        for idx, manager in enumerate(valid_managers):
            sales_data = manager_sales_dict[manager].values
            n = len(sales_data)
            x = np.arange(n)
            x_future = np.arange(n, n + 6)

            sales_coeffs = np.polyfit(x, sales_data, 2)
            sales_poly = np.poly1d(sales_coeffs)
            sales_fitted = sales_poly(x)
            sales_forecast = sales_poly(x_future)

            axes1[idx].plot(x, sales_data, 'b-', alpha=0.6, linewidth=1.5, label='Реальні дані')
            axes1[idx].plot(x, sales_fitted, 'r-', linewidth=2, label='МНК модель')
            axes1[idx].plot(x_future, sales_forecast, 'g--', linewidth=2, label='Прогноз')

            axes1[idx].axvline(x=n - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axes1[idx].set_title(f'Менеджер: {manager}', fontsize=12, fontweight='bold')
            axes1[idx].set_xlabel('Місяць')
            axes1[idx].set_ylabel('Sales')
            axes1[idx].legend(fontsize=9)
            axes1[idx].grid(True, alpha=0.3)

        for idx in range(n_managers, len(axes1)):
            axes1[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, '6_1_manager_sales_forecast.png'),
                    dpi=300, bbox_inches='tight')
        print(f"\nГрафік збережено: {GRAPHS_DIR}/6_1_manager_sales_forecast.png")
        plt.close()

        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        fig2.suptitle('ПРОГНОЗУВАННЯ ПРИБУТКУ У МЕНЕДЖЕРІВ',
                      fontsize=16, fontweight='bold')

        if n_managers == 1:
            axes2 = np.array([axes2])
        axes2 = axes2.flatten()

        for idx, manager in enumerate(valid_managers):
            profit_data = manager_profit_dict[manager].values
            n = len(profit_data)
            x = np.arange(n)
            x_future = np.arange(n, n + 6)

            profit_coeffs = np.polyfit(x, profit_data, 2)
            profit_poly = np.poly1d(profit_coeffs)
            profit_fitted = profit_poly(x)
            profit_forecast = profit_poly(x_future)

            axes2[idx].plot(x, profit_data, 'b-', alpha=0.6, linewidth=1.5, label='Реальні дані')
            axes2[idx].plot(x, profit_fitted, 'r-', linewidth=2, label='МНК модель')
            axes2[idx].plot(x_future, profit_forecast, 'g--', linewidth=2, label='Прогноз')

            axes2[idx].axvline(x=n - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axes2[idx].set_title(f'Менеджер: {manager}', fontsize=12, fontweight='bold')
            axes2[idx].set_xlabel('Місяць')
            axes2[idx].set_ylabel('Profit')
            axes2[idx].legend(fontsize=9)
            axes2[idx].grid(True, alpha=0.3)

        for idx in range(n_managers, len(axes2)):
            axes2[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, '6_2_manager_profit_forecast.png'),
                    dpi=300, bbox_inches='tight')
        print(f"\nГрафік збережено: {GRAPHS_DIR}/6_2_manager_profit_forecast.png")
        plt.close()

    def forecast_by_region(self, periods=6):
        """Прогнозування по регіонам"""
        print("\n" + "-" * 80)
        print("ПРОГНОЗУВАННЯ ПО РЕГІОНАМ")

        regions = self.df_merged['Region'].unique()

        for region in regions:
            region_data = self.df_merged[self.df_merged['Region'] == region]

            monthly_region = region_data.groupby('YearMonth').agg({
                'Sales': 'sum',
                'Profit': 'sum'
            })

            print(f"\n{'-' * 40}")
            print(f"РЕГІОН: {region}")
            print(f"{'-' * 40}")

            n = len(monthly_region)
            x = np.arange(n)
            x_future = np.arange(n, n + periods)

            sales_data = monthly_region['Sales'].values
            sales_coeffs = np.polyfit(x, sales_data, 2)
            sales_poly = np.poly1d(sales_coeffs)
            sales_forecast = sales_poly(x_future)

            print("\nПрогноз ПРОДАЖІВ на 6 місяців:")
            for j, val in enumerate(sales_forecast, 1):
                print(f"Місяць +{j}: {val:.2f}")

            profit_data = monthly_region['Profit'].values
            profit_coeffs = np.polyfit(x, profit_data, 2)
            profit_poly = np.poly1d(profit_coeffs)
            profit_forecast = profit_poly(x_future)

            print("\nПрогноз ПРИБУТКУ на 6 місяців:")
            for j, val in enumerate(profit_forecast, 1):
                print(f"Місяць +{j}: {val:.2f}")

    def forecast_by_manager(self, periods=6):
        """Прогнозування у менеджерів з розподілом для спільних регіонів"""
        print("\n" + "-" * 80)
        print("ПРОГНОЗУВАННЯ У МЕНЕДЖЕРІВ")

        managers = self.users['Manager'].unique()

        for manager in managers:
            manager_regions = self.users[self.users['Manager'] == manager]['Region'].tolist()

            region_manager_counts = {}
            for region in manager_regions:
                count = len(self.users[self.users['Region'] == region])
                region_manager_counts[region] = count

            manager_data = self.df_merged[self.df_merged['Region'].isin(manager_regions)].copy()

            manager_data['Manager_Count'] = manager_data['Region'].map(region_manager_counts)
            manager_data['Sales'] = manager_data['Sales'] / manager_data['Manager_Count']
            manager_data['Profit'] = manager_data['Profit'] / manager_data['Manager_Count']

            monthly_manager = manager_data.groupby('YearMonth').agg({
                'Sales': 'sum',
                'Profit': 'sum'
            })

            if len(monthly_manager) < 3:
                print(f"\n{'-' * 40}")
                print(f"МЕНЕДЖЕР: {manager}")
                print(f"{'-' * 40}")
                print("Недостатньо даних для прогнозування (потрібно мінімум 3 місяці)")
                continue

            print(f"\n{'-' * 40}")
            print(f"МЕНЕДЖЕР: {manager}")
            print(f"{'-' * 40}")

            n = len(monthly_manager)
            x = np.arange(n)
            x_future = np.arange(n, n + periods)

            sales_data = monthly_manager['Sales'].values
            sales_coeffs = np.polyfit(x, sales_data, 2)
            sales_poly = np.poly1d(sales_coeffs)
            sales_forecast = sales_poly(x_future)

            print("\nПрогноз ПРОДАЖІВ на 6 місяців:")
            for j, val in enumerate(sales_forecast, 1):
                print(f"Місяць +{j}: {val:.2f}")

            profit_data = monthly_manager['Profit'].values
            profit_coeffs = np.polyfit(x, profit_data, 2)
            profit_poly = np.poly1d(profit_coeffs)
            profit_forecast = profit_poly(x_future)

            print("\nПрогноз ПРИБУТКУ на 6 місяців:")
            for j, val in enumerate(profit_forecast, 1):
                print(f"Місяць +{j}: {val:.2f}")

def main():
    """Головна функція"""
    print("\n" + "-" * 80)
    print("                     ЛАБОРАТОРНА РОБОТА №8")
    print("ПРОЕКТНИЙ ПРАКТИКУМ З ПРОГНОЗУВАННЯ ЕКОНОМІЧНИХ ПОКАЗНИКІВ")
    print("-" * 80 + "\n")


    filename = 'Data_Set_11.xlsx'
    analyzer = MultiProcessAnalyzer(filename)

    try:
        if not analyzer.load_data():
            return

        analyzer.create_time_series()

        analyzer.test_stationarity_and_seasonality()

        analyzer.visualize_stationarity_tests()
        analyzer.visualize_seasonality()

        analyzer.analyze_properties()

        analyzer.visualize_all_processes()

        analyzer.statistical_analysis()

        analyzer.build_forecast_models()

        analyzer.forecast_and_visualize(periods=6)

        print("\n" + "-" * 80)
        print("ПРОГРАМА УСПІШНО ЗАВЕРШЕНА")


    except Exception as e:
        print(f"\nПомилка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()