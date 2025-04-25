import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# 步骤1：读取数据并预处理
# --------------------------------------------------
df = pd.read_csv('/Users/sean/2024-2025/CS200/leisureARIMA.csv', parse_dates=['year'], index_col='year')

# Check data format
print("First 5 rows:\n", df.head())
df = df.asfreq('AS')  # 'AS'表示年度起始频率
print("\nTime frequency:", df.index.freq)

# 步骤2：平稳性检测（ADF检验）
# --------------------------------------------------
def adf_test(series):
    result = adfuller(series)
    print('\nADF Test Results:')
    print(f'ADF Statistic: {result[0]:.3f}')
    print(f'p-value: {result[1]:.3f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.3f}')
    if result[1] < 0.05:
        print("Conclusion: Series is stationary (no differencing needed)")
    else:
        print("Conclusion: Series is non-stationary (differencing required)")

adf_test(df['leisure_time'])

# 步骤3：可视化原始序列
# --------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df['leisure_time'], label='Leisure Time', marker='o')
plt.title('Annual Trend of Leisure Time')
plt.xlabel('Year')
plt.ylabel('Leisure Time (units)')
plt.grid(True)
plt.legend()
plt.show()

# 步骤4：拟合ARIMA(1,1,1)模型
# --------------------------------------------------
model = ARIMA(df['leisure_time'], order=(1,1,1))
results = model.fit()
print("\nModel Summary:")
print(results.summary())

# 步骤5：模型诊断（残差分析）
# --------------------------------------------------
# Residuals plot
residuals = results.resid
plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Residuals', color='orange')
plt.axhline(y=0, linestyle='--', color='gray')
plt.title('Residual Series')
plt.xlabel('Year')
plt.legend()
plt.show()

# Residual distribution
plt.figure(figsize=(8, 4))
residuals.plot(kind='kde', title='Residual Distribution')
plt.show()

# Ljung-Box test
lb_test = acorr_ljungbox(residuals, lags=10)
print("\nLjung-Box Test p-values:")
print(lb_test['lb_pvalue'])
if all(lb_test['lb_pvalue'] > 0.05):
    print("Conclusion: Residuals are white noise (model valid)")
else:
    print("Conclusion: Residuals have autocorrelation (model needs improvement)")

# 步骤6：预测下一个时段
# --------------------------------------------------
forecast = results.get_forecast(steps=1)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

next_year = df.index[-1] + pd.DateOffset(years=1)
print(f"\nForecast for {next_year.year}:")
print(f"Predicted value: {forecast_mean.values[0]:.2f}")
print(f"95% Confidence Interval: [{forecast_conf_int.iloc[0,0]:.2f}, {forecast_conf_int.iloc[0,1]:.2f}]")

# 步骤7：可视化预测结果
# --------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['leisure_time'], label='Historical Data', marker='o')
plt.plot(next_year, forecast_mean, 'ro', markersize=8, label='Forecast')
plt.fill_betweenx(
    y=[forecast_conf_int.iloc[0,0], forecast_conf_int.iloc[0,1]],
    x=next_year,
    color='pink',
    alpha=0.3,
    label='95% Confidence Interval'
)
plt.title('ARIMA(1,1,1) Forecast')
plt.xlabel('Year')
plt.ylabel('Leisure Time (units)')
plt.grid(True)
plt.legend()
plt.show()