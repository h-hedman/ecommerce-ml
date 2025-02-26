# Author: Hayden Hedman
# Project: Time series analysis of synthetic e-commerce data
# Date: 2025-02-26
# --------------------------------------------------------------------------------------------------------------
# DESCRIPTION: 
## This model predicts revenue conversion in e-commerce using synthetic data.
## It leverages features such as time-based trends, seasonal patterns, and random noise.
## The model applies both ARIMA and SARIMA techniques to forecast future sales and evaluates performance using multiple metrics.
#
# OBJECTIVE: 
## Develop and evaluate a time series forecasting model that accurately predicts sales trends,
## incorporating seasonality and trend effects to improve forecasting accuracy.
#
# Hypothesis:
## H0: Time-based trends, seasonality, and noise do not significantly influence sales forecasting accuracy.
## H1: Incorporating time trends, seasonality, and noise improves sales prediction accuracy.
# --------------------------------------------------------------------------------------------------------------
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# --------------------------------------------------------------------------------------------------------------
# 1. Simulate e-commerce sales data (trend + seasonality + noise)
# --------------------------------------------------------------------------------------------------------------
np.random.seed(42)
days = 365  # 1 year of daily data
time_index = np.arange(days)
seasonality = 10 * np.sin(2 * np.pi * time_index / 30)  # Monthly seasonality
trend = 100 + 0.5 * time_index  # Linear trend
noise = np.random.normal(scale=5, size=days)  # Random noise
sales_data = trend + seasonality + noise
# --------------------------------------------------------------------------------------------------------------
# Create DataFrame
sales_df = pd.DataFrame({'date': pd.date_range(start='2025-01-01', periods=days, freq='D'), 'sales': sales_data})
sales_df.set_index('date', inplace=True)

# Split data into train and test sets (last 30 days for testing)
train_data, test_data = sales_df.iloc[:-30], sales_df.iloc[-30:]
# --------------------------------------------------------------------------------------------------------------
# 2. ARIMA Model 
# --------------------------------------------------------------------------------------------------------------
# Auto-Regressive Integrated Moving Average (ARIMA)
arima_order = (5,1,2)  # (p, d, q) parameters chosen arbitrarily
arima_model = ARIMA(train_data, order=arima_order)
arima_fit = arima_model.fit()

# Forecast using ARIMA
arima_forecast = arima_fit.forecast(steps=30)

# Calculate Error Metrics for ARIMA
arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
arima_mae = mean_absolute_error(test_data, arima_forecast)
arima_r2 = r2_score(test_data, arima_forecast)
print(f'ARIMA RMSE: {arima_rmse:.2f}')
print(f'ARIMA MAE: {arima_mae:.2f}')
print(f'ARIMA R-squared: {arima_r2:.2f}')
# --------------------------------------------------------------------------------------------------------------
# 3. SARIMA Model
# --------------------------------------------------------------------------------------------------------------
# Seasonal Auto-Regressive Integrated Moving Average (SARIMA)
sarima_order = (5,1,2)  # Non-seasonal parameters (p, d, q)
seasonal_order = (1,1,1,30)  # Seasonal parameters (P, D, Q, S) with monthly seasonality
sarima_model = SARIMAX(train_data, order=sarima_order, seasonal_order=seasonal_order)
sarima_fit = sarima_model.fit()

# Forecast using SARIMA
sarima_forecast = sarima_fit.forecast(steps=30)

# Calculate Error Metrics for SARIMA
sarima_rmse = np.sqrt(mean_squared_error(test_data, sarima_forecast))
sarima_mae = mean_absolute_error(test_data, sarima_forecast)
sarima_r2 = r2_score(test_data, sarima_forecast)
print(f'SARIMA RMSE: {sarima_rmse:.2f}')
print(f'SARIMA MAE: {sarima_mae:.2f}')
print(f'SARIMA R-squared: {sarima_r2:.2f}')
# --------------------------------------------------------------------------------------------------------------
# 4. Plot Results
# --------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(sales_df.index, sales_df['sales'], label='Actual Sales', linestyle='dashed', alpha=0.7)
plt.plot(test_data.index, arima_forecast, label='ARIMA Forecast', color='red')
plt.plot(test_data.index, sarima_forecast, label='SARIMA Forecast', color='blue')
plt.axvline(sales_df.index[-30], color='black', linestyle='dotted', label='Train/Test Split')
plt.legend()
plt.title('E-commerce Sales Forecast using ARIMA and SARIMA')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
# --------------------------------------------------------------------------------------------------------------