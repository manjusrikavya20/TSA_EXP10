# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 25/10/2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
# PROGRAM: BMW Data Forecasting using SARIMA Model

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load and clean the dataset
df = pd.read_csv('bmw dataset.csv')

# Automatically detect numeric column to forecast
numeric_cols = df.select_dtypes(include=np.number).columns
if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found in the dataset.")
value_col = numeric_cols[0]

# If dataset has a Date column, use it for sorting
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    df = df.reset_index(drop=True)
    df = df[['Date', value_col]].copy()
    df.columns = ['Date', 'Value']
else:
    # If no Date column, just create a sequential index
    df = df[[value_col]].copy()
    df['Date'] = np.arange(len(df))
    df.columns = ['Value', 'Date']

print("Cleaned dataset:")
print(df.head())

# Plot BMW trend
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Value'], marker='o', color='blue')
plt.title('BMW Data Over Time')
plt.xlabel('Date / Time Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Check Stationarity (ADF Test)
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    if result[1] <= 0.05:
        print("The data is stationary.")
    else:
        print("The data is NOT stationary. Differencing may be required.")

check_stationarity(df['Value'])

# Plot ACF and PACF
plt.figure(figsize=(10, 4))
plot_acf(df['Value'], lags=20)
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(df['Value'], lags=20)
plt.show()

# Split data into training and testing sets (80%-20%)
train_size = int(len(df) * 0.8)
train, test = df['Value'][:train_size], df['Value'][train_size:]

# Fit SARIMA model
# Adjust the order (p,d,q) and seasonal_order (P,D,Q,s) if needed
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
result = model.fit(disp=False)

# Make predictions
pred = result.predict(start=len(train), end=len(train)+len(test)-1)

# Evaluate model performance
rmse = np.sqrt(mean_squared_error(test, pred))
print(f"\nRoot Mean Square Error (RMSE): {rmse:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, pred, label='Predicted', color='red')
plt.title('SARIMA Model - BMW Data Forecast')
plt.xlabel('Index (Time)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

```

### OUTPUT:
ORIGINAL DATA:

<img width="1108" height="569" alt="image" src="https://github.com/user-attachments/assets/bdc1f636-f9a6-41cf-862c-0c3d22f5c094" />

AUTOCORRELATION:

<img width="699" height="538" alt="image" src="https://github.com/user-attachments/assets/bcb80920-229f-4869-9dd7-5214f62c7e81" />

PARTIAL AUTOCORRELATION:

<img width="724" height="547" alt="image" src="https://github.com/user-attachments/assets/4ffaec55-ae54-429a-88bd-f77d75a9d7e6" />

SARIMA MODEL:

<img width="1085" height="582" alt="image" src="https://github.com/user-attachments/assets/26f7a086-a75c-434f-a115-9712e2e8c523" />

RMSE VALUE:

<img width="357" height="35" alt="image" src="https://github.com/user-attachments/assets/bdd86c3a-a88d-44ac-965e-489ead9a43f4" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
