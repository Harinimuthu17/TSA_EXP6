# Ex.No: 6               HOLT WINTERS METHOD
### Date: 

## Name: M.HARINI
## Reg No: 212222240035
## Date: 

# Ex.No: 6  HOLT WINTERS METHOD

## AIM:
To implement Holt-winters model on tesla stock prediction and make future predictions.

## ALGORITHM:

1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
    
## PROGRAM:
```
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

file_path = 'Microsoft_Stock.csv'
data = pd.read_csv(file_path, index_col='Date', parse_dates=True)

data_open = data[['Open']]
data_monthly = data_open.resample('MS').mean()

scaler = MinMaxScaler()
data_scaled = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(), index=data_monthly.index)

train_data = data_scaled[:int(len(data_scaled) * 0.8)]
test_data = data_scaled[int(len(data_scaled) * 0.8):]

fitted_model_add = ExponentialSmoothing(
    train_data, trend='add', seasonal='add', seasonal_periods=12
).fit()

test_predictions_add = fitted_model_add.forecast(len(test_data))

print("MAE (Additive):", mean_absolute_error(test_data, test_predictions_add))
print("RMSE (Additive):", mean_squared_error(test_data, test_predictions_add, squared=False))

# First Plot: Additive Holt-Winters Predictions
plt.figure(figsize=(6, 4))
plt.plot(train_data, label='Train Data', color='black')
plt.plot(test_data, label='Test Data', color='green')
plt.plot(test_predictions_add, label='Additive Predictions', color='red')
plt.title('Train, Test, and Additive Holt-Winters Predictions')
plt.legend(loc='best')
plt.tight_layout()  # Adjust layout to add spacing
plt.show()  # Show the first plot separately

# Shift and fit the multiplicative model
shift_value = abs(min(data_scaled)) + 1
data_shifted = data_scaled + shift_value

final_model_mul = ExponentialSmoothing(data_shifted, trend='mul', seasonal='mul', seasonal_periods=12).fit()

forecast_predictions_mul = final_model_mul.forecast(steps=12)

forecast_predictions = forecast_predictions_mul - shift_value

# Second Plot: Multiplicative Holt-Winters Forecast
plt.figure(figsize=(6, 4))
data_scaled.plot(label='Current Data', legend=True)
forecast_predictions.plot(label='Forecasted Data', legend=True)
plt.title('Forecast using Multiplicative Holt-Winters Model (Shifted Data)')
plt.xlabel('Date')
plt.ylabel('Price (Scaled)')
plt.legend(loc='best')
plt.tight_layout()  # Adjust layout to add spacing
plt.show() 

```

## OUTPUT:

EVALUTION






TEST_PREDICTION



![Screenshot 2024-10-21 105217](https://github.com/user-attachments/assets/1b318ede-5452-49ba-a210-4233b223feea)

FINAL_PREDICTION

![Screenshot 2024-10-21 105237](https://github.com/user-attachments/assets/d04a2115-c98c-4e17-8bdd-354dcd17de74)



## RESULT:

Therefore a python program has been executed successfully based on the Holt Winters Method model.



FINAL_PREDICTION

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
