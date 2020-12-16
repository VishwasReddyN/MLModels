import matplotlib.pyplot as pt
import numpy as np
import pandas as pd

data = pd.read_excel("./TimeSeriesData.xlsx", header= None)

total = list(data.iloc[:72,1].values)

train = list(data.iloc[:60,1].values)

test = list(data.iloc[60:72,1].values)

train_time =list(np.arange(60))

total_time = list(np.arange(72))

test_time = list(np.arange(60,72))


from statsmodels.tsa.ar_model import AR

# fit model
model = AR(train)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(train), len(train)+11)
print(len(train))
# print(test-yhat)

pt.plot(train_time,train, color = 'blue' )

pt.plot(test_time, test, color = 'red')

pt.plot(test_time,yhat, color = 'black')

pt.show()

forecast_errors = [test[i]-yhat[i] for i in range(len(test))]
print('Forecast Errors: %s' % forecast_errors) # forecast_error = expected_value - predicted_value

bias = sum(forecast_errors) * 1.0/len(test)
print('Bias: %f' % bias) # mean_forecast_error = mean(forecast_error)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(test, yhat)
print('MAE: %f' % mae) # mean_absolute_error = mean( abs(forecast_error) )

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(test, yhat)
print('MSE: %f' % mse) # mean_squared_error = mean(forecast_error^2)

from sklearn.metrics import mean_squared_error
from math import sqrt

mse = mean_squared_error(test, yhat)
rmse = sqrt(mse)
print('RMSE: %f' % rmse) # rmse = sqrt(mean_squared_error)