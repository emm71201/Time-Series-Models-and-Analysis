import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Model:
    def __init__(self):
        pass
    def one_step_mse(self, train, model):
        #start = train.shape[0] - model.shape[0]
        start = 1
        return np.mean((train[start:] - model)**2)
    def forecast_mse(self, test, forecast):
        return np.mean((test - forecast)**2)

class Average_Method(Model):
    def __init__(self):
        super().__init__()
    def one_step_ahead(self, train):
        return [train[:t].mean() for t in range(1, train.shape[0])]
    def hstep_forecast(self, train, test):
        return np.full(test.shape, train.mean())


train = np.array([112,118,132,129,121,135,148,136,119])
test = np.array([104,118,115,126,141])
avg_method = Average_Method()
model = avg_method.one_step_ahead(train)
forecast = avg_method.hstep_forecast(train, test)

# ploting
plt.plot([i for i in range(train.shape[0])], train, '.-', label = 'Training Data')
plt.plot([t for t in range(train.shape[0], train.shape[0] + test.shape[0])], test, ".-", label='Test Data')
#plt.plot([i for i in range(1,train.shape[0])], model, '.-', label = 'Model Data')
plt.plot([t for t in range(train.shape[0], train.shape[0] + test.shape[0])], forecast, ".-", label='Forecast')
plt.legend()
plt.show()

print(f"\nAVG METHOD MSE = {avg_method.one_step_mse(train, model):.2f}")
print(f"\nAVG METHOD FORECAST MSE = {avg_method.forecast_mse(test, forecast):.2f}")