#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import numpy as np

#%%
# My setup of matplotlib
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc('text', usetex=False)
plt.rc('font', family='times')
mpl.rcParams['mathtext.fontset'] = 'cm'

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
fontsize = 16
#%%[markdown]
# Problem 1
#tute1_url = "https://github.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/blob/main/tute1.csv"
#tute1 = get_csv_from_web(tute1_url)
tute1 = pd.read_csv("tute1.csv")
tute1['Date'] = pd.to_datetime(tute1['Date'])
tute1.set_index('Date', inplace=True)
print("data obtained")
#%%
plt.figure(figsize=(9,7))
tute1['Sales'].plot()
tute1['AdBudget'].plot()
tute1['GDP'].plot()
plt.ylabel('USD ($)')
plt.grid()
plt.title("Time series of Sales, Ad Budget and GDP for Tute 1")
plt.legend(loc=2, framealpha=0.4)
plt.savefig("tute1_series.pdf")
plt.show()
#%%
# Problem 2
print(f"The Sales mean is : {tute1['Sales'].mean():.2f} and the variance is : {tute1['Sales'].var():.2f} with standard deviation : {tute1['Sales'].std():.2f} median: {tute1['Sales'].median():.2f}.")
print(f"The AdBudget mean is : {tute1['AdBudget'].mean():.2f} and the variance is : {tute1['AdBudget'].var():.2f} with standard deviation : {tute1['AdBudget'].std():.2f} median: {tute1['AdBudget'].median():.2f}.")
print(f"The GDP mean is : {tute1['GDP'].mean():.2f} and the variance is : {tute1['GDP'].var():.2f} with standard deviation : {tute1['GDP'].std():.2f} median: {tute1['GDP'].median():.2f}.")
# %%
# Problem 3: Use rolling mean and variance to test whether a variable is stationary or not
def Cal_rolling_mean_var(data):

    """ take list and compute the rolling mean and variance
    Params: 
        Data: a list (numpy array, pandas series, ... a 1-D iterable) of float
    Return 
        rolling mean (list of floats), rolling variance (list of floats)
      """
    
    rolling_mean = np.array([])
    rolling_variance = np.array([])
    rolling_data = np.array([])

    #iterate over the data
    for item in data:

        rolling_data = np.append(rolling_data, item) # insert a new item from data
        rolling_mean = np.append(rolling_mean, rolling_data.mean()) # compute the new mean
        rolling_variance = np.append(rolling_variance, rolling_data.var()) # compute the new variance
    
    return rolling_mean, rolling_variance

#%%
sales_roll_mean, sales_roll_var = Cal_rolling_mean_var(tute1['Sales'])
adbudget_roll_mean, addbudget_roll_var = Cal_rolling_mean_var(tute1['AdBudget'])
gdp_roll_mean, gdp_roll_var = Cal_rolling_mean_var(tute1['GDP'])
# %%[markdown]
# Rolling Mean and Variance for Sales
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,10))

ax1.plot(sales_roll_mean, color='midnightblue')
ax2.plot(sales_roll_var, color='royalblue')

ax1.set_title("Rolling Mean Sales")
ax2.set_title("Rolling Variance Sales")
ax1.set_xlabel("Samples")
ax2.set_xlabel("Samples")
ax1.set_ylabel("Mean")
ax2.set_ylabel("Variance")
plt.savefig("roll_mean_variance_sales.pdf")
plt.show()
# %%
# Rolling Mean and Variance for AdBudget
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,10))

ax1.plot(adbudget_roll_mean, color='midnightblue')
ax2.plot(addbudget_roll_var, color='royalblue')

ax1.set_title("Rolling Mean Ad Budget")
ax2.set_title("Rolling Variance for Ad Budget")
ax1.set_xlabel("Samples")
ax2.set_xlabel("Samples")
ax1.set_ylabel("Mean")
ax2.set_ylabel("Variance")
plt.savefig("roll_mean_variance_adBudget.pdf")
plt.show()
# %%
# Rolling Mean and Variance for GDP
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,10))

ax1.plot(gdp_roll_mean, color='midnightblue')
ax2.plot(gdp_roll_var, color='royalblue')

ax1.set_title("Rolling Mean GDP")
ax2.set_title("Rolling Variance for GDP")
ax1.set_xlabel("Samples")
ax2.set_xlabel("Samples")
ax1.set_ylabel("Mean")
ax2.set_ylabel("Variance")
plt.savefig("roll_mean_variance_GDP.pdf")
plt.show()

#%%[markdown]
# Problem 4 answered in the report
# %%[markdown]
# Problem 5
def ADF_Cal(x):
    result = adfuller(x)
    print("\tADF Statistic: %f" %result[0]) 
    print('\tp-value: %f' % result[1]) 
    print('\tCritical Values:')
    for key, value in result[4].items():
        print('\t\t%s: %.3f' % (key, value))

print('ADF Test for Sales:')
ADF_Cal(tute1['Sales'])
print()
print('ADF Test for AdBudget:')
ADF_Cal(tute1['AdBudget'])
print()
print('ADF Test for GDP:')
ADF_Cal(tute1['GDP'])
# %%
# Problem 6
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items(): 
        kpss_output['Critical Value (%s)'%key] = value
    print(kpss_output)

print('1. Sales:\n')
print(kpss_test(tute1['Sales']))
print()
print('2. Ad Budget:\n')
print(kpss_test(tute1['AdBudget']))
print()
print('3. GDP:\n')
print(kpss_test(tute1['GDP']))
######################################################### END OF TUTE 1 ###################
# %%[markdown]
## Air Passenger dataset
# %%
#air_passenger_url = "https://github.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/blob/main/AirPassengers.csv"
air_passenger = pd.read_csv("AirPassengers.csv")
air_passenger['Month'] = pd.to_datetime(air_passenger['Month'])
air_passenger.set_index('Month', inplace=True)
# %%
plt.figure(figsize=(9,5))
air_passenger['#Passengers'].plot()
plt.ylabel('Number of Passengers')
plt.xlabel("Time")
plt.grid()
plt.title("Air Passenger Time Series")
plt.savefig("airpassenger_series.pdf")
plt.show()
# %%
# Problem 2
print(f"The #Passengers mean is : {air_passenger['#Passengers'].mean():.2f} and the variance is : {air_passenger['#Passengers'].var():.2f} with standard deviation : {air_passenger['#Passengers'].std():.2f} median: {air_passenger['#Passengers'].median():.2f}.")
# %%
# Problem 3
airp_roll_mean, airp_roll_var = Cal_rolling_mean_var(air_passenger['#Passengers'])
# Rolling Mean and Variance for Number of Passengers
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,10))

ax1.plot(airp_roll_mean, color='midnightblue')
ax2.plot(airp_roll_var, color='royalblue')

ax1.set_title("Rolling Mean Number of Air Passengers")
ax2.set_title("Rolling Variance Number of Air Passengers")
ax1.set_xlabel("Samples")
ax2.set_xlabel("Samples")
ax1.set_ylabel("Mean")
ax2.set_ylabel("Variance")
plt.savefig("roll_mean_variance_airp.pdf")
plt.show()
# Problem 4 answered on the report
# %%
# Problem 5: ADF TEST
print("ADF TEST for the Number of Air Passengers:\n")
ADF_Cal(air_passenger['#Passengers'])
# %%
# Problem 6: KPSS TEST
kpss_test(air_passenger['#Passengers'])
# %%
def seasonal_differencing(data, p=1):
    """ perform the seasonal differencing of time series 
    Parameters:
        data: an iterable of floats
        p: the period. By default, we set p = 1. 
        We assume that the length of data is at least p + 1
        
        y'_t = y_t - y_{t - p}

        This function reduces to y'_{p+k} = y_{p+k} - y_{k} 
            where k ranges from 0 to len(data) - p - 1

    Return a numpy array of length len(data) - p
    """

    if len(data) < p + 1:
        print(" This function required the time series to have at least p + 1 data ")
        return

    return np.array([data[p + k] - data[k] for k in range(len(data) - p)])
# %%
# First order seasonal differencing
airp_first_diff = seasonal_differencing(air_passenger['#Passengers'], 1)
plt.figure(figsize=(9,5))
plt.plot(airp_first_diff)
plt.ylabel('First Order Differences of Number of Passengers')
plt.xlabel("Time")
plt.grid()
plt.title("Air Passenger Time Series First Order Differencing")
plt.savefig("airpassenger_first_order_diff.pdf")
plt.show()
# %%
airp_first_diff_mean, airp_first_diff_var = Cal_rolling_mean_var(airp_first_diff)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,10))

ax1.plot(airp_first_diff_mean, color='midnightblue')
ax2.plot(airp_first_diff_var, color='royalblue')

ax1.set_title("Rolling Mean After First Order Differencing")
ax2.set_title("Rolling Variance After First Order Differencing")
ax1.set_xlabel("Samples")
ax2.set_xlabel("Samples")
ax1.set_ylabel("Mean")
ax2.set_ylabel("Variance")
plt.savefig("airp_first_diff_roll_mean_variance.pdf")
plt.show()

# %%
# Second order seasonal differencing
airp_second_diff = seasonal_differencing(air_passenger['#Passengers'], 2)
plt.figure(figsize=(9,5))
plt.plot(airp_second_diff)
plt.ylabel('First Order Differences of Number of Passengers')
plt.xlabel("Time")
plt.grid()
plt.title("Air Passenger Time Series First Order Differencing")
#plt.savefig("airpassenger_first_order_diff.pdf")
plt.show()
# %%
airp_second_diff_mean, airp_second_diff_var = Cal_rolling_mean_var(airp_second_diff)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,10))

ax1.plot(airp_second_diff_mean, color='midnightblue')
ax2.plot(airp_second_diff_var, color='royalblue')

ax1.set_title("Rolling Mean After Second Order Differencing")
ax2.set_title("Rolling Variance After First Order Differencing")
ax1.set_xlabel("Samples")
ax2.set_xlabel("Samples")
ax1.set_ylabel("Mean")
ax2.set_ylabel("Variance")
plt.savefig("airp_second_diff_roll_mean_variance.pdf")
plt.show()

# %%
# third order seasonal differencing
airp_third_diff = seasonal_differencing(air_passenger['#Passengers'], 3)
plt.figure(figsize=(9,5))
plt.plot(airp_third_diff)
plt.ylabel('First Order Differences of Number of Passengers')
plt.xlabel("Time")
plt.grid()
plt.title("Air Passenger Time Series First Order Differencing")
#plt.savefig("airpassenger_first_order_diff.pdf")
plt.show()
# %%
airp_third_diff_mean, airp_third_diff_var = Cal_rolling_mean_var(airp_third_diff)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,10))

ax1.plot(airp_third_diff_mean, color='midnightblue')
ax2.plot(airp_third_diff_var, color='royalblue')

ax1.set_title("Rolling Mean After Third Order Differencing")
ax2.set_title("Rolling Variance After Third Order Differencing")
ax1.set_xlabel("Samples")
ax2.set_xlabel("Samples")
ax1.set_ylabel("Mean")
ax2.set_ylabel("Variance")
plt.savefig("airp_third_diff_roll_mean_variance.pdf")
plt.show()
# %%
# Log transform and then do 1st order differencing
transformed = seasonal_differencing(np.log(air_passenger['#Passengers']))
plt.figure(figsize=(9,5))
plt.plot(transformed)
plt.ylabel('First Order Differences of Number of Passengers')
plt.xlabel("Time")
plt.grid()
plt.title("Air Passenger Time Series First Order Differencing")
#plt.savefig("airpassenger_first_order_diff.pdf")
plt.show()
# %%
transformed_roll_mean, transformed_roll_var = Cal_rolling_mean_var(transformed)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,10))

ax1.plot(transformed_roll_mean, color='midnightblue')
ax2.plot(transformed_roll_var, color='royalblue')

ax1.set_title("Rolling Mean After First Order Differencing the Log of Number of Passengers")
ax2.set_title("Rolling Variance After First Order Differencing the Log of Number of Passengers")
ax1.set_xlabel("Samples")
ax2.set_xlabel("Samples")
ax1.set_ylabel("Mean")
ax2.set_ylabel("Variance")
plt.savefig("airp_transformed_roll_mean_variance.pdf")
plt.show()
# %%
print("ADF Test for log transformation followed by 1st order differencing:\n")
ADF_Cal(transformed)
# %%
print("KPSS Test:\n")
kpss_test(transformed)
# %%
