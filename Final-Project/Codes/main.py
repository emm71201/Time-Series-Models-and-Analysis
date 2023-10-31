from functools import lru_cache
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tools
tools.pyplot_setup()

# My setup of matplotlib
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc('text', usetex=False)
plt.rc('font', family='times')
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
fontsize = 18

# get data
PATH = "../dataset/jena_climate_2009_2016.csv"

# read data and resample every 60 minutes
@lru_cache
def get_data(PATH, resample_rule):
    raw_data = pd.read_csv(PATH)
    raw_data = raw_data.set_index(pd.DatetimeIndex(raw_data['Date Time']))
    raw_data.drop(columns=['Date Time'], inplace=True)
    raw_data = raw_data.resample(resample_rule).mean()

    return raw_data
# data = get_data(PATH, "60T")
# data.to_pickle("raw_data_resampled.pkl")
# data
data = pd.read_pickle("raw_data_resampled.pkl")
data.dropna(inplace=True)
print(f"The final shape of the data is {data.shape}")

# change columns
new_columns = ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 'sh',
'H2OC', 'rho', 'wv', 'max. wv', 'wd']
mapper = {data.columns[i]:new_columns[i] for i in range(len(new_columns))}
data = data.rename(columns=mapper)

# plot dependent variable: T
tools.plot_series(data['T'], "Temperature (C)", "Time", "Temperature")

# Plot ACF
number_lags=60
size = data.shape[0]
acf_list = tools.ACF_Calc(data['T'], number_lags)
fig = tools.plot_ACF(acf_list, number_lags, size)
plt.show()

# correlation heat map
heatmap = tools.plot_correlation_heatmap(data, "Correlation Heatmap")

# train test split
split=0.2
dependent_col = 'T'
X_train, X_test, y_train, y_test = tools.dataframe_train_test_split(data, dependent_col, split)
# print(X_test.shape[0]/X_train.shape[0])







