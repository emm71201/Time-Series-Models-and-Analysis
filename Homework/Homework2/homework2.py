# Author: Edison Murairi

# modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns


np.random.seed(6313)

# My setup of matplotlib
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc('text', usetex=False)
plt.rc('font', family='times')
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
fontsize = 17

# Problem 2
mean = 0
std = 1
size = 1000
X = np.random.normal(mean, std, size=size)
print(f"The sample mean is {X.mean():.3f}")
print(f"The sample standard deviation is {X.std(ddof=1):.3f}")
data = pd.DataFrame({'X':X}, index=pd.date_range("2000-01-01", "2000-12-31", periods=size))
data.plot(kind="hist")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Problem 3
def ACF(data, tau):
    """ compute the ACF at some lag = tau """
    # handle negative lag
    if tau < 0:
        return ACF(data, -tau)
    mean = data.mean()
    AA = sum((data[t] - mean) * (data[t-tau] - mean) for t in range(tau,len(data)))
    BB = sum((data[t] - mean)**2 for t in range(len(data)))
    return AA/BB
    
def ACF_Calc(data, number_lags):

    try:
        data = np.array(data)
    except:
        print("The data must be a one - D array")
        return
    
    mean = data.mean()
    BB = sum((data[t] - mean)**2 for t in range(len(data)))

    tmp = np.array([sum((data[t] - mean) * (data[t-tau] - mean) \
                        for t in range(tau,len(data))) for tau in range(number_lags+1)])/BB

    return np.concatenate((tmp[::-1][:-1], tmp))

def plot_ACF(acf_list, number_lags, size, ax=None):

    
    x = np.array([i for i in range(-number_lags, number_lags+1)])
    
    #print(len(x), len(acf_list))

    #insignificance band
    xfill = [i for i in range(-number_lags, number_lags+1)]
    ydown = np.full((len(xfill), ), -1.96/np.sqrt(size) )
    yup = np.full((len(xfill), ), 1.96/np.sqrt(size) )

    if not ax is None:
        ax.fill_between(xfill, ydown, yup, alpha=0.5)
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel("ACF")
        ax.stem(x, acf_list, markerfmt="magenta", basefmt='C2')

        return ax

    fig = plt.figure()
    plt.fill_between(xfill, ydown, yup, alpha=0.5)
    plt.xlabel(r"$\tau$")
    plt.ylabel("ACF")
    plt.stem(x, acf_list, markerfmt="magenta", basefmt='C2')

    return fig

#part a
nlags = 4
data = np.array([3,9,27,81,243])
acf_list = ACF_Calc(data, nlags)
myplot = plot_ACF(acf_list, nlags, data.shape[0])
plt.show()

# part b
nlags = 20
acf_list = ACF_Calc(X, nlags)
myplot = plot_ACF(acf_list, nlags, 1000)
plt.show()

# Problem 4
from pandas_datareader import data
import yfinance as yf
yf.pdr_override()

# mpl.rcParams.update(mpl.rcParamsDefault)
# plt.rc('text', usetex=False)
# plt.rc('font', family='times')
# mpl.rcParams['mathtext.fontset'] = 'cm'
# plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
# fontsize = 16

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9,9))
stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
stocks = np.array(stocks).reshape((3,2))
stocks_data={}
for i in range(3):
    for j in range(2):
        ax = axes[i,j]
        stock = stocks[i,j]

        df = data.get_data_yahoo(stocks[i,j], start='2000-01-01', end='2023-01-31')
        stocks_data[stock] = df
        ax.plot(df['Adj Close'], color='blue', label=stock)
        ax.grid()
        ax.set_xlabel('Time')
        ax.set_ylabel('Close Value')
        ax.legend(loc=2)

plt.tight_layout()        
plt.show()

# Part (b) -- ACF
nlags = 50
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9,9))
for i in range(3):
    for j in range(2):
        ax = axes[i,j]
        stock = stocks[i,j]
        close_values = stocks_data[stock]['Adj Close']
        size = len(close_values)
        acf_list = ACF_Calc(close_values, nlags)
        ax = plot_ACF(acf_list, nlags, size, ax)
        ax.set_title(stock)

plt.tight_layout() 
plt.show()
                         

