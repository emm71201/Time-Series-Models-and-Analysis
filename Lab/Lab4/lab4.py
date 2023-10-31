import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

# My setup of matplotlib
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc('text', usetex=False)
plt.rc('font', family='times')
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
fontsize = 18

#load data
url = "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/daily-min-temperatures.csv"
df = pd.read_csv(url)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(df['Date'], inplace=True)
df.drop("Date", axis=1, inplace=True)
print("Data loaded")

# print(df.head())

def get_moving_averages(values, m):

    left, right = 0, m-1
    moving_averages = np.array([])
    while right < len(values):

        moving_averages = np.append(moving_averages, np.mean(values[left:right + 1]))
        left += 1
        right += 1

    return moving_averages

def check_input_is_digit(m):
    while not m.isdigit():
        print("A window side must be an integer\n")
        m = input("Enter a valid window size: m = ")
    return int(m)
def calc_moving_average(values):

    m = input("Insert the window size: m = ")
    m = check_input_is_digit(m)
    while m == 1 or m == 2:
        print("A valid window size must be different from 1 and 2")
        m = input("Enter a valid window size: m = ")
    m = check_input_is_digit(m)
    if m % 2 == 1:
        return get_moving_averages(values, m)

    print("Your window size is even\n")
    print("You must enter a second MA window size\n")
    print("The second MA window size must be even\n")
    folding = input("Insert the second MA window size: m = ")
    folding = check_input_is_digit(folding)
    while folding % 2 == 0:
        print("The second MA size must be even\n")
        folding = input("Insert the second MA window size: m = ")
        folding = check_input_is_digit(folding)

    return get_moving_averages(get_moving_averages(values, m), folding)

def plot_moving_averages(ax, values, moving_averages, title=""):

    x1 = [i for i in range(values.shape[0])]
    diff_lengths = values.shape[0] - moving_averages.shape[0]
    delta = diff_lengths//2
    x2 = [i for i in range(delta, values.shape[0] - delta)]


    cutoff = 50
    values = values[:cutoff ]
    x1 = x1[:cutoff]
    if moving_averages.shape[0] > cutoff:
        moving_averages = moving_averages[:cutoff]
        x2 = x2[:cutoff]

    ax.plot(x1, values, label='Original')
    ax.plot(x2, moving_averages, label='MA')
    ax.set_title(title)
    ax.set_ylabel("Temp")
    ax.set_xlabel("Time")
    ax.legend()

    return ax

fig, axes = plt.subplots(2,2, figsize=(10,9))
fig.subplots_adjust(hspace=0.5)
ma_sizes = [3,5,7,9]
count = 0
for i in range(2):
    for j in range(2):
        m = ma_sizes[count]
        moving_averages = get_moving_averages(df['Temp'], m)
        ax = axes[i,j]
        ax = plot_moving_averages(ax, df['Temp'], moving_averages, f"{m}-MA")
        count += 1
plt.savefig("Odd-MA.pdf", bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2,2, figsize=(10,9))
fig.subplots_adjust(hspace=0.5)
ma_sizes = [(2,4),(2,6),(2,8),(2,10)]
count = 0
for i in range(2):
    for j in range(2):
        m2,m1 = ma_sizes[count]
        av1 = get_moving_averages(df['Temp'], m1)
        av2 = get_moving_averages(av1, m2)
        ax = axes[i,j]
        ax = plot_moving_averages(ax, df['Temp'], moving_averages, f"{m2}$\\times${m1}-MA")
        count += 1
plt.savefig("Even-MA.pdf", bbox_inches='tight')
plt.show()

####### Problem 4 ######
m = 3
moving_averages = get_moving_averages(df['Temp'], m)
def adf_test(values, label=""):

    adf = adfuller(values)
    print(f"ADF TEST FOR {label.capitalize()}:")
    print(f"\tADF Statistics: {adf[0]:.3f}")
    print(f"\tp-value: {adf[1]:.3f}")
    print("\tCritical Values:")
    for key, value in adf[4].items():
        print('\t\t%s: %.3f' % (key, value))

    return adf

adf_test(df['Temp'], "Original data")
adf_test(moving_averages, "3-MA")

################ Problem 5 ###############################################
temp = pd.Series(df['Temp'])
stl = STL(temp, period=365)
result = stl.fit()
fig = result.plot()
plt.xlabel("Time")
# plt.ylabel("Temp")
plt.savefig("stld_fit.pdf", bbox_inches='tight')
plt.show()


T = result.trend
S = result.seasonal
R = result.resid
############### Problem 6 ############################################
plt.figure(figsize=(11, 6))
plt.plot(df['Temp'], label = "Original data")
plt.plot(T, label="Seasonally adjusted")
plt.xlabel("Time")
plt.ylabel("Temp")
plt.legend()
plt.savefig("seasonally-adjusted.pdf", bbox_inches='tight')
plt.show()

############# Problem 7 and Problem 8 ###################################
def strength(T,S,R, type="trend"):
    if type == "trend":
        total = T + R
    elif type == "seasonality":
        total = S + R
    else:
        return
    return max(0, 1 - R.var() / total.var())

print(f"The strength of trend for this data set is {strength(T,S,R,'trend'):.4f}")
print(f"The strength of seasonality for this data set is {strength(T,S,R,'seasonality'):.4f}")




