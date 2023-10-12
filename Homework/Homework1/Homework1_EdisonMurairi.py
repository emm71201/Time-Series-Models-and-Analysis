#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
np.random.seed(6313)
#%%

def correlation_coefficent_cal(x,y):

    """ compute the correlation between two datasets x and y """

    mean_x, mean_y = np.mean(x), np.mean(y)

    return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) /\
          np.sqrt(sum((x[i] - mean_x)**2 for i in range(len(x)))\
                   * sum((y[i] - mean_y)**2 for i in range(len(y))))


# %%
x = [1,2,3,4,5]
y = [1,2,3,4,5]
z = [-1,-2,-3,-4,-5]
g = [1,1,0,-1,-1,0,1]
h = [0,1,1,1,-1,-1,-1]
# %%
print(f"The correlation coefficient between x and y is {correlation_coefficent_cal(x,y)}")
print(f"The correlation coefficient between x and z is {correlation_coefficent_cal(x,z)}")
print(f"The correlation coefficient between g and h is {correlation_coefficent_cal(g,h)}")
# %%
# Problem 2: Answered in the report
# Problem 3
tute1_url = "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv"
tute1 = pd.read_csv(tute1_url, index_col=0)
# tute1['Date'] = pd.to_datetime(tute1['Date'])
# tute1.set_index('Date', inplace=True)
print("Tute 1 Loaded")
# %%
# My setup of matplotlib
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc('text', usetex=False)
plt.rc('font', family='times')
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
fontsize = 16
#%%
# scatterplot : x axis Sales, y - axis GDP
ax = tute1.plot.scatter(x = 'Sales', y='GDP', figsize=(9,5), fontsize=12)
ax.set_xlabel('Sales', fontsize=13)
ax.set_ylabel('GDP', fontsize=13)
ax.set_title(f"Correlation Coefficient = {correlation_coefficent_cal(tute1['Sales'], tute1['GDP']):.2f}")
# plt.savefig("Figures/sales-gdp.pdf")
plt.show()
# %%
# scatterplot : x axis Sales, y - axis AdBudget
ax = tute1.plot.scatter(x = 'Sales', y='AdBudget', figsize=(9,5), fontsize=12)
ax.set_xlabel('Sales', fontsize=13)
ax.set_ylabel('AdBudget', fontsize=13)
ax.set_title(f"Correlation Coefficient = {correlation_coefficent_cal(tute1['Sales'], tute1['AdBudget']):.2f}")
#plt.savefig("Figures/sales-adbudget.pdf")
plt.show()

# %%
# scatterplot : x axis GDP, y - axis AdBudget
ax = tute1.plot.scatter(x = 'GDP', y='AdBudget', figsize=(9,5), fontsize=12)
ax.set_xlabel('GDP', fontsize=13)
ax.set_ylabel('AdBudget', fontsize=13)
ax.set_title(f"Correlation Coefficient = {correlation_coefficent_cal(tute1['GDP'], tute1['AdBudget']):.2f}")
#plt.savefig("Figures/gdp-adbudget.pdf")
plt.show()
# %%
# Problem 6 - Using the Seaborn package and pairplot() function, graph the correlation matrix for the tute1.csv dataset.
# Plot the Dataframe using the following options. Explain the graphs and justify the cross correlations. [15pts]
# a. kind="kde"
# b. kind="hist"
# c. diag_kind="hist"
#%%
# kind = kde
sns.pairplot(tute1, kind="kde")
#plt.savefig("Figures/tute1-pairplot-kde.pdf")
plt.show()
# %%
# kind = "hist"
sns.pairplot(tute1, kind="hist")
#plt.savefig("Figures/tute1-pairplot-hist.pdf")
plt.show()
# %%
sns.pairplot(tute1, diag_kind="hist")
#plt.savefig("Figures/tute1-pairplot-diagkind-kde.pdf")
plt.show()
# %%
# Problem 7 -- Using the Seaborn package and heatmap() function, graph the correlation
# matrix for the tute1.csv dataset.
# Explain the depicted correlation matrix

corr = tute1[["Sales", "AdBudget", "GDP"]].corr()
sns.heatmap(corr, annot=True, cmap='Blues')
#plt.savefig("Figures/tute1-corr.pdf")
plt.show()
#%%
# Develop a python program that asks a user to input numerical numbers: mean, variance & number
# of observations [default values : mean = 0, variance = 1, observations = 1000]. Then generates a
# random variable x that is normally distributed with above statistics. Create the following random
# variables: [15pts]
# a. 𝑦 = 𝑥
#%%
tmp_mean = input("Mean: ")
tmp_variance = input("Variance: ")
try:
    mean = float(tmp_mean)
except:
    print("Invalid value entered. Using the default, mean = 0")
    mean = 0

try:
    variance = float(tmp_variance)
except:
    print("Invalid value entered. Using the default, variance = 1")
    variance = 1
x = np.random.normal(mean, variance, 1000)
y = x**2
z = x**3
# %%
print(f"Using Mean = {mean}, Variance = {variance}\n")
print(f"\tThe correlation coefficient between x & y = {correlation_coefficent_cal(x,y):.3f}")
print(f"\tThe correlation coefficient between x & z = {correlation_coefficent_cal(x,z):.3f}")
# %%
# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax1,ax2 = axes
ax1.plot(x,y, ".")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
ax1.set_title(r"Plot of $y = x^2$")

ax2.plot(x,z, ".")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"z")
ax2.set_title(r"Plot of $z = x^3$")
#plt.savefig("Figures/xyz.pdf")
plt.show()
# %%
