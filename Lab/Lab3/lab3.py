import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.random.seed(6331)

url = "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/autos.clean.csv"
columns = ["price", 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size',\
    'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg','highway-mpg']
independent_cols = columns[1:]

df = pd.read_csv(url)[columns]
X = df[independent_cols]
Y = df[["price"]]
print("Data succesfully obtained")

#problem 1
# X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)
train, test = train_test_split(df, test_size=0.20)
print(f"The size of the train set is {train.shape[0]}")
print(f"The size of the test set is {test.shape[0]}")

#Problem 2
corr_matrix = train.corr()
sns.heatmap(corr_matrix)
# plt.show()

X_train = train.to_numpy()[:,1:]
y_train = train["price"].to_numpy()
#Problem 3

S,V,D = np.linalg.svd(X_train)
print(f"The minimum lambda from SVD is {V.min():.2f}")
print(f"The condition number is {np.sqrt(V.max()/V.min()):.2f}")

# Problem 4: standardize the dataset
standardizer = StandardScaler()
X_train = standardizer.fit_transform(X_train)

# Problem 5
def make_xymatrix(X,y):
    firstcol = np.array([[1] for _ in range(X.shape[0])])
    return np.hstack((firstcol, X)), y
def LSE_equation(X,Y):

    return np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y

X,Y = make_xymatrix(X_train, y_train)
beta = LSE_equation(X,Y)
print(beta)

# Problem 6
model  = sm.OLS(y_train, sm.add_constant(X_train))
results = model.fit()
print(results.summary())

# Problem 7
