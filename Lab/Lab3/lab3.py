#%%
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(6331)

url = "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/autos.clean.csv"
columns = ["price", 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size',\
    'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg','highway-mpg']
predictors = columns[1:]

df = pd.read_csv(url)[columns]
X = df[predictors]
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
plt.title("Correlation map")
plt.tight_layout()
# plt.savefig("correlation_map.pdf")
# plt.show()

X_train = train.to_numpy()[:,1:]
y_train = train["price"].to_numpy()
print("\n===== END of Problem 2 ============\n")
#Problem 3
print("\n===== Problem 3=====================\n")
S,V,D = np.linalg.svd(X_train)

print(f"\nThe eigenvalues from SVD are:\n")
for val in V:
    print(f"{val:.2e}", sep=", ")

print(f"\nThe minimum lambda from SVD is {V.min():.2f}\n")

print(f"\nThe maximum lambda from SVD is {V.max():.2f}\n")

print(f"\nThe condition number is {V.max()/V.min():.2f}\n")

print("\n===== END of Problem 3 ============\n")

print("\n===== Problem 4 =====================\n")
# Problem 4: standardize the dataset
standardizer = StandardScaler()
X_train = standardizer.fit_transform(X_train)
print("\nThe dataset has been standardized\n")

print("\n===== END of Problem 4 ============\n")

print("\n===== Problem 5 =====================\n")
# Problem 5
def make_xymatrix(X,y):
    firstcol = np.array([[1] for _ in range(X.shape[0])])
    return np.hstack((firstcol, X)), y
def LSE_equation(X,Y):

    return np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y

X,Y = make_xymatrix(X_train, y_train)
beta = LSE_equation(X,Y)
print("\nThe regression model coefficients are: \n")
for val in beta:
    print(f"{val:.2f}", sep=",")

print()
print("\n===== END of Problem 5 ============\n")

print("\n===== Problem 6 =====================\n")
# Problem 6
model  = sm.OLS(y_train, sm.add_constant(X_train))
results = model.fit()
print(results.summary())

# print(results.summary().as_latex())

print("\n===== END of Problem 6 ===============\n")

# Problem 7
def find_largestpvalue_predictor(model, predictors):

    index = np.argmax(model.pvalues)
    predictor = predictors[index-1]
    pvalue = model.pvalues[index]

    return predictor

def compute_metrcic(model, metric = "rsquared_adj"):

    if metric == "rsquared_adj":
        metric_value = model.rsquared_adj
    elif metric == "aic":
        metric_value = model.aic
    elif metric == "bic":
        metric_value = model.bic
    
    return metric_value

#%%
def backward_regression(train, predictors, metric="rsquared_adj"):

    curr_metric = None
    curr_model = None

    y = train["price"]
    X = standardizer.fit_transform(train[predictors].to_numpy())
    tmp_model = sm.OLS(y, sm.add_constant(X))
    tmp_model = tmp_model.fit()
    tmp_metric = compute_metrcic(tmp_model, metric)


    if metric == "rsquared_adj":
        # maximize the rsquared_adjusted
        curr_metric = 0
        while tmp_metric > curr_metric:
            curr_metric = tmp_metric
            curr_model = tmp_model
            largest_pvalue_predictor = find_largestpvalue_predictor(curr_model,predictors)

            # remove the predictor with the largest p-value
            predictors.remove(largest_pvalue_predictor)

            # fit the new model
            X = standardizer.fit_transform(train[predictors].to_numpy())
            tmp_model = sm.OLS(y, sm.add_constant(X))
            tmp_model = tmp_model.fit()
            tmp_metric = compute_metrcic(tmp_model, metric)


        return curr_model, predictors+[largest_pvalue_predictor]
    
    else:
        # minimize the metric
        curr_metric = 1e10
        while tmp_metric < curr_metric:
            curr_metric = tmp_metric
            curr_model = tmp_model
            largest_pvalue_predictor = find_largestpvalue_predictor(curr_model,predictors)

            # remove the predictor with the largest p-value
            predictors.remove(largest_pvalue_predictor)

            # fit the new model
            X = standardizer.fit_transform(train[predictors].to_numpy())
            tmp_model = sm.OLS(y, sm.add_constant(X))
            tmp_model = tmp_model.fit()
            tmp_metric = compute_metrcic(tmp_model, metric)


        return curr_model, predictors+[largest_pvalue_predictor]

    return

metrics = ["rsquared_adj", "aic", "bic"]
mymap = {"rsquared_adj":"ADJUSTED R^2", "aic":"AIC", "bic":"BIC"}
print("Using eliminating the variable with the highest Pvalue:\n")
count = 1
for metric in metrics:
    preds = copy.deepcopy(predictors)
    print(f"\t {count}. {mymap[metric]}:\n")
    print("\t \t The model summary is: \n")
    model, predictors = backward_regression(train, preds, metric)
    print(model.summary())
    print("\t \t The predictors selected are")
    print(predictors)
    
    count += 1

#%%

def find_largest_vif(train, predictors):
    X = train[predictors]
    curr_vif = 0
    curr_pred = None
    for i in range(len(predictors)):
        predictor = predictors[i]
        tmp_vif = variance_inflation_factor(X.values, i)
        if tmp_vif > curr_vif:
            curr_vif = tmp_vif
            curr_pred = predictor
    
    return curr_pred

def VIF_method(train, predictors, metric="rsquared_adj"):

    curr_metric = None
    curr_model = None

    y = train["price"]
    X = standardizer.fit_transform(train[predictors].to_numpy())
    tmp_model = sm.OLS(y, sm.add_constant(X))
    tmp_model = tmp_model.fit()
    tmp_metric = compute_metrcic(tmp_model, metric)

    if metric == "rsquared_adj":
        
        # maximize the rsquared_adjusted
        curr_metric = 0
        while tmp_metric > curr_metric:
            
            curr_metric = tmp_metric
            curr_model = tmp_model

            largest_vif_predictor = find_largest_vif(train[predictors], predictors)

            # remove the predictor with the largest p-value

            predictors.remove(largest_vif_predictor)

            # fit the new model
            X = standardizer.fit_transform(train[predictors].to_numpy())
            tmp_model = sm.OLS(y, sm.add_constant(X))
            tmp_model = tmp_model.fit()
            tmp_metric = compute_metrcic(tmp_model, metric)
            print(curr_metric, tmp_metric)


        return curr_model, predictors+[largest_vif_predictor]
    
    else:
        # minimize the metric
        curr_metric = 1e10
        while tmp_metric < curr_metric:
            curr_metric = tmp_metric
            curr_model = tmp_model
            largest_vif_predictor = find_largest_vif(train[predictors], predictors)

            # remove the predictor with the largest p-value
            predictors.remove(largest_vif_predictor)

            # fit the new model
            X = standardizer.fit_transform(train[predictors].to_numpy())
            tmp_model = sm.OLS(y, sm.add_constant(X))
            tmp_model = tmp_model.fit()
            tmp_metric = compute_metrcic(tmp_model, metric)


        return curr_model, predictors+[largest_vif_predictor]

    return

#%%
metrics = ["rsquared_adj", "aic", "bic"]
mymap = {"rsquared_adj":"ADJUSTED R^2", "aic":"AIC", "bic":"BIC"}
print("VIF METHOD:\n")
count = 1
for metric in metrics:
    preds = copy.deepcopy(predictors)
    print(f"\n\t {count}. {mymap[metric]}:\n")
    print("\n\t \t The model summary is: \n")
    model, predictors = VIF_method(train, preds, metric)
    print(model.summary())
    print("\n\t \t The predictors selected are")
    print(predictors)
    
    count += 1