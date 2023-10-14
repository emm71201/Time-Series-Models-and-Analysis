import copy
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as sms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.simplefilter('ignore')
np.random.seed(6331)

# My setup of matplotlib
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc('text', usetex=False)
plt.rc('font', family='times')
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
fontsize = 18

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

print("\n============ Problem 7 ==================\n")
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
    with open(f"{metric}.tex", "w") as f:
        f.write(model.summary().as_latex())
    print("\t \t The predictors selected are")
    print(predictors)
    
    count += 1

print("\n============= END of Problem 7 =================\n")

print("\n============= Problem 8 =================\n")

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
    with open(f"vif-{metric}.tex", "w") as f:
        f.write(model.summary().as_latex())
    print("\n\t \t The predictors selected are")
    print(predictors)
    
    count += 1

print("\n============= END of Problem 8 ===================\n")

print("\n=============== Problem 9 ===================\n")
print("\nAnswered in the report\n")
print("============= END of Problem 9 ===================")

print("\n============= Problem 10 ===================\n")
def get_final_model(train):
    predictors = ['engine-size', 'stroke', 'compression-ratio', 'peak-rpm', 'horsepower', 'width']
    y = train["price"]
    X = standardizer.fit_transform(train[predictors].to_numpy())
    model = sm.OLS(y, sm.add_constant(X))
    model = model.fit()
    return model, predictors

final_model, final_predictors = get_final_model(train)
print(final_model.summary())
print("\n============= END of Problem 10 ===================\n")


print("\n============= Problem 11 ===================\n")

def predict(test, model, predictors):
    X = standardizer.fit_transform(test[predictors].to_numpy())
    predictions = model.predict(sm.add_constant(X))
    return predictions

train_prices = train['price'].to_numpy()

tt = [len(train_prices)+jj for jj in range(test.shape[0])]
predictions = predict(test, final_model, final_predictors)
test_prices = test['price'].to_numpy()
plt.figure(figsize=(10,7))
plt.plot(train_prices, label="Training set")
plt.plot(tt,test_prices, label="Testing set")
plt.plot(tt, predictions, label="Predictions")
plt.xlabel("Item")
plt.ylabel("Price")
plt.title("Prices of the training set, testing set and the predictions made")
plt.legend()
plt.tight_layout()
#plt.savefig("price-plot.pdf")
plt.show()

print("\n============= END of Problem 11 ===================\n")

print("\n============= Problem 12 ===================\n")

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


error = predictions - test_prices
nlags = 20
acf_list = ACF_Calc(error, nlags)
error_acfplot = plot_ACF(acf_list, nlags, len(acf_list))
#plt.savefig("error-acf.pdf")
plt.show()

print("\n============= END of Problem 12 ===================\n")

print("\n============= Problem 13 ===================\n")

tresult = sms.weightstats.ttest_ind(test_prices, predictions)
print(f"\n\tThe T Test resulrs are: \n\nT-value = {tresult[0]:.4f}\nP-value = {tresult[1]:.5f}\ndf = {tresult[2]}\n")

# ftest
print("\n F-TEST\n\n")
Xtest = standardizer.fit_transform(test[final_predictors].to_numpy())
print(final_model.f_test(sm.add_constant(Xtest)).summary())

print("\n============= END of Problem 13 ===================\n")

print("\n============= END of LAB3 ===================\n")