import numpy as np
import pandas as pd
import helpers as hlp
from matplotlib import pyplot as plt
import matplotlib as mpl

# My setup of matplotlib
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc('text', usetex=False)
plt.rc('font', family='times')
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
fontsize = 20

def print_info(model, forecast, model_residual, forecast_residual):
    print("\n\nPRINTING DATA \n\n")
    print(f"One step predictions:")
    for item in model:
        print(f"\t{item:.2f}", end=", ")
    print(f"\nH steps predictions:")
    for item in forecast:
        print(f"\t{item:.2f}", end=", ")
    print(f"\nTraining set residuals:")
    for item in model_residual:
        print(f"\t{item:.2f}", end=", ")
    print(f"\nTest set residuals:")
    for item in forecast_residual:
        print(f"\t{item:.2f}", end=", ")
    print(f"\nTraining set square residuals:")
    for item in model_residual ** 2:
        print(f"\t{item:.2f}", end=", ")
    print(f"\nTest set square residuals:")
    for item in forecast_residual ** 2:
        print(f"\t{item:.2f}", end=", ")

    print("\n\nFINISHED PRINTING\n\n")

def plotting(train, test, model, forecast, model_name):

    x1 = [i for i in range(train.shape[0])]
    x2 = [t for t in range(train.shape[0], train.shape[0] + test.shape[0])]
    x3 = [t for t in range(train.shape[0], train.shape[0] + test.shape[0])]

    fig = plt.figure(figsize=(7,3))
    plt.title(f"{model_name}")
    plt.plot(x1, train, '.-', label = 'Training Data')
    #plt.plot(x1[1:], model, ".-", label="Model")
    plt.plot(x2, test, ".-", label='Test Data')
    plt.plot(x3, forecast, ".-", label='Forecast')
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()

    return fig

class Model:
    def __init__(self):
        pass

    def one_step_mse(self, train, model):
        train_start = 2
        model_start = model.shape[0] - (train.shape[0] - train_start)
        return np.mean((train[train_start:] - model[model_start:])**2)
    def forecast_mse(self, test, forecast):
        return np.mean((test - forecast)**2)
    def model_residual(self, train, model):
        train_start = 2
        model_start = model.shape[0] - (train.shape[0] - train_start)
        return train[train_start:] - model[model_start:]
    def forecast_residual(self, test, forecast):
        return test - forecast

    def training_residual_Qvalue(self, train, model, h = None):
        if h is None:
            h = len(train)
        residuals = self.model_residual(train, model)
        return (len(train)-2) * sum(hlp.ACF(residuals, k)**2 for k in range(2, h))

    def forecast_residual_Qvalue(self, test, forecast, T, h=None):
        if h is None:
            h = len(test)
        residuals = self.forecast_residual(test, forecast)
        return T * sum(hlp.ACF(residuals, k) for k in range(h + 1))

class Average_Method(Model):
    def __init__(self):
        super().__init__()
    def one_step_ahead(self, train):
        return np.array([train[:t].mean() for t in range(1, train.shape[0])])
    def forecast(self, train, nsteps):
        return np.full(nsteps, train.mean())

class Naive_Method(Model):
    def __init__(self):
        super().__init__()

    def one_step_ahead(self, train):
        return np.array([train[t-1] for t in range(1, train.shape[0])])

    def forecast(self, train, nsteps):
        return np.full(nsteps, train[-1])

class Drift_Method(Model):

    def __init__(self):
        super().__init__()
    def one_step_ahead(self, train):

        return np.array([train[t] + (train[t] - train[0])/t for t in range(2,train.shape[0])])

    def forecast(self, train, nsteps):

        return np.array([train[-1] + h * (train[-1] - train[0])/(train.shape[0]-1) for h in range(1,nsteps+1)])

class Exponential_Method(Model):

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def one_step_ahead(self, train):

        lo = train[0]

        return np.array([(1 - self.alpha)**t * lo + \
            sum(self.alpha * (1 - self.alpha)**j * train[t - j - 1] for j in range(t) ) for t in range(1, train.shape[0])])

    def forecast(self, train, nsteps):
        lo = train[0]
        t = train.shape[0]
        return np.array([(1 - self.alpha)**t * lo + \
                sum(self.alpha * (1 - self.alpha)**j * train[t - j - 1] for j in range(t)) for _ in range(nsteps)])


train = np.array([112,118,132,129,121,135,148,136,119])
test = np.array([104,118,115,126,141])
nsteps = test.shape[0]

print("="*52)
print("="*18, "AVERAGE METHOD", "="*18)
print("="*52)

# avg method
avg_method = Average_Method()
avg_model = avg_method.one_step_ahead(train)
avg_forecast = avg_method.forecast(train, nsteps)
avg_model_residual = avg_method.model_residual(train, avg_model)
avg_forecast_residual = avg_method.forecast_residual(test, avg_forecast)

print_info(avg_model, avg_forecast, avg_model_residual, avg_forecast_residual)


myplot = plotting(train, test, avg_model, avg_forecast, "Average Method")
plt.savefig("avg-method-plt.pdf", bbox_inches="tight")
plt.show()


print(f"\nAVG METHOD 1-STEP MSE = {avg_method.one_step_mse(train, avg_model):.2f}")
print(f"\nAVG METHOD H-STEPS MSE = {avg_method.forecast_mse(test, avg_forecast):.2f}")
print(f"\nAVG METHOD VARIANCE OF PREDICTION ERROR = {avg_model_residual.var():.2f}")
print(f"\nAVG METHOD VARIANCE OF FORECAST ERROR = {avg_forecast_residual.var():.2f}")
print(f"\nAVG METHOD TRAINING RESIDUALS Q-VALUE = {avg_method.training_residual_Qvalue(train, avg_model, 5):.2f}")

print("="*50)
print("="*18, "NAIVE METHOD", "="*18)
print("="*50)

naive_method = Naive_Method()
naive_model = naive_method.one_step_ahead(train)
naive_forecast = naive_method.forecast(train,nsteps)
naive_model_residual = naive_method.model_residual(train, naive_model)
naive_forecast_residual = naive_method.forecast_residual(test, naive_forecast)

print_info(naive_model, naive_forecast, naive_model_residual, naive_forecast_residual)

myplot = plotting(train, test, naive_model, naive_forecast, "Naive Method")
plt.savefig("naive-method-plt.pdf", bbox_inches="tight")
plt.show()

print(f"\nNAIVE METHOD 1-STEP MSE = {naive_method.one_step_mse(train, naive_model):.2f}")
print(f"\nNAIVE METHOD FORECAST H-STEPS MSE = {naive_method.forecast_mse(test, naive_forecast):.2f}")
print(f"\nNAIVE METHOD VARIANCE OF PREDICTION ERROR = {naive_model_residual.var():.2f}")
print(f"\nNAIVE METHOD VARIANCE OF FORECAST ERROR = {naive_forecast_residual.var():.2f}")
print(f"\nNAIVE METHOD TRAINING RESIDUALS Q-VALUE = {naive_method.training_residual_Qvalue(train, naive_model, 5):.2f}")


print("="*50)
print("="*18, "DRIFT METHOD", "="*18)
print("="*50)
# drift method
drift_method = Drift_Method()
drift_model = drift_method.one_step_ahead(train)
drift_forecast = drift_method.forecast(train, nsteps)
drift_model_residual = drift_method.model_residual(train, drift_model)
drift_forecast_residual = drift_method.forecast_residual(test, drift_forecast)

print_info(drift_model, drift_forecast, drift_model_residual, drift_forecast_residual)

myplot = plotting(train, test, drift_model, drift_forecast, "Drift Method")
plt.savefig("drift-method-plt.pdf", bbox_inches="tight")
plt.show()

print(f"\nDRIFT METHOD 1-STEP MSE = {drift_method.one_step_mse(train, drift_model):.2f}")
print(f"\nDRIFT METHOD FORECAST H-STEPS MSE = {drift_method.forecast_mse(test, drift_forecast):.2f}")
print(f"\nDRIFT METHOD VARIANCE OF PREDICTION ERROR = {drift_model_residual.var():.2f}")
print(f"\nDRIFT METHOD VARIANCE OF FORECAST ERROR = {drift_forecast_residual.var():.2f}")
print(f"\nDRIFT METHOD TRAINING RESIDUALS Q-VALUE = {drift_method.training_residual_Qvalue(train, drift_model, 5):.2f}")

print("="*50)
print("="*18, "EXPONENTIAL METHOD", "="*18)
print("="*50)

alpha = 0.5
ses_method = Exponential_Method(alpha)
ses_model = ses_method.one_step_ahead(train)
ses_forecast = ses_method.forecast(train, nsteps)
ses_model_residual = ses_method.model_residual(train, ses_model)
ses_forecast_residual = ses_method.forecast_residual(test, ses_forecast)


print_info(ses_model, ses_forecast, ses_model_residual, ses_forecast_residual)

myplot = plotting(train, test, ses_model, ses_forecast, "SES Method")
plt.savefig("ses-method-plt.pdf", bbox_inches="tight")
plt.show()

print()
print(f"\nSES METHOD 1-STEP MSE = {ses_method.one_step_mse(train, ses_model):.2f}")
print(f"\nSES METHOD FORECAST H-STEPS MSE = {ses_method.forecast_mse(test, ses_forecast):.2f}")
print(f"\nSES METHOD VARIANCE OF PREDICTION ERROR = {ses_model_residual.var():.2f}")
print(f"\nSES METHOD VARIANCE OF FORECAST ERROR = {ses_forecast_residual.var():.2f}")
print(f"\nSES METHOD TRAINING RESIDUALS Q-VALUE = {ses_method.training_residual_Qvalue(train, ses_model, 5):.2f}")


print("="*52)
print("="*18, "QUESTION 9", "="*18)
print("="*52)

alphas = [0, 0.25, 0.75, 0.99]
ses_methods = (Exponential_Method(alpha) for alpha in alphas)
ses_data = [(ses.one_step_ahead(train), ses.forecast(train, nsteps)) for ses in ses_methods]

def plotting_modified(ax, train, test, model, forecast, model_name):

    x1 = [i for i in range(train.shape[0])]
    x2 = [t for t in range(train.shape[0], train.shape[0] + test.shape[0])]
    x3 = [t for t in range(train.shape[0], train.shape[0] + test.shape[0])]

    #fig = plt.figure(figsize=(7,3))
    ax.set_title(f"{model_name}")
    ax.plot(x1, train, '.-', label = 'Training Data')
    #plt.plot(x1[1:], model, ".-", label="Model")
    ax.plot(x2, test, ".-", label='Test Data')
    ax.plot(x3, forecast, ".-", label='Forecast')
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.legend(framealpha=0.5)

    return ax

# My setup of matplotlib
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc('text', usetex=False)
plt.rc('font', family='times')
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
fontsize = 8
fig, axes = plt.subplots(2,2, figsize=(8,5))
fig.subplots_adjust(hspace=1)
count = 0
for i in range(2):
    for j in range(2):
        model, forecast = ses_data[count]
        model_name = f"SES alpha = {alphas[count]}"
        axes[i,j] = plotting_modified(axes[i,j], train, test, model, forecast, model_name)
        count += 1
plt.savefig("ses_methods.pdf", bbox_inches='tight')
plt.show()
