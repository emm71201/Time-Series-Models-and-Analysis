import numpy as np
import statsmodels.api as sm
import pandas as pd

X = np.array([[i] for i in range(1,6)])
Y = np.array([[2], [4], [5], [4], [5]])
X = sm.add_constant(X)

model = sm.OLS(Y,X).fit()
print("finished")
print(model.summary())

