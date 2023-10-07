import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

size = 1000
nfeatures = 4
X =  []
for i in range(nfeatures):
    X.append(np.random.normal(0, 1, size))
    #X = np.vstack(X,np.random.normal(0, 1, size))
X = np.array(X).transpose()

s,v,d = np.linalg.svd(X)
print(f"{np.sqrt(v.max()/v.min()):.2f}")

X = pd.DataFrame(X, columns=['A', 'B', 'C', 'D'])
vif_data1 = pd.DataFrame()
vif_data1['features'] = X.columns
vif_data1['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data1)