import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, seasonal_decompose


url = "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/AirPassengers.csv"
data = pd.read_csv(url)

Temp = data["#Passengers"]
Temp = pd.Series(np.array(Temp), index = pd.date_range('1981-01-01', periods=len(Temp), freq='m'), name = 'Temp')
Temp_df = pd.DataFrame(Temp)

Temp_df.plot()
plt.show()

#==================
# STL decomposition
#======================
STL = STL(Temp)
res = STL.fit()

T = res.trend
S = res.seasonal
R = res.resid

def strength(T,R):
    T,R = np.array(T), np.array(R)
    AT = T + R
    AS = T + S

    return max(0,1 - R.var()/AS.var()),  max(0,1 - R.var()/AT.var())

Ft, Fs = strength(T,R)
print(f"Ft = {Ft:.2f}, Fs = {Fs:.2f}")

fig = res.plot()
plt.show()
