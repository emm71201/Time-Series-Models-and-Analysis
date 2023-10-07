#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import acf

N = 1000
mean = 0
std = 1

np.random.seed(6313)
y = np.random.normal(mean, std, size = N)
date = pd.date_range(start='2000-01-01', periods = len(y), freq = 'D')

df = pd.DataFrame(data=y, columns=['y'], index=date)
df.plot()
plt.grid()
plt.xlabel('Date')
plt.ylabel('Mag')
plt.title('White Noise')
plt.tight_layout()
plt.show()


# %%
