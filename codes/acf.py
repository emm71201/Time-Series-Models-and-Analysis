#%%
import numpy as np
# %%
def ACF(data, tau):

    if tau == 0:
        return 1

    if tau > 0:

        mean = np.mean(data)

        numerator = sum((data[t] - mean)*(data[t-tau] - mean) for t in range(tau, len(data)))
        denominator = sum((data[t] - mean)**2 for t in range(len(data)))

        return numerator/denominator
    
    return ACF(data, -tau)

def Cal_ACF(data):

    res = [ACF(data, i) for i in range(len(data))]
    res = res[::-1] + res

    return np.array(res)
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 20
mean = 0
std = 1

#date = pd.date_range(start='2000-01-01', periods = len(y), freq = 'D')

# Create an auto regressive process
# y(t) - 0.9y(t-1) = e(t)
###################################

np.random.seed(6313)
e = np.random.normal(mean, std, size=N)
y = np.zeros(len(e))
for t in range(len(e)):
    if t == 0:
        y[t] = e[t]
    else:
        y[t] = 0.9 * y[t-1] + e[t]


# df = pd.DataFrame(data=y, columns=['y'], index=date)
# df.plot()
# plt.grid()
# plt.xlabel('Date')
# plt.ylabel('Mag')
# plt.title('White Noise')
# plt.tight_layout()
# plt.show()
# %%
plt.plot(Cal_ACF(y), marker=".")
m = 1.96/np.sqrt(len(y))
plt.axhspan(-m, m, color='cyan')
plt.show()
# %%
