# simulare AR(1)
# y(t) + 0.5y(t-1) = e(t)
#==========================

import numpy as np
np.random.seed(6313)
T = 1000
mean = 0
var = 1

e = np.random.normal(mean, var, T)
y = np.zeros(len(e))
for t in range(len(e)):
    if t==0:
        y[t] = e[t]
    else:
        #y[t] = #
        pass


from scipy import signal
num = [1,0]
den = [1,0.5]
sys = (num, den, 1)
_, y_dlsim = signal.dlsim(sys, e)
print(y_dlsim)