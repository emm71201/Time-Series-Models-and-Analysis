import numpy as np
def ACF(data, tau):
    """ compute the ACF at some lag = tau """
    # handle negative lag
    if tau < 0:
        return ACF(data, -tau)
    mean = data.mean()
    AA = sum((data[t] - mean) * (data[t - tau] - mean) for t in range(tau, len(data)))
    BB = sum((data[t] - mean) ** 2 for t in range(len(data)))
    return AA / BB

def ACF_Calc(data, number_lags):
    try:
        data = np.array(data)
    except:
        print("The data must be a one - D array")
        return

    mean = data.mean()
    BB = sum((data[t] - mean) ** 2 for t in range(len(data)))

    tmp = np.array([sum((data[t] - mean) * (data[t - tau] - mean) \
                        for t in range(tau, len(data))) for tau in range(number_lags + 1)]) / BB

    return np.concatenate((tmp[::-1][:-1], tmp))


def plot_ACF(acf_list, number_lags, size, ax=None):
    x = np.array([i for i in range(-number_lags, number_lags + 1)])

    # print(len(x), len(acf_list))

    # insignificance band
    xfill = [i for i in range(-number_lags, number_lags + 1)]
    ydown = np.full((len(xfill),), -1.96 / np.sqrt(size))
    yup = np.full((len(xfill),), 1.96 / np.sqrt(size))

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