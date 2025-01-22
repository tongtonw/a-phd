import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Bayesian Information Criterion to analyze models.
# I = -2 * log(L)+k * log(n)

def bic(n, k, mse):
    # n is the number of observations
    # k is the number of predictor variables in the regression model
    sigma = mse* n/(n-k)
    L = -n/2*np.log10(2*np.pi)-n/2*np.log(sigma)-mse *n/2/sigma
    bic = -2 * np.log10(L) + k * np.log10(n)
    # likelihood1 = np.exp(-bic / 2)
    return bic


# probability for  Mi  is， a posterior probability distribution
# def post_prob():
#     p1 = likelihood1 * pM1_pri / np.sum(likelihood1 * pM1_pri, likelihood2 * pM2_pri)
#     p2 = likelihood2 * pM2_pri / np.sum(likelihood1 * pM1_pri, likelihood2 * pM2_pri)


#
#
# def bayes_factor():
#     bf = likelihood1 * pM1 / (likelihood2 * pM2)
def weight(I1, I2):
    w1 = np.exp(-I1 / 2)
    w2 = np.exp(-I2 / 2)
    total = w1 + w2

    return w1 / total, w2 / total


data = pd.read_csv('../error_m.csv').values

data_gp = pd.read_csv('../err_gp_510.csv').values

n = len(data)
k = 9
mse = np.sum(data[1791:1950, 9] ** 2) / len(data[1791:1950, 9])
mseg = np.sum(data_gp[:, 9] ** 2) / len(data_gp[:, 9])
print(mse, mseg)

Im = bic(len(data[1791:1950, 9]), k, mse)
Ig = bic(len(data_gp[:, 9]), k, mseg)
print(Im, Ig)
I1, I2 = weight(Im, Ig)
print('physics weight: ', I1, 'data weight: ', I2)

#  get average r
# print(data[1791, 3] == data_gp[0, 3])
# rAvg = I1 * data[1791:, 6] + I2 * data_gp[:, 6]
# e_avg = data_gp[:, 3] - rAvg
# mse_avg = np.sum(e_avg ** 2)/len(e_avg)

uAvg = I1 * data[1791:1950, 6] + I2 * data_gp[:, 6]
eu_avg = data_gp[:, 1] - uAvg

# plt.figure()
# plt.boxplot([data[1791:, 9], data_gp[:, 9], e_avg], labels=['physics', 'data', 'average'], showmeans=True)
# plt.show()

plt.figure()
plt.boxplot([data[1791:1950, 9], data_gp[:, 9], eu_avg], labels=['physics', 'data', 'average'], showmeans=True)
plt.show()
