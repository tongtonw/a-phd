import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

data = pd.read_csv('../err_gp_xy_test.csv')

file = '../zigzag_data.csv'
Dz = pd.read_csv(file)

#
feature_in = Dz.iloc[:, [10, 11, 19, 15, 16, 23, 1, 2, 3, 4, 25, 26]].values
knottoms = 0.514

input_d = feature_in * [1, 1, np.pi / 180, knottoms, knottoms, np.pi / 180, 1, 1 / 100 * 203, 1, 1 / 100 * 203, 1,
                        knottoms]

loc = [570, 800]  # 20/20, 60%
# loc = [850, 1000] # 30/30, 60%
# loc = [1791,1951]
# loc = [0, 499]
x_test = input_d[loc[0]:loc[1], :]

# get probablity

mu_n, sigma_n = data['n_hat'].values, data['std_n'].values  # mean and standard deviation
mu_e, sigma_e = data['e_hat'].values, data['std_e'].values  # mean and standard deviation

mu0, sigma0 = mu_n[5], sigma_n[5]
s = np.random.normal(mu0, sigma0, 100)
mu0e, sigma0e = mu_e[5], sigma_e[5]
se = np.random.normal(mu0e, sigma0e, 100)
# correlation coefficient between X and Y
cor_xy = np.corrcoef(s, se)[0, 1]
print(cor_xy)

centroid_n, centroid_e = x_test[5, 0]-x_test[0, 0], x_test[5, 1]-x_test[0, 1]
x, y = np.mgrid[mu0 - sigma0:mu0 + sigma0:.01,
       mu0e - sigma0e:mu0e + sigma0e:.01]

pos = np.dstack((x, y))
cov = [[sigma0 ** 2, cor_xy * sigma0 * sigma0e], [cor_xy * sigma0 * sigma0e, sigma0e ** 2]]
rv = multivariate_normal([mu0, mu0e], cov)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x_test[:10, 0]-x_test[0, 0], x_test[:10, 1]-x_test[0, 1])
ax2.pcolormesh(x+centroid_n, y+ centroid_e, rv.pdf(pos), cmap='RdBu_r')

# plt.plot(bins, 1 / (sigma0 * np.sqrt(2 * np.pi)) *
#          np.exp(- (bins - mu0) ** 2 / (2 * sigma0 ** 2)),
#          linewidth=2, color='r')

plt.show()
# points = np.linspace(-5, 5, 100)
# pdf = norm.pdf(points, 0, 1)
# plt.plot(points, pdf, color='r')
