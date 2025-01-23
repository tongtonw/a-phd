import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def multivariate_gaussian_likelihood(D, x, sigma):
    n = len(x)
    det_sigma = np.linalg.det(sigma)
    norm_factor = (2*np.pi) ** (n/2) * np.sqrt(det_sigma)
    exp_factor = np.exp(-0.5 * (D-x).T @ np.linalg.inv(sigma) @ (D - x))
    return exp_factor / norm_factor


def BMA(p_m1, x1, var1, x2, var2, D):
    p_m2 = 1 - p_m1
    x_bma = p_m1 * x1 + p_m2 * x2

    var_bma_sq = (p_m1 * np.outer(x1 -x_bma, x1-x_bma)) + (p_m2 * np.outer(x2 -x_bma, x2-x_bma))

    var_bma = np.sqrt(var_bma_sq)
    # likelihood
    p_d_given_m1 = multivariate_gaussian_likelihood(D, x1, var1)
    p_d_given_m2 = multivariate_gaussian_likelihood(D, x2, var2)

    p_m1_prior = p_m1
    p_m2_prior = p_m2

    # posterior
    p_m1_posterior = (p_d_given_m1 * p_m1_prior) / (p_d_given_m1 * p_m1_prior + p_d_given_m2 * p_m2_prior)
    p_m2_posterior = (p_d_given_m2 * p_m2_prior) / (p_d_given_m1 * p_m1_prior + p_d_given_m2 * p_m2_prior)

    return x_bma, var_bma, p_m1_posterior, p_m2_posterior

# zigzag motion prediction error 
data = pd.read_csv('../error_m.csv').values[1791:1950, :]
data_gp = pd.read_csv('../err_gp_510.csv').values
# uvr prediction
u_gp = data_gp[:, 4:7]
var_u_gp = data_gp[:, 10:13]
u_math = data[:,  4:7]
var_u_math = np.std(data[:, 7:10], axis=0)
measurement = data_gp[:, 1:4]

# docking motion prediction error
# n_path = 2
# data = pd.read_csv('../error_m_docking{}.csv'.format(n_path)).values
# data_gp = pd.read_csv('../err_gp_xy_test_docks{}.csv'.format(n_path)).values

# NE prediction
# u_gp = data_gp[:, 4:7]
# var_u_gp = data_gp[:, 10:13]
# u_math = data[:, 13:16]
# var_u_math = np.std(data[:, 16:], axis=0)
# measurement = data_gp[:, 1:4]

print('math err_u std', var_u_math)

print(len(u_gp), len(u_math))
n = len(u_gp)
p_m1 = 0.5
uAvg = []
varAvg = []
P_m1 = []
for i in range(n):
    # print(var_u_gp[i])
    var1 = np.diag(var_u_gp[i]**2)
    # print(var1)
    var2 = np.diag(var_u_math**2)
    # print(var2)
    x_avg, var_avg, p_m1_post, p_m2_post = BMA(p_m1, u_gp[i], var1, u_math[i], var2, measurement[i])
    uAvg.append(x_avg)
    varAvg.append(var_avg)
    p_m1 = p_m1_post
    P_m1.append(p_m1)

eu_avg = measurement - np.asarray(uAvg)

# plt.figure()
# plt.boxplot([data[1791:, 9], data_gp[:, 9], e_avg], labels=['physics', 'data', 'average'], showmeans=True)
# plt.show()
plt.figure()
plt.boxplot([data[:, 7], data_gp[:, 7], eu_avg[:, 0]], labels=['physics', 'gp', 'bma'], showmeans=True)
plt.figure()
plt.boxplot([data[:, 8], data_gp[:, 8], eu_avg[:, 1]], labels=['physics', 'gp', 'bma'], showmeans=True)
plt.figure()
plt.boxplot([data[:, 9], data_gp[:, 9], eu_avg[:, 2]], labels=['physics', 'gp', 'bma'], showmeans=True)
plt.figure()
plt.plot(P_m1)
plt.show()
# docking
# plt.figure()
# plt.boxplot([data[:, 16], data_gp[:, 7], eu_avg[:, 0]], labels=['physics', 'gp', 'bma'], showmeans=True)
# plt.figure()
# plt.boxplot([data[:, 17], data_gp[:, 8], eu_avg[:, 1]], labels=['physics', 'gp', 'bma'], showmeans=True)
# plt.figure()
# plt.boxplot([data[:, 18], data_gp[:, 9], eu_avg[:, 2]], labels=['physics', 'gp', 'bma'], showmeans=True)
# plt.figure()
# plt.plot(P_m1)
# plt.show()
