import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data_m = pd.read_csv('../error_m_ne.csv').values
data_gp = pd.read_csv('../err_gp_xy_train.csv').values

X_1 = data_m[:498, 13].reshape(-1, 1)
X_2 = data_gp[:, 4].reshape(-1, 1)
X_3 = data_gp[:, 10].reshape(-1, 1)
# plt.figure()
# plt.plot(X_3)
# plt.show()
# X_4 = np.sqrt((X_1 - X_2) ** 2)
Y_true = data_m[:498, 10].reshape(-1, 1)
X = np.hstack([X_1, X_2, X_3])

test_gp = pd.read_csv('../err_gp_xy_test.csv').values

testX_1 = data_m[571:700, 13].reshape(-1, 1)
testX_2 = test_gp[:-100, 4].reshape(-1, 1)
testX_3 = test_gp[:-100, 10].reshape(-1, 1)

testX_4 = np.sqrt((testX_1 - testX_2) ** 2)
testY_true = data_m[571:700, 10].reshape(-1, 1)
Xtest = np.hstack([testX_1, testX_2, testX_3])


# X = np.vstack([X, Xtest])
# Y_true = np.vstack([Y_true, testY_true])

# plt.figure()
# plt.plot((X[:, 0].reshape(-1, 1) - Y_true))
# plt.plot((X[:, 1].reshape(-1, 1) - Y_true))
# plt.show()
#
# plt.figure()
# plt.plot(tf.math.exp(-(X[:, 0].reshape(-1, 1) - Y_true) ** 2 / 2 / 0.3 ** 2))
# plt.plot(tf.math.exp(-(X[:, 1].reshape(-1, 1) - Y_true) ** 2 / 2 / X[:, 2].reshape(-1, 1) ** 2))
# plt.show()


def loss_fn(y_pred, data):
    """ y_pred is the weight of physics model"""
    y_true = data[:, 0]
    i = data[:, 1]
    k = data[:, 2]
    std = data[:, 3]

    mean_ = (y_pred * i + (1 - y_pred) * k - y_true) ** 2
    # prob_ = -np.log10(
    #     (y_pred * np.exp(-(i - y_true) ** 2 / 2 / 0.3 ** 2) / 0.3 / np.sqrt(2*3.14) +
    #      (1-y_pred) * np.exp(-(k - y_true) ** 2 / 2 / std ** 2) / std / np.sqrt(2*3.14)))
    # mean_ = (y_pred * i + (1 - y_pred) * k - y_true) ** 2
    # prob_ = -tf.experimental.numpy.log10(
    #     (y_pred * tf.math.exp(-(i - y_true) ** 2 / 2 / 0.001 ** 2) + (1 - y_pred) * tf.math.exp(-(k - y_true) ** 2 / 2 / std ** 2)))
    prob_ = y_pred ** 2 * 0.3 ** 2 + (1 - y_pred) ** 2 * std ** 2

    return prob_ + mean_


Y = np.hstack([Y_true, X])

wp = []
for j in range(len(X)):
    res = minimize(loss_fn, x0=0.5, args=(Y[j, :].reshape(1, -1),), method='L-BFGS-B', options={'disp': False},
                   bounds=((0, 1),))
    # print(res.x)
    wp.append(res.x)

plt.figure()
plt.plot(wp)
plt.show()

# pd.DataFrame(wp, columns=['w']).to_csv('../Pweights_train.csv')
# pd.DataFrame(wp, columns=['w']).to_csv('../Pweights_train.csv')
# checkpoint_path = './tmp/checkpoint'
# checkpoint_dir = os.path.dirname(checkpoint_path)

# model.save(checkpoint_path)


# Y_pre = model.predict(X)
# plt.figure()
# plt.plot(Y_pre)
# # plt.show()
# wp = Y_pre[:498].reshape(-1, 1)
wd = 1-np.array(wp)
weightY = wp * X_1 + wd * X_2
stdY = np.sqrt(np.array(wp) ** 2 * 0.3 ** 2 + wd ** 2 * X_3 ** 2)
#
ans = np.hstack([np.array(wp), weightY, stdY])
pd.DataFrame(ans, columns=['w', 'yw', 'stdyw']).to_csv('../weighted_train.csv')
#
# plt.figure()
# plt.plot(X_1, label='model')
# plt.plot(X_2, label='gp')
# plt.plot(weightY, label='weighted')
# plt.plot(Y_true[:498, :], label='true')
# plt.legend()
#
# m_e = data_m[:498, 16].reshape(-1, 1)
# d_e = data_gp[:, 7].reshape(-1, 1)
# w_e = weightY - Y_true[:498, :]
#
# plt.figure()
# plt.boxplot([m_e.flatten(), d_e.flatten(), w_e.flatten()], labels=['physics', 'data', 'average'], showmeans=True)
#
# # Predict on test dataset-----------------------------------------------------------------------------------
# # test_gp = pd.read_csv('../err_gp_xy_test.csv').values
# # testX_2 = test_gp[:, 4].reshape(-1, 1)
# # testX_3 = test_gp[:, 10].reshape(-1, 1)
# # testX_1 = data_m[571:800, 13].reshape(-1, 1)
#
# # testX_4 = np.sqrt((testX_1 - testX_2) ** 2)
# # testY_true = data_m[571:800, 10].reshape(-1, 1)
# # Xtest = np.hstack([testX_1, testX_2, testX_3])
#
# testY_pre = model.predict(Xtest)
# plt.figure()
# plt.plot(testY_pre)
#
# testwp = testY_pre[:].reshape(-1, 1)
# testwd = 1-testwp
#
# weightYtest = testwp * testX_1 + testwd * testX_2
#
# plt.figure()
# plt.plot(testX_1, label='model')
# plt.plot(testX_2, label='gp')
# plt.plot(weightYtest, label='weighted')
# plt.plot(testY_true, label='true')
# plt.legend()
#
# testm_e = data_m[571:800, 16].reshape(-1, 1)
# testd_e = test_gp[:, 7].reshape(-1, 1)
# testw_e = weightYtest - testY_true
#
# plt.figure()
# plt.boxplot([testm_e.flatten(), testd_e.flatten(), testw_e.flatten()], labels=['physics', 'data', 'average'],
#             showmeans=True)
#
# plt.show()
