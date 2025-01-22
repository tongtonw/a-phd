from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
import pandas as pd


class Train():

    def model(self):
        kernel = 2.0 ** 2 * RBF(length_scale=100.0) + WhiteKernel()
        mlp = make_pipeline(
            StandardScaler(),
            # MLPRegressor(hidden_layer_sizes=(12, 24, 12), max_iter=10000)
            GaussianProcessRegressor(kernel=kernel)
        )
        return mlp

    def train(self, x_train, y_train):
        gp = self.model()

        gp.fit(x_train, y_train)
        # print(mlp.score(x_test, y_test))
        return gp

    def pre(self, X, yu, yv, yr, x_test, save, file_name):
        # test_size = 0.2
        # dof = ['u', 'v', 'r']
        # X, y = shuffle(input_d, onput_d, random_state=0)

        gpu = self.train(X, yu)

        gpv = self.train(X, yv)
        gpr = self.train(X, yr)

        sol_mean = []

        for j in range(len(x_test) - 1):
            mean_y_pred, std_y_pred = gpu.predict(x_test[j].reshape(1, -1), return_std=True)
            mean_y_pred_v, std_y_pred_v = gpv.predict(x_test[j].reshape(1, -1), return_std=True)
            mean_y_pred_r, std_y_pred_r = gpr.predict(x_test[j].reshape(1, -1), return_std=True)

            delta = x_test[j + 1] - x_test[j]
            # sol_meanDict = {'delta_u': delta[0],
            #                 'delta_v': delta[1],
            #                 'delta_r': delta[2],
            #                 'u_hat': mean_y_pred[0],
            #                 'v_hat': mean_y_pred_v[0],
            #                 'r_hat': mean_y_pred_r[0],
            #
            #                 'error_u': delta[0] - mean_y_pred[0],
            #                 'error_v': delta[1] - mean_y_pred_v[0],
            #                 'error_r': delta[2] - mean_y_pred_r[0],
            #                 'std_u': std_y_pred[0],
            #                 'std_v': std_y_pred_v[0],
            #                 'std_r': std_y_pred_r[0]
            #                 }
            sol_meanDict = {'delta_n': delta[0],
                            'delta_e': delta[1],
                            'delta_psi': delta[2],
                            'n_hat': mean_y_pred[0],
                            'e_hat': mean_y_pred_v[0],
                            'psi_hat': mean_y_pred_r[0],

                            'error_n': delta[0] - mean_y_pred[0],
                            'error_e': delta[1] - mean_y_pred_v[0],
                            'error_psi': delta[2] - mean_y_pred_r[0],
                            'std_n': std_y_pred[0],
                            'std_e': std_y_pred_v[0],
                            'std_psi': std_y_pred_r[0]
                            }
            sol_mean.append(sol_meanDict)

        mean = pd.DataFrame(sol_mean)

        if save:
            # joblib.dump(gpu, '../gp_u.pkl')
            # joblib.dump(gpv, '../gp_v.pkl')
            # joblib.dump(gpr, '../gp_r.pkl')
            mean.to_csv('../{}.csv'.format(file_name))

        return mean

    # def pre_mul(self, X, yu, yv, yr, horizon, x_test):
    #     gpu = self.train(X, yu)
    #
    #     gpv = self.train(X, yv)
    #     gpr = self.train(X, yr)
    #
    #     # horizon = 20
    #     step_for = 1
    #
    #     sol_mean = []
    #     sol_std = []
    #
    #     while step_for <= horizon:
    #         mean_y_pred, std_y_pred = gpu.predict(x_test, return_std=True)
    #         mean_y_pred_v, std_y_pred_v = gpv.predict(x_test, return_std=True)
    #         mean_y_pred_r, std_y_pred_r = gpr.predict(x_test, return_std=True)
    #
    #         x_test[0, 0] = mean_y_pred + x_test[0, 0]
    #
    #         x_test[0, 1] = mean_y_pred_v + x_test[0, 1]
    #         x_test[0, 2] = mean_y_pred_r + x_test[0, 2]
    #
    #         sol_meanDict = {'u': mean_y_pred[0], 'v': mean_y_pred_v[0], 'r': mean_y_pred_r[0]}
    #         sol_stdDict = {'u': std_y_pred[0], 'v': std_y_pred_v[0], 'r': std_y_pred_r[0]}
    #         sol_mean.append(sol_meanDict)
    #         sol_std.append(sol_stdDict)
    #         # sol_mean.append(mean_y_pred[0])
    #         # sol_std.append(std_y_pred[0])
    #         step_for += 1
    #
    #         mean = pd.DataFrame(sol_mean)
    #         std = pd.DataFrame(sol_std)
    #
    #         xout = np.zeros([horizon, 3])
    #         xout[0, 0] = x_test[0, 0]
    #         xout[0, 1] = x_test[0, 1]
    #         xout[0, 2] = x_test[0, 2]
    #         for i in range(len(mean) - 1):
    #             xout[i + 1, 0] = xout[i, 0] + mean.iloc[i, 0]
    #             xout[i + 1, 1] = xout[i, 1] + mean.iloc[i, 1]
    #             xout[i + 1, 2] = xout[i, 2] + mean.iloc[i, 2]
    #
    #         fig1 = plt.figure(1)
    #         plt.plot(range(30), X[500:500 + 30, 0], color="black", linestyle="dashed",
    #                  label="Measurements")
    #         plt.plot(range(30), xout[:, 0], color="tab:blue", alpha=0.4, label="Gaussian process")
    #         plt.fill_between(
    #             range(30),
    #             xout[:, 0] - std['u'],
    #             xout[:, 0] + std['u'],
    #             color="tab:blue",
    #             alpha=0.2,
    #         )
    #         plt.legend()
    #         plt.xlabel("index")
    #         plt.ylabel("u")
    #
    #         fig2 = plt.figure(2)
    #         plt.plot(range(30), X[500:500 + 30, 1], color="black", linestyle="dashed",
    #                  label="Measurements")
    #         plt.plot(range(30), xout[:, 1], color="tab:blue", alpha=0.4, label="Gaussian process")
    #         plt.fill_between(
    #             range(30),
    #             xout[:, 1] - std['v'],
    #             xout[:, 1] + std['v'],
    #             color="tab:blue",
    #             alpha=0.2,
    #         )
    #         plt.legend()
    #         plt.xlabel("index")
    #         plt.ylabel("v")
    #
    #         fig3 = plt.figure(3)
    #         plt.plot(range(30), X[500:500 + 30, 2], color="black", linestyle="dashed",
    #                  label="Measurements")
    #         plt.plot(range(30), xout[:, 2], color="tab:blue", alpha=0.4, label="Gaussian process")
    #         plt.fill_between(
    #             range(30),
    #             xout[:, 2] - std['r'],
    #             xout[:, 2] + std['r'],
    #             color="tab:blue",
    #             alpha=0.2,
    #         )
    #         plt.legend()
    #         plt.xlabel("index")
    #         plt.ylabel("r")


if __name__ == '__main__':
    gppre = Train()

    file = '../zigzag_data.csv'
    Dz = pd.read_csv(file)

    feature_in = Dz.iloc[:, [10, 11, 19, 15, 16, 23, 1, 2, 3, 4, 25, 26]].values
    knottoms = 0.514

    input_d = feature_in * [1, 1, np.pi / 180, knottoms, knottoms, np.pi / 180, 1, 1 / 100 * 203, 1, 1 / 100 * 203, 1, knottoms]

    plt.figure(4)
    plt.subplot(1, 1, 1)
    plt.plot(feature_in[:, 0], feature_in[:, 1])

    # plt.subplot(5, 1, 3)
    # plt.plot(feature_in[:, 2])
    #
    # plt.subplot(5, 1, 4)
    # plt.plot(feature_in[:, 3], label='azi pt')
    # plt.plot(feature_in[:, 5], label='azi sb')
    # plt.legend()
    # plt.subplot(5, 1, 5)
    # plt.plot(feature_in[:, 4], label='rpm pt')
    # plt.plot(feature_in[:, 6], label='rpm sb')
    # plt.legend()

    onput_d = input_d[1:500, 0] - input_d[:499, 0]
    onput_dv = input_d[1:500, 1] - input_d[:499, 1]
    onput_dr = input_d[1:500, 2] - input_d[:499, 2]

    loc = [570, 800] # 20/20, 60%
    # loc = [850, 1000] # 30/30, 60%
    # loc = [1791,1951]
    # loc = [0, 499]
    x_test = input_d[loc[0]:loc[1], :]
    x_train = input_d[:499, :]
    # mean = gppre.pre(x_train, onput_d, onput_dv, onput_dr, x_test, True, 'err_gp_xy_test')

    sol_m = pd.read_csv('../error_m_ne.csv').values

    mean = pd.read_csv('../err_gp_xy_test.csv')

    fig1 = plt.figure(1)
    plt.plot(range(len(x_test)-1), x_test[1:, 0], color="black", linestyle="dashed",
             label="Measurements")
    plt.plot(range(len(x_test)-1), x_test[:-1, 0] + mean['n_hat'], color="tab:blue", alpha=0.4, label="Gaussian process")
    plt.fill_between(
        range(len(x_test)-1),
        x_test[:-1, 0] + mean['n_hat'] - mean['std_n'],
        x_test[:-1, 0] + mean['n_hat'] + mean['std_n'],
        color="tab:blue",
        alpha=0.2,
    )
    plt.plot(range(len(x_test)-1), x_test[:-1, 0] + sol_m[loc[0]:loc[1]-1, 13],  color="tab:red", alpha=0.4, label="Model projections")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("n")

    fig1.savefig('../figures/gpm-n-test.png',
                bbox_inches='tight', dpi=800)

    fig2 = plt.figure(2)
    plt.plot(range(len(x_test) - 1), x_test[1:, 1], color="black", linestyle="dashed",
             label="Measurements")
    plt.plot(range(len(x_test) - 1), x_test[:-1, 1] + mean['e_hat'], color="tab:blue", alpha=0.4,
             label="Gaussian process")
    plt.fill_between(
        range(len(x_test) - 1),
        x_test[:-1, 1] + mean['e_hat'] - mean['std_e'],
        x_test[:-1, 1] + mean['e_hat'] + mean['std_e'],
        color="tab:blue",
        alpha=0.2,
    )
    plt.plot(range(len(x_test) - 1), x_test[:-1, 1] + sol_m[loc[0]:loc[1] - 1, 14], color="tab:red", alpha=0.4,
             label="Model projections")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("e")

    fig2.savefig('../figures/gpm-e-test.png',
                bbox_inches='tight', dpi=800)

    fig3 = plt.figure(3)
    plt.plot(range(len(x_test) - 1), x_test[1:, 2], color="black", linestyle="dashed",
             label="Measurements")
    plt.plot(range(len(x_test) - 1), x_test[:-1, 2] + mean['psi_hat'], color="tab:blue", alpha=0.4,
             label="Gaussian process")
    plt.fill_between(
        range(len(x_test) - 1),
        x_test[:-1, 2] + mean['psi_hat'] - mean['std_psi'],
        x_test[:-1, 2] + mean['psi_hat'] + mean['std_psi'],
        color="tab:blue",
        alpha=0.2,
    )
    plt.plot(range(len(x_test) - 1), x_test[:-1, 2] + sol_m[loc[0]:loc[1] - 1, 15], color="tab:red", alpha=0.4,
             label="Model projections")
    # plt.plot(range(len(x_test) - 1), x_test[:-1, 2] + sol_m[loc[0]:loc[1] - 1, 3], color="tab:green", alpha=0.4,
    #          label="Truth")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("psi")

    fig3.savefig('../figures/gpm-psi-test.png',
                bbox_inches='tight', dpi=800)

    plt.figure(5)
    plt.subplot(3, 1, 1)
    plt.plot(mean['n_hat'], mean['error_n'], '.')
    plt.xlabel('$\Delta$ uhat')
    plt.ylabel('$\Delta$ u - $\Delta$ uhat')
    plt.subplot(3, 1, 2)
    plt.plot(mean['e_hat'], mean['error_e'], '.')
    plt.xlabel('$\Delta$ vhat')
    plt.ylabel('e')
    plt.subplot(3, 1, 3)
    plt.plot(mean['psi_hat'], mean['error_psi'], '.')
    plt.xlabel('$\Delta$ rhat')
    plt.ylabel('e')
    plt.tight_layout()

plt.show()
