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

    def pre(self, X, yu, yv, yr, x_test):
        # test_size = 0.2
        # dof = ['u', 'v', 'r']
        # X, y = shuffle(input_d, onput_d, random_state=0)

        gpu = self.train(X, yu)

        gpv = self.train(X, yv)
        gpr = self.train(X, yr)

        # save model
        joblib.dump(gpu, '../gp_u.pkl')
        joblib.dump(gpv, '../gp_v.pkl')
        joblib.dump(gpr, '../gp_r.pkl')

        sol_mean = []
        # x_test = X[500, :].reshape(1, -1)
        for j in range(len(x_test) - 1):
            mean_y_pred, std_y_pred = gpu.predict(x_test[j].reshape(1, -1), return_std=True)
            mean_y_pred_v, std_y_pred_v = gpv.predict(x_test[j].reshape(1, -1), return_std=True)
            mean_y_pred_r, std_y_pred_r = gpr.predict(x_test[j].reshape(1, -1), return_std=True)

            delta = x_test[j + 1] - x_test[j]
            sol_meanDict = {'delta_u': delta[0],
                            'delta_v': delta[1],
                            'delta_r': delta[2],
                            'u_hat': mean_y_pred[0],
                            'v_hat': mean_y_pred_v[0],
                            'r_hat': mean_y_pred_r[0],

                            'error_u': delta[0] - mean_y_pred[0],
                            'error_v': delta[1] - mean_y_pred_v[0],
                            'error_r': delta[2] - mean_y_pred_r[0]
                            }
            sol_mean.append(sol_meanDict)

        mean = pd.DataFrame(sol_mean)
        mean.to_csv('../error_gp.csv')
        return mean

    def pre_mul(self, X, yu, yv, yr, horizon, x_test):
        gpu = self.train(X, yu)

        gpv = self.train(X, yv)
        gpr = self.train(X, yr)

        # horizon = 20
        step_for = 1

        sol_mean = []
        sol_std = []

        while step_for <= horizon:
            mean_y_pred, std_y_pred = gpu.predict(x_test, return_std=True)
            mean_y_pred_v, std_y_pred_v = gpv.predict(x_test, return_std=True)
            mean_y_pred_r, std_y_pred_r = gpr.predict(x_test, return_std=True)

            x_test[0, 0] = mean_y_pred + x_test[0, 0]

            x_test[0, 1] = mean_y_pred_v + x_test[0, 1]
            x_test[0, 2] = mean_y_pred_r + x_test[0, 2]

            sol_meanDict = {'u': mean_y_pred[0], 'v': mean_y_pred_v[0], 'r': mean_y_pred_r[0]}
            sol_stdDict = {'u': std_y_pred[0], 'v': std_y_pred_v[0], 'r': std_y_pred_r[0]}
            sol_mean.append(sol_meanDict)
            sol_std.append(sol_stdDict)
            # sol_mean.append(mean_y_pred[0])
            # sol_std.append(std_y_pred[0])
            step_for += 1

            mean = pd.DataFrame(sol_mean)
            std = pd.DataFrame(sol_std)

            xout = np.zeros([horizon, 3])
            xout[0, 0] = x_test[0, 0]
            xout[0, 1] = x_test[0, 1]
            xout[0, 2] = x_test[0, 2]
            for i in range(len(mean) - 1):
                xout[i + 1, 0] = xout[i, 0] + mean.iloc[i, 0]
                xout[i + 1, 1] = xout[i, 1] + mean.iloc[i, 1]
                xout[i + 1, 2] = xout[i, 2] + mean.iloc[i, 2]

            fig1 = plt.figure(1)
            plt.plot(range(30), X[500:500 + 30, 0], color="black", linestyle="dashed",
                     label="Measurements")
            plt.plot(range(30), xout[:, 0], color="tab:blue", alpha=0.4, label="Gaussian process")
            plt.fill_between(
                range(30),
                xout[:, 0] - std['u'],
                xout[:, 0] + std['u'],
                color="tab:blue",
                alpha=0.2,
            )
            plt.legend()
            plt.xlabel("index")
            plt.ylabel("u")

            fig2 = plt.figure(2)
            plt.plot(range(30), X[500:500 + 30, 1], color="black", linestyle="dashed",
                     label="Measurements")
            plt.plot(range(30), xout[:, 1], color="tab:blue", alpha=0.4, label="Gaussian process")
            plt.fill_between(
                range(30),
                xout[:, 1] - std['v'],
                xout[:, 1] + std['v'],
                color="tab:blue",
                alpha=0.2,
            )
            plt.legend()
            plt.xlabel("index")
            plt.ylabel("v")

            fig3 = plt.figure(3)
            plt.plot(range(30), X[500:500 + 30, 2], color="black", linestyle="dashed",
                     label="Measurements")
            plt.plot(range(30), xout[:, 2], color="tab:blue", alpha=0.4, label="Gaussian process")
            plt.fill_between(
                range(30),
                xout[:, 2] - std['r'],
                xout[:, 2] + std['r'],
                color="tab:blue",
                alpha=0.2,
            )
            plt.legend()
            plt.xlabel("index")
            plt.ylabel("r")


if __name__ == '__main__':
    gppre = Train()

    file = '../zigzag_data.csv'
    Dz = pd.read_csv(file)

    feature_in = Dz.iloc[:, [15, 16, 23, 1, 2, 3, 4, 25, 26]].values
    knottoms = 0.514

    input_d = feature_in * [knottoms, knottoms, np.pi / 180, 1, 1 / 100 * 203, 1, 1 / 100 * 203, 1, knottoms]

    onput_d = input_d[1:-500, 0] - input_d[:-501, 0]
    onput_dv = input_d[1:-500, 1] - input_d[:-501, 1]
    onput_dr = input_d[1:-500, 2] - input_d[:-501, 2]

    x_test = input_d[-500:, :]
    mean = gppre.pre(input_d[:-501, :], onput_d, onput_dv, onput_dr, x_test)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(mean['u_hat'], mean['error_u'], '.')
    plt.xlabel('$\Delta$ uhat')
    plt.ylabel('$\Delta$ u - $\Delta$ uhat')
    plt.subplot(3, 1, 2)
    plt.plot(mean['v_hat'], mean['error_v'], '.')
    plt.xlabel('$\Delta$ vhat')
    plt.ylabel('e')
    plt.subplot(3, 1, 3)
    plt.plot(mean['r_hat'], mean['error_r'], '.')
    plt.xlabel('$\Delta$ rhat')
    plt.ylabel('e')
    plt.tight_layout()
# save model
# joblib.dump(mlp, './m_env_model_gp_%s.pkl' % idof)

plt.show()
