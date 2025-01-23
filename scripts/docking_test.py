import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from run_model import *
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D, Bidirectional, LSTM, Embedding, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


# print(' TF Version:', tf.__version__, '\n',  # 2.7.0
#       'TFP Version:', tfp.__version__)  # 0.15.0

# if tf.test.gpu_device_name() != '/device:GPU:0':
#   print('WARNING: GPU device not found.')
# else:
#   print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

@tf.function
def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


def BayesRegress(x, y, x_test, y_test):
    # std_scaler = StandardScaler()
    # x_norm = std_scaler.fit_transform(x)
    # y_norm = std_scaler.fit_transform(y.reshape(-1, 1))
    #
    # print(std_scaler.mean_)
    reg = make_pipeline(
        StandardScaler(),
        BayesianRidge(tol=1e-6, compute_score=True, n_iter=5000))

    # reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
    init = [1 / np.var(y), 1.0]
    # init = [1 / np.var(y), 1.0]  # Default values
    reg[1].set_params(alpha_init=init[0], lambda_init=init[1])

    reg.fit(x, y)
    ymean, ystd = reg.predict(x_test, return_std=True)
    # ymean = std_scaler.inverse_transform(ymean.reshape(-1, 1))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    xid = range(len(y_test))
    ax.scatter(xid, y_test, s=5, alpha=0.5, label="observation")
    ax.plot(xid, ymean, color="red", label="predict mean")
    ax.fill_between(
        xid, ymean.ravel() - ystd, ymean.ravel() + ystd, color="pink", alpha=0.5, label="predict std"
    )
    text = "$\\alpha={:.1f}$\n$\\lambda={:.3f}$\n$L={:.1f}$".format(
        reg[1].alpha_, reg[1].lambda_, reg[1].scores_[-1]
    )
    ax.text(0.05, 0.6, text, fontsize=12)
    plt.tight_layout()
    plt.show()


def prob_ml_aleatoric(x, y, x_test, y_test):
    event_shape = 1
    model = Sequential([
        Dense(32, activation="sigmoid"),
        Dense(16, activation="sigmoid"),
        Dense(units=tfpl.IndependentNormal.params_size(event_shape)),
        tfpl.IndependentNormal(event_shape)
    ])
    model.compile(loss=nll, optimizer='adam')
    model.fit(x, y, epochs=100, verbose=0)

    print('Loss:', str(model.evaluate(x, y, verbose=False)))
    plot_2sd_data(model, x, y)
    plot_2sd_data(model, x_test, y_test)


def plot_2sd_data(model, x, y):
    model_distribution = model(x)
    model_sample = model_distribution.sample()
    model_means = model_distribution.mean()
    print('Model mean:', tf.reduce_mean(model_means).numpy())
    print('Mean of the data:', y.mean())

    model_std = model_distribution.stddev()

    y_m2sd = model_means - 2 * model_std
    y_p2sd = model_means + 2 * model_std

    fig, (ax1, ax2) = plt.subplots(1, 2)
    xid = range(len(y))
    ax1.scatter(xid, y, alpha=0.4, label='Data')
    ax1.scatter(xid, model_sample, s=5, alpha=0.4, color='black', label='Model Samples')
    ax1.legend()

    ax2.scatter(xid, y, alpha=0.4, label='Data')
    ax2.plot(xid, model_means, color='black', alpha=0.8, label='model $\mu$')
    ax2.plot(xid, y_m2sd, color='green', alpha=0.8, label='model $\mu \pm 2 \sigma$',
             linewidth=2)
    ax2.plot(xid, y_p2sd, color='green', alpha=0.8, linewidth=2)
    ax2.legend()

    plt.show()


def gen_training_data(n_path):
    x_train = np.array([]).reshape(-1, 12)
    train_x = np.array([])
    train_y = np.array([])
    train_z = np.array([])

    for i_path in range(n_path):
        file = '../00 Non-elongated Gunnerus Docking Data/Data/RealData_extracted/' \
               'Scenario{}.csv'.format(i_path)
        print('read scenario {}.'.format(i_path))
        Dz = pd.read_csv(file)

        feature_in = Dz.iloc[:2000, [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 12, 13]].values
        knottoms = 0.514

        input_d = feature_in * [1, 1, np.pi / 180, 1, 1, np.pi / 180, 1, 1 / 100 * 203, 1, 1 / 100 * 203, 1,
                                1]

        X = input_d[:-1, :]
        onput_d = input_d[1:, 0] - input_d[:-1, 0]
        onput_dv = input_d[1:, 1] - input_d[:-1, 1]
        onput_dr = input_d[1:, 2] - input_d[:-1, 2]

        x_train = np.concatenate((x_train, X), axis=0)
        train_x = np.concatenate((train_x, onput_d), axis=0)
        train_y = np.concatenate((train_y, onput_dv), axis=0)
        train_z = np.concatenate((train_z, onput_dr), axis=0)
    all = np.concatenate((x_train, train_x.reshape(-1, 1), train_y.reshape(-1, 1), train_z.reshape(-1, 1)), axis=1)
    pd.DataFrame(all).to_csv('../ml_xy_TrainData_docks{}.csv'.format(n_path), index=False)
    return True


def gen_test_data(n_path):
    file = '../00 Non-elongated Gunnerus Docking Data/Data/RealData_extracted/' \
           '/Scenario{}.csv'.format(n_path)
    Dz = pd.read_csv(file)

    feature_in = Dz.iloc[:2000, [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 12, 13]].values

    input_d = feature_in * [1, 1, np.pi / 180, 1, 1, np.pi / 180, 1, 1 / 100 * 203, 1, 1 / 100 * 203, 1,
                            1]

    x_test = input_d[:-1, :]
    y_test = input_d[1:, 0] - input_d[:-1, 0]
    return x_test, y_test


def model(n_path):
    # ------------------------- visulize model errors--------------------
    file = '../00 Non-elongated Gunnerus Docking Data/Data/RealData_extracted/' \
           '/Scenario{}.csv'.format(n_path)
    Dz = pd.read_csv(file)

    feature_in = Dz.iloc[:, [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 12, 13]].values
    knottoms = 0.514

    input_d = feature_in * [1, 1, np.pi / 180, 1, 1, np.pi / 180, 1, 1 / 100 * 203, 1, 1 / 100 * 203, 1,
                            1]

    x_test = input_d[:-1, :]
    err_m = error_gen(x_test, 'error_m_docking{}'.format(n_path))
    # err_m = pd.read_csv('../error_m_docking{}.csv'.format(n_path))
                         
    e_n = err_m.iloc[:, 16]
    std_n = 0.1

    plt.figure()
    plt.scatter(err_m.iloc[:, 10], err_m.iloc[:, 13], s=1, c='r')
    left, right = plt.xlim()
    plt.plot(np.arange(left - 0.2, right + 0.2), np.arange(left - 0.2, right + 0.2), 'b')
    plt.fill_between(np.linspace(left, right, len(e_n)), np.linspace(left, right, len(e_n)) - std_n,
                     np.linspace(left, right, len(e_n)) + std_n, color="tab:blue",
                     alpha=0.2)
    plt.xlabel('Observations $\Delta x$')
    plt.ylabel('Predictions $\Delta \hat{x}$')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    n_path = 2
    # gen_training_data(n_path)
    # print('training data generated.')
    TrainD = pd.read_csv('../ml_xy_TrainData_docks{}.csv'.format(n_path)).values
    x_train, train_x, train_y, train_z = TrainD[:, :-3], TrainD[:, -3], TrainD[:, -2], TrainD[:, -1]
    x_test, y_test = gen_test_data(n_path)
    prob_ml_aleatoric(x_train, train_x, x_test, y_test)
    # BayesRegress(x_train, train_x, x_test, y_test)
    print('train done.')

# while ml:
#     # ----------------------- train ML model -----------------------------
#     x_train = np.array([]).reshape(-1, 12)
#     train_x = np.array([])
#     train_y = np.array([])
#     train_z = np.array([])
#
#     file = '../00 Non-elongated Gunnerus Docking Data/Data/RealData_extracted/' \
#            '/Scenario{}.csv'.format(n_path)
#     Dz = pd.read_csv(file)
#
#     feature_in = Dz.iloc[:, [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 12, 13]].values
#     knottoms = 0.514
#
#     input_d = feature_in * [1, 1, np.pi / 180, 1, 1, np.pi / 180, 1, 1 / 100 * 203, 1, 1 / 100 * 203, 1,
#                             1]
#
#     x_test = input_d[:-1, :]
#
#     # gppre = Train()
#     # mean = gppre.pre(x_train, train_x, train_y, train_z, x_test, True, 'err_gp_xy_test_docks{}'.format(n_path))
#
#     fig1 = plt.figure(1)
#     plt.plot(range(len(x_test) - 1), x_test[1:, 0], color="black", linestyle="dashed",
#              label="Measurements")
#     plt.plot(range(len(x_test) - 1), x_test[:-1, 0] + mean['n_hat'], color="tab:blue", alpha=0.4,
#              label="Gaussian process")
#     plt.fill_between(
#         range(len(x_test) - 1),
#         x_test[:-1, 0] + mean['n_hat'] - mean['std_n'],
#         x_test[:-1, 0] + mean['n_hat'] + mean['std_n'],
#         color="tab:blue",
#         alpha=0.2,
#     )
#     # plt.plot(range(len(x_test)-1), x_test[:-1, 0] + sol_m[loc[0]:loc[1]-1, 13],  color="tab:red", alpha=0.4, label="Model projections")
#     plt.legend()
#     plt.xlabel("time [s]")
#     plt.ylabel("n")
