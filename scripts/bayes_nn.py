import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from run_model import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D, Bidirectional, LSTM, Embedding, GlobalMaxPooling1D, Input, \
    BatchNormalization
from tensorflow.keras.models import Sequential

import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers

# print(' TF Version:', tf.__version__, '\n',  # 2.7.0
#       'TFP Version:', tfp.__version__)  # 0.15.0

# if tf.test.gpu_device_name() != '/device:GPU:0':
#     print('WARNING: GPU device not found.')
# else:
#     print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))


@tf.function
def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


hidden_units = [24, 12]
learning_rate = 0.01
num_epochs = 10


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = Sequential(
        [
            tfpl.DistributionLambda(
                lambda t: tfd.Laplace(loc=tf.zeros(n), scale=2 * tf.ones(n))
            # tfpl.DistributionLambda(
            #     lambda t: tfd.MultivariateNormalDiag(
            #         loc=tf.zeros(n), scale_diag=tf.ones(n))

            )
        ]
    )
    return prior_model

def spike_and_slab(event_shape, dtype):
    distribution = tfd.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype),
                scale=1.0*tf.ones(event_shape, dtype=dtype)),
                            reinterpreted_batch_ndims=1),
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype),
                scale=10.0*tf.ones(event_shape, dtype=dtype)),
                            reinterpreted_batch_ndims=1)],
    name='spike_and_slab')
    return distribution
def get_prior(kernel_size, bias_size, dtype=None):
    """
    This function should create the prior distribution, consisting of the
    "spike and slab" distribution that is described above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the prior distribution.
    """
    n = kernel_size+bias_size
    prior_model = Sequential([tfpl.DistributionLambda(lambda t : spike_and_slab(n, dtype))])
    return prior_model
def get_posterior(kernel_size, bias_size, dtype=None):
    """
    This function should create the posterior distribution as specified above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the posterior distribution.
    """
    n = kernel_size + bias_size
    return Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n)
    ])
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfpl.VariableLayer(
                2 * n, dtype=dtype
                # tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            # tfpl.MultivariateNormalTriL(n),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(

                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + 0.1 * tf.nn.softplus(t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ]
    )
    return posterior_model


def prob_ml_epistemic(train_size):
    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    inputs = Input(shape=(9,), dtype=tf.float64)
    # features = BatchNormalization()(inputs)
    features = tfpl.DenseVariational(units=hidden_units[0],
                                      make_prior_fn=get_prior,
                                      make_posterior_fn=get_posterior,
                                      kl_weight=1 / train_size, activation="tanh")(inputs)
    features = tfpl.DenseVariational(units=hidden_units[1],
                                     make_prior_fn=get_prior,
                                     make_posterior_fn=get_posterior,
                                     kl_weight=1 / train_size, activation="tanh")(features)

    distribution_params = Dense(units=2)(features)
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

    # event_shape = 1
    # model = Sequential([
    #     Dense(32, activation=tf.nn.tanh),
    #     Dense(units=tfpl.IndependentNormal.params_size(event_shape)),
    #     tfpl.IndependentNormal(event_shape)
    # ])
    # model.compile(loss=nll, optimizer='adam')
    # model.fit(x, y, epochs=300, verbose=0)
    #
    # print('Loss:', str(model.evaluate(x, y, verbose=False)))
    # plot_2sd_data(model, x, y)
    # plot_2sd_data(model, x_test, y_test)


def run_experiment(model, loss, x_train, y_train, x_test, y_test):
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test))
    print("Model training finished.")
    _, rmse = model.evaluate(x_train, y_train, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")


def plot_2sd_data(model, x, y):
    model_distribution = model(x)
    model_sample = model_distribution.sample()
    model_means = model_distribution.mean()
    print('Model mean:', tf.reduce_mean(model_means).numpy())
    print('Mean of the data:', y.mean())

    model_std = model_distribution.stddev()

    y_m2sd = model_means - 2 * model_std
    y_p2sd = model_means + 2 * model_std

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=True, dpi=128)
    xid = range(len(y))
    ax1.scatter(xid, y, alpha=0.4, label='Data')
    ax1.scatter(xid, model_sample, alpha=0.4, color='black', label='Model Samples')
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

    x_test = input_d[:-1, 3:]
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
    x_train, train_x, train_y, train_z = TrainD[:, 3:-3], TrainD[:, -3], TrainD[:, -2], TrainD[:, -1]
    x_test, y_test = gen_test_data(n_path)

    # hidden_units = [8, 8]
    # learning_rate = 0.01
    # num_epochs = 1000
    prob_bnn_model = prob_ml_epistemic(x_train.shape[0])

    run_experiment(prob_bnn_model, nll, x_train, train_x, x_test, y_test)
    print('train done.')
    # ----- predict--------------------
    prediction_distribution = prob_bnn_model(x_train)
    prediction_mean = prediction_distribution.mean().numpy().tolist()
    prediction_stdv = prediction_distribution.stddev().numpy()

    # The 95% CI is computed as mean ± (1.96 * stdv)
    upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
    lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
    prediction_stdv = prediction_stdv.tolist()

    sample = 10
    for idx in range(sample):
        print(
            f"Prediction mean: {round(prediction_mean[idx][0], 2)}, "
            f"stddev: {round(prediction_stdv[idx][0], 2)}, "
            f"95% CI: [{round(upper[idx][0], 2)} - {round(lower[idx][0], 2)}]"
            f" - Actual: {train_x[idx]}"
        )

    fig, ax2 = plt.subplots(1, 1)
    xid = range(len(train_x))
    # ax1.scatter(xid, y_test, alpha=0.4, label='Data')
    # ax1.scatter(xid, model_sample, alpha=0.4, color='black', label='Model Samples')
    # ax1.legend()

    ax2.scatter(xid, train_x, s=5, alpha=0.4, label='Data')
    ax2.plot(xid, prediction_mean, color='black', alpha=0.8, label='model $\mu$')
    ax2.plot(xid, lower, color='green', alpha=0.8, label='model $\mu \pm 2 \sigma$',
             linewidth=1)
    ax2.plot(xid, upper, color='green', alpha=0.8, linewidth=1)
    ax2.legend()

    plt.show()
    # prob_ml(x_train, train_x, x_test, y_test)

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
