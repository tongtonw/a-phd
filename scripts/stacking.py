import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import RNN, Dense, Layer, SimpleRNN, LSTM, Input, TimeDistributed
from tensorflow.keras import Sequential, Model
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
import os

data_m = pd.read_csv('../error_m_ne.csv').values
data_gp = pd.read_csv('../err_gp_xy_train.csv').values

X_1 = data_m[:498, 13].reshape(-1, 1)
X_2 = data_gp[:, 4].reshape(-1, 1)
X_3 = data_gp[:, 10].reshape(-1, 1)
Y_true = data_m[:498, 10].reshape(-1, 1)
X = np.hstack([X_1, X_2, X_3])


def baseline_model():
    n_model = 3

    inputs = Input(shape=(n_model,))
    # (batch_size, input_dim)
    dens1 = Dense(12, activation='tanh')(inputs)
    dens2 = Dense(12, activation='tanh')(dens1)
    output = Dense(1, activation='sigmoid')(dens2)

    model = Model(inputs, output)
    model.compile(loss=loss_fn, optimizer=Adam(lr=0.0005))

    return model


def loss_fn(data, y_pred):
    y_true = data[:, 0]
    i = data[:, 1]
    k = data[:, 2]
    std = data[:, 3]
    mean_ = (y_pred * i + (1 - y_pred) * k - y_true) ** 2
    prob_ = -tf.experimental.numpy.log10(
        (y_pred * tf.math.exp(-(i - y_true) ** 2 / 2)) + (1 - y_pred) * tf.math.exp(-(k - y_true) ** 2 / 2 / std ** 2))

    return prob_ + mean_


Y = np.hstack([Y_true, X])
model = baseline_model()
model.fit(X, Y, epochs=20, verbose=1)
print('train done')

#
# checkpoint_path = './tmp/checkpoint'
# checkpoint_dir = os.path.dirname(checkpoint_path)
# model.save(checkpoint_path)


Y_pre = model.predict(X)
plt.figure()
plt.plot(Y_pre)

weightY = Y_pre * X_1 + (1 - Y_pre) * X_2

# ans = np.hstack([Y_pre, weightY])
# pd.DataFrame(ans, columns=['w', 'yw']).to_csv('../weighted_train.csv')

plt.figure()
plt.plot(X_1, label='model')
plt.plot(X_2, label='gp')
plt.plot(weightY, label='weighted')
plt.plot(Y_true, label='true')
plt.legend()

m_e = data_m[:498, 16].reshape(-1, 1)
d_e = data_gp[:, 7].reshape(-1, 1)
w_e = weightY - Y_true

plt.figure()
plt.boxplot([m_e.flatten(), d_e.flatten(), w_e.flatten()], labels=['physics', 'data', 'average'], showmeans=True)

# Predict on test dataset-----------------------------------------------------------------------------------
test_gp = pd.read_csv('../err_gp_xy_test.csv').values
testX_2 = test_gp[:, 4].reshape(-1, 1)
testX_3 = test_gp[:, 10].reshape(-1, 1)
testX_1 = data_m[571:800, 13].reshape(-1, 1)
testY_true = data_m[571:800, 10].reshape(-1, 1)
Xtest = np.hstack([testX_1, testX_2, testX_3])

testY_pre = model.predict(Xtest)
plt.figure()
plt.plot(testY_pre)

weightYtest = testY_pre * testX_1 + (1 - testY_pre) * testX_2

plt.figure()
plt.plot(testX_1, label='model')
plt.plot(testX_2, label='gp')
plt.plot(weightYtest, label='weighted')
plt.plot(testY_true, label='true')
plt.legend()

testm_e = data_m[571:800, 16].reshape(-1, 1)
testd_e = test_gp[:, 7].reshape(-1, 1)
testw_e = weightYtest - testY_true

plt.figure()
plt.boxplot([testm_e.flatten(), testd_e.flatten(), testw_e.flatten()], labels=['physics', 'data', 'average'],
            showmeans=True)

plt.show()
