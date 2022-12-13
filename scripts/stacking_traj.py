import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import RNN, Dense, Layer, SimpleRNN, LSTM, Input, TimeDistributed
from tensorflow.keras import Sequential, Model
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
import os
from keras.models import load_model
import matplotlib as mpl

data_m = pd.read_csv('../error_m_ne.csv').values
data_gp = pd.read_csv('../err_gp_xy_train.csv').values

X_1 = data_m[:498, 13].reshape(-1, 1)
X_2 = data_gp[:, 4].reshape(-1, 1)
X_3 = data_gp[:, 10].reshape(-1, 1)
Y_true = data_m[:498, 10].reshape(-1, 1)
X = np.hstack([X_1, X_2, X_3])

dw = pd.read_csv('../weighted_train.csv').values
w = dw[:, 1]
yw = dw[:, 2].reshape(-1, 1)

x1 = 0
x = []
for j in range(len(w)):
    x1 += yw[j, 0]
    x.append(x1)

def num2color(values, cmap):
    """将数值映射为颜色"""
    norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
    cmap = mpl.cm.get_cmap(cmap)
    return [cmap(norm(val)) for val in values]

colors = num2color(w, "RdBu")

plt.figure()
plt.scatter(np.arange(len(x)), x, c=w, cmap="Blues")
plt.colorbar()
plt.show()

model = load_model('my_model.h5')
Y = np.hstack([Y_true, X])

model.predict()
print('train done')


checkpoint_path = './tmp/checkpoint'
checkpoint_dir = os.path.dirname(checkpoint_path)
model.save(checkpoint_path)


Y_pre = model.predict(X)
plt.figure()
plt.plot(Y_pre)

weightY = Y_pre * X_1 + (1 - Y_pre) * X_2

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
