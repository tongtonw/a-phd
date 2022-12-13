import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

file = '../zigzag_data.csv'
Dz = pd.read_csv(file)

feature_in = Dz.iloc[:, [15, 16, 23, 1, 2, 3, 4, 25, 26]].values
knottoms = 0.514

input_d = feature_in * [knottoms, knottoms, np.pi / 180, 1, 1 / 100 * 203, 1, 1 / 100 * 203, 1, knottoms]

# plt.figure(4)
# plt.subplot(5, 1, 1)
# plt.plot(feature_in[:, 0])
# plt.subplot(5, 1, 2)
# plt.plot(feature_in[:, 1])
#
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
#
# sol_m = pd.read_csv('../error_m.csv')
loc = [570, 800]

plt.figure()
plt.subplot(2, 1, 1)
plt.hist(input_d[:499, 3])
plt.hist(input_d[loc[0]:loc[1], 3])

plt.subplot(2, 1, 2)
plt.hist(input_d[:499, 4])
plt.hist(input_d[loc[0]:loc[1], 4])

plt.show()
