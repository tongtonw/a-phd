from run_model import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = '../Trajectory1.csv'
Dz = pd.read_csv(file)
#
# feature_in = Dz.iloc[:, [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 12, 13]].values
# knottoms = 0.514
#
# input_d = feature_in * [1, 1, np.pi / 180, knottoms, knottoms, np.pi / 180, 1, 1 / 100 * 203, 1, 1 / 100 * 203, 1,
#                         knottoms]
#
# Xtest = input_d
# err_m = error_gen(Xtest, 'error_m_daily')


err_m = pd.read_csv('../error_m_daily.csv')

plt.figure()
plt.subplot(4, 1, 1)
plt.plot(Dz.iloc[:, [3]].values)
plt.subplot(4, 1, 2)
plt.plot(Dz.iloc[:, [16]].values)
plt.plot(Dz.iloc[:, [18]].values)
plt.subplot(4, 1, 3)
plt.plot(Dz.iloc[:, [17]].values)
plt.plot(Dz.iloc[:, [19]].values)
plt.subplot(4, 1, 4)
plt.plot(err_m['error_u'], label='error')
plt.plot(err_m['delta_u'], label='$\Delta$ u')
plt.plot(err_m['u_hat'], label='$\Delta$ uhat')
plt.legend()

# Alldata = pd.concat(Dz, err_m, axis=1)
# plt.figure()
# sns.pairplot(Alldata)

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(err_m['error_u'], label='error')
plt.plot(err_m['delta_u'], label='$\Delta$ u')
plt.plot(err_m['u_hat'], label='$\Delta$ uhat')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(err_m['error_v'])
plt.subplot(3, 1, 3)
plt.plot(err_m['error_r'])

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(err_m.iloc[11000:12500, 4], err_m.iloc[11000:12500, 7], '.')
plt.xlabel('$\Delta$ uhat')
plt.ylabel('$\Delta$ u - $\Delta$ uhat')
plt.subplot(3, 1, 2)
plt.plot(err_m['v_hat'], err_m['error_v'], '.')
plt.xlabel('$\Delta$ vhat')
plt.ylabel('e')
plt.subplot(3, 1, 3)
plt.plot(err_m['r_hat'], err_m['error_r'], '.')
plt.xlabel('$\Delta$ rhat')
plt.ylabel('e')
plt.tight_layout()

plt.show()