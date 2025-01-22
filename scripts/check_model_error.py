import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
sol_m = pd.read_csv('../error_m_ne.csv')
loc = [0, 499]

e_n = sol_m.iloc[:, 16]
std_n = 0.3
print(std_n)
plt.figure()
plt.scatter(sol_m.iloc[:, 10], sol_m.iloc[:, 13], s=1, c='r')

#
# loc = [570, 800]
# plt.figure()
# f= sns.jointplot(data=sol_m, x="delta_n", y="n_hat", s=1)
#
# # plt.scatter(sol_m.iloc[:, 10], sol_m.iloc[:, 13], s=1, c='r')
left, right = plt.xlim()
plt.plot(np.arange(left - 0.2, right + 0.2), np.arange(left - 0.2, right + 0.2), 'b')
plt.fill_between(np.linspace(left, right, len(e_n)), np.linspace(left, right, len(e_n)) - std_n, np.linspace(left, right, len(e_n)) + std_n, color="tab:blue",
        alpha=0.2)
plt.xlabel('Observations $\Delta x$')
plt.ylabel('Predictions $\Delta \hat{x}$')
plt.grid()
plt.savefig('../figures/m_n_e.png',
            bbox_inches='tight', dpi=800)
# allData = np.hstack([sol_m.iloc[loc[0]:loc[1], 13:16].values, input_d[loc[0]:loc[1], :5]])
# allData = pd.DataFrame(allData, columns=['error_n', 'error_e', 'error_psi', 'u', 'v', 'r', 'delta', 'rpm'])
# sns.kdeplot(data=allData, x='delta', y='rpm')
# plt.scatter(allData['delta'], allData['rpm'])
# allData = np.hstack([sol_m.iloc[loc[0]:loc[1], 13:14].values, input_d[loc[0]:loc[1], :5]])
# allData = pd.DataFrame(allData, columns=['error_n', 'u', 'v', 'r', 'delta', 'rpm'])
# sns.pairplot(allData, kind="kde")
plt.show()
