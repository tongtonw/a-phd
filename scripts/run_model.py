import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math_model import *
import seaborn as sns


def get_mbp(fmu, vrs, x0, y0, psi0, u0, v0, r0, rudder_cd_pt, rpm_cd_pt, rudder_cd_sb, rpm_cd_sb, global_wind_direction,
            global_wind_speed, delta_t):
    mbp = ModelPredict(fmu, vrs, x0, y0, psi0, u0, v0, r0, rudder_cd_pt, rpm_cd_pt, rudder_cd_sb, rpm_cd_sb,
                       global_wind_direction,
                       global_wind_speed,
                       delta_t=delta_t,
                       step_size=1)
    sol = mbp.simulate()
    return sol


def data_gen(fmu, vrs, ini0):
    feature_dict = []
    sol_m = get_mbp(fmu=fmu,
                    vrs=vrs,
                    x0=ini0[0, 0],
                    y0=ini0[0, 1],
                    psi0=ini0[0, 2],
                    u0=ini0[0, 3],
                    v0=ini0[0, 4],
                    r0=ini0[0, 5],
                    rudder_cd_pt=ini0[0, 6],
                    rpm_cd_pt=ini0[0, 7],
                    rudder_cd_sb=ini0[0, 8],
                    rpm_cd_sb=ini0[0, 9],
                    global_wind_direction=ini0[0, 10],
                    global_wind_speed=ini0[0, 11],
                    delta_t=1)

    featuresDict = {'delta_nhat': sol_m.iloc[0, 1],
                    'delta_ehat': sol_m.iloc[0, 2],
                    'delta_psihat': sol_m.iloc[0, 3],
                    'delta_uhat': sol_m.iloc[0, 4],
                    'delta_vhat': sol_m.iloc[0, 5],
                    'delta_rhat': sol_m.iloc[0, 6]}

    feature_dict.append(featuresDict)

    return pd.DataFrame(feature_dict)


def error_gen(input_d, Outfile):
    fmu, vrs = fmuinitialize(fmu_filename='../PMAzimuth.fmu')
    feature_dict = []

    istep = 0
    nstep = len(input_d)
    while istep < nstep - 1:
        x0 = input_d[istep, :].reshape(1, -1)
        """x0=ini0[0, 0],
            y0=ini0[0, 1],
            psi0=ini0[0, 2],
            u0=ini0[0, 3],
            v0=ini0[0, 4],
            r0=ini0[0, 5],
            rudder_cd_pt=ini0[0, 6],
            rpm_cd_pt=ini0[0, 7],
            rudder_cd_sb=ini0[0, 8],
            rpm_cd_sb=ini0[0, 9],
            global_wind_direction=ini0[0, 10],
            global_wind_speed"""
        features = data_gen(fmu, vrs, x0)
        x1 = input_d[istep + 1, :].reshape(1, -1)

        delta = x1 - x0

        errorDict = {'delta_u': delta[0, 3],
                     'delta_v': delta[0, 4],
                     'delta_r': delta[0, 5],

                     'u_hat': features.iloc[0, 3],
                     'v_hat': features.iloc[0, 4],
                     'r_hat': features.iloc[0, 5],

                     'error_u': delta[0, 3] - features.iloc[0, 3],
                     'error_v': delta[0, 4] - features.iloc[0, 4],
                     'error_r': delta[0, 5] - features.iloc[0, 5],

                     'delta_n': delta[0, 0],
                     'delta_e': delta[0, 1],
                     'delta_psi': delta[0, 2],

                     'n_hat': features.iloc[0, 0],
                     'e_hat': features.iloc[0, 1],
                     'psi_hat': features.iloc[0, 2],

                     'error_n': delta[0, 0] - features.iloc[0, 0],
                     'error_e': delta[0, 1] - features.iloc[0, 1],
                     'error_psi': delta[0, 2] - features.iloc[0, 2],

                     }
        feature_dict.append(errorDict)


        istep += 1
    pd.DataFrame(feature_dict).to_csv('../{}.csv'.format(Outfile))
    return pd.DataFrame(feature_dict)


if __name__ == '__main__':
    file = '../zigzag_data.csv'
    Dz = pd.read_csv(file)

    #
    feature_in = Dz.iloc[:, [10, 11, 19, 15, 16, 23, 1, 2, 3, 4, 25, 26]].values
    knottoms = 0.514

    input_d = feature_in * [1, 1, np.pi / 180, knottoms, knottoms, np.pi / 180, 1, 1 / 100 * 203, 1, 1 / 100 * 203, 1,
                            knottoms]

    Xtest = input_d

    err_m = error_gen(Xtest, 'error_m_ne')
    # err_m = pd.read_csv('../error_m.csv')

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(Dz.iloc[:, [2]].values)
    plt.subplot(4, 1, 2)
    plt.plot(Dz.iloc[:, [1]].values)
    plt.subplot(4, 1, 3)
    plt.plot(Dz.iloc[:, [15]].values)
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
    plt.plot(err_m['error_n'], label='error')
    plt.plot(err_m['delta_n'], label='$\Delta$ u')
    plt.plot(err_m['n_hat'], label='$\Delta$ uhat')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(err_m['error_e'])
    plt.subplot(3, 1, 3)
    plt.plot(err_m['error_psi'])

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(err_m['e_hat'], err_m['error_e'], '.')
    plt.xlabel('$\Delta$ uhat')
    plt.ylabel('$\Delta$ u - $\Delta$ uhat')
    plt.subplot(3, 1, 2)
    plt.plot(err_m['n_hat'], err_m['error_n'], '.')
    plt.xlabel('$\Delta$ vhat')
    plt.ylabel('e')
    plt.subplot(3, 1, 3)
    plt.plot(err_m['psi_hat'], err_m['error_psi'], '.')
    plt.xlabel('$\Delta$ rhat')
    plt.ylabel('e')
    plt.tight_layout()

    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(err_m['u_hat'] + err_m['error_u'], err_m['u_hat'], '.')
    # plt.xlabel('$\Delta$ u')
    # plt.ylabel('$\Delta$ uhat')
    # plt.subplot(3, 1, 2)
    # plt.plot(err_m['v_hat'] + err_m['error_v'], err_m['v_hat'], '.')
    # plt.xlabel('$\Delta$ v')
    # plt.ylabel('$\Delta$ vhat')
    # plt.subplot(3, 1, 3)
    # plt.plot(err_m['r_hat'] + err_m['error_r'], err_m['r_hat'], '.')
    # plt.xlabel('$\Delta$ r')
    # plt.ylabel('$\Delta$ rhat')
    # plt.tight_layout()

    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.hist(err_m['error_u'].values)
    # plt.subplot(3, 1, 2)
    # plt.hist(err_m['error_v'].values)
    # plt.subplot(3, 1, 3)
    # plt.hist(err_m['error_r'].values)

    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(Xtest[:, 7])
    # plt.plot(Xtest[:, 9])
    # plt.title('rpm')
    # plt.subplot(2, 1, 2)
    # plt.plot(Xtest[:, 6])
    # plt.plot(Xtest[:, 8])
    # plt.title('azimuth angle')
    # plt.tight_layout()
    #
    # plt.figure()
    # plt.plot(sol_m['surge_vel'] + Xtest[0, 3], label='model predictions')
    # plt.plot(Xtest[1:16, 3], label='truth')
    #
    # plt.figure()
    # plt.plot(sol_m['sway_vel'] + Xtest[0, 4], label='model predictions')
    # plt.plot(Xtest[1:16, 4], label='truth')
    #
    # plt.figure()
    # plt.plot(sol_m['yaw_vel'] + Xtest[0, 5], label='model predictions')
    # plt.plot(Xtest[1:16, 5], label='truth')
    #
    # sol_k = kinematic(Xtest[0, 2], Xtest[0, 3], Xtest[0, 4], Xtest[0, 5], 15)
    #
    # plt.figure()
    # plt.plot(sol_m['North'] + Xtest[0, 0], sol_m['East'] + Xtest[0, 1], label='model predictions')
    # plt.plot(Xtest[1:16, 0],
    #          Xtest[1:16, 1], label='truth')
    # plt.plot([Xtest[0, 0], sol_k[0] + Xtest[0, 0]],
    #          [Xtest[0, 1], sol_k[1] + Xtest[0, 1]], label='kinematic')
    #
    # plt.legend()
plt.show()
