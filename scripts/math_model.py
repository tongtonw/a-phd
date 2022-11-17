from fmpy import read_model_description, extract
from fmpy.fmi1 import FMU1Slave
import pandas as pd
import numpy as np
from numpy.linalg import inv


def fmuinitialize(fmu_filename):
    # define the model name and simulation parameters
    # fmu_filename = './PMAzimuth.fmu'
    # read the model description
    model_description = read_model_description(fmu_filename)
    # collect the value references
    vrs = {}
    for variable in model_description.modelVariables:
        vrs[variable.name] = variable.valueReference
    # extract the FMU
    unzipdir = extract(fmu_filename)

    fmu = FMU1Slave(guid=model_description.guid,
                    unzipDirectory=unzipdir,
                    modelIdentifier=model_description.coSimulation.modelIdentifier,
                    instanceName='instance1')
    # initialize
    fmu.instantiate()

    return fmu, vrs


class ModelPredict:
    def __init__(self, fmu, vrs, x0, y0, psi0, u0, v0, r0, rudder_cd_pt, rpm_cd_pt, rudder_cd_sb, rpm_cd_sb,
                 global_wind_direction,
                 global_wind_speed,
                 delta_t=30,
                 step_size=1):
        """
        :param fmu: PMAzimuth.fmu
        :param vrs:
        :param x0: initial north position, m, get from Lat, Lon
        :param y0: initial east position, m, get from Lat, Lon
        :param psi0: initial heading position, rad
        :param u0: initial surge speed, m/s, connnected with "Gunnerus/SeapathGPSVtg/SpeedKmHr"
        :param v0: initial sway speed, m/s, --> "Gunnerus/SeapathGPSVbw/TransGroundSpeed"
        :param r0: initial yaw speed, rad/s, --> "Gunnerus/SeapathMRU_rates/YawRate"
        :param rudder_cd_pt: portside thruster rudder angle, deg, --> "Gunnerus/hcx_port_mp/AzimuthFeedback"
        :param rpm_cd_pt: portside thruster speed, rpm, --> "Gunnerus/hcx_port_mp/RPMFeedback"
        :param rudder_cd_sb: starbord thruster rudder angle, deg, --> "Gunnerus/hcx_stbd_mp/AzimuthFeedback"
        :param rpm_cd_sb: starbord thruster speed, rpm, --> "Gunnerus/hcx_stbd_mp/RPMFeedback"
        :param global_wind_direction: deg --> "Gunnerus/dpWind/Wind_Direction"
        :param global_wind_speed: m/s --> "Gunnerus/dpWind/Wind_Speed"
        :param delta_t: prediction horizon, s, by default it is 30 second.
        :param step_size: by default it is 1 second.

        The unit of mqtt side needs double check and consist with those input variables.
        Output variables will be a dataframe with size [30, *] if the delta_t = 30 (prediction horizon)

        """

        self.x0 = x0
        self.y0 = y0
        self.psi0 = psi0
        self.u0 = u0
        self.v0 = v0
        self.r0 = r0
        self.rudder_cd_pt = rudder_cd_pt
        self.rpm_cd_pt = rpm_cd_pt
        self.rudder_cd_sb = rudder_cd_sb
        self.rpm_cd_sb = rpm_cd_sb
        self.delta_t = delta_t
        self.step_size = step_size

        self.global_wind_direction = global_wind_direction  # deg
        self.global_wind_speed = global_wind_speed  # m/s

        self.fmu, self.vrs = fmu, vrs

        wind_ang = np.arange(0, 181, 10) / 180 * np.pi

        # x direction
        cx = np.array(
            [-0.53, -0.59, -0.65, -0.59, -0.51, -0.47, -0.4, -0.29, -0.2, -0.18, -0.14, -0.05, 0.12, 0.37, 0.61, 0.82,
             0.86,
             0.72, 0.62])
        self.p_x = np.polyfit(wind_ang, cx, deg=5)

        # y direction
        cy = np.array(
            [0, 0.22, 0.42, 0.66, 0.83, 0.9, 0.88, 0.87, 0.86, 0.85, 0.83, 0.82, 0.81, 0.73, 0.58, 0.46, 0.26, 0.09, 0])
        self.p_y = np.polyfit(wind_ang, cy, deg=5)

        # n direction
        cn = np.array(
            [0, 0.05, 0.1, 0.135, 0.149, 0.148, 0.114, 0.093, 0.075, 0.04, 0.02, -0.013, -0.035, -0.041, -0.045, -0.04,
             -0.029, -0.014, 0])
        self.p_n = np.polyfit(wind_ang, cn, deg=5)

    def Merge(self, dict1, dict2):
        res = {**dict1, **dict2}
        return res

    def ship(self, x, F):
        # vessel parameters
        L_pp = 33.9
        # m = 493.472 * 1025
        m = 557.385 * 1025
        x_g = 1.8
        r_z = L_pp / 2
        I_z = m * r_z ** 2

        # 'Hydrodynamic Derivatives from Trondheim team report
        # Xudot = -51730.5698077703
        # Yvdot = -416325.955012191
        # Yrdot = 1269437.4753285
        # Nvdot = 1269437.4753285
        # Nrdot = -36399737.2119462

        # hydrodynamic derivatives from FullScale Matlab code
        Xudot = -100000
        Yvdot, Yrdot, Nvdot, Nrdot = -453413, -26543, 2425837, -3800854

        # hydrodynamic derivatives from VesselFMU
        # Xudot, Yvdot, Yrdot, Nvdot, Nrdot = -79320.5895, -408192.6004, 331671.6384, 331671.6384, -12245783.9277

        M = np.array([[m - Xudot, 0, 0],
                      [0, m - Yvdot, m * x_g - Yrdot],
                      [0, m * x_g - Nvdot, I_z - Nrdot]])

        etadot = np.array([x[3] * np.cos(x[2]) - x[4] * np.sin(x[2]),
                           x[3] * np.sin(x[2]) + x[4] * np.cos(x[2]),
                           x[5]]).reshape(-1, 1)

        C_RB = np.array([[0, 0, - m * (x_g * x[5][0] + x[4][0])],
                         [0, 0, m * x[3][0]],
                         [m * (x_g * x[5][0] + x[4][0]), -m * x[3][0], 0]])

        C_A = np.array([[0, 0, Yvdot * x[4][0] + 1 / 2 * (Yrdot * x[5][0] + Nvdot * x[5][0])],
                        [0, 0, -Xudot * x[3][0]],
                        [-Yvdot * x[4][0] - 1 / 2 * (Yrdot * x[5][0] + Nvdot * x[5][0]), Xudot * x[3][0], 0]])

        nu = np.array([x[3], x[4], x[5]]).reshape(-1, 1)

        vdot = np.dot(inv(M), F - (C_RB + C_A).dot(nu))
        ans = np.vstack([etadot, vdot]).reshape(-1, 1)
        return ans

    def hydrodynamic(self, u, v, r):
        """
        Hydrodynamic Derivatives from Trondheim team report
        """
        # Yuv = -9365.71971498149
        # Yur = 223824.070356427
        # Nuv = 196681.47332015
        # Nur = -2670249.09211213
        # Xvv = 1769.34745818901
        # Xvvvvdivuu = 51954.5854342518
        # Xrv = -245237.104934441
        # Xrr = 0
        # X0 = 0
        # Xu = 476.250244140625
        # Xuu = -1306.41870117188
        # Xuuu = 385.845001220703
        # Xuuuu = -46.3575286865234
        # Xuuuuu = -1.4837589263916
        #
        # X_h = np.sign(u) * ((
        #             Xuuuuu * u ** 5 + Xuuuu * u ** 4 + Xuuu * u ** 3 + Xuu * u ** 2 + Xu * u + X0)) + Xrr * r ** 2 + Xrv * r * v + Xvv * v * v + Xvvvvdivuu * v ** 4 * u ** 2
        #
        # Y_h = Yuv * u * v + Yur * u * r
        #
        # N_h = Nuv * u * v + Nur * u * r

        """ 
         hydrodynamic derivatives from VesselFMU
        """
        # Xudot, Yvdot, Yrdot, Nvdot, Nrdot = -79320.5895, -408192.6004, 331671.6384, 331671.6384, -12245783.9277
        # Xuu, Xvv, Xrr, Xrv, Xuvv, Xrvu, Xurr, Xumodv = -2100, 2880.5065, -29267.3371, -323707.1228, -2371.1792, \
        #                                                33926.4945, 83774.6548, 619.643
        # Yuv, Yur, Yuur, Yuuv, Yvvv, Yrrr, Yrrv, Yvvr, Ymodrv, Ymodvv, Ymodrr, Ymodvr = -31000.2779, 277765.7816, \
        #                                                                                11112.4159, 777.3214, \
        #                                                                                -11198.6727, 21117955.8908, \
        #                                                                                -12912371.0982, 102679.0573, \
        #                                                                                485260.3423, -21640.6686, \
        #                                                                                -1020386.1325, 321944.5769
        # Nuv, Nur, Nuur, Nuuv, Nvvv, Nrrr, Nrrv, Nvvr, Nmodrv, Nmodvv, Nmodrr, Nmodvr = 309051.5592, -2413124.664, \
        #                                                                                -187538.9302, -14041.6851, \
        #                                                                                -56669.589, -179720233.7453, \
        #                                                                                -585678.1567, -4278941.6707, \
        #                                                                                1626540.037, 116643.7578, \
        #                                                                                -9397835.7612, -2258197.6812
        # X_h = (Xuu * u * u) + Xvv * v ** 2 + Xrr * r ** 2 + Xrv * r * v + Xuvv * u * v ** 2 + Xrvu * u * r * v + Xurr * \
        #       u * r ** 2 + Xumodv * u * np.abs(v)
        #
        # Y_h = Yuv * u * v + Yur * u * r + Yuur * u ** 2 * r + Yuuv * u ** 2 * v + Yvvv * v ** 3 + Yrrr * r ** 3 + Yrrv \
        #       * r ** 2 * v + Yvvr * v ** 2 * r + Ymodrv * v * np.abs(r) + Ymodvv * np.abs(v) * v + Ymodrr * np.abs(r) \
        #       * r + Ymodvr * np.abs(v) * r
        #
        # N_h = Nuv * u * v + Nur * u * r + Nuur * u * u * r + Nuuv * u * u * v + Nvvv * v ** 3 + Nrrr * r ** 3 + Nrrv * \
        #       r * r * v + Nvvr * v * v * r + Nmodrv * np.abs(r) * v + Nmodvv * np.abs(v) * v + Nmodrr * np.abs(r) * r \
        #       + Nmodvr * np.abs(v) * r
        # #
        # Fh = np.array([X_h, Y_h, N_h]).reshape(-1, 1)

        """ 
         hydrodynamic derivatives from FullScale Matlab code
        """
        X_coeff = np.array([17496.74972, -4475.434732, -15679.11679, 175249.6048])
        Y_coeff = np.array([1093539, -185048, 4177435, -29940, 723602, 1570919, -16493927, -226243])
        N_coeff = np.array([-8614233, -508092, -2180055, - 70853, -3434695, -1307083, 7020848, -3912253])
        X_h = -(2.6528 * u ** 4 - 58.026 * u ** 3 + 480.12 * u ** 2 - 1756.2 * u + 2400.8) * 1000 + X_coeff.dot(
            np.array([v ** 2, u * (v ** 2), u * v * r, v * r]))

        Y_h = Y_coeff.dot(
            np.array([r, v, r * np.abs(r), v * np.abs(v), r * np.abs(v), v * np.abs(r), v * r ** 2, v ** 2 * r]))

        N_h = N_coeff.dot(
            np.array([r, v, r * np.abs(r), v * np.abs(v), r * np.abs(v), v * np.abs(r), v * r ** 2, v ** 2 * r]))

        Fh = np.array([X_h[0], Y_h[0], N_h[0]]).reshape(-1, 1)
        return Fh

    def get_wind_coef(self, attack_angle):
        CX = np.polyval(self.p_x, attack_angle)
        CY = np.polyval(self.p_y, attack_angle)
        CN = np.polyval(self.p_n, attack_angle)
        return [CX, CY, CN]

    def get_wind_force(self, psi, u, v, r):
        Loa = 33.9
        Alw = 172
        Afw = 81
        rhoa = 1.247

        # global_w_dir = np.mod(self.global_wind_direction + psi * 180 / np.pi, 360)
        # global_w_spd = np.abs(u - self.global_wind_speed)
        # realAngle = np.deg2rad(global_w_dir) - psi
        # u_wr = global_w_spd * np.cos(realAngle) + u
        # v_wr = global_w_spd * np.sin(realAngle) + v

        realAngle = np.deg2rad(self.global_wind_direction) - psi
        u_wr = self.global_wind_speed * np.cos(realAngle) + u
        v_wr = self.global_wind_speed * np.sin(realAngle) + v

        relVel = np.sqrt(u_wr ** 2 + v_wr ** 2)
        attackangle = np.arctan2(v_wr, u_wr)

        [Cx, Cy, Cn] = self.get_wind_coef(np.abs(attackangle))

        tauwx = 0.5 * rhoa * relVel ** 2 * Cx * Alw
        tauwy = -np.sign(attackangle) * 0.5 * rhoa * relVel ** 2 * Cy * Alw
        tauwn = -np.sign(attackangle) * 0.5 * rhoa * relVel ** 2 * Cn * Alw * Loa

        tauw = np.array([tauwx, tauwy, tauwn]).reshape(-1, 1)

        return tauw

    def get_one_side_f(self, ps, time, step_size, x0, rudder_cmd, rpm_cmd):
        if ps == 'P':
            ly = -2.7
        elif ps == 'S':
            ly = 2.7

        vr_inputs = [self.vrs['input_act_angle'], self.vrs['input_act_revs']]  # deg and rpm
        vr_outputs4 = [self.vrs['output_force_surge'], self.vrs['output_force_sway'], self.vrs['output_torque']]
        parameter_blinds = ['input_x_rel_ap', "input_z_rel_bl", "input_prop_diam", "input_distancetohull",
                            "input_bilgeradius", "input_rho", "input_lpp"]
        parameter_reference = []
        for para in parameter_blinds:
            parameter_reference.append(self.vrs[para])

        self.fmu.setReal(parameter_reference, [0., 0.55, 1.9, 1.5, 3., 1025., 33.9])
        self.fmu.setReal([self.vrs['input_y_rel_cl']], [ly])

        self.fmu.setReal([self.vrs['input_cg_x_rel_ap']], [15.52])
        self.fmu.setReal([self.vrs['input_cg_y_rel_cl']], [0.])
        self.fmu.setReal([self.vrs['input_cg_z_rel_bl']], [3.624])

        self.fmu.setReal([vr_inputs[0]], [rudder_cmd])
        self.fmu.setReal([vr_inputs[1]], [rpm_cmd])
        self.fmu.setReal([self.vrs['input_cg_surge_vel']], [x0[3]])
        self.fmu.setReal([self.vrs['input_cg_sway_vel']], [x0[4]])
        self.fmu.setReal([self.vrs['input_yaw_vel']], [x0[5] * 180 / 3.14])  # input_yaw_vel [deg/s]

        self.fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
        val_outputs = {'output_force_surge': self.fmu.getReal([vr_outputs4[0]])[0],
                       'output_force_sway': self.fmu.getReal([vr_outputs4[1]])[0],
                       'output_torque': self.fmu.getReal([vr_outputs4[2]])[0]}

        fy = self.fmu.getReal([vr_outputs4[1]])[0]
        fx = self.fmu.getReal([vr_outputs4[0]])[0]
        # fn = - fy * 15.52 - fy * 15.52
        fn = - fy * 15.52 - fx * ly
        val_outputs['output_torque_calculation'] = fn
        Faz = np.array([fx, fy, fn]).reshape(-1, 1)
        return Faz

    def simulate(self):
        rows = []

        ini_s = np.array([self.x0,
                          self.y0,
                          self.psi0,
                          self.u0,
                          self.v0,
                          self.r0]).reshape(-1, 1)

        val_cmd = {'pt_rpm': self.rpm_cd_pt, 'pt_azi': self.rudder_cd_pt, 'sb_rpm': self.rpm_cd_sb,
                   'sb_azi': self.rudder_cd_sb}
        val_ini_s = {'ini_n': ini_s[0][0], 'ini_e': ini_s[1][0], 'ini_psi': ini_s[2][0], 'ini_u': ini_s[3][0],
                     'ini_v': ini_s[4][0], ' ini_r': ini_s[5][0]}
        x0 = ini_s
        time = 0
        while time < self.delta_t:
            F_pt = self.get_one_side_f('P', time, 1., x0, rudder_cmd=self.rudder_cd_pt,
                                       rpm_cmd=self.rpm_cd_pt)
            F_sb = self.get_one_side_f('S', time, 1., x0, rudder_cmd=self.rudder_cd_sb,
                                       rpm_cmd=self.rpm_cd_sb)
            Fhs = self.hydrodynamic(x0[3], x0[4], x0[5])
            Fw = self.get_wind_force(x0[2], x0[3], x0[4], x0[5])
            Fto = Fhs + F_pt + F_sb + Fw
            xdot = self.ship(x0, Fto)

            x0 = x0 + xdot * self.step_size

            # relative states
            val_state = {'North': x0[0][0] - ini_s[0][0],
                         'East': x0[1][0] - ini_s[1][0],
                         'yaw': x0[2][0] - ini_s[2][0],
                         'surge_vel': x0[3][0] - ini_s[3][0],
                         'sway_vel': x0[4][0] - ini_s[4][0],
                         'yaw_vel': x0[5][0] - ini_s[5][0]}

            val_time = {'time': time}

            rowsDict = self.Merge(self.Merge(self.Merge(val_time, val_state), val_cmd), val_ini_s)
            # [t, N, E, psi, u, v, r, port_rpm, port_azi, star_rpm, star_azi, real_n, real_e, real_psi, real_u,
            # real_v, # real_r, ini_n, ini_e, ini_psi, ini_u, ini_v, ini_r]
            rows.append(rowsDict)
            time += self.step_size

        result = pd.DataFrame(rows)
        return result
