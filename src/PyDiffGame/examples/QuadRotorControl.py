# Imports

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
from math import cos, sin
from scipy.optimize import nnls
from numpy import sin, cos, arctan
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from ContinuousPyDiffGame import ContinuousPyDiffGame

# Global Constants

g = 9.81
sum_theta_init = -3 / g
Ixx = 7.5e-3
Iyy = 7.5e-3
Izz = 1.3e-2
m = 0.65
l = 0.23
Jr = 6e-5
b = 3.13e-5
d = 7.5e-7

a1 = (Iyy - Izz) / Ixx
a2 = Jr / Ixx
a3 = (Izz - Ixx) / Iyy
a4 = Jr / Iyy
a5 = (Ixx - Iyy) / Izz

b1 = l / Ixx
b2 = l / Iyy
b3 = 1 / Izz

v_d_s_0_2 = 0
h_d_s_0_2 = 0
sum_theta = sum_theta_init
sum_theta_2 = 0


# Low-Level Control

def quad_rotor_state_diff_eqn_for_given_pqrT(X, _, p, q, r, T, Plast):
    phi, dPhidt, theta, dThetadt, psi, dPsidt, z, dzdt, x, dxdt, y, dydt = X
    u_x = cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)
    u_y = cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)

    K = cos(phi) * cos(theta) / m
    U = low_level_angular_rate_controller([dPhidt, dThetadt, dPsidt], p, q, r, T, Plast)
    omegas_squared_coeffs = np.array([[b] * 4,
                                      [0, -b, 0, b],
                                      [b, 0, -b, 0],
                                      [-d, d, -d, d]
                                      ])
    u1, u2, u3, u4 = U
    omegas_squared = nnls(omegas_squared_coeffs, np.array([u1, u2, u3, u4]))[0]
    omegas = np.sqrt(omegas_squared)

    u5 = d * (omegas[0] - omegas[1] + omegas[2] - omegas[3])

    dPhiddt = dThetadt * (dPsidt * a1 + u5 * a2) + b1 * u2
    dThetaddt = dPhidt * (dPsidt * a3 - u5 * a4) + b2 * u3
    dPsiddt = dThetadt * dPhidt * a5 + b3 * u4
    dzddt = g - K * u1
    dxddt = u_x * u1 / m
    dyddt = u_y * u1 / m

    return np.array([dPhidt, dPhiddt, dThetadt, dThetaddt, dPsidt, dPsiddt, dzdt,
                     dzddt, dxdt, dxddt, dydt, dyddt], dtype='float64')


def low_level_angular_rate_controller(x, p, q, r, T, Plast):
    B1 = np.array([[b1],
                   [0],
                   [0]])

    B2 = np.array([[0],
                   [b2],
                   [0]])

    B3 = np.array([[0],
                   [0],
                   [b3]])

    R1 = np.array([[0.1]])
    R2 = np.array([[0.1]])
    R3 = np.array([[0.1]])

    B = [B1, B2, B3]
    R = [R1, R2, R3]
    P_sol = Plast
    reduced_X = np.array(x) - np.array([p, q, r])
    reduced_X_tr = reduced_X.forward_time
    inv_Rs = [inv(r) for r in R]
    B_t = [b.T for b in B]
    U_angular = np.array([- r @ b @ p @ reduced_X_tr for r, b, p in zip(inv_Rs, B_t, P_sol)]).reshape(3, )
    u2, u3, u4 = U_angular
    U = [T, u2, u3, u4]

    return U


def get_P_quad_given_angular_rates(x, P_sol):
    A = np.array([[0, (1 / 2) * a1 * x[2], (1 / 2) * a1 * x[1]],
                  [(1 / 2) * a3 * x[2], 0, (1 / 2) * a3 * x[0]],
                  [(1 / 2) * a5 * x[1], (1 / 2) * a5 * x[0], 0]])

    B1 = np.array([[b1],
                   [0],
                   [0]])

    B2 = np.array([[0],
                   [b2],
                   [0]])

    B3 = np.array([[0],
                   [0],
                   [b3]])

    Q1 = np.array([[1000, 0, 0],
                   [0, 10, 0],
                   [0, 0, 10]])

    Q2 = np.array([[10, 0, 0],
                   [0, 1000, 0],
                   [0, 0, 10]])

    Q3 = np.array([[10, 0, 0],
                   [0, 10, 0],
                   [0, 0, 1000]])

    R1 = np.array([[0.1]])
    R2 = np.array([[0.1]])
    R3 = np.array([[0.1]])

    B = [B1, B2, B3]
    R = [R1, R2, R3]
    Q = [Q1, Q2, Q3]
    P = ContinuousPyDiffGame(A=A, B=B, Q=Q, R=R, P_f=P_sol, show_legend=False).solve_game_and_simulate_state_space()
    Plast = P[-1]

    return Plast


# High-Level Control

def get_mf_numerator(F3, R11, F1, R31, a_y, R12, R32):
    return F3 * R11 - F1 * R31 + a_y * (R12 * R31 - R11 * R32)


def get_mf_denominator(F2, R11, F1, R21, a_y, R22, R12):
    return - F2 * R11 + F1 * R21 + a_y * (R11 * R22 - R12 * R21)


def get_mc_numerator(mf_numerator, a_z, R31, R13, R11, R33):
    return mf_numerator + a_z * (R13 * R31 - R11 * R33)


def get_mc_denominator(mf_denominator, a_z, R11, R23):
    return mf_denominator - a_z * R11 * R23


def hpf_ode_v_d_s(v_d_s, _, f_a, f_b, P_z_tilda):
    return f_a * v_d_s + f_b * P_z_tilda


def hpf_ode_h_d_s(h_d_s, _, f_a, f_b, P_y_tilda):
    return f_a * h_d_s + f_b * P_y_tilda


def calculate_Bs(u_sizes, dividing_matrix, B):
    block_matrix = B @ dividing_matrix
    Bs = []

    last = 0
    for u_size in u_sizes:
        Bs += [block_matrix[:, last:u_size + last]]
        last = u_size

    return Bs


def wall_punishment(wall_distance, a_y):
    return 3 * (10 ** 2) * (wall_distance / a_y) ** 2


def get_higher_level_control2(state, st, a_y):
    global v_d_s_0_2, h_d_s_0_2, sum_theta, sum_theta_2
    # a_y = 1
    a_z = -2.5

    x = state[8]
    y = state[10]
    z = state[6]
    phi = state[0]
    theta = state[2]
    psi = state[4]

    sphi = sin(phi)
    cphi = cos(phi)
    stheta = sin(theta)
    ctheta = cos(theta)
    spsi = sin(psi)
    cpsi = cos(psi)

    sectheta = 1 / ctheta
    tanpsi = spsi / cpsi

    vp_x = sectheta * (sphi * stheta - cphi * tanpsi)
    vp_y = sectheta * (- cphi * stheta - sphi * tanpsi)

    R11 = ctheta * cpsi
    R21 = cpsi * stheta * sphi - cphi * spsi
    R31 = cpsi * stheta * cphi + sphi * spsi
    R12 = ctheta * spsi
    R22 = spsi * stheta * sphi + cphi * cpsi
    R32 = spsi * stheta * cphi - sphi * cpsi
    R13 = - stheta
    R23 = ctheta * sphi
    R33 = ctheta * cphi
    r = np.array([[R11, R21, R31], [R12, R22, R32], [R13, R23, R33]])

    curr_loc = np.array([[x], [y], [z]])
    [F1, F2, F3] = r.T @ curr_loc

    mfr_numerator = get_mf_numerator(F3, R11, F1, R31, a_y, R12, R32)
    mfr_denominator = get_mf_denominator(F2, R11, F1, R21, a_y, R22, R12)
    mfl_numerator = get_mf_numerator(F3, R11, F1, R31, -a_y, R12, R32)
    mfl_denominator = get_mf_denominator(F2, R11, F1, R21, -a_y, R22, R12)

    mfr = mfr_numerator / mfr_denominator
    mfl = mfl_numerator / mfl_denominator

    mcr = get_mc_numerator(mfr_numerator, a_z, R31, R13, R11, R33) / get_mc_denominator(mfr_denominator, a_z, R11, R23)
    mcl = get_mc_numerator(mfl_numerator, a_z, R31, R13, R11, R33) / get_mc_denominator(mfl_denominator, a_z, R11, R23)

    at_mcl = arctan(mcl)
    at_mcr = arctan(mcr)
    at_mfl = arctan(mfl)
    at_mfr = arctan(mfr)

    p_y_tilda = at_mcl + at_mcr - at_mfl - at_mfr
    p_z_tilda = at_mfl - at_mfr + at_mcl - at_mcr
    phi_tilda = at_mfl + at_mfr + at_mcl + at_mcr

    f_a = -10
    f_b = 8
    f_c = -12.5
    f_d = 10

    v_d_s = v_d_s_0_2
    h_d_s = h_d_s_0_2
    v_d = f_c * v_d_s + f_d * p_z_tilda
    h_d = f_c * h_d_s + f_d * p_y_tilda

    data_points = 100
    t = np.linspace(st, st + 0.1, data_points)
    v_d_s = np.mean(odeint(func=hpf_ode_v_d_s, y0=v_d_s_0_2, t=t, args=(f_a, f_b, p_z_tilda)))
    h_d_s = np.mean(odeint(func=hpf_ode_h_d_s, y0=h_d_s_0_2, t=t, args=(f_a, f_b, p_y_tilda)))

    v_d_s_0_2 = v_d_s
    h_d_s_0_2 = h_d_s

    Q1 = np.array([[1000, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1000, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1000, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0.1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0.1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 10, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0.05, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    R1 = np.array([[10, 0, 0, 0],
                   [0, 10, 0, 0],
                   [0, 0, 10, 0],
                   [0, 0, 0, 0.01]])

    A = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [g, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0]])

    B = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, -1 / m],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    dividing_matrix = np.array([[1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 1],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0]])

    u_sizes = [4, 1]
    Bs = calculate_Bs(u_sizes, dividing_matrix, B)

    R2 = np.array([[10]])

    max_punishment = wall_punishment(a_y, a_y)
    curr_punishment = min(max_punishment, wall_punishment(p_y_tilda[0], a_y))

    Q_wall_0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, curr_punishment]])
    Q_speed_up_0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1000, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    R = [R1, R2]
    if abs(p_y_tilda[0] / a_y) < 0.5:
        Q = [0.01 * Q1, 0.01 * Q_speed_up_0]
    else:
        Q = [0.01 * Q1, 0.01 * Q_wall_0]
    P_sol = [0.01 * Q1, 0.01 * Q1]

    Psol = ContinuousPyDiffGame(A=A, B=Bs, Q=Q, R=R, P_f=P_sol, show_legend=False).\
        solve_game_and_simulate_state_space()
    Plast = Psol[-1]
    N = 2
    M = 9
    P_size = M ** 2
    Plast = [(Plast[i * P_size:(i + 1) * P_size]).reshape(M, M) for i in range(N)]

    inv_Rs = [inv(r) for r in R]
    B_t = [b.forward_time for b in Bs]

    U_Agenda1 = - inv_Rs[0] @ B_t[0] @ Plast[0] @ np.array(
        [-phi_tilda[0], -vp_y, -vp_x, -p_y_tilda[0], -h_d[0], -p_z_tilda[0], -v_d[0], sum_theta, sum_theta_2])
    U_Agenda2 = - inv_Rs[1] @ B_t[1] @ Plast[1] @ np.array(
        [-phi_tilda[0], -vp_y, -vp_x, -p_y_tilda[0], -h_d[0], -p_z_tilda[0], -v_d[0], sum_theta, sum_theta_2])

    Us = [U_Agenda1, U_Agenda2]

    U_all_Out = dividing_matrix @ np.concatenate(Us).ravel().forward_time
    p_r, q_r, r_r, t_r = U_all_Out

    tilda_state = np.array(
        [phi_tilda[0], vp_y, vp_x, p_y_tilda[0], h_d[0], p_z_tilda[0], v_d[0], sum_theta, sum_theta_2])
    sum_theta = sum_theta - vp_y * 0.1
    sum_theta_2 = sum_theta_2 - vp_y * 0.1

    return p_r, q_r, r_r, t_r + m * g, tilda_state


# Simulation

Q1 = np.array([[1000, 0, 0],
               [0, 10, 0],
               [0, 0, 10]])
Q2 = np.array([[10, 0, 0],
               [0, 1000, 0],
               [0, 0, 10]])
Q3 = np.array([[10, 0, 0],
               [0, 10, 0],
               [0, 0, 1000]])
Q = [Q1, Q2, Q3]
M = 3
P_size = M ** 2
N = 3
tTotal = [0]
tTotal_low = [0]
a_ys = [0.55]
quad_rotor_state_PD_dynamic = {}
quad_rotor_omega_D_dynamic = {}
quad_rotor_state_PD_dynamic_low = {}
tilda_state_dynamic = {}

for a_y in a_ys:

    T_start = 0
    deltaTstate = 0.1
    X_rotor_0 = np.array([0.1, 0, 0, 0, 0.1, 0, -1, 0, 0, 0, 0.3, 0])
    omega_rotor_0 = np.array([0, 0, 0])
    quad_rotor_state_PD_dynamic[a_y] = [X_rotor_0]
    quad_rotor_omega_D_dynamic[a_y] = [omega_rotor_0]
    quad_rotor_state_PD_dynamic_low[a_y] = [X_rotor_0]
    tilda_state = np.array([0, 0, 0, 0, 0, 0, 0, sum_theta_init, 0])
    tilda_state_dynamic[a_y] = [tilda_state]

    X_rotor_0_PD = X_rotor_0
    Plast = [Q1, Q2, Q3]
    v_d_s_0_2 = 0
    h_d_s_0_2 = 0
    sum_theta = sum_theta_init
    sum_theta_2 = 0

    for i in tqdm(range(300)):
        # p_r, q_r, r_r, t_r, tilda_state_l = get_higher_level_control(X_rotor_0_PD,T_start, a_y)
        p_r2, q_r2, r_r2, t_r2, tilda_state_l2 = get_higher_level_control2(X_rotor_0_PD, T_start, a_y)
        tilda_state = [tilda_state_l2]
        tilda_state_dynamic[a_y] = np.append(tilda_state_dynamic[a_y], tilda_state, axis=0)
        T_end = T_start + deltaTstate
        data_points = 100
        t = np.linspace(T_start, T_end, data_points)
        Plast = get_P_quad_given_angular_rates([X_rotor_0_PD[1], X_rotor_0_PD[3], X_rotor_0_PD[5]], Plast)
        Plast = [(Plast[k * P_size:(k + 1) * P_size]).reshape(M, M) for k in range(N)]
        quad_rotor_state_PD = odeint(quad_rotor_state_diff_eqn_for_given_pqrT, X_rotor_0_PD, t,
                                     args=(p_r2, q_r2, r_r2, t_r2, Plast))
        omega_rotor_i = np.array([p_r2, q_r2, r_r2])
        X_rotor_0_PD = quad_rotor_state_PD[-1]
        T_start = T_end
        quad_rotor_state_PD_dynamic[a_y] = np.append(quad_rotor_state_PD_dynamic[a_y],
                                                     quad_rotor_state_PD, axis=0)
        quad_rotor_omega_D_dynamic[a_y] = np.append(quad_rotor_omega_D_dynamic[a_y],
                                                    [omega_rotor_i], axis=0)
        quad_rotor_state_PD_dynamic_low[a_y] = np.append(quad_rotor_state_PD_dynamic_low[a_y],
                                                         [X_rotor_0_PD], axis=0)
        if a_y == a_ys[0]:
            tTotal = np.append(tTotal, t)
            tTotal_low = np.append(tTotal_low, T_end)

angles = {'phi': [0, 0], 'theta': [2, 1], 'psi': [4, 2]}
positions = {'x': [8, 7], 'z': [6, 5], 'y': [10, 3]}
velocities = {'y_dot': [11, 4], 'z_dot': [7, 6], 'x_dot': 9}

plot_vars = {**angles, **positions, **velocities}


def plot_var(var):
    var_indices = plot_vars[var]

    for a_y in quad_rotor_state_PD_dynamic_low.keys():
        plt.figure(dpi=130)

        plt.title('$a _y = \ $' + str(a_y) + '$ \ [m]$', fontsize=16)

        plt.plot(tTotal_low[1:], quad_rotor_state_PD_dynamic_low
                                 [a_y][1:, var_indices[0] if var != 'x_dot' else var_indices])

        if var in positions.keys():
            plt.plot(tTotal_low[1:], tilda_state_dynamic[a_y][1:, var_indices[1]])
        else:
            plt.plot(tTotal_low[1:], -tilda_state_dynamic[a_y][1:, var_indices[1]])

        if var in positions.keys():
            plt.legend(['$' + var + ' \\ [m]$', '$\\tilde{' + var + '} \\ [m]$'])
        else:
            if 'dot' in var:
                plt.legend(['$\\dot{' + var[0] + '} \\ \\left[ \\frac{m}{sec} \\right]$',
                            '$\\tilde{\\dot{' + var[0] + '}} \\ \\left[ \\frac{m}{sec} \\right]$'])
            else:
                plt.legend(['$\\' + var + ' \\ [rad]$', '$\\tilde{\\' + var + '} \\ [rad]$'])

        plt.xlabel('$t \ [sec]$', fontsize=16)
        plt.grid()
        plt.show()


for curr_a_y, var in quad_rotor_state_PD_dynamic_low.items():
    plt.figure(dpi=300)
    plt.plot(tTotal_low[1:], var[1:, 9])
    plt.plot(tTotal_low[1:], abs(tilda_state_dynamic[curr_a_y][1:, 3] / curr_a_y))
    plt.plot(tTotal_low[1:], 0.5 * np.ones(len(tTotal_low[1:])))
    plt.grid()
    plt.xlabel('$t \ [sec]$', fontsize=12)
    # plt.title('$a_y = ' + str(curr_a_y) + ' \ [m]$', fontsize=12)
    #   plt.legend(['$\\dot{x} \ [\\frac{m}{s}]$', '$\\frac{\\tilde{y}}{a_y}$'])
    plt.legend(['Forward Velocity', 'Wall Distance Measure'])
    plt.show()

for var in ['x', 'y', 'y_dot', 'z', 'z_dot', 'phi', 'theta', 'psi']:
    plot_var(var)

time_ratio1 = len(tTotal_low) / max(tTotal_low)
time_ratio2 = len(tTotal) / max(tTotal)

for a_y in quad_rotor_omega_D_dynamic.keys():
    fig, ax = plt.subplots(1, dpi=150)

    t01 = int(time_ratio1 * 0.8)
    t1 = int(time_ratio1 * 1.1)
    t02 = int(time_ratio2 * 0.8)
    t2 = int(time_ratio2 * 1)

    x1 = tTotal_low[t01:t1]
    y1 = quad_rotor_omega_D_dynamic[a_y][t01:t1, 1]
    x2 = tTotal[t02:t2]
    y2 = quad_rotor_state_PD_dynamic[a_y][t02:t2, 3]

    ax.step(x1, y1, linewidth=1)
    ax.plot(x2, y2, linewidth=1)
    plt.xlabel('$t \ [sec]$', fontsize=14)
    plt.grid()

    axins = zoomed_inset_axes(ax, 3.5, borderpad=3)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    xlim = [0.898, 0.905]
    ylim = [0.193, 0.205]

    axins.set_xlim(xlim)
    axins.set_ylim(ylim)
    axins.step(x1, y1, linewidth=1)
    axins.plot(x2, y2, linewidth=1)
    plt.grid()

    fig.legend(['$\\dot{\\theta}_d \\ \\left[ \\frac{rad}{sec} \\right]$',
                '$\\dot{\\theta} \\ \\left[ \\frac{rad}{sec} \\right]$'],
               bbox_to_anchor=(0.28, 0.35) if a_y != 0.6 else (0.3, 0.35))

    plt.show()

for sol in quad_rotor_state_PD_dynamic_low.values():
    plt.figure(dpi=300)
    plt.plot(tTotal_low[0:], sol[0:, 0:6])
    plt.xlabel('$t \ [sec]$', fontsize=14)
    plt.legend(['$\\phi[rad]$', '$\\dot{\phi}\\left[ \\frac{rad}{sec} \\right]$', '$\\theta[rad]$',
                '$\\dot{\\theta}\\left[ \\frac{rad}{sec} \\right]$', '$\\psi[rad]$',
                '$\\dot{\psi}\\left[ \\frac{rad}{sec} \\right]$'], ncol=2, loc='upper right')
    plt.grid()
    plt.show()

for sol in quad_rotor_state_PD_dynamic_low.values():
    plt.figure(dpi=300)
    plt.plot(tTotal_low[0:], sol[0:, 6:8], tTotal_low[0:], sol[0:, 10:12])
    plt.xlabel('$t \ [sec]$', fontsize=14)
    plt.legend(
        ['$z[m]$', '$\\dot{z}\\left[ \\frac{m}{sec} \\right]$', '$y[m]$', '$\\dot{y}\\left[ \\frac{m}{sec} \\right]$'])
    plt.grid()
    plt.show()
