import numpy as np
from numpy import pi
from control import ctrb
# import sympy as sp

from scipy.linalg import solve_continuous_are

# import matplotlib.animation as animation
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.lines as lines
# from scipy.integrate import solve_ivp
# from time import time
# from termcolor import colored

from ContinuousPyDiffGame import ContinuousPyDiffGame

# from DiscretePyDiffGame import DiscretePyDiffGame

m_c = 500
m_p = 10
g = 9.81
L = 15
l = L / 2  # CoM of uniform rod
I = 1 / 12 * m_p * (L ** 2)

x_0 = np.array([20,  # x
                pi / 3,  # theta
                0,  # x_dot
                0]
               )  # theta_dot
x_T = np.array([0,  # x
                pi,  # theta
                0,  # x_dot
                0]  # theta_dot
               )

# # original linear system
D = m_c * m_p * l ** 2 + I * (m_c + m_p)
a21 = m_p ** 2 * g * l ** 2 / D
a31 = m_p * g * l * (m_c + m_p) / D

# a21, a31 = sp.var('a21 a31')

A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, a21, 0, 0],
              [0, a31, 0, 0]])

# p11, p12, p13, p14, p22, p23, p24, p33, p34, p44 = sp.var('p11 p12 p13 p14 p22 p23 p24 p33 p34 p44')
# ps = [p11, p12, p13, p14, p22, p23, p24, p33, p34, p44]

# P = np.array([[p11, p12, p13, p14],
#               [p12, p22, p23, p24],
#               [p13, p23, p33, p34],
#               [p14, p24, p34, p44]
#               ])
# P = sp.Matrix(P)

b21 = (m_p * l ** 2 + I) / D
b31 = m_p * l / D
b22 = b31
b32 = (m_c + m_p) / D
#

# b21, b22, b31, b32 = sp.var('b21 b22 b31 b32')

B = np.array([[0, 0],
              [0, 0],
              [b21, b22],
              [b31, b32]])
# B = sp.Matrix(B)


q_s = 0.1
q_m = 10
q_l = 100

# q_s, q_m, q_l = sp.var('q_s q_m q_l')

Q_x = np.diag([q_l, q_s, q_m, q_s])
# Q_x = sp.Matrix(Q_x)

Q_theta = np.diag([q_s, q_l, q_s, q_m])
# Q_theta = sp.Matrix(Q_theta)

Q = (Q_x + Q_theta) / 2

r = 0.0001
# r = sp.var('r')
R = np.diag([r, r])
# print(np.linalg.matrix_rank(ctrb(A, B)))

# def is_pos_def(x):
#     return np.all(np.linalg.eigvals(x) > 0)

P = solve_continuous_are(A, B, Q, R)
K = np.matrix(np.linalg.inv(R) @ B.T @ P)
A_cl = A - B @ K
w, v = np.linalg.eig(A_cl)
print(w)


# print(x)
# print(is_pos_def(x))

# sys = P @ A + A.T @ P - P @ B @ np.linalg.inv(R) @ B.T @ P + Q
# PB = P @ B
# print(sp.latex(PB @ PB.T))

#
#
# one_player_game = ContinuousPyDiffGame(A=A,
#                                        B=B,
#                                        Q=Q,
#                                        R=R,
#                                        x_0=x_0,
#                                        x_T=x_T,
#                                        )
# one_player_game.solve_game_and_simulate_state_space()
# print(f'One player cost: {one_player_game.get_costs()}')
#
#
# R_x = np.diag([r, r])
# R_theta = np.array([r / 10])
#
# # M_x = np.array([[1],
# #                [1]])
# #
# # M_theta = np.array([[1, 0],
# #                     [0, 1]])
# #
# # two_player_game = ContinuousPyDiffGame(A=A,
# #                                        B=[B @ M1,
# #                                           B @ M2],
# #                                        Q=[Q_x, Q_theta],
# #                                        R=[R_x, R_theta],
# #                                        x_0=x_0,
# #                                        x_T=x_T,
# #                                        )
# #
# # two_player_game.solve_game_and_simulate_state_space()
# # print(f'Two player cost: {two_player_game.get_costs()}')
#
# # for L in [int(1000 / i) for i in range(1, 10)]:
# #     cont_game = ContinuousPyDiffGame(A=A,
# #                                      B=[B @ M1,
# #                                         B @ M2],
# #                                      Q=[Q_x, Q_theta],
# #                                      R=[R_x, R_theta],
# #                                      x_0=x_0,
# #                                      x_T=x_T,
# #                                      T_f=T_f,
# #                                      L=L,
# #                                      )
# #
# #     cont_game.solve_game_and_simulate_state_space()
# #     cont_costs = cont_game.get_costs()
# #
# #     d_game = DiscretePyDiffGame(A=A,
# #                                 B=[B @ M1,
# #                                    B @ M2],
# #                                 Q=[Q_x, Q_theta],
# #                                 R=[R_x, R_theta],
# #                                 x_0=x_0,
# #                                 T_f=T_f,
# #                                 L=L
# #                                 )
# #
# #     d_game.solve_game_and_simulate_state_space()
# #     d_costs = d_game.get_costs()
# #
# #     print(colored('#' * 10 + f' L = {L} ' + '#' * 10, 'blue'))
# #     print(colored(f'Continuous costs: {cont_costs}\n'
# #                   f'Continuous costs sum: {cont_costs.sum()}', 'red'))
# #     print(colored(f'Discrete costs: {d_costs}\n'
# #                   f'Discrete costs sum: {d_costs.sum()}', 'red'))