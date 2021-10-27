import numpy as np
from numpy import pi
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from scipy.integrate import solve_ivp
import scipy.linalg as linalg
from time import time

from PyDiffGame import ContinuousPyDiffGame

# fig = plt.figure()
# ax = fig.add_subplot(111,
#                      aspect='equal',
#                      xlim=(-5, 5),
#                      ylim=(-1, 1),
#                      title="Inverted Pendulum Simulation")
# ax.grid()

# animation parameters
# origin = (0, 0)
dt = 0.02
frames = 200
t_span = [0.0, frames * dt]

m1 = 500
m2 = 10
g = 9.81
L = 15
l = L / 2  # CoM of uniform rod
I = 1 / 12 * m2 * (L ** 2)
b = 1  # dampening coefficient

# original linear system
denominator = m1 * m2 * (l ** 2) + I * (m1 + m2)
A21 = (m2 ** 2) * g * (l ** 2) / denominator
A31 = m2 * g * l * (m1 + m2) / denominator
A22 = -b * (m2 * (l ** 2) + I) / denominator
A32 = -m2 * l * b / denominator
B2 = (m2 * (l ** 2) + I) / denominator
B3 = m2 * l / denominator

A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, A21, A22, 0],
              [0, A31, A32, 0]])

B21 = (m2 * (l ** 2) + I) / denominator
B31 = m2 * l / denominator
B22 = B31
B32 = (m1 + m2) / denominator

B = np.array([[0, 0],
              [0, 0],
              [B21, B22],
              [B31, B32]])

Q = np.array([[1, 0, 0, 0],
              [0, 10, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

R_constant = 0.001

R = np.array([[R_constant, 0],
              [0, R_constant]])

P = np.matrix(linalg.solve_continuous_are(A, B, Q, R))
K = np.matrix(linalg.inv(R) * (B.T * P))
# eigVals, eigVecs = linalg.eig(A - B * K)[0:2]
ss = np.array([0., 0., 0., 0.])


# print(f"eigVecs:\n{eigVecs}")
# print(f"eigVals:\n{eigVals}")
# print(f"K:\n{K}")
#
# pendulumArm = lines.Line2D(xdata=origin,
#                            ydata=origin,
#                            color='r')
#
# cart = patches.Rectangle(xy=origin,
#                          width=0.5,
#                          height=0.15,
#                          color='b')
#
#
# def stateSpace(_, i):
#     # returns state space model evaluated at initial conditions
#     S = np.sin(i[1])
#     C = np.cos(i[1])
#     denominator = (m2 * l * C) ** 2 - (m1 + m2) * (m2 * (l ** 2) + I)
#     # nonlinear system
#     ss[0] = i[2]
#     ss[1] = i[3]
#     ss[2] = (1 / denominator) * (
#             -(m2 ** 2) * g * (l ** 2) * C * S - m2 * l * S * (m2 * (l ** 2) + I) * (i[3] ** 2) + b * (
#             m2 * (l ** 2) + I) * i[2]) - ((m2 * (l ** 2) + I) / denominator) * u(i)
#     ss[3] = (1 / denominator) * (
#             (m1 + m2) * m2 * g * l * S + (m2 ** 2) * (l ** 2) * S * C * (i[3] ** 2) - b * m2 * l * C * i[2]) + (
#                     (m2 * l * C) / denominator) * u(i)
#
#     return ss


initial = np.array([200,  # x
                    pi * (1 - 2 / 3),  # theta
                    -100,  # x_dot
                    0])  # theta_dot
final = np.array([0,  # x
                  pi,  # theta
                  0,  # x_dot
                  0]  # theta_dot
                 )
# u = lambda x: np.matmul(-K, (x - final))
# ts = np.linspace(t_span[0], t_span[1], frames)
# pendulum_state = solve_ivp(fun=stateSpace,
#                            t_span=t_span,
#                            y0=initial,
#                            t_eval=ts,
#                            rtol=1e-8
#                            )
#
#
# def init():
#     ax.add_patch(cart)
#     ax.add_line(pendulumArm)
#
#     return pendulumArm, cart
#
#
# def animate(i):
#     xPos = pendulum_state.y[0][i]
#     theta = pendulum_state.y[1][i]
#     x = [origin[0] + xPos, origin[0] + xPos + L * np.sin(theta)]
#     y = [origin[1], origin[1] - L * np.cos(theta)]
#     pendulumArm.set_xdata(x)
#     pendulumArm.set_ydata(y)
#     cartPos = [origin[0] + xPos - cart.get_width() / 2, origin[1] - cart.get_height()]
#     cart.set_xy(cartPos)
#
#     return pendulumArm, cart
#
#
# t0 = time()
# animate(0)  # sample time required to evaluate animate() function
# t1 = time()
# interval = 1000 * dt - (t1 - t0)
#
# anim = animation.FuncAnimation(fig=fig,
#                                func=animate,
#                                init_func=init,
#                                frames=frames,
#                                interval=interval,
#                                blit=True)
# plt.show()

one_player_game = ContinuousPyDiffGame(A=A,
                                       B=[B],
                                       Q=[Q],
                                       R=[R],
                                       x_0=initial,
                                       x_T=final,
                                       T_f=30
                                       )
one_player_game.solve_game_and_simulate_state_space()

M1 = np.array([[10, 0],
               [0, 10]])
M2 = np.array([[1],
               [1]])

B = [B @ M1,
     B @ M2]

two_player_game = ContinuousPyDiffGame(A=A,
                                       B=B,
                                       Q=[Q] * 2,
                                       R=[R, np.array([R_constant])],
                                       x_0=initial,
                                       x_T=final,
                                       T_f=30
                                       )
two_player_game.solve_game_and_simulate_state_space()
