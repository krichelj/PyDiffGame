import numpy as np
from numpy import pi
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from scipy.integrate import solve_ivp
from time import time

from PyDiffGame import ContinuousPyDiffGame

m_c = 500
m_p = 10
g = 9.81
L = 15
l = L / 2  # CoM of uniform rod
I = 1 / 12 * m_p * (L ** 2)
b = 0  # dampening coefficient

initial = np.array([20,  # x
                    pi / 3,  # theta
                    0,  # x_dot
                    0]
                   )  # theta_dot
final = np.array([0,  # x
                  pi,  # theta
                  0,  # x_dot
                  0]  # theta_dot
                 )

# original linear system
denominator = m_c * m_p * (l ** 2) + I * (m_c + m_p)
A21 = (m_p ** 2) * g * (l ** 2) / denominator
A31 = m_p * g * l * (m_c + m_p) / denominator
A22 = -b * (m_p * (l ** 2) + I) / denominator
A32 = -m_p * l * b / denominator
B2 = (m_p * (l ** 2) + I) / denominator
B3 = m_p * l / denominator

A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, A21, A22, 0],
              [0, A31, A32, 0]])

B21 = (m_p * (l ** 2) + I) / denominator
B31 = m_p * l / denominator
B22 = B31
B32 = (m_c + m_p) / denominator

B = np.array([[0, 0],
              [0, 0],
              [B21, B22],
              [B31, B32]])

Q_x = np.diag([100, 10, 1, 1])
Q_theta = np.diag([1, 1, 100, 10])

R_constant = 0.001
R_one_player = np.array([[R_constant, 0],
                         [0, R_constant]])
R_x = np.array([[R_constant/10, 0],
                [0, R_constant]])
R_theta = np.array([R_constant/10])

# R = np.array([[R_constant, 0],
#               [0, R_constant]])


def simulate(K):
    u = lambda x: np.matmul(-K, (x - final))  # [[F],[M]]
    fig = plt.figure()
    ax = fig.add_subplot(111,
                         aspect='equal',
                         xlim=(-(initial[0] + 1), initial[0] + 1),
                         ylim=(-(L + 5), L + 5),
                         title="Inverted Pendulum Simulation")
    ax.grid()

    # animation parameters
    origin = (0, 0)
    dt = 0.01
    frames = 1000
    t_span = [0.0, frames * dt]

    ss = np.array([0., 0., 0., 0.])

    pendulumArm = lines.Line2D(xdata=origin,
                               ydata=origin,
                               color='r')

    cart = patches.Rectangle(xy=origin,
                             width=0.5,
                             height=0.15,
                             color='b')

    def stateSpace(_, i):
        # returns state space model evaluated at initial conditions
        x, theta, x_dot, theta_dot = i

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        D = (m_p * l * cos_theta) ** 2 - (m_c + m_p) * (m_p * (l ** 2) + I)

        u_i = u(i)
        F = u_i[0, 0]
        M = u_i[0, 1]

        # nonlinear system
        ss[0] = x_dot
        ss[1] = theta_dot
        ss[2] = (1 / D) * (b * (m_p * (l ** 2) + I) * x_dot
                           - m_p * l * (theta_dot ** 2) * sin_theta * (m_p * (l ** 2) + I)
                           - (m_p ** 2) * g * (l ** 2) * cos_theta * sin_theta
                           - (m_p * (l ** 2) + I) * F
                           + m_p * l * M)
        ss[3] = (1 / D) * (- b * m_p * l * cos_theta * x_dot
                           + (m_p ** 2) * (l ** 2) * (theta_dot ** 2) * sin_theta * cos_theta
                           + (m_c + m_p) * m_p * g * l * sin_theta
                           + m_p * l * cos_theta * F
                           - (m_c + m_p) * M)

        return ss

    def init():
        ax.add_patch(cart)
        ax.add_line(pendulumArm)

        return pendulumArm, cart

    def animate(i):
        xPos = pendulum_state.y[0][i]
        theta = pendulum_state.y[1][i]
        x = [origin[0] + xPos, origin[0] + xPos + L * np.sin(theta)]
        y = [origin[1], origin[1] - L * np.cos(theta)]
        pendulumArm.set_xdata(x)
        pendulumArm.set_ydata(y)
        cartPos = [origin[0] + xPos - cart.get_width() / 2, origin[1] - cart.get_height()]
        cart.set_xy(cartPos)

        return pendulumArm, cart

    ts = np.linspace(t_span[0], t_span[1], frames)
    pendulum_state = solve_ivp(fun=stateSpace,
                               t_span=t_span,
                               y0=initial,
                               t_eval=ts,
                               rtol=1e-8
                               )

    t0 = time()
    animate(0)  # sample time required to evaluate animate() function
    t1 = time()
    interval = 1000 * dt - (t1 - t0)

    anim = animation.FuncAnimation(fig=fig,
                                   func=animate,
                                   init_func=init,
                                   frames=frames,
                                   interval=interval,
                                   blit=True
                                   )
    plt.show()


T_f = 40

one_player_game = ContinuousPyDiffGame(A=A,
                                       B=[B],
                                       Q=[(Q_x + Q_theta) / 2],
                                       R=[R_one_player],
                                       x_0=initial,
                                       x_T=final,
                                       T_f=T_f
                                       )
one_player_game.solve_game_and_simulate_state_space()
# simulate(one_player_game.K[0])
M2 = np.array([[1],
               [1]])

for i in range(1, 10):
    M1 = np.array([[i, 0],
                   [0, i]])

    two_player_game = ContinuousPyDiffGame(A=A,
                                           B=[B @ M1,
                                              B @ M2],
                                           Q=[Q_x, Q_theta],
                                           R=[R_x, R_theta],
                                           x_0=initial,
                                           x_T=final,
                                           T_f=T_f
                                           )
    two_player_game.solve_game_and_simulate_state_space()
# simulate(two_player_game.K)
