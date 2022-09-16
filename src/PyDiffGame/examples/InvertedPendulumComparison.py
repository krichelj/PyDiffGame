from __future__ import annotations

import numpy as np
from time import time
from numpy import pi, sin, cos
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from typing import Final, ClassVar, Optional

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.PyDiffGameComparison import PyDiffGameComparison
from PyDiffGame.Objective import GameObjective, LQRObjective


class InvertedPendulumComparison(PyDiffGameComparison):
    __q_default: Final[ClassVar[float]] = 50
    __r_default: Final[ClassVar[float]] = 10

    def __init__(self,
                 m_c: float,
                 m_p: float,
                 p_L: float,
                 q: Optional[float] = __q_default,
                 r: Optional[float] = __r_default,
                 x_0: Optional[np.array] = None,
                 x_T: Optional[np.array] = None,
                 T_f: Optional[float] = None,
                 epsilon: Optional[float] = PyDiffGame.epsilon_default,
                 L: Optional[int] = PyDiffGame.L_default,
                 eta: Optional[int] = PyDiffGame.eta_default):
        self.__m_c = m_c
        self.__m_p = m_p
        self.__p_L = p_L
        self.__l = self.__p_L / 2  # CoM of uniform rod
        self.__I = 1 / 12 * self.__m_p * self.__p_L ** 2  # center mass moment of inertia of uniform rod

        # # original linear system
        linearized_D = self.__m_c * self.__m_p * self.__l ** 2 + self.__I * (self.__m_c + self.__m_p)
        a32 = self.__m_p * PyDiffGame.g * self.__l ** 2 / linearized_D
        a42 = self.__m_p * PyDiffGame.g * self.__l * (self.__m_c + self.__m_p) / linearized_D

        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, a32, 0, 0],
                      [0, a42, 0, 0]])

        b21 = (m_p * self.__l ** 2 + self.__I) / linearized_D
        b31 = m_p * self.__l / linearized_D
        b22 = b31
        b32 = (m_c + m_p) / linearized_D

        B = np.array([[0, 0],
                      [0, 0],
                      [b21, b22],
                      [b31, b32]])
        M1 = B[2, :].reshape(1, 2)
        M2 = B[3, :].reshape(1, 2)
        Ms = [M1, M2]
        Q_x = np.array([[q, 0, 2 * q, 0],
                        [0, 0, 0, 0],
                        [2 * q, 0, 3 * q, 0],
                        [0, 0, 0, 0]])
        Q_theta = np.array([[0, 0, 0, 0],
                            [0, q, 0, 2 * q],
                            [0, 0, 0, 0],
                            [0, 2 * q, 0, 3 * q]])
        Q_lqr = Q_theta + Q_x
        Qs = [Q_x, Q_theta]

        R_lqr = np.diag([r] * 2)

        Rs = [np.array([r])] * 2

        self.__origin = (0.0, 0.0)

        state_variables_names = ['x',
                                 '\\theta',
                                 '\\dot{x}',
                                 '\\dot{\\theta}']

        args = {'A': A,
                'B': B,
                'x_0': x_0,
                'x_T': x_T,
                'T_f': T_f,
                'state_variables_names': state_variables_names,
                'epsilon': epsilon,
                'L': L,
                'eta': eta,
                'force_finite_horizon': T_f is not None}

        lqr_objective = [LQRObjective(Q=Q_lqr, R_ii=R_lqr)]
        game_objectives = [GameObjective(Q=Q, R_ii=R, M_i=M_i) for Q, R, M_i in zip(Qs, Rs, Ms)]
        games_objectives = [lqr_objective, game_objectives]

        super().__init__(args=args,
                         games_objectives=games_objectives,
                         continuous=True)

    def _simulate_non_linear_system(self,
                                    i: int,
                                    plot: bool = False) -> np.array:
        game = self._games[i]
        K = game.K
        x_T = game.x_T

        def nonlinear_state_space(_, x_t: np.array) -> np.array:
            x_t = x_t - x_T

            if game.is_LQR():
                u_t = - K[0] @ x_t
                F_t, M_t = u_t.T
            else:
                K_x, K_theta = K
                v_x = - K_x @ x_t
                v_theta = - K_theta @ x_t
                v = np.array([v_x, v_theta])
                F_t, M_t = game.M_inv @ v

            x, theta, x_dot, theta_dot = x_t

            theta_ddot = 1 / (self.__m_p * self.__l ** 2 + self.__I - (self.__m_p * self.__l) ** 2 * cos(theta) ** 2 /
                              (self.__m_p + self.__m_c)) * (M_t - self.__m_p * self.__l *
                                                            (cos(theta) / (self.__m_p + self.__m_c) *
                                                             (F_t + self.__m_p * self.__l * sin(theta)
                                                              * theta_dot ** 2) + PyDiffGame.g * sin(theta)))
            x_ddot = 1 / (self.__m_p + self.__m_c) * (F_t + self.__m_p * self.__l * (sin(theta) * theta_dot ** 2 -
                                                                                     cos(theta) * theta_ddot))
            if isinstance(theta_ddot, np.ndarray):
                theta_ddot = theta_ddot[0]
                x_ddot = x_ddot[0]

            non_linear_x = np.array([x_dot, theta_dot, x_ddot, theta_ddot],
                                    dtype=float)

            return non_linear_x

        pendulum_state = solve_ivp(fun=nonlinear_state_space,
                                   t_span=[0.0, game.T_f],
                                   y0=game.x_0,
                                   t_eval=game.forward_time,
                                   rtol=game.epsilon)

        Y = pendulum_state.y

        if plot:
            game.plot_state_variables(state_variables=Y.T,
                                      linear_system=False)

        return Y

    def run_animation(self,
                      i: int) -> (Line2D, Rectangle):
        game = self._games[i]
        game._x_non_linear = self._simulate_non_linear_system(i=i,
                                                              plot=True)
        x_t, theta_t, x_dot_t, theta_dot_t = game._x_non_linear

        pendulumArm = Line2D(xdata=self.__origin,
                             ydata=self.__origin,
                             color='r')
        cart = Rectangle(xy=self.__origin,
                         width=0.5,
                         height=0.15,
                         color='b')

        fig = plt.figure()
        x_max = max(abs(max(x_t)), abs(min(x_t)))
        square_side = 1.1 * min(max(self.__p_L, x_max), 3 * self.__p_L)

        ax = fig.add_subplot(111,
                             aspect='equal',
                             xlim=(-square_side, square_side),
                             ylim=(-square_side, square_side),
                             title=f"Inverted Pendulum {'LQR' if game.is_LQR() else 'Game'} Simulation")

        def init() -> (Line2D, Rectangle):
            ax.add_patch(cart)
            ax.add_line(pendulumArm)

            return pendulumArm, cart

        def animate(i: int) -> (Line2D, Rectangle):
            x_i, theta_i = x_t[i], theta_t[i]
            pendulum_x_coordinates = [x_i, x_i + self.__p_L * sin(theta_i)]
            pendulum_y_coordinates = [0, - self.__p_L * cos(theta_i)]
            pendulumArm.set_xdata(x=pendulum_x_coordinates)
            pendulumArm.set_ydata(y=pendulum_y_coordinates)

            cart_x_y = [x_i - cart.get_width() / 2, - cart.get_height()]
            cart.set_xy(xy=cart_x_y)

            return pendulumArm, cart

        ax.grid()
        t0 = time()
        animate(0)
        t1 = time()

        frames = game.L
        interval = game.T_f - (t1 - t0)

        anim = FuncAnimation(fig=fig,
                             func=animate,
                             init_func=init,
                             frames=frames,
                             interval=interval,
                             blit=True)
        plt.show()


x_0 = np.array([0,  # x
                0,  # theta
                0,  # x_dot
                0]  # theta_dot
               )
x_T = np.array([10,  # x
                pi,  # theta
                0,  # x_dot
                0]  # theta_dot
               )

m_c, m_p, p_L = 10, 5, 3
epsilon = 10 ** (-3)

inverted_pendulum = InvertedPendulumComparison(m_c=m_c,
                                               m_p=m_p,
                                               p_L=p_L,
                                               x_0=x_0,
                                               x_T=x_T,
                                               epsilon=epsilon)
inverted_pendulum.__call__(calculate_costs=True,
                           x_only_costs=True)
inverted_pendulum.plot_two_state_spaces(non_linear=True)
