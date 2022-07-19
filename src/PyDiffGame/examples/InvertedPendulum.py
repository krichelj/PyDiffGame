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


class InvertedPendulum(PyDiffGameComparison):
    __q_s_default: Final[ClassVar[float]] = 1
    __q_m_default: Final[ClassVar[float]] = 100 * __q_s_default
    __q_l_default: Final[ClassVar[float]] = 100 * __q_m_default
    __r_default: Final[ClassVar[float]] = 1

    def __init__(self,
                 m_c: float,
                 m_p: float,
                 p_L: float,
                 q_s: Optional[float] = __q_s_default,
                 q_m: Optional[float] = __q_m_default,
                 q_l: Optional[float] = __q_l_default,
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
        self.__I = 1 / 12 * self.__m_p * self.__p_L ** 2

        # # original linear system
        linearized_D = self.__m_c * self.__m_p * self.__l ** 2 + self.__I * (self.__m_c + self.__m_p)
        a21 = self.__m_p ** 2 * PyDiffGame.g * self.__l ** 2 / linearized_D
        a31 = self.__m_p * PyDiffGame.g * self.__l * (self.__m_c + self.__m_p) / linearized_D

        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, a21, 0, 0],
                      [0, a31, 0, 0]])

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

        Q_x = np.diag([q_l, q_s, q_m, q_s])
        Q_theta = np.diag([q_s, q_l, q_s, q_m])
        Q_lqr = (Q_x + Q_theta) / 2
        Qs = [Q_x, Q_theta]

        R_lqr = np.diag([r, r])
        Rs = [np.array([r * 1000])] * 2

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

        lqr_objective = LQRObjective(Q=Q_lqr, R_ii=R_lqr)
        game_objectives = [GameObjective(Q=Q, R_ii=R, M_i=M_i) for Q, R, M_i in zip(Qs, Rs, Ms)]
        objectives = [lqr_objective] + game_objectives

        super().__init__(continuous=True,
                         args=args,
                         objectives=objectives)

        self.__lqr, self.__game = self._games

    def __get_next_non_linear_x(self, x_t: np.array, F_t: float, M_t: float) -> np.array:
        x, theta, x_dot, theta_dot = x_t
        masses = self.__m_p + self.__m_c
        m_p_l = self.__m_p * self.__l
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        g_sin = PyDiffGame.g * sin_theta
        sin_dot_sq = sin_theta * (theta_dot ** 2)

        theta_ddot = 1 / (m_p_l * self.__l + self.__I - (m_p_l ** 2) * (cos_theta ** 2) / masses) * \
                     (M_t - m_p_l * (cos_theta / masses * (F_t + m_p_l * sin_dot_sq) + g_sin))
        x_ddot = 1 / masses * (F_t + m_p_l * (sin_dot_sq - cos_theta * theta_ddot))

        non_linear_x = np.array([x_dot, theta_dot, x_ddot, theta_ddot], dtype=float)

        return non_linear_x

    def __simulate_lqr_non_linear_system(self) -> np.array:
        K = self.__lqr.K[0]

        def stateSpace(_, x_t: np.array) -> np.array:
            u_t = - K @ (x_t - self.__game.x_T)
            F_t, M_t = u_t.T
            return self.__get_next_non_linear_x(x_t, F_t, M_t)

        pendulum_state = solve_ivp(fun=stateSpace,
                                   t_span=[0.0, self.__game.T_f],
                                   y0=self.__game.x_0,
                                   t_eval=self.__game.forward_time,
                                   rtol=1e-8)

        Y = pendulum_state.y
        self.__game.plot_temporal_state_variables(state_variables=Y.T,
                                                  linear_system=False)

        return Y

    def __simulate_game_non_linear_system(self) -> np.array:
        K_x, k_theta = self.__game.K

        def stateSpace(_, x_t: np.array) -> np.array:
            x_curr = x_t - self.__game.x_T
            v_x = - K_x[0] @ x_curr
            v_theta = - k_theta[0] @ x_curr
            F_t, M_t = self.__game.M_inv @ np.array([[v_x], [v_theta]])

            return self.__get_next_non_linear_x(x_t, F_t, M_t)

        pendulum_state = solve_ivp(fun=stateSpace,
                                   t_span=[0.0, self.__game.T_f],
                                   y0=self.__game.x_0,
                                   t_eval=self.__game.forward_time,
                                   rtol=1e-8)

        Y = pendulum_state.y
        self.__game.plot_temporal_state_variables(state_variables=Y.T,
                                                  linear_system=False)

        return Y

    def __run_animation(self) -> (Line2D, Rectangle):
        pend_x, pend_theta, pend_x_dot, pend_theta_dot = self.__simulate_game_non_linear_system()

        pendulumArm = Line2D(xdata=self.__origin,
                             ydata=self.__origin,
                             color='r')
        cart = Rectangle(xy=self.__origin,
                         width=0.5,
                         height=0.15,
                         color='b')

        fig = plt.figure()
        ax = fig.add_subplot(111,
                             aspect='equal',
                             xlim=(-10, 10),
                             ylim=(-5, 5),
                             title="Inverted LQR Pendulum Simulation")

        def init() -> (Line2D, Rectangle):
            ax.add_patch(cart)
            ax.add_line(pendulumArm)

            return pendulumArm, cart

        def animate(i: int) -> (Line2D, Rectangle):
            x_i, theta_i = pend_x[i], pend_theta[i]
            pendulum_x_coordinates = [x_i, x_i + self.__p_L * sin(theta_i)]
            pendulum_y_coordinates = [0, - self.__p_L * cos(theta_i)]
            pendulumArm.set_xdata(pendulum_x_coordinates)
            pendulumArm.set_ydata(pendulum_y_coordinates)

            cart_x_y = [x_i - cart.get_width() / 2, - cart.get_height()]
            cart.set_xy(cart_x_y)

            return pendulumArm, cart

        ax.grid()
        t0 = time()
        animate(0)
        t1 = time()
        interval = self.__game.T_f - (t1 - t0)

        anim = FuncAnimation(fig=fig,
                             func=animate,
                             init_func=init,
                             frames=self.__game.L,
                             interval=interval,
                             blit=True)
        plt.show()

    def run_simulations(self, plot_state_space: Optional[bool] = True, run_animation: Optional[bool] = True):
        super().run_simulations(plot_state_space=plot_state_space)

        if run_animation:
            self.__run_animation()


x_0 = np.array([0,  # x
                pi / 3,  # theta
                2,  # x_dot
                4]  # theta_dot
               )
x_T = np.array([7,  # x
                pi / 4,  # theta
                2,  # x_dot
                6]  # theta_dot
               )

m_c_i, m_p_i, p_L_i = 50, 8, 3
epsilon = 10 ** (-5)

inverted_pendulum = InvertedPendulum(m_c=m_c_i,
                                     m_p=m_p_i,
                                     p_L=p_L_i,
                                     x_0=x_0,
                                     x_T=x_T,
                                     epsilon=epsilon
                                     )
inverted_pendulum.run_simulations()
