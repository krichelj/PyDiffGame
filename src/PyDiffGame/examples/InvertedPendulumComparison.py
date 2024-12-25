from __future__ import annotations

import numpy as np
import scipy as sp
from time import time
import matplotlib
import matplotlib.pyplot as plt

from typing import Optional

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.PyDiffGameLQRComparison import PyDiffGameLQRComparison
from PyDiffGame.Objective import GameObjective, LQRObjective


class InvertedPendulumComparison(PyDiffGameLQRComparison):
    def __init__(self,
                 m_c: float,
                 m_p: float,
                 p_L: float,
                 q: float,
                 r: Optional[float] = 1,
                 x_0: Optional[np.array] = None,
                 x_T: Optional[np.array] = None,
                 T_f: Optional[float] = None,
                 epsilon_x: Optional[float] = PyDiffGame.epsilon_x_default,
                 epsilon_P: Optional[float] = PyDiffGame.epsilon_P_default,
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

        Q_x = q * np.array([[1, 0, 2, 0],
                            [0, 0, 0, 0],
                            [2, 0, 4, 0],
                            [0, 0, 0, 0]])
        Q_theta = q * np.array([[0, 0, 0, 0],
                                [0, 1, 0, 2],
                                [0, 0, 0, 0],
                                [0, 2, 0, 4]])
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
                'epsilon_x': epsilon_x,
                'epsilon_P': epsilon_P,
                'L': L,
                'eta': eta,
                'force_finite_horizon': T_f is not None}

        lqr_objective = [LQRObjective(Q=Q_lqr, R_ii=R_lqr)]
        game_objectives = [GameObjective(Q=Q, R_ii=R, M_i=M_i) for Q, R, M_i in zip(Qs, Rs, Ms)]
        games_objectives = [lqr_objective, game_objectives]

        super().__init__(args=args,
                         games_objectives=games_objectives,
                         continuous=True)

    def __simulate_non_linear_system(self,
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

            theta_ddot = 1 / (
                    self.__m_p * self.__l ** 2 + self.__I - (self.__m_p * self.__l) ** 2 * np.cos(theta) ** 2 /
                    (self.__m_p + self.__m_c)) * (M_t - self.__m_p * self.__l *
                                                  (np.cos(theta) / (self.__m_p + self.__m_c) *
                                                   (F_t + self.__m_p * self.__l * np.sin(theta)
                                                    * theta_dot ** 2) + PyDiffGame.g * np.sin(theta)))
            x_ddot = 1 / (self.__m_p + self.__m_c) * (F_t + self.__m_p * self.__l * (np.sin(theta) * theta_dot ** 2 -
                                                                                     np.cos(theta) * theta_ddot))
            if isinstance(theta_ddot, np.ndarray):
                theta_ddot = theta_ddot[0]
                x_ddot = x_ddot[0]

            non_linear_x = np.array([x_dot, theta_dot, x_ddot, theta_ddot],
                                    dtype=float)

            return non_linear_x

        pendulum_state = sp.integrate.solve_ivp(fun=nonlinear_state_space,
                                                t_span=[0.0, game.T_f],
                                                y0=game.x_0,
                                                t_eval=game.forward_time,
                                                rtol=game.epsilon)

        Y = pendulum_state.y

        if plot:
            game.plot_state_variables(state_variables=Y.T,
                                      linear_system=False)

        return Y

    def __run_animation(self,
                        i: int) -> (matplotlib.lines.Line2D, matplotlib.patches.Rectangle):
        game = self._games[i]
        game._x_non_linear = self.__simulate_non_linear_system(i=i,
                                                               plot=True)
        x_t, theta_t, x_dot_t, theta_dot_t = game._x_non_linear

        pendulumArm = matplotlib.lines.Line2D(xdata=self.__origin,
                                              ydata=self.__origin,
                                              color='r')
        cart = matplotlib.patches.Rectangle(xy=self.__origin,
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

        def init() -> (matplotlib.lines.Line2D, matplotlib.patches.Rectangle):
            ax.add_patch(cart)
            ax.add_line(pendulumArm)

            return pendulumArm, cart

        def animate(i: int) -> (matplotlib.lines.Line2D, matplotlib.patches.Rectangle):
            x_i, theta_i = x_t[i], theta_t[i]
            pendulum_x_coordinates = [x_i, x_i + self.__p_L * np.sin(theta_i)]
            pendulum_y_coordinates = [0, - self.__p_L * np.cos(theta_i)]
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

        anim = matplotlib.animationFuncAnimation(fig=fig,
                                                 func=animate,
                                                 init_func=init,
                                                 frames=frames,
                                                 interval=interval,
                                                 blit=True)
        plt.show()


def multiprocess_worker_function(x_T: float,
                                 theta_0: float,
                                 m_c: float,
                                 m_p: float,
                                 p_L: float,
                                 q: float,
                                 epsilon_x: float,
                                 epsilon_P: float) -> int:
    x_T = np.array([x_T,  # x
                    theta_0,  # theta
                    0,  # x_dot
                    0]  # theta_dot
                   )
    x_0 = np.zeros_like(x_T)

    inverted_pendulum_comparison = \
        InvertedPendulumComparison(m_c=m_c,
                                   m_p=m_p,
                                   p_L=p_L,
                                   q=q,
                                   x_0=x_0,
                                   x_T=x_T,
                                   epsilon_x=epsilon_x,
                                   epsilon_P=epsilon_P)  # game class
    is_max_lqr = \
        inverted_pendulum_comparison(plot_state_spaces=False,
                                     run_animations=False
                                     )

    # inverted_pendulum_comparison.plot_two_state_spaces(non_linear=True)
    return int(is_max_lqr)


if __name__ == '__main__':
    x_Ts = [10 ** p for p in [2]]
    theta_Ts = [np.pi / 2 + np.pi / n for n in [10]]
    m_cs = [10 ** p for p in [1, 2]]
    m_ps = [10 ** p for p in [0, 1, 2]]
    p_Ls = [10 ** p for p in [0, 1]]
    qs = [10 ** p for p in [-2, -1, 0, 1]]
    epsilon_xs = [10 ** (-7)]
    epsilon_Ps = [10 ** (-3)]
    params = [x_Ts, theta_Ts, m_cs, m_ps, p_Ls, qs, epsilon_xs, epsilon_Ps]

    PyDiffGameLQRComparison.run_multiprocess(multiprocess_worker_function=multiprocess_worker_function,
                                             values=params)
