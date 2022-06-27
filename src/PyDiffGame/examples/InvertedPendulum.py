import numpy as np
from time import time
from numpy import pi, sin, cos
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame


class InvertedPendulum(ContinuousPyDiffGame):
    __q_s_default = 1
    __q_m_default = 100 * __q_s_default
    __q_l_default = 100 * __q_m_default
    __r_default = 1

    def __init__(self,
                 m_c: float,
                 m_p: float,
                 p_L: float,
                 q_s: float = __q_s_default,
                 q_m: float = __q_m_default,
                 q_l: float = __q_l_default,
                 r: float = __r_default,
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 epsilon: float = PyDiffGame._epsilon_default,
                 L: int = PyDiffGame._L_default,
                 regular_LQR: bool = False,
                 show_animation: bool = True
                 ):
        self.__m_c = m_c
        self.__m_p = m_p
        self.__p_L = p_L
        self.__l = self.__p_L / 2  # CoM of uniform rod
        self.__I = 1 / 12 * self.__m_p * self.__p_L ** 2

        # # original linear system
        linearized_D = self.__m_c * self.__m_p * self.__l ** 2 + self.__I * (self.__m_c + self.__m_p)
        a21 = self.__m_p ** 2 * PyDiffGame._g * self.__l ** 2 / linearized_D
        a31 = self.__m_p * PyDiffGame._g * self.__l * (self.__m_c + self.__m_p) / linearized_D

        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, a21, 0, 0],
                      [0, a31, 0, 0]])

        b21 = (m_p * self.__l ** 2 + self.__I) / linearized_D
        b31 = m_p * self.__l / linearized_D
        b22 = b31
        b32 = (m_c + m_p) / linearized_D

        B_x = np.array([[0],
                        [0],
                        [1],
                        [0]])
        B_theta = np.array([[0],
                            [0],
                            [0],
                            [1]])
        B_lqr = np.array([[0, 0],
                          [0, 0],
                          [b21, b22],
                          [b31, b32]])
        B_multiplayer = [B_x, B_theta]

        Q_x = np.diag([q_l, q_s, q_m, q_s])
        Q_theta = np.diag([q_s, q_l, q_s, q_m])
        Q_lqr = (Q_x + Q_theta) / 2
        Q_multiplayer = [Q_x, Q_theta]

        R_lqr = np.diag([r, r])
        R_multiplayer = [np.array([r * 1000])] * 2

        self.__origin = (0.0, 0.0)
        self.__show_animation = show_animation

        state_variables_names = ['x', '\\theta', '\\dot{x}', '\\dot{\\theta}']

        super().__init__(A=A,
                         B=B_lqr if regular_LQR else B_multiplayer,
                         Q=Q_lqr if regular_LQR else Q_multiplayer,
                         R=R_lqr if regular_LQR else R_multiplayer,
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         state_variables_names=state_variables_names,
                         epsilon=epsilon,
                         L=L,
                         force_finite_horizon=T_f is not None
                         )

    def __simulate_non_linear_system(self) -> np.array:
        K = self._K[0]

        def stateSpace(_, x_t: np.array) -> np.array:
            u_t = - K @ (x_t - self._x_T)
            F_t, M_t = u_t.T
            x, theta, x_dot, theta_dot = x_t
            masses = self.__m_p + self.__m_c
            m_p_l = self.__m_p * self.__l
            sin_theta = sin(theta)
            cos_theta = cos(theta)
            g_sin = PyDiffGame._g * sin_theta
            sin_dot_sq = sin_theta * (theta_dot ** 2)
            theta_ddot = 1 / (m_p_l * self.__l + self.__I - (m_p_l ** 2) * (cos_theta ** 2) / masses) * \
                             (M_t - m_p_l * (cos_theta / masses * (F_t + m_p_l * sin_dot_sq) + g_sin))
            x_ddot = 1 / masses * (F_t + m_p_l * (sin_dot_sq - cos_theta * theta_ddot))

            return np.array([x_dot, theta_dot, x_ddot, theta_ddot])

        pendulum_state = solve_ivp(fun=stateSpace,
                                   t_span=[0.0, self._T_f],
                                   y0=self._x_0,
                                   t_eval=self._forward_time,
                                   rtol=1e-8)

        Y = pendulum_state.y
        self._plot_temporal_state_variables(state_variables=Y.T,
                                            linear_system=False)

        return Y

    def __run_animation(self):
        pend_x, pend_theta, pend_x_dot, pend_theta_dot = self.__simulate_non_linear_system()

        pendulumArm = Line2D(xdata=self.__origin,
                             ydata=self.__origin,
                             color='r')
        cart = Rectangle(xy=self.__origin,
                         width=0.5,
                         height=0.15,
                         color='b')

        if self._x_T is None:
            x_limits = (-5, 5)
        else:
            x = np.max(np.abs(self._x[:, 0]))
            x_limits = (-x, x)

        fig = plt.figure()
        ax = fig.add_subplot(111,
                             aspect='equal',
                             xlim=(-10, 10),
                             ylim=(-5, 5),
                             title="Inverted Pendulum Simulation")

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
        interval = self._T_f - (t1 - t0)

        anim = FuncAnimation(fig=fig,
                             func=animate,
                             init_func=init,
                             frames=self._L,
                             interval=interval,
                             blit=True)
        plt.show()

    def run_simulation(self, plot_state_space: bool = True):
        super(InvertedPendulum, self).run_simulation(plot_state_space)

        if self.__show_animation:
            self.__run_animation()

        self.__simulate_non_linear_system()


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

lqr_inverted_pendulum = InvertedPendulum(m_c=m_c_i,
                                         m_p=m_p_i,
                                         p_L=p_L_i,
                                         x_0=x_0,
                                         x_T=x_T,
                                         regular_LQR=True,
                                         show_animation=False,
                                         epsilon=epsilon
                                         )
lqr_inverted_pendulum.run_simulation(plot_state_space=True)

# game_inverted_pendulum = InvertedPendulum(m_c=m_c_i,
#                                           m_p=m_p_i,
#                                           p_L=p_L_i,
#                                           x_0=x_0,
#                                           x_T=x_T,
#                                           show_animation=False,
#                                           epsilon=epsilon
#                                           )
# game_inverted_pendulum.run_simulation(plot_state_space=False)
# lqr_inverted_pendulum.plot_two_state_spaces(game_inverted_pendulum)
