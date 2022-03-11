import numpy as np
from time import time
from numpy import pi, sin, cos
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame

g = 9.81


class InvertedPendulum(ContinuousPyDiffGame):

    def __init__(self,
                 m_c: float,
                 m_p: float,
                 p_L: float,
                 q_s: float = 100,
                 q_m: float = 100,
                 q_l: float = 1000,
                 r: float = 0.001,
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 L: int = PyDiffGame._L_default,
                 multiplayer: bool = True,
                 regular_LQR: bool = False
                 ):
        self.__m_c = m_c
        self.__m_p = m_p
        self.__p_L = p_L
        self.__l = self.__p_L / 2  # CoM of uniform rod
        self.__I = 1 / 12 * self.__m_p * self.__p_L ** 2

        # # original linear system
        linearized_D = self.__m_c * self.__m_p * self.__l ** 2 + self.__I * (self.__m_c + self.__m_p)
        a21 = self.__m_p ** 2 * g * self.__l ** 2 / linearized_D
        a31 = self.__m_p * g * self.__l * (self.__m_c + self.__m_p) / linearized_D

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

        Q_x = np.diag([q_l, q_s, q_m, q_s])
        Q_theta = np.diag([q_s, q_l, q_s, q_m])

        self.__origin = (0.0, 0.0)
        self.__regular_LQR = regular_LQR
        multi = multiplayer and not regular_LQR

        B_x = np.array([[0],
                        [0],
                        [1],
                        [0]])
        B_theta = np.array([[0],
                            [0],
                            [0],
                            [1]])

        super().__init__(A=A,
                         B=[B_x, B_theta] if multi else B,
                         Q=[Q_x, Q_theta] if multi else (Q_x + Q_theta) / 2,
                         R=[np.array([r * 1000])] * 2 if multi else np.diag([r, r]),
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         L=L
                         )

    def run_simulation(self):
        if self.__regular_LQR:
            P = np.array(solve_continuous_are(self._A,
                                              self._B[0],
                                              self._Q[0],
                                              self._R[0])
                         )
            K = np.array(np.linalg.inv(self._R[0]) @ self._B[0].T @ P)
        else:
            # self.solve_game_and_simulate_state_space()
            self.solve_game_simulate_state_space_and_plot()
            K = self._K[0]

        def stateSpace(_, x_t: np.array) -> np.array:
            u_t = - K @ (x_t - self._x_T)
            F_t, M_t = u_t.T
            x, theta, x_dot, theta_dot = x_t
            one_over_masses = 1 / (self.__m_p + self.__m_c)
            m_p_l = self.__m_p * self.__l
            sin_theta = sin(theta)
            cos_theta = cos(theta)
            g_sin = g * sin_theta
            sin_dot_sq = sin_theta * theta_dot ** 2
            non_linear_D = m_p_l * self.__l + self.__I - m_p_l ** 2 * cos_theta ** 2 * one_over_masses
            theta_ddot = 1 / non_linear_D * \
                         (M_t - m_p_l * (cos_theta * one_over_masses * (F_t + m_p_l * sin_dot_sq) + g_sin))
            x_ddot = one_over_masses * (F_t + m_p_l * (sin_dot_sq - cos_theta * theta_ddot))
            x_dot_t = np.array([x_dot, theta_dot, float(x_ddot), float(theta_ddot)])

            return x_dot_t

        pendulum_state = solve_ivp(fun=stateSpace,
                                   t_span=[0.0, self._T_f],
                                   y0=self._x_0,
                                   t_eval=self._forward_time,
                                   rtol=1e-8)
        Y = pendulum_state.y

        pendulumArm = lines.Line2D(xdata=self.__origin,
                                   ydata=self.__origin,
                                   color='r')
        cart = patches.Rectangle(xy=self.__origin,
                                 width=0.5,
                                 height=0.15,
                                 color='b')

        def init():
            ax.add_patch(cart)
            ax.add_line(pendulumArm)
            return pendulumArm, cart

        def animate(i):
            xPos = Y[0][i]
            theta = Y[1][i]
            x = [self.__origin[0] + xPos, self.__origin[0] + xPos + self.__p_L * sin(theta)]
            y = [self.__origin[1], self.__origin[1] - self.__p_L * np.cos(theta)]
            pendulumArm.set_xdata(x)
            pendulumArm.set_ydata(y)
            cartPos = [self.__origin[0] + xPos - cart.get_width() / 2, self.__origin[1] - cart.get_height()]
            cart.set_xy(cartPos)

            return pendulumArm, cart

        fig = plt.figure()
        ax = fig.add_subplot(111,
                             aspect='equal',
                             xlim=(-5, 5),
                             ylim=(-1, 1),
                             title="Inverted Pendulum Simulation")
        ax.grid()
        t0 = time()
        animate(0)
        t1 = time()
        interval = self._T_f - (t1 - t0)

        anim = animation.FuncAnimation(fig=fig,
                                       func=animate,
                                       init_func=init,
                                       frames=self._L,
                                       interval=interval,
                                       blit=True)
        plt.show()


x_0 = np.array([20,  # x
                pi / 3,  # theta
                0,  # x_dot
                0]  # theta_dot
               )
x_T = np.array([0,  # x
                pi,  # theta
                0,  # x_dot
                0]  # theta_dot
               )

ip = InvertedPendulum(m_c=10,
                      m_p=30,
                      p_L=0.7,
                      x_0=x_0,
                      x_T=x_T,
                      # regular_LQR=True
                      )
ip.run_simulation()
