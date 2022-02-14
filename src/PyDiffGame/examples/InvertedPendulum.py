import numpy as np
from time import time
from numpy import pi
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from scipy.integrate import solve_ivp
# from scipy.linalg import solve_continuous_are

from PyDiffGame import PyDiffGame
from ContinuousPyDiffGame import ContinuousPyDiffGame


class InvertedPendulum(ContinuousPyDiffGame):
    g = 9.81

    def __init__(self,
                 m_c: float,
                 m_p: float,
                 p_L: float,
                 q_s: float = 0.1,
                 q_m: float = 10,
                 q_l: float = 100,
                 r: float = 0.0001,
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 L: int = PyDiffGame._L_default,
                 multiplayer: bool = True
                 ):
        self.m_c = m_c
        self.m_p = m_p
        self.p_L = p_L
        self.l = self.p_L / 2  # CoM of uniform rod
        self.I = 1 / 12 * self.m_p * self.p_L ** 2

        # # original linear system
        D = self.m_c * self.m_p * self.l ** 2 + self.I * (self.m_c + self.m_p)
        a21 = self.m_p ** 2 * InvertedPendulum.g * self.l ** 2 / D
        a31 = self.m_p * InvertedPendulum.g * self.l * (self.m_c + self.m_p) / D

        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, a21, 0, 0],
                      [0, a31, 0, 0]])

        b21 = (m_p * self.l ** 2 + self.I) / D
        b31 = m_p * self.l / D
        b22 = b31
        b32 = (m_c + m_p) / D

        B = np.array([[0, 0],
                      [0, 0],
                      [b21, b22],
                      [b31, b32]])

        Q_x = np.diag([q_l, q_s, q_m, q_s])
        Q_theta = np.diag([q_s, q_l, q_s, q_m])
        R = np.diag([r, r])

        self.origin = (0.0, 0.0)
        # self.dt = 0.02
        # self.frames = 1000

        super().__init__(A=A,
                         B=B,
                         Q=[Q_x, Q_theta] if multiplayer else (Q_x + Q_theta) / 2,
                         R=R,
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         L=L
                         )
        self.t_span = [0.0, self._L * self._delta]

    def run_simulation(self):
        self.solve_game_and_simulate_state_space()

        ts = np.linspace(self.t_span[0], self.t_span[1], self._L)
        ss = np.zeros((4,))

        K = self.K[0]

        def stateSpace(_, x: np.array):
            u = - K @ (x - self._x_T)
            u_x, u_theta = u.T

            # returns state space model evaluated at initial conditions
            S = np.sin(x[1])
            C = np.cos(x[1])
            denominator = self.m_p * self.l * C ** 2 - (self.m_c + self.m_p) * (self.m_p * self.l ** 2 + self.I)

            # nonlinear system
            ss[0] = x[2]
            ss[1] = x[3]
            ss[2] = 1 / denominator * (
                    -self.m_p ** 2 * InvertedPendulum.g * self.l ** 2 * C * S - self.m_p * self.l * S *
                    (self.m_p * self.l ** 2 + self.I) * x[3] ** 2
            ) - (self.m_p * (self.l ** 2) + self.I) / denominator * u_x
            ss[3] = 1 / denominator * (
                    (self.m_c + self.m_p) * self.m_p * InvertedPendulum.g * self.l * S +
                    self.m_p ** 2 * self.l ** 2 * S * C * x[3] ** 2
            ) + self.m_p * self.l * C / denominator * u_theta

            return ss

        pendulum_state = solve_ivp(fun=stateSpace,
                                   t_span=self.t_span,
                                   y0=self._x_0,
                                   t_eval=ts,
                                   rtol=1e-8)
        Y = pendulum_state.y

        pendulumArm = lines.Line2D(xdata=self.origin, ydata=self.origin, color='r')
        cart = patches.Rectangle(xy=self.origin, width=0.5, height=0.15, color='b')

        def init():
            ax.add_patch(cart)
            ax.add_line(pendulumArm)
            return pendulumArm, cart

        def animate(i):
            xPos = Y[0][i]
            theta = Y[1][i]
            x = [self.origin[0] + xPos, self.origin[0] + xPos + self.p_L * np.sin(theta)]
            y = [self.origin[1], self.origin[1] - self.p_L * np.cos(theta)]
            pendulumArm.set_xdata(x)
            pendulumArm.set_ydata(y)
            cartPos = [self.origin[0] + xPos - cart.get_width() / 2, self.origin[1] - cart.get_height()]
            cart.set_xy(cartPos)

            return pendulumArm, cart

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', xlim=(-5, 5), ylim=(-1, 1), title="Inverted Pendulum Simulation")
        ax.grid()
        t0 = time()
        animate(0)  # sample time required to evaluate animate() function
        t1 = time()
        interval = self._L * self._delta - (t1 - t0)

        anim = animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=self._L, interval=interval,
                                       blit=True)
        plt.show()


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
ip = InvertedPendulum(m_c=11,
                      m_p=2,
                      p_L=0.5,
                      x_0=x_0,
                      x_T=x_T,
                      multiplayer=False)
ip.run_simulation()
