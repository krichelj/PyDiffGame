<p align="center">
    <img alt="Logo" src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/logo.png"/>
</p>


  * [What is this?](#what-is-this)
  * [Local Installation](#local-installation)
  * [Input Parameters](#input-Parameters)

# Additional Information
  * [Mathematical Description](Math.md)
  * [Supported Scenarios](Scenarios.md)
  * [References](Math.md#references)

# What is this?
`PyDiffGame` is a Python implementation of a multi-objective control systems simulator based on Nash Equilibrium differential game. 
The method relies on the formulation given in:

 - The thesis work "_Differential Games for Compositional Handling of Competing Control Tasks_"
   ([Research Gate](https://www.researchgate.net/publication/359819808_Differential_Games_for_Compositional_Handling_of_Competing_Control_Tasks))

 - The conference article "_Composition of Dynamic Control Objectives Based on Differential Games_" 
([IEEE](https://ieeexplore.ieee.org/document/9480269) | 
[Research Gate](https://www.researchgate.net/publication/353452024_Composition_of_Dynamic_Control_Objectives_Based_on_Differential_Games))

# Local Installation
To clone Git repository locally run this from the command prompt:
```
git clone https://github.com/krichelj/PyDiffGame.git
```


# Input Parameters

The package contains a file named `PyDiffGame.py` and an abstract class of the same name. An object of this class represents an instance of differential game. 
Once the object is created, it can be simulated using the `run_simulation` class method.
All the constants are defined in the [Mathematical Description](Math.md) section.
The input parameters to instantiate a `PyDiffGame` object are:

* `A` : 2-d `np.array` of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}n,n">)
>The system dynamics matrix
* `B` : `list` of 2-d `np.array` objects of len(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}N">), each array <img src="https://render.githubusercontent.com/render/math?math=\color{yellow}B_i"> of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}n, m_i">)
>System input matrices for each control objective
* `Q` : `list` of 2-d `np.array` objects of len(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}N">), each array <img src="https://render.githubusercontent.com/render/math?math=\color{yellow}Q_i"> of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}n, n">)
>Cost function state weights for each control objective
* `R` : `list` of 2-d `np.array` objects of len(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}N">), each array <img src="https://render.githubusercontent.com/render/math?math=\color{yellow}R_{i}"> of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}m_i, m_i">)
>Cost function input weights for each control objective
* `x_0` : 1-d `np.array` of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}n">), optional
>Initial state vector
* `x_T` : 1-d `np.array` of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}n">), optional
>Final state vector, in case of signal tracking
* `T_f` : positive `float`, optional, default = `10`
>System dynamics horizon. Should be given in the case of finite horizon
* `P_f` : `list` of 2-d `np.array` objects of len(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}N">), each array <img src="https://render.githubusercontent.com/render/math?math=\color{yellow}P_{f_i}"> of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}n,n">), optional, default = uncoupled solution of `scipy's solve_are`
>
>Final condition for the Riccati equation array. Should be given in the case of finite horizon
* `show_legend` : `boolean`, optional, default = `True`
>Indicates whether to display a legend in the plots
* `state_variables_names` : `list` of `str` objects of len(<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}n">), optional
>The state variables' names to display
* `epsilon` : `float` in the interval <img src="https://render.githubusercontent.com/render/math?math=\color{yellow}(0,1)">, optional, default = `10 ** (-7)`
>Numerical convergence threshold
* `L` : positive `int`, optional, default = `1000`
>Number of data points
* `eta` : positive `int`, optional, default = `5`
>The number of last matrix norms to consider for convergence
* `debug` : `boolean`, optional, default = `False`
>Indicates whether to display debug information

# Tutorial

This example is based on a 'Hello-World' example explained in the M.Sc. Thesis showcasing this work, which can be found [here](https://www.researchgate.net/publication/359819808_Differential_Games_for_Compositional_Handling_of_Competing_Control_Tasks).
Let us consider the following input parameters for the instantiation of an `InvertedPendulum` object and 
corresponding call for `run_simulation`:

```python
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

        theta_ddot = 1 / (m_p_l * self.__l + self.__I - (m_p_l ** 2) * (cos_theta ** 2) / masses) *
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
        self.__game.plot_state_variables(state_variables=Y.T,
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
        self.__game.plot_state_variables(state_variables=Y.T,
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

```

This will result in the following plot that compares the two systems performance:

<p align="center">
    <img alt="Logo" src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/200_15_1.png"/>
</p>

# Acknowledgments
This research was supported in part by the Helmsley Charitable Trust through the Agricultural, Biological and Cognitive Robotics Initiative and by the Marcus Endowment Fund both at Ben-Gurion University of the Negev, Israel.
This research was also supported by The Israeli Smart Transportation Research Center (ISTRC) by The Technion and Bar-Ilan Universities, Israel.

<p align="center">
    <a href="https://istrc.net.technion.ac.il/">
    <img src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/Logo_ISTRC_Green_English.png" width="180"  alt=""/>
    </a>
&emsp;
&emsp;
&emsp;
&emsp;
<a href="https://in.bgu.ac.il/en/Pages/default.aspx">
<img src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/BGU-logo-round.png" width="150"  alt=""/>
</a>
&emsp;
&emsp;
&emsp;
&emsp;
<a href="https://in.bgu.ac.il/en/robotics/Pages/default.aspx">
<img src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/logo_abc.png" width="180"  alt=""/>
</a>
</p>
