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
from typing import Optional

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.PyDiffGameComparison import PyDiffGameComparison
from PyDiffGame.Objective import GameObjective, LQRObjective


class InvertedPendulumComparison(PyDiffGameComparison):
    def __init__(self,
                 m_c: float,
                 m_p: float,
                 p_L: float,
                 q: float,
                 r: Optional[float] = 1,
                 x_0: Optional[np.array] = None,
                 x_T: Optional[np.array] = None,
                 T_f: Optional[float] = None,
                 epsilon: Optional[float] = PyDiffGame.epsilon_x_default,
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

    def __run_animation(self,
                        i: int) -> (Line2D, Rectangle):
        game = self._games[i]
        game._x_non_linear = self.__simulate_non_linear_system(i=i,
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


def multiprocess_worker_function(x_T: float,
                                 theta_0: float,
                                 m_c: float,
                                 m_p: float,
                                 p_L: float,
                                 q: float,
                                 epsilon: float) -> int:
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
                                   epsilon=epsilon)  # game class
    is_max_lqr = \
        inverted_pendulum_comparison(plot_state_spaces=False,
                                     run_animations=False,
                                     print_costs=True,
                                     non_linear_costs=True,
                                     agnostic_costs=True)
    return int(is_max_lqr)
    # inverted_pendulum_comparison.plot_two_state_spaces(non_linear=True)


if __name__ == '__main__':
    x_Ts = [10 ** p for p in [2]]
    theta_Ts = [pi / 2 + pi / n for n in [10]]
    m_cs = [10 ** p for p in [1, 2]]
    m_ps = [10 ** p for p in [0, 1, 2]]
    p_Ls = [10 ** p for p in [0, 1]]
    qs = [10 ** p for p in [-2, -1, 0, 1]]
    epsilons = [10 ** (-3)]
    params = [x_Ts, theta_Ts, m_cs, m_ps, p_Ls, qs, epsilons]

    PyDiffGameComparison.run_multiprocess(multiprocess_worker_function=multiprocess_worker_function,
                                          values=params)

```

This will result in the following plot that compares the two systems performance:

<p align="center">
    <img alt="Logo" src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/tut.png"/>
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
