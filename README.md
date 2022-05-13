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
The method relies on the formulation given in the article "_Composition of Dynamic Control Objectives Based on Differential Games_" 
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
import numpy as np
from numpy import pi
from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame

g = 9.81


class InvertedPendulum(ContinuousPyDiffGame):
    def __init__(self,
                 m_c: float,
                 m_p: float,
                 p_L: float,
                 q_s: float = 1,
                 q_m: float = 100,
                 q_l: float = 10000,
                 r: float = 1,
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 epsilon: float = PyDiffGame._epsilon_default,
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

        state_variables_names = ['x', '\\theta', '\\dot{x}', '\\dot{\\theta}']

        super().__init__(A=A,
                         B=[B_x, B_theta] if multi else B,
                         Q=[Q_x, Q_theta] if multi else (Q_x + Q_theta) / 2,
                         R=[np.array([r * 1000])] * 2 if multi else np.diag([r, r]),
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         state_variables_names=state_variables_names,
                         epsilon=epsilon,
                         L=L,
                         force_finite_horizon=T_f is not None
                         )

    def run_simulation(self, plot_state_space: bool = True):
        super(InvertedPendulum, self).run_simulation(plot_state_space)


x_0_1 = np.array([20,  # x
                  pi / 3,  # theta
                  0,  # x_dot
                  0]  # theta_dot
                 )
x_T_1 = np.array([-3,  # x
                  pi / 4,  # theta
                  10,  # x_dot
                  5]  # theta_dot
                 )

m_c_i, m_p_i, p_L_i = 50, 8, 3
lqr_inverted_pendulum = InvertedPendulum(m_c=m_c_i,
                                         m_p=m_p_i,
                                         p_L=p_L_i,
                                         x_0=x_0_1,
                                         x_T=x_T_1,
                                         regular_LQR=True,
                                         epsilon=10**(-5)
                                         )
lqr_inverted_pendulum.run_simulation(plot_state_space=False)
game_inverted_pendulum = InvertedPendulum(m_c=m_c_i,
                                          m_p=m_p_i,
                                          p_L=p_L_i,
                                          x_0=x_0_1,
                                          x_T=x_T_1,
                                          epsilon=10 ** (-5)
                                          )
game_inverted_pendulum.run_simulation(plot_state_space=False)
lqr_inverted_pendulum.plot_two_state_spaces(game_inverted_pendulum)

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
    <img src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/Logo_ISTRC_Green_English.png" width="180" />
    </a>
&emsp;
&emsp;
&emsp;
&emsp;
<a href="https://in.bgu.ac.il/en/Pages/default.aspx">
<img src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/BGU-logo-round.png" width="150" />
</a>
&emsp;
&emsp;
&emsp;
&emsp;
<a href="https://in.bgu.ac.il/en/robotics/Pages/default.aspx">
<img src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/logo_abc.png" width="180" />
</a>
</p>
