# PyDiffGame
  * [What is this?](#what-is-this)
  * [Local Installation](#local-installation)
  * [Required Input](#input-Parameters)

## Additional Information
  * [Mathematical Description](Math.md)
  * [Supported Scenarios](Scenarios.md)
  * [References](Math.md#references)

## What is this?
`PyDiffGame` is a Python package that allows to solve equations arising in models that involve the use of differential 
games for finding optimal controllers of a multi-objective system with multiple controlled objects.

## Local Installation
To clone Git repository locally run this from the command prompt:
```
git clone https://github.com/krichelj/PyDiffGame.git
```

## Input Parameters

The package a file named `PyDiffGame.py`. The main function is `solve_diff_game`.
All the constants are defined in the [Mathematical Description](Math.md) section.
The input parameters for `solve_diff_game` are:

* `A` : numpy 2-d array, of shape(<img src="https://render.githubusercontent.com/render/math?math=M,M">)
>The system dynamics matrix
* `B` : list of numpy 2-d arrays, of len(<img src="https://render.githubusercontent.com/render/math?math=N">), each matrix <img src="https://render.githubusercontent.com/render/math?math=B_j"> of shape(<img src="https://render.githubusercontent.com/render/math?math=M,k_j">), <img src="https://render.githubusercontent.com/render/math?math=j=1...N">
>System input matrices for each control agenda
* `Q` : list of numpy 2-d arrays, of len(<img src="https://render.githubusercontent.com/render/math?math=N">), each matrix <img src="https://render.githubusercontent.com/render/math?math=Q_j"> of shape(<img src="https://render.githubusercontent.com/render/math?math=M,M">), <img src="https://render.githubusercontent.com/render/math?math=j=1...N">
>Cost function state weights for each control agenda
* `R` : list of numpy 2-d arrays, of len(<img src="https://render.githubusercontent.com/render/math?math=N">), each matrix <img src="https://render.githubusercontent.com/render/math?math=R_{j}"> of shape(<img src="https://render.githubusercontent.com/render/math?math=k_j,k_j">), <img src="https://render.githubusercontent.com/render/math?math=j=1...N">
>Cost function input weights for each control agenda
* `cl` : boolean
>Indicates whether to render the closed (True) or open (False) loop behaviour
* `T_f` : positive float, optional, default = 5
>System dynamics horizon. Should be given in the case of finite horizon
* `X_0` : numpy 2-d array, of shape(<img src="https://render.githubusercontent.com/render/math?math=M">), optional
>Initial state vector
* `P_f` : list of numpy 2-d arrays, of shape(<img src="https://render.githubusercontent.com/render/math?math=M, M">), optional, default = solution of scipy's `solve_continuous_are`
>Final condition for the Riccati equation matrix. Should be given in the case of finite horizon
* `data_points` : positive int, optional, default = 1000
>Number of data points to evaluate the Riccati equation matrix. Should be given in the case of finite horizon

## Tutorial

Let us consider the following input parameters and invocation of the `solve_diff_game` function:

```python
import numpy as np
from PyDiffGame import solve_diff_game

A = np.array([[-2, 1],
                  [1, 4]])
B = [np.array([[1, 0],
               [0, 1]]),
     np.array([[0],
               [1]]),
     np.array([[1],
               [0]])]
Q = [np.array([[1, 0],
               [0, 1]]),
     np.array([[1, 0],
               [0, 10]]),
     np.array([[10, 0],
               [0, 1]])
     ]
R = [np.array([[100, 0],
               [0, 200]]),
     np.array([[5]]),
     np.array([[7]])]

cl = False
X0 = np.array([10, 20])
T_f = 5
P_f = [np.array([[10, 0],
                 [0, 20]]),
       np.array([[30, 0],
                 [0, 40]]),
       np.array([[50, 0],
                 [0, 60]])]
data_points = 1000
show_legend = True

P = solve_diff_game(A=A, B=B, Q=Q, R=R, cl=cl, X0=X0, T_f=T_f, P_f=P_f, data_points=data_points, 
                        show_legend=show_legend) # the function returns P
```

This will result in the following plot for the Riccati equation solution:

<img src="https://github.com/krichelj/PyDiffGame/blob/master/images/tut1_riccati.png" width="640" alt="EKF pic">

and the following state space simulation:

<img src="https://github.com/krichelj/PyDiffGame/blob/master/images/tut1_state.png" width="640" alt="EKF pic">



