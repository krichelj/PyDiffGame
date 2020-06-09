# PyDiffGame
  * [What is this?](#what-is-this)
  * [Required Input](#input-Parameters)
  * [References](#references)
  
## Additional Information
  * [Mathematical Description](Math.md)
  * [Supported Scenarios](Scenarios.md)
  
## What is this?
`PyDiffGame` is a Python package that solves differential games for continuous-time control theory implementations.
This package allows to solve equations arising in models that involve the use of differential games
for finding optimal controllers of a multi-objective system with multiple controlled objects.

## Input Parameters

The package a file named `PyDiffGame.py`. The main function is `solve_diff_game`.
The input parameters for `solve_diff_game` are:

* `m` : list of positive ints, of len(<img src="https://render.githubusercontent.com/render/math?math=n">)
>Lengths of controlled objects' state vectors
* `A` : numpy 2-d array, of shape(<img src="https://render.githubusercontent.com/render/math?math=M,M">)
>The system dynamics matrix
* `B` : list of <img src="https://render.githubusercontent.com/render/math?math=j"> numpy 2-d arrays, of len(<img src="https://render.githubusercontent.com/render/math?math=N">), each matrix <img src="https://render.githubusercontent.com/render/math?math=B_j"> of shape(<img src="https://render.githubusercontent.com/render/math?math=M,k_j">)
>System input matrices for each control agenda
* `Q` : list of <img src="https://render.githubusercontent.com/render/math?math=j"> numpy 2-d arrays, of len(<img src="https://render.githubusercontent.com/render/math?math=N">), each matrix <img src="https://render.githubusercontent.com/render/math?math=Q_j"> of shape(<img src="https://render.githubusercontent.com/render/math?math=M,M">)
>Cost function state weights for each control agenda
* `R` : list of <img src="https://render.githubusercontent.com/render/math?math=j"> numpy 2-d arrays, of len(<img src="https://render.githubusercontent.com/render/math?math=N">), each matrix <img src="https://render.githubusercontent.com/render/math?math=R_{j}"> of shape(<img src="https://render.githubusercontent.com/render/math?math=k_j,k_j">)
>Cost function input weights for each control agenda
* `cl` : boolean
>Indicates whether to render the closed (True) or open (False) loop behaviour
* `T_f` : positive float <= 100, optional, default = 5
>System dynamics horizon. Should be given in the case of finite horizon
* `X_0` : numpy 2-d array, of shape(<img src="https://render.githubusercontent.com/render/math?math=M">)
>Initial state vector
* `P_f` : numpy 2-d array, of shape(<img src="https://render.githubusercontent.com/render/math?math=N \cdot M^2">), optional, default = solution of scipy's `solve_continuous_are`
>Final condition for the Riccati equation matrix. Should be given in the case of finite horizon
* `data_points` : positive int <= 10000, optional, default = 10000
>Number of data points to evaluate the Riccati equation matrix. Should be given in the case of finite horizon

## Tutorial

Let us consider the following input parameters:

```
m = [2]

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

cl = True
T_f = 5
P_f = np.array([10, 0, 0, 20, 30, 0, 0, 40, 50, 0, 0, 60])
data_points = 1000

X0 = np.array([10, 20])
```

Invoking the function `solve_diff_game` as such:
```
solve_diff_game(m, A, B, Q, R, cl, X0, T_f, P_f, data_points)
```

will result in the following plot for the Riccati equation solution:

<img src="https://github.com/krichelj/PyDiffGame/blob/master/images/tut1_riccati.png" width="640" alt="EKF pic">

and the following state space simulation:

<img src="https://github.com/krichelj/PyDiffGame/blob/master/images/tut1_state.png" width="640" alt="EKF pic">



