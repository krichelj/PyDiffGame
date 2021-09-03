<p align="center">
    <img alt="Logo" src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/logo.png"/>
</p>

  * [What is this?](#what-is-this)
  * [Local Installation](#local-installation)
  * [Required Input](#input-Parameters)

## Additional Information
  * [Mathematical Description](Math.md)
  * [Supported Scenarios](Scenarios.md)
  * [References](Math.md#references)

## What is this?
`PyDiffGame` is a Python Multi-Objective Control Systems Simulator based on Nash Equilibrium Differential Game. 
Based on the article "_Composition of Dynamic Control Objectives Based on Differential Games_" 
([IEEE](https://ieeexplore.ieee.org/document/9480269) | 
[Research Gate](https://www.researchgate.net/publication/353452024_Composition_of_Dynamic_Control_Objectives_Based_on_Differential_Games))
## Local Installation
To clone Git repository locally run this from the command prompt:
```
git clone https://github.com/krichelj/PyDiffGame.git
```

## Input Parameters

The package contains a file named `PyDiffGame.py` and a class of the same name.
An object of the class represents an instance of differential game. Once the object is created,
the method that calls for the game to begin is `play_the_game`.
All the constants are defined in the [Mathematical Description](Math.md) section.
The input parameters to instantiate a `PyDiffGame` object are:

* `A` : numpy 2-d array, of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{red}n,n">)
>The system dynamics matrix
* `B` : list of numpy 2-d arrays, of len(<img src="https://render.githubusercontent.com/render/math?math=\color{red}N">), each matrix <img src="https://render.githubusercontent.com/render/math?math=\color{red}B_i"> of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{red}n, k_j">), <img src="https://render.githubusercontent.com/render/math?math=\color{red}i=1...N">
>System input matrices for each control objective
* `Q` : list of numpy 2-d arrays, of len(<img src="https://render.githubusercontent.com/render/math?math=\color{red}N">), each matrix <img src="https://render.githubusercontent.com/render/math?math=\color{red}Q_i"> of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{red}n, n">), <img src="https://render.githubusercontent.com/render/math?math=\color{red}i=1...N">
>Cost function state weights for each control objective
* `R` : list of numpy 2-d arrays, of len(<img src="https://render.githubusercontent.com/render/math?math=\color{red}N">), each matrix <img src="https://render.githubusercontent.com/render/math?math=\color{red}R_{i}"> of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{red}k_j,k_j">), <img src="https://render.githubusercontent.com/render/math?math=\color{red}j=1...N">
>Cost function input weights for each control objective
* `cl` : boolean
>Indicates whether to render the closed (True) or open (False) loop behaviour
* `T_f` : positive float, optional, default = 5
>System dynamics horizon. Should be given in the case of finite horizon
* `x_0` : numpy 1-d array, of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{red}n">), optional
>Initial state vector
* `P_f` : list of numpy 2-d arrays, of shape(<img src="https://render.githubusercontent.com/render/math?math=\color{red}n, n">), optional, default = solution of scipy's `solve_continuous_are`
>Final condition for the Riccati equation matrix. Should be given in the case of finite horizon
* `data_points` : positive int, optional, default = 1000
>Number of data points to evaluate the Riccati equation matrix. Should be given in the case of finite horizon
* `show_legend` : boolean
>Indicates whether to display a legend in the plots (True) or not (False)
> 
## Tutorial

Let us consider the following input parameters for the instantiation of a `PyDiffGame` object and 
corresponding call for `play_the_game`:

```python
import numpy as np
from PyDiffGame import ContinuousPyDiffGame, DiscretePyDiffGame

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
x_0 = np.array([10, 20])
T_f = 5
P_f = [np.array([[1, 0],
                 [0, 200]]),
       np.array([[3, 0],
                 [0, 4]]),
       np.array([[500, 0],
                 [0, 600]])]
data_points = 1000
show_legend = True

discrete_game = DiscretePyDiffGame(A=A,
                                   B=B,
                                   Q=Q,
                                   R=R,
                                   x_0=x_0,
                                   P_f=P_f,
                                   T_f=T_f,
                                   data_points=data_points,
                                   show_legend=show_legend)
discrete_game.solve_game()

continuous_game = ContinuousPyDiffGame(A=A,
                                       B=B,
                                       Q=Q,
                                       R=R,
                                       cl=cl,
                                       x_0=x_0,
                                       P_f=P_f,
                                       T_f=T_f,
                                       data_points=data_points,
                                       show_legend=show_legend)
P = continuous_game.solve_game()

n = A.shape[0]
P_size = n ** 2
N = len(Q)
print([(P[-1][i * P_size:(i + 1) * P_size]).reshape(n, n) for i in range(N)])
```
The output will be:

```
>>> [array([[0.21206958, 0.02483988],
>>>         [0.02483988, 0.11586253]]), 
>>> array([[ 1.20634199,  6.38917789],
>>>         [ 6.38917789, 42.59805757]]), 
>>> array([[2.26874197, 0.32396162],
>>>         [0.32396162, 0.18060845]])]
```

This will result in the following plot for the Riccati equation solution:

<p align="center">
    <img alt="Logo" src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/tut1_riccati.png"/>
</p>

with the corresponding state space simulation:

<p align="center">
    <img alt="State space simulation" src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/tut1_state.png"/>
</p>