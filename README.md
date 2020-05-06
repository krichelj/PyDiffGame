# PyDiffGame

`PyDiffGame` is a Python package that solves differential games for continuous-time control theory implementations.
This package allows to solve equations arising in models that involve the use of differential games
for finding optimal controllers of a multi-objective system with multiple controlled objects.

## Supported Scenarios
`PyDiffGame` supports several scenarios that can arise based on the designated control law 
of the input system  and the reqired horizon to solve the system for. 

## Required Input
This package receives control parameters for <img src="https://render.githubusercontent.com/render/math?math=m"> control agendas that correspond to <img src="https://render.githubusercontent.com/render/math?math=m"> controlled objects.
Each control object <img src="https://render.githubusercontent.com/render/math?math=x_i"> has <img src="https://render.githubusercontent.com/render/math?math=n"> controlled coordinates, meaning:
<img src="https://render.githubusercontent.com/render/math?math=x_i \in \mathbb{R}^n \ : \ \forall 1 \leq i \leq m">.
It is assumed that each controlled object <img src="https://render.githubusercontent.com/render/math?math=x_i"> adheres to the following model:

<img src="https://render.githubusercontent.com/render/math?math=\dot{x_i} = A_i x_i %2B B_i u_i \ : \ A_i, B_i \in \mathbb{R}^{n \times n}, u_i \in  \mathbb{R}^n">

and has a corresponding cost function:

<img src="https://render.githubusercontent.com/render/math?math=J_i = \int_0^{T_{i}} (x_i^TQ_ix_i %2B u_i^TR_iu_i)dt \ : \ Q_i, R_i \in \mathbb{R}^{n \times n}">

Thus the input includes <img src="https://render.githubusercontent.com/render/math?math=\{ A_i \}_{i=1}^m, \{ B_i \}_{i=1}^m, \{ Q_i \}_{i=1}^m, \{ R_i \}_{i=1}^m">
### Open Loop Finite Horizon

In such cases the follwing Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=\frac{dP_i}{dt} = - A^T P_i - P_i A - Q_i %2B P_i \sum_{j=1}^m S_j P_j  \ : \ \forall 1 \leq i \leq m">
<img src="https://render.githubusercontent.com/render/math?math=S_j = B_j R_j^{-1} B_j^T">

### Closed Loop Finite Horizon

In such cases the follwing Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=\frac{dP_i}{dt} = - A^T P_i - P_i A - Q_i %2B P_i \big(\sum_{j=1}^m S_j P_j\big)  %2B \big(\sum_{\substack{j=1 \\ j \neq i}}^mP_jS_j\big)  P_i\ : \ \forall 1 \leq i \leq m">
<img src="https://render.githubusercontent.com/render/math?math=S_j = B_j R_j^{-1} B_j^T">

### Closed Loop Infinite Horizon

In such cases the follwing Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=0 = P_i A_c %2B A_c^T P_i %2B Q_i %2B \sum_{j=1}^m P_jB_j R_{jj}^{-T}R_{ij}R_{jj}^{-1}B_j^TP_j \ : \ \forall 1 \leq i \leq m">
<img src="https://render.githubusercontent.com/render/math?math=A_c = A - \sum_{i=1}^m S_iP_i \ , \ S_i = B_i R_{ii}^{-1}B_i^T">

The package is still in working progress will include Open Loop Infinite Horizon in the future.

# References
- A Differential Game Approach to Formation Control, Dongbing Gu, IEEE Transactions on Control Systems Technology, 2008
- Optimal Control - Third Edition, Frank L. Lewis, Draguna L. Vrabie, Vassilis L. Syrmos, 2012 by John Wiley & Sons, 2012
