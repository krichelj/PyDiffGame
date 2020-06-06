# PyDiffGame

`PyDiffGame` is a Python package that solves differential games for continuous-time control theory implementations.
This package allows to solve equations arising in models that involve the use of differential games
for finding optimal controllers of a multi-objective system with multiple controlled objects.

## Mathematical Description and Required Input
This package receives control parameters for <img src="https://render.githubusercontent.com/render/math?math=N \in \mathbb{N}"> control agendas acting upon <img src="https://render.githubusercontent.com/render/math?math=n \in \mathbb{N}"> controlled objects. Let us denote each controlled object as a state variable of the form:

<img src="https://render.githubusercontent.com/render/math?math=x_i \in \mathbb{R}^{m_i}">

where: 

<img src="https://render.githubusercontent.com/render/math?math=m_i \in \mathbb{N} \ : \ \forall 1 \leq i \leq n">


Let us define the augmented state variable for all of these objects:

<img src="https://render.githubusercontent.com/render/math?math=x = [x_1, ..., x_n]^T">

where:

<img src="https://render.githubusercontent.com/render/math?math=x \in \mathbb{R}^{M}, M = \sum_{i=1}^n m_i">

Using this definition, let us assume the objects adhere to the following continuous-time model:

<img src="https://render.githubusercontent.com/render/math?math=\dot{x} = A x %2B \sum_{j=1}^NB_{j} u_j \ : \ x(0) = x_0">

where:

<img src="https://render.githubusercontent.com/render/math?math=A \in \mathbb{R}^{M \times M}, B_{j} \in \mathbb{R}^{M \times k_j} , u_j \in  \mathbb{R}^{k_j}, k_j \in  \mathbb{N} \ : \ \forall 1 \leq j \leq N">

Let us define for each control agenda a weighted cost function to be minimized:

<img src="https://render.githubusercontent.com/render/math?math=J_i = \int_0^{T_f} (x^TQ_ix %2B \sum_{j=1}^N u_j^TR_{ij}u_j)dt \ : \ \forall 1 \leq i \leq N">

where:

<img src="https://render.githubusercontent.com/render/math?math=Q_i \in \mathbb{R}^{M \times M}, R_{ij} \in \mathbb{R}^{k_j \times k_j} \ : \ \forall 1 \leq i \leq j \leq N">

such that:

<img src="https://render.githubusercontent.com/render/math?math=Q_i \geq 0, R_{ii} > 0 \ : \ \forall 1 \leq i \leq N, R_{ij} \geq 0 \ : \ \forall 1 \leq i < j \leq N">

i.e. <img src="https://render.githubusercontent.com/render/math?math=Q_i, R_{ij}"> are positive semi-definite for all <img src="https://render.githubusercontent.com/render/math?math=1 \leq i < j \leq N"> and <img src="https://render.githubusercontent.com/render/math?math=R_{ii}"> is positive-definite for all <img src="https://render.githubusercontent.com/render/math?math=1 \leq i  \leq N">.

The horizon <img src="https://render.githubusercontent.com/render/math?math=T_f"> is defined as such:

<img src="https://render.githubusercontent.com/render/math?math=T_f \in \mathbb{R} \cup \{ \infty \}">

For the finite horizon case, minimizing the aforementioned cost functions result in solving a set of <img src="https://render.githubusercontent.com/render/math?math=N"> coupled differential Riccati equations for the matrix <img src="https://render.githubusercontent.com/render/math?math=P_i"> backwards in time using <img src="https://render.githubusercontent.com/render/math?math=\psi"> iterations and with the final condition:

<img src="https://render.githubusercontent.com/render/math?math=P_i(T_f) = P_{f_i} \ : \ \forall 1 \leq i \leq N">

where:

<img src="https://render.githubusercontent.com/render/math?math=P_{f_i} \geq 0 \ : \ \forall 1 \leq i \leq N">

Thus in the finite horizon case, the input includes <img src="https://render.githubusercontent.com/render/math?math=\{ m_i \}_{i=1}^N, A, \{ B_i \}_{i=1}^N, \{ Q_i \}_{i=1}^N, \{ \{ R_{ij} \}_{i=1}^N\}_{j=1}^N, T_f, x_0, \{ P_{f_i} \}_{i=1}^N, \psi"> and for the infinite horizon case - the same input holds but without the values for <img src="https://render.githubusercontent.com/render/math?math=P_{f}">.

## Supported Scenarios
`PyDiffGame` supports several scenarios that can arise based on the designated control law 
of the input system  and the required horizon to solve the system for.

The following scenarios assume no coupling weighting for the inputs of the agendas in the cost function, i.e.:

<img src="https://render.githubusercontent.com/render/math?math=R_{ij}=0_{k_i \times k_i} \ : \ \forall 1 \leq i < j \leq N">

### Open Loop Finite Horizon

In such cases the following Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=\frac{dP_i}{dt} = - A^T P_i - P_i A - Q_i %2B P_i \sum_{j=1}^N S_j P_j  \ , \ P_i(T_f) = P_{f_i} \ : \ \forall 1 \leq i \leq N">
<img src="https://render.githubusercontent.com/render/math?math=S_j = B_j R_{jj}^{-1} B_j^T \ : \ \forall 1 \leq i \leq N">

### Closed Loop Finite Horizon

In such cases the follwing Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=\frac{dP_i}{dt} = - A^T P_i - P_i A - Q_i %2B P_i \big(\sum_{j=1}^N S_j P_j\big)  %2B \big(\sum_{\substack{j=1 \\ j \neq i}}^NP_jS_j\big)  P_i \ , \ P_i(T_f) = P_{f_i} \ : \ \forall 1 \leq i \leq N">
<img src="https://render.githubusercontent.com/render/math?math=S_j = B_j R_{jj}^{-1} B_j^T \ : \ \forall 1 \leq i \leq N">

### Closed Loop Infinite Horizon

In such cases the follwing Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=0 = P_i A_c %2B A_c^T P_i %2B Q_i %2B \sum_{j=1}^N P_jB_j R_{jj}^{-T}R_{ij}R_{jj}^{-1}B_j^TP_j \ , \ P_i(T_f) = P_{f_i} \ : \ \forall 1 \leq i \leq N">
<img src="https://render.githubusercontent.com/render/math?math=A_c = A - \sum_{i=1}^N S_iP_i \ , \ S_i = B_i R_{ii}^{-1}B_i^T">

The package is still in working progress will include Open Loop Infinite Horizon in the future.

# References
- **A Differential Game Approach to Formation Control**, Dongbing Gu, IEEE Transactions on Control Systems Technology, 2008
- **Optimal Control - Third Edition**, Frank L. Lewis, Draguna L. Vrabie, Vassilis L. Syrmos, 2012 by John Wiley & Sons, 2012
- **A Survey of Nonsymmetric Riccati Equations**, Gerhard Freiling, Linear Algebra and its Applications, 2002
