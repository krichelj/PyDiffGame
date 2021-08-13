## Supported Scenarios
`PyDiffGame` supports several scenarios that can arise based on the designated control law 
of the input system  and the required horizon to solve the system for.

The following scenarios assume no coupling weighting for the inputs of the agendas in the cost function, i.e.:

<img src="https://render.githubusercontent.com/render/math?math=\color{white}R_{ij}=0_{k_i \times k_i} \ : \ \forall 1 \leq i \neq j \leq N">

### Open Loop Finite Horizon

In such cases the following Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=\color{white}\frac{dP_i}{dt} = - A^T P_i - P_i A - Q_i %2B P_i \sum_{j=1}^N S_j P_j  \ , \ P_i(T_f) = P_{f_i} \ : \ \forall 1 \leq i \leq N">
<img src="https://render.githubusercontent.com/render/math?math=\color{white}S_j = B_j R_{j}^{-1} B_j^T \ : \ \forall 1 \leq j \leq N">

### Closed Loop Finite Horizon

In such cases the following Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=\color{white}\frac{dP_i}{dt} = - A^T P_i - P_i A - Q_i %2B P_i \big(\sum_{j=1}^N S_j P_j\big)  %2B \big(\sum_{\substack{j=1 \\ j \neq i}}^NP_jS_j\big)  P_i \ , \ P_i(T_f) = P_{f_i} \ : \ \forall 1 \leq i \leq N">
<img src="https://render.githubusercontent.com/render/math?math=\color{white}S_j = B_j R_{j}^{-1} B_j^T \ : \ \forall 1 \leq j \leq N">