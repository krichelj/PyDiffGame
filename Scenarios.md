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

In such cases the following Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=\frac{dP_i}{dt} = - A^T P_i - P_i A - Q_i %2B P_i \big(\sum_{j=1}^N S_j P_j\big)  %2B \big(\sum_{\substack{j=1 \\ j \neq i}}^NP_jS_j\big)  P_i \ , \ P_i(T_f) = P_{f_i} \ : \ \forall 1 \leq i \leq N">
<img src="https://render.githubusercontent.com/render/math?math=S_j = B_j R_{jj}^{-1} B_j^T \ : \ \forall 1 \leq i \leq N">

### Closed Loop Infinite Horizon

In such cases the following Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=0 = P_i A_c %2B A_c^T P_i %2B Q_i %2B \sum_{j=1}^N P_jB_j R_{jj}^{-T}R_{ij}R_{jj}^{-1}B_j^TP_j \ , \ P_i(T_f) = P_{f_i} \ : \ \forall 1 \leq i \leq N">
<img src="https://render.githubusercontent.com/render/math?math=A_c = A - \sum_{i=1}^N S_iP_i \ , \ S_i = B_i R_{ii}^{-1}B_i^T">

The package is still in working progress will include Open Loop Infinite Horizon in the future.
