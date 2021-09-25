# Supported Scenarios
`PyDiffGame` supports several scenarios that can arise based on the designated control law 
of the input system  and the required horizon to solve the system for.

The following scenarios assume no coupling weighting for the inputs of the agendas in the cost function, i.e.:

<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}R_{ij}=0_{k_i \times k_i} \ : \ \forall 1 \leq i \neq j \leq N">

## Open Loop Finite Horizon

In such cases the following Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}\frac{dP_i}{dt} = - A^T P_i - P_i A - Q_i %2B P_i \sum_{j=1}^N B_j R_{jj}^{-1} B_j^T P_j  \ , \ P_i(T_f) = P_{f_i} \ : \ \forall 1 \leq i \leq N">

## Closed Loop Finite Horizon

In such cases the following Riccati equation arise:

<img src="https://render.githubusercontent.com/render/math?math=\color{yellow}\frac{dP_i}{dt} = - A^T P_i - P_i A - Q_i %2B P_i \sum_{j=1}^N B_j R_{jj}^{-1} B_j^T P_j  %2B \big(\sum_{\substack{j=1 \\ j \neq i}}^N P_j B_j R_{jj}^{-1} B_j^T\big)  P_i \ , \ P_i(T_f) = P_{f_i} \ : \ \forall 1 \leq i \leq N">
