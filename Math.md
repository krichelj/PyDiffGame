## Mathematical Description
This package receives control parameters for <img src="https://render.githubusercontent.com/render/math?math=\color{red}N \in \mathbb{N}"> control objectives acting upon a state vector:

<img src="https://render.githubusercontent.com/render/math?math=\color{red}\mathbf{x} \in \mathbb{R}^{n}\ : \  n \in  \mathbb{N}">

Using this definition, let us assume the state vector adheres the following continuous-time model:

<img src="https://render.githubusercontent.com/render/math?math=\color{red}\dot{\mathbf{x}} = A \mathbf{x} %2B \sum_{j=1}^NB_{j} \mathbf{u}_j \ : \ \mathbf{x}(0) = \mathbf{x}_0">

where:

<img src="https://render.githubusercontent.com/render/math?math=\color{red}A \in \mathbb{R}^{n \times n}, B_{j} \in \mathbb{R}^{n \times m_j} , \mathbf{u}_j \in  \mathbb{R}^{m_j}, m_j \in  \mathbb{N} \ : \ \forall 1 \leq j \leq N">

Let us define for each control objective a weighted cost function to be minimized:

<img src="https://render.githubusercontent.com/render/math?math=\color{red}J_i = \int_0^{T_f} (\mathbf{x}^TQ_i\mathbf{x} %2B \sum_{j=1}^N \mathbf{u}_j^TR_{j}\mathbf{u}_j)dt %2B \mathbf{x}(T_f)^TQ_{f_i}\mathbf{x}(T_f) \ : \ \forall 1 \leq i \leq N">

where:

<img src="https://render.githubusercontent.com/render/math?math=\color{red}Q_i, Q_{f_i} \in \mathbb{R}^{n \times n}, Q_i, Q_{f_i} \geq 0, R_{j} \in \mathbb{R}^{m_j \times m_j}, R_{i} > 0 \ : \ \forall 1 \leq i \leq N">

i.e. <img src="https://render.githubusercontent.com/render/math?math=\color{red}Q_i, Q_{f_i}"> is positive semi-definite and <img src="https://render.githubusercontent.com/render/math?math=\color{red}R_{j}"> is positive-definite for all <img src="https://render.githubusercontent.com/render/math?math=\color{red}1 \leq i \leq j \leq N">.

The horizon <img src="https://render.githubusercontent.com/render/math?math=\color{red}T_f"> is defined as such:

<img src="https://render.githubusercontent.com/render/math?math=\color{red}T_f \in \mathbb{R} \cup \{ \infty \}">

For the finite horizon case (<img src="https://render.githubusercontent.com/render/math?math=\color{red}T_f < \infty ">), minimizing the aforementioned cost functions result in solving a set of <img src="https://render.githubusercontent.com/render/math?math=\color{red}N"> coupled differential Riccati equations for the matrix <img src="https://render.githubusercontent.com/render/math?math=\color{red}P_i"> backwards in time using <img src="https://render.githubusercontent.com/render/math?math=\color{red}\psi"> data points between 0 and <img src="https://render.githubusercontent.com/render/math?math=\color{red}T_f">, and with the final condition:

<img src="https://render.githubusercontent.com/render/math?math=\color{red}P_i(T_f) = P_{f_i} \ : \ \forall 1 \leq i \leq N">

where:

<img src="https://render.githubusercontent.com/render/math?math=\color{red}P_{f_i} \geq 0 \ : \ \forall 1 \leq i \leq N">

Thus in the finite horizon case, the input includes <img src="https://render.githubusercontent.com/render/math?math=\color{red}A, \{ B_i \}_{i=1}^N, \{ Q_i \}_{i=1}^N,\{ Q_{f_i} \}_{i=1}^N,  \{ R_{i}  \}_{i=1}^N, T_f, \mathbf{x}_0, \{ P_{f_i} \}_{i=1}^N, \psi"> and for the infinite horizon case - we set <img src="https://render.githubusercontent.com/render/math?math=\color{red}Q_{f_i} = 0 \ : \ \forall 1 \leq i \leq N"> and then the same input holds but without the values for <img src="https://render.githubusercontent.com/render/math?math=\color{red}P_{f}"> and <img src="https://render.githubusercontent.com/render/math?math=\color{red}\psi">.

# References
- **_A Differential Game Approach to Formation Control_**, Dongbing Gu, IEEE Transactions on Control Systems Technology, 2008
- **_Optimal Control - Third Edition_**, Frank L. Lewis, Draguna L. Vrabie, Vassilis L. Syrmos, 2012 by John Wiley & Sons, 2012
- **_A Survey of Nonsymmetric Riccati Equations_**, Gerhard Freiling, Linear Algebra and its Applications, 2002
