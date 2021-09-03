import numpy as np
from numpy.linalg import eigvals, inv, norm
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, expm, fractional_matrix_power
from scipy.integrate import odeint
import warnings
from typing import List, Optional, Union, Tuple
from scipy.optimize import fsolve
from abc import ABC, abstractmethod
from quadpy import quad

data_points_default = 1000
epsilon_default = 1/(10**3)
delta_T_default = 0.5
last_norms_number_default = 3


class PyDiffGame(ABC):
    """
    Differential game abstract base class

    Parameters
    ----------
    A: numpy 2-d array of shape(n, n)
        The system dynamics matrix
    B: list of numpy 2-d arrays of len(N), each matrix B_j of shape(n, k_j)
        System input matrices for each control objective
    Q: list of numpy 2-d arrays of len(N), each matrix Q_j of shape(n, M)
        Cost function state weights for each control objective
    R: list of numpy 2-d arrays of len(N), each matrix R_j of shape(k_j, k_j)
        Cost function input weights for each control objective
    T_f: positive float, optional, default = 5
        System dynamics horizon. Should be given in the case of finite horizon
    x_0: numpy 1-d array of shape(n), optional
        Initial state vector
    P_f: list of numpy 2-d arrays of len(N), each matrix P_f_j of shape(n, n), optional,
        default = solution of scipy's solve_continuous_are
        Final condition for the Riccati equation matrix. Should be given in the case of finite horizon
    data_points: positive int, optional, default = 1000
        Number of data points to evaluate the Riccati equation matrix. Should be given in the case of finite horizon
    show_legend: boolean, optional, default = True
        Indicates whether to display a legend in the plots (True) or not (False)
    epsilon: float, optional, default = 10e-8
        The convergence threshold for numerical convergence
    delta_T: float, optional, default = 0.5
        Sampling interval
    last_norms_number: int, optional, default = 5
        The number of last matrix norms to consider for convergence
    """

    def __init__(self, A: np.ndarray, B: List[np.ndarray], Q: List[np.ndarray], R: List[np.ndarray],
                 x_0: Optional[np.ndarray] = None, T_f: Optional[Union[float, int]] = None,
                 P_f: Optional[List[np.ndarray]] = None, data_points: int = data_points_default,
                 show_legend: bool = True, epsilon: float = epsilon_default, delta_T: float = delta_T_default,
                 last_norms_number: int = last_norms_number_default):

        # input and shape parameters
        self.A = A
        self.B = B
        self.n = self.A.shape[0]
        self.N = len(self.B)
        self.P_size = self.n ** 2
        self.Q = Q
        self.R = R
        self.x_0 = x_0
        self.infinite_horizon = T_f is None
        self.T_f = 20 if self.infinite_horizon else T_f
        self.P_f = self.get_care_P_f() if P_f is None else P_f
        self.data_points = data_points
        self.show_legend = show_legend

        self.forward_time = np.linspace(0, self.T_f, self.data_points)
        self.backward_time = np.array(list(reversed(self.forward_time)))

        self.check_input()

        # additional parameters
        self.A_t = A.T
        self.S_matrices = [self.B[i] @ inv(self.R[i]) @ self.B[i].T for i in range(self.N)]
        self.P_rows_num = self.N * self.n
        K_rows_num = sum([b.shape[1] for b in self.B])
        self.Z_row_num = self.P_rows_num + K_rows_num

        self.A_cl = np.empty_like(self.A)
        self.P = []
        self.K = []

        self.epsilon = epsilon
        self.delta_T = delta_T
        self.x = self.x_0
        self.last_norms_number = last_norms_number
        self.converged = False

    def check_input(self):
        """
        Input checking method

        Raises
        ------
        Case-specific errors
        """

        if self.A.shape != (self.n, self.n):
            raise ValueError('A must be a square 2-d numpy array with shape nxn')
        if any([B_i.shape[0] != self.n for B_i in self.B]):
            raise ValueError('B must be a list of 2-d numpy arrays with n rows')
        if not all([B_i.shape[1] == R_i.shape[0] == R_i.shape[1]
                    and np.all(eigvals(R_i) >= 0) for B_i, R_i in zip(self.B, self.R)]):
            raise ValueError('R must be a list of square 2-d positive definite numpy arrays with shape '
                             'corresponding to the second dimensions of the arrays in B')
        if any([Q_i.shape != (self.n, self.n) for Q_i in self.Q]):
            raise ValueError('Q must be a list of square 2-d positive semi-definite numpy arrays with shape nxn')
        if any([np.all(eigvals(Q_i) < 0) for Q_i in self.Q]):
            if all([np.all(eigvals(Q_i) >= -1e-15) for Q_i in self.Q]):
                warnings.warn("Warning: there is a matrix in Q that has negative (but really small) eigenvalues")
            else:
                raise ValueError('Q must contain positive semi-definite numpy arrays')
        if self.x_0 is not None and self.x_0.shape != (self.n,):
            raise ValueError('x_0 must be a 1-d numpy array with length n')
        if self.T_f and self.T_f <= 0:
            raise ValueError('T_f must be a positive real number')
        if not all([P_f_i.shape[0] == P_f_i.shape[1] == self.n for P_f_i in self.P_f]):
            raise ValueError('P_f must be a list of 2-d positive semi-definite numpy arrays with shape nxn')
        if not all([eig >= 0 for eig_set in [eigvals(P_f_i) for P_f_i in self.P_f] for eig in eig_set]):
            if all([eig >= -1e-15 for eig_set in [eigvals(P_f_i) for P_f_i in self.P_f] for eig in eig_set]):
                warnings.warn("Warning: there is a matrix in P_f that has negative (but really small) eigenvalues")
            else:
                raise ValueError('P_f must contain positive semi-definite numpy arrays')
        if self.data_points <= 0:
            raise ValueError('The number of data points must be a positive integer')

    def get_care_P_f(self) -> np.array:
        """
        Evaluates the algebraic Riccati equation

        Returns
        ----------
        P_f: list of numpy 2-d arrays, of len(N), of shape(n, n), solution of scipy's solve_continuous_are
            Final condition for the Riccati equation matrix. Should be given in the case of finite horizon
        """

        return [solve_continuous_are(self.A, B_i, Q_i, R_ii) for B_i, Q_i, R_ii in zip(self.B, self.Q, self.R)]

    @abstractmethod
    def update_P_from_last_state(self, **args):
        """
        Updates the matrices in P after forward propagation of the state through time
        """
        pass

    @abstractmethod
    def update_A_cl_from_last_state(self, **args):
        """
        Updates the closed-loop control dynamics after forward propagation of the state through time
        """
        pass

    @abstractmethod
    def is_closed_loop_stable(self) -> bool:
        """
        Tests Lyapunov stability of the closed loop

        Returns
        ----------
        is_stable: boolean
            Indicates whether the system has Lyapunov stability or not
        """
        pass

    def converge_DREs_to_AREs(self):
        last_norms = []

        while not self.converged:
            self.backward_time = np.linspace(self.T_f, self.T_f - self.delta_T, self.data_points)
            self.update_P_from_last_state()
            self.P_f = self.P[-1]
            last_norms += [norm(self.P_f)]

            if len(last_norms) > self.last_norms_number:
                last_norms.pop(0)

            if len(last_norms) == self.last_norms_number:
                self.converged = all([abs(norm_i - norm_i1) < self.epsilon for norm_i, norm_i1
                                      in zip(last_norms, last_norms[1:])])

            self.T_f -= self.delta_T

    @abstractmethod
    def solve_game(self, **args):
        """
        Propagate the game through time, solve it and display the state plot wth respect to time
        """
        pass

    @abstractmethod
    def simulate_state_space(self, **args):
        """
        Propagate the game through time, solve it and display the state plot wth respect to time
        """
        pass

    def plot(self, t: np.ndarray, mat: Tuple[np.ndarray, dict], is_P: bool):
        """
        Displays plots for the state variables with respect to time and the convergence of the values of P
        """
        V = range(1, self.N + 1)
        U = range(1, self.n + 1)

        plt.figure(dpi=130)
        plt.plot(t, mat)
        plt.xlabel('Time')

        if self.show_legend:
            legend = tuple(
                ['${P' + str(i) + '}_{' + str(j) + str(k) + '}$' for i in V for j in U for k in U] if is_P else
                ['${\mathbf{x}}_{' + str(j) + '}$' for j in U])
            plt.legend(legend, loc='upper left' if is_P else 'best', ncol=int(self.n / 2),
                       prop={'size': int(20 / self.n)})

        plt.grid()
        plt.show()


class ContinuousPyDiffGame(PyDiffGame):
    """
    Continuous differential game base class

    Considers the system:
    dx(t)/dt = Ax(t) + sum_{j=1}^N B_j v_j(t)

    with the finite-horizon cost functions:
    J_i = x(T_f)^T F_i(T_f) x(T_f) + int_{t=0}^T_f [x(t)^T Q_i x(t) + sum_{j=1}^N u_j(t)^T R_{ij} u_j(t)]dt

    or with the infinite-horizon cost functions:
    J_i = int_{t=0}^infty [x(t)^T Q_i x(t) + sum_{j=1}^N u_j(t)^T R_{ij} u_j(t)]dt

    Parameters
    ----------
    cl: boolean
        Indicates whether to render the closed (True) or open (False) loop behaviour
    """

    def __init__(self, A: np.ndarray, B: List[np.ndarray], Q: List[np.ndarray], R: List[np.ndarray],
                 cl: bool, x_0: Optional[np.ndarray] = None, T_f: Optional[Union[float, int]] = None,
                 P_f: Optional[List[np.ndarray]] = None, data_points: int = data_points_default,
                 show_legend: bool = True, epsilon: float = epsilon_default, delta_T: float = delta_T_default,
                 last_norms_number: int = last_norms_number_default):

        super().__init__(A=A,
                         B=B,
                         Q=Q,
                         R=R,
                         x_0=x_0,
                         T_f=T_f,
                         P_f=P_f,
                         data_points=data_points,
                         show_legend=show_legend,
                         epsilon=epsilon,
                         delta_T=delta_T,
                         last_norms_number=last_norms_number)

        self.cl = cl
        self.dx_dt = None

        def solve_N_coupled_diff_riccati(_: float, P_t: np.ndarray):
            """
            ODEInt Solver function

            Parameters
            ----------
            _: float
                Integration point in time
            P_t: numpy 2-d array of shape(n, n)
                Current integrated solution
            """
            P_t_matrices = [(P_t[i * self.P_size:(i + 1) * self.P_size]).reshape(self.n, self.n) for i in range(self.N)]
            SP_sum = sum([S_i @ P_t_i for S_i, P_t_i in zip(self.S_matrices, P_t_matrices)])
            dP_tdt = np.zeros((self.P_size,))

            for i in range(self.N):
                P_t_i = P_t_matrices[i]
                Q_i = Q[i]

                dP_t_idt = - self.A_t @ P_t_i - P_t_i @ A - Q_i + P_t_i @ SP_sum

                if self.cl:
                    PS_sum_i = (SP_sum - self.S_matrices[i] @ P_t_matrices[i]).T
                    dP_t_idt = dP_t_idt + PS_sum_i @ P_t_i

                dP_t_idt = dP_t_idt.reshape(self.P_size)
                dP_tdt = dP_t_idt if i == 0 else np.concatenate((dP_tdt, dP_t_idt), axis=0)

            dP_tdt = np.ravel(dP_tdt)

            return dP_tdt

        self.ode_func = solve_N_coupled_diff_riccati

    def update_P_from_last_state(self):
        y_0 = np.ravel(self.P_f)
        self.P = odeint(func=self.ode_func,
                        y0=y_0,
                        t=self.backward_time,
                        tfirst=True)

    def solve_game(self):
        """
        Differential game main evolution method for the continuous case

        Returns
        -------
        P : numpy 2-d array, of shape(n, n) or of shape(n, n)
            Solution to the continuous-time set of Riccati equations
        """

        if self.infinite_horizon:
            self.converge_DREs_to_AREs()
        else:
            self.update_P_from_last_state()

            self.plot(t=self.backward_time,
                      mat=self.P,
                      is_P=True)

        if self.x_0 is not None:
            is_stable = self.is_closed_loop_stable()
            print(f"The system is {'NOT ' if not is_stable else ''}stable")
            self.x = self.simulate_state_space()

            self.plot(t=self.forward_time,
                      mat=self.x,
                      is_P=False)

    def update_A_cl_from_last_state(self, P_matrices: List[np.ndarray]):
        self.A_cl = self.A - sum([a @ b for a, b in zip(self.S_matrices, P_matrices)])

    def is_closed_loop_stable(self) -> bool:
        """
        Tests Lyapunov stability of the closed loop:
        A continuous dynamic system governed by the matrix A_cl has Lyapunov stability iff:

        Re(eig) <= 0 forall eigenvalues of A_cl and there is at most one real eigenvalue at the origin
        """
        closed_loop_eigenvalues = eigvals(self.A_cl)
        closed_loop_eigenvalues_real_values = [eig.real for eig in closed_loop_eigenvalues]
        num_of_zeros = [int(v) for v in closed_loop_eigenvalues_real_values].count(0)
        non_positive_eigenvalues = all([eig_real_value <= 0 for eig_real_value in closed_loop_eigenvalues_real_values])
        at_most_one_zero_eigenvalue = num_of_zeros <= 1

        stability = non_positive_eigenvalues and at_most_one_zero_eigenvalue

        return stability

    def simulate_state_space(self) -> np.array:
        j = len(self.P) - 1

        def state_diff_eqn(x: np.ndarray, _):
            nonlocal j

            P_matrices_j = [(self.P[j][i * self.P_size:(i + 1) * self.P_size]).reshape(self.n, self.n)
                            for i in range(self.N)]
            self.update_A_cl_from_last_state(P_matrices_j)
            dxdt = self.A_cl @ x

            if j > 0:
                j -= 1

            return np.ravel(dxdt)

        x = odeint(func=state_diff_eqn,
                   y0=self.x_0,
                   t=self.forward_time)

        return x


class DiscretePyDiffGame(PyDiffGame):
    """
    Discrete differential game base class


    Considers the system:
    x[k+1] = Ax[k] + sum_{j=1}^N B_j v_j[k]

    with the infinite-horizon cost functions:
    J_i = sum_{k=1}^infty [ x[k]^T Q_i x[k] + sum_{j=1}^N u_j[k]^T R_{ij} u_j[k] ]
    """

    def __init__(self, A: np.ndarray, B: List[np.ndarray], Q: List[np.ndarray], R: List[np.ndarray],
                 x_0: Optional[np.ndarray] = None, T_f: Optional[Union[float, int]] = None,
                 P_f: Optional[List[np.ndarray]] = None, data_points: int = data_points_default,
                 show_legend: bool = True, epsilon: float = epsilon_default, delta_T: float = delta_T_default,
                 last_norms_number: int = last_norms_number_default):

        e_A = expm(A)
        A_discrete = fractional_matrix_power(e_A, delta_T)

        e_AT, err = quad(lambda T: np.array([fractional_matrix_power(e_A, t) for t in T]).swapaxes(0, 2), 0, delta_T)
        B_discrete = [e_AT @ B_i for B_i in B]

        super().__init__(A=A_discrete,
                         B=B_discrete,
                         Q=Q,
                         R=R,
                         x_0=x_0,
                         T_f=T_f,
                         P_f=P_f,
                         data_points=data_points,
                         show_legend=show_legend,
                         epsilon=epsilon,
                         delta_T=delta_T,
                         last_norms_number=last_norms_number)
        self.Z = None

    def update_K_from_last_Z(self):
        self.K = []

        for i in range(self.N):
            K_i_offset = sum([self.B[j].shape[1] for j in range(i)]) if i else 0
            first_K_i_index = self.P_rows_num + K_i_offset
            second_K_i_index = first_K_i_index + self.B[i].shape[1]
            K_i = self.Z[first_K_i_index:second_K_i_index, :]

            self.K += [K_i]

    def is_closed_loop_stable(self) -> bool:
        closed_loop_eigenvalues = eigvals(self.A_cl)
        closed_loop_eigenvalues_absolute_values = [abs(eig) for eig in closed_loop_eigenvalues]
        print(closed_loop_eigenvalues_absolute_values)
        stability = all([eig_norm < 1 for eig_norm in closed_loop_eigenvalues_absolute_values])

        return stability

    def update_P_from_last_state(self):
        self.P = [self.Z[i * self.n:(i + 1) * self.n, :] for i in range(self.N)]

    def update_A_cl_from_last_state(self):
        self.A_cl = self.A - sum([B_i @ K_i for B_i, K_i in zip(self.B, self.K)])

    def update_parameters_from_last_Z(self):
        self.update_P_from_last_state()
        self.update_K_from_last_Z()
        self.update_A_cl_from_last_state()

    def update_Z(self):
        Z_P = []
        Z_K = []

        for i in range(self.N):
            K_i = self.K[i]
            P_i = self.P[i]
            R_ii = self.R[i]
            Q_i = self.Q[i]
            B_i = self.B[i]
            B_i_T_P_i = B_i.T @ P_i

            coefficient = R_ii + B_i_T_P_i @ B_i
            inv_coefficient = inv(coefficient)
            coefficient = inv_coefficient @ B_i_T_P_i
            A_cl_i = self.A_cl + B_i @ K_i

            equation_K_i = K_i - coefficient @ A_cl_i
            Z_K += [equation_K_i]

            equation_P_i = Q_i + K_i.T @ R_ii @ K_i - P_i + self.A_cl.T @ P_i @ self.A_cl
            Z_P += [equation_P_i]

        self.Z = np.concatenate((*Z_P, *Z_K), axis=0)

    def simulate_state_space(self):
        x_0 = self.x_0.reshape((self.n, 1))

        x_t = x_0
        x_T = np.array([x_t]).reshape(1, self.n)

        for _ in self.forward_time[:-1]:
            x_t_1 = self.A_cl @ x_t
            x_T = np.concatenate((x_T, x_t_1.reshape(1, self.n)), axis=0)
            x_t = x_t_1

        # print(f"On iteration {j}, the system is {'NOT ' if not is_Y_stable else ''}stable")
        self.plot(t=self.forward_time, mat=x_T, is_P=False)

    def initialize_Z(self, random_values_multiplier: float = 100):
        P = []
        K = []

        for i in range(self.N):
            B_i_columns_num = self.B[i].shape[1]
            P += [random_values_multiplier * (np.random.rand(self.n, self.n) - 0.5 * np.ones((self.n, self.n)))]
            K += [random_values_multiplier * (np.random.rand(B_i_columns_num, self.n) -
                                              0.5 * np.ones((B_i_columns_num, self.n)))]

        self.Z = np.concatenate((*P, *K), axis=0)

    def solve_game(self, num_of_simulations: int = 1):
        """
        Differential game main evolution method for the discrete case

        Notes
        ----------

        This method is used to solve for the optimal controllers K_i and the riccati equations solutions P_i
        for n >= i >= 1
        """

        warnings.filterwarnings('ignore', 'The iteration is not making good progress')
        j = 0

        def get_equations(Z: np.array):
            nonlocal j

            j += 1
            self.Z = Z.reshape((self.Z_row_num, self.n))

            self.update_parameters_from_last_Z()
            self.update_Z()
            is_Z_stable = self.is_closed_loop_stable()
            print(f"On iteration {j}, the system is {'NOT ' if not is_Z_stable else ''}stable")

            return Z.ravel()

        # is_stable = False
        # is_Z_0_stable = False

        for _ in range(num_of_simulations):
            # while not is_Z_0_stable:
            self.initialize_Z()
            self.update_A_cl_from_last_state()
            is_Z_0_stable = self.is_closed_loop_stable()

            if is_Z_0_stable:
                print(f"Generated random initial guess which is stable")

            self.Z = fsolve(get_equations, self.Z)
            self.Z = np.array(self.Z).reshape((self.Z_row_num, self.n))
            j = 0

            self.simulate_state_space()
            # is_Z_0_stable = False


if __name__ == '__main__':
    A = np.array([[-0.2, 10],
                  [8, -0.6]])
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
    R = [np.array([[10, 0],
                   [0, 20]]),
         np.array([[5]]),
         np.array([[7]])]

    cl = True
    x_0 = np.array([10, 20])
    T_f = 5
    P_f = [np.array([[10, 0],
                     [0, 20]]),
           np.array([[30, 0],
                     [0, 40]]),
           np.array([[50, 0],
                     [0, 60]])]
    data_points = 1000
    show_legend = True

    # for data_points in [i * 200 for i in range(1, 2)]:
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

    # discrete_game = DiscretePyDiffGame(A=A,
    #                                    B=B,
    #                                    Q=Q,
    #                                    R=R,
    #                                    x_0=x_0,
    #                                    # T_f=T_f,
    #                                    data_points=data_points,
    #                                    show_legend=show_legend)
    # discrete_game.solve_game(num_of_simulations=1)
