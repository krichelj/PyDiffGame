# imports

import numpy as np
from numpy.linalg import eigvals, inv, norm
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, solve_discrete_are, expm
from scipy.integrate import odeint
import warnings
from typing import List, Optional, Union, Tuple
from scipy.optimize import fsolve
from abc import ABC, abstractmethod
from quadpy import quad

# globals

T_f_default = 10
data_points_default = 1000
epsilon_default = 1 / (10 ** 8)
delta_T_default = 0.5
last_norms_number_default = 5


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
                 x_0: np.ndarray = None, T_f: Union[float, int] = None,
                 P_f: List[np.ndarray] = None, data_points: int = data_points_default,
                 show_legend: bool = True, epsilon: float = epsilon_default, delta_T: float = delta_T_default,
                 last_norms_number: int = last_norms_number_default, debug: bool = False):

        # input and shape parameters
        self._A = A
        self._B = B
        self._n = self._A.shape[0]
        self._N = len(self._B)
        self._P_size = self._n ** 2
        self._Q = Q
        self._R = R
        self._x_0 = x_0
        self._infinite_horizon = T_f is None
        self._T_f = T_f_default if self._infinite_horizon else T_f
        self._P_f = self._get_are_P_f() if P_f is None else P_f
        self._data_points = data_points
        self.__show_legend = show_legend
        self._forward_time = np.linspace(0, self._T_f, self._data_points)
        self._backward_time = self._forward_time[::-1]

        self.__check_input()

        # additional parameters
        self._A_cl = np.empty_like(self._A)
        self._P = []
        self._psi = []

        self.__epsilon = epsilon
        self.__delta_T = delta_T
        self._x = self._x_0
        self.__last_norms_number = last_norms_number
        self.__converged = False
        self.debug = debug

    def __check_input(self):
        """
        Input checking method

        Raises
        ------
        Case-specific errors
        """

        if self._A.shape != (self._n, self._n):
            raise ValueError('A must be a square 2-d numpy array with shape nxn')
        if any([B_i.shape[0] != self._n for B_i in self._B]):
            raise ValueError('B must be a list of 2-d numpy arrays with n rows')
        if not all([B_i.shape[1] == R_i.shape[0] == R_i.shape[1]
                    and np.all(eigvals(R_i) >= 0) for B_i, R_i in zip(self._B, self._R)]):
            raise ValueError('R must be a list of square 2-d positive definite numpy arrays with shape '
                             'corresponding to the second dimensions of the arrays in B')
        if any([Q_i.shape != (self._n, self._n) for Q_i in self._Q]):
            raise ValueError('Q must be a list of square 2-d positive semi-definite numpy arrays with shape nxn')
        if any([np.all(eigvals(Q_i) < 0) for Q_i in self._Q]):
            if all([np.all(eigvals(Q_i) >= -1e-15) for Q_i in self._Q]):
                warnings.warn("Warning: there is a matrix in Q that has negative (but really small) eigenvalues")
            else:
                raise ValueError('Q must contain positive semi-definite numpy arrays')
        if self._x_0 is not None and self._x_0.shape != (self._n,):
            raise ValueError('x_0 must be a 1-d numpy array with length n')
        if self._T_f and self._T_f <= 0:
            raise ValueError('T_f must be a positive real number')
        if not all([P_f_i.shape[0] == P_f_i.shape[1] == self._n for P_f_i in self._P_f]):
            raise ValueError('P_f must be a list of 2-d positive semi-definite numpy arrays with shape nxn')
        if not all([eig >= 0 for eig_set in [eigvals(P_f_i) for P_f_i in self._P_f] for eig in eig_set]):
            if all([eig >= -1e-15 for eig_set in [eigvals(P_f_i) for P_f_i in self._P_f] for eig in eig_set]):
                warnings.warn("Warning: there is a matrix in P_f that has negative (but really small) eigenvalues")
            else:
                raise ValueError('P_f must contain positive semi-definite numpy arrays')
        if self._data_points <= 0:
            raise ValueError('The number of data points must be a positive integer')

    @abstractmethod
    def _get_are_P_f(self) -> np.array:
        """
        Evaluates the algebraic Riccati equation

        Returns
        ----------
        P_f: list of numpy 2-d arrays, of len(N), of shape(n, n), solution of scipy's solve_are
            Final condition for the Riccati equation matrix. Should be given in the case of finite horizon
        """

        pass

    @abstractmethod
    def _update_P_from_last_state(self, **args):
        """
        Updates the matrices in P after forward propagation of the state through time
        """
        pass

    @abstractmethod
    def _update_controllers_from_last_state(self, **args):
        """
        Updates the controllers K after forward propagation of the state through time
        """
        pass

    @abstractmethod
    def _update_closed_loop_from_last_state(self, **args):
        """
        Updates the closed-loop control dynamics after forward propagation of the state through time
        """
        pass

    @abstractmethod
    def _is_closed_loop_stable(self, **args) -> bool:
        """
        Tests Lyapunov stability of the closed loop

        Returns
        ----------
        is_stable: boolean
            Indicates whether the system has Lyapunov stability or not
        """
        pass

    def _converge_DREs_to_AREs(self):
        """
        In case of infinite-horizon, this method solves the game as backwards convergence of the differential
        finite-horizon game for repeated consecutive steps until the matrix norm converges
        """
        last_norms = []

        while not self.__converged:
            self._backward_time = np.linspace(self._T_f, self._T_f - self.__delta_T, self._data_points)
            self._update_P_from_last_state()
            self._P_f = self._P[-1]
            last_norms += [norm(self._P_f)]

            if len(last_norms) > self.__last_norms_number:
                last_norms.pop(0)

            if len(last_norms) == self.__last_norms_number:
                self.__converged = all([abs(norm_i - norm_i1) < self.__epsilon for norm_i, norm_i1
                                        in zip(last_norms, last_norms[1:])])

            self._T_f -= self.__delta_T

    @abstractmethod
    def solve_game(self, **args):
        """
        Propagates the game through time, solves for it and plots the state plot wth respect to time
        """
        pass

    def _plot(self, t: np.ndarray, mat: Union[Tuple[np.array, dict], np.array], is_P: bool, title: str = None):
        """
        Displays plots for the state variables with respect to time and the convergence of the values of P
        """
        V = range(1, self._N + 1)
        U = range(1, self._n + 1)

        plt.figure(dpi=130)
        plt.plot(t, mat)
        plt.xlabel('Time')

        if title:
            plt.title(title)

        if self.__show_legend:
            legend = tuple(
                ['${P' + str(i) + '}_{' + str(j) + str(k) + '}$' for i in V for j in U for k in U] if is_P else
                ['${\mathbf{x}}_{' + str(j) + '}$' for j in U])
            plt.legend(legend, loc='upper left' if is_P else 'best', ncol=int(self._n / 2),
                       prop={'size': int(20 / self._n)})

        plt.grid()
        plt.show()

    def _plot_state_space(self):
        """
        Plots the state plot wth respect to time
        """
        self._plot(t=self._forward_time,
                   mat=self._x,
                   is_P=False,
                   title=f'{self._data_points} sampling points')

    @abstractmethod
    def _simulate_state_space(self):
        """
        Propagates the game through time and solves for it
        """
        pass

    def _simulate_and_plot_state_space(self):
        """
        Propagates the game through time, solves for it and plots the state plot wth respect to time
        """
        self._simulate_state_space()
        self._plot_state_space()


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
                 last_norms_number: int = last_norms_number_default, debug: bool = False):

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
                         last_norms_number=last_norms_number,
                         debug=debug)

        self.__S_matrices = [self._B[i] @ inv(self._R[i]) @ self._B[i].T for i in range(self._N)]
        self.__cl = cl
        self.__dx_dt = None

        def __solve_N_coupled_diff_riccati(_: float, P_t: np.ndarray):
            """
            ODEInt Solver function

            Parameters
            ----------
            _: float
                Integration point in time
            P_t: numpy 2-d array of shape(n, n)
                Current integrated solution
            """
            P_t = [(P_t[i * self._P_size:(i + 1) * self._P_size]).reshape(self._n, self._n) for i in range(self._N)]
            SP_sum = sum([S_i @ P_t_i for S_i, P_t_i in zip(self.__S_matrices, P_t)])
            dP_tdt = np.zeros((self._P_size,))

            for i in range(self._N):
                P_t_i = P_t[i]
                Q_i = Q[i]

                dP_t_idt = - self._A.T @ P_t_i - P_t_i @ A - Q_i + P_t_i @ SP_sum  # open-loop

                if self.__cl:  # closed-loop
                    PS_sum_i = (SP_sum - self.__S_matrices[i] @ P_t[i]).T
                    dP_t_idt = dP_t_idt + PS_sum_i @ P_t_i

                dP_t_idt = dP_t_idt.reshape(self._P_size)
                dP_tdt = dP_t_idt if i == 0 else np.concatenate((dP_tdt, dP_t_idt), axis=0)

            dP_tdt = np.ravel(dP_tdt)

            return dP_tdt

        self.__ode_func = __solve_N_coupled_diff_riccati

    def _get_are_P_f(self) -> np.array:
        """
        Evaluates the algebraic Riccati equation

        Returns
        ----------
        P_f: list of numpy 2-d arrays, of len(N), of shape(n, n), solution of scipy's solve_continuous_are
            Final condition for the Riccati equation matrix. Should be given in the case of finite horizon
        """

        return [solve_continuous_are(self._A, B_i, Q_i, R_ii) for B_i, Q_i, R_ii in zip(self._B, self._Q, self._R)]

    def _update_closed_loop_from_last_state(self):
        self._A_cl = self._A - sum([B_i @ K_i for B_i, K_i in zip(self._B, self._psi)])

    def _update_P_from_last_state(self):
        y_0 = np.ravel(self._P_f)
        self._P = odeint(func=self.__ode_func,
                         y0=y_0,
                         t=self._backward_time,
                         tfirst=True)

    def solve_game(self):
        """
        Differential game main evolution method for the continuous case

        Returns
        -------
        P : numpy 2-d array, of shape(n, n) or of shape(n, n)
            Solution to the continuous-time set of Riccati equations
        """

        if self._infinite_horizon:
            self._converge_DREs_to_AREs()
        else:
            self._update_P_from_last_state()
            self._plot(t=self._backward_time,
                       mat=self._P,
                       is_P=True)

        if self._x_0 is not None:
            if self.debug:
                is_stable = self._is_closed_loop_stable()
                print(f"The system is {'NOT ' if not is_stable else ''}stable")

            self._simulate_and_plot_state_space()

    def _update_controllers_from_last_state(self, t: int):
        P_t = [(self._P[t][i * self._P_size:(i + 1) * self._P_size]).reshape(self._n, self._n) for i in range(self._N)]
        self._psi = [inv(R_i) @ B_i.T @ P_i for R_i, B_i, P_i in zip(self._R, self._B, P_t)]

    def _is_closed_loop_stable(self) -> bool:
        """
        Tests Lyapunov stability of the closed loop:
        A continuous dynamic system governed by the matrix A_cl has Lyapunov stability iff:

        Re(eig) <= 0 forall eigenvalues of A_cl and there is at most one real eigenvalue at the origin
        """
        closed_loop_eigenvalues = eigvals(self._A_cl)
        closed_loop_eigenvalues_real_values = [eig.real for eig in closed_loop_eigenvalues]
        num_of_zeros = [int(v) for v in closed_loop_eigenvalues_real_values].count(0)
        non_positive_eigenvalues = all([eig_real_value <= 0 for eig_real_value in closed_loop_eigenvalues_real_values])
        at_most_one_zero_eigenvalue = num_of_zeros <= 1
        print(f'A_cl = \n{self._A_cl}\neig=\n{closed_loop_eigenvalues_real_values}')
        stability = non_positive_eigenvalues and at_most_one_zero_eigenvalue

        return stability

    def _simulate_state_space(self):
        t = len(self._P) - 1

        def state_diff_eqn(x: np.ndarray, _):
            nonlocal t

            self._update_controllers_from_last_state(t)
            self._update_closed_loop_from_last_state()
            dxdt = self._A_cl @ x
            dxdt = dxdt.ravel()

            if t > 0:
                t -= 1

            return dxdt

        self._x = odeint(func=state_diff_eqn,
                         y0=self._x_0,
                         t=self._forward_time)


class DiscretePyDiffGame(PyDiffGame):
    """
    Discrete differential game base class


    Considers the system:
    x[k+1] = Ax[k] + sum_{j=1}^N B_j v_j[k]

    with the infinite-horizon cost functions:
    J_i = sum_{k=1}^infty [ x[k]^T Q_i x[k] + sum_{j=1}^N u_j[k]^T R_{ij} u_j[k] ]
    """

    def __init__(self, A: np.ndarray, B: List[np.ndarray], Q: List[np.ndarray], R: List[np.ndarray],
                 is_input_discrete: bool, x_0: Optional[np.ndarray] = None, T_f: Optional[Union[float, int]] = None,
                 P_f: Optional[List[np.ndarray]] = None, data_points: int = data_points_default,
                 show_legend: bool = True, epsilon: float = epsilon_default, delta_T: float = delta_T_default,
                 last_norms_number: int = last_norms_number_default, debug: bool = False):

        if not is_input_discrete:
            delta_A = delta_T * A
            A_tilda = expm(delta_A)
            e_AT, err = quad(lambda T: np.array([expm(t * A) for t in T]).swapaxes(0, 2), 0, delta_T)
            A = A_tilda
            B_tilda = e_AT
            B = [B_tilda @ B_i for B_i in B]
            Q = [delta_T * Q_i for Q_i in Q]
            R = [delta_T * R_i for R_i in R]

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
                         last_norms_number=last_norms_number,
                         debug=debug)

        self._psi_rows_num = sum([B_i.shape[1] for B_i in self._B])
        self.debug_counter = 0

        def __solve_for_psi_k(psi_k_previous: np.array, k_1: int) -> np.array:
            psi_k_previous = psi_k_previous.reshape((self._N,
                                                     self._psi_rows_num,
                                                     self._n))
            P_k_1 = self._P[k_1]
            psi_k = np.zeros_like(psi_k_previous)

            for i in range(self._N):
                i_indices = (i, self._get_psi_i_indices(i))
                psi_k_previous_i = psi_k_previous[i_indices]
                P_k_1_i = P_k_1[i]
                R_ii = self._R[i]
                B_i = self._B[i]
                B_i_T = B_i.T

                A_cl_k_i = self._A

                for j in range(self._N):
                    if j != i:
                        B_j = self._B[j]
                        psi_k_previous_j = psi_k_previous[j, self._get_psi_i_indices(j)]
                        A_cl_k_i = A_cl_k_i - B_j @ psi_k_previous_j

                psi_k_i = psi_k_previous_i - inv(R_ii + B_i_T @ P_k_1_i @ B_i) @ B_i_T @ P_k_1_i @ A_cl_k_i
                psi_k[i_indices] = psi_k_i

            return psi_k.ravel()

        self.f_solve_func = __solve_for_psi_k

    def _get_are_P_f(self) -> np.array:
        """
        Evaluates the algebraic Riccati equation

        Returns
        ----------
        P_f: list of numpy 2-d arrays, of len(N), of shape(n, n), solution of scipy's solve_continuous_are
            Final condition for the Riccati equation matrix. Should be given in the case of finite horizon
        """

        return [solve_discrete_are(self._A, B_i, Q_i, R_ii) for B_i, Q_i, R_ii in zip(self._B, self._Q, self._R)]

    def _get_psi_i_indices(self, i: int) -> range:
        psi_i_offset = sum([self._B[j].shape[1] for j in range(i)]) if i else 0
        first_psi_i_index = psi_i_offset
        second_psi_i_index = first_psi_i_index + self._B[i].shape[1]

        return range(first_psi_i_index, second_psi_i_index)

    def _update_closed_loop_from_last_state(self, k: int):
        A_cl_k = self._A
        psi_k = self._psi[k]

        for i in range(self._N):
            psi_k_i = psi_k[i, self._get_psi_i_indices(i)]
            A_cl_k = A_cl_k - self._B[i] @ psi_k_i

        self._A_cl[k] = A_cl_k

    def _get_backwards_t_1_index(self, t: float) -> int:
        return int(np.where(self._backward_time == t)[0])

    def _update_controllers_from_last_state(self, k_1: int):
        psi_k_1 = self._psi[k_1]
        # psi_k_0 = np.random.rand(*psi_k_1.shape)
        psi_k = fsolve(func=self.f_solve_func,
                       x0=psi_k_1,
                       args=tuple([k_1]))
        psi_k_reshaped = np.array(psi_k).reshape((self._N,
                                                  self._psi_rows_num,
                                                  self._n))

        if self.debug:
            k = k_1 - 1
            f_psi_k = self.f_solve_func(psi_k_previous=psi_k, k_1=k)
            close_to_zero_array = np.isclose(f_psi_k, np.zeros_like(psi_k))
            is_close_to_zero = all(close_to_zero_array)
            print(f"The current solution is {'NOT ' if not is_close_to_zero else ''}close to zero"
                  f"\npsi_k:\n{psi_k_reshaped}\nf_psi_k:\n{f_psi_k}")

        self._psi[k_1 - 1] = psi_k_reshaped

    def _is_closed_loop_stable(self, k: int) -> bool:
        A_cl_k = self._A_cl[k]
        A_cl_k_eigenvalues = eigvals(A_cl_k)
        A_cl_k_eigenvalues_absolute_values = [abs(eig) for eig in A_cl_k_eigenvalues]
        if self.debug:
            print(A_cl_k_eigenvalues_absolute_values)
        stability = all([eig_norm < 1 for eig_norm in A_cl_k_eigenvalues_absolute_values])

        return stability

    def _update_P_from_last_state(self, k_1: int):
        k = k_1 - 1
        P_k_1 = self._P[k_1]
        psi_k = self._psi[k]
        A_cl_k = self._A_cl[k]
        P_k = np.zeros_like(P_k_1)

        for i in range(self._N):
            psi_k_i = psi_k[i, self._get_psi_i_indices(i)]
            P_k_1_i = P_k_1[i]
            R_ii = self._R[i]
            Q_i = self._Q[i]

            P_k[i] = A_cl_k.T @ P_k_1_i @ A_cl_k + psi_k_i.T @ R_ii @ psi_k_i + Q_i

        self._P[k] = P_k

    def _simulate_state_space(self):
        x_k = self._x_0
        self._x[0] = x_k

        for k in range(1, self._data_points - 1):
            A_cl_k = self._A_cl[k]
            x_k_1 = A_cl_k @ x_k
            self._x[k] = x_k_1
            x_k = x_k_1

    def _initialize(self):
        self._P = np.random.rand(self._data_points,
                                 self._N,
                                 self._n,
                                 self._n)
        self._psi = np.random.rand(self._data_points,
                                   self._N,
                                   self._psi_rows_num,
                                   self._n)
        self._A_cl = np.zeros((self._data_points,
                               self._n,
                               self._n))
        self._P[-1] = np.array(self._Q).reshape((self._N,
                                                 self._n,
                                                 self._n))
        self._update_closed_loop_from_last_state(k=-1)
        self._x = np.zeros((self._data_points,
                            self._n))
        self._x[0] = self._x_0

    def solve_game(self, num_of_simulations: int = 1):
        """
        Differential game main evolution method for the discrete case

        Notes
        ----------

        This method is used to solve for the optimal controllers K_i and the riccati equations solutions P_i
        for n >= i >= 1
        """

        # warnings.filterwarnings('ignore', 'The iteration is not making good progress')

        for _ in range(num_of_simulations):
            self._initialize()

            for backwards_index, t in enumerate(self._backward_time[:-1]):
                k_1 = self._data_points - 1 - backwards_index
                k = k_1 - 1

                if self.debug:
                    print('#' * 30 + f' t = {round(t, 3)} ' + '#' * 30)

                self._update_controllers_from_last_state(k_1)

                if self.debug:
                    print(f'psi_k+1:\n{self._psi[k_1]}')

                self._update_closed_loop_from_last_state(k)

                if self.debug:
                    print(f'A_cl_k:\n{self._A_cl[k]}')
                    stable_k = self._is_closed_loop_stable(k)
                    print(f"The system is {'NOT ' if not stable_k else ''}stable")

                self._update_P_from_last_state(k)

            self._simulate_and_plot_state_space()


if __name__ == '__main__':
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
    x_0 = np.array([100, -200])
    T_f = 5
    P_f = [np.array([[1, 0],
                     [0, 2]]),
           np.array([[3, 0],
                     [0, 4]]),
           np.array([[5, 0],
                     [0, 6]])]
    # data_points = 1000
    show_legend = True

    for data_points in [i * 100 for i in range(2, 10)]:
        continuous_game = ContinuousPyDiffGame(A=A,
                                               B=B,
                                               Q=Q,
                                               R=R,
                                               cl=cl,
                                               x_0=x_0,
                                               # P_f=P_f,
                                               # T_f=T_f,
                                               data_points=data_points,
                                               show_legend=show_legend)
        continuous_game.solve_game()

        discrete_game = DiscretePyDiffGame(A=A,
                                           B=B,
                                           Q=Q,
                                           R=R,
                                           is_input_discrete=False,
                                           x_0=x_0,
                                           # T_f=T_f,
                                           data_points=data_points,
                                           show_legend=show_legend,
                                           debug=False)
        discrete_game.solve_game(num_of_simulations=1)