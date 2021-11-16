# imports
from __future__ import annotations
import numpy as np
from numpy.linalg import eigvals, inv, norm
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.integrate import odeint, quad
import warnings
from typing import Union, Callable, Any
from scipy.optimize import fsolve
from abc import ABC, abstractmethod
from quadpy import quad as quadpy


class PyDiffGame(ABC):
    """
    Differential game abstract base class


    Parameters
    ----------
    A: numpy 2-d array of shape(n, n)
        The system dynamics matrix
    B: numpy array of numpy 2-d arrays of len(N), each matrix B_j of shape(n, k_j)
        System input matrices for each control objective
    Q: numpy array of numpy 2-d arrays of len(N), each matrix Q_j of shape(n, M)
        Cost function state weights for each control objective
    R: numpy array of numpy 2-d arrays of len(N), each matrix R_j of shape(k_j, k_j)
        Cost function input weights for each control objective
    x_0: numpy 1-d array of shape(n), optional
        Initial state vector
    x_T: numpy 1-d array of shape(n), optional
        Final state vector, in case of signal tracking
    T_f: positive float, optional, default = 10
        System dynamics horizon. Should be given in the case of finite horizon
    P_f: numpy array of numpy 2-d arrays of len(N), each matrix P_f_j of shape(n, n), optional,
        default = uncoupled solution of scipy's solve_are
        Final condition for the Riccati equation matrix. Should be given in the case of finite horizon
    data_points: positive int, optional, default = 1000
        Number of data points to evaluate the Riccati equation matrix. Should be given in the case of finite horizon
    show_legend: boolean, optional, default = True
        Indicates whether to display a legend in the plots (True) or not (False)
    epsilon: float, optional, default = 1 / (10 ** 8)
        The convergence threshold for numerical convergence
    delta_T: float, optional, default = 0.5
        Sampling interval
    last_norms_number: int, optional, default = 5
        The number of last matrix norms to consider for convergence
    debug: boolean, optional, default = False
        Indicates whether to display debug information or not
    """

    __T_f_default = 5
    _data_points_default = 1000
    _epsilon_default = 10 ** (-3)
    _delta_T_default = 0.01
    _last_norms_number_default = 3

    def __init__(self,
                 A: np.array,
                 B: list[np.array],
                 Q: list[np.array],
                 R: list[np.array],
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 P_f: list[np.array] = None,
                 data_points: int = _data_points_default,
                 show_legend: bool = True,
                 epsilon: float = _epsilon_default,
                 delta_T: float = _delta_T_default,
                 last_norms_number: int = _last_norms_number_default,
                 force_finite_horizon: bool = False,
                 debug: bool = False
                 ):

        self.__continuous = isinstance(self, ContinuousPyDiffGame)

        # input and shape parameters
        self._A = A
        self._B = B
        self._n = self._A.shape[0]
        self._N = len(self._B)
        self._P_size = self._n ** 2
        self._Q = Q
        self._R = R
        self._x_0 = x_0
        self._x_T = x_T
        self.__infinite_horizon = (not force_finite_horizon) and (T_f is None or P_f is None)
        self._T_f = PyDiffGame.__T_f_default if T_f is None else T_f
        self._P_f = self.__get_are_P_f() if P_f is None else P_f
        self._data_points = data_points
        self.__show_legend = show_legend
        self._forward_time = np.linspace(0, self._T_f, self._data_points)
        self._backward_time = self._forward_time[::-1]

        self.__verify_input()

        # additional parameters
        self._A_cl = np.empty_like(self._A)
        self._P = []
        self._K: list[np.array] = []

        self.__epsilon = epsilon
        self._delta_T = delta_T
        self._x = self._x_0
        self.__last_norms_number = last_norms_number
        self._converged = False
        self._debug = debug

    def __verify_input(self):
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
        if not all([(B_i.shape[1] == R_i.shape[0] == R_i.shape[1] if R_i.ndim > 1 else B_i.shape[1] == R_i.shape[0])
                    and (np.all(eigvals(R_i) > 0) if R_i.ndim > 1 else R_i > 0) for B_i, R_i in zip(self._B, self._R)]):
            raise ValueError('R must be a list of square 2-d positive definite numpy arrays with shape '
                             'corresponding to the second dimensions of the arrays in B')
        if any([Q_i.shape != (self._n, self._n) for Q_i in self._Q]):
            raise ValueError('Q must be a list of square 2-d positive semi-definite numpy arrays with shape nxn')
        if any([np.all(eigvals(Q_i) < 0) for Q_i in self._Q]):
            if all([np.all(eigvals(Q_i) >= -1e-7) for Q_i in self._Q]):
                warnings.warn("Warning: there is a matrix in Q that has negative (but really small) eigenvalues")
            else:
                raise ValueError('Q must contain positive semi-definite numpy arrays')
        if self._x_0 is not None and self._x_0.shape != (self._n,):
            raise ValueError('x_0 must be a 1-d numpy array with length n')
        if self._x_T is not None and self._x_T.shape != (self._n,):
            raise ValueError('x_T must be a 1-d numpy array with length n')
        if self._T_f and self._T_f <= 0:
            raise ValueError('T_f must be a positive real number')
        if not all([P_f_i.shape[0] == P_f_i.shape[1] == self._n for P_f_i in self._P_f]):
            raise ValueError('P_f must be a list of 2-d positive semi-definite numpy arrays with shape nxn')
        if not all([eig >= 0 for eig_set in [eigvals(P_f_i) for P_f_i in self._P_f] for eig in eig_set]):
            warnings.warn("Warning: there is a matrix in P_f that has negative eigenvalues. Convergence may not occur")
        if self._data_points <= 0:
            raise ValueError('The number of data points must be a positive integer')

    def _converge_DREs_to_AREs(self):
        """
        Solves the game as backwards convergence of the differential
        finite-horizon game for repeated consecutive steps until the matrix norm converges
        """

        last_norms = []

        while not self._converged:
            self._backward_time = np.linspace(self._T_f, self._T_f - self._delta_T, self._data_points)
            self._update_P_from_last_state()
            self._P_f = self._P[-1]
            last_norms += [norm(self._P_f)]

            if len(last_norms) > self.__last_norms_number:
                last_norms.pop(0)

            if len(last_norms) == self.__last_norms_number:
                self._converged = all([abs(norm_i - norm_i1) < self.__epsilon for norm_i, norm_i1
                                       in zip(last_norms, last_norms[1:])])

            self._T_f -= self._delta_T

    def _post_convergence(method: Callable) -> Callable:
        """
        A decorator static-method to apply on methods that can only be called after convergence
        """

        def verify_convergence(self: PyDiffGame, *args, **kwargs) -> Any:
            if not self.converged:
                raise RuntimeError('Must first simulate the differential game')
            method(self, *args, **kwargs)

        return verify_convergence

    @_post_convergence
    def _plot(self, t: np.array, mat: np.array, is_P: bool, title: str = None):
        """
        Displays plots for the state variables with respect to time and the convergence of the values of P

        Parameters
        ----------
        t: numpy array of len(data_points)
            The time axis information to plot
        mat: numpy array
            The y-axis information to plot
        is_P: boolean
            Indicates whether to accommodate plotting for all the values in P_i or just for x
        title: str, optional
            The plot title to display
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
            plt.legend(legend, loc='upper left' if is_P else 'best',
                       ncol=int(self._n / 2),
                       prop={'size': int(40 / self._n)})

        plt.grid()
        plt.show()

    def __plot_variables(self, mat):
        """
        Plots the state plot wth respect to time
        """

        self._plot(t=self._forward_time,
                   mat=mat,
                   is_P=False,
                   title=f"{('Continuous' if self.__continuous else 'Discrete')}, "
                         f"{('Infinite' if self.__infinite_horizon else 'Finite')} Horizon, {self._N}-Player Game, "
                         f"{self._data_points} sampling points")

    def _plot_state_space(self):
        self.__plot_variables(mat=self._x)

    def _plot_Y(self, C: np.array):
        Y = C @ self._x.T
        self.__plot_variables(mat=Y.T)

    def __get_are_P_f(self) -> list[np.array]:
        """
        Solves the uncoupled set of algebraic Riccati equations to use as initial guesses for fsolve and odeint

        Returns
        ----------
        P_f: numpy array of numpy 2-d arrays, of len(N), of shape(n, n), solution of scipy's solve_are
            Final condition for the Riccati equation matrix
        """

        are_solver = solve_continuous_are if self.__continuous else solve_discrete_are
        return [are_solver(self._A, B_i, Q_i, R_ii) for B_i, Q_i, R_ii in zip(self._B, self._Q, self._R)]

    @abstractmethod
    def _update_K_from_last_state(self, *args):
        """
        Updates the controllers K after forward propagation of the state through time
        """

        pass

    @abstractmethod
    def _get_K_i(self, *args) -> np.array:
        pass

    def _update_A_cl_from_last_state(self, k: int = None):
        """
        Updates the closed-loop control dynamics with the updated controllers based on the relation:
        A_cl = A - sum_{i=1}^N B_i K_i

        Parameters
        ----------
        k: int, optional
            The current k'th sample index, in case the controller is time-dependant
        """

        A_cl = self._A - sum([self._B[i] @ (self._get_K_i(i, k) if k is not None else self._get_K_i(i))
                              for i in range(self._N)])
        if k is not None:
            self._A_cl[k] = A_cl
        else:
            self._A_cl = A_cl

    @abstractmethod
    def _update_P_from_last_state(self, *args):
        """
        Updates the matrices P after forward propagation of the state through time
        """

        pass

    @abstractmethod
    def is_A_cl_stable(self, *args) -> bool:
        """
        Tests Lyapunov stability of the closed loop

        Returns
        ----------
        is_stable: boolean
            Indicates whether the system has Lyapunov stability or not
        """

        pass

    @abstractmethod
    def _solve_finite_horizon(self):
        """
        Solves for the finite horizon case
        """

        pass

    def _plot_finite_horizon_convergence(self):
        """
        Plots the convergence of the values for the matrices P_i
        """

        self._plot(t=self._backward_time,
                   mat=self._P,
                   is_P=True)

    def _solve_and_plot_finite_horizon(self):
        """
        Solves for the finite horizon case and plots the convergence of the values for the matrices P_i
        """

        self._solve_finite_horizon()
        self._plot_finite_horizon_convergence()

    @abstractmethod
    def _solve_infinite_horizon(self):
        """
        Solves for the infinite horizon case using the finite horizon case
        """

        pass

    @abstractmethod
    @_post_convergence
    def _solve_state_space(self):
        """
        Propagates the game through time and solves for it
        """

        pass

    def solve_game(self):
        if self.__infinite_horizon:
            self._solve_infinite_horizon()
        else:
            self._solve_and_plot_finite_horizon()

    def simulate_state_space(self):
        if self._x_0 is not None:
            if self._debug:
                print(f"The system is {'NOT ' if not self.is_A_cl_stable() else ''}stable")

            self._solve_state_space()
            self._plot_state_space()

    def solve_game_and_simulate_state_space(self):
        """
        Propagates the game through time, solves for it and plots the state with respect to time
        """
        self.solve_game()
        self.simulate_state_space()

    def calculate_J(self) -> np.array:
        J = [quad(func=lambda x: x, a=0, b=self._T_f) for i in range(self._N)]

        return np.array(J)

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def Q(self):
        return self._Q

    @property
    def R(self):
        return self._R

    @property
    def K(self):
        return self._K

    @property
    def forward_time(self):
        return self._forward_time

    @property
    def converged(self):
        return self._converged

    def __len__(self) -> int:
        """
        We define the length of a differential game to be its number of objectives
        """

        return self._N

    def __eq__(self, other: PyDiffGame) -> bool:
        """
        We define two differential games equal iff the system dynamics and cost matrices are all equal
        """

        return all([s == o for s, o in zip([self.A, self.B, self.Q, self.R], [other.A, other.B, other.Q, other.R])])

    _post_convergence = staticmethod(_post_convergence)


class ContinuousPyDiffGame(PyDiffGame):
    """
    Continuous differential game base class


    Considers the system:
    dx(t)/dt = A x(t) + sum_{j=1}^N B_j v_j(t)

    Parameters
    ----------
    cl: boolean, optional, default = True
        Indicates whether to generate the closed (True) or open (False) loop behaviour
    """

    def __init__(self,
                 A: np.array,
                 B: list[np.array],
                 Q: list[np.array],
                 R: list[np.array],
                 cl: bool = True,
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 P_f: list[np.array] = None,
                 data_points: int = PyDiffGame._data_points_default,
                 show_legend: bool = True,
                 epsilon: float = PyDiffGame._epsilon_default,
                 delta_T: float = PyDiffGame._delta_T_default,
                 last_norms_number: int = PyDiffGame._last_norms_number_default,
                 force_finite_horizon: bool = False,
                 debug: bool = False
                 ):

        super().__init__(A=A,
                         B=B,
                         Q=Q,
                         R=R,
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         P_f=P_f,
                         data_points=data_points,
                         show_legend=show_legend,
                         epsilon=epsilon,
                         delta_T=delta_T,
                         last_norms_number=last_norms_number,
                         force_finite_horizon=force_finite_horizon,
                         debug=debug)

        S_matrices = [B_i @ inv(R_i) @ B_i.T if R_i.ndim > 1 else 1 / R_i * B_i @ B_i.T
                      for B_i, R_i in zip(self._B, self._R)]
        self.__cl = cl
        self.__dx_dt = None

        def __solve_N_coupled_diff_riccati(_: float, P_t: np.array) -> np.array:
            """
            odeint Coupled Differential Matrix Riccati Equations Solver function

            Parameters
            ----------
            _: float
                Integration point in time
            P_t: numpy array of numpy 2-d arrays, of len(N), of shape(n, n)
                Current integrated solution

            Returns
            ----------
            dP_tdt: numpy array of numpy 2-d arrays, of len(N), of shape(n, n)
                Current calculated value for the time-derivative of the matrices P_t
            """

            P_t = [(P_t[i * self._P_size:(i + 1) * self._P_size]).reshape(self._n, self._n) for i in range(self._N)]
            SP_sum = sum([S_i @ P_t_i for S_i, P_t_i in zip(S_matrices, P_t)])
            dP_tdt = np.zeros((self._P_size,))

            for i in range(self._N):
                P_t_i = P_t[i]
                Q_i = Q[i]

                dP_t_idt = - self._A.T @ P_t_i - P_t_i @ A - Q_i + P_t_i @ SP_sum  # open-loop

                if self.__cl:  # closed-loop
                    PS_sum_i = (SP_sum - S_matrices[i] @ P_t_i).T
                    dP_t_idt = dP_t_idt + PS_sum_i @ P_t_i

                dP_t_idt = dP_t_idt.reshape(self._P_size)
                dP_tdt = dP_t_idt if not i else np.concatenate((dP_tdt, dP_t_idt), axis=0)

            dP_tdt = np.ravel(dP_tdt)

            return dP_tdt

        self.__ode_func = __solve_N_coupled_diff_riccati

    def _update_K_from_last_state(self, t: int):
        """
        After the matrices P_i are obtained, this method updates the controllers at time t by the rule:
        K_i(t) = R^{-1}_ii B^T_i P_i(t)

        Parameters
        ----------
        t: int
            Current point in time
        """

        P_t = [(self._P[t][i * self._P_size:(i + 1) * self._P_size]).reshape(self._n, self._n) for i in range(self._N)]
        self._K = [(inv(R_ii) @ B_i.T if R_ii.ndim > 1 else 1 / R_ii * B_i.T) @ P_i
                   for R_ii, B_i, P_i in zip(self._R, self._B, P_t)]

    def _get_K_i(self, i: int) -> np.array:
        return self._K[i]

    def _update_P_from_last_state(self):
        """
        Evaluates the matrices P by backwards-solving a set of coupled time-continuous Riccati differential equations,
        with P_f as the terminal condition

        For the open-loop case:
        dP_i(t)/dt = - A^T P_i(t) - P_i(t) A - Q_i + P_i(t) sum_{j=1}^N B_j R^{-1}_{jj} B^T_j P_j(t)

        and for the closed loop case:
        dP_idt = - A^T P_i(t) - P_i(t) A - Q_i + P_i(t) sum_{j=1}^N B_j R^{-1}_{jj} B^T_j P_j(t) +
                    [sum_{j=1, j!=i}^N P_j(t) B_j R^{-1}_{jj} B^T_j] P_i(t)
        """

        self._P = odeint(func=self.__ode_func,
                         y0=np.ravel(self._P_f),
                         t=self._backward_time,
                         tfirst=True)

    def _solve_finite_horizon(self):
        """
        Considers the following set of finite-horizon cost functions:
        J_i = x(T_f)^T F_i(T_f) x(T_f) + int_{t=0}^T_f [x(t)^T Q_i x(t) + sum_{j=1}^N u_j(t)^T R_{ij} u_j(t)] dt

        In the continuous-time case, the matrices P_i can be solved for, unrelated to the controllers K_i.
        Then when simulating the state space progression, the controllers K_i can be computed as functions of P_i
        for each time interval
        """

        self._update_P_from_last_state()
        self._converged = True
        #
        # # plot the convergence of the values of the P matrices
        # self._plot(t=self._backward_time,
        #            mat=self._P,
        #            is_P=True)

    def _solve_infinite_horizon(self):
        """
        Considers the following finite-horizon cost functions:
        J_i = int_{t=0}^infty [x(t)^T Q_i x(t) + sum_{j=1}^N u_j(t)^T R_{ij} u_j(t)] dt
        """

        self._converge_DREs_to_AREs()

    def is_A_cl_stable(self) -> bool:
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

    @PyDiffGame._post_convergence
    def _solve_state_space(self):
        """
        Propagates the game forward through time and solves for it by solving the continuous differential equation:
        dx(t)/dt = A_cl(t) x(t)
        In each step, the controllers K_i are calculated with the values of P_i evaluated beforehand.
        """

        t = 0

        def state_diff_eqn(x_t: np.array, _: float) -> np.array:
            """
            odeint State Variables Solver function

            Parameters
            ----------
            x_t: numpy 1-d array of shape(n)
                Current integrated state variables
            _: float
                Current time

            Returns
            ----------
            dx_t_dt: numpy 1-d array of shape(n)
                Current calculated value for the time-derivative of the state variable vector x_t
            """

            nonlocal t

            self._update_K_from_last_state(t=t)
            self._update_A_cl_from_last_state()
            dx_t_dt = self._A_cl @ ((x_t - self._x_T) if self._x_T is not None else x_t)
            dx_t_dt = dx_t_dt.ravel()

            if t:
                t += 1

            return dx_t_dt

        self._x = odeint(func=state_diff_eqn,
                         y0=self._x_0,
                         t=self._forward_time)


class DiscretePyDiffGame(PyDiffGame):
    """
    Discrete differential game base class


    Considers the system:
    x[k+1] = A_tilda x[k] + sum_{j=1}^N B_j_tilda v_j[k]

    where A_tilda and each B_j_tilda are in discrete form, meaning they correlate to a discrete system,
    which, at the sampling points, is assumed to be equal to some equivalent continuous system of the form:
    dx(t)/dt = A x(t) + sum_{j=1}^N B_j v_j(t)

    Parameters
    ----------
    is_input_discrete: boolean, optional, default = False
        Indicates whether the input matrices A, B, Q, R are in discrete form or whether they need to be discretized
    """

    def __init__(self,
                 A: np.array,
                 B: list[np.array],
                 Q: list[np.array],
                 R: list[np.array],
                 is_input_discrete: bool = False,
                 x_0: np.array = None,
                 T_f: Union[float, int] = None,
                 P_f: list[np.array] = None,
                 data_points: int = PyDiffGame._data_points_default,
                 show_legend: bool = True,
                 epsilon: float = PyDiffGame._epsilon_default,
                 delta_T: float = PyDiffGame._delta_T_default,
                 last_norms_number: int = PyDiffGame._last_norms_number_default,
                 force_finite_horizon: bool = False,
                 debug: bool = False
                 ):

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
                         force_finite_horizon=force_finite_horizon,
                         debug=debug
                         )

        if not is_input_discrete:
            self.__discretize_game()

        self.__K_rows_num = sum([B_i.shape[1] for B_i in self._B])

        def __solve_for_K_k(K_k_previous: np.array, k_1: int) -> np.array:
            """
            fsolve Controllers Solver function

            Parameters
            ----------
            K_k_previous: np.array
                The previous estimation for the controllers at the k'th sample point
            k_1: int
                The k+1 index

            Returns
            ----------
            dP_tdt: list of numpy 2-d arrays, of len(N), of shape(n, n)
                Current calculated value for the time-derivative of the matrices P_t
            """

            K_k_previous = K_k_previous.reshape((self._N,
                                                 self.__K_rows_num,
                                                 self._n))
            P_k_1 = self._P[k_1]
            K_k = np.zeros_like(K_k_previous)

            for i in range(self._N):
                i_indices = (i, self.__get_K_i_shape(i))
                K_k_previous_i = K_k_previous[i_indices]
                P_k_1_i = P_k_1[i]
                R_ii = self._R[i]
                B_i = self._B[i]
                B_i_T = B_i.T

                A_cl_k_i = self._A

                for j in range(self._N):
                    if j != i:
                        B_j = self._B[j]
                        K_k_previous_j = K_k_previous[j, self.__get_K_i_shape(j)]
                        A_cl_k_i = A_cl_k_i - B_j @ K_k_previous_j

                K_k_i = K_k_previous_i - inv(R_ii + B_i_T @ P_k_1_i @ B_i) @ B_i_T @ P_k_1_i @ A_cl_k_i
                K_k[i_indices] = K_k_i

            return K_k.ravel()

        self.f_solve_func = __solve_for_K_k

    def __discretize_game(self):
        """
        The input to the discrete game can either given in continuous or discrete form.
        If it is not in discrete form, the game gets discretized using this method in the following manner:
        A = exp(delta_T A)
        B_j = int_{t=0}^delta_T exp(tA) dt B_j
        """

        A_tilda = np.exp(self._delta_T * self._A)
        e_AT, err = quadpy(lambda T: np.array([np.exp(t * self._A) for t in T]).swapaxes(0, 2).swapaxes(0, 1),
                           0,
                           self._delta_T)

        self._A = A_tilda
        B_tilda = e_AT
        self._B = [B_tilda @ B_i for B_i in self._B]
        self._Q = [self._delta_T * Q_i for Q_i in self._Q]
        self._R = [self._delta_T * R_i for R_i in self._R]

    def __simpson_integrate_Q(self):
        """
        Use Simpson's approximation for the definite integral of the state cost function term:
        x(t) ^T Q_f_i x(t) + int_{t=0}^T_f x(t) ^T Q_i x(t) ~ sum_{k=0)^K  x(t) ^T Q_i_k x(t)

        where:
        Q_i_0 = 1 / 3 * Q_i
        Q_i_1, ..., Q_i_K-1 = 4 / 3 * Q_i
        Q_i_2, ..., Q_i_K-2 = 2 / 3 * Q_i
        """
        Q = np.array(self._Q)
        self._Q = np.zeros((self._data_points,
                            self._N,
                            self._n,
                            self._n))
        self._Q[0] = 1 / 3 * Q
        self._Q[1::2] = 4 / 3 * Q
        self._Q[::2] = 2 / 3 * Q

    def __initialize_finite_horizon(self):
        """
        Initializes the calculated parameters for the finite horizon case by the following steps:
            1. The matrices P_i and controllers K_i are initialized randomly for all sample points
            2. The closed-loop dynamics matrix A_cl is first set to zero for all sample points
            3. The terminal conditions P_f_i are set to Q_i
            4. The resulting closed-loop dynamics matrix A_cl for the last sampling point is updated
            5. The state is initialized with its initial value, if given
         """

        self._P = np.random.rand(self._data_points,
                                 self._N,
                                 self._n,
                                 self._n)
        self._K = np.random.rand(self._data_points,
                                 self._N,
                                 self.__K_rows_num,
                                 self._n)
        self._A_cl = np.zeros((self._data_points,
                               self._n,
                               self._n))
        self.__simpson_integrate_Q()
        self._P[-1] = np.array(self._Q[-1]).reshape((self._N,
                                                     self._n,
                                                     self._n))
        self._update_A_cl_from_last_state(k=-1)
        self._x = np.zeros((self._data_points,
                            self._n))
        self._x[0] = self._x_0

    def __get_K_i_shape(self, i: int) -> range:
        """
        Returns the i'th controller shape indices

        Parameters
        ----------
        i: int
            The desired controller index
        """

        K_i_offset = sum([self._B[j].shape[1] for j in range(i)]) if i else 0
        first_K_i_index = K_i_offset
        second_K_i_index = first_K_i_index + self._B[i].shape[1]

        return range(first_K_i_index, second_K_i_index)

    def _update_K_from_last_state(self, k_1: int):
        """
        After initializing, in order to solve for the controllers K_i and matrices P_i,
        we start by solving for the controllers by the update rule:
        K_i(t) = R^{-1}_ii B^T_i P_i(t)

        Parameters
        ----------
        k_1: int
            The current k+1 sample index
        """

        K_k_1 = self._K[k_1]
        K_k_0 = np.random.rand(*K_k_1.shape)
        K_k = fsolve(func=self.f_solve_func,
                     x0=K_k_0,
                     args=tuple([k_1]))
        K_k_reshaped = np.array(K_k).reshape((self._N,
                                              self.__K_rows_num,
                                              self._n))

        if self._debug:
            k = k_1 - 1
            f_K_k = self.f_solve_func(K_k_previous=K_k, k_1=k)
            close_to_zero_array = np.isclose(f_K_k, np.zeros_like(K_k))
            is_close_to_zero = all(close_to_zero_array)
            print(f"The current solution is {'NOT ' if not is_close_to_zero else ''}close to zero"
                  f"\nK_k:\n{K_k_reshaped}\nf_K_k:\n{f_K_k}")

        self._K[k_1 - 1] = K_k_reshaped

    def _get_K_i(self, i: int, k: int) -> np.array:
        return self._K[k][i, self.__get_K_i_shape(i)]

    def _update_P_from_last_state(self, k_1: int):
        """
        Updates the matrices P_i with
        A_cl[k] = A - sum_{i=1}^N B_i K_i[k]

        Parameters
        ----------
        k_1: int
            The current k'th sample index
        """

        k = k_1 - 1
        P_k_1 = self._P[k_1]
        K_k = self._K[k]
        A_cl_k = self._A_cl[k]
        Q_k = self._Q[k]
        P_k = np.zeros_like(P_k_1)

        for i in range(self._N):
            K_k_i = K_k[i, self.__get_K_i_shape(i)]
            P_k_1_i = P_k_1[i]
            R_ii = self._R[i]
            Q_k_i = Q_k[i]

            P_k[i] = A_cl_k.T @ P_k_1_i @ A_cl_k + (K_k_i.T @ R_ii if R_ii.ndim > 1 else K_k_i.T * 1 / R_ii) \
                     @ K_k_i + Q_k_i

        self._P[k] = P_k

    def is_A_cl_stable(self, k: int) -> bool:
        """
        Tests Lyapunov stability of the closed loop:
        A discrete dynamic system governed by the matrix A_cl has Lyapunov stability iff:
        |eig| <= 1 forall eigenvalues of A_cl
        """

        A_cl_k = self._A_cl[k]
        A_cl_k_eigenvalues = eigvals(A_cl_k)
        A_cl_k_eigenvalues_absolute_values = [abs(eig) for eig in A_cl_k_eigenvalues]

        if self._debug:
            print(A_cl_k_eigenvalues_absolute_values)

        stability = all([eig_norm < 1 for eig_norm in A_cl_k_eigenvalues_absolute_values])

        return stability

    def _solve_finite_horizon(self):
        """
        Solves the system with the finite-horizon cost functions:
        J_i = sum_{k=1}^T_f [ x[k]^T Q_i x[k] + sum_{j=1}^N u_j[k]^T R_{ij} u_j[k] ]

        In the discrete-time case, the matrices P_i have to be solved simultaneously with the controllers K_i
        """

        pass

    def _solve_infinite_horizon(self):
        """
        Solves the system with the infinite-horizon cost functions:
        J_i = sum_{k=1}^infty [ x[k]^T Q_i x[k] + sum_{j=1}^N u_j[k]^T R_{ij} u_j[k] ]
        """

        self.__initialize_finite_horizon()

        for backwards_index, t in enumerate(self._backward_time[:-1]):
            k_1 = self._data_points - 1 - backwards_index
            k = k_1 - 1

            if self._debug:
                print('#' * 30 + f' t = {round(t, 3)} ' + '#' * 30)

            self._update_K_from_last_state(k_1=k_1)

            if self._debug:
                print(f'K_k+1:\n{self._K[k_1]}')

            self._update_A_cl_from_last_state(k)

            if self._debug:
                print(f'A_cl_k:\n{self._A_cl[k]}')
                stable_k = self.is_A_cl_stable(k)
                print(f"The system is {'NOT ' if not stable_k else ''}stable")

            self._update_P_from_last_state(k)

    @PyDiffGame._post_convergence
    def _solve_state_space(self):
        """
        Propagates the game through time and solves for it by solving the discrete difference equation:
        x[k+1] = A_cl[k] x[k]
        In each step, the controllers K_i are calculated with the values of P_i evaluated beforehand.
        """

        x_1_k = self._x_0

        for k in range(1, self._data_points - 1):
            A_cl_k = self._A_cl[k]
            x_k = A_cl_k @ x_1_k
            self._x[k] = x_k
            x_1_k = x_k
