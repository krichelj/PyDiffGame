# imports
from __future__ import annotations
import time
import numpy as np
from numpy.linalg import eigvals, norm, LinAlgError
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, solve_discrete_are
import warnings
from typing import Callable, Union
from abc import ABC, abstractmethod


class PyDiffGame(ABC):
    """
    Differential game abstract base class


    Parameters
    ----------
    A: numpy 2-d array of shape(n, n)
        The system dynamics matrix
    B: numpy array of numpy 2-d arrays of len(N), each matrix B_i of shape(n, m_i)
        System input matrices for each control objective
    Q: numpy array of numpy 2-d arrays of len(N), each matrix Q_i of shape(n, M)
        Cost function state weights for each control objective
    R: numpy array of numpy 2-d arrays of len(N), each matrix R_i of shape(m_i, m_i)
        Cost function input weights for each control objective
    x_0: numpy 1-d array of shape(n), optional
        Initial state vector
    x_T: numpy 1-d array of shape(n), optional
        Final state vector, in case of signal tracking
    T_f: positive float, optional, default = 10
        System dynamics horizon. Should be given in the case of finite horizon
    P_f: numpy array of numpy 2-d arrays of len(N), each matrix P_f_i of shape(n, n), optional,
        default = uncoupled solution of scipy's solve_are
        Final condition for the Riccati equation matrix. Should be given in the case of finite horizon
    show_legend: boolean, optional, default = True
        Indicates whether to display a legend in the plots (True) or not (False)
    epsilon: float, optional, default = 1 / (10 ** 8)
        The convergence threshold for numerical convergence
    L: int, optional, default = 1000
        Number of data points
    last_norms_number: int, optional, default = 5
        The number of last matrix norms to consider for convergence
    debug: boolean, optional, default = False
        Indicates whether to display debug information or not
    """

    # class fields
    __T_f_default: int = 10
    _L_default: int = 1000
    _epsilon_default: float = 10 ** (-3)
    _last_norms_number_default: int = 5

    def __init__(self,
                 A: np.array,
                 B: Union[list[np.array], np.array],
                 Q: Union[list[np.array], np.array],
                 R: Union[list[np.array], np.array],
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 P_f: list[np.array] = None,
                 show_legend: bool = True,
                 epsilon: float = _epsilon_default,
                 L: int = _L_default,
                 last_norms_number: int = _last_norms_number_default,
                 force_finite_horizon: bool = False,
                 debug: bool = False
                 ):

        self.__continuous = 'ContinuousPyDiffGame' in [c.__name__ for c in type(self).__bases__]

        # input and shape parameters
        self._A = A
        self._B = B if isinstance(B, list) else [B]
        self._n = self._A.shape[0]
        self._N = len(self._B)
        self._P_size = self._n ** 2
        self._Q = Q if isinstance(Q, list) else [Q]
        self._R = R if isinstance(R, list) else [R]
        self._x_0 = x_0
        self._x_T = x_T
        self.__infinite_horizon = (not force_finite_horizon) and (T_f is None or P_f is None)
        self._T_f = PyDiffGame.__T_f_default if T_f is None else T_f
        self._P_f = self.__get_are_P_f() if P_f is None else P_f
        self._L = L
        self._delta = self._T_f / self._L
        self.__show_legend = show_legend
        self._forward_time = np.linspace(start=0, stop=self._T_f, num=self._L)
        self._backward_time = self._forward_time[::-1]

        self.__verify_input()

        # additional parameters
        self._A_cl = np.empty_like(self._A)
        self._P = []
        self._K: list[np.array] = []

        self.__epsilon = epsilon

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
        if self._L <= 0:
            raise ValueError('The number of data points must be a positive integer')

    def _converge_DREs_to_AREs(self):
        """
        Solves the game as backwards convergence of the differential
        finite-horizon game for repeated consecutive steps until the matrix norm converges
        """

        last_norms = []
        converged = False
        T_f_curr = self._T_f

        while not converged:
            self._backward_time = np.linspace(T_f_curr, T_f_curr - self._delta, self._L)
            self._update_Ps_from_last_state()
            self._P_f = self._P[-1]
            last_norms += [norm(self._P_f)]

            if len(last_norms) > self.__last_norms_number:
                last_norms.pop(0)

            if len(last_norms) == self.__last_norms_number:
                converged = all([abs(norm_i - norm_i1) < self.__epsilon for norm_i, norm_i1
                                 in zip(last_norms, last_norms[1:])])

            T_f_curr -= self._delta

    def _post_convergence(method: Callable) -> Callable:
        """
        A decorator static-method to apply on methods that need only be called after convergence
        """

        def verify_convergence(self: PyDiffGame, *args, **kwargs):
            if not self._converged:
                raise RuntimeError('Must first simulate the differential game')
            return method(self, *args, **kwargs)

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

    def __plot_variables(self, mat: np.array):
        """
        Displays plots for the state variables with respect to time

        Parameters
        ----------

        mat: numpy array
            The y-axis information to plot
        """

        self._plot(t=self._forward_time,
                   mat=mat,
                   is_P=False,
                   title=f"{('Continuous' if self.__continuous else 'Discrete')}, "
                         f"{('Infinite' if self.__infinite_horizon else 'Finite')} Horizon, {self._N}-Player Game, "
                         f"{self._L} sampling points")

    def _plot_state_space(self):
        """
        Plots the state vector variables wth respect to time
        """

        self.__plot_variables(mat=self._x)

    def _plot_Y(self, C: np.array):
        """
        Plots the output vector variables wth respect to time

        Parameters
        ----------

        C: numpy array
            The output coefficients with respect to state
        """

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
        P_f = []

        for B_i, Q_i, R_ii in zip(self._B, self._Q, self._R):
            try:
                P_f_i = are_solver(self._A, B_i, Q_i, R_ii)
            except LinAlgError:
                rand = np.random.rand(*self._A.shape)
                P_f_i = rand @ rand.T

            P_f += [P_f_i]

        return P_f

    @abstractmethod
    def _update_K_from_last_state(self, *args):
        """
        Updates the controllers K after forward propagation of the state through time
        """

        pass

    @abstractmethod
    def _get_K_i(self, *args) -> np.array:
        """
        Returns the i'th element of the controllers K
        """

        pass

    @abstractmethod
    def _get_P_f_i(self, i: int) -> np.array:
        """
        Returns the i'th element of the final condition matrices P_f

        Parameters
        ----------
        i: int
            The required index
        """

        pass

    def _update_A_cl_from_last_state(self, k: int = None):
        """
        Updates the closed-loop dynamics with the updated controllers based on the relation:
        A_cl = A - sum_{i=1}^N B_i K_i

        Parameters
        ----------
        k: int, optional
            The current k'th sample index, in case the controller is time-dependant.
            In this case the update rule is:
            A_cl[k] = A - sum_{i=1}^N B_i K_i[k]
        """

        A_cl = self._A - sum([self._B[i] @ (self._get_K_i(i, k) if k is not None else self._get_K_i(i))
                              for i in range(self._N)])
        if k is not None:
            self._A_cl[k] = A_cl
        else:
            self._A_cl = A_cl

    @abstractmethod
    def _update_Ps_from_last_state(self, *args):
        """
        Updates the matrices {P_i}_{i=1}^N after forward propagation of the state through time
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

    @_post_convergence
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

        self._converged = True

    @_post_convergence
    def simulate_state_space(self):
        if self._x_0 is not None:
            if self._debug:
                print(f"The system is {'NOT ' if not self.is_A_cl_stable() else ''}stable")

            self._solve_state_space()

    def solve_game_and_simulate_state_space(self):
        """
        Propagates the game through time, solves for it and plots the state with respect to time
        """

        self.solve_game()
        self.simulate_state_space()

    def solve_game_simulate_state_space_and_plot(self):
        self.solve_game_and_simulate_state_space()

        if self._x_0 is not None:
            self._plot_state_space()

    @_post_convergence
    def get_costs(self) -> np.array:
        """
        Calculates the cost function value using the formula:
        J_i = int_{t=0}^T_f [ x(t)^T ( Q_i + sum_{j=1}^N  K_j(t)^T R_{ij} K_j(t) ) x(t) ] dt
        """

        costs = []

        for i in range(self._N):
            Q_i = self._Q[i]
            R_ii = self._R[i]

            cost_i = 0

            for l in range(self._L):
                x_l = self._x[l]
                K_i_l = self._get_K_i(i) if self.__continuous else self._get_K_i(i, l)
                cost_i += x_l.T @ (Q_i + (K_i_l.T @ R_ii @ K_i_l if R_ii.ndim > 1 else R_ii * K_i_l.T @ K_i_l)) @ x_l

            costs += [cost_i]

        return np.array(costs)

    def __len__(self) -> int:
        """
        We define the length of a differential game to be its number of objectives (N)
        """

        return self._N

    def __eq__(self, other: PyDiffGame) -> bool:
        """
        Let G1, J1 and G2, J2 be two differential games and their respective costs.
        We define G1 == G2 iff:
             - len(G1) == len(G2)
             - sum_{i=1}^N J1[i] == sum_{i=1}^N J2[i]
        """

        if len(self) != len(other):
            return False

        return self.get_costs().sum() == other.get_costs().sum()

    def __lt__(self, other: PyDiffGame) -> bool:
        """
        Let G1, J1 and G2, J2 be two differential games and their respective costs.
        We define G1 < G2 iff:
            - len(G1) == len(G2) := N, else they are not comparable
            - sum_{i=1}^N J1[i] < sum_{i=1}^N J2[i]
        """

        if len(self) != len(other):
            raise ValueError('The lengths of the differential games do not match, so they are non-comparable')

        return self.get_costs().sum() < other.get_costs().sum()

    _post_convergence = staticmethod(_post_convergence)


def timed(f: Callable) -> Callable:
    def time_wrapper(*args, **kwargs):
        start = time.time()
        output = f(*args, **kwargs)
        end = time.time()
        print(f'{f.__name__} function took {(end - start) * 1000.0} ms')

        return output

    return time_wrapper