# imports
from __future__ import annotations
import numpy as np
import math
from numpy.linalg import eigvals, norm
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, solve_discrete_are
import warnings
from typing import Callable, Any
from abc import ABC, abstractmethod


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

        self.__continuous = self.__class__.__name__ == 'ContinuousPyDiffGame'

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

        print(self.__infinite_horizon)

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
        converged = False

        while not converged:
            self._backward_time = np.linspace(self._T_f, self._T_f - self._delta_T, self._data_points)
            self._update_P_from_last_state()
            self._P_f = self._P[-1]
            last_norms += [norm(self._P_f)]

            if len(last_norms) > self.__last_norms_number:
                last_norms.pop(0)

            if len(last_norms) == self.__last_norms_number:
                converged = all([abs(norm_i - norm_i1) < self.__epsilon for norm_i, norm_i1
                                 in zip(last_norms, last_norms[1:])])

            self._T_f -= self._delta_T

    def _post_convergence(method: Callable) -> Callable:
        """
        A decorator static-method to apply on methods that can only be called after convergence
        """

        def verify_convergence(self: PyDiffGame, *args, **kwargs) -> Any:
            if not self._converged:
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

        self._converged = True

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

    def calculate_costs(self, add_noise: bool = False) -> np.array:

        def cost_function_as_quadratic_expression(i: int) -> int:
            P_f_i = (self._P_f[i * self._P_size:(i + 1) * self._P_size]).reshape(self._n, self._n)

            if add_noise:
                costs = [int(self._x[-1].T @ (P_f_i +
                                              np.random.normal(loc=0,
                                                               scale=np.mean(P_f_i),
                                                               size=P_f_i.shape)) @ self._x[-1])
                         for _ in range(100000)]

                return math.ceil(np.mean(costs))

            return int(self._x[-1].T @ P_f_i @ self._x[-1])

        J_quadratic = [cost_function_as_quadratic_expression(i) for i in range(self._N)]

        return np.array(J_quadratic)

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