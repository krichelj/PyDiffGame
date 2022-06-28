from __future__ import annotations

import os
from pathlib import Path
import time
from scipy import interpolate
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
    A: 2-d np.array of shape(n, n)
        The system dynamics matrix
    B: list of 2-d np.arrays of len(N), each array B_i of shape(n, m_i)
        System input matrices for each control objective
    Q: list of 2-d np.arrays of len(N), each array Q_i of shape(n, n)
        Cost function state weights for each control objective
    R: list of 2-d np.arrays of len(N), each array R_i of shape(m_i, m_i)
        Cost function input weights for each control objective
    x_0: 1-d np.array of shape(n), optional
        Initial state vector
    x_T: 1-d np.array of shape(n), optional
        Final state vector, in case of signal tracking
    T_f: positive float, optional, default = 10
        System dynamics horizon. Should be given in the case of finite horizon
    P_f: list of 2-d np.arrays of len(N), each array P_f_i of shape(n, n), optional
        default = uncoupled solution of scipy's solve_are
        Final condition for the Riccati equation array. Should be given in the case of finite horizon
    show_legend: boolean, optional, default = True
        Indicates whether to display a legend in the plots
    state_variables_names: list of strings of len(n), optional
        The state variables' names to display
    epsilon: float in the interval (0,1), optional, default = 10 ** (-7)
        Numerical convergence threshold
    L: positive int, optional, default = 1000
        Number of data points
    eta: positive int, optional, default = 5
        The number of last matrix norms to consider for convergence
    debug: boolean, optional, default = False
        Indicates whether to display debug information
    """

    # class fields
    __T_f_default: int = 5
    _L_default: int = 1000
    _epsilon_default: float = 10 ** (-7)
    _eta_default: int = 5
    _g = 9.81

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
                 state_variables_names: list[str] = None,
                 epsilon: float = _epsilon_default,
                 L: int = _L_default,
                 eta: int = _eta_default,
                 force_finite_horizon: bool = False,
                 debug: bool = False):

        self.__continuous = 'ContinuousPyDiffGame' in [c.__name__ for c in type(self).__bases__]
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
        self.__state_variables_names = state_variables_names
        self._forward_time = np.linspace(start=0,
                                         stop=self._T_f,
                                         num=self._L)
        self._backward_time = self._forward_time[::-1]
        self._A_cl = np.empty_like(self._A)
        self._P: list[np.array] = []
        self._K: list[np.array] = []
        self.__epsilon = epsilon
        self._x = self._x_0
        self.__eta = eta
        self._converged = False
        self._debug = debug
        self._fig = None

        self._verify_input()

    def _verify_input(self):
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
                    and (np.all(eigvals(R_i) > 0) if R_i.ndim > 1 else R_i > 0)
                    for B_i, R_i in zip(self._B, self._R)]):
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
        if len(self.__state_variables_names) != self._n:
            raise ValueError('The parameter state_variables_names must be of length n')
        if not 0 < self.__epsilon < 1:
            raise ValueError('The convergence tolerance epsilon must be in the open interval (0,1)')
        if self._L <= 0:
            raise ValueError('The number of data points must be a positive integer')
        if self.__eta <= 0:
            raise ValueError('The number of last matrix norms to consider for convergence must be a positive integer')

    def _converge_DREs_to_AREs(self):
        """
        Solves the game as backwards convergence of the differential
        finite-horizon game for repeated consecutive steps until the matrix norm converges
        """

        x_converged = False
        P_converged = False
        last_norms = []
        curr_iteration_T_f = self._T_f
        x_T = self._x_T if self._x_T is not None else np.zeros_like(self._x_0)

        while not x_converged:

            while not P_converged:
                self._backward_time = np.linspace(start=self._T_f,
                                                  stop=self._T_f - self._delta,
                                                  num=self._L)
                self._update_Ps_from_last_state()
                self._P_f = self._P[-1]
                last_norms += [norm(self._P_f)]

                if len(last_norms) > self.__eta:
                    last_norms.pop(0)

                if len(last_norms) == self.__eta:
                    P_converged = all([abs(norm_i - norm_i1) < self.__epsilon for norm_i, norm_i1
                                       in zip(last_norms, last_norms[1:])])
                self._T_f -= self._delta

            self._T_f = curr_iteration_T_f

            if self._x_0 is not None:
                curr_x_T_f_norm = self.simulate_x_T_f()
                x_converged = norm(curr_x_T_f_norm - x_T) < self.__epsilon
                curr_iteration_T_f += 1
                self._T_f = curr_iteration_T_f
                self._forward_time = np.linspace(start=0,
                                                 stop=self._T_f,
                                                 num=self._L)
            else:
                x_converged = True

    def _post_convergence(f: Callable) -> Callable:
        """
        A decorator static-method to apply on methods that need only be called after convergence

        Parameters
        ----------
        f: callable
            The method requiring convergence

        Returns
        ----------
        decorated_f: callable
            A decorated version of the method requiring convergence
        """

        def decorated_f(self: PyDiffGame, *args, **kwargs):
            if not self._converged:
                raise RuntimeError('Must first simulate the differential game')
            return f(self, *args, **kwargs)

        return decorated_f

    @_post_convergence
    def _plot_temporal_variables(self,
                                 t: np.array,
                                 variables: np.array,
                                 is_P: bool,
                                 title: str = None,
                                 two_state_spaces: bool = False):
        """
        Displays plots for the state variables with respect to time and the convergence of the values of P

        Parameters
        ----------
        t: 1-d np.array array of len(L)
            The time axis information to plot
        variables: 2-d np.arrays of shape(L, n)
            The y-axis information to plot
        is_P: boolean
            Indicates whether to accommodate plotting for all the values in P_i or just for x
        title: str, optional
            The plot title to display
        two_state_spaces: bool, optional
            Indicates whether to plot two state spaces
        """

        num_of_variables = variables.shape[1]
        variables_range = range(1, num_of_variables + 1)

        self._fig = plt.figure(dpi=150)
        self._fig.set_size_inches(8, 6)
        plt.plot(t, variables)
        plt.xlabel('Time')

        if title:
            plt.title(title)

        if self.__show_legend:
            if is_P:
                labels = ['${P' + str(i) + '}_{' + str(j) + str(k) + '}$' for i in range(1, self._N + 1)
                          for j in variables_range for k in variables_range]
            elif self.__state_variables_names is not None:
                if two_state_spaces:
                    labels = [f'${name}_{1}$' for name in self.__state_variables_names] + \
                             [f'${name}_{2}$' for name in self.__state_variables_names]
                else:
                    labels = [f'${name}$' for name in self.__state_variables_names]
            else:
                labels = ["${\mathbf{x}}_{" + str(j) + "}$" for j in variables_range]

            plt.legend(labels=labels,
                       loc='upper left' if is_P else 'best',
                       ncol=int(num_of_variables / 2),
                       prop={'size': int(120 / num_of_variables)})

        plt.grid()

        plt.show()

    def __get_temporal_state_variables_title(self,
                                             linear_system: bool = True,
                                             two_state_spaces=False):
        if two_state_spaces:
            title = f"{self._N}-Player Game"
        else:
            title = f"{('Linear' if linear_system else 'Nonlinear')}, " \
                    f"{('Continuous' if self.__continuous else 'Discrete')}, " \
                    f"{('Infinite' if self.__infinite_horizon else 'Finite')} Horizon, " \
                    f"{self._N}-Player Game" \
                    f"{f', {self._L} Sampling Points'}"

        return title

    def _plot_temporal_state_variables(self,
                                       state_variables: np.array,
                                       linear_system: bool = True):
        """
        Displays plots for the state variables with respect to time

        Parameters
        ----------

        state_variables: 2-d np.array of shape(L, n)
            The temporal state variables y-axis information to plot
        """

        self._plot_temporal_variables(t=self._forward_time,
                                      variables=state_variables,
                                      is_P=False,
                                      title=self.__get_temporal_state_variables_title(linear_system=linear_system))

    def __plot_x(self):
        """
        Plots the state vector variables wth respect to time
        """

        self._plot_temporal_state_variables(state_variables=self._x)

    def __plot_y(self,
                 C: np.array):
        """
        Plots an output vector y = C x^T wth respect to time

        Parameters
        ----------

        C: 2-d np.array array of shape(p, n)
            The output coefficients with respect to state
        """

        y = C @ self._x.T
        self._plot_temporal_state_variables(state_variables=y.T)

    @_post_convergence
    def save_figure(self,
                    figure_path: Union[str, Path] = Path(fr'{os.getcwd()}/figures'),
                    filename: str = 'last_image'):
        """
        Saves the current figure

        Parameters
        ----------

        figure_path: str or Path, optional, default = figures sub-folder of the current directory
            The desired saved figure's file path
        filename: str, optional
            The desired saved figure's filename
        """

        if self._x_0 is None:
            raise RuntimeError('No parameter x_0 was defined to illustrate a figure')

        if not figure_path.is_dir():
            os.mkdir(figure_path)

        self._fig.savefig(Path(fr'{figure_path}/{filename}'))

    def __augment_two_state_space_vectors(self,
                                          other: PyDiffGame) -> (np.array, np.array):
        """
        Plots the state variables of two converged state spaces

        Parameters
        ----------
        other: PyDiffGame
            Other converged differential game to augment the current one with

        Returns
        ----------
        longer_period: 1-d np.array of len(max(self.L, other.L))
            The longer time period of the two games
        augmented_state_space: 2-d np.array of shape(max(self.L, other.L), self.n + other.n)
            Augmented state space with interpolated and extrapolated values for both games
        """

        f_self = interpolate.interp1d(x=self._forward_time,
                                      y=self._x,
                                      axis=0,
                                      fill_value="extrapolate")

        f_other = interpolate.interp1d(x=other.forward_time,
                                       y=other.x,
                                       axis=0,
                                       fill_value="extrapolate")

        longer_period = self._forward_time if max(self._forward_time) > max(other.forward_time) \
            else other.forward_time

        self_state_space = f_self(longer_period)
        other_state_space = f_other(longer_period)

        augmented_state_space = np.concatenate((self_state_space, other_state_space), axis=1)

        return longer_period, augmented_state_space

    def plot_two_state_spaces(self,
                              other: PyDiffGame):
        """
        Plots the state variables of two converged state spaces

        Parameters
        ----------

        other: PyDiffGame
            Other converged differential game to plot along with the current one
        """

        longer_period, augmented_state_space = self.__augment_two_state_space_vectors(other)

        self_title = self.__get_temporal_state_variables_title(two_state_spaces=True)
        other_title = other.__get_temporal_state_variables_title(two_state_spaces=True)

        self._plot_temporal_variables(t=longer_period,
                                      variables=augmented_state_space,
                                      is_P=False,
                                      title=f"{self_title} " + "$(T_{f_1} = " +
                                            f"{self._T_f})$ \nvs\n {other_title} " + "$(T_{f_2} = " + f"{other.T_f})$",
                                      two_state_spaces=True
                                      )

    def __get_are_P_f(self) -> list[np.array]:
        """
        Solves the uncoupled set of algebraic Riccati equations to use as initial guesses for fsolve and odeint

        Returns
        ----------
        P_f: list of 2-d np.arrays, of len(N), of shape(n, n), solution of scipy's solve_are
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
    def _update_K_from_last_state(self,
                                  *args):
        """
        Updates the controllers K after forward propagation of the state through time
        """

        pass

    @abstractmethod
    def _get_K_i(self,
                 *args) -> np.array:
        """
        Returns the i'th element of the currently calculated controllers

        Returns
        ----------
        K_i: numpy array of numpy 2-d arrays, of len(N), of shape(n, n), solution of scipy's solve_are
            Final condition for the Riccati equation matrix
        """

        pass

    @abstractmethod
    def _get_P_f_i(self,
                   i: int) -> np.array:
        """
        Returns the i'th element of the final condition matrices P_f

        Parameters
        ----------
        i: int
            The required index
        """

        pass

    def _update_A_cl_from_last_state(self,
                                     k: int = None):
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
    def _update_Ps_from_last_state(self,
                                   *args):
        """
        Updates the matrices {P_i}_{i=1}^N after forward propagation of the state through time
        """

        pass

    @abstractmethod
    def is_A_cl_stable(self,
                       *args) -> bool:
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
    def __plot_finite_horizon_convergence(self):
        """
        Plots the convergence of the values for the matrices P_i
        """

        self._plot_temporal_variables(t=self._backward_time,
                                      variables=self._P,
                                      is_P=True)

    def __solve_and_plot_finite_horizon(self):
        """
        Solves for the finite horizon case and plots the convergence of the values for the matrices P_i
        """

        self._solve_finite_horizon()
        self.__plot_finite_horizon_convergence()

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

    def __solve_game(self):
        if self.__infinite_horizon:
            self._solve_infinite_horizon()
        else:
            self.__solve_and_plot_finite_horizon()

        self._converged = True

    def __solve_game_and_simulate_state_space(self):
        """
        Propagates the game through time, solves for it and plots the state with respect to time
        """

        self.__solve_game()

        if self._x_0 is not None:
            self._solve_state_space()

    def __solve_game_simulate_state_space_and_plot(self):
        """
        Propagates the game through time, solves for it and plots the state with respect to time
        """

        self.__solve_game_and_simulate_state_space()

        if self._x_0 is not None:
            self.__plot_x()

    def simulate_x_T_f(self) -> np.array:
        """
        Evaluates the space vector at the terminal time T_f

        Returns
        ----------
        x_T_f: numpy 1-d array of shape(n)
            Evaluated terminal state vector x(T_f)
        """

        pass

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

    def run_simulation(self,
                       plot_state_space: bool = True):
        if plot_state_space:
            self.__solve_game_simulate_state_space_and_plot()
        else:
            self.__solve_game_and_simulate_state_space()

    @property
    def P(self) -> list:
        return self._P

    @property
    def x(self) -> np.array:
        return self._x

    @property
    def forward_time(self) -> np.array:
        return self._forward_time

    @property
    def T_f(self) -> float:
        return self._T_f

    def __len__(self) -> int:
        """
        We define the length of a differential game to be its number of objectives (N)
        """

        return self._N

    def __eq__(self,
               other: PyDiffGame) -> bool:
        """
        Let G1, J1 and G2, J2 be two differential games and their respective costs.
        We define G1 == G2 iff:
             - len(G1) == len(G2)
             - sum_{i=1}^N J1[i] == sum_{i=1}^N J2[i]
        """

        return len(self) == len(other) and self.get_costs().sum() == other.get_costs().sum()

    def __lt__(self,
               other: PyDiffGame) -> bool:
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
