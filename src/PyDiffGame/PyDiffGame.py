from __future__ import annotations

import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sp
import sympy
import warnings
from typing import Callable, Final, ClassVar, Optional, Sequence
from abc import ABC, abstractmethod

from PyDiffGame.Objective import Objective, GameObjective, LQRObjective


class PyDiffGame(ABC, Callable, Sequence):
    """
    Differential game abstract base class


    Parameters
    ----------

    A: 2-d np.array of shape(n, n)
        The system dynamics matrix
    B: sequence of 2-d np.arrays of len(N), each array B_i of shape(n, m_i)
        System input matrices for each control objective
    objectives: sequence of Objective objects
        Desired objectives for the game
    x_0: 1-d np.array of shape(n), optional
        Initial state vector
    x_T: 1-d np.array of shape(n), optional
        Final state vector, in case of signal tracking
    T_f: positive float, optional
        System dynamics horizon. Should be given in the case of finite horizon
    P_f: sequence of 2-d np.arrays of len(N), each array P_f_i of shape(n, n), optional
        default = uncoupled solution of scipy's solve_are
        Final condition for the Riccati equation array. Should be given in the case of finite horizon
    show_legend: bool, optional
        Indicates whether to display a legend in the plots
    state_variables_names: sequence of strings of len(n), optional
        The state variables' names to display
    epsilon_x: float in the interval (0,1), optional
        Numerical convergence threshold for the state vector of the system
    epsilon_P: float in the interval (0,1), optional
        Numerical convergence threshold for the matrices P_i
    L: positive int, optional
        Number of data points
    eta: positive int, optional
        The number of last matrix norms to consider for convergence
    debug: bool, optional
        Indicates whether to display debug information
    """

    # class fields
    __T_f_default: Final[ClassVar[int]] = 20
    _epsilon_x_default: Final[ClassVar[float]] = 10e-7
    _epsilon_P_default: Final[ClassVar[float]] = 10e-7
    _eigvals_tol: Final[ClassVar[float]] = 10e-7
    _L_default: Final[ClassVar[int]] = 1000
    _eta_default: Final[ClassVar[int]] = 3
    _g: Final[ClassVar[float]] = 9.81
    __x_max_convergence_iterations: Final[ClassVar[int]] = 25
    __p_max_convergence_iterations: Final[ClassVar[int]] = 25

    _default_figures_path = Path(fr'{os.getcwd()}/figures')
    _default_figures_filename = 'image'

    @classmethod
    @property
    def epsilon_x_default(cls) -> float:
        return cls._epsilon_x_default

    @classmethod
    @property
    def epsilon_P_default(cls) -> float:
        return cls._epsilon_P_default

    @classmethod
    @property
    def L_default(cls) -> int:
        return cls._L_default

    @classmethod
    @property
    def eta_default(cls) -> float:
        return cls._eta_default

    @classmethod
    @property
    def g(cls) -> float:
        return cls._g

    @classmethod
    @property
    def default_figures_path(cls) -> Path:
        return cls._default_figures_path

    @classmethod
    @property
    def default_figures_filename(cls) -> str:
        return cls._default_figures_filename

    def __init__(self,
                 A: np.array,
                 B: Optional[np.array] = None,
                 Bs: Optional[Sequence[np.array]] = None,
                 Qs: Optional[Sequence[np.array]] = None,
                 Rs: Optional[Sequence[np.array]] = None,
                 Ms: Optional[Sequence[np.array]] = None,
                 objectives: Optional[Sequence[Objective]] = None,
                 x_0: Optional[np.array] = None,
                 x_T: Optional[np.array] = None,
                 T_f: Optional[float] = None,
                 P_f: Optional[Sequence[np.array] | np.array] = None,
                 show_legend: Optional[bool] = True,
                 state_variables_names: Optional[Sequence[str]] = None,
                 epsilon_x: Optional[float] = _epsilon_x_default,
                 epsilon_P: Optional[float] = _epsilon_P_default,
                 L: Optional[int] = _L_default,
                 eta: Optional[int] = _eta_default,
                 force_finite_horizon: Optional[bool] = False,
                 debug: Optional[bool] = False):

        self.__continuous = 'ContinuousPyDiffGame' in [c.__name__ for c in
                                                       set(type(self).__bases__).union({self.__class__})]
        self._A = A
        self._B = B
        self._Bs = Bs
        self._n = self._A.shape[0]
        self._m = self._B.shape[1] if self._B is not None else self._Bs[0].shape[1]
        self.__objectives = objectives
        self._N = len(objectives) if objectives else len(Qs)
        self._P_size = self._n ** 2
        self._Qs = Qs if Qs else [o.Q for o in objectives]
        self._Rs = Rs if Rs else [o.R for o in objectives]
        self._x_0 = x_0
        self._x_T = x_T
        self.__infinite_horizon = (not force_finite_horizon) and (T_f is None or P_f is None)
        self._T_f = self.__T_f_default if T_f is None else T_f

        self._Ms = Ms

        if self._Ms is None and self.__objectives is not None:
            self._Ms = [o.M_i for o in self.__objectives if isinstance(o, GameObjective)]

        self._M = None

        if self._Bs is None:
            if self._Ms is not None and len(self._Ms):
                self._M = np.concatenate(self._Ms, axis=0)
                self._M_inv = np.linalg.inv(self._M)
                l = 0
                self._Bs = []

                for M_i in self._Ms:
                    m_i = M_i.shape[0]
                    M_i_tilde = self._M_inv[:, l:l + m_i]
                    BM_i_tilde = self._B @ M_i_tilde
                    B_i = BM_i_tilde.reshape((self._n, m_i))
                    self._Bs += [B_i]
                    l += m_i
            else:
                self._Bs = [B]

        self._P_f = self.__get_are_P_f() if P_f is None else P_f
        # self._P_f = [np.eye(self._n)] * self._N if P_f is None else P_f
        self._L = L
        self._delta = self._T_f / self._L
        self.__show_legend = show_legend
        self.__state_variables_names = state_variables_names
        self._forward_time = np.linspace(start=0,
                                         stop=self._T_f,
                                         num=self._L)
        self._backward_time = self._forward_time[::-1]
        self._A_cl = np.empty_like(self._A)
        self._P = []
        self._K = []
        self.__epsilon_x = epsilon_x
        self.__epsilon_P = epsilon_P
        self._x = self._x_0
        self._x_non_linear = None
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

        ValueError
            If any of the class parameters don't meet the specified requirements
        """

        if self._N == 0:
            raise ValueError('At least one objective must be specified')
        if self._N > 1 and self._Ms and not all([M_i.shape[0] == R_ii.shape[0] == R_ii.shape[1]
                                                 if R_ii.ndim > 1 else M_i.shape[0] == R_ii.shape[0]
                                                 for M_i, R_ii in zip(self._Ms, self._Rs)]):
            raise ValueError('R must be a sequence of square 2-d numpy arrays with shape '
                             'corresponding to the second dimensions of the arrays in M')

        if not 1 > self.__epsilon_x > 0:
            raise ValueError('The state convergence tolerance must be in the open interval (0,1)')
        if not 1 > self.__epsilon_P > 0:
            raise ValueError('The convergence tolerance of the matrix P must be in the open interval (0,1)')

        if self._A.shape != (self._n, self._n):
            raise ValueError(f'A must be a square 2-d numpy array with shape nxn = {self._n}x{self._n}')
        if self._B is not None:
            if self._B.shape[0] != self._n:
                raise ValueError(f'B must be a 2-d numpy array with n = {self._n} rows')
            if not self.is_fully_controllable():
                warnings.warn("Warning: The given system is not fully controllable")
        else:
            if not all([B_i.shape[0] == self._n and B_i.shape[1] == R_ii.shape[0] == R_ii.shape[1]
                        if R_ii.ndim > 1 else B_i.shape[1] == R_ii.shape[0] for B_i, R_ii in zip(self._Bs, self._Rs)]):
                raise ValueError('R must be a sequence of square 2-d numpy arrays with shape '
                                 'corresponding to the second dimensions of the arrays in Bs')
        if not all([np.all(np.linalg.eigvals(R_ii) > 0) if R_ii.ndim > 1 else R_ii > 0
                    for R_ii in self._Rs]):
            raise ValueError('R must be a sequence of square 2-d positive definite numpy arrays')
        if not all([Q_i.shape == (self._n, self._n) for Q_i in self._Qs]):
            raise ValueError(f'Q must be a sequence of square 2-d positive semi-definite numpy arrays with shape nxn '
                             f'= {self._n}x{self._n}')
        if not all([all([e_i >= 0 or PyDiffGame._eigvals_tol > abs(e_i) >= 0 for e_i in np.linalg.eigvals(Q_i)])
                    for Q_i in self._Qs]):
            raise ValueError('Q must contain positive semi-definite numpy arrays')

        if self._x_0 is not None and self._x_0.shape != (self._n,):
            raise ValueError(f'x_0 must be a 1-d numpy array with length n = {self._n}')
        if self._x_T is not None and self._x_T.shape != (self._n,):
            raise ValueError(f'x_T must be a 1-d numpy array with length n= {self._n}')
        if self._T_f is not None and self._T_f <= 0:
            raise ValueError('T_f must be a positive real number')
        if not all([P_f_i.shape[0] == P_f_i.shape[1] == self._n for P_f_i in self._P_f]):
            raise ValueError(f'P_f must be a sequence of 2-d positive semi-definite numpy arrays with shape nxn = '
                             f'{self._n}x{self._n}')
        # if not all([eig >= 0 for eig_set in [np.linalg.eigvals(P_f_i) for P_f_i in self._P_f] for eig in eig_set]):
        #     warnings.warn("Warning: there is a matrix in P_f that has negative eigenvalues.
        #     Convergence may not occur")
        if self.__state_variables_names and len(self.__state_variables_names) != self._n:
            raise ValueError(f'The parameter state_variables_names must be of length n = {self._n}')

        if self._L <= 0:
            raise ValueError('The number of data points must be a positive integer')
        if self.__eta <= 0:
            raise ValueError('The number of last matrix norms to consider for convergence must be a positive integer')

    def is_fully_controllable(self) -> bool:
        """
        Tests full controllability of an LTI system by the equivalent condition of full rank
        of the controllability Gramian

        Returns
        -------

        fully_controllable : bool
            Whether the system is fully controllable
        """

        controllability_matrix = np.concatenate([self._B] +
                                                [np.linalg.matrix_power(self._A, i) @ self._B
                                                 for i in range(1, self._n)],
                                                axis=1)
        controllability_matrix_rank = np.linalg.matrix_rank(A=controllability_matrix,
                                                            tol=10e-5)
        fully_controllable = controllability_matrix_rank == self._n

        return fully_controllable

    def is_LQR(self) -> bool:
        """
        Indicates whether the game has only one objective and is just a regular LQR

        Returns
        -------

        is_lqr : bool
            Whether the system is LQR
        """

        return len(self) == 1

    def __print_eigenvalues(self):
        """
        Prints the eigenvalues of the matrices Q_i and R_ii for all 1 <= i <= N
        with the greatest common divisor of their elements d as a parameter
        """

        d = sympy.symbols('d')

        for matrix in self._Rs + self._Qs:
            curr_gcd = math.gcd(*matrix.ravel())
            sympy_matrix = sympy.Matrix((matrix / curr_gcd).astype(int)) * d
            eigenvalues_multiset = sympy_matrix.eigenvals(multiple=True)
            print(f"{repr(sympy_matrix)}: {eigenvalues_multiset}\n")

    def __print_characteristic_polynomials(self):
        """
        Prints the characteristic polynomials of the matrices Q_i and R_ii for all 1 <= i <= N
        as a function of L and with the greatest common divisor of all their elements d as a parameter
        """

        d = sympy.symbols('d')
        L = sympy.symbols('L')

        for matrix in self._Rs + self._Qs:
            curr_gcd = math.gcd(*matrix.ravel())
            sympy_matrix = sympy.Matrix((matrix / curr_gcd).astype(int)) * d
            characteristic_polynomial = sympy_matrix.charpoly(x=L).as_expr()
            factored_characteristic_polynomial_expression = sympy.factor(f=characteristic_polynomial)
            print(f"{repr(sympy_matrix)}: {factored_characteristic_polynomial_expression}\n")

    def _converge_DREs_to_AREs(self):
        """
        Solves the game as backwards convergence of the differential
        finite-horizon game for repeated consecutive steps until the matrix norm converges
        """

        x_converged = False
        P_converged = False
        last_eta_norms = []
        curr_iteration_T_f = self._T_f
        x_T = self._x_T if self._x_T is not None else np.zeros_like(self._x_0)
        x_convergence_iterations = 0

        with tqdm(total=self.__x_max_convergence_iterations) as x_progress_bar:
            while (not x_converged) and x_convergence_iterations < self.__x_max_convergence_iterations:
                x_convergence_iterations += 1
                P_convergence_iterations = 0

                while (not P_converged) and P_convergence_iterations < self.__p_max_convergence_iterations:
                    P_convergence_iterations += 1
                    self._backward_time = np.linspace(start=self._T_f,
                                                      stop=self._T_f - self._delta,
                                                      num=self._L)
                    self._update_Ps_from_last_state()
                    self._P_f = self._P[-1]
                    last_eta_norms += [np.linalg.norm(self._P_f)]

                    if len(last_eta_norms) > self.__eta:
                        last_eta_norms.pop(0)

                    if len(last_eta_norms) == self.__eta:
                        P_converged = all([abs(norm_i - norm_i1) < self.__epsilon_P for norm_i, norm_i1
                                           in zip(last_eta_norms, last_eta_norms[1:])])
                    self._T_f -= self._delta

                self._T_f = curr_iteration_T_f

                if self._x_0 is not None:
                    curr_x_T = self._simulate_curr_x_T()
                    x_T_difference = x_T - curr_x_T
                    x_T_difference_norm = x_T_difference.T @ x_T_difference
                    curr_x_T_norm = curr_x_T.T @ curr_x_T

                    if curr_x_T_norm > 0:
                        convergence_ratio = x_T_difference_norm / curr_x_T_norm
                        x_converged = convergence_ratio < self.__epsilon_x

                    curr_iteration_T_f += 1
                    self._T_f = curr_iteration_T_f
                    self._forward_time = np.linspace(start=0,
                                                     stop=self._T_f,
                                                     num=self._L)
                else:
                    x_converged = True

                x_progress_bar.update(1)

    def _post_convergence(f: Callable) -> Callable:
        """
        A decorator static-method to apply on methods that need only be called after convergence

        Parameters
        ----------

        f: Callable
            The method requiring convergence

        Returns
        ----------

        decorated_f: Callable
            A decorated version of the method requiring convergence
        """

        def decorated_f(self: PyDiffGame,
                        *args,
                        **kwargs):
            if not self._converged:
                raise RuntimeError('Must first simulate the differential game')
            return f(self,
                     *args,
                     **kwargs)

        return decorated_f

    @_post_convergence
    def __plot_temporal_variables(self,
                                  t: np.array,
                                  temporal_variables: np.array,
                                  is_P: bool,
                                  title: Optional[str] = None,
                                  output_variables_names: Optional[Sequence[str]] = None,
                                  two_state_spaces: Optional[bool] = False):
        """
        Displays plots for the state variables with respect to time and the convergence of the values of P

        Parameters
        ----------

        t: 1-d np.array array of len(L)
            The time axis information to plot
        temporal_variables: 2-d np.arrays of shape(L, n)
            The y-axis information to plot
        is_P: bool
            Indicates whether to accommodate plotting for all the values in P_i or just for x
        title: str, optional
            The plot title to display
        two_state_spaces: bool, optional
            Indicates whether to plot two state spaces
        """

        self._fig = plt.figure(dpi=150)
        self._fig.set_size_inches(8, 6)
        plt.plot(t[:temporal_variables.shape[0]], temporal_variables)
        plt.xlabel('Time $[s]$')

        if title:
            plt.title(title)

        if self.__show_legend:
            if is_P:
                labels = ['${P' + str(i) + '}_{' + str(j) + str(k) + '}$' for i in range(1, self._N + 1)
                          for j in range(1, self._n + 1) for k in range(1, self._n + 1)]
            elif output_variables_names is not None:
                labels = output_variables_names
            elif self.__state_variables_names is not None:
                if two_state_spaces:
                    labels = [f'${name}_{1}$' for name in self.__state_variables_names] + \
                             [f'${name}_{2}$' for name in self.__state_variables_names]
                else:
                    labels = [f'${name}$' for name in self.__state_variables_names]
            else:
                labels = ["${\mathbf{x}}_{" + str(j) + "}$" for j in range(1, self._n + 1)]

            plt.legend(labels=labels,
                       loc='upper left' if is_P else 'best',
                       ncol=2,
                       prop={'size': int(100 / self._n)},
                       # bbox_to_anchor=(1, 0.75)
                       )

        plt.grid()
        plt.show()

    def __get_temporal_state_variables_title(self,
                                             linear_system: Optional[bool] = True,
                                             two_state_spaces: Optional[bool] = False) -> str:
        """
        Returns the title of a temporal figure to plot

        Parameters
        ----------

        linear_system: bool, optional
            Whether the system is linear
        two_state_spaces: bool, optional
            Whether the plot is of two state spaces

        Returns
        -------

        title : str
            The title of the figure
        """

        if two_state_spaces:
            title = f"{self._N}-Player Game"
        else:
            title = f"{('' if linear_system else 'Nonlinear, ')}" \
                    f"{('Continuous' if self.__continuous else 'Discrete')}, " \
                    f"{('Infinite' if self.__infinite_horizon else 'Finite')} Horizon" \
                    f"{(f', {self._N}-Player Game' if self._N > 1 else ' LQR')}" \
                    f"{f', {self._L} Sampling Points'}"

        return title

    def plot_state_variables(self,
                             state_variables: np.array,
                             linear_system: Optional[bool] = True,
                             output_variables_names: Optional[Sequence[str]] = None, ):
        """
        Displays plots for the state variables with respect to time

        Parameters
        ----------

        state_variables: 2-d np.array of shape(L, n)
            The temporal state variables y-axis information to plot
        linear_system: bool, optional
            Indicates whether the system is linear or not
        output_variables_names: optional
            Names of the output variables
        """

        self.__plot_temporal_variables(t=self._forward_time,
                                       temporal_variables=state_variables,
                                       is_P=False,
                                       title=self.__get_temporal_state_variables_title(linear_system=linear_system),
                                       output_variables_names=output_variables_names)

    def __plot_x(self):
        """
        Plots the state vector variables wth respect to time
        """

        self.plot_state_variables(state_variables=self._x)

    def plot_y(self,
               C: np.array,
               output_variables_names: Optional[Sequence[str]] = None,
               save_figure: Optional[bool] = False,
               figure_path: Optional[str | Path] = _default_figures_filename,
               figure_filename: Optional[str] = _default_figures_filename):
        """
        Plots an output vector y = C x^T wth respect to time

        Parameters
        ----------

        C: 2-d np.array array of shape(p, n)
            The output coefficients with respect to state
        output_variables_names: Sequence of len(p)
            Names of output variables
        save_figure: bool
            Whether to save the figure or not
        figure_path: str or Path, optional, default = figures sub-folder of the current directory
            The desired saved figure's file path
        figure_filename: str, optional
            The desired saved figure's filename
        """

        y = C @ self._x.T
        self.plot_state_variables(state_variables=y.T,
                                  output_variables_names=output_variables_names)
        if save_figure:
            self.__save_figure(figure_path=figure_path,
                               figure_filename=figure_filename)

    @_post_convergence
    def __save_figure(self,
                      figure_path: Optional[str | Path] = _default_figures_filename,
                      figure_filename: Optional[str] = _default_figures_filename):
        """
        Saves the current figure

        Parameters
        ----------

        figure_path: str or Path, optional, default = figures sub-folder of the current directory
            The desired saved figure's file path
        figure_filename: str, optional
            The desired saved figure's filename
        """

        if self._x_0 is None:
            raise RuntimeError('No parameter x_0 was defined to illustrate a figure')

        if figure_filename == PyDiffGame._default_figures_filename:
            figure_filename += '0'

        if not figure_path.is_dir():
            os.mkdir(figure_path)

        figure_full_filename = Path(fr'{figure_path}/{figure_filename}.png')

        while figure_full_filename.is_file():
            curr_name = figure_full_filename.name
            curr_num = int(curr_name.split('.png')[0][-1])
            next_num = curr_num + 1
            figure_full_filename = Path(fr'{figure_path}/image{next_num}.png')

        self._fig.savefig(figure_full_filename)

    def __augment_two_state_space_vectors(self,
                                          other: PyDiffGame,
                                          non_linear: bool = False) -> (np.array, np.array):
        """
        Plots the state variables of two converged state spaces

        Parameters
        ----------

        other: PyDiffGame
            Other converged differential game to augment the current one with
        non_linear: bool, optional
            Indicates whether to augment the non-linear states or not

        Returns
        ----------

        longer_period: 1-d np.array of len(max(self.L, other.L))
            The longer time period of the two games
        augmented_state_space: 2-d np.array of shape(max(self.L, other.L), self.n + other.n)
            Augmented state space with interpolated and extrapolated values for both games
        """

        f_self = sp.interpolate.interp1d(x=self._forward_time,
                                         y=self._x if not non_linear else self._x_non_linear.T,
                                         axis=0,
                                         fill_value="extrapolate")

        f_other = sp.interpolate.interp1d(x=other.forward_time,
                                          y=other.x if not non_linear else other._x_non_linear.T,
                                          axis=0,
                                          fill_value="extrapolate")

        longer_period = self._forward_time if max(self._forward_time) > max(other.forward_time) \
            else other.forward_time

        self_state_space = f_self(longer_period)
        other_state_space = f_other(longer_period)

        augmented_state_space = np.concatenate((self_state_space, other_state_space), axis=1)

        return longer_period, augmented_state_space

    def plot_two_state_spaces(self,
                              other: PyDiffGame,
                              non_linear: bool = False):
        """
        Plots the state variables of two converged state spaces

        Parameters
        ----------

        other: PyDiffGame
            Other converged differential game to plot along with the current one
        non_linear: bool, optional
            Indicates whether to plot the non-linear states or not
        """

        longer_period, augmented_state_space = self.__augment_two_state_space_vectors(other=other,
                                                                                      non_linear=non_linear)

        self_title = self.__get_temporal_state_variables_title(two_state_spaces=True)
        other_title = other.__get_temporal_state_variables_title(two_state_spaces=True)

        self.__plot_temporal_variables(t=longer_period,
                                       temporal_variables=augmented_state_space,
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

        P_f: list of 2-d numpy arrays, of len(N), of shape(n, n), solution of scipy's solve_are
            Final condition for the Riccati equation matrix
        """

        are_solver = sp.linalg.solve_continuous_are if self.__continuous else sp.linalg.solve_discrete_are
        P_f = []

        for B_i, Q_i, R_ii in zip(self._Bs, self._Qs, self._Rs):
            try:
                P_f_i = are_solver(self._A, B_i, Q_i, R_ii)
            except np.linalg.LinAlgError:
                # rand = np.random.rand(*self._A.shape)
                # P_f_i = rand @ rand.T
                P_f_i = Q_i

            P_f += [P_f_i]

        return P_f

    @abstractmethod
    def _update_K_from_last_state(self,
                                  *args,
                                  **kwargs):
        """
        Updates the controllers K after forward propagation of the state through time
        """

        pass

    @abstractmethod
    def _get_K_i(self,
                 *args,
                 **kwargs) -> np.array:
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

        Returns
        ----------

        P_f_i: 2-d numpy array of shape(n, n)
            i'th element of the final condition matrices P_f
        """

        pass

    def _update_A_cl_from_last_state(self,
                                     k: Optional[int] = None):
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

        A_cl = self._A - sum([self._Bs[i] @ (self._get_K_i(i, k) if k is not None else self._get_K_i(i))
                              for i in range(self._N)])
        if k is not None:
            self._A_cl[k] = A_cl
        else:
            self._A_cl = A_cl

    @abstractmethod
    def _update_Ps_from_last_state(self,
                                   *args,
                                   **kwargs):
        """
        Updates the matrices {P_i}_{i=1}^N after forward propagation of the state through time
        """

        pass

    @abstractmethod
    def is_A_cl_stable(self,
                       *args,
                       **kwargs) -> bool:
        """
        Tests Lyapunov stability of the closed loop

        Returns
        ----------

        is_stable: bool
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

        self.__plot_temporal_variables(t=self._backward_time,
                                       temporal_variables=self._P,
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

    def __solve_game_simulate_state_space_and_plot(self,
                                                   plot_Mx: Optional[bool] = False,
                                                   M: Optional[np.array] = None,
                                                   output_variables_names: Optional[Sequence[str]] = None):
        """
        Propagates the game through time, solves for it and plots the state with respect to time
        """

        self.__solve_game_and_simulate_state_space()

        if self._x_0 is not None:
            self.__plot_x()

            if plot_Mx and self._n % self._m == 0:
                C = np.kron(a=np.eye(int(self._n / self._m)),
                            b=M)
                self.plot_y(C=C,
                            output_variables_names=output_variables_names)

    def _simulate_curr_x_T(self) -> np.array:
        """
        Evaluates the current value of the state space vector at the terminal time T_f

        Returns
        ----------

        x_T_f: 1-d numpy array of shape(n)
            Evaluated terminal state vector x(T_f)
        """

        pass

    @_post_convergence
    def __get_x_l_tilde(self,
                        x: np.array,
                        l: int) -> np.array:
        """
        Returns the state difference at the lth segment

        Parameters
        ----------

        x: 1-d numpy array of shape(n)
            The solved state vector to consider
        l: int
            The desired segment

        Returns
        ----------

        x_l_tilde: 1-d numpy array of shape(n)
            State difference at the lth segment
        """

        x_l = x[l]
        x_l_tilde = self.x_T - x_l

        return x_l_tilde

    @_post_convergence
    def __get_v_i_l_tilde(self,
                          x: np.array,
                          i: int,
                          l: int,
                          v_i_T: np.array) -> np.array:
        """
        Returns the ith virtual input difference at the lth segment

        Parameters
        ----------

        x: 1-d numpy array of shape(n)
            The solved state vector to consider
        i: int
            The desired input index
        l: int
            The desired segment
        v_i_T: 1-d numpy array of shape(m_i)
            The terminal value of the ith input

        Returns
        ----------

        v_i_l_tilde: 1-d numpy array of shape(m_i)
            ith virtual input difference at the lth segment
        """

        x_l = x[l]

        K_i_l = self._get_K_i(i) if self.__continuous else self._get_K_i(i, l)
        v_i_l = - K_i_l @ x_l
        v_i_l_tilde = v_i_T - v_i_l

        return v_i_l_tilde

    @_post_convergence
    def __get_u_l_tilde(self,
                        x: np.array,
                        l: int,
                        x_l_tilde: np.array) -> np.array:
        """
        Returns the actual input difference at the lth segment

        Parameters
        ----------

        x: 1-d numpy array of shape(n)
            The solved state vector to consider
        l: int
            The desired segment
        x_l_tilde: 1-d numpy array of shape(n)
            The state difference at the lth segment

        Returns
        ----------

        u_l_tilde: 1-d numpy array of shape(m)
            Actual input difference at the lth segment
        """

        if self.is_LQR():
            k_T = self._get_K_i(0) if self.__continuous else self._get_K_i(0, self._L - 1)
            u_l_tilde = - k_T @ x_l_tilde
        else:
            v_l_tilde = []

            for i in range(self._N):
                k_i_T = self._get_K_i(i) if self.__continuous else self._get_K_i(i, self._L - 1)
                v_i_T = - k_i_T @ self.x_T

                v_i_l_tilde = self.__get_v_i_l_tilde(x=x,
                                                     i=i,
                                                     l=l,
                                                     v_i_T=v_i_T)

                v_l_tilde += [v_i_l_tilde]

            v_l_tilde = np.array(v_l_tilde)
            u_l_tilde = self._M_inv @ v_l_tilde

        return u_l_tilde

    @_post_convergence
    def __get_x_and_u_l_tilde(self,
                              x: np.array,
                              l: int) -> (np.array, np.array):
        """
        Returns the state and actual input difference at the lth segment

        Parameters
        ----------

        x: 1-d numpy array of shape(n)
            The solved state vector to consider
        l: int
            The desired segment

        Returns
        ----------

        x_l_tilde: 1-d numpy array of shape(n)
            State difference at the lth segment
        u_l_tilde: 1-d numpy array of shape(m)
            Actual input difference at the lth segment
        """

        x_l_tilde = self.__get_x_l_tilde(x=x,
                                         l=l)
        u_l_tilde = self.__get_u_l_tilde(x=x,
                                         l=l,
                                         x_l_tilde=x_l_tilde)

        return x_l_tilde, u_l_tilde

    def __get_x(self,
                non_linear: bool) -> np.array:
        """
        Returns the desired state vector to consider

        Parameters
        ----------

        non_linear: bool
            Whether to return the non-linear state

        Returns
        ----------

        x: 1-d numpy array of shape(n)
            Desired state vector
        """

        x = self._x if not non_linear else self._x_non_linear.T
        assert len(x) == self._L

        return x

    @_post_convergence
    def get_cost(self,
                 lqr_objective: Objective,
                 non_linear: Optional[bool] = False,
                 x_only: Optional[bool] = False) -> float:
        """
        Calculates the cost functionals values using the formula:
        J = int_{t=0}^T_f J(t) dt =
            int_{t=0}^T_f [ x_tilde(t)^T Q x_tilde(t) + u_tilde^T(t) R u_tilde(t) ) ] dt

        and the corresponding Trapezoidal rule approximation:
        J ~ delta / 2 sum_{l=1}^{L} [ J(l) + J(l - 1) ]
          = delta * [ sum_{l=1}^{L-1} J(l) + 1/2 ( J(0) + J(L) ) ]

        Parameters
        ----------

        lqr_objective: LQRObjective
            LQR objective to get Q and R from
        non_linear: bool, optional
            Indicates whether to calculate the cost with respect to the nonlinear state
        x_only: bool, optional
            Indicates whether to calculate the cost only with respect to x(t)

        Returns
        ----------

        log_cost: float
            game cost
        """

        x = self.__get_x(non_linear=non_linear)

        cost = 0
        Q = lqr_objective.Q
        R = lqr_objective.R

        for l in range(0, self._L):
            x_l_tilde = self.__get_x_l_tilde(x=x,
                                             l=l)
            cost_l = float(x_l_tilde.T @ Q @ x_l_tilde)

            if not x_only:
                u_l_tilde = self.__get_u_l_tilde(x=x,
                                                 l=l,
                                                 x_l_tilde=x_l_tilde)
                cost_l += float(u_l_tilde.T @ R @ u_l_tilde)

            if l in [0, self._L - 1]:
                cost_l *= 0.5

            cost += cost_l

        return cost

    @_post_convergence
    def get_agnostic_costs(self,
                           non_linear: Optional[bool] = False) -> float:
        """
        Calculates the agnostic functional value using the formula:
        J_agnostic = int_{t=0}^T_f J_agnostic(t) dt =
                     int_{t=0}^T_f [ ||x_tilde(t)||^2 + ||u_tilde(t)||^2 ] dt =
                     int_{t=0}^T_f [ x_tilde(t)^T x_tilde(t) + u_tilde(t)^T u_tilde(t) ] dt

        and the corresponding Trapezoidal rule approximation:
        J ~ delta / 2 sum_{l=1}^L [ J(l) + J(l - 1) ] =
            delta * [ sum_{l=1}^{L-1} J(l) + 1/2 ( J(0) + J(L) ) ]

        Parameters
        ----------

        non_linear: bool, optional
            Indicates whether to calculate the cost with respect to the nonlinear state

        Returns
        ----------

        log_cost: float
            agnostic game cost
        """

        x = self.__get_x(non_linear=non_linear)

        cost = sum((0.5 if l in [0, self._L - 1] else 1) * sum(y_l.T @ y_l
                                                               for y_l in self.__get_x_and_u_l_tilde(x=x,
                                                                                                     l=l))
                   for l in range(0, self._L)) * self._delta

        return cost

    def __call__(self,
                 plot_state_space: Optional[bool] = True,
                 plot_Mx: Optional[bool] = False,
                 M: Optional[np.array] = None,
                 output_variables_names: Optional[Sequence[str]] = None,
                 save_figure: Optional[bool] = False,
                 figure_path: Optional[str | Path] = _default_figures_path,
                 figure_filename: Optional[str] = _default_figures_filename,
                 print_characteristic_polynomials: Optional[bool] = False,
                 print_eigenvalues: Optional[bool] = False):
        """
        Runs the game simulation

        Parameters
        ----------

        plot_state_space: bool, optional
            Whether to plot the solved game state space
        save_figure: bool, optional
            Whether to save the state space plot figure
        print_characteristic_polynomials: bool, optional
            Whether to print the characteristic polynomials of the matrices Q_i and R_ii for all 1 <= i <= N
        print_eigenvalues: bool, optional
            Whether to print the eigenvalues of the matrices Q_i and R_ii for all 1 <= i <= N

        """

        if print_characteristic_polynomials:
            self.__print_characteristic_polynomials()
        if print_eigenvalues:
            self.__print_eigenvalues()

        if plot_state_space:
            self.__solve_game_simulate_state_space_and_plot(plot_Mx=plot_Mx,
                                                            M=M,
                                                            output_variables_names=output_variables_names)
            if save_figure:
                self.__save_figure(figure_path=figure_path,
                                   figure_filename=figure_filename)
        else:
            self.__solve_game_and_simulate_state_space()

    def __getitem__(self,
                    i: int) -> Objective:
        """
        Returns the i'th objective of the game

        Parameters
        ----------

        i: int
            The objective index to consider

        Returns
        ----------

        o_i: Objective
            i'th objective of the game
        """

        return self.__objectives[i]

    def __len__(self) -> int:
        """
        We define the length of a differential game to be its number of objectives (N)

        Returns
        ----------

        len: int
            The length of the game
        """

        return self._N

    @property
    def P(self) -> list[np.array]:
        return self._P

    @property
    def K(self) -> list[np.array]:
        return self._K

    @property
    def x_0(self) -> np.array:
        return self._x_0

    @property
    def x_T(self) -> np.array:
        return self._x_T

    @property
    def x(self) -> np.array:
        return self._x

    @property
    def forward_time(self) -> np.array:
        return self._forward_time

    @property
    def T_f(self) -> float:
        return self._T_f

    @property
    def L(self) -> int:
        return self._L

    @property
    def Bs(self) -> Sequence:
        return self._Bs

    @property
    def M_inv(self) -> np.array:
        return self._M_inv

    @property
    def epsilon(self) -> float:
        return self.__epsilon_x

    _post_convergence = staticmethod(_post_convergence)
