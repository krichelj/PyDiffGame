import numpy as np
from scipy.optimize import fsolve
import quadpy
from numpy.linalg import eigvals, inv
from typing import Union

from PyDiffGame import PyDiffGame


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
        The input to the discrete game can either be given in continuous or discrete form.
        If it is not in discrete form, the model gets discretized using this method in the following manner:
        A = exp(delta_T A)
        B_i = int_{t=0}^delta_T exp(tA) dt B_i

        The performance index gets discretized in the following manner:
        Q_i = Q_i * delta_T
        R_i = Q_i / delta_T
        """

        A_tilda = np.exp(self._A * self._delta_T)
        e_AT = quadpy.quad(f=lambda T: np.array([np.exp(t * self._A) for t in T]).swapaxes(0, 2).swapaxes(0, 1),
                           a=0,
                           b=self._delta_T)[0]

        self._A = A_tilda
        B_tilda = e_AT
        self._B = [B_tilda @ B_i for B_i in self._B]
        self._Q = [Q_i * self._delta_T for Q_i in self._Q]
        self._R = [R_i / self._delta_T for R_i in self._R]

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

        Q = np.array(self._Q)
        self._Q = np.zeros((self._data_points,
                            self._N,
                            self._n,
                            self._n))
        self._Q[0] = Q
        self._Q[1::2] = Q
        self._Q[::2] = Q

        # self.__simpson_integrate_Q()
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

            self._update_A_cl_from_last_state(k=k)

            if self._debug:
                print(f'A_cl_k:\n{self._A_cl[k]}')
                stable_k = self.is_A_cl_stable(k)
                print(f"The system is {'NOT ' if not stable_k else ''}stable")

            self._update_P_from_last_state(k_1=k)

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