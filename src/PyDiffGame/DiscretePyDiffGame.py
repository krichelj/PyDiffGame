import numpy as np
from scipy.optimize import fsolve
import quadpy
from numpy.linalg import eigvals, inv
from typing import Sequence, Optional

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.Objective import Objective


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
                 B: Sequence[np.array] | np.array,
                 objectives: Sequence[Objective],
                 is_input_discrete: Optional[bool] = False,
                 x_0: Optional[np.array] = None,
                 x_T: Optional[np.array] = None,
                 T_f: Optional[float] = None,
                 P_f: Optional[Sequence[np.array] | np.array] = None,
                 show_legend: Optional[bool] = True,
                 state_variables_names: Optional[Sequence] = None,
                 epsilon: Optional[float] = PyDiffGame._epsilon_default,
                 L: Optional[int] = PyDiffGame._L_default,
                 eta: Optional[int] = PyDiffGame._eta_default,
                 force_finite_horizon: Optional[bool] = False,
                 debug: Optional[bool] = False):

        super().__init__(A=A,
                         B=B,
                         objectives=objectives,
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         P_f=P_f,
                         show_legend=show_legend,
                         state_variables_names=state_variables_names,
                         epsilon=epsilon,
                         L=L,
                         eta=eta,
                         force_finite_horizon=force_finite_horizon,
                         debug=debug
                         )

        if not is_input_discrete:
            self.__discretize_game()

        self.__K_rows_num = sum([B_i.shape[1] for B_i in self._Bs])

        def __solve_for_K_k(K_k_previous: np.array,
                            k_1: int) -> np.array:
            """
            fsolve Controllers Solver function

            Parameters
            ----------
            K_k_previous: numpy array
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
                i_indices = self.__get_indices_tuple(i)
                K_k_previous_i = K_k_previous[i_indices]
                P_k_1_i = P_k_1[i]
                R_ii = self._R[i]
                B_i = self._Bs[i]
                B_i_T = B_i.T

                A_cl_k_i = self._A

                for j in range(self._N):
                    if j != i:
                        B_j = self._Bs[j]
                        j_indices = self.__get_indices_tuple(j)
                        K_k_previous_j = K_k_previous[j_indices]
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

        A_tilda = np.exp(self._A * self._delta)
        e_AT = quadpy.quad(f=lambda T: np.array([np.exp(t * self._A) for t in T]).swapaxes(0, 2).swapaxes(0, 1),
                           a=0,
                           b=self._delta)[0]

        self._A = A_tilda
        B_tilda = e_AT
        self._Bs = [B_tilda @ B_i for B_i in self._Bs]
        self._Q = [Q_i * self._delta for Q_i in self._Q]
        self._R = [R_i / self._delta for R_i in self._Rs]

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
        self._Q = np.zeros((self._L,
                            self._N,
                            self._n,
                            self._n))
        self._Q[0] = 1 / 3 * Q
        self._Q[1::2] = 4 / 3 * Q
        self._Q[::2] = 2 / 3 * Q

    def __initialize_finite_horizon(self):
        """
        Initializes the calculated parameters for the finite horizon case by the following steps:
            - The matrices P_i and controllers K_i are initialized randomly for all sample points
            - The closed-loop dynamics matrix A_cl is first set to zero for all sample points
            - The terminal conditions P_f_i are set to Q_i
            - The resulting closed-loop dynamics matrix A_cl for the last sampling point is updated
            - The state is initialized with its initial value, if given
         """

        self._P = np.random.rand(self._L,
                                 self._N,
                                 self._n,
                                 self._n)
        self._K = np.random.rand(self._L,
                                 self._N,
                                 self.__K_rows_num,
                                 self._n)
        self._A_cl = np.zeros((self._L,
                               self._n,
                               self._n))

        self._P[-1] = np.array(self._Q).reshape((self._N,
                                                 self._n,
                                                 self._n))
        self._update_A_cl_from_last_state(k=-1)
        self._x = np.zeros((self._L,
                            self._n))
        self._x[0] = self._x_0

    def __get_K_i_shape(self,
                        i: int) -> range:
        """
        Returns the i'th controller shape indices

        Parameters
        ----------
        i: int
            The desired controller index
        """

        K_i_offset = sum([self._Bs[j].shape[1] for j in range(i)]) if i else 0
        first_K_i_index = K_i_offset
        second_K_i_index = first_K_i_index + self._Bs[i].shape[1]

        return range(first_K_i_index, second_K_i_index)

    def __get_indices_tuple(self, i: int):
        return i, self.__get_K_i_shape(i)

    def _update_K_from_last_state(self,
                                  k_1: int):
        """
        After initializing, in order to solve for the controllers K_i and matrices P_i,
        we start by solving for the controllers by the update rule:
        K_i[k] = [ R_ii + B_i^T P_i[k+1] B_i ] ^ {-1} B_i^T P_i[k+1] [ A - sum_{j=1, j!=i}^N B_j K_j[k] ]

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

        k = k_1 - 1

        if self._debug:
            f_K_k = self.f_solve_func(K_k_previous=K_k, k_1=k)
            close_to_zero_array = np.isclose(f_K_k, np.zeros_like(K_k))
            is_close_to_zero = all(close_to_zero_array)
            print(f"The current solution is {'NOT ' if not is_close_to_zero else ''}close to zero"
                  f"\nK_k:\n{K_k_reshaped}\nf_K_k:\n{f_K_k}")

        self._K[k] = K_k_reshaped

    def _get_K_i(self,
                 i: int,
                 k: int) -> np.array:
        return self._K[k][self.__get_indices_tuple(i)]

    def _get_P_f_i(self,
                   i: int) -> np.array:
        return self._P_f[i]

    def _update_Ps_from_last_state(self,
                                   k_1: int):
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
        P_k = np.zeros_like(P_k_1)

        for i in range(self._N):
            i_indices = self.__get_indices_tuple(i)
            K_k_i = K_k[i_indices]
            K_k_i_T = K_k_i.T
            P_k_1_i = P_k_1[i]
            R_ii = self._R[i]
            Q_i = self._Q[i]

            P_k[i] = A_cl_k.T @ P_k_1_i @ A_cl_k + (K_k_i_T @ R_ii if R_ii.ndim > 1 else K_k_i_T * R_ii) @ K_k_i + Q_i

        self._P[k] = P_k

    def is_A_cl_stable(self,
                       k: int) -> bool:
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
            k_1 = self._L - 1 - backwards_index
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

            self._update_Ps_from_last_state(k_1=k_1)

    @PyDiffGame._post_convergence
    def _solve_state_space(self):
        """
        Propagates the game through time and solves for it by solving the discrete difference equation:
        x[k+1] = A_cl[k] x[k]
        In each step, the controllers K_i are calculated with the values of P_i evaluated beforehand.
        """

        x_1_k = self._x_0

        for k in range(1, self._L - 1):
            A_cl_k = self._A_cl[k]
            x_k = A_cl_k @ x_1_k
            self._x[k] = x_k
            x_1_k = x_k
