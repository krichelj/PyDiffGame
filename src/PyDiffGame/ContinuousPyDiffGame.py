import numpy as np
from scipy.integrate import odeint
from numpy.linalg import eigvals, inv
from typing import Union

from PyDiffGame.PyDiffGame import PyDiffGame


class ContinuousPyDiffGame(PyDiffGame):
    """
    Continuous differential game base class


    Considers the system:
    dx(t)/dt = A x(t) + sum_{j=1}^N B_j v_j(t)
    """

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
                 epsilon: float = PyDiffGame._epsilon_default,
                 L: int = PyDiffGame._L_default,
                 eta: int = PyDiffGame._eta_default,
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
                         show_legend=show_legend,
                         epsilon=epsilon,
                         L=L,
                         eta=eta,
                         force_finite_horizon=force_finite_horizon,
                         debug=debug)

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
            # SP_sum = sum([S_i @ P_t_i for S_i, P_t_i in zip(S_matrices, P_t)])
            dP_tdt = np.zeros((self._P_size,))
            A_cl_t = self._A - sum([(self._B[j] @ inv(self._R[j])
                                     if self._R[j].ndim > 1 else self._B[j] / self._R[j]) @ self._B[j].T @ P_t[j]
                                    for j in range(self._N)])

            for i in range(self._N):
                P_t_i = P_t[i]
                B_i = self._B[i]
                R_ii = self._R[i]
                Q_i = self._Q[i]

                dP_t_idt = - A_cl_t.T @ P_t_i - P_t_i @ A_cl_t - Q_i - P_t_i @ \
                           (B_i @ inv(R_ii) if R_ii.ndim > 1 else B_i / R_ii) @ B_i.T @ P_t_i

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

    def _get_P_f_i(self, i: int) -> np.array:
        if len(self._P_f) == self._N:
            return self._P_f[i]

        return self._P_f[i * self._P_size:(i + 1) * self._P_size].reshape(self._n, self._n)

    def _update_Ps_from_last_state(self):
        """
        With P_f as the terminal condition, evaluates the matrices P_i by backwards-solving this set of
        coupled time-continuous Riccati differential equations:

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

        self._update_Ps_from_last_state()
        self._converged = True

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
        # print(f'A_cl = \n{self._A_cl}\neig=\n{closed_loop_eigenvalues_real_values}')
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
            Scipy's odeint State Variables Solver function

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

            t += 1

            return dx_t_dt

        self._x = odeint(func=state_diff_eqn,
                         y0=self._x_0,
                         t=self._forward_time)

    def simulate_x_T_f(self) -> np.array:

        def state_diff_eqn(x_t: np.array, _: float) -> np.array:
            self._update_K_from_last_state(t=-1)
            self._update_A_cl_from_last_state()
            dx_t_dt = self._A_cl @ ((x_t - self._x_T) if self._x_T is not None else x_t)
            dx_t_dt = dx_t_dt.ravel()

            return dx_t_dt

        simulated_x_T = odeint(func=state_diff_eqn,
                               y0=self._x_0,
                               t=self._forward_time)[-1]

        return simulated_x_T

