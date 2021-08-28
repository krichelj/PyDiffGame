import numpy as np
from numpy.linalg import eigvals, inv, norm
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint
import warnings
from typing import List, Optional, Union, Tuple
from scipy.optimize import fsolve
from abc import ABC


class PyDiffGame(ABC):
    """
    Differential game abstract base class

    Parameters
    ----------
    A: numpy 2-d array, of shape(n, n)
        The system dynamics matrix
    B: list of numpy 2-d arrays, of len(N), each matrix B_j of shape(n, k_j)
        System input matrices for each control objective
    Q: list of numpy 2-d arrays, of len(N), each matrix Q_j of shape(n, M)
        Cost function state weights for each control objective
    R: list of numpy 2-d arrays, of len(N), each matrix R_j of shape(k_j, k_j)
        Cost function input weights for each control objective
    T_f: positive float, optional, default = 5
        System dynamics horizon. Should be given in the case of finite horizon
    x_0: numpy 1-d array, of shape(n), optional
        Initial state vector
    P_f: list of numpy 2-d arrays, , of len(N), of shape(n, n), optional, default = solution of scipy's solve_continuous_are
        Final condition for the Riccati equation matrix. Should be given in the case of finite horizon
    data_points: positive int, optional, default = 1000
        Number of data points to evaluate the Riccati equation matrix. Should be given in the case of finite horizon
    show_legend: boolean
        Indicates whether to display a legend in the plots (True) or not (False)
    """

    def __init__(self, A: np.ndarray, B: List[np.ndarray], Q: List[np.ndarray], R: List[np.ndarray],
                 x_0: Optional[np.ndarray] = None, T_f: Optional[Union[float, int]] = None,
                 P_f: Optional[List[np.ndarray]] = None, data_points: int = 1000, show_legend: bool = True):

        # input parameters
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x_0 = x_0
        self.infinite_horizon = T_f is None
        self.T_f = 20 if self.infinite_horizon else T_f
        self.P_f = self.get_care_P_f() if P_f is None else P_f
        self.data_points = data_points
        self.show_legend = show_legend

        self.forward_time = np.linspace(0, self.T_f, self.data_points)
        self.backward_time = reversed(self.forward_time)
        self.n = A.shape[0]
        self.N = len(B)
        self.P_size = self.n ** 2

        self.check_input()

        # additional parameters

        self.A_t = A.T
        self.S_matrices = [self.B[i] @ inv(self.R[i]) @ self.B[i].T for i in range(self.N)]
        self.P_rows_num = self.N * self.n
        K_rows_num = sum([b.shape[1] for b in self.B])
        self.Y_row_num = self.P_rows_num + K_rows_num
        self.A_cl = self.A

    def check_input(self):
        """
        Input checking method

        Raises
        ------
        Case-specific errors
        """

        if self.A.shape != (self.n, self.n):
            raise ValueError('A must be a square 2-d numpy array with shape MxM')
        if any([B_i.shape[0] != self.n for B_i in self.B]):
            raise ValueError('B must be a list of 2-d numpy arrays with M rows')
        if not all([B_i.shape[1] == R_i.shape[0] == R_i.shape[1]
                    and np.all(eigvals(R_i) >= 0) for B_i, R_i in zip(self.B, self.R)]):
            raise ValueError('R must be a list of square 2-d positive definite numpy arrays with shape '
                             'corresponding to the second dimensions of the arrays in B')
        if any([Q_i.shape != (self.n, self.n) for Q_i in self.Q]):
            raise ValueError('Q must be a list of square 2-d positive semi-definite numpy arrays with shape MxM')
        if any([np.all(eigvals(Q_i) < 0) for Q_i in self.Q]):
            if all([np.all(eigvals(Q_i) >= -1e-15) for Q_i in self.Q]):
                warnings.warn("Warning: there is a matrix in Q that has negative (but really small) eigenvalues")
            else:
                raise ValueError('Q must contain positive semi-definite numpy arrays')
        if self.x_0 is not None and self.x_0.shape != (self.n,):
            raise ValueError('x_0 must be a 1-d numpy array with length M')
        if self.T_f and self.T_f <= 0:
            raise ValueError('T_f must be a positive real number')
        if P_f:
            if not all([P_i.shape[0] == P_i.shape[1] == self.n for P_i in self.P_f]):
                raise ValueError('P_f must be a list of 2-d positive semi-definite numpy arrays with shape MxM')
            if not all([eig >= 0 for eig_set in [eigvals(P_f_i) for P_f_i in self.P_f] for eig in eig_set]):
                if all([eig >= -1e-15 for eig_set in [eigvals(P_f_i) for P_f_i in self.P_f] for eig in eig_set]):
                    warnings.warn("Warning: there is a matrix in P_f that has negative (but really small) eigenvalues")
                else:
                    raise ValueError('P_f must contain positive semi-definite numpy arrays')
        if self.data_points <= 0:
            raise ValueError('The number of data points must be a positive integer')

    def get_care_P_f(self):
        P_f = np.zeros((self.N * self.P_size,))

        for i in range(self.N):
            P_f_i = solve_continuous_are(self.A, self.B[i], self.Q[i], self.R[i]).reshape((self.P_size,))

            if i == 0:
                P_f = P_f_i
            else:
                P_f = np.concatenate((P_f, P_f_i), axis=0)

        return P_f

    def solve_game(self, **args):
        pass

    def simulate_state_space(self, **args):
        pass

    def plot(self, t: np.ndarray, mat: Tuple[np.ndarray, dict], is_P: bool):
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
                 P_f: Optional[List[np.ndarray]] = None, data_points: int = 1000, show_legend: bool = True):

        super().__init__(A, B, Q, R, x_0, T_f, P_f, data_points, show_legend)
        self.cl = cl

    def solve_game(self, epsilon: float = 10e-8, delta_T: float = 5, delta_T_points: int = None) \
            -> np.ndarray:
        """
        Differential game main evolution method for the continuous case

        Parameters
        ----------
        epsilon: float
            The convergence criterion constant
        delta_T: float
            The interval length
        delta_T_points: int
            Number of calculation points

        Returns
        -------
        P : numpy 2-d array, of shape(n, n) or of shape(n, n)
            Solution to the continuous-time set of Riccati equations
        """
        P = np.array([])

        if not delta_T_points:
            delta_T_points = self.data_points

        if self.infinite_horizon:
            last_Ps = []
            convergence = False

            while not convergence:
                self.backward_time = np.linspace(self.T_f, self.T_f - delta_T, delta_T_points)
                P = odeint(func=ContinuousPyDiffGame.solve_N_coupled_diff_riccati,
                           y0=np.ravel(self.P_f),
                           t=self.backward_time,
                           args=(self.A,
                                 self.A_t,
                                 self.S_matrices,
                                 self.Q,
                                 self.cl),
                           tfirst=True)

                self.P_f = P[-1]

                P_matrix_norm = norm(self.P_f)
                last_Ps.append(P_matrix_norm)

                if len(last_Ps) > 3:
                    last_Ps.pop(0)

                if len(last_Ps) == 3:
                    if abs(last_Ps[2] - last_Ps[1]) < epsilon and abs(last_Ps[1] - last_Ps[0]) < epsilon:
                        convergence = True

                self.T_f -= delta_T

        else:
            P = odeint(func=ContinuousPyDiffGame.solve_N_coupled_diff_riccati,
                       y0=np.ravel(self.P_f),
                       t=self.backward_time,
                       args=(self.A,
                             self.A_t,
                             self.S_matrices,
                             self.Q,
                             self.cl),
                       tfirst=True)

            self.plot(t=self.backward_time,
                      mat=P,
                      is_P=True)

        if self.x_0 is not None:
            x = self.simulate_state_space(self.forward_time, P)

            self.plot(t=self.forward_time,
                      mat=x,
                      is_P=False)

        return P

    def get_dxdt(self, x: np.ndarray, P_matrices: List[np.ndarray]):
        self.A_cl = self.A - sum([a @ b for a, b in zip(self.S_matrices, P_matrices)])
        dxdt = self.A_cl @ x

        return dxdt

    @staticmethod
    def solve_N_coupled_diff_riccati(_, P: np.ndarray, A: np.ndarray, A_t: np.ndarray, S_matrices: List[np.ndarray],
                                     Q: List[np.ndarray], cl: bool):
        n = A.shape[0]
        N = len(Q)
        P_size = n ** 2
        P_matrices = [(P[i * P_size:(i + 1) * P_size]).reshape(n, n) for i in range(N)]
        SP_sum = sum(a @ b for a, b in zip(S_matrices, P_matrices))
        dPdt = np.zeros((P_size,))

        for i in range(N):
            P_i = P_matrices[i]
            Q_i = Q[i]

            dPidt = - A_t @ P_i - P_i @ A - Q_i + P_i @ SP_sum

            if cl:
                PS_sum = (SP_sum - S_matrices[i] @ P_matrices[i]).transpose()
                dPidt = dPidt + PS_sum @ P_i

            dPidt = dPidt.reshape(P_size)

            if i == 0:
                dPdt = dPidt
            else:
                dPdt = np.concatenate((dPdt, dPidt), axis=0)
        return np.ravel(dPdt)

    def simulate_state_space(self, t: np.ndarray, P: np.ndarray) -> np.array:
        j = len(P) - 1

        def state_diff_eqn(x: np.ndarray, _):
            nonlocal j

            P_matrices_j = [(P[j][i * self.P_size:(i + 1) * self.P_size]).reshape(self.n, self.n)
                            for i in range(self.N)]
            dxdt = self.get_dxdt(x, P_matrices_j)

            if j > 0:
                j -= 1

            return np.ravel(dxdt)

        x = odeint(state_diff_eqn,
                   self.x_0, t)

        return x


class DiscretePyDiffGame(PyDiffGame):
    """
    Discrete differential game base class


    Considers the system:
    x[k+1] = Ax[k] + sum_{j=1}^N B_j v_j[k]

    with the infinite-horizon cost functions:
    J_i = sum_{k=1}^infty [ x[k]^T Q_i x[k] + sum_{j=1}^N u_j[k]^T R_{ij} u_j[k] ]

    """
    def get_K_i(self, Y: np.array, i: int) -> np.array:
        K_i_offset = sum([self.B[j].shape[1] for j in range(i)]) if i else 0
        first_K_i_index = self.P_rows_num + K_i_offset
        second_K_i_index = first_K_i_index + self.B[i].shape[1]
        K_i = Y[first_K_i_index:second_K_i_index, :]

        return K_i

    def get_K(self, Y: np.array) -> List[np.array]:
        return [self.get_K_i(Y, i) for i in range(self.N)]

    def update_K_and_A_cl(self, Y: np.array):
        K = self.get_K(Y)
        self.A_cl = self.A - sum([self.B[i] @ K[i] for i in range(self.N)])

    def check_discrete_stability(self) -> bool:
        closed_loop_eigenvalues = eigvals(self.A_cl)
        closed_loop_eigenvalues_absolute_values = [abs(eig) for eig in closed_loop_eigenvalues]
        stability = all([eig_norm < 1 for eig_norm in closed_loop_eigenvalues_absolute_values])

        # print(f'A_k:\n{A_k}\nEigenvalues:\n{closed_loop_eigenvalues_absolute_values}')
        return stability

    def get_discrete_parameters(self, Y: np.array) -> Tuple[List[np.array], List[np.array], np.array]:

        Y = Y.reshape((self.Y_row_num, self.n))

        P = []
        K = []
        A_k = self.A

        for i in range(self.N):
            P_i = Y[i * self.n:(i + 1) * self.n, :]
            P += [P_i]
            K_i = self.get_K_i(Y, i)
            K += [K_i]

            A_k = A_k - self.B[i] @ K_i

        return P, K, A_k

    def evaluate_discrete_equations(self, P: List[np.array], K: List[np.array], A_k: np.array) -> np.array:
        equations_P = []
        equations_K = []

        for i in range(self.N):
            K_i = K[i]
            P_i = P[i]
            R_ii = self.R[i]
            Q_i = self.Q[i]
            B_i = self.B[i]
            B_i_T_P_i = B_i.T @ P_i

            coefficient = R_ii + B_i_T_P_i @ B_i
            inv_coefficient = inv(coefficient)
            coefficient = inv_coefficient @ B_i_T_P_i
            A_k_i = A_k + B_i @ K_i

            equation_K_i = K_i - coefficient @ A_k_i
            equations_K += [equation_K_i]

            equation_P_i = Q_i + K_i.T @ R_ii @ K_i - P_i + A_k.T @ P_i @ A_k
            equations_P += [equation_P_i]

        equations = np.concatenate((*equations_P, *equations_K), axis=0).ravel()

        return equations

    def simulate_state_space(self, Y: np.array):
        self.update_K_and_A_cl(Y)
        x_0 = self.x_0.reshape((self.n, 1))

        x_t = x_0
        x_T = np.array([x_t]).reshape(1, self.n)

        for _ in self.forward_time[:-1]:
            x_t_1 = self.A_cl @ x_t
            x_T = np.concatenate((x_T, x_t_1.reshape(1, self.n)), axis=0)
            x_t = x_t_1

        # print(f"On iteration {j}, the system is {'NOT ' if not is_Y_stable else ''}stable")
        self.plot(t=self.forward_time, mat=x_T, is_P=False)

    def generate_Y_0(self) -> np.array:
        random_values_multiplier = 1
        P_0 = [random_values_multiplier * np.random.rand(self.n, self.n) for _ in range(self.N)]
        K_0 = [random_values_multiplier * np.random.rand(b.shape[1], self.n) for b in self.B]
        Y_0 = np.concatenate((*P_0, *K_0), axis=0)

        return Y_0

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

        def get_equations(Y: np.array):
            nonlocal j

            j += 1
            P, K, A_k = self.get_discrete_parameters(Y.ravel())
            Y = self.evaluate_discrete_equations(P, K, A_k)
            is_Y_stable = self.check_discrete_stability()
            # print(f"On iteration {j}, the system is {'NOT ' if not is_Y_stable else ''}stable")

            return Y

        Y = None
        # is_stable = False
        is_Y_0_stable = False

        for _ in range(num_of_simulations):
            while not is_Y_0_stable:
                Y_0 = self.generate_Y_0()

                self.update_K_and_A_cl(Y_0)
                is_Y_0_stable = self.check_discrete_stability()

                if is_Y_0_stable:
                    print(f"Generated random initial guess which is stable")

                Y = fsolve(get_equations, Y_0)
                j = 0
                Y = np.array(Y).reshape((self.Y_row_num, self.n))

                # A_k = self.get_A_k(Y)
                # is_stable = self.check_discrete_stability(A_k)

            self.simulate_state_space(Y)
            is_Y_0_stable = False


if __name__ == '__main__':
    A = np.array([[-0.1, 0.11],
                  [0.1, 0.4]])
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
    x_0 = np.array([10, 20])
    T_f = 5
    P_f = [np.array([[1, 0],
                     [0, 200]]),
           np.array([[3, 0],
                     [0, 4]]),
           np.array([[500, 0],
                     [0, 600]])]
    data_points = 1000
    show_legend = True

    discrete_game = DiscretePyDiffGame(A=A,
                                       B=B,
                                       Q=Q,
                                       R=R,
                                       x_0=x_0,
                                       P_f=P_f,
                                       T_f=T_f,
                                       data_points=data_points,
                                       show_legend=show_legend)
    discrete_game.solve_game()

    # continuous_game = ContinuousPyDiffGame(A=A,
    #                                        B=B,
    #                                        Q=Q,
    #                                        R=R,
    #                                        cl=cl,
    #                                        x_0=x_0,
    #                                        P_f=P_f,
    #                                        T_f=T_f,
    #                                        data_points=data_points,
    #                                        show_legend=show_legend)
    # P = continuous_game.solve_game()

    # n = A.shape[0]
    # N = len(Q)

    # print([(P[-1][i * P_size:(i + 1) * P_size]).reshape(n, n) for i in range(N)])
