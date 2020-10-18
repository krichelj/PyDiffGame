import numpy as np
from numpy.linalg import eigvals, inv, norm
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint
import warnings
from typing import List


class PyDiffGame:

    @staticmethod
    def check_input(A, B, Q, R, X0, T_f, P_f, data_points):
        M = A.shape[0]

        if A.shape != (M, M):
            raise ValueError('A must be a square 2-d numpy array with shape MxM')
        if any([B_i.shape[0] != M for B_i in B]):
            raise ValueError('B must be a list of 2-d numpy arrays with M rows')
        if not all([B_i.shape[1] == R_i.shape[0] == R_i.shape[1]
                    and np.all(eigvals(R_i) >= 0) for B_i, R_i in zip(B, R)]):
            raise ValueError('R must be a list of square 2-d positive definite numpy arrays with shape '
                             'corresponding to the second dimensions of the arrays in B')
        if any([Q_i.shape != (M, M) for Q_i in Q]):
            raise ValueError('Q must be a list of square 2-d positive semi-definite numpy arrays with shape MxM')
        if any([np.all(eigvals(Q_i) < 0) for Q_i in Q]):
            if all([np.all(eigvals(Q_i) >= -1e-15) for Q_i in Q]):
                warnings.warn("Warning: there is a matrix in Q that has negative (but really small) eigenvalues")
            else:
                raise ValueError('Q must contain positive semi-definite numpy arrays')
        if X0 is not None and X0.shape != (M,):
            raise ValueError('X0 must be a 1-d numpy array with length M')
        if T_f is not None and T_f <= 0:
            raise ValueError('T_f must be a positive real number')
        if P_f is not None:
            if not all([P_i.shape[0] == P_i.shape[1] == M for P_i in P_f]):
                raise ValueError('P_f must be a list of 2-d positive semi-definite numpy arrays with shape MxM')
            if not all([eig >= 0 for eig_set in [eigvals(P_f_i) for P_f_i in P_f] for eig in eig_set]):
                if all([eig >= -1e-15 for eig_set in [eigvals(P_f_i) for P_f_i in P_f] for eig in eig_set]):
                    warnings.warn("Warning: there is a matrix in P_f that has negative (but really small) eigenvalues")
                else:
                    raise ValueError('P_f must contain positive semi-definite numpy arrays')
        if data_points <= 0:
            raise ValueError('The number of data points must be a positive integer')

    def __init__(self, A: np.ndarray, B: List[np.ndarray], Q: List[np.ndarray], R: List[np.ndarray],
                 cl: bool, X0: np.ndarray = None, T_f: float or int = None, P_f: List[np.ndarray] = None,
                 data_points: int = 10000, show_legend: bool = True):
        PyDiffGame.check_input(A, B, Q, R, X0, T_f, P_f, data_points)

        self.__A = A
        self.__B = B
        self.__M = A.shape[0]
        self.__N = len(B)
        self.__P_size = self.__M ** 2
        self.__Q = Q
        self.__R = R
        self.__cl = cl
        self.__X0 = X0
        self.__P_f = self.get_care_P_f() if P_f is None else P_f
        self.__show_legend = show_legend
        self.__infinite_horizon = T_f is None
        self.__current_T_f = 20 if self.__infinite_horizon else T_f
        self.__t = np.linspace(self.__current_T_f, 0, data_points)
        self.__A_t = A.transpose()
        self.__S_matrices = [B[i] @ inv(R[i]) @ B[i].transpose() for i in range(self.__N)]

    def play_the_game(self):
        P = []

        if self.__infinite_horizon:
            a_level = 10e-8
            delta_T = 5
            delta_T_points = 10

            last_Ps = []
            convergence = False

            while not convergence:
                self.__t = np.linspace(self.__current_T_f, self.__current_T_f - delta_T, delta_T_points)
                P = odeint(func=PyDiffGame.solve_N_coupled_diff_riccati, y0=np.ravel(self.__P_f), t=self.__t,
                           args=(self.__M, self.__N, self.__A, self.__A_t, self.__S_matrices, self.__Q, self.__cl),
                           tfirst=True)
                self.__P_f = P[-1]

                P_matrix_norm = norm(self.__P_f)
                last_Ps.append(P_matrix_norm)

                if len(last_Ps) > 3:
                    last_Ps.pop(0)

                if len(last_Ps) == 3:
                    if abs(last_Ps[2] - last_Ps[1]) < a_level and abs(last_Ps[1] - last_Ps[0]) < a_level:
                        convergence = True

                self.__current_T_f -= delta_T

        else:
            P = odeint(func=PyDiffGame.solve_N_coupled_diff_riccati, y0=np.ravel(self.__P_f), t=self.__t,
                       args=(self.__M, self.__N, self.__A, self.__A_t, self.__S_matrices, self.__Q, self.__cl),
                       tfirst=True)
            self.plot(self.__t, P, True)

        if self.__X0 is not None:
            forward_t = self.__t[::-1]
            X = self.simulate_state_space(forward_t, P)
            self.plot(forward_t, X, False)

        return P

    def get_care_P_f(self):
        P_f = np.zeros((self.__N * self.__P_size,))

        for i in range(self.__N):
            P_f_i = solve_continuous_are(self.__A, self.__B[i], self.__Q[i], self.__R[i]).reshape((self.__P_size,))

            if i == 0:
                P_f = P_f_i
            else:
                P_f = np.concatenate((P_f, P_f_i), axis=0)

        return P_f

    @staticmethod
    def solve_N_coupled_diff_riccati(_, P, M, N, A, A_t, S_matrices, Q, cl):
        P_size = M ** 2
        P_matrices = [(P[i * P_size:(i + 1) * P_size]).reshape(M, M) for i in range(N)]
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

    def plot(self, t, mat, is_P):
        V = range(1, self.__N + 1)
        U = range(1, self.__M + 1)

        plt.figure(dpi=130)
        plt.plot(t, mat)
        plt.xlabel('Time')

        if self.__show_legend:
            legend = tuple(
                ['${P' + str(i) + '}_{' + str(j) + str(k) + '}$' for i in V for j in U for k in U] if is_P else
                ['${X' + (str(j) if self.__M > 1 else '') + '}_{' + str(j) + '}$' for j in U])
            plt.legend(legend, loc='upper left' if is_P else 'best', ncol=int(self.__M / 2),
                       prop={'size': int(20 / self.__M)})

        plt.grid()
        plt.show()

    def simulate_state_space(self, t, P):
        j = len(P) - 1

        def get_dXdt(X, P_matrices):
            SP_sum = sum(a @ b for a, b in zip(self.__S_matrices, P_matrices))
            A_cl = self.__A - SP_sum
            dXdt = A_cl @ X

            return dXdt

        def finite_horizon_state_diff_eqn(X, _):
            nonlocal j

            P_matrices_j = [(P[j][i * self.__P_size:(i + 1) * self.__P_size]).reshape(self.__M, self.__M)
                            for i in range(self.__N)]
            dXdt = get_dXdt(X, P_matrices_j)

            if j > 0:
                j -= 1
            return np.ravel(dXdt)

        def infinite_horizon_state_diff_eqn(X, _):
            P_matrices = [(P[i * self.__P_size:(i + 1) * self.__P_size]).reshape(self.__M, self.__M)
                          for i in range(self.__N)]
            dXdt = get_dXdt(X, P_matrices)

            return np.ravel(dXdt)

        X = odeint(infinite_horizon_state_diff_eqn if self.__infinite_horizon else finite_horizon_state_diff_eqn,
                   self.__X0, t)

        return X


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

    cl = False
    X0 = np.array([10, 20])
    T_f = 5
    P_f = [np.array([[10, 0],
                     [0, 20]]),
           np.array([[30, 0],
                     [0, 40]]),
           np.array([[50, 0],
                     [0, 60]])]
    data_points = 1000
    show_legend = True

    P = PyDiffGame(A=A, B=B, Q=Q, R=R, cl=cl, X0=X0, P_f =P_f, T_f=T_f, data_points=data_points,
                   show_legend=show_legend).play_the_game()
    print(P[-1])
    P = PyDiffGame(A=A, B=B, Q=Q, R=R, cl=cl, show_legend=show_legend).play_the_game()
    print(P[-1])
