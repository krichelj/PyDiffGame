import numpy as np
from numpy.linalg import eigvals, inv
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint
import warnings


def get_care_P_f(A, B, Q, R):
    M = A.shape[0]
    N = len(B)
    P_size = M ** 2
    P_f = np.zeros((N * P_size,))

    for i in range(N):
        P_f_i = solve_continuous_are(A, B[i], Q[i], R[i]).reshape((P_size,))

        if i == 0:
            P_f = P_f_i
        else:
            P_f = np.concatenate((P_f, P_f_i), axis=0)

    return P_f


def check_input(A, B, Q, R, X0, T_f, P_f, data_points):
    M = A.shape[0]

    if not (isinstance(A, np.ndarray) and A.shape == (M, M)):
        raise ValueError('A must be a square 2-d numpy array with shape MxM')
    if not (isinstance(B, list) and all([isinstance(B_i, np.ndarray) and B_i.shape[0] == M for B_i in B])):
        raise ValueError('B must be a list of 2-d numpy arrays with M rows')
    if not (isinstance(R, list) and all([isinstance(R_i, np.ndarray) and B_i.shape[1] == R_i.shape[0] == R_i.shape[1]
                                         and np.all(eigvals(R_i) >= 0) for B_i, R_i in zip(B, R)])):
        raise ValueError('R must be a list of square 2-d positive definite numpy arrays with shape '
                         'corresponding to the second dimensions of the arrays in B')
    if not (isinstance(Q, list) and all([isinstance(Q_i, np.ndarray) and Q_i.shape == (M, M) for Q_i in Q])):
        raise ValueError('Q must be a list of square 2-d positive semi-definite numpy arrays with shape MxM')
    if not all([np.all(eigvals(Q_i) >= 0) for Q_i in Q]):
        if all([np.all(eigvals(Q_i) >= -1e-15) for Q_i in Q]):
            warnings.warn("Warning: there is a matrix in Q that has negative (but really small) eigenvalues")
        else:
            raise ValueError('Q must contain positive semi-definite numpy arrays')
    if X0 is not None and not (isinstance(X0, np.ndarray) and X0.shape == (M,)):
        raise ValueError('X0 must be a 1-d numpy array with length M')
    if T_f is not None and not (isinstance(T_f, (float, int)) and T_f > 0):
        raise ValueError('T_f must be a positive real number')
    if P_f is not None:
        if not (isinstance(P_f, list) and
                all([isinstance(P_i, np.ndarray) and P_i.shape[0] == P_i.shape[1] == M for P_i in P_f])):
            raise ValueError('P_f must be a list of 2-d positive semi-definite numpy arrays with shape MxM')
        if not all([eig >= 0 for eig_set in [eigvals(P_f_i) for P_f_i in P_f] for eig in eig_set]):
            if all([eig >= -1e-15 for eig_set in [eigvals(P_f_i) for P_f_i in P_f] for eig in eig_set]):
                warnings.warn("Warning: there is a matrix in P_f that has negative (but really small) eigenvalues")
            else:
                raise ValueError('P_f must contain positive semi-definite numpy arrays')
    if not (isinstance(data_points, int) and data_points > 0):
        raise ValueError('The number of data points must be a positive integer')


def solve_N_coupled_riccati(P, M, N, A, A_t, S_matrices, Q, cl):
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


def solve_N_coupled_diff_riccati(_, P, M, N, A, A_t, S_matrices, Q, cl):
    sol = solve_N_coupled_riccati(P, M, N, A, A_t, S_matrices, Q, cl)
    return sol


def plot(M, N, s, mat, is_P, show_legend=True):
    V = range(1, N + 1)
    U = range(1, M + 1)

    plt.figure(dpi=130)
    plt.plot(s, mat)
    plt.xlabel('Time')

    if show_legend:
        legend = tuple(['${P' + str(i) + '}_{' + str(j) + str(k) + '}$' for i in V for j in U for k in U] if is_P else
                       ['${X' + (str(j) if M > 1 else '') + '}_{' + str(j) + '}$' for j in U])
        plt.legend(legend, loc='upper left' if is_P else 'best', ncol=int(M / 2), prop={'size': int(20 / M)})

    plt.grid()
    plt.show()


def simulate_state_space(P, M, A, S_matrices, N, X0, t, finite_horizon):
    j = len(P) - 1
    P_size = M ** 2

    def get_dXdt(X, P_matrices):
        SP_sum = sum(a @ b for a, b in zip(S_matrices, P_matrices))
        A_cl = A - SP_sum
        dXdt = A_cl @ X

        return dXdt

    def finite_horizon_state_diff_eqn(X, _):
        nonlocal j

        P_matrices_j = [(P[j][i * P_size:(i + 1) * P_size]).reshape(M, M) for i in range(N)]
        dXdt = get_dXdt(X, P_matrices_j)

        if j > 0:
            j -= 1
        return np.ravel(dXdt)

    def infinite_horizon_state_diff_eqn(X, _):
        P_matrices = [(P[i * P_size:(i + 1) * P_size]).reshape(M, M) for i in range(N)]
        dXdt = get_dXdt(X, P_matrices)

        return np.ravel(dXdt)

    X = odeint(finite_horizon_state_diff_eqn if finite_horizon else infinite_horizon_state_diff_eqn, X0, t)

    return X


def solve_diff_game(A, B, Q, R, cl, X0=None, T_f=None, P_f=None, data_points=10000, show_legend=True):
    check_input(A, B, Q, R, X0, T_f, P_f, data_points)
    M = A.shape[0]
    N = len(B)

    infinite_horizon = T_f is None
    current_T_f = 5 if infinite_horizon else T_f
    t = np.linspace(current_T_f, 0, data_points)
    A_t = A.transpose()
    S_matrices = [B[i] @ inv(R[i]) @ B[i].transpose() for i in range(N)]

    if P_f is None:
        P_f = get_care_P_f(A, B, Q, R)

    P = odeint(func=solve_N_coupled_diff_riccati, y0=np.ravel(P_f), t=t,
               args=(M, N, A, A_t, S_matrices, Q, cl), tfirst=True, rtol=1e-4 if infinite_horizon else 1.49012e-8)

    if T_f:
        plot(M, N, t, P, True, show_legend)

    if X0 is not None:
        forward_t = t[::-1]
        X = simulate_state_space(P, M, A, S_matrices, N, X0, forward_t, current_T_f)
        plot(M, N, forward_t, X, False)

    output = P[-1] if infinite_horizon else P
    return output


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

    P = solve_diff_game(A=A, B=B, Q=Q, R=R, cl=cl, T_f=T_f, data_points=data_points, show_legend=show_legend)
    print(P[-1])
    P = solve_diff_game(A=A, B=B, Q=Q, R=R, cl=cl, show_legend=show_legend)
    print(P)
