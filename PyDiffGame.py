import numpy as np
from numpy.linalg import eigvals, inv
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint, solve_ivp
import time


def get_P_f(m, B):
    M = sum(m)
    N = len(B)
    P_size = M ** 2
    P_f = np.zeros((N * P_size,))

    for i in range(N):
        random_matrix = np.random.rand(M, M)
        P_f_i = random_matrix.transpose() @ random_matrix
        P_f_i = P_f_i.reshape((P_size,))

        if i == 0:
            P_f = P_f_i
        else:
            P_f = np.concatenate((P_f, P_f_i), axis=0)
    return P_f


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


def check_input(m, A, B, Q, R, X0, T_f, P_f, data_points):
    M = sum(m)
    N = len(B)
    P_size = M ** 2

    if not all([isinstance(m_i, int) and m_i > 0 for m_i in m]):
        raise ValueError('The state variables dimensions must all be positive integers')
    if not A.shape == (M, M):
        raise ValueError('The system matrix A must be of size MxM')
    if not all([B_i.shape[0] == M for B_i in B]):
        raise ValueError('The matrices B_i must all have M rows')
    if not all([B_i.shape[1] == R_i.shape[0] == R_i.shape[1] for B_i, R_i in zip(B, R)]):
        raise ValueError('The matrices B_i, R_i must have corresponding dimensions')
    if not all([np.all(eigvals(Q_i) >= 0) for Q_i in Q]):
        raise ValueError('The weight matrices Q_i must all be positive semi-definite')
    if not all([np.all(eigvals(R_i) > 0) for R_i in R]):
        raise ValueError('The weight matrices R_i must all be positive definite')
    if not X0.shape == (M, ):
        raise ValueError('The initial state vector should be of dimension M')

    valid_T_f = isinstance(T_f, (float, int)) and T_f < 100
    valid_P_f = P_f is not None
    valid_data_points = isinstance(data_points, int) and data_points < 10000
    finite_horizon = [valid_T_f, valid_P_f, valid_data_points]

    if any(finite_horizon) and not all(finite_horizon):
        raise ValueError('For finite horizon - T_f, P_f and the number of data points must all be defined as required')
    if valid_T_f and T_f <= 0:
        raise ValueError('The horizon must be positive')
    if valid_P_f and not \
            all([eig >= 0 for eig_set in [eigvals(P_f[i * P_size:(i + 1) * P_size].reshape(M, M)) for i in range(N)]
                 for eig in eig_set]):
        raise ValueError('Final matrices P_f must all be positive semi-definite')
    if valid_data_points and data_points <= 0:
        raise ValueError('There has to be a positive number of data points')


def solve_N_coupled_riccati(_, P, M, N, A, B, Q, R, cl):
    P_size = M ** 2

    P_matrices = [(P[i * P_size:(i + 1) * P_size]).reshape(M, M) for i in range(N)]
    S_matrices = [B[i] @ inv(R[i]) @ B[i].transpose() for i in range(N)]
    SP_sum = sum(a @ b for a, b in zip(S_matrices, P_matrices))
    A_t = A.transpose()
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


def plot(m, s, mat, is_P):
    M = sum(m)
    V = range(1, len(m) + 1)
    U = range(1, int(M) + 1)

    legend = tuple(['${' + ('P' if is_P else 'X') + str(i) + '}_{' + str(j) + str(k) + '}$' for i in V for j in U for k in U])

    plt.figure(dpi=130)
    plt.plot(s, mat)
    plt.xlabel('Time')
    plt.legend(legend, loc='best', ncol=int(M / 2), prop={'size': int(20 / M)})
    plt.grid()
    plt.show()


def simulate_state_space(P, M, A, R, B, N, X0, t):
    j = len(P) - 1
    P_size = M ** 2

    S_matrices = [B[i] @ inv(R[i]) @ B[i].transpose() for i in range(N)]

    def stateDiffEqn(X, _):
        nonlocal j

        P_matrices_j = [(P[j][i * P_size:(i + 1) * P_size]).reshape(M, M) for i in range(N)]
        SP_sum = sum(a @ b for a, b in zip(S_matrices, P_matrices_j))
        A_cl = A - SP_sum
        dXdt = A_cl @ X

        if j > 0:
            j -= 1
        return dXdt

    X = odeint(stateDiffEqn, X0, t)

    return X


def solve_diff_game(m, A, B, Q, R, cl, X0, T_f=5, P_f=None, data_points=10000):
    check_input(m, A, B, Q, R, X0, T_f, P_f, data_points)
    M = sum(m)
    N = len(B)
    t = np.linspace(T_f, 0, data_points)

    if P_f is None:
        P_f = get_care_P_f(A, B, Q, R)
        sol = solve_ivp(fun=solve_N_coupled_riccati, t_span=[T_f, 0], y0=P_f, args=(M, N, A, B, Q, R, cl), t_eval=t)
        P = np.swapaxes(sol.y, 0, 1)
    else:
        P = odeint(func=solve_N_coupled_riccati, y0=P_f, t=t, args=(M, N, A, B, Q, R, cl), tfirst=True)

    plot(m, t, P, True)

    forward_t = t[::-1]
    X = simulate_state_space(P, M, A, R, B, N, X0, forward_t)
    plot(m, forward_t, X, False)


if __name__ == '__main__':
    # start = time.time()
    m = [2, 2]

    A = np.diag([2, 1, 1, 4])
    B = [np.diag([2, 1, 1, 2]),
         np.diag([1, 2, 2, 1])]
    Q = [np.diag([2, 1, 2, 2]),
         np.diag([1, 2, 3, 4])]
    R = [np.diag([100, 200, 100, 200]),
         np.diag([100, 300, 200, 400])]
    coupled_R = [np.diag([100, 200, 100, 200]),
                 np.diag([100, 300, 200, 400])]

    cl = True
    T_f = 3
    P_f = get_care_P_f(A, B, Q, R)
    data_points = 1000

    X0 = np.array([10, 20, 30, 100])

    solve_diff_game(m, A, B, Q, R, cl, X0, T_f, P_f, data_points)
    # end = time.time()

    # print(end - start)
