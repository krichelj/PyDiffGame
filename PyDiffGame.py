import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def get_S_matrices(R, B, m):
    S_matrices = []

    for i in range(0, m):
        R_i = R[i]
        R_i_inv = inv(R_i)

        B_i = B[i]
        B_i_t = B[i].transpose()

        S_i = B_i @ R_i_inv @ B_i_t
        S_matrices.append(S_i)

    return S_matrices


def get_M_sum(P_matrices, S_matrices, m, n):
    M = np.zeros((n, n))

    for i in range(0, m):
        S_i = S_matrices[i]
        P_i = P_matrices[i]

        M = M + S_i @ P_i

    return M


def get_P_matrices(P, N, M):
    P_matrices = [np.array(P[M * i:M * (i + 1)]).reshape(n, n) for i in range(0, N)]

    return P_matrices


def get_parameters(P, A, B, R, m, N):
    A_t = A.transpose()

    M = sum(m)
    P_matrices = get_P_matrices(P, m, N, M)
    S_matrices = get_S_matrices(R, B, m)
    SP_sum = get_M_sum(P_matrices, S_matrices, m, n)

    return A_t, P_matrices, S_matrices, M


def get_P_f(m):
    M = sum(m)
    P_size = M ** 2
    N = len(m)

    P_f = np.zeros((len(m)*P_size,))

    for i in range(0, N):
        if i == 0:
            P_f = np.array([0] * P_size)
        else:
            P_f_i = i * 1000
            P_f = np.concatenate((P_f, [P_f_i] * P_size), axis=None)
    return P_f


def get_W_i(P_matrices, S_matrices, m, n, i):
    W_i = np.zeros((n, n))

    for j in range(0, m):
        if j != i:
            S_j = S_matrices[j]
            P_j = P_matrices[j]

            W_i = W_i + P_j @ S_j

    return W_i


def solve_riccati(P, A, B, Q, R, m, N, is_closed_loop):
    M = sum(m)
    A_t, P_matrices, S_matrices, SP_sum = get_parameters(P, A, B, R, m, N)
    dPdt = np.zeros((N * m,))

    for i in range(0, m):

        P_i = P_matrices[i]
        Q_i = Q[i]

        dP_idt = - A_t @ P_i - P_i @ A - Q_i + P_i @ M

        if is_closed_loop:
            W_i = get_W_i(P_matrices, S_matrices, m, n, i)
            dP_idt = dP_idt + W_i @ P_i

        dP_idt = np.array(dP_idt).reshape(N)

        if i == 0:
            dPdt = dP_idt
        else:
            dPdt = np.concatenate((dPdt, dP_idt), axis=None)

    return dPdt


def solve_m_coupled_riccati(P, _, A, B, Q, R, m, N):
    is_closed_loop = False
    dPdt = solve_riccati(P, A, B, Q, R, m, N, is_closed_loop)

    return dPdt


def solve_m_coupled_riccati_cl(P, _, A, B, Q, R, m, n, N):
    is_closed_loop = True
    dPdt = solve_riccati(P, A, B, Q, R, m, n, N, is_closed_loop)

    return dPdt


def plot(s, P_or_X, m, n, is_P):
    V = range(1, m + 1)
    U = range(1, n + 1)
    legend = tuple(['${P' + str(i) + '}_{' + str(j) + str(k) + '}$' for i in V for j in U for k in U] if is_P else
                   ['${X' + str(i) + '}_{' + str(j) + '}$' for i in V for j in U])

    plt.figure(dpi=130)
    plt.plot(s, P_or_X)
    plt.xlabel('Time')
    plt.legend(legend, loc='best', ncol=int(m / 2), prop={'size': int(20 / m)})
    plt.grid()
    plt.show()
