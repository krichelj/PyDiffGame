import numpy as np
from scipy.integrate import odeint
from numpy.linalg import inv, eigvals
import matplotlib.pyplot as plt

from PyDiffGame import get_P_f


def get_SP_sum(S_matrices, P_matrices, M, N):
    SP_sum = np.zeros((M,))

    for j in range(N):
        SP_sum = SP_sum + S_matrices[j] @ P_matrices[j]

    return SP_sum


def get_PS_sum(S_matrices, P_matrices, M, N, i):
    PS_sum = np.zeros((M,))

    for j in range(N):
        if j != i:
            PS_sum = PS_sum + P_matrices[j] @ S_matrices[j]

    return PS_sum


def solve_N_coupled_riccati_cl(P, _, M, A, B, Q, R, N, cl):
    P_size = M ** 2

    P_matrices = [(P[i * P_size:(i + 1) * P_size]).reshape(M, M) for i in range(N)]
    S_matrices = [B[i] @ inv(R[i]) @ B[i].transpose() for i in range(N)]
    SP_sum = get_SP_sum(S_matrices, P_matrices, M, N)
    A_t = A.transpose()
    dPdt = np.zeros((M, M))

    for i in range(N):
        P_i = P_matrices[i]
        Q_i = Q[i]

        dPidt = - A_t @ P_i - P_i @ A - Q_i + P_i @ SP_sum

        if cl:
            PS_sum = get_PS_sum(S_matrices, P_matrices, M, N, i)
            dPidt = dPidt + PS_sum @ P_i

        dPidt = dPidt.reshape(P_size)

        if i == 0:
            dPdt = dPidt
        else:
            dPdt = np.concatenate((dPdt, dPidt), axis=None)

    return dPdt


def plot(s, P, M, N):
    V = range(1, N + 1)
    U = range(1, M + 1)

    legend = tuple(['${P' + str(i) + '}_{' + str(j) + str(k) + '}$' for i in V for j in U for k in U])

    plt.figure(dpi=130)
    plt.plot(s, P)
    plt.xlabel('Time')
    plt.legend(legend, loc='best', ncol=int(M / 2), prop={'size': int(20 / M)})
    plt.grid()
    plt.show()


def check_input(m, Q, R, T_f, P_f, N):
    M = sum(m)
    P_size = M ** 2

    if not all([m_i > 0 for m_i in m]):
        raise ValueError('The state variables dimensions must all be positive')
    if T_f <= 0:
        raise ValueError('The horizon must be positive')
    if not all([np.all(eigvals(Q_i) >= 0) for Q_i in Q]):
        raise ValueError('The weight matrices Q_i must all be positive semi-definite')
    if not all([eig >= 0 for eig_set in [eigvals(P_f[i*P_size:(i+1)*P_size].reshape(M, M)) for i in range(N)]
                for eig in eig_set]):
        raise ValueError('Final matrices P_f must all be positive semi-definite')
    if not all([np.all(eigvals(R_i) > 0) for R_i in R]):
        raise ValueError('The weight matrices R_i must all be positive definite')


def run(m, A, B, Q, R, T_f):
    M = sum(m)
    N = len(B)
    P_f = get_P_f(m, N)

    check_input(m, Q, R, T_f, P_f, N)

    iterations = 5000
    s = np.linspace(T_f, 0, iterations)

    cl = True
    P_cl = odeint(solve_N_coupled_riccati_cl, P_f, s, args=(M, A, B, Q, R, N, cl))
    plot(s, P_cl, M, N)
    cl = False
    P_ol = odeint(solve_N_coupled_riccati_cl, P_f, s, args=(M, A, B, Q, R, N, cl))
    plot(s, P_ol, M, N)


if __name__ == '__main__':
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

    T_f = 5

    run(m, A, B, Q, R, T_f)
