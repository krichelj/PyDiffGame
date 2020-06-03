import numpy as np
from scipy.integrate import odeint
from numpy.linalg import inv, norm


from PyDiffGame import check_input, plot, get_care_P_f


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


def solve_N_coupled_riccati(P, _, M, A, B, Q, R, N, cl):
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


def run(m, A, B, Q, R, T_f):
    M = sum(m)
    N = len(B)

    P_f = get_care_P_f(M, N, A, B, Q, R)
    check_input(m, A, B, Q, R, T_f, P_f, N)

    iterations = 300
    s = np.linspace(T_f, 0, iterations)

    cl = True
    P_cl = odeint(solve_N_coupled_riccati, P_f, s, args=(M, A, B, Q, R, N, cl))
    plot(m, s, P_cl)
    # cl = False
    # P_ol = odeint(solve_N_coupled_riccati, P_f, s, args=(M, A, B, Q, R, N, cl))
    # plot(m, s, P_ol)


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
