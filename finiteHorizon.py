import numpy as np
from scipy.integrate import odeint, solve_ivp
from numpy.linalg import inv


from PyDiffGame import check_input, plot, get_care_P_f


def solve_N_coupled_riccati(_, P, M, N, A, B, Q, R, cl):
    P_size = M ** 2

    P_matrices = [(P[i * P_size:(i + 1) * P_size]).reshape(M, M) for i in range(N)]
    S_matrices = [B[i] @ inv(R[i]) @ B[i].transpose() for i in range(N)]
    SP_sum = sum(a @ b for a, b in zip(S_matrices, P_matrices))
    A_t = A.transpose()
    dPdt = np.zeros((M, M))

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
            dPdt = np.concatenate((dPdt, dPidt), axis=None)

    return dPdt


def solve_finite_horizon(m, A, B, Q, R, T_f, P_f, iterations, cl, events=None):
    check_input(m, A, B, Q, R, T_f, P_f)

    M = sum(m)
    N = len(B)
    s = np.linspace(T_f, 0, iterations)

    sol = solve_ivp(solve_N_coupled_riccati, [T_f, 0], P_f, args=(M, N, A, B, Q, R, cl), t_eval=s, events=events)
    y = sol.y

    P = [[y[j, i] for j in range(y.shape[0])] for i in range(y.shape[1])]

    return s, P


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
    P_f = get_care_P_f(A, B, Q, R)
    iterations = 300
    cl = True

    s, P = solve_finite_horizon(m, A, B, Q, R, T_f, P_f, iterations, cl)
    plot(m, s, P)