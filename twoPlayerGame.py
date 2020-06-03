import numpy as np
from scipy.integrate import odeint
from numpy.linalg import inv

from PyDiffGame import get_P_f, check_input, plot


def get_Ss(B, R):
    B1 = B[0]
    B2 = B[1]

    R11 = R[0]
    R22 = R[1]

    R11_inv = inv(R11)
    R22_inv = inv(R22)

    S11 = B1 @ R11_inv @ B1.transpose()
    S22 = B2 @ R22_inv @ B2.transpose()

    return S11, S22


def solve_two_coupled_riccati(P, _, M, A, B, Q, R, cl):
    P_size = M ** 2

    P1 = P[0:P_size].reshape(M, M)
    P2 = P[P_size:2 * P_size].reshape(M, M)
    Q1 = Q[0]
    Q2 = Q[1]

    S11, S22 = get_Ss(B, R)

    A_t = A.transpose()

    dP1dt = - A_t @ P1 - P1 @ A - Q1 + P1 @ (S11 @ P1 + S22 @ P2)
    dP2dt = - A_t @ P2 - P2 @ A - Q2 + P2 @ (S22 @ P2 + S11 @ P1)

    if cl:
        dP1dt = dP1dt + P2 @ S22 @ P1
        dP2dt = dP1dt + P1 @ S11 @ P2

    dP1dt = dP1dt.reshape(P_size)
    dP2dt = dP2dt.reshape(P_size)

    dPdt = np.concatenate((dP1dt, dP2dt), axis=None)

    return dPdt


def run(m, A, B, Q, R, T_f):
    M = sum(m)
    N = 2
    P_f = get_P_f(m, N)

    s = np.linspace(T_f, 0, iterations)

    cl = True
    P_cl = odeint(solve_two_coupled_riccati, P_f, s, args=(M, A, B, Q, R, cl))
    plot(m, s, P_cl)
    cl = False
    P_ol = odeint(solve_two_coupled_riccati, P_f, s, args=(M, A, B, Q, R, cl))
    plot(m, s, P_ol)


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
    iterations = 5000

    run(m, A, B, Q, R, T_f)
