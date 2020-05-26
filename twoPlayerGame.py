import numpy as np
from scipy.integrate import odeint
from numpy.linalg import inv
import matplotlib.pyplot as plt

from PyDiffGame import get_P_f


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


def solve_two_coupled_riccati_cl(P, _, m, A, B, Q, R, N):
    M = sum(m)
    P_size = M ** 2

    P1 = P[0:P_size].reshape(M, M)
    P2 = P[P_size:2 * P_size].reshape(M, M)
    Q1 = Q[0]
    Q2 = Q[1]

    S11, S22 = get_Ss(B, R)

    A_t = A.transpose()
    dP1dt = - A_t @ P1 - P1 @ A - Q1 + P1 @ (S11 @ P1 + S22 @ P2)
    dP2dt = - A_t @ P2 - P2 @ A - Q2 + P2 @ (S22 @ P2 + S11 @ P1)
    dP1dt = dP1dt.reshape(P_size)
    dP2dt = dP2dt.reshape(P_size)

    dPdt = np.concatenate((dP1dt, dP2dt), axis=None)

    return dPdt


def plot(s, P, m):
    M = sum(m)
    V = range(1, len(m) + 1)
    U = range(1, M + 1)

    legend = tuple(['${P' + str(i) + '}_{' + str(j) + str(k) + '}$' for i in V for j in U for k in U])

    plt.figure(dpi=130)
    plt.plot(s, P)
    plt.xlabel('Time')
    plt.legend(legend, loc='best', ncol=int(M / 2), prop={'size': int(20 / M)})
    plt.grid()
    plt.show()


# def sim_state_space(P, A, R, B, n, m, N, T_f, iterations):
#     M = sum(m)
#     P_size = M ** 2
#     j = len(P)
#
#     def stateDiffEqn(X, _):
#         nonlocal j
#
#         X = np.array(X).reshape(n, )
#         P_j = P[j - 1]
#         P1 = np.array(P[0:P_size]).reshape(M, M)
#         P2 = np.array(P[P_size:2 * P_size]).reshape(M, M)
#
#         S11, S22 = get_Ss(B, R)
#
#         A_cl = A - S11 @
#
#         dXdt = A_cl @ X
#         dXdt = np.array(dXdt).reshape(n * m)
#
#         j -= 1
#         return dXdt


if __name__ == '__main__':

    def run(m, A, B, Q, R, T_f, P_f, iterations):
        M = sum(m)
        N = len(B)
        s = np.linspace(T_f, 0, iterations)
        P_cl = odeint(solve_two_coupled_riccati_cl, P_f, s, args=(m, A, B, Q, R, N))
        plot(s, P_cl, m)

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

    P_f = get_P_f(m)
    T_f = 5
    iterations = 5000
    run(m, A, B, Q, R, T_f, P_f, iterations)