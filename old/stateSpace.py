import numpy as np
from scipy.integrate import odeint

from PyDiffGame import get_S_matrices, plot


def get_X0(m, n):
    X0 = np.zeros((n*m, ))

    for i in range(0, m):
        if i == 0:
            X0 = np.array([1] * n)
        else:
            x_i = i * 10
            X0 = np.concatenate((X0, [x_i] * n), axis=None)

    return X0


def sim_state_space(P, A, R, B, n, m, N, T_f, iterations):
    j = len(P)

    S_matrices = get_S_matrices(R, B, m)

    def stateDiffEqn(X, _):
        nonlocal j

        X = np.array(X).reshape(n, m)

        A_cl = A
        P_j = P[j - 1]

        for k in range(0, m):
            S_k = S_matrices[k]
            P_j_k = np.array(P_j[N * k:N * (k + 1)]).reshape(n, n)

            A_cl = A_cl - S_k @ P_j_k

        dXdt = A_cl @ X
        dXdt = np.array(dXdt).reshape(n * m)

        j -= 1
        return dXdt

    X0 = get_X0(m, n)
    t = np.linspace(0, T_f, iterations)

    X = odeint(stateDiffEqn, X0, t)
    is_P = False
    plot(t, X, m, n, is_P)
