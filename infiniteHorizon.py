import numpy as np

from PyDiffGame import get_care_P_f
from finiteHorizon import solve_finite_horizon


def solve_infinite_horizon(m, A, B, Q, R, T_f, P_f, iterations, cl):
    M = sum(m)
    N = len(B)
    P_size = M ** 2

    def hit_ground(t, y, M, N, A, B, Q, R, cl):
        return y[0]

    hit_ground.terminal = True

    _, P = solve_finite_horizon(m, A, B, Q, R, T_f, P_f, iterations, cl, hit_ground)


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
    iterations = 300
    P_f = get_care_P_f(A, B, Q, R)
    cl = True

    solve_infinite_horizon(m, A, B, Q, R, T_f, P_f, iterations, cl)
