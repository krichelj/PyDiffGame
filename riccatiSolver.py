import numpy as np
from scipy.integrate import odeint, solve_ivp
import time

from PyDiffGame import check_input, get_care_P_f, plot, solve_N_coupled_riccati


def solve_riccati(m, A, B, Q, R, cl, T_f=5, P_f=None, data_points=5000):
    check_input(m, A, B, Q, R, T_f, P_f, data_points)
    M = sum(m)
    N = len(B)
    t = np.linspace(T_f, 0, data_points)

    if P_f is None:
        P_f = get_care_P_f(A, B, Q, R)
        sol = solve_ivp(fun=solve_N_coupled_riccati, t_span=[T_f, 0], y0=P_f, args=(M, N, A, B, Q, R, cl),
                        t_eval=t, rtol=1e-3, atol=1e-3)
        P = np.swapaxes(sol.y, 0, 1)
    else:
        P = odeint(func=solve_N_coupled_riccati, y0=P_f, t=t, args=(M, N, A, B, Q, R, cl), tfirst=True,
                   rtol=1e-3, atol=1e-3)

    return t, P


if __name__ == '__main__':
    start = time.time()
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

    t, P = solve_riccati(m, A, B, Q, R, cl)
    end = time.time()

    print(end - start)
    plot(m, t, P)
