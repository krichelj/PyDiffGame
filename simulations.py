import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve

from PyDiffGame import get_P_f, plot, solve_m_coupled_riccati, solve_m_coupled_riccati_cl
from stateSpace import sim_state_space
from closedLoopAlgebraic import get_coupled_R, solve_algebraic_riccati


def run_open_loop_simulation(P_f, s, A, B, Q, R, m, n, N):
    P = odeint(solve_m_coupled_riccati, P_f, s, args=(A, B, Q, R, m, n, N))
    is_P = True
    plot(s, P, m, n, is_P)

    return P


def run_closed_loop_simulation(P_f, s, A, B, Q, R, m, n, N):
    P_cl = odeint(solve_m_coupled_riccati_cl, P_f, s, args=(A, B, Q, R, m, n, N))
    is_P = True
    plot(s, P_cl, m, n, is_P)

    return P_cl


def run_closed_loop_algebraic_simulation():
    coupled_R = get_coupled_R(m, n)
    P0 = [np.ones((n, n)) * 300] * m

    sol = fsolve(solve_algebraic_riccati, P0, args=(A, B, Q, R, coupled_R, m, n, N))
    sol[sol < 1 * 10 ** (-6)] = 0

    for i in range(m):
        print('P' + str(i + 1) + ' = ' + str(sol[i * N:(i + 1) * N].reshape(n, n)))


if __name__ == '__main__':
    m = 2
    n = 2
    N = n ** 2

    A = np.array([[2, 0],
                  [0, 2]])
    B = [np.array([[1, 0],
                   [0, 1]])] * m
    Q = [np.array([[5, 0],
                   [0, 6]])] * m
    R = [np.array([[100, 0],
                   [0, 300]])] * m

    P_f = get_P_f(m, N)
    T_f = 5
    iterations = 5000
    s = np.linspace(T_f, 0, iterations)

    output_P = run_open_loop_simulation(P_f, s, A, B, Q, R, m, n, N)
    sim_state_space(output_P, A, R, B, n, m, N, T_f, iterations)

    # output_P_cl = run_closed_loop_simulation(P_f, s, A, B, Q, R, m, n, N)
    # sim_state_space(output_P_cl, A, R, B, n, m, N, T_f, iterations)

    # run_closed_loop_algebraic_simulation()
