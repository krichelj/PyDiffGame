import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def get_ranges(n, m):
    U = range(1, n + 1)
    V = range(1, m + 1)

    return U, V


def solve_m_coupled_riccati(P_f, A, B, R, Q, m, s, n, N):
    S_matrices = []

    def get_parameters(P):
        nonlocal S_matrices

        P_matrices = []

        M = np.zeros((n, n))

        for i in range(0, m):
            R_i = R[i]
            R_i_inv = np.linalg.inv(R_i)

            B_i = B[i]
            B_i_t = B[i].transpose()

            S_i = B_i @ R_i_inv @ B_i_t

            S_matrices.append(S_i)

            P_i = np.array(P[N * i:N * (i + 1)]).reshape(n, n)
            P_matrices.append(P_i)

            M = M + S_i @ P_i

        return P_matrices, M

    def coupled_riccati_ode_solver(P, t):
        P_matrices, M = get_parameters(P)
        A_t = A.transpose()
        dPdt = np.zeros((n, n))

        for i in range(m):

            P_i = P_matrices[i]
            Q_i = Q[i]

            dP_idt = P_i @ M - P_i @ A - A_t @ P_i - Q_i
            dP_idt = np.array(dP_idt).reshape(N)

            if i == 0:
                dPdt = dP_idt
            else:
                dPdt = np.concatenate((dPdt, dP_idt), axis=None)

        return dPdt

    return_P = odeint(coupled_riccati_ode_solver, P_f, s)

    return return_P, S_matrices


def plot_riccati(s, P, m, n):
    plt.figure(dpi=130)
    plt.plot(s, P)
    plt.xlabel('Time')

    U, V = get_ranges(n, m)

    P_legend = tuple(['${P' + str(i) + '}_{' + str(j) + str(k) + '}$' for i in V for j in U for k in U])

    plt.legend(P_legend, loc='best', ncol=int(m / 2), prop={'size': int(30 * 1 / m)})
    plt.grid()
    plt.show()


def get_P_f(m):
    P_f = 0

    for i in range(0, m):
        if i == 0:
            P_f = np.array([0, 0, 0, 0])
        else:
            k = i * 10
            P_f = np.concatenate((P_f, [k, k, k, k]), axis=None)

    return P_f


def simStateSpace(P, A, n, N, m, S_matrices, T_f, iterations):
    j = len(P)

    def stateDiffEqn(X, t):

        nonlocal j

        X = np.array(X).reshape(n, m)

        A_cl = A
        P_j = P[j - 1]

        for k in range(0, m):
            S_k = S_matrices[k]
            P_j_k = np.array(P_j[N * k:N * (k + 1)]).reshape(n, n)

            A_cl = A_cl - S_k @ P_j_k

        dXdt = np.dot(A_cl, X)
        dXdt = np.array(dXdt).reshape(n * m)

        j -= 1
        return dXdt

    X0 = 0

    for i in range(0, m):
        if i == 0:
            X0 = np.array([1] * n)
        else:
            e = i * 10
            X0 = np.concatenate((X0, [e] * n), axis=None)

    t = np.linspace(0, T_f, iterations)

    X = odeint(stateDiffEqn, X0, t)
    plt.figure(dpi=130)
    plt.plot(t, X)
    plt.xlabel('Time')

    U, V = get_ranges(n, m)
    X_legend = ['${X' + str(i) + '}_{' + str(j) + '}$'
                for i in V for j in U]

    plt.legend(tuple(X_legend), ncol=int(m / 2), loc='best',
               prop={'size': int(50 / m)})
    plt.grid()
    plt.show()


if __name__ == '__main__':
    input_m = 3
    input_n = 2
    input_N = input_n ** 2

    input_A = np.array([[2, 0],
                        [0, 2]])

    input_B = [np.array([[1, 0],
                         [0, 1]])] * input_m
    input_R = [np.array([[100, 0],
                         [0, 300]])] * input_m
    input_Q = [np.array([[5, 0],
                         [0, 6]])] * input_m

    input_P_f = get_P_f(input_m)
    input_T_f = 5
    input_iterations = 5000
    input_s = np.linspace(input_T_f, 0, input_iterations)

    output_P, S_matrices = solve_m_coupled_riccati(input_P_f, input_A, input_B, input_R, input_Q, input_m, input_s,
                                                   input_n, input_N)

    plot_riccati(input_s, output_P, input_m, input_n)

    simStateSpace(output_P, input_A, input_n, input_N, input_m, S_matrices, input_T_f, input_iterations)
