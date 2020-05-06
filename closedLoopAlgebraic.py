from numpy.linalg import inv
import numpy as np

from PyDiffGame import get_P_matrices


def get_A_cl(P_matrices, A, B, R, m):
    A_cl = A

    for i in range(m):
        P_i = P_matrices[i]

        B_i = B[i]
        B_i_t = B_i.transpose()
        R_i = R[i]
        R_i_inv = inv(R_i)

        S_i = B_i @ R_i_inv @ B_i_t @ P_i
        A_cl = A_cl - S_i

    A_cl_t = A_cl.transpose()

    return A_cl, A_cl_t


def get_curr_sum(P_matrices, B, R, coupled_R, n, m, i):
    G = np.zeros((n, n))

    for j in range(m):
        P_j = P_matrices[j]

        B_j = B[j]
        B_j_t = B_j.transpose()
        R_j = R[j]
        R_j_inv = inv(R_j)
        R_j_inv_t = R_j_inv.transpose()

        R_i_j = coupled_R[i][j]

        G = G + P_j @ B_j @ R_j_inv_t @ R_i_j @ R_j_inv @ B_j_t @ P_j

    return G


def solve_algebraic_riccati(P, *data):
    A, B, Q, R, coupled_R, m, n, N = data

    P_matrices = get_P_matrices(P, m, n, N)
    A_cl, A_cl_t = get_A_cl(P_matrices, A, B, R, m)

    solution_matrices = []
    equations_to_solve = []

    for i in range(m):
        P_i = P[i * N:(i + 1) * N].reshape(n, n)
        solution_matrices.append(P_i)

        Q_i = Q[i]
        G = get_curr_sum(P_matrices, B, R, coupled_R, n, m, i)

        curr_equation = P_i @ A_cl + A_cl_t @ P_i + Q_i + G
        curr_equation = np.array(curr_equation).reshape(N)

        equations_to_solve.append(curr_equation)

    output = np.concatenate(equations_to_solve, axis=None)
    return output


def get_coupled_R(m, n):
    coupled_R = []

    for i in range(m):
        curr_Rs = []
        for j in range(m):
            if j == i:
                curr_Rs.append(np.eye(n))
            else:
                curr_Rs.append(np.zeros((n, n)))
        coupled_R.append(curr_Rs)

    return coupled_R