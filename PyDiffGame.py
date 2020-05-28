import numpy as np
from numpy.linalg import eigvals
import matplotlib.pyplot as plt

def get_P_f(m, N):
    M = sum(m)
    P_size = M ** 2

    P_f = np.zeros((N * P_size,))

    for i in range(N):
        random_matrix = np.random.rand(M, M)
        P_f_i = random_matrix.transpose() @ random_matrix
        P_f_i = P_f_i.reshape((P_size,))

        if i == 0:
            P_f = P_f_i
        else:
            P_f = np.concatenate((P_f, P_f_i), axis=None)
    return P_f


def check_input(m, Q, R, T_f, P_f, N):
    M = sum(m)
    P_size = M ** 2

    if not all([m_i > 0 for m_i in m]):
        raise ValueError('The state variables dimensions must all be positive')
    if T_f <= 0:
        raise ValueError('The horizon must be positive')
    if not all([np.all(eigvals(Q_i) >= 0) for Q_i in Q]):
        raise ValueError('The weight matrices Q_i must all be positive semi-definite')
    if not all([eig >= 0 for eig_set in [eigvals(P_f[i * P_size:(i + 1) * P_size].reshape(M, M)) for i in range(N)]
                for eig in eig_set]):
        raise ValueError('Final matrices P_f must all be positive semi-definite')
    if not all([np.all(eigvals(R_i) > 0) for R_i in R]):
        raise ValueError('The weight matrices R_i must all be positive definite')


def plot(m, s, P):
    M = sum(m)
    print(len(m))
    V = range(1, len(m) + 1)
    U = range(1, int(M) + 1)

    legend = tuple(['${P' + str(i) + '}_{' + str(j) + str(k) + '}$' for i in V for j in U for k in U])

    plt.figure(dpi=130)
    plt.plot(s, P)
    plt.xlabel('Time')
    plt.legend(legend, loc='best', ncol=int(M / 2), prop={'size': int(20 / M)})
    plt.grid()
    plt.show()
