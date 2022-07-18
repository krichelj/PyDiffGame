import numpy as np
from numpy.linalg import inv
from typing import Collection, Optional

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.PyDiffGameComparison import PyDiffGameComparison


class MassesWithSpringsComparison(PyDiffGameComparison):
    def __init__(self,
                 N: int,
                 m: float,
                 k: float,
                 q: float,
                 r: float,
                 Ms: Collection[np.array],
                 x_0: Optional[np.array] = None,
                 x_T: Optional[np.array] = None,
                 T_f: Optional[float] = None,
                 epsilon: Optional[float] = PyDiffGame.epsilon_default,
                 L: Optional[int] = PyDiffGame.L_default,
                 eta: Optional[int] = PyDiffGame.eta_default):
        M = m * np.eye(N)
        K = - 2 * k * np.eye(N) + k * np.array([[int(abs(i - j) == 1) for j in range(N)] for i in range(N)])
        M_inv = inv(M)
        N_z = np.zeros((N, N))
        N_e = np.eye(N)
        A = np.concatenate([np.concatenate([N_z, N_e],
                                           axis=1),
                            np.concatenate([M_inv @ K, N_z],
                                           axis=1)],
                           axis=0)
        B = np.concatenate([N_z, M_inv], axis=0)
        Qs = [np.diag([0.0] * i + [q] + [0.0] * (N - 1) + [q ** 2] + [0.0] * (N - i - 1)) for i in range(N)]
        Rs = [np.array([r])] * N
        R_lqr = r * N_e
        Q_lqr = sum(Qs) / N

        state_variables_names = ['x_{' + str(i) + '}' for i in range(1, N + 1)] + \
                                ['\\dot{x}_{' + str(i) + '}' for i in range(1, N + 1)]
        LQR_args = {'A': A,
                    'B': B,
                    'Q': Q_lqr,
                    'R': R_lqr,
                    'x_0': x_0,
                    'x_T': x_T,
                    'T_f': T_f,
                    'state_variables_names': state_variables_names,
                    'epsilon': epsilon,
                    'L': L,
                    'eta': eta,
                    'force_finite_horizon': T_f is not None}

        super().__init__(continuous=True,
                         args=LQR_args,
                         QRMs=[(Qs, Rs, Ms)])


N = 2
m, k = 10, 100
q, r = 100, 2000

M1 = 1 / (2 * m) * np.array([[1, -1]])
M2 = 1 / (2 * m) * np.array([[1, 1]])

# M_3 = 1 / (2 * m) * np.array([[0, 0, 1]])

# o = ortho_group.rvs(dim=N)

Ms = [M1, M2]
# Ms = [1 / (2 * m) * random(size=(1, N)) for _ in range(N)]

x_0 = np.array([i for i in range(1, N + 1)] + [0] * N)
x_T = 10 * x_0

epsilon = 10 ** (-3)

masses_with_springs = MassesWithSpringsComparison(N=N,
                                                  m=m,
                                                  k=k,
                                                  q=q,
                                                  r=r,
                                                  Ms=Ms,
                                                  x_0=x_0,
                                                  x_T=x_T,
                                                  epsilon=epsilon)
masses_with_springs.run_simulations()

# game = masses_with_springs.games[-1]
#
# for i in range(len(game)):
#     M_i = game._Ms[i]
#     R_ii = game._R[i].reshape((M_i.shape[0], M_i.shape[0]))
#
#     R_i = M_i.T @ R_ii @ M_i
#     R_i_inv = inv(R_i)
#
#     P_i = game._get_P_f_i(i)
#
#     K_i = R_i_inv @ game._B.T @ P_i
#     print(K_i)
#     print(game._get_K_i(i))
