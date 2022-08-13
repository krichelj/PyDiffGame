import numpy as np
from typing import Sequence, Optional

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.LQRPyDiffGameComparison import LQRPyDiffGameComparison
from PyDiffGame.Objective import GameObjective, LQRObjective


class MassesWithSpringsComparison(LQRPyDiffGameComparison):
    def __init__(self,
                 N: int,
                 m: float,
                 k: float,
                 q: float,
                 r: float,
                 Ms: Sequence[np.array],
                 x_0: Optional[np.array] = None,
                 x_T: Optional[np.array] = None,
                 T_f: Optional[float] = None,
                 epsilon: Optional[float] = PyDiffGame.epsilon_default,
                 L: Optional[int] = PyDiffGame.L_default,
                 eta: Optional[int] = PyDiffGame.eta_default):
        M = m * np.eye(N)
        K = k * (- 2 * np.eye(N) + np.array([[int(abs(i - j) == 1) for j in range(N)] for i in range(N)]))
        M_inv = np.linalg.inv(M)
        N_z = np.zeros((N, N))
        N_e = np.eye(N)
        M_inv_K = M_inv @ K
        A = np.concatenate([np.concatenate([N_z, N_e],
                                           axis=1),
                            np.concatenate([M_inv_K, N_z],
                                           axis=1)],
                           axis=0)
        B = np.concatenate([N_z,
                            M_inv],
                           axis=0)
        Qs = [np.diag([0.0] * i + [q] + [0.0] * (N - 1) + [q] + [0.0] * (N - i - 1)) for i in range(N)]
        Rs = [np.array([r])] * N
        R_lqr = r * N_e
        Q_lqr = sum(Qs) / N

        state_variables_names = ['x_{' + str(i) + '}' for i in range(1, N + 1)] + \
                                ['\\dot{x}_{' + str(i) + '}' for i in range(1, N + 1)]
        args = {'A': A,
                'B': B,
                'x_0': x_0,
                'x_T': x_T,
                'T_f': T_f,
                'state_variables_names': state_variables_names,
                'epsilon': epsilon,
                'L': L,
                'eta': eta,
                'force_finite_horizon': T_f is not None}

        lqr_objective = LQRObjective(Q=Q_lqr, R_ii=R_lqr)
        game_objectives = [GameObjective(Q=Q, R_ii=R, M_i=M_i) for Q, R, M_i in zip(Qs, Rs, Ms)]

        super().__init__(args=args,
                         LQR_objective=lqr_objective,
                         game_objectives=game_objectives,
                         continuous=True)


N = 4
m = 10
k = m * 5

q = 1
r = 5 * q

# M1 = 1 / (2 * m) * np.array([[1, -1]])
# M2 = 1 / (2 * m) * np.array([[1, 1]])
# Ms = [M1, M2]

Ms = [1 / (2 * m) * np.random.random(size=(1, N)) for _ in range(N)]

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
