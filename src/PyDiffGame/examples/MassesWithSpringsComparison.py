import numpy as np
from typing import Sequence, Optional

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.PyDiffGameComparison import PyDiffGameComparison
from PyDiffGame.Objective import GameObjective, LQRObjective


class MassesWithSpringsComparison(PyDiffGameComparison):
    def __init__(self,
                 N: int,
                 m: float,
                 k: float,
                 q: float,
                 r: float,
                 Ms: Optional[Sequence[np.array]] = None,
                 x_0: Optional[np.array] = None,
                 x_T: Optional[np.array] = None,
                 T_f: Optional[float] = None,
                 epsilon: Optional[float] = PyDiffGame.epsilon_default,
                 L: Optional[int] = PyDiffGame.L_default,
                 eta: Optional[int] = PyDiffGame.eta_default):
        M = m * np.eye(N)
        K = k * (- 2 * np.eye(N) + np.array([[int(abs(i - j) == 1) for j in range(N)] for i in range(N)]))
        M_inv = np.linalg.inv(M)

        if Ms is None:
            Ms = [row.reshape(1, N) for row in np.linalg.eig(M_inv @ K)[1]]

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

        lqr_objective = [LQRObjective(Q=Q_lqr,
                                      R_ii=R_lqr)]
        game_objectives = [GameObjective(Q=Q,
                                         R_ii=R,
                                         M_i=M_i) for Q, R, M_i in zip(Qs, Rs, Ms)]
        games_objectives = [lqr_objective,
                            game_objectives]

        super().__init__(args=args,
                         games_objectives=games_objectives,
                         continuous=True)


def multiprocess_worker_function(N: int) -> int:
    k = 1
    m = k * 10

    q = 1
    r = 5 * q

    x_0 = np.array([i for i in range(1, N + 1)] + [0] * N)
    x_T = 10 * x_0
    epsilon = 10 ** (-5)

    masses_with_springs = MassesWithSpringsComparison(N=N,
                                                      m=m,
                                                      k=k,
                                                      q=q,
                                                      r=r,
                                                      x_0=x_0,
                                                      x_T=x_T,
                                                      epsilon=epsilon)
    is_max_lqr = masses_with_springs(plot_state_spaces=False,
                                     print_costs=True,
                                     agnostic_costs=True)

    return int(is_max_lqr)


if __name__ == '__main__':
    Ns = list(range(4, 8))
    params = [Ns]
    PyDiffGameComparison.run_multiprocess(multiprocess_worker_function=multiprocess_worker_function,
                                          params=params)
