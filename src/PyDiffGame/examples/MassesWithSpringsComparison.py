import numpy as np
from typing import Sequence, Optional

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.PyDiffGameLQRComparison import PyDiffGameLQRComparison
from PyDiffGame.Objective import GameObjective, LQRObjective


class MassesWithSpringsComparison(PyDiffGameLQRComparison):
    def __init__(self,
                 N: int,
                 m: float,
                 k: float,
                 q: float | Sequence[float],
                 r: float,
                 Ms: Optional[Sequence[np.array]] = None,
                 x_0: Optional[np.array] = None,
                 x_T: Optional[np.array] = None,
                 T_f: Optional[float] = None,
                 epsilon_x: Optional[float] = PyDiffGame.epsilon_x_default,
                 epsilon_P: Optional[float] = PyDiffGame.epsilon_P_default,
                 L: Optional[int] = PyDiffGame.L_default,
                 eta: Optional[int] = PyDiffGame.eta_default):
        N_e = np.eye(N)
        N_z = np.zeros((N, N))

        M_masses = m * N_e
        K = k * (- 2 * N_e + np.array([[int(abs(i - j) == 1) for j in range(N)] for i in range(N)]))
        M_masses_inv = np.linalg.inv(M_masses)
        M_inv_K = M_masses_inv @ K

        if Ms is None:
            eigenvectors = np.linalg.eig(-M_inv_K)[1]
            Ms = [eigenvector.reshape(1, N) / eigenvector[0] for eigenvector in eigenvectors]

        A = np.block([[N_z, N_e],
                      [M_inv_K, N_z]])
        B = np.block([[N_z],
                      [M_masses_inv]])

        Qs = [np.diag([0.0] * i + [q] + [0.0] * (N - 1) + [q] + [0.0] * (N - i - 1))
              if isinstance(q, (int, float)) else
              np.diag([0.0] * i + [q[i]] + [0.0] * (N - 1) + [q[i]] + [0.0] * (N - i - 1)) for i in range(N)]


        # Qs = [np.array([[int(i == j)]*2*N for j in range(2 * N)]) for i in range(N)]
        # Qs = [Qs[i].T + Qs[i] - np.diag([0] * i + [1] + [0]*(N - i + 1)) for i in range(N)]

        M = np.concatenate(Ms,
                           axis=0)
        Q_mat = np.kron(a=np.eye(2),
                        b=M)

        Qs = [Q_mat.T @ Q @ Q_mat for Q in Qs]

        Rs = [np.array([r])] * N
        R_lqr = r * N_e
        Q_lqr = q * np.eye(2 * N) if isinstance(q, (int, float)) else np.diag(2 * q)

        state_variables_names = ['x_{' + str(i) + '}' for i in range(1, N + 1)] + \
                                ['\\dot{x}_{' + str(i) + '}' for i in range(1, N + 1)]
        args = {'A': A,
                'B': B,
                'x_0': x_0,
                'x_T': x_T,
                'T_f': T_f,
                'state_variables_names': state_variables_names,
                'epsilon_x': epsilon_x,
                'epsilon_P': epsilon_P,
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
                         M=M,
                         games_objectives=games_objectives,
                         continuous=True)


def multiprocess_worker_function(N: int,
                                 k: float,
                                 m: float,
                                 q: float,
                                 r: float,
                                 epsilon_x: float,
                                 epsilon_P: float):
    x_0_vals = [10 * i for i in range(1, N + 1)]
    x_0 = np.array(x_0_vals + [0] * N)
    x_T = np.array(list(reversed([x ** 3 for x in x_0_vals])) + [0] * N)
    # x_T = 10 * x_0

    masses_with_springs = MassesWithSpringsComparison(N=N,
                                                      m=m,
                                                      k=k,
                                                      q=q,
                                                      r=r,
                                                      x_0=x_0,
                                                      x_T=x_T,
                                                      epsilon_x=epsilon_x,
                                                      epsilon_P=epsilon_P)
    # output_variables_names = ['$x_1 + x_2$', '$x_1 - x_2$', '$\\dot{x}_1 + \\dot{x}_2$', '$\\dot{x}_1 - \\dot{x}_2$']
    output_variables_names = [f'$n_{i + 1}$' for i in range(N)] + ['$\\dot{n}_{' + str(i + 1) + '}$' for i in range(N)]
    masses_with_springs(plot_state_spaces=True,
                        plot_Mx=True,
                        output_variables_names=output_variables_names,
                        save_figure=True
                        )


if __name__ == '__main__':
    Ns = [8]
    ks = [10]
    ms = [50]
    qs = [[10 * (10 ** (N - (i+1))) for i in range(N)] for N in Ns]
    rs = [1]
    epsilon_xs = [10 ** (-8)]
    epsilon_Ps = [10 ** (-8)]

    # N, k, m, q, r, epsilon_x, epsilon_P = Ns[0], ks[0], ms[0], qs[0], rs[0], epsilon_xs[0], epsilon_Ps[0]
    # x_0 = np.array([10 * i for i in range(1, N + 1)] + [0] * N)
    # x_T = x_0 * 10
    #
    # masses_with_springs = MassesWithSpringsComparison(N=N,
    #                                                   m=m,
    #                                                   k=k,
    #                                                   q=q,
    #                                                   r=r,
    #                                                   x_0=x_0,
    #                                                   x_T=x_T,
    #                                                   epsilon_x=epsilon_x,
    #                                                   epsilon_P=epsilon_P)
    #
    # # output_variables_names = ['$x_1 + x_2$', '$x_1 - x_2$', '$\\dot{x}_1 + \\dot{x}_2$', '$\\dot{x}_1 - \\dot{x}_2$']
    # output_variables_names = [f'$n_{i}$' for i in range(N)] + ['$\\dot{n}_{' + str(i) + '}$' for i in range(N)]
    #
    # masses_with_springs(plot_state_spaces=True,
    #                     plot_Mx=True,
    #                     output_variables_names=output_variables_names,
    #                     save_figure=False
    #                     )

    params = [Ns, ks, ms, qs, rs, epsilon_xs, epsilon_Ps]
    PyDiffGameLQRComparison.run_multiprocess(multiprocess_worker_function=multiprocess_worker_function,
                                             values=params)
