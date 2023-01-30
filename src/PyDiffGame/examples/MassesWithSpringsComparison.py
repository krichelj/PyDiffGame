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
        K = k * (2 * N_e - np.array([[int(abs(i - j) == 1) for j in range(N)] for i in range(N)]))
        M_masses_inv = np.linalg.inv(M_masses)

        M_inv_K = M_masses_inv @ K

        if Ms is None:
            eigenvectors = np.linalg.eig(M_inv_K)[1]
            Ms = [eigenvector.reshape(1, N) for eigenvector in eigenvectors]
            # Ms = [a / np.linalg.norm(a) ** 2 for a in Ms]

        A = np.block([[N_z, N_e],
                      [-M_inv_K, N_z]])
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
        R_lqr = 1 / 4 * r * N_e
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

        self.figure_filename_generator = lambda g: ('LQR' if g.is_LQR() else f'{N}-players') + \
                                                   f'_large_{1 if q[0] > q[1] else 2}'

        super().__init__(args=args,
                         M=M,
                         games_objectives=games_objectives,
                         continuous=True)

    @staticmethod
    def are_all_orthogonal(vectors: Sequence[np.array]) -> bool:
        """
        Check if a set of vectors are all orthogonal to each other.

        Parameters
        ----------

        vectors: sequence of vectors of len(n)
            A sequence of numpy arrays representing the vectors

        Returns
        ----------

        all_orthogonal: bool
            True if all the vectors are orthogonal to each other, False otherwise
        """

        all_orthogonal = not any([any([abs(vectors[i] @ vectors[j].T) > 1e-10 for j in range(i + 1, len(vectors))])
                                  for i in range(len(vectors))])

        return all_orthogonal

    @staticmethod
    def are_all_unit_vectors(vectors: Sequence[np.array]) -> bool:
        """
        Check if a set of vectors are all of length 1.

        Parameters
        ----------

        vectors: sequence of vectors of len(n)
            A sequence of numpy arrays representing the vectors

        Returns
        ----------

        all_orthogonal: bool
            True if all the vectors are orthogonal to each other, False otherwise
        """

        all_unit_vectors = all([abs(np.linalg.norm(v) - 1) < 1e-10 for v in vectors])

        return all_unit_vectors


def multiprocess_worker_function(N: int,
                                 k: float,
                                 m: float,
                                 q: float,
                                 r: float,
                                 epsilon_x: float,
                                 epsilon_P: float):
    x_0 = np.array([10 * i for i in range(1, N + 1)] + [0] * N)
    x_T = x_0 * 10 if N == 2 else np.array([(10 * i) ** 3 for i in range(1, N + 1)] + [0] * N)

    output_variables_names = ['$x_1 + x_2$', '$x_1 - x_2$', '$\\dot{x}_1 + \\dot{x}_2$', '$\\dot{x}_1 - \\dot{x}_2$'] \
        if N == 2 else [f'$q_{i}$' for i in range(1, N + 1)] + ['$\\dot{q}_{' + str(i) + '}$' for i in range(1, N + 1)]

    masses_with_springs = MassesWithSpringsComparison(N=N,
                                                      m=m,
                                                      k=k,
                                                      q=q,
                                                      r=r,
                                                      x_0=x_0,
                                                      x_T=x_T,
                                                      T_f=25,
                                                      epsilon_x=epsilon_x,
                                                      epsilon_P=epsilon_P)

    masses_with_springs(plot_state_spaces=True,
                        plot_Mx=True,
                        output_variables_names=output_variables_names,
                        save_figure=True,
                        figure_filename=masses_with_springs.figure_filename_generator
                        )


if __name__ == '__main__':
    d = 0.2
    N = 8

    Ns = [N]
    ks = [10]
    ms = [50]
    rs = [1]
    epsilon_xs = [10e-7]
    epsilon_Ps = [10e-7]
    qs = [[500, 2000], [500, 250]] if N == 2 else \
        [[500 * (1 + d) ** i for i in range(N)], [500 * (1 - d) ** i for i in range(N)]]

    # N, k, m, r, epsilon_x, epsilon_P = Ns[0], ks[0], ms[0], rs[0], epsilon_xs[0], epsilon_Ps[0]
    # q = [500, 2000] if N == 2 else [500 * 1.3 ** i for i in range(N)]
    # x_0 = np.array([10 * i for i in range(1, N + 1)] + [0] * N)
    # x_T = x_0 * 10 if N == 2 else np.array([(10 * i) ** 3 for i in range(1, N + 1)] + [0] * N)
    #
    # masses_with_springs = MassesWithSpringsComparison(N=N,
    #                                                   m=m,
    #                                                   k=k,
    #                                                   q=q,
    #                                                   r=r,
    #                                                   x_0=x_0,
    #                                                   x_T=x_T,
    #                                                   # T_f=25,
    #                                                   epsilon_x=epsilon_x,
    #                                                   epsilon_P=epsilon_P)
    #
    # output_variables_names = ['$x_1 + x_2$', '$x_1 - x_2$', '$\\dot{x}_1 + \\dot{x}_2$', '$\\dot{x}_1 - \\dot{x}_2$'] \
    #     if N == 2 else [f'$q_{i}$' for i in range(1, N + 1)] + ['$\\dot{q}_{' + str(i) + '}$' for i in range(1, N + 1)]
    #
    # masses_with_springs(plot_state_spaces=True,
    #                     plot_Mx=True,
    #                     output_variables_names=output_variables_names,
    #                     save_figure=True,
    #                     figure_filename=masses_with_springs.figure_filename_generator
    #                     )

    params = [Ns, ks, ms, qs, rs, epsilon_xs, epsilon_Ps]
    PyDiffGameLQRComparison.run_multiprocess(multiprocess_worker_function=multiprocess_worker_function,
                                             values=params)
