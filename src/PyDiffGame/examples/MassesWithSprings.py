import numpy as np
from numpy.linalg import inv

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.PyDiffGame import PyDiffGameComparison
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame
from PyDiffGame.LQR import ContinuousLQR


class MassesWithSpringsComparison(PyDiffGameComparison):
    def __init__(self,
                 N: int,
                 m: float,
                 k: float,
                 q: float,
                 r: float,
                 Ms: list[np.array],
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 epsilon: float = PyDiffGame.epsilon_default,
                 L: int = PyDiffGame.L_default,
                 eta: int = PyDiffGame.eta_default):
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

        A_2 = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [-2 * k / m, k / m, 0, 0],
                        [k / m, -2 * k / m, 0, 0]])

        # A_3 = np.array([[0, 0, 0, 1, 0, 0],
        #                 [0, 0, 0, 0, 1, 0],
        #                 [0, 0, 0, 0, 0, 1],
        #                 [-2 * k / m, k / m, 0, 0, 0, 0],
        #                 [k / m, -2 * k / m, k / m, 0, 0, 0],
        #                 [0, k / m, -2 * k / m, 0, 0, 0]])

        B = np.concatenate([N_z, M_inv], axis=0)

        B_2 = 1 / m * np.array([[0, 0],
                                [0, 0],
                                [1, 0],
                                [0, 1]])

        Qs = [np.diag([0.0] * i + [q] + [0.0] * (N - 1) + [q ** 2] + [0.0] * (N - i - 1)) for i in range(N)]
        Rs = [np.array([r])] * N
        R_lqr = r * N_e

        state_variables_names = ['x_{' + str(i) + '}' for i in range(1, N + 1)] + \
                                ['\\dot{x}_{' + str(i) + '}' for i in range(1, N + 1)]
        args = {'A': A,
                'B': B,
                'Q': sum(Qs) / N,
                'R': R_lqr,
                'x_0': x_0,
                'x_T': x_T,
                'T_f': T_f,
                'state_variables_names': state_variables_names,
                'epsilon': epsilon,
                'L': L,
                'eta': eta,
                'force_finite_horizon': T_f is not None}

        games = {'LQR': ContinuousLQR(**args),
                 f'{N}_player_game': ContinuousPyDiffGame(**args | {'Q': Qs, 'R': Rs, 'Ms': Ms})}

        super().__init__(games=games)


N = 2
m, k = 10, 100
q, r = 100, 1000

M_1 = 1 / (2 * m) * np.array([[1, 1]])
M_2 = 1 / (2 * m) * np.array([[1, -1]])
Ms = [M_1, M_2]

x_0 = np.array([1,  # x_1
                2,  # x_2
                0,  # x_1_dot
                0]  # x_2_dot
               )
x_T = np.array([10,  # x_1
                20,  # x_2
                0,  # x_1_dot
                0]  # x_2_dot
               )

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

# lqr_masses.run_simulation(plot_state_space=True)
# # lqr_masses.save_figure(filename='LQR')
#
# game_masses = MassesWithSprings(m=m,
#                                 k=k,
#                                 q=q,
#                                 r=r,
#                                 x_0=x_0,
#                                 x_T=x_T,
#                                 epsilon=epsilon
#                                 )
# game_masses.run_simulation(plot_state_space=True)
# # game_masses.save_figure(filename='game')
#
# print(f'cost difference: {lqr_masses.get_costs() - game_masses.get_costs().max()}')
