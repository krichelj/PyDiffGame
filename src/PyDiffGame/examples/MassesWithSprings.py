import numpy as np
from numpy.linalg import inv

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.PyDiffGame import PyDiffGameComparison
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame
from PyDiffGame.LQR import ContinuousLQR


class MassesWithSpringsComparison(PyDiffGameComparison):
    def __init__(self,
                 m: float,
                 k: float,
                 q: float,
                 r: float,
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 epsilon: float = PyDiffGame.epsilon_default,
                 L: int = PyDiffGame.L_default,
                 eta: int = PyDiffGame.eta_default):

        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [-2 * k / m, k / m, 0, 0],
                      [k / m, -2 * k / m, 0, 0]])

        B = 1 / m * np.array([[0, 0],
                              [0, 0],
                              [1, 0],
                              [0, 1]])

        M_1 = 1 / (2 * m) * np.array([[1, 1]])
        M_2 = 1 / (2 * m) * np.array([[1, -1]])
        m_1 = M_1.shape[0]
        m_2 = M_2.shape[0]

        M = np.concatenate([M_1, M_2])
        M_inv = inv(M)
        M_1_tilde = M_inv[:, 0:m_1]
        M_2_tilde = M_inv[:, m_1:m_1 + m_2]

        n = A.shape[0]
        B_1 = (B @ M_1_tilde).reshape((n, m_1))
        B_2 = (B @ M_2_tilde).reshape((n, m_2))

        B_multiplayer = [B_1, B_2]

        Q_1 = np.diag([q, 0, q ** 2, 0])
        Q_2 = np.diag([0, q, 0, q ** 2])
        Q_multiplayer = [Q_1, Q_2]
        R_lqr = np.diag([r, r])
        R_multiplayer = [np.array([r])] * 2

        state_variables_names = ['x_1', 'x_2', '\\dot{x}_1', '\\dot{x}_2']

        optimizers = {**{f'LQR_{i}': ContinuousLQR(A=A,
                                                   B=B,
                                                   Q=Q_i,
                                                   R=R_lqr,
                                                   x_0=x_0,
                                                   x_T=x_T,
                                                   T_f=T_f,
                                                   state_variables_names=state_variables_names,
                                                   epsilon=epsilon,
                                                   L=L,
                                                   eta=eta,
                                                   force_finite_horizon=T_f is not None)
                         for i, Q_i in enumerate(Q_multiplayer)},
                      **{'game': ContinuousPyDiffGame(A=A,
                                                      B=B_multiplayer,
                                                      Q=Q_multiplayer,
                                                      R=R_multiplayer,
                                                      x_0=x_0,
                                                      x_T=x_T,
                                                      T_f=T_f,
                                                      state_variables_names=state_variables_names,
                                                      epsilon=epsilon,
                                                      L=L,
                                                      eta=eta,
                                                      force_finite_horizon=T_f is not None)}}

        super().__init__(optimizers=optimizers)


m, k = 10, 100
q, r = 1000, 10000

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

masses_with_springs = MassesWithSpringsComparison(m=m,
                                                  k=k,
                                                  q=q,
                                                  r=r,
                                                  x_0=x_0,
                                                  x_T=x_T,
                                                  epsilon=epsilon
                                                  )
masses_with_springs.run_optimizers()

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
