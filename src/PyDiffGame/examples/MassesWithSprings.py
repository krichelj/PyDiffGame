import numpy as np
from numpy.linalg import inv
from PyDiffGame.PyDiffGame import PyDiffGame

from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame


class MassesWithSprings(ContinuousPyDiffGame):
    def __init__(self,
                 m: float,
                 k: float,
                 q: float,
                 r: float,
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 epsilon: float = PyDiffGame._epsilon_default,
                 L: int = PyDiffGame._L_default,
                 regular_LQR: bool = False,
                 ):
        self.__m = m
        self.__k = k

        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [-2 * self.__k / self.__m, self.__k / self.__m, 0, 0],
                      [self.__k / self.__m, -2 * self.__k / self.__m, 0, 0]])

        B = 1 / self.__m * np.array([[0, 0],
                                     [0, 0],
                                     [1, 0],
                                     [0, 1]])

        M_1 = 1 / (2 * self.__m) * np.array([[1, 1]])
        M_2 = 1 / (2 * self.__m) * np.array([[1, -1]])
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

        Q_1 = np.diag([q, 0, q ** 2, q ** 2])
        Q_2 = np.diag([0, q, q ** 2, q ** 2])
        Q_lqr = np.diag([q, q, q ** 2, q ** 2])
        Q_multiplayer = [Q_1, Q_2]
        R_lqr = np.diag([r, r])
        R_multiplayer = [np.array([r])] * 2

        self.__origin = (0.0, 0.0)

        state_variables_names = ['x_1', 'x_2', '\\dot{x}_1', '\\dot{x}_2']

        super().__init__(A=A,
                         B=B if regular_LQR else B_multiplayer,
                         Q=Q_lqr if regular_LQR else Q_multiplayer,
                         R=R_lqr if regular_LQR else R_multiplayer,
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         state_variables_names=state_variables_names,
                         epsilon=epsilon,
                         L=L,
                         force_finite_horizon=T_f is not None)


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

lqr_masses = MassesWithSprings(m=m,
                               k=k,
                               q=q,
                               r=r,
                               x_0=x_0,
                               x_T=x_T,
                               regular_LQR=True,
                               epsilon=epsilon
                               )
lqr_masses.run_simulation(plot_state_space=True)
# lqr_masses.save_figure(filename='LQR')

game_masses = MassesWithSprings(m=m,
                                k=k,
                                q=q,
                                r=r,
                                x_0=x_0,
                                x_T=x_T,
                                epsilon=epsilon
                                )
game_masses.run_simulation(plot_state_space=True)
# game_masses.save_figure(filename='game')

# print(f'cost difference: {lqr_masses.get_costs() - game_masses.get_costs().max()}')
