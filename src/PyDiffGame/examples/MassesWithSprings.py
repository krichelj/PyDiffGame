import numpy as np
from PyDiffGame.PyDiffGame import PyDiffGame

from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame


class MassesWithSprings(ContinuousPyDiffGame):
    __q_default = 100
    __r_default = 1

    def __init__(self,
                 m: float,
                 k: float,
                 q: float = __q_default,
                 r: float = __r_default,
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

        B_1 = np.array([[0],
                        [0],
                        [1 / 2],
                        [1 / 2]])
        B_2 = np.array([[0],
                        [0],
                        [1 / 2],
                        [-1 / 2]])

        B_lqr = 1 / self.__m * np.array([[0, 0],
                                         [0, 0],
                                         [1, 0],
                                         [0, 1]])
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
                         B=B_lqr if regular_LQR else B_multiplayer,
                         Q=Q_lqr if regular_LQR else Q_multiplayer,
                         R=R_lqr if regular_LQR else R_multiplayer,
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         state_variables_names=state_variables_names,
                         epsilon=epsilon,
                         L=L,
                         force_finite_horizon=T_f is not None
                         )


m, k, r = 10, 100, 1000

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
                               r=r,
                               x_0=x_0,
                               x_T=x_T,
                               regular_LQR=True,
                               epsilon=epsilon
                               )
lqr_masses.run_simulation()

game_masses = MassesWithSprings(m=m,
                                k=k,
                                r=r,
                                x_0=x_0,
                                x_T=x_T,
                                epsilon=epsilon
                                )
game_masses.run_simulation()
