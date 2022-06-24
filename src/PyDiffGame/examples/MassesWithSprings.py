import numpy as np
from PyDiffGame.PyDiffGame import PyDiffGame

from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame


class MassesWithSprings(ContinuousPyDiffGame):
    __q_s_default = 1
    __q_m_default = 100
    __q_l_default = 10000
    __r_default = 1

    def __init__(self,
                 m: float,
                 k: float,
                 q_s: float = __q_s_default,
                 q_m: float = __q_m_default,
                 q_l: float = __q_l_default,
                 r: float = __r_default,
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 epsilon: float = PyDiffGame._epsilon_default,
                 L: int = PyDiffGame._L_default,
                 regular_LQR: bool = False,
                 show_animation: bool = True
                 ):
        self.__m = m
        self.__k = k

        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [-2 * self.__k / self.__m, 1 * self.__k / self.__m, 0, 0],
                      [-1 * self.__k / self.__m, 2 * self.__k / self.__m, 0, 0]])

        B_lqr = 1 / self.__k * np.array([[0, 0],
                                         [0, 0],
                                         [1, 0],
                                         [0, 1]])

        Q_x = np.diag([q_l, q_s, q_m, q_s])
        Q_theta = np.diag([q_s, q_l, q_s, q_m])
        Q_lqr = (Q_x + Q_theta) / 2

        R_lqr = np.diag([r, r])

        self.__origin = (0.0, 0.0)
        self.__show_animation = show_animation

        state_variables_names = ['x_1', 'x_2', '\\dot{x}_1', '\\dot{x}_2']

        super().__init__(A=A,
                         B=B_lqr,
                         Q=Q_lqr,
                         R=R_lqr,
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         state_variables_names=state_variables_names,
                         epsilon=epsilon,
                         L=L,
                         force_finite_horizon=T_f is not None
                         )


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
sys = MassesWithSprings(m=1,
                        k=1,
                        r=100000,
                        x_0=x_0,
                        x_T=x_T,
                        regular_LQR=True,
                        show_animation=False,
                        epsilon=epsilon
                        )
sys.solve_game_simulate_state_space_and_plot()
