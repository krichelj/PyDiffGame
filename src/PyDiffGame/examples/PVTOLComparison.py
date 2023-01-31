import numpy as np
from typing import Optional

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.PyDiffGameLQRComparison import PyDiffGameLQRComparison
from PyDiffGame.Objective import GameObjective, LQRObjective


class PVTOLComparison(PyDiffGameLQRComparison):
    def __init__(self,
                 x_0: Optional[np.array] = None,
                 x_T: Optional[np.array] = None,
                 T_f: Optional[float] = None,
                 epsilon_x: Optional[float] = PyDiffGame.epsilon_x_default,
                 epsilon_P: Optional[float] = PyDiffGame.epsilon_P_default,
                 L: Optional[int] = PyDiffGame.L_default,
                 eta: Optional[int] = PyDiffGame.eta_default):
        m = 4  # mass of aircraft
        J = 0.0475  # inertia around pitch axis
        r = 0.25  # distance to center of force
        c = 0.05  # damping factor (estimated)

        I_3 = np.eye(3)
        Z_3 = np.zeros((3, 3))

        A_11 = Z_3
        A_12 = I_3
        A_21 = np.array([[0, 0, -PyDiffGame.g],
                         [0, 0, 0],
                         [0, 0, 0]])
        A_22 = np.diag([-c / m, -c / m, 0])

        A = np.block([[A_11, A_12],
                      [A_21, A_22]])

        # Input matrix
        B = np.array(
            [[0, 0],
             [0, 0],
             [0, 0],
             [1 / m, 0],
             [0, 1 / m],
             [r / J, 0]]
        )

        I_2 = np.eye(2)

        M = I_2
        Ms = [M[0, :].reshape(1, M.shape[1]), M[1, :].reshape(1, M.shape[1])]

        # q_s = 1
        # q_l = q_s * 10000

        q_x = 10
        q_theta = 10
        q_y = 10000

        r_x_theta = 10
        r_y = 10000

        # Q_y = np.diag([q_s, q_l, q_s] * 2)
        # Q_x_theta = np.diag([q_l, q_s, q_l]*2)

        Q_x_theta = np.diag([q_x, 0, q_theta] + [0] * 3)
        Q_y = np.diag([0, q_y, 0] + [0] * 3)

        Qs = [Q_x_theta, Q_y]
        Rs = [np.array([r_x_theta]), np.array([r_y])]

        R_lqr = np.diag([r_y, r_x_theta])
        Q_lqr = np.diag([q_x, q_y, q_theta] + [0] * 3)

        variables = ['x', 'y', '\\theta']

        state_variables_names = [f'{v}(t)' for v in variables] + \
                                ['\\dot{' + str(v) + '}(t)' for v in variables]

        lqr_objective = [LQRObjective(Q=Q_lqr,
                                      R_ii=R_lqr)]
        game_objectives = [GameObjective(Q=Q_i,
                                         R_ii=R_i,
                                         M_i=M_i) for Q_i, R_i, M_i in zip(Qs, Rs, Ms)]
        games_objectives = [lqr_objective,
                            game_objectives]

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

        super().__init__(args=args,
                         M=M,
                         games_objectives=games_objectives,
                         continuous=True)


if __name__ == '__main__':
    epsilon_x, epsilon_P = 10 ** (-6), 10 ** (-6)

    x_0 = np.zeros((6,))
    x_d = 10
    y_d = 50
    x_T = np.array([x_d, y_d] + [0] * 4)
    T_f = 20

    pvtol = PVTOLComparison(x_0=x_0,
                            x_T=x_T,
                            T_f=T_f,
                            epsilon_x=epsilon_x,
                            epsilon_P=epsilon_P)
    pvtol(plot_state_spaces=True,
          save_figure=True
          )
