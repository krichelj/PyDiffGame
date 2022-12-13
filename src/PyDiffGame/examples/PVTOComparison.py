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
        g = 9.81  # gravitational constant
        c = 0.05  # damping factor (estimated)

        A = np.array(
            [[0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, -g / m, -c / m, 0, 0],
             [0, 0, 0, 0, -c / m, 0],
             [0, 0, 0, 0, 0, 0]]
        )

        # Input matrix
        B = np.array(
            [[0, 0],
             [0, 0],
             [0, 0],
             [1 / m, 0],
             [0, 1 / m],
             [r / J, 0]]
        )

        I = np.eye(2)

        M = I
        Ms = [M[0, :].reshape(1, M.shape[1]), M[1, :].reshape(1, M.shape[1])]

        q_x = 10
        q_y = 10
        q_theta = 10

        r_x = 1
        r_theta = 1

        Qs = [np.diag([0, q_y, 0] + [0] * 3), np.diag([q_x, 0, q_theta] + [0] * 3)]
        Rs = [np.array([r_x]), np.array([r_theta])]

        R_lqr = r * I
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
    epsilon_x, epsilon_P = 10 ** (-4), 10 ** (-4)

    x_0 = np.zeros((6,))
    x_d = 10
    y_d = 50
    x_T = np.array([x_d, y_d] + [0] * 4)

    pvtol = PVTOLComparison(x_0=x_0,
                            x_T=x_T,
                            epsilon_x=epsilon_x,
                            epsilon_P=epsilon_P)
    pvtol(plot_state_spaces=True,
          save_figure=True)
