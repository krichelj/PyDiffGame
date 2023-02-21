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

        r_x = 1 / m
        r_y = 1
        r_theta = r / J

        M_x = np.array([r_x, 0])
        M_y = np.array([0, r_y])
        M_theta = np.array([r_theta, 0])

        Ms = [M_x.reshape(1, 2),
              M_y.reshape(1, 2),
              M_theta.reshape(1, 2)]

        q_x = 100
        q_y = 10
        q_theta = 2 * np.pi / 5


        Q_x = np.diag([q_x, 0, 0] * 2)
        Q_y = np.diag([0, q_y, 0] * 2)
        Q_theta = np.diag([0, 0, q_theta] * 2)

        Qs = [Q_x, Q_y, Q_theta]
        Rs = [np.array([r_x]), np.array([r_y]), np.array([r_theta])]

        variables = ['x', 'y', '\\theta']

        self.state_variables_names = [f'{v}(t)' for v in variables] + \
                                     ['\\dot{' + str(v) + '}(t)' for v in variables]

        Q_lqr = np.diag([q_x, q_y, q_theta] * 2)
        R_lqr = np.diag([r_x + r_theta, r_y])
        assert np.all(np.abs(Q_lqr - (Q_x + Q_y + Q_theta)) < epsilon_x)

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
                'state_variables_names': self.state_variables_names,
                'epsilon_x': epsilon_x,
                'epsilon_P': epsilon_P,
                'L': L,
                'eta': eta}

        super().__init__(args=args,
                         M=M,
                         games_objectives=games_objectives,
                         continuous=True)


if __name__ == '__main__':
    epsilon_x, epsilon_P = 10 ** (-8), 10 ** (-8)

    x_0 = np.zeros((6,))
    x_d = 20
    y_d = 50
    x_T = np.array([x_d, y_d] + [0] * 4)

    pvtol = PVTOLComparison(x_0=x_0,
                            x_T=x_T,
                            # T_f=T_f,
                            epsilon_x=epsilon_x,
                            epsilon_P=epsilon_P)
    pvtol(plot_state_spaces=True,
          save_figure=True)
