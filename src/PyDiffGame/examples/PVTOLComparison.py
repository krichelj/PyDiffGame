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

        A_11 = np.zeros((3, 3))
        A_12 = np.eye(3)
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

        psi = 1

        M_x = [1, 0]
        M_y = [0, 1]
        M_theta = [1, 0]

        Ms = [np.array(M_i).reshape(1, 2) for M_i in [M_x, M_y, M_theta]]
        M = np.concatenate(Ms, axis=0)

        self.figure_filename_generator = lambda g: ('LQR_' if g.is_LQR() else '') + f"PVTOL{str(psi).replace('.', '')}"

        Q_x_diag = [psi, 0, 0]
        Q_y_diag = [0, 1, 0]
        Q_theta_diag = [0, 0, 1]

        r_x = 1
        r_y = 1
        r_theta = 1

        Qs = [np.diag(Q_i_diag * 2) for Q_i_diag in [Q_x_diag, Q_y_diag, Q_theta_diag]]
        Rs = [np.array([r_i]) for r_i in [r_x, r_y, r_theta]]

        variables = ['x', 'y', '\\theta']

        self.state_variables_names = [f'{v}(t)' for v in variables] + ['\\dot{' + str(v) + '}(t)' for v in variables]

        Q_lqr = sum(Qs)
        R_lqr = np.diag([r_x + r_theta, r_y])

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
                         games_objectives=games_objectives)


if __name__ == '__main__':
    epsilon_x = epsilon_P = 10 ** (-8)

    x_0 = np.zeros((6,))
    x_d = y_d = 50
    x_T = np.array([x_d, y_d] + [0] * 4)
    T_f = 25

    pvtol = PVTOLComparison(x_0=x_0,
                            x_T=x_T,
                            T_f=T_f,
                            epsilon_x=epsilon_x,
                            epsilon_P=epsilon_P)
    pvtol(plot_state_spaces=True,
          save_figure=True,
          figure_filename=pvtol.figure_filename_generator)
