"""Planar vertical take-off and landing (PVTOL) aircraft: differential game vs. LQR.

The PVTOL is a classic 6-state planar aircraft model with horizontal position
``x``, vertical position ``y`` and pitch angle ``theta`` (plus their velocities).
The two physical inputs are the thrust and the rolling moment.  The physical
input space is decomposed along the ``x``, ``y`` and ``theta`` channels so that
each becomes one player of a differential game, and the result is compared
against a single LQR controller acting on the whole system.

Run directly to solve a small instance and report the costs::

    python -m PyDiffGame.examples.PVTOLComparison
"""

from __future__ import annotations

import numpy as np

from PyDiffGame.base import PyDiffGame
from PyDiffGame.comparison import PyDiffGameLQRComparison
from PyDiffGame.objective import GameObjective, LQRObjective
from PyDiffGame.plotting import show


class PVTOLComparison(PyDiffGameLQRComparison):
    """Compare a per-channel differential game with an LQR for the PVTOL aircraft."""

    def __init__(
        self,
        x_0: np.ndarray | None = None,
        x_T: np.ndarray | None = None,
        T_f: float | None = None,
        epsilon_x: float = PyDiffGame.epsilon_x_default,
        epsilon_P: float = PyDiffGame.epsilon_P_default,
        L: int = PyDiffGame.L_default,
        eta: int = PyDiffGame.eta_default,
    ) -> None:
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

        Q_x_diag = [psi, 0, 0]
        Q_y_diag = [0, 1, 0]
        Q_theta_diag = [0, 0, 1]

        r_x = 1
        r_y = 1
        r_theta = 1

        Qs = [np.diag(Q_i_diag * 2) for Q_i_diag in [Q_x_diag, Q_y_diag, Q_theta_diag]]
        Rs = [r_x, r_y, r_theta]

        variables = ["x", "y", r"\theta"]
        state_variables_names = [rf"{v}(t)" for v in variables] + [
            rf"\dot{{{v}}}(t)" for v in variables
        ]

        Q_lqr = sum(Qs)
        R_lqr = np.diag([r_x + r_theta, r_y])

        lqr_objective = [LQRObjective(Q=Q_lqr, R=R_lqr)]
        game_objectives = [
            GameObjective(Q=Q_i, R=R_i, M=M_i)
            for Q_i, R_i, M_i in zip(Qs, Rs, Ms)
        ]

        super().__init__(
            A=A,
            B=B,
            games_objectives=[lqr_objective, game_objectives],
            continuous=True,
            x_0=x_0,
            x_T=x_T,
            T_f=T_f,
            L=L,
            eta=eta,
            epsilon_x=epsilon_x,
            epsilon_P=epsilon_P,
            state_variables_names=state_variables_names,
        )


def main(*, plot: bool = True, save_figure: bool = False) -> None:
    """Solve a single PVTOL comparison and print the costs."""

    x_0 = np.zeros((6,))
    x_d = y_d = 50
    x_T = np.array([x_d, y_d] + [0] * 4, dtype=np.float64)
    T_f = 25.0

    comparison = PVTOLComparison(x_0=x_0, x_T=x_T, T_f=T_f, L=300)
    comparison.run(plot_state_spaces=plot, save_figure=save_figure)

    lqr_cost, game_cost = comparison.costs()
    print(f"LQR cost   = {lqr_cost:.4g}")
    print(f"Game cost  = {game_cost:.4g}")
    print(f"All games controllable: {comparison.are_all_controllable()}")
    if plot:
        show()


if __name__ == "__main__":
    main()
