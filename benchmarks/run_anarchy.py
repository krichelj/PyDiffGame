"""Nominal regime, part 2: the honest *price of anarchy*.

Two carts coupled by a spring, each with its own actuator and its own objective
(player i wants cart i at the origin). The Nash equilibrium of this genuine
2-player game is a *decentralized* design; scored on the joint (sum) cost it can
only match or exceed the centralized LQR -- the price of anarchy. This is the
honest flip side of the lossless masses result: when objectives genuinely
compete, the game pays a small nominal price (and buys decentralization /
compositionality in return). The robustness boost is measured separately.
"""

from __future__ import annotations

import numpy as np

from benchmarks.harness import ControlProblem
from benchmarks.pdg_design import game_controller, lqr_controller
from benchmarks.runner import compare
from PyDiffGame.objective import GameObjective


def main() -> None:
    m, k, q, r = 1.0, 2.0, 100.0, 1.0
    A = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-k / m, k / m, 0.0, 0.0],
            [k / m, -k / m, 0.0, 0.0],
        ]
    )
    B = np.array([[0.0, 0.0], [0.0, 0.0], [1.0 / m, 0.0], [0.0, 1.0 / m]])  # one actuator per cart

    # Each player controls its own actuator and cares only about its own cart.
    objectives = [
        GameObjective(Q=np.diag([q, 0.0, 1.0, 0.0]), R=r, M=np.array([[1.0, 0.0]])),
        GameObjective(Q=np.diag([0.0, q, 0.0, 1.0]), R=r, M=np.array([[0.0, 1.0]])),
    ]
    # Centralized LQR optimizes the joint (sum) objective -- the shared yardstick.
    Q_shared = np.diag([q, q, 1.0, 1.0])
    R_shared = r * np.eye(2)

    problem = ControlProblem(
        name="coupled carts (price of anarchy)",
        A=A,
        B=B,
        Q=Q_shared,
        R=R_shared,
        x_0=np.array([1.0, -1.0, 0.0, 0.0]),  # opposite displacement -> spring engaged, coupling matters
        x_T=np.zeros(4),
        T_f=12.0,
        tracked=(0, 1),
        state_names=("x1", "x2", "v1", "v2"),
    )

    controllers = [lqr_controller(problem), game_controller(problem, objectives)]
    compare(problem, controllers, out_prefix="coupled_carts_anarchy")


if __name__ == "__main__":
    main()
