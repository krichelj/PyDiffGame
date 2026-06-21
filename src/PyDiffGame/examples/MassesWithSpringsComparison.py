"""Coupled masses-and-springs benchmark: differential game vs. LQR.

``N`` unit-coupled masses connected by springs form a ``2N``-dimensional linear
system (positions and velocities).  The physical input space is decomposed along
the *modal* directions of ``M^{-1}K`` so that each vibration mode becomes one
player of a differential game, and the result is compared against a single
LQR controller acting on the whole system.

Run directly to solve a small instance and report the costs::

    python -m PyDiffGame.examples.MassesWithSpringsComparison
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from PyDiffGame.base import PyDiffGame
from PyDiffGame.comparison import PyDiffGameLQRComparison
from PyDiffGame.objective import GameObjective, LQRObjective
from PyDiffGame.plotting import show


class MassesWithSpringsComparison(PyDiffGameLQRComparison):
    """Compare a modal differential game with an LQR for ``N`` masses on springs."""

    def __init__(
        self,
        N: int,
        m: float,
        k: float,
        q: float | Sequence[float],
        r: float,
        x_0: np.ndarray | None = None,
        x_T: np.ndarray | None = None,
        T_f: float | None = None,
        epsilon_x: float = PyDiffGame.epsilon_x_default,
        epsilon_P: float = PyDiffGame.epsilon_P_default,
        L: int = PyDiffGame.L_default,
        eta: int = PyDiffGame.eta_default,
    ) -> None:
        I_N = np.eye(N)
        Z_N = np.zeros((N, N))

        mass_matrix = m * I_N
        stiffness = k * (2 * I_N - np.array([[int(abs(i - j) == 1) for j in range(N)] for i in range(N)]))
        mass_inv = np.linalg.inv(mass_matrix)
        mass_inv_stiffness = mass_inv @ stiffness

        # Modal directions: eigh gives orthonormal eigenvectors, so M is orthogonal.
        _, eigenvectors = np.linalg.eigh(mass_inv_stiffness)
        Ms = [eigenvectors[:, i].reshape(1, N) for i in range(N)]
        M = np.concatenate(Ms, axis=0)
        assert np.allclose(np.linalg.inv(M), M.T, atol=1e-9), "modal transform must be orthogonal"

        A = np.block([[Z_N, I_N], [-mass_inv_stiffness, Z_N]])
        B = np.block([[Z_N], [mass_inv]])

        q_per_mode = [q] * N if isinstance(q, (int, float)) else list(q)
        modal_to_state = np.kron(np.eye(2), M)
        game_objectives = []
        for i, (q_i, M_i) in enumerate(zip(q_per_mode, Ms)):
            modal_weight = np.diag([0.0] * i + [q_i] + [0.0] * (N - 1) + [q_i] + [0.0] * (N - i - 1))
            Q_i = modal_to_state.T @ modal_weight @ modal_to_state
            game_objectives.append(GameObjective(Q=Q_i, R=r, M=M_i))

        Q_lqr = (q if isinstance(q, (int, float)) else 1.0) * np.eye(2 * N)
        if not isinstance(q, (int, float)):
            Q_lqr = np.diag(list(q) + list(q))
        lqr_objective = [LQRObjective(Q=Q_lqr, R=0.25 * r * I_N)]

        state_variables_names = [f"x_{{{i}}}" for i in range(1, N + 1)] + [
            rf"\dot{{x}}_{{{i}}}" for i in range(1, N + 1)
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


def main(N: int = 2, *, plot: bool = True, save_figure: bool = False) -> None:
    """Solve a single masses-and-springs comparison and print the costs."""

    m, k, r = 50.0, 10.0, 1.0
    q = [500.0, 2000.0] if N == 2 else [500.0 * 1.2**i for i in range(N)]
    x_0 = np.array([10.0 * i for i in range(1, N + 1)] + [0.0] * N)
    x_T = x_0 * 10.0

    comparison = MassesWithSpringsComparison(N=N, m=m, k=k, q=q, r=r, x_0=x_0, x_T=x_T, T_f=25.0, L=300)
    comparison.run(plot_state_spaces=plot, save_figure=save_figure)

    lqr_cost, game_cost = comparison.costs()
    print(f"LQR cost   = {lqr_cost:.4g}")
    print(f"Game cost  = {game_cost:.4g}")
    print(f"All games controllable: {comparison.are_all_controllable()}")
    if plot:
        show()


if __name__ == "__main__":
    main()
