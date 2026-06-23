"""Masses-on-springs system: PyDiffGame modal game vs centralized LQR.

N identical masses in a line, each connected to its neighbours and the walls by
springs. The physical input space is decomposed along the modal directions of
M^{-1}K, so each vibration mode becomes one player of a differential game.
"""

from __future__ import annotations

import numpy as np

from benchmarks.harness import ControlProblem
from PyDiffGame.objective import GameObjective


def build(N: int = 2, m: float = 50.0, k: float = 10.0, r: float = 1.0, T_f: float = 25.0):
    """Return ``(problem, game_objectives)`` for the masses benchmark."""
    q = [500.0, 2000.0] if N == 2 else [500.0 * (1.2**i) for i in range(N)]

    I_N, Z_N = np.eye(N), np.zeros((N, N))
    stiffness = k * (2 * I_N - np.eye(N, k=1) - np.eye(N, k=-1))
    mass_inv_stiffness = (1.0 / m) * stiffness

    # Modal decomposition (orthonormal eigenvectors -> orthogonal modal transform).
    _, eigenvectors = np.linalg.eigh(mass_inv_stiffness)
    Ms = [eigenvectors[:, i].reshape(1, N) for i in range(N)]
    modal_to_state = np.kron(np.eye(2), np.concatenate(Ms, axis=0))

    A = np.block([[Z_N, I_N], [-mass_inv_stiffness, Z_N]])
    B = np.block([[Z_N], [(1.0 / m) * I_N]])

    game_objectives = []
    for i, (q_i, M_i) in enumerate(zip(q, Ms)):
        modal_weight = np.diag([0.0] * i + [q_i] + [0.0] * (N - 1) + [q_i] + [0.0] * (N - i - 1))
        Q_i = modal_to_state.T @ modal_weight @ modal_to_state
        game_objectives.append(GameObjective(Q=Q_i, R=r, M=M_i))

    # Shared yardstick = aggregate of the game's objectives (the genuine monolithic optimum).
    Q_shared = modal_to_state.T @ np.diag(q + q) @ modal_to_state
    R_shared = r * np.eye(N)

    x_0 = (
        np.array([10.0, 20.0, 0.0, 0.0])
        if N == 2
        else np.concatenate([10.0 * (np.arange(N) + 1), np.zeros(N)])
    )
    x_T = x_0 * 10.0

    state_names = tuple([f"x{i + 1}" for i in range(N)] + [f"v{i + 1}" for i in range(N)])
    problem = ControlProblem(
        name=f"{N} masses on springs",
        A=A,
        B=B,
        Q=Q_shared,
        R=R_shared,
        x_0=x_0,
        x_T=x_T,
        T_f=T_f,
        tracked=tuple(range(N)),
        state_names=state_names,
    )
    return problem, game_objectives
