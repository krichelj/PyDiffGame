"""Adapters turning designs into harness ``Controller`` objects (aggregate gain)."""

from __future__ import annotations

import control as ct
import numpy as np

from benchmarks.harness import Controller, ControlProblem
from PyDiffGame import ContinuousPyDiffGame
from PyDiffGame.objective import GameObjective


def aggregate_gain(game: ContinuousPyDiffGame) -> np.ndarray:
    """Constant physical-input gain ``K`` (u = -K x) from a solved infinite-horizon game."""
    gains = game.K  # list of per-player gains for the infinite-horizon design
    if game.N == 1:
        return np.asarray(gains[0], dtype=float)
    stacked = np.concatenate([np.asarray(k, dtype=float) for k in gains], axis=0)
    m_inv = game.M_inv
    return m_inv @ stacked if m_inv is not None else stacked


def lqr_controller(
    problem: ControlProblem, *, name: str = "LQR (python-control)", color: str = "#d97706"
) -> Controller:
    """Centralized LQR baseline from python-control on the shared (Q, R) yardstick."""
    K, _, _ = ct.lqr(problem.A, problem.B, problem.Q, problem.R)
    return Controller(name=name, K=np.asarray(K, dtype=float), color=color, meta={"kind": "lqr"})


def game_controller(
    problem: ControlProblem,
    objectives: list[GameObjective],
    *,
    name: str = "PyDiffGame (Nash)",
    color: str = "#059669",
) -> Controller:
    """Build the infinite-horizon Nash game for ``objectives`` and return its aggregate gain."""
    game = ContinuousPyDiffGame(A=problem.A, objectives=objectives, B=problem.B).solve()
    K = aggregate_gain(game)
    residual = max(float(np.max(np.abs(r))) for r in game.algebraic_riccati_residuals())
    return Controller(
        name=name,
        K=K,
        color=color,
        meta={"kind": "game", "are_residual": residual, "stable": bool(game.is_closed_loop_stable())},
    )
