"""PyDiffGame — Nash-equilibrium solutions to linear-quadratic differential games.

PyDiffGame solves multi-objective dynamical control problems by reducing the
Game Hamilton–Jacobi–Bellman equations to coupled algebraic / differential
Riccati equations.

Quick start
-----------
>>> import numpy as np
>>> from PyDiffGame import ContinuousLQR
>>> A = np.array([[0.0, 1.0], [0.0, 0.0]])
>>> B = np.array([[0.0], [1.0]])
>>> lqr = ContinuousLQR(A=A, B=B, Q=np.eye(2), R=1.0).solve()
>>> lqr.is_closed_loop_stable()
True

The public API is intentionally small:

* :class:`~PyDiffGame.objective.Objective` (+ :func:`LQRObjective`,
  :func:`GameObjective` helpers) — describe each player's cost.
* :class:`~PyDiffGame.continuous.ContinuousPyDiffGame` /
  :class:`~PyDiffGame.discrete.DiscretePyDiffGame` — the solvers.
* :class:`~PyDiffGame.lqr.ContinuousLQR` / :class:`~PyDiffGame.lqr.DiscreteLQR`
  — single-objective convenience wrappers.
* :class:`~PyDiffGame.comparison.PyDiffGameLQRComparison` — compare designs.
"""

from __future__ import annotations

from PyDiffGame.base import PyDiffGame
from PyDiffGame.comparison import PyDiffGameLQRComparison
from PyDiffGame.continuous import ContinuousPyDiffGame
from PyDiffGame.discrete import DiscretePyDiffGame
from PyDiffGame.lqr import ContinuousLQR, DiscreteLQR
from PyDiffGame.objective import GameObjective, LQRObjective, Objective

__version__ = "2.0.0"

__all__ = [
    "PyDiffGame",
    "ContinuousPyDiffGame",
    "DiscretePyDiffGame",
    "ContinuousLQR",
    "DiscreteLQR",
    "Objective",
    "LQRObjective",
    "GameObjective",
    "PyDiffGameLQRComparison",
    "__version__",
]
