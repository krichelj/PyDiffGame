"""Convenience Linear-Quadratic Regulator (LQR) classes.

An LQR is a differential game with a single objective and no input
decomposition.  These thin wrappers offer the familiar ``(A, B, Q, R)``
signature on top of the general game solvers.
"""

from __future__ import annotations

from PyDiffGame._typing import ArrayLike
from PyDiffGame.continuous import ContinuousPyDiffGame
from PyDiffGame.discrete import DiscretePyDiffGame
from PyDiffGame.objective import LQRObjective


class ContinuousLQR(ContinuousPyDiffGame):
    """Continuous-time LQR: ``dx/dt = A x + B u``."""

    def __init__(self, A: ArrayLike, B: ArrayLike, Q: ArrayLike, R: ArrayLike, **kwargs) -> None:
        super().__init__(A, [LQRObjective(Q=Q, R=R)], B=B, **kwargs)


class DiscreteLQR(DiscretePyDiffGame):
    """Discrete-time LQR: ``x[k+1] = A x[k] + B u[k]``."""

    def __init__(self, A: ArrayLike, B: ArrayLike, Q: ArrayLike, R: ArrayLike, **kwargs) -> None:
        super().__init__(A, [LQRObjective(Q=Q, R=R)], B=B, **kwargs)


__all__ = ["ContinuousLQR", "DiscreteLQR"]
