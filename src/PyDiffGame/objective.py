"""Control objectives for a differential game.

A differential game is specified by a collection of :class:`Objective` instances,
one per *player* (a.k.a. virtual control objective).  Each objective carries its
own state-weight matrix :math:`Q_i`, input-weight matrix :math:`R_{ii}` and,
optionally, a *decomposition* matrix :math:`M_i` that maps the shared physical
input space onto the player's virtual input space.

A plain Linear-Quadratic Regulator (LQR) is simply a game with a single
objective and no decomposition matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from PyDiffGame._typing import ArrayLike, FloatArray


def _as_matrix(value: ArrayLike) -> FloatArray:
    """Coerce ``value`` into a 2-D float array.

    Scalars and 1-D inputs are promoted so that an input weight given as a bare
    number (a common shorthand for single-input players) becomes a ``1 x 1``
    matrix.
    """

    array = np.atleast_2d(np.asarray(value, dtype=np.float64))
    return array


@dataclass(frozen=True, slots=True)
class Objective:
    """A single player's quadratic cost objective.

    Parameters
    ----------
    Q:
        State-weight matrix of shape ``(n, n)``.  Must be symmetric positive
        semi-definite.
    R:
        Input-weight matrix of shape ``(m_i, m_i)`` (or a scalar for a
        single-input player).  Must be symmetric positive definite.
    M:
        Optional decomposition matrix of shape ``(m_i, m)`` mapping the physical
        input onto this player's virtual input.  ``None`` denotes a plain LQR
        objective acting directly on the full input.
    """

    Q: FloatArray
    R: FloatArray
    M: FloatArray | None = None
    #: Whether this objective participates as an LQR (no decomposition).
    is_lqr: bool = field(init=False)

    def __post_init__(self) -> None:
        # ``frozen=True`` forbids normal attribute assignment, so we go through
        # ``object.__setattr__`` to normalise the matrices once at construction.
        object.__setattr__(self, "Q", _as_matrix(self.Q))
        object.__setattr__(self, "R", _as_matrix(self.R))
        if self.M is not None:
            object.__setattr__(self, "M", _as_matrix(self.M))
        object.__setattr__(self, "is_lqr", self.M is None)
        self._validate()

    def _validate(self) -> None:
        n = self.Q.shape[0]
        if self.Q.shape != (n, n):
            raise ValueError(f"Q must be square, got shape {self.Q.shape}")
        if not np.allclose(self.Q, self.Q.T):
            raise ValueError("Q must be symmetric")
        q_eigvals = np.linalg.eigvalsh(self.Q)
        if np.any(q_eigvals < -1e-7):
            raise ValueError("Q must be positive semi-definite")

        m_i = self.R.shape[0]
        if self.R.shape != (m_i, m_i):
            raise ValueError(f"R must be square, got shape {self.R.shape}")
        if not np.allclose(self.R, self.R.T):
            raise ValueError("R must be symmetric")
        if np.any(np.linalg.eigvalsh(self.R) <= 0):
            raise ValueError("R must be positive definite")

        if self.M is not None and self.M.shape[0] != m_i:
            raise ValueError(f"M must have m_i = {m_i} rows to match R, got {self.M.shape[0]}")

    @property
    def m_i(self) -> int:
        """Number of (virtual) inputs controlled by this objective."""

        return self.R.shape[0]


def LQRObjective(Q: ArrayLike, R: ArrayLike) -> Objective:
    """Convenience constructor for a plain LQR objective (no decomposition)."""

    return Objective(Q=_as_matrix(Q), R=_as_matrix(R), M=None)


def GameObjective(Q: ArrayLike, R: ArrayLike, M: ArrayLike) -> Objective:
    """Convenience constructor for a game player's objective with decomposition ``M``."""

    return Objective(Q=_as_matrix(Q), R=_as_matrix(R), M=_as_matrix(M))


__all__ = ["Objective", "LQRObjective", "GameObjective"]
