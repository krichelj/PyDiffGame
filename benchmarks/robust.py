"""Benchmark adapters over PyDiffGame's robust H-infinity design.

Thin wrappers so the benchmark scripts exercise the *package* feature
(:class:`PyDiffGame.ContinuousHInfinityControl` and
:func:`PyDiffGame.robust.worst_case_l2_gain`) rather than a private copy.
"""

from __future__ import annotations

import numpy as np

from PyDiffGame import ContinuousHInfinityControl
from PyDiffGame.robust import _symmetric_sqrt as _sqrtm_psd  # noqa: F401 (re-exported for plots)
from PyDiffGame.robust import worst_case_l2_gain

FloatArray = np.ndarray


def h_infinity_gain(A, B, Q, R, B_w, gamma):
    """Solve PyDiffGame's H-infinity disturbance game at ``gamma``; return ``(P, K)``."""
    design = ContinuousHInfinityControl(A, B, B_w, Q, R, gamma=gamma).solve()
    return design.P, design.K


def min_gamma(A, B, Q, R, B_w, **_ignored) -> float:
    """Optimal ``gamma*`` of PyDiffGame's H-infinity disturbance game."""
    return ContinuousHInfinityControl(A, B, B_w, Q, R).optimal_gamma()


def closed_loop_hinf_norm(A, B, K, Q, R, B_w, *, n_grid: int = 2500):
    """Worst-case L2 gain ||G_zw||inf of an arbitrary state-feedback gain ``K``.

    ``n_grid=2500`` is dense enough to catch the lightly-damped resonant peaks in
    this catalogue (verified against the 6000-point default) while keeping the
    suite fast.
    """
    return worst_case_l2_gain(A, B, K, Q, R, B_w, n_grid=n_grid)
