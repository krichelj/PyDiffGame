"""Tests for the :class:`~PyDiffGame.objective.Objective` data model."""

import numpy as np
import pytest

from PyDiffGame import GameObjective, LQRObjective, Objective


def test_scalar_R_is_promoted_to_matrix():
    obj = LQRObjective(Q=np.eye(2), R=2.0)
    assert obj.R.shape == (1, 1)
    assert obj.m_i == 1
    assert obj.is_lqr


def test_game_objective_carries_decomposition():
    obj = GameObjective(Q=np.eye(2), R=1.0, M=np.array([[1.0, 0.0]]))
    assert obj.M is not None
    assert not obj.is_lqr


def test_objective_is_frozen():
    obj = LQRObjective(Q=np.eye(2), R=1.0)
    with pytest.raises(Exception):
        obj.Q = np.zeros((2, 2))  # type: ignore[misc]


@pytest.mark.parametrize(
    "Q, R, match",
    [
        (np.array([[1.0, 2.0], [0.0, 1.0]]), 1.0, "symmetric"),
        (-np.eye(2), 1.0, "semi-definite"),
        (np.eye(2), -1.0, "positive definite"),
    ],
)
def test_invalid_weights_raise(Q, R, match):
    with pytest.raises(ValueError, match=match):
        Objective(Q=Q, R=R)


def test_M_row_mismatch_raises():
    with pytest.raises(ValueError, match="rows to match R"):
        GameObjective(Q=np.eye(2), R=np.eye(2), M=np.array([[1.0, 0.0]]))
