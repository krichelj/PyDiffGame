"""Multi-player game correctness via coupled-Riccati residuals and stability."""

import numpy as np
import pytest

from PyDiffGame import ContinuousPyDiffGame, GameObjective, LQRObjective


@pytest.fixture
def two_player_game():
    A = np.array(
        [[0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 0.0]]
    )
    B = np.eye(4)[:, [1, 3]]
    objectives = [
        GameObjective(Q=np.diag([1.0, 0.1, 0.0, 0.0]), R=1.0, M=np.array([[1.0, 0.0]])),
        GameObjective(Q=np.diag([0.0, 0.0, 1.0, 0.1]), R=1.0, M=np.array([[0.0, 1.0]])),
    ]
    return ContinuousPyDiffGame(A=A, objectives=objectives, B=B)


def test_coupled_are_residuals_near_zero(two_player_game):
    game = two_player_game.solve()
    residuals = game.algebraic_riccati_residuals()
    assert len(residuals) == 2
    for residual in residuals:
        assert np.max(np.abs(residual)) < 1e-6


def test_game_closed_loop_is_stable(two_player_game):
    game = two_player_game.solve()
    assert game.is_closed_loop_stable()


def test_single_player_game_equals_lqr():
    """A one-objective game must reduce to the standard LQR solution."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    game = ContinuousPyDiffGame(A=A, objectives=[LQRObjective(Q=np.eye(2), R=1.0)], B=B).solve()
    from scipy.linalg import solve_continuous_are

    np.testing.assert_allclose(game.P[0], solve_continuous_are(A, B, np.eye(2), [[1.0]]), atol=1e-8)


def test_len_and_indexing(two_player_game):
    assert len(two_player_game) == 2
    assert two_player_game[0].is_lqr is False
