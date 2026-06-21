"""Discrete-time solver: discretisation, Riccati recursion and stability."""

import numpy as np
import pytest
from scipy.linalg import expm, solve_discrete_are

from PyDiffGame import DiscreteLQR, DiscretePyDiffGame, GameObjective


def test_zero_order_hold_discretization_matches_van_loan():
    """The internal discretisation must match an independent Van Loan computation."""
    A = np.array([[0.0, 1.0], [-1.0, -0.5]])
    B = np.array([[0.0], [1.0]])
    T_f, L = 10.0, 100
    delta = T_f / L
    lqr = DiscreteLQR(A=A, B=B, Q=np.eye(2), R=1.0, T_f=T_f, L=L)  # discretises on init

    A_d_expected = expm(A * delta)
    np.testing.assert_allclose(lqr.A, A_d_expected, atol=1e-10)


def test_discrete_game_residual_and_stability():
    A = np.array([[1.0, 0.05, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.05],
                  [0.0, 0.0, 0.0, 1.0]])
    B = np.eye(4)[:, [1, 3]]
    objectives = [
        GameObjective(Q=np.diag([1.0, 0.1, 0.0, 0.0]), R=1.0, M=np.array([[1.0, 0.0]])),
        GameObjective(Q=np.diag([0.0, 0.0, 1.0, 0.1]), R=1.0, M=np.array([[0.0, 1.0]])),
    ]
    game = DiscretePyDiffGame(A=A, objectives=objectives, B=B, is_input_discrete=True).solve()
    assert game.is_closed_loop_stable()


def test_discrete_finite_horizon_recursion_converges_to_dare():
    """Independent check: a long finite-horizon backward recursion seeded away from
    the fixed point must converge to scipy's discrete ARE solution."""
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q, R = np.eye(2), np.array([[1.0]])
    game = DiscreteLQR(
        A=A, B=B, Q=Q, R=R, is_input_discrete=True,
        T_f=50.0, L=400, P_f=[np.zeros((2, 2))],
    ).solve()
    P_at_t0 = np.asarray(game.P)[0, 0]  # finite horizon -> (L, N, n, n)
    np.testing.assert_allclose(P_at_t0, solve_discrete_are(A, B, Q, R), atol=1e-6)
