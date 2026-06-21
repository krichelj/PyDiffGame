"""LQR correctness: the solver must match scipy's algebraic Riccati solutions."""

import numpy as np
import pytest
from scipy.linalg import solve_continuous_are, solve_discrete_are

from PyDiffGame import ContinuousLQR, DiscreteLQR


@pytest.fixture
def continuous_system():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.diag([2.0, 1.0])
    R = np.array([[0.5]])
    return A, B, Q, R


def test_continuous_lqr_matches_scipy_care(continuous_system):
    A, B, Q, R = continuous_system
    lqr = ContinuousLQR(A=A, B=B, Q=Q, R=R).solve()
    np.testing.assert_allclose(lqr.P[0], solve_continuous_are(A, B, Q, R), atol=1e-8)


def test_continuous_lqr_gain_and_stability(continuous_system):
    A, B, Q, R = continuous_system
    lqr = ContinuousLQR(A=A, B=B, Q=Q, R=R).solve()
    expected_K = np.linalg.solve(R, B.T) @ lqr.P[0]
    np.testing.assert_allclose(lqr.K[0], expected_K, atol=1e-8)
    assert lqr.is_closed_loop_stable()


def test_discrete_lqr_matches_scipy_dare():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    lqr = DiscreteLQR(A=A, B=B, Q=Q, R=R, is_input_discrete=True).solve()
    np.testing.assert_allclose(lqr.P[0], solve_discrete_are(A, B, Q, R), atol=1e-8)
    assert lqr.is_closed_loop_stable()


def test_riccati_residual_is_zero(continuous_system):
    A, B, Q, R = continuous_system
    lqr = ContinuousLQR(A=A, B=B, Q=Q, R=R).solve()
    P = lqr.P[0]
    S = B @ np.linalg.solve(R, B.T)
    residual = A.T @ P + P @ A - P @ S @ P + Q
    assert np.max(np.abs(residual)) < 1e-8


def test_finite_horizon_dre_converges_to_care_independently(continuous_system):
    """Independent of scipy's algebraic solver.

    The infinite-horizon LQR path delegates to ``solve_continuous_are`` (so it is
    bit-for-bit identical, hence a 0.0 difference). Here we instead integrate the
    *differential* Riccati equation backward from a non-equilibrium terminal
    condition (P_f = 0) over a long horizon, using only our own ODE code, and
    require that P(t=0) converges to scipy's algebraic solution.
    """
    A, B, Q, R = continuous_system
    game = ContinuousLQR(A=A, B=B, Q=Q, R=R, T_f=30.0, L=600, P_f=[np.zeros_like(A)]).solve()
    P_at_t0 = np.asarray(game.P)[0, 0]  # finite horizon storage: (L, N, n, n)
    care = solve_continuous_are(A, B, Q, R)
    error = np.max(np.abs(P_at_t0 - care))
    assert 0 < error < 1e-6, f"expected small nonzero convergence error, got {error:.2e}"
