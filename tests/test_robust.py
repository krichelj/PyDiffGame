"""Tests for the H-infinity (robust) differential-game state feedback."""

from __future__ import annotations

import control as ct
import numpy as np
import pytest

from PyDiffGame import ContinuousHInfinityControl


@pytest.fixture
def cart():
    """Double-integrator cart with a force disturbance."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    B_w = np.array([[0.0], [1.0]])
    Q = np.diag([1.0, 0.0])
    R = np.array([[1.0]])
    return A, B, B_w, Q, R


def test_gare_residual_is_tiny(cart):
    A, B, B_w, Q, R = cart
    h = ContinuousHInfinityControl(A, B, B_w, Q, R, gamma_margin=1.3).solve()
    assert h.gare_residual() < 1e-8
    assert h.is_closed_loop_stable()


def test_more_robust_than_lqr(cart):
    """The H-infinity game must reduce the worst-case L2 gain below the LQR's."""
    A, B, B_w, Q, R = cart
    h = ContinuousHInfinityControl(A, B, B_w, Q, R, gamma_margin=1.3).solve()
    g_hinf, _ = h.worst_case_gain()

    # LQR closed-loop worst-case gain on the same weighted output, same metric.
    K_lqr, _, _ = ct.lqr(A, B, Q, R)
    A_cl = A - B @ np.asarray(K_lqr)
    C = np.vstack([np.sqrt(Q), -np.sqrt(R) @ np.asarray(K_lqr)])
    eye = np.eye(A.shape[0])
    omegas = np.concatenate([[0.0], np.logspace(-3, 3, 4000)])
    g_lqr = max(
        float(np.linalg.svd(C @ np.linalg.solve(1j * w * eye - A_cl, B_w), compute_uv=False)[0])
        for w in omegas
    )
    assert g_hinf < g_lqr  # strictly more robust on the formal metric


def test_gamma_limit_approaches_optimal(cart):
    """As gamma -> gamma*, the closed-loop H-infinity norm approaches gamma*."""
    A, B, B_w, Q, R = cart
    base = ContinuousHInfinityControl(A, B, B_w, Q, R)
    gstar = base.optimal_gamma()
    near = ContinuousHInfinityControl(A, B, B_w, Q, R, gamma=gstar * 1.02).solve()
    g_norm, _ = near.worst_case_gain()
    # the achieved worst-case gain cannot beat gamma*, and should be close to it
    assert g_norm >= gstar - 1e-3
    assert g_norm <= gstar * 1.5


def test_infeasible_gamma_raises(cart):
    A, B, B_w, Q, R = cart
    gstar = ContinuousHInfinityControl(A, B, B_w, Q, R).optimal_gamma()
    with pytest.raises(ValueError):
        ContinuousHInfinityControl(A, B, B_w, Q, R, gamma=gstar * 0.5).solve()


def test_gamma_star_admissibility_is_monotone():
    """gamma* must sit on a clean admissible/inadmissible boundary (no boundary jitter).

    Regression for the Hamiltonian imaginary-axis instability: just above gamma*
    must be admissible and just below must not, with the validated predicate.
    """
    A = np.array([[0.0, 1.0], [-1.0, -0.2]])  # lightly damped mass-spring-damper
    B = np.array([[0.0], [1.0]])
    B_w = np.array([[0.0], [1.0]])
    Q = np.diag([1.0, 0.0])
    R = np.array([[0.1]])
    design = ContinuousHInfinityControl(A, B, B_w, Q, R)
    gstar = design.optimal_gamma()
    assert design._is_admissible(gstar * 1.05)
    assert not design._is_admissible(gstar * 0.95)


def test_gamma_star_does_not_floor_for_small_gamma():
    """A heavily-damped, weakly-disturbed plant has a tiny gamma* that must not floor."""
    A = np.array([[-5.0, 1.0], [0.0, -6.0]])  # well-damped, stable
    B = np.array([[0.0], [1.0]])
    B_w = np.array([[1e-3], [0.0]])  # very weak disturbance -> tiny gamma*
    Q = np.diag([1.0, 0.0])
    R = np.array([[1.0]])
    gstar = ContinuousHInfinityControl(A, B, B_w, Q, R).optimal_gamma()
    assert gstar < 1e-3  # must report the true tiny optimum, not a 1e-3 floor


def test_unstable_plant_is_stabilised(cart):
    """An open-loop-unstable plant is stabilised robustly (inverted-pendulum-like)."""
    A = np.array([[0.0, 1.0], [5.0, 0.0]])  # unstable (eigenvalues +/- sqrt(5))
    B = np.array([[0.0], [1.0]])
    B_w = np.array([[0.0], [1.0]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    h = ContinuousHInfinityControl(A, B, B_w, Q, R, gamma_margin=1.5).solve()
    assert h.is_closed_loop_stable()
    assert h.gare_residual() < 1e-7
