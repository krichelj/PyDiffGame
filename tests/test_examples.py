"""Smoke and regression tests for the bundled example simulations.

These guard the physics-level bugs that unit tests on the solver core cannot
see: that each example actually runs and produces finite costs, that the
inverted-pendulum nonlinear model is consistent with its linear design, and
that the masses comparison is scored fairly (the decomposed game matches the
monolithic LQR optimum).
"""

from __future__ import annotations

import numpy as np
import pytest

from PyDiffGame.examples.InvertedPendulumComparison import InvertedPendulumComparison
from PyDiffGame.examples.InvertedPendulumComparison import main as pendulum_main
from PyDiffGame.examples.MassesWithSpringsComparison import MassesWithSpringsComparison
from PyDiffGame.examples.MassesWithSpringsComparison import main as masses_main
from PyDiffGame.examples.PVTOLComparison import main as pvtol_main


@pytest.mark.parametrize("main", [masses_main, pendulum_main, pvtol_main])
def test_example_main_runs(main) -> None:
    """Each example's ``main`` solves, simulates and reports without error."""
    main(plot=False)


def test_quadrotor_and_legacy_modules_import() -> None:
    """The heavier scaffolding modules at least import cleanly."""
    import PyDiffGame.examples.PVTOL  # noqa: F401
    import PyDiffGame.examples.QuadRotorControl  # noqa: F401


def test_masses_game_matches_monolithic_lqr() -> None:
    """On the shared aggregate objective the modal game recovers the LQR optimum."""
    x_0 = np.array([10.0, 20.0, 0.0, 0.0])
    comparison = MassesWithSpringsComparison(
        N=2, m=50.0, k=10.0, q=[500.0, 2000.0], r=1.0, x_0=x_0, x_T=x_0 * 10.0, T_f=25.0, L=300
    )
    comparison.run(plot_state_spaces=False)
    lqr_cost, game_cost = comparison.costs()
    assert np.isfinite(lqr_cost) and np.isfinite(game_cost)
    # the game is a decomposed design, so its cost is >= the optimum, and here
    # the modal decomposition is lossless => they coincide.
    assert game_cost >= lqr_cost - 1e-3 * abs(lqr_cost)
    assert abs(game_cost - lqr_cost) <= 1e-6 * abs(lqr_cost)


def test_pendulum_nonlinear_model_linearizes_to_design() -> None:
    """The nonlinear cart-pendulum EOM must linearize to the design (A, B)."""
    comparison = InvertedPendulumComparison(
        m_c=10.0, m_p=1.0, p_L=1.0, q=1.0, r=1.0, x_0=np.zeros(4), x_T=np.zeros(4), L=50
    )
    lqr_game = comparison[0]
    A = lqr_game.A
    B = lqr_game.Bs[0]  # the LQR player drives the full physical input

    def state_dot(state: np.ndarray, force: float, moment: float) -> np.ndarray:
        _x, theta, _x_dot, theta_dot = state
        x_ddot, theta_ddot = comparison._nonlinear_acceleration(theta, theta_dot, force, moment)
        return np.array([_x_dot, theta_dot, x_ddot, theta_ddot])

    eps = 1e-6
    origin = np.zeros(4)
    A_num = np.zeros((4, 4))
    for j in range(4):
        step = np.zeros(4)
        step[j] = eps
        A_num[:, j] = (state_dot(origin + step, 0.0, 0.0) - state_dot(origin - step, 0.0, 0.0)) / (2 * eps)
    B_num = np.zeros((4, 2))
    for j, (force, moment) in enumerate([(eps, 0.0), (0.0, eps)]):
        B_num[:, j] = (state_dot(origin, force, moment) - state_dot(origin, -force, -moment)) / (2 * eps)

    np.testing.assert_allclose(A_num, A, atol=1e-5)
    np.testing.assert_allclose(B_num, B, atol=1e-5)
