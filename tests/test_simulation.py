"""Simulation, cost evaluation and the comparison orchestrator."""

import numpy as np
import pytest

from PyDiffGame import ContinuousLQR, GameObjective, LQRObjective, PyDiffGameLQRComparison


def test_regulated_state_converges_to_origin():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    lqr = ContinuousLQR(
        A=A, B=B, Q=np.eye(2), R=1.0, x_0=np.array([5.0, 0.0]), T_f=20.0, L=400
    ).solve().simulate()
    assert np.allclose(lqr.x[-1], 0.0, atol=1e-2)
    assert lqr.x.shape == (400, 2)


def test_tracking_state_converges_to_target():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    x_T = np.array([3.0, 0.0])
    lqr = ContinuousLQR(
        A=A, B=B, Q=np.eye(2), R=1.0, x_0=np.zeros(2), x_T=x_T, T_f=20.0, L=400
    ).solve().simulate()
    assert np.allclose(lqr.x[-1], x_T, atol=1e-2)


def test_cost_is_positive_and_requires_simulation():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    obj = LQRObjective(Q=np.eye(2), R=1.0)
    lqr = ContinuousLQR(A=A, B=B, Q=np.eye(2), R=1.0, x_0=np.array([1.0, 1.0]), T_f=10.0, L=200)
    with pytest.raises(RuntimeError):
        lqr.cost(obj)  # not solved yet
    lqr.solve().simulate()
    assert lqr.cost(obj) > 0.0


def test_comparison_runs_and_reports_costs():
    A = np.array(
        [[0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 0.0]]
    )
    B = np.eye(4)[:, [1, 3]]
    x_0 = np.array([10.0, 0.0, 20.0, 0.0])
    M1, M2 = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    lqr_group = [LQRObjective(Q=np.eye(4), R=np.eye(2))]
    game_group = [
        GameObjective(Q=np.diag([1.0, 0.1, 0.0, 0.0]), R=1.0, M=M1),
        GameObjective(Q=np.diag([0.0, 0.0, 1.0, 0.1]), R=1.0, M=M2),
    ]
    comparison = PyDiffGameLQRComparison(
        A=A, B=B, games_objectives=[lqr_group, game_group], x_0=x_0, T_f=15.0, L=200
    )
    comparison.run(plot_state_spaces=False)
    costs = comparison.costs()
    assert len(comparison) == 2
    assert all(c > 0 for c in costs)
