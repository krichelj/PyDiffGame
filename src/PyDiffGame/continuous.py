r"""Continuous-time differential game solver.

Solves the control design for

.. math:: \dot{x}(t) = A x(t) + \sum_{i=1}^N B_i v_i(t)

via the coupled differential / algebraic Riccati equations.  The feedback Nash
gains are :math:`K_i = R_{ii}^{-1} B_i^\top P_i`, and the closed loop is
:math:`A_{cl} = A - \sum_i B_i K_i`.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

from PyDiffGame._typing import FloatArray
from PyDiffGame.base import PyDiffGame


class ContinuousPyDiffGame(PyDiffGame):
    """Continuous-time differential game / LQR solver."""

    def _solve_are(self, B: FloatArray, Q: FloatArray, R: FloatArray) -> FloatArray:
        return solve_continuous_are(self._A, B, Q, R)

    # ------------------------------------------------------------------ #
    # Coupled Riccati dynamics
    # ------------------------------------------------------------------ #
    def _dP_dt(self, _t: float, P_flat: FloatArray) -> FloatArray:
        r"""RHS of the coupled matrix Riccati ODE (forward-time derivative).

        .. math:: \dot P_i = -\left(A_{cl}^\top P_i + P_i A_{cl}
                  + Q_i + P_i S_i P_i\right),\qquad A_{cl}=A-\sum_j S_j P_j
        """

        Ps = self._unflatten(P_flat)
        A_cl = self._A - sum(S_j @ P_j for S_j, P_j in zip(self._S, Ps))
        derivatives = [
            -(A_cl.T @ P_i + P_i @ A_cl + Q_i + P_i @ S_i @ P_i)
            for P_i, S_i, Q_i in zip(Ps, self._S, self._Qs)
        ]
        return np.concatenate([d.ravel() for d in derivatives])

    def _unflatten(self, P_flat: FloatArray) -> list[FloatArray]:
        size = self._n * self._n
        return [P_flat[i * size : (i + 1) * size].reshape(self._n, self._n) for i in range(self._N)]

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    def solve(self) -> ContinuousPyDiffGame:
        if self._infinite_horizon:
            self._solve_infinite_horizon()
        else:
            self._solve_finite_horizon()
        self._solved = True
        return self

    def _solve_infinite_horizon(self) -> None:
        if self.is_lqr:
            P = self._solve_are(self._Bs[0], self._Qs[0], self._Rs[0])
            self._P = [P]
        else:
            self._P = self._converge_to_algebraic_solution()

        self._K = [np.linalg.solve(R_i, B_i.T) @ P_i
                   for R_i, B_i, P_i in zip(self._Rs, self._Bs, self._P)]
        self._K_aggregate = self._aggregate_gain(self._K)
        self._A_cl = self._closed_loop(self._K)

    def _converge_to_algebraic_solution(self) -> list[FloatArray]:
        """Integrate the coupled DREs backwards over repeated horizons until steady state."""

        Ps = list(self._P_f)
        previous_norm = np.inf
        for _ in range(self.max_P_iterations):
            solution = solve_ivp(
                fun=self._dP_dt,
                t_span=(self._T_f, 0.0),
                y0=np.concatenate([P.ravel() for P in Ps]),
                method="LSODA",
                t_eval=[0.0],
                rtol=1e-8,
                atol=1e-10,
            )
            Ps = self._unflatten(solution.y[:, -1])
            norm = sum(float(np.linalg.norm(P)) for P in Ps)
            if abs(norm - previous_norm) < self._epsilon_P:
                break
            previous_norm = norm
        return Ps

    def _solve_finite_horizon(self) -> None:
        backward_time = np.linspace(self._T_f, 0.0, self._L)
        solution = solve_ivp(
            fun=self._dP_dt,
            t_span=(self._T_f, 0.0),
            y0=np.concatenate([P.ravel() for P in self._P_f]),
            method="LSODA",
            t_eval=backward_time,
            rtol=1e-8,
            atol=1e-10,
        )
        # Reorder so index 0 corresponds to t = 0 (forward time).
        Ps_t = solution.y[:, ::-1]
        self._P = np.stack(
            [self._unflatten(Ps_t[:, l]) for l in range(self._L)]
        )  # (L, N, n, n)

        gains_t, aggregate_t = [], []
        for l in range(self._L):
            player_gains = [
                np.linalg.solve(R_i, B_i.T) @ self._P[l, i]
                for i, (R_i, B_i) in enumerate(zip(self._Rs, self._Bs))
            ]
            gains_t.append(player_gains)
            aggregate_t.append(self._aggregate_gain(player_gains))
        self._K = gains_t
        self._K_aggregate_t = aggregate_t
        self._A_cl = self._closed_loop(gains_t[0])

    # ------------------------------------------------------------------ #
    # Gains
    # ------------------------------------------------------------------ #
    def _aggregate_gain(self, player_gains: list[FloatArray]) -> FloatArray:
        """Physical-input gain ``K`` such that ``u = -K x``."""

        if self.is_lqr:
            return player_gains[0]
        stacked = np.concatenate(player_gains, axis=0)
        return self._M_inv @ stacked if self._M_inv is not None else stacked

    def _gain_at(self, l: int) -> FloatArray:
        return self._K_aggregate if self._infinite_horizon else self._K_aggregate_t[l]

    def _closed_loop_at(self, l: int) -> FloatArray:
        if self._infinite_horizon:
            return self._A_cl
        return self._closed_loop(self._K[l])

    # ------------------------------------------------------------------ #
    # Simulate
    # ------------------------------------------------------------------ #
    def simulate(self) -> ContinuousPyDiffGame:
        self._require_solved()
        if self._x_0 is None:
            raise RuntimeError("simulate() requires x_0")
        target = self._x_T if self._x_T is not None else np.zeros(self._n)

        def dx_dt(t: float, x: FloatArray) -> FloatArray:
            l = min(int(t / self._delta), self._L - 1)
            return self._closed_loop_at(l) @ (x - target)

        solution = solve_ivp(
            fun=dx_dt,
            t_span=(0.0, self._T_f),
            y0=self._x_0,
            method="LSODA",
            t_eval=self._forward_time,
            rtol=1e-8,
            atol=1e-10,
        )
        self._x = solution.y.T
        return self

    # ------------------------------------------------------------------ #
    # Stability
    # ------------------------------------------------------------------ #
    def is_closed_loop_stable(self) -> bool:
        """Hurwitz test: all eigenvalues have non-positive real part, at most one at 0."""

        self._require_solved()
        eigenvalues = np.linalg.eigvals(self._A_cl)
        real_parts = eigenvalues.real
        zeros = int(np.sum(np.abs(real_parts) < self.eigenvalue_tolerance))
        return bool(np.all(real_parts <= self.eigenvalue_tolerance) and zeros <= 1)

    def algebraic_riccati_residuals(self) -> list[FloatArray]:
        r"""Coupled-ARE residuals :math:`A_{cl}^\top P_i + P_i A_{cl} + Q_i + P_i S_i P_i`.

        Only meaningful for the infinite-horizon case, where each residual should
        be close to zero.
        """

        self._require_solved()
        Ps = self._P if isinstance(self._P, list) else [self._P[0, i] for i in range(self._N)]
        A_cl = self._closed_loop(self._K if isinstance(self._K, list) else self._K[0])
        return [A_cl.T @ P_i + P_i @ A_cl + Q_i + P_i @ S_i @ P_i
                for P_i, S_i, Q_i in zip(Ps, self._S, self._Qs)]


__all__ = ["ContinuousPyDiffGame"]
