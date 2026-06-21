r"""Discrete-time differential game solver.

Considers control design for

.. math:: x[k+1] = \tilde A x[k] + \sum_{i=1}^N \tilde B_i v_i[k]

If the matrices are supplied in continuous form (the default) they are first
discretised with a zero-order hold using the Van Loan matrix-exponential
identity.  The coupled feedback gains at each step are obtained by solving a
*linear* block system (the discrete coupled-Riccati gain equations are linear in
the gains given the next-step cost matrices), which is both faster and far more
robust than the root-finding the previous implementation attempted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from PyDiffGame._typing import ArrayLike, FloatArray
from PyDiffGame.base import PyDiffGame
from PyDiffGame.objective import Objective


class DiscretePyDiffGame(PyDiffGame):
    """Discrete-time differential game / LQR solver.

    Parameters
    ----------
    is_input_discrete:
        When ``True`` the matrices ``A``, ``B`` and the objectives are taken to
        be already in discrete form.  When ``False`` (default) the continuous
        data are discretised with sampling period ``delta = T_f / L``.
    """

    def __init__(
        self,
        A: ArrayLike,
        objectives: Sequence[Objective],
        *,
        is_input_discrete: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(A, objectives, **kwargs)
        if not is_input_discrete:
            self._discretize()
        # Riccati seeds must be computed from the (possibly discretised) data.
        self._P_f = self._uncoupled_are_solutions()

    def _solve_are(self, B: FloatArray, Q: FloatArray, R: FloatArray) -> FloatArray:
        return solve_discrete_are(self._A, B, Q, R)

    def _discretize(self) -> None:
        r"""Zero-order-hold discretisation via the Van Loan identity.

        .. math:: \exp\!\left(\begin{bmatrix}A & I\\0 & 0\end{bmatrix}\delta\right)
                  = \begin{bmatrix}\tilde A & \Gamma\\0 & I\end{bmatrix},
                  \qquad \tilde B_i = \Gamma B_i

        The quadratic cost weights are scaled by the sampling period so the
        discrete sum approximates the continuous integral.
        """

        n, delta = self._n, self._delta
        augmented = np.zeros((2 * n, 2 * n))
        augmented[:n, :n] = self._A
        augmented[:n, n:] = np.eye(n)
        exponential = expm(augmented * delta)
        A_d = exponential[:n, :n]
        gamma = exponential[:n, n:]

        self._A = A_d
        self._Bs = [gamma @ B_i for B_i in self._Bs]
        self._Qs = [Q_i * delta for Q_i in self._Qs]
        self._Rs = [R_i * delta for R_i in self._Rs]

    # ------------------------------------------------------------------ #
    # Coupled gain solve (one backward step)
    # ------------------------------------------------------------------ #
    def _step_gains(self, Ps_next: list[FloatArray]) -> list[FloatArray]:
        r"""Feedback gains for one backward step given next-step matrices ``P[k+1]``.

        Solves the linear block system arising from

        .. math:: K_i = (R_{ii} + B_i^\top P_i B_i)^{-1} B_i^\top P_i
                        \Big(A - \sum_{j\neq i} B_j K_j\Big)
        """

        G = [
            np.linalg.solve(R_i + B_i.T @ P_i @ B_i, B_i.T @ P_i)
            for B_i, R_i, P_i in zip(self._Bs, self._Rs, Ps_next)
        ]
        offsets = np.cumsum([0] + [B_i.shape[1] for B_i in self._Bs])
        m = int(offsets[-1])
        C = np.zeros((m, m))
        D = np.zeros((m, self._n))
        for i in range(self._N):
            rows = slice(offsets[i], offsets[i + 1])
            D[rows] = G[i] @ self._A
            for j in range(self._N):
                cols = slice(offsets[j], offsets[j + 1])
                if i == j:
                    C[rows, cols] = np.eye(self._Bs[i].shape[1])
                else:
                    C[rows, cols] = G[i] @ self._Bs[j]
        K_stacked = np.linalg.solve(C, D)
        return [K_stacked[offsets[i] : offsets[i + 1]] for i in range(self._N)]

    def _step_P(self, Ps_next: list[FloatArray], gains: list[FloatArray]) -> list[FloatArray]:
        A_cl = self._closed_loop(gains)
        return [
            A_cl.T @ P_i @ A_cl + K_i.T @ R_i @ K_i + Q_i
            for P_i, K_i, R_i, Q_i in zip(Ps_next, gains, self._Rs, self._Qs)
        ]

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    def solve(self) -> DiscretePyDiffGame:
        if self._infinite_horizon:
            self._solve_infinite_horizon()
        else:
            self._solve_finite_horizon()
        self._solved = True
        return self

    def _solve_infinite_horizon(self) -> None:
        if self.is_lqr:
            P = self._solve_are(self._Bs[0], self._Qs[0], self._Rs[0])
            Ps = [P]
            gains = self._step_gains(Ps)
        else:
            Ps = list(self._P_f)
            previous = np.inf
            for _ in range(self.max_P_iterations):
                gains = self._step_gains(Ps)
                Ps = self._step_P(Ps, gains)
                norm = sum(float(np.linalg.norm(P)) for P in Ps)
                if abs(norm - previous) < self._epsilon_P:
                    break
                previous = norm
            gains = self._step_gains(Ps)

        self._P = Ps
        self._K = gains
        self._K_aggregate = self._aggregate_gain(gains)
        self._A_cl = self._closed_loop(gains)

    def _solve_finite_horizon(self) -> None:
        Ps_t: list[list[FloatArray]] = [list(self._P_f)]
        gains_t: list[list[FloatArray]] = []
        for _ in range(self._L - 1):
            gains = self._step_gains(Ps_t[-1])
            Ps_t.append(self._step_P(Ps_t[-1], gains))
            gains_t.append(gains)
        gains_t.append(self._step_gains(Ps_t[-1]))

        # Index 0 == t = 0 (forward time): reverse the backward sweep.
        Ps_t.reverse()
        gains_t.reverse()
        self._P = np.stack([np.stack(step) for step in Ps_t])  # (L, N, n, n)
        self._K = gains_t
        self._K_aggregate_t = [self._aggregate_gain(g) for g in gains_t]
        self._A_cl = self._closed_loop(gains_t[0])

    # ------------------------------------------------------------------ #
    # Gains
    # ------------------------------------------------------------------ #
    def _aggregate_gain(self, player_gains: list[FloatArray]) -> FloatArray:
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
    def simulate(self) -> DiscretePyDiffGame:
        self._require_solved()
        if self._x_0 is None:
            raise RuntimeError("simulate() requires x_0")
        target = self._x_T if self._x_T is not None else np.zeros(self._n)

        x = np.zeros((self._L, self._n))
        x[0] = self._x_0
        for k in range(self._L - 1):
            x[k + 1] = self._closed_loop_at(k) @ (x[k] - target) + target
        self._x = x
        return self

    # ------------------------------------------------------------------ #
    # Stability
    # ------------------------------------------------------------------ #
    def is_closed_loop_stable(self) -> bool:
        """Schur test: the closed-loop spectral radius is strictly below one."""

        self._require_solved()
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(self._A_cl))))
        return spectral_radius < 1.0


__all__ = ["DiscretePyDiffGame"]
