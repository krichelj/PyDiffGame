r"""Robust (:math:`H_\infty`) state feedback as a controller-vs-disturbance game.

This completes the differential-games family in :mod:`PyDiffGame`: alongside the
single-objective :class:`~PyDiffGame.lqr.ContinuousLQR` and the :math:`N`-player
Nash :class:`~PyDiffGame.continuous.ContinuousPyDiffGame`, the *robust* design is
the two-player **zero-sum** differential game in which the controller minimises
and an adversarial disturbance maximises

.. math::
    J = \int_0^\infty x^\top Q x + u^\top R u - \gamma^2 w^\top w \; dt ,
    \qquad \dot x = A x + B u + B_w w .

Its saddle point is the :math:`H_\infty` state-feedback controller
:math:`u = -K x`, :math:`K = R^{-1} B^\top P`, where :math:`P` solves the game
algebraic Riccati equation

.. math::
    A^\top P + P A + Q - P\,(B R^{-1} B^\top - \gamma^{-2} B_w B_w^\top)\,P = 0 .

The quadratic term is *indefinite* (the disturbance subtracts), which is exactly
why an ordinary LQR Riccati solver cannot produce it and the game formulation is
required. The controller minimises the worst-case :math:`L_2` gain from the
disturbance to the weighted performance output — the robustness an LQR does not
optimise.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import schur

from PyDiffGame._typing import ArrayLike, FloatArray


def _as_matrix(value: ArrayLike) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    return arr.reshape(1, 1) if arr.ndim == 0 else arr


def _symmetric_sqrt(matrix: FloatArray) -> FloatArray:
    """Symmetric positive-semidefinite square root of ``matrix``."""
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T


class ContinuousHInfinityControl:
    r"""Continuous-time :math:`H_\infty` state feedback via the disturbance game.

    Parameters
    ----------
    A, B : array_like
        State and control matrices of :math:`\dot x = A x + B u + B_w w`.
    B_w : array_like
        Disturbance input matrix :math:`B_w`.
    Q, R : array_like
        Symmetric weights of the performance output (``Q`` PSD, ``R`` PD).
    gamma : float, optional
        Robustness level. Smaller is more robust but harder; it must exceed the
        optimal :math:`\gamma^\star`. If omitted, :math:`\gamma^\star` is found by
        bisection and ``gamma = gamma_margin * gamma_star`` is used.
    gamma_margin : float, default ``1.3``
        Multiplier applied to :math:`\gamma^\star` when ``gamma`` is not given.
    """

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        B_w: ArrayLike,
        Q: ArrayLike,
        R: ArrayLike,
        *,
        gamma: float | None = None,
        gamma_margin: float = 1.3,
    ) -> None:
        self._A = _as_matrix(A)
        self._B = _as_matrix(B)
        self._B_w = _as_matrix(B_w)
        self._Q = _as_matrix(Q)
        self._R = _as_matrix(R)
        self._n = self._A.shape[0]
        if self._A.shape != (self._n, self._n):
            raise ValueError("A must be square")
        for name, mat, rows in (
            ("B", self._B, self._n),
            ("B_w", self._B_w, self._n),
            ("Q", self._Q, self._n),
        ):
            if mat.shape[0] != rows:
                raise ValueError(f"{name} must have {rows} rows")
        if gamma_margin <= 1.0:
            raise ValueError("gamma_margin must be > 1")

        self._gamma_margin = float(gamma_margin)
        self._gamma_star: float | None = None
        self._gamma = None if gamma is None else float(gamma)
        self._P: FloatArray | None = None
        self._K: FloatArray | None = None

    # ------------------------------------------------------------------ #
    # GARE solve (Hamiltonian / ordered Schur)
    # ------------------------------------------------------------------ #
    def _solve_gare(self, gamma: float) -> FloatArray:
        R_inv = np.linalg.inv(self._R)
        S = self._B @ R_inv @ self._B.T - (1.0 / gamma**2) * (self._B_w @ self._B_w.T)
        hamiltonian = np.block([[self._A, -S], [-self._Q, -self._A.T]])
        _, vectors, stable_dim = schur(hamiltonian, sort="lhp")
        if stable_dim != self._n:
            raise ValueError(
                f"no stabilising H-infinity solution at gamma={gamma:.4g} "
                f"(stable subspace dimension {stable_dim} != {self._n}); gamma is below gamma*"
            )
        basis = vectors[:, : self._n]
        upper, lower = basis[: self._n, :], basis[self._n :, :]
        if abs(np.linalg.det(upper)) < 1e-12:
            raise ValueError(f"degenerate Schur basis at gamma={gamma:.4g}")
        P = np.real(lower @ np.linalg.inv(upper))
        return 0.5 * (P + P.T)

    def _stabilizing_solution(self, gamma: float) -> tuple[FloatArray | None, bool]:
        """Solve the GARE at ``gamma`` and validate a stabilising PSD solution.

        Robust admissibility predicate: rather than trust the (boundary-brittle)
        stable-subspace dimension count alone, ``gamma`` is admissible only if the
        recovered ``P`` is symmetric PSD *and* the closed loop ``A - B K`` is
        strictly Hurwitz with a small margin. This makes admissibility monotone
        through :math:`\\gamma^\\star` and immune to the imaginary-axis eigenvalue
        jitter of the Hamiltonian at the boundary.
        """
        try:
            P = self._solve_gare(gamma)
        except ValueError:
            return None, False
        if float(np.min(np.linalg.eigvalsh(P))) < -1e-7:
            return P, False
        K = np.linalg.inv(self._R) @ self._B.T @ P
        spectral_abscissa = float(np.max(np.linalg.eigvals(self._A - self._B @ K).real))
        return P, spectral_abscissa < -1e-7

    def _is_admissible(self, gamma: float) -> bool:
        return self._stabilizing_solution(gamma)[1]

    def optimal_gamma(self, *, iterations: int = 100) -> float:
        r"""Smallest :math:`\gamma` admitting a stabilising solution (:math:`\gamma^\star`).

        Brackets :math:`\gamma^\star` from *both* sides adaptively — doubling the
        upper bound up until admissible and halving the lower bound down until
        inadmissible — so it never silently floors out on systems whose
        :math:`\gamma^\star` is tiny, then bisects on the validated admissibility
        predicate (:meth:`_stabilizing_solution`), which is monotone through the
        boundary.
        """
        if self._gamma_star is not None:
            return self._gamma_star
        upper = 1.0
        for _ in range(60):
            if self._is_admissible(upper):
                break
            upper *= 2.0
        else:
            raise RuntimeError("could not bracket a feasible upper gamma")
        lower = upper
        for _ in range(60):
            lower *= 0.5
            if not self._is_admissible(lower):
                break
        else:
            # admissible all the way down: gamma* is effectively zero
            self._gamma_star = float(lower)
            return self._gamma_star
        for _ in range(iterations):
            mid = np.sqrt(lower * upper)
            if self._is_admissible(mid):
                upper = mid
            else:
                lower = mid
            if upper / lower < 1.0 + 1e-6:
                break
        self._gamma_star = float(upper)
        return self._gamma_star

    def solve(self) -> ContinuousHInfinityControl:
        """Solve the disturbance game and store the controller. Returns ``self``."""
        gamma = self._gamma if self._gamma is not None else self.optimal_gamma() * self._gamma_margin
        self._gamma = float(gamma)
        self._P = self._solve_gare(gamma)
        self._K = np.linalg.inv(self._R) @ self._B.T @ self._P
        return self

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    def _require_solved(self) -> None:
        if self._K is None:
            raise RuntimeError("call solve() first")

    @property
    def K(self) -> FloatArray:
        self._require_solved()
        assert self._K is not None
        return self._K

    @property
    def P(self) -> FloatArray:
        self._require_solved()
        assert self._P is not None
        return self._P

    @property
    def gamma(self) -> float:
        self._require_solved()
        assert self._gamma is not None
        return self._gamma

    @property
    def A_cl(self) -> FloatArray:
        return self._A - self._B @ self.K

    def is_closed_loop_stable(self, tolerance: float = 1e-9) -> bool:
        return bool(np.all(np.linalg.eigvals(self.A_cl).real < tolerance))

    def gare_residual(self) -> float:
        """Max-abs residual of the game algebraic Riccati equation (should be ~0)."""
        self._require_solved()
        P, R_inv = self.P, np.linalg.inv(self._R)
        S = self._B @ R_inv @ self._B.T - (1.0 / self.gamma**2) * (self._B_w @ self._B_w.T)
        res = self._A.T @ P + P @ self._A + self._Q - P @ S @ P
        return float(np.max(np.abs(res)))

    def worst_case_gain(self, *, n_grid: int = 6000) -> tuple[float, float]:
        r"""Closed-loop worst-case :math:`L_2` gain of this :math:`H_\infty` design.

        See :func:`worst_case_l2_gain`. Returns ``(peak_gain, peak_frequency)``.
        """
        self._require_solved()
        return worst_case_l2_gain(self._A, self._B, self.K, self._Q, self._R, self._B_w, n_grid=n_grid)


def worst_case_l2_gain(
    A: ArrayLike,
    B: ArrayLike,
    K: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    B_w: ArrayLike,
    *,
    n_grid: int = 6000,
) -> tuple[float, float]:
    r"""Worst-case :math:`L_2` gain :math:`\lVert G_{zw}\rVert_\infty` of any state-feedback gain.

    The induced gain from disturbance ``w`` to the weighted performance output
    :math:`z = [Q^{1/2} x;\, R^{1/2} u]` for the closed loop ``A - B K``, i.e.
    :math:`\max_\omega \sigma_{\max}(C (j\omega I - (A-BK))^{-1} B_w)`, by a
    log-spaced frequency sweep refined around the peak (no slycot needed). Lower
    is more robust; this is the formal robustness metric used to compare an
    :math:`H_\infty` design against, say, an LQR. Returns ``inf`` if unstable.
    """
    A, B, K = _as_matrix(A), _as_matrix(B), _as_matrix(K)
    Q, R, B_w = _as_matrix(Q), _as_matrix(R), _as_matrix(B_w)
    A_cl = A - B @ K
    eigenvalues = np.linalg.eigvals(A_cl)
    if np.max(eigenvalues.real) >= 0:
        return float("inf"), float("nan")
    C = np.vstack([_symmetric_sqrt(Q), -_symmetric_sqrt(R) @ K])
    eye = np.eye(A_cl.shape[0])

    def sigma_max(omega: float) -> float:
        transfer = C @ np.linalg.solve(1j * omega * eye - A_cl, B_w)
        return float(np.linalg.svd(transfer, compute_uv=False)[0])

    # Upper frequency derived from the fastest closed-loop mode so the sweep can
    # never miss a high-frequency peak (with a generous safety factor and floor).
    top = max(1.0e5, 50.0 * float(np.max(np.abs(eigenvalues))))
    grid = np.concatenate([[0.0], np.logspace(-4, np.log10(top), n_grid)])
    values = np.array([sigma_max(w) for w in grid])
    peak = int(np.argmax(values))
    lo = grid[max(peak - 1, 0)]
    hi = grid[min(peak + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, n_grid // 2)
    fine_values = np.array([sigma_max(w) for w in fine])
    best = int(np.argmax(fine_values))
    if fine_values[best] >= values[peak]:
        return float(fine_values[best]), float(fine[best])
    return float(values[peak]), float(grid[peak])


__all__ = ["ContinuousHInfinityControl", "worst_case_l2_gain"]
