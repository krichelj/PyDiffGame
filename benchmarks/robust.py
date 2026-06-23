"""Robust (H-infinity) state feedback as a controller-vs-disturbance game.

H-infinity control is the saddle point of a two-player zero-sum differential
game: the controller minimises, an adversarial disturbance maximises,
``J = integral x'Qx + u'Ru - gamma^2 w'w dt``. The resulting controller
minimises the worst-case L2 gain from disturbance ``w`` to the weighted
performance output ``z = [Q^{1/2} x ; R^{1/2} u]`` — exactly what a plain LQR
does *not* optimise. This module solves that game and measures the formal
robustness metric (closed-loop H-infinity norm) so the boost is verifiable.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import schur

FloatArray = np.ndarray


def _sqrtm_psd(M: FloatArray) -> FloatArray:
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def h_infinity_gain(
    A: FloatArray, B: FloatArray, Q: FloatArray, R: FloatArray, B_w: FloatArray, gamma: float
) -> tuple[FloatArray, FloatArray]:
    """Solve the H-infinity GARE via the Hamiltonian / ordered-Schur method.

    Returns ``(P, K)`` with feedback ``u = -K x``. Raises if no stabilising
    solution exists at this ``gamma`` (i.e. gamma is below the optimal gamma*).
    The effective quadratic term ``S = B R^{-1} B' - gamma^{-2} B_w B_w'`` is
    *indefinite*, which is exactly why a standard LQR Riccati solver cannot do
    this and the game formulation is needed.
    """
    n = A.shape[0]
    R = np.atleast_2d(R)
    R_inv = np.linalg.inv(R)
    S = B @ R_inv @ B.T - (1.0 / gamma**2) * (B_w @ B_w.T)
    H = np.block([[A, -S], [-Q, -A.T]])
    T, Z, sdim = schur(H, sort="lhp")
    if sdim != n:
        raise ValueError(
            f"no stabilising H-infinity solution at gamma={gamma:.4g} (stable dim {sdim} != {n})"
        )
    U = Z[:, :n]
    X1, X2 = U[:n, :], U[n:, :]
    if abs(np.linalg.det(X1)) < 1e-12:
        raise ValueError(f"singular Schur block at gamma={gamma:.4g}")
    P = np.real(X2 @ np.linalg.inv(X1))
    P = 0.5 * (P + P.T)
    K = R_inv @ B.T @ P
    return P, K


def min_gamma(
    A: FloatArray,
    B: FloatArray,
    Q: FloatArray,
    R: FloatArray,
    B_w: FloatArray,
    *,
    lo: float = 1e-2,
    hi: float = 1e4,
) -> float:
    """Bisection for the smallest gamma admitting a stabilising H-infinity solution (gamma*)."""
    # ensure hi is feasible
    feasible = False
    for _ in range(40):
        try:
            P, _ = h_infinity_gain(A, B, Q, R, B_w, hi)
            if np.all(np.linalg.eigvalsh(P) > -1e-8):
                feasible = True
                break
        except ValueError:
            pass
        hi *= 2
    if not feasible:
        raise RuntimeError("could not find a feasible gamma upper bound")
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        try:
            P, _ = h_infinity_gain(A, B, Q, R, B_w, mid)
            ok = np.all(np.linalg.eigvalsh(P) > -1e-8)
        except ValueError:
            ok = False
        if ok:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.0 + 1e-4:
            break
    return hi


def closed_loop_hinf_norm(
    A: FloatArray, B: FloatArray, K: FloatArray, Q: FloatArray, R: FloatArray, B_w: FloatArray
) -> tuple[float, float]:
    """Worst-case L2 gain ||G_zw||_inf from disturbance w to z=[Q^{1/2}x; R^{1/2}u].

    ``||G||_inf = max_omega sigma_max(C (jw I - A_cl)^{-1} B_w)`` evaluated by a
    log-spaced frequency sweep refined around the peak (no slycot needed).
    Returns ``(gpeak, fpeak)``; lower is more robust. This is the formal,
    provable robustness metric. Returns ``inf`` if the closed loop is unstable.
    """
    A_cl = A - B @ K
    if np.max(np.linalg.eigvals(A_cl).real) >= 0:
        return float("inf"), float("nan")
    C = np.vstack([_sqrtm_psd(Q), -_sqrtm_psd(np.atleast_2d(R)) @ K])
    n = A_cl.shape[0]
    eye = np.eye(n)

    def sigma_max(omega: float) -> float:
        G = C @ np.linalg.solve(1j * omega * eye - A_cl, B_w)
        return float(np.linalg.svd(G, compute_uv=False)[0])

    grid = np.concatenate([[0.0], np.logspace(-4, 5, 6000)])
    vals = np.array([sigma_max(w) for w in grid])
    i = int(np.argmax(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, 3000)
    fvals = np.array([sigma_max(w) for w in fine])
    j = int(np.argmax(fvals))
    if fvals[j] >= vals[i]:
        return float(fvals[j]), float(fine[j])
    return float(vals[i]), float(grid[i])
