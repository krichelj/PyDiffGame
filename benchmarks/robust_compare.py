"""Robustness showdown: H-infinity differential game vs LQR under disturbance.

Produces (a) the formal metric — closed-loop worst-case L2 gain ||G_zw||_inf —
and (b) an animated time-domain comparison driving both closed loops with the
single worst-case sinusoidal disturbance (at the frequency where the LQR is most
vulnerable), so the robustness boost is visible, not just asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import control as ct
import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from benchmarks.harness import _style
from benchmarks.robust import closed_loop_hinf_norm, h_infinity_gain, min_gamma

RESULTS = Path(__file__).resolve().parent / "results"
FloatArray = np.ndarray


@dataclass
class RobustProblem:
    name: str
    A: FloatArray
    B: FloatArray
    B_w: FloatArray
    Q: FloatArray
    R: FloatArray
    x_0: FloatArray
    T_f: float
    output_idx: int  # which state to plot as the "performance output"
    output_name: str
    gamma_factor: float = 1.3  # robustness level: gamma = factor * gamma*


def _sigma_curve(A, B, K, Q, R, B_w, omegas):
    from benchmarks.robust import _sqrtm_psd

    A_cl = A - B @ K
    C = np.vstack([_sqrtm_psd(Q), -_sqrtm_psd(np.atleast_2d(R)) @ K])
    eye = np.eye(A_cl.shape[0])
    return np.array(
        [
            float(np.linalg.svd(C @ np.linalg.solve(1j * w * eye - A_cl, B_w), compute_uv=False)[0])
            for w in omegas
        ]
    )


def _simulate(A, B, K, B_w, x0, t, w_signal):
    def f(ti, x):
        return (A - B @ K) @ x + B_w @ np.atleast_1d(w_signal(ti))

    sol = solve_ivp(
        f, (t[0], t[-1]), x0, t_eval=t, method="LSODA", rtol=1e-9, atol=1e-11, max_step=(t[-1] - t[0]) / 400
    )
    return sol.y.T


def run(problem: RobustProblem, *, render_gif: bool = True) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    A, B, B_w, Q, R = problem.A, problem.B, problem.B_w, problem.Q, problem.R

    K_lqr, _, _ = ct.lqr(A, B, Q, R)
    K_lqr = np.asarray(K_lqr)
    gstar = min_gamma(A, B, Q, R, B_w)
    _, K_hinf = h_infinity_gain(A, B, Q, R, B_w, gstar * problem.gamma_factor)

    g_lqr, f_lqr = closed_loop_hinf_norm(A, B, K_lqr, Q, R, B_w)
    g_hinf, f_hinf = closed_loop_hinf_norm(A, B, K_hinf, Q, R, B_w)
    reduction = 100.0 * (g_lqr - g_hinf) / g_lqr

    # Time response to the single worst-case sinusoid (at the LQR's peak frequency).
    # Computed always (cheap) so the report has the time-domain peak even without a GIF.
    t = np.linspace(0.0, problem.T_f, 700)
    omega = max(f_lqr, 1e-3)

    def w_sig(ti):
        return np.sin(omega * ti)

    x_lqr = _simulate(A, B, K_lqr, B_w, problem.x_0, t, w_sig)
    x_hinf = _simulate(A, B, K_hinf, B_w, problem.x_0, t, w_sig)
    oi = problem.output_idx
    peak_lqr = float(np.abs(x_lqr[:, oi]).max())
    peak_hinf = float(np.abs(x_hinf[:, oi]).max())

    report = {
        "problem": problem.name,
        "gamma_star": gstar,
        "gamma_used": gstar * problem.gamma_factor,
        "hinf_norm": {"LQR": g_lqr, "Hinf_game": g_hinf, "reduction_pct": reduction},
        "time_domain_peak": {"LQR": peak_lqr, "Hinf_game": peak_hinf},
        "K_lqr": K_lqr.tolist(),
        "K_hinf": K_hinf.tolist(),
        "peak_freq": {"LQR": f_lqr, "Hinf_game": f_hinf},
    }

    print(f"\n=== {problem.name} — robustness (H-infinity game vs LQR) ===")
    print(f"  gamma* = {gstar:.4g},  gamma used = {gstar * problem.gamma_factor:.4g}")
    print(f"  ||G_zw||inf   LQR = {g_lqr:.4g}   H-inf = {g_hinf:.4g}   ({reduction:+.1f}%)")
    verdict = "BOOST" if reduction > 1.0 else ("tie" if abs(reduction) <= 1.0 else "WORSE")
    print(f"  VERDICT: worst-case L2 gain {verdict} for the H-infinity game.")

    if render_gif:
        _style()
        fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        # right: formal sigma_max(omega) curves (static)
        omegas = np.logspace(-2, 2, 250)
        ax_f.semilogx(
            omegas, _sigma_curve(A, B, K_lqr, Q, R, B_w, omegas), color="#d97706", lw=2.2, label="LQR"
        )
        ax_f.semilogx(
            omegas, _sigma_curve(A, B, K_hinf, Q, R, B_w, omegas), color="#059669", lw=2.2, label="H∞ game"
        )
        ax_f.axhline(g_lqr, color="#d97706", ls=(0, (1, 3)), lw=1)
        ax_f.axhline(g_hinf, color="#059669", ls=(0, (1, 3)), lw=1)
        ax_f.set_title(
            f"worst-case gain σmax(ω):  {g_lqr:.2f} → {g_hinf:.2f}  ({reduction:+.0f}%)", fontweight="bold"
        )
        ax_f.set_xlabel("frequency ω [rad/s]")
        ax_f.set_ylabel("σmax of G(jω)")
        ax_f.legend(loc="upper right", fontsize=10)

        # left: animated time response under worst-case disturbance
        ax_t.set_title(f"{problem.name}: response to worst-case disturbance", fontweight="bold")
        ax_t.set_xlabel("time [s]")
        ax_t.set_ylabel(problem.output_name)
        ax_t.set_xlim(t[0], t[-1])
        allo = np.concatenate([x_lqr[:, oi], x_hinf[:, oi]])
        pad = 0.1 * (allo.max() - allo.min() + 1e-9)
        ax_t.set_ylim(allo.min() - pad, allo.max() + pad)
        (ln_l,) = ax_t.plot(
            [], [], color="#d97706", lw=2.2, label=f"LQR  (peak {np.abs(x_lqr[:, oi]).max():.2f})"
        )
        (ln_h,) = ax_t.plot(
            [], [], color="#059669", lw=2.2, label=f"H∞ game  (peak {np.abs(x_hinf[:, oi]).max():.2f})"
        )
        ax_t.legend(loc="upper right", fontsize=10)

        stride = 9
        frames = range(1, len(t), stride)

        def update(fi):
            ln_l.set_data(t[:fi], x_lqr[:fi, oi])
            ln_h.set_data(t[:fi], x_hinf[:fi, oi])
            return ln_l, ln_h

        fig.tight_layout()
        ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
        out = str(RESULTS / f"robust_{problem.name.replace(' ', '_')}.gif")
        ani.save(out, writer=animation.PillowWriter(fps=25))
        plt.close(fig)
        report["gif"] = out
        print(f"  time-domain peak |{problem.output_name}|:  LQR={peak_lqr:.3g}  H-inf={peak_hinf:.3g}")
        print(f"  GIF: {out}")

    (RESULTS / f"robust_{problem.name.replace(' ', '_')}.json").write_text(
        json.dumps(report, indent=2, default=float)
    )
    return report


def main() -> None:
    cart = RobustProblem(
        name="cart with disturbance",
        A=np.array([[0.0, 1.0], [0.0, 0.0]]),
        B=np.array([[0.0], [1.0]]),
        B_w=np.array([[0.0], [1.0]]),
        Q=np.diag([1.0, 0.0]),
        R=np.array([[1.0]]),
        x_0=np.array([0.0, 0.0]),
        T_f=40.0,
        output_idx=0,
        output_name="position",
    )
    run(cart)


if __name__ == "__main__":
    main()
