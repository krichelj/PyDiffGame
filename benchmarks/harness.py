"""Benchmark harness: rigorously compare a PyDiffGame design against a
python-control baseline on a common state-feedback control problem.

The harness is deliberately controller-agnostic: a *controller* is reduced to an
aggregate state-feedback gain ``K`` (so ``u = -K (x - x_T)``), which is exactly
what both an LQR (``control.lqr``) and a PyDiffGame game produce. Everything
downstream — closed-loop simulation (with optional disturbances), the metric
suite, and the animations — is shared, so the comparison is apples-to-apples.

Honesty note: on a single shared quadratic cost the centralized LQR is optimal
by construction, so a Nash game can at best tie it. The interesting regimes are
robustness (bounded / worst-case disturbances), decentralized actuation, and
compositionality — the metrics below are built to expose all of them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

FloatArray = np.ndarray
Disturbance = Callable[[float, FloatArray], FloatArray]  # (t, x) -> additive state-rate disturbance


# --------------------------------------------------------------------------- #
# Problem + controller descriptions
# --------------------------------------------------------------------------- #
@dataclass
class ControlProblem:
    """A linear control problem ``xdot = A x + B u`` with a tracking target.

    ``Q`` / ``R`` define the *shared yardstick* both controllers are scored on
    (the aggregate objective). ``tracked`` selects which state indices are the
    "outputs" used for settling-time / overshoot metrics (e.g. positions).
    """

    name: str
    A: FloatArray
    B: FloatArray
    Q: FloatArray
    R: FloatArray
    x_0: FloatArray
    x_T: FloatArray
    T_f: float
    tracked: tuple[int, ...]
    state_names: tuple[str, ...]
    L: int = 1000

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.B.shape[1]

    @property
    def time(self) -> FloatArray:
        return np.linspace(0.0, self.T_f, self.L)


@dataclass
class Controller:
    """A state-feedback controller reduced to its aggregate gain ``K`` (u = -K (x - x_T))."""

    name: str
    K: FloatArray
    color: str
    meta: dict = field(default_factory=dict)


@dataclass
class Rollout:
    t: FloatArray
    x: FloatArray  # (L, n)
    u: FloatArray  # (L, m) total applied input (feedforward + feedback)
    controller: Controller
    u_ff: FloatArray  # constant setpoint feedforward (same for every controller)


# --------------------------------------------------------------------------- #
# Closed-loop simulation
# --------------------------------------------------------------------------- #
def setpoint_feedforward(A: FloatArray, B: FloatArray, x_T: FloatArray) -> FloatArray:
    """Constant input that best holds ``x_T`` as an equilibrium: A x_T + B u_ff ≈ 0.

    Computed once from the plant (least-squares via the pseudo-inverse), so it is
    identical for every controller and the comparison stays fair. For setpoints
    that are already equilibria (e.g. the origin) this is just zero.
    """
    return -np.linalg.pinv(B) @ (A @ x_T)


def simulate(
    problem: ControlProblem,
    controller: Controller,
    *,
    disturbance: Disturbance | None = None,
    x_0: FloatArray | None = None,
) -> Rollout:
    """Integrate ``xdot = A x + B u + d`` with ``u = u_ff - K (x - x_T)`` and recover ``u(t)``."""
    A, B, K = problem.A, problem.B, controller.K
    x_T = problem.x_T
    x0 = problem.x_0 if x_0 is None else x_0
    t_eval = problem.time
    u_ff = setpoint_feedforward(A, B, x_T)

    def f(t: float, x: FloatArray) -> FloatArray:
        u = u_ff - K @ (x - x_T)
        xdot = A @ x + B @ u
        if disturbance is not None:
            xdot = xdot + disturbance(t, x)
        return xdot

    sol = solve_ivp(
        f,
        (0.0, problem.T_f),
        x0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-9,
        atol=1e-11,
        max_step=problem.T_f / 200,
    )
    x = sol.y.T  # (L, n)
    u = u_ff + (-(x - x_T) @ K.T)  # total input (L, m)
    return Rollout(t=t_eval, x=x, u=u, controller=controller, u_ff=u_ff)


# --------------------------------------------------------------------------- #
# Metric suite
# --------------------------------------------------------------------------- #
def lqr_cost(problem: ControlProblem, roll: Rollout) -> float:
    """Shared-yardstick quadratic cost J = ∫ e^T Q e + δu^T R δu dt.

    ``e = x - x_T`` and ``δu = u - u_ff`` (the control *beyond* the constant
    holding input) — this is exactly the regulation objective both the LQR and
    the game are designed against, so it is the fair common yardstick.
    """
    e = roll.x - problem.x_T
    du = roll.u - roll.u_ff
    stage = np.einsum("ti,ij,tj->t", e, problem.Q, e) + np.einsum("ti,ij,tj->t", du, problem.R, du)
    return float(np.trapezoid(stage, roll.t))


def control_effort(roll: Rollout) -> dict[str, float]:
    """Effort of the feedback control ``δu = u - u_ff`` (the part each controller chooses)."""
    du = roll.u - roll.u_ff
    energy = float(np.trapezoid(np.sum(du**2, axis=1), roll.t))
    peak = float(np.max(np.abs(du))) if du.size else 0.0
    return {"energy": energy, "peak": peak}


def settling_time(problem: ControlProblem, roll: Rollout, *, frac: float = 0.02) -> float:
    """Time after which every tracked output stays within ``frac`` of its step forever."""
    idx = np.array(problem.tracked)
    step = np.abs(problem.x_T[idx] - problem.x_0[idx])
    band = np.maximum(frac * step, 1e-9)
    err = np.abs(roll.x[:, idx] - problem.x_T[idx])
    within = np.all(err <= band, axis=1)
    # last index where it leaves the band; settling time is the next sample
    if within.all():
        return float(roll.t[0])
    left = np.where(~within)[0]
    last = left[-1]
    if last + 1 >= len(roll.t):
        return float("inf")
    return float(roll.t[last + 1])


def overshoot_pct(problem: ControlProblem, roll: Rollout) -> float:
    """Worst tracked-output overshoot beyond the target, as % of the step."""
    idx = np.array(problem.tracked)
    step = problem.x_T[idx] - problem.x_0[idx]
    over = 0.0
    for j, s in zip(idx, step):
        if abs(s) < 1e-12:
            continue
        traj = roll.x[:, j]
        beyond = (traj - problem.x_T[j]) / s  # >0 means past the target in the step direction
        over = max(over, float(np.max(beyond)) * 100.0)
    return max(over, 0.0)


def steady_state_error(problem: ControlProblem, roll: Rollout) -> float:
    return float(np.linalg.norm(roll.x[-1] - problem.x_T))


def metrics(problem: ControlProblem, roll: Rollout) -> dict[str, float]:
    eff = control_effort(roll)
    return {
        "cost": lqr_cost(problem, roll),
        "settling_time": settling_time(problem, roll),
        "overshoot_pct": overshoot_pct(problem, roll),
        "control_energy": eff["energy"],
        "control_peak": eff["peak"],
        "ss_error": steady_state_error(problem, roll),
    }


# --------------------------------------------------------------------------- #
# Robustness: cost under disturbances
# --------------------------------------------------------------------------- #
def worst_case_impulse_cost(problem: ControlProblem, controller: Controller, magnitude: float = 1.0) -> float:
    """Max shared-cost over unit impulse disturbances injected on each state rate.

    Approximates the H-infinity-flavoured worst case by sweeping a persistent
    push along each +/- state-rate direction and taking the worst resulting cost.
    """
    worst = 0.0
    n = problem.n
    for j in range(n):
        for sign in (+1.0, -1.0):
            d_vec = np.zeros(n)
            d_vec[j] = sign * magnitude

            def dist(t: float, x: FloatArray, d_vec=d_vec) -> FloatArray:
                return d_vec if t < problem.T_f * 0.25 else np.zeros(n)

            roll = simulate(problem, controller, disturbance=dist)
            worst = max(worst, lqr_cost(problem, roll))
    return worst


def monte_carlo_disturbance_cost(
    problem: ControlProblem, controller: Controller, *, trials: int = 64, sigma: float = 0.5, seed: int = 0
) -> dict[str, float]:
    """Mean / 95th-percentile shared-cost under random band-limited disturbances."""
    rng = np.random.default_rng(seed)
    switch = problem.T_f / 5.0
    costs = []
    for _ in range(trials):
        # piecewise-constant random push on the state rates
        amp = rng.normal(0.0, sigma, size=problem.n)

        def dist(t: float, x: FloatArray, amp=amp, switch=switch) -> FloatArray:
            phase = 1.0 if int(t / switch) % 2 == 0 else -1.0
            return amp * phase

        costs.append(lqr_cost(problem, simulate(problem, controller, disturbance=dist)))
    arr = np.array(costs)
    return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)), "max": float(arr.max())}


# --------------------------------------------------------------------------- #
# Plot / animation helpers
# --------------------------------------------------------------------------- #
INK = "#0f172a"
GRID = "#cbd5e1"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "cm",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.alpha": 0.6,
            "axes.edgecolor": "#94a3b8",
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
        }
    )


def comparison_panels_gif(
    problem: ControlProblem, rollouts: list[Rollout], out_path: str, *, fps: int = 25, stride: int = 4
) -> str:
    """Render an animated comparison: tracked outputs + control signals over time.

    Generic (works for any system) — system-specific animators can be added
    separately. Returns the output path.
    """
    import matplotlib.animation as animation

    _style()
    t = problem.time
    frames = range(1, len(t), stride)

    fig, (ax_x, ax_u) = plt.subplots(1, 2, figsize=(11, 4.2))
    idx = np.array(problem.tracked)

    # static targets
    for j in idx:
        ax_x.axhline(problem.x_T[j], color="#94a3b8", ls=(0, (2, 3)), lw=1.0)

    lines_x: dict = {}
    lines_u: dict = {}
    for roll in rollouts:
        for j in idx:
            (ln,) = ax_x.plot([], [], color=roll.controller.color, lw=2.0, alpha=0.9)
            lines_x[(roll.controller.name, j)] = ln
        for k in range(problem.m):
            (ln,) = ax_u.plot([], [], color=roll.controller.color, lw=1.8, alpha=0.9)
            lines_u[(roll.controller.name, k)] = ln

    all_x = np.concatenate([r.x[:, idx] for r in rollouts], axis=0)
    all_u = np.concatenate([r.u for r in rollouts], axis=0)
    ax_x.set_xlim(t[0], t[-1])
    ax_x.set_ylim(all_x.min() - 0.1 * abs(all_x.min()) - 1, all_x.max() * 1.1 + 1)
    ax_u.set_xlim(t[0], t[-1])
    upad = 0.1 * (all_u.max() - all_u.min() + 1e-9)
    ax_u.set_ylim(all_u.min() - upad, all_u.max() + upad)
    ax_x.set_title(f"{problem.name} — tracked outputs", fontweight="bold")
    ax_u.set_title("control signals", fontweight="bold")
    ax_x.set_xlabel("time [s]")
    ax_u.set_xlabel("time [s]")

    handles = [
        plt.Line2D([0], [0], color=r.controller.color, lw=2.5, label=r.controller.name) for r in rollouts
    ]
    ax_x.legend(handles=handles, loc="lower right", fontsize=9)

    def update(fi: int):
        for roll in rollouts:
            for j in idx:
                lines_x[(roll.controller.name, j)].set_data(t[:fi], roll.x[:fi, j])
            for k in range(problem.m):
                lines_u[(roll.controller.name, k)].set_data(t[:fi], roll.u[:fi, k])
        return list(lines_x.values()) + list(lines_u.values())

    fig.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
