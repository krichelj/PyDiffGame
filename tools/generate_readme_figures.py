"""Regenerate the figures used in the project README.

Run with::

    uv run python tools/generate_readme_figures.py

All figures are written to ``images/readme/`` as PNGs. The trajectory and cost
figures are produced from the *actual* v2.0 solver output (no precomputed
assets), so they always reflect the current behaviour of the package.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.patches import FancyArrowPatch, Rectangle

from PyDiffGame.examples.MassesWithSpringsComparison import MassesWithSpringsComparison

OUT = Path(__file__).resolve().parents[1] / "images" / "readme"

# A cohesive, modern palette reused across every figure.
INK = "#0f172a"  # slate-900, text / strokes
MUTED = "#64748b"  # slate-500
GRID = "#cbd5e1"  # slate-300
BLUE = "#2563eb"  # mode / state 1
RED = "#dc2626"  # mode / state 2
GREEN = "#059669"  # accent (game)
AMBER = "#d97706"  # accent (lqr)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 12,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "cm",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.alpha": 0.7,
            "grid.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.prop_cycle": cycler(color=[BLUE, RED, GREEN, AMBER]),
        }
    )


# --------------------------------------------------------------------------- #
# Figure 1 — masses-and-springs system schematic
# --------------------------------------------------------------------------- #
def _spring(ax, x0: float, x1: float, y: float, coils: int = 6, amp: float = 0.16) -> None:
    """Draw a zig-zag spring between (x0, y) and (x1, y)."""
    lead = 0.12 * (x1 - x0)
    xs = [x0, x0 + lead]
    ys = [y, y]
    n = coils * 2
    body = np.linspace(x0 + lead, x1 - lead, n + 1)
    for i, xb in enumerate(body[1:-1], start=1):
        xs.append(xb)
        ys.append(y + amp * (1 if i % 2 else -1))
    xs += [x1 - lead, x1]
    ys += [y, y]
    ax.plot(xs, ys, color=INK, lw=1.8, solid_joinstyle="round", zorder=2)


def _wall(ax, x: float, y: float, h: float, side: str) -> None:
    ax.plot([x, x], [y - h, y + h], color=INK, lw=2.5, zorder=3)
    n = 7
    for yy in np.linspace(y - h, y + h, n):
        dx = -0.18 if side == "left" else 0.18
        ax.plot([x, x + dx], [yy, yy + 0.16], color=INK, lw=1.2, zorder=3)


def figure_schematic() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9.2, 2.4))
    y = 0.0
    box_w, box_h = 1.1, 0.9
    # geometry: wall - spring - m1 - spring - m2 - spring - wall
    xL, xR = 0.0, 10.0
    c1, c2 = 3.4, 6.6
    _wall(ax, xL, y, 1.0, "left")
    _wall(ax, xR, y, 1.0, "right")
    _spring(ax, xL, c1 - box_w / 2, y)
    _spring(ax, c1 + box_w / 2, c2 - box_w / 2, y)
    _spring(ax, c2 + box_w / 2, xR, y)

    for cx, label, colour in ((c1, r"$m_1$", BLUE), (c2, r"$m_2$", RED)):
        ax.add_patch(
            Rectangle(
                (cx - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                facecolor=colour,
                edgecolor=INK,
                lw=1.6,
                alpha=0.92,
                zorder=4,
            )
        )
        ax.text(cx, y, label, ha="center", va="center", color="white", fontsize=15, zorder=5)
        # control input arrow
        ax.add_patch(
            FancyArrowPatch(
                (cx, y + box_h / 2 + 0.12),
                (cx, y + box_h / 2 + 0.95),
                arrowstyle="-|>",
                mutation_scale=16,
                lw=2.0,
                color=GREEN,
                zorder=5,
            )
        )
    ax.text(c1, y + box_h / 2 + 1.12, r"$u_1$", ha="center", color=GREEN, fontsize=14)
    ax.text(c2, y + box_h / 2 + 1.12, r"$u_2$", ha="center", color=GREEN, fontsize=14)
    # spring stiffness labels
    for sx in (1.7, 5.0, 8.3):
        ax.text(sx, y - 0.55, r"$k$", ha="center", color=MUTED, fontsize=12)

    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(-1.4, 2.2)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "masses_schematic.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Solve the benchmark once and reuse for the trajectory + cost figures
# --------------------------------------------------------------------------- #
def _solve_benchmark():
    m, k, r = 50.0, 10.0, 1.0
    q = [500.0, 2000.0]
    x_0 = np.array([10.0, 20.0, 0.0, 0.0])
    x_T = x_0 * 10.0
    comp = MassesWithSpringsComparison(N=2, m=m, k=k, q=q, r=r, x_0=x_0, x_T=x_T, T_f=25.0, L=300)
    comp.run(plot_state_spaces=False)
    lqr_game, game = comp.games
    lqr_cost, game_cost = comp.costs()
    return lqr_game, game, lqr_cost, game_cost, x_T


def figure_trajectories(lqr_game, game, lqr_cost, game_cost, x_T) -> None:
    _style()
    t = lqr_game.forward_time
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    panels = (
        (axes[0], lqr_game, "Monolithic LQR", lqr_cost, AMBER),
        (axes[1], game, "Modal differential game", game_cost, GREEN),
    )
    for ax, g, title, cost, accent in panels:
        ax.plot(t, g.x[:, 0], color=BLUE, lw=2.2, label=r"$x_1(t)$")
        ax.plot(t, g.x[:, 1], color=RED, lw=2.2, label=r"$x_2(t)$")
        ax.axhline(x_T[0], color=BLUE, ls=(0, (4, 4)), lw=1.2, alpha=0.7)
        ax.axhline(x_T[1], color=RED, ls=(0, (4, 4)), lw=1.2, alpha=0.7)
        ax.set_title(title, color=accent)
        ax.set_xlabel(r"time  $t$  $[\mathrm{s}]$")
        ax.set_xlim(t[0], t[-1])
        ax.text(
            0.97,
            0.05,
            rf"cost $= {cost:.2e}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=accent,
            fontsize=11,
            fontweight="bold",
        )
    axes[0].set_ylabel("mass position")
    axes[1].legend(loc="lower right", bbox_to_anchor=(1.0, 0.16), fontsize=11)
    fig.suptitle(
        "Tracking two coupled masses to a target — LQR vs. differential game",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    # dashed = targets
    fig.text(0.5, -0.02, "dashed lines: per-mass targets $x_T$", ha="center", color=MUTED, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "masses_game_vs_lqr.png")
    plt.close(fig)


def figure_cost(lqr_cost, game_cost) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    labels = ["Monolithic\nLQR", "Modal\ngame"]
    values = [lqr_cost, game_cost]
    bars = ax.bar(labels, values, color=[AMBER, GREEN], width=0.6, edgecolor=INK, lw=1.2, alpha=0.92)
    ax.set_ylabel("cost on the LQR objective")
    ax.set_title("Near-optimal cost from a fully decomposed design")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    overhead = 100.0 * (game_cost - lqr_cost) / lqr_cost
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.2e}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
        )
    ax.annotate(
        f"+{overhead:.0f}% vs. optimal",
        xy=(1, game_cost),
        xytext=(0.5, max(values) * 1.12),
        ha="center",
        color=GREEN,
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    fig.savefig(OUT / "masses_cost.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_schematic()
    lqr_game, game, lqr_cost, game_cost, x_T = _solve_benchmark()
    figure_trajectories(lqr_game, game, lqr_cost, game_cost, x_T)
    figure_cost(lqr_cost, game_cost)
    print(f"Wrote figures to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(" -", p.relative_to(OUT.parents[1]))


if __name__ == "__main__":
    main()
