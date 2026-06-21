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
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

from PyDiffGame.examples.MassesWithSpringsComparison import MassesWithSpringsComparison

OUT = Path(__file__).resolve().parents[1] / "images" / "readme"

# A cohesive, modern palette reused across every figure.
INK = "#0f172a"  # slate-900, text / strokes
MUTED = "#64748b"  # slate-500
GRID = "#cbd5e1"  # slate-300
BLUE = "#2563eb"  # state 1
RED = "#dc2626"  # state 2
GREEN = "#059669"  # accent (game)
AMBER = "#d97706"  # accent (lqr)

# Schematic-specific colours (chosen to match the original TikZ figure).
NAVY = "#15317e"  # displacement dimension lines / labels
FORCE = "#c00000"  # force arrows / labels
STRUCT = "#3a3a3a"  # springs and structural outlines
MASS_FILL = "#f6e0e0"  # light red mass fill
MASS_EDGE = "#8b1a1a"  # dark red mass border
HATCH_FACE = "#ededed"  # hatched wall / ground fill


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
# Figure 1 — masses-and-springs system schematic (faithful to the TikZ original)
# --------------------------------------------------------------------------- #
def _coil(ax, x0: float, x1: float, y: float, *, radius: float = 0.32) -> None:
    """Draw a helical coil spring between ``(x0, y)`` and ``(x1, y)``."""
    lead = 0.08 * (x1 - x0)
    ax.plot([x0, x0 + lead], [y, y], color=STRUCT, lw=2.0, solid_capstyle="round", zorder=2)
    ax.plot([x1 - lead, x1], [y, y], color=STRUCT, lw=2.0, solid_capstyle="round", zorder=2)

    span = (x1 - lead) - (x0 + lead)
    n = max(6, int(round(span / 0.42)))
    step = span / n
    for i in range(n):
        cx = x0 + lead + step * (i + 0.5)
        ax.add_patch(
            Ellipse(
                (cx, y),
                width=step * 1.75,
                height=2 * radius,
                fill=False,
                edgecolor=STRUCT,
                lw=2.0,
                zorder=2,
            )
        )


def _hatched(ax, x: float, y: float, w: float, h: float) -> None:
    """Draw a hatched structural block (wall or ground)."""
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=HATCH_FACE,
            edgecolor=INK,
            lw=1.6,
            hatch="////",
            zorder=1,
        )
    )


def figure_schematic() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(11.0, 3.1))

    floor_top = 0.0
    mh, mw = 1.0, 1.35  # mass height / width
    y_c = floor_top + mh / 2  # spring + mass centreline
    wall_h = 1.55
    xw_l, xw_r = 1.0, 13.0  # wall inner faces

    # ground channel: left wall, floor, right wall (hatching outside)
    _hatched(ax, 0.4, floor_top, xw_l - 0.4, wall_h)  # left wall
    _hatched(ax, xw_r, floor_top, 0.6, wall_h)  # right wall
    _hatched(ax, 0.4, floor_top - 0.7, (xw_r + 0.6) - 0.4, 0.7)  # ground

    # two masses resting on the floor
    m1_l, m2_l = 3.9, 8.4
    masses = ((m1_l, r"$m$"), (m2_l, r"$m$"))

    # three springs in series: wall - m1 - m2 - wall
    _coil(ax, xw_l, m1_l, y_c)
    _coil(ax, m1_l + mw, m2_l, y_c)
    _coil(ax, m2_l + mw, xw_r, y_c)

    # spring stiffness labels
    for sx in ((xw_l + m1_l) / 2, (m1_l + mw + m2_l) / 2, (m2_l + mw + xw_r) / 2):
        ax.text(sx, floor_top + mh + 0.18, r"$k$", ha="center", va="bottom", color=STRUCT, fontsize=15)

    for ml, label in masses:
        ax.add_patch(
            Rectangle(
                (ml, floor_top),
                mw,
                mh,
                facecolor=MASS_FILL,
                edgecolor=MASS_EDGE,
                lw=2.0,
                zorder=4,
            )
        )
        ax.text(
            ml + mw / 2,
            floor_top + mh / 2,
            label,
            ha="center",
            va="center",
            color=MASS_EDGE,
            fontsize=18,
            zorder=5,
        )

    # horizontal force arrows F_1(t), F_2(t) on top of each mass
    for ml, flabel in ((m1_l, r"$F_1(t)$"), (m2_l, r"$F_2(t)$")):
        x_start = ml + mw - 0.15
        ax.add_patch(
            FancyArrowPatch(
                (x_start, floor_top + mh - 0.18),
                (x_start + 1.05, floor_top + mh - 0.18),
                arrowstyle="-|>",
                mutation_scale=20,
                lw=2.6,
                color=FORCE,
                zorder=6,
            )
        )
        ax.text(
            x_start + 0.5, floor_top + mh + 0.12, flabel, ha="center", va="bottom", color=FORCE, fontsize=15
        )

    # displacement dimension lines x_1(t), x_2(t), measured from the left wall
    dim1_y = floor_top + mh + 1.05
    dim2_y = floor_top + mh + 2.0
    # reference + per-mass drop lines (dashed, navy)
    ax.plot([xw_l, xw_l], [floor_top, dim2_y + 0.15], color=NAVY, ls=(0, (5, 4)), lw=1.3, zorder=3)
    ax.plot([m1_l, m1_l], [floor_top + mh, dim1_y], color=NAVY, ls=(0, (5, 4)), lw=1.3, zorder=3)
    ax.plot([m2_l, m2_l], [floor_top + mh, dim2_y], color=NAVY, ls=(0, (5, 4)), lw=1.3, zorder=3)
    for x_end, dy, dlabel in ((m1_l, dim1_y, r"$x_1(t)$"), (m2_l, dim2_y, r"$x_2(t)$")):
        ax.annotate(
            "",
            xy=(x_end, dy),
            xytext=(xw_l, dy),
            arrowprops={"arrowstyle": "-|>", "color": NAVY, "lw": 1.8, "mutation_scale": 18},
        )
        ax.text((xw_l + x_end) / 2, dy + 0.12, dlabel, ha="center", va="bottom", color=NAVY, fontsize=16)

    ax.set_xlim(0.1, 14.0)
    ax.set_ylim(floor_top - 0.9, dim2_y + 0.7)
    ax.set_aspect("equal")
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
