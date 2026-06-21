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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

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
def _coil(ax, x0: float, x1: float, y: float, *, radius: float = 0.26, coils_per_unit: float = 5.5) -> None:
    """Draw a continuous, tightly-wound helical coil spring between ``(x0, y)`` and ``(x1, y)``.

    The coil is a single parametric polyline: an axial advance plus a circular
    ``(cos, sin)`` component so consecutive turns overlap into visible loops
    (a slinky-like helix). A ``sin`` envelope tapers the radius to zero at both
    ends, so the helix lands exactly on the axis at ``(xa, y)`` / ``(xb, y)`` —
    no doubling-back lead, no stray hook. ``coils_per_unit`` is shared across
    springs so every spring has the same pitch.
    """
    lead = 0.06 * (x1 - x0)
    xa, xb = x0 + lead, x1 - lead
    span = xb - xa
    n = max(11, int(round(span * coils_per_unit)))
    theta = np.linspace(0.0, 2.0 * np.pi * n, n * 100)
    # Flat-top (Tukey) envelope: uniform coil amplitude across the middle, with a
    # short cosine taper to zero over the last ~12% at each end so the helix lands
    # cleanly on the axis (no spindle shape, no stray hook).
    t_norm = theta / (2.0 * np.pi * n)
    alpha = 0.12
    env = np.ones_like(theta)
    left, right = t_norm < alpha, t_norm > 1.0 - alpha
    env[left] = 0.5 * (1.0 - np.cos(np.pi * t_norm[left] / alpha))
    env[right] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - t_norm[right]) / alpha))
    cx = xa + span * t_norm + radius * env * np.cos(theta)
    cy = y + radius * env * np.sin(theta)
    # short straight leads from the attachment points onto the (on-axis) coil ends
    ax.plot([x0, xa], [y, y], color=STRUCT, lw=2.0, solid_capstyle="round", zorder=2)
    ax.plot([xb, x1], [y, y], color=STRUCT, lw=2.0, solid_capstyle="round", zorder=2)
    ax.plot(cx, cy, color=STRUCT, lw=2.0, solid_capstyle="round", solid_joinstyle="round", zorder=2)


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
    mh, mw = 1.1, 0.95  # mass height / width (upright boxes, as in the original)
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

    # spring stiffness labels, just above the coils
    for sx in ((xw_l + m1_l) / 2, (m1_l + mw + m2_l) / 2, (m2_l + mw + xw_r) / 2):
        ax.text(sx, y_c + 0.42, r"$k$", ha="center", va="bottom", color=STRUCT, fontsize=15)

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
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [2.05, 1.0]})

    # ---- overlay: the decomposed game coincides with the monolithic LQR ---
    ax.plot(t, lqr_game.x[:, 0], color=BLUE, lw=4.2, alpha=0.30, solid_capstyle="round")
    ax.plot(t, lqr_game.x[:, 1], color=RED, lw=4.2, alpha=0.30, solid_capstyle="round")
    ax.plot(t, game.x[:, 0], color=BLUE, lw=1.7, ls=(0, (5, 3)))
    ax.plot(t, game.x[:, 1], color=RED, lw=1.7, ls=(0, (5, 3)))
    for tgt, col in ((x_T[0], BLUE), (x_T[1], RED)):
        ax.axhline(tgt, color=col, ls=(0, (1, 3)), lw=1.0, alpha=0.55)
    ax.set_xlabel(r"time  $t$  $[\mathrm{s}]$")
    ax.set_ylabel("mass position")
    ax.set_xlim(t[0], t[-1])
    handles = [
        Line2D([0], [0], color=BLUE, lw=3.5, alpha=0.5),
        Line2D([0], [0], color=RED, lw=3.5, alpha=0.5),
        Line2D([0], [0], color=INK, lw=4.2, alpha=0.30),
        Line2D([0], [0], color=INK, lw=1.7, ls=(0, (5, 3))),
    ]
    ax.legend(
        handles,
        [r"mass $x_1$", r"mass $x_2$", "LQR (thick)", "game (dashed)"],
        loc="lower right",
        ncol=2,
        fontsize=10,
    )

    # ---- difference panel: they match to numerical precision --------------
    diff = np.max(np.abs(game.x - lqr_game.x), axis=1)
    axd.semilogy(t, np.maximum(diff, 1e-16), color=GREEN, lw=2.0)
    axd.set_title("difference (game − LQR)", color=GREEN)
    axd.set_xlabel(r"time  $t$  $[\mathrm{s}]$")
    axd.set_ylabel(r"$\max_i\,\left|x_i^{\mathrm{game}}-x_i^{\mathrm{LQR}}\right|$", fontsize=10)
    axd.set_xlim(t[0], t[-1])

    fig.suptitle(
        "Tracking two coupled masses — the decomposed game is the monolithic optimum",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.86, bottom=0.16, wspace=0.30, left=0.08, right=0.97)
    fig.savefig(OUT / "masses_game_vs_lqr.png")
    plt.close(fig)


def figure_cost(lqr_cost, game_cost) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    labels = ["Monolithic\nLQR", "Modal\ngame"]
    values = [lqr_cost, game_cost]
    bars = ax.bar(labels, values, color=[AMBER, GREEN], width=0.58, edgecolor=INK, lw=1.3)
    ax.set_ylabel("cost on the shared objective")
    ax.set_title("Modal decomposition is lossless")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    gap = 100.0 * (game_cost - lqr_cost) / lqr_cost
    gap_str = "Δcost ≈ 0%" if abs(gap) < 0.05 else f"Δcost {gap:+.1f}%"
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3e}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
        )
    ax.text(
        0.5,
        max(values) * 1.15,
        f"the game recovers the optimum  ({gap_str})",
        ha="center",
        color=GREEN,
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_ylim(0, max(values) * 1.30)
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
