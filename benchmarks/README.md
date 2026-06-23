# PyDiffGame benchmark study — where a differential game actually helps

This directory is an honest, reproducible head-to-head between **PyDiffGame** and
the **[python-control](https://python-control.readthedocs.io/)** package across a
catalogue of standard control systems (carts, vehicles, aircraft, drones and
flexible structures), with rendered GIFs and a formal verification of *where*
the differential-game approach beats classical optimal control — and where it
honestly does not.

## TL;DR (the honest scientific bottom line)

1. **On a single shared quadratic cost, a differential game cannot beat a
   centralized LQR — at best it ties it.** This is not a tuning failure, it is
   theory: the centralized LQR is the optimum of that cost, and a Nash game is a
   *constrained* (decomposed) design, so its cost is `>= LQR` (price of anarchy),
   with equality when the decomposition is lossless. We verify the *lossless*
   case directly: on the masses-on-springs system PyDiffGame's modal game
   reproduces the python-control LQR **to ~5e-12 on every metric**, disturbances
   included. No boost — and we say so.

2. **The real, formally-verifiable win is robustness.** Robust (H-infinity)
   control *is* a differential game — the saddle point of a controller-vs-
   adversarial-disturbance zero-sum game. PyDiffGame now ships this as
   `ContinuousHInfinityControl`, and it **provably reduces the worst-case
   disturbance gain** an LQR leaves on the table — at a documented nominal-cost
   price, and only when the plant has worst-case gain to recover.

## What is measured

For every system we design two state-feedback controllers on the **same**
weights `(Q, R)`:

| controller | what it optimizes |
| --- | --- |
| `control.lqr` (python-control) | nominal cost (no disturbance) |
| `PyDiffGame.ContinuousHInfinityControl` | worst-case disturbance gain (the game) |

and report, on the same closed loop:

- **`‖G_zw‖∞`** — the closed-loop worst-case L2 gain from the disturbance to the
  weighted performance output `z = [Q^{1/2}x; R^{1/2}u]` (the formal robustness
  metric; lower is more robust). Computed slycot-free by a refined frequency
  sweep.
- **time-domain peak** of the output under the single worst-case sinusoidal
  disturbance (at the LQR's most vulnerable frequency).
- **nominal LQ cost penalty** — how much nominal performance the robust design
  gives up (always `>= 0`, since the LQR is the nominal optimum; this is the
  *price* of robustness, reported honestly alongside the gain).

## Results

See [`results/ROBUSTNESS_REPORT.md`](results/ROBUSTNESS_REPORT.md) for the full
auto-generated table, and `results/robust_<system>.gif` for each animation
(left: time response to the worst-case disturbance; right: the `σmax(ω)` curves
whose peak *is* `‖G_zw‖∞`).

Headline (honest): all 10 systems show a *relative* worst-case-gain reduction,
but only the ones with a **non-negligible absolute gain** matter in practice —
**7/10 are practically significant** (inverted pendulum +35%, PVTOL/quadrotor
+26%, flexible two-mass / cart / DC motor ~+22%, ...), exactly the lightly-damped
and unstable plants where the LQR leaves a sharp resonant peak. The two cars
(cruise, bicycle) have an absolute worst-case gain of ~0 — the disturbance is
already rejected by *any* reasonable controller — so their real relative
reductions are **not practically meaningful**, and we say so rather than headline
a "10/10 win". (The bicycle's earlier apparent "tie" turned out to be a `γ*`
numerical artifact, caught by the methodology review and fixed; the corrected
result is a real-but-immaterial relative reduction.)

## Reproduce

```bash
uv run --extra dev python -m benchmarks.run_masses          # nominal: game == LQR (lossless)
uv run --extra dev python -m benchmarks.robust_compare      # one robust comparison + GIF
uv run --extra dev python -m benchmarks.run_robust_suite    # the full 10-system suite + report
```

## Rigor

The catalogue models were verified entry-for-entry against the controls /
vehicle-dynamics / flight-dynamics literature, and the comparison methodology
(GARE solve, `γ*` search, the worst-case-gain metric, the nominal-cost
accounting, and the fairness of scoring both controllers on the same output) was
adversarially reviewed; the review hardened PyDiffGame's `γ*` search against a
boundary numerical instability (now regression-tested).
