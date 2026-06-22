<p align="center">
    <img alt="PyDiffGame logo" src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/logo.gif" width="420"/>
</p>

<p align="center">
    <i>Nash-equilibrium control for multi-objective dynamical systems, built on coupled Riccati equations.</i>
</p>

<p align="center">
<a href="https://github.com/krichelj/PyDiffGame/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/krichelj/PyDiffGame/actions/workflows/tests.yml/badge.svg?branch=master"></a>
<a href="https://pypi.org/project/PyDiffGame/"><img alt="PyPI" src="https://img.shields.io/pypi/v/PyDiffGame.svg"></a>
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg"></a>
<a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://mypy-lang.org/"><img alt="Checked with mypy" src="https://img.shields.io/badge/mypy-checked-2a6db2.svg"></a>
<a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/managed%20with-uv-261230.svg"></a>
</p>

---

* [What is this?](#what-is-this)
* [Why differential games?](#why-differential-games)
* [Installation](#installation)
* [Quick start](#quick-start)
* [Input parameters](#input-parameters)
* [Tutorial: masses on springs](#tutorial-masses-on-springs)
* [More examples](#more-examples)
* [Testing and development](#testing-and-development)
* [Citing](#citing)
* [Acknowledgments](#acknowledgments)
* [Star history](#star-history)

# What is this?

[`PyDiffGame`](https://github.com/krichelj/PyDiffGame) is a Python implementation of a
**Nash-equilibrium solution to differential games**. It reduces the Game
Hamilton–Jacobi–Bellman (GHJB) equations to a set of *coupled* Game Algebraic and
Differential Riccati equations, and solves them to synthesize feedback controllers for
multi-objective dynamical control systems.

In one sentence: where a classical Linear-Quadratic Regulator (LQR) folds every control
task into **one** quadratic cost and solves **one** Riccati equation, `PyDiffGame` keeps
each task as a separate *player*, and solves the coupled Riccati system whose fixed point
is their Nash equilibrium. A plain LQR is simply the one-player special case.

The method follows the formulation in:

- The thesis _“Differential Games for Compositional Handling of Competing Control Tasks”_
  ([ResearchGate](https://www.researchgate.net/publication/359819808_Differential_Games_for_Compositional_Handling_of_Competing_Control_Tasks))
- The conference paper _“Composition of Dynamic Control Objectives Based on Differential Games”_
  ([IEEE](https://ieeexplore.ieee.org/document/9480269) ·
  [ResearchGate](https://www.researchgate.net/publication/353452024_Composition_of_Dynamic_Control_Objectives_Based_on_Differential_Games))

The package requires **Python ≥ 3.11** and is tested on CPython 3.11, 3.12, 3.13 and 3.14.

# Why differential games?

| Classical LQR | `PyDiffGame` |
| --- | --- |
| A single, hand-blended cost $\int x^\top Q x + u^\top R u$ | One objective **per task / player**, each with its own $Q_i, R_i$ |
| One Algebraic Riccati Equation | A **coupled** system of Riccati equations solved to its Nash fixed point |
| Re-tune the whole weight matrix to add a task | **Compose** tasks by adding a player — the others keep their own cost |
| Continuous time only, by convention | Continuous **and** discrete time, finite or infinite horizon |

`PyDiffGame` ships both regulation (drive the state to the origin) and **signal tracking**
(drive the state to a target $x_T$), a one-call **LQR-vs-game comparison** harness, cost
accounting on a common yardstick, and ready-made plotting.

# Installation

PyDiffGame is published on [PyPI](https://pypi.org/project/PyDiffGame/) and is
managed with [**uv**](https://docs.astral.sh/uv/). Add it to your project:

```bash
uv add PyDiffGame
```

To run the bundled examples (which additionally need
[`python-control`](https://python-control.readthedocs.io/)):

```bash
uv add "PyDiffGame[examples]"
```

<details>
<summary><b>Prefer pip?</b> It works as a fallback.</summary>

```bash
pip install PyDiffGame
pip install "PyDiffGame[examples]"   # with the examples extra
```

</details>

To work on the package itself, clone it and sync the locked development
environment with uv:

```bash
git clone https://github.com/krichelj/PyDiffGame.git
cd PyDiffGame
uv sync --extra dev          # creates .venv with the exact locked dependencies
uv run pre-commit install    # enable the formatting / lint / type-check hooks
```

Then run anything through `uv run` (`uv run pytest`,
`uv run python -m PyDiffGame.examples.MassesWithSpringsComparison`, …). Pip users
can instead `pip install -e ".[dev]"`, though uv is recommended for the exact
locked environment.

# Quick start

A plain Linear-Quadratic Regulator is just a one-objective game. The continuous solver
matches `scipy.linalg.solve_continuous_are` exactly:

```python
import numpy as np
from PyDiffGame import ContinuousLQR

A = np.array([[0.0, 1.0],
              [0.0, 0.0]])
B = np.array([[0.0],
              [1.0]])

lqr = ContinuousLQR(A=A, B=B, Q=np.eye(2), R=1.0).solve()

print(lqr.K[0])                     # optimal feedback gain
print(lqr.is_closed_loop_stable())  # True
```

For a multi-player differential game, give one [`Objective`](#input-parameters) per
player. Each player owns a slice of the physical input through a decomposition matrix `M`:

```python
import numpy as np
from PyDiffGame import ContinuousPyDiffGame, GameObjective

A = np.array([[0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0, 0.0]])
B = np.eye(4)[:, [1, 3]]

objectives = [
    GameObjective(Q=np.diag([1.0, 0.1, 0.0, 0.0]), R=1.0, M=np.array([[1.0, 0.0]])),
    GameObjective(Q=np.diag([0.0, 0.0, 1.0, 0.1]), R=1.0, M=np.array([[0.0, 1.0]])),
]

game = ContinuousPyDiffGame(A=A, objectives=objectives, B=B).solve()

# Each converged P_i drives its coupled algebraic Riccati residual to ~0:
print(max(np.max(np.abs(r)) for r in game.algebraic_riccati_residuals()))  # ~1e-14
print(game.is_closed_loop_stable())                                        # True
```

# Input parameters

A game is described by a system matrix `A`, a set of
[`Objective`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/objective.py)
objects (one per player), and an input description (`B` together with each objective's
decomposition matrix `M`, or per-player matrices `Bs`). Construct one of the concrete
solvers —
[`ContinuousPyDiffGame`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/continuous.py)
or
[`DiscretePyDiffGame`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/discrete.py)
— with the following parameters:

| Parameter | Type | Meaning |
| --- | --- | --- |
| `A` | `np.ndarray` $(n, n)$ | System dynamics matrix |
| `objectives` | `Sequence[Objective]` of length $N$ | One objective per player; each carries $Q_i$, $R_{ii}$ and optionally $M_i$ |
| `B` | `np.ndarray` $(n, m)$, optional | Full input matrix, used with each objective's decomposition matrix $M_i$ |
| `Bs` | `Sequence[np.ndarray]`, optional | Per-player input matrices $B_i$ of shape $(n, m_i)$ — an alternative to `B` + `M` |
| `x_0` | `np.ndarray` $(n,)$, optional | Initial state vector |
| `x_T` | `np.ndarray` $(n,)$, optional | Final (target) state for signal tracking |
| `T_f` | positive `float`, optional | Finite-horizon length; omit (or `None`) for the infinite-horizon problem |
| `P_f` | `Sequence[np.ndarray]`, optional | Terminal Riccati condition (default: uncoupled algebraic Riccati solutions) |
| `L` | positive `int`, default `1000` | Number of time samples |
| `eta` | positive `int`, default `5` | Number of trailing matrix norms inspected for convergence |
| `epsilon_x`, `epsilon_P` | `float` in $(0, 1)$, optional | Convergence tolerances for the state and the Riccati matrices |
| `state_variables_names` | `Sequence[str]` of length $n$, optional | LaTeX names (without `$`) for state variables, used in plots |
| `show_legend` | `bool`, default `True` | Whether plots include a legend |
| `debug` | `bool`, default `False` | Emit verbose diagnostics while solving |

An [`Objective`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/objective.py) takes:

* `Q` — `np.ndarray` $(n, n)$, symmetric positive **semi**-definite state weight
* `R` — `np.ndarray` $(m_i, m_i)$ (or a scalar), symmetric positive definite input weight
* `M` — `np.ndarray` $(m_i, m)$, optional decomposition matrix (`None` for a plain LQR objective)

The helpers `LQRObjective(Q, R)` and `GameObjective(Q, R, M)` are thin constructors for
the two common cases.

# Tutorial: masses on springs

To show the package in action we compare a differential game against an LQR on a chain of
masses connected by springs — a textbook coupled, oscillatory system:

<p align="center">
    <img alt="Two masses connected by springs between two walls" src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/readme/masses_schematic.png" width="760"/>
</p>

The physical input space is decomposed along the **modal** directions of $M^{-1}K$, so each
vibration mode becomes one player of a game. The full example lives in
[`MassesWithSpringsComparison.py`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/examples/MassesWithSpringsComparison.py);
here is its essence:

```python
import numpy as np
from PyDiffGame import GameObjective, LQRObjective, PyDiffGameLQRComparison

N, m, k, r = 2, 50.0, 10.0, 1.0
q = [500.0, 2000.0]                       # per-mode state weights

I_N, Z_N = np.eye(N), np.zeros((N, N))
mass_inv_stiffness = (1.0 / m) * k * (2 * I_N - np.eye(N, k=1) - np.eye(N, k=-1))

# Modal decomposition (orthonormal, so the modal transform is orthogonal):
_, eigenvectors = np.linalg.eigh(mass_inv_stiffness)
Ms = [eigenvectors[:, i].reshape(1, N) for i in range(N)]
modal_to_state = np.kron(np.eye(2), np.concatenate(Ms, axis=0))

A = np.block([[Z_N, I_N], [-mass_inv_stiffness, Z_N]])
B = np.block([[Z_N], [(1.0 / m) * I_N]])

game_objectives = []
for i, (q_i, M_i) in enumerate(zip(q, Ms)):
    modal_weight = np.diag([0.0] * i + [q_i] + [0.0] * (N - 1) + [q_i] + [0.0] * (N - i - 1))
    game_objectives.append(GameObjective(Q=modal_to_state.T @ modal_weight @ modal_to_state, R=r, M=M_i))

# The LQR baseline optimizes exactly the aggregate of the game's objectives
# (sum of the per-mode Q_i, and the matching physical input weight r * I), so
# it is the genuine monolithic optimum to compare the decomposed game against:
modal_weight_total = np.diag(q + q)
lqr_objective = [LQRObjective(Q=modal_to_state.T @ modal_weight_total @ modal_to_state, R=r * I_N)]

x_0 = np.array([10.0, 20.0, 0.0, 0.0])
comparison = PyDiffGameLQRComparison(
    A=A, B=B,
    games_objectives=[lqr_objective, game_objectives],
    x_0=x_0, x_T=x_0 * 10.0, T_f=25.0, L=300,
)

comparison.run(plot_state_spaces=True)
lqr_cost, game_cost = comparison.costs()
print(f"LQR cost = {lqr_cost:.4g},  game cost = {game_cost:.4g}")
```

Run the bundled example end-to-end:

```bash
uv run python -m PyDiffGame.examples.MassesWithSpringsComparison
```

The monolithic LQR is, by construction, the optimal controller for the *aggregate* of the
game's objectives. The **fully decomposed** modal game — where each vibration mode is
solved as an independent player, coupled only through the shared dynamics — reproduces that
monolithic optimum **to numerical precision**: the two state trajectories coincide
(they differ by ~10⁻⁷) and the costs are equal:

<p align="center">
    <img alt="State trajectories: the decomposed game reproduces the monolithic LQR" src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/readme/masses_game_vs_lqr.png" width="860"/>
</p>

<p align="center">
    <img alt="Cost comparison: the modal game recovers the LQR optimum" src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/readme/masses_cost.png" width="440"/>
</p>

For this modally-decoupled system the decomposition is **lossless** — and it buys
**compositionality**: you can add, drop or re-weight a control task by editing a single
player, without re-tuning one monolithic cost matrix.

> The figures above are regenerated from the live solver by
> [`tools/generate_readme_figures.py`](https://github.com/krichelj/PyDiffGame/blob/master/tools/generate_readme_figures.py)
> (`uv run python tools/generate_readme_figures.py`), so they always match the current code.

# More examples

The [`src/PyDiffGame/examples`](https://github.com/krichelj/PyDiffGame/tree/master/src/PyDiffGame/examples)
directory contains further worked comparisons:

| Example | System |
| --- | --- |
| [`MassesWithSpringsComparison.py`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/examples/MassesWithSpringsComparison.py) | Chain of masses coupled by springs (the tutorial above) |
| [`InvertedPendulumComparison.py`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/examples/InvertedPendulumComparison.py) | Inverted pendulum on a cart |
| [`PVTOL.py`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/examples/PVTOL.py) · [`PVTOLComparison.py`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/examples/PVTOLComparison.py) | Planar vertical take-off & landing aircraft |
| [`QuadRotorControl.py`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/examples/QuadRotorControl.py) | Quadrotor attitude / position control |

# Testing and development

The package ships with a `pytest` suite that validates the solvers against analytical
ground truth — LQR solutions are checked against `scipy`'s algebraic Riccati solvers, and
the games against their coupled-Riccati residuals and closed-loop stability. All tooling
runs through uv:

```bash
uv sync --extra dev
uv run pytest                                # run the test suite
uv run ruff format src/PyDiffGame tests      # auto-format (black-compatible)
uv run ruff check  src/PyDiffGame tests      # lint
uv run mypy        src/PyDiffGame            # type-check
```

Continuous integration runs the formatter check, linter, type checker and full suite on
Python 3.11–3.14. See [CONTRIBUTING.md](https://github.com/krichelj/PyDiffGame/blob/master/CONTRIBUTING.md) for details.

# Citing

If you use this work, please cite our paper:

```bibtex
@inproceedings{pydiffgame_paper,
    author={Kricheli, Joshua Shay and Sadon, Aviran and Arogeti, Shai and Regev, Shimon and Weiss, Gera},
    booktitle={29th Mediterranean Conference on Control and Automation (MED 2021)},
    title={{Composition of Dynamic Control Objectives Based on Differential Games}},
    year={2021},
    pages={298-304},
    doi={10.1109/MED51440.2021.9480269}}
```

Further details can be found in the [citation document](https://github.com/krichelj/PyDiffGame/blob/master/CITATIONS.bib).

# Acknowledgments

This research was supported in part by the Leona M. and Harry B. Helmsley Charitable Trust
through the _‘Agricultural, Biological and Cognitive Robotics Initiative’_ (‘ABC’) and by
the Marcus Endowment Fund, both at Ben-Gurion University of the Negev, Israel. It was also
supported by The _‘Israeli Smart Transportation Research Center’_ (‘ISTRC’) by The Technion
and Bar-Ilan Universities, Israel.

<p align="center">
<a href="https://istrc.net.technion.ac.il/">
<img src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/Logo_ISTRC_Green_English.png" height="80" alt="ISTRC"/>
</a>
&emsp;
<a href="https://in.bgu.ac.il/en/Pages/default.aspx">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZ8GbtJiX8lNUygX7-inRBuWESK438jWbRjQ&s">
  <source media="(prefers-color-scheme: light)" srcset="https://tamrur.bgu.ac.il/restore/BGU.sig.png">
  <img alt="Ben-Gurion University of the Negev" src="https://tamrur.bgu.ac.il/restore/BGU.sig.png" height="80">
</picture>
</a>
&emsp;
<a href="https://helmsleytrust.org/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://helmsleytrust.org/wp-content/themes/helmsley/assets/img/helmsley-charitable-trust-logo-white.png">
  <source media="(prefers-color-scheme: light)" srcset="https://helmsleytrust.org/wp-content/themes/helmsley/assets/img/helmsley-charitable-trust-logo-blue.png">
  <img alt="Helmsley Charitable Trust" src="https://helmsleytrust.org/wp-content/themes/helmsley/assets/img/helmsley-charitable-trust-logo-blue.png" height="80">
</picture>
</a>
</p>

# Star history

<p align="center">
<a href="https://star-history.com/#krichelj/PyDiffGame&Date">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=krichelj/PyDiffGame&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=krichelj/PyDiffGame&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=krichelj/PyDiffGame&type=Date" width="600"/>
</picture>
</a>
</p>
