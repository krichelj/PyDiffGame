<p align="center">
    <img alt="Logo" src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/images/logo.png"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Tests](https://github.com/krichelj/PyDiffGame/actions/workflows/tests.yml/badge.svg)](https://github.com/krichelj/PyDiffGame/actions/workflows/tests.yml) [![Upload Python Package](https://github.com/krichelj/PyDiffGame/actions/workflows/python-publish.yml/badge.svg)](https://github.com/krichelj/PyDiffGame/actions/workflows/python-publish.yml) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)


  * [What is this?](#what-is-this)
  * [Installation](#installation)
  * [Quick start](#quick-start)
  * [Input parameters](#input-parameters)
  * [Tutorial](#tutorial)
  * [Testing and development](#testing-and-development)
  * [Authors](#authors)
  * [Acknowledgments](#acknowledgments)
  * [Star History](#star-history)

# What is this?

[`PyDiffGame`](https://github.com/krichelj/PyDiffGame) is a Python implementation of a Nash Equilibrium solution to differential games, based on a reduction of Game Hamilton-Jacobi-Bellman (GHJB) equations to coupled Game Algebraic and Differential Riccati equations, associated with multi-objective dynamical control systems.
The method relies on the formulation given in:

 - The thesis work "_Differential Games for Compositional Handling of Competing Control Tasks_"
   ([Research Gate](https://www.researchgate.net/publication/359819808_Differential_Games_for_Compositional_Handling_of_Competing_Control_Tasks))

 - The conference article "_Composition of Dynamic Control Objectives Based on Differential Games_"
([IEEE](https://ieeexplore.ieee.org/document/9480269) |
[Research Gate](https://www.researchgate.net/publication/353452024_Composition_of_Dynamic_Control_Objectives_Based_on_Differential_Games))

The package requires **Python >= 3.11** and is tested on CPython 3.11, 3.12, 3.13 and 3.14.

# Installation

Install the latest release from PyPI:

```bash
pip install PyDiffGame
```

To run the bundled examples (which additionally need [`python-control`](https://python-control.readthedocs.io/)):

```bash
pip install "PyDiffGame[examples]"
```

To work on the package itself, install it from source in editable mode with the development extras (tests, linter, type checker):

```bash
git clone https://github.com/krichelj/PyDiffGame.git
cd PyDiffGame
pip install -e ".[dev]"
```

# Quick start

A plain Linear-Quadratic Regulator is just a one-objective game. The continuous solver matches `scipy.linalg.solve_continuous_are` exactly:

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

For a multi-player differential game, give one [`Objective`](#input-parameters) per player. Each player owns a slice of the physical input through a decomposition matrix `M`:

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

A game is described by a system matrix `A`, a set of [`Objective`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/objective.py) objects (one per player), and an input description (`B` together with each objective's decomposition matrix `M`, or per-player matrices `Bs`). Construct one of the concrete solvers — [`ContinuousPyDiffGame`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/continuous.py) or [`DiscretePyDiffGame`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/discrete.py) — with the following parameters:

* `A` : `np.ndarray` of shape $(n, n)$
>System dynamics matrix
* `objectives` : `Sequence` of `Objective` of length $N$
>One objective per player; each carries $Q_i$, $R_{ii}$ and optionally $M_i$
* `B` : `np.ndarray` of shape $(n, m)$, optional
>Full input matrix, used together with each objective's decomposition matrix $M_i$
* `Bs` : `Sequence` of `np.ndarray`, optional
>Per-player input matrices $B_i$ of shape $(n, m_i)$, an alternative to `B` + `M`
* `x_0` : `np.ndarray` of shape $(n,)$, optional
>Initial state vector
* `x_T` : `np.ndarray` of shape $(n,)$, optional
>Final (target) state vector for signal tracking
* `T_f` : positive `float`, optional
>Finite horizon length. Omit (or pass `None`) for the infinite-horizon problem
* `P_f` : `Sequence` of `np.ndarray`, optional, default = uncoupled algebraic Riccati solutions
>Terminal condition for the Riccati equation(s)
* `L` : positive `int`, optional, default `1000`
>Number of time samples
* `eta` : positive `int`, optional, default `5`
>Number of trailing matrix norms inspected for convergence
* `epsilon_x`, `epsilon_P` : `float` in $(0, 1)$, optional
>Convergence tolerances for the state and for the Riccati matrices
* `state_variables_names` : `Sequence` of `str` of length $n$, optional
>LaTeX names (without `$`) for the state variables, used in plots
* `show_legend` : `bool`, optional, default `True`
>Whether plots include a legend
* `debug` : `bool`, optional, default `False`
>Emit verbose diagnostics while solving

An [`Objective`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/objective.py) takes:

* `Q` : `np.ndarray` of shape $(n, n)$ — symmetric positive semi-definite state weight
* `R` : `np.ndarray` of shape $(m_i, m_i)$ (or a scalar) — symmetric positive definite input weight
* `M` : `np.ndarray` of shape $(m_i, m)$, optional — decomposition matrix (`None` for a plain LQR objective)

The helpers `LQRObjective(Q, R)` and `GameObjective(Q, R, M)` are thin constructors for the two common cases.

# Tutorial

To demonstrate the package we compare a differential game against an LQR on a system of masses connected by springs:

<p align="center">
    <img align=top src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/src/PyDiffGame/examples/figures/2/two_masses_tikz.png" width="797" height="130"/>
</p>

The physical input space is decomposed along the *modal* directions of $M^{-1}K$ so that each vibration mode becomes one player of a game. The full example lives in [`MassesWithSpringsComparison.py`](https://github.com/krichelj/PyDiffGame/blob/master/src/PyDiffGame/examples/MassesWithSpringsComparison.py); here is its essence:

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

lqr_objective = [LQRObjective(Q=np.diag(q + q), R=0.25 * r * I_N)]

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

Running the bundled example end-to-end:

```bash
python -m PyDiffGame.examples.MassesWithSpringsComparison
```

produces, for two players, the state-space trajectories of the game vs. the LQR:

<p align="center">
    <img align=top src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/src/PyDiffGame/examples/figures/2/2-players_large_1.png" width="400" height="300"/>
    <img align=top src="https://raw.githubusercontent.com/krichelj/PyDiffGame/master/src/PyDiffGame/examples/figures/2/LQR_large_1.png" width="400" height="300"/>
</p>

Other worked examples in [`src/PyDiffGame/examples`](https://github.com/krichelj/PyDiffGame/tree/master/src/PyDiffGame/examples) cover an inverted pendulum on a cart, a PVTOL aircraft and a quadrotor.

# Testing and development

The package ships with a `pytest` suite that validates the solvers against analytical ground truth — the LQR solutions are checked against `scipy`'s algebraic Riccati solvers, the games against their coupled-Riccati residuals and closed-loop stability:

```bash
pip install -e ".[dev]"
pytest                                   # run the test suite
ruff check src/PyDiffGame tests          # lint
```

Continuous integration runs the linter and the full suite on Python 3.11–3.14.

# Authors

If you use this work, please cite our paper:
```
@inproceedings{pydiffgame_paper,
    author={Kricheli, Joshua Shay and Sadon, Aviran and Arogeti, Shai and Regev, Shimon and Weiss, Gera},
    booktitle={29th Mediterranean Conference on Control and Automation (MED 2021)},
    title={{Composition of Dynamic Control Objectives Based on Differential Games}},
    year={2021},
    volume={},
    number={},
    pages={298-304},
    doi={10.1109/MED51440.2021.9480269}}
```

Further details can be found in the [citation document](CITATIONS.bib).

# Acknowledgments

This research was supported in part by the Leona M. and Harry B. Helmsley Charitable Trust through the '_Agricultural, Biological and Cognitive Robotics Initiative_' ('ABC') and by the Marcus Endowment Fund both at Ben-Gurion University of the Negev, Israel.
This research was also supported by The '_Israeli Smart Transportation Research Center_' ('ISTRC') by The Technion and Bar-Ilan Universities, Israel.

<p align="center">
<a href="https://istrc.net.technion.ac.il/">
<img src="https://github.com/krichelj/PyDiffGame/blob/master/images/Logo_ISTRC_Green_English.png?raw=true" height="80" alt="ISTRC"/>
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

# Star History

<p align="center">
<a href="https://star-history.com/#krichelj/PyDiffGame&Date">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=krichelj/PyDiffGame&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=krichelj/PyDiffGame&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=krichelj/PyDiffGame&type=Date" width="600"/>
</picture>
</a>
</p>
