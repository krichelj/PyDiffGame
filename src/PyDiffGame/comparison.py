"""Compare several control designs (e.g. an LQR baseline vs. a multi-player game).

A :class:`PyDiffGameLQRComparison` builds one game per objective group, all
sharing the same model (``A``, ``B``, ``x_0`` ...), then solves, simulates and
optionally plots them and reports their costs on a common yardstick.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from PyDiffGame._typing import ArrayLike
from PyDiffGame.base import PyDiffGame
from PyDiffGame.continuous import ContinuousPyDiffGame
from PyDiffGame.discrete import DiscretePyDiffGame
from PyDiffGame.objective import Objective


class PyDiffGameLQRComparison:
    """Build and compare several games that share the same linear model.

    Parameters
    ----------
    A:
        Shared dynamics matrix.
    B:
        Shared input matrix.
    games_objectives:
        A sequence of objective groups; one game is built per group.  A group
        with a single LQR objective yields an LQR; a group of game objectives
        (each carrying an ``M``) yields a multi-player game.
    continuous:
        Whether to build continuous- or discrete-time games.
    **game_kwargs:
        Forwarded to every game (``x_0``, ``x_T``, ``T_f``, ``L`` ...).
    """

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        games_objectives: Sequence[Sequence[Objective]],
        *,
        continuous: bool = True,
        **game_kwargs,
    ) -> None:
        game_class = ContinuousPyDiffGame if continuous else DiscretePyDiffGame
        self._games: list[PyDiffGame] = [
            game_class(A, list(objectives), B=B, **game_kwargs)
            for objectives in games_objectives
        ]
        self._lqr_objective: Objective | None = next(
            (game[0] for game in self._games if game.is_lqr), None
        )

    @property
    def games(self) -> list[PyDiffGame]:
        return self._games

    def are_all_controllable(self) -> bool:
        return all(game.is_controllable() for game in self._games)

    def solve(self) -> PyDiffGameLQRComparison:
        for game in self._games:
            game.solve()
        return self

    def run(
        self,
        *,
        plot_state_spaces: bool = True,
        save_figure: bool = False,
        figure_filename: str | Callable[[PyDiffGame], str] = PyDiffGame.default_figures_filename,
    ) -> PyDiffGameLQRComparison:
        """Solve, simulate and (optionally) plot every game in the comparison."""

        for game in self._games:
            name = figure_filename(game) if callable(figure_filename) else figure_filename
            game.run(
                plot_state_space=plot_state_spaces,
                save_figure=save_figure,
                figure_filename=name,
            )
        return self

    def __call__(self, **kwargs) -> PyDiffGameLQRComparison:
        return self.run(**kwargs)

    def costs(self, objective: Objective | None = None) -> list[float]:
        """Cost of every (solved, simulated) game measured by a common objective."""

        objective = objective or self._lqr_objective
        if objective is None:
            raise ValueError("no LQR objective available; pass one explicitly")
        return [game.cost(objective) for game in self._games]

    @staticmethod
    def run_multiprocess(
        worker: Callable[..., object],
        values: Sequence[Sequence[object]],
    ) -> None:
        """Run ``worker`` over the Cartesian product of ``values`` in parallel."""

        start = time.perf_counter()
        combinations = list(itertools.product(*values))
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(worker, *combo) for combo in combinations]
            for future in tqdm(futures, total=len(futures)):
                future.result()
        print(f"Total time: {time.perf_counter() - start:.3f} s")

    def __len__(self) -> int:
        return len(self._games)

    def __getitem__(self, i: int) -> PyDiffGame:
        return self._games[i]


__all__ = ["PyDiffGameLQRComparison"]
