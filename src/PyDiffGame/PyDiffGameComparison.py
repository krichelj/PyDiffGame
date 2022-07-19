from typing import Sequence, Any, Optional
from abc import ABC

from PyDiffGame.LQR import ContinuousLQR, DiscreteLQR
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame
from PyDiffGame.DiscretePyDiffGame import DiscretePyDiffGame
from PyDiffGame.Objective import GameObjective


class PyDiffGameComparison(ABC):

    def __init__(self,
                 continuous: bool,
                 args: dict[str, Any],
                 Os: Sequence[GameObjective]):
        LQR_class = ContinuousLQR if continuous else DiscreteLQR
        GameClass = ContinuousPyDiffGame if continuous else DiscretePyDiffGame

        self._games = [LQR_class(**args)] + [GameClass(**args | {'Q': Qs, 'R': Rs, 'Ms': Ms}) for Qs, Rs, Ms in Os]

    def run_simulations(self, plot_state_space: Optional[bool] = True):
        for game in self._games:
            game.run_simulation(plot_state_space=plot_state_space)
