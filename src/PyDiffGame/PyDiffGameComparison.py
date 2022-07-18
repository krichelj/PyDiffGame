import numpy as np
from typing import Collection, Any, Optional
from abc import ABC

from PyDiffGame.LQR import ContinuousLQR, DiscreteLQR
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame
from PyDiffGame.DiscretePyDiffGame import DiscretePyDiffGame


class PyDiffGameComparison(ABC):

    def __init__(self,
                 continuous: bool,
                 args: dict[str, Any],
                 QRMs: Collection[Collection[Collection[np.array]]]):

        LQR_class = ContinuousLQR if continuous else DiscreteLQR
        GameClass = ContinuousPyDiffGame if continuous else DiscretePyDiffGame

        self._games = [LQR_class(**args)] + [GameClass(**args | {'Q': Qs, 'R': Rs, 'Ms': Ms}) for Qs, Rs, Ms in QRMs]

    def run_simulations(self, plot_state_space: Optional[bool] = True):
        for game in self._games:
            game.run_simulation(plot_state_space=plot_state_space)
