from typing import Sequence, Any, Optional
from abc import ABC

from PyDiffGame.LQR import ContinuousLQR, DiscreteLQR
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame
from PyDiffGame.DiscretePyDiffGame import DiscretePyDiffGame
from PyDiffGame.Objective import Objective, LQRObjective, GameObjective


class PyDiffGameComparison(ABC):

    def __init__(self,
                 continuous: bool,
                 args: dict[str, Any],
                 objectives: Sequence[Objective]):
        LQR_class = ContinuousLQR if continuous else DiscreteLQR
        GameClass = ContinuousPyDiffGame if continuous else DiscretePyDiffGame

        self._games = [LQR_class(**args | {'objective': o}) for o in objectives if isinstance(o, LQRObjective)] + \
                      [GameClass(**args | {'objectives': [o for o in objectives if isinstance(o, GameObjective)]})]

    def run_simulations(self, plot_state_space: Optional[bool] = True):
        for game in self._games:
            game.run_simulation(plot_state_space=plot_state_space)
