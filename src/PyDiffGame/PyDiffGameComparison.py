from typing import Sequence, Any, Optional
from abc import ABC

import numpy as np

from PyDiffGame.LQR import ContinuousLQR, DiscreteLQR
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame
from PyDiffGame.DiscretePyDiffGame import DiscretePyDiffGame
from PyDiffGame.Objective import Objective, LQRObjective, GameObjective


class PyDiffGameComparison(ABC):
    """
    Differential games comparison abstract base class


    Parameters
    ----------
    args: sequence of 2-d np.arrays of len(N), each array B_i of shape(n, m_i)
        System input matrices for each control objective
    objectives: sequence of Objective objects
        Desired objectives for the game
    continuous: boolean, optional
        Indicates whether to consider a continuous or discrete control design
    """

    def __init__(self,
                 args: dict[str, Any],
                 objectives: Sequence[Objective],
                 continuous: bool = True):
        LQR_class = ContinuousLQR if continuous else DiscreteLQR
        GameClass = ContinuousPyDiffGame if continuous else DiscretePyDiffGame

        self.__args = args
        self._verify_input()

        self._games = [LQR_class(**self.__args | {'objective': o}) for o in objectives if isinstance(o, LQRObjective)] \
                      + [GameClass(**self.__args | {'objectives': [o for o in objectives
                                                                        if isinstance(o, GameObjective)]})]

    def _verify_input(self):
        """
        Input checking method

        Raises
        ------
        Case-specific errors
        """

        for c in ['A', 'B']:
            c_name = 'dynamics' if c == 'A' else 'input'
            if c not in self.__args.keys():
                raise ValueError(f"Passed arguments must contain the {c_name} matrix {c}")
            if isinstance(self.__args[c], type(np.array)):
                raise ValueError(f'The {c_name} matrix {c} must be a numpy array')

    def run_simulations(self, plot_state_space: Optional[bool] = True, run_animation: Optional[bool] = True):
        for game in self._games:
            game.run_simulation(plot_state_space=plot_state_space)

            if run_animation:
                game
