from typing import Sequence, Any, Optional
from abc import ABC

import numpy as np

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.LQR import ContinuousLQR, DiscreteLQR
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame
from PyDiffGame.DiscretePyDiffGame import DiscretePyDiffGame
from PyDiffGame.Objective import LQRObjective, GameObjective


class LQRPyDiffGameComparison(ABC):
    """
    LQR vs. differential game comparison abstract base class


    Parameters
    ----------
    args: sequence of 2-d np.arrays of len(N), each array B_i of shape(n, m_i)
        System input matrices for each control objective
    LQR_objective: LQRObjective object
        Desired LQR objective for the comparison
    game_objectives: Sequence of GameObjective objects
        Desired game objectives for the comparison
    continuous: boolean, optional
        Indicates whether to consider a continuous or discrete control design
    """

    def __init__(self,
                 args: dict[str, Any],
                 LQR_objective: LQRObjective,
                 game_objectives: Sequence[GameObjective],
                 continuous: bool = True):
        LQR_class = ContinuousLQR if continuous else DiscreteLQR
        GameClass = ContinuousPyDiffGame if continuous else DiscretePyDiffGame

        self.__args = args
        self._verify_input()

        self._LQR = LQR_class(**self.__args | {'objective': LQR_objective})
        self._game = GameClass(**self.__args | {'objectives': [o for o in game_objectives]})

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

    def run_simulations(self,
                        plot_state_spaces: Optional[bool] = True,
                        run_animations: Optional[bool] = True,
                        calculate_costs: Optional[bool] = True):


        def run_simulation(tool: PyDiffGame, LQR: bool):
            tool.run_simulation(plot_state_space=plot_state_spaces)

            if run_animations:
                self.run_animation(LQR=LQR)

        run_simulation(tool=self._LQR,
                       LQR=True)

        run_simulation(tool=self._game,
                       LQR=False)

        if calculate_costs:
            LQR_cost = self._LQR.get_costs()
            game_costs = self._game.get_costs()
            game_total_cost = sum(game_costs)
            costs_differences = LQR_cost - game_total_cost

            print(costs_differences)

    def _simulate_non_linear_system(self,
                                    *args):
        """
        Simulates the corresponding non-linear system's progression through time
        """

        pass

    def run_animation(self,
                      LQR: bool):
        """
        Animates the state progression of the system
        """

        pass
