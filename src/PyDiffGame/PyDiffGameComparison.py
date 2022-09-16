from typing import Sequence, Any, Optional, Callable
from abc import ABC

import numpy as np

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame
from PyDiffGame.DiscretePyDiffGame import DiscretePyDiffGame
from PyDiffGame.Objective import GameObjective


class PyDiffGameComparison(ABC, Callable, Sequence):
    """
    Differential games comparison abstract base class


    Parameters
    ----------
    args: sequence of 2-d np.arrays of len(N), each array B_i of shape(n, m_i)
        System input matrices for each control objective
    games_objectives: Sequence of Sequences of GameObjective objects
        Each game objectives for the comparison
    continuous: boolean, optional
        Indicates whether to consider a continuous or discrete control design
    """

    def __init__(self,
                 args: dict[str, Any],
                 games_objectives: Sequence[Sequence[GameObjective]],
                 continuous: bool = True):
        GameClass = ContinuousPyDiffGame if continuous else DiscretePyDiffGame

        self.__args = args
        self._verify_input()

        self._games = {i: GameClass(**self.__args | {'objectives': [o for o in game_i_objectives]})
                       for i, game_i_objectives in enumerate(games_objectives)}

    def _verify_input(self):
        """
        Input checking method

        Raises
        ------
        Case-specific errors
        """

        for mat in ['A', 'B']:
            mat_name = 'dynamics' if mat == 'A' else 'input'
            if mat not in self.__args.keys():
                raise ValueError(f"Passed arguments must contain the {mat_name} matrix {mat}")
            if isinstance(self.__args[mat], type(np.array)):
                raise ValueError(f'The {mat_name} matrix {mat} must be a numpy array')

    def are_fully_controllable(self):
        return all(game.is_fully_controllable() for game in self._games.values())

    def plot_two_state_spaces(self,
                              i: Optional[int] = 0,
                              j: Optional[int] = 1,
                              non_linear: bool = False):
        game_i = self._games[i]
        game_j = self._games[j]
        game_i.plot_two_state_spaces(other=game_j,
                                     non_linear=non_linear)

    def _simulate_non_linear_system(self,
                                    *args) -> np.array:
        """
        Simulates the corresponding non-linear system's progression through time
        """

        pass

    def run_animation(self,
                      i: int):
        """
        Animates the state progression of the system
        """

        pass

    def __call__(self,
                 plot_state_spaces: Optional[bool] = True,
                 run_animations: Optional[bool] = True,
                 calculate_costs: Optional[bool] = False,
                 x_only_costs: Optional[bool] = False):
        """
        Runs the comparison
        """

        for i, game_i in self._games.items():
            game_i(plot_state_space=plot_state_spaces)

            if run_animations:
                self.run_animation(i=i)

            if calculate_costs:
                game_i_costs = game_i.get_costs(x_only=x_only_costs)
                print(f'Game {i} costs: {game_i_costs}')

    def __getitem__(self, i: int) -> PyDiffGame:
        return self._games[i]

    def __len__(self):
        return len(self._games)
