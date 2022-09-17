import numpy as np
from termcolor import colored
from typing import Sequence, Any, Optional, Callable
from abc import ABC

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
        """
        Tests each game's full controllability as an LTI system
        """

        return all(game.is_fully_controllable() for game in self._games.values())

    def plot_two_state_spaces(self,
                              i: Optional[int] = 0,
                              j: Optional[int] = 1,
                              non_linear: bool = False):
        """
        Plot the state spaces of two games
        """

        self._games[i].plot_two_state_spaces(other=self._games[j],
                                             non_linear=non_linear)

    def _simulate_non_linear_system(self,
                                    i: int,
                                    plot: bool = False) -> np.array:
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
                 print_characteristic_polynomials: Optional[bool] = False,
                 print_eigenvalues: Optional[bool] = False,
                 plot_state_spaces: Optional[bool] = True,
                 run_animations: Optional[bool] = True,
                 print_costs: Optional[bool] = False,
                 non_linear_costs: Optional[bool] = False,
                 agnostic_costs: Optional[bool] = False,
                 x_only_costs: Optional[bool] = False) -> bool:
        """
        Runs the comparison
        """

        costs = []
        res = False

        for i, game_i in self._games.items():
            game_i(plot_state_space=plot_state_spaces,
                   print_characteristic_polynomials=print_characteristic_polynomials,
                   print_eigenvalues=print_eigenvalues)

            if run_animations:
                self.run_animation(i=i)
            elif non_linear_costs:
                game_i._x_non_linear = self._simulate_non_linear_system(i=i,
                                                                        plot=False)

            if print_costs:
                game_i_costs = game_i.get_agnostic_costs(non_linear=non_linear_costs) \
                    if agnostic_costs else game_i.get_costs(non_linear=non_linear_costs,
                                                            x_only=x_only_costs)

                costs += [game_i_costs]

        if print_costs:
            costs = np.array(costs)
            maximal_cost_game = costs.argmax()
            is_max_lqr = self._games[maximal_cost_game].is_LQR()

            print('\n' + '#' * 100 + '\n' + colored(f"\nGame {maximal_cost_game + 1} has the most cost "
                                                    f"{'which is LQR! :)' if is_max_lqr else ' :('}\n",
                                                    'green' if is_max_lqr else 'red') + '#' * 50
                  + '\n\n' + '\n'.join([f"Game {i + 1} {'non-linear ' if non_linear_costs else ''}"
                                        f"{'agnostic ' if agnostic_costs else ''}"
                                        f"cost: 10^{round(cost_i, 3)}" for i, cost_i in enumerate(costs)]))

            res = is_max_lqr

        return res

    def __getitem__(self,
                    i: int) -> PyDiffGame:
        """
        Returns the i'th game
        """

        return self._games[i]

    def __len__(self) -> int:
        """
        We define the length of a comparison to be its number of games
        """

        return len(self._games)
