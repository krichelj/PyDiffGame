import numpy as np
from tqdm import tqdm
from time import time
from termcolor import colored
import itertools
from concurrent.futures import ProcessPoolExecutor

import inspect
from typing import Sequence, Any, Optional, Callable
from abc import ABC

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame
from PyDiffGame.DiscretePyDiffGame import DiscretePyDiffGame
from PyDiffGame.Objective import GameObjective


class PyDiffGameLQRComparison(ABC, Callable, Sequence):
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
                 M: np.array,
                 continuous: bool = True):
        GameClass = ContinuousPyDiffGame if continuous else DiscretePyDiffGame

        self.__args = args
        self.__verify_input()

        self._games = {}
        self._lqr_objective = None
        self.__M = M

        for i, game_i_objectives in enumerate(games_objectives):
            game_i = GameClass(**self.__args | {'objectives': [o for o in game_i_objectives]})
            self._games[i] = game_i

            if game_i.is_LQR():
                self._lqr_objective = game_i_objectives[0]

    def __verify_input(self):
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

    def __simulate_non_linear_system(self,
                                     i: int,
                                     plot: bool = False) -> np.array:
        """
        Simulates the corresponding non-linear system's progression through time
        """

        pass

    def __run_animation(self,
                        i: int):
        """
        Animates the state progression of the system
        """

        pass

    def __call__(self,
                 plot_state_spaces: Optional[bool] = True,
                 plot_Mx: Optional[bool] = False,
                 output_variables_names: Optional[Sequence[str]] = None,
                 save_figure: Optional[bool] = False,
                 run_animations: Optional[bool] = True,
                 print_characteristic_polynomials: Optional[bool] = False,
                 print_eigenvalues: Optional[bool] = False):
        """
        Runs the comparison
        """

        for i, game_i in self._games.items():
            game_i(plot_state_space=plot_state_spaces,
                   plot_Mx=plot_Mx,
                   M=self.__M,
                   output_variables_names=output_variables_names,
                   save_figure=save_figure,
                   print_characteristic_polynomials=print_characteristic_polynomials,
                   print_eigenvalues=print_eigenvalues)

            if run_animations:
                self.__run_animation(i=i)

    @staticmethod
    def run_multiprocess(multiprocess_worker_function: Callable[[Any], None],
                         values: Sequence[Sequence]):
        t_start = time()
        combos = list(itertools.product(*values))

        with ProcessPoolExecutor() as executor:
            submittals = {str(combo): executor.submit(multiprocess_worker_function, *combo) for combo in combos}

            delimiter = ', '
            names = inspect.signature(multiprocess_worker_function).parameters.keys()

            for combo, submittal in tqdm(iterable=submittals.items(), total=len(submittals)):
                values = combo[1:-1].split(delimiter)
                print_str = f"""{colored(text=f"{delimiter.join([f'{n}={v}' for n, v in zip(names, values)])}", 
                                            color='blue')}"""
                print(print_str)
                submittal.result()

        print(f'Total time: {time() - t_start}')

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
