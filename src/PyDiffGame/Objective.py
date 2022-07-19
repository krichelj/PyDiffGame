import numpy as np
from abc import ABC
from typing import Sequence


class Objective(ABC):
    def __init__(self,
                 Qs: Sequence[np.array],
                 Rs: Sequence[np.array]):
        self._Qs = Qs
        self._Rs = Rs

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        res = self._Qs[self._i], self._i
        return


class LQRObjective(Objective):
    def __init__(self,
                 Q: np.array,
                 R: np.array):
        super().__init__(Qs=[Q],
                         Rs=[R])


class GameObjective(Objective):
    def __init__(self,
                 Qs: Sequence[np.array],
                 Rs: Sequence[np.array],
                 Ms: Sequence[np.array]):
        self.__Ms = Ms
        super().__init__(Qs=Qs,
                         Rs=Rs)
