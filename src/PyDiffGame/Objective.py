import numpy as np
from abc import ABC


class Objective(ABC):
    """
    Objective abstract base class


    Parameters
    ----------
    Q: 2-d np.array of shape(n, n)
        Cost function state weights for the objective
    R_ii: 2-d np.array of shape(m_i, m_i)
        Cost function input weights for the objective with regard to the i'th input
    """

    def __init__(self,
                 Q: np.array,
                 R_ii: np.array):
        self._Q = Q
        self._R_ii = R_ii

    @property
    def Q(self) -> np.array:
        return self._Q

    @property
    def R(self) -> np.array:
        return self._R_ii


class LQRObjective(Objective):
    """
    LQR objective marker class
    """

    pass


class GameObjective(Objective):
    """
    Game objective class


    Parameters
    ----------
    M_i: 2-d np.array of shape(m_i, m)
        Division matrix for the i'th virtual input
    """

    def __init__(self,
                 Q: np.array,
                 R_ii: np.array,
                 M_i: np.array):
        self.__M_i = M_i
        super().__init__(Q=Q,
                         R_ii=R_ii)

    @property
    def M_i(self) -> np.array:
        return self.__M_i
