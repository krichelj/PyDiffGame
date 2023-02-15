import numpy as np
from typing import Sequence, Optional
from abc import ABC

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame
from PyDiffGame.DiscretePyDiffGame import DiscretePyDiffGame
from PyDiffGame.Objective import LQRObjective


class LQR(PyDiffGame, ABC):
    """
    LQR abstract base class


    Considers a differential game with only one objective
    """

    def __init__(self,
                 A: np.array,
                 B: np.array,
                 objective: LQRObjective,
                 x_0: Optional[np.array] = None,
                 x_T: Optional[np.array] = None,
                 T_f: Optional[float] = None,
                 P_f: Optional[np.array] = None,
                 show_legend: Optional[bool] = True,
                 state_variables_names: Optional[Sequence[str]] = None,
                 epsilon_x: Optional[float] = PyDiffGame._epsilon_x_default,
                 L: Optional[int] = PyDiffGame._L_default,
                 eta: Optional[int] = PyDiffGame._eta_default,
                 debug: Optional[bool] = False):
        super().__init__(A=A,
                         B=B,
                         objectives=[objective],
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         P_f=P_f,
                         show_legend=show_legend,
                         state_variables_names=state_variables_names,
                         epsilon_x=epsilon_x,
                         L=L,
                         eta=eta,
                         debug=debug)


class ContinuousLQR(LQR, ContinuousPyDiffGame):
    """
    Continuous LQR marker class


    Considers control design according to the system:
    dx(t)/dt = A x(t) + B u(t)
    """

    pass


class DiscreteLQR(LQR, DiscretePyDiffGame):
    """
    Discrete LQR marker class


    Considers control design according to the system:
    x[k+1] = A_tilda x[k] +  B_tilda u[k]

    where A_tilda and B_tilda are in discrete form, meaning they correlate to a discrete system,
    which, at the sampling points, is assumed to be equal to some equivalent continuous system of the form:
    dx(t)/dt = A x(t) +  B u(t)
    """

    pass
