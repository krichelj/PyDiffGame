import numpy as np

from PyDiffGame.PyDiffGame import PyDiffGame
from PyDiffGame.ContinuousPyDiffGame import ContinuousPyDiffGame
from PyDiffGame.DiscretePyDiffGame import DiscretePyDiffGame


class ContinuousLQR(ContinuousPyDiffGame):
    """
    Continuous LQR base class


    Considers the system:
    dx(t)/dt = A x(t) + B u(t)
    """
    def __init__(self,
                 A: np.array,
                 B: np.array,
                 Q: np.array,
                 R: np.array,
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 P_f: np.array = None,
                 show_legend: bool = True,
                 state_variables_names: list = None,
                 epsilon: float = PyDiffGame.epsilon_default,
                 L: int = PyDiffGame.L_default,
                 eta: int = PyDiffGame.eta_default,
                 force_finite_horizon: bool = False,
                 debug: bool = False):

        super().__init__(A=A,
                         B=B,
                         Q=Q,
                         R=R,
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         P_f=P_f,
                         show_legend=show_legend,
                         state_variables_names=state_variables_names,
                         epsilon=epsilon,
                         L=L,
                         eta=eta,
                         force_finite_horizon=force_finite_horizon,
                         debug=debug)


class DiscreteLQR(DiscretePyDiffGame):
    """
    Discrete differential game base class


    Considers the system:
    x[k+1] = A_tilda x[k] +  B_tilda u[k]

    where A_tilda and B_tilda are in discrete form, meaning they correlate to a discrete system,
    which, at the sampling points, is assumed to be equal to some equivalent continuous system of the form:
    dx(t)/dt = A x(t) +  B u(t)

    """
    def __init__(self,
                 A: np.array,
                 B: np.array,
                 Q: np.array,
                 R: np.array,
                 x_0: np.array = None,
                 x_T: np.array = None,
                 T_f: float = None,
                 P_f: np.array = None,
                 show_legend: bool = True,
                 state_variables_names: list = None,
                 epsilon: float = PyDiffGame.epsilon_default,
                 L: int = PyDiffGame.L_default,
                 eta: int = PyDiffGame.eta_default,
                 force_finite_horizon: bool = False,
                 debug: bool = False):

        super().__init__(A=A,
                         B=B,
                         Q=Q,
                         R=R,
                         x_0=x_0,
                         x_T=x_T,
                         T_f=T_f,
                         P_f=P_f,
                         show_legend=show_legend,
                         state_variables_names=state_variables_names,
                         epsilon=epsilon,
                         L=L,
                         eta=eta,
                         force_finite_horizon=force_finite_horizon,
                         debug=debug)
