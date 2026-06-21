"""Abstract base class for differential games.

:class:`PyDiffGame` holds everything that is independent of the time domain:
construction and validation of the problem data, the
virtual-input decomposition, controllability tests, cost evaluation, the
solve/simulate orchestration and plotting.  The continuous- and discrete-time
specifics live in :mod:`PyDiffGame.continuous` and :mod:`PyDiffGame.discrete`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from PyDiffGame import plotting
from PyDiffGame._typing import ArrayLike, FloatArray, PathInput
from PyDiffGame.objective import Objective


class PyDiffGame(ABC):
    r"""Differential game abstract base class.

    Considers control design for the linear system

    .. math:: \dot{x}(t) = A x(t) + \sum_{i=1}^N B_i v_i(t)

    where each player :math:`i` minimises a quadratic cost weighted by
    :math:`Q_i` (state) and :math:`R_{ii}` (input).  When the players share a
    single physical input :math:`u` through decomposition matrices
    :math:`M_i`, pass the full input matrix ``B`` together with objectives that
    carry an ``M``; otherwise pass the per-player matrices ``Bs`` directly.

    Parameters
    ----------
    A:
        System dynamics matrix of shape ``(n, n)``.
    objectives:
        One :class:`~PyDiffGame.objective.Objective` per player.
    B:
        Full input matrix of shape ``(n, m)``, used together with each
        objective's decomposition matrix ``M``.
    Bs:
        Per-player input matrices, an alternative to ``B`` + ``M``.
    x_0, x_T:
        Optional initial / terminal state vectors of shape ``(n,)``.
    T_f:
        Finite horizon length.  ``None`` (the default) solves the
        infinite-horizon problem.
    P_f:
        Terminal Riccati condition(s); defaults to the uncoupled algebraic
        Riccati solutions.
    L:
        Number of time samples.
    eta:
        Number of trailing matrix norms inspected for convergence.
    epsilon_x, epsilon_P:
        Convergence tolerances for the state and for the Riccati matrices.
    state_variables_names:
        LaTeX names (without ``$``) for the ``n`` state variables, used in plots.
    show_legend:
        Whether plots include a legend.
    debug:
        Emit verbose diagnostics while solving.
    """

    T_f_default: Final[float] = 20.0
    epsilon_x_default: Final[float] = 1e-7
    epsilon_P_default: Final[float] = 1e-7
    L_default: Final[int] = 1000
    eta_default: Final[int] = 5
    eigenvalue_tolerance: Final[float] = 1e-7
    #: Standard gravitational acceleration, handy for the mechanical examples.
    g: Final[float] = 9.81

    max_x_iterations: Final[int] = 100
    max_P_iterations: Final[int] = 100

    default_figures_path: Final[Path] = Path.cwd() / "figures"
    default_figures_filename: Final[str] = "image"

    def __init__(
        self,
        A: ArrayLike,
        objectives: Sequence[Objective],
        *,
        B: ArrayLike | None = None,
        Bs: Sequence[ArrayLike] | None = None,
        x_0: ArrayLike | None = None,
        x_T: ArrayLike | None = None,
        T_f: float | None = None,
        P_f: Sequence[ArrayLike] | None = None,
        L: int = L_default,
        eta: int = eta_default,
        epsilon_x: float = epsilon_x_default,
        epsilon_P: float = epsilon_P_default,
        state_variables_names: Sequence[str] | None = None,
        show_legend: bool = True,
        debug: bool = False,
    ) -> None:
        self._A = np.asarray(A, dtype=np.float64)
        self._n = self._A.shape[0]
        self._objectives = list(objectives)
        self._N = len(self._objectives)
        self._Qs = [o.Q for o in self._objectives]
        self._Rs = [o.R for o in self._objectives]

        self._B = None if B is None else np.asarray(B, dtype=np.float64)
        self._M: FloatArray | None = None
        self._M_inv: FloatArray | None = None
        self._Bs = self._resolve_input_matrices(Bs)
        self._m = sum(B_i.shape[1] for B_i in self._Bs)

        self._x_0 = None if x_0 is None else np.asarray(x_0, dtype=np.float64).ravel()
        self._x_T = None if x_T is None else np.asarray(x_T, dtype=np.float64).ravel()

        self._infinite_horizon = T_f is None
        self._T_f = float(self.T_f_default if T_f is None else T_f)
        self._L = int(L)
        self._eta = int(eta)
        self._epsilon_x = float(epsilon_x)
        self._epsilon_P = float(epsilon_P)
        self._delta = self._T_f / self._L

        self._state_variables_names = None if state_variables_names is None else list(state_variables_names)
        self._show_legend = bool(show_legend)
        self._debug = bool(debug)

        self._forward_time = np.linspace(0.0, self._T_f, self._L)

        # Per-player feedback gains S_i = B_i R_ii^{-1} B_i^T.
        self._S = [B_i @ np.linalg.solve(R_i, B_i.T) for B_i, R_i in zip(self._Bs, self._Rs)]

        self._P_f = (
            self._uncoupled_are_solutions()
            if P_f is None
            else [np.asarray(P_f_i, dtype=np.float64) for P_f_i in P_f]
        )

        # Solver outputs, populated by ``solve`` / ``simulate``.  ``_P`` / ``_K``
        # are intentionally dynamic: a list of constant matrices for the
        # infinite-horizon case, a time-indexed array for the finite-horizon one.
        self._P: Any = []
        self._K: Any = []
        self._A_cl: FloatArray = np.empty_like(self._A)
        self._x: FloatArray | None = None
        self._solved = False
        self._fig: Any = None

        self._validate()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _resolve_input_matrices(self, Bs: Sequence[ArrayLike] | None) -> list[FloatArray]:
        """Determine the per-player input matrices ``B_i``.

        Either ``Bs`` is given explicitly, or it is derived from the full input
        matrix ``B`` and the objectives' decomposition matrices ``M_i`` through
        :math:`B_i = B M^{-1}[:, \\text{block}_i]` where ``M`` stacks the
        ``M_i``.
        """

        if Bs is not None:
            return [np.asarray(B_i, dtype=np.float64) for B_i in Bs]

        if self._B is None:
            raise ValueError("Either B or Bs must be provided")

        decomposition = [o.M for o in self._objectives if o.M is not None]
        if not decomposition:
            # Plain LQR / single shared input: one player drives the full input.
            return [self._B]

        self._M = np.concatenate(decomposition, axis=0)
        try:
            M_inv = np.linalg.inv(self._M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(self._M)
        self._M_inv = M_inv

        Bs_resolved: list[FloatArray] = []
        column = 0
        for M_i in decomposition:
            m_i = M_i.shape[0]
            Bs_resolved.append(self._B @ M_inv[:, column : column + m_i])
            column += m_i
        return Bs_resolved

    def _uncoupled_are_solutions(self) -> list[FloatArray]:
        """Solve each player's decoupled algebraic Riccati equation.

        Used as the terminal Riccati condition / iteration seed.  Falls back to
        ``Q_i`` when the decoupled equation has no stabilising solution.
        """

        solutions: list[FloatArray] = []
        for B_i, Q_i, R_i in zip(self._Bs, self._Qs, self._Rs):
            try:
                solutions.append(self._solve_are(B_i, Q_i, R_i))
            except (np.linalg.LinAlgError, ValueError):
                solutions.append(Q_i)
        return solutions

    @abstractmethod
    def _solve_are(self, B: FloatArray, Q: FloatArray, R: FloatArray) -> FloatArray:
        """Solve the (continuous or discrete) algebraic Riccati equation for ``(A, B)``."""

    def _validate(self) -> None:
        if self._N == 0:
            raise ValueError("At least one objective must be specified")
        if self._A.shape != (self._n, self._n):
            raise ValueError(f"A must be square (n, n) = ({self._n}, {self._n})")
        for Q_i in self._Qs:
            if Q_i.shape != (self._n, self._n):
                raise ValueError(f"every Q must have shape ({self._n}, {self._n})")
        for B_i, R_i in zip(self._Bs, self._Rs):
            if B_i.shape[0] != self._n:
                raise ValueError(f"every B_i must have n = {self._n} rows")
            if B_i.shape[1] != R_i.shape[0]:
                raise ValueError("each B_i column count must match its R_ii size")
        for vec, name in ((self._x_0, "x_0"), (self._x_T, "x_T")):
            if vec is not None and vec.shape != (self._n,):
                raise ValueError(f"{name} must have length n = {self._n}")
        if self._T_f <= 0:
            raise ValueError("T_f must be a positive real number")
        if self._L <= 0:
            raise ValueError("L (number of samples) must be a positive integer")
        if self._eta <= 0:
            raise ValueError("eta must be a positive integer")
        if not 0 < self._epsilon_x < 1:
            raise ValueError("epsilon_x must lie in the open interval (0, 1)")
        if not 0 < self._epsilon_P < 1:
            raise ValueError("epsilon_P must lie in the open interval (0, 1)")
        if self._state_variables_names is not None and len(self._state_variables_names) != self._n:
            raise ValueError(f"state_variables_names must have length n = {self._n}")
        if self._B is not None and not self.is_controllable():
            import warnings

            warnings.warn("the given system is not fully controllable", stacklevel=2)

    # ------------------------------------------------------------------ #
    # System properties
    # ------------------------------------------------------------------ #
    def is_controllable(self) -> bool:
        """Whether ``(A, B)`` is fully controllable (full-rank controllability matrix)."""

        if self._B is None:
            B = np.concatenate(self._Bs, axis=1)
        else:
            B = self._B
        controllability = np.concatenate(
            [np.linalg.matrix_power(self._A, i) @ B for i in range(self._n)], axis=1
        )
        return int(np.linalg.matrix_rank(controllability)) == self._n

    @property
    def is_lqr(self) -> bool:
        """Whether the game reduces to a single-objective LQR problem."""

        return self._N == 1

    def _closed_loop(self, gains: Sequence[FloatArray]) -> FloatArray:
        r"""Closed-loop matrix :math:`A - \sum_i B_i K_i` for the given gains."""

        return self._A - sum(B_i @ K_i for B_i, K_i in zip(self._Bs, gains))

    # ------------------------------------------------------------------ #
    # Solve / simulate orchestration
    # ------------------------------------------------------------------ #
    @abstractmethod
    def solve(self) -> PyDiffGame:
        """Solve the Riccati equation(s) and store ``P`` and the gains ``K``."""

    @abstractmethod
    def simulate(self) -> PyDiffGame:
        """Propagate the closed-loop state trajectory from ``x_0``."""

    @abstractmethod
    def is_closed_loop_stable(self) -> bool:
        """Whether the converged closed loop is stable in the relevant sense."""

    def _require_solved(self) -> None:
        if not self._solved:
            raise RuntimeError("the game must be solved first; call solve()")

    def run(
        self,
        *,
        plot_state_space: bool = True,
        save_figure: bool = False,
        figure_path: PathInput = default_figures_path,
        figure_filename: str = default_figures_filename,
    ) -> PyDiffGame:
        """Solve, simulate (if ``x_0`` is set) and optionally plot the state."""

        self.solve()
        if self._x_0 is not None:
            self.simulate()
            if plot_state_space:
                self.plot_state_variables(
                    save_figure=save_figure,
                    figure_path=figure_path,
                    figure_filename=figure_filename,
                )
        return self

    def __call__(self, **kwargs) -> PyDiffGame:
        """Alias for :meth:`run` so a game instance is callable."""

        return self.run(**kwargs)

    # ------------------------------------------------------------------ #
    # Cost
    # ------------------------------------------------------------------ #
    def cost(self, objective: Objective, *, state_only: bool = False) -> float:
        r"""Trapezoidal approximation of an objective's cost along the trajectory.

        .. math:: J \approx \delta \left[ \tfrac12 (J_0 + J_L)
                  + \sum_{l=1}^{L-1} J_l \right]
        """

        self._require_solved()
        if self._x is None:
            raise RuntimeError("no state trajectory; call simulate() with an x_0")

        Q, R = objective.Q, objective.R
        target = self._x_T if self._x_T is not None else np.zeros(self._n)
        gains = self._gain_at  # bound method, time-aware per subclass

        total = 0.0
        for l in range(self._L):
            x_tilde = target - self._x[l]
            cost_l = float(x_tilde @ Q @ x_tilde)
            if not state_only:
                u_tilde = -gains(l) @ x_tilde
                cost_l += float(u_tilde @ R @ u_tilde)
            if l in (0, self._L - 1):
                cost_l *= 0.5
            total += cost_l
        return total * self._delta

    @abstractmethod
    def _gain_at(self, l: int) -> FloatArray:
        """Aggregate physical-input gain at sample ``l`` (time-aware per domain)."""

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def plot_state_variables(
        self,
        *,
        save_figure: bool = False,
        figure_path: PathInput = default_figures_path,
        figure_filename: str = default_figures_filename,
    ) -> None:
        """Plot the state trajectory against time."""

        self._require_solved()
        if self._x is None:
            raise RuntimeError("no state trajectory; call simulate() with an x_0")

        labels = None
        if self._show_legend:
            if self._state_variables_names is not None:
                labels = [f"${name}$" for name in self._state_variables_names]
            else:
                labels = [rf"$\mathbf{{x}}_{{{j}}}$" for j in range(1, self._n + 1)]

        self._fig = plotting.plot_temporal(
            self._forward_time,
            self._x,
            labels=labels,
            show_legend=self._show_legend,
        )
        if save_figure:
            plotting.save_figure(self._fig, figure_path, figure_filename)

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #
    @property
    def A(self) -> FloatArray:
        return self._A

    @property
    def Bs(self) -> list[FloatArray]:
        return self._Bs

    @property
    def n(self) -> int:
        return self._n

    @property
    def N(self) -> int:
        return self._N

    @property
    def P(self) -> list[FloatArray] | FloatArray:
        self._require_solved()
        return self._P

    @property
    def K(self) -> list[FloatArray] | FloatArray:
        self._require_solved()
        return self._K

    @property
    def x(self) -> FloatArray | None:
        return self._x

    @property
    def x_0(self) -> FloatArray | None:
        return self._x_0

    @property
    def x_T(self) -> FloatArray | None:
        return self._x_T

    @property
    def forward_time(self) -> FloatArray:
        return self._forward_time

    @property
    def T_f(self) -> float:
        return self._T_f

    @property
    def L(self) -> int:
        return self._L

    @property
    def M_inv(self) -> FloatArray | None:
        return self._M_inv

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, i: int) -> Objective:
        return self._objectives[i]

    def __repr__(self) -> str:
        domain = type(self).__name__
        horizon = "infinite" if self._infinite_horizon else f"T_f={self._T_f:g}"
        kind = "LQR" if self.is_lqr else f"{self._N}-player game"
        return f"<{domain}: {kind}, n={self._n}, {horizon}>"


__all__ = ["PyDiffGame"]
