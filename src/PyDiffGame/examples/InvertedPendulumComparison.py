"""Inverted-pendulum-on-a-cart benchmark: differential game vs. LQR.

A uniform rod pinned to a cart forms the classic underactuated inverted
pendulum.  Linearising about the upright equilibrium yields a ``4``-dimensional
linear system (cart position, pendulum angle and their velocities) driven by a
two-dimensional physical input (cart force ``F`` and pendulum moment ``M``).
The input is decomposed so that the cart and the pendulum each become one player
of a differential game, and the result is compared against a single LQR
controller acting on the whole system.

The nonlinear closed loop can additionally be simulated by feeding the designed
gains back into the full (un-linearised) equations of motion.

Run directly to solve a single instance and report the costs and stability::

    python -m PyDiffGame.examples.InvertedPendulumComparison
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from PyDiffGame.base import PyDiffGame
from PyDiffGame.comparison import PyDiffGameLQRComparison
from PyDiffGame.objective import GameObjective, LQRObjective
from PyDiffGame.plotting import show


class InvertedPendulumComparison(PyDiffGameLQRComparison):
    """Compare a two-player differential game with an LQR for the cart-pendulum."""

    def __init__(
        self,
        m_c: float,
        m_p: float,
        p_L: float,
        q: float,
        r: float = 1.0,
        x_0: np.ndarray | None = None,
        x_T: np.ndarray | None = None,
        T_f: float | None = None,
        epsilon_x: float = PyDiffGame.epsilon_x_default,
        epsilon_P: float = PyDiffGame.epsilon_P_default,
        L: int = PyDiffGame.L_default,
        eta: int = PyDiffGame.eta_default,
    ) -> None:
        self._m_c = m_c
        self._m_p = m_p
        self._p_L = p_L
        self._l = p_L / 2  # centre of mass of the uniform rod
        self._I = 1 / 12 * m_p * p_L**2  # moment of inertia of the uniform rod about its centre

        # Linearised system about the upright equilibrium. The cart<->pendulum
        # coupling terms (a32, and the off-diagonal inputs b22, b31) are negative
        # in the standard convention; gravity destabilises the angle (a42 > 0).
        # These matrices are exactly the Jacobian of ``_nonlinear_acceleration``
        # at the origin (asserted by tests/test_examples.py).
        linearized_D = m_c * m_p * self._l**2 + self._I * (m_c + m_p)
        a32 = -m_p * PyDiffGame.g * self._l**2 / linearized_D
        a42 = m_p * PyDiffGame.g * self._l * (m_c + m_p) / linearized_D
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, a32, 0, 0], [0, a42, 0, 0]], dtype=float)

        b21 = (m_p * self._l**2 + self._I) / linearized_D
        b31 = -m_p * self._l / linearized_D
        b22 = b31
        b32 = (m_c + m_p) / linearized_D
        B = np.array([[0, 0], [0, 0], [b21, b22], [b31, b32]], dtype=float)

        # Each physical-input row becomes one player's decomposition matrix.
        M1 = B[2, :].reshape(1, 2)
        M2 = B[3, :].reshape(1, 2)
        Ms = [M1, M2]

        Q_x = q * np.array([[1, 0, 2, 0], [0, 0, 0, 0], [2, 0, 4, 0], [0, 0, 0, 0]], dtype=float)
        Q_theta = q * np.array([[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0], [0, 2, 0, 4]], dtype=float)
        Q_lqr = Q_x + Q_theta
        Qs = [Q_x, Q_theta]

        R_lqr = np.diag([r] * 2)
        Rs = [np.array([[r]])] * 2

        self._origin = (0.0, 0.0)

        state_variables_names = [
            "x",
            r"\theta",
            r"\dot{x}",
            r"\dot{\theta}",
        ]

        lqr_objective = [LQRObjective(Q=Q_lqr, R=R_lqr)]
        game_objectives = [GameObjective(Q=Q_i, R=R_i, M=M_i) for Q_i, R_i, M_i in zip(Qs, Rs, Ms)]
        games_objectives = [lqr_objective, game_objectives]

        super().__init__(
            A=A,
            B=B,
            games_objectives=games_objectives,
            continuous=True,
            x_0=x_0,
            x_T=x_T,
            T_f=T_f,
            L=L,
            eta=eta,
            epsilon_x=epsilon_x,
            epsilon_P=epsilon_P,
            state_variables_names=state_variables_names,
        )

    def _nonlinear_acceleration(
        self, theta: float, theta_dot: float, F: float, M: float
    ) -> tuple[float, float]:
        """Cart and pendulum accelerations from the full nonlinear EOM.

        Upright convention: gravity destabilises the angle. The Jacobian of this
        map at the origin equals the linear design matrices ``A``, ``B`` (checked
        in ``tests/test_examples.py``), so the gains designed on the linear model
        are applied to a consistent nonlinear plant.
        """

        m_p, m_c, length, inertia = self._m_p, self._m_c, self._l, self._I
        den = m_p * length**2 + inertia - (m_p * length) ** 2 * np.cos(theta) ** 2 / (m_p + m_c)
        theta_ddot = (
            M
            - m_p
            * length
            * (
                np.cos(theta) / (m_p + m_c) * (F + m_p * length * np.sin(theta) * theta_dot**2)
                - PyDiffGame.g * np.sin(theta)
            )
        ) / den
        x_ddot = (F + m_p * length * (np.sin(theta) * theta_dot**2 - np.cos(theta) * theta_ddot)) / (
            m_p + m_c
        )
        return x_ddot, theta_ddot

    def simulate_nonlinear_system(self, i: int) -> np.ndarray:
        """Simulate the full nonlinear cart-pendulum closed loop for game ``i``.

        The feedback gains designed on the linear model are fed back into the
        un-linearised equations of motion. Works for both the infinite-horizon
        (constant gain) and finite-horizon (time-varying gain) designs. Returns
        the state trajectory with shape ``(n, L)``.
        """

        game = self._games[i]
        game.solve()
        x_T = game.x_T if game.x_T is not None else np.zeros(game.n)
        forward_time = game.forward_time
        n_samples = len(forward_time)
        delta = game.T_f / n_samples

        def nonlinear_state_space(t: float, x_t: np.ndarray) -> np.ndarray:
            x_tilde = x_t - x_T
            # Aggregate physical-input gain at this time (already maps the
            # per-player gains back to the physical input u = [F, M]).
            sample = min(int(t / delta), n_samples - 1)
            F_t, M_t = (-game._gain_at(sample) @ x_tilde).ravel()
            _x, theta, _x_dot, theta_dot = x_t
            x_ddot, theta_ddot = self._nonlinear_acceleration(theta, theta_dot, F_t, M_t)
            return np.array([_x_dot, theta_dot, x_ddot, theta_ddot], dtype=float)

        pendulum_state = solve_ivp(
            fun=nonlinear_state_space,
            t_span=[0.0, game.T_f],
            y0=game.x_0,
            t_eval=game.forward_time,
            rtol=1e-6,
        )

        return pendulum_state.y


def multiprocess_worker_function(
    x_T: float,
    theta_0: float,
    m_c: float,
    m_p: float,
    p_L: float,
    q: float,
    epsilon_x: float,
    epsilon_P: float,
) -> int:
    """Worker for a parameter sweep: returns 1 if the LQR cost is the largest."""

    x_T_vec = np.array([x_T, theta_0, 0.0, 0.0])
    x_0 = np.zeros_like(x_T_vec)

    comparison = InvertedPendulumComparison(
        m_c=m_c,
        m_p=m_p,
        p_L=p_L,
        q=q,
        x_0=x_0,
        x_T=x_T_vec,
        T_f=10.0,
        L=300,
        epsilon_x=epsilon_x,
        epsilon_P=epsilon_P,
    )
    comparison.run(plot_state_spaces=False)
    lqr_cost, game_cost = comparison.costs()
    return int(lqr_cost >= game_cost)


def main(*, plot: bool = True, save_figure: bool = False) -> None:
    """Solve a single inverted-pendulum comparison and print the costs."""

    m_c, m_p, p_L, q, r = 10.0, 1.0, 1.0, 1.0, 1.0
    x_T = np.array([5.0, 0.0, 0.0, 0.0])
    x_0 = np.array([0.0, np.pi / 18, 0.0, 0.0])

    # The upright pendulum is open-loop unstable, so the coupled infinite-horizon
    # game need not have a stabilising Nash equilibrium; use a finite horizon.
    comparison = InvertedPendulumComparison(
        m_c=m_c, m_p=m_p, p_L=p_L, q=q, r=r, x_0=x_0, x_T=x_T, T_f=10.0, L=300
    )
    comparison.run(plot_state_spaces=plot, save_figure=save_figure)

    lqr_cost, game_cost = comparison.costs()
    print(f"LQR cost   = {lqr_cost:.4g}")
    print(f"Game cost  = {game_cost:.4g}")
    print(f"All games controllable: {comparison.are_all_controllable()}")
    print(f"LQR  closed loop stable: {comparison[0].is_closed_loop_stable()}")
    print(f"Game closed loop stable: {comparison[1].is_closed_loop_stable()}")
    if plot:
        show()


if __name__ == "__main__":
    main(plot=False)
