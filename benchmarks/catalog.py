"""A catalogue of standard LTI control systems of increasing complexity.

Each entry is a representative linearized model from the controls literature,
with a disturbance input ``B_w`` placed where physical disturbances act (forces,
gusts, load torques, road banks). Used to benchmark the H-infinity differential
game against an LQR across cars, aircraft and drones. Models are illustrative
but physically reasonable; exact coefficients are documented per system.
"""

from __future__ import annotations

import numpy as np

from benchmarks.robust_compare import RobustProblem


def cart() -> RobustProblem:
    """1. Double-integrator cart; force input, force disturbance (the simplest case)."""
    return RobustProblem(
        name="cart (double integrator)",
        A=np.array([[0.0, 1.0], [0.0, 0.0]]),
        B=np.array([[0.0], [1.0]]),
        B_w=np.array([[0.0], [1.0]]),
        Q=np.diag([1.0, 0.0]),
        R=np.array([[1.0]]),
        x_0=np.zeros(2),
        T_f=40.0,
        output_idx=0,
        output_name="position",
    )


def mass_spring_damper() -> RobustProblem:
    """2. Lightly-damped mass-spring-damper (m=1, k=1, c=0.2) — strong resonance."""
    m, k, c = 1.0, 1.0, 0.2
    return RobustProblem(
        name="mass-spring-damper",
        A=np.array([[0.0, 1.0], [-k / m, -c / m]]),
        B=np.array([[0.0], [1.0 / m]]),
        B_w=np.array([[0.0], [1.0 / m]]),
        Q=np.diag([1.0, 0.0]),
        R=np.array([[0.1]]),
        x_0=np.zeros(2),
        T_f=50.0,
        output_idx=0,
        output_name="position",
    )


def dc_motor() -> RobustProblem:
    """3. DC-motor position servo (MATLAB params); disturbance = load torque on speed."""
    J, b, K, Re, L = 0.01, 0.1, 0.01, 1.0, 0.5
    A = np.array([[0.0, 1.0, 0.0], [0.0, -b / J, K / J], [0.0, -K / L, -Re / L]])
    B = np.array([[0.0], [0.0], [1.0 / L]])
    B_w = np.array([[0.0], [1.0 / J], [0.0]])  # load torque enters on angular acceleration
    return RobustProblem(
        name="DC motor servo",
        A=A,
        B=B,
        B_w=B_w,
        Q=np.diag([1.0, 0.0, 0.0]),
        R=np.array([[0.01]]),
        x_0=np.zeros(3),
        T_f=6.0,
        output_idx=0,
        output_name="shaft angle",
    )


def cruise_control() -> RobustProblem:
    """4. Car longitudinal speed with engine lag; disturbance = road grade / drag."""
    m, b, tau = 1000.0, 50.0, 0.5
    A = np.array([[-b / m, 1.0 / m], [0.0, -1.0 / tau]])  # [speed, engine force]
    B = np.array([[0.0], [1.0 / tau]])
    B_w = np.array([[1.0 / m], [0.0]])  # disturbance force on the car body
    return RobustProblem(
        name="car cruise control",
        A=A,
        B=B,
        B_w=B_w,
        Q=np.diag([1.0, 0.0]),
        R=np.array([[1e-4]]),
        x_0=np.zeros(2),
        T_f=30.0,
        output_idx=0,
        output_name="speed error",
    )


def bicycle_lateral() -> RobustProblem:
    """5. Car lateral dynamics (linear bicycle model); disturbance = side wind / bank.

    States [v_y, r] (lateral velocity, yaw rate), input = steering angle.
    Params: m=1500 kg, Iz=2250, a=1.2, b=1.6 m, Cf=Cr=6e4 N/rad, Vx=20 m/s.
    """
    m, Iz, a, b, Cf, Cr, Vx = 1500.0, 2250.0, 1.2, 1.6, 6.0e4, 6.0e4, 20.0
    A = np.array(
        [
            [-(Cf + Cr) / (m * Vx), -Vx - (a * Cf - b * Cr) / (m * Vx)],
            [-(a * Cf - b * Cr) / (Iz * Vx), -(a**2 * Cf + b**2 * Cr) / (Iz * Vx)],
        ]
    )
    B = np.array([[Cf / m], [a * Cf / Iz]])
    B_w = np.array([[1.0 / m], [0.0]])  # lateral wind force
    return RobustProblem(
        name="car lateral (bicycle)",
        A=A,
        B=B,
        B_w=B_w,
        Q=np.diag([1.0, 1.0]),
        R=np.array([[1.0]]),
        x_0=np.zeros(2),
        T_f=12.0,
        output_idx=0,
        output_name="lateral velocity",
    )


def inverted_pendulum() -> RobustProblem:
    """6. Inverted pendulum on a cart, linearised upright (unstable); disturbance = cart force.

    Classic Franklin/MATLAB parameters: M=0.5, m=0.2, b=0.1, I=0.006, g=9.8, l=0.3.
    """
    M, mp, b, inertia, g, length = 0.5, 0.2, 0.1, 0.006, 9.8, 0.3
    p = inertia * (M + mp) + M * mp * length**2
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -(inertia + mp * length**2) * b / p, (mp**2 * g * length**2) / p, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -(mp * length * b) / p, mp * g * length * (M + mp) / p, 0.0],
        ]
    )
    B = np.array([[0.0], [(inertia + mp * length**2) / p], [0.0], [mp * length / p]])
    B_w = B.copy()  # disturbance is a force on the cart, same channel as the input
    return RobustProblem(
        name="inverted pendulum",
        A=A,
        B=B,
        B_w=B_w,
        Q=np.diag([5.0, 0.0, 20.0, 0.0]),
        R=np.array([[1.0]]),
        x_0=np.zeros(4),
        T_f=8.0,
        output_idx=2,
        output_name="pendulum angle",
    )


def aircraft_short_period() -> RobustProblem:
    """7. Aircraft longitudinal short-period mode; disturbance = vertical gust on alpha.

    Representative lightly-damped short-period: states [alpha, q], elevator input.
    """
    A = np.array([[-0.7, 1.0], [-5.0, -0.7]])
    B = np.array([[-0.10], [-8.0]])
    B_w = np.array([[1.0], [0.0]])  # vertical gust perturbs angle of attack
    return RobustProblem(
        name="aircraft short-period",
        A=A,
        B=B,
        B_w=B_w,
        Q=np.diag([10.0, 1.0]),
        R=np.array([[1.0]]),
        x_0=np.zeros(2),
        T_f=12.0,
        output_idx=0,
        output_name="angle of attack",
    )


def quadrotor_planar() -> RobustProblem:
    """8. Drone: planar quadrotor single lateral axis [y, ydot, phi, phidot].

    Small-angle: ydot-dot = -g*phi, phidot-dot = tau/Ix. Underactuated;
    disturbance = lateral wind force on the body.
    """
    g, Ix = 9.81, 0.0123
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -g, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    B = np.array([[0.0], [0.0], [0.0], [1.0 / Ix]])
    B_w = np.array([[0.0], [1.0], [0.0], [0.0]])  # wind force on lateral acceleration
    return RobustProblem(
        name="quadrotor (planar)",
        A=A,
        B=B,
        B_w=B_w,
        Q=np.diag([10.0, 1.0, 1.0, 0.1]),
        R=np.array([[0.1]]),
        x_0=np.zeros(4),
        T_f=10.0,
        output_idx=0,
        output_name="lateral position",
    )


CATALOG = [
    cart,
    mass_spring_damper,
    dc_motor,
    cruise_control,
    bicycle_lateral,
    inverted_pendulum,
    aircraft_short_period,
    quadrotor_planar,
]
