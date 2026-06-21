# pvtol_lqr.m - LQR design for vectored thrust aircraft
# RMM, 14 Jan 03
#
# This file works through an LQR based design problem, using the
# planar vertical takeoff and landing (PVTOL) aircraft example from
# Astrom and Murray, Chapter 5.  It is intended to demonstrate the
# basic functionality of the python-control package.
#

from __future__ import annotations

import control as ct
import matplotlib.pyplot as plt  # MATLAB-like plotting functions
import numpy as np

from PyDiffGame.plotting import show

#
# System dynamics
#
# These are the dynamics for the PVTOL system, written in state space
# form.
#

# System parameters
m = 4       # mass of aircraft
J = 0.0475  # inertia around pitch axis
r = 0.25    # distance to center of force
g = 9.8     # gravitational constant
c = 0.05    # damping factor (estimated)

# State space dynamics
xe = [0, 0, 0, 0, 0, 0]  # equilibrium point of interest
ue = [0, m * g]  # (note these are lists, not matrices)

A = np.array(
    [[0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, (-ue[0]*np.sin(xe[2]) - ue[1]*np.cos(xe[2]))/m, -c/m, 0, 0],
     [0, 0, (ue[0]*np.cos(xe[2]) - ue[1]*np.sin(xe[2]))/m, 0, -c/m, 0],
     [0, 0, 0, 0, 0, 0]]
)

# Input matrix
B = np.array(
    [[0, 0], [0, 0], [0, 0],
     [np.cos(xe[2])/m, -np.sin(xe[2])/m],
     [np.sin(xe[2])/m, np.cos(xe[2])/m],
     [r/J, 0]]
)

# Output matrix
C = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
D = np.array([[0, 0], [0, 0]])

#
# Construct inputs and outputs corresponding to steps in xy position
#
# The vectors xd and yd correspond to the states that are the desired
# equilibrium states for the system.  The matrices Cx and Cy are the
# corresponding outputs.
#
# The way these vectors are used is to compute the closed loop system
# dynamics as
#
#   xdot = Ax + B u  =>  xdot = (A-BK)x + K xd
#      u = -K(x - xd)       y = Cx
#
# The closed loop dynamics can be simulated using the "step" command,
# with K*xd as the input vector (assumes that the "input" is unit size,
# so that xd corresponds to the desired steady state.
#

xd = np.array([[1], [0], [0], [0], [0], [0]])
yd = np.array([[0], [1], [0], [0], [0], [0]])

#
# Extract the relevant dynamics for use with SISO library
#
# We define the 'lat' and 'alt' index vectors to consist of the states
# that are relevant to the lateral (x) and vertical (y) dynamics.
#

# Indices for the parts of the state that we want
lat = (0, 2, 3, 5)
alt = (1, 4)

# Decoupled dynamics
Ax = A[np.ix_(lat, lat)]
Bx = B[np.ix_(lat, [0])]
Cx = C[np.ix_([0], lat)]
Dx = D[np.ix_([0], [0])]

Ay = A[np.ix_(alt, alt)]
By = B[np.ix_(alt, [1])]
Cy = C[np.ix_([1], alt)]
Dy = D[np.ix_([1], [1])]


def main() -> None:
    """Run the LQR design study and draw the comparison plots."""

    # Label the plot
    plt.clf()
    plt.suptitle("LQR controllers for vectored thrust aircraft (pvtol-lqr)")

    #
    # LQR design
    #

    # Start with a diagonal weighting
    Qx1 = np.diag([1, 1, 1, 1, 1, 1])
    Qu1a = np.diag([1, 1])
    K1a, _X, _E = ct.lqr(A, B, Qx1, Qu1a)

    # Close the loop: xdot = Ax - B K (x-xd)
    #
    # Note: python-control requires we do this 1 input at a time.

    # Step response for the first input
    H1ax = ct.ss(Ax - Bx @ K1a[np.ix_([0], lat)],
                 Bx @ K1a[np.ix_([0], lat)] @ xd[lat, :], Cx, Dx)
    Tx, Yx = ct.step_response(H1ax, T=np.linspace(0, 10, 100))

    # Step response for the second input
    H1ay = ct.ss(Ay - By @ K1a[np.ix_([1], alt)],
                 By @ K1a[np.ix_([1], alt)] @ yd[alt, :], Cy, Dy)
    Ty, Yy = ct.step_response(H1ay, T=np.linspace(0, 10, 100))

    plt.subplot(221)
    plt.title("Identity weights")
    plt.plot(Tx.T, Yx.T, '-', Ty.T, Yy.T, '--')
    plt.plot([0, 10], [1, 1], 'k-')

    plt.axis([0, 10, -0.1, 1.4])
    plt.ylabel('position')
    plt.legend(('x', 'y'), loc='lower right')

    # Look at different input weightings
    Qu1a = np.diag([1, 1])
    K1a, _X, _E = ct.lqr(A, B, Qx1, Qu1a)
    H1ax = ct.ss(Ax - Bx @ K1a[np.ix_([0], lat)],
                 Bx @ K1a[np.ix_([0], lat)] @ xd[lat, :], Cx, Dx)

    Qu1b = (40 ** 2)*np.diag([1, 1])
    K1b, _X, _E = ct.lqr(A, B, Qx1, Qu1b)
    H1bx = ct.ss(Ax - Bx @ K1b[np.ix_([0], lat)],
                 Bx @ K1b[np.ix_([0], lat)] @ xd[lat, :], Cx, Dx)

    Qu1c = (200 ** 2)*np.diag([1, 1])
    K1c, _X, _E = ct.lqr(A, B, Qx1, Qu1c)
    H1cx = ct.ss(Ax - Bx @ K1c[np.ix_([0], lat)],
                 Bx @ K1c[np.ix_([0], lat)] @ xd[lat, :], Cx, Dx)

    T1, Y1 = ct.step_response(H1ax, T=np.linspace(0, 10, 100))
    T2, Y2 = ct.step_response(H1bx, T=np.linspace(0, 10, 100))
    T3, Y3 = ct.step_response(H1cx, T=np.linspace(0, 10, 100))

    plt.subplot(222)
    plt.title("Effect of input weights")
    plt.plot(T1.T, Y1.T, 'b-')
    plt.plot(T2.T, Y2.T, 'b-')
    plt.plot(T3.T, Y3.T, 'b-')
    plt.plot([0, 10], [1, 1], 'k-')

    plt.axis([0, 10, -0.1, 1.4])

    plt.text(5.3, 0.4, r'$\rho$')

    # Output weighting - change Qx to use outputs
    Qx2 = C.T @ C
    Qu2 = 0.1 * np.diag([1, 1])
    K2, _X, _E = ct.lqr(A, B, Qx2, Qu2)

    H2x = ct.ss(Ax - Bx @ K2[np.ix_([0], lat)],
                Bx @ K2[np.ix_([0], lat)] @ xd[lat, :], Cx, Dx)
    H2y = ct.ss(Ay - By @ K2[np.ix_([1], alt)],
                By @ K2[np.ix_([1], alt)] @ yd[alt, :], Cy, Dy)

    plt.subplot(223)
    plt.title("Output weighting")
    T2x, Y2x = ct.step_response(H2x, T=np.linspace(0, 10, 100))
    T2y, Y2y = ct.step_response(H2y, T=np.linspace(0, 10, 100))
    plt.plot(T2x.T, Y2x.T, T2y.T, Y2y.T)
    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend(('x', 'y'), loc='lower right')

    #
    # Physically motivated weighting
    #
    # Shoot for 1 cm error in x, 10 cm error in y.  Try to keep the angle
    # less than 5 degrees in making the adjustments.  Penalize side forces
    # due to loss in efficiency.
    #

    Qx3 = np.diag([100, 10, 2*np.pi/5, 0, 0, 0])
    Qu3 = 0.1*np.diag([1, 10])
    K3, _X, _E = ct.lqr(A, B, Qx3, Qu3)

    H3x = ct.ss(Ax - Bx @ K3[np.ix_([0], lat)],
                Bx @ K3[np.ix_([0], lat)] @ xd[lat, :], Cx, Dx)
    H3y = ct.ss(Ay - By @ K3[np.ix_([1], alt)],
                By @ K3[np.ix_([1], alt)] @ yd[alt, :], Cy, Dy)
    plt.subplot(224)
    T3x, Y3x = ct.step_response(H3x, T=np.linspace(0, 10, 100))
    T3y, Y3y = ct.step_response(H3y, T=np.linspace(0, 10, 100))
    plt.plot(T3x.T, Y3x.T, T3y.T, Y3y.T)
    plt.title("Physically motivated weights")
    plt.xlabel('time')
    plt.legend(('x', 'y'), loc='lower right')
    plt.tight_layout()

    show()


if __name__ == "__main__":
    main()
