"""Reproduce every number quoted on chapter A1 of "Advanced MPC and Learning MPC",
"State-space models and the double integrator" (a01-state-space-models-and-the-double-integrator.html).

Usage:  python a01_numbers.py
Needs numpy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
"""
from __future__ import annotations

import numpy as np

from advanced_mpc_lmpc.collision import vehicle_matrices
from advanced_mpc_lmpc.linear import rollout

np.set_printoptions(precision=4, suppress=True)


def section(title):
    print(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
section("Part 2: the sampled double integrator (zero-order hold) and the canonical matrices")
for Ts in (1.0, 0.5, 0.25):
    A, B = vehicle_matrices(Ts, "zoh")
    print(f"Ts = {Ts}: A = {A.tolist()}, B = {B.ravel().tolist()}")
A, B = vehicle_matrices(1.0, "zoh")
Ae, Be = vehicle_matrices(1.0, "euler")
print("Euler at Ts = 1: A =", Ae.tolist(), "B =", Be.ravel().tolist())

# --------------------------------------------------------------------------------------------------------------
section("Hero: the cart under three input sequences from x0 = (-3, 0), Ts = 1 (zero-order hold)")
x0 = np.array([-3.0, 0.0])
sequences = {
    "push then brake (1, 1, 0, -1, -1)": [1, 1, 0, -1, -1],
    "brake earlier (1, 1, -1, -1, 0)": [1, 1, -1, -1, 0],
    "half-size inputs, 8 moves": [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
}
for name, U in sequences.items():
    X = rollout(A, B, x0, np.asarray(U, float)[:, None])
    print(f"{name}: states", [tuple(np.round(x, 3).tolist()) for x in X])
    print(f"   final state {np.round(X[-1], 3)}, max |v| = {np.abs(X[:, 1]).max():.3f}, max |p| = {np.abs(X[:, 0]).max():.3f}",
          "box |p|<=4, |v|<=3 respected" if (np.abs(X[:, 0]) <= 4).all() and (np.abs(X[:, 1]) <= 3).all() else "LEAVES the box")

# --------------------------------------------------------------------------------------------------------------
section("Part 3: Euler against zero-order hold, same inputs (1, 1, 0, -1, -1) from (-3, 0)")
U = np.array([1, 1, 0, -1, -1.0])[:, None]
Xz = rollout(A, B, x0, U)
Xe = rollout(Ae, Be, x0, U)
print("ZOH   positions", np.round(Xz[:, 0], 3), "velocities", np.round(Xz[:, 1], 3))
print("Euler positions", np.round(Xe[:, 0], 3), "velocities", np.round(Xe[:, 1], 3))
print("position difference at each step", np.round(Xz[:, 0] - Xe[:, 0], 3), "= Ts^2/2 * cumulative input")
print("velocities identical:", np.allclose(Xz[:, 1], Xe[:, 1]))


def exact_continuous(x0, U, Ts):
    """The continuous-time cart with piecewise-constant acceleration: ZOH is exact at the samples."""
    p, v = x0
    out = [(p, v)]
    for a in np.asarray(U).ravel():
        p, v = p + Ts * v + 0.5 * Ts**2 * a, v + Ts * a
        out.append((p, v))
    return np.asarray(out)


print("ZOH equals the continuous cart at the samples:", np.allclose(Xz, exact_continuous(x0, U, 1.0)))

# --------------------------------------------------------------------------------------------------------------
section("Part 3: Euler's position lags the true one by Ts/2 times the velocity (push 2.5 s, brake 2.5 s)")
for Ts in (1.0, 0.5, 0.25, 0.1):
    steps = int(round(5 / Ts))
    a_seq = np.r_[np.ones(steps // 2), -np.ones(steps - steps // 2)]
    Az, Bz = vehicle_matrices(Ts, "zoh"); Au, Bu = vehicle_matrices(Ts, "euler")
    Xz = rollout(Az, Bz, x0, a_seq[:, None]); Xe = rollout(Au, Bu, x0, a_seq[:, None])
    gap = Xz[:, 0] - Xe[:, 0]
    print(f"Ts = {Ts:5.2f}: largest position gap {gap.max():.4f} = Ts/2 * max v = {Ts / 2 * Xz[:, 1].max():.4f};"
          f" final positions ZOH {Xz[-1, 0]:.4f}, Euler {Xe[-1, 0]:.4f}")

# --------------------------------------------------------------------------------------------------------------
section("Part 4: eigenvalues of A (the open loop) and the canonical constraint sets")
print("eigenvalues of A:", np.linalg.eigvals(A), "(a double integrator is marginally stable: not Schur)")
print("state box |p| <= 4, |v| <= 3 as H x <= h:")
H = np.vstack([np.eye(2), -np.eye(2)]); h = np.array([4, 3, 4, 3.0])
print("H =", H.tolist(), "h =", h.tolist())
print("input interval |u| <= 1; from (-3, 0) with u = 1 held, the velocity reaches 3 after 3 steps, at position", rollout(A, B, x0, np.ones((3, 1)))[-1, 0])

# --------------------------------------------------------------------------------------------------------------
section("Exercise 2: the double integrator with Ts = 0.5, rollout of (0.4, 0.2, 0, -0.2, -0.3, 0.1) from (-2, 0.25)")
A5, B5 = vehicle_matrices(0.5, "zoh")
X5 = rollout(A5, B5, [-2.0, 0.25], np.array([0.4, 0.2, 0, -0.2, -0.3, 0.1])[:, None])
print("A =", A5.tolist(), "B =", B5.ravel().tolist())
print("states", [tuple(np.round(x, 4).tolist()) for x in X5])

# --------------------------------------------------------------------------------------------------------------
section("Exercise 4: the unstable scalar plant x+ = 1.2 x + u with u = 0, from x = 1")
x = 1.0
xs = [x]
for _ in range(10):
    x = 1.2 * x; xs.append(x)
print("x after 10 steps:", round(xs[-1], 4), "; eigenvalue 1.2 > 1, not Schur")
