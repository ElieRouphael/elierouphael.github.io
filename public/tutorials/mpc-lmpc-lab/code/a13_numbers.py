"""Reproduce every number quoted on chapter A13 of "Advanced MPC and Learning MPC",
"The learned terminal value and the LMPC problem" (a13-the-learned-terminal-value-and-the-lmpc-problem.html).

Usage:  python a13_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
"""
from __future__ import annotations

import numpy as np

from advanced_mpc_lmpc.lmpc import LMPCController, convex_safe_value, stage_cost, trajectory_cost_to_go
from advanced_mpc_lmpc.mpc import LinearMPC

np.set_printoptions(precision=4, suppress=True)


def section(title):
    print(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
section("Hero and Part 2: the scalar integrator x+ = x + u, q(x, u) = x^2 + 0.1 u^2, two successful runs")
runs = {"A: 4 -> 3 -> 2 -> 1 -> 0": ([4, 3, 2, 1, 0], [-1, -1, -1, -1]),
        "B: -3 -> -2 -> -1 -> 0": ([-3, -2, -1, 0], [1, 1, 1])}
S, V, Us = [[0.0]], [0.0], [0.0]
for name, (xs, us) in runs.items():
    J = trajectory_cost_to_go(np.array(xs, float)[:, None], np.array(us, float)[:, None], [[1.0]], [[0.1]])
    print(f"{name}: costs-to-go {[round(float(j), 3) for j in J]}")
    S += [[float(x)] for x in xs[:-1]]; V += [float(j) for j in J[:-1]]; Us += [float(u) for u in us]
S, V, Us = np.array(S), np.array(V), np.array(Us)
print("stored (x, J) pairs:", sorted((float(s[0]), round(float(v), 1)) for s, v in zip(S, V)))
for x in (2.5, 1.5, 3.7, -0.5):
    val, lam, ok = convex_safe_value([x], S, V)
    print(f"V_c({x}) = {val:.4f} with weights", {float(S[i, 0]): round(float(lam[i]), 3) for i in np.where(lam > 1e-6)[0]})

section("Part 3: the decrease at x = 2.5")
val, lam, _ = convex_safe_value([2.5], S, V)
u = float(lam @ Us); xp = 2.5 + u
vp = convex_safe_value([xp], S, V)[0]
h = 2.5**2 + 0.1 * u**2
print(f"weights give u = {u}, x+ = {xp}, V_c(x+) = {vp:.4f}, q(x, u) = {h:.4f}, V_c(x+) - V_c(x) = {vp - val:.4f} <= -q: slack {-(vp - val) - h:.4f}")
print("the slack is the convexity gap of x^2 between 2.5^2 and the average of 2^2 and 3^2:", 0.5 * (4 + 9) - 6.25)

section("Part 2 widget: a worse third run through 2, 1.6, 1.0, 0.4")
xs, us = [2, 1.6, 1.0, 0.4, 0], [-0.4, -0.6, -0.6, -0.4]
Jw = trajectory_cost_to_go(np.array(xs, float)[:, None], np.array(us, float)[:, None], [[1.0]], [[0.1]])
print("its costs-to-go:", [round(float(j), 3) for j in Jw])
S2 = np.vstack([S, np.array(xs[:-1], float)[:, None]]); V2 = np.r_[V, Jw[:-1]]
for x in (2.0, 1.6, 1.0, 0.4, 0.2, 0.7):
    print(f"V_c({x}): before {convex_safe_value([x], S, V)[0]:.4f}, after {convex_safe_value([x], S2, V2)[0]:.4f}")

# --------------------------------------------------------------------------------------------------------------
section("Parts 4 and 5: the LMPC problem on the canonical cart, from the slow N = 2 seed of chapter A12")
A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])
XLB, XUB, ULB, UUB = [-4, -3], [4, 3], [-1], [1]
XS = np.array([-3.5, 2.5])


def bridge(X, U):
    X, U = list(X), list(U); x = np.asarray(X[-1], float)
    for u in -np.linalg.solve(np.hstack([A @ B, B]), A @ A @ x):
        x = A @ x + B @ [u]; U.append(np.array([u])); X.append(x)
    X[-1] = np.zeros(2); return np.asarray(X), np.asarray(U)


mpc2 = LinearMPC(A, B, Q, R, 2, P=np.zeros((2, 2)), x_lower=XLB, x_upper=XUB, u_lower=ULB, u_upper=UUB)
x = XS.copy(); X, U = [x.copy()], []
while np.linalg.norm(x) > 1e-3:
    u = mpc2.solve(x).U[0]; x = A @ x + B @ u; U.append(u); X.append(x.copy())
X, U = bridge(X, U)
J = trajectory_cost_to_go(X, U, Q, R)
print(f"seed: {len(U)} inputs, cost {J[0]:.6f}; costs-to-go along it: {[round(float(j), 3) for j in J[:8]]} ...")
for N in (2, 3, 4):
    ctrl = LMPCController(A, B, Q, R, N, XLB, XUB, ULB, UUB); ctrl.add_trajectory(X, U)
    s = ctrl.solve(XS)
    act = np.where(s.lambdas > 1e-6)[0]
    print(f"N = {N}: first LMPC input {s.U[0, 0]:.4f} (the seed applied {U[0, 0]:.4f}), objective {s.objective:.6f} <= stored cost {J[0]:.6f},"
          f" terminal state {s.terminal_state}, {len(act)} active weights", {int(i): round(float(s.lambdas[i]), 4) for i in act},
          "on stored rows", [tuple(np.round(s.safe_states[i], 3).tolist()) for i in act], f"with values {[round(float(s.safe_values[i]), 3) for i in act]}")
    if N == 3:
        # the shifted sequence is feasible at the next state: solve from x1 with the shifted warm start, and check the appended input
        x1 = A @ XS + B @ s.U[0]
        u_app = s.lambdas @ s.safe_inputs
        x_app = A @ s.terminal_state + B @ u_app
        xdata = s.safe_states[s.next_index].T @ s.lambdas
        print(f"   shifted sequence: appended input {u_app[0]:.4f} (admissible), successor {x_app} = mixture of stored successors {xdata}; "
              f"solve from x1 = {x1}: success {ctrl.solve(x1, warm=ctrl.shifted_warm_start(x1, s)).success}")
        s1 = ctrl.solve(x1)
        print(f"   value decrease: J(x1) = {s1.objective:.6f} <= J(x0) - q(x0, u0) = {s.objective - stage_cost(XS, s.U[0], Q, R):.6f}")

section("The stage cost by hand at x = (-2, 1), u = 0.5")
print("x'Qx =", (-2.0)**2 * 1 + 1**2 * 0.2, " R u^2 =", 0.2 * 0.25, " total", stage_cost([-2, 1], [0.5], Q, R))
