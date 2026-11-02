"""Reproduce every number quoted on lab B2 of "Advanced MPC and Learning MPC",
"Learning MPC on a repeated task" (b02-learning-mpc-on-a-repeated-task.html).

Usage:  python b02_numbers.py            # about two minutes: every learning run of the page
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).

The task: the canonical cart x+ = Ax + Bu, |p| <= 4, |v| <= 3, |u| <= 1, q = x'diag(1, 0.2)x + 0.2 u^2,
repeated from x_S = (-3.5, 2.5) to the origin. A run stops once ||x|| <= 1e-3 and is completed by two
deadbeat inputs, so that it ends exactly on the target.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import ConvexHull

from advanced_mpc_lmpc.lmpc import LMPCController, convex_safe_value, solve_ftocp_l1, stage_cost, trajectory_cost_to_go
from advanced_mpc_lmpc.mpc import LinearMPC

np.set_printoptions(precision=4, suppress=True)
A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])
XLB, XUB, ULB, UUB = [-4, -3], [4, 3], [-1], [1]
XS = np.array([-3.5, 2.5]); TOL = 1e-3


def section(title):
    print(f"\n=== {title}")


def bridge(X, U):
    X, U = list(X), list(U); x = np.asarray(X[-1], float)
    for u in -np.linalg.solve(np.hstack([A @ B, B]), A @ A @ x):
        x = A @ x + B @ [u]; U.append(np.array([u])); X.append(x)
    X[-1] = np.zeros(2); return np.asarray(X), np.asarray(U)


def seed_run(N):
    mpc = LinearMPC(A, B, Q, R, N, P=np.zeros((2, 2)), x_lower=XLB, x_upper=XUB, u_lower=ULB, u_upper=UUB)
    x = XS.copy(); X, U = [x.copy()], []
    while np.linalg.norm(x) > TOL:
        s = mpc.solve(x); assert s.success; u = s.U[0]; x = A @ x + B @ u; U.append(u); X.append(x.copy())
    return bridge(X, U)


def lmpc_iteration(ctrl, window):
    x = XS.copy(); X, U = [x.copy()], []; warm = None; first = None; nq = 0; t0 = time.perf_counter()
    while np.linalg.norm(x) > TOL:
        s = ctrl.solve(x, last_n_trajectories=window, warm=warm); nq += 1
        assert s.success, "LMPC problem infeasible"
        first = first or s
        u = s.U[0]; x = A @ x + B @ u; U.append(u); X.append(x.copy()); warm = ctrl.shifted_warm_start(x, s)
    X, U = bridge(X, U); ctrl.add_trajectory(X, U)
    return X, U, first, (time.perf_counter() - t0) / nq


# --------------------------------------------------------------------------------------------------------------
section("Part 1: the long-horizon reference (a converged open-loop solve) and the seeds")
for N in (10, 20, 30, 40):
    s = solve_ftocp_l1(A, B, Q, R, np.zeros((0, 2)), N, XS, XLB, XUB, ULB, UUB)
    print(f"N_ref = {N}: open-loop cost {s.objective:.9f}, terminal norm {np.linalg.norm(s.X[-1]):.1e}")
JREF = solve_ftocp_l1(A, B, Q, R, np.zeros((0, 2)), 30, XS, XLB, XUB, ULB, UUB).objective
for N in (2, 3, 5):
    X, U = seed_run(N); J = trajectory_cost_to_go(X, U, Q, R)
    print(f"seed MPC N = {N}: {len(U)} inputs, first input {U[0, 0]:.4f}, cost {J[0]:.6f}, gap {J[0] - JREF:.6f} ({100 * (J[0] - JREF) / JREF:.1f} percent)")
XSEED, USEED = seed_run(2)
JSEED = trajectory_cost_to_go(XSEED, USEED, Q, R)
print("the seed's states:", [tuple(np.round(x, 3).tolist()) for x in XSEED[:8]], "...")
print("its costs-to-go:", np.round(JSEED[:8], 3).tolist(), "...; stored samples after the seed:", len(USEED) + 1)

# --------------------------------------------------------------------------------------------------------------
section("Parts 2 and 3: the learning iterations for N = 2, 3, 4 with a window of four trajectories")
RESULTS = {}
for N in (2, 3, 4):
    ctrl = LMPCController(A, B, Q, R, N, XLB, XUB, ULB, UUB); ctrl.add_trajectory(XSEED, USEED)
    rows = []
    for it in range(1, 10):
        X, U, first, ms = lmpc_iteration(ctrl, 4)
        rows.append((it, ctrl.iteration_costs[-1], len(U), first.U[0, 0], 1000 * ms, len(ctrl.safe_data(4)[0])))
    RESULTS[N] = (ctrl, rows)
    print(f"N = {N}:  iteration  cost  change  gap/J*  steps  first input  ms/QP  safe states in window")
    prev = JSEED[0]
    for it, c, steps, u0, ms, M in rows:
        print(f"   {it:2d}  {c:.6f}  {c - prev:+.2e}  {(c - JREF) / JREF:.2e}  {steps:2d}  {u0:.4f}  {ms:5.1f}  {M}")
        prev = c
    inc = max(np.diff([JSEED[0]] + [r[1] for r in rows]))
    print(f"   largest iteration-to-iteration change {inc:+.2e} (negative means never an increase)")

# --------------------------------------------------------------------------------------------------------------
section("Part 2: the first LMPC solve from the start, N = 4")
ctrl4 = LMPCController(A, B, Q, R, 4, XLB, XUB, ULB, UUB); ctrl4.add_trajectory(XSEED, USEED)
s = ctrl4.solve(XS); act = np.where(s.lambdas > 1e-6)[0]
print(f"first input {s.U[0, 0]:.4f} (seed: {USEED[0, 0]:.4f}); objective {s.objective:.6f}; terminal state {s.terminal_state};"
      f" active weights {[(int(i), round(float(s.lambdas[i]), 4)) for i in act]} on {[tuple(np.round(s.safe_states[i], 3).tolist()) for i in act]}")

# --------------------------------------------------------------------------------------------------------------
section("Part 4: hull growth against value learning (N = 4)")
ctrl, rows = RESULTS[4]
pts = lambda trajs: np.vstack([np.zeros((1, 2))] + [X[:-1] for X in trajs])
for k in (1, 2, 4, len(ctrl.trajectories)):
    print(f"hull area after {k - 1} iterations ({k} stored trajectories): {ConvexHull(pts(ctrl.trajectories[:k])).volume:.4f}")
print("the cost kept falling meanwhile:", [round(r[1], 6) for r in rows[:4]])
Jlast = ctrl.qfun[-1]; Xl, Ul = ctrl.trajectories[-1], ctrl.inputs[-1]
res = max(abs(Jlast[k] - stage_cost(Xl[k], Ul[k], Q, R) - Jlast[k + 1]) for k in range(len(Ul)))
print(f"backward-recursion residual on the last iteration: {res:.1e}; inputs saturated on the last iteration: {int(np.sum(np.abs(Ul) > 1 - 1e-6))} of {len(Ul)}, on the seed: {int(np.sum(np.abs(USEED) > 1 - 1e-6))} of {len(USEED)}")
XR = solve_ftocp_l1(A, B, Q, R, np.zeros((0, 2)), 30, XS, XLB, XUB, ULB, UUB)
print(f"the reference saturates on {int(np.sum(np.abs(XR.U) > 1 - 1e-6))} steps; largest seed input {np.abs(USEED).max():.3f}")

# --------------------------------------------------------------------------------------------------------------
section("Part 5: the window against all data (N = 4, five iterations)")
ctrl_all = LMPCController(A, B, Q, R, 4, XLB, XUB, ULB, UUB); ctrl_all.add_trajectory(XSEED, USEED)
for it in range(1, 6):
    X, U, first, ms = lmpc_iteration(ctrl_all, None)
    print(f"   all data, iteration {it}: cost {ctrl_all.iteration_costs[-1]:.6f} (window: {RESULTS[4][1][it - 1][1]:.6f}), "
          f"{len(ctrl_all.safe_data(None)[0])} barycentric variables, {1000 * ms:.1f} ms/QP (window: {RESULTS[4][1][it - 1][4]:.1f})")

# --------------------------------------------------------------------------------------------------------------
section("Part 6: the plateau for N = 2 is real: a brute-force search over (u0, u1) with the learned value")
ctrl2, rows2 = RESULTS[2]
S, V, Us, nxt = ctrl2.safe_data(4)
best = np.inf; best_u = (0.0, 0.0); half = 1.0


def two_step_cost(u0, u1):
    x1 = A @ XS + B @ [u0]
    if not ((x1 <= XUB).all() and (x1 >= XLB).all()): return np.inf
    x2 = A @ x1 + B @ [u1]
    if not ((x2 <= XUB).all() and (x2 >= XLB).all()): return np.inf
    val, lam, ok = convex_safe_value(x2, S, V)
    return stage_cost(XS, [u0], Q, R) + stage_cost(x1, [u1], Q, R) + val if ok else np.inf


for refine in range(4):   # a 41 x 41 grid, refined three times around the best point
    g0 = np.clip(np.linspace(best_u[0] - half, best_u[0] + half, 41), -1, 1)
    g1 = np.clip(np.linspace(best_u[1] - half, best_u[1] + half, 41), -1, 1)
    for u0 in g0:
        for u1 in g1:
            c = two_step_cost(u0, u1)
            if c < best: best, best_u = c, (u0, u1)
    half /= 10
print(f"best two-step plan found by the grid: cost {best:.6f} at (u0, u1) = ({best_u[0]:.4f}, {best_u[1]:.4f}); the stored cost of the last N = 2 iteration is {rows2[-1][1]:.6f}, its first two inputs {ctrl2.inputs[-1][0, 0]:.4f}, {ctrl2.inputs[-1][1, 0]:.4f}")
print(f"so with the data it has, the N = 2 controller cannot do better than replay: the plateau {100 * (rows2[-1][1] - JREF) / JREF:.2f} percent above J* is a property of the pessimistic learned value, not of the solver")

# --------------------------------------------------------------------------------------------------------------
section("Exercise: a different seed (N = 3 plain MPC) and N = 4 LMPC")
X3, U3 = seed_run(3)
ctrl = LMPCController(A, B, Q, R, 4, XLB, XUB, ULB, UUB); ctrl.add_trajectory(X3, U3)
costs = [ctrl.iteration_costs[0]]
for it in range(5):
    lmpc_iteration(ctrl, 4); costs.append(ctrl.iteration_costs[-1])
print("costs from the N = 3 seed:", [round(c, 6) for c in costs], f"final gap {costs[-1] - JREF:.1e}")
