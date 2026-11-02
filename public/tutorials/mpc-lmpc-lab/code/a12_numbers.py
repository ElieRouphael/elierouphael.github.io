"""Reproduce every number quoted on chapter A12 of "Advanced MPC and Learning MPC",
"Sampled and convex safe sets" (a12-sampled-and-convex-safe-sets.html).

Usage:  python a12_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).

Three successful runs of the canonical cart from the repeated start x_S = (-3.5, 2.5): plain
receding-horizon MPC with N = 2 and N = 3 and no terminal ingredients, and the saturated LQR law.
Each runs until the state is within 1e-3 of the origin and is then bridged there by two deadbeat inputs, so that it ends exactly on the target.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from advanced_mpc_lmpc.linear import dlqr
from advanced_mpc_lmpc.lmpc import LMPCController, convex_safe_value, trajectory_cost_to_go
from advanced_mpc_lmpc.mpc import LinearMPC

np.set_printoptions(precision=4, suppress=True)

A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])
XLB, XUB, ULB, UUB = [-4, -3], [4, 3], [-1], [1]
XS = np.array([-3.5, 2.5])


def section(title):
    print(f"\n=== {title}")


def bridge(X, U):
    """Two deadbeat inputs u = -[AB B]^{-1} A^2 x take the last state exactly to the origin."""
    X, U = list(X), list(U)
    x = np.asarray(X[-1], float)
    uu = -np.linalg.solve(np.hstack([A @ B, B]), A @ A @ x)
    for u in uu:
        x = A @ x + B @ [u]; U.append(np.array([u])); X.append(x)
    X[-1] = np.zeros(2)
    return np.asarray(X), np.asarray(U)


def run_plain_mpc(N, tol=1e-3):
    mpc = LinearMPC(A, B, Q, R, N, P=np.zeros((2, 2)), x_lower=XLB, x_upper=XUB, u_lower=ULB, u_upper=UUB)
    x = XS.copy(); X, U = [x.copy()], []
    for _ in range(200):
        if np.linalg.norm(x) <= tol: break
        s = mpc.solve(x); assert s.success
        u = s.U[0]; x = A @ x + B @ u; U.append(u); X.append(x.copy())
    return bridge(X, U)


def run_saturated_lqr(tol=1e-3):
    K, _, _ = dlqr(A, B, Q, R)
    x = XS.copy(); X, U = [x.copy()], []
    for _ in range(200):
        if np.linalg.norm(x) <= tol: break
        u = np.clip(-K @ x, -1, 1); x = A @ x + B @ u
        assert (x <= XUB).all() and (x >= XLB).all()
        U.append(u); X.append(x.copy())
    return bridge(X, U)


# --------------------------------------------------------------------------------------------------------------
section("Hero: three successful runs from x_S = (-3.5, 2.5)")
runs = {"plain MPC, N = 2": run_plain_mpc(2), "plain MPC, N = 3": run_plain_mpc(3), "saturated LQR": run_saturated_lqr()}
ctrl = LMPCController(A, B, Q, R, 3, XLB, XUB, ULB, UUB)
for name, (X, U) in runs.items():
    ctrl.add_trajectory(X, U)
    J = trajectory_cost_to_go(X, U, Q, R)
    print(f"{name}: {len(U)} inputs (last two are the bridge), cost J_0 = {J[0]:.4f}, first input {U[0, 0]:.4f}, "
          f"largest |v| {np.abs(X[:, 1]).max():.3f}, first states {[tuple(np.round(x, 3).tolist()) for x in X[1:4]]}")
S, V, Usafe, nxt = ctrl.safe_data()
print(f"sampled safe set: {len(S)} states (the target once, then every non-terminal state of the three runs)")
hull = ConvexHull(S)
print(f"convex safe set: {len(hull.vertices)} vertices, area {hull.volume:.4f} (the box has area 48, C_inf 39.25)")

# --------------------------------------------------------------------------------------------------------------
section("Part 1: what a stored state certifies (the hero's answer for a stored state)")
k = 5
print(f"stored state {k}: s = {S[k]}, from run index and step as stored; input {Usafe[k, 0]:.4f}, successor {S[nxt[k]]},"
      f" dynamics residual {np.linalg.norm(A @ S[k] + B @ Usafe[k] - S[nxt[k]]):.1e}, cost-to-go {V[k]:.4f}")

# --------------------------------------------------------------------------------------------------------------
section("Part 3: a hull point that is on no run, the hero's default probe (-1, 1.8)")
probe = np.array([-1.0, 1.8])
val, lam, ok = convex_safe_value(probe, S, V)
act = np.where(lam > 1e-6)[0]
print("probe inside the hull:", all(hull.equations[:, :2] @ probe + hull.equations[:, 2] <= 1e-9))
print("barycentric weights chosen by the LP:", {int(i): round(float(lam[i]), 4) for i in act}, "on", [tuple(np.round(S[i], 3).tolist()) for i in act])
print("reconstructed state S'lambda =", S.T @ lam, " sum lambda =", lam.sum())
u_mix = float(lam @ Usafe[:, 0]); x_dyn = A @ probe + B @ [u_mix]; x_data = S[nxt].T @ lam
print(f"mixed input {u_mix:.4f} (admissible: |u| <= 1), A x + B u = {x_dyn}, mixture of stored successors = {x_data},"
      f" difference {np.abs(x_dyn - x_data).max():.1e}")
print("successor inside the hull:", all(hull.equations[:, :2] @ x_dyn + hull.equations[:, 2] <= 1e-9))

# --------------------------------------------------------------------------------------------------------------
section("Part 4 widget: four stored states of the N = 2 run with the weights (0.10, 0.25, 0.30, 0.35)")
X2, U2 = runs["plain MPC, N = 2"]
chosen = [1, 4, 7, 10]
lam4 = np.array([0.10, 0.25, 0.30, 0.35])
xd = lam4 @ X2[chosen]; ud = lam4 @ U2[chosen, 0]
print("combination", xd, "input", round(ud, 4), "A x + B u =", A @ xd + B @ [ud], "mixture of successors =", lam4 @ X2[[c + 1 for c in chosen]])

# --------------------------------------------------------------------------------------------------------------
section("Part 5: the nonlinear counter-example f(x, u) = x^2 + u")
for th in (0.5, 0.25):
    x = th * 2.0
    print(f"theta = {th}: mixed state {x}, f(x, 0) = {x * x}, mixture of successors {th * 4.0}")

# --------------------------------------------------------------------------------------------------------------
section("Exercise: the hull with only the N = 2 run, and with all three")
for subset in (["plain MPC, N = 2"], ["plain MPC, N = 2", "plain MPC, N = 3"], list(runs)):
    pts = np.vstack([np.zeros((1, 2))] + [runs[n][0][:-1] for n in subset])
    h = ConvexHull(pts)
    print(f"{' + '.join(subset)}: {len(pts)} states, hull area {h.volume:.4f}")
