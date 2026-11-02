"""Reproduce every number quoted on chapter A11 of "Advanced MPC and Learning MPC",
"Terminal ingredients and the stability theorem" (a11-terminal-ingredients-and-the-stability-theorem.html).

Usage:  python a11_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from advanced_mpc_lmpc.linear import dlqr
from advanced_mpc_lmpc.mpc import LinearMPC
from advanced_mpc_lmpc.polytopes import (HPolytope, closed_loop_admissible_set_scalar_input, maximal_control_invariant,
                                         maximal_invariant, n_step_controllable_sets)

np.set_printoptions(precision=4, suppress=True)

A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])
XLB, XUB = np.array([-4.0, -3.0]), np.array([4.0, 3.0])
X = HPolytope.box(XLB, XUB)
K, P, eig = dlqr(A, B, Q, R); Acl = A - B @ K
ZERO = HPolytope(np.vstack([np.eye(2), -np.eye(2)]), np.zeros(4))


def section(title):
    print(f"\n=== {title}")


def area(S):
    V = S.vertices_2d()
    return ConvexHull(V).volume if len(V) >= 3 else 0.0


def stage(x, u):
    return float(x @ Q @ x + u @ R @ u)


def full_value(x0, sol):
    Xp, U = sol.X, sol.U
    return float(x0 @ Q @ x0) + sum(float(x @ Q @ x) for x in Xp[:-1]) + sum(float(u @ R @ u) for u in U) + float(Xp[-1] @ P @ Xp[-1])


def ingredients(umax):
    XK = closed_loop_admissible_set_scalar_input(X, K, -umax, umax)
    Xf, hist = maximal_invariant(Acl, XK)
    return XK, Xf, len(hist) - 1


def controller(design, N, umax, Xf):
    kw = dict(x_lower=XLB, x_upper=XUB, u_lower=[-umax], u_upper=[umax])
    if design == "zero":
        return LinearMPC(A, B, Q, R, N, P=Q, terminal_H=ZERO.H, terminal_h=ZERO.h, **kw)
    if design == "set":
        return LinearMPC(A, B, Q, R, N, P=P, terminal_H=Xf.H, terminal_h=Xf.h, **kw)
    return LinearMPC(A, B, Q, R, N, P=P, **kw)


def loop(mpc, x0, steps=40, tol=1e-3):
    x = np.asarray(x0, float).copy(); xs, us, vals, stages = [x.copy()], [], [], []
    for t in range(steps):
        sol = mpc.solve(x)
        if not sol.success:
            return dict(xs=np.array(xs), us=np.array(us), vals=vals, stages=stages, infeasible_at=t)
        vals.append(full_value(x, sol)); u = sol.U[0]; stages.append(stage(x, u)); us.append(u.copy())
        x = A @ x + B @ u; xs.append(x.copy())
        if np.linalg.norm(x) < tol:
            break
    return dict(xs=np.array(xs), us=np.array(us), vals=vals, stages=stages, infeasible_at=-1)


# --------------------------------------------------------------------------------------------------------------
section("Part 3: the LQR ingredients, full motor |u| <= 1 and weaker motor |u| <= 0.5")
print("K =", K.ravel().round(4), "P =", P.round(4).tolist(), "rho(A - BK) =", np.abs(eig).max().round(4))
print("DARE identity residual:", np.abs(Acl.T @ P @ Acl - P + Q + K.T @ R @ K).max())
for umax in (1.0, 0.5):
    XK, Xf, iters = ingredients(umax)
    V = Xf.vertices_2d()
    print(f"\n|u| <= {umax}: X_K has {len(XK.vertices_2d())} vertices, area {area(XK):.3f}; X_f after {iters} iterations has {len(V)} vertices, area {area(Xf):.3f}:")
    for v in V:
        u = float(-(K @ v).item()); vn = Acl @ v
        res = float(vn @ P @ vn - v @ P @ v + v @ Q @ v + R.item() * u * u)
        print(f"   vertex ({v[0]:7.4f}, {v[1]:7.4f}): u = -Kv = {u:7.4f}, in X: {X.contains(v)}, successor ({vn[0]:7.4f}, {vn[1]:7.4f}) in X_f: {Xf.contains(vn, 1e-7)}, V_f(x+) - V_f(x) + q = {res:.1e}")
    C, hC = maximal_control_invariant(A, B, X, -umax, umax)
    print(f"   C_inf: {len(C.vertices_2d())} vertices, area {area(C):.3f}, {len(hC) - 1} iterations")

# --------------------------------------------------------------------------------------------------------------
section("Parts 4-5: feasible regions of the two designs against the horizon (areas)")
for umax in (0.5, 1.0):
    XK, Xf, _ = ingredients(umax)
    Kz = n_step_controllable_sets(A, B, X, ZERO, -umax, umax, 12)
    Kf = n_step_controllable_sets(A, B, X, Xf, -umax, umax, 12)
    C, _ = maximal_control_invariant(A, B, X, -umax, umax)
    az = [round(area(S), 2) for S in Kz]; af = [round(area(S), 2) for S in Kf]
    print(f"|u| <= {umax}: C_inf area {area(C):.2f}")
    print("   x_N = 0   regions, N = 0..12:", az)
    print("   x_N in X_f regions, N = 0..12:", af)
    print("   fraction of C_inf at N = 5: zero", round(az[5] / area(C), 3), "set", round(af[5] / area(C), 3),
          "; the zero design fills C_inf at N =", next(n for n, a in enumerate(az) if abs(a - area(C)) < 1e-2),
          ", the set design at N =", next(n for n, a in enumerate(af) if abs(a - area(C)) < 1e-2))

# --------------------------------------------------------------------------------------------------------------
section("Hero: weaker motor |u| <= 0.5, N = 5: a start feasible with X_f but not with x_N = 0")
umax = 0.5
XK, Xf, _ = ingredients(umax)
Kz = n_step_controllable_sets(A, B, X, ZERO, -umax, umax, 5)[-1]
Kf = n_step_controllable_sets(A, B, X, Xf, -umax, umax, 5)[-1]
cands = [(float(p), float(v)) for p in np.arange(-4, 4.01, 0.5) for v in np.arange(-3, 3.01, 0.5) if Kf.contains([p, v], 1e-9) and not Kz.contains([p, v], 1e-9)]
print(f"{len(cands)} grid states (spacing 0.5) are in the X_f region and not in the zero region, e.g. {cands[:6]}")
x0 = np.array([-3.5, -0.5]) if any(abs(c[0] + 3.5) < 1e-9 and abs(c[1] + 0.5) < 1e-9 for c in cands) else np.array(cands[0])
print("chosen start:", x0, "in zero region:", Kz.contains(x0, 1e-9), "in X_f region:", Kf.contains(x0, 1e-9))
for design in ("zero", "set", "none"):
    r = loop(controller(design, 5, umax, Xf), x0)
    if r["infeasible_at"] == 0:
        print(f"design {design}: infeasible at the start"); continue
    resid = np.array([r["vals"][t + 1] - r["vals"][t] + r["stages"][t] for t in range(len(r["vals"]) - 1)])
    print(f"design {design}: {len(r['us'])} steps, final {r['xs'][-1].round(5)}, saturated on {int(np.sum(np.isclose(np.abs(r['us'][:, 0]), umax, atol=1e-6)))},"
          f" first input {r['us'][0, 0]:.4f}, values {np.round(r['vals'][:4], 3)}, largest residual {resid.max():.1e}, positive on {int(np.sum(resid > 1e-8))} of {len(resid)}")
r = loop(controller("set", 5, umax, Xf), x0)
sol = controller("set", 5, umax, Xf).solve(x0)
print("terminal design: predicted terminal state of the first plan", sol.X[-1].round(4), "in X_f:", Xf.contains(sol.X[-1], 1e-7))
print("value drops from", round(r["vals"][0], 4), "to", round(r["vals"][1], 4), "by", round(r["vals"][0] - r["vals"][1], 4), "against the first stage cost", round(r["stages"][0], 4))

# --------------------------------------------------------------------------------------------------------------
section("Part 6: inside X_f the MPC is the LQR (full motor, N = 5)")
umax = 1.0
XK, Xf, _ = ingredients(umax)
ctrl = controller("set", 5, umax, Xf)
rng = np.random.default_rng(1); V = Xf.vertices_2d(); worst_J = worst_u = 0.0
for _ in range(200):
    w = rng.dirichlet(np.ones(len(V))); x = w @ V
    sol = ctrl.solve(x)
    worst_J = max(worst_J, abs(full_value(x, sol) - float(x @ P @ x))); worst_u = max(worst_u, abs(sol.U[0, 0] + float((K @ x).item())))
print(f"over 200 random states of X_f: |J_5^*(x) - x'Px| <= {worst_J:.1e}, |u_0^*(x) + Kx| <= {worst_u:.1e}")

# --------------------------------------------------------------------------------------------------------------
section("Part 7: the design without terminal ingredients can raise the value (weaker motor, N = 3)")
umax = 0.5
XK, Xf, _ = ingredients(umax)
ctrl_none = controller("none", 3, umax, Xf); ctrl_set = controller("set", 3, umax, Xf)
rises, lost = [], []
for p in np.arange(-4, 4.01, 0.5):
    for v in np.arange(-3, 3.01, 0.5):
        r = loop(ctrl_none, [p, v], steps=30)
        if r["infeasible_at"] > 0:
            lost.append((float(p), float(v), r["infeasible_at"]))
        elif r["infeasible_at"] < 0 and len(r["vals"]) > 1:
            resid = np.array([r["vals"][t + 1] - r["vals"][t] + r["stages"][t] for t in range(len(r["vals"]) - 1)])
            if resid.max() > 1e-6:
                rises.append((float(p), float(v), int(np.sum(resid > 1e-6)), resid.max()))
print(f"grid starts where the cost-only design is feasible, then loses feasibility: {len(lost)}, e.g. {lost[:4]}")
print(f"grid starts where its value fails to drop by q on some step: {len(rises)}, e.g. {[(p, v, n) for p, v, n, _ in rises[:4]]}")
if lost:
    p, v, k = lost[0]
    print(f"   from ({p}, {v}) the terminal-set design with N = 3 is {'feasible' if ctrl_set.solve(np.array([p, v])).success else 'infeasible at once'}")

# --------------------------------------------------------------------------------------------------------------
section("Part 5: the zero terminal constraint: minimum-norm plan and local controllability (full motor)")
N = 5
C5 = np.hstack([np.linalg.matrix_power(A, N - 1 - j) @ B for j in range(N)])
for x in (np.array([0.1, 0.1]), np.array([0.5, -0.3])):
    U = -np.linalg.pinv(C5) @ np.linalg.matrix_power(A, N) @ x
    print(f"x = {x}: minimum-norm 5-step plan to the origin U = {U.round(4)}, largest |u| = {np.abs(U).max():.4f}, linear in x")

# --------------------------------------------------------------------------------------------------------------
section("Exercise: the terminal set shrinks with the input bound")
for umax in (1.0, 0.75, 0.5, 0.25):
    XK, Xf, it = ingredients(umax)
    print(f"|u| <= {umax}: X_K area {area(XK):.3f}, X_f {len(Xf.vertices_2d())} vertices, area {area(Xf):.3f}, {it} iterations")

# --------------------------------------------------------------------------------------------------------------
section("Exercise: the terminal weight R trades aggressiveness for admissibility (|u| <= 0.5, N = 5)")
for Rv in (0.02, 0.2, 2.0):
    Kr, Pr, er = dlqr(A, B, Q, np.array([[Rv]]))
    XKr = closed_loop_admissible_set_scalar_input(X, Kr, -0.5, 0.5); Xfr, _ = maximal_invariant(A - B @ Kr, XKr)
    reg = n_step_controllable_sets(A, B, X, Xfr, -0.5, 0.5, 5)[-1]
    print(f"R = {Rv}: K = {Kr.ravel().round(3)}, rho = {np.abs(er).max():.3f}, area(X_f) = {area(Xfr):.2f}, area(N = 5 region) = {area(reg):.2f}")
