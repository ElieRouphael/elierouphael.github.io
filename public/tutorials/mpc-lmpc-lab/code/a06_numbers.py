"""Reproduce every number quoted on chapter A6 of "Advanced MPC and Learning MPC",
"Receding horizon: from a plan to feedback" (a06-receding-horizon-from-a-plan-to-feedback.html).

Usage:  python a06_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
"""
from __future__ import annotations

import numpy as np

from advanced_mpc_lmpc.linear import dlqr
from advanced_mpc_lmpc.mpc import LinearMPC
from advanced_mpc_lmpc.polytopes import HPolytope, closed_loop_admissible_set_scalar_input, maximal_invariant

np.set_printoptions(precision=4, suppress=True)

# the canonical system
A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])
XLB, XUB = np.array([-4.0, -3.0]), np.array([4.0, 3.0])
X = HPolytope.box(XLB, XUB)
K, P, eig = dlqr(A, B, Q, R)


def section(title):
    print(f"\n=== {title}")


def stage(x, u):
    return float(x @ Q @ x + u @ R @ u)


def full_value(x0, sol, Pterm):
    """The full finite-horizon value x0'Qx0 + sum_{k=1}^{N-1} x_k'Qx_k + sum u'Ru + x_N'P x_N."""
    Xp, U = sol.X, sol.U
    J = float(x0 @ Q @ x0) + sum(float(x @ Q @ x) for x in Xp[:-1]) + sum(float(u @ R @ u) for u in U)
    return J + float(Xp[-1] @ Pterm @ Xp[-1])


def make(design, N, umax=1.0):
    """design: 'PQ' (P = Q, no terminal set), 'P' (LQR P, no set), 'set' (LQR P and X_f)."""
    if design == "PQ":
        return LinearMPC(A, B, Q, R, N, P=Q, x_lower=XLB, x_upper=XUB, u_lower=[-umax], u_upper=[umax]), Q
    XK = closed_loop_admissible_set_scalar_input(X, K, -umax, umax)
    Xf, _ = maximal_invariant(A - B @ K, XK)
    if design == "P":
        return LinearMPC(A, B, Q, R, N, P=P, x_lower=XLB, x_upper=XUB, u_lower=[-umax], u_upper=[umax]), P
    return LinearMPC(A, B, Q, R, N, P=P, x_lower=XLB, x_upper=XUB, u_lower=[-umax], u_upper=[umax],
                     terminal_H=Xf.H, terminal_h=Xf.h), P


def loop(mpc, Pterm, x0, steps=25, tol=1e-3):
    x = np.asarray(x0, float).copy()
    xs, us, vals, plans, stages = [x.copy()], [], [], [], []
    for t in range(steps):
        sol = mpc.solve(x)
        if not sol.success:
            return dict(xs=np.array(xs), us=np.array(us), vals=vals, plans=plans, stages=stages, infeasible_at=t)
        vals.append(full_value(x, sol, Pterm)); plans.append(sol.X.copy())
        u = sol.U[0]; stages.append(stage(x, u)); us.append(u.copy())
        x = A @ x + B @ u; xs.append(x.copy())
        if np.linalg.norm(x) < tol:
            break
    return dict(xs=np.array(xs), us=np.array(us), vals=vals, plans=plans, stages=stages, infeasible_at=-1)


# --------------------------------------------------------------------------------------------------------------
section("Hero: three terminal designs, N = 5, from x0 = (-3.6, 2.4), |u| <= 1")
x0 = np.array([-3.6, 2.4])
for design, name in (("PQ", "P = Q, no terminal set"), ("P", "LQR cost P, no terminal set"), ("set", "LQR cost P and invariant terminal set X_f")):
    mpc, Pterm = make(design, 5)
    r = loop(mpc, Pterm, x0)
    resid = np.array([r["vals"][t + 1] - r["vals"][t] + r["stages"][t] for t in range(len(r["vals"]) - 1)])
    gap = max(np.abs(np.r_[[r["xs"][0]], r["plans"][0]][: len(r["xs"])] - r["xs"][: len(r["plans"][0]) + 1]).max()
              for _ in [0])
    print(f"{name}: {len(r['us'])} moves, final {r['xs'][-1].round(4)}, input saturated on "
          f"{int(np.sum(np.isclose(np.abs(r['us'][:, 0]), 1.0, atol=1e-6)))}, closed-loop cost {sum(r['stages']):.4f}, "
          f"value fails to drop by q on {int(np.sum(resid > 1e-7))} of {len(resid)} steps (largest residual {resid.max():.1e}), "
          f"plan at step 0 against what was applied: largest gap {gap:.4f}")
    print("   values J_N^*:", np.round(r["vals"][:6], 3), "... velocity peak", r["xs"][:, 1].max().round(3))
mpc, Pterm = make("set", 5)
r = loop(mpc, Pterm, x0)
print("Terminal design, first plan from x0:", [tuple(np.round(x, 3).tolist()) for x in r["plans"][0]])
print("Applied states:", [tuple(np.round(x, 3).tolist()) for x in r["xs"][:6]])

# --------------------------------------------------------------------------------------------------------------
section("Part 2: the plan made at each step against what is applied (terminal design)")
for t in range(min(4, len(r["plans"]))):
    plan = np.vstack([r["xs"][t], r["plans"][t]])
    actual = r["xs"][t: t + len(plan)]
    print(f"step {t}: plan ends at {plan[-1].round(3)}; largest difference from the applied states over the overlap {np.abs(plan[:len(actual)] - actual).max():.2e}")

# --------------------------------------------------------------------------------------------------------------
section("Part 4: losing recursive feasibility (no terminal set, P = Q) on a grid of starts")
found = []
for umax in (1.0, 0.5):
    for N in (1, 2, 3):
        mpc, Pterm = make("PQ", N, umax)
        for p0 in np.arange(-4, 4.01, 0.5):
            for v0 in np.arange(-3, 3.01, 0.5):
                rr = loop(mpc, Pterm, [p0, v0], steps=15)
                if rr["infeasible_at"] > 0:
                    found.append((umax, N, p0, v0, rr["infeasible_at"]))
print(f"{len(found)} (u_max, N, p0, v0) combinations are feasible at the start and lose feasibility later")
by = {}
for umax, N, p0, v0, k in found:
    by.setdefault((umax, N), []).append((p0, v0, k))
for key, lst in sorted(by.items()):
    ks = sorted(set(k for *_, k in lst))
    print(f"  |u| <= {key[0]}, N = {key[1]}: {len(lst)} starts, infeasible after {ks} step(s); e.g. {lst[0][:2]} after {lst[0][2]}")
umax, N, p0, v0, k = max(found, key=lambda f: (f[4], -f[1]))
print(f"longest: |u| <= {umax}, N = {N}, start ({p0}, {v0}): feasible for {k} steps, then no admissible plan")
mpc, Pterm = make("PQ", N, umax)
rr = loop(mpc, Pterm, [p0, v0], steps=15)
print("   its states:", [tuple(np.round(x, 3).tolist()) for x in rr["xs"]], "inputs", rr["us"].ravel().round(3))
mpc_s, Pt = make("set", N, umax)
sol = mpc_s.solve(np.array([p0, v0]))
print(f"   the terminal-set design from the same start with the same N: {'feasible' if sol.success else 'infeasible at once (the honest answer)'}")

# --------------------------------------------------------------------------------------------------------------
section("Part 5: the horizon-one pathology on x+ = a x + u, q = x^2 + r u^2")
for a, rr_ in ((1.2, 4.0), (1.1, 5.0)):
    Ks, Ps, es = dlqr(np.array([[a]]), np.array([[1.0]]), np.array([[1.0]]), np.array([[rr_]]))
    print(f"a = {a}, r = {rr_}: N = 1 without terminal cost gives u = 0 and x+ = {a} x (eigenvalue {a}); "
          f"with the Riccati terminal cost P = {Ps.item():.4f}: u = -{Ks.item():.4f} x, eigenvalue {es[0].real:.4f}; "
          f"x after 15 steps from 1 without the terminal cost: {a**15:.1f}")

# --------------------------------------------------------------------------------------------------------------
section("Exercise: the receding-horizon loop against the open-loop plan executed to the end (terminal design, N = 5)")
mpc, Pterm = make("set", 5)
sol = mpc.solve(x0)
x = x0.copy(); cost_open = 0.0
for u in sol.U:
    cost_open += stage(x, u); x = A @ x + B @ u
print(f"executing the first plan open loop for 5 steps costs {cost_open:.4f} and ends at {x.round(4)};")
r = loop(mpc, Pterm, x0)
print(f"the receding-horizon loop costs {sum(r['stages'][:5]):.4f} over its first 5 steps and {sum(r['stages']):.4f} in all ({len(r['us'])} steps); with an exact model and no disturbance the two agree over the first 5 moves to {abs(cost_open - sum(r['stages'][:5])):.1e}")
