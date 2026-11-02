"""Reproduce every number quoted on lab B6 of "Advanced MPC and Learning MPC",
"Capstone: one design, end to end" (b06-capstone-one-design-end-to-end.html).

Usage:  python b06_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
The grid certificate solves about twenty thousand quadratic programs; the script takes a few minutes.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import ConvexHull

from advanced_mpc_lmpc.linear import dlqr, prediction_matrices
from advanced_mpc_lmpc.lmpc import LMPCController, convex_safe_value, stage_cost, trajectory_cost_to_go
from advanced_mpc_lmpc.mpc import LinearMPC
from advanced_mpc_lmpc.polytopes import (HPolytope, closed_loop_admissible_set_scalar_input, maximal_control_invariant,
                                         maximal_invariant, n_step_controllable_sets)
from advanced_mpc_lmpc.qp import feasibility_margin, solve_qp

np.set_printoptions(precision=4, suppress=True)
A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]]); Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])
XLB, XUB, UMIN, UMAX = np.array([-4.0, -3.0]), np.array([4.0, 3.0]), -1.0, 1.0
X = HPolytope.box(XLB, XUB)
XS = np.array([-3.5, 2.5])
area = lambda P: ConvexHull(P.vertices_2d()).volume if len(P.vertices_2d()) >= 3 else 0.0
q = lambda x, u: float(x @ Q @ x + R[0, 0] * float(np.asarray(u).reshape(-1)[0]) ** 2)


def section(title):
    print(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
section("Part 1: the LQR terminal ingredients")
K, P, eig = dlqr(A, B, Q, R); K = K.reshape(-1); AK = A - B @ K[None, :]
XK = closed_loop_admissible_set_scalar_input(X, K, UMIN, UMAX)
XF, hist = maximal_invariant(AK, XK)
CINF, _ = maximal_control_invariant(A, B, X, UMIN, UMAX)
print(f"K = {K.round(4).tolist()}, P = {P.round(4).tolist()}, |eig(A - BK)| = {abs(eig[0]):.4f}")
print(f"DARE identity residual {np.abs(AK.T @ P @ AK - P + Q + K[:, None] @ R @ K[None, :]).max():.1e}")
print(f"X_f: {len(XF.vertices_2d())} vertices after {len(hist) - 1} iterations, area {area(XF):.3f}; X_K area {area(XK):.3f}; C_inf area {area(CINF):.3f}; X area {area(X):.1f}")
for v in XF.vertices_2d():
    u = -K @ v; xn = AK @ v
    print(f"  vertex {v.round(4).tolist()}: u = {u:.4f}, in X {X.contains(v)}, u in U {UMIN - 1e-9 <= u <= UMAX + 1e-9}, A_K x in X_f {XF.contains(xn, 1e-7)}, "
          f"decrease residual {xn @ P @ xn - v @ P @ v + q(v, u):.1e}")

# --------------------------------------------------------------------------------------------------------------
section("Part 2: the feasible sets X_N = K_N(X_f)")
KXF = n_step_controllable_sets(A, B, X, XF, UMIN, UMAX, 6)
for N, S in enumerate(KXF):
    nested = N == 0 or all(S.contains(v, 1e-7) for v in KXF[N - 1].vertices_2d())
    print(f"  N = {N}: area {area(S):.4f} ({100 * area(S) / area(CINF):.1f} percent of C_inf), nested {nested}, equal to C_inf {S.approximately_equal(CINF, 1e-7)}")

# --------------------------------------------------------------------------------------------------------------
section("Part 3: the grid certificate (terminal design)")
GRID = [np.array([p, v]) for v in np.arange(-3, 3.01, 0.5) for p in np.arange(-4, 4.01, 0.5)]
print(f"grid: {len(GRID)} states")


def mpc_terminal(N): return LinearMPC(A, B, Q, R, N, P=P, x_lower=XLB, x_upper=XUB, u_lower=[UMIN], u_upper=[UMAX], terminal_H=XF.H, terminal_h=XF.h)
def mpc_plain(N): return LinearMPC(A, B, Q, R, N, P=np.zeros((2, 2)), x_lower=XLB, x_upper=XUB, u_lower=[UMIN], u_upper=[UMAX])


def constraints(mpc, x0):
    """The condensed inequality rows G U <= w of the MPC problem at x0 (for the feasibility LP)."""
    n, N = 2, mpc.N; Sx, Su = mpc.Sx, mpc.Su; free = Sx @ x0
    G = [Su, -Su, np.eye(N), -np.eye(N)]; w = [np.tile(XUB, N) - free, np.tile(XLB, N) * -1 + free, np.ones(N) * UMAX, -np.ones(N) * UMIN]
    if mpc.terminal_H is not None:
        G.append(np.asarray(mpc.terminal_H) @ Su[-n:]); w.append(np.asarray(mpc.terminal_h) - np.asarray(mpc.terminal_H) @ free[-n:])
    return np.vstack(G), np.concatenate(w)


def full_value(mpc, x0, sol):
    J = q(x0, sol.U[0]); X_ = sol.X
    for k in range(1, mpc.N): J += q(X_[k - 1], sol.U[k])
    return J + float(X_[-1] @ mpc.P @ X_[-1])


def closed_loop(mpc, x0, steps=40, tol=1e-3):
    x = x0.copy(); xs = [x.copy()]; us = []; vals = []; stages = []; infeasible_at = -1; qps = 0
    for t in range(steps):
        G, w = constraints(mpc, x)
        if feasibility_margin(mpc.N, A_ub=G, b_ub=w) > 1e-6: infeasible_at = t; break
        sol = mpc.solve(x); qps += 1
        u = float(sol.U[0, 0]); vals.append(full_value(mpc, x, sol)); stages.append(q(x, u)); us.append(u)
        x = A @ x + B[:, 0] * u; xs.append(x.copy())
        if np.linalg.norm(x) < tol: break
    resid = [vals[t + 1] - vals[t] + stages[t] for t in range(len(vals) - 1)]
    return dict(xs=np.array(xs), us=us, vals=vals, stages=stages, resid=resid, infeasible_at=infeasible_at, qps=qps)


REF = mpc_terminal(12)
_jinf = {}
def jinf(x):
    k = tuple(np.round(x, 6))
    if k not in _jinf: _jinf[k] = full_value(REF, x, REF.solve(x))
    return _jinf[k]


def grid_run(mpc):
    runs = []
    for x0 in GRID:
        G, w = constraints(mpc, x0)
        if feasibility_margin(mpc.N, A_ub=G, b_ub=w) > 1e-6: continue
        r = closed_loop(mpc, x0)
        r["x0"] = x0; r["doomed"] = not CINF.contains(x0, 1e-9)
        r["converged"] = r["infeasible_at"] < 0 and np.linalg.norm(r["xs"][-1]) <= 1e-3
        r["enter"] = next((i for i, x in enumerate(r["xs"]) if XF.contains(x, 1e-9)), -1)
        r["ratio"] = sum(r["stages"]) / jinf(x0) if r["converged"] and not r["doomed"] and np.linalg.norm(x0) > 0 else np.nan
        runs.append(r)
    return runs


t0 = time.time(); total_qps = 0
for N in (2, 3, 5):
    runs = grid_run(mpc_terminal(N)); qps = sum(r["qps"] for r in runs); total_qps += qps
    print(f"  N = {N}: feasible starts {len(runs)}, lost feasibility {sum(r['infeasible_at'] >= 0 for r in runs)}, "
          f"max decrease residual {max(max(r['resid']) for r in runs if r['resid']):.1e}, max steps to X_f {max(r['enter'] for r in runs)}, "
          f"QPs {qps}, worst cost / J* {np.nanmax([r['ratio'] for r in runs]):.4f}, all converged {all(r['converged'] for r in runs)}")
print(f"  {total_qps} closed-loop QPs in {time.time() - t0:.0f} s")
rng = np.random.default_rng(0); m3 = mpc_terminal(3); dJ = []; dU = []
V = XF.vertices_2d()
for _ in range(200):
    lam = rng.dirichlet(np.ones(len(V))); x = lam @ V
    sol = m3.solve(x); dJ.append(abs(full_value(m3, x, sol) - x @ P @ x)); dU.append(abs(sol.U[0, 0] + K @ x))
print(f"  on 200 random states of X_f: |J_3* - V_f| at most {max(dJ):.1e}, |u_0* + Kx| at most {max(dU):.1e}")
x_deg = np.array([4.0, 0.5]); G, w = constraints(mpc_terminal(5), x_deg)
print(f"  the vertex (4, 0.5) of C_inf: feasibility margin {feasibility_margin(5, A_ub=G, b_ub=w):.1e}, u_0 forced to {-(4.5 - 4) / 0.5:.0f}; the QP solver reports success {mpc_terminal(5).solve(x_deg).success}")

# --------------------------------------------------------------------------------------------------------------
section("Part 4: removing the terminal ingredients (plain design)")
KX = n_step_controllable_sets(A, B, X, X, UMIN, UMAX, 5)
for N in range(1, 6):
    print(f"  K_{N}(X): area {area(KX[N]):.2f}, doomed area {area(KX[N]) - area(CINF):.2f}")
J12 = {tuple(x): jinf(x) for x in GRID if CINF.contains(x, 1e-9)}
m30 = mpc_plain(30); gap = max(abs(J12[tuple(x)] - full_value(m30, x, m30.solve(x))) for x in GRID if CINF.contains(x, 1e-9) and np.linalg.norm(x) > 0)
print(f"  J_12* of the terminal design against J_30^0 of the plain problem on the grid: differ by at most {gap:.1e}")
for N in range(1, 6):
    runs = grid_run(mpc_plain(N))
    lost = [r for r in runs if r["infeasible_at"] >= 0]; lostC = [r for r in lost if not r["doomed"]]
    noconv = [r for r in runs if r["infeasible_at"] < 0 and not r["converged"]]
    ratios = [r["ratio"] for r in runs if not np.isnan(r["ratio"])]
    resid_all = [v for r in runs for v in r["resid"]]
    print(f"  N = {N}: feasible {len(runs)} (doomed {sum(r['doomed'] for r in runs)}), lost feasibility {len(lost)} ({len(lostC)} from inside C_inf), "
          f"no convergence {len(noconv)}, worst cost / J* {max(ratios) if ratios else float('nan'):.4f}, starts > 1.2 J* {sum(v > 1.2 for v in ratios)}, "
          f"decrease residual > 0 at {sum(v > 1e-9 for v in resid_all)} of {len(resid_all)} steps, value increases at {sum(r['vals'][t + 1] > r['vals'][t] + 1e-9 for r in runs for t in range(len(r['vals']) - 1))}")
    if N == 2:
        r = next(r for r in runs if np.allclose(r["x0"], XS))
        print(f"    from x_S = (-3.5, 2.5): {len(r['us'])} steps, overshoot to p = {r['xs'][:, 0].max():.2f}, cost ratio {r['ratio']:.2f}")
r1 = closed_loop(mpc_plain(1), np.array([-1.0, 2.0]))
print(f"  plain N = 1 from (-1, 2): infeasible at step {r1['infeasible_at']}, at {r1['xs'][-1].round(2).tolist()}")

# --------------------------------------------------------------------------------------------------------------
section("Part 5: Learning MPC from the plain N = 2 trajectory")
JS = jinf(XS)
print(f"J_inf*(x_S) = {JS:.6f}")
seed = closed_loop(mpc_plain(2), XS)
Xs, Us = [x.copy() for x in seed["xs"]], list(seed["us"])
def complete(Xs, Us):
    x = Xs[-1]; M = np.c_[A @ B, B]; ub = -np.linalg.solve(M, A @ A @ x)
    for u in ub:
        x = A @ x + B[:, 0] * u; Us.append(float(u)); Xs.append(x.copy())
    Xs[-1] = np.zeros(2)
    return np.array(Xs), np.array(Us)[:, None], ub
X0, U0, bridge = complete(Xs, Us)
J0 = trajectory_cost_to_go(X0, U0, Q, R)[0]
print(f"first trajectory: {len(U0)} steps (bridge inputs {bridge.round(8).tolist()}), cost {J0:.4f} = {J0 / JS:.2f} J*")
results = {}
for N in (2, 3, 4):
    ctrl = LMPCController(A, B, Q, R, N, XLB, XUB, [UMIN], [UMAX]); ctrl.add_trajectory(X0, U0)
    for j in range(10):
        x = XS.copy(); Xj = [x.copy()]; Uj = []; warm = None
        for t in range(60):
            if np.linalg.norm(x) <= 1e-3: break
            sol = ctrl.solve(x, warm=warm)
            assert sol.success, f"LMPC infeasible N={N} iteration {j} step {t}: {sol.message}"
            u = float(sol.U[0, 0]); x = A @ x + B[:, 0] * u; Xj.append(x.copy()); Uj.append(u)
            warm = ctrl.shifted_warm_start(x, sol)
        Xc, Uc, _ = complete(Xj, Uj); ctrl.add_trajectory(Xc, Uc)
    costs = np.array(ctrl.iteration_costs); results[N] = ctrl
    print(f"  N = {N}: relative gaps (J^j - J*)/J* for j = 0, 1, 2, 5, 10: " + ", ".join(f"{(costs[j] - JS) / JS:.2e}" for j in (0, 1, 2, 5, 10))
          + f"; largest increase {max(np.diff(costs)):.1e}; stored states {sum(len(U) for U in ctrl.inputs)}")
# the N = 2 plateau is not a solver artefact: brute force over (u_0, u_1) with V_c by its LP
ctrl = results[2]; S, Vv, Usafe, _ = ctrl.safe_data()
def plan_cost(u0, u1):
    x1 = A @ XS + B[:, 0] * u0; x2 = A @ x1 + B[:, 0] * u1
    if np.any(np.abs(x1) > XUB + 1e-9) or np.any(np.abs(x2) > XUB + 1e-9): return np.inf
    v, _, ok = convex_safe_value(x2, S, Vv)
    return q(XS, u0) + q(x1, u1) + v if ok else np.inf
best = np.inf; grid = np.linspace(-1, 1, 41); lo, hi = (-1, 1), (-1, 1)
for refine in range(4):
    g0 = np.linspace(lo[0], lo[1], 41); g1 = np.linspace(hi[0], hi[1], 41)
    vals = np.array([[plan_cost(a, b) for b in g1] for a in g0]); i, j = np.unravel_index(np.argmin(vals), vals.shape); best = vals[i, j]
    d0, d1 = (lo[1] - lo[0]) / 40, (hi[1] - hi[0]) / 40
    lo = (max(-1, g0[i] - d0), min(1, g0[i] + d0)); hi = (max(-1, g1[j] - d1), min(1, g1[j] + d1))
print(f"  brute-force best N = 2 plan at x_S with the iteration-10 data: {best:.6f} against the stored cost {ctrl.iteration_costs[-1]:.6f}")

# --------------------------------------------------------------------------------------------------------------
section("Part 6: B.1 the condensed QP and B.2 the KKT conditions at x_S with N = 5")
m5 = mpc_terminal(5); G, w = constraints(m5, XS); Sx, Su = m5.Sx, m5.Su
Qbar = np.kron(np.eye(5), Q); Qbar[-2:, -2:] = P; H = 2 * (Su.T @ Qbar @ Su + 0.2 * np.eye(5)); f = 2 * Su.T @ Qbar @ (Sx @ XS); c = XS @ Q @ XS + XS @ Sx.T @ Qbar @ Sx @ XS
print(f"  {len(w)} constraint rows, smallest eigenvalue of H {np.linalg.eigvalsh(H).min():.3f}")
rng = np.random.default_rng(1); err = 0.0; agree = True
for _ in range(1000):
    U = rng.uniform(-1, 1, 5); Xp = (Sx @ XS + Su @ U).reshape(5, 2)
    J = q(XS, U[0]) + sum(q(Xp[k - 1], U[k]) for k in range(1, 5)) + Xp[-1] @ P @ Xp[-1]
    err = max(err, abs(0.5 * U @ H @ U + f @ U + c - J)); agree &= bool((G @ U <= w + 1e-9).all() == (np.all(np.abs(Xp) <= XUB + 1e-9) and XF.contains(Xp[-1], 1e-9)))
print(f"  on 1000 random input sequences: cost formula agrees to {err:.1e}, constraint rows agree with the rollout in every case: {agree}")
r = solve_qp(H, f, A_ub=G, b_ub=w)
act = np.where(w - G @ r.x <= 1e-7)[0]
print(f"  U* = {r.x.round(4).tolist()}, J_5*(x_S) = {0.5 * r.x @ H @ r.x + f @ r.x + c:.6f}, active rows {act.tolist()} with multipliers {r.lam[act].round(4).tolist()}")
print("  (row 26 is u_1 >= -1 in the ordering: 10 state upper rows, 10 state lower rows, 5 input upper, 5 input lower, terminal rows)")

section("B.6: the barycentric terminal cost on random convex combinations")
for N in (2, 4):
    S, Vv, Usafe, _ = results[N].safe_data(); rng = np.random.default_rng(7); resid = []; excess = []; rel = []
    for _ in range(40):
        idx = rng.choice(len(S), 3, replace=False); lam = rng.dirichlet(np.ones(3)); x = lam @ S[idx]
        v, lamv, ok = convex_safe_value(x, S, Vv)
        if not ok: continue
        u = float(lamv @ Usafe[:, 0]); xn = A @ x + B[:, 0] * u; vn, _, ok2 = convex_safe_value(xn, S, Vv)
        if not ok2: continue
        resid.append(vn - v + q(x, u)); Jx = jinf(x); excess.append(v - Jx)
        if Jx >= 0.1: rel.append((v - Jx) / Jx)
    print(f"  N = {N} data: decrease residual at most {max(resid):.1e}, V_c - J* at least {min(excess):.1e}, median over-estimate {100 * np.median(rel):.0f} percent on {len(rel)} points")
