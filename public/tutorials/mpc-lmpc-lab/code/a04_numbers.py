"""Reproduce every number quoted on chapter A4 of "Advanced MPC and Learning MPC",
"Solving a QP you can read" (a04-solving-a-qp-you-can-read.html).

Usage:  python a04_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
"""
from __future__ import annotations

import time

import numpy as np

from advanced_mpc_lmpc.linear import prediction_matrices
from advanced_mpc_lmpc.qp import feasibility_margin, solve_qp, solve_qp_slsqp, stack_inequalities

np.set_printoptions(precision=5, suppress=True, linewidth=120)


def section(title):
    print(f"\n=== {title}")


A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = 0.2
x0 = np.array([-3.0, 0.0])


def mpc_qp(N, x0, umax=1.0):
    """The condensed QP of chapter A2 for the canonical system: returns the arguments of solve_qp."""
    Sx, Su = prediction_matrices(A, B, N)
    Qbar = np.kron(np.eye(N), Q); Rbar = R * np.eye(N)
    H = 2 * (Su.T @ Qbar @ Su + Rbar); f = 2 * Su.T @ Qbar @ Sx @ x0
    A_ub = np.vstack([Su, -Su]); b_ub = np.r_[np.tile([4, 3.0], N) - Sx @ x0, np.tile([4, 3.0], N) + Sx @ x0]
    return dict(H=H, f=f, A_ub=A_ub, b_ub=b_ub, lb=-umax * np.ones(N), ub=umax * np.ones(N))


# --------------------------------------------------------------------------------------------------------------
section("Part 1: the stacked inequalities of the hand QP with bounds")
G, g = stack_inequalities(2, A_ub=[[1.0, 1.0]], b_ub=[1.0], lb=[0.0, 0.0], ub=[2.0, np.inf])
print("G =\n", G, "\ng =", g, " (the user's row, then the finite upper bound, then the finite lower bounds)")

# --------------------------------------------------------------------------------------------------------------
section("Hero and Part 3: the iterates on the hand QP  min ||z - c||^2  s.t.  z1 + z2 <= 1")
c = np.array([1.2, 0.8]); H2 = 2 * np.eye(2); f2 = -2 * c
trace = []
r = solve_qp(H2, f2, A_ub=[[1.0, 1.0]], b_ub=[1.0], trace=trace)
print("iteration   z1       z2        s        lam      residual     mu")
for k, (z, s, lam, resid, mu) in enumerate(trace):
    print(f"{k:5d}   {z[0]:8.5f} {z[1]:8.5f}  {s[0]:8.5f} {lam[0]:8.5f}   {resid:9.2e}  {mu:9.2e}")
print(f"converged in {r.nit} iterations: z* = {r.x}, lambda* = {r.lam}, KKT residual {r.kkt_residual:.1e}, success {r.success}")
print("the starting point: z = 0, s = max(g - G z, 1) =", max(1.0 - 0.0, 1.0), ", lam = 1")

# --------------------------------------------------------------------------------------------------------------
section("Part 6: the active-set polish on the hand QP")
# the last iterate before the polish is in the trace; the polish solves the KKT system on {rows with lam > s}
z_last, s_last, lam_last, _, _ = trace[-1]
print("last interior iterate z =", z_last, " s =", s_last, " lam =", lam_last, "-> active set:", np.where(lam_last > s_last)[0])
M = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [1.0, 1.0, 0.0]])
sol = np.linalg.solve(M, np.r_[2 * c, 1.0])
print("polished KKT solve [z; lambda] =", sol, " exactly (0.7, 0.3, 1)")

# --------------------------------------------------------------------------------------------------------------
section("Part 7: three problems, interior point against SLSQP")
problems = {
    "hand QP (2 variables, 1 row)": dict(H=H2, f=f2, A_ub=[[1.0, 1.0]], b_ub=[1.0]),
    "MPC QP, N = 8 from (-3, 0) (8 variables, 48 rows)": mpc_qp(8, x0),
    "MPC QP, N = 20 (20 variables, 120 rows)": mpc_qp(20, x0),
    "infeasible: z1 <= -1 and -z1 <= -1": dict(H=np.eye(2), f=np.zeros(2), A_ub=[[1.0, 0.0], [-1.0, 0.0]], b_ub=[-1.0, -1.0]),
}
for name, kw in problems.items():
    reps = 20
    t = time.perf_counter()
    for _ in range(reps): r = solve_qp(**kw)
    t_ipm = (time.perf_counter() - t) / reps
    t = time.perf_counter()
    for _ in range(reps): r2 = solve_qp_slsqp(**kw)
    t_sl = (time.perf_counter() - t) / reps
    print(f"\n{name}")
    print(f"  interior point: {r.nit:3d} iterations, {1e3 * t_ipm:6.2f} ms, success {r.success}, KKT residual {r.kkt_residual:.1e}, violation {r.max_violation:.1e}")
    print(f"  SLSQP:          {r2.nit:3d} iterations, {1e3 * t_sl:6.2f} ms, success {r2.success}, violation {r2.max_violation:.1e}, message: {r2.message}")
    if r.success and r2.success:
        print(f"  solutions agree to {np.abs(r.x - r2.x).max():.1e}; objective difference {abs(r.objective - r2.objective):.1e}")
    if "MPC" in name and r.success:
        act = np.where(r.lam > 1e-9)[0]
        n_rows = len(kw["b_ub"]); N = len(kw["f"])
        kinds = ["state row" if i < n_rows else "upper bound" if i < n_rows + N else "lower bound" for i in act]
        print(f"  active set: {len(act)} rows ({', '.join(f'{k}{i}' for k, i in zip(kinds, act))}); saturated inputs: {int(np.sum(np.abs(np.abs(r.x) - 1) < 1e-9))}")

# --------------------------------------------------------------------------------------------------------------
section("Part 8: feasibility certified by an independent LP")
kw = problems["infeasible: z1 <= -1 and -z1 <= -1"]
print("margin for the infeasible pair:", feasibility_margin(2, A_ub=kw["A_ub"], b_ub=kw["b_ub"]), " (positive: no z satisfies both rows)")
kw = mpc_qp(8, x0)
print("margin for the N = 8 MPC QP:", feasibility_margin(8, A_ub=kw["A_ub"], b_ub=kw["b_ub"], lb=kw["lb"], ub=kw["ub"]), " (zero: feasible)")
kw = mpc_qp(3, np.array([3.0, 2.0]))
r = solve_qp(**kw)
print("from (3, 2) with N = 3: solver success", r.success, ", margin", round(feasibility_margin(3, A_ub=kw["A_ub"], b_ub=kw["b_ub"], lb=kw["lb"], ub=kw["ub"]), 4),
      " (the next position 3 + 2 + u/2 >= 4.5 leaves the box whatever u)")

# --------------------------------------------------------------------------------------------------------------
section("Part 4: sigma and the number of iterations on the N = 8 MPC QP")
kw = mpc_qp(8, x0)
for sigma in (0.01, 0.1, 0.5):
    r = solve_qp(**kw, sigma=sigma)
    print(f"sigma = {sigma}: {r.nit} iterations, residual {r.kkt_residual:.1e}")
for tol in (1e-6, 1e-9, 1e-12):
    r = solve_qp(**kw, tol=tol)
    print(f"tol = {tol}: {r.nit} iterations")

# --------------------------------------------------------------------------------------------------------------
section("Exercise 2: a degenerate problem where the polish matters (weakly active bound)")
r = solve_qp(np.eye(3), np.array([1.0, 2.0, 3.0]), A_eq=[[1, 1, 1]], b_eq=[1], lb=[0, 0, 0], ub=[1, 1, 1])
print("z* =", r.x, " nu =", r.nu, " lam (rows: 3 upper, 3 lower) =", r.lam, " iterations", r.nit)
