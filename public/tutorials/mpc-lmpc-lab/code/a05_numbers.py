"""Reproduce every number quoted on chapter A5 of "Advanced MPC and Learning MPC",
"The finite-horizon problem" (a05-the-finite-horizon-problem.html).

Usage:  python a05_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
The unicycle section uses SciPy's SLSQP (a nonlinear program); everything else is the package's QP solver.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from advanced_mpc_lmpc.linear import prediction_matrices
from advanced_mpc_lmpc.qp import feasibility_margin, solve_qp

np.set_printoptions(precision=4, suppress=True, linewidth=120)

A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = 0.2; P = Q
PMAX, VMAX = 4.0, 3.0


def section(title):
    print(f"\n=== {title}")


def difference_matrix(N):
    D = np.eye(N); D[np.arange(1, N), np.arange(N - 1)] = -1.0
    return D


def build(N, x0, umax=1.0, dumax=0.5, u_prev=0.0, terminal=True, soft=None, vmax=VMAX):
    """The condensed CFTOC: cost, box on every predicted state, input bound, rate bound, optional x_N = 0.
    soft = rho adds slacks on the velocity bound with an l1 penalty rho (exact penalty)."""
    x0 = np.asarray(x0, float)
    Sx, Su = prediction_matrices(A, B, N)
    Qbar = np.kron(np.eye(N), Q); Qbar[-2:, -2:] = P; Rbar = R * np.eye(N)
    H = 2 * (Su.T @ Qbar @ Su + Rbar); f = 2 * Su.T @ Qbar @ Sx @ x0
    const = x0 @ (Sx.T @ Qbar @ Sx + Q) @ x0
    rows, rhs, tags = [], [], []
    Sxx = Sx @ x0
    for k in range(N):
        rp, rv = Su[2 * k], Su[2 * k + 1]
        rows += [rp, -rp, rv, -rv]; rhs += [PMAX - Sxx[2 * k], PMAX + Sxx[2 * k], vmax - Sxx[2 * k + 1], vmax + Sxx[2 * k + 1]]
        tags += [("position", k + 1)] * 2 + [("velocity", k + 1)] * 2
    D = difference_matrix(N)
    for k in range(N):
        off = -u_prev if k == 0 else 0.0
        rows += [D[k], -D[k]]; rhs += [dumax - off, dumax + off]; tags += [("rate", k)] * 2
    for k in range(N):
        ek = np.zeros(N); ek[k] = 1
        rows += [ek, -ek]; rhs += [umax, umax]; tags += [("input", k)] * 2
    G = np.array(rows); g = np.array(rhs)
    E = Su[-2:] if terminal else None; e = -Sxx[-2:] if terminal else None
    if soft is not None:   # slacks eps_k >= 0 on the 2N velocity rows: G_v U - eps <= g_v, cost + rho * sum eps
        pos = [i for i, t in enumerate(tags) if t[0] == "velocity"]
        nz = N + len(pos)
        H2 = np.zeros((nz, nz)); H2[:N, :N] = H
        f2 = np.r_[f, soft * np.ones(len(pos))]
        G2 = np.zeros((len(g), nz)); G2[:, :N] = G
        for j, i in enumerate(pos): G2[i, N + j] = -1.0
        E2 = None if E is None else np.c_[E, np.zeros((2, len(pos)))]
        lb = np.r_[-np.inf * np.ones(N), np.zeros(len(pos))]
        return dict(H=H2, f=f2, A_ub=G2, b_ub=g, A_eq=E2, b_eq=e, lb=lb), dict(Sx=Sx, Su=Su, const=const, tags=tags, N=N, x0=x0, npos=len(pos))
    return dict(H=H, f=f, A_ub=G, b_ub=g, A_eq=E, b_eq=e), dict(Sx=Sx, Su=Su, const=const, tags=tags, N=N, x0=x0)


def report(kw, info, label):
    r = solve_qp(**kw)
    N = info["N"]; U = r.x[:N]
    X = np.vstack([info["x0"], (info["Sx"] @ info["x0"] + info["Su"] @ U).reshape(N, 2)])
    J = r.objective + info["const"] if "npos" not in info else 0.5 * U @ kw["H"][:N, :N] @ U + kw["f"][:N] @ U + info["const"]
    print(f"{label}: success {r.success}, violation {r.max_violation:.1e}, {r.nit} iterations")
    if not r.success:
        return r, U, X, J
    print(f"  physical cost J = {J:.4f} (QP objective {r.objective:.4f}, dropped constant {info['const']:.4f})")
    print(f"  inputs {np.round(U, 4)}")
    print(f"  states {[tuple(np.round(x, 3).tolist()) for x in X[1:]]}")
    print(f"  largest |u| = {np.abs(U).max():.4f}, largest |du| = {np.abs(np.diff(np.r_[0.0, U])).max():.4f}, peak |v| = {np.abs(X[:, 1]).max():.4f}")
    act = [(info["tags"][i], r.lam[i]) for i in np.where(r.lam > 1e-8)[0]]
    kinds = {}
    for (kind, _), _ in act: kinds[kind] = kinds.get(kind, 0) + 1
    print(f"  active inequalities: {len(act)} of {len(kw['b_ub'])} -> {kinds}")
    for (kind, step), lam in sorted(act, key=lambda t: -t[1])[:4]:
        print(f"    {kind} at step {step}: multiplier {lam:.4f}")
    if kw.get("A_eq") is not None:
        print(f"  terminal state {np.round(X[-1], 6)}, multipliers nu = {np.round(r.nu, 4)}")
    else:
        print(f"  terminal state {np.round(X[-1], 4)} (no terminal equality)")
    return r, U, X, J


# --------------------------------------------------------------------------------------------------------------
section("Hero and Part 4: the complete problem, N = 8 from (-3, 0), |u| <= 1, |du| <= 0.5, x_N = 0")
kw, info = build(8, [-3.0, 0.0])
r, U, X, J = report(kw, info, "with x_N = 0")
U_hero = U.copy()
ev = np.linalg.eigvalsh(kw["H"]); print(f"  smallest eigenvalue of H {ev[0]:.4f}: a convex QP")

section("Part 4: the same without the terminal equality")
kw2, info2 = build(8, [-3.0, 0.0], terminal=False)
r2, U2, X2, J2 = report(kw2, info2, "without x_N = 0")
print(f"  cost difference {J - J2:.4f}: the price of landing exactly")

section("Part 4: shadow price check on the most expensive rate bound")
# relax the first rate bound by delta and compare the cost change with lambda * delta
kw3, info3 = build(8, [-3.0, 0.0], dumax=0.5)
idx = [i for i, t in enumerate(info3["tags"]) if t == ("rate", 0)][0]
lam0 = solve_qp(**kw3).lam[idx]
kw3["b_ub"] = kw3["b_ub"].copy(); kw3["b_ub"][idx] += 1e-3
J3 = solve_qp(**kw3).objective + info3["const"]
print(f"  multiplier on the first rate bound {lam0:.4f}; relaxing it by 0.001 lowers J by {J - J3:.6f} = {(J - J3) / 1e-3:.4f} per unit")

# --------------------------------------------------------------------------------------------------------------
section("Part 5: an infeasible problem and its certificate")
kw4, info4 = build(4, [-3.0, 0.0])
r4 = solve_qp(**kw4)
m = feasibility_margin(4, A_ub=kw4["A_ub"], b_ub=kw4["b_ub"], A_eq=kw4["A_eq"], b_eq=kw4["b_eq"])
print(f"N = 4 with x_N = 0: solver success {r4.success}; feasibility margin {m:.4f} (positive: no sequence reaches the origin in 4 steps under the rate bound)")
for N in (5, 6, 7):
    kw5, info5 = build(N, [-3.0, 0.0])
    m = feasibility_margin(N, A_ub=kw5["A_ub"], b_ub=kw5["b_ub"], A_eq=kw5["A_eq"], b_eq=kw5["b_eq"])
    print(f"N = {N}: margin {m:.4f}")

section("Part 5: a softened comfort bound |v| <= 1 on the hero problem, and the exact penalty")
kwh, infoh = build(8, [-3.0, 0.0], vmax=1.0)
rh = solve_qp(**kwh)
Uh = rh.x; Xh = (infoh["Sx"] @ np.array([-3.0, 0.0]) + infoh["Su"] @ Uh).reshape(8, 2)
act = [(infoh["tags"][i], float(rh.lam[i])) for i in np.where(rh.lam > 1e-8)[0]]
vel = [(t, round(l, 4)) for t, l in act if t[0] == "velocity"]
lam_max = max(l for (k, _), l in act if k == "velocity")
Jh = rh.objective + infoh["const"]
print(f"hard |v| <= 1: success {rh.success}, cost {Jh:.4f} (against {J:.4f} with |v| <= 3), peak |v| {np.abs(Xh[:, 1]).max():.4f}")
print("  active velocity rows and multipliers:", vel, " largest", round(lam_max, 4))
for rho in (0.25 * lam_max, 0.5 * lam_max, lam_max, 2 * lam_max):
    kws, infos = build(8, [-3.0, 0.0], vmax=1.0, soft=rho)
    rs = solve_qp(**kws)
    N = 8; U = rs.x[:N]; eps = rs.x[N:]
    Xs = (infos["Sx"] @ infos["x0"] + infos["Su"] @ U).reshape(N, 2)
    Jq = 0.5 * U @ kws["H"][:N, :N] @ U + kws["f"][:N] @ U + infos["const"]
    print(f"rho = {rho:.4f}: largest slack {eps.max():.4f}, peak |v| {np.abs(Xs[:, 1]).max():.4f}, quadratic cost {Jq:.4f}, inputs differ from the hard ones by {np.abs(U - Uh).max():.1e}")
kwi, _ = build(6, [3.6, 1.5], terminal=False)
print("from (3.6, 1.5) the hard problem is infeasible (margin", round(feasibility_margin(6, A_ub=kwi["A_ub"], b_ub=kwi["b_ub"]), 4),
      "): softening the velocity bound does not help, the position bound is the one that cannot hold; a slack on the position rows would be needed")

# --------------------------------------------------------------------------------------------------------------
section("Part 6: the nonlinear CFTOC, a unicycle to a terminal pose (SciPy SLSQP)")
Ts, Nu = 0.2, 30
alpha, beta = 0.05, 0.10


def rollout_unicycle(Uflat):
    U = Uflat.reshape(Nu, 2); z = np.zeros(3); Z = [z.copy()]
    for v, w in U:
        z = z + Ts * np.array([v * np.cos(z[2]), v * np.sin(z[2]), w]); Z.append(z.copy())
    return np.asarray(Z)


def unicycle_cost(Uflat):
    U = Uflat.reshape(Nu, 2)
    return alpha * np.sum(U ** 2) + beta * np.sum(np.diff(U, axis=0) ** 2)


rate = NonlinearConstraint(lambda U: np.diff(U.reshape(Nu, 2), axis=0).ravel(), -0.2, 0.2)
bounds = [(-1, 1)] * (2 * Nu)
U0 = np.c_[np.r_[0.8 * np.ones(10), 0.6 * np.ones(10), 0.4 * np.ones(10)], np.r_[0.3 * np.ones(10), -0.3 * np.ones(10), np.zeros(10)]]


def solve_unicycle(zf, Uinit):
    tc = NonlinearConstraint(lambda U: rollout_unicycle(U)[-1] - zf, np.zeros(3), np.zeros(3))
    return minimize(unicycle_cost, Uinit.reshape(-1), method="SLSQP", bounds=bounds, constraints=[tc, rate],
                    options={"ftol": 1e-10, "maxiter": 2000})


guesses = {"notebook": U0, "straight": np.c_[0.5 * np.ones(Nu), np.zeros(Nu)],
           "reverse": np.c_[-0.4 * np.ones(Nu), np.r_[0.5 * np.ones(10), np.zeros(10), -0.5 * np.ones(10)]],
           "random": np.random.default_rng(1).uniform(-0.5, 0.5, (Nu, 2))}
for zf in (np.array([2.0, 1.0, 0.0]), np.array([0.0, 1.0, np.pi])):
    sols = {k: solve_unicycle(zf, g) for k, g in guesses.items()}
    ref = sols["notebook"]
    print(f"target {zf}: success {[s.success for s in sols.values()]}, costs {[round(float(s.fun), 4) for s in sols.values()]},"
          f" terminal residual {np.abs(rollout_unicycle(ref.x)[-1] - zf).max():.1e},"
          f" max input difference between starts {max(np.abs(s.x - ref.x).max() for s in sols.values()):.1e}")

# --------------------------------------------------------------------------------------------------------------
section("Exercise 5: the rate matrix D for N = 4 and the previous input")
D4 = difference_matrix(4); U4 = np.array([0.3, 0.5, 0.4, 0.1]); u_prev = 0.2
print("D =\n", D4, "\nD U + (-u_prev, 0, 0, 0) =", D4 @ U4 + np.r_[-u_prev, 0, 0, 0], " np.diff:", np.diff(np.r_[u_prev, U4]))

section("Exercise 6: the sparse form of the hero problem agrees")
N = 8; x0 = np.array([-3.0, 0.0]); Sx, Su = prediction_matrices(A, B, N)
nz = 3 * N
Hs = np.zeros((nz, nz)); Hs[:2 * N, :2 * N] = 2 * np.kron(np.eye(N), Q); Hs[2 * N:, 2 * N:] = 2 * R * np.eye(N)
E = np.zeros((2 * N + 2, nz)); e = np.zeros(2 * N + 2)
for k in range(N):
    E[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.eye(2)
    if k: E[2 * k:2 * k + 2, 2 * (k - 1):2 * k] = -A
    else: e[0:2] = A @ x0
    E[2 * k:2 * k + 2, 2 * N + k] = -B.ravel()
E[2 * N:, 2 * (N - 1):2 * N] = np.eye(2)       # x_N = 0
D = difference_matrix(N); Gs = np.zeros((2 * N, nz)); Gs[:N, 2 * N:] = D; Gs[N:, 2 * N:] = -D
gs = 0.5 * np.ones(2 * N)
lb = np.r_[np.tile([-4, -3.0], N), -np.ones(N)]; ub = np.r_[np.tile([4, 3.0], N), np.ones(N)]
rs = solve_qp(Hs, np.zeros(nz), A_ub=Gs, b_ub=gs, A_eq=E, b_eq=e, lb=lb, ub=ub)
print("sparse inputs agree with the condensed ones to", np.abs(rs.x[2 * N:] - U_hero).max(), "; variables", nz, "against", N)
