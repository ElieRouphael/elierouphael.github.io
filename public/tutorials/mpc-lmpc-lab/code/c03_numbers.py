"""Reproduce every number quoted on chapter C3 of "Advanced MPC and Learning MPC",
"Nonlinear MPC by successive linearisation" (c03-nonlinear-mpc-by-successive-linearisation.html).

Usage:  python c03_numbers.py
Needs numpy, scipy (for the SLSQP comparison only) and the course package's QP solver (PYTHONPATH=src).

Everything in this file except `solve_qp` is new code written for this chapter and is not in the package:
the unicycle model and its Jacobians, the time-varying prediction matrices, the successive-linearisation
loop (a Gauss-Newton SQP with an optional elastic terminal constraint and trust region), and the
real-time-iteration closed loop.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from advanced_mpc_lmpc.qp import solve_qp

np.set_printoptions(precision=4, suppress=True, linewidth=120)

TS, N = 0.2, 30
ALPHA, BETA = 0.05, 0.10
UMAX, DUMAX = 1.0, 0.2
Z0 = np.zeros(3)
ZF = np.array([2.0, 1.0, 0.0])


def section(title):
    print(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
# The unicycle, its rollout and its Jacobians (forward Euler, one sample)

def step(z, u):
    """z = (X, Y, theta), u = (v, omega): one Euler step of the unicycle."""
    x, y, th = z
    v, w = u
    return np.array([x + TS * v * np.cos(th), y + TS * v * np.sin(th), th + TS * w])


def jacobians(z, u):
    """A_k = df/dz and B_k = df/du at the linearisation point (z, u)."""
    th = z[2]; v = u[0]
    A = np.array([[1.0, 0.0, -TS * v * np.sin(th)],
                  [0.0, 1.0, TS * v * np.cos(th)],
                  [0.0, 0.0, 1.0]])
    B = np.array([[TS * np.cos(th), 0.0],
                  [TS * np.sin(th), 0.0],
                  [0.0, TS]])
    return A, B


def rollout(z0, U):
    Z = [np.asarray(z0, float)]
    for u in U:
        Z.append(step(Z[-1], u))
    return np.asarray(Z)


def ltv_prediction(As, Bs):
    """Time-varying prediction matrices: with dz_{k+1} = A_k dz_k + B_k du_k,
    [dz_1; ...; dz_N] = Sx dz_0 + Su [du_0; ...; du_{N-1}].
    Block row i of Sx is A_{i-1} ... A_0; block (i, j) of Su is A_{i-1} ... A_{j+1} B_j, and B_i on the diagonal."""
    n, m = Bs[0].shape; N = len(As)
    Sx = np.zeros((N * n, n)); Su = np.zeros((N * n, N * m))
    Phi = np.eye(n)
    for i in range(N):
        Phi = As[i] @ Phi
        Sx[i * n:(i + 1) * n] = Phi
        P = np.eye(n)                                 # P = A_{i-1} ... A_{j+1}, built backwards in j
        for j in range(i, -1, -1):
            Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = P @ Bs[j]
            P = P @ As[j]
    return Sx, Su


# --------------------------------------------------------------------------------------------------------------
# The cost and the input constraints are exactly quadratic / linear in the stacked U

def cost_matrices(N):
    """J(U) = alpha ||U||^2 + beta ||D U||^2 = 0.5 U'HU with H = 2(alpha I + beta D'D); D differences consecutive inputs."""
    m = 2
    D = np.zeros(((N - 1) * m, N * m))
    for k in range(N - 1):
        D[k * m:(k + 1) * m, k * m:(k + 1) * m] = -np.eye(m)
        D[k * m:(k + 1) * m, (k + 1) * m:(k + 2) * m] = np.eye(m)
    H = 2.0 * (ALPHA * np.eye(N * m) + BETA * D.T @ D)
    return H, D


def cost(U):
    U = np.asarray(U).reshape(-1, 2)
    return ALPHA * np.sum(U ** 2) + BETA * np.sum(np.diff(U, axis=0) ** 2)


def rate_rows(N, D, u_prev=None):
    """G U <= g for the rate bounds |u_k - u_{k-1}| <= DUMAX; with u_prev the first move is bounded against it too."""
    G = np.vstack([D, -D]); g = np.full(2 * D.shape[0], DUMAX)
    if u_prev is not None:
        E0 = np.zeros((2, N * 2)); E0[:, :2] = np.eye(2)
        G = np.vstack([G, E0, -E0]); g = np.r_[g, DUMAX + u_prev, DUMAX - u_prev]
    return G, g


# --------------------------------------------------------------------------------------------------------------
# Successive linearisation: linearise the terminal pose along the current plan, solve a QP in the step dU, repeat

def sqp(z0, zf, U0, N, tol=1e-6, maxiter=50, trust=None, rho=None, u_prev=None, log=None):
    """Returns the plan U, the number of iterations and a history of (cost, terminal residual, largest |dU|, QP iterations).
    rho=None: the linearised terminal pose is a hard equality M dU = defect.
    rho=number: the elastic mode, M dU + s_plus - s_minus = defect with the l1 penalty rho 1'(s_plus + s_minus),
    an exact penalty that keeps every QP feasible. trust bounds every entry of dU."""
    m, n = 2, 3
    H, D = cost_matrices(N); G, g = rate_rows(N, D, u_prev)
    U = np.asarray(U0, float).reshape(N, m).copy()
    hist = []
    for it in range(1, maxiter + 1):
        Z = rollout(z0, U)
        As, Bs = zip(*[jacobians(Z[k], U[k]) for k in range(N)])
        Sx, Su = ltv_prediction(list(As), list(Bs))
        M = Su[-n:]                                   # d z_N / d U along the plan
        defect = zf - Z[-1]                           # how far the plan's end is from the pose
        f = H @ U.reshape(-1)                         # J(U + dU) = 0.5 dU'H dU + (H U)' dU + J(U)
        lb = -UMAX - U.reshape(-1); ub = UMAX - U.reshape(-1)
        if trust is not None:
            lb = np.maximum(lb, -trust); ub = np.minimum(ub, trust)
        gd = g - G @ U.reshape(-1)
        if rho is None:
            res = solve_qp(H, f, A_ub=G, b_ub=gd, A_eq=M, b_eq=defect, lb=lb, ub=ub)
            dU = res.x
        else:
            nz = N * m
            Hs = np.zeros((nz + 2 * n, nz + 2 * n)); Hs[:nz, :nz] = H
            fs = np.r_[f, rho * np.ones(2 * n)]
            res = solve_qp(Hs, fs, A_ub=np.c_[G, np.zeros((len(gd), 2 * n))], b_ub=gd,
                           A_eq=np.c_[M, np.eye(n), -np.eye(n)], b_eq=defect,
                           lb=np.r_[lb, np.zeros(2 * n)], ub=np.r_[ub, np.full(2 * n, np.inf)])
            dU = res.x[:nz]
        if not res.success:
            hist.append((cost(U), np.abs(defect).max(), np.nan, res.message)); break
        U = U + dU.reshape(N, m)
        resid = np.abs(zf - rollout(z0, U)[-1]).max()
        hist.append((cost(U), resid, np.abs(dU).max(), res.nit))
        if log is not None:
            log.append(U.copy())
        if resid < tol and np.abs(dU).max() < tol:
            break
    return U, it, hist


U0 = np.c_[np.r_[0.8 * np.ones(10), 0.6 * np.ones(10), 0.4 * np.ones(10)],
           np.r_[0.3 * np.ones(10), -0.3 * np.ones(10), np.zeros(10)]]      # the guess of chapter A5

section("Part 1: the problem, the guess, and why it is nonlinear")
Zg = rollout(Z0, U0)
print(f"the guess (drive, turn, unwind) costs {cost(U0):.4f} and ends at {Zg[-1]}, {np.abs(ZF - Zg[-1]).max():.3f} from the pose {ZF}")
A10, B10 = jacobians(Zg[10], U0[10])
print(f"Jacobians at step 10 of the guess (theta = {Zg[10][2]:.3f}, v = {U0[10][0]}):\nA_10 =\n{A10}\nB_10 =\n{B10}")

section("Part 3: the time-varying prediction matrices against a perturbed rollout")
As, Bs = zip(*[jacobians(Zg[k], U0[k]) for k in range(N)])
Sx, Su = ltv_prediction(list(As), list(Bs))
dU = 1e-4 * np.random.default_rng(0).standard_normal((N, 2))
pred = (Su @ dU.reshape(-1)).reshape(N, 3)
actual = rollout(Z0, U0 + dU)[1:] - Zg[1:]
print(f"Sx is {Sx.shape}, Su is {Su.shape}; for an input perturbation of size 1e-4 the linear prediction of the state"
      f" change is off by {np.abs(pred - actual).max():.1e}, second order in the perturbation")
print(f"the last block row of Su, dz_N/dU, is {Su[-3:].shape}; its rank is {np.linalg.matrix_rank(Su[-3:])}")

section("Part 4: successive linearisation from the guess (hard linearised terminal equality)")
t0 = time.perf_counter()
log = []
U_sqp, its, hist = sqp(Z0, ZF, U0, N, log=log)
t_sqp = time.perf_counter() - t0
print("iteration   cost       terminal residual   largest |dU|   QP iterations")
for i, (c, r, d, nit) in enumerate(hist, 1):
    print(f"{i:9d}   {c:.6f}   {r:.2e}            {d:.2e}       {nit}")
print(f"converged in {its} iterations ({t_sqp * 1000:.0f} ms): cost {cost(U_sqp):.6f}, terminal residual {hist[-1][1]:.1e}")
first_below = next((i for i, h in enumerate(hist, 1) if h[1] < 1e-6), None)
ratios = [hist[i][1] / hist[i - 1][1] for i in range(2, 8)]
print(f"terminal residual below 1e-6 from iteration {first_below}; residual ratios between consecutive iterations "
      f"{np.round(ratios, 3).tolist()}: a roughly constant factor, linear convergence")
Zs = rollout(Z0, U_sqp)
print(f"path: ends at {Zs[-1]}; largest |v| = {np.abs(U_sqp[:, 0]).max():.4f}, largest |omega| = {np.abs(U_sqp[:, 1]).max():.4f},"
      f" largest rate = {np.abs(np.diff(U_sqp, axis=0)).max():.4f} (bounds 1, 1, 0.2): no constraint is active")
print(f"first inputs {U_sqp[:3].round(4).tolist()}, inputs at the turn {U_sqp[12:15].round(4).tolist()}")

section("Part 4: other guesses, hard equality")
guesses = {"straight ahead, v = 0.5": np.c_[0.5 * np.ones(N), np.zeros(N)], "all zeros": np.zeros((N, 2)),
           "random in [-0.5, 0.5]": np.random.default_rng(1).uniform(-0.5, 0.5, (N, 2))}
for name, Ug in guesses.items():
    Ux, itx, hx = sqp(Z0, ZF, Ug, N)
    ok = hx[-1][1] < 1e-6
    print(f"{name:24s}: {'converged' if ok else 'QP INFEASIBLE'} after {itx:2d} iterations, cost {cost(Ux):.6f}, residual {hx[-1][1]:.1e},"
          f" max |U - U_sqp| = {np.abs(Ux - U_sqp).max():.1e}" + ("" if ok else f"; rank of dz_N/dU at the guess: "
          f"{np.linalg.matrix_rank(ltv_prediction(*map(list, zip(*[jacobians(z, u) for z, u in zip(rollout(Z0, Ug)[:-1], Ug)])))[1][-3:])}"))

section("Part 5: the elastic mode (l1 penalty rho on the terminal defect) from every guess, and with a trust region")
for rho in (20.0,):
    for name, Ug in {"the guess": U0, **guesses}.items():
        Ux, itx, hx = sqp(Z0, ZF, Ug, N, rho=rho)
        print(f"rho = {rho}, {name:24s}: {itx:2d} iterations, cost {cost(Ux):.6f}, residual {hx[-1][1]:.1e}, max |U - U_sqp| = {np.abs(Ux - U_sqp).max():.1e}")
print(f"the guess itself violates the rate bound: its largest rate is {np.abs(np.diff(U0, axis=0)).max():.1f} against 0.2, repaired by the first step")
for trust in (0.1, 0.25):
    Ut, itt, ht = sqp(Z0, ZF, U0, N, rho=20.0, trust=trust)
    ok = ht[-1][1] < 1e-6
    print(f"rho = 20 with the trust region |dU| <= {trust}, from the guess: " + (f"{itt} iterations, cost {cost(Ut):.6f}, residual {ht[-1][1]:.1e}, max |U - U_sqp| = {np.abs(Ut - U_sqp).max():.1e}" if ok else
          f"QP infeasible at iteration {itt}: the rate violation of the guess cannot be repaired within the radius"))
for rho in (0.1, 1.0):
    Ux, itx, hx = sqp(Z0, ZF, U0, N, rho=rho)
    print(f"rho = {rho}: {itx} iterations, cost {cost(Ux):.6f}, residual {hx[-1][1]:.1e}"
          + (" (below the largest terminal multiplier: the penalty is not exact and the pose is missed)" if hx[-1][1] > 1e-6 else ""))
ZB = np.array([0.0, 1.0, np.pi])
Ub, itb, hb = sqp(Z0, ZB, U0, N)
Ube, itbe, hbe = sqp(Z0, ZB, U0, N, rho=20.0)
print(f"target (0, 1, pi) behind the start: hard equality {itb} iterations, cost {cost(Ub):.4f}, residual {hb[-1][1]:.1e};"
      f" elastic {itbe} iterations, cost {cost(Ube):.4f}, residual {hbe[-1][1]:.1e} (chapter A5's SLSQP: 0.4941)")

section("Part 6: the terminal multipliers, and SciPy's SLSQP on the same problem")
H, D = cost_matrices(N); G, g = rate_rows(N, D)
Zs = rollout(Z0, U_sqp); As, Bs = zip(*[jacobians(Zs[k], U_sqp[k]) for k in range(N)]); M = ltv_prediction(list(As), list(Bs))[1][-3:]
res = solve_qp(H, H @ U_sqp.reshape(-1), A_ub=G, b_ub=g - G @ U_sqp.reshape(-1), A_eq=M, b_eq=ZF - Zs[-1],
               lb=-UMAX - U_sqp.reshape(-1), ub=UMAX - U_sqp.reshape(-1))
print(f"at the solution the step is {np.abs(res.x).max():.1e} and the multipliers of the three terminal rows are {res.nu}:"
      f" the price of moving the pose by one unit in x, y and theta; the largest is {np.abs(res.nu).max():.3f}, so any rho above it is exact")
rate = NonlinearConstraint(lambda U: np.diff(U.reshape(N, 2), axis=0).ravel(), -DUMAX, DUMAX)
tc = NonlinearConstraint(lambda U: rollout(Z0, U.reshape(N, 2))[-1] - ZF, np.zeros(3), np.zeros(3))
t0 = time.perf_counter()
sl = minimize(lambda U: cost(U), U0.reshape(-1), method="SLSQP", bounds=[(-UMAX, UMAX)] * (2 * N), constraints=[tc, rate],
              options={"ftol": 1e-10, "maxiter": 2000})
t_sl = time.perf_counter() - t0
print(f"SLSQP from the guess: success {sl.success}, {sl.nit} iterations ({t_sl * 1000:.0f} ms), cost {sl.fun:.6f},"
      f" residual {np.abs(rollout(Z0, sl.x.reshape(N, 2))[-1] - ZF).max():.1e}, max |U_slsqp - U_sqp| = {np.abs(sl.x.reshape(N, 2) - U_sqp).max():.1e}")

section("Part 7: the real-time iteration in closed loop: one QP per sample, a push, late pushes and noise")


def closed_loop(push, push_at=10, rho=20.0, noise=0.0, seed=0, iters=1):
    """Shrinking horizon: at sample t the plan has N - t moves; `iters` QPs per sample (1 = the real-time iteration),
    warm-started by the shifted plan, the first move rate-bounded against the input just applied, the terminal
    pose elastic so that the QP stays feasible when the push makes the pose unreachable in the time left."""
    rng = np.random.default_rng(seed)
    z = Z0.copy(); Z = [z.copy()]; applied = []; plan = U_sqp.copy(); u_prev = None; n_qp = 0
    for t in range(N):
        horizon = N - t
        Up, itx, _ = sqp(z, ZF, plan[:horizon], horizon, maxiter=iters, rho=rho, u_prev=u_prev)
        n_qp += itx
        u = Up[0]; applied.append(u.copy()); u_prev = u
        z = step(z, u)
        if t == push_at:
            z = z + push
        z = z + noise * rng.standard_normal(3)
        Z.append(z.copy())
        plan = np.vstack([Up[1:], Up[-1:]]) if horizon > 1 else Up
    Z = np.asarray(Z); Ua = np.asarray(applied)
    ok = np.abs(np.diff(Ua, axis=0)).max() <= DUMAX + 1e-9 and np.abs(Ua).max() <= UMAX + 1e-9
    return Z, Ua, np.abs(Z[-1] - ZF), n_qp, ok


for push, at in ((np.zeros(3), 10), (np.array([0.0, 0.15, 0.10]), 10), (np.array([0.0, 0.40, 0.30]), 10),
                 (np.array([0.0, 0.15, 0.10]), 25), (np.array([0.0, 0.15, 0.10]), 28)):
    Z, Ua, err, n_qp, ok = closed_loop(push, at)
    print(f"push {push} at sample {at:2d}: {n_qp} QPs, final pose {Z[-1]}, pose error {err}, closed-loop cost {cost(Ua):.4f},"
          f" input and rate bounds respected: {ok}")
Z, Ua, err, _, ok = closed_loop(np.zeros(3), noise=0.01, seed=3)
print(f"noise 0.01 on every state, no push: pose error {err}, cost {cost(Ua):.4f}, bounds respected: {ok}")
for rho in (2.0, 0.5):
    Z, Ua, err, _, ok = closed_loop(np.array([0.0, 0.15, 0.10]), 25, rho=rho)
    print(f"push at 25 with rho = {rho}: pose error {err}, cost {cost(Ua):.4f} (a cheaper loop that misses by more)")
Z, Ua, err, n_qp, ok = closed_loop(np.array([0.0, 0.15, 0.10]), 10, iters=30)
print(f"re-converging at every sample instead (push at 10): {n_qp} QPs in all, pose error {err}, cost {cost(Ua):.4f}")
