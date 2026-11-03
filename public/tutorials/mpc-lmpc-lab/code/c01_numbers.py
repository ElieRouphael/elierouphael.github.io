"""Reproduce every number quoted on chapter C1 of "Advanced MPC and Learning MPC",
"Tube MPC for a bounded disturbance" (c01-tube-mpc-for-a-bounded-disturbance.html).

Usage:  python c01_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
Runs in a few seconds.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from advanced_mpc_lmpc.linear import dlqr
from advanced_mpc_lmpc.mpc import LinearMPC
from advanced_mpc_lmpc.polytopes import (HPolytope, affine_image_2d, closed_loop_admissible_set_scalar_input,
                                         maximal_control_invariant, maximal_invariant, minkowski_sum_2d,
                                         n_step_controllable_sets)

np.set_printoptions(precision=4, suppress=True)


def section(title):
    print(f"\n=== {title}")


def area(P):
    V = P.vertices_2d()
    return ConvexHull(V).volume if len(V) >= 3 else 0.0


# the canonical system
A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])
X = HPolytope.box([-4.0, -3.0], [4.0, 3.0]); UMAX = 1.0
K, P, eig = dlqr(A, B, Q, R); AK = A - B @ K
N = 5

# --------------------------------------------------------------------------------------------------------------
section("Part 1: the disturbed plant, and the error dynamics under u = v - K (x - z)")
print("K =", K.ravel(), " rho(A - BK) =", abs(eig).max().round(4))
print("e+ = (A - BK) e + w: the error between plant and nominal state is driven by w alone")


def rpi_sets(wmax, s_max=12):
    """F_s = W + A_K W + ... + A_K^{s-1} W, the partial sums of the minimal robust positively invariant set."""
    W = HPolytope.box([-wmax, -wmax], [wmax, wmax])
    F, out = W, [W]
    for s in range(2, s_max + 1):
        F = minkowski_sum_2d(F, affine_image_2d(W, np.linalg.matrix_power(AK, s - 1)))
        out.append(F)
    return W, out


def support(P, rows):
    """Support function h_P(H_i) = max_{x in P} H_i x for every row of `rows`."""
    V = P.vertices_2d()
    return np.array([max(float(r @ v) for v in V) for r in rows])


def scaled(P, factor):
    return HPolytope.from_vertices_2d(P.vertices_2d() * factor)


# --------------------------------------------------------------------------------------------------------------
section("Part 2: the minimal robust positively invariant set as a limit of Minkowski sums, |w_i| <= 0.1")
WMAX = 0.1
W, Fs = rpi_sets(WMAX)
for s, F in enumerate(Fs, 1):
    print(f"F_{s:2d}: {len(F.vertices_2d()):2d} vertices, area {area(F):.4f}")
S_USE = 8
Fs8 = Fs[S_USE - 1]
AKs = np.linalg.matrix_power(AK, S_USE)
alpha = max(np.abs(AKs @ v).max() / WMAX for v in W.vertices_2d())
print(f"alpha with s = {S_USE}: A_K^{S_USE} W is inside alpha W with alpha = {alpha:.2e} ({alpha:.6f})")
E = scaled(Fs8, 1.0 / (1.0 - alpha))
print(f"E = F_{S_USE} / (1 - alpha): {len(E.vertices_2d())} vertices, area {area(E):.4f}, "
      f"extent |e_1| <= {support(E, np.array([[1.0, 0.0]]))[0]:.4f}, |e_2| <= {support(E, np.array([[0.0, 1.0]]))[0]:.4f}")
# robust positive invariance check: A_K E + W inside E
AKE_W = minkowski_sum_2d(affine_image_2d(E, AK), W)
print("A_K E + W inside E:", all(E.contains(v, 1e-9) for v in AKE_W.vertices_2d()),
      f"(support gap at most {np.max(support(AKE_W, E.H) - E.h):.2e})")

# --------------------------------------------------------------------------------------------------------------
section("Part 3: the tightened constraints X - E and U - K E")
hx_tight = X.h - support(E, X.H)
ku = support(E, K)[0]
umax_tight = UMAX - ku
print("box rows (p<=4, v<=3, -p<=4, -v<=3) tightened to", hx_tight.round(4))
print(f"max |K e| over E = {ku:.4f}, so the nominal input bound is |v| <= {umax_tight:.4f}")
Xt = HPolytope(X.H, hx_tight)

# --------------------------------------------------------------------------------------------------------------
section("Part 4: the nominal MPC on the tightened sets, N = 5")
XKt = closed_loop_admissible_set_scalar_input(Xt, K, -umax_tight, umax_tight)
Xft, hist = maximal_invariant(AK, XKt)
Ct, histC = maximal_control_invariant(A, B, Xt, -umax_tight, umax_tight)
regt = n_step_controllable_sets(A, B, Xt, Xft, -umax_tight, umax_tight, N)[-1]
XK = closed_loop_admissible_set_scalar_input(X, K, -UMAX, UMAX)
Xf, _ = maximal_invariant(AK, XK)
C, _ = maximal_control_invariant(A, B, X, -UMAX, UMAX)
reg = n_step_controllable_sets(A, B, X, Xf, -UMAX, UMAX, N)[-1]
print(f"tightened: X_f {len(Xft.vertices_2d())} vertices, area {area(Xft):.3f} ({len(hist) - 1} iterations); "
      f"C_inf area {area(Ct):.3f} ({len(histC) - 1} iterations); N = 5 region area {area(regt):.3f}")
print(f"nominal (chapter A11): X_f area {area(Xf):.3f}; C_inf area {area(C):.3f}; N = 5 region area {area(reg):.3f}")
print("the N = 5 tube region plus E (area of the Minkowski sum):", round(area(minkowski_sum_2d(regt, E)), 3))

mpc_t = LinearMPC(A, B, Q, R, N, P=P, x_lower=[-hx_tight[2], -hx_tight[3]], x_upper=[hx_tight[0], hx_tight[1]],
                  u_lower=[-umax_tight], u_upper=[umax_tight], terminal_H=Xft.H, terminal_h=Xft.h)
mpc_n = LinearMPC(A, B, Q, R, N, P=P, x_lower=[-4, -3], x_upper=[4, 3], u_lower=[-UMAX], u_upper=[UMAX],
                  terminal_H=Xf.H, terminal_h=Xf.h)


# --------------------------------------------------------------------------------------------------------------
# The tube loop (quoted on the page).
def tube_loop(mpc, x0, wmax, steps=40, seed=0, K=K, E=E):
    """Tube MPC: plan for the nominal state z, apply u = v - K (x - z), and let the plant be disturbed."""
    rng = np.random.default_rng(seed)
    x = np.array(x0, float); z = x.copy()
    xs, zs, us, ws = [x.copy()], [z.copy()], [], []
    for t in range(steps):
        sol = mpc.solve(z)                                   # the nominal problem, from the nominal state
        if not sol.success:
            return dict(xs=np.array(xs), zs=np.array(zs), us=np.array(us), infeasible_at=t)
        v = float(sol.U[0].item())
        u = v - float((K @ (x - z)).item())                  # the tube feedback keeps x near z
        w = rng.uniform(-wmax, wmax, 2)
        x = A @ x + B.ravel() * u + w                        # the plant, disturbed
        z = A @ z + B.ravel() * v                            # the nominal model, undisturbed
        xs.append(x.copy()); zs.append(z.copy()); us.append(u); ws.append(w)
    return dict(xs=np.array(xs), zs=np.array(zs), us=np.array(us), ws=np.array(ws), infeasible_at=-1)


def nominal_loop(mpc, x0, wmax, steps=40, seed=0):
    """Chapter A11's nominal MPC applied to the disturbed plant: re-solve from the measured state."""
    rng = np.random.default_rng(seed)
    x = np.array(x0, float); xs, us = [x.copy()], []
    for t in range(steps):
        sol = mpc.solve(x)
        if not sol.success:
            return dict(xs=np.array(xs), us=np.array(us), infeasible_at=t)
        u = float(sol.U[0].item())
        x = A @ x + B.ravel() * u + rng.uniform(-wmax, wmax, 2)
        xs.append(x.copy()); us.append(u)
    return dict(xs=np.array(xs), us=np.array(us), infeasible_at=-1)


section("Part 5: the experiment, disturbed loops from (-3, 2), |w_i| <= 0.1, 40 steps, five seeds")
x0 = [-3.0, 2.0]
print("start inside the tube region:", regt.contains(x0), "; inside chapter A11's region:", reg.contains(x0))
for seed in range(5):
    r = tube_loop(mpc_t, x0, WMAX, seed=seed)
    e = r["xs"] - r["zs"]
    in_E = all(E.contains(ei, 1e-9) for ei in e)
    viol = sum(not X.contains(xi, 1e-9) for xi in r["xs"])
    print(f"seed {seed}: box violations {viol}, max |u| {np.abs(r['us']).max():.3f}, every error in E: {in_E}, "
          f"max |e| = ({np.abs(e[:, 0]).max():.3f}, {np.abs(e[:, 1]).max():.3f}), z_40 = {r['zs'][-1].round(4)}, "
          f"x_40 = {r['xs'][-1].round(3)}, |x_40| = {np.linalg.norm(r['xs'][-1]):.3f}")
r0 = tube_loop(mpc_t, x0, WMAX, seed=0)
print("seed 0, first inputs u =", r0["us"][:5].round(4), "; nominal plan's first move v_0 =",
      round(float(mpc_t.solve(np.array(x0)).U[0].item()), 4))
print("seed 0, |z_t|:", np.linalg.norm(r0["zs"], axis=1)[[0, 2, 4, 6, 8, 10]].round(4))

section("Part 5: chapter A11's nominal MPC on the same disturbed plant (no tube)")
for seed in range(5):
    r = nominal_loop(mpc_n, x0, WMAX, seed=seed)
    viol = [t for t, xi in enumerate(r["xs"]) if not X.contains(xi, 1e-9)]
    print(f"seed {seed}: infeasible at step {r['infeasible_at']}, box violations {len(viol)}"
          + (f" at steps {viol[:6]}" if viol else "") + f", max |u| {np.abs(r['us']).max():.3f}"
          + (f", worst state {r['xs'][viol[0]].round(3)}" if viol else "") + f", final {r['xs'][-1].round(3)}")
# every grid start of chapter A11's region, three seeds each: where does the nominal controller fail?
fails, total = [], 0
for p_ in np.arange(-4, 4.01, 0.5):
    for v_ in np.arange(-3, 3.01, 0.5):
        s0 = [float(p_), float(v_)]
        if not reg.contains(s0, 1e-7):
            continue
        for seed in range(3):
            total += 1
            rr = nominal_loop(mpc_n, s0, WMAX, seed=seed)
            viol = [t for t, xi in enumerate(rr["xs"]) if not X.contains(xi, 1e-9)]
            if rr["infeasible_at"] >= 0 or viol:
                fails.append((s0, seed, rr["infeasible_at"], viol[:2], regt.contains(s0, 1e-7)))
print(f"grid of starts in chapter A11's region (spacing 0.5), 3 seeds each: {total} nominal runs, {len(fails)} fail "
      f"(infeasible or outside the box), all from starts on the boundary of C_inf; "
      f"{sum(f[4] for f in fails)} of those starts are accepted by the tube controller")
for s0, seed, inf, viol, _ in fails[:6]:
    print(f"   start {s0}, seed {seed}: infeasible at step {inf}" + (f", outside the box at step {viol[0]}" if viol else ""))
ex = nominal_loop(mpc_n, [3.0, 1.5], WMAX, seed=1)
print(f"the page's example, (3, 1.5) with seed 1: x_1 = {ex['xs'][1].round(4)} (p_1 > 4), then no feasible plan; "
      f"the tube controller's region does not contain (3, 1.5): {not regt.contains([3.0, 1.5], 1e-7)}")

# --------------------------------------------------------------------------------------------------------------
section("Part 6: the price of robustness against the disturbance size")
for wmax in (0.05, 0.1, 0.15):
    W_, Fs_ = rpi_sets(wmax)
    F8 = Fs_[S_USE - 1]; a_ = max(np.abs(AKs @ v).max() / wmax for v in W_.vertices_2d())
    E_ = scaled(F8, 1 / (1 - a_))
    h_ = X.h - support(E_, X.H); ku_ = support(E_, K)[0]; um_ = UMAX - ku_
    Xt_ = HPolytope(X.H, h_)
    XKt_ = closed_loop_admissible_set_scalar_input(Xt_, K, -um_, um_); Xft_, _ = maximal_invariant(AK, XKt_)
    Ct_, _ = maximal_control_invariant(A, B, Xt_, -um_, um_)
    regt_ = n_step_controllable_sets(A, B, Xt_, Xft_, -um_, um_, N)[-1]
    mpc_ = LinearMPC(A, B, Q, R, N, P=P, x_lower=[-h_[2], -h_[3]], x_upper=[h_[0], h_[1]], u_lower=[-um_], u_upper=[um_],
                     terminal_H=Xft_.H, terminal_h=Xft_.h)
    rr = tube_loop(mpc_, x0, wmax, seed=0, E=E_)
    fin = f"|x_40| = {np.linalg.norm(rr['xs'][-1]):.3f}" if rr["infeasible_at"] < 0 else f"nominal problem infeasible at step {rr['infeasible_at']}"
    print(f"|w_i| <= {wmax}: E area {area(E_):.4f}, box rows {h_[:2].round(3)}, |v| <= {um_:.3f}, X_f area {area(Xft_):.3f}, "
          f"C_inf area {area(Ct_):.3f}, N = 5 region {area(regt_):.3f}, start (-3, 2) feasible: {regt_.contains(x0)}; seed 0 loop: {fin}")

# --------------------------------------------------------------------------------------------------------------
section("Exercise: the error stays in E for the worst-case corner disturbance")
x = np.array(x0); z = x.copy(); worst = 0.0
for t in range(40):
    sol = mpc_t.solve(z); v = float(sol.U[0].item()); u = v - float((K @ (x - z)).item())
    w = WMAX * np.sign(E.H[0] @ (x - z) if t else np.ones(2)) * np.ones(2)
    x = A @ x + B.ravel() * u + w; z = A @ z + B.ravel() * v
    worst = max(worst, float(np.max(E.H @ (x - z) - E.h)))
print(f"corner disturbances w = +-0.1 every step: largest support-function excess of e over E = {worst:.2e} (never positive)")

# --------------------------------------------------------------------------------------------------------------
section("Exercise: the scaling factor for s = 4, 6, 8, and the w_max = 0.2 row")
for s_ in (4, 6, 8):
    a_ = max(np.abs(np.linalg.matrix_power(AK, s_) @ v).max() / WMAX for v in W.vertices_2d())
    print(f"s = {s_}: alpha = {a_:.4e}, area of F_s / (1 - alpha) = {area(scaled(Fs[s_ - 1], 1 / (1 - a_))):.4f}")
W2, Fs2 = rpi_sets(0.2); a2 = max(np.abs(AKs @ v).max() / 0.2 for v in W2.vertices_2d()); E2 = scaled(Fs2[S_USE - 1], 1 / (1 - a2))
h2 = X.h - support(E2, X.H); um2 = UMAX - support(E2, K)[0]; Xt2 = HPolytope(X.H, h2)
XKt2 = closed_loop_admissible_set_scalar_input(Xt2, K, -um2, um2); Xft2, _ = maximal_invariant(AK, XKt2)
reg2 = n_step_controllable_sets(A, B, Xt2, Xft2, -um2, um2, N)[-1]
print(f"w_max = 0.2: E area {area(E2):.3f}, box {h2[0]:.3f} x {h2[1]:.3f}, |v| <= {um2:.3f}, N = 5 region area {area(reg2):.2f}, (-3, 2) feasible: {reg2.contains(x0)}")
