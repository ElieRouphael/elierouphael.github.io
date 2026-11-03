"""Reproduce every number quoted on chapter C2 of "Advanced MPC and Learning MPC",
"Reference tracking and offset-free MPC" (c02-reference-tracking-and-offset-free-mpc.html).

Usage:  python c02_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
Runs in a few seconds.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import place_poles
from scipy.spatial import ConvexHull

from advanced_mpc_lmpc.linear import dlqr
from advanced_mpc_lmpc.mpc import LinearMPC
from advanced_mpc_lmpc.polytopes import HPolytope, closed_loop_admissible_set_scalar_input, maximal_invariant

np.set_printoptions(precision=4, suppress=True)


def section(title):
    print(f"\n=== {title}")


def area(P):
    V = P.vertices_2d()
    return ConvexHull(V).volume if len(V) >= 3 else 0.0


# the canonical system, with the position as the measured output
A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]]); C = np.array([[1.0, 0.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])
XLB, XUB, UMAX = np.array([-4.0, -3.0]), np.array([4.0, 3.0]), 1.0
K, P, eig = dlqr(A, B, Q, R); AK = A - B @ K
N = 5
D = -0.2                                               # the slope: the plant adds d to every input


# --------------------------------------------------------------------------------------------------------------
# The target and the deviation problem (quoted on the page).
def target(r, d_hat=0.0):
    """Steady state (x_s, u_s) with C x_s = r for x+ = A x + B (u + d): solve [[I - A, -B], [C, 0]] [x_s; u_s] = [B d; r]."""
    M = np.block([[np.eye(2) - A, -B], [C, np.zeros((1, 1))]])
    sol = np.linalg.solve(M, np.r_[B.ravel() * d_hat, r])
    return sol[:2], float(sol[2])


def tracking_mpc(x_s, u_s):
    """Chapter A11's regulator in deviation variables delta = x - x_s, delta_u = u - u_s, with the box and the
    terminal set shifted to the setpoint; returns the controller and its terminal set (in deviation coordinates)."""
    lo, hi = XLB - x_s, XUB - x_s
    ulo, uhi = -UMAX - u_s, UMAX - u_s
    Xd = HPolytope.box(lo, hi)
    XK = closed_loop_admissible_set_scalar_input(Xd, K, ulo, uhi)
    Xf, _ = maximal_invariant(AK, XK)
    mpc = LinearMPC(A, B, Q, R, N, P=P, x_lower=lo, x_upper=hi, u_lower=[ulo], u_upper=[uhi], terminal_H=Xf.H, terminal_h=Xf.h)
    return mpc, Xf


section("Part 1: steady-state targets for the cart, y = p")
for r in (2.0, -1.0, 3.5):
    xs, us = target(r)
    print(f"r = {r:5.1f}: x_s = {xs}, u_s = {us:.3f}")
print("the target equations have a unique solution: det of the 3 x 3 system =",
      round(np.linalg.det(np.block([[np.eye(2) - A, -B], [C, np.zeros((1, 1))]])), 4))

section("Part 2: the terminal set shifted to the setpoint")
for r in (0.0, 2.0, 3.5, 3.8):
    xs, us = target(r)
    mpc, Xf = tracking_mpc(xs, us)
    V = Xf.vertices_2d()
    print(f"r = {r:4.1f}: terminal set {len(V)} vertices, area {area(Xf):.3f}, position extent in deviation "
          f"[{V[:, 0].min():.3f}, {V[:, 0].max():.3f}] (the box leaves {XUB[0] - r:.1f} to the right)")

# --------------------------------------------------------------------------------------------------------------
section("Part 3: the slope, and the offset of plain tracking MPC")
# analytic: near the setpoint the controller is the LQR, delta+ = A_K delta + B d, fixed point (I - A_K)^{-1} B d
delta_ss = np.linalg.solve(np.eye(2) - AK, B.ravel() * D)
print(f"analytic steady-state deviation under u = u_s - K delta with the slope d = {D}: delta_ss = {delta_ss}, "
      f"position offset {delta_ss[0]:.4f}, input at steady state u_s - K delta_ss = {-float((K @ delta_ss).item()):.4f} "
      f"(= -d: the error supplies the input that cancels the slope)")


def run_plain(r_seq, d, x0=(0.0, 0.0)):
    x = np.array(x0, float); xs, us = [x.copy()], []
    for r in r_seq:
        x_s, u_s = target(r)
        mpc, _ = tracking_mpc(x_s, u_s)
        sol = mpc.solve(x - x_s)
        assert sol.success, "tracking problem infeasible"
        u = u_s + float(sol.U[0].item())
        x = A @ x + B.ravel() * (u + d)
        xs.append(x.copy()); us.append(u)
    return np.array(xs), np.array(us)


r_seq = [2.0] * 30 + [-1.0] * 30
Xp, Up = run_plain(r_seq, D)
print(f"plain tracking MPC, r = 2 then -1, slope d = {D}: position after 30 steps {Xp[30, 0]:.4f} (offset {Xp[30, 0] - 2:.4f}), "
      f"after 60 steps {Xp[60, 0]:.4f} (offset {Xp[60, 0] + 1:.4f}); input at the end {Up[-1]:.4f}")
print("the offset does not depend on the setpoint: it is (I - A_K)^{-1} B d in position,", round(delta_ss[0], 4))

# --------------------------------------------------------------------------------------------------------------
section("Part 4: the augmented model and a Luenberger observer for (x, d)")
Aa = np.block([[A, B], [np.zeros((1, 2)), np.ones((1, 1))]])     # input-disturbance model: x+ = A x + B (u + d), d+ = d
Ba = np.vstack([B, np.zeros((1, 1))])
Ca = np.hstack([C, np.zeros((1, 1))])
Ao = np.block([[A, np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]])   # output-disturbance model: y = p + d
Co = np.array([[1.0, 0.0, 1.0]])
obs = np.vstack([Ca @ np.linalg.matrix_power(Aa, k) for k in range(3)])
print("augmented A =", Aa.tolist(), "; observability matrix rank", np.linalg.matrix_rank(obs), "of 3")
print("detectability condition rank [[I - A, -B], [C, 0]] =",
      np.linalg.matrix_rank(np.block([[np.eye(2) - A, -B], [C, np.zeros((1, 1))]])), "of 3")


def observer_gain(poles, Am=Aa, Cm=Ca):
    """The current-estimator gain: xhat_t = pred_t + L (y_t - C pred_t), pred_{t+1} = A xhat_t + B u_t.
    Its error dynamics are A (I - L C), whose eigenvalues equal those of A - (A L) C, so the predictor-form
    gain from pole placement, Lp with eig(A - Lp C) = poles, gives L = A^{-1} Lp."""
    Lp = place_poles(Am.T, Cm.T, poles).gain_matrix.T
    return np.linalg.solve(Am, Lp)


POLES = (0.5, 0.55, 0.6)
for poles in ((0.5, 0.55, 0.6), (0.3, 0.35, 0.4), (0.8, 0.85, 0.9)):
    Lg = observer_gain(poles)
    print(f"poles {poles}: L = {Lg.ravel()}, error eigenvalues {np.sort(np.linalg.eigvals(Aa @ (np.eye(3) - Lg @ Ca)).real).round(3)}")
L_OBS = observer_gain(POLES)
print("gains to six decimals, as hard-coded in the page script (pole placement is not ported to JavaScript):",
      {str(pl): np.round(observer_gain(pl).ravel(), 6).tolist() for pl in ((0.3, 0.35, 0.4), (0.5, 0.55, 0.6), (0.8, 0.85, 0.9))})


def run_offset_free(r_seq, d, Lg, x0=(0.0, 0.0), model="input"):
    """Offset-free tracking: estimate (x, d) from the position, recompute the target from d_hat, control the estimate."""
    Am, Cm = (Aa, Ca) if model == "input" else (Ao, Co)
    x = np.array(x0, float); pred = np.zeros(3)
    xs, us, dh, targets, failures = [x.copy()], [], [], [], []
    for t, r in enumerate(r_seq):
        y = float((C @ x).item())
        xh = pred + Lg.ravel() * (y - float((Cm @ pred).item()))        # correct the prediction with the measurement
        x_s, u_s = target(r, xh[2]) if model == "input" else target(r - xh[2])
        mpc, _ = tracking_mpc(x_s, u_s)
        sol = mpc.solve(xh[:2] - x_s)
        if not sol.success:                                               # the estimate left the feasible region
            u = us[-1] if us else 0.0; failures.append(t)
        else:
            u = u_s + float(sol.U[0].item())
        x = A @ x + B.ravel() * (u + d)                                   # the plant: the slope acts on the input
        pred = Am @ xh + Ba.ravel() * u                                   # predict the next estimate
        xs.append(x.copy()); us.append(u); dh.append(xh[2]); targets.append(x_s[0])
    return np.array(xs), np.array(us), np.array(dh), np.array(targets), failures


def settled(X, lo, hi, r, tol=0.01):
    """First step t in [lo, hi) from which the position stays within tol of r up to hi; None if never."""
    return next((t for t in range(lo, hi) if all(abs(X[k, 0] - r) < tol for k in range(t, hi + 1))), None)


Xo, Uo, Dh, Ts_, F0 = run_offset_free(r_seq, D, L_OBS)
s2, s3 = settled(Xo, 0, 30, 2.0), settled(Xo, 30, 60, -1.0)
print(f"offset-free MPC (input-disturbance model, observer poles {POLES}): position after 30 steps {Xo[30, 0]:.4f}, "
      f"after 60 {Xo[60, 0]:.4f}; d_hat after 5, 10, 30 steps = {Dh[5]:.4f}, {Dh[10]:.4f}, {Dh[29]:.4f} (true {D}); "
      f"within 0.01 of r = 2 from step {s2}, of r = -1 from step {s3} ({None if s3 is None else s3 - 30} after the change); "
      f"largest |u| {np.abs(Uo).max():.3f}; input at the end {Uo[-1]:.4f}; infeasible steps: {F0 or 'none'}")
print("plain MPC's position at steps 5, 10, 30, 40, 60:", Xp[[5, 10, 30, 40, 60], 0].round(4), "; offset-free:", Xo[[5, 10, 30, 40, 60], 0].round(4))
print("plain MPC's first inputs:", Up[:4].round(4), "; offset-free:", Uo[:4].round(4))
for poles in ((0.3, 0.35, 0.4), (0.8, 0.85, 0.9)):
    Xq, Uq, Dq, _, Fq = run_offset_free(r_seq, D, observer_gain(poles))
    print(f"observer poles {poles}: within 0.01 of r = 2 from step {settled(Xq, 0, 30, 2.0)}, d_hat after 5 and 10 steps "
          f"{Dq[5]:.4f}, {Dq[10]:.4f}, largest |u| {np.abs(Uq).max():.3f}, infeasible steps {Fq or 'none'}")

section("Part 5: the output-disturbance model on the same plant")
rank_in = np.linalg.matrix_rank(np.block([[np.eye(2) - A, -B], [C, np.zeros((1, 1))]]))
rank_out = np.linalg.matrix_rank(np.block([[np.eye(2) - A, np.zeros((2, 1))], [C, np.ones((1, 1))]]))
print(f"detectability rank test [[I - A, -B_d], [C, C_d]], needs 3: input-disturbance model {rank_in}, output-disturbance model {rank_out}")
print("observability matrix rank of (A_o, C_o):", np.linalg.matrix_rank(np.vstack([Co @ np.linalg.matrix_power(Ao, k) for k in range(3)])), "of 3:",
      "a constant position and a constant output offset give the same measurements, so the model is not detectable")
Xy, Uy, Dy, _, Fy = run_offset_free([2.0] * 30, D, observer_gain(POLES, Ao, Co), model="output")
print(f"running it anyway (pole placement still returns a gain): position after 10 and 30 steps {Xy[10, 0]:.1f}, {Xy[30, 0]:.1f}, "
      f"d_hat after 30 steps {Dy[29]:.1e}, tracking problem infeasible from step {Fy[0] if Fy else None}: the loop diverges")

# --------------------------------------------------------------------------------------------------------------
section("Exercise: no slope, both controllers agree; a slope of +0.3")
Xa, _ = run_plain([2.0] * 30, 0.0); Xb, _, Db, _, _ = run_offset_free([2.0] * 30, 0.0, L_OBS)
print(f"d = 0: plain reaches {Xa[30, 0]:.4f}, offset-free {Xb[30, 0]:.4f}, d_hat stays {abs(Db).max():.1e}")
Xc, Uc = run_plain([2.0] * 30, 0.3); Xd_, Ud, Dd, _, _ = run_offset_free([2.0] * 30, 0.3, L_OBS)
print(f"d = +0.3: plain offset {Xc[30, 0] - 2:.4f} (= {np.linalg.solve(np.eye(2) - AK, B.ravel() * 0.3)[0]:.4f}), "
      f"offset-free position {Xd_[30, 0]:.4f}, d_hat {Dd[-1]:.4f}, largest |u| {np.abs(Ud).max():.3f}")
