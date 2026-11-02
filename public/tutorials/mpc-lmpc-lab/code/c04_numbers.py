"""Reproduce every number quoted on chapter C4 of "Advanced MPC and Learning MPC",
"Learning MPC racing: closing the loop" (c04-learning-mpc-racing-closing-the-loop.html).

Usage:  python c04_numbers.py          # drive the laps and print everything the page quotes (about a minute)
        python c04_numbers.py data     # print the page's data file, interactive/c04-data.js
Needs numpy and the course package (its QP solver, advanced_mpc_lmpc.qp.solve_qp).

The track, the dynamic bicycle model of the 1:10-scale car, the noisy path-following controller that drives the
seed laps and the local regression of lab B4 are copied verbatim from the course package's racing module
(advanced_mpc_lmpc/racing.py as shipped in the course download), so that this script runs on its own: the
first half of the file is that copy, the second half is the chapter's own controller.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass

import numpy as np

from advanced_mpc_lmpc.qp import solve_qp

DATA = len(sys.argv) > 1 and sys.argv[1] == "data"
out = (lambda *a, **k: None) if DATA else print
np.set_printoptions(precision=4, suppress=True)
G = 9.81


def section(title):
    out(f"\n=== {title}")


# ==============================================================================================================
# Part one: the track, the car, the seed laps and the local regression, copied from the course package.
# ==============================================================================================================

@dataclass
class Track:
    s: np.ndarray
    xy: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    length: float
    half_width: float

    @property
    def ds(self):
        return np.diff(np.r_[self.s, self.length])

    def wrap(self, s):
        return np.mod(np.asarray(s, float), self.length)

    def curvature_at(self, s):
        grid = np.r_[self.s, self.length]
        return np.interp(self.wrap(s), grid, np.r_[self.curvature, self.curvature[0]])

    def normal(self):
        return np.c_[-np.sin(self.heading), np.cos(self.heading)]

    def to_global(self, ey):
        ey = np.broadcast_to(np.asarray(ey, float), self.s.shape)
        return self.xy + ey[:, None] * self.normal()


def _demo_curve(t):
    return 45 * np.cos(t) + 5 * np.cos(2 * t), 30 * np.sin(t) + 6 * np.sin(3 * t)


def _demo_curve_d1(t):
    return -45 * np.sin(t) - 10 * np.sin(2 * t), 30 * np.cos(t) + 18 * np.cos(3 * t)


def _demo_curve_d2(t):
    return -45 * np.cos(t) - 20 * np.cos(2 * t), -30 * np.sin(t) - 54 * np.sin(3 * t)


def build_demo_track(n=400, half_width=3.0, dense=200_000):
    t = np.linspace(0.0, 2.0 * np.pi, dense + 1)
    speed = np.hypot(*_demo_curve_d1(t))
    s_dense = np.r_[0.0, np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(t))]
    length = float(s_dense[-1])
    s = np.arange(n) * length / n
    tq = np.interp(s, s_dense, t)
    half = n // 2
    if n % 2 == 0:
        tq[half + 1:] = 2.0 * np.pi - tq[1:half][::-1]
    x, y = _demo_curve(tq)
    dx, dy = _demo_curve_d1(tq)
    ddx, ddy = _demo_curve_d2(tq)
    curvature = (dx * ddy - dy * ddx) / np.hypot(dx, dy) ** 3
    if n % 2 == 0:
        curvature[half + 1:] = curvature[1:half][::-1]
    heading = np.unwrap(np.arctan2(dy, dx))
    return Track(s=s, xy=np.c_[x, y], heading=heading, curvature=curvature, length=length, half_width=float(half_width))


def scale_track_geometry(track, factor, half_width=None):
    return Track(s=track.s * factor, xy=track.xy * factor, heading=track.heading.copy(), curvature=track.curvature / factor,
                 length=track.length * factor, half_width=float(track.half_width * factor if half_width is None else half_width))


def track_global_position(track, s, ey):
    s = track.wrap(np.atleast_1d(s))
    ey = np.broadcast_to(np.asarray(ey, float), s.shape)
    grid = np.r_[track.s, track.length]

    def periodic(values):
        return np.interp(s, grid, np.r_[values, values[0]])

    x, y = periodic(track.xy[:, 0]), periodic(track.xy[:, 1])
    cos_h, sin_h = periodic(np.cos(track.heading)), periodic(np.sin(track.heading))
    norm = np.hypot(cos_h, sin_h)
    return np.c_[x - ey * sin_h / norm, y + ey * cos_h / norm]


@dataclass(frozen=True)
class VehicleParams:
    m: float = 1.98
    lf: float = 0.125
    lr: float = 0.125
    Iz: float = 0.024
    B: float = 1.0
    C: float = 1.25
    peak_friction: float = 0.8

    @property
    def D(self):
        return self.peak_friction * self.m * G / 2.0


BARC = VehicleParams()
LMPC_INPUT_BOUNDS = np.array([[-0.5, -10.0], [0.5, 10.0]])


def bicycle_dynamics(x, u, kappa, params=BARC):
    vx, vy, r, epsi, _, ey = (float(v) for v in x)
    delta, a = float(u[0]), float(u[1])
    alpha_f = delta - math.atan2(vy + params.lf * r, vx)
    alpha_r = -math.atan2(vy - params.lr * r, vx)
    Fyf = params.D * math.sin(params.C * math.atan(params.B * alpha_f))
    Fyr = params.D * math.sin(params.C * math.atan(params.B * alpha_r))
    sdot = (vx * math.cos(epsi) - vy * math.sin(epsi)) / (1.0 - kappa * ey)
    return np.array([
        a - Fyf * math.sin(delta) / params.m + r * vy,
        (Fyf * math.cos(delta) + Fyr) / params.m - r * vx,
        (params.lf * Fyf * math.cos(delta) - params.lr * Fyr) / params.Iz,
        r - kappa * sdot,
        sdot,
        vx * math.sin(epsi) + vy * math.cos(epsi),
    ])


def pid_control(x, target_speed=0.8, rng=None, steer_noise=0.25, accel_noise=0.10):
    rng = np.random.default_rng() if rng is None else rng
    steer_dither = float(np.clip(steer_noise * rng.standard_normal(), -0.9, 0.9))
    accel_dither = float(np.clip(accel_noise * rng.standard_normal(), -0.2, 0.2))
    return np.array([-0.6 * x[5] - 0.9 * x[3] + steer_dither, 1.5 * (target_speed - x[0]) + accel_dither])


def step_plant(track, x, u, dt=0.1, inner_dt=0.001):
    """Hold u for dt while integrating the bicycle model with explicit Euler steps of inner_dt."""
    x = np.asarray(x, float).copy()
    for _ in range(int(round(dt / inner_dt))):
        x = x + inner_dt * bicycle_dynamics(x, u, float(track.curvature_at(x[4])))
    return x


def simulate_pid_lap(track, seed=0, target_speed=0.8, dt=0.1, inner_dt=0.001, x0=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0), max_steps=2000):
    rng = np.random.default_rng(seed)
    X, U = [np.asarray(x0, float)], []
    while X[-1][4] < track.length:
        if len(U) >= max_steps:
            raise RuntimeError("the lap did not finish within max_steps")
        u = np.clip(pid_control(X[-1], target_speed, rng), LMPC_INPUT_BOUNDS[0], LMPC_INPUT_BOUNDS[1])
        U.append(u)
        X.append(step_plant(track, X[-1], u, dt, inner_dt))
    return np.array(X), np.array(U)


def kinematic_affine_linearization(x, track, dt=0.1):
    vx, vy, r, epsi, s, ey = (float(v) for v in x)
    kappa = float(track.curvature_at(s))
    den = 1.0 - kappa * ey
    c, sn = math.cos(epsi), math.sin(epsi)
    tangential = vx * c - vy * sn
    d_tangential_depsi = -vx * sn - vy * c
    sdot = tangential / den
    A = np.zeros((3, 6))
    A[0] = [-dt * kappa * c / den, dt * kappa * sn / den, dt, 1.0 - dt * kappa * d_tangential_depsi / den, 0.0, -dt * kappa ** 2 * tangential / den ** 2]
    A[1] = [dt * c / den, -dt * sn / den, 0.0, dt * d_tangential_depsi / den, 1.0, dt * kappa * tangential / den ** 2]
    A[2] = [dt * sn, dt * c, 0.0, dt * tangential, 0.0, 1.0]
    f = np.array([epsi + dt * (r - kappa * sdot), s + dt * sdot, ey + dt * (vx * sn + vy * c)])
    return A, f - A @ np.asarray(x, float), kappa


def _weighted_ridge(Phi, Y, w, ridge):
    weighted = Phi * w[:, None]
    normal = Phi.T @ weighted + ridge * np.eye(Phi.shape[1])
    return np.linalg.solve(normal, weighted.T @ Y)


def _structured_dynamic_fit(X, U, Xnext, w, ridge):
    ones = np.ones((len(X), 1))
    A, B, c = np.zeros((3, 6)), np.zeros((3, 2)), np.zeros(3)
    theta_vx = _weighted_ridge(np.hstack([X[:, :3], U[:, [1]], ones]), Xnext[:, 0], w, ridge)
    A[0, :3], B[0, 1], c[0] = theta_vx[:3], theta_vx[3], theta_vx[4]
    theta_lat = _weighted_ridge(np.hstack([X[:, :3], U[:, [0]], ones]), Xnext[:, 1:3], w, ridge)
    A[1:3, :3], B[1:3, 0], c[1:3] = theta_lat[:3].T, theta_lat[3], theta_lat[4]
    return A, B, c


class LocalRacingModel:
    """Lab B4's local regression: per lap the nearest transitions in the scaled l1 metric, Epanechnikov weights,
    structured weighted ridge rows for vx, vy, r and the exact Frenet rows."""

    def __init__(self, track, dt=0.1, max_points_per_lap=7, bandwidth=5.0, ridge=1e-8, scaling=(0.1, 1.0, 1.0, 1.0, 1.0)):
        self.track, self.dt, self.m, self.h, self.ridge = track, dt, int(max_points_per_lap), float(bandwidth), float(ridge)
        self.scaling = np.asarray(scaling, float)
        self.laps = []

    def add_trajectory(self, X, U):
        X, U = np.asarray(X, float), np.asarray(U, float)
        self.laps.append((X[:-1], U, X[1:]))

    def fit(self, x, u):
        query = np.r_[np.asarray(x, float)[:3], np.asarray(u, float)]
        rows_X, rows_U, rows_next, weights, fallback = [], [], [], [], 0
        for Xk, Uk, Xn in self.laps:
            d = np.abs(np.hstack([Xk[:, :3], Uk]) - query) @ self.scaling
            order = np.argsort(d, kind="stable")
            inside = order[d[order] < self.h]
            if len(inside):
                idx, h_lap = inside[:self.m], self.h
            else:
                idx, h_lap, fallback = order[:self.m], 1.05 * float(d[order[:self.m]].max()), fallback + 1
            w = 0.75 * (1.0 - (d[idx] / h_lap) ** 2)
            rows_X.append(Xk[idx]); rows_U.append(Uk[idx]); rows_next.append(Xn[idx]); weights.append(w)
        A_dyn, B_dyn, c_dyn = _structured_dynamic_fit(np.vstack(rows_X), np.vstack(rows_U), np.vstack(rows_next), np.concatenate(weights), self.ridge)
        A_kin, c_kin, _ = kinematic_affine_linearization(x, self.track, self.dt)
        return np.vstack([A_dyn, A_kin]), np.vstack([B_dyn, np.zeros((3, 2))]), np.r_[c_dyn, c_kin], fallback


def periodic_minimum_time_speed_profile(track, mu=1.0, g=G, vmax=35.0, amin=-6.0, amax=3.0, tol=1e-9, max_iter=100):
    """Lab B3's planner: the largest speed-squared profile under the lateral cap and the acceleration bounds."""
    ds = track.ds
    n = len(ds)
    zmax = np.minimum(vmax ** 2, mu * g / np.maximum(np.abs(track.curvature), 1e-9))
    z = zmax.copy()
    for _ in range(max_iter):
        z_previous = z.copy()
        for i in range(n):
            j = (i + 1) % n
            z[j] = min(z[j], z[i] + 2.0 * amax * ds[i])
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            z[i] = min(z[i], z[j] - 2.0 * amin * ds[i])
        if np.max(np.abs(z - z_previous)) <= tol:
            break
    z_next = np.roll(z, -1)
    return {"time": float(np.sum(2.0 * ds / (np.sqrt(z) + np.sqrt(z_next)))), "v": np.sqrt(z), "a": (z_next - z) / (2.0 * ds)}


# ==============================================================================================================
# Part two: the Learning MPC racing controller.
# ==============================================================================================================

DT, N = 0.1, 14
RATE = np.diag([5.0, 50.0])          # penalty on (delta_k - delta_{k-1}, a_k - a_{k-1})
UMIN, UMAX = LMPC_INPUT_BOUNDS
VX_MIN, VX_MAX = 0.1, 4.0            # a speed box, so that the affine model is never asked about a stopped or absurd car
MODEL_LAPS, SS_LAPS, SS_POINTS = 4, 4, 12
WRAP_POINTS = 40                     # stored states of the lap's own start appended beyond the finish line


class RacingLMPC:
    """Learning MPC for lap-time racing: a local affine time-varying model refitted along the horizon from the
    stored laps, a local convex safe set of stored states around the predicted terminal point with their
    costs-to-go in samples to the finish line, an input-rate cost, the lane and input constraints, and one QP
    per sample solved by the course's interior-point method."""

    def __init__(self, track):
        self.track, self.L, self.W = track, track.length, track.half_width
        self.laps = []                 # dicts: X, U, steps, ok (respected the lane)

    def add_lap(self, X, U, ok=True):
        self.laps.append({"X": np.asarray(X, float), "U": np.asarray(U, float), "steps": len(U), "ok": bool(ok)})

    def good_laps(self):
        return [lap for lap in self.laps if lap["ok"]]

    def model(self):
        """The local model's data: the most recent successful laps."""
        m = LocalRacingModel(self.track, DT)
        for lap in self.good_laps()[-MODEL_LAPS:]:
            m.add_trajectory(lap["X"], lap["U"])
        return m

    def safe_set(self, s_terminal):
        """SS_POINTS stored states of each of the SS_LAPS fastest laps, nearest in progress to s_terminal, with
        their cost-to-go in samples; the lap's own first WRAP_POINTS states, shifted by one lap length, stand for
        the lap after the finish line, with negative cost-to-go."""
        laps = sorted(self.good_laps(), key=lambda lap: lap["steps"])[:SS_LAPS]
        S, J, U = [], [], []
        for lap in laps:
            X, T = lap["X"], lap["steps"]
            Xw = X[:WRAP_POINTS].copy(); Xw[:, 4] += self.L
            Xe = np.vstack([X[:T], Xw])
            Je = np.r_[T - np.arange(T), -np.arange(WRAP_POINTS)].astype(float)
            Ue = np.vstack([lap["U"][:T], lap["U"][:WRAP_POINTS]])
            idx = np.argsort(np.abs(Xe[:, 4] - s_terminal), kind="stable")[:SS_POINTS]
            S.append(Xe[idx]); J.append(Je[idx]); U.append(Ue[idx])
        return np.vstack(S), np.concatenate(J), np.vstack(U)

    def solve(self, x0, u_prev, U_guess, model):
        """One Learning MPC step: linearise along the guess, build the QP in (U, lambda), solve it."""
        n, m = 6, 2
        A_list, B_list, c_list, xbar, fallbacks = [], [], [], np.asarray(x0, float).copy(), 0
        for k in range(N):                                   # the LTV model along the guessed trajectory
            A, B, c, fb = model.fit(xbar, U_guess[k]); fallbacks += fb
            A_list.append(A); B_list.append(B); c_list.append(c)
            xbar = A @ xbar + B @ U_guess[k] + c
        P, Gm, d = [np.eye(n)], [np.zeros((n, m * N))], [np.zeros(n)]
        for k in range(N):                                   # x_{k+1} = A_k x_k + B_k u_k + c_k, stacked in U
            Gk = A_list[k] @ Gm[k]; Gk[:, m * k:m * k + m] += B_list[k]
            P.append(A_list[k] @ P[k]); Gm.append(Gk); d.append(A_list[k] @ d[k] + c_list[k])
        base = [P[k] @ x0 + d[k] for k in range(N + 1)]      # the part of x_k that does not depend on U
        S, J, _ = self.safe_set(float(base[N][4]))
        M = len(S)
        nz = m * N + M
        # cost: input rates, and the learned terminal value lambda' J
        D = np.eye(m * N)
        for k in range(1, N):
            D[m * k:m * k + m, m * (k - 1):m * k] = -np.eye(m)
        Rbar = np.kron(np.eye(N), RATE)
        p = np.zeros(m * N); p[:m] = u_prev
        H = np.zeros((nz, nz)); H[:m * N, :m * N] = 2.0 * D.T @ Rbar @ D
        f = np.zeros(nz); f[:m * N] = -2.0 * D.T @ Rbar @ p; f[m * N:] = J
        # lane and speed constraints on every predicted state
        A_ub, b_ub = [], []
        for k in range(1, N + 1):
            for row, sign, lim in ((5, 1.0, self.W), (5, -1.0, self.W), (0, 1.0, VX_MAX), (0, -1.0, -VX_MIN)):
                r = np.zeros(nz); r[:m * N] = sign * Gm[k][row]
                A_ub.append(r); b_ub.append(lim - sign * base[k][row])
        # terminal state in the convex hull of the local safe set
        A_eq = np.zeros((n + 1, nz)); b_eq = np.zeros(n + 1)
        A_eq[:n, :m * N] = Gm[N]; A_eq[:n, m * N:] = -S.T; b_eq[:n] = -base[N]
        A_eq[n, m * N:] = 1.0; b_eq[n] = 1.0
        lb = np.r_[np.tile(UMIN, N), np.zeros(M)]; ub = np.r_[np.tile(UMAX, N), np.ones(M)]
        res = solve_qp(H, f, A_ub=np.vstack(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub)
        U = res.x[:m * N].reshape(N, m); lam = res.x[m * N:]
        X = np.array([base[k] + Gm[k] @ res.x[:m * N] for k in range(N + 1)])
        return {"success": res.success, "U": U, "lam": lam, "X": X, "S": S, "J": J, "terminal": S.T @ lam,
                "value": float(lam @ J), "nit": res.nit, "fallbacks": fallbacks}

    def drive_lap(self, x_start, u_prev, snapshots=None, snap_every=25):
        """One lap of closed-loop Learning MPC from x_start; the model and the safe set come from stored laps."""
        model = self.model()
        fastest = min(self.good_laps(), key=lambda lap: lap["steps"])
        x = np.asarray(x_start, float).copy()
        k0 = int(np.argmin(np.abs(fastest["X"][:fastest["steps"], 4] - x[4])))
        U_guess = np.array([fastest["U"][min(k0 + k, fastest["steps"] - 1)] for k in range(N)])
        X, U, stats = [x.copy()], [], {"qps": 0, "nit": 0, "fallbacks": 0, "model_fallbacks": 0, "solve_time": 0.0, "max_ey": 0.0}
        while x[4] < self.L:
            t0 = time.perf_counter()
            sol = self.solve(x, u_prev, U_guess, model)
            stats["solve_time"] += time.perf_counter() - t0; stats["qps"] += 1; stats["nit"] += sol["nit"]; stats["model_fallbacks"] += sol["fallbacks"]
            if sol["success"]:
                u = sol["U"][0]
                U_guess = np.vstack([sol["U"][1:], sol["U"][-1:]])
            else:                                            # the rule of lab B4's exercise 11: replay the fastest stored lap
                stats["fallbacks"] += 1
                k = int(np.argmin(np.hypot(fastest["X"][:fastest["steps"], 4] - x[4], fastest["X"][:fastest["steps"], 5] - x[5])))
                u = fastest["U"][k]
                U_guess = np.array([fastest["U"][min(k + j, fastest["steps"] - 1)] for j in range(N)])
            if snapshots is not None and len(U) % snap_every == 0 and sol["success"]:
                snapshots.append({"step": len(U), "x0": x.copy(), "X": sol["X"], "S": sol["S"], "J": sol["J"], "lam": sol["lam"],
                                  "terminal": sol["terminal"], "u": u.copy(), "value": sol["value"]})
            x = step_plant(self.track, x, u)
            stats["max_ey"] = max(stats["max_ey"], abs(float(x[5])))
            U.append(u.copy()); X.append(x.copy()); u_prev = u
            if len(U) > 2000:
                raise RuntimeError("the lap did not finish")
        return np.array(X), np.array(U), stats


# --------------------------------------------------------------------------------------------------------------
def main():
    t_all = time.perf_counter()
    track = scale_track_geometry(build_demo_track(n=400, half_width=3.0), factor=0.08, half_width=0.4)
    section("The track and the plant (lab B4's)")
    out(f"scaled track: length {track.length:.3f} m, half-width {track.half_width} m, curvature range "
        f"[{track.curvature.min():.3f}, {track.curvature.max():.3f}] 1/m, tightest radius {1 / track.curvature.max():.3f} m")
    out("car: BARC bicycle model, inputs |delta| <= 0.5 rad, |a| <= 10 m/s^2, sample 0.1 s, inner Euler step 1 ms")

    section("Seed laps: the noisy path follower at a target speed of 0.8 m/s (seeds 0 to 3)")
    lmpc = RacingLMPC(track)
    for seed in range(4):
        X, U = simulate_pid_lap(track, seed=seed)
        lmpc.add_lap(X, U, ok=bool(np.all(np.abs(X[:, 5]) <= track.half_width + 1e-9)))
        out(f"seed {seed}: {len(U)} samples = {DT * len(U):.1f} s, max |e_y| {np.abs(X[:, 5]).max():.3f} m, "
            f"max v_x {X[:, 0].max():.3f} m/s")

    section("Learning MPC laps (N = 14, rate penalty diag(5, 50), safe set 4 laps x 12 points)")
    x = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_prev = np.zeros(2)
    snapshots, snapshot_lap = [], 3
    lap_rows = []
    for lap_no in range(1, 21):
        snaps = snapshots if lap_no == snapshot_lap else None
        X, U, st = lmpc.drive_lap(x, u_prev, snapshots=snaps)
        ok = st["max_ey"] <= track.half_width + 1e-9
        lmpc.add_lap(X, U, ok=ok)
        u_in = bool(np.all(U >= UMIN - 1e-9) and np.all(U <= UMAX + 1e-9))
        lap_rows.append({"lap": lap_no, "steps": len(U), "time": DT * len(U), "ok": ok, "inputs_ok": u_in, "max_ey": st["max_ey"],
                         "max_vx": float(X[:, 0].max()), "qps": st["qps"], "nit": st["nit"] / max(st["qps"], 1),
                         "fallbacks": st["fallbacks"], "model_fallbacks": st["model_fallbacks"], "ms_per_qp": 1000 * st["solve_time"] / max(st["qps"], 1),
                         "start": "standing" if lap_no == 1 else "flying"})
        out(f"lap {lap_no:2d}: {len(U):3d} samples = {DT * len(U):5.1f} s ({lap_rows[-1]['start']} start), lane {'respected' if ok else 'VIOLATED'} "
            f"(max |e_y| {st['max_ey']:.3f} m), inputs {'within bounds' if u_in else 'OUT OF BOUNDS'}, max v_x {X[:, 0].max():.3f} m/s, "
            f"{st['qps']} QPs, {st['nit'] / max(st['qps'], 1):.1f} iterations each, {1000 * st['solve_time'] / max(st['qps'], 1):.1f} ms each, "
            f"{st['fallbacks']} fallbacks, {st['model_fallbacks']} model support fallbacks")
        x = X[-1].copy(); x[4] -= track.length                 # the next lap starts where this one ended
        u_prev = U[-1].copy()

    section("Summary of the learning")
    seed_times = [DT * lap["steps"] for lap in lmpc.laps[:4]]
    times = [r["time"] for r in lap_rows]
    best = min(range(len(times)), key=lambda i: times[i])
    out(f"seed laps: {', '.join(f'{t:.1f}' for t in seed_times)} s; fastest seed {min(seed_times):.1f} s")
    out(f"Learning MPC lap times: {', '.join(f'{t:.1f}' for t in times)} s")
    out(f"fastest lap: lap {lap_rows[best]['lap']}, {times[best]:.1f} s, {100 * (1 - times[best] / min(seed_times)):.1f} percent faster than the fastest seed")
    stalled = next((r["lap"] for i, r in enumerate(lap_rows) if i >= 2 and all(times[i] >= times[j] - 1e-9 for j in range(i))), None)
    improvements = [times[i] - times[i - 1] for i in range(1, len(times))]
    out(f"lap-to-lap changes (s): {', '.join(f'{d:+.1f}' for d in improvements)}")
    out(f"first lap not faster than every earlier lap: {stalled}; laps violating the lane: {[r['lap'] for r in lap_rows if not r['ok']] or 'none'}; "
        f"laps with inputs out of bounds: {[r['lap'] for r in lap_rows if not r['inputs_ok']] or 'none'}")
    out(f"total QPs {sum(r['qps'] for r in lap_rows)}, total fallbacks {sum(r['fallbacks'] for r in lap_rows)}, "
        f"model support fallbacks {sum(r['model_fallbacks'] for r in lap_rows)}, mean solve time {np.mean([r['ms_per_qp'] for r in lap_rows]):.1f} ms per QP")

    section("The fastest lap against lab B3's centreline speed planner on the same track")
    best_lap = lmpc.laps[4 + best]
    plan = periodic_minimum_time_speed_profile(track, mu=BARC.peak_friction, g=G, vmax=35.0, amin=-10.0, amax=10.0)
    Xb = best_lap["X"][:best_lap["steps"]]
    v_plan_at = np.interp(track.wrap(Xb[:, 4]), np.r_[track.s, track.length], np.r_[plan["v"], plan["v"][0]])
    out(f"planner (mu = {BARC.peak_friction}, |a| <= 10, centreline, no tyre dynamics): lap time {plan['time']:.2f} s, "
        f"speed {plan['v'].min():.2f} to {plan['v'].max():.2f} m/s")
    out(f"fastest Learning MPC lap: {times[best]:.1f} s, speed {Xb[:, 0].min():.2f} to {Xb[:, 0].max():.2f} m/s, "
        f"mean ratio v_x / v_plan {np.mean(Xb[:, 0] / v_plan_at):.2f}, lowest ratio {np.min(Xb[:, 0] / v_plan_at):.2f} at s = {Xb[np.argmin(Xb[:, 0] / v_plan_at), 4]:.2f} m")
    out(f"mean |e_y| on the fastest lap {np.abs(Xb[:, 5]).mean():.3f} m, max {np.abs(Xb[:, 5]).max():.3f} m of {track.half_width}")

    section(f"One QP of lap {snapshot_lap}: the snapshots the page's second widget replays")
    for sn in snapshots[:3]:
        act = np.where(sn["lam"] > 1e-6)[0]
        out(f"step {sn['step']}: x0 = ({sn['x0'][0]:.3f}, {sn['x0'][1]:.3f}, {sn['x0'][2]:.3f}, {sn['x0'][3]:.3f}, {sn['x0'][4]:.3f}, {sn['x0'][5]:.3f}), "
            f"u = ({sn['u'][0]:.3f}, {sn['u'][1]:.3f}), terminal s {sn['terminal'][4]:.3f}, learned value {sn['value']:.2f} samples, "
            f"{len(act)} active weights, safe-set costs {sn['J'].min():.0f} to {sn['J'].max():.0f}")
    out(f"{len(snapshots)} snapshots recorded")
    out(f"\nscript time {time.perf_counter() - t_all:.0f} s")

    if DATA:
        r4 = lambda a: np.round(np.asarray(a, float), 4).tolist()
        inner, outer = track.to_global(-track.half_width), track.to_global(track.half_width)
        data = {
            "track": {"length": round(track.length, 4), "halfWidth": track.half_width, "s": r4(track.s), "xy": r4(track.xy),
                      "inner": r4(inner), "outer": r4(outer), "curvature": r4(track.curvature)},
            "seeds": [{"X": r4(lap["X"]), "U": r4(lap["U"]), "steps": lap["steps"]} for lap in lmpc.laps[:4]],
            "laps": [{"X": r4(lap["X"]), "U": r4(lap["U"]), "steps": lap["steps"], "ok": lap["ok"], "stats": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}}
                     for lap, row in zip(lmpc.laps[4:], lap_rows)],
            "plan": {"time": round(plan["time"], 3), "v": r4(plan["v"])},
            "snapshotLap": snapshot_lap,
            "snapshots": [{"step": sn["step"], "x0": r4(sn["x0"]), "X": r4(sn["X"]), "S": r4(sn["S"]), "J": r4(sn["J"]), "lam": r4(sn["lam"]),
                           "terminal": r4(sn["terminal"]), "u": r4(sn["u"]), "value": round(sn["value"], 3)} for sn in snapshots],
        }
        print("/* Chapter C4: the RC-car track, the four seed laps, the Learning MPC laps and the QP snapshots of one lap, computed in Python by code/c04_numbers.py. */")
        print("window.C04_DATA = " + json.dumps(data, separators=(",", ":")) + ";")


if __name__ == "__main__":
    main()
