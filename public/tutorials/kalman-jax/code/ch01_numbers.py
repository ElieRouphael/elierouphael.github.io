"""Reproduce every number quoted on chapter 01 of "Kalman Filters with JAX"
(public/tutorials/kalman-jax/01-tracking-a-vehicle-with-a-kalman-filter.html).

Usage:  python ch01_numbers.py          # print everything the page quotes
        python ch01_numbers.py data     # print the page's data file, interactive/ch01-data.js
Needs numpy; the last two sections also use JAX and Optax when they are installed, and are skipped otherwise.

The page runs on the kalman_vehicle_tracking repository's CSV as first committed (commit adf4b35,
data/example_data.csv), shipped next to this script as data/vehicle_example_data_original.csv. The repository
has since regenerated its CSV with a seed, which gives different numbers; section 10 prints those, the
numbers a reader of the download sees. The repository's original twenty-line class is copied verbatim below so
that the script runs alone; if the CSV is not next to it, the script reads it from GitHub at that commit.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE / "data" / "vehicle_example_data_original.csv"
CSV_URL = "https://raw.githubusercontent.com/ElieRouphael/kalman_vehicle_tracking/adf4b35738/data/example_data.csv"
DATA = len(sys.argv) > 1 and sys.argv[1] == "data"
out = (lambda *a, **k: None) if DATA else print


def section(title):
    out(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
# The repository's class, verbatim from kalman/kalman_filter.py at commit adf4b35.
class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x0):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P
        return self.x
# --------------------------------------------------------------------------------------------------------------


def load(path):
    if path.exists():
        text = path.read_text()
    else:                                                       # the script was downloaded on its own
        with urllib.request.urlopen(CSV_URL) as r:
            text = r.read().decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    col = lambda k: np.array([float(r[k]) for r in rows])
    return {k: col(k) for k in ("time", "acceleration", "true_position", "noisy_position", "true_velocity", "noisy_velocity")}


def run_class(zs, dt, Q, R, P0=None, x0=None):
    """The repository's tracker loop: predict, then update, for each measurement. Returns the estimates."""
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    kf = KalmanFilter(F, H, np.asarray(Q, float), np.array([[R]], float),
                      np.eye(2) if P0 is None else P0, np.array([[0], [0]]) if x0 is None else x0)
    est, vel = [], []
    for z in zs:
        kf.predict()
        x = kf.update(np.array([[z]]))
        est.append(x[0, 0]); vel.append(x[1, 0])
    return np.array(est), np.array(vel)


def run(zs, F, H, Q, R, P0, joseph=False):
    """A general linear filter (any state size, scalar measurement) that also returns gains, innovations and S."""
    n = F.shape[0]
    x, P = np.zeros(n), P0.copy()
    xs, gains, ys, Ss = [], [], [], []
    for z in zs:
        x, P = F @ x, F @ P @ F.T + Q
        y = z - H @ x
        S = H @ P @ H + R
        K = P @ H / S
        x = x + K * y
        if joseph:
            A = np.eye(n) - np.outer(K, H)
            P = A @ P @ A.T + R * np.outer(K, K)
        else:
            P = P - np.outer(K, H @ P)
        xs.append(x.copy()); gains.append(K.copy()); ys.append(y); Ss.append(S)
    return np.array(xs), np.array(gains), np.array(ys), np.array(Ss)


rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
THIRDS = [(0, 333), (333, 666), (666, 1000)]                     # as the page: 333, 333 and 334 samples
thirds = lambda a, b: [rmse(a[i:j], b[i:j]) for i, j in THIRDS]
fmt = lambda v, d=2: ", ".join(f"{x:.{d}f}" for x in v)

D = load(CSV)
t, truth, zs = D["time"], D["true_position"], D["noisy_position"]
N, dt = len(t), float(t[1] - t[0])
F2, H2 = np.array([[1.0, dt], [0.0, 1.0]]), np.array([1.0, 0.0])
v_true = 30 * t * np.cos(t ** 2)                                   # the derivative of 15 sin(t^2)
a_true = 30 * np.cos(t ** 2) - 60 * t ** 2 * np.sin(t ** 2)
Q_repo = np.array([[1.0, 0.0], [0.0, 3.0]])

if DATA:
    # The page's data file: the recorded columns and the repository's own estimates, rounded to 4 decimals.
    est, vel = run_class(zs, dt, Q_repo, 25)
    r4 = lambda a: [0 if round(float(v), 4) == 0 else (int(round(float(v), 4)) if round(float(v), 4).is_integer() else round(float(v), 4)) for v in a]
    kv = {"dt": dt, "time": r4(t), "true": r4(truth), "noisy": r4(zs), "trueVel": r4(D["true_velocity"]),
          "noisyVel": r4(D["noisy_velocity"]), "est": r4(est), "vel": r4(vel)}
    sys.stdout.write(
        "/* Kalman filters with JAX, chapter 01 data: the kalman_vehicle_tracking repository's recorded simulation (data/example_data.csv,\n"
        "   1000 samples at 0.01 s: a chirp position 15 sin(t^2) with N(0, 5^2) measurement noise, and the unrelated \"true_velocity\" 2t and\n"
        "   \"noisy_velocity\" columns), with the repository's own Kalman estimates for the page's equivalence check.\n"
        "   Written by code/ch01_numbers.py data with NumPy 2.5 on a CPU. */\n"
        "window.KV = " + json.dumps(kv, separators=(",", ":")) + ";\n")
    sys.exit()

out("numpy", np.__version__)

section("1. The data (Part 1)")
out(f"{N} samples at dt = {dt} s, t from {t[0]} to {t[-1]:.2f} s")
out(f"sensor RMSE {rmse(zs, truth):.2f} m; by thirds {fmt(thirds(zs, truth))} m")
out(f"chirp frequency at the end: t/pi = {t[-1] / np.pi:.2f} Hz; true velocity reaches +-{np.abs(v_true).max():.0f} m/s")
out(f"CSV true_velocity is 2t: max |true_velocity - 2t| = {np.abs(D['true_velocity'] - 2 * t).max():.2e}; "
    f"acceleration column constant at {set(D['acceleration'].round(9))}")
out(f"RMS of the true velocity, the score of always answering 0: {rmse(0 * t, v_true):.1f} m/s")
out(f"chirp acceleration: RMS by thirds {fmt([rmse(0 * t[i:j], a_true[i:j]) for i, j in THIRDS], 0)} m/s^2, "
    f"peak |a| {np.abs(a_true).max():.0f} m/s^2")

section("2. The repository's filter (Part 2 and the hero)")
est, vel = run_class(zs, dt, Q_repo, 25)
out(f"Q = diag(1, 3), R = 25: position RMSE {rmse(est, truth):.2f} m against the sensor's {rmse(zs, truth):.2f} "
    f"({100 * (1 - rmse(est, truth) / rmse(zs, truth)):.0f}% better)")
out(f"by thirds: filter {fmt(thirds(est, truth))} m, sensor {fmt(thirds(zs, truth))} m")
xs, gains, ys, Ss = run(zs, F2, H2, Q_repo, 25.0, np.eye(2))
out(f"final gain ({gains[-1][0]:.4f}, {gains[-1][1]:.4f}); the general filter matches the class to "
    f"{np.abs(xs[:, 0] - est).max():.1e} m")
# The page runs on the data file, which holds the CSV rounded to 4 decimals, and compares with the repository's
# estimates, also rounded to 4 decimals.
est_page, vel_page = run_class(np.round(zs, 4), dt, Q_repo, 25)
out(f"page against repository: max |position| difference {np.abs(est_page - np.round(est, 4)).max():.2e} m, "
    f"velocity {np.abs(vel_page - np.round(vel, 4)).max():.2e} m/s (the rounding of the data file)")
xj, *_ = run(zs, F2, H2, Q_repo, 25.0, np.eye(2), joseph=True)
out(f"Joseph form against P - KHP, float64: max |estimate difference| {np.abs(xj[:, 0] - xs[:, 0]).max():.1e} m "
    f"(relative to 15 m: {np.abs(xj[:, 0] - xs[:, 0]).max() / 15:.0e})")

section("3. The gain's limits (Part 2, question 9)")
for R in (1e-2, 1e-8):
    _, g, _, _ = run(zs, F2, H2, Q_repo, R, np.eye(2))
    out(f"R = {R:g}: final gain ({g[-1][0]:.8f}, {g[-1][1]:.4f})")
# As R -> 0 the gain tends to (1, P_pv/P_pp) of the predicted covariance: compute that ratio's limit directly.
P = np.eye(2)
for _ in range(5000):
    Pp = F2 @ P @ F2.T + Q_repo
    P = np.array([[0.0, 0.0], [0.0, Pp[1, 1] - Pp[0, 1] ** 2 / Pp[0, 0]]])     # update with R = 0
out(f"R = 0 exactly: P_pv/P_pp of the predicted covariance tends to {Pp[0, 1] / Pp[0, 0]:.4f}")

section("4. Q in units of acceleration (Part 3, question 4's piecewise-constant form)")
out(f"Q_vv = sigma_a^2 dt^2 = 3 means sigma_a = sqrt(3)/dt = {np.sqrt(3) / dt:.0f} m/s^2")
out(f"Q_pp = sigma_a^2 dt^4 / 4 = 1 would need sigma_a = 2/dt^2 = {2 / dt ** 2:.0f} m/s^2")

section("5. The sweep over the scale of Q (Part 3)")
grid = [i / 20 for i in range(-40, 61)]                         # log10 scale from -2 to 3 in steps of 0.05, as the page
sweep = {e: run_class(zs, dt, 10 ** e * Q_repo, 25) for e in grid}
errs = {e: rmse(v[0], truth) for e, v in sweep.items()}
best = min(errs, key=errs.get)
out(f"scale 1: {errs[0.0]:.4f} m; scale 0.01: {errs[-2.0]:.2f} m; scale 1000: {errs[3.0]:.2f} m (sensor {rmse(zs, truth):.2f})")
out(f"grid minimum {errs[best]:.4f} m at scale 10^{best:.2f} = {10 ** best:.2f}")
fine = {s: rmse(run_class(zs, dt, s * Q_repo, 25)[0], truth) for s in np.round(np.arange(4.0, 12.01, 0.1), 1)}
fbest = min(fine, key=fine.get)
out(f"fine minimum {fine[fbest]:.5f} m at scale {fbest}; scale 7: {fine[7.0]:.5f} m (page: {fine[7.0]:.2f}); "
    f"scale 10: {fine[10.0]:.5f} m (page: {fine[10.0]:.2f})")
out(f"by thirds at scale {fbest}: {fmt(thirds(run_class(zs, dt, fbest * Q_repo, 25)[0], truth))} m")
vel_errs = [rmse(v[1], v_true) for v in sweep.values()]
out(f"velocity RMSE over every scale from 0.01 to 1000: {min(vel_errs):.1f} to {max(vel_errs):.1f} m/s")

section("6. The velocity estimate (Part 4)")
out(f"velocity RMSE {rmse(vel, v_true):.1f} m/s against {rmse(0 * t, v_true):.1f} for always answering 0")
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
out(f"correlation with the true velocity: overall {corr(vel, v_true):.2f}; by thirds "
    f"{fmt([corr(vel[i:j], v_true[i:j]) for i, j in THIRDS])}")
out(f"size: the estimate's RMS {rmse(vel, 0 * t):.1f} m/s, peak {np.abs(vel).max():.0f}; the truth's RMS {rmse(v_true, 0 * t):.1f}, "
    f"peak {np.abs(v_true).max():.0f}; by thirds, estimate {fmt([rmse(vel[i:j], 0 * t[i:j]) for i, j in THIRDS], 1)} "
    f"against truth {fmt([rmse(v_true[i:j], 0 * t[i:j]) for i, j in THIRDS], 1)}")


def q_pc(sa):
    """Question 4's piecewise-constant acceleration Q, sigma_a in m/s^2."""
    return sa ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])


def q_int(q):
    """Chapter 02's integrated white-noise Q, intensity q (sigma_a = sqrt(q) in m/s^(3/2))."""
    return q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])


sas = np.round(10 ** np.arange(2.5, 4.51, 0.01), 0)
pc = {sa: run_class(zs, dt, q_pc(sa), 25) for sa in sas}
pc_err = {sa: rmse(v[0], truth) for sa, v in pc.items()}
sa_best = min(pc_err, key=pc_err.get)
out(f"question 4's Q: best sigma_a {sa_best:.0f} m/s^2 -> position {pc_err[sa_best]:.2f} m "
    f"(thirds {fmt(thirds(pc[sa_best][0], truth))}), velocity {rmse(pc[sa_best][1], v_true):.1f} m/s")
iq = {s: run_class(zs, dt, q_int(s ** 2), 25) for s in np.arange(300, 801, 5)}
iq_err = {s: rmse(v[0], truth) for s, v in iq.items()}
s_best = min(iq_err, key=iq_err.get)
out(f"chapter 02's integrated Q: best sigma_a {s_best} m/s^1.5 -> position {iq_err[s_best]:.3f} m, "
    f"velocity {rmse(iq[s_best][1], v_true):.1f} m/s")

section("7. The lag (Part 4 and its chart)")
w = 50
starts = list(range(0, N - w + 1, 10))
tt = [t[i + w // 2] for i in starts]
errK = [rmse(est[i:i + w], truth[i:i + w]) for i in starts]
errM = [rmse(zs[i:i + w], truth[i:i + w]) for i in starts]
first = next(tt[k] for k in range(len(tt)) if errK[k] > errM[k] and k > 5)
sustained = next(tt[k] for k in range(len(tt)) if all(errK[j] > errM[j] for j in range(k, len(tt))))
out(f"half-second windows: first crossing at {first:.2f} s, worse in every window from {sustained:.2f} s")
phi_sensor = 2 * np.degrees(np.arcsin(5 / (15 * np.sqrt(2))))
out(f"a sine of amplitude 15 m is as far from the truth as the sensor's 5 m at a lag of {phi_sensor:.1f} degrees, "
    f"1/{360 / phi_sensor:.1f} of a period ({phi_sensor / 360 / 3.18:.3f} s at 3.18 Hz)")
for deg in (60, 90):
    phi = np.radians(deg)
    out(f"a sine of amplitude A lagged by {deg} degrees: RMS error {np.sqrt(2) * np.sin(phi / 2):.4f} A; "
        f"the estimate 0: {1 / np.sqrt(2):.4f} A")

section("8. Exercise 2: an acceleration state")
F3 = np.array([[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, 1]])
H3 = np.array([1.0, 0.0, 0.0])
ca = {e: run(zs, F3, H3, np.diag([0.0, 0.0, 10.0 ** e]), 25.0, np.eye(3))[0][:, 0] for e in np.arange(-2, 8.01, 0.5)}
ca_err = {e: rmse(v, truth) for e, v in ca.items()}
e_best = min(ca_err, key=ca_err.get)
out(f"best q_a = 10^{e_best:.1f}: {ca_err[e_best]:.2f} m, thirds {fmt(thirds(ca[e_best], truth))}")
out(f"q_a = 10^3: {ca_err[3.0]:.2f} m, thirds {fmt(thirds(ca[3.0], truth))}")

section("9. Exercise 3: the innovations")
for scale in (1, 10):
    _, _, y, S = run(zs, F2, H2, scale * Q_repo, 25.0, np.eye(2))
    nis = y ** 2 / S
    out(f"Q x {scale}: var(y)/mean(S) {y.var() / S.mean():.2f}, by thirds {fmt([y[i:j].var() / S[i:j].mean() for i, j in THIRDS])}; "
        f"the mean of y^2/S {nis.mean():.2f}, by thirds {fmt([nis[i:j].mean() for i, j in THIRDS])}")

section("10. What the download prints (the repository after its October 2026 corrections)")
# data/simulate_data.py now: rng = default_rng(0), position 15 sin(t^2) + N(0, 5^2), velocity sensor N(0, 1).
rng = np.random.default_rng(0)
t_new = np.arange(0, 10, 0.01)
truth_new = 15 * np.sin(t_new ** 2)
z_new = truth_new + rng.normal(0, 5, size=t_new.shape)
for scale in (10, 1):
    xn, *_ = run(z_new, F2, H2, scale * Q_repo, 25.0, np.eye(2), joseph=True)
    out(f"regenerated CSV, Q = {scale} x diag(1, 3), solve and Joseph form: measurements {rmse(z_new, truth_new):.3f} m, "
        f"Kalman {rmse(xn[:, 0], truth_new):.3f} m")


section("11. Without the ground truth: the innovations' likelihood (Part 3 and its chart)")


def run_batch(zs, Qs, Rs):
    """The class's filter for B pairs (Q, R) at once, vectorised over the pairs: the same arithmetic as run_class
    written out for a 2-state, 1-measurement filter. Returns the estimates, velocities, innovations y and their S."""
    Qs, Rs = np.asarray(Qs, float), np.asarray(Rs, float)
    B = len(Rs)
    x0, x1, P00, P01, P10, P11 = np.zeros(B), np.zeros(B), np.ones(B), np.zeros(B), np.zeros(B), np.ones(B)
    est, vel, ys, Ss = (np.empty((B, len(zs))) for _ in range(4))
    for k, z in enumerate(zs):
        x0, x1 = x0 + dt * x1, x1                                                       # predict
        P00, P01, P10, P11 = (P00 + dt * (P01 + P10) + dt * dt * P11 + Qs[:, 0, 0], P01 + dt * P11 + Qs[:, 0, 1],
                              P10 + dt * P11 + Qs[:, 1, 0], P11 + Qs[:, 1, 1])
        S = P00 + Rs                                                                    # update
        k0, k1, y = P00 / S, P10 / S, z - x0
        x0, x1 = x0 + k0 * y, x1 + k1 * y
        P00, P01, P10, P11 = P00 - k0 * P00, P01 - k0 * P01, P10 - k1 * P00, P11 - k1 * P01
        est[:, k], vel[:, k], ys[:, k], Ss[:, k] = x0, x1, y, S
    return est, vel, ys, Ss


def nll(ys, Ss):
    """The negative log-likelihood per step of the measurements under the filter's predictions (J3, Part 7)."""
    return np.mean(0.5 * (np.log(2 * np.pi) + np.log(Ss) + ys ** 2 / Ss), axis=-1)


b_est, *_ = run_batch(zs, [Q_repo], [25.0])
out(f"the vectorised filter matches the class to {np.abs(b_est[0] - est).max():.1e} m")
# The chart's grid, as the page computes it live: 101 scales, R = 25.
g_est, g_vel, g_y, g_S = run_batch(zs, [10 ** e * Q_repo for e in grid], np.full(len(grid), 25.0))
g_nll = nll(g_y, g_S)
gi = int(np.argmin(g_nll))
out(f"chart grid ({len(grid)} runs of the filter): likelihood smallest at scale 10^{grid[gi]:.2f} = {10 ** grid[gi]:.1f} (RMSE {rmse(g_est[gi], truth):.2f} m), "
    f"RMSE smallest at 10^{best:.2f} = {10 ** best:.1f}; NLL per step from {g_nll.min():.2f} to {g_nll.max():.2f}, "
    f"{g_nll[grid.index(0.0)]:.2f} at scale 1")
# The same refinement as the RMSE's in section 5: scales 4.0 to 12.0 in steps of 0.1.
f_scales = np.round(np.arange(4.0, 12.01, 0.1), 1)
f_est, f_vel, f_y, f_S = run_batch(zs, [s_ * Q_repo for s_ in f_scales], np.full(len(f_scales), 25.0))
fi = int(np.argmin(nll(f_y, f_S)))
out(f"fine: likelihood smallest at scale {f_scales[fi]}: RMSE {rmse(f_est[fi], truth):.4f} m (page: "
    f"{rmse(f_est[fi], truth):.2f}), against the RMSE's best {fine[fbest]:.4f} m at {fbest}: "
    f"{1000 * (rmse(f_est[fi], truth) - fine[fbest]):.0f} mm worse; mean y^2/S there {np.mean(f_y[fi] ** 2 / f_S[fi]):.2f}")
# Chapter 02's integrated white-noise Q(sigma_a) with R = sigma_z^2, both free: a grid on (log sigma_a, log sigma_z),
# then three finer grids around the best point.
la, lz = np.log(np.geomspace(100, 3000, 60)), np.log(np.geomspace(2, 10, 60))
for width in (None, 0.1, 0.01, 0.001):
    if width is not None:
        la, lz = np.linspace(a_best - width, a_best + width, 41), np.linspace(z_best - width, z_best + width, 41)
    A_, Z_ = (m.ravel() for m in np.meshgrid(la, lz))
    i_est, i_vel, i_y, i_S = run_batch(zs, [q_int(np.exp(2 * a)) for a in A_], np.exp(2 * Z_))
    i_nll = nll(i_y, i_S)
    ii = int(np.argmin(i_nll))
    a_best, z_best = A_[ii], Z_[ii]
sa_ml, sz_ml = np.exp(a_best), np.exp(z_best)
out(f"chapter 02's Q(sigma_a), sigma_a and sigma_z free: likelihood smallest at sigma_a = {sa_ml:.2f}, "
    f"sigma_z = {sz_ml:.4f} (true 5); NLL per step {i_nll[ii]:.6f}")
out(f"  there: position RMSE {rmse(i_est[ii], truth):.3f} m (thirds {fmt(thirds(i_est[ii], truth))}), velocity "
    f"{rmse(i_vel[ii], v_true):.1f} m/s, mean y^2/S {np.mean(i_y[ii] ** 2 / i_S[ii]):.3f}")
out(f"  with the truth, the same Q's best (section 6): sigma_a {s_best} -> {iq_err[s_best]:.3f} m")

section("12. The smoother (Part 4 and the hero's 'smoother (offline)')")


def filter_and_smoother(zs, Q, R=25.0):
    """The class's filter, keeping its predictions, then the Rauch-Tung-Striebel backward pass (J5, Part 7)."""
    x, P = np.zeros(2), np.eye(2)
    xf, Pf, xp, Pp = [], [], [], []
    for z in zs:
        x, P = F2 @ x, F2 @ P @ F2.T + Q
        xp.append(x); Pp.append(P)
        K = P @ H2 / (H2 @ P @ H2 + R)
        x, P = x + K * (z - H2 @ x), P - np.outer(K, H2 @ P)
        xf.append(x); Pf.append(P)
    xs = list(xf)                                     # the last filtered estimate is already smoothed
    for k in range(len(zs) - 2, -1, -1):
        G = Pf[k] @ F2.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + G @ (xs[k + 1] - xp[k + 1])
    return np.array(xf), np.array(xs)


for name, Qm in (("the repository's Q", Q_repo), ("Q x 10", 10 * Q_repo),
                 (f"question 4's Q, sigma_a = {sa_best:.0f}", q_pc(sa_best))):
    xf_, xs_ = filter_and_smoother(zs, Qm)
    out(f"{name}: position {rmse(xf_[:, 0], truth):.3f} -> {rmse(xs_[:, 0], truth):.3f} m; by thirds "
        f"{fmt(thirds(xf_[:, 0], truth))} -> {fmt(thirds(xs_[:, 0], truth))} m; velocity "
        f"{rmse(xf_[:, 1], v_true):.2f} -> {rmse(xs_[:, 1], v_true):.2f} m/s")
out(f"filter here against the class: {np.abs(filter_and_smoother(zs, Q_repo)[0][:, 0] - est).max():.1e} m; "
    f"sensor by thirds {fmt(thirds(zs, truth))} m; answering 0 for the velocity {rmse(0 * t, v_true):.1f} m/s")

section("13. Optional, in JAX: Part 2's box and Adam on the likelihood (Part 3)")
try:
    import jax
    import jax.numpy as jnp
    import optax
except ImportError:
    jax = None
    out("JAX or Optax is not installed: sections 13 and 14 skipped (pip install jax optax)")

if jax is not None:
    out("jax", jax.__version__, "optax", optax.__version__)

    def run_tracker(zs, F, H, Q, R, x0, P0):            # Part 2's box, verbatim
        def step(carry, z):                    # pure: the state comes in and goes out
            x, P = carry
            x, P = F @ x, F @ P @ F.T + Q          # predict
            S = H @ P @ H.T + R                    # update
            K = P @ H.T @ jnp.linalg.inv(S)
            x = x + K @ (z - H @ x)
            P = P - K @ H @ P
            return (x, P), x                   # the new carry, and one output
        _, xs = jax.lax.scan(step, (x0, P0), zs)
        return xs

    def run_tracker_joseph(zs, F, H, Q, R, x0, P0):     # the same, with the repository's later Joseph form
        def step(carry, z):
            x, P = carry
            x, P = F @ x, F @ P @ F.T + Q
            S = H @ P @ H.T + R
            K = P @ H.T @ jnp.linalg.inv(S)
            x = x + K @ (z - H @ x)
            A = jnp.eye(2, dtype=P.dtype) - K @ H
            P = A @ P @ A.T + K @ R @ K.T
            return (x, P), (x, jnp.linalg.eigvalsh(P)[0])
        _, out_ = jax.lax.scan(step, (x0, P0), zs)
        return out_

    def jax_model():
        F = jnp.array([[1.0, dt], [0.0, 1.0]]); H = jnp.array([[1.0, 0.0]])
        Q, R = jnp.diag(jnp.array([1.0, 3.0])), jnp.array([[25.0]])
        return F, H, Q, R, jnp.zeros(2), jnp.eye(2)

    # JAX's default, float32 (J1, Part 4).
    m32 = jax_model()
    x32 = np.asarray(run_tracker(jnp.asarray(zs)[:, None], *m32), np.float64)
    xj32, mins32 = run_tracker_joseph(jnp.asarray(zs)[:, None], *m32)
    xj32 = np.asarray(xj32, np.float64)
    out(f"float32 ({m32[0].dtype}): the box against the float64 class {np.abs(x32[:, 0] - est).max():.1e} m; "
        f"Joseph against textbook {np.abs(xj32[:, 0] - x32[:, 0]).max():.1e} m; "
        f"smallest eigenvalue of P {float(mins32.min()):.2f}")
    # float64, switched on for the rest of the section.
    jax.config.update("jax_enable_x64", True)
    m64 = jax_model()
    x64 = np.asarray(run_tracker(jnp.asarray(zs, jnp.float64)[:, None], *m64))
    out(f"float64 ({m64[0].dtype}): the box against the class {np.abs(x64[:, 0] - est).max():.1e} m")

    # Adam on the likelihood, theta = (log sigma_a, log sigma_z), chapter 02's Q(sigma_a), as J6, Part 6.
    def nll_jax(theta, zs):
        sa, sz = jnp.exp(theta[0]), jnp.exp(theta[1])
        Q = sa ** 2 * jnp.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        F = jnp.array([[1.0, dt], [0.0, 1.0]])

        def step(carry, z):
            x, P = carry
            x, P = F @ x, F @ P @ F.T + Q
            S = P[0, 0] + sz ** 2
            K, y = P[:, 0] / S, z - x[0]
            return (x + K * y, P - jnp.outer(K, P[0, :])), 0.5 * (jnp.log(2 * jnp.pi) + jnp.log(S) + y ** 2 / S)
        _, terms = jax.lax.scan(step, (jnp.zeros(2), jnp.eye(2)), zs)
        return jnp.mean(terms)

    z64 = jnp.asarray(zs, jnp.float64)
    out(f"the JAX likelihood at the numpy minimum: {float(nll_jax(jnp.log(jnp.array([sa_ml, sz_ml])), z64)):.6f}")
    optimizer = optax.adam(0.05)
    theta = jnp.log(jnp.array([100.0, 1.0]))
    opt_state = optimizer.init(theta)

    @jax.jit
    def update(theta, opt_state):
        loss, grads = jax.value_and_grad(nll_jax)(theta, z64)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(theta, updates), opt_state, loss

    path = []
    for _ in range(1000):
        theta, opt_state, _ = update(theta, opt_state)
        path.append(np.exp(np.asarray(theta)))
    path = np.array(path)
    close = np.all(np.abs(path / np.array([sa_ml, sz_ml]) - 1) < 1e-3, axis=1)
    settled = next(k for k in range(len(close)) if close[k:].all()) + 1
    out(f"Adam(0.05) from sigma_a = 100, sigma_z = 1: within 0.1% of the minimum, and stays, from step {settled}; "
        f"after 300 steps ({path[299][0]:.1f}, {path[299][1]:.3f}); after 1000 ({path[-1][0]:.2f}, {path[-1][1]:.4f})")
    hess = np.asarray(jax.hessian(nll_jax)(theta, z64))
    out(f"Hessian of the NLL per step at the minimum, in (log sigma_a, log sigma_z): eigenvalues "
        f"{fmt(np.linalg.eigvalsh(hess), 3)}")

    section("14. Exercise 5: fit the noise with Optax (the solution's 300 Adam steps, float64)")
    # The solution's loop is section 13's: Adam(0.05) on nll_jax from (100, 1); path[299] is theta after 300 steps.
    sa300, sz300 = path[299]
    e300, _, y300, S300 = run_batch(zs, [q_int(sa300 ** 2)], [sz300 ** 2])
    out(f"after 300 steps: sigma_a {sa300:.1f}, sigma_z {sz300:.3f}, NLL per step "
        f"{float(nll_jax(jnp.log(jnp.asarray(path[299])), z64)):.4f} (numpy filter: {nll(y300, S300)[0]:.4f}); "
        f"position RMSE {rmse(e300[0], truth):.2f} m ({rmse(e300[0], truth):.4f})")
