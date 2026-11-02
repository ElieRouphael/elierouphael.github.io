"""Reproduce every number quoted on lab B3 of "Advanced MPC and Learning MPC",
"Frenet coordinates and minimum-time racing" (b03-frenet-coordinates-and-minimum-time-racing.html).

Usage:  python b03_numbers.py
Needs numpy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).

The page runs on the lab notebook's closed-form track (X = 45 cos t + 5 cos 2t, Y = 30 sin t + 6 sin 3t) with
the notebook's closed-lap planner, both re-implemented below exactly as in the notebook and the browser demo.
The package's `synthetic_track` and `minimum_time_speed_profile` are a simpler open-loop version on a 300 m
loop; the last section runs them so that the code walk's numbers are reproducible too. Takes about a minute.
"""
from __future__ import annotations

import numpy as np

from advanced_mpc_lmpc.racing import minimum_time_speed_profile, synthetic_track

np.set_printoptions(precision=4, suppress=True)
G = 9.81
W = 3.0


def section(title):
    print(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
# the notebook's track: closed-form curve, arc length integrated on a dense grid, uniform sampling in arc length
def curve(t): return np.c_[45 * np.cos(t) + 5 * np.cos(2 * t), 30 * np.sin(t) + 6 * np.sin(3 * t)]
def curve_d1(t): return np.c_[-45 * np.sin(t) - 10 * np.sin(2 * t), 30 * np.cos(t) + 18 * np.cos(3 * t)]
def curve_d2(t): return np.c_[-45 * np.cos(t) - 20 * np.cos(2 * t), -30 * np.sin(t) - 54 * np.sin(3 * t)]


_T = np.linspace(0, 2 * np.pi, 200001)
_SP = np.linalg.norm(curve_d1(_T), axis=1)
_S = np.r_[0, np.cumsum(0.5 * (_SP[1:] + _SP[:-1]) * np.diff(_T))]
LENGTH = float(_S[-1])


def build_track(n, half_width=W):
    s = np.arange(n) * LENGTH / n
    t = np.interp(s, _S, _T)
    half = n // 2
    if n % 2 == 0:                                   # symmetric sampling, as the notebook
        t[half + 1:] = 2 * np.pi - t[n - np.arange(half + 1, n)]
    d1, d2 = curve_d1(t), curve_d2(t)
    K = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
    if n % 2 == 0:
        K[half + 1:] = K[n - np.arange(half + 1, n)]
    heading = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
    ds = np.diff(np.r_[s, LENGTH])
    return dict(n=n, s=s, xy=curve(t), K=K, heading=heading, ds=ds, length=LENGTH, half_width=half_width)


def planner(track, mu=1.0, vmax=35.0, amin=-6.0, amax=3.0, tol=1e-9, max_iter=100):
    """The closed-lap forward/backward planner in z = v^2, repeated to a fixed point."""
    n, ds, K = track["n"], track["ds"], track["K"]
    zmax = np.minimum(vmax**2, mu * G / np.maximum(np.abs(K), 1e-9))
    z = zmax.copy()
    for it in range(1, max_iter + 1):
        prev = z.copy()
        for i in range(n):
            j = (i + 1) % n; z[j] = min(z[j], z[i] + 2 * amax * ds[i])
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n; z[i] = min(z[i], z[j] - 2 * amin * ds[i])
        if np.max(np.abs(z - prev)) <= tol:
            break
    zn = np.roll(z, -1)
    a = (zn - z) / (2 * ds)
    v = np.sqrt(z)
    seg = 2 * ds / (v + np.sqrt(zn))
    return dict(z=z, v=v, a=a, zmax=zmax, time=float(seg.sum()), iterations=it, seg=seg)


def braking_zone(track, sol, amin):
    """Where full braking before the tightest corner starts: distance upstream and entry speed."""
    n, a, apex = track["n"], sol["a"], int(np.argmax(np.abs(track["K"])))
    k = apex
    while abs(a[(k - 1) % n] - amin) > 1e-6:
        k -= 1
    end = k
    while abs(a[(k - 1) % n] - amin) <= 1e-6:
        k -= 1
    start = k % n
    return dict(start=start, end=end % n, apex=apex, upstream=track["s"][apex] - track["s"][start])


T240 = build_track(240)
APEX = int(np.argmax(np.abs(T240["K"])))

# --------------------------------------------------------------------------------------------------------------
section("Part 2: the synthetic track")
print(f"length {LENGTH:.3f} m; 240 samples, ds = {LENGTH / 240:.3f} m")
print(f"curvature range [{T240['K'].min():.5f}, {T240['K'].max():.5f}] 1/m")
print(f"tightest sample s = {T240['s'][APEX]:.3f} m, K = {T240['K'][APEX]:.6f}, radius {1 / abs(T240['K'][APEX]):.3f} m")
half = 240 // 2
mirror = half + int(np.argmax(np.abs(T240["K"][half:])))
print(f"its mirror image is at s = {T240['s'][mirror]:.1f}")
print(f"Frenet margin min(1 - |K| W) = {np.min(1 - np.abs(T240['K']) * W):.4f}")
print(f"first sample: ({T240['xy'][0, 0]:.1f}, {T240['xy'][0, 1]:.1f}), driven counter-clockwise")

# --------------------------------------------------------------------------------------------------------------
section("Part 3: the Frenet rates at the example state (v_x 12, v_y 0.4, r 0.15, e_psi 0.05, K 0.02, e_y 0.5)")
vx, vy, r, epsi, K, ey = 12.0, 0.4, 0.15, 0.05, 0.02, 0.5
vt = vx * np.cos(epsi) - vy * np.sin(epsi); den = 1 - K * ey; sdot = vt / den
eydot = vx * np.sin(epsi) + vy * np.cos(epsi); epsidot = r - K * sdot
print(f"v_t = {vt:.3f}, 1 - K e_y = {den:.2f}, sdot = {sdot:.3f}, eydot = {eydot:.3f}, epsidot = {epsidot:.4f}")
print(f"spatial: de_y/ds = {eydot / sdot:.4f}, de_psi/ds = {epsidot / sdot:.4f} rad/m, dt/ds = {1 / sdot:.4f} s/m")

# --------------------------------------------------------------------------------------------------------------
section("Part 4: the curvature cap and the naive profile")
print(f"cap at the tightest corner with mu = 1: sqrt(g / |K|) = {np.sqrt(G / abs(T240['K'][APEX])):.3f} m/s")
zmax = np.minimum(35**2, G / np.maximum(np.abs(T240["K"]), 1e-9))
a_naive = (np.roll(zmax, -1) - zmax) / (2 * T240["ds"])
print(f"naive profile z = zmax: implied a in [{a_naive.min():.1f}, {a_naive.max():.1f}] m/s^2, "
      f"{int(np.sum((a_naive > 3 + 1e-8) | (a_naive < -6 - 1e-8)))} of 240 segments violate [-6, 3]")

# --------------------------------------------------------------------------------------------------------------
section("Part 5: the exact segment time")
v0, v1, ds = 10.0, 14.0, 30.0
a = (v1**2 - v0**2) / (2 * ds)
print(f"from 10 to 14 m/s over 30 m: a = {a:.2f} m/s^2, time (v1 - v0)/a = {(v1 - v0) / a:.2f} s, formula 2 ds/(v0+v1) = {2 * ds / (v0 + v1):.2f} s")

# --------------------------------------------------------------------------------------------------------------
section("Part 6: the planner at the lab's setting (240 samples, mu 1, a in [-6, 3], vmax 35)")
sol = planner(T240)
print(f"sweeps {sol['iterations']}, lap time {sol['time']:.4f} s, speeds {sol['v'].min():.3f} to {sol['v'].max():.3f} m/s")
print(f"acceleration range [{sol['a'].min():.1f}, {sol['a'].max():.1f}]; segments at a_max {int(np.sum(np.abs(sol['a'] - 3) < 1e-6))}, "
      f"at a_min {int(np.sum(np.abs(sol['a'] + 6) < 1e-6))}, samples on the cap {int(np.sum(np.abs(sol['z'] - sol['zmax']) < 1e-6))}")
print(f"top-speed limit active: {bool(np.any(np.abs(sol['z'] - 35**2) < 1e-6))}; elapsed time at the finish line {sol['seg'].sum():.4f} s")
bz = braking_zone(T240, sol, -6)
print(f"full braking before the apex starts {bz['upstream']:.1f} m upstream at {sol['v'][bz['start']]:.1f} m/s; apex speed {sol['v'][APEX]:.3f}")

# --------------------------------------------------------------------------------------------------------------
section("Part 7: friction, authority and grid")
laps = {mu: planner(T240, mu=mu)["time"] for mu in (0.6, 0.8, 1.0, 1.2)}
print("lap times by friction:", {k: round(v, 4) for k, v in laps.items()})
print(f"weaker braking (a_min = -3.5): +{planner(T240, amin=-3.5)['time'] - sol['time']:.2f} s; "
      f"weaker acceleration (a_max = 1.5): +{planner(T240, amax=1.5)['time'] - sol['time']:.2f} s")
grid = {}
for n in (60, 120, 240, 480, 960, 1920):
    t = build_track(n); grid[n] = (planner(t)["time"], np.abs(t["K"]).max())
print("grid: " + ", ".join(f"{n}: {T:.4f} s (max |K| {mk:.4f})" for n, (T, mk) in grid.items()))
Ts = [grid[n][0] for n in (60, 120, 240, 480, 960, 1920)]
print("increments:", [round(Ts[i + 1] - Ts[i], 4) for i in range(5)])
tail = Ts[-1] + (Ts[-1] - Ts[-2])        # geometric tail with ratio 1/2
print(f"extrapolated limit about {tail:.3f} s; the 240-sample time is {tail - grid[240][0]:.3f} s optimistic, the 60-sample one {tail - grid[60][0]:.2f} s")
v12, v06 = planner(T240, mu=1.2)["v"], planner(T240, mu=0.6)["v"]
d = v12 - v06
print(f"mu 1.2 against 0.6: largest speed gain {d.max():.1f} m/s at s = {T240['s'][d.argmax()]:.1f}, smallest {d.min():.1f} at s = {T240['s'][d.argmin()]:.0f}")
s06 = planner(T240, mu=0.6)
print(f"with mu = 0.6 the cap is active on {100 * np.mean(np.abs(s06['z'] - s06['zmax']) < 1e-6):.0f} percent of the samples")

# --------------------------------------------------------------------------------------------------------------
section("Part 8: the friction-circle diagnostic")
fu = np.hypot(sol["a"], sol["z"] * T240["K"]) / G
print(f"largest utilisation {fu.max():.4f}, {int(np.sum(fu > 1 + 1e-8))} of 240 segments exceed one")
print(f"at the apex the decoupled planner still accelerates at {sol['a'][APEX]:.2f} m/s^2")

# --------------------------------------------------------------------------------------------------------------
section("Exercise 8: braking distance in spatial form")
z0, zf = 20.91**2, 8.71**2
print(f"({z0:.1f} - {zf:.1f}) / 12 = {(z0 - zf) / 12:.1f} m")

section("Exercise 10: braking sweep before the tightest corner")
for amin in (-3, -4.5, -6, -9):
    s_ = planner(T240, amin=amin); b = braking_zone(T240, s_, amin)
    print(f"a_min = {amin:5.1f}: braking starts {b['upstream']:.1f} m upstream from {s_['v'][b['start']]:.1f} m/s, lap {s_['time']:.2f} s, apex {s_['v'][APEX]:.3f}")

section("The code behind it: the package's open-loop planner on its 300 m curvature profile")
s300 = np.linspace(0, 300, 301); kap = synthetic_track(s300)
pk = minimum_time_speed_profile(s300, kap)
print(f"curvature range [{kap.min():.4f}, {kap.max():.4f}] 1/m, tightest radius {1 / np.abs(kap).max():.1f} m")
print(f"success {pk['success']}, time {pk['time']:.4f} s from v0 = 12 m/s, speeds {pk['v'].min():.2f} to {pk['v'].max():.2f}, "
      f"a in [{pk['a'].min():.2f}, {pk['a'].max():.2f}]")
