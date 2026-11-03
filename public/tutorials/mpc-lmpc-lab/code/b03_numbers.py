"""Reproduce every number quoted on lab B3 of "Advanced MPC and Learning MPC",
"Frenet coordinates and minimum-time racing" (b03-frenet-coordinates-and-minimum-time-racing.html).

Usage:  python b03_numbers.py
Needs numpy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).

Everything comes from the package's racing module: the demo track (a closed-form curve sampled uniformly in
arc length), the Frenet rates, the periodic minimum-time speed planner in squared speed, and the friction-circle
diagnostic. Takes about a minute (the 1920-sample grid is the slow part).
"""
from __future__ import annotations

import numpy as np

from advanced_mpc_lmpc.racing import (G, build_demo_track, frenet_rates, friction_circle_utilization,
                                      lateral_speed_squared_cap, periodic_minimum_time_speed_profile)

np.set_printoptions(precision=4, suppress=True)
W = 3.0


def section(title):
    print(f"\n=== {title}")


def plan(track, **kw):
    """The package's closed-lap planner with the lab's defaults (mu 1, vmax 35, a in [-6, 3])."""
    return periodic_minimum_time_speed_profile(track, **{**dict(mu=1.0, vmax=35.0, amin=-6.0, amax=3.0), **kw})


def braking_zone(track, sol, amin):
    """Where full braking before the tightest corner starts: distance upstream and entry speed."""
    n, a, apex = len(track.s), sol["a"], int(np.argmax(np.abs(track.curvature)))
    k = apex
    while abs(a[(k - 1) % n] - amin) > 1e-6:
        k -= 1
    end = k
    while abs(a[(k - 1) % n] - amin) <= 1e-6:
        k -= 1
    start = k % n
    return dict(start=start, end=end % n, apex=apex, upstream=track.s[apex] - track.s[start])


T240 = build_demo_track(240, half_width=W)
APEX = int(np.argmax(np.abs(T240.curvature)))

# --------------------------------------------------------------------------------------------------------------
section("Part 2: the synthetic track (build_demo_track with 240 samples)")
print(f"length {T240.length:.3f} m; 240 samples, ds = {T240.length / 240:.3f} m")
print(f"curvature range [{T240.curvature.min():.5f}, {T240.curvature.max():.5f}] 1/m")
print(f"tightest sample s = {T240.s[APEX]:.3f} m, K = {T240.curvature[APEX]:.6f}, radius {1 / abs(T240.curvature[APEX]):.3f} m")
half = 240 // 2
mirror = half + int(np.argmax(np.abs(T240.curvature[half:])))
print(f"its mirror image is at s = {T240.s[mirror]:.1f}")
print(f"Frenet margin min(1 - |K| W) = {np.min(1 - np.abs(T240.curvature) * W):.4f}")
print(f"first sample: ({T240.xy[0, 0]:.1f}, {T240.xy[0, 1]:.1f}), driven counter-clockwise")

# --------------------------------------------------------------------------------------------------------------
section("Part 3: the Frenet rates at the example state (v_x 12, v_y 0.4, r 0.15, e_psi 0.05, K 0.02, e_y 0.5)")
rates = frenet_rates(12.0, 0.4, 0.15, 0.05, 0.02, 0.5)
print({k: round(float(v), 4) for k, v in rates.items()})

# --------------------------------------------------------------------------------------------------------------
section("Part 4: the curvature cap and the naive profile")
print(f"cap at the tightest corner with mu = 1: sqrt(g / |K|) = {np.sqrt(G / abs(T240.curvature[APEX])):.3f} m/s")
zmax = lateral_speed_squared_cap(T240.curvature, mu=1.0, vmax=35.0)
a_naive = (np.roll(zmax, -1) - zmax) / (2 * T240.ds)
print(f"naive profile z = zmax: implied a in [{a_naive.min():.1f}, {a_naive.max():.1f}] m/s^2, "
      f"{int(np.sum((a_naive > 3 + 1e-8) | (a_naive < -6 - 1e-8)))} of 240 segments violate [-6, 3]")

# --------------------------------------------------------------------------------------------------------------
section("Part 5: the exact segment time")
v0, v1, ds = 10.0, 14.0, 30.0
a = (v1**2 - v0**2) / (2 * ds)
print(f"from 10 to 14 m/s over 30 m: a = {a:.2f} m/s^2, time (v1 - v0)/a = {(v1 - v0) / a:.2f} s, formula 2 ds/(v0+v1) = {2 * ds / (v0 + v1):.2f} s")

# --------------------------------------------------------------------------------------------------------------
section("Part 6: the planner at the lab's setting (240 samples, mu 1, a in [-6, 3], vmax 35)")
sol = plan(T240)
print(f"success {sol['success']}, sweeps {sol['iterations']}, lap time {sol['time']:.4f} s, speeds {sol['v'].min():.3f} to {sol['v'].max():.3f} m/s")
print(f"acceleration range [{sol['a'].min():.1f}, {sol['a'].max():.1f}]; segments at a_max {int(np.sum(np.abs(sol['a'] - 3) < 1e-6))}, "
      f"at a_min {int(np.sum(np.abs(sol['a'] + 6) < 1e-6))}, samples on the cap {int(np.sum(np.abs(sol['z'] - sol['zmax']) < 1e-6))}")
print(f"top-speed limit active: {bool(np.any(np.abs(sol['z'] - 35**2) < 1e-6))}; elapsed time at the finish line {sol['segment_time'].sum():.4f} s")
bz = braking_zone(T240, sol, -6)
print(f"full braking before the apex starts {bz['upstream']:.1f} m upstream at {sol['v'][bz['start']]:.1f} m/s; apex speed {sol['v'][APEX]:.3f}")

# --------------------------------------------------------------------------------------------------------------
section("Part 7: friction, authority and grid")
laps = {mu: plan(T240, mu=mu)["time"] for mu in (0.6, 0.8, 1.0, 1.2)}
print("lap times by friction:", {k: round(v, 4) for k, v in laps.items()})
print(f"weaker braking (a_min = -3.5): +{plan(T240, amin=-3.5)['time'] - sol['time']:.2f} s; "
      f"weaker acceleration (a_max = 1.5): +{plan(T240, amax=1.5)['time'] - sol['time']:.2f} s")
grid = {}
for n in (60, 120, 240, 480, 960, 1920):
    t = build_demo_track(n, half_width=W); grid[n] = (plan(t)["time"], np.abs(t.curvature).max())
print("grid: " + ", ".join(f"{n}: {T:.4f} s (max |K| {mk:.4f})" for n, (T, mk) in grid.items()))
Ts = [grid[n][0] for n in (60, 120, 240, 480, 960, 1920)]
print("increments:", [round(Ts[i + 1] - Ts[i], 4) for i in range(5)])
tail = Ts[-1] + (Ts[-1] - Ts[-2])        # geometric tail with ratio 1/2
print(f"extrapolated limit about {tail:.3f} s; the 240-sample time is {tail - grid[240][0]:.3f} s optimistic, the 60-sample one {tail - grid[60][0]:.2f} s")
v12, v06 = plan(T240, mu=1.2)["v"], plan(T240, mu=0.6)["v"]
d = v12 - v06
print(f"mu 1.2 against 0.6: largest speed gain {d.max():.1f} m/s at s = {T240.s[d.argmax()]:.1f}, smallest {d.min():.1f} at s = {T240.s[d.argmin()]:.0f}")
s06 = plan(T240, mu=0.6)
print(f"with mu = 0.6 the cap is active on {100 * np.mean(np.abs(s06['z'] - s06['zmax']) < 1e-6):.0f} percent of the samples")

# --------------------------------------------------------------------------------------------------------------
section("Part 8: the friction-circle diagnostic")
fu = friction_circle_utilization(sol["z"], sol["a"], T240.curvature, mu=1.0)
print(f"largest utilisation {fu.max():.4f}, {int(np.sum(fu > 1 + 1e-8))} of 240 segments exceed one")
print(f"at the apex the decoupled planner still accelerates at {sol['a'][APEX]:.2f} m/s^2")

# --------------------------------------------------------------------------------------------------------------
section("Exercise 8: braking distance in spatial form")
z0, zf = 20.91**2, 8.71**2
print(f"({z0:.1f} - {zf:.1f}) / 12 = {(z0 - zf) / 12:.1f} m")

section("Exercise 10: braking sweep before the tightest corner")
for amin in (-3, -4.5, -6, -9):
    s_ = plan(T240, amin=amin); b = braking_zone(T240, s_, amin)
    print(f"a_min = {amin:5.1f}: braking starts {b['upstream']:.1f} m upstream from {s_['v'][b['start']]:.1f} m/s, lap {s_['time']:.2f} s, apex {s_['v'][APEX]:.3f}")

section("Exercise 13: the two passes written by hand against the package's planner")
ds = T240.ds; n = 240; z = lateral_speed_squared_cap(T240.curvature, mu=1.0, vmax=35.0)
for sweep in range(1, 101):
    prev = z.copy()
    for i in range(n):
        j = (i + 1) % n; z[j] = min(z[j], z[i] + 2 * 3.0 * ds[i])
    for i in range(n - 1, -1, -1):
        j = (i + 1) % n; z[i] = min(z[i], z[j] + 2 * 6.0 * ds[i])
    if np.max(np.abs(z - prev)) <= 1e-9:
        break
v = np.sqrt(z); T_hand = float(np.sum(2 * ds / (v + np.roll(v, -1))))
print(f"hand-written passes: {sweep} sweeps, lap {T_hand:.4f} s; package: {sol['iterations']} sweeps, {sol['time']:.4f} s; "
      f"profiles agree to {np.abs(v - sol['v']).max():.1e} m/s")
print(f"the package's default 400-sample track: lap {plan(build_demo_track(400, half_width=W))['time']:.4f} s")
