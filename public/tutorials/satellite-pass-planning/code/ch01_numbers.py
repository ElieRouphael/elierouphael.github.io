"""Chapter 01, Circular Orbits from Newton: every number quoted on the page.

Circular speed and period from satplan's constants, how they change with altitude, the geostationary
altitude with the package's mean Earth radius, and the three integrators of the page (explicit Euler,
symplectic Euler, RK4) run in NumPy exactly as interactive/ch01.js runs them in the browser, compared
with the closed-form circle of satplan.geometry.CircularOrbit.

Usage:  python code/ch01_numbers.py
satplan is imported from the course archive (satplan/ next to code/) or, failing that, from the
repository path in SATPLAN_REPO (default: the author's working copy).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent, Path(os.environ.get("SATPLAN_REPO", r"C:/Users/rouphaee/Documents/PostDoc_v2/satellite-pass-planning"))):
    if (cand / "satplan").is_dir():
        sys.path.insert(0, str(cand))
        break

from satplan.geometry import MU_EARTH, OMEGA_EARTH, R_EARTH, CircularOrbit  # noqa: E402

ALT = 550e3
R_EQ_WGS84 = 6_378_137.0  # m, equatorial radius, used by the textbook geostationary altitude


def section(title):
    print("\n== " + title)


# ---------------------------------------------------------------- circular orbit at 550 km
def circular(alt):
    a = R_EARTH + alt
    v = math.sqrt(MU_EARTH / a)
    T = 2 * math.pi * math.sqrt(a**3 / MU_EARTH)
    return a, v, T


section("circular orbit at 550 km")
a, v, T = circular(ALT)
orb = CircularOrbit(ALT, 0.0, 0.0, 0.0)
print(f"a = R_EARTH + 550 km = {a / 1e3:.1f} km")
print(f"gravity at the surface mu/R^2 = {MU_EARTH / R_EARTH**2:.3f} m/s^2; at 550 km mu/a^2 = {MU_EARTH / a**2:.3f} m/s^2 "
      f"({MU_EARTH / a**2 / (MU_EARTH / R_EARTH**2) * 100:.1f} % of the surface value)")
print(f"circular speed v = sqrt(mu/a) = {v:.1f} m/s = {v / 1e3:.3f} km/s = {v * 3.6:,.0f} km/h")
print(f"period T = 2 pi sqrt(a^3/mu) = {T:.1f} s = {T / 60:.2f} min (CircularOrbit.period_s = {orb.period_s:.3f} s)")
print(f"revolutions per day (86,400 s) = {86400 / T:.2f}")
print(f"angular rate n = 2 pi / T = {2 * math.pi / T:.4e} rad/s = {360 / T * 60:.3f} deg/min")
print(f"circumference 2 pi a = {2 * math.pi * a / 1e3:,.0f} km")
print(f"specific energy E = -mu/(2a) = {-MU_EARTH / (2 * a) / 1e6:.3f} MJ/kg")

section("altitude table (circular orbits)")
T_sid = 2 * math.pi / OMEGA_EARTH
a_geo = (MU_EARTH * (T_sid / (2 * math.pi)) ** 2) ** (1 / 3)
for alt_km in (200, 400, 550, 800, 1000, 2000, 5000, 10000, 20000):
    aa, vv, TT = circular(alt_km * 1e3)
    print(f"  {alt_km:>6} km: v = {vv / 1e3:.3f} km/s, T = {TT / 60:7.1f} min = {TT / 3600:6.2f} h, {86400 / TT:5.2f} rev/day")
aa, vv, TT = circular(a_geo - R_EARTH)
print(f"  geostationary: T = sidereal day 2 pi / OMEGA_EARTH = {T_sid:.1f} s = {T_sid / 3600:.4f} h")
print(f"  a_geo = (mu (T/2 pi)^2)^(1/3) = {a_geo / 1e3:,.1f} km, v = {vv / 1e3:.3f} km/s")
print(f"  altitude above the package's mean radius (6371 km): {(a_geo - R_EARTH) / 1e3:,.0f} km")
print(f"  altitude above the equatorial radius (6378.137 km), the textbook value: {(a_geo - R_EQ_WGS84) / 1e3:,.0f} km")
print(f"  difference: {(R_EQ_WGS84 - R_EARTH) / 1e3:.1f} km")
print(f"  slider range 200 km .. {(a_geo - R_EARTH) / 1e3:.0f} km")
# doubling the radius
print(f"  Kepler's third law: doubling a multiplies T by 2^1.5 = {2 ** 1.5:.3f}")


# ---------------------------------------------------------------- integrators
def accel(r):
    """Newton's gravity per unit mass: r'' = -mu r / |r|^3."""
    return -MU_EARTH * r / np.linalg.norm(r) ** 3


def step_euler(r, v, h):
    """Explicit Euler: both updates use the state at the start of the step."""
    return r + h * v, v + h * accel(r)


def step_symplectic(r, v, h):
    """Symplectic (semi-implicit) Euler: kick the velocity with the old position, then drift with the new velocity."""
    v = v + h * accel(r)
    return r + h * v, v


def step_rk4(r, v, h):
    """Classical fourth-order Runge-Kutta on the state (r, v)."""
    k1r, k1v = v, accel(r)
    k2r, k2v = v + h / 2 * k1v, accel(r + h / 2 * k1r)
    k3r, k3v = v + h / 2 * k2v, accel(r + h / 2 * k2r)
    k4r, k4v = v + h * k3v, accel(r + h * k3r)
    return r + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r), v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)


STEPPERS = {"explicit Euler": step_euler, "symplectic Euler": step_symplectic, "RK4": step_rk4}


def energy(r, v):
    """Specific orbital energy: kinetic plus potential, per kilogram."""
    return 0.5 * float(v @ v) - MU_EARTH / float(np.linalg.norm(r))


def kepler_reference(r0, k):
    """Closed-form radius and position of the planar orbit that starts at (r0, 0) with speed k * v_circ along +y."""
    vc = math.sqrt(MU_EARTH / r0)
    v0 = k * vc
    ae = 1.0 / (2.0 / r0 - v0 * v0 / MU_EARTH)
    e = abs(k * k - 1.0)
    n = math.sqrt(MU_EARTH / ae**3)
    start_at_perigee = k >= 1.0

    def at(t):
        if e < 1e-15:
            u = n * t
            return r0, (r0 * math.cos(u), r0 * math.sin(u))
        M = n * t + (0.0 if start_at_perigee else math.pi)  # mean anomaly from perigee
        E = M
        for _ in range(50):
            E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
        rr = ae * (1 - e * math.cos(E))
        nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
        ang = nu if start_at_perigee else nu - math.pi  # angle from the start direction (+x)
        return rr, (rr * math.cos(ang), rr * math.sin(ang))

    return ae, e, 2 * math.pi / n, at


def run(method, h, revs, k=1.0, alt=ALT):
    """Integrate `revs` periods of the reference orbit with step h. Returns final errors and extremes."""
    r0 = R_EARTH + alt
    ae, e, Te, ref = kepler_reference(r0, k)
    n_steps = math.ceil(revs * Te / h - 1e-9)
    r, w = np.array([r0, 0.0]), np.array([0.0, k * math.sqrt(MU_EARTH / r0)])
    E0 = energy(r, w)
    step = STEPPERS[method]
    max_dE = 0.0
    rmin = rmax = r0
    max_dr = 0.0
    hit = None
    for i in range(1, n_steps + 1):
        r, w = step(r, w, h)
        d = float(np.linalg.norm(r))
        if d < R_EARTH:
            hit = i * h
            break
        rmin, rmax = min(rmin, d), max(rmax, d)
        max_dE = max(max_dE, abs(energy(r, w) - E0) / abs(E0))
        max_dr = max(max_dr, abs(d - ref(i * h)[0]))
    t_end = (hit if hit is not None else n_steps * h)
    rr, pos = ref(t_end)
    return {
        "steps": n_steps, "t_end": t_end, "hit": hit,
        "dE_end": (energy(r, w) - E0) / abs(E0), "max_dE": max_dE,
        "dr_end": float(np.linalg.norm(r)) - rr, "max_dr": max_dr,
        "pos_err": math.hypot(r[0] - pos[0], r[1] - pos[1]),
        "rmin": rmin, "rmax": rmax, "evals": n_steps * (4 if method == "RK4" else 1),
    }


section("integrators, the page's default: h = 60 s, 5 revolutions, circular start at 550 km")
H0, N0 = 60.0, 5
print(f"steps per revolution T/h = {T / H0:.1f}; total steps ceil(5 T / h) = {math.ceil(N0 * T / H0)}; "
      f"angle per step n h = {360 * H0 / T:.2f} deg")
for m in STEPPERS:
    s = run(m, H0, N0)
    print(f"  {m:<17} dE/|E| end {s['dE_end']:+.3e}  max {s['max_dE']:.3e} | radius error end {s['dr_end'] / 1e3:+.3f} km  "
          f"max {s['max_dr'] / 1e3:.3f} km | position error vs closed form {s['pos_err'] / 1e3:,.3f} km | "
          f"r in [{s['rmin'] / 1e3:,.1f}, {s['rmax'] / 1e3:,.1f}] km | f evals {s['evals']}")

section("explicit Euler, first step (r perpendicular to v)")
hw = H0 * 2 * math.pi / T
print(f"n h = {hw:.5f} rad; |r1|/|r0| = sqrt(1 + (n h)^2) = {math.sqrt(1 + hw * hw):.6f}  -> +{(math.sqrt(1 + hw * hw) - 1) * a / 1e3:.2f} km in one step")
print(f"crude estimate of growth per revolution if every step were like the first: (1 + (nh)^2)^(T/2h) = "
      f"{(1 + hw * hw) ** (T / H0 / 2):.3f}")

section("one revolution, h = 60 s")
for m in STEPPERS:
    s = run(m, H0, 1)
    print(f"  {m:<17} dE/|E| end {s['dE_end']:+.3e} | radius error end {s['dr_end'] / 1e3:+.4f} km | position error {s['pos_err'] / 1e3:,.4f} km")

section("step-size study, 5 revolutions (dE/|E| at the end; position error vs closed form, km)")
for h in (1, 2, 5, 10, 20, 30, 60, 120, 240):
    row = []
    for m in STEPPERS:
        s = run(m, float(h), N0)
        row.append(f"{m}: {s['dE_end']:+.2e} / {s['pos_err'] / 1e3:,.4g} km")
    print(f"  h = {h:>3} s  " + "  |  ".join(row))

section("order of accuracy: position error after 5 revolutions, ratio when h is halved (60 -> 30 s)")
for m in STEPPERS:
    e60, e30 = run(m, 60.0, N0)["pos_err"], run(m, 30.0, N0)["pos_err"]
    print(f"  {m:<17} {e60 / 1e3:,.4g} km -> {e30 / 1e3:,.4g} km, ratio {e60 / e30:.2f} (log2 {math.log2(e60 / e30):.2f})")

section("symplectic Euler's energy band (largest |dE/E| over 5 revolutions) against the step")
for h in (120.0, 60.0, 30.0, 15.0):
    print(f"  h = {h:g} s: {run('symplectic Euler', h, N0)['max_dE']:.3e}")

section("symplectic Euler, long run: 50 revolutions at h = 60 s")
for m in ("symplectic Euler", "RK4"):
    s = run(m, 60.0, 50)
    print(f"  {m:<17} max |dE/E| {s['max_dE']:.3e}, dE/E end {s['dE_end']:+.3e}, max radius error {s['max_dr'] / 1e3:.3f} km, "
          f"position error {s['pos_err'] / 1e3:,.1f} km")

section("RK4 at h = 10 s, 5 revolutions")
s = run("RK4", 10.0, N0)
print(f"  dE/E end {s['dE_end']:+.3e}, position error {s['pos_err']:.4f} m, f evals {s['evals']}")

section("closed form cost: CircularOrbit.position_eci for one time")
p = orb.position_eci(5 * T)
print(f"  position_eci(5 T) = ({p[0] / 1e3:.3f}, {p[1] / 1e3:.3f}, {p[2] / 1e3:.3f}) km  (back at the start, exactly)")

section("non-circular start (speed k * v_circ at 550 km, direction perpendicular to r)")
for k in (0.95, 0.98, 1.02, 1.05, 1.10, 1.20):
    ae, e, Te, _ = kepler_reference(a, k)
    rp, ra = ae * (1 - e), ae * (1 + e)
    print(f"  k = {k:.2f}: a = {ae / 1e3:,.0f} km, e = {e:.4f}, perigee alt {(rp - R_EARTH) / 1e3:,.0f} km, "
          f"apogee alt {(ra - R_EARTH) / 1e3:,.0f} km, period {Te / 60:.1f} min")
k_min = math.sqrt(2 * R_EARTH / (a + R_EARTH))
print(f"  slowest start whose perigee clears the surface: k = sqrt(2 R / (a + R)) = {k_min:.4f} "
      f"(speed {k_min * v:.0f} m/s, {v - k_min * v:.0f} m/s below circular)")
print(f"  escape speed sqrt(2) v = {math.sqrt(2) * v:.0f} m/s")
s = run("RK4", 60.0, 5, k=1.05)
print(f"  RK4, k = 1.05, h = 60 s, 5 revs of the ellipse: dE/E end {s['dE_end']:+.3e}, position error {s['pos_err'] / 1e3:.3f} km")
s = run("explicit Euler", 60.0, 5, k=0.95)
print(f"  explicit Euler, k = 0.95: hits the surface at t = {s['hit']} s" if s["hit"] else "  explicit Euler, k = 0.95: no hit")
s = run("RK4", 60.0, 5, k=0.95)
print(f"  RK4, k = 0.95: hits the surface at t = {s['hit']} s = {s['hit'] / 60:.0f} min" if s["hit"] else "  RK4 k=0.95 no hit")

section("equal cost: 1,912 evaluations of the acceleration over 5 revolutions")
for m, h in (("RK4", 60.0), ("symplectic Euler", 15.0), ("explicit Euler", 15.0)):
    s = run(m, h, N0)
    print(f"  {m:<17} h = {h:g} s: {s['evals']} evaluations, max |dE/E| {s['max_dE']:.3e}, position error {s['pos_err'] / 1e3:,.3f} km")

section("small facts quoted in the text")
print(f"  steps for 5 revolutions at h = 1 s: {math.ceil(5 * T / 1.0 - 1e-9):,}")
print(f"  10 s grid over the small instance's 6 h: {len(np.arange(0, 6 * 3600 + 1, 10))} instants")
print(f"  start speed 1.05 v_c = {1.05 * v:.0f} m/s")
print(f"  mass ratio Earth / 420 t spacecraft = {MU_EARTH / 6.674e-11 / 4.2e5:.1e}")

section("exercise 1: a 90-minute circular orbit")
a90 = (MU_EARTH * (90 * 60) ** 2 / (4 * math.pi**2)) ** (1 / 3)
print(f"  altitude {(a90 - R_EARTH) / 1e3:.0f} km, speed {math.sqrt(MU_EARTH / a90):.0f} m/s, "
      f"{math.sqrt(MU_EARTH / a90) - v:.0f} m/s faster than at 550 km")

section("exercise 3: leapfrog (Stormer-Verlet), 5 revolutions")


def step_leapfrog(r, v, h):
    v = v + h / 2 * accel(r)
    r = r + h * v
    return r, v + h / 2 * accel(r)


STEPPERS["leapfrog"] = step_leapfrog
for h in (60.0, 30.0):
    s = run("leapfrog", h, N0)
    print(f"  h = {h:g} s: max |dE/E| {s['max_dE']:.3e}, position error {s['pos_err'] / 1e3:.3f} km")
s50 = run("leapfrog", 60.0, 50)
print(f"  50 revolutions at 60 s: max |dE/E| {s50['max_dE']:.3e}")

section("exercise 4: RK4 with equal-angle steps (2 pi / 100) against a fixed step, one revolution of an ellipse")
for kk in (1.05, 1.20):
    ae, e, Te, ref = kepler_reference(a, kk)
    theta = 2 * math.pi / 100
    r, w, t, n_ad = np.array([a, 0.0]), np.array([0.0, kk * v]), 0.0, 0
    while t < Te - 1e-9:
        h = theta * float(r @ r) / abs(r[0] * w[1] - r[1] * w[0])
        h = min(h, Te - t)
        r, w = step_rk4(r, w, h)
        t += h
        n_ad += 1
    err_ad = math.hypot(*(r - np.array(ref(Te)[1])))
    hf = Te / n_ad
    r, w = np.array([a, 0.0]), np.array([0.0, kk * v])
    for _ in range(n_ad):
        r, w = step_rk4(r, w, hf)
    err_fx = math.hypot(*(r - np.array(ref(Te)[1])))
    print(f"  k = {kk:.2f} (e = {e:.4f}): {n_ad} steps either way: equal-angle error {err_ad:,.2f} m, "
          f"fixed step ({hf:.1f} s) error {err_fx:,.2f} m, ratio fixed / equal-angle {err_fx / err_ad:.2f}")
