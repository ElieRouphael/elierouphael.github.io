"""Chapter 02, Placing an Orbit in Space: every number quoted on the page, and interactive/ch02-data.js.

Rotation matrices about x and z, their order, the expansion of R_z(RAAN) R_x(i) (a cos u, a sin u, 0)
checked against satplan.geometry.CircularOrbit.position_eci, and the constellations that
satplan.scenario.make_constellation builds for the benchmark's three sizes (seed 0), exported for the
page's 3D views.

Usage:  python code/ch02_numbers.py
satplan is imported from the course archive (satplan/ next to code/) or, failing that, from the
repository path in SATPLAN_REPO (default: the author's working copy).
"""
from __future__ import annotations

import json
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

from satplan.geometry import R_EARTH, CircularOrbit  # noqa: E402
from satplan.scenario import ScenarioConfig, make_constellation  # noqa: E402

OUT = HERE.parent / "interactive" / "ch02-data.js"
SIZES = {  # as scripts/run_benchmark.py
    "small": dict(n_sats=4, n_targets=120, horizon_h=6.0),
    "medium": dict(n_sats=8, n_targets=250, horizon_h=12.0),
    "large": dict(n_sats=12, n_targets=600, horizon_h=12.0),
}


def section(title):
    print("\n== " + title)


def Rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def km(v):
    return "(" + ", ".join(f"{x / 1e3:,.1f}" for x in v) + ") km"


# ---------------------------------------------------------------- the hero's default state
section("hero default: i = 97.6 deg, RAAN = 40 deg, u = 60 deg, 550 km")
I0, O0, U0 = 97.6, 40.0, 60.0
a = R_EARTH + 550e3
i, O, u = map(math.radians, (I0, O0, U0))
p_plane = np.array([a * math.cos(u), a * math.sin(u), 0.0])
p_tilt = Rx(i) @ p_plane
p_eci = Rz(O) @ p_tilt
pkg = CircularOrbit(550e3, I0, O0, U0).position_eci(0.0)
print(f"a = {a / 1e3:.1f} km")
print("step 1, in the orbital plane (a cos u, a sin u, 0):", km(p_plane))
print("step 2, tilted by i about x, R_x(i) p:", km(p_tilt))
print("step 3, turned by RAAN about z, R_z(RAAN) R_x(i) p:", km(p_eci))
print("CircularOrbit(550e3, 97.6, 40, 60).position_eci(0):", km(pkg), f" max difference {np.abs(pkg - p_eci).max():.2e} m")
wrong = Rx(i) @ Rz(O) @ p_plane
print("wrong order R_x(i) R_z(RAAN) p:", km(wrong), f" distance from the right answer {np.linalg.norm(wrong - p_eci) / 1e3:,.0f} km")
print(f"latitude of the satellite asin(z/a) = {math.degrees(math.asin(p_eci[2] / a)):.2f} deg")
print(f"right ascension atan2(y, x) = {math.degrees(math.atan2(p_eci[1], p_eci[0])):.2f} deg")

section("non-commutation, the matrices for i = 97.6, RAAN = 40")
np.set_printoptions(precision=4, suppress=True)
print("R_z(RAAN) R_x(i) =\n", Rz(O) @ Rx(i))
print("R_x(i) R_z(RAAN) =\n", Rx(i) @ Rz(O))
print(f"largest entry difference {np.abs(Rz(O) @ Rx(i) - Rx(i) @ Rz(O)).max():.4f}")
e1 = np.array([1.0, 0, 0])
print("simplest case, 90 deg each, applied to x = (1, 0, 0):")
print("  R_z(90) R_x(90) x =", np.round(Rz(math.pi / 2) @ Rx(math.pi / 2) @ e1, 12))
print("  R_x(90) R_z(90) x =", np.round(Rx(math.pi / 2) @ Rz(math.pi / 2) @ e1, 12))

section("the formula equals position_eci: 10,000 random (i, RAAN, u, t)")
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(10000):
    ii, oo, uu = rng.uniform(0, 180), rng.uniform(0, 360), rng.uniform(0, 360)
    t = rng.uniform(0, 43200)
    orb = CircularOrbit(550e3, ii, oo, uu)
    uu_t = math.radians(uu) + 2 * math.pi * t / orb.period_s
    q = Rz(math.radians(oo)) @ Rx(math.radians(ii)) @ np.array([a * math.cos(uu_t), a * math.sin(uu_t), 0.0])
    worst = max(worst, float(np.abs(q - orb.position_eci(t)).max()))
print(f"largest coordinate difference {worst:.2e} m")

section("inclination 97.6 deg")
print(f"retrograde: i > 90; past polar by {I0 - 90:.1f} deg; highest latitude reached 180 - i = {180 - I0:.1f} deg")
print(f"z component of the orbit normal cos i = {math.cos(math.radians(I0)):.4f}")
# velocity at the ascending node (u = 0): direction of d/du of the formula, (-sin O cos i ... )
vdir = Rz(O) @ Rx(i) @ np.array([0.0, 1.0, 0.0])
print(f"velocity direction at the ascending node for RAAN 40: {np.round(vdir, 4)}; "
      f"eastward component (along z x r) {np.dot(vdir, np.cross([0, 0, 1], Rz(O) @ e1)):+.4f} (negative = westward)")


# ---------------------------------------------------------------- planes
def normal(inc_deg, raan_deg):
    ii, oo = math.radians(inc_deg), math.radians(raan_deg)
    return np.array([math.sin(oo) * math.sin(ii), -math.cos(oo) * math.sin(ii), math.cos(ii)])


def plane_angle(n1, n2):
    """Angle between two orbit planes as sets (0..90 deg) and between their orientations (0..180)."""
    c = float(np.clip(np.dot(n1, n2), -1, 1))
    ang = math.degrees(math.acos(c))
    return min(ang, 180 - ang), ang


section("why RAAN is spread over 180 deg and not 360 deg (i = 97.6)")
for d in (90.0, 180.0, 120.0, 60.0, 240.0):
    geo, ori = plane_angle(normal(I0, 0.0), normal(I0, d))
    print(f"  planes at RAAN 0 and {d:g}: {geo:.1f} deg apart as planes (normals {ori:.1f} deg apart)")
print(f"  formula: the plane at RAAN + 180 is 2 (i - 90) = {2 * (I0 - 90):.1f} deg from the plane at RAAN, flown the other way")
print("  normal vector of the plane with (i, RAAN): (sin RAAN sin i, -cos RAAN sin i, cos i)")

# ---------------------------------------------------------------- constellations
section("make_constellation, seed 0, the benchmark's three sizes")
export = {}
for name, kw in SIZES.items():
    cfg = ScenarioConfig(seed=0, **kw)
    orbits = make_constellation(cfg)
    n_planes = max(1, int(round(np.sqrt(cfg.n_sats))))
    per_plane = int(np.ceil(cfg.n_sats / n_planes))
    print(f"\n  {name}: {cfg.n_sats} satellites, sqrt = {math.sqrt(cfg.n_sats):.3f} -> {n_planes} planes, "
          f"per_plane = ceil({cfg.n_sats}/{n_planes}) = {per_plane}; RAAN step 180/{n_planes} = {180 / n_planes:g} deg; "
          f"phase step in a plane 360/{per_plane} = {360 / per_plane:g} deg; offset between planes "
          f"360/({n_planes}*{per_plane}) = {360 / (n_planes * per_plane):g} deg")
    rows = []
    for k, o in enumerate(orbits):
        plane, slot = k % n_planes, k // n_planes
        raan_nom = 180.0 * plane / n_planes
        phase_nom = 360.0 * slot / per_plane + 360.0 * plane / (n_planes * per_plane)
        rows.append(dict(k=k, plane=plane, slot=slot, raan=o.raan_deg, phase=o.phase_deg, raanNom=raan_nom, phaseNom=phase_nom))
        print(f"    sat {k}: plane {plane}, slot {slot}: RAAN {raan_nom:6.1f} {o.raan_deg - raan_nom:+.2f} = {o.raan_deg:7.2f} deg, "
              f"phase {phase_nom:6.1f} {o.phase_deg - phase_nom:+.2f} = {o.phase_deg:7.2f} deg")
    counts = [sum(1 for r in rows if r["plane"] == p) for p in range(n_planes)]
    print(f"    satellites per plane: {counts}")
    # geometry facts: plane angles and closest approach between satellites over the horizon
    planes = sorted({r["plane"] for r in rows})
    first = {p: next(r for r in rows if r["plane"] == p) for p in planes}
    for p in planes:
        for q in planes:
            if q > p:
                g, _ = plane_angle(normal(I0, first[p]["raan"]), normal(I0, first[q]["raan"]))
                print(f"    planes {p} and {q} (first satellites): {g:.1f} deg apart")
    # smallest angle between the planes of two satellites in different nominal planes, as built and with the RAAN
    # spread over 360 deg instead of 180 deg (same jitter)
    def min_sep(raans):
        best = 180.0
        for r1 in rows:
            for r2 in rows:
                if r1["plane"] < r2["plane"]:
                    best = min(best, plane_angle(normal(I0, raans[r1["k"]]), normal(I0, raans[r2["k"]]))[0])
        return best
    built = [r["raan"] for r in rows]
    alt360 = [2 * r["raanNom"] + (r["raan"] - r["raanNom"]) for r in rows]
    print(f"    smallest angle between planes of different nominal planes: {min_sep(built):.1f} deg as built, "
          f"{min_sep(alt360):.1f} deg if RAAN were spread over 360 deg")
    times = np.arange(0, cfg.horizon_h * 3600 + 1, 10.0)
    pos = np.stack([o.position_eci(times) for o in orbits])  # (S, T, 3)
    best = (1e18, None)
    for s1 in range(len(orbits)):
        for s2 in range(s1 + 1, len(orbits)):
            d = np.linalg.norm(pos[s1] - pos[s2], axis=-1)
            j = int(np.argmin(d))
            if d[j] < best[0]:
                z = pos[s1, j, 2]
                best = (float(d[j]), (s1, s2, float(times[j]), math.degrees(math.asin(z / a))))
    s1, s2, tt, lat = best[1]
    print(f"    closest approach over {cfg.horizon_h:g} h on the 10 s grid: satellites {s1} and {s2}, "
          f"{best[0] / 1e3:,.0f} km at t = {tt:.0f} s (latitude {lat:.1f} deg)")
    export[name] = {
        "nSats": cfg.n_sats, "nPlanes": n_planes, "perPlane": per_plane, "horizon": int(cfg.horizon_h * 3600),
        "orbits": [[o.altitude_m, o.inclination_deg, round(o.raan_deg, 6), round(o.phase_deg, 6)] for o in orbits],
        "plane": [r["plane"] for r in rows], "slot": [r["slot"] for r in rows],
        "raanNom": [r["raanNom"] for r in rows], "phaseNom": [r["phaseNom"] for r in rows],
    }

section("satellites per revolution: one satellite crosses the equator twice per period")
T = CircularOrbit(550e3, I0, 0, 0).period_s
print(f"period {T:.1f} s; time between two satellites of one plane in the medium layout (120 deg apart): {T / 3:.0f} s = {T / 3 / 60:.1f} min")
print(f"small layout, two per plane (180 deg apart): {T / 2:.0f} s = {T / 2 / 60:.1f} min")

header = ("/* Satellite Pass Planning from Scratch, chapter 02: the constellations of satplan.scenario.make_constellation\n"
          "   for the benchmark's three sizes (seed 0), written by code/ch02_numbers.py. orbits: [altitude m, inclination,\n"
          "   RAAN, phase at t = 0] in degrees; plane, slot and the nominal RAAN and phase before the random jitter. */\n")
OUT.write_text(header + "window.SAT_CH02 = " + json.dumps({"period": T, "sizes": export}, separators=(",", ":")) + ";\n",
               encoding="utf-8")
print(f"\nwrote {OUT} ({OUT.stat().st_size} bytes)")

section("small facts quoted in the text")
from satplan.geometry import MU_EARTH, OMEGA_EARTH  # noqa: E402
print(f"Earth rotation during one revolution: {math.degrees(OMEGA_EARTH * T):.2f} deg")


def angles(r, v):
    h = np.cross(r, v); h /= np.linalg.norm(h)
    inc = np.degrees(np.arccos(h[2]))
    raan = np.degrees(np.arctan2(h[0], -h[1])) % 360
    node = np.array([np.cos(np.radians(raan)), np.sin(np.radians(raan)), 0.0])
    u = np.degrees(np.arctan2(np.dot(np.cross(node, r), h), np.dot(node, r))) % 360
    return inc, raan, u


n_rate = 2 * math.pi / T
v_eci = Rz(O) @ Rx(i) @ np.array([-a * n_rate * math.sin(u), a * n_rate * math.cos(u), 0.0])
print(f"exercise 1: |v| = {np.linalg.norm(v_eci):.1f} m/s")
print("exercise 2: angles(r, v) = (%.4f, %.4f, %.4f)" % angles(p_eci, v_eci))
for n in (6, 10):
    orbs = make_constellation(ScenarioConfig(n_sats=n))
    P = int(round(np.sqrt(n)))
    print(f"exercise 3: n = {n}: {P} planes, counts {[sum(1 for k in range(n) if k % P == p) for p in range(P)]}, "
          f"RAANs {[round(o.raan_deg, 2) for o in orbs]}")
