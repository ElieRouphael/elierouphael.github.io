"""Chapter 03, The Earth Turns Underneath: every number quoted on the page.

Usage:  python code/ch03_numbers.py
satplan is imported from the course archive (satplan/ next to code/) or, failing that, from
the repository path in SATPLAN_REPO (default: the author's working copy).

Prints, labelled: the sidereal and solar day, the orbit period and revolutions per day, the westward
shift of the ground track per revolution, the maximum latitude, passes per day over a point against its
latitude (one satellite, 24 h, 30 s grid, as the page's widget computes it), the revisit of the ten
maritime areas in the course's small instance, and the J2 nodal regression with what it would shift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent, Path(os.environ.get("SATPLAN_REPO", r"C:/Users/rouphaee/Documents/PostDoc_v2/satellite-pass-planning"))):
    if (cand / "satplan").is_dir():
        sys.path.insert(0, str(cand))
        break

from satplan.geometry import OMEGA_EARTH, R_EARTH, CircularOrbit, boolean_runs, ground_points_eci, visibility_mask  # noqa: E402
from satplan.scenario import GROUND_STATIONS, MARITIME_AREAS, ScenarioConfig, build_instance, make_constellation  # noqa: E402

DAY = 86400.0
YEAR_DAYS = 365.2422  # tropical year, mean solar days
J2 = 1.08263e-3       # Earth's second zonal harmonic (dimensionless)
R_EQ = 6_378_137.0    # m, equatorial radius (the reference radius of J2)
MU = 3.986004418e14


def sub_point(r, t):
    """Geocentric latitude and Earth-fixed longitude (deg) of the sub-satellite point, as sat.js subPoint."""
    r = np.atleast_2d(r)
    lat = np.degrees(np.arcsin(r[:, 2] / np.linalg.norm(r, axis=1)))
    lon = np.degrees(np.arctan2(r[:, 1], r[:, 0]) - OMEGA_EARTH * np.asarray(t))
    lon = (lon + 180) % 360 - 180
    return lat, lon


def main() -> None:
    print("== Part 1: the Earth's rotation")
    sid = 2 * np.pi / OMEGA_EARTH
    print(f"OMEGA_EARTH = {OMEGA_EARTH} rad/s = {np.degrees(OMEGA_EARTH):.6e} deg/s = {np.degrees(OMEGA_EARTH) * 3600:.4f} deg/h")
    print(f"sidereal day 2 pi / OMEGA = {sid:.2f} s = {int(sid // 3600)} h {int(sid % 3600 // 60)} min {sid % 60:.1f} s")
    print(f"solar day - sidereal day = {DAY - sid:.1f} s = {(DAY - sid) / 60:.2f} min")
    print(f"rotation in one solar day = {np.degrees(OMEGA_EARTH) * DAY:.4f} deg; 2 pi/solar day would be {2 * np.pi / DAY:.6e} rad/s")
    print(f"extra rotation per solar day 360/{YEAR_DAYS} = {360 / YEAR_DAYS:.4f} deg/day")
    # the error of using the solar rate instead, after 6 h, in longitude and km at the equator
    err6 = np.degrees(OMEGA_EARTH - 2 * np.pi / DAY) * 6 * 3600
    print(f"using 2 pi / 86400 s instead: longitude error after 6 h = {err6:.4f} deg = {np.radians(err6) * R_EARTH / 1e3:.1f} km at the equator")

    print("\n== Part 2: the orbit and its ground track")
    orb = CircularOrbit(550e3, 97.6, 0.0, 0.0)
    T = orb.period_s
    print(f"a = {orb.semi_major_axis / 1e3:.0f} km, period T = {T:.3f} s = {T / 60:.2f} min, revolutions per solar day = {DAY / T:.3f}")
    dlon = np.degrees(OMEGA_EARTH * T)
    print(f"westward shift per revolution = OMEGA * T = {OMEGA_EARTH * T:.5f} rad = {dlon:.2f} deg = {np.radians(dlon) * R_EARTH / 1e3:.0f} km at the equator")
    print(f"number of tracks to fill 360 deg: 360 / shift = {360 / dlon:.2f}")
    # footprint half-width at the 55 deg cone (chapter 04): Earth central angle lambda = 90 - eps - eta
    k = R_EARTH / orb.semi_major_axis
    eps = np.degrees(np.arccos(np.sin(np.radians(55.0)) / k))
    lam = 90.0 - 55.0 - eps
    print(f"footprint half-width (chapter 04): lambda = {lam:.2f} deg = {np.radians(lam) * R_EARTH / 1e3:.0f} km; swath = {2 * np.radians(lam) * R_EARTH / 1e3:.0f} km")
    print(f"track spacing at latitude phi = {np.radians(dlon) * R_EARTH / 1e3:.0f} cos(phi) km; equals the swath at phi = {np.degrees(np.arccos(2 * lam / dlon)):.1f} deg")
    # check the shift from successive ascending-node longitudes
    t = np.arange(0, 3 * T, 1.0)
    lat, lon = sub_point(orb.position_eci(t), t)
    asc = np.flatnonzero((lat[:-1] < 0) & (lat[1:] >= 0))
    lons = lon[asc]
    print(f"ascending-node longitudes found by sampling at 1 s: {np.round(lons, 2).tolist()} -> differences {np.round(np.diff(lons), 2).tolist()}")
    # maximum latitude, the repository test
    o30 = CircularOrbit(550e3, 97.6, 30.0, 0.0)
    p = o30.position_eci(np.linspace(0, o30.period_s, 4000))
    latmax = np.degrees(np.arcsin(p[:, 2] / np.linalg.norm(p, axis=1))).max()
    print(f"maximum latitude (test_max_latitude_equals_inclination_supplement, 4000 samples): {latmax:.4f} deg; 180 - i = {180 - 97.6:.1f}")
    # latitude where the track turns: the satellite is then moving west relative to the ground?
    # ground speed of the sub-satellite point (inertial) and Earth surface speed at the equator
    v_sub = 2 * np.pi * R_EARTH / T
    print(f"sub-satellite point speed (inertial, on the sphere) = {v_sub:.0f} m/s; Earth surface at equator = {OMEGA_EARTH * R_EARTH:.0f} m/s")
    # heading at the equator crossing relative to the ground (deg west of north) for the descending/ascending pass
    i = np.radians(97.6)
    n = 2 * np.pi / T
    vn = n * np.sin(i)              # northward angular rate at the ascending node (rad/s of arc)
    ve = n * np.cos(i) - OMEGA_EARTH  # eastward rate relative to the ground
    print(f"ascending-node ground heading: {np.degrees(np.arctan2(ve, vn)):.2f} deg from north (negative = west of north); inertial: {np.degrees(np.arctan2(n * np.cos(i), vn)):.2f}")

    print("\n== Part 3: revisit")
    # passes per day over a point against its latitude: one satellite, 24 h, 30 s grid, 12 longitudes,
    # visibility as the targets' (55 deg cone, 10 deg elevation); the page's widget does the same.
    times = np.arange(0, DAY + 1, 30.0)
    lats = np.arange(0, 90, 5.0)
    lons = np.arange(-180, 180, 30.0)
    for inc in (97.6, 53.0):
        sat = CircularOrbit(550e3, inc, 0.0, 0.0).position_eci(times)
        print(f"passes per 24 h over a point (mean over 12 longitudes), one satellite at {inc} deg, 30 s grid:")
        for la in lats:
            g = ground_points_eci(np.full(lons.size, la), lons, times)
            m = visibility_mask(sat, g, 55.0, 10.0)
            cnt = [len(boolean_runs(m[:, j])) for j in range(lons.size)]
            print(f"  lat {la:4.0f}: mean {np.mean(cnt):5.2f}, min {min(cnt)}, max {max(cnt)}")

    # the small instance: passes over each maritime area, from its opportunities. Windows of the same satellite
    # that start less than 20 min after the previous one ends belong to the same pass (the page's widget does the same).
    cfg = ScenarioConfig(n_sats=4, n_targets=120, horizon_h=6.0, seed=0)
    inst, meta = build_instance(cfg)
    orbits = meta["orbits"]
    tgrid = np.arange(0, cfg.horizon_h * 3600 + 1, cfg.time_step_s, dtype=float)
    print("\nsmall instance: passes over each maritime area (opportunities of its targets, any of the 4 satellites, 6 h)")
    gaps_all, npass = [], []
    for k, (name, *_box) in enumerate(MARITIME_AREAS):
        ops = sorted((o for o in inst.opportunities if meta["area"][o.target] == k), key=lambda o: o.window_start)
        passes = []  # [sat, start, end, number of opportunities]
        for o in ops:
            for p in passes:
                if p[0] == o.sat and o.window_start <= p[2] + 1200:
                    p[2] = max(p[2], o.window_end)
                    p[3] += 1
                    break
            else:
                passes.append([o.sat, o.window_start, o.window_end, 1])
        passes.sort(key=lambda p: p[1])
        gaps = [(passes[q + 1][1] - passes[q][2]) / 60 for q in range(len(passes) - 1)]
        gaps_all += gaps
        npass.append(len(passes))
        desc = ", ".join(f"sat {p[0]} at {p[1] / 3600:.2f} h ({(p[2] - p[1]) / 60:.1f} min, {p[3]} opp.)" for p in passes)
        print(f"  {name:26s}: {len(ops):2d} opportunities, {len(passes)} passes: {desc}; gaps (min) = {[round(g) for g in gaps]}")
    short = [g for g in gaps_all if g < 60]
    long_ = [g for g in gaps_all if g >= 60]
    print(f"  all areas: passes per area {npass}, {len(gaps_all)} gaps; short gaps {len(short)} from {min(short):.0f} to {max(short):.0f} min; "
          f"long gaps {len(long_)} from {min(long_) / 60:.1f} to {max(long_) / 60:.1f} h")
    # the polar station
    for name, la, lo in GROUND_STATIONS:
        npass = 0
        for ob in orbits:
            m = visibility_mask(ob.position_eci(tgrid), ground_points_eci([la], [lo], tgrid), None, 5.0)[:, 0]
            npass += len(boolean_runs(m))
        per = [len(boolean_runs(visibility_mask(ob.position_eci(tgrid), ground_points_eci([la], [lo], tgrid), None, 5.0)[:, 0])) for ob in orbits]
        print(f"  station {name} ({la}, {lo}): {npass} contacts in 6 h over the 4 satellites, per satellite {per}; revolutions in 6 h = {6 * 3600 / T:.2f}")
    print("  contacts per satellite (merged over stations, from build_instance):", [len(c) for c in meta["contacts"]])

    print("\n== Part 4: what J2 would add")
    a = orb.semi_major_axis
    n = np.sqrt(MU / a**3)
    for inc in (97.6,):
        rate = -1.5 * J2 * (R_EQ / a) ** 2 * n * np.cos(np.radians(inc))  # rad/s
        print(f"a = {a / 1e3:.0f} km, n = {n:.6e} rad/s, i = {inc}: dOmega/dt = {rate:.4e} rad/s = {np.degrees(rate) * DAY:.4f} deg/day")
    sun = 360 / YEAR_DAYS
    print(f"Sun's apparent motion: 360 / {YEAR_DAYS} = {sun:.4f} deg/day")
    cos_ss = -np.radians(sun) / DAY / (1.5 * J2 * (R_EQ / a) ** 2 * n)
    print(f"sun-synchronous inclination at a = {a / 1e3:.0f} km: {np.degrees(np.arccos(cos_ss)):.2f} deg")
    a2 = R_EQ + 550e3
    n2 = np.sqrt(MU / a2**3)
    cos_ss2 = -np.radians(sun) / DAY / (1.5 * J2 * (R_EQ / a2) ** 2 * n2)
    print(f"(with 550 km above the equatorial radius, a = {a2 / 1e3:.0f} km: {np.degrees(np.arccos(cos_ss2)):.2f} deg)")
    rate = -1.5 * J2 * (R_EQ / a) ** 2 * n * np.cos(np.radians(97.6))
    for hours in (6, 12, 24):
        d = rate * hours * 3600
        print(f"RAAN drift over {hours:2d} h: {np.degrees(d):.3f} deg -> {d * R_EARTH / 1e3:.1f} km east at the equator, "
              f"{d * R_EARTH * np.cos(np.radians(45)) / 1e3:.1f} km at 45 deg")
    print(f"drift over a year: {np.degrees(rate) * DAY * YEAR_DAYS:.1f} deg")
    # what an equatorial (i = 0) or polar orbit would do
    for inc in (0.0, 53.0, 90.0):
        r = -1.5 * J2 * (R_EQ / a) ** 2 * n * np.cos(np.radians(inc))
        print(f"  i = {inc:4.0f}: dOmega/dt = {np.degrees(r) * DAY:+.3f} deg/day")
    # the second J2 effect: the argument of latitude advances at n (1 + 3/4 J2 (R/a)^2 (8 cos^2 i - 2)) for e = 0
    k = 0.75 * J2 * (R_EQ / a) ** 2 * (8 * np.cos(np.radians(97.6)) ** 2 - 2)
    for hours in (6, 12):
        du = k * n * hours * 3600
        print(f"along-track (argument of latitude) offset over {hours} h with the Kepler period kept: {np.degrees(du):.2f} deg = "
              f"{du * a / 1e3:.0f} km in orbit = {abs(du) / n:.1f} s of timing")
    print(f"relative change of the argument-of-latitude rate: {k:.3e}; period that would match it: {T / (1 + k):.2f} s vs {T:.2f} s")
    # The a in that formula is the theory's mean semi-major axis, not the satellite's average distance from the
    # centre. Compare at the same average distance instead: integrate the J2 equations of motion (point mass + J2)
    # from a circular state at r = a on the ascending node, and measure the node-to-node period against Kepler's
    # period for the time-averaged radius.
    from scipy.integrate import solve_ivp

    def rhs(_t, s):
        r, v = s[:3], s[3:]
        rn = np.linalg.norm(r)
        f = 1.5 * J2 * MU * R_EQ**2 / rn**5
        zz = 5 * r[2] ** 2 / rn**2
        return np.concatenate([v, -MU * r / rn**3 + f * np.array([r[0] * (zz - 1), r[1] * (zz - 1), r[2] * (zz - 3)])])

    inc = np.radians(97.6)
    v0 = np.sqrt(MU / a)
    s0 = [a, 0.0, 0.0, 0.0, v0 * np.cos(inc), v0 * np.sin(inc)]
    sol = solve_ivp(rhs, [0, 30 * T], s0, method="DOP853", rtol=1e-11, atol=1e-5, dense_output=True)
    tt = np.linspace(0, 30 * T, 300001)
    st = sol.sol(tt)
    zs = st[2]
    up = np.flatnonzero((zs[:-1] < 0) & (zs[1:] >= 0))
    tn = np.array([tt[q] - zs[q] * (tt[q + 1] - tt[q]) / (zs[q + 1] - zs[q]) for q in up])
    TN = (tn[-1] - tn[0]) / (len(tn) - 1)
    sel = (tt >= tn[0]) & (tt < tn[-1])
    rbar = np.linalg.norm(st[:3], axis=0)[sel].mean()
    TK = 2 * np.pi * np.sqrt(rbar**3 / MU)
    rel = TN / TK - 1
    abar = (MU * (TN * (1 + k) / (2 * np.pi)) ** 2) ** (1 / 3)  # mean semi-major axis that the formula would need
    print(f"numerical J2 orbit: nodal period {TN:.2f} s, time-averaged radius {rbar / 1e3:.1f} km, Kepler period there {TK:.2f} s: "
          f"longer by {TN - TK:.2f} s per revolution ({rel:.2e}); the formula's mean a for this orbit = {abar / 1e3:.1f} km, "
          f"{(rbar - abar) / 1e3:.1f} km below the average radius")
    vorb = 2 * np.pi * a / T
    for hours in (6, 12):
        lag = rel / (1 + rel) * hours * 3600
        print(f"  at the same average radius, lag after {hours} h = {lag:.1f} s = {lag * vorb / 1e3:.0f} km along the track")
    print(f"  period that would match it: {T * (1 + rel):.2f} s vs {T:.2f} s")

    print("\n== Exercises")
    Trep = sid / 15
    arep = (MU * Trep**2 / (4 * np.pi**2)) ** (1 / 3)
    print(f"ex. 2: 15 revolutions per sidereal day: T = {Trep:.2f} s, a = {arep / 1e3:.0f} km, altitude = {(arep - R_EARTH) / 1e3:.0f} km; omega T = {np.degrees(OMEGA_EARTH * Trep):.2f} deg")
    for h in (300e3, 550e3, 800e3, 1200e3):
        aa = R_EARTH + h
        nn = np.sqrt(MU / aa**3)
        c = -np.radians(sun) / DAY / (1.5 * J2 * (R_EQ / aa) ** 2 * nn)
        print(f"ex. 3: sun-synchronous inclination at {h / 1e3:.0f} km (above the mean radius): {np.degrees(np.arccos(c)):.2f} deg")
    hs = np.arange(300e3, 20000e3, 10e3)
    cs = np.array([-np.radians(sun) / DAY / (1.5 * J2 * (R_EQ / (R_EARTH + h)) ** 2 * np.sqrt(MU / (R_EARTH + h) ** 3)) for h in hs])
    print(f"ex. 3: no sun-synchronous inclination above about {hs[np.argmax(cs < -1)] / 1e3:.0f} km")

    # ex. 4: the small instance's windows with the J2 node drift added to every orbit
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class DriftingOrbit(CircularOrbit):
        raan_rate: float = 0.0  # rad/s

        def position_eci(self, t):
            t = np.asarray(t, dtype=float)
            a = self.semi_major_axis
            u = np.radians(self.phase_deg) + 2.0 * np.pi * t / self.period_s
            inc = np.radians(self.inclination_deg)
            raan = np.radians(self.raan_deg) + self.raan_rate * t
            cu, su = np.cos(u), np.sin(u)
            x = a * (np.cos(raan) * cu - np.sin(raan) * su * np.cos(inc))
            y = a * (np.sin(raan) * cu + np.cos(raan) * su * np.cos(inc))
            z = a * (su * np.sin(inc))
            return np.stack([x, y, z], axis=-1)

    from satplan.scenario import _windows, make_targets
    tlat, tlon, _area, _w, tdur = make_targets(cfg)
    base, drift = [], []
    for s_, ob in enumerate(orbits):
        rr = -1.5 * J2 * (R_EQ / ob.semi_major_axis) ** 2 * (2 * np.pi / ob.period_s) * np.cos(np.radians(ob.inclination_deg))
        dob = DriftingOrbit(ob.altitude_m, ob.inclination_deg, ob.raan_deg, ob.phase_deg, rr)
        base += [(s_, j, ws, we) for j, ws, we, _ in _windows(ob, tlat, tlon, tgrid, 55.0, 10.0) if we - ws >= tdur[j]]
        drift += [(s_, j, ws, we) for j, ws, we, _ in _windows(dob, tlat, tlon, tgrid, 55.0, 10.0) if we - ws >= tdur[j]]
    def match(w, pool):
        c = [q for q in pool if q[0] == w[0] and q[1] == w[1] and q[2] <= w[3] and w[2] <= q[3]]
        return c[0] if c else None
    moved = [max(abs(m[2] - w[2]), abs(m[3] - w[3])) for w in base if (m := match(w, drift))]
    lost = sum(1 for w in base if match(w, drift) is None)
    new_ = sum(1 for w in drift if match(w, base) is None)
    print(f"ex. 4: opportunities without / with J2 node drift: {len(base)} / {len(drift)}; disappear {lost}, appear {new_}; "
          f"edges moved: max {max(moved)} s, by more than 10 s: {sum(1 for d in moved if d > 10)}, unchanged: {sum(1 for d in moved if d == 0)}")



if __name__ == "__main__":
    main()
