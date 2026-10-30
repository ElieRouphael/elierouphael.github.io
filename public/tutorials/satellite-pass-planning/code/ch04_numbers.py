"""Chapter 04, Seeing the Ground: every number quoted on the page, and interactive/ch04-data.js.

Usage:  python code/ch04_numbers.py
satplan is imported from the course archive (satplan/ next to code/) or, failing that, from
the repository path in SATPLAN_REPO (default: the author's working copy).

Prints, labelled: the visibility triangle at 550 km (elevation at the 55 deg cone, footprints, slant
ranges, the altitude where the elevation mask would start to bind), the check of those limits against
visibility_mask, pass durations against cross-track offset, the hero demo's default pass, the window
lengths of the course's small instance (kept and dropped by the duration filter), and the error of the
10 s grid measured with exact (bisected) window edges. Writes the windows to interactive/ch04-data.js.
"""
from __future__ import annotations

import json
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
from satplan.scenario import ScenarioConfig, _windows, build_instance, make_constellation, make_targets  # noqa: E402

OUT = HERE.parent / "interactive" / "ch04-data.js"
H = 550e3


def triangle(h, eps_deg=None, eta_deg=None):
    """Given the elevation or the off-nadir angle at the edge, return (eps, eta, lam) in degrees and the slant range in m."""
    k = R_EARTH / (R_EARTH + h)
    if eps_deg is not None:
        eta = np.degrees(np.arcsin(k * np.cos(np.radians(eps_deg))))
        eps = eps_deg
    else:
        eta = eta_deg
        eps = np.degrees(np.arccos(np.sin(np.radians(eta_deg)) / k))
    lam = 90.0 - eps - eta
    rng = R_EARTH * np.sin(np.radians(lam)) / np.sin(np.radians(eta))
    return eps, eta, lam, rng


def visible_scalar(orbit, lat, lon, t, max_off, min_el):
    return bool(visibility_mask(orbit.position_eci([t]), ground_points_eci([lat], [lon], [t]), max_off, min_el)[0, 0])


def edge(orbit, lat, lon, t_in, t_out, max_off, min_el, iters=50):
    """Bisect the visibility edge between a visible time t_in and an invisible time t_out."""
    for _ in range(iters):
        tm = 0.5 * (t_in + t_out)
        if visible_scalar(orbit, lat, lon, tm, max_off, min_el):
            t_in = tm
        else:
            t_out = tm
    return 0.5 * (t_in + t_out)


def ef_unit(r, t):
    """Earth-fixed unit vector of an inertial position at time t (rotate by -OMEGA t about z)."""
    th = OMEGA_EARTH * t
    x, y, z = r
    v = np.array([x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th), z])
    return v / np.linalg.norm(v)


def offset_target(orbit, t0, offset_m):
    """The point at a cross-track distance offset_m (positive = left of the ground track) from the sub-satellite
    point at t0, as the page's hero demo places it. Returns (lat, lon) in degrees."""
    p0 = ef_unit(orbit.position_eci(t0), t0)
    d = ef_unit(orbit.position_eci(t0 + 0.5), t0 + 0.5) - ef_unit(orbit.position_eci(t0 - 0.5), t0 - 0.5)
    d = d - np.dot(d, p0) * p0
    d /= np.linalg.norm(d)
    c = np.cross(p0, d)
    b = offset_m / R_EARTH
    g = np.cos(b) * p0 + np.sin(b) * c
    return float(np.degrees(np.arcsin(g[2]))), float(np.degrees(np.arctan2(g[1], g[0])))


def pass_window(orbit, lat, lon, t0, max_off, min_el, span=600.0):
    """Exact window around t0 (bisected edges) and the window sampled on the absolute 10 s grid."""
    ts = np.arange(np.floor((t0 - span) / 10) * 10, t0 + span + 1e-9, 1.0)
    m = visibility_mask(orbit.position_eci(ts), ground_points_eci([lat], [lon], ts), max_off, min_el)[:, 0]
    rs = boolean_runs(m)
    if not rs:
        return None
    i0, i1 = max(rs, key=lambda r: r[1] - r[0])
    a = edge(orbit, lat, lon, ts[i0], ts[i0] - 1.0, max_off, min_el)
    b = edge(orbit, lat, lon, ts[i1], ts[i1] + 1.0, max_off, min_el)
    grid = ts[(ts % 10) == 0]
    mg = visibility_mask(orbit.position_eci(grid), ground_points_eci([lat], [lon], grid), max_off, min_el)[:, 0]
    rg = [r for r in boolean_runs(mg) if grid[r[0]] <= b and grid[r[1]] >= a]
    ws, we = (grid[rg[0][0]], grid[rg[0][1]]) if rg else (None, None)
    return a, b, ws, we


def main() -> None:
    print("== Part 1-2: the triangle at 550 km")
    k = R_EARTH / (R_EARTH + H)
    print(f"R / (R + h) = {k:.5f}; horizon (eps = 0): eta = {np.degrees(np.arcsin(k)):.2f} deg, lambda = {90 - np.degrees(np.arcsin(k)):.2f} deg "
          f"= {np.radians(90 - np.degrees(np.arcsin(k))) * R_EARTH / 1e3:.0f} km")
    cases = [("targets, 55 deg cone", dict(eta_deg=55.0)), ("10 deg elevation alone", dict(eps_deg=10.0)),
             ("stations, 5 deg elevation", dict(eps_deg=5.0)), ("nadir", None)]
    for name, kw in cases:
        if kw is None:
            print(f"  {name:28s}: eps = 90, eta = 0, lambda = 0, range = {H / 1e3:.0f} km")
            continue
        eps, eta, lam, rng = triangle(H, **kw)
        print(f"  {name:28s}: eps = {eps:.2f} deg, eta = {eta:.2f} deg, lambda = {lam:.2f} deg = {np.radians(lam) * R_EARTH / 1e3:.0f} km, "
              f"slant range = {rng / 1e3:.0f} km, footprint area = {2 * np.pi * R_EARTH**2 * (1 - np.cos(np.radians(lam))) / 1e12:.2f} million km^2")
    h_bind = R_EARTH * (np.cos(np.radians(10)) / np.sin(np.radians(55)) - 1)
    print(f"the 10 deg mask starts to bind (eps at the 55 deg cone falls below 10 deg) above h = {h_bind / 1e3:.0f} km; "
          f"there lambda = 25 deg = {np.radians(25) * R_EARTH / 1e3:.0f} km")
    for h in (400e3, 800e3, 1200e3):
        eps, eta, lam, rng = triangle(h, eta_deg=55.0)
        print(f"  h = {h / 1e3:.0f} km: elevation at the cone edge {eps:.1f} deg, lambda {lam:.2f} deg = {np.radians(lam) * R_EARTH / 1e3:.0f} km")

    # check against visibility_mask: equatorial orbit over (0, 0) at t = 0, points along the equator
    orb0 = CircularOrbit(H, 0.0, 0.0, 0.0)
    lons = np.arange(0, 25, 0.001)
    g = ground_points_eci(np.zeros(lons.size), lons, [0.0])
    sat = orb0.position_eci([0.0])
    for name, mo, me in (("55 deg cone + 10 deg", 55.0, 10.0), ("10 deg only", None, 10.0), ("5 deg only (stations)", None, 5.0)):
        m = visibility_mask(sat, g, mo, me)[0]
        print(f"  visibility_mask along the equator, {name}: last visible longitude = {lons[m].max():.3f} deg")

    print("\n== Part 3: pass duration against cross-track offset")
    T = orb0.period_s
    n = 2 * np.pi / T
    eps, eta, lam, _ = triangle(H, eta_deg=55.0)
    L = np.radians(lam)
    print(f"overhead pass, no Earth rotation: 2 lambda / n = {2 * L / n:.1f} s")
    orb = CircularOrbit(H, 97.6, 0.0, 0.0)
    t0 = 643.0  # sub-satellite point near 40 deg N on the ascending part of the first revolution
    print(f"hero orbit: 550 km, 97.6 deg, RAAN 0, phase 0; t0 = {t0} s, sub-point at t0: lat/lon = "
          f"{np.round(np.degrees([np.arcsin(ef_unit(orb.position_eci(t0), t0)[2]), np.arctan2(ef_unit(orb.position_eci(t0), t0)[1], ef_unit(orb.position_eci(t0), t0)[0])]), 3).tolist()}")
    print(" offset (km) | formula 2/n acos(cos lam / cos beta) | simulated exact | sampled (10 s grid) | max elev | min off-nadir")
    for off in (0, 100, 200, 300, 400, 500, 600, 700, 800, 850, 870):
        lat, lon = offset_target(orb, t0, off * 1e3)
        b = off * 1e3 / R_EARTH
        form = 2 / n * np.arccos(np.cos(L) / np.cos(b)) if b < L else 0.0
        w = pass_window(orb, lat, lon, t0, 55.0, 10.0)
        ts = np.arange(t0 - 400, t0 + 400, 0.5)
        sp = orb.position_eci(ts)
        gp = ground_points_eci([lat], [lon], ts)[:, 0]
        los = gp - sp
        rng = np.linalg.norm(los, axis=1)
        up = gp / np.linalg.norm(gp, axis=1, keepdims=True)
        el = np.degrees(np.arcsin(-np.sum(los * up, axis=1) / rng)).max()
        nad = -sp / np.linalg.norm(sp, axis=1, keepdims=True)
        on = np.degrees(np.arccos(np.clip(np.sum(los * nad, axis=1) / rng, -1, 1))).min()
        if w is None:
            print(f"  {off:4d} | {form:6.1f} | not visible | | {el:.1f} | {on:.1f}")
        else:
            a, bb, ws, we = w
            print(f"  {off:4d} | {form:6.1f} | {bb - a:6.1f} ({a:.1f} to {bb:.1f}) | "
                  f"{(we - ws) if ws is not None else 'none'} ({ws} to {we}) | {el:.1f} | {on:.1f}")
    # the same overhead pass, right and left, shows Earth rotation's asymmetry
    for off in (-300, 300):
        lat, lon = offset_target(orb, t0, off * 1e3)
        a, bb, ws, we = pass_window(orb, lat, lon, t0, 55.0, 10.0)
        print(f"  offset {off:+d} km (+ = left of track): exact {bb - a:.1f} s, sampled {we - ws:.0f} s")
    # hero default: 300 km, 55 deg, 10 deg, grid offset 0 s; and the station mode
    lat, lon = offset_target(orb, t0, 300e3)
    a, bb, ws, we = pass_window(orb, lat, lon, t0, 55.0, 10.0)
    print(f"HERO default (300 km, 55 deg, 10 deg): target ({lat:.3f}, {lon:.3f}); exact [{a:.2f}, {bb:.2f}] = {bb - a:.1f} s; "
          f"sampled [{ws:.0f}, {we:.0f}] = {we - ws:.0f} s; lost {ws - a:.1f} s at the start and {bb - we:.1f} s at the end")
    a, bb, ws, we = pass_window(orb, lat, lon, t0, None, 5.0)
    print(f"HERO station mode (no cone, 5 deg) at 300 km: exact {bb - a:.1f} s, sampled {we - ws:.0f} s")
    lat0, lon0 = offset_target(orb, t0, 0.0)
    a, bb, ws, we = pass_window(orb, lat0, lon0, t0, None, 5.0)
    print(f"station directly under the track: exact {bb - a:.1f} s = {(bb - a) / 60:.1f} min")

    print("\n== Part 4: windows of the small instance")
    cfg = ScenarioConfig(n_sats=4, n_targets=120, horizon_h=6.0, seed=0)
    inst, meta = build_instance(cfg)
    lat, lon, area, weight, duration = make_targets(cfg)
    orbits = make_constellation(cfg)
    horizon = int(cfg.horizon_h * 3600)
    times = np.arange(0, horizon + 1, cfg.time_step_s, dtype=float)
    raw = []
    for s, o in enumerate(orbits):
        for j, ws, we, _u in _windows(o, lat, lon, times, cfg.max_off_nadir_deg, cfg.min_elevation_deg):
            raw.append((s, j, ws, we))
    kept = [r for r in raw if r[3] - r[2] >= duration[r[1]]]
    dropped = [r for r in raw if r[3] - r[2] < duration[r[1]]]
    L_opp = np.array([o.window_end - o.window_start for o in inst.opportunities])
    print(f"raw windows (no duration filter): {len(raw)}; kept: {len(kept)} (instance has {len(inst.opportunities)} opportunities); dropped: {len(dropped)}")
    print(f"targets with at least one raw window: {len({r[1] for r in raw})}; with an opportunity: {len({o.target for o in inst.opportunities})}")
    lens = np.array([r[3] - r[2] for r in raw])
    print(f"raw sampled lengths: min {lens.min()}, median {np.median(lens):.0f}, max {lens.max()} s; zero-length (one sample) windows: {(lens == 0).sum()}")
    print(f"opportunity window lengths: min {L_opp.min()}, quartiles {np.percentile(L_opp, [25, 50, 75]).tolist()}, max {L_opp.max()} s, mean {L_opp.mean():.1f}")
    print(f"opportunities with window <= 2 x duration: {sum(1 for o in inst.opportunities if o.window_end - o.window_start <= 2 * o.duration)}")
    for d in (30, 60, 90, 120):
        nk = sum(1 for r in kept if duration[r[1]] == d)
        nd = sum(1 for r in dropped if duration[r[1]] == d)
        print(f"  duration {d:3d} s: windows {nk + nd}, kept {nk}, dropped {nd}")
    clen = [c[1] - c[0] for cs in meta["contacts"] for c in cs]
    print(f"ground contacts (merged over the 4 stations, per satellite): {len(clen)}, lengths from {min(clen)} to {max(clen)} s, median {np.median(clen):.0f} s")
    edge_cut = [r for r in raw if r[2] == 0 or r[3] == horizon]
    print(f"windows touching the horizon ends (t = 0 or {horizon} s): {len(edge_cut)}")

    # exact edges by bisection, for every raw window not cut by the horizon
    print("\n== Part 5: the 10 s grid against exact edges")
    exact = []
    for s, j, ws, we in raw:
        o = orbits[s]
        a = edge(o, lat[j], lon[j], float(ws), float(ws) - 10.0, cfg.max_off_nadir_deg, cfg.min_elevation_deg) if ws > 0 else 0.0
        b = edge(o, lat[j], lon[j], float(we), float(we) + 10.0, cfg.max_off_nadir_deg, cfg.min_elevation_deg) if we < horizon else float(horizon)
        exact.append((a, b))
    inner = [q for q, r in enumerate(raw) if 0 < r[2] and r[3] < horizon]
    ls = np.array([raw[q][2] - exact[q][0] for q in inner])
    le = np.array([exact[q][1] - raw[q][3] for q in inner])
    tot = ls + le
    print(f"windows inside the horizon: {len(inner)}")
    print(f"lost at the start: mean {ls.mean():.2f} s, max {ls.max():.2f} s; at the end: mean {le.mean():.2f} s, max {le.max():.2f} s")
    print(f"total underestimate: mean {tot.mean():.2f} s, median {np.median(tot):.2f}, max {tot.max():.2f} s, min {tot.min():.2f} s")
    rel = tot / np.array([exact[q][1] - exact[q][0] for q in inner])
    print(f"relative underestimate: median {100 * np.median(rel):.1f} %")
    marginal = [q for q, r in enumerate(raw) if r[3] - r[2] < duration[r[1]] <= exact[q][1] - exact[q][0]]
    print(f"dropped windows whose exact length would have been long enough: {len(marginal)}")
    for q in marginal:
        s, j, ws, we = raw[q]
        print(f"   sat {s} target {j}: sampled {we - ws} s, exact {exact[q][1] - exact[q][0]:.1f} s, needs {duration[j]} s")
    # the tasks never leave true visibility: a task inside [ws, we] is inside [a, b]
    assert all(exact[q][0] <= raw[q][2] and raw[q][3] <= exact[q][1] for q in range(len(raw)))
    print("check: every sampled window lies inside its exact window (the grid is conservative)")
    ex_len = np.array([b - a for a, b in exact])
    print(f"exact lengths: median {np.median(ex_len):.1f} s, max {ex_len.max():.1f} s")

    # exercise 4: the same instance on finer grids
    import dataclasses
    import time
    t_ = time.perf_counter()
    build_instance(cfg)
    print(f"ex. 4: build_instance with time_step_s = 10 takes {time.perf_counter() - t_:.2f} s")
    for step in (5, 1):
        t_ = time.perf_counter()
        inst2, _ = build_instance(dataclasses.replace(cfg, time_step_s=step))
        print(f"ex. 4: build_instance with time_step_s = {step} takes {time.perf_counter() - t_:.2f} s")
        have = {(o.sat, o.target) for o in inst.opportunities}
        extra = sorted({(o.sat, o.target) for o in inst2.opportunities} - have)
        print(f"       (satellite, target) pairs with an opportunity only on the finer grid: {extra}")
        L2 = np.array([o.window_end - o.window_start for o in inst2.opportunities])
        st2 = inst2.stats()
        print(f"ex. 4: time_step_s = {step}: {len(inst2.opportunities)} opportunities, {st2['targets_with_opportunity']} targets reachable, "
              f"mean window {L2.mean():.1f} s, {st2['conflict_pairs']} conflict pairs")
    print(f"       time_step_s = 10: {len(inst.opportunities)} opportunities, {inst.stats()['targets_with_opportunity']} targets reachable, "
          f"mean window {L_opp.mean():.1f} s, {inst.stats()['conflict_pairs']} conflict pairs")

    data = {
        "horizon": horizon,
        # [sat, target, ws, we, duration, exact start, exact end, kept]
        "windows": [[s, j, ws, we, int(duration[j]), round(exact[q][0], 2), round(exact[q][1], 2), int(we - ws >= duration[j])]
                    for q, (s, j, ws, we) in enumerate(raw)],
    }
    OUT.write_text("/* Satellite Pass Planning from Scratch, chapter 04: every visibility window of the small instance before the\n"
                   "   duration filter, with exact (bisected) edges. Written by code/ch04_numbers.py. */\n"
                   "window.SAT_CH04 = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"\nwrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
