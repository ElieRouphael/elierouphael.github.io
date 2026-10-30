"""Chapter 05, From Geometry to a Planning Problem: every number quoted on the page.

Builds the course's small instance (4 satellites, 120 targets, 6 h, seed 0) with satplan and prints,
labelled, the numbers the chapter quotes: how targets are drawn, the windows and the opportunities,
the ground contacts and memory segments, the revolutions, the opportunities that straddle a segment
boundary (the approximation of the midpoint rule), the unreachable targets, and the statistics.

Usage:  python code/ch05_numbers.py
satplan is imported from the course archive (satplan/ next to code/) or, failing that, from the
repository path in SATPLAN_REPO (default: the author's working copy).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent, Path(os.environ.get("SATPLAN_REPO", r"C:/Users/rouphaee/Documents/PostDoc_v2/satellite-pass-planning"))):
    if (cand / "satplan").is_dir():
        sys.path.insert(0, str(cand))
        break

import numpy as np  # noqa: E402

from satplan.geometry import ground_points_eci, visibility_mask  # noqa: E402
from satplan.scenario import (  # noqa: E402
    GROUND_STATIONS, MARITIME_AREAS, PRIORITY_PROBS, PRIORITY_WEIGHTS, ScenarioConfig, _windows, build_instance,
    make_constellation, make_targets,
)


def straddles(inst, meta) -> dict:
    """Windows that contain a revolution boundary or a contact start, and those where a task could be
    placed wholly on the side of the boundary that the midpoint rule did not choose."""
    P, H = meta["period_s"], inst.horizon_s
    rev_b = [k * P for k in range(1, int(H // P) + 1)]
    out = Counter()
    for o in inst.opportunities:
        d, mid = o.duration, 0.5 * (o.window_start + o.window_end)
        bounds = [("revolution boundary", B) for B in rev_b] + [("contact start", c0) for c0, _ in meta["contacts"][o.sat]]
        movable = False
        for kind, B in bounds:
            if o.window_start < B < o.window_end:
                out[f"windows containing a {kind}"] += 1
                after = mid >= B
                if (after and o.window_start + d <= B) or ((not after) and o.latest_start >= B):
                    out[f"  ... where a task can lie wholly on the other side of the {kind}"] += 1
                    movable = True
        if movable:  # a window can contain both a revolution boundary and a contact start: count it once
            out["distinct windows with a task that can lie wholly on the wrong side"] += 1
        if any(o.window_start <= c1 and c0 <= o.window_end for c0, c1 in meta["contacts"][o.sat]):
            out["windows overlapping a ground contact in time"] += 1
    return dict(out)


def main() -> None:
    cfg = ScenarioConfig(n_sats=4, n_targets=120, horizon_h=6.0, seed=0)
    inst, meta = build_instance(cfg)
    opps = inst.opportunities
    P = meta["period_s"]
    H = inst.horizon_s
    print("== scenario ==")
    print("maritime areas:", len(MARITIME_AREAS), " ground stations:", len(GROUND_STATIONS))
    print("expected weight per target:", float(np.dot(PRIORITY_WEIGHTS, PRIORITY_PROBS)),
          " x 120 =", float(np.dot(PRIORITY_WEIGHTS, PRIORITY_PROBS)) * 120)
    lat, lon, area, weight, duration = make_targets(cfg)
    print("targets per area:", {MARITIME_AREAS[a][0]: int(c) for a, c in sorted(Counter(area.tolist()).items())})
    print("priority counts:", dict(sorted(Counter(weight.tolist()).items())))
    print("duration counts:", dict(sorted(Counter(duration.tolist()).items())))
    print("total weight:", int(weight.sum()), " total requested listening [s]:", int(duration.sum()))
    # area sizes (km^2, spherical) to show that targets per area does not follow area
    R = 6371.0
    for name, la0, la1, lo0, lo1 in MARITIME_AREAS:
        a_km2 = R * R * np.radians(lo1 - lo0) * (np.sin(np.radians(la1)) - np.sin(np.radians(la0)))
        print(f"   area {name:<26} {a_km2/1e3:8.0f} thousand km^2")

    print("\n== orbits ==")
    print("period [s]:", round(P, 2), " = min", round(P / 60, 2), "  revolutions in 6 h:", round(H / P, 3))
    print("revolution boundaries [s]:", [round(k * P, 1) for k in range(1, int(H // P) + 1)])
    for s, o in enumerate(meta["orbits"]):
        print(f"   sat {s}: RAAN {o.raan_deg:.2f}  phase {o.phase_deg:.2f}")

    print("\n== windows and opportunities ==")
    times = np.arange(0, H + 1, cfg.time_step_s, dtype=float)
    raw_total, too_short = 0, 0
    raw_targets = set()
    for s, orbit in enumerate(meta["orbits"]):
        w = _windows(orbit, lat, lon, times, cfg.max_off_nadir_deg, cfg.min_elevation_deg)
        raw_total += len(w)
        for j, ws, we, _ in w:
            raw_targets.add(j)
            if we - ws < duration[j]:
                too_short += 1
    print("raw visibility windows:", raw_total, " too short for the listening time:", too_short,
          " kept:", raw_total - too_short)
    print("opportunities:", len(opps))
    print("opportunities per satellite:", [sum(o.sat == s for o in opps) for s in range(inst.n_sats)])
    lens = np.array([o.window_end - o.window_start for o in opps])
    print("window length [s] min / median / max:", int(lens.min()), int(np.median(lens)), int(lens.max()))
    slack = np.array([o.latest_start - o.earliest_start for o in opps])
    print("start-time freedom latest-earliest [s] min / median / max:", int(slack.min()), int(np.median(slack)), int(slack.max()))
    bt = inst.by_target()
    print("opportunities per reachable target:", dict(sorted(Counter(len(v) for v in bt.values()).items())))
    o0 = opps[0]
    print("first opportunity:", o0)
    print("   earliest_start", o0.earliest_start, " latest_start", o0.latest_start)
    # the target with the most opportunities (the page's default selection)
    jmax = max(bt, key=lambda j: (len(bt[j]), -j))
    print("target with most opportunities:", jmax, "n =", len(bt[jmax]), "area", MARITIME_AREAS[area[jmax]][0],
          "weight", weight[jmax], "duration", duration[jmax], "opps", bt[jmax])
    print("   their (sat, ws, we, rev, mem):", [(opps[i].sat, opps[i].window_start, opps[i].window_end, opps[i].revolution, opps[i].memory_segment) for i in bt[jmax]])

    print("\n== unreachable targets ==")
    unreach = [j for j in range(inst.n_targets) if j not in bt]
    print("unreachable:", len(unreach), " weight lost:", int(sum(weight[j] for j in unreach)))
    for j in unreach:
        n_raw = 0
        best = []
        for s, orbit in enumerate(meta["orbits"]):
            for jj, ws, we, _ in _windows(orbit, lat[j:j + 1], lon[j:j + 1], times, cfg.max_off_nadir_deg, cfg.min_elevation_deg):
                n_raw += 1
                best.append(we - ws)
        print(f"   target {j}: area {MARITIME_AREAS[area[j]][0]}, lat {lat[j]:.2f} lon {lon[j]:.2f}, weight {weight[j]}, "
              f"duration {duration[j]} s, raw windows {n_raw}, their lengths {sorted(best)}")
    print("targets with at least one raw window:", len(raw_targets))

    print("\n== ground contacts and memory segments ==")
    for s, cs in enumerate(meta["contacts"]):
        d = [c1 - c0 for c0, c1 in cs]
        print(f"   sat {s}: {len(cs)} contacts, starts {[c0 for c0, _ in cs]}, lengths {d}")
    print("contacts in total:", sum(len(c) for c in meta["contacts"]))
    # which stations see each contact (contacts are merged across stations)
    gs_lat = np.array([g[1] for g in GROUND_STATIONS]); gs_lon = np.array([g[2] for g in GROUND_STATIONS])
    station_count = Counter()
    for s, orbit in enumerate(meta["orbits"]):
        m = visibility_mask(orbit.position_eci(times), ground_points_eci(gs_lat, gs_lon, times), None, cfg.station_min_elevation_deg)
        for c0, c1 in meta["contacts"][s]:
            i0, i1 = c0 // cfg.time_step_s, c1 // cfg.time_step_s
            names = tuple(GROUND_STATIONS[k][0] for k in range(len(GROUND_STATIONS)) if m[i0:i1 + 1, k].any())
            station_count[names] += 1
    print("contacts by station(s) seen:", dict(station_count))
    print("memory segments used per satellite:", [sorted({o.memory_segment for o in opps if o.sat == s}) for s in range(inst.n_sats)])
    print("revolutions used per satellite:", [sorted({o.revolution for o in opps if o.sat == s}) for s in range(inst.n_sats)])

    print("\n== the midpoint rule: windows straddling a boundary ==")
    for k, v in straddles(inst, meta).items():
        print(f"   {k}: {v}")
    for o in opps:
        for c0, c1 in meta["contacts"][o.sat]:
            if o.window_start < c0 < o.window_end:
                print(f"   straddler: opp {o.idx} sat {o.sat} window [{o.window_start}, {o.window_end}] duration {o.duration}, "
                      f"contact starts {c0}, midpoint {0.5 * (o.window_start + o.window_end)}, segment {o.memory_segment}")
    med, med_meta = build_instance(ScenarioConfig(n_sats=8, n_targets=250, horizon_h=12.0, seed=0))
    print("   medium seed 0:", len(med.opportunities), "opportunities")
    for k, v in straddles(med, med_meta).items():
        print(f"   medium seed 0, {k}: {v}")
    sm = straddles(med, med_meta)
    movable = sm.get("distinct windows with a task that can lie wholly on the wrong side", 0)
    print(f"   medium seed 0: windows whose task could be charged to the wrong segment: {movable} of "
          f"{len(med.opportunities)} = {100 * movable / len(med.opportunities):.1f} %")

    print("\n== statistics ==")
    print("small seed 0 stats():", inst.stats())
    print("max transition [s]:", inst.max_transition())
    print("total requested listening of reachable targets [s]:", int(sum(duration[j] for j in bt)))
    groups = inst.budget_groups()
    print("binding budget groups:", len(groups), Counter(k for k, _, _ in groups))
    print("medium and large sizes: results/results.md (opps, pairs) and results/benchmark.json (stats)")
    print("growth small -> large seed 0: opportunities x", round(4826 / 168, 1), " pairs x", round(145039 / 876, 1))

    print("\n== passes (windows of one satellite separated by less than 10 min) ==")
    for s in range(inst.n_sats):
        ws = sorted(opps_s := [o for o in opps if o.sat == s], key=lambda o: o.window_start)
        passes, cur = [], [ws[0]]
        for o in ws[1:]:
            if o.window_start > max(q.window_end for q in cur) + 600:
                passes.append(cur); cur = [o]
            else:
                cur.append(o)
        passes.append(cur)
        print(f"   sat {s}: {len(passes)} passes, sizes {[len(p) for p in passes]}, "
              f"starts {[p[0].window_start for p in passes]}, ends {[max(q.window_end for q in p) for p in passes]}")

    print("\n== exercise 3: the small configuration over 12 h ==")
    i12, _ = build_instance(ScenarioConfig(n_sats=4, n_targets=120, horizon_h=12.0, seed=0))
    st12 = i12.stats()
    print("   ", {k: st12[k] for k in ("targets_with_opportunity", "opportunities", "conflict_pairs", "total_weight", "reachable_weight")})

    print("\n== exercise 4: revolutions counted from the ascending node ==")
    from dataclasses import replace
    from satplan.cpsat import solve_cpsat
    phases = [o.phase_deg for o in meta["orbits"]]
    new = [replace(o, revolution=int(phases[o.sat] / 360.0 + 0.5 * (o.window_start + o.window_end) / P)) for o in opps]
    changed = sum(a.revolution != b.revolution for a, b in zip(opps, new))
    inst2 = replace(inst, opportunities=new, pairs=[])
    inst2.build_pairs()
    print("   opportunities that change revolution:", changed, " budget groups:", len(inst2.budget_groups()))
    r1 = solve_cpsat(inst, 30, workers=8, seed=0)
    r2 = solve_cpsat(inst2, 30, workers=8, seed=0)
    print("   CP-SAT, midpoint // period:", r1["objective"], r1["status"], " node-crossing revolutions:", r2["objective"], r2["status"])


if __name__ == "__main__":
    main()
