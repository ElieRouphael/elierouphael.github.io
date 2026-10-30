"""Chapter 06, Constraints and an Independent Checker: every number quoted on the page.

Builds the course's small instance (4 satellites, 120 targets, 6 h, seed 0) with satplan and prints,
labelled: the reconfiguration times, how much the pointing direction turns inside a window (the
midpoint approximation), the triangle inequality of s(a, b), the breakdown of the conflict-pair
sweep and its three cases, the brute-force check of the sweep, the budget groups, and what the
independent checker says about the course's greedy and CP-SAT plans and about broken plans.

The plans are read from interactive/instance-small.js (the ones the page uses), so that the numbers
refer to exactly the plans on screen; CP-SAT is multi-threaded and a rerun could return another optimum.

Usage:  python code/ch06_numbers.py
satplan is imported from the course archive (satplan/ next to code/) or, failing that, from the
repository path in SATPLAN_REPO (default: the author's working copy).
"""
from __future__ import annotations

import itertools
import json
import math
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

from satplan.checker import check_plan, plan_value  # noqa: E402
from satplan.geometry import ground_points_eci, pointing_angle_deg  # noqa: E402
from satplan.scenario import ScenarioConfig, build_instance  # noqa: E402

JS = HERE.parent / "interactive" / "instance-small.js"


def load_plans():
    text = JS.read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    p = data["python"]
    return ({int(k): v for k, v in p["greedy"]["plan"].items()}, {int(k): v for k, v in p["cpsat"]["plan"].items()})


def los(orbit, lat, lon, t):
    """Unit vector satellite -> target in the inertial frame at time t."""
    g = ground_points_eci([lat], [lon], [t])[0, 0]
    s = orbit.position_eci(t)
    u = g - s
    return u / np.linalg.norm(u)


def main() -> None:
    inst, meta = build_instance(ScenarioConfig(n_sats=4, n_targets=120, horizon_h=6.0, seed=0))
    opps = inst.opportunities
    greedy, cpsat = load_plans()
    by_sat = inst.by_satellite()

    print("== the decision ==")
    print("opportunities:", len(opps), " subsets: 2^", len(opps), "=", f"{2.0 ** len(opps):.2e}")
    print("tasks: greedy", len(greedy), "value", plan_value(inst, greedy), "| CP-SAT", len(cpsat), "value", plan_value(inst, cpsat))

    print("\n== reconfiguration times ==")
    print("settle", inst.settle_s, "s, slew", inst.slew_rate_deg_s, "deg/s, max_transition", inst.max_transition(), "s")
    s_all, sym = [], True
    for ids in by_sat.values():
        for a, b in itertools.combinations(ids, 2):
            s_ab, s_ba = inst.transition(a, b), inst.transition(b, a)
            sym &= s_ab == s_ba
            s_all.append(s_ab)
    s_all = np.array(s_all)
    print("same-satellite pairs:", len(s_all), " s(a,b) == s(b,a) for all:", sym)
    print("s over all same-satellite pairs min / median / max:", s_all.min(), int(np.median(s_all)), s_all.max())
    sp = np.array([p.s_ab for p in inst.pairs])
    print("s over the 876 conflict pairs min / median / max:", sp.min(), int(np.median(sp)), sp.max())

    print("\n== pointing inside a window (the midpoint approximation) ==")
    lat, lon, orbits = meta["lat"], meta["lon"], meta["orbits"]
    turn, dev = [], []
    for o in opps:
        orb = orbits[o.sat]
        u0, u1 = los(orb, lat[o.target], lon[o.target], o.window_start), los(orb, lat[o.target], lon[o.target], o.window_end)
        turn.append(pointing_angle_deg(u0, u1))
        mid = np.asarray(o.pointing)
        dev.append(max(pointing_angle_deg(los(orb, lat[o.target], lon[o.target], t), mid)
                       for t in np.arange(o.window_start, o.window_end + 1, 10)))
    turn, dev = np.array(turn), np.array(dev)
    print("angle turned between window start and end [deg] min / median / max:", round(turn.min(), 1), round(float(np.median(turn)), 1), round(turn.max(), 1))
    print("largest angle from the midpoint direction inside a window [deg] median / max:", round(float(np.median(dev)), 1), round(dev.max(), 1))
    print("   as seconds of slew at 2 deg/s, median / max:", round(float(np.median(dev)) / 2, 1), round(dev.max() / 2, 1))
    # a sample opportunity, for the page
    o = opps[0]
    orb = orbits[o.sat]
    print(f"sample opp 0 (sat {o.sat}, window [{o.window_start}, {o.window_end}]): angle to midpoint direction at "
          f"start {pointing_angle_deg(los(orb, lat[o.target], lon[o.target], o.window_start), np.asarray(o.pointing)):.1f}, "
          f"end {pointing_angle_deg(los(orb, lat[o.target], lon[o.target], o.window_end), np.asarray(o.pointing)):.1f}, "
          f"start-to-end {turn[0]:.1f} deg")
    # consequence for the plans: redo each consecutive transition with the directions at the actual times
    for name, plan in (("greedy", greedy), ("CP-SAT", cpsat)):
        n, short, diffs = 0, 0, []
        per = {}
        for i, t in plan.items():
            per.setdefault(opps[i].sat, []).append(i)
        for s, ids in per.items():
            ids.sort(key=lambda i: plan[i])
            for a, b in zip(ids, ids[1:]):
                oa, ob = opps[a], opps[b]
                end_a = plan[a] + oa.duration
                ua = los(orbits[s], lat[oa.target], lon[oa.target], end_a)
                ub = los(orbits[s], lat[ob.target], lon[ob.target], plan[b])
                s_true = int(math.ceil(inst.settle_s + pointing_angle_deg(ua, ub) / inst.slew_rate_deg_s))
                n += 1
                diffs.append(s_true - inst.transition(a, b))
                if plan[b] - end_a < s_true:
                    short += 1
        diffs = np.array(diffs)
        print(f"{name}: consecutive transitions {n}; true-time transition minus midpoint transition [s] min / median / max: "
              f"{diffs.min()} / {np.median(diffs):g} / {diffs.max()}; gaps too short with true-time angles: {short}")

    print("\n== triangle inequality ==")
    n_tri, worst_angle, worst_s = 0, 1e9, 1e9
    for ids in by_sat.values():
        U = np.array([opps[i].pointing for i in ids])
        A = np.degrees(np.arccos(np.clip(U @ U.T, -1, 1)))
        S = np.ceil(inst.settle_s + A / inst.slew_rate_deg_s)
        m = len(ids)
        # slack[a, b, c] = X[a, b] + X[b, c] - X[a, c] over distinct a, b, c
        sa = A[:, :, None] + A[None, :, :] - A[:, None, :]
        ss = S[:, :, None] + S[None, :, :] - S[:, None, :]
        mask = np.ones((m, m, m), bool)
        idx = np.arange(m)
        mask[idx, idx, :] = False; mask[:, idx, idx] = False; mask[idx, :, idx] = False
        n_tri += int(mask.sum())
        worst_angle = min(worst_angle, float(sa[mask].min()))
        worst_s = min(worst_s, float(ss[mask].min()))
    print("ordered triples on a satellite:", n_tri, " min angle slack [deg]:", f"{worst_angle:.2e}", " min transition slack [s]:", worst_s)

    print("\n== conflict-pair sweep ==")
    s_max = inst.max_transition()
    c = Counter()
    for ids in by_sat.values():
        ids = sorted(ids, key=lambda i: opps[i].window_start)
        for p, a in enumerate(ids):
            oa = opps[a]
            for q, b in enumerate(ids[p + 1:]):
                ob = opps[b]
                if ob.window_start >= oa.window_end + s_max:
                    c["skipped by the break (b starts after a's window end + s_max)"] += len(ids) - (p + 1) - q
                    break
                if oa.target == ob.target:
                    c["same target"] += 1
                    continue
                s_ab, s_ba = inst.transition(a, b), inst.transition(b, a)
                if oa.window_end + s_ab <= ob.earliest_start:
                    c["a then b always holds"] += 1
                    continue
                if ob.window_end + s_ba <= oa.earliest_start:
                    c["b then a always holds"] += 1
                    continue
                c["kept"] += 1
    total = sum(math.comb(len(v), 2) for v in by_sat.values())
    print("same-satellite pairs:", total, dict(c), " sum:", sum(c.values()))
    both = sum(p.a_first_ok and p.b_first_ok for p in inst.pairs)
    a_only = sum(p.a_first_ok and not p.b_first_ok for p in inst.pairs)
    b_only = sum(p.b_first_ok and not p.a_first_ok for p in inst.pairs)
    none = sum(not p.a_first_ok and not p.b_first_ok for p in inst.pairs)
    print("pairs:", len(inst.pairs), " both orders:", both, " only a first:", a_only, " only b first:", b_only,
          " one order in total:", a_only + b_only, " neither:", none)
    # windows that overlap in time among the kept pairs
    ov = sum(opps[p.a].window_start < opps[p.b].window_end and opps[p.b].window_start < opps[p.a].window_end for p in inst.pairs)
    print("kept pairs whose windows overlap in time:", ov)
    # brute force, as tests/test_planning.py::test_pair_sweep_skips_only_free_pairs, on this instance
    listed = {(p.a, p.b) for p in inst.pairs} | {(p.b, p.a) for p in inst.pairs}
    checked = 0
    for ids in by_sat.values():
        for a, b in itertools.combinations(ids, 2):
            if (a, b) in listed or opps[a].target == opps[b].target:
                continue
            free_ab = opps[a].window_end + inst.transition(a, b) <= opps[b].earliest_start
            free_ba = opps[b].window_end + inst.transition(b, a) <= opps[a].earliest_start
            assert free_ab or free_ba
            checked += 1
    print("brute force: unlisted different-target pairs checked free:", checked)
    # examples for the page's pair widget defaults
    for kind, pred in (("both", lambda p: p.a_first_ok and p.b_first_ok), ("one", lambda p: p.a_first_ok != p.b_first_ok),
                       ("neither", lambda p: not p.a_first_ok and not p.b_first_ok)):
        p = next(p for p in inst.pairs if pred(p))
        oa, ob = opps[p.a], opps[p.b]
        print(f"first '{kind}' pair: {p}; a window [{oa.window_start}, {oa.window_end}] d {oa.duration}; "
              f"b window [{ob.window_start}, {ob.window_end}] d {ob.duration}")

    print("\n== budgets ==")
    duty, mem = {}, {}
    for o in opps:
        duty.setdefault((o.sat, o.revolution), []).append(o.idx)
        mem.setdefault((o.sat, o.memory_segment), []).append(o.idx)
    groups = inst.budget_groups()
    print("duty groups:", len(duty), " memory groups:", len(mem), " total:", len(duty) + len(mem),
          " kept:", len(groups), Counter(k for k, _, _ in groups))
    for kind, lim, ids in groups:
        o0 = opps[ids[0]]
        key = (o0.sat, o0.revolution if kind == "duty" else o0.memory_segment)
        tot = sum(opps[i].duration for i in ids)
        print(f"   {kind:<6} sat {key[0]} seg {key[1]}: {len(ids)} opps, {tot} s requested of {lim} s "
              f"(x{tot / lim:.2f}); used greedy {sum(opps[i].duration for i in ids if i in greedy)}, "
              f"CP-SAT {sum(opps[i].duration for i in ids if i in cpsat)}")
    ratios = [sum(opps[i].duration for i in ids) / lim for _, lim, ids in groups]
    print("oversubscription of kept groups min / max:", round(min(ratios), 2), round(max(ratios), 2))
    full = [(k, ids) for k, lim, ids in groups if sum(opps[i].duration for i in ids if i in cpsat) == lim]
    print("kept groups exactly full in the CP-SAT plan:", len(full), Counter(k for k, _ in full))
    full_g = [(k, ids) for k, lim, ids in groups if sum(opps[i].duration for i in ids if i in greedy) == lim]
    print("kept groups exactly full in the greedy plan:", len(full_g), Counter(k for k, _ in full_g))

    print("\n== the checker ==")
    print("greedy:", check_plan(inst, greedy), " CP-SAT:", check_plan(inst, cpsat))
    bad = dict(cpsat)
    i0 = sorted(cpsat)[0]
    bad[i0] = opps[i0].latest_start + 10
    print("CP-SAT with one start 10 s past its latest start:", check_plan(inst, bad))
    # add an opportunity of an already collected target
    tgt = {opps[i].target for i in cpsat}
    extra = next(o.idx for o in opps if o.idx not in cpsat and o.target in tgt)
    bad = dict(cpsat); bad[extra] = opps[extra].earliest_start
    print("CP-SAT plus a second opportunity of a collected target:", check_plan(inst, bad)[:3])
    print("CP-SAT plus every unused opportunity at its earliest start: violations =",
          len(check_plan(inst, {**{o.idx: o.earliest_start for o in opps}, **cpsat})))

    print("\n== the page's default demo ==")
    # busiest 40-minute window of the CP-SAT plan per satellite
    for s in range(inst.n_sats):
        ts = sorted((cpsat[i], i) for i in cpsat if opps[i].sat == s)
        best = max(((sum(1 for t, _ in ts if t0 <= t < t0 + 2400), t0) for t0, _ in ts), default=(0, 0))
        print(f"   sat {s}: CP-SAT tasks {len(ts)}, busiest 40 min holds {best[0]} tasks from t = {best[1]}")
    S, T0, T1 = 3, 12300, 12800
    print(f"default view: satellite {S}, [{T0}, {T1}] s")
    ids = [o.idx for o in opps if o.sat == S and o.window_end > T0 and o.window_start < T1]
    print("   opportunities in view:", len(ids), ids)
    chosen = sorted((cpsat[i], i) for i in ids if i in cpsat)
    print("   CP-SAT tasks in view:", [(i, t, opps[i].duration, opps[i].target, opps[i].weight) for t, i in chosen])
    for (ta, a), (tb, b) in zip(chosen, chosen[1:]):
        print(f"   {a} -> {b}: s = {inst.transition(a, b)} s, gap end-to-start {tb - ta - opps[a].duration} s")
    sub = {i: cpsat[i] for i in cpsat if opps[i].sat == S}
    print("   satellite's CP-SAT plan:", len(sub), "tasks, value", plan_value(inst, sub), "checker:", check_plan(inst, sub))
    for kind, lim, gids in groups:
        if opps[gids[0]].sat == S:
            o0 = opps[gids[0]]
            print(f"   {kind} seg {o0.revolution if kind == 'duty' else o0.memory_segment}: used {sum(opps[i].duration for i in gids if i in sub)} of {lim}")
    for r in sorted({o.revolution for o in opps if o.sat == S}):
        print(f"   satellite {S} revolution {r}: requested {sum(o.duration for o in opps if o.sat == S and o.revolution == r)} s, "
              f"CP-SAT uses {sum(opps[i].duration for i in sub if opps[i].revolution == r)} s")
    for m in sorted({o.memory_segment for o in opps if o.sat == S}):
        print(f"   satellite {S} memory segment {m}: requested {sum(o.duration for o in opps if o.sat == S and o.memory_segment == m)} s, "
              f"CP-SAT uses {sum(opps[i].duration for i in sub if opps[i].memory_segment == m)} s")

    print("\n== experiments with CP-SAT (exercises and Part 5) ==")
    from dataclasses import replace
    from satplan.cpsat import solve_cpsat
    from satplan.instance import Opportunity, PlanningInstance
    r = solve_cpsat(replace(inst, memory_budget_s=10**6), 30, workers=8, seed=0)
    print("   memory budget removed: CP-SAT", r["objective"], r["status"], "(with 600 s: 219)")
    # the default pass alone, with the power left by the rest of the optimal plan (240 s) and with 300 s
    pass_ids = [o.idx for o in opps if o.sat == S and 12300 <= o.window_start and o.window_end <= 12800]
    for duty in (240, 300):
        sub_opps = [Opportunity(k, 0, k, opps[i].window_start, opps[i].window_end, opps[i].duration, opps[i].weight,
                                opps[i].pointing, 0, 0) for k, i in enumerate(pass_ids)]
        sub = PlanningInstance(1, len(sub_opps), inst.horizon_s, sub_opps, inst.settle_s, inst.slew_rate_deg_s, duty,
                               inst.memory_budget_s, [o.weight for o in sub_opps], [o.duration for o in sub_opps])
        sub.build_pairs()
        rr = solve_cpsat(sub, 30, workers=8, seed=0)
        print(f"   default pass alone ({len(pass_ids)} opportunities), power {duty} s: best value {rr['objective']} {rr['status']}, "
              f"tasks {sorted(pass_ids[k] for k in rr['plan'])}")
    # exercise 1: a break that uses s_ab instead of s_max
    def sweep_bad(I):
        out = set()
        for ids in I.by_satellite().values():
            ids = sorted(ids, key=lambda i: I.opportunities[i].window_start)
            for p_, a_ in enumerate(ids):
                oa = I.opportunities[a_]
                for b_ in ids[p_ + 1:]:
                    ob = I.opportunities[b_]
                    if ob.window_start >= oa.window_end + I.transition(a_, b_):
                        break
                    if oa.target == ob.target:
                        continue
                    if oa.window_end + I.transition(a_, b_) <= ob.earliest_start:
                        continue
                    out.add((a_, b_))
        return out
    test_inst, _ = build_instance(ScenarioConfig(n_sats=3, n_targets=40, horizon_h=4, seed=3))
    for name, I in (("course instance", inst), ("test instance", test_inst)):
        good = {(p.a, p.b) for p in I.pairs}
        print(f"   exercise 1, {name}: pairs {len(good)}, with the s_ab break {len(sweep_bad(I) & good)} found, lost {len(good - sweep_bad(I))}")


if __name__ == "__main__":
    main()
