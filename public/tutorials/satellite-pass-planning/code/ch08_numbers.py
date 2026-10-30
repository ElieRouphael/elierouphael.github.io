"""Chapter 08, Constraint Programming with CP-SAT: every number quoted on the page.

Prints, labelled: the size of the CP-SAT model on the course's small instance; CP-SAT runs on it (8 workers
with and without the greedy hint, with the best plan and the best bound recorded over time; 1 worker, which
is deterministic, with and without the hint and without NoOverlap); the anytime trace of the benchmark's
medium run (read from results/benchmark.json) and of a fresh 60 s medium run that also records the bound;
the toy sub-instance of the page's branch and bound (one pass of satellite 0, 21 opportunities), the
branch-and-bound node counts for every setting of the widget, and the check that its optimum equals
CP-SAT's; and how the node count grows with the number of opportunities.
With --write, writes interactive/ch08-data.js (the page's traces and toy). CP-SAT with 8 workers is not
deterministic, so a rerun prints slightly different traces and medium values than the recorded file; the
1-worker runs, the toy and the branch-and-bound counts are deterministic.

Usage:  python code/ch08_numbers.py [--write]        (about three minutes: one 60 s medium run)
satplan is imported from the course archive (satplan/ next to code/) or, failing that, from the
repository path in SATPLAN_REPO (default: the author's working copy).
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = None
for cand in (HERE.parent, Path(os.environ.get("SATPLAN_REPO", r"C:/Users/rouphaee/Documents/PostDoc_v2/satellite-pass-planning"))):
    if (cand / "satplan").is_dir():
        sys.path.insert(0, str(cand))
        REPO = cand
        break

from ortools.sat.python import cp_model  # noqa: E402

from satplan.checker import check_plan  # noqa: E402
from satplan.cpsat import build_cpsat, solve_cpsat  # noqa: E402
from satplan.greedy import solve_greedy  # noqa: E402
from satplan.instance import PlanningInstance  # noqa: E402
from satplan.scenario import ScenarioConfig, build_instance  # noqa: E402

OUT = HERE.parent / "interactive" / "ch08-data.js"
SMALL = dict(n_sats=4, n_targets=120, horizon_h=6.0)
MEDIUM = dict(n_sats=8, n_targets=250, horizon_h=12.0)


# ------------------------------------------------------------------ CP-SAT with value and bound recorded
def run_cpsat(inst, time_limit, workers, hint=None, seed=0, no_overlap=True):
    """solve_cpsat's model and parameters, with both the best plan and the best bound recorded over time."""
    m, x, s = build_cpsat(inst, use_no_overlap=no_overlap)
    if hint is not None:
        for o in inst.opportunities:
            chosen = o.idx in hint
            m.add_hint(x[o.idx], chosen)
            if chosen:
                m.add_hint(s[o.idx], int(round(hint[o.idx])))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    t0 = time.perf_counter()
    bounds = []
    solver.best_bound_callback = lambda b: bounds.append((time.perf_counter() - t0, b))

    class Cb(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.pts = []

        def on_solution_callback(self):
            self.pts.append((time.perf_counter() - t0, self.objective_value))

    cb = Cb()
    status = solver.solve(m, cb)
    plan = {o.idx: float(solver.value(s[o.idx])) for o in inst.opportunities if solver.boolean_value(x[o.idx])}
    assert not check_plan(inst, plan)
    return {"status": solver.status_name(status), "value": round(solver.objective_value),
            "bound": solver.best_objective_bound, "time": solver.wall_time, "values": cb.pts, "bounds": bounds,
            "branches": solver.num_branches, "conflicts": solver.num_conflicts, "plan": plan}


def model_size(inst):
    bt = inst.by_target()
    pairs = inst.pairs
    both = sum(p.a_first_ok and p.b_first_ok for p in pairs)
    excl = sum(not p.a_first_ok and not p.b_first_ok for p in pairs)
    return {
        "x": len(inst.opportunities), "s": len(inst.opportunities),
        "atmostone": sum(len(v) > 1 for v in bt.values()),
        "targets_one_opp": sum(len(v) == 1 for v in bt.values()),
        "nooverlap": inst.n_sats,
        "clauses": excl, "one_order": len(pairs) - both - excl, "two_orders": both,
        "order_literals": both, "enforced_precedences": (len(pairs) - both - excl) + 2 * both,
        "budgets": len(inst.budget_groups()),
        "duty_groups": sum(k == "duty" for k, _, _ in inst.budget_groups()),
        "memory_groups": sum(k == "memory" for k, _, _ in inst.budget_groups()),
        "domain_mean": sum(o.latest_start - o.earliest_start + 1 for o in inst.opportunities) / len(inst.opportunities),
    }


# ------------------------------------------------------------------ the page's branch and bound
def sub_instance(inst, ids):
    """The opportunities `ids` as an instance of their own, re-indexed 0..n-1 in order of window start."""
    ids = sorted(ids, key=lambda i: (inst.opportunities[i].window_start, i))
    opps = [replace(inst.opportunities[i], idx=k) for k, i in enumerate(ids)]
    si = PlanningInstance(inst.n_sats, inst.n_targets, inst.horizon_s, opps, inst.settle_s, inst.slew_rate_deg_s,
                          inst.duty_budget_s, inst.memory_budget_s, inst.target_weights, inst.target_durations)
    si.build_pairs()
    return si, ids


def branch_and_bound(si, bound="simple", start=0, include_first=True, cap=5_000_000):
    """Depth-first branch and bound over one satellite's opportunities in order of window start, exactly as
    interactive/ch08.js runs it. A node decides the first k opportunities; 'take' places opportunity k at its
    earliest start after the last task taken (sequence fixed by window start). Bounds: 'simple' adds the weight
    of every remaining target; 'fit' only those whose opportunity could still be taken on its own; 'knap'
    solves the power budget's fractional knapsack over those (the budget's linear relaxation).
    Returns (best value, nodes explored, nodes pruned, best plan, trace of (node, best, global bound))."""
    O = si.opportunities
    n = len(O)
    tr = [[si.transition(a, b) if a != b else 0 for b in range(n)] for a in range(n)]
    duty, mem = si.duty_budget_s, si.memory_budget_s

    def start_of(node, o):
        k, val, last, tlast, taken, ud, um, plan = node
        if o.target in taken:
            return None
        if ud.get(o.revolution, 0) + o.duration > duty or um.get(o.memory_segment, 0) + o.duration > mem:
            return None
        t = o.window_start if last < 0 else max(o.window_start, tlast + O[last].duration + tr[last][o.idx])
        return t if t <= o.latest_start else None

    def ub(node):
        k, val, last, tlast, taken, ud, um, plan = node
        if bound == "knap":
            per_rev = {}
            for o in O[k:]:
                if start_of(node, o) is not None:
                    per_rev.setdefault(o.revolution, []).append(o)
            tot = val
            for rev in sorted(per_rev):
                cap_left = duty - ud.get(rev, 0)
                for o in sorted(per_rev[rev], key=lambda o: (-o.weight / o.duration, o.idx)):
                    if cap_left <= 0:
                        break
                    tot += min(1.0, cap_left / o.duration) * o.weight
                    cap_left -= o.duration
            return tot
        best = {}
        for o in O[k:]:
            if o.target in taken or (bound == "fit" and start_of(node, o) is None):
                continue
            best[o.target] = max(best.get(o.target, 0), o.weight)
        return val + sum(best.values())

    root = (0, 0, -1, 0, frozenset(), {}, {}, ())
    best, best_plan = start, None
    explored = pruned = 0
    stack = [(root, ub(root))]
    trace = [(0, best, ub(root))]
    while stack and explored < cap:
        node, b = stack.pop()
        explored += 1
        k, val, last, tlast, taken, ud, um, plan = node
        if val > best:
            best, best_plan = val, plan
        if b <= best:
            pruned += 1
        elif k < n:
            o = O[k]
            kids = [(k + 1, val, last, tlast, taken, ud, um, plan)]
            t = start_of(node, o)
            if t is not None:
                ud2, um2 = dict(ud), dict(um)
                ud2[o.revolution] = ud2.get(o.revolution, 0) + o.duration
                um2[o.memory_segment] = um2.get(o.memory_segment, 0) + o.duration
                kids.append((k + 1, val + o.weight, k, t, taken | {o.target}, ud2, um2, plan + ((k, t),)))
            if not include_first:
                kids.reverse()
            for c in kids:
                stack.append((c, ub(c)))
        glob = max([best] + [bb for _, bb in stack])
        if trace[-1][1] != best or trace[-1][2] != glob or not stack:
            trace.append((explored, best, glob))
    return best, explored, pruned, dict(best_plan or ()), trace


def main() -> None:
    write = "--write" in sys.argv[1:]
    data = {}
    print("== small instance, seed 0: the model ==")
    inst, meta = build_instance(ScenarioConfig(seed=0, **SMALL))
    ms = model_size(inst)
    print(ms)
    g = solve_greedy(inst)
    print("greedy", g["objective"])

    print("\n== CP-SAT on small, 8 workers, 20 s limit ==")
    runs = {}
    for name, hint in (("hint", g["plan"]), ("nohint", None)):
        r = run_cpsat(inst, 20.0, 8, hint=hint)
        runs[name] = r
        print(f"{name}: {r['status']} value {r['value']} bound {r['bound']:.0f} wall {r['time']:.2f} s, "
              f"{len(r['values'])} improving plans, first plan {r['values'][0][1]:.0f} at {r['values'][0][0]:.3f} s, "
              f"optimum found at {r['values'][-1][0]:.3f} s")
        print("   values:", [(round(t, 3), v) for t, v in r["values"]])
        print("   bounds (first 6, last 3):", [(round(t, 3), round(b, 1)) for t, b in r["bounds"][:6]],
              [(round(t, 3), round(b, 1)) for t, b in r["bounds"][-3:]], "count", len(r["bounds"]))
    print("sum of the weights of all opportunities (every x_o = 1):", sum(o.weight for o in inst.opportunities))
    hb = [[round(t, 3), round(b, 2)] for t, b in runs["hint"]["bounds"]]
    if hb[-1][1] != runs["hint"]["bound"]:  # the final proof is reported by the status, not by the callback
        hb.append([round(runs["hint"]["time"], 3), runs["hint"]["bound"]])
    data["smallRun"] = {"values": [[round(t, 3), v] for t, v in runs["hint"]["values"]],
                        "bounds": hb,
                        "time": round(runs["hint"]["time"], 3)}

    print("\n== CP-SAT on small, 1 worker (deterministic) ==")
    one = {}
    for name, hint, nov in (("hint", g["plan"], True), ("nohint", None, True), ("hint, no NoOverlap", g["plan"], False),
                            ("nohint, no NoOverlap", None, False)):
        r = run_cpsat(inst, 60.0, 1, hint=hint, no_overlap=nov)
        one[name] = r
        print(f"{name}: {r['status']} value {r['value']} wall {r['time']:.2f} s branches {r['branches']} "
              f"conflicts {r['conflicts']} improving plans {len(r['values'])}")
    data["oneWorker"] = {k: {"time": round(v["time"], 2), "branches": v["branches"], "conflicts": v["conflicts"]}
                         for k, v in one.items()}

    print("\n== medium instance, seed 0: benchmark trace (results/benchmark.json) ==")
    bench = json.loads((REPO / "results" / "benchmark.json").read_text())
    rec = next(r for r in bench["records"] if r["size"] == "medium" and r["seed"] == 0)
    cm = next(x for x in rec["results"] if x["method"] == "cpsat+hint")
    gm_b = next(x for x in rec["results"] if x["method"] == "greedy")
    print(f"benchmark: value {cm['objective']} bound {cm['bound']:.0f} status {cm['status']} time {cm['time_s']:.1f} s; "
          f"greedy {gm_b['objective']}; {len(cm['trace'])} improving plans")
    print("   trace:", [(round(t, 2), v) for t, v in cm["trace"]])
    print(f"   gap at the time limit: {100 * (cm['bound'] - cm['objective']) / cm['bound']:.1f} %")
    for sz in ("small", "large"):
        for r in bench["records"]:
            if r["size"] == sz:
                c = next(x for x in r["results"] if x["method"] == "cpsat+hint")
                print(f"   {sz} seed {r['seed']}: value {c['objective']} bound {c['bound']:.0f} {c['status']} {c['time_s']:.1f} s")
    data["benchMedium"] = {"values": [[round(t, 2), v] for t, v in cm["trace"]], "bound": cm["bound"],
                           "value": cm["objective"], "time": round(cm["time_s"], 1), "greedy": gm_b["objective"]}

    print("\n== medium instance, seed 0: a fresh 60 s run recording the bound ==")
    im, _ = build_instance(ScenarioConfig(seed=0, **MEDIUM))
    gm = solve_greedy(im)
    mm = model_size(im)
    print("model:", {k: mm[k] for k in ("x", "clauses", "one_order", "two_orders", "budgets", "atmostone", "nooverlap")}, "pairs", len(im.pairs))
    r = run_cpsat(im, 60.0, 8, hint=gm["plan"], seed=0)
    print(f"fresh run: {r['status']} value {r['value']} bound {r['bound']:.0f} wall {r['time']:.1f} s, "
          f"{len(r['values'])} improving plans; gap {100 * (r['bound'] - r['value']) / r['bound']:.1f} %")
    print("   values:", [(round(t, 2), v) for t, v in r["values"]])
    bnd = r["bounds"]
    print("   bounds: first", [(round(t, 2), round(b, 1)) for t, b in bnd[:4]], "count", len(bnd))
    for tq in (1, 5, 10, 30, 60):
        bb = [b for t, b in bnd if t <= tq]
        vv = [v for t, v in r["values"] if t <= tq]
        print(f"   at {tq:>2} s: best plan {vv[-1] if vv else None}  best bound {round(bb[-1], 1) if bb else None}")
    # thin the bound trace for the page: keep the first, every change of at least 0.5, and the last
    thin, lastb = [], None
    for t, b in bnd:
        if lastb is None or lastb - b >= 0.5:
            thin.append([round(t, 2), round(b, 1)])
            lastb = b
    if bnd and thin[-1][1] != round(bnd[-1][1], 1):
        thin.append([round(bnd[-1][0], 2), round(bnd[-1][1], 1)])
    data["freshMedium"] = {"values": [[round(t, 2), v] for t, v in r["values"]], "bounds": thin,
                           "time": round(r["time"], 1), "value": r["value"], "bound": r["bound"],
                           "greedy": gm["objective"]}

    print("\n== the toy: satellite 0, first pass (windows inside 0-900 s) ==")
    ids = [o.idx for o in inst.opportunities if o.sat == 0 and o.window_start >= 0 and o.window_end <= 900]
    si, ids = sub_instance(inst, ids)
    O = si.opportunities
    tg = solve_greedy(si)
    tc = run_cpsat(si, 20.0, 8)
    areas = sorted({meta["area"][o.target] for o in O})
    print(f"{len(O)} opportunities (course indices {ids}), {len({o.target for o in O})} targets, "
          f"areas {[ [a[0] for a in __import__('satplan.scenario', fromlist=['MARITIME_AREAS']).MARITIME_AREAS][k] for k in areas]}")
    print(f"pairs {len(si.pairs)}, budget groups {[(k, lim, len(v)) for k, lim, v in si.budget_groups()]}, "
          f"total listening {sum(o.duration for o in O)} s, total weight {sum(si.target_weights[t] for t in {o.target for o in O})}")
    print(f"revolutions {sorted({o.revolution for o in O})}, memory segments {sorted({o.memory_segment for o in O})}")
    print(f"greedy {tg['objective']} ({len(tg['plan'])} tasks), CP-SAT {tc['value']} {tc['status']} ({len(tc['plan'])} tasks)")
    print(f"2^{len(O)} = {2 ** len(O)} subsets")
    table = {}
    for bound in ("simple", "fit", "knap"):
        for start_name, st in (("zero", 0), ("greedy", tg["objective"])):
            for inc in (True, False):
                best, ex, pr, plan, trace = branch_and_bound(si, bound, st, inc)
                key = f"{bound}|{start_name}|{'take' if inc else 'leave'}"
                table[key] = [best, ex, pr]
                print(f"B&B {key:<22} best {best} nodes {ex} pruned {pr}")
                if plan:
                    assert not check_plan(si, plan) and sum(O[i].weight for i in plan) == best
    assert all(v[0] == tc["value"] for v in table.values()), "the restricted search must reach CP-SAT's optimum"
    print("all settings reach CP-SAT's optimum:", tc["value"])
    best, ex, pr, plan, trace = branch_and_bound(si, "simple", 0, True)
    print("default trace (first 8, last 3):", trace[:8], trace[-3:])
    bound_root = {b: round(branch_and_bound(si, b, 0, True, cap=1)[4][0][2], 2) for b in ("simple", "fit", "knap")}
    print("root bounds:", bound_root)
    s_nb, _ = sub_instance(inst, ids)
    s_nb.duty_budget_s = 10 ** 6
    print("toy without the power budget: CP-SAT", run_cpsat(s_nb, 20.0, 8)["value"],
          "(memory budget kept; 'all remaining' root bound 49)")
    data["toy"] = {
        "courseIdx": ids, "settle": si.settle_s, "slewRate": si.slew_rate_deg_s, "dutyBudget": si.duty_budget_s,
        "memoryBudget": si.memory_budget_s, "nSats": inst.n_sats, "nTargets": inst.n_targets,
        "targetWeights": si.target_weights, "targetDurations": si.target_durations,
        "opps": [{"idx": o.idx, "sat": o.sat, "target": o.target, "ws": o.window_start, "we": o.window_end,
                  "dur": o.duration, "w": o.weight, "u": [float(v) for v in o.pointing], "rev": o.revolution,
                  "mem": o.memory_segment} for o in O],
        "tr": [[si.transition(a, b) if a != b else 0 for b in range(len(O))] for a in range(len(O))],
        "greedy": {"value": tg["objective"], "plan": {str(k): v for k, v in tg["plan"].items()}},
        "cpsat": {"value": tc["value"], "status": tc["status"], "plan": {str(k): v for k, v in tc["plan"].items()}},
        "bb": table, "rootBounds": bound_root,
    }

    print("\n== growth: the first n opportunities of satellite 3, knapsack bound, greedy start ==")
    s3 = sorted((o for o in inst.opportunities if o.sat == 3), key=lambda o: (o.window_start, o.idx))
    growth = []
    for n in (10, 20, 30, 40, 50, 64):
        sub, _ = sub_instance(inst, [o.idx for o in s3[:n]])
        gs = solve_greedy(sub)
        res = {}
        for bound in ("simple", "knap"):
            t0 = time.perf_counter()
            best, ex, pr, _, _ = branch_and_bound(sub, bound, gs["objective"], True, cap=500_000)
            res[bound] = (best, ex, round(time.perf_counter() - t0, 1))
        opt = run_cpsat(sub, 20.0, 8)["value"]
        growth.append([n, res["simple"][1], res["knap"][1], res["knap"][0], opt])
        print(f"n {n}: simple bound {res['simple']}  knapsack bound {res['knap']}  CP-SAT optimum {opt}"
              f"{'' if res['knap'][0] == opt else '  (fixed order misses the optimum)'}")
    data["growth"] = growth

    print("\n== a stretch where the fixed order by window start is not exact: satellite 0, windows in 17200-17600 s ==")
    ids2 = [o.idx for o in inst.opportunities if o.sat == 0 and o.window_start >= 17200 and o.window_end <= 17600]
    s2, _ = sub_instance(inst, ids2)
    g2 = solve_greedy(s2)
    b2 = branch_and_bound(s2, "knap", g2["objective"], True)
    c2 = run_cpsat(s2, 20.0, 8)
    print(f"{len(ids2)} opportunities: greedy {g2['objective']}, fixed-order branch and bound {b2[0]}, CP-SAT {c2['value']} "
          f"({c2['status']})")

    if not write:
        print("\n(run with --write to rewrite interactive/ch08-data.js)")
        return
    OUT.write_text("/* Satellite Pass Planning from Scratch, chapter 08: solver traces, the branch-and-bound toy and its "
                   "checks,\n   written by code/ch08_numbers.py. */\nwindow.SAT_CH08 = "
                   + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"\nwrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
