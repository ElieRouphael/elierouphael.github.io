"""Chapter 10, Rolling Horizons and Reading a Benchmark: every number quoted on the page, and
interactive/ch10-data.js.

What it runs (about 9 minutes on the author's 14-thread laptop; CP-SAT always with 8 workers):
  * small instance, seed 0: rolling-horizon CP-SAT with blocks of 30, 60, 120, 180 min and the whole 6 h,
    a total time limit of 20 s per run (the benchmark's small limit), each run twice with identical settings, against the whole-horizon optimum 219
    shipped in interactive/instance-small.js; the plans are exported for the page's timeline;
  * medium instance, seed 0: the same block sizes (whole = 12 h), a total time limit of 60 s per run
    (the benchmark's medium limit), each run twice;
  * run-to-run noise: CP-SAT (greedy hint, as in the benchmark) on medium seed 0, 30 s, random seeds 0 to 4;
  * one capstone check: CP-SAT on medium seed 0, 60 s, hinted with the 60-min rolling plan instead of greedy;
  * the 20-minute benchmark's numbers and the memory sweep, read from results/benchmark.json (cited, not rerun).
Every plan goes through satplan.checker.

Usage:  python code/ch10_numbers.py
satplan is imported from the course archive (satplan/ next to code/) or, failing that, from the
repository path in SATPLAN_REPO (default: the author's working copy).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = None
for cand in (HERE.parent, Path(os.environ.get("SATPLAN_REPO", r"C:/Users/rouphaee/Documents/PostDoc_v2/satellite-pass-planning"))):
    if (cand / "satplan").is_dir():
        sys.path.insert(0, str(cand))
        REPO = cand
        break

from satplan.checker import check_plan, plan_value  # noqa: E402
from satplan.cpsat import solve_cpsat  # noqa: E402
from satplan.greedy import solve_greedy  # noqa: E402
from satplan.rolling import solve_rolling  # noqa: E402
from satplan.scenario import ScenarioConfig, build_instance  # noqa: E402

OUT = HERE.parent / "interactive" / "ch10-data.js"
SMALL = dict(n_sats=4, n_targets=120, horizon_h=6.0)
MEDIUM = dict(n_sats=8, n_targets=250, horizon_h=12.0)
BLOCKS_MIN = [30, 60, 120, 180]
WORKERS = 8


def checked(inst, plan, value=None):
    errs = check_plan(inst, plan)
    assert not errs, errs[:3]
    v = plan_value(inst, plan)
    assert value is None or v == value
    return v


def per_block(inst, plan, block_s):
    """Priority collected per block, a task counted in the block where its window starts (as rolling.py)."""
    nb = -(-inst.horizon_s // block_s)
    out = [0] * nb
    for i in plan:
        o = inst.opportunities[i]
        out[min(o.window_start // block_s, nb - 1)] += o.weight
    return out


def sweep(name, inst, limit, whole_min, reps=2):
    """Each block length is run `reps` times with the same settings (seed 0): the first run's plan is kept for the
    page's timeline, and all values are kept to show that even the rolling horizon varies from run to run."""
    rows = []
    for bm in BLOCKS_MIN + [whole_min]:
        nb = -(-inst.horizon_s // (bm * 60))
        row = {"block_min": bm, "n_blocks": nb, "per_block_limit_s": round(limit / nb, 2), "values": [], "times": []}
        for rep in range(reps):
            r = solve_rolling(inst, bm * 60, limit, workers=WORKERS, seed=0)
            v = checked(inst, r["plan"], r["objective"])
            row["values"].append(v)
            row["times"].append(round(r["time_s"], 2))
            if rep == 0:
                row["value"], row["time_s"] = v, round(r["time_s"], 2)
                row["plan"] = {str(k): t for k, t in r["plan"].items()}
        rows.append(row)
        print(f"   [{name}] blocks {bm:4d} min ({nb:2d} blocks, {limit / nb:5.2f} s each): values {row['values']}, "
              f"times {row['times']} s")
    return rows


def main():
    t_all = time.perf_counter()
    data = {"limits": {"small": 20.0, "medium": 60.0, "noise": 30.0, "workers": WORKERS}}

    # ---------------- small instance: block sizes against the whole-horizon optimum
    inst, _ = build_instance(ScenarioConfig(seed=0, **SMALL))
    js = (HERE.parent / "interactive" / "instance-small.js").read_text(encoding="utf-8")
    shipped = json.loads(js[js.index("{"): js.rindex("}") + 1])
    opt_plan = {int(k): v for k, v in shipped["python"]["cpsat"]["plan"].items()}
    opt = checked(inst, opt_plan, 219)
    print(f"[small] whole-horizon optimum shipped in instance-small.js: {opt} (passes the checker)")
    rows = sweep("small", inst, 20.0, 360)
    for r in rows:
        plan = {int(k): v for k, v in r["plan"].items()}
        bs = r["block_min"] * 60
        rb, ob = per_block(inst, plan, bs), per_block(inst, opt_plan, bs)
        r["per_block"], r["opt_per_block"] = rb, ob
        tr = {inst.opportunities[i].target for i in plan}
        to = {inst.opportunities[i].target for i in opt_plan}
        r["only_opt"], r["only_roll"] = sorted(to - tr), sorted(tr - to)
        lost = sum(inst.target_weights[j] for j in to - tr)
        gained = sum(inst.target_weights[j] for j in tr - to)
        print(f"      {r['block_min']} min: per block rolling {rb} vs optimum {ob}; targets only in optimum "
              f"{len(to - tr)} (priority {lost}), only in rolling {len(tr - to)} (priority {gained})")
    data["small"] = {"opt": opt, "rows": rows}

    # ---------------- medium instance
    instm, _ = build_instance(ScenarioConfig(seed=0, **MEDIUM))
    rowsm = sweep("medium", instm, 60.0, 720)
    for r in rowsm:
        r.pop("plan")
    data["medium"] = {"rows": rowsm}

    # ---------------- run-to-run noise of CP-SAT on medium seed 0
    g = solve_greedy(instm)
    checked(instm, g["plan"], g["objective"])
    noise = []
    for seed in range(5):
        c = solve_cpsat(instm, 30.0, workers=WORKERS, hint=g["plan"], seed=seed)
        checked(instm, c["plan"], c["objective"])
        noise.append({"seed": seed, "value": c["objective"], "bound": c["bound"], "status": c["status"],
                      "time_s": round(c["time_s"], 1)})
        print(f"   [noise] CP-SAT medium seed 0, 30 s, random_seed {seed}: value {c['objective']}, bound {c['bound']:.0f}, "
              f"{c['status']}")
    vals = [n["value"] for n in noise]
    print(f"   [noise] values {vals}: min {min(vals)}, max {max(vals)}, spread {max(vals) - min(vals)}, "
          f"mean {sum(vals) / len(vals):.1f}; bounds {sorted({n['bound'] for n in noise})}")
    data["noise"] = noise

    # ---------------- capstone check: hint CP-SAT with the 60-min rolling plan
    r60 = solve_rolling(instm, 3600, 60.0, workers=WORKERS, seed=0)
    checked(instm, r60["plan"], r60["objective"])
    c = solve_cpsat(instm, 60.0, workers=WORKERS, hint=r60["plan"], seed=0)
    checked(instm, c["plan"], c["objective"])
    data["seeded"] = {"rolling": r60["objective"], "value": c["objective"], "bound": c["bound"]}
    print(f"   [seeded] CP-SAT medium seed 0, 60 s, hinted with the 60-min rolling plan ({r60['objective']}): "
          f"value {c['objective']}, bound {c['bound']:.0f}")

    # ---------------- the benchmark, cited
    bj = json.loads((REPO / "results" / "benchmark.json").read_text())
    bench = []
    for r in bj["records"]:
        res = {x["method"]: x for x in r["results"]}
        gr, cp, mi, rh = res["greedy"], res["cpsat+hint"], res["milp-highs"], res["rolling-cpsat-60min"]
        best = min(b for b in (cp["bound"], mi["bound"]) if b is not None)
        bench.append({"size": r["size"], "seed": r["seed"], "limit": r["time_limit_s"], "opps": r["stats"]["opportunities"],
                      "pairs": r["stats"]["conflict_pairs"], "greedy": gr["objective"], "cpsat": cp["objective"],
                      "rolling": rh["objective"], "milp": mi["objective"], "cpsat_bound": round(cp["bound"], 3),
                      "milp_bound": round(mi["bound"], 3), "best_bound": round(best, 3),
                      "t": [round(gr["time_s"], 2), round(cp["time_s"], 2), round(rh["time_s"], 2), round(mi["time_s"], 2)]})
        bv = max(gr["objective"], cp["objective"], rh["objective"], mi["objective"])
        print(f"   [bench] {r['size']} {r['seed']}: best bound {best:.0f}, best value {bv}; "
              + ", ".join(f"{k} {v} ({100 * (best - v) / best:.1f} %)" for k, v in
                          (("greedy", gr["objective"]), ("cpsat", cp["objective"]), ("rolling", rh["objective"]),
                           ("milp", mi["objective"]))))
    data["bench"] = bench
    data["memory"] = bj["memory_sweep"]
    print("   [memory sweep, cited]", [(m["memory_budget_s"], m["cpsat"], m["cpsat_bound"], m["status"]) for m in bj["memory_sweep"]])

    # medium seed 0: the best value ever seen in this script and the benchmark, for the true-gap bracket
    seen = [557] + vals + [v for r in rowsm for v in r["values"]] + [c["objective"]]
    best_seen = max(seen)
    bounds = [575.0, 576.0] + [n["bound"] for n in noise] + [c["bound"]]  # benchmark CP-SAT and HiGHS, then this script
    best_bound = min(bounds)
    print(f"   [medium 0] best value seen anywhere: {best_seen}; best bound proved anywhere: {best_bound:.0f} -> the optimum "
          f"is in [{best_seen}, {best_bound:.0f}]; CP-SAT's benchmark 557 is between {100 * (best_seen - 557) / best_seen:.1f} % "
          f"and {100 * (best_bound - 557) / best_bound:.1f} % below it (the benchmark's gap against 575: "
          f"{100 * (575 - 557) / 575:.1f} %)")
    data["best_seen_medium"] = best_seen
    data["best_bound_medium"] = best_bound

    OUT.write_text("/* Satellite Pass Planning from Scratch, chapter 10: rolling-horizon block sweeps (small and medium, seed 0),\n"
                   "   CP-SAT run-to-run noise and the benchmark's numbers, exported by code/ch10_numbers.py. */\n"
                   "window.SAT_CH10 = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB); total {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
