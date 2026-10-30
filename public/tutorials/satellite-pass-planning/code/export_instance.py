"""Export the course's shared planning instance to interactive/instance-small.js.

The instance is the benchmark's "small" size, seed 0: 4 satellites, 120 targets, a 6-hour horizon.
The file carries everything the chapter pages need to recompute the problem in the browser
(orbits, targets, ground contacts, opportunities with their pointing vectors), plus the Python
results the pages check themselves against: the instance statistics, the conflict pairs, the
greedy plan and an optimal CP-SAT plan with its proven bound.

Usage:  python code/export_instance.py
satplan is imported from the course archive (satplan/ next to code/) or, failing that, from
the repository path in SATPLAN_REPO (default: the author's working copy).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent, Path(os.environ.get("SATPLAN_REPO", r"C:/Users/rouphaee/Documents/PostDoc_v2/satellite-pass-planning"))):
    if (cand / "satplan").is_dir():
        sys.path.insert(0, str(cand))
        break

from satplan.checker import check_plan, plan_value  # noqa: E402
from satplan.cpsat import solve_cpsat  # noqa: E402
from satplan.greedy import solve_greedy  # noqa: E402
from satplan.scenario import GROUND_STATIONS, MARITIME_AREAS, ScenarioConfig, build_instance  # noqa: E402

OUT = HERE.parent / "interactive" / "instance-small.js"


def export(cfg: ScenarioConfig, out: Path, var: str) -> dict:
    inst, meta = build_instance(cfg)
    g = solve_greedy(inst)
    c = solve_cpsat(inst, time_limit_s=60.0, workers=8, hint=g["plan"], seed=0)
    for r in (g, c):
        assert not check_plan(inst, r["plan"]), r["method"]
    data = {
        "config": cfg.__dict__,
        "settle": inst.settle_s, "slewRate": inst.slew_rate_deg_s,
        "dutyBudget": inst.duty_budget_s, "memoryBudget": inst.memory_budget_s,
        "nSats": inst.n_sats, "nTargets": inst.n_targets, "horizon": inst.horizon_s,
        "period": meta["period_s"],
        "orbits": [[o.altitude_m, o.inclination_deg, o.raan_deg, o.phase_deg] for o in meta["orbits"]],
        "targets": [[round(float(a), 6), round(float(b), 6), int(k)] for a, b, k in zip(meta["lat"], meta["lon"], meta["area"])],
        "targetWeights": inst.target_weights, "targetDurations": inst.target_durations,
        "areas": [a[0] for a in MARITIME_AREAS], "areaBoxes": [a[1:] for a in MARITIME_AREAS],
        "stations": [list(s) for s in GROUND_STATIONS],
        "contacts": meta["contacts"],
        "opps": [
            {"idx": o.idx, "sat": o.sat, "target": o.target, "ws": o.window_start, "we": o.window_end,
             "dur": o.duration, "w": o.weight, "u": [float(v) for v in o.pointing],
             "rev": o.revolution, "mem": o.memory_segment}
            for o in inst.opportunities
        ],
        "python": {
            "stats": inst.stats(),
            "pairs": [[p.a, p.b, p.s_ab, p.s_ba, int(p.a_first_ok), int(p.b_first_ok)] for p in inst.pairs],
            "greedy": {"value": g["objective"], "plan": {str(k): v for k, v in g["plan"].items()}},
            "cpsat": {"value": c["objective"], "bound": c["bound"], "status": c["status"],
                      "time_s": round(c["time_s"], 2), "trace": [[round(a, 3), b] for a, b in c["trace"]],
                      "plan": {str(k): v for k, v in c["plan"].items()}},
        },
    }
    header = (f"/* Satellite Pass Planning from Scratch: the shared planning instance ({cfg.n_sats} satellites, "
              f"{cfg.n_targets} targets, {cfg.horizon_h:g} h, seed {cfg.seed}),\n   exported by code/export_instance.py from the "
              "satplan package, with Python's greedy and CP-SAT plans for the pages' checks. */\n")
    out.write_text(header + f"window.{var} = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    return data


def main() -> None:
    d = export(ScenarioConfig(n_sats=4, n_targets=120, horizon_h=6.0, seed=0), OUT, "SAT_SMALL")
    p = d["python"]
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    print("stats", p["stats"])
    print("greedy", p["greedy"]["value"], "cpsat", p["cpsat"]["value"], p["cpsat"]["status"], "bound", p["cpsat"]["bound"])


if __name__ == "__main__":
    main()
