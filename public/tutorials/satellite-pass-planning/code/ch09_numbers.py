"""Chapter 09, The Same Problem as a MILP: every number quoted on the page, and interactive/ch09-data.js.

What it computes (about one minute in all):
  * the size of the big-M MILP of satplan/milp.py on the small, medium and large instances (seed 0);
  * its LP relaxation bound (scipy.optimize.milp with every integrality set to 0, i.e. HiGHS's LP solver),
    against three weaker or simpler relaxations, on small and medium (and large if it solves in time);
  * how fractional the LP optimum is, and how many precedence rows it leaves switched off;
  * the smallest big-M of every pair against a naive M equal to the horizon;
  * one real two-way pair of the small instance for the page's (t_a, t_b) widget;
  * one MILP solve of the small instance (independent check of the optimum 219);
  * the 20-minute benchmark's numbers, read from results/benchmark.json (cited, not rerun).

Usage:  python code/ch09_numbers.py
satplan is imported from the course archive (satplan/ next to code/) or, failing that, from the
repository path in SATPLAN_REPO (default: the author's working copy).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

HERE = Path(__file__).resolve().parent
REPO = None
for cand in (HERE.parent, Path(os.environ.get("SATPLAN_REPO", r"C:/Users/rouphaee/Documents/PostDoc_v2/satellite-pass-planning"))):
    if (cand / "satplan").is_dir():
        sys.path.insert(0, str(cand))
        REPO = cand
        break

from satplan.checker import check_plan  # noqa: E402
from satplan.milp import build_milp, solve_milp  # noqa: E402
from satplan.scenario import ScenarioConfig, build_instance  # noqa: E402

OUT = HERE.parent / "interactive" / "ch09-data.js"
SIZES = {
    "small": dict(n_sats=4, n_targets=120, horizon_h=6.0),
    "medium": dict(n_sats=8, n_targets=250, horizon_h=12.0),
    "large": dict(n_sats=12, n_targets=600, horizon_h=12.0),
}


def build_variant(inst, mode="tight"):
    """The model of satplan/milp.py, with the precedence rows changed by `mode`:
    'tight'  : exactly milp.py (smallest valid big-M);
    'naive'  : every big-M replaced by the horizon length (a common first choice);
    'noprec' : precedence rows dropped, mutual-exclusion rows x_a + x_b <= 1 kept;
    'bare'   : only at-most-one per target and the budgets."""
    opps = inst.opportunities
    n = len(opps)
    both = [p for p in inst.pairs if p.a_first_ok and p.b_first_ok]
    y_index = {(p.a, p.b): 2 * n + k for k, p in enumerate(both)}
    nv = 2 * n + len(both)
    rows, cols, vals, lo, hi = [], [], [], [], []

    def add_row(entries, lb, ub):
        r = len(lo)
        for c, v in entries:
            rows.append(r); cols.append(c); vals.append(v)
        lo.append(lb); hi.append(ub)

    for ids in inst.by_target().values():
        if len(ids) > 1:
            add_row([(i, 1.0) for i in ids], -np.inf, 1.0)
    H = float(inst.horizon_s)
    for p in inst.pairs:
        a, b = p.a, p.b
        oa, ob = opps[a], opps[b]
        gab, gba = oa.duration + p.s_ab, ob.duration + p.s_ba
        Mab = H if mode == "naive" else gab - ob.earliest_start + oa.latest_start
        Mba = H if mode == "naive" else gba - oa.earliest_start + ob.latest_start
        if not p.a_first_ok and not p.b_first_ok:
            if mode != "bare":
                add_row([(a, 1.0), (b, 1.0)], -np.inf, 1.0)
            continue
        if mode in ("noprec", "bare"):
            continue
        if p.a_first_ok and not p.b_first_ok:
            add_row([(n + b, 1.0), (n + a, -1.0), (a, -Mab), (b, -Mab)], gab - 2 * Mab, np.inf)
        elif p.b_first_ok and not p.a_first_ok:
            add_row([(n + a, 1.0), (n + b, -1.0), (a, -Mba), (b, -Mba)], gba - 2 * Mba, np.inf)
        else:
            y = y_index[(a, b)]
            add_row([(n + b, 1.0), (n + a, -1.0), (a, -Mab), (b, -Mab), (y, -Mab)], gab - 3 * Mab, np.inf)
            add_row([(n + a, 1.0), (n + b, -1.0), (a, -Mba), (b, -Mba), (y, Mba)], gba - 2 * Mba, np.inf)
    for _, limit, ids in inst.budget_groups():
        add_row([(i, float(opps[i].duration)) for i in ids], -np.inf, float(limit))
    A = coo_matrix((vals, (rows, cols)), shape=(len(lo), nv)).tocsr()
    c = np.zeros(nv)
    c[:n] = -np.array([o.weight for o in opps], dtype=float)
    lb, ub = np.zeros(nv), np.ones(nv)
    lb[n:2 * n] = [o.earliest_start for o in opps]
    ub[n:2 * n] = [o.latest_start for o in opps]
    return c, LinearConstraint(A, lo, hi), Bounds(lb, ub)


def lp_bound(c, cons, bounds, limit=300.0):
    t0 = time.perf_counter()
    res = milp(c, constraints=cons, bounds=bounds, integrality=np.zeros(len(c)),
               options={"time_limit": limit, "disp": False})
    dt = time.perf_counter() - t0
    if res.x is None:
        return None, None, dt
    return -res.fun, res.x, dt


def analyse(name, inst):
    n = len(inst.opportunities)
    opps = inst.opportunities
    st = inst.stats()
    out = {"stats": st}
    # 1. the size of milp.py's model, and the check that build_variant('tight') is milp.py
    c0, cons0, b0, integ = build_milp(inst)
    c1, cons1, b1 = build_variant(inst, "tight")
    assert cons0.A.shape == cons1.A.shape and abs(cons0.A - cons1.A).max() == 0
    one_way = sum(p.a_first_ok != p.b_first_ok for p in inst.pairs)
    out["size"] = {"vars": int(len(c0)), "binary": int(integ.sum()), "continuous": int(n), "y_vars": int(len(c0) - 2 * n),
                   "rows": int(cons0.A.shape[0]), "nonzeros": int(cons0.A.nnz), "one_way": int(one_way)}
    print(f"[{name}] opportunities {n}, pairs {st['conflict_pairs']} (two-way {st['pairs_both_orders']}, "
          f"one-way {one_way}, exclusive {st['pairs_mutually_exclusive']})")
    print(f"   MILP: {len(c0)} variables ({int(integ.sum())} binary: {n} x + {len(c0) - 2 * n} y; {n} continuous t), "
          f"{cons0.A.shape[0]} rows, {cons0.A.nnz} nonzeros")
    # 2. big-M sizes
    Ms = []
    for p in inst.pairs:
        oa, ob = opps[p.a], opps[p.b]
        if p.a_first_ok:
            Ms.append(oa.duration + p.s_ab - ob.earliest_start + oa.latest_start)
        if p.b_first_ok:
            Ms.append(ob.duration + p.s_ba - oa.earliest_start + ob.latest_start)
    Ms = np.array(Ms, dtype=float)
    out["bigM"] = {"count": int(len(Ms)), "median": float(np.median(Ms)), "max": float(Ms.max()),
                   "min": float(Ms.min()), "horizon": int(inst.horizon_s)}
    print(f"   smallest big-M per precedence row: median {np.median(Ms):.0f} s, min {Ms.min():.0f}, max {Ms.max():.0f}; "
          f"horizon {inst.horizon_s} s ({inst.horizon_s / np.median(Ms):.0f} x the median)")
    # 3. LP relaxations
    lps = {}
    for mode in ("tight", "naive", "noprec", "bare"):
        cc, cs, bb = build_variant(inst, mode)
        val, x, dt = lp_bound(cc, cs, bb)
        lps[mode] = {"bound": None if val is None else round(val, 4), "time_s": round(dt, 2)}
        print(f"   LP relaxation [{mode:6s}]: {('%.2f' % val) if val is not None else 'none'}  ({dt:.1f} s)")
        if mode == "tight" and x is not None:
            xs = x[:n]
            frac = int(np.sum((xs > 1e-6) & (xs < 1 - 1e-6)))
            half = int(np.sum(np.abs(xs - 0.5) < 1e-6))
            ones = int(np.sum(xs > 1 - 1e-6))
            # precedence rows with x_a + x_b <= 1 at the LP optimum are fully switched off
            off = sum(1 for p in inst.pairs if (p.a_first_ok or p.b_first_ok) and xs[p.a] + xs[p.b] <= 1 + 1e-6)
            act = sum(1 for p in inst.pairs if (p.a_first_ok or p.b_first_ok))
            sel = float(np.sum(xs))
            lps["tight"].update({"fractional": frac, "half": half, "ones": ones, "pairs_off": off, "pairs_prec": act,
                                 "sum_x": round(sel, 2)})
            hist = np.histogram(xs, bins=10, range=(0, 1))[0]
            lps["tight"]["hist"] = [int(h) for h in hist]
            print(f"      LP optimum: {frac} of {n} x fractional ({half} exactly 1/2), {ones} at 1; "
                  f"sum of x = {sel:.2f}; precedence pairs with x_a + x_b <= 1 (row switched off): {off} of {act}")
    out["lp"] = lps
    return out


def pick_pair(inst):
    """A two-way pair of the small instance with long, overlapping windows, for the widget."""
    opps = inst.opportunities
    best = None
    for p in inst.pairs:
        if not (p.a_first_ok and p.b_first_ok):
            continue
        oa, ob = opps[p.a], opps[p.b]
        la, lb = oa.latest_start - oa.earliest_start, ob.latest_start - ob.earliest_start
        ov = min(oa.window_end, ob.window_end) - max(oa.window_start, ob.window_start)
        score = min(la, lb) + 0.5 * ov
        if best is None or score > best[0]:
            best = (score, p)
    p = best[1]
    oa, ob = opps[p.a], opps[p.b]
    gab, gba = oa.duration + p.s_ab, ob.duration + p.s_ba
    d = {"a": p.a, "b": p.b, "sat": oa.sat, "ta": oa.target, "tb": ob.target,
         "ea": oa.earliest_start, "la": oa.latest_start, "eb": ob.earliest_start, "lb": ob.latest_start,
         "da": oa.duration, "db": ob.duration, "sab": p.s_ab, "sba": p.s_ba, "gab": gab, "gba": gba,
         "M1": gab - ob.earliest_start + oa.latest_start, "M2": gba - oa.earliest_start + ob.latest_start,
         "wa": oa.weight, "wb": ob.weight}
    print(f"   widget pair: opps {p.a} and {p.b} on satellite {oa.sat} (targets {oa.target}, {ob.target}): "
          f"a starts in [{d['ea']}, {d['la']}], b in [{d['eb']}, {d['lb']}], durations {d['da']}, {d['db']} s, "
          f"transitions s_ab {p.s_ab}, s_ba {p.s_ba} s -> g_ab {gab}, g_ba {gba}; M1 {d['M1']}, M2 {d['M2']}")
    return d


def benchmark():
    path = REPO / "results" / "benchmark.json"
    d = json.loads(path.read_text())
    recs = []
    for r in d["records"]:
        res = {x["method"]: x for x in r["results"]}
        g, c, m, rh = res["greedy"], res["cpsat+hint"], res["milp-highs"], res["rolling-cpsat-60min"]
        recs.append({"size": r["size"], "seed": r["seed"], "opps": r["stats"]["opportunities"],
                     "pairs": r["stats"]["conflict_pairs"], "limit": r["time_limit_s"],
                     "greedy": g["objective"], "cpsat": c["objective"], "rolling": rh["objective"], "milp": m["objective"],
                     "cpsat_bound": round(c["bound"], 3), "milp_bound": round(m["bound"], 3) if m["bound"] is not None else None,
                     "t_greedy": round(g["time_s"], 2), "t_cpsat": round(c["time_s"], 4),  # 4 decimals: the page's
                     "t_rolling": round(rh["time_s"], 2), "t_milp": round(m["time_s"], 4),  # ratios round correctly
                     "cpsat_status": c["status"], "milp_status": m["status"]})
    print("[benchmark.json, cited]")
    for r in recs:
        ratio = r["t_milp"] / r["t_cpsat"] if r["size"] == "small" else None
        print(f"   {r['size']} {r['seed']}: greedy {r['greedy']}, CP-SAT {r['cpsat']} (bound {r['cpsat_bound']:.0f}, "
              f"{r['t_cpsat']} s), MILP {r['milp']} (bound {r['milp_bound']:.2f}, {r['t_milp']} s, {r['milp_status']})"
              + (f", MILP/CP-SAT time ratio {ratio:.1f}" if ratio else ""))
    return recs


def main():
    data = {}
    insts = {}
    for name, kw in SIZES.items():
        inst, _ = build_instance(ScenarioConfig(seed=0, **kw))
        insts[name] = inst
        data[name] = analyse(name, inst)
        if name == "small":
            data["pair"] = pick_pair(inst)
    # one MILP solve of the small instance
    r = solve_milp(insts["small"], 60.0)
    assert not check_plan(insts["small"], r["plan"])
    data["small"]["milp_solve"] = {"value": r["objective"], "bound": r["bound"], "time_s": round(r["time_s"], 1),
                                   "status": r["status"]}
    print(f"[small] MILP solve: value {r['objective']}, bound {r['bound']:.2f}, {r['status']}, {r['time_s']:.1f} s "
          f"(benchmark run: 8.8 s); plan passes the checker")
    # the same MILP with every big-M equal to the horizon (exercise 2)
    cn, csn, bn = build_variant(insts["small"], "naive")
    integ = np.ones(len(cn)); integ[len(insts["small"].opportunities):2 * len(insts["small"].opportunities)] = 0
    t0 = time.perf_counter()
    rn = milp(cn, constraints=csn, bounds=bn, integrality=integ, options={"time_limit": 120.0, "disp": False})
    dtn = time.perf_counter() - t0
    data["small"]["milp_naive"] = {"value": int(round(-rn.fun)), "status": int(rn.status), "time_s": round(dtn, 1)}
    print(f"[small] MILP with M = horizon: value {-rn.fun:.0f}, status {rn.status} (0 = optimal), {dtn:.1f} s "
          f"(tight M in the same script: {r['time_s']:.1f} s)")
    for name in ("small", "medium"):
        inst = insts[name]
        # clique inequalities (exercise 3): opportunities pairwise incompatible (same target, or neither order fits)
        adj = {i: set() for i in range(len(inst.opportunities))}
        for ids in inst.by_target().values():
            for i in ids:
                adj[i].update(j for j in ids if j != i)
        for p in inst.pairs:
            if not p.a_first_ok and not p.b_first_ok:
                adj[p.a].add(p.b); adj[p.b].add(p.a)
        w = [o.weight for o in inst.opportunities]
        cliques = set()
        for v in range(len(adj)):
            cl = [v]
            for u in sorted(adj[v], key=lambda u: (-w[u], u)):
                if all(u in adj[q] for q in cl):
                    cl.append(u)
            if len(cl) >= 3:
                cliques.add(tuple(sorted(cl)))
        cc, cs, bb = build_variant(inst, "tight")
        n = len(inst.opportunities)
        extra = coo_matrix(([1.0] * sum(len(q) for q in cliques),
                            ([k for k, q in enumerate(cliques) for _ in q], [i for q in cliques for i in q])),
                           shape=(len(cliques), len(cc))).tocsr()
        A2 = vstack([cs.A, extra]).tocsr()
        cs2 = LinearConstraint(A2, np.concatenate([cs.lb, -np.inf * np.ones(len(cliques))]),
                               np.concatenate([cs.ub, np.ones(len(cliques))]))
        val, _, _ = lp_bound(cc, cs2, bb)
        data[name]["lp"]["cliques"] = {"bound": round(val, 4), "count": len(cliques)}
        # a time-indexed formulation would need one binary per (opportunity, start second)
        ti1 = sum(o.latest_start - o.earliest_start + 1 for o in inst.opportunities)
        ti10 = sum((o.latest_start - o.earliest_start) // 10 + 1 for o in inst.opportunities)
        data[name]["time_indexed"] = {"per_second": ti1, "per_10s": ti10}
        print(f"[{name}] LP with {len(cliques)} greedy clique inequalities: {val:.2f} "
              f"(without: {data[name]['lp']['tight']['bound']:.2f}); time-indexed binaries: {ti1} on a 1 s grid, "
              f"{ti10} on a 10 s grid ({n} opportunities)")
    # the integer optimum with the precedence rows deleted: how much of the LP-to-optimum distance is reconfiguration
    s_inst = insts["small"]; ns = len(s_inst.opportunities)
    cp_, csp, bp = build_variant(s_inst, "noprec")
    integ_p = np.ones(len(cp_)); integ_p[ns:2 * ns] = 0
    rp = milp(cp_, constraints=csp, bounds=bp, integrality=integ_p, options={"time_limit": 60.0, "disp": False})
    print(f"[small] integer optimum with the precedence rows deleted: {-rp.fun:.0f} (status {rp.status}); "
          f"so of LP {data['small']['lp']['tight']['bound']:.2f} - 219, {-rp.fun - 219:.0f} points are reconfiguration")
    lp, opt = data["small"]["lp"]["tight"]["bound"], 219
    print(f"[small] LP bound {lp:.2f} vs optimum {opt}: {100 * (lp - opt) / opt:.1f} % above; floor (integer weights) {int(np.floor(lp + 1e-9))}")
    lpm = data["medium"]["lp"]["tight"]["bound"]
    print(f"[medium] LP bound {lpm:.2f} vs CP-SAT bound 575 and best value 557: {100 * (lpm - 575) / 575:.1f} % above the bound")
    data["bench"] = benchmark()
    OUT.write_text("/* Satellite Pass Planning from Scratch, chapter 09: LP relaxations, model sizes, a real pair and the\n"
                   "   benchmark's numbers, exported by code/ch09_numbers.py. */\nwindow.SAT_CH09 = "
                   + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
