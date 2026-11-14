"""Reproduce every number quoted on chapter B2 of "Which Job Should the Machine Do Next?",
"Dispatching rules in a dynamic shop" (b02-dispatching-rules-in-a-dynamic-shop.html).

Usage:  python b02_numbers.py          (about 15 s)
Needs numpy, simpy and the repository's package `jobshop` (set JOBSHOP_REPO to the repository,
or run from the course zip, where jobshop/ sits next to code/).

The benchmark numbers come from the repository's results/runs.csv (never re-run here). The short
simulations below (rule runs, 20 days each) re-create those runs exactly and measure what the CSV
does not record: queue lengths at decisions, reroutes after breakdowns, the effect of switching
rerouting off, and an ATCS controller written for exercise 4. ATCS is NOT part of the repository.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JOBSHOP_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

from jobshop.checker import check_schedule  # noqa: E402
from jobshop.metrics import bootstrap_ci, run_metrics  # noqa: E402
from jobshop.rules import RULES, RuleController, route_earliest_finish, workload  # noqa: E402
from jobshop.scenario import LEVELS, make_scenario, make_workshop  # noqa: E402
from jobshop.shop import Shop  # noqa: E402

WS = make_workshop()
LV = list(LEVELS)                      # ["no breakdowns", "frequent breakdowns"]
RULE_NAMES = ["FIFO", "SPT", "EDD", "SST"]


def section(title):
    print(f"\n=== {title}")


# ---------------------------------------------------------------------------------------------
# One machine: the two queues of the hero (the paint machine of notation.md, six orders on m2)
# ---------------------------------------------------------------------------------------------
# name, family, processing time, due date, queue position (FIFO key)
PAINT = dict(start=0, orders=[("A", 0, 20, 30, 1), ("B", 1, 10, 25, 2), ("C", 0, 15, 70, 3), ("D", 1, 25, 60, 4)],
             setup=lambda f, g: 0 if f == g else 10, names={0: "blue", 1: "red"})
M2 = WS.setup[2]
SIX = dict(start=2, orders=[("J0", 0, 26, 60, 1), ("J1", 1, 22, 40, 2), ("J2", 2, 19, 95, 3),
                            ("J3", 0, 8, 45, 4), ("J4", 3, 20, 70, 5), ("J5", 1, 15, 110, 6)],
           setup=lambda f, g: M2[f][g], names={f: f"family {f}" for f in range(4)})
K_VALUES = [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10, 100]      # the hero's slider positions


def run(inst, seq):
    """Timing as in the simulator: changeover, then processing; tardiness max(0, C - d)."""
    t, fam, rows = 0, inst["start"], []
    for o in seq:
        name, f, p, d, _ = o
        s = inst["setup"](fam, f)
        rows.append((name, t, t + s, t + s + p, max(0, t + s + p - d)))
        t, fam = t + s + p, f
    return rows


def total(rows):
    return sum(r[4] for r in rows)


def averages(inst):
    """p-bar: mean processing time of the orders; s-bar: mean changeover over ordered pairs of distinct orders."""
    o = inst["orders"]
    pbar = float(np.mean([x[2] for x in o]))
    sbar = float(np.mean([inst["setup"](a[1], b[1]) for a in o for b in o if a is not b]))
    return pbar, sbar


def index(inst, rule, o, t, fam, k1=1.0, k2=1.0):
    """Priority of order o when the machine is free at t, set up for fam. Smallest wins, except ATCS (largest)."""
    name, f, p, d, pos = o
    s = inst["setup"](fam, f)
    if rule == "FIFO":
        return (pos,)
    if rule == "SPT":
        return (p, pos)
    if rule == "EDD":
        return (d, pos)
    if rule == "SST":
        return (s, d, pos)                       # ties by due date, as jobshop/rules.py:_sst
    pbar, sbar = averages(inst)
    return (p and (1 / p) * math.exp(-max(d - p - t, 0) / (k1 * pbar)) * math.exp(-s / (k2 * sbar)),)


def dispatch(inst, rule, k1=1.0, k2=1.0, trace=False):
    waiting, seq, t, fam = list(inst["orders"]), [], 0, inst["start"]
    while waiting:
        keys = {o[0]: index(inst, rule, o, t, fam, k1, k2) for o in waiting}
        if trace:
            print(f"  t = {t:3d}, set up for {inst['names'][fam]}:",
                  ", ".join(f"{n} {v[0]:.5f}" if rule == "ATCS" else f"{n} {v}" for n, v in keys.items()))
        o = (max(waiting, key=lambda o: (keys[o[0]], -o[4])) if rule == "ATCS"
             else min(waiting, key=lambda o: keys[o[0]]))
        waiting.remove(o)
        seq.append(o)
        s = inst["setup"](fam, o[1])
        t, fam = t + s + o[2], o[1]
    return seq


def best(inst):
    return min(itertools.permutations(inst["orders"]), key=lambda s: (total(run(inst, s)), "".join(x[0] for x in s)))


section("Part 2 and hero: the paint machine, indices at t = 0 (machine set up for blue)")
for rule in RULE_NAMES:
    print(f"{rule}: indices at t = 0", {o[0]: index(PAINT, rule, o, 0, 0) for o in PAINT["orders"]})
for inst_name, inst in (("paint machine", PAINT), ("six orders on m2, set up for family 2", SIX)):
    print(f"-- {inst_name}")
    for rule in RULE_NAMES:
        seq = dispatch(inst, rule)
        rows = run(inst, seq)
        print(f"  {rule:5s} order {' '.join(x[0] for x in seq)}  total tardiness {total(rows)}"
              f"  changeover minutes {sum(r[2] - r[1] for r in rows)}  per order (name, completion, tardiness)",
              [(r[0], r[3], r[4]) for r in rows])
    b = best(inst)
    n_orders = math.factorial(len(inst["orders"]))
    print(f"  best of all {n_orders} orders: {' '.join(x[0] for x in b)}  total tardiness {total(run(inst, b))}")
    pbar, sbar = averages(inst)
    print(f"  p-bar = {pbar:.4f}, s-bar = {sbar:.4f}")
pairs = [(a, b) for a in PAINT["orders"] for b in PAINT["orders"] if a is not b]
print("paint: ordered pairs of distinct orders", len(pairs), "of which with a colour change",
      sum(PAINT["setup"](a[1], b[1]) > 0 for a, b in pairs))
print("six-order changeover matrix of m2 (from, to):", M2)
print("Part 6: changeover minutes, six orders: SST", sum(r[2] - r[1] for r in run(SIX, dispatch(SIX, "SST"))),
      " EDD", sum(r[2] - r[1] for r in run(SIX, dispatch(SIX, "EDD"))),
      " best order", sum(r[2] - r[1] for r in run(SIX, best(SIX))),
      " (changeovers of the best order:", [r[2] - r[1] for r in run(SIX, best(SIX))], ")")

section("Exercise 1: a static index dispatched with every job present is a sort; a dynamic one need not be")
for inst_name, inst in (("paint", PAINT), ("six orders", SIX)):
    for rule in ("FIFO", "SPT", "EDD", "SST", "ATCS"):
        srt = sorted(inst["orders"], key=lambda o: index(inst, rule, o, 0, inst["start"]),
                     reverse=(rule == "ATCS"))
        dis = dispatch(inst, rule)
        print(f"  {inst_name:10s} {rule:5s} sorted by the index at t = 0: {' '.join(x[0] for x in srt)}"
              f" (total {total(run(inst, srt))}); dispatched: {' '.join(x[0] for x in dis)} (total {total(run(inst, dis))});"
              f" equal: {srt == dis}")

section("Part 7: ATCS on the paint machine by hand, k1 = k2 = 1 (k1 p-bar = 17.5, k2 s-bar = 6.667)")
seq = dispatch(PAINT, "ATCS", 1, 1, trace=True)
rows = run(PAINT, seq)
print("  order", "".join(x[0] for x in seq), " completions", [r[3] for r in rows], " tardiness", [r[4] for r in rows],
      " total", total(rows))
print("  factors at t = 0: exp(-10/17.5) =", round(math.exp(-10 / 17.5), 4), " exp(-15/17.5) =", round(math.exp(-15 / 17.5), 4),
      " exp(-55/17.5) =", round(math.exp(-55 / 17.5), 4), " exp(-35/17.5) =", round(math.exp(-35 / 17.5), 4),
      " exp(-10/6.667) =", round(math.exp(-10 / (20 / 3)), 4))
print("  exercise 3, k1 = 2, k2 = 1: k1 p-bar =", 2 * 17.5, "; C at t = 20: exp(-35/35) =", round(math.exp(-1), 4),
      "; A at t = 0: exp(-10/35) =", round(math.exp(-10 / 35), 4))
seq = dispatch(PAINT, "ATCS", 2, 1, trace=True)
print("  order", "".join(x[0] for x in seq), " total", total(run(PAINT, seq)))

section("Part 7 and hero: ATCS over the slider grid (rows k1, columns k2), order and total tardiness")
for inst_name, inst in (("paint", PAINT), ("six orders", SIX)):
    print(f"-- {inst_name}; k2 =", K_VALUES)
    for k1 in K_VALUES:
        cells = []
        for k2 in K_VALUES:
            s = dispatch(inst, "ATCS", k1, k2)
            cells.append(f"{''.join(x[0][-1] for x in s)}:{total(run(inst, s))}")
        print(f"  k1 = {k1:<5}", " ".join(cells))
    rows = run(inst, dispatch(inst, "ATCS", 1, 1))
    print(f"  hero default k1 = k2 = 1: changeover minutes {sum(r[2] - r[1] for r in rows)}, total tardiness {total(rows)}")

# ---------------------------------------------------------------------------------------------
# The shop: runs.csv, and short simulations that re-create the rule runs
# ---------------------------------------------------------------------------------------------
ROWS = list(csv.DictReader(open(REPO / "results" / "runs.csv", encoding="utf-8")))


def col(level, method, key):
    r = sorted((x for x in ROWS if x["level"] == level and x["method"] == method), key=lambda x: int(x["seed"]))
    return np.array([float(x[key]) for x in r])


section("Part 4: the four rules on the 20 test days (results/runs.csv), mean [95 % bootstrap interval]")
for level in LV:
    print(f"-- {level}")
    for m in RULE_NAMES:
        mt, mx, su, ts = (col(level, m, k) for k in ("mean_tardiness", "max_tardiness", "setup_time", "tardy_share"))
        lo, hi = bootstrap_ci(mt)
        lo2, hi2 = bootstrap_ci(mx)
        print(f"  {m:5s} mean tardiness {mt.mean():5.1f} [{lo:.1f}, {hi:.1f}]  max tardiness {mx.mean():4.0f} [{lo2:.0f}, {hi2:.0f}]"
              f"  changeover minutes {su.mean():5.0f}  late share {100 * ts.mean():4.1f} %  makespan {col(level, m, 'makespan').mean():.0f}")
    for m in ("Greedy replan", "CP-SAT"):
        print(f"  {m}: mean tardiness {col(level, m, 'mean_tardiness').mean():.1f}, changeover minutes {col(level, m, 'setup_time').mean():.0f}")
    days_best = {m: 0 for m in RULE_NAMES}
    for i in range(20):
        days_best[min(RULE_NAMES, key=lambda m: col(level, m, "mean_tardiness")[i])] += 1
    print("  days on which each rule is the best of the four:", days_best)
for level in LV:
    d = col(level, "SST", "mean_tardiness") - col(level, "EDD", "mean_tardiness")
    lo, hi = bootstrap_ci(d)
    dmax = col(level, "SST", "max_tardiness") - col(level, "EDD", "max_tardiness")
    lo3, hi3 = bootstrap_ci(dmax)
    print(f"{level}: SST - EDD mean tardiness {d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}], SST better on {(d < 0).sum()}, worse on {(d > 0).sum()} of 20 days;"
          f" max tardiness {dmax.mean():+.0f} [{lo3:+.0f}, {hi3:+.0f}]")
    print(f"  changeover minutes saved by SST against EDD: {(col(level, 'EDD', 'setup_time') - col(level, 'SST', 'setup_time')).mean():.0f}")
# Part 4, point 2: the arrivals do not depend on the breakdown level (scenario.py), so the change of
# SST - EDD between the two levels is itself a paired difference over the same 20 seeds
dd = ((col(LV[1], "SST", "mean_tardiness") - col(LV[1], "EDD", "mean_tardiness"))
      - (col(LV[0], "SST", "mean_tardiness") - col(LV[0], "EDD", "mean_tardiness")))
lo, hi = bootstrap_ci(dd)
print(f"change of SST - EDD from no breakdowns to frequent breakdowns, paired over the seeds: {dd.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]")
for m in RULE_NAMES:
    a, b = col(LV[0], m, "mean_tardiness").mean(), col(LV[1], m, "mean_tardiness").mean()
    print(f"{m}: breakdowns multiply mean tardiness by {b / a:.1f} ({a:.1f} -> {b:.1f})")

# per-day data of the results widget (pasted into interactive/b02.js as DAYS)
DAYS = {lv: {m: {k: [round(v, 3) for v in col(lv, m, k)] for k in ("mean_tardiness", "max_tardiness", "setup_time")}
             for m in RULE_NAMES} for lv in LV}
print("DAYS (for b02.js) =", json.dumps(DAYS, separators=(",", ":")))


class Instrumented(RuleController):
    """The repository's RuleController, unchanged, plus counters; `reroute=False` switches rerouting off."""

    def __init__(self, rule, reroute=True):
        super().__init__(rule)
        self.allow_reroute = reroute
        self.queue_at_decision, self.reroutes, self.checked = [], 0, 0

    def sequence(self, shop, m):
        self.queue_at_decision.append(len(m.queue))
        return super().sequence(shop, m)

    def on_event(self, shop, kind, **info):
        if kind != "breakdown" or not self.allow_reroute:
            return
        m = info["machine"]
        for jr in list(m.queue):
            self.checked += 1
            target = route_earliest_finish(shop, jr)
            if target != m.idx:
                shop.reroute(jr, target)
                self.reroutes += 1


def simulate(level, seed, controller):
    sc = make_scenario(WS, seed, mtbf=LEVELS[level])
    log = Shop(WS, sc, controller).run()
    return sc, log, run_metrics(WS, sc, log)


section("Parts 1, 3 and 5: the rule runs re-simulated (they match runs.csv exactly), with counters")
STATS = {}
for level in LV:
    for m in RULE_NAMES:
        q, rr, chk, mt, mt_off, proc, setup = [], 0, 0, [], [], [], []
        for seed in range(20):
            c = Instrumented(m)
            sc, log, met = simulate(level, seed, c)
            assert abs(met["mean_tardiness"] - col(level, m, "mean_tardiness")[seed]) < 1e-9
            q += c.queue_at_decision
            rr += c.reroutes
            chk += c.checked
            mt.append(met["mean_tardiness"])
            proc.append(sum(s.end - s.start for s in log.segments if s.kind == "proc"))
            setup.append(met["setup_time"])
            mt_off.append(simulate(level, seed, Instrumented(m, reroute=False))[2]["mean_tardiness"])
        q = np.array(q)
        STATS[level, m] = dict(proc_per_job=np.mean(proc) / 60, setup_per_job=np.mean(setup) / 60)
        print(f"{level:19s} {m:5s} decisions per day {len(q) / 20:3.0f}; with >= 2 jobs waiting {100 * (q >= 2).mean():4.1f} %"
              f" (one job only: {100 * (q < 2).mean():4.1f} %); mean queue at a decision {q.mean():.2f};"
              f" reroutes per day {rr / 20:4.1f} ({chk / 20:.1f} queued jobs re-examined);"
              f" mean tardiness {np.mean(mt):.1f}, without rerouting {np.mean(mt_off):.1f}"
              f" (rerouting worth {np.mean(mt_off) - np.mean(mt):.1f} min)")
print("all rule runs re-created exactly from the repository's code")

section("Part 3: the first reroute on test day 0, frequent breakdowns, shortest setup first")


class Spy(RuleController):
    found = None

    def on_event(self, shop, kind, **info):
        if kind == "breakdown" and Spy.found is None:
            m = info["machine"]
            for jr in m.queue:
                target = route_earliest_finish(shop, jr)
                if target != m.idx:
                    rows = []
                    for mm, p in WS.alternatives(jr.family, jr.op).items():
                        M = shop.machines[mm]
                        tr = WS.travel[jr.station][WS.station(mm)]
                        free = shop.expected_free(M) + workload(shop, M, skip=jr)
                        rows.append(dict(machine=mm, travel=tr, arrival=shop.now + tr, down=M.down,
                                         expected_up=shop.expected_up(M), remaining_current=shop.remaining_current(M),
                                         committed=workload(shop, M, skip=jr), free=free, p=p,
                                         key=max(shop.now + tr, free) + p,
                                         queue=[(j.idx, j.family, j.op) for j in M.queue if j is not jr],
                                         inbound=[(j.idx, j.family, j.op, WS.proc(j.family, j.op, mm)) for j in M.inbound]))
                    Spy.found = dict(t=shop.now, failed=m.idx, job=jr.idx, family=jr.family, op=jr.op, due=jr.due,
                                     station=jr.station, target=target, rows=rows)
                    break
        super().on_event(shop, kind, **info)


sc = make_scenario(WS, 0, mtbf=120.0)
Shop(WS, sc, Spy("SST")).run()
f = Spy.found
print(f"minute {f['t']}: machine m{f['failed']} fails; job {f['job']} (family {f['family']}, operation {f['op']}, due {f['due']})"
      f" waits in its queue at station {f['station']}; moved to m{f['target']}")
for r in f["rows"]:
    print("  ", r)
repair = [s for s in sc.downtime[f["failed"]] if s[0] == f["t"]]
real_up = repair[0][1]
print("  the real repair:", repair, "(the controller assumes breakdown + MTTR = ", f["t"] + 20, ")")
r1 = [r for r in f["rows"] if r["machine"] == f["target"]][0]
r4 = [r for r in f["rows"] if r["machine"] == f["failed"]][0]
print(f"  moving finishes {r4['key'] - r1['key']} min earlier by the estimate")
print(f"  routing widget: stay key = t + MTTR + {r4['remaining_current']} + {r4['committed']} + {r4['p']}"
      f" = {f['t']} + MTTR + {r4['remaining_current'] + r4['committed'] + r4['p']}"
      f" (with the real repair end {real_up}: max({f['t']}, {real_up} + {r4['remaining_current']} + {r4['committed']}) + {r4['p']} = {real_up + r4['remaining_current'] + r4['committed'] + r4['p']});"
      f" move key = max({r1['arrival']}, {f['t']} + R) + {r1['p']}; with MTTR = 20 the job moves while R < "
      f"{f['t'] + 20 + r4['remaining_current'] + r4['committed'] + r4['p'] - r1['p'] - f['t']}")
stay_const = f["t"] + r4["remaining_current"] + r4["committed"] + r4["p"]
move_const = f["t"] + r1["p"]
print(f"  routing widget threshold: stay = {stay_const} + MTTR; move = {r1['arrival'] + r1['p']} if R <= {r1['travel']}, else {move_const} + R;"
      f" the job moves iff R < {stay_const - move_const} + MTTR; work counted ahead on m4: {r4['remaining_current'] + r4['committed']} min")

section("Part 5: why shortest setup first gains under breakdowns, a rough load calculation")
lam_inv = 20.83
A = 120 / 140
print(f"mean gap between arrivals {lam_inv} min, 5 machines: {5 * lam_inv / 60 * 60:.2f} machine-minutes per job; availability A = {A:.3f}, downtime {100 * (1 - A):.0f} %")
fr = float(np.mean([WS.min_remaining_work(f, 0) for f in range(4)]))
print("mean fastest-route work of a job (uniform family mix):", fr, f"-> the course's load rho = {fr:.1f} / {5 * lam_inv:.2f} = {fr / (5 * lam_inv):.2f}")
print("processing per job, every rule (FIFO is the one chapter B1 quotes):",
      {f"{lv} {m}": round(float(STATS[lv, m]["proc_per_job"]), 1) for lv in LV for m in RULE_NAMES})
LOADS = {}
for level in LV:
    for m in ("EDD", "SST"):
        s = STATS[level, m]
        busy = (s["proc_per_job"] + s["setup_per_job"]) / (5 * lam_inv)
        rho = busy / (A if level == LV[1] else 1.0)
        LOADS[level, m] = rho / (1 - rho)
        print(f"  {level:19s} {m}: processing {s['proc_per_job']:.1f} + changeover {s['setup_per_job']:.1f} min per job"
              f" -> effective load rho_eff {rho:.3f}, rho_eff/(1-rho_eff) = {rho / (1 - rho):.2f}")

for level in LV:
    print(f"  {level}: waiting factor EDD / SST = {LOADS[level, 'EDD'] / LOADS[level, 'SST']:.2f}")

section("Part 5: the M/M/1 derivation, evaluated at the effective loads (a qualitative model, not the shop)")
# birth-death balance: lambda pi_n = mu pi_{n+1}  ->  pi_n = (1 - rho) rho^n,  L = rho / (1 - rho)
for rho in (0.5, 0.8):
    pis = [(1 - rho) * rho ** n for n in range(2000)]
    print(f"  check by summation, rho = {rho}: sum pi_n = {sum(pis):.6f}, sum n pi_n = {sum(n * x for n, x in enumerate(pis)):.6f},"
          f" rho/(1-rho) = {rho / (1 - rho):.6f}")
for level in LV:
    rho_e = LOADS[level, "EDD"] / (1 + LOADS[level, "EDD"])          # back from rho/(1-rho) to rho
    rho_s = LOADS[level, "SST"] / (1 + LOADS[level, "SST"])
    saved = STATS[level, "EDD"]["setup_per_job"] - STATS[level, "SST"]["setup_per_job"]
    dL = LOADS[level, "EDD"] - LOADS[level, "SST"]
    print(f"  {level}: changeover saved by SST {saved:.1f} min per job -> load lowered by {rho_e - rho_s:.3f}"
          f" ({rho_e:.3f} -> {rho_s:.3f}); L = rho/(1-rho) falls by {dL:.2f} jobs, {dL / saved:.2f} jobs per minute saved per job;"
          f" slope dL/drho = 1/(1-rho)^2 at EDD's load {1 / (1 - rho_e) ** 2:.1f}")
# Step 5 split: the load drop from saved changeovers alone is lambda * Delta s / (5 A); the rest of the table's gap
# rho_e - rho_s comes from the small difference in processing per job between EDD and SST (both print as 67.1 / 67.4)
for level in LV:
    Alv = A if level == LV[1] else 1.0
    saved = STATS[level, "EDD"]["setup_per_job"] - STATS[level, "SST"]["setup_per_job"]
    dproc = STATS[level, "EDD"]["proc_per_job"] - STATS[level, "SST"]["proc_per_job"]
    rho_e = LOADS[level, "EDD"] / (1 + LOADS[level, "EDD"])
    rho_s = LOADS[level, "SST"] / (1 + LOADS[level, "SST"])
    d_setup = saved / (5 * lam_inv * Alv)
    d_proc = dproc / (5 * lam_inv * Alv)
    rho_c = rho_e - d_setup                                            # EDD's load with only the changeovers saved
    dL_setup = rho_e / (1 - rho_e) - rho_c / (1 - rho_c)
    print(f"  Step 5 split, {level}: Delta s = {saved:.2f} min per job, changeover-only drop lambda*Delta s/(5A) ="
          f" {saved:.2f}/({5 * lam_inv:.2f}*{Alv:.3f}) = {d_setup:.3f}; processing difference EDD - SST {dproc:+.2f} min per job"
          f" -> {d_proc:+.4f}; sum {d_setup + d_proc:.4f} = table gap {rho_e - rho_s:.4f}")
    print(f"    changeovers alone: rho_eff {rho_e:.3f} -> {rho_c:.3f}, L {rho_e / (1 - rho_e):.2f} -> {rho_c / (1 - rho_c):.2f},"
          f" falls by {dL_setup:.2f} jobs, {dL_setup / saved:.2f} jobs per minute saved per job")
slope = [1 / (1 - LOADS[lv, "EDD"] / (1 + LOADS[lv, "EDD"])) ** 2 for lv in LV]
print(f"  the same saved minute is worth {slope[1] / slope[0]:.1f} times more at the breakdown load (ratio of the slopes)")
for level in LV:
    fe, fs = col(level, "EDD", "mean_flow_time").mean(), col(level, "SST", "mean_flow_time").mean()
    print(f"  measured (runs.csv), {level}: mean flow time EDD {fe:.1f}, SST {fs:.1f} min; gap {fe - fs:.1f} min; ratio {fe / fs:.2f}")

section("Part 6: what a non-delay rule cannot do (paint numbers)")
# at minute 0 only B (red, 10 min, due 25) waits, the machine is set up for blue; A (blue, 20 min, due 30) arrives at minute 1
start_now = dict(B=(0 + 10, 0 + 10 + 10), A=(20 + 10, 20 + 10 + 20))
wait = dict(A=(1, 21), B=(21 + 10, 21 + 10 + 10))
for name, plan in (("start B now (every rule)", start_now), ("wait one minute for A", wait)):
    late = {k: max(0, v[1] - {"A": 30, "B": 25}[k]) for k, v in plan.items()}
    print(f"  {name}: completions { {k: v[1] for k, v in plan.items()} }, tardiness {late}, total {sum(late.values())}")

# ---------------------------------------------------------------------------------------------
# Exercise 4: an ATCS controller for the shop (a sketch; not in the repository)
# ---------------------------------------------------------------------------------------------
SBAR = [float(np.mean([WS.setup[m][f][g] for f in range(4) for g in range(4) if f != g])) for m in range(5)]


class ATCSController(RuleController):
    """Earliest-finish routing and rerouting as RuleController; sequencing by the ATCS index."""

    def __init__(self, k1, k2):
        super().__init__("SST")
        self.name, self.k1, self.k2 = "ATCS", k1, k2

    def sequence(self, shop, m):
        q = m.queue
        if len(q) == 1:
            return q[0]
        p = {j.idx: shop.ws.proc(j.family, j.op, m.idx) for j in q}
        pbar = float(np.mean(list(p.values())))

        def I(j):
            rest = shop.ws.min_remaining_work(j.family, j.op + 1)       # fastest work after this operation
            slack = j.due - rest - p[j.idx] - shop.now
            s = shop.ws.setup_time(m.idx, m.last_family, j.family)
            return math.exp(-max(slack, 0) / (self.k1 * pbar)) * math.exp(-s / (self.k2 * SBAR[m.idx])) / p[j.idx]

        return max(q, key=lambda j: (I(j), -j.idx))


section("Exercise 4: ATCS in the shop; s-bar per machine = mean off-diagonal changeover")
print("  s-bar per machine:", [round(s, 3) for s in SBAR])
GRID1, GRID2 = [0.5, 1, 2, 3, 5], [0.25, 0.5, 1, 2, 5]
VAL = range(5000, 5010)
score = {}
for k1 in GRID1:
    for k2 in GRID2:
        v = [np.mean([simulate(lv, s, ATCSController(k1, k2))[2]["mean_tardiness"] for s in VAL]) for lv in LV]
        score[k1, k2] = float(np.mean(v))
kbest = min(score, key=score.get)
print("  validation score (mean of the two levels, seeds 5000-5009) per (k1, k2):",
      {f"{k[0]},{k[1]}": round(v, 2) for k, v in score.items()})
print(f"  chosen on validation: k1 = {kbest[0]}, k2 = {kbest[1]} (score {score[kbest]:.2f});"
      f" spread of the 25 scores {min(score.values()):.2f} to {max(score.values()):.2f}")
for lv, ref in zip(LV, ("EDD", "SST")):
    vals, errors = [], 0
    for s in range(20):
        sc, log, met = simulate(lv, s, ATCSController(*kbest))
        errors += len(check_schedule(WS, sc, log))
        vals.append(met["mean_tardiness"])
    vals = np.array(vals)
    d = vals - col(lv, ref, "mean_tardiness")
    lo, hi = bootstrap_ci(d)
    print(f"  test, {lv}: ATCS mean tardiness {vals.mean():.1f} against {ref} {col(lv, ref, 'mean_tardiness').mean():.1f}"
          f" and CP-SAT {col(lv, 'CP-SAT', 'mean_tardiness').mean():.1f}; paired ATCS - {ref} {d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}],"
          f" better on {(d < 0).sum()} of 20 days; checker errors {errors}")
