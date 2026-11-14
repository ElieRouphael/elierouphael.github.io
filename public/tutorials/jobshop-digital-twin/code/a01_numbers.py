"""Reproduce every number quoted on chapter A1 of "Which Job Should the Machine Do Next?",
"What a schedule is" (a01-what-a-schedule-is.html).

Usage:  python a01_numbers.py
Run from the course folder (jobshop/ next to code/) or with JOBSHOP_REPO pointing at the repository.
Needs numpy and simpy (for the one short simulation of Part 8). Finishes in a few seconds.
"""
from __future__ import annotations

import csv
import os
import sys
from itertools import permutations
from pathlib import Path

REPO = Path(os.environ.get("JOBSHOP_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from jobshop.checker import check_schedule  # noqa: E402
from jobshop.metrics import run_metrics  # noqa: E402
from jobshop.rules import RuleController  # noqa: E402
from jobshop.scenario import LEVELS, make_scenario, make_workshop  # noqa: E402
from jobshop.shop import Shop  # noqa: E402


def section(title):
    print(f"\n=== {title}")


# ---------------------------------------------------------------- the paint machine (notation.md#paint)
SETUP, START = 10, "blue"
ORDERS = {  # name: (colour, painting time p_j, due date d_j)
    "A": ("blue", 20, 30), "B": ("red", 10, 25), "C": ("blue", 15, 70), "D": ("red", 25, 60)}
RELEASE_NONE = {n: 0 for n in ORDERS}                   # all four wait at minute 0 (the default)
RELEASE_STAGGER = {"A": 0, "B": 5, "C": 10, "D": 15}    # the hero's "arrive five minutes apart" variant


def schedule(seq, release=RELEASE_NONE, idle=None):
    """Semi-active timetable of a sequence: the changeover starts as soon as the machine is free and
    the order has arrived (the repository's rule); `idle` = {name: extra minutes held before it}."""
    idle = idle or {}
    t, colour, rows = 0, START, []
    for n in seq:
        c, p, d = ORDERS[n]
        s = 0 if c == colour else SETUP
        ready = max(t, release[n]) + idle.get(n, 0)
        S = ready + s
        C = S + p
        rows.append(dict(name=n, r=release[n], p=p, d=d, setup=s, free=t, ready=ready, S=S, C=C,
                         L=C - d, T=max(0, C - d), E=max(0, d - C), F=C - release[n], U=int(C > d)))
        t, colour = C, c
    return rows


def measures(rows):
    return dict(Cmax=max(r["C"] for r in rows), sumC=sum(r["C"] for r in rows), sumF=sum(r["F"] for r in rows),
                Lmax=max(r["L"] for r in rows), Tmax=max(r["T"] for r in rows), sumT=sum(r["T"] for r in rows),
                meanT=sum(r["T"] for r in rows) / len(rows), sumU=sum(r["U"] for r in rows),
                sumE=sum(r["E"] for r in rows), setups=sum(r["setup"] for r in rows),
                changes=sum(1 for r in rows if r["setup"]))


def show(seq, release=RELEASE_NONE, idle=None):
    rows = schedule(seq, release, idle)
    for r in rows:
        print(f"  {r['name']}: changeover [{r['ready']},{r['S']}) paint [{r['S']},{r['C']})  r={r['r']} d={r['d']}"
              f"  C={r['C']} L={r['L']} T={r['T']} F={r['F']} E={r['E']}")
    m = measures(rows)
    print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()})
    return m


def rule_seq(key):
    waiting, colour, seq = list(ORDERS), START, []
    while waiting:
        n = min(waiting, key=lambda x: key(x, colour))
        seq.append(n); waiting.remove(n); colour = ORDERS[n][0]
    return "".join(seq)


ARR = {n: i for i, n in enumerate(ORDERS)}
RULESEQ = {
    "FIFO": rule_seq(lambda n, c: ARR[n]),
    "SPT": rule_seq(lambda n, c: (ORDERS[n][1], ARR[n])),
    "EDD": rule_seq(lambda n, c: (ORDERS[n][2], ARR[n])),
    "SST": rule_seq(lambda n, c: (0 if ORDERS[n][0] == c else SETUP, ARR[n])),
}

section("Part 1 (plain words) and hero: first come, first served, ABCD")
show("ABCD")

section("Part 1 and hero presets: the four rules (all orders waiting at minute 0)")
for rule, seq in RULESEQ.items():
    m = measures(schedule(seq))
    print(f"  {rule}: {seq}  sum T = {m['sumT']}  sum C = {m['sumC']}  Cmax = {m['Cmax']}  Lmax = {m['Lmax']}  "
          f"tardy = {m['sumU']}  changeovers = {m['changes']}")

section("Parts 1, 3 and 6 (widget 2): all 24 sequences, every measure, the best for each (ties broken alphabetically)")
from math import factorial  # noqa: E402
print(f"  number of sequences: 4! = {factorial(4)}, and for 10 orders 10! = {factorial(10)}")
allm ={"".join(s): measures(schedule(s)) for s in permutations(ORDERS)}
for key in ("sumT", "sumC", "Cmax", "Lmax", "sumU", "sumE"):
    vals = sorted(set(m[key] for m in allm.values()))
    best = min(allm, key=lambda s: (allm[s][key], s))
    winners = sorted(s for s in allm if allm[s][key] == allm[best][key])
    print(f"  {key}: range {min(vals)} to {max(vals)}, {len(vals)} distinct values; best {allm[best][key]} by {winners}")
print("  makespan values and how many sequences reach each:",
      {v: sum(1 for m in allm.values() if m["Cmax"] == v) for v in sorted(set(m["Cmax"] for m in allm.values()))})
print("  makespan = total painting time 70 + 10 x number of colour changes, for every sequence:",
      all(m["Cmax"] == 70 + 10 * m["changes"] for m in allm.values()))
print("  sequences with the minimum makespan 80:", sorted(s for s in allm if allm[s]["Cmax"] == 80),
      [allm[s]["sumT"] for s in sorted(s for s in allm if allm[s]["Cmax"] == 80)])
r_sumT = {s: sorted(allm, key=lambda x: (allm[x]["sumT"], x)).index(s) + 1 for s in allm}
print("  rank of each rule's sequence by sum T (1 = best of 24):", {k: r_sumT[v] for k, v in RULESEQ.items()})
print("  best for sum T (ABDC):"); show("ABDC")
print("  best for sum C:"); show(min(allm, key=lambda s: (allm[s]["sumC"], s)))

print("  ranks of ABCD (1 = best of 24) on Cmax, sum C, Lmax, sum T:",
      [sorted(allm, key=lambda x: (allm[x][k], x)).index("ABCD") + 1 for k in ("Cmax", "sumC", "Lmax", "sumT")])
ms_sorted = sorted(allm[s]["Cmax"] for s in allm)
tied = [i + 1 for i, v in enumerate(ms_sorted) if v == allm["ABCD"]["Cmax"]]
print(f"  Part 6: FIFO's makespan {allm['ABCD']['Cmax']} is shared by {len(tied)} sequences, ranks {tied[0]} to {tied[-1]}")
print("  Part 3 (lateness is not tardiness): sum L = sum C - sum d for ABDC:",
      allm["ABDC"]["sumC"], "-", sum(o[2] for o in ORDERS.values()), "=",
      sum(r["L"] for r in schedule("ABDC")), "; sum T =", allm["ABDC"]["sumT"])

section("Part 3, Part 4 and exercise 2: orders arrive five minutes apart (r = 0, 5, 10, 15)")
for seq in ("ABCD", "DCBA", "ABDC"):
    print(f"  {seq}:"); show(seq, RELEASE_STAGGER)
alls = {"".join(s): measures(schedule(s, RELEASE_STAGGER)) for s in permutations(ORDERS)}
print("  sum F = sum C - sum r for all 24 sequences (sum r = 30):",
      all(m["sumF"] == m["sumC"] - 30 for m in alls.values()))
print("  widget 2 caption with arrivals: makespan values", sorted(set(m["Cmax"] for m in alls.values())),
      "; equal to 70 + 10 x changes for every sequence:", all(m["Cmax"] == 70 + 10 * m["changes"] for m in alls.values()),
      "; DCBA idle minutes:", alls["DCBA"]["Cmax"] - 70 - alls["DCBA"]["setups"])


def dynamic_rule(key, release):
    """A rule applied when the machine frees, among the orders that have arrived (as the hero's presets)."""
    waiting, t, colour, seq = list(ORDERS), 0, START, []
    while waiting:
        avail = [n for n in waiting if release[n] <= t] or [n for n in waiting if release[n] == min(release[w] for w in waiting)]
        n = min(avail, key=lambda x: key(x, colour))
        c, p, _ = ORDERS[n]
        t = max(t, release[n]) + (0 if c == colour else SETUP) + p
        seq.append(n); waiting.remove(n); colour = c
    return "".join(seq)


KEYS = {"FIFO": lambda n, c: ARR[n], "SPT": lambda n, c: (ORDERS[n][1], ARR[n]), "EDD": lambda n, c: (ORDERS[n][2], ARR[n]),
        "SST": lambda n, c: (0 if ORDERS[n][0] == c else SETUP, ARR[n])}
for rule, key in KEYS.items():
    sq = dynamic_rule(key, RELEASE_STAGGER)
    print(f"  {rule} with arrivals: {sq}  sum T = {alls[sq]['sumT']}  sum F = {alls[sq]['sumF']}  Cmax = {alls[sq]['Cmax']}"
          f"  (all waiting: {dynamic_rule(key, RELEASE_NONE)})")
bC = min(alls, key=lambda s: (alls[s]["sumC"], s)); bT = min(alls, key=lambda s: (alls[s]["sumT"], s))
print(f"  best sum T with arrivals: {bT} ({alls[bT]['sumT']})")
bF = min(alls, key=lambda s: (alls[s]["sumF"], s))
print(f"  best sum C: {bC} ({alls[bC]['sumC']}); best sum F: {bF} ({alls[bF]['sumF']})")

section("Part 2 (schedule) and Part 3 (measures): the three-job example with release dates, one machine, no changeovers")
J3 = {"1": (0, 8, 10), "2": (5, 4, 12), "3": (12, 6, 25)}   # name: (r, p, d)


def one_machine(seq, jobs):
    t, out = 0, []
    for n in seq:
        r, p, d = jobs[n]
        S = max(t, r); C = S + p
        out.append((n, r, p, d, S, C, C - d, max(0, C - d), C - r))
        t = C
    return out


for seq in ("123", "213"):
    rows = one_machine(seq, J3)
    print(f"  sequence {seq}:")
    for n, r, p, d, S, C, L, T, F in rows:
        print(f"    job {n}: r={r} p={p} d={d}  S={S} C={C} L={L} T={T} F={F}")
    print(f"    Cmax={max(x[5] for x in rows)} sumC={sum(x[5] for x in rows)} sumF={sum(x[8] for x in rows)} "
          f"Lmax={max(x[6] for x in rows)} sumT={sum(x[7] for x in rows)} meanT={sum(x[7] for x in rows) / 3:.2f} "
          f"idle={max(x[5] for x in rows) - sum(x[2] for x in rows)}")
print("  weighted: with w = (3, 1, 1), sum wC for 123 =", sum(w * x[5] for w, x in zip((3, 1, 1), one_machine("123", J3))),
      " for 213 =", sum(w * x[5] for w, x in zip((1, 3, 1), one_machine("213", J3))))

section("Part 10, exercise 2 (b): C_max and F_max, one machine, (r, p) = (0, 10), (0, 10), (10, 1)")
JF = {"1": (0, 10, 0), "2": (0, 10, 0), "3": (10, 1, 0)}
for seq in ("".join(s) for s in permutations("123")):
    rows = one_machine(seq, JF)
    print(f"  {seq}: C = {[x[5] for x in rows]}  F = {[x[8] for x in rows]}  Cmax = {max(x[5] for x in rows)}"
          f"  Fmax = {max(x[8] for x in rows)}")

section("Part 4 (regular measures) and the timetable widget: delay B, and everything after it, by 5 minutes in FIFO ABCD")
m0 = show("ABCD"); m5 = show("ABCD", idle={"B": 5})
print("  every regular measure is no smaller:",
      all(m5[k] >= m0[k] for k in ("Cmax", "sumC", "sumF", "Lmax", "Tmax", "sumT", "sumU")),
      " total earliness:", m0["sumE"], "->", m5["sumE"])

section("Part 4, the left-shift theorem: (LS) applied to the delayed FIFO timetable gives back the semi-active one")
delayed = schedule("ABCD", idle={"B": 5}); tight = schedule("ABCD")
print("  delayed starts:", {r["name"]: r["S"] for r in delayed}, " (LS) starts:", {r["name"]: r["S"] for r in tight},
      " every end no later:", all(t_["C"] <= d_["C"] for t_, d_ in zip(tight, delayed)))

section("Part 4: a semi-active schedule that is not non-delay can be better (two jobs, one machine)")
J2 = {"1": (0, 10, 100), "2": (1, 1, 2)}   # name: (r, p, d)
for seq in ("12", "21"):
    rows = one_machine(seq, J2)
    print(f"  {seq}: " + ", ".join(f"job {n} [{S},{C}) T={T}" for n, r, p_, d, S, C, L, T, F in rows),
          " sum T =", sum(x[7] for x in rows), " idle before the first start:", rows[0][4])

section("Part 8 (pitfalls): total lateness is total completion time minus a constant")
sum_d = sum(o[2] for o in ORDERS.values())
print(f"  sum d = {sum_d}; FIFO: sum L = {sum(r['L'] for r in schedule('ABCD'))} = {allm['ABCD']['sumC']} - {sum_d}")
bL = min(allm, key=lambda s: (sum(r["L"] for r in schedule(s)), s))
print(f"  the order minimising sum L is {bL}, the same as for sum C; its sum T = {allm[bL]['sumT']}")

section("Part 3 (measures): the CP-SAT objective 10 sum T + sum C for the paint sequences")
for s in ("ABDC", "ABCD", "ACBD"):
    print(f"  {s}: 10*{allm[s]['sumT']} + {allm[s]['sumC']} = {10 * allm[s]['sumT'] + allm[s]['sumC']}")
bestJ = min(allm, key=lambda s: (10 * allm[s]["sumT"] + allm[s]["sumC"], s))
print("  best of 24 for 10 sum T + sum C:", bestJ, 10 * allm[bestJ]["sumT"] + allm[bestJ]["sumC"])

section("Part 10, exercise 1: the EDD sequence BADC by hand")
show("BADC")

section("Part 10, exercise 5: C, D, A, B with arrivals 0, 5, 10, 15 (forced idle time)")
show("CDAB", RELEASE_STAGGER)

section("Part 5 (notation): P2||Cmax on five jobs 3, 3, 2, 2, 2, by brute force and by largest-first")
P = [3, 3, 2, 2, 2]
best = min(max(sum(p for p, a in zip(P, asg) if a == 0), sum(p for p, a in zip(P, asg) if a == 1))
           for asg in np.ndindex(*(2,) * len(P)))
loads = [0, 0]
for p in sorted(P, reverse=True):
    loads[loads.index(min(loads))] += p
print(f"  optimal makespan {best} (3+3 on one machine, 2+2+2 on the other); largest-first gives {max(loads)} {loads}; "
      f"lower bound sum p / 2 = {sum(P) / 2}")

# ---------------------------------------------------------------- the workshop's benchmark (results/runs.csv)
METHODS = ["FIFO", "SPT", "EDD", "SST", "DQN", "Greedy replan", "CP-SAT"]


def js1(x):
    """One decimal, rounded half up on the exact binary value, as JavaScript's toFixed(1) does in the demo."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(x).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


with open(REPO / "results" / "runs.csv", newline="") as fh:
    runs = [r for r in csv.DictReader(fh) if r["method"] in METHODS]
print(f"\n{len(runs)} runs of the seven methods; checker errors in total: {sum(int(r['checker_errors']) for r in runs)}")
with open(REPO / "results" / "runs.csv", newline="") as fh:
    every = list(csv.DictReader(fh))
print(f"all {len(every)} runs in runs.csv (the seven methods and the agent's five training replicas): "
      f"{sum(int(r['checker_errors']) for r in every)} checker errors")
# Part 2 (schedule): 480 = (7 methods + 5 DQN training seeds) x 20 days x 2 levels, of which 280 are the seven methods
names = sorted(set(r["method"] for r in every)); days = sorted(set(int(r["seed"]) for r in every))
levels = sorted(set(r["level"] for r in every))
print(f"  methods in runs.csv: {len(names)} = 7 headline + {len([n for n in names if n not in METHODS])} DQN training seeds "
      f"{[n for n in names if n not in METHODS]}; days {len(days)}; levels {len(levels)}; "
      f"{len(names)} x {len(days)} x {len(levels)} = {len(names) * len(days) * len(levels)}, "
      f"of which 7 x {len(days)} x {len(levels)} = {7 * len(days) * len(levels)} are the seven methods")
COLS = ["makespan", "utilisation", "mean_flow_time", "mean_tardiness", "max_tardiness", "tardy_share"]

section("Part 6 (objective), widget 4, result line, Parts 8, 9 and summary: mean over the 20 test days, per method")
spread = {}
for level in LEVELS:
    print(f"  -- {level}")
    print("  " + "method".ljust(14) + "".join(c.rjust(16) for c in COLS))
    table = {}
    for m in METHODS:
        rows = [r for r in runs if r["level"] == level and r["method"] == m]
        assert len(rows) == 20
        table[m] = {c: float(np.mean([float(r[c]) for r in rows])) for c in COLS}
        print("  " + m.ljust(14) + "".join(f"{table[m][c]:16.3f}" for c in COLS))
    for c in COLS:
        vals = [table[m][c] for m in METHODS]
        lo, hi = min(vals), max(vals)
        spread[(level, c)] = (lo, hi)
        print(f"  {c:15s} best {lo:9.3f}  worst {hi:9.3f}  worst/best {hi / lo:6.2f}  (worst-best)/best {100 * (hi - lo) / lo:7.1f} %")
    # JSON-like line for the demo (interactive/a01.js holds these values)
    print("  rounded as on the page (makespan, utilisation %, mean flow time, mean tardiness):",
          {m: (js1(table[m]['makespan']), js1(100 * table[m]['utilisation']), js1(table[m]['mean_flow_time']),
               js1(table[m]['mean_tardiness'])) for m in METHODS})
    # the values interactive/a01.js holds (RUNS): [makespan, utilisation %, mean tardiness, mean flow time]
    print("  demo data:", {m: [round(table[m]["makespan"], 3), round(100 * table[m]["utilisation"], 3),
                               round(table[m]["mean_tardiness"], 3), round(table[m]["mean_flow_time"], 3)] for m in METHODS})
    lo_t, hi_t = spread[(level, "mean_tardiness")]
    print(f"  mean tardiness, two decimals: highest {hi_t:.2f}, lowest {lo_t:.2f}, ratio {hi_t / lo_t:.2f};"
          f" rounded {hi_t:.1f} / {lo_t:.1f} = {round(hi_t, 1) / round(lo_t, 1):.1f}")
    for c in ("mean_tardiness", "makespan"):
        order = sorted(METHODS, key=lambda m: table[m][c])
        print(f"  ranking by mean {c} (1 = lowest): " + ", ".join(f"{i + 1}. {m}" for i, m in enumerate(order)))

section("Part 6 (objective): per test day, how much the methods' makespans differ, and what sets the makespan")
ws = make_workshop()
for level in LEVELS:
    rng_ms, rng_t, gap = [], [], []
    for seed in range(20):
        rows = [r for r in runs if r["level"] == level and int(r["seed"]) == seed]
        ms = [float(r["makespan"]) for r in rows]; mt = [float(r["mean_tardiness"]) for r in rows]
        rng_ms.append(max(ms) - min(ms)); rng_t.append(max(mt) - min(mt))
        last = make_scenario(ws, seed).jobs[-1].arrival
        gap.append(np.mean(ms) - last)
    print(f"  {level}: makespan range across the 7 methods on a day: mean {np.mean(rng_ms):.1f} min, "
          f"max {max(rng_ms):.0f}; mean tardiness range: mean {np.mean(rng_t):.1f} min")
    print(f"  {level}: makespan minus the last arrival, mean over days {np.mean(gap):.1f} min")
lasts = [make_scenario(ws, s).jobs[-1].arrival for s in range(20)]
print(f"  last arrival over the 20 test days: mean {np.mean(lasts):.1f}, from {min(lasts)} to {max(lasts)}")
for level in LEVELS:
    rel = []
    for seed in range(20):
        ms = [float(r["makespan"]) for r in runs if r["level"] == level and int(r["seed"]) == seed]
        rel.append((max(ms) - min(ms)) / min(ms))
    get = lambda m, c: {int(r["seed"]): float(r[c]) for r in runs if r["level"] == level and r["method"] == m}
    ms_c, ms_f = get("CP-SAT", "makespan"), get("FIFO", "makespan")
    mt_c, mt_f = get("CP-SAT", "mean_tardiness"), get("FIFO", "mean_tardiness")
    print(f"  {level}: per-day makespan range relative to the day's best: mean {100 * np.mean(rel):.1f} %; "
          f"CP-SAT shorter makespan than FIFO on {sum(ms_c[s] < ms_f[s] for s in range(20))} of 20 days, "
          f"lower mean tardiness on {sum(mt_c[s] < mt_f[s] for s in range(20))} of 20")
    best_ms = min(METHODS, key=lambda m: np.mean(list(get(m, "makespan").values())))
    best_mt = min(METHODS, key=lambda m: np.mean(list(get(m, "mean_tardiness").values())))
    print(f"  {level}: best method by mean makespan {best_ms}, by mean tardiness {best_mt}")

# ---------------------------------------------------------------- the code walk: run_metrics on one simulated day
section("Part 7 (code walk): FIFO on test day 0 without breakdowns, metrics recomputed from the log")
scenario = make_scenario(ws, 0, mtbf=LEVELS["no breakdowns"])
log = Shop(ws, scenario, RuleController("FIFO")).run()
errors = check_schedule(ws, scenario, log)
met = run_metrics(ws, scenario, log)
print("  checker errors:", len(errors), " segments:", len(log.segments), " AGV legs:", len(log.legs))
print("  run_metrics:", {k: round(v, 4) for k, v in met.items()})
row = next(r for r in runs if r["level"] == "no breakdowns" and r["seed"] == "0" and r["method"] == "FIFO")
print("  runs.csv row agrees:", all(abs(float(row[k]) - met[k]) < 1e-9 for k in met))
C = {}
for s in log.segments:
    C[s.job] = max(C.get(s.job, 0), s.end)
T = [max(0, C[j.idx] - j.due) for j in scenario.jobs]
print(f"  by hand: total tardiness {sum(T)} over {len(T)} jobs = mean {sum(T) / len(T):.4f}; "
      f"late jobs {sum(t > 0 for t in T)}; largest {max(T)}")
F = [C[j.idx] - j.arrival for j in scenario.jobs]
print(f"  sum C = {sum(C.values())}, sum r = {sum(j.arrival for j in scenario.jobs)}, sum F = {sum(F)}"
      f"  (sum C - sum r = {sum(C.values()) - sum(j.arrival for j in scenario.jobs)})")
proc = sum(s.end - s.start for s in log.segments if s.kind == "proc")
setup = sum(s.end - s.start for s in log.segments if s.kind == "setup")
print(f"  processing minutes {proc}, changeover minutes {setup}, 5 x makespan = {5 * met['makespan']:.0f}, "
      f"utilisation {proc / (5 * met['makespan']):.4f}; with changeovers counted {(proc + setup) / (5 * met['makespan']):.4f}")
j0 = scenario.jobs[0]
print(f"  job 0: family {j0.family}, r = {j0.arrival}, d = {j0.due}, C = {C[0]}, F = {C[0] - j0.arrival}, T = {max(0, C[0] - j0.due)}")
print(f"  first jobs (idx, family, r, d): {[(j.idx, j.family, j.arrival, j.due) for j in scenario.jobs[:3]]}")

section("Part 10, exercise 6 (coding): your own metrics against run_metrics, EDD and SST on test days 0-2, no breakdowns")


def my_metrics(scenario, log):
    C = {}
    for s in log.segments:
        C[s.job] = max(C.get(s.job, 0), s.end)
    jobs = scenario.jobs
    return dict(sumC=sum(C[j.idx] for j in jobs), sumF=sum(C[j.idx] - j.arrival for j in jobs),
                sum_r=sum(j.arrival for j in jobs), Cmax=max(C.values()),
                meanT=sum(max(0, C[j.idx] - j.due) for j in jobs) / len(jobs),
                Lmax=max(C[j.idx] - j.due for j in jobs))


for rule in ("EDD", "SST"):
    for seed in range(3):
        sc = make_scenario(ws, seed, mtbf=LEVELS["no breakdowns"])
        lg = Shop(ws, sc, RuleController(rule)).run()
        mine, theirs = my_metrics(sc, lg), run_metrics(ws, sc, lg)
        # the identity sum F = sum C - sum r, checked against run_metrics' independently computed mean flow time
        ok = abs(mine["meanT"] - theirs["mean_tardiness"]) < 1e-9 and mine["Cmax"] == theirs["makespan"] \
            and abs((mine["sumC"] - mine["sum_r"]) / len(sc.jobs) - theirs["mean_flow_time"]) < 1e-9
        print(f"  {rule} day {seed}: Cmax {mine['Cmax']}  mean T {mine['meanT']:.2f}  Lmax {mine['Lmax']}  "
              f"sum F {mine['sumF']} = sum C {mine['sumC']} - sum r {mine['sum_r']}   agrees: {ok}")
