"""Reproduce every number quoted on chapter C2 of "Which Job Should the Machine Do Next?",
"How CP-SAT finds and proves a plan" (c02-how-cp-sat-finds-and-proves-a-plan.html).

Usage:  python c02_numbers.py            (about 1 to 2 minutes; the CP-SAT runs use 8 threads)
Needs the repository's environment (numpy, simpy, ortools). Run from the course zip, where
jobshop/ sits next to code/, or set JOBSHOP_REPO to the repository folder.

Parts of the page and where their numbers come from:
  Part 1-2  the paint machine: the search tree and branch and bound        -> section "PAINT"
  Part 3    constraint propagation on a five-operation example              -> section "PROPAGATION"
  Part 4    a learned clause on the paint machine                           -> section "CLAUSE"
  Part 5    cores on the paint machine (and exercise 3)                     -> section "CORES"
  Part 5    large neighbourhood search on the paint machine (exercise 4)    -> section "LNS"
  Part 5-7  CP-SAT on real snapshots of a simulated day                     -> sections "SNAPSHOTS" ...
  Part 7    what deterministic time counts; the GREEDY fallback             -> "DETERMINISTIC TIME", "FALLBACK"
  Part 6    the replan statistics of the benchmark (results/runs.csv)       -> section "RUNS.CSV"
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(os.environ.get("JOBSHOP_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

from ortools.sat.python import cp_model  # noqa: E402

from jobshop.cpsat import CPSATController, solve_snapshot, take_snapshot, transport_lag, warm_start  # noqa: E402
from jobshop.scenario import make_scenario, make_workshop  # noqa: E402
from jobshop.shop import simulate  # noqa: E402

T_START = time.perf_counter()
import ortools  # noqa: E402
print("OR-Tools", ortools.__version__, "(the page's numbers are for 9.15)")


def section(title):
    print(f"\n=== {title}")


# ==============================================================================================================
# PAINT: one machine set up for blue, a colour change takes 10 min (notation.md, the worked example)
# ==============================================================================================================
ORDERS = {"A": ("blue", 20, 30), "B": ("red", 10, 25), "C": ("blue", 15, 70), "D": ("red", 25, 60)}  # colour, p, due
SETUP = 10


def run(seq, t=0, fam="blue"):
    """End time, last colour, total tardiness and start times (after the changeover) of a partial sequence."""
    g, starts = 0, {}
    for n in seq:
        f, p, d = ORDERS[n]
        t += SETUP if f != fam else 0
        starts[n] = t
        t += p
        fam = f
        g += max(0, t - d)
    return t, fam, g, starts


def lower_bound(prefix, kind):
    """kind 'none' and 'sofar': the tardiness of the orders already placed (tardiness is never negative).
    kind 'alone': plus, for each remaining order, its tardiness if it ran alone right now (after a
    changeover if its colour differs from the machine's)."""
    t, fam, g, _ = run(prefix)
    if kind in ("none", "sofar"):
        return g
    rest = [n for n in ORDERS if n not in prefix]
    return g + sum(max(0, t + (SETUP if ORDERS[n][0] != fam else 0) + ORDERS[n][1] - ORDERS[n][2]) for n in rest)


def branch_and_bound(kind="alone", order="bound", start=None):
    """Depth-first branch and bound over the orders, exactly as interactive/c02.js does it.
    A node is a prefix (the orders placed first). Its bound is computed when it is generated; when it
    is taken from the stack it is pruned if bound >= best known value (never with kind 'none'),
    evaluated if complete, expanded otherwise. Returns counts and the trace."""
    ub, inc = (math.inf, None) if start is None else (start[1], start[0])
    stack = [((), lower_bound((), kind))]
    expanded = leaves = pruned = 0
    trace = []
    while stack:
        node, lb = stack.pop()
        if kind != "none" and lb >= ub:
            pruned += 1
            trace.append(f"prune {''.join(node) or 'root'} (bound {lb} >= best {ub})")
            continue
        if len(node) == len(ORDERS):
            leaves += 1
            better = lb < ub
            if better:
                ub, inc = lb, "".join(node)
            trace.append(f"leaf {''.join(node)} = {lb}" + ("  -> new best" if better else ""))
            continue
        expanded += 1
        kids = [(node + (n,), lower_bound(node + (n,), kind)) for n in ORDERS if n not in node]
        if order == "bound":
            kids.sort(key=lambda kv: (kv[1], kv[0]))
        trace.append(f"expand {''.join(node) or 'root'} (bound {lb}): children " + ", ".join(f"{''.join(k)}:{v}" for k, v in kids))
        stack.extend(reversed(kids))
    return dict(explored=expanded + leaves, expanded=expanded, leaves=leaves, pruned=pruned,
                visited=expanded + leaves + pruned, best=ub, plan=inc), trace


section("PAINT. Part 1: every sequence of the four orders and its total tardiness")
values = {"".join(p): run(p)[2] for p in itertools.permutations("ABCD")}
for s, v in sorted(values.items(), key=lambda kv: (kv[1], kv[0])):
    print(f"  {s}: {v}", end="")
print()
print("number of sequences:", len(values), " best:", min(values.values()), "by", [s for s, v in values.items() if v == min(values.values())])
print("FIFO ABCD", values["ABCD"], " SPT BCAD", values["BCAD"], " EDD BADC", values["BADC"], " SST ACBD", values["ACBD"])
t, fam, g, st = run("ABDC")
print("ABDC starts (after changeovers):", st, " end", t)

section("PAINT. Part 1: the size of the search tree")
levels = [math.perm(4, k) for k in range(5)]
print("nodes per level (prefixes of length 0..4):", levels, " total:", sum(levels), " leaves:", levels[-1])
for n in (4, 5, 8, 10, 12, 15, 20, 25):
    print(f"  n = {n}: n! = {math.factorial(n):,} ({math.factorial(n):.1e}); tree nodes sum_k n!/(n-k)! = "
          f"{sum(math.perm(n, k) for k in range(n + 1)):,} = floor(e n!) = {math.floor(math.e * math.factorial(n)) if n < 15 else 'n/a'}")

section("PAINT. Part 2: the lower bounds at the root and its children")
for kind in ("sofar", "alone"):
    print(kind, "root:", lower_bound((), kind), " children:", {n: lower_bound((n,), kind) for n in "ABCD"})
t, fam, g, _ = run("A")
print("after A alone: machine free at", t, "blue; B alone ends", t + SETUP + 10, "(due 25), C ends", t + 15, "(due 70), D ends", t + SETUP + 25, "(due 60)")

section("PAINT. Part 2: branch and bound, all settings (hero demo and table)")
starts = {"none": None, "SST": ("ACBD", 50), "optimum": ("ABDC", 40)}
print(f"{'bound':6} {'order':6} {'start':8} explored expanded leaves pruned visited best plan")
for kind in ("none", "sofar", "alone"):
    for order in ("alpha", "bound"):
        for sname, s in starts.items():
            r, _ = branch_and_bound(kind, order, s)
            print(f"{kind:6} {order:6} {sname:8} {r['explored']:8} {r['expanded']:8} {r['leaves']:6} {r['pruned']:6} {r['visited']:7} {r['best']:4} {r['plan']}")
print("default of the hero: bound 'alone', best bound first, no starting plan. Trace:")
r, trace = branch_and_bound("alone", "bound", None)
for k, line in enumerate(trace, 1):  # one line per Step of the hero; the incumbent appears at step 5
    print(f"  step {k}:", line)
print("  ->", r)

# ==============================================================================================================
# PROPAGATION: five operations, two machines, one precedence (Part 3 and its widget)
# ==============================================================================================================
# P, Q, R on machine 1 and U, V on machine 2 (letters distinct from the paint orders A-D)
TASKS = {"P": dict(m=1, p=4, est=0, lct=20), "Q": dict(m=1, p=3, est=1, lct=8), "R": dict(m=1, p=3, est=2, lct=9),
         "U": dict(m=2, p=3, est=0, lct=20), "V": dict(m=2, p=5, est=3, lct=10)}
PREC = [("P", "U", 2)]  # U starts at least 2 min after P ends (a transport lag)


def propagate_once(dom, edge_finding):
    """The first deduction found, scanning precedences, then timetabling, then edge-finding (as the widget)."""
    for a, b, lag in PREC:
        v = dom[a]["est"] + dom[a]["p"] + lag
        if v > dom[b]["est"]:
            return ("precedence", b, "est", v, f"{a} ends at {dom[a]['est'] + dom[a]['p']} at the earliest, plus the lag {lag}")
        v = dom[b]["lct"] - dom[b]["p"] - lag
        if v < dom[a]["lct"]:
            return ("precedence", a, "lct", v, f"{b} must start by {dom[b]['lct'] - dom[b]['p']}, minus the lag {lag}")
    for m in (1, 2):
        names = [n for n in dom if dom[n]["m"] == m]
        for j in names:
            for i in names:
                if i == j:
                    continue
                lst, ect = dom[i]["lct"] - dom[i]["p"], dom[i]["est"] + dom[i]["p"]
                if lst >= ect:
                    continue
                if dom[j]["est"] < ect and dom[j]["est"] + dom[j]["p"] > lst:
                    return ("timetabling", j, "est", ect, f"{i} occupies [{lst}, {ect}) in every solution")
                if dom[j]["lct"] > lst and dom[j]["lct"] - dom[j]["p"] < ect:
                    return ("timetabling", j, "lct", lst, f"{i} occupies [{lst}, {ect}) in every solution")
    if edge_finding:
        for m in (1, 2):
            names = [n for n in dom if dom[n]["m"] == m]
            for i in names:
                others = [n for n in names if n != i]
                for mask in range(1, 2 ** len(others)):
                    om = [others[k] for k in range(len(others)) if mask >> k & 1]
                    p_om = sum(dom[n]["p"] for n in om)
                    est_all = min(dom[n]["est"] for n in om + [i])
                    lct_om = max(dom[n]["lct"] for n in om)
                    if est_all + p_om + dom[i]["p"] > lct_om:
                        v = max(min(dom[n]["est"] for n in sub) + sum(dom[n]["p"] for n in sub)
                                for r in range(1, len(om) + 1) for sub in itertools.combinations(om, r))
                        if v > dom[i]["est"]:
                            return ("edge-finding", i, "est", v,
                                    f"{i} cannot end before all of {{{', '.join(om)}}}: {est_all} + {p_om} + {dom[i]['p']} > {lct_om}")
                    lct_all = max(dom[n]["lct"] for n in om + [i])
                    est_om = min(dom[n]["est"] for n in om)
                    if lct_all - p_om - dom[i]["p"] < est_om:
                        v = min(max(dom[n]["lct"] for n in sub) - sum(dom[n]["p"] for n in sub)
                                for r in range(1, len(om) + 1) for sub in itertools.combinations(om, r))
                        if v < dom[i]["lct"]:
                            return ("edge-finding", i, "lct", v,
                                    f"{i} cannot start after all of {{{', '.join(om)}}}: {lct_all} - {p_om} - {dom[i]['p']} < {est_om}")
    return None


def propagate(edge_finding):
    dom = {n: dict(t) for n, t in TASKS.items()}
    steps = []
    while (d := propagate_once(dom, edge_finding)) is not None:
        rule, task, field, value, why = d
        steps.append((rule, task, field, dom[task][field], value, why))
        dom[task][field] = value
    return steps, dom


section("PROPAGATION. Part 3: domains shrinking to a fixed point")
print("initial windows [est, lct] and start ranges [est, lct - p]:")
for n, t in TASKS.items():
    print(f"  {n}: machine {t['m']}, p = {t['p']}, window [{t['est']}, {t['lct']}], start in [{t['est']}, {t['lct'] - t['p']}] ({t['lct'] - t['p'] - t['est'] + 1} values)")
for ef in (True, False):
    steps, dom = propagate(ef)
    print(f"edge-finding {'on' if ef else 'off'}: {len(steps)} deductions")
    for k, (rule, task, field, old, new, why) in enumerate(steps, 1):
        print(f"  {k}. {rule:12} {field}_{task}: {old} -> {new}   ({why})")
    print("  fixed point:", {n: (t["est"], t["lct"]) for n, t in dom.items()})
    print("  start values left:", {n: t["lct"] - t["p"] - t["est"] + 1 for n, t in dom.items()},
          " product of the domain sizes:", " x ".join(str(t["lct"] - t["p"] - t["est"] + 1) for t in dom.values()), "=",
          math.prod(t["lct"] - t["p"] - t["est"] + 1 for t in dom.values()),
          " initially:", " x ".join(str(t["lct"] - t["p"] - t["est"] + 1) for t in TASKS.values()), "=",
          math.prod(t["lct"] - t["p"] - t["est"] + 1 for t in TASKS.values()))
# the edge-finding deduction checked by enumeration: P last among P, Q, R in every feasible order
feasible = []
for order in itertools.permutations("PQR"):
    for sP in range(0, 16):  # every start of P in its window [0, 15 - 4]; Q, R left-justified around it
        t, ok = 0, True
        for n in order:
            start = sP if n == "P" else max(t, TASKS[n]["est"])
            ok &= start >= t and start >= TASKS[n]["est"]
            t = start + TASKS[n]["p"]
            ok &= t <= (15 if n == "P" else TASKS[n]["lct"])
        if ok:
            feasible.append(("".join(order), sP))
print("orders of P, Q, R on machine 1 that meet the windows (with lct_P = 15), and P's feasible starts:",
      {o: (min(s for oo, s in feasible if oo == o), max(s for oo, s in feasible if oo == o)) for o in sorted({o for o, _ in feasible})})
# exercise 2: Q's window [1, 9] and R's [2, 10]
saved = {n: dict(t) for n, t in TASKS.items()}
TASKS["Q"]["lct"], TASKS["R"]["lct"] = 9, 10
steps, dom = propagate(True)
print("exercise 2, Q in [1, 9] and R in [2, 10]: edge-finding test for P: 0 + 6 + 4 =", 0 + 6 + 4, "> 10 ?", 10 > 10,
      "; deductions:", [(r, t, f, o, n) for r, t, f, o, n, _ in steps], "; P first then Q, R ends at", 4 + 3 + 3)
TASKS.clear(); TASKS.update(saved)

# ==============================================================================================================
# CLAUSE: a learned clause on the paint machine (Part 4)
# ==============================================================================================================
section("CLAUSE. Part 4: after the plan ABDC (40), look for total tardiness <= 39")
t, fam, g, st = run("D")
print("decide D first: D painted", st["D"], "to", t, "; then B starts at", t, "at the earliest (red, no change) and A at", t + SETUP, "(after a change)")
print("  so tardiness of A >=", t + SETUP + 20 - 30, ", of B >=", t + 10 - 25, ", total >=", (t + SETUP + 20 - 30) + (t + 10 - 25), "> 39: conflict")
print("  learned clause: not [S_A >= 45] or not [S_B >= 35], i.e. [S_A <= 44] or [S_B <= 34]")
print("  lifted: S_A >= 35 and S_B >= 35 give T_A + T_B >=", (35 + 20 - 30) + (35 + 10 - 25), "> 39")
ruled = [(s, values[s], run(s)[3]["A"], run(s)[3]["B"]) for s in values if run(s)[3]["A"] >= 35 and run(s)[3]["B"] >= 35]
print(f"sequences in which A and B both start at minute 35 or later: {len(ruled)} of 24:", [(s, v) for s, v, _, _ in sorted(ruled)])
print("  their smallest total tardiness:", min(v for _, v, _, _ in ruled))
ruled2 = [s for s in values if run(s)[3]["A"] >= 45 and run(s)[3]["B"] >= 35]
print(f"with the unlifted clause (S_A >= 45 and S_B >= 35): {len(ruled2)} sequences:", sorted(ruled2))
print("sequences with D first (the decision cut):", sum(1 for s in values if s[0] == "D"))
t, fam, g, st = run("CD")
print("branch C first, D second: D ends at", t, "; B starts >=", t, "and A >=", t + SETUP, ": the clause fires without new reasoning")
print("lazy literals: with chapter C1's domains [0, H], H = 110, the four starts would need 4 x 110 =", 4 * 110,
      "literals [S_j >= v] if created in advance; the conflict above uses 2")

section("CORES. Part 5 and exercise 3: sets of orders that cannot all be on time")
subsets_all = [c for r in range(1, 5) for c in itertools.combinations("ABCD", r)]
least = {c: min(sum(max(0, run(s)[3][n] + ORDERS[n][1] - ORDERS[n][2]) for n in c) for s in values) for c in subsets_all}
cores = [c for c in subsets_all if least[c] > 0]
minimal = [c for c in cores if not any(set(d) < set(c) for d in cores)]
print("smallest total tardiness of each set of orders over the 24 sequences:", {"".join(c): v for c, v in least.items()})
print("minimal cores (every proper subset can be on time):", [("".join(c), least[c]) for c in minimal])
print("core {A, B}: A first -> A 0-20, change, B 30-40, T_B =", 40 - 25, "; B first -> change, B 10-20, change, A 30-50, T_A =", 50 - 30)
print("pairs of disjoint minimal cores:", [("".join(a), "".join(b)) for a, b in itertools.combinations(minimal, 2) if not set(a) & set(b)])

section("LNS. Part 5 and exercise 4: large neighbourhood search on the paint machine")


def lns_pairs(s):
    """Free two positions (the two other orders stay in place) and re-solve exactly: both orderings of the pair."""
    for i, j in itertools.combinations(range(4), 2):
        l = list(s)
        l[i], l[j] = l[j], l[i]
        yield f"positions {i + 1},{j + 1}", "".join(l)


def lns_window(s, w=3):
    """Free w consecutive positions (a time window) and re-solve exactly: every ordering of the freed orders."""
    for i in range(4 - w + 1):
        for perm in itertools.permutations(s[i:i + w]):
            yield f"positions {i + 1}-{i + w}", s[:i] + "".join(perm) + s[i + w:]


def lns_descent(s, hood):
    """Move to the best neighbour while it improves; ties broken alphabetically (min over (value, sequence))."""
    path = [(s, values[s])]
    while True:
        v, t = min((values[t], t) for _, t in hood(s))
        if v >= values[s]:
            return path
        s = t
        path.append((s, v))


print("from SST's ACBD (50), free any two positions:", [(n, t, values[t]) for n, t in lns_pairs("ACBD")])
print("  best neighbour no better than 50: ACBD is a local optimum of this neighbourhood:",
      min(values[t] for _, t in lns_pairs("ACBD")) >= values["ACBD"])
print("from ACBD, free three consecutive positions:", [(n, t, values[t]) for n, t in lns_window("ACBD")])
print("  descent:", lns_descent("ACBD", lns_window), "; ACBD is", values["ACBD"] - min(values.values()), "above the optimum")
t, fam, g, st = run("ABDC")
print("  ABDC, order by order (start after any change, end, tardiness):",
      [(n, st[n], st[n] + ORDERS[n][1], max(0, st[n] + ORDERS[n][1] - ORDERS[n][2])) for n in "ABDC"], "total", g)
for name, hood in (("two positions", lns_pairs), ("three consecutive", lns_window)):
    minima = sorted(((s, v) for s, v in values.items() if all(values[t] >= v for _, t in hood(s))), key=lambda e: e[1])
    stuck = sum(1 for s in values if lns_descent(s, hood)[-1][1] > min(values.values()))
    print(f"  neighbourhood '{name}': local optima {minima}; of the 24 starting sequences, {stuck} end above the optimum"
          " (ties broken alphabetically)")

    def ends(s, hood=hood):
        """Every final value a descent from s can reach, over all ways of breaking ties between best neighbours."""
        best = min(values[t] for _, t in hood(s))
        if best >= values[s]:
            return {values[s]}
        return set().union(*(ends(t) for _, t in hood(s) if values[t] == best))

    opt = min(values.values())
    tied = sorted((s, sorted(ends(s))) for s in values if len(ends(s)) > 1)
    always = sum(1 for s in values if min(ends(s)) > opt)
    sometimes = sum(1 for s in values if max(ends(s)) > opt)
    print(f"    starts whose end depends on tie-breaking: {tied}")
    print(f"    over all tie choices, between {always} and {sometimes} of the 24 starts end above the optimum")

section("LP. Part 5 recalls chapter C1's result: the big-M relaxation bounds the paint machine's tardiness by 0")
M = 120  # at least twice the longest changeover + painting, 2 x 35
pts = {n: 0.0 for n in ORDERS}  # every start at 0, every ordering variable y = 1/2
ok = all(pts[b] >= pts[a] + SETUP + ORDERS[a][1] - M * 0.5 and pts[a] >= pts[b] + SETUP + ORDERS[b][1] - M * 0.5
         for a, b in itertools.combinations(ORDERS, 2))
print("all starts 0 with y = 1/2 satisfies every relaxed disjunction (M = 120):", ok,
      "; LP bound on total tardiness:", sum(max(0, ORDERS[n][1] - ORDERS[n][2]) for n in ORDERS), "; optimum:", min(values.values()))
print("the argument needs M / 2 >= s + max p =", SETUP + max(o[1] for o in ORDERS.values()), "i.e. M >=", 2 * (SETUP + max(o[1] for o in ORDERS.values())),
      "; with M = 120 a pair reads 0 >= 10 + p -", M // 2)
print("edge-finding deduction for P (Part 3): est_P = max(", 1 + 3, ",", 2 + 3, ",", 1 + 6, ") = 7, over {Q}, {R}, {Q, R}")

# ==============================================================================================================
# SNAPSHOTS: CP-SAT on real snapshots of a simulated day (Parts 5-7)
# ==============================================================================================================
WS = make_workshop()
SNAPS = []


class Recorder(CPSATController):
    """Quick re-planning (the greedy ablation, no solver), recording the snapshot of every replan."""

    def on_event(self, shop, kind, **info):
        jobs = sorted(shop.active_jobs(), key=lambda j: (j.due, j.idx))[: self.max_jobs]
        SNAPS.append((kind, take_snapshot(shop, jobs, self.allowance), dict(self.plan)))
        super().on_event(shop, kind, **info)


LOG: list[str] = []
LAST = {}


class LoggingSolver(cp_model.CpSolver):
    """CpSolver that keeps its search log and itself, so the numbers below can be read after solve_snapshot."""

    def solve(self, model, callback=None):
        self.parameters.log_search_progress = True
        self.parameters.log_to_stdout = False
        self.log_callback = lambda text: LOG.extend(text.splitlines())
        LAST["solver"] = self
        status = super().solve(model, callback)
        LAST["status"] = self.status_name(status)
        return status


cp_model.CpSolver = LoggingSolver  # solve_snapshot creates its solver through cp_model.CpSolver()


def solve(snap, prev, **kw):
    LOG.clear()
    plan = solve_snapshot(WS, snap, previous_plan=prev, **kw)
    s = LAST["solver"]
    return plan, dict(status=plan.status, obj=plan.objective, bound=int(round(s.best_objective_bound)),
                      solver_status=LAST["status"], det=s.deterministic_time, wall=plan.wall_s,
                      branches=s.num_branches, conflicts=s.num_conflicts, log=list(LOG))


EVENT = re.compile(r"^#(\d+|Bound)\s+([\d.]+)s\s+best:(\S+)\s+next:\[(\S*?),(\S*?)\]\s+(\S+)")


def events(log):
    out = []
    for line in log:
        m = EVENT.match(line)
        if m:
            out.append((m.group(1), float(m.group(2)), m.group(3), m.group(4), m.group(5), m.group(6)))
    return out


section("SNAPSHOTS: one simulated day, seed 0, frequent breakdowns, 60 jobs, run by quick re-planning")
scenario = make_scenario(WS, 0, n_jobs=60, mtbf=120.0)
rec = Recorder(use_solver=False)
simulate(WS, scenario, rec)
sizes = [len(s.jobs) for _, s, _ in SNAPS]
print("replans (snapshots):", len(SNAPS), " non-empty:", sum(1 for n in sizes if n), " empty (status EMPTY, no solver call):",
      sum(1 for n in sizes if not n), " largest:", max(sizes), "jobs")
print("snapshot sizes:", Counter(sizes))

section("SNAPSHOTS. Part 6: every third non-empty snapshot, solved as in the benchmark (budget 0.2, hint, 8 workers)")
sample = [i for i, n in enumerate(sizes) if n][::3]
stats = []
for i in sample:
    kind, snap, prev = SNAPS[i]
    plan, r = solve(snap, prev)
    ws_obj = warm_start(WS, snap, 2, prev).objective
    stats.append((len(snap.jobs), r["status"], r["obj"], ws_obj, r["bound"], r["wall"]))
print("solved:", len(stats), " proven optimal:", sum(s[1] == "OPTIMAL" for s in stats),
      f"({100 * sum(s[1] == 'OPTIMAL' for s in stats) / len(stats):.0f} %)",
      " improved on the warm start:", sum(s[2] < s[3] for s in stats), " ended on the warm start (status GREEDY):", sum(s[1] == "GREEDY" for s in stats))
for lo, hi in ((1, 4), (5, 6), (7, 8), (9, 12)):
    grp = [s for s in stats if lo <= s[0] <= hi]
    if grp:
        gaps = [s[2] - s[4] for s in grp if s[4] is not None]
        print(f"  {lo}-{hi} jobs: {len(grp)} snapshots, {sum(s[1] == 'OPTIMAL' for s in grp)} optimal, "
              f"largest gap {max(gaps)}, mean wall {sum(s[5] for s in grp) / len(grp):.2f} s")

# --------------------------------------------------------------------------------------------------------------
I_MID = 101  # an 8-job snapshot: found at once with the hint, proven optimal only with a larger budget
kind, snap, prev = SNAPS[I_MID]
assert len(snap.jobs) == 8, "the simulated day changed: pick another snapshot"
n_ops = sum(WS.n_ops(j.family) - j.first_op for j in snap.jobs)
section(f"SNAPSHOTS. Part 6: snapshot {I_MID} ({kind} at minute {snap.now}): {len(snap.jobs)} jobs, {n_ops} operations to place")
ws_plan = warm_start(WS, snap, 2, prev)
print("warm start (best greedy plan or previous plan re-timed):", ws_plan.objective)
budgets = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
sweep = {"budgets": budgets, "warm": ws_plan.objective}
for use in (True, False):
    rows = []
    for b in budgets:
        plan, r = solve(snap, prev, det_time=b, use_warm_start=use)
        rows.append([r["solver_status"], r["obj"], r["bound"]])
        gap = None if r["obj"] is None else r["obj"] - r["bound"]
        rel = "-" if gap is None else f"{100 * gap / r['obj']:.1f} %"
        print(f"  hint {'on ' if use else 'off'} budget {b:5}: {r['solver_status']:8} objective {r['obj']}  bound {r['bound']}  "
              f"gap {gap} ({rel})  reported det. time {r['det']:.2f}  wall {r['wall']:.2f} s")
        if use and b == 0.8:
            log08 = r["log"]
    sweep["hint" if use else "nohint"] = rows
print("hero-2 data (interactive/c02.js, SWEEP):", json.dumps(sweep))
print(f"  largest budget / smallest: {budgets[-1] / budgets[0]:.0f}-fold")
ev = events(log08)
print("hint on, budget 0.8: solutions found by", [(e[0], e[2], e[5]) for e in ev if e[0] != "Bound"])
print("  bound improvements by worker:", Counter(e[5] for e in ev if e[0] == "Bound"))
print("  hint line of the log:", [l for l in log08 if "hint" in l.lower() and "complete" in l.lower()][:1])
print("  subsolvers:", [l for l in log08 if "subsolvers:" in l])
print("  batch:", [l for l in log08 if "batch" in l.lower()][:2])

# --------------------------------------------------------------------------------------------------------------
I_BIG = max(range(len(SNAPS)), key=lambda i: (len(SNAPS[i][1].jobs), -i))
kind, snap, prev = SNAPS[I_BIG]
n_ops = sum(WS.n_ops(j.family) - j.first_op for j in snap.jobs)
section(f"SNAPSHOTS. Part 6: the largest snapshot, {I_BIG} ({kind} at minute {snap.now}): {len(snap.jobs)} jobs, {n_ops} operations")
print("warm start:", warm_start(WS, snap, 2, prev).objective)
for use in (True, False):
    plan, r = solve(snap, prev, det_time=0.2, use_warm_start=use)
    print(f"  hint {'on ' if use else 'off'} budget 0.2: {r['solver_status']} objective {r['obj']} bound {r['bound']} gap {r['obj'] - r['bound']}"
          f" = {(r['obj'] - r['bound']) / 10:.1f} min of total tardiness if total completion stays the same"
          f" ({100 * (r['obj'] - r['bound']) / r['obj']:.1f} % of the objective)")
    if use:
        big_log = r["log"]
        with_hint = r["obj"]
print(f"  without / with the hint: {r['obj'] / with_hint:.2f} times; completion times alone contribute at least "
      f"{len(snap.jobs)} x {snap.now} = {len(snap.jobs) * snap.now:,} to any plan's objective")
ev = events(big_log)
print("  first bound (initial domains):", [e[3] for e in ev if e[0] == "Bound"][:2], " solutions:", [(e[2], e[5]) for e in ev if e[0] != "Bound"])
print("  model as given to the solver:", [l for l in big_log if l.startswith("#Variables")][:1], [l.strip() for l in big_log if " in [" in l][:3])
# the domains of the start/end/arrival variables: [now, horizon] (lines 158-165 of jobshop/cpsat.py)
horizon = max(snap.machine_free + [snap.now])
max_setup = max(max(max(row) for row in mat) for mat in WS.setup)
max_lag = 2 + max(max(row) for row in WS.travel)
for job in snap.jobs:
    horizon = max(horizon, max(job.ready.values()))
for job in snap.jobs:
    for op in range(job.first_op, WS.n_ops(job.family)):
        horizon += max(WS.alternatives(job.family, op).values()) + max_setup + max_lag
print(f"  time variables in [{snap.now}, {horizon}]: {horizon - snap.now + 1} values each; 3 per operation -> {3 * n_ops} variables;"
      f" an eager order encoding would need {3 * n_ops * (horizon - snap.now):,} literals [x >= v]")


def each_job_alone(snap):
    """A valid lower bound on the model's objective: every job alone, machines free at machine_free, no changeover."""
    total = 0
    for j in snap.jobs:
        E = {}
        for op in range(j.first_op, WS.n_ops(j.family)):
            new = {}
            for k, p in WS.alternatives(j.family, op).items():
                if op == j.first_op:
                    if k not in j.ready:
                        continue
                    arrive = j.ready[k]
                else:
                    arrive = min(E[k2] + transport_lag(WS, k2, k, 2) for k2 in E)
                new[k] = max(arrive, snap.machine_free[k]) + p
            E = new
        end = min(E.values())
        total += 10 * max(0, end - j.due) + end
    return total


print("  exercise 6: each-job-alone bound of this snapshot:", each_job_alone(snap), "(CP-SAT's bound at budget 0.2 above);"
      " of snapshot", I_MID, ":", each_job_alone(SNAPS[I_MID][1]), "(CP-SAT: 7540 at budgets up to 0.1, 7740 at 0.2)")

# --------------------------------------------------------------------------------------------------------------
section(f"SNAPSHOTS. Part 7: reproducibility, snapshot {I_MID}, budget 0.2, hint on")
kind, snap, prev = SNAPS[I_MID]
runs = []
for seed in (0, 0, 1):
    plan, r = solve(snap, prev, seed=seed)
    runs.append((seed, plan, r))
    print(f"  random_seed {seed}: {r['solver_status']} objective {r['obj']} bound {r['bound']} branches {r['branches']} conflicts {r['conflicts']} wall {r['wall']:.3f} s")
print("  the two seed-0 runs give the same plan:", runs[0][1].ops == runs[1][1].ops, "; same objective and bound:",
      (runs[0][2]["obj"], runs[0][2]["bound"]) == (runs[1][2]["obj"], runs[1][2]["bound"]),
      "; same counters:", (runs[0][2]["branches"], runs[0][2]["conflicts"]) == (runs[1][2]["branches"], runs[1][2]["conflicts"]))
print("  seed 1 against seed 0: same plan:", runs[0][1].ops == runs[2][1].ops)

# --------------------------------------------------------------------------------------------------------------
# DETERMINISTIC TIME. Part 7: what the response's deterministic_time counts, and what interleave_search buys.
# The solver is the same as solve_snapshot's; OVERRIDE changes num_workers / interleave_search after the
# repository has set them (solve_snapshot itself is not modified).
# --------------------------------------------------------------------------------------------------------------
OVERRIDE: dict = {}
_base_solve = LoggingSolver.solve


def _solve_with_override(self, model, callback=None):
    for k, v in OVERRIDE.items():
        setattr(self.parameters, k, v)
    return _base_solve(self, model, callback)


LoggingSolver.solve = _solve_with_override
UNIT = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def task_dtime_sum(log):
    """Sum of the last column ('dtime', total per subsolver) of the log's 'Task timing' table."""
    total, inside = 0.0, False
    for line in log:
        if line.startswith("Task timing"):
            inside = True
            continue
        if inside:
            if not line.strip().startswith("'"):
                break
            m = re.search(r"([\d.]+)(ns|us|ms|s)\s*$", line)
            total += float(m.group(1)) * UNIT[m.group(2)]
    return total


section(f"DETERMINISTIC TIME. Part 7: snapshot {I_MID}, reported deterministic_time against the limit")
kind, snap, prev = SNAPS[I_MID]
for b in (0.2, 0.8):
    for w in (1, 8):
        for il in (True, False):
            OVERRIDE.clear()
            OVERRIDE.update(num_workers=w, interleave_search=il)
            plan, r = solve(snap, prev, det_time=b)
            print(f"  limit {b}  num_workers {w}  interleave {str(il):5}: {r['solver_status']:8} reported deterministic_time {r['det']:.3f}"
                  f" = {r['det'] / b:.1f} x the limit; 'Task timing' dtime column summed over subsolvers {task_dtime_sum(r['log']):.3f}"
                  f" (the rest is presolve)")
OVERRIDE.clear()
print("  -> with num_workers 1 the reported figure equals the limit; with 8 it is the work of all subsolvers added up."
      " Values with interleave False change from one execution to the next (thread timing).")

section(f"DETERMINISTIC TIME. Part 7: three solves of snapshot {I_MID} at 0.2 without interleave_search (8 workers)")
OVERRIDE.update(interleave_search=False)
plans_off = []
for k in range(3):
    plan, r = solve(snap, prev)
    plans_off.append((plan.ops, r["bound"]))
    print(f"  run {k + 1}: {r['solver_status']} objective {r['obj']} bound {r['bound']} (changes from one execution to the next)")
OVERRIDE.clear()
print("  same plan in the three runs:", all(p == plans_off[0][0] for p, _ in plans_off),
      "; distinct bounds:", sorted({bd for _, bd in plans_off}), "; with interleave_search the bound was", runs[0][2]["bound"], "in both runs")

section("FALLBACK. Part 7: a budget that runs out in presolve, before the search has loaded the hint")
for i, b in ((I_MID, 0.01), (I_MID, 0.025), (I_BIG, 0.025), (I_BIG, 0.05)):
    kind, snap, prev = SNAPS[i]
    plan, r = solve(snap, prev, det_time=b)
    hint_line = [l for l in r["log"] if "hint" in l.lower() and "complete" in l.lower()][:1]
    started = any(l.startswith("Starting deterministic search") for l in r["log"])
    print(f"  snapshot {i} ({len(snap.jobs)} jobs), budget {b}: solver {r['solver_status']}, search started: {started},"
          f" returned status {plan.status} objective {plan.objective}; {hint_line}")

# ==============================================================================================================
# RUNS.CSV: the replan statistics of the benchmark (20 test days per level)
# ==============================================================================================================
section("RUNS.CSV. Part 6: CP-SAT replans in results/runs.csv")
rows = [r for r in csv.DictReader(open(REPO / "results" / "runs.csv", encoding="utf-8")) if r["method"] == "CP-SAT"]
tot = Counter()
for level in ("no breakdowns", "frequent breakdowns"):
    c = Counter()
    for r in rows:
        if r["level"] == level:
            for k in ("replans", "replans_optimal", "replans_not_improved", "replans_capped"):
                c[k] += int(r[k])
            c["runs"] += 1
            c["decision"] += float(r["decision_time_s"])
    tot += c
    print(f"  {level}: {c['runs']} runs, {c['replans']} replans, {c['replans_optimal']} proven optimal "
          f"({100 * c['replans_optimal'] / c['replans']:.0f} % of all replanning events, empty ones included), "
          f"{c['replans_not_improved']} fell back to the warm start (status GREEDY), "
          f"{c['replans_capped']} capped, {c['decision'] / c['replans']:.2f} s per replan")
print(f"  both levels: {tot['replans']:,} replans, {tot['replans_optimal']:,} proven optimal ({100 * tot['replans_optimal'] / tot['replans']:.0f} %), "
      f"{tot['replans_not_improved']} fell back to the warm start; runs.csv does not record how many replans were EMPTY")

print(f"\n(total running time {time.perf_counter() - T_START:.0f} s)")
