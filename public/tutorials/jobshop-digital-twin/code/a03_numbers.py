"""Reproduce every number quoted on chapter A3 of "Which Job Should the Machine Do Next?",
"The flexible job shop" (a03-the-flexible-job-shop.html).

Usage:  python a03_numbers.py
Needs numpy and the repository's `jobshop` package (next to `code/` in the course zip, or set
JOBSHOP_REPO to the repository root). Runs in a few seconds: everything is enumeration.
"""
from __future__ import annotations

import itertools
import math
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(os.environ.get("JOBSHOP_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

import inspect  # noqa: E402

from jobshop.cpsat import CPSATController, solve_snapshot, transport_lag  # noqa: E402

# defaults of the repository's planner: the allowance delta (Part 'extras') and the job cap (Part 'workshop' fold)
PLANNER_ARGS = inspect.signature(CPSATController.__init__).parameters
ALLOWANCE = PLANNER_ARGS["allowance"].default
MAX_JOBS = PLANNER_ARGS["max_jobs"].default
assert ALLOWANCE == inspect.signature(solve_snapshot).parameters["allowance"].default
from jobshop.scenario import LOAD_STATION, make_scenario, make_workshop  # noqa: E402


def section(title):
    print(f"\n=== {title}")


# ==============================================================================================================
# The small instances of the chapter (the hero and the flexible widget compute the same in the browser).
# Operation (j, k) = (machine, minutes). Three jobs, three machines, each job visits every machine once.
# ==============================================================================================================
TOY = [
    [(0, 2), (1, 7), (2, 7)],   # job 0: M0 -> M1 -> M2
    [(1, 7), (2, 4), (0, 5)],   # job 1: M1 -> M2 -> M0
    [(2, 2), (0, 6), (1, 2)],   # job 2: M2 -> M0 -> M1
]
N_MACHINES = 3


def pairs_of(jobs):
    """The disjunctive pairs: every two operations of different jobs on the same machine, (u, v) with u < v."""
    ops = [(j, k) for j, r in enumerate(jobs) for k in range(len(r))]
    out = []
    for m in range(N_MACHINES):
        on = [o for o in ops if jobs[o[0]][o[1]][0] == m]
        out += [(u, v) for u, v in itertools.combinations(on, 2) if u[0] != v[0]]
    return out


def longest_paths(jobs, arcs_dis, tau=0, s=0):
    """Earliest start of every operation for a (partial) selection, or None if the graph has a cycle.

    arcs_dis: list of (u, v) meaning "u before v on their machine". Arc weights, as on the page
    (the source node is written 0 on the page, the sink *):
      source 0 -> (j, 0)     tau + chg_v               (the cart from the load station)
      (j, k) -> (j, k + 1)   p_jk + tau + chg_v        (transport, then the changeover once the job is there)
      u -> v (disjunctive)   p_u + s                   (the machine is free, then the changeover)
      (j, last) -> sink *    p_j,last
    chg_v (the dict `sig`) = s if some disjunctive arc enters v (v is not first on its machine), else 0.
    """
    ops = [(j, k) for j, r in enumerate(jobs) for k in range(len(r))]
    has_pred = {o: False for o in ops}
    for _, v in arcs_dis:
        has_pred[v] = True
    sig = {o: (s if has_pred[o] else 0) for o in ops}
    p = {o: jobs[o[0]][o[1]][1] for o in ops}
    succ = {o: [] for o in ops}
    for j, r in enumerate(jobs):
        for k in range(len(r) - 1):
            succ[(j, k)].append(((j, k + 1), p[(j, k)] + tau + sig[(j, k + 1)]))
    for u, v in arcs_dis:
        succ[u].append((v, p[u] + s))
    indeg = Counter(v for u in ops for v, _ in succ[u])
    start = {o: (tau + sig[o] if o[1] == 0 else 0) for o in ops}
    order = [o for o in ops if indeg[o] == 0]
    i = 0
    while i < len(order):
        u = order[i]
        i += 1
        for v, w in succ[u]:
            start[v] = max(start[v], start[u] + w)
            indeg[v] -= 1
            if indeg[v] == 0:
                order.append(v)
    if len(order) < len(ops):
        return None
    return start


def makespan(jobs, start):
    return max(start[(j, len(r) - 1)] + r[-1][1] for j, r in enumerate(jobs))


def selection_arcs(pairs, bits):
    return [(u, v) if b else (v, u) for (u, v), b in zip(pairs, bits)]


def transitive_on_each_machine(jobs, arcs):
    """True if no machine has a 3-cycle among its own operations (the selection is an order on each machine)."""
    before = set(arcs)
    for m in range(N_MACHINES):
        on = [(j, k) for j, r in enumerate(jobs) for k, (mm, _) in enumerate(r) if mm == m]
        for a, b, c in itertools.permutations(on, 3):
            if (a, b) in before and (b, c) in before and (c, a) in before:
                return False
    return True


def enumerate_selections(jobs, tau=0, s=0):
    pairs = pairs_of(jobs)
    n_trans = n_acyclic = 0
    values = []
    for bits in itertools.product([1, 0], repeat=len(pairs)):
        arcs = selection_arcs(pairs, bits)
        n_trans += transitive_on_each_machine(jobs, arcs)
        st = longest_paths(jobs, arcs, tau, s)
        if st is not None:
            n_acyclic += 1
            values.append((makespan(jobs, st), bits))
    return pairs, n_trans, n_acyclic, values


def name(o):
    return f"J{o[0]}.{o[1]}"


# --------------------------------------------------------------------------------------------------------------
section("Hero: the classical 3 x 3 instance, its disjunctive pairs and every selection")
pairs, n_trans, n_acyclic, values = enumerate_selections(TOY)
print("operations: 9, conjunctive arcs (inside jobs): 6, disjunctive pairs:", len(pairs))
print("pairs:", ", ".join(f"{name(u)}-{name(v)} on M{TOY[u[0]][u[1]][0]}" for u, v in pairs))
print(f"selections: 2^{len(pairs)} = {2 ** len(pairs)}; an order on every machine: {n_trans} = 3!^3 = {math.factorial(3) ** 3}; "
      f"acyclic: {n_acyclic}")
loads = [sum(p for r in TOY for m, p in r if m == mm) for mm in range(N_MACHINES)]
lengths = [sum(p for _, p in r) for r in TOY]
print("machine loads M0, M1, M2:", loads, " job lengths J0, J1, J2:", lengths, " lower bound max =", max(loads + lengths))
best = min(v for v, _ in values)
worst = max(v for v, _ in values)
optimal = [b for v, b in values if v == best]
print(f"optimum makespan {best} (reached by {len(optimal)} selection(s)); worst acyclic selection {worst}; "
      f"optimum minus the lower bound: {best - max(loads + lengths)}")
dist = Counter(v for v, _ in values)
print("makespans of the 64 acyclic selections (value: count):", dict(sorted(dist.items())))
opt_arcs = selection_arcs(pairs, optimal[0])
st = longest_paths(TOY, opt_arcs)
print("optimal selection:", ", ".join(f"{name(u)} before {name(v)}" for u, v in opt_arcs))
for m in range(N_MACHINES):
    seq = sorted(((st[o], o) for o in st if TOY[o[0]][o[1]][0] == m))
    print(f"  M{m}: " + "  ".join(f"{name(o)} [{t}, {t + TOY[o[0]][o[1]][1]})" for t, o in seq))


def critical_path(jobs, arcs, tau=0, s=0):
    """One longest path from the source to the sink, as a list of operations."""
    st = longest_paths(jobs, arcs, tau, s)
    cmax = makespan(jobs, st)
    p = {o: jobs[o[0]][o[1]][1] for o in st}
    has_pred = {o: any(v == o for _, v in arcs) for o in st}
    sig = {o: (s if has_pred[o] else 0) for o in st}
    preds = {o: [] for o in st}
    for j, r in enumerate(jobs):
        for k in range(len(r) - 1):
            preds[(j, k + 1)].append(((j, k), p[(j, k)] + tau + sig[(j, k + 1)]))
    for u, v in arcs:
        preds[v].append((u, p[u] + s))
    end = next((j, len(r) - 1) for j, r in enumerate(jobs) if st[(j, len(r) - 1)] + r[-1][1] == cmax)
    path = [end]
    while True:
        v = path[-1]
        tight = [u for u, w in preds[v] if st[u] + w == st[v]]
        if not tight:
            break
        path.append(tight[0])
    return path[::-1], cmax


cp, cmax = critical_path(TOY, opt_arcs)
print("a critical path of the optimum:", " -> ".join(name(o) for o in cp), f"length {cmax} =",
      " + ".join(str(TOY[o[0]][o[1]][1]) for o in cp))

# a selection from one global job priority, J0 before J1 before J2 on every machine, and the reverse
for label, prio in (("J0 > J1 > J2", (0, 1, 2)), ("J2 > J1 > J0", (2, 1, 0))):
    arcs = [(u, v) if prio.index(u[0]) < prio.index(v[0]) else (v, u) for u, v in pairs]
    st_p = longest_paths(TOY, arcs)
    print(f"priority {label} on every machine: acyclic {st_p is not None}, makespan {makespan(TOY, st_p)}")

# the smallest cycles: two oriented pairs (on two machines) that already close a cycle with the job arcs
two_cycles = []
for (a, b) in itertools.combinations(range(len(pairs)), 2):
    for ba, bb in itertools.product([1, 0], repeat=2):
        arcs = selection_arcs([pairs[a], pairs[b]], [ba, bb])
        if longest_paths(TOY, arcs) is None:
            two_cycles.append(arcs)
print("partial selections of two pairs that already close a cycle:", len(two_cycles))
for arcs in two_cycles[:3]:
    print("   ", ", ".join(f"{name(u)} -> {name(v)}" for u, v in arcs))

# --------------------------------------------------------------------------------------------------------------
section("Hero toggles: transport tau = 1 between stations, changeover s = 1 between two different jobs")
for tau, s in ((0, 0), (1, 0), (0, 1), (1, 1)):
    _, _, n_ac, vals = enumerate_selections(TOY, tau, s)
    b = min(v for v, _ in vals)
    nb = sum(1 for v, _ in vals if v == b)
    same = [bb for v, bb in vals if v == b]
    print(f"tau = {tau}, s = {s}: acyclic selections {n_ac}, optimum makespan {b} ({nb} optimal selection(s)); "
          f"the plain optimum's selection gives {makespan(TOY, longest_paths(TOY, opt_arcs, tau, s))}; "
          f"same optimal selection as plain: {optimal[0] in same}")

# --------------------------------------------------------------------------------------------------------------
section("Part 'flexible': the instance table (eligible machines and minutes) and every assignment")
FLEX = [
    [{0: 2}, {1: 7, 2: 6}, {2: 7}],
    [{1: 7, 2: 6}, {2: 4}, {0: 5}],
    [{2: 2}, {0: 6}, {1: 2, 0: 3}],
]
for j in range(3):
    print(f"J{j}: " + " | ".join(
        f"step {k}: " + " or ".join(f"M{m}: {p}" + (" (classical)" if TOY[j][k][0] == m else "")
                                   for m, p in sorted(FLEX[j][k].items(), key=lambda kv: kv[0] != TOY[j][k][0]))
        for k in range(3)))
flex_ops = [(j, k) for j in range(3) for k in range(3) if len(FLEX[j][k]) > 1]
print("flexible operations:", [name(o) for o in flex_ops], " assignments: 2^3 =", 2 ** len(flex_ops))


def best_for_assignment(assign):
    """Best makespan over every order on every machine, for jobs with a fixed assignment."""
    on = {m: [(j, k) for j in range(3) for k in range(3) if assign[j][k][0] == m] for m in range(N_MACHINES)}
    n_orders = n_ac = 0
    best_v, best_orders = None, None
    for perms in itertools.product(*[list(itertools.permutations(on[m])) for m in range(N_MACHINES)]):
        n_orders += 1
        arcs = [(a, b) for seq in perms for a, b in zip(seq, seq[1:])]
        st_a = longest_paths(assign, arcs)
        if st_a is None:
            continue
        n_ac += 1
        v = makespan(assign, st_a)
        if best_v is None or v < best_v:
            best_v, best_orders = v, perms
    return best_v, n_orders, n_ac, best_orders


rows = []
for choice in itertools.product(*[sorted(FLEX[j][k]) for j, k in flex_ops]):
    assign = [[(min(FLEX[j][k]), FLEX[j][k][min(FLEX[j][k])]) for k in range(3)] for j in range(3)]
    for (j, k), m in zip(flex_ops, choice):
        assign[j][k] = (m, FLEX[j][k][m])
    b, n_orders, n_ac, _ = best_for_assignment(assign)
    n_pairs_a = len(pairs_of(assign))
    same_job = sum(1 for m in range(N_MACHINES) for u, v in itertools.combinations(
        [(j, k) for j in range(3) for k in range(3) if assign[j][k][0] == m], 2) if u[0] == v[0])
    load = [sum(p for r in assign for mm, p in r if mm == m) for m in range(N_MACHINES)]
    lb = max(load + [sum(p for _, p in r) for r in assign])
    tag = []
    if choice == (1, 1, 1):
        tag.append("= the classical instance")
    if all(FLEX[j][k][m] == min(FLEX[j][k].values()) for (j, k), m in zip(flex_ops, choice)):
        tag.append("= fastest machine everywhere")
    rows.append((choice, b, n_orders, n_ac, load, lb))
    print(f"J0.1 on M{choice[0]}, J1.0 on M{choice[1]}, J2.2 on M{choice[2]}: loads {load}, lower bound {lb}, "
          f"disjunctive pairs {n_pairs_a} (+{same_job} same-job pairs ordered by the route), "
          f"machine orders {n_orders}, acyclic {n_ac}, best makespan {b} {' '.join(tag)}")
gbest = min(r[1] for r in rows)
worst_flex = max(r[1] for r in rows)
print(f"worst best-makespan over the assignments: {worst_flex}, reached by "
      f"{[('M%d, M%d, M%d' % c) for c, bb, *_ in rows if bb == worst_flex]}")
print(f"best over all assignments: {gbest}, reached by {sum(1 for r in rows if r[1] == gbest)} of {len(rows)} assignments; "
      f"total (assignment, order) pairs examined: {sum(r[2] for r in rows)}")

# ==============================================================================================================
# The course's workshop
# ==============================================================================================================
ws = make_workshop()
section("Part 'workshop': routes, eligible machines and processing times (make_workshop(), seed 0)")
sizes = Counter()
all_p = []
for f in range(ws.n_families):
    cells = []
    for k in range(ws.n_ops(f)):
        alt = ws.alternatives(f, k)
        sizes[len(alt)] += 1
        all_p += list(alt.values())
        cells.append(", ".join(f"m{m}: {p}" for m, p in sorted(alt.items())))
    print(f"family {f}: " + " | ".join(cells) + f" | fastest route {ws.min_remaining_work(f, 0)}"
          f", slowest route {sum(max(a.values()) for a in ws.routings[f])}")
print("machines:", ws.n_machines, " carts (AGVs):", ws.n_agvs, " families:", ws.n_families)
print("operations per family:", [ws.n_ops(f) for f in range(ws.n_families)], " total operation types:", sum(sizes.values()))
print("operations with 1, 2, 3 eligible machines:", sizes[1], sizes[2], sizes[3],
      " mean |M_fk| =", round(sum(k * v for k, v in sizes.items()) / sum(sizes.values()), 3))
print("processing times from", min(all_p), "to", max(all_p), "min")
W = [ws.min_remaining_work(f, 0) for f in range(ws.n_families)]
print("fastest-route work W_f:", W, " mean", sum(W) / len(W))
elig_count = Counter(m for f in range(ws.n_families) for k in range(ws.n_ops(f)) for m in ws.alternatives(f, k))
print("number of operation types each machine can do (m0..m4):", [elig_count[m] for m in range(ws.n_machines)])

section("Part 'workshop': assignment choices, the product of |M_fk| over a job's operations")
per_family = [math.prod(len(ws.alternatives(f, k)) for k in range(ws.n_ops(f))) for f in range(ws.n_families)]
print("assignment choices per family:", per_family, " = ",
      ["x".join(str(len(ws.alternatives(f, k))) for k in range(ws.n_ops(f))) for f in range(ws.n_families)])
print("geometric mean per job (uniform family mix):", round(math.prod(per_family) ** 0.25, 2))

scen = make_scenario(ws, 0)
fams = [j.family for j in scen.jobs]
print("test day seed 0: families of the 60 jobs in arrival order:", fams)
print("jobs per family:", [fams.count(f) for f in range(ws.n_families)], " operations:", sum(ws.n_ops(f) for f in fams))
log_assign = sum(math.log10(per_family[f]) for f in fams)
print(f"assignment choices for the day: product = 10^{log_assign:.1f}, about 10^{log_assign:.0f}")
logs = []
for seed in range(20):
    fs = [j.family for j in make_scenario(ws, seed).jobs]
    logs.append(sum(math.log10(per_family[f]) for f in fs))
print(f"over the 20 test days: 10^{min(logs):.1f} to 10^{max(logs):.1f}")


def fastest_machine(f, k):
    """Fastest eligible machine, ties to the lower index (the assignment used to count orders)."""
    return min(ws.alternatives(f, k).items(), key=lambda kv: (kv[1], kv[0]))[0]


section("Part 'workshop': operations per machine, disjunctive pairs and orders, fastest-machine assignment")
ties = [(f, k) for f in range(ws.n_families) for k in range(ws.n_ops(f))
        if list(ws.alternatives(f, k).values()).count(min(ws.alternatives(f, k).values())) > 1]
print("operation types with a tie for the fastest machine (broken to the lower index):", ties)
print("fastest machine per family and operation:", [[fastest_machine(f, k) for k in range(ws.n_ops(f))] for f in range(ws.n_families)])


def counts_for(families):
    """Operations per machine, disjunctive pairs (different jobs only), log10 of the orientations, of the
    orders of every machine (prod n_m!) and of those orders that respect the routes (prod n_m! / 2^q)."""
    n_m = Counter(fastest_machine(f, k) for f in families for k in range(ws.n_ops(f)))
    nm = [n_m[m] for m in range(ws.n_machines)]
    all_pairs = sum(n * (n - 1) // 2 for n in nm)
    # pairs of operations of the same job on the same machine: already ordered by the route
    same = sum(c * (c - 1) // 2 for f in families
               for c in Counter(fastest_machine(f, k) for k in range(ws.n_ops(f))).values())
    n_pairs = all_pairs - same
    log_orient = n_pairs * math.log10(2)
    log_perm = sum(math.lgamma(n + 1) / math.log(10) for n in nm)
    # the same-job pairs are disjoint, so each halves the permutations independently
    log_orders = log_perm - same * math.log10(2)
    return nm, all_pairs, same, n_pairs, log_orient, log_perm, log_orders


nm, all_pairs, same, n_pairs, log_orient, log_perm, log_orders = counts_for(fams)
print(f"60 jobs: operations per machine {nm} (max/min = {max(nm) / min(nm):.1f}); about 10^{log_perm:.0f} orders of every machine;"
      f" pairs of operations sharing a machine {all_pairs}, of which {same} "
      f"belong to the same job; disjunctive pairs {n_pairs}, orientations 2^{n_pairs} = 10^{log_orient:.0f}; "
      f"orders of every machine prod n_m! = 10^{log_perm:.1f}; respecting the routes prod n_m!/2^{same} = 10^{log_orders:.1f}")
print("Part 'workshop' fold: the planner of C3 plans at most max_jobs =", MAX_JOBS, "jobs (CPSATController default)")
for n in sorted({1, 2, 3, 5, 10, MAX_JOBS}):
    nm_n, all_n, same_n, pairs_n, lo_n, lp_n, lord_n = counts_for(fams[:n])
    la_n = sum(math.log10(per_family[f]) for f in fams[:n])
    print(f"first {n:2d} jobs: assignments 10^{la_n:.1f}, ops per machine {nm_n}, same-job pairs {same_n}, pairs {pairs_n}, "
          f"orientations 10^{lo_n:.1f}, prod n_m! 10^{lp_n:.1f}, machine orders respecting the routes 10^{lord_n:.1f}")

section("Part 'workshop': changeovers s_m(f, g) and travel times tau(a, b); the triangle inequality (A2 #setups, C1)")
offdiag = [ws.setup[m][f][g] for m in range(ws.n_machines) for f in range(ws.n_families) for g in range(ws.n_families) if f != g]
print("changeovers from", min(offdiag), "to", max(offdiag), "min;", len(offdiag), "off-diagonal entries; mean",
      round(sum(offdiag) / len(offdiag), 2))
asym = sum(1 for m in range(ws.n_machines) for f in range(ws.n_families) for g in range(f + 1, ws.n_families)
           if ws.setup[m][f][g] != ws.setup[m][g][f])
print("asymmetric pairs s_m(f,g) != s_m(g,f):", asym, "of", ws.n_machines * 6)
viol = [(m, f, g, h) for m in range(ws.n_machines) for f in range(4) for g in range(4) for h in range(4)
        if len({f, g, h}) == 3 and ws.setup[m][f][h] > ws.setup[m][f][g] + ws.setup[m][g][h]]
print("triangle inequality s_m(f,h) <= s_m(f,g) + s_m(g,h) violated in", len(viol), "of",
      ws.n_machines * 24, "triples (f, g, h distinct)")
print("example violations:", [(m, f, g, h, ws.setup[m][f][h], ws.setup[m][f][g], ws.setup[m][g][h]) for m, f, g, h in viol[:3]])
pmin_m = [min(p for f in range(4) for k in range(ws.n_ops(f)) for mm, p in ws.alternatives(f, k).items() if mm == m)
          for m in range(ws.n_machines)]
print("shortest processing time on each machine:", pmin_m)
weak = [(m, f, g, h) for m, f, g, h in viol if ws.setup[m][f][h] > ws.setup[m][f][g] + pmin_m[m] + ws.setup[m][g][h]]
print("weak triangle s_m(f,h) <= s_m(f,g) + p_min(m) + s_m(g,h) violated in", len(weak), "triples")
min_nonzero = min(offdiag)
print(f"Part 'extras' (One caution): plain triangle broken in {len(viol)} triples, weak form in {len(weak)}; "
      f"why: largest changeover {max(offdiag)} <= smallest {min_nonzero} + shortest operation {min(pmin_m)} + smallest {min_nonzero}"
      f" = {2 * min_nonzero + min(pmin_m)}")
tv = [ws.travel[a][b] for a in range(ws.n_machines + 1) for b in range(ws.n_machines + 1) if a != b]
print("travel times between distinct stations:", sorted(set(tv)), " from the load station:",
      [ws.travel[LOAD_STATION][ws.station(m)] for m in range(ws.n_machines)])
print("Part 'extras': planner allowance delta (CPSATController default) =", ALLOWANCE, "min")
print("planner's transport lag tau + delta between machines 1 and 4:", transport_lag(ws, 1, 4, ALLOWANCE),
      "= travel", ws.travel[ws.station(1)][ws.station(4)], f"+ {ALLOWANCE}; same machine:", transport_lag(ws, 3, 3, ALLOWANCE))

section("Part 'constraints': a worked case of constraints 2 and 5 (release, first ride, changeover)")
r_j, f_j, m_j = 100, 0, 1
p_j = ws.proc(f_j, 0, m_j)
ride = ws.travel[LOAD_STATION][ws.station(m_j)]
A = r_j + ride
s_case = ws.setup_time(m_j, 2, f_j)
for free in (98, 105):
    S = max(free, A) + s_case
    print(f"family-{f_j} job released at r = {r_j}, step 0 on machine {m_j} ({p_j} min): tau(0, st({m_j})) = {ride}, "
          f"alpha = {A}; machine last ran family 2, free from {free}; s_{m_j}(2, {f_j}) = {s_case}; "
          f"S = max({free}, {A}) + {s_case} = {S}, E = {S + p_j}")

section("Part 'constraints': the planner's objective weight, 10 * tardiness + end (jobshop/cpsat.py)")
import inspect  # noqa: E402
import jobshop.cpsat as _cp  # noqa: E402
print("objective terms in cpsat.py:", sorted({ln.strip() for ln in inspect.getsource(_cp).splitlines() if "10 * " in ln}))

section("Part 'code': the literals of make_workshop (jobshop/scenario.py), checked against the drawn workshop")
print("operations per family drawn from integers(3, 5) = 3 or 4; drawn:", [ws.n_ops(f) for f in range(ws.n_families)])
print("eligible machines per operation drawn from [1, 2, 2, 3]: P(1) = 1/4, P(2) = 1/2, P(3) = 1/4; drawn 1, 2, 3:",
      sizes[1], sizes[2], sizes[3], "of 16 (expected 4, 8, 4)")
print("base time integers(8, 25) = 8 to 24 min, scaled per machine by uniform(0.8, 1.3): times must lie in",
      [round(0.8 * 8), round(1.3 * 24)], "; drawn", min(all_p), "to", max(all_p))
print("changeovers integers(3, 11) = 3 to 10; layout: load station at (0, 1), machines on a 4 x 3 grid;"
      " travel = 1 + (d + 1) // 2 for Manhattan distance d")

section("Exercise: two jobs on two machines with opposite routes")
TWO = [[(0, 3), (1, 2)], [(1, 4), (0, 1)]]
pairs2 = []
ops2 = [(j, k) for j in range(2) for k in range(2)]
for m in range(2):
    on = [o for o in ops2 if TWO[o[0]][o[1]][0] == m]
    pairs2 += list(itertools.combinations(on, 2))
for bits in itertools.product([1, 0], repeat=2):
    arcs = selection_arcs(pairs2, bits)
    st2 = longest_paths(TWO, arcs)
    print(", ".join(f"{name(u)} -> {name(v)}" for u, v in arcs), ":",
          "cycle" if st2 is None else f"makespan {makespan(TWO, st2)}; " + ", ".join(
              f"{name(o)} [{st2[o]}, {st2[o] + TWO[o[0]][o[1]][1]})" for o in sorted(st2, key=lambda o: (st2[o], o))))
