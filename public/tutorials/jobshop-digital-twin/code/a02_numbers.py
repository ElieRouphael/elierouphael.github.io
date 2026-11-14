"""Reproduce every number quoted on chapter A2 of "Which Job Should the Machine Do Next?",
"One machine: when a rule is optimal, and when it cannot be" (a02-one-machine-when-rules-are-optimal.html).

Usage:  JOBSHOP_REPO=<path to the repository> python a02_numbers.py
Needs only the standard library and the repository's package `jobshop` (for its rules and its workshop).
Runs in well under a minute.
"""
from __future__ import annotations

import math
import os
import sys
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

REPO = Path(os.environ.get("JOBSHOP_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

from jobshop import rules  # noqa: E402
from jobshop.scenario import make_workshop  # noqa: E402


def section(title):
    print(f"\n=== {title}")


# ------------------------------------------------------------------ the paint machine (notation.md#paint)
# name: (colour, painting time, due minute, arrival order). The machine is set up for blue.
PAINT = {"A": ("blue", 20, 30, 1), "B": ("red", 10, 25, 2), "C": ("blue", 15, 70, 3), "D": ("red", 25, 60, 4)}
START = "blue"


def run(seq, setup=10, start=START, orders=PAINT):
    """Same timing rule as the simulator: an order starts after the changeover it needs."""
    t, colour, rows = 0, start, []
    for name in seq:
        c, p, due, _ = orders[name]
        s = 0 if c == colour else setup
        start_t = t + s
        end = start_t + p
        rows.append(dict(name=name, free=t, start=start_t, end=end, lateness=end - due, late=max(0, end - due), setup=s))
        t, colour = end, c
    return rows


def measures(rows):
    return dict(T=sum(r["late"] for r in rows), C=sum(r["end"] for r in rows), Lmax=max(r["lateness"] for r in rows),
                Cmax=rows[-1]["end"], changes=sum(1 for r in rows if r["setup"]), tardy=sum(1 for r in rows if r["late"]))


# The repository's rules, called on a stub of the shop: the paint queue becomes the queue of machine 0.
# `op` is set to the order's position so that the stub's `proc` can return each order's own time.
def repo_rule_sequence(rule, setup=10, orders=PAINT, start=START):
    fam = {"blue": 0, "red": 1}
    names = list(orders)
    proc = {(fam[orders[n][0]], k): orders[n][1] for k, n in enumerate(names)}
    ws = SimpleNamespace(proc=lambda f, op, m: proc[(f, op)],
                         setup_time=lambda m, prev, f: 0 if prev is None or prev == f else setup)
    shop = SimpleNamespace(ws=ws)
    jobs = [SimpleNamespace(idx=k, family=fam[orders[n][0]], op=k, due=orders[n][2], queued_at=orders[n][3], name=n)
            for k, n in enumerate(names)]
    m = SimpleNamespace(idx=0, queue=jobs[:], last_family=fam[start])
    seq = []
    while m.queue:
        j = rules.pick(rule, shop, m)
        seq.append(j.name)
        m.queue.remove(j)
        m.last_family = j.family
    return seq


# the toolkit's rules (interactive/jobshop-lib.js): the same keys, SST tied by arrival instead of due date
def toolkit_rule_sequence(rule, setup=10, orders=PAINT, start=START):
    waiting, colour, seq = list(orders), start, []
    key = {"FIFO": lambda n, c: (orders[n][3],),
           "SPT": lambda n, c: (orders[n][1], orders[n][3]),
           "EDD": lambda n, c: (orders[n][2], orders[n][3]),
           "SST": lambda n, c: (0 if orders[n][0] == c else setup, orders[n][3])}[rule]
    while waiting:
        nxt = min(waiting, key=lambda n: key(n, colour))
        seq.append(nxt); waiting.remove(nxt); colour = orders[nxt][0]
    return seq


RULE_NAMES = ["FIFO", "SPT", "EDD", "SST"]

section("Part 1 (paint) and hero: the four rules and the best of the 24 orders, changeover 10 min")
ALL = ["".join(p) for p in permutations("ABCD")]
values = {s: measures(run(s))["T"] for s in ALL}
best = min(ALL, key=lambda s: (values[s], s))
seqs = {}
for r in RULE_NAMES:
    a, b = "".join(repo_rule_sequence(r)), "".join(toolkit_rule_sequence(r))
    assert a == b, (r, a, b)
    seqs[r] = a
seqs["Best"] = best
for name, s in seqs.items():
    rows = run(s); m = measures(rows)
    better = sum(1 for v in values.values() if v < m["T"])
    print(f"{name:5s} {s}  total tardiness {m['T']:3d}  changeovers {m['changes']}  makespan {m['Cmax']}  "
          f"sum C {m['C']}  Lmax {m['Lmax']}  late orders {m['tardy']}  orders strictly better: {better} of 24  | "
          + ", ".join(f"{r['name']} {r['start']}-{r['end']} late {r['late']}" for r in rows))
print("the repository's rules.pick and the toolkit's ruleSequence give the same four sequences: True")
print("number of orders:", len(ALL), "= 4!")

section("Hero histogram: total tardiness of all 24 orders (setup 10)")
dist = {}
for s in ALL:
    dist.setdefault(values[s], []).append(s)
for v in sorted(dist):
    print(f"  {v:3d} min: {len(dist[v]):2d} orders  {' '.join(dist[v])}")
vals = sorted(values.values())
print("min", vals[0], "max", vals[-1], "median of the 24:", (vals[11] + vals[12]) / 2,
      "mean", round(sum(vals) / 24, 2), "optimal orders:", dist[vals[0]])
print("orders at or below SST's 50:", sum(1 for v in vals if v <= 50), "; strictly below 50:", sum(1 for v in vals if v < 50))

section("Hero slider: the changeover time 0, 5, 10, 15, 20 min (the repository's rules; SST ties broken by due date)")
for setup in (0, 5, 10, 15, 20):
    vs = {s: measures(run(s, setup))["T"] for s in ALL}
    b = min(ALL, key=lambda s: (vs[s], s))
    parts = []
    for r in RULE_NAMES:
        s = "".join(repo_rule_sequence(r, setup))
        parts.append(f"{r} {s} {vs[s]} (beaten by {sum(1 for v in vs.values() if v < vs[s])})")
    print(f"setup {setup:2d}: " + "  ".join(parts) + f"  best {b} {vs[b]}  (optimal orders: {sum(1 for v in vs.values() if v == vs[b])})")
    print("          all 24:", " ".join(f"{s}:{vs[s]}" for s in ALL))
print("toolkit SST (ties by arrival) at setup 0:", "".join(toolkit_rule_sequence("SST", 0)),
      "; repository SST (ties by due date) at setup 0:", "".join(repo_rule_sequence("SST", 0)))

# ------------------------------------------------------------------ Part 3: interchange, SPT and WSPT
section("Part 3 (interchange): sum of completion times on the paint machine without changeovers")
for pair in ("AB", "BA"):
    rows = run(pair, 0)
    print(f"opening: {pair[0]} then {pair[1]} completes at", [r["end"] for r in rows], "sum", measures(rows)["C"])
print("C and D then start at", run("ABCD", 0)[2]["start"], "after AB and at", run("BACD", 0)[2]["start"], "after BA; saving",
      measures(run("ABCD", 0))["C"] - measures(run("BACD", 0))["C"], "= p_A - p_B")
for name in ("ABCD", "BCAD"):
    rows = run(name, 0)
    print(name, "completions", [r["end"] for r in rows], "sum C", measures(rows)["C"])
p_spt = sorted(v[1] for v in PAINT.values())
print("formula sum_q (n-q+1) p_[q] for SPT:", " + ".join(f"{4 - q}*{p}" for q, p in enumerate(p_spt)), "=",
      sum((4 - q) * p for q, p in enumerate(p_spt)))
optC0 = min(measures(run(s, 0))["C"] for s in ALL)
print("best sum C over the 24 orders, no changeover:", optC0, "; orders reaching it:",
      [s for s in ALL if measures(run(s, 0))["C"] == optC0])
print("swap A,B at the front (no changeover): ABCD -> BACD changes sum C by",
      measures(run("BACD", 0))["C"] - measures(run("ABCD", 0))["C"], "= p_B - p_A =", 10 - 20)

section("Part 3: with the 10-min changeover, SPT no longer minimises sum C")
optC = min(ALL, key=lambda s: (measures(run(s))["C"], s))
print("SPT BCAD sum C", measures(run("BCAD"))["C"], "; best", optC, measures(run(optC))["C"],
      "; completions of the best", [r["end"] for r in run(optC)])

section("Exercise 1 (WSPT): weight 2 on order D, weight 1 on the others, no changeover")
print("swap C before D: w_D p_C - w_C p_D = 2*15 - 25 =", 2 * 15 - 25, "> 0, so D goes first")
w = {"A": 1, "B": 1, "C": 1, "D": 2}
ratio = {n: PAINT[n][1] / w[n] for n in PAINT}
wspt = "".join(sorted(PAINT, key=lambda n: ratio[n]))
wC = lambda s: sum(w[r["name"]] * r["end"] for r in run(s, 0))
print("ratios p/w:", ratio, "WSPT order", wspt, "weighted sum", wC(wspt), "; SPT order BCAD weighted sum", wC("BCAD"),
      "; best over 24:", min(wC(s) for s in ALL))

# ------------------------------------------------------------------ Part 4: Jackson's rule
section("Part 4 (jackson): maximum lateness")
for pair in ("AB", "BA"):
    rows = run(pair, 0)
    print(f"two orders, no changeover: {pair[0]} then {pair[1]} latenesses", [r["lateness"] for r in rows],
          "worst", measures(rows)["Lmax"])
for setup in (0, 10):
    e = "".join(toolkit_rule_sequence("EDD", setup))
    rows = run(e, setup)
    best_L = min(measures(run(s, setup))["Lmax"] for s in ALL)
    print(f"setup {setup}: EDD {e} lateness", [r["lateness"] for r in rows], "Lmax", measures(rows)["Lmax"],
          " best Lmax over 24:", best_L, " orders reaching it:", [s for s in ALL if measures(run(s, setup))["Lmax"] == best_L])

# ------------------------------------------------------------------ Part 5: total tardiness
section("Part 5 (tardiness): the three-job counterexample X(10, 30), Y(20, 10), Z(30, 20), no changeovers")
XYZ = {"X": ("-", 10, 30, 1), "Y": ("-", 20, 10, 2), "Z": ("-", 30, 20, 3)}
for s in ("".join(p) for p in permutations("XYZ")):
    rows = run(s, 0, "-", XYZ)
    print(s, "completions", [r["end"] for r in rows], "tardiness", [r["late"] for r in rows], "total", measures(rows)["T"])
print("SPT = XYZ, EDD = YZX, best = YXZ")

section("Part 5: the swap of two adjacent orders C (15, 70) and D (25, 60) as a function of their start t")


def pair_cost(t, first, second):
    (p1, d1), (p2, d2) = first, second
    return max(0, t + p1 - d1) + max(0, t + p1 + p2 - d2)


Cj, Dj = (15, 70), (25, 60)
for t in (0, 20, 30, 40, 45, 50, 60):
    print(f"t = {t:2d}: C first (SPT) {pair_cost(t, Cj, Dj):3d}, D first (EDD) {pair_cost(t, Dj, Cj):3d}")
cross = [t for t in range(0, 100) if pair_cost(t, Cj, Dj) < pair_cost(t, Dj, Cj)]
print("C first is strictly better from t =", cross[0], "on; D first strictly better for t in",
      [t for t in range(100) if pair_cost(t, Dj, Cj) < pair_cost(t, Cj, Dj)][0], "to",
      [t for t in range(100) if pair_cost(t, Dj, Cj) < pair_cost(t, Cj, Dj)][-1])
print("Exercise 3: Phi_CD(t) = (t-55)^+ + (t-20)^+, Phi_DC(t) = (t-35)^+ + (t-30)^+; kinks at", sorted({55, 20, 35, 30}),
      "; on [35, 55]: t-20 against 2t-65, equal at t =", (65 - 20), "; beyond 55: 2t-75 against 2t-65")
print("sum C of the pair: C first minus D first =", (2 * 0 + 2 * 15 + 25) - (2 * 0 + 2 * 25 + 15), "= p_C - p_D at every t")

# ------------------------------------------------------------------ Part 7: changeovers and the travelling salesman
section("Part 7 (setups): the paint machine as a travelling salesman, makespan = 70 + changeovers")
for s in ("ABCD", "ACBD", "BADC"):
    print(s, "changeovers", measures(run(s))["changes"], "makespan", measures(run(s))["Cmax"])
print("smallest makespan over the 24 orders:", min(measures(run(s))["Cmax"] for s in ALL))

ws = make_workshop()
S = [[list(row) for row in ws.setup[m]] for m in range(ws.n_machines)]
section("Part 7: the workshop's changeover matrices s_m(f, g)")
for m in range(ws.n_machines):
    print(f"machine {m}:", S[m])
asym = [(m, f, g, S[m][f][g], S[m][g][f]) for m in range(5) for f in range(4) for g in range(f + 1, 4) if S[m][f][g] != S[m][g][f]]
print("asymmetric pairs:", len(asym), "of", 5 * 6, "; example machine 0: s(0,2) =", S[0][0][2], ", s(2,0) =", S[0][2][0])
viol = [(m, f, h, g, S[m][f][g], S[m][f][h] + S[m][h][g]) for m in range(5) for f in range(4) for g in range(4)
        for h in range(4) if len({f, g, h}) == 3 and S[m][f][g] > S[m][f][h] + S[m][h][g]]
print("triangle-inequality violations s(f,g) > s(f,h) + s(h,g):", len(viol), viol[:6])
print("range of changeovers off the diagonal:", min(S[m][f][g] for m in range(5) for f in range(4) for g in range(4) if f != g),
      "to", max(S[m][f][g] for m in range(5) for f in range(4) for g in range(4) if f != g))


def held_karp(D, start=None):
    """Shortest path visiting every city once, from `start` (None: free start, as a machine that has run nothing)."""
    n = len(D)
    G = {}
    for j in range(n):
        G[(1 << j, j)] = (0 if start is None else D[start][j], [j])
    for size in range(2, n + 1):
        for mask in range(1 << n):
            if bin(mask).count("1") != size:
                continue
            for j in range(n):
                if not mask >> j & 1:
                    continue
                prev = mask ^ (1 << j)
                G[(mask, j)] = min((G[(prev, i)][0] + D[i][j], G[(prev, i)][1] + [j]) for i in range(n) if prev >> i & 1)
    return min(G[((1 << n) - 1, j)] for j in range(n))


def nearest_neighbour(D, first):
    path, left, cost = [first], set(range(len(D))) - {first}, 0
    while left:
        nxt = min(left, key=lambda g: (D[path[-1]][g], g))
        cost += D[path[-1]][nxt]; path.append(nxt); left.remove(nxt)
    return cost, path


section("Part 7: one batch per family; the best family order on each machine (Held-Karp) against SST (nearest neighbour)")
print("family orders per machine: 4! =", math.factorial(4))
for m in range(5):
    opt = held_karp(S[m])
    worst = max((sum(S[m][a][b] for a, b in zip(p, p[1:])), list(p)) for p in permutations(range(4)))
    nn = [nearest_neighbour(S[m], f) for f in range(4)]
    nn_best_first = []
    for f in range(4):
        # best path that starts with family f
        bf = min((sum(S[m][a][b] for a, b in zip(p, p[1:])), list(p)) for p in permutations(range(4)) if p[0] == f)
        nn_best_first.append((f, nn[f], bf))
    print(f"machine {m}: best path {opt[1]} total {opt[0]}, worst {worst[1]} total {worst[0]}; "
          + "; ".join(f"from family {f}: SST {c}{p} vs best {b[0]}{b[1]}" for f, (c, p), b in nn_best_first))

# ------------------------------------------------------------------ Part 8: counting orders
section("Part 8 (counting): n!, Stirling, brute force at 10^9 sequences per second, the dynamic programmes")
for n in (4, 5, 8, 10, 12, 13, 14, 15, 20, 25, 60):
    f = math.factorial(n)
    st = math.sqrt(2 * math.pi * n) * (n / math.e) ** n
    years = f / 1e9 / (3600 * 24 * 365.25)
    print(f"n = {n:2d}: n! = {f:,}  ({f:.3e});  Stirling {st:.4e}, ratio {st / f:.4f};  log10 n! = {math.log10(f):.2f};  "
          f"at 1e9/s: {1e3 * f / 1e9:.3g} ms = {f / 1e9:.3g} s = {f / 1e9 / 60:.3g} min = {years:.3g} years;  n 2^n = {n * 2 ** n:,} ({n * 2 ** n:.2e});  "
          f"n^2 2^n = {n * n * 2 ** n:,} ({n * n * 2 ** n:.2e})")
print("the table of Part 8, rounded as on the page (2 significant digits):")
for n in (4, 10, 15, 20, 25, 60):
    f = math.factorial(n)
    print(f"  n = {n:2d}: n! {f:.1e}; at 1e9/s {f / 1e9:.2e} s = {f / 1e9 / 60:.3g} min = {f / 1e9 / (3600 * 24 * 365.25):.1e} years; "
          f"n 2^n {n * 2 ** n:.1e}; n^2 2^n {n * n * 2 ** n:.1e}")
print("ln n! against n ln n - n + 1 for n = 20:", round(math.lgamma(21), 3), round(20 * math.log(20) - 20 + 1, 3))

# ------------------------------------------------------------------ the random small instances (mulberry32, as the demo)
M32 = 0xFFFFFFFF


def mulberry32(seed):
    """The toolkit's JS.rng, bit for bit: the demo draws the same instances."""
    a = [seed & M32]

    def r():
        a[0] = (a[0] + 0x6D2B79F5) & M32
        x = a[0]
        t = ((x ^ (x >> 15)) * (1 | x)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return r


SETUP0 = S[0]                     # changeovers of the workshop's machine 0, families 0-3
ZERO = [[0] * 4 for _ in range(4)]


def instance(r, n, theta, R=0.6):
    p = [5 + math.floor(r() * 26) for _ in range(n)]          # 5 to 30 min, as the workshop's operations
    fam = [math.floor(r() * 4) for _ in range(n)]
    P = sum(p)
    d = [max(0, math.floor(P * (1 - theta + R * (r() - 0.5)) + 0.5)) for _ in range(n)]
    return p, fam, d


def total_tardiness(p, fam, d, seq, St):
    t, prev, s = 0, None, 0
    for j in seq:
        t += (0 if prev is None else St[prev][fam[j]]) + p[j]
        prev = fam[j]
        s += max(0, t - d[j])
    return s


def brute(p, fam, d, St):
    n, best = len(p), [math.inf]

    def dfs(t, prev, used, acc):          # prefix sharing: each prefix is evaluated once
        if acc >= best[0]:
            return
        if used == (1 << n) - 1:
            best[0] = acc
            return
        for j in range(n):
            if not used >> j & 1:
                tj = t + (0 if prev is None else St[prev][fam[j]]) + p[j]
                dfs(tj, fam[j], used | 1 << j, acc + max(0, tj - d[j]))
    dfs(0, None, 0, 0)
    return best[0]


def subset_dp(p, d):
    """1||sum T_j: F(S) = min over the last job j of S of F(S - j) + max(0, p(S) - d_j)."""
    n = len(p)
    F = [0] + [math.inf] * ((1 << n) - 1)
    P = [0] * (1 << n)
    for S_ in range(1, 1 << n):
        low = (S_ & -S_).bit_length() - 1
        P[S_] = P[S_ ^ (1 << low)] + p[low]
        F[S_] = min(F[S_ ^ (1 << j)] + max(0, P[S_] - d[j]) for j in range(n) if S_ >> j & 1)
    return F[-1]


def naive_state_dp(p, fam, d, St):
    """The wrong dynamic programme for setups: state (set, last job), keeping only the least tardiness."""
    n = len(p)
    G = {(1 << j, j): (max(0, p[j] - d[j]), p[j]) for j in range(n)}
    for mask in sorted(range(1, 1 << n), key=lambda x: bin(x).count("1")):
        for j in range(n):
            if (mask, j) not in G:
                continue
            cost, t = G[(mask, j)]
            for k in range(n):
                if mask >> k & 1:
                    continue
                tk = t + St[fam[j]][fam[k]] + p[k]
                cand = (cost + max(0, tk - d[k]), tk)
                if (mask | 1 << k, k) not in G or cand < G[(mask | 1 << k, k)]:
                    G[(mask | 1 << k, k)] = cand
    return min(G[((1 << n) - 1, j)][0] for j in range(n))


def repo_sequence(rule, p, fam, d, St):
    """The repository's rules.pick on a one-machine queue: arrival order = index, queued_at = index."""
    ws_ = SimpleNamespace(proc=lambda f, op, m: p[op], setup_time=lambda m, prev, f: 0 if prev is None else St[prev][f])
    shop = SimpleNamespace(ws=ws_)
    m = SimpleNamespace(idx=0, last_family=None,
                        queue=[SimpleNamespace(idx=k, op=k, family=fam[k], due=d[k], queued_at=k) for k in range(len(p))])
    seq = []
    while m.queue:
        j = rules.pick(rule, shop, m)
        seq.append(j.idx); m.queue.remove(j); m.last_family = j.family
    return seq


def fixed1(numer, denom):
    """numer / denom to one decimal, halves rounded up, in integers (the demo rounds the same way)."""
    q = (20 * numer + denom) // (2 * denom)
    return f"{q // 10}.{q % 10}"


def experiment(n, theta, with_setups, count=300, seed=2026):
    r = mulberry32(seed)
    St = SETUP0 if with_setups else ZERO
    keys_ = RULE_NAMES + ["best4"]
    hits, excess = {k: 0 for k in keys_}, {k: 0 for k in keys_}
    dp_agree, naive_wrong, opt_sum, zero_opt = 0, 0, 0, 0
    for _ in range(count):
        p, fam, d = instance(r, n, theta)
        opt = brute(p, fam, d, St)
        opt_sum += opt; zero_opt += opt == 0
        if not with_setups:
            dp_agree += subset_dp(p, d) == opt
        else:
            naive_wrong += naive_state_dp(p, fam, d, St) != opt
        vals = {k: total_tardiness(p, fam, d, repo_sequence(k, p, fam, d, St), St) for k in RULE_NAMES}
        vals["best4"] = min(vals.values())
        for k, v in vals.items():
            hits[k] += v == opt
            excess[k] += v - opt
    return dict(hits=hits, excess=excess, dp_agree=dp_agree, naive_wrong=naive_wrong, opt_sum=opt_sum,
                zero_opt=zero_opt, count=count)


def show(n, theta, ws_, res):
    h, e, c = res["hits"], res["excess"], res["count"]
    keys_ = RULE_NAMES + ["best4"]
    print(f"n = {n}, theta = {theta}, {'with' if ws_ else 'no  '} changeovers: optimal on "
          + ", ".join(f"{k} {fixed1(100 * h[k], c)}%" for k in keys_)
          + " | mean excess (min) " + ", ".join(f"{k} {fixed1(e[k], c)}" for k in keys_)
          + f" | mean optimum {fixed1(res['opt_sum'], c)}, optimum 0 on {res['zero_opt']}"
          + (f" | subset DP = brute force on {res['dp_agree']} of {c}" if not ws_ else
             f" | (set, last) DP wrong on {res['naive_wrong']} of {c}"))


section("Part 7 (setups) widget, defaults: 300 instances, n = 6, theta = 0.5, R = 0.6, seed 2026, machine 0's changeovers")
r = mulberry32(2026)
p, fam, d = instance(r, 6, 0.5)
print("first instance: p =", p, "families =", fam, "due =", d)
print("due dates: round(P * (1 - theta + 0.6 * (u - 0.5))), u uniform on [0, 1), clipped at 0: uniform on [(", round(1 - 0.3, 1),
      "- theta) P, (", round(1 + 0.3, 1), "- theta) P], P = sum of the p_j")
g = mulberry32(1)
print("mulberry32(1) first three draws (compare with JS.rng(1)):", [round(g(), 10) for _ in range(3)])
for ws_ in (False, True):
    show(6, 0.5, ws_, experiment(6, 0.5, ws_))

section("Part 7: the same experiment for every slider position (n = 4..7, theta = 0.2..0.8)")
for n in (4, 5, 6, 7):
    for theta in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for ws_ in (False, True):
            show(n, theta, ws_, experiment(n, theta, ws_))

# ------------------------------------------------------------------ Part 8: the subset recursion by hand
section("Part 8 (counting): F(S) by hand on X(10, 30), Y(20, 10), Z(30, 20), smallest sets first")
names3, p3, d3 = "XYZ", [10, 20, 30], [30, 10, 20]
F3, terms = {0: (0, "")}, 0
for size in (1, 2, 3):
    for mask in range(1, 8):
        if bin(mask).count("1") != size:
            continue
        pS = sum(p3[k] for k in range(3) if mask >> k & 1)
        cands = []
        for j in range(3):
            if mask >> j & 1:
                prev, order = F3[mask ^ (1 << j)]
                cands.append((prev + max(0, pS - d3[j]), order + names3[j], f"{names3[j]}: {prev} + {max(0, pS - d3[j])}"))
                terms += 1
        best_v = min(c[0] for c in cands)
        F3[mask] = min((c[0], c[1]) for c in cands)
        ties = sum(1 for c in cands if c[0] == best_v)
        print(f"S = {', '.join(names3[k] for k in range(3) if mask >> k & 1):8s} p(S) = {pS:2d}  " + "; ".join(c[2] for c in cands)
              + f"  ->  F(S) = {best_v} ({F3[mask][1] if ties == 1 else 'a tie'})")
print("optimum", F3[7][0], "order", F3[7][1], "; terms of the recursion:", terms, "; listing the 6 orders:", 6 * 3, "terms")

# ------------------------------------------------------------------ Exercise 5: where the (set, last job) state fails
section("Exercise 5 (and Part 8): a four-job instance where the (set, last job) recursion is wrong, machine 0's changeovers")
p4, f4, d4 = [5, 10, 5, 5], [3, 3, 3, 2], [15, 35, 35, 30]


def finishes(seq, p_, f_, St):
    t, prev, out = 0, None, []
    for j in seq:
        t += (0 if prev is None else St[prev][f_[j]]) + p_[j]
        prev = f_[j]
        out.append(t)
    return out


print("p =", p4, "families =", f4, "d =", d4, "; s_0(2,3) =", SETUP0[2][3], ", s_0(3,2) =", SETUP0[3][2])
opt4 = brute(p4, f4, d4, SETUP0)
print("optimum (search over orders):", opt4, "; optimal orders:",
      [q for q in permutations(range(4)) if total_tardiness(p4, f4, d4, q, SETUP0) == opt4],
      "; (set, last) recursion:", naive_state_dp(p4, f4, d4, SETUP0))
print("3,0,1,2 finishes at", finishes((3, 0, 1, 2), p4, f4, SETUP0), "tardiness", total_tardiness(p4, f4, d4, (3, 0, 1, 2), SETUP0))
for pre in ((3, 0, 1), (0, 3, 1)):
    print(f"partial order {pre}: tardiness {total_tardiness(p4, f4, d4, pre, SETUP0)}, done at {finishes(pre, p4, f4, SETUP0)[-1]}; "
          f"then job 2 ends at {finishes(pre + (2,), p4, f4, SETUP0)[-1]}, total {total_tardiness(p4, f4, d4, pre + (2,), SETUP0)}")
print("why nothing is on time: job 3 first -> earliest end of job 0:", min(finishes(q, p4, f4, SETUP0)[q.index(0)] for q in permutations(range(4)) if q[0] == 3),
      "; job 3 last -> it ends at", finishes((0, 1, 2, 3), p4, f4, SETUP0)[-1],
      "; job 3 in the middle -> last job ends at", min(finishes(q, p4, f4, SETUP0)[-1] for q in permutations(range(4)) if q[0] != 3 and q[-1] != 3))

# ------------------------------------------------------------------ Part 9 (code): how many decisions a simulated day makes
section("Part 9 (code): calls to rules.pick on one simulated day (seed 0, 60 jobs, SST, no breakdowns)")
from jobshop.rules import RuleController  # noqa: E402
from jobshop.scenario import make_scenario  # noqa: E402
from jobshop.shop import Shop  # noqa: E402


class CountingController(RuleController):
    def __init__(self, rule):
        super().__init__(rule)
        self.calls, self.queue_sizes = 0, []

    def sequence(self, shop, m):
        self.calls += 1
        self.queue_sizes.append(len(m.queue))
        return super().sequence(shop, m)


sc = make_scenario(ws, 0)
cc = CountingController("SST")
Shop(ws, sc, cc).run()
print("jobs", len(sc.jobs), "; operations", sum(ws.n_ops(j.family) for j in sc.jobs), "; calls to pick", cc.calls,
      "; largest queue scanned", max(cc.queue_sizes), "; mean queue scanned", round(sum(cc.queue_sizes) / len(cc.queue_sizes), 2))
