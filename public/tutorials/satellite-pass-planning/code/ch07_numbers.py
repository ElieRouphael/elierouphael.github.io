"""Chapter 07, A Greedy Plan and What It Leaves Behind: every number quoted on the page.

Prints, labelled, the greedy result on the course's small instance and its decisions, the gaps on the
three small seeds, the hand-made counterexamples (checked with CP-SAT), the other orderings (the same
greedy with a different target order, including random orders drawn with the page's random generator,
a port of Sat.rng), and the drop-and-refill local improvement, on the small and the medium instance.
Writes nothing: the page recomputes the small-instance numbers live from interactive/instance-small.js
and must agree with what this script prints.

Usage:  python code/ch07_numbers.py
satplan is imported from the course archive (satplan/ next to code/) or, failing that, from the
repository path in SATPLAN_REPO (default: the author's working copy).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent, Path(os.environ.get("SATPLAN_REPO", r"C:/Users/rouphaee/Documents/PostDoc_v2/satellite-pass-planning"))):
    if (cand / "satplan").is_dir():
        sys.path.insert(0, str(cand))
        break

import numpy as np  # noqa: E402

from satplan.checker import check_plan, plan_value  # noqa: E402
from satplan.cpsat import solve_cpsat  # noqa: E402
from satplan.greedy import solve_greedy  # noqa: E402
from satplan.instance import Opportunity, PlanningInstance  # noqa: E402
from satplan.scenario import ScenarioConfig, build_instance  # noqa: E402

SMALL = dict(n_sats=4, n_targets=120, horizon_h=6.0)
MEDIUM = dict(n_sats=8, n_targets=250, horizon_h=12.0)


# ------------------------------------------------------------------ the page's random generator
class PageRng:
    """Sat.rng(seed) from interactive/sat.js, bit for bit (xorshift128+ on 32-bit words)."""

    def __init__(self, seed: int):
        M = 0xFFFFFFFF
        self.s0 = (seed & M) or 1
        self.s1 = (0x9E3779B9 ^ self.s0) & M

    def next(self) -> float:
        M = 0xFFFFFFFF
        x, y = self.s0, self.s1
        self.s0 = y
        x = (x ^ (x << 23)) & M
        x = (x ^ (x >> 17)) & M
        x = (x ^ y ^ (y >> 26)) & M
        self.s1 = x
        return ((self.s0 + self.s1) & M) / 4294967296


def shuffled(items, rng: PageRng):
    """Fisher-Yates from the end, as ch07.js does: j = floor(next() * (k + 1))."""
    a = list(items)
    for k in range(len(a) - 1, 0, -1):
        j = int(rng.next() * (k + 1))
        a[k], a[j] = a[j], a[k]
    return a


# ------------------------------------------------------------------ greedy with a given target order
class Greedy:
    """satplan.greedy.solve_greedy, split so that the target order can be chosen and a partial plan can be
    refilled. With the package's order it reproduces solve_greedy exactly (checked in main)."""

    def __init__(self, inst: PlanningInstance):
        self.inst = inst
        self.opps = inst.opportunities
        self.by_target = inst.by_target()
        n = len(self.opps)
        self.tr = {}
        bysat = inst.by_satellite()
        for ids in bysat.values():
            for a in ids:
                for b in ids:
                    if a != b:
                        self.tr[a, b] = inst.transition(a, b)

    def groups(self, o):
        return [("duty", o.sat, o.revolution), ("memory", o.sat, o.memory_segment)]

    def limit(self, g):
        return self.inst.duty_budget_s if g[0] == "duty" else self.inst.memory_budget_s

    def run(self, order, plan=None, log=None, room_first=False):
        """Insert the targets of `order` (skipping those already in `plan`) one by one, as greedy.py does.
        `log`, if a list, receives (target, outcome) with outcome 'placed', 'budget', 'blocked' or 'mixed'."""
        opps, tr = self.opps, self.tr
        plan = dict(plan or {})
        schedule = {s: [] for s in range(self.inst.n_sats)}
        used = {}
        taken = set()
        for i, t in plan.items():
            o = opps[i]
            schedule[o.sat].append((t, i))
            taken.add(o.target)
            for g in self.groups(o):
                used[g] = used.get(g, 0) + o.duration

        def fits(o, t):
            for t_p, p in schedule[o.sat]:
                if t >= t_p:
                    if t < t_p + opps[p].duration + tr[p, o.idx]:
                        return False
                elif t + o.duration + tr[o.idx, p] > t_p:
                    return False
            return True

        for j in order:
            if j in taken:
                continue
            seen = set()
            placed = False
            if room_first:  # exercise 1: the opportunity whose revolution has the most power budget left
                key = lambda i: (used.get(("duty", opps[i].sat, opps[i].revolution), 0), opps[i].window_start)
            else:
                key = lambda i: opps[i].window_start
            for i in sorted(self.by_target[j], key=key):
                o = opps[i]
                if any(used.get(g, 0) + o.duration > self.limit(g) for g in self.groups(o)):
                    seen.add("budget")
                    continue
                cands = [o.earliest_start] + [t_p + opps[p].duration + tr[p, i] for t_p, p in schedule[o.sat]]
                cands = sorted(c for c in cands if o.earliest_start <= c <= o.latest_start)
                t_ok = next((c for c in cands if fits(o, c)), None)
                if t_ok is None:
                    seen.add("blocked")
                    continue
                plan[i] = float(t_ok)
                schedule[o.sat].append((float(t_ok), i))
                taken.add(j)
                for g in self.groups(o):
                    used[g] = used.get(g, 0) + o.duration
                placed = True
                break
            if log is not None:
                log.append((j, "placed" if placed else (seen.pop() if len(seen) == 1 else "mixed")))
        return plan

    def value(self, plan):
        return int(sum(self.opps[i].weight for i in plan))

    # target orders -------------------------------------------------------------
    def first_seen(self):
        return list(self.by_target)  # dict order: order of first appearance among the opportunities

    def order_by(self, key):
        base = self.first_seen()
        rank = {j: k for k, j in enumerate(base)}
        return sorted(base, key=lambda j: key(j) + (rank[j],))

    def orders(self):
        W, D, n = self.inst.target_weights, self.inst.target_durations, lambda j: len(self.by_target[j])
        return {
            "density": self.order_by(lambda j: (-W[j] / D[j], n(j))),
            "weight": self.order_by(lambda j: (-W[j], n(j))),
            "shortest": self.order_by(lambda j: (D[j], -W[j])),
            "fewest": self.order_by(lambda j: (n(j), -W[j] / D[j])),
        }


def one_for_two(G: Greedy, plan: dict, slack: int = 600):
    """Exercise 2: remove one task r and insert two refused targets (any pair with an opportunity on r's satellite
    whose window is within `slack` s of r's), each at its earliest feasible start; accept if they are worth more
    than r. Tasks in increasing opportunity index, pairs in increasing target order; repeat until no move helps."""
    opps = G.opps
    plan = dict(plan)
    moves = 0
    while True:
        done = False
        taken = {opps[i].target for i in plan}
        for r in sorted(plan):
            o_r = opps[r]
            near = sorted({j for j in G.by_target if j not in taken for i in G.by_target[j]
                           if opps[i].sat == o_r.sat and opps[i].window_end >= o_r.window_start - slack
                           and opps[i].window_start <= o_r.window_end + slack})
            rest = {i: t for i, t in plan.items() if i != r}
            for a in range(len(near)):
                for b in range(a + 1, len(near)):
                    cand = G.run([near[a], near[b]], rest)
                    if len(cand) == len(rest) + 2 and G.value(cand) > G.value(plan):
                        plan, moves, done = cand, moves + 1, True
                        break
                if done:
                    break
            if done:
                break
        if not done:
            return plan, moves


def random_restarts(G: Greedy, n_orders: int, seed: int):
    """n_orders random target orders from Sat.rng(seed); returns the list of values and the best plan."""
    rng = PageRng(seed)
    base = G.first_seen()
    vals, best, best_plan = [], -1, None
    for _ in range(n_orders):
        p = G.run(shuffled(base, rng))
        v = G.value(p)
        vals.append(v)
        if v > best:
            best, best_plan = v, p
    return vals, best_plan


def drop_and_refill(G: Greedy, plan: dict, order):
    """Local improvement: remove one task, refill with the greedy (same target order), keep the result if it
    collects more. Tasks are tried in increasing opportunity index; restart after each improvement; stop when
    no single removal helps. Returns the final plan and the list of (value, removed opportunity)."""
    plan = dict(plan)
    history = []
    improved = True
    while improved:
        improved = False
        v0 = G.value(plan)
        for r in sorted(plan):
            rest = {i: t for i, t in plan.items() if i != r}
            cand = G.run(order, rest)
            if G.value(cand) > v0:
                plan = cand
                history.append((G.value(cand), r))
                improved = True
                break
    return plan, history


def ruin_and_recreate(G: Greedy, plan: dict, iters: int, seed: int, window_s: float):
    """Large-neighbourhood search with the page's random numbers: pick a satellite and a stretch of `window_s`
    seconds at random, remove the tasks that start in it, refill with the greedy in a random target order, and
    keep the result if it collects at least as much. Returns the final plan and the (iteration, value) improvements."""
    rng = PageRng(seed)
    opps, base = G.opps, G.first_seen()
    cur, v = dict(plan), G.value(plan)
    hist = [(0, v)]
    for it in range(1, iters + 1):
        s = int(rng.next() * G.inst.n_sats)
        t0 = rng.next() * (G.inst.horizon_s - window_s)
        rest = {i: t for i, t in cur.items() if not (opps[i].sat == s and t0 <= t < t0 + window_s)}
        cand = G.run(shuffled(base, rng), rest)
        vc = G.value(cand)
        if vc >= v:
            if vc > v:
                hist.append((it, vc))
            cur, v = cand, vc
    return cur, hist


# ------------------------------------------------------------------ hand-made counterexamples
def toy(tasks, n_sats=1, duty=10_000, memory=10_000):
    """tasks: list of (target, weight, duration, ws, we, revolution, pointing angle in deg about z).
    Pointing angle gives the reconfiguration time 10 s + |angle difference| / 2."""
    opps, W, D = [], {}, {}
    for k, (j, w, d, ws, we, rev, ang) in enumerate(tasks):
        u = (float(np.cos(np.radians(ang))), float(np.sin(np.radians(ang))), 0.0)
        opps.append(Opportunity(k, 0, j, ws, we, d, w, u, rev, 0))
        W[j], D[j] = w, d
    nt = max(W) + 1
    inst = PlanningInstance(n_sats, nt, 10_000, opps, 10.0, 2.0, duty, memory,
                            [W.get(j, 0) for j in range(nt)], [D.get(j, 1) for j in range(nt)])
    inst.build_pairs()
    return inst


COUNTER = {
    # A dense task whose window sits between two others: it blocks both.
    "block": dict(tasks=[(0, 4, 30, 25, 55, 0, 0), (1, 3, 30, 0, 40, 0, 0), (2, 3, 30, 50, 90, 0, 0)]),
    # A task with two passes takes the first one and fills that revolution's budget.
    "early": dict(tasks=[(0, 4, 60, 0, 120, 0, 0), (0, 4, 60, 300, 420, 1, 0), (1, 3, 60, 130, 260, 0, 0)],
                  duty=100),
    # Value density on a knapsack: a short dense task leaves no room for a long valuable one.
    "knap": dict(tasks=[(0, 2, 20, 0, 150, 0, 0), (1, 9, 95, 0, 150, 0, 0)], duty=100),
}


def main() -> None:
    np.set_printoptions(precision=3)
    print("== small instance, seed 0 ==")
    inst, meta = build_instance(ScenarioConfig(seed=0, **SMALL))
    g = solve_greedy(inst)
    G = Greedy(inst)
    orders = G.orders()
    log = []
    mine = G.run(orders["density"], log=log)
    assert mine == g["plan"], "the order-parameterised greedy must reproduce solve_greedy"
    assert not check_plan(inst, mine)
    c = solve_cpsat(inst, time_limit_s=60.0, workers=8, hint=g["plan"], seed=0)
    opt = c["objective"]
    print(f"greedy value {g['objective']}, tasks {len(g['plan'])}, time {g['time_s']:.3f} s")
    print(f"cpsat value {opt} bound {c['bound']:.0f} status {c['status']} tasks {len(c['plan'])}")
    print(f"greedy below optimum: {100 * (opt - g['objective']) / opt:.1f} %  (missing {opt - g['objective']})")
    print(f"targets considered (reachable): {len(log)}")
    from collections import Counter
    print("outcomes:", dict(Counter(o for _, o in log)))
    wsum = {k: sum(inst.target_weights[j] for j, o in log if o == k) for k in ("placed", "budget", "blocked", "mixed")}
    print("weight by outcome:", wsum)
    # the optimal plan the page draws is the one exported in interactive/instance-small.js
    import json
    js = (HERE.parent / "interactive" / "instance-small.js").read_text(encoding="utf-8")
    exported = json.loads(js[js.index("{"):js.rstrip().rindex(";")])["python"]["cpsat"]
    cplan = {int(k): v for k, v in exported["plan"].items()}
    assert not check_plan(inst, cplan) and plan_value(inst, cplan) == opt
    print(f"exported optimal plan: value {exported['value']}, {len(cplan)} tasks")
    gt = {inst.opportunities[i].target for i in g["plan"]}
    ct = {inst.opportunities[i].target for i in cplan}
    print(f"targets: greedy {len(gt)}, optimum {len(ct)}, common {len(gt & ct)}, only optimum {len(ct - gt)} "
          f"(weight {sum(inst.target_weights[j] for j in ct - gt)}), only greedy {len(gt - ct)} "
          f"(weight {sum(inst.target_weights[j] for j in gt - ct)})")
    W, D = inst.target_weights, inst.target_durations
    print("only optimum (target, weight, duration):", sorted((j, W[j], D[j]) for j in ct - gt))
    print("only greedy (target, weight, duration):", sorted((j, W[j], D[j]) for j in gt - ct))
    tiers = {}
    for j, o in log:
        tiers.setdefault((W[j], D[j]), {}).setdefault(o, 0)
        tiers[W[j], D[j]][o] += 1
    print("outcomes by (weight, duration) in greedy order:", tiers)
    print("total weight of opportunities (every x = 1):", sum(o.weight for o in inst.opportunities))
    dens = [round(inst.target_weights[j] / inst.target_durations[j], 4) for j in orders["density"]]
    print("distinct densities in order:", sorted(set(dens), reverse=True))
    # first decisions
    for j, o in log[:6]:
        print(f"  step target {j} w {inst.target_weights[j]} d {inst.target_durations[j]} -> {o}")
    k_first_reject = next(k for k, (j, o) in enumerate(log) if o != "placed")
    print(f"first rejection at step {k_first_reject + 1}: target {log[k_first_reject][0]} ({log[k_first_reject][1]})")
    # running value at checkpoints
    run_v, acc = [], 0
    pl = iter(sorted(mine))
    for j, o in log:
        if o == "placed":
            acc += inst.target_weights[j]
        run_v.append(acc)
    print("value after 10/20/40/60/80/114 steps:", [run_v[k - 1] for k in (10, 20, 40, 60, 80, len(log))])

    print(f"2^168 = {2.0 ** len(inst.opportunities):.2e} subsets; at 1e9 per second: "
          f"{2.0 ** len(inst.opportunities) / 1e9 / 3.156e7:.1e} years")
    print("\n== the three small seeds (exercises 1 and 3: other opportunity rule, other tie-breaks) ==")
    for seed in (0, 1, 2):
        i2, _ = build_instance(ScenarioConfig(seed=seed, **SMALL))
        g2 = solve_greedy(i2)
        G2 = Greedy(i2)
        W2, D2, n2 = i2.target_weights, i2.target_durations, lambda j: len(G2.by_target[j])
        no_tie = G2.order_by(lambda j: (-W2[j] / D2[j],))
        rev_tie = G2.order_by(lambda j: (-W2[j] / D2[j], -n2(j)))
        room = G2.run(G2.orders()["density"], room_first=True)
        vals = [G2.value(G2.run(o)) for o in (no_tie, rev_tie)]
        for p_ in (room, G2.run(no_tie), G2.run(rev_tie)):
            assert not check_plan(i2, p_)
        print(f"seed {seed}: greedy {g2['objective']} time {g2['time_s']:.3f} s; most-room opportunity rule "
              f"{G2.value(room)}; no tie-break {vals[0]}; most opportunities first {vals[1]}")

    import math
    print(f"number of target orders: {len(log)}! = 10^{math.lgamma(len(log) + 1) / math.log(10):.1f}")
    print("\n== other orderings, small instance ==")
    for name, od in orders.items():
        p = G.run(od)
        assert not check_plan(inst, p)
        print(f"order {name:<9} value {G.value(p)}  tasks {len(p)}  gap {100 * (opt - G.value(p)) / opt:.1f} %")
    for n_orders in (1, 10, 100, 1000):
        vals, bp = random_restarts(G, n_orders, seed=7)
        assert not check_plan(inst, bp)
        print(f"random orders, Sat.rng(7), first {n_orders}: best {max(vals)}  mean {np.mean(vals):.1f}  "
              f"min {min(vals)}  median {np.median(vals):.0f}  share >= 193: {np.mean(np.array(vals) >= 193):.3f}")
    t0 = time.perf_counter()
    vals, _ = random_restarts(G, 1000, seed=7)
    print(f"  1000 random orders took {time.perf_counter() - t0:.2f} s in Python")
    # histogram of values for the page (bins of 5)
    h = Counter((v // 5) * 5 for v in vals)
    print("  histogram (bin start: count):", dict(sorted(h.items())))

    print("\n== drop-and-refill local improvement, small instance ==")
    t0 = time.perf_counter()
    lp, hist = drop_and_refill(G, mine, orders["density"])
    assert not check_plan(inst, lp)
    print(f"from greedy {G.value(mine)}: improvements {hist}  final {G.value(lp)} "
          f"gap {100 * (opt - G.value(lp)) / opt:.1f} %  ({time.perf_counter() - t0:.2f} s)")
    vals, bp = random_restarts(G, 100, seed=7)
    lp2, hist2 = drop_and_refill(G, bp, orders["density"])
    assert not check_plan(inst, lp2)
    print(f"from the best of 100 random orders ({G.value(bp)}): final {G.value(lp2)}, steps {len(hist2)}")

    t0 = time.perf_counter()
    sp, sm = one_for_two(G, mine)
    assert not check_plan(inst, sp)
    print(f"exercise 2, 1-for-2 swaps from greedy: {G.value(sp)} after {sm} moves ({time.perf_counter() - t0:.1f} s)")
    ti = toy(COUNTER["early"]["tasks"], duty=100)
    Gt = Greedy(ti)
    print("exercise 1 on the 'early' toy, most-room rule:", Gt.value(Gt.run(Gt.orders()["density"], room_first=True)))

    print("\n== ruin and recreate (page defaults: 60 min stretch, Sat.rng(1), 2000 iterations) ==")
    for win_min in (10, 30, 60, 90):
        t0 = time.perf_counter()
        rp, rh = ruin_and_recreate(G, mine, 2000, 1, win_min * 60)
        assert not check_plan(inst, rp)
        print(f"stretch {win_min} min, seed 1: final {G.value(rp)} (gap {100 * (opt - G.value(rp)) / opt:.1f} %), "
              f"improvements {rh}  ({time.perf_counter() - t0:.2f} s)")
    finals = []
    for seed in range(1, 11):
        rp, rh = ruin_and_recreate(G, mine, 2000, seed, 3600)
        finals.append(G.value(rp))
    print(f"stretch 60 min, seeds 1..10: {finals}  min {min(finals)} max {max(finals)}")
    rp, rh = ruin_and_recreate(G, mine, 10000, 1, 3600)
    print(f"stretch 60 min, seed 1, 10000 iterations: {G.value(rp)}, last improvement at {rh[-1]}")

    print("\n== counterexamples (greedy vs CP-SAT optimum) ==")
    for name, spec in COUNTER.items():
        ti = toy(spec["tasks"], duty=spec.get("duty", 10_000))
        tg = solve_greedy(ti)
        tc = solve_cpsat(ti, time_limit_s=10, workers=1)
        assert not check_plan(ti, tg["plan"]) and not check_plan(ti, tc["plan"])
        print(f"{name}: greedy {tg['objective']} plan {tg['plan']}   optimum {tc['objective']} ({tc['status']}) "
              f"plan {tc['plan']}")
    # the blocking example as the dense task's weight varies (page slider: 1..8)
    row = []
    for wa in range(1, 9):
        tasks = [(0, wa, 30, 25, 55, 0, 0), (1, 3, 30, 0, 40, 0, 0), (2, 3, 30, 50, 90, 0, 0)]
        ti = toy(tasks)
        row.append((wa, solve_greedy(ti)["objective"], solve_cpsat(ti, 10, workers=1)["objective"]))
    print("block, weight of A -> (greedy, optimum):", row)
    for name in ("early", "knap"):
        row = []
        for b in range(60 if name == "early" else 20, 201, 10 if name == "early" else 5):
            ti = toy(COUNTER[name]["tasks"], duty=b)
            row.append((b, solve_greedy(ti)["objective"], solve_cpsat(ti, 10, workers=1)["objective"]))
        print(f"{name}, power budget -> (greedy, optimum):", row)

    print("\n== medium instance, seed 0 ==")
    im, _ = build_instance(ScenarioConfig(seed=0, **MEDIUM))
    gm = solve_greedy(im)
    Gm = Greedy(im)
    om = Gm.orders()
    assert Gm.run(om["density"]) == gm["plan"]
    print(f"greedy {gm['objective']} in {gm['time_s']:.3f} s; benchmark CP-SAT 557, best bound 575 (results.md)")
    for name, od in om.items():
        p = Gm.run(od)
        assert not check_plan(im, p)
        print(f"order {name:<9} value {Gm.value(p)}")
    t0 = time.perf_counter()
    vals, bp = random_restarts(Gm, 100, seed=7)
    print(f"100 random orders: best {max(vals)} mean {np.mean(vals):.1f} min {min(vals)} "
          f"({time.perf_counter() - t0:.1f} s)")
    t0 = time.perf_counter()
    lpm, histm = drop_and_refill(Gm, gm["plan"], om["density"])
    assert not check_plan(im, lpm)
    print(f"drop-and-refill from greedy: {Gm.value(lpm)} after {len(histm)} improvements "
          f"({time.perf_counter() - t0:.1f} s)")
    t0 = time.perf_counter()
    rp, rh = ruin_and_recreate(Gm, gm["plan"], 2000, 1, 3600)
    assert not check_plan(im, rp)
    print(f"ruin and recreate, 60 min, seed 1, 2000 iterations: {Gm.value(rp)} ({time.perf_counter() - t0:.1f} s), "
          f"last improvements {rh[-3:]}")


if __name__ == "__main__":
    main()
