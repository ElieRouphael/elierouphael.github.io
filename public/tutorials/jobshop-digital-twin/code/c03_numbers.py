"""Reproduce every number quoted on chapter C3 of "Which Job Should the Machine Do Next?",
"Rescheduling on a rolling horizon" (c03-rescheduling-on-a-rolling-horizon.html).

Usage:
    python c03_numbers.py          # print every number of the page (about 40 s)
    python c03_numbers.py --js     # also print the data embedded in interactive/c03.js

Needs the repository (github.com/ElieRouphael/jobshop-digital-twin) and its requirements
(simpy, ortools, numpy). Set JOBSHOP_REPO to the repository folder, or run the script from the
course zip, where jobshop/ sits next to code/. The benchmark results are read from
results/runs.csv (never rerun here); everything else is a short simulation.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("JOBSHOP_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from jobshop import cpsat  # noqa: E402
from jobshop.checker import check_schedule  # noqa: E402
from jobshop.metrics import bootstrap_ci, run_metrics  # noqa: E402
from jobshop.scenario import LEVELS, Scenario, make_scenario, make_workshop  # noqa: E402
from jobshop.shop import JobState, Shop  # noqa: E402

EMIT_JS = "--js" in sys.argv
WS = make_workshop()
SETTINGS = {"det_time": 0.2, "allowance": 2, "max_jobs": 25, "strict": True, "workers": 8}  # scripts/evaluate.py
T_START = time.perf_counter()


def section(title):
    print(f"\n=== {title}")


# ==============================================================================================
section("Part 1 (idea): the paint machine with a surprise order E")
# One machine set up for blue, colour change 10 min (notation.md). Orders A-D wait at t = 0; order E
# arrives later. Three ways to run the machine: plan once at t = 0 and append E at the end (strict plan
# following), re-plan when E arrives (orders already started are frozen), and a clairvoyant plan that
# knew E from the start. The same computation runs in the browser (interactive/c03.js).
PAINT = [("A", "blue", 20, 30, 0), ("B", "red", 10, 25, 0), ("C", "blue", 15, 70, 0), ("D", "red", 25, 60, 0)]
SETUP = 10


def run(seq, orders, t0=0, colour="blue"):
    """(rows, total tardiness): a changeover starts once the machine is free and the order is there."""
    by = {o[0]: o for o in orders}
    t, c, rows, tot = t0, colour, [], 0
    for name in seq:
        _, col, p, due, rel = by[name]
        free = max(t, rel)
        start = free + (0 if col == c else SETUP)
        end = start + p
        rows.append((name, free, start, end, max(0, end - due)))
        tot += max(0, end - due)
        t, c = end, col
    return rows, tot


def best(orders, names, t0, colour):
    out = None
    for seq in itertools.permutations(names):
        _, v = run(seq, orders, t0, colour)
        if out is None or v < out[1] or (v == out[1] and "".join(seq) < "".join(out[0])):
            out = (seq, v)
    return out


def surprise(e_colour="red", e_p=5, e_due=50, e_arr=35):
    orders = PAINT + [("E", e_colour, e_p, e_due, e_arr)]
    once_seq, once_plan_value = best(PAINT, "ABCD", 0, "blue")
    rows, _ = run(once_seq, PAINT)
    # plan once: E waits until the plan is done
    once_rows, once = run(list(once_seq) + ["E"], orders)
    # re-plan at e_arr: orders whose changeover began before e_arr are frozen
    frozen = [r for r in rows if r[1] < e_arr]
    t_free = frozen[-1][3] if frozen else e_arr
    colour = ({o[0]: o[1] for o in PAINT}[frozen[-1][0]]) if frozen else "blue"
    rest = [n for n in "ABCD" if n not in {r[0] for r in frozen}] + ["E"]
    tail_seq, _ = best(orders, rest, max(t_free, e_arr), colour)
    re_seq = [r[0] for r in frozen] + list(tail_seq)
    re_rows, replan = run(re_seq, orders)
    clair_seq, clair = best(orders, "ABCDE", 0, "blue")
    return dict(once_seq="".join(once_seq), once_plan=once_plan_value, once=once, frozen="".join(r[0] for r in frozen),
                t_free=t_free, re_seq="".join(re_seq), replan=replan, clair_seq="".join(clair_seq), clair=clair,
                once_rows=once_rows, re_rows=re_rows)


DEFAULT_E = ("blue", 5, 50, 35)   # the widget's default surprise order
for args, label in ((DEFAULT_E, "default of the widget"), (("red", 5, 50, 35), "the same order in red"),
                    (("red", 5, 50, 5), "red, arriving at 5"), (("blue", 10, 45, 15), "blue, 10 min, due 45, arriving at 15")):
    x = surprise(*args)
    print(f"E = {args} ({label}):")
    print(f"  plan at t = 0 for A-D: {x['once_seq']}, total tardiness {x['once_plan']}")
    print(f"  plan once, E appended: {x['once_seq']}E, total {x['once']}  rows (name, free, start, end, late) {x['once_rows']}")
    print(f"  re-plan at {args[3]}: frozen {x['frozen']} (machine free at {x['t_free']}), sequence {x['re_seq']}, total {x['replan']}"
          f"  rows {x['re_rows']}")
    print(f"  clairvoyant (knew E at t = 0): {x['clair_seq']}, total {x['clair']}; re-planning recovers "
          f"{x['once'] - x['replan']} of the {x['once'] - x['clair']} minutes the clairvoyant saves")

# hand check of the default re-plan at 35: B is in progress until 40 (red); remaining C, D, E
print("re-plan at 35, default E: all six orders of C, D, E after B (machine free at 40, set up for red):")
for seq in itertools.permutations("CDE"):
    rows, v = run(list(seq), PAINT + [("E",) + DEFAULT_E], 40, "red")
    print("  ", "".join(seq), [(r[0], r[2], r[3], r[4]) for r in rows], "tardiness of C, D, E:", v, " total with A, B:", v + 15)
x = surprise(*DEFAULT_E)
print("the clairvoyant searches all", len(list(itertools.permutations("ABCDE"))), "orders of A-E")
rows, v = run(list(x["clair_seq"]), PAINT + [("E",) + DEFAULT_E])
print("clairvoyant rows (name, free, start, end, late):", rows)


# ==============================================================================================
section("Hero: one simulated morning, test day 0, frequent breakdowns, CP-SAT re-planning")
# The first 12 orders of test day 0. The 13th arrives at minute 314: before that minute the controller has
# seen exactly what it sees in the full 60-order day, so the replay of minutes [0, 314) is the full day's.
FULL = make_scenario(WS, 0, n_jobs=60, mtbf=LEVELS["frequent breakdowns"])
N_MORNING = 12
T_END = FULL.jobs[N_MORNING].arrival
MORNING = Scenario(FULL.seed, FULL.jobs[:N_MORNING], FULL.downtime, FULL.mttr)
print(f"first {N_MORNING} orders, arrivals", [j.arrival for j in MORNING.jobs], f"; order {N_MORNING} arrives at {T_END}")
print("orders (idx, family, arrival, due):", [(j.idx, j.family, j.arrival, j.due) for j in MORNING.jobs])
print("downtime before minute", T_END, ":", {m: [s for s in FULL.downtime[m] if s[0] < T_END] for m in range(5)})


class Recorder(cpsat.CPSATController):
    """The repository's controller, unchanged, with a record of every event, snapshot and plan."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.events = []

    def on_event(self, shop, kind, **info):
        who = info["job"].idx if kind == "arrival" else info["machine"].idx
        states = [(j.idx, j.state.value, j.op, j.dest, j.eta) for j in sorted(shop.active_jobs(), key=lambda j: j.idx)]
        machines = [(m.down, m.current.idx if m.current else None, len(m.queue)) for m in shop.machines]
        n_active = len(shop.active_jobs())
        rec = {"t": shop.now, "kind": kind, "who": who, "states": states, "machines": machines, "n_active": n_active}
        _CAPTURE.clear()
        super().on_event(shop, kind, **info)
        snap, plan = _CAPTURE[-1] if _CAPTURE else (None, None)
        rec["snap"], rec["plan"] = snap, dict(plan.ops) if plan else {}
        rec["status"], rec["objective"] = (plan.status, plan.objective) if plan else ("EMPTY", None)
        rec["warm"] = cpsat.warm_start(shop.ws, snap, self.allowance, self._prev) if snap and snap.jobs else None
        self._prev = dict(self.plan)
        self.events.append(rec)

    _prev: dict = {}


_CAPTURE: list = []
_solve = cpsat.solve_snapshot


def _recording_solve(ws, snap, **kw):
    plan = _solve(ws, snap, **kw)
    _CAPTURE.append((snap, plan))
    return plan


cpsat.solve_snapshot = _recording_solve  # the controller looks the function up in its module at call time

t0 = time.perf_counter()
cp = Recorder(**SETTINGS)
cp_log = Shop(WS, MORNING, cp).run()
cp_wall = time.perf_counter() - t0
gr = Recorder(**SETTINGS, use_solver=False)
gr_log = Shop(WS, MORNING, gr).run()
for name, sc, log in (("CP-SAT", MORNING, cp_log), ("Greedy replan", MORNING, gr_log)):
    assert check_schedule(WS, sc, log) == [], name
print("both runs pass the independent checker")

ev = [e for e in cp.events if e["t"] < T_END]
ev_g = [e for e in gr.events if e["t"] < T_END]
assert [(e["t"], e["kind"], e["who"]) for e in ev] == [(e["t"], e["kind"], e["who"]) for e in ev_g]
kinds = {k: sum(e["kind"] == k for e in ev) for k in ("arrival", "breakdown", "repair")}
print(f"events before minute {T_END}: {len(ev)} ({kinds['arrival']} arrivals, {kinds['breakdown']} breakdowns, "
      f"{kinds['repair']} repairs), the same for both controllers")
print(f"one event every {T_END / len(ev):.1f} minutes on average")
planned = [e for e in ev if e["snap"] and e["snap"].jobs]
print(f"replans with at least one job to plan: {len(planned)}; with none: {len(ev) - len(planned)} "
      f"(the first arrival is at minute {MORNING.jobs[0].arrival})")
print("jobs in the shop at an event: at most", max(e["n_active"] for e in ev), "; jobs planned: at most",
      max(len(e["snap"].jobs) for e in planned))
statuses = {s: sum(e["status"] == s for e in planned) for s in ("OPTIMAL", "FEASIBLE", "GREEDY")}
print("CP-SAT status over these replans:", statuses)
better = sum(e["warm"] is not None and e["objective"] < e["warm"].objective for e in planned)
print(f"replans where CP-SAT improved on the greedy warm start: {better} of {len(planned)}")
gains = [e["warm"].objective - e["objective"] for e in planned]
print("objective gain over the warm start (10 sum T + sum C): total", sum(gains), " largest", max(gains))
print(f"morning runs to completion (all {N_MORNING} orders): CP-SAT mean tardiness "
      f"{run_metrics(WS, MORNING, cp_log)['mean_tardiness']:.2f}, greedy replan {run_metrics(WS, MORNING, gr_log)['mean_tardiness']:.2f} "
      f"(these 12-order runs are not the benchmark; CP-SAT run took {cp_wall:.0f} s of wall clock here)")

# Hero hint, checked: the full 60-order day, stopped at minute T_END, makes the same events, the same
# snapshots' plans, statuses and objectives as the 12-order morning (both controllers).


class _Stop(Exception):
    pass


class UntilTEnd(Recorder):
    def on_event(self, shop, kind, **info):
        if shop.now >= T_END:
            raise _Stop
        super().on_event(shop, kind, **info)


for name, kw, morning_events in (("CP-SAT", {}, ev), ("Greedy replan", {"use_solver": False}, ev_g)):
    full = UntilTEnd(**SETTINGS, **kw)
    try:
        Shop(WS, FULL, full).run()
    except _Stop:
        pass
    sig = lambda es: [(e["t"], e["kind"], e["who"], sorted(e["plan"].items()), e["status"], e["objective"]) for e in es if e["t"] < T_END]
    assert sig(full.events) == sig(morning_events), name
    print(f"hero check, {name}: the full 60-order day up to minute {T_END} has the same {len(sig(full.events))} events, "
          f"plans, statuses and objectives as the 12-order morning")

# Part 3 (loop), "try this" pointers of the hero: queued orders the new plan sends to another machine,
# and repairs at which no operation changes machine. Event numbers are 1-based, as in the hero.
for i, e in enumerate(ev):
    moves = [(j, dest, e["plan"][(j, op)][0]) for j, stt, op, dest, _ in e["states"]
             if stt == "queued" and (j, op) in e["plan"] and e["plan"][(j, op)][0] != dest]
    if moves:
        print(f"  event {i + 1} (minute {e['t']}, {e['kind']} of {e['who']}): queued order(s) sent elsewhere "
              f"(order, from machine, to machine) {moves}")

# ----------------------------------------------------------------------------------------------
section("Part 2 (snapshot): the snapshot at one event, read by hand")
SHOW_T = 117
e_show = next(e for e in ev if e["t"] == SHOW_T)
snap = e_show["snap"]
print(f"event at minute {SHOW_T}: {e_show['kind']} of {'job' if e_show['kind'] == 'arrival' else 'machine'} {e_show['who']}")
print("job states (idx, state, op, dest, eta):", e_show["states"])
print("machines (down, current job, queue length):", e_show["machines"])
print("machine_free:", snap.machine_free, " machine_family:", snap.machine_family)
for pj in snap.jobs:
    print(f"  PlanJob idx={pj.idx} family={pj.family} due={pj.due} first_op={pj.first_op} ready={pj.ready}")
print("plan (job, op) -> (machine, start, end):", sorted(e_show["plan"].items()))
print("status", e_show["status"], " objective", e_show["objective"], " warm start", e_show["warm"].objective)
print("stations: load station 0, machine m at station m + 1; travel times used in the hand reading:")
for a, b, why in ((1, 2, "order 4, at m0 (station 1), to m1 (station 2)"), (5, 2, "order 2, at m4 (station 5), to m1"),
                  (4, 1, "order 3, at m3 (station 4), to m0"), (4, 3, "order 3, at m3, to m2 (station 3)"),
                  (0, 1, "order 5, at the load station 0, to m0")):
    print(f"  tau({a}, {b}) = {WS.travel[a][b]}   {why}")
print("changeover of m3 from family 0 to family 2:", WS.setup_time(3, 0, 2))
print("repairs before minute 117 still running: m2 down since 116 -> expected 116 + 20 = 136; m3 down since 109 -> 129")


# ----------------------------------------------------------------------------------------------
section("Part 4 (greedy): list scheduling by hand on the minute-117 snapshot")
# A copy of the loop of jobshop/cpsat.py:greedy_plan that records the candidates of each step;
# the assertion below checks it builds exactly the repository's plan.


def greedy_trace(ws, snap, allowance, rule, previous_plan=None):
    free, family = list(snap.machine_free), list(snap.machine_family)
    next_op = {j.idx: j.first_op for j in snap.jobs}
    last_done, ops, todo, steps = {}, {}, {j.idx: j for j in snap.jobs}, []
    while todo:
        cands = []
        for job in todo.values():
            op = next_op[job.idx]
            for k, p in ws.alternatives(job.family, op).items():
                if op == job.first_op:
                    if k not in job.ready:
                        continue
                    arrive = job.ready[k]
                else:
                    k_prev, end_prev = last_done[job.idx]
                    arrive = end_prev + cpsat.transport_lag(ws, k_prev, k, allowance)
                su = ws.setup_time(k, family[k], job.family)
                start = max(free[k], arrive) + su
                end = start + p
                if rule == "finish":
                    key = (end, job.due, job.idx, k)
                elif rule == "start":
                    key = (start, job.due, end, job.idx, k)
                elif rule == "edd":
                    key = (job.due, job.idx, end, k)
                else:
                    old = previous_plan.get((job.idx, op)) if previous_plan else None
                    key = (0, old[1], job.idx, k) if old is not None and old[0] == k else (1, end, job.idx, k)
                cands.append({"key": key, "job": job.idx, "op": op, "m": k, "free": free[k], "arrive": arrive,
                              "setup": su, "S": start, "p": p, "E": end, "due": job.due})
        b = min(cands, key=lambda c: c["key"])
        steps.append((cands, b))
        ops[(b["job"], b["op"])] = (b["m"], b["S"], b["E"])
        free[b["m"]], family[b["m"]] = b["E"], todo[b["job"]].family
        last_done[b["job"]] = (b["m"], b["E"])
        next_op[b["job"]] += 1
        if next_op[b["job"]] == ws.n_ops(todo[b["job"]].family):
            del todo[b["job"]]
    return ops, steps


prev_of_117 = ev[ev.index(e_show) - 1]["plan"]
for rule in ("finish", "start", "edd", "previous"):
    ops, steps = greedy_trace(WS, snap, 2, rule, prev_of_117)
    ref = cpsat.greedy_plan(WS, snap, 2, rule, prev_of_117)
    assert ops == ref.ops, rule
    first3 = [(b["job"], b["op"], f"m{b['m']}", b["S"], b["E"]) for _, b in steps[:3]]
    print(f"key '{rule}': objective {ref.objective}; first three placements (order, op, machine, S, E): {first3}")
    if rule == "finish":
        print(f"  {len(steps)} placement steps (N = operations to plan); step 1 candidates "
              "(order, op, machine, phi_m, alpha, changeover, S, p, E, due):")
        for c in sorted(steps[0][0], key=lambda c: (c["job"], c["m"])):
            print(f"    order {c['job']} op {c['op']} on m{c['m']}: phi {c['free']}, alpha {c['arrive']}, s {c['setup']}, "
                  f"S {c['S']}, p {c['p']}, E {c['E']}, due {c['due']}")
        print("  step 2 picks:", (steps[1][1]["job"], steps[1][1]["op"], steps[1][1]["m"], steps[1][1]["S"], steps[1][1]["E"]))
print("warm start at 117 = the cheapest of the four:", e_show["warm"].objective)
print("first plan of the day (minute 51): warm_start builds", 3, "greedy plans, there is no previous plan yet")

# ----------------------------------------------------------------------------------------------
section("Part 7 (nervous): how much consecutive plans differ, morning of test day 0")


def nervousness(events):
    """Per event: operations in both the previous and the new plan; share moved to another machine;
    mean |change of planned start|; machines whose next planned operation changed."""
    out = []
    for prev, cur in zip(events, events[1:]):
        P, Q, t = prev["plan"], cur["plan"], cur["t"]
        common = [o for o in Q if o in P]
        if not common:
            out.append(None)
            continue
        moved = sum(P[o][0] != Q[o][0] for o in common)
        shift = sum(abs(P[o][1] - Q[o][1]) for o in common) / len(common)
        nxt = lambda plan, m: min(((s, o) for o, (k, s, _) in plan.items() if k == m and s >= t), default=(None, None))[1]
        machines = [m for m in range(WS.n_machines) if nxt(Q, m) is not None]
        changed = sum(nxt(P, m) != nxt(Q, m) for m in machines)
        out.append({"t": t, "common": len(common), "moved": moved, "shift": shift, "next_changed": changed,
                    "next_total": len(machines)})
    return out


def summary(nv):
    rows = [r for r in nv if r]
    common = sum(r["common"] for r in rows)
    return {"pairs": len(rows), "common": common, "moved": sum(r["moved"] for r in rows),
            "moved_share": sum(r["moved"] for r in rows) / common,
            "shift": sum(r["shift"] * r["common"] for r in rows) / common,
            "next_changed": sum(r["next_changed"] for r in rows), "next_total": sum(r["next_total"] for r in rows),
            "calm": sum(r["moved"] == 0 and r["shift"] == 0 for r in rows)}


nv_cp, nv_gr = nervousness(ev), nervousness(ev_g)
for name, nv in (("CP-SAT", nv_cp), ("Greedy replan", nv_gr)):
    s = summary(nv)
    print(f"{name}: {s['pairs']} consecutive pairs of plans, {s['common']} operations in both plans of a pair; "
          f"{s['moved']} moved to another machine ({100 * s['moved_share']:.1f} %), mean |start change| {s['shift']:.1f} min, "
          f"next operation changed on {s['next_changed']} of {s['next_total']} (machine, re-plan) pairs "
          f"({100 * s['next_changed'] / s['next_total']:.0f} %); pairs of plans that left every shared operation "
          f"unchanged (same machine, same start): {s['calm']}")
by_kind = {}
for e, r in zip(ev[1:], nv_cp[0:]):
    if r:
        by_kind.setdefault(e["kind"], []).append(r)
for k, rs in by_kind.items():
    c = sum(r["common"] for r in rs)
    print(f"  CP-SAT after {'an' if k == 'arrival' else 'a'} {k}: {len(rs)} events, mean |start change| "
          f"{sum(r['shift'] * r['common'] for r in rs) / c:.1f} min, moved {sum(r['moved'] for r in rs)} of {c}")
r117 = nv_cp[ev.index(e_show) - 1]
print(f"  the hero's minute-117 pair (event {ev.index(e_show) + 1}): {r117['moved']} of {r117['common']} shared operations "
      f"moved machine, starts moved {r117['shift']:.1f} min on average")
for i, (e, r) in enumerate(zip(ev[1:], nv_cp)):
    if r and e["kind"] == "repair" and r["moved"] == 0 and e["t"] == 128:
        print(f"  event {i + 2} (minute {e['t']}, repair of m{e['who']}): moved 0 of {r['common']}, starts moved {r['shift']:.2f} min")

# ----------------------------------------------------------------------------------------------
section("Part 6 (blind): drift, the AGV lag and certainty equivalence")


def first_proc_starts(log):
    s = {}
    for g in log.segments:
        if g.kind == "proc":
            s[(g.job, g.op)] = min(s.get((g.job, g.op), 10 ** 9), g.start)
    return s


def drift(events, log):
    """Actual processing start minus the start planned by the last plan made before it."""
    starts = first_proc_starts(log)
    out = []
    for o, s_act in starts.items():
        if s_act >= T_END:
            continue
        plans = [e for e in events if e["t"] <= s_act and o in e["plan"]]
        if plans:
            out.append(s_act - plans[-1]["plan"][o][1])
    return out


dr = drift(ev, cp_log)
print(f"CP-SAT, operations started before minute {T_END}: {len(dr)}; actual minus planned start: "
      f"on time (0) {sum(x == 0 for x in dr)}, early {sum(x < 0 for x in dr)}, late {sum(x > 0 for x in dr)}; "
      f"mean {st.mean(dr):+.2f} min, largest delay {max(dr)} min, largest advance {min(dr)} min")


class TimedShop(Shop):
    """The repository's shop, unchanged, with a note of when each job asks for an AGV."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.asked = {}

    def _send(self, jr, machine):
        if jr.station != self.ws.station(machine):
            self.asked.setdefault(jr.idx, []).append(self.now)
        super()._send(jr, machine)


def agv_waits(shop, log):
    """For each loaded AGV leg: wait = start of the loaded leg - time the job asked for transport."""
    waits = []
    for j, asks in shop.asked.items():
        loaded = sorted(g.start for g in log.legs if g.job == j)
        assert len(loaded) == len(asks)
        waits += [s - a for s, a in zip(loaded, asks)]
    return waits


w = []
for seed in range(20):
    sc = make_scenario(WS, seed, mtbf=LEVELS["frequent breakdowns"])
    shop = TimedShop(WS, sc, cpsat.CPSATController(**SETTINGS, use_solver=False))
    w += agv_waits(shop, shop.run())
w = np.array(w)
print(f"AGV wait before a loaded move, greedy replan, 20 test days, frequent breakdowns: {len(w)} moves, "
      f"median {np.median(w):.0f} min, mean {w.mean():.2f} min, 90th percentile {np.percentile(w, 90):.0f} min, "
      f"zero wait {100 * (w == 0).mean():.0f} %, more than 2 min {100 * (w > 2).mean():.0f} %")
print("travel times between stations:", sorted({x for row in WS.travel for x in row if x}), "min; allowance delta = 2 min")


two = [max(0, d - 20) for d in (10, 30)]
print(f"certainty equivalence, two-point version: D = 10 or 30 with probability 1/2 -> tardiness {two}, mean {st.mean(two)}, "
      f"planned (D = 20) {max(0, 20 - 20)}")
print("certainty equivalence: a repair lasts uniform [10, 30] min, the planner assumes 20. A job planned to finish")
print("exactly at its due date behind the repair has planned tardiness 0 and expected tardiness")
xs = np.linspace(10, 30, 200001)
print(f"  E[max(0, D - 20)] = {np.mean(np.maximum(0, xs - 20)):.3f} min (exact: 10^2 / 40 = {100 / 40}); with slack beta: (10 - beta)^2 / 40,"
      f" e.g. beta = 5: {25 / 40:.3f} min")
# Pitfall 2: the breakdown of m3 at minute 109 (event 20 of the hero)
e109 = next(e for e in ev if e["t"] == 109)
print("pitfall 2: at minute 109 order 3 is", [s for s in e109["states"] if s[0] == 3], "; the plan puts (3, 0) on",
      e109["plan"][(3, 0)], "; m3 expected back at", e109["snap"].machine_free[3],
      "; real repair", [d for d in FULL.downtime[3] if d[0] == 109])

section("Exercise 5 (re-plan only at arrivals): greedy re-planning that ignores breakdowns and repairs")


class ArrivalsOnly(cpsat.CPSATController):
    def on_event(self, shop, kind, **info):
        if kind == "arrival":
            super().on_event(shop, kind, **info)


tard = []
for seed in range(20):
    sc = make_scenario(WS, seed, mtbf=LEVELS["frequent breakdowns"])
    log = Shop(WS, sc, ArrivalsOnly(**SETTINGS, use_solver=False)).run()
    assert check_schedule(WS, sc, log) == []
    tard.append(run_metrics(WS, sc, log)["mean_tardiness"])
print(f"greedy re-planning at arrivals only, frequent breakdowns, 20 test days: mean tardiness {np.mean(tard):.2f} min "
      f"(at every event: 24.03 min, Part 5 below)")


# ----------------------------------------------------------------------------------------------
section("Part 5 (cap): jobs in the shop, the 25-job cap, and the size of the model (also Part 3: strict vs non-strict)")
maxact, results, per_cap = [], {}, {}


class Probe(cpsat.CPSATController):
    def on_event(self, shop, kind, **info):
        maxact.append(len(shop.active_jobs()))
        super().on_event(shop, kind, **info)


for level, mtbf in LEVELS.items():
    for label, kw in (("cap 25", {}), ("cap 10", {"max_jobs": 10}), ("cap 5", {"max_jobs": 5}), ("no cap", {"max_jobs": 10 ** 6}),
                      ("non-strict", {"strict": False})):
        tard, setup, peak, capped, replans = [], [], 0, 0, 0
        for seed in range(20):
            sc = make_scenario(WS, seed, mtbf=mtbf)
            maxact.clear()
            c = Probe(**{**SETTINGS, **kw}, use_solver=False)
            log = Shop(WS, sc, c).run()
            met = run_metrics(WS, sc, log)
            tard.append(met["mean_tardiness"])
            setup.append(met["setup_time"])
            peak, capped, replans = max(peak, max(maxact)), capped + c.n_capped, replans + c.n_replans
        results[(level, label)] = (np.mean(tard), peak, capped, replans)
        per_cap[(level, label)] = (np.array(tard), np.array(setup))
        print(f"greedy replan, {level}, {label}: mean tardiness {np.mean(tard):.2f} min, changeover time per day "
              f"{np.mean(setup):.0f} min, most jobs in the shop at an event (all active orders, including those on their "
              f"last operation) {peak}, capped replans {capped} of {replans}")
    for label in ("cap 10", "cap 5"):
        dd = per_cap[(level, label)][0] - per_cap[(level, "cap 25")][0]
        lo, hi = bootstrap_ci(dd)
        print(f"  paired, {level}: {label} - cap 25 = {dd.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] min, "
              f"lower on {(dd < 0).sum()} of 20 days, higher on {(dd > 0).sum()}; changeover time per day "
              f"{per_cap[(level, label)][1].mean() - per_cap[(level, 'cap 25')][1].mean():+.0f} min")


def model_size(n_per_family):
    """Circuit arcs and optional intervals of a snapshot with n_per_family fresh jobs of each family."""
    cand = [0] * WS.n_machines
    for f in range(WS.n_families):
        for op in range(WS.n_ops(f)):
            for m in WS.alternatives(f, op):
                cand[m] += n_per_family
    arcs = sum(1 + 3 * c + c * (c - 1) for c in cand if c)
    return sum(cand), arcs, cand


per_job = [sum(len(WS.alternatives(f, op)) for op in range(WS.n_ops(f))) for f in range(WS.n_families)]
print("optional intervals per fresh job, by family:", per_job, " mean", st.mean(per_job))
print("arcs of the circuit on a machine with c candidate operations: 1 + 3c + c(c - 1) = (c + 1)^2")
a_m = model_size(1)[2]
print(f"candidates per machine for one fresh job of each family q_m = {a_m}; for n jobs, c_m = n q_m / 4 and the arcs are "
      f"exactly sum (c_m + 1)^2 = {sum(a * a for a in a_m) / 16} n^2 + {2 * sum(a_m) / 4:g} n + {WS.n_machines}")
for n in (1, 2, 5, 6, 10):
    ivs, arcs, cand = model_size(n)
    print(f"  {4 * n:g} fresh jobs (n/4 per family): {ivs:g} optional intervals, candidates per machine {cand}, "
          f"circuit arcs {arcs:g}")

# ----------------------------------------------------------------------------------------------
section("Part 3 (loop) and Part 8 (results): replans per day, solver statistics and paired comparisons, from results/runs.csv")
rows = list(csv.DictReader(open(REPO / "results" / "runs.csv", newline="", encoding="utf-8")))
for r in rows:
    for k in ("mean_tardiness", "replans", "replans_optimal", "replans_not_improved", "replans_capped", "decision_time_s", "setup_time", "makespan"):
        r[k] = float(r[k])
    r["seed"] = int(r["seed"])


def col(level, method, key="mean_tardiness"):
    return np.array([r[key] for r in sorted((r for r in rows if r["level"] == level and r["method"] == method), key=lambda r: r["seed"])])


for level in LEVELS:
    rp = col(level, "CP-SAT", "replans")
    print(f"{level}: CP-SAT replans per day {rp.mean():.1f} (min {rp.min():.0f}, max {rp.max():.0f}); "
          f"proven optimal {100 * col(level, 'CP-SAT', 'replans_optimal').sum() / rp.sum():.0f} %; "
          f"not improved on the warm start {col(level, 'CP-SAT', 'replans_not_improved').sum():.0f} of {rp.sum():.0f}; "
          f"capped {col(level, 'CP-SAT', 'replans_capped').sum():.0f}; "
          f"seconds per replan {col(level, 'CP-SAT', 'decision_time_s').sum() / rp.sum():.2f}; "
          f"greedy replan seconds per replan {1000 * col(level, 'Greedy replan', 'decision_time_s').sum() / col(level, 'Greedy replan', 'replans').sum():.2f} ms")
ratio = (col("frequent breakdowns", "CP-SAT", "decision_time_s").sum() / col("frequent breakdowns", "CP-SAT", "replans").sum()) / (
    col("frequent breakdowns", "Greedy replan", "decision_time_s").sum() / col("frequent breakdowns", "Greedy replan", "replans").sum())
print(f"a CP-SAT replan takes about {ratio:.0f} times as long as a greedy one, about {round(ratio, -3):.0f} "
      f"(frequent breakdowns, wall clock, indicative)")
mk = col("frequent breakdowns", "CP-SAT", "makespan").mean()
cycle = 120 + 20
print(f"expected replans, frequent breakdowns: 60 arrivals + 2 x 5 x makespan / (MTBF + MTTR) = 60 + 10 x {mk:.0f} / {cycle} "
      f"= {60 + 10 * mk / cycle:.1f}  (C_max = {mk:.0f}: the mean makespan of the CP-SAT runs with breakdowns in runs.csv)")
rp = col("frequent breakdowns", "CP-SAT", "replans")
mks = col("frequent breakdowns", "CP-SAT", "makespan")
gap = rp - (60 + 10 * mks / cycle)
print(f"  observed {rp.mean():.1f}, day-to-day standard deviation {rp.std(ddof=1):.1f}, standard error of the mean "
      f"{rp.std(ddof=1) / np.sqrt(len(rp)):.1f}; day by day, observed minus 60 + 10 x makespan_i / 140: mean {gap.mean():+.1f}, "
      f"standard error {gap.std(ddof=1) / np.sqrt(len(gap)):.1f}")
for level in LEVELS:
    print(f"  {level}: mean time between events = mean makespan / mean replans = "
          f"{col(level, 'CP-SAT', 'makespan').mean():.0f} / {col(level, 'CP-SAT', 'replans').mean():.1f} = "
          f"{col(level, 'CP-SAT', 'makespan').mean() / col(level, 'CP-SAT', 'replans').mean():.1f} min")

methods = ["FIFO", "SPT", "EDD", "SST", "Greedy replan", "CP-SAT"]
per_seed = {}
for level in LEVELS:
    means = {m: col(level, m).mean() for m in methods}
    best_rule = min(["FIFO", "SPT", "EDD", "SST"], key=lambda m: means[m])
    ref, g, c = col(level, best_rule), col(level, "Greedy replan"), col(level, "CP-SAT")
    print(f"{level}: best rule {best_rule} {means[best_rule]:.1f}, greedy replan {means['Greedy replan']:.1f}, CP-SAT {means['CP-SAT']:.1f}")
    for name, dd in (("Greedy replan - best rule", g - ref), ("CP-SAT - greedy replan", c - g), ("CP-SAT - best rule", c - ref)):
        lo, hi = bootstrap_ci(dd)
        print(f"  {name}: {dd.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] min, better on {(dd < 0).sum()} of 20, worse on {(dd > 0).sum()}")
    total = (c - ref).mean()
    print(f"  decomposition: {(g - ref).mean():+.2f} + {(c - g).mean():+.2f} = {total:+.2f}; re-planning share "
          f"{100 * (g - ref).mean() / total:.0f} %, solver share {100 * (c - g).mean() / total:.0f} %; "
          f"CP-SAT reduction against the best rule {100 * -total / means[best_rule]:.0f} %, greedy {100 * -(g - ref).mean() / means[best_rule]:.0f} %")
    print(f"  setup time per day: best rule {col(level, best_rule, 'setup_time').mean():.0f}, greedy {col(level, 'Greedy replan', 'setup_time').mean():.0f}, "
          f"CP-SAT {col(level, 'CP-SAT', 'setup_time').mean():.0f} min")
    per_seed[level] = {"rule": best_rule, "r": ref.tolist(), "g": g.tolist(), "c": c.tolist()}
readme = (REPO / "README.md").read_text(encoding="utf-8")
import re  # noqa: E402
flat = " ".join(readme.split())
print("README, strict plan following for CP-SAT, chosen on validation seeds:",
      re.search(r"machines follow the planned order strictly[^;]*", flat).group(0))

# ----------------------------------------------------------------------------------------------
section("Part 4 (greedy) and Exercise 2: re-timing a plan, the 'previous' warm start")
viol = 0
checked = 0
for e in planned:
    if e["snap"] and len(e["snap"].jobs) and e["plan"]:
        again = cpsat.greedy_plan(WS, e["snap"], 2, "previous", e["plan"])
        checked += 1
        viol += again.objective > e["objective"]
print(f"re-timing each CP-SAT plan of the morning on its own snapshot (the case sigma' = sigma): {checked} plans, "
      f"objective increased in {viol}")


def cost_on(plan_ops, snap):
    """J of a plan, counted over the jobs of a snapshot (10 x tardiness + completion)."""
    return sum(10 * max(0, plan_ops[(pj.idx, WS.n_ops(pj.family) - 1)][2] - pj.due) + plan_ops[(pj.idx, WS.n_ops(pj.family) - 1)][2]
               for pj in snap.jobs)


# later snapshots: re-time plan i-1 on snapshot i, when snapshot i has no order plan i-1 did not plan
later, rep_rows = {}, []
for prev, cur in zip(ev, ev[1:]):
    s = cur["snap"]
    if not (s and s.jobs and prev["plan"]):
        continue
    if not all((pj.idx, op) in prev["plan"] for pj in s.jobs for op in range(pj.first_op, WS.n_ops(pj.family))):
        continue  # a new order: outside the proposition
    old = cost_on(prev["plan"], s)
    again = cpsat.greedy_plan(WS, s, 2, "previous", prev["plan"]).objective
    r = later.setdefault(cur["kind"], [0, 0, 0])
    r[0] += 1
    r[1] += again > old
    r[2] += cur["objective"] > old
    if cur["kind"] == "repair":
        dur = next(b - a for a, b in FULL.downtime[cur["who"]] if b == cur["t"])
        rep_rows.append((1, dur, again - old))
        print(f"  repair at minute {cur['t']} of m{cur['who']} after {dur} min ({'later' if dur > 20 else 'not later'} than MTTR = 20): "
              f"previous plan on the same orders {old}, re-timed {again} ({again - old:+d}), new CP-SAT plan {cur['objective']}")
print(f"  repairs no later than MTTR (the nominal case): {sum(rr[0] for rr in rep_rows if rr[1] <= 20)}, re-timing increased the cost "
      f"in {sum(rr[2] > 0 for rr in rep_rows if rr[1] <= 20)}; repairs later than MTTR: {sum(rr[1] > 20 for rr in rep_rows)}, "
      f"increased in {sum(rr[2] > 0 for rr in rep_rows if rr[1] > 20)} (by {[rr[2] for rr in rep_rows if rr[1] > 20 and rr[2] > 0]})")
for k, (n_, up_retime, up_new) in later.items():
    print(f"  consecutive events, at {'an' if k == 'arrival' else 'a'} {k} with no new order: {n_} pairs; re-timed previous plan "
          f"costs more than the previous plan on the same orders in {up_retime}; the new CP-SAT plan costs more in {up_new}")

# Exercise 2, first counterexample: availability later than planned (a breakdown)
for F in (0, 5):
    c1, c2 = F + 10, F + 20
    print(f"  two-order counterexample, machine free at {F}: completions {c1}, {c2}, objective "
          f"{10 * max(0, c1 - 10) + c1 + 10 * max(0, c2 - 20) + c2}")


# Exercise 2, second counterexample: earlier availability but the machine set up for another family.
def one_machine(seq, free, fam, setup=5):
    t, out, J = free, [], 0
    for name, f, p, due in seq:
        s = t + (setup if fam is not None and fam != f else 0)
        e = s + p
        out.append((name, t, s, e, max(0, e - due)))
        J += 10 * max(0, e - due) + e
        t, fam = e, f
    return out, J


XY = [("X", 0, 10, 20), ("Y", 1, 10, 35)]
for free, fam in ((10, 0), (8, 1)):
    rows_, J = one_machine(XY, free, fam)
    print(f"  changeover counterexample: free at {free}, set up for family {fam}: (name, free, start, end, late) {rows_}, J = {J}")

# ----------------------------------------------------------------------------------------------
section("Exercise 4 (count the re-plans): the hero's morning against the renewal prediction")
pred = N_MORNING + 2 * WS.n_machines * T_END / cycle
print(f"failures per machine expected in the window (elementary renewal theorem): {T_END} / {cycle} = {T_END / cycle:.2f}; "
      f"the hero's lower chart spans minutes 0 to {T_END + 6}")
n_dt = kinds["breakdown"] + kinds["repair"]
print(f"prediction for the morning: {N_MORNING} + 2 x 5 x {T_END} / {cycle} = {pred:.1f} events, of which "
      f"{2 * WS.n_machines * T_END / cycle:.1f} downtime events ({WS.n_machines * T_END / cycle:.1f} breakdowns); observed "
      f"{len(ev)}: {kinds['breakdown']} breakdowns + {kinds['repair']} repairs = {n_dt}")
counts = []
for seed in range(2000):
    sc = make_scenario(WS, seed, n_jobs=1, mtbf=LEVELS["frequent breakdowns"])
    counts.append(sum((a < T_END) + (b < T_END) for m in range(WS.n_machines) for a, b in sc.downtime[m]))
counts = np.array(counts)
print(f"downtime events (breakdowns + repairs) before minute {T_END}, 2000 simulated mornings (seeds 0-1999): mean "
      f"{counts.mean():.1f}, standard deviation {counts.std(ddof=1):.1f}, {n_dt} or more on {100 * (counts >= n_dt).mean():.1f} % "
      f"of mornings; test day 0 has {counts[0]}")

print(f"\n(total running time {time.perf_counter() - T_START:.0f} s)")

# ==============================================================================================
if EMIT_JS:
    def plan_rows(plan, snap):
        """[job, op, machine, start, end, setup] with the changeover before each planned start."""
        fam = {pj.idx: pj.family for pj in snap.jobs}
        out = []
        for m in range(WS.n_machines):
            prev = snap.machine_family[m]
            for (j, o), (k, s, e) in sorted(((key, v) for key, v in plan.items() if v[0] == m), key=lambda x: x[1][1]):
                out.append([j, o, k, s, e, WS.setup_time(k, prev, fam[j])])
                prev = fam[j]
        return out

    kind_code = {"arrival": 0, "breakdown": 1, "repair": 2}
    state_code = {"new": 0, "wait_transport": 1, "in_transit": 2, "queued": 3, "processing": 4}
    E = []
    for i, e in enumerate(ev):
        s = e["snap"]
        rec = {"t": e["t"], "k": kind_code[e["kind"]], "w": e["who"], "st": e["status"][0] if e["status"] != "EMPTY" else "-",
               "obj": e["objective"], "warm": e["warm"].objective if e["warm"] else None,
               "free": s.machine_free if s else [], "fam": s.machine_family if s else [],
               "down": [int(m[0]) for m in e["machines"]], "cur": [m[1] for m in e["machines"]],
               "jobs": [[x[0], state_code[x[1]], x[2], x[3]] for x in e["states"]],
               "ready": {pj.idx: [pj.first_op, pj.ready] for pj in s.jobs} if s else {},
               "plan": plan_rows(e["plan"], s) if s and s.jobs else [],
               "wplan": plan_rows(e["warm"].ops, s) if e["warm"] else [],
               }
        E.append(rec)
    segs = [[g.machine, g.job, g.op, int(g.kind == "proc"), g.start, g.end] for g in cp_log.segments if g.start < T_END]
    legs = [[g.agv, g.job if g.job is not None else -1, g.start, g.end] for g in cp_log.legs if g.start < T_END]
    down = [[m, a, b] for m in range(5) for a, b in FULL.downtime[m] if a < T_END]
    jobs = [[j.idx, j.family, j.arrival, j.due] for j in MORNING.jobs]
    nerv = [[r["t"], r["common"], r["moved"], round(r["shift"], 2), r["next_changed"], r["next_total"]] if r else None for r in nv_cp]
    nerv_g = [[r["t"], r["common"], r["moved"], round(r["shift"], 2), r["next_changed"], r["next_total"]] if r else None for r in nv_gr]
    data = {"tEnd": T_END, "events": E, "segs": segs, "legs": legs, "down": down, "jobs": jobs,
            "nerv": nerv, "nervG": nerv_g, "perSeed": per_seed}
    print("\n/* ---- data for interactive/c03.js ---- */")
    print("const DATA = " + json.dumps(data, separators=(",", ":")) + ";")
