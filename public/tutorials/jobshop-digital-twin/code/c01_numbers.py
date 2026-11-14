"""Reproduce every number quoted on chapter C1 of "Which Job Should the Machine Do Next?",
"A constraint-programming model of the shop" (c01-a-constraint-programming-model.html).

Usage:
    JOBSHOP_REPO=<path to the repository> python c01_numbers.py
(inside the course zip, the repository's jobshop/ sits next to code/ and JOBSHOP_REPO is not needed).
Needs the repository's environment: numpy, simpy and ortools (CP-SAT, plus GLOP and SCIP for the MILP part).
Runs in well under a minute.
"""
from __future__ import annotations

import itertools
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(os.environ.get("JOBSHOP_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

from ortools.linear_solver import pywraplp  # noqa: E402
from ortools.sat.python import cp_model  # noqa: E402

import jobshop.cpsat as cpsat  # noqa: E402
from jobshop.cpsat import CPSATController, PlanJob, Snapshot, plan_cost, take_snapshot, transport_lag  # noqa: E402
from jobshop.scenario import Workshop, make_scenario, make_workshop  # noqa: E402
from jobshop.shop import simulate  # noqa: E402


def section(title):
    print(f"\n=== {title}")


# ----------------------------------------------------------------------------------------------------------------
# The paint machine of notation.md: set up for blue, a colour change takes 10 min, every order waits at t = 0.
NAMES = "ABCD"
COLOUR = {"A": "blue", "B": "red", "C": "blue", "D": "red"}
P = {"A": 20, "B": 10, "C": 15, "D": 25}
DUE = {"A": 30, "B": 25, "C": 70, "D": 60}
SETUP = 10
START_COLOUR = "blue"


def s(f, g):
    """Changeover from colour f to colour g (f = None: nothing run yet)."""
    return 0 if f is None or f == g else SETUP


def run(seq):
    """Packed schedule of a sequence: each order starts as soon as the machine and its changeover allow."""
    t, col, rows = 0, START_COLOUR, []
    for n in seq:
        st = t + s(col, COLOUR[n])
        rows.append((n, st, st + P[n]))
        t, col = st + P[n], COLOUR[n]
    return rows


def sums(rows):
    T = sum(max(0, e - DUE[n]) for n, _, e in rows)
    C = sum(e for _, _, e in rows)
    return T, C


# horizon of the hand-written paint model (Part 'paint'): latest availability 0 + sum of (p + max setup); nothing moves,
# so no transport term. (solve_snapshot itself always adds max_lag = allowance + longest travel: see Part 'paint'.)
H = 0 + sum(P[n] + SETUP for n in NAMES)

section("Part 'objective': the horizon bound for the hand-written paint model")
print(f"H = 0 + sum over the four orders of (p + max setup 10) = {sum(P.values())} + 4 x 10 = {H}")

section("Part 'objective': linearised tardiness, order A (due 30)")
for C in (25, 30, 40):
    print(f"C_A = {C}: T_A >= {C - 30} and T_A >= 0, smallest feasible T_A = {max(0, C - 30)}")
print("tardiness widget at load: C_A = 40, T_A = 25: feasible, 10 T_A =", 10 * 25,
      "; lowering T_A to 10 saves", 10 * (25 - 10), "and gives 10 T_A =", 10 * 10)

section("Brute force over the 24 sequences (packed schedules), as in chapter A2")
table = []
for seq in itertools.permutations(NAMES):
    T, C = sums(run(seq))
    table.append(("".join(seq), T, C))
best_T = min(table, key=lambda r: (r[1], r[0]))
print("minimum total tardiness:", best_T[1], "by", best_T[0], " (ties:", [r[0] for r in table if r[1] == best_T[1]], ")")
for name in ("ABCD", "ABDC", "BADC", "ACBD", "BCAD"):
    T, C = next((r[1], r[2]) for r in table if r[0] == name)
    print(f"  {name}: total tardiness {T}, total completion {C}, 10 T + C = {10 * T + C}")

section("Part 'objective': what the weight w in  w * sum T + sum C  does (brute force)")
for w in (0, 1, 2, 5, 10, 100):
    seq, T, C = min(table, key=lambda r: (w * r[1] + r[2], r[0]))
    ties = [(r[0], r[1], r[2]) for r in table if w * r[1] + r[2] == w * T + C and r[0] != seq]
    print(f"  w = {w:3d}: best sequence {seq}, total tardiness {T}, total completion {C}, objective {w * T + C}"
          f"  ties: {ties}")
# the smallest weight from which the tardiness-optimal sequence wins
for w in range(0, 101):
    seq, T, C = min(table, key=lambda r: (w * r[1] + r[2], r[0]))
    if T == best_T[1]:
        print("  smallest integer weight at which the minimum-tardiness plan is chosen:", w); break

section("Hero: the presets")
for label, seq in (("FIFO ABCD", "ABCD"), ("optimum ABDC", "ABDC")):
    rows = run(seq)
    T, C = sums(rows)
    print(f"  {label}: starts {[(n, st) for n, st, _ in rows]}, sum T = {T}, sum C = {C}, 10 T + C = {10 * T + C}")
# the opening state: arcs of ABDC, starts packed as if there were no changeovers
naive = {"A": 0, "B": 20, "D": 30, "C": 55}
viol = []
order = ["A", "B", "D", "C"]
prev_end, prev_col = 0, START_COLOUR
for n in order:
    need = prev_end + s(prev_col, COLOUR[n])
    if naive[n] < need:
        viol.append(f"S_{n} = {naive[n]} < {need}")
    prev_end, prev_col = naive[n] + P[n], COLOUR[n]
print("  opening state (ABDC, changeovers forgotten): starts", naive, "violations:", viol)

# ----------------------------------------------------------------------------------------------------------------
section("Part 'paint': the paint machine written as a CP-SAT model by hand, objective sum T_j")


def paint_model(weight_T=1, weight_C=0):
    m = cp_model.CpModel()
    S = {n: m.new_int_var(0, H, f"S_{n}") for n in NAMES}
    E = {n: m.new_int_var(0, H, f"E_{n}") for n in NAMES}
    iv = {n: m.new_interval_var(S[n], P[n], E[n], f"I_{n}") for n in NAMES}
    T = {n: m.new_int_var(0, H, f"T_{n}") for n in NAMES}
    m.add_no_overlap(iv.values())
    node = {n: i + 1 for i, n in enumerate(NAMES)}
    arcs, lit = [], {}
    for n in NAMES:
        first = m.new_bool_var(f"0->{n}")
        arcs.append((0, node[n], first)); lit[("0", n)] = first
        m.add(S[n] >= 0 + s(START_COLOUR, COLOUR[n])).only_enforce_if(first)
        last = m.new_bool_var(f"{n}->0")
        arcs.append((node[n], 0, last)); lit[(n, "0")] = last
        for k in NAMES:
            if k != n:
                follows = m.new_bool_var(f"{n}->{k}")
                arcs.append((node[n], node[k], follows)); lit[(n, k)] = follows
                m.add(S[k] >= E[n] + s(COLOUR[n], COLOUR[k])).only_enforce_if(follows)
    m.add_circuit(arcs)
    for n in NAMES:
        m.add(T[n] >= E[n] - DUE[n])
    m.minimize(weight_T * sum(T.values()) + weight_C * sum(E.values()))
    return m, S, E, T, lit, arcs


def proto_of(model):
    return model.Proto() if hasattr(model, "Proto") else model.proto


KINDS = ("interval", "no_overlap", "circuit", "exactly_one", "linear", "bool_or", "bool_and", "at_most_one", "lin_max")


def kind(c):
    """Which constraint a ConstraintProto holds (works with the protobuf and the pybind proto of OR-Tools 9.15)."""
    if hasattr(c, "WhichOneof"):
        return c.WhichOneof("constraint")
    for k in KINDS:
        has = getattr(c, "has_" + k)
        if (has() if callable(has) else has):
            return k
    return "other"


def describe(model):
    pr = proto_of(model)
    kinds = Counter(kind(c) for c in pr.constraints)
    n_bool = sum(1 for v in pr.variables if list(v.domain) == [0, 1])
    enforced = sum(1 for c in pr.constraints if kind(c) == "linear" and len(c.enforcement_literal))
    plain = kinds.get("linear", 0) - enforced
    arcs = sum(len(c.circuit.tails) for c in pr.constraints if kind(c) == "circuit")
    return dict(variables=len(pr.variables), booleans=n_bool, integers=len(pr.variables) - n_bool,
                intervals=kinds.get("interval", 0), no_overlap=kinds.get("no_overlap", 0),
                circuits=kinds.get("circuit", 0), circuit_arcs=arcs, exactly_one=kinds.get("exactly_one", 0),
                enforced_linear=enforced, plain_linear=plain, constraints=len(pr.constraints))


m, S, E, T, lit, arcs = paint_model()
solver = cp_model.CpSolver()
solver.parameters.num_workers = 8
solver.parameters.random_seed = 0
status = solver.solve(m)
seq = sorted(NAMES, key=lambda n: solver.value(S[n]))
print("status:", solver.status_name(status), " objective sum T =", int(solver.objective_value),
      " best bound =", int(solver.best_objective_bound))
print("sequence:", "".join(seq), " starts:", {n: solver.value(S[n]) for n in seq},
      " ends:", {n: solver.value(E[n]) for n in seq}, " T:", {n: solver.value(T[n]) for n in seq})
print("arcs set to 1:", [k for k, v in lit.items() if solver.boolean_value(v)])
print("model size:", describe(m))
print("brute force optimum of chapter A2:", best_T[1], " -> agree:", int(solver.objective_value) == best_T[1])

m2, S2, E2, T2, _, _ = paint_model(10, 1)
status2 = solver.solve(m2)
seq2 = sorted(NAMES, key=lambda n: solver.value(S2[n]))
print("with the repository's objective 10 sum T + sum C:", solver.status_name(status2), int(solver.objective_value),
      "sequence", "".join(seq2), "sum T =", sum(solver.value(T2[n]) for n in NAMES),
      "sum C =", sum(solver.value(E2[n]) for n in NAMES))

section("Part 'paint': the repository's solve_snapshot on the paint machine (one family per order)")
fam = {n: i for i, n in enumerate(NAMES)}
paint_ws = Workshop(
    n_machines=1, n_agvs=1,
    routings=tuple(({0: P[n]},) for n in NAMES),
    setup=(tuple(tuple(s(COLOUR[a], COLOUR[b]) for b in NAMES) for a in NAMES),),
    travel=((0, 0), (0, 0)),
)
paint_snap = Snapshot(0, [0], [fam["A"]], [PlanJob(fam[n], fam[n], DUE[n], 0, {0: 0}) for n in NAMES])
_cap, _orig = [], cp_model.CpSolver.solve
cp_model.CpSolver.solve = lambda self, model, *a, **k: (_cap.append(model), _orig(self, model, *a, **k))[1]
plan = cpsat.solve_snapshot(paint_ws, paint_snap, det_time=5.0)
cp_model.CpSolver.solve = _orig
order_ = sorted(plan.ops.items(), key=lambda kv: kv[1][1])
print("status:", plan.status, " objective 10 sum T + sum C =", plan.objective,
      " sequence:", "".join(NAMES[j] for (j, _), _ in order_),
      " sum T =", sum(max(0, plan.ops[(fam[n], 0)][2] - DUE[n]) for n in NAMES))
_ps = describe(_cap[-1])
print("  its model: circuit arcs", _ps["circuit_arcs"], "(20 succession/first/last arcs + 4 self-loops + 1 idle loop)")
_mlag = 2 + max(max(row) for row in paint_ws.travel)
print(f"  its horizon: max_lag = allowance 2 + longest travel {max(max(row) for row in paint_ws.travel)} = {_mlag};"
      f" H = 0 + {sum(P.values())} + 4 x ({SETUP} + {_mlag}) = {sum(P.values()) + 4 * (SETUP + _mlag)};"
      f" time domain of its first variable: {list(proto_of(_cap[-1]).variables[0].domain)}")

# ----------------------------------------------------------------------------------------------------------------
section("Part 'circuit': two identical paint machines (widget), best assignment by brute force")


def best_on_machine(orders):
    best = None
    for p in itertools.permutations(orders):
        T_, C_ = sums(run(p)) if p else (0, 0)
        if best is None or T_ < best[0] or (T_ == best[0] and "".join(p) < best[1]):
            best = (T_, "".join(p), C_)
    return best


results = []
for assign in itertools.product((0, 1), repeat=4):
    groups = [[n for n, a in zip(NAMES, assign) if a == k] for k in (0, 1)]
    b0, b1 = best_on_machine(groups[0]), best_on_machine(groups[1])
    results.append((b0[0] + b1[0], "".join(str(a) for a in assign), b0[1], b1[1]))
results.sort()
print("best total tardiness with two machines:", results[0][0], " assignments reaching it:",
      [(r[1], r[2], r[3]) for r in results if r[0] == results[0][0]])
print("everything on machine 0 (assignment 0000):", next(r for r in results if r[1] == "0000"))
n_ops = 4
print(f"arcs per machine with all {n_ops} orders as candidates: (n + 1)^2 = {(n_ops + 1) ** 2}")

section("Part 'circuit': is the triangle inequality s(f,h) <= s(f,g) + s(g,h) true for the workshop's changeovers?")
WS = make_workshop()
viol, total = [], 0
viol_p = 0
min_p = min(min(alt.values()) for r in WS.routings for alt in r)
for mach in range(WS.n_machines):
    for f, g, h in itertools.permutations(range(WS.n_families), 3):
        total += 1
        if WS.setup[mach][f][h] > WS.setup[mach][f][g] + WS.setup[mach][g][h]:
            viol.append((mach, f, g, h, WS.setup[mach][f][h], WS.setup[mach][f][g], WS.setup[mach][g][h]))
        if WS.setup[mach][f][h] > WS.setup[mach][f][g] + min_p + WS.setup[mach][g][h]:
            viol_p += 1
print(f"triples (machine, f, g, h) with f, g, h distinct: {total}; violations: {len(viol)}")
print("first violations (m, f, g, h, s(f,h), s(f,g), s(g,h)):", viol[:3])
print(f"shortest processing time in the workshop: {min_p} min; violations of s(f,h) <= s(f,g) + p_min + s(g,h): {viol_p}")
v = viol[0]
print(f"example: on machine {v[0]}, {v[1]} -> {v[3]} directly costs {v[4]} min, {v[1]} -> {v[2]} -> {v[3]} costs {v[5]} + {v[6]} = {v[5] + v[6]} min of changeovers")

section("Part 'nooverlap': three operations that fit pairwise but not together in a 60-minute window")
durs, window = (20, 25, 30), 60
print("pairwise sums:", [a + b for a, b in itertools.combinations(durs, 2)], "<=", window, "; total", sum(durs), ">", window)

section("Part 'precedence': transport lags tau + delta used by the planner (delta = 2)")
lags = sorted({transport_lag(WS, a, b, 2) for a in range(5) for b in range(5) if a != b})
print("distinct lags between two different machines:", lags, "; zero when the job stays on its machine")
print("example: machine 1 -> machine 3:", transport_lag(WS, 1, 3, 2), "= 2 +", WS.travel[2][4])
print("  a family-0 job ending operation 0 on machine 1 at minute 30 is at machine 3 at", 30 + transport_lag(WS, 1, 3, 2),
      "; operation 1 of family 0 may run on", WS.alternatives(0, 1))
print("ell_max = delta + longest travel (solve_snapshot's max_lag) =", 2 + max(max(row) for row in WS.travel))
pairs_max = max(len(WS.alternatives(f, k - 1)) * len(WS.alternatives(f, k)) for f in range(4) for k in range(1, 4))
print("largest number of lag constraints for one pair of consecutive operations |M_j,k-1| x |M_jk|:", pairs_max)
print("eligible machines per operation:", sorted({len(alt) for r in WS.routings for alt in r}),
      "; family 0, operation 0:", WS.alternatives(0, 0))

# ----------------------------------------------------------------------------------------------------------------
section("Part 'milp': the paint machine as a big-M MILP (disjunctive binaries y_ab), SCIP and its LP relaxation (GLOP)")
M = H + SETUP
print(f"big-M: M = H + max setup = {H} + {SETUP} = {M}")


def bigM(relax: bool, M=M):
    lp = pywraplp.Solver.CreateSolver("GLOP" if relax else "SCIP")
    Sv = {n: lp.NumVar(0, H - P[n], f"S_{n}") for n in NAMES}
    Tv = {n: lp.NumVar(0, lp.infinity(), f"T_{n}") for n in NAMES}
    y = {}
    for a, b in itertools.combinations(NAMES, 2):
        y[(a, b)] = lp.NumVar(0, 1, f"y_{a}{b}") if relax else lp.IntVar(0, 1, f"y_{a}{b}")
        # y = 1: a before b (not necessarily directly); y = 0: b before a
        lp.Add(Sv[b] >= Sv[a] + P[a] + s(COLOUR[a], COLOUR[b]) - M * (1 - y[(a, b)]))
        lp.Add(Sv[a] >= Sv[b] + P[b] + s(COLOUR[b], COLOUR[a]) - M * y[(a, b)])
    for n in NAMES:
        lp.Add(Sv[n] >= s(START_COLOUR, COLOUR[n]))  # the changeover from blue, if this order comes first
        lp.Add(Tv[n] >= Sv[n] + P[n] - DUE[n])
    lp.Minimize(sum(Tv.values()))
    st = lp.Solve()
    return lp, st, Sv, Tv, y


milp, st, Sv, Tv, y = bigM(False)
print("MILP domains: S_j in [0, H - p_j] =", {n: (0, H - P[n]) for n in NAMES}, ", T_j >= 0")
print("MILP (SCIP): optimal =", st == pywraplp.Solver.OPTIMAL, " sum T =", round(milp.Objective().Value(), 6),
      " sequence:", "".join(sorted(NAMES, key=lambda n: Sv[n].solution_value())),
      " variables:", milp.NumVariables(), "(of which binary:", len(y), ") constraints:", milp.NumConstraints(),
      " branch-and-bound nodes:", milp.nodes())
lp, st, Sv, Tv, y = bigM(True)
print("LP relaxation (GLOP): sum T bound =", round(lp.Objective().Value(), 6),
      " starts:", {n: round(Sv[n].solution_value(), 2) for n in NAMES},
      " y:", {"".join(k): round(v.solution_value(), 3) for k, v in y.items()})
lp, st, Sv, Tv, y = bigM(True, M=1000)
print("LP relaxation with M = 1000: bound =", round(lp.Objective().Value(), 6))
# the two-order picture of the widget: orders A and B, p_A = 20, p_B = 10, changeover 10 either way
print("widget (A and B only): A before B needs S_B - S_A >= 30; B before A needs S_B - S_A <= -20;")
print(f"  relaxed with y in [0, 1] and M = {M}: 30 - M <= S_B - S_A <= M - 20, i.e. {30 - M} <= S_B - S_A <= {M - 20};"
      " it is nonempty for every y once M >= 50")
print(f"  the band for one y contains the diagonal S_B - S_A = 0 iff 30 - M(1 - y) <= 0 <= -20 + M y,"
      f" i.e. 20/M <= y <= 1 - 30/M: at M = {M}, {20 / M:.4f} <= y <= {1 - 30 / M:.4f}; at y = 0.1 the band is"
      f" [{30 - M * 0.9:g}, {-20 + M * 0.1:g}]")
yw = 0.5
print(f"  widget at load, y_AB = {yw}, M = {M}: {30 - M * (1 - yw):g} <= S_B - S_A <= {-20 + M * yw:g};"
      f" the LP's (S_A, S_B) = (0, 10) has S_B - S_A = 10, inside")

# ----------------------------------------------------------------------------------------------------------------
section("Part 'code': solve_snapshot's model on chapter C2's snapshot 101 (seed 0, 60 jobs, frequent breakdowns, "
        "quick re-planning)")
# Built exactly as code/c02_numbers.py builds it: the greedy re-planning loop (no solver) records a snapshot at every
# replan; snapshot number 101 (0-based) is an arrival at minute 891 with 8 jobs.
t0 = time.perf_counter()
SNAPS = []


class Recorder(CPSATController):
    """Quick re-planning (the greedy ablation, no solver), recording the snapshot of every replan."""

    def on_event(self, shop, kind, **info):
        jobs = sorted(shop.active_jobs(), key=lambda j: (j.due, j.idx))[: self.max_jobs]
        SNAPS.append((kind, take_snapshot(shop, jobs, self.allowance), dict(self.plan)))
        super().on_event(shop, kind, **info)


scenario = make_scenario(WS, 0, n_jobs=60, mtbf=120.0)
simulate(WS, scenario, Recorder(use_solver=False))
I_SNAP = 101
kind_, big, prev_plan = SNAPS[I_SNAP]
assert len(big.jobs) == 8, "the simulated day changed: chapter C2 must change snapshot too"


def n_remaining(snap):
    return sum(WS.n_ops(j.family) - j.first_op for j in snap.jobs)


print(f"{len(SNAPS)} snapshots recorded; snapshot {I_SNAP}: {kind_} at minute {big.now}, {len(big.jobs)} jobs, "
      f"{n_remaining(big)} operations to plan")
print("  machine_free:", big.machine_free, " machine_family:", big.machine_family)

captured = []
orig_solve = cp_model.CpSolver.solve


def capture(self, model, *args, **kwargs):
    captured.append(model)
    return orig_solve(self, model, *args, **kwargs)


cp_model.CpSolver.solve = capture
plan = cpsat.solve_snapshot(WS, big, allowance=2, det_time=0.2, previous_plan=prev_plan)  # as in C2
cp_model.CpSolver.solve = orig_solve
size = describe(captured[-1])
pr = proto_of(captured[-1])
per_machine = [len(c.circuit.tails) for c in pr.constraints if kind(c) == "circuit"]
cands = []
for k in range(WS.n_machines):
    n = 0
    for j in big.jobs:
        for op in range(j.first_op, WS.n_ops(j.family)):
            alts = WS.alternatives(j.family, op)
            if op == j.first_op:
                alts = {kk: p for kk, p in alts.items() if kk in j.ready}
            n += k in alts
    cands.append(n)
print("  model size:", size)
print("  rows of the table:", size["exactly_one"], "+", size["no_overlap"], "+", size["circuits"], "+",
      size["enforced_linear"], "+", size["plain_linear"], "+ interval constraints", size["intervals"], "=",
      size["exactly_one"] + size["no_overlap"] + size["circuits"] + size["enforced_linear"] + size["plain_linear"]
      + size["intervals"], "; all constraints:", size["constraints"])
print("  candidate operations per machine n_k:", cands, " arcs per circuit:", per_machine,
      " (n_k + 1)^2:", [(n + 1) ** 2 for n in cands])
presence = sum(cands)
print(f"  presence literals (= optional intervals): {presence}; circuit Booleans created (idle, first, last, follows):"
      f" {sum(1 + 2 * n + n * (n - 1) for n in cands)}")
hz = max(big.machine_free + [big.now] + [max(j.ready.values()) for j in big.jobs])
print("  horizon: start", hz, "-> H =", pr.variables[0].domain[1], f"(H - now = {pr.variables[0].domain[1] - big.now})")
nops = n_remaining(big)
print(f"  integer variables: 3 x {nops} operations (S, E, alpha) + {len(big.jobs)} tardiness = {3 * nops + len(big.jobs)}"
      f" (model: {size['integers']}); plain linear: {nops} 's >= a' + {len(big.jobs)} tardiness bounds ="
      f" {nops + len(big.jobs)} (model: {size['plain_linear']})")
print(f"  solved with the benchmark's budget 0.2 and C2's hint: status {plan.status}, objective {plan.objective},"
      f" check plan_cost = {plan_cost(big, WS, plan.ops)}")
pairs = sum(n * (n - 1) // 2 for n in cands)
print(f"  a pairwise big-M MILP of the same snapshot would need {presence} assignment binaries and {pairs} order binaries y_ab")
print(f"total time {time.perf_counter() - t0:.1f} s")

# ----------------------------------------------------------------------------------------------------------------
section("Part 'circuit', reason 3: a changeover enforced against a non-adjacent predecessor (D arrives at minute 50)")
R = {"A": 0, "B": 0, "C": 0, "D": 50}
# sequence A B D C: B ends at 40, D (red, like B) needs no changeover after B
print("direct successor B -> D: S_D >= max(E_B, r_D) + s(red, red) = max(40, 50) + 0 =", max(40, 50) + 0)
print("if the changeover were also imposed from A (blue): S_D >= r_D + s(blue, red) = 50 + 10 =", 50 + 10)


def run_release(seq):
    t, col, rows = 0, START_COLOUR, []
    for n in seq:
        st = max(t, R[n]) + s(col, COLOUR[n])  # the changeover starts once the machine is free and the order is there
        rows.append((n, st, st + P[n]))
        t, col = st + P[n], COLOUR[n]
    return rows


def run_pairwise(seq):
    """Earliest starts if every earlier order a imposed S_b >= E_a + s(a, b) and S_b >= r_b + s(a, b)."""
    out, prev_end, col = {}, 0, START_COLOUR
    for i, n in enumerate(seq):
        lb = max(max(prev_end, R[n]) + s(col, COLOUR[n]), R[n] + s(START_COLOUR, COLOUR[n]))
        for a in seq[:i]:
            lb = max(lb, out[a][1] + s(COLOUR[a], COLOUR[n]), R[n] + s(COLOUR[a], COLOUR[n]))
        out[n] = (lb, lb + P[n])
        prev_end, col = lb + P[n], COLOUR[n]
    return out


section("Exercise 2: plan A, C, B, D with D arriving at minute 50, model rule against pairwise rule")
_m = {n: (a, b) for n, a, b in run_release("ACBD")}
_p = run_pairwise("ACBD")
print("model rule, A C B D:", _m, " total tardiness", sum(max(0, b - DUE[n]) for n, (a, b) in _m.items()))
print("  B: max(E_C = 35, r_B = 0) + 10 =", max(35, 0) + 10, "; D: max(E_B = 55, r_D = 50) + 0 =", max(55, 50))
print("pairwise rule, A C B D:", _p, " total tardiness", sum(max(0, b - DUE[n]) for n, (a, b) in _p.items()))
print("  D from A and from C (blue): r_D + 10 =", 50 + 10, "; B from A: E_A + 10 = 30, from C: E_C + 10 = 45")
_pw = sorted((sum(max(0, b - DUE[n]) for n, (a, b) in run_pairwise(q).items()), "".join(q))
             for q in itertools.permutations(NAMES))
print("pairwise rule, brute force: best", _pw[0][0], "by", [q for v, q in _pw if v == _pw[0][0]],
      "; true tardiness of those plans:", {q: sums(run_release(q))[0] for v, q in _pw if v == _pw[0][0]})

section("Exercise 4: the paint machine with order D arriving at minute 50 (brute force and CP-SAT)")
rel = sorted(((sums(run_release(p))[0], "".join(p)) for p in itertools.permutations(NAMES)))
print("brute force: best total tardiness", rel[0][0], "by", [q for v, q in rel if v == rel[0][0]])
print("the old optimum ABDC now gives", next(v for v, q in rel if q == "ABDC"), "; its schedule:", run_release("ABDC"))


def paint_with_release(release):
    m = cp_model.CpModel()
    S = {n: m.new_int_var(0, 200, f"S_{n}") for n in NAMES}
    E = {n: m.new_int_var(0, 200, f"E_{n}") for n in NAMES}
    T = {n: m.new_int_var(0, 200, f"T_{n}") for n in NAMES}
    iv = [m.new_interval_var(S[n], P[n], E[n], f"I_{n}") for n in NAMES]
    m.add_no_overlap(iv)
    for n in NAMES:
        m.add(S[n] >= release[n])  # S >= alpha, and here alpha = r
    node = {n: i + 1 for i, n in enumerate(NAMES)}
    arcs = []
    for a in NAMES:
        first = m.new_bool_var("")
        arcs.append((0, node[a], first))
        m.add(S[a] >= release[a] + s(START_COLOUR, COLOUR[a])).only_enforce_if(first)
        arcs.append((node[a], 0, m.new_bool_var("")))
        for b in NAMES:
            if a != b:
                z = m.new_bool_var("")
                arcs.append((node[a], node[b], z))
                m.add(S[b] >= E[a] + s(COLOUR[a], COLOUR[b])).only_enforce_if(z)
                if s(COLOUR[a], COLOUR[b]):
                    m.add(S[b] >= release[b] + s(COLOUR[a], COLOUR[b])).only_enforce_if(z)
        m.add(T[a] >= E[a] - DUE[a])
    m.add_circuit(arcs)
    m.minimize(sum(T.values()))
    sol = cp_model.CpSolver()
    sol.parameters.num_workers = 8
    st = sol.solve(m)
    return sol.status_name(st), int(sol.objective_value), "".join(sorted(NAMES, key=lambda n: sol.value(S[n])))


print("CP-SAT with r_D = 50:", paint_with_release(R))

section("Exercise 6: solve_snapshot's model on real_snapshot() of tests/test_cpsat.py")
sys.path.insert(0, str(REPO / "tests"))
from test_cpsat import real_snapshot  # noqa: E402

captured.clear()
cp_model.CpSolver.solve = capture
tplan = cpsat.solve_snapshot(WS, real_snapshot(), allowance=2, det_time=0.5)
cp_model.CpSolver.solve = orig_solve
tsnap = real_snapshot()
print("jobs:", len(tsnap.jobs), " operations:", n_remaining(tsnap), " model:", describe(captured[-1]))
tc = []
for k in range(WS.n_machines):
    n = 0
    for j in tsnap.jobs:
        for op in range(j.first_op, WS.n_ops(j.family)):
            alts = WS.alternatives(j.family, op)
            if op == j.first_op:
                alts = {kk: p for kk, p in alts.items() if kk in j.ready}
            n += k in alts
    tc.append(n)
print("  n_k:", tc, " sum (n_k + 1)^2 =", sum((n + 1) ** 2 for n in tc), " status", tplan.status, "objective", tplan.objective)

section("Check your understanding: a machine with 3 candidate operations; the arc z_DC of the paint machine")
print("arc literals with 3 candidates: (3 + 1)^2 =", (3 + 1) ** 2, "; of which self-loops reusing presence literals: 3;"
      " new Booleans: 1 + 2 x 3 + 3 x 2 =", 1 + 2 * 3 + 3 * 2)
print("z_DC (red -> blue, changeover", s(COLOUR["D"], COLOUR["C"]), "): S_C >= E_D +", s(COLOUR["D"], COLOUR["C"]),
      "and S_C >= alpha_C +", s(COLOUR["D"], COLOUR["C"]), "= 0 +", s(COLOUR["D"], COLOUR["C"]))
print("order A completing at 25 (due 30): T_A >= -5 and T_A >= 0; feasible T_A in [0, H] = [0,", H, "]; the optimum picks 0")

# ----------------------------------------------------------------------------------------------------------------
section("Exercise 5: chapter A3's flexible 3 x 3 instance in CP-SAT (makespan), classical and flexible")
# A3 Part 'flexible': J0: M0 2, M1 7, M2 7; J1: M1 7, M2 4, M0 5; J2: M2 2, M0 6, M1 2;
# made flexible: J0.1 and J1.0 also on M2 in 6 min, J2.2 also on M0 in 3 min.
JOBS3 = [[{0: 2}, {1: 7}, {2: 7}], [{1: 7}, {2: 4}, {0: 5}], [{2: 2}, {0: 6}, {1: 2}]]
FLEX3 = {(0, 1): {2: 6}, (1, 0): {2: 6}, (2, 2): {0: 3}}
FAM3 = [0, 0, 1]  # "Going further": J0 and J1 in family 0, J2 in family 1


def solve_flex(flexible, fixed=None, setup=0, arcs_out=None):
    """The page's solution code; `fixed` pins the machine of the three flexible operations (for the enumeration);
    `setup` > 0 adds the "Going further" family changeovers through one add_circuit per machine."""
    m = cp_model.CpModel()
    elig = {(j, k): ({**op, **FLEX3.get((j, k), {})} if flexible else dict(op))
            for j, job in enumerate(JOBS3) for k, op in enumerate(job)}
    if fixed:
        for key, mach in fixed.items():
            elig[key] = {mach: elig[key][mach]}
    H = 60  # the page's horizon, larger than the total of the slowest times (printed below)
    S, E, x, on = {}, {}, {}, {0: [], 1: [], 2: []}
    for (j, k), e in elig.items():
        S[j, k], E[j, k] = m.new_int_var(0, H, ""), m.new_int_var(0, H, "")
        x[j, k] = {i: m.new_bool_var("") for i in e}
        for i, p in e.items():
            on[i].append(m.new_optional_interval_var(S[j, k], p, E[j, k], x[j, k][i], ""))
        m.add_exactly_one(x[j, k].values())
        if k:
            m.add(S[j, k] >= E[j, k - 1])
    for i, ivs in on.items():
        m.add_no_overlap(ivs)
        if not setup:
            continue
        cand = [o for o in elig if i in elig[o]]  # nodes 1, 2, ... of machine i
        arcs = [(0, 0, m.new_bool_var(""))]  # the idle loop
        for u, a in enumerate(cand, 1):
            arcs += [(0, u, m.new_bool_var("")), (u, 0, m.new_bool_var("")), (u, u, x[a][i].Not())]
            for v, b in enumerate(cand, 1):
                if v != u:
                    z, c = m.new_bool_var(""), setup * (FAM3[a[0]] != FAM3[b[0]])
                    arcs.append((u, v, z))
                    m.add(S[b] >= E[a] + c).only_enforce_if(z)
                    if c and b[1]:  # alpha_b = end of the job's previous step
                        m.add(S[b] >= E[b[0], b[1] - 1] + c).only_enforce_if(z)
        m.add_circuit(arcs)
        if arcs_out is not None:
            arcs_out[f"M{i}"] = len(arcs)
    cmax = m.new_int_var(0, H, "")
    m.add_max_equality(cmax, [E[j, 2] for j in range(3)])
    m.minimize(cmax)
    sol = cp_model.CpSolver()
    st = sol.solve(m)
    mu = {f"J{j}.{k}": f"M{i}" for (j, k) in FLEX3 for i, v in x[j, k].items() if sol.boolean_value(v)}
    return sol.status_name(st), int(sol.value(cmax)), mu


_work = sum(max({**op, **FLEX3.get((j, k), {})}.values()) for j, job in enumerate(JOBS3) for k, op in enumerate(job))
print("horizon used on the page: 60; total of the slowest eligible times:", _work, "-> safe:", 60 > _work)
print("classical assignment:", solve_flex(False))
print("with flexibility:", solve_flex(True), "(which optimal routing is returned may depend on the search)")
enum = {}
for choice in itertools.product(*[sorted({**JOBS3[j][k], **FLEX3[j, k]}) for (j, k) in FLEX3]):
    enum["".join(f"M{c}" for c in choice)] = solve_flex(True, dict(zip(FLEX3, choice)))[1]
print("best makespan of each of the 8 assignments of J0.1, J1.0, J2.2:", enum)
print("assignments reaching", min(enum.values()), ":", sum(v == min(enum.values()) for v in enum.values()), "of", len(enum))


# "Going further" of Exercise 5: family changeovers on every machine, through one circuit per machine.
section("Exercise 5, going further: family changeovers (J0, J1 family 0; J2 family 1) through add_circuit per machine")


def brute_flex(flexible, setup):
    """Independent check: every assignment of the flexible operations and every order on every machine; for each,
    the earliest starts S_b = max(E of the machine predecessor, alpha_b) + changeover (none for a machine's first
    operation, alpha_b the end of the job's previous step, 0 for a first step); orders that form a cycle are skipped.
    The makespan is regular, so the best of these semi-active schedules is the optimum."""
    keys = list(FLEX3) if flexible else []
    best = None
    for choice in itertools.product(*[sorted({**JOBS3[j][k], **FLEX3[j, k]}) for (j, k) in keys]):
        mu = {(j, k): next(iter(op)) for j, job in enumerate(JOBS3) for k, op in enumerate(job)}
        mu.update(dict(zip(keys, choice)))
        p = {o: {**JOBS3[o[0]][o[1]], **(FLEX3.get(o, {}) if flexible else {})}[mu[o]] for o in mu}
        per = [[o for o in mu if mu[o] == i] for i in range(3)]
        for orders in itertools.product(*[itertools.permutations(ops) for ops in per]):
            pred = {b: a for seq in orders for a, b in zip(seq, seq[1:])}
            S = {o: 0 for o in mu}
            for _ in range(len(mu) + 2):  # longest-path relaxation; converges iff the orders are acyclic
                changed = False
                for o in mu:
                    alpha = S[o[0], o[1] - 1] + p[o[0], o[1] - 1] if o[1] else 0
                    if o in pred:
                        a = pred[o]
                        val = max(S[a] + p[a], alpha) + setup * (FAM3[a[0]] != FAM3[o[0]])
                    else:
                        val = alpha
                    if val != S[o]:
                        S[o], changed = val, True
                if not changed:
                    break
            else:
                continue
            cm = max(S[j, 2] + p[j, 2] for j in range(3))
            if best is None or cm < best[0]:
                best = (cm, {f"J{j}.{k}": f"M{c}" for (j, k), c in zip(keys, choice)},
                        {f"M{i}": [f"J{j}.{k}" for j, k in seq] for i, seq in enumerate(orders)})
    return best


print("changeover between families on the page: 2 minutes (and 1 minute for the closing remark)")
for su in (0, 1, 2):
    arcs = {}
    cl, fl = solve_flex(False, setup=su), solve_flex(True, setup=su, arcs_out=arcs)
    bc, bf = brute_flex(False, su), brute_flex(True, su)
    print(f"setup {su}: CP-SAT classical {cl[:2]}, flexible {fl[:2]};  brute force classical {bc[0]}, flexible {bf[0]}"
          f"  -> agree: {cl[1] == bc[0] and fl[1] == bf[0]}")
    if su:
        print(f"  circuit arcs per machine (flexible): {arcs} = (n_m + 1)^2 with n_m =",
              {k: round(v ** 0.5) - 1 for k, v in arcs.items()})
    if su == 2:
        print("  a best flexible schedule (brute force): routing", bf[1], " machine orders", bf[2])
print("flexibility's gain without changeovers:", solve_flex(False)[1] - solve_flex(True)[1],
      "; with 2-minute changeovers:", solve_flex(False, setup=2)[1] - solve_flex(True, setup=2)[1])
