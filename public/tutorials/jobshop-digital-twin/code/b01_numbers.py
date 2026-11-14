"""Reproduce every number quoted on chapter B1 of "Which Job Should the Machine Do Next?",
"Discrete-event simulation" (b01-discrete-event-simulation.html).

Usage:
    JOBSHOP_REPO=path/to/jobshop-digital-twin python b01_numbers.py
(from the course zip, where jobshop/ sits next to code/, no variable is needed).
Needs numpy and simpy (the repository's requirements). Runs in a few seconds: the benchmark
numbers come from results/runs.csv; the only simulations are rule runs, a few milliseconds each.
"""
from __future__ import annotations

import csv
import heapq
import os
import sys
import time
from math import factorial
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JOBSHOP_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

from jobshop.metrics import run_metrics  # noqa: E402
from jobshop.rules import RuleController  # noqa: E402
from jobshop.scenario import LEVELS, Job, Scenario, Workshop, make_scenario, make_workshop, mean_interarrival  # noqa: E402
from jobshop.shop import Shop, simulate  # noqa: E402

WS = make_workshop()
SEEDS = range(20)          # the 20 test days
MTBF, MTTR = 120.0, 20.0   # the "frequent breakdowns" level


def section(title):
    print(f"\n=== {title}")


# ------------------------------------------------------------------------------------------------
section("Part 1 (why): how fast a simulated day runs, from results/runs.csv")
with open(REPO / "results" / "runs.csv", newline="") as fh:
    RUNS = list(csv.DictReader(fh))
rule_rows = [r for r in RUNS if r["method"] in ("FIFO", "SPT", "EDD", "SST") and r["level"] == "no breakdowns"]
wall = np.median([float(r["run_wall_s"]) for r in rule_rows])
mk = np.mean([float(r["makespan"]) for r in rule_rows])
print(f"rule runs without breakdowns ({len(rule_rows)} of them): median wall time of one simulated day {1e3 * wall:.1f} ms")
print(f"mean simulated day (makespan) of the same runs: {mk:.0f} min = {mk / 60:.1f} h")
print(f"speed-up over real time: {mk * 60 / wall:.1e}")
print("test runs in results/runs.csv:", len(RUNS))


# ------------------------------------------------------------------------------------------------
# The hero's tiny simulator, written exactly like interactive/b01.js: one or two identical machines,
# one shared queue, blue/red jobs, a 10-minute colour change, breakdowns that suspend the work.
class Mini:
    CHANGE = 10

    def __init__(self, jobs, n_machines=1, rule="FIFO", downtime=None):
        self.jobs = jobs                      # [(name, colour, p, arrival)]
        self.rule = rule
        self.t = 0
        self.seq = 0
        self.events = []                      # heap of (time, seq, kind, data)
        self.cancelled = set()
        self.queue = []
        self.m = [dict(job=None, last="blue", down=False, rem_s=0, rem_p=0, seg=None, ev=None, kind="")
                  for _ in range(n_machines)]
        self.done = {}
        self.trace = []
        self.n_events = 0
        self.downtime = downtime or {}        # machine -> [(down, up)]
        self.dt_idx = {k: 0 for k in self.downtime}
        self.push(jobs[0][3], "arrival", 0)
        for k, spans in self.downtime.items():
            if spans:
                self.push(spans[0][0], "failure", k)

    def push(self, t, kind, data):
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, data))
        return self.seq

    def pending(self):
        return sorted((t, s, k, d) for t, s, k, d in self.events if s not in self.cancelled)

    def start_segment(self, k):
        m = self.m[k]
        if m["rem_s"] > 0:
            m["kind"], m["seg"] = "setup", self.t
            m["ev"] = self.push(self.t + m["rem_s"], "setup done", k)
        else:
            m["kind"], m["seg"] = "proc", self.t
            m["ev"] = self.push(self.t + m["rem_p"], "done", k)

    def try_start(self, k):
        m = self.m[k]
        if m["job"] is not None or m["down"] or not self.queue:
            return None
        key = {"FIFO": lambda j: (j[3], j[0]), "SPT": lambda j: (j[2], j[3], j[0]),
               "SST": lambda j: (0 if j[1] == m["last"] else self.CHANGE, j[3], j[0])}[self.rule]
        j = min(self.queue, key=key)
        self.queue.remove(j)
        m["job"] = j
        m["rem_s"] = 0 if j[1] == m["last"] else self.CHANGE
        m["rem_p"] = j[2]
        m["last"] = j[1]
        self.start_segment(k)
        return j

    def step(self):
        while self.events and self.events[0][1] in self.cancelled:
            heapq.heappop(self.events)
        t, _, kind, data = heapq.heappop(self.events)
        self.t = t
        self.n_events += 1
        note = ""
        if kind == "arrival":
            j = self.jobs[data]
            self.queue.append(j)
            if data + 1 < len(self.jobs):
                self.push(self.jobs[data + 1][3], "arrival", data + 1)
            started = [self.try_start(k) for k in range(len(self.m))]
            note = f"{j[0]} ({j[1]}, {j[2]} min) arrives" + "".join(
                f"; machine {k} starts {s[0]}" for k, s in enumerate(started) if s)
        elif kind == "setup done":
            m = self.m[data]
            m["rem_s"] = 0
            self.start_segment(data)
            note = f"changeover done on machine {data}, {m['job'][0]} is processed"
        elif kind == "done":
            m = self.m[data]
            j = m["job"]
            self.done[j[0]] = t
            m["job"] = None
            nxt = self.try_start(data)
            note = f"{j[0]} leaves (flow time {t - j[3]})" + (f"; {self.rule} picks {nxt[0]}" if nxt else "")
        elif kind == "failure":
            m = self.m[data]
            m["down"] = True
            if m["job"] is not None:
                el = t - m["seg"]
                m["rem_s" if m["kind"] == "setup" else "rem_p"] -= el
                self.cancelled.add(m["ev"])
                note = f"machine {data} fails during the {m['kind']} of {m['job'][0]}"
            else:
                note = f"machine {data} fails while idle"
            self.push(self.downtime[data][self.dt_idx[data]][1], "repair", data)
        elif kind == "repair":
            m = self.m[data]
            m["down"] = False
            self.dt_idx[data] += 1
            if self.dt_idx[data] < len(self.downtime[data]):
                self.push(self.downtime[data][self.dt_idx[data]][0], "failure", data)
            if m["job"] is not None:
                self.start_segment(data)
                note = f"machine {data} repaired, resumes {m['job'][0]}"
            else:
                self.try_start(data)
                note = f"machine {data} repaired"
        self.trace.append((t, kind, note, [(tt, kk) for tt, _, kk, _ in self.pending()]))
        return len(self.done) < len(self.jobs)

    def run(self):
        while self.step():
            pass
        return self


# The hero's random day (interactive/b01.js: N_RANDOM, P_CHOICES, MEAN_P, MTBF, MTTR), quoted in the hero hint
HERO_N, HERO_P = 30, (10, 15, 20, 25)
section("Hero (hint): the random day of the demo")
print(f"{HERO_N} orders, blue or red with probability 1/2, processing time one of {HERO_P} min "
      f"({min(HERO_P)} to {max(HERO_P)} min), mean {np.mean(HERO_P)} min; breakdowns MTBF {MTBF:.0f}, MTTR {MTTR:.0f} min")


# The worked example: the paint orders of chapters A1-A2 arriving over time, one breakdown.
PAINT = [("A", "blue", 20, 0), ("B", "red", 10, 5), ("C", "blue", 15, 12), ("D", "red", 25, 40)]
section("Part 2 (events): the hand-traced event list, FIFO, machine down from 25 to 33")
ex = Mini(PAINT, 1, "FIFO", {0: [(25, 33)]}).run()
for t, kind, note, pend in ex.trace:
    print(f"t = {t:3d}  {kind:10s}  {note:55s}  next: " + ", ".join(f"{k} @{tt}" for tt, k in pend))
ev_minutes = sorted({t for t, *_ in ex.trace})
print("events processed:", ex.n_events, "at", len(ev_minutes), "distinct minutes; clock at the end:", ex.t, "min")
print("a 1-minute time-stepped loop visits the instants 0 to", ex.t, "=", ex.t + 1, "instants,",
      ex.t + 1 - len(ev_minutes), "of them with no event")
flows = {j[0]: ex.done[j[0]] - j[3] for j in PAINT}
print("completions", ex.done, " flow times", flows, " sum", sum(flows.values()), " mean F_bar =", np.mean(list(flows.values())))
# N(t) by hand: integral = sum of flow times; Little's law on [0, 108]
grid = np.arange(ex.t)
N = np.array([sum(1 for j in PAINT if j[3] <= s < ex.done[j[0]]) for s in grid])
print("integral of N(t) dt =", int(N.sum()), " L = integral / T =", round(N.sum() / ex.t, 3),
      " lambda_hat = 4 /", ex.t, " lambda_hat * F_bar =", round(4 / ex.t * np.mean(list(flows.values())), 3))
pieces = [int(s) for s in grid if s == 0 or N[s] != N[s - 1]] + [ex.t]
print("N(t) pieces:", [(a, b, int(N[a])) for a, b in zip(pieces, pieces[1:])],
      " areas:", " + ".join(str((b - a) * int(N[a])) for a, b in zip(pieces, pieces[1:])))
print("the repair lasts", 33 - 25, "min; without it every later order would leave that much earlier")
ex2 = Mini(PAINT, 1, "FIFO").run()
print("same orders without the breakdown: completions", ex2.done)
ex3 = Mini(PAINT, 1, "SST", {0: [(25, 33)]}).run()
print("exercise: SST with the breakdown: completions", ex3.done, " events", ex3.n_events,
      " sum of flow times", sum(ex3.done[j[0]] - j[3] for j in PAINT))
f3 = {j[0]: ex3.done[j[0]] - j[3] for j in PAINT}
print("  SST flow times", f3, f" L = {sum(f3.values())}/{ex3.t} = {sum(f3.values()) / ex3.t:.3f}")
print("  SST trace:")
for t, kind, note, pend in ex3.trace:
    print(f"  t = {t:3d}  {kind:10s}  {note:55s}  next: " + ", ".join(f"{k} @{tt}" for tt, k in pend))

section("Part 2: the repository's simulator on the same example (1 machine, travel 0)")
# families 0..3 = A, B, C, D; a colour change costs 10 min; the carts take no time
colour = ["blue", "red", "blue", "red"]
tiny = Workshop(n_machines=1, n_agvs=1, routings=(({0: 20},), ({0: 10},), ({0: 15},), ({0: 25},)),
                setup=(tuple(tuple(0 if colour[f] == colour[g] else 10 for g in range(4)) for f in range(4)),),
                travel=((0, 0), (0, 0)))
sc = Scenario(0, tuple(Job(i, i, a, 999) for i, (_, _, _, a) in enumerate(PAINT)), (((25, 33),),), mttr=8.0)
log = simulate(tiny, sc, RuleController("FIFO"))
for s in log.segments:
    print(f"  {PAINT[s.job][0]} {s.kind:5s} {s.start:3d}-{s.end:3d}")
rep = {PAINT[j][0]: max(s.end for s in log.segments if s.job == j) for j in range(4)}
print("repository completions", rep, " equal to the hand trace:", rep == ex.done)

section("Part 2: how many SimPy events one simulated day takes (seed 0, EDD)")
for level, mtbf in LEVELS.items():
    sc0 = make_scenario(WS, 0, mtbf=mtbf)
    shop = Shop(WS, sc0, RuleController("EDD"))
    shop.env.process(shop._arrivals())
    for m in shop.machines:
        if sc0.downtime[m.idx]:
            shop.env.process(shop._breakdowns(m))
    n, times = 0, []
    while not shop._all_done.processed:
        times.append(shop.env.peek())     # the time of the event about to be executed
        shop.env.step()
        n += 1
    distinct = len(np.unique(times))
    print(f"{level}: {n} SimPy events (bookkeeping included) for {shop.now} simulated minutes, "
          f"on {distinct} distinct minutes ({100 * distinct / shop.now:.0f} %); "
          f"a one-minute loop would be empty in {100 * (1 - distinct / shop.now):.0f} % of its steps")


# ------------------------------------------------------------------------------------------------
section("Part 3 (poisson): the arrival stream of the 20 test days")
m_ia = mean_interarrival(WS, 0.60)
lam = 1 / m_ia
print(f"1/lambda = {m_ia:.2f} min, lambda = {lam:.4f} per min = {60 * lam:.2f} per hour")
print(f"variance of a gap 1/lambda^2 = {m_ia ** 2:.0f} min^2, standard deviation {m_ia:.2f} min")
raw = []        # the exponential gaps exactly as make_scenario draws them
ints = []       # the integer gaps after flooring the arrival times
last = []
for s in SEEDS:
    rng = np.random.default_rng([s, 1])
    g = rng.exponential(m_ia, size=60)
    raw.append(g)
    sc = make_scenario(WS, s)
    a = np.array([j.arrival for j in sc.jobs])
    assert np.array_equal(a, np.floor(np.cumsum(g)).astype(int))
    ints.append(np.diff(np.r_[0, a]))
    last.append(a[-1])
raw = np.concatenate(raw); ints = np.concatenate(ints)
print(f"{raw.size} gaps: mean {raw.mean():.2f} min, standard deviation {raw.std(ddof=1):.2f}, "
      f"coefficient of variation {raw.std(ddof=1) / raw.mean():.2f}; standard error of the mean {raw.std(ddof=1) / np.sqrt(raw.size):.2f}")
print(f"distance of the mean from 1/lambda: {(m_ia - raw.mean()) / (raw.std(ddof=1) / np.sqrt(raw.size)):.2f} standard errors "
      f"(with the theoretical standard error {m_ia / np.sqrt(raw.size):.2f}: {(m_ia - raw.mean()) / (m_ia / np.sqrt(raw.size)):.2f})")
print(f"integer gaps after flooring: mean {ints.mean():.2f}, shortest {ints.min()}, longest {ints.max()}, "
      f"share of zero gaps {np.mean(ints == 0):.3f}")
print(f"last arrival: mean {np.mean(last):.0f} min (theory 60/lambda = {60 * m_ia:.0f})")
per_seed = [g.mean() for g in np.split(raw, 20)]
print(f"per-day mean gap from {min(per_seed):.1f} to {max(per_seed):.1f} min")
s_, t_ = 20.0, 20.0
cond = np.mean(raw[raw > s_] > s_ + t_) if np.any(raw > s_) else float("nan")
print(f"memoryless check, s = t = 20: P(X > 40 | X > 20) = {cond:.3f} from {np.sum(raw > s_)} gaps, "
      f"P(X > 20) = {np.mean(raw > t_):.3f}, theory exp(-20/20.83) = {np.exp(-t_ / m_ia):.3f}")
p0, n0 = np.exp(-t_ / m_ia), int(np.sum(raw > s_))
se_c = np.sqrt(p0 * (1 - p0) / n0)
print(f"  standard error of the conditional share from {n0} gaps: sqrt({p0:.2f} x {1 - p0:.2f} / {n0}) = {se_c:.3f}; "
      f"{cond:.3f} is {(p0 - cond) / se_c:.1f} standard errors below {p0:.3f}")
se_f = np.sqrt(p0 * (1 - p0) / raw.size)
print(f"  standard error of the fresh share from {raw.size} gaps: {se_f:.3f}; "
      f"{np.mean(raw > t_):.3f} is {(p0 - np.mean(raw > t_)) / se_f:.1f} standard errors below {p0:.3f}")
print(f"mean remaining gap after 20 min already waited: {np.mean(raw[raw > 20] - 20):.2f} min (theory {m_ia:.2f})")
# a large sample, for the exact-looking version of the same check
big = np.random.default_rng(0).exponential(m_ia, size=1_000_000)
print(f"1,000,000 draws: P(X > 40 | X > 20) = {np.mean(big[big > 20] > 40):.4f}, P(X > 20) = {np.mean(big > 20):.4f}")
# counts per hour: Poisson with mean 60 lambda
mu_h = 60 * lam
counts = []
for s in SEEDS:
    a = np.array([j.arrival for j in make_scenario(WS, s).jobs])
    hours = int(a[-1] // 60)
    counts += [int(np.sum((a >= 60 * h) & (a < 60 * (h + 1)))) for h in range(hours)]
counts = np.array(counts)
print(f"arrivals per hour, N_arr(t + 60) - N_arr(t): theory Poisson mean = variance = {mu_h:.2f}; measured over {counts.size} full hours: "
      f"mean {counts.mean():.2f}, variance {counts.var(ddof=1):.2f}")
print(f"P(no arrival in an hour) theory {np.exp(-mu_h):.3f} (one hour in {1 / np.exp(-mu_h):.1f}), measured {np.mean(counts == 0):.3f} = {100 * np.mean(counts == 0):.1f} % (one hour in {1 / np.mean(counts == 0):.1f})")
p5 = 1 - sum(np.exp(-mu_h) * mu_h ** k / factorial(k) for k in range(5))
print(f"P(5 or more arrivals in an hour) theory {p5:.3f}, measured {np.mean(counts >= 5):.3f}")

section("Part 3: superposition and splitting")
fam_gaps = {f: [] for f in range(4)}
for s in SEEDS:
    sc = make_scenario(WS, s)
    for f in range(4):
        a = np.array([j.arrival for j in sc.jobs if j.family == f])
        fam_gaps[f] += list(np.diff(a))
print(f"splitting: each family is a Poisson stream of rate lambda/4; the wait X^(f) for a family-f order has mean {4 * m_ia:.2f} min")
for f in range(4):
    g = np.array(fam_gaps[f])
    print(f"  family {f}: {g.size} gaps, mean {g.mean():.1f}, standard deviation {g.std(ddof=1):.1f}")
allg = np.concatenate([np.array(fam_gaps[f]) for f in range(4)])
print(f"  all families: {allg.size} gaps, mean {allg.mean():.1f}, standard error about {allg.std(ddof=1) / np.sqrt(allg.size):.1f}")
# The same measurement on the generator itself: 60 Poisson arrivals, uniform families, gaps counted
# inside the day only. Long gaps that would run past the last order are never observed (an edge effect).
mc = np.random.default_rng(3)
tot, cntg = 0.0, 0
for _ in range(20_000):
    arr = np.cumsum(mc.exponential(m_ia, size=60))
    fam = mc.integers(0, 4, size=60)
    for f in range(4):
        d = np.diff(arr[fam == f])
        tot += d.sum(); cntg += d.size
print(f"  Monte Carlo of the within-day measurement (20,000 days): expected measured mean family gap {tot / cntg:.1f} min "
      f"(not {4 * m_ia:.2f})")
print(f"first failure among 5 working machines, MTBF {MTBF:.0f}: exponential with mean {MTBF / 5:.0f} min")
r = np.random.default_rng(1)
mins = r.exponential(MTBF, size=(200_000, 5)).min(axis=1)
print(f"  simulated, {mins.size:,} draws: mean {mins.mean():.2f}, standard deviation {mins.std():.2f}")


# ------------------------------------------------------------------------------------------------
section("Part 4 (load): rho by hand and what it ignores")
W = [WS.min_remaining_work(f, 0) for f in range(WS.n_families)]
EW = float(np.mean(W))
print("fastest-route work per family:", W, " mean", EW)
print(f"1/lambda = E[W_f(j)]/(rho m) = {EW}/(0.60 x 5) = {EW / 3:.2f} min;  rho = lambda E[W_f(j)]/m = {EW / m_ia / 5:.2f}")
mean_alt = float(np.mean([sum(np.mean(list(alt.values())) for alt in WS.routings[f]) for f in range(4)]))
print(f"if every operation took the average of its eligible machines: work per job = {mean_alt:.2f}, rho = {mean_alt / m_ia / 5:.2f}")
off = [WS.setup[m][f][g] for m in range(5) for f in range(4) for g in range(4) if f != g]
print(f"mean changeover between different families: {np.mean(off):.2f} min (range {min(off)}-{max(off)})")
print(f"random order: P(family change) = 3/4, so about {0.75 * np.mean(off):.2f} min per operation, "
      f"{4 * 0.75 * np.mean(off):.1f} min per job -> rho = {(EW + 4 * 0.75 * np.mean(off)) / m_ia / 5:.2f}")
print(f"with availability A = {MTBF / (MTBF + MTTR):.3f} the fastest-route load per available machine is {0.60 / (MTBF / (MTBF + MTTR)):.2f}")
moves = sorted({WS.travel[a][b] for a in range(len(WS.travel)) for b in range(len(WS.travel)) if a != b})
print(f"transport: a cart move between two different stations takes {' or '.join(map(str, moves))} min (make_workshop().travel)")
# the hero's random day: processing 17.5 min on average, a 10-min colour change half the time under FIFO, A = 0.857
fac = (17.5 + 0.5 * 10) / 17.5 / (MTBF / (MTBF + MTTR))
print(f"hero random day: real load = slider x (17.5 + 0.5 x 10)/17.5 / A = {fac:.2f} x slider; "
      f"overloaded above slider = {1 / fac:.2f}")
print(f"  (Part 6) busy fraction u = slider x (17.5 + 0.5 x 10)/17.5 = {(17.5 + 5) / 17.5:.2f} x slider; "
      f"u < A = {MTBF / (MTBF + MTTR):.3f} means slider < {MTBF / (MTBF + MTTR) / ((17.5 + 5) / 17.5):.2f}")
print("measured in results/runs.csv (mean over 20 days): processing and changeover per job, busy share")
for level in LEVELS:
    for meth in ("FIFO", "SST", "CP-SAT"):
        rr = [x for x in RUNS if x["level"] == level and x["method"] == meth]
        proc = np.mean([float(x["utilisation"]) * 5 * float(x["makespan"]) / 60 for x in rr])
        setup = np.mean([float(x["setup_time"]) / 60 for x in rr])
        print(f"  {level:20s} {meth:7s} processing {proc:.1f} min/job, changeover {setup:.1f} min/job, "
              f"load with both = {(proc + setup) / m_ia / 5:.2f}")


# ------------------------------------------------------------------------------------------------
section("Part 5 (breakdowns): availability and breakdown counts on the 20 test days")
A = MTBF / (MTBF + MTTR)
print(f"A = MTBF/(MTBF+MTTR) = {MTBF:.0f}/{MTBF + MTTR:.0f} = {A:.3f} ({100 * A:.1f} %); downtime share 1 - A = {1 - A:.4f}")
print(f"repair uniform [0.5, 1.5] x MTTR = [{0.5 * MTTR:.0f}, {1.5 * MTTR:.0f}] min: mean {MTTR:.0f}, "
      f"standard deviation {MTTR / np.sqrt(12):.2f}")
T = 1400
print(f"expected breakdowns per machine in a day of T = {T} min: T/(MTBF+MTTR) = {T / (MTBF + MTTR):.1f}; "
      f"for the 5 machines {5 * T / (MTBF + MTTR):.0f}")
ups, downs, frac, cnt = [], [], [], []
for s in SEEDS:
    sc = make_scenario(WS, s, mtbf=MTBF)
    for m in range(5):
        spans = sc.downtime[m]
        prev = 0
        for d, u in spans:
            ups.append(d - prev); downs.append(u - d); prev = u
        frac.append(sum(max(0, min(u, T) - min(d, T)) for d, u in spans) / T)
        cnt.append(sum(1 for d, _ in spans if d < T))
ups, downs, frac, cnt = map(np.array, (ups, downs, frac, cnt))
print(f"all {ups.size} generated cycles (to the generator's far horizon): mean up {ups.mean():.1f} min, "
      f"mean repair {downs.mean():.2f} min, availability {ups.sum() / (ups.sum() + downs.sum()):.3f}")
print(f"first {T} min of each of the 100 machine-days: downtime share mean {frac.mean():.4f} "
      f"(from {frac.min():.3f} to {frac.max():.3f}), breakdowns mean {cnt.mean():.2f} (from {cnt.min()} to {cnt.max()})")
print(f"breakdowns per machine-day: variance {cnt.var(ddof=1):.2f}")
cyc_mu = MTBF + MTTR
cyc_var = MTBF ** 2 + MTTR ** 2 / 12          # Exp(MTBF) up time + uniform [10, 30] repair
print(f"renewal theory: cycle mean {cyc_mu:.0f}, variance {cyc_var:.0f}; Var B(T) ~ T sigma^2 / mu^3 = {T * cyc_var / cyc_mu ** 3:.2f}")
print(f"  = (T/mu) x CV^2 = {T / cyc_mu:.0f} x {cyc_var / cyc_mu ** 2:.2f}; with exponential repairs CV^2 would be "
      f"({MTBF ** 2 + MTTR ** 2:.0f}/{cyc_mu ** 2:.0f}) = {(MTBF ** 2 + MTTR ** 2) / cyc_mu ** 2:.2f}; an exponential gap has CV^2 = 1")
print(f"in percent: theory downtime {100 * (1 - A):.1f} %, measured {100 * frac.mean():.1f} %; "
      f"single machine-days from {100 * frac.min():.1f} % to {100 * frac.max():.1f} %")
mk_b = np.mean([float(x["makespan"]) for x in RUNS if x["level"] == "frequent breakdowns" and x["method"] in ("FIFO", "SPT", "EDD", "SST")])
print(f"mean day length under breakdowns (rules' makespan): {mk_b:.0f} min")


# ------------------------------------------------------------------------------------------------
section("Part 6 (little): L = lambda F_bar (usually written L = lambda W)")
print(f"opening illustration: {3} orders an hour x {2} hours each = {3 * 2} orders inside on average")
print("F_bar = mean flow time from results/runs.csv, nominal L = lambda F_bar = F_bar / 20.83 (Going further block)")
for level in LEVELS:
    for meth in ("FIFO", "SPT", "EDD", "SST", "DQN", "Greedy replan", "CP-SAT"):
        w = np.mean([float(x["mean_flow_time"]) for x in RUNS if x["level"] == level and x["method"] == meth])
        print(f"  {level:20s} {meth:14s} F_bar = {w:6.1f} min   L = lambda F_bar = {w / m_ia:5.2f} jobs")
wf = np.mean([float(x["mean_flow_time"]) for x in RUNS if x["level"] == "no breakdowns" and x["method"] == "FIFO"])
wc = np.mean([float(x["mean_flow_time"]) for x in RUNS if x["level"] == "no breakdowns" and x["method"] == "CP-SAT"])
print(f"FIFO minus CP-SAT, no breakdowns: {wf / m_ia - wc / m_ia:.2f} fewer in lambda F_bar with CP-SAT")
section("Part 6: Little's law applied to the machines, busy fraction u = lambda E[omega] / m, omega_j the work actually done")
proc_ex = sum(j[2] for j in PAINT)
chg_ex = sum(1 for a, b in zip(["blue"] + [j[1] for j in PAINT], [j[1] for j in PAINT]) if a != b) * Mini.CHANGE
print(f"worked example: processing {proc_ex} + changeover {chg_ex} = {proc_ex + chg_ex} busy minutes of {ex.t}: "
      f"u = {(proc_ex + chg_ex) / ex.t:.3f} = (4/{ex.t}) x E[omega] with E[omega] = {(proc_ex + chg_ex) / 4:.0f} -> {4 / ex.t * (proc_ex + chg_ex) / 4:.3f}")
for level in LEVELS:
    for meth in ("FIFO", "SST", "CP-SAT"):
        rr = [x for x in RUNS if x["level"] == level and x["method"] == meth]
        day_u = np.mean([float(x["utilisation"]) + float(x["setup_time"]) / (5 * float(x["makespan"])) for x in rr])
        print(f"  {level:20s} {meth:7s} busy fraction over its own day (n/T) E[omega]/m = {day_u:.3f} "
              f"(with the nominal lambda: see Part 4)")
print("job-minutes in the shop per day, n F_bar = integral of N(t) dt (60 jobs a day):")
jm = {}
for level in LEVELS:
    for meth in ("FIFO", "SPT", "EDD", "SST", "DQN", "Greedy replan", "CP-SAT"):
        w = np.mean([float(x["mean_flow_time"]) for x in RUNS if x["level"] == level and x["method"] == meth])
        w = round(w, 1)                       # the W quoted on the page, so that 60 x W checks by hand
        jm[level, meth] = 60 * w
        rr = [x for x in RUNS if x["level"] == level and x["method"] == meth]
        # a day starts empty at 0 and ends empty at the makespan, so L_day = n W / C_max exactly
        Lday = np.mean([60 * float(x["mean_flow_time"]) / float(x["makespan"]) for x in rr])
        print(f"  {level:20s} {meth:14s} n F_bar = 60 x {w:.1f} = {60 * w:,.0f} job-minutes "
              f"({100 * (1 - 60 * w / jm[level, 'FIFO']):.0f} % fewer than FIFO); day-average L = n F_bar / C_max = {Lday:.2f}")
print(f"FIFO minus CP-SAT, no breakdowns: {jm['no breakdowns', 'FIFO'] - jm['no breakdowns', 'CP-SAT']:,.0f} job-minutes a day")
print("re-running the four rules on the 20 test days, N(t) measured from the executed log:")
t0 = time.perf_counter()
diff = 0.0
for level, mtbf in LEVELS.items():
    for rule in ("FIFO", "SPT", "EDD", "SST"):
        Lfull, LWfull, lamT, Wd = [], [], [], []
        for s in SEEDS:
            sc = make_scenario(WS, s, mtbf=mtbf)
            lg = simulate(WS, sc, RuleController(rule))
            met = run_metrics(WS, sc, lg)
            csv_w = [float(x["mean_flow_time"]) for x in RUNS if x["level"] == level and x["method"] == rule and int(x["seed"]) == s][0]
            diff = max(diff, abs(met["mean_flow_time"] - csv_w))
            comp = {}
            for sg in lg.segments:
                comp[sg.job] = max(comp.get(sg.job, 0), sg.end)
            r_ = np.array([j.arrival for j in sc.jobs]); c_ = np.array([comp[j.idx] for j in sc.jobs])
            Tend = int(c_.max())
            minutes = np.arange(Tend)
            # N(t) counted minute by minute from the log (all times are integers, so this is exact)
            Nt = ((r_[None, :] <= minutes[:, None]) & (minutes[:, None] < c_[None, :])).sum(axis=1)
            Lfull.append(Nt.mean()); lamT.append(len(r_) / Tend); Wd.append((c_ - r_).mean())
            LWfull.append(len(r_) / Tend * (c_ - r_).mean())
        print(f"  {level:20s} {rule}: day average L = {np.mean(Lfull):.3f}, (n/T) F_bar = {np.mean(LWfull):.3f} "
              f"(n/T = 1/{1 / np.mean(lamT):.1f} min);  nominal lambda F_bar = {np.mean(Wd) / m_ia:.2f}")
print(f"rerun agrees with runs.csv mean_flow_time to {diff:.1e}; 160 runs in {time.perf_counter() - t0:.1f} s")


# ------------------------------------------------------------------------------------------------
section("Part 7 (simpy): the two small SimPy programs of the chapter")
import simpy  # noqa: E402


def arrivals(env, gaps):
    for name, g in zip("ABCD", gaps):
        yield env.timeout(g)              # sleep g simulated minutes
        print(f"t = {env.now}: order {name} arrives")


env = simpy.Environment()
env.process(arrivals(env, [0, 5, 7, 28]))
env.run()                                 # t = 0, 5, 12, 40


def machine(env):
    yield env.timeout(20)                 # A is processed from 0 to 20
    left = 10                             # then the colour change for B
    while left > 0:
        start = env.now
        try:
            yield env.timeout(left)
            left = 0
        except simpy.Interrupt as stop:
            left -= env.now - start       # keep the work already done
            print(f"t = {env.now}: interrupted, {left} min of changeover left")
            yield env.timeout(stop.cause) # wait out the repair
    print(f"t = {env.now}: changeover done")


def breakdown(env, victim):
    yield env.timeout(25)
    victim.interrupt(8)                   # down for 8 minutes


env = simpy.Environment()
proc = env.process(machine(env))
env.process(breakdown(env, proc))
env.run()

section("Part 8 (code): the defaults of make_scenario")
import inspect  # noqa: E402
_d = inspect.signature(make_scenario).parameters
print(f"n_jobs {_d['n_jobs'].default}, load {_d['load'].default}, mttr {_d['mttr'].default}; due date r_j + k W_f "
      f"with k uniform in [{_d['tightness'].default[0]:.0f}, {_d['tightness'].default[1]:.0f}]")

section("Exercise 4: a LIFO controller through the controller interface, 20 test days, no breakdowns")
from jobshop.checker import check_schedule  # noqa: E402
from jobshop.rules import route_earliest_finish  # noqa: E402


class LIFO:
    """Last come, first served: the job that entered the queue most recently."""
    name, decision_time_s, n_replans = "LIFO", 0.0, 0

    def route(self, shop, jr):
        return route_earliest_finish(shop, jr)

    def sequence(self, shop, m):
        return max(m.queue, key=lambda j: (j.queued_at, -j.idx))

    def on_event(self, shop, kind, **info):
        pass


res, errs = [], 0
for s in SEEDS:
    sc = make_scenario(WS, s)
    lg = simulate(WS, sc, LIFO())
    errs += len(check_schedule(WS, sc, lg))
    res.append(run_metrics(WS, sc, lg))
fifo = [x for x in RUNS if x["level"] == "no breakdowns" and x["method"] == "FIFO"]
print(f"LIFO: checker errors {errs}; mean tardiness {np.mean([r['mean_tardiness'] for r in res]):.1f} min, "
      f"max tardiness {np.mean([r['max_tardiness'] for r in res]):.0f}, mean flow time {np.mean([r['mean_flow_time'] for r in res]):.1f} min")
print(f"FIFO (runs.csv): mean tardiness {np.mean([float(x['mean_tardiness']) for x in fifo]):.1f}, "
      f"max tardiness {np.mean([float(x['max_tardiness']) for x in fifo]):.0f}, mean flow time {np.mean([float(x['mean_flow_time']) for x in fifo]):.1f}")

section("Exercises")
print(f"Ex. memoryless: P(X > 30 | X > 10) = P(X > 20) = exp(-20/20.83) = {np.exp(-20 / m_ia):.3f}")
a2 = 60 / 80
print(f"Ex. MTBF 60, MTTR 20: A = {60 / 80:.2f}; fastest-route load per available machine = {0.60 / a2:.2f}")
print(f"Ex. halving 1/lambda to {m_ia / 2:.2f} min: rho = {EW / (m_ia / 2) / 5:.2f} > 1, the queue grows without bound")


section("Exercise 5 (exercises): a one-machine SimPy simulator written from scratch, the code printed on the page")
EX5 = r'''
import itertools, random
import simpy

class Machine:
    def __init__(self, env):
        self.env, self.colour, self.queue = env, "blue", []
        self.up, self.working, self.busy = True, False, 0.0
        self.flows, self.work, self.wake = [], [], env.event()
        self.inside, self.area, self.last, self.log = 0, 0.0, 0.0, []
        self.proc = env.process(self.run())

    def count(self, step):                      # +1 arrival, -1 departure
        now = self.env.now                      # area under N(t), measured
        self.area += self.inside * (now - self.last)
        self.inside, self.last = self.inside + step, now
        self.log.append((now, self.inside))

    def poke(self):                             # an arrival or a repair
        if not self.wake.triggered:
            self.wake.succeed()

    def wait_until(self, ok):
        while not ok():
            self.wake = self.env.event()
            yield self.wake

    def run(self):
        while True:
            yield from self.wait_until(lambda: self.queue and self.up)
            colour, p, arrival = self.queue.pop(0)                # FIFO
            left = (10 if colour != self.colour else 0) + p       # colour change
            self.colour = colour
            self.work.append(left)
            while left > 0:
                yield from self.wait_until(lambda: self.up)
                start, self.working = self.env.now, True
                try:
                    yield self.env.timeout(left)
                    left = 0
                except simpy.Interrupt:
                    left -= self.env.now - start                  # keep the work done
                self.working = False
                self.busy += self.env.now - start
            self.flows.append(self.env.now - arrival)
            self.count(-1)

def arrivals(env, mc, orders):                  # orders: (colour, p, gap)
    for colour, p, gap in orders:
        yield env.timeout(gap)
        mc.queue.append((colour, p, env.now))
        mc.count(+1)
        mc.poke()

def breakdowns(env, mc, cycles):                # cycles: (up time, repair time)
    for up, repair in cycles:
        yield env.timeout(up)                   # calendar time, busy or idle
        mc.up = False
        if mc.working:
            mc.proc.interrupt()
        yield env.timeout(repair)
        mc.up = True
        mc.poke()

def run(orders, cycles, n):
    env = simpy.Environment()
    mc = Machine(env)
    env.process(arrivals(env, mc, orders))
    env.process(breakdowns(env, mc, cycles))
    while len(mc.flows) < n:                    # until the last job has left
        env.step()
    return env.now, mc

# (a) Part 2: gaps 0, 5, 7, 28; down from 25 to 33
T, mc = run([("blue", 20, 0), ("red", 10, 5), ("blue", 15, 7), ("red", 25, 28)], [(25, 8)], 4)
F = sum(mc.flows) / 4
print(mc.log)                                   # (time, N) at every change
print(T, mc.flows, f"L {mc.area / T:.3f} (n/T)F {4 / T * F:.3f} busy {mc.busy / T:.3f}")

# (b) 50,000 orders, mean gap 40; MTBF 120, repairs uniform on [10, 30]
n, lam, arr, brk = 50_000, 1 / 40, random.Random(1), random.Random(2)
orders = ((arr.choice(("blue", "red")), arr.choice((10, 15, 20, 25)), arr.expovariate(lam)) for _ in range(n))
cycles = ((brk.expovariate(1 / 120), brk.uniform(10, 30)) for _ in itertools.count())
T, mc = run(orders, cycles, n)
F, omega = sum(mc.flows) / n, sum(mc.work) / n
print(f"L {mc.area / T:.3f} (n/T)F {n / T * F:.3f} lambda F {lam * F:.3f}")
print(f"busy {mc.busy / T:.4f} (n/T)E[omega] {n / T * omega:.4f} lambda E[omega] {lam * omega:.4f} E[omega] {omega:.2f}")
'''
exec(compile(EX5, "exercise5", "exec"), {"__name__": "exercise5"})
print("(a) the hand trace; (b) the identities hold exactly on a window that starts and ends empty, the nominal values differ by chance")
print(f"  (b) nominal E[omega] = 17.5 + 10/2 = {17.5 + 10 / 2:.2f} min (half the jobs change colour under random colours)")
