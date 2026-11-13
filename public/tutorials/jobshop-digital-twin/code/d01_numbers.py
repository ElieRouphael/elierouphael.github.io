"""Reproduce every number quoted on chapter D1 of "Which Job Should the Machine Do Next?",
"Dispatching as reinforcement learning" (d01-dispatching-as-reinforcement-learning.html).

Usage (from the course folder, or with JOBSHOP_REPO pointing at the repository):
    JOBSHOP_REPO=path/to/jobshop-digital-twin python d01_numbers.py
Needs numpy, simpy and torch (the repository's requirements). About one minute.

Sections, in the order of the page:
  Hero / Part 1   the toy: tabular Q-learning picks a rule on random six-order paint-machine instances
                  (a line-by-line port of interactive/d01.js, same seeded generator, same numbers)
  Part 2          the agent's state, actions and decisions in the workshop (jobshop/dqn.py)
  Part 3          the telescoping reward, checked on short simulated runs; what discounting changes
  Part 4          Bellman's equation on a two-decision example
  Part 5          overestimation: E[max] >= max E, the coin example and the Gaussian value
  Part 6          the hyper-parameters of DQNConfig and the size of the network
  Part 7          training curves and checkpoint selection, from models/training.json
  Part 8          the test results, from results/runs.csv (never re-run here)
  Data            the arrays embedded in interactive/d01.js for the training-curve widget
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(os.environ.get("JOBSHOP_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

T0 = time.perf_counter()


def section(title: str) -> None:
    print(f"\n=== {title}")


# =====================================================================================================
# The toy of the hero: one machine, two colours, six orders, all waiting at t = 0.
# Everything below mirrors interactive/d01.js operation by operation (same generator, same order of
# random draws, same tie-breaking), so the page and this script print the same numbers.
# =====================================================================================================
MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & MASK


def mulberry32(seed: int):
    """JS.rng of interactive/jobshop-lib.js, bit for bit."""
    state = [seed & MASK]

    def r() -> float:
        a = (state[0] + 0x6D2B79F5) & MASK
        state[0] = a
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & MASK) ^ t
        return ((t ^ (t >> 14)) & MASK) / 4294967296

    return r


N_ORDERS = 6
RULE_NAMES = ("FIFO", "SPT", "EDD", "SST")


def toy_instance(r, spread: float):
    """Six orders (idx, colour 0 blue / 1 red, painting time 5-25, due date >= painting time)."""
    raw = []
    for _ in range(N_ORDERS):
        colour = 1 if r() < 0.5 else 0
        p = 5 + math.floor(r() * 21)
        raw.append((colour, p))
    total = sum(p for _, p in raw)
    return [(j, c, p, p + math.floor(r() * spread * total)) for j, (c, p) in enumerate(raw)]


class Toy:
    def __init__(self, setup: int = 5, spread: float = 0.7):
        self.S = setup
        self.spread = spread
        rv, rt = mulberry32(999), mulberry32(777)
        self.val = [toy_instance(rv, spread) for _ in range(200)]
        self.test = [toy_instance(rt, spread) for _ in range(200)]

    def setup(self, f: int, g: int) -> int:
        return 0 if f == g else self.S

    def key(self, a: int, o, f: int):
        j, c, p, d = o
        return ((j,), (p, j), (d, j), (self.setup(f, c), d, j))[a]   # SST ties by due date, as jobshop/rules.py

    def pick(self, a: int, wait, f: int):
        best = wait[0]
        for o in wait[1:]:
            if self.key(a, o, f) < self.key(a, best, f):
                best = o
        return best

    @staticmethod
    def committed(t: int, done_tardiness: int, wait) -> int:
        """E(t): finished orders in full, a waiting order as if it started now with no changeover."""
        return done_tardiness + sum(max(0, t + p - d) for _, _, p, d in wait)

    def state(self, t: int, f: int, wait) -> int:
        n = len(wait)
        size = 0 if n == 2 else (1 if n <= 4 else 2)
        e = self.pick(2, wait, f)                      # the most urgent order
        if e[1] == f:
            colour = 1                                 # it has the current colour: EDD and SST agree
        elif any(o[1] == f for o in wait):
            colour = 2                                 # it needs a change, another order does not: they disagree
        else:
            colour = 0                                 # every order needs a change: EDD and SST agree
        tight = 1 if e[3] - t - self.setup(f, e[1]) - e[2] < self.S else 0
        return (size * 3 + colour) * 2 + tight

    def episode(self, inst, policy, learn=None, trace=None):
        t, f, wait, done_t, pending = 0, 0, list(inst), 0, None
        while wait:
            if len(wait) >= 2:
                s, E = self.state(t, f, wait), self.committed(t, done_t, wait)
                if pending is not None:
                    ps, pa, pE = pending
                    if learn:
                        learn(ps, pa, -(E - pE), s, False)
                    if trace is not None:
                        trace.append((ps, pa, pE, -(E - pE)))
                a = policy(s)
                pending = (s, a, E)
                o = self.pick(a, wait, f)
            else:
                o = wait[0]
            wait.remove(o)
            t += self.setup(f, o[1]) + o[2]
            f = o[1]
            done_t += max(0, t - o[3])
        ps, pa, pE = pending
        if learn:
            learn(ps, pa, -(done_t - pE), None, True)
        if trace is not None:
            trace.append((ps, pa, pE, -(done_t - pE)))
        return done_t

    def seq_tardiness(self, seq) -> int:
        t, f, total = 0, 0, 0
        for _, c, p, d in seq:
            t += self.setup(f, c) + p
            f = c
            total += max(0, t - d)
        return total

    def mean_rule(self, insts, a: int) -> float:
        return sum(self.episode(i, lambda s: a) for i in insts) / len(insts)

    def best_succession(self, inst) -> int:
        """Best of the 4^5 = 1024 successions of rules (one rule per decision, in order), in hindsight:
        the ceiling of any rule selector that may switch rules at every decision."""
        best = math.inf
        for succ in itertools.product(range(4), repeat=N_ORDERS - 1):
            it = iter(succ)
            best = min(best, self.episode(inst, lambda s, it=it: next(it)))
        return best

    def room(self, insts):
        """(best fixed rule chosen on validation, its test score, hindsight single rule, hindsight succession, optimum)"""
        v = [self.mean_rule(self.val, a) for a in range(4)]
        b = min(range(4), key=lambda a: v[a])
        single = np.mean([min(self.episode(i, lambda s, a=a: a) for a in range(4)) for i in insts])
        succ = np.mean([self.best_succession(i) for i in insts])
        opt = np.mean([min(self.seq_tardiness(p) for p in itertools.permutations(i)) for i in insts])
        return b, self.mean_rule(insts, b), single, succ, opt

    def train(self, seed: int = 1, episodes: int = 5000, alpha: float = 0.1, gamma: float = 1.0, every: int = 100):
        r = mulberry32(seed)
        Q = [[0.0] * 4 for _ in range(18)]
        visits = [0] * 18
        done = [0]

        def greedy_of(Qm):
            def g(s):
                q, b = Qm[s], 0
                for a in range(1, 4):
                    if q[a] > q[b]:
                        b = a
                return b
            return g

        greedy = greedy_of(Q)

        def behaviour(s):
            visits[s] += 1
            eps = max(0.05, 1 - done[0] / (0.6 * episodes))
            if r() < eps:
                return math.floor(r() * 4)
            return greedy(s)

        def learn(s, a, rew, s2, terminal):
            target = rew + (0.0 if terminal else gamma * max(Q[s2]))
            Q[s][a] += alpha * (target - Q[s][a])

        curve, best = [], (math.inf, None, 0)
        for ep in range(episodes + 1):
            if ep % every == 0:
                v = sum(self.episode(i, greedy) for i in self.val) / len(self.val)
                curve.append(v)
                if v < best[0]:
                    best = (v, [q[:] for q in Q], ep)
            if ep == episodes:
                break
            self.episode(toy_instance(r, self.spread), behaviour, learn)
            done[0] += 1
        Qb = best[1]
        gb = greedy_of(Qb)
        test = sum(self.episode(i, gb) for i in self.test) / len(self.test)
        return {"curve": curve, "best_val": best[0], "best_ep": best[2], "test": test, "Q": Qb,
                "policy": [gb(s) for s in range(18)], "visits": visits, "greedy": gb}


section("Hero and Part 1: the toy, default settings (changeover 5 min, due-date spread 0.7, seed 1, "
        "5000 episodes, step size 0.1, gamma 1)")
toy = Toy(5, 0.7)
val_rules = [toy.mean_rule(toy.val, a) for a in range(4)]
test_rules = [toy.mean_rule(toy.test, a) for a in range(4)]
best_rule = min(range(4), key=lambda a: val_rules[a])
print("validation, total tardiness of the six orders (min), FIFO SPT EDD SST:", [round(v, 1) for v in val_rules])
print("test, same:                                                          ", [round(v, 1) for v in test_rules])
print(f"best fixed rule on validation: {RULE_NAMES[best_rule]}; on test it gives {test_rules[best_rule]:.1f}")
hind = [min(toy.episode(i, lambda s, a=a: a) for a in range(4)) for i in toy.test]
opt = [min(toy.seq_tardiness(p) for p in itertools.permutations(i)) for i in toy.test]
print(f"test, best of the four rules on each instance (hindsight): {np.mean(hind):.1f}")
print(f"test, optimal sequence of each instance (all 720 orders): {np.mean(opt):.1f}")
print(f"validation, optimal sequence (the dotted line of the hero): "
      f"{np.mean([min(toy.seq_tardiness(p) for p in itertools.permutations(i)) for i in toy.val]):.1f}")
succ5 = np.mean([toy.best_succession(i) for i in toy.test])
print(f"test, best succession of the four rules for each instance (4^5 = 1024 successions, hindsight): {succ5:.1f}"
      f" (optimum {np.mean(opt):.1f}: successions come within {succ5 - np.mean(opt):.1f} min of it)")
print(f"room between the best fixed rule and the best rule per instance (a lower bound on the room): "
      f"{test_rules[best_rule] - np.mean(hind):.1f} min")
print(f"room between the best fixed rule and the best succession per instance (the room of a rule selector): "
      f"{test_rules[best_rule] - succ5:.1f} min")
res = toy.train(1)
print(f"agent: validation curve starts at {res['curve'][0]:.1f} (untrained greedy = FIFO everywhere), "
      f"best checkpoint at episode {res['best_ep']} with validation {res['best_val']:.1f}; test {res['test']:.1f}")
print(f"agent minus best fixed rule on test: {res['test'] - test_rules[best_rule]:+.1f} min "
      f"({100 * (res['test'] - test_rules[best_rule]) / test_rules[best_rule]:+.0f} %)")
print("checkpoints (validation) every 100 episodes, first 11:", [round(v, 1) for v in res["curve"][:11]])
reachable = [s for s in range(18) if res["visits"][s] > 0]
print(f"states visited during training: {len(reachable)} of 18 (state = size x colour situation x urgency)")
print("greedy rule of the best checkpoint per state (size, colour, tight) -> rule [visits]:")
for s in range(18):
    size, rest = divmod(s, 6)
    colour, tight = divmod(rest, 2)
    print(f"   s={s:2d} size={('2', '3-4', '5-6')[size]:>3} colour={('none same', 'urgent same', 'conflict')[colour]:<11} "
          f"tight={tight}  {RULE_NAMES[res['policy'][s]]:<4} [{res['visits'][s]}]  Q={[round(q, 1) for q in res['Q'][s]]}")
seeds = [toy.train(sd) for sd in range(1, 6)]
print("five training seeds 1-5, test of the best checkpoint:", [round(x["test"], 1) for x in seeds])
print("decisions per episode in the toy:", N_ORDERS - 1, "(the last order needs no decision)")

section("Hero and exercise 6 (Room to learn): the toy at changeovers 0, 5, 10 and 15 min, same seed, test set")
for S in (0, 5, 10, 15):
    tS = toy if S == 5 else Toy(S, 0.7)
    b, best_t, single, succ, optS = tS.room(tS.test)
    agent = res["test"] if S == 5 else tS.train(1)["test"]
    print(f"changeover {S:2d}: test rules {[round(tS.mean_rule(tS.test, a), 1) for a in range(4)]}; best on validation "
          f"{RULE_NAMES[b]} {best_t:.1f}; agent {agent:.1f} (gain {best_t - agent:.1f}); hindsight single rule "
          f"{single:.1f}; hindsight succession {succ:.1f}; optimum {optS:.1f}; room (best rule - succession) "
          f"{best_t - succ:.1f}, its lower bound (best rule - single rule) {best_t - single:.1f}")

section("Part 3: the telescoping identity on one toy run (test instance 0, the trained agent)")
trace: list = []
T_inst = toy.episode(toy.test[0], res["greedy"], trace=trace)
print("orders (idx, colour, p, due):", toy.test[0], " letters A-F = idx 0-5")


def state_label(s: int) -> str:
    """The situation as the telescoping widget prints it."""
    return (f"{('2', '3-4', '5-6')[s // 6]} waiting, "
            f"{('all need a change', 'urgent has the colour', 'conflict')[(s % 6) // 2]}, {('slack', 'tight')[s % 2]}")


for k, (s, a, E, rew) in enumerate(trace):
    print(f"   decision {k}: state {s:2d} ({state_label(s)}), rule {RULE_NAMES[a]}, committed tardiness E = {E}, "
          f"reward {rew}")
print(f"sum of rewards = {sum(x[3] for x in trace)}; total tardiness = {T_inst}")
for g in (0.9, 0.97):
    print(f"discounted return with gamma {g}: {sum(g ** k * x[3] for k, x in enumerate(trace)):.2f}")


def hand_trace(tt: "Toy", inst, rules, names="ABCDEF") -> None:
    """E(t) term by term at each decision (Part 3 and exercise 1). rules: one rule per decision."""
    t, f, wait, done, Es, k = 0, 0, list(inst), [], [], 0
    while len(wait) >= 2:
        terms = [f"{names[o[0]]}: {t}+{o[2]}-{o[3]} = {t + o[2] - o[3]}" for o in wait]
        E = sum(T for _, T in done) + sum(max(0, t + o[2] - o[3]) for o in wait)
        Es.append(E)
        print(f"   decision {k} at t = {t}: finished (order, T) {[(names[j], T) for j, T in done]}; waiting, "
              f"t + p - d: {terms}; E = {E}; rule {RULE_NAMES[rules[k]]}")
        o = tt.pick(rules[k], wait, f)
        wait.remove(o)
        t += tt.setup(f, o[1]) + o[2]
        f = o[1]
        done.append((o[0], max(0, t - o[3])))
        print(f"      -> {names[o[0]]} ends at {t} (due {o[3]}, late {max(0, t - o[3])})")
        k += 1
    o = wait[0]
    t += tt.setup(f, o[1]) + o[2]
    done.append((o[0], max(0, t - o[3])))
    total = sum(T for _, T in done)
    Es.append(total)
    print(f"   end at t = {t}: (order, T) {[(names[j], T) for j, T in done]}; total tardiness {total}; "
          f"rewards {[-(b - a) for a, b in zip(Es, Es[1:])]}, sum {-(total - Es[0])}")


print("by hand: test instance 0 under the agent's rules, E(t) term by term")
hand_trace(toy, toy.test[0], [a for _, a, _, _ in trace])

section("Exercise 1: the rewards of the paint example (notation.md): changeover 10 min, set up for blue")
PAINT = [(0, 0, 20, 30), (1, 1, 10, 25), (2, 0, 15, 70), (3, 1, 25, 60)]   # A blue, B red, C blue, D red
toyP = Toy(10, 0.7)
for a in (3, 0):
    print(f"{RULE_NAMES[a]} at every decision:")
    hand_trace(toyP, PAINT, [a] * 3)

# =====================================================================================================
# The workshop agent (jobshop/dqn.py)
# =====================================================================================================
import torch  # noqa: E402

from jobshop.dqn import (ACTIONS, STATE_DIM, DQNConfig, DQNController, Learner, QNetwork,  # noqa: E402
                         estimated_tardiness, features, load_policy, total_tardiness)
from jobshop.rules import RuleController, pick  # noqa: E402
from jobshop.scenario import LEVELS, make_scenario, make_workshop  # noqa: E402
from jobshop.shop import Shop  # noqa: E402

torch.set_num_threads(1)
WS = make_workshop()
CFG = DQNConfig()

section("Part 2: state and actions")
print("actions:", ACTIONS, " state dimension:", STATE_DIM, "= 4 rules x 5 job features + 9 queue/shop features")


class Probe(DQNController):
    """Random rule at each decision; records E(t) and the state (as tests/test_dqn.py)."""

    def __init__(self, seed=0):
        super().__init__()
        self.r = np.random.default_rng(seed)
        self.E, self.n = [], 0

    def sequence(self, shop, m):
        if len(m.queue) == 1:
            return m.queue[0]
        features(shop, m)
        self.E.append(estimated_tardiness(shop))
        self.n += 1
        return pick(ACTIONS[int(self.r.integers(4))], shop, m)


section("Part 3: the telescoping identity in the workshop")
config0 = replace(DQNConfig(), min_buffer_size=10**9)     # collect transitions only, no optimisation
# seed 91: the repository's test scenario; 100001 / 100002: training-range scenarios; 5000 with MTBF 120: exercise 4
for seed, n_jobs, mtbf in ((91, 40, 120.0), (100_001, 60, None), (100_002, 60, 120.0), (5000, 60, 120.0)):
    learner = Learner(config0, seed=0)
    shop = Shop(WS, make_scenario(WS, seed, n_jobs=n_jobs, mtbf=mtbf), DQNController(learner=learner))
    shop.run()
    T = total_tardiness(shop)
    learner.end_episode(T)
    tr = list(learner.buffer._buffer)
    rew = np.array([x.reward for x in tr])
    disc = sum(CFG.gamma ** k * x for k, x in enumerate(rew))
    print(f"scenario seed {seed}, {n_jobs} jobs, {'no breakdowns' if mtbf is None else 'MTBF 120'}: "
          f"{len(tr)} decisions, total tardiness {T} min, sum of rewards x {CFG.reward_scale:.0f} = "
          f"{rew.sum() * CFG.reward_scale:.6f} (difference = E at the first decision: "
          f"{T + rew.sum() * CFG.reward_scale:.0f}); every reward <= 0: {bool((rew <= 0).all())}; "
          f"rewards equal to 0: {int((rew == 0).sum())}; "
          f"discounted return (gamma {CFG.gamma}) x 50 = {disc * CFG.reward_scale:.1f}")
first_E = []
counts = []
for seed in range(100_000, 100_010):
    for mtbf in LEVELS.values():
        pr = Probe(seed)
        sh = Shop(WS, make_scenario(WS, seed, n_jobs=60, mtbf=mtbf), pr)
        sh.run()
        first_E.append(pr.E[0])
        counts.append(pr.n)
        assert all(b >= a for a, b in zip(pr.E, pr.E[1:]))
        assert estimated_tardiness(sh) == total_tardiness(sh)
print(f"20 training scenarios (seeds 100000-100009, both levels) under a random policy: E at the first decision "
      f"= 0 in {sum(e == 0 for e in first_E)} of 20, otherwise {sorted(e for e in first_E if e)} min "
      f"(the sum of the rewards is -(T - E(t_0)) / 50, exact only when E(t_0) = 0); E never decreased; "
      f"E at the end = total tardiness; decisions per run {min(counts)}-{max(counts)}, mean {np.mean(counts):.0f}")
print(f"effective horizon of gamma = {CFG.gamma}: 1/(1-gamma) = {1 / (1 - CFG.gamma):.1f} (about {1 / (1 - CFG.gamma):.0f}) decisions; "
      f"weight of the reward 79 decisions ahead: gamma^79 = {CFG.gamma ** 79:.3f}")
print(f"weight after 23 decisions: gamma^23 = {CFG.gamma ** 23:.3f} (half-life ln 2 / -ln gamma = "
      f"{math.log(2) / -math.log(CFG.gamma):.1f} decisions)")

section("Part 4: Bellman's equation on the two-decision example")
# Example on the page (gamma = 1): from s0, action a gives reward -2 and leads to s1; action b gives -4 and ends.
# From s1, action a ends with reward -6, action b ends with reward -3.
Q_s1 = {"a": -6, "b": -3}
Q_s0 = {"a": -2 + max(Q_s1.values()), "b": -4}
print("Q*(s1, .) =", Q_s1, " Q*(s0, .) =", Q_s0, " optimal first action:", max(Q_s0, key=Q_s0.get),
      " (the myopic choice, best immediate reward, is a: -2 > -4)")
for g in (0.5, 0.8):
    qa = -2 + g * max(Q_s1.values())
    print(f"with gamma = {g}: Q(s0, a) = {qa:.1f}, Q(s0, b) = -4 -> {'a' if qa > -4 else 'b'}  (check question 1 for 0.8)")
print("gamma at which a and b tie: -2 - 3 gamma = -4 -> gamma =", round(2 / 3, 3))

section("Exercise 2: Q-learning by hand on the two-decision example (alpha 0.5, gamma 1, Q = 0 at the start)")
Qh = {("s0", "a"): 0.0, ("s0", "b"): 0.0, ("s1", "a"): 0.0, ("s1", "b"): 0.0}
CYCLE = [("s0", "a", -2, "s1"), ("s1", "a", -6, None), ("s0", "b", -4, None), ("s1", "b", -3, None)]


def q_step(s, a, rwd, s2, alpha=0.5, gamma=1.0):
    target = rwd + (0.0 if s2 is None else gamma * max(Qh[(s2, "a")], Qh[(s2, "b")]))
    Qh[(s, a)] += alpha * (target - Qh[(s, a)])
    return target


def q_table():
    return (f"Q(s0,a) {Qh[('s0', 'a')]:.2f}, Q(s0,b) {Qh[('s0', 'b')]:.2f}, "
            f"Q(s1,a) {Qh[('s1', 'a')]:.2f}, Q(s1,b) {Qh[('s1', 'b')]:.2f}")


for n, tr_ in enumerate(CYCLE + CYCLE[:1], start=1):
    y = q_step(*tr_)
    print(f"   step {n}: transition {tr_}, target {y:.2f} -> {q_table()}")
for _ in range(19):
    for tr_ in CYCLE[1:] + CYCLE[:1]:
        q_step(*tr_)
print(f"   after 20 passes through the four transitions: {q_table()}  (Q* = -5, -4, -6, -3)")

section("Part 5: overestimation, E[max] >= max E")
vals = [1, -1]
emax = np.mean([max(x, y) for x in vals for y in vals])
print(f"coin example: each estimate is 0 +/- 1 with probability 1/2, independent: E[max] = {emax}, max E = 0")
dbl = np.mean([b2 if a1 >= a2 else b1 for a1 in vals for a2 in vals for b1 in vals for b2 in vals])
print(f"double estimator (select with A, evaluate with an independent B): mean {dbl}")
print(f"two Gaussian estimates N(0, sigma^2): E[max] = sigma / sqrt(pi) = {1 / math.sqrt(math.pi):.3f} sigma")
rg = np.random.default_rng(0)
xa, xb = rg.normal(size=(2, 400_000)), rg.normal(size=(2, 400_000))
print(f"   Monte Carlo check: {np.maximum(xa[0], xa[1]).mean():.3f}; double estimator "
      f"{np.where(xa[0] >= xa[1], xb[0], xb[1]).mean():+.3f}")

section("Exercise 3: the bias of a maximum, by hand")
print(f"E|Z| for Z ~ N(0, 2 sigma^2) = sigma * sqrt(2) * sqrt(2/pi) = {math.sqrt(2) * math.sqrt(2 / math.pi):.4f} sigma; "
      f"half of it = {math.sqrt(2) * math.sqrt(2 / math.pi) / 2:.4f} = 1/sqrt(pi)")

section("Part 6 (code walk): the TD targets of tests/test_dqn.py on hand-made Q-values")
from jobshop.dqn import compute_td_targets  # noqa: E402


class FixedQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor([values], dtype=torch.float32)

    def forward(self, x):
        return self.values.expand(x.shape[0], -1)


online, target = FixedQ([0.0, 9.0, 0.0, 0.0]), FixedQ([4.0, 1.0, 7.0, 0.0])
args = (torch.zeros(1, 1), torch.zeros(1, STATE_DIM), torch.zeros(1, 1), online, target, 1.0)
print(f"online (0, 9, 0, 0), target (4, 1, 7, 0), r = 0, gamma = 1: Double DQN target "
      f"{compute_td_targets(*args, double_dqn=True).item():.0f}, plain DQN target "
      f"{compute_td_targets(*args, double_dqn=False).item():.0f}")

section("Part 6: the hyper-parameters (DQNConfig) and the network")
for k, v in asdict(CFG).items():
    print(f"   {k} = {v}")
net = QNetwork(STATE_DIM, len(ACTIONS), CFG.hidden_sizes)
print("network parameters (weights and biases):", sum(p.numel() for p in net.parameters()), "= 29*64+64 + 64*64+64 + 64*4+4")
print(f"epsilon schedule: {CFG.epsilon_end} + {CFG.epsilon_start - CFG.epsilon_end:.2f} exp(-N / {CFG.epsilon_decay_steps})")
print(f"epsilon after 30000 decisions: {CFG.epsilon_end + (CFG.epsilon_start - CFG.epsilon_end) * math.exp(-1):.3f}; "
      f"after 118760 (seed 3, end of training): "
      f"{CFG.epsilon_end + (CFG.epsilon_start - CFG.epsilon_end) * math.exp(-118760 / 30000):.3f}")
print(f"soft update: the target keeps (1 - tau)^n of an old value; half-life ln 2 / tau = "
      f"{math.log(2) / CFG.tau_soft_update:.0f} gradient steps")

section("Part 7: training curves and checkpoint selection (models/training.json)")
tj = json.loads((REPO / "models" / "training.json").read_text())
print("selected training seed:", tj["selected_seed"])
curves = {}
for run in tj["runs"]:
    h = run["history"]
    val = [(x["episode"], x["validation"]) for x in h if "validation" in x]
    curves[run["seed"]] = (h, val)
    vv = [v for _, v in val]
    print(f"seed {run['seed']}: best validation {run['best_validation']:.2f} at episode {run['best_episode']}, "
          f"{run['steps']} decisions ({run['steps'] / 1500:.0f} per episode), {run['wall_time_s'] / 60:.0f} min; "
          f"first check {vv[0]:.2f}, last {vv[-1]:.2f}, range {min(vv):.2f}-{max(vv):.2f}, "
          f"final epsilon {h[-1]['epsilon']:.3f}")
allv = np.array([[v for _, v in curves[s][1]] for s in sorted(curves)])
print(f"validation checks per seed: {allv.shape[1]}; mean over seeds of the std of the last 20 checks: "
      f"{allv[:, -20:].std(axis=1).mean():.2f}")
print(f"mean of the five best validations {np.mean([r['best_validation'] for r in tj['runs']]):.2f}; "
      f"mean of the five final validations {allv[:, -1].mean():.2f}")
first_train = np.mean([np.mean([x["train_mean_tardiness"] for x in curves[s][0][:100]]) for s in curves])
last_train = np.mean([np.mean([x["train_mean_tardiness"] for x in curves[s][0][-100:]]) for s in curves])
print(f"training episodes, mean tardiness per job: first 100 episodes {first_train:.1f}, last 100 {last_train:.1f} "
      f"(mixed levels, exploring policy)")

print("validation score of the rules on the same scenarios (seeds 5000-5009, both levels, 60 jobs):")
val_rule = {}
for rule in ACTIONS:
    vs = []
    for mtbf in LEVELS.values():
        for seed in range(5000, 5010):
            sh = Shop(WS, make_scenario(WS, seed, n_jobs=60, mtbf=mtbf), RuleController(rule))
            sh.run()
            vs.append(total_tardiness(sh) / 60)
    val_rule[rule] = float(np.mean(vs))
    print(f"   {rule}: {val_rule[rule]:.2f}")
best_val_rule = min(val_rule, key=val_rule.get)
print(f"best rule on validation: {best_val_rule} {val_rule[best_val_rule]:.2f}; selected agent 21.98 is "
      f"{val_rule[best_val_rule] - 21.98:.2f} min lower on the scenarios it was selected on")
seeds_below = sum(r["best_validation"] < val_rule[best_val_rule] for r in tj["runs"])
print(f"training seeds whose best checkpoint beats the best rule on validation: {seeds_below} of 5")
frac_below = np.mean(allv < val_rule[best_val_rule])
print(f"share of all {allv.size} validation checks (5 seeds x 60) below the best rule: {100 * frac_below:.0f} %")
from jobshop.dqn import validate  # noqa: E402
print(f"re-validated models/dqn.pt: {validate(WS, load_policy(REPO / 'models' / 'dqn.pt'), 60):.2f}; "
      f"dqn_seed0-4.pt (exercise 5): "
      + ", ".join(f"{validate(WS, load_policy(REPO / 'models' / f'dqn_seed{k}.pt'), 60):.2f}" for k in range(5)))
print(f"mean decisions (= gradient steps, less the 1000 of warm-up) per training seed: "
      f"{np.mean([r['steps'] for r in tj['runs']]):.0f}")

# =====================================================================================================
section("Part 8: test results (results/runs.csv; 20 test seeds per level)")
from jobshop.metrics import bootstrap_ci  # noqa: E402

rows = list(csv.DictReader(open(REPO / "results" / "runs.csv", encoding="utf-8")))
by = {}
for r in rows:
    by.setdefault((r["level"], r["method"]), {})[int(r["seed"])] = r


def series(level, method, col="mean_tardiness"):
    d = by[(level, method)]
    return np.array([float(d[s][col]) for s in sorted(d)])


def cell(x, fmt="{:.1f}"):
    lo, hi = bootstrap_ci(x)
    return f"{fmt.format(np.mean(x))} [{fmt.format(lo)}, {fmt.format(hi)}]"


for level in LEVELS:
    rules = {m: series(level, m) for m in ACTIONS}
    best = min(rules, key=lambda m: rules[m].mean())
    dqn = series(level, "DQN")
    d = dqn - rules[best]
    dmax = series(level, "DQN", "max_tardiness") - series(level, best, "max_tardiness")
    oracle = np.min(np.vstack(list(rules.values())), axis=0)
    print(f"{level}: rules " + ", ".join(f"{m} {cell(rules[m])}" for m in ACTIONS))
    print(f"   DQN {cell(dqn)}; best rule {best}; DQN - {best} = {cell(d, '{:+.1f}')}, "
          f"better on {int((d < 0).sum())}, equal {int((d == 0).sum())}, worse {int((d > 0).sum())} of 20; "
          f"max tardiness difference {cell(dmax, '{:+.0f}')}")
    print(f"   best rule chosen per day in hindsight: {oracle.mean():.1f} (a lower bound on the room: "
          f"{rules[best].mean() - oracle.mean():.2f} = {rules[best].mean():.2f} - {oracle.mean():.2f} "
          f"min below {best}); which rule wins per day: "
          + str({m: int(sum(np.argmin(np.vstack(list(rules.values())), axis=0) == i)) for i, m in enumerate(ACTIONS)}))
    print(f"   total changeover time per day, mean of the 20 days: DQN {series(level, 'DQN', 'setup_time').mean():.0f}, "
          + ", ".join(f"{m} {series(level, m, 'setup_time').mean():.0f}" for m in ACTIONS)
          + f", CP-SAT {series(level, 'CP-SAT', 'setup_time').mean():.0f} min")
    print(f"   the five training seeds on test: "
          + ", ".join(f"{k}: {series(level, f'DQN:{k}').mean():.1f}" for k in range(5)))
    print(f"   CP-SAT {series(level, 'CP-SAT').mean():.1f}, Greedy replan {series(level, 'Greedy replan').mean():.1f}")

section("Part 8: which rules the selected agent uses (test seeds 0-4, both levels; re-run here)")
policy = load_policy(REPO / "models" / "dqn.pt")
for level, mtbf in LEVELS.items():
    counts = np.zeros(4)
    for seed in range(5):
        ctrl = DQNController(policy)
        sh = Shop(WS, make_scenario(WS, seed, n_jobs=60, mtbf=mtbf), ctrl)
        sh.run()
        counts += ctrl.action_counts
        assert abs(total_tardiness(sh) / 60 - float(by[(level, "DQN")][seed]["mean_tardiness"])) < 1e-9
    share = counts / counts.sum()
    print(f"{level}: {int(counts.sum())} decisions; share " + ", ".join(f"{a} {100 * s:.0f} %" for a, s in zip(ACTIONS, share))
          + "  (mean tardiness identical to runs.csv)")

# =====================================================================================================
section("Data embedded in interactive/d01.js (TRAIN constant)")
emb = {"rules": {k: round(v, 2) for k, v in val_rule.items()}, "selected": tj["selected_seed"], "seeds": []}
for run in tj["runs"]:
    h = run["history"]
    blocks = [round(float(np.mean([x["train_mean_tardiness"] for x in h[i:i + 25]])), 2) for i in range(0, 1500, 25)]
    emb["seeds"].append({"seed": run["seed"], "best": round(run["best_validation"], 2), "bestEp": run["best_episode"],
                         "val": [round(x["validation"], 2) for x in h if "validation" in x], "train": blocks,
                         "eps": [round(h[i - 1]["epsilon"], 3) for i in range(25, 1501, 25)]})
print("const TRAIN = " + json.dumps(emb, separators=(",", ":")) + ";")
print(f"\n(done in {time.perf_counter() - T0:.0f} s)")
