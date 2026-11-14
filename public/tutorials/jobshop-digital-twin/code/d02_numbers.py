"""Reproduce every number quoted on chapter D2 of "Which Job Should the Machine Do Next?",
"A fair benchmark" (d02-a-fair-benchmark.html).

Usage (from the course folder, or with JOBSHOP_REPO pointing at the repository):
    python code/d02_numbers.py
Needs numpy, simpy and the repository's jobshop package. The benchmark numbers come from
results/runs.csv (the 40-minute evaluation is never rerun); the extra simulations (a pool of
500 fresh days for the four rules, the bugged simulator) take about half a minute.
"""
from __future__ import annotations

import csv
import itertools
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JOBSHOP_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

from jobshop.checker import check_schedule  # noqa: E402
from jobshop.metrics import bootstrap_ci, run_metrics  # noqa: E402
from jobshop.rules import RuleController  # noqa: E402
from jobshop.scenario import LEVELS, make_scenario, make_workshop  # noqa: E402
from jobshop.shop import Shop  # noqa: E402

T0 = time.perf_counter()
METHODS = ["FIFO", "SPT", "EDD", "SST", "DQN", "Greedy replan", "CP-SAT"]
RULES4 = ["FIFO", "SPT", "EDD", "SST"]
LV = list(LEVELS)                        # "no breakdowns", "frequent breakdowns"
CALM, BD = LV


def section(title):
    print(f"\n=== {title}")


def ci(v, **kw):
    lo, hi = bootstrap_ci(v, **kw)
    return f"[{lo:.1f}, {hi:.1f}]"


# ---------------------------------------------------------------- the data: results/runs.csv
rows = list(csv.DictReader(open(REPO / "results" / "runs.csv", encoding="utf-8")))
TOT: dict[tuple[str, str], list[int]] = {}       # total tardiness (minutes) per seed 0..19
MT: dict[tuple[str, str], np.ndarray] = {}       # mean tardiness per job, per seed
for level in LV:
    for m in METHODS + [f"DQN:{k}" for k in range(5)]:
        r = sorted((x for x in rows if x["level"] == level and x["method"] == m), key=lambda x: int(x["seed"]))
        assert [int(x["seed"]) for x in r] == list(range(20))
        TOT[(level, m)] = [int(round(60 * float(x["mean_tardiness"]))) for x in r]
        MT[(level, m)] = np.array([t / 60 for t in TOT[(level, m)]])
n_checked = sum(int(x["checker_errors"]) == 0 for x in rows)


# ---------------------------------------------------------------- the browser's bootstrap (mulberry32), exactly
def mulberry32(seed):
    a = seed & 0xFFFFFFFF

    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r


def js_boot_paired(values, B=10_000, seed=0):
    """interactive/d02.js:bootMeans = jobshop-lib.js:bootstrapMean: mulberry32(seed + 12345), floor quantiles."""
    r, n = mulberry32(seed + 12345), len(values)
    means = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += values[math.floor(r() * n)]
        means.append(s / n)
    means.sort()
    return means[math.floor(0.025 * (B - 1))], means[math.floor(0.975 * (B - 1))]


def js_boot_unpaired(a, b, B=10_000, seed=0):
    """interactive/d02.js:bootUnpaired: resample the two methods' days independently."""
    r, n = mulberry32(seed + 54321), len(a)
    diffs = []
    for _ in range(B):
        sa = 0.0
        for _ in range(n):
            sa += a[math.floor(r() * n)]
        sb = 0.0
        for _ in range(n):
            sb += b[math.floor(r() * n)]
        diffs.append(sa / n - sb / n)
    diffs.sort()
    return diffs[math.floor(0.025 * (B - 1))], diffs[math.floor(0.975 * (B - 1))]


def unpaired_ci(a, b, B=10_000, seed=0):
    """Percentile interval for mean(a) - mean(b) when the two samples are treated as independent."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    da = rng.choice(a, size=(B, len(a))).mean(axis=1)
    db = rng.choice(b, size=(B, len(b))).mean(axis=1)
    lo, hi = np.quantile(da - db, [0.025, 0.975])
    return float(lo), float(hi)


# ================================================================ Part 1: the same days for every method
section("Part 1 (crn): one bad day for everybody, seed 7")
for level in LV:
    print(f"  {level}: day 7 mean tardiness", {m: round(MT[(level, m)][7], 1) for m in METHODS})
    for m in ("SST", "CP-SAT"):
        v = MT[(level, m)]
        print(f"    {m}: day 7 rank among the 20 days = {int((v > v[7]).sum()) + 1}; range over days "
              f"{v.min():.1f} to {v.max():.1f}")
for level in LV:
    worst = {m: int(np.argmax(MT[(level, m)])) for m in METHODS}
    print(f"  {level}: worst day of each method (seed):", worst)
print("  runs in runs.csv that passed the checker:", n_checked, "of", len(rows))
n_dqn_agents = len({x["method"] for x in rows if x["method"].startswith("DQN:")})
n_levels, n_seeds = len({x["level"] for x in rows}), len({x["seed"] for x in rows})
print(f"  runs.csv = ({len(METHODS)} methods + {n_dqn_agents} DQN agents) x {n_seeds} days x {n_levels} levels = "
      f"{(len(METHODS) + n_dqn_agents) * n_seeds * n_levels} rows (file: {len(rows)}); headline runs of the seven methods: "
      f"{len(METHODS) * n_seeds * n_levels}")
print("  headline means (min):", {lv: {m: round(MT[(lv, m)].mean(), 1) for m in METHODS} for lv in LV})

# ================================================================ Part 2: Var(X - Y) and the correlations
section("Part 2 (variance): the three-day example by hand, X = (10, 20, 60), Y = (14, 26, 68)")
hx, hy = np.array([10.0, 20.0, 60.0]), np.array([14.0, 26.0, 68.0])
print(f"  means {hx.mean():g} and {hy.mean():g}; s_X^2 = {hx.var(ddof=1):g}, s_Y^2 = {hy.var(ddof=1):g}, "
      f"Cov = {np.cov(hx, hy)[0, 1]:g}, r = {np.corrcoef(hx, hy)[0, 1]:.4f}")
print(f"  s_X^2 + s_Y^2 - 2 Cov = {hx.var(ddof=1) + hy.var(ddof=1) - 2 * np.cov(hx, hy)[0, 1]:g}; "
      f"differences {[float(v) for v in hx - hy]}, their variance {np.var(hx - hy, ddof=1):g}")
print(f"  squared deviations: X {[float(v) for v in (hx - hx.mean()) ** 2]}, Y {[float(v) for v in (hy - hy.mean()) ** 2]}")
import inspect  # noqa: E402
print(f"  (rho here is a correlation; the shop's load of B1 is make_scenario's default "
      f"load = {inspect.signature(make_scenario).parameters['load'].default:.2f})")

section("Part 2 (variance): across-seed correlations of mean tardiness, Pearson")
for level in LV:
    C = np.corrcoef(np.array([MT[(level, m)] for m in METHODS]))
    print(f"  {level}:")
    print("      " + " ".join(f"{m[:6]:>7}" for m in METHODS))
    for i, m in enumerate(METHODS):
        print(f"  {m[:6]:>6} " + " ".join(f"{C[i, j]:7.2f}" for j in range(len(METHODS))))
    off = C[np.triu_indices(len(METHODS), 1)]
    print(f"  {level}: the 21 pairwise correlations range from {off.min():.2f} to {off.max():.2f}, median {np.median(off):.2f}")
    rr = C[np.ix_(range(4), range(4))][np.triu_indices(4, 1)]
    print(f"  {level}: among the four rules, from {rr.min():.2f} to {rr.max():.2f}, mean {rr.mean():.2f}")

section("Part 2 (variance): the identity on CP-SAT minus SST, frequent breakdowns (the hero's default)")
X, Y = MT[(BD, "CP-SAT")], MT[(BD, "SST")]
D = X - Y
n = len(D)
vx, vy, cxy = X.var(ddof=1), Y.var(ddof=1), np.cov(X, Y)[0, 1]
r_xy = np.corrcoef(X, Y)[0, 1]
print(f"  sd CP-SAT {X.std(ddof=1):.2f}, sd SST {Y.std(ddof=1):.2f}, covariance {cxy:.1f}, correlation {r_xy:.3f}")
print(f"  Var X + Var Y = {vx:.1f} + {vy:.1f} = {vx + vy:.1f};  - 2 Cov = {-2 * cxy:.1f};  Var(X - Y) = {vx + vy - 2 * cxy:.1f}"
      f" (direct: {D.var(ddof=1):.1f}); sd of the daily difference {D.std(ddof=1):.2f}")
c_fac = 2 * X.std(ddof=1) * Y.std(ddof=1) / (vx + vy)
print(f"  c = 2 sx sy / (sx^2 + sy^2) = {c_fac:.3f}; 1 - r c = {1 - r_xy * c_fac:.3f}; lower bound 1 - r = {1 - r_xy:.3f}")
print(f"  variance ratio Var(X-Y)/(Var X + Var Y) = {D.var(ddof=1) / (vx + vy):.3f}; sd ratio {math.sqrt(D.var(ddof=1) / (vx + vy)):.3f}")
se_p, se_u = D.std(ddof=1) / math.sqrt(n), math.sqrt((vx + vy) / n)
print(f"  standard error of the mean difference: paired {se_p:.2f}, unpaired {se_u:.2f}; "
      f"normal half-widths 1.96 x SE: {1.96 * se_p:.2f} and {1.96 * se_u:.2f}")
print(f"  mean difference {D.mean():.2f}; t(19, 0.975) = 2.093: t-interval [{D.mean() - 2.093 * se_p:.1f}, {D.mean() + 2.093 * se_p:.1f}]")
print(f"  repository bootstrap_ci, paired: {ci(D)}; unpaired (independent resamples): "
      f"[{unpaired_ci(X, Y)[0]:.1f}, {unpaired_ci(X, Y)[1]:.1f}]")
wp = np.diff(bootstrap_ci(D))[0]
wu = np.diff(unpaired_ci(X, Y))[0]
print(f"  widths: paired {wp:.1f}, unpaired {wu:.1f}, ratio {wu / wp:.1f}; seeds needed by the unpaired design for the "
      f"paired width: about {(wu / wp) ** 2:.0f} x 20 = {20 * (wu / wp) ** 2:.0f}")
print(f"  days better / equal / worse: {(D < 0).sum()} / {(D == 0).sum()} / {(D > 0).sum()}")

section("Part 2 (variance): paired against unpaired for every comparison the README reports")
for level in LV:
    best = min(RULES4, key=lambda m: MT[(level, m)].mean())
    for a, b in (("CP-SAT", best), ("Greedy replan", best), ("CP-SAT", "Greedy replan"), ("DQN", best)):
        A, Bv = MT[(level, a)], MT[(level, b)]
        d = A - Bv
        lo, hi = bootstrap_ci(d)
        ulo, uhi = unpaired_ci(A, Bv)
        print(f"  {level:>19} {a} - {b}: {d.mean():+.1f} paired [{lo:+.1f}, {hi:+.1f}] width {hi - lo:.1f}; "
              f"unpaired [{ulo:+.1f}, {uhi:+.1f}] width {uhi - ulo:.1f}; r = {np.corrcoef(A, Bv)[0, 1]:.2f}; "
              f"better/equal/worse {(d < 0).sum()}/{(d == 0).sum()}/{(d > 0).sum()}")

section("Hero: the browser's own resamples (mulberry32), default pair CP-SAT - SST, frequent breakdowns")
Xl, Yl = [t / 60 for t in TOT[(BD, "CP-SAT")]], [t / 60 for t in TOT[(BD, "SST")]]
Dl = [x - y for x, y in zip(Xl, Yl)]
for nn in (5, 10, 15, 20):
    jp = js_boot_paired(Dl[:nn])
    ju = js_boot_unpaired(Xl[:nn], Yl[:nn])
    rp = bootstrap_ci(D[:nn])
    ru = unpaired_ci(X[:nn], Y[:nn])
    print(f"  first {nn:2d} days: mean {np.mean(Dl[:nn]):+.1f}; browser paired [{jp[0]:.1f}, {jp[1]:.1f}] width {jp[1] - jp[0]:.1f}, "
          f"unpaired [{ju[0]:.1f}, {ju[1]:.1f}] width {ju[1] - ju[0]:.1f};  numpy paired [{rp[0]:.1f}, {rp[1]:.1f}] "
          f"width {rp[1] - rp[0]:.1f}, unpaired width {ru[1] - ru[0]:.1f}; r = {np.corrcoef(X[:nn], Y[:nn])[0, 1]:.2f}")

# ================================================================ Part 3: the bootstrap
section("Part 3 (bootstrap): the hand-sized example, three days with differences -3, -6, -15")
h = np.array([-3.0, -6.0, -15.0])
allm = sorted(sum(c) / 3 for c in itertools.product(h, repeat=3))
vals, counts = np.unique(np.round(allm, 4), return_counts=True)
print("  27 equally likely resamples; distinct means and how many resamples give each:")
print("   ", ", ".join(f"{v:g} (x{c})" for v, c in zip(vals, counts)))
plug = ((h - h.mean()) ** 2).mean()
print(f"  squared deviations {[float(v) for v in (h - h.mean()) ** 2]}")
print(f"  sample mean {h.mean():g}; plug-in variance (1/n) sum (x - mean)^2 = {plug:g}; s^2 (1/(n-1)) = {h.var(ddof=1):g}")
print(f"  exact bootstrap variance of the resample mean = {np.var(allm):.4f} = plug-in / n = {plug / 3:.4f}; "
      f"bootstrap SE {math.sqrt(plug / 3):.2f}; usual SE s/sqrt(n) = {h.std(ddof=1) / math.sqrt(3):.2f}")
print(f"  P(resample mean = -15) = 1/27 = {1 / 27:.3f} > 0.025, so the 95 % percentile interval is the data range "
      f"[{min(allm):g}, {max(allm):g}]")
print("  the repository's bootstrap_ci on these three values:", ci(h))

section("Part 3 (bootstrap) and its Going further block (what B buys): Monte Carlo error, 100 bootstrap seeds, B = 10,000")
sig_hat = math.sqrt(((D - D.mean()) ** 2).mean())
print(f"  the 20 days of CP-SAT - SST: plug-in sd {sig_hat:.2f}; exact bootstrap SE sigma_hat / sqrt(20) = {sig_hat / math.sqrt(20):.2f} "
      f"(against s / sqrt(n) = {D.std(ddof=1) / math.sqrt(20):.2f}); 20^20 = {20 ** 20:.2e} ordered resamples")
ends = np.array([bootstrap_ci(D, seed=s) for s in range(100)])
print(f"  CP-SAT - SST, frequent breakdowns: lower end {ends[:, 0].min():.2f} to {ends[:, 0].max():.2f} "
      f"(sd {ends[:, 0].std():.3f}); upper end {ends[:, 1].min():.2f} to {ends[:, 1].max():.2f} (sd {ends[:, 1].std():.3f})")
ends1 = np.array([bootstrap_ci(D, seed=s, resamples=1000) for s in range(100)])
print(f"  with B = 1,000 instead: sd of the ends {ends1[:, 0].std():.3f} and {ends1[:, 1].std():.3f}")
endsu = np.array([unpaired_ci(X, Y, seed=s) for s in range(100)])
print(f"  unpaired interval, B = 10,000, 100 seeds: lower end {endsu[:, 0].min():.2f} to {endsu[:, 0].max():.2f} "
      f"(sd {endsu[:, 0].std():.3f}); upper end {endsu[:, 1].min():.2f} to {endsu[:, 1].max():.2f} (sd {endsu[:, 1].std():.3f})")

section("Part 3 (bootstrap): basic (pivot) interval against percentile interval, CP-SAT - SST, frequent breakdowns")
qlo, qhi = bootstrap_ci(D)
print(f"  mean {D.mean():.2f}; percentile [q.025, q.975] = [{qlo:.1f}, {qhi:.1f}]; "
      f"basic [2 mean - q.975, 2 mean - q.025] = [{2 * D.mean() - qhi:.1f}, {2 * D.mean() - qlo:.1f}]; "
      f"t-interval [{D.mean() - 2.093 * se_p:.1f}, {D.mean() + 2.093 * se_p:.1f}]")
print(f"  distances of the percentile ends from the mean: {D.mean() - qlo:.2f} below, {qhi - D.mean():.2f} above "
      f"(unequal: the bootstrap distribution is skewed)")


def t_cdf(x, nu):
    """Student t cumulative distribution by trapezoid integration of the density (no scipy)."""
    xs = np.linspace(-60, x, 400001)
    f = math.gamma((nu + 1) / 2) / (math.sqrt(nu * math.pi) * math.gamma(nu / 2)) * (1 + xs ** 2 / nu) ** (-(nu + 1) / 2)
    return float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(xs)))


print(f"  t(19) quantile check: P(t19 <= 2.093) = {t_cdf(2.093, 19):.4f}")
cov_norm = 2 * t_cdf(1.96 * math.sqrt(19 / 20), 19) - 1
print(f"  normal data, n = 20: the interval mean +- 1.96 sigma_hat / sqrt(n) covers P(|t19| < 1.96 sqrt(19/20)) = {100 * cov_norm:.1f} %")
rng_n = np.random.default_rng(5)
hits_n = 0
for k in range(1000):
    z = rng_n.standard_normal(20)
    lo_, hi_ = bootstrap_ci(z, resamples=2000, seed=k)
    hits_n += lo_ <= 0 <= hi_
print(f"  simulated: percentile interval on 1000 normal samples of 20 (B = 2,000) covers {hits_n / 10:.1f} %")

section("Part 3 (bootstrap): sample sd of CP-SAT - SST over the first n days, frequent breakdowns")
print("  daily differences, days 0-9:", [round(float(v), 1) for v in D[:10]])
print("  " + ", ".join(f"n={nn}: {D[:nn].std(ddof=1):.2f}" for nn in range(3, 21)))

section("Part 3 (bootstrap): how the paired interval shrinks with the number of days (numpy, first n seeds)")
for nn in (5, 10, 15, 20):
    lo, hi = bootstrap_ci(D[:nn])
    ulo, uhi = unpaired_ci(X[:nn], Y[:nn])
    print(f"  n = {nn:2d}: paired [{lo:.1f}, {hi:.1f}] width {hi - lo:.1f}; unpaired width {uhi - ulo:.1f}; "
          f"sd of the difference {D[:nn].std(ddof=1):.2f}")
sdd_dqn = (MT[(BD, "DQN")] - MT[(BD, "SST")]).std(ddof=1)
mdd = (MT[(BD, "DQN")] - MT[(BD, "SST")]).mean()
print(f"  DQN - SST, frequent breakdowns: mean {mdd:+.2f}, sd of the daily difference {sdd_dqn:.2f}")
print(f"    days for a half-width of 1.3 min: (1.96 sd / 1.3)^2 = {(1.96 * sdd_dqn / 1.3) ** 2:.0f}; "
      f"for 80 % power at a true +1.3: ((1.96 + 0.84) sd / 1.3)^2 = {((1.96 + 0.84) * sdd_dqn / 1.3) ** 2:.0f}")

# ================================================================ the pool of fresh days (rules only)
section("Pool: the four rules on 500 fresh days (seeds 200000-200499, used nowhere else), both levels")
ws = make_workshop()
POOL = {}
t = time.perf_counter()
for level in LV:
    vals_l = {m: [] for m in RULES4}
    for s in range(200_000, 200_500):
        sc = make_scenario(ws, s, mtbf=LEVELS[level])
        for m in RULES4:
            log = Shop(ws, sc, RuleController(m)).run()
            vals_l[m].append(run_metrics(ws, sc, log)["mean_tardiness"])
    POOL[level] = {m: np.array(v) for m, v in vals_l.items()}
print(f"  4000 runs in {time.perf_counter() - t:.0f} s")
for level in LV:
    P = POOL[level]
    print(f"  {level}: pool means " + ", ".join(f"{m} {P[m].mean():.2f} (SE {P[m].std(ddof=1) / math.sqrt(500):.2f})" for m in RULES4))
    C = np.corrcoef(np.array([P[m] for m in RULES4]))
    print(f"  {level}: pool correlations among the rules from {C[np.triu_indices(4, 1)].min():.2f} to "
          f"{C[np.triu_indices(4, 1)].max():.2f}; SST-EDD {C[2, 3]:.2f}")

section("Part 2, Going further (truly independent days): SST on 20 days, EDD on 20 other days, frequent breakdowns")
P = POOL[BD]
a_same, b_same = P["SST"][:20], P["EDD"][:20]
b_other = P["EDD"][20:40]
lo, hi = bootstrap_ci(a_same - b_same)
ulo, uhi = unpaired_ci(a_same, b_other)
print(f"  common days: SST - EDD {np.mean(a_same - b_same):+.1f} [{lo:+.1f}, {hi:+.1f}] width {hi - lo:.1f}")
print(f"  independent days: {a_same.mean() - b_other.mean():+.1f} [{ulo:+.1f}, {uhi:+.1f}] width {uhi - ulo:.1f}")
print(f"  pool truth: SST - EDD = {P['SST'].mean() - P['EDD'].mean():+.2f}")
# the width over many such experiments
rng = np.random.default_rng(1)
wp_, wu_ = [], []
for _ in range(200):
    idx = rng.permutation(500)
    s1, s2 = idx[:20], idx[20:40]
    l1, h1 = bootstrap_ci(P["SST"][s1] - P["EDD"][s1], resamples=2000)
    l2, h2 = unpaired_ci(P["SST"][s1], P["EDD"][s2], B=2000)
    wp_.append(h1 - l1); wu_.append(h2 - l2)
print(f"  over 200 such experiments (B = 2,000): mean width common days {np.mean(wp_):.1f}, independent days {np.mean(wu_):.1f}")

section("Part 3, Going further (coverage): the 95 % percentile interval on 1000 samples of 20 pool days (B = 2,000)")
for level in LV:
    P = POOL[level]
    for label, v in ((f"SST mean", P["SST"]), ("SST - EDD paired", P["SST"] - P["EDD"])):
        truth = v.mean()
        rng = np.random.default_rng(2)
        hits = 0
        for k in range(1000):
            sub = v[rng.choice(500, size=20, replace=False)]
            lo, hi = bootstrap_ci(sub, resamples=2000, seed=k)
            hits += lo <= truth <= hi
        skew = float(((v - v.mean()) ** 3).mean() / v.std() ** 3)
        print(f"  {level:>19}, {label}: coverage {hits / 10:.1f} %  (pool skewness {skew:.2f})")

# ================================================================ Part 4: training / validation / test
section("Part 4 (separation): the five DQN training seeds on the test days")
import json  # noqa: E402

from jobshop import dqn as _dqn  # noqa: E402

cfg = _dqn.DQNConfig()
print(f"  test seeds 0-19 (scripts/evaluate.py); validation seeds {_dqn.VALIDATION_SEEDS[0]}-{_dqn.VALIDATION_SEEDS[-1]}; "
      f"training seeds {_dqn.TRAIN_SEED_BASE} + 1000 x training seed + episode, episodes 1-{cfg.episodes}, "
      f"so {_dqn.TRAIN_SEED_BASE + 1} to {_dqn.TRAIN_SEED_BASE + 1000 * 4 + cfg.episodes}; validation every {cfg.validate_every} episodes")
training = json.loads((REPO / "models" / "training.json").read_text())
# the days each agent trains on, exactly as dqn.train draws them: (scenario seed, breakdown level)
_lv = list(LEVELS.values())
days = {k: {(_dqn.TRAIN_SEED_BASE + 1_000 * k + e, _lv[e % len(_lv)]) for e in range(1, cfg.episodes + 1)}
        for k in (run["seed"] for run in training["runs"])}
ks = sorted(days)
print(f"  {len(ks)} agents x {cfg.episodes} episodes = {sum(len(v) for v in days.values()):,} episodes; "
      f"distinct (seed, level) days {len(set().union(*days.values())):,}; days shared by consecutive agents: "
      + ", ".join(f"{a}&{b} {len(days[a] & days[b])}" for a, b in zip(ks, ks[1:]))
      + f"; by agents two apart: {max(len(days[a] & days[a + 2]) for a in ks[:-2])}")
print("  selected seed (best validation):", training["selected_seed"])
for run in training["runs"]:
    k = run["seed"]
    print(f"  training seed {k}: validation {run['best_validation']:.2f}, test calm {MT[(CALM, f'DQN:{k}')].mean():.1f}, "
          f"test breakdowns {MT[(BD, f'DQN:{k}')].mean():.1f}")
for level in LV:
    test = {k: MT[(level, f"DQN:{k}")].mean() for k in range(5)}
    kb = min(test, key=test.get)
    print(f"  {level}: chosen on the test would be seed {kb} with {test[kb]:.1f}; chosen on validation: seed "
          f"{training['selected_seed']} with {test[training['selected_seed']]:.1f}; spread {min(test.values()):.1f} "
          f"to {max(test.values()):.1f}")
for level in LV:
    best = min(RULES4, key=lambda m: MT[(level, m)].mean())
    for k in range(5):
        dk = MT[(level, f"DQN:{k}")] - MT[(level, best)]
        print(f"  {level}: DQN:{k} - {best} paired {dk.mean():+.2f} {ci(dk)}")
print("  the 10 pairwise differences among the five agents, paired intervals (* = excludes zero):")
n_excl = 0
for level in LV:
    for i, j in itertools.combinations(range(5), 2):
        dij = MT[(level, f"DQN:{i}")] - MT[(level, f"DQN:{j}")]
        lo, hi = bootstrap_ci(dij)
        flag = "*" if hi < 0 or lo > 0 else " "
        n_excl += flag == "*"
        print(f"    {level:>19} DQN:{i} - DQN:{j}: {dij.mean():+.2f} [{lo:+.1f}, {hi:+.1f}] {flag}")
print(f"  {n_excl} of 20 pairwise agent intervals exclude zero")

# ================================================================ Part 5: the winner's curse
section("Part 5 (winners): expected maximum of K independent standard normals, e_K")


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def e_max(K):
    xs = np.linspace(-9, 9, 36001)
    f = xs * K * np.exp(-xs ** 2 / 2) / math.sqrt(2 * math.pi) * np.array([Phi(x) for x in xs]) ** (K - 1)
    return float(np.trapezoid(f, xs) if hasattr(np, "trapezoid") else np.trapz(f, xs))


EK = {K: e_max(K) for K in (2, 3, 4, 5, 7, 10, 20)}
print("  " + ", ".join(f"e_{K} = {v:.3f}" for K, v in EK.items()), f"; 1/sqrt(pi) = {1 / math.sqrt(math.pi):.3f}")
print("  the large-K asymptote sqrt(2 ln K): " + ", ".join(f"K={K}: {math.sqrt(2 * math.log(K)):.2f}" for K in (4, 10, 20)))
sig = MT[(BD, "SST")].std(ddof=1)
rr = np.corrcoef(np.array([MT[(BD, m)] for m in RULES4]))[np.triu_indices(4, 1)].mean()
print(f"  demo defaults: sigma = sd of SST across the test days, frequent breakdowns = {sig:.1f} min; rho = 0.90; K = 4; n = 20")
for K, rho, nn in ((4, 0.0, 20), (4, 0.9, 20), (10, 0.0, 20), (10, 0.9, 20), (4, 0.9, 5), (4, 0.0, 5)):
    print(f"    K = {K:2d}, rho = {rho:.2f}, n = {nn:2d}: expected optimism sigma sqrt(1 - rho) e_K / sqrt(n) = "
          f"{sig * math.sqrt(1 - rho) * EK[K] / math.sqrt(nn):.2f} min")
# the same model simulated, as the demo does
rng = np.random.default_rng(3)
K, rho, nn, R = 4, 0.9, 20, 20000
Z0 = rng.standard_normal((R, nn, 1))
E = rng.standard_normal((R, nn, K))
Xs = sig * (math.sqrt(rho) * Z0 + math.sqrt(1 - rho) * E)
print(f"    simulated (R = {R}): mean of the winner's 20-day mean = {Xs.mean(axis=1).min(axis=1).mean():.2f} "
      f"(the true mean is 0 for every method)")

section("Part 5, Going further (the four rules on the pool): 2000 random 20-day test sets")
for level in LV:
    P = POOL[level]
    M = np.array([P[m] for m in RULES4])           # 4 x 500
    truth = M.mean(axis=1)
    rng = np.random.default_rng(4)
    opt, right, holdout = [], 0, []
    for _ in range(2000):
        idx = rng.permutation(500)
        test, val = idx[:20], idx[20:30]
        means = M[:, test].mean(axis=1)
        k = int(np.argmin(means))
        opt.append(means[k] - truth[k])
        right += k == int(np.argmin(truth))
        kv = int(np.argmin(M[:, val].mean(axis=1)))       # chosen on 10 separate days, reported on the 20 test days
        holdout.append(M[kv, test].mean() - truth[kv])
    print(f"  {level}: true best {RULES4[int(np.argmin(truth))]} ({truth.min():.2f}); picked it in {right / 20:.1f} % of the test sets; "
          f"mean optimism of 'best on the test' {np.mean(opt):+.2f} min ({100 * np.mean(opt) / truth.min():+.0f} % of the true best); "
          f"chosen on 10 other days, reported on the test: {np.mean(holdout):+.2f} min")

# ================================================================ Part 6: reporting
section("Part 6 (reporting): CP-SAT against SST, frequent breakdowns, the full report")
A, Bv = MT[(BD, "CP-SAT")], MT[(BD, "SST")]
d = A - Bv
print(f"  CP-SAT {A.mean():.1f} {ci(A)}, SST {Bv.mean():.1f} {ci(Bv)}; difference {d.mean():+.1f} {ci(d)}")
print(f"  relative reduction 1 - {A.mean():.1f}/{Bv.mean():.1f} = {100 * (1 - A.mean() / Bv.mean()):.0f} %; "
      f"standardised effect d_z = mean/sd of the difference = {d.mean() / d.std(ddof=1):.2f}")
print(f"  days better {(d < 0).sum()} of 20; smallest gain {-d.max():.1f} min, largest {-d.min():.1f} min; median difference {np.median(d):+.1f} (median gain {-np.median(d):.1f})")
print(f"  probability that 20 of 20 days favour one side if each day were a coin toss: 2 x 0.5^20 = {2 * 0.5 ** 20:.1e}")
dd = MT[(BD, "DQN")] - MT[(BD, "SST")]
print(f"  DQN - SST: {dd.mean():+.1f} {ci(dd)}, d_z = {dd.mean() / dd.std(ddof=1):.2f}, better on {(dd < 0).sum()} of 20")
dc = MT[(CALM, "CP-SAT")] - MT[(CALM, "EDD")]
print(f"  calm, CP-SAT - EDD: {dc.mean():+.1f} {ci(dc)}, d_z = {dc.mean() / dc.std(ddof=1):.2f}, better on {(dc < 0).sum()} of 20;"
      f" relative reduction {100 * (1 - MT[(CALM, 'CP-SAT')].mean() / MT[(CALM, 'EDD')].mean()):.0f} %")
pairs = len(METHODS) * (len(METHODS) - 1) // 2
print(f"  pairs among 7 methods: {pairs}; at 95 % each, expected false alarms if no pair differed: 0.05 x {pairs} = {0.05 * pairs:.2f} per level")

# ================================================================ Part 7: the checker
section("Part 7 (checker): the small instance on the paint machine of A1 (notation.md #paint)")
# set up for blue, colour change 10 min; A blue 20 min, then B red 10 min
a_end = 20
b_end_skip, b_end_true = a_end + 10, a_end + 10 + 10
print(f"  A (blue) 0 to {a_end}; B (red) with the 10-minute changeover skipped ends at {b_end_skip}, "
      f"with it at {b_end_true}")

section("Part 7 (checker): a simulator that forgets changeovers, the four rules on the 20 test days, frequent breakdowns")
ws_bug = replace(ws, setup=tuple(tuple(tuple(0 for _ in r) for r in mat) for mat in ws.setup))
for m in RULES4:
    good, bad, errs, bad_runs = [], [], 0, 0
    for s in range(20):
        sc = make_scenario(ws, s, mtbf=LEVELS[BD])
        log = Shop(ws, sc, RuleController(m)).run()
        good.append(run_metrics(ws, sc, log)["mean_tardiness"])
        log_b = Shop(ws_bug, sc, RuleController(m)).run()
        bad.append(run_metrics(ws, sc, log_b)["mean_tardiness"])
        e = check_schedule(ws, sc, log_b)
        errs += len(e); bad_runs += bool(e)
    print(f"  {m}: real simulator {np.mean(good):.1f} min, bugged simulator {np.mean(bad):.1f} min; checker: {bad_runs} of 20 runs "
          f"rejected, {errs} violations, e.g. '{e[0][:90]}...'")
print("  (the real simulator's values equal runs.csv:", all(abs(np.mean(good) - MT[(BD, 'SST')].mean()) < 1e-9 for _ in [0]), ")")

section("Part 7 (checker): the injected-violation tests of tests/test_checker.py")
try:
    sys.path.insert(0, str(REPO / "tests"))
    import copy

    import test_checker as tc
    from jobshop.shop import simulate
    names = [k for k in dir(tc) if k.startswith("test_")]
    base = simulate(tc.WS, tc.SCENARIO, RuleController("EDD"))
    print(f"  valid EDD log (seed 42, 40 jobs, MTBF 120): {len(base.segments)} segments, {len(base.legs)} AGV legs, "
          f"{len(check_schedule(tc.WS, tc.SCENARIO, base))} violations")
    caught = 0
    for k in sorted(names, key=lambda k: getattr(tc, k).__code__.co_firstlineno):
        lg = copy.deepcopy(base)
        try:
            getattr(tc, k)(lg)
            caught += 1
            first = check_schedule(tc.WS, tc.SCENARIO, lg)
            print(f"    {k}: caught, {len(first)} violation(s): {first[0][:80]}")
        except AssertionError:
            print(f"    {k}: NOT caught")
    print(f"  {caught} of {len(names)} injected violations caught")
except ImportError as exc:
    print("  tests/ not found next to jobshop/:", exc)

# ================================================================ Exercises
section("Exercise 1: pairing for DQN - SST, frequent breakdowns")
xa, ya = MT[(BD, "DQN")], MT[(BD, "SST")]
sxa, sya, ra = xa.std(ddof=1), ya.std(ddof=1), np.corrcoef(xa, ya)[0, 1]
vpa = sxa ** 2 + sya ** 2 - 2 * ra * sxa * sya
print(f"  s_X = {sxa:.2f}, s_Y = {sya:.2f}, r = {ra:.3f}; s_X^2 + s_Y^2 = {sxa ** 2 + sya ** 2:.1f}; "
      f"paired variance {vpa:.1f} (sd {math.sqrt(vpa):.2f})")
print(f"  standard errors over 20 days: paired {math.sqrt(vpa / 20):.2f}, unpaired {math.sqrt((sxa ** 2 + sya ** 2) / 20):.2f}")
print(f"  independent days per method to match 20 paired days: 20 (s_X^2 + s_Y^2) / Var(X - Y) = "
      f"{20 * (sxa ** 2 + sya ** 2) / vpa:.0f}")
print(f"  by hand with the rounded values: s_X^2 = {26.01 ** 2:.1f}, s_Y^2 = {24.02 ** 2:.1f}, "
      f"2 r s_X s_Y = {2 * 0.945 * 26.01 * 24.02:.0f}, paired variance about {26.01 ** 2 + 24.02 ** 2 - 2 * 0.945 * 26.01 * 24.02:.0f}")
print(f"  if r were -0.5 instead: -2 r s_X s_Y = s_X s_Y = {sxa * sya:.1f}; paired variance {sxa ** 2 + sya ** 2 + sxa * sya:.1f}")

section("Exercise 2: the 256 resamples of four days (-2, -4, -6, -20)")
h4 = np.array([-2.0, -4.0, -6.0, -20.0])
m4 = np.array(sorted(sum(c) / 4 for c in itertools.product(h4, repeat=4)))
v4, c4 = np.unique(np.round(m4, 4), return_counts=True)
cum = np.cumsum(c4) / 256
print("  distinct means (count, cumulative probability):")
print("   ", ", ".join(f"{v:g} ({c}, {p:.4f})" for v, c, p in zip(v4, c4, cum)))
q_lo = v4[np.argmax(cum >= 0.025)]
q_hi = v4[np.argmax(cum >= 0.975)]
print(f"  mean {h4.mean():g}; P(all four = -20) = 1/256 = {1 / 256:.4f}; exact 2.5 % quantile (smallest v with "
      f"P(mean <= v) >= 0.025) = {q_lo:g}, 97.5 % quantile = {q_hi:g}; data range [{h4.min():g}, {h4.max():g}]")
print(f"  squared deviations {[float(v) for v in (h4 - h4.mean()) ** 2]}")
print(f"  plug-in variance {((h4 - h4.mean()) ** 2).mean():g}, exact bootstrap variance of the mean {m4.var():g}")
print("  repository bootstrap_ci on the four values:", ci(h4))

section("Exercise 3: two candidates, e_2 and the chance that the worse rule wins (EDD vs SST, pool, breakdowns)")
print(f"  e_2 = {EK[2]:.3f}, 1/sqrt(pi) = {1 / math.sqrt(math.pi):.3f}")
P = POOL[BD]
d3 = P["EDD"] - P["SST"]                       # worse minus better, mean delta > 0
dl3, sd3 = d3.mean(), d3.std(ddof=1)
sE3, sS3 = P["EDD"].std(ddof=1), P["SST"].std(ddof=1)
su3 = math.sqrt(sE3 ** 2 + sS3 ** 2)
zp, zu = dl3 * math.sqrt(20) / sd3, dl3 * math.sqrt(20) / su3
print(f"  delta = EDD - SST = {dl3:.2f}; s_Delta = {sd3:.2f}; spreads EDD {sE3:.2f}, SST {sS3:.2f}; "
      f"sqrt of the sum of squares {su3:.2f}")
print(f"  n = 20 common days: z = {zp:.2f}, P(EDD wins) = Phi(-z) = {Phi(-zp):.3f}; "
      f"independent days: z = {zu:.2f}, P = {Phi(-zu):.3f}")
rng = np.random.default_rng(5)
wp3 = wu3 = 0
R3 = 10_000
for _ in range(R3):
    idx = rng.permutation(500)
    a3, b3 = idx[:20], idx[20:40]
    wp3 += P["EDD"][a3].mean() < P["SST"][a3].mean()
    wu3 += P["EDD"][b3].mean() < P["SST"][a3].mean()
print(f"  on {R3:,} random 20-day sets of the pool, EDD wins {100 * wp3 / R3:.1f} % (common days), "
      f"{100 * wu3 / R3:.1f} % (EDD on 20 other days)")

section("Exercise 5: the min-flow invariant on seed 3, EDD, a valid run and a shifted job")
from jobshop.shop import simulate  # noqa: E402

ws6 = make_workshop()
sc6 = make_scenario(ws6, 3, mtbf=120.0)


def check_min_flow(ws, sc, log):
    end = {}
    for s in log.segments:
        end[s.job] = max(end.get(s.job, s.end), s.end)
    return [f"job {j.idx} done at {end[j.idx]}, needs at least {j.arrival + ws.min_remaining_work(j.family, 0)}"
            for j in sc.jobs if end[j.idx] < j.arrival + ws.min_remaining_work(j.family, 0)]


log6 = simulate(ws6, sc6, RuleController("EDD"))
print("  valid run:", check_min_flow(ws6, sc6, log6))
log6.segments[:] = [replace(s, start=s.start - 1000, end=s.end - 1000) if s.job == 10 else s for s in log6.segments]
print("  job 10 shifted:", check_min_flow(ws6, sc6, log6)[:1])

print(f"\ntotal time {time.perf_counter() - T0:.0f} s")
