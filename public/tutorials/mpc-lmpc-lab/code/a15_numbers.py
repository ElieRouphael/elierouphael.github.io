"""Reproduce every number quoted on chapter A15 of "Advanced MPC and Learning MPC",
"Disjunctive constraints and Big-M" (a15-disjunctive-constraints-and-big-m.html).

Usage:  python a15_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
The system is lab B1's longitudinal point mass (explicit Euler, T_s = 1 s), not the canonical cart.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)   # the infeasible pass branch makes the solver overflow before it reports failure

from advanced_mpc_lmpc.collision import (choose_collision_strategy, conflict_bounds, solve_collision_strategy, strategy_grid,
                                         terminal_position_extrema_no_speed_limits, vehicle_matrices)

np.set_printoptions(precision=4, suppress=True)


def section(title):
    print(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
section("Part 1: the conflict interval")
d_max, d_min = conflict_bounds(d_bike=100.0, l_car=4.5, w_bike=8.0)
print(f"yield side d <= d_max = {d_max}; pass side d >= d_min = {d_min}; forbidden gap {d_min - d_max} = l_car + w_bike")
print("two safe points", d_max - 1, "and", d_min + 1, "have the midpoint", 0.5 * (d_max - 1 + d_min + 1), "= the centre of the crossing: not safe")

# --------------------------------------------------------------------------------------------------------------
section("Part 3: the smallest valid Big-M for a position envelope [d_low, d_high]")
for d_low, d_high in ((0.0, 300.0), (0.0, 150.0), (50.0, 200.0)):
    M = max(d_high - d_max, d_min - d_low)
    print(f"envelope [{d_low}, {d_high}]: M >= max({d_high} - {d_max}, {d_min} - {d_low}) = {M}")
print("the round M = 5000 used by the lab is valid, and 24 times larger than needed for [0, 300]")

section("Part 3: what the LP relaxation sees (s continuous in [0, 1])")
M = 206.25
for s in (0.0, 0.25, 0.5, 0.75, 1.0):
    lo, hi = d_min - M * s, d_max + M * (1 - s)
    print(f"s = {s:4.2f}: relaxed constraints d >= {lo:8.3f} and d <= {hi:8.3f} -> allowed positions [{max(lo, 0):.3f}, {min(hi, 300):.3f}]")
print("at s = 0.5 the relaxed set contains the whole crossing: the relaxation's bound on the cost says nothing about avoidance")

# --------------------------------------------------------------------------------------------------------------
section("Part 4: enumerating the two branches from the nominal state (60, 5), N = 4, mu = 0.10")
x0 = np.array([60.0, 5.0])
for strategy in ("yield", "pass"):
    sol = solve_collision_strategy(x0, strategy, N=4, mu=0.10)
    print(f"{strategy:5s}: feasible {sol.success}, cost {sol.objective:.2f}" + (f", final state {sol.X[-1].round(3).tolist()}, largest violation {sol.max_violation:.1e}" if sol.success else f" ({sol.message})"))
best, sols = choose_collision_strategy(x0, N=4, mu=0.10)
print("chosen:", best.strategy, "with cost", round(best.objective, 2))
A, B = vehicle_matrices(1.0, "euler")
lo, hi = terminal_position_extrema_no_speed_limits(x0, N=4, Ts=1.0, mu=0.10)
print(f"closed-form reachable terminal positions under Euler: [{lo:.3f}, {hi:.3f}]; d_min = {d_min} is out of reach, d_max = {d_max} is not binding")

section("Part 4: a state where both branches are feasible, mu = 0.15, x0 = (30, 17)")
x1 = np.array([30.0, 17.0])
for Ru in (0.1, 1.0, 1000.0):
    best, sols = choose_collision_strategy(x1, N=4, Ts=1.0, mu=0.15, v_max=32.0, v_ref=20.0, Qv=3.0, Ru=Ru, Pv=3.0,
                                           discretization="euler", yield_boundary="geometric")
    print(f"R = {Ru:6.1f}: yield cost {sols[0].objective:8.2f} (feasible {sols[0].success}), pass cost {sols[1].objective:8.2f} (feasible {sols[1].success}) -> {best.strategy}")

section("Part 5: why one binary per manoeuvre: a per-step choice would drive through the crossing")
once = solve_collision_strategy(np.array([0.0, 19.0]), "yield", N=8, conflict_steps=(4,), mu=0.10)
window = solve_collision_strategy(np.array([0.0, 19.0]), "yield", N=8, conflict_steps=(4, 5, 6), mu=0.10)
print("from (0, 19): yield enforced only at step 4 is", "feasible" if once.success else "infeasible", "with d_4, d_5, d_6 =", once.X[4:7, 0].round(1).tolist() if once.success else "-",
      "; enforced over steps 4-6 it is", "feasible" if window.success else "infeasible")

# --------------------------------------------------------------------------------------------------------------
section("Exercises")
print("1. a_j appears in d_N exactly N - 1 - j times; the last acceleration never moves d_N under Euler")
print("2. M for [0, 300]:", max(300 - d_max, d_min - 0.0), "; for [0, 1000]:", max(1000 - d_max, d_min - 0.0))
print("3. three binaries, eight combinations; with m manoeuvres 2^m branch QPs:", [2 ** m for m in (1, 3, 5, 10, 20)])
count, _ = strategy_grid(np.arange(0, 100, 5), np.arange(0, 33), N=4, mu=0.10)
print("5. build it: on the 660-state grid at mu = 0.10,", int((count == 0).sum()), "states have no feasible branch")
sol = solve_collision_strategy(np.array([85.0, 3.0]), "yield", N=4, mu=0.10)
print("   the state (85, 3): yield feasible", sol.success, "ending exactly on the bound, final state", sol.X[-1].round(3).tolist())
print("6. two obstacles in a row with 'before or after' each: 4 combinations, but 'after the first and before the second' needs the gap between them to fit the car")
