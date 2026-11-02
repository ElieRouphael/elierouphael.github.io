"""Reproduce every number quoted on lab B1 of "Advanced MPC and Learning MPC",
"Vehicle–bike collision avoidance" (b01-vehicle-bike-collision-avoidance.html).

Usage:  python b01_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
Every QP is solved by the package's interior-point solver; the whole script takes about a minute.
"""
from __future__ import annotations

import time

import numpy as np

from advanced_mpc_lmpc.collision import (choose_collision_strategy, conflict_bounds, solve_collision_strategy,
                                         strategy_grid, terminal_position_extrema_no_speed_limits, vehicle_matrices)

np.set_printoptions(precision=4, suppress=True)
G = 9.81
D_VALUES = np.arange(0, 100, 5)        # 20 positions
V_VALUES = np.arange(0, 33, 1)         # 33 speeds: 660 initial states


def section(title):
    print(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
section("Part 1: the scene")
d_max, d_min = conflict_bounds()
print(f"yield bound d_max = {d_max}, pass bound d_min = {d_min}, forbidden gap {d_min - d_max} m")
A, B = vehicle_matrices(1.0, "euler")
print("Euler model at Ts = 1: A =", A.tolist(), "B =", B.ravel().tolist())
for mu in (0.10, 0.05):
    print(f"mu = {mu}: |a| <= {mu * G:.4f} m/s^2")

# --------------------------------------------------------------------------------------------------------------
section("Part 3: the nominal scenario, x0 = (60, 5), N = 4, mu = 0.10")
x0 = np.array([60.0, 5.0])
best, sols = choose_collision_strategy(x0, N=4, mu=0.10)
for s in sols:
    print(f"{s.strategy:5s}: success {s.success}, cost {s.objective:.2f}, x_N = {s.X[-1].round(3).tolist()}, "
          f"inputs {s.U.ravel().round(3).tolist()}")
print("selected:", best.strategy)
print("yield bound active?", abs(sols[0].X[-1, 0] - d_max) < 1e-6, f"(d_4 = {sols[0].X[-1, 0]:.3f} against {d_max})")

# --------------------------------------------------------------------------------------------------------------
section("Part 4: analytic reachability under Euler")
lo, hi = terminal_position_extrema_no_speed_limits(x0, N=4, mu=0.10)
print(f"d_4 in [{lo:.3f}, {hi:.3f}] with every position-relevant acceleration at +-mu g")
for mu in (0.10, 0.12):
    width = mu * G * 4 * 3
    print(f"mu = {mu}: reachable width mu g Ts^2 N(N-1) = {width:.2f} m against the 12.5 m gap")
mu_crit = 12.5 / (G * 4 * 3)
print(f"critical friction for N = 4: {mu_crit:.4f}")

# --------------------------------------------------------------------------------------------------------------
section("Part 5: friction and feasibility, the strategy maps (660 states, two QPs each)")
t0 = time.time()
maps = {}
for mu in (0.10, 0.05):
    count, bestc = strategy_grid(D_VALUES, V_VALUES, N=4, mu=mu)
    maps[mu] = (count, bestc)
    none = int((count == 0).sum()); both = int((count == 2).sum()); yield_ok = int((bestc == 0).sum()); pass_ok = int((bestc == 1).sum())
    print(f"mu = {mu}: no strategy {none}, both {both}, yield cheaper {yield_ok}, pass cheaper {pass_ok}")
    if mu == 0.10:
        bad = [(int(D_VALUES[j]), int(V_VALUES[i])) for i, j in zip(*np.where(count == 0))]
        print("   states with no strategy:", bad)
        print("   sum d0 + 4 v0 for them:", [d + 4 * v for d, v in bad])
none10 = int((maps[0.10][0] == 0).sum()); none05 = int((maps[0.05][0] == 0).sum())
print(f"low friction adds {none05 - none10} states without a strategy; inclusion F(0.05) subset F(0.10) holds state by state:",
      bool(np.all((maps[0.05][0] > 0) <= (maps[0.10][0] > 0))))
print(f"grid time {time.time() - t0:.1f} s")
s85 = solve_collision_strategy(np.array([85.0, 3.0]), "yield", N=4, mu=0.10)
print(f"state (85, 3), yield: success {s85.success}, d_4 = {s85.X[-1, 0]:.3f} (bound {d_max}), inputs {s85.U.ravel().round(3).tolist()}")

# --------------------------------------------------------------------------------------------------------------
section("Part 6: what the weights change, mu = 0.15 at (30, 17)")
for Ru in (0.1, 1.0, 1000.0):
    best, sols = choose_collision_strategy(np.array([30.0, 17.0]), N=4, mu=0.15, Ru=Ru)
    y, p = sols
    print(f"R = {Ru:6.1f}: yield cost {y.objective:8.2f} sum a^2 {np.sum(y.U**2):6.3f} | pass cost {p.objective:8.2f} "
          f"sum a^2 {np.sum(p.U**2):6.3f} | feasible {[s.success for s in sols]} | chosen {best.strategy}")

# --------------------------------------------------------------------------------------------------------------
section("Part 7: yield is not stop")
for name, kw in (("A geometric yield", dict()), ("B full stop", dict(terminal_stop=True)),
                 ("C centre-line", dict(yield_boundary="center_line"))):
    s = solve_collision_strategy(x0, "yield", N=4, mu=0.10, **kw)
    print(f"{name}: success {s.success}" + (f", x_4 = {s.X[-1].round(3).tolist()}" if s.success else ""))
print(f"time to brake from 5 m/s to rest at {0.1 * G:.3f} m/s^2: {5 / (0.1 * G):.2f} s")
yield_ok = np.array([[solve_collision_strategy(np.array([d0, v0]), "yield", N=4, mu=0.10).success for d0 in D_VALUES] for v0 in V_VALUES])
stop_ok = np.array([[solve_collision_strategy(np.array([d0, v0]), "yield", N=4, mu=0.10, terminal_stop=True).success for d0 in D_VALUES] for v0 in V_VALUES])
print(f"yield feasible from {yield_ok.sum()} states, a full stop from {stop_ok.sum()}; a stop needs v0 <= N mu g = {4 * 0.1 * G:.2f}")

# --------------------------------------------------------------------------------------------------------------
section("Part 8: the occupancy window k = 4, 5, 6 with N = 8")
win = solve_collision_strategy(x0, "yield", N=8, conflict_steps=(4, 5, 6), mu=0.10)
print(f"from (60, 5): yield over the window: success {win.success}, cost {win.objective:.1f}, "
      f"d_4..d_8 = {win.X[4:, 0].round(2).tolist()}, inputs {win.U.ravel().round(3).tolist()}")
pw = solve_collision_strategy(x0, "pass", N=8, conflict_steps=(4, 5, 6), mu=0.10)
print(f"from (60, 5): pass over the window: success {pw.success}")
x7 = np.array([0.0, 19.0])
once = solve_collision_strategy(x7, "yield", N=8, conflict_steps=(4,), mu=0.10)
window = solve_collision_strategy(x7, "yield", N=8, conflict_steps=(4, 5, 6), mu=0.10)
print(f"from (0, 19): single-instant yield success {once.success}, d_4, d_5, d_6 = {once.X[4:7, 0].round(1).tolist()}; "
      f"window yield success {window.success}")
n_fooled = 0
for v0 in V_VALUES:
    for d0 in D_VALUES:
        a = solve_collision_strategy(np.array([d0, v0]), "yield", N=8, conflict_steps=(4,), mu=0.10).success
        b = solve_collision_strategy(np.array([d0, v0]), "yield", N=8, conflict_steps=(4, 5, 6), mu=0.10).success
        n_fooled += int(a and not b)
print(f"grid states where the single-instant yield is feasible and the window yield is not: {n_fooled}")

# --------------------------------------------------------------------------------------------------------------
section("Exercise 8: the exact zero-order-hold model")
Az, Bz = vehicle_matrices(1.0, "zoh")
full = np.full((4, 1), 0.1 * G)
from advanced_mpc_lmpc.linear import rollout
print(f"full throttle from (60, 5): d_4 = {rollout(A, B, x0, full)[-1, 0]:.3f} (Euler), {rollout(Az, Bz, x0, full)[-1, 0]:.3f} (ZOH)")
count_z, best_z = strategy_grid(D_VALUES, V_VALUES, N=4, mu=0.10, discretization="zoh")
print(f"ZOH map: no strategy {int((count_z == 0).sum())}, both {int((count_z == 2).sum())} (Euler: {none10}, {int((maps[0.10][0] == 2).sum())})")

# --------------------------------------------------------------------------------------------------------------
section("Exercise 9: planning for the worst friction in [0.05, 0.10]")
print(f"robust bound |a| <= {0.05 * G:.4f} m/s^2; states without a strategy {none05} instead of {none10}, {none05 - none10} given up")
