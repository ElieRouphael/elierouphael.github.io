"""Reproduce every number quoted on lab B5 of "Advanced MPC and Learning MPC",
"MPC, Learning MPC and reinforcement learning" (b05-mpc-lmpc-and-reinforcement-learning.html).

Usage:  python b05_numbers.py            # about a minute: the Q-learning runs are 200 000 steps each
Needs numpy and the course package (for the LQR and the terminal set of chapter A10). The lattice
dynamic programming is the same as chapter A14's script; the Q-learning uses NumPy's random stream
with seed 0, which is what interactive/b05-data.js stores for comparison with the browser's runs.
"""
from __future__ import annotations

import numpy as np

from advanced_mpc_lmpc.linear import dlqr
from advanced_mpc_lmpc.polytopes import HPolytope, closed_loop_admissible_set_scalar_input, maximal_invariant

np.set_printoptions(precision=4, suppress=True)
A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])
K, P, eig_cl = dlqr(A, B, Q, R); Acl = A - B @ K


def section(title):
    print(f"\n=== {title}")


def build_lattice(h, p_max=4.0, v_max=3.0, u_max=1.0):
    Pm, Vm, Am = round(2 * p_max / h), round(v_max / h), round(u_max / h)
    I0 = (Pm + Vm) // 2 + 1
    ij = np.array([(i, j) for j in range(-Vm, Vm + 1) for i in range(-I0, I0 + 1) if abs(2 * i - j) <= Pm])
    index = -np.ones((2 * I0 + 1, 2 * Vm + 1), dtype=int)
    index[ij[:, 0] + I0, ij[:, 1] + Vm] = np.arange(len(ij))
    a = np.arange(-Am, Am + 1)
    i_next = ij[:, [0]] + ij[:, [1]] + a
    j_next = ij[:, [1]] + a
    inside = (np.abs(j_next) <= Vm) & (np.abs(2 * i_next - j_next) <= Pm)
    succ = np.where(inside, index[np.clip(i_next + I0, 0, 2 * I0), np.clip(j_next + Vm, 0, 2 * Vm)], -1)
    S = np.c_[(2 * ij[:, 0] - ij[:, 1]) * h / 2, ij[:, 1] * h]
    U = a * h
    qsa = np.einsum("ni,ij,nj->n", S, Q, S)[:, None] + R[0, 0] * U[None, :] ** 2
    return S, U, succ, qsa


# --------------------------------------------------------------------------------------------------------------
section("The h = 0.25 lattice, J* by undiscounted value iteration, the terminal set O_inf")
S4, U4, succ4, q4 = build_lattice(0.25); n4, m4 = q4.shape
q4_hard = np.where(succ4 >= 0, q4, np.inf)


def bellman_hard(J):
    Qsa = q4_hard + J[np.maximum(succ4, 0)]
    return Qsa.min(axis=1), Qsa.argmin(axis=1)


J_iter = [np.zeros(n4)]
while True:
    J_new, _ = bellman_hard(J_iter[-1])
    if np.array_equal(J_new, J_iter[-1]): break
    J_iter.append(J_new)
J_star4 = J_iter[-1]; fin4 = np.isfinite(J_star4)
X_box = HPolytope.box([-4.0, -3.0], [4.0, 3.0])
O_inf, _ = maximal_invariant(Acl, closed_loop_admissible_set_scalar_input(X_box, K, -1.0, 1.0))
in_O = np.all(O_inf.H @ S4.T <= O_inf.h[:, None] + 1e-9, axis=0)
xPx4 = np.einsum("ni,ij,nj->n", S4, P, S4)
origin4 = int(np.flatnonzero((S4 == 0).all(axis=1))[0])
print(f"{n4} states, {m4} inputs, {fin4.sum()} viable (finite J*), {in_O.sum()} lattice states in O_inf")

# --------------------------------------------------------------------------------------------------------------
section("Part 2: three terminal designs, MPC as the greedy policy of T^(N-1) V_f, closed loops from every feasible start")
designs = {"A": np.zeros(n4), "B": np.where(in_O, xPx4, np.inf), "C": np.where(in_O, J_star4, np.inf)}


def receding_horizon(kappa, F, steps=80):
    start = np.flatnonzero(F)
    s, cost, ok = start.copy(), np.zeros(len(start)), np.ones(len(start), bool)
    for _ in range(steps):
        ok &= F[s]
        a = kappa[s]
        cost += np.where(ok, q4[s, a], 0.0)
        s_next = succ4[s, a]
        ok &= s_next >= 0
        s = np.where(s_next >= 0, s_next, s)
    return start, cost, ok & (s == origin4)


print("design  N  feasible  doomed  failed  optimal  mean subopt %  max subopt %")
checks = {}
for name, Vf in designs.items():
    J_prev = Vf.copy(); upper = mono = bounded = True
    for N in range(1, 13):
        J_N, a_N = bellman_hard(J_prev); F = np.isfinite(J_N)
        start, cost, ok = receding_horizon(a_N, F)
        nz = ok & (start != origin4)
        sub = np.maximum(cost[nz] - J_star4[start[nz]], 0.0) / J_star4[start[nz]]
        if N in (1, 2, 3, 4, 5, 8, 12):
            print(f"  {name}  {N:2d}  {F.sum():8d}  {np.sum(F & ~fin4):6d}  {np.sum(~ok):6d}  {np.sum(ok & (np.abs(cost - J_star4[start]) < 1e-9)):7d}  "
                  f"{100 * sub.mean() if sub.size else float('nan'):13.3f}  {100 * sub.max() if sub.size else float('nan'):12.3f}")
        upper &= bool(np.all(J_N[F] >= J_star4[F] - 1e-9)); mono &= bool(np.all(J_N <= J_prev + 1e-9)); bounded &= bool(np.all(cost[ok] <= J_N[start[ok]] + 1e-9))
        J_prev = J_N
    checks[name] = (upper, mono, bounded)
stay = np.any((succ4 >= 0) & in_O[np.maximum(succ4, 0)], axis=1)
print(f"terminal states with a lattice input that stays in X_f = O_inf: {np.sum(stay & in_O)} of {in_O.sum()}")
excess = bellman_hard(designs["B"])[0][in_O] - designs["B"][in_O]
print(f"design B: the decrease condition V_f >= T V_f fails at {np.sum(excess > 1e-12)} of {in_O.sum()} terminal states (largest excess {excess.max():.4f})")
for name, (upper, mono, bounded) in checks.items():
    print(f"design {name}: J_N >= J* for all N: {upper}; J_N non-increasing in N: {mono}; closed-loop cost <= J_N(x_0): {bounded}")
# design A at N = 1: the coasting example from (-1, 2)
s = int(np.flatnonzero(np.all(S4 == [-1, 2], axis=1))[0]); JA1, aA1 = bellman_hard(designs["A"])
path = [S4[s].tolist()]
for _ in range(3):
    s = succ4[s, aA1[s]]; path.append(S4[s].tolist() if s >= 0 else None)
    if s < 0: break
print("design A, N = 1, from (-1, 2) coasting:", path)

# --------------------------------------------------------------------------------------------------------------
section("Part 3: Learning MPC on the lattice from a sluggish seed, x_S = (-4, 0)")


def rollout(policy, s0, max_steps=200):
    states, acts = [s0], []
    while states[-1] != origin4 and len(acts) < max_steps:
        a = int(policy[states[-1]]); acts.append(a); states.append(int(succ4[states[-1], a]))
        assert states[-1] >= 0
    return np.array(states), np.array(acts)


xQx = np.einsum("ni,ij,nj->n", S4, Q, S4)
s_start = int(np.flatnonzero(np.all(S4 == [-4.0, 0.0], axis=1))[0])
q_lazy = np.where(succ4 >= 0, 0.005 * xQx[:, None] + U4[None, :] ** 2, np.inf)   # the seed: optimal for a sluggish cost
J = np.zeros(n4)
for _ in range(10000):
    J_new = (q_lazy + J[np.maximum(succ4, 0)]).min(axis=1)
    if np.array_equal(J_new, J): break
    J = J_new
seed = rollout((q_lazy + J[np.maximum(succ4, 0)]).argmin(axis=1), s_start)
print(f"J*(x_S) = {J_star4[s_start]:.3f}; seed: {len(seed[1])} steps, cost {np.sum(q4[seed[0][:-1], seed[1]]):.3f}, "
      f"{100 * (np.sum(q4[seed[0][:-1], seed[1]]) / J_star4[s_start] - 1):.1f} percent above J*")
gaps = []
for N_L in [1, 2, 3, 4]:
    V = np.full(n4, np.inf); V[origin4] = 0.0
    states, acts = seed; costs = []
    for j in range(9):
        ctg = np.cumsum(q4[states[:-1], acts][::-1])[::-1]
        costs.append(ctg[0]); gaps.append(np.min(ctg - J_star4[states[:-1]]))
        if N_L == 3 and j == 1: diag = np.max(np.abs(ctg - J_star4[states[:-1]]))
        V[states[:-1]] = np.minimum(V[states[:-1]], ctg)
        J_tilde = V.copy()
        for _ in range(N_L - 1): J_tilde, _ = bellman_hard(J_tilde)
        states, acts = rollout(bellman_hard(J_tilde)[1], s_start)
    c = np.array(costs)
    print(f"  N = {N_L}: iteration costs {np.round(c, 3)}; never increases: {bool(np.all(np.diff(c) <= 1e-9))}; final gap to J*: {100 * (c[-1] / J_star4[s_start] - 1):.1f}%")
print(f"smallest stored J^j_t - J*(x^j[t]) over all runs and iterations: {min(gaps):.2e}; N = 3, iteration 1: max |J^1_t - J*(x^1[t])| = {diag:.2e}")

# --------------------------------------------------------------------------------------------------------------
section("Part 5: tabular Q-learning on the h = 0.5 crash problem of chapter A14 (gamma = 0.95, crash cost 100), seed 0")
S3, U3, succ3, q3 = build_lattice(0.5); n3, m3 = q3.shape
gamma, c_crash = 0.95, 100.0
succ_c = np.vstack([np.where(succ3 >= 0, succ3, n3), np.full((1, m3), n3)])
cost_c = np.vstack([q3 + c_crash * (succ3 < 0), np.zeros((1, m3))])
J = np.zeros(n3 + 1)
for k in range(1, 5000):
    Qsa = cost_c + gamma * J[succ_c]; J_new = Qsa.min(axis=1)
    if gamma / (1 - gamma) * np.max(np.abs(J_new - J)) < 1e-12: break
    J = J_new
J_star3 = J_new; Q_star3 = cost_c + gamma * J_star3[succ_c]; opt_action3 = Q_star3 <= J_star3[:, None] + 1e-9
viable3 = np.ones(n3, bool)
while True:
    nxt = viable3 & np.any((succ3 >= 0) & viable3[np.maximum(succ3, 0)], axis=1)
    if np.array_equal(nxt, viable3): break
    viable3 = nxt
print(f"{n3} states, {m3} inputs, {viable3.sum()} viable; value iteration: greedy policy optimal after 6 sweeps = {6 * n3 * m3} backups over {n3 * m3} pairs, no crash")


def q_learning(n_steps, alpha, eps, Q0=0.0, horizon=30, seed=0, every=2000):
    rng = np.random.default_rng(seed)
    explore = rng.random(n_steps) < eps; rand_a = rng.integers(m3, size=n_steps); starts = rng.integers(n3, size=n_steps)
    Qtab = [[Q0] * m3 for _ in range(n3)]; visits = [[0] * m3 for _ in range(n3)]
    succ_l, cost_l = succ3.tolist(), cost_c[:n3].tolist()
    s, t_ep, crashes, log = int(starts[0]), 0, 0, []
    for t in range(n_steps):
        row = Qtab[s]
        a = int(rand_a[t]) if explore[t] else row.index(min(row))
        s2, c = succ_l[s][a], cost_l[s][a]
        target = c if s2 < 0 else c + gamma * min(Qtab[s2])            # the temporal-difference target
        visits[s][a] += 1
        step = visits[s][a] ** -0.6 if alpha == "1/n^0.6" else alpha
        row[a] += step * (target - row[a])
        crashes += s2 < 0; t_ep += 1
        if s2 < 0 or t_ep == horizon: s, t_ep = int(starts[t]), 0
        else: s = s2
        if (t + 1) % every == 0:
            greedy = np.argmin(np.array(Qtab), axis=1)
            log.append((t + 1, opt_action3[np.arange(n3), greedy][viable3].mean(), crashes))
    return np.array(Qtab), np.array(log)


configs = [("alpha=1, eps=0.1", dict(alpha=1.0, eps=0.1)), ("alpha=0.1, eps=0.1", dict(alpha=0.1, eps=0.1)),
           ("alpha=1/n^0.6, eps=0.1", dict(alpha="1/n^0.6", eps=0.1)), ("alpha=1, eps=0, Q0=0 (optimistic)", dict(alpha=1.0, eps=0.0, Q0=0.0)),
           ("alpha=1, eps=0, Q0=200 (pessimistic)", dict(alpha=1.0, eps=0.0, Q0=200.0))]
for name, cfg in configs:
    Qtab, log = q_learning(200_000, **cfg)
    hit = np.flatnonzero(log[:, 1] == 1.0)
    first = f"{int(log[hit[0], 0])} (crashes so far {int(log[hit[0], 2])})" if hit.size else "never"
    greedy_crash = int(np.sum((succ3[np.arange(n3), Qtab.argmin(axis=1)] < 0)[viable3]))
    print(f"  {name:>37}: all optimal from step {first}; final fraction {log[-1, 1]:.3f}; crashes {int(log[-1, 2])}; viable states whose greedy input crashes: {greedy_crash}")
    if name == configs[0][0]: Q_first = Qtab
best = Q_star3[:n3].argmin(axis=1)
err_best = np.abs(Q_first[np.arange(n3), best] - Q_star3[np.arange(n3), best])[viable3]
print(f"alpha=1, eps=0.1 after 200 000 steps: max |Q - Q*| on (viable state, optimal input) = {err_best.max():.2e}; over all pairs = {np.abs(Q_first - Q_star3[:n3]).max():.1f}")
print(f"largest crash target: {cost_c[:n3].max():.1f} (a crash target is only its one-step cost)")
