"""Reproduce every number quoted on chapter A14 of "Advanced MPC and Learning MPC",
"Bellman: dynamic programming, value and policy iteration"
(a14-bellman-dynamic-programming-value-and-policy-iteration.html).

Usage:  python a14_numbers.py
Needs numpy and the course package (for the LQR and the polytopes of chapters A9 and A10).
The dynamic programming itself is written here in full, so that the script is the course's DP code.
"""
from __future__ import annotations

import itertools

import numpy as np

from advanced_mpc_lmpc.linear import dlqr
from advanced_mpc_lmpc.polytopes import (HPolytope, closed_loop_admissible_set_scalar_input,
                                         maximal_control_invariant, maximal_invariant)

np.set_printoptions(precision=4, suppress=True)
A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])


def section(title):
    print(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
section("Part 1: the integer chain, x+ = x + u, |x| <= 10, u in {-2..2}, cost x^2 + u^2, x_N = 0, N = 5")
x_chain = np.arange(-10, 11); u_chain = np.arange(-2, 3); N_chain = 5
x_next = x_chain[:, None] + u_chain[None, :]
admissible = np.abs(x_next) <= 10
succ_chain = np.clip(x_next, -10, 10) + 10
q_chain = np.where(admissible, x_chain[:, None] ** 2 + u_chain[None, :] ** 2, np.inf)
J_chain = [None] * (N_chain + 1); mu_chain = [None] * N_chain
J_chain[N_chain] = np.where(x_chain == 0, 0.0, np.inf)                 # the terminal constraint as an infinite cost
for k in range(N_chain - 1, -1, -1):
    Qk = q_chain + J_chain[k + 1][succ_chain]                          # q(x, u) + J_{k+1}(f(x, u))
    J_chain[k] = Qk.min(axis=1)
    mu_chain[k] = np.where(np.isfinite(J_chain[k]), u_chain[Qk.argmin(axis=1)], np.nan)
seqs = np.array(list(itertools.product(u_chain, repeat=N_chain)))     # brute force: every input sequence
xs = np.tile(x_chain.astype(float), (len(seqs), 1)); cost = np.zeros_like(xs)
for k in range(N_chain):
    u = seqs[:, [k]]; cost += xs ** 2 + u ** 2; xs = xs + u; cost[np.abs(xs) > 10] = np.inf
cost[xs != 0] = np.inf
J_brute = cost.min(axis=0); fin = np.isfinite(J_brute)
print("input sequences per initial state:", len(seqs))
print("same feasible states:", np.array_equal(fin, np.isfinite(J_chain[0])), "; max |J_DP - J_brute|:", np.max(np.abs(J_chain[0][fin] - J_brute[fin])))
print("stage-cost evaluations, brute force:", N_chain * len(seqs) * len(x_chain), "; DP:", N_chain * len(x_chain) * len(u_chain))
for k in range(N_chain + 1):
    d = x_chain[np.isfinite(J_chain[k])]; print(f"  k={k}: dom J_k = {{{d.min()}, ..., {d.max()}}}")
print("J_0(x) for x = 0..10:", J_chain[0][10:])
print("mu_k(2) for k = 0..4:", [float(mu_chain[k][12]) for k in range(N_chain)])

# --------------------------------------------------------------------------------------------------------------
section("Part 2: the linear-quadratic case, the Riccati recursion from three terminal weights")
K, P, eig_cl = dlqr(A, B, Q, R); Acl = A - B @ K; rho = np.max(np.abs(eig_cl))
print("P =", P.tolist(), "K =", K.ravel(), f"rho(A_cl) = {rho:.4f}, rho^2 = {rho ** 2:.4f}")


def riccati_step(P_next):
    H = R + B.T @ P_next @ B
    K_k = np.linalg.solve(H, B.T @ P_next @ A)
    return Q + A.T @ P_next @ A - A.T @ P_next @ B @ K_k, K_k


for label, Pf in [("P_f = 0", np.zeros((2, 2))), ("P_f = Q", Q.copy()), ("P_f = 10P", 10 * P)]:
    Pj, errs, gains = Pf.copy(), [np.linalg.norm(Pf - P, 2)], []
    for j in range(25):
        Pj, Kj = riccati_step(Pj); errs.append(np.linalg.norm(Pj - P, 2)); gains.append(Kj.ravel())
    print(f"  {label:>9}: ||P_(N-25) - P|| = {errs[-1]:.1e}, max |K_(N-25) - K| = {np.max(np.abs(gains[-1] - K.ravel())):.1e}, "
          f"mean rate per step over j = 2..12: {(errs[12] / errs[2]) ** 0.1:.3f}")
E0, Pj, lo, hi = 9 * P, 10 * P, [], []
for j in range(1, 11):
    Pj, _ = riccati_step(Pj); Aj = np.linalg.matrix_power(Acl, j)
    lo.append(np.linalg.eigvalsh(Pj - P).min() / np.linalg.norm(Pj - P, 2))
    hi.append(np.linalg.eigvalsh(Aj.T @ E0 @ Aj - (Pj - P)).min() / np.linalg.norm(Pj - P, 2))
print(f"  sandwich from P_f = 10P, j = 1..10: min eig(E_j)/||E_j|| = {min(lo):.2e}, min eig(bound - E_j)/||E_j|| = {min(hi):.2e} (both >= -rounding)")


# --------------------------------------------------------------------------------------------------------------
def build_lattice(h, p_max=4.0, v_max=3.0, u_max=1.0):
    """The double integrator on the lattice {v in hZ, p + v/2 in hZ} with inputs u in hZ: mapped into itself exactly.
    Returns states S (n x 2), inputs U (m,), successor indices succ (n x m, -1 outside the box), stage costs qsa."""
    Pm, Vm, Am = round(2 * p_max / h), round(v_max / h), round(u_max / h)
    I0 = (Pm + Vm) // 2 + 1
    ij = np.array([(i, j) for j in range(-Vm, Vm + 1) for i in range(-I0, I0 + 1) if abs(2 * i - j) <= Pm])
    index = -np.ones((2 * I0 + 1, 2 * Vm + 1), dtype=int)
    index[ij[:, 0] + I0, ij[:, 1] + Vm] = np.arange(len(ij))
    a = np.arange(-Am, Am + 1)
    i_next = ij[:, [0]] + ij[:, [1]] + a                 # (p + v/2)+ = (p + v/2) + v + u
    j_next = ij[:, [1]] + a                              # v+ = v + u
    inside = (np.abs(j_next) <= Vm) & (np.abs(2 * i_next - j_next) <= Pm)
    succ = np.where(inside, index[np.clip(i_next + I0, 0, 2 * I0), np.clip(j_next + Vm, 0, 2 * Vm)], -1)
    S = np.c_[(2 * ij[:, 0] - ij[:, 1]) * h / 2, ij[:, 1] * h]
    U = a * h
    qsa = np.einsum("ni,ij,nj->n", S, Q, S)[:, None] + R[0, 0] * U[None, :] ** 2
    return S, U, succ, qsa


section("Part 3: the discounted crash-state problem on the h = 0.5 lattice; value iteration")
h3, gamma, c_crash = 0.5, 0.95, 100.0
S3, U3, succ3, q3 = build_lattice(h3); n3, m3 = q3.shape
X_next = np.transpose(S3 @ A.T + U3[:, None, None] * B.ravel(), (1, 0, 2))
inside3 = (np.abs(X_next[..., 0]) <= 4) & (np.abs(X_next[..., 1]) <= 3)
print(f"lattice h = {h3}: {n3} states, {m3} inputs; max |S[succ] - (Ax + Bu)| = {np.abs(S3[succ3[inside3]] - X_next[inside3]).max()}; "
      f"-1 exactly outside the box: {np.array_equal(inside3, succ3 >= 0)}")
crash = n3
succ_c = np.vstack([np.where(succ3 >= 0, succ3, crash), np.full((1, m3), crash)])
cost_c = np.vstack([q3 + c_crash * (succ3 < 0), np.zeros((1, m3))])


def bellman(J):
    """The discounted Bellman operator of the crash problem: (TJ, greedy action indices)."""
    Qsa = cost_c + gamma * J[succ_c]
    return Qsa.min(axis=1), Qsa.argmin(axis=1)


J = np.zeros(n3 + 1)
for k in range(1, 5000):
    J_new, _ = bellman(J)
    if gamma / (1 - gamma) * np.max(np.abs(J_new - J)) < 1e-12:
        break
    J = J_new
J_star3 = J_new; mu_star3 = bellman(J_star3)[1]
Q_star3 = cost_c + gamma * J_star3[succ_c]; opt_action3 = Q_star3 <= J_star3[:, None] + 1e-9
print(f"J* certified after {k} iterations from J_0 = 0; states with tied optimal actions: {int((opt_action3[:n3].sum(axis=1) > 1).sum())}")
viable3 = np.ones(n3, bool)
while True:
    nxt = viable3 & np.any((succ3 >= 0) & viable3[np.maximum(succ3, 0)], axis=1)
    if np.array_equal(nxt, viable3): break
    viable3 = nxt
s_next = succ3[np.arange(n3), mu_star3[:n3]]
print(f"viable states: {viable3.sum()} of {n3}; max J* on viable = {J_star3[:n3][viable3].max():.2f}; min J* on non-viable = {J_star3[:n3][~viable3].min():.2f}")
print("the optimal policy keeps every viable state viable:", bool(np.all(s_next[viable3] >= 0) and np.all(viable3[s_next[viable3]])))
rng = np.random.default_rng(0)
for name, J0 in [("J_0 = 0", np.zeros(n3 + 1)), ("J_0 = 2000", np.full(n3 + 1, 2000.0)), ("J_0 ~ U[0, 2000]", rng.uniform(0, 2000, n3 + 1))]:
    Jk, errs, k_pol = J0.copy(), [np.max(np.abs(J0 - J_star3))], None
    for kk in range(1, 151):
        Jk, mu = bellman(Jk); errs.append(np.max(np.abs(Jk - J_star3)))
        if k_pol is None and opt_action3[np.arange(n3 + 1), mu].all(): k_pol = kk
    errs = np.array(errs); bound = errs[0] * gamma ** np.arange(151); zero = np.flatnonzero(errs == 0)
    print(f"  {name:>16}: ||J_0 - J*|| = {errs[0]:7.1f}; max_(k>=1) err/bound = {np.max(errs[1:] / bound[1:]):.6f}; "
          f"k = 50: err {errs[50]:.2e}, bound {bound[50]:.2e}; greedy optimal from k = {k_pol}; exact from k = {zero[0] if zero.size else None}")

# --------------------------------------------------------------------------------------------------------------
section("Part 4: policy iteration from the coasting policy u = 0")


def evaluate(mu):
    rows = np.arange(n3 + 1); P_mu = np.zeros((n3 + 1, n3 + 1)); P_mu[rows, succ_c[rows, mu]] = 1.0
    return np.linalg.solve(np.eye(n3 + 1) - gamma * P_mu, cost_c[rows, mu])


mu = np.full(n3 + 1, m3 // 2)
for k in range(50):
    J_mu = evaluate(mu); Qsa = cost_c + gamma * J_mu[succ_c]
    keep = Qsa[np.arange(n3 + 1), mu] <= Qsa.min(axis=1) + 1e-10
    mu_new = np.where(keep, mu, Qsa.argmin(axis=1)); changed = int(np.sum(mu_new != mu))
    print(f"  k = {k}: ||J_mu_k - J*|| = {np.max(np.abs(J_mu - J_star3)):.3e}, actions changed = {changed}")
    if changed == 0: break
    mu = mu_new
print(f"deterministic stationary policies on the {n3} lattice states: {m3}^{n3} = 10^{n3 * np.log10(m3):.1f}")

# --------------------------------------------------------------------------------------------------------------
section("Part 5: the undiscounted hard-constrained problem on the h = 0.25 lattice")
h4 = 0.25
S4, U4, succ4, q4 = build_lattice(h4); n4, m4 = q4.shape
q4_hard = np.where(succ4 >= 0, q4, np.inf)


def bellman_hard(J):
    Qsa = q4_hard + J[np.maximum(succ4, 0)]
    return Qsa.min(axis=1), Qsa.argmin(axis=1)


J_iter = [np.zeros(n4)]
while True:
    J_new, _ = bellman_hard(J_iter[-1])
    if np.array_equal(J_new, J_iter[-1]): break
    J_iter.append(J_new)
J_star4 = J_iter[-1]; mu_star4 = U4[bellman_hard(J_star4)[1]]; fin4 = np.isfinite(J_star4)
xQx = np.einsum("ni,ij,nj->n", S4, Q, S4)
print(f"lattice h = {h4}: {n4} states, {m4} inputs; exact fixed point at k = {len(J_iter) - 1}; "
      f"c_min = {xQx[xQx > 0].min():.4f}, max finite J* = {J_star4[fin4].max():.2f} (worst-case bound J*/c_min = {J_star4[fin4].max() / xQx[xQx > 0].min():.0f})")
print("finite-cost states per iteration |dom J_k|:", [int(np.isfinite(Jk).sum()) for Jk in J_iter])
X_box = HPolytope.box([-4.0, -3.0], [4.0, 3.0])
C_inf, C_hist = maximal_control_invariant(A, B, X_box, -1.0, 1.0)
O_inf, _ = maximal_invariant(Acl, closed_loop_admissible_set_scalar_input(X_box, K, -1.0, 1.0))


def inside(poly, pts, tol=1e-9):
    return np.all(poly.H @ pts.T <= poly.h[:, None] + tol, axis=0)


xPx4 = np.einsum("ni,ij,nj->n", S4, P, S4); in_O = inside(O_inf, S4)
sel = in_O & (xPx4 > 0); gap = (J_star4[sel] - xPx4[sel]) / xPx4[sel]
print(f"min over finite states of J* - x'Px = {np.min(J_star4[fin4] - xPx4[fin4]):.3g}")
print(f"lattice states in O_inf: {in_O.sum()}; relative gap (J* - x'Px)/x'Px there: median {np.median(gap):.4f}, mean {gap.mean():.4f}, max {gap.max():.4f}")
print(" k   lattice states in C_k (polytope)   |dom J_k| (value iteration)")
for k, Ck in enumerate(C_hist):
    Jk = J_iter[min(k, len(J_iter) - 1)]
    print(f"{k:2d}   {inside(Ck, S4).sum():30d}   {int(np.isfinite(Jk).sum()):27d}")
print("every finite-cost lattice state lies in C_inf:", bool(np.all(inside(C_inf, S4)[fin4])), "; lattice states in C_inf with infinite cost:", int(np.sum(inside(C_inf, S4) & ~fin4)))
survive = np.array([sum(np.isfinite(Jk[i]) for Jk in J_iter) - 1 for i in range(n4)])
print("doomed states leaving X after 1, 2, 3 steps:", [int(np.sum((survive == s) & ~fin4)) for s in (0, 1, 2)])
bang = np.sum(np.abs(mu_star4[fin4]) == 1.0)
print(f"optimal input at the input bound on {bang} of {fin4.sum()} viable states; from (-4, 0): J* = {J_star4[np.flatnonzero(np.all(S4 == [-4, 0], axis=1))[0]]:.3f}")
s = int(np.flatnonzero(np.all(S4 == [-4, 0], axis=1))[0]); steps = 0
pol = bellman_hard(J_star4)[1]
while S4[s].any(): s = succ4[s, pol[s]]; steps += 1
print(f"  the optimal trajectory from (-4, 0) reaches the origin in {steps} steps")

# --------------------------------------------------------------------------------------------------------------
section("Part 6: the curse of dimensionality, 30 points per axis")
for n_dim in [2, 4, 6, 8, 12]:
    pts = 30.0 ** n_dim
    print(f"  n = {n_dim:2d}: {pts:10.3g} values, {pts * 8 / 1e9:10.3g} GB per float64 table")
