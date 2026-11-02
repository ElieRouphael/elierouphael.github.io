"""Reproduce every number quoted on chapter A9 of "Advanced MPC and Learning MPC",
"Pre, Post and controllable sets" (a09-pre-post-and-controllable-sets.html).

Usage:  python a09_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from advanced_mpc_lmpc.polytopes import (HPolytope, n_step_controllable_sets, predecessor_scalar_input,
                                         reachable_set_scalar_input_2d)

np.set_printoptions(precision=4, suppress=True)


def section(title):
    print(f"\n=== {title}")


def area(P):
    V = P.vertices_2d()
    return ConvexHull(V).volume if len(V) >= 3 else 0.0


A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
X = HPolytope.box([-4, -3], [4, 3]); umin, umax = -1.0, 1.0

# --------------------------------------------------------------------------------------------------------------
section("Parts 1-3: Post and Pre of the small box S = [-0.4, 0.4] x [-0.35, 0.35] with |u| <= 1")
S = HPolytope.box([-0.4, -0.35], [0.4, 0.35])
Post = reachable_set_scalar_input_2d(S, A, B, umin, umax)
print("Post(S) = A S (+) B U: hull of 4 x 2 = 8 points Av_i + Bu, vertices", Post.vertices_2d().tolist(), "area", round(area(Post), 4))
bu = (S.H @ B).ravel()
print("Fourier-Motzkin: the target rows give u-coefficients", bu.tolist(), "plus the two input bounds:",
      f"{(bu > 0).sum() + 1} upper and {(bu < 0).sum() + 1} lower bounds -> {((bu > 0).sum() + 1) * ((bu < 0).sum() + 1)} pairwise rows")
Pre = predecessor_scalar_input(S, A, B, umin, umax)
print(f"Pre_U(S): {len(Pre.h)} rows survive redundancy removal; vertices", X.intersect(Pre).vertices_2d().tolist(), "area", round(area(Pre), 4))
print("Pre and Post are mirror images here (symmetric S and U, linear map):",
      np.allclose(np.sort(Pre.vertices_2d(), axis=0), np.sort(-Post.vertices_2d(), axis=0)))

section("Part 3: the witness input at x = (-0.65, 0.45)")
x = np.array([-0.65, 0.45]); Ax = A @ x
lo, hi = umin, umax
for i in range(len(S.h)):
    a = (S.H[i] @ B).item(); b = float(S.h[i] - S.H[i] @ Ax)
    if a > 1e-12: hi = min(hi, b / a)
    elif a < -1e-12: lo = max(lo, b / a)
print(f"x in S? {S.contains(x)}; in Pre(S)? {Pre.contains(x)}; witness interval u in [{lo:.4f}, {hi:.4f}]")
res = linprog(np.zeros(1), A_ub=(S.H @ B), b_ub=S.h - S.H @ Ax, bounds=[(umin, umax)], method="highs")
u = float(res.x[0]); print(f"a linear program with a zero objective returns u = {u:.4f}, sending x to {(Ax + B.ravel() * u).tolist()}, on the boundary of S")
print("the other end sends it to", (Ax + B.ravel() * lo).tolist())

# --------------------------------------------------------------------------------------------------------------
section("Hero and Part 4: controllable sets to the target box [-0.25, 0.25]^2, N = 6")
target = HPolytope.box([-0.25, -0.25], [0.25, 0.25])
Ks = n_step_controllable_sets(A, B, X, target, umin, umax, 6)
print("vertices of K_0..K_6:", [len(K.vertices_2d()) for K in Ks])
print("areas of K_0..K_6:  ", [round(area(K), 3) for K in Ks])
print("nested (every vertex of K_j lies in K_{j+1}):", all(all(Ks[j + 1].contains(v, 1e-7) for v in Ks[j].vertices_2d()) for j in range(6)))
print("the target is control invariant (K_0 inside K_1):", all(Ks[1].contains(v, 1e-7) for v in Ks[0].vertices_2d()))
pv = np.linspace(-4, 4, 161); vv = np.linspace(-3, 3, 121)
hist = [0] * 7; out = 0
for p_ in pv:
    for v_ in vv:
        k = next((j for j, K in enumerate(Ks) if K.contains([p_, v_], 1e-9)), -1)
        if k < 0: out += 1
        else: hist[k] += 1
print(f"grid of 161 x 121 = {161 * 121} states: {out} are in the box but not in K_6; N_min histogram (0..6):", hist)
for probe in ([3.8, -1.0], [-2.0, 1.0], [3.8, 2.5]):
    k = next((j for j, K in enumerate(Ks) if K.contains(probe, 1e-9)), -1)
    print(f"probe {probe}: N_min = {k if k >= 0 else 'more than 6 (not in K_6)'}")

section("Part 5: reachable against controllable")
R1 = reachable_set_scalar_input_2d(X, A, B, umin, umax)
print("Post(X), one step from anywhere in the box: vertices", R1.vertices_2d().tolist(), "area", round(area(R1), 2), "(the box has 48)")
print("it sticks out of the box: a cart at the edge moving outward leaves in one step whatever the input")

# --------------------------------------------------------------------------------------------------------------
section("Exercises")
print("1. Fourier-Motzkin pair: a1'x + 2u <= g1 and a2'x - u <= g2 -> (a1 + 2 a2)'x <= g1 + 2 g2")
S2 = HPolytope.box([-1, -0.5], [1, 0.5])
P2 = predecessor_scalar_input(S2, A, B, -0.5, 0.5)
print("2. Pre of [-1,1]x[-0.5,0.5] with |u| <= 0.5:", len(P2.h), "rows, vertices", X.intersect(P2).vertices_2d().tolist())
for um in (0.5, 1.0, 2.0):
    K6 = n_step_controllable_sets(A, B, X, target, -um, um, 6)[-1]
    print(f"3. |u| <= {um}: area of K_6 = {area(K6):.3f} ({len(K6.vertices_2d())} vertices)")
Kz = n_step_controllable_sets(A, B, X, HPolytope(np.vstack([np.eye(2), -np.eye(2)]), np.zeros(4)), umin, umax, 8)
print("4. to the origin exactly (target {0}): areas of K_1..K_8:", [round(area(K), 2) for K in Kz[1:]])
# build it: the predecessor by projection of the lifted polytope, checked against the package
G = np.c_[S.H @ A, S.H @ B]; g = S.h.copy()
G = np.vstack([G, [[0, 0, 1], [0, 0, -1]]]); g = np.r_[g, umax, -umin]
pos = [i for i in range(len(g)) if G[i, 2] > 1e-12]; neg = [i for i in range(len(g)) if G[i, 2] < -1e-12]; zer = [i for i in range(len(g)) if abs(G[i, 2]) <= 1e-12]
rows = [(G[i, :2], g[i]) for i in zer] + [(G[p, :2] / G[p, 2] - G[n, :2] / G[n, 2], g[p] / G[p, 2] - g[n] / G[n, 2]) for n in neg for p in pos]
mine = HPolytope(np.array([r for r, _ in rows]), np.array([b for _, b in rows])).remove_redundancy()
print(f"5. build it: {len(rows)} pairwise rows -> {len(mine.h)} after redundancy; same set as the package:", mine.approximately_equal(Pre))
