"""Reproduce every number quoted on chapter A10 of "Advanced MPC and Learning MPC",
"Positive and control invariance" (a10-positive-and-control-invariance.html).

Usage:  python a10_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from advanced_mpc_lmpc.linear import dlqr
from advanced_mpc_lmpc.polytopes import (HPolytope, closed_loop_admissible_set_scalar_input, linear_preimage,
                                         maximal_control_invariant, maximal_invariant, n_step_controllable_sets)

np.set_printoptions(precision=4, suppress=True)


def section(title):
    print(f"\n=== {title}")


def area(P):
    V = P.vertices_2d()
    return ConvexHull(V).volume if len(V) >= 3 else 0.0


A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])
X = HPolytope.box([-4, -3], [4, 3]); umin, umax = -1.0, 1.0
K, P, eig = dlqr(A, B, Q, R); Acl = A - B @ K

# --------------------------------------------------------------------------------------------------------------
section("Part 2: the LQR law and where it is admissible")
print("K =", K.ravel(), "closed-loop eigenvalues", eig, "spectral radius", round(float(np.max(np.abs(eig))), 4))
for corner in ([-4, 3], [4, 3], [4, -3], [-4, -3]):
    print(f"  at the corner {corner}: -Kx = {(-K @ np.asarray(corner, float)).item():.3f}", "(admissible)" if abs((-K @ np.asarray(corner, float)).item()) <= 1 else "(over the bound)")
XK = closed_loop_admissible_set_scalar_input(X, K, umin, umax)
print("X_K = {x in X : |-Kx| <= 1}:", len(XK.vertices_2d()), "vertices", XK.vertices_2d().tolist(), "area", round(area(XK), 3))

section("Part 2: the maximal positively invariant set O_inf by the fixed point inside X_K")
O, hist = maximal_invariant(Acl, XK)
print("iterations:", len(hist) - 1, "; vertices along the way:", [len(h.vertices_2d()) for h in hist], "; areas:", [round(area(h), 3) for h in hist])
print("O_inf vertices", O.vertices_2d().tolist(), "area", round(area(O), 3))
print("check: A_cl maps every vertex of O_inf back inside O_inf:", all(O.contains(Acl @ v, 1e-7) for v in O.vertices_2d()))
print("check: the LQR input at every vertex is within the bound:", [round((-K @ v).item(), 4) for v in O.vertices_2d()])
Om1 = XK.intersect(linear_preimage(XK, Acl))
print("Omega_1 = X_K intersect A_cl^{-1}(X_K) already has", len(Om1.vertices_2d()), "vertices; Omega_2 = Omega_1:", Om1.approximately_equal(O))

# --------------------------------------------------------------------------------------------------------------
section("Part 3: the maximal control-invariant set C_inf by the fixed point from X")
C, histC = maximal_control_invariant(A, B, X, umin, umax)
print("iterations:", len(histC) - 1, "; vertices along the way:", [len(h.vertices_2d()) for h in histC], "; areas:", [round(area(h), 2) for h in histC])
print("C_inf vertices", C.vertices_2d().tolist(), "area", round(area(C), 3), "of the box's", area(X))
print(f"C_inf is {area(C) / area(O):.1f} times O_inf; every vertex of O_inf lies in C_inf:", all(C.contains(v, 1e-7) for v in O.vertices_2d()))

# --------------------------------------------------------------------------------------------------------------
section("Part 4: four states that all satisfy the box now")
for s in ([3.8, -1.0], [3.8, 2.5], [0.0, 2.8], [-3.5, -2.2]):
    print(f"  {s}: in X {X.contains(s)}, in C_inf {C.contains(s, 1e-7)}, in O_inf {O.contains(s, 1e-7)}")
print("  (3.8, 2.5): next position p+ = 3.8 + 2.5 + 0.5u >=", 3.8 + 2.5 - 0.5, "whatever u; (-3.5, -2.2): p+ <=", -3.5 - 2.2 + 0.5)

section("Part 4: the minimum-effort safe action at x = (3, 0.8)")
x = np.array([3.0, 0.8]); Ax = A @ x
lo, hi = umin, umax
for i in range(len(C.h)):
    a = (C.H[i] @ B).item(); b = float(C.h[i] - C.H[i] @ Ax)
    if a > 1e-12: hi = min(hi, b / a)
    elif a < -1e-12: lo = max(lo, b / a)
u = min(max(0.0, lo), hi)
print(f"inputs that keep the next state in C_inf: [{lo:.4f}, {hi:.4f}]; the least |u| is {u:.4f}, sending x to {(Ax + B.ravel() * u).round(4).tolist()}, in C_inf: {C.contains(Ax + B.ravel() * u, 1e-7)}")

# --------------------------------------------------------------------------------------------------------------
section("Exercises: a weaker motor, |u| <= 0.4 and |u| <= 0.5")
target = HPolytope.box([-0.25, -0.25], [0.25, 0.25])
K6 = n_step_controllable_sets(A, B, X, target, umin, umax, 6)[-1]
print(f"|u| <= 1.0: K_6 area {area(K6):.3f}, C_inf {area(C):.3f} ({len(C.vertices_2d())} vertices, {len(histC) - 1} iterations), O_inf {area(O):.3f}")
for um in (0.5, 0.4):
    XK2 = closed_loop_admissible_set_scalar_input(X, K, -um, um)
    O2, h2 = maximal_invariant(Acl, XK2); C2, hc2 = maximal_control_invariant(A, B, X, -um, um)
    K62 = n_step_controllable_sets(A, B, X, target, -um, um, 6)[-1]
    print(f"|u| <= {um}: K_6 area {area(K62):.3f}, C_inf {area(C2):.3f} ({len(C2.vertices_2d())} vertices, {len(hc2) - 1} iterations), "
          f"O_inf {area(O2):.3f} ({len(O2.vertices_2d())} vertices, {len(h2) - 1} iterations), X_K {len(XK2.vertices_2d())} vertices, area {area(XK2):.3f}")
    print("   O_inf vertices", O2.vertices_2d().tolist())

section("Exercise: a different gain (R = 2, gentler) and its invariant set")
K2, P2, e2 = dlqr(A, B, Q, np.array([[2.0]]))
XK3 = closed_loop_admissible_set_scalar_input(X, K2, umin, umax); O3, h3 = maximal_invariant(A - B @ K2, XK3)
print("K =", K2.ravel(), "radius", round(float(np.max(np.abs(e2))), 4), "; X_K area", round(area(XK3), 3), "; O_inf", len(O3.vertices_2d()), "vertices, area", round(area(O3), 3), "in", len(h3) - 1, "iterations")

section("Exercise: build the control-invariant iteration yourself, compare with the package")
from advanced_mpc_lmpc.polytopes import predecessor_scalar_input
Cmine = X
for it in range(100):
    nxt = X.intersect(predecessor_scalar_input(Cmine, A, B, umin, umax))
    if nxt.approximately_equal(Cmine):
        break
    Cmine = nxt
print("stopped after", it, "iterations; same set as the package:", Cmine.approximately_equal(C))
