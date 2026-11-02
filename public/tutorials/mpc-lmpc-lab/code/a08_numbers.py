"""Reproduce every number quoted on chapter A8 of "Advanced MPC and Learning MPC",
"Polytopes and set operations" (a08-polytopes-and-set-operations.html).

Usage:  python a08_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from advanced_mpc_lmpc.polytopes import HPolytope, affine_image_2d, linear_preimage, minkowski_sum_2d

np.set_printoptions(precision=4, suppress=True)


def section(title):
    print(f"\n=== {title}")


def area(P):
    V = P.vertices_2d()
    return ConvexHull(V).volume if len(V) >= 3 else 0.0


# --------------------------------------------------------------------------------------------------------------
section("Part 1: a five-sided polygon, V to H and back")
V5 = np.array([[-2, -1], [0.5, -1.5], [2.2, -0.2], [1.2, 1.8], [-1.4, 1.4]], float)
P5 = HPolytope.from_vertices_2d(V5)
print("five vertices ->", len(P5.h), "half-spaces with unit normals")
print("H =\n", P5.H, "\nh =", P5.h)
print("vertices recovered from H:", P5.vertices_2d().tolist())
probe = np.array([1.5, 0.0])
print("probe (1.5, 0): H x - h =", P5.H @ probe - P5.h, "-> inside:", P5.contains(probe))
probe = np.array([2.0, 1.0])
print("probe (2, 1):   H x - h =", P5.H @ probe - P5.h, "-> inside:", P5.contains(probe),
      "(rows violated:", (np.where(P5.H @ probe - P5.h > 1e-9)[0] + 1).tolist(), ")")

# --------------------------------------------------------------------------------------------------------------
section("Part 1: the box |x_1| <= 4, |x_2| <= 3 as H x <= h")
X = HPolytope.box([-4, -3], [4, 3])
print("H =", X.H.tolist(), "h =", X.h.tolist(), "area", area(X))

# --------------------------------------------------------------------------------------------------------------
section("Part 2: a redundant row, removed by a linear program")
unit = HPolytope.box([-1, -1], [1, 1])
Pr = HPolytope(np.vstack([unit.H, [[1.0, 0.0]]]), np.r_[unit.h, 2.0])
print("unit box plus the row x_1 <= 2:", len(Pr.h), "rows ->", len(Pr.remove_redundancy().h), "after redundancy removal")
twice = HPolytope(np.vstack([unit.H, unit.H[:1]]), np.r_[unit.h, unit.h[:1]])
print("unit box with its first row duplicated:", len(twice.h), "rows ->", len(twice.remove_redundancy().h),
      "(a duplicate is tested against the rows still kept, so one copy survives)")


def naive_remove_redundancy(P, tol=1e-8):
    """The trap: test every row against ALL the other original rows."""
    from scipy.optimize import linprog
    keep = []
    for i in range(len(P.h)):
        idx = [j for j in range(len(P.h)) if j != i]
        res = linprog(-P.H[i], A_ub=P.H[idx], b_ub=P.h[idx], bounds=[(None, None)] * 2, method="highs")
        if (not res.success) or (-res.fun > P.h[i] + tol):
            keep.append(i)
    return HPolytope(P.H[keep], P.h[keep])


naive = naive_remove_redundancy(twice)
print("the naive test (each row against all the others) keeps", len(naive.h), "rows: both copies are dropped and the box",
      "loses its side x_1 <= 1:" , "contains (5, 0)?", naive.contains([5, 0]))

# --------------------------------------------------------------------------------------------------------------
section("Hero: the four operations at the slider's default t = 1")
P1 = HPolytope.box([-2, -1.3], [2, 1.3])
P2 = HPolytope.from_vertices_2d([[-1.5, 0], [0, -2], [1.5, 0], [0, 2]])
I = P1.intersect(P2)
print(f"intersection: 4 + 4 = 8 stacked rows, {len(I.h)} kept; {len(I.vertices_2d())} vertices", I.vertices_2d().tolist())
M = np.array([[1.2, 0.8], [0.2, 1.4]]); c = np.array([0.6, -0.2])
Y = affine_image_2d(HPolytope.box([-1, -1], [1, 1]), M, c)
print("affine image of the unit square: vertices", Y.vertices_2d().tolist(), f"; area scaled by |det M| = {abs(np.linalg.det(M)):.2f}:", round(area(Y), 3))
S = HPolytope.box([-1, -0.7], [1, 0.7]); Ap = np.array([[1, 0.7], [0, 0.85]])
Pre = linear_preimage(S, Ap)
print("linear preimage of the box under A = [[1, 0.7], [0, 0.85]]: vertices", Pre.vertices_2d().tolist())
print("   its rows are H_S A:", (S.H @ Ap).tolist(), "with the same h:", S.h.tolist())
As = HPolytope.box([-1, -0.5], [1, 0.5]); Bs = HPolytope.from_vertices_2d([[-0.4, 0], [0, -0.8], [0.4, 0], [0, 0.8]])
Sm = minkowski_sum_2d(As, Bs)
print(f"Minkowski sum: 4 x 4 = 16 pairwise sums, hull keeps {len(Sm.vertices_2d())}:", Sm.vertices_2d().tolist())

# --------------------------------------------------------------------------------------------------------------
section("Part 7: how many rows Fourier-Motzkin creates for the box target of chapter A9")
Sb = HPolytope.box([-0.4, -0.35], [0.4, 0.35])
A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
bu = (Sb.H @ B).ravel()
pos, neg = (bu > 0).sum() + 1, (bu < 0).sum() + 1     # plus one input bound on each side
print(f"{pos} upper and {neg} lower bounds on u -> {pos * neg} pairwise rows before redundancy removal")

# --------------------------------------------------------------------------------------------------------------
section("Exercises")
print("1. H-representation of -2 <= x1 <= 3, -1 <= x2 <= 4:", HPolytope.box([-2, -1], [3, 4]).H.tolist(), HPolytope.box([-2, -1], [3, 4]).h.tolist())
Pex = HPolytope(np.vstack([HPolytope.box([-2, -1], [3, 4]).H, [[1.0, 0.0]]]), np.r_[HPolytope.box([-2, -1], [3, 4]).h, 5.0])
print("2. with x1 <= 5 added:", len(Pex.h), "->", len(Pex.remove_redundancy().h), "rows")
Mex = np.array([[1.0, 1.0], [0.0, 2.0]])
print("3. affine image of [0,1]^2 under M = [[1,1],[0,2]]:", affine_image_2d(HPolytope.box([0, 0], [1, 1]), Mex).vertices_2d().tolist(), "area", round(area(affine_image_2d(HPolytope.box([0, 0], [1, 1]), Mex)), 3))
Aint = HPolytope.box([-2], [1]); Bint = HPolytope.box([-1], [3])
print("4. [-2, 1] (+) [-1, 3] = [", -2 - 1, ",", 1 + 3, "]")
box1 = HPolytope.box([0, 0], [1, 2]); box2 = HPolytope.box([-1, 1], [1, 1.5])
print("   boxes: [0,1]x[0,2] (+) [-1,1]x[1,1.5] =", minkowski_sum_2d(box1, box2).vertices_2d().tolist())
# build-it exercise: vertices from pairwise facet intersections, checked against the package
H, h = P5.H, P5.h
pts = []
for i in range(len(h)):
    for j in range(i + 1, len(h)):
        Mij = H[[i, j]]
        if abs(np.linalg.det(Mij)) < 1e-10:
            continue
        x = np.linalg.solve(Mij, h[[i, j]])
        if np.all(H @ x <= h + 1e-8):
            pts.append(x)
pts = np.unique(np.round(np.asarray(pts), 10), axis=0)
print("5. build it: pairwise facet intersections that are feasible:", len(pts), "points; the package's vertices_2d gives", len(P5.vertices_2d()))
print("   same set:", np.allclose(np.sort(pts, axis=0), np.sort(P5.vertices_2d(), axis=0)))
