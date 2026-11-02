"""Reproduce every number quoted on chapter A3 of "Advanced MPC and Learning MPC",
"Convexity, quadratic programs and the KKT conditions" (a03-convexity-qps-and-kkt.html).

Usage:  python a03_numbers.py
Needs numpy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
"""
from __future__ import annotations

import numpy as np

from advanced_mpc_lmpc.qp import solve_qp

np.set_printoptions(precision=5, suppress=True)


def section(title):
    print(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
section("Part 2: convexity of a quadratic, P = [[2, alpha], [alpha, 1]]")
for alpha in (1.4, np.sqrt(2.0), 1.5):
    print(f"alpha = {alpha:.4f}: eigenvalues {np.linalg.eigvalsh([[2.0, alpha], [alpha, 1.0]])}, determinant {2 - alpha**2:.4f}")
print("two fixed examples:", np.linalg.eigvalsh([[3.0, 0.8], [0.8, 1.5]]), np.linalg.eigvalsh([[1.0, 0.0], [0.0, -0.6]]))

# --------------------------------------------------------------------------------------------------------------
section("Hero and Part 4: the hand QP  min ||z - c||^2  s.t.  z1 + z2 <= 1,  c = (1.2, 0.8)")
c = np.array([1.2, 0.8]); a = np.array([1.0, 1.0]); b = 1.0
Pm = 2 * np.eye(2); q = -2 * c
res = solve_qp(Pm, q, A_ub=a[None, :], b_ub=[b])
z, lam = res.x, res.lam[0]
print("z* =", z, " lambda* =", round(lam, 10), " f* = ||z*-c||^2 =", round(float((z - c) @ (z - c)), 6), " iterations", res.nit)
grad = Pm @ z + q
print("gradient of the objective at z*:", grad, " = -lambda* a:", -lam * a)
print("KKT residuals: stationarity", np.abs(grad + lam * a).max(), " primal", a @ z - b, " dual lam >= 0:", lam >= 0,
      " complementary slackness lam*(a'z-b) =", lam * (a @ z - b))
print("hand solution: lambda* = a'c - b =", a @ c - b, ", z* = c - lambda/2 a =", c - (a @ c - b) / 2 * a)


def dual(l):
    y = q + l * a
    return -0.5 * y @ np.linalg.solve(Pm, y) - b * l


print("dual function at lambda*:", round(dual(lam), 6), " primal (constant c'c dropped):", round(res.objective, 6),
      " gap", abs(dual(lam) - res.objective))
ls = np.linspace(0, 3, 301); d = np.array([dual(l) for l in ls])
print("the dual is maximised at lambda =", round(ls[d.argmax()], 2), " (grid of 0.01)")

section("Part 4: the multiplier as a shadow price, df*/db = -lambda*")
def fstar(bb):
    r = solve_qp(Pm, q, A_ub=a[None, :], b_ub=[bb])
    return float((r.x - c) @ (r.x - c))
h = 1e-6
for bb in (0.8, 1.0, 1.5, 1.9, 2.1):
    lam_b = max(a @ c - bb, 0)
    print(f"b = {bb}: lambda* = {lam_b:.3f}, numerical df*/db = {(fstar(bb + h) - fstar(bb - h)) / (2 * h):.6f}")

section("Part 4: an inactive constraint has a zero multiplier (b = 2.5)")
r = solve_qp(Pm, q, A_ub=a[None, :], b_ub=[2.5])
print("z* =", r.x, " lambda* =", r.lam[0], " a'z* - b =", a @ r.x - 2.5)

# --------------------------------------------------------------------------------------------------------------
section("Part 5: a QP with two constraints, one active and one not")
G = np.array([[1.0, 1.0], [1.0, -1.0]]); g = np.array([1.0, 2.0])
r = solve_qp(Pm, q, A_ub=G, b_ub=g)
print("z* =", r.x, " multipliers", r.lam, " slacks", g - G @ r.x)

section("Part 5: an equality constraint and its multiplier nu")
r = solve_qp(Pm, q, A_eq=a[None, :], b_eq=[1.0])
print("z* =", r.x, " nu =", r.nu, " (stationarity 2(z - c) + nu a = 0)")

# --------------------------------------------------------------------------------------------------------------
section("Exercise 2: KKT by hand for c = (3, 1), b = 2")
c2 = np.array([3.0, 1.0])
r = solve_qp(Pm, -2 * c2, A_ub=a[None, :], b_ub=[2.0])
print("z* =", r.x, " lambda* =", r.lam[0], " hand: lambda = a'c - b =", a @ c2 - 2.0, " f* =", float((r.x - c2) @ (r.x - c2)))

section("Exercise 5: the unconstrained minimiser of a strictly convex quadratic, P = [[3, 0.8], [0.8, 1.5]], q = (-1, 2)")
P5 = np.array([[3.0, 0.8], [0.8, 1.5]]); q5 = np.array([-1.0, 2.0])
z5 = -np.linalg.solve(P5, q5)
print("z* = -P^{-1} q =", z5, " gradient there:", P5 @ z5 + q5, " value", 0.5 * z5 @ P5 @ z5 + q5 @ z5)
