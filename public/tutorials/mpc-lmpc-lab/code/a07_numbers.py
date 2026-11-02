"""Reproduce every number quoted on chapter A7 of "Advanced MPC and Learning MPC",
"Lyapunov stability and LQR" (a07-lyapunov-stability-and-lqr.html).

Usage:  python a07_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from advanced_mpc_lmpc.linear import dlqr

np.set_printoptions(precision=4, suppress=True)

A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])


def section(title):
    print(f"\n=== {title}")


def riccati(P):
    S = R + B.T @ P @ B
    Kk = np.linalg.solve(S, B.T @ P @ A)
    return Q + A.T @ P @ A - A.T @ P @ B @ Kk, Kk


# --------------------------------------------------------------------------------------------------------------
section("Part 1: the open loop is not Schur")
print("eigenvalues of A:", np.linalg.eigvals(A), "spectral radius", np.abs(np.linalg.eigvals(A)).max())
print("A^k for k = 10:", np.linalg.matrix_power(A, 10).tolist(), "(the position entry grows linearly: coasting)")

# --------------------------------------------------------------------------------------------------------------
section("Part 2: a Lyapunov function for a fixed gain, K = (0.5, 1): solve the Lyapunov equation")
K0 = np.array([[0.5, 1.0]]); Acl0 = A - B @ K0
print("eigenvalues of A - B K0:", np.linalg.eigvals(Acl0).round(4), "spectral radius", np.abs(np.linalg.eigvals(Acl0)).max().round(4))
P0 = solve_discrete_lyapunov(Acl0.T, np.eye(2))
print("P solving Acl' P Acl - P = -I:", P0.round(4).tolist(), "eigenvalues", np.linalg.eigvalsh(P0).round(4), "(positive definite)")
x = np.array([-3.6, 2.4]); Vs = []
for k in range(6):
    Vs.append(float(x @ P0 @ x)); x = Acl0 @ x
print("V = x'P x along the loop from (-3.6, 2.4):", np.round(Vs, 3), "ratios", np.round(np.array(Vs[1:]) / np.array(Vs[:-1]), 3))

# --------------------------------------------------------------------------------------------------------------
section("Part 3: the infinite-horizon LQR of the canonical system")
K, P, eig = dlqr(A, B, Q, R)
Acl = A - B @ K
print("K =", K.ravel().round(4), " P =", P.round(4).tolist())
print("closed-loop eigenvalues", eig.round(4), " spectral radius", np.abs(eig).max().round(4), " |eig|^2 =", (np.abs(eig).max() ** 2).round(4))
resid = Acl.T @ P @ Acl - P + Q + K.T @ R @ K
print("DARE identity residual |Acl' P Acl - P + Q + K'RK|_max =", np.abs(resid).max())
print("P - (Q + A'PA - A'PB (R + B'PB)^-1 B'PA) =", np.abs(P - riccati(P)[0]).max())
print("eigenvalues of P:", np.linalg.eigvalsh(P).round(4))

# --------------------------------------------------------------------------------------------------------------
section("Hero: the LQR closed loop from (-3.6, 2.4)")
x = np.array([-3.6, 2.4]); Vs, qs, us, xs = [], [], [], [x.copy()]
for k in range(8):
    u = float(-(K @ x).item()); Vs.append(float(x @ P @ x)); qs.append(float(x @ Q @ x + R.item() * u * u)); us.append(u)
    x = A @ x + B @ np.array([u]); xs.append(x.copy())
print("first move u0 = -K x0 =", round(us[0], 4), "(beyond the bound |u| <= 1 that the constrained chapters impose)")
print("V(x_k):", np.round(Vs, 4))
print("q(x_k, u_k):", np.round(qs, 4))
print("V(x_{k+1}) - V(x_k) + q(x_k, u_k):", np.array([Vs[k + 1] - Vs[k] + qs[k] for k in range(len(Vs) - 1)]))
print("states:", [tuple(np.round(v, 3).tolist()) for v in xs[:5]])
print("the loop is a spiral (complex eigenvalues): V falls to 1e-6 of its start after", next(k for k, v in enumerate(Vs) if v < 1e-6 * Vs[0]) if any(v < 1e-6 * Vs[0] for v in Vs) else "more than 8", "steps")

# --------------------------------------------------------------------------------------------------------------
section("Hero sliders: other weights")
for q2, r in ((0.2, 0.2), (0.2, 0.02), (0.2, 2.0), (1.0, 0.2), (0.05, 0.2)):
    Kq, Pq, eq = dlqr(A, B, np.diag([1.0, q2]), np.array([[r]]))
    print(f"Q = diag(1, {q2}), R = {r}: K = {Kq.ravel().round(4)}, rho = {np.abs(eq).max():.4f}, first move from (-3.6, 2.4) = {-(Kq @ np.array([-3.6, 2.4])).item():.3f}")

# --------------------------------------------------------------------------------------------------------------
section("Part 5: the Riccati recursion from a terminal weight P_f (the finite-horizon value)")
for name, Pf in (("0", np.zeros((2, 2))), ("Q", Q), ("10 P", 10 * P), ("P", P)):
    Pk = Pf.copy(); errs = [np.abs(Pk - P).max()]
    for k in range(40):
        Pk, Kk = riccati(Pk); errs.append(np.abs(Pk - P).max())
    first = next((k for k, e in enumerate(errs) if e < 1e-12), None)
    rates = [errs[k + 1] / errs[k] for k in range(8, 13) if errs[k] > 1e-13]
    print(f"P_f = {name}: error after 1, 2, 5, 10, 20 steps = {errs[1]:.2e}, {errs[2]:.2e}, {errs[5]:.2e}, {errs[10]:.2e}, {errs[20]:.2e};"
          f" below 1e-12 after {first} steps; mean rate per step over steps 8-13 {np.mean(rates) if rates else float('nan'):.4f} against rho^2 = {np.abs(eig).max() ** 2:.4f}")
    if errs[2] > 1e-13:
        print(f"   geometric mean rate from step 2 to step 10: {(errs[10] / errs[2]) ** (1 / 8):.4f}; step by step the ratio wobbles with period three"
              f" (complex eigenvalues): {[round(float(errs[k + 1] / errs[k]), 3) for k in range(1, 10)]}")
# the sandwich inequality
Pk = 10 * P
for k in range(3):
    Pn, Kk = riccati(Pk)
    AX = A - B @ Kk
    lower = AX.T @ (Pk - P) @ AX; upper = Acl.T @ (Pk - P) @ Acl; mid = Pn - P
    print(f"sandwich at step {k + 1} from 10P: eig(mid - lower) >= 0: {np.linalg.eigvalsh(mid - lower).min():.2e}, eig(upper - mid) >= 0: {np.linalg.eigvalsh(upper - mid).min():.2e}")
    Pk = Pn

# --------------------------------------------------------------------------------------------------------------
section("Exercise: the wrong sign convention")
print("eigenvalues of A - B K:", np.linalg.eigvals(A - B @ K).round(4))
print("eigenvalues of A + B K:", np.linalg.eigvals(A + B @ K).round(4), "(one outside the unit circle: positive feedback)")

# --------------------------------------------------------------------------------------------------------------
section("Exercise: the decrease identity checked on random states")
rng = np.random.default_rng(0)
errs = [float(x @ (Acl.T @ P @ Acl - P + Q + K.T @ R @ K) @ x) for x in rng.normal(size=(100, 2))]
print("largest |V(Acl x) - V(x) + q(x, -Kx)| over 100 random states:", max(abs(e) for e in errs))

# --------------------------------------------------------------------------------------------------------------
section("Exercise: a scalar DARE by hand, x+ = 1.2 x + u, q = x^2 + 4 u^2")
a, r = 1.2, 4.0
# P = a^2 P - a^2 P^2/(r + P) + 1  ->  P^2 + (1 - a^2 - r) P... solve the quadratic
Ks, Ps, es = dlqr(np.array([[a]]), np.array([[1.0]]), np.array([[1.0]]), np.array([[r]]))
# multiply the scalar DARE P = a^2 P - a^2 P^2 / (r + P) + 1 by (r + P): P^2 + (r - a^2 r - 1) P - r = 0
b = r - a**2 * r - 1
Phand = (-b + np.sqrt(b * b + 4 * r)) / 2
print(f"P from the DARE = {Ps.item():.4f}; by hand, the positive root of P^2 + (r - a^2 r - 1) P - r = 0, i.e. P^2 - {-b:.2f} P - {r} = 0, is {Phand:.4f}; K = aP/(r + P) = {a * Phand / (r + Phand):.4f}, eigenvalue a - K = {a - a * Phand / (r + Phand):.4f}")
