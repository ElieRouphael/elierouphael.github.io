"""Reproduce every number quoted on chapter A2 of "Advanced MPC and Learning MPC",
"Prediction matrices and condensing" (a02-prediction-matrices-and-condensing.html).

Usage:  python a02_numbers.py
Needs numpy, scipy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).
"""
from __future__ import annotations

import time

import numpy as np

from advanced_mpc_lmpc.linear import prediction_matrices, rollout
from advanced_mpc_lmpc.mpc import LinearMPC
from advanced_mpc_lmpc.qp import solve_qp

np.set_printoptions(precision=4, suppress=True, linewidth=120)

A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
Q = np.diag([1.0, 0.2]); R = np.array([[0.2]]); P = Q      # P = Q until chapter A7 supplies the Riccati P
x0 = np.array([-3.0, 0.0])


def section(title):
    print(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
section("Part 1: the prediction matrices for N = 3, by hand and by prediction_matrices")
Sx3, Su3 = prediction_matrices(A, B, 3)
print("A^2 =", np.linalg.matrix_power(A, 2).tolist(), " A^3 =", np.linalg.matrix_power(A, 3).tolist())
print("AB =", (A @ B).ravel().tolist(), " A^2 B =", (A @ A @ B).ravel().tolist())
print("Sx =\n", Sx3, "\nSu =\n", Su3)
Sx_hand = np.array([[1, 1], [0, 1], [1, 2], [0, 1], [1, 3], [0, 1]], float)
Su_hand = np.array([[0.5, 0, 0], [1, 0, 0], [1.5, 0.5, 0], [1, 1, 0], [2.5, 1.5, 0.5], [1, 1, 1]])
print("match the hand-derived matrices:", np.allclose(Sx3, Sx_hand), np.allclose(Su3, Su_hand))

# --------------------------------------------------------------------------------------------------------------
section("Hero: prediction against rollout, N = 6 from (-3, 0), U = (0.8, 0.2, -0.4, -0.5, 0.3, 0.1)")
U6 = np.array([0.8, 0.2, -0.4, -0.5, 0.3, 0.1])
Sx6, Su6 = prediction_matrices(A, B, 6)
Xc = (Sx6 @ x0 + Su6 @ U6).reshape(6, 2)
Xr = rollout(A, B, x0, U6[:, None])[1:]
print("predicted states", [tuple(np.round(x, 3).tolist()) for x in Xc])
print("largest difference from the step-by-step rollout:", np.abs(Xc - Xr).max())

# --------------------------------------------------------------------------------------------------------------
section("Part 2: condensing the cost, N = 6, Q = diag(1, 0.2), R = 0.2, P = Q")
N = 6
Qbar = np.kron(np.eye(N), Q); Qbar[-2:, -2:] = P
Rbar = np.kron(np.eye(N), R)
H = 2 * (Su6.T @ Qbar @ Su6 + Rbar)
f = 2 * Su6.T @ Qbar @ Sx6 @ x0
c = x0 @ (Sx6.T @ Qbar @ Sx6 + Q) @ x0
ev = np.linalg.eigvalsh(H)
print("eigenvalues of H:", np.round(ev, 4), " condition number:", round(ev[-1] / ev[0], 1))
print("H is positive definite (smallest eigenvalue > 0):", ev[0] > 0, " and 2 lambda_min(Rbar) =", 2 * 0.2)
Ustar = -np.linalg.solve(H, f)
print("unconstrained minimiser U* = -H^{-1} f =", np.round(Ustar, 3), " first push", round(Ustar[0], 3), "> the bound 1")
Xstar = (Sx6 @ x0 + Su6 @ Ustar).reshape(N, 2)
print("its states", [tuple(np.round(x, 3).tolist()) for x in Xstar], " dropped constant c(x0) =", round(c, 4))
# the two ways of writing the cost agree on a random U
rng = np.random.default_rng(0)
for _ in range(3):
    U = rng.normal(size=N)
    X = rollout(A, B, x0, U[:, None])
    J_direct = sum(X[k] @ Q @ X[k] + R[0, 0] * U[k] ** 2 for k in range(N)) + X[N] @ P @ X[N]
    J_cond = 0.5 * U @ H @ U + f @ U + c
    print(f"random U: direct sum {J_direct:.6f}, condensed form {J_cond:.6f}, difference {abs(J_direct - J_cond):.1e}")

# --------------------------------------------------------------------------------------------------------------
section("Part 3: condensing the state constraint |p_k| <= 4 for N = 3 from (-3, 0)")
Hx = np.array([[1.0, 0.0], [-1.0, 0.0]]); hx = np.array([4.0, 4.0])
Hbar = np.kron(np.eye(3), Hx); hbar = np.tile(hx, 3)
print("Hbar Su =\n", Hbar @ Su3)
print("hbar - Hbar Sx x0 =", hbar - Hbar @ Sx3 @ x0)

section("Part 3: the difference matrix D for N = 5 and the rate check")
from advanced_mpc_lmpc.lmpc import prediction_matrices as _pm  # noqa: F401  (same function; keeps the import list honest)


def difference_matrix(N):
    D = np.eye(N)
    D[np.arange(1, N), np.arange(N - 1)] = -1.0
    return D


D5 = difference_matrix(5)
print("D =\n", D5)
U5 = np.array([0.3, 0.5, 0.4, 0.1, -0.2]); u_prev = 0.2
print("D U + (-u_prev, 0, ...) =", D5 @ U5 + np.r_[-u_prev, np.zeros(4)], " np.diff:", np.diff(np.r_[u_prev, U5]))

# --------------------------------------------------------------------------------------------------------------
section("Part 4: sparse against condensed, N = 8 from (-3, 0), box |p|<=4, |v|<=3, |u|<=1, P = Q")
N = 8
Sx, Su = prediction_matrices(A, B, N)
Qbar = np.kron(np.eye(N), Q); Rbar = np.kron(np.eye(N), R)
Hc = 2 * (Su.T @ Qbar @ Su + Rbar); fc = 2 * Su.T @ Qbar @ Sx @ x0
A_ub = np.vstack([Su, -Su]); b_ub = np.r_[np.tile([4, 3.0], N) - Sx @ x0, np.tile([4, 3.0], N) + Sx @ x0]
t = time.perf_counter(); rc = solve_qp(Hc, fc, A_ub=A_ub, b_ub=b_ub, lb=-np.ones(N), ub=np.ones(N)); tc = time.perf_counter() - t
Uc = rc.x; Xc = (Sx @ x0 + Su @ Uc).reshape(N, 2)
cost_c = x0 @ Q @ x0 + sum(Xc[k] @ Q @ Xc[k] for k in range(N - 1)) + Xc[-1] @ P @ Xc[-1] + R[0, 0] * np.sum(Uc ** 2)
# sparse: z = (x_1..x_N, u_0..u_{N-1}), dynamics as equalities
nz = 2 * N + N
Hs = np.zeros((nz, nz)); Hs[:2 * N, :2 * N] = 2 * Qbar; Hs[2 * N:, 2 * N:] = 2 * Rbar
E = np.zeros((2 * N, nz)); e = np.zeros(2 * N)
for k in range(N):
    E[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.eye(2)
    if k:
        E[2 * k:2 * k + 2, 2 * (k - 1):2 * k] = -A
    else:
        e[0:2] = A @ x0
    E[2 * k:2 * k + 2, 2 * N + k] = -B.ravel()
lb = np.r_[np.tile([-4, -3.0], N), -np.ones(N)]; ub = np.r_[np.tile([4, 3.0], N), np.ones(N)]
t = time.perf_counter(); rs = solve_qp(Hs, np.zeros(nz), A_eq=E, b_eq=e, lb=lb, ub=ub); ts = time.perf_counter() - t
Us = rs.x[2 * N:]; Xs = rs.x[:2 * N].reshape(N, 2)
print(f"condensed: {N} variables, {A_ub.shape[0] + 2 * N} inequalities, {rc.nit} iterations, {1e3 * tc:.1f} ms, cost {cost_c:.6f}")
print(f"sparse:    {nz} variables, {2 * N} equalities, {rs.nit} iterations, {1e3 * ts:.1f} ms")
print("inputs agree to", np.abs(Uc - Us).max(), "; states agree to", np.abs(Xc - Xs).max())
print("first inputs", np.round(Uc[:4], 4), " saturated steps:", int(np.sum(np.abs(np.abs(Uc) - 1) < 1e-7)))
print("sparse Hessian nonzeros:", int(np.count_nonzero(Hs)), "of", nz * nz, "; condensed Hessian nonzeros:", int(np.count_nonzero(Hc)), "of", N * N)
print("sparse equality matrix nonzeros:", int(np.count_nonzero(E)), "of", E.size)

# --------------------------------------------------------------------------------------------------------------
section("Part 4: LinearMPC builds the same condensed problem")
mpc = LinearMPC(A, B, Q, R, N, P=P, x_lower=[-4, -3], x_upper=[4, 3], u_lower=[-1], u_upper=[1])
sol = mpc.solve(x0)
print("LinearMPC first inputs", np.round(sol.U[:4, 0], 4), " agree with the hand-built QP to", np.abs(sol.U[:, 0] - Uc).max())

# --------------------------------------------------------------------------------------------------------------
section("Exercise 3: the fourth block row of Sx and Su (N = 4)")
Sx4, Su4 = prediction_matrices(A, B, 4)
print("A^4 =", Sx4[-2:].tolist()); print("row block 4 of Su =", Su4[-2:].tolist())

section("Exercise 6: sizes for n = 8, m = 2, N = 30")
n, m, N30 = 8, 2, 30
print("sparse variables", N30 * n + N30 * m, "+", N30 * n, "equalities; condensed variables", N30 * m)
