/* Advanced MPC and Learning MPC: the Learning MPC toolkit shared by chapters A12, A13 and B2.
   Generic over the task (A, B, Q, R, an optional l1 weight q1, box bounds, the start x_S, target 0):
   stage cost and roll-out cost-to-go, the finite-horizon problem without terminal ingredients (the seed
   controller and the long-horizon reference), the safe-set database with a trajectory window, the LMPC
   quadratic program with barycentric terminal variables, the barycentric terminal value as a linear
   program, the exact two-step bridge to the origin, and the closed-loop runs. Mirrors the course
   package's lmpc.py. Needs mpc-lib.js. */
(function (global) {
"use strict";
const { zeros, mv, mm, tr, dot, solve, solveQP, predictionMatrices } = global.MPC;

/* a task: the system, the cost, the bounds and the repeated start */
function makeTask(o) {
  const t = Object.assign({ q1: 0, n: 2 }, o);
  t.n = t.A.length;
  return t;
}
const stageCost = (t, x, u) => dot(x, mv(t.Q, x)) + t.R * u * u + t.q1 * x.reduce((s, v) => s + Math.abs(v), 0);
function costToGo(t, X, U) { const J = new Array(X.length).fill(0); for (let k = U.length - 1; k >= 0; k--) J[k] = stageCost(t, X[k], U[k]) + J[k + 1]; return J; }

/* the common QP pieces for a horizon N from x0: stage costs on x_1..x_{N-1} (x_0 is a constant, x_N is priced
   by the terminal value or not at all), state bounds on x_1..x_N, input bounds, and when q1 > 0 the l1 epigraph
   variables e >= |q1 x_k| for k = 1..N-1. `extra` variables (the lambdas) sit between U and the epigraph block. */
function buildCore(t, x0, N, extra) {
  const n = t.n, { Sx, Su } = predictionMatrices(t.A, t.B, N), nu = N, ne = t.q1 > 0 ? (N - 1) * n : 0, nz = nu + extra + ne, Sxx = mv(Sx, x0);
  const Qbar = zeros(N * n, N * n); for (let k = 0; k < N - 1; k++) for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Qbar[k * n + r][k * n + c] = t.Q[r][c];
  const H = zeros(nz, nz), f = new Array(nz).fill(0), SuTQ = mm(tr(Su), Qbar), Huu = mm(SuTQ, Su);
  for (let i = 0; i < nu; i++) for (let j = 0; j < nu; j++) H[i][j] = 2 * (Huu[i][j] + (i === j ? t.R : 0));
  const fu = mv(SuTQ, Sxx).map(v => 2 * v); for (let i = 0; i < nu; i++) f[i] = fu[i]; for (let i = 0; i < ne; i++) f[nu + extra + i] = 1;
  const G = [], g = [];
  for (let r = 0; r < N * n; r++) { const k = r % n, row = new Array(nz).fill(0); for (let j = 0; j < nu; j++) row[j] = Su[r][j]; G.push(row); g.push(t.xub[k] - Sxx[r]); G.push(row.map(v => -v)); g.push(-t.xlb[k] + Sxx[r]); }
  if (ne) for (let k = 1; k < N; k++) for (let i = 0; i < n; i++) { const r = (k - 1) * n + i, eidx = nu + extra + (k - 1) * n + i; const row = new Array(nz).fill(0); for (let j = 0; j < nu; j++) row[j] = t.q1 * Su[r][j]; row[eidx] = -1; G.push(row); g.push(-t.q1 * Sxx[r]); const row2 = row.map(v => -v); row2[eidx] = -1; G.push(row2); g.push(t.q1 * Sxx[r]); }
  const lb = new Array(nz).fill(-Infinity), ub = new Array(nz).fill(Infinity); for (let j = 0; j < nu; j++) { lb[j] = t.umin; ub[j] = t.umax; } for (let i = 0; i < ne; i++) lb[nu + extra + i] = 0;
  return { Sx, Su, Sxx, nu, ne, nz, H, f, G, g, lb, ub };
}
const unstack = (t, x0, c, U, N) => { const Xp = c.Sxx.map((v, i) => v + dot(c.Su[i], U)), X = []; for (let k = 0; k < N; k++) X.push(Xp.slice(k * t.n, (k + 1) * t.n)); return X; };

/* the finite-horizon problem with no terminal ingredients: the seed controller and the long-horizon reference */
function solveFTOCP(t, x0, N) {
  const c = buildCore(t, x0, N, 0), r = solveQP(c.H, c.f, c.G, c.g, [], [], c.lb, c.ub), U = r.z.slice(0, N), X = [x0, ...unstack(t, x0, c, U, N)];
  let obj = 0; for (let k = 0; k < N; k++) obj += stageCost(t, X[k], U[k]);
  return { success: r.success, U, X, objective: obj, nit: r.nit };
}

/* the safe-set database */
function makeStore() { return { trajectories: [], inputs: [], qfun: [], costs: [] }; }
function addTrajectory(t, store, X, U) { const J = costToGo(t, X, U); store.trajectories.push(X); store.inputs.push(U); store.qfun.push(J); store.costs.push(J[0]); }
/* the target first (value 0, continuation u = 0 to itself), then every non-terminal state of the trajectories in the window,
   each with its stored cost-to-go, its continuation input and the index of its stored successor */
function safeData(store, window) {
  const n = store.trajectories.length, start = window == null || window === 0 ? 0 : Math.max(0, n - Math.max(window, 1)), S = [[0, 0]], V = [0], Us = [0], next = [0], from = [[-1, -1]];
  for (let it = start; it < n; it++) { const X = store.trajectories[it], U = store.inputs[it], J = store.qfun[it], T = U.length, base = S.length; for (let k = 0; k < T; k++) { S.push(X[k]); V.push(J[k]); Us.push(U[k]); next.push(k < T - 1 ? base + k + 1 : 0); from.push([it, k]); } }
  return { S, V, Us, next, from };
}

/* the LMPC problem: variables [U (N), lambda (M), e]; x_N = S' lambda, 1' lambda = 1, lambda >= 0, cost + lambda' V */
function lmpcSolve(t, store, x0, N, window) {
  const { S, V, Us, next, from } = safeData(store, window), M = S.length, c = buildCore(t, x0, N, M), nu = c.nu, n = t.n;
  for (let r = 0; r < M; r++) { c.f[nu + r] = V[r]; c.lb[nu + r] = 0; c.ub[nu + r] = 1; }
  const E = [], e = []; for (let i = 0; i < n; i++) { const row = new Array(c.nz).fill(0), r = (N - 1) * n + i; for (let j = 0; j < nu; j++) row[j] = c.Su[r][j]; for (let k = 0; k < M; k++) row[nu + k] = -S[k][i]; E.push(row); e.push(-c.Sxx[r]); }
  const one = new Array(c.nz).fill(0); for (let k = 0; k < M; k++) one[nu + k] = 1; E.push(one); e.push(1);
  const r = solveQP(c.H, c.f, c.G, c.g, E, e, c.lb, c.ub), U = r.z.slice(0, N), lam = r.z.slice(nu, nu + M), X = unstack(t, x0, c, U, N);
  let obj = 0; const Xf = [x0, ...X]; for (let k = 0; k < N; k++) obj += stageCost(t, Xf[k], U[k]); obj += dot(lam, V);
  return { success: r.success, U, X, lam, objective: obj, S, V, Us, next, from, nit: r.nit, terminal: Xf[N].map((_, d) => dot(lam, S.map(s => s[d]))) };
}

/* the barycentric terminal value as a linear program (H = 0), returning the weights it chose */
function convexSafeValue(x, S, V) { const M = S.length, H = zeros(M, M), E = [...x.map((_, d) => S.map(s => s[d])), new Array(M).fill(1)], e = [...x, 1]; const r = solveQP(H, V.slice(), [], [], E, e, new Array(M).fill(0), new Array(M).fill(1)); return { value: dot(r.z, V), lam: r.z, success: r.success }; }

/* the exact two-step bridge to the origin: u = -[AB B]^{-1} A^2 x, so that stored trajectories end exactly at the target */
function closeExactly(t, X, U) {
  let x = X[X.length - 1]; if (Math.hypot(...x) > 1e-2) throw new Error("bridge requested too far from the target");
  const AB = mv(t.A, t.B.map(r => r[0])), M = [[AB[0], t.B[0][0]], [AB[1], t.B[1][0]]], A2x = mv(t.A, mv(t.A, x)), uu = solve(M, A2x.map(v => -v));
  for (const u of uu) { x = mv(t.A, x).map((v, i) => v + t.B[i][0] * u); U.push(u); X.push(x); }
  X[X.length - 1] = [0, 0]; return [X, U];
}
/* a run of plain receding-horizon MPC with horizon N and no terminal ingredients, bridged to the target */
function runSeed(t, N, maxSteps = 200, tol = 1e-3) {
  let x = t.xS.slice(); const X = [x], U = [];
  for (let s = 0; s < maxSteps; s++) { if (Math.hypot(...x) <= tol) break; const r = solveFTOCP(t, x, N); if (!r.success) return { X, U, failed: "infeasible" }; const u = r.U[0]; x = mv(t.A, x).map((v, i) => v + t.B[i][0] * u); U.push(u); X.push(x); }
  if (Math.hypot(...X[X.length - 1]) > tol) return { X, U, failed: "did not reach the target" };
  closeExactly(t, X, U); return { X, U };
}
/* a run of any state-feedback law u = law(x) (clipped to the input bounds), bridged to the target */
function runLaw(t, law, maxSteps = 200, tol = 1e-3) {
  let x = t.xS.slice(); const X = [x], U = [];
  for (let s = 0; s < maxSteps; s++) { if (Math.hypot(...x) <= tol) break; const u = Math.min(t.umax, Math.max(t.umin, law(x))); x = mv(t.A, x).map((v, i) => v + t.B[i][0] * u); if (x.some((v, i) => v > t.xub[i] + 1e-9 || v < t.xlb[i] - 1e-9)) return { X, U, failed: "left the box" }; U.push(u); X.push(x); }
  if (Math.hypot(...X[X.length - 1]) > tol) return { X, U, failed: "did not reach the target" };
  closeExactly(t, X, U); return { X, U };
}
/* one LMPC iteration from x_S, added to the store on success */
function runIteration(t, store, N, window, maxSteps = 200, tol = 1e-3) {
  let x = t.xS.slice(); const X = [x], U = []; let first = null, nits = 0;
  for (let s = 0; s < maxSteps; s++) { if (Math.hypot(...x) <= tol) break; const r = lmpcSolve(t, store, x, N, window); if (!r.success) return { X, U, failed: "LMPC QP infeasible at step " + s }; if (!first) first = r; nits += r.nit; const u = r.U[0]; x = mv(t.A, x).map((v, i) => v + t.B[i][0] * u); U.push(u); X.push(x); }
  if (Math.hypot(...X[X.length - 1]) > tol) return { X, U, failed: "did not reach the target" };
  closeExactly(t, X, U); addTrajectory(t, store, X, U); return { X, U, first, cost: store.costs[store.costs.length - 1], nits };
}

global.LMPC = { makeTask, stageCost, costToGo, buildCore, solveFTOCP, makeStore, addTrajectory, safeData, lmpcSolve, convexSafeValue, closeExactly, runSeed, runLaw, runIteration };
})(window);
