/* Advanced MPC and Learning MPC, chapter 09, Capstone: the live demos.
   The numerics are chapter 03's verified block (polytope toolkit of chapter 02, LQR by Riccati iteration, the
   condensed MPC quadratic program with a primal-dual interior-point solver, receding-horizon closed loops), plus
   the condensed QP data of exercise B.1, the feasibility LP of A.3, and the LMPC of chapter 05 with its
   barycentric terminal cost. Everything on the page is recomputed here. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const eye = n => zeros(n, n).map((row, i) => row.map((_, j) => i === j ? 1 : 0));
const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const mv = (A, x) => A.map(r => r.reduce((s, v, k) => s + v * x[k], 0));
const tr = A => A[0].map((_, j) => A.map(r => r[j]));
const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
const sub = (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j]));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const quad = (P, x) => dot(x, mv(P, x));
function solve(A0, b0) {
  const n = A0.length, A = A0.map((r, i) => [...r, b0[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; if (Math.abs(A[c][c]) < 1e-14) A[c][c] = 1e-14;
    for (let r = 0; r < n; r++) { if (r === c) continue; const m = A[r][c] / A[c][c]; if (!m) continue; for (let k = c; k <= n; k++) A[r][k] -= m * A[c][k]; }
  }
  return A.map((r, i) => r[n] / r[i]);
}
const inv = A => tr(A[0].map((_, j) => solve(A, A.map((r, i) => i === j ? 1 : 0))));
function eig2(M) { const t = M[0][0] + M[1][1], d = M[0][0] * M[1][1] - M[0][1] * M[1][0], disc = t * t / 4 - d; return disc >= 0 ? [[t / 2 + Math.sqrt(disc), 0], [t / 2 - Math.sqrt(disc), 0]] : [[t / 2, Math.sqrt(-disc)], [t / 2, -Math.sqrt(-disc)]]; }
function dlqr(A, B, Q, R, iters = 10000) {
  let P = Q;
  for (let it = 0; it < iters; it++) { const BtP = mm(tr(B), P), S = add(R, mm(BtP, B)), K = mm(inv(S), mm(BtP, A)); const Pn = add(Q, sub(mm(tr(A), mm(P, A)), mm(tr(mm(BtP, A)), K))); const diff = Math.max(...sub(Pn, P).flat().map(Math.abs)); P = Pn; if (diff < 1e-13) break; }
  const BtP = mm(tr(B), P), K = mm(inv(add(R, mm(BtP, B))), mm(BtP, A)), Acl = sub(A, mm(B, K));
  return { K, P, Acl, eig: A.length === 2 ? eig2(Acl) : [[Acl[0][0], 0]] };
}
/* ---- polytopes {H, h} in 2-D (chapter 02's toolkit) ---- */
const poly = (H, h) => ({ H, h });
const box = (lo, hi) => poly([[1, 0], [0, 1], [-1, 0], [0, -1]], [hi[0], hi[1], -lo[0], -lo[1]]);
const contains = (P, x, tol = 1e-8) => P.H.every((row, i) => dot(row, x) <= P.h[i] + tol);
function hull(pts) { const P = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]), cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); const lo = [], up = []; for (const p of P) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 1e-12) lo.pop(); lo.push(p); } for (const p of P.slice().reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 1e-12) up.pop(); up.push(p); } return lo.slice(0, -1).concat(up.slice(0, -1)); }
function vertices(P, tol = 1e-8) { const pts = []; for (let i = 0; i < P.H.length; i++) for (let j = i + 1; j < P.H.length; j++) { const det = P.H[i][0] * P.H[j][1] - P.H[i][1] * P.H[j][0]; if (Math.abs(det) < 1e-10) continue; const x = [(P.h[i] * P.H[j][1] - P.H[i][1] * P.h[j]) / det, (P.H[i][0] * P.h[j] - P.h[i] * P.H[j][0]) / det]; if (contains(P, x, tol)) pts.push(x); } const uniq = []; for (const p of pts) if (!uniq.some(q => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-7)) uniq.push(p); return uniq.length < 3 ? uniq : hull(uniq); }
function removeRedundancy(P) { const V = vertices(P); if (V.length < 3) return P; const H = [], h = []; for (let i = 0; i < P.H.length; i++) { const m = Math.max(...V.map(v => dot(P.H[i], v))); if (m > P.h[i] - 1e-7 && !H.some((row, k) => Math.abs(row[0] - P.H[i][0]) < 1e-9 && Math.abs(row[1] - P.H[i][1]) < 1e-9 && Math.abs(h[k] - P.h[i]) < 1e-9)) { H.push(P.H[i]); h.push(P.h[i]); } } return poly(H, h); }
const intersect = (P, Q) => removeRedundancy(poly([...P.H, ...Q.H], [...P.h, ...Q.h]));
const preimage = (S, A) => removeRedundancy(poly(S.H.map(row => mv(tr(A), row)), S.h.slice()));
function predecessor(S, A, B, umin, umax) { const ax = S.H.map(row => mv(tr(A), row)), bu = S.H.map(row => row[0] * B[0][0] + row[1] * B[1][0]), c = S.h.slice(); ax.push([0, 0], [0, 0]); bu.push(1, -1); c.push(umax, -umin); const pos = [], neg = [], zer = []; bu.forEach((b, i) => (b > 1e-12 ? pos : b < -1e-12 ? neg : zer).push(i)); const H = [], h = []; for (const i of zer) { H.push(ax[i]); h.push(c[i]); } for (const n of neg) for (const p of pos) { H.push([ax[p][0] / bu[p] - ax[n][0] / bu[n], ax[p][1] / bu[p] - ax[n][1] / bu[n]]); h.push(c[p] / bu[p] - c[n] / bu[n]); } return removeRedundancy(poly(H, h)); }
const approxEqual = (P, Q, tol = 1e-6) => { const V1 = vertices(P), V2 = vertices(Q); if (!V1.length || !V2.length) return V1.length === V2.length; return V1.every(v => contains(Q, v, tol)) && V2.every(v => contains(P, v, tol)); };
function maximalInvariant(A, X, maxIter = 100) { let Om = removeRedundancy(X); const history = [Om]; for (let i = 0; i < maxIter; i++) { const nxt = intersect(X, preimage(Om, A)); history.push(nxt); if (approxEqual(nxt, Om)) return { set: nxt, history }; Om = nxt; } return { set: Om, history }; }
function maximalControlInvariant(A, B, X, umin, umax, maxIter = 100) { let C = removeRedundancy(X); const history = [C]; for (let i = 0; i < maxIter; i++) { const nxt = intersect(X, predecessor(C, A, B, umin, umax)); history.push(nxt); if (approxEqual(nxt, C)) return { set: nxt, history }; C = nxt; } return { set: C, history }; }
function controllableSets(A, B, X, target, umin, umax, N) { const sets = [intersect(target, X)]; for (let i = 0; i < N; i++) sets.push(intersect(X, predecessor(sets[sets.length - 1], A, B, umin, umax))); return sets; }
const closedLoopAdmissible = (X, K, umin, umax) => intersect(X, poly([[-K[0], -K[1]], [K[0], K[1]]], [umax, -umin]));
const area = V => V.length < 3 ? 0 : Math.abs(V.reduce((s, v, i) => { const w = V[(i + 1) % V.length]; return s + v[0] * w[1] - w[0] * v[1]; }, 0)) / 2;
const zeroTarget = () => poly([[1, 0], [0, 1], [-1, 0], [0, -1]], [0, 0, 0, 0]);
/* ---- a primal-dual interior point for  min 1/2 z'Hz + f'z,  G z <= g,  E z = e ---- */
function solveQP(H, f, G, g, E = [], e = [], iters = 80) {
  const n = f.length, m = g.length, p = e.length; let z = new Array(n).fill(0), lam = new Array(m).fill(1), nu = new Array(p).fill(0), s = g.map((gi, i) => Math.max(gi - dot(G[i], z), 1));
  const Gt = m ? tr(G) : null, Et = p ? tr(E) : null;
  for (let it = 0; it < iters; it++) {
    const rd = mv(H, z).map((v, i) => v + f[i] + (m ? dot(Gt[i], lam) : 0) + (p ? dot(Et[i], nu) : 0)), rp = G.map((row, i) => dot(row, z) + s[i] - g[i]), re = E.map((row, i) => dot(row, z) - e[i]);
    const mu = m ? dot(s, lam) / m : 0, sigma = 0.1, rc = s.map((si, i) => sigma * mu - si * lam[i]);
    if (Math.max(...rd.map(Math.abs), ...rp.map(Math.abs), ...re.map(Math.abs), 0) < 1e-10 && mu < 1e-11) break;
    const D = lam.map((l, i) => l / s[i]), M = zeros(n + p, n + p), rhs = new Array(n + p).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let v = H[i][j]; for (let k = 0; k < m; k++) v += G[k][i] * D[k] * G[k][j]; M[i][j] = v; }
    for (let k = 0; k < p; k++) for (let i = 0; i < n; i++) { M[i][n + k] = E[k][i]; M[n + k][i] = E[k][i]; }
    const t = rc.map((v, i) => (v + lam[i] * rp[i]) / s[i]);
    for (let i = 0; i < n; i++) rhs[i] = -rd[i] - (m ? dot(Gt[i], t) : 0);
    for (let k = 0; k < p; k++) rhs[n + k] = -re[k];
    const sol = solve(M, rhs), dz = sol.slice(0, n), dnu = sol.slice(n), ds = rp.map((v, i) => -v - dot(G[i], dz)), dl = rc.map((v, i) => (v - lam[i] * ds[i]) / s[i]);
    let a = 1; for (let i = 0; i < m; i++) { if (ds[i] < 0) a = Math.min(a, -0.99 * s[i] / ds[i]); if (dl[i] < 0) a = Math.min(a, -0.99 * lam[i] / dl[i]); }
    z = z.map((v, i) => v + a * dz[i]); s = s.map((v, i) => v + a * ds[i]); lam = lam.map((v, i) => v + a * dl[i]); nu = nu.map((v, i) => v + a * dnu[i]);
  }
  return { z, lam, obj: 0.5 * quad(H, z) + dot(f, z), maxViolation: Math.max(...g.map((gi, i) => dot(G[i], z) - gi), ...e.map((ei, i) => Math.abs(dot(E[i], z) - ei)), 0) };
}
function predictionMatrices(A, B, N) { const n = 2, Sx = zeros(N * n, n), Su = zeros(N * n, N); let Ap = eye(n); const powers = [Ap]; for (let i = 1; i <= N; i++) { Ap = mm(A, Ap); powers.push(Ap); } for (let i = 0; i < N; i++) { for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Sx[i * n + r][c] = powers[i + 1][r][c]; for (let j = 0; j <= i; j++) { const blk = mm(powers[i - j], B); for (let r = 0; r < n; r++) Su[i * n + r][j] = blk[r][0]; } } return { Sx, Su }; }
/* the condensed MPC: terminal = {kind: "zero"} | {kind: "set", H, h} | {kind: "none"} */
function makeMPC(A, B, Q, R, N, P, xlb, xub, umin, umax, terminal) {
  const n = 2, { Sx, Su } = predictionMatrices(A, B, N), Qbar = zeros(N * n, N * n);
  for (let i = 0; i < N; i++) { const W = i === N - 1 ? P : Q; for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Qbar[i * n + r][i * n + c] = W[r][c]; }
  const SuTQ = mm(tr(Su), Qbar), H = mm(SuTQ, Su).map((r, i) => r.map((v, j) => 2 * (v + (i === j ? R : 0)))), F = mm(SuTQ, Sx).map(r => r.map(v => 2 * v)), SxTQSx = mm(tr(Sx), mm(Qbar, Sx));
  return { A, B, N, Q, R, P, Sx, Su, terminal, solve(x0) {
    const f = mv(F, x0), Sxx = mv(Sx, x0), G = [], g = [], E = [], e = [];
    for (let r = 0; r < N * n; r++) { const k = r % n; G.push(Su[r]); g.push(xub[k] - Sxx[r]); G.push(Su[r].map(v => -v)); g.push(-xlb[k] + Sxx[r]); }
    for (let j = 0; j < N; j++) { const ej = new Array(N).fill(0); ej[j] = 1; G.push(ej); g.push(umax); G.push(ej.map(v => -v)); g.push(-umin); }
    const SuN = Su.slice(-n), SxNx = Sxx.slice(-n);
    if (terminal.kind === "zero") { E.push(SuN[0], SuN[1]); e.push(-SxNx[0], -SxNx[1]); }
    else if (terminal.kind === "set") terminal.H.forEach((row, i) => { G.push(SuN[0].map((_, j) => row[0] * SuN[0][j] + row[1] * SuN[1][j])); g.push(terminal.h[i] - dot(row, SxNx)); });
    const r = solveQP(H, f, G, g, E, e), X = []; const Xs = Sxx.map((v, i) => v + dot(Su[i], r.z)); for (let i = 0; i < N; i++) X.push([Xs[i * n], Xs[i * n + 1]]);
    const J = quad(Q, x0) + r.obj + quad(SxTQSx, x0);   // the full finite-horizon value: x0'Qx0 + sum_1^{N-1} x'Qx + sum u'Ru + xN'PxN
    return { U: r.z, X, J, feasible: r.maxViolation < 1e-6 };
  } };
}
function closedLoop(mpc, x0, steps = 40, tol = 1e-3) {
  let x = x0.slice(); const xs = [x.slice()], us = [], vals = [], stages = []; let infeasibleAt = -1;
  for (let t = 0; t < steps; t++) { const sol = mpc.solve(x); if (!sol.feasible) { infeasibleAt = t; break; } const u = sol.U[0]; vals.push(sol.J); stages.push(quad(mpc.Q, x) + mpc.R * u * u); us.push(u); x = mv(mpc.A, x).map((v, i) => v + mpc.B[i][0] * u); xs.push(x.slice()); if (Math.hypot(...x) < tol) break; }
  const resid = vals.slice(1).map((v, t) => v - vals[t] + stages[t]);
  return { xs, us, vals, stages, resid, infeasibleAt };
}
/* ---- chapter 09 additions: the condensed QP data, a feasibility LP, the LMPC of chapter 05 on this system ---- */
function condensed(mpc, x0) {   /* exercise B.1: 1/2 U'HU + f'U + c with G U <= w; the rows are named for the KKT widget */
  const n = 2, N = mpc.N, { Sx, Su } = mpc, Qbar = zeros(N * n, N * n);
  for (let i = 0; i < N; i++) { const W = i === N - 1 ? mpc.P : mpc.Q; for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Qbar[i * n + r][i * n + c] = W[r][c]; }
  const SuTQ = mm(tr(Su), Qbar), H = mm(SuTQ, Su).map((r, i) => r.map((v, j) => 2 * (v + (i === j ? mpc.R : 0)))), free = mv(Sx, x0), f = mv(SuTQ, free).map(v => 2 * v), c = quad(mpc.Q, x0) + dot(free, mv(Qbar, free));
  const G = [], w = [], names = [];
  for (let r = 0; r < N * n; r++) { const k = r % n, s = k ? "v" : "p", t = Math.floor(r / n) + 1; G.push(Su[r]); w.push((k ? 3 : 4) - free[r]); names.push(`${s}_${t} <= max`); }
  for (let r = 0; r < N * n; r++) { const k = r % n, s = k ? "v" : "p", t = Math.floor(r / n) + 1; G.push(Su[r].map(v => -v)); w.push((k ? 3 : 4) + free[r]); names.push(`${s}_${t} >= min`); }
  for (let j = 0; j < N; j++) { const ej = new Array(N).fill(0); ej[j] = 1; G.push(ej); w.push(1); names.push(`u_${j} <= 1`); }
  for (let j = 0; j < N; j++) { const ej = new Array(N).fill(0); ej[j] = -1; G.push(ej); w.push(1); names.push(`u_${j} >= -1`); }
  if (mpc.terminal.kind === "set") { const SuN = Su.slice(-n), SxNx = free.slice(-n); mpc.terminal.H.forEach((row, i) => { G.push(SuN[0].map((_, j) => row[0] * SuN[0][j] + row[1] * SuN[1][j])); w.push(mpc.terminal.h[i] - dot(row, SxNx)); names.push(`terminal row ${i}`); }); }
  return { H, f, c, G, w, names };
}
function feasibilityMargin(G, w) {   /* the LP of A.3: smallest s >= 0 with G U <= w + s; s = 0 certifies feasibility */
  const n = G[0].length, Gs = G.map(row => [...row, -1]), gs = w.slice(); Gs.push(new Array(n).fill(0).concat([-1])); gs.push(0);
  const r = solveQP(zeros(n + 1, n + 1), new Array(n).fill(0).concat([1]), Gs, gs, [], [], 120); return Math.max(r.z[n], 0);
}
/* LMPC on this system (chapter 05): variables [U (N), lambda (M)], x_N = S'lambda, cost = stage costs + J'lambda, all stored data */
function lmpcSolve(mpc, store, x0) {
  const n = 2, N = mpc.N, M = store.S.length, nz = N + M, { Sx, Su } = mpc, Sxx = mv(Sx, x0), Qbar = zeros(N * n, N * n);
  for (let k = 0; k < N - 1; k++) for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Qbar[k * n + r][k * n + c] = mpc.Q[r][c];
  const SuTQ = mm(tr(Su), Qbar), Huu = mm(SuTQ, Su), H = zeros(nz, nz), f = new Array(nz).fill(0);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) H[i][j] = 2 * (Huu[i][j] + (i === j ? mpc.R : 0));
  const fu = mv(SuTQ, Sxx); for (let i = 0; i < N; i++) f[i] = 2 * fu[i]; for (let r = 0; r < M; r++) f[N + r] = store.J[r];
  const G = [], g = [];
  for (let r = 0; r < N * n; r++) { const k = r % n, row = new Array(nz).fill(0); for (let j = 0; j < N; j++) row[j] = Su[r][j]; G.push(row); g.push((k ? 3 : 4) - Sxx[r]); G.push(row.map(v => -v)); g.push((k ? 3 : 4) + Sxx[r]); }
  for (let j = 0; j < N; j++) { const row = new Array(nz).fill(0); row[j] = 1; G.push(row); g.push(1); G.push(row.map(v => -v)); g.push(1); }
  for (let r = 0; r < M; r++) { const row = new Array(nz).fill(0); row[N + r] = -1; G.push(row); g.push(0); }
  const E = [], e = []; for (let i = 0; i < n; i++) { const row = new Array(nz).fill(0), r = (N - 1) * n + i; for (let j = 0; j < N; j++) row[j] = Su[r][j]; for (let k = 0; k < M; k++) row[N + k] = -store.S[k][i]; E.push(row); e.push(-Sxx[r]); }
  const one = new Array(nz).fill(0); for (let k = 0; k < M; k++) one[N + k] = 1; E.push(one); e.push(1);
  const r = solveQP(H, f, G, g, E, e, 120), U = r.z.slice(0, N), lam = r.z.slice(N), Xs = Sxx.map((v, i) => v + dot(Su[i], U)), X = [x0]; for (let k = 0; k < N; k++) X.push([Xs[k * n], Xs[k * n + 1]]);
  let obj = quad(mpc.Q, x0) + mpc.R * U[0] * U[0]; for (let k = 1; k < N; k++) obj += quad(mpc.Q, X[k]) + mpc.R * U[k] * U[k]; obj += dot(lam, store.J);
  return { U, lam, X, objective: obj, feasible: r.maxViolation < 1e-6 };
}
const stageCost = (Q, R, x, u) => quad(Q, x) + R * u * u;
function addTrajectory(store, Q, R, X, U) { const J = new Array(X.length).fill(0); for (let k = U.length - 1; k >= 0; k--) J[k] = stageCost(Q, R, X[k], U[k]) + J[k + 1]; for (let k = 0; k < U.length; k++) { store.S.push(X[k]); store.J.push(J[k]); store.U.push(U[k]); } store.costs.push(J[0]); store.trajectories.push(X); }
function completeToOrigin(A, B, X, U) {   /* two-step deadbeat completion [AB B][u0; u1] = -A^2 x, which reaches the origin exactly */
  const x = X[X.length - 1], b = [B[0][0], B[1][0]], AB = mv(A, b), A2x = mv(A, mv(A, x)), det = AB[0] * b[1] - b[0] * AB[1], u0 = (-A2x[0] * b[1] + b[0] * A2x[1]) / det, u1 = (-AB[0] * A2x[1] + AB[1] * A2x[0]) / det;
  let xc = x; for (const u of [u0, u1]) { xc = mv(A, xc).map((v, i) => v + b[i] * u); U.push(u); X.push(xc); } X[X.length - 1] = [0, 0]; return { X, U, bridge: [u0, u1] };
}
function convexSafeValue(x, S, J) { const M = S.length, G = [], g = []; for (let r = 0; r < M; r++) { const row = new Array(M).fill(0); row[r] = -1; G.push(row); g.push(0); } const E = [S.map(s => s[0]), S.map(s => s[1]), new Array(M).fill(1)], e = [x[0], x[1], 1]; const r = solveQP(zeros(M, M), J.slice(), G, g, E, e, 120); return { value: dot(r.z, J), lam: r.z, ok: r.maxViolation < 1e-6 }; }
/* NET-END */

function legend(ctx, lines, x, y, narrow) { ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; const out = narrow ? lines.flatMap(l => l.split(" · ")) : lines; out.forEach((l, i) => ctx.fillText(l, x, y + 13 * i)); }
function drawPoly(fr, V, stroke, width = 1.5, dash = null, fill = null) { if (V.length < 3) return; const { ctx, X, Y } = fr; ctx.beginPath(); V.forEach((v, i) => i ? ctx.lineTo(X(v[0]), Y(v[1])) : ctx.moveTo(X(v[0]), Y(v[1]))); ctx.closePath(); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]); } }
const phaseFrame = (cv, narrow, big) => frame(cv, narrow ? 1.15 : (big ? 1.5 : 1.7), { x: [-4.6, 4.6], y: [-3.5, 3.5], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, -2, -1, 0, 1, 2, 3] });

/* ---- the system and its sets, computed once ---- */
const A = [[1, 1], [0, 1]], B = [[0.5], [1]], Q = [[1, 0], [0, 0.2]], R = 0.2, XLB = [-4, -3], XUB = [4, 3], UMIN = -1, UMAX = 1, XBOX = box(XLB, XUB);
const LQ = dlqr(A, B, Q, [[R]]), K = LQ.K[0], P = LQ.P, RHO = Math.hypot(LQ.eig[0][0], LQ.eig[0][1]);
const XK = closedLoopAdmissible(XBOX, K, UMIN, UMAX), XF_RES = maximalInvariant(LQ.Acl, XK), XF = XF_RES.set, CINF = maximalControlInvariant(A, B, XBOX, UMIN, UMAX).set;
const KXF = controllableSets(A, B, XBOX, XF, UMIN, UMAX, 6), KX = controllableSets(A, B, XBOX, XBOX, UMIN, UMAX, 5);
const VX = vertices(XBOX), VXK = vertices(XK), VXF = vertices(XF), VC = vertices(CINF);
const mpcT = N => makeMPC(A, B, Q, R, N, P, XLB, XUB, UMIN, UMAX, { kind: "set", H: XF.H, h: XF.h }), mpcP = N => makeMPC(A, B, Q, R, N, [[0, 0], [0, 0]], XLB, XUB, UMIN, UMAX, { kind: "none" });
const GRID = []; for (let v = -3; v <= 3.001; v += 0.5) for (let p = -4; p <= 4.001; p += 0.5) GRID.push([Math.round(p * 2) / 2, Math.round(v * 2) / 2]);
const REF12 = mpcT(12), JINF = new Map(); const jinf = x => { const k = x.join(","); if (!JINF.has(k)) JINF.set(k, REF12.solve(x).J); return JINF.get(k); };
const XS = [-3.5, 2.5];
function gridRun(design, N) {   /* every feasible grid start, closed loop for 40 steps */
  const mpc = design === "terminal" ? mpcT(N) : mpcP(N), runs = [];
  for (const x0 of GRID) { const { G, w } = condensed(mpc, x0); if (feasibilityMargin(G, w) > 1e-6) continue; const r = closedLoop(mpc, x0, 40, 1e-3); const inC = contains(CINF, x0, 1e-9), conv = r.infeasibleAt < 0 && Math.hypot(...r.xs[r.xs.length - 1]) <= 1e-3, cost = r.stages.reduce((s, v) => s + v, 0); runs.push({ x0, ...r, doomed: !inC, converged: conv, ratio: conv && inC && Math.hypot(...x0) > 0 ? cost / jinf(x0) : NaN, enter: r.xs.findIndex(x => contains(XF, x, 1e-9)) }); }
  return runs;
}
const CACHE = {}; const runsFor = (design, N) => CACHE[design + N] || (CACHE[design + N] = gridRun(design, N));
function summary(runs) { const feas = runs.length, lost = runs.filter(r => r.infeasibleAt >= 0).length, doomed = runs.filter(r => r.doomed).length, lostC = runs.filter(r => r.infeasibleAt >= 0 && !r.doomed).length, noconv = runs.filter(r => r.infeasibleAt < 0 && !r.converged).length, ratios = runs.map(r => r.ratio).filter(v => !Number.isNaN(v)), resid = Math.max(-Infinity, ...runs.flatMap(r => r.resid)), enter = Math.max(-1, ...runs.map(r => r.enter)); return { feas, lost, doomed, lostC, noconv, worst: ratios.length ? Math.max(...ratios) : NaN, over20: ratios.filter(v => v > 1.2).length, resid, enter }; }

/* =====================================================================
   Hero: the grid certificate
   ===================================================================== */
const hs = { design: "terminal", N: 3, probe: XS.slice() };
function drawHero() {
  const runs = runsFor(hs.design, hs.N), S = summary(runs), cv = $("#gridC"), narrow = narrowOf(cv), fr = phaseFrame(cv, narrow, true), { ctx, X, Y } = fr; cv._fr = fr;
  drawPoly(fr, VX, alpha("--ink", 0.5), 1, [3, 3]); drawPoly(fr, VC, alpha("--ink", 0.6), 2);
  const sets = hs.design === "terminal" ? KXF : KX, FN = sets[Math.min(hs.N, sets.length - 1)]; drawPoly(fr, vertices(FN), css("--post"), 1.6, [6, 3]); drawPoly(fr, VXF, css("--post"), 2);
  runs.forEach(r => { ctx.strokeStyle = r.infeasibleAt >= 0 ? alpha("--warm", 0.8) : alpha("--post", 0.25); ctx.lineWidth = r.infeasibleAt >= 0 ? 1.4 : 0.8; ctx.beginPath(); r.xs.forEach((x, i) => i ? ctx.lineTo(X(x[0]), Y(x[1])) : ctx.moveTo(X(x[0]), Y(x[1]))); ctx.stroke(); });
  runs.forEach(r => { ctx.fillStyle = r.doomed ? css("--warm") : alpha("--ink", 0.7); ctx.beginPath(); ctx.arc(X(r.x0[0]), Y(r.x0[1]), r.doomed ? 3.2 : 2, 0, 2 * Math.PI); ctx.fill(); });
  const pr = runs.find(r => r.x0[0] === hs.probe[0] && r.x0[1] === hs.probe[1]);
  if (pr) { ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2.2; ctx.beginPath(); pr.xs.forEach((x, i) => i ? ctx.lineTo(X(x[0]), Y(x[1])) : ctx.moveTo(X(x[0]), Y(x[1]))); ctx.stroke(); }
  ctx.fillStyle = css("--bg"); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(hs.probe[0]), Y(hs.probe[1]), 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  legend(ctx, ["dots: feasible grid starts (orange = doomed) · grey: C_inf · dashed blue: X_N · blue: X_f · orange lines: lost feasibility · black: the probe"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* value along the probe */
  const cv2 = $("#valC"), fr2 = frame(cv2, narrow ? 1.5 : 2.6, { x: [0, 20], y: [-10, 2], xlabel: "time step t", ylabel: "log10 of the optimal value", xticks: [0, 5, 10, 15, 20], yticks: [-10, -8, -6, -4, -2, 0, 2] }), ctx2 = fr2.ctx;
  if (pr && pr.vals.length) { plotLine(fr2, pr.vals.map((_, t) => t), pr.vals.map(v => Math.log10(Math.max(v, 1e-12))), { color: pr.infeasibleAt >= 0 ? css("--warm") : css("--post"), width: 2.2 }); const v0 = Math.max(pr.vals[0], 1e-12); plotLine(fr2, [0, 20], [Math.log10(v0), Math.log10(v0) + 20 * 2 * Math.log10(RHO)], { color: alpha("--ink", 0.5), width: 1.2, dash: [5, 4] }); }
  legend(ctx2, ["J_N*(x(t)) along the probe's closed loop · dashed: the slope of ρ(A − BK)^{2t}"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  $("#hNV").textContent = hs.N; $$("#demo [data-design]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.design === hs.design)));
  $("#rFeas").textContent = `${S.feas} of ${GRID.length}` + (S.doomed ? ` (${S.doomed} doomed)` : ""); $("#rLost").textContent = String(S.lost); $("#rRes").textContent = Number.isFinite(S.resid) ? S.resid.toExponential(1) : "–"; $("#rWorst").textContent = Number.isNaN(S.worst) ? "–" : fmt(S.worst, 4);
  const nb = hs.design === "terminal" && hs.N === 3 && hs.probe[0] === XS[0] && hs.probe[1] === XS[1];
  let text = `${hs.design === "terminal" ? "MPC with the LQR terminal ingredients" : "Plain MPC, no terminal cost and no terminal set"}, [[N = ${hs.N}]], from the ${S.feas} feasible starts of the ${GRID.length}-state grid${hs.design === "terminal" ? ` (every grid state of [[\\mathcal X_{${hs.N}} = \\mathcal K_{${hs.N}}(\\mathcal X_f)]])` : `, the grid states of [[\\mathcal K_{${hs.N}}(\\mathcal X)]]${S.doomed ? `, of which ${S.doomed} are doomed: feasible now, outside [[\\mathcal C_\\infty]]` : ""}`}. `;
  if (hs.design === "terminal") text += `Every closed loop stayed feasible${S.lost ? " except " + S.lost : ""}, entered [[\\mathcal X_f]] within ${S.enter} steps and converged; the largest decrease residual [[J_N^\\star(x^+) - J_N^\\star(x) + q(x, u)]] over all steps is ${S.resid.toExponential(1)}, zero up to solver tolerance, and the worst closed-loop cost is ${fmt(S.worst, 4)} times [[J_\\infty^\\star]]. `;
  else text += `${S.lost} closed loops lost feasibility (${S.lostC} from starts inside [[\\mathcal C_\\infty]], ${S.lost - S.lostC} from doomed starts), ${S.noconv} stayed feasible without converging in 40 steps; ${Number.isNaN(S.worst) ? "no converged run to compare" : `among the converged runs the worst cost is ${fmt(S.worst, 4)} times [[J_\\infty^\\star]] and ${S.over20} starts pay more than 20 percent extra`}. ${S.resid > 1e-6 ? `The decrease inequality fails: the largest residual is ${S.resid.toExponential(1)}, as it must with [[V_f = 0]], where a longer horizon can only add cost.` : ""} `;
  if (pr) text += `The probe at [[(${fmt(pr.x0[0], 1)}, ${fmt(pr.x0[1], 1)})]]: ${pr.infeasibleAt >= 0 ? `the QP became infeasible at step ${pr.infeasibleAt}` : `${pr.us.length} steps, closed-loop cost ${fmt(pr.stages.reduce((s, v) => s + v, 0), 4)} against [[J_\\infty^\\star = ${fmt(jinf(pr.x0), 4)}]]${pr.enter >= 0 ? `, inside [[\\mathcal X_f]] from step ${pr.enter}` : ""}, largest residual ${Math.max(...pr.resid, -Infinity).toExponential(1)}`}${nb ? ". This is the notebook's start for the comparisons of Parts 4 and 5" : ""}. `;
  else text += "The probe is not a feasible start for this design. ";
  text += "Tap a grid state to move the probe.";
  caption($("#heroStatus"), text);
}
$("#hN").addEventListener("input", () => { hs.N = +$("#hN").value; drawHero(); });
$$("#demo [data-design]").forEach(b => b.addEventListener("click", () => { hs.design = b.dataset.design; drawHero(); }));
$("#gridC").addEventListener("click", ev => { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), fr = cv._fr, x = fr.x0 + (ev.clientX - r.left - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), y = fr.y1 - (ev.clientY - r.top - fr.pad.t) / fr.ih * (fr.y1 - fr.y0); hs.probe = [clamp(Math.round(x * 2) / 2, -4, 4), clamp(Math.round(y * 2) / 2, -3, 3)]; drawHero(); });
$("#hReset").addEventListener("click", () => { hs.design = "terminal"; hs.N = 3; $("#hN").value = 3; hs.probe = XS.slice(); drawHero(); });

/* =====================================================================
   Part 1: the terminal ingredients
   ===================================================================== */
function drawIngredients() {
  const cv = $("#ingC"), narrow = narrowOf(cv), fr = phaseFrame(cv, narrow, false), { ctx, X, Y } = fr;
  drawPoly(fr, VX, alpha("--ink", 0.5), 1, [3, 3]); drawPoly(fr, VC, alpha("--ink", 0.6), 2); drawPoly(fr, VXK, css("--warm"), 1.4, [5, 4]); drawPoly(fr, VXF, css("--post"), 2.4, null, alpha("--post", 0.1));
  VXF.forEach(v => { let x = v; ctx.strokeStyle = alpha("--post", 0.8); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(x[0]), Y(x[1])); for (let i = 0; i < 6; i++) { x = mv(LQ.Acl, x); ctx.lineTo(X(x[0]), Y(x[1])); } ctx.stroke(); });
  legend(ctx, ["dotted: X · grey: C_inf · dashed orange: X_K = {x in X : |Kx| <= 1} · blue: X_f = O_inf with LQR closed loops from its vertices"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const rows = VXF.map(v => { const u = -dot(K, v), xn = mv(LQ.Acl, v); return { v, u, inX: contains(XBOX, v, 1e-9), inU: u >= UMIN - 1e-9 && u <= UMAX + 1e-9, inXf: contains(XF, xn, 1e-9), res: quad(P, xn) - quad(P, v) + quad(Q, v) + R * u * u }; });
  $("#ingT").innerHTML = "<thead><tr><th>vertex (p, v)</th><th>u = −Kx</th><th>x in X</th><th>u in U</th><th>A_K x in X_f</th><th>decrease residual</th></tr></thead><tbody>" + rows.map(r => `<tr><td>(${fmt(r.v[0], 4)}, ${fmt(r.v[1], 4)})</td><td>${fmt(r.u, 4)}</td><td class="words">${r.inX ? "yes" : "no"}</td><td class="words">${r.inU ? "yes" : "no"}</td><td class="words">${r.inXf ? "yes" : "no"}</td><td>${r.res.toExponential(1)}</td></tr>`).join("") + "</tbody>";
  caption($("#ingCap"), `[[K = (${fmt(K[0], 4)}, ${fmt(K[1], 4)})]], [[P = \\begin{pmatrix} ${fmt(P[0][0], 4)} & ${fmt(P[0][1], 4)} \\\\ ${fmt(P[1][0], 4)} & ${fmt(P[1][1], 4)} \\end{pmatrix}]], closed-loop eigenvalues of modulus ${fmt(RHO, 4)}. The invariant-set recursion stops after ${XF_RES.history.length - 1} iterations with a ${VXF.length}-vertex parallelogram whose long edges lie on the lines [[|Kx| = 1]]: the LQR input is [[\\pm 1]] at every vertex, so the input bound, not the state box, limits [[\\mathcal X_f]]. Conditions 1 and 2 hold at every vertex (hence on the whole convex set, since the maps are linear) and the decrease residual of condition 3 is at machine precision, as the DARE identity [[A_K^\\top P A_K - P = -(Q + K^\\top R K)]] predicts. Areas: [[\\mathcal X]] ${fmt(area(VX), 2)}, [[\\mathcal X_K]] ${fmt(area(VXK), 2)}, [[\\mathcal X_f]] ${fmt(area(VXF), 2)}, [[\\mathcal C_\\infty]] ${fmt(area(VC), 2)}: the finite horizon has to steer into a terminal set that covers ${fmt(100 * area(VXF) / area(VC), 0)} percent of the states that could be kept in the box.`);
}

/* =====================================================================
   Part 2: the feasible sets
   ===================================================================== */
function drawFeasible() {
  const cv = $("#fsC"), narrow = narrowOf(cv), fr = phaseFrame(cv, narrow, false), { ctx } = fr;
  drawPoly(fr, VX, alpha("--ink", 0.5), 1, [3, 3]); drawPoly(fr, VC, alpha("--ink", 0.35), 5);
  [0, 1, 2, 3, 5].forEach((N, i) => drawPoly(fr, vertices(KXF[N]), alpha("--post", 0.35 + 0.65 * i / 4), 1.8));
  legend(ctx, ["grey: C_inf · blue, lighter to darker: X_0 = X_f, X_1, X_2, X_3, X_5"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const rows = KXF.map((S, N) => ({ N, a: area(vertices(S)), nested: N === 0 || vertices(KXF[N - 1]).every(v => contains(S, v, 1e-9)), eq: approxEqual(S, CINF, 1e-7) }));
  $("#fsT").innerHTML = "<thead><tr><th>N</th><th>area of X_N</th><th>fraction of C_inf</th><th>X_(N−1) inside X_N</th><th>X_N = C_inf</th></tr></thead><tbody>" + rows.map(r => `<tr><td>${r.N}</td><td>${fmt(r.a, 4)}</td><td>${fmt(r.a / area(VC), 4)}</td><td class="words">${r.nested ? "yes" : "no"}</td><td class="words">${r.eq ? "yes" : "no"}</td></tr>`).join("") + "</tbody>";
  const first = rows.find(r => r.eq);
  caption($("#fsCap"), `The feasible sets [[\\mathcal X_N = \\mathcal K_N(\\mathcal X_f)]] computed with chapter 02's recursion. They are nested and grow quickly: ${fmt(100 * rows[1].a / area(VC), 0)} percent of [[\\mathcal C_\\infty]] for [[N = 1]], ${fmt(100 * rows[3].a / area(VC), 0)} percent for [[N = 3]], and all of it from [[N = ${first ? first.N : "?"}]]. From that horizon on the terminal constraint costs no region of attraction; below it, states of [[\\mathcal C_\\infty \\setminus \\mathcal X_N]] are rejected although some controller could keep them in [[\\mathcal X]]. Each [[\\mathcal X_N]] is control invariant: from a state of [[\\mathcal K_N]] some admissible input reaches [[\\mathcal K_{N-1} \\subseteq \\mathcal K_N]]. That set-level statement is recursive feasibility.`);
}

/* =====================================================================
   Part 4: without terminal ingredients
   ===================================================================== */
const pl = { rows: {}, pending: [1, 2, 3, 4, 5] };
function drawPlain() {
  const cv = $("#plC"), narrow = narrowOf(cv), fr = phaseFrame(cv, narrow, false), { ctx, X, Y } = fr;
  drawPoly(fr, vertices(KX[2]), null, 0, null, alpha("--warm", 0.25)); drawPoly(fr, VC, null, 0, null, css("--bg")); drawPoly(fr, VX, alpha("--ink", 0.5), 1, [3, 3]); drawPoly(fr, vertices(KX[1]), css("--warm"), 1.4, [5, 4]); drawPoly(fr, vertices(KX[2]), css("--warm"), 1.6); drawPoly(fr, VC, alpha("--ink", 0.6), 2.2);
  const ex = [[1, [-1, 2], css("--ink")], [2, [0, 3], css("--warm")]].map(([N, x0, col]) => { const r = closedLoop(mpcP(N), x0, 40, 1e-3); ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.beginPath(); r.xs.forEach((x, i) => i ? ctx.lineTo(X(x[0]), Y(x[1])) : ctx.moveTo(X(x[0]), Y(x[1]))); ctx.stroke(); const xe = r.xs[r.xs.length - 1]; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(xe[0]), Y(xe[1]), 5, 0, 2 * Math.PI); ctx.fill(); return { N, x0, r }; });
  legend(ctx, ["shaded: doomed states of K_2(X) · dashed orange: K_1(X) · orange: K_2(X) · grey: C_inf · black: plain N = 1 from (−1, 2) · orange: plain N = 2 from (0, 3)"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const rows = [1, 2, 3, 4, 5].map(N => pl.rows[N] ? { N, ...pl.rows[N] } : { N });
  $("#plT").innerHTML = "<thead><tr><th>N</th><th>area K_N(X)</th><th>doomed area</th><th>feasible starts</th><th>doomed starts</th><th>lost, in C_inf</th><th>lost, doomed</th><th>no convergence</th><th>worst cost / J_inf</th><th>starts > 1.2 J_inf</th></tr></thead><tbody>" + rows.map(r => `<tr><td>${r.N}</td><td>${fmt(area(vertices(KX[r.N])), 2)}</td><td>${fmt(area(vertices(KX[r.N])) - area(VC), 2)}</td>${r.feas === undefined ? "<td colspan=\"7\" class=\"words\">computing…</td>" : `<td>${r.feas}</td><td>${r.doomed}</td><td>${r.lostC}</td><td>${r.lost - r.lostC}</td><td>${r.noconv}</td><td>${Number.isNaN(r.worst) ? "–" : fmt(r.worst, 4)}</td><td>${r.over20}</td>`}</tr>`).join("") + "</tbody>";
  const e1 = ex[0].r, e2 = ex[1].r;
  caption($("#plCap"), `The plain controller's feasible set [[\\mathcal K_N(\\mathcal X)]] shrinks towards [[\\mathcal C_\\infty]] and contains doomed states for short horizons: a doomed area of ${fmt(area(vertices(KX[1])) - area(VC), 2)} at [[N = 1]] and ${fmt(area(vertices(KX[2])) - area(VC), 2)} at [[N = 2]] (the shaded slivers), none from [[N = 3]]. From [[(-1, 2)]] with [[N = 1]] the controller coasts ([[u_0 = 0]] minimises [[q(x_0, u_0)]] whenever [[Ax \\in \\mathcal X]]) until the wall, ${e1.infeasibleAt >= 0 ? `and the QP is infeasible at step ${e1.infeasibleAt}, at [[(${fmt(e1.xs[e1.xs.length - 1][0], 1)}, ${fmt(e1.xs[e1.xs.length - 1][1], 1)})]], where one braking step is not enough` : "and converges"}; from the doomed start [[(0, 3)]] with [[N = 2]] ${e2.infeasibleAt >= 0 ? `feasibility is lost at step ${e2.infeasibleAt}` : "it survives"}. The table runs the plain controller from every feasible grid start for [[N = 1, \\ldots, 5]], computed here in the background${pl.pending.length ? " (still running)" : ""}: ${pl.rows[1] ? `with [[N = 1]], ${pl.rows[1].noconv} feasible starts never converge and ${pl.rows[1].lostC} starts inside [[\\mathcal C_\\infty]] lose feasibility; ` : ""}${pl.rows[2] ? `with [[N = 2]] both doomed starts fail and the converged runs cost up to ${fmt(pl.rows[2].worst, 2)} times [[J_\\infty^\\star]], ${pl.rows[2].over20} of them more than 20 percent extra; ` : ""}${pl.rows[3] && pl.rows[5] ? `from [[N = 3]] nothing visibly goes wrong, worst ratios ${fmt(pl.rows[3].worst, 4)}, ${fmt(pl.rows[4].worst, 4)} and ${fmt(pl.rows[5].worst, 4)}. ` : ""}What is lost is the certificate: with [[V_f = 0]], [[J_N^0(x^+) - J_N^0(x) + q(x, u_0^\\star)]] equals [[J_N^0(x^+) - J_{N-1}^0(x^+) \\ge 0]]: the decrease inequality with the wrong sign.`);
}
function plainNext() { if (!pl.pending.length) return; const N = pl.pending.shift(); pl.rows[N] = summary(runsFor("plain", N)); drawPlain(); setTimeout(plainNext, 30); }

/* =====================================================================
   Part 5: Learning MPC from the poor trajectory
   ===================================================================== */
const lm = { N: 4, runs: {}, running: null };
function seedTrajectory() { if (lm.seed) return lm.seed; const r = closedLoop(mpcP(2), XS, 40, 1e-3), X = r.xs.map(x => x.slice()), U = r.us.slice(); const c = completeToOrigin(A, B, X, U); lm.seed = { X: c.X, U: c.U, bridge: c.bridge }; return lm.seed; }
function lmpcRun(N, done) {
  const seed = seedTrajectory(), store = { S: [], J: [], U: [], costs: [], trajectories: [] }, mpc = mpcT(N); addTrajectory(store, Q, R, seed.X, seed.U);
  const res = { store, N, iterations: [], failed: null }; lm.runs[N] = res; let j = 0;
  const step = () => { if (j >= 10) { lm.running = null; done(true); return; } let x = XS.slice(); const X = [x], U = []; for (let t = 0; t < 60 && Math.hypot(...x) > 1e-3; t++) { const s = lmpcSolve(mpc, store, x); if (!s.feasible) { res.failed = `iteration ${j + 1}, step ${t}`; lm.running = null; done(true); return; } x = mv(A, x).map((v, i) => v + B[i][0] * s.U[0]); X.push(x); U.push(s.U[0]); } const c = completeToOrigin(A, B, X, U); addTrajectory(store, Q, R, c.X, c.U); res.iterations.push({ X: c.X, U: c.U }); j++; done(false); setTimeout(step, 10); };
  lm.running = N; setTimeout(step, 10);
}
function drawLMPC() {
  const res = lm.runs[lm.N], JS = jinf(XS), cv = $("#lmC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.5 : 2.4, { x: [0, 10], y: [-9, 0.5], xlabel: "iteration j (0 = first trajectory)", ylabel: "log10 of (J^j − J*) / J*", xticks: [0, 2, 4, 6, 8, 10], yticks: [-8, -6, -4, -2, 0] }), { ctx } = fr;
  [2, 3, 4].forEach((N, i) => { const r = lm.runs[N]; if (!r) return; const g = r.store.costs.map(c => Math.log10(Math.max((c - JS) / JS, 1e-9))); plotLine(fr, g.map((_, j) => j), g, { color: alpha("--post", 0.35 + 0.65 * i / 2), width: N === lm.N ? 2.6 : 1.4 }); });
  legend(ctx, ["LMPC from the plain N = 2 trajectory, horizons 2, 3, 4 (darker = longer; thick = selected)"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const cv2 = $("#lmTraj"), fr2 = phaseFrame(cv2, narrow, false), c2 = fr2.ctx, X2 = fr2.X, Y2 = fr2.Y;
  drawPoly(fr2, VC, alpha("--ink", 0.6), 2); drawPoly(fr2, VXF, css("--post"), 1.4);
  if (res) { const hullPts = hull(res.store.trajectories.flat()); drawPoly(fr2, hullPts, null, 0, null, alpha("--post", 0.1)); }
  const seed = seedTrajectory(); c2.strokeStyle = css("--warm"); c2.lineWidth = 1.8; c2.beginPath(); seed.X.forEach((x, i) => i ? c2.lineTo(X2(x[0]), Y2(x[1])) : c2.moveTo(X2(x[0]), Y2(x[1]))); c2.stroke();
  const opt = closedLoop(REF12, XS, 12, 1e-6); c2.strokeStyle = alpha("--ink", 0.7); c2.setLineDash([5, 4]); c2.lineWidth = 1.5; c2.beginPath(); opt.xs.forEach((x, i) => i ? c2.lineTo(X2(x[0]), Y2(x[1])) : c2.moveTo(X2(x[0]), Y2(x[1]))); c2.stroke(); c2.setLineDash([]);
  if (res && res.iterations.length) { const last = res.iterations[res.iterations.length - 1]; c2.strokeStyle = css("--post"); c2.lineWidth = 2.2; c2.beginPath(); last.X.forEach((x, i) => i ? c2.lineTo(X2(x[0]), Y2(x[1])) : c2.moveTo(X2(x[0]), Y2(x[1]))); c2.stroke(); }
  legend(c2, ["orange: first trajectory (plain N = 2) · blue: last LMPC iteration · dashed: optimal · shaded: convex safe set"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  $$("#lmW [data-n]").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.n === lm.N)));
  const seedCost = seed.U.reduce((s, u, k) => s + stageCost(Q, R, seed.X[k], u), 0);
  let text = `[[J_\\infty^\\star(x_S) = ${fmt(JS, 6)}]] at [[x_S = (-3.5, 2.5)]], from the terminal design with [[N = 12]]. The first trajectory is the plain [[N = 2]] closed loop of Part 4, stopped at [[\\|x\\| \\le 10^{-3}]] and completed to the origin by the two deadbeat inputs ${seed.bridge.map(u => u.toExponential(1)).join(", ")}: ${seed.U.length} steps, cost ${fmt(seedCost, 6)}, that is ${fmt(seedCost / JS, 2)} times the optimum. `;
  if (res) { const c = res.store.costs, inc = Math.max(-Infinity, ...c.slice(1).map((v, j) => v - c[j])); text += `LMPC with [[N = ${lm.N}]] and all stored data${lm.running === lm.N ? `, iteration ${res.iterations.length + 1} of 10 running` : ""}: after ${res.iterations.length} iteration${res.iterations.length === 1 ? "" : "s"} the cost is ${fmt(c[c.length - 1], 6)}, ${((c[c.length - 1] - JS) / JS).toExponential(2)} above [[J_\\infty^\\star]] relatively; the largest increase between iterations is ${Number.isFinite(inc) ? inc.toExponential(1) : "none yet"} (guaranteed non-positive up to solver tolerance); ${res.store.S.length} stored states${res.failed ? `; then it failed at ${res.failed}` : ""}. `; }
  const done = [2, 3, 4].filter(N => lm.runs[N] && lm.runs[N].iterations.length === 10);
  if (done.length) text += `Final relative gaps: ${done.map(N => `[[N = ${N}]]: ${((lm.runs[N].store.costs[10] - JS) / JS).toExponential(2)}`).join(", ")}. `;
  text += `One iteration removes almost all of the excess for every horizon; then the horizons part ways. With [[N = 4]] the gap keeps falling and the trajectory joins the optimal one; with [[N = 2]] and [[N = 3]] it stalls after the first iteration, and the notebook's brute-force search confirms that the plateau is not a solver artefact: the best [[N = 2]] plan at [[x_S]] with the data of iteration 10 costs exactly the stored trajectory's cost. The mechanism is the terminal cost: to improve, a plan must end elsewhere in [[CS^j]], where [[V_c^j]] interpolates the costs of suboptimal trajectories and over-estimates the optimal cost-to-go. A gain that appears only later in the tail is invisible to a short plan.`;
  caption($("#lmCap"), text);
}
$$("#lmW [data-n]").forEach(b => b.addEventListener("click", () => { lm.N = +b.dataset.n; if (!lm.runs[lm.N] && !lm.running) lmpcRun(lm.N, () => { drawLMPC(); setTimeout(drawB6, 20); }); drawLMPC(); }));

/* =====================================================================
   Part 6: the KKT conditions at x_S
   ===================================================================== */
function drawKKT() {
  const N = +$("#kkN").value, mpc = mpcT(N), { H, f, c, G, w, names } = condensed(mpc, XS), r = solveQP(H, f, G, w, [], [], 120), U = r.z, J = 0.5 * quad(H, U) + dot(f, U) + c;
  const grad = mv(H, U).map((v, i) => v + f[i]), active = names.map((_, i) => i).filter(i => w[i] - dot(G[i], U) <= 1e-7), mu = active.map(i => r.lam[i]);
  const stat = Math.sqrt(grad.reduce((s, v, i) => s + (v + active.reduce((t, a, k) => t + G[a][i] * mu[k], 0)) ** 2, 0));
  let price = null; if (active.length) { const k = mu.indexOf(Math.max(...mu)), i = active[k], eps = 1e-4, w2 = w.slice(); w2[i] += eps; const r2 = solveQP(H, f, G, w2, [], [], 120), J2 = 0.5 * quad(H, r2.z) + dot(f, r2.z) + c; price = { name: names[i], mu: mu[k], dJ: J2 - J, pred: -mu[k] * eps }; }
  $("#kkNV").textContent = N;
  let ev = Infinity; { const M = H.map(row => row.slice()); for (let s = 0; s < 50; s++) for (let p = 0; p < N; p++) for (let q = p + 1; q < N; q++) { if (Math.abs(M[p][q]) < 1e-14) continue; const th = (M[q][q] - M[p][p]) / (2 * M[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), cs = 1 / Math.sqrt(t * t + 1), sn = t * cs; for (let k = 0; k < N; k++) { const a = M[k][p], b = M[k][q]; M[k][p] = cs * a - sn * b; M[k][q] = sn * a + cs * b; } for (let k = 0; k < N; k++) { const a = M[p][k], b = M[q][k]; M[p][k] = cs * a - sn * b; M[q][k] = sn * a + cs * b; } } ev = Math.min(...M.map((row, i) => row[i])); }
  caption($("#kkCap"), `The condensed QP at [[x_S]] with [[N = ${N}]] has ${G.length} constraint rows and a Hessian whose smallest eigenvalue is ${fmt(ev, 3)}, so the minimiser is unique: U* = (${U.map(u => fmt(u, 4)).join(", ")}), [[J_${N}^\\star(x_S) = ${fmt(J, 6)}]]. Active constraints: ${active.length ? active.map((i, k) => `${names[i]} with multiplier ${fmt(mu[k], 4)}`).join("; ") : "none"}; stationarity residual [[\\|HU^\\star + f + G_A^\\top \\mu\\|]] of ${stat.toExponential(1)}. ${price ? `A multiplier is the price of its constraint: relaxing "${price.name}" by [[10^{-4}]] changes the optimal cost by ${price.dJ.toExponential(4)}, against [[-\\mu_i \\varepsilon = ${price.pred.toExponential(4)}]].` : ""} ${N === 5 ? "This is the notebook's computation, which finds the single active constraint [[u_1 \\ge -1]] with multiplier 0.4944." : ""} Without inequalities the KKT system of the uncondensed problem reduces to the Riccati recursion, and with [[P]] from the DARE [[P_k = P]] for all [[k]]: that is why the MPC equals the LQR inside [[\\mathcal X_f]].`);
}
$("#kkN").addEventListener("input", drawKKT);

/* =====================================================================
   Part 6: the barycentric terminal cost, checked on random points
   ===================================================================== */
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
function drawB6() {
  const res = lm.runs[lm.N]; if (!res || res.iterations.length < 10) { caption($("#b6Cap"), "Waiting for the LMPC run of Part 5 to finish…"); return; }
  const { S, J, U } = res.store, r = rng(7), resid = [], excess = [], rel = [];
  for (let k = 0; k < 40; k++) { const idx = []; while (idx.length < 3) { const i = Math.floor(r() * S.length); if (!idx.includes(i)) idx.push(i); } const a = [r(), r(), r()], t = a.reduce((s, v) => s + v, 0), lam0 = a.map(v => v / t), x = [0, 1].map(d => idx.reduce((s, i, k) => s + lam0[k] * S[i][d], 0)); const v = convexSafeValue(x, S, J); if (!v.ok) continue; const u = dot(v.lam, U), xn = mv(A, x).map((q, i) => q + B[i][0] * u), vn = convexSafeValue(xn, S, J); if (!vn.ok) continue; resid.push(vn.value - v.value + stageCost(Q, R, x, u)); const Jx = jinf(x); excess.push(v.value - Jx); if (Jx >= 0.1) rel.push((v.value - Jx) / Jx); }
  rel.sort((a, b) => a - b);
  caption($("#b6Cap"), `With the ${S.length} stored states of the [[N = ${lm.N}]] run, ${resid.length} random convex combinations of three stored states: the decrease residual [[V_c(x^+) - V_c(x) + q(x, u)]] with [[u = \\sum_i \\lambda_i u_i]] is at most ${Math.max(...resid).toExponential(1)} (inequality (b) holds to round-off), [[V_c - J_\\infty^\\star]] is at least ${Math.min(...excess).toExponential(1)} (never below the optimum), and between stored states [[V_c]] over-estimates [[J_\\infty^\\star]] by a median of ${fmt(100 * rel[Math.floor(rel.length / 2)], 0)} percent over the ${rel.length} points with [[J_\\infty^\\star \\ge 0.1]]. The notebook's random points give medians of 57 percent with the [[N = 2]] data and 77 percent with the [[N = 4]] data. LMPC still reaches the optimum with [[N = 4]] because [[V_c]] only needs to be accurate near the optimal trajectory, where the stored states accumulate.`);
}

function drawAll() { drawHero(); drawIngredients(); drawFeasible(); drawPlain(); drawLMPC(); drawKKT(); drawB6(); }
drawAll();
setTimeout(plainNext, 50);
lmpcRun(4, () => { drawLMPC(); setTimeout(drawB6, 20); });
onRedraw(() => { drawHero(); drawIngredients(); drawFeasible(); drawPlain(); drawLMPC(); });
})();
