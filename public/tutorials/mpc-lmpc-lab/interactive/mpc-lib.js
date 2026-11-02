/* Advanced MPC and Learning MPC: the numerical toolkit shared by every chapter's demos.
   Plain JavaScript, no dependencies: dense linear algebra, the discrete Riccati equation,
   exact 2-D polytopes (vertices, redundancy, intersection, images, preimages, Minkowski
   sums, Fourier–Motzkin predecessors, invariant and controllable sets), a primal-dual
   interior-point QP solver with an active-set polish, the condensed linear MPC and its
   closed loop. The algorithms mirror the course package (advanced_mpc_lmpc), so the pages
   compute what the Python computes. Loaded after ../kit/kit.js, before interactive/<id>.js. */
(function (global) {
"use strict";

/* ---------------- dense linear algebra ---------------- */
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const eye = n => zeros(n, n).map((row, i) => row.map((_, j) => i === j ? 1 : 0));
const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const mv = (A, x) => A.map(r => r.reduce((s, v, k) => s + v * x[k], 0));
const tr = A => A[0].map((_, j) => A.map(r => r[j]));
const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
const sub = (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j]));
const scale = (A, s) => A.map(r => r.map(v => v * s));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const quad = (P, x) => dot(x, mv(P, x));
const norm = x => Math.hypot(...x);
const matpow = (A, k) => { let R = eye(A.length); for (let i = 0; i < k; i++) R = mm(R, A); return R; };
/* Gaussian elimination with partial pivoting; A x = b */
function solve(A0, b0) {
  const n = A0.length, A = A0.map((r, i) => [...r, b0[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; if (Math.abs(A[c][c]) < 1e-14) A[c][c] = 1e-14;
    const pivot = A[c];
    for (let r = c + 1; r < n; r++) { const m = A[r][c] / pivot[c]; if (!m) continue; const row = A[r]; for (let k = c; k <= n; k++) row[k] -= m * pivot[k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = A[r][n]; for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k]; x[r] = s / A[r][r]; }
  return x;
}
const inv = A => tr(A[0].map((_, j) => solve(A, A.map((r, i) => i === j ? 1 : 0))));
/* eigenvalues of a 2x2 matrix as [[re, im], [re, im]] */
function eig2(M) { const t = M[0][0] + M[1][1], d = M[0][0] * M[1][1] - M[0][1] * M[1][0], disc = t * t / 4 - d; return disc >= 0 ? [[t / 2 + Math.sqrt(disc), 0], [t / 2 - Math.sqrt(disc), 0]] : [[t / 2, Math.sqrt(-disc)], [t / 2, -Math.sqrt(-disc)]]; }
/* eigenvalues of a symmetric 2x2 matrix, ascending */
const eigsym2 = M => { const t = M[0][0] + M[1][1], d = M[0][0] * M[1][1] - M[0][1] * M[1][0], s = Math.sqrt(Math.max(0, t * t / 4 - d)); return [t / 2 - s, t / 2 + s]; };

/* ---------------- LQR: the discrete Riccati equation by iteration, u = -K x ---------------- */
function dlqr(A, B, Q, R, iters = 10000) {
  let P = Q;
  for (let it = 0; it < iters; it++) {
    const BtP = mm(tr(B), P), S = add(R, mm(BtP, B)), K = mm(inv(S), mm(BtP, A));
    const Pn = add(Q, sub(mm(tr(A), mm(P, A)), mm(tr(mm(BtP, A)), K)));
    const diff = Math.max(...sub(Pn, P).flat().map(Math.abs)); P = Pn; if (diff < 1e-14) break;
  }
  const BtP = mm(tr(B), P), K = mm(inv(add(R, mm(BtP, B))), mm(BtP, A)), Acl = sub(A, mm(B, K));
  return { K, P, Acl, eig: A.length === 2 ? eig2(Acl) : [[Acl[0][0], 0]] };
}
/* the finite-horizon Riccati recursion from a terminal weight, returning every P_k and K_k */
function riccatiRecursion(A, B, Q, R, Pf, N) {
  const Ps = [Pf], Ks = [];
  let P = Pf;
  for (let k = 0; k < N; k++) {
    const BtP = mm(tr(B), P), K = mm(inv(add(R, mm(BtP, B))), mm(BtP, A));
    P = add(Q, sub(mm(tr(A), mm(P, A)), mm(tr(mm(BtP, A)), K)));
    Ps.unshift(P); Ks.unshift(K);
  }
  return { Ps, Ks };
}

/* ---------------- polytopes {H, h} = {x : H x <= h} in two dimensions ---------------- */
const poly = (H, h) => ({ H, h });
const box = (lo, hi) => poly([[1, 0], [0, 1], [-1, 0], [0, -1]], [hi[0], hi[1], -lo[0], -lo[1]]);
const contains = (P, x, tol = 1e-8) => P.H.every((row, i) => dot(row, x) <= P.h[i] + tol);
/* Andrew's monotone chain: counter-clockwise hull without collinear points */
function hull(pts) {
  const P = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]), cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  if (P.length < 3) return P;
  const lo = [], up = [];
  for (const p of P) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 1e-12) lo.pop(); lo.push(p); }
  for (const p of P.slice().reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 1e-12) up.pop(); up.push(p); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
/* vertices: pairwise intersection of facets, kept if feasible, hull-ordered */
function vertices(P, tol = 1e-8) {
  const pts = [];
  for (let i = 0; i < P.H.length; i++) for (let j = i + 1; j < P.H.length; j++) {
    const det = P.H[i][0] * P.H[j][1] - P.H[i][1] * P.H[j][0]; if (Math.abs(det) < 1e-10) continue;
    const x = [(P.h[i] * P.H[j][1] - P.H[i][1] * P.h[j]) / det, (P.H[i][0] * P.h[j] - P.h[i] * P.H[j][0]) / det];
    if (contains(P, x, tol)) pts.push(x);
  }
  const uniq = []; for (const p of pts) if (!uniq.some(q => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-7)) uniq.push(p);
  return uniq.length < 3 ? uniq : hull(uniq);
}
/* H-representation from the hull's edges: outward unit normals of a counter-clockwise polygon */
function fromVertices(V) {
  const Hv = hull(V), H = [], h = [];
  for (let i = 0; i < Hv.length; i++) { const p = Hv[i], q = Hv[(i + 1) % Hv.length], n = [q[1] - p[1], -(q[0] - p[0])], len = Math.hypot(...n); H.push([n[0] / len, n[1] / len]); h.push((n[0] * p[0] + n[1] * p[1]) / len); }
  return poly(H, h);
}
/* keep a row only if some vertex touches it (and drop exact duplicates) */
function removeRedundancy(P) {
  const V = vertices(P); if (V.length < 3) return P;
  const H = [], h = [];
  for (let i = 0; i < P.H.length; i++) {
    const m = Math.max(...V.map(v => dot(P.H[i], v)));
    if (m > P.h[i] - 1e-7 && !H.some((row, k) => Math.abs(row[0] - P.H[i][0]) < 1e-9 && Math.abs(row[1] - P.H[i][1]) < 1e-9 && Math.abs(h[k] - P.h[i]) < 1e-9)) { H.push(P.H[i]); h.push(P.h[i]); }
  }
  return poly(H, h);
}
const intersect = (P, Q) => removeRedundancy(poly([...P.H, ...Q.H], [...P.h, ...Q.h]));
const affineImage = (P, M, c = [0, 0]) => fromVertices(vertices(P).map(v => mv(M, v).map((y, i) => y + c[i])));
const minkowski = (P, Q) => { const VP = vertices(P), VQ = vertices(Q), pts = []; for (const p of VP) for (const q of VQ) pts.push([p[0] + q[0], p[1] + q[1]]); return fromVertices(pts); };
const post = (S, A, B, umin, umax) => fromVertices(vertices(S).flatMap(x => [umin, umax].map(u => mv(A, x).map((y, i) => y + B[i][0] * u))));
const preimage = (S, A) => removeRedundancy(poly(S.H.map(row => mv(tr(A), row)), S.h.slice()));   // {x : A x in S}
/* the controlled predecessor for a scalar input by Fourier–Motzkin elimination of u */
function predecessor(S, A, B, umin, umax) {
  const ax = S.H.map(row => mv(tr(A), row)), bu = S.H.map(row => row[0] * B[0][0] + row[1] * B[1][0]), c = S.h.slice();
  ax.push([0, 0], [0, 0]); bu.push(1, -1); c.push(umax, -umin);
  const pos = [], neg = [], zer = []; bu.forEach((b, i) => (b > 1e-12 ? pos : b < -1e-12 ? neg : zer).push(i));
  const H = [], h = [];
  for (const i of zer) { H.push(ax[i]); h.push(c[i]); }
  for (const n of neg) for (const p of pos) { H.push([ax[p][0] / bu[p] - ax[n][0] / bu[n], ax[p][1] / bu[p] - ax[n][1] / bu[n]]); h.push(c[p] / bu[p] - c[n] / bu[n]); }
  return removeRedundancy(poly(H, h));
}
const approxEqual = (P, Q, tol = 1e-6) => { const V1 = vertices(P), V2 = vertices(Q); if (!V1.length || !V2.length) return V1.length === V2.length; return V1.every(v => contains(Q, v, tol)) && V2.every(v => contains(P, v, tol)); };
function maximalInvariant(A, X, maxIter = 100) { let Om = removeRedundancy(X); const history = [Om]; for (let i = 0; i < maxIter; i++) { const nxt = intersect(X, preimage(Om, A)); history.push(nxt); if (approxEqual(nxt, Om)) return { set: nxt, history }; Om = nxt; } return { set: Om, history }; }
function maximalControlInvariant(A, B, X, umin, umax, maxIter = 100) { let C = removeRedundancy(X); const history = [C]; for (let i = 0; i < maxIter; i++) { const nxt = intersect(X, predecessor(C, A, B, umin, umax)); history.push(nxt); if (approxEqual(nxt, C)) return { set: nxt, history }; C = nxt; } return { set: C, history }; }
function controllableSets(A, B, X, target, umin, umax, N) { const sets = [intersect(target, X)]; for (let i = 0; i < N; i++) sets.push(intersect(X, predecessor(sets[sets.length - 1], A, B, umin, umax))); return sets; }
const closedLoopAdmissible = (X, K, umin, umax) => intersect(X, poly([[-K[0], -K[1]], [K[0], K[1]]], [umax, -umin]));
const area = V => V.length < 3 ? 0 : Math.abs(V.reduce((s, v, i) => { const w = V[(i + 1) % V.length]; return s + v[0] * w[1] - w[0] * v[1]; }, 0)) / 2;
const zeroTarget = () => poly([[1, 0], [0, 1], [-1, 0], [0, -1]], [0, 0, 0, 0]);
/* the interval of inputs that send x into S in one step, or null */
function witnessInterval(x, S, A, B, umin, umax) {
  let lo = umin, hi = umax; const Ax = mv(A, x);
  for (let i = 0; i < S.H.length; i++) { const a = S.H[i][0] * B[0][0] + S.H[i][1] * B[1][0], b = S.h[i] - dot(S.H[i], Ax); if (Math.abs(a) < 1e-12) { if (b < -1e-9) return null; } else if (a > 0) hi = Math.min(hi, b / a); else lo = Math.max(lo, b / a); }
  return lo > hi + 1e-9 ? null : [lo, hi];
}

/* ---------------- a primal-dual interior point for  min 1/2 z'Hz + f'z,  G z <= g,  E z = e,  lb <= z <= ub ----------------
   The same algorithm as the course package's qp.py: bounds are stacked as inequality rows,
   Newton steps on the perturbed KKT system with sigma = 0.1, the fraction-to-the-boundary rule,
   and an active-set polish at the end. Returns the multipliers lam (of the stacked rows:
   G first, then the finite upper bounds, then the finite lower bounds) and nu. */
function solveQP(H, f, G = [], g = [], E = [], e = [], lb = null, ub = null, opts = {}) {
  const n = f.length, tol = opts.tol || 1e-10, iters = opts.iters || 100, sigma = 0.1;
  const rows = G.map(r => r.slice()), rhs = g.slice();
  if (ub) for (let i = 0; i < n; i++) if (ub[i] < Infinity) { const r = new Array(n).fill(0); r[i] = 1; rows.push(r); rhs.push(ub[i]); }
  if (lb) for (let i = 0; i < n; i++) if (lb[i] > -Infinity) { const r = new Array(n).fill(0); r[i] = -1; rows.push(r); rhs.push(-lb[i]); }
  const m = rhs.length, p = e.length, Gt = m ? tr(rows) : null, Et = p ? tr(E) : null;
  let z = new Array(n).fill(0);
  if (opts.x0) z = opts.x0.slice();
  if (lb) z = z.map((v, i) => lb[i] > -Infinity ? Math.max(v, lb[i]) : v);
  if (ub) z = z.map((v, i) => ub[i] < Infinity ? Math.min(v, ub[i]) : v);
  let lam = new Array(m).fill(1), s = rhs.map((gi, i) => Math.max(gi - dot(rows[i], z), 1)), nu = new Array(p).fill(0);
  let resid = Infinity, mu = 0, nit = 0;
  for (nit = 1; nit <= iters; nit++) {
    const rd = mv(H, z).map((v, i) => v + f[i] + (m ? dot(Gt[i], lam) : 0) + (p ? dot(Et[i], nu) : 0));
    const rp = rows.map((row, i) => dot(row, z) + s[i] - rhs[i]), re = E.map((row, i) => dot(row, z) - e[i]);
    mu = m ? dot(s, lam) / m : 0;
    const rc = s.map((si, i) => sigma * mu - si * lam[i]);
    resid = Math.max(...rd.map(Math.abs), ...rp.map(Math.abs), ...re.map(Math.abs), 0);
    if (resid < tol && mu < tol) break;
    const D = lam.map((l, i) => l / s[i]), M = zeros(n + p, n + p), b = new Array(n + p).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] = H[i][j];
    for (let k = 0; k < m; k++) { const row = rows[k], Dk = D[k]; for (let i = 0; i < n; i++) { const a = row[i]; if (!a) continue; const Da = Dk * a; for (let j = 0; j < n; j++) if (row[j]) M[i][j] += Da * row[j]; } }
    for (let k = 0; k < p; k++) for (let i = 0; i < n; i++) { M[i][n + k] = E[k][i]; M[n + k][i] = E[k][i]; }
    const t = rc.map((v, i) => (v + lam[i] * rp[i]) / s[i]);
    for (let i = 0; i < n; i++) b[i] = -rd[i] - (m ? dot(Gt[i], t) : 0);
    for (let k = 0; k < p; k++) b[n + k] = -re[k];
    const sol = solve(M, b), dz = sol.slice(0, n), dnu = sol.slice(n);
    const ds = rp.map((v, i) => -v - dot(rows[i], dz)), dl = rc.map((v, i) => (v - lam[i] * ds[i]) / s[i]);
    let a = 1; for (let i = 0; i < m; i++) { if (ds[i] < 0) a = Math.min(a, -0.99 * s[i] / ds[i]); if (dl[i] < 0) a = Math.min(a, -0.99 * lam[i] / dl[i]); }
    z = z.map((v, i) => v + a * dz[i]); s = s.map((v, i) => v + a * ds[i]); lam = lam.map((v, i) => v + a * dl[i]); nu = nu.map((v, i) => v + a * dnu[i]);
    if (!z.every(Number.isFinite)) break;
  }
  /* polish on the active set */
  if (z.every(Number.isFinite) && resid < 1e-4) {
    const act = []; for (let i = 0; i < m; i++) if (lam[i] > s[i]) act.push(i);
    const k = act.length, Mp = zeros(n + k + p, n + k + p), bp = new Array(n + k + p).fill(0);
    for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) Mp[i][j] = H[i][j]; bp[i] = -f[i]; }
    act.forEach((r, q) => { for (let i = 0; i < n; i++) { Mp[i][n + q] = rows[r][i]; Mp[n + q][i] = rows[r][i]; } bp[n + q] = rhs[r]; });
    for (let q = 0; q < p; q++) { for (let i = 0; i < n; i++) { Mp[i][n + k + q] = E[q][i]; Mp[n + k + q][i] = E[q][i]; } bp[n + k + q] = e[q]; }
    const sol = solve(Mp, bp), zp = sol.slice(0, n), lp = sol.slice(n, n + k), nup = sol.slice(n + k);
    const feas = zp.every(Number.isFinite) && rows.every((row, i) => dot(row, zp) - rhs[i] <= 1e-9) && lp.every(v => v >= -1e-9) && E.every((row, i) => Math.abs(dot(row, zp) - e[i]) <= 1e-9);
    if (feas) { z = zp; nu = nup; lam = new Array(m).fill(0); act.forEach((r, q) => lam[r] = Math.max(lp[q], 0)); s = rhs.map((gi, i) => gi - dot(rows[i], z)); const rd = mv(H, z).map((v, i) => v + f[i] + (m ? dot(Gt[i], lam) : 0) + (p ? dot(Et[i], nu) : 0)); resid = Math.max(...rd.map(Math.abs), ...E.map((row, i) => Math.abs(dot(row, z) - e[i])), 0); mu = 0; }
  }
  const maxViolation = Math.max(...rows.map((row, i) => dot(row, z) - rhs[i]), ...E.map((row, i) => Math.abs(dot(row, z) - e[i])), 0);
  const success = z.every(Number.isFinite) && resid < 10 * tol && mu < 10 * tol && maxViolation <= 1e-7;
  return { z, lam: lam.slice(0, G.length), lamBounds: lam.slice(G.length), nu, obj: 0.5 * quad(H, z) + dot(f, z), success, maxViolation, nit, resid };
}

/* ---------------- condensed linear MPC on a 2-D system with a scalar input ---------------- */
function predictionMatrices(A, B, N) {
  const n = A.length, Sx = zeros(N * n, n), Su = zeros(N * n, N); let Ap = eye(n); const powers = [Ap];
  for (let i = 1; i <= N; i++) { Ap = mm(A, Ap); powers.push(Ap); }
  for (let i = 0; i < N; i++) { for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Sx[i * n + r][c] = powers[i + 1][r][c]; for (let j = 0; j <= i; j++) { const blk = mm(powers[i - j], B); for (let r = 0; r < n; r++) Su[i * n + r][j] = blk[r][0]; } }
  return { Sx, Su };
}
/* terminal = {kind: "zero"} | {kind: "set", H, h} | {kind: "none"}. R is a number (scalar input). */
function makeMPC(A, B, Q, R, N, P, xlb, xub, umin, umax, terminal = { kind: "none" }) {
  const n = A.length, { Sx, Su } = predictionMatrices(A, B, N), Qbar = zeros(N * n, N * n);
  for (let i = 0; i < N; i++) { const W = i === N - 1 ? P : Q; for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Qbar[i * n + r][i * n + c] = W[r][c]; }
  const SuTQ = mm(tr(Su), Qbar), H = mm(SuTQ, Su).map((r, i) => r.map((v, j) => 2 * (v + (i === j ? R : 0)))), F = mm(SuTQ, Sx).map(r => r.map(v => 2 * v)), SxTQSx = mm(tr(Sx), mm(Qbar, Sx));
  return { A, B, N, Q, R, P, Sx, Su, H, terminal, xlb, xub, umin, umax,
    build(x0) {
      const f = mv(F, x0), Sxx = mv(Sx, x0), G = [], g = [], E = [], e = [];
      for (let r = 0; r < N * n; r++) { const k = r % n; G.push(Su[r]); g.push(xub[k] - Sxx[r]); G.push(Su[r].map(v => -v)); g.push(-xlb[k] + Sxx[r]); }
      const SuN = Su.slice(-n), SxNx = Sxx.slice(-n);
      if (terminal.kind === "zero") { for (let r = 0; r < n; r++) { E.push(SuN[r]); e.push(-SxNx[r]); } }
      else if (terminal.kind === "set") terminal.H.forEach((row, i) => { G.push(SuN[0].map((_, j) => row.reduce((s, hr, r) => s + hr * SuN[r][j], 0))); g.push(terminal.h[i] - dot(row, SxNx)); });
      return { H, f, G, g, E, e, lb: new Array(N).fill(umin), ub: new Array(N).fill(umax), Sxx, constant: quad(Q, x0) + quad(SxTQSx, x0) };
    },
    solve(x0, opts) {
      const qp = this.build(x0), r = solveQP(qp.H, qp.f, qp.G, qp.g, qp.E, qp.e, qp.lb, qp.ub, opts);
      const Xs = qp.Sxx.map((v, i) => v + dot(Su[i], r.z)), X = []; for (let i = 0; i < N; i++) X.push(Xs.slice(i * n, i * n + n));
      return { U: r.z, X, J: r.obj + qp.constant, feasible: r.success, qp: r };   // J: the full value x0'Qx0 + sum x'Qx + sum R u^2 + xN'PxN
    } };
}
/* the receding-horizon loop with the full value J_N^* and the Lyapunov residual at each step */
function closedLoop(mpc, x0, steps = 40, tol = 1e-3) {
  let x = x0.slice(); const xs = [x.slice()], us = [], vals = [], stages = []; let infeasibleAt = -1;
  for (let t = 0; t < steps; t++) {
    const sol = mpc.solve(x); if (!sol.feasible) { infeasibleAt = t; break; }
    const u = sol.U[0]; vals.push(sol.J); stages.push(quad(mpc.Q, x) + mpc.R * u * u); us.push(u);
    x = mv(mpc.A, x).map((v, i) => v + mpc.B[i][0] * u); xs.push(x.slice());
    if (norm(x) < tol) break;
  }
  const resid = vals.slice(1).map((v, t) => v - vals[t] + stages[t]);
  return { xs, us, vals, stages, resid, infeasibleAt };
}
const rollout = (A, B, x0, U) => { const xs = [x0.slice()]; let x = x0; for (const u of U) { x = mv(A, x).map((v, i) => v + B[i][0] * u); xs.push(x); } return xs; };

/* ---------------- the course's canonical system ---------------- */
const SYS = {
  A: [[1, 1], [0, 1]], B: [[0.5], [1]], Q: [[1, 0], [0, 0.2]], R: 0.2,
  xlb: [-4, -3], xub: [4, 3], umin: -1, umax: 1,
  X: box([-4, -3], [4, 3]),
};
SYS.lqr = dlqr(SYS.A, SYS.B, SYS.Q, [[SYS.R]]);

/* ---------------- drawing helpers shared by the chapters ---------------- */
function legend(ctx, lines, x, y, narrow) { const css = global.Kit.css; ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; const out = narrow ? lines.flatMap(l => l.split(" · ")) : lines; out.forEach((l, i) => ctx.fillText(l, x, y + 13 * i)); }
function drawPoly(fr, V, fill, stroke, width = 1.5, dash = null) { if (V.length < 3) return; const { ctx, X, Y } = fr; ctx.beginPath(); V.forEach((v, i) => i ? ctx.lineTo(X(v[0]), Y(v[1])) : ctx.moveTo(X(v[0]), Y(v[1]))); ctx.closePath(); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]); } }
function clickPoint(ev) { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), fr = cv._fr; if (!fr) return null; return [fr.x0 + (ev.clientX - r.left - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), fr.y1 - (ev.clientY - r.top - fr.pad.t) / fr.ih * (fr.y1 - fr.y0)]; }
function drawPath(fr, pts, color, width = 1.6, dots = 2) { const { ctx, X, Y } = fr; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); pts.forEach((x, i) => i ? ctx.lineTo(X(x[0]), Y(x[1])) : ctx.moveTo(X(x[0]), Y(x[1]))); ctx.stroke(); if (dots) { ctx.fillStyle = color; pts.forEach(x => { ctx.beginPath(); ctx.arc(X(x[0]), Y(x[1]), dots, 0, 2 * Math.PI); ctx.fill(); }); } }
const fmtM = (M, d = 3) => "\\begin{pmatrix} " + M.map(r => r.map(v => v.toFixed(d)).join(" & ")).join(" \\\\ ") + " \\end{pmatrix}";
const fmtV = (v, d = 3) => "(" + v.map(x => x.toFixed(d)).join(", ") + ")";
const sci = (v, d = 1) => (v === 0 ? "0" : v.toExponential(d).replace(/e([+-])(\d+)/, (_, s, n) => " \\times 10^{" + (s === "-" ? "-" : "") + parseInt(n, 10) + "}"));

global.MPC = { zeros, eye, mm, mv, tr, add, sub, scale, dot, quad, norm, matpow, solve, inv, eig2, eigsym2,
  dlqr, riccatiRecursion,
  poly, box, contains, hull, vertices, fromVertices, removeRedundancy, intersect, affineImage, minkowski, post, preimage, predecessor,
  approxEqual, maximalInvariant, maximalControlInvariant, controllableSets, closedLoopAdmissible, area, zeroTarget, witnessInterval,
  solveQP, predictionMatrices, makeMPC, closedLoop, rollout, SYS,
  legend, drawPoly, clickPoint, drawPath, fmtM, fmtV, sci };
})(window);
