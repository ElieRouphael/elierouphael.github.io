/* Advanced MPC and Learning MPC, chapter 03, MPC stability and terminal sets: the live demos.
   Everything is computed here: the scalar horizon-one pathology and its Riccati repair; the exact
   2-D polytope toolkit of chapter 02 (controllable sets to a zero target or to an invariant terminal
   set, the maximal control-invariant set, the LQR terminal set); a condensed MPC with either a zero
   terminal equality or a terminal set and terminal cost, solved by an interior-point method; and the
   closed loop with the full finite-horizon value J_N and the Lyapunov residual ΔJ + q at every step.
   The defaults reproduce the notebook's numbers. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp, fit } = window.Kit;

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
  return { z, obj: 0.5 * quad(H, z) + dot(f, z), maxViolation: Math.max(...g.map((gi, i) => dot(G[i], z) - gi), ...e.map((ei, i) => Math.abs(dot(E[i], z) - ei)), 0) };
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
/* NET-END */

const A = [[1, 1], [0, 1]], B = [[0], [1]], Xbox = box([-5, -5], [5, 5]), UMIN = -0.5, UMAX = 0.5, Q = [[1, 0], [0, 1]];
const RVALS = [1, 2, 5, 10, 20, 50, 100];
function legend(ctx, lines, x, y, narrow) { ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; const out = narrow ? lines.flatMap(l => l.split(" · ")) : lines; out.forEach((l, i) => ctx.fillText(l, x, y + 13 * i)); }
function drawPoly(fr, V, fill, stroke, width = 1.5, dash = null) { if (V.length < 3) return; const { ctx, X, Y } = fr; ctx.beginPath(); V.forEach((v, i) => i ? ctx.lineTo(X(v[0]), Y(v[1])) : ctx.moveTo(X(v[0]), Y(v[1]))); ctx.closePath(); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]); } }
function clickPoint(ev) { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), fr = cv._fr; if (!fr) return null; return [fr.x0 + (ev.clientX - r.left - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), fr.y1 - (ev.clientY - r.top - fr.pad.t) / fr.ih * (fr.y1 - fr.y0)]; }
const CINF = maximalControlInvariant(A, B, Xbox, UMIN, UMAX), CINFV = vertices(CINF.set);

/* =====================================================================
   Hero: terminal ingredients, feasible regions and the closed loop
   ===================================================================== */
const hs = { design: "set", probe: [4.5, 0.5], cache: {} };
function heroSets() {
  const N = +$("#hN").value, R = RVALS[+$("#hR").value], key = `${N}|${R}`; if (hs.cache[key]) return hs.cache[key];
  const L = dlqr(A, B, Q, [[R]]), XK = closedLoopAdmissible(Xbox, L.K[0], UMIN, UMAX), Xf = maximalInvariant(L.Acl, XK);
  const zeroN = controllableSets(A, B, Xbox, zeroTarget(), UMIN, UMAX, N), termN = controllableSets(A, B, Xbox, Xf.set, UMIN, UMAX, N);
  return hs.cache[key] = { N, R, L, XK, Xf: Xf.set, XfIters: Xf.history.length - 1, XfV: vertices(Xf.set), zeroV: vertices(zeroN[N]), termV: vertices(termN[N]), zeroN: zeroN[N], termN: termN[N] };
}
function heroRun() {
  const r = heroSets(), key = `${r.N}|${r.R}|${hs.design}|${hs.probe.map(v => v.toFixed(4)).join(",")}`; if (hs.cache[key]) return hs.cache[key];
  const terminal = hs.design === "zero" ? { kind: "zero" } : hs.design === "set" ? { kind: "set", H: r.Xf.H, h: r.Xf.h } : { kind: "none" };
  const mpc = makeMPC(A, B, Q, r.R, r.N, hs.design === "zero" ? Q : r.L.P, [-5, -5], [5, 5], UMIN, UMAX, terminal);
  return hs.cache[key] = closedLoop(mpc, hs.probe, 40, 1e-3);
}
function drawHero() {
  const r = heroSets(), run = heroRun(), cv = $("#planeC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.1 : 1.35, { x: [-5.6, 5.6], y: [-5.6, 5.6], xlabel: "x₁", ylabel: "x₂", xticks: [-5, -2.5, 0, 2.5, 5], yticks: [-5, -2.5, 0, 2.5, 5] }), { ctx, X, Y } = fr; cv._fr = fr;
  drawPoly(fr, vertices(Xbox), alpha("--ink", 0.04), alpha("--ink", 0.35), 1, [3, 3]);
  drawPoly(fr, CINFV, null, css("--warm"), 1.5, [5, 4]);
  drawPoly(fr, r.termV, alpha("--post", 0.12), css("--post"), 2);
  drawPoly(fr, r.zeroV, alpha("--c5", 0.18), css("--c5"), 1.8);
  drawPoly(fr, r.XfV, alpha("--post", 0.3), css("--post"), 1.2, [2, 3]);
  /* the closed loop */
  const p = hs.probe; ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.6; ctx.beginPath(); run.xs.forEach((x, i) => i ? ctx.lineTo(X(x[0]), Y(x[1])) : ctx.moveTo(X(x[0]), Y(x[1]))); ctx.stroke();
  ctx.fillStyle = css("--ink"); run.xs.forEach(x => { ctx.beginPath(); ctx.arc(X(x[0]), Y(x[1]), 2, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = run.infeasibleAt === 0 ? css("--warm") : css("--ink"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--bg"); ctx.lineWidth = 1.5; ctx.stroke();
  legend(ctx, ["purple: x_N = 0 region · blue: x_N ∈ X_f region", "dotted: X_f · dashed orange: C_∞"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* value and residual against time */
  const cv2 = $("#valueC"), T = Math.max(10, run.vals.length), fr2 = frame(cv2, narrow ? 1.5 : 2.3, { x: [0, T], y: [-6, 3], xlabel: "closed-loop step t", yticks: [-6, -4, -2, 0, 2], yfmt: v => "10" + String(v).replace("-", "⁻").replace(/\d/g, ch => "⁰¹²³⁴⁵⁶⁷⁸⁹"[ch]) }), ctx2 = fr2.ctx;
  if (run.vals.length) { plotLine(fr2, run.vals.map((_, t) => t), run.vals.map(v => Math.max(-6, Math.log10(Math.max(v, 1e-300)))), { color: css("--post"), width: 2 }); ctx2.fillStyle = css("--post"); run.vals.forEach((v, t) => { ctx2.beginPath(); ctx2.arc(fr2.X(t), fr2.Y(Math.max(-6, Math.log10(Math.max(v, 1e-300)))), 2.2, 0, 2 * Math.PI); ctx2.fill(); }); }
  legend(ctx2, ["J*_N(x(t)), the full finite-horizon value, log scale"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  $("#hNV").textContent = r.N; $("#hRV").textContent = r.R; $$("#demo [data-design]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.design === hs.design)));
  $("#rZero").textContent = fmt(area(r.zeroV), 2); $("#rTerm").textContent = fmt(area(r.termV), 2); $("#rXf").textContent = `${r.XfV.length} vertices, area ${fmt(area(r.XfV), 2)}`;
  const nb = r.N === 5 && r.R === 10 && Math.abs(p[0] - 4.5) < 1e-9 && Math.abs(p[1] - 0.5) < 1e-9, bad = run.resid.filter(v => v > 1e-6).length, maxRes = run.resid.length ? Math.max(...run.resid) : NaN, last = run.xs[run.xs.length - 1];
  const inZero = contains(r.zeroN, p, 1e-7), inTerm = contains(r.termN, p, 1e-7);
  $("#rLoop").textContent = run.infeasibleAt === 0 ? "infeasible at the start" : run.infeasibleAt > 0 ? `infeasible at step ${run.infeasibleAt}` : `${run.us.length} steps, ${bad} of ${run.resid.length} residuals > 0`;
  let text = `[[N = ${r.N}]], [[R = ${r.R}]], [[K = (${fmt(r.L.K[0][0], 4)}, ${fmt(r.L.K[0][1], 4)})]], [[\\rho(A - BK) = ${fmt(Math.hypot(...r.L.eig[0]), 4)}]]${nb && hs.design === "set" ? ", the notebook's run" : ""}. The terminal set [[\\mathcal X_f]] (${r.XfV.length} vertices, ${r.XfIters} iterations) is the largest region where the LQR law stays within [[|u| \\le 0.5]] and inside the box forever. From it, the [[N]]-step region [[x_N \\in \\mathcal X_f]] (blue) has area ${fmt(area(r.termV), 2)} against ${fmt(area(r.zeroV), 2)} for [[x_N = 0]] (purple) and ${fmt(area(CINFV), 2)} for the limit [[\\mathcal C_\\infty]]. `;
  text += `The start [[x_0 = (${fmt(p[0], 2)}, ${fmt(p[1], 2)})]] is ${inZero ? "inside" : "outside"} the zero-terminal region and ${inTerm ? "inside" : "outside"} the terminal-set region. `;
  const dname = hs.design === "zero" ? "design A, [[x_N = 0]]" : hs.design === "set" ? "design B, [[x_N \\in \\mathcal X_f]] with the Riccati cost [[P]]" : "no terminal constraint, Riccati cost [[P]] only";
  if (run.infeasibleAt === 0) text += `Under ${dname} the first problem is infeasible: no admissible ${r.N}-step plan meets the terminal requirement from here. The theorem says nothing from outside its feasible set.`;
  else if (run.infeasibleAt > 0) text += `Under ${dname} the loop ran ${run.infeasibleAt} steps and then found no feasible plan: feasible now did not mean feasible later. This cannot happen under designs A or B, whose shifted sequence is always a feasible plan.`;
  else text += `Under ${dname} the loop ${Math.hypot(...last) < 1e-3 ? `reaches [[\\lVert x \\rVert < 10^{-3}]] in ${run.us.length} steps` : `runs the ${run.us.length}-step cap and ends at [[(${fmt(last[0], 4)}, ${fmt(last[1], 4)})]]`}, with [[|u|]] saturated on ${run.us.filter(u => Math.abs(Math.abs(u) - 0.5) < 1e-6).length} of them. The Lyapunov residual [[J_N^\\star(x_{t+1}) - J_N^\\star(x_t) + q(x_t, u_t)]] is at most ${maxRes.toExponential(1)}: ${bad === 0 ? (hs.design === "none" ? "never positive beyond solver tolerance here, although no theorem guarantees it for this design" : "never positive beyond solver tolerance, as the theorem demands") : `positive on ${bad} of ${run.resid.length} steps, which the theorem forbids under designs A and B but does not rule out here`}. ${hs.design === "set" && bad === 0 ? "It is in fact nearly an equality: once the input leaves saturation the MPC coincides with the LQR, whose value drops by exactly [[q]] each step." : ""}`;
  caption($("#heroStatus"), text + " Tap or click the plane to move the start.");
}
["#hN", "#hR"].forEach(id => $(id).addEventListener("input", drawHero));
$$("#demo [data-design]").forEach(bt => bt.addEventListener("click", () => { hs.design = bt.dataset.design; drawHero(); }));
$("#planeC").addEventListener("click", ev => { const p = clickPoint(ev); if (p) { hs.probe = [clamp(p[0], -5, 5), clamp(p[1], -5, 5)]; drawHero(); } });
$("#hReset").addEventListener("click", () => { $("#hN").value = 5; $("#hR").value = 3; hs.design = "set"; hs.probe = [4.5, 0.5]; drawHero(); });

/* =====================================================================
   Part 3: the scalar horizon-one pathology
   ===================================================================== */
function drawScalar() {
  const a = +$("#sA").value, r = +$("#sR").value, L = dlqr([[a]], [[1]], [[1]], [[r]]), P = L.P[0][0], K = L.K[0][0], lam = a - K;
  const steps = 15; let x1 = 1, x2 = 1; const xs1 = [x1], xs2 = [x2]; for (let t = 0; t < steps; t++) { x1 = a * x1; xs1.push(x1); x2 = a * x2 - K * x2; xs2.push(x2); }
  const cv = $("#scalarC"), narrow = narrowOf(cv), ymax = Math.min(20, Math.max(2, Math.max(...xs1))), fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, steps], y: [-0.5, ymax], xlabel: "closed-loop step", xticks: [0, 5, 10, 15] }), { ctx } = fr;
  plotLine(fr, xs1.map((_, t) => t), xs1, { color: css("--warm"), width: 2 }); plotLine(fr, xs2.map((_, t) => t), xs2, { color: css("--post"), width: 2 });
  ctx.strokeStyle = alpha("--ink", 0.3); seg(ctx, fr.pad.l, fr.Y(0), fr.w - fr.pad.r, fr.Y(0));
  legend(ctx, ["orange: no terminal cost, u = 0", "blue: with the Riccati terminal cost"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#sAV").textContent = fmt(a, 2); $("#sRV").textContent = fmt(r, 1);
  const nb = a === 1.2 && r === 4;
  caption($("#scalarCap"), `[[x^+ = ${fmt(a, 2)}x + u]], [[q = x^2 + ${fmt(r, 1)}u^2]]${nb ? ", the notebook's numbers" : ""}. With [[N = 1]] and no terminal cost the objective [[x_0^2 + ru_0^2]] does not contain [[x_1]], so [[u_0^\\star = 0]] and the loop is [[x^+ = ${fmt(a, 2)}x]]: ${a > 1 ? `unstable, ${fmt(xs1[steps], 1)} after ${steps} steps` : a === 1 ? "marginally stable, the state never moves" : "stable only because the plant already was"}. Adding the Riccati terminal cost [[Px_1^2]] with [[P = ${fmt(P, 4)}]] gives [[u^\\star = -\\frac{Pa}{r + P}x = -${fmt(K, 4)}x]], the scalar LQR law, and the closed-loop eigenvalue [[a - K = ${fmt(lam, 4)}]]${Math.abs(lam) < 1 ? ": stable" : ""}. One number, [[P]], carries the whole infinite future into a one-step problem. ` + (r > 15 ? "A heavy input penalty makes the repair timid: the eigenvalue creeps towards 1 but stays below it, since the DARE always yields a stabilising gain." : a >= 1.5 ? "A more unstable plant needs a larger P and a larger gain; the Riccati solution adapts automatically." : "Exercise 1 asks for the same with a = 1.1 and r = 5: the eigenvalue without terminal cost is 1.1, and with it 0.648."));
}
["#sA", "#sR"].forEach(id => $(id).addEventListener("input", drawScalar));

/* =====================================================================
   Part 5: horizon length and the zero-terminal feasible region
   ===================================================================== */
const ha = { cache: {} };
function drawAreas() {
  const umax = +$("#aU").value, key = String(umax);
  if (!ha.cache[key]) { const sets = controllableSets(A, B, Xbox, zeroTarget(), -umax, umax, 20), C = maximalControlInvariant(A, B, Xbox, -umax, umax); ha.cache[key] = { areas: sets.map(S => area(vertices(S))), cArea: area(vertices(C.set)), sets }; }
  const d = ha.cache[key], cv = $("#areasC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, 21], y: [0, Math.max(d.cArea, ...d.areas) * 1.15], xlabel: "horizon N", xticks: [1, 5, 10, 15, 20] }), { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--warm"); ctx.setLineDash([5, 4]); seg(ctx, fr.pad.l, Y(d.cArea), fr.w - fr.pad.r, Y(d.cArea)); ctx.setLineDash([]);
  d.areas.forEach((a, N) => { if (N === 0) return; const w = (X(1) - X(0)) * 0.7; ctx.fillStyle = [5, 10, 20].includes(N) ? css("--post") : alpha("--post", 0.45); ctx.fillRect(X(N) - w / 2, Y(a), w, Y(0) - Y(a)); });
  legend(ctx, ["bars: area of the x_N = 0 region", "dashed: area of C_∞"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#aUV").textContent = fmt(umax, 1);
  const first = d.areas.findIndex(a => a > d.cArea - 1e-6);
  caption($("#areasCap"), `[[|u| \\le ${fmt(umax, 1)}]]${umax === 0.5 ? ", the notebook's bound" : ""}: the region from which the origin can be reached exactly in [[N]] steps grows with [[N]]: a line segment of zero area at [[N = 1]], then ${fmt(d.areas[2], 1)} at [[N = 2]] and ${fmt(d.areas[3], 1)} at [[N = 3]], to ${fmt(d.areas[5], 2)} at [[N = 5]] (${fmt(100 * d.areas[5] / d.cArea, 1)}% of [[\\mathcal C_\\infty]]), ${fmt(d.areas[10], 2)} at [[N = 10]] (${fmt(100 * d.areas[10] / d.cArea, 1)}%) and ${fmt(d.areas[20], 2)} at [[N = 20]], ${first > 0 ? `reaching the whole of [[\\mathcal C_\\infty]] (area ${fmt(d.cArea, 2)}) at [[N = ${first}]]` : "still short of " + fmt(d.cArea, 2)}: from every state that can be kept safe forever, the origin can be reached in ${first > 0 ? first : "more than 20"} steps with this input authority. The horizon is a lever on the region of attraction; its price is paid online, in decision variables and constraints. ${umax < 0.5 ? "With less input authority the regions grow more slowly in N, and the limit C_∞ itself is smaller." : umax > 0.5 ? "With more authority the sets fill C_∞ in fewer steps." : "Exercise 7 asks for N = 3, 5, 8, 12: areas 4.0, 16.1, 29.2 and 37.3."}`);
}
$("#aU").addEventListener("input", drawAreas);

/* =====================================================================
   Part 7: the LQR terminal region and its checks
   ===================================================================== */
function drawTerminal() {
  const R = RVALS[+$("#tR").value], L = dlqr(A, B, Q, [[R]]), K = L.K[0], XK = closedLoopAdmissible(Xbox, K, UMIN, UMAX), Xf = maximalInvariant(L.Acl, XK), V = vertices(Xf.set), XKV = vertices(XK);
  const cv = $("#termC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.15 : 1.5, { x: [-5.6, 5.6], y: [-5.6, 5.6], xlabel: "x₁", ylabel: "x₂", xticks: [-5, -2.5, 0, 2.5, 5], yticks: [-5, -2.5, 0, 2.5, 5] }), { ctx, X, Y } = fr;
  drawPoly(fr, vertices(Xbox), alpha("--ink", 0.04), alpha("--ink", 0.35), 1, [3, 3]); drawPoly(fr, XKV, alpha("--c5", 0.12), css("--c5"), 1.5, [5, 4]); drawPoly(fr, V, alpha("--post", 0.3), css("--post"), 2);
  /* one LQR trajectory from each vertex, two steps */
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; V.forEach(v => { let x = v; ctx.beginPath(); ctx.moveTo(X(x[0]), Y(x[1])); for (let k = 0; k < 3; k++) { x = mv(L.Acl, x); ctx.lineTo(X(x[0]), Y(x[1])); } ctx.stroke(); });
  legend(ctx, ["dashed purple: X_K · blue: X_f", "black: LQR loop from each vertex"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const rows = V.map(v => { const u = -dot(K, v), vn = mv(L.Acl, v); return { v, u, inside: contains(Xf.set, vn, 1e-7), res: quad(L.P, vn) - quad(L.P, v) + quad(Q, v) + R * u * u }; });
  $("#termT").innerHTML = "<thead><tr><th>vertex</th><th>u = −Kv</th><th>next in X<sub>f</sub></th><th>p(x⁺) − p(x) + q</th></tr></thead><tbody>" + rows.map(r => `<tr><td>(${fmt(r.v[0], 3)}, ${fmt(r.v[1], 3)})</td><td>${fmt(r.u, 4)}</td><td class="words">${r.inside ? "yes" : "no"}</td><td>${r.res.toExponential(1)}</td></tr>`).join("") + "</tbody>";
  $("#tRV").textContent = R;
  const rho = Math.hypot(...L.eig[0]), nb = R === 10, resid = Math.max(...sub(add(mm(tr(L.Acl), mm(L.P, L.Acl)), add(Q, mm(tr(L.K), mm([[R]], L.K)))), L.P).flat().map(Math.abs));
  caption($("#termCap"), `[[R = ${R}]]${nb ? ", the notebook's value" : ""}: [[K = (${fmt(K[0], 4)}, ${fmt(K[1], 4)})]], [[P = \\begin{pmatrix} ${fmt(L.P[0][0], 3)} & ${fmt(L.P[0][1], 3)} \\\\ ${fmt(L.P[1][0], 3)} & ${fmt(L.P[1][1], 3)} \\end{pmatrix}]], [[\\rho(A_{\\mathrm{cl}}) = ${fmt(rho, 4)}]], DARE identity residual ${resid.toExponential(1)}. The LQR input is admissible on [[\\mathcal X_K]] (${XKV.length} vertices); the fixed point [[\\Omega_{j+1} = \\mathcal X_K \\cap A_{\\mathrm{cl}}^{-1}(\\Omega_j)]] stops after ${Xf.history.length - 1} iterations with [[\\mathcal X_f]] of ${V.length} vertices and area ${fmt(area(V), 2)}. The table checks the three assumptions at every vertex: the state is in the box, the input is within [[\\pm 0.5]] (the largest is ${fmt(Math.max(...rows.map(r => Math.abs(r.u))), 4)}), the next state stays in [[\\mathcal X_f]], and the terminal decrease [[p(A_{\\mathrm{cl}} x) - p(x) + q(x, -Kx)]] is zero to ${Math.max(...rows.map(r => Math.abs(r.res))).toExponential(0)}: equality, as the DARE identity promises. ` + (R <= 2 ? "A cheap input gives an aggressive gain, which saturates sooner: the admissible strip is narrow and the terminal set small." : R >= 50 ? "An expensive input gives a gentle gain that rarely saturates: the terminal set is large, but the loop inside it converges slowly (exercise 8)." : "Exercise 8 trades these off: smaller R, smaller terminal set; larger R, larger set but a slower local loop."));
}
$("#tR").addEventListener("input", drawTerminal);

function drawAll() { drawHero(); drawScalar(); drawAreas(); drawTerminal(); }
drawAll();
onRedraw(drawAll);
})();
