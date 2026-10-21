/* Advanced MPC and Learning MPC, chapter 00, Roadmap and mathematical primer: the live demos.
   Everything is computed here: the double integrator, the stacked prediction matrices, the condensed
   QP (solved by a primal-dual interior-point method to 1e-9), the discrete Riccati equation (by
   iteration), the maximal positively invariant set (the fixed-point algorithm on H-polytopes, with
   vertex enumeration in 2-D) and the receding-horizon loop with the notebook's stop rule. The default
   settings reproduce the notebook's numbers.  Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp, fit } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
/* ---- small dense linear algebra ---- */
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
const maxAbs = A => Math.max(...A.flat().map(Math.abs));
function solve(A0, b0) {   // Gaussian elimination with partial pivoting
  const n = A0.length, A = A0.map((r, i) => [...r, b0[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < n; r++) { if (r === c) continue; const m = A[r][c] / A[c][c]; if (!m) continue; for (let k = c; k <= n; k++) A[r][k] -= m * A[c][k]; }
  }
  return A.map((r, i) => r[n] / r[i]);
}
const inv = A => tr(A[0].map((_, j) => solve(A, A.map((r, i) => i === j ? 1 : 0))));
/* eigenvalues of a 2x2 matrix, as [re, im] pairs */
function eig2(M) { const t = M[0][0] + M[1][1], d = M[0][0] * M[1][1] - M[0][1] * M[1][0], disc = t * t / 4 - d; return disc >= 0 ? [[t / 2 + Math.sqrt(disc), 0], [t / 2 - Math.sqrt(disc), 0]] : [[t / 2, Math.sqrt(-disc)], [t / 2, -Math.sqrt(-disc)]]; }
const specRadius = M => Math.max(...eig2(M).map(([re, im]) => Math.hypot(re, im)));

/* ---- Part II: stacked prediction matrices, [x1;...;xN] = Sx x0 + Su [u0;...;uN-1] ---- */
function predictionMatrices(A, B, N) {
  const n = A.length, m = B[0].length, Sx = zeros(N * n, n), Su = zeros(N * n, N * m);
  let Ap = eye(n); const powers = [Ap];
  for (let i = 1; i <= N; i++) { Ap = mm(A, Ap); powers.push(Ap); }
  for (let i = 0; i < N; i++) {
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Sx[i * n + r][c] = powers[i + 1][r][c];
    for (let j = 0; j <= i; j++) { const blk = mm(powers[i - j], B); for (let r = 0; r < n; r++) for (let c = 0; c < m; c++) Su[i * n + r][j * m + c] = blk[r][c]; }
  }
  return { Sx, Su };
}
const rollout = (A, B, x0, U) => { const xs = [x0.slice()]; let x = x0; for (const u of U) { x = mv(A, x).map((v, i) => v + mv(B, u)[i]); xs.push(x); } return xs; };

/* ---- Part V: the discrete algebraic Riccati equation by iteration, u = -K x ---- */
function dlqr(A, B, Q, R, iters = 5000) {
  let P = Q;
  for (let it = 0; it < iters; it++) {
    const BtP = mm(tr(B), P), S = add(R, mm(BtP, B)), K = mm(inv(S), mm(BtP, A));
    const Pn = add(Q, sub(mm(tr(A), mm(P, A)), mm(tr(mm(BtP, A)), K)));
    const diff = maxAbs(sub(Pn, P)); P = Pn; if (diff < 1e-13) break;
  }
  const BtP = mm(tr(B), P), K = mm(inv(add(R, mm(BtP, B))), mm(BtP, A)), Acl = sub(A, mm(B, K));
  return { K, P, Acl, eig: eig2(Acl) };
}

/* ---- a primal-dual interior-point method for  min 1/2 z'Hz + f'z  s.t.  G z <= g ---- */
function solveQP(H, f, G, g, iters = 60) {
  const n = f.length, m = g.length; let z = new Array(n).fill(0), lam = new Array(m).fill(1), s = g.map((gi, i) => Math.max(gi - dot(G[i], z), 1));
  const Gt = tr(G);
  for (let it = 0; it < iters; it++) {
    const rd = mv(H, z).map((v, i) => v + f[i] + dot(Gt[i], lam)), rp = G.map((row, i) => dot(row, z) + s[i] - g[i]);
    const mu = dot(s, lam) / m, sigma = 0.1, rc = s.map((si, i) => sigma * mu - si * lam[i]);
    if (Math.max(...rd.map(Math.abs)) < 1e-10 && Math.max(...rp.map(Math.abs)) < 1e-10 && mu < 1e-11) break;
    const D = lam.map((l, i) => l / s[i]), M = H.map((row, i) => row.map((v, j) => v + G.reduce((acc, gr, k) => acc + gr[i] * D[k] * gr[j], 0)));
    const t = rc.map((v, i) => (v + lam[i] * rp[i]) / s[i]), rhs = rd.map((v, i) => -v - dot(Gt[i], t));
    const dz = solve(M, rhs), ds = rp.map((v, i) => -v - dot(G[i], dz)), dl = rc.map((v, i) => (v - lam[i] * ds[i]) / s[i]);   // from s dl + lam ds = rc
    let a = 1; for (let i = 0; i < m; i++) { if (ds[i] < 0) a = Math.min(a, -0.99 * s[i] / ds[i]); if (dl[i] < 0) a = Math.min(a, -0.99 * lam[i] / dl[i]); }
    z = z.map((v, i) => v + a * dz[i]); s = s.map((v, i) => v + a * ds[i]); lam = lam.map((v, i) => v + a * dl[i]);
  }
  return { z, obj: 0.5 * quad(H, z) + dot(f, z), maxViolation: Math.max(...g.map((gi, i) => dot(G[i], z) - gi)) };
}

/* ---- the condensed linear MPC of the notebook (LinearMPC) ---- */
function makeMPC(A, B, Q, R, N, P, xlb, xub, ulb, uub, terminal) {
  const n = A.length, { Sx, Su } = predictionMatrices(A, B, N);
  const Qbar = zeros(N * n, N * n); for (let i = 0; i < N; i++) { const W = i === N - 1 ? P : Q; for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Qbar[i * n + r][i * n + c] = W[r][c]; }
  const Rbar = zeros(N, N); for (let i = 0; i < N; i++) Rbar[i][i] = R[0][0];
  const SuTQ = mm(tr(Su), Qbar), H = scale(add(mm(SuTQ, Su), Rbar), 2), F = scale(mm(SuTQ, Sx), 2);   // f(x0) = F x0
  const SxTQSx = mm(tr(Sx), mm(Qbar, Sx));   // the constant c(x0) = x0' SxTQSx x0; V_N(x0) also counts x0'Qx0, which the condensed objective drops
  return { N, n, Sx, Su, H, F, SxTQSx, Q, R, P, xlb, xub, ulb, uub, terminal, A, B,
    solve(x0) {
      const f = mv(F, x0), G = [], g = [], Sxx = mv(Sx, x0);
      for (let r = 0; r < N * n; r++) { const k = r % n; if (xub) { G.push(Su[r]); g.push(xub[k] - Sxx[r]); } if (xlb) { G.push(Su[r].map(v => -v)); g.push(-xlb[k] + Sxx[r]); } }
      for (let j = 0; j < N; j++) { const e = new Array(N).fill(0); e[j] = 1; G.push(e); g.push(uub[0]); G.push(e.map(v => -v)); g.push(-ulb[0]); }
      if (terminal) { const SuN = Su.slice(-n), SxNx = Sxx.slice(-n); terminal.H.forEach((row, i) => { G.push(SuN[0].map((_, j) => row.reduce((s, v, k) => s + v * SuN[k][j], 0))); g.push(terminal.h[i] - dot(row, SxNx)); }); }
      const res = solveQP(H, f, G, g), X = []; const Xs = mv(Sx, x0).map((v, r) => v + dot(Su[r], res.z)); for (let i = 0; i < N; i++) X.push(Xs.slice(i * n, (i + 1) * n));
      return { U: res.z, X, objective: res.obj, value: res.obj + quad(SxTQSx, x0) + quad(Q, x0), feasible: res.maxViolation < 1e-5 };
    } };
}
/* simulate_receding_horizon: apply the first move, stop when ||x|| < tol or after `steps` moves */
function recedingHorizon(mpc, x0, steps = 25, tol = 1e-3) {
  let x = x0.slice(); const xs = [x.slice()], us = [], objs = [], values = [], plans = []; let infeasible = false;
  for (let t = 0; t < steps; t++) {
    const sol = mpc.solve(x); if (!sol.feasible) { infeasible = true; break; }
    us.push(sol.U[0]); objs.push(sol.objective); values.push(sol.value); plans.push(sol.X);
    x = mv(mpc.A, x).map((v, i) => v + mpc.B[i][0] * sol.U[0]); xs.push(x.slice());
    if (Math.hypot(...x) < tol) break;
  }
  return { xs, us, objs, values, plans, infeasible };
}

/* ---- Part VI: H-polytopes in 2-D: vertices by pairwise intersection, redundancy by the vertex test ---- */
function vertices2d(H, h, tol = 1e-8) {
  const pts = [];
  for (let i = 0; i < H.length; i++) for (let j = i + 1; j < H.length; j++) {
    const det = H[i][0] * H[j][1] - H[i][1] * H[j][0]; if (Math.abs(det) < 1e-10) continue;
    const x = (h[i] * H[j][1] - H[i][1] * h[j]) / det, y = (H[i][0] * h[j] - h[i] * H[j][0]) / det;
    if (H.every((row, k) => row[0] * x + row[1] * y <= h[k] + tol)) pts.push([x, y]);
  }
  const uniq = []; for (const p of pts) if (!uniq.some(q => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-7)) uniq.push(p);
  if (uniq.length < 3) return uniq;
  const cx = uniq.reduce((s, p) => s + p[0], 0) / uniq.length, cy = uniq.reduce((s, p) => s + p[1], 0) / uniq.length;
  return uniq.sort((p, q) => Math.atan2(p[1] - cy, p[0] - cx) - Math.atan2(q[1] - cy, q[0] - cx));
}
function removeRedundancy(H, h) {
  const V = vertices2d(H, h); if (V.length < 3) return { H, h };
  const keepH = [], keeph = [];
  for (let i = 0; i < H.length; i++) {
    const m = Math.max(...V.map(v => H[i][0] * v[0] + H[i][1] * v[1]));
    if (m > h[i] - 1e-8 && !keepH.some((row, k) => Math.abs(row[0] - H[i][0]) < 1e-9 && Math.abs(row[1] - H[i][1]) < 1e-9 && Math.abs(keeph[k] - h[i]) < 1e-9)) { keepH.push(H[i]); keeph.push(h[i]); }
  }
  return { H: keepH, h: keeph };
}
const polyBox = (lo, hi) => ({ H: [[1, 0], [0, 1], [-1, 0], [0, -1]], h: [hi[0], hi[1], -lo[0], -lo[1]] });
const polyIntersect = (P, Qp) => removeRedundancy([...P.H, ...Qp.H], [...P.h, ...Qp.h]);
const preimage = (S, A) => removeRedundancy(S.H.map(row => mv(tr(A), row)), S.h.slice());   // {x : A x in S}
const sameSet = (P, Qp, tol = 1e-6) => { const V1 = vertices2d(P.H, P.h), V2 = vertices2d(Qp.H, Qp.h); const inside = (V, R) => V.every(v => R.H.every((row, k) => row[0] * v[0] + row[1] * v[1] <= R.h[k] + tol)); return V1.length > 0 && V2.length > 0 && inside(V1, Qp) && inside(V2, P); };
/* maximal positively invariant subset of X under x+ = A x: Omega <- X ∩ Pre(Omega) until it stops changing */
function maximalInvariant(A, X, maxIter = 100) {
  let Om = removeRedundancy(X.H, X.h); const history = [Om];
  for (let it = 0; it < maxIter; it++) { const nxt = polyIntersect(X, preimage(Om, A)); history.push(nxt); if (sameSet(nxt, Om)) return { set: nxt, history }; Om = nxt; }
  return { set: Om, history };
}
/* NET-END */

/* ---------------- the chapter's system ---------------- */
const Ts = 1, A = [[1, Ts], [0, 1]], B = [[0.5 * Ts * Ts], [Ts]];
const Qn = [[1, 0], [0, 0.2]], Rn = [[0.1]];                       // Part III and V
const LQR = dlqr(A, B, Qn, Rn);
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
/* short legend lines inside a chart; on narrow charts the lines are split further at " · " */
function legend(ctx, lines, x, y, narrow) { ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; const out = narrow ? lines.flatMap(l => l.split(" · ")) : lines; out.forEach((l, i) => ctx.fillText(l, x, y + 13 * i)); }

/* =====================================================================
   Hero: receding-horizon MPC on the double integrator
   ===================================================================== */
const hs = { term: "none" };
const hero = { cache: {} };
function heroRun() {
  const N = +$("#hN").value, umax = +$("#hU").value, p0 = +$("#hP0").value, v0 = +$("#hV0").value, key = `${N}|${umax}|${p0}|${v0}|${hs.term}`;
  if (hero.cache[key]) return hero.cache[key];
  const Q = [[2, 0], [0, 0.4]], R = [[0.2]], xlb = [-10, -3], xub = [10, 3];
  let P = Q, terminal = null;
  if (hs.term !== "none") { const L = dlqr(A, B, Q, R); P = L.P; if (hs.term === "set") { const Xcl = removeRedundancy([...polyBox(xlb, xub).H, L.K[0], L.K[0].map(v => -v)], [...polyBox(xlb, xub).h, umax, umax]); terminal = maximalInvariant(L.Acl, Xcl).set; } }
  const mpc = makeMPC(A, B, Q, R, N, P, xlb, xub, [-umax], [umax], terminal), run = recedingHorizon(mpc, [p0, v0], 25, 1e-3);
  const ell = run.us.map((u, t) => quad(Q, run.xs[t]) + R[0][0] * u * u);
  const viol = run.values.slice(1).map((v, t) => v - run.values[t] + ell[t]);   // V(x_{t+1}) - V(x_t) + l(x_t,u_t): <= 0 under the theorem
  return hero.cache[key] = Object.assign(run, { N, umax, p0, v0, ell, viol, terminal, closedLoopCost: ell.reduce((s, v) => s + v, 0) });
}
function drawHero() {
  const r = heroRun(), cv = $("#mpcC"), narrow = narrowOf(cv), T = Math.max(12, r.xs.length), xs = r.xs, w = cv.clientWidth || cv.parentElement.clientWidth;
  const h = narrow ? 420 : 360, { ctx } = fit(cv, w / h), padL = 44, padR = 10, ph = (h - 40) / 3, X = t => padL + t / (T - 1) * (w - padL - padR);
  const panel = (k, y0, y1, ticks, label) => { const top = 8 + k * ph, bot = top + ph - 26, Y = v => bot - (v - y0) / (y1 - y0) * (bot - top); ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle"; ctx.textAlign = "right"; for (const t of ticks) { ctx.strokeStyle = css("--grid"); seg(ctx, padL, Y(t), w - padR, Y(t)); ctx.fillStyle = css("--muted"); ctx.fillText(String(t), padL - 5, Y(t)); } ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillText(label, padL + 4, top + 10); return { Y, top, bot }; };
  const pMax = Math.max(10, Math.ceil(Math.abs(r.p0))), P1 = panel(0, -pMax * 1.35, pMax * 1.35, [-pMax, 0, pMax], "position p"), P2 = panel(1, -4.2, 4.2, [-3, 0, 3], "velocity v, bound ±3"), P3 = panel(2, -r.umax * 1.5, r.umax * 1.5, [-r.umax, 0, r.umax].map(v => +v.toFixed(2)), "input u, bound ±" + fmt(r.umax, 1));
  const line = (pts, Y, token, width = 2) => { ctx.strokeStyle = css(token); ctx.lineWidth = width; ctx.beginPath(); pts.forEach((v, t) => t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v))); ctx.stroke(); ctx.fillStyle = css(token); pts.forEach((v, t) => { ctx.beginPath(); ctx.arc(X(t), Y(v), 2.6, 0, 2 * Math.PI); ctx.fill(); }); };
  for (const [P, v] of [[P2, 3], [P2, -3], [P3, r.umax], [P3, -r.umax]]) { ctx.strokeStyle = alpha("--warm", 0.7); ctx.setLineDash([5, 4]); seg(ctx, padL, P.Y(v), w - padR, P.Y(v)); ctx.setLineDash([]); }
  /* the plan made at the chosen step, as a ghost */
  const tp = clamp(+$("#hStep").value, 0, r.plans.length - 1); $("#hStep").max = Math.max(0, r.plans.length - 1);
  line(xs.map(x => x[0]), P1.Y, "--post"); line(xs.map(x => x[1]), P2.Y, "--post");
  let planGap = 0;
  if (r.plans[tp]) { const plan = [r.xs[tp], ...r.plans[tp]]; plan.forEach((x, i) => { const act = r.xs[tp + i]; if (act) planGap = Math.max(planGap, Math.abs(x[0] - act[0]), Math.abs(x[1] - act[1])); }); ctx.strokeStyle = css("--warm"); ctx.setLineDash([3, 3]); ctx.lineWidth = 2; [[0, P1], [1, P2]].forEach(([k, P]) => { ctx.beginPath(); plan.forEach((x, i) => i ? ctx.lineTo(X(tp + i), P.Y(x[k])) : ctx.moveTo(X(tp), P.Y(x[k]))); ctx.stroke(); }); ctx.setLineDash([]); }
  ctx.strokeStyle = css("--c5"); ctx.lineWidth = 2; ctx.beginPath(); r.us.forEach((u, t) => { if (t === 0) ctx.moveTo(X(0), P3.Y(u)); else ctx.lineTo(X(t), P3.Y(r.us[t - 1])); ctx.lineTo(X(t), P3.Y(u)); ctx.lineTo(X(t + 1), P3.Y(u)); }); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; for (const t of [0, 5, 10, 15, 20, 25]) if (t < T) ctx.fillText(String(t), X(t), h - 6);
  $("#hNV").textContent = r.N; $("#hUV").textContent = fmt(r.umax, 1); $("#hP0V").textContent = fmt(r.p0, 1); $("#hV0V").textContent = fmt(r.v0, 1); $("#hStepV").textContent = tp;
  const last = r.xs[r.xs.length - 1], nb = r.N === 8 && r.umax === 1 && r.p0 === -7 && r.v0 === 0 && hs.term === "none";
  $("#rMoves").textContent = r.infeasible ? "infeasible" : String(r.us.length); $("#rFinal").textContent = `(${fmt(last[0], 4)}, ${fmt(last[1], 4)})`; $("#rCost").textContent = fmt(r.closedLoopCost, 2);
  const bad = r.viol.filter(v => v > 1e-6).length; $("#rLyap").textContent = r.viol.length ? `${bad} of ${r.viol.length}` : "–";
  let text = r.infeasible ? (r.us.length === 0 ? `No admissible ${r.N}-step plan exists from (${fmt(r.p0, 1)}, ${fmt(r.v0, 1)}): the first QP is infeasible and the loop never starts. A state can satisfy the constraints and still have no way to keep satisfying them (Part 4). ` : `From (${fmt(r.p0, 1)}, ${fmt(r.v0, 1)}) the QP was feasible for ${r.us.length} step${r.us.length === 1 ? "" : "s"} and then found no admissible plan: feasible now did not mean feasible later. This loss of recursive feasibility is what the terminal set of Part 7 rules out, at the price of refusing some starts up front. `) : `${r.us.length} moves${nb ? ", the notebook's run" : ""}: the loop stops when [[\\lVert x \\rVert < 10^{-3}]], here at (${fmt(last[0], 4)}, ${fmt(last[1], 4)}). The velocity peaks at ${fmt(Math.max(...xs.map(x => x[1])), 2)} and the input sits on its bound for ${r.us.filter(u => Math.abs(Math.abs(u) - r.umax) < 1e-6).length} of the moves. `;
  if (!r.infeasible) {
    text += `The dashed orange curve is the plan made at step ${tp}: ${r.N} predicted states, of which only the first move is applied before the problem is solved again. ${planGap < 5e-4 ? "It coincides with what was then applied: the model is exact and nothing disturbs the cart, so re-solving changes nothing." : planGap < 0.05 ? `It differs from what was then applied by at most ${planGap.toFixed(3)}: with an exact model and no disturbance, re-solving changes the plan only where the end of the horizon moved.` : `It differs from what was then applied by up to ${planGap.toFixed(2)} in position or velocity: the short horizon made the plan end where the next solve, seeing one step further, chose differently.`} `;
    text += hs.term === "none" ? `With the notebook's setting, terminal cost [[P = Q]] and no terminal set, the optimal value decreased by at least the stage cost on ${r.viol.length - bad} of ${r.viol.length} steps; the theorem of Part 7 is not in force, and the convergence seen here is luck of the example, not a guarantee.` : hs.term === "cost" ? `With the LQR matrix [[P]] as terminal cost but no terminal set, the value decreased by at least the stage cost on ${r.viol.length - bad} of ${r.viol.length} steps: the terminal controller may violate the input bound beyond the horizon, so the proof's second condition can fail.` : `With the LQR terminal cost and the invariant terminal set [[\\mathcal X_f]] (${r.terminal.h.length} inequalities), the value [[V_N^*(x_t)]] decreased by at least [[\\ell(x_t, u_t)]] on every one of the ${r.viol.length} steps, as Part 7 proves it must${bad ? ` (${bad} exceptions within solver tolerance)` : ""}.`;
  }
  caption($("#heroStatus"), text);
}
["#hN", "#hU", "#hP0", "#hV0", "#hStep"].forEach(id => $(id).addEventListener("input", drawHero));
$$("#demo [data-term]").forEach(bt => bt.addEventListener("click", () => { hs.term = bt.dataset.term; $$("#demo [data-term]").forEach(b => b.setAttribute("aria-pressed", String(b === bt))); drawHero(); }));
$("#hReset").addEventListener("click", () => { $("#hN").value = 8; $("#hU").value = 1; $("#hP0").value = -7; $("#hV0").value = 0; $("#hStep").value = 0; hs.term = "none"; $$("#demo [data-term]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.term === "none"))); drawHero(); });

/* =====================================================================
   Part 1: a rollout in the phase plane
   ===================================================================== */
const SEQS = { notebook: [1, 1, 0, -1, -1], brake: [1, 1, -1, -1, 0], gentle: [0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5] };
let seqKey = "notebook";
function drawRoll() {
  const U = SEQS[seqKey].map(u => [u]), xs = rollout(A, B, [-5, 0], U), cv = $("#rollC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.4 : 2.2, { x: [-6, 3], y: [-1.5, 2.8], xlabel: "position p", ylabel: "velocity v", xticks: [-6, -4, -2, 0, 2], yticks: [-1, 0, 1, 2] }), { ctx, X, Y } = fr;
  ctx.strokeStyle = alpha("--ink", 0.25); seg(ctx, X(0), fr.pad.t, X(0), Y(fr.y0)); seg(ctx, X(fr.x0), Y(0), X(fr.x1), Y(0));
  plotLine(fr, xs.map(x => x[0]), xs.map(x => x[1]), { color: css("--post"), width: 2 });
  ctx.fillStyle = css("--post"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left";
  xs.forEach((x, k) => { ctx.beginPath(); ctx.arc(X(x[0]), Y(x[1]), 3.5, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = css("--muted"); ctx.fillText(k === 0 ? "x₀" : `x${k}`, X(x[0]) + 6, Y(x[1]) - 6); ctx.fillStyle = css("--post"); });
  $$("#rollW [data-seq]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.seq === seqKey)));
  caption($("#rollCap"), `Inputs [[u = (${U.map(u => u[0]).join(", ")})]] from [[x_0 = (-5, 0)]]: the states are ${xs.map(x => `(${fmt(x[0], 1)}, ${fmt(x[1], 1)})`).join(", ")}. Each step moves the velocity by [[u]] and the position by [[v + u/2]]: the curve is pinned to the dynamics, and no sequence of inputs can make the state jump sideways in this plane. ` + (seqKey === "notebook" ? "The notebook's sequence accelerates for two steps, coasts, and brakes for two: it arrives at [[(1, 0)]], overshooting the origin by one unit." : seqKey === "brake" ? "Braking one step earlier lands exactly on the origin at rest in four moves, with a fifth move of zero to stay there." : "Half-size inputs take eight moves to come to rest, a little short of the origin."));
}
$$("#rollW [data-seq]").forEach(bt => bt.addEventListener("click", () => { seqKey = bt.dataset.seq; drawRoll(); }));

/* =====================================================================
   Part 2: the prediction matrices
   ===================================================================== */
function drawPred() {
  const N = +$("#pN").value, { Sx, Su } = predictionMatrices(A, B, N), U = [0.8, 0.2, -0.4, -0.5, 0.3, 0.1].slice(0, N), x0 = [-5, 0];
  const Xc = mv(Sx, x0).map((v, r) => v + dot(Su[r], U)), Xr = rollout(A, B, x0, U.map(u => [u])).slice(1).flat(), err = Math.max(...Xc.map((v, i) => Math.abs(v - Xr[i])));
  const table = (M, name) => `<div class="mat"><span class="mlabel">${name}</span><table class="num">` + M.map(r => "<tr>" + r.map(v => `<td>${Number.isInteger(v) ? v : fmt(v, 1)}</td>`).join("") + "</tr>").join("") + "</table></div>";
  $("#pMats").innerHTML = table(Sx, `S<sub>x</sub> (${2 * N}×2)`) + table(Su, `S<sub>u</sub> (${2 * N}×${N})`);
  $("#pNV").textContent = N;
  caption($("#pCap"), `Horizon [[N = ${N}]]: [[\\mathcal S_x]] stacks [[A, A^2, \\dots, A^{${N}}]], and row block [[i]] of [[\\mathcal S_u]] holds [[A^{i-1}B, \\dots, B]] followed by zeros, because [[u_j]] cannot influence [[x_i]] for [[j \\ge i]]. For the double integrator [[A^i = \\begin{pmatrix} 1 & i \\\\ 0 & 1 \\end{pmatrix}]] and [[A^iB = (i + \\tfrac12,\\ 1)]], which is what the columns show. With [[x_0 = (-5, 0)]] and [[U = (${U.join(", ")})]], the condensed prediction and a step-by-step rollout agree to ${err.toExponential(1)}.`);
}
$("#pN").addEventListener("input", drawPred);

/* =====================================================================
   Part 5: LQR, the DARE and the Lyapunov decrease
   ===================================================================== */
function drawLqr() {
  const q2 = +$("#lQ").value, r = +$("#lR").value, Q = [[1, 0], [0, q2]], R = [[r]], L = dlqr(A, B, Q, R);
  const resid = maxAbs(add(sub(mm(tr(L.Acl), mm(L.P, L.Acl)), L.P), add(Q, mm(tr(L.K), mm(R, L.K)))));
  let x = [-5, 1.5]; const Vs = [], traj = [];
  for (let k = 0; k < 25; k++) { Vs.push(quad(L.P, x)); traj.push(x); x = mv(L.Acl, x); }
  const cv = $("#lqrC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, 24], y: [-14, 2], xlabel: "time step k", yticks: [-12, -8, -4, 0], yfmt: v => "10" + String(v).replace("-", "⁻").replace(/\d/g, ch => "⁰¹²³⁴⁵⁶⁷⁸⁹"[ch]) }), { ctx } = fr;
  plotLine(fr, Vs.map((_, k) => k), Vs.map(v => Math.max(-14, Math.log10(Math.max(v, 1e-300)))), { color: css("--post"), width: 2.2 });
  legend(ctx, ["V(x_k) = x_k' P x_k along u = -Kx · from (-5, 1.5)"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#lQV").textContent = fmt(q2, 2); $("#lRV").textContent = fmt(r, 2);
  const nb = q2 === 0.2 && r === 0.1, [e1] = L.eig, rho = specRadius(L.Acl);
  caption($("#lqrCap"), `[[Q = \\mathrm{diag}(1, ${fmt(q2, 2)})]], [[R = ${fmt(r, 2)}]]${nb ? ", the notebook's numbers" : ""}: [[K = (${fmt(L.K[0][0], 4)},\\ ${fmt(L.K[0][1], 4)})]], [[P = \\begin{pmatrix} ${fmt(L.P[0][0], 4)} & ${fmt(L.P[0][1], 4)} \\\\ ${fmt(L.P[1][0], 4)} & ${fmt(L.P[1][1], 4)} \\end{pmatrix}]], closed-loop eigenvalues [[${fmt(e1[0], 4)} \\pm ${fmt(Math.abs(e1[1]), 4)}i]]${Math.abs(e1[1]) < 1e-12 ? ` and ${fmt(L.eig[1][0], 4)}` : ""}, spectral radius ${fmt(rho, 4)}. The identity [[A_{\\mathrm{cl}}^\\top P A_{\\mathrm{cl}} - P = -(Q + K^\\top R K)]] holds to ${resid.toExponential(1)}. Along the loop [[V]] falls by exactly [[\\ell(x, -Kx)]] at every step: ${fmt(Vs[0], 2)}, ${fmt(Vs[1], 3)}, ${fmt(Vs[2], 4)}, … ` + (r >= 1 ? "A heavy input penalty makes the controller timid: the eigenvalues move towards 1 and the decrease slows." : r <= 0.02 ? "A cheap input gives an aggressive controller: the eigenvalues approach 0 and the loop is nearly dead-beat; its first move from this state is " + fmt(-dot(L.K[0], [-5, 1.5]), 1) + ", far beyond the bound of 1 that the constrained examples impose." : "The unconstrained LQR move from this state is " + fmt(-dot(L.K[0], [-5, 1.5]), 2) + "; Part 6 asks where that move respects the bound of 1."));
}
$("#lQ").addEventListener("input", drawLqr); $("#lR").addEventListener("input", drawLqr);

/* =====================================================================
   Part 6: the maximal positively invariant set under LQR
   ===================================================================== */
function drawInv() {
  const umax = +$("#iU").value, vmax = +$("#iV").value, K = LQR.K[0], box = polyBox([-8, -vmax], [8, vmax]);
  const Xcl = removeRedundancy([...box.H, K, K.map(v => -v)], [...box.h, umax, umax]), { set: Xf, history } = maximalInvariant(LQR.Acl, Xcl);
  const Vb = vertices2d(box.H, box.h), Vc = vertices2d(Xcl.H, Xcl.h), Vf = vertices2d(Xf.H, Xf.h);
  const cv = $("#invC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.3 : 1.9, { x: [-9, 9], y: [-5.5, 5.5], xlabel: "position p", ylabel: "velocity v", xticks: [-8, -4, 0, 4, 8], yticks: [-4, -2, 0, 2, 4] }), { ctx, X, Y } = fr;
  const poly = (V, fill, stroke) => { if (V.length < 3) return; ctx.beginPath(); V.forEach((v, i) => i ? ctx.lineTo(X(v[0]), Y(v[1])) : ctx.moveTo(X(v[0]), Y(v[1]))); ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); };
  poly(Vb, alpha("--ink", 0.06), alpha("--ink", 0.35)); poly(Vc, alpha("--c5", 0.18), css("--c5")); poly(Vf, alpha("--post", 0.35), css("--post"));
  /* one LQR trajectory from a vertex of Xf, to show it stays inside */
  if (Vf.length) { let x = Vf[0]; ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(X(x[0]), Y(x[1])); for (let k = 0; k < 8; k++) { x = mv(LQR.Acl, x); ctx.lineTo(X(x[0]), Y(x[1])); } ctx.stroke(); }
  legend(ctx, ["grey: state box · purple: X_cl, |−Kx| ≤ u_max", "blue: X_f, maximal invariant · black: LQR path"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#iUV").textContent = fmt(umax, 1); $("#iVV").textContent = fmt(vmax, 1);
  const nb = umax === 1 && vmax === 3, shrink = Vf.length && Vc.length ? 1 - polyArea(Vf) / polyArea(Vc) : 0;
  caption($("#invCap"), `[[u_{\\max} = ${fmt(umax, 1)}]], [[|v| \\le ${fmt(vmax, 1)}]]${nb ? ", the notebook's setting" : ""}: [[\\mathcal X_{\\mathrm{cl}}]] has ${Vc.length} vertices${Vc.length ? ` (${Vc.map(v => `(${fmt(v[0], 2)}, ${fmt(v[1], 2)})`).join(", ")})` : ""}; the fixed-point iteration stops after ${history.length - 1} iteration${history.length - 1 === 1 ? "" : "s"} with [[\\mathcal X_f]] of ${Vf.length} vertices${Vf.length ? ` (${Vf.map(v => `(${fmt(v[0], 2)}, ${fmt(v[1], 2)})`).join(", ")})` : ""}, ${fmt(100 * shrink, 0)}% smaller in area. The black path is the LQR loop from a vertex of [[\\mathcal X_f]]: it never leaves. ` + (umax >= 2.5 ? "With a generous input bound the input constraint hardly binds and [[\\mathcal X_f]] nearly fills the admissible box." : umax <= 0.4 ? "With a tight input bound, [[\\mathcal X_f]] is a thin sliver: from a state outside it the LQR move would exceed the bound, so a terminal set must be this small for the proof to hold." : "The two sloped sides of [[\\mathcal X_f]] are the input bound; the invariant set cuts the admissible set where one LQR step would leave it."));
}
const polyArea = V => Math.abs(V.reduce((s, v, i) => { const w = V[(i + 1) % V.length]; return s + v[0] * w[1] - w[0] * v[1]; }, 0)) / 2;
$("#iU").addEventListener("input", drawInv); $("#iV").addEventListener("input", drawInv);

/* =====================================================================
   Part 9: sampled states and their convex hull
   ===================================================================== */
const TRAJ = [[[-5, 0], [-4.2, 1], [-2.7, 1.4], [-1.2, 1], [0, 0]], [[-5, 0], [-4, 1.4], [-2.2, 1.6], [-0.8, 0.8], [0, 0]], [[-5, 0], [-3.8, 1.7], [-1.8, 1.5], [-0.5, 0.6], [0, 0]]];
const ss = { probe: [-2.5, 0.8] };
function hull2d(pts) { const P = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]), cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); const lo = [], up = []; for (const p of P) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); } for (const p of P.slice().reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); } return lo.slice(0, -1).concat(up.slice(0, -1)); }
function drawSafe() {
  const pts = TRAJ.flat(), Hh = hull2d(pts), cv = $("#safeC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.1, { x: [-5.6, 0.6], y: [-0.4, 2.1], xlabel: "state component x₁", ylabel: "x₂", xticks: [-5, -4, -3, -2, -1, 0], yticks: [0, 1, 2] }), { ctx, X, Y } = fr; cv._fr = fr;
  ctx.beginPath(); Hh.forEach((v, i) => i ? ctx.lineTo(X(v[0]), Y(v[1])) : ctx.moveTo(X(v[0]), Y(v[1]))); ctx.closePath(); ctx.fillStyle = alpha("--post", 0.14); ctx.fill(); ctx.strokeStyle = alpha("--post", 0.6); ctx.stroke();
  const tokens = ["--post", "--warm", "--c4"];
  TRAJ.forEach((trj, i) => { plotLine(fr, trj.map(p => p[0]), trj.map(p => p[1]), { color: css(tokens[i]), width: 1.8 }); ctx.fillStyle = css(tokens[i]); trj.forEach(p => { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.2, 0, 2 * Math.PI); ctx.fill(); }); });
  const q = ss.probe, inside = Hh.every((v, i) => { const w = Hh[(i + 1) % Hh.length]; return (w[0] - v[0]) * (q[1] - v[1]) - (w[1] - v[1]) * (q[0] - v[0]) >= -1e-9; });
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), 5, 0, 2 * Math.PI); ctx.fill();
  const near = pts.map((p, k) => [Math.hypot(p[0] - q[0], p[1] - q[1]), k]).sort((a, b) => a[0] - b[0])[0], tk = Math.floor(near[1] / 5), st = near[1] % 5;
  caption($("#safeCap"), `The three coloured paths are the sampled safe set [[\\mathcal{SS}]]: 15 stored states with a known continuation to the origin. The shaded hull is the convex safe set [[\\mathcal{CS}]]. The black probe at [[(${fmt(q[0], 2)}, ${fmt(q[1], 2)})]] is ${inside ? "inside the hull: it can be written as a convex combination of stored states, and for a linear system with convex constraints the matching combination of their continuations is itself an admissible path to the origin." : "outside the hull: nothing stored certifies a safe continuation from here, so an LMPC would not accept it as a terminal state."} Its nearest stored state is step ${st} of iteration ${tk}, at distance ${fmt(near[0], 2)}. Tap or click to move the probe. (These paths are sketches drawn to look like decelerations; the notebook does not simulate them, and exercise 5 asks what their safety claim rests on.)`);
}
$("#safeC").addEventListener("click", ev => { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), fr = cv._fr; if (!fr) return; ss.probe = [fr.x0 + (ev.clientX - r.left - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), fr.y1 - (ev.clientY - r.top - fr.pad.t) / fr.ih * (fr.y1 - fr.y0)]; drawSafe(); });

/* =====================================================================
   Part 10: Frenet coordinates on a circular centreline
   ===================================================================== */
function drawFrenet() {
  const Rt = 10, s = +$("#fS").value, ey = +$("#fE").value, th = s / Rt, c = [Rt * Math.cos(th), Rt * Math.sin(th)], tg = [-Math.sin(th), Math.cos(th)], nm = [Math.cos(th), Math.sin(th)], veh = [c[0] + ey * nm[0], c[1] + ey * nm[1]];
  const cv = $("#frenetC"), w = cv.clientWidth || cv.parentElement.clientWidth, h = Math.min(w, 380), { ctx } = fit(cv, w / h), sc = h / 30, X = x => w / 2 + x * sc, Y = y => h / 2 - y * sc;
  ctx.strokeStyle = alpha("--ink", 0.5); ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.arc(X(0), Y(0), Rt * sc, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = alpha("--ink", 0.15); ctx.beginPath(); ctx.arc(X(0), Y(0), (Rt + 3) * sc, 0, 2 * Math.PI); ctx.stroke(); ctx.beginPath(); ctx.arc(X(0), Y(0), (Rt - 3) * sc, 0, 2 * Math.PI); ctx.stroke();
  const arrow = (from, to, token) => { ctx.strokeStyle = css(token); ctx.fillStyle = css(token); ctx.lineWidth = 2; seg(ctx, X(from[0]), Y(from[1]), X(to[0]), Y(to[1])); const ang = Math.atan2(-(to[1] - from[1]), to[0] - from[0]); ctx.beginPath(); ctx.moveTo(X(to[0]), Y(to[1])); ctx.lineTo(X(to[0]) - 9 * Math.cos(ang - 0.4), Y(to[1]) - 9 * Math.sin(ang - 0.4)); ctx.lineTo(X(to[0]) - 9 * Math.cos(ang + 0.4), Y(to[1]) - 9 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill(); };
  arrow(c, [c[0] + 3 * tg[0], c[1] + 3 * tg[1]], "--post"); if (Math.abs(ey) > 0.05) arrow(c, veh, "--warm");
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(c[0]), Y(c[1]), 4, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(veh[0]), Y(veh[1]), 6, 0, 2 * Math.PI); ctx.fill();
  legend(ctx, ["dashed: centreline, R = 10 · faint: edges, W = 3", "blue arrow: s direction · orange: e_y"], 8, 14, w < 520);
  $("#fSV").textContent = fmt(s, 1); $("#fEV").textContent = fmt(ey, 1);
  caption($("#frenetCap"), `[[s = ${fmt(s, 1)}]] along the centreline, [[e_y = ${fmt(ey, 1)}]] ${ey >= 0 ? "to the left" : "to the right"}: the reference point is at angle [[\\theta = s / R = ${fmt(th, 3)}]], so the vehicle sits at [[(X, Y) = ((R + e_y)\\cos\\theta,\\ (R + e_y)\\sin\\theta) = (${fmt(veh[0], 2)}, ${fmt(veh[1], 2)})]]. The track boundary is the one inequality [[|e_y| \\le W = 3]]${Math.abs(ey) > 3 ? ", violated here" : ", satisfied here"}; in Cartesian coordinates the same boundary is a curve. Along this circle the curvature is constant, [[\\kappa = 1/R = 0.1]]; a real track's [[\\kappa(s)]] varies, and the racing chapters plan speed against it.`);
}
$("#fS").addEventListener("input", drawFrenet); $("#fE").addEventListener("input", drawFrenet);

function drawAll() { drawHero(); drawRoll(); drawPred(); drawLqr(); drawInv(); drawSafe(); drawFrenet(); }
drawAll();
onRedraw(drawAll);
})();
