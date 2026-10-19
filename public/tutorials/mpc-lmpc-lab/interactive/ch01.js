/* Advanced MPC and Learning MPC, chapter 01, Convex optimisation and CFTOC: the live demos.
   The condensed CFTOC of Part IV (state, input, input-rate and terminal-equality constraints) is
   assembled and solved here by a primal-dual interior-point method that also returns the Lagrange
   multipliers, so the active constraints and their shadow prices can be read off. The KKT example,
   the dual function, the prediction matrices and the sparsity patterns are computed too. The
   unicycle solutions come from Python (ch01-data.js), since the notebook's SLSQP is not reproduced
   here. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp, fit } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const eye = n => zeros(n, n).map((row, i) => row.map((_, j) => i === j ? 1 : 0));
const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const mv = (A, x) => A.map(r => r.reduce((s, v, k) => s + v * x[k], 0));
const tr = A => A[0].map((_, j) => A.map(r => r[j]));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const quad = (P, x) => dot(x, mv(P, x));
function solve(A0, b0) {   // Gaussian elimination with partial pivoting
  const n = A0.length, A = A0.map((r, i) => [...r, b0[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    if (Math.abs(A[c][c]) < 1e-14) A[c][c] = 1e-14;
    for (let r = 0; r < n; r++) { if (r === c) continue; const m = A[r][c] / A[c][c]; if (!m) continue; for (let k = c; k <= n; k++) A[r][k] -= m * A[c][k]; }
  }
  return A.map((r, i) => r[n] / r[i]);
}
const eig2 = M => { const t = M[0][0] + M[1][1], d = M[0][0] * M[1][1] - M[0][1] * M[1][0], disc = Math.sqrt(Math.max(0, t * t / 4 - d)); return [t / 2 - disc, t / 2 + disc]; };

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
/* D such that [u0, u1-u0, ..., u_{N-1}-u_{N-2}] = D U (scalar input) */
function differenceMatrix(N) { const D = zeros(N, N); D[0][0] = 1; for (let k = 1; k < N; k++) { D[k][k - 1] = -1; D[k][k] = 1; } return D; }

/* primal-dual interior point for  min 1/2 z'Hz + f'z  s.t.  G z <= g,  E z = e;  returns multipliers too */
function solveQP(H, f, G, g, E = [], e = [], iters = 80) {
  const n = f.length, m = g.length, p = e.length;
  let z = new Array(n).fill(0), lam = new Array(m).fill(1), nu = new Array(p).fill(0), s = g.map((gi, i) => Math.max(gi - dot(G[i], z), 1));
  const Gt = m ? tr(G) : null, Et = p ? tr(E) : null;
  for (let it = 0; it < iters; it++) {
    const rd = mv(H, z).map((v, i) => v + f[i] + (m ? dot(Gt[i], lam) : 0) + (p ? dot(Et[i], nu) : 0));
    const rp = G.map((row, i) => dot(row, z) + s[i] - g[i]), re = E.map((row, i) => dot(row, z) - e[i]);
    const mu = m ? dot(s, lam) / m : 0, sigma = 0.1, rc = s.map((si, i) => sigma * mu - si * lam[i]);
    const err = Math.max(...rd.map(Math.abs), ...rp.map(Math.abs), ...re.map(Math.abs), 0);
    if (err < 1e-10 && mu < 1e-11) break;
    const D = lam.map((l, i) => l / s[i]);
    const M = zeros(n + p, n + p), rhs = new Array(n + p).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let v = H[i][j]; for (let k = 0; k < m; k++) v += G[k][i] * D[k] * G[k][j]; M[i][j] = v; }
    for (let k = 0; k < p; k++) for (let i = 0; i < n; i++) { M[i][n + k] = E[k][i]; M[n + k][i] = E[k][i]; }
    const t = rc.map((v, i) => (v + lam[i] * rp[i]) / s[i]);
    for (let i = 0; i < n; i++) rhs[i] = -rd[i] - (m ? dot(Gt[i], t) : 0);
    for (let k = 0; k < p; k++) rhs[n + k] = -re[k];
    const sol = solve(M, rhs), dz = sol.slice(0, n), dnu = sol.slice(n);
    const ds = rp.map((v, i) => -v - dot(G[i], dz)), dl = rc.map((v, i) => (v - lam[i] * ds[i]) / s[i]);
    let a = 1; for (let i = 0; i < m; i++) { if (ds[i] < 0) a = Math.min(a, -0.99 * s[i] / ds[i]); if (dl[i] < 0) a = Math.min(a, -0.99 * lam[i] / dl[i]); }
    z = z.map((v, i) => v + a * dz[i]); s = s.map((v, i) => v + a * ds[i]); lam = lam.map((v, i) => v + a * dl[i]); nu = nu.map((v, i) => v + a * dnu[i]);
  }
  return { z, lam, nu, obj: 0.5 * quad(H, z) + dot(f, z), maxViolation: Math.max(...g.map((gi, i) => dot(G[i], z) - gi), ...e.map((ei, i) => Math.abs(dot(E[i], z) - ei)), 0) };
}

/* Part IV: the condensed linear-quadratic CFTOC on the double integrator (Ts = 0.5) */
function buildCFTOC({ N = 12, x0 = [-4, 0], umax = 0.8, dumax = 0.5, pmax = 5, vmax = 2, terminal = true, uPrev = 0, Q = [[1, 0], [0, 0.2]], P = [[8, 0], [0, 1]], R = 0.1 } = {}) {
  const Ts = 0.5, A = [[1, Ts], [0, 1]], B = [[0.5 * Ts * Ts], [Ts]], n = 2, { Sx, Su } = predictionMatrices(A, B, N);
  const Qbar = zeros(N * n, N * n); for (let i = 0; i < N; i++) { const W = i === N - 1 ? P : Q; for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Qbar[i * n + r][i * n + c] = W[r][c]; }
  const Rbar = zeros(N, N); for (let i = 0; i < N; i++) Rbar[i][i] = R;
  const SuTQ = mm(tr(Su), Qbar), H = mm(SuTQ, Su).map((r, i) => r.map((v, j) => 2 * (v + Rbar[i][j]))), f = mv(mm(SuTQ, Sx), x0).map(v => 2 * v);
  const constant = quad(mm(tr(Sx), mm(Qbar, Sx)), x0), Sxx = mv(Sx, x0);
  const G = [], g = [], tags = [];   // tags identify each inequality: {kind, step}
  for (let k = 0; k < N; k++) { const rp = Su[k * n], rv = Su[k * n + 1]; G.push(rp); g.push(pmax - Sxx[k * n]); tags.push({ kind: "position", step: k + 1 }); G.push(rp.map(v => -v)); g.push(pmax + Sxx[k * n]); tags.push({ kind: "position", step: k + 1 }); G.push(rv); g.push(vmax - Sxx[k * n + 1]); tags.push({ kind: "velocity", step: k + 1 }); G.push(rv.map(v => -v)); g.push(vmax + Sxx[k * n + 1]); tags.push({ kind: "velocity", step: k + 1 }); }
  const D = differenceMatrix(N);
  for (let k = 0; k < N; k++) { const off = k === 0 ? -uPrev : 0; G.push(D[k]); g.push(dumax - off); tags.push({ kind: "rate", step: k }); G.push(D[k].map(v => -v)); g.push(dumax + off); tags.push({ kind: "rate", step: k }); }
  for (let k = 0; k < N; k++) { const ek = new Array(N).fill(0); ek[k] = 1; G.push(ek); g.push(umax); tags.push({ kind: "input", step: k }); G.push(ek.map(v => -v)); g.push(umax); tags.push({ kind: "input", step: k }); }
  const E = terminal ? [Su[(N - 1) * n], Su[(N - 1) * n + 1]] : [], e = terminal ? [-Sxx[(N - 1) * n], -Sxx[(N - 1) * n + 1]] : [];
  return { A, B, N, Ts, x0, Sx, Su, H, f, constant, G, g, tags, E, e, umax, dumax, pmax, vmax, terminal };
}
function solveCFTOC(prob) {
  const r = solveQP(prob.H, prob.f, prob.G, prob.g, prob.E, prob.e), N = prob.N, n = 2, Xs = mv(prob.Sx, prob.x0).map((v, i) => v + dot(prob.Su[i], r.z)), X = [];
  for (let i = 0; i < N; i++) X.push([Xs[i * n], Xs[i * n + 1]]);
  const active = prob.tags.map((t, i) => ({ ...t, lam: r.lam[i], slack: prob.g[i] - dot(prob.G[i], r.z) })).filter(t => t.lam > 1e-6);
  return { U: r.z, X, cost: r.obj + prob.constant, objective: r.obj, lam: r.lam, nu: r.nu, active, feasible: r.maxViolation < 1e-6 };
}

/* Part II: the analytic QP  min ||z - c||^2  s.t.  a'z <= b  with a = (1, 1): KKT gives lambda = max(a'c - b, 0) (since ||a||^2 = 2), z = c - (lambda/2) a */
function kktExample(c, b) { const lam = Math.max(c[0] + c[1] - b, 0), z = [c[0] - lam / 2, c[1] - lam / 2]; return { z, lam, fstar: (z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2, objNoConst: (z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2 - dot(c, c) }; }
/* its dual function d(lambda) = -1/2 (q + a lambda)' P^{-1} (q + a lambda) - b lambda, with P = 2I, q = -2c */
function dualValue(c, b, lam) { const y = [-2 * c[0] + lam, -2 * c[1] + lam]; return -0.25 * (y[0] * y[0] + y[1] * y[1]) - b * lam; }
/* NET-END */

/* ---------------- helpers ---------------- */
function legend(ctx, lines, x, y, narrow) { ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; const out = narrow ? lines.flatMap(l => l.split(" · ")) : lines; out.forEach((l, i) => ctx.fillText(l, x, y + 13 * i)); }
const kindName = { position: "position bound", velocity: "velocity bound", rate: "rate bound", input: "input bound" };

/* =====================================================================
   Hero: the CFTOC of Part IV, live
   ===================================================================== */
const hero = { cache: {} };
function heroRun() {
  const N = +$("#hN").value, umax = +$("#hU").value, dumax = +$("#hD").value, p0 = +$("#hP0").value, v0 = +$("#hV0").value, terminal = $("#hTerm").getAttribute("aria-pressed") === "true";
  const key = [N, umax, dumax, p0, v0, terminal].join("|"); if (hero.cache[key]) return hero.cache[key];
  const prob = buildCFTOC({ N, umax, dumax, x0: [p0, v0], terminal }), sol = solveCFTOC(prob);
  return hero.cache[key] = { prob, sol, N, umax, dumax, p0, v0, terminal };
}
function drawHero() {
  const r = heroRun(), { prob, sol } = r, cv = $("#cftocC"), narrow = narrowOf(cv), w = cv.clientWidth || cv.parentElement.clientWidth, h = narrow ? 430 : 370, { ctx } = fit(cv, w / h);
  const padL = 44, padR = 10, ph = (h - 40) / 3, T = r.N, X = k => padL + k / T * (w - padL - padR);
  const panel = (k, y0, y1, ticks, label) => { const top = 8 + k * ph, bot = top + ph - 26, Y = v => bot - (v - y0) / (y1 - y0) * (bot - top); ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle"; ctx.textAlign = "right"; for (const t of ticks) { ctx.strokeStyle = css("--grid"); seg(ctx, padL, Y(t), w - padR, Y(t)); ctx.fillStyle = css("--muted"); ctx.fillText(String(t), padL - 5, Y(t)); } ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillText(label, padL + 4, top + 10); return { Y, top, bot }; };
  const pm = Math.max(5, Math.ceil(Math.abs(r.p0)) + 1), P1 = panel(0, -pm * 1.3, pm * 1.3, [-5, 0, 5], "position p, bound ±5"), P2 = panel(1, -2.8, 2.8, [-2, 0, 2], "velocity v, bound ±2"), P3 = panel(2, -r.umax * 1.5, r.umax * 1.5, [-r.umax, 0, r.umax].map(v => +v.toFixed(2)), "input u, bound ±" + fmt(r.umax, 1) + ", rate ±" + fmt(r.dumax, 1));
  for (const [P, v] of [[P1, 5], [P1, -5], [P2, 2], [P2, -2], [P3, r.umax], [P3, -r.umax]]) { ctx.strokeStyle = alpha("--warm", 0.7); ctx.setLineDash([5, 4]); seg(ctx, padL, P.Y(v), w - padR, P.Y(v)); ctx.setLineDash([]); }
  if (!sol.feasible) { ctx.fillStyle = css("--warm"); ctx.font = "13px " + css("--font-ui"); ctx.textAlign = "center"; ctx.fillText("infeasible: no input sequence satisfies all the constraints", w / 2, h / 2); }
  else {
    const xs = [prob.x0, ...sol.X];
    const line = (pts, Y, token) => { ctx.strokeStyle = css(token); ctx.lineWidth = 2; ctx.beginPath(); pts.forEach((v, t) => t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v))); ctx.stroke(); ctx.fillStyle = css(token); pts.forEach((v, t) => { ctx.beginPath(); ctx.arc(X(t), Y(v), 2.6, 0, 2 * Math.PI); ctx.fill(); }); };
    line(xs.map(x => x[0]), P1.Y, "--post"); line(xs.map(x => x[1]), P2.Y, "--post");
    ctx.strokeStyle = css("--c5"); ctx.lineWidth = 2; ctx.beginPath(); sol.U.forEach((u, t) => { if (t === 0) ctx.moveTo(X(0), P3.Y(u)); else { ctx.lineTo(X(t), P3.Y(sol.U[t - 1])); ctx.lineTo(X(t), P3.Y(u)); } if (t === sol.U.length - 1) ctx.lineTo(X(t + 1), P3.Y(u)); }); ctx.stroke();
    /* mark the active constraints: a ring where a bound is touched, sized by its multiplier */
    const maxLam = Math.max(1e-9, ...sol.active.map(a => a.lam));
    for (const a of sol.active) { const rad = 4 + 7 * Math.sqrt(a.lam / maxLam); ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.6; ctx.beginPath(); if (a.kind === "input") ctx.arc(X(a.step + 0.5), P3.Y(sol.U[a.step]), rad, 0, 2 * Math.PI); else if (a.kind === "rate") ctx.arc(X(a.step), P3.Y(sol.U[a.step]), rad, 0, 2 * Math.PI); else if (a.kind === "position") ctx.arc(X(a.step), P1.Y(xs[a.step][0]), rad, 0, 2 * Math.PI); else ctx.arc(X(a.step), P2.Y(xs[a.step][1]), rad, 0, 2 * Math.PI); ctx.stroke(); }
    if (r.terminal) { ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(T), P1.Y(0), 5, 0, 2 * Math.PI); ctx.fill(); ctx.beginPath(); ctx.arc(X(T), P2.Y(0), 5, 0, 2 * Math.PI); ctx.fill(); }
  }
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; for (let k = 0; k <= T; k += (T > 14 ? 4 : 2)) ctx.fillText(fmt(k * prob.Ts, 1) + "s", X(k), h - 6);
  $("#hNV").textContent = r.N; $("#hUV").textContent = fmt(r.umax, 1); $("#hDV").textContent = fmt(r.dumax, 1); $("#hP0V").textContent = fmt(r.p0, 1); $("#hV0V").textContent = fmt(r.v0, 1);
  const nb = r.N === 12 && r.umax === 0.8 && r.dumax === 0.5 && r.p0 === -4 && r.v0 === 0 && r.terminal;
  if (!sol.feasible) { $("#rCost").textContent = "–"; $("#rAct").textContent = "–"; $("#rMax").textContent = "–"; $("#rTerm").textContent = "–"; caption($("#heroStatus"), `No input sequence satisfies every constraint from [[(${fmt(r.p0, 1)}, ${fmt(r.v0, 1)})]] with [[N = ${r.N}]]${r.terminal ? " and the terminal equality" : ""}: the QP is infeasible. The solver reports it; nothing else can be read from this run. A longer horizon, a looser input or rate bound, or dropping the terminal equality gives the optimiser something to work with.`); return; }
  const maxU = Math.max(...sol.U.map(Math.abs)), dU = sol.U.map((u, k) => u - (k ? sol.U[k - 1] : 0)), maxdU = Math.max(...dU.map(Math.abs)), xN = sol.X[sol.X.length - 1];
  const counts = {}; sol.active.forEach(a => counts[a.kind] = (counts[a.kind] || 0) + 1); const top = sol.active.slice().sort((a, b) => b.lam - a.lam)[0];
  $("#rCost").textContent = fmt(sol.cost, 3); $("#rAct").textContent = String(sol.active.length); $("#rMax").textContent = `${fmt(maxU, 3)} / ${fmt(maxdU, 3)}`; $("#rTerm").textContent = `(${fmt(Math.abs(xN[0]) < 5e-4 ? 0 : xN[0], 3)}, ${fmt(Math.abs(xN[1]) < 5e-4 ? 0 : xN[1], 3)})`;
  let text = `${nb ? "The notebook's problem: " : ""}cost [[J = ${fmt(sol.cost, 3)}]] (the QP's own objective is ${fmt(sol.objective, 2)}; the dropped constant [[x_0^\\top S_x^\\top \\bar Q S_x x_0 = ${fmt(prob.constant, 1)}]] is added back). The largest input is ${fmt(maxU, 3)} and the largest step change ${fmt(maxdU, 3)}, against bounds of ${fmt(r.umax, 1)} and ${fmt(r.dumax, 1)}. `;
  text += sol.active.length ? `${sol.active.length} inequality constraints are active (ringed; ring size follows the multiplier): ${Object.entries(counts).map(([k, v]) => `${v} ${kindName[k]}${v > 1 ? "s" : ""}`).join(", ")}. The most expensive is the ${kindName[top.kind]} at step ${top.step}, with multiplier [[\\lambda = ${fmt(top.lam, 2)}]]: relaxing that one bound by a small [[\\delta]] would lower the cost by about [[${fmt(top.lam, 2)}\\,\\delta]]. ` : "No inequality is active: the bounds do not bind and the problem is, locally, unconstrained. ";
  text += r.terminal ? `The terminal equality lands the state at [[x_N = (${fmt(Math.abs(xN[0]) < 5e-4 ? 0 : xN[0], 3)}, ${fmt(Math.abs(xN[1]) < 5e-4 ? 0 : xN[1], 3)})]] with multipliers [[\\nu = (${fmt(sol.nu[0], 2)}, ${fmt(sol.nu[1], 2)})]].` : `Without the terminal equality the horizon ends at [[x_N = (${fmt(xN[0], 3)}, ${fmt(xN[1], 3)})]], close to the origin but not on it; the terminal weight [[P]] pulls, the equality would pin.`;
  caption($("#heroStatus"), text);
}
["#hN", "#hU", "#hD", "#hP0", "#hV0"].forEach(id => $(id).addEventListener("input", drawHero));
$("#hTerm").addEventListener("click", () => { const b = $("#hTerm"); b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true")); drawHero(); });
$("#hReset").addEventListener("click", () => { $("#hN").value = 12; $("#hU").value = 0.8; $("#hD").value = 0.5; $("#hP0").value = -4; $("#hV0").value = 0; $("#hTerm").setAttribute("aria-pressed", "true"); drawHero(); });

/* =====================================================================
   Part 1: convex sets, the segment test
   ===================================================================== */
const cs = { pts: [[0.25, 0.30], [1.70, 0.45]], shape: "polytope", next: 0 };
const inPoly = ([x, y]) => x >= 0 && y >= 0 && x + y <= 2.4 && x <= 2 && y <= 1.8;
const inNotch = ([x, y]) => (x >= 0 && y >= 0 && x <= 2 && y <= 0.7) || (x >= 1.3 && x <= 2 && y >= 0 && y <= 1.8);   // an L: union of two boxes, not convex
const inside = p => cs.shape === "polytope" ? inPoly(p) : inNotch(p);
function drawSet() {
  const cv = $("#setC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.3 : 1.7, { x: [-0.3, 2.6], y: [-0.3, 2.1], xlabel: "z₁", ylabel: "z₂", xticks: [0, 1, 2], yticks: [0, 1, 2] }), { ctx, X, Y } = fr; cv._fr = fr;
  const V = cs.shape === "polytope" ? [[0, 0], [2, 0], [2, 0.4], [0.6, 1.8], [0, 1.8]] : [[0, 0], [2, 0], [2, 1.8], [1.3, 1.8], [1.3, 0.7], [0, 0.7]];
  ctx.beginPath(); V.forEach((v, i) => i ? ctx.lineTo(X(v[0]), Y(v[1])) : ctx.moveTo(X(v[0]), Y(v[1]))); ctx.closePath(); ctx.fillStyle = alpha("--post", 0.16); ctx.fill(); ctx.strokeStyle = css("--post"); ctx.lineWidth = 1.5; ctx.stroke();
  const [a, b] = cs.pts, n = 200; let out = 0; const segs = [];
  for (let i = 0; i <= n; i++) { const t = i / n, p = [t * a[0] + (1 - t) * b[0], t * a[1] + (1 - t) * b[1]]; const ok = inside(p); if (!ok) out++; segs.push([p, ok]); }
  for (let i = 1; i < segs.length; i++) { ctx.strokeStyle = segs[i][1] && segs[i - 1][1] ? css("--ink") : css("--warm"); ctx.lineWidth = 2.5; seg(ctx, X(segs[i - 1][0][0]), Y(segs[i - 1][0][1]), X(segs[i][0][0]), Y(segs[i][0][1])); }
  cs.pts.forEach((p, i) => { ctx.fillStyle = inside(p) ? css("--post") : css("--warm"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 6, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText(i ? "y" : "x", X(p[0]) + 8, Y(p[1]) - 6); });
  $$("#setW [data-shape]").forEach(bt => bt.setAttribute("aria-pressed", String(bt.dataset.shape === cs.shape)));
  const bothIn = inside(a) && inside(b);
  caption($("#setCap"), (cs.shape === "polytope" ? `The polytope [[\\{z : z \\ge 0,\\ z_1 + z_2 \\le 2.4,\\ z_1 \\le 2,\\ z_2 \\le 1.8\\}]], five half-spaces intersected, as in the notebook. ` : `An L-shaped set, the union of two boxes. A union of convex sets is in general not convex. `) + `The segment [[\\theta x + (1 - \\theta) y]] between [[x = (${fmt(a[0], 2)}, ${fmt(a[1], 2)})]] and [[y = (${fmt(b[0], 2)}, ${fmt(b[1], 2)})]] ` + (!bothIn ? "has an endpoint outside the set, so it says nothing about convexity; convexity is about segments between points of the set. " : out === 0 ? "stays inside: of 201 sample points along it, none leaves. " : `leaves the set: ${out} of its 201 sample points lie outside (drawn orange), so this set is not convex. `) + `Tap or click to move the points, alternating [[x]] and [[y]].`);
}
$("#setC").addEventListener("click", ev => { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), fr = cv._fr; if (!fr) return; cs.pts[cs.next] = [fr.x0 + (ev.clientX - r.left - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), fr.y1 - (ev.clientY - r.top - fr.pad.t) / fr.ih * (fr.y1 - fr.y0)]; cs.next = 1 - cs.next; drawSet(); });
$$("#setW [data-shape]").forEach(bt => bt.addEventListener("click", () => { cs.shape = bt.dataset.shape; drawSet(); }));

/* =====================================================================
   Part 1: a quadratic's level sets and its Hessian
   ===================================================================== */
function contours(fr, F, levels, g, color) {   // marching squares on the grid g (values F[i][j] at (g[i], g[j]))
  const { ctx, X, Y } = fr; ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.beginPath();
  const lerp = (p, q, fp, fq, lv) => { const t = (lv - fp) / (fq - fp); return [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]; };
  for (const lv of levels) for (let i = 0; i < g.length - 1; i++) for (let j = 0; j < g.length - 1; j++) {
    const c = [[g[i], g[j], F[i][j]], [g[i + 1], g[j], F[i + 1][j]], [g[i + 1], g[j + 1], F[i + 1][j + 1]], [g[i], g[j + 1], F[i][j + 1]]], pts = [];
    for (let k = 0; k < 4; k++) { const p = c[k], q = c[(k + 1) % 4]; if ((p[2] - lv) * (q[2] - lv) < 0) pts.push(lerp(p, q, p[2], q[2], lv)); }
    if (pts.length >= 2) { ctx.moveTo(X(pts[0][0]), Y(pts[0][1])); ctx.lineTo(X(pts[1][0]), Y(pts[1][1])); if (pts.length === 4) { ctx.moveTo(X(pts[2][0]), Y(pts[2][1])); ctx.lineTo(X(pts[3][0]), Y(pts[3][1])); } }
  }
  ctx.stroke();
}
const qs = { mode: "alpha" };
function drawQuad() {
  const al = +$("#qA").value, P = qs.mode === "alpha" ? [[2, al], [al, 1]] : qs.mode === "convex" ? [[3, 0.8], [0.8, 1.5]] : [[1, 0], [0, -0.6]], [e1, e2] = eig2(P);
  const cv = $("#quadC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.2 : 1.6, { x: [-2, 2], y: [-2, 2], xlabel: "z₁", ylabel: "z₂", xticks: [-2, -1, 0, 1, 2], yticks: [-2, -1, 0, 1, 2] }), { ctx, X, Y } = fr;
  const g = Array.from({ length: 81 }, (_, i) => -2 + 4 * i / 80), F = g.map(a => g.map(b => 0.5 * (P[0][0] * a * a + 2 * P[0][1] * a * b + P[1][1] * b * b)));
  const fmax = Math.max(...F.flat()), fmin = Math.min(...F.flat());
  contours(fr, F, Array.from({ length: 10 }, (_, k) => fmax * (k + 1) / 11), g, css("--post"));
  if (fmin < -1e-9) contours(fr, F, Array.from({ length: 6 }, (_, k) => fmin * (k + 1) / 7), g, css("--warm"));
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(0), Y(0), 3.5, 0, 2 * Math.PI); ctx.fill();
  if (e1 > 1e-9) { /* eigenvector of the small eigenvalue: the long axis */ const vx = P[0][1] !== 0 ? [e1 - P[1][1], P[0][1]] : (P[0][0] < P[1][1] ? [1, 0] : [0, 1]), nn = Math.hypot(...vx); ctx.strokeStyle = alpha("--ink", 0.5); ctx.setLineDash([3, 3]); seg(ctx, X(-1.8 * vx[0] / nn), Y(-1.8 * vx[1] / nn), X(1.8 * vx[0] / nn), Y(1.8 * vx[1] / nn)); ctx.setLineDash([]); }
  $("#qAV").textContent = fmt(al, 2); $("#qA").disabled = qs.mode !== "alpha"; $$("#quadW [data-q]").forEach(bt => bt.setAttribute("aria-pressed", String(bt.dataset.q === qs.mode)));
  const verdict = e1 > 1e-9 ? "positive definite: strictly convex, elliptical level sets with a single minimum at the origin" : e1 > -1e-9 ? "positive semidefinite but singular: convex, with a whole line of minimisers (the dashed direction), not strictly convex" : "indefinite: a saddle; the blue levels are positive, the orange negative, and the origin is a minimum along one direction and a maximum along the other";
  caption($("#quadCap"), `[[P = \\begin{pmatrix} ${fmt(P[0][0], 2)} & ${fmt(P[0][1], 2)} \\\\ ${fmt(P[1][0], 2)} & ${fmt(P[1][1], 2)} \\end{pmatrix}]], eigenvalues ${fmt(e1, 3)} and ${fmt(e2, 3)}: ${verdict}. ` + (qs.mode === "alpha" ? `For [[P = \\begin{pmatrix} 2 & \\alpha \\\\ \\alpha & 1 \\end{pmatrix}]] the determinant is [[2 - \\alpha^2]], so [[f]] is convex exactly when [[|\\alpha| \\le \\sqrt 2 \\approx 1.414]] (exercise 1); the trace is 3, so the eigenvalues are [[\\tfrac{3 \\pm \\sqrt{1 + 4\\alpha^2}}{2}]].` : qs.mode === "convex" ? "The notebook's convex example." : "The notebook's indefinite example: no amount of searching finds a minimum, because there is none; the function is unbounded below along [[z_2]]."));
}
$("#qA").addEventListener("input", () => { qs.mode = "alpha"; drawQuad(); });
$$("#quadW [data-q]").forEach(bt => bt.addEventListener("click", () => { qs.mode = bt.dataset.q; drawQuad(); }));

/* =====================================================================
   Part 2: KKT by hand, shadow prices and the dual
   ===================================================================== */
function drawKKT() {
  const c = [+$("#kC1").value, +$("#kC2").value], b = +$("#kB").value, k = kktExample(c, b);
  const cv = $("#kktC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.2 : 1.7, { x: [-0.6, 2.4], y: [-0.6, 2.2], xlabel: "z₁", ylabel: "z₂", xticks: [0, 1, 2], yticks: [0, 1, 2] }), { ctx, X, Y } = fr;
  /* feasible half-plane z1 + z2 <= b */
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  ctx.beginPath(); ctx.moveTo(X(-1), Y(b + 1)); ctx.lineTo(X(b + 1), Y(-1)); ctx.lineTo(X(-1), Y(-1)); ctx.closePath(); ctx.fillStyle = alpha("--post", 0.14); ctx.fill();
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 1.5; seg(ctx, X(-1), Y(b + 1), X(b + 1), Y(-1));
  /* level sets of ||z - c||^2 */
  for (const rad of [0.25, 0.5, 0.75, 1, 1.25]) { ctx.strokeStyle = alpha("--ink", 0.18); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X(c[0]), Y(c[1]), rad * fr.iw / (fr.x1 - fr.x0), 0, 2 * Math.PI); ctx.stroke(); }
  if (k.fstar > 1e-9) { ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(X(c[0]), Y(c[1]), Math.sqrt(k.fstar) * fr.iw / (fr.x1 - fr.x0), 0, 2 * Math.PI); ctx.stroke(); }
  /* gradient and the multiplier direction at z* */
  if (k.lam > 1e-9) { ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2; seg(ctx, X(k.z[0]), Y(k.z[1]), X(k.z[0] + k.lam / 2), Y(k.z[1] + k.lam / 2)); }
  ctx.restore();
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(c[0]), Y(c[1]), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(k.z[0]), Y(k.z[1]), 5.5, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText("c", X(c[0]) + 7, Y(c[1]) - 6); ctx.fillText("z*", X(k.z[0]) + 7, Y(k.z[1]) + 12);
  legend(ctx, ["shaded: z₁ + z₂ ≤ b · circles: level sets of ‖z − c‖²", "orange: active level, λa/2"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* dual function panel */
  const cv2 = $("#dualC"), fr2 = frame(cv2, narrow ? 1.5 : 2.4, { x: [0, 3], y: [-3.2, 1.2], xlabel: "λ", ylabel: "d(λ)", xticks: [0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1] }), ctx2 = fr2.ctx;
  const ls = Array.from({ length: 121 }, (_, i) => 3 * i / 120), ds = ls.map(l => dualValue(c, b, l));
  plotLine(fr2, ls, ds, { color: css("--post"), width: 2 });
  ctx2.strokeStyle = css("--warm"); ctx2.setLineDash([4, 4]); seg(ctx2, fr2.X(0), fr2.Y(k.objNoConst), fr2.X(3), fr2.Y(k.objNoConst)); ctx2.setLineDash([]);
  ctx2.fillStyle = css("--warm"); ctx2.beginPath(); ctx2.arc(fr2.X(k.lam), fr2.Y(dualValue(c, b, k.lam)), 5, 0, 2 * Math.PI); ctx2.fill();
  legend(ctx2, ["blue: the dual function d(λ) · dashed: the primal optimum p*"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  $("#kC1V").textContent = fmt(c[0], 2); $("#kC2V").textContent = fmt(c[1], 2); $("#kBV").textContent = fmt(b, 2);
  const nb = c[0] === 1.2 && c[1] === 0.8 && b === 1, act = k.lam > 1e-9;
  caption($("#kktCap"), `[[c = (${fmt(c[0], 2)}, ${fmt(c[1], 2)})]], [[b = ${fmt(b, 2)}]]${nb ? ", the notebook's numbers" : ""}. ` + (act ? `The unconstrained minimiser [[c]] violates [[z_1 + z_2 \\le b]] (${fmt(c[0] + c[1], 2)} > ${fmt(b, 2)}), so the constraint is active: stationarity [[2z - 2c + \\lambda a = 0]] gives [[z = c - \\tfrac{\\lambda}{2} a]], and [[a^\\top z = b]] gives [[\\lambda^* = a^\\top c - b = ${fmt(k.lam, 3)}]], [[z^* = (${fmt(k.z[0], 3)}, ${fmt(k.z[1], 3)})]], [[f^* = \\lambda^{*2}/2 = ${fmt(k.fstar, 3)}]]. The orange arrow at [[z^*]] is [[\\tfrac{\\lambda}{2} a]], the gradient of the objective pulled back onto the constraint: the shadow price. Relaxing [[b]] by [[\\delta]] lowers [[f^*]] by [[\\lambda^* \\delta]] to first order ([[df^*/db = -\\lambda^*]]); the notebook's numerical derivative of [[f^*(b)]] matches [[-\\lambda^* = ${fmt(-k.lam, 2)}]] to [[2 \\times 10^{-8}]]. ` : `The unconstrained minimiser [[c]] already satisfies [[z_1 + z_2 \\le b]] (${fmt(c[0] + c[1], 2)} ≤ ${fmt(b, 2)}), so [[z^* = c]], [[\\lambda^* = 0]] by complementary slackness and [[f^* = 0]]: the constraint is inactive and relaxing it buys nothing. `) + `On the right, the dual function [[d(\\lambda) = -\\tfrac14 \\lVert -2c + \\lambda a \\rVert^2 - b\\lambda]] is concave; its maximum ${fmt(dualValue(c, b, k.lam), 3)} at [[\\lambda = ${fmt(k.lam, 2)}]] equals the primal optimum [[p^* = f^* - c^\\top c = ${fmt(k.objNoConst, 3)}]] (the QP drops the constant [[c^\\top c]]): strong duality, zero gap. Every other [[\\lambda \\ge 0]] gives a lower bound, which is weak duality.`);
}
["#kC1", "#kC2", "#kB"].forEach(id => $(id).addEventListener("input", drawKKT));

/* =====================================================================
   Part 3/4: sparse versus condensed structure
   ===================================================================== */
function drawStruct() {
  const N = +$("#sN").value, n = +$("#sn").value, m = +$("#sm").value, nz = N * n + N * m;
  /* sparse: H block diagonal (dense blocks), dynamics equality rows [I, -A, -B] */
  const Hs = zeros(nz, nz); for (let k = 0; k < N; k++) { for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Hs[k * n + r][k * n + c] = 1; for (let r = 0; r < m; r++) for (let c = 0; c < m; c++) Hs[N * n + k * m + r][N * n + k * m + c] = 1; }
  const Es = zeros(N * n, nz); for (let k = 0; k < N; k++) for (let r = 0; r < n; r++) { for (let c = 0; c < n; c++) { Es[k * n + r][k * n + c] = 1; if (k) Es[k * n + r][(k - 1) * n + c] = 1; } for (let c = 0; c < m; c++) Es[k * n + r][N * n + k * m + c] = 1; }
  /* condensed: H dense (N m)^2, state-constraint rows F Su: block lower triangular */
  const Hc = zeros(N * m, N * m).map(r => r.map(() => 1)), Gc = zeros(N * n, N * m); for (let i = 0; i < N; i++) for (let r = 0; r < n; r++) for (let j = 0; j <= i; j++) for (let c = 0; c < m; c++) Gc[i * n + r][j * m + c] = 1;
  const count = M => M.flat().reduce((s, v) => s + v, 0);
  const draw = (id, M, title) => { const cv = $(id), w = cv.clientWidth || cv.parentElement.clientWidth, rows = M.length, cols = M[0].length, cell = Math.max(1, Math.floor(Math.min((w - 8) / cols, 160 / rows))), h = rows * cell + 22, { ctx } = fit(cv, w / h); ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText(`${title} ${rows}×${cols}`, 2, 12); const x0 = Math.max(2, (w - cols * cell) / 2); ctx.fillStyle = alpha("--ink", 0.08); ctx.fillRect(x0, 18, cols * cell, rows * cell); ctx.fillStyle = css("--post"); M.forEach((r, i) => r.forEach((v, j) => { if (v) ctx.fillRect(x0 + j * cell, 18 + i * cell, Math.max(1, cell - (cell > 3 ? 1 : 0)), Math.max(1, cell - (cell > 3 ? 1 : 0))); })); };
  draw("#sHs", Hs, "sparse H"); draw("#sEs", Es, "sparse dynamics E"); draw("#sHc", Hc, "condensed H"); draw("#sGc", Gc, "condensed F̄ S_u");
  $("#sNV").textContent = N; $("#snV").textContent = n; $("#smV").textContent = m;
  caption($("#structCap"), `[[n = ${n}]], [[m = ${m}]], [[N = ${N}]]: the sparse problem has [[Nn + Nm = ${nz}]] decision variables with a block-diagonal Hessian (${count(Hs)} nonzeros of ${nz * nz}) and [[Nn = ${N * n}]] dynamics equalities whose matrix has only ${count(Es)} nonzeros; the condensed problem has [[Nm = ${N * m}]] variables, a dense Hessian (${count(Hc)} nonzeros, every entry) and dense lower-triangular constraint rows (${count(Gc)} of ${N * n * N * m}). The sparse matrices grow linearly with [[N]], the condensed ones quadratically. For the notebook's exercise 6, [[n = 8]], [[m = 2]], [[N = 30]]: 300 sparse variables against 60 condensed; the sparse problem is five times larger and can still be faster, because a structure-exploiting solver factorises its banded KKT system in time linear in [[N]], while the dense condensed Hessian costs [[\\mathcal O((Nm)^3)]].`);
}
["#sN", "#sn", "#sm"].forEach(id => $(id).addEventListener("input", drawStruct));

/* =====================================================================
   Part 5: the unicycle (solutions from Python)
   ===================================================================== */
const UD = window.CH01_DATA.targets, uni = { target: "(2, 1, 0)" };
function drawUni() {
  const d = UD[uni.target], Z = d.Z, U = d.U, cv = $("#uniC"), narrow = narrowOf(cv), xs = Z.map(z => z[0]), ys = Z.map(z => z[1]);
  const x0 = Math.min(-0.4, ...xs) - 0.3, x1 = Math.max(0.4, ...xs) + 0.3, y0 = Math.min(-0.4, ...ys) - 0.3, y1 = Math.max(0.4, ...ys) + 0.3;
  const fr = frame(cv, narrow ? 1.2 : 1.6, { x: [x0, x1], y: [y0, y1], xlabel: "x", ylabel: "y" }), { ctx, X, Y } = fr;
  plotLine(fr, xs, ys, { color: css("--post"), width: 2 }); ctx.fillStyle = css("--post"); Z.forEach(z => { ctx.beginPath(); ctx.arc(X(z[0]), Y(z[1]), 2.2, 0, 2 * Math.PI); ctx.fill(); });
  /* heading ticks every 5 steps */
  ctx.strokeStyle = alpha("--ink", 0.6); ctx.lineWidth = 1.2; for (let k = 0; k < Z.length; k += 5) { const [x, y, th] = Z[k]; seg(ctx, X(x), Y(y), X(x + 0.18 * Math.cos(th)), Y(y - 0.18 * Math.sin(th) * -1)); }
  const zf = Z[Z.length - 1]; ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(0), Y(0), 5, 0, 2 * Math.PI); ctx.fill(); ctx.beginPath(); ctx.arc(X(zf[0]), Y(zf[1]), 5, 0, 2 * Math.PI); ctx.fill();
  const cv2 = $("#uniUC"), fr2 = frame(cv2, narrow ? 1.5 : 2.4, { x: [0, 6], y: [-1.1, 1.1], xlabel: "time [s]", xticks: [0, 2, 4, 6], yticks: [-1, -0.5, 0, 0.5, 1] }), ctx2 = fr2.ctx;
  for (const v of [1, -1]) { ctx2.strokeStyle = alpha("--warm", 0.7); ctx2.setLineDash([5, 4]); seg(ctx2, fr2.X(0), fr2.Y(v), fr2.X(6), fr2.Y(v)); ctx2.setLineDash([]); }
  const stepLine = (vals, token) => { ctx2.strokeStyle = css(token); ctx2.lineWidth = 2; ctx2.beginPath(); vals.forEach((v, k) => { const t = 0.2 * k; if (k === 0) ctx2.moveTo(fr2.X(0), fr2.Y(v)); else { ctx2.lineTo(fr2.X(t), fr2.Y(vals[k - 1])); ctx2.lineTo(fr2.X(t), fr2.Y(v)); } if (k === vals.length - 1) ctx2.lineTo(fr2.X(t + 0.2), fr2.Y(v)); }); ctx2.stroke(); };
  stepLine(U.map(u => u[0]), "--post"); stepLine(U.map(u => u[1]), "--c5");
  legend(ctx2, ["blue: speed v · purple: turn rate ω · dashed: bounds ±1"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  $$("#uniW [data-t]").forEach(bt => bt.setAttribute("aria-pressed", String(bt.dataset.t === uni.target)));
  const costs = Object.values(d.guess_costs), spread = Math.max(...costs) - Math.min(...costs), diff = Math.max(...Object.values(d.guess_maxdiff)), maxV = Math.max(...U.map(u => Math.abs(u[0]))), maxW = Math.max(...U.map(u => Math.abs(u[1])));
  caption($("#uniCap"), `Target pose [[z_f = ${uni.target}]]${uni.target === "(2, 1, 0)" ? ", the notebook's" : ""}: SciPy's SLSQP finds a path with cost ${fmt(d.cost, 4)}, largest speed ${fmt(maxV, 3)} and largest turn rate ${fmt(maxW, 3)}, well inside the bounds of 1; the rate bounds of 0.2 per step shape the smooth ramps. The ticks along the path show the heading. Five initial guesses (the notebook's, straight ahead, turning the other way first, driving in reverse, and random inputs) all converge to the same solution: the costs differ by ${spread.toExponential(1)} and the inputs by at most ${diff.toExponential(1)}. Nonconvexity means multiple local optima are possible, not certain; with this smooth cost and these rate limits the landscape is benign, and that cannot be known before trying several starts (exercise 7).`);
}
$$("#uniW [data-t]").forEach(bt => bt.addEventListener("click", () => { uni.target = bt.dataset.t; drawUni(); }));

function drawAll() { drawHero(); drawSet(); drawQuad(); drawKKT(); drawStruct(); drawUni(); }
drawAll();
onRedraw(drawAll);
})();
