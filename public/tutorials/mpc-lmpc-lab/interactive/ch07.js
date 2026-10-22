/* Advanced MPC and Learning MPC, chapter 07, Data-driven local models for racing: the live demos.
   The laps come from Python (ch07-data.js: the nonlinear bicycle simulator under the noisy PID controller,
   whose random dither cannot be reproduced bit for bit in a browser). Everything else is recomputed here:
   the scaled-l1 neighbour selection per lap, the Epanechnikov weights, the structured weighted ridge
   regression of the dynamic rows, the analytic linearisation of the Frenet kinematics, the held-out
   validation, the horizon rollouts and the exercise sweeps. Needs ../kit/kit.js and ch07-data.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const D = window.CH07_DATA;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
function solve(A0, b0) {
  const n = A0.length, A = A0.map((r, i) => [...r, b0[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; const pivot = A[c];
    for (let r = c + 1; r < n; r++) { const m = A[r][c] / pivot[c]; if (!m) continue; const row = A[r]; for (let k = c; k <= n; k++) row[k] -= m * pivot[k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = A[r][n]; for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k]; x[r] = s / A[r][r]; }
  return x;
}
/* condition number of a small symmetric positive matrix: ratio of extreme eigenvalues by Jacobi rotations */
function cond2(M) { const n = M.length, A = M.map(r => r.slice()); for (let sweep = 0; sweep < 60; sweep++) { let off = 0; for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q]; if (off < 1e-30) break; for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) { if (Math.abs(A[p][q]) < 1e-300) continue; const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c; for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; } for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; } } } const ev = A.map((r, i) => Math.abs(r[i])); return Math.max(...ev) / Math.min(...ev); }
/* track helpers: periodic interpolation on the 400-sample grid */
const TR = D.track, NT = TR.s.length;
function periodic(values, s) { const L = TR.length, q = ((s % L) + L) % L, i = Math.min(NT - 1, Math.floor(q / L * NT)), s0 = TR.s[i], s1 = i + 1 < NT ? TR.s[i + 1] : L, v1 = i + 1 < NT ? values[i + 1] : values[0], f = (q - s0) / (s1 - s0); return values[i] + f * (v1 - values[i]); }
const curvatureAt = s => periodic(TR.K, s);
const COSH = TR.heading.map(Math.cos), SINH = TR.heading.map(Math.sin), XS = TR.xy.map(p => p[0]), YS = TR.xy.map(p => p[1]);
function globalPosition(s, ey) { const x = periodic(XS, s), y = periodic(YS, s), ch = periodic(COSH, s), sh = periodic(SINH, s), nrm = Math.hypot(ch, sh); return [x - ey * sh / nrm, y + ey * ch / nrm]; }
/* exact affine linearisation of the Euler-discretised Frenet kinematics at x (rows e_psi, s, e_y) */
function kinematicLinearization(x, dt = 0.1) {
  const [vx, vy, r, epsi, s, ey] = x, kappa = curvatureAt(s), den = 1 - kappa * ey, c = Math.cos(epsi), sn = Math.sin(epsi), tang = vx * c - vy * sn, dtang = -vx * sn - vy * c, sdot = tang / den;
  const A = [[-dt * kappa * c / den, dt * kappa * sn / den, dt, 1 - dt * kappa * dtang / den, 0, -dt * kappa * kappa * tang / (den * den)], [dt * c / den, -dt * sn / den, 0, dt * dtang / den, 1, dt * kappa * tang / (den * den)], [dt * sn, dt * c, 0, dt * tang, 0, 1]];
  const f = [epsi + dt * (r - kappa * sdot), s + dt * sdot, ey + dt * (vx * sn + vy * c)];
  return { A, c: f.map((v, i) => v - A[i].reduce((p, a, j) => p + a * x[j], 0)), kappa };
}
/* weighted ridge least squares: returns theta (columns of Y) and the condition number of the normal matrix */
function weightedRidge(Phi, Y, w, ridge) { const p = Phi[0].length, m = Y[0].length, N = Phi.map((row, i) => [row.reduce((s, v, j) => s + v * v * 0, 0)]) && []; const normal = Array.from({ length: p }, () => new Array(p).fill(0)), rhs = Array.from({ length: p }, () => new Array(m).fill(0)); for (let k = 0; k < Phi.length; k++) { const ph = Phi[k], wk = w[k]; for (let i = 0; i < p; i++) { const wi = wk * ph[i]; for (let j = 0; j < p; j++) normal[i][j] += wi * ph[j]; for (let l = 0; l < m; l++) rhs[i][l] += wi * Y[k][l]; } } for (let i = 0; i < p; i++) normal[i][i] += ridge; const theta = []; for (let l = 0; l < m; l++) theta.push(solve(normal, rhs.map(r => r[l]))); return { theta, cond: cond2(normal) }; }
/* structured fit of the dynamic rows; opts: offset (affine constant), vxDelta (steering in the vx row), latAcc (acceleration in the lateral rows) */
function structuredFit(X, U, Xn, w, ridge, opts = {}) {
  const offset = opts.offset !== false, one = offset ? [1] : [], vxCols = [1].concat(opts.vxDelta ? [0] : []), latCols = [0].concat(opts.latAcc ? [1] : []);
  const Phiv = X.map((x, k) => [x[0], x[1], x[2], ...vxCols.map(c => U[k][c]), ...one]), fv = weightedRidge(Phiv, Xn.map(x => [x[0]]), w, ridge), tv = fv.theta[0];
  const Phil = X.map((x, k) => [x[0], x[1], x[2], ...latCols.map(c => U[k][c]), ...one]), fl = weightedRidge(Phil, Xn.map(x => [x[1], x[2]]), w, ridge);
  const A = [[tv[0], tv[1], tv[2], 0, 0, 0], [fl.theta[0][0], fl.theta[0][1], fl.theta[0][2], 0, 0, 0], [fl.theta[1][0], fl.theta[1][1], fl.theta[1][2], 0, 0, 0]], B = [[0, 0], [0, 0], [0, 0]], c = [0, 0, 0];
  vxCols.forEach((col, i) => B[0][col] = tv[3 + i]); latCols.forEach((col, i) => { B[1][col] = fl.theta[0][3 + i]; B[2][col] = fl.theta[1][3 + i]; });
  if (offset) { c[0] = tv[3 + vxCols.length]; c[1] = fl.theta[0][3 + latCols.length]; c[2] = fl.theta[1][3 + latCols.length]; }
  return { A, B, c, cond: [fv.cond, fl.cond] };
}
function assemble(dyn, kin) { return { A: [...dyn.A, ...kin.A], B: [...dyn.B, [0, 0], [0, 0], [0, 0]], c: [...dyn.c, ...kin.c] }; }
function makeLaps(list) { return list.map(l => ({ X: l.X.slice(0, -1), U: l.U, Xn: l.X.slice(1) })); }
/* the local model */
function localFit(laps, x, u, p) {
  const sc = p.scaling || [0.1, 1, 1, 1, 1], h = p.bandwidth ?? 5, m = p.maxPoints ?? 7, ridge = p.ridge ?? 1e-8, q = [x[0], x[1], x[2], u[0], u[1]];
  const rowsX = [], rowsU = [], rowsN = [], w = [], sel = [];
  for (const lap of laps) {
    const d = lap.X.map((xi, k) => sc[0] * Math.abs(xi[0] - q[0]) + sc[1] * Math.abs(xi[1] - q[1]) + sc[2] * Math.abs(xi[2] - q[2]) + sc[3] * Math.abs(lap.U[k][0] - q[3]) + sc[4] * Math.abs(lap.U[k][1] - q[4]));
    const order = d.map((v, i) => i).sort((a, b) => d[a] - d[b] || a - b), inside = order.filter(i => d[i] < h);
    let idx, hl, flag; if (inside.length) { idx = inside.slice(0, m); hl = h; flag = false; } else { idx = order.slice(0, m); hl = 1.05 * Math.max(...idx.map(i => d[i])); flag = true; }
    const wl = idx.map(i => 0.75 * (1 - (d[i] / hl) ** 2)); idx.forEach((i, k) => { rowsX.push(lap.X[i]); rowsU.push(lap.U[i]); rowsN.push(lap.Xn[i]); w.push(wl[k]); }); sel.push({ idx, d: idx.map(i => d[i]), w: wl, fallback: flag });
  }
  const dyn = structuredFit(rowsX, rowsU, rowsN, w, ridge, p), kin = kinematicLinearization(x, 0.1), M = assemble(dyn, kin);
  return { ...M, cond: dyn.cond, sel, kappa: kin.kappa, fallback: sel.some(s => s.fallback) };
}
function supportDistance(laps, x, u, sc = [0.1, 1, 1, 1, 1]) { const q = [x[0], x[1], x[2], u[0], u[1]]; let best = Infinity; for (const lap of laps) lap.X.forEach((xi, k) => { const d = sc[0] * Math.abs(xi[0] - q[0]) + sc[1] * Math.abs(xi[1] - q[1]) + sc[2] * Math.abs(xi[2] - q[2]) + sc[3] * Math.abs(lap.U[k][0] - q[3]) + sc[4] * Math.abs(lap.U[k][1] - q[4]); if (d < best) best = d; }); return best; }
function globalDyn(laps, ridge = 1e-8, opts = {}) { const X = laps.flatMap(l => l.X), U = laps.flatMap(l => l.U), Xn = laps.flatMap(l => l.Xn); return structuredFit(X, U, Xn, X.map(() => 1), ridge, opts); }
function globalFit(gdyn, x) { return assemble(gdyn, kinematicLinearization(x, 0.1)); }
const apply = (M, x, u) => M.A.map((row, i) => row.reduce((s, a, j) => s + a * x[j], 0) + M.B[i][0] * u[0] + M.B[i][1] * u[1] + M.c[i]);
function rollout(fitter, x0, U) { const X = [x0], diags = []; for (const u of U) { const M = fitter(X[X.length - 1], u); diags.push(M); X.push(apply(M, X[X.length - 1], u)); } return { X, diags }; }
/* NET-END */

const LAPS = makeLaps(D.laps), VAL = D.val, NV = VAL.U.length;
const rmse = (E, cols) => cols.map(j => Math.sqrt(E.reduce((s, e) => s + e[j] * e[j], 0) / E.length));
function validate(p, gopts) { const g = globalDyn(LAPS, p.ridge ?? 1e-8, gopts || p), El = [], Eg = [], sup = [], cond = [], fb = []; for (let k = 0; k < NV; k++) { const x = VAL.X[k], u = VAL.U[k], M = localFit(LAPS, x, u, p), xl = apply(M, x, u), xg = apply(globalFit(g, x), x, u), t = VAL.X[k + 1]; El.push(xl.map((v, j) => v - t[j])); Eg.push(xg.map((v, j) => v - t[j])); sup.push(supportDistance(LAPS, x, u, p.scaling)); cond.push(Math.max(...M.cond)); fb.push(M.fallback); } return { El, Eg, sup, cond, fb, rl: rmse(El, [0, 1, 2, 3, 4, 5]), rg: rmse(Eg, [0, 1, 2, 3, 4, 5]), g }; }
const median = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const quant = (a, q) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.round(q * (s.length - 1)))]; };
function legend(ctx, lines, x, y, narrow) { ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; const out = narrow ? lines.flatMap(l => l.split(" · ")) : lines; out.forEach((l, i) => ctx.fillText(l, x, y + 13 * i)); }
function mapProj(fr, xr, yr) { const sc = Math.min(fr.iw / (xr[1] - xr[0]), fr.ih / (yr[1] - yr[0])), cx = (xr[0] + xr[1]) / 2, cy = (yr[0] + yr[1]) / 2; return { X: x => fr.pad.l + fr.iw / 2 + (x - cx) * sc, Y: y => fr.pad.t + fr.ih / 2 - (y - cy) * sc, sc }; }

/* =====================================================================
   Hero: the local model at a query of the held-out lap
   ===================================================================== */
const hs = { q: 120, bandwidth: 5, maxPoints: 7, ridge: 1e-8, offset: true, vxDelta: false, latAcc: false, scaleVx: 0.1, val: null };
const params = () => ({ bandwidth: hs.bandwidth, maxPoints: hs.maxPoints, ridge: hs.ridge, offset: hs.offset, vxDelta: hs.vxDelta, latAcc: hs.latAcc, scaling: [hs.scaleVx, 1, 1, 1, 1] });
const NB = () => hs.bandwidth === 5 && hs.maxPoints === 7 && hs.ridge === 1e-8 && hs.offset && !hs.vxDelta && !hs.latAcc && hs.scaleVx === 0.1;
function drawHero() {
  if (!hs.val) hs.val = validate(params());
  const V = hs.val, k = hs.q, x = VAL.X[k], u = VAL.U[k], M = localFit(LAPS, x, u, params()), xl = apply(M, x, u), xg = apply(globalFit(V.g, x), x, u), t = VAL.X[k + 1];
  /* map */
  const cv = $("#mapC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.55, { x: [-4.5, 4.5], y: [-3.2, 3.2], xticks: [], yticks: [] }), { ctx } = fr, P = mapProj(fr, [-4.5, 4.5], [-3.2, 3.2]); cv._P = P;
  const edge = ey => TR.s.map((s, i) => [TR.xy[i][0] - ey * SINH[i], TR.xy[i][1] + ey * COSH[i]]);
  ctx.lineWidth = 1; ctx.strokeStyle = alpha("--ink", 0.4); [edge(-TR.W), edge(TR.W)].forEach(B => { ctx.beginPath(); B.forEach((p, i) => i ? ctx.lineTo(P.X(p[0]), P.Y(p[1])) : ctx.moveTo(P.X(p[0]), P.Y(p[1]))); ctx.closePath(); ctx.stroke(); });
  LAPS.forEach(l => { ctx.strokeStyle = alpha("--post", 0.25); ctx.lineWidth = 1; ctx.beginPath(); l.X.forEach((xi, i) => { const p = globalPosition(xi[4], xi[5]); i ? ctx.lineTo(P.X(p[0]), P.Y(p[1])) : ctx.moveTo(P.X(p[0]), P.Y(p[1])); }); ctx.stroke(); });
  ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.8; ctx.beginPath(); VAL.X.forEach((xi, i) => { const p = globalPosition(xi[4], xi[5]); i ? ctx.lineTo(P.X(p[0]), P.Y(p[1])) : ctx.moveTo(P.X(p[0]), P.Y(p[1])); }); ctx.stroke();
  M.sel.forEach((s, j) => s.idx.forEach(i => { const xi = LAPS[j].X[i], p = globalPosition(xi[4], xi[5]); ctx.fillStyle = css("--post"); ctx.beginPath(); ctx.arc(P.X(p[0]), P.Y(p[1]), 3, 0, 2 * Math.PI); ctx.fill(); }));
  const pq = globalPosition(x[4], x[5]); ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(P.X(pq[0]), P.Y(pq[1]), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--bg"); ctx.lineWidth = 1.5; ctx.stroke();
  legend(ctx, ["faint blue: 4 training laps · orange: held-out lap · black: the query · blue dots: its selected neighbours"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* data cloud */
  const cv2 = $("#cloudC"), fr2 = frame(cv2, narrow ? 1.4 : 2.2, { x: [0.45, 0.85], y: [-0.55, 0.55], xlabel: "v_x [m/s]", ylabel: "steering δ [rad]", xticks: [0.5, 0.6, 0.7, 0.8], yticks: [-0.5, 0, 0.5] }), ctx2 = fr2.ctx, X2 = fr2.X, Y2 = fr2.Y;
  LAPS.forEach(l => l.X.forEach((xi, i) => { ctx2.fillStyle = alpha("--post", 0.22); ctx2.beginPath(); ctx2.arc(X2(xi[0]), Y2(l.U[i][0]), 1.6, 0, 2 * Math.PI); ctx2.fill(); }));
  VAL.X.slice(0, -1).forEach((xi, i) => { ctx2.fillStyle = alpha("--warm", 0.35); ctx2.beginPath(); ctx2.arc(X2(xi[0]), Y2(VAL.U[i][0]), 1.6, 0, 2 * Math.PI); ctx2.fill(); });
  M.sel.forEach((s, j) => s.idx.forEach((i, kk) => { const xi = LAPS[j].X[i]; ctx2.fillStyle = css("--post"); ctx2.beginPath(); ctx2.arc(X2(xi[0]), Y2(LAPS[j].U[i][0]), 2.5 + 3 * s.w[kk] / 0.75, 0, 2 * Math.PI); ctx2.fill(); }));
  ctx2.fillStyle = css("--ink"); ctx2.beginPath(); ctx2.arc(X2(x[0]), Y2(u[0]), 5.5, 0, 2 * Math.PI); ctx2.fill();
  legend(ctx2, ["two of the five neighbour-selection features · dot size = kernel weight"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  /* readouts */
  $("#hQV").textContent = k; $("#hHV").textContent = fmt(hs.bandwidth, 1); $("#hMV").textContent = hs.maxPoints; $("#hRV").textContent = hs.ridge === 0 ? "0" : hs.ridge.toExponential(0);
  const el = Math.hypot(...xl.slice(0, 3).map((v, j) => v - t[j])), eg = Math.hypot(...xg.slice(0, 3).map((v, j) => v - t[j]));
  $("#rErr").textContent = `${el.toExponential(2)} / ${eg.toExponential(2)}`; $("#rVx").textContent = `${V.rl[0].toExponential(2)} / ${V.rg[0].toExponential(2)}`; $("#rVy").textContent = `${V.rl[1].toExponential(2)} / ${V.rg[1].toExponential(2)}`; $("#rR").textContent = `${V.rl[2].toExponential(2)} / ${V.rg[2].toExponential(2)}`;
  const nSel = M.sel.reduce((s, q) => s + q.idx.length, 0), dmax = Math.max(...M.sel.flatMap(s => s.d)), wmin = Math.min(...M.sel.flatMap(s => s.w)), nb = NB() && k === 120;
  caption($("#heroStatus"), `${nb ? "The notebook's query, " : "Query "}sample ${k} of the held-out lap: x = (${x.map(v => fmt(v, 3)).join(", ")}), u = (${fmt(u[0], 3)}, ${fmt(u[1], 3)}), local curvature [[K = ${fmt(M.kappa, 3)}]]. The scaled [[\\ell_1]] distance picks ${nSel} transitions, ${hs.maxPoints} from each training lap, all within ${fmt(dmax, 3)} of the query against a bandwidth of ${fmt(hs.bandwidth, 1)}${dmax < 0.2 * hs.bandwidth ? `, so the kernel weights are nearly uniform (minimum ${fmt(wmin, 3)} of 0.75) and the count, not the bandwidth, does the selecting` : ""}${M.fallback ? "; some lap had nothing inside the bandwidth and fell back to its nearest points" : ""}. The learned rows give [[a_{11} = ${fmt(M.A[0][0], 3)}]], [[b_{12} = ${fmt(M.B[0][1], 3)}]], [[b_{31} = ${fmt(M.B[2][0], 3)}]] and offsets c = (${M.c.slice(0, 3).map(v => fmt(v, 3)).join(", ")}); normal-matrix condition numbers ${M.cond.map(v => v.toExponential(1)).join(" and ")}. One step ahead the local model misses the simulator's [[(v_x, v_y, r)]] by ${el.toExponential(2)}, the global structured model by ${eg.toExponential(2)}. Over the whole held-out lap the one-step RMSE is ${V.rl.slice(0, 3).map(v => v.toExponential(2)).join(", ")} for the local model against ${V.rg.slice(0, 3).map(v => v.toExponential(2)).join(", ")} for the global one${NB() ? ", the notebook's figures: the yaw rate gains a factor " + fmt(V.rg[2] / V.rl[2], 1) : ""}; ${V.fb.filter(Boolean).length} of ${NV} queries needed the fallback; median condition number ${median(V.cond).toExponential(2)}. The kinematic rows are identical in both models, so their errors (${V.rl.slice(3).map(v => v.toExponential(2)).join(", ")}) coincide.`);
}
function heroRecompute() { hs.val = null; drawHero(); drawKernel(); drawMatrices(); drawHorizon(); drawRegimes(); }
$("#hQ").addEventListener("input", () => { hs.q = +$("#hQ").value; drawHero(); drawKernel(); drawMatrices(); });
$("#hH").addEventListener("input", () => { hs.bandwidth = +$("#hH").value; heroRecompute(); });
$("#hM").addEventListener("input", () => { hs.maxPoints = +$("#hM").value; heroRecompute(); });
$("#hR").addEventListener("input", () => { hs.ridge = [0, 1e-8, 1e-4, 1e-2, 1][+$("#hR").value]; heroRecompute(); });
const toggles = { hOff: "offset", hVxD: "vxDelta", hLatA: "latAcc", hScale: "scaleVx" };
Object.entries(toggles).forEach(([id, key]) => $("#" + id).addEventListener("click", () => { if (key === "offset") hs.offset = !hs.offset; else if (key === "scaleVx") hs.scaleVx = hs.scaleVx === 0.1 ? 1 : 0.1; else hs[key] = !hs[key]; $("#" + id).setAttribute("aria-pressed", String(key === "offset" ? !hs.offset : key === "scaleVx" ? hs.scaleVx === 1 : hs[key])); heroRecompute(); }));
$("#hReset").addEventListener("click", () => { Object.assign(hs, { q: 120, bandwidth: 5, maxPoints: 7, ridge: 1e-8, offset: true, vxDelta: false, latAcc: false, scaleVx: 0.1 }); $("#hQ").value = 120; $("#hH").value = 5; $("#hM").value = 7; $("#hR").value = 1; Object.keys(toggles).forEach(id => $("#" + id).setAttribute("aria-pressed", "false")); heroRecompute(); });
$("#mapC").addEventListener("click", ev => { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top, P = cv._P; let best = 0, bd = Infinity; VAL.X.slice(0, -1).forEach((xi, i) => { const p = globalPosition(xi[4], xi[5]), d = Math.hypot(P.X(p[0]) - x, P.Y(p[1]) - y); if (d < bd) { bd = d; best = i; } }); hs.q = best; $("#hQ").value = best; drawHero(); drawKernel(); drawMatrices(); });

/* =====================================================================
   Part 4: the kernel weights of the selected points
   ===================================================================== */
function drawKernel() {
  const k = hs.q, M = localFit(LAPS, VAL.X[k], VAL.U[k], params()), h = hs.bandwidth, cv = $("#kernC"), narrow = narrowOf(cv), dmax = Math.max(h, ...M.sel.flatMap(s => s.d)) * 1.05;
  const fr = frame(cv, narrow ? 1.5 : 2.4, { x: [0, dmax], y: [0, 0.8], xlabel: "scaled distance to the query", ylabel: "kernel weight", yticks: [0, 0.25, 0.5, 0.75] }), { ctx, X, Y } = fr;
  const dd = Array.from({ length: 101 }, (_, i) => dmax * i / 100); plotLine(fr, dd, dd.map(d => Math.max(0, 0.75 * (1 - (d / h) ** 2))), { color: alpha("--ink", 0.5), width: 1.4, dash: [5, 4] });
  M.sel.forEach((s, j) => s.d.forEach((d, i) => { ctx.fillStyle = alpha("--post", 0.35 + 0.65 * j / 3); ctx.beginPath(); ctx.arc(X(d), Y(s.w[i]), 4, 0, 2 * Math.PI); ctx.fill(); }));
  legend(ctx, [`dashed: Epanechnikov kernel, h = ${fmt(h, 1)} · dots: selected points (darker = later lap)`], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const all = M.sel.flatMap(s => s.d);
  caption($("#kernCap"), `The ${all.length} selected transitions lie between ${fmt(Math.min(...all), 3)} and ${fmt(Math.max(...all), 3)} from the query in the scaled metric, ${Math.max(...all) < 0.25 * h ? `far inside the bandwidth, so every weight is within ${fmt(100 * (1 - Math.min(...M.sel.flatMap(s => s.w)) / 0.75), 1)} percent of the maximum 3/4. On this RC-car data the whole training cloud spans only 4.74 in the scaled metric, less than [[h = 5]]: for any query inside the envelope every stored transition is inside the kernel, and the seven-points-per-lap rule does the actual selection` : `and the kernel shape now matters: weights range from ${fmt(Math.min(...M.sel.flatMap(s => s.w)), 3)} to ${fmt(Math.max(...M.sel.flatMap(s => s.w)), 3)}`}. Lower the bandwidth in the demo to below 0.5 to see the kernel discriminate${M.fallback ? "; a lap with nothing inside the bandwidth falls back to its nearest points, with the bandwidth widened to cover them" : ""}.`);
}

/* =====================================================================
   Part 6 and 7: the assembled matrices at the query
   ===================================================================== */
function drawMatrices() {
  const k = hs.q, M = localFit(LAPS, VAL.X[k], VAL.U[k], params()), names = ["v_x", "v_y", "r", "e_ψ", "s", "e_y"], f = v => Math.abs(v) < 1e-12 ? "0" : fmt(v, 4);
  $("#matT").innerHTML = `<thead><tr><th>row</th>${names.map(n => `<th>${n}</th>`).join("")}<th>δ</th><th>a_x</th><th>c</th></tr></thead><tbody>` + M.A.map((row, i) => `<tr><td class="words">${names[i]}⁺ ${i < 3 ? "(learned)" : "(analytic)"}</td>${row.map(v => `<td>${f(v)}</td>`).join("")}<td>${f(M.B[i][0])}</td><td>${f(M.B[i][1])}</td><td>${f(M.c[i])}</td></tr>`).join("") + "</tbody>";
  caption($("#matCap"), `The assembled affine model [[x^+ = A x + B u + c]] at sample ${k}. Rows one to three are regression coefficients, not derivatives of the plant: here [[a_{11} = ${fmt(M.A[0][0], 3)}]] and [[a_{12} = ${fmt(M.A[0][1], 3)}]], values no physical car has, because over the ${M.sel.reduce((s, q) => s + q.idx.length, 0)} selected transitions [[v_y]] barely varies and its coefficient is poorly determined (condition numbers ${M.cond.map(v => v.toExponential(1)).join(", ")}); coefficients close to the true local derivatives fit the same data almost equally well. What is validated is the prediction. Rows four to six are the exact Jacobian of the Euler-discretised Frenet kinematics at the query with [[K = ${fmt(M.kappa, 3)}]] frozen: [[\\partial e_\\psi^+ / \\partial r = T_s = 0.1]], [[\\partial s^+ / \\partial s = 1]], [[\\partial e_y^+ / \\partial e_y = 1]], and the offsets [[c = f(x_q) - A x_q]] make the affine map exact at the query. The zero blocks are the structure: the kinematic rows take no input, the [[v_x]] row takes only [[a_x]], the lateral rows only [[\\delta]]${hs.vxDelta || hs.latAcc ? " (relaxed by your toggles)" : ""}.`);
}

/* =====================================================================
   Part 9: the horizon experiment
   ===================================================================== */
const hz = { start: 145, H: 14 };
function drawHorizon() {
  if (!hs.val) hs.val = validate(params());
  const start = Math.min(hz.start, NV - hz.H), H = hz.H, U = VAL.U.slice(start, start + H), T = VAL.X.slice(start, start + H + 1), g = hs.val.g;
  const loc = rollout((x, u) => localFit(LAPS, x, u, params()), VAL.X[start], U), glo = rollout((x, u) => globalFit(g, x), VAL.X[start], U);
  const en = (X) => X.map((x, i) => Math.hypot(...x.slice(0, 3).map((v, j) => v - T[i][j]))), el = en(loc.X), eg = en(glo.X), rm = X => Math.sqrt(X.reduce((s, x, i) => s + x.slice(0, 3).reduce((p, v, j) => p + (v - T[i][j]) ** 2, 0), 0) / (3 * X.length));
  const cv = $("#horC"), narrow = narrowOf(cv), allv = [...T, ...loc.X, ...glo.X].map(x => x[0]), lo = Math.min(...allv) - 0.03, hi = Math.max(...allv) + 0.03;
  const fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, H], y: [lo, hi], xlabel: "prediction step", ylabel: "v_x [m/s]", xticks: Array.from({ length: Math.floor(H / (H > 20 ? 5 : 2)) + 1 }, (_, i) => i * (H > 20 ? 5 : 2)), yfmt: v => fmt(v, 2) }), { ctx } = fr, st = T.map((_, i) => i);
  plotLine(fr, st, T.map(x => x[0]), { color: css("--ink"), width: 2 }); plotLine(fr, st, loc.X.map(x => x[0]), { color: css("--post"), width: 2, dash: [6, 3] }); plotLine(fr, st, glo.X.map(x => x[0]), { color: css("--warm"), width: 2, dash: [2, 3] });
  legend(ctx, ["black: nonlinear simulator · blue dashed: local LTV prediction · orange dotted: global affine prediction"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const cv2 = $("#horE"), fr2 = frame(cv2, narrow ? 1.5 : 2.6, { x: [0, H], y: [0, Math.max(...el, ...eg) * 1.1], xlabel: "prediction step", ylabel: "error norm", xticks: fr.xticks, yfmt: v => fmt(v, 2) });
  plotLine(fr2, st, el, { color: css("--post"), width: 2 }); plotLine(fr2, st, eg, { color: css("--warm"), width: 2 }); legend(fr2.ctx, ["dynamic-state error ‖(ṽ_x, ṽ_y, r̃)‖ · blue: local · orange: global"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  const Av = loc.diags.map(M => Math.sqrt(M.A.flat().reduce((s, v, i) => s + (v - loc.diags[0].A.flat()[i]) ** 2, 0))), Bv = loc.diags.map(M => Math.sqrt(M.B.flat().reduce((s, v, i) => s + (v - loc.diags[0].B.flat()[i]) ** 2, 0))), sup = loc.X.slice(0, -1).map((x, i) => supportDistance(LAPS, x, U[i], params().scaling));
  $("#hzSV").textContent = start; $("#hzHV").textContent = H; let jump = 1; for (let i = 2; i <= H; i++) if (el[i] - el[i - 1] > el[jump] - el[jump - 1]) jump = i; const nb = start === 145 && H === 14 && NB();
  caption($("#horCap"), `${nb ? "The notebook's experiment: " : ""}from sample ${start} of the held-out lap, both models are fed the ${H} inputs the simulator actually received and re-fitted at every predicted state. Dynamic-state RMSE over the horizon: local ${rm(loc.X).toExponential(3)}, global ${rm(glo.X).toExponential(3)}${rm(loc.X) < rm(glo.X) ? `, a factor ${fmt(rm(glo.X) / rm(loc.X), 1)} in favour of the local model` : ", the global model wins here: one-step accuracy does not promise long-horizon accuracy once errors feed back into the neighbour selection"}. The errors are not monotone: the local error jumps most at step ${jump} (from ${el[jump - 1].toExponential(2)} to ${el[jump].toExponential(2)}), where the true trajectory has a sharp transient, it can shrink again when the trajectories come back together, and the final-step errors are ${el[H].toExponential(2)} against ${eg[H].toExponential(2)}. The model is time varying: [[\\|A_k - A_0\\|_F]] reaches ${fmt(Math.max(...Av), 2)} and [[\\|B_k - B_0\\|_F]] ${fmt(Math.max(...Bv), 2)} along the horizon, and not smoothly, because a small change of the query swaps neighbours and the nearly collinear data then move the coefficients abruptly. Data support along the horizon: the nearest stored transition is never farther than ${fmt(Math.max(...sup), 3)}, far inside the bandwidth ${fmt(hs.bandwidth, 1)}; a predicted point crossing the bandwidth would be an explicit warning that the optimiser plans where the local dynamics are unsupported.`);
}
$("#hzS").addEventListener("input", () => { hz.start = +$("#hzS").value; drawHorizon(); }); $("#hzH").addEventListener("input", () => { hz.H = +$("#hzH").value; drawHorizon(); });

/* =====================================================================
   Part 10: an out-of-distribution query
   ===================================================================== */
function drawOOD() {
  const t = +$("#oodT").value, xq = VAL.X[120], uq = VAL.U[120], xo = [4.0, 1.0, 4.0, 0.2, 5.0, 0.2], uo = [0.5, 5.0], x = xq.map((v, i) => v + t * (xo[i] - v)), u = uq.map((v, i) => v + t * (uo[i] - v));
  const d = supportDistance(LAPS, x, u), M = localFit(LAPS, x, u, { bandwidth: 5, maxPoints: 7, ridge: 1e-8 });
  const cv = $("#oodC"), narrow = narrowOf(cv), ts = Array.from({ length: 41 }, (_, i) => i / 40), ds = ts.map(tt => supportDistance(LAPS, xq.map((v, i) => v + tt * (xo[i] - v)), uq.map((v, i) => v + tt * (uo[i] - v))));
  const fr = frame(cv, narrow ? 1.5 : 2.5, { x: [0, 1], y: [0, 10], xlabel: "position between the notebook's query (0) and the out-of-distribution point (1)", ylabel: "nearest scaled distance", yticks: [0, 5, 10] }), { ctx, X, Y } = fr;
  plotLine(fr, ts, ds, { color: css("--post"), width: 2.2 }); ctx.strokeStyle = alpha("--ink", 0.5); ctx.setLineDash([5, 4]); seg(ctx, X(0), Y(5), X(1), Y(5)); ctx.setLineDash([]); ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(t), Y(d), 5, 0, 2 * Math.PI); ctx.fill();
  legend(ctx, ["dashed: kernel bandwidth h = 5"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#oodTV").textContent = fmt(t, 2);
  caption($("#oodCap"), `At [[t = ${fmt(t, 2)}]] the query is x = (${x.map(v => fmt(v, 2)).join(", ")}), u = (${fmt(u[0], 2)}, ${fmt(u[1], 2)}) and its nearest stored transition is ${fmt(d, 3)} away in the scaled metric${d > 5 ? ", beyond the bandwidth: no lap has a point inside the kernel, every lap falls back to its seven nearest transitions with a widened bandwidth, and the fit is flagged on " + M.sel.filter(s => s.fallback).length + " of 4 laps" : d > 2.5 ? ", still inside the bandwidth but far from any experience" : ", well supported"}. The fit still returns numbers ([[a_{11} = ${fmt(M.A[0][0], 3)}]], [[b_{12} = ${fmt(M.B[0][1], 3)}]]); that proves nothing. The notebook's extreme point, four times the speed and yaw rate ever recorded, sits at distance 8.546: the correct conclusion is that the query is extrapolative and the model's reliability unverified, which a controller should treat as a warning, not as a prediction.`);
}
$("#oodT").addEventListener("input", drawOOD);

/* =====================================================================
   Part 11: the sweeps of exercises 3 to 9, computed live
   ===================================================================== */
let sweepsDone = false;
function drawSweeps() {
  if (sweepsDone) return; sweepsDone = true;
  const rows = [], base = validate({ bandwidth: 5, maxPoints: 7, ridge: 1e-8 });
  const line = (name, V) => rows.push(`<tr><td class="words">${name}</td><td>${V.rl[0].toExponential(2)}</td><td>${V.rl[1].toExponential(2)}</td><td>${V.rl[2].toExponential(2)}</td><td>${V.fb.filter(Boolean).length}</td><td>${median(V.cond).toExponential(1)}</td></tr>`);
  line("notebook setting: h = 5, 7 per lap, λ = 1e−8", base);
  for (const h of [0.5, 1, 2, 10]) line(`bandwidth h = ${h}`, validate({ bandwidth: h, maxPoints: 7, ridge: 1e-8 }));
  for (const m of [3, 5, 15, 30, 80]) line(`${m} points per lap`, validate({ bandwidth: 5, maxPoints: m, ridge: 1e-8 }));
  for (const l of [1e-4, 1e-2, 1]) line(`ridge λ = ${l}`, validate({ bandwidth: 5, maxPoints: 7, ridge: l }));
  line("no affine offset (exercise 6)", validate({ bandwidth: 5, maxPoints: 7, ridge: 1e-8, offset: false }));
  line("steering added to the v_x row (exercise 7)", validate({ bandwidth: 5, maxPoints: 7, ridge: 1e-8, vxDelta: true }));
  line("acceleration added to the lateral rows (exercise 8)", validate({ bandwidth: 5, maxPoints: 7, ridge: 1e-8, latAcc: true }));
  line("v_x unscaled in the metric (exercise 2)", validate({ bandwidth: 5, maxPoints: 7, ridge: 1e-8, scaling: [1, 1, 1, 1, 1] }));
  line("global structured model", { rl: base.rg, fb: [], cond: [1] });
  $("#sweepT").innerHTML = "<thead><tr><th>setting</th><th>RMSE v_x</th><th>RMSE v_y</th><th>RMSE r</th><th>fallbacks</th><th>median cond.</th></tr></thead><tbody>" + rows.join("") + "</tbody>";
  caption($("#sweepCap"), `Held-out one-step RMSE of the three learned states for the exercise variants, all recomputed here. Bandwidth (exercise 3): irrelevant from [[h = 1]] upwards, since the data cloud is narrower than that; only at [[h = 0.5]] does the kernel start to discriminate, and it helps [[v_x]] slightly. Points per lap (exercise 4): five beat seven, and 15, 30 or 80 are progressively worse; more data is not better when the extra points come from different operating conditions, which is the whole argument for locality, while the condition number improves with more points. Ridge (exercise 5): [[10^{-4}]] is harmless, [[10^{-2}]] already biases the fit and [[\\lambda = 1]] quadruples the error; conditioning is bought with bias. Structure: dropping the offset (exercise 6) nearly doubles the yaw-rate error; adding steering to the [[v_x]] row (exercise 7) cuts the [[v_x]] error by a factor ${fmt(base.rl[0] / validate({ bandwidth: 5, maxPoints: 7, ridge: 1e-8, vxDelta: true }).rl[0], 1)}, because this car steers up to 0.5 rad and the neglected [[-F_{y,f}\\sin\\delta / m]] term is not small, so here the extra parameter is justified; adding acceleration to the lateral rows (exercise 8) helps nothing. Removing the 0.1 scale on [[v_x]] (exercise 2) changes the selected neighbours little, because [[v_x]] spans only 0.33 m/s in this data; the scale matters for data with a wider speed range.`);
}

/* =====================================================================
   Exercise 9: where locality helps most
   ===================================================================== */
function drawRegimes() {
  if (!hs.val) hs.val = validate(params());
  const V = hs.val, Kv = VAL.X.slice(0, -1).map(x => Math.abs(curvatureAt(x[4]))), vx = VAL.X.slice(0, -1).map(x => x[0]), mK = median(Kv), mv = median(vx), en = E => E.map(e => Math.hypot(e[0], e[1], e[2]));
  const el = en(V.El), eg = en(V.Eg), rms = (a, mask) => Math.sqrt(a.reduce((s, v, i) => s + (mask[i] ? v * v : 0), 0) / mask.filter(Boolean).length);
  const regimes = [["low curvature", Kv.map(k => k < mK)], ["high curvature", Kv.map(k => k >= mK)], ["low speed", vx.map(v => v < mv)], ["high speed", vx.map(v => v >= mv)]];
  $("#regT").innerHTML = "<thead><tr><th>regime (median split)</th><th>local</th><th>global</th><th>ratio</th></tr></thead><tbody>" + regimes.map(([n, m]) => `<tr><td class="words">${n}</td><td>${rms(el, m).toExponential(2)}</td><td>${rms(eg, m).toExponential(2)}</td><td>${fmt(rms(eg, m) / rms(el, m), 2)}</td></tr>`).join("") + "</tbody>";
  const corr = (a, b) => { const ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length; let sab = 0, saa = 0, sbb = 0; a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; sbb += (b[i] - mb) ** 2; }); return sab / Math.sqrt(saa * sbb); };
  caption($("#regCap"), `Dynamic-state error norm of the one-step predictions on the held-out lap, split at the medians of [[|K|]] (${fmt(mK, 3)}) and [[v_x]] (${fmt(mv, 3)}). Locality helps everywhere, most at high speed and low curvature, least at low speed; the global model averages incompatible local behaviour and is worst where the car is fastest. Error against data support: the correlation between the nearest-data distance and the error norm is ${fmt(corr(V.sup, el), 2)}, positive but weak, as the notebook says: distance is a geometric proxy, and every held-out query has a stored transition within ${fmt(Math.max(...V.sup), 2)} of it, a small fraction of the bandwidth, so this lap tests interpolation, not extrapolation.`);
}

function drawAll() { drawHero(); drawKernel(); drawMatrices(); drawHorizon(); drawOOD(); drawRegimes(); drawSweeps(); }
drawAll();
onRedraw(() => { drawHero(); drawKernel(); drawHorizon(); drawOOD(); });
})();
