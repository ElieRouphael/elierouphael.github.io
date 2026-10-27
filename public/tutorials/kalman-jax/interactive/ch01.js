/* Kalman filters with JAX, chapter 01, The filter and its nonlinear cousins: the live demos. The
   linear Kalman filter (Joseph-form update), the extended filter and the unscented filter of the
   repository are implemented here for the two-state position-velocity model, with an optional
   emulation of 32-bit arithmetic that reproduces the UKF's failure under JAX's default precision.
   The repository's own rollout and JAX's results come from ch01-data.js. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
/* Small dense-matrix helpers (arrays of rows). */
const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const mv = (A, x) => A.map(r => r.reduce((s, v, k) => s + v * x[k], 0));
const tr = A => A[0].map((_, j) => A.map(r => r[j]));
const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
const sub = (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j]));
const eye = n => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
const scale = (A, s) => A.map(r => r.map(v => v * s));
function solve(A0, B0) {   // A X = B for a matrix right-hand side, Gaussian elimination with pivoting
  const n = A0.length, m = B0[0].length, A = A0.map((r, i) => [...r, ...B0[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < n; r++) { if (r === c) continue; const f = A[r][c] / A[c][c]; if (!f) continue; for (let k = c; k < n + m; k++) A[r][k] -= f * A[c][k]; }
  }
  return A.map((r, i) => r.slice(n).map(v => v / r[i]));
}
/* Model of the repository: constant-velocity motion sampled at dt, position measured. */
function model(dt = 0.1, processStd = 0.1, measStd = 0.5) {
  return { F: [[1, dt], [0, 1]], H: [[1, 0]], Q: scale([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]], processStd ** 2), R: [[measStd ** 2]] };
}
/* Linear predict and Joseph-form update, as kalman_jax.filters. */
function kfPredict(x, P, F, Q) { return { x: mv(F, x), P: add(mm(mm(F, P), tr(F)), Q) }; }
function kfUpdate(x, P, z, H, R) {
  const S = add(mm(mm(H, P), tr(H)), R);                     // innovation covariance
  const K = tr(solve(S, mm(H, tr(P))));                      // P H^T S^-1 without an explicit inverse
  const innov = z.map((v, i) => v - mv(H, x)[i]);
  const xn = x.map((v, i) => v + mv(K, innov)[i]);
  const IKH = sub(eye(x.length), mm(K, H));
  const Pn = add(mm(mm(IKH, P), tr(IKH)), mm(mm(K, R), tr(K)));
  return { x: xn, P: Pn, innov, S, K };
}
function runKF(zs, { F, H, Q, R }, x0, P0) {
  let x = x0, P = P0; const out = [], gains = [], Ps = [];
  for (const z of zs) { const p = kfPredict(x, P, F, Q); const u = kfUpdate(p.x, p.P, [z], H, R); x = u.x; P = u.P; out.push(x); gains.push(u.K.map(r => r[0])); Ps.push(P); }
  return { xs: out, gains, Ps };
}
/* Extended filter with user-supplied f, h and their Jacobians; on a linear model it is the linear filter. */
function runEKF(zs, f, Fj, h, Hj, Q, R, x0, P0) {
  let x = x0, P = P0; const out = [];
  for (const z of zs) { const Fx = Fj(x); const xp = f(x), Pp = add(mm(mm(Fx, P), tr(Fx)), Q); const Hx = Hj(xp); const S = add(mm(mm(Hx, Pp), tr(Hx)), R); const K = tr(solve(S, mm(Hx, tr(Pp)))); const innov = [z - h(xp)[0]]; x = xp.map((v, i) => v + mv(K, innov)[i]); const IKH = sub(eye(2), mm(K, Hx)); P = add(mm(mm(IKH, Pp), tr(IKH)), mm(mm(K, R), tr(K))); out.push(x); }
  return out;
}
/* Unscented filter with the repository's sigma-point construction. `rnd` rounds every intermediate
   result (identity for double precision, Math.fround to emulate JAX's default float32). */
function ukfWeights(n, alpha, beta, kappa, rnd) {
  const lambda = rnd(rnd(alpha * alpha) * (n + kappa) - n), sc = rnd(n + lambda);
  const Wm = new Array(2 * n + 1).fill(rnd(1 / (2 * sc))), Wc = Wm.slice();
  Wm[0] = rnd(lambda / sc); Wc[0] = rnd(rnd(lambda / sc) + rnd(1 - rnd(alpha * alpha) + beta));
  return { sc, Wm, Wc };
}
function chol2(P, rnd) {   // lower Cholesky factor of a 2x2 matrix, rounded like the reference
  const l11 = rnd(Math.sqrt(P[0][0])), l21 = rnd(P[1][0] / l11), l22 = rnd(Math.sqrt(rnd(P[1][1] - rnd(l21 * l21))));
  return [[l11, 0], [l21, l22]];
}
function sigmaPoints(x, P, alpha, beta, kappa, rnd, jitter = 1e-9) {
  const n = x.length, { sc, Wm, Wc } = ukfWeights(n, alpha, beta, kappa, rnd);
  const L = chol2(P.map((r, i) => r.map((v, j) => rnd(v + (i === j ? jitter : 0)))), rnd);
  const s = rnd(Math.sqrt(sc)), cols = [0, 1].map(j => L.map(r => rnd(s * r[j])));   // columns of the factor, as the repository fixed it
  const pts = [x.slice()];
  for (let j = 0; j < n; j++) pts.push(x.map((v, i) => rnd(v + cols[j][i])));
  for (let j = 0; j < n; j++) pts.push(x.map((v, i) => rnd(v - cols[j][i])));
  return { pts, Wm, Wc };
}
function ukfStep(x, P, z, f, h, Q, R, alpha, beta, kappa, rnd) {
  const { pts, Wm, Wc } = sigmaPoints(x, P, alpha, beta, kappa, rnd);
  const fx = pts.map(p => f(p).map(rnd));
  const xp = [0, 1].map(i => fx.reduce((s, p, k) => rnd(s + rnd(Wm[k] * p[i])), 0));
  let Pp = [[0, 0], [0, 0]];
  fx.forEach((p, k) => { const d = p.map((v, i) => rnd(v - xp[i])); for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) Pp[i][j] = rnd(Pp[i][j] + rnd(Wc[k] * rnd(d[i] * d[j]))); });
  Pp = Pp.map((r, i) => r.map((v, j) => rnd(v + Q[i][j])));
  const sp = sigmaPoints(xp, Pp, alpha, beta, kappa, rnd);
  const hz = sp.pts.map(p => rnd(h(p)[0]));
  const zp = hz.reduce((s, v, k) => rnd(s + rnd(sp.Wm[k] * v)), 0);
  let Pzz = 0; const Pxz = [0, 0];
  hz.forEach((v, k) => { const dz = rnd(v - zp); Pzz = rnd(Pzz + rnd(sp.Wc[k] * rnd(dz * dz))); const dx = sp.pts[k].map((q, i) => rnd(q - xp[i])); for (let i = 0; i < 2; i++) Pxz[i] = rnd(Pxz[i] + rnd(sp.Wc[k] * rnd(dx[i] * dz))); });
  Pzz = rnd(Pzz + R[0][0]);
  const K = Pxz.map(v => rnd(v / Pzz)), innov = rnd(z - zp);
  const xn = xp.map((v, i) => rnd(v + rnd(K[i] * innov)));
  let Pn = Pp.map((r, i) => r.map((v, j) => rnd(v - rnd(rnd(K[i] * Pzz) * K[j]))));
  Pn = [[Pn[0][0], rnd(0.5 * rnd(Pn[0][1] + Pn[1][0]))], [rnd(0.5 * rnd(Pn[0][1] + Pn[1][0])), Pn[1][1]]];
  return { x: xn, P: Pn };
}
function runUKF(zs, f, h, Q, R, x0, P0, { alpha = 1e-3, beta = 2, kappa = 0, float32 = false } = {}) {
  const rnd = float32 ? Math.fround : (v => v);
  let x = x0.map(rnd), P = P0.map(r => r.map(rnd)); const out = []; let nanFrom = -1;
  zs.forEach((z, t) => { const s = ukfStep(x, P, rnd(z), f, h, Q, R, alpha, beta, kappa, rnd); x = s.x; P = s.P; if (nanFrom < 0 && x.some(v => !isFinite(v))) nanFrom = t; out.push(x); });
  return { xs: out, nanFrom };
}
const rmse2 = (A, B) => [0, 1].map(i => Math.sqrt(A.reduce((s, a, t) => s + (a[i] - B[t][i]) ** 2, 0) / A.length));
/* Steady-state gain and covariance by iterating the Riccati recursion. */
function steadyState({ F, H, Q, R }, iters = 500) {
  let P = eye(2), K = null, S = null;
  for (let i = 0; i < iters; i++) { const Pp = add(mm(mm(F, P), tr(F)), Q); S = add(mm(mm(H, Pp), tr(H)), R); K = tr(solve(S, mm(H, tr(Pp)))); const IKH = sub(eye(2), mm(K, H)); P = add(mm(mm(IKH, Pp), tr(IKH)), mm(mm(K, R), tr(K))); }
  return { K: K.map(r => r[0]), P, S: S[0][0] };
}
function makeRng(seed) {
  let s0 = (seed >>> 0) || 1, s1 = 0x9e3779b9 ^ s0;
  const next = () => { let x = s0, y = s1; s0 = y; x ^= x << 23; x ^= x >>> 17; x ^= y ^ (y >>> 26); s1 = x; return ((s0 + s1) >>> 0) / 4294967296; };
  let spare = null;
  const normal = () => { if (spare !== null) { const v = spare; spare = null; return v; } let u1 = 0; while (u1 <= 1e-12) u1 = next(); const u2 = next(), r = Math.sqrt(-2 * Math.log(u1)); spare = r * Math.sin(2 * Math.PI * u2); return r * Math.cos(2 * Math.PI * u2); };
  return { next, normal };
}
/* The repository's generator with the page's random numbers: process noise with covariance Q, position noise. */
function generate(n, mdl, rng, x0 = [0, 1]) {
  const { F, Q, R } = mdl, L = chol2(Q, v => v), xs = [], zs = []; let x = x0;
  for (let t = 0; t < n; t++) { const e = [rng.normal(), rng.normal()], w = mv(L, e); x = mv(F, x).map((v, i) => v + w[i]); xs.push(x); zs.push(x[0] + Math.sqrt(R[0][0]) * rng.normal()); }
  return { xs, zs };
}
/* NET-END */

if (typeof module !== "undefined" && module.exports) {
  module.exports = { model, runKF, runEKF, runUKF, rmse2, steadyState, makeRng, generate, ukfWeights };
  return;
}

const D = window.KJ1, MDL = { F: D.F, H: D.H, Q: D.Q, R: D.R };
const x0 = [0, 0], P0 = eye(2);
const f = x => mv(MDL.F, x), h = x => mv(MDL.H, x);

/* ============================================================ hero */
const hero = { source: "repo", procStd: 0.1, measStd: 0.5, seed: 1, show: { kf: true, ekf: false, ukf: true }, range: [0, 200] };
let H0 = null;
function runHero() {
  const mdl = hero.source === "repo" ? MDL : model(0.1, hero.procStd, hero.measStd);
  const data = hero.source === "repo" ? { xs: D.xs, zs: D.zs } : generate(1000, mdl, makeRng(hero.seed * 101 + 7));
  const kf = runKF(data.zs, mdl, x0, P0);
  const ekf = runEKF(data.zs, x => mv(mdl.F, x), () => mdl.F, x => mv(mdl.H, x), () => mdl.H, mdl.Q, mdl.R, x0, P0);
  const ukf = runUKF(data.zs, x => mv(mdl.F, x), x => mv(mdl.H, x), mdl.Q, mdl.R, x0, P0, { alpha: 1.0 });
  H0 = { mdl, data, kf, ekf, ukf, r: { kf: rmse2(data.xs, kf.xs), ekf: rmse2(data.xs, ekf), ukf: rmse2(data.xs, ukf.xs) }, measRmse: Math.sqrt(data.zs.reduce((s, z, t) => s + (z - data.xs[t][0]) ** 2, 0) / data.zs.length), ss: steadyState(mdl) };
}
function drawHero() {
  if (!H0) runHero();
  const { data, kf, ekf, ukf, r } = H0, [a, b] = hero.range, t = Array.from({ length: b - a }, (_, i) => (a + i) * 0.1);
  const sl = arr => arr.slice(a, b);
  const cv = $("#heroP"), narrow = narrowOf(cv, 600);
  const pos = sl(data.xs).map(x => x[0]), lo = Math.min(...pos, ...sl(data.zs)) - 0.5, hi = Math.max(...pos, ...sl(data.zs)) + 0.5;
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [t[0], t[t.length - 1]], y: [lo, hi], xlabel: "time [s]", ylabel: "position" });
  fr.ctx.fillStyle = alpha("--muted", 0.55); sl(data.zs).forEach((z, i) => { fr.ctx.beginPath(); fr.ctx.arc(fr.X(t[i]), fr.Y(z), 1.6, 0, 2 * Math.PI); fr.ctx.fill(); });
  plotLine(fr, t, pos, { color: css("--ink"), width: 2 });
  if (hero.show.kf) plotLine(fr, t, sl(kf.xs).map(x => x[0]), { color: css("--post"), width: 1.6 });
  if (hero.show.ekf) plotLine(fr, t, sl(ekf).map(x => x[0]), { color: css("--c4"), width: 1.4, dash: [6, 4] });
  if (hero.show.ukf) plotLine(fr, t, sl(ukf.xs).map(x => x[0]), { color: css("--warm"), width: 1.4, dash: [2, 3] });
  const cvv = $("#heroV"), vel = sl(data.xs).map(x => x[1]);
  const frv = frame(cvv, narrow ? 2.4 : 4.5, { x: [t[0], t[t.length - 1]], y: [Math.min(...vel) - 0.4, Math.max(...vel) + 0.4], xlabel: "time [s]", ylabel: "velocity (not measured)" });
  plotLine(frv, t, vel, { color: css("--ink"), width: 2 });
  if (hero.show.kf) plotLine(frv, t, sl(kf.xs).map(x => x[1]), { color: css("--post"), width: 1.6 });
  if (hero.show.ekf) plotLine(frv, t, sl(ekf).map(x => x[1]), { color: css("--c4"), width: 1.4, dash: [6, 4] });
  if (hero.show.ukf) plotLine(frv, t, sl(ukf.xs).map(x => x[1]), { color: css("--warm"), width: 1.4, dash: [2, 3] });
  $("#rKF").textContent = `${fmt(r.kf[0], 4)} / ${fmt(r.kf[1], 4)}`; $("#rEKF").textContent = `${fmt(r.ekf[0], 4)} / ${fmt(r.ekf[1], 4)}`; $("#rUKF").textContent = `${fmt(r.ukf[0], 4)} / ${fmt(r.ukf[1], 4)}`;
  $("#rMeas").textContent = fmt(H0.measRmse, 4); $("#rGain").textContent = `(${fmt(H0.ss.K[0], 4)}, ${fmt(H0.ss.K[1], 4)})`;
  const isRepo = hero.source === "repo";
  caption($("#heroStatus"), `${isRepo ? "The repository's 1000-step rollout (JAX random key 42)" : `A fresh 1000-step rollout with process noise ${fmt(hero.procStd, 2)} and measurement noise ${fmt(hero.measStd, 2)}`}, samples ${a} to ${b} shown: true position (black) and velocity, the noisy position measurements (dots), and the estimates of the linear filter (blue), the extended filter (green, dashed) and the unscented filter with [[\\alpha = 1]] (orange, dotted). Position RMSE: measurements alone ${fmt(H0.measRmse, 4)}, linear filter ${fmt(r.kf[0], 4)}, EKF ${fmt(r.ekf[0], 4)}, UKF ${fmt(r.ukf[0], 4)}; velocity RMSE ${fmt(r.kf[1], 4)}, ${fmt(r.ekf[1], 4)}, ${fmt(r.ukf[1], 4)}${isRepo ? ", the script's printed 0.1554 and 0.1351 (JAX runs in 32-bit, the page in 64-bit; they agree to four decimals)" : ""}. On a linear model the three filters are the same filter: the EKF's Jacobians are the model matrices, and the unscented transform of a linear map is exact. The filter divides the measurement error by ${fmt(H0.measRmse / r.kf[0], 1)} and recovers a velocity it never measures; its steady-state gain is [[(${fmt(H0.ss.K[0], 3)}, ${fmt(H0.ss.K[1], 3)})]]: each new position moves the position estimate by a tenth of the innovation and the velocity by six hundredths.`);
}
$$("#heroF .btn").forEach(b => b.addEventListener("click", () => { hero.show[b.dataset.f] = !hero.show[b.dataset.f]; b.setAttribute("aria-pressed", hero.show[b.dataset.f]); drawHero(); }));
$("#hProc").addEventListener("input", e => { hero.procStd = +e.target.value; $("#hProcV").textContent = fmt(hero.procStd, 2); hero.source = "fresh"; sync(); runHero(); drawHero(); });
$("#hMeas").addEventListener("input", e => { hero.measStd = +e.target.value; $("#hMeasV").textContent = fmt(hero.measStd, 2); hero.source = "fresh"; sync(); runHero(); drawHero(); });
$("#hNew").addEventListener("click", () => { hero.seed++; hero.source = "fresh"; sync(); runHero(); drawHero(); });
$("#hRepo").addEventListener("click", () => { hero.source = "repo"; sync(); runHero(); drawHero(); });
$("#hWin").addEventListener("input", e => { const s = +e.target.value; hero.range = [s, s + 200]; $("#hWinV").textContent = `${s / 10} to ${(s + 200) / 10} s`; drawHero(); });
function sync() { $("#hRepo").setAttribute("aria-pressed", hero.source === "repo"); }
$("#hProcV").textContent = fmt(hero.procStd, 2); $("#hMeasV").textContent = fmt(hero.measStd, 2); $("#hWinV").textContent = "0 to 20 s";

/* ============================================================ the UKF in 32-bit arithmetic */
const uk = { alpha: 1e-3, kappa: 0 };
function drawUKF() {
  const r32 = runUKF(D.zs, f, h, MDL.Q, MDL.R, x0, P0, { alpha: uk.alpha, kappa: uk.kappa, float32: true });
  const r64 = runUKF(D.zs, f, h, MDL.Q, MDL.R, x0, P0, { alpha: uk.alpha, kappa: uk.kappa, float32: false });
  const kf = runKF(D.zs, MDL, x0, P0).xs;
  const w = ukfWeights(2, uk.alpha, 2, uk.kappa, v => v);
  const cv = $("#ukfC"), narrow = narrowOf(cv);
  const n = 400, t = Array.from({ length: n }, (_, i) => i * 0.1);
  const d32 = r32.xs.slice(0, n).map((x, i) => isFinite(x[0]) ? Math.abs(x[0] - kf[i][0]) : NaN), d64 = r64.xs.slice(0, n).map((x, i) => Math.abs(x[0] - kf[i][0]));
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, 40], y: [-16, 1], xlabel: "time [s]", ylabel: "log10 |UKF − linear KF| (position)", yticks: [-15, -10, -5, 0] });
  plotLine(fr, t, d64.map(v => Math.log10(Math.max(v, 1e-16))), { color: css("--post"), width: 1.6 });
  plotLine(fr, t.filter((_, i) => isFinite(d32[i])), d32.filter(isFinite).map(v => Math.log10(Math.max(v, 1e-16))), { color: css("--warm"), width: 1.8 });
  if (r32.nanFrom >= 0 && r32.nanFrom < n) { fr.ctx.strokeStyle = css("--warm"); fr.ctx.setLineDash([4, 3]); seg(fr.ctx, fr.X(r32.nanFrom * 0.1), fr.pad.t, fr.X(r32.nanFrom * 0.1), fr.Y(-16)); fr.ctx.setLineDash([]); }
  const e32 = r32.nanFrom < 0 ? rmse2(D.xs, r32.xs) : null;
  caption($("#ukfCap"), `The unscented filter on the repository's data with [[\\alpha = ${uk.alpha}]], [[\\kappa = ${uk.kappa}]], [[\\beta = 2]]: distance from the linear filter's position estimate in double precision (blue) and with every intermediate result rounded to 32-bit (orange, how JAX runs by default). The sigma-point weights are [[W_0 = ${fmt(w.Wm[0], 3)}]] on the centre and [[W_i = ${fmt(w.Wm[1], 3)}]] on each of the four spread points, with the spread scaled by [[\\sqrt{n + \\lambda} = ${w.sc > 0 ? fmt(Math.sqrt(w.sc), 4) : "undefined"}]]. ` + (uk.alpha <= 0.01 ? `With [[\\alpha = ${uk.alpha}]] the centre weight is ${fmt(w.Wm[0], 0)} and the spread points sit [[${fmt(Math.sqrt(w.sc), 4)}]] standard deviations from the mean: the covariance is reconstructed as a difference of numbers a million times larger than itself, which 32-bit arithmetic cannot represent. ${r32.nanFrom >= 0 ? `The emulated filter loses the covariance's positive-definiteness and turns to NaN at step ${r32.nanFrom} (JAX's run fails at step ${D.py.ukfNanFrom}; the exact step depends on the order of operations). ` : ""}In double precision the same weights work, to ${fmt(Math.max(...d64) * 1e7, 1)} parts in ten million of the linear filter. ` : `With [[\\alpha = ${uk.alpha}]] the weights are of order one, the spread points sit about one standard deviation out, and both precisions track the linear filter${e32 ? `: RMSE ${fmt(e32[0], 4)} and ${fmt(e32[1], 4)} in 32-bit against the linear filter's 0.1554 and 0.1351` : ""}. `) + `The repository's default [[\\alpha = 10^{-3}]] is the textbook value, written for double precision; the fix proposed for the repository is [[\\alpha = 1]], and the page's hero demo uses it.`);
}
$("#ukA").addEventListener("input", e => { uk.alpha = +(10 ** +e.target.value).toPrecision(2); $("#ukAV").textContent = uk.alpha; drawUKF(); });
$("#ukAV").textContent = uk.alpha;

/* ============================================================ gain and covariance convergence */
function drawGain() {
  const kf = runKF(D.zs, MDL, x0, P0), ss = steadyState(MDL), n = 60, t = Array.from({ length: n }, (_, i) => i * 0.1);
  const cv = $("#gainC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, 6], y: [0, 1.02], xlabel: "time [s]", ylabel: "Kalman gain and posterior sd", xticks: [0, 1, 2, 3, 4, 5, 6], yticks: [0, 0.25, 0.5, 0.75, 1] });
  plotLine(fr, t, kf.gains.slice(0, n).map(k => k[0]), { color: css("--post"), width: 2 });
  plotLine(fr, t, kf.gains.slice(0, n).map(k => k[1]), { color: css("--post"), width: 1.4, dash: [6, 4] });
  plotLine(fr, t, kf.Ps.slice(0, n).map(P => Math.sqrt(P[0][0])), { color: css("--warm"), width: 2 });
  plotLine(fr, t, kf.Ps.slice(0, n).map(P => Math.sqrt(P[1][1])), { color: css("--warm"), width: 1.4, dash: [6, 4] });
  const conv = kf.gains.findIndex((k, i) => i > 0 && Math.abs(k[0] - ss.K[0]) < 1e-3 && Math.abs(k[1] - ss.K[1]) < 1e-3);
  caption($("#gainCap"), `The first six seconds of the repository's run, started from [[P_0 = I]]: the Kalman gain on the position (blue, solid) and on the velocity (blue, dashed), and the posterior standard deviations of position (orange, solid) and velocity (orange, dashed). Neither depends on the measurements, only on [[F, H, Q, R]] and [[P_0]]: the gain starts near one, because the prior is ignorant, and settles within [[10^{-3}]] of its steady-state value [[(${fmt(ss.K[0], 4)}, ${fmt(ss.K[1], 4)})]] after ${conv} steps; the posterior standard deviations settle at ${fmt(Math.sqrt(ss.P[0][0]), 4)} and ${fmt(Math.sqrt(ss.P[1][1]), 4)}, against a measurement noise of 0.5. The steady state is the fixed point of the Riccati recursion, computed on the page by iterating it; the achieved position RMSE of ${fmt(D.py.rmse.linear[0], 4)} agrees with the predicted ${fmt(Math.sqrt(ss.P[0][0]), 4)} because the model is the one that generated the data.`);
}

function drawAll() { drawHero(); drawUKF(); drawGain(); }
drawAll();
onRedraw(drawAll);
})();
