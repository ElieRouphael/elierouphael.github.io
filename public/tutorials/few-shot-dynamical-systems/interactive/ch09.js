/* Learning to Identify New Dynamical Systems, chapter 09, Summary: the capstone demo. The whole
   pipeline of the course runs on one new system in the browser: a designed 6 s experiment
   (chapter 06), the two-number context with its Laplace posterior (chapter 04), prediction of a
   fresh input against the average system and the true coefficients (chapter 05), the misfit test
   against a threshold calibrated live on 60 in-family systems (chapter 07), and feedforward
   tracking (chapter 08). The system can be pushed out of the family (chapter 07). The pretrained
   family, the designed input, the reference and the shift direction come from ch09-data.js.
   Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the smoke test) */
const DT = 0.1, X0 = [0.2, 0.0];
const SIGMA_Y = 0.01, SIGMA_W = 0.01, RHO = 0.8;
const LOW = [0.05, 0.01, 0.05, 0.0], HIGH = [5.0, 2.0, 3.0, 5.0];
const ETA0 = [1.20, 0.32, 0.90, 1.00], M = [[0.55, 0.12], [0.07, 0.11], [0.16, -0.09], [0.30, 0.45]], EPS_SD = [0.015, 0.004, 0.008, 0.04];
const U_MAX = 2.0;
const actuator = u => 0.3 * Math.tanh(u / 0.6);
const dActuator = u => { const c = Math.cosh(u / 0.6); return 0.5 / (c * c); };
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
function simulate(sys, u, noise) {
  const [k, d, b, a] = sys.eta;
  let q = X0[0], v = X0[1], w = 0;
  const qs = [q];
  for (let t = 0; t < u.length; t++) {
    const ut = t === 0 ? 0 : u[t - 1];
    if (noise) w = RHO * w + SIGMA_W * noise();
    v += DT * (-k * q - d * v + b * actuator(ut) - a * q * q * q - sys.gamma * q ** 5 + sys.bias + w);
    q += DT * v;
    qs.push(q);
  }
  return { q: qs, y: noise ? qs.map(x => x + SIGMA_Y * noise()) : qs.slice() };
}
function predictJac(eta, u) {
  const [k, d, b, a] = eta, T = u.length;
  let q = X0[0], v = X0[1], dq = [0, 0, 0, 0], dv = [0, 0, 0, 0];
  const out = new Float64Array(T + 1), J = Array.from({ length: T + 1 }, () => [0, 0, 0, 0]);
  out[0] = q;
  for (let t = 0; t < T; t++) {
    const f = actuator(t === 0 ? 0 : u[t - 1]), q2 = q * q, s = -k - 3 * a * q2, extra = [-q, -v, f, -q2 * q];
    const ndv = dv.map((dvi, i) => dvi + DT * (s * dq[i] - d * dvi + extra[i]));
    v += DT * (-k * q - d * v + b * f - a * q2 * q); q += DT * v;
    dv = ndv; dq = dq.map((dqi, i) => dqi + DT * dv[i]);
    out[t + 1] = q; J[t + 1] = dq.slice();
  }
  return { q: out, J };
}
const predict = (eta, u) => Array.from(predictJac(eta, u).q);
const rmse = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);
function solve(A0, b0) {
  const n = A0.length, A = A0.map((r, i) => [...r, b0[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    if (Math.abs(A[c][c]) < 1e-300) return null;
    const piv = A[c];
    for (let r = 0; r < n; r++) { if (r === c) continue; const m = A[r][c] / piv[c]; if (!m) continue; const row = A[r]; for (let k = c; k <= n; k++) row[k] -= m * piv[k]; }
  }
  return A.map((r, i) => r[n] / r[i]);
}
const inv2 = A => { const d = A[0][0] * A[1][1] - A[0][1] * A[1][0]; return [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]]; };
const decode = (fam, z) => fam.mean.map((m, j) => clip(m + fam.U[j][0] * z[0] + fam.U[j][1] * z[1], LOW[j], HIGH[j]));
function sensitivity(fam, z, u) {
  const raw = fam.mean.map((m, j) => m + fam.U[j][0] * z[0] + fam.U[j][1] * z[1]), live = raw.map((v, j) => v > LOW[j] && v < HIGH[j] ? 1 : 0);
  const { q, J } = predictJac(decode(fam, z), u);
  return { q, S: J.map(row => [0, 1].map(k => row.reduce((s, v, j) => s + live[j] * v * fam.U[j][k], 0))) };
}
function adapt(fam, u, y, t = 1) {
  const sig = fam.sigma, s2 = sig * sig;
  let z = [0, 0];
  const cost = zz => { const q = predictJac(decode(fam, zz), u).q; let s = 0; for (let i = 0; i < y.length; i++) s += (q[i] - y[i]) ** 2; return s / s2 + t * t * (zz[0] ** 2 + zz[1] ** 2); };
  let f = cost(z), lam = 1e-3;
  for (let it = 0; it < 2000; it++) {
    const { q, S } = sensitivity(fam, z, u);
    const H = [[t * t, 0], [0, t * t]], g = [t * t * z[0], t * t * z[1]];
    for (let i = 0; i < y.length; i++) { const r = (q[i] - y[i]) / s2; g[0] += S[i][0] * r; g[1] += S[i][1] * r; H[0][0] += S[i][0] * S[i][0] / s2; H[0][1] += S[i][0] * S[i][1] / s2; H[1][1] += S[i][1] * S[i][1] / s2; }
    H[1][0] = H[0][1];
    if (Math.hypot(g[0], g[1]) < 1e-12) break;
    let accepted = false;
    for (let tries = 0; tries < 40 && !accepted; tries++) {
      const A = [[H[0][0] * (1 + lam), H[0][1]], [H[1][0], H[1][1] * (1 + lam)]];
      const step = solve(A, [-g[0], -g[1]]); if (!step) { lam *= 10; continue; }
      const zn = [z[0] + step[0], z[1] + step[1]], fn = cost(zn);
      if (fn < f) { const rel = (f - fn) / Math.max(f, 1e-300); z = zn; f = fn; lam = Math.max(lam / 10, 1e-15); accepted = true; if (rel < 1e-14 || Math.hypot(step[0], step[1]) < 1e-10) break; }
      else lam *= 10;
    }
    if (!accepted || lam > 1e8) break;
  }
  const { q, S } = sensitivity(fam, z, u), prec = [[t * t, 0], [0, t * t]];
  for (const r of S) { prec[0][0] += r[0] * r[0] / s2; prec[0][1] += r[0] * r[1] / s2; prec[1][1] += r[1] * r[1] / s2; } prec[1][0] = prec[0][1];
  return { z, cov: inv2(prec), misfit: rmse(Array.from(q), y) };
}
function project(fam, eta) {
  const A = fam.U.map((row, j) => row.map(u => u / fam.scale[j])), b = eta.map((v, j) => (v - fam.mean[j]) / fam.scale[j]);
  const N = [[0, 0], [0, 0]], r = [0, 0];
  A.forEach((row, i) => { for (let p = 0; p < 2; p++) { r[p] += row[p] * b[i]; for (let q = 0; q < 2; q++) N[p][q] += row[p] * row[q]; } });
  return solve(N, r);
}
function predictInputJac(eta, u) {
  const [k, d, b, a] = eta, T = u.length, q = predict(eta, u);
  const J = Array.from({ length: T + 1 }, () => new Float64Array(T));
  for (let s = 0; s < T - 1; s++) {
    let dv = DT * b * dActuator(u[s]), dq = DT * dv;
    J[s + 2][s] = dq;
    for (let t = s + 2; t < T; t++) { dv += DT * (-(k + 3 * a * q[t] * q[t]) * dq - d * dv); dq += DT * dv; J[t + 1][s] = dq; }
  }
  return { q, J };
}
function feedforward(eta, ref, lam = 0.05) {
  const T = ref.length - 1;
  let u = new Array(T).fill(0);
  const cost = uu => { const q = predict(eta, uu); let s = 0; for (let t = 0; t <= T; t++) s += (q[t] - ref[t]) ** 2; for (let t = 0; t < T - 1; t++) s += lam * (uu[t + 1] - uu[t]) ** 2; return s; };
  let f = cost(u), mu = 1e-3;
  for (let it = 0; it < 200; it++) {
    const { q, J } = predictInputJac(eta, u);
    const H = Array.from({ length: T }, () => new Float64Array(T)), g = new Float64Array(T);
    for (let t = 0; t <= T; t++) { const row = J[t], r = q[t] - ref[t]; for (let i = 0; i < T; i++) { const ji = row[i]; if (!ji) continue; g[i] += ji * r; const Hi = H[i]; for (let j = 0; j < T; j++) Hi[j] += ji * row[j]; } }
    for (let t = 0; t < T - 1; t++) { const du = u[t + 1] - u[t]; g[t] -= lam * du; g[t + 1] += lam * du; H[t][t] += lam; H[t + 1][t + 1] += lam; H[t][t + 1] -= lam; H[t + 1][t] -= lam; }
    const free = []; for (let i = 0; i < T; i++) if (!(u[i] <= -U_MAX + 1e-12 && g[i] > 0) && !(u[i] >= U_MAX - 1e-12 && g[i] < 0)) free.push(i);
    let gn = 0; free.forEach(i => { gn = Math.max(gn, Math.abs(g[i])); });
    if (gn < 1e-12) break;
    let accepted = false;
    for (let tries = 0; tries < 30 && !accepted; tries++) {
      const A = free.map(i => free.map(j => i === j ? H[i][i] * (1 + mu) + 1e-30 : H[i][j]));
      const step = solve(A, free.map(i => -g[i])); if (!step) { mu *= 10; continue; }
      let alpha_ = 1, un, fn;
      for (let ls = 0; ls < 12; ls++) { un = u.slice(); free.forEach((i, k) => { un[i] = clip(u[i] + alpha_ * step[k], -U_MAX, U_MAX); }); fn = cost(un); if (fn < f) break; alpha_ *= 0.5; }
      if (fn < f) { const rel = (f - fn) / Math.max(f, 1e-300); u = un; f = fn; mu = Math.max(mu / 10, 1e-15); accepted = true; if (rel < 1e-13) return u; }
      else mu *= 10;
    }
    if (!accepted) break;
  }
  return u;
}
function makeRng(seed) {
  let s0 = (seed >>> 0) || 1, s1 = 0x9e3779b9 ^ s0;
  const next = () => { let x = s0, y = s1; s0 = y; x ^= x << 23; x ^= x >>> 17; x ^= y ^ (y >>> 26); s1 = x; return ((s0 + s1) >>> 0) / 4294967296; };
  let spare = null;
  const normal = () => { if (spare !== null) { const v = spare; spare = null; return v; } let u1 = 0; while (u1 <= 1e-12) u1 = next(); const u2 = next(), r = Math.sqrt(-2 * Math.log(u1)); spare = r * Math.sin(2 * Math.PI * u2); return r * Math.cos(2 * Math.PI * u2); };
  return { next, normal, uniform: (lo, hi) => lo + (hi - lo) * next() };
}
function multisine(T, rng, n = 3, fmin = 0.02, fmax = 0.8) {
  const f = []; for (let i = 0; i < n; i++) f.push(Math.exp(rng.uniform(Math.log(fmin), Math.log(fmax))));
  const s = new Array(T).fill(0);
  for (let i = 0; i < n; i++) { const p = rng.uniform(0, 2 * Math.PI); for (let t = 0; t < T; t++) s[t] += Math.sin(2 * Math.PI * f[i] * DT * t + p); }
  const m = Math.max(...s.map(Math.abs)); return s.map(v => v / m);
}
function sampleSystem(rng) {
  const c = [rng.uniform(-1, 1), rng.uniform(-1, 1)];
  const eta = ETA0.map((e, j) => e + M[j][0] * c[0] + M[j][1] * c[1] + EPS_SD[j] * rng.normal());
  return { eta, context: c, gamma: 0.035 + 0.012 * rng.uniform(0, 1), bias: 0.006 * c[0] + 0.002 * rng.normal() };
}
const shiftSystem = (sys, dir, delta) => Object.assign({}, sys, { eta: sys.eta.map((v, j) => clip(v + delta * dir[j], LOW[j] + 0.02, HIGH[j])) });
const quantile = (arr, p) => { const a = [...arr].sort((x, y) => x - y), i = (a.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return a[lo] + (a[hi] - a[lo]) * (i - lo); };
/* The whole pipeline on one system: experiment, context, posterior, prediction, misfit, tracking. */
function pipeline(fam, sys, uDesign, ref, rng, threshold) {
  const y = simulate(sys, uDesign, rng.normal).y, ad = adapt(fam, uDesign, y);
  const uv = multisine(300, rng, 6), qv = simulate(sys, uv).q;
  const etas = { none: fam.mean, context: decode(fam, ad.z), reference: sys.eta };
  const pred = {}, track = {}, uff = {}, qff = {};
  for (const m of Object.keys(etas)) { pred[m] = rmse(predict(etas[m], uv), qv); uff[m] = feedforward(etas[m], ref, 0.05); qff[m] = simulate(sys, uff[m]).q; track[m] = rmse(qff[m], ref); }
  return { y, z: ad.z, cov: ad.cov, misfit: ad.misfit, flagged: ad.misfit > threshold, zTrue: project(fam, sys.eta), uv, qv, pred, track, uff, qff, etas };
}
/* NET-END */

if (typeof module !== "undefined" && module.exports) {
  module.exports = { simulate, adapt, decode, project, feedforward, makeRng, multisine, sampleSystem, shiftSystem, quantile, pipeline, rmse };
  return;
}

const D = window.FS09, FAM = D.family;
const TC = D.ref.map((_, t) => DT * t), TV = Array.from({ length: 301 }, (_, t) => DT * t);
function eig2(A) {
  const a = A[0][0], b = A[0][1], c = A[1][1], tr = a + c, det = a * c - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
  let v1 = Math.abs(b) > 1e-14 ? [l1 - c, b] : (a >= c ? [1, 0] : [0, 1]); const n1 = Math.hypot(...v1); v1 = v1.map(x => x / n1);
  return { values: [l1, l2], vectors: [[v1[0], -v1[1]], [v1[1], v1[0]]] };
}
function ellipsePath(ctx, X, Y, m, cov, nStd) {
  const { values, vectors } = eig2(cov), a = nStd * Math.sqrt(Math.max(0, values[0])), b = nStd * Math.sqrt(Math.max(0, values[1]));
  ctx.beginPath();
  for (let i = 0; i <= 90; i++) { const th = 2 * Math.PI * i / 90, ex = a * Math.cos(th), ey = b * Math.sin(th); const px = m[0] + vectors[0][0] * ex + vectors[0][1] * ey, py = m[1] + vectors[1][0] * ex + vectors[1][1] * ey; i ? ctx.lineTo(X(px), Y(py)) : ctx.moveTo(X(px), Y(py)); }
  ctx.closePath();
}
const star = (ctx, x, y) => { ctx.beginPath(); for (let i = 0; i < 10; i++) { const r = i % 2 ? 4 : 9, a = -Math.PI / 2 + i * Math.PI / 5; ctx.lineTo(x + r * Math.cos(a), y + r * Math.sin(a)); } ctx.closePath(); ctx.fill(); };

/* calibration of the misfit threshold on 60 fresh in-family systems, once */
const calRng = makeRng(4242), cal = [];
for (let i = 0; i < 60; i++) { const s = sampleSystem(calRng); cal.push(adapt(FAM, D.uDesign, simulate(s, D.uDesign, calRng.normal).y).misfit); }
const THRESHOLD = quantile(cal, 0.95);

const st = { seed: 9, delta: 0 };
let P = null, SYS = null;
function run() {
  const rng = makeRng(st.seed * 2029 + 11);
  SYS = shiftSystem(sampleSystem(rng), D.direction, st.delta);
  P = pipeline(FAM, SYS, D.uDesign, D.ref, rng, THRESHOLD);
}
function draw() {
  if (!P) run();
  const narrow = narrowOf($("#capZ"), 600);
  // context plane
  const cz = $("#capZ"), lim = Math.max(3, Math.abs(P.zTrue[0]) + 0.5, Math.abs(P.zTrue[1]) + 0.5, Math.abs(P.z[0]) + 0.5, Math.abs(P.z[1]) + 0.5);
  const fr = frame(cz, narrow ? 1.1 : 1.0, { x: [-lim, lim], y: [-lim, lim], xlabel: "z₁", ylabel: "z₂" });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--rule"); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.ellipse(X(0), Y(0), X(2) - X(0), Y(0) - Y(2), 0, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = alpha("--muted", 0.5); FAM.z.forEach(z => { ctx.beginPath(); ctx.arc(X(z[0]), Y(z[1]), 3, 0, 2 * Math.PI); ctx.fill(); });
  ctx.strokeStyle = css(P.flagged ? "--warm" : "--post"); ctx.lineWidth = 2; ellipsePath(ctx, X, Y, P.z, P.cov, 2); ctx.stroke();
  ctx.fillStyle = css(P.flagged ? "--warm" : "--post"); ctx.beginPath(); ctx.arc(X(P.z[0]), Y(P.z[1]), 5, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = css("--ink"); star(ctx, X(P.zTrue[0]), Y(P.zTrue[1]));
  // prediction
  const cp = $("#capP");
  let ymax = 0.3; P.qv.forEach(v => { ymax = Math.max(ymax, Math.abs(v)); }); ymax = Math.ceil(ymax * 10) / 10;
  const frP = frame(cp, narrow ? 1.5 : 2.0, { x: [0, 30], y: [-ymax, ymax], xlabel: "time [s]", ylabel: "prediction of a fresh input", xticks: [0, 10, 20, 30] });
  plotLine(frP, TV, P.qv, { color: css("--ink"), width: 2.4 });
  plotLine(frP, TV, predict(FAM.mean, P.uv), { color: css("--muted"), width: 1.3 });
  plotLine(frP, TV, predict(P.etas.context, P.uv), { color: css("--post"), width: 1.6 });
  // tracking
  const ct = $("#capT");
  const frT = frame(ct, narrow ? 1.5 : 2.0, { x: [0, 12], y: [-0.25, 0.35], xlabel: "time [s]", ylabel: "feedforward tracking", xticks: [0, 4, 8, 12], yticks: [-0.2, 0, 0.2] });
  plotLine(frT, TC, D.ref, { color: css("--ink"), width: 2.6 });
  plotLine(frT, TC, P.qff.none, { color: css("--muted"), width: 1.3 });
  plotLine(frT, TC, P.qff.context, { color: css("--post"), width: 1.6 });
  plotLine(frT, TC, P.qff.reference, { color: css("--ink"), width: 1.2, dash: [5, 4] });
  // readouts
  $("#rEta").textContent = `(${SYS.eta.map(v => fmt(v, 3)).join(", ")})`;
  $("#rZ").textContent = `(${P.z.map(v => fmt(v, 2)).join(", ")}) ± (${fmt(Math.sqrt(P.cov[0][0]), 2)}, ${fmt(Math.sqrt(P.cov[1][1]), 2)})`;
  $("#rMis").textContent = `${fmt(P.misfit, 4)} against ${fmt(THRESHOLD, 4)}: ${P.flagged ? "flagged" : "in family"}`;
  $("#rPred").textContent = `${fmt(P.pred.none, 4)} / ${fmt(P.pred.context, 4)} / ${fmt(P.pred.reference, 4)}`;
  $("#rTrack").textContent = `${fmt(P.track.none, 4)} / ${fmt(P.track.context, 4)} / ${fmt(P.track.reference, 4)}`;
  const dz = [P.zTrue[0] - P.z[0], P.zTrue[1] - P.z[1]], Pr = inv2(P.cov), maha = Math.sqrt(dz[0] * (Pr[0][0] * dz[0] + Pr[0][1] * dz[1]) + dz[1] * (Pr[1][0] * dz[0] + Pr[1][1] * dz[1]));
  const dist = (() => { const best = decode(FAM, P.zTrue); const r = SYS.eta.map((v, j) => (v - best[j]) / FAM.scale[j]); return Math.sqrt(r.reduce((s, v) => s + v * v, 0)); })();
  caption($("#capStatus"), `A new system${st.delta > 0 ? ` pushed [[\\delta = ${fmt(st.delta, 1)}]] out of the family along chapter 07's direction (whitened distance from the family's plane ${fmt(dist, 1)})` : " drawn from the family"}, coefficients [[(${SYS.eta.map(v => fmt(v, 3)).join(", ")})]]. The designed 6 s experiment of chapter 08 is run on it with noise and disturbance from the page's generator; chapter 04's adaptation gives [[\\hat z = (${P.z.map(v => fmt(v, 2)).join(", ")})]] with posterior standard deviations ${fmt(Math.sqrt(P.cov[0][0]), 2)} and ${fmt(Math.sqrt(P.cov[1][1]), 2)}; the system's best context in the family (star) lies ${fmt(maha, 1)} posterior standard deviations away${maha > 3 ? ", outside the ellipse: the designed experiment's posterior is precise and biased, as chapter 06 found" : ""}. Chapter 07's misfit test, calibrated live on 60 in-family systems (threshold ${fmt(THRESHOLD, 4)}; Python's calibration gives ${fmt(D.threshold, 4)}), reads ${fmt(P.misfit, 4)} and ${P.flagged ? "flags the system" : "lets it pass"}. Middle: the disturbance-free response to a fresh 30 s input (black) predicted by the average system (grey, RMSE ${fmt(P.pred.none, 4)}) and the adapted context (blue, ${fmt(P.pred.context, 4)}); the true coefficients would give ${fmt(P.pred.reference, 4)}. Right: chapter 08's feedforward tracking of the reference from each model: ${fmt(P.track.none, 4)}, ${fmt(P.track.context, 4)} and ${fmt(P.track.reference, 4)} (true coefficients dashed). ` + (st.delta >= 0.4 ? `Outside the family the context's prediction error no longer approaches the true coefficients' and the test fires on only some systems: the lesson of chapter 07.` : `Six seconds and two numbers take the system most of the way from the average member to the truth, for prediction and for control alike.`));
}
$("#capNew").addEventListener("click", () => { st.seed++; run(); draw(); });
$("#capD").addEventListener("input", e => { st.delta = +e.target.value; $("#capDV").textContent = fmt(st.delta, 1); run(); draw(); });
$("#capDV").textContent = fmt(st.delta, 1);
draw();
onRedraw(draw);
})();
