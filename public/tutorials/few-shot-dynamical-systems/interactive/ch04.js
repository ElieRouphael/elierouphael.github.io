/* Learning to Identify New Dynamical Systems, chapter 04, Few-shot identification of a new system:
   the live demos. The pretrained family of chapter 03 is frozen (ch04-data.js) and only the
   two-number context of a new system is estimated here, by maximum a posteriori with a standard
   normal prior, solved by Levenberg–Marquardt with an exact Jacobian (chain rule through the
   decoder), plus the Laplace posterior; the from-scratch fit of chapter 02 runs alongside for
   comparison, and fresh systems, experiments and validation inputs come from the page's own
   random generator. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const DT = 0.1, X0 = [0.2, 0.0];
const SIGMA_Y = 0.01, SIGMA_W = 0.01, RHO = 0.8;
const LOW = [0.05, 0.01, 0.05, 0.0], HIGH = [5.0, 2.0, 3.0, 5.0], GENERIC = [1.0, 0.5, 1.0, 0.5];
const ETA0 = [1.20, 0.32, 0.90, 1.00], M = [[0.55, 0.12], [0.07, 0.11], [0.16, -0.09], [0.30, 0.45]], EPS_SD = [0.015, 0.004, 0.008, 0.04];
const actuator = u => 0.3 * Math.tanh(u / 0.6);
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
    for (let r = 0; r < n; r++) { if (r === c) continue; const m = A[r][c] / A[c][c]; if (!m) continue; for (let k = c; k <= n; k++) A[r][k] -= m * A[c][k]; }
  }
  return A.map((r, i) => r[n] / r[i]);
}
const inv2 = A => { const d = A[0][0] * A[1][1] - A[0][1] * A[1][0]; return [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]]; };
/* Chapter 02's bounded from-scratch fit (Levenberg–Marquardt, projection onto the box). */
function fitLS(u, y, guess = GENERIC) {
  let x = guess.map((g, i) => clip(g, LOW[i], HIGH[i]));
  const cost = eta => { const q = predictJac(eta, u).q; let s = 0; for (let i = 0; i < y.length; i++) s += (q[i] - y[i]) ** 2; return s; };
  let f = cost(x), lam = 1e-3;
  for (let it = 0; it < 300; it++) {
    const { q, J } = predictJac(x, u);
    const H = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], g = [0, 0, 0, 0];
    for (let t = 0; t < y.length; t++) { const r = q[t] - y[t], Jt = J[t]; for (let i = 0; i < 4; i++) { g[i] += Jt[i] * r; for (let j = 0; j < 4; j++) H[i][j] += Jt[i] * Jt[j]; } }
    const free = [0, 1, 2, 3].filter(i => !(x[i] <= LOW[i] + 1e-12 && g[i] > 0) && !(x[i] >= HIGH[i] - 1e-12 && g[i] < 0));
    if (!free.length || Math.max(...free.map(i => Math.abs(g[i]))) < 1e-14) break;
    let accepted = false;
    for (let tries = 0; tries < 40 && !accepted; tries++) {
      const A = free.map((i, a) => free.map((j, b) => a === b ? H[i][j] * (1 + lam) + 1e-30 : H[i][j]));
      const step = solve(A, free.map(i => -g[i])); if (!step) { lam *= 10; continue; }
      const xn = x.slice(); free.forEach((i, k) => { xn[i] = clip(x[i] + step[k], LOW[i], HIGH[i]); });
      const fn = cost(xn);
      if (fn < f) { const rel = (f - fn) / Math.max(f, 1e-300), dx = Math.max(...free.map(i => Math.abs(xn[i] - x[i]) / Math.max(1e-12, Math.abs(x[i])))); x = xn; f = fn; lam = Math.max(lam / 3, 1e-15); accepted = true; if (rel < 1e-13 || dx < 1e-10) return { eta: x, sse: f }; }
      else lam *= 4;
    }
    if (!accepted) break;
  }
  return { eta: x, sse: f };
}
/* The decoder of chapter 03 (clipped to the admissible box). */
const decode = (fam, z) => fam.mean.map((m, j) => clip(m + fam.U[j][0] * z[0] + fam.U[j][1] * z[1], LOW[j], HIGH[j]));
/* dynutils.adapt with an adjustable prior weight: minimise ||(q(decode(z)) - y) / sigma||^2 + t^2 ||z||^2.
   Returns the MAP context, the Laplace covariance (S^T S / sigma^2 + t^2 I)^-1, the output misfit and the
   two precision matrices. t = 0 is plain least squares in z. */
function adapt(fam, u, y, t = 1) {
  const sig = fam.sigma, s2 = sig * sig;
  let z = [0, 0];
  /* Chain rule through the decoder; a coefficient pinned by the clip has zero derivative with respect to z. */
  const sens = zz => { const raw = fam.mean.map((m, j) => m + fam.U[j][0] * zz[0] + fam.U[j][1] * zz[1]), live = raw.map((v, j) => v > LOW[j] && v < HIGH[j] ? 1 : 0); const { q, J } = predictJac(decode(fam, zz), u); const S = J.map(row => [0, 1].map(k => row.reduce((s, v, j) => s + live[j] * v * fam.U[j][k], 0))); return { q, S }; };
  const cost = zz => { const q = predictJac(decode(fam, zz), u).q; let s = 0; for (let i = 0; i < y.length; i++) s += (q[i] - y[i]) ** 2; return s / s2 + t * t * (zz[0] ** 2 + zz[1] ** 2); };
  let f = cost(z), lam = 1e-3, iters = 0;
  for (; iters < 2000; iters++) {
    const { q, S } = sens(z);
    const H = [[t * t, 0], [0, t * t]], g = [t * t * z[0], t * t * z[1]];
    for (let i = 0; i < y.length; i++) { const r = (q[i] - y[i]) / s2; g[0] += S[i][0] * r; g[1] += S[i][1] * r; H[0][0] += S[i][0] * S[i][0] / s2; H[0][1] += S[i][0] * S[i][1] / s2; H[1][1] += S[i][1] * S[i][1] / s2; }
    H[1][0] = H[0][1];
    if (Math.hypot(g[0], g[1]) < 1e-12) break;
    let accepted = false;
    for (let tries = 0; tries < 40 && !accepted; tries++) {
      const A = [[H[0][0] * (1 + lam), H[0][1]], [H[1][0], H[1][1] * (1 + lam)]];
      const step = solve(A, [-g[0], -g[1]]); if (!step) { lam *= 10; continue; }
      const zn = [z[0] + step[0], z[1] + step[1]], fn = cost(zn);
      if (fn < f) { const rel = (f - fn) / Math.max(f, 1e-300); z = zn; f = fn; lam = Math.max(lam / 10, 1e-15); accepted = true; if (rel < 1e-14 || Math.hypot(step[0], step[1]) < 1e-10) { iters++; break; } }
      else lam *= 10;
    }
    if (!accepted) break;
    if (lam > 1e8) break;
  }
  const { q, S } = sens(z);
  const data = [[0, 0], [0, 0]]; for (let i = 0; i < y.length; i++) { data[0][0] += S[i][0] * S[i][0] / s2; data[0][1] += S[i][0] * S[i][1] / s2; data[1][1] += S[i][1] * S[i][1] / s2; } data[1][0] = data[0][1];
  const prec = [[data[0][0] + t * t, data[0][1]], [data[1][0], data[1][1] + t * t]];
  const cov = inv2(prec);
  return { z, cov, misfit: rmse(Array.from(q), y), dataPrecision: data, prec, iters };
}
/* dynutils.Family.project: best context for known coefficients, in the whitened metric. */
function project(fam, eta) {
  const A = fam.U.map((row, j) => row.map(u => u / fam.scale[j])), b = eta.map((v, j) => (v - fam.mean[j]) / fam.scale[j]);
  const N = [[0, 0], [0, 0]], r = [0, 0];
  A.forEach((row, i) => { for (let p = 0; p < 2; p++) { r[p] += row[p] * b[i]; for (let q = 0; q < 2; q++) N[p][q] += row[p] * row[q]; } });
  return solve(N, r);
}
/* Eigen-decomposition of a symmetric 2x2: { values (descending), vectors (columns) }. */
function eig2(A) {
  const a = A[0][0], b = A[0][1], c = A[1][1], tr = a + c, det = a * c - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
  let v1 = Math.abs(b) > 1e-14 ? [l1 - c, b] : (a >= c ? [1, 0] : [0, 1]); const n1 = Math.hypot(...v1); v1 = v1.map(x => x / n1);
  return { values: [l1, l2], vectors: [[v1[0], -v1[1]], [v1[1], v1[0]]] };
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
/* dynutils.sample_system with the page's generator. */
function sampleSystem(rng) {
  const c = [rng.uniform(-1, 1), rng.uniform(-1, 1)];
  const eta = ETA0.map((e, j) => e + M[j][0] * c[0] + M[j][1] * c[1] + EPS_SD[j] * rng.normal());
  return { eta, context: c, gamma: 0.035 + 0.012 * rng.uniform(0, 1), bias: 0.006 * c[0] + 0.002 * rng.normal() };
}
/* One complete comparison on one system: T samples of data, 30 s validation. */
function compare(fam, sys, T, rng, t = 1) {
  const u = multisine(T, rng), y = simulate(sys, u, rng.normal).y;
  const uv = multisine(300, rng, 6), qv = simulate(sys, uv).q;
  const ad = adapt(fam, u, y, t), sc = fitLS(u, y);
  return { u, y, uv, qv, ad, sc, rmse: { none: rmse(predict(fam.mean, uv), qv), scratch: rmse(predict(sc.eta, uv), qv), context: rmse(predict(decode(fam, ad.z), uv), qv), truth: rmse(predict(sys.eta, uv), qv) } };
}
/* NET-END */

if (typeof module !== "undefined" && module.exports) {
  module.exports = { simulate, predict, fitLS, adapt, decode, project, eig2, makeRng, multisine, sampleSystem, compare, rmse };
  return;
}

const D = window.FS04, FAM = D.family;
const timeAxis = n => Array.from({ length: n }, (_, t) => DT * t);
function ellipsePath(ctx, X, Y, mean, cov, nStd) {
  const { values, vectors } = eig2(cov), a = nStd * Math.sqrt(Math.max(0, values[0])), b = nStd * Math.sqrt(Math.max(0, values[1]));
  ctx.beginPath();
  for (let i = 0; i <= 90; i++) { const th = 2 * Math.PI * i / 90, ex = a * Math.cos(th), ey = b * Math.sin(th); const px = mean[0] + vectors[0][0] * ex + vectors[0][1] * ey, py = mean[1] + vectors[1][0] * ex + vectors[1][1] * ey; i ? ctx.lineTo(X(px), Y(py)) : ctx.moveTo(X(px), Y(py)); }
  ctx.closePath();
}

/* ============================================================ hero */
const hero = { T: 20, t: 1, source: "nb", seed: 1 };
let H = null;
function runHero() {
  if (hero.source === "nb") {
    const ad = adapt(FAM, D.u_short, D.y_short, hero.t), sc = fitLS(D.u_short, D.y_short);
    H = { sys: D.nu, u: D.u_short, y: D.y_short, uv: D.u_val, qv: D.q_val, ad, sc, rmse: { none: rmse(predict(FAM.mean, D.u_val), D.q_val), scratch: rmse(predict(sc.eta, D.u_val), D.q_val), context: rmse(predict(decode(FAM, ad.z), D.u_val), D.q_val), truth: rmse(predict(D.nu.eta, D.u_val), D.q_val) } };
  } else {
    const rng = makeRng(hero.seed * 7919 + 11), sys = sampleSystem(rng);
    H = Object.assign({ sys }, compare(FAM, sys, hero.T, rng, hero.t));
  }
  H.zTrue = project(FAM, H.sys.eta);
}
function drawHero() {
  if (!H) runHero();
  const { ad, sc, uv, qv, zTrue } = H;
  const cz = $("#heroZ"), narrow = narrowOf(cz, 600);
  const lim = Math.max(3, Math.abs(zTrue[0]) + 0.4, Math.abs(zTrue[1]) + 0.4, Math.abs(ad.z[0]) + 0.4, Math.abs(ad.z[1]) + 0.4);
  const fr = frame(cz, narrow ? 1.1 : 1.0, { x: [-lim, lim], y: [-lim, lim], xlabel: "z₁", ylabel: "z₂" });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--rule"); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.ellipse(X(0), Y(0), X(2) - X(0), Y(0) - Y(2), 0, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = alpha("--muted", 0.55); FAM.z.forEach(z => { ctx.beginPath(); ctx.arc(X(z[0]), Y(z[1]), 3, 0, 2 * Math.PI); ctx.fill(); });
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 2; ellipsePath(ctx, X, Y, ad.z, ad.cov, 2); ctx.stroke();
  ctx.fillStyle = css("--post"); ctx.beginPath(); ctx.arc(X(ad.z[0]), Y(ad.z[1]), 5, 0, 2 * Math.PI); ctx.fill();
  // true system as a star
  ctx.fillStyle = css("--ink"); ctx.beginPath(); for (let i = 0; i < 10; i++) { const r = i % 2 ? 4 : 9, a = -Math.PI / 2 + i * Math.PI / 5; ctx.lineTo(X(zTrue[0]) + r * Math.cos(a), Y(zTrue[1]) + r * Math.sin(a)); } ctx.closePath(); ctx.fill();
  const cp = $("#heroP"), t = timeAxis(uv.length + 1);
  let ymax = 0.3; qv.forEach(v => { ymax = Math.max(ymax, Math.abs(v)); }); const pNone = predict(FAM.mean, uv), pSc = predict(sc.eta, uv), pCx = predict(decode(FAM, ad.z), uv); [pNone, pSc, pCx].forEach(p => p.forEach(v => { ymax = Math.max(ymax, Math.min(1.5, Math.abs(v))); })); ymax = Math.ceil(ymax * 10) / 10;
  const frP = frame(cp, narrow ? 1.5 : 2.2, { x: [0, 30], y: [-ymax, ymax], xlabel: "time [s]", ylabel: "position", xticks: [0, 5, 10, 15, 20, 25, 30] });
  plotLine(frP, t, qv, { color: css("--ink"), width: 2.6 });
  plotLine(frP, t, pNone, { color: css("--muted"), width: 1.5 });
  plotLine(frP, t, pSc, { color: css("--warm"), width: 1.5 });
  plotLine(frP, t, pCx, { color: css("--post"), width: 1.8 });
  $("#rNone").textContent = fmt(H.rmse.none, 4); $("#rScr").textContent = fmt(H.rmse.scratch, 4); $("#rCtx").textContent = fmt(H.rmse.context, 4); $("#rTruth").textContent = fmt(H.rmse.truth, 4);
  $("#rZ").textContent = `(${fmt(ad.z[0], 3)}, ${fmt(ad.z[1], 3)}) ± (${fmt(Math.sqrt(ad.cov[0][0]), 3)}, ${fmt(Math.sqrt(ad.cov[1][1]), 3)})`;
  $("#rMis").textContent = fmt(ad.misfit, 4) + " (σ̂ " + fmt(FAM.sigma, 4) + ")";
  const dz = [zTrue[0] - ad.z[0], zTrue[1] - ad.z[1]], P = inv2(ad.cov), maha = Math.sqrt(dz[0] * (P[0][0] * dz[0] + P[0][1] * dz[1]) + dz[1] * (P[1][0] * dz[0] + P[1][1] * dz[1]));
  const ev = eig2(ad.dataPrecision).values;
  const nat = e => ({ f: Math.sqrt(e[0]) / (2 * Math.PI), zeta: e[1] / (2 * Math.sqrt(e[0])) });
  const nS = nat(sc.eta), nT = nat(H.sys.eta);
  let text = `${hero.source === "nb" ? "The notebook's new system" : "A fresh member of the family"} seen through ${fmt(DT * H.u.length, 1)} s of data (${H.u.length} samples)${hero.source === "nb" ? ", a third of its " + fmt(2 * Math.PI / Math.sqrt(H.sys.eta[0]), 1) + " s period" : ""}. Left: the training contexts (grey), the posterior for [[z]] with its two-sigma ellipse (blue) and the true system's context (star), [[z = (${fmt(zTrue[0], 2)}, ${fmt(zTrue[1], 2)})]], which lies ${fmt(maha, 2)} posterior standard deviations from the estimate${maha <= 2 ? ", inside the ellipse" : ", outside the ellipse"}. Right: the true response to a 30 s input the model has never seen (black) against the mean system (grey, RMSE ${fmt(H.rmse.none, 4)}), the four-coefficient fit from scratch (orange, ${fmt(H.rmse.scratch, 4)}) and the adapted two-number context (blue, ${fmt(H.rmse.context, 4)})${hero.source === "nb" && hero.t === 1 ? ", the notebook's 0.056, 0.158 and 0.014" : ""}. `;
  text += `The from-scratch fit explains the short record with [[\\hat\\eta = (${sc.eta.map(v => fmt(v, 3)).join(", ")})]]: ${nS.zeta < 0.03 ? "a barely damped oscillator" : "an oscillator"} at ${fmt(nS.f, 3)} Hz with damping ratio ${fmt(nS.zeta, 3)}, against the truth's ${fmt(nT.f, 3)} Hz and ${fmt(nT.zeta, 3)}${sc.eta.some((v, j) => v <= LOW[j] + 1e-9 || v >= HIGH[j] - 1e-9) ? ", with coefficients parked on the bounds" : ""}. `;
  text += `The data's precision about [[z]] has eigenvalues ${fmt(ev[0], 1)} and ${fmt(ev[1], 2)} against the prior's ${fmt(hero.t * hero.t, 2)} per direction: one direction of the context is pinned by the record, the other is ${ev[1] < 1 ? "mostly the prior's guess" : "also determined by the data"}. The true coefficients themselves would score ${fmt(H.rmse.truth, 4)}, the floor set by the quintic term, the bias and the clipped decoder.`;
  caption($("#heroStatus"), text);
}
$("#hT").addEventListener("input", e => { hero.T = +e.target.value; $("#hTV").textContent = fmt(DT * hero.T, 1) + " s"; hero.source = "fresh"; sync(); runHero(); drawHero(); });
$("#hPrior").addEventListener("input", e => { hero.t = +e.target.value; $("#hPriorV").textContent = hero.t === 0 ? "0 (no prior)" : fmt(hero.t, 2); runHero(); drawHero(); });
$("#hNew").addEventListener("click", () => { hero.seed++; hero.source = "fresh"; sync(); runHero(); drawHero(); });
$("#hNb").addEventListener("click", () => { hero.source = "nb"; hero.T = 20; $("#hT").value = 20; $("#hTV").textContent = "2.0 s"; sync(); runHero(); drawHero(); });
function sync() { $("#hNb").setAttribute("aria-pressed", hero.source === "nb"); }
$("#hTV").textContent = fmt(DT * hero.T, 1) + " s"; $("#hPriorV").textContent = fmt(hero.t, 2);

/* ============================================================ prior x likelihood = posterior */
const pr = { t: 1 };
function drawPrior() {
  const ad = adapt(FAM, D.u_short, D.y_short, pr.t), zTrue = D.py.z_true;
  const cv = $("#priC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.1 : 1.6, { x: [-3.2, 3.2], y: [-3.6, 2.8], xlabel: "z₁", ylabel: "z₂", xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2] });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--muted", 0.45); FAM.z.forEach(z => { ctx.beginPath(); ctx.arc(X(z[0]), Y(z[1]), 2.5, 0, 2 * Math.PI); ctx.fill(); });
  // prior: covariance I / t^2
  if (pr.t > 0) { ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ellipsePath(ctx, X, Y, [0, 0], [[1 / (pr.t * pr.t), 0], [0, 1 / (pr.t * pr.t)]], 2); ctx.stroke(); ctx.setLineDash([]); }
  // likelihood alone (t = 0)
  const like = adapt(FAM, D.u_short, D.y_short, 0);
  ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.8; ellipsePath(ctx, X, Y, like.z, like.cov, 2); ctx.stroke();
  ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(like.z[0]), Y(like.z[1]), 4, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 2.2; ellipsePath(ctx, X, Y, ad.z, ad.cov, 2); ctx.stroke();
  ctx.fillStyle = css("--post"); ctx.beginPath(); ctx.arc(X(ad.z[0]), Y(ad.z[1]), 5, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = css("--ink"); ctx.beginPath(); for (let i = 0; i < 10; i++) { const r = i % 2 ? 4 : 9, a = -Math.PI / 2 + i * Math.PI / 5; ctx.lineTo(X(zTrue[0]) + r * Math.cos(a), Y(zTrue[1]) + r * Math.sin(a)); } ctx.closePath(); ctx.fill();
  const val = rmse(predict(decode(FAM, ad.z), D.u_val), D.q_val), valL = rmse(predict(decode(FAM, like.z), D.u_val), D.q_val);
  const dz = [zTrue[0] - ad.z[0], zTrue[1] - ad.z[1]], P = inv2(ad.cov), maha = Math.sqrt(dz[0] * (P[0][0] * dz[0] + P[0][1] * dz[1]) + dz[1] * (P[1][0] * dz[0] + P[1][1] * dz[1]));
  caption($("#priCap"), `The notebook's 2 s experiment. Orange: the likelihood alone (prior weight 0), a long ellipse because the short record barely constrains [[z_2]]; its centre [[(${fmt(like.z[0], 2)}, ${fmt(like.z[1], 2)})]] predicts the validation input with RMSE ${fmt(valL, 4)}. Grey dashed: the two-sigma circle of the prior with weight [[${fmt(pr.t, 2)}]]${pr.t === 1 ? " (the standard normal of the family)" : pr.t === 0 ? " (switched off)" : ""}. Blue: the posterior, centre [[(${fmt(ad.z[0], 2)}, ${fmt(ad.z[1], 2)})]], standard deviations ${fmt(Math.sqrt(ad.cov[0][0]), 2)} and ${fmt(Math.sqrt(ad.cov[1][1]), 2)}, validation RMSE ${fmt(val, 4)}; the truth (star) is ${fmt(maha, 2)} posterior standard deviations away. ` + (pr.t >= 2.5 ? `A tight prior pulls the estimate towards the mean system and shrinks the ellipse, but for the wrong reason: it is more confident without having seen more data, and the truth now sits far outside it. ` : pr.t === 0 ? `With no prior the estimate is the best this experiment can do on its own: on this draw it is also the most accurate, because the true system happens to lie two standard deviations out along the poorly determined direction. ` : pr.t === 1 ? `With the family's own prior the estimate sits between the likelihood and the origin: on this particular draw the prior pulls it away from a truth that lies two standard deviations out, and the validation error is worse than without it. Chapter 05 shows that over many systems the prior wins on average, which is what a prior is for. ` : "") + `Posterior precision is data precision plus prior precision; the ellipse can only shrink as the prior tightens, whatever the data say.`);
}
$("#pT").addEventListener("input", e => { pr.t = +e.target.value; $("#pTV").textContent = pr.t === 0 ? "0 (no prior)" : fmt(pr.t, 2); drawPrior(); });
$("#pTV").textContent = fmt(pr.t, 2);

/* ============================================================ exercise 2: twenty systems */
const ex2 = { seed: 3 };
let ex2Cache = null;
function drawEx2() {
  if (!ex2Cache || ex2Cache.seed !== ex2.seed) {
    const rng = makeRng(ex2.seed * 104729 + 7), rows = [];
    for (let i = 0; i < 20; i++) { const sys = sampleSystem(rng); const c = compare(FAM, sys, 20, rng); rows.push(c.rmse); }
    ex2Cache = { seed: ex2.seed, rows };
  }
  const rows = ex2Cache.rows, py = D.py.ex2;
  const cv = $("#ex2C"), narrow = narrowOf(cv);
  const lo = 0.003, hi = 0.3, lg = v => Math.log10(v);
  const fr = frame(cv, narrow ? 1.2 : 1.5, { x: [lg(lo), lg(hi)], y: [lg(lo), lg(hi)], xlabel: "from scratch, RMSE", ylabel: "pretrained + context, RMSE", xticks: [-2, -1.5, -1, -0.5], yticks: [-2, -1.5, -1, -0.5], xfmt: v => (10 ** v).toFixed(v < -1.6 ? 3 : 2), yfmt: v => (10 ** v).toFixed(v < -1.6 ? 3 : 2) });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--muted"); ctx.setLineDash([4, 3]); seg(ctx, X(lg(lo)), Y(lg(lo)), X(lg(hi)), Y(lg(hi))); ctx.setLineDash([]);
  ctx.fillStyle = alpha("--muted", 0.5); py.scratch.forEach((s, i) => { ctx.beginPath(); ctx.arc(X(lg(Math.min(hi, Math.max(lo, s)))), Y(lg(Math.min(hi, Math.max(lo, py.context[i])))), 3.5, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = css("--post"); rows.forEach(r => { ctx.beginPath(); ctx.arc(X(lg(Math.min(hi, Math.max(lo, r.scratch)))), Y(lg(Math.min(hi, Math.max(lo, r.context)))), 4.5, 0, 2 * Math.PI); ctx.fill(); });
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const wins = rows.filter(r => r.context < r.scratch).length, pyWins = py.context.filter((c, i) => c < py.scratch[i]).length;
  const ratios = rows.map(r => r.scratch / r.context).sort((a, b) => a - b), pyR = py.scratch.map((s, i) => s / py.context[i]).sort((a, b) => a - b);
  caption($("#ex2Cap"), `Twenty new systems, each with its own 2 s experiment and 30 s validation input (blue: the page's draws; grey: the notebook's). Points below the diagonal are wins for the adapted context. Page: mean RMSE ${fmt(mean(rows.map(r => r.scratch)), 4)} from scratch against ${fmt(mean(rows.map(r => r.context)), 4)} with the context, ${wins} wins of 20, from-scratch-to-context ratio from ${fmt(ratios[0], 2)} to ${fmt(ratios[19], 1)}, median ${fmt(ratios[10], 1)}. Notebook: ${fmt(mean(py.scratch), 4)} against ${fmt(mean(py.context), 4)}, ${pyWins} wins of 20, ratio from ${fmt(pyR[0], 2)} to ${fmt(pyR[19], 1)}. The context wins on average by a wide margin, but not "on essentially every draw" as the notebook's solution says: from scratch wins ${20 - pyWins} of the notebook's 20 pairs, twice by a factor above 2, when the random 2 s input happens to excite the system well and the prior pulls the context the wrong way. Chapter 05 measures how this gap closes as the budget grows.`);
}
$("#ex2New").addEventListener("click", () => { ex2.seed++; drawEx2(); });

function drawAll() { drawHero(); drawPrior(); drawEx2(); }
drawAll();
onRedraw(drawAll);
})();
