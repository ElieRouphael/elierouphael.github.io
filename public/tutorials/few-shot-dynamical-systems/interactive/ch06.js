/* Learning to Identify New Dynamical Systems, chapter 06, Choosing the best experiment: the live
   demos. The Bayesian D-optimal design criterion of the notebook (expected log-determinant of the
   Fisher information of the context over the training contexts) is computed here with the exact
   output sensitivities of chapter 04, so candidate experiments are scored in the browser; the
   posteriors of section 2 and a scaled-down version of the 60-trial study are rerun live. The
   notebook's candidates, experiments and study results come from its own run (ch06-data.js).
   Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, histogram } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const DT = 0.1, X0 = [0.2, 0.0];
const SIGMA_Y = 0.01, SIGMA_W = 0.01, RHO = 0.8;
const LOW = [0.05, 0.01, 0.05, 0.0], HIGH = [5.0, 2.0, 3.0, 5.0];
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
const std = a => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
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
const decode = (fam, z) => fam.mean.map((m, j) => clip(m + fam.U[j][0] * z[0] + fam.U[j][1] * z[1], LOW[j], HIGH[j]));
/* Output sensitivity to the context, S = J_eta U, with clipped coefficients given zero derivative. */
function sensitivity(fam, z, u) {
  const raw = fam.mean.map((m, j) => m + fam.U[j][0] * z[0] + fam.U[j][1] * z[1]), live = raw.map((v, j) => v > LOW[j] && v < HIGH[j] ? 1 : 0);
  const { q, J } = predictJac(decode(fam, z), u);
  return { q, S: J.map(row => [0, 1].map(k => row.reduce((s, v, j) => s + live[j] * v * fam.U[j][k], 0))) };
}
/* Fisher information of the context for input u at context z: S^T S / sigma^2 + I. */
function fisher(fam, u, z) {
  const { S } = sensitivity(fam, z, u), s2 = fam.sigma * fam.sigma, F = [[1, 0], [0, 1]];
  for (const r of S) { F[0][0] += r[0] * r[0] / s2; F[0][1] += r[0] * r[1] / s2; F[1][1] += r[1] * r[1] / s2; }
  F[1][0] = F[0][1]; return F;
}
const logdet2 = F => Math.log(F[0][0] * F[1][1] - F[0][1] * F[1][0]);
/* The design score: expected log det over the training contexts. */
const designScore = (fam, u) => fam.z.reduce((s, z) => s + logdet2(fisher(fam, u, z)), 0) / fam.z.length;
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
function sampleSystem(rng) {
  const c = [rng.uniform(-1, 1), rng.uniform(-1, 1)];
  const eta = ETA0.map((e, j) => e + M[j][0] * c[0] + M[j][1] * c[1] + EPS_SD[j] * rng.normal());
  return { eta, context: c, gamma: 0.035 + 0.012 * rng.uniform(0, 1), bias: 0.006 * c[0] + 0.002 * rng.normal() };
}
/* Dominant frequency of an input by a zero-padded periodogram. */
function peakFreq(u) {
  const n = u.length, m = u.reduce((s, v) => s + v, 0) / n, N = 8 * n; let best = 0, bf = 0;
  for (let k = 1; k <= N / 2; k++) { let re = 0, im = 0; for (let t = 0; t < n; t++) { const ph = 2 * Math.PI * k * t / N; re += (u[t] - m) * Math.cos(ph); im -= (u[t] - m) * Math.sin(ph); } const P = re * re + im * im; if (P > best) { best = P; bf = k / (N * DT); } }
  return bf;
}
/* One trial of the section-3 study at budget T: a pool of `poolSize` candidates scored, the best,
   the worst and an unscored random experiment run on a fresh system, each adapted and validated. */
function designTrial(fam, T, rng, poolSize = 200) {
  let bestU = null, worstU = null, bestS = -Infinity, worstS = Infinity;
  for (let c = 0; c < poolSize; c++) { const u = multisine(T, rng), s = designScore(fam, u); if (s > bestS) { bestS = s; bestU = u; } if (s < worstS) { worstS = s; worstU = u; } }
  const randomU = multisine(T, rng), sys = sampleSystem(rng), uv = multisine(300, rng, 6), qv = simulate(sys, uv).q;
  const out = {};
  for (const [m, u] of [["worst", worstU], ["context", randomU], ["designed", bestU]]) { const y = simulate(sys, u, rng.normal).y; const ad = adapt(fam, u, y); out[m] = rmse(predict(decode(fam, ad.z), uv), qv); }
  return out;
}
/* NET-END */

if (typeof module !== "undefined" && module.exports) {
  module.exports = { simulate, predict, adapt, decode, project, fisher, designScore, sensitivity, makeRng, multisine, sampleSystem, designTrial, rmse, std, peakFreq, eig2 };
  return;
}

const D = window.FS06, FAM = D.family, T40 = 40;
const timeAxis = n => Array.from({ length: n }, (_, t) => DT * t);
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
function ellipsePath(ctx, X, Y, m, cov, nStd) {
  const { values, vectors } = eig2(cov), a = nStd * Math.sqrt(Math.max(0, values[0])), b = nStd * Math.sqrt(Math.max(0, values[1]));
  ctx.beginPath();
  for (let i = 0; i <= 90; i++) { const th = 2 * Math.PI * i / 90, ex = a * Math.cos(th), ey = b * Math.sin(th); const px = m[0] + vectors[0][0] * ex + vectors[0][1] * ey, py = m[1] + vectors[1][0] * ex + vectors[1][1] * ey; i ? ctx.lineTo(X(px), Y(py)) : ctx.moveTo(X(px), Y(py)); }
  ctx.closePath();
}
const star = (ctx, x, y) => { ctx.beginPath(); for (let i = 0; i < 10; i++) { const r = i % 2 ? 4 : 9, a = -Math.PI / 2 + i * Math.PI / 5; ctx.lineTo(x + r * Math.cos(a), y + r * Math.sin(a)); } ctx.closePath(); ctx.fill(); };

/* ============================================================ hero: scoring candidates */
const hero = { pool: null, seed: 0 };
function setPool(cands, scores, label) {
  const motion = cands.map(u => std(predict(FAM.mean, u)));
  let ib = 0, iw = 0; scores.forEach((s, i) => { if (s > scores[ib]) ib = i; if (s < scores[iw]) iw = i; });
  let mb = 0; motion.forEach((m, i) => { if (m > motion[mb]) mb = i; });
  const mx = mean(motion), sx = mean(scores), cov = mean(motion.map((m, i) => (m - mx) * (scores[i] - sx))), corr = cov / (std(motion) * std(scores));
  hero.pool = { cands, scores, motion, ib, iw, mb, corr, label };
}
function scorePool(cands) { return cands.map(u => designScore(FAM, u)); }
function drawHero() {
  if (!hero.pool) setPool(D.candidates, scorePool(D.candidates), "notebook");
  const P = hero.pool, narrow = narrowOf($("#heroH"), 600);
  const lo = Math.floor(Math.min(...P.scores)), hi = Math.ceil(Math.max(...P.scores));
  const fr = frame($("#heroH"), narrow ? 1.5 : 1.3, { x: [lo, hi], y: [0, 1], xlabel: "expected log det information", yticks: false, pad: { l: 10 } });
  const counts = new Float64Array(30), w = (hi - lo) / 30; P.scores.forEach(s => { counts[Math.min(29, Math.floor((s - lo) / w))]++; });
  histogram(fr, P.scores, lo, hi, 30, 0.92 / Math.max(...counts), alpha("--muted", 0.6));
  fr.ctx.lineWidth = 2; fr.ctx.strokeStyle = css("--c4"); seg(fr.ctx, fr.X(P.scores[P.ib]), fr.pad.t, fr.X(P.scores[P.ib]), fr.Y(0)); fr.ctx.strokeStyle = css("--c5"); seg(fr.ctx, fr.X(P.scores[P.iw]), fr.pad.t, fr.X(P.scores[P.iw]), fr.Y(0));
  const fr2 = frame($("#heroS"), narrow ? 1.5 : 1.3, { x: [0, Math.max(...P.motion) * 1.08], y: [lo, hi], xlabel: "std of predicted position", ylabel: "score" });
  fr2.ctx.fillStyle = alpha("--muted", 0.55); P.motion.forEach((m, i) => { fr2.ctx.beginPath(); fr2.ctx.arc(fr2.X(m), fr2.Y(P.scores[i]), 2.4, 0, 2 * Math.PI); fr2.ctx.fill(); });
  [[P.ib, "--c4"], [P.iw, "--c5"], [P.mb, "--warm"]].forEach(([i, c]) => { fr2.ctx.fillStyle = css(c); fr2.ctx.beginPath(); fr2.ctx.arc(fr2.X(P.motion[i]), fr2.Y(P.scores[i]), 5, 0, 2 * Math.PI); fr2.ctx.fill(); });
  const t = timeAxis(T40 + 1), qb = predict(FAM.mean, P.cands[P.ib]), qw = predict(FAM.mean, P.cands[P.iw]);
  const fr3 = frame($("#heroR"), narrow ? 1.5 : 1.3, { x: [0, 4], y: [-0.5, 0.5], xlabel: "time [s]", ylabel: "mean system's response (thin: input ÷ 3)", xticks: [0, 1, 2, 3, 4], yticks: [-0.4, -0.2, 0, 0.2, 0.4] });
  plotLine(fr3, t.slice(0, T40), P.cands[P.ib].map(v => 0.3 * v), { color: alpha("--c4", 0.5), width: 1 }); plotLine(fr3, t.slice(0, T40), P.cands[P.iw].map(v => 0.3 * v), { color: alpha("--c5", 0.5), width: 1 });
  plotLine(fr3, t, qb, { color: css("--c4"), width: 2 }); plotLine(fr3, t, qw, { color: css("--c5"), width: 2 });
  $("#rBest").textContent = fmt(P.scores[P.ib], 3) + (P.label === "notebook" ? " (SciPy " + fmt(D.scores[D.best], 3) + ")" : ""); $("#rWorst").textContent = fmt(P.scores[P.iw], 3) + (P.label === "notebook" ? " (SciPy " + fmt(D.scores[D.worst], 3) + ")" : "");
  $("#rCorr").textContent = fmt(P.corr, 3); $("#rFreq").textContent = `${fmt(peakFreq(P.cands[P.ib]), 3)} / ${fmt(peakFreq(P.cands[P.iw]), 3)} Hz`;
  const Fb = FAM.z.map(z => eig2(fisher(FAM, P.cands[P.ib], z)).values), Fw = FAM.z.map(z => eig2(fisher(FAM, P.cands[P.iw], z)).values);
  const mb = [mean(Fb.map(v => v[0])), mean(Fb.map(v => v[1]))], mw = [mean(Fw.map(v => v[0])), mean(Fw.map(v => v[1]))];
  caption($("#heroStatus"), `${P.label === "notebook" ? "The notebook's 300 candidate experiments" : "300 fresh candidates drawn by the page"}, 4 s three-sine inputs, scored by the expected log-determinant of the Fisher information over the 20 training contexts${P.label === "notebook" ? ", recomputed here with exact sensitivities (SciPy's finite differences agree to the ninth decimal)" : ""}. Left: the scores, from ${fmt(Math.min(...P.scores), 2)} to ${fmt(Math.max(...P.scores), 2)}, best (green) and worst (purple) marked. Middle: score against how much the mean system moves under each input; correlation ${fmt(P.corr, 2)}${P.label === "notebook" ? ", the notebook's 0.89" : ""}. The orange point is the candidate that moves the system most: it is not the best, it ranks ${P.scores.filter(s => s > P.scores[P.mb]).length + 1} of 300 by score. Right: the best input swings the mean system through two-sided motions of standard deviation ${fmt(P.motion[P.ib], 3)} with its energy near ${fmt(peakFreq(P.cands[P.ib]), 2)} Hz, inside the family's resonance band; the worst shakes it at ${fmt(peakFreq(P.cands[P.iw]), 2)} Hz, which the mass filters out (standard deviation ${fmt(P.motion[P.iw], 3)}). In Fisher terms the best input is worth ${fmt(mb[0], 0)} and ${fmt(mb[1], 0)} units of prior along the two posterior axes, the worst ${fmt(mw[0], 0)} and ${fmt(mw[1], 1)}: the worst experiment leaves one direction of the context almost entirely to the prior.`);
}
$("#hNew").addEventListener("click", () => { hero.seed++; const rng = makeRng(hero.seed * 4241 + 3); const cands = Array.from({ length: 300 }, () => multisine(T40, rng)); setPool(cands, scorePool(cands), "fresh"); drawHero(); });
$("#hNb").addEventListener("click", () => { setPool(D.candidates, scorePool(D.candidates), "notebook"); drawHero(); });

/* ============================================================ single-sine sweep */
const sw = { f: 0.17 };
function drawSweep() {
  const freqs = Array.from({ length: 60 }, (_, i) => Math.exp(Math.log(0.03) + (Math.log(0.8) - Math.log(0.03)) * i / 59));
  if (!sw.curve) sw.curve = freqs.map(f => designScore(FAM, Array.from({ length: T40 }, (_, t) => Math.sin(2 * Math.PI * f * DT * t))));
  const u = Array.from({ length: T40 }, (_, t) => Math.sin(2 * Math.PI * sw.f * DT * t)), s = designScore(FAM, u);
  const cv = $("#sweepC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.4 : 2.4, { x: [Math.log10(0.03), Math.log10(0.8)], y: [4.5, 10.5], xlabel: "sine frequency [Hz] (log)", ylabel: "score", xticks: [-1.5, -1, -0.5], xfmt: v => (10 ** v).toFixed(2), yticks: [5, 6, 7, 8, 9, 10] });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--post", 0.12); ctx.fillRect(X(Math.log10(0.116)), fr.pad.t, X(Math.log10(0.218)) - X(Math.log10(0.116)), fr.ih);
  ctx.strokeStyle = css("--c4"); ctx.setLineDash([4, 3]); seg(ctx, X(Math.log10(0.03)), Y(hero.pool ? hero.pool.scores[hero.pool.ib] : D.scores[D.best]), X(Math.log10(0.8)), Y(hero.pool ? hero.pool.scores[hero.pool.ib] : D.scores[D.best])); ctx.setLineDash([]);
  plotLine(fr, freqs.map(Math.log10), sw.curve, { color: css("--warm"), width: 2 });
  ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(Math.log10(sw.f)), Y(s), 5, 0, 2 * Math.PI); ctx.fill();
  const imax = sw.curve.indexOf(Math.max(...sw.curve));
  caption($("#sweepCap"), `A single 4 s sine of frequency [[f = ${fmt(sw.f, 3)}]] Hz scores ${fmt(s, 2)}. Over the sweep, pure sines score between ${fmt(Math.min(...sw.curve), 2)} and ${fmt(Math.max(...sw.curve), 2)}, with the maximum at ${fmt(freqs[imax], 2)} Hz, just above the family's resonance band (shaded, 0.116 to 0.218 Hz), and every one of them is far below the best three-sine candidate (dashed, ${fmt(hero.pool ? hero.pool.scores[hero.pool.ib] : D.scores[D.best], 2)}). A sine at resonance moves the cart most, but one frequency alone cannot separate stiffness from damping and gain; the designed multisines win because they straddle the band and excite several frequencies at once. "Informative" is not the same as "large".`);
}
$("#swF").addEventListener("input", e => { sw.f = +e.target.value; $("#swFV").textContent = fmt(sw.f, 3) + " Hz"; drawSweep(); });
$("#swFV").textContent = fmt(sw.f, 3) + " Hz";

/* ============================================================ section 2: three experiments */
const s2 = { extra: null, seed: 0 };
function drawSec2() {
  const zTrue = project(FAM, D.nu.eta), exps = [["worst", "--c5"], ["random", "--post"], ["designed", "--c4"]];
  const res = exps.map(([k, c]) => { const e = D.sec2[k]; const ad = adapt(FAM, e.u, e.y); return { k, c, ad, py: e }; });
  if (s2.extra) res.push(s2.extra);
  const cv = $("#s2C"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.05 : 1.3, { x: [-2.6, 2.6], y: [-3.2, 2.6], xlabel: "z₁", ylabel: "z₂", xticks: [-2, -1, 0, 1, 2], yticks: [-3, -2, -1, 0, 1, 2] });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--muted", 0.45); FAM.z.forEach(z => { ctx.beginPath(); ctx.arc(X(z[0]), Y(z[1]), 2.5, 0, 2 * Math.PI); ctx.fill(); });
  res.forEach(r => { ctx.strokeStyle = css(r.c); ctx.lineWidth = 2; ellipsePath(ctx, X, Y, r.ad.z, r.ad.cov, 2); ctx.stroke(); ctx.fillStyle = css(r.c); ctx.beginPath(); ctx.arc(X(r.ad.z[0]), Y(r.ad.z[1]), 4.5, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = css("--ink"); star(ctx, X(zTrue[0]), Y(zTrue[1]));
  const det = c => c[0][0] * c[1][1] - c[0][1] * c[1][0];
  const mah = r => { const d = [zTrue[0] - r.ad.z[0], zTrue[1] - r.ad.z[1]], P = inv2(r.ad.cov); return Math.sqrt(d[0] * (P[0][0] * d[0] + P[0][1] * d[1]) + d[1] * (P[1][0] * d[0] + P[1][1] * d[1])); };
  $("#s2T tbody").innerHTML = res.map(r => `<tr><td class="words">${r.k}</td><td>(${r.ad.z.map(v => fmt(v, 3)).join(", ")})</td><td>${det(r.ad.cov).toExponential(2)}</td><td>${fmt(mah(r), 2)}</td><td>${fmt(r.ad.misfit, 4)}</td></tr>`).join("");
  caption($("#s2Cap"), `The notebook's new system, [[z_{\\text{true}} = (${fmt(zTrue[0], 2)}, ${fmt(zTrue[1], 2)})]] (star), identified from three different 4 s experiments: the worst candidate (purple), a random one (blue) and the designed one (green), each adapted live from the notebook's recorded measurements (SciPy's estimates agree to five decimals). The posterior determinants fall from ${det(res[0].ad.cov).toExponential(1)} to ${det(res[2].ad.cov).toExponential(1)}, a factor of ${fmt(det(res[0].ad.cov) / det(res[2].ad.cov), 0)}: the designed experiment shrinks the uncertainty region by two and a half orders of magnitude. But the smallest ellipse does not contain the true system: the truth lies ${fmt(mah(res[2]), 2)} posterior standard deviations from the designed estimate, ${fmt(mah(res[1]), 2)} from the random one and ${fmt(mah(res[0]), 2)} from the worst, which is the only one honest about its ignorance. The Laplace posterior assumes white noise and a correct model, while the real disturbance is coloured and the model lacks the quintic term and the bias; a small ellipse is a promise, not a guarantee.` + (s2.extra ? ` The orange ellipse is a fresh random 4 s experiment on the same system, simulated and adapted by the page.` : ""));
}
$("#s2New").addEventListener("click", () => { s2.seed++; const rng = makeRng(s2.seed * 911 + 5); const u = multisine(T40, rng), y = simulate(D.nu, u, rng.normal).y; s2.extra = { k: "fresh random", c: "--warm", ad: adapt(FAM, u, y) }; drawSec2(); });

/* ============================================================ section 3: the study */
const st = { i: 1, own: null, running: false, seed: 0 };
function fillStudy() {
  $("#stT tbody").innerHTML = D.budgets.map((T, i) => { const s = D.stats[i]; return `<tr><td>${fmt(DT * T, 1)} s</td><td>${fmt(s.random, 4)}</td><td>${fmt(s.designed, 4)}</td><td>${fmt(s.worst, 4)}</td><td>${fmt(100 * s.reduction, 1)}%</td><td>${fmt(100 * s.win, 0)}%</td><td>[${s.ci[0] >= 0 ? "+" : ""}${fmt(s.ci[0], 4)}, +${fmt(s.ci[1], 4)}]</td><td>${fmt(100 * s.designed_worse_than_worst, 0)}%</td></tr>`; }).join("");
}
function drawStudy() {
  const cv = $("#stC"), narrow = narrowOf(cv);
  const sec = D.budgets.map(b => DT * b), lg = Math.log10;
  const fr = frame(cv, narrow ? 1.3 : 2.2, { x: [lg(1.2), lg(15)], y: [0, 0.05], xlabel: "length of the experiment [s]", ylabel: "mean prediction RMSE", xticks: sec.map(lg), xfmt: v => (10 ** v).toFixed(10 ** v < 10 ? 1 : 0), yticks: [0, 0.01, 0.02, 0.03, 0.04, 0.05] });
  const { ctx, X, Y } = fr;
  [["worst", "--c5"], ["context", "--post"], ["designed", "--c4"]].forEach(([m, c]) => { const ys = D.err[m].map(mean); plotLine(fr, sec.map(lg), ys, { color: css(c), width: 2 }); ys.forEach((v, i) => { ctx.fillStyle = css(c); ctx.beginPath(); ctx.arc(X(lg(sec[i])), Y(v), 3.5, 0, 2 * Math.PI); ctx.fill(); }); });
  if (st.own) { const o = st.own; [["worst", "--c5"], ["context", "--post"], ["designed", "--c4"]].forEach(([m, c]) => { if (!o.rows.length) return; const v = mean(o.rows.map(r => r[m])); ctx.strokeStyle = css(c); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(lg(DT * o.T)), Y(v), 6, 0, 2 * Math.PI); ctx.stroke(); }); }
  let text = `The notebook's study: 60 new systems per budget; for each, a pool of 200 candidates is scored, the designed (green) and the worst (purple) experiment are picked, and a random experiment (blue) is drawn without looking at its score. Mean validation RMSE against budget. At 1.5 s the designed and random experiments are indistinguishable (${fmt(100 * D.stats[0].reduction, 1)}% reduction, ${fmt(100 * D.stats[0].win, 0)}% wins, interval containing zero); from 3 s on, design cuts the mean error by ${fmt(100 * D.stats[1].reduction, 0)}%, ${fmt(100 * D.stats[2].reduction, 0)}% and ${fmt(100 * D.stats[3].reduction, 0)}% and wins about three paired trials in four. `;
  if (st.own) { const o = st.own; if (o.rows.length) { const r = mean(o.rows.map(x => x.context)), d = mean(o.rows.map(x => x.designed)), w = o.rows.filter(x => x.designed < x.context).length; text += `Page's rerun at ${fmt(DT * o.T, 1)} s with ${o.rows.length}${o.rows.length < o.n ? " of " + o.n : ""} trials and 100-candidate pools (hollow circles): random ${fmt(r, 4)}, designed ${fmt(d, 4)}, reduction ${fmt(100 * (r - d) / r, 0)}%, designed wins ${w} of ${o.rows.length}.`; } else text += "Computing the page's rerun…"; }
  else text += "Rerun a budget with the page's own draws to see how much these figures move.";
  caption($("#stCap"), text);
}
function runStudy() {
  if (st.running) return;
  st.running = true; st.seed++; st.own = { T: D.budgets[st.i], n: 20, rows: [] };
  const rng = makeRng(st.seed * 2719 + st.i), o = st.own;
  const step = () => { if (o.rows.length >= o.n) { st.running = false; $("#stStatus").textContent = ""; drawStudy(); return; } o.rows.push(designTrial(FAM, o.T, rng, 100)); $("#stStatus").textContent = `trial ${o.rows.length} of ${o.n}`; drawStudy(); setTimeout(step, 0); };
  setTimeout(step, 0);
}
$$("#stSeg .btn").forEach(b => b.addEventListener("click", () => { $$("#stSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); st.i = +b.dataset.i; }));
$("#stRun").addEventListener("click", runStudy);

function drawAll() { drawHero(); drawSweep(); drawSec2(); drawStudy(); }
fillStudy();
drawAll();
onRedraw(drawAll);
})();
