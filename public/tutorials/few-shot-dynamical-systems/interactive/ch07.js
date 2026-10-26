/* Learning to Identify New Dynamical Systems, chapter 07, When the shared knowledge does not apply:
   the live demos. New systems are pushed out of the pretrained family along a direction the
   two-number context cannot express; the context adaptation (chapter 04), the from-scratch fit
   (chapter 02) and the output-misfit test run in the browser, on the notebook's study (ch07-data.js)
   and on fresh draws. Needs ../kit/kit.js. */
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
const decode = (fam, z) => fam.mean.map((m, j) => clip(m + fam.U[j][0] * z[0] + fam.U[j][1] * z[1], LOW[j], HIGH[j]));
function sensitivity(fam, z, u) {
  const raw = fam.mean.map((m, j) => m + fam.U[j][0] * z[0] + fam.U[j][1] * z[1]), live = raw.map((v, j) => v > LOW[j] && v < HIGH[j] ? 1 : 0);
  const { q, J } = predictJac(decode(fam, z), u);
  return { q, S: J.map(row => [0, 1].map(k => row.reduce((s, v, j) => s + live[j] * v * fam.U[j][k], 0))) };
}
/* dynutils.adapt: MAP context, Laplace covariance and the output misfit. */
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
  const { q } = sensitivity(fam, z, u);
  return { z, misfit: rmse(Array.from(q), y) };
}
function project(fam, eta) {
  const A = fam.U.map((row, j) => row.map(u => u / fam.scale[j])), b = eta.map((v, j) => (v - fam.mean[j]) / fam.scale[j]);
  const N = [[0, 0], [0, 0]], r = [0, 0];
  A.forEach((row, i) => { for (let p = 0; p < 2; p++) { r[p] += row[p] * b[i]; for (let q = 0; q < 2; q++) N[p][q] += row[p] * row[q]; } });
  return solve(N, r);
}
/* Unit vector orthogonal (in physical units) to both columns of U, closest to the damping axis:
   the projection of e_d onto the orthogonal complement, by Gram-Schmidt. */
function complementDirection(fam, axis = 1) {
  const cols = [0, 1].map(k => fam.U.map(r => r[k]));
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const q1 = cols[0].map(v => v / Math.sqrt(dot(cols[0], cols[0])));
  let q2 = cols[1].map((v, i) => v - dot(cols[1], q1) * q1[i]); const n2 = Math.sqrt(dot(q2, q2)); q2 = q2.map(v => v / n2);
  let e = [0, 0, 0, 0]; e[axis] = 1;
  let p = e.map((v, i) => v - dot(e, q1) * q1[i] - dot(e, q2) * q2[i]); const np_ = Math.sqrt(dot(p, p));
  return p.map(v => v / np_);
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
/* Shift a system's coefficients by delta along `dir`, clipped as in the notebook. */
const shiftSystem = (sys, dir, delta) => Object.assign({}, sys, { eta: sys.eta.map((v, j) => clip(v + delta * dir[j], LOW[j] + 0.02, HIGH[j])) });
/* One trial of the shift study: errors of the four predictors, the misfit, the whitened distance from the family's plane. */
function shiftTrial(fam, sys, u, y, uv, qv, withScratch = true) {
  const ad = adapt(fam, u, y), best = decode(fam, project(fam, sys.eta));
  const out = { context: rmse(predict(decode(fam, ad.z), uv), qv), best: rmse(predict(best, uv), qv), none: rmse(predict(fam.mean, uv), qv), misfit: ad.misfit, z: ad.z };
  if (withScratch) out.scratch = rmse(predict(fitLS(u, y).eta, uv), qv);
  const r = sys.eta.map((v, j) => (v - best[j]) / fam.scale[j]); out.dist = Math.sqrt(r.reduce((s, v) => s + v * v, 0));
  return out;
}
function drawShiftTrial(fam, dir, delta, T, rng, withScratch = true) {
  const sys = shiftSystem(sampleSystem(rng), dir, delta), u = multisine(T, rng), y = simulate(sys, u, rng.normal).y, uv = multisine(300, rng, 6), qv = simulate(sys, uv).q;
  return shiftTrial(fam, sys, u, y, uv, qv, withScratch);
}
const median = arr => { const a = [...arr].sort((x, y) => x - y), n = a.length; return n % 2 ? a[(n - 1) / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]); };
/* numpy's default (linear) quantile. */
const quantile = (arr, p) => { const a = [...arr].sort((x, y) => x - y), i = (a.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return a[lo] + (a[hi] - a[lo]) * (i - lo); };
const freqResp = (eta, om) => 0.5 * eta[2] / Math.hypot(eta[0] - om * om, eta[1] * om);
/* NET-END */

if (typeof module !== "undefined" && module.exports) {
  module.exports = { simulate, predict, fitLS, adapt, decode, project, complementDirection, makeRng, multisine, sampleSystem, shiftSystem, shiftTrial, drawShiftTrial, rmse, median, quantile };
  return;
}

const D = window.FS07, FAM = D.family, SH = D.shifts, T = D.T;
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const NAMES = ["k", "d", "b", "α"];

/* ============================================================ hero: the shift study */
const hero = { show: { scratch: true, context: true, best: false, none: false }, n: 30, own: null, running: false, seed: 0 };
function runOwn() {
  if (hero.running) return;
  hero.running = true; hero.seed++;
  hero.own = { rows: SH.map(() => []), thr: null, cal: [] };
  const rng = makeRng(hero.seed * 7877 + 13), o = hero.own;
  const step = (phase, i, n) => {
    if (phase === 0) { // calibration: 60 in-family misfits
      const t0 = performance.now();
      while (o.cal.length < 60 && performance.now() - t0 < 40) { const sys = sampleSystem(rng), u = multisine(T, rng), y = simulate(sys, u, rng.normal).y; o.cal.push(adapt(FAM, u, y).misfit); }
      $("#hStatus").textContent = `calibrating the threshold: ${o.cal.length} of 60 in-family systems`;
      if (o.cal.length >= 60) { o.thr = quantile(o.cal, 0.95); setTimeout(() => step(1, 0, 0), 0); } else setTimeout(() => step(0, 0, 0), 0);
      return;
    }
    if (i >= SH.length) { hero.running = false; $("#hStatus").textContent = ""; drawHero(); return; }
    const t0 = performance.now();
    while (o.rows[i].length < hero.n && performance.now() - t0 < 40) o.rows[i].push(drawShiftTrial(FAM, D.direction, SH[i], T, rng));
    $("#hStatus").textContent = `shift ${fmt(SH[i], 1)}: system ${o.rows[i].length} of ${hero.n}`;
    drawHero();
    if (o.rows[i].length >= hero.n) setTimeout(() => step(1, i + 1, 0), 0); else setTimeout(() => step(1, i, 0), 0);
  };
  setTimeout(() => step(0, 0, 0), 0);
}
function drawHero() {
  const cv = $("#heroE"), narrow = narrowOf(cv, 600);
  const COL = { scratch: "--warm", context: "--post", best: "--c4", none: "--muted" };
  const fr = frame(cv, narrow ? 1.3 : 1.5, { x: [-0.03, 0.63], y: [0, 0.03], xlabel: "distance δ from the learned family", ylabel: "median prediction RMSE", xticks: SH, yticks: [0, 0.01, 0.02, 0.03] });
  const { ctx, X, Y } = fr;
  Object.keys(COL).filter(m => hero.show[m]).forEach(m => { const ys = D.err[m].map(median); plotLine(fr, SH, ys.map(v => Math.min(v, 0.03)), { color: css(COL[m]), width: 2 }); ys.forEach((v, i) => { ctx.fillStyle = css(COL[m]); ctx.beginPath(); ctx.arc(X(SH[i]), Y(Math.min(v, 0.03)), 3.5, 0, 2 * Math.PI); ctx.fill(); }); });
  if (hero.own) { const o = hero.own; Object.keys(COL).filter(m => hero.show[m]).forEach(m => { const pts = o.rows.map((r, i) => r.length ? [SH[i], median(r.map(x => x[m]))] : null).filter(Boolean); if (pts.length > 1) plotLine(fr, pts.map(p => p[0]), pts.map(p => Math.min(p[1], 0.03)), { color: css(COL[m]), width: 1.4, dash: [2, 3] }); pts.forEach(p => { ctx.strokeStyle = css(COL[m]); ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(X(p[0]), Y(Math.min(p[1], 0.03)), 4.5, 0, 2 * Math.PI); ctx.stroke(); }); }); }
  const cf = $("#heroF"), frF = frame(cf, narrow ? 1.3 : 1.5, { x: [-0.03, 0.63], y: [-3, 103], xlabel: "distance δ from the learned family", ylabel: "systems flagged [%]", xticks: SH, yticks: [0, 25, 50, 75, 100] });
  frF.ctx.strokeStyle = css("--muted"); frF.ctx.setLineDash([3, 3]); seg(frF.ctx, frF.X(-0.03), frF.Y(5), frF.X(0.63), frF.Y(5)); frF.ctx.setLineDash([]);
  const fl = D.flagged.map(r => 100 * mean(r)); plotLine(frF, SH, fl, { color: css("--ink"), width: 2 }); fl.forEach((v, i) => { frF.ctx.fillStyle = css("--ink"); frF.ctx.beginPath(); frF.ctx.arc(frF.X(SH[i]), frF.Y(v), 3.5, 0, 2 * Math.PI); frF.ctx.fill(); });
  if (hero.own && hero.own.thr !== null) { const o = hero.own, pts = o.rows.map((r, i) => r.length ? [SH[i], 100 * r.filter(x => x.misfit > o.thr).length / r.length] : null).filter(Boolean); if (pts.length > 1) plotLine(frF, pts.map(p => p[0]), pts.map(p => p[1]), { color: css("--ink"), width: 1.4, dash: [2, 3] }); pts.forEach(p => { frF.ctx.strokeStyle = css("--ink"); frF.ctx.lineWidth = 1.4; frF.ctx.beginPath(); frF.ctx.arc(frF.X(p[0]), frF.Y(p[1]), 4.5, 0, 2 * Math.PI); frF.ctx.stroke(); }); }
  $("#rThr").textContent = fmt(D.threshold, 4) + (hero.own && hero.own.thr !== null ? ` (page: ${fmt(hero.own.thr, 4)})` : "");
  $("#rFlag").textContent = fl.map(v => fmt(v, 0) + "%").join(", ");
  const wins = SH.map((_, i) => D.err.context[i].filter((v, n) => v < D.err.scratch[i][n]).length);
  $("#rWins").textContent = wins.map(w => w + "/30").join(", ");
  let text = `The notebook's study: 30 new systems at each of seven distances [[\\delta]] from the family, pushed along the direction [[(${D.direction.map(v => fmt(v, 2)).join(", ")})]] in [[(k, d, b, \\alpha)]], which the two-number context cannot express, and identified from 12 s of data. Left: median validation error. The pretrained context (blue) degrades steadily, from ${fmt(median(D.err.context[0]), 4)} to ${fmt(median(D.err.context[6]), 4)}, while the from-scratch fit (orange) is unaffected and even improves, from ${fmt(median(D.err.scratch[0]), 4)} to ${fmt(median(D.err.scratch[6]), 4)}, because the shifted systems are more heavily damped and move less. The context wins ${wins[0]} of 30 pairs at [[\\delta = 0]] and ${wins[3]} of 30 at [[\\delta = 0.3]]. Right: the misfit test flags ${fmt(fl[0], 0)}% of the unshifted systems against a 5% target (3 of 30: within sampling noise) and rises to ${fmt(Math.max(...fl), 0)}% at the largest shifts, not the "three quarters" of the notebook's text. `;
  if (hero.own && hero.own.rows.some(r => r.length)) text += `The page's rerun (dotted, hollow points; threshold recalibrated on 60 fresh in-family systems) shows the same shape with different numbers.`;
  else text += `Rerun the study with the page's own draws to see how much thirty systems per point move these curves.`;
  caption($("#heroStatus"), text);
}
$$("#heroM .btn").forEach(b => b.addEventListener("click", () => { hero.show[b.dataset.m] = !hero.show[b.dataset.m]; b.setAttribute("aria-pressed", hero.show[b.dataset.m]); drawHero(); }));
$("#hN").addEventListener("input", e => { hero.n = +e.target.value; $("#hNV").textContent = hero.n; });
$("#hRun").addEventListener("click", runOwn);
$("#hNV").textContent = hero.n;

/* ============================================================ direction explorer */
const dx = { delta: 0.3, dir: "nb" };
const omegas = Array.from({ length: 300 }, (_, i) => 0.05 + (3 - 0.05) * i / 299);
function drawDir() {
  const dir = dx.dir === "nb" ? D.direction : dx.dir === "proj" ? D.projDir : D.inside;
  const base = { eta: FAM.mean.slice() }, sys = shiftSystem(base, dir, dx.delta), best = decode(FAM, project(FAM, sys.eta));
  const resid = sys.eta.map((v, j) => (v - best[j]) / FAM.scale[j]), dist = Math.sqrt(resid.reduce((s, v) => s + v * v, 0));
  const cb = $("#dirB"), narrow = narrowOf(cb, 600);
  const frB = frame(cb, narrow ? 1.6 : 1.5, { x: [-0.5, 3.5], y: [0, 2], xticks: false, yticks: [0, 0.5, 1, 1.5, 2], ylabel: "coefficient" });
  const { ctx, X, Y } = frB;
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let j = 0; j < 4; j++) {
    ctx.fillStyle = alpha("--muted", 0.45); ctx.fillRect(X(j - 0.36), Y(Math.min(2, FAM.mean[j])), X(j - 0.12) - X(j - 0.36), Y(0) - Y(Math.min(2, FAM.mean[j])));
    ctx.fillStyle = alpha("--warm", 0.85); ctx.fillRect(X(j - 0.12), Y(Math.min(2, sys.eta[j])), X(j + 0.12) - X(j - 0.12), Y(0) - Y(Math.min(2, sys.eta[j])));
    ctx.fillStyle = alpha("--post", 0.85); ctx.fillRect(X(j + 0.12), Y(Math.min(2, best[j])), X(j + 0.36) - X(j + 0.12), Y(0) - Y(Math.min(2, best[j])));
    ctx.fillStyle = css("--ink-2"); ctx.fillText(NAMES[j], X(j), Y(0) + 5);
  }
  const cf = $("#dirF"), frF = frame(cf, narrow ? 1.6 : 1.5, { x: [0, 3 / (2 * Math.PI)], y: [0, 3], xlabel: "frequency [Hz]", ylabel: "|q / u|", xticks: [0, 0.1, 0.2, 0.3, 0.4] });
  const fh = o => o / (2 * Math.PI);
  plotLine(frF, omegas.map(fh), omegas.map(o => freqResp(FAM.mean, o)), { color: css("--muted"), width: 1.2, dash: [4, 3] });
  plotLine(frF, omegas.map(fh), omegas.map(o => freqResp(sys.eta, o)), { color: css("--warm"), width: 2.2 });
  plotLine(frF, omegas.map(fh), omegas.map(o => freqResp(best, o)), { color: css("--post"), width: 1.8 });
  const nat = e => ({ f: Math.sqrt(e[0]) / (2 * Math.PI), zeta: e[1] / (2 * Math.sqrt(e[0])) }), nS = nat(sys.eta), nB = nat(best);
  caption($("#dirCap"), `The mean system (grey) pushed by [[\\delta = ${fmt(dx.delta, 2)}]] along ${dx.dir === "nb" ? "the notebook's direction" : dx.dir === "proj" ? "the damping axis projected onto the complement" : "the family's first direction (exercise 1)"} [[(${dir.map(v => fmt(v, 2)).join(", ")})]] becomes the orange system [[\\eta = (${sys.eta.map(v => fmt(v, 3)).join(", ")})]], natural frequency ${fmt(nS.f, 3)} Hz and damping ratio ${fmt(nS.zeta, 3)}. The closest member of the family (blue, the projection onto the plane) is [[(${best.map(v => fmt(v, 3)).join(", ")})]], ${fmt(nB.f, 3)} Hz and ${fmt(nB.zeta, 3)}; what it cannot represent is a residual of ${fmt(dist, 1)} whitened units (coefficient differences divided by their estimation standard deviations: ${resid.map((v, j) => `${NAMES[j]} ${fmt(v, 1)}`).join(", ")}). ` + (dx.dir === "inside" ? "Along a direction inside the family the residual is zero whatever the shift: the context simply moves to meet it." : dx.dir === "nb" ? "The residual is almost entirely damping: in whitened units the shift moves [[d]] by 182 per unit of [[\\delta]] and the other coefficients by about 30, and the family's plane runs nearly parallel to the damping axis. The frequency responses show the consequence: the shifted system's resonance is flatter than anything the family can produce at that frequency." : "This is the basis-free choice of a direction orthogonal to the family that moves damping most; the notebook's vector is one element of an SVD basis of the same two-dimensional complement, 25° away from this one. Both leave the family; the study's conclusions do not depend on which is used."));
}
$("#dxD").addEventListener("input", e => { dx.delta = +e.target.value; $("#dxDV").textContent = fmt(dx.delta, 2); drawDir(); });
$$("#dxSeg .btn").forEach(b => b.addEventListener("click", () => { $$("#dxSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); dx.dir = b.dataset.d; drawDir(); }));
$("#dxDV").textContent = fmt(dx.delta, 2);

/* ============================================================ misfit distributions and the threshold */
const th = { p: 0.95 };
function drawMisfit() {
  const thr = quantile(D.inFamily, th.p);
  const cv = $("#misC"), narrow = narrowOf(cv);
  const lo = 0.008, hi = 0.03;
  const fr = frame(cv, narrow ? 1.2 : 2.0, { x: [-1.5, 6.5], y: [lo, hi], xticks: [-1, 0, 1, 2, 3, 4, 5, 6], xfmt: v => v < 0 ? "cal." : fmt(SH[v], 1), xlabel: "distance δ (cal. = the 60 in-family calibration systems)", ylabel: "output misfit of the adapted model", yticks: [0.01, 0.015, 0.02, 0.025, 0.03] });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]); seg(ctx, X(-1.5), Y(thr), X(6.5), Y(thr)); ctx.setLineDash([]);
  ctx.strokeStyle = alpha("--muted", 0.7); ctx.setLineDash([2, 3]); seg(ctx, X(-1.5), Y(FAM.sigma), X(6.5), Y(FAM.sigma)); ctx.setLineDash([]);
  const dots = (xs, vals, col) => vals.forEach((v, n) => { const jit = ((n * 37) % 11 - 5) / 18; ctx.fillStyle = v > thr ? css("--warm") : alpha(col, 0.55); ctx.beginPath(); ctx.arc(X(xs + jit), Y(clip(v, lo, hi)), 3, 0, 2 * Math.PI); ctx.fill(); });
  dots(-1, D.inFamily, "--muted"); SH.forEach((_, i) => dots(i, D.misfits[i], "--post"));
  SH.forEach((_, i) => { const m = median(D.misfits[i]); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; seg(ctx, X(i - 0.35), Y(m), X(i + 0.35), Y(m)); });
  const fa = 100 * D.inFamily.filter(v => v > thr).length / 60, rates = SH.map((_, i) => 100 * D.misfits[i].filter(v => v > thr).length / 30);
  $("#misT tbody").innerHTML = `<tr><td>${fmt(100 * th.p, 1)}th</td><td>${fmt(thr, 4)}</td><td>${fmt(fa, 1)}%</td>${rates.map(r => `<td>${fmt(r, 0)}%</td>`).join("")}</tr>`;
  caption($("#misCap"), `Every system's output misfit, the residual RMS of the adapted model on its own 12 s record: the 60 calibration systems on the left, then the 30 systems at each shift (medians as bars; dots above the threshold in orange). The threshold is the ${fmt(100 * th.p, 1)}th percentile of the calibration misfits, ${fmt(thr, 4)}; the family's residual level [[\\hat\\sigma = ${fmt(FAM.sigma, 4)}]] is dotted. Misfits are dominated by the coloured disturbance, which is why the in-family cloud sits between 0.010 and 0.023 and why the shifted systems' clouds overlap it so much: at [[\\delta = 0.3]] the median misfit is ${fmt(median(D.misfits[3]), 4)} against ${fmt(median(D.misfits[0]), 4)} unshifted, a difference smaller than the spread. Detection lags degradation: at that shift the adapter is already worse than the from-scratch fit on ${30 - D.err.context[3].filter((v, n) => v < D.err.scratch[3][n]).length} systems of 30, while only ${fmt(rates[3], 0)}% are flagged. Move the percentile to trade false alarms against detection; the notebook's exercise 2 does this at the 99th.`);
}
$("#thP").addEventListener("input", e => { th.p = +e.target.value; $("#thPV").textContent = fmt(100 * th.p, 1) + "th"; drawMisfit(); });
$("#thPV").textContent = fmt(100 * th.p, 1) + "th";

/* ============================================================ exercise 1: inside against outside */
const ex1 = { delta: 0.4, seed: 1, cache: {} };
function drawEx1() {
  const key = `${ex1.delta}-${ex1.seed}`;
  if (!ex1.cache[key]) {
    const rng = makeRng(ex1.seed * 5801 + Math.round(100 * ex1.delta)), rows = { inside: [], outside: [] };
    for (let n = 0; n < 30; n++) { rows.inside.push(drawShiftTrial(FAM, D.inside, ex1.delta, T, rng, false)); rows.outside.push(drawShiftTrial(FAM, D.direction, ex1.delta, T, rng, false)); }
    ex1.cache[key] = rows;
  }
  const rows = ex1.cache[key], thr = D.threshold;
  const cv = $("#ex1C"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, 0.035], y: [-0.6, 1.6], xlabel: "validation RMSE of the adapted context", yticks: false, xticks: [0, 0.01, 0.02, 0.03], pad: { l: 12 } });
  const { ctx, X, Y } = fr;
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  [["inside", "--c4", 1], ["outside", "--warm", 0]].forEach(([k, c, yy]) => { rows[k].forEach((r, n) => { const jit = ((n * 37) % 11 - 5) / 22; ctx.fillStyle = r.misfit > thr ? css("--ink") : alpha(c, 0.6); ctx.beginPath(); ctx.arc(X(Math.min(0.035, r.context)), Y(yy + jit), 3.2, 0, 2 * Math.PI); ctx.fill(); }); const m = median(rows[k].map(r => r.context)); ctx.strokeStyle = css(c); ctx.lineWidth = 2.5; seg(ctx, X(m), Y(yy - 0.35), X(m), Y(yy + 0.35)); ctx.fillStyle = css("--ink-2"); ctx.fillText(k === "inside" ? "shift inside the family" : "shift outside the family", fr.pad.l + 4, Y(yy + 0.38)); });
  const mi = median(rows.inside.map(r => r.context)), mo = median(rows.outside.map(r => r.context)), fi = rows.inside.filter(r => r.misfit > thr).length, fo = rows.outside.filter(r => r.misfit > thr).length;
  const di = median(rows.inside.map(r => r.dist)), dO = median(rows.outside.map(r => r.dist));
  caption($("#ex1Cap"), `Thirty new systems shifted by [[\\delta = ${fmt(ex1.delta, 1)}]] along the family's first direction (green) and thirty along the notebook's orthogonal direction (orange), each adapted from 12 s; black dots are flagged by the notebook's threshold. Inside the family: median error ${fmt(mi, 4)}, ${fi} of 30 flagged, median whitened distance from the plane ${fmt(di, 1)}. Outside: ${fmt(mo, 4)}, ${fo} of 30 flagged, distance ${fmt(dO, 1)}. ${ex1.delta === 0.4 ? `The notebook's run at 0.4 gives 0.0104 inside (2 of 30 flagged) against 0.0129 outside. ` : ""}Moving along a direction the context can represent leaves both the error and the flag rate near the in-family baseline, because the adapter simply finds a [[z]] that reaches that part of coefficient space; it is specifically the unrepresentable component of a shift that hurts, and that the misfit sees, late.`);
}
$("#e1D").addEventListener("input", e => { ex1.delta = +e.target.value; $("#e1DV").textContent = fmt(ex1.delta, 1); drawEx1(); });
$("#e1New").addEventListener("click", () => { ex1.seed++; drawEx1(); });
$("#e1DV").textContent = fmt(ex1.delta, 1);

function drawAll() { drawHero(); drawDir(); drawMisfit(); drawEx1(); }
drawAll();
onRedraw(drawAll);
})();
