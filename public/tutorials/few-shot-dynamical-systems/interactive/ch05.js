/* Learning to Identify New Dynamical Systems, chapter 05, When is pretraining worth it: the live
   demos. The notebook's budget study (five experiment lengths, forty new systems each, four ways of
   predicting a fresh 30 s input) is shown from its own run (ch05-data.js) and rerun here with the
   page's random draws: new systems, experiments, from-scratch fits (chapter 02's solver) and
   context adaptations (chapter 04's) all computed in the browser. Needs ../kit/kit.js. */
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
/* A family of rank r: U is 4 x r. */
const rankOf = fam => fam.U[0].length;
const decode = (fam, z) => fam.mean.map((m, j) => clip(m + fam.U[j].reduce((s, u, k) => s + u * z[k], 0), LOW[j], HIGH[j]));
function adapt(fam, u, y, t = 1) {
  const r = rankOf(fam), sig = fam.sigma, s2 = sig * sig;
  let z = new Array(r).fill(0);
  /* Chain rule through the decoder; a coefficient pinned by the clip has zero derivative with respect to z. */
  const sens = zz => { const raw = fam.mean.map((m, j) => m + fam.U[j].reduce((s, uu, k) => s + uu * zz[k], 0)), live = raw.map((v, j) => v > LOW[j] && v < HIGH[j] ? 1 : 0); const { q, J } = predictJac(decode(fam, zz), u); const S = J.map(row => Array.from({ length: r }, (_, k) => row.reduce((s, v, j) => s + live[j] * v * fam.U[j][k], 0))); return { q, S }; };
  const cost = zz => { const q = predictJac(decode(fam, zz), u).q; let s = 0; for (let i = 0; i < y.length; i++) s += (q[i] - y[i]) ** 2; return s / s2 + t * t * zz.reduce((a, v) => a + v * v, 0); };
  let f = cost(z), lam = 1e-3;
  for (let it = 0; it < 2000; it++) {
    const { q, S } = sens(z);
    const H = Array.from({ length: r }, (_, a) => Array.from({ length: r }, (_, b) => a === b ? t * t : 0)), g = z.map(v => t * t * v);
    for (let i = 0; i < y.length; i++) { const res = (q[i] - y[i]) / s2; for (let a = 0; a < r; a++) { g[a] += S[i][a] * res; for (let b = 0; b < r; b++) H[a][b] += S[i][a] * S[i][b] / s2; } }
    if (Math.sqrt(g.reduce((s, v) => s + v * v, 0)) < 1e-12) break;
    let accepted = false;
    for (let tries = 0; tries < 40 && !accepted; tries++) {
      const A = H.map((row, a) => row.map((v, b) => a === b ? v * (1 + lam) : v));
      const step = solve(A, g.map(v => -v)); if (!step) { lam *= 10; continue; }
      const zn = z.map((v, k) => v + step[k]), fn = cost(zn);
      if (fn < f) { const rel = (f - fn) / Math.max(f, 1e-300); z = zn; f = fn; lam = Math.max(lam / 10, 1e-15); accepted = true; if (rel < 1e-14 || Math.sqrt(step.reduce((s, v) => s + v * v, 0)) < 1e-10) break; }
      else lam *= 10;
    }
    if (!accepted || lam > 1e8) break;
  }
  return { z, misfit: rmse(Array.from(sens(z).q), y) };
}
function project(fam, eta) {
  const r = rankOf(fam), A = fam.U.map((row, j) => row.map(u => u / fam.scale[j])), b = eta.map((v, j) => (v - fam.mean[j]) / fam.scale[j]);
  const N = Array.from({ length: r }, () => new Array(r).fill(0)), rhs = new Array(r).fill(0);
  A.forEach((row, i) => { for (let p = 0; p < r; p++) { rhs[p] += row[p] * b[i]; for (let q = 0; q < r; q++) N[p][q] += row[p] * row[q]; } });
  return solve(N, rhs);
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
/* One system at budget T: the five errors of the study (fams: { context, rank1? }). */
function trial(fam, sys, u, y, uv, qv, extra = {}) {
  const sc = fitLS(u, y), ad = adapt(fam, u, y);
  const out = { none: rmse(predict(fam.mean, uv), qv), scratch: rmse(predict(sc.eta, uv), qv), context: rmse(predict(decode(fam, ad.z), uv), qv), reference: rmse(predict(sys.eta, uv), qv), best: rmse(predict(decode(fam, project(fam, sys.eta)), uv), qv) };
  if (extra.rank1) out.rank1 = rmse(predict(decode(extra.rank1, adapt(extra.rank1, u, y).z), uv), qv);
  return out;
}
function drawTrial(fam, T, rng, extra) {
  const sys = sampleSystem(rng), u = multisine(T, rng), y = simulate(sys, u, rng.normal).y, uv = multisine(300, rng, 6), qv = simulate(sys, uv).q;
  return trial(fam, sys, u, y, uv, qv, extra);
}
const median = arr => { const a = [...arr].sort((x, y) => x - y), n = a.length; return n % 2 ? a[(n - 1) / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]); };
const quantile = (arr, p) => { const a = [...arr].sort((x, y) => x - y), i = (a.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return a[lo] + (a[hi] - a[lo]) * (i - lo); };
/* NET-END */

if (typeof module !== "undefined" && module.exports) {
  module.exports = { simulate, predict, fitLS, adapt, decode, project, makeRng, multisine, sampleSystem, trial, drawTrial, rmse, median, quantile };
  return;
}

const D = window.FS05, FAM = D.family, BUD = D.budgets, SEC = BUD.map(b => DT * b);
const METHODS = ["none", "scratch", "context", "reference", "best"];
const LABEL = { none: "no adaptation", scratch: "from scratch", context: "pretrained + context", reference: "reference (true η)", best: "best in family" };
const COL = { none: "--muted", scratch: "--warm", context: "--post", reference: "--ink", best: "--c4" };
const RANK1 = { mean: FAM.mean, U: FAM.U.map(r => [r[0]]), sigma: FAM.sigma, scale: FAM.scale };
const lg = Math.log10;

/* ============================================================ hero: the budget curves */
const hero = { n: 40, show: { none: true, scratch: true, context: true, reference: true, best: false }, seed: 1, own: null, running: false };
function runOwn() {
  if (hero.running) return;
  hero.running = true; hero.own = { err: Object.fromEntries(METHODS.map(m => [m, BUD.map(() => [])])), done: 0 };
  const rng = makeRng(hero.seed * 6007 + 17);
  const step = (i, n) => {
    if (i >= BUD.length) { hero.running = false; $("#hStatus").textContent = ""; drawHero(); return; }
    const t0 = performance.now();
    while (n < hero.n && performance.now() - t0 < 40) { const r = drawTrial(FAM, BUD[i], rng); METHODS.forEach(m => hero.own.err[m][i].push(r[m])); n++; }
    $("#hStatus").textContent = `computing: budget ${fmt(SEC[i], 1)} s, system ${n} of ${hero.n}`;
    if (n >= hero.n) { hero.own.done = i + 1; drawHero(); setTimeout(() => step(i + 1, 0), 0); } else setTimeout(() => step(i, n), 0);
  };
  setTimeout(() => step(0, 0), 0);
}
function drawHero() {
  const cv = $("#heroC"), narrow = narrowOf(cv, 600);
  const fr = frame(cv, narrow ? 1.2 : 1.9, { x: [lg(1.2), lg(32)], y: [lg(0.0015), lg(0.4)], xlabel: narrow ? "experiment length [s]" : "length of the experiment on the new system [s]", ylabel: narrow ? "prediction RMSE (median, IQR)" : "prediction RMSE (median, interquartile band)", xticks: SEC.map(lg), xfmt: v => (10 ** v).toFixed(10 ** v < 10 ? 1 : 0), yticks: [-2.5, -2, -1.5, -1, -0.5], yfmt: v => String(+(10 ** v).toPrecision(2)) });
  const { ctx, X, Y } = fr;
  const band = (errs, col, a) => { ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip(); ctx.fillStyle = alpha(col, a); ctx.beginPath(); errs.forEach((e, i) => { const v = lg(quantile(e, 0.75)); i ? ctx.lineTo(X(lg(SEC[i])), Y(v)) : ctx.moveTo(X(lg(SEC[i])), Y(v)); }); for (let i = errs.length - 1; i >= 0; i--) ctx.lineTo(X(lg(SEC[i])), Y(lg(quantile(errs[i], 0.25)))); ctx.closePath(); ctx.fill(); ctx.restore(); };
  METHODS.filter(m => hero.show[m]).forEach(m => { band(D.err[m], COL[m], 0.12); plotLine(fr, SEC.map(lg), D.err[m].map(e => lg(median(e))), { color: css(COL[m]), width: 2, dash: m === "reference" ? [6, 4] : null }); D.err[m].forEach((e, i) => { ctx.fillStyle = css(COL[m]); ctx.beginPath(); ctx.arc(X(lg(SEC[i])), Y(lg(median(e))), 3.5, 0, 2 * Math.PI); ctx.fill(); }); });
  if (hero.own && hero.own.done) {
    const k = hero.own.done;
    METHODS.filter(m => hero.show[m]).forEach(m => { const errs = hero.own.err[m].slice(0, k); plotLine(fr, SEC.slice(0, k).map(lg), errs.map(e => lg(median(e))), { color: css(COL[m]), width: 1.4, dash: [2, 3] }); errs.forEach((e, i) => { ctx.strokeStyle = css(COL[m]); ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(X(lg(SEC[i])), Y(lg(median(e))), 4.5, 0, 2 * Math.PI); ctx.stroke(); }); });
  }
  const ratio = BUD.map((_, i) => median(D.err.scratch[i]) / median(D.err.context[i]));
  const ownR = hero.own && hero.own.done === BUD.length ? BUD.map((_, i) => median(hero.own.err.scratch[i]) / median(hero.own.err.context[i])) : null;
  $("#rRatio").textContent = ratio.map(v => fmt(v, 1) + "×").join(", ");
  $("#rOwn").textContent = ownR ? ownR.map(v => fmt(v, 1) + "×").join(", ") : (hero.own ? "computing…" : "not run yet");
  let text = `The notebook's study (solid lines, points at the five budgets, bands from the first to the third quartile over 40 new systems). From scratch starts above doing nothing at 1.5 s (median ${fmt(median(D.err.scratch[0]), 4)} against ${fmt(median(D.err.none[0]), 4)}) and falls steadily; the pretrained context starts at ${fmt(median(D.err.context[0]), 4)} and improves more slowly; the ratio of the two medians is ${ratio.map(v => fmt(v, 1)).join(", ")} at ${SEC.map(s => fmt(s, 1)).join(", ")} s, the notebook's printed figures. The reference, the true coefficients in the learner's model, sits at ${fmt(median(D.err.reference[2]), 4)}; the best context the family can offer for each system (switch it on) sits at ${fmt(median(D.err.best[2]), 4)}. `;
  if (ownR) text += `The page's own ${hero.n} systems per budget (dotted, hollow points) give ratios ${ownR.map(v => fmt(v, 1)).join(", ")}: the same picture with different numbers, which is the right way to read a 40-system study.`;
  else if (hero.own) text += `The page's own draws are being computed and appear as dotted lines.`;
  else text += `Press the button to rerun the study with the page's own draws; it takes a second or two.`;
  caption($("#heroStatus"), text);
}
$$("#heroM .btn").forEach(b => b.addEventListener("click", () => { hero.show[b.dataset.m] = !hero.show[b.dataset.m]; b.setAttribute("aria-pressed", hero.show[b.dataset.m]); drawHero(); }));
$("#hN").addEventListener("input", e => { hero.n = +e.target.value; $("#hNV").textContent = hero.n; });
$("#hRun").addEventListener("click", () => { hero.seed++; runOwn(); });
$("#hNV").textContent = hero.n;

/* ============================================================ the table */
function fillTable() {
  const rows = BUD.map((T, i) => {
    const med = m => median(D.err[m][i]);
    const winsS = D.err.context[i].filter((v, n) => v < D.err.scratch[i][n]).length, winsN = D.err.context[i].filter((v, n) => v < D.err.none[i][n]).length, worse = D.err.scratch[i].filter((v, n) => v > D.err.none[i][n]).length;
    return `<tr><td>${fmt(SEC[i], 1)} s</td><td>${fmt(med("none"), 4)}</td><td>${fmt(med("scratch"), 4)}</td><td>${fmt(med("context"), 4)}</td><td>${fmt(med("best"), 4)}</td><td>${fmt(med("reference"), 4)}</td><td>${fmt(med("scratch") / med("context"), 1)}×</td><td>${winsS}/40</td><td>${worse}/40</td></tr>`;
  });
  $("#budT tbody").innerHTML = rows.join("");
}

/* ============================================================ per-system strip */
const strip = { i: 0 };
function drawStrip() {
  const i = strip.i, cv = $("#stripC"), narrow = narrowOf(cv);
  const ms = ["none", "scratch", "context", "best", "reference"];
  const fr = frame(cv, narrow ? 1.1 : 2.0, { x: [lg(0.0008), lg(0.6)], y: [-0.6, ms.length - 0.4], xlabel: "validation RMSE (log scale)", yticks: false, xticks: [-3, -2.5, -2, -1.5, -1, -0.5], xfmt: v => String(+(10 ** v).toPrecision(2)), pad: { l: 12 } });
  const { ctx, X, Y } = fr;
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ms.forEach((m, k) => {
    const yy = ms.length - 1 - k, errs = D.err[m][i];
    ctx.fillStyle = alpha(COL[m], 0.55); errs.forEach((e, n) => { const jit = ((n * 37) % 11 - 5) / 22; ctx.beginPath(); ctx.arc(X(lg(e)), Y(yy + jit), 3.2, 0, 2 * Math.PI); ctx.fill(); });
    const md = median(errs); ctx.strokeStyle = css(COL[m]); ctx.lineWidth = 2.5; seg(ctx, X(lg(md)), Y(yy - 0.35), X(lg(md)), Y(yy + 0.35));
    ctx.fillStyle = css("--ink-2"); ctx.fillText(LABEL[m], fr.pad.l + 4, Y(yy + 0.38));
  });
  const worse = D.err.scratch[i].filter((v, n) => v > D.err.none[i][n]).length, winsS = D.err.context[i].filter((v, n) => v < D.err.scratch[i][n]).length;
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  caption($("#stripCap"), `Budget ${fmt(SEC[i], 1)} s: the 40 systems' errors for each method (one dot each, medians as bars). ${i === 0 ? `From scratch is worse than doing nothing on ${worse} of 40 systems: with 15 samples four free coefficients overfit almost every time. ` : i === 4 ? `By 25 s from scratch is worse than doing nothing on ${worse} system only, and the context still wins ${winsS} of 40 pairs, by a median factor of ${fmt(median(D.err.scratch[i]) / median(D.err.context[i]), 1)}. ` : `From scratch is worse than doing nothing on ${worse} of 40 systems, and the context wins ${winsS} of 40 pairs. `}Means are more sensitive than medians to the long right tail: from scratch averages ${fmt(mean(D.err.scratch[i]), 4)} against a median of ${fmt(median(D.err.scratch[i]), 4)}, the context ${fmt(mean(D.err.context[i]), 4)} against ${fmt(median(D.err.context[i]), 4)}. The reference and the best-in-family dots never move with the budget, because neither uses the experiment: they are floors, at ${fmt(median(D.err.reference[i]), 4)} and ${fmt(median(D.err.best[i]), 4)}.`);
}
$$("#stripSeg .btn").forEach(b => b.addEventListener("click", () => { $$("#stripSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); strip.i = +b.dataset.i; drawStrip(); }));

/* ============================================================ exercise 2: one number or two */
const ex2 = { T: 15, seed: 2, cache: {} };
function drawEx2() {
  const key = `${ex2.T}-${ex2.seed}`;
  if (!ex2.cache[key]) { const rng = makeRng(ex2.seed * 3571 + ex2.T), rows = []; for (let n = 0; n < 40; n++) rows.push(drawTrial(FAM, ex2.T, rng, { rank1: RANK1 })); ex2.cache[key] = rows; }
  const rows = ex2.cache[key];
  const cv = $("#ex2C"), narrow = narrowOf(cv);
  const lo = lg(0.002), hi = lg(0.3);
  const fr = frame(cv, narrow ? 1.1 : 1.6, { x: [lo, hi], y: [lo, hi], xlabel: "two-number context, RMSE", ylabel: "one-number context, RMSE", xticks: [-2.5, -2, -1.5, -1], yticks: [-2.5, -2, -1.5, -1], xfmt: v => String(+(10 ** v).toPrecision(1)), yfmt: v => String(+(10 ** v).toPrecision(1)) });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--muted"); ctx.setLineDash([4, 3]); seg(ctx, X(lo), Y(lo), X(hi), Y(hi)); ctx.setLineDash([]);
  if (ex2.T === 15) { ctx.fillStyle = alpha("--muted", 0.5); D.ex2.rank2.forEach((v, n) => { ctx.beginPath(); ctx.arc(X(lg(clip(v, 10 ** lo, 10 ** hi))), Y(lg(clip(D.ex2.rank1[n], 10 ** lo, 10 ** hi))), 3.5, 0, 2 * Math.PI); ctx.fill(); }); }
  ctx.fillStyle = css("--post"); rows.forEach(r => { ctx.beginPath(); ctx.arc(X(lg(clip(r.context, 10 ** lo, 10 ** hi))), Y(lg(clip(r.rank1, 10 ** lo, 10 ** hi))), 4.5, 0, 2 * Math.PI); ctx.fill(); });
  const m1 = median(rows.map(r => r.rank1)), m2 = median(rows.map(r => r.context)), wins = rows.filter(r => r.context < r.rank1).length;
  const pm1 = median(D.ex2.rank1), pm2 = median(D.ex2.rank2), pw = D.ex2.rank2.filter((v, n) => v < D.ex2.rank1[n]).length;
  caption($("#ex2Cap"), `Forty new systems at ${fmt(DT * ex2.T, 1)} s, each adapted with the full two-number context and with a one-number context (the family's first direction only). Page: medians ${fmt(m1, 4)} with one number against ${fmt(m2, 4)} with two, two numbers better on ${wins} of 40. ${ex2.T === 15 ? `Notebook, same budget (grey): ${fmt(pm1, 4)} against ${fmt(pm2, 4)}, two numbers better on ${pw} of 40. At the shortest budget the second number barely matters, because 1.5 s do not determine it and the prior fills it in either way; the notebook's solution calls the one-number context "noticeably worse" here, which the numbers do not support. ` : `Compare with 1.5 s: the gap between one and two numbers grows with the budget, because longer experiments start to determine the second direction, which a one-number context cannot use. `}Points above the diagonal are systems where the second direction helped.`);
}
$$("#ex2Seg .btn").forEach(b => b.addEventListener("click", () => { $$("#ex2Seg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); ex2.T = +b.dataset.t; drawEx2(); }));
$("#ex2New").addEventListener("click", () => { ex2.seed++; drawEx2(); });

function drawAll() { drawHero(); drawStrip(); drawEx2(); }
fillTable();
drawAll();
onRedraw(drawAll);
})();
