/* Learning to Identify New Dynamical Systems, chapter 02, Identifying one system from scratch:
   the live demos. Output-error least squares is run here by a bounded Levenberg–Marquardt method
   with an exact Jacobian from the sensitivity equations of the learner's model, so experiments of
   any length can be fitted in the browser; the notebook's own 60 SciPy estimates and its first
   experiment of each length come from Python (ch02-data.js) for the comparison. The plant, the
   test system and the random inputs follow dynutils. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, histogram } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const DT = 0.1, X0 = [0.2, 0.0];
const SIGMA_Y = 0.01, SIGMA_W = 0.01, RHO = 0.8;
const LOW = [0.05, 0.01, 0.05, 0.0], HIGH = [5.0, 2.0, 3.0, 5.0], GENERIC = [1.0, 0.5, 1.0, 0.5];
const NAMES = ["k", "d", "b", "α"];
const actuator = u => 0.3 * Math.tanh(u / 0.6);
/* The true plant (dynutils.simulate). `noise` = { normal, sigmaY, disturbance } or null. */
function simulate(sys, u, noise) {
  const [k, d, b, a] = sys.eta;
  let q = X0[0], v = X0[1], w = 0;
  const qs = [q];
  for (let t = 0; t < u.length; t++) {
    const ut = t === 0 ? 0 : u[t - 1];
    if (noise && noise.disturbance) w = RHO * w + SIGMA_W * noise.normal();
    v += DT * (-k * q - d * v + b * actuator(ut) - a * q * q * q - sys.gamma * q ** 5 + sys.bias + w);
    q += DT * v;
    qs.push(q);
  }
  const y = noise ? qs.map(x => x + noise.sigmaY * noise.normal()) : qs.slice();
  return { q: qs, y };
}
/* The learner's model with its sensitivities: returns { q (T+1), J (T+1 x 4) }. */
function predictJac(eta, u) {
  const [k, d, b, a] = eta, T = u.length;
  let q = X0[0], v = X0[1];
  let dq = [0, 0, 0, 0], dv = [0, 0, 0, 0];
  const out = new Float64Array(T + 1), J = Array.from({ length: T + 1 }, () => [0, 0, 0, 0]);
  out[0] = q;
  for (let t = 0; t < T; t++) {
    const f = actuator(t === 0 ? 0 : u[t - 1]), q2 = q * q, s = -k - 3 * a * q2;
    const acc = -k * q - d * v + b * f - a * q2 * q;
    const extra = [-q, -v, f, -q2 * q];
    const ndv = dv.map((dvi, i) => dvi + DT * (s * dq[i] - d * dvi + extra[i]));
    v += DT * acc; q += DT * v;
    dv = ndv; dq = dq.map((dqi, i) => dqi + DT * dv[i]);
    out[t + 1] = q; J[t + 1] = dq.slice();
  }
  return { q: out, J };
}
const predict = (eta, u) => Array.from(predictJac(eta, u).q);
const rmse = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);
const sse = (eta, u, y) => { const q = predictJac(eta, u).q; let s = 0; for (let i = 0; i < y.length; i++) s += (q[i] - y[i]) ** 2; return s; };
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
function inverse(A) {
  const n = A.length, M = A.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => i === j ? 1 : 0)]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c]; if (Math.abs(piv) < 1e-300) return null;
    for (let k = 0; k < 2 * n; k++) M[c][k] /= piv;
    for (let r = 0; r < n; r++) { if (r === c) continue; const m = M[r][c]; if (!m) continue; for (let k = 0; k < 2 * n; k++) M[r][k] -= m * M[c][k]; }
  }
  return M.map(r => r.slice(n));
}
/* Output-error least squares: min sum (y - q(eta; u))^2 over LOW <= eta <= HIGH (optionally with
   some coordinates frozen) by Levenberg–Marquardt with projection onto the box. Returns the
   estimate, the residual sum of squares, the Gauss-Newton covariance sigma^2 (J^T J)^-1 over the
   free coordinates, and the iteration count. */
function fitLS(u, y, guess = GENERIC, opts = {}) {
  const frozen = opts.frozen || [false, false, false, false];
  let x = guess.map((g, i) => Math.min(HIGH[i], Math.max(LOW[i], g)));
  const cost = eta => sse(eta, u, y);
  let f = cost(x), lam = 1e-3, iters = 0;
  for (; iters < 300; iters++) {
    const { q, J } = predictJac(x, u);
    const H = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], g = [0, 0, 0, 0];
    for (let t = 0; t < y.length; t++) { const r = q[t] - y[t], Jt = J[t]; for (let i = 0; i < 4; i++) { g[i] += Jt[i] * r; for (let j = 0; j < 4; j++) H[i][j] += Jt[i] * Jt[j]; } }
    // free coordinates: not frozen, and not pinned at a bound the gradient pushes against
    const free = [0, 1, 2, 3].filter(i => !frozen[i] && !(x[i] <= LOW[i] + 1e-12 && g[i] > 0) && !(x[i] >= HIGH[i] - 1e-12 && g[i] < 0));
    if (!free.length) break;
    const Hf = free.map(i => free.map(j => H[i][j])), gf = free.map(i => g[i]);
    const gnorm = Math.max(...gf.map(Math.abs));
    if (gnorm < 1e-14) break;
    let accepted = false;
    for (let tries = 0; tries < 40 && !accepted; tries++) {
      const A = Hf.map((r, i) => r.map((v, j) => i === j ? v * (1 + lam) + 1e-30 : v));
      const step = solve(A, gf.map(v => -v));
      if (!step) { lam *= 10; continue; }
      const xn = x.slice();
      free.forEach((i, k) => { xn[i] = Math.min(HIGH[i], Math.max(LOW[i], x[i] + step[k])); });
      const fn = cost(xn);
      if (fn < f) {
        const rel = (f - fn) / Math.max(f, 1e-300), dx = Math.max(...free.map((i, k) => Math.abs(xn[i] - x[i]) / Math.max(1e-12, Math.abs(x[i]))));
        x = xn; f = fn; lam = Math.max(lam / 3, 1e-15); accepted = true;
        if (rel < 1e-13 || dx < 1e-10) { iters++; return finish(); }
      } else lam *= 4;
    }
    if (!accepted) break;
  }
  return finish();
  function finish() {
    const { J } = predictJac(x, u);
    const free = [0, 1, 2, 3].filter(i => !frozen[i]);
    const H = free.map(i => free.map(j => { let s = 0; for (let t = 0; t < y.length; t++) s += J[t][i] * J[t][j]; return s; }));
    const sigma2 = f / y.length, Hi = inverse(H);
    const se = [NaN, NaN, NaN, NaN];
    if (Hi) free.forEach((i, k) => { se[i] = Math.sqrt(Math.max(0, sigma2 * Hi[k][k])); });
    return { eta: x, sse: f, rms: Math.sqrt(sigma2), se, iters };
  }
}
/* Profile of the residual sum of squares along alpha, the other three re-optimised. */
function alphaProfile(u, y, alphas) {
  return alphas.map(a => fitLS(u, y, [GENERIC[0], GENERIC[1], GENERIC[2], a], { frozen: [false, false, false, true] }).sse);
}
/* xorshift128+ with Box–Muller normals. */
function makeRng(seed) {
  let s0 = (seed >>> 0) || 1, s1 = 0x9e3779b9 ^ s0;
  const next = () => { let x = s0, y = s1; s0 = y; x ^= x << 23; x ^= x >>> 17; x ^= y ^ (y >>> 26); s1 = x; return ((s0 + s1) >>> 0) / 4294967296; };
  let spare = null;
  const normal = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u1 = 0; while (u1 <= 1e-12) u1 = next();
    const u2 = next(), r = Math.sqrt(-2 * Math.log(u1));
    spare = r * Math.sin(2 * Math.PI * u2);
    return r * Math.cos(2 * Math.PI * u2);
  };
  return { next, normal, uniform: (lo, hi) => lo + (hi - lo) * next() };
}
/* dynutils.multisine: n sines with log-uniform frequencies in [fmin, fmax] Hz and random phases, peak 1. */
function multisine(T, rng, n = 3, fmin = 0.02, fmax = 0.8) {
  const f = [], ph = [];
  for (let i = 0; i < n; i++) f.push(Math.exp(rng.uniform(Math.log(fmin), Math.log(fmax))));
  const s = new Array(T).fill(0);
  for (let i = 0; i < n; i++) { const p = rng.uniform(0, 2 * Math.PI); for (let t = 0; t < T; t++) s[t] += Math.sin(2 * Math.PI * f[i] * DT * t + p); }
  const m = Math.max(...s.map(Math.abs));
  return s.map(v => v / m);
}
const median = arr => { const a = [...arr].sort((x, y) => x - y), n = a.length; return n % 2 ? a[(n - 1) / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]); };
/* Median relative error per coefficient of a set of estimates. */
const medianRelErr = (ests, truth) => truth.map((tv, j) => median(ests.map(e => Math.abs(e[j] - tv) / tv)));
/* A repeated-experiment study: `runs` fresh inputs of T samples on `sys`, each fitted from scratch. */
function study(sys, T, runs, rng, noise) {
  const ests = [];
  for (let r = 0; r < runs; r++) { const u = multisine(T, rng); const y = simulate(sys, u, { normal: rng.normal, sigmaY: noise.sigmaY, disturbance: noise.disturbance }).y; ests.push(fitLS(u, y).eta); }
  return ests;
}
/* NET-END */

if (typeof module !== "undefined" && module.exports) {
  module.exports = { simulate, predict, predictJac, fitLS, alphaProfile, makeRng, multisine, rmse, medianRelErr, study, sse, GENERIC, LOW, HIGH };
  return;
}

const D = window.FS02, TEST = D.test;
const timeAxis = n => Array.from({ length: n }, (_, t) => DT * t);
const VAL_U = multisine(300, makeRng(77), 3), VAL_Q = simulate(TEST, VAL_U).q;   // a fixed fresh validation input
const COLS = ["--post", "--warm", "--c4", "--c5"];

/* ============================================================ hero: one experiment, one fit */
const hero = { T: 40, seed: 1, source: "nb40" };
let heroState = null;
function runHero() {
  let u, y;
  if (hero.source === "fresh") { const rng = makeRng(hero.seed * 1009 + hero.T); u = multisine(hero.T, rng); y = simulate(TEST, u, { normal: rng.normal, sigmaY: SIGMA_Y, disturbance: true }).y; }
  else { const e = hero.source === "nb40" ? D.exp40 : D.exp400; u = e.u; y = e.y; }
  const fit = fitLS(u, y), clean = simulate(TEST, u).q, qfit = predict(fit.eta, u);
  heroState = { u, y, fit, clean, qfit, val: rmse(predict(fit.eta, VAL_U), VAL_Q), valTrue: rmse(predict(TEST.eta, VAL_U), VAL_Q), valGen: rmse(predict(GENERIC, VAL_U), VAL_Q) };
}
function drawHero() {
  if (!heroState) runHero();
  const { u, y, fit, clean, qfit } = heroState, T = u.length, t = timeAxis(T + 1), tEnd = DT * T;
  const cv = $("#heroC"), narrow = narrowOf(cv, 600);
  let ymax = 0.3; y.forEach(v => { ymax = Math.max(ymax, Math.abs(v)); }); ymax = Math.ceil(ymax * 10) / 10;
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, tEnd], y: [-ymax, ymax], xlabel: "time [s]", ylabel: "position" });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--muted", 0.8); y.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(t[i]), Y(v), T > 100 ? 1.3 : 2, 0, 2 * Math.PI); ctx.fill(); });
  plotLine(fr, t, clean, { color: css("--ink"), width: 1.4, dash: [5, 4] });
  plotLine(fr, t, qfit, { color: css("--warm"), width: 2 });
  // bars: estimate against truth
  const cb = $("#heroB"), frB = frame(cb, narrow ? 2.2 : 3.2, { x: [-0.5, 3.5], y: [0, 1.6], xticks: false, yticks: [0, 0.5, 1, 1.5], ylabel: "estimate / true value" });
  const c2 = frB.ctx;
  c2.strokeStyle = css("--muted"); c2.setLineDash([3, 3]); seg(c2, frB.X(-0.5), frB.Y(1), frB.X(3.5), frB.Y(1)); c2.setLineDash([]);
  c2.font = "11px " + css("--font-mono"); c2.textAlign = "center"; c2.textBaseline = "top";
  for (let j = 0; j < 4; j++) {
    const r = fit.eta[j] / TEST.eta[j], se = fit.se[j] / TEST.eta[j];
    c2.fillStyle = alpha(COLS[j], 0.75); c2.fillRect(frB.X(j - 0.3), frB.Y(Math.min(r, 1.6)), frB.X(j + 0.3) - frB.X(j - 0.3), frB.Y(0) - frB.Y(Math.min(r, 1.6)));
    if (isFinite(se)) { c2.strokeStyle = css("--ink"); c2.lineWidth = 1.2; seg(c2, frB.X(j), frB.Y(Math.max(0, Math.min(1.6, r - se))), frB.X(j), frB.Y(Math.min(1.6, r + se))); }
    c2.fillStyle = css("--ink-2"); c2.fillText(NAMES[j] + (r > 1.6 ? " ↑" : ""), frB.X(j), frB.Y(0) + 5);
  }
  $("#rEst").textContent = `(${fit.eta.map(v => fmt(v, 3)).join(", ")})`;
  $("#rTrue").textContent = `(${TEST.eta.map(v => fmt(v, 3)).join(", ")})`;
  $("#rSE").textContent = `(${fit.se.map(v => fmt(v, 3)).join(", ")})`;
  $("#rRes").textContent = fmt(fit.rms, 4) + " (noise 0.01)";
  $("#rVal").textContent = `${fmt(heroState.val, 4)} vs ${fmt(heroState.valTrue, 4)} true`;
  const rel = fit.eta.map((v, j) => 100 * Math.abs(v - TEST.eta[j]) / TEST.eta[j]);
  const nb = hero.source !== "fresh" ? (hero.source === "nb40" ? D.exp40 : D.exp400) : null;
  const atB = fit.eta.map((v, j) => v <= LOW[j] + 1e-9 ? "lower" : v >= HIGH[j] - 1e-9 ? "upper" : null);
  let text = `${hero.source === "fresh" ? `A fresh ${fmt(DT * T, 1)} s experiment on the notebook's test system` : `The notebook's first ${hero.source === "nb40" ? "4" : "40"} s experiment`} (dots: measurements; dashed: the disturbance-free truth; orange: the fitted model's response). Levenberg–Marquardt from the generic guess [[(1, 0.5, 1, 0.5)]] converged in ${fit.iters} iterations to [[\\hat\\eta = (${fit.eta.map(v => fmt(v, 3)).join(", ")})]] against the truth [[(${TEST.eta.map(v => fmt(v, 3)).join(", ")})]]: errors of ${rel.slice(0, 3).map((r, j) => `${fmt(r, 0)}% in ${NAMES[j]}`).join(", ")} and ${fmt(rel[3], 0)}% in [[\\alpha]]${atB[3] ? `, which sits at its ${atB[3]} bound` : ""}. `;
  if (nb) text += `SciPy's bounded trust-region solver found [[(${nb.fit.map(v => fmt(v, 3)).join(", ")})]] on the same data, the same minimum${Math.max(...nb.fit.map((v, j) => Math.abs(v - fit.eta[j]))) < 2e-3 ? "" : " up to the flat direction"}. `;
  text += `The residual RMS is ${fmt(fit.rms, 4)}${fit.rms > 0.013 ? ", well above the sensor noise 0.01: the coloured disturbance is in the residual and the model cannot absorb it" : ", about the sensor noise"}. The Gauss–Newton standard errors (whiskers) claim ${fmt(100 * fit.se[0] / TEST.eta[0], 0)}% on [[k]] and ${isFinite(fit.se[3]) ? fmt(100 * fit.se[3] / TEST.eta[3], 0) + "%" : "nothing useful"} on [[\\alpha]]; on a fresh 30 s input the fitted model predicts the disturbance-free response with RMSE ${fmt(heroState.val, 4)}, against ${fmt(heroState.valTrue, 4)} for the true coefficients and ${fmt(heroState.valGen, 4)} for the generic guess.`;
  caption($("#heroStatus"), text);
}
$("#hT").addEventListener("input", e => { hero.T = +e.target.value; $("#hTV").textContent = fmt(DT * hero.T, 1) + " s"; hero.source = "fresh"; syncSrc(); runHero(); drawHero(); });
$("#hNew").addEventListener("click", () => { hero.seed++; hero.source = "fresh"; syncSrc(); runHero(); drawHero(); });
$$("#heroSrc .btn").forEach(b => b.addEventListener("click", () => { hero.source = b.dataset.src; if (hero.source !== "fresh") { hero.T = hero.source === "nb40" ? 40 : 400; $("#hT").value = hero.T; $("#hTV").textContent = fmt(DT * hero.T, 1) + " s"; } syncSrc(); runHero(); drawHero(); }));
function syncSrc() { $$("#heroSrc .btn").forEach(x => x.setAttribute("aria-pressed", x.dataset.src === hero.source)); }
$("#hTV").textContent = fmt(DT * hero.T, 1) + " s";

/* ============================================================ the notebook's 60 estimates */
const hist = { T: 400 };
function drawHist() {
  const ests = hist.T === 40 ? D.est40 : D.est400, rel = medianRelErr(ests, TEST.eta);
  for (let j = 0; j < 4; j++) {
    const cv = $("#histC" + j), vals = ests.map(e => e[j]), lo = j === 3 ? 0 : 0, hi = j === 3 ? 5 : Math.max(2 * TEST.eta[j], ...vals) * 1.05;
    const fr = frame(cv, 1.3, { x: [lo, hi], y: [0, 1], xlabel: NAMES[j], yticks: false, pad: { l: 8 } });
    const bins = 15, counts = new Float64Array(bins), w = (hi - lo) / bins;
    vals.forEach(v => { const i = Math.min(bins - 1, Math.floor((v - lo) / w)); counts[i]++; });
    const cmax = Math.max(...counts);
    histogram(fr, vals, lo, hi, bins, 0.9 / cmax, alpha(COLS[j], 0.7));
    fr.ctx.strokeStyle = css("--ink"); fr.ctx.lineWidth = 2; seg(fr.ctx, fr.X(TEST.eta[j]), fr.pad.t, fr.X(TEST.eta[j]), fr.Y(0));
  }
  const a = ests.map(e => e[3]), atLo = a.filter(v => v < 1e-6).length, atHi = a.filter(v => v > 5 - 1e-6).length;
  caption($("#histCap"), `SciPy's 30 from-scratch estimates on ${hist.T === 40 ? "4" : "40"} s experiments (the notebook's run), one histogram per coefficient, the true value as a vertical line. Median relative errors: ${NAMES.map((n, j) => `${n} ${fmt(100 * rel[j], 1)}%`).join(", ")}, the notebook's printed figures. ${hist.T === 40 ? `Four seconds is less than one period of this member (${fmt(2 * Math.PI / Math.sqrt(TEST.eta[0]), 1)} s), so many coefficient combinations fit equally well and the estimates scatter widely; ` : `With 40 s, [[k]], [[d]] and [[b]] are within a few percent, but `}[[\\alpha]] is spread over its whole admissible range [[[0, 5]]], with ${atLo} of 30 estimates pinned at 0 and ${atHi} at 5: a median relative error of ${fmt(100 * rel[3], 0)}% means the hardening term is simply not identified${hist.T === 400 ? ", even from 40 s of data. The notebook's text calls this \"uncertain by tens of percent\"; the number is one hundred" : ""}.`);
}
$$("#histSeg .btn").forEach(b => b.addEventListener("click", () => { $$("#histSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); hist.T = +b.dataset.t; drawHist(); }));

/* ============================================================ the cost along alpha */
const prof = { which: "nb400" };
const ALPHAS = Array.from({ length: 26 }, (_, i) => i * 0.2);
const profCache = {};
function drawProf() {
  const e = prof.which === "nb400" ? D.exp400 : D.exp40, key = prof.which;
  if (!profCache[key]) profCache[key] = alphaProfile(e.u, e.y, ALPHAS);
  const p = profCache[key], n = e.y.length, floor = n * SIGMA_Y * SIGMA_Y;
  const cv = $("#profC"), narrow = narrowOf(cv);
  const lo = Math.min(...p, floor) * 0.9, hi = Math.max(...p) * 1.05;
  const fr = frame(cv, narrow ? 1.5 : 2.4, { x: [0, 5], y: [lo, hi], xlabel: "α (others re-optimised)", ylabel: "residual sum of squares", xticks: [0, 1, 2, 3, 4, 5] });
  fr.ctx.strokeStyle = alpha("--muted", 0.8); fr.ctx.setLineDash([3, 3]); seg(fr.ctx, fr.X(0), fr.Y(floor), fr.X(5), fr.Y(floor)); fr.ctx.setLineDash([]);
  fr.ctx.strokeStyle = css("--ink"); fr.ctx.lineWidth = 1.5; seg(fr.ctx, fr.X(TEST.eta[3]), fr.pad.t, fr.X(TEST.eta[3]), fr.Y(lo));
  plotLine(fr, ALPHAS, p, { color: css("--warm"), width: 2.2 });
  const imin = p.indexOf(Math.min(...p)), span = (Math.max(...p) - Math.min(...p)) / Math.min(...p);
  caption($("#profCap"), `The notebook's first ${prof.which === "nb400" ? "40" : "4"} s experiment: the residual sum of squares when [[\\alpha]] is held fixed and [[k, d, b]] are re-optimised (a profile), the true [[\\alpha = ${fmt(TEST.eta[3], 3)}]] as a vertical line and the sensor-noise floor [[(T + 1)\\,\\sigma_y^2 = ${fmt(floor, 4)}]] dotted. Across the whole admissible range the cost changes by only ${fmt(100 * span, 1)}%, ${prof.which === "nb400" ? "falling monotonically towards the upper bound, where the fit lands: the coloured disturbance, which keeps the cost " + fmt(p[imin] / floor, 1) + " times above the noise floor, has tilted a nearly flat valley" : "with its minimum at the bound " + fmt(ALPHAS[imin], 1) + ": the valley is flat and whichever way the noise tilts it decides the estimate"}. Nothing in the data pins [[\\alpha]]: the cubic term contributes a few percent of the restoring force at these deflections, and that few percent is the size of the unmodelled effects.`);
}
$$("#profSeg .btn").forEach(b => b.addEventListener("click", () => { $$("#profSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); prof.which = b.dataset.w; drawProf(); }));

/* ============================================================ exercise 1: starting guesses */
function fillGuess() {
  const { u, y } = D.ex1;
  const guesses = { "generic (1, 0.5, 1, 0.5)": GENERIC, "near the truth (×1.05)": TEST.eta.map(v => 1.05 * v), "far corner (0.9 × upper bounds)": HIGH.map(v => 0.9 * v) };
  const rows = Object.entries(guesses).map(([name, g]) => { const f = fitLS(u, y, g); return `<tr><td class="words">${name}</td><td>(${f.eta.map(v => fmt(v, 4)).join(", ")})</td><td>${f.sse.toExponential(4)}</td><td>${f.iters}</td></tr>`; });
  $("#guessT tbody").innerHTML = rows.join("");
  const py = Object.values(D.ex1.fits);
  caption($("#guessCap"), `Exercise 1 on the notebook's own 4 s experiment: three starting guesses, three identical minima, here and in SciPy ([[(${py[0].map(v => fmt(v, 4)).join(", ")})]] from all three guesses). The notebook's solution says that different starting points "can converge to different, equally-fitting solutions"; on this data they do not. The scatter across experiments is a property of the data, as the notebook also says, but not because the solver wanders: each experiment has one minimum, and the minimum moves with the noise.`);
}

/* ============================================================ exercise 2: noise sources */
const ex2 = { sigma: 0.001, dist: false, runs: 30 };
let ex2Cache = {};
function drawEx2() {
  const key = `${ex2.sigma}-${ex2.dist}`;
  if (!ex2Cache[key]) ex2Cache[key] = study(TEST, 40, ex2.runs, makeRng(2024), { sigmaY: ex2.sigma, disturbance: ex2.dist });
  const ests = ex2Cache[key], rel = medianRelErr(ests, TEST.eta);
  const base = ex2Cache["0.01-true"] || (ex2Cache["0.01-true"] = study(TEST, 40, ex2.runs, makeRng(2024), { sigmaY: 0.01, disturbance: true }));
  const relB = medianRelErr(base, TEST.eta);
  const cv = $("#ex2C"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.6 : 2.6, { x: [-0.5, 3.5], y: [0, 120], xticks: false, yticks: [0, 25, 50, 75, 100], ylabel: "median relative error [%]" });
  const { ctx, X, Y } = fr;
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let j = 0; j < 4; j++) {
    ctx.fillStyle = alpha("--muted", 0.35); ctx.fillRect(X(j - 0.36), Y(Math.min(120, 100 * relB[j])), X(j) - X(j - 0.36) - 2, Y(0) - Y(Math.min(120, 100 * relB[j])));
    ctx.fillStyle = alpha(COLS[j], 0.8); ctx.fillRect(X(j) + 2, Y(Math.min(120, 100 * rel[j])), X(j + 0.36) - X(j) - 2, Y(0) - Y(Math.min(120, 100 * rel[j])));
    ctx.fillStyle = css("--ink-2"); ctx.fillText(NAMES[j], X(j), Y(0) + 5);
  }
  const a = ests.map(e => e[3]);
  caption($("#ex2Cap"), `Thirty 4 s experiments on the test system, fitted from scratch, with the page's own random numbers. Grey: the notebook's setting ([[\\sigma_y = 0.01]], disturbance on), median errors ${NAMES.map((n, j) => `${n} ${fmt(100 * relB[j], 0)}%`).join(", ")}. Coloured: [[\\sigma_y = ${ex2.sigma}]], disturbance ${ex2.dist ? "on" : "off"}: ${NAMES.map((n, j) => `${n} ${fmt(100 * rel[j], 0)}%`).join(", ")}, with [[\\alpha]] estimates from ${fmt(Math.min(...a), 2)} to ${fmt(Math.max(...a), 2)}. ` + (ex2.sigma === 0.001 && !ex2.dist ? "This is the notebook's exercise 2 as written: its helper calls the simulator with noise switched off, which removes the coloured disturbance along with the sensor noise, and then adds a tenth of the sensor noise back. Every coefficient improves several-fold, [[\\alpha]] included; the notebook's solution says it \"improves a little\". " : ex2.sigma === 0.001 && ex2.dist ? "A quieter sensor alone, disturbance kept: the three linear coefficients gain a little and [[\\alpha]] nothing; the disturbance dominates the residual. " : ex2.sigma === 0.01 && !ex2.dist ? "The disturbance alone switched off, sensor noise kept: [[\\alpha]] is still undetermined; in 4 s the sensor noise alone is enough to hide a few-percent term. " : "") + `In the notebook's setting the residual carries a coloured disturbance that the model cannot absorb; only when both the sensor and the plant are quiet does a 4 s record say anything about [[\\alpha]]. (Over 40 s with neither, the page's fits put [[\\alpha]] within a few percent.)`);
}
$$("#ex2S .btn").forEach(b => b.addEventListener("click", () => { $$("#ex2S .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); ex2.sigma = +b.dataset.s; drawEx2(); }));
$$("#ex2D .btn").forEach(b => b.addEventListener("click", () => { $$("#ex2D .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); ex2.dist = b.dataset.d === "on"; drawEx2(); }));

/* ============================================================ equation error: noise amplification */
function fillEqErr() {
  const s = SIGMA_Y, d1 = Math.sqrt(2) * s / DT, d2 = Math.sqrt(6) * s / (DT * DT);
  $("#eqV1").textContent = fmt(d1, 3); $("#eqV2").textContent = fmt(d2, 2);
}

function drawAll() { drawHero(); drawHist(); drawProf(); drawEx2(); }
fillGuess(); fillEqErr();
drawAll();
onRedraw(drawAll);
})();
