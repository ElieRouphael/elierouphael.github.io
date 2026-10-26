/* Learning to Identify New Dynamical Systems, chapter 08, From a better model to better control:
   the live demos. A model-based feedforward input is computed here for each candidate model by
   bounded Levenberg–Marquardt on the input sequence, with the exact Jacobian of the predicted
   position with respect to the inputs (the linearised dynamics), and applied to the true system.
   Context adaptations use chapter 04's solver; the notebook's systems, experiments and SciPy
   results come from its own run (ch08-data.js). Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const DT = 0.1, X0 = [0.2, 0.0];
const SIGMA_Y = 0.01, SIGMA_W = 0.01, RHO = 0.8;
const LOW = [0.05, 0.01, 0.05, 0.0], HIGH = [5.0, 2.0, 3.0, 5.0], GENERIC = [1.0, 0.5, 1.0, 0.5];
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
/* Predicted positions of the learner's model and their Jacobian with respect to the inputs:
   the dynamics linearised along the trajectory, driven by the actuator slope at each input. */
function predictInputJac(eta, u) {
  const [k, d, b, a] = eta, T = u.length, q = predict(eta, u);
  const J = Array.from({ length: T + 1 }, () => new Float64Array(T));
  for (let s = 0; s < T - 1; s++) {
    let dv = DT * b * dActuator(u[s]), dq = DT * dv;      // u_s acts at step s + 1 (one-sample delay)
    J[s + 2][s] = dq;
    for (let t = s + 2; t < T; t++) { dv += DT * (-(k + 3 * a * q[t] * q[t]) * dq - d * dv); dq += DT * dv; J[t + 1][s] = dq; }
  }
  return { q, J };
}
/* dynutils.feedforward: minimise sum (q_t(u) - r_t)^2 + lam sum (u_{t+1} - u_t)^2 over |u| <= 2,
   by Levenberg–Marquardt with projection onto the box. Returns the input and the iteration count. */
function feedforward(eta, ref, lam = 0.05) {
  const T = ref.length - 1;
  let u = new Array(T).fill(0);
  const cost = uu => { const q = predict(eta, uu); let s = 0; for (let t = 0; t <= T; t++) s += (q[t] - ref[t]) ** 2; for (let t = 0; t < T - 1; t++) s += lam * (uu[t + 1] - uu[t]) ** 2; return s; };
  let f = cost(u), mu = 1e-3, iters = 0;
  for (; iters < 200; iters++) {
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
      let alpha_ = 1, un, fn;     // backtracking line search along the damped Gauss–Newton step
      for (let ls = 0; ls < 12; ls++) { un = u.slice(); free.forEach((i, k) => { un[i] = clip(u[i] + alpha_ * step[k], -U_MAX, U_MAX); }); fn = cost(un); if (fn < f) break; alpha_ *= 0.5; }
      if (fn < f) { const rel = (f - fn) / Math.max(f, 1e-300); u = un; f = fn; mu = Math.max(mu / 10, 1e-15); accepted = true; if (rel < 1e-13) { iters++; return { u, cost: f, iters }; } }
      else mu *= 10;
    }
    if (!accepted) break;
  }
  return { u, cost: f, iters };
}
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
  return { z, misfit: rmse(Array.from(sensitivity(fam, z, u).q), y) };
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
      if (fn < f) { const rel = (f - fn) / Math.max(f, 1e-300), dx = Math.max(...free.map(i => Math.abs(xn[i] - x[i]) / Math.max(1e-12, Math.abs(x[i])))); x = xn; f = fn; lam = Math.max(lam / 3, 1e-15); accepted = true; if (rel < 1e-13 || dx < 1e-10) return { eta: x }; }
      else lam *= 4;
    }
    if (!accepted) break;
  }
  return { eta: x };
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
const totalVariation = u => { let s = 0; for (let t = 0; t < u.length - 1; t++) s += Math.abs(u[t + 1] - u[t]); return s; };
/* NET-END */

if (typeof module !== "undefined" && module.exports) {
  module.exports = { simulate, predict, predictInputJac, feedforward, adapt, decode, fitLS, makeRng, multisine, sampleSystem, rmse, totalVariation };
  return;
}

const D = window.FS08, FAM = D.family, REF = D.ref, TC = REF.map((_, t) => DT * t);
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const median = arr => { const a = [...arr].sort((x, y) => x - y), n = a.length; return n % 2 ? a[(n - 1) / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]); };
const MODELS = [["none", "--muted", "average-system model"], ["context", "--post", "adapted model (6 s, designed)"], ["scratch", "--warm", "from scratch (same 6 s)"], ["reference", "--ink", "true coefficients"]];

/* ============================================================ hero: one system, four models */
const hero = { i: 0, lam: 0.05, noise: false, seed: 1, show: { none: true, context: true, scratch: false, reference: true }, fresh: null };
const ffCache = {};
function ffFor(key, eta, lam) { const k = key + "|" + lam; if (!ffCache[k]) ffCache[k] = feedforward(eta, REF, lam).u; return ffCache[k]; }
function heroSystem() {
  if (hero.i < 15) { const s = D.systems[hero.i]; const ad = adapt(FAM, D.uDesign, s.y); const sc = fitLS(D.uDesign, s.y); return { sys: { eta: s.eta, gamma: s.gamma, bias: s.bias }, etas: { none: FAM.mean, context: decode(FAM, ad.z), scratch: sc.eta, reference: s.eta }, key: "nb" + hero.i, z: ad.z, py: s }; }
  if (!hero.fresh || hero.fresh.seed !== hero.seed) { const rng = makeRng(hero.seed * 3301 + 7), sys = sampleSystem(rng), y = simulate(sys, D.uDesign, rng.normal).y, ad = adapt(FAM, D.uDesign, y), sc = fitLS(D.uDesign, y); hero.fresh = { seed: hero.seed, sys, etas: { none: FAM.mean, context: decode(FAM, ad.z), scratch: sc.eta, reference: sys.eta }, key: "fresh" + hero.seed, z: ad.z }; }
  return hero.fresh;
}
function drawHero() {
  const H = heroSystem(), rng = makeRng(hero.seed * 17 + hero.i), noise = hero.noise ? rng.normal : null;
  const runs = {};
  MODELS.forEach(([m]) => { const u = ffFor(H.key + m, H.etas[m], hero.lam); const q = simulate(H.sys, u, noise).q; runs[m] = { u, q, err: rmse(q, REF), clean: rmse(simulate(H.sys, u).q, REF), own: rmse(predict(H.etas[m], u), REF) }; });
  const cv = $("#heroQ"), narrow = narrowOf(cv, 600);
  const fr = frame(cv, narrow ? 1.5 : 2.3, { x: [0, 12], y: [-0.25, 0.35], xlabel: "time [s]", ylabel: "position", xticks: [0, 2, 4, 6, 8, 10, 12], yticks: [-0.2, -0.1, 0, 0.1, 0.2, 0.3] });
  plotLine(fr, TC, REF, { color: css("--ink"), width: 3 });
  MODELS.filter(([m]) => hero.show[m]).forEach(([m, c]) => plotLine(fr, TC, runs[m].q, { color: css(c), width: 1.7, dash: m === "reference" ? [6, 4] : null }));
  const cu = $("#heroU"), fru = frame(cu, narrow ? 3 : 4.5, { x: [0, 12], y: [-1, 1], xlabel: "time [s]", ylabel: "feedforward input u", xticks: [0, 2, 4, 6, 8, 10, 12], yticks: [-1, -0.5, 0, 0.5, 1] });
  MODELS.filter(([m]) => hero.show[m]).forEach(([m, c]) => plotLine(fru, TC.slice(0, 120), runs[m].u, { color: css(c), width: 1.5, dash: m === "reference" ? [6, 4] : null }));
  MODELS.forEach(([m]) => { $("#r_" + m).textContent = `${fmt(runs[m].clean, 4)} / ${fmt(runs[m].own, 4)}`; });
  const py = H.py;
  let text = `${hero.i < 15 ? `The notebook's new system ${hero.i + 1} of 15` : "A fresh member of the family"}, coefficients [[(${H.sys.eta.map(v => fmt(v, 3)).join(", ")})]], context estimated from the designed 6 s experiment${hero.i < 15 ? " (SciPy's and the page's contexts agree to five decimals)" : ""}. Each model's feedforward input is computed by least squares on 120 inputs with smoothing [[\\lambda = ${fmt(hero.lam, 3)}]], then applied to the true system${hero.noise ? " with a disturbance and noise realisation from the page's generator" : " without disturbance"}. Tracking RMSE: ${MODELS.filter(([m]) => hero.show[m]).map(([m, , l]) => `${l} ${fmt(runs[m].err, 4)}`).join(", ")}. `;
  text += `Each model believes its own input tracks the reference to about ${fmt(mean(MODELS.map(([m]) => runs[m].own)), 4)}; the difference between that promise and the achieved error is the model error, and feedforward cannot correct it while running. `;
  if (hero.i < 15 && hero.lam === 0.05) text += `SciPy's inputs for this system give ${fmt(py.errClean.none, 4)}, ${fmt(py.errClean.context, 4)} and ${fmt(py.errClean.reference, 4)} without disturbance, and ${fmt(py.err.none, 4)}, ${fmt(py.err.context, 4)}, ${fmt(py.err.reference, 4)} with the notebook's realisation.`;
  caption($("#heroStatus"), text);
}
$("#hSys").addEventListener("input", e => { hero.i = +e.target.value; $("#hSysV").textContent = hero.i < 15 ? `notebook's system ${hero.i + 1}` : "fresh system"; drawHero(); });
$("#hLam").addEventListener("input", e => { hero.lam = +(10 ** +e.target.value).toPrecision(2); $("#hLamV").textContent = fmt(hero.lam, 3); drawHero(); });
$("#hNoise").addEventListener("click", () => { hero.noise = !hero.noise; $("#hNoise").setAttribute("aria-pressed", hero.noise); $("#hNoise").textContent = hero.noise ? "disturbance on" : "disturbance off"; drawHero(); });
$("#hNew").addEventListener("click", () => { hero.seed++; hero.i = 15; $("#hSys").value = 15; $("#hSysV").textContent = "fresh system"; drawHero(); });
$$("#heroM .btn").forEach(b => b.addEventListener("click", () => { hero.show[b.dataset.m] = !hero.show[b.dataset.m]; b.setAttribute("aria-pressed", hero.show[b.dataset.m]); drawHero(); }));
$("#hSysV").textContent = "notebook's system 1"; $("#hLamV").textContent = fmt(hero.lam, 3);

/* ============================================================ the fifteen systems */
const fs = { own: null, running: false, seed: 0 };
function drawFifteen() {
  const cv = $("#fifC"), narrow = narrowOf(cv);
  const rows = [["none", "--muted"], ["context", "--post"], ["scratch", "--warm"], ["reference", "--ink"]];
  const fr = frame(cv, narrow ? 1.1 : 1.9, { x: [0, 0.17], y: [-0.6, rows.length - 0.4], xlabel: "tracking RMSE", yticks: false, xticks: [0, 0.05, 0.1, 0.15], pad: { l: 12 } });
  const { ctx, X, Y } = fr;
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  rows.forEach(([m, c], k) => {
    const yy = rows.length - 1 - k;
    const clean = D.systems.map(s => s.errClean[m]), noisy = m === "scratch" ? null : D.systems.map(s => s.err[m]);
    clean.forEach((e, n) => { const jit = ((n * 37) % 11 - 5) / 20; ctx.fillStyle = alpha(c, 0.85); ctx.beginPath(); ctx.arc(X(Math.min(0.17, e)), Y(yy + 0.12 + jit), 3.2, 0, 2 * Math.PI); ctx.fill(); });
    if (noisy) noisy.forEach((e, n) => { const jit = ((n * 37) % 11 - 5) / 20; ctx.strokeStyle = alpha(c, 0.8); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(X(Math.min(0.17, e)), Y(yy - 0.22 + jit), 3.2, 0, 2 * Math.PI); ctx.stroke(); });
    ctx.strokeStyle = css(c); ctx.lineWidth = 2.5; seg(ctx, X(mean(clean)), Y(yy + 0.3), X(mean(clean)), Y(yy - 0.05)); if (noisy) seg(ctx, X(mean(noisy)), Y(yy - 0.05), X(mean(noisy)), Y(yy - 0.4));
    ctx.fillStyle = css("--ink-2"); ctx.fillText(MODELS.find(x => x[0] === m)[2], fr.pad.l + 4, Y(yy + 0.42));
    if (fs.own && fs.own.rows.length) { const v = mean(fs.own.rows.map(r => r[m])); ctx.strokeStyle = css(c); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(v), Y(yy + 0.12), 6, 0, 2 * Math.PI); ctx.stroke(); }
  });
  const ce = D.ctrlErr, wins = D.systems.filter(s => s.err.context < s.err.reference).length, winsC = D.systems.filter(s => s.errClean.context < s.errClean.reference).length;
  const cleanMean = m => mean(D.systems.map(s => s.errClean[m]));
  let text = `The notebook's 15 new systems. Filled dots and upper bars: tracking error without disturbance, the pure model error: ${fmt(cleanMean("none"), 4)} for the average system, ${fmt(cleanMean("context"), 4)} for the adapted context, ${fmt(cleanMean("scratch"), 4)} for a from-scratch fit on the same 6 s experiment, ${fmt(cleanMean("reference"), 4)} for the true coefficients (the quintic term and the bias are not in the model). Hollow dots and lower bars: with the notebook's disturbance realisation, the printed means ${fmt(mean(ce.none), 4)}, ${fmt(mean(ce.context), 4)} and ${fmt(mean(ce.reference), 4)}. The adapted model cuts the average-system error by two thirds but does not match the true coefficients: its mean is ${fmt(mean(ce.context) / mean(ce.reference), 2)} times theirs, and it is better on ${wins} of 15 systems with disturbance and ${winsC} of 15 without. The notebook's text says it "matches" and "can even be marginally better"; the numbers say within 40%, and never better once the disturbance's luck is removed. `;
  if (fs.own && fs.own.rows.length) text += `Page's rerun on ${fs.own.rows.length} fresh systems (hollow circles, no disturbance): ${MODELS.map(([m, , l]) => `${l} ${fmt(mean(fs.own.rows.map(r => r[m])), 4)}`).join(", ")}.`;
  else text += `Rerun on fresh systems to see the same ordering with other numbers.`;
  caption($("#fifCap"), text);
}
function runFifteen() {
  if (fs.running) return; fs.running = true; fs.seed++; fs.own = { rows: [] };
  const rng = makeRng(fs.seed * 9311 + 5);
  const step = () => { if (fs.own.rows.length >= 15) { fs.running = false; $("#fifStatus").textContent = ""; drawFifteen(); return; } const sys = sampleSystem(rng), y = simulate(sys, D.uDesign, rng.normal).y, ad = adapt(FAM, D.uDesign, y), sc = fitLS(D.uDesign, y); const etas = { none: FAM.mean, context: decode(FAM, ad.z), scratch: sc.eta, reference: sys.eta }; const row = {}; for (const [m] of MODELS) row[m] = rmse(simulate(sys, feedforward(etas[m], REF, 0.05).u).q, REF); fs.own.rows.push(row); $("#fifStatus").textContent = `system ${fs.own.rows.length} of 15`; drawFifteen(); setTimeout(step, 0); };
  setTimeout(step, 0);
}
$("#fifRun").addEventListener("click", runFifteen);

/* ============================================================ exercise 1: random against designed */
const e1 = { own: null, running: false, seed: 0 };
function drawEx1() {
  const cv = $("#ex1C"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.3 : 2.4, { x: [0, 0.06], y: [-0.6, 1.6], xlabel: "tracking RMSE of the adapted model", yticks: false, xticks: [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06], pad: { l: 12 } });
  const { ctx, X, Y } = fr;
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  const sets = [["designed experiment (notebook)", D.ctrlErr.context, "--c4", 1], ["random experiment (notebook)", D.ex1, "--post", 0]];
  sets.forEach(([lab, vals, c, yy]) => { vals.forEach((e, n) => { const jit = ((n * 37) % 11 - 5) / 22; ctx.fillStyle = alpha(c, 0.75); ctx.beginPath(); ctx.arc(X(Math.min(0.06, e)), Y(yy + jit), 3.2, 0, 2 * Math.PI); ctx.fill(); }); ctx.strokeStyle = css(c); ctx.lineWidth = 2.5; seg(ctx, X(mean(vals)), Y(yy - 0.35), X(mean(vals)), Y(yy + 0.35)); ctx.fillStyle = css("--ink-2"); ctx.fillText(lab, fr.pad.l + 4, Y(yy + 0.38)); });
  if (e1.own && e1.own.rows.length) { [["designed", 1, "--c4"], ["random", 0, "--post"]].forEach(([k, yy, c]) => { const v = mean(e1.own.rows.map(r => r[k])); ctx.strokeStyle = css(c); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(v), Y(yy), 6, 0, 2 * Math.PI); ctx.stroke(); }); }
  let text = `The notebook's exercise 1: adapt from a random 6 s experiment instead of the designed one, on 15 further systems, and track. Means with the notebook's disturbance realisations: ${fmt(mean(D.ex1), 4)} for the random experiments against ${fmt(mean(D.ctrlErr.context), 4)} for the designed ones (on different systems), with the average-system model at ${fmt(mean(D.ctrlErr.none), 4)} and the true coefficients at ${fmt(mean(D.ctrlErr.reference), 4)}. The notebook's solution says the random experiment's error "sits above the designed experiment's, closer to the midpoint"; it is in fact slightly lower here. Fifteen systems and a disturbance floor of about 0.011 in quadrature leave no room to see the design advantage that chapter 06 measured on prediction. `;
  if (e1.own && e1.own.rows.length) { const o = e1.own; text += `Page's paired rerun (hollow circles): ${o.rows.length} fresh systems, each adapted from both a designed and a random 6 s experiment, no disturbance: designed ${fmt(mean(o.rows.map(r => r.designed)), 4)}, random ${fmt(mean(o.rows.map(r => r.random)), 4)}, designed better on ${o.rows.filter(r => r.designed < r.random).length} of ${o.rows.length}.`; }
  else text += `The page can rerun it paired, both experiments on the same systems and without disturbance, which is the comparison that could show the difference.`;
  caption($("#ex1Cap"), text);
}
function runEx1() {
  if (e1.running) return; e1.running = true; e1.seed++; e1.own = { rows: [] };
  const rng = makeRng(e1.seed * 7001 + 3);
  const step = () => { if (e1.own.rows.length >= 15) { e1.running = false; $("#ex1Status").textContent = ""; drawEx1(); return; } const sys = sampleSystem(rng), yd = simulate(sys, D.uDesign, rng.normal).y, ur = multisine(60, rng), yr = simulate(sys, ur, rng.normal).y; const zd = adapt(FAM, D.uDesign, yd).z, zr = adapt(FAM, ur, yr).z; e1.own.rows.push({ designed: rmse(simulate(sys, feedforward(decode(FAM, zd), REF, 0.05).u).q, REF), random: rmse(simulate(sys, feedforward(decode(FAM, zr), REF, 0.05).u).q, REF) }); $("#ex1Status").textContent = `system ${e1.own.rows.length} of 15`; drawEx1(); setTimeout(step, 0); };
  setTimeout(step, 0);
}
$("#ex1Run").addEventListener("click", runEx1);

/* ============================================================ exercise 2: the smoothness penalty */
const e2 = { lam: 0.5 };
function drawEx2() {
  const sys = { eta: D.ex2.eta, gamma: D.ex2.gamma, bias: D.ex2.bias };
  const u = ffFor("ex2", sys.eta, e2.lam), q = simulate(sys, u).q, u05 = ffFor("ex2", sys.eta, 0.05), q05 = simulate(sys, u05).q;
  const cv = $("#ex2U"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 2.2 : 3.6, { x: [0, 12], y: [-0.8, 0.8], ylabel: "input u", xticks: [0, 2, 4, 6, 8, 10, 12], yticks: [-0.5, 0, 0.5] });
  plotLine(fr, TC.slice(0, 120), u05, { color: alpha("--muted", 0.8), width: 1.3 }); plotLine(fr, TC.slice(0, 120), u, { color: css("--warm"), width: 2 });
  const cq = $("#ex2Q"), frq = frame(cq, narrow ? 2.2 : 3.6, { x: [0, 12], y: [-0.25, 0.35], xlabel: "time [s]", ylabel: "position", xticks: [0, 2, 4, 6, 8, 10, 12], yticks: [-0.2, 0, 0.2] });
  plotLine(frq, TC, REF, { color: css("--ink"), width: 2.5 }); plotLine(frq, TC, q05, { color: alpha("--muted", 0.8), width: 1.3 }); plotLine(frq, TC, q, { color: css("--warm"), width: 2 });
  caption($("#ex2Cap"), `The notebook's exercise-2 system with the true coefficients, [[\\lambda = ${fmt(e2.lam, 3)}]] (orange) against the notebook's [[\\lambda = 0.05]] (grey). Total variation of the input ${fmt(totalVariation(u), 2)} against ${fmt(totalVariation(u05), 2)}, largest input ${fmt(Math.max(...u.map(Math.abs)), 2)} against ${fmt(Math.max(...u05.map(Math.abs)), 2)}; tracking RMSE without disturbance ${fmt(rmse(q, REF), 4)} against ${fmt(rmse(q05, REF), 4)}. A heavier penalty produces a gentler input that changes less from sample to sample, at the cost of tracking the two sigmoid steps of the reference less closely; the optimisation trades tracking accuracy for input smoothness exactly as [[\\lambda]] dictates, whichever model supplied the coefficients. Python's sweep on this system: ${Object.entries(D.ex2.sweep).map(([l, v]) => `λ = ${l}: ${fmt(v.err, 4)}`).join(", ")}.`);
}
$("#e2Lam").addEventListener("input", e => { e2.lam = +(10 ** +e.target.value).toPrecision(2); $("#e2LamV").textContent = fmt(e2.lam, 3); drawEx2(); });
$("#e2LamV").textContent = fmt(e2.lam, 3);

function drawAll() { drawHero(); drawFifteen(); drawEx1(); drawEx2(); }
drawAll();
onRedraw(drawAll);
})();
