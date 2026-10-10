/* TrustDrive chapter 09, Advanced ideas: the live demos.
   The notebook's NIS streams, CUSUM, split conformal prediction and adaptive conformal
   inference, with their own seeded random numbers.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { makeRng, quantileSorted } = window.Stats;

/* ---------------- the methods ---------------- */
const GATE = 9.21;
/* CUSUM g_k = max(0, g_{k-1} + z_k - kref), alarm above h, restarting from zero after an alarm */
function cusum(z, kref, h) {
  const g = new Float64Array(z.length), alarms = [];
  let s = 0;
  for (let t = 0; t < z.length; t++) {
    s = Math.max(0, s + z[t] - kref); g[t] = s;
    if (s > h) { alarms.push(t); s = 0; }
  }
  return { g, alarms };
}
/* P(chi^2_{2m} > t) = e^{-t/2} sum_{i<m} (t/2)^i / i!, and the noncentral chi^2_2 tail as a Poisson mixture */
function chi2EvenSf(t, m) { let term = Math.exp(-t / 2), s = 0; for (let i = 0; i < m; i++) { s += term; term *= (t / 2) / (i + 1); } return s; }
function ncx2Sf(t, lam) {
  let w = Math.exp(-lam / 2), s = 0;
  for (let j = 0; j < 200; j++) { s += w * chi2EvenSf(t, j + 1); w *= (lam / 2) / (j + 1); if (w < 1e-16 && j > lam) break; }
  return s;
}
/* ridge regression on random cosine features, solved by Cholesky */
function ridgeFit(Phi, y, reg) {
  const n = Phi.length, d = Phi[0].length, A = Array.from({ length: d }, () => new Float64Array(d)), b = new Float64Array(d);
  for (let r = 0; r < n; r++) { const p = Phi[r]; for (let i = 0; i < d; i++) { b[i] += p[i] * y[r]; const pi = p[i]; if (pi) for (let j = 0; j <= i; j++) A[i][j] += pi * p[j]; } }
  for (let i = 0; i < d; i++) A[i][i] += reg;
  for (let j = 0; j < d; j++) {                                              // A = L L^T, in place (lower triangle)
    let s = A[j][j]; for (let k = 0; k < j; k++) s -= A[j][k] * A[j][k]; A[j][j] = Math.sqrt(s);
    for (let i = j + 1; i < d; i++) { let v = A[i][j]; for (let k = 0; k < j; k++) v -= A[i][k] * A[j][k]; A[i][j] = v / A[j][j]; }
  }
  const z = new Float64Array(d), w = new Float64Array(d);
  for (let i = 0; i < d; i++) { let v = b[i]; for (let k = 0; k < i; k++) v -= A[i][k] * z[k]; z[i] = v / A[i][i]; }
  for (let i = d - 1; i >= 0; i--) { let v = z[i]; for (let k = i + 1; k < d; k++) v -= A[k][i] * w[k]; w[i] = v / A[i][i]; }
  return w;
}
/* the conformal half-width: the ceil((n + 1)(1 - alpha))-th smallest score */
function conformalQ(sortedScores, alpha) {
  const k = Math.ceil((sortedScores.length + 1) * (1 - alpha));
  return k > sortedScores.length ? Infinity : sortedScores[k - 1];
}
/* online conformal along a stream, as the notebook: the interval at t is the (1 - alpha_t) quantile
   (linear interpolation, as np.quantile) of all earlier scores; the first `warm` steps count as covered */
function onlineConformal(scores, alpha, { adaptive = false, gamma = 0.02, warm = 50 } = {}) {
  const T = scores.length, covered = new Uint8Array(T), q = new Float64Array(T), at = new Float64Array(T), sorted = [];
  let a = alpha;
  const insert = v => { let lo = 0, hi = sorted.length; while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] < v) lo = m + 1; else hi = m; } sorted.splice(lo, 0, v); };
  for (let t = 0; t < T; t++) {
    if (t < warm) { covered[t] = 1; if (adaptive) a += gamma * alpha; at[t] = a; q[t] = NaN; insert(scores[t]); continue; }
    const lvl = adaptive ? 1 - clamp(a, 0, 1) : 1 - alpha;
    const qt = quantileSorted(sorted, lvl), c = scores[t] <= qt;
    covered[t] = c ? 1 : 0; q[t] = qt;
    if (adaptive) a += gamma * (alpha - (c ? 0 : 1));
    at[t] = a; insert(scores[t]);
  }
  return { covered, q, at };
}

/* =====================================================================
   THE DEMO: CUSUM AGAINST THE SINGLE-SAMPLE GATE
   ===================================================================== */
const N = 400, CHANGE = 200;
const hero = { seed: 11, base: null };                                       // a typical stream: two false gate alarms, none for CUSUM
function makeBase(seed) {                                                    // the stream's randomness, kept while the sliders move
  const r = makeRng(seed);
  return { e: Array.from({ length: N }, () => r.exponential(0.5)), n1: Array.from({ length: N }, () => r.normal()), n2: Array.from({ length: N }, () => r.normal()) };
}
function stream(lam) {
  const B = hero.base, s = Math.sqrt(lam);
  return B.e.map((v, t) => (t < CHANGE ? v : (s + B.n1[t]) ** 2 + B.n2[t] ** 2));   // chi^2_2, then noncentral chi^2_2(lam)
}
function updateHero() {
  hero.base = hero.base || makeBase(hero.seed);
  const lam = +$("#hL").value, h = +$("#hH").value, k = +$("#hK").value;
  $("#hLV").textContent = `${fmt(lam, 1)} (mean ${fmt(2 + lam, 1)})`; $("#hHV").textContent = String(h); $("#hKV").textContent = fmt(k, 2);
  const z = stream(lam), { g, alarms } = cusum(z, k, h);
  const gateHits = z.map((v, t) => (v > GATE ? t : -1)).filter(t => t >= 0);
  const cDet = alarms.find(t => t >= CHANGE), gDet = gateHits.find(t => t >= CHANGE);
  const cFalse = alarms.filter(t => t < CHANGE).length, gFalse = gateHits.filter(t => t < CHANGE).length;
  const narrow = narrowOf($("#zC")), t = Array.from({ length: N }, (_, i) => i);
  const top = Math.min(40, Math.max(16, 1.05 * Math.max(...z)));
  const fr = frame($("#zC"), narrow ? 1.6 : 2.8, { x: [0, N], y: [0, top], xlabel: "step k", ylabel: "NIS", yfmt: v => String(Math.round(v)) });
  const shadeAfter = F => { F.ctx.fillStyle = alpha("--geom", 0.07); F.ctx.fillRect(F.X(CHANGE), F.pad.t, F.X(N) - F.X(CHANGE), F.ih); };
  shadeAfter(fr);
  plotLine(fr, t, z.map(v => Math.min(v, top)), { color: alpha("--warm", 0.85), width: 1 });
  fr.ctx.strokeStyle = css("--ink"); fr.ctx.lineWidth = 1.2; fr.ctx.setLineDash([5, 4]); seg(fr.ctx, fr.X(0), fr.Y(GATE), fr.X(N), fr.Y(GATE)); fr.ctx.setLineDash([]);
  fr.ctx.fillStyle = css("--geom"); gateHits.forEach(i => fr.ctx.fillRect(fr.X(i) - 0.75, fr.Y(0) - 7, 1.5, 7));
  const gTop = Math.max(1.3 * h, 1.05 * Math.max(...g));
  const gr = frame($("#gC"), narrow ? 1.9 : 3.4, { x: [0, N], y: [0, gTop], xlabel: "step k", ylabel: "g", yfmt: v => String(Math.round(v)) });
  shadeAfter(gr);
  plotLine(gr, t, Array.from(g), { color: css("--c4"), width: 1.8 });
  gr.ctx.strokeStyle = css("--ink"); gr.ctx.lineWidth = 1.2; gr.ctx.setLineDash([5, 4]); seg(gr.ctx, gr.X(0), gr.Y(h), gr.X(N), gr.Y(h)); gr.ctx.setLineDash([]);
  gr.ctx.fillStyle = css("--geom"); alarms.forEach(i => gr.ctx.fillRect(gr.X(i) - 1, gr.Y(0) - 9, 2, 9));
  const d = v => (v === undefined ? "none" : v === CHANGE ? "at once" : `${v - CHANGE} step${v - CHANGE === 1 ? "" : "s"}`);
  $("#rC").textContent = d(cDet); $("#rG").textContent = d(gDet); $("#rFG").textContent = String(gFalse); $("#rFC").textContent = String(cFalse);
  const pAfter = ncx2Sf(GATE, lam);
  let text = lam === 0 ? `No change at all: the second half is in control too. The gate fires on ${gateHits.length} of ${N} samples (about 1% expected), the CUSUM ${alarms.length} time${alarms.length === 1 ? "" : "s"}.`
    : `After the change, each sample crosses the gate with probability ${fmt(pAfter, 2)}, and the mean rises from 2 to ${fmt(2 + lam, 1)}. ` +
      (cDet !== undefined ? `The CUSUM detects it after <b>${cDet - CHANGE} steps</b>` : "The CUSUM does not detect it") +
      (gDet !== undefined ? `; the gate first fires ${gDet === CHANGE ? "at once" : `after ${gDet - CHANGE}`}. ` : "; the gate never fires after it. ") +
      `Before the change, the gate raised ${gFalse} false alarm${gFalse === 1 ? "" : "s"} and the CUSUM ${cFalse}.`;
  if (lam > 0 && k <= 2) text += " With the reference at or below the in-control mean of 2, the sum drifts upwards even without a change: alarms become frequent.";
  else if (lam > 0 && k >= 2 + lam) text += ` With the reference above the out-of-control mean (${fmt(2 + lam, 1)}), the sum drains even after the change: the CUSUM goes nearly blind.`;
  caption($("#heroStatus"), text);
}
["#hL", "#hH", "#hK"].forEach(id => $(id).addEventListener("input", updateHero));
$("#hNew").addEventListener("click", () => { hero.seed++; hero.base = makeBase(hero.seed); updateHero(); });

/* =====================================================================
   PART 2: AVERAGE RUN LENGTHS
   ===================================================================== */
const HS = [3, 5, 7, 9, 11, 13], ARL = { c0: [], c1: [], done: false };
function runArl() {
  const r = makeRng(17), R0 = 400, R1 = 3000;
  const jobs = HS.flatMap(h => [[h, 0], [h, 1]]);
  let j = 0;
  const slice = () => {
    const t0 = performance.now();
    while (j < jobs.length && performance.now() - t0 < 25) {
      const [h, after] = jobs[j++], R = after ? R1 : R0, sl = Math.sqrt(6);
      let total = 0;
      for (let rep = 0; rep < R; rep++) {
        let g = 0, n = 0;
        do { n++; const z = after ? (sl + r.normal()) ** 2 + r.normal() ** 2 : r.exponential(0.5); g = Math.max(0, g + z - 5); } while (g <= h);
        total += n;
      }
      (after ? ARL.c1 : ARL.c0).push(total / R);
    }
    if (j < jobs.length) { $("#arlCap").textContent = `Simulating… ${Math.round(100 * j / jobs.length)}%`; setTimeout(slice, 0); }
    else { ARL.done = true; drawArl(); }
  };
  slice();
}
const GATE_T = Array.from({ length: 13 }, (_, i) => 5 + i * 0.75);           // gate thresholds 5 to 14
const gate0 = GATE_T.map(t => Math.exp(t / 2)), gate1 = GATE_T.map(t => 1 / ncx2Sf(t, 6));
const logInterp = (x, xs, ys) => {                                           // interpolate in log-log
  const lx = Math.log(x);
  for (let i = 1; i < xs.length; i++) if (Math.log(xs[i]) >= lx) { const a = Math.log(xs[i - 1]), b = Math.log(xs[i]), f = (lx - a) / (b - a); return Math.exp(Math.log(ys[i - 1]) + f * (Math.log(ys[i]) - Math.log(ys[i - 1]))); }
  return NaN;
};
function drawArl() {
  if (!ARL.done) return;
  const cv = $("#arlC"), L10 = Math.log10;
  const fr = frame(cv, narrowOf(cv) ? 1.2 : 1.7, { x: [L10(8), L10(8000)], y: [L10(1), L10(10)], xlabel: "ARL₀, mean steps between false alarms", ylabel: "ARL₁, detection delay",
    xticks: [1, 2, 3], xfmt: v => String(10 ** v), yticks: [0, L10(2), L10(5), 1], yfmt: v => String(Math.round(10 ** v)) });
  const { ctx, X, Y } = fr;
  [[gate0, gate1, "--geom"], [ARL.c0, ARL.c1, "--c4"]].forEach(([a0, a1, col]) => {
    plotLine(fr, a0.map(L10), a1.map(L10), { color: css(col), width: 2 });
    a0.forEach((v, i) => { ctx.fillStyle = css(col); ctx.beginPath(); ctx.arc(X(L10(v)), Y(L10(a1[i])), 3.5, 0, 2 * Math.PI); ctx.fill(); });
  });
  const at = n => [logInterp(n, gate0, gate1), logInterp(n, ARL.c0, ARL.c1)];
  const [g1, c1] = at(100), [g3, c3] = at(300), [g10, c10] = at(1000);
  caption($("#arlCap"),
    `Lower is better at each false-alarm interval. At one false alarm per 100 steps, the CUSUM detects in ${fmt(c1, 1)} steps against ${fmt(g1, 1)} for the gate; per 300, ${fmt(c3, 1)} against ${fmt(g3, 1)}; per 1,000, <b>${fmt(c10, 1)} against ${fmt(g10, 1)}</b>. ` +
    `The CUSUM's points are simulated here (thresholds 3 to 13, up to ${Math.round(ARL.c0[ARL.c0.length - 1])} steps between false alarms); the gate's are exact.`);
}

/* =====================================================================
   PARTS 3 AND 4: CONFORMAL PREDICTION
   ===================================================================== */
const f = x => Math.sin(1.5 * x), NOISE = 0.1;
const ordinal = n => n + ([11, 12, 13].includes(n % 100) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th"));
const CF = (() => {
  const r = makeRng(29), D = 200;
  const W = Array.from({ length: D }, () => r.normal(0, 1.5)), B = Array.from({ length: D }, () => 2 * Math.PI * r.uniform());
  const phi = x => W.map((w, i) => Math.cos(w * x + B[i]));
  const Xtr = Array.from({ length: 1200 }, () => -2 + 4 * r.uniform()), Ytr = Xtr.map(x => f(x) + r.normal(0, NOISE));
  const w = ridgeFit(Xtr.slice(0, 400).map(phi), Ytr.slice(0, 400), 1e-2);
  const model = x => { const p = phi(x); let s = 0; for (let i = 0; i < D; i++) s += p[i] * w[i]; return s; };
  const cal = Xtr.slice(400).map((x, i) => Math.abs(Ytr[400 + i] - model(x))).sort((a, b) => a - b);
  const testU = Array.from({ length: 3000 }, () => r.uniform()), testN = Array.from({ length: 3000 }, () => r.normal(0, NOISE));
  const grid = Array.from({ length: 361 }, (_, i) => -3 + 9.5 * i / 360), gridY = grid.map(model);
  const T = 4000, xs = Array.from({ length: T }, (_, i) => -2 + 7 * i / (T - 1));
  const scores = xs.map(x => Math.abs(f(x) + r.normal(0, NOISE) - model(x)));
  return { model, cal, testU, testN, grid, gridY, xs, scores, trainX: Xtr.slice(0, 400), trainY: Ytr.slice(0, 400) };
})();
function drawConformal() {
  const a = +$("#cA").value, c = +$("#cS").value, q = conformalQ(CF.cal, a);
  $("#cAV").textContent = `${fmt(a, 2)} (target coverage ${Math.round(100 * (1 - a))}%)`; $("#cSV").textContent = `[${fmt(c - 2, 1)}, ${fmt(c + 2, 1)}]`;
  const tx = CF.testU.map(u => c - 2 + 4 * u), ty = tx.map((x, i) => f(x) + CF.testN[i]);
  const cov = tx.map((x, i) => Math.abs(ty[i] - CF.model(x)) <= q);
  const share = cov.filter(Boolean).length / cov.length;
  const cv = $("#cfC"), fr = frame(cv, narrowOf(cv) ? 1.3 : 2.4, { x: [-3, 6.5], y: [-2.2, 2.2], xlabel: "input x", yfmt: v => fmt(v, 0) });
  const { ctx, X, Y, pad, ih } = fr;
  ctx.fillStyle = alpha("--ink", 0.05); ctx.fillRect(X(-2), pad.t, X(2) - X(-2), ih);
  ctx.font = "600 10px " + css("--font-ui"); ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.fillText("trained and calibrated here", X(0), pad.t + 12);
  const gy = CF.gridY.map(v => clamp(v, -5, 5));
  plotLine(fr, CF.grid.concat(CF.grid.slice().reverse()), gy.map(v => v + q).concat(gy.map(v => v - q).reverse()), { fill: alpha("--path", 0.18) });
  plotLine(fr, CF.grid, CF.grid.map(f), { color: css("--ink"), width: 1.2, dash: [5, 4] });
  plotLine(fr, CF.grid, gy, { color: css("--path"), width: 2.2 });
  ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t, fr.iw, ih); ctx.clip();
  for (let i = 0; i < 300; i++) { ctx.fillStyle = alpha(cov[i] ? "--c4" : "--geom", 0.75); ctx.beginPath(); ctx.arc(X(tx[i]), Y(ty[i]), 2.4, 0, 2 * Math.PI); ctx.fill(); }
  ctx.restore();
  const inside = c - 2 >= -2 - 1e-9 && c + 2 <= 2 + 1e-9;
  caption($("#cfCap"),
    `With [[\\alpha = ${fmt(a, 2)}]] and 800 calibration scores, [[q]] is the ${ordinal(Math.ceil(801 * (1 - a)))} smallest: <b>${fmt(q, 3)}</b>. On 3,000 test points from [${fmt(c - 2, 1)}, ${fmt(c + 2, 1)}], the band covers <b>${Math.round(1000 * share) / 10}%</b>` +
    (inside ? `, against a guarantee of at least ${Math.round(100 * (1 - a))}% on average over calibration sets (any one set varies by about 1%).`
      : `: the test points are no longer exchangeable with the calibration points, and the guarantee no longer applies. ${share < 1 - a - 0.05 ? "The model extrapolates badly where it never saw data, and its constant-width band cannot know." : ""}`));
}
["#cA", "#cS"].forEach(id => $(id).addEventListener("input", drawConformal));

let FIXED = null;
function drawAci() {
  const gamma = 10 ** +$("#aG").value;
  $("#aGV").textContent = fmt(gamma, gamma < 0.01 ? 4 : 3);
  FIXED = FIXED || onlineConformal(CF.scores, 0.1);
  const aci = onlineConformal(CF.scores, 0.1, { adaptive: true, gamma });
  const run = c => { let s = 0; return Array.from(c, (v, i) => (s += v) / (i + 1)); };
  const xs = CF.xs, narrow = narrowOf($("#aciCov"));
  const fr = frame($("#aciCov"), narrow ? 1.25 : 1.15, { x: [-2, 5], y: [0.5, 1.02], xlabel: "input x (drifting)", yticks: [0.5, 0.75, 0.9, 1], yfmt: v => fmt(v, 2) });
  const shadeOod = F => { F.ctx.fillStyle = alpha("--geom", 0.07); F.ctx.fillRect(F.X(2), F.pad.t, F.X(5) - F.X(2), F.ih); };
  shadeOod(fr);
  fr.ctx.strokeStyle = css("--ink"); fr.ctx.lineWidth = 1.2; fr.ctx.setLineDash([5, 4]); seg(fr.ctx, fr.X(-2), fr.Y(0.9), fr.X(5), fr.Y(0.9)); fr.ctx.setLineDash([]);
  plotLine(fr, xs, run(FIXED.covered), { color: css("--geom"), width: 2 });
  plotLine(fr, xs, run(aci.covered), { color: css("--c4"), width: 2 });
  const qs = (o) => Array.from(o.q, v => (Number.isFinite(v) ? v : NaN));
  const wmax = Math.max(1.6, ...qs(FIXED).filter(Number.isFinite), ...qs(aci).filter(Number.isFinite)) * 1.05;
  const gr = frame($("#aciW"), narrow ? 1.25 : 1.15, { x: [-2, 5], y: [0, Math.min(wmax, 3)], xlabel: "input x (drifting)", yfmt: v => fmt(v, 1) });
  shadeOod(gr);
  const valid = i => i >= 50;
  plotLine(gr, xs.filter((_, i) => valid(i)), Array.from(FIXED.q).filter((_, i) => valid(i)), { color: css("--geom"), width: 1.8 });
  plotLine(gr, xs.filter((_, i) => valid(i)), Array.from(aci.q).filter((_, i) => valid(i)).map(v => Math.min(v, 3)), { color: css("--c4"), width: 1.8 });
  const mean = (arr, pred) => { let s = 0, n = 0; arr.forEach((v, i) => { if (pred(i)) { s += v; n++; } }); return s / n; };
  const lr = o => mean(o.covered, i => i >= 100), ood = o => mean(o.covered, i => xs[i] > 2);
  const neg = Array.from(aci.at).filter((v, i) => i >= 50 && v <= 0).length / (aci.at.length - 50);
  const qIn = mean(aci.q, i => i >= 50 && Math.abs(xs[i]) < 0.05), qEnd = aci.q[aci.q.length - 1];
  caption($("#aciCap"),
    `Long-run coverage: fixed level <b>${fmt(lr(FIXED), 2)}</b>, ACI <b>${fmt(lr(aci), 2)}</b> (target 0.90). Once the input leaves the training domain (shaded), the fixed level covers ${Math.round(100 * ood(FIXED))}% of the points, ACI ${Math.round(100 * ood(aci))}%. ` +
    `ACI pays with width: its half-width grows from about ${fmt(qIn, 2)} in the domain to ${fmt(qEnd, 2)} at the end. ` +
    (gamma < 0.005 ? "With so small a rate, it adapts too slowly to keep up with the drift." : gamma > 0.08 ? "With so large a rate, the level swings hard from step to step." : `Its level [[\\alpha_t]] was at or below zero on ${Math.round(100 * neg)}% of the steps.`));
}
$("#aG").addEventListener("input", drawAci);

/* =====================================================================
   start-up
   ===================================================================== */
updateHero(); drawConformal(); drawAci(); runArl();
onRedraw(() => { updateHero(); drawArl(); drawConformal(); drawAci(); });
})();
