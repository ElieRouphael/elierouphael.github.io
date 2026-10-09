/* Chapter 09, Monte Carlo and Metropolis: the live demos.
   The five measurements, the A/B posteriors and the golf data are the notebook's, copied
   exactly; the samplers are the notebook's random-walk Metropolis, run in the browser,
   so every run gives slightly different numbers. Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, css, fmt, pct, int, caption, frame, plotLine, histogram, alpha, onRedraw, reduceMotion, narrowOf } = window.Kit;
const { makeRng, linspace, mean, sortedCopy, quantileSorted, ess, rhat } = window.Stats;
const rng = makeRng();

/* =====================================================================
   THE DEMO: A METROPOLIS CHAIN, ONE STEP AT A TIME
   ===================================================================== */
const Y4 = [3.6, 4.9, 3.2, 4.4, 4.1];
const Y1 = Y4.map(v => +(v - 3.04).toFixed(2));        // the same five, shifted to average 1.00
const RANGE = [-1.5, 6.5], MAXS = 20000;
const mh = { y: Y4, prior: "cauchy", step: 1, chain: new Float64Array(MAXS), S: 0, acc: 0, cur: 0, lq: 0, last: null, running: false, single: false, essV: null, essN: 0, essT: 0 };
function logq(mu) {
  let s = 0;
  for (const v of mh.y) s += (v - mu) ** 2;
  return -0.5 * s - (mh.prior === "cauchy" ? Math.log1p(mu * mu) : 0.5 * mu * mu);
}
const priorPdf = mu => (mh.prior === "cauchy" ? 1 / (Math.PI * (1 + mu * mu)) : Math.exp(-0.5 * mu * mu) / Math.sqrt(2 * Math.PI));
/* the exact posterior on the notebook's grid (-2 to 8, 4001 points) */
let exact = null;
function setTarget() {
  const g = linspace(-2, 8, 4001), dg = g[1] - g[0];
  const lq = g.map(logq), top = Math.max(...lq);
  let z = 0, m = 0;
  lq.forEach((l, i) => { const d = Math.exp(l - top); z += d; m += g[i] * d; });
  const dens = x => Math.exp(logq(x) - top) / (z * dg);
  const xs = linspace(RANGE[0], RANGE[1], 400);
  exact = { mean: m / z, dens, xs, ys: xs.map(dens), prior: xs.map(priorPdf) };
}
function resetChain() {
  mh.S = 0; mh.acc = 0; mh.cur = 0; mh.lq = logq(0); mh.last = null; mh.essV = null; mh.single = false;
}
function step() {
  if (mh.S >= MAXS) return false;
  const from = mh.cur, prop = from + mh.step * rng.normal(), lqp = logq(prop);
  const logr = lqp - mh.lq, ok = Math.log(rng.uniform()) < logr;
  if (ok) { mh.cur = prop; mh.lq = lqp; mh.acc++; }
  mh.chain[mh.S++] = mh.cur;
  mh.last = { from, prop, logr, ok };
  return true;
}
const kept = () => mh.chain.subarray(Math.floor(mh.S / 10), mh.S);
function drawDemo() {
  const cv = $("#mhC"), narrow = narrowOf(cv);
  const yTop = Math.max(...exact.ys) * 1.25;
  const fr = frame(cv, narrow ? 1.35 : 2.1, { x: RANGE, y: [0, yTop], yticks: false, xlabel: "μ" });
  const { ctx, X, Y } = fr;
  const k = kept();
  if (k.length >= 100) histogram(fr, k, RANGE[0], RANGE[1], 80, 1 / (k.length * 0.1), alpha("--post", 0.45));
  else {                                        // too few draws for a histogram: one tick per draw
    ctx.strokeStyle = alpha("--post", 0.7); ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (const v of k) { ctx.moveTo(X(v), Y(0)); ctx.lineTo(X(v), Y(0) - 22); }
    ctx.stroke();
  }
  plotLine(fr, exact.xs, exact.prior, { color: css("--prior"), width: 1.4, dash: [5, 4] });
  plotLine(fr, exact.xs, exact.ys, { color: css("--ink"), width: 2 });
  // the exact mean
  ctx.strokeStyle = alpha("--ink", 0.5); ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(exact.mean), Y(0)); ctx.lineTo(X(exact.mean), fr.pad.t); ctx.stroke(); ctx.setLineDash([]);
  // the current point and the last proposal, on the curve
  const onCurve = x => Y(Math.min(yTop, exact.dens(x)));
  const base = Y(0) - 10;
  if (mh.last) {
    const { from, prop, ok } = mh.last, col = css(ok ? "--c4" : "--warm");
    const inside = prop >= RANGE[0] && prop <= RANGE[1];
    const px = Math.min(fr.pad.l + fr.iw - 2, Math.max(fr.pad.l + 2, X(prop)));
    // the jump, as an arc from where the chain was to where it proposed to go
    ctx.strokeStyle = col; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(X(from), base);
    ctx.quadraticCurveTo((X(from) + px) / 2, base - Math.min(60, 12 + Math.abs(px - X(from)) * 0.35), px, base); ctx.stroke();
    if (inside) {
      ctx.beginPath(); ctx.moveTo(px, base); ctx.lineTo(px, onCurve(prop)); ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(px, onCurve(prop), 5, 0, 2 * Math.PI); ctx.fill();
    } else {
      ctx.font = "600 11px " + css("--font-ui"); ctx.fillStyle = col;
      ctx.textAlign = prop > RANGE[1] ? "right" : "left";
      ctx.fillText(`${prop > RANGE[1] ? "" : "← "}${fmt(prop, 1)}${prop > RANGE[1] ? " →" : ""}`, px + (prop > RANGE[1] ? -2 : 2), base - 22);
    }
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(px, base, 3.5, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(X(mh.cur), Y(0)); ctx.lineTo(X(mh.cur), onCurve(mh.cur)); ctx.stroke();
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(mh.cur), onCurve(mh.cur), 5.5, 0, 2 * Math.PI); ctx.fill();
  drawTrace();
  readouts();
}
function drawTrace() {
  const cv = $("#mhT"), narrow = narrowOf(cv);
  const T = Math.max(mh.S, 60);
  const fr = frame(cv, narrow ? 2.1 : 3.6, { x: [0, T], y: RANGE, xlabel: "step", ylabel: "μ", yticks: [0, 2, 4, 6], xfmt: v => int(v) });
  const { ctx, X, Y } = fr;
  const burn = Math.floor(mh.S / 10);
  if (burn) { ctx.fillStyle = alpha("--prior", 0.18); ctx.fillRect(X(0), fr.pad.t, X(burn) - X(0), fr.ih); }
  ctx.strokeStyle = alpha("--ink", 0.5); ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(0), Y(exact.mean)); ctx.lineTo(X(T), Y(exact.mean)); ctx.stroke(); ctx.setLineDash([]);
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  ctx.strokeStyle = css("--post"); ctx.lineWidth = mh.S > 400 ? 0.8 : 1.4; ctx.lineJoin = "round";
  ctx.beginPath();
  if (mh.S <= 2 * fr.iw) {
    ctx.moveTo(X(0), Y(0));
    for (let s = 0; s < mh.S; s++) ctx.lineTo(X(s + 1), Y(mh.chain[s]));
  } else {                                      // one vertical stroke per pixel column
    const cols = Math.ceil(fr.iw), lo = new Float64Array(cols).fill(Infinity), hi = new Float64Array(cols).fill(-Infinity);
    for (let s = 0; s < mh.S; s++) {
      const c = Math.min(cols - 1, Math.floor((s + 1) / T * cols)), v = mh.chain[s];
      if (v < lo[c]) lo[c] = v;
      if (v > hi[c]) hi[c] = v;
    }
    let prev = 0;
    for (let c = 0; c < cols; c++) {
      if (lo[c] === Infinity) continue;
      const x = fr.pad.l + c + 0.5;
      ctx.moveTo(x, Y(Math.max(hi[c], prev))); ctx.lineTo(x, Y(Math.min(lo[c], prev)));
      prev = mh.chain[Math.min(mh.S - 1, Math.floor((c + 1) / cols * T) - 1)];
    }
  }
  ctx.stroke();
  ctx.restore();
}
function readouts() {
  const k = kept();
  $("#rS").textContent = int(mh.S);
  $("#rAcc").textContent = mh.S ? pct(mh.acc / mh.S, 0) : "–";
  $("#rMean").textContent = k.length ? fmt(mean(k), 3) : "–";
  $("#rExact").textContent = fmt(exact.mean, 3);
  const now = performance.now();
  if (k.length >= 20 && (!mh.running || mh.essV == null || now - mh.essT > 250)) { mh.essV = ess(k); mh.essN = k.length; mh.essT = now; }
  $("#rEss").textContent = k.length >= 20 ? `${int(mh.essV)} of ${int(mh.essN)}` : "–";
}
const fmtStep = d => (d >= 10 ? d.toFixed(0) : d >= 1 ? d.toFixed(1) : d.toPrecision(2));
function ratioTex(logr) {
  if (logr > 9.2 || logr < -9.2) return `e^{${fmt(logr, 1)}}`;
  const r = Math.exp(logr);
  return r >= 100 ? fmt(r, 0) : r >= 0.01 ? fmt(r, 2) : r.toPrecision(2);
}
function status() {
  const el = $("#mhStatus");
  if (mh.running) return;                       // read out once the chain stops, not every frame
  if (!mh.S) {
    el.innerHTML = "The chain starts at <b>μ = 0</b>, where the prior is centred, far from the measurements. Press <b>1 step</b> to propose a move, or <b>run</b>.";
    return;
  }
  if (mh.single && mh.last) {
    const { from, prop, logr, ok } = mh.last;
    caption(el,
      `Step ${int(mh.S)}: from μ = ${fmt(from, 2)}, the proposal is μ* = ${fmt(prop, 2)}. ` +
      `[[\\frac{q(\\mu^*)}{q(\\mu)} = ${ratioTex(logr)}]], ` +
      (logr >= 0 ? "above 1 (uphill), so the move is always accepted. "
        : Math.exp(logr) < 1e-4 ? "so the move is practically never accepted (probability below 0.01%). "
        : `so the move is accepted with probability ${pct(Math.exp(logr), Math.exp(logr) < 0.01 ? 2 : 0)}. `) +
      (ok ? "It was <b>accepted</b>." : `It was <b>rejected</b>: the chain stays at μ = ${fmt(from, 2)}, and that value is recorded again.`));
    return;
  }
  const k = kept(), a = mh.acc / mh.S;
  let advice = "";
  if (mh.S >= 300) {
    advice = a > 0.8 ? " Almost every move is accepted because each step is tiny: the chain crawls (look at the trace), and the effective sample size is small. Increase the step size."
      : a < 0.15 ? " Most proposals land where the posterior is negligible and are rejected, so the trace is flat for long stretches. Decrease the step size."
      : " An acceptance rate of 40–50% is about right for one parameter: the chain moves freely.";
  }
  el.innerHTML = `After ${int(mh.S)} steps, the draws after burn-in average <b>${fmt(mean(k), 3)}</b>; the exact posterior mean is <b>${fmt(exact.mean, 3)}</b>.` + advice +
    (mh.S >= MAXS ? " The chain is full; press <b>start over</b> for a new one." : "");
}
function update() { drawDemo(); status(); }
function setRunning(on) {
  mh.running = on;
  $("#mhRun").setAttribute("aria-pressed", on ? "true" : "false");
  $("#mhRun").textContent = on ? "pause" : "run";
  if (on) { mh.lastT = performance.now(); mh.owed = 0; requestAnimationFrame(tick); }
}
/* Speed depends on time, not frame rate: 12 steps a second for the first 30, then the
   chain grows about 3.3 times every second (20,000 steps in about 8 seconds). */
function tick(now) {
  if (!mh.running) return;
  const dt = Math.min(0.5, Math.max(0, (now - mh.lastT) / 1000));
  mh.lastT = now;
  mh.owed += mh.S < 30 ? 12 * dt : mh.S * Math.expm1(1.2 * dt);
  const n = Math.floor(mh.owed);
  mh.owed -= n;
  for (let i = 0; i < n; i++) if (!step()) break;
  if (mh.S >= MAXS) { mh.running = false; setRunning(false); update(); return; }
  drawDemo();
  requestAnimationFrame(tick);
}
$("#mh1").addEventListener("click", () => { setRunning(false); step(); mh.single = true; update(); });
$("#mh100").addEventListener("click", () => { setRunning(false); for (let i = 0; i < 100; i++) step(); mh.single = false; update(); });
$("#mhRun").addEventListener("click", () => {
  mh.single = false;
  if (mh.running) { setRunning(false); update(); return; }
  if (mh.S >= MAXS) resetChain();
  if (reduceMotion) { while (step()); update(); return; }
  setRunning(true);
});
$("#mhReset").addEventListener("click", () => { setRunning(false); resetChain(); update(); });
function restart() { resetChain(); if (mh.running) drawDemo(); else update(); }
$("#mhStep").addEventListener("input", () => {
  mh.step = 10 ** +$("#mhStep").value;
  $("#mhStepV").textContent = fmtStep(mh.step);
  restart();
});
$("#mhPrior").addEventListener("change", () => { mh.prior = $("#mhPrior").value; setTarget(); restart(); });
$("#mhData").addEventListener("change", () => { mh.y = $("#mhData").value === "4" ? Y4 : Y1; setTarget(); restart(); });

/* =====================================================================
   PART 1: MONTE CARLO ERROR ON THE A/B TEST
   ===================================================================== */
const SIZES = [10, 30, 100, 300, 1000, 3000, 10000], P_EXACT = 0.9586;
let mcEst = null, mcS = 0, mcTimer = null;
function simulateMc() {
  clearTimeout(mcTimer);
  const S = mcS = SIZES[+$("#mcS").value];
  $("#mcSV").textContent = int(S);
  mcEst = Array.from({ length: 100 }, () => {
    let hits = 0;
    for (let s = 0; s < S; s++) if (rng.beta(44, 236) > rng.beta(29, 241)) hits++;
    return hits / S;
  });
  drawMc();
}
function drawMc() {
  const S = mcS;
  const cv = $("#mcC"), B = 40, lo = 0.6, hi = 1.0, w = (hi - lo) / B;
  const counts = new Array(B).fill(0);
  for (const v of mcEst) counts[Math.min(B - 1, Math.max(0, Math.floor((v - lo) / w + 1e-9)))]++;
  const fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [lo, hi], y: [0, Math.max(...counts) * 1.15], xticks: [0.6, 0.7, 0.8, 0.9, 1], xfmt: v => fmt(v, 1), xlabel: "estimate of P(B better)", yfmt: v => String(Math.round(v)) });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--post", 0.75);
  counts.forEach((c, i) => { if (c) ctx.fillRect(X(lo + i * w) + 0.3, Y(c), Math.max(1.5, X(lo + (i + 1) * w) - X(lo + i * w) - 0.6), Y(0) - Y(c)); });
  ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2.2;
  ctx.beginPath(); ctx.moveTo(X(P_EXACT), Y(0)); ctx.lineTo(X(P_EXACT), fr.pad.t); ctx.stroke();
  ctx.font = "600 11px " + css("--font-ui"); ctx.fillStyle = css("--warm"); ctx.textAlign = "right";
  ctx.fillText("exact 0.9586", X(P_EXACT) - 5, fr.pad.t + 10);
  const m = mean(mcEst), sd = Math.sqrt(mcEst.reduce((s, v) => s + (v - m) ** 2, 0) / (mcEst.length - 1));
  const theory = Math.sqrt(P_EXACT * (1 - P_EXACT) / S);
  caption($("#mcCap"),
    `With ${int(S)} draws per estimate, the 100 estimates (bars) have a standard deviation of <b>${fmt(sd, sd < 0.01 ? 4 : 3)}</b>; ` +
    `the formula [[\\sqrt{\\frac{p(1-p)}{S}}]] with p = 0.9586 gives <b>${fmt(theory, theory < 0.01 ? 4 : 3)}</b>. ` +
    (S === 10 ? "With only 10 draws, each estimate is a multiple of 0.1. " : "") +
    (S < 10000 ? "Each move to the right multiplies S by about 3 and divides the spread by about √3." : "That is a thousand times more draws than the first setting, and a spread about 32 times smaller."));
}
/* The two largest settings need up to two million beta draws, which takes a noticeable
   moment on a phone: while the slider is dragged, they wait until it rests. */
$("#mcS").addEventListener("input", () => {
  const S = SIZES[+$("#mcS").value];
  clearTimeout(mcTimer);
  if (S <= 1000) { simulateMc(); return; }
  $("#mcSV").textContent = int(S);
  $("#mcCap").textContent = `Drawing 100 estimates of ${int(S)} draws each…`;
  mcTimer = setTimeout(simulateMc, 200);
});
$("#mcAgain").addEventListener("click", simulateMc);

/* =====================================================================
   PART 4: THREE STEP SIZES (the notebook's: 0.05, 1 and 25, from 4.0)
   ===================================================================== */
const TUNE = [0.05, 1, 25];
let tune = null;
(() => {
  const box = $("#tuneBox");
  TUNE.forEach((d, i) => {
    const wrap = document.createElement("div");
    wrap.innerHTML = `<p class="chart-title" id="tuneT${i}"></p><canvas id="tuneC${i}" aria-label="Trace of the chain with step size ${d}"></canvas>`;
    box.appendChild(wrap);
  });
})();
function logq4(mu) {                            // the notebook's target: Cauchy prior, original data
  let s = 0;
  for (const v of Y4) s += (v - mu) ** 2;
  return -0.5 * s - Math.log1p(mu * mu);
}
function simulateTune() {
  tune = TUNE.map(d => {
    const c = new Float64Array(5000);
    let cur = 4, lq = logq4(cur), acc = 0;
    for (let s = 0; s < 5000; s++) {
      const p = cur + d * rng.normal(), lp = logq4(p);
      if (Math.log(rng.uniform()) < lp - lq) { cur = p; lq = lp; acc++; }
      c[s] = cur;
    }
    return { c, acc: acc / 5000, ess: ess(c.subarray(500)) };
  });
  drawTune();
}
function drawTune() {
  tune.forEach((t, i) => {
    const cv = $(`#tuneC${i}`);
    $(`#tuneT${i}`).textContent = `step ${TUNE[i]}: acceptance ${pct(t.acc, 0)}, effective sample size ${int(t.ess)} of 4,500`;
    const fr = frame(cv, narrowOf(cv) ? 2.6 : 5, { x: [0, 5000], y: [1.5, 6.5], yticks: [2, 4, 6], xlabel: i === 2 ? "step" : null, xfmt: v => int(v) });
    plotLine(fr, Array.from({ length: 5000 }, (_, s) => s), t.c, { color: css("--post"), width: 0.7 });
  });
  const [a, b, c] = tune;
  caption($("#tuneCap"),
    `<b>Too small</b> (0.05): ${pct(a.acc, 0)} of moves accepted, but the chain wanders slowly, and its 4,500 kept draws are worth about <b>${int(a.ess)}</b> independent ones. ` +
    `<b>About right</b> (1): ${pct(b.acc, 0)} accepted, worth about <b>${int(b.ess)}</b>. ` +
    `<b>Too large</b> (25): ${pct(c.acc, 0)} accepted; the flat stretches are the chain staying put, worth about <b>${int(c.ess)}</b>.`);
}
$("#tuneAgain").addEventListener("click", simulateTune);

/* =====================================================================
   PART 5: GOLF PUTTS, FOUR CHAINS (and an optional fifth, lost far away)
   ===================================================================== */
const DIST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], MADE = [53, 51, 50, 36, 34, 20, 21, 15, 9, 3], N_ATT = 60;
const STARTS = [[-4, 1], [6, -1], [0, 0], [4, 0.5]], FAR = [20, 5], STEPS = [0.15, 0.015], NMAX = 12000, XC = 11;
const CHAIN_COL = ["--post", "--warm", "--c4", "--c5"];
const logaddexp0 = e => (e > 0 ? e + Math.log1p(Math.exp(-e)) : Math.log1p(Math.exp(e)));
function logqGolf(a, b) {
  let s = 0;
  for (let j = 0; j < DIST.length; j++) { const eta = a + b * DIST[j]; s += MADE[j] * eta - N_ATT * logaddexp0(eta); }
  return s - a * a / 50 - b * b / 2;
}
/* run in (a, b), or centred in (a + 11 b, b); always store a and b */
function golfChain(start, centred) {
  const A = new Float64Array(NMAX), Bv = new Float64Array(NMAX), moved = new Uint8Array(NMAX);
  const lq = (u, b) => logqGolf(centred ? u - XC * b : u, b);
  let u = centred ? start[0] + XC * start[1] : start[0], b = start[1], l = lq(u, b);
  for (let s = 0; s < NMAX; s++) {
    const pu = u + STEPS[0] * rng.normal(), pb = b + STEPS[1] * rng.normal(), lp = lq(pu, pb);
    if (Math.log(rng.uniform()) < lp - l) { u = pu; b = pb; l = lp; moved[s] = 1; }
    A[s] = centred ? u - XC * b : u; Bv[s] = b;
  }
  return { a: A, b: Bv, moved, start };
}
const golf = { centred: false, chains: null, far: null };
function simulateGolf() {
  golf.centred = $("#gCentre").checked;
  golf.chains = STARTS.map(s => golfChain(s, golf.centred));
  golf.far = golfChain(FAR, golf.centred);
  summariseGolf();
}
function summariseGolf() {
  const N = +$("#gN").value, burn = +$("#gB").value, withFar = $("#gFar").checked;
  const chains = withFar ? [...golf.chains, golf.far] : golf.chains;
  const acc = chains.map(c => c.moved.subarray(0, N).reduce((s, v) => s + v, 0) / N);   // over the steps in use
  const keptOf = (c, key) => c[key].subarray(burn, N);
  const param = key => {
    const parts = chains.map(c => keptOf(c, key)), all = sortedCopy(parts.flatMap(p => Array.from(p)));
    return { mean: mean(all), lo: quantileSorted(all, 0.025), hi: quantileSorted(all, 0.975), rhat: rhat(parts), ess: parts.reduce((s, p) => s + ess(p), 0) };
  };
  const pa = param("a"), pb = param("b");
  // intercept at 11 feet (the centred parameter)
  const ac = chains.map(c => { const k = keptOf(c, "a"), kb = keptOf(c, "b"); return k.map((v, i) => v + XC * kb[i]); });
  const acAll = sortedCopy(ac.flatMap(p => Array.from(p)));
  const pc = { mean: mean(acAll), lo: quantileSorted(acAll, 0.025), hi: quantileSorted(acAll, 0.975), rhat: rhat(ac), ess: ac.reduce((s, p) => s + ess(p), 0) };
  // 50% distance, -a/b
  const half = [];
  chains.forEach(c => { const k = keptOf(c, "a"), kb = keptOf(c, "b"); for (let i = 0; i < k.length; i++) half.push(-k[i] / kb[i]); });
  const hs = sortedCopy(half);
  // band of P(made) against distance, from every 20th kept draw
  const xs = linspace(0, 22, 45), curves = xs.map(() => []);
  chains.forEach(c => { const k = keptOf(c, "a"), kb = keptOf(c, "b"); for (let i = 0; i < k.length; i += 20) xs.forEach((x, j) => curves[j].push(1 / (1 + Math.exp(-(k[i] + kb[i] * x))))); });
  const band = curves.map(v => { const s = sortedCopy(v); return [quantileSorted(s, 0.025), quantileSorted(s, 0.5), quantileSorted(s, 0.975)]; });
  golf.summary = { N, burn, withFar, chains, acc, pa, pb, pc, hs, xs, band, total: chains.length * (N - burn) };
  drawGolf();
}
function drawGolf() {
  const g = golf.summary, centred = golf.centred;
  // left: the paths, first 3,000 steps
  const cvP = $("#gPath"), aspect = narrowOf(cvP) ? 1.15 : 1.1, shown = Math.min(3000, g.N);
  const u = (c, s) => (centred ? c.a[s] + XC * c.b[s] : c.a[s]);
  let xlo = Infinity, xhi = -Infinity, ylo = Infinity, yhi = -Infinity;
  g.chains.forEach(c => {
    const s0 = centred ? c.start[0] + XC * c.start[1] : c.start[0];
    xlo = Math.min(xlo, s0); xhi = Math.max(xhi, s0); ylo = Math.min(ylo, c.start[1]); yhi = Math.max(yhi, c.start[1]);
    for (let s = 0; s < shown; s++) { const x = u(c, s), y = c.b[s]; if (x < xlo) xlo = x; if (x > xhi) xhi = x; if (y < ylo) ylo = y; if (y > yhi) yhi = y; }
  });
  const px = (xhi - xlo) * 0.06, py = (yhi - ylo) * 0.08;
  const fr = frame(cvP, aspect, { x: [xlo - px, xhi + px], y: [ylo - py, yhi + py], xlabel: centred ? "intercept at 11 ft" : "a (intercept)", ylabel: "b (slope)", yfmt: v => String(+v.toFixed(2)) });
  const { ctx, X, Y } = fr;
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  g.chains.forEach((c, j) => {
    ctx.strokeStyle = j < 4 ? alpha(CHAIN_COL[j], 0.8) : alpha("--ink", 0.55); ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(X(centred ? c.start[0] + XC * c.start[1] : c.start[0]), Y(c.start[1]));
    for (let s = 0; s < shown; s++) ctx.lineTo(X(u(c, s)), Y(c.b[s]));
    ctx.stroke();
  });
  g.chains.forEach(c => { ctx.fillStyle = css("--ink"); const x = X(centred ? c.start[0] + XC * c.start[1] : c.start[0]), y = Y(c.start[1]); ctx.fillRect(x - 4, y - 4, 8, 8); });
  ctx.restore();
  // right: the chance of making a putt
  const cvC = $("#gCurve");
  const gr = frame(cvC, aspect, { x: [0, 22], y: [0, 1], xlabel: "distance (feet)", yticks: [0, 0.25, 0.5, 0.75, 1], yfmt: v => pct(v, 0) });
  const c2 = gr.ctx;
  c2.beginPath();
  g.xs.forEach((x, i) => (i ? c2.lineTo(gr.X(x), gr.Y(g.band[i][2])) : c2.moveTo(gr.X(x), gr.Y(g.band[i][2]))));
  for (let i = g.xs.length - 1; i >= 0; i--) c2.lineTo(gr.X(g.xs[i]), gr.Y(g.band[i][0]));
  c2.closePath(); c2.fillStyle = alpha("--post", 0.3); c2.fill();
  plotLine(gr, g.xs, g.band.map(b => b[1]), { color: css("--post"), width: 2 });
  DIST.forEach((d, j) => { c2.fillStyle = css("--warm"); c2.beginPath(); c2.arc(gr.X(d), gr.Y(MADE[j] / N_ATT), 4, 0, 2 * Math.PI); c2.fill(); });
  // table
  const row = (name, p) => `<tr><td>${name}</td><td>${fmt(p.mean, 3)}<span class="iv">(${fmt(p.lo, 3)}, ${fmt(p.hi, 3)})</span></td><td>${fmt(p.rhat, 3)}</td><td>${int(p.ess)}</td></tr>`;
  $("#gT tbody").innerHTML = row("a", g.pa) + row("b", g.pb) + row("a + 11b", g.pc);
  const worst = Math.max(g.pa.rhat, g.pb.rhat);
  const verdict = worst < 1.01 ? "All the chains agree: <b>R-hat is below 1.01</b> for both parameters."
    : worst < 1.05 ? `R-hat reaches <b>${fmt(worst, 3)}</b>, above 1.01: the chains roughly agree, but they should run longer.`
    : `R-hat reaches <b>${fmt(worst, 3)}</b>: the chains disagree, and these draws can't be trusted yet.`;
  caption($("#gCap"),
    `Acceptance rates: ${g.acc.map(a => fmt(a, 2)).join(", ")}. ${verdict} ` +
    (g.withFar && g.burn < 1000 && worst >= 1.05 ? "The fifth chain (grey) spends its first few hundred steps walking in from (20, 5), and those steps are still counted. " : "") +
    `The slope's ${int(g.total)} kept draws are worth about <b>${int(g.pb.ess)}</b> independent ones${centred ? " (centred)" : ""}. ` +
    `Bands: 95% credible band of the success probability; dots: the observed rates. ` +
    `The 50% distance [[-\\frac{a}{b}]]: median <b>${fmt(quantileSorted(g.hs, 0.5), 1)} feet</b>, 95% interval (${fmt(quantileSorted(g.hs, 0.025), 1)}, ${fmt(quantileSorted(g.hs, 0.975), 1)}).`);
}
$("#gCentre").addEventListener("change", simulateGolf);
$("#gAgain").addEventListener("click", simulateGolf);
["#gFar", "#gN", "#gB"].forEach(id => $(id).addEventListener("change", summariseGolf));

/* =====================================================================
   start-up
   ===================================================================== */
mh.step = 10 ** +$("#mhStep").value;
$("#mhStepV").textContent = fmtStep(mh.step);
setTarget(); resetChain(); update();
simulateMc(); simulateTune(); simulateGolf();
onRedraw(() => { drawDemo(); drawMc(); drawTune(); drawGolf(); });
})();
