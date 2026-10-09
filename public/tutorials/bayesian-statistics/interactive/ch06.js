/* Chapter 06, Continuous data: the live demos.
   Numbers quoted in the text come from the chapter's notebook; these widgets recompute
   the same conjugate updates in the browser. Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf } = window.Kit;
const { pdf, lgamma, makeRng, linspace, densityQuantiles, tQuantile } = window.Stats;
const rng = makeRng();

/* ---------------- helpers ---------------- */
const gammaQ = (a, b, ps) => densityQuantiles(x => pdf.gamma(x, a, b), 0, (a + 14 * Math.sqrt(a)) / b, ps, 6001);
const invGammaPdf = (x, a, b) => (x <= 0 ? 0 : Math.exp(a * Math.log(b) - lgamma(a) - (a + 1) * Math.log(x) - b / x));

/* =====================================================================
   WEIGHING A SAMPLE (the demo at the top)
   ===================================================================== */
const NOTEBOOK_READINGS = [10.41, 10.12, 10.63, 10.27, 10.48];
const TRUE_MASS = 10.38;                                // hidden: "take a reading" uses it
const wt = { m0: 10, s0: 0.5, sigma: 0.3, readings: NOTEBOOK_READINGS.slice() };
let wFrame = null;
function weighing() {
  const n = wt.readings.length, ybar = n ? wt.readings.reduce((a, b) => a + b, 0) / n : NaN;
  const p0 = 1 / wt.s0 ** 2, pd = 1 / wt.sigma ** 2, P = p0 + n * pd;
  const m1 = (wt.m0 * p0 + (n ? n * ybar * pd : 0)) / P;
  return { n, ybar, p0, pd, P, m1, s1: Math.sqrt(1 / P) };
}
function drawWeighing() {
  const w = weighing();
  const pts = [wt.m0 - 3.5 * wt.s0, wt.m0 + 3.5 * wt.s0, w.m1 - 4 * w.s1, w.m1 + 4 * w.s1, ...wt.readings];
  let lo = Math.min(...pts) - 0.15, hi = Math.max(...pts) + 0.15;
  lo = Math.max(lo, 5); hi = Math.min(hi, 15);
  const xs = linspace(lo, hi, 700);
  const prior = xs.map(x => pdf.normal(x, wt.m0, wt.s0));
  const lik = w.n ? xs.map(x => pdf.normal(x, w.ybar, wt.sigma / Math.sqrt(w.n))) : null;
  const post = xs.map(x => pdf.normal(x, w.m1, w.s1));
  const top = Math.max(...post, ...prior, ...(lik || [0])) * 1.1;
  const canvas = $("#wC");
  wFrame = frame(canvas, narrowOf(canvas) ? 1.5 : 2.5, { x: [lo, hi], y: [0, top], yticks: false, xlabel: "mass μ (g)", pad: { b: 46 } });
  plotLine(wFrame, xs, prior, { color: css("--prior"), width: 1.6, fill: alpha("--prior", 0.25) });
  if (lik) plotLine(wFrame, xs, lik, { color: css("--lik"), width: 1.8, dash: [6, 4] });
  plotLine(wFrame, xs, post, { color: css("--post"), width: 2.6, fill: alpha("--post", 0.18) });
  const { ctx, X, Y } = wFrame;
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.6;
  for (const r of wt.readings) { ctx.beginPath(); ctx.moveTo(X(r), Y(0) + 20); ctx.lineTo(X(r), Y(0) + 30); ctx.stroke(); }

  // the precision bar
  const pc = $("#precC");
  const total = w.P, scaleMax = total * 1.04;
  const fr = frame(pc, narrowOf(pc) ? 4 : 8, { x: [0, scaleMax], y: [0, 1], yticks: false, xfmt: v => fmt(v, 0), pad: { t: 4 } });
  const by = fr.Y(0.85), bh = fr.Y(0.1) - fr.Y(0.85);
  const seg = (a, b, color) => { fr.ctx.fillStyle = color; fr.ctx.fillRect(fr.X(a), by, Math.max(0.5, fr.X(b) - fr.X(a) - (fr.X(b) - fr.X(a) > 4 ? 1 : 0)), bh); };
  seg(0, w.p0, css("--prior"));
  for (let i = 0; i < w.n; i++) seg(w.p0 + i * w.pd, w.p0 + (i + 1) * w.pd, i % 2 ? alpha("--lik", 0.7) : css("--lik"));

  $("#wM0V").textContent = fmt(wt.m0, 2); $("#wS0V").textContent = fmt(wt.s0, 2); $("#wSigV").textContent = fmt(wt.sigma, 2);
  $("#rN").textContent = w.n;
  $("#rYbar").textContent = w.n ? fmt(w.ybar, 3) : "–";
  $("#rM1").textContent = fmt(w.m1, 3);
  $("#rS1").textContent = fmt(w.s1, 3);
  $("#rCI").textContent = `(${fmt(w.m1 - 1.96 * w.s1, 3)}, ${fmt(w.m1 + 1.96 * w.s1, 3)})`;
  $("#rWorth").textContent = fmt(wt.sigma ** 2 / wt.s0 ** 2, 2);
  caption($("#wStatus"), w.n
    ? `Precision: the prior's [[${fmt(w.p0, 1)}]] plus ${w.n} reading${w.n > 1 ? "s" : ""} × [[${fmt(w.pd, 1)}]] = [[${fmt(w.P, 1)}]], so the posterior sd is [[\\frac{1}{\\sqrt{${fmt(w.P, 1)}}} = ${fmt(w.s1, 3)}]] g. ` +
      `The posterior mean ${fmt(w.m1, 3)} weights the supplier's ${fmt(wt.m0, 2)} by ${fmt(w.p0 / w.P, 2)} and the readings' average ${fmt(w.ybar, 3)} by ${fmt(1 - w.p0 / w.P, 2)}.`
    : `No readings yet: the posterior is the supplier's prior, with precision [[1/${fmt(wt.s0, 2)}^2 = ${fmt(w.p0, 1)}]]. Click the chart or take a reading.`);
}
$("#wC").addEventListener("click", ev => {
  if (!wFrame || wt.readings.length >= 200) return;
  const r = $("#wC").getBoundingClientRect();
  const x = wFrame.x0 + (ev.clientX - r.left - wFrame.pad.l) / wFrame.iw * (wFrame.x1 - wFrame.x0);
  if (x < wFrame.x0 || x > wFrame.x1) return;
  wt.readings.push(Math.round(x * 100) / 100);
  drawWeighing();
});
$("#wTake").addEventListener("click", () => { if (wt.readings.length < 200) { wt.readings.push(Math.round(rng.normal(TRUE_MASS, wt.sigma) * 100) / 100); drawWeighing(); } });
$("#wNotebook").addEventListener("click", () => { Object.assign(wt, { m0: 10, s0: 0.5, sigma: 0.3, readings: NOTEBOOK_READINGS.slice() }); syncWeighing(); drawWeighing(); });
$("#wClear").addEventListener("click", () => { wt.readings = []; drawWeighing(); });
$("#wM0").addEventListener("input", e => { wt.m0 = +e.target.value; drawWeighing(); });
$("#wS0").addEventListener("input", e => { wt.s0 = +e.target.value; drawWeighing(); });
$("#wSig").addEventListener("input", e => { wt.sigma = +e.target.value; drawWeighing(); });
function syncWeighing() { $("#wM0").value = wt.m0; $("#wS0").value = wt.s0; $("#wSig").value = wt.sigma; }

/* =====================================================================
   WAITING FOR THE BUS
   ===================================================================== */
const WAITS = [12.5, 7.1, 15.8, 9.3, 21.0, 11.2, 8.4, 14.6, 10.9, 13.7];
const BUS_A0 = 4, BUS_B0 = 40;
let busK = 10;
function drawBus() {
  const used = WAITS.slice(0, busK), tot = used.reduce((a, b) => a + b, 0);
  const a = BUS_A0 + busK, b = BUS_B0 + tot;
  $("#busKV").textContent = busK ? `${busK} (${used.join(", ")} min)` : "none yet";
  const aspect = narrowOf($("#busRate")) ? 1.5 : 1.3;
  // the rate
  const ls = linspace(0.0005, 0.25, 500);
  const pr = ls.map(x => pdf.gamma(x, BUS_A0, BUS_B0)), po = ls.map(x => pdf.gamma(x, a, b));
  const fr = frame($("#busRate"), aspect, { x: [0, 0.25], y: [0, Math.max(...po, ...pr) * 1.1], yticks: false, xlabel: "λ per minute" });
  plotLine(fr, ls, pr, { color: css("--prior"), width: 1.5, fill: alpha("--prior", 0.25) });
  plotLine(fr, ls, po, { color: css("--post"), width: 2.4, fill: alpha("--post", 0.18) });
  // the mean wait, one over lambda: an inverse-gamma
  const ws = linspace(0.2, 40, 600);
  const wpr = ws.map(x => invGammaPdf(x, BUS_A0, BUS_B0)), wpo = ws.map(x => invGammaPdf(x, a, b));
  const gr = frame($("#busWait"), aspect, { x: [0, 40], y: [0, Math.max(...wpo, ...wpr) * 1.25], yticks: false, xlabel: "minutes" });
  plotLine(gr, ws, wpr, { color: css("--prior"), width: 1.5, fill: alpha("--prior", 0.25) });
  plotLine(gr, ws, wpo, { color: css("--post"), width: 2.4, fill: alpha("--post", 0.18) });
  const eInv = b / (a - 1), invE = b / a;
  const mark = (v, color, dash, label, row) => {
    gr.ctx.strokeStyle = color; gr.ctx.lineWidth = 1.6; gr.ctx.setLineDash(dash);
    gr.ctx.beginPath(); gr.ctx.moveTo(gr.X(v), gr.Y(0)); gr.ctx.lineTo(gr.X(v), gr.pad.t + 12 * row + 4); gr.ctx.stroke(); gr.ctx.setLineDash([]);
    gr.ctx.font = "600 10px " + css("--font-ui"); gr.ctx.fillStyle = color;
    const right = gr.X(v) + 4 + gr.ctx.measureText(label).width <= gr.w - gr.pad.r;   // else write it on the left of the line
    gr.ctx.textAlign = right ? "left" : "right";
    gr.ctx.fillText(label, gr.X(v) + (right ? 4 : -4), gr.pad.t + 12 * row + 8);
  };
  mark(eInv, css("--post"), [6, 4], "posterior mean wait", 0);
  mark(invE, css("--lik"), [], "one over the mean rate", 1);
  if (busK) mark(tot / busK, css("--ink"), [2, 3], "sample mean", 2);
  const [l1, l2] = gammaQ(a, b, [0.025, 0.975]);
  const pNext = (b / (b + 20)) ** a, plug = Math.exp(-20 * a / b);
  caption($("#busCap"),
    `Posterior Gamma(${a}, ${fmt(b, 1).replace(/\.0$/, "")}). The mean wait: [[\\mathbb{E}\\big[\\frac{1}{\\lambda}\\big] = ${fmt(eInv, 2)}]] minutes, but [[\\frac{1}{\\mathbb{E}[\\lambda]} = ${fmt(invE, 2)}]]` +
    (busK ? `, and the sample mean is ${fmt(tot / busK, 2)}` : "") +
    `. 95% interval for the mean wait: (${fmt(1 / l2, 1)}, ${fmt(1 / l1, 1)}) minutes. ` +
    `[[P(\\text{next wait} > 20) = ${fmt(pNext, 3)}]], against ${fmt(plug, 3)} for the plug-in exponential.`);
}
$("#busK").addEventListener("input", e => { busK = +e.target.value; drawBus(); });

/* =====================================================================
   BATTERY LIFE: UNKNOWN MEAN AND VARIANCE
   ===================================================================== */
const LIFE = [9.8, 10.4, 11.1, 9.2, 10.9, 10.1, 9.6, 10.7, 11.4, 10.0, 9.9, 10.6];
const B_ALPHA = 2, B_BETA = 1;
const bt = { k: 12, m: 10, w: 1, draws: null, post: null };
function computeBattery() {
  const data = LIFE.slice(0, bt.k), n = data.length;
  const ybar = data.reduce((a, b) => a + b, 0) / n;
  const s2 = data.reduce((a, v) => a + (v - ybar) ** 2, 0) / (n - 1);
  const a1 = B_ALPHA + n / 2;
  const b1 = B_BETA + (n - 1) * s2 / 2 + bt.w * n / (2 * (bt.w + n)) * (ybar - bt.m) ** 2;
  const mStar = (n * ybar + bt.w * bt.m) / (n + bt.w);
  const gam = Math.sqrt(b1 / ((n + bt.w) * a1)), nu = 2 * B_ALPHA + n;
  const N = 20000, mu = new Float64Array(N), sd = new Float64Array(N);
  for (let i = 0; i < N; i++) {                          // composition sampling
    const sig2 = 1 / rng.gamma(a1, b1);
    sd[i] = Math.sqrt(sig2);
    mu[i] = rng.normal(mStar, Math.sqrt(sig2 / (n + bt.w)));
  }
  const tq = tQuantile(0.975, nu), cq = tQuantile(0.975, n - 1);
  bt.post = { n, ybar, s2, a1, b1, mStar, gam, nu, tq, cq };
  bt.draws = { mu, sd };
}
function drawBattery() {
  const { n, ybar, s2, a1, b1, mStar, gam, nu, tq, cq } = bt.post, { mu, sd } = bt.draws;
  $("#bKV").textContent = n; $("#bMV").textContent = fmt(bt.m, 1); $("#bWV").textContent = bt.w;
  const aspect = narrowOf($("#bJoint")) ? 1.3 : 1.15;
  const lo = mStar - 5 * gam, hi = mStar + 5 * gam;
  const sdSorted = Float64Array.from(sd).sort(), sdHi = sdSorted[Math.floor(0.995 * sd.length)] * 1.05;
  const fr = frame($("#bJoint"), aspect, { x: [lo, hi], y: [0, sdHi], xlabel: "μ (hours)", ylabel: "σ", yfmt: v => fmt(v, 1) });
  fr.ctx.fillStyle = alpha("--post", 0.35);
  for (let i = 0; i < 2500; i++) if (mu[i] > lo && mu[i] < hi && sd[i] < sdHi) fr.ctx.fillRect(fr.X(mu[i]) - 1, fr.Y(sd[i]) - 1, 2, 2);
  // marginal of mu: histogram of the draws, the analytic t, and a normal with sigma fixed at the sample sd
  const B = 60, wdt = (hi - lo) / B, counts = new Array(B).fill(0);
  for (const v of mu) { const i = Math.floor((v - lo) / wdt); if (i >= 0 && i < B) counts[i]++; }
  const dens = counts.map(c => c / (mu.length * wdt));
  const xs = linspace(lo, hi, 400);
  const tDens = xs.map(x => pdf.t((x - mStar) / gam, nu) / gam);
  const normSd = Math.sqrt(s2 / (n + bt.w)), nDens = xs.map(x => pdf.normal(x, mStar, normSd));
  const top = Math.max(...dens, ...tDens, ...nDens) * 1.1;
  const gr = frame($("#bMarg"), aspect, { x: [lo, hi], y: [0, top], yticks: false, xlabel: "μ (hours)" });
  gr.ctx.fillStyle = alpha("--post", 0.35);
  dens.forEach((d, i) => gr.ctx.fillRect(gr.X(lo + i * wdt) + 0.3, gr.Y(d), Math.max(0.6, gr.X(lo + (i + 1) * wdt) - gr.X(lo + i * wdt) - 0.6), gr.Y(0) - gr.Y(d)));
  plotLine(gr, xs, nDens, { color: css("--prior"), width: 1.6, dash: [5, 4] });
  plotLine(gr, xs, tDens, { color: css("--post"), width: 2.4 });
  const sorted = Float64Array.from(mu).sort(), q = p => sorted[Math.floor(p * sorted.length)];
  const cl = ybar - cq * Math.sqrt(s2 / n), ch = ybar + cq * Math.sqrt(s2 / n);
  caption($("#bCap"),
    `${n} phones, sample mean ${fmt(ybar, 3)} and sd ${fmt(Math.sqrt(s2), 3)}. The posterior of μ is a t with ${nu} degrees of freedom (blue curve), ` +
    `95% credible interval (${fmt(mStar - tq * gam, 3)}, ${fmt(mStar + tq * gam, 3)}); the 20,000 sampled μ (histogram) give (${fmt(q(0.025), 3)}, ${fmt(q(0.975), 3)}). ` +
    `The classical t interval is (${fmt(cl, 3)}, ${fmt(ch, 3)}). The dashed grey normal pretends σ is known exactly: its tails are too thin. ` +
    `Posterior mean of σ²: ${fmt(b1 / (a1 - 1), 3)}.`);
}
function updateBattery() { computeBattery(); drawBattery(); }
$("#bK").addEventListener("input", e => { bt.k = +e.target.value; updateBattery(); });
$("#bM").addEventListener("input", e => { bt.m = +e.target.value; updateBattery(); });
$("#bW").addEventListener("input", e => { bt.w = +e.target.value; updateBattery(); });
$("#bAgain").addEventListener("click", updateBattery);

/* =====================================================================
   start-up
   ===================================================================== */
syncWeighing();
drawWeighing(); drawBus(); updateBattery();
onRedraw(() => { drawWeighing(); drawBus(); drawBattery(); });
})();
