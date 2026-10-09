/* Chapter 08, Bayesian linear regression: the live demos.
   The 25 days are the notebook's simulated data (numpy seed 8), copied exactly; the
   widgets recompute the same posterior in the browser. Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, caption, frame, plotLine, alpha, onRedraw, narrowOf } = window.Kit;
const { pdf, makeRng, linspace, tCdf, tQuantile, quantile } = window.Stats;
const rng = makeRng();

const TEMP = [19.2, 33.7, 19.0, 29.3, 31.1, 20.6, 21.6, 20.2, 14.4, 22.5, 17.3, 17.7, 16.1, 16.3, 29.9, 21.3, 17.6, 25.0, 25.3, 26.2, 32.0, 15.3, 20.2, 18.3, 12.4];
const DRINKS = [104, 170, 108, 130, 179, 102, 90, 110, 102, 115, 116, 118, 82, 64, 140, 130, 89, 114, 118, 138, 164, 99, 123, 91, 74];
const CURVED = [108, 204, 113, 142, 199, 104, 91, 112, 124, 115, 126, 126, 96, 77, 154, 131, 98, 115, 120, 141, 188, 117, 125, 98, 108];  // + 0.3 (x - 23)^2, rounded

/* ---------------- simple linear regression with the reference prior ---------------- */
function fit(xs, ys) {
  const n = xs.length;
  let sx = 0, sxx = 0, sy = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sxx += xs[i] * xs[i]; sy += ys[i]; sxy += xs[i] * ys[i]; }
  const det = n * sxx - sx * sx;
  const inv = [sxx / det, -sx / det, n / det];           // (X^T X)^-1 as [a, b, c] for [[a, b], [b, c]]
  const b1 = (n * sxy - sx * sy) / det, b0 = (sy - b1 * sx) / n;
  let rss = 0;
  for (let i = 0; i < n; i++) rss += (ys[i] - b0 - b1 * xs[i]) ** 2;
  const nu = n - 2, s2 = rss / nu;
  return { n, b0, b1, inv, rss, nu, s2, se0: Math.sqrt(s2 * inv[0]), se1: Math.sqrt(s2 * inv[2]), det };
}
/* exact draws by composition: sigma^2, then (b0, b1) given sigma^2 */
function sample(f, S) {
  const L11 = Math.sqrt(f.inv[0]), L21 = f.inv[1] / L11, L22 = Math.sqrt(Math.max(0, f.inv[2] - L21 * L21));
  const b0 = new Float64Array(S), b1 = new Float64Array(S), sig = new Float64Array(S);
  for (let s = 0; s < S; s++) {
    const sigma = Math.sqrt(1 / rng.gamma(f.nu / 2, f.rss / 2));
    const z1 = rng.normal(), z2 = rng.normal();
    sig[s] = sigma; b0[s] = f.b0 + sigma * L11 * z1; b1[s] = f.b1 + sigma * (L21 * z1 + L22 * z2);
  }
  return { b0, b1, sig };
}

/* =====================================================================
   THE DEMO: CLICK TO ADD OR REMOVE DAYS
   ===================================================================== */
const XR = [10, 36];
const demo = { xs: TEMP.slice(), ys: DRINKS.slice(), fit: null, draws: null, frame: null, msg: "" };
const GX = linspace(XR[0], XR[1], 60);
/* Draws and bands are recomputed only when the data change; resizing just redraws them. */
function updateDemo() {
  demo.fit = fit(demo.xs, demo.ys);
  const d = demo.draws = sample(demo.fit, 4000), S = d.b0.length;
  demo.bands = GX.map(x => {
    const m = new Float64Array(S), p = new Float64Array(S);
    for (let s = 0; s < S; s++) { m[s] = d.b0[s] + d.b1[s] * x; p[s] = m[s] + d.sig[s] * rng.normal(); }
    m.sort(); p.sort();
    return [m[Math.floor(0.025 * S)], m[Math.floor(0.975 * S)], p[Math.floor(0.025 * S)], p[Math.floor(0.975 * S)]];
  });
  drawDemo();
}
function drawDemo() {
  const f = demo.fit, d = demo.draws, bands = demo.bands, gx = GX;
  const showPred = $("#rPred").checked, showCred = $("#rCred").checked, showLines = $("#rLines").checked;
  const lo = Math.min(...demo.ys, ...(showPred ? bands.map(b => b[2]) : bands.map(b => b[0]))), hi = Math.max(...demo.ys, ...(showPred ? bands.map(b => b[3]) : bands.map(b => b[1])));
  const pad = (hi - lo) * 0.06;
  const canvas = $("#rC");
  demo.frame = frame(canvas, narrowOf(canvas) ? 1.25 : 1.9, { x: XR, y: [lo - pad, hi + pad], xlabel: "daily high (°C)", ylabel: "drinks", yfmt: v => String(Math.round(v)) });
  const fr = demo.frame, { ctx, X, Y } = fr;
  const poly = (loI, hiI, color) => {
    ctx.beginPath();
    gx.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(bands[i][hiI])) : ctx.moveTo(X(x), Y(bands[i][hiI]))));
    for (let i = gx.length - 1; i >= 0; i--) ctx.lineTo(X(gx[i]), Y(bands[i][loI]));
    ctx.closePath(); ctx.fillStyle = color; ctx.fill();
  };
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  if (showPred) poly(2, 3, alpha("--post", 0.12));
  if (showCred) poly(0, 1, alpha("--post", 0.32));
  if (showLines) {
    ctx.strokeStyle = alpha("--post", 0.16); ctx.lineWidth = 1;
    for (let s = 0; s < 60; s++) { ctx.beginPath(); ctx.moveTo(X(XR[0]), Y(d.b0[s] + d.b1[s] * XR[0])); ctx.lineTo(X(XR[1]), Y(d.b0[s] + d.b1[s] * XR[1])); ctx.stroke(); }
  }
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 2.2;
  ctx.beginPath(); ctx.moveTo(X(XR[0]), Y(f.b0 + f.b1 * XR[0])); ctx.lineTo(X(XR[1]), Y(f.b0 + f.b1 * XR[1])); ctx.stroke();
  ctx.restore();
  demo.xs.forEach((x, i) => { ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(x), Y(demo.ys[i]), 3.6, 0, 2 * Math.PI); ctx.fill(); });
  // readouts (analytic t)
  const tq = tQuantile(0.975, f.nu);
  $("#rBig").textContent = `${fmt(f.b1, 2)} (${fmt(f.b1 - tq * f.se1, 2)} to ${fmt(f.b1 + tq * f.se1, 2)})`;
  $("#rN").textContent = f.n;
  $("#rSlope").textContent = `${fmt(f.b1, 2)} ± ${fmt(tq * f.se1, 2)}`;
  $("#rInt").textContent = `${fmt(f.b0, 2)} ± ${fmt(tq * f.se0, 2)}`;
  $("#rSig").textContent = fmt(quantile(d.sig, 0.5), 2);
  $("#rP4").textContent = fmt(1 - tCdf((4 - f.b1) / f.se1, f.nu), 3);
  const changed = f.n !== 25 || demo.xs.some((x, i) => x !== TEMP[i] || demo.ys[i] !== DRINKS[i]);
  $("#rStatus").innerHTML = (demo.msg ? demo.msg + " " : "") + (changed
    ? `With your ${f.n} days, each extra °C sells <b>${fmt(f.b1, 2)}</b> drinks (95%: ${fmt(f.b1 - tq * f.se1, 2)} to ${fmt(f.b1 + tq * f.se1, 2)}). The true slope is 4.5.`
    : `Each thin line is one draw from the posterior; the thick line is least squares. The 95% interval for the slope, 3.11 to 5.20, is exactly the classical t interval. The true slope, 4.5, is inside.`);
  demo.msg = "";
}
$("#rC").addEventListener("click", ev => {
  const fr = demo.frame, r = $("#rC").getBoundingClientRect();
  const px = ev.clientX - r.left, py = ev.clientY - r.top;
  if (px < fr.pad.l || px > fr.w - fr.pad.r || py < fr.pad.t || py > fr.pad.t + fr.ih) return;
  const hit = demo.xs.findIndex((x, i) => Math.hypot(fr.X(x) - px, fr.Y(demo.ys[i]) - py) < 9);
  if (hit >= 0) {
    if (demo.xs.length <= 3) { demo.msg = "<b>At least 3 days are needed</b> to fit a line and estimate the noise."; drawDemo(); return; }
    demo.xs.splice(hit, 1); demo.ys.splice(hit, 1);
  } else {
    if (demo.xs.length >= 200) return;
    const x = fr.x0 + (px - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), y = fr.y0 + (fr.pad.t + fr.ih - py) / fr.ih * (fr.y1 - fr.y0);
    demo.xs.push(Math.round(x * 10) / 10); demo.ys.push(Math.round(y));
  }
  updateDemo();
});
["#rLines", "#rCred", "#rPred"].forEach(id => $(id).addEventListener("change", drawDemo));
$("#rReset").addEventListener("click", () => { demo.xs = TEMP.slice(); demo.ys = DRINKS.slice(); updateDemo(); });
$("#rAgain").addEventListener("click", updateDemo);

/* =====================================================================
   INTERCEPT AND SLOPE, AND A DERIVED QUANTITY (the notebook's data)
   ===================================================================== */
const NB = fit(TEMP, DRINKS), XBAR = TEMP.reduce((a, b) => a + b, 0) / TEMP.length;
let nbDraws = sample(NB, 4000);
function drawJoint() {
  const centred = $("#jCentre").checked;
  const { b0, b1 } = nbDraws, S = b0.length;
  const a = Array.from(b0, (v, i) => (centred ? v + b1[i] * XBAR : v));
  const aspect = narrowOf($("#jC")) ? 1.3 : 1.15;
  const ax = [quantile(a, 0.002), quantile(a, 0.998)], bx = [quantile(b1, 0.002), quantile(b1, 0.998)];
  const fr = frame($("#jC"), aspect, { x: ax, y: bx, xlabel: centred ? "intercept at 21.7 °C" : "intercept β₀", ylabel: "slope β₁", yfmt: v => fmt(v, 1), xfmt: v => String(Math.round(v)) });
  fr.ctx.fillStyle = alpha("--post", 0.35);
  for (let i = 0; i < 2500; i++) fr.ctx.fillRect(fr.X(a[i]) - 1, fr.Y(b1[i]) - 1, 2, 2);
  const corr = (() => { const ma = a.reduce((s, v) => s + v, 0) / S, mb = b1.reduce((s, v) => s + v, 0) / S; let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < S; i++) { sab += (a[i] - ma) * (b1[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b1[i] - mb) ** 2; } return sab / Math.sqrt(saa * sbb); })();
  // temperature for 150 expected drinks
  const t150 = Array.from(b0, (v, i) => (150 - v) / b1[i]);
  const lo = 24, hi = 40, B = 48, w = (hi - lo) / B, counts = new Array(B).fill(0);
  for (const v of t150) { const i = Math.floor((v - lo) / w); if (i >= 0 && i < B) counts[i]++; }
  const dens = counts.map(c => c / (S * w));
  const gr = frame($("#tC"), aspect, { x: [lo, hi], y: [0, Math.max(...dens) * 1.12], yticks: false, xlabel: "°C" });
  dens.forEach((dd, i) => { gr.ctx.fillStyle = alpha("--post", 0.7); gr.ctx.fillRect(gr.X(lo + i * w) + 0.4, gr.Y(dd), Math.max(0.8, gr.X(lo + (i + 1) * w) - gr.X(lo + i * w) - 0.8), gr.Y(0) - gr.Y(dd)); });
  caption($("#jCap"),
    `${centred ? "Centred" : "Uncentred"}: the posterior correlation between intercept and slope is <b>${fmt(corr, 2)}</b>` +
    (centred ? `, and the intercept is now the expected sales at the average temperature, about ${fmt(NB.b0 + NB.b1 * XBAR, 1)} drinks. ` : `: a steeper line needs a lower intercept. `) +
    `Right: the temperature at which expected sales reach 150 drinks, computed for each draw: median ${fmt(quantile(t150, 0.5), 1)} °C, 95% interval (${fmt(quantile(t150, 0.025), 1)}, ${fmt(quantile(t150, 0.975), 1)}) °C.`);
}
$("#jCentre").addEventListener("change", drawJoint);

/* =====================================================================
   CREDIBLE VERSUS PREDICTIVE AT ONE TEMPERATURE
   ===================================================================== */
function drawBand() {
  const x = +$("#bX").value;
  $("#bXV").textContent = `${fmt(x, 1)} °C`;
  const m = NB.b0 + NB.b1 * x;
  const q = NB.inv[0] + 2 * NB.inv[1] * x + NB.inv[2] * x * x;          // x^T (X^T X)^-1 x
  const scM = Math.sqrt(NB.s2 * q), scP = Math.sqrt(NB.s2 * (1 + q));
  const ys = linspace(m - 4.5 * scP, m + 4.5 * scP, 500);
  const tM = ys.map(y => pdf.t((y - m) / scM, NB.nu) / scM), tP = ys.map(y => pdf.t((y - m) / scP, NB.nu) / scP);
  const canvas = $("#bC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.6 : 2.8, { x: [ys[0], ys[ys.length - 1]], y: [0, Math.max(...tM) * 1.1], yticks: false, xlabel: "drinks", xfmt: v => String(Math.round(v)) });
  plotLine(fr, ys, tP, { color: css("--lik"), width: 2, fill: alpha("--lik", 0.15) });
  plotLine(fr, ys, tM, { color: css("--post"), width: 2.4, fill: alpha("--post", 0.2) });
  const tq = tQuantile(0.975, NB.nu);
  const p170 = 1 - tCdf((170 - m) / scP, NB.nu);
  caption($("#bCap"),
    `At ${fmt(x, 1)} °C, the expected sales are between <b>${fmt(m - tq * scM, 1)}</b> and <b>${fmt(m + tq * scM, 1)}</b> drinks (95%, blue), ` +
    `but a single day's sales are between <b>${fmt(m - tq * scP, 1)}</b> and <b>${fmt(m + tq * scP, 1)}</b> (95%, orange). ` +
    `[[P(\\text{more than 170 drinks}) = ${fmt(p170, 3)}]]. Both are t distributions with ${NB.nu} degrees of freedom; the predictive one adds the day-to-day noise [[s^2]] to the variance.`);
}
$("#bX").addEventListener("input", drawBand);

/* =====================================================================
   RIDGE: THE PENALTY IS A PRIOR
   ===================================================================== */
const XC = TEMP.map(x => x - XBAR);
function ridge(lam, k) {
  let n = k, sx = 0, sxx = 0, sy = 0, sxy = 0;
  for (let i = 0; i < k; i++) { sx += XC[i]; sxx += XC[i] * XC[i]; sy += DRINKS[i]; sxy += XC[i] * DRINKS[i]; }
  const a = 1e-8 + n, b = sx, c = lam + sxx, det = a * c - b * b;
  return { b0: (c * sy - b * sxy) / det, b1: (a * sxy - b * sy) / det };   // (V0^-1 + X^T X)^-1 X^T y
}
const LAMS = linspace(-3, 4, 200);
function drawRidge() {
  const e = +$("#lam").value, lam = 10 ** e;
  $("#lamV").textContent = lam >= 100 ? fmt(lam, 0) : lam >= 1 ? fmt(lam, 1) : lam.toPrecision(2);
  const aspect = narrowOf($("#lamC")) ? 1.3 : 1.15;
  const s25 = LAMS.map(l => ridge(10 ** l, 25).b1), s5 = LAMS.map(l => ridge(10 ** l, 5).b1);
  const fr = frame($("#lamC"), aspect, { x: [-3, 4], y: [-0.3, 5.3], xticks: [-3, -1, 1, 3], xfmt: v => `10^${v}`, yfmt: v => fmt(v, 0), xlabel: "λ" });
  plotLine(fr, LAMS, s25, { color: css("--post"), width: 2.2 });
  plotLine(fr, LAMS, s5, { color: css("--warm"), width: 2.2, dash: [6, 4] });
  const r25 = ridge(lam, 25), r5 = ridge(lam, 5);
  [[r25.b1, "--post"], [r5.b1, "--warm"]].forEach(([v, t]) => { fr.ctx.fillStyle = css(t); fr.ctx.beginPath(); fr.ctx.arc(fr.X(e), fr.Y(v), 4.5, 0, 2 * Math.PI); fr.ctx.fill(); });
  fr.ctx.strokeStyle = css("--muted"); fr.ctx.lineWidth = 1; fr.ctx.beginPath(); fr.ctx.moveTo(fr.X(e), fr.pad.t); fr.ctx.lineTo(fr.X(e), fr.Y(-0.3)); fr.ctx.stroke();
  const gr = frame($("#lamL"), aspect, { x: XR, y: [50, 190], xlabel: "°C", yfmt: v => String(Math.round(v)) });
  TEMP.forEach((x, i) => { gr.ctx.fillStyle = i < 5 ? css("--warm") : alpha("--ink", 0.55); gr.ctx.beginPath(); gr.ctx.arc(gr.X(x), gr.Y(DRINKS[i]), i < 5 ? 4.2 : 3, 0, 2 * Math.PI); gr.ctx.fill(); });
  const line = (r, tok, dash) => plotLine(gr, XR, XR.map(x => r.b0 + r.b1 * (x - XBAR)), { color: css(tok), width: 2.2, dash });
  line(r25, "--post"); line(r5, "--warm", [6, 4]);
  caption($("#lamCap"),
    `At λ = ${$("#lamV").textContent}, the posterior mean slope is <b>${fmt(r25.b1, 2)}</b> with all 25 days (blue) and <b>${fmt(r5.b1, 2)}</b> with only the first 5 (orange, dashed; those days are the orange dots). ` +
    (lam < 0.1 ? "The prior is weak: both are close to least squares." : lam > 300 ? "The prior insists the slope is near 0, and both lines go flat." : "Tighten it further: the 5-day slope gives way first."));
}
$("#lam").addEventListener("input", drawRidge);

/* =====================================================================
   POSTERIOR PREDICTIVE CHECKS
   ===================================================================== */
const TEMP2 = TEMP.map(x => x * x);
function stats(ys) {
  const f = fit(TEMP, ys);
  const r = ys.map((y, i) => y - f.b0 - f.b1 * TEMP[i]);
  const maxAbs = Math.max(...r.map(Math.abs));
  const mr = r.reduce((a, b) => a + b, 0) / r.length, mt = TEMP2.reduce((a, b) => a + b, 0) / TEMP2.length;
  let srt = 0, srr = 0, stt = 0;
  for (let i = 0; i < r.length; i++) { srt += (r[i] - mr) * (TEMP2[i] - mt); srr += (r[i] - mr) ** 2; stt += (TEMP2[i] - mt) ** 2; }
  return [maxAbs, srt / Math.sqrt(srr * stt)];
}
const ppc = { data: "straight", reps: null, obs: null };
function simulatePpc() {
  const ys = ppc.data === "curved" ? CURVED : DRINKS;
  const f = fit(TEMP, ys), d = sample(f, 2000), reps = [];
  for (let s = 0; s < 2000; s++) {
    const yrep = TEMP.map(x => d.b0[s] + d.b1[s] * x + d.sig[s] * rng.normal());
    reps.push(stats(yrep));
  }
  ppc.reps = reps; ppc.obs = stats(ys);
}
function drawPpc() {
  const shares = [0, 1].map(j => ppc.reps.filter(t => t[j] >= ppc.obs[j]).length / ppc.reps.length);
  [["#ppc1", 0], ["#ppc2", 1]].forEach(([id, j]) => {
    const vals = ppc.reps.map(t => t[j]), obs = ppc.obs[j];
    const lo = Math.min(quantile(vals, 0.002), obs), hi = Math.max(quantile(vals, 0.998), obs);
    const pad = (hi - lo) * 0.06, B = 40, w = (hi - lo + 2 * pad) / B, counts = new Array(B).fill(0);
    for (const v of vals) { const i = Math.floor((v - lo + pad) / w); if (i >= 0 && i < B) counts[i]++; }
    const canvas = $(id);
    const fr = frame(canvas, narrowOf(canvas) ? 1.5 : 1.3, { x: [lo - pad, hi + pad], y: [0, Math.max(...counts) * 1.15], yticks: false, xfmt: v => (j ? fmt(v, 1) : String(Math.round(v))) });
    counts.forEach((c, i) => { fr.ctx.fillStyle = alpha("--prior", 0.75); fr.ctx.fillRect(fr.X(lo - pad + i * w) + 0.4, fr.Y(c), Math.max(0.8, fr.X(lo - pad + (i + 1) * w) - fr.X(lo - pad + i * w) - 0.8), fr.Y(0) - fr.Y(c)); });
    fr.ctx.strokeStyle = css("--warm"); fr.ctx.lineWidth = 2.6;
    fr.ctx.beginPath(); fr.ctx.moveTo(fr.X(obs), fr.Y(0)); fr.ctx.lineTo(fr.X(obs), fr.pad.t); fr.ctx.stroke();
    fr.ctx.font = "600 11px " + css("--font-ui"); fr.ctx.fillStyle = css("--warm"); fr.ctx.textAlign = fr.X(obs) > fr.w * 0.7 ? "right" : "left";
    fr.ctx.fillText(`observed: ${pct(shares[j], shares[j] < 0.01 ? 1 : 0)} beyond`, fr.X(obs) + (fr.X(obs) > fr.w * 0.7 ? -5 : 5), fr.pad.t + 10);
  });
  $$("#ppcw [data-d]").forEach(b => b.setAttribute("aria-pressed", b.dataset.d === ppc.data ? "true" : "false"));
  caption($("#ppcCap"),
    `Grey: the statistic in 2,000 data sets replicated from the posterior predictive. Orange: the real data's value. ` +
    `Share of replicates at least as large: largest residual <b>${fmt(shares[0], 2)}</b>, curvature <b>${fmt(shares[1], shares[1] < 0.01 ? 4 : 2)}</b>. ` +
    (ppc.data === "curved"
      ? (shares[1] < 0.01 ? "The curvature statistic is far out in the tail: the straight-line model cannot produce data this curved. The largest residual sees nothing." : "")
      : "Neither is extreme: nothing here that the model can't reproduce."));
}
$$("#ppcw [data-d]").forEach(b => b.addEventListener("click", () => { ppc.data = b.dataset.d; simulatePpc(); drawPpc(); }));
$("#ppcAgain").addEventListener("click", () => { simulatePpc(); drawPpc(); });

/* =====================================================================
   start-up
   ===================================================================== */
updateDemo(); drawJoint(); drawBand(); drawRidge(); simulatePpc(); drawPpc();
onRedraw(() => { drawDemo(); drawJoint(); drawBand(); drawRidge(); drawPpc(); });
})();
