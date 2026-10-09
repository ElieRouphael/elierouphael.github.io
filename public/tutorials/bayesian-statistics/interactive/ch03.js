/* Chapter 03, Likelihood and frequentist inference: the live demos.
   Numbers quoted in the text come from the chapter's notebook; these widgets recompute
   the same quantities in the browser. Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, int, caption, frame, plotLine, alpha, reduceMotion, onRedraw } = window.Kit;
const { pdf, pmf, makeRng, linspace, cumulative } = window.Stats;
const rng = makeRng();
const narrowOf = canvas => canvas.clientWidth < 520;
const Z = { 0.8: 1.2816, 0.9: 1.6449, 0.95: 1.96, 0.99: 2.5758 };

/* the two intervals, for y successes in n trials */
function wald(y, n, z) {
  const t = y / n, se = Math.sqrt(t * (1 - t) / n);
  return [t - z * se, t + z * se];
}
function wilson(y, n, z) {
  const centre = (y + z * z / 2) / (n + z * z);
  const half = z / (n + z * z) * Math.sqrt(y * (n - y) / n + z * z / 4);
  return [centre - half, centre + half];
}
const INTERVALS = { wald, wilson };
/* exact long-run share of intervals that contain theta */
function coverage(theta, n, interval, z = 1.96) {
  let c = 0;
  for (let y = 0; y <= n; y++) {
    const [lo, hi] = interval(y, n, z);
    if (lo <= theta && theta <= hi) c += pmf.binomial(y, n, theta);
  }
  return c;
}

/* =====================================================================
   REPEATED EXPERIMENTS (the demo at the top)
   ===================================================================== */
const CI_NS = [5, 10, 20, 50, 100, 200];
const SHOW = 100;
const ci = { theta: 0.65, nIdx: 2, type: "wald", level: 0.95, runs: [], timer: null };
const ciN = () => CI_NS[ci.nIdx];

function drawCi() {
  const canvas = $("#ciC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.1 : 1.75, { x: [0, 1], y: [0, SHOW + 1], yticks: false, xlabel: "success rate θ", pad: { t: 6 } });
  const { ctx, X, Y } = fr;
  const recent = ci.runs.slice(-SHOW).reverse();          // newest at the top
  const rowH = Math.max(1, Y(0) - Y(1));
  recent.forEach((r, i) => {
    const yy = Y(SHOW - i);
    ctx.strokeStyle = r.hit ? css("--post") : css("--warm");
    ctx.lineWidth = r.hit ? Math.min(2, Math.max(1, rowH * 0.45)) : Math.min(3.2, Math.max(1.6, rowH * 0.8));
    ctx.beginPath(); ctx.moveTo(X(Math.max(0, r.lo)), yy); ctx.lineTo(X(Math.min(1, r.hi)), yy); ctx.stroke();
    if (Math.abs(r.hi - r.lo) < 1e-9) {                     // a collapsed interval: a single point
      ctx.fillStyle = ctx.strokeStyle; ctx.fillRect(X(r.lo) - 1.5, yy - 1.5, 3, 3);
    }
    ctx.fillStyle = css("--ink");
    ctx.beginPath(); ctx.arc(X(r.y / ciN()), yy, Math.min(2.2, Math.max(1, rowH * 0.4)), 0, 2 * Math.PI); ctx.fill();
  });
  ctx.strokeStyle = css("--ink"); ctx.setLineDash([5, 4]); ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(X(ci.theta), fr.pad.t); ctx.lineTo(X(ci.theta), Y(0)); ctx.stroke(); ctx.setLineDash([]);
  ctx.font = "600 11px " + css("--font-ui"); ctx.fillStyle = css("--ink");
  ctx.textAlign = ci.theta > 0.8 ? "right" : "left";
  ctx.strokeStyle = css("--surface"); ctx.lineWidth = 4; ctx.lineJoin = "round";
  ctx.strokeText(`true θ = ${fmt(ci.theta, 2)}`, X(ci.theta) + (ci.theta > 0.8 ? -6 : 6), fr.pad.t + 12);
  ctx.fillText(`true θ = ${fmt(ci.theta, 2)}`, X(ci.theta) + (ci.theta > 0.8 ? -6 : 6), fr.pad.t + 12);

  const N = ci.runs.length, hits = ci.runs.filter(r => r.hit).length;
  const exact = coverage(ci.theta, ciN(), INTERVALS[ci.type], Z[ci.level]);
  $("#ciThetaV").textContent = fmt(ci.theta, 2);
  $("#ciNV").textContent = ciN();
  $("#rExp").textContent = int(N);
  $("#rHit").textContent = N ? `${int(hits)} (${pct(hits / N, 1)})` : "–";
  $("#rProm").textContent = pct(ci.level, 0);
  $("#rExact").textContent = pct(exact, 1);
  const name = ci.type === "wald" ? "Wald" : "Wilson";
  let msg;
  if (!N) msg = `The true success rate is fixed at <b>${fmt(ci.theta, 2)}</b>. Press <b>repeat once</b>: the player throws ${ciN()} times, and we compute a ${pct(ci.level, 0)} ${name} interval from the result.`;
  else if (N === 1) {
    const r = ci.runs[0];
    msg = `She made ${r.y} of ${ciN()}: the interval is (${fmt(Math.max(0, r.lo), 2)}, ${fmt(Math.min(1, r.hi), 2)}), and it ${r.hit ? "<b>caught</b>" : "<b>missed</b>"} the true rate. One interval tells you nothing about how reliable the method is. Repeat it many times.`;
  } else {
    msg = `<b>${int(hits)}</b> of ${int(N)} intervals caught the true rate: <b>${pct(hits / N, 1)}</b>. The promise is ${pct(ci.level, 0)}; the exact long-run rate for this setting is ${pct(exact, 1)}.`;
    if (exact < ci.level - 0.03) {
      const small = ciN() < 30, edge = ci.theta < 0.15 || ci.theta > 0.85;
      const why = small && edge ? "so few throws and a rate this close to the edge" : small ? "so few throws" : edge ? "a rate this close to the edge" : "this setting";
      msg += ` The promise is broken here: the normal approximation behind the ${name} interval fails for ${why}${ci.type === "wald" ? ". Try the Wilson interval" : ""}.`;
    }
    else msg += " Each interval either caught it or not; the percentage describes the method, not any one interval.";
  }
  $("#ciStatus").innerHTML = msg;
}
function runOnce() {
  const n = ciN(), y = rng.binomial(n, ci.theta);
  const [lo, hi] = INTERVALS[ci.type](y, n, Z[ci.level]);
  ci.runs.push({ y, lo, hi, hit: lo <= ci.theta && ci.theta <= hi });
}
function repeat(k) {
  clearInterval(ci.timer);
  if (reduceMotion || k === 1) { for (let i = 0; i < k; i++) runOnce(); drawCi(); return; }
  let left = k;
  ci.timer = setInterval(() => {
    for (let i = 0; i < 5 && left > 0; i++, left--) runOnce();
    drawCi();
    if (left <= 0) clearInterval(ci.timer);
  }, 25);
}
function resetCi() { clearInterval(ci.timer); ci.runs = []; drawCi(); }
$("#ciTheta").addEventListener("input", e => { ci.theta = +e.target.value / 100; resetCi(); });
$("#ciN").addEventListener("input", e => { ci.nIdx = +e.target.value; resetCi(); });
$("#ciType").addEventListener("change", e => { ci.type = e.target.value; resetCi(); });
$("#ciLevel").addEventListener("change", e => { ci.level = +e.target.value; resetCi(); });
$("#ciOne").addEventListener("click", () => repeat(1));
$("#ciHundred").addEventListener("click", () => repeat(100));
$("#ciClear").addEventListener("click", resetCi);

/* =====================================================================
   TWO READINGS OF ONE FORMULA
   ===================================================================== */
const lik = { theta: 0.5, n: 20, y: 13, log: false };
function drawLik() {
  const { theta, n, y } = lik;
  $("#likY").max = n;
  if (lik.y > n) { lik.y = n; }
  $("#likY").value = lik.y;
  $("#likThetaV").textContent = fmt(theta, 2);
  $("#likNV").textContent = n;
  $("#likYV").textContent = lik.y;
  const aspect = narrowOf($("#likBars")) ? 1.45 : 1.25;
  // left: the distribution of the number of makes for this theta
  const ks = Array.from({ length: n + 1 }, (_, k) => k), ps = ks.map(k => pmf.binomial(k, n, theta));
  const fr = frame($("#likBars"), aspect, { x: [-0.7, n + 0.7], y: [0, Math.max(...ps) * 1.15], yfmt: v => fmt(v, 2), xlabel: "makes" });
  const bw = Math.max(1, (fr.X(1) - fr.X(0)) * 0.72);
  ks.forEach((k, i) => {
    fr.ctx.fillStyle = k === lik.y ? css("--warm") : alpha("--prior", 0.8);
    fr.ctx.fillRect(fr.X(k) - bw / 2, fr.Y(ps[i]), bw, fr.Y(0) - fr.Y(ps[i]));
  });
  // right: the likelihood (or log-likelihood) of each theta for these data
  const ts = linspace(0.001, 0.999, 500);
  const L = t => pmf.binomial(lik.y, n, t);
  const mle = lik.y / n;
  const here = L(theta);
  let gr;
  if (lik.log) {
    const ls = ts.map(t => Math.log(L(t))), top = Math.log(L(Math.min(0.999, Math.max(0.001, mle))));
    gr = frame($("#likCurve"), aspect, { x: [0, 1], y: [top - 12, top + 0.8], yfmt: v => fmt(v, 0), xlabel: "θ" });
    plotLine(gr, ts, ls, { color: css("--lik"), width: 2.2 });
    dot(gr, theta, Math.log(here));
  } else {
    const ls = ts.map(L), top = Math.max(...ls) * 1.15;
    gr = frame($("#likCurve"), aspect, { x: [0, 1], y: [0, top], yfmt: v => fmt(v, 2), xlabel: "θ" });
    plotLine(gr, ts, ls, { color: css("--lik"), width: 2.2, fill: alpha("--lik", 0.12) });
    dot(gr, theta, here);
  }
  gr.ctx.strokeStyle = css("--ink"); gr.ctx.setLineDash([4, 3]); gr.ctx.lineWidth = 1.2;
  gr.ctx.beginPath(); gr.ctx.moveTo(gr.X(mle), gr.pad.t); gr.ctx.lineTo(gr.X(mle), gr.Y(gr.y0)); gr.ctx.stroke(); gr.ctx.setLineDash([]);
  caption($("#likCap"),
    `At θ = ${fmt(theta, 2)}, the probability of exactly ${lik.y} make${lik.y === 1 ? "" : "s"} in ${n} throws is <b>${fmt(here, 3)}</b>` +
    (lik.log ? ` (log: ${fmt(Math.log(here), 2)})` : "") + `: the orange bar on the left and the dot on the right. ` +
    `The bars add up to 1. The area under the likelihood curve is [[\\frac{1}{${n + 1}} \\approx ${fmt(1 / (n + 1), 3)}]], not 1. ` +
    `The curve peaks at the MLE, [[\\hat\\theta = \\frac{${lik.y}}{${n}} = ${fmt(mle, 2)}]] (the dashed line).`);
}
function dot(fr, x, y) {
  fr.ctx.fillStyle = css("--warm"); fr.ctx.strokeStyle = css("--surface"); fr.ctx.lineWidth = 2;
  fr.ctx.beginPath(); fr.ctx.arc(fr.X(x), fr.Y(y), 5, 0, 2 * Math.PI); fr.ctx.fill(); fr.ctx.stroke();
}
$("#likTheta").addEventListener("input", e => { lik.theta = +e.target.value / 100; drawLik(); });
$("#likN").addEventListener("input", e => { lik.n = +e.target.value; drawLik(); });
$("#likY").addEventListener("input", e => { lik.y = +e.target.value; drawLik(); });
$("#likLog").addEventListener("change", e => { lik.log = e.target.checked; drawLik(); });

/* =====================================================================
   CURVATURE: MORE DATA, SHARPER PEAK
   ===================================================================== */
const CURV_NS = [20, 40, 80, 160, 320];
let curvIdx = 0;
function drawCurv() {
  const n = CURV_NS[curvIdx], y = Math.round(0.65 * n), mle = y / n;
  const ll = t => y * Math.log(t) + (n - y) * Math.log(1 - t);
  const J = n / (mle * (1 - mle)), se = 1 / Math.sqrt(J);
  const ts = linspace(0.3, 0.95, 400);
  const canvas = $("#curvC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.4 : 2.4, { x: [0.3, 0.95], y: [-6, 0.4], yticks: [-6, -4, -2, 0], xlabel: "θ", ylabel: "ℓ(θ) − ℓ(θ̂)" });
  const cut = -(1.96 ** 2) / 2;
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--muted"); ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(fr.pad.l, Y(cut)); ctx.lineTo(fr.w - fr.pad.r, Y(cut)); ctx.stroke(); ctx.setLineDash([]);
  plotLine(fr, ts, ts.map(t => -((t - mle) ** 2) / (2 * se * se)), { color: css("--post"), width: 1.8, dash: [6, 4] });
  plotLine(fr, ts, ts.map(t => ll(t) - ll(mle)), { color: css("--lik"), width: 2.4 });
  const lo = mle - 1.96 * se, hi = mle + 1.96 * se;
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(X(lo), Y(cut)); ctx.lineTo(X(hi), Y(cut)); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left";
  ctx.fillText("1.92 below the top", fr.pad.l + 4, Y(cut) - 5);
  $("#curvNV").textContent = `${n} throws, ${y} made`;
  caption($("#curvCap"),
    `Orange: the log-likelihood, shifted so that its top is at 0. Dashed blue: the parabola with the same curvature at the peak. ` +
    `Information [[J = ${fmt(J, 1)}]], standard error [[\\text{SE} = ${fmt(se, 3)}]], and the 95% Wald interval (${fmt(lo, 2)}, ${fmt(hi, 2)}), the thick bar: ` +
    `it is where the parabola drops [[\\frac{1.96^2}{2} \\approx 1.92]] below its top.` +
    (n === 20 ? " With 20 throws the real curve is visibly lopsided compared with the parabola." : n >= 160 ? " With this much data the curve and the parabola almost coincide." : ""));
}
$("#curvN").addEventListener("input", e => { curvIdx = +e.target.value; drawCurv(); });

/* =====================================================================
   EXACT COVERAGE
   ===================================================================== */
const cov = { nIdx: 2, theta: 0.05, wald: true, wilson: false };
const COV_GRID = linspace(0.002, 0.998, 499);
function drawCov() {
  const n = CI_NS[cov.nIdx];
  const canvas = $("#covC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.4 : 2.4, { x: [0, 1], y: [0, 1.02], yticks: [0, 0.25, 0.5, 0.75, 0.95], yfmt: v => pct(v, 0), xlabel: "true θ" });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--ink"); ctx.setLineDash([5, 4]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(fr.pad.l, Y(0.95)); ctx.lineTo(fr.w - fr.pad.r, Y(0.95)); ctx.stroke(); ctx.setLineDash([]);
  const lines = [];
  if (cov.wald) lines.push(["Wald", wald, css("--warm")]);
  if (cov.wilson) lines.push(["Wilson", wilson, css("--post")]);
  for (const [, f, color] of lines) plotLine(fr, COV_GRID, COV_GRID.map(t => coverage(t, n, f)), { color, width: 1.6 });
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(cov.theta), fr.pad.t); ctx.lineTo(X(cov.theta), Y(0)); ctx.stroke();
  for (const [, f, color] of lines) {
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(X(cov.theta), Y(coverage(cov.theta, n, f)), 4.5, 0, 2 * Math.PI); ctx.fill();
  }
  $("#covNV").textContent = n;
  $("#covTV").textContent = fmt(cov.theta, 2);
  const parts = lines.map(([name, f]) => `${name} <b>${fmt(coverage(cov.theta, n, f), 3)}</b>`);
  caption($("#covCap"),
    (parts.length ? `At θ = ${fmt(cov.theta, 2)} with n = ${n}, the exact coverage of the "95%" interval is: ${parts.join(", ")}.` : "Tick an interval to draw its coverage.") +
    ` The dashed line is the promised 95%. The curves are jagged because y can only take ${n + 1} values: as θ moves, values of y enter and leave the set whose interval catches θ.`);
}
$("#covN").addEventListener("input", e => { cov.nIdx = +e.target.value; drawCov(); });
$("#covT").addEventListener("input", e => { cov.theta = +e.target.value / 100; drawCov(); });
$("#covWald").addEventListener("change", e => { cov.wald = e.target.checked; drawCov(); });
$("#covWilson").addEventListener("change", e => { cov.wilson = e.target.checked; drawCov(); });

/* =====================================================================
   A PREVIEW: PRIOR × LIKELIHOOD = POSTERIOR
   ===================================================================== */
const PRIORS = { flat: [1, 1], informed: [30, 10] };
let priorKey = "flat";
const BGRID = linspace(0, 1, 4001);
function betaSummary(a, b) {
  const F = cumulative(t => pdf.beta(t, a, b), BGRID);
  const q = p => { const i = F.findIndex(v => v >= p); return BGRID[Math.max(0, i)]; };
  const above = 1 - F[2000];                               // P(theta > 0.5): grid point 2000 is 0.5
  return { lo: q(0.025), hi: q(0.975), above, mean: a / (a + b) };
}
function drawPost() {
  const [a0, b0] = PRIORS[priorKey], a = a0 + 13, b = b0 + 7;
  const ts = linspace(0.002, 0.998, 500);
  const post = ts.map(t => pdf.beta(t, a, b));
  const prior = ts.map(t => pdf.beta(t, a0, b0));
  const like = ts.map(t => pdf.beta(t, 14, 8));             // the likelihood, scaled to area 1
  const top = Math.max(...post, ...prior, ...like) * 1.12;
  const canvas = $("#postC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.4 : 2.5, { x: [0, 1], y: [0, top], yticks: false, xlabel: "success rate θ", pad: { b: 58 } });
  const s = betaSummary(a, b);
  const right = ts.filter(t => t > 0.5);
  plotLine(fr, right, right.map(t => pdf.beta(t, a, b)), { fill: alpha("--post", 0.22) });
  plotLine(fr, ts, prior, { color: css("--prior"), width: 2, dash: priorKey === "flat" ? [4, 4] : null });
  plotLine(fr, ts, like, { color: css("--lik"), width: 1.8, dash: [6, 4] });
  plotLine(fr, ts, post, { color: css("--post"), width: 2.6 });
  // interval bars under the axis
  const { ctx, X, Y } = fr;
  const bar = (lo, hi, yy, color, label) => {
    ctx.strokeStyle = color; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(X(lo), yy); ctx.lineTo(X(hi), yy); ctx.stroke();
    ctx.fillStyle = color; ctx.font = "600 11px " + css("--font-ui"); ctx.textAlign = "right";
    ctx.fillText(label, X(lo) - 6, yy + 4);
  };
  bar(s.lo, s.hi, Y(0) + 32, css("--post"), "95% credible interval");
  bar(0.441, 0.859, Y(0) + 46, css("--warm"), "95% Wald interval");
  $$("#post [data-prior]").forEach(btn => btn.setAttribute("aria-pressed", btn.dataset.prior === priorKey ? "true" : "false"));
  const pAbove = s.above > 0.999 ? "more than 0.999" : fmt(s.above, 3);
  caption($("#postCap"), priorKey === "flat"
    ? `With a flat prior the posterior is just the likelihood, rescaled to area 1: Beta(14, 8). It answers the first question, [[P(\\theta > 0.5 \\mid y) = ${pAbove}]] (the shaded area), and the second: θ is between ${fmt(s.lo, 2)} and ${fmt(s.hi, 2)} with probability 95%.`
    : `The prior Beta(30, 10) is worth 40 earlier throws with 30 made. The posterior Beta(43, 17) sits between prior and likelihood, at a mean of ${fmt(s.mean, 2)}, and is narrower than either: θ is between ${fmt(s.lo, 2)} and ${fmt(s.hi, 2)} with probability 95%, and [[P(\\theta > 0.5 \\mid y)]] is ${pAbove}. That answers the third question.`);
}
$$("#post [data-prior]").forEach(btn => btn.addEventListener("click", () => { priorKey = btn.dataset.prior; drawPost(); }));

/* =====================================================================
   start-up
   ===================================================================== */
$("#ciTheta").value = Math.round(ci.theta * 100); $("#ciN").value = ci.nIdx;
$("#likTheta").value = Math.round(lik.theta * 100); $("#likN").value = lik.n; $("#likY").value = lik.y;
drawCi(); drawLik(); drawCurv(); drawCov(); drawPost();
onRedraw(() => { drawCi(); drawLik(); drawCurv(); drawCov(); drawPost(); });
})();
