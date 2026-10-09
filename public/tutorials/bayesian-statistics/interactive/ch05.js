/* Chapter 05, Counts and proportions: the live demos.
   Numbers quoted in the text come from the chapter's notebook; these widgets recompute
   the same conjugate updates in the browser. Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, int, caption, frame, plotLine, alpha, onRedraw, narrowOf } = window.Kit;
const { pdf, pmf, lgamma, lchoose, makeRng, linspace, cumulative, densityQuantiles } = window.Stats;
const rng = makeRng();
const signed = v => (v >= 0 ? "+" : "−") + Math.abs(Math.round(100 * v)) + "%";

/* ---------------- beta and gamma helpers (numerical quantiles on a local grid) ---------------- */
const betaMean = (a, b) => a / (a + b);
const betaSd = (a, b) => Math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)));
const gammaMean = (a, b) => a / b, gammaSd = (a, b) => Math.sqrt(a) / b;
const betaRange = (a, b) => [Math.max(0, betaMean(a, b) - 9 * betaSd(a, b)), Math.min(1, betaMean(a, b) + 9 * betaSd(a, b))];
const betaQ = (a, b, ps) => { const [lo, hi] = betaRange(a, b); return densityQuantiles(x => pdf.beta(x, a, b), lo, hi, ps); };
const gammaQ = (a, b, ps) => densityQuantiles(x => pdf.gamma(x, a, b), 0, gammaMean(a, b) + 12 * gammaSd(a, b), ps);

/* =====================================================================
   THE A/B TEST (the demo at the top)
   ===================================================================== */
const ab = { nA: 240, yA: 26, nB: 250, yB: 41, m: 0.10, k: 30, draws: null };
const S = 20000;
function posteriors() {
  const a0 = ab.m * ab.k, b0 = (1 - ab.m) * ab.k;
  return { a0, b0, A: [a0 + ab.yA, b0 + ab.nA - ab.yA], B: [a0 + ab.yB, b0 + ab.nB - ab.yB] };
}
/* P(theta_B > theta_A) = integral of p_A(t) (1 - F_B(t)) dt, on a grid covering both posteriors */
function probBetter(A, B) {
  const [la, ha] = betaRange(...A), [lb, hb] = betaRange(...B);
  const xs = linspace(Math.min(la, lb), Math.max(ha, hb), 6001);
  const FB = cumulative(x => pdf.beta(x, ...B), xs), tot = FB[FB.length - 1];
  const f = xs.map((x, i) => pdf.beta(x, ...A) * (1 - FB[i] / tot));
  let s = 0;
  for (let i = 1; i < xs.length; i++) s += 0.5 * (f[i] + f[i - 1]) * (xs[i] - xs[i - 1]);
  return s;
}
/* Recompute only when the inputs change; redraws (resize, theme) reuse the same draws. */
function computeAb() {
  const { A, B } = posteriors();
  const tA = new Float64Array(S), tB = new Float64Array(S), lift = new Float64Array(S);
  for (let i = 0; i < S; i++) { tA[i] = rng.beta(...A); tB[i] = rng.beta(...B); lift[i] = tB[i] / tA[i] - 1; }
  ab.draws = { tA, tB, lift, pBetter: probBetter(A, B) };
}
function updateAb() { computeAb(); drawAb(); }
function drawAb() {
  const { a0, b0, A, B } = posteriors();
  const { lift } = ab.draws;
  const sorted = Float64Array.from(lift).sort();
  const lq = p => sorted[Math.min(S - 1, Math.floor(p * S))];
  const pBetter = ab.draws.pBetter, p20 = lift.reduce((c, v) => c + (v > 0.2 ? 1 : 0), 0) / S;
  const [qa1, qa2] = betaQ(...A, [0.025, 0.975]), [qb1, qb2] = betaQ(...B, [0.025, 0.975]);
  // chart 1: prior and the two posteriors
  const canvas = $("#abC");
  const hiX = Math.min(1, Math.max(betaQ(a0, b0, [0.995])[0], betaMean(...A) + 6 * betaSd(...A), betaMean(...B) + 6 * betaSd(...B)));
  const xs = linspace(0, hiX, 600);
  const curves = [[a0, b0], A, B].map(([a, b]) => xs.map(x => pdf.beta(x, a, b)));
  const top = Math.max(...curves[1], ...curves[2]) * 1.1;
  const fr = frame(canvas, narrowOf(canvas) ? 1.6 : 2.6, { x: [0, hiX], y: [0, top], yticks: false, xfmt: v => pct(v, 0), xlabel: "conversion rate" });
  plotLine(fr, xs, curves[0], { color: css("--prior"), width: 1.4, fill: alpha("--prior", 0.25) });
  plotLine(fr, xs, curves[1], { color: css("--lik"), width: 2.4, fill: alpha("--lik", 0.15) });
  plotLine(fr, xs, curves[2], { color: css("--post"), width: 2.4, fill: alpha("--post", 0.15) });
  // chart 2: the lift
  const lo = lq(0.005), hi = lq(0.995), BINS = 60, w = (hi - lo) / BINS, counts = new Array(BINS).fill(0);
  for (const v of lift) { const i = Math.floor((v - lo) / w); if (i >= 0 && i < BINS) counts[i]++; }
  const dens = counts.map(c => c / (S * w));
  const lc = $("#liftC");
  const gr = frame(lc, narrowOf(lc) ? 2.2 : 4, { x: [lo, hi], y: [0, Math.max(...dens) * 1.12], yticks: false, xfmt: v => signed(v), xlabel: "lift of B over A" });
  dens.forEach((d, i) => {
    const l = lo + i * w;
    gr.ctx.fillStyle = l + w / 2 > 0 ? alpha("--post", 0.8) : alpha("--prior", 0.8);
    gr.ctx.fillRect(gr.X(l) + 0.4, gr.Y(d), Math.max(0.8, gr.X(l + w) - gr.X(l) - 0.8), gr.Y(0) - gr.Y(d));
  });
  if (lo < 0 && hi > 0) { gr.ctx.strokeStyle = css("--ink"); gr.ctx.lineWidth = 1.4; gr.ctx.beginPath(); gr.ctx.moveTo(gr.X(0), gr.pad.t); gr.ctx.lineTo(gr.X(0), gr.Y(0)); gr.ctx.stroke(); }
  // readouts
  $("#bigP").textContent = fmt(pBetter, 3);
  $("#rA").textContent = `${fmt(betaMean(...A), 3)} (${fmt(qa1, 3)}, ${fmt(qa2, 3)})`;
  $("#rB").textContent = `${fmt(betaMean(...B), 3)} (${fmt(qb1, 3)}, ${fmt(qb2, 3)})`;
  $("#rLift").textContent = signed(lq(0.5));
  $("#rLiftCI").textContent = `(${signed(lq(0.025))}, ${signed(lq(0.975))})`;
  $("#rLift20").textContent = fmt(p20, 3);
  $("#pmV").textContent = pct(ab.m, 0);
  $("#pkV").textContent = int(ab.k);
  const verdict = pBetter > 0.95 ? "B is very probably better" : pBetter > 0.8 ? "B is probably better, but not certainly" : pBetter > 0.2 ? "the data can't tell the two apart" : "A is probably better";
  $("#abStatus").innerHTML = `<b>${verdict}</b>: P(B better) = ${fmt(pBetter, 3)}. The most likely lift is around ${signed(lq(0.5))}, and anything from ${signed(lq(0.025))} to ${signed(lq(0.975))} is compatible with the data (95%).` +
    (ab.k > ab.nA + ab.nB ? " The prior is worth more visitors than the test itself, so it pulls both rates towards its typical value." : "");
  drawMc(); drawBb();
}
function readAbInputs() {
  const v = id => Math.max(0, Math.round(+$(id).value || 0));
  ab.nA = Math.max(1, v("#nA")); ab.nB = Math.max(1, v("#nB"));
  ab.yA = Math.min(v("#yA"), ab.nA); ab.yB = Math.min(v("#yB"), ab.nB);
}
["#nA", "#yA", "#nB", "#yB"].forEach(id => $(id).addEventListener("input", () => { readAbInputs(); updateAb(); }));
$("#pm").addEventListener("input", e => { ab.m = +e.target.value / 100; updateAb(); });
$("#pk").addEventListener("input", e => { ab.k = +e.target.value; updateAb(); });

/* ---------------- Monte Carlo pairs ---------------- */
function drawMc() {
  const { tA, tB } = ab.draws, n = 2500;
  const all = [...tA.slice(0, n), ...tB.slice(0, n)].sort((a, b) => a - b);
  const lo = all[Math.floor(0.002 * all.length)], hi = all[Math.floor(0.998 * all.length)];
  const pad = (hi - lo) * 0.08;
  const canvas = $("#mcC");
  const side = Math.min(canvas.parentElement.clientWidth, 460);
  canvas.style.maxWidth = side + "px"; canvas.style.marginInline = "auto";
  const fr = frame(canvas, 1, { x: [lo - pad, hi + pad], y: [lo - pad, hi + pad], xfmt: v => pct(v, 0), yfmt: v => pct(v, 0), xlabel: "θA", ylabel: "θB" });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(X(lo - pad), Y(lo - pad)); ctx.lineTo(X(hi + pad), Y(hi + pad)); ctx.stroke(); ctx.setLineDash([]);
  let above = 0;
  for (let i = 0; i < n; i++) {
    const b = tB[i] > tA[i]; if (b) above++;
    ctx.fillStyle = b ? alpha("--post", 0.55) : alpha("--lik", 0.8);
    ctx.fillRect(X(tA[i]) - 1.2, Y(tB[i]) - 1.2, 2.4, 2.4);
  }
  caption($("#mcCap"), `${int(n)} pairs drawn from the two posteriors. <b>${int(above)}</b> lie above the diagonal, where B is better: ${fmt(above / n, 3)}. ` +
    `The exact value from the integral is ${$("#bigP").textContent}; Monte Carlo's answer wobbles around it by about ${fmt(Math.sqrt(0.04 / n) * 2, 3)} from draw to draw, and more draws shrink the wobble.`);
}
$("#mcAgain").addEventListener("click", () => updateAb());

/* ---------------- predicting future buyers of B ---------------- */
const bb = { m: 100, k: 20 };
function betabinomPmf(k, m, a, b) {
  return Math.exp(lchoose(m, k) + lgamma(a + k) + lgamma(b + m - k) - lgamma(a + b + m) - (lgamma(a) + lgamma(b) - lgamma(a + b)));
}
function drawBb() {
  const { B } = posteriors(), { m } = bb;
  $("#bbK").max = m; bb.k = Math.min(bb.k, m); $("#bbK").value = bb.k;
  $("#bbMV").textContent = m; $("#bbKV").textContent = bb.k;
  const ks = Array.from({ length: m + 1 }, (_, k) => k);
  const pred = ks.map(k => betabinomPmf(k, m, ...B)), plug = ks.map(k => pmf.binomial(k, m, ab.yB / ab.nB));
  const mom = ps => { const mu = ps.reduce((s, p, k) => s + k * p, 0); return [mu, Math.sqrt(ps.reduce((s, p, k) => s + (k - mu) ** 2 * p, 0))]; };
  const [mp, sp] = mom(pred), [mu, su] = mom(plug);
  const hiK = Math.min(m, Math.ceil(Math.max(mp + 5 * sp, mu + 5 * su))), loK = Math.max(0, Math.floor(Math.min(mp - 5 * sp, mu - 5 * su)));
  const top = Math.max(...pred, ...plug) * 1.15;
  const canvas = $("#bbC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.5 : 2.6, { x: [loK - 0.7, hiK + 0.7], y: [0, top], yfmt: v => fmt(v, 2), xlabel: `buyers among the next ${m} visitors` });
  const bw = Math.max(0.8, (fr.X(1) - fr.X(0)) * 0.38);
  for (let k = loK; k <= hiK; k++) {
    fr.ctx.fillStyle = alpha("--prior", k >= bb.k ? 1 : 0.5); fr.ctx.fillRect(fr.X(k) - bw - 0.3, fr.Y(plug[k]), bw, fr.Y(0) - fr.Y(plug[k]));
    fr.ctx.fillStyle = alpha("--post", k >= bb.k ? 1 : 0.45); fr.ctx.fillRect(fr.X(k) + 0.3, fr.Y(pred[k]), bw, fr.Y(0) - fr.Y(pred[k]));
  }
  const tail = ps => ps.reduce((s, p, k) => s + (k >= bb.k ? p : 0), 0);
  caption($("#bbCap"),
    `Beta-binomial predictive: mean ${fmt(mp, 2)}, sd ${fmt(sp, 2)}, [[P(\\tilde y \\ge ${bb.k}) = ${fmt(tail(pred), 3)}]]. ` +
    `Plug-in Binomial(${m}, ${fmt(ab.yB / ab.nB, 3)}): mean ${fmt(mu, 2)}, sd ${fmt(su, 2)}, ${fmt(tail(plug), 3)}. ` +
    "The predictive is wider, and its mean is pulled a little towards the prior's typical rate.");
}
$("#bbM").addEventListener("input", e => { bb.m = +e.target.value; drawBb(); });
$("#bbK").addEventListener("input", e => { bb.k = +e.target.value; drawBb(); });

/* =====================================================================
   SHRINKAGE: A PRIOR IS IMAGINARY DATA
   ===================================================================== */
const sh = { m: 0.75, k: 40, n: 20, y: 13 };
function drawShrink() {
  $("#sY").max = sh.n; sh.y = Math.min(sh.y, sh.n); $("#sY").value = sh.y;
  $("#sMV").textContent = fmt(sh.m, 2); $("#sKV").textContent = sh.k; $("#sNV").textContent = sh.n; $("#sYV").textContent = sh.y;
  const a0 = sh.m * sh.k, b0 = (1 - sh.m) * sh.k, a1 = a0 + sh.y, b1 = b0 + sh.n - sh.y;
  const xs = linspace(0.001, 0.999, 600);
  const prior = xs.map(x => pdf.beta(x, a0, b0)), post = xs.map(x => pdf.beta(x, a1, b1));
  const lik = sh.n ? xs.map(x => pdf.beta(x, sh.y + 1, sh.n - sh.y + 1)) : null;
  const top = Math.min(40, Math.max(...post, ...prior.slice(3, -3), ...(lik || [0]).slice(3, -3)) * 1.1);
  const canvas = $("#shrinkC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.4 : 2.4, { x: [0, 1], y: [0, top], yticks: false, pad: { b: 62 } });
  plotLine(fr, xs, prior, { color: css("--prior"), width: 1.6, fill: alpha("--prior", 0.25) });
  if (lik) plotLine(fr, xs, lik, { color: css("--lik"), width: 1.8, dash: [6, 4] });
  plotLine(fr, xs, post, { color: css("--post"), width: 2.6, fill: alpha("--post", 0.15) });
  // the tug of war between prior mean and data
  const { ctx, X, Y } = fr, yy = Y(0) + 40;
  const pm = sh.m, dm = sh.n ? sh.y / sh.n : null, qm = a1 / (a1 + b1);
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 2;
  if (dm !== null) { ctx.beginPath(); ctx.moveTo(X(pm), yy); ctx.lineTo(X(dm), yy); ctx.stroke(); }
  // labels above the line point away from each other, so close values don't overlap
  const dotAt = (v, color, label, where) => {
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(X(v), yy, 5.5, 0, 2 * Math.PI); ctx.fill();
    ctx.font = "600 11px " + css("--font-ui");
    ctx.textAlign = where === "left" ? "right" : where === "right" ? "left" : "center";
    const x = where === "left" ? X(v) + 6 : where === "right" ? X(v) - 6 : Math.min(fr.w - 44, Math.max(44, X(v)));
    ctx.fillText(label, x, where === "below" ? yy + 18 : yy - 10);
  };
  const priorLeft = dm === null || pm <= dm;
  dotAt(pm, css("--prior"), "prior mean", priorLeft ? "left" : "right");
  if (dm !== null) dotAt(dm, css("--lik"), "data proportion", priorLeft ? "right" : "left");
  dotAt(qm, css("--post"), "posterior mean", "below");
  const w = sh.k / (sh.k + sh.n);
  caption($("#shrinkCap"),
    `Prior Beta(${fmt(a0, 1).replace(/\.0$/, "")}, ${fmt(b0, 1).replace(/\.0$/, "")}), worth ${sh.k} observations` +
    (sh.n ? `, plus ${sh.y} successes in ${sh.n} trials, gives the posterior Beta(${fmt(a1, 1).replace(/\.0$/, "")}, ${fmt(b1, 1).replace(/\.0$/, "")}). ` +
      `Its mean is [[${fmt(qm, 3)} = ${fmt(w, 2)} \\times ${fmt(pm, 2)} + ${fmt(1 - w, 2)} \\times ${fmt(dm, 2)}]]: the prior's weight is its worth ${sh.k} out of ${sh.k + sh.n} observations in total.`
      : `, and no data yet: the posterior is the prior.`));
}
[["#sM", v => (sh.m = v / 100)], ["#sK", v => (sh.k = v)], ["#sN", v => (sh.n = v)], ["#sY", v => (sh.y = v)]]
  .forEach(([id, set]) => $(id).addEventListener("input", e => { set(+e.target.value); drawShrink(); }));

/* =====================================================================
   THE CAFE: POISSON-GAMMA
   ===================================================================== */
const NOTEBOOK_DAYS = [6, 3, 7, 5, 8, 4, 6];
const cafe = { days: NOTEBOOK_DAYS.slice(), m: 3, k: 2 };
function buildDays() {
  $("#cafeDays").innerHTML = cafe.days.map((c, i) =>
    `<label>day ${i + 1}<input type="number" min="0" max="99" step="1" value="${c}" data-i="${i}" aria-label="Complaints on day ${i + 1}"></label>`).join("");
  $$("#cafeDays input").forEach(inp => inp.addEventListener("input", () => {
    const v = Math.round(+inp.value);
    if (isFinite(v) && v >= 0 && v <= 99) { cafe.days[+inp.dataset.i] = v; drawCafe(); }
  }));
}
function negbinPmf(k, a, p) { return Math.exp(lgamma(k + a) - lgamma(a) - lgamma(k + 1) + a * Math.log(p) + k * Math.log(1 - p)); }
function drawCafe() {
  const n = cafe.days.length, tot = cafe.days.reduce((a, b) => a + b, 0);
  const a0 = cafe.m * cafe.k, b0 = cafe.k, a1 = a0 + tot, b1 = b0 + n;
  $("#cMV").textContent = fmt(cafe.m, 1); $("#cKV").textContent = fmt(cafe.k, 1);
  const hiX = Math.max(10, gammaMean(a1, b1) + 5 * gammaSd(a1, b1), gammaMean(a0, b0) + 3 * gammaSd(a0, b0));
  const xs = linspace(0.01, hiX, 600);
  const prior = xs.map(x => pdf.gamma(x, a0, b0)), post = xs.map(x => pdf.gamma(x, a1, b1));
  const lik = n ? xs.map(x => pdf.gamma(x, tot + 1, n)) : null;
  const top = Math.max(...post, ...prior.slice(2), ...(lik || [0])) * 1.1;
  const c1 = $("#cafeC");
  const fr = frame(c1, narrowOf(c1) ? 1.6 : 2.8, { x: [0, hiX], y: [0, top], yticks: false });
  plotLine(fr, xs, prior, { color: css("--prior"), width: 1.6, fill: alpha("--prior", 0.25) });
  if (lik) plotLine(fr, xs, lik, { color: css("--lik"), width: 1.8, dash: [6, 4] });
  plotLine(fr, xs, post, { color: css("--post"), width: 2.6, fill: alpha("--post", 0.15) });
  // tomorrow: negative binomial predictive, and a Poisson with the same mean for comparison
  const p = b1 / (b1 + 1), mean = a1 / b1, varc = a1 / b1 + a1 / b1 ** 2;
  const kMax = Math.max(15, Math.ceil(mean + 5 * Math.sqrt(varc)));
  const ks = Array.from({ length: kMax + 1 }, (_, k) => k);
  const nb = ks.map(k => negbinPmf(k, a1, p)), po = ks.map(k => pmf.poisson(k, mean));
  const c2 = $("#cafeP");
  const gr = frame(c2, narrowOf(c2) ? 1.8 : 3.4, { x: [-0.7, kMax + 0.7], y: [0, Math.max(...nb, ...po) * 1.15], yfmt: v => fmt(v, 2), xticks: kMax <= 20 ? ks : undefined });
  const bw = Math.max(1, (gr.X(1) - gr.X(0)) * 0.6);
  ks.forEach(k => { gr.ctx.fillStyle = alpha("--post", k >= 10 ? 1 : 0.55); gr.ctx.fillRect(gr.X(k) - bw / 2, gr.Y(nb[k]), bw, gr.Y(0) - gr.Y(nb[k])); });
  gr.ctx.fillStyle = css("--ink");
  ks.forEach(k => { gr.ctx.beginPath(); gr.ctx.arc(gr.X(k), gr.Y(po[k]), 2.6, 0, 2 * Math.PI); gr.ctx.fill(); });
  const [q1, q2] = gammaQ(a1, b1, [0.025, 0.975]);
  const tail = nb.reduce((s, v, k) => s + (k >= 10 ? v : 0), 0) + Math.max(0, 1 - nb.reduce((s, v) => s + v, 0));
  const w = b0 / b1;
  const g = (v) => fmt(v, 1).replace(/\.0$/, "");
  caption($("#cafeCap"),
    `Prior Gamma(${g(a0)}, ${g(b0)}) plus ${tot} complaints in ${n} days gives the posterior <b>Gamma(${g(a1)}, ${g(b1)})</b>` +
    (n ? `, with mean [[${fmt(mean, 3)} = ${fmt(w, 2)} \\times ${fmt(cafe.m, 1)} + ${fmt(1 - w, 2)} \\times ${fmt(tot / n, 3)}]]` : "") +
    ` and a 95% interval of (${fmt(q1, 2)}, ${fmt(q2, 2)}) complaints per day. ` +
    `Tomorrow (blue bars): mean ${fmt(mean, 2)} and variance ${fmt(varc, 2)}, against ${fmt(mean, 2)} for a Poisson with the same mean (black dots). ` +
    `[[P(\\tilde y \\ge 10) = ${fmt(tail, 4)}]] (the dark bars).`);
}
$("#cM").addEventListener("input", e => { cafe.m = +e.target.value; drawCafe(); });
$("#cK").addEventListener("input", e => { cafe.k = +e.target.value; drawCafe(); });
$("#cafeAdd").addEventListener("click", () => { if (cafe.days.length < 30) { cafe.days.push(5); buildDays(); drawCafe(); } });
$("#cafeDay8").addEventListener("click", () => { cafe.days = NOTEBOOK_DAYS.concat([12]); buildDays(); drawCafe(); });
$("#cafeReset").addEventListener("click", () => { cafe.days = NOTEBOOK_DAYS.slice(); cafe.m = 3; cafe.k = 2; $("#cM").value = 3; $("#cK").value = 2; buildDays(); drawCafe(); });

/* =====================================================================
   start-up
   ===================================================================== */
$("#nA").value = ab.nA; $("#yA").value = ab.yA; $("#nB").value = ab.nB; $("#yB").value = ab.yB;
$("#pm").value = Math.round(ab.m * 100); $("#pk").value = ab.k;
$("#sM").value = Math.round(sh.m * 100); $("#sK").value = sh.k; $("#sN").value = sh.n; $("#sY").value = sh.y;
$("#cM").value = cafe.m; $("#cK").value = cafe.k;
buildDays();
updateAb(); drawShrink(); drawCafe();
onRedraw(() => { drawAb(); drawShrink(); drawCafe(); });
})();
