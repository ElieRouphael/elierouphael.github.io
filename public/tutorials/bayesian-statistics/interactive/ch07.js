/* Chapter 07, Choosing priors: the live demos.
   Numbers quoted in the text come from the chapter's notebook; these widgets recompute
   the same quantities in the browser. Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, int, caption, frame, plotLine, alpha, onRedraw, narrowOf } = window.Kit;
const { lgamma, lchoose, integrate, makeRng, linspace, densityQuantiles } = window.Stats;
const rng = makeRng();

/* ---------------- beta helpers with the constant computed once ---------------- */
function betaDens(a, b) {
  const c = lgamma(a + b) - lgamma(a) - lgamma(b);
  return x => (x <= 0 || x >= 1 ? 0 : Math.exp(c + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)));
}
function betaQ(a, b, ps) {
  const m = a / (a + b), sd = Math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)));
  const lo = Math.max(0, m - 12 * sd), hi = Math.min(1, m + 12 * sd);
  return densityQuantiles(betaDens(a, b), lo, hi, ps, 1601);
}

/* =====================================================================
   ELICITATION (the demo at the top)
   ===================================================================== */
const el = { mode: 0.2, high: 0.4 };
/* mode m fixes beta = (alpha - 1)(1/m - 1) + 1; bisection on alpha for the 90th percentile */
function elicit(m, h) {
  const betaOf = a => (a - 1) * (1 / m - 1) + 1;
  const q90 = a => betaQ(a, betaOf(a), [0.9])[0];
  let lo = 1.0005, hi = 3000;
  if (q90(lo) <= h || q90(hi) >= h) return null;      // 90th percentile out of reach
  for (let k = 0; k < 42; k++) { const mid = Math.sqrt(lo * hi); if (q90(mid) > h) lo = mid; else hi = mid; }
  const a = Math.sqrt(lo * hi);
  return { a, b: betaOf(a) };
}
function drawElicit() {
  $("#elModeV").textContent = pct(el.mode, 0);
  $("#elHighV").textContent = pct(el.high, 0);
  const canvas = $("#elC");
  const res = el.high > el.mode ? elicit(el.mode, el.high) : null;
  if (!res) {
    frame(canvas, narrowOf(canvas) ? 1.6 : 2.6, { x: [0, 1], y: [0, 1], yticks: false });
    $("#elBig").textContent = "no fit";
    ["#rMean", "#rCI", "#rWorth"].forEach(s => { $(s).textContent = "–"; });
    $("#elFeedback").textContent = el.high <= el.mode
      ? "These two statements contradict each other: the value you'd be surprised to exceed must be above your best guess."
      : "No beta prior can say both things: even the flattest single-peaked prior puts its 90th percentile below this. Lower the upper value, or move the best guess.";
    return;
  }
  const { a, b } = res, f = betaDens(a, b);
  const [q025, q975] = betaQ(a, b, [0.025, 0.975]);
  const xs = linspace(0.001, 0.999, 600), ys = xs.map(f);
  const fr = frame(canvas, narrowOf(canvas) ? 1.6 : 2.6, { x: [0, 1], y: [0, Math.max(...ys) * 1.15], yticks: false, xfmt: v => pct(v, 0), xlabel: "adoption rate" });
  plotLine(fr, xs, ys, { color: css("--prior"), width: 2.2, fill: alpha("--prior", 0.22) });
  const tail = xs.filter(x => x >= el.high);
  plotLine(fr, tail, tail.map(f), { fill: alpha("--warm", 0.45) });
  const low = xs.filter(x => x <= q025);
  plotLine(fr, low, low.map(f), { fill: alpha("--post", 0.4) });
  const { ctx, X, Y } = fr;
  ctx.font = "600 11px " + css("--font-ui"); ctx.textAlign = "center";
  [[el.mode, css("--ink"), "best guess"], [el.high, css("--warm"), "10% above"]].forEach(([v, c, label], k) => {
    ctx.strokeStyle = c; ctx.lineWidth = 1.5; ctx.setLineDash(k ? [5, 4] : []);
    ctx.beginPath(); ctx.moveTo(X(v), Y(0)); ctx.lineTo(X(v), fr.pad.t + 14 + 12 * k); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = c; ctx.fillText(label, X(v), fr.pad.t + 10 + 12 * k);
  });
  const mean = a / (a + b);
  $("#elBig").textContent = `Beta(${fmt(a, 2)}, ${fmt(b, 2)})`;
  $("#rMean").textContent = pct(mean, 1);
  $("#rCI").textContent = `(${pct(q025, 1)}, ${pct(q975, 1)})`;
  $("#rWorth").textContent = fmt(a + b, 1);
  $("#elFeedback").innerHTML =
    `"So you are saying: the single most likely adoption rate is ${pct(el.mode, 0)}; there is one chance in ten that it is above ${pct(el.high, 0)} (orange); ` +
    `there is a 2.5% chance that it is below ${pct(q025, 1)} (blue); on average you expect ${pct(mean, 1)}; ` +
    `and your opinion is worth about as much as watching <b>${Math.round(a + b)} users</b>." If any of this sounds wrong to the expert, change the statements.`;
}
$("#elMode").addEventListener("input", e => { el.mode = +e.target.value / 100; drawElicit(); });
$("#elHigh").addEventListener("input", e => { el.high = +e.target.value / 100; drawElicit(); });

/* =====================================================================
   THE HALDANE PRIOR: AN INFINITE AREA
   ===================================================================== */
const hal = { y: 0, k: 2 };
const N_HAL = 20;
/* the unnormalised posterior theta^(y-1) (1-theta)^(n-y-1), and the same per decade of log10(theta) */
const halF = t => Math.exp((hal.y - 1) * Math.log(t) + (N_HAL - hal.y - 1) * Math.log(1 - t));
const perDecade = u => { const t = 10 ** u; return t >= 1 ? 0 : halF(t) * t * Math.LN10; };
function drawHal() {
  const u0 = -hal.k;
  const area = integrate(perDecade, u0, 0, 4000);
  const us = linspace(-16, 0, 800), gs = us.map(perDecade);
  const canvas = $("#halC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.6 : 2.8, { x: [-16, 0], y: [0, Math.max(...gs) * 1.15], xticks: [-16, -12, -8, -4, 0], xfmt: v => (v === 0 ? "1" : `10^${v}`), yticks: false, xlabel: "θ (log scale)" });
  const shade = us.filter(u => u >= u0);
  plotLine(fr, shade, shade.map(perDecade), { fill: alpha(hal.y ? "--post" : "--warm", 0.35) });
  plotLine(fr, us, gs, { color: css("--ink-2"), width: 2 });
  fr.ctx.strokeStyle = css("--ink"); fr.ctx.lineWidth = 1.4;
  fr.ctx.beginPath(); fr.ctx.moveTo(fr.X(u0), fr.Y(0)); fr.ctx.lineTo(fr.X(u0), fr.pad.t); fr.ctx.stroke();
  $("#halEV").textContent = `10^−${hal.k}`;
  $$("#hal [data-y]").forEach(b => b.setAttribute("aria-pressed", +b.dataset.y === hal.y ? "true" : "false"));
  caption($("#halCap"),
    `The curve is the posterior's area per factor of 10 in θ (each decade on this axis). ` +
    (hal.y === 0
      ? `With 0 successes it is flat on the left, so <b>every</b> decade adds the same area, about 2.3, and there are infinitely many decades: the area from [[10^{-${hal.k}}]] to 1 is <b>${fmt(area, 2)}</b>, and it never stops growing. There is no posterior.`
      : `With 1 success the likelihood's θ cancels the prior's [[\\theta^{-1}]], the curve dies away on the left, and the area settles: from [[10^{-${hal.k}}]] to 1 it is <b>${fmt(area, 4)}</b>, already essentially [[\\frac{1}{19} = ${fmt(1 / 19, 4)}]]. The posterior is Beta(1, 19).`));
}
$$("#hal [data-y]").forEach(b => b.addEventListener("click", () => { hal.y = +b.dataset.y; drawHal(); }));
$("#halE").addEventListener("input", e => { hal.k = +e.target.value; drawHal(); });

/* =====================================================================
   THE SAME PRIOR ON TWO SCALES
   ===================================================================== */
const LO_PRIORS = {
  flat: { f: () => 1, name: "the flat prior Beta(1, 1)" },
  jeff: { f: t => 1 / (Math.PI * Math.sqrt(t * (1 - t))), name: "Jeffreys' Beta(½, ½)" },
  hal: { f: t => 1 / (t * (1 - t)), name: "the Haldane prior", improper: true },
};
let loKey = "flat";
function drawLo() {
  const P = LO_PRIORS[loKey];
  const aspect = narrowOf($("#loT")) ? 1.5 : 1.25;
  const ts = linspace(0.002, 0.998, 500), ys = ts.map(P.f);
  const capT = P.improper ? 30 : 4;
  const fr = frame($("#loT"), aspect, { x: [0, 1], y: [0, Math.min(capT, Math.max(...ys)) * 1.1], yticks: false, xlabel: "θ" });
  plotLine(fr, ts, ys.map(v => Math.min(v, capT * 1.2)), { color: css("--post"), width: 2.4, fill: alpha("--post", 0.18) });
  const ps = linspace(-8, 8, 600);
  const th = p => 1 / (1 + Math.exp(-p));
  const yp = ps.map(p => { const t = th(p); return P.f(t) * t * (1 - t); });   // change of variables
  const gr = frame($("#loP"), aspect, { x: [-8, 8], y: [0, Math.max(...yp) * 1.25], yticks: false, xlabel: "φ = log(θ / (1 − θ))" });
  plotLine(gr, ps, yp, { color: css("--post"), width: 2.4, fill: alpha("--post", 0.18) });
  $$("#lo [data-p]").forEach(b => b.setAttribute("aria-pressed", b.dataset.p === loKey ? "true" : "false"));
  const text = {
    flat: "Flat on θ, but on the log-odds scale it is the logistic density, bunched around φ = 0 (θ = ½): it is quite sure that θ is not extreme, which is hardly \"no opinion\".",
    jeff: "Jeffreys' prior piles up a little at both ends of θ, and on the log-odds scale it becomes [[\\frac{1}{\\pi}\\sqrt{\\theta(1-\\theta)}]], wider than the logistic. It is also exactly what Jeffreys' rule gives if you start from the log-odds: the rule doesn't care which scale you use.",
    hal: "Flat on the log-odds scale, so it says nothing about φ, but on θ it is [[\\theta^{-1}(1-\\theta)^{-1}]], piled up at 0 and 1, with infinite area: improper on both scales. (The left panel is cut off at the top.)",
  };
  caption($("#loCap"), text[loKey]);
}
$$("#lo [data-p]").forEach(b => b.addEventListener("click", () => { loKey = b.dataset.p; drawLo(); }));

/* =====================================================================
   COVERAGE: WALD, JEFFREYS, FLAT
   ===================================================================== */
const COV_NS = [5, 10, 20, 50, 100, 200];
const cov = { nIdx: 2, theta: 0.05, show: { wald: true, jeff: true, flat: false } };
const intervalCache = new Map();
function intervals(n) {
  if (intervalCache.has(n)) return intervalCache.get(n);
  const wald = [], jeff = [], flat = [];
  for (let y = 0; y <= n; y++) {
    const t = y / n, se = Math.sqrt(t * (1 - t) / n);
    wald.push([t - 1.96 * se, t + 1.96 * se]);
    jeff.push(betaQ(y + 0.5, n - y + 0.5, [0.025, 0.975]));
    flat.push(betaQ(y + 1, n - y + 1, [0.025, 0.975]));
  }
  const res = { wald, jeff, flat, lc: Array.from({ length: n + 1 }, (_, y) => lchoose(n, y)) };
  intervalCache.set(n, res);
  return res;
}
function coverage(theta, n, ivs, lc) {
  const l1 = Math.log(theta), l2 = Math.log(1 - theta);
  let c = 0;
  for (let y = 0; y <= n; y++) { const [lo, hi] = ivs[y]; if (lo <= theta && theta <= hi) c += Math.exp(lc[y] + y * l1 + (n - y) * l2); }
  return c;
}
const COV_GRID = linspace(0.002, 0.998, 499);
const COV_STYLE = { wald: ["--warm", "Wald"], jeff: ["--post", "Jeffreys"], flat: ["--prior", "flat prior"] };
function drawCov() {
  const n = COV_NS[cov.nIdx], iv = intervals(n);
  const canvas = $("#covC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.4 : 2.4, { x: [0, 1], y: [0, 1.02], yticks: [0, 0.25, 0.5, 0.75, 0.95], yfmt: v => pct(v, 0), xlabel: "true θ" });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--ink"); ctx.setLineDash([5, 4]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(fr.pad.l, Y(0.95)); ctx.lineTo(fr.w - fr.pad.r, Y(0.95)); ctx.stroke(); ctx.setLineDash([]);
  const parts = [];
  for (const key of ["flat", "wald", "jeff"]) {
    if (!cov.show[key]) continue;
    const [tok, name] = COV_STYLE[key];
    plotLine(fr, COV_GRID, COV_GRID.map(t => coverage(t, n, iv[key], iv.lc)), { color: css(tok), width: 1.6 });
    const c = coverage(cov.theta, n, iv[key], iv.lc);
    ctx.fillStyle = css(tok); ctx.beginPath(); ctx.arc(X(cov.theta), Y(c), 4.5, 0, 2 * Math.PI); ctx.fill();
    parts.push(`${name} <b>${fmt(c, 3)}</b>`);
  }
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(cov.theta), fr.pad.t); ctx.lineTo(X(cov.theta), Y(0)); ctx.stroke();
  $("#covNV").textContent = n; $("#covTV").textContent = fmt(cov.theta, 2);
  caption($("#covCap"), parts.length
    ? `Exact coverage of the "95%" intervals at θ = ${fmt(cov.theta, 2)} with n = ${n}: ${parts.join(", ")}. The dashed line is the promised 95%.`
    : "Tick an interval to draw its coverage.");
}
$("#covN").addEventListener("input", e => { cov.nIdx = +e.target.value; drawCov(); });
$("#covT").addEventListener("input", e => { cov.theta = +e.target.value / 100; drawCov(); });
[["#cW", "wald"], ["#cJ", "jeff"], ["#cF", "flat"]].forEach(([id, key]) => $(id).addEventListener("change", e => { cov.show[key] = e.target.checked; drawCov(); }));

/* =====================================================================
   PRIOR PREDICTIVE CHECK: COMPLAINTS PER DAY
   ===================================================================== */
const PP = { vague: [0.001, 0.001, '"vague" Gamma(0.001, 0.001)'], weak: [2, 0.5, "weakly informative Gamma(2, 0.5)"] };
const pp = { key: "weak", days: null };
const poisson = lam => (lam < 30 ? rng.poisson(lam) : Math.max(0, Math.round(rng.normal(lam, Math.sqrt(lam)))));
function simulatePp() {
  const [a, b] = PP[pp.key], N = 100000, days = new Float64Array(N);
  for (let i = 0; i < N; i++) days[i] = poisson(rng.gamma(a, b));
  pp.days = days.sort();
}
function drawPp() {
  const days = pp.days, N = days.length, MAXK = 30;
  const counts = new Array(MAXK + 2).fill(0);
  for (const d of days) counts[Math.min(MAXK + 1, d)]++;
  const share = counts.map(c => c / N);
  const canvas = $("#ppC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.6 : 2.8, { x: [-0.8, MAXK + 1.8], y: [0, Math.max(...share) * 1.12], yfmt: v => pct(v, 0),
    xticks: [0, 5, 10, 15, 20, 25, 30, 31], xfmt: v => (v === 31 ? "31+" : String(v)), xlabel: "complaints in one day" });
  const bw = Math.max(1, (fr.X(1) - fr.X(0)) * 0.72);
  share.forEach((s, k) => { fr.ctx.fillStyle = k === MAXK + 1 ? css("--warm") : alpha("--post", 0.85); fr.ctx.fillRect(fr.X(k) - bw / 2, fr.Y(s), bw, fr.Y(0) - fr.Y(s)); });
  const q = p => days[Math.min(N - 1, Math.floor(p * N))];
  $$("#pp [data-pp]").forEach(b => b.setAttribute("aria-pressed", b.dataset.pp === pp.key ? "true" : "false"));
  caption($("#ppCap"),
    `100,000 simulated days under the ${PP[pp.key][2]} prior. P(0 complaints) = <b>${fmt(share[0], 2)}</b>; ` +
    `median ${int(q(0.5))}, 90% quantile ${int(q(0.9))}, 99% ${int(q(0.99))}, 99.9% ${int(q(0.999))}. ` +
    (pp.key === "vague"
      ? `Almost every day has no complaints at all, yet ${pct(share[MAXK + 1], 2)} of days have more than 30 (the orange bar), up to hundreds. No café works like that: this prior is not "vague", it is absurd.`
      : "A café with typically a few complaints a day, rarely more than 15: plausible, and still wide enough for the data to move it."));
}
$$("#pp [data-pp]").forEach(b => b.addEventListener("click", () => { pp.key = b.dataset.pp; simulatePp(); drawPp(); }));
$("#ppAgain").addEventListener("click", () => { simulatePp(); drawPp(); });

/* =====================================================================
   SENSITIVITY ANALYSIS
   ===================================================================== */
const SENS_NS = [10, 20, 50, 100, 200, 400];
const SENS_PRIORS = [
  ["flat Beta(1, 1)", 1, 1, "--prior"],
  ["Jeffreys Beta(½, ½)", 0.5, 0.5, "--post"],
  ["Beta(5, 5)", 5, 5, "--warm"],
  ["Beta(2, 8)", 2, 8, "--c4"],
  ["Beta(20, 20)", 20, 20, "--c5"],
];
let sensIdx = 0;
function drawSens() {
  const n = SENS_NS[sensIdx], y = Math.round(0.3 * n);
  $("#sensNV").textContent = `${y} of ${n}`;
  const xs = linspace(0.002, 0.998, 600);
  const curves = SENS_PRIORS.map(([, a, b]) => xs.map(betaDens(a + y, b + n - y)));
  const top = Math.max(...curves.flat()) * 1.1;
  const canvas = $("#sensC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.5 : 2.6, { x: [0, 1], y: [0, top], yticks: false, xlabel: "θ" });
  fr.ctx.strokeStyle = css("--ink"); fr.ctx.setLineDash([4, 4]); fr.ctx.lineWidth = 1;
  fr.ctx.beginPath(); fr.ctx.moveTo(fr.X(0.3), fr.pad.t); fr.ctx.lineTo(fr.X(0.3), fr.Y(0)); fr.ctx.stroke(); fr.ctx.setLineDash([]);
  SENS_PRIORS.forEach(([, , , tok], i) => plotLine(fr, xs, curves[i], { color: css(tok), width: 2 }));
  const rows = SENS_PRIORS.map(([name, a, b, tok]) => {
    const A = a + y, B = b + n - y, [lo, hi] = betaQ(A, B, [0.025, 0.975]);
    return { name, tok, mean: A / (A + B), lo, hi };
  });
  $("#sensT tbody").innerHTML = rows.map(r =>
    `<tr><td><span class="swatch" style="background:var(${r.tok})"></span>${r.name}</td><td>${fmt(r.mean, 3)}</td><td>(${fmt(r.lo, 3)}, ${fmt(r.hi, 3)})</td></tr>`).join("");
  const spread = Math.max(...rows.map(r => r.mean)) - Math.min(...rows.map(r => r.mean));
  caption($("#sensCap"), `The five posterior means span ${fmt(spread, 3)}` +
    (spread > 0.1 ? ": with this little data, the prior visibly changes the answer. Report it." : spread > 0.03 ? ": the priors still matter a little." : ": the data dominate, and the prior is a detail."));
}
$("#sensN").addEventListener("input", e => { sensIdx = +e.target.value; drawSens(); });

/* =====================================================================
   start-up
   ===================================================================== */
$("#elMode").value = 20; $("#elHigh").value = 40;
simulatePp();
drawElicit(); drawHal(); drawLo(); drawCov(); drawPp(); drawSens();
onRedraw(() => { drawElicit(); drawHal(); drawLo(); drawCov(); drawPp(); drawSens(); });
})();
