/* Chapter 02, Random variables and distributions: the live demos.
   Numbers quoted in the text come from the chapter's notebook; these widgets recompute
   the same quantities in the browser. Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, int, caption, frame, plotLine, alpha, reduceMotion, onRedraw } = window.Kit;
const { pdf, cdf, pmf, makeRng, linspace, cumulative, mean, variance } = window.Stats;
const rng = makeRng();
const narrowOf = canvas => canvas.clientWidth < 520;

/* =====================================================================
   THE CENTRAL LIMIT THEOREM (the demo at the top)
   ===================================================================== */
const NS = [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 200, 500];
const MAX_AVERAGES = 100000;
const SOURCES = {
  die: { mu: 3.5, sd: Math.sqrt(35 / 12), lattice: true, support: [1, 6], draw: () => rng.integer(1, 6),
    values: [1, 2, 3, 4, 5, 6].map(v => [v, 1 / 6]), view: [0.5, 6.5] },
  exponential: { mu: 1, sd: 1, support: [0, Infinity], draw: () => rng.exponential(1),
    density: x => pdf.exponential(x, 1), view: [0, 6] },
  rare: { mu: 0.05, sd: Math.sqrt(0.05 * 0.95), lattice: true, support: [0, 1], draw: () => rng.bernoulli(0.05),
    values: [[0, 0.95], [1, 0.05]], view: [-0.5, 1.5] },
  uniform: { mu: 0.5, sd: Math.sqrt(1 / 12), support: [0, 1], draw: () => rng.uniform(),
    density: x => pdf.uniform(x, 0, 1), view: [-0.25, 1.25] },
  humps: { mu: 0, sd: Math.sqrt(4.25), support: [-Infinity, Infinity], draw: () => (rng.uniform() < 0.5 ? -2 : 2) + rng.normal(0, 0.5),
    density: x => 0.5 * pdf.normal(x, -2, 0.5) + 0.5 * pdf.normal(x, 2, 0.5), view: [-5, 5] },
  cauchy: { mu: NaN, sd: NaN, support: [-Infinity, Infinity], draw: () => Math.tan(Math.PI * (rng.uniform() - 0.5)),
    density: x => 1 / (Math.PI * (1 + x * x)), view: [-10, 10], fixed: [-10, 10] },
};
const clt = { source: "exponential", nIdx: 7, avgs: [], timer: null };
const cltSrc = () => SOURCES[clt.source];
const cltN = () => NS[clt.nIdx];

function cltRange() {
  const s = cltSrc(), n = cltN();
  if (s.fixed) return s.fixed;
  let lo = s.mu - 4.5 * s.sd / Math.sqrt(n), hi = s.mu + 4.5 * s.sd / Math.sqrt(n);
  lo = Math.max(lo, s.support[0]); hi = Math.min(hi, s.support[1]);
  if (s.lattice) { lo -= 0.5 / n; hi += 0.5 / n; }
  return [lo, hi];
}
/* Histogram on the density scale: [[left, right, height], ...] */
function cltHistogram() {
  const s = cltSrc(), n = cltN(), N = clt.avgs.length, [lo, hi] = cltRange();
  if (!N) return [];
  const bars = [];
  if (s.lattice) {
    // averages sit on the grid k/n; group neighbouring grid points if there are too many
    const g = Math.max(1, Math.ceil((hi - lo) * n / 120));
    const counts = new Map();
    for (const a of clt.avgs) { const key = Math.floor(Math.round(a * n) / g); counts.set(key, (counts.get(key) || 0) + 1); }
    const width = g / n;
    for (const [key, c] of counts) {
      const left = (key * g - 0.5) / n;
      bars.push([left, left + width, c / (N * width)]);
    }
  } else {
    const B = 70, w = (hi - lo) / B, counts = new Array(B).fill(0);
    for (const a of clt.avgs) { const i = Math.floor((a - lo) / w); if (i >= 0 && i < B) counts[i]++; }
    counts.forEach((c, i) => { if (c) bars.push([lo + i * w, lo + (i + 1) * w, c / (N * w)]); });
  }
  return bars;
}
function drawCltSource() {
  const s = cltSrc(), canvas = $("#cltSrc");
  const [v0, v1] = s.view;
  if (s.values) {
    const top = Math.max(...s.values.map(v => v[1])) * 1.2;
    const fr = frame(canvas, narrowOf(canvas) ? 3 : 5, { x: [v0, v1], y: [0, top], yticks: false, xticks: s.values.map(v => v[0]) });
    const bw = Math.min(28, (fr.X(v0 + 1) - fr.X(v0)) * 0.5);
    fr.ctx.fillStyle = css("--prior");
    for (const [v, p] of s.values) fr.ctx.fillRect(fr.X(v) - bw / 2, fr.Y(p), bw, fr.Y(0) - fr.Y(p));
  } else {
    const xs = linspace(v0, v1, 300), ys = xs.map(s.density);
    const top = Math.max(...ys) * 1.2;
    const fr = frame(canvas, narrowOf(canvas) ? 3 : 5, { x: [v0, v1], y: [0, top], yticks: false });
    plotLine(fr, xs, ys, { color: css("--prior"), fill: alpha("--prior", 0.25) });
  }
}
function drawClt() {
  const s = cltSrc(), n = cltN(), canvas = $("#cltC");
  const [lo, hi] = cltRange();
  const bars = cltHistogram();
  const normalOk = isFinite(s.mu);
  const sdN = s.sd / Math.sqrt(n);
  const peak = normalOk ? pdf.normal(s.mu, s.mu, sdN) : 0;
  let top = Math.max(peak * 1.15, ...bars.map(b => b[2] * 1.08));
  if (!(top > 0)) top = s.density ? s.density(s.mu || 0) * 1.2 : 1;
  const fr = frame(canvas, narrowOf(canvas) ? 1.5 : 2.4, { x: [lo, hi], y: [0, top], yticks: false });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--prior", 0.75);
  for (const [l, r, hgt] of bars) {
    const x0 = Math.max(X(l), fr.pad.l), x1 = Math.min(X(r), fr.w - fr.pad.r);
    if (x1 > x0) ctx.fillRect(x0 + 0.3, Y(Math.min(hgt, top)), Math.max(0.6, x1 - x0 - 0.6), Y(0) - Y(Math.min(hgt, top)));
  }
  if (normalOk) {
    const xs = linspace(lo, hi, 300);
    plotLine(fr, xs, xs.map(x => pdf.normal(x, s.mu, sdN)), { color: css("--post"), width: 2.4 });
  }
  // readouts
  const N = clt.avgs.length;
  const m = N ? mean(clt.avgs) : NaN, sd = N > 1 ? Math.sqrt(variance(clt.avgs)) : NaN;
  $("#rCount").textContent = int(N);
  $("#rMean").textContent = N ? fmt(m, 3) : "–";
  $("#rSd").textContent = N > 1 ? fmt(sd, 3) : "–";
  $("#rMu").textContent = normalOk ? fmt(s.mu, 3) : "does not exist";
  $("#rSdTh").textContent = normalOk ? fmt(sdN, 3) : "does not exist";
  $("#cltNV").textContent = `n = ${n}`;
  let msg;
  if (!N) msg = `Press <b>draw</b>. Each average is of <b>n = ${n}</b> draws, and the histogram collects the averages.`;
  else if (clt.source === "cauchy") msg = `<b>${int(N)}</b> averages of n = ${n} draws, and they are as wild as single draws: the Cauchy has no mean and no variance, so the central limit theorem does not apply. Some averages land far outside this window.`;
  else if (n === 1) msg = `With n = 1, each "average" is a single draw, so the histogram shows the shape of one draw, ${clt.source === "rare" || clt.source === "die" ? "a few separate values" : "not a bell"}. Increase n.`;
  else msg = `<b>${int(N)}</b> averages of n = ${n} draws. Their spread, ${fmt(sd, 3)}, matches the theory's σ/√n = ${fmt(sdN, 3)}. The blue curve is the normal distribution that the central limit theorem predicts${n >= 30 && clt.source !== "rare" ? ", and the histogram hugs it" : ""}.`;
  $("#cltStatus").innerHTML = msg;
  $("#cltDraw").disabled = $("#cltDraw10").disabled = N >= MAX_AVERAGES;
}
function addAverages(count) {
  const s = cltSrc(), n = cltN();
  for (let k = 0; k < count && clt.avgs.length < MAX_AVERAGES; k++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += s.draw();
    clt.avgs.push(sum / n);
  }
}
function drawAverages(total) {
  clearInterval(clt.timer);
  if (reduceMotion || total > 1000) { addAverages(total); drawClt(); return; }
  let left = total;
  clt.timer = setInterval(() => {
    addAverages(Math.min(100, left)); left -= 100;
    drawClt();
    if (left <= 0) clearInterval(clt.timer);
  }, 30);
}
function resetClt() { clearInterval(clt.timer); clt.avgs = []; drawCltSource(); drawClt(); }
$("#cltSource").addEventListener("change", e => { clt.source = e.target.value; resetClt(); });
$("#cltN").addEventListener("input", e => { clt.nIdx = +e.target.value; resetClt(); });
$("#cltDraw").addEventListener("click", () => drawAverages(1000));
$("#cltDraw10").addEventListener("click", () => drawAverages(10000));
$("#cltClear").addEventListener("click", resetClt);
$$("[data-clt]").forEach(b => b.addEventListener("click", () => {
  const [src, n] = b.dataset.clt.split(",");
  clt.source = src; clt.nIdx = NS.indexOf(+n);
  $("#cltSource").value = src; $("#cltN").value = clt.nIdx;
  resetClt(); drawAverages(10000);
  $("#demo").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}));

/* =====================================================================
   PROBABILITY IS AREA
   ===================================================================== */
const AREA = {
  tri: { f: x => (x >= 0 && x <= 1 ? 2 * x : 0), F: x => Math.min(1, Math.max(0, x)) ** 2, view: [0, 1], a: 0.5, b: 1, name: "f(x) = 2x" },
  exp: { f: x => pdf.exponential(x, 1), F: x => cdf.exponential(x, 1), view: [0, 6], a: 1, b: 2, name: "the exponential" },
  normal: { f: x => pdf.normal(x), F: x => cdf.normal(x), view: [-4, 4], a: -1, b: 1, name: "the standard normal" },
  beta: { f: x => pdf.beta(x, 2, 5), view: [0, 1], a: 0.1, b: 0.4, name: "Beta(2, 5)" },
};
// numerical CDF where there is no simple formula
{
  const d = AREA.beta, xs = linspace(0, 1, 2001), Fs = cumulative(d.f, xs);
  d.F = x => { if (x <= 0) return 0; if (x >= 1) return 1; const i = Math.min(1999, Math.floor(x * 2000)); const t = x * 2000 - i; return Fs[i] + t * (Fs[i + 1] - Fs[i]); };
}
const area = { dist: "tri", a: 0.5, b: 1 };
const toSlider = (d, v) => Math.round((v - d.view[0]) / (d.view[1] - d.view[0]) * 1000);
const fromSlider = (d, s) => d.view[0] + (d.view[1] - d.view[0]) * s / 1000;
function syncArea() { const d = AREA[area.dist]; $("#areaA").value = toSlider(d, area.a); $("#areaB").value = toSlider(d, area.b); }
function drawArea() {
  const d = AREA[area.dist];
  const a = Math.min(area.a, area.b), b = Math.max(area.a, area.b);
  const xs = linspace(d.view[0], d.view[1], 400), ys = xs.map(d.f);
  const top = Math.max(...ys) * 1.15;
  const aspect = narrowOf($("#areaPdf")) ? 1.5 : 1.35;
  const fr = frame($("#areaPdf"), aspect, { x: d.view, y: [0, top] });
  const sx = linspace(a, b, 200);
  plotLine(fr, sx, sx.map(d.f), { fill: alpha("--post", 0.28) });
  plotLine(fr, xs, ys, { color: css("--post") });
  for (const v of [a, b]) {
    fr.ctx.strokeStyle = css("--warm"); fr.ctx.lineWidth = 1.5;
    fr.ctx.beginPath(); fr.ctx.moveTo(fr.X(v), fr.Y(0)); fr.ctx.lineTo(fr.X(v), fr.Y(d.f(v))); fr.ctx.stroke();
  }
  const Fa = d.F(a), Fb = d.F(b);
  const gr = frame($("#areaCdf"), aspect, { x: d.view, y: [0, 1.05], yticks: [0, 0.5, 1] });
  plotLine(gr, xs, xs.map(d.F), { color: css("--ink-2") });
  const { ctx, X, Y } = gr;
  ctx.setLineDash([3, 3]); ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1;
  for (const [v, F] of [[a, Fa], [b, Fb]]) { ctx.beginPath(); ctx.moveTo(gr.pad.l, Y(F)); ctx.lineTo(X(v), Y(F)); ctx.lineTo(X(v), Y(0)); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.strokeStyle = css("--warm"); ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(X(b), Y(Fa)); ctx.lineTo(X(b), Y(Fb)); ctx.stroke();
  for (const [v, F] of [[a, Fa], [b, Fb]]) { ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(v), Y(F), 4, 0, 2 * Math.PI); ctx.fill(); }
  $("#areaAV").textContent = fmt(area.a, 2);
  $("#areaBV").textContent = fmt(area.b, 2);
  const over1 = area.dist === "tri" || area.dist === "beta";
  caption($("#areaCap"),
    `[[P(${fmt(a, 2)} \\le X \\le ${fmt(b, 2)}) = F(${fmt(b, 2)}) - F(${fmt(a, 2)}) = ${fmt(Fb, 3)} - ${fmt(Fa, 3)} = ${fmt(Fb - Fa, 3)}]]: ` +
    `the shaded area on the left is the orange jump on the right.` +
    (over1 ? ` Notice that the density of ${d.name} goes above 1: it is probability per unit of x, not a probability.` : "") +
    (area.dist === "normal" && Math.abs(a + 1) < 0.01 && Math.abs(b - 1) < 0.01 ? " Within one standard deviation of the mean: about 68%." : ""));
}
$("#areaDist").addEventListener("change", e => { area.dist = e.target.value; const d = AREA[area.dist]; area.a = d.a; area.b = d.b; syncArea(); drawArea(); });
$("#areaA").addEventListener("input", e => { area.a = fromSlider(AREA[area.dist], +e.target.value); drawArea(); });
$("#areaB").addEventListener("input", e => { area.b = fromSlider(AREA[area.dist], +e.target.value); drawArea(); });

/* =====================================================================
   THE INVERSE-CDF METHOD, FOR f(x) = 2x
   ===================================================================== */
const inv = { xs: [], last: null };
const INV_BINS = 40;
function drawInv() {
  const aspect = narrowOf($("#invF")) ? 1.3 : 1.2;
  const fr = frame($("#invF"), aspect, { x: [0, 1], y: [0, 1], yticks: [0, 0.5, 1], xlabel: "x", ylabel: "U" });
  const grid = linspace(0, 1, 200);
  plotLine(fr, grid, grid.map(x => x * x), { color: css("--ink-2") });
  if (inv.last) {
    const { ctx, X, Y } = fr, u = inv.last, x = Math.sqrt(u);
    ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(X(0), Y(u)); ctx.lineTo(X(x), Y(u)); ctx.lineTo(X(x), Y(0)); ctx.stroke();
    ctx.fillStyle = css("--warm");
    ctx.beginPath(); ctx.arc(X(0), Y(u), 4, 0, 2 * Math.PI); ctx.fill();
    ctx.beginPath(); ctx.arc(X(x), Y(0), 4, 0, 2 * Math.PI); ctx.fill();
    ctx.font = "600 11px " + css("--font-ui"); ctx.textAlign = "left";
    ctx.fillText(`U = ${fmt(u, 2)}`, X(0) + 6, u > 0.85 ? Y(u) + 14 : Y(u) - 6);
    ctx.textAlign = x > 0.8 ? "right" : "center";
    ctx.fillText(`X = ${fmt(x, 2)}`, x > 0.8 ? X(x) - 8 : X(x), Y(0) - 8);
  }
  const N = inv.xs.length, counts = new Array(INV_BINS).fill(0);
  for (const x of inv.xs) counts[Math.min(INV_BINS - 1, Math.floor(x * INV_BINS))]++;
  const dens = counts.map(c => (N ? c / (N / INV_BINS) : 0));
  const hr = frame($("#invH"), aspect, { x: [0, 1], y: [0, Math.max(2.3, ...dens.map(d => d * 1.05))], yticks: [0, 1, 2], xlabel: "x" });
  hr.ctx.fillStyle = alpha("--prior", 0.75);
  dens.forEach((d, i) => { if (d) hr.ctx.fillRect(hr.X(i / INV_BINS) + 0.5, hr.Y(d), hr.X(1 / INV_BINS) - hr.X(0) - 1, hr.Y(0) - hr.Y(d)); });
  plotLine(hr, [0, 1], [0, 2], { color: css("--post"), width: 2.2 });
  if (N) {
    const m = mean(inv.xs);
    hr.ctx.strokeStyle = css("--warm"); hr.ctx.setLineDash([4, 3]); hr.ctx.lineWidth = 1.5;
    hr.ctx.beginPath(); hr.ctx.moveTo(hr.X(m), hr.Y(0)); hr.ctx.lineTo(hr.X(m), hr.pad.t); hr.ctx.stroke(); hr.ctx.setLineDash([]);
  }
  let cap;
  if (!N) cap = "No draws yet. Press <b>draw 1</b> to follow one uniform number through the CDF.";
  else {
    const m = mean(inv.xs), v = variance(inv.xs);
    cap = `<b>${int(N)}</b> draw${N > 1 ? "s" : ""}. Their mean is <b>${fmt(m, 4)}</b> (exact [[\\frac{2}{3} \\approx 0.6667]], the dashed line)` +
      (N > 1 ? ` and their variance <b>${fmt(v, 4)}</b> (exact [[\\frac{1}{18} \\approx 0.0556]]).` : ".") +
      (N < 1000 ? " With more draws, both settle on the exact values and the histogram on the line 2x." : "");
  }
  caption($("#invCap"), cap);
}
function invDraw(k) {
  for (let i = 0; i < k && inv.xs.length < 200000; i++) { const u = rng.uniform(); inv.last = u; inv.xs.push(Math.sqrt(u)); }
  if (k > 1) inv.last = null;
  drawInv();
}
$("#inv1").addEventListener("click", () => invDraw(1));
$("#inv100").addEventListener("click", () => invDraw(100));
$("#inv10k").addEventListener("click", () => invDraw(10000));
$("#invReset").addEventListener("click", () => { inv.xs = []; inv.last = null; drawInv(); });

/* =====================================================================
   DISCRETE FAMILIES
   ===================================================================== */
const disc = { fam: "poisson", p: 0.3, n: 10, lam: 3, k: 5, cmp: false };
function discSpec() {
  const { fam, p, n, lam } = disc;
  switch (fam) {
    case "bernoulli": return { xs: [0, 1], f: k => pmf.bernoulli(k, p), mean: p, varc: p * (1 - p), name: `\\text{Bernoulli}(${fmt(p, 2)})` };
    case "binomial": return { xs: range(0, n), f: k => pmf.binomial(k, n, p), mean: n * p, varc: n * p * (1 - p), name: `\\text{Binomial}(${n}, ${fmt(p, 2)})` };
    case "geometric": {
      let hi = 1, c = 0;
      while (c < 0.995 && hi < 200) { c += pmf.geometric(hi, p); hi++; }
      return { xs: range(1, Math.max(10, hi)), f: k => pmf.geometric(k, p), mean: 1 / p, varc: (1 - p) / (p * p), name: `\\text{Geometric}(${fmt(p, 2)})` };
    }
    default: return { xs: range(0, Math.max(10, Math.ceil(lam + 4.5 * Math.sqrt(lam)))), f: k => pmf.poisson(k, lam), mean: lam, varc: lam, name: `\\text{Poisson}(${fmt(lam, 1).replace(/\.0$/, "")})` };
  }
}
function range(a, b) { return Array.from({ length: b - a + 1 }, (_, i) => a + i); }
function drawDisc() {
  $$("#disc [data-for]").forEach(el => { el.style.display = el.dataset.for.split(" ").includes(disc.fam) ? "" : "none"; });
  const s = discSpec();
  const lo = s.xs[0], hi = s.xs[s.xs.length - 1];
  const kMax = hi;
  $("#discK").max = kMax; $("#discK").min = lo;
  disc.k = Math.min(Math.max(disc.k, lo), kMax);
  $("#discK").value = disc.k;
  $("#discPV").textContent = fmt(disc.p, 2);
  $("#discNV").textContent = disc.n;
  $("#discLV").textContent = fmt(disc.lam, 1);
  $("#discKV").textContent = disc.k;
  const ps = s.xs.map(s.f);
  const showCmp = disc.fam === "binomial" && disc.cmp;
  const poi = showCmp ? s.xs.map(k => pmf.poisson(k, s.mean)) : [];
  const top = Math.max(...ps, ...poi) * 1.15;
  const canvas = $("#discC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.5 : 2.6, { x: [lo - 0.7, hi + 0.7], y: [0, top],
    xticks: s.xs.length <= 16 ? s.xs : undefined, yfmt: v => fmt(v, 2) });
  const { ctx, X, Y } = fr;
  const bw = Math.max(1, Math.min(34, (X(lo + 1) - X(lo)) * 0.7));
  s.xs.forEach((k, i) => {
    ctx.fillStyle = k >= disc.k ? css("--warm") : css("--post");
    ctx.fillRect(X(k) - bw / 2, Y(ps[i]), bw, Y(0) - Y(ps[i]));
  });
  if (showCmp) {
    plotLine(fr, s.xs, poi, { color: css("--ink"), width: 1.4, dash: [4, 3] });
    ctx.fillStyle = css("--ink");
    s.xs.forEach((k, i) => { ctx.beginPath(); ctx.arc(X(k), Y(poi[i]), 3, 0, 2 * Math.PI); ctx.fill(); });
  }
  // mean marker
  ctx.strokeStyle = css("--ink-2"); ctx.lineWidth = 1.5; ctx.setLineDash([2, 3]);
  ctx.beginPath(); ctx.moveTo(X(s.mean), Y(0)); ctx.lineTo(X(s.mean), fr.pad.t); ctx.stroke(); ctx.setLineDash([]);
  const tail = s.xs.reduce((acc, k, i) => acc + (k >= disc.k ? ps[i] : 0), 0);
  caption($("#discCap"),
    `[[${s.name}]]: mean ${fmt(s.mean, 3).replace(/\.?0+$/, "")} (the dotted line), variance ${fmt(s.varc, 3).replace(/\.?0+$/, "")}. ` +
    `The orange bars are [[P(X \\ge ${disc.k}) = ${fmt(tail, 4)}]].` +
    (showCmp ? ` Black dots: Poisson(${fmt(s.mean, 2).replace(/\.?0+$/, "")}), the Poisson with the same mean. The more trials and the smaller p, the closer they get.` : ""));
}
$("#discFam").addEventListener("change", e => { disc.fam = e.target.value; drawDisc(); });
$("#discP").addEventListener("input", e => { disc.p = +e.target.value / 100; drawDisc(); });
$("#discN").addEventListener("input", e => { disc.n = +e.target.value; drawDisc(); });
$("#discL").addEventListener("input", e => { disc.lam = +e.target.value / 10; drawDisc(); });
$("#discK").addEventListener("input", e => { disc.k = +e.target.value; drawDisc(); });
$("#discCmp").addEventListener("change", e => { disc.cmp = e.target.checked; drawDisc(); });
function syncDisc() {
  $("#discFam").value = disc.fam; $("#discP").value = Math.round(disc.p * 100);
  $("#discN").value = disc.n; $("#discL").value = Math.round(disc.lam * 10); $("#discCmp").checked = disc.cmp;
}
$$("[data-disc]").forEach(b => b.addEventListener("click", () => {
  const [what, x, k] = b.dataset.disc.split(",");
  if (what === "poisson") Object.assign(disc, { fam: "poisson", lam: +x, k: +k, cmp: false });
  else Object.assign(disc, { fam: "binomial", n: +x, p: Math.round(300 / +x) / 100, k: +k, cmp: true });
  syncDisc(); drawDisc();
  $("#disc").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
}));

/* =====================================================================
   CONTINUOUS FAMILIES, AND THE SCIPY TRAP
   ===================================================================== */
const FAMS = {
  uniform: { params: [["a", "lower end a", -2, 2, 0.1, 0], ["b", "upper end b", -1, 5, 0.1, 1]], view: () => [-3, 6],
    f: (x, p) => pdf.uniform(x, p.a, p.b), mean: p => (p.a + p.b) / 2, varc: p => (p.b - p.a) ** 2 / 12,
    scipy: p => `stats.uniform(loc=${p.a}, scale=${fmt(p.b - p.a, 1)})` },
  exponential: { params: [["lam", "rate λ", 0.1, 5, 0.1, 1]], view: p => [0, Math.min(30, Math.max(6, 6 / p.lam))],
    f: (x, p) => pdf.exponential(x, p.lam), mean: p => 1 / p.lam, varc: p => 1 / p.lam ** 2,
    scipy: p => `stats.expon(scale=1/${p.lam})`, trap: p => [x => pdf.exponential(x, 1 / p.lam), `stats.expon(scale=${p.lam})`, 1 / (1 / p.lam)] },
  gamma: { params: [["al", "shape α", 0.5, 20, 0.5, 3], ["be", "rate β", 0.2, 5, 0.1, 2]],
    view: p => [0, Math.max(4, p.al / p.be + 5 * Math.sqrt(p.al) / p.be, p.trap ? p.al * p.be + 4 * Math.sqrt(p.al) * p.be : 0)],
    f: (x, p) => pdf.gamma(x, p.al, p.be), mean: p => p.al / p.be, varc: p => p.al / p.be ** 2,
    scipy: p => `stats.gamma(a=${p.al}, scale=1/${p.be})`, trap: p => [x => pdf.gamma(x, p.al, 1 / p.be), `stats.gamma(a=${p.al}, scale=${p.be})`, p.al * p.be] },
  beta: { params: [["al", "α", 0.2, 20, 0.1, 2], ["be", "β", 0.2, 20, 0.1, 5]], view: () => [0, 1],
    f: (x, p) => pdf.beta(x, p.al, p.be), mean: p => p.al / (p.al + p.be),
    varc: p => p.al * p.be / ((p.al + p.be) ** 2 * (p.al + p.be + 1)), scipy: p => `stats.beta(${p.al}, ${p.be})` },
  normal: { params: [["mu", "mean μ", -3, 3, 0.1, 0], ["sd", "standard deviation σ", 0.2, 3, 0.1, 1]], view: () => [-6, 6],
    f: (x, p) => pdf.normal(x, p.mu, p.sd), mean: p => p.mu, varc: p => p.sd ** 2,
    scipy: p => `stats.norm(loc=${p.mu}, scale=${p.sd})`, trap: p => [x => pdf.normal(x, p.mu, p.sd ** 2), `stats.norm(loc=${p.mu}, scale=${fmt(p.sd ** 2, 2)})`, p.mu] },
  t: { params: [["nu", "degrees of freedom ν", 1, 50, 1, 3]], view: () => [-6, 6],
    f: (x, p) => pdf.t(x, p.nu), mean: p => (p.nu > 1 ? 0 : NaN), varc: p => (p.nu > 2 ? p.nu / (p.nu - 2) : p.nu > 1 ? Infinity : NaN),
    scipy: p => `stats.t(df=${p.nu})` },
};
const cont = { fam: "beta", p: {}, trap: false };
const roundTo = (v, step) => +(Math.round(v / step) * step).toFixed(3);
function buildContParams() {
  const F = FAMS[cont.fam];
  cont.p = {};
  $("#contParams").innerHTML = F.params.map(([key, label, lo, hi, step, def]) => {
    cont.p[key] = def;
    return `<label class="field">${label}<span class="row"><input type="range" data-k="${key}" min="${lo}" max="${hi}" step="${step}" value="${def}"><span class="readout-line" data-v="${key}"></span></span></label>`;
  }).join("");
  $$("#contParams input").forEach(inp => inp.addEventListener("input", () => {
    const step = +inp.step;
    cont.p[inp.dataset.k] = roundTo(+inp.value, step);
    if (cont.fam === "uniform" && cont.p.b <= cont.p.a + 0.05) cont.p.b = roundTo(cont.p.a + 0.1, 0.1);
    drawCont();
  }));
  $("#contTrapRow").style.display = F.trap ? "" : "none";
}
function drawCont() {
  const F = FAMS[cont.fam], p = cont.p;
  const showTrap = cont.trap && F.trap;
  const [v0, v1] = F.view({ ...p, trap: showTrap });
  const xs = linspace(v0, v1, 500);
  const ys = xs.map(x => F.f(x, p));
  const inner = ys.slice(5, -5).filter(isFinite);
  let top = Math.min(6, Math.max(...inner) * 1.15);
  let trapYs = null;
  if (showTrap) { const [g] = F.trap(p); trapYs = xs.map(g); top = Math.max(top, Math.min(6, Math.max(...trapYs.filter(isFinite)) * 1.15)); }
  const canvas = $("#contC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.5 : 2.6, { x: [v0, v1], y: [0, top], yfmt: v => fmt(v, 1) });
  if (cont.fam === "t") plotLine(fr, xs, xs.map(x => pdf.normal(x)), { color: css("--prior"), width: 1.6, dash: [5, 4] });
  plotLine(fr, xs, ys.map(y => (isFinite(y) ? y : top * 2)), { color: css("--post"), width: 2.4, fill: alpha("--post", 0.15) });
  if (trapYs) plotLine(fr, xs, trapYs, { color: css("--warm"), width: 2, dash: [6, 4] });
  for (const [key] of F.params) $(`#contParams [data-v="${key}"]`).textContent = String(p[key]);
  const m = F.mean(p), v = F.varc(p);
  const show = x => (isNaN(x) ? "undefined" : x === Infinity ? "infinite" : fmt(x, 3).replace(/\.?0+$/, ""));
  $("#contScipy").innerHTML = `in scipy: <code>${F.scipy(p)}</code>`;
  let cap = `Mean <b>${show(m)}</b>, variance <b>${show(v)}</b>.`;
  if (cont.fam === "t") cap += ` The dashed grey curve is the standard normal: the t has heavier tails, and it approaches the normal as ν grows.${p.nu <= 2 ? " With ν ≤ 2 the tails are so heavy that the variance is infinite." : ""}`;
  if (cont.fam === "beta" && p.al < 1 && p.be < 1) cap += " With both parameters below 1, the density piles up at both ends: a proportion that is probably close to 0 or close to 1.";
  if (cont.fam === "beta" && p.al > 1 && p.be > 1) cap += ` Its peak (mode) is at ${fmt((p.al - 1) / (p.al + p.be - 2), 3)}, as exercise 5 shows.`;
  if (showTrap) {
    const [, call, wrongMean] = F.trap(p);
    const same = cont.fam === "normal" ? Math.abs(p.sd - 1) < 1e-9 : cont.fam === "gamma" ? Math.abs(p.be - 1) < 1e-9 : Math.abs(p.lam - 1) < 1e-9;
    cap += same ? " With this setting the mistake happens to be invisible (rate 1 or sd 1); change it to see the dashed curve move."
      : ` Dashed orange: what <code>${call}</code> gives instead, ${cont.fam === "normal" ? `a spread of ${fmt(p.sd ** 2, 2)} instead of ${p.sd}` : `a mean of ${show(wrongMean)} instead of ${show(m)}`}. It looks perfectly reasonable, which is what makes the mistake dangerous.`;
  }
  caption($("#contCap"), cap);
}
$("#contFam").addEventListener("change", e => { cont.fam = e.target.value; buildContParams(); drawCont(); });
$("#contTrap").addEventListener("change", e => { cont.trap = e.target.checked; drawCont(); });

/* =====================================================================
   FROM A GRID OF HYPOTHESES TO A DENSITY
   ===================================================================== */
const GRIDS = [6, 11, 21, 51, 101, 1001];
const CURED = 3, PATIENTS = 10;
let gridIdx = 1;
function drawGrid() {
  const m = GRIDS[gridIdx], spacing = 1 / (m - 1);
  const thetas = linspace(0, 1, m);
  const un = thetas.map(t => t ** CURED * (1 - t) ** (PATIENTS - CURED));
  const s = un.reduce((a, b) => a + b, 0);
  const dens = un.map(u => u / s / spacing);              // probability per unit of theta
  const xs = linspace(0, 1, 400), curve = xs.map(x => pdf.beta(x, CURED + 1, PATIENTS - CURED + 1));
  const top = Math.max(...curve, ...dens) * 1.12;
  const canvas = $("#gridC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.6 : 2.6, { x: [-0.02, 1.02], y: [0, top], xticks: [0, 0.25, 0.5, 0.75, 1], xfmt: v => `${Math.round(v * 100)}%`, yfmt: v => fmt(v, 1) });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--post", m > 200 ? 0.5 : 0.8);
  thetas.forEach((t, i) => {
    const l = Math.max(0, t - spacing / 2), r = Math.min(1, t + spacing / 2);
    ctx.fillRect(X(l) + (m <= 101 ? 0.5 : 0), Y(dens[i]), Math.max(0.5, X(r) - X(l) - (m <= 101 ? 1 : 0)), Y(0) - Y(dens[i]));
  });
  plotLine(fr, xs, curve, { color: css("--warm"), width: 2.2 });
  $("#gridV").textContent = int(m);
  caption($("#gridCap"),
    `${int(m)} hypotheses, ${fmt(spacing, m > 101 ? 3 : 2)} apart. Each bar is its posterior probability divided by the spacing; ` +
    `the orange curve is the Beta(4, 8) density, [[p(\\theta \\mid y) \\propto \\theta^{3} (1 - \\theta)^{7}]]. ` +
    (m <= 11 ? "With so few hypotheses the bars are a coarse staircase." : m >= 1001 ? "The bars and the curve are now indistinguishable: in the limit, the sum is the integral." : "The finer the grid, the closer the bars come to the curve."));
}
$("#gridM").addEventListener("input", e => { gridIdx = +e.target.value; drawGrid(); });

/* =====================================================================
   start-up
   ===================================================================== */
$("#cltN").value = clt.nIdx;
syncArea();
syncDisc();
buildContParams();
drawCltSource(); drawClt();
drawArea(); drawInv(); drawDisc(); drawCont(); drawGrid();
onRedraw(() => { drawCltSource(); drawClt(); drawArea(); drawInv(); drawDisc(); drawCont(); drawGrid(); });
})();
