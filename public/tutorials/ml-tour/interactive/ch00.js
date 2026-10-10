/* A Tour of Machine Learning, chapter 00, What machine learning is: the live demos.
   The notebook's experiment: polynomials in the Legendre basis on [0, 1], fitted by least
   squares (Householder QR) or ridge regression (Cholesky), with 5-fold cross-validation.
   Each demo draws its own random data, so its numbers vary a little from the notebook's.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, int, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { makeRng } = window.Stats;

/* ---------------- the notebook's model ---------------- */
const SIGMA = 0.3, f = x => Math.sin(2 * Math.PI * x);
function features(x, degree) {                                                // legendre.legvander(2x - 1, degree)
  const t = 2 * x - 1, row = [1];
  if (degree >= 1) row.push(t);
  for (let k = 1; k < degree; k++) row.push(((2 * k + 1) * t * row[k] - k * row[k - 1]) / (k + 1));
  return row;
}
/* least squares by Householder QR (needs n >= degree + 1) */
function lstsq(X, y) {
  const n = X.length, m = X[0].length, A = X.map(r => r.slice()), b = y.slice();
  for (let j = 0; j < m; j++) {
    let norm = 0; for (let i = j; i < n; i++) norm += A[i][j] * A[i][j]; norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const alpha0 = A[j][j] > 0 ? -norm : norm, v = new Float64Array(n);
    for (let i = j; i < n; i++) v[i] = A[i][j]; v[j] -= alpha0;
    let vv = 0; for (let i = j; i < n; i++) vv += v[i] * v[i];
    if (vv === 0) continue;
    for (let c = j; c < m; c++) { let s = 0; for (let i = j; i < n; i++) s += v[i] * A[i][c]; s *= 2 / vv; for (let i = j; i < n; i++) A[i][c] -= s * v[i]; }
    let s = 0; for (let i = j; i < n; i++) s += v[i] * b[i]; s *= 2 / vv; for (let i = j; i < n; i++) b[i] -= s * v[i];
  }
  const w = new Float64Array(m);
  for (let j = m - 1; j >= 0; j--) { let s = b[j]; for (let c = j + 1; c < m; c++) s -= A[j][c] * w[c]; w[j] = s / A[j][j]; }
  return w;
}
/* ridge: (X'X + lam P) w = X'y, P = diag(0, 1, ..., 1), by Cholesky */
function ridge(X, y, lam) {
  const m = X[0].length, G = Array.from({ length: m }, () => new Float64Array(m)), r = new Float64Array(m);
  X.forEach((row, i) => { for (let a = 0; a < m; a++) { r[a] += row[a] * y[i]; for (let c = 0; c <= a; c++) G[a][c] += row[a] * row[c]; } });
  for (let a = 1; a < m; a++) G[a][a] += lam;
  for (let j = 0; j < m; j++) {
    let s = G[j][j]; for (let k = 0; k < j; k++) s -= G[j][k] * G[j][k]; G[j][j] = Math.sqrt(s);
    for (let i = j + 1; i < m; i++) { let v = G[i][j]; for (let k = 0; k < j; k++) v -= G[i][k] * G[j][k]; G[i][j] = v / G[j][j]; }
  }
  const z = new Float64Array(m), w = new Float64Array(m);
  for (let i = 0; i < m; i++) { let v = r[i]; for (let k = 0; k < i; k++) v -= G[i][k] * z[k]; z[i] = v / G[i][i]; }
  for (let i = m - 1; i >= 0; i--) { let v = z[i]; for (let k = i + 1; k < m; k++) v -= G[k][i] * w[k]; w[i] = v / G[i][i]; }
  return w;
}
function fit(x, y, degree, lam = 0) { const X = x.map(v => features(v, degree)); return lam === 0 ? lstsq(X, y) : ridge(X, y, lam); }
const predict = (w, x) => { const p = features(x, w.length - 1); let s = 0; for (let k = 0; k < p.length; k++) s += p[k] * w[k]; return s; };
const rmse = (w, x, y) => Math.sqrt(x.reduce((s, v, i) => s + (predict(w, v) - y[i]) ** 2, 0) / x.length);
/* 5-fold cross-validation, folds as np.array_split: the first n % k folds get one extra point */
function cvRmse(x, y, degree, lam = 0, k = 5) {
  const n = x.length, base = Math.floor(n / k), extra = n % k;
  let start = 0, sse = [];
  for (let f2 = 0; f2 < k; f2++) {
    const size = base + (f2 < extra ? 1 : 0), test = [], train = [];
    for (let i = 0; i < n; i++) (i >= start && i < start + size ? test : train).push(i);
    start += size;
    const w = fit(train.map(i => x[i]), train.map(i => y[i]), degree, lam);
    sse.push(test.reduce((s, i) => s + (predict(w, x[i]) - y[i]) ** 2, 0) / test.length);
  }
  return Math.sqrt(sse.reduce((a, b) => a + b, 0) / k);
}
function sample(n, rng) { const x = Array.from({ length: n }, () => rng.uniform()); return { x, y: x.map(v => f(v) + rng.normal(0, SIGMA)) }; }
const TEST = sample(5000, makeRng(99));
const GRID = Array.from({ length: 201 }, (_, i) => i / 200);
const L10 = Math.log10;
function drawTruth(fr) { plotLine(fr, GRID, GRID.map(f), { color: css("--muted"), width: 1.4, dash: [6, 4] }); }
function dots(fr, x, y, col, r = 3.2) { const { ctx, X, Y } = fr; ctx.fillStyle = col; x.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(v), Y(y[i]), r, 0, 2 * Math.PI); ctx.fill(); }); }

/* =====================================================================
   THE DEMO: BEND A POLYNOMIAL UNTIL IT BREAKS
   ===================================================================== */
const hero = { seed: 1, all: null };
const MAXD = 14;
function updateHero() {
  hero.all = hero.all || sample(100, makeRng(hero.seed));
  const n = +$("#hN").value, maxd = Math.min(MAXD, n - 1), dIn = $("#hD");
  dIn.max = String(maxd); if (+dIn.value > maxd) dIn.value = String(maxd);
  const d = +dIn.value, x = hero.all.x.slice(0, n), y = hero.all.y.slice(0, n);
  $("#hDV").textContent = String(d); $("#hNV").textContent = String(n);
  const curves = Array.from({ length: maxd + 1 }, (_, k) => { const w = fit(x, y, k); return { w, tr: rmse(w, x, y), te: rmse(w, TEST.x, TEST.y) }; });
  const best = curves.reduce((b, c, k) => (c.te < curves[b].te ? k : b), 0), cur = curves[d];
  const narrow = narrowOf($("#fitC"));
  const fr = frame($("#fitC"), narrow ? 1.35 : 2.2, { x: [0, 1], y: [-2, 2], xlabel: "x", ylabel: "y", yticks: [-2, -1, 0, 1, 2], xfmt: v => fmt(v, 1) });
  drawTruth(fr);
  plotLine(fr, GRID, GRID.map(v => clamp(predict(cur.w, v), -5, 5)), { color: css("--post"), width: 2.4 });
  dots(fr, x, y, css("--warm"), n > 50 ? 2.4 : 3.4);
  const top = 3, bot = L10(0.05);                                              // log10 of 1000 and 0.05
  const er = frame($("#errC"), narrow ? 1.5 : 2.6, { x: [-0.5, maxd + 0.5], y: [bot, top], xlabel: "polynomial degree", ylabel: "RMSE",
    yticks: [-1, 0, 1, 2, 3], yfmt: v => String(10 ** v), xticks: Array.from({ length: maxd + 1 }, (_, k) => k).filter(k => k % 2 === 0 || maxd < 10) });
  const { ctx, X, Y } = er;
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1.2; ctx.setLineDash([2, 4]); seg(ctx, X(-0.5), Y(L10(SIGMA)), X(maxd + 0.5), Y(L10(SIGMA))); ctx.setLineDash([]);
  ctx.font = "600 10px " + css("--font-ui"); ctx.fillStyle = css("--muted"); ctx.textAlign = "left"; ctx.fillText("noise level 0.3", X(-0.4), Y(L10(SIGMA)) + 12);
  ctx.fillStyle = alpha("--ink", 0.07); ctx.fillRect(X(d - 0.45), er.pad.t, X(d + 0.45) - X(d - 0.45), er.ih);
  const ks = curves.map((_, k) => k);
  plotLine(er, ks, curves.map(c => clamp(L10(c.tr), bot, top)), { color: css("--post"), width: 2 });
  plotLine(er, ks, curves.map(c => clamp(L10(c.te), bot, top)), { color: css("--warm"), width: 2 });
  curves.forEach((c, k) => { [[c.tr, "--post"], [c.te, "--warm"]].forEach(([v, col]) => { ctx.fillStyle = css(col); ctx.beginPath(); ctx.arc(X(k), Y(clamp(L10(v), bot, top)), k === d ? 5 : 3, 0, 2 * Math.PI); ctx.fill(); }); });
  const verdict = cur.te > 1.15 * curves[best].te ? (d < best ? "underfits" : "overfits") : "about right";
  $("#rTr").textContent = fmt(cur.tr, 3); $("#rTe").textContent = cur.te < 1000 ? fmt(cur.te, 3) : cur.te.toExponential(1);
  $("#rBest").textContent = `${best} (test ${fmt(curves[best].te, 3)})`; $("#rV").textContent = verdict;
  let text = `Degree ${d}: training error <b>${fmt(cur.tr, 3)}</b>, test error <b>${cur.te < 1000 ? fmt(cur.te, 3) : cur.te.toExponential(1)}</b>. `;
  if (verdict === "underfits") text += `Too rigid: the curve misses the shape of the data, and errs even on the ${n} points it was fitted to.`;
  else if (verdict === "overfits") text += cur.tr < SIGMA ? `The fit is tighter than the noise itself (training error below 0.3): it is following the noise in these ${n} points, which new data will not share.` : `More flexible than these ${n} points can support: the fit chases the noise, and new points pay for it.`;
  else text += `Close to the best this data allows: the test error is near the noise level, 0.3, which no model can beat.`;
  text += ` The lowest test error is at degree ${best}.`;
  if (n >= 60 && d >= 8 && verdict !== "overfits") text += ` With ${n} points, even a flexible polynomial is held in place: more data cuts the variance.`;
  caption($("#heroStatus"), text);
}
["#hD", "#hN"].forEach(id => $(id).addEventListener("input", updateHero));
$("#hNew").addEventListener("click", () => { hero.seed++; hero.all = null; updateHero(); drawRidge(true); });

/* =====================================================================
   PART 1: THREE KINDS OF QUESTION
   ===================================================================== */
const FAM = (() => {
  const r = makeRng(4);
  const reg = Array.from({ length: 40 }, () => { const x = r.uniform(); return [x, 0.3 + 0.9 * x + r.normal(0, 0.12)]; });
  const blob = (cx, cy, s, k) => Array.from({ length: k }, () => [cx + r.normal(0, s), cy + r.normal(0, s)]);
  const cls = [...blob(0.32, 0.62, 0.11, 30).map(p => [...p, 0]), ...blob(0.66, 0.36, 0.11, 30).map(p => [...p, 1])];
  const clu = [...blob(0.25, 0.3, 0.07, 25), ...blob(0.72, 0.32, 0.08, 25), ...blob(0.5, 0.75, 0.07, 25)];
  return { reg, cls, clu };
})();
let famMode = "reg";
function kmeans(pts, cents, it = 10) {
  let lab = [];
  for (let t = 0; t < it; t++) {
    lab = pts.map(p => cents.reduce((b, c, k) => ((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 < (p[0] - cents[b][0]) ** 2 + (p[1] - cents[b][1]) ** 2 ? k : b), 0));
    cents = cents.map((c, k) => { const m = pts.filter((_, i) => lab[i] === k); return m.length ? [m.reduce((s, p) => s + p[0], 0) / m.length, m.reduce((s, p) => s + p[1], 0) / m.length] : c; });
  }
  return { lab, cents };
}
function drawFam() {
  $$("#fam [data-fam]").forEach(b => b.setAttribute("aria-pressed", b.dataset.fam === famMode ? "true" : "false"));
  const cv = $("#famC"), fr = frame(cv, narrowOf(cv) ? 1.3 : 2.4, { x: [0, 1], y: [0, 1.25], xticks: false, yticks: false });
  const { ctx, X, Y } = fr, cols = [css("--post"), css("--warm"), css("--c4")];
  let text;
  if (famMode === "reg") {
    const xs = FAM.reg.map(p => p[0]), ys = FAM.reg.map(p => p[1]), w = fit(xs, ys, 1);
    dots(fr, xs, ys, css("--warm"));
    plotLine(fr, [0, 1], [predict(w, 0), predict(w, 1)], { color: css("--post"), width: 2.4 });
    text = "<b>Regression</b>: every point comes with a numerical answer (its height), and the model learns to predict the answer from the input: here, a line. Chapters 01 and 08.";
  } else if (famMode === "cls") {
    const m0 = [0, 0], m1 = [0, 0]; let n0 = 0, n1 = 0;
    FAM.cls.forEach(p => { if (p[2]) { m1[0] += p[0]; m1[1] += p[1]; n1++; } else { m0[0] += p[0]; m0[1] += p[1]; n0++; } });
    m0[0] /= n0; m0[1] /= n0; m1[0] /= n1; m1[1] /= n1;
    const mx = (m0[0] + m1[0]) / 2, my = (m0[1] + m1[1]) / 2, dx = m1[0] - m0[0], dy = m1[1] - m0[1];
    const yAt = x => my - dx * (x - mx) / dy;                                  // the perpendicular bisector of the two means
    ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.8; ctx.setLineDash([6, 4]); seg(ctx, X(0), Y(yAt(0)), X(1), Y(yAt(1))); ctx.setLineDash([]); ctx.restore();
    FAM.cls.forEach(p => { ctx.fillStyle = cols[p[2]]; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.4, 0, 2 * Math.PI); ctx.fill(); });
    text = "<b>Classification</b>: every point comes with a category (its colour), and the model learns a boundary that separates the categories, so that a new point can be labelled by which side it falls on. Chapters 03, 04, 06, 07 and 11.";
  } else {
    const { lab, cents } = kmeans(FAM.clu, [[0.1, 0.1], [0.9, 0.1], [0.5, 1.0]]);
    FAM.clu.forEach((p, i) => { ctx.fillStyle = cols[lab[i]]; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.4, 0, 2 * Math.PI); ctx.fill(); });
    ctx.fillStyle = css("--ink"); ctx.font = "700 16px " + css("--font-ui"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
    cents.forEach(c => ctx.fillText("×", X(c[0]), Y(c[1])));
    ctx.textBaseline = "alphabetic";
    text = "<b>Clustering</b>: the points come with no answers at all. The algorithm finds the groups by itself (here, k-means, with a cross at each group's centre), and the colours are its discovery, not given labels. Chapter 05; chapter 09 is unsupervised too.";
  }
  caption($("#famCap"), text);
}
$$("#fam [data-fam]").forEach(b => b.addEventListener("click", () => { famMode = b.dataset.fam; drawFam(); }));

/* =====================================================================
   PART 4: BIAS AND VARIANCE
   ===================================================================== */
const BV = { seed: 7 }, BV_SETS = 2000, BV_GRID = Array.from({ length: 101 }, (_, i) => i / 100);
function drawBv() {
  const d = +$("#bD").value, n = +$("#bN").value;
  $("#bDV").textContent = String(d); $("#bNV").textContent = String(n);
  const r = makeRng(BV.seed), preds = [], gaps = [];
  for (let s = 0; s < BV_SETS; s++) {
    const tr = sample(n, r), w = fit(tr.x, tr.y, d);
    preds.push(BV_GRID.map(v => predict(w, v)));
    gaps.push(Math.max(Math.min(...tr.x), 1 - Math.max(...tr.x)));   // the widest empty stretch at an end of [0, 1]
  }
  const mean = BV_GRID.map((_, j) => preds.reduce((a, p) => a + p[j], 0) / BV_SETS);
  const bias2 = BV_GRID.reduce((a, v, j) => a + (mean[j] - f(v)) ** 2, 0) / BV_GRID.length;
  // each set's share of the variance (the shares average to it), and the worst 1% of sets
  const share = preds.map(p => p.reduce((a, v, j) => a + (v - mean[j]) ** 2, 0) / BV_GRID.length);
  const variance = share.reduce((a, v) => a + v, 0) / BV_SETS;
  const worst = share.map((v, i) => i).sort((a, b) => share[b] - share[a]).slice(0, BV_SETS / 100);
  const topShare = worst.reduce((a, i) => a + share[i], 0) / (variance * BV_SETS);
  const median = v => { const s = [...v].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
  const cv = $("#bvC"), fr = frame(cv, narrowOf(cv) ? 1.35 : 2.4, { x: [0, 1], y: [-2.2, 2.2], xlabel: "x", yticks: [-2, -1, 0, 1, 2], xfmt: v => fmt(v, 1) });
  for (let s = 0; s < 30; s++) plotLine(fr, BV_GRID, preds[s].map(v => clamp(v, -5, 5)), { color: alpha("--post", 0.28), width: 1 });
  drawTruth(fr);
  plotLine(fr, BV_GRID, mean.map(v => clamp(v, -5, 5)), { color: css("--post"), width: 2.6 });
  const total = bias2 + variance + SIGMA ** 2, f4 = v => (v < 0.001 ? v.toExponential(1) : v < 100 ? fmt(v, v < 1 ? 3 : 1) : v.toExponential(1));
  const lead = bias2 > variance ? "Bias dominates: the fits agree with each other, but their average misses the true curve."
    : bias2 < 0.05 ? "Variance dominates: the average fit follows the true curve, but each fit swings with its own training set."
    : "Variance dominates: the wildest fits swing so far that they drag even the average away from the true curve.";
  caption($("#bvCap"),
    `Over ${int(BV_SETS)} training sets of ${n} points, at degree ${d}: bias² <b>${f4(bias2)}</b>, variance <b>${f4(variance)}</b>, noise 0.09, so an expected test MSE of ${f4(total)}. ${lead}` +
    (topShare > 0.4 ? ` The worst 1% of the training sets carry ${pct(topShare, 0)} of this variance: unlucky draws that leave an empty stretch at one end (median ${fmt(median(worst.map(i => gaps[i])))}, against ${fmt(median(gaps))} over all sets), where the polynomial shoots off. Another batch would give a quite different value.` : "") +
    (n >= 50 && d >= 5 && topShare <= 0.4 ? " With this many points per set, the high-degree fits are held in place: more data cuts the variance, not the bias." : ""));
}
["#bD", "#bN"].forEach(id => $(id).addEventListener("input", drawBv));
$("#bAgain").addEventListener("click", () => { BV.seed++; drawBv(); });

/* =====================================================================
   PART 5: RIDGE REGRESSION AT DEGREE 12
   ===================================================================== */
const LAMS = Array.from({ length: 33 }, (_, i) => -6 + 0.25 * i), RG = { curves: null, x: null, y: null };
function drawRidge(resample = false) {
  if (resample || !RG.curves) {
    const all = hero.all || sample(100, makeRng(hero.seed));
    RG.x = all.x.slice(0, 15); RG.y = all.y.slice(0, 15);
    RG.curves = LAMS.map(l => { const lam = 10 ** l, w = fit(RG.x, RG.y, 12, lam); return { tr: rmse(w, RG.x, RG.y), te: rmse(w, TEST.x, TEST.y), cv: cvRmse(RG.x, RG.y, 12, lam) }; });
  }
  const li = Math.round((+$("#gL").value + 6) / 0.25), lam = 10 ** LAMS[li], c = RG.curves[li];
  $("#gLV").textContent = lam >= 0.01 ? fmt(lam, lam < 1 ? 2 : 0) : lam.toExponential(0);
  const w = fit(RG.x, RG.y, 12, lam), narrow = narrowOf($("#rgFit"));
  const fr = frame($("#rgFit"), narrow ? 1.25 : 1.15, { x: [0, 1], y: [-2, 2], xlabel: "x", yticks: [-2, -1, 0, 1, 2], xfmt: v => fmt(v, 1) });
  drawTruth(fr);
  plotLine(fr, GRID, GRID.map(v => clamp(predict(w, v), -5, 5)), { color: css("--post"), width: 2.4 });
  dots(fr, RG.x, RG.y, css("--warm"));
  const bot = L10(0.05), top = 2;
  const er = frame($("#rgErr"), narrow ? 1.25 : 1.15, { x: [-6, 2], y: [bot, top], xlabel: "penalty λ", xticks: [-6, -4, -2, 0, 2], xfmt: v => (v === 0 ? "1" : `1e${v}`), yticks: [-1, 0, 1, 2], yfmt: v => String(10 ** v) });
  const { ctx, X, Y } = er;
  ctx.strokeStyle = alpha("--ink", 0.5); ctx.lineWidth = 1; ctx.setLineDash([3, 3]); seg(ctx, X(LAMS[li]), er.pad.t, X(LAMS[li]), Y(bot)); ctx.setLineDash([]);
  [["tr", "--post"], ["cv", "--c4"], ["te", "--warm"]].forEach(([k, col]) => plotLine(er, LAMS, RG.curves.map(q => clamp(L10(q[k]), bot, top)), { color: css(col), width: 2 }));
  const bestCv = RG.curves.reduce((b, q, i) => (q.cv < RG.curves[b].cv ? i : b), 0);
  caption($("#rgCap"),
    `With [[\\lambda = ${$("#gLV").textContent}]]: training error ${fmt(c.tr, 3)}, cross-validation ${c.cv < 100 ? fmt(c.cv, 3) : c.cv.toExponential(1)}, test error <b>${c.te < 100 ? fmt(c.te, 3) : c.te.toExponential(1)}</b>. ` +
    (LAMS[li] <= -4 ? "So little penalty that the degree-12 polynomial still chases the noise. " : LAMS[li] >= 1 ? "So much penalty that the curve is flattened: underfitting again. " : "The penalty holds the polynomial near a smooth curve. ") +
    `Cross-validation, which never sees the test points, would pick [[\\lambda = ${(10 ** LAMS[bestCv]) >= 0.01 ? fmt(10 ** LAMS[bestCv], 10 ** LAMS[bestCv] < 1 ? 2 : 0) : (10 ** LAMS[bestCv]).toExponential(0)}]], with a test error of ${fmt(RG.curves[bestCv].te, 3)}.`);
}
$("#gL").addEventListener("input", () => drawRidge());

/* =====================================================================
   start-up
   ===================================================================== */
updateHero(); drawFam(); drawBv(); drawRidge();
onRedraw(() => { updateHero(); drawFam(); drawBv(); drawRidge(); });
})();
