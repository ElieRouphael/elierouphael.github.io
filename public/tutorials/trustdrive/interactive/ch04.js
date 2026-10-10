/* TrustDrive chapter 04, Uncertainty and ensembles: the live demos.
   The ensemble is the notebook's recipe: ridge regressions on 200 random cosine features,
   each fitted to its own bootstrap resample. It trains here with its own random draws,
   so its numbers vary a little from the notebook's. Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf } = window.Kit;
const { makeRng, erf } = window.Stats;
const f = x => Math.sin(1.5 * x), D_FEAT = 200, SCALE = 1.5, LAM = 1e-2, N_TRAIN = 300, N_EVAL = 2000;
const XS = Array.from({ length: 400 }, (_, i) => -5 + 10 * i / 399);
const uniform = (rng, lo, hi, n) => Float64Array.from({ length: n }, () => lo + (hi - lo) * rng.uniform());
const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
function seg(ctx, x0, y0, x1, y1) { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }

/* ---------------- one member: random cosine features, ridge regression ---------------- */
function drawFeatures(rng, D) {
  return { W: Float64Array.from({ length: D }, () => rng.normal(0, SCALE)), b: Float64Array.from({ length: D }, () => 2 * Math.PI * rng.uniform()) };
}
/* Solve (P^T P + lam I) w = P^T y by Cholesky. P holds the features of the resampled inputs, row by row. */
function fitRidge(feat, X, Y, idx) {
  const D = feat.W.length, n = idx.length, P = new Float64Array(n * D), G = new Float64Array(D * D), r = new Float64Array(D);
  for (let k = 0; k < n; k++) { const x = X[idx[k]]; for (let j = 0; j < D; j++) P[k * D + j] = Math.cos(x * feat.W[j] + feat.b[j]); }
  for (let k = 0; k < n; k++) {
    const y = Y[idx[k]], row = k * D;
    for (let i = 0; i < D; i++) { const pi = P[row + i]; r[i] += pi * y; const gi = i * D; for (let j = 0; j <= i; j++) G[gi + j] += pi * P[row + j]; }
  }
  for (let i = 0; i < D; i++) G[i * D + i] += LAM;
  for (let j = 0; j < D; j++) {                                    // Cholesky, lower triangle in place
    let s = G[j * D + j];
    for (let k = 0; k < j; k++) s -= G[j * D + k] * G[j * D + k];
    const d = Math.sqrt(s); G[j * D + j] = d;
    for (let i = j + 1; i < D; i++) { let t = G[i * D + j]; for (let k = 0; k < j; k++) t -= G[i * D + k] * G[j * D + k]; G[i * D + j] = t / d; }
  }
  const z = new Float64Array(D), w = new Float64Array(D);
  for (let i = 0; i < D; i++) { let t = r[i]; for (let k = 0; k < i; k++) t -= G[i * D + k] * z[k]; z[i] = t / G[i * D + i]; }
  for (let i = D - 1; i >= 0; i--) { let t = z[i]; for (let k = i + 1; k < D; k++) t -= G[k * D + i] * w[k]; w[i] = t / G[i * D + i]; }
  return w;
}
function predict(feat, w, xs) {
  const out = new Float64Array(xs.length), D = w.length;
  for (let k = 0; k < xs.length; k++) { let s = 0; const x = xs[k]; for (let j = 0; j < D; j++) s += w[j] * Math.cos(x * feat.W[j] + feat.b[j]); out[k] = s; }
  return out;
}
/* mean and population variance across members (np.mean(0), np.var(0)) */
function moments(rows) {
  const M = rows.length, n = rows[0].length, mu = new Float64Array(n), v = new Float64Array(n);
  for (const r of rows) for (let k = 0; k < n; k++) mu[k] += r[k] / M;
  for (const r of rows) for (let k = 0; k < n; k++) v[k] += (r[k] - mu[k]) ** 2 / M;
  return { mu, v };
}

/* =====================================================================
   THE DEMO: TRAIN AN ENSEMBLE, MEMBER BY MEMBER
   ===================================================================== */
const ens = { seed: 1, gen: 0, members: [], data: null, stats: null };
function makeData() {
  const a = +$("#eA").value, noise = +$("#eN").value, rng = makeRng(ens.seed * 7919 + 13);
  const set = (lo, hi, n) => { const X = uniform(rng, lo, hi, n); return { X, Y: X.map(x => f(x) + rng.normal(0, noise)) }; };
  ens.data = { a, noise, tr: set(-a, a, N_TRAIN), val: set(-a, a, N_TRAIN), inD: set(-a, a, N_EVAL), outD: set(a + 1, a + 3, N_EVAL), te: set(-(a + 1), a + 1, N_EVAL) };
  ens.members = [];
}
function trainMember(s) {
  const rng = makeRng(ens.seed * 100003 + 1000 + s), d = ens.data;
  const feat = drawFeatures(rng, D_FEAT);
  const idx = Int32Array.from({ length: N_TRAIN }, () => Math.floor(rng.uniform() * N_TRAIN));   // bootstrap resample
  const w = fitRidge(feat, d.tr.X, d.tr.Y, idx);
  return { xs: predict(feat, w, XS), val: predict(feat, w, d.val.X), inD: predict(feat, w, d.inD.X), outD: predict(feat, w, d.outD.X), te: predict(feat, w, d.te.X) };
}
function computeStats() {
  const M = Math.min(+$("#eM").value, ens.members.length), mem = ens.members.slice(0, M), d = ens.data;
  if (!M) { ens.stats = null; return; }
  const g = key => moments(mem.map(m => m[key]));
  const xs = g("xs"), val = g("val"), inD = g("inD"), outD = g("outD"), te = g("te");
  const sigA2 = val.mu.reduce((s, m, k) => s + (d.val.Y[k] - m) ** 2, 0) / val.mu.length;
  const avg = sel => { let s = 0, n = 0; XS.forEach((x, k) => { if (sel(x)) { s += xs.v[k]; n++; } }); return Math.sqrt(s / n); };
  const z = (set, m) => Array.from(m.mu, (mu, k) => Math.abs(set.Y[k] - mu) / Math.sqrt(m.v[k] + sigA2));
  const zIn = z(d.inD, inD), zOut = z(d.outD, outD);
  const nll = (mu, v, Y) => { let s = 0; for (let k = 0; k < Y.length; k++) s += 0.5 * Math.log(2 * Math.PI * v[k]) + 0.5 * (Y[k] - mu[k]) ** 2 / v[k]; return s / Y.length; };
  ens.stats = {
    M, xs, sigA2, zIn, zOut,
    epiIn: avg(x => x > -d.a && x < d.a), epiOut: avg(x => Math.abs(x) > d.a + 1),
    covIn: zIn.filter(v => v <= 1.96).length / zIn.length, covOut: zOut.filter(v => v <= 1.96).length / zOut.length,
    nllSingle: nll(mem[0].te, new Float64Array(N_EVAL).fill(sigA2), d.te.Y),
    nllEns: nll(te.mu, te.v.map(v => v + sigA2), d.te.Y),
  };
}
/* Train the missing members in slices of about 25 ms, redrawing after each slice. */
function train() {
  const gen = ++ens.gen, target = +$("#eM").value;
  const step = () => {
    if (gen !== ens.gen) return;                                       // a newer request took over
    const t0 = performance.now();
    while (ens.members.length < target && performance.now() - t0 < 25) ens.members.push(trainMember(ens.members.length));
    computeStats(); drawEns(); ensStatus(ens.members.length < target);
    if (ens.members.length < target) setTimeout(step, 0);
    else { drawDec(); drawCal(); }
  };
  step();
}
function drawEns() {
  const st = ens.stats, d = ens.data, cv = $("#ensC");
  const fr = frame(cv, narrowOf(cv) ? 1.25 : 2.1, { x: [-5, 5], y: [-2.5, 2.5], xlabel: "x", yfmt: v => fmt(v, 0) });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--c4", 0.07); ctx.fillRect(X(-d.a), fr.pad.t, X(d.a) - X(-d.a), fr.ih);
  ctx.strokeStyle = css("--c4"); ctx.lineWidth = 1.2; ctx.setLineDash([4, 4]);
  [-d.a, d.a].forEach(a => seg(ctx, X(a), fr.pad.t, X(a), Y(-2.5))); ctx.setLineDash([]);
  if (st) {
    const band = (k, color) => {
      const up = XS.map((_, i) => st.xs.mu[i] + 2 * Math.sqrt(st.xs.v[i] + k)), lo = XS.map((_, i) => st.xs.mu[i] - 2 * Math.sqrt(st.xs.v[i] + k));
      ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
      ctx.beginPath(); XS.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(up[i])) : ctx.moveTo(X(x), Y(up[i]))));
      for (let i = XS.length - 1; i >= 0; i--) ctx.lineTo(X(XS[i]), Y(lo[i]));
      ctx.closePath(); ctx.fillStyle = color; ctx.fill(); ctx.restore();
    };
    band(st.sigA2, alpha("--warm", 0.2));
    band(0, alpha("--c5", 0.32));
    if ($("#eShow").checked) ens.members.slice(0, st.M).forEach(m => plotLine(fr, XS, m.xs, { color: alpha("--c5", 0.22), width: 1 }));
    plotLine(fr, XS, st.xs.mu, { color: css("--c5"), width: 2.2 });
  }
  plotLine(fr, XS, XS.map(f), { color: css("--ink"), width: 1.3, dash: [5, 4] });
  ctx.fillStyle = alpha("--accent", 0.45);
  d.tr.X.forEach((x, k) => { ctx.beginPath(); ctx.arc(X(x), Y(d.tr.Y[k]), 1.6, 0, 2 * Math.PI); ctx.fill(); });
  // readouts
  $("#rAl").textContent = st ? fmt(Math.sqrt(st.sigA2), 3) : "–";
  $("#rEin").textContent = st && st.M > 1 ? fmt(st.epiIn, 3) : "–";
  $("#rEout").textContent = st && st.M > 1 ? fmt(st.epiOut, 2) : "–";
  $("#rCov").textContent = st ? fmt(st.covIn, 2) : "–";
}
function ensStatus(training) {
  const st = ens.stats, d = ens.data, el = $("#ensStatus");
  if (training || !st) { el.textContent = `Training member ${ens.members.length + 1} of ${$("#eM").value}…`; return; }
  if (st.M === 1) { caption(el, "With a single model there is nothing to disagree with: the epistemic band vanishes, and the model is equally sure of itself everywhere, including far outside the data, where it is wildly wrong."); return; }
  caption(el,
    `Inside the data (green), the ${st.M} members agree to within <b>${fmt(st.epiIn, 3)}</b>, far less than the label noise (estimated at ${fmt(Math.sqrt(st.sigA2), 3)}, true ${fmt(d.noise, 2)}). ` +
    `Beyond [[|x| = ${fmt(d.a + 1, 1)}]] they fan out, with a spread of <b>${fmt(st.epiOut, 2)}</b>: the purple band is the ensemble saying "I don't know". ` +
    `The light band adds the irreducible noise. In the data, the 95% intervals cover ${fmt(100 * st.covIn, 0)}% of fresh points.`);
}
$("#eM").addEventListener("input", () => { $("#eMV").textContent = $("#eM").value; train(); });
["#eA", "#eN"].forEach(id => $(id).addEventListener("input", () => { labels(); makeData(); train(); }));
$("#eNew").addEventListener("click", () => { ens.seed++; makeData(); train(); });
$("#eShow").addEventListener("change", drawEns);
function labels() {
  $("#eMV").textContent = $("#eM").value;
  $("#eAV").textContent = `${fmt(+$("#eA").value, 2)}`;
  $("#eNV").textContent = fmt(+$("#eN").value, 2);
}

/* =====================================================================
   PART 3: THE TWO PARTS OF THE VARIANCE AT ONE INPUT
   ===================================================================== */
const fmtVar = v => (v < 1e-3 ? v.toExponential(1) : fmt(v, v < 0.1 ? 4 : 2));   // each variance at its own scale
function drawDec() {
  const st = ens.stats; if (!st) return;
  const x = +$("#dX").value, k = Math.round((x + 5) / 10 * 399), M = st.M;
  $("#dXV").textContent = `${fmt(x, 2)}${Math.abs(x) < ens.data.a ? " (inside the data)" : " (outside the data)"}`;
  const ys = ens.members.slice(0, M).map(m => m.xs[k]), mu = st.xs.mu[k], ve = st.xs.v[k], va = st.sigA2;
  const sd = Math.sqrt(ve + va), lo = Math.min(...ys, mu - 2.2 * sd, f(XS[k])), hi = Math.max(...ys, mu + 2.2 * sd, f(XS[k]));
  const pad = 0.08 * (hi - lo || 1), aspect = narrowOf($("#decPts")) ? 1.25 : 1.15;
  const fr = frame($("#decPts"), aspect, { x: [0, M + 1], y: [lo - pad, hi + pad], xticks: false, yfmt: v => fmt(v, Math.abs(hi - lo) < 1 ? 2 : 1) });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--warm", 0.18); ctx.fillRect(X(0), Y(mu + 2 * sd), X(M + 1) - X(0), Y(mu - 2 * sd) - Y(mu + 2 * sd));
  ctx.fillStyle = alpha("--c5", 0.3); ctx.fillRect(X(0), Y(mu + 2 * Math.sqrt(ve)), X(M + 1) - X(0), Y(mu - 2 * Math.sqrt(ve)) - Y(mu + 2 * Math.sqrt(ve)));
  ctx.strokeStyle = css("--c5"); ctx.lineWidth = 2; seg(ctx, X(0), Y(mu), X(M + 1), Y(mu));
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.3; ctx.setLineDash([5, 4]); seg(ctx, X(0), Y(f(XS[k])), X(M + 1), Y(f(XS[k]))); ctx.setLineDash([]);
  ctx.fillStyle = css("--c5"); ys.forEach((y, i) => { ctx.beginPath(); ctx.arc(X(i + 1), Y(y), 3.2, 0, 2 * Math.PI); ctx.fill(); });
  // the split, as one stacked bar of 100%
  const cv = $("#decBar"), gr = frame(cv, aspect, { x: [0, 1], y: [0, 1], xticks: false, yticks: [0, 0.5, 1], yfmt: v => `${Math.round(v * 100)}%` });
  const share = va / (va + ve), bx0 = gr.X(0.3), bx1 = gr.X(0.7);
  gr.ctx.fillStyle = alpha("--warm", 0.75); gr.ctx.fillRect(bx0, gr.Y(share), bx1 - bx0, gr.Y(0) - gr.Y(share));
  gr.ctx.fillStyle = alpha("--c5", 0.75); gr.ctx.fillRect(bx0, gr.Y(1), bx1 - bx0, gr.Y(share) - gr.Y(1));
  gr.ctx.font = "600 11px " + css("--font-ui"); gr.ctx.textAlign = "left";
  gr.ctx.fillStyle = css("--warm"); gr.ctx.fillText("aleatoric", gr.X(0.72), gr.Y(share / 2) + 4);
  gr.ctx.fillStyle = css("--c5"); gr.ctx.fillText("epistemic", gr.X(0.72), gr.Y((1 + share) / 2) + 4);
  caption($("#decCap"),
    `At [[x = ${fmt(x, 2)}]], the ${M} members predict between ${fmt(Math.min(...ys), 2)} and ${fmt(Math.max(...ys), 2)}; the truth is ${fmt(f(XS[k]), 2)}. ` +
    `Variance: aleatoric <b>${fmtVar(va)}</b> + epistemic <b>${fmtVar(ve)}</b>. ` +
    (share > 0.9 ? "Here the noise dominates: the members know this region." : share < 0.1 ? "Here ignorance dominates: the members have never seen data like this." : "Here both matter."));
}
$("#dX").addEventListener("input", drawDec);

/* =====================================================================
   PART 4: CALIBRATION AND THE NEGATIVE LOG-LIKELIHOOD
   ===================================================================== */
const TS = Array.from({ length: 40 }, (_, i) => 0.05 + 2.95 * i / 39);
function drawCal() {
  const st = ens.stats; if (!st) return;
  const nominal = TS.map(t => 2 * Phi(t) - 1), emp = zs => TS.map(t => zs.filter(z => z <= t).length / zs.length);
  const aspect = narrowOf($("#calC")) ? 1.1 : 1.05;
  const fr = frame($("#calC"), aspect, { x: [0, 1], y: [0, 1], xlabel: "nominal coverage", yfmt: v => fmt(v, 1), xfmt: v => fmt(v, 1) });
  plotLine(fr, [0, 1], [0, 1], { color: css("--ink"), width: 1.2, dash: [5, 4] });
  plotLine(fr, nominal, emp(st.zIn), { color: css("--c4"), width: 2.2 });
  plotLine(fr, nominal, emp(st.zOut), { color: css("--geom"), width: 2.2 });
  // the two negative log-likelihoods, as horizontal bars from zero
  const vals = [["one model", st.nllSingle], ["ensemble", st.nllEns]];
  const lo = Math.min(0, ...vals.map(v => v[1])) - 0.5, hi = Math.max(0, ...vals.map(v => v[1])) * 1.15 + 0.5;
  const gr = frame($("#nllC"), aspect, { x: [lo, hi], y: [0, 2], yticks: false, xlabel: "NLL", xfmt: v => fmt(v, 0) });
  gr.ctx.strokeStyle = alpha("--ink", 0.5); gr.ctx.lineWidth = 1; seg(gr.ctx, gr.X(0), gr.pad.t, gr.X(0), gr.Y(0));
  vals.forEach(([name, v], i) => {
    const y0 = gr.Y(1.7 - i), y1 = gr.Y(1.25 - i);
    gr.ctx.fillStyle = i ? alpha("--c5", 0.75) : alpha("--geom", 0.75);
    gr.ctx.fillRect(Math.min(gr.X(0), gr.X(v)), y0, Math.abs(gr.X(v) - gr.X(0)), y1 - y0);
    gr.ctx.font = "600 11px " + css("--font-ui"); gr.ctx.fillStyle = css("--ink"); gr.ctx.textAlign = "left";
    gr.ctx.fillText(`${name}: ${fmt(v, 2)}`, gr.X(Math.max(0, v)) + 6 > gr.w - 90 ? gr.X(0) + 6 : gr.X(Math.max(0, v)) + 6, y1 + 14);
  });
  caption($("#calCap"),
    `The 95% intervals cover <b>${fmt(st.covIn, 2)}</b> of fresh points inside the data (green, close to the diagonal) and <b>${fmt(st.covOut, 2)}</b> far from it (red, above the diagonal: too wide, conservative). ` +
    `On test points from [[-${+(ens.data.a + 1).toFixed(2)}]] to ${+(ens.data.a + 1).toFixed(2)}, the single model, which reports only the noise, scores an NLL of <b>${fmt(st.nllSingle, 2)}</b>; the ensemble, which adds its disagreement, <b>${fmt(st.nllEns, 2)}</b>.`);
}

/* =====================================================================
   PART 5: SHARED BLIND SPOTS
   ===================================================================== */
const weak = { seed: 3, share: true, data: null };
(() => { const rng = makeRng(424242); const X = uniform(rng, -2, 2, N_TRAIN); weak.data = { X, Y: X.map(x => f(x) + rng.normal(0, 0.1)) }; })();
const MASK = XS.map(x => x > -2 && x < 2);
function drawWeak() {
  const D = +$("#wD").value;
  $("#wDV").textContent = D;
  $$("#weak [data-share]").forEach(b => b.setAttribute("aria-pressed", (b.dataset.share === "1") === weak.share ? "true" : "false"));
  const shared = drawFeatures(makeRng(weak.seed * 31 + 7), D), rows = [];
  for (let s = 0; s < 25; s++) {
    const rng = makeRng(weak.seed * 977 + 101 * s + 5);
    const feat = weak.share ? shared : drawFeatures(rng, D);
    const idx = Int32Array.from({ length: N_TRAIN }, () => Math.floor(rng.uniform() * N_TRAIN));
    rows.push(predict(feat, fitRidge(feat, weak.data.X, weak.data.Y, idx), XS));
  }
  const { mu, v } = moments(rows);
  const xs = XS.filter((_, k) => MASK[k]), err = XS.map((x, k) => Math.abs(mu[k] - f(x))).filter((_, k) => MASK[k]), band = Array.from(v).map(s => 2 * Math.sqrt(s)).filter((_, k) => MASK[k]);
  const over = err.filter((e, k) => e > band[k]).length / err.length;
  const cv = $("#weakC"), top = Math.max(0.05, ...err, ...band) * 1.1;
  const fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [-2, 2], y: [0, top], xlabel: "x (inside the training data)", yfmt: v => fmt(v, 2) });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--geom", 0.15);
  for (let k = 0; k < xs.length - 1; k++) if (err[k] > band[k]) {
    ctx.beginPath(); ctx.moveTo(X(xs[k]), Y(band[k])); ctx.lineTo(X(xs[k + 1]), Y(band[k + 1])); ctx.lineTo(X(xs[k + 1]), Y(err[k + 1])); ctx.lineTo(X(xs[k]), Y(err[k])); ctx.closePath(); ctx.fill();
  }
  plotLine(fr, xs, band, { color: css("--c5"), width: 2.2 });
  plotLine(fr, xs, err, { color: css("--geom"), width: 2.2 });
  const meanErr = err.reduce((a, b) => a + b, 0) / err.length, meanBand = band.reduce((a, b) => a + b, 0) / band.length;
  caption($("#weakCap"),
    `${D} random feature${D > 1 ? "s" : ""} per member, ${weak.share ? "the same for all 25 members" : "drawn independently for each member"}. ` +
    `The actual error exceeds the reported 2σ band over <b>${Math.round(100 * over)}%</b> of the training range (mean error ${fmt(meanErr, 3)}, mean band ${fmt(meanBand, 3)}). ` +
    (over > 0.5 ? "The members agree, so the ensemble is confident, and it is wrong: <b>confidently wrong</b>." : over > 0.1 ? "The band misses the error in places." : "The band covers the error: the members disagree where they are unsure."));
}
$$("#weak [data-share]").forEach(b => b.addEventListener("click", () => { weak.share = b.dataset.share === "1"; drawWeak(); }));
$("#wD").addEventListener("input", drawWeak);
$("#wAgain").addEventListener("click", () => { weak.seed++; drawWeak(); });

/* =====================================================================
   start-up
   ===================================================================== */
labels(); makeData(); drawWeak(); train();
onRedraw(() => { drawEns(); drawDec(); drawCal(); drawWeak(); });
})();
