/* Which Job Should the Machine Do Next?, chapter D2, a fair benchmark: the demos.
   Hero: the 20 test days of results/runs.csv for any two methods, their daily differences, and the
   bootstrap of the mean difference, paired against unpaired. Widget 1 (Part 3): resampling the 20 days by
   hand. Widget 2 (Part 5): the winner's curse in the equal-means model. Needs ../kit/kit.js and
   jobshop-lib.js. code/d02_numbers.py prints the numbers of the page, including this file's default
   intervals (its mulberry32 port reproduces bootMeans and bootUnpaired exactly). */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, alpha, onRedraw, narrowOf, histogram } = window.Kit;
const { rng } = window.JS;

/* ---------------- the data: total tardiness (minutes, 60 jobs) per test day, seeds 0..19, from
   results/runs.csv of the repository; mean tardiness per order = total / 60. */
const METHODS = ["FIFO", "SPT", "EDD", "SST", "DQN", "Greedy replan", "CP-SAT"];
const NAMES = { FIFO: "First come, first served (FIFO)", SPT: "Shortest job first (SPT)", EDD: "Most urgent first (EDD)",
  SST: "Shortest setup first (SST)", DQN: "Self-learning agent (DQN)", "Greedy replan": "Quick re-planning (Greedy replan)",
  "CP-SAT": "Optimiser re-planning (CP-SAT)" };
const LEVELS = ["no breakdowns", "frequent breakdowns"];
const TOT = {
  "no breakdowns": {
    FIFO: [408, 1190, 221, 554, 154, 118, 1050, 5846, 1961, 468, 1289, 280, 1415, 149, 340, 153, 206, 2490, 232, 126],
    SPT: [427, 689, 260, 347, 68, 65, 658, 3348, 1586, 388, 908, 162, 930, 124, 89, 204, 237, 1158, 199, 140],
    EDD: [177, 696, 125, 158, 111, 18, 312, 2836, 1139, 139, 753, 112, 589, 71, 107, 60, 289, 558, 96, 72],
    SST: [368, 556, 132, 362, 285, 54, 638, 3101, 1264, 581, 646, 187, 656, 207, 185, 217, 153, 1417, 148, 65],
    DQN: [226, 506, 135, 279, 86, 41, 393, 2503, 913, 201, 900, 149, 783, 196, 134, 67, 246, 947, 203, 27],
    "Greedy replan": [36, 108, 42, 307, 5, 2, 240, 983, 606, 49, 158, 19, 189, 43, 27, 0, 7, 795, 19, 51],
    "CP-SAT": [2, 45, 1, 0, 1, 0, 31, 742, 298, 25, 37, 1, 50, 2, 17, 1, 0, 347, 0, 0],
  },
  "frequent breakdowns": {
    FIFO: [1970, 4283, 2400, 3399, 1321, 859, 9210, 10196, 5967, 4730, 2058, 1930, 5373, 1926, 1917, 1087, 2866, 8664, 1191, 1656],
    SPT: [2238, 2488, 2910, 2097, 943, 864, 6256, 7645, 4149, 2495, 2317, 1332, 4270, 1299, 1571, 674, 1667, 6892, 1240, 1177],
    EDD: [1532, 2647, 1875, 2213, 950, 677, 4281, 6449, 2689, 2337, 1482, 980, 3922, 1055, 1558, 703, 2328, 5494, 793, 1192],
    SST: [1074, 2724, 1429, 2338, 752, 608, 4488, 5942, 3628, 2013, 1117, 1423, 2695, 1337, 1605, 793, 1282, 3844, 862, 1025],
    DQN: [1750, 2675, 1887, 1906, 906, 516, 4668, 5438, 3279, 1981, 1151, 974, 3664, 1097, 1623, 506, 1756, 5308, 603, 818],
    "Greedy replan": [467, 1731, 1856, 1379, 314, 183, 3378, 3599, 2346, 2014, 604, 667, 1332, 695, 1202, 256, 663, 4846, 678, 625],
    "CP-SAT": [559, 1258, 829, 1350, 112, 95, 2713, 3818, 2252, 1753, 474, 633, 1923, 356, 1021, 219, 662, 2714, 211, 237],
  },
};
const MT = (level, m) => TOT[level][m].map(t => t / 60);

/* ---------------- statistics ---------------- */
const mean = xs => { let s = 0; for (const v of xs) s += v; return s / xs.length; };
function sd(xs) { const m = mean(xs); let s = 0; for (const v of xs) s += (v - m) * (v - m); return Math.sqrt(s / (xs.length - 1)); }
function corr(a, b) {
  const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : NaN;
}
/* The resampling of jobshop-lib.js:bootstrapMean, keeping the sorted means for the histogram. */
function bootMeans(values, B = 10000, seed = 0) {
  const r = rng(seed + 12345), n = values.length, out = new Float64Array(B);
  for (let b = 0; b < B; b++) { let s = 0; for (let i = 0; i < n; i++) s += values[Math.floor(r() * n)]; out[b] = s / n; }
  return out.sort();
}
/* Unpaired: each method's days resampled separately, then the difference of the two means. */
function bootUnpaired(a, b, B = 10000, seed = 0) {
  const r = rng(seed + 54321), n = a.length, out = new Float64Array(B);
  for (let k = 0; k < B; k++) {
    let sa = 0; for (let i = 0; i < n; i++) sa += a[Math.floor(r() * n)];
    let sb = 0; for (let i = 0; i < n; i++) sb += b[Math.floor(r() * n)];
    out[k] = sa / n - sb / n;
  }
  return out.sort();
}
const quant = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))))];
const sgn = v => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v), 1);
const sgn2 = v => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v), 2);
const iv = (lo, hi) => `[${sgn(lo)}, ${sgn(hi)}]`;

/* ---------------- the hero ---------------- */
const st = { level: 1, a: "CP-SAT", b: "SST" };
function fillSelect(sel, value) {
  sel.innerHTML = METHODS.map(m => `<option value="${m}"${m === value ? " selected" : ""}>${NAMES[m]}</option>`).join("");
}
fillSelect($("#hA"), st.a); fillSelect($("#hB"), st.b);

function drawHero() {
  const level = LEVELS[st.level], n = +$("#hN").value;
  const A = MT(level, st.a).slice(0, n), Bv = MT(level, st.b).slice(0, n), D = A.map((v, i) => v - Bv[i]);
  const cA = css("--accent"), cB = css("--warm"), ink = css("--ink"), muted = css("--muted");
  const same = st.a === st.b;

  /* 1: the two methods day by day */
  {
    const cv = $("#dayC"), narrow = narrowOf(cv), top = Math.max(1, ...A, ...Bv) * 1.08;
    const fr = frame(cv, narrow ? 1.5 : 3.0, { x: [-0.6, n - 0.4], y: [0, top], xlabel: "test day (seed)", ylabel: "mean tardiness (min)",
      xticks: Array.from({ length: n }, (_, i) => i).filter(i => n <= 10 || !narrow || i % 2 === 0) });
    const { ctx, X, Y } = fr;
    ctx.strokeStyle = alpha("--muted", 0.6); ctx.lineWidth = 1.5;
    for (let i = 0; i < n; i++) { ctx.beginPath(); ctx.moveTo(X(i), Y(A[i])); ctx.lineTo(X(i), Y(Bv[i])); ctx.stroke(); }
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = cB; ctx.beginPath(); ctx.arc(X(i), Y(Bv[i]), 4.2, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = cA; ctx.beginPath(); ctx.arc(X(i), Y(A[i]), 4.2, 0, 2 * Math.PI); ctx.fill();
    }
  }
  $("#dayLegend").innerHTML = `<span><span class="sw" style="background:${cA}"></span>X: ${NAMES[st.a]}</span><span><span class="sw" style="background:${cB}"></span>Y: ${NAMES[st.b]}</span>`;

  /* 2: the daily differences */
  {
    const cv = $("#diffC"), narrow = narrowOf(cv);
    const lo = Math.min(-1, ...D) * 1.1, hi = Math.max(1, ...D) * 1.1;
    const fr = frame(cv, narrow ? 2.0 : 4.2, { x: [-0.6, n - 0.4], y: [lo, hi], xlabel: "test day", ylabel: "X − Y (min)",
      xticks: Array.from({ length: n }, (_, i) => i).filter(i => n <= 10 || !narrow || i % 2 === 0) });
    const { ctx, X, Y } = fr, bw = Math.max(2, 0.6 * (X(1) - X(0)));
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = D[i] < 0 ? cA : D[i] > 0 ? cB : muted;
      const y0 = Y(0), y1 = Y(D[i]);
      ctx.fillRect(X(i) - bw / 2, Math.min(y0, y1), bw, Math.max(1, Math.abs(y1 - y0)));
    }
    ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(fr.pad.l, Y(0)); ctx.lineTo(fr.w - fr.pad.r, Y(0)); ctx.stroke();
    ctx.setLineDash([5, 4]); ctx.strokeStyle = ink; ctx.beginPath(); ctx.moveTo(fr.pad.l, Y(mean(D))); ctx.lineTo(fr.w - fr.pad.r, Y(mean(D))); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = ink; ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "right";
    ctx.fillText(`mean ${sgn(mean(D))}`, fr.w - fr.pad.r - 2, Y(mean(D)) + (mean(D) < 0 ? 13 : -5));
  }

  /* 3: the bootstrap distributions */
  const P = bootMeans(D), U = bootUnpaired(A, Bv);
  const pLo = quant(P, 0.025), pHi = quant(P, 0.975), uLo = quant(U, 0.025), uHi = quant(U, 0.975);
  {
    const cv = $("#bootC"), narrow = narrowOf(cv);
    let lo = Math.min(quant(U, 0.002), quant(P, 0.002), 0), hi = Math.max(quant(U, 0.998), quant(P, 0.998), 0);
    const pad0 = 0.05 * (hi - lo || 1); lo -= pad0; hi += pad0;
    const bins = 70, w = (hi - lo) / bins, scale = 1 / (P.length * w);
    const peak = c => { const h = new Float64Array(bins); for (const v of c) { const i = Math.floor((v - lo) / w); if (i >= 0 && i < bins) h[i]++; } return Math.max(...h) * scale; };
    const ytop = Math.max(peak(P), peak(U), 1e-6) * 1.45;
    const fr = frame(cv, narrow ? 1.5 : 2.8, { x: [lo, hi], y: [0, ytop], yticks: false, xlabel: "mean of X − Y over the days (min)" });
    const { ctx, X, Y } = fr;
    const dark = window.Kit.isDark();     /* the warm wash is too faint on a dark surface */
    const uCounts = histogram(fr, U, lo, hi, bins, scale, alpha("--warm", dark ? 0.7 : 0.45));
    if (dark) {                            /* and a 1 px outline in --warm keeps each bar visible */
      ctx.save(); ctx.strokeStyle = cB; ctx.lineWidth = 1;
      uCounts.forEach((c, i) => { if (!c) return; const x0 = X(lo + i * w), x1 = X(lo + (i + 1) * w), h = c * scale; ctx.strokeRect(x0 + 0.8, Y(h) + 0.5, Math.max(0.8, x1 - x0 - 1.6), Y(0) - Y(h) - 0.5); });
      ctx.restore();
    }
    histogram(fr, P, lo, hi, bins, scale, alpha("--accent", 0.6));
    ctx.strokeStyle = muted; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(0), fr.pad.t); ctx.lineTo(X(0), Y(0)); ctx.stroke(); ctx.setLineDash([]);
    const bar = (l, h, y, c, label) => {
      ctx.strokeStyle = c; ctx.fillStyle = c; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(X(l), y); ctx.lineTo(X(h), y); ctx.stroke();
      ctx.lineWidth = 1.5; [l, h].forEach(v => { ctx.beginPath(); ctx.moveTo(X(v), y - 5); ctx.lineTo(X(v), y + 5); ctx.stroke(); });
      ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(label, Math.min(fr.w - fr.pad.r - 60, Math.max(fr.pad.l + 60, X((l + h) / 2))), y - 6); ctx.textBaseline = "alphabetic";
    };
    bar(uLo, uHi, fr.pad.t + 16, cB, `unpaired ${iv(uLo, uHi)}`);
    bar(pLo, pHi, fr.pad.t + 40, cA, `paired ${iv(pLo, pHi)}`);
  }

  /* readouts and caption */
  const r = corr(A, Bv), sa = sd(A), sb = sd(Bv), sdD = sd(D), nb = D.filter(v => v < 0).length, ne = D.filter(v => v === 0).length, nw = n - nb - ne;
  const cov = r * sa * sb;
  $("#rA").textContent = fmt(mean(A), 1) + " min"; $("#rB").textContent = fmt(mean(Bv), 1) + " min";
  $("#rD").textContent = sgn(mean(D)) + " min"; $("#rR").textContent = isFinite(r) ? fmt(r, 3) : "–";
  $("#rP").textContent = iv(pLo, pHi); $("#rU").textContent = iv(uLo, uHi);
  $("#rW").textContent = pHi - pLo > 1e-9 ? fmt((uHi - uLo) / (pHi - pLo), 1) + "×" : "–";
  $("#rC").textContent = `${nb} / ${ne} / ${nw}`;
  $("#hNV").textContent = n;
  $$("#hLevel .btn").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.l === st.level)));
  const verdict = same ? "The two methods are the same: every daily difference is zero, and so is the paired interval; the unpaired interval still has a width, because resampling the two copies separately breaks their pairing."
    : pHi < 0 ? `The paired interval lies below zero: on these ${n} days, <b>${st.a}</b> is less late than ${st.b}.`
    : pLo > 0 ? `The paired interval lies above zero: on these ${n} days, <b>${st.b}</b> is less late than ${st.a}.`
    : `The paired interval contains zero: these ${n} days do not show which method is less late.`;
  caption($("#heroStatus"), `<b>${st.a} − ${st.b}</b>, ${level}, first ${n} test days: mean difference ${sgn(mean(D))} min, better on ${nb} days, equal on ${ne}, worse on ${nw}. ` +
    (same ? "" : `The days move together, [[r = ${fmt(r, 3)}]], so [[s_X^2 + s_Y^2 - 2\\,\\widehat{\\mathrm{Cov}} = ${fmt(sa * sa, 1)} + ${fmt(sb * sb, 1)} - ${fmt(2 * cov, 1)} = ${fmt(sdD * sdD, 1)}]] for the daily difference, against ${fmt(sa * sa + sb * sb, 1)} for unrelated days; ` +
    `the paired interval is ${fmt(pHi - pLo, 1)} minutes wide, the unpaired one ${fmt(uHi - uLo, 1)}. `) + verdict);
}
$$("#hLevel .btn").forEach(b => b.addEventListener("click", () => { st.level = +b.dataset.l; drawHero(); }));
$("#hA").addEventListener("change", e => { st.a = e.target.value; drawHero(); });
$("#hB").addEventListener("change", e => { st.b = e.target.value; drawHero(); });
$("#hN").addEventListener("input", drawHero);

/* ---------------- widget 1: resampling by hand (CP-SAT − SST, frequent breakdowns) ---------------- */
const RS = (() => { const a = MT("frequent breakdowns", "CP-SAT"), b = MT("frequent breakdowns", "SST"); return a.map((v, i) => v - b[i]); })();
const rs = { r: rng(2026), means: [], last: null };
$("#rsChips").innerHTML = RS.map((d, i) => `<span data-i="${i}">d${i}: ${sgn(d)}</span>`).join("");
function drawOne() {
  const n = RS.length, pick = Array.from({ length: n }, () => Math.floor(rs.r() * n));
  rs.last = pick; rs.means.push(pick.reduce((s, i) => s + RS[i], 0) / n);
}
function drawRS() {
  const n = RS.length, counts = new Array(n).fill(0);
  if (rs.last) rs.last.forEach(i => counts[i]++);
  $$("#rsChips span").forEach((el, i) => {
    el.className = counts[i] >= 2 ? "on2" : counts[i] === 1 ? "on" : "";
    el.textContent = `d${i}: ${sgn(RS[i])}` + (counts[i] >= 2 ? ` ×${counts[i]}` : "");
  });
  const cv = $("#rsC"), narrow = narrowOf(cv), lo = -24, hi = -6, bins = 54, w = (hi - lo) / bins;
  const m = rs.means.length, scale = m ? 1 / (m * w) : 1;
  const fr = frame(cv, narrow ? 1.8 : 3.4, { x: [lo, hi], y: [0, 0.3], yticks: false, xlabel: "resample mean of the 20 differences (min)" });
  const { ctx, X, Y } = fr;
  if (m) histogram(fr, rs.means, lo, hi, bins, scale, alpha("--accent", 0.6));
  const m0 = mean(RS);
  ctx.strokeStyle = css("--ink"); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(X(m0), fr.pad.t); ctx.lineTo(X(m0), Y(0)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = css("--ink"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText(`sample mean ${sgn(m0)}`, X(m0) + 4, fr.pad.t + 10);
  const plug = Math.sqrt(RS.reduce((s, v) => s + (v - m0) ** 2, 0) / n) / Math.sqrt(n);
  let text;
  if (!m) text = `Press <b>Draw one resample</b>: 20 days are picked at random, with replacement, from the 20 above; some appear twice, some not at all. The mean of their differences is one bootstrap replicate. The exact bootstrap standard error is [[\\hat\\sigma/\\sqrt{n} = ${fmt(plug, 2)}]] minutes.`;
  else {
    const sorted = Float64Array.from(rs.means).sort(), seen = new Set(rs.last).size;
    const lastMean = rs.means[m - 1];
    text = m === 1
      ? `Last resample: ${seen} distinct days of 20 (the others are left out, and the orange ones count twice or more), mean ${sgn(lastMean)} min: one bootstrap replicate of the sample mean ${sgn(m0)}. Draw more to see how much the replicates spread; the exact bootstrap standard error is [[\\hat\\sigma/\\sqrt{n} = ${fmt(plug, 2)}]] minutes.`
      : `Last resample: ${seen} distinct days of 20, mean ${sgn(lastMean)} min. After <b>${m}</b> resamples the replicates have standard deviation ${fmt(sd(rs.means), 2)} min, against the exact bootstrap value [[\\hat\\sigma/\\sqrt{n} = ${fmt(plug, 2)}]]` +
      (m >= 1000 ? `; their 2.5&nbsp;% and 97.5&nbsp;% quantiles give the interval ${iv(quant(sorted, 0.025), quant(sorted, 0.975))}, the percentile bootstrap with [[B = ${m}]].` : ". Draw 1,000 to read an interval.");
  }
  caption($("#rsCap"), text);
}
$("#rsOne").addEventListener("click", () => { drawOne(); drawRS(); });
$("#rsMany").addEventListener("click", () => { for (let k = 0; k < 1000; k++) drawOne(); drawRS(); });
$("#rsReset").addEventListener("click", () => { rs.r = rng(2026); rs.means = []; rs.last = null; drawRS(); });

/* ---------------- widget 2: the winner's curse ---------------- */
/* Phi by the complementary error function of Numerical Recipes (erfcc, fractional error below 1.2e-7). */
function erfc(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const Phi = x => 0.5 * erfc(-x / Math.SQRT2);
const EK = {};
function eK(K) {          /* E[max of K standard normals] = int w K phi(w) Phi(w)^(K-1) dw, trapezoid on [-9, 9] */
  if (EK[K] != null) return EK[K];
  const N = 6000, a = -9, h = 18 / N; let s = 0;
  for (let i = 0; i <= N; i++) { const w = a + i * h, f = w * K * Math.exp(-w * w / 2) / Math.sqrt(2 * Math.PI) * Math.pow(Phi(w), K - 1); s += (i === 0 || i === N ? 0.5 : 1) * f; }
  return (EK[K] = s * h);
}
const SIGMA = sd(MT("frequent breakdowns", "SST"));      // SST's spread over the test days under breakdowns, 24.0 min
const wc = { seed: 7 };
function normals(r) { return () => { const u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }; }
function drawWC() {
  const K = +$("#wcK").value, rho = +$("#wcR").value, n = +$("#wcN").value, R = 4000;
  $("#wcKV").textContent = K; $("#wcRV").textContent = fmt(rho, 2); $("#wcNV").textContent = n;
  const z = normals(rng(1000 + wc.seed)), a = Math.sqrt(rho), b = Math.sqrt(1 - rho);
  const won = new Float64Array(R), fresh = new Float64Array(R), sums = new Float64Array(K);
  for (let e = 0; e < R; e++) {
    sums.fill(0);
    for (let i = 0; i < n; i++) { const day = a * z(); for (let k = 0; k < K; k++) sums[k] += SIGMA * (day + b * z()); }
    let best = 0; for (let k = 1; k < K; k++) if (sums[k] < sums[best]) best = k;
    won[e] = sums[best] / n;
    let f = 0; for (let i = 0; i < n; i++) f += SIGMA * (a * z() + b * z());     // the winner on n fresh days
    fresh[e] = f / n;
  }
  const theory = -SIGMA * Math.sqrt(1 - rho) * eK(K) / Math.sqrt(n), spread = SIGMA / Math.sqrt(n);
  const lo = -3.6 * spread + Math.min(0, theory), hi = 3.6 * spread, bins = 60, w = (hi - lo) / bins, scale = 1 / (R * w);
  const cv = $("#wcC"), narrow = narrowOf(cv);
  const ytop = 1.6 / (Math.sqrt(2 * Math.PI) * spread * Math.max(0.2, Math.min(1, Math.sqrt(1 - rho) + 0.25)));
  const fr = frame(cv, narrow ? 1.6 : 3.0, { x: [lo, hi], y: [0, ytop], yticks: false, xlabel: "reported mean minus the true mean (min)" });
  const { ctx, X, Y } = fr;
  histogram(fr, fresh, lo, hi, bins, scale, alpha("--accent", 0.45));
  histogram(fr, won, lo, hi, bins, scale, alpha("--warm", 0.6));
  /* right = true puts the label to the right of both this line and the truth line, clear of the curse markers */
  const vline = (x, c, dash, label, dy, right = false) => {
    ctx.strokeStyle = c; ctx.setLineDash(dash); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(x), fr.pad.t); ctx.lineTo(X(x), Y(0)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = c; ctx.font = "11px " + css("--font-mono");
    if (right) { ctx.textAlign = "left"; ctx.fillText(label, X(Math.max(x, 0)) + 4, fr.pad.t + dy); }
    else { ctx.textAlign = x < 0 ? "right" : "left"; ctx.fillText(label, X(x) + (x < 0 ? -4 : 4), fr.pad.t + dy); }
  };
  const mw = mean(won), mf = mean(fresh);
  vline(0, css("--muted"), [3, 3], "truth 0", 10);
  vline(theory, css("--ink"), [6, 4], `theory ${sgn2(theory)}`, 24);
  /* the two histograms' own means, so the shift between them is visible even when it is small */
  vline(mw, css("--warm"), [2, 2], `winner on its days ${sgn2(mw)}`, 38);
  vline(mf, css("--accent"), [2, 2], `on fresh days ${sgn2(mf)}`, 24, true);
  caption($("#wcCap"), `<span style="color:${css("--warm")}">■</span> the winner's score on the days it was chosen on, <span style="color:${css("--accent")}">■</span> the same winner on [[n = ${n}]] fresh days; 4,000 experiments, [[\\sigma = ${fmt(SIGMA, 1)}]] min. ` +
    `${K} candidates with the same true mean: the chosen one looks ${fmt(-mw, 2)} min better than it is on average (theory [[\\sigma\\sqrt{1-\\rho}\\, e_{${K}}/\\sqrt{n} = ${fmt(-theory, 2)}]], with [[e_{${K}} = ${fmt(eK(K), 3)}]]); on fresh days the error averages ${sgn(mf)}. ` +
    (rho >= 0.85 ? "Common days keep the curse small: only the method-specific part of the luck can pick the winner." : "Lower correlation, more method-specific luck, a larger curse."));
}
["#wcK", "#wcR", "#wcN"].forEach(id => $(id).addEventListener("input", drawWC));
$("#wcAgain").addEventListener("click", () => { wc.seed++; drawWC(); });

function drawAll() { drawHero(); drawRS(); drawWC(); }
drawOne();             // open with one resample on show
drawAll();
onRedraw(drawAll);
})();
