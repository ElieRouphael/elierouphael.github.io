/* Chapter 04, From prior to posterior: the live demos.
   Numbers quoted in the text come from the chapter's notebook; these widgets recompute
   the same quantities in the browser on the same grid. Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, caption, frame, plotLine, alpha, reduceMotion, onRedraw, narrowOf } = window.Kit;
const { pdf, pmf, lchoose, linspace } = window.Stats;

/* ---------------- the grid, as in the notebook ---------------- */
const G = 2001;
const THETA = linspace(0, 1, G), DT = 1 / (G - 1);
const LOG_T = THETA.map(Math.log), LOG_1T = THETA.map(t => Math.log(1 - t));
const normalise = f => { const s = f.reduce((a, b) => a + b, 0) * DT; return s > 0 ? f.map(v => v / s) : null; };
/* log of theta^k (1 - theta)^(m - k) on the grid, safe at the ends */
const logKernel = (i, k, m) => (k ? k * LOG_T[i] : 0) + (m - k ? (m - k) * LOG_1T[i] : 0);

/* summaries of a density on the grid (the notebook's summarise, plus HPD in pieces) */
function summarise(post, level = 0.95) {
  const w = post.map(v => v * DT);
  const cdf = []; w.reduce((a, v, i) => (cdf[i] = a + v), 0);
  const q = p => THETA[Math.min(G - 1, cdf.findIndex(c => c >= p))];
  const mean = w.reduce((a, v, i) => a + v * THETA[i], 0);
  const sd = Math.sqrt(w.reduce((a, v, i) => a + v * (THETA[i] - mean) ** 2, 0));
  let modeIdx = 0; post.forEach((v, i) => { if (v > post[modeIdx]) modeIdx = i; });
  // HPD: take the highest-density cells until they hold `level`; may come out in pieces
  const order = post.map((v, i) => i).sort((a, b) => post[b] - post[a]);
  const inside = new Uint8Array(G);
  let acc = 0;
  for (const i of order) { inside[i] = 1; acc += w[i]; if (acc >= level) break; }
  const pieces = [];
  for (let i = 0; i < G; i++) {
    if (inside[i] && (i === 0 || !inside[i - 1])) pieces.push([THETA[i], THETA[i]]);
    if (inside[i]) pieces[pieces.length - 1][1] = THETA[i];
  }
  const above = w.reduce((a, v, i) => a + (THETA[i] > 0.5 ? v : 0), 0);
  return { mean, sd, median: q(0.5), mode: THETA[modeIdx], et: [q((1 - level) / 2), q((1 + level) / 2)], hpd: pieces, above };
}

/* =====================================================================
   DRAW YOUR PRIOR (the demo at the top)
   ===================================================================== */
const CP = 201;                                         // control points of the drawn prior
const CP_X = linspace(0, 1, CP);
const PRESETS = {                                      // the notebook's three priors
  flat: () => 1,
  informed: t => pdf.normal(t, 0.75, 0.05),
  sceptical: t => pdf.normal(t, 0.3, 0.06) + pdf.normal(t, 0.9, 0.04),
};
const st = { heights: null, preset: "flat", n: 20, y: 13, post: null, prior: null, lik: null };
function setPreset(name) {
  const h = CP_X.map(PRESETS[name]);
  const top = Math.max(...h);
  st.heights = h.map(v => v / top * 0.8);
  st.preset = name;
}
function priorOnGrid() {
  if (PRESETS[st.preset]) return THETA.map(PRESETS[st.preset]);   // exact, as in the notebook
  return THETA.map(t => {                                           // a hand-drawn prior: interpolate

    const x = t * (CP - 1), i = Math.min(CP - 2, Math.floor(x)), f = x - i;
    return st.heights[i] * (1 - f) + st.heights[i + 1] * f;
  });
}
function compute() {
  st.prior = normalise(priorOnGrid());
  const lk = THETA.map((_, i) => Math.exp(logKernel(i, st.y, st.n)));
  st.lik = normalise(lk);                                 // the likelihood, scaled to area 1
  st.post = st.prior ? normalise(st.prior.map((p, i) => p * lk[i])) : null;
}

let priorFrame = null;
function drawPrior() {
  const canvas = $("#priorC");
  priorFrame = frame(canvas, narrowOf(canvas) ? 2.2 : 3.6, { x: [0, 1], y: [0, 1.05], yticks: false });
  plotLine(priorFrame, CP_X, st.heights, { color: css("--prior"), width: 2, fill: alpha("--prior", 0.3) });
  const { ctx, w, pad } = priorFrame;
  if (!drawn) {
    ctx.font = "600 12px " + css("--font-ui"); ctx.fillStyle = css("--muted"); ctx.textAlign = "right";
    ctx.fillText("drag here to draw ↔", w - pad.r - 4, pad.t + 14);
  }
}
function drawPost() {
  const canvas = $("#postC");
  if (!st.post) {
    const fr = frame(canvas, narrowOf(canvas) ? 1.7 : 2.8, { x: [0, 1], y: [0, 1], yticks: false, xlabel: "success rate θ" });
    fr.ctx.font = "600 13px " + css("--font-ui"); fr.ctx.fillStyle = css("--warm"); fr.ctx.textAlign = "center";
    fr.ctx.fillText("Your prior is zero wherever these data are possible.", fr.w / 2, fr.h / 2);
    return;
  }
  const top = Math.max(...st.post, ...st.prior, ...st.lik) * 1.1;
  const fr = frame(canvas, narrowOf(canvas) ? 1.7 : 2.8, { x: [0, 1], y: [0, top], yticks: false, xlabel: "success rate θ" });
  plotLine(fr, THETA, st.prior, { color: css("--prior"), width: 1.6, fill: alpha("--prior", 0.25) });
  plotLine(fr, THETA, st.lik, { color: css("--lik"), width: 1.8, dash: [6, 4] });
  plotLine(fr, THETA, st.post, { color: css("--post"), width: 2.6, fill: alpha("--post", 0.18) });
}
function updateDemo() {
  compute();
  drawPrior(); drawPost();
  $("#dN").value = st.n; $("#dY").max = st.n; $("#dY").value = st.y;
  $("#dNV").textContent = st.n; $("#dYV").textContent = st.y;
  $$("[data-prior]").forEach(b => b.setAttribute("aria-pressed", b.dataset.prior === st.preset ? "true" : "false"));
  let msg;
  if (!st.post) {
    ["#rMean", "#rMode", "#rCI", "#rAbove"].forEach(s => { $(s).textContent = "–"; });
    msg = "A posterior needs the prior to allow the values the data point to. Draw some height where the likelihood is, or pick a preset.";
  } else {
    const s = summarise(st.post), sp = summarise(st.prior);
    $("#rMean").textContent = fmt(s.mean, 3);
    $("#rMode").textContent = fmt(s.mode, 3);
    $("#rCI").textContent = `(${fmt(s.et[0], 3)}, ${fmt(s.et[1], 3)})`;
    $("#rAbove").textContent = s.above > 0.9995 ? "> 0.999" : fmt(s.above, 3);
    const two = s.hpd.length > 1 ? " Your posterior has more than one peak: the data have not been able to decide between them." : "";
    if (!st.n) msg = `No throws yet, so the posterior is just your prior, centred at ${fmt(sp.mean, 2)}. Add some throws.`;
    else if (st.preset === "flat") msg = `With a flat prior, the posterior has exactly the shape of the likelihood: the blue curve lies on the dashed orange one. Its peak is at the data's [[\\frac{${st.y}}{${st.n}}]] = ${fmt(st.y / st.n, 2)}.`;
    else {
      const mle = st.y / st.n, between = (s.mean - sp.mean) * (s.mean - mle) <= 1e-9;
      msg = `Your prior is centred at ${fmt(sp.mean, 2)} and the data say [[\\frac{${st.y}}{${st.n}}]] = ${fmt(mle, 2)}. ` +
        (between ? `The posterior settles in between, at ${fmt(s.mean, 2)}` +
            (s.sd < Math.min(sp.sd, summarise(st.lik).sd) ? ", and it is narrower than either: two sources of information agree." : ".")
          : `The posterior mean is ${fmt(s.mean, 2)}, not between the two: the whole shape of the prior matters, not just its centre.`) + two;
    }
  }
  caption($("#postStatus"), msg);
  drawSum(); drawPred();
}

/* drawing on the prior canvas */
let drawing = false, drawn = false, lastCp = null;
function pointerToCp(ev) {
  const r = $("#priorC").getBoundingClientRect(), fr = priorFrame;
  const t = Math.min(1, Math.max(0, (ev.clientX - r.left - fr.pad.l) / fr.iw));
  const h = Math.min(1, Math.max(0, (fr.pad.t + fr.ih - (ev.clientY - r.top)) / fr.ih * 1.05));
  return [Math.round(t * (CP - 1)), h];
}
function paint(ev) {
  const [i, h] = pointerToCp(ev);
  if (lastCp) {
    const [i0, h0] = lastCp, lo = Math.min(i0, i), hi = Math.max(i0, i);
    for (let j = lo; j <= hi; j++) st.heights[j] = hi === lo ? h : h0 + (h - h0) * (j - i0) / (i - i0);
  } else st.heights[i] = h;
  lastCp = [i, h];
  st.preset = "own"; drawn = true;
  updateDemo();
}
$("#priorC").addEventListener("pointerdown", ev => { drawing = true; lastCp = null; $("#priorC").setPointerCapture(ev.pointerId); paint(ev); });
$("#priorC").addEventListener("pointermove", ev => { if (drawing) paint(ev); });
["pointerup", "pointercancel"].forEach(t => $("#priorC").addEventListener(t, () => { drawing = false; lastCp = null; }));

$("#dN").addEventListener("input", e => { st.n = +e.target.value; st.y = Math.min(st.y, st.n); updateDemo(); });
$("#dY").addEventListener("input", e => { st.y = +e.target.value; updateDemo(); });
$$("[data-prior]").forEach(b => b.addEventListener("click", () => { setPreset(b.dataset.prior); updateDemo(); }));
$$("[data-try]").forEach(b => b.addEventListener("click", () => {
  setPreset(b.dataset.try); st.n = 20; st.y = 13; updateDemo();
  $("#demo").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}));

/* =====================================================================
   READING THE POSTERIOR: point estimates and intervals
   ===================================================================== */
let level = 0.95;
function drawSum() {
  const canvas = $("#sumC");
  $("#sumLV").textContent = pct(level, 0);
  if (!st.post) { frame(canvas, narrowOf(canvas) ? 1.5 : 2.6, { x: [0, 1], y: [0, 1], yticks: false }); caption($("#sumCap"), "No posterior to read: see the demo at the top."); return; }
  const s = summarise(st.post, level);
  const top = Math.max(...st.post) * 1.18;
  const fr = frame(canvas, narrowOf(canvas) ? 1.5 : 2.6, { x: [0, 1], y: [0, top], yticks: false, pad: { b: 56 } });
  const { ctx, X, Y } = fr;
  plotLine(fr, THETA, st.post, { color: css("--post"), width: 2.4, fill: alpha("--post", 0.15) });
  const marks = [["mean", s.mean, [6, 4]], ["median", s.median, [2, 3]], ["mode", s.mode, null]];
  ctx.font = "600 11px " + css("--font-ui"); ctx.textAlign = "center";
  marks.forEach(([name, v, dash], k) => {
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.4; ctx.setLineDash(dash || []);
    ctx.beginPath(); ctx.moveTo(X(v), Y(0)); ctx.lineTo(X(v), fr.pad.t + 14 + 12 * k); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = css("--surface"); ctx.lineWidth = 4; ctx.lineJoin = "round";
    ctx.strokeText(name, X(v), fr.pad.t + 10 + 12 * k);
    ctx.fillStyle = css("--ink"); ctx.fillText(name, X(v), fr.pad.t + 10 + 12 * k);
  });
  ctx.lineWidth = 4;
  ctx.strokeStyle = css("--post");
  ctx.beginPath(); ctx.moveTo(X(s.et[0]), Y(0) + 32); ctx.lineTo(X(s.et[1]), Y(0) + 32); ctx.stroke();
  ctx.strokeStyle = css("--warm");
  for (const [a, b] of s.hpd) { ctx.beginPath(); ctx.moveTo(X(a), Y(0) + 46); ctx.lineTo(X(Math.max(b, a + 0.002)), Y(0) + 46); ctx.stroke(); }
  const L = pct(level, 0);
  const etW = s.et[1] - s.et[0], hpdW = s.hpd.reduce((a, [p, q]) => a + q - p, 0);
  const hpdText = s.hpd.map(([a, b]) => `(${fmt(a, 3)}, ${fmt(b, 3)})`).join(" and ");
  caption($("#sumCap"),
    `Mean ${fmt(s.mean, 3)}, median ${fmt(s.median, 3)}, mode ${fmt(s.mode, 3)}. ` +
    `The ${L} equal-tailed interval (${fmt(s.et[0], 3)}, ${fmt(s.et[1], 3)}) is ${fmt(etW, 3)} wide; ` +
    `the ${L} HPD region ${hpdText} is ${fmt(hpdW, 3)} wide${s.hpd.length > 1 ? `, in ${s.hpd.length} pieces, because the posterior has more than one peak` : ", never wider than the equal-tailed one"}.` +
    (s.hpd.length > 1 && st.post[Math.round(s.mean * (G - 1))] < 0.5 * Math.max(...st.post) ? " Notice the mean: it sits in a valley of the posterior, a best guess that is itself implausible." : ""));
}
$("#sumL").addEventListener("input", e => { level = +e.target.value / 100; drawSum(); });

/* =====================================================================
   PREDICTING NEW DATA
   ===================================================================== */
const pred = { m: 10, k: 8, prior: false };
function predictive(dens, m) {
  const w = dens.map(v => v * DT);
  return Array.from({ length: m + 1 }, (_, k) => {
    const lc = lchoose(m, k);
    let s = 0;
    for (let i = 0; i < G; i++) if (w[i]) s += w[i] * Math.exp(lc + logKernel(i, k, m));
    return s;
  });
}
const moments = ps => { const mu = ps.reduce((a, p, k) => a + k * p, 0); return [mu, Math.sqrt(ps.reduce((a, p, k) => a + (k - mu) ** 2 * p, 0))]; };
function drawPred() {
  const { m } = pred;
  $("#predK").max = m; pred.k = Math.min(pred.k, m); $("#predK").value = pred.k;
  $("#predMV").textContent = m; $("#predKV").textContent = pred.k;
  const dens = pred.prior ? st.prior : st.post;
  const canvas = $("#predC");
  if (!dens) { frame(canvas, narrowOf(canvas) ? 1.5 : 2.6, { x: [0, 1], y: [0, 1], yticks: false }); caption($("#predCap"), "No posterior: see the demo at the top."); return; }
  const centre = pred.prior ? summarise(st.prior).mean : st.n ? st.y / st.n : summarise(st.post).mean;
  const pp = predictive(dens, m);
  const plug = Array.from({ length: m + 1 }, (_, k) => pmf.binomial(k, m, centre));
  const top = Math.max(...pp, ...plug) * 1.15;
  const fr = frame(canvas, narrowOf(canvas) ? 1.5 : 2.6, { x: [-0.7, m + 0.7], y: [0, top], xticks: m <= 15 ? pp.map((_, k) => k) : undefined, yfmt: v => fmt(v, 2), xlabel: "made, out of the next " + m });
  const { ctx, X, Y } = fr;
  const bw = Math.max(1, (X(1) - X(0)) * 0.36);
  pp.forEach((p, k) => {
    ctx.fillStyle = alpha("--prior", k >= pred.k ? 1 : 0.55); ctx.fillRect(X(k) - bw - 0.5, Y(plug[k]), bw, Y(0) - Y(plug[k]));
    ctx.fillStyle = alpha("--post", k >= pred.k ? 1 : 0.45); ctx.fillRect(X(k) + 0.5, Y(p), bw, Y(0) - Y(p));
  });
  const [mp, sp] = moments(pp), [mu, su] = moments(plug);
  const tail = a => a.reduce((s, p, k) => s + (k >= pred.k ? p : 0), 0);
  caption($("#predCap"),
    `${pred.prior ? "Prior" : "Posterior"} predictive: mean ${fmt(mp, 2)}, sd ${fmt(sp, 2)}, and [[P(\\tilde y \\ge ${pred.k}) = ${fmt(tail(pp), 3)}]]. ` +
    `Plug-in Binomial(${m}, ${fmt(centre, 2)}): mean ${fmt(mu, 2)}, sd ${fmt(su, 2)}, and ${fmt(tail(plug), 3)}. ` +
    (sp > su + 0.005 ? "The predictive is wider: it also admits that θ is uncertain." : "Here the two nearly agree: θ is already pinned down well."));
}
$("#predM").addEventListener("input", e => { pred.m = +e.target.value; drawPred(); });
$("#predK").addEventListener("input", e => { pred.k = +e.target.value; drawPred(); });
$("#predPrior").addEventListener("change", e => { pred.prior = e.target.checked; drawPred(); });

/* =====================================================================
   ONE THROW AT A TIME
   ===================================================================== */
const NOTEBOOK_ORDER = "X.XX.XXXX.X.X..XXXX.".split("").map(c => c === "X");
const seq = { order: NOTEBOOK_ORDER.slice(), step: 0, timer: null };
const FINAL = normalise(THETA.map((_, i) => Math.exp(logKernel(i, 13, 20))));
function sequentialBelief(steps) {
  let b = normalise(THETA.map(() => 1));
  for (let s = 0; s < steps; s++) { const made = seq.order[s]; b = normalise(b.map((v, i) => v * (made ? THETA[i] : 1 - THETA[i]))); }
  return b;
}
function drawSeq() {
  const belief = sequentialBelief(seq.step);
  const top = Math.max(...belief, ...FINAL) * 1.12;
  const canvas = $("#seqC");
  const fr = frame(canvas, narrowOf(canvas) ? 1.6 : 2.8, { x: [0, 1], y: [0, top], yticks: false, xlabel: "success rate θ" });
  plotLine(fr, THETA, FINAL, { color: css("--lik"), width: 1.8, dash: [6, 4] });
  plotLine(fr, THETA, belief, { color: css("--post"), width: 2.6, fill: alpha("--post", 0.18) });
  $("#seqThrows").innerHTML = seq.order.map((made, i) =>
    `<span class="${i < seq.step - 1 ? "done" : i === seq.step - 1 ? "now" : ""}">${made ? "X" : "."}</span>`).join("");
  const made = seq.order.slice(0, seq.step).filter(Boolean).length;
  const mean = summarise(belief).mean;
  let cap;
  if (!seq.step) cap = "Before any throw: a flat prior. The dashed orange curve is the posterior from all 20 throws at once, for comparison.";
  else if (seq.step < seq.order.length) cap = `After ${seq.step} throw${seq.step > 1 ? "s" : ""} (${made} made, ${seq.step - made} missed), the posterior mean is ${fmt(mean, 3)}. Each new throw multiplies the current posterior by θ (a make) or 1 − θ (a miss), then normalises.`;
  else {
    const diff = belief.reduce((a, v, i) => Math.max(a, Math.abs(v - FINAL[i])), 0);
    const diffTex = diff === 0 ? "0" : diff.toExponential(1).replace("e", "\\times 10^{").replace("+", "") + "}";
    cap = `All 20 throws: the blue curve lies exactly on the dashed one. The largest difference on the grid is [[${diffTex}]]: rounding error, nothing more. Shuffle the order and play again: the end point never moves.`;
  }
  caption($("#seqCap"), cap);
  $("#seqNext").disabled = seq.step >= seq.order.length;
}
function setPlaying(on) {
  clearInterval(seq.timer); seq.timer = null;
  $("#seqPlay").setAttribute("aria-pressed", on ? "true" : "false");
  $("#seqPlay").textContent = on ? "pause" : "play";
  if (on) {
    if (seq.step >= seq.order.length) seq.step = 0;
    seq.timer = setInterval(() => {
      seq.step++; drawSeq();
      if (seq.step >= seq.order.length) setPlaying(false);
    }, reduceMotion ? 700 : 380);
  }
}
$("#seqNext").addEventListener("click", () => { setPlaying(false); seq.step = Math.min(seq.order.length, seq.step + 1); drawSeq(); });
$("#seqPlay").addEventListener("click", () => setPlaying(!seq.timer));
$("#seqShuffle").addEventListener("click", () => {
  setPlaying(false);
  const a = seq.order;
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  seq.step = 0; drawSeq();
});
$("#seqReset").addEventListener("click", () => { setPlaying(false); seq.order = NOTEBOOK_ORDER.slice(); seq.step = 0; drawSeq(); });

/* =====================================================================
   start-up
   ===================================================================== */
setPreset("flat");
updateDemo();
drawSeq();
onRedraw(() => { drawPrior(); drawPost(); drawSum(); drawPred(); drawSeq(); });
})();
