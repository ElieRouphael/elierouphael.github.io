(() => {
"use strict";
const D = window.DQN_DATA;
const $ = (sel, root = document) => root.querySelector(sel);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const byName = Object.fromEntries(D.variants.map(v => [v.name, v]));
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------------- math and code rendering ---------------- */
/* Chrome's MathML ignores mathvariant="script"/"double-struck", so map those
   letters to their Unicode math characters before rendering. */
const LETTERS = {
  "\\mathcal{L}": "ℒ", "\\mathcal{D}": "𝒟", "\\mathcal D": "𝒟", "\\mathcal{S}": "𝒮", "\\mathcal S": "𝒮",
  "\\mathcal{A}": "𝒜", "\\mathcal A": "𝒜", "\\mathcal{M}": "ℳ", "\\mathcal{T}": "𝒯", "\\mathbb{E}": "𝔼", "\\mathbb{R}": "ℝ",
};
function renderMath(root = document) {
  if (!window.katex) return;
  root.querySelectorAll(".m, .math-block").forEach(node => {
    if (node.dataset.rendered) return;
    let tex = node.textContent;
    for (const [cmd, ch] of Object.entries(LETTERS)) tex = tex.split(cmd).join(ch);
    try {
      katex.render(tex, node, {
        output: "mathml",
        displayMode: node.classList.contains("math-block"),
        throwOnError: false,
        strict: false,
      });
      node.dataset.rendered = "1";
    } catch (err) { /* leave the TeX source visible */ }
  });
}
if (window.hljs) document.querySelectorAll("pre code").forEach(block => hljs.highlightElement(block));

/* ---------------- labels and colours ---------------- */
const LABEL = {
  baseline_fixed: "defaults (Part 4)",
  deeper_net: "deeper network",
  slow_exploration: "slow exploration",
  fast_exploration: "fast exploration",
  double_dqn: "Double DQN",
  step_target_update: "target sync per step",
  improved_dqn: "improved recipe",
};
const COLOR = {
  baseline_fixed: "--neutral-series",
  deeper_net: "--series-3",
  slow_exploration: "--series-4",
  fast_exploration: "--orange",
  double_dqn: "--series-5",
  step_target_update: "--series-6",
  improved_dqn: "--accent",
};
const EXTRA = {
  light: { "--series-3": "#1baf7a", "--series-4": "#eda100", "--series-5": "#4a3aa7", "--series-6": "#e87ba4" },
  dark:  { "--series-3": "#199e70", "--series-4": "#c98500", "--series-5": "#9085e9", "--series-6": "#d55181" },
};
function isDark() {
  const t = document.documentElement.getAttribute("data-theme");
  if (t) return t === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}
function colorOf(name) {
  const token = COLOR[name] || "--neutral-series";
  return EXTRA[isDark() ? "dark" : "light"][token] || css(token);
}
const fmt = (x, d = 0) => (x == null || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
const pct = (x) => `${Math.round(x * 100)}%`;

/* ---------------- tiny SVG chart kit ---------------- */
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}
function niceStep(max, count = 5) {
  for (const s of [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 125, 250, 500]) if (max / s <= count) return s;
  return 1000;
}
function makeTooltip(container) {
  let tip = container.querySelector(".tooltip");
  if (!tip) { tip = document.createElement("div"); tip.className = "tooltip"; tip.hidden = true; container.appendChild(tip); }
  return tip;
}
function placeTip(tip, container, x, y, above = true) {
  tip.hidden = false;
  const cw = container.clientWidth, w = tip.offsetWidth;
  tip.style.left = `${Math.min(Math.max(x - w / 2, 0), Math.max(0, cw - w))}px`;
  tip.style.top = above ? `${y - tip.offsetHeight - 10}px` : `${y}px`;
}

/**
 * Line chart with optional CI bands, area fill, markers and a crosshair tooltip.
 * series: [{name, color, x, y, low?, high?, area?, width?, opacity?, fmt?}]
 */
function lineChart(container, { series, xMax, yMax = 500, xLabel, yLabel, height = 300, yFormat = v => v, markers = [] }) {
  container.innerHTML = "";
  const W = Math.max(300, container.clientWidth || 600);
  const H = height;
  const m = { l: 46, r: 14, t: 10, b: 36 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const sx = x => m.l + (x / xMax) * iw;
  const sy = y => m.t + ih - (Math.min(y, yMax) / yMax) * ih;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": yLabel });
  container.appendChild(svg);

  const ys = niceStep(yMax);
  for (let t = 0; t <= yMax + 1e-9; t += ys) {
    el("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), stroke: css("--grid"), "stroke-width": 1 }, svg);
    const lab = el("text", { x: m.l - 8, y: sy(t) + 4, "text-anchor": "end" }, svg); lab.textContent = yFormat(+t.toFixed(2));
  }
  const xs = niceStep(xMax, W < 500 ? 4 : 6);
  for (let t = 0; t <= xMax + 1e-9; t += xs) {
    const lab = el("text", { x: sx(t), y: H - m.b + 16, "text-anchor": "middle" }, svg); lab.textContent = t;
  }
  el("line", { x1: m.l, x2: W - m.r, y1: sy(0), y2: sy(0), stroke: css("--rule"), "stroke-width": 1 }, svg);
  const xt = el("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, svg); xt.textContent = xLabel;

  for (const mk of markers) {
    el("line", { x1: sx(mk.x), x2: sx(mk.x), y1: m.t, y2: m.t + ih, stroke: css("--ink-2"), "stroke-width": 1, "stroke-dasharray": "4 4" }, svg);
    const t = el("text", { x: sx(mk.x) + 6, y: m.t + 12 }, svg); t.textContent = mk.label; t.setAttribute("style", `fill:${css("--ink-2")}`);
  }
  for (const s of series) {
    if (s.area) {
      let d = `M${sx(s.x[0])},${sy(0)}`;
      s.x.forEach((x, i) => { d += `L${sx(x)},${sy(s.y[i])}`; });
      d += `L${sx(s.x[s.x.length - 1])},${sy(0)}Z`;
      el("path", { d, fill: s.color, "fill-opacity": 0.1, stroke: "none" }, svg);
    }
    if (s.low && s.high) {
      let d = "";
      s.x.forEach((x, i) => { d += `${i ? "L" : "M"}${sx(x)},${sy(s.high[i])}`; });
      for (let i = s.x.length - 1; i >= 0; i--) d += `L${sx(s.x[i])},${sy(s.low[i])}`;
      el("path", { d: d + "Z", fill: s.color, "fill-opacity": 0.12, stroke: "none" }, svg);
    }
  }
  for (const s of series) {
    let d = "";
    s.x.forEach((x, i) => { if (s.y[i] != null) d += `${d ? "L" : "M"}${sx(x)},${sy(s.y[i])}`; });
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-opacity": s.opacity ?? 1, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  }

  const tip = makeTooltip(container);
  const cross = el("line", { y1: m.t, y2: m.t + ih, stroke: css("--muted"), "stroke-width": 1, visibility: "hidden" }, svg);
  const dots = series.map(s => el("circle", { r: 4.5, fill: s.color, stroke: css("--surface"), "stroke-width": 2, visibility: "hidden" }, svg));
  const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
  function move(evt) {
    const rect = svg.getBoundingClientRect();
    const px = (evt.clientX - rect.left) * (W / rect.width);
    const xv = (px - m.l) / iw * xMax;
    const ref = series[series.length - 1];
    let idx = 0, best = Infinity;
    ref.x.forEach((x, i) => { const dd = Math.abs(x - xv); if (dd < best) { best = dd; idx = i; } });
    const x = ref.x[idx];
    cross.setAttribute("x1", sx(x)); cross.setAttribute("x2", sx(x)); cross.setAttribute("visibility", "visible");
    const rows = [];
    series.forEach((s, k) => {
      const j = s.x.indexOf(x);
      const y = j >= 0 ? s.y[j] : null;
      if (y == null) { dots[k].setAttribute("visibility", "hidden"); return; }
      dots[k].setAttribute("cx", sx(x)); dots[k].setAttribute("cy", sy(y)); dots[k].setAttribute("visibility", "visible");
      const ci = s.low ? ` <span style="opacity:.7">[${fmt(s.low[j])}–${fmt(s.high[j])}]</span>` : "";
      rows.push({ y, html: `<div><span class="sw" style="background:${s.color}"></span>${s.name}: ${s.fmt ? s.fmt(y) : fmt(y)}${ci}</div>` });
    });
    rows.sort((a, b) => b.y - a.y);
    tip.innerHTML = `<div style="opacity:.7">${xLabel} ${x}</div>` + rows.slice(0, 11).map(r => r.html).join("");
    placeTip(tip, container, sx(x) * (rect.width / W), m.t * (rect.height / H) + 4, false);
  }
  function leave() { tip.hidden = true; cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); }
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerdown", move);
  hit.addEventListener("pointerleave", leave);
}

/* Dot plot: one row per configuration, one dot per seed, mean bar and CI whisker. */
function dotPlot(container, variants, metric) {
  container.innerHTML = "";
  const W = Math.max(300, container.clientWidth || 600);
  const narrow = W < 560;
  const rowH = 38;
  const m = { l: narrow ? 128 : 180, r: 16, t: 8, b: 34 };
  const H = m.t + m.b + rowH * variants.length;
  const iw = W - m.l - m.r;
  const sx = v => m.l + (v / 500) * iw;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Test return per seed by configuration" });
  container.appendChild(svg);
  for (const t of [0, 100, 200, 300, 400, 500]) {
    el("line", { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b, stroke: css("--grid"), "stroke-width": 1 }, svg);
    const lab = el("text", { x: sx(t), y: H - m.b + 16, "text-anchor": "middle" }, svg); lab.textContent = t;
  }
  const xt = el("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, svg);
  xt.textContent = "mean test return per seed";
  const tip = makeTooltip(container);

  variants.forEach((v, row) => {
    const cy = m.t + rowH * row + rowH / 2;
    const color = colorOf(v.name);
    const label = el("text", { x: m.l - 12, y: cy + 4, "text-anchor": "end" }, svg);
    label.textContent = LABEL[v.name] || v.name;
    label.setAttribute("style", `fill:${css("--ink-2")};font-size:${narrow ? 10.5 : 12}px`);
    const scores = metric === "best" ? v.best_scores : v.final_scores;
    const mean = metric === "best" ? v.best_mean : v.final_mean;
    const ci = metric === "best" ? v.best_ci : v.final_ci;
    el("line", { x1: sx(ci[0]), x2: sx(ci[1]), y1: cy, y2: cy, stroke: css("--ink-2"), "stroke-width": 1.5 }, svg);
    scores.forEach((s, i) => {
      const jitter = ((i * 7919) % 11 - 5) * 1.6;   // deterministic jitter
      const dot = el("circle", { cx: sx(s), cy: cy + jitter, r: 4.5, fill: color, "fill-opacity": 0.85, stroke: css("--surface"), "stroke-width": 2 }, svg);
      const show = () => {
        tip.innerHTML = `${LABEL[v.name]}<br>seed ${v.seeds[i]}: <b>${fmt(s, 1)}</b>`;
        const rect = svg.getBoundingClientRect();
        placeTip(tip, container, sx(s) * rect.width / W, (cy + jitter) * rect.height / H);
      };
      dot.addEventListener("pointerenter", show);
      dot.addEventListener("pointerdown", show);
      dot.addEventListener("pointerleave", () => { tip.hidden = true; });
    });
    el("rect", { x: sx(mean) - 1.5, y: cy - 11, width: 3, height: 22, rx: 1.5, fill: css("--ink") }, svg);
  });
}

/* ---------------- widget: discounting ---------------- */
function renderGamma() {
  const g = +$("#gamma").value;
  $("#gammaOut").textContent = g.toFixed(3);
  const horizon = 1 / (1 - g);
  $("#gHorizon").textContent = fmt(horizon, horizon < 100 ? 1 : 0);
  $("#gForever").textContent = fmt(horizon, 1);
  $("#g10").textContent = fmt((1 - g ** 10) / (1 - g), 1);
  $("#g100").textContent = `${(100 * g ** 100).toFixed(g ** 100 < 0.01 ? 2 : 1)}%`;
  const K = 500;
  const x = Array.from({ length: K + 1 }, (_, k) => k);
  const y = x.map(k => g ** k);
  lineChart($("#gammaChart"), {
    series: [{ name: "γ^k", color: css("--accent"), x, y, area: true, fmt: v => v.toFixed(3) }],
    xMax: K, yMax: 1, xLabel: "steps ahead k =", yLabel: "weight", height: 200,
    markers: horizon <= K ? [{ x: horizon, label: `horizon ≈ ${fmt(horizon)}` }] : [],
  });
}
$("#gamma").addEventListener("input", renderGamma);

/* ---------------- widget: overestimation ---------------- */
function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function randn(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function renderBias() {
  const K = +$("#nActions").value, sigma = +$("#noise").value;
  $("#nActionsOut").textContent = K;
  $("#noiseOut").textContent = sigma.toFixed(1);
  const rng = mulberry32(12345);
  const N = 4000;
  let single = 0, dbl = 0;
  for (let n = 0; n < N; n++) {
    let bestA = -Infinity, argA = 0;
    const b = new Array(K);
    for (let k = 0; k < K; k++) {
      const a = sigma * randn(rng);
      b[k] = sigma * randn(rng);
      if (a > bestA) { bestA = a; argA = k; }
    }
    single += bestA;
    dbl += b[argA];
  }
  single /= N; dbl /= N;
  const container = $("#biasChart");
  container.innerHTML = "";
  const W = Math.max(300, container.clientWidth || 600);
  const narrow = W < 520;
  const rows = [["true value of the best action", 0, css("--neutral-series")], ["single estimator (DQN)", single, css("--orange")], ["double estimator (Double DQN)", dbl, css("--accent")]];
  const m = { l: narrow ? 150 : 220, r: 56, t: 6, b: 26 };
  const rowH = 34, H = m.t + m.b + rowH * rows.length;
  const iw = W - m.l - m.r;
  const xMax = Math.max(1, Math.ceil(sigma * 3.1));
  const sx = v => m.l + (Math.max(0, v) / xMax) * iw;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Average estimated value of the best action" });
  container.appendChild(svg);
  const step = niceStep(xMax, 5);
  for (let t = 0; t <= xMax + 1e-9; t += step) {
    el("line", { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b, stroke: css("--grid"), "stroke-width": 1 }, svg);
    const lab = el("text", { x: sx(t), y: H - m.b + 16, "text-anchor": "middle" }, svg); lab.textContent = +t.toFixed(2);
  }
  rows.forEach(([name, val, color], i) => {
    const cy = m.t + rowH * i + rowH / 2;
    const lab = el("text", { x: m.l - 10, y: cy + 4, "text-anchor": "end" }, svg);
    lab.textContent = name; lab.setAttribute("style", `fill:${css("--ink-2")};font-size:${narrow ? 10.5 : 12}px`);
    const w = Math.max(3, sx(val) - m.l);
    el("rect", { x: m.l, y: cy - 10, width: w, height: 20, rx: 4, fill: color }, svg);
    const v = el("text", { x: m.l + w + 6, y: cy + 4 }, svg);
    v.textContent = (val >= 0 ? "+" : "") + val.toFixed(2);
    v.setAttribute("style", `fill:${css("--ink")}`);
  });
}
$("#nActions").addEventListener("input", renderBias);
$("#noise").addEventListener("input", renderBias);

/* ---------------- widget: Q-learning in a corridor ----------------
   Cells 0..6. Cell 0 ends the episode with reward 0, cell 6 with reward +1.
   The agent starts in cell 3; action 0 moves left, action 1 moves right. */
const CN = 7, CSTART = 3, CG = 0.9, CLIMIT = 50;
const cTerm = s => s === 0 || s === CN - 1;
const cReward = s2 => (s2 === CN - 1 ? 1 : 0);
function cSweep(Q) {   // one application of the Bellman optimality operator
  const next = Q.map(row => row.slice());
  for (let s = 1; s < CN - 1; s++) {
    for (const a of [0, 1]) {
      const s2 = s + (a ? 1 : -1);
      next[s][a] = cReward(s2) + (cTerm(s2) ? 0 : CG * Math.max(...Q[s2]));
    }
  }
  return next;
}
const cZeros = () => Array.from({ length: CN }, () => [0, 0]);
const C_STAR = (() => { let Q = cZeros(); for (let i = 0; i < 200; i++) Q = cSweep(Q); return Q; })();
const cRng = mulberry32(2024);
let cQ, cPos, cEpisodes, cSteps, cEpSteps, cLast, cSweeps;
function cReset() {
  cQ = cZeros(); cPos = CSTART; cEpisodes = 0; cSteps = 0; cEpSteps = 0; cLast = null; cSweeps = 0;
}
function cGreedy(s) {
  const [l, r] = cQ[s];
  if (l === r) return cRng() < 0.5 ? 0 : 1;   // break ties at random
  return r > l ? 1 : 0;
}
function cStep() {
  const eps = +$("#cEps").value, alpha = +$("#cAlpha").value;
  const s = cPos;
  const explore = cRng() < eps;
  const tie = cQ[s][0] === cQ[s][1];
  const a = explore ? (cRng() < 0.5 ? 0 : 1) : cGreedy(s);
  const s2 = s + (a ? 1 : -1);
  const r = cReward(s2), term = cTerm(s2);
  const old = cQ[s][a];
  const boot = term ? 0 : Math.max(...cQ[s2]);
  const target = r + CG * boot;
  cQ[s][a] = old + alpha * (target - old);
  cLast = { s, a, r, s2, term, old, boot, alpha, explore, tie, now: cQ[s][a] };
  cSteps++; cEpSteps++;
  cPos = s2;
  const truncated = !term && cEpSteps >= CLIMIT;
  if (term || truncated) { cEpisodes++; cEpSteps = 0; cPos = CSTART; cLast.ended = term ? "terminated" : "truncated"; }
  return term || truncated;
}
function cRenderQ() {
  const f = x => x.toFixed(2);
  const shade = v => {
    const p = Math.round(Math.max(0, Math.min(1, v)) * 85);
    return { bg: `color-mix(in srgb, var(--accent) ${p}%, var(--surface))`, light: p > 50 };
  };
  $("#corridor").innerHTML = Array.from({ length: CN }, (_, s) => {
    if (s === 0) return `<div class="cell term">cell 0<br>end<br>r = 0</div>`;
    if (s === CN - 1) return `<div class="cell term goal">cell 6<br>goal<br>r = +1</div>`;
    const rows = [0, 1].map(a => {
      const sh = shade(cQ[s][a]);
      return `<div class="q${sh.light ? " light" : ""}" style="background:${sh.bg}">${a ? "→" : "←"} ${f(cQ[s][a])}</div>`;
    }).join("");
    const here = s === cPos;
    return `<div class="cell${here ? " here" : ""}"><div class="idx">${here ? '<span class="agent" title="agent"></span>' : ""}${s}</div>${rows}<div class="star">Q*<br>${f(C_STAR[s][0])}<br>${f(C_STAR[s][1])}</div></div>`;
  }).join("");
  let err = 0;
  for (let s = 1; s < CN - 1; s++) for (const a of [0, 1]) err = Math.max(err, Math.abs(cQ[s][a] - C_STAR[s][a]));
  $("#cStats").innerHTML = `<span>episodes <b>${cEpisodes}</b></span><span>steps <b>${cSteps}</b></span><span>value-iteration sweeps <b>${cSweeps}</b></span><span>max |Q − Q*| <b>${err.toFixed(3)}</b></span>`;
  const L = cLast;
  if (!L) { $("#cLog").textContent = "Press a button to start. Every Q-value starts at 0."; return; }
  if (L.vi) { $("#cLog").innerHTML = "Value iteration: every entry set to r + γ · max Q(s′, ·), using the known model.<br>The error shrinks by at least a factor γ = 0.9 per sweep."; return; }
  const arrow = L.a ? "→" : "←";
  const how = L.explore ? "explore (random)" : L.tie ? "greedy (tie, broken at random)" : "greedy";
  const next = L.term ? `cell ${L.s2} is terminal, so no future term` : `max Q(${L.s2}, ·) = ${f(L.boot)}`;
  $("#cLog").innerHTML = `${how}: in cell ${L.s} go ${arrow}, land in cell ${L.s2}, r = ${L.r}${L.ended ? ` (${L.ended})` : ""}; ${next}<br>Q(${L.s}, ${arrow}) ← ${f(L.old)} + ${L.alpha} × [${L.r} + 0.9 × ${f(L.boot)} − ${f(L.old)}] = <b>${f(L.now)}</b>`;
}
function buildCorridor() {
  cReset();
  const upd = () => { $("#cEpsOut").textContent = (+$("#cEps").value).toFixed(2); $("#cAlphaOut").textContent = (+$("#cAlpha").value).toFixed(2); };
  $("#cEps").addEventListener("input", upd);
  $("#cAlpha").addEventListener("input", upd);
  upd();
  $("#cStep").addEventListener("click", () => { cStep(); cRenderQ(); });
  $("#cEpisode").addEventListener("click", () => { while (!cStep()); cRenderQ(); });
  $("#cMany").addEventListener("click", () => { for (let i = 0; i < 50; i++) while (!cStep()); cRenderQ(); });
  $("#cVI").addEventListener("click", () => { cQ = cSweep(cQ); cSweeps++; cLast = { vi: true }; cRenderQ(); });
  $("#cReset").addEventListener("click", () => { cReset(); cRenderQ(); });
  cRenderQ();
}

/* ---------------- result charts ---------------- */
function renderEpsilon() {
  const names = ["fast_exploration", "baseline_fixed", "slow_exploration"].filter(n => byName[n]);
  const series = names.map(n => {
    const eps = byName[n].epsilon_curve;
    return { name: LABEL[n], color: colorOf(n), x: eps.map((_, i) => i * 5 + 1), y: eps.map(e => e * 100), fmt: v => `${fmt(v)}%` };
  });
  lineChart($("#epsChart"), { series, xMax: 300, yMax: 100, xLabel: "episode", yLabel: "epsilon (%)", height: 220, yFormat: v => `${v}%` });
  const T = { fast_exploration: "2,500", baseline_fixed: "8,000 (defaults)", slow_exploration: "20,000" };
  $("#epsLegend").innerHTML = names.map(n => `<span class="chip" style="--c:${colorOf(n)};cursor:default"><span class="sw"></span>T = ${T[n]} steps</span>`).join("");
}

let spagVariant = "baseline_fixed";
function renderSpaghetti() {
  const v = byName[spagVariant];
  const color = colorOf(spagVariant);
  const series = v.greedy_per_seed.map((ys, i) => ({
    name: `seed ${v.seeds[i]}`, color, x: ys.map((_, j) => (j + 1) * 10), y: ys, width: 1.2, opacity: 0.45,
  }));
  series.push({ name: "mean", color: css("--ink"), x: v.greedy_curve.episode, y: v.greedy_curve.mean, width: 3 });
  lineChart($("#spagChart"), { series, xMax: 300, xLabel: "episode", yLabel: "greedy return" });
}
function buildSpagSeg() {
  const choices = ["baseline_fixed", "deeper_net", "improved_dqn"].filter(n => byName[n]);
  const seg = $("#spagSeg");
  seg.innerHTML = choices.map(n => `<button type="button" data-v="${n}" aria-pressed="${n === spagVariant}">${LABEL[n]}</button>`).join("");
  seg.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    spagVariant = b.dataset.v;
    seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b));
    renderSpaghetti();
  });
}

let dotMetric = "final";
function renderDots() { dotPlot($("#dotChart"), D.variants, dotMetric); }
function segToggle(id, onChange) {
  $(id).addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    $(id).querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b));
    onChange(b.dataset.metric);
  });
}

const curveOn = new Set(["baseline_fixed", "slow_exploration", "improved_dqn"]);
function renderCurves() {
  const series = D.variants.filter(v => curveOn.has(v.name)).map(v => ({
    name: LABEL[v.name], color: colorOf(v.name),
    x: v.greedy_curve.episode, y: v.greedy_curve.mean, low: v.greedy_curve.low, high: v.greedy_curve.high,
  }));
  if (!series.length) { $("#curveChart").innerHTML = ""; return; }
  lineChart($("#curveChart"), { series, xMax: 300, xLabel: "episode", yLabel: "greedy return", height: 320 });
}
function buildCurveLegend() {
  const legend = $("#curveLegend");
  legend.innerHTML = D.variants.map(v => `<button type="button" class="chip" data-v="${v.name}" aria-pressed="${curveOn.has(v.name)}" style="--c:${colorOf(v.name)}"><span class="sw"></span>${LABEL[v.name]}</button>`).join("");
  legend.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    const n = b.dataset.v;
    curveOn.has(n) ? curveOn.delete(n) : curveOn.add(n);
    b.setAttribute("aria-pressed", curveOn.has(n));
    renderCurves();
  });
}

function buildTable() {
  const head = "<tr><th>configuration</th><th>what changes</th><th class='n'>final mean</th><th class='n'>95% CI</th><th class='n'>perfect</th><th class='n'>best-ckpt mean</th><th class='n'>95% CI</th><th class='n'>perfect</th></tr>";
  const rows = D.variants.map(v => `<tr><td>${LABEL[v.name]}</td><td style="font-family:var(--font-body)">${v.description}</td><td class="n">${fmt(v.final_mean, 1)}</td><td class="n">${fmt(v.final_ci[0])}–${fmt(v.final_ci[1])}</td><td class="n">${pct(v.final_frac_perfect)}</td><td class="n">${fmt(v.best_mean, 1)}</td><td class="n">${fmt(v.best_ci[0])}–${fmt(v.best_ci[1])}</td><td class="n">${pct(v.best_frac_perfect)}</td></tr>`).join("");
  $("#summaryTable").innerHTML = `<table>${head}${rows}</table>`;
}

/* How each configuration's test episodes end, as 100% stacked bars. */
const FAIL = [
  ["time_limit", "reached 500 steps (success)", "--accent"],
  ["cart_off_track", "cart left the track", "--orange"],
  ["pole_fell", "pole fell", "--violet"],
];
let failMetric = "final";
function renderFail() {
  const container = $("#failChart");
  container.innerHTML = "";
  const FM = D.failure_modes; if (!FM) return;
  const W = Math.max(300, container.clientWidth || 600);
  const narrow = W < 560;
  const rowH = 32, barH = 20;
  const m = { l: narrow ? 128 : 180, r: 16, t: 4, b: 30 };
  const variants = D.variants.filter(v => FM[v.name]);
  const H = m.t + m.b + rowH * variants.length;
  const iw = W - m.l - m.r;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "How test episodes end, by configuration" });
  container.appendChild(svg);
  for (const t of [0, 25, 50, 75, 100]) {
    const lab = el("text", { x: m.l + iw * t / 100, y: H - m.b + 16, "text-anchor": "middle" }, svg); lab.textContent = `${t}%`;
  }
  const tip = makeTooltip(container);
  variants.forEach((v, row) => {
    const counts = FM[v.name][failMetric];
    const total = FAIL.reduce((a, [k]) => a + counts[k], 0);
    const cy = m.t + rowH * row + rowH / 2;
    const label = el("text", { x: m.l - 12, y: cy + 4, "text-anchor": "end" }, svg);
    label.textContent = LABEL[v.name];
    label.setAttribute("style", `fill:${css("--ink-2")};font-size:${narrow ? 10.5 : 12}px`);
    let x = m.l;
    FAIL.forEach(([key, name, token]) => {
      const w = iw * counts[key] / total;
      if (w <= 0) return;
      const x0 = x;
      const seg = el("rect", { x: x0 + 1, y: cy - barH / 2, width: Math.max(0, w - 2), height: barH, rx: 3, fill: css(token) }, svg);
      if (w > 34) {
        const t = el("text", { x: x0 + w / 2, y: cy + 4, "text-anchor": "middle" }, svg);
        t.textContent = `${Math.round(100 * counts[key] / total)}%`;
        t.setAttribute("style", `fill:${isDark() ? "#101418" : "#ffffff"};font-weight:600;pointer-events:none`);
      }
      const show = () => {
        tip.innerHTML = `${LABEL[v.name]}<br><span class="sw" style="background:${css(token)}"></span>${name}: <b>${counts[key]}</b> of ${total}`;
        const rect = svg.getBoundingClientRect();
        placeTip(tip, container, (x0 + w / 2) * rect.width / W, (cy - barH / 2) * rect.height / H);
      };
      seg.addEventListener("pointerenter", show);
      seg.addEventListener("pointerdown", show);
      seg.addEventListener("pointerleave", () => { tip.hidden = true; });
      x += w;
    });
  });
  $("#failLegend").innerHTML = FAIL.map(([, name, token]) => `<span class="chip" style="--c:${css(token)};cursor:default"><span class="sw" style="height:10px;width:10px;border-radius:3px"></span>${name}</span>`).join("");
}

/* ---------------- CartPole physics (same equations and constants as Gymnasium) ---------------- */
const G = 9.8, MC = 1.0, MP = 0.1, TOTAL = MC + MP, L = 0.5, PML = MP * L, FORCE = 10, TAU = 0.02;
const THETA_LIMIT = 12 * 2 * Math.PI / 360, X_LIMIT = 2.4, MAX_STEPS = 500;

function physicsStep(s, action) {
  const [x, xd, th, thd] = s;
  const force = action === 1 ? FORCE : -FORCE;
  const c = Math.cos(th), sn = Math.sin(th);
  const temp = (force + PML * thd * thd * sn) / TOTAL;
  const thacc = (G * sn - c * temp) / (L * (4 / 3 - MP * c * c / TOTAL));
  const xacc = temp - PML * thacc * c / TOTAL;
  return [x + TAU * xd, xd + TAU * xacc, th + TAU * thd, thd + TAU * thacc];
}

/* The trained network in plain JS: Linear -> ReLU -> ... -> Linear. */
function forward(net, input) {
  let h = input;
  net.layers.forEach((layer, li) => {
    const out = new Array(layer.b.length);
    for (let i = 0; i < out.length; i++) {
      let sum = layer.b[i];
      const row = layer.W[i];
      for (let j = 0; j < h.length; j++) sum += row[j] * h[j];
      out[i] = li < net.layers.length - 1 ? Math.max(0, sum) : sum;
    }
    h = out;
  });
  return h;
}

/* ---------------- text that depends on the numbers ---------------- */
function bindText() {
  const T = window.DQN_TEXT ? window.DQN_TEXT(D, forward) : {};
  document.querySelectorAll("[data-bind]").forEach(node => {
    const key = node.dataset.bind;
    if (T[key] != null) node.innerHTML = T[key];
  });
  if (T.results_prose) $("#resultsProse").innerHTML = T.results_prose;
}

function renderAll() { renderGamma(); renderBias(); renderEpsilon(); renderSpaghetti(); renderDots(); renderFail(); renderCurves(); }

bindText();
renderMath();
buildCorridor();
buildSpagSeg();
segToggle("#dotSeg", m => { dotMetric = m; renderDots(); });
segToggle("#failSeg", m => { failMetric = m; renderFail(); });
buildCurveLegend();
buildTable();
renderAll();
let resizeTimer, lastWidth = window.innerWidth;
window.addEventListener("resize", () => {
  if (window.innerWidth === lastWidth) return;
  lastWidth = window.innerWidth;
  clearTimeout(resizeTimer); resizeTimer = setTimeout(renderAll, 150);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
new MutationObserver(renderAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

/* ---------------- the live demo ---------------- */
const canvas = $("#sim");
const ctx = canvas.getContext("2d");
let state, steps, best = 0, ended = null, pauseUntil = 0, humanDir = 0, lastQ = null;
function reset() {
  state = [0, 0, 0, 0].map(() => Math.random() * 0.1 - 0.05);
  steps = 0; ended = null;
  setStatus("");
}
function setStatus(html) { $("#status").innerHTML = html; }

function chooseAction() {
  const mode = $("#policy").value;
  lastQ = null;
  if (mode === "random") return Math.random() < 0.5 ? 0 : 1;
  if (mode === "human") return humanDir > 0 ? 1 : humanDir < 0 ? 0 : (steps % 2);
  const q = forward(D.networks[mode], state);
  lastQ = q;
  return q[1] > q[0] ? 1 : 0;
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
function draw(action) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cw = canvas.clientWidth, ch = cw / 2.2;
  if (canvas.width !== Math.round(cw * dpr)) { canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const scale = cw / (2 * X_LIMIT * 1.15);
  const groundY = ch * 0.78;
  const cx = cw / 2 + state[0] * scale;

  ctx.strokeStyle = css("--track"); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cw / 2 - X_LIMIT * scale, groundY); ctx.lineTo(cw / 2 + X_LIMIT * scale, groundY); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = `11px ${css("--font-mono")}`; ctx.textAlign = "center";
  for (const s of [-1, 1]) {
    const lx = cw / 2 + s * X_LIMIT * scale;
    ctx.fillRect(lx - 1, groundY - 8, 2, 16);
    ctx.fillText(`${s * 2.4} m`, lx, groundY + 24);
  }
  const cartW = Math.max(44, scale * 0.5), cartH = cartW * 0.36;
  const pivotY = groundY - cartH - 4;
  const poleLen = Math.min(ch * 0.62, scale * 2 * L * 1.6);
  ctx.fillStyle = css("--grid");
  ctx.beginPath(); ctx.moveTo(cx, pivotY);
  ctx.arc(cx, pivotY, poleLen * 0.9, -Math.PI / 2 - THETA_LIMIT, -Math.PI / 2 + THETA_LIMIT); ctx.closePath(); ctx.fill();

  if (action != null && !ended) {
    const dir = action === 1 ? 1 : -1;
    ctx.strokeStyle = css("--accent"); ctx.fillStyle = css("--accent"); ctx.lineWidth = 3;
    const ax = cx - dir * (cartW / 2 + 8), ay = groundY - cartH / 2 - 4;
    ctx.beginPath(); ctx.moveTo(ax - dir * 22, ay); ctx.lineTo(ax, ay); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ax + dir * 2, ay); ctx.lineTo(ax - dir * 7, ay - 6); ctx.lineTo(ax - dir * 7, ay + 6); ctx.closePath(); ctx.fill();
  }
  ctx.fillStyle = css("--cart");
  roundRect(cx - cartW / 2, groundY - cartH - 4, cartW, cartH, 5); ctx.fill();
  ctx.fillStyle = css("--track");
  for (const s of [-1, 1]) { ctx.beginPath(); ctx.arc(cx + s * cartW * 0.28, groundY - 3, 4, 0, Math.PI * 2); ctx.fill(); }
  const th = state[2];
  ctx.strokeStyle = ended === "terminated" ? css("--bug") : css("--pole");
  ctx.lineWidth = Math.max(6, cartW * 0.13); ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(cx, pivotY); ctx.lineTo(cx + Math.sin(th) * poleLen, pivotY - Math.cos(th) * poleLen); ctx.stroke();
  ctx.fillStyle = css("--surface");
  ctx.beginPath(); ctx.arc(cx, pivotY, 3.5, 0, Math.PI * 2); ctx.fill();
}

function updatePanel(action) {
  $("#stepCount").textContent = steps;
  $("#bestCount").textContent = best;
  $("#progress").style.width = `${(steps / MAX_STEPS) * 100}%`;
  $("#sx").textContent = `${state[0].toFixed(2)} m`;
  $("#sxd").textContent = `${state[1].toFixed(2)} m/s`;
  $("#sth").textContent = `${(state[2] * 180 / Math.PI).toFixed(1)}°`;
  $("#sthd").textContent = `${state[3].toFixed(2)} rad/s`;
  const qL = $("#qL"), qR = $("#qR");
  if (lastQ) {
    const lo = Math.min(...lastQ) - 1, hi = Math.max(...lastQ, lo + 1);
    const w = q => `${Math.max(4, ((q - lo) / (hi - lo)) * 100)}%`;
    qL.style.width = w(lastQ[0]); qR.style.width = w(lastQ[1]);
    qL.classList.toggle("chosen", action === 0); qR.classList.toggle("chosen", action === 1);
    $("#qLv").textContent = lastQ[0].toFixed(1); $("#qRv").textContent = lastQ[1].toFixed(1);
  } else {
    qL.style.width = "0"; qR.style.width = "0";
    $("#qLv").textContent = "–"; $("#qRv").textContent = "–";
  }
}

let acc = 0, last = performance.now(), lastAction = null, running = false;
function tick(now) {
  acc += Math.min(now - last, 100); last = now;
  const dt = TAU * 1000;
  while (acc >= dt) {
    acc -= dt;
    if (ended) { if (now > pauseUntil) reset(); continue; }
    lastAction = chooseAction();
    state = physicsStep(state, lastAction);
    steps++;
    best = Math.max(best, steps);
    if (Math.abs(state[0]) > X_LIMIT || Math.abs(state[2]) > THETA_LIMIT) {
      ended = "terminated"; pauseUntil = now + 1400;
      const why = Math.abs(state[0]) > X_LIMIT ? "the cart left the track" : "the pole fell";
      setStatus(`<span class="bad">terminated at step ${steps}: ${why} (d = 1)</span>`);
    } else if (steps >= MAX_STEPS) {
      ended = "truncated"; pauseUntil = now + 2200;
      setStatus(`<span class="good">truncated at step 500: the pole is still up, so d = 0</span>`);
    }
  }
  draw(lastAction);
  updatePanel(lastAction);
  requestAnimationFrame(tick);
}
function start() { if (!running) { running = true; last = performance.now(); requestAnimationFrame(tick); } }

function poke(dir) { if (!ended) state = [state[0], state[1], state[2], state[3] + dir * 0.55]; }
$("#pokeL").addEventListener("click", () => { poke(-1); start(); });
$("#pokeR").addEventListener("click", () => { poke(1); start(); });
$("#resetBtn").addEventListener("click", () => { best = 0; reset(); start(); });
$("#policy").addEventListener("change", () => { best = 0; reset(); start(); });
window.addEventListener("keydown", e => {
  if ($("#policy").value !== "human") return;
  if (e.key === "ArrowLeft") { humanDir = -1; e.preventDefault(); start(); }
  if (e.key === "ArrowRight") { humanDir = 1; e.preventDefault(); start(); }
});
window.addEventListener("keyup", e => { if (e.key === "ArrowLeft" || e.key === "ArrowRight") humanDir = 0; });
for (const [id, dir] of [["pokeL", -1], ["pokeR", 1]]) {
  const b = $("#" + id);
  b.addEventListener("pointerdown", () => { if ($("#policy").value === "human") humanDir = dir; });
  b.addEventListener("pointerup", () => { humanDir = 0; });
  b.addEventListener("pointerleave", () => { humanDir = 0; });
}

reset();
draw(null);
updatePanel(null);
new ResizeObserver(() => { if (!running) draw(lastAction); }).observe(canvas);
if (reduceMotion) setStatus("Paused because your system asks for reduced motion. Press reset to start.");
else start();
})();
