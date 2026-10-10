/* A Tour of Machine Learning, chapter 02, Finding minima and maxima: the live demos.
   The hero runs the notebook's gradient descent on its loss L(w) = a w^2 - 2 b w + c, with
   a, b, c computed in Python from the notebook's 50 points (rng = default_rng(0)).
   Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;

/* ---------------- shared helpers ---------------- */
function dot(ctx, x, y, r, fill, stroke, lw = 2) { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); } }
function arrow(ctx, x0, y0, x1, y1, col, lw = 2.2) {
  const a = Math.atan2(y1 - y0, x1 - x0), L = 9;
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw; seg(ctx, x0, y0, x1 - 0.6 * L * Math.cos(a), y1 - 0.6 * L * Math.sin(a));
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - L * Math.cos(a - 0.4), y1 - L * Math.sin(a - 0.4)); ctx.lineTo(x1 - L * Math.cos(a + 0.4), y1 - L * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill();
}
const range = (n, f) => Array.from({ length: n }, (_, i) => f(i));
const signed = (v, d) => (v < 0 ? "−" : "") + fmt(Math.abs(v), d);
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const sci = v => { if (!isFinite(v)) return "overflow"; const a = Math.abs(v); if (a !== 0 && (a < 1e-3 || a >= 1e5)) { const [m, e] = v.toExponential(1).split("e"); return `${m.replace("-", "−")} × 10${String(+e).split("").map(ch => SUP[ch]).join("")}`; } return signed(v, 4); };

/* =====================================================================
   THE DEMO: CHOOSE A LEARNING RATE
   ===================================================================== */
const A = 36.250118330485215, B = 84.61420224081971, C = 201.5331750049264;
const WSTAR = B / A, LMIN = C - B * B / A;
const loss = w => A * w * w - 2 * B * w + C, grad = w => 2 * A * w - 2 * B;
const hero = { eta: 0.002 };
function etaLabel(e) { return e < 0.01 ? fmt(e, 4) : fmt(e, 4); }
function runGD(eta, T) { const path = [0]; let w = 0; for (let t = 0; t < T; t++) { w = w - eta * grad(w); path.push(w); } return path; }
function drawHero() {
  const eta = hero.eta, T = +$("#hT").value, q = 1 - 2 * A * eta, path = runGD(eta, T);
  $("#hEV").textContent = etaLabel(eta); $("#hTV").textContent = String(T);
  // the loss and the path
  const cv = $("#lossC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.3 : 2.4, { x: [-2, 6], y: [0, 700], xlabel: "w", ylabel: "L(w)", yticks: [0, 200, 400, 600] });
  const { ctx, X, Y } = fr, ws = range(161, i => -2 + i * 0.05);
  plotLine(fr, ws, ws.map(loss), { color: css("--post"), width: 2.2 });
  const shown = path.slice(0, 61), inside = w => w >= -2 && w <= 6;
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  ctx.strokeStyle = alpha("--warm", 0.75); ctx.lineWidth = 1.2;
  for (let i = 1; i < shown.length; i++) seg(ctx, X(shown[i - 1]), Y(loss(shown[i - 1])), X(shown[i]), Y(loss(shown[i])));
  ctx.restore();
  shown.forEach((w, i) => { if (inside(w) && loss(w) <= 700) dot(ctx, X(w), Y(loss(w)), i === 0 ? 4.5 : 3, css("--warm")); });
  const escaped = shown.findIndex(w => !inside(w) || loss(w) > 700);
  if (escaped > 0) { ctx.fillStyle = css("--warm"); ctx.font = "600 11px " + css("--font-ui"); ctx.textAlign = shown[escaped] > WSTAR ? "right" : "left"; ctx.fillText(`step ${escaped} leaves the chart`, shown[escaped] > WSTAR ? fr.w - fr.pad.r - 4 : fr.pad.l + 4, fr.pad.t + 12); }
  dot(ctx, X(WSTAR), Y(LMIN), 6, css("--c4"));
  // the distance to the minimum, log scale
  const dv = $("#distC"), df = frame(dv, narrow ? 1.6 : 2.6, { x: [0, T], y: [-16, 16], xlabel: "step", yticks: [-15, -10, -5, 0, 5, 10, 15], yfmt: v => (v === 0 ? "1" : `1e${v}`) });
  // stop the line where the distance reaches floating-point precision (about 1e-15 here): below that it is rounding, not progress
  const floor = path.findIndex(w => Math.abs(w - WSTAR) < 1e-14), end = floor >= 0 ? floor : T;
  const lg = path.slice(0, end + 1).map(w => Math.log10(Math.max(1e-16, Math.abs(w - WSTAR))));
  df.ctx.strokeStyle = alpha("--ink", 0.35); df.ctx.setLineDash([3, 3]); seg(df.ctx, df.X(0), df.Y(0), df.X(T), df.Y(0)); df.ctx.setLineDash([]);
  plotLine(df, lg.map((_, t) => t), lg.map(v => clamp(v, -16, 16)), { color: css("--warm"), width: 2 });
  if (floor >= 0 && floor < T) {
    df.ctx.fillStyle = css("--muted"); df.ctx.font = "600 10px " + css("--font-ui"); df.ctx.textAlign = df.X(floor) > df.pad.l + df.iw / 2 ? "right" : "left";
    df.ctx.fillText("rounding error reached", df.X(floor) + (df.ctx.textAlign === "left" ? 6 : -6), df.Y(lg[lg.length - 1]) - 6);
  }
  // readouts
  const wT = path[T], k = path.findIndex(w => Math.abs(w - WSTAR) < 5e-5);
  $("#rQ").textContent = signed(q, 4); $("#rW").textContent = sci(wT); $("#rL").textContent = sci(loss(wT));
  $("#rK").textContent = k >= 0 ? String(k) : Math.abs(q) >= 1 ? "never" : `more than ${T}`;
  let text = `With [[\\eta = ${fmt(eta, 4)}]], each step multiplies the distance to the minimum by [[1 - 2a\\eta = ${fmt(q, 4)}]]. `;
  if (Math.abs(q) < 1e-6) text += `<b>One step lands exactly on the minimum</b>, [[w^* = ${fmt(WSTAR, 4)}]]: this is [[\\eta = \\frac{1}{2a}]], the step Newton's method would take (exercise 3).`;
  else if (Math.abs(q + 1) < 1e-6) text += `<b>The edge of stability</b>, [[\\eta = \\frac{1}{a}]]: the factor is exactly −1, so [[w]] bounces between 0 and [[${fmt(2 * WSTAR, 4)}]] forever, neither converging nor exploding.`;
  else if (q > 0) text += q > 0.97 ? `<b>Too timid</b>: every step lands on the same side, a little closer. It is safe, but ${k >= 0 ? `it needs ${k} steps` : `${T} steps are not enough`} to match [[w^*]] to 4 decimals.` : `<b>Steady</b>: every step lands on the same side of the minimum, a fixed fraction closer. ${k >= 0 ? `After ${k} steps [[w]] matches [[w^* = ${fmt(WSTAR, 4)}]] to 4 decimals.` : `More steps are needed to match [[w^*]] to 4 decimals.`}`;
  else if (q > -1) text += `<b>Zig-zag</b>: the factor is negative, so each step jumps across the minimum, but it shrinks the distance, so the path still converges${k >= 0 ? `, matching [[w^*]] to 4 decimals after ${k} steps` : ""}.`;
  else text += `<b>Explodes</b>: the factor is below −1, so each step overshoots further than the last, and the loss grows at every step. [[\\eta]] is above [[\\frac{1}{a} = ${fmt(1 / A, 4)}]], the limit the curvature sets.`;
  caption($("#heroStatus"), text);
}
function setEta(e) { hero.eta = e; $("#hE").value = String(Math.log10(e)); drawHero(); }
$("#hE").addEventListener("input", e => { hero.eta = 10 ** +e.target.value; drawHero(); });
$("#hT").addEventListener("input", drawHero);
$$("#demo [data-eta]").forEach(b => b.addEventListener("click", () => {
  const v = b.dataset.eta; setEta(v === "half" ? 1 / (2 * A) : v === "edge" ? 1 / A : +v);
}));

/* =====================================================================
   PART 1: FROM CHORD TO TANGENT
   ===================================================================== */
function drawChord() {
  const x0 = +$("#cX").value, h = 10 ** +$("#cH").value, cv = $("#chordC");
  $("#cXV").textContent = signed(x0, 1); $("#cHV").textContent = h >= 0.1 ? fmt(h, 2) : h >= 0.01 ? fmt(h, 3) : fmt(h, 4);
  const fr = frame(cv, narrowOf(cv) ? 1.3 : 2.4, { x: [-1.5, 5.5], y: [-2, 30], xlabel: "x", yticks: [0, 10, 20, 30] });
  const { ctx, X, Y } = fr, xs = range(141, i => -1.5 + i * 0.05);
  plotLine(fr, xs, xs.map(v => v * v), { color: css("--post"), width: 2.4 });
  const slope = 2 * x0, chordSlope = 2 * x0 + h;
  plotLine(fr, [-1.5, 5.5], [-1.5, 5.5].map(v => x0 * x0 + slope * (v - x0)), { color: css("--ink"), width: 1.6, dash: [6, 4] });
  plotLine(fr, [-1.5, 5.5], [-1.5, 5.5].map(v => x0 * x0 + chordSlope * (v - x0)), { color: css("--warm"), width: 2 });
  dot(ctx, X(x0), Y(x0 * x0), 5, css("--ink")); dot(ctx, X(x0 + h), Y((x0 + h) ** 2), 5, css("--warm"));
  const hs = h >= 0.1 ? fmt(h, 2) : h >= 0.01 ? fmt(h, 3) : fmt(h, 4), cs = fmt(chordSlope, h >= 0.1 ? 2 : h >= 0.01 ? 3 : 4);
  caption($("#chordCap"), `The chord from [[x = ${fmt(x0, 1)}]] to [[x + h = ${fmt(x0 + h, 4).replace(/\.?0+$/, "")}]] has slope [[\\frac{f(x+h) - f(x)}{h} = 2x + h = ${cs}]]. The tangent's slope is [[f'(x) = 2x = ${fmt(slope, 0)}]]: the gap between the two is exactly [[h = ${hs}]], and it vanishes as [[h]] shrinks.` + (Math.abs(x0) < 1e-9 ? " At x = 0 the tangent is flat: a critical point, here the minimum." : ""));
}
["#cX", "#cH"].forEach(id => $(id).addEventListener("input", drawChord));

/* =====================================================================
   PART 2: BOWL, SADDLE AND NARROW VALLEY, L = p w1^2 + q w2^2
   ===================================================================== */
const SURF = { bowl: [1, 3, "w_1^2 + 3w_2^2"], saddle: [1, -1, "w_1^2 - w_2^2"], valley: [1, 10, "w_1^2 + 10\\,w_2^2"] };
const surf = { kind: "bowl", pt: [1.2, 0.8], fr: null, drag: false };
function drawSurf() {
  $$("#surf [data-surf]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.surf === surf.kind)));
  const [p, q, texL] = SURF[surf.kind], th = +$("#sA").value * Math.PI / 180, u = [Math.cos(th), Math.sin(th)];
  $("#sAV").textContent = `${$("#sA").value}° from the w₁ axis`;
  const cv = $("#surfC"), fr = surf.fr = frame(cv, 1, { x: [-2, 2], y: [-2, 2], xlabel: "w₁", ylabel: "w₂", xticks: [-2, -1, 0, 1, 2], yticks: [-2, -1, 0, 1, 2] });
  const { ctx, X, Y } = fr, Lf = (a, b) => p * a * a + q * b * b, step = surf.kind === "valley" ? 2 : 0.5;
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  const cell = 3, shade = new Map();                                            // one colour per (sign, band), looked up once
  const colour = (neg, band) => { const key = (neg ? -1 : 1) * (band + 1); if (!shade.has(key)) shade.set(key, alpha(neg ? "--warm" : "--post", 0.06 + Math.min(band, 10) * 0.025 + (band % 2) * 0.05)); return shade.get(key); };
  for (let px = fr.pad.l; px < fr.pad.l + fr.iw; px += cell) for (let py = fr.pad.t; py < fr.pad.t + fr.ih; py += cell) {
    const a = -2 + (px + cell / 2 - fr.pad.l) / fr.iw * 4, b = 2 - (py + cell / 2 - fr.pad.t) / fr.ih * 4, v = Lf(a, b);
    ctx.fillStyle = colour(v < 0, Math.floor(Math.abs(v) / step)); ctx.fillRect(px, py, cell, cell);
  }
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]);
  seg(ctx, X(-2.9 * u[0]), Y(-2.9 * u[1]), X(2.9 * u[0]), Y(2.9 * u[1])); ctx.setLineDash([]);
  ctx.restore();
  const [w1, w2] = surf.pt, g = [2 * p * w1, 2 * q * w2], gn = Math.hypot(...g);
  if (gn > 1e-9) arrow(ctx, X(w1), Y(w2), X(w1 - 0.6 * g[0] / gn), Y(w2 - 0.6 * g[1] / gn), css("--c4"));
  dot(ctx, X(w1), Y(w2), 6, css("--surface"), css("--ink"), 2.2);
  dot(ctx, X(0), Y(0), 3.5, css("--ink"));
  // the slice L(t u) = k t^2, k = p cos^2 + q sin^2 = (1/2) u'Hu
  const k = p * u[0] ** 2 + q * u[1] ** 2, sv = $("#sliceC"), sAspect = narrowOf(sv) ? 1.5 : 1, ylim = surf.kind === "valley" ? [-2, 42] : [-4.5, 13];
  const sf = frame(sv, sAspect, { x: [-2, 2], y: ylim, xlabel: "distance along the direction", xticks: [-2, -1, 0, 1, 2] });
  const ts = range(81, i => -2 + i * 0.05);
  plotLine(sf, ts, ts.map(t => k * t * t), { color: css(k >= 0 ? "--post" : "--warm"), width: 2.4 });
  dot(sf.ctx, sf.X(0), sf.Y(0), 3.5, css("--ink"));
  const kind = surf.kind === "saddle" ? "a <b>saddle</b>: the curvature is positive along w₁ and negative along w₂, so the origin is a minimum along some directions and a maximum along others" : surf.kind === "bowl" ? "a <b>minimum</b>: the curvature is positive in every direction, so every direction leads uphill" : "a <b>minimum</b> at the bottom of a long, narrow valley: every direction leads uphill, but ten times more steeply along w₂ than along w₁ (exercise 4)";
  const hh = 2 * k;
  caption($("#surfCap"), `For [[L = ${texL}]], along this direction [[h^\\top H h = ${fmt(hh, 2).replace("-", "−")}]] for a step [[h]] of length 1, so the slice bends ${Math.abs(hh) < 1e-9 ? "neither way: it is flat" : hh > 0 ? "up" : "down"}. The origin is ${kind}. At the point you dragged, [[\\nabla L = (${fmt(g[0], 2)}, ${fmt(g[1], 2)})]]; the green arrow, [[-\\nabla L]], is the way gradient descent would step.`);
}
(() => {
  const cv = $("#surfC");
  const move = ev => {
    if (!surf.drag || !surf.fr) return;
    const r = cv.getBoundingClientRect(), fr = surf.fr;
    surf.pt = [clamp(-2 + (ev.clientX - r.left - fr.pad.l) / fr.iw * 4, -2, 2), clamp(2 - (ev.clientY - r.top - fr.pad.t) / fr.ih * 4, -2, 2)];
    drawSurf();
  };
  cv.addEventListener("pointerdown", ev => { surf.drag = true; cv.setPointerCapture(ev.pointerId); move(ev); });
  cv.addEventListener("pointermove", move);
  ["pointerup", "pointercancel"].forEach(t => cv.addEventListener(t, () => { surf.drag = false; }));
  $$("#surf [data-surf]").forEach(b => b.addEventListener("click", () => { surf.kind = b.dataset.surf; drawSurf(); }));
  $("#sA").addEventListener("input", drawSurf);
})();

/* =====================================================================
   PART 3: THREE STEPS BY HAND, L(w) = (w - 3)^2 FROM w = 0
   ===================================================================== */
const num = v => { const s = String(+v.toFixed(6)); return s === "-0" ? "0" : s.replace("-", "−"); };
function drawSteps() {
  const eta = +$("#tE").value, q = 1 - 2 * eta;
  $("#tEV").textContent = fmt(eta, 2);
  const ws = [0]; for (let t = 0; t < 6; t++) ws.push(ws[t] - eta * 2 * (ws[t] - 3));
  // the notebook's table: steps 0 to 2 in full, then w_3 and L(w_3)
  const rows = [0, 1, 2].map(t => { const w = ws[t], g = 2 * (w - 3), st = -eta * g; return `<tr><td>${t}</td><td>${num(w)}</td><td>${num((w - 3) ** 2)}</td><td>${num(g)}</td><td>${st > 0 ? "+" : ""}${num(st)}</td><td>${num(ws[t + 1])}</td></tr>`; });
  rows.push(`<tr><td>3</td><td>${num(ws[3])}</td><td>${num((ws[3] - 3) ** 2)}</td><td></td><td></td><td></td></tr>`);
  $("#stepT tbody").innerHTML = rows.join("");
  const cv = $("#stepC"), shown = ws.slice(0, 6), lo = Math.min(-1, ...shown) - 0.5, hi = Math.max(7, ...shown) + 0.5, top = Math.max(16, ...shown.map(w => (w - 3) ** 2)) * 1.05;
  const fr = frame(cv, narrowOf(cv) ? 1.4 : 2.6, { x: [lo, hi], y: [0, top], xlabel: "w", ylabel: "L(w)" });
  const { ctx, X, Y } = fr, xs = range(201, i => lo + (hi - lo) * i / 200);
  plotLine(fr, xs, xs.map(w => (w - 3) ** 2), { color: css("--post"), width: 2.2 });
  ctx.strokeStyle = alpha("--warm", 0.8); ctx.lineWidth = 1.4;
  for (let i = 1; i < shown.length; i++) seg(ctx, X(shown[i - 1]), Y((shown[i - 1] - 3) ** 2), X(shown[i]), Y((shown[i] - 3) ** 2));
  shown.forEach((w, i) => { dot(ctx, X(w), Y((w - 3) ** 2), 4, css("--warm")); ctx.fillStyle = css("--ink-2"); ctx.font = "600 10px " + css("--font-ui"); ctx.textAlign = "center"; if (i < 4) ctx.fillText(`w${"₀₁₂₃"[i]}`, X(w), Y((w - 3) ** 2) - 9); });
  dot(ctx, X(3), Y(0), 5, css("--c4"));
  const dist = ws.slice(0, 4).map(w => num(Math.abs(w - 3))).join(", ");
  let text = `The factor is [[1 - 2\\eta = ${fmt(q, 2)}]], so the distance to the minimum goes ${dist}. `;
  if (Math.abs(q) < 1e-9) text += "<b>One step lands exactly on the minimum</b>, [[w = 3]].";
  else if (Math.abs(q + 1) < 1e-9) text += "<b>The edge</b>: the factor is −1, and [[w]] bounces between 0 and 6 forever.";
  else if (q > 0) text += q > 0.8 ? "Steady but slow: every step lands on the same side, a little closer." : "Steady: every step lands on the same side of the minimum, a fixed fraction closer.";
  else if (q > -1) text += "<b>Zig-zag</b>: each step jumps across the minimum, but closer each time.";
  else text += "<b>Explodes</b>: each step jumps across the minimum and lands further away than before.";
  caption($("#stepCap"), text);
}
$("#tE").addEventListener("input", drawSteps);
$$("#steps [data-te]").forEach(b => b.addEventListener("click", () => { $("#tE").value = b.dataset.te; drawSteps(); }));

/* =====================================================================
   PART 5: THE SECOND-DERIVATIVE TEST, LIVE
   ===================================================================== */
const FN = {
  cubic: { f: x => x ** 3 - 6 * x * x + 9 * x + 1, d1: x => 3 * x * x - 12 * x + 9, d2: x => 6 * x - 12, x: [-0.5, 4.5], y: [-6, 12], crit: [1, 3], tex: "x^3 - 6x^2 + 9x + 1", x0: 1, flat: [2] },
  x3: { f: x => x ** 3, d1: x => 3 * x * x, d2: x => 6 * x, x: [-1.5, 1.5], y: [-3.4, 3.4], crit: [0], tex: "x^3", x0: 0, flat: [0] },
  x4: { f: x => x ** 4, d1: x => 4 * x ** 3, d2: x => 12 * x * x, x: [-1.5, 1.5], y: [-1, 5], crit: [0], tex: "x^4", x0: 0, flat: [] },
  g: { f: x => x ** 4 - 4 * x * x, d1: x => 4 * x ** 3 - 8 * x, d2: x => 12 * x * x - 8, x: [-2.3, 2.3], y: [-5.5, 8], crit: [-Math.SQRT2, 0, Math.SQRT2], tex: "x^4 - 4x^2", x0: Math.SQRT2, flat: [] },
};
const tay = { fn: "cubic" };
function setFn(name) {
  tay.fn = name; const F = FN[name], s = $("#yX");
  s.min = String(F.x[0]); s.max = String(F.x[1]); s.value = String(F.x0);
  $$("#taylor [data-fn]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.fn === name)));
  drawTaylor();
}
function drawTaylor() {
  const F = FN[tay.fn]; let x0 = +$("#yX").value;
  const near = [...F.crit, ...F.flat].find(c => Math.abs(c - x0) < 0.04); if (near !== undefined) x0 = near;     // snap to the special points
  const f0 = F.f(x0), d1 = F.d1(x0), d2 = F.d2(x0), zero = v => Math.abs(v) < 1e-9, f3 = v => fmt(Math.abs(v) < 5e-4 ? 0 : v, 3);
  $("#yXV").textContent = Math.abs(Math.abs(x0) - Math.SQRT2) < 1e-12 ? (x0 < 0 ? "−√2" : "√2") : signed(x0, 2);
  const cv = $("#taylorC"), fr = frame(cv, narrowOf(cv) ? 1.3 : 2.4, { x: F.x, y: F.y, xlabel: "x" });
  const { ctx, X, Y } = fr, xs = range(241, i => F.x[0] + (F.x[1] - F.x[0]) * i / 240);
  plotLine(fr, xs, xs.map(F.f), { color: css("--post"), width: 2.4 });
  plotLine(fr, xs, xs.map(x => f0 + d1 * (x - x0) + 0.5 * d2 * (x - x0) ** 2), { color: css("--warm"), width: 2 });
  plotLine(fr, F.x, F.x.map(x => f0 + d1 * (x - x0)), { color: css("--ink"), width: 1.6, dash: [6, 4] });
  F.crit.forEach(c => dot(ctx, X(c), Y(F.f(c)), 3.5, alpha("--ink", 0.5)));
  dot(ctx, X(x0), Y(f0), 6, css("--surface"), css("--ink"), 2.4);
  const xs0 = $("#yXV").textContent, vals = `[[f(${xs0.replace("−", "-").replace("√2", "\\sqrt{2}")}) = ${f3(f0)}]], [[f' = ${f3(d1)}]], [[f'' = ${f3(d2)}]]`;
  let text = `For [[f(x) = ${F.tex}]], at [[x = ${xs0.replace("−", "-").replace("√2", "\\sqrt{2}")}]]: ${vals}. `;
  if (zero(d1)) {
    if (d2 > 1e-9) text += "<b>A critical point with [[f'' > 0]]: a local minimum.</b> The tangent is flat, and the Taylor parabola opens upward.";
    else if (d2 < -1e-9) text += "<b>A critical point with [[f'' < 0]]: a local maximum.</b> The tangent is flat, and the Taylor parabola opens downward.";
    else text += tay.fn === "x3" ? "<b>A critical point where the test is silent</b>: [[f'' = 0]] too, so the parabola is flat. The curve keeps rising on both sides: an inflection point, not an extremum." : "<b>A critical point where the test is silent</b>: [[f'' = 0]] too, so the parabola is flat. Yet every nearby point is higher: a minimum, decided by the fourth-order term.";
  } else {
    text += `Not a critical point: the slope is ${d1 > 0 ? "positive, so the curve is rising" : "negative, so the curve is falling"} here.`;
    if (zero(d2)) text += " [[f'' = 0]] here: an inflection point, where the curve switches from bending down to bending up.";
    else text += ` Slide to a point where the tangent is flat; the grey dots mark them.`;
  }
  caption($("#taylorCap"), text);
}
$("#yX").addEventListener("input", drawTaylor);
$$("#taylor [data-fn]").forEach(b => b.addEventListener("click", () => setFn(b.dataset.fn)));

/* =====================================================================
   start-up
   ===================================================================== */
function drawAll() { drawHero(); drawChord(); drawSurf(); drawSteps(); drawTaylor(); }
setEta(0.002); drawAll();
onRedraw(drawAll);
})();
