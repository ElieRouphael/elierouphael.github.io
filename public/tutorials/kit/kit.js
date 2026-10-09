/* Shared helpers for the interactive tutorial pages: math, code highlighting,
   canvas sizing, theme colours and redraws. Load after KaTeX and highlight.js,
   before the chapter's own script. */
(function (global) {
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t === "dark" || (t !== "light" && darkQuery.matches);
};
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Numbers: fmt(0.08757, 3) -> "0.088"; pct(0.08757) -> "8.8%" (more digits for small values). */
const fmt = (v, d = 2) => (v == null || !isFinite(v)) ? "–" : v.toFixed(d);
function pct(p, d) {
  if (p == null || !isFinite(p)) return "–";
  if (d == null) d = p === 0 || p === 1 ? 0 : p < 0.001 || p > 0.999 ? 2 : p < 0.1 || p > 0.9 ? 1 : 0;
  return (100 * p).toFixed(d) + "%";
}
const int = n => Math.round(n).toLocaleString("en-US");

/* ---------------- math and code ---------------- */
function renderTo(node, source) {
  try {
    katex.render(source, node, {
      displayMode: node.classList.contains("math-block"), throwOnError: false, strict: false,
    });
    return true;
  } catch (err) { return false; /* leave the TeX source visible */ }
}
function renderMath(root = document) {
  if (!global.katex) return;
  $$(".m, .math-block", root).forEach(node => {
    if (node.dataset.rendered) return;
    node.dataset.tex = node.textContent;
    if (renderTo(node, node.dataset.tex)) node.dataset.rendered = "1";
  });
  fitMath(root);
}
/* A display formula may carry a multi-line version in data-narrow. It is used only
   when the one-line version does not fit its box (phones, narrow boxes). */
function fitMath(root = document) {
  if (!global.katex) return;
  $$(".math-block[data-narrow]", root).forEach(node => {
    if (!node.dataset.rendered || !node.clientWidth) return;   // hidden, e.g. in a closed <details>
    if (node.dataset.shown !== "wide") { renderTo(node, node.dataset.tex); node.dataset.shown = "wide"; }
    if (node.scrollWidth > node.clientWidth + 1) { renderTo(node, node.dataset.narrow); node.dataset.shown = "narrow"; }
  });
}
document.addEventListener("toggle", ev => { if (ev.target.open) fitMath(ev.target); }, true);
/* Render TeX into an element from code (for readouts that change). */
function tex(el, source, display = false) {
  if (global.katex) katex.render(source, el, { displayMode: display, throwOnError: false, strict: false });
  else el.textContent = source;
}

/* Set an element's HTML; [[...]] marks TeX, rendered inline with KaTeX (fractions stacked,
   allowed to wrap). For captions that change as a widget is used. */
function caption(el, html) {
  const parts = [];
  el.innerHTML = html.replace(/\[\[(.+?)\]\]/g, (_, t) => { parts.push(t); return '<span class="cap-tex"></span>'; });
  $$(".cap-tex", el).forEach((span, i) => tex(span, parts[i]));
}

/* ---------------- canvas ---------------- */
/* Size a canvas to its CSS box times the device pixel ratio; draw in CSS pixels. */
function fit(canvas, aspect) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = aspect ? w / aspect : canvas.clientHeight;
  if (aspect) canvas.style.height = h + "px";
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

/* ---------------- charts ---------------- */
/* About `count` round tick values between lo and hi (steps of 1, 2 or 5 times a power of 10). */
function ticks(lo, hi, count = 5) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(12));
  return out;
}
const tickLabel = v => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) ? v.toExponential(0) : String(+v.toPrecision(6));

/* Size the canvas and draw axes for data in [x0, x1] x [y0, y1]. Returns the context and
   the maps X(x), Y(y) from data to CSS pixels. Options: pad, xticks, yticks (arrays, or
   false to hide), xfmt, yfmt, xlabel, ylabel, ygrid. */
function frame(canvas, aspect, o) {
  const { ctx, w, h } = fit(canvas, aspect);
  const pad = Object.assign({ l: o.yticks === false ? 10 : 42, r: 12, t: o.ylabel ? 24 : 12, b: o.xlabel ? 38 : 24 }, o.pad);
  const [x0, x1] = o.x, [y0, y1] = o.y;
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const X = v => pad.l + (v - x0) / (x1 - x0) * iw;
  const Y = v => pad.t + ih - (v - y0) / (y1 - y0) * ih;
  ctx.font = "11px " + css("--font-mono");
  ctx.lineWidth = 1;
  if (o.yticks !== false) {
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of o.yticks || ticks(y0, y1, 4)) {
      if (o.ygrid !== false) { ctx.strokeStyle = css("--grid"); ctx.beginPath(); ctx.moveTo(pad.l, Math.round(Y(t)) + 0.5); ctx.lineTo(w - pad.r, Math.round(Y(t)) + 0.5); ctx.stroke(); }
      ctx.fillStyle = css("--muted");
      ctx.fillText((o.yfmt || tickLabel)(t), pad.l - 6, Y(t));
    }
  }
  ctx.strokeStyle = css("--muted");
  ctx.beginPath(); ctx.moveTo(pad.l, Math.round(Y(y0)) + 0.5); ctx.lineTo(w - pad.r, Math.round(Y(y0)) + 0.5); ctx.stroke();
  if (o.xticks !== false) {
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = css("--muted");
    for (const t of o.xticks || ticks(x0, x1, Math.max(3, Math.floor(iw / 70)))) {
      ctx.beginPath(); ctx.moveTo(Math.round(X(t)) + 0.5, Y(y0)); ctx.lineTo(Math.round(X(t)) + 0.5, Y(y0) + 4); ctx.stroke();
      ctx.fillText((o.xfmt || tickLabel)(t), Math.min(w - pad.r - 8, Math.max(pad.l + 4, X(t))), Y(y0) + 6);
    }
  }
  if (o.xlabel) { ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText(o.xlabel, w - pad.r, h - 2); }
  if (o.ylabel) { ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(o.ylabel, 2, 0); }
  ctx.textBaseline = "alphabetic";
  return { ctx, w, h, X, Y, pad, iw, ih, x0, x1, y0, y1 };
}
/* Draw y = f(x) (or the points xs, ys) inside a frame, clipped to its box. */
function plotLine(fr, xs, ys, { color, width = 2, dash = null, fill = null } = {}) {
  const { ctx, X, Y, pad, iw, ih } = fr;
  ctx.save();
  ctx.beginPath(); ctx.rect(pad.l, pad.t - 2, iw, ih + 2); ctx.clip();
  ctx.beginPath();
  xs.forEach((x, i) => { const px = X(x), py = Y(ys[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
  if (fill) {
    ctx.lineTo(X(xs[xs.length - 1]), Y(fr.y0)); ctx.lineTo(X(xs[0]), Y(fr.y0)); ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();
    ctx.beginPath();
    xs.forEach((x, i) => { const px = X(x), py = Y(ys[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
  }
  if (color) { ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.setLineDash(dash || []); ctx.stroke(); }
  ctx.restore();
}
/* A colour with transparency, from a CSS colour token such as "--post". */
function alpha(token, a) {
  const c = css(token);
  if (c.startsWith("#") && c.length === 7) {
    const v = parseInt(c.slice(1), 16);
    return `rgba(${(v >> 16) & 255}, ${(v >> 8) & 255}, ${v & 255}, ${a})`;
  }
  return c;
}

/* ---------------- redraw on resize, theme change and font load ---------------- */
const drawers = [() => fitMath()];
function onRedraw(fn) { drawers.push(fn); }
function redrawAll() { drawers.forEach(fn => fn()); }
let resizeTimer = null;
new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(redrawAll, 60); }).observe(document.body);
darkQuery.addEventListener("change", redrawAll);
new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
if (document.fonts) document.fonts.ready.then(redrawAll);

renderMath();
window.addEventListener("load", () => renderMath());
if (global.hljs) $$("pre code").forEach(b => { if (!b.classList.contains("nohl")) hljs.highlightElement(b); });

global.Kit = { $, $$, css, isDark, reduceMotion, fmt, pct, int, renderMath, tex, caption, fit, ticks, tickLabel, frame, plotLine, alpha, onRedraw, redrawAll };
})(window);
