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

global.Kit = { $, $$, css, isDark, reduceMotion, fmt, pct, int, renderMath, tex, fit, onRedraw, redrawAll };
})(window);
