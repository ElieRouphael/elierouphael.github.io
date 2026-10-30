/* Kalman filters with JAX, chapter J4, vmap: the live demos.
   1. vmap(linear_kf_predict, in_axes=...) for every choice of batched arguments: JAX's own jaxprs.
   2. Average NEES and NIS over 1000 simulated runs, for a filter that assumes wrong noise levels.
   3. J3's maximum-likelihood noise estimate on 1000 datasets: a histogram, chapter 02's dataset marked.
   4. The timing table of batched filters.
   All numbers come from code/j4_numbers.py through j4-data.js. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, histogram } = window.Kit;
const D = window.J4;
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const grp = n => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

/* ============================================================ 1. the jaxprs */
const NAMES = ["x", "P", "F", "Q"], SHAPE1 = ["(2,)", "(2, 2)", "(2, 2)", "(2, 2)"], SHAPEB = ["(1000, 2)", "(1000, 2, 2)", "(1000, 2, 2)", "(1000, 2, 2)"];
const hero = { key: "b---" };
const eqCount = jx => (jx.match(/^ {4}[a-z_]+:/gm) || []).length;
const highlight = jx => esc(jx).replace(/([a-z_]+:f32\[1000[^\]]*\])/g, '<span class="b">$1</span>');

function drawHero() {
  const key = hero.key, c = D.hero.cases[key];
  $("#jxOne").innerHTML = esc(D.hero.one);
  $("#axesLabel").textContent = "(" + key.split("").map(ch => ch === "b" ? "0" : "None").join(", ") + ")";
  $$("#axToggles [data-arg]").forEach(b => b.setAttribute("aria-pressed", key[+b.dataset.arg] === "b"));
  $$("#axPresets [data-p]").forEach(b => b.setAttribute("aria-pressed", b.dataset.p === key));
  $("#rIn").textContent = NAMES.map((n, i) => `${n} ${key[i] === "b" ? SHAPEB[i] : SHAPE1[i]}`).join(", ");
  if (c.error) {
    $("#jxBatch").innerHTML = `<span class="err">${esc(c.error)}</span>`;
    $("#rOut").textContent = "–"; $("#rOps").textContent = "–"; $("#rLoops").textContent = "–";
    caption($("#heroStatus"), "<b>Nothing is batched, so there is nothing to map over.</b> <code>vmap</code> refuses: it needs at least one argument with a batch axis to know the batch size (or an explicit <code>axis_size</code>).");
    return;
  }
  $("#jxBatch").innerHTML = highlight(c.jaxpr);
  $("#rOut").textContent = `x_pred (${c.out[0].join(", ")}), P_pred (${c.out[1].join(", ")})`;
  $("#rOps").textContent = `${eqCount(D.hero.one)} → ${eqCount(c.jaxpr)}`;
  $("#rLoops").textContent = /\b(while|scan)\[/.test(c.jaxpr) ? "yes" : "none: one pass over the whole batch";

  const notes = [];
  const batchedDots = (c.jaxpr.match(/\(\[0\], \[0\]\)\)/g) || []).length;
  const xB = key[0] === "b" || key[2] === "b";                       // x_pred = F @ x
  const pB = key[1] === "b" || key[2] === "b" || key[3] === "b";     // P_pred = F @ P @ F.T + Q
  if (batchedDots) notes.push(`${batchedDots === 1 ? "One product has" : `${batchedDots} products have`} both operands batched, and became a <code>dot_general</code> with batch dimensions <code>([0], [0])</code>: 1000 independent 2 × 2 products in one operation.`);
  if (key[0] === "b" && key[2] === "-") notes.push("<code>F @ x</code> with one shared <code>F</code> and 1000 states became a single matrix product with the states as columns, then a transpose to put the batch axis back in front.");
  if (!xB) notes.push("<code>x_pred = F @ x</code> uses only unbatched arguments, so it is computed once; because every output gets a batch axis by default, it is then broadcast to 1000 identical copies. <code>out_axes=(None, 0)</code> would return it once.");
  if (!pB) notes.push("<code>P_pred</code> uses only unbatched arguments, so it is computed once; because every output gets a batch axis by default, it is then broadcast to 1000 identical copies. <code>out_axes=(0, None)</code> would return it once.");
  if (/f32\[1,2,2\] = broadcast_in_dim/.test(c.jaxpr)) notes.push("Where an unbatched value meets a batched one, in the addition of <code>Q</code>, the unbatched one is broadcast across the batch first.");
  if (key[2] === "b" && key.slice(0, 2) !== "bb") notes.push("Batching <code>F</code> turns its transpose into a batched transpose too. With 1000 copies of one model, this does 1000 times the work of sharing it: use <code>None</code> where an argument is the same for every element.");
  if (key === "bbbb") notes.push("Everything is batched: the program has the same five operations as for one input, each on 1000 at once.");
  if (!notes.length) notes.push("Each operation on a batched value became the same operation with a batch axis; operations on shared values run once.");
  caption($("#heroStatus"), notes.join(" "));
}
$$("#axToggles [data-arg]").forEach(b => b.addEventListener("click", () => {
  const i = +b.dataset.arg, k = hero.key.split("");
  k[i] = k[i] === "b" ? "-" : "b"; hero.key = k.join(""); drawHero();
}));
$$("#axPresets [data-p]").forEach(b => b.addEventListener("click", () => { hero.key = b.dataset.p; drawHero(); }));
drawHero();

/* ============================================================ 2. NEES and NIS */
const ne = { fa: 1, fz: 1 };
const facLabel = f => f === 1 ? "true" : `×${f}`;
$("#faBtns").innerHTML = `<span class="hint" style="align-self:center">process noise σa assumed</span>` +
  D.FA.map(f => `<button class="btn ghost" type="button" data-fa="${f}">${facLabel(f)}</button>`).join("");
$("#fzBtns").innerHTML = `<span class="hint" style="align-self:center">measurement noise σz assumed</span>` +
  D.FZ.map(f => `<button class="btn ghost" type="button" data-fz="${f}">${facLabel(f)}</button>`).join("");
$("#neesLegend").innerHTML = `<span><i style="background:var(--accent)"></i>average NEES</span><span><i class="box" style="background:var(--accent)"></i>its 95% band</span>` +
  `<span><i style="background:var(--warm)"></i>average NIS</span><span><i class="box" style="background:var(--warm)"></i>its 95% band</span>`;
const fkey = f => (Number.isInteger(f) ? f.toFixed(1) : String(f));
/* For a consistent filter the innovations are independent, so the NIS averaged over all T steps of all runs
   has mean 1 and standard deviation sqrt(2 / (T * runs)): a 95% band of 1 +/- 1.96 of those. */
const nisAllBand = n => [1 - 1.96 * Math.sqrt(2 / n), 1 + 1.96 * Math.sqrt(2 / n)];
function drawNees() {
  $$("#faBtns [data-fa]").forEach(b => b.setAttribute("aria-pressed", +b.dataset.fa === ne.fa));
  $$("#fzBtns [data-fz]").forEach(b => b.setAttribute("aria-pressed", +b.dataset.fz === ne.fz));
  const r = D.nees[`${fkey(ne.fa)}_${fkey(ne.fz)}`];
  const cv = $("#neesC");
  const L = v => Math.log2(Math.max(v, 0.125));
  const fr = frame(cv, narrowOf(cv) ? 1.2 : 2.3, {
    x: [1, r.anees.length], y: [L(0.2), L(32)], yticks: [-2, -1, 0, 1, 2, 3, 4, 5].map(v => v),
    yfmt: v => String(+(2 ** v).toPrecision(2)), xlabel: "filter step", ygrid: true,
  });
  const { ctx, X, Y } = fr;
  const band = (lo, hi, tok) => { ctx.fillStyle = alpha(tok, 0.16); ctx.fillRect(X(1), Y(L(hi)), X(r.anees.length) - X(1), Y(L(lo)) - Y(L(hi))); };
  band(D.band_nees[0], D.band_nees[1], "--accent");
  band(D.band_nis[0], D.band_nis[1], "--warm");
  const xs = r.anees.map((_, i) => i + 1);
  plotLine(fr, xs, r.anis.map(L), { color: css("--warm"), width: 1.6 });
  plotLine(fr, xs, r.anees.map(L), { color: css("--accent"), width: 1.8 });
  const T = r.anees.length;
  const inside = r.anees.filter(v => v >= D.band_nees[0] && v <= D.band_nees[1]).length;
  const insideI = r.anis.filter(v => v >= D.band_nis[0] && v <= D.band_nis[1]).length;
  const [alo, ahi] = nisAllBand(T * 1000);
  $("#nE").textContent = `${r.anees_all.toFixed(3)} (expect 2)`;
  $("#nI").textContent = `${r.anis_all.toFixed(3)} (expect 1; for ${grp(T * 1000)} innovations, 95% between ${alo.toFixed(3)} and ${ahi.toFixed(3)})`;
  $("#nIn").textContent = `${inside} of ${T}`;
  $("#nInI").textContent = `${insideI} of ${T}`;
  const vE = inside >= 0.9 * T ? `the NEES is inside its band at ${inside} of ${T} steps: consistent`
    : `the NEES is ${r.anees_all > 2 ? "too high, so the filter is overconfident" : "too low, so the filter is too cautious"} (inside its band at only ${inside} of ${T} steps)`;
  const vI = r.anis_all > ahi ? `the NIS averaged over all ${grp(T * 1000)} innovations is too high`
    : r.anis_all < alo ? `the NIS averaged over all ${grp(T * 1000)} innovations is too low`
    : `the NIS averaged over all innovations is consistent`;
  let msg = `The filter assumes [[\\sigma_a]] ${ne.fa === 1 ? "equal to the true 0.1" : `${ne.fa} times the true value`} and [[\\sigma_z]] ${ne.fz === 1 ? "equal to the true 0.5" : `${ne.fz} times the true value`}. Step by step, ${vE}, and the NIS is inside its band at ${insideI} of ${T} steps; overall, ${vI}. `;
  if (ne.fa === 1 && ne.fz === 1) msg += "Both curves stay in their bands apart from the occasional step, about 5% of them, as a 95% band should allow.";
  else if (ne.fz === 1 && ne.fa > 1) msg += "Step by step the NIS barely moves: the innovations are dominated by the measurement noise, so an overstated process noise hides inside the per-step band. Only the average over a great many innovations gives it away, while the NEES, which needs the true state, sees it at every step.";
  else if (ne.fz === 1 && ne.fa < 1) msg += "Understating the process noise makes the filter trust its model too much; the NEES sees it at once. The NIS rises only a little.";
  else msg += "A wrong measurement noise moves both tests: the innovations are mostly measurement noise, so the NIS is a good test of [[\\sigma_z]].";
  caption($("#neesCap"), msg);
}
$$("#faBtns [data-fa]").forEach(b => b.addEventListener("click", () => { ne.fa = +b.dataset.fa; drawNees(); }));
$$("#fzBtns [data-fz]").forEach(b => b.addEventListener("click", () => { ne.fz = +b.dataset.fz; drawNees(); }));
drawNees();
onRedraw(drawNees);

/* ============================================================ 3. the estimates on 1000 datasets */
const FI = D.fits;
const fitS = { p: "sa" };
const PAR = {
  sa: { vals: FI.sa, truth: 0.1, k42: FI.key42[0], lo: 0.03, hi: 0.17, bins: 35, med: FI.sa_median, q05: FI.sa_q05, q95: FI.sa_q95, name: "σa", tex: "\\hat\\sigma_a" },
  sz: { vals: FI.sz, truth: 0.5, k42: FI.key42[1], lo: 0.46, hi: 0.54, bins: 32, med: FI.sz_median, q05: FI.sz_q05, q95: FI.sz_q95, name: "σz", tex: "\\hat\\sigma_z" },
};
$("#fitLegend").innerHTML = `<span><i style="background:var(--accent)"></i>the 1000 estimates</span><span><i style="background:var(--ink)"></i>the true value</span><span><i style="background:var(--warm)"></i>chapter 02's dataset (key 42)</span>`;
function drawFit() {
  const P = PAR[fitS.p];
  $$("#fitSeg [data-p]").forEach(b => b.setAttribute("aria-pressed", b.dataset.p === fitS.p));
  const w = (P.hi - P.lo) / P.bins;
  const counts = new Array(P.bins).fill(0);
  P.vals.forEach(v => { const i = Math.floor((v - P.lo) / w); if (i >= 0 && i < P.bins) counts[i]++; });
  const cv = $("#fitC");
  const fr = frame(cv, narrowOf(cv) ? 1.3 : 2.4, { x: [P.lo, P.hi], y: [0, Math.max(...counts) * 1.12], xlabel: P.name, ylabel: "datasets" });
  histogram(fr, P.vals, P.lo, P.hi, P.bins, 1, alpha("--accent", 0.75));
  const { ctx, X, Y } = fr;
  ctx.lineWidth = 2; ctx.strokeStyle = css("--ink"); ctx.setLineDash([5, 4]);
  seg(ctx, X(P.truth), Y(fr.y1), X(P.truth), Y(0)); ctx.setLineDash([]);
  ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2.4;
  seg(ctx, X(P.k42), Y(fr.y1), X(P.k42), Y(0));
  const rank = fitS.p === "sa" ? FI.sa_at_or_below_key42 : FI.sz_at_or_above_key42;
  $("#fMed").textContent = P.med.toFixed(4);
  $("#fRange").textContent = `${P.q05.toFixed(3)} and ${P.q95.toFixed(3)}`;
  $("#f42").textContent = `${P.k42.toFixed(4)}: ${fitS.p === "sa" ? `${rank} of 1000 are this low or lower` : `${rank} of 1000 are this high or higher`}`;
  const msg = fitS.p === "sa"
    ? `The estimates centre on the true 0.1 (median ${P.med.toFixed(3)}) and spread over about ±25%. Chapter 02's dataset, at ${P.k42.toFixed(3)}, is in the far left tail: only ${rank} of the 1000 datasets give an estimate as low. J3's method was not biased; its one dataset was unusual.`
    : `[[${P.tex}]] is pinned down far more tightly, within about ±4% of the true 0.5. Chapter 02's ${P.k42.toFixed(3)} is unremarkable: ${rank} of 1000 datasets give an estimate at least as high. The measurement noise shows in every innovation; the small process noise must be inferred from slow drifts.`;
  caption($("#fitCap"), msg);
}
$$("#fitSeg [data-p]").forEach(b => b.addEventListener("click", () => { fitS.p = b.dataset.p; drawFit(); }));
drawFit();
onRedraw(drawFit);

/* ============================================================ 4. the timing table */
const LO = Math.log10(0.3), HI = Math.log10(1000);
const bw = v => `${(100 * Math.max(0, Math.min(1, (Math.log10(v) - LO) / (HI - LO)))).toFixed(1)}%`;
const ms = v => v >= 100 ? `${v.toFixed(0)} ms` : v >= 10 ? `${v.toFixed(1)} ms` : `${v.toFixed(2)} ms`;
$("#timeBody").innerHTML = D.timing.map(r => `<tr><td class="n">${grp(r.B)}</td><td class="n">${r.loop_ms ? ms(r.loop_ms) : "not run"}</td><td class="n">${ms(r.vmap_ms)}</td>` +
  `<td class="n">${r.loop_ms ? `${(r.loop_ms / r.vmap_ms).toFixed(r.loop_ms / r.vmap_ms < 10 ? 1 : 0)}×` : "–"}</td>` +
  `<td class="barc">${r.loop_ms ? `<div class="tb loop" style="width:${bw(r.loop_ms)}"></div>` : ""}<div class="tb vm" style="width:${bw(r.vmap_ms)}"></div></td></tr>`).join("");
})();
