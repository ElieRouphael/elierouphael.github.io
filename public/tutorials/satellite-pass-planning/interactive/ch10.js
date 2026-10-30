/* Satellite Pass Planning from Scratch, chapter 10, Rolling Horizons and Reading a Benchmark: the live demos.
   (1) the hero: the rolling-horizon plans of the small instance (blocks of 30 min to the whole 6 h), block by
       block, against the whole-horizon optimum shipped with the course; every plan is re-checked with sat.js;
   (2) the four cases of a new task against a fixed one (rolling.py), on an illustrative timeline;
   (3) the block-size sweep on small and medium (value and time);
   (4) the benchmark's gaps as ranges: against the best proven bound and against the best known plan;
   (5) the memory sweep of results.md with the measured run-to-run noise.
   Needs ../kit/kit.js, sat.js, instance-small.js and ch10-data.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, int, ticks, caption, frame, plotLine, alpha, onRedraw, narrowOf, fit } = window.Kit;
const Sat = window.Sat, I = window.SAT_SMALL, D = window.SAT_CH10;

const dot = (ctx, x, y, r, color) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); };
const sum = a => a.reduce((s, v) => s + v, 0);
const blockLabel = r => r.n_blocks === 1 ? "whole horizon" : `${r.block_min} min`;

/* ---------------- (1) the hero: rolling plans on the small instance ---------------- */
const OPT = new Map(Object.entries(I.python.cpsat.plan).map(([k, v]) => [+k, v]));
const optTargets = new Set([...OPT.keys()].map(i => I.opps[i].target));
const rowsS = D.small.rows;
const plans = rowsS.map(r => new Map(Object.entries(r.plan).map(([k, v]) => [+k, v])));
const checks = plans.map(p => ({ errors: Sat.checkPlan(I, p).length, value: Sat.planValue(I, p) }));
const optCheck = { errors: Sat.checkPlan(I, OPT).length, value: Sat.planValue(I, OPT) };
const hs = { r: 1, k: rowsS[1].n_blocks };   // 60-minute blocks, all solved
function blockOf(o, bs, nb) { return Math.min(Math.floor(o.ws / bs), nb - 1); }

function drawHero() {
  const row = rowsS[hs.r], plan = plans[hs.r], bs = row.block_min * 60, nb = row.n_blocks;
  const rollTargets = new Set([...plan.keys()].map(i => I.opps[i].target));
  const cv = $("#heroC"), narrow = narrowOf(cv, 560);
  const lane = narrow ? 9 : 12, gapS = narrow ? 10 : 14, left = narrow ? 38 : 50, top = 8, bottom = 30;
  const w = cv.clientWidth || cv.parentElement.clientWidth;
  const h = top + I.nSats * (2 * lane + 3 + gapS) + bottom;
  const { ctx } = fit(cv, w / h);
  const right = w - 8, X = t => left + t / I.horizon * (right - left);
  const laneY = (s, which) => top + s * (2 * lane + 3 + gapS) + (which ? lane + 3 : 0);
  ctx.font = "11px " + css("--font-mono");
  // blocks: done (plain), current (tinted), future (greyed on the rolling lanes)
  for (let b = 0; b < nb; b++) {
    const x0 = X(b * bs), x1 = X(Math.min(I.horizon, (b + 1) * bs));
    if (b === hs.k - 1) { ctx.fillStyle = alpha("--c4", 0.10); ctx.fillRect(x0, top - 4, x1 - x0, h - top - bottom + 2); }
    if (b >= hs.k) { ctx.fillStyle = alpha("--muted", 0.10); for (let s = 0; s < I.nSats; s++) ctx.fillRect(x0, laneY(s, 1), x1 - x0, lane); }
    if (b > 0) { ctx.strokeStyle = alpha("--ink", 0.55); ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(Math.round(x0) + 0.5, top - 4); ctx.lineTo(Math.round(x0) + 0.5, h - bottom + 2); ctx.stroke(); ctx.setLineDash([]); }
  }
  // satellite labels and lane frames
  for (let s = 0; s < I.nSats; s++) {
    ctx.fillStyle = css("--ink-2"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(`sat ${s}`, left - 6, laneY(s, 0) + lane + 1);
    ctx.strokeStyle = css("--grid");
    ctx.strokeRect(left + 0.5, laneY(s, 0) + 0.5, right - left, lane);
    ctx.strokeRect(left + 0.5, laneY(s, 1) + 0.5, right - left, lane);
  }
  const drawTask = (i, t, which, other) => {
    const o = I.opps[i], only = !other.has(o.target);
    ctx.fillStyle = only ? css(which ? "--c4" : "--warm") : alpha("--ink", 0.55);
    const x0 = X(t), x1 = X(t + o.dur);
    ctx.fillRect(x0, laneY(o.sat, which) + 1, Math.max(only ? 3 : 2, x1 - x0), lane - 2);
  };
  for (const [i, t] of OPT) drawTask(i, t, 0, rollTargets);
  for (const [i, t] of plan) if (blockOf(I.opps[i], bs, nb) < hs.k) drawTask(i, t, 1, optTargets);
  // time axis
  ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let hr = 0; hr <= I.horizon / 3600; hr++) { const x = X(hr * 3600); ctx.fillRect(Math.round(x), h - bottom + 2, 1, 4); ctx.fillText(`${hr} h`, Math.min(right - 10, Math.max(left + 8, x)), h - bottom + 8); }
  ctx.textBaseline = "alphabetic";

  // the per-block difference chart
  const cv2 = $("#heroD");
  const diff = row.per_block.map((v, b) => v - row.opt_per_block[b]);
  const cum = []; diff.reduce((s, v, b) => (cum[b] = s + v), 0);
  const lim = Math.max(8, ...diff.map(Math.abs), ...cum.map(Math.abs)) + 2;
  const fr = frame(cv2, narrow ? 2.6 : 5.2, { x: [0, I.horizon], y: [-lim, lim], xticks: [0, 3600, 7200, 10800, 14400, 18000, 21600], xfmt: v => `${v / 3600} h`, ylabel: "rolling minus optimum, per block", pad: { l: left } });
  fr.ctx.strokeStyle = css("--muted"); fr.ctx.beginPath(); fr.ctx.moveTo(fr.X(0), fr.Y(0) + 0.5); fr.ctx.lineTo(fr.X(I.horizon), fr.Y(0) + 0.5); fr.ctx.stroke();
  diff.forEach((v, b) => {
    const x0 = fr.X(b * bs) + 2, x1 = fr.X(Math.min(I.horizon, (b + 1) * bs)) - 2;
    fr.ctx.fillStyle = b < hs.k ? css(v >= 0 ? "--c4" : "--warm") : alpha("--muted", 0.3);
    fr.ctx.fillRect(x0, Math.min(fr.Y(0), fr.Y(v)), Math.max(1, x1 - x0), Math.abs(fr.Y(v) - fr.Y(0)));
  });
  const kx = [0], ky = [0];
  for (let b = 0; b < hs.k; b++) { kx.push(Math.min(I.horizon, (b + 1) * bs)); ky.push(cum[b]); }
  plotLine(fr, kx, ky, { color: css("--ink"), width: 1.6 });
  ky.forEach((v, j) => dot(fr.ctx, fr.X(kx[j]), fr.Y(v), 2.5, css("--ink")));

  // readouts and caption
  const c = checks[hs.r];
  const rollSoFar = sum(row.per_block.slice(0, hs.k)), optSoFar = sum(row.opt_per_block.slice(0, hs.k));
  $("#hBlockV").textContent = blockLabel(row);
  $("#hStepV").textContent = `${hs.k} of ${nb}`;
  $("#hStep").max = nb; $("#hStep").value = hs.k;
  $("#rBlocks").textContent = `${nb} × ${fmt(row.per_block_limit_s, 2)} s`;
  $("#rVal").textContent = `${row.value} / ${D.small.opt}`;
  $("#rRep").textContent = (row.values || [row.value]).join(", ");
  $("#rSoFar").textContent = `${rollSoFar} / ${optSoFar}`;
  $("#rTime").textContent = `${fmt(row.time_s, 2)} s`;
  $("#rCheck").textContent = c.errors === 0 && c.value === row.value && optCheck.errors === 0 ? "both feasible" : `${c.errors} violations`;
  const first = diff[0], lastNeg = diff.reduce((m, v) => Math.min(m, v), 0);
  const whole = nb === 1;
  const done = hs.k === nb;
  const lostT = row.only_opt.length, gainT = row.only_roll.length;
  caption($("#heroStatus"), `Top lane of each satellite: the whole-horizon optimum of the small instance (219, from instance-small.js); bottom lane: the rolling-horizon plan with ${whole ? "a single block, i.e. the whole-horizon CP-SAT model without a hint" : `${row.block_min}-minute blocks (dashed borders)`}, computed by code/ch10_numbers.py with a total time limit of ${D.limits.small} s (first of two runs with identical settings; the second gave ${(row.values || [row.value]).slice(1).join(", ") || "the same"}). Grey tasks collect a target both plans collect; orange, a target only the optimum collects; green, one only the rolling plan collects. ${whole ? "" : `After ${hs.k} of ${nb} blocks the rolling plan holds ${rollSoFar} points against ${optSoFar} for the optimum in the same blocks. `}${done ? `In total it collects <b>${row.value}</b> against 219: ${row.value === 219 ? "no loss on this instance" : `a loss of ${219 - row.value}`}, with ${lostT} targets only in the optimum and ${gainT} only in the rolling plan.` : "Move the slider to the last block to see the end."} ${!whole && first > 0 ? `The bars below show the pattern of a myopic decomposition: the first block takes ${first} more points than the optimum does there, because nothing tells it that the budgets and satellite time it uses will be wanted later; the bill arrives in later blocks${lastNeg < 0 ? `, up to ${-lastNeg} points in one block` : ""}.` : ""} ${c.errors === 0 && optCheck.errors === 0 ? "Both plans pass the course's checker in your browser (sat.js)." : `The browser's checker reports ${c.errors + optCheck.errors} violations: please report this.`}`);
}
$$("#hBlockSeg .btn").forEach((btn, j) => btn.addEventListener("click", () => {
  hs.r = j; hs.k = rowsS[j].n_blocks;
  $$("#hBlockSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === btn));
  drawHero();
}));
$("#hStep").addEventListener("input", e => { hs.k = +e.target.value; drawHero(); });

/* ---------------- (2) the four cases of a new task against a fixed one ---------------- */
const fc = { tf: 600, df: 60, sfi: 40, sif: 40, di: 45, e: 450, w: 300 };
function drawCases() {
  const cv = $("#caseC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 2.0 : 4.2, { x: [250, 1050], y: [0, 3], yticks: false, xlabel: "time [s]", xticks: [300, 400, 500, 600, 700, 800, 900, 1000] });
  const { ctx, X, Y } = fr;
  const l = fc.e + fc.w;
  const after = fc.tf + fc.df + fc.sfi, before = fc.tf - fc.di - fc.sif;
  // fixed task and its reconfiguration margins
  ctx.fillStyle = alpha("--ink", 0.75); ctx.fillRect(X(fc.tf), Y(2.7), X(fc.tf + fc.df) - X(fc.tf), Y(2.1) - Y(2.7));
  ctx.fillStyle = alpha("--ink", 0.18); ctx.fillRect(X(fc.tf + fc.df), Y(2.7), X(after) - X(fc.tf + fc.df), Y(2.1) - Y(2.7));
  ctx.fillRect(X(before + fc.di), Y(2.7), X(fc.tf) - X(before + fc.di), Y(2.1) - Y(2.7));
  ctx.font = "11px " + css("--font-mono"); ctx.fillStyle = css("--ink-2"); ctx.textAlign = "left";
  ctx.fillText("fixed task f", X(fc.tf), Y(2.78));
  // forbidden starts of i: (before, after)
  ctx.fillStyle = alpha("--warm", 0.25); ctx.fillRect(X(before), Y(1.75), X(after) - X(before), Y(0.35) - Y(1.75));
  ctx.strokeStyle = css("--warm"); ctx.setLineDash([4, 3]);
  for (const v of [before, after]) { ctx.beginPath(); ctx.moveTo(X(v) + 0.5, Y(2.0)); ctx.lineTo(X(v) + 0.5, Y(0.2)); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.fillStyle = css("--warm"); ctx.textAlign = "center";
  ctx.fillText(`before = ${before}`, X(before) - 6, Y(0.12));
  ctx.fillText(`after = ${after}`, X(after) + 6, Y(0.12));
  // the window of possible starts of i, and the part that survives
  ctx.fillStyle = alpha("--post", 0.25); ctx.fillRect(X(fc.e), Y(1.4), Math.max(2, X(l) - X(fc.e)), Y(0.7) - Y(1.4));
  ctx.fillStyle = css("--post");
  const keep = [[fc.e, Math.min(l, before)], [Math.max(fc.e, after), l]];
  for (const [a, b] of keep) if (b >= a) ctx.fillRect(X(a), Y(1.3), Math.max(2, X(b) - X(a)), Y(0.8) - Y(1.3));
  ctx.fillStyle = css("--ink-2"); ctx.textAlign = "left";
  ctx.fillText(`possible starts of i: [${fc.e}, ${l}]`, X(Math.min(fc.e, 800)), Y(1.5));
  // the case, exactly as rolling.py decides it
  const afterOk = after <= l, beforeOk = before >= fc.e;
  let kase, code;
  if (after <= fc.e || before >= l) { kase = "no interference whatever the start"; code = "continue"; }
  else if (!afterOk && !beforeOk) { kase = "i cannot be selected"; code = "m.add(x[i] == 0)"; }
  else if (afterOk && !beforeOk) { kase = "i must start after f"; code = "m.add(s[i] >= after).only_enforce_if(x[i])"; }
  else if (beforeOk && !afterOk) { kase = "i must end before f"; code = "m.add(s[i] <= before).only_enforce_if(x[i])"; }
  else { kase = "either side: a new order literal"; code = "o = m.new_bool_var(\"\") and two enforced rows"; }
  $("#cEV").textContent = `${fc.e} s`; $("#cWV").textContent = `${fc.w} s`;
  $("#cCase").textContent = kase; $("#cCode").textContent = code;
  caption($("#caseCap"), `An illustrative fixed task f (black) starts at ${fc.tf} s and lasts ${fc.df} s; a new task i of ${fc.di} s needs ${fc.sif} s of reconfiguration before f and ${fc.sfi} s after it (grey margins). So i must start at or before [[\\text{before} = t_f - d_i - s(i, f) = ${before}]] or at or after [[\\text{after} = t_f + d_f + s(f, i) = ${after}]]: the orange interval is forbidden. The light blue bar is the window of possible starts of i, [${fc.e}, ${l}]; the dark blue parts survive. Here [[\\text{after} \\le \\ell_i]] is ${afterOk} and [[\\text{before} \\ge e_i]] is ${beforeOk}, so rolling.py's case is: <b>${kase}</b>.`);
}
$("#cE").addEventListener("input", e => { fc.e = +e.target.value; drawCases(); });
$("#cW").addEventListener("input", e => { fc.w = +e.target.value; drawCases(); });

/* ---------------- (3) the block-size sweep ---------------- */
function drawSweep() {
  const cats = rowsS.map(blockLabel).map(s => s === "whole horizon" ? "whole" : s.replace(" min", ""));
  const refS = D.small.opt, refM = D.best_bound_medium;
  const vS = rowsS.map(r => 100 * r.value / refS), vM = D.medium.rows.map(r => 100 * r.value / refM);
  const cv = $("#swV"), narrow = narrowOf(cv, 360);
  const lo = Math.floor(Math.min(...vS, ...vM) - 1), hi = 100.5;
  const fr = frame(cv, narrow ? 1.25 : 1.35, { x: [-0.5, cats.length - 0.5], y: [lo, hi], xticks: cats.map((_, i) => i), xfmt: i => cats[i], yfmt: v => v + "%", xlabel: "block length [min]", ylabel: "value / best bound" });
  const xs = cats.map((_, i) => i);
  plotLine(fr, xs, vS, { color: css("--post"), width: 2 }); plotLine(fr, xs, vM, { color: css("--warm"), width: 2 });
  fr.ctx.font = "11px " + css("--font-mono"); fr.ctx.textAlign = "center";
  xs.forEach(i => {
    dot(fr.ctx, fr.X(i), fr.Y(vS[i]), 4, css("--post")); dot(fr.ctx, fr.X(i), fr.Y(vM[i]), 4, css("--warm"));
    fr.ctx.fillStyle = css("--post"); fr.ctx.fillText(String(rowsS[i].value), fr.X(i), fr.Y(vS[i]) - 8);
    fr.ctx.fillStyle = css("--warm"); fr.ctx.fillText(String(D.medium.rows[i].value), fr.X(i), fr.Y(vM[i]) + 16);
    // the repeated runs, same settings: hollow rings
    for (const [rowsX, ref, col] of [[rowsS, refS, "--post"], [D.medium.rows, refM, "--warm"]]) {
      (rowsX[i].values || []).slice(1).forEach(v => { fr.ctx.strokeStyle = css(col); fr.ctx.lineWidth = 1.5; fr.ctx.beginPath(); fr.ctx.arc(fr.X(i), fr.Y(100 * v / ref), 6, 0, 2 * Math.PI); fr.ctx.stroke(); fr.ctx.lineWidth = 1; });
    }
  });
  const cv2 = $("#swT");
  const lt = v => Math.log10(Math.max(0.05, v));
  const fr2 = frame(cv2, narrow ? 1.25 : 1.35, { x: [-0.5, cats.length - 0.5], y: [-1.5, 2], xticks: cats.map((_, i) => i), xfmt: i => cats[i], yticks: [-1, 0, 1, 2], yfmt: v => String(10 ** v >= 1 ? 10 ** v : (10 ** v).toFixed(1)), xlabel: "block length [min]", ylabel: "solve time [s], log" });
  const tS = rowsS.map(r => lt(r.time_s)), tM = D.medium.rows.map(r => lt(r.time_s));
  plotLine(fr2, xs, tS, { color: css("--post"), width: 2 }); plotLine(fr2, xs, tM, { color: css("--warm"), width: 2 });
  xs.forEach(i => { dot(fr2.ctx, fr2.X(i), fr2.Y(tS[i]), 4, css("--post")); dot(fr2.ctx, fr2.X(i), fr2.Y(tM[i]), 4, css("--warm")); });
  for (const [lim, col] of [[D.limits.small, "--post"], [D.limits.medium, "--warm"]]) { fr2.ctx.strokeStyle = alpha(col, 0.7); fr2.ctx.setLineDash([4, 3]); fr2.ctx.beginPath(); fr2.ctx.moveTo(fr2.X(-0.5), fr2.Y(lt(lim))); fr2.ctx.lineTo(fr2.X(cats.length - 0.5), fr2.Y(lt(lim))); fr2.ctx.stroke(); fr2.ctx.setLineDash([]); }
  const m = D.medium.rows, best = m.reduce((a, r) => r.value > a.value ? r : a, m[0]);
  const bestRoll = m.slice(0, -1).reduce((a, r) => r.value > a.value ? r : a, m[0]);
  let nonMono = null;
  for (let i = 1; i < m.length - 1; i++) if (m[i].value < m[i - 1].value) { nonMono = { a: m[i - 1], b: m[i], allOpt: m[i].time_s < m[i].per_block_limit_s && m[i - 1].time_s < m[i - 1].per_block_limit_s }; break; }
  caption($("#swCap"), `Rolling-horizon CP-SAT on seed 0 of the small instance (blue, total limit ${D.limits.small} s, values over the optimum 219) and the medium instance (orange, total limit ${D.limits.medium} s, values over the best bound proved anywhere for it, ${refM}), for blocks of 30, 60, 120 and 180 minutes and a single block covering the whole horizon. Left: value of the first run (filled dots, labelled) and of a second run with identical settings (rings); right: wall time of the first run, with the time limits dashed. On small the values are ${rowsS.map(r => r.value).join(", ")} for the five block lengths: short blocks lose a few points, long ones ${rowsS[3].value === refS ? "lose nothing" : "come close"}, and every run takes under a second. On medium they are ${m.map(r => r.value).join(", ")}; the best rolling plan uses ${bestRoll.block_min}-minute blocks (${bestRoll.value} in ${fmt(bestRoll.time_s, 1)} s), and the single block ${best === m[m.length - 1] ? "is best of all" : "reaches"} ${m[m.length - 1].value} but needs the full ${fmt(m[m.length - 1].time_s, 0)} s. ${nonMono ? `The curve is not monotone: ${nonMono.b.block_min}-minute blocks (${nonMono.b.value}) do worse than ${nonMono.a.block_min}-minute ones (${nonMono.a.value})${nonMono.allOpt ? ", although every block was solved to optimality: the borders fall elsewhere, and optimal block plans are not unique" : ": the blocks are larger and slower to solve, and the borders fall elsewhere"}.` : "Here the value grows with the block length, as the simplest argument predicts."}`);
}

/* ---------------- (4) gaps as ranges ---------------- */
const METHODS = [
  { k: "greedy", label: "greedy", c: "--prior" },
  { k: "cpsat", label: "CP-SAT", c: "--post" },
  { k: "rolling", label: "rolling", c: "--c4" },
  { k: "milp", label: "MILP", c: "--warm" },
];
let gapMode = "range";
function drawGaps() {
  const cv = $("#gapC"), narrow = narrowOf(cv, 600);
  const recs = D.bench, n = recs.length, cap = 25;
  const fr = frame(cv, narrow ? 1.1 : 2.3, { x: [0, n], y: [0, cap], xticks: false, yticks: [0, 5, 10, 15, 20, 25], yfmt: v => v + "%", ylabel: "gap of the plan", pad: { b: narrow ? 30 : 40 } });
  const { ctx, X, Y } = fr;
  const gw = X(1) - X(0), bw = gw * 0.8 / METHODS.length;
  ctx.font = "11px " + css("--font-mono");
  recs.forEach((r, i) => {
    const bestKnown = Math.max(r.greedy, r.cpsat, r.rolling, r.milp);
    METHODS.forEach((m, j) => {
      const v = r[m.k];
      const gHi = 100 * (r.best_bound - v) / r.best_bound, gLo = 100 * (bestKnown - v) / bestKnown;
      const xc = X(i) + gw * 0.1 + (j + 0.5) * bw;
      const top = gapMode === "known" ? gLo : gHi, bot = gapMode === "bound" ? gHi : gLo;
      ctx.fillStyle = alpha(m.c, gapMode === "range" ? 0.35 : 1);
      if (gapMode === "range") ctx.fillRect(xc - bw * 0.35, Y(Math.min(cap, top)), bw * 0.7, Math.max(1.5, Y(Math.min(cap, bot)) - Y(Math.min(cap, top))));
      else ctx.fillRect(xc - bw * 0.35, Y(Math.min(cap, top)), bw * 0.7, Math.max(1.5, Y(0) - Y(Math.min(cap, top))));
      if (gapMode === "range") { dot(ctx, xc, Y(Math.min(cap, gHi)), 3, css(m.c)); dot(ctx, xc, Y(Math.min(cap, gLo)), 2.2, css("--ink")); }
      if (top > cap) { ctx.fillStyle = css(m.c); ctx.textAlign = "center"; ctx.fillText(`${fmt(top, 0)}%`, xc, Y(cap) + 12); }
    });
    ctx.fillStyle = css("--ink-2"); ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(narrow ? `${r.size[0].toUpperCase()}${r.seed}` : `${r.size} ${r.seed}`, X(i + 0.5), Y(0) + 6);
    if (!narrow) { ctx.fillStyle = css("--muted"); ctx.fillText(`bound ${fmt(r.best_bound, 0)}`, X(i + 0.5), Y(0) + 20); }
    ctx.textBaseline = "alphabetic";
  });
  const m0 = D.bench.find(r => r.size === "medium" && r.seed === 0);
  const txt = {
    range: `Each bar is the range in which the <b>true</b> gap of a plan must lie: its top (coloured dot) is the gap against the best bound either exact solver proved, as in results.md; its bottom (black dot) is the gap against the best plan any method found on that instance. The optimum is somewhere between those two numbers, so the true gap is too. On small the two coincide: the bound is the optimum. On medium and large the ranges are wide: CP-SAT's medium seed 0 plan, 557, is 3.1% below the bound ${fmt(m0.best_bound, 0)} but 0% below the best known plan, and the true value is in between.`,
    bound: `The gaps as results.md reports them: (best bound − value) / best bound, with the best bound proved by CP-SAT or HiGHS on that instance. On small the bound is the optimum and these are true gaps. On medium and large they are upper estimates: the bound is loose, so every method looks worse than it is, by an unknown amount that is the same for all methods on one instance. Off-scale bars (the MILP on large) are labelled.`,
    known: `Gaps against the best plan any method found, (best known − value) / best known: a lower estimate of the true gap, since the optimum is at least as good as the best known plan. The method that found the best plan scores 0% by definition, which says nothing about how far it is from the optimum. Read together with the bound view, this brackets the truth.`,
  };
  caption($("#gapCap"), txt[gapMode]);
}
$$("#gapSeg .btn").forEach(btn => btn.addEventListener("click", () => {
  gapMode = btn.dataset.m;
  $$("#gapSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === btn));
  drawGaps();
}));

/* ---------------- (5) the memory sweep with the noise band ---------------- */
function drawMemory() {
  const M = D.memory, cats = M.map(r => r.memory_budget_s >= 1e5 ? "no limit" : String(r.memory_budget_s));
  const cv = $("#memC"), narrow = narrowOf(cv);
  const noise = D.noise.map(r => r.value), nlo = Math.min(...noise), nhi = Math.max(...noise);
  const fr = frame(cv, narrow ? 1.3 : 2.4, { x: [-0.5, M.length - 0.5], y: [440, 590], xticks: M.map((_, i) => i), xfmt: i => cats[i], xlabel: "memory between two contacts [s of recording]", ylabel: "collected priority, medium seed 0" });
  const xs = M.map((_, i) => i);
  const i600 = M.findIndex(r => r.memory_budget_s === 600);
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--warm", 0.22); ctx.fillRect(X(-0.5), Y(nhi), X(M.length - 0.5) - X(-0.5), Math.max(1.5, Y(nlo) - Y(nhi)));
  plotLine(fr, xs, M.map(r => r.cpsat_bound), { color: css("--ink"), width: 1.4, dash: [5, 4] });
  plotLine(fr, xs, M.map(r => r.greedy), { color: css("--prior"), width: 2 });
  plotLine(fr, xs, M.map(r => r.cpsat), { color: css("--post"), width: 2 });
  if (i600 >= 0) D.noise.forEach((r, j) => dot(ctx, X(i600) + (j - 2) * 5, Y(r.value), 2.5, css("--warm")));
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center";
  xs.forEach(i => {
    dot(ctx, X(i), Y(M[i].cpsat), 4, css("--post")); dot(ctx, X(i), Y(M[i].greedy), 3.5, css("--prior"));
    ctx.fillStyle = css("--post"); ctx.fillText(String(M[i].cpsat), X(i), Y(M[i].cpsat) - 9);
    if (M[i].status === "OPTIMAL") { ctx.fillStyle = css("--ink"); ctx.fillText("proved optimal", X(i) + (narrow ? 22 : 0), Y(M[i].cpsat) + 16); }
  });
  caption($("#memCap"), `The memory sweep of results.md on medium seed 0 (cited): CP-SAT with the greedy hint and 30 s per budget (blue), its proven bound (dashed), and greedy (grey). The orange band, drawn across the chart for comparison, is the spread measured by this chapter's script: five CP-SAT runs at the default memory of 600 s with the same 30 s limit and random seeds 0 to 4 gave ${noise.join(", ")} (dots), a spread of ${nhi - nlo} points. At 200 s the memory is the binding constraint: the value drops to ${M[0].cpsat}, and CP-SAT proves it optimal: with so few tasks able to fit, the search closes quickly. At 300 s the bound is still lower (${fmt(M[1].cpsat_bound, 0)}), so memory still binds. From 450 s on, the bound stays at ${fmt(M[2].cpsat_bound, 0)} and the values ${M.slice(2).map(r => r.cpsat).join(", ")} ${M.slice(2).every(r => r.cpsat >= nlo - 1 && r.cpsat <= nhi + 1) ? "all lie inside the noise band, to within a point" : "differ by no more than a few points, the size of the measured noise"}: as far as these runs can tell, more memory changes nothing, and what limits collection there is reconfiguration time and the power budget.`);
}

function drawAll() { drawHero(); drawCases(); drawSweep(); drawGaps(); drawMemory(); }
drawAll(); onRedraw(drawAll);
})();
