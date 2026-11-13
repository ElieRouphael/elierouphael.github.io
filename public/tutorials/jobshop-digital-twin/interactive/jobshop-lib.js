/* Which Job Should the Machine Do Next? — the toolkit shared by every chapter's demos.
   Plain JavaScript, no dependencies beyond ../kit/kit.js (window.Kit), loaded before
   interactive/<id>.js. Chapters must not edit this file: chapter-specific code lives in the
   chapter's own interactive/<id>.js. Everything here mirrors the Python of the repository
   (github.com/ElieRouphael/jobshop-digital-twin) or the course's worked example, so the pages
   compute what the Python computes.

   window.JS = {
     PAINT, WORKSHOP, FAM, RULES,
     rng, expo, permutations,
     runSequence, ruleSequence, totalTardiness, bestSequence,
     setupTime, minRemainingWork,
     drawGantt, bootstrapMean, mean, fmtMin
   }
*/
(function (global) {
"use strict";
const Kit = global.Kit;

/* ---------------- the worked example of chapter A2 (and the landing page) ----------------
   One paint machine, set up for blue; a colour change takes 10 min; four orders in this
   order of arrival. Same timing rule as the repository's simulator: an order starts after
   the changeover it needs, and its lateness is max(0, completion - due). */
const PAINT = {
  setup: 10,
  start: "blue",
  orders: [
    { name: "A", family: "blue", p: 20, due: 30, arrival: 1 },
    { name: "B", family: "red",  p: 10, due: 25, arrival: 2 },
    { name: "C", family: "blue", p: 15, due: 70, arrival: 3 },
    { name: "D", family: "red",  p: 25, due: 60, arrival: 4 },
  ],
};

/* ---------------- the repository's workshop: make_workshop() with its default seed ----------------
   Machines and families are numbered from 0 as in the code. routings[f][k] = {machine: minutes};
   setup[m][f][g] = changeover on machine m from family f to family g (0 on the diagonal; a machine
   that has run nothing yet needs none); travel[a][b] between stations, where the load station is 0
   and machine m stands at station m + 1. Printed from jobshop/scenario.py on 2026-11-13. */
const WORKSHOP = {
  nMachines: 5, nAgvs: 2, nFamilies: 4,
  routings: [
    [{1: 6, 2: 10}, {2: 26, 3: 30}, {1: 10, 4: 7}, {0: 8, 2: 8, 3: 7}],
    [{0: 9}, {1: 18, 3: 18}, {2: 22, 4: 19}, {2: 15, 3: 17}],
    [{0: 16, 2: 19, 3: 16}, {1: 17, 3: 24, 4: 24}, {1: 18, 4: 21}, {0: 21, 4: 19}],
    [{1: 26, 4: 23}, {4: 17}, {0: 20, 1: 16, 2: 20}, {0: 17, 2: 19}],
  ],
  setup: [
    [[0, 7, 10, 3], [6, 0, 9, 9], [6, 6, 0, 6], [7, 4, 9, 0]],
    [[0, 3, 6, 5], [8, 0, 8, 8], [10, 10, 0, 4], [3, 4, 8, 0]],
    [[0, 10, 10, 8], [10, 0, 9, 3], [3, 9, 0, 3], [10, 9, 10, 0]],
    [[0, 5, 4, 7], [10, 0, 5, 10], [6, 9, 0, 4], [6, 5, 4, 0]],
    [[0, 10, 9, 4], [10, 0, 10, 5], [6, 7, 0, 8], [6, 4, 10, 0]],
  ],
  travel: [
    [0, 2, 2, 3, 3, 3], [2, 0, 2, 2, 2, 3], [2, 2, 0, 3, 3, 3],
    [3, 2, 3, 0, 2, 3], [3, 2, 3, 2, 0, 2], [3, 3, 3, 3, 2, 0],
  ],
  meanInterarrival: 20.83,   // minutes, mean_interarrival(ws, load=0.60)
  mtbf: 120, mttr: 20,       // the "frequent breakdowns" level
  allowance: 2,              // minutes the CP-SAT planner adds to each move for the AGV to come
};
function setupTime(m, prevFamily, family) {
  return prevFamily == null ? 0 : WORKSHOP.setup[m][prevFamily][family];
}
function minRemainingWork(family, op = 0) {
  return WORKSHOP.routings[family].slice(op).reduce((s, alt) => s + Math.min(...Object.values(alt)), 0);
}

/* Colours: one per product family (index 0-3 as in the code), and the two paint colours.
   They are fixed (not theme tokens) so a family keeps its colour across every chapter. */
const FAM = { colours: ["#2a78d6", "#e0703a", "#3f9b5b", "#9b6bd1"], blue: "#2a78d6", red: "#d9534f",
              setup: "#9aa4b0", down: "#d9534f" };

/* ---------------- seeded randomness (mulberry32), so demos are reproducible ---------------- */
function rng(seed = 1) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const expo = (r, mean) => -mean * Math.log(1 - r());
function permutations(arr) {
  if (arr.length <= 1) return [arr.slice()];
  const out = [];
  arr.forEach((x, i) => permutations([...arr.slice(0, i), ...arr.slice(i + 1)]).forEach(p => out.push([x, ...p])));
  return out;
}

/* ---------------- one machine: evaluate a sequence, apply a rule, find the best ----------------
   orders: [{name, family, p, due, arrival}]; seq: array of names; setup: number, or a function
   (prevFamily, family) -> minutes. Returns [{name, free, start, end, late, setup}] where `free` is
   when the machine became free for this order and [free, start) is the changeover. */
function runSequence(orders, seq, setup = PAINT.setup, startFamily = PAINT.start, t0 = 0) {
  const by = Object.fromEntries(orders.map(o => [o.name, o]));
  const sf = typeof setup === "function" ? setup : (a, b) => (a == null || a === b ? 0 : setup);
  let t = t0, fam = startFamily;
  return seq.map(name => {
    const o = by[name], s = sf(fam, o.family), start = t + s, end = start + o.p;
    const row = { name, free: t, start, end, late: Math.max(0, end - o.due), setup: s };
    t = end; fam = o.family;
    return row;
  });
}
const totalTardiness = rows => rows.reduce((s, r) => s + r.late, 0);
/* The four classic rules, applied each time the machine is free: key(order, currentFamily) -> sort key. */
const RULES = {
  FIFO: { name: "First come, first served", key: (o) => [o.arrival] },
  SPT:  { name: "Shortest job first",       key: (o) => [o.p, o.arrival] },
  EDD:  { name: "Most urgent first",        key: (o) => [o.due, o.arrival] },
  SST:  { name: "Shortest setup first",     key: (o, fam, setup) => [setup(fam, o.family), o.arrival] },
};
function lexLess(a, b) { for (let i = 0; i < a.length; i++) { if (a[i] !== b[i]) return a[i] < b[i]; } return false; }
function ruleSequence(orders, rule, setup = PAINT.setup, startFamily = PAINT.start) {
  const sf = typeof setup === "function" ? setup : (a, b) => (a == null || a === b ? 0 : setup);
  const waiting = orders.slice(), seq = [];
  let fam = startFamily;
  while (waiting.length) {
    let best = 0;
    for (let i = 1; i < waiting.length; i++)
      if (lexLess(RULES[rule].key(waiting[i], fam, sf), RULES[rule].key(waiting[best], fam, sf))) best = i;
    const o = waiting.splice(best, 1)[0];
    seq.push(o.name); fam = o.family;
  }
  return seq;
}
/* Exhaustive search over all n! orders (n <= 8 in the browser); ties broken alphabetically. */
function bestSequence(orders, setup = PAINT.setup, startFamily = PAINT.start) {
  let best = null, bestVal = Infinity, count = 0;
  for (const p of permutations(orders.map(o => o.name))) {
    count++;
    const v = totalTardiness(runSequence(orders, p, setup, startFamily));
    if (v < bestVal || (v === bestVal && p.join("") < best.join(""))) { best = p; bestVal = v; }
  }
  return { seq: best, value: bestVal, count };
}

/* ---------------- drawing a schedule ----------------
   drawGantt(canvas, blocks, {lanes, t0, t1, aspect, laneLabels, due})
   blocks: [{lane, start, end, colour, label, kind}] with kind "work" | "setup" | "down" | "agv".
   lanes: number of lanes (lane 0 at the top). due: optional [{lane, t, label}] tick marks.
   Uses the page's theme tokens for text and grid, and redraws with Kit.onRedraw if you register it. */
function drawGantt(canvas, blocks, o = {}) {
  const lanes = o.lanes || 1 + Math.max(0, ...blocks.map(b => b.lane));
  const t0 = o.t0 ?? 0, t1 = o.t1 ?? Math.max(1, ...blocks.map(b => b.end));
  const labelW = o.labelWidth ?? (o.laneLabels ? 92 : 8);
  const laneH = o.laneHeight ?? 30;
  const { ctx, w, h } = Kit.fit(canvas, o.aspect || null);
  if (!o.aspect) { canvas.style.height = (lanes * laneH + 30) + "px"; }
  const H = o.aspect ? h : lanes * laneH + 30;
  const X = t => labelW + (t - t0) / (t1 - t0) * (w - labelW - 10);
  const css = Kit.css;
  ctx.clearRect(0, 0, w, H);
  ctx.font = "11px " + css("--font-mono");
  ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of Kit.ticks(t0, t1, Math.max(4, Math.floor((w - labelW) / 70)))) {
    ctx.strokeStyle = css("--grid") || css("--rule");
    ctx.beginPath(); ctx.moveTo(Math.round(X(t)) + 0.5, 0); ctx.lineTo(Math.round(X(t)) + 0.5, lanes * laneH); ctx.stroke();
    ctx.fillText(Kit.tickLabel(t), X(t), lanes * laneH + 6);
  }
  if (o.laneLabels) {
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillStyle = css("--ink");
    o.laneLabels.forEach((l, i) => ctx.fillText(l, labelW - 8, i * laneH + laneH / 2));
  }
  for (const b of blocks) {
    const y = b.lane * laneH, x0 = X(Math.max(t0, b.start)), x1 = X(Math.min(t1, b.end));
    if (x1 <= x0) continue;
    const pad = b.kind === "agv" ? laneH * 0.32 : b.kind === "down" ? 2 : laneH * 0.14;
    ctx.globalAlpha = b.kind === "down" ? 0.28 : 1;
    ctx.fillStyle = b.colour || (b.kind === "setup" ? FAM.setup : b.kind === "down" ? FAM.down : css("--accent"));
    if (b.kind === "setup") {        // striped changeover
      ctx.save(); ctx.beginPath(); ctx.rect(x0, y + pad, x1 - x0, laneH - 2 * pad); ctx.clip();
      ctx.fillStyle = css("--surface") || "#fff"; ctx.fillRect(x0, y + pad, x1 - x0, laneH - 2 * pad);
      ctx.strokeStyle = FAM.setup; ctx.lineWidth = 2;
      for (let s = x0 - laneH; s < x1; s += 6) { ctx.beginPath(); ctx.moveTo(s, y + laneH); ctx.lineTo(s + laneH, y); ctx.stroke(); }
      ctx.restore();
    } else {
      ctx.fillRect(x0, y + pad, x1 - x0, laneH - 2 * pad);
    }
    ctx.globalAlpha = 1;
    if (b.label && x1 - x0 > 14 && b.kind !== "setup") {
      ctx.fillStyle = "#ffffff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.font = "600 11px " + css("--font-mono");
      ctx.fillText(b.label, (x0 + x1) / 2, y + laneH / 2);
      ctx.font = "11px " + css("--font-mono");
    }
  }
  for (const d of o.due || []) {
    const x = X(d.t), y = d.lane * laneH;
    ctx.strokeStyle = FAM.red; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, y + 2); ctx.lineTo(x, y + laneH - 2); ctx.stroke();
  }
  return { X, laneH, w, h: H };
}

/* ---------------- statistics ---------------- */
const mean = xs => xs.reduce((s, v) => s + v, 0) / xs.length;
/* Percentile bootstrap interval for the mean, as jobshop/metrics.py:bootstrap_ci (B resamples). */
function bootstrapMean(values, B = 10000, conf = 0.95, seed = 0) {
  const r = rng(seed + 12345), n = values.length, means = new Float64Array(B);
  for (let b = 0; b < B; b++) { let s = 0; for (let i = 0; i < n; i++) s += values[Math.floor(r() * n)]; means[b] = s / n; }
  means.sort();
  const a = (1 - conf) / 2, q = p => means[Math.min(B - 1, Math.max(0, Math.floor(p * (B - 1))))];
  return [q(a), q(1 - a)];
}
const fmtMin = v => (v == null || !isFinite(v)) ? "–" : (Math.round(v * 10) / 10).toString() + " min";

global.JS = { PAINT, WORKSHOP, FAM, RULES, rng, expo, permutations, runSequence, ruleSequence,
              totalTardiness, bestSequence, setupTime, minRemainingWork, drawGantt, bootstrapMean, mean, fmtMin };
})(window);
