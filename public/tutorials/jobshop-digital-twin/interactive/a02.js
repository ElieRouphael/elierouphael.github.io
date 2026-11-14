/* Which Job Should the Machine Do Next?, chapter A2, one machine: when a rule is optimal. The demos.
   Hero: the paint machine under the four rules, the best order or an order you build, with the
   histogram of the total tardiness of all 24 orders. Swap widget: two adjacent orders exchanged, as a
   function of the time they start. Random widget: how often each rule is optimal on small random
   instances, drawn with the toolkit's seeded generator exactly as code/a02_numbers.py draws them.
   Needs ../kit/kit.js and jobshop-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, fit, alpha, onRedraw, narrowOf } = window.Kit;
const JS = window.JS;
const RULES4 = ["FIFO", "SPT", "EDD", "SST"];

/* The repository's keys (jobshop/rules.py): FIFO (queued_at, idx), SPT (p, idx), EDD (due, idx),
   SST (setup, due, idx). Applied each time the machine is free. */
function lexLess(a, b) { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i]; return false; }
function repoKey(rule, o, fam, sf) {
  if (rule === "FIFO") return [o.arrival, o.idx];
  if (rule === "SPT") return [o.p, o.idx];
  if (rule === "EDD") return [o.due, o.idx];
  return [sf(fam, o.family), o.due, o.idx];
}
function repoSequence(orders, rule, sf, startFamily) {
  const waiting = orders.slice(), seq = [];
  let fam = startFamily;
  while (waiting.length) {
    let b = 0;
    for (let i = 1; i < waiting.length; i++) if (lexLess(repoKey(rule, waiting[i], fam, sf), repoKey(rule, waiting[b], fam, sf))) b = i;
    const o = waiting.splice(b, 1)[0];
    seq.push(o.name); fam = o.family;
  }
  return seq;
}

/* ================================================================ hero: the paint machine */
const PAINT = JS.PAINT.orders.map((o, i) => Object.assign({ idx: i }, o));
const hs = { setup: 10, mode: "SST", custom: [] };
const paintSetup = s => (a, b) => (a == null || a === b ? 0 : s);

function heroData() {
  const s = hs.setup, sf = paintSetup(s);
  const all = JS.permutations(PAINT.map(o => o.name)).map(p => {
    const rows = JS.runSequence(PAINT, p, s, JS.PAINT.start);
    return { seq: p.join(""), T: JS.totalTardiness(rows), cmax: rows[rows.length - 1].end };
  });
  const best = JS.bestSequence(PAINT, s, JS.PAINT.start);
  const rules = {};
  RULES4.forEach(r => { const q = repoSequence(PAINT, r, sf, JS.PAINT.start).join(""); rules[r] = { seq: q, T: all.find(a => a.seq === q).T }; });
  rules.Best = { seq: best.seq.join(""), T: best.value };
  return { all, rules, tmax: Math.max(...all.map(a => a.cmax)) };
}

function drawHero() {
  const D = heroData(), s = hs.setup;
  const seq = hs.mode === "custom" ? hs.custom.slice() : D.rules[hs.mode].seq.split("");
  const rows = JS.runSequence(PAINT, seq, s, JS.PAINT.start);
  const complete = seq.length === 4;
  /* the machine's timeline, with the due dates on a second lane */
  const blocks = [];
  rows.forEach(r => {
    const o = PAINT.find(x => x.name === r.name);
    if (r.start > r.free) blocks.push({ lane: 0, start: r.free, end: r.start, kind: "setup" });
    blocks.push({ lane: 0, start: r.start, end: r.end, colour: o.family === "blue" ? JS.FAM.blue : JS.FAM.red, label: r.name, kind: "work" });
  });
  const cv = $("#hGantt");
  const g = JS.drawGantt(cv, blocks, { lanes: 2, t0: 0, t1: Math.max(D.tmax, 80), laneLabels: ["machine", "due dates"], laneHeight: 34 });
  const ctx = cv.getContext("2d");
  ctx.font = "600 11px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const dueX = PAINT.map(o => g.X(o.due));
  PAINT.forEach(o => {
    const r = rows.find(x => x.name === o.name), x = g.X(o.due), y = g.laneH * 1.5, late = r && r.late > 0;
    ctx.fillStyle = late ? JS.FAM.red : css("--muted");
    ctx.beginPath(); ctx.moveTo(x, y - 7); ctx.lineTo(x + 6, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 6, y); ctx.closePath(); ctx.fill();
    /* the letter goes right of its diamond, or left, or above when another diamond is within 24 px */
    const crowdR = dueX.some(z => z > x && z - x < 24), crowdL = dueX.some(z => z < x && x - z < 24) || x - g.X(0) < 20;
    ctx.fillStyle = css("--ink");
    if (!crowdR) ctx.fillText(o.name, x + 13, y);
    else if (!crowdL) ctx.fillText(o.name, x - 13, y);
    else ctx.fillText(o.name, x, y - 13);
    if (late) { /* a line from the due date to the completion: the tardiness */
      const yy = y + (o.idx % 2 ? 9 : -9);
      ctx.strokeStyle = JS.FAM.red; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, yy); ctx.lineTo(g.X(r.end), yy); ctx.moveTo(g.X(r.end), yy - 4); ctx.lineTo(g.X(r.end), yy + 4); ctx.stroke();
    }
  });
  /* histogram of the 24 totals */
  const hc = $("#hHist"), narrow = narrowOf(hc);
  const counts = {}; D.all.forEach(a => { counts[a.T] = (counts[a.T] || 0) + 1; });
  const maxT = Math.max(...D.all.map(a => a.T)), maxC = Math.max(...Object.values(counts));
  const fr = frame(hc, narrow ? 1.5 : 2.6, { x: [-5, maxT + 10], y: [0, maxC], pad: { t: 64, l: 34 }, xlabel: "total tardiness of the order (min)", ylabel: "orders", yticks: Array.from({ length: maxC + 1 }, (_, i) => i) });
  const cur = complete ? D.all.find(a => a.seq === seq.join("")).T : null;
  const bw = Math.max(3, Math.min(14, fr.iw / (maxT + 15) * 4));
  Object.entries(counts).forEach(([v, c]) => {
    v = +v;
    fr.ctx.fillStyle = v === cur ? css("--accent") : alpha("--ink", 0.28);
    fr.ctx.fillRect(fr.X(v) - bw / 2, fr.Y(c), bw, fr.Y(0) - fr.Y(c));
  });
  /* the rules' positions above their bars; a label that would touch one already placed moves up a row */
  const placed = [];
  fr.ctx.font = "600 11px " + css("--font-mono"); fr.ctx.textAlign = "center"; fr.ctx.textBaseline = "bottom";
  ["Best", ...RULES4].map(r => ({ r, v: D.rules[r].T })).sort((a, b) => a.v - b.v).forEach(({ r, v }) => {
    const text = r === "Best" ? "best" : r, w = fr.ctx.measureText(text).width + 6, x = fr.X(v), base = fr.Y(counts[v]) - 4;
    let y = base;
    while (placed.some(q => Math.abs(q.x - x) < (q.w + w) / 2 && Math.abs(q.y - y) < 12)) y -= 13;
    placed.push({ x, y, w });
    fr.ctx.fillStyle = r === "Best" ? css("--warm") : (r === hs.mode ? css("--accent") : css("--ink"));
    fr.ctx.fillText(text, x, y);
  });
  /* readouts */
  const T = JS.totalTardiness(rows), changes = rows.filter(r => r.setup > 0).length;
  $("#hSetupV").textContent = s + " min";
  $$("#demo [data-rule]").forEach(b => b.setAttribute("aria-pressed", String(hs.mode === b.dataset.rule)));
  $$("#demo [data-pick]").forEach(b => { b.disabled = hs.mode === "custom" && hs.custom.length < 4 && hs.custom.includes(b.dataset.pick); });
  $("#rSeq").textContent = complete ? seq.join(" ") : (seq.join(" ") || "–") + " …";
  $("#rT").textContent = complete ? T + " min" : "–";
  $("#rCh").textContent = complete ? String(changes) : "–";
  $("#rCmax").textContent = complete ? rows[rows.length - 1].end + " min" : "–";
  const better = complete ? D.all.filter(a => a.T < T).length : null;
  $("#rRank").textContent = complete ? `${better} of 24` : "–";
  /* caption */
  const name = { FIFO: "First come, first served (FIFO)", SPT: "Shortest job first (SPT)", EDD: "Most urgent first (EDD)", SST: "Shortest setup first (SST)", Best: "The best of the 24 orders" }[hs.mode];
  const lateList = rows.filter(r => r.late > 0).map(r => `${r.name} ${r.late}`).join(", ");
  let msg;
  if (!complete) msg = `Pick the orders one after the other: ${seq.length} of 4 chosen. The machine starts set up for blue, and every colour change costs ${s} min.`;
  else {
    msg = `<b>${hs.mode === "custom" ? "Your order" : name}</b> paints ${seq.join(" → ")}: ${changes} colour change${changes === 1 ? "" : "s"}, finish at minute ${rows[rows.length - 1].end}, total tardiness <b>${T} min</b>${lateList ? ` (late: ${lateList})` : " (nothing late)"}. `;
    msg += better === 0 ? `No order of the 24 does better${D.all.filter(a => a.T === T).length > 1 ? "; other orders tie with it" : ""}. ` : `${better} of the 24 orders ${better === 1 ? "does" : "do"} better; the best, ${D.rules.Best.seq}, is ${D.rules.Best.T} min late in total. `;
    const r = D.rules;
    msg += `With ${s}-min changeovers the rules score FIFO ${r.FIFO.T}, SPT ${r.SPT.T}, EDD ${r.EDD.T}, SST ${r.SST.T}.`;
    if (s === 0) msg += " Without changeovers shortest setup first falls back on its tie-break, the due date, and becomes most urgent first; here EDD is late on nothing, so it is optimal.";
    else if (r.SST.T === r.Best.T) msg += " Changeovers now dominate, and avoiding them is optimal.";
  }
  caption($("#heroStatus"), msg);
}
$$("#demo [data-rule]").forEach(b => b.addEventListener("click", () => { hs.mode = b.dataset.rule; hs.custom = []; drawHero(); }));
$$("#demo [data-pick]").forEach(b => b.addEventListener("click", () => {
  if (hs.mode !== "custom" || hs.custom.length === 4) { hs.mode = "custom"; hs.custom = []; }
  if (!hs.custom.includes(b.dataset.pick)) hs.custom.push(b.dataset.pick);
  drawHero();
}));
$("#hClear").addEventListener("click", () => { hs.mode = "custom"; hs.custom = []; drawHero(); });
$("#hSetup").addEventListener("input", () => { hs.setup = +$("#hSetup").value; drawHero(); });

/* ================================================================ the swap of two adjacent orders */
const ws = { obj: "T" };
const pos = x => Math.max(0, x);
function pairCost(obj, t, a, b) {          /* a first, then b, starting at t */
  const Ca = t + a.p, Cb = t + a.p + b.p;
  if (obj === "C") return Ca + Cb;
  if (obj === "L") return Math.max(Ca - a.d, Cb - b.d);
  return pos(Ca - a.d) + pos(Cb - b.d);
}
function drawSwap() {
  const i = { p: +$("#wPi").value, d: +$("#wDi").value }, j = { p: +$("#wPj").value, d: +$("#wDj").value };
  ["Pi", "Di", "Pj", "Dj"].forEach(k => { $("#w" + k + "V").textContent = $("#w" + k).value; });
  $$("#swapW [data-obj]").forEach(b => b.setAttribute("aria-pressed", String(ws.obj === b.dataset.obj)));
  const ts = Array.from({ length: 101 }, (_, k) => k), ci = ts.map(t => pairCost(ws.obj, t, i, j)), cj = ts.map(t => pairCost(ws.obj, t, j, i));
  const lo = Math.min(0, ...ci, ...cj), hi = Math.max(10, ...ci, ...cj);
  const cv = $("#swapC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.3 : 2.4, { x: [0, 100], y: [lo, hi * 1.05], xlabel: "start time t of the pair (min)", ylabel: { C: "Cᵢ + Cⱼ", L: "max(Lᵢ, Lⱼ)", T: "Tᵢ + Tⱼ" }[ws.obj] });
  /* shade where each order is strictly better */
  ts.forEach(t => {
    if (t === 100 || ci[t] === cj[t]) return;
    fr.ctx.fillStyle = ci[t] < cj[t] ? alpha("--accent", 0.10) : alpha("--warm", 0.12);
    fr.ctx.fillRect(fr.X(t), fr.pad.t, fr.X(t + 1) - fr.X(t), fr.ih);
  });
  window.Kit.plotLine(fr, ts, ci, { color: css("--accent"), width: 2.2 });
  window.Kit.plotLine(fr, ts, cj, { color: css("--warm"), width: 2.2, dash: [6, 4] });
  fr.ctx.font = "11px " + css("--font-mono"); fr.ctx.textAlign = "left"; fr.ctx.fillStyle = css("--accent");
  fr.ctx.fillText("i first (solid)", fr.pad.l + 6, fr.pad.t + 12); fr.ctx.fillStyle = css("--warm"); fr.ctx.fillText("j first (dashed)", fr.pad.l + 6, fr.pad.t + 26);
  fr.ctx.fillStyle = css("--muted"); fr.ctx.fillText("shaded: where that order is strictly better", fr.pad.l + 6, fr.pad.t + 40);
  /* where the preference changes */
  const better = ts.map(t => Math.sign(cj[t] - ci[t]));   /* +1: i first better */
  const runs = []; let k = 0;
  while (k < ts.length) { let e = k; while (e + 1 < ts.length && better[e + 1] === better[k]) e++; runs.push([better[k], ts[k], ts[e]]); k = e + 1; }
  const word = v => v > 0 ? "<b>i first</b>" : v < 0 ? "<b>j first</b>" : "a tie";
  const desc = runs.map(([v, a, b]) => a === b ? `${word(v)} at [[t = ${a}]]` : `${word(v)} for [[t \\in [${a}, ${b}]]]`).join(", ");
  const paintCD = i.p === 15 && i.d === 70 && j.p === 25 && j.d === 60;
  let msg = `Order [[i]]${paintCD ? " (C of the paint machine)" : ""} takes ${i.p} min, due at ${i.d}; order [[j]]${paintCD ? " (D)" : ""} takes ${j.p} min, due at ${j.d}. `;
  if (ws.obj === "C") msg += `The two lines are parallel: i first minus j first is [[p_i - p_j = ${i.p - j.p}]] at every start time, so the shorter order goes first wherever the pair sits. That independence from [[t]] is what makes shortest job first an index rule.`;
  else if (ws.obj === "L") msg += `Most urgent first never loses: with [[d_${i.d <= j.d ? "i" : "j"} \\le d_${i.d <= j.d ? "j" : "i"}]] the line of ${i.d <= j.d ? "i" : "j"} first stays on or below the other at every [[t]]. Over [[t \\in [0, 100]]]: ${desc}.`;
  else msg += `Over [[t \\in [0, 100]]]: ${desc}. Which order is better depends on when the pair starts, that is, on the jobs placed before it, so no index computed from one job alone can rank them for total tardiness.`;
  caption($("#swapCap"), msg);
}
$$("#swapW [data-obj]").forEach(b => b.addEventListener("click", () => { ws.obj = b.dataset.obj; drawSwap(); }));
["#wPi", "#wDi", "#wPj", "#wDj"].forEach(id => $(id).addEventListener("input", drawSwap));
$("#wReset").addEventListener("click", () => { $("#wPi").value = 15; $("#wDi").value = 70; $("#wPj").value = 25; $("#wDj").value = 60; drawSwap(); });

/* ================================================================ how often is a rule optimal? */
const S0 = JS.WORKSHOP.setup[0];                          /* the changeovers of the workshop's machine 0 */
const ZERO = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
function instance(r, n, theta, R = 0.6) {
  const p = [], fam = [], d = [];
  for (let k = 0; k < n; k++) p.push(5 + Math.floor(r() * 26));
  for (let k = 0; k < n; k++) fam.push(Math.floor(r() * 4));
  const P = p.reduce((a, b) => a + b, 0);
  for (let k = 0; k < n; k++) d.push(Math.max(0, Math.floor(P * (1 - theta + R * (r() - 0.5)) + 0.5)));
  return { p, fam, d };
}
function tardiness(I, seq, St) {
  let t = 0, prev = null, s = 0;
  for (const j of seq) { t += (prev == null ? 0 : St[prev][I.fam[j]]) + I.p[j]; prev = I.fam[j]; s += Math.max(0, t - I.d[j]); }
  return s;
}
function brute(I, St) {             /* every order, prefixes shared, a branch cut when it cannot win */
  const n = I.p.length, full = (1 << n) - 1;
  let best = Infinity;
  (function dfs(t, prev, used, acc) {
    if (acc >= best) return;
    if (used === full) { best = acc; return; }
    for (let j = 0; j < n; j++) if (!(used >> j & 1)) {
      const tj = t + (prev == null ? 0 : St[prev][I.fam[j]]) + I.p[j];
      dfs(tj, I.fam[j], used | 1 << j, acc + Math.max(0, tj - I.d[j]));
    }
  })(0, null, 0, 0);
  return best;
}
function ruleSeqIdx(I, rule, St) {
  const orders = I.p.map((p, k) => ({ name: k, idx: k, arrival: k, p, due: I.d[k], family: I.fam[k] }));
  return repoSequence(orders, rule, (a, b) => (a == null ? 0 : St[a][b]), null);
}
const fixed1 = (numer, denom) => { const q = Math.floor((20 * numer + denom) / (2 * denom)); return `${Math.floor(q / 10)}.${q % 10}`; };
const cache = {};
function experiment(n, theta, withSetups, count = 300, seed = 2026) {
  const key = [n, theta, withSetups].join("|");
  if (cache[key]) return cache[key];
  const r = JS.rng(seed), St = withSetups ? S0 : ZERO, keys = [...RULES4, "best4"];
  const hits = Object.fromEntries(keys.map(k => [k, 0])), excess = Object.fromEntries(keys.map(k => [k, 0]));
  let optSum = 0;
  for (let c = 0; c < count; c++) {
    const I = instance(r, n, theta), opt = brute(I, St);
    optSum += opt;
    const v = {}; RULES4.forEach(k => { v[k] = tardiness(I, ruleSeqIdx(I, k, St), St); });
    v.best4 = Math.min(...RULES4.map(k => v[k]));
    keys.forEach(k => { hits[k] += v[k] === opt; excess[k] += v[k] - opt; });
  }
  return (cache[key] = { hits, excess, optSum, count });
}
function drawRand() {
  const n = +$("#xN").value, theta = +$("#xTheta").value / 10, withS = $("#xSetup").checked;
  $("#xNV").textContent = n; $("#xThetaV").textContent = theta.toFixed(1);
  const E = experiment(n, theta, withS), keys = [...RULES4, "best4"], c = E.count;
  const cv = $("#randC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.4 : 2.6, { x: [0, 100], y: [0, 5], pad: { l: 96, t: 8 }, yticks: false, xticks: [0, 20, 40, 60, 80, 100], xfmt: v => v + "%", xlabel: "instances on which the rule is optimal" });
  const ctx = fr.ctx, labels = { FIFO: "FIFO", SPT: "SPT", EDD: "EDD", SST: "SST", best4: "best of four" };
  keys.forEach((k, idx) => {
    const y0 = fr.Y(5 - idx) + 4, h = fr.Y(4 - idx) - fr.Y(5 - idx) - 8, v = 100 * E.hits[k] / c;
    ctx.fillStyle = k === "best4" ? css("--warm") : css("--accent");
    ctx.fillRect(fr.X(0), y0, Math.max(1, fr.X(v) - fr.X(0)), h);
    ctx.fillStyle = css("--ink"); ctx.font = "12px " + css("--font-mono"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(labels[k], fr.pad.l - 8, y0 + h / 2);
    ctx.textAlign = "left"; ctx.fillStyle = css("--muted");
    ctx.fillText(`${fixed1(100 * E.hits[k], c)}%  (+${fixed1(E.excess[k], c)} min)`, Math.min(fr.X(v) + 6, fr.w - 150), y0 + h / 2);
  });
  const h = E.hits;
  caption($("#randCap"), `${c} random instances of [[n = ${n}]] orders (processing times 5 to 30 min, four families, due dates uniform between [[(0.7-\\theta)\\sum p_j]] and [[(1.3-\\theta)\\sum p_j]], rounded and never below 0, with [[\\theta = ${theta.toFixed(1)}]])${withS ? ", with the changeovers of the workshop's machine 0" : ", without changeovers"}; the optimum of each is found exactly, by a search over orders that drops a prefix once it is as late as the best order found. FIFO is optimal on ${fixed1(100 * h.FIFO, c)}&nbsp;%, SPT on ${fixed1(100 * h.SPT, c)}&nbsp;%, EDD on ${fixed1(100 * h.EDD, c)}&nbsp;%, SST on ${fixed1(100 * h.SST, c)}&nbsp;%, and the best of the four on ${fixed1(100 * h.best4, c)}&nbsp;%. In brackets: the average number of minutes of total tardiness above the optimum. The mean optimum is ${fixed1(E.optSum, c)} min.${withS ? "" : " Without changeovers SST's first key is always 0, so it is EDD."}`);
}
["#xN", "#xTheta"].forEach(id => $(id).addEventListener("input", drawRand));
$("#xSetup").addEventListener("change", drawRand);

function drawAll() { drawHero(); drawSwap(); drawRand(); }
drawAll();
onRedraw(drawAll);
})();
