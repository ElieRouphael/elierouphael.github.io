/* Which Job Should the Machine Do Next?, chapter A1, what a schedule is: the demos.
   Hero: the paint machine with a reorderable sequence; the Gantt chart, the per-order table and
   every measure update live. Widget 2: all 24 sequences on four measures. Widget 3: a hand-made
   timetable and a small checker (capacity, changeover, release). Widget 4: the seven methods of the
   benchmark on makespan, utilisation, flow time and tardiness (means from results/runs.csv).
   Needs ../kit/kit.js and jobshop-lib.js. Every number agrees with code/a01_numbers.py. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, alpha, onRedraw, narrowOf, fit } = window.Kit;
const { PAINT, FAM, RULES, drawGantt, permutations } = window.JS;

const NAMES = PAINT.orders.map(o => o.name);
const BY = Object.fromEntries(PAINT.orders.map(o => [o.name, o]));
const REL = { none: { A: 0, B: 0, C: 0, D: 0 }, stagger: { A: 0, B: 5, C: 10, D: 15 } };
const colourOf = n => BY[n].family === "blue" ? FAM.blue : FAM.red;
const setupFor = (prev, fam) => (prev == null || prev === fam) ? 0 : PAINT.setup;

/* The semi-active timetable of a sequence, as a01_numbers.py:schedule. The changeover starts as soon
   as the machine is free and the order has arrived (the repository's rule). */
function schedule(seq, rel) {
  let t = 0, fam = PAINT.start;
  return seq.map(n => {
    const o = BY[n], s = setupFor(fam, o.family), ready = Math.max(t, rel[n]), S = ready + s, C = S + o.p;
    const row = { name: n, r: rel[n], p: o.p, d: o.due, setup: s, free: t, ready, S, C,
                  L: C - o.due, T: Math.max(0, C - o.due), E: Math.max(0, o.due - C), F: C - rel[n], U: C > o.due ? 1 : 0 };
    t = C; fam = o.family;
    return row;
  });
}
function measures(rows) {
  const sum = k => rows.reduce((s, r) => s + r[k], 0), max = k => Math.max(...rows.map(r => r[k]));
  return { Cmax: max("C"), sumC: sum("C"), sumF: sum("F"), Lmax: max("L"), Tmax: max("T"), sumT: sum("T"),
           meanT: sum("T") / rows.length, sumU: sum("U"), sumE: sum("E"), setups: sum("setup"),
           changes: rows.filter(r => r.setup).length };
}
/* A rule applied each time the machine frees, among the orders that have arrived (all of them when
   every order waits at minute 0, which gives the sequences of notation.md). */
function ruleSeq(rule, rel) {
  const waiting = NAMES.slice(), seq = [];
  let t = 0, fam = PAINT.start;
  while (waiting.length) {
    let avail = waiting.filter(n => rel[n] <= t);
    if (!avail.length) { t = Math.min(...waiting.map(n => rel[n])); avail = waiting.filter(n => rel[n] <= t); }
    const key = n => RULES[rule].key(BY[n], fam, setupFor);
    let best = avail[0];
    for (const n of avail.slice(1)) { const a = key(n), b = key(best); for (let i = 0; i < a.length; i++) { if (a[i] !== b[i]) { if (a[i] < b[i]) best = n; break; } } }
    const o = BY[best], s = setupFor(fam, o.family);
    t = Math.max(t, rel[best]) + s + o.p; fam = o.family;
    seq.push(best); waiting.splice(waiting.indexOf(best), 1);
  }
  return seq;
}
const ALL = permutations(NAMES);
function allMeasures(rel) { return ALL.map(s => ({ seq: s, key: s.join(""), m: measures(schedule(s, rel)) })); }
function bestBy(list, k) { return list.slice().sort((a, b) => a.m[k] - b.m[k] || (a.key < b.key ? -1 : 1))[0]; }
function rankBy(list, k, key) { return list.slice().sort((a, b) => a.m[k] - b.m[k] || (a.key < b.key ? -1 : 1)).findIndex(x => x.key === key) + 1; }

const st = { seq: ["A", "B", "C", "D"], stagger: false, preset: "FIFO" };
const rel = () => st.stagger ? REL.stagger : REL.none;

/* ---------------- the Gantt chart of a timetable ---------------- */
function drawTimetable(cv, rows, o = {}) {
  const narrow = narrowOf(cv, 560);
  const blocks = [], due = [];
  rows.forEach(r => {
    if (r.setup) blocks.push({ lane: 0, start: r.S - r.setup, end: r.S, kind: "setup" });
    blocks.push({ lane: 0, start: r.S, end: r.C, colour: colourOf(r.name), label: r.name, kind: "work" });
  });
  if (o.perOrder) NAMES.forEach((n, i) => {
    const r = rows.find(x => x.name === n), lane = i + 1;
    if (r.T > 0) blocks.push({ lane, start: r.d, end: r.C, colour: FAM.down, kind: "down" });   /* late: shaded behind */
    blocks.push({ lane, start: r.r, end: r.C, colour: alpha("--muted", 0.55), kind: "agv" });
    blocks.push({ lane, start: r.S, end: r.C, colour: colourOf(n), kind: "agv" });
    due.push({ lane, t: r.d });
  });
  const lanes = o.perOrder ? 5 : 1;
  const labels = o.perOrder ? ["machine", ...NAMES.map(n => narrow ? n : "order " + n)] : ["machine"];
  const g = drawGantt(cv, blocks, { lanes, t0: 0, t1: o.t1 || 130, laneLabels: labels, labelWidth: narrow ? 58 : 82,
                                    laneHeight: o.perOrder ? 32 : 34, due });
  if (o.perOrder) {   /* due-date labels, above the order's bar, on a surface-coloured backing */
    const ctx = cv.getContext("2d");
    ctx.font = "9px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "top";
    NAMES.forEach((n, i) => {
      const r = rows.find(x => x.name === n), txt = "d=" + r.d, tw = ctx.measureText(txt).width;
      const x = Math.min(g.X(r.d) + 3, g.w - tw - 4), y = (i + 1) * g.laneH + 1;
      ctx.fillStyle = css("--surface"); ctx.fillRect(x - 1, y - 1, tw + 3, 10);
      ctx.fillStyle = css("--ink"); ctx.fillText(txt, x, y);
    });
  }
  return g;
}

/* ---------------- hero ---------------- */
function chips() {
  const box = $("#seqChips");
  box.innerHTML = st.seq.map((n, i) => `<span class="chip" draggable="true" data-i="${i}" style="--c:${colourOf(n)}">` +
    `<button class="btn mv" data-i="${i}" data-d="-1" aria-label="Move ${n} earlier" ${i === 0 ? "disabled" : ""}>&#8592;</button>` +
    `<b>${n}</b><button class="btn mv" data-i="${i}" data-d="1" aria-label="Move ${n} later" ${i === 3 ? "disabled" : ""}>&#8594;</button></span>`).join("");
  $$(".mv", box).forEach(b => b.addEventListener("click", () => {
    const i = +b.dataset.i, j = i + +b.dataset.d; [st.seq[i], st.seq[j]] = [st.seq[j], st.seq[i]];
    st.preset = null; drawAllLinked(); const nb = $(`.mv[data-i="${j}"][data-d="${b.dataset.d}"]`, box) || $(`.mv[data-i="${j}"]`, box); if (nb && !nb.disabled) nb.focus();
  }));
  $$(".chip", box).forEach(c => {
    c.addEventListener("dragstart", e => { e.dataTransfer.setData("text/plain", c.dataset.i); c.classList.add("dragging"); });
    c.addEventListener("dragend", () => c.classList.remove("dragging"));
    c.addEventListener("dragover", e => e.preventDefault());
    c.addEventListener("drop", e => {
      e.preventDefault(); const from = +e.dataTransfer.getData("text/plain"), to = +c.dataset.i;
      if (isNaN(from) || from === to) return;
      const [x] = st.seq.splice(from, 1); st.seq.splice(to, 0, x); st.preset = null; drawAllLinked();
    });
  });
}
function presets() {
  $$("#presets .btn").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.p === st.preset)));
}
function drawHero() {
  const R = rel(), rows = schedule(st.seq, R), m = measures(rows), list = allMeasures(R), key = st.seq.join("");
  chips(); presets();
  drawTimetable($("#heroG"), rows, { perOrder: true });
  const order = NAMES.map(n => rows.find(r => r.name === n));
  $("#heroT").innerHTML = `<thead><tr><th>Order</th><th>colour</th><th><span class="m">r_j</span></th><th><span class="m">p_j</span></th><th><span class="m">d_j</span></th><th><span class="m">S_j</span></th><th><span class="m">C_j</span></th><th><span class="m">L_j</span></th><th><span class="m">T_j</span></th><th><span class="m">F_j</span></th></tr></thead><tbody>` +
    order.map(r => `<tr><td>${r.name}</td><td>${BY[r.name].family}</td><td>${r.r}</td><td>${r.p}</td><td>${r.d}</td><td>${r.S}</td><td>${r.C}</td><td>${r.L}</td><td>${r.T ? "<b>" + r.T + "</b>" : 0}</td><td>${r.F}</td></tr>`).join("") + "</tbody>";
  window.Kit.renderMath($("#heroT"));
  const set = (id, v) => { $(id).textContent = v; };
  set("#rCmax", m.Cmax); set("#rSumC", m.sumC); set("#rSumF", m.sumF); set("#rLmax", m.Lmax); set("#rTmax", m.Tmax);
  set("#rSumT", m.sumT); set("#rMeanT", fmt(m.meanT, 2)); set("#rU", `${m.sumU} of 4`); set("#rSetup", `${m.setups} min (${m.changes})`);
  set("#rRank", `${rankBy(list, "sumT", key)} of 24`);
  const late = order.filter(r => r.T > 0), idle = rows.filter(r => r.ready > r.free);
  const sumR = order.reduce((s, r) => s + r.r, 0), idleMin = m.Cmax - 70 - m.setups;
  caption($("#heroStatus"), `Sequence <b>${key}</b>. ` + rows.map(r => `${r.name} ${r.setup ? `changes colour ${r.S - r.setup}–${r.S}, then ` : ""}paints ${r.S}–${r.C}`).join("; ") + ". " +
    (idle.length ? `The machine waits for ${idle.map(r => `${r.name} (arrives at ${r.r})`).join(", ")}: an order cannot start before it is released. ` : "") +
    `Tardiness [[\\sum_j T_j = ${order.map(r => r.T).join(" + ")} = ${m.sumT}]]` + (late.length ? `, from ${late.map(r => `${r.name} (done at ${r.C}, due ${r.d})`).join(", ")}` : "") +
    `; mean [[\\bar T = ${m.sumT}/4 = ${fmt(m.meanT, 2)}]]. Makespan [[C_{\\max} = ${m.Cmax} = 70 + ${m.setups}${idleMin ? " + " + idleMin : ""}]]: the painting time 70 is fixed; only ${m.changes === 1 ? "the one colour change" : "the " + m.changes + " colour changes"}${idleMin ? " and the " + idleMin + " idle minutes" : ""} ${m.changes === 1 && !idleMin ? "moves" : "move"} it. ` +
    `Flow times sum to [[\\sum F_j = \\sum C_j - \\sum r_j = ${m.sumC} - ${sumR} = ${m.sumF}]]. This order is number ${rankBy(list, "sumT", key)} of 24 for total tardiness and number ${rankBy(list, "sumC", key)} for total completion time.`);
}
$$("#presets .btn").forEach(b => b.addEventListener("click", () => {
  const p = b.dataset.p, list = allMeasures(rel());
  st.seq = p === "bestT" ? bestBy(list, "sumT").seq.slice() : p === "bestC" ? bestBy(list, "sumC").seq.slice() : ruleSeq(p, rel());
  st.preset = p; drawAllLinked();
}));
$("#stagger").addEventListener("change", e => {
  st.stagger = e.target.checked;
  if (st.preset) { const list = allMeasures(rel()); st.seq = st.preset === "bestT" ? bestBy(list, "sumT").seq.slice() : st.preset === "bestC" ? bestBy(list, "sumC").seq.slice() : ruleSeq(st.preset, rel()); }
  drawAllLinked();
});

/* ---------------- widget 2: all 24 sequences on four measures ---------------- */
const ROWS = [["Cmax", "makespan Cmax"], ["sumC", "total completion ΣC"], ["Lmax", "worst lateness Lmax"], ["sumT", "total tardiness ΣT"]];
let allHits = [];
function drawAll24() {
  const cv = $("#allC"), narrow = narrowOf(cv, 560), R = rel(), list = allMeasures(R), key = st.seq.join("");
  const rowH = narrow ? 74 : 64, { ctx, w } = fit(cv, cv.clientWidth / (ROWS.length * rowH + 8));
  const left = 12, right = w - 12;
  allHits = [];
  ROWS.forEach(([k, label], ri) => {
    const y0 = ri * rowH + 4, vals = list.map(x => x.m[k]), lo = Math.min(...vals), hi = Math.max(...vals);
    const X = v => left + 8 + (hi === lo ? 0.5 : (v - lo) / (hi - lo)) * (right - left - 16);
    ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(label, left, y0);
    ctx.textAlign = "right"; ctx.fillText(`${lo} … ${hi}`, right, y0);
    const base = y0 + rowH - 16;
    ctx.strokeStyle = css("--rule"); ctx.beginPath(); ctx.moveTo(left, base + 0.5); ctx.lineTo(right, base + 0.5); ctx.stroke();
    const stack = {}, best = bestBy(list, k).m[k];
    list.slice().sort((a, b) => (a.key === key) - (b.key === key)).forEach(x => {
      const v = x.m[k], n = stack[v] = (stack[v] || 0) + 1, px = X(v), py = base - 5 - (n - 1) * 7;
      const cur = x.key === key;
      ctx.beginPath(); ctx.arc(px, cur ? base - 5 : py, cur ? 6 : 3.2, 0, 2 * Math.PI);
      ctx.fillStyle = cur ? css("--accent") : v === best ? css("--warm") : alpha("--ink", 0.35); ctx.fill();
      if (cur) { ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1; }
      allHits.push({ x: px, y: cur ? base - 5 : py, key: x.key, seq: x.seq });
    });
    const cv0 = list.find(x => x.key === key).m[k];
    ctx.fillStyle = css("--accent"); ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.font = "600 11px " + css("--font-mono");
    ctx.fillText(String(cv0), Math.min(right - 12, Math.max(left + 12, X(cv0))), base + 3);
  });
  const ranks = ROWS.map(([k]) => rankBy(list, k, key)), bT = bestBy(list, "sumT"), bC = bestBy(list, "Cmax"), bS = bestBy(list, "sumC");
  caption($("#allCap"), `Each dot is one of the 24 orders of A, B, C, D${st.stagger ? " (arriving five minutes apart)" : ""}; equal values stack. <b>Blue</b>: the hero's sequence ${key}, ranked ${ranks[0]}, ${ranks[1]}, ${ranks[2]} and ${ranks[3]} of 24 on the four measures (ties broken alphabetically). <b>Orange</b>: the best value on each row. ` +
    `The best order for total tardiness, ${bT.key}, has [[\\sum T_j = ${bT.m.sumT}]] but makespan ${bT.m.Cmax}; the best for total completion time, ${bS.key}, has [[\\sum C_j = ${bS.m.sumC}]] but [[\\sum T_j = ${bS.m.sumT}]]; ${bC.key} reaches the shortest makespan, ${bC.m.Cmax}, with [[\\sum T_j = ${bC.m.sumT}]]. ` +
    (list.every(x => x.m.Cmax === 70 + 10 * x.m.changes)
      ? `The makespan takes only ${new Set(list.map(x => x.m.Cmax)).size} values, because it is the painting time 70 plus 10 per colour change. `
      : `The makespan takes ${new Set(list.map(x => x.m.Cmax)).size} values: the painting time 70, plus 10 per colour change, plus any minutes the machine waits for an order to arrive. `) +
    `Click a dot to load that sequence into the hero.`);
}
$("#allC").addEventListener("click", e => {
  const b = e.target.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
  let best = null, bd = 12 * 12;
  for (const h of allHits) { const d = (h.x - x) ** 2 + (h.y - y) ** 2; if (d < bd) { bd = d; best = h; } }
  if (best) { st.seq = best.seq.slice(); st.preset = null; drawAllLinked(); }
});

/* ---------------- widget 3: a hand-made timetable and a small checker ---------------- */
const fz = { S: { A: 0, B: 30, C: 50, D: 75 }, stagger: false, meas: false };
function checkTimetable(S, R) {
  const order = NAMES.slice().sort((a, b) => S[a] - S[b] || (a < b ? -1 : 1));
  const errs = [], rows = [];
  let prevEnd = 0, prevFam = PAINT.start, prevName = null;
  order.forEach(n => {
    const o = BY[n], s = setupFor(prevFam, o.family), C = S[n] + o.p, setupStart = S[n] - s;
    /* one message per order: the binding side of S - s >= max(C_prev, r) */
    if (setupStart < Math.max(prevEnd, R[n])) {
      const what = s ? `its ${s}-minute colour change` : "its painting";
      if (prevName != null && prevEnd >= R[n])
        errs.push(s ? `${n} starts painting at ${S[n]}, so ${what} from ${BY[prevName].family} would start at ${setupStart}, while ${prevName} is still painting until ${prevEnd} (capacity and changeover).`
                    : `${n} starts at ${S[n]}, while ${prevName} is still painting until ${prevEnd}: two orders on one machine at once (capacity).`);
      else if (R[n] > 0) errs.push(`${n}: ${what} would start at ${setupStart}, but the order only arrives at ${R[n]} (release).`);
      else errs.push(`${n}: ${what} would start at ${setupStart}, before the day starts at minute 0.`);
    }
    rows.push({ name: n, r: R[n], p: o.p, d: o.due, setup: s, free: prevEnd, ready: setupStart, S: S[n], C,
                L: C - o.due, T: Math.max(0, C - o.due), E: Math.max(0, o.due - C), F: C - R[n], U: C > o.due ? 1 : 0 });
    prevEnd = Math.max(prevEnd, C); prevFam = o.family; prevName = n;
  });
  return { order, errs, rows };
}
function drawFeas() {
  const R = fz.stagger ? REL.stagger : REL.none;
  NAMES.forEach(n => { $("#fz" + n).value = fz.S[n]; });
  const { order, errs, rows } = checkTimetable(fz.S, R), m = measures(rows);
  const tight = measures(schedule(order, R));
  const g = drawTimetable($("#feasC"), rows, { perOrder: false, t1: 140 });
  const ctx = $("#feasC").getContext("2d");
  if (fz.stagger) { ctx.fillStyle = css("--muted"); ctx.font = "10px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "top";
    NAMES.forEach(n => { ctx.fillText("r" + n, g.X(R[n]), 1); }); }
  const valid = errs.length === 0;
  const slack = valid ? schedule(order, R).filter((r, i) => r.S < rows[i].S).map((r, i) => r.name) : [];
  $("#feasErr").innerHTML = valid ? `<b>Valid.</b> ` + (slack.length ? `Not as early as possible: ${slack.join(", ")} could start earlier in the same order.` : `And as early as possible for the order ${order.join("")}${fz.meas ? " (semi-active)" : ""}.`)
    : `<b>Not valid</b> (${errs.length} violation${errs.length > 1 ? "s" : ""}):<br>` + errs.map(e => "· " + e).join("<br>");
  const fin = NAMES.map(n => `${n} at ${rows.find(r => r.name === n).C}`).join(", ");
  caption($("#feasCap"), `Order on the machine, read from the start times: <b>${order.join("")}</b>. ` +
    (!valid ? (fz.meas ? `A measure computed on an invalid timetable means nothing: the repository's checker flags such a run, and no result is published.` : ``)
    : !fz.meas ? `The orders finish ${fin}. Tick "show the measures" once you have read Parts 3 and 4.`
    : `Measures of this timetable: [[\\sum T_j = ${m.sumT}]], [[\\sum C_j = ${m.sumC}]], [[C_{\\max} = ${m.Cmax}]], [[L_{\\max} = ${m.Lmax}]], total earliness [[\\sum e_j = ${m.sumE}]]. ` +
      (slack.length ? `The same order pushed as early as possible gives [[\\sum T_j = ${tight.sumT}]], [[\\sum C_j = ${tight.sumC}]], [[C_{\\max} = ${tight.Cmax}]], [[L_{\\max} = ${tight.Lmax}]], [[\\sum e_j = ${tight.sumE}]]: every regular measure is at least as good, while total earliness, which is not regular, ${tight.sumE > m.sumE ? "grows" : "stays the same"}.`
                    : `No order can start earlier, so no regular measure can be improved without changing the order.`)));
}
NAMES.forEach(n => $("#fz" + n).addEventListener("input", e => { const v = parseInt(e.target.value, 10); if (!isNaN(v)) { fz.S[n] = Math.max(-20, Math.min(200, v)); drawFeas(); } }));
$("#fzShift").addEventListener("click", () => {
  const R = fz.stagger ? REL.stagger : REL.none, { order } = checkTimetable(fz.S, R);
  schedule(order, R).forEach(r => { fz.S[r.name] = r.S; }); drawFeas();
});
$("#fzHero").addEventListener("click", () => { fz.stagger = st.stagger; $("#fzStagger").checked = fz.stagger; schedule(st.seq, rel()).forEach(r => { fz.S[r.name] = r.S; }); drawFeas(); });
$("#fzDelay").addEventListener("click", () => { const sB = fz.S.B; NAMES.forEach(n => { if (fz.S[n] >= sB) fz.S[n] += 5; }); drawFeas(); });
$("#fzStagger").addEventListener("change", e => { fz.stagger = e.target.checked; drawFeas(); });
$("#fzMeas").addEventListener("change", e => { fz.meas = e.target.checked; drawFeas(); });

/* ---------------- widget 4: the benchmark, four measures ---------------- */
/* Means over the 20 test days of results/runs.csv: [makespan, utilisation %, mean tardiness, mean flow time],
   printed by a01_numbers.py ("demo data"). */
const RUNS = {
  "no breakdowns": { FIFO: [1351.1, 60.221, 15.542, 157.437], SPT: [1348.25, 60.288, 9.989, 147.213], EDD: [1352.0, 60.061, 7.015, 145.583], SST: [1339.45, 60.699, 9.352, 141.248], DQN: [1344.7, 60.389, 7.446, 144.033], "Greedy replan": [1356.35, 59.963, 3.072, 133.039], "CP-SAT": [1341.35, 60.425, 1.333, 124.928] },
  "frequent breakdowns": { FIFO: [1430.2, 56.908, 60.836, 227.008], SPT: [1433.6, 56.609, 45.437, 208.977], EDD: [1432.95, 56.676, 37.631, 202.859], SST: [1395.9, 58.317, 34.149, 191.033], DQN: [1412.5, 57.83, 35.422, 198.777], "Greedy replan": [1416.5, 57.27, 24.029, 183.126], "CP-SAT": [1415.35, 57.161, 19.324, 177.145] },
};
const MEAS = [["makespan (min)", 0, 1], ["utilisation (%)", 1, 1], ["mean flow time (min)", 3, 1], ["mean tardiness (min)", 2, 1]];
let runLevel = "no breakdowns";
function drawRuns() {
  const cv = $("#runsC"), narrow = narrowOf(cv, 620), data = RUNS[runLevel], methods = Object.keys(data);
  const cols = narrow ? 1 : 2, panelH = methods.length * 17 + 34, rowsN = Math.ceil(MEAS.length / cols);
  const { ctx, w } = fit(cv, cv.clientWidth / (rowsN * panelH + 6));
  const pw = w / cols, lab = 96;
  const ratios = [];
  MEAS.forEach(([title, idx, dec], k) => {
    const x0 = (k % cols) * pw, y0 = Math.floor(k / cols) * panelH, vals = methods.map(m => data[m][idx]);
    const hi = Math.max(...vals), lo = Math.min(...vals), W = pw - lab - 50;
    ratios.push(hi / lo);
    ctx.fillStyle = css("--ink"); ctx.font = "600 11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(title, x0 + 4, y0 + 2);
    ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "right";
    ctx.fillText(`max/min ${fmt(hi / lo, 2)}`, x0 + pw - 8, y0 + 2);
    methods.forEach((m, i) => {
      const y = y0 + 22 + i * 17, v = data[m][idx], bw = v / hi * W;
      ctx.fillStyle = css("--muted"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.font = "11px " + css("--font-mono");
      ctx.fillText(m === "Greedy replan" ? "Greedy" : m, x0 + lab - 6, y + 6);
      ctx.fillStyle = m === "CP-SAT" ? css("--accent") : v === lo ? css("--warm") : alpha("--ink", 0.3);
      ctx.fillRect(x0 + lab, y, bw, 12);
      ctx.fillStyle = css("--ink"); ctx.textAlign = "left";
      ctx.fillText(fmt(v, dec), x0 + lab + bw + 4, y + 6);
    });
  });
  $$("#runsLv .btn").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.l === runLevel)));
  const ms = methods.map(m => data[m][0]), mt = methods.map(m => data[m][2]);
  const bestMs = methods[ms.indexOf(Math.min(...ms))], bestMt = methods[mt.indexOf(Math.min(...mt))];
  const pct = r => fmt(100 * (r - 1), 1) + " %";
  caption($("#runsCap"), `<b>${runLevel[0].toUpperCase() + runLevel.slice(1)}</b>, mean over the 20 test days; bars start at zero, orange marks the smallest value of each panel (for utilisation, higher is not better: pitfall 3). The highest mean makespan is ${pct(ratios[0])} above the lowest, the highest utilisation ${pct(ratios[1])} above the lowest: on these two measures every method looks the same. Flow time separates them a little (${pct(ratios[2])}), mean tardiness a lot (a factor of ${fmt(ratios[3], 2)}). ` +
    `The shortest makespan belongs to ${bestMs}, the lowest tardiness to ${bestMt} (blue): the two measures do not even agree on the winner.`);
}
$$("#runsLv .btn").forEach(b => b.addEventListener("click", () => { runLevel = b.dataset.l; drawRuns(); }));

function drawAllLinked() { drawHero(); drawAll24(); }
function drawEverything() { drawAllLinked(); drawFeas(); drawRuns(); }
drawEverything();
onRedraw(drawEverything);
})();
