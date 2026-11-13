/* Which Job Should the Machine Do Next?, chapter A3, the flexible job shop: the demos.
   Hero: the disjunctive graph of a 3-job, 3-machine job shop. The reader orients each
   disjunctive pair; cycles are detected, and for an acyclic selection the longest path, the
   makespan and the semi-active Gantt chart follow. Second widget: the flexible version, where
   three operations may change machine and every order on every machine is enumerated. Third
   widget: how many assignments and orders a day of the course's workshop has.
   Same computations as code/a03_numbers.py. Needs ../kit/kit.js and jobshop-lib.js. */
(() => {
"use strict";
const { $, $$, css, caption, frame, plotLine, onRedraw } = window.Kit;
const { WORKSHOP, FAM, drawGantt, rng } = window.JS;

/* ---------------- the instance: operation (j, k) = [machine, minutes] ---------------- */
const TOY = [
  [[0, 2], [1, 7], [2, 7]],   // job 0: M0 -> M1 -> M2
  [[1, 7], [2, 4], [0, 5]],   // job 1: M1 -> M2 -> M0
  [[2, 2], [0, 6], [1, 2]],   // job 2: M2 -> M0 -> M1
];
const NM = 3;
const nm = o => `J${o[0]}.${o[1]}`;
const opsOf = jobs => jobs.flatMap((r, j) => r.map((_, k) => [j, k]));
const id = (o) => o[0] * 10 + o[1];

/* the disjunctive pairs, in the order of the numbers script: per machine, (u, v) with u before v in (j, k) order,
   operations of different jobs only (two steps of one job on one machine are already ordered by its route) */
function pairsOf(jobs) {
  const ops = opsOf(jobs), out = [];
  for (let m = 0; m < NM; m++) {
    const on = ops.filter(o => jobs[o[0]][o[1]][0] === m);
    for (let a = 0; a < on.length; a++) for (let b = a + 1; b < on.length; b++) if (on[a][0] !== on[b][0]) out.push([on[a], on[b]]);
  }
  return out;
}

/* Earliest starts for the arcs chosen so far (arcs: [[u, v], ...] meaning u before v on their machine).
   Arc weights as on the page: source -> (j,0): tau + sigma; (j,k) -> (j,k+1): p + tau + sigma;
   u -> v: p_u + s; sigma_v = s when an arc of the selection enters v. Returns
   {start, sig, cycle (list of operations) or null, cmax, crit (operations), critArcs (Set of "u>v")}. */
function evaluate(jobs, arcs, tau = 0, s = 0) {
  const ops = opsOf(jobs), P = o => jobs[o[0]][o[1]][1];
  const hasPred = new Set(arcs.map(a => id(a[1])));
  const sig = {}; ops.forEach(o => { sig[id(o)] = hasPred.has(id(o)) ? s : 0; });
  const succ = {}, pred = {};
  ops.forEach(o => { succ[id(o)] = []; pred[id(o)] = []; });
  const add = (u, v, w) => { succ[id(u)].push([v, w]); pred[id(v)].push([u, w]); };
  jobs.forEach((r, j) => r.forEach((_, k) => { if (k + 1 < r.length) add([j, k], [j, k + 1], P([j, k]) + tau + sig[id([j, k + 1])]); }));
  arcs.forEach(([u, v]) => add(u, v, P(u) + s));
  const indeg = {}; ops.forEach(o => { indeg[id(o)] = pred[id(o)].length; });
  const start = {}; ops.forEach(o => { start[id(o)] = o[1] === 0 ? tau + sig[id(o)] : 0; });
  const queue = ops.filter(o => indeg[id(o)] === 0), done = new Set();
  for (let i = 0; i < queue.length; i++) {
    const u = queue[i]; done.add(id(u));
    for (const [v, w] of succ[id(u)]) {
      start[id(v)] = Math.max(start[id(v)], start[id(u)] + w);
      if (--indeg[id(v)] === 0) queue.push(v);
    }
  }
  if (done.size < ops.length) {
    /* every operation left has a predecessor that is also left: walk backwards until one repeats */
    let v = ops.find(o => !done.has(id(o)));
    const seen = new Map(), walk = [];
    while (!seen.has(id(v))) { seen.set(id(v), walk.length); walk.push(v); v = pred[id(v)].map(x => x[0]).find(u => !done.has(id(u))); }
    const cyc = walk.slice(seen.get(id(v))).reverse();   // forward order
    return { start, sig, cycle: cyc, cmax: null, crit: [], critArcs: new Set() };
  }
  let cmax = 0, last = null;
  jobs.forEach((r, j) => { const o = [j, r.length - 1], e = start[id(o)] + P(o); if (e > cmax) { cmax = e; last = o; } });
  const crit = [last], critArcs = new Set();
  for (;;) {
    const v = crit[crit.length - 1], t = pred[id(v)].find(([u, w]) => start[id(u)] + w === start[id(v)]);
    if (!t) break;
    critArcs.add(id(t[0]) + ">" + id(v)); crit.push(t[0]);
  }
  crit.reverse();
  return { start, sig, cycle: null, cmax, crit, critArcs };
}
const arcsOf = (pairs, sel) => pairs.flatMap((p, i) => sel[i] == null ? [] : [sel[i] ? [p[0], p[1]] : [p[1], p[0]]]);

/* every complete selection of a job shop: how many are acyclic, and the best makespan */
function enumerate(jobs, tau, s) {
  const pairs = pairsOf(jobs), n = pairs.length;
  let acyclic = 0, best = Infinity, bestSel = null;
  for (let mask = 0; mask < (1 << n); mask++) {
    const sel = pairs.map((_, i) => ((mask >> (n - 1 - i)) & 1) ? 0 : 1);
    const r = evaluate(jobs, arcsOf(pairs, sel), tau, s);
    if (r.cycle) continue;
    acyclic++;
    if (r.cmax < best) { best = r.cmax; bestSel = sel; }
  }
  return { total: 1 << n, acyclic, best, bestSel };
}

/* ---------------- the disjunctive graph as SVG ---------------- */
/* Two layouts: wide (jobs as rows, steps left to right) and, for a holder narrower than NARROW px,
   tall (jobs as columns, steps top to bottom), so that labels keep their size on a phone. */
const NARROW = 540, NW = 66, NH = 42;
const LAYOUTS = {
  wide: { w: 600, h: 330, pos: o => ({ x: [130, 300, 470][o[1]], y: [55, 165, 275][o[0]] }), src: { x: 26, y: 165 }, snk: { x: 574, y: 165 } },
  tall: { w: 340, h: 540, pos: o => ({ x: [58, 170, 282][o[0]], y: [118, 270, 422][o[1]] }), src: { x: 170, y: 24 }, snk: { x: 170, y: 516 } },
};
const svgEl = (tag, attrs, text) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e; };
const qpt = (a, c, b, t) => ({ x: (1 - t) * (1 - t) * a.x + 2 * t * (1 - t) * c.x + t * t * b.x, y: (1 - t) * (1 - t) * a.y + 2 * t * (1 - t) * c.y + t * t * b.y });
function bowFor(a, b, avoid) {
  const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy), nx = -dy / L, ny = dx / L;
  for (const f of [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.42, -0.42]) {
    const c = { x: (a.x + b.x) / 2 + nx * f * L, y: (a.y + b.y) / 2 + ny * f * L };
    let ok = true;
    for (let i = 1; i < 30 && ok; i++) { const p = qpt(a, c, b, i / 30); for (const q of avoid) if (Math.abs(p.x - q.x) < NW / 2 + 6 && Math.abs(p.y - q.y) < NH / 2 + 6) { ok = false; break; } }
    if (ok) return c;
  }
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}
function arrowHead(g, p, dir, cls, size = 9) {
  const L = Math.hypot(dir.x, dir.y) || 1, ux = dir.x / L, uy = dir.y / L;
  const pts = [[p.x + ux * size, p.y + uy * size], [p.x - ux * size * 0.6 - uy * size * 0.62, p.y - uy * size * 0.6 + ux * size * 0.62], [p.x - ux * size * 0.6 + uy * size * 0.62, p.y - uy * size * 0.6 - ux * size * 0.62]];
  g.appendChild(svgEl("polygon", { points: pts.map(q => q.join(",")).join(" "), class: cls }));
}
/* Draw the graph. pairs/sel: disjunctive pairs and their orientation (null = unordered);
   res: an evaluate() result for highlighting; onPick(i): click on pair i (omit for a static graph). */
function drawGraph(holder, jobs, pairs, sel, res, { tau = 0, s = 0, onPick = null, label } = {}) {
  const LY = (holder.clientWidth || 600) < NARROW ? LAYOUTS.tall : LAYOUTS.wide, pos = LY.pos, SRC = LY.src, SNK = LY.snk;
  const svg = svgEl("svg", { viewBox: `0 0 ${LY.w} ${LY.h}`, role: "img", "aria-label": label || "Disjunctive graph" });
  if (LY === LAYOUTS.tall) svg.style.maxWidth = "430px";
  svg.style.marginInline = "auto";
  const gArcs = svgEl("g", {}), gLab = svgEl("g", { "pointer-events": "none" }), gHit = svgEl("g", {}), gNodes = svgEl("g", {});
  /* a weight label sitting on its arc, over a small backing so that crossing strokes do not hide it */
  const weightLabel = (x, y, text, cls = "wlab") => {
    const wd = 7.4 * String(text).length + 6;
    gLab.appendChild(svgEl("rect", { x: x - wd / 2, y: y - 8, width: wd, height: 16, rx: 4, class: "wbg" }));
    gLab.appendChild(svgEl("text", { x, y: y + 4, class: cls }, text));
  };
  const ops = opsOf(jobs), nodes = ops.map(pos).concat([SRC, SNK]);
  const cycSet = new Set(), cycArcs = new Set();
  if (res && res.cycle) { res.cycle.forEach((o, i) => { cycSet.add(id(o)); cycArcs.add(id(o) + ">" + id(res.cycle[(i + 1) % res.cycle.length])); }); }
  const critSet = new Set(res && !res.cycle ? res.crit.map(id) : []);
  const P = o => jobs[o[0]][o[1]][1];
  /* conjunctive arcs, source and sink arcs */
  const straight = (a, b, cls, w, critical, cyc = false) => {
    const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
    const trimA = a === SRC ? 12 : Math.min(NW / 2 / Math.abs(ux || 1e-9), NH / 2 / Math.abs(uy || 1e-9));
    const trimB = b === SNK ? 12 : Math.min(NW / 2 / Math.abs(ux || 1e-9), NH / 2 / Math.abs(uy || 1e-9));
    const p0 = { x: a.x + ux * trimA, y: a.y + uy * trimA }, p1 = { x: b.x - ux * (trimB + 8), y: b.y - uy * (trimB + 8) };
    gArcs.appendChild(svgEl("line", { x1: p0.x, y1: p0.y, x2: p1.x, y2: p1.y, class: cls + (critical ? " crit" : "") }));
    arrowHead(gArcs, { x: p1.x, y: p1.y }, { x: ux, y: uy }, "head" + (critical ? " crit" : cyc ? " cyc" : ""), critical || cyc ? 10 : 8);
    if (w != null) weightLabel((p0.x + p1.x) / 2, (p0.y + p1.y) / 2, w);
  };
  const sigOf = o => (res ? res.sig[id(o)] : 0);
  const critFirst = res && !res.cycle && res.crit.length ? id(res.crit[0]) : null;
  const critLast = res && !res.cycle && res.crit.length ? id(res.crit[res.crit.length - 1]) : null;
  jobs.forEach((r, j) => {
    const first = [j, 0], wS = tau + sigOf(first);
    straight(SRC, pos(first), "conj", wS ? String(wS) : null, critFirst === id(first));
    r.forEach((_, k) => {
      if (k + 1 < r.length) {
        const a = [j, k], b = [j, k + 1], w = P(a) + tau + sigOf(b);
        const inCycle = cycArcs.has(id(a) + ">" + id(b));
        straight(pos(a), pos(b), inCycle ? "conj cyc" : "conj", String(w), res && !res.cycle && res.critArcs.has(id(a) + ">" + id(b)), inCycle);
      }
    });
    const lastOp = [j, r.length - 1];
    straight(pos(lastOp), SNK, "conj", String(P(lastOp)), critLast === id(lastOp));
  });
  /* disjunctive pairs */
  pairs.forEach(([u, v], i) => {
    const a = pos(u), b = pos(v), avoid = nodes.filter(q => q !== a && q !== b && !(q.x === a.x && q.y === a.y) && !(q.x === b.x && q.y === b.y));
    const c = bowFor(a, b, avoid), d = `M${a.x},${a.y} Q${c.x},${c.y} ${b.x},${b.y}`;
    const o = sel[i], fwd = o === 1, tail = fwd ? u : v, head = fwd ? v : u, key = id(tail) + ">" + id(head);
    let cls = "dis";
    if (o != null) cls += cycArcs.has(key) ? " cyc" : res && !res.cycle && res.critArcs.has(key) ? " crit" : " on";
    gArcs.appendChild(svgEl("path", { d, class: cls }));
    if (o != null) {
      const t0 = 0.56, p = qpt(fwd ? a : b, c, fwd ? b : a, t0), q = qpt(fwd ? a : b, c, fwd ? b : a, t0 + 0.02);
      arrowHead(gArcs, p, { x: q.x - p.x, y: q.y - p.y }, "head " + cls.replace("dis", "").trim(), 10);
      if (s) { const r = qpt(fwd ? a : b, c, fwd ? b : a, 0.36); weightLabel(r.x, r.y, "+" + s, "wlab sw"); }
    }
    if (onPick) {
      const hit = svgEl("path", { d, class: "hit", tabindex: "-1" });
      hit.appendChild(svgEl("title", {}, `${nm(u)} and ${nm(v)} on M${jobs[u[0]][u[1]][0]}: click to order`));
      hit.addEventListener("click", () => onPick(i));
      gHit.appendChild(hit);
    }
  });
  /* nodes */
  const node = (p, l1, l2, fill, ring) => {
    const g = svgEl("g", {});
    if (ring) g.appendChild(svgEl("rect", { x: p.x - NW / 2 - 4, y: p.y - NH / 2 - 4, width: NW + 8, height: NH + 8, rx: 12, class: ring }));
    g.appendChild(svgEl("rect", { x: p.x - NW / 2, y: p.y - NH / 2, width: NW, height: NH, rx: 9, fill }));
    g.appendChild(svgEl("text", { x: p.x, y: p.y - 3, class: "n1" }, l1));
    g.appendChild(svgEl("text", { x: p.x, y: p.y + 13, class: "n2" }, l2));
    gNodes.appendChild(g);
  };
  ops.forEach(o => node(pos(o), nm(o), `M${jobs[o[0]][o[1]][0]} · ${P(o)}`, FAM.colours[o[0]], cycSet.has(id(o)) ? "ring-cyc" : critSet.has(id(o)) ? "ring-crit" : null));
  [[SRC, "α"], [SNK, "ω"]].forEach(([p, l]) => { gNodes.appendChild(svgEl("circle", { cx: p.x, cy: p.y, r: 13, class: "st" })); gNodes.appendChild(svgEl("text", { x: p.x, y: p.y + 5, class: "stl" }, l)); });
  svg.append(gArcs, gLab, gHit, gNodes);
  holder.replaceChildren(svg);
}

/* Gantt chart of a schedule: one lane per machine, changeovers striped before the operation.
   split: while some pairs are unordered, operations of one machine may overlap, so each machine lane is
   split into one thin row per job and an overlap shows as two bars side by side in time. */
function ganttOf(canvas, jobs, res, s, t1, split = false) {
  const blocks = [], nj = jobs.length;
  const pxPerMin = Math.max(1, (canvas.clientWidth || 600) - 50) / t1;
  opsOf(jobs).forEach(o => {
    const m = jobs[o[0]][o[1]][0], st = res.start[id(o)], p = jobs[o[0]][o[1]][1], lane = split ? m * nj + o[0] : m;
    if (res.sig[id(o)]) blocks.push({ lane, start: st - res.sig[id(o)], end: st, kind: "setup" });
    const fits = p * pxPerMin >= 7 * nm(o).length + 4;   // drop the label of a bar narrower than its text
    blocks.push({ lane, start: st, end: st + p, colour: FAM.colours[o[0]], label: fits ? nm(o) : "", kind: "work" });
  });
  const laneLabels = split ? Array.from({ length: NM * nj }, (_, i) => (i % nj === (nj - 1) >> 1 ? "M" + Math.floor(i / nj) : "")) : ["M0", "M1", "M2"];
  const g = drawGantt(canvas, blocks, { lanes: split ? NM * nj : NM, laneHeight: split ? 16 : 48, t0: 0, t1, laneLabels, labelWidth: 40 });
  if (split) {   // thin rules between the machines
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = css("--rule"); ctx.lineWidth = 1;
    for (let m = 1; m < NM; m++) { const y = Math.round(m * nj * g.laneH) + 0.5; ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(g.w - 10, y); ctx.stroke(); }
  }
}

/* ================================ hero ================================ */
const HP = pairsOf(TOY);
const hero = { sel: HP.map(() => null), tau: 0, s: 0 };
const optCache = {};
const optFor = (tau, s) => (optCache[tau + "," + s] ||= enumerate(TOY, tau, s));

function buildPairButtons() {
  const box = $("#dgPairs");
  box.innerHTML = "";
  for (let m = 0; m < NM; m++) {
    const row = document.createElement("div"); row.className = "pairrow";
    const lab = document.createElement("span"); lab.className = "pm"; lab.textContent = "M" + m; row.appendChild(lab);
    HP.forEach(([u, v], i) => {
      if (TOY[u[0]][u[1]][0] !== m) return;
      const b = document.createElement("button"); b.className = "btn pb"; b.type = "button"; b.dataset.i = i;
      b.addEventListener("click", () => pick(i)); row.appendChild(b);
    });
    box.appendChild(row);
  }
}
function pick(i) { hero.sel[i] = hero.sel[i] == null ? 1 : 1 - hero.sel[i]; drawHero(); }

function drawHero() {
  const { tau, s, sel } = hero, arcs = arcsOf(HP, sel), res = evaluate(TOY, arcs, tau, s), opt = optFor(tau, s);
  const nOn = sel.filter(x => x != null).length;
  drawGraph($("#dgGraph"), TOY, HP, sel, res, { tau, s, onPick: pick, label: "Disjunctive graph of the three-job, three-machine instance; click a dashed pair to order it" });
  $$("#dgPairs .pb").forEach(b => {
    const i = +b.dataset.i, [u, v] = HP[i], o = sel[i];
    b.textContent = o == null ? `${nm(u)} · ${nm(v)}` : o ? `${nm(u)} → ${nm(v)}` : `${nm(v)} → ${nm(u)}`;
    b.setAttribute("aria-pressed", o != null ? "true" : "false");
    b.setAttribute("aria-label", `Pair ${nm(u)} and ${nm(v)}: ` + (o == null ? "not ordered" : o ? `${nm(u)} first` : `${nm(v)} first`));
  });
  const cv = $("#dgGantt");
  $("#rOn").textContent = `${nOn} of ${HP.length}`;
  $("#rOpt").textContent = `${opt.best} min`;
  $("#rAc").textContent = `${opt.acyclic} of ${opt.total}`;
  if (res.cycle) {
    drawGantt(cv, [], { lanes: NM, laneHeight: 48, t0: 0, t1: 45, laneLabels: ["M0", "M1", "M2"], labelWidth: 40 });
    $("#rCyc").textContent = "yes"; $("#rCmax").textContent = "none"; $("#rCrit").textContent = "–";
    const P = res.cycle.map(o => TOY[o[0]][o[1]][1]);
    caption($("#heroStatus"), `<b>A cycle: no schedule exists.</b> ${res.cycle.map(nm).join(" → ")} → ${nm(res.cycle[0])}. Each arc says "the head starts after the tail ends", so following the cycle gives [[S_{\\text{${nm(res.cycle[0])}}} \\ge S_{\\text{${nm(res.cycle[0])}}} + ${P.join(" + ")}]], impossible since the durations are positive. Flip one of the red arcs.`);
    return;
  }
  const t1 = Math.max(45, res.cmax + 2);
  ganttOf(cv, TOY, res, s, t1, nOn < HP.length);
  $("#rCyc").textContent = "none";
  $("#rCmax").textContent = (nOn < HP.length ? "≥ " : "") + res.cmax + " min";
  $("#rCrit").textContent = res.crit.map(nm).join(" → ");
  if (nOn === 0) {
    caption($("#heroStatus"), `No pair is ordered yet. Only the routes constrain the operations, so every job starts at once and the longest path is the longest job, ${res.cmax} min: a lower bound. Each machine lane is split into one row per job, so you can see operations of the same machine overlap. Click a dashed pair, in the graph or in the panel, to decide which of its two operations goes first.`);
  } else if (nOn < HP.length) {
    caption($("#heroStatus"), `${nOn} of ${HP.length} pairs ordered, no cycle so far. The chart is the earliest schedule that respects the arcs chosen so far; operations of an unordered pair may still overlap (each machine lane has one row per job, so overlaps show), so the longest path, ${res.cmax} min, is a lower bound on every completion of this selection.`);
  } else {
    const gap = res.cmax - opt.best;
    caption($("#heroStatus"), `<b>An acyclic selection: a schedule.</b> Each operation starts at the length of the longest path from α to it, which gives the semi-active schedule above, and the makespan is the longest path from α to ω: [[C_{\\max} = ${res.cmax}]] min, along ${res.crit.map(nm).join(" → ")}. ${gap === 0 ? "This is the optimum: no acyclic selection does better." : `The optimum is ${opt.best} min, ${gap} min shorter: to shorten this schedule, flip a pair on the highlighted critical path.`}`);
  }
}
function heroPreset(kind) {
  if (kind === "clear") hero.sel = HP.map(() => null);
  if (kind === "random") { const r = rng(Math.floor(Math.random() * 1e9)); hero.sel = HP.map(() => (r() < 0.5 ? 1 : 0)); }
  if (kind === "opt") hero.sel = optFor(hero.tau, hero.s).bestSel.slice();
  if (kind === "prio") hero.sel = HP.map(([u, v]) => (u[0] < v[0] ? 1 : 0));
  drawHero();
}
buildPairButtons();
$("#dgClear").addEventListener("click", () => heroPreset("clear"));
$("#dgRandom").addEventListener("click", () => heroPreset("random"));
$("#dgOpt").addEventListener("click", () => heroPreset("opt"));
$("#dgPrio").addEventListener("click", () => heroPreset("prio"));
$("#dgTau").addEventListener("change", e => { hero.tau = e.target.checked ? 1 : 0; drawHero(); });
$("#dgSet").addEventListener("change", e => { hero.s = e.target.checked ? 1 : 0; drawHero(); });

/* ================================ the flexible version ================================ */
const FLEX = [
  [{ 0: 2 }, { 1: 7, 2: 6 }, { 2: 7 }],
  [{ 1: 7, 2: 6 }, { 2: 4 }, { 0: 5 }],
  [{ 2: 2 }, { 0: 6 }, { 1: 2, 0: 3 }],
];
const FLEX_OPS = [[0, 1], [1, 0], [2, 2]];
const flex = { choice: [1, 1, 1] };   // the machines of the classical instance
function permutations(a) { if (a.length <= 1) return [a.slice()]; const out = []; a.forEach((x, i) => permutations([...a.slice(0, i), ...a.slice(i + 1)]).forEach(p => out.push([x, ...p]))); return out; }
function assignmentOf(choice) {
  const jobs = FLEX.map(r => r.map(alt => { const m = +Object.keys(alt)[0]; return [m, alt[m]]; }));
  FLEX_OPS.forEach((o, i) => { jobs[o[0]][o[1]] = [choice[i], FLEX[o[0]][o[1]][choice[i]]]; });
  return jobs;
}
function bestForAssignment(jobs) {
  const on = [0, 1, 2].map(m => opsOf(jobs).filter(o => jobs[o[0]][o[1]][0] === m));
  const perms = on.map(permutations);
  let n = 0, nac = 0, best = null;
  for (const p0 of perms[0]) for (const p1 of perms[1]) for (const p2 of perms[2]) {
    n++;
    const arcs = [p0, p1, p2].flatMap(seq => seq.slice(1).map((v, i) => [seq[i], v]));
    const r = evaluate(jobs, arcs);
    if (r.cycle) continue;
    nac++;
    if (!best || r.cmax < best.res.cmax) best = { res: r, orders: [p0, p1, p2] };
  }
  return { n, nac, best };
}
const flexAll = (() => {
  const out = [];
  for (const a of [1, 2]) for (const b of [1, 2]) for (const c of [0, 1]) {
    const jobs = assignmentOf([a, b, c]);
    out.push({ choice: [a, b, c], best: bestForAssignment(jobs).best.res.cmax });
  }
  return out;
})();
function buildFlexButtons() {
  const box = $("#fxChoice"); box.innerHTML = "";
  FLEX_OPS.forEach((o, i) => {
    const row = document.createElement("div"); row.className = "pairrow";
    const lab = document.createElement("span"); lab.className = "pm"; lab.textContent = nm(o); row.appendChild(lab);
    Object.entries(FLEX[o[0]][o[1]]).forEach(([m, p]) => {
      const b = document.createElement("button"); b.type = "button"; b.className = "btn pb"; b.dataset.i = i; b.dataset.m = m;
      b.textContent = `M${m} · ${p} min`;
      b.addEventListener("click", () => { flex.choice[i] = +m; drawFlex(); });
      row.appendChild(b);
    });
    box.appendChild(row);
  });
}
function drawFlex() {
  $$("#fxChoice .pb").forEach(b => b.setAttribute("aria-pressed", flex.choice[+b.dataset.i] === +b.dataset.m ? "true" : "false"));
  const jobs = assignmentOf(flex.choice), { n, nac, best } = bestForAssignment(jobs);
  /* the pairs of this assignment, oriented as in the best orders */
  const pairs = pairsOf(jobs), rank = {};
  best.orders.forEach(seq => seq.forEach((o, i) => { rank[id(o)] = i; }));
  const sel = pairs.map(([u, v]) => (rank[id(u)] < rank[id(v)] ? 1 : 0));
  const res = evaluate(jobs, arcsOf(pairs, sel));
  drawGraph($("#fxGraph"), jobs, pairs, sel, res, { label: "Disjunctive graph of the chosen assignment, oriented as in its best schedule" });
  ganttOf($("#fxGantt"), jobs, res, 0, 30);
  const load = [0, 1, 2].map(m => opsOf(jobs).filter(o => jobs[o[0]][o[1]][0] === m).reduce((t, o) => t + jobs[o[0]][o[1]][1], 0));
  const lb = Math.max(...load, ...jobs.map(r => r.reduce((t, x) => t + x[1], 0)));
  const gbest = Math.min(...flexAll.map(r => r.best)), nBest = flexAll.filter(r => r.best === gbest).length;
  const fastest = flex.choice.every((m, i) => FLEX[FLEX_OPS[i][0]][FLEX_OPS[i][1]][m] === Math.min(...Object.values(FLEX[FLEX_OPS[i][0]][FLEX_OPS[i][1]])));
  const classical = flex.choice.join() === "1,1,1";
  $("#fxLoad").textContent = load.join(", ");
  $("#fxPairs").textContent = pairs.length;
  $("#fxOrders").textContent = `${nac} of ${n}`;
  $("#fxBest").textContent = `${res.cmax} min`;
  caption($("#fxCap"), `${classical ? "<b>The classical instance of the hero.</b> " : fastest ? "<b>The fastest machine for every operation.</b> " : ""}Machine loads ${load.join(", ")} min, so no schedule can end before ${lb} min (the busiest machine, or the longest job). The assignment puts ${pairs.length} disjunctive pairs in the graph and leaves ${n} ways to order the machines, ${nac} of them acyclic; the best ends at <b>${res.cmax} min</b>${res.cmax === lb ? ", on the lower bound" : ""}. Over all 8 assignments the best is ${gbest} min, reached by ${nBest} of them.${fastest ? " Choosing the fastest machine everywhere piles five operations on M2: the shortest operations make the longest schedule." : ""}`);
}
buildFlexButtons();

/* ================================ counting the workshop's choices ================================ */
/* families of the 60 jobs of test day 0, in arrival order (make_scenario(make_workshop(), 0), printed by a03_numbers.py) */
const DAY0 = [0, 3, 3, 2, 1, 1, 0, 1, 2, 3, 3, 0, 0, 2, 3, 3, 0, 0, 2, 3, 2, 0, 0, 3, 0, 3, 2, 2, 1, 1, 1, 2, 0, 2, 2, 2, 1, 0, 2, 0, 3, 2, 3, 1, 1, 1, 1, 3, 3, 2, 3, 0, 1, 3, 3, 2, 0, 1, 0, 3];
const R = WORKSHOP.routings;
const perFamily = R.map(ops => ops.reduce((t, alt) => t * Object.keys(alt).length, 1));
const fastestMachine = (f, k) => Object.entries(R[f][k]).map(([m, p]) => [+m, p]).sort((a, b) => a[1] - b[1] || a[0] - b[0])[0][0];
const log10fact = n => { let s = 0; for (let i = 2; i <= n; i++) s += Math.log10(i); return s; };
/* as counts_for in the numbers script: pairs of different jobs only; orders that respect the routes =
   prod n_m! / 2^q, q = pairs of operations of one job on one machine (disjoint, each halves the count) */
const sameJob = f => { const c = [0, 0, 0, 0, 0]; R[f].forEach((_, k) => { c[fastestMachine(f, k)]++; }); return c.reduce((t, x) => t + x * (x - 1) / 2, 0); };
function countsFor(n) {
  const fams = DAY0.slice(0, n), nm5 = [0, 0, 0, 0, 0];
  fams.forEach(f => R[f].forEach((_, k) => { nm5[fastestMachine(f, k)]++; }));
  const same = fams.reduce((t, f) => t + sameJob(f), 0);
  const pairs = nm5.reduce((t, x) => t + x * (x - 1) / 2, 0) - same;
  const perm = nm5.reduce((t, x) => t + log10fact(x), 0);
  return { assign: fams.reduce((t, f) => t + Math.log10(perFamily[f]), 0), nm5, same, pairs, orient: pairs * Math.log10(2), perm, orders: perm - same * Math.log10(2) };
}
const COUNTS = Array.from({ length: 60 }, (_, i) => countsFor(i + 1));
function drawCount() {
  const n = +$("#cN").value, c = COUNTS[n - 1], cv = $("#cPlot");
  const fr = frame(cv, cv.clientWidth < 520 ? 1.25 : 2.2, { x: [1, 60], y: [0, 340], xlabel: "jobs of test day 0", ylabel: "log10 of the count", xticks: [1, 10, 20, 30, 40, 50, 60], yticks: [0, 100, 200, 300] });
  const xs = COUNTS.map((_, i) => i + 1);
  plotLine(fr, xs, COUNTS.map(r => r.assign), { color: css("--accent"), width: 2.2 });
  plotLine(fr, xs, COUNTS.map(r => r.orders), { color: css("--warm"), width: 2.2 });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--muted"); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(n), Y(0)); ctx.lineTo(X(n), Y(340)); ctx.stroke(); ctx.setLineDash([]);
  [[c.assign, "--accent"], [c.orders, "--warm"]].forEach(([v, t]) => { ctx.fillStyle = css(t); ctx.beginPath(); ctx.arc(X(n), Y(v), 4.5, 0, 2 * Math.PI); ctx.fill(); });
  ctx.font = "12px " + css("--font-ui"); ctx.textAlign = "left";
  ctx.fillStyle = css("--warm"); ctx.fillText("orders on the machines that respect the routes", fr.pad.l + 10, Y(275));
  ctx.fillStyle = css("--accent"); ctx.fillText("assignments, Π |M_jk|", fr.pad.l + 10, Y(250));
  $("#cNV").textContent = n;
  $("#cOps").textContent = 4 * n;
  $("#cAs").innerHTML = "10<sup>" + c.assign.toFixed(1) + "</sup>";
  $("#cNm").textContent = c.nm5.join(", ");
  $("#cPairs").textContent = c.pairs.toLocaleString("en-US");
  $("#cOr").innerHTML = "10<sup>" + c.orient.toFixed(c.orient < 100 ? 1 : 0) + "</sup>";
  $("#cOrd").innerHTML = "10<sup>" + c.orders.toFixed(1) + "</sup>";
  caption($("#cCap"), `The first ${n} job${n > 1 ? "s" : ""} of test day 0 have ${4 * n} operations. Choosing a machine for each gives [[\\prod |\\mathcal M_{jk}| \\approx 10^{${c.assign.toFixed(1)}}]] assignments. With every operation on its fastest machine the five machines receive ${c.nm5.join(", ")} operations, hence ${c.pairs.toLocaleString("en-US")} disjunctive pairs (${c.same} more pairs share a machine but belong to one job, already ordered by its route) and [[2^{${c.pairs}} \\approx 10^{${c.orient.toFixed(c.orient < 100 ? 1 : 0)}}]] orientations. Only those that order each machine as the routes allow can be acyclic: [[\\prod_m n_m! / 2^{${c.same}} \\approx 10^{${c.orders.toFixed(1)}}]] of them. These are counts of a static problem in which every job is present at once; release times rule most of them out in the real shop, and the planner of chapter <a href="c03-rescheduling-on-a-rolling-horizon.html">C3</a> never looks at more than 25 jobs.`);
}
$("#cN").addEventListener("input", drawCount);

function drawAll() { drawHero(); drawFlex(); drawCount(); }
drawAll();
onRedraw(drawAll);
})();
