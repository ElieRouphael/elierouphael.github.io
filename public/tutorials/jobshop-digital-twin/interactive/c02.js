/* Which Job Should the Machine Do Next?, chapter C2, how CP-SAT finds and proves a plan: the demos.
   Hero: depth-first branch and bound over the paint machine's four orders: the tree of the 24
   sequences, the lower bound of every node, pruning, a node counter and the optimality gap.
   Widget 2: bounds propagation on five operations (precedence, timetabling, edge-finding) one
   deduction at a time. Widget 3: CP-SAT's best plan and best bound on a real 8-job snapshot as the
   deterministic budget grows, with and without the hint (data printed by code/c02_numbers.py).
   Every count here is the one code/c02_numbers.py prints. Needs ../kit/kit.js and jobshop-lib.js. */
(() => {
"use strict";
const { $, css, fmt, caption, fit, frame, alpha, onRedraw, narrowOf, reduceMotion } = window.Kit;
const { PAINT, FAM, runSequence, totalTardiness, drawGantt } = window.JS;

/* ====================================================================== hero: branch and bound */
const ORD = Object.fromEntries(PAINT.orders.map(o => [o.name, o]));
const NAMES = PAINT.orders.map(o => o.name);
const START_PLANS = { none: null, SST: ["ACBD", 50], optimum: ["ABDC", 40] };

function stateOf(prefix) {
  const rows = runSequence(PAINT.orders, prefix);
  const t = rows.length ? rows[rows.length - 1].end : 0;
  const fam = prefix.length ? ORD[prefix[prefix.length - 1]].family : PAINT.start;
  return { rows, t, fam, g: totalTardiness(rows) };
}
/* the three bounds of Part 2: 'none' and 'sofar' use the tardiness already incurred ('none' never prunes);
   'alone' adds, for each remaining order, its tardiness if it ran alone right now */
function aloneEnd(o, t, fam) { return t + (o.family !== fam ? PAINT.setup : 0) + o.p; }
function bound(prefix, kind) {
  const { t, fam, g } = stateOf(prefix);
  if (kind !== "alone") return g;
  return g + NAMES.filter(n => !prefix.includes(n)).reduce((s, n) => s + Math.max(0, aloneEnd(ORD[n], t, fam) - ORD[n].due), 0);
}

/* the tree: every prefix of every permutation, leaves in alphabetical order */
const TREE = [];          // {key, depth, x (0..1), parent}
(function build() {
  const leaves = [];
  const perms = a => a.length <= 1 ? [a] : a.flatMap((x, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map(p => [x, ...p]));
  perms(NAMES).forEach((p, i) => leaves.push({ key: p.join(""), x: (i + 0.5) / 24 }));
  const xs = {};
  leaves.forEach(l => { for (let d = 0; d <= 4; d++) { const k = l.key.slice(0, d); (xs[k] = xs[k] || []).push(l.x); } });
  Object.entries(xs).forEach(([k, v]) => TREE.push({ key: k, depth: k.length, x: v.reduce((s, z) => s + z, 0) / v.length, parent: k.slice(0, -1) }));
})();
const LEAVES_UNDER = k => TREE.filter(n => n.depth === 4 && n.key.startsWith(k)).length;

const bb = {};
function resetBB() {
  const kind = $("#bbBound").value, order = $("#bbOrder").value, start = START_PLANS[$("#bbStart").value];
  Object.assign(bb, {
    kind, order, ub: start ? start[1] : Infinity, inc: start ? start[0] : null, startName: $("#bbStart").value,
    stack: [["", bound([], kind)]], lb: { "": bound([], kind) }, mark: {}, expanded: 0, leaves: 0, pruned: 0,
    current: null, msg: "", done: false, cut: 0,
  });
  bb.mark[""] = "open";
  bb.msg = `Nothing explored yet. The root stands for all 24 sequences; its bound is ${bb.lb[""]}` +
    (start ? `, and the starting plan ${start[0]} gives an upper bound of ${start[1]} before any search.` : ", and there is no plan yet, so the upper bound is infinite.");
}
function stepBB() {
  if (bb.done) return;
  const [key, lb] = bb.stack.pop();
  bb.current = key;
  const label = key || "the root";
  if (bb.kind !== "none" && lb >= bb.ub) {
    bb.pruned++; bb.mark[key] = "pruned"; bb.cut += LEAVES_UNDER(key);
    bb.msg = `Prune ${label}: its bound ${lb} is not below the best plan's ${bb.ub}, so none of its ${LEAVES_UNDER(key)} sequence${LEAVES_UNDER(key) > 1 ? "s" : ""} can do better.`;
  } else if (key.length === 4) {
    bb.leaves++;
    if (lb < bb.ub) {
      const old = bb.ub; bb.ub = lb; bb.inc = key; bb.mark[key] = "best";
      Object.keys(bb.mark).forEach(k => { if (k !== key && bb.mark[k] === "best") bb.mark[k] = "leaf"; });
      bb.msg = `Leaf ${key}: total tardiness ${lb}, better than ${old === Infinity ? "nothing" : old}. It becomes the best plan: the upper bound drops to ${lb}.`;
    } else {
      bb.mark[key] = "leaf";
      bb.msg = `Leaf ${key}: total tardiness ${lb}, no better than ${bb.ub}.`;
    }
  } else {
    bb.expanded++; bb.mark[key] = "expanded";
    let kids = NAMES.filter(n => !key.includes(n)).map(n => [key + n, bound([...key, n], bb.kind)]);
    if (bb.order === "bound") kids.sort((a, b) => a[1] - b[1] || (a[0] < b[0] ? -1 : 1));
    kids.forEach(([k, v]) => { bb.lb[k] = v; bb.mark[k] = "open"; });
    for (let i = kids.length - 1; i >= 0; i--) bb.stack.push(kids[i]);
    bb.msg = `Expand ${label} (bound ${lb}${bb.ub < Infinity ? `, below the best ${bb.ub}` : ""}): its children ` +
      kids.map(([k, v]) => `${k} (${v})`).join(", ") + (bb.order === "bound" ? ", taken best bound first." : ", taken in alphabetical order.");
  }
  if (!bb.stack.length) {
    bb.done = true;
    bb.msg += ` The stack is empty: every sequence has been evaluated or ruled out, so ${bb.inc} = ${bb.ub} is optimal, proven after ${bb.expanded + bb.leaves + bb.pruned} nodes.`;
  }
}
const globalLB = () => Math.min(bb.ub, ...bb.stack.map(s => s[1]));

function drawTree() {
  const cv = $("#bbTree"), narrow = narrowOf(cv), { ctx, w, h } = fit(cv, narrow ? 1.25 : 2.3);
  const top = 16, bottom = 26, X = x => 10 + x * (w - 20), Y = d => top + d * (h - top - bottom) / 4;
  const pos = n => [X(n.x), Y(n.depth)];
  const by = Object.fromEntries(TREE.map(n => [n.key, n]));
  const prunedAbove = k => { for (let d = 0; d < k.length; d++) if (bb.mark[k.slice(0, d)] === "pruned") return true; return false; };
  ctx.lineWidth = 1;
  for (const n of TREE) {
    if (!n.depth) continue;
    const [x1, y1] = pos(n), [x0, y0] = pos(by[n.parent]);
    ctx.strokeStyle = bb.mark[n.key] && bb.mark[n.key] !== "pruned" ? alpha("--ink", 0.55) : prunedAbove(n.key) || bb.mark[n.key] === "pruned" ? alpha("--warm", 0.25) : css("--rule");
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
  const r = narrow ? 3.2 : 4.2;
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const n of TREE) {
    const [x, y] = pos(n), m = bb.mark[n.key];
    ctx.beginPath(); ctx.arc(x, y, n.depth <= 2 ? r + 1.5 : r, 0, 2 * Math.PI);
    if (!m) { ctx.fillStyle = prunedAbove(n.key) ? alpha("--warm", 0.18) : css("--surface"); ctx.fill(); ctx.strokeStyle = prunedAbove(n.key) ? alpha("--warm", 0.35) : css("--muted"); ctx.stroke(); }
    else if (m === "open") { ctx.fillStyle = css("--surface"); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = css("--accent"); ctx.stroke(); ctx.lineWidth = 1; }
    else if (m === "expanded") { ctx.fillStyle = css("--ink"); ctx.fill(); }
    else if (m === "leaf") { ctx.fillStyle = css("--muted"); ctx.fill(); }
    else if (m === "best") { ctx.fillStyle = css("--c4"); ctx.fill(); }
    else if (m === "pruned") { ctx.fillStyle = css("--warm"); ctx.fill(); }
    if (n.key === bb.current) { ctx.beginPath(); ctx.arc(x, y, r + 5, 0, 2 * Math.PI); ctx.strokeStyle = css("--accent"); ctx.lineWidth = 1.5; ctx.stroke(); ctx.lineWidth = 1; }
    /* labels: the letter added at this node for the upper levels, the bound once known */
    if (n.depth >= 1 && n.depth <= 2) {
      ctx.fillStyle = css("--ink"); ctx.fillText(n.key.slice(-1), x + (n.depth === 1 ? -14 : -10), y - 9);
    }
    if (bb.lb[n.key] != null && n.depth <= (narrow ? 1 : 2) && m) {
      ctx.fillStyle = m === "pruned" ? css("--warm") : css("--muted"); ctx.fillText(String(bb.lb[n.key]), x + (n.depth === 1 ? 15 : 11), y - 9);
    }
  }
  /* leaf values along the bottom for evaluated leaves */
  ctx.textBaseline = "top";
  for (const n of TREE) {
    if (n.depth !== 4 || !(bb.mark[n.key] === "leaf" || bb.mark[n.key] === "best")) continue;
    const [x, y] = pos(n);
    ctx.fillStyle = bb.mark[n.key] === "best" ? css("--c4") : css("--muted");
    ctx.fillText(String(bb.lb[n.key]), x, y + 8);
  }
  ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = css("--muted");
  ctx.fillText(narrow ? "root" : "root: all 24 sequences", 4, 0);
}

function drawNodeGantt() {
  const key = bb.current ?? "", prefix = [...key], { rows, t, fam } = stateOf(prefix);
  const blocks = [];
  rows.forEach(rw => {
    if (rw.setup) blocks.push({ lane: 0, start: rw.free, end: rw.start, kind: "setup" });
    blocks.push({ lane: 0, start: rw.start, end: rw.end, colour: ORD[rw.name].family === "blue" ? FAM.blue : FAM.red, label: rw.name, kind: "work" });
  });
  const due = [];
  NAMES.forEach((n, i) => {
    const o = ORD[n];
    due.push({ lane: i + 1, t: o.due });
    if (prefix.includes(n)) return;
    const s = o.family !== fam ? PAINT.setup : 0;
    if (s) blocks.push({ lane: i + 1, start: t, end: t + s, kind: "setup" });
    blocks.push({ lane: i + 1, start: t + s, end: t + s + o.p, colour: o.family === "blue" ? FAM.blue : FAM.red, label: n, kind: "work" });
  });
  const labels = ["machine", ...NAMES.map(n => prefix.includes(n) ? `${n} placed` : `${n} alone`)];
  drawGantt($("#bbGantt"), blocks, { lanes: 5, t0: 0, t1: 120, laneLabels: labels, due, laneHeight: 24 });
}

function drawHero() {
  drawTree(); drawNodeGantt();
  const lb = globalLB(), ub = bb.ub;
  $("#rVisited").textContent = `${bb.expanded + bb.leaves + bb.pruned} of 65`;
  $("#rExpanded").textContent = String(bb.expanded);
  $("#rLeaves").textContent = `${bb.leaves} of 24`;
  $("#rPruned").textContent = `${bb.pruned} (${bb.cut} sequences cut)`;
  $("#rUB").textContent = ub === Infinity ? "none yet" : `${bb.inc} = ${ub}`;
  $("#rLB").textContent = String(lb === Infinity ? 0 : lb);
  $("#rGap").textContent = ub === Infinity ? "∞" : String(ub - lb);
  $("#rStatus").textContent = bb.done ? "OPTIMAL" : ub < Infinity ? "FEASIBLE" : "UNKNOWN";
  const key = bb.current;
  let detail = "";
  if (key != null && key.length < 4 && bb.kind === "alone") {
    const { t, fam, g } = stateOf([...key]);
    const parts = NAMES.filter(n => !key.includes(n)).map(n => { const e = aloneEnd(ORD[n], t, fam); return `${n} alone ends at ${e} (due ${ORD[n].due}: ${Math.max(0, e - ORD[n].due)})`; });
    detail = ` The chart shows the bound of ${key || "the root"}: the orders placed so far are already ${g} min late in total; ${parts.join(", ")}; the sum, ${bound([...key], "alone")}, is a lower bound for every sequence starting with ${key || "anything"}.`;
  }
  caption($("#heroStatus"), bb.msg + detail);
}

let timer = null;
function stopRun() { if (timer) { clearInterval(timer); timer = null; } $("#bbRun").textContent = "Run"; $("#bbRun").setAttribute("aria-pressed", "false"); }
$("#bbStep").addEventListener("click", () => { stopRun(); stepBB(); drawHero(); });
$("#bbRun").addEventListener("click", () => {
  if (timer) { stopRun(); return; }
  if (bb.done) { resetBB(); }
  if (reduceMotion) { while (!bb.done) stepBB(); drawHero(); return; }
  $("#bbRun").textContent = "Pause"; $("#bbRun").setAttribute("aria-pressed", "true");
  timer = setInterval(() => { stepBB(); drawHero(); if (bb.done) stopRun(); }, 420);
});
$("#bbReset").addEventListener("click", () => { stopRun(); resetBB(); drawHero(); });
["#bbBound", "#bbOrder", "#bbStart"].forEach(id => $(id).addEventListener("change", () => { stopRun(); resetBB(); drawHero(); }));

/* ====================================================================== widget 2: propagation */
/* P, Q, R on machine 1; U, V on machine 2 (not the paint orders A-D) */
const TASKS0 = { P: { m: 1, p: 4, est: 0, lct: 20 }, Q: { m: 1, p: 3, est: 1, lct: 8 }, R: { m: 1, p: 3, est: 2, lct: 9 },
                 U: { m: 2, p: 3, est: 0, lct: 20 }, V: { m: 2, p: 5, est: 3, lct: 10 } };
const PREC = [["P", "U", 2]];
const subsets = a => a.reduce((acc, x) => acc.concat(acc.map(s => [...s, x])), [[]]).filter(s => s.length);
function propagateOnce(dom, ef) {
  for (const [a, b, lag] of PREC) {
    let v = dom[a].est + dom[a].p + lag;
    if (v > dom[b].est) return ["precedence", b, "est", v, `${a} ends at ${dom[a].est + dom[a].p} at the earliest, plus the lag ${lag}`];
    v = dom[b].lct - dom[b].p - lag;
    if (v < dom[a].lct) return ["precedence", a, "lct", v, `${b} must start by ${dom[b].lct - dom[b].p}, minus the lag ${lag}`];
  }
  for (const m of [1, 2]) {
    const names = Object.keys(dom).filter(n => dom[n].m === m);
    for (const j of names) for (const i of names) {
      if (i === j) continue;
      const lst = dom[i].lct - dom[i].p, ect = dom[i].est + dom[i].p;
      if (lst >= ect) continue;
      if (dom[j].est < ect && dom[j].est + dom[j].p > lst) return ["timetabling", j, "est", ect, `${i} occupies [${lst}, ${ect}) in every solution`];
      if (dom[j].lct > lst && dom[j].lct - dom[j].p < ect) return ["timetabling", j, "lct", lst, `${i} occupies [${lst}, ${ect}) in every solution`];
    }
  }
  if (ef) for (const m of [1, 2]) {
    const names = Object.keys(dom).filter(n => dom[n].m === m);
    for (const i of names) {
      const others = names.filter(n => n !== i);
      /* the same subset order as the Python (bit masks 1, 2, 3, ...) */
      for (let mask = 1; mask < 1 << others.length; mask++) {
        const om = others.filter((_, k) => mask >> k & 1), pOm = om.reduce((s, n) => s + dom[n].p, 0);
        const estAll = Math.min(...[...om, i].map(n => dom[n].est)), lctOm = Math.max(...om.map(n => dom[n].lct));
        if (estAll + pOm + dom[i].p > lctOm) {
          const v = Math.max(...subsets(om).map(s => Math.min(...s.map(n => dom[n].est)) + s.reduce((q, n) => q + dom[n].p, 0)));
          if (v > dom[i].est) return ["edge-finding", i, "est", v, `${i} cannot end before all of {${om.join(", ")}}: ${estAll} + ${pOm} + ${dom[i].p} > ${lctOm}`];
        }
        const lctAll = Math.max(...[...om, i].map(n => dom[n].lct)), estOm = Math.min(...om.map(n => dom[n].est));
        if (lctAll - pOm - dom[i].p < estOm) {
          const v = Math.min(...subsets(om).map(s => Math.max(...s.map(n => dom[n].lct)) - s.reduce((q, n) => q + dom[n].p, 0)));
          if (v < dom[i].lct) return ["edge-finding", i, "lct", v, `${i} cannot start after all of {${om.join(", ")}}: ${lctAll} - ${pOm} - ${dom[i].p} < ${estOm}`];
        }
      }
    }
  }
  return null;
}
const pr = { dom: null, steps: [], done: false };
function resetProp() { pr.dom = JSON.parse(JSON.stringify(TASKS0)); pr.steps = []; pr.done = false; }
function stepProp() {
  if (pr.done) return;
  const d = propagateOnce(pr.dom, $("#pEF").checked);
  if (!d) { pr.done = true; return; }
  const [rule, task, field, value, why] = d;
  pr.steps.push({ rule, task, field, old: pr.dom[task][field], value, why });
  pr.dom[task][field] = value;
  if (!propagateOnce(pr.dom, $("#pEF").checked)) pr.done = true;
}
const starts = dom => Object.values(dom).reduce((s, t) => s * (t.lct - t.p - t.est + 1), 1);
function drawProp() {
  const cv = $("#propC"), narrow = narrowOf(cv), { ctx, w, h } = fit(cv, narrow ? 1.3 : 2.6);
  const names = Object.keys(TASKS0), padL = 58, padR = 12, top = 16, GAP = 1.0, rowH = (h - top - 30) / (names.length + GAP);
  const X = t => padL + t / 20 * (w - padL - padR);
  ctx.font = "11px " + css("--font-mono");
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let t = 0; t <= 20; t += narrow ? 5 : 2) {
    ctx.strokeStyle = css("--grid"); ctx.beginPath(); ctx.moveTo(Math.round(X(t)) + 0.5, top); ctx.lineTo(Math.round(X(t)) + 0.5, h - 24); ctx.stroke();
    ctx.fillStyle = css("--muted"); ctx.fillText(String(t), X(t), h - 20);
  }
  const last = pr.steps.length ? pr.steps[pr.steps.length - 1] : null;
  names.forEach((n, k) => {
    const y = top + (k + (k >= 3 ? GAP : 0)) * rowH, t0 = TASKS0[n], t = pr.dom[n], hh = rowH * 0.62, yy = y + (rowH - hh) / 2;
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillStyle = css("--ink");
    ctx.fillText(`${n} · p=${t.p}`, padL - 6, y + rowH / 2);
    /* the original window, outlined; the current one, filled */
    ctx.strokeStyle = css("--muted"); ctx.setLineDash([3, 3]); ctx.strokeRect(X(t0.est) + 0.5, yy + 0.5, X(t0.lct) - X(t0.est), hh); ctx.setLineDash([]);
    ctx.fillStyle = alpha("--accent", 0.22); ctx.fillRect(X(t.est), yy, X(t.lct) - X(t.est), hh);
    /* compulsory part [lst, ect) */
    const lst = t.lct - t.p, ect = t.est + t.p;
    if (lst < ect) { ctx.fillStyle = css("--accent"); ctx.fillRect(X(lst), yy, X(ect) - X(lst), hh); }
    /* the earliest placement, as a thin bar */
    ctx.fillStyle = css("--ink"); ctx.fillRect(X(t.est), yy + hh + 2, X(t.est + t.p) - X(t.est), 2.5);
    if (last && last.task === n) {
      ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2;
      const xv = X(last.value); ctx.beginPath(); ctx.moveTo(xv, yy - 3); ctx.lineTo(xv, yy + hh + 3); ctx.stroke(); ctx.lineWidth = 1;
    }
  });
  /* the precedence P -> U */
  const yA = top + 0 * rowH + rowH / 2, yD = top + (3 + GAP) * rowH + rowH / 2, xA = X(pr.dom.P.est + pr.dom.P.p), xD = X(pr.dom.U.est);
  ctx.strokeStyle = css("--warm"); ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(xA, yA); ctx.lineTo(xD, yD); ctx.stroke(); ctx.setLineDash([]);
  /* a rule between the two machines */
  const ySep = Math.round(top + (3 + GAP * 0.35) * rowH) + 0.5;
  ctx.strokeStyle = css("--rule"); ctx.beginPath(); ctx.moveTo(4, ySep); ctx.lineTo(w - padR, ySep); ctx.stroke();
  ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillStyle = css("--muted");
  ctx.fillText("machine 1", 4, top - 2);
  ctx.fillText("machine 2", 4, top + (3 + GAP) * rowH - 2);
  ctx.textBaseline = "alphabetic";
  $("#pLog").innerHTML = pr.steps.length ? pr.steps.map(s => `<li><b>${s.rule}</b>: <span class="m">${s.field === "est" ? "\\mathrm{est}" : "\\mathrm{lct}"}_{${s.task}}: ${s.old} \\to ${s.value}</span> (${s.why})</li>`).join("") : "<li>No deduction yet.</li>";
  window.Kit.renderMath($("#pLog"));
  const ef = $("#pEF").checked;
  caption($("#propCap"), `${pr.steps.length} deduction${pr.steps.length === 1 ? "" : "s"}${pr.done ? ", fixed point reached: no rule can shrink a window further" : ""}. ` +
    `Windows now: ${Object.entries(pr.dom).map(([n, t]) => `${n} [${t.est}, ${t.lct}]`).join(", ")}. ` +
    `Product of the numbers of possible start values: ${starts(pr.dom).toLocaleString("en-US")}, from ${starts(TASKS0).toLocaleString("en-US")}` +
    (pr.done ? (ef ? `: edge-finding moved P behind Q and R, and the precedence carried that to U.` : `: without edge-finding, P could still start at minute 0, which no feasible plan allows; search would have to find that out by branching.`) : "."));
}
$("#pStep").addEventListener("click", () => { stepProp(); drawProp(); });
$("#pAll").addEventListener("click", () => { while (!pr.done) stepProp(); drawProp(); });
$("#pReset").addEventListener("click", () => { resetProp(); drawProp(); });
$("#pEF").addEventListener("change", () => { resetProp(); drawProp(); });

/* ====================================================================== widget 3: the budget sweep */
/* printed by code/c02_numbers.py ("hero-2 data"): snapshot 101 of the simulated day, 8 jobs, 19 operations */
const SWEEP = {"budgets": [0.025, 0.05, 0.1, 0.2, 0.4, 0.8], "warm": 7962, "hint": [["FEASIBLE", 7940, 7540], ["FEASIBLE", 7940, 7540], ["FEASIBLE", 7940, 7540], ["FEASIBLE", 7940, 7740], ["FEASIBLE", 7940, 7859], ["OPTIMAL", 7940, 7940]], "nohint": [["UNKNOWN", null, 7540], ["UNKNOWN", null, 7540], ["FEASIBLE", 8202, 7702], ["FEASIBLE", 7961, 7847], ["FEASIBLE", 7940, 7858], ["OPTIMAL", 7940, 7940]]};
let sweepMode = "hint";
function drawSweep() {
  const cv = $("#sweepC"), narrow = narrowOf(cv);
  const lx = b => Math.log2(b / 0.025);
  const fr = frame(cv, narrow ? 1.15 : 2.1, { x: [-0.3, lx(0.8) + 0.3], y: [7450, 8300], xlabel: "deterministic budget (log scale)", ylabel: "objective  10·ΣT + ΣC",
    xticks: SWEEP.budgets.map(lx), xfmt: v => String(+(0.025 * 2 ** v).toPrecision(2)), yticks: [7500, 7700, 7900, 8100, 8300], pad: { l: 46 } });
  const { ctx, X, Y } = fr;
  /* the warm start */
  ctx.strokeStyle = css("--muted"); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(X(-0.3), Y(SWEEP.warm)); ctx.lineTo(X(lx(0.8) + 0.3), Y(SWEEP.warm)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = css("--muted"); ctx.textAlign = "left"; ctx.font = "11px " + css("--font-mono");
  ctx.fillText(`warm start ${SWEEP.warm}`, X(-0.2), Y(SWEEP.warm) - 6);
  const draw = (mode, strong) => {
    const rows = SWEEP[mode], a = strong ? 1 : 0.28;
    ctx.globalAlpha = a;
    /* gap band */
    rows.forEach((r, i) => { if (r[1] == null) return; ctx.fillStyle = alpha("--accent", 0.12); const x = X(lx(SWEEP.budgets[i])); ctx.fillRect(x - 5, Y(Math.min(8300, r[1])), 10, Y(r[2]) - Y(Math.min(8300, r[1]))); });
    const line = (vals, colour) => {
      ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.beginPath(); let on = false;
      vals.forEach((v, i) => { if (v == null) { on = false; return; } const x = X(lx(SWEEP.budgets[i])), y = Y(Math.min(8300, v)); on ? ctx.lineTo(x, y) : ctx.moveTo(x, y); on = true; });
      ctx.stroke(); ctx.lineWidth = 1;
      vals.forEach((v, i) => { if (v == null) return; ctx.fillStyle = colour; ctx.beginPath(); ctx.arc(X(lx(SWEEP.budgets[i])), Y(Math.min(8300, v)), 3.5, 0, 2 * Math.PI); ctx.fill(); });
    };
    line(rows.map(r => r[1]), css("--accent"));
    line(rows.map(r => r[2]), css("--warm"));
    if (strong) {
      ctx.textAlign = "center"; ctx.fillStyle = css("--ink");
      rows.forEach((r, i) => {
        const x = X(lx(SWEEP.budgets[i]));
        ctx.fillText(r[0] === "OPTIMAL" ? "OPT" : r[0] === "FEASIBLE" ? "FEAS" : "UNK", x, Y(8300) + 2);
        if (r[1] == null) { ctx.fillStyle = css("--muted"); ctx.fillText("no plan", x, Y(8150)); ctx.fillStyle = css("--ink"); }
      });
    }
    ctx.globalAlpha = 1;
  };
  draw(sweepMode === "hint" ? "nohint" : "hint", false);
  draw(sweepMode, true);
  /* legend in the empty lower right corner (every bound there is above 7700) */
  /* short labels on a phone, so the legend stays right of the 0.1-0.2 segment of the bound line */
  ctx.textAlign = "right"; ctx.fillStyle = css("--accent"); ctx.fillText(narrow ? "blue: best plan" : "blue: best plan (upper bound)", fr.w - fr.pad.r - 4, Y(7600));
  ctx.fillStyle = css("--warm"); ctx.fillText(narrow ? "orange: bound" : "orange: best bound (lower bound)", fr.w - fr.pad.r - 4, Y(7520));
  caption($("#sweepCap"), sweepMode === "hint"
    ? `With the hint, the plan (blue) is optimal from the smallest budget; the rest of the budget only raises the bound (orange) until it meets the plan at 0.8. The table below gives the values.`
    : `Without the hint, there is no plan at the two smallest budgets, and the plan reaches the optimum only at 0.4; the proof still comes at 0.8. The table below gives the values.`);
  $$btn().forEach(b => b.setAttribute("aria-pressed", String(b.dataset.mode === sweepMode)));
}
const $$btn = () => [...document.querySelectorAll("#sweepW [data-mode]")];
$$btn().forEach(b => b.addEventListener("click", () => { sweepMode = b.dataset.mode; drawSweep(); }));

/* ====================================================================== start */
resetBB(); resetProp();
function drawAll() { drawHero(); drawProp(); drawSweep(); }
drawAll();
onRedraw(drawAll);
})();
