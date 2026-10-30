/* Satellite Pass Planning from Scratch, chapter 08, Constraint Programming with CP-SAT: the live demos.
   (1) A depth-first branch and bound written from scratch, on a toy cut from the course's small instance
   (satellite 0's first pass, 21 opportunities), with three bounds, two starting plans and two branching orders;
   it runs node for node as branch_and_bound() in code/ch08_numbers.py, and the page compares its node counts
   with the Python ones. (2) Propagation on one conflict pair: the enforced precedences of cpsat.py shrinking
   the start-time domains, fixing the order literal, or failing and producing a learned clause. (3) CP-SAT's
   anytime traces (best plan and best bound over time) on the small and medium instances, from ch08-data.js.
   Needs ../kit/kit.js, sat.js, instance-small.js and ch08-data.js. */
(() => {
"use strict";
const { $, css, fmt, caption, frame, fit, alpha, onRedraw, narrowOf, seg, plotLine } = window.Kit;
const Sat = window.Sat, I = window.SAT_SMALL, D = window.SAT_CH08, T = D.toy;
const O = T.opps, n = O.length, TR = T.tr;
const latest = o => o.we - o.dur;
const int = v => Math.round(v).toLocaleString("en-US");

/* ---------------- (1) branch and bound on the toy ---------------- */
/* A node has decided opportunities 0..k-1 (in order of window start): taken ones are in `plan` as [index, start],
   with the satellite's sequence fixed by window start. `last`, `tlast`: the last task taken and its start;
   ud / um: seconds used per revolution / memory segment. */
function makeBB(bound, start, takeFirst) {
  const startOf = (nd, o) => {
    if (nd.taken.has(o.target)) return null;
    if ((nd.ud.get(o.rev) || 0) + o.dur > T.dutyBudget || (nd.um.get(o.mem) || 0) + o.dur > T.memoryBudget) return null;
    const t = nd.last < 0 ? o.ws : Math.max(o.ws, nd.tlast + O[nd.last].dur + TR[nd.last][o.idx]);
    return t <= latest(o) ? t : null;
  };
  const ub = nd => {
    if (bound === "knap") {
      const perRev = new Map();
      for (let q = nd.k; q < n; q++) { const o = O[q]; if (startOf(nd, o) !== null) { if (!perRev.has(o.rev)) perRev.set(o.rev, []); perRev.get(o.rev).push(o); } }
      let tot = nd.val;
      for (const rev of [...perRev.keys()].sort((a, b) => a - b)) {
        let cap = T.dutyBudget - (nd.ud.get(rev) || 0);
        const items = perRev.get(rev).slice().sort((a, b) => (-a.w / a.dur) - (-b.w / b.dur) || a.idx - b.idx);
        for (const o of items) { if (cap <= 0) break; tot += Math.min(1, cap / o.dur) * o.w; cap -= o.dur; }
      }
      return tot;
    }
    const best = new Map();
    for (let q = nd.k; q < n; q++) {
      const o = O[q];
      if (nd.taken.has(o.target) || (bound === "fit" && startOf(nd, o) === null)) continue;
      best.set(o.target, Math.max(best.get(o.target) || 0, o.w));
    }
    let s = nd.val; for (const v of best.values()) s += v; return s;
  };
  const root = { k: 0, val: 0, last: -1, tlast: 0, taken: new Set(), ud: new Map(), um: new Map(), plan: [] };
  const st = { best: start, bestPlan: null, explored: 0, pruned: 0, stack: [[root, ub(root)]], trace: [], last: null, newBest: false };
  st.rootBound = st.stack[0][1];
  st.trace.push([0, st.best, st.rootBound]);
  st.done = () => st.stack.length === 0;
  st.step = () => {
    if (!st.stack.length) return;
    const [nd, b] = st.stack.pop();
    st.explored++;
    st.newBest = false;
    if (nd.val > st.best) { st.best = nd.val; st.bestPlan = nd.plan; st.newBest = true; }
    let action;
    if (b <= st.best) { st.pruned++; action = "pruned"; }
    else if (nd.k < n) {
      const o = O[nd.k];
      const kids = [{ k: nd.k + 1, val: nd.val, last: nd.last, tlast: nd.tlast, taken: nd.taken, ud: nd.ud, um: nd.um, plan: nd.plan }];
      const t = startOf(nd, o);
      if (t !== null) {
        const ud = new Map(nd.ud), um = new Map(nd.um);
        ud.set(o.rev, (ud.get(o.rev) || 0) + o.dur); um.set(o.mem, (um.get(o.mem) || 0) + o.dur);
        kids.push({ k: nd.k + 1, val: nd.val + o.w, last: nd.k, tlast: t, taken: new Set(nd.taken).add(o.target), ud, um, plan: nd.plan.concat([[nd.k, t]]) });
      }
      if (!takeFirst) kids.reverse();
      for (const c of kids) st.stack.push([c, ub(c)]);
      action = t === null ? "branch-leave-only" : "branch";
      st.takeStart = t;
    } else action = "leaf";
    let glob = st.best; for (const [, bb] of st.stack) if (bb > glob) glob = bb;
    const lastT = st.trace[st.trace.length - 1];
    if (lastT[1] !== st.best || lastT[2] !== glob || !st.stack.length) st.trace.push([st.explored, st.best, glob]);
    st.glob = glob;
    st.last = { nd, b, action };
  };
  st.glob = st.rootBound;
  return st;
}

const bb = { bound: "simple", start: "zero", take: true, st: null, timer: null };
const keyOf = () => `${bb.bound}|${bb.start}|${bb.take ? "take" : "leave"}`;
const startValue = () => bb.start === "greedy" ? T.greedy.value : 0;
function newSearch() { stopRun(); bb.st = makeBB(bb.bound, startValue(), bb.take); }
function finishSearch() { while (!bb.st.done()) bb.st.step(); }
const BOUND_WORDS = {
  simple: "every remaining target's priority",
  fit: "the priority of each remaining opportunity that could still be taken on its own",
  knap: "the power budget's fractional knapsack over those opportunities",
};

function drawBBTimeline() {
  const cv = $("#bbT"), narrow = narrowOf(cv, 600);
  const fr = frame(cv, narrow ? 1.05 : 2.1, { x: [0, 900], y: [0, n], yticks: false, xlabel: "time [s] on satellite 0", pad: { l: narrow ? 30 : 52, t: 6 } });
  const { ctx, X, Y } = fr;
  const st = bb.st, nd = st.last ? st.last.nd : null;
  const takenAt = new Map(nd ? nd.plan : []), incumbent = new Map(st.bestPlan || []);
  ctx.font = "10px " + css("--font-mono"); ctx.textBaseline = "middle"; ctx.textAlign = "right";
  for (let q = 0; q < n; q++) {
    const o = O[q], y1 = Y(n - q), y0 = Y(n - q - 1), hh = (y0 - y1);
    const decided = nd && q < nd.k;
    ctx.fillStyle = css("--muted");
    ctx.fillText(narrow ? String(o.w) : `${o.w} · ${o.dur}s`, fr.pad.l - 4, (y0 + y1) / 2);
    if (decided && !takenAt.has(q)) { ctx.fillStyle = alpha("--ink", 0.07); ctx.fillRect(X(o.ws), y1 + hh * 0.15, X(o.we) - X(o.ws), hh * 0.7); }
    ctx.strokeStyle = decided ? alpha("--ink", 0.25) : alpha("--ink", 0.55);
    ctx.strokeRect(X(o.ws) + 0.5, y1 + hh * 0.15, X(o.we) - X(o.ws), hh * 0.7);
    if (decided && !takenAt.has(q)) { ctx.strokeStyle = alpha("--ink", 0.35); seg(ctx, X(o.ws), y1 + hh * 0.5, X(o.we), y1 + hh * 0.5); }
    if (takenAt.has(q)) { const t = takenAt.get(q); ctx.fillStyle = css("--post"); ctx.fillRect(X(t), y1 + hh * 0.12, Math.max(2, X(t + o.dur) - X(t)), hh * 0.5); }
    if (incumbent.has(q)) { const t = incumbent.get(q); ctx.fillStyle = css("--c4"); ctx.fillRect(X(t), y1 + hh * 0.68, Math.max(2, X(t + o.dur) - X(t)), hh * 0.2); }
  }
  if (nd && nd.k < n && st.last.action !== "pruned") {   // the opportunity branched on next
    const o = O[nd.k], y1 = Y(n - nd.k), y0 = Y(n - nd.k - 1);
    ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2; ctx.strokeRect(X(o.ws) - 1.5, y1 + 0.5, X(o.we) - X(o.ws) + 3, y0 - y1 - 1); ctx.lineWidth = 1;
  }
}
function drawBBChart() {
  const cv = $("#bbC"), narrow = narrowOf(cv, 600), st = bb.st;
  const total = (T.bb[keyOf()] || [0, st.explored])[1];
  const ymax = Math.ceil(Math.max(st.rootBound, T.cpsat.value) / 5) * 5 + 2;
  const fr = frame(cv, narrow ? 2.0 : 4.4, { x: [0, Math.max(total, st.explored, 1)], y: [0, ymax], xlabel: "nodes explored", ylabel: "priority" });
  const { ctx, X, Y } = fr;
  const tr = st.trace.slice();
  if (tr[tr.length - 1][0] !== st.explored) tr.push([st.explored, st.best, st.glob]);
  const stepXY = (col) => { const xs = [], ys = []; tr.forEach(([x, b, g], q) => { const v = col === 1 ? b : g; if (q) { xs.push(x); ys.push(col === 1 ? tr[q - 1][1] : tr[q - 1][2]); } xs.push(x); ys.push(v); }); return [xs, ys]; };
  const [bx, by] = stepXY(2), [ix, iy] = stepXY(1);
  plotLine(fr, bx, by, { color: css("--warm"), width: 2 });
  plotLine(fr, ix, iy, { color: css("--c4"), width: 2.2 });
  ctx.strokeStyle = alpha("--ink", 0.5); ctx.setLineDash([3, 3]); seg(ctx, X(0), Y(T.cpsat.value), X(fr.x1), Y(T.cpsat.value)); ctx.setLineDash([]);
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "right"; ctx.textBaseline = "bottom";
  ctx.fillText(`CP-SAT optimum ${T.cpsat.value}`, X(fr.x1) - 2, Y(T.cpsat.value) - 2);
}
function bbCaption() {
  const st = bb.st, L = st.last, key = keyOf(), py = T.bb[key];
  const setting = `Bound: ${BOUND_WORDS[bb.bound]}; starting plan: ${bb.start === "greedy" ? `the greedy's, worth ${T.greedy.value}` : "none (best = 0)"}; branch "${bb.take ? "take" : "leave"}" first.`;
  if (!L) return `${setting} The root decides nothing: its bound is ${fmt(st.rootBound, bb.bound === "knap" ? 2 : 0)}. Press <b>step</b> to explore one node, or <b>run</b>.`;
  const nd = L.nd, takenN = nd.plan.length;
  let s = `<b>Node ${int(st.explored)}</b>: ${nd.k} of ${n} opportunities decided (${takenN} taken, ${nd.k - takenN} left), value ${nd.val}, bound ${fmt(L.b, bb.bound === "knap" ? 2 : 0)}. `;
  if (st.newBest) s += `It is a plan better than any so far: <b>new best ${st.best}</b>. `;
  if (L.action === "pruned") s += `Its bound is not above the best plan known (${st.best}), so nothing below it can do better: <b>pruned</b>. `;
  else if (L.action === "branch") { const o = O[nd.k]; s += `Branch on opportunity ${nd.k + 1} (priority ${o.w}, ${o.dur} s, window ${o.ws}–${o.we} s, outlined): take it at ${st.takeStart} s, or leave it. `; }
  else if (L.action === "branch-leave-only") { const o = O[nd.k]; s += `Opportunity ${nd.k + 1} cannot be taken here (budget, its target already served, or no start left in its window), so the only child leaves it. `; }
  else s += "All opportunities are decided: a leaf. ";
  s += `Best plan ${st.best}, global bound ${fmt(st.glob, bb.bound === "knap" ? 2 : 0)}, ${int(st.pruned)} nodes pruned.`;
  if (st.done()) {
    const match = py && py[0] === st.best && py[1] === st.explored && py[2] === st.pruned;
    const ok = st.bestPlan ? Sat.checkPlan(T, new Map(st.bestPlan.map(([q, t]) => [q, t]))).length === 0 : true;
    s += ` <b>Search complete</b> after ${int(st.explored)} nodes, against ${int(2 ** n)} subsets for complete enumeration: the best plan, ${st.best}, is <b>proved optimal</b> because no open node is left whose bound exceeds it. CP-SAT finds ${T.cpsat.value} on the same opportunities (${T.cpsat.status}). ${py ? (match ? "Node and prune counts match <code>ch08_numbers.py</code> exactly." : `<b>The Python run explored ${py[1]} nodes and pruned ${py[2]}: this page differs.</b>`) : ""} ${st.bestPlan ? (ok ? "The best plan passes <code>Sat.checkPlan</code>." : "<b>The best plan fails the checker.</b>") : "The greedy's starting plan was never beaten."}`;
  }
  return `${setting} ${s}`;
}
function drawBB() {
  drawBBTimeline(); drawBBChart();
  const st = bb.st;
  $("#rNodes").textContent = int(st.explored);
  $("#rPruned").textContent = int(st.pruned);
  $("#rBest").textContent = String(st.best);
  $("#rGlob").textContent = fmt(st.glob, bb.bound === "knap" ? 2 : 0);
  const py = T.bb[keyOf()];
  $("#rPy").textContent = py ? `${int(py[1])} nodes, best ${py[0]}` : "–";
  caption($("#bbStatus"), bbCaption());
}
function stopRun() { if (bb.timer) { cancelAnimationFrame(bb.timer); bb.timer = null; } const b = $("#bbRun"); b.textContent = "run"; b.setAttribute("aria-pressed", "false"); }
function runAnim() {
  if (bb.timer) { stopRun(); return; }
  if (bb.st.done()) newSearch();
  const total = (T.bb[keyOf()] || [0, 2000])[1], per = Math.max(1, Math.ceil(total / 300));
  $("#bbRun").textContent = "pause"; $("#bbRun").setAttribute("aria-pressed", "true");
  const tick = () => { for (let q = 0; q < per && !bb.st.done(); q++) bb.st.step(); drawBB(); if (bb.st.done()) { stopRun(); return; } bb.timer = requestAnimationFrame(tick); };
  bb.timer = requestAnimationFrame(tick);
}
function syncBB() {
  for (const b of ["simple", "fit", "knap"]) $("#bnd-" + b).setAttribute("aria-pressed", String(bb.bound === b));
  $("#stZero").setAttribute("aria-pressed", String(bb.start === "zero")); $("#stGreedy").setAttribute("aria-pressed", String(bb.start === "greedy"));
  $("#brTake").setAttribute("aria-pressed", String(bb.take)); $("#brLeave").setAttribute("aria-pressed", String(!bb.take));
}
for (const b of ["simple", "fit", "knap"]) $("#bnd-" + b).addEventListener("click", () => { bb.bound = b; syncBB(); newSearch(); finishSearch(); drawBB(); });
$("#stZero").addEventListener("click", () => { bb.start = "zero"; syncBB(); newSearch(); finishSearch(); drawBB(); });
$("#stGreedy").addEventListener("click", () => { bb.start = "greedy"; syncBB(); newSearch(); finishSearch(); drawBB(); });
$("#brTake").addEventListener("click", () => { bb.take = true; syncBB(); newSearch(); finishSearch(); drawBB(); });
$("#brLeave").addEventListener("click", () => { bb.take = false; syncBB(); newSearch(); finishSearch(); drawBB(); });
$("#bbStep").addEventListener("click", () => { stopRun(); if (bb.st.done()) newSearch(); bb.st.step(); drawBB(); });
$("#bbRun").addEventListener("click", runAnim);
$("#bbReset").addEventListener("click", () => { newSearch(); drawBB(); });
$("#bbEnd").addEventListener("click", () => { stopRun(); finishSearch(); drawBB(); });

/* ---------------- (2) propagation on one conflict pair ---------------- */
const PAIRS = I.python.pairs.map(([a, b, sab, sba, af, bf]) => ({ a, b, sab, sba, aFirst: !!af, bFirst: !!bf }));
const KINDS = {
  both: PAIRS.filter(p => p.aFirst && p.bFirst),
  one: PAIRS.filter(p => p.aFirst !== p.bFirst),
  none: PAIRS.filter(p => !p.aFirst && !p.bFirst),
};
const pp = { kind: "both", idx: 0, xa: "free", xb: "free", ord: "free", lbA: null, lbB: null };
function curPair() { return KINDS[pp.kind][pp.idx]; }
function resetPairState() {
  const p = curPair(), A = I.opps[p.a], B = I.opps[p.b];
  pp.lbA = A.ws; pp.lbB = B.ws;
  const ra = $("#ppLa"), rb = $("#ppLb");
  ra.min = A.ws; ra.max = latest(A); ra.step = 1; ra.value = A.ws;
  rb.min = B.ws; rb.max = latest(B); rb.step = 1; rb.value = B.ws;
  $("#ppIdx").max = KINDS[pp.kind].length - 1; $("#ppIdx").value = pp.idx;
}
/* Bounds propagation of cpsat.py's constraints for this one pair, as a CP solver would do it. */
function propagate() {
  const p = curPair(), A = I.opps[p.a], B = I.opps[p.b];
  const gab = A.dur + p.sab, gba = B.dur + p.sba;
  let da = [pp.lbA, latest(A)], db = [pp.lbB, latest(B)];
  let xa = pp.xa, xb = pp.xb, ord = pp.ord;
  const notes = []; let conflict = null;
  const fitsAB = () => da[0] + gab <= db[1], fitsBA = () => db[0] + gba <= da[1];
  if (pp.kind === "none") {
    if (xa === "1" && xb === "1") conflict = "the clause [[\\lnot x_a \\lor \\lnot x_b]] is violated: both tasks cannot be selected";
    else if (xa === "1" && xb === "free") { xb = "0"; notes.push("the clause [[\\lnot x_a \\lor \\lnot x_b]] with [[x_a = 1]] forces [[x_b = 0]] (unit propagation, no search)"); }
    else if (xb === "1" && xa === "free") { xa = "0"; notes.push("the clause [[\\lnot x_a \\lor \\lnot x_b]] with [[x_b = 1]] forces [[x_a = 0]] (unit propagation, no search)"); }
    else notes.push("the clause [[\\lnot x_a \\lor \\lnot x_b]] says nothing about start times; it only forbids selecting both");
  } else if (xa === "1" && xb === "1") {
    if (pp.kind === "one") {
      ord = p.aFirst ? "ab" : "ba";
      notes.push(`only "${p.aFirst ? "a then b" : "b then a"}" can ever fit, so the model enforces that precedence directly, without an order literal`);
    } else if (ord === "free") {
      const ab = fitsAB(), ba = fitsBA();
      if (ab && !ba) { ord = "ab"; notes.push("with these bounds “b then a” no longer fits, so propagation sets the order literal to “a then b”"); }
      else if (!ab && ba) { ord = "ba"; notes.push("with these bounds “a then b” no longer fits, so propagation sets the order literal to “b then a”"); }
      else if (!ab && !ba) conflict = "neither order fits between the current bounds";
      else notes.push("both orders still fit: propagation cannot decide the order literal; search must branch on it, or other constraints will");
    }
    if (!conflict && ord === "ab") { db[0] = Math.max(db[0], da[0] + gab); da[1] = Math.min(da[1], db[1] - gab); notes.push(`[[s_b \\ge s_a + ${gab}]] is enforced: the earliest start of b rises to ${db[0]} and the latest start of a falls to ${da[1]}`); }
    if (!conflict && ord === "ba") { da[0] = Math.max(da[0], db[0] + gba); db[1] = Math.min(db[1], da[1] - gba); notes.push(`[[s_a \\ge s_b + ${gba}]] is enforced: the earliest start of a rises to ${da[0]} and the latest start of b falls to ${db[1]}`); }
    if (!conflict && (da[0] > da[1] || db[0] > db[1])) conflict = `a domain is empty after enforcing "${ord === "ab" ? "a then b" : "b then a"}"`;
  } else {
    notes.push(`the precedence${pp.kind === "both" ? "s are" : " is"} enforced only if [[x_a]] and [[x_b]] are both true (half-reification, <code>only_enforce_if</code>); until then the start times are untouched`);
  }
  let learned = null;
  if (conflict) {
    const lits = ["\\lnot x_a", "\\lnot x_b"];
    if (pp.kind === "both" && pp.ord !== "free") lits.push(pp.ord === "ab" ? "\\lnot o_{ab}" : "o_{ab}");
    if (pp.kind !== "none") {
      if (pp.lbA > A.ws) lits.push(`\\lbrack s_a < ${pp.lbA} \\rbrack`);
      if (pp.lbB > B.ws) lits.push(`\\lbrack s_b < ${pp.lbB} \\rbrack`);
    }
    learned = lits.join(" \\lor ");
  }
  return { p, A, B, gab, gba, da, db, xa, xb, ord, notes, conflict, learned };
}
function drawPair() {
  const r = propagate(), { A, B } = r;
  const lo = Math.min(A.ws, B.ws) - 20, hi = Math.max(A.we, B.we) + 20;
  const cv = $("#ppC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.8 : 3.6, { x: [lo, hi], y: [0, 2], yticks: false, xlabel: "start time [s] on satellite " + A.sat, pad: { l: 40 } });
  const { ctx, X, Y } = fr;
  ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle"; ctx.textAlign = "right"; ctx.fillStyle = css("--muted");
  ctx.fillText("s_a", fr.pad.l - 6, Y(1.5)); ctx.fillText("s_b", fr.pad.l - 6, Y(0.5));
  const lane = (o, dom, yc, sel) => {
    ctx.strokeStyle = alpha("--ink", 0.4); ctx.setLineDash([4, 3]); ctx.strokeRect(X(o.ws), Y(yc + 0.38), X(o.we) - X(o.ws), Y(yc - 0.38) - Y(yc + 0.38)); ctx.setLineDash([]);
    ctx.fillStyle = alpha("--muted", 0.25); ctx.fillRect(X(o.ws), Y(yc + 0.1), Math.max(2, X(latest(o)) - X(o.ws)), Y(yc - 0.1) - Y(yc + 0.1));
    if (sel !== "0" && dom[0] <= dom[1]) { ctx.fillStyle = r.conflict ? css("--warm") : css("--post"); ctx.fillRect(X(dom[0]), Y(yc + 0.16), Math.max(3, X(dom[1]) - X(dom[0])), Y(yc - 0.16) - Y(yc + 0.16)); }
    ctx.fillStyle = css("--ink-2"); ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    ctx.fillText(`target ${o.target}, priority ${o.w}, ${o.dur} s: window ${o.ws}–${o.we}`, X(o.ws) + 2, Y(yc + 0.4));
    ctx.textBaseline = "middle";
  };
  lane(A, r.da, 1.5, r.xa); lane(B, r.db, 0.5, r.xb);
  const dtxt = (d, sel) => sel === "0" ? "not selected" : d[0] > d[1] ? "empty" : `[${d[0]}, ${d[1]}]`;
  $("#ppDa").textContent = dtxt(r.da, r.xa); $("#ppDb").textContent = dtxt(r.db, r.xb);
  $("#ppG").textContent = `${r.gab} / ${r.gba}`;
  $("#ppIdxV").textContent = `${pp.idx + 1} of ${KINDS[pp.kind].length}`;
  $("#ppLaV").textContent = String(pp.lbA); $("#ppLbV").textContent = String(pp.lbB);
  const kindWords = { both: "both orders fit: one order literal and two enforced precedences", one: "only one order fits: one enforced precedence", none: "neither order fits: one clause" };
  let s = `Opportunities ${r.p.a} (a) and ${r.p.b} (b) of the small instance, on satellite ${A.sat}: ${kindWords[pp.kind]}. Gaps between starts: [[d_a + s(a,b) = ${r.gab}]] s and [[d_b + s(b,a) = ${r.gba}]] s. The dashed boxes are the windows, the grey bars the start times the window allows, the coloured bars the start times left after propagation. `;
  s += r.notes.map(t => t[0].toUpperCase() + t.slice(1) + ".").join(" ");
  if (r.conflict) s += ` <b>Conflict</b>: ${r.conflict}. A clause-learning solver records why, as a new clause that is true in every plan: [[${r.learned}]]. It will never try this combination again, anywhere in the search.`;
  caption($("#ppCap"), s);
}
for (const k of ["both", "one", "none"]) $("#pk-" + k).addEventListener("click", () => {
  pp.kind = k; pp.idx = 0; pp.ord = "free"; $("#ppOrd").value = "free";
  for (const q of ["both", "one", "none"]) $("#pk-" + q).setAttribute("aria-pressed", String(q === k));
  $("#ppOrd").disabled = k !== "both"; resetPairState(); drawPair();
});
$("#ppIdx").addEventListener("input", e => { pp.idx = +e.target.value; resetPairState(); drawPair(); });
$("#ppXa").addEventListener("change", e => { pp.xa = e.target.value; drawPair(); });
$("#ppXb").addEventListener("change", e => { pp.xb = e.target.value; drawPair(); });
$("#ppOrd").addEventListener("change", e => { pp.ord = e.target.value; drawPair(); });
$("#ppLa").addEventListener("input", e => { pp.lbA = +e.target.value; drawPair(); });
$("#ppLb").addEventListener("input", e => { pp.lbB = +e.target.value; drawPair(); });

/* ---------------- (3) CP-SAT's anytime traces ---------------- */
const tr = { which: "small" };
function stepLine(points, tEnd) {
  const xs = [], ys = [];
  points.forEach(([t, v], q) => { if (q) { xs.push(t); ys.push(points[q - 1][1]); } xs.push(t); ys.push(v); });
  if (points.length) { xs.push(tEnd); ys.push(points[points.length - 1][1]); }
  return [xs, ys];
}
function drawTrace() {
  const cv = $("#trC"), narrow = narrowOf(cv);
  let s;
  if (tr.which === "small") {
    const R = D.smallRun, opt = I.python.cpsat.value, g = I.python.greedy.value;
    const tEnd = R.time;
    const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, Math.ceil(tEnd * 10) / 10], y: [120, 270], xlabel: "wall time [s]", ylabel: "priority", yticks: [150, 193, 219, 250] });
    const [vx, vy] = stepLine(R.values, tEnd), [bx, by] = stepLine(R.bounds, tEnd);
    plotLine(fr, bx, by, { color: css("--warm"), width: 2 });
    plotLine(fr, vx, vy, { color: css("--post"), width: 2.2 });
    fr.ctx.strokeStyle = css("--c4"); fr.ctx.setLineDash([2, 3]); seg(fr.ctx, fr.X(0), fr.Y(g), fr.X(fr.x1), fr.Y(g)); fr.ctx.setLineDash([]);
    const b0 = R.bounds[0], hit = R.values.find(v => v[1] === opt);
    s = `The small instance, CP-SAT with 8 workers and the greedy hint, as recorded by <code>ch08_numbers.py</code> (blue: best plan so far; orange: best bound so far, clipped above 270; green dotted: the greedy's ${g}). The first bound, ${fmt(b0[1], 0)} at ${fmt(b0[0], 3)} s, is the sum of the priorities of all ${I.opps.length} opportunities: every [[x_o = 1]]. It falls within a few hundredths of a second (${R.bounds.slice(1, 4).map(b => fmt(b[1], 0)).join(", ")}); the first plan is worth ${fmt(R.values[0][1], 0)} at ${fmt(R.values[0][0], 3)} s${(() => { const h = R.values.find(v => v[1] === g); return h && h !== R.values[0] ? `, the hint's ${g} follows at ${fmt(h[0], 3)} s` : ""; })()}, and ${R.values.length} improving plans are reported in all. The optimum ${opt} is found at ${hit ? fmt(hit[0], 3) : "?"} s, and the search ends at ${fmt(tEnd, 2)} s when the bound reaches it too: status OPTIMAL.`;
  } else {
    const B = D.benchMedium, F = D.freshMedium;
    const tEnd = Math.max(B.time, F.time);
    const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, 62], y: [440, 600], xlabel: "wall time [s]", ylabel: "priority", yticks: [450, 500, 518, 550, 575, 600] });
    const [bx, by] = stepLine(F.bounds, F.time);
    plotLine(fr, bx, by, { color: css("--warm"), width: 2 });
    const [v1x, v1y] = stepLine(B.values, B.time), [v2x, v2y] = stepLine(F.values, F.time);
    plotLine(fr, v1x, v1y, { color: css("--post"), width: 2.2 });
    plotLine(fr, v2x, v2y, { color: css("--c5"), width: 1.8, dash: [5, 3] });
    fr.ctx.strokeStyle = css("--c4"); fr.ctx.setLineDash([2, 3]); seg(fr.ctx, fr.X(0), fr.Y(B.greedy), fr.X(fr.x1), fr.Y(B.greedy)); fr.ctx.setLineDash([]);
    const gapB = (B.bound - B.value) / B.bound, gapF = (F.bound - F.value) / F.bound;
    const firstAt = F.bounds.find(b => b[1] <= F.bound + 1e-9);
    s = `The medium instance (8 satellites, 1373 opportunities), 60 s limit, 8 workers, greedy hint. Blue: the benchmark's run (<code>results/benchmark.json</code>), from ${fmt(B.values[0][1], 0)} to <b>${B.value}</b> at the limit. Purple dashed: the run of <code>ch08_numbers.py</code>, same model, same parameters, same seed, from ${fmt(F.values[0][1], 0)} to <b>${F.value}</b>. Orange: that run's best bound, which reaches ${fmt(F.bound, 0)} at ${firstAt ? fmt(firstAt[0], 1) : "?"} s and never moves again (the benchmark's final bound is ${fmt(B.bound, 0)} too). Neither run proves optimality: status FEASIBLE, gaps ${fmt(100 * gapB, 1)} % and ${fmt(100 * gapF, 1)} %. The ${Math.abs(B.value - F.value)} point${Math.abs(B.value - F.value) === 1 ? "" : "s"} between the two runs are the run-to-run noise of a multi-threaded search stopped at a time limit (chapter 10).`;
    void tEnd;
  }
  $("#tr-small").setAttribute("aria-pressed", String(tr.which === "small"));
  $("#tr-medium").setAttribute("aria-pressed", String(tr.which === "medium"));
  caption($("#trCap"), s);
}
$("#tr-small").addEventListener("click", () => { tr.which = "small"; drawTrace(); });
$("#tr-medium").addEventListener("click", () => { tr.which = "medium"; drawTrace(); });

/* ---------------- start ---------------- */
syncBB(); newSearch(); finishSearch();
resetPairState();
function drawAll() { drawBB(); drawPair(); drawTrace(); }
drawAll(); onRedraw(drawAll);
})();
