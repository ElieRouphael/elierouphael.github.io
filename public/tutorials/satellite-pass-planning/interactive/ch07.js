/* Satellite Pass Planning from Scratch, chapter 07, A Greedy Plan and What It Leaves Behind: the live demos.
   The greedy of satplan/greedy.py, re-run here with a choice of target order and a record of every decision
   (it reproduces Sat.greedy and the Python plan exactly; the page checks it at load), stepped through on the
   course's small instance; three hand-made counterexamples solved by brute force; other target orders and
   random restarts; and two local improvements (drop one task and refill, ruin and recreate), every plan
   validated by Sat.checkPlan. code/ch07_numbers.py prints the same numbers in Python.
   Needs ../kit/kit.js, sat.js and instance-small.js. */
(() => {
"use strict";
const { $, css, fmt, caption, frame, fit, alpha, onRedraw, narrowOf, seg } = window.Kit;
const Sat = window.Sat, I = window.SAT_SMALL;
const OPT = I.python.cpsat.value, OPT_PLAN = new Map(Object.entries(I.python.cpsat.plan).map(([k, v]) => [+k, v]));

/* ---------------- the greedy, with a chosen target order ---------------- */
/* run(order, plan0, log): insert the targets of `order` one by one (skipping those already collected in
   plan0), each on its earliest opportunity that passes the budgets and fits, at the earliest feasible start
   among the window start and the ends-plus-transition of the tasks already on that satellite: greedy.py
   line for line. With log = [] it records, per target, every opportunity tried and why it failed. */
function makeGreedy(inst) {
  const opps = inst.opps, N = opps.length;
  const byTarget = new Map();
  for (const o of opps) { if (!byTarget.has(o.target)) byTarget.set(o.target, []); byTarget.get(o.target).push(o.idx); }
  for (const ids of byTarget.values()) ids.sort((a, b) => opps[a].ws - opps[b].ws);   // stable, as Python's sorted
  const trc = new Int32Array(N * N).fill(-1);
  const tr = (a, b) => { const k = a * N + b; if (trc[k] < 0) trc[k] = Sat.transition(inst, a, b); return trc[k]; };
  const firstSeen = [...byTarget.keys()];
  const groupKeys = o => ["duty:" + o.sat + ":" + o.rev, "memory:" + o.sat + ":" + o.mem];
  const limitOf = g => g.startsWith("duty") ? inst.dutyBudget : inst.memoryBudget;
  const valueOf = plan => { let v = 0; for (const i of plan.keys()) v += opps[i].w; return v; };

  function run(order, plan0, log) {
    const plan = new Map(plan0 || []);
    const schedule = []; for (let s = 0; s < inst.nSats; s++) schedule.push([]);
    const used = new Map(), taken = new Set();
    for (const [i, t] of plan) {
      const o = opps[i]; schedule[o.sat].push([t, i]); taken.add(o.target);
      for (const g of groupKeys(o)) used.set(g, (used.get(g) || 0) + o.dur);
    }
    const fits = (o, t) => {
      for (const [tp, p] of schedule[o.sat]) {
        if (t >= tp) { if (t < tp + opps[p].dur + tr(p, o.idx)) return false; }
        else if (t + o.dur + tr(o.idx, p) > tp) return false;
      }
      return true;
    };
    /* the tasks whose "no start here" zone (tp - d - s(o,p), tp + d_p + s(p,o)) cuts o's start interval */
    const blockersOf = o => schedule[o.sat].map(([tp, p]) => ({ p, tp, lo: tp - o.dur - tr(o.idx, p), hi: tp + opps[p].dur + tr(p, o.idx) }))
      .filter(b => b.lo < Sat.latest(o) && b.hi > o.ws);
    let value = valueOf(plan);
    for (const j of order) {
      if (taken.has(j)) continue;
      const tries = []; let placed = false;
      for (const i of byTarget.get(j)) {
        const o = opps[i], gs = groupKeys(o);
        const full = gs.find(g => (used.get(g) || 0) + o.dur > limitOf(g));
        if (full !== undefined) {
          if (log) tries.push({ i, res: "budget", group: full, used: used.get(full) || 0, limit: limitOf(full) });
          continue;
        }
        const cands = [o.ws];
        for (const [tp, p] of schedule[o.sat]) cands.push(tp + opps[p].dur + tr(p, i));
        const ok = cands.filter(c => c >= o.ws && c <= Sat.latest(o)).sort((x, y) => x - y);
        const t = ok.find(c => fits(o, c));
        if (t === undefined) { if (log) tries.push({ i, res: "blocked", blockers: blockersOf(o) }); continue; }
        if (log) tries.push({ i, res: "placed", start: t, neighbours: schedule[o.sat].length });
        plan.set(i, t); schedule[o.sat].push([t, i]); taken.add(j);
        for (const g of gs) used.set(g, (used.get(g) || 0) + o.dur);
        value += o.w; placed = true;
        break;
      }
      if (log) {
        const kinds = new Set(tries.map(x => x.res));
        log.push({ target: j, tries, value, outcome: placed ? "placed" : kinds.size === 1 ? [...kinds][0] : "mixed" });
      }
    }
    return plan;
  }

  /* target orders: a key (array, compared left to right), ties by order of first appearance */
  function orderBy(key) {
    return firstSeen.map((j, r) => [j, key(j).concat([r])]).sort((A, B) => {
      for (let q = 0; q < A[1].length; q++) if (A[1][q] !== B[1][q]) return A[1][q] - B[1][q];
      return 0;
    }).map(x => x[0]);
  }
  const W = inst.targetWeights, D = inst.targetDurations, nOf = j => byTarget.get(j).length;
  const orders = {
    density: orderBy(j => [-W[j] / D[j], nOf(j)]),
    weight: orderBy(j => [-W[j], nOf(j)]),
    shortest: orderBy(j => [D[j], -W[j]]),
    fewest: orderBy(j => [nOf(j), -W[j] / D[j]]),
  };
  return { run, valueOf, orders, firstSeen, byTarget, tr };
}

/* Fisher-Yates from the end with the page's generator, as ch07_numbers.py's shuffled(). */
function shuffled(items, rng) {
  const a = items.slice();
  for (let k = a.length - 1; k > 0; k--) { const j = Math.floor(rng.next() * (k + 1)); const x = a[k]; a[k] = a[j]; a[j] = x; }
  return a;
}
const G = makeGreedy(I);
const LOG = [];
const GPLAN = G.run(G.orders.density, null, LOG);
const STEPS = LOG.length;
/* self-check against the shared port and the Python plan */
const pyPlan = I.python.greedy.plan, satG = Sat.greedy(I);
const SAME = GPLAN.size === Object.keys(pyPlan).length && [...GPLAN].every(([i, t]) => pyPlan[String(i)] === t)
  && satG.plan.size === GPLAN.size && [...GPLAN].every(([i, t]) => satG.plan.get(i) === t) && Sat.checkPlan(I, GPLAN).length === 0;

/* ---------------- shared drawing helpers ---------------- */
const WALPHA = { 1: 0.35, 2: 0.55, 4: 0.8, 8: 1 };
const wcol = w => alpha("--post", WALPHA[w] || 1);
const density = j => I.targetWeights[j] / I.targetDurations[j];
const groupWords = g => { const [kind, s, k] = g.split(":"); return kind === "duty" ? `the power budget of revolution ${k} on satellite ${s}` : `the memory budget of segment ${k} on satellite ${s}`; };

/* ---------------- hero: the greedy, decision by decision ---------------- */
const hero = { k: STEPS, showOpt: true, timer: null };
function planAfter(k) { const m = new Map(); for (let q = 0; q < k; q++) { const e = LOG[q]; const pl = e.tries.find(x => x.res === "placed"); if (pl) m.set(pl.i, pl.start); } return m; }
function tallies(k) { const c = { placed: 0, budget: 0, blocked: 0, mixed: 0 }; for (let q = 0; q < k; q++) c[LOG[q].outcome]++; return c; }
function focusTry(e) { return e.tries.find(x => x.res === "placed") || e.tries.find(x => x.res === "blocked") || e.tries[0]; }

function drawGantt() {
  const cv = $("#gantt"), narrow = narrowOf(cv, 600);
  const { ctx, w, h } = fit(cv, narrow ? 1.4 : 2.8);
  const padL = 40, padR = 8, padT = 6, padB = 22, iw = w - padL - padR;
  const X = t => padL + t / I.horizon * iw, rowH = (h - padT - padB) / I.nSats;
  ctx.font = "11px " + css("--font-mono");
  for (let s = 0; s < I.nSats; s++) {
    const y0 = padT + s * rowH;
    ctx.fillStyle = alpha("--ink", 0.035); ctx.fillRect(padL, y0 + 1, iw, rowH - 2);
    ctx.fillStyle = alpha("--c4", 0.2);
    for (const [c0, c1] of I.contacts[s]) ctx.fillRect(X(c0), y0 + 1, Math.max(1, X(c1) - X(c0)), rowH - 2);
    ctx.fillStyle = css("--muted"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText("sat " + s, padL - 5, y0 + rowH / 2);
  }
  ctx.strokeStyle = alpha("--muted", 0.6); ctx.setLineDash([3, 3]);
  for (let r = 1; r * I.period < I.horizon; r++) seg(ctx, Math.round(X(r * I.period)) + 0.5, padT, Math.round(X(r * I.period)) + 0.5, h - padB);
  ctx.setLineDash([]);
  ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let hr = 0; hr <= 6; hr++) ctx.fillText(hr + " h", Math.min(w - padR - 10, Math.max(padL + 8, X(hr * 3600))), h - padB + 6);
  const plan = planAfter(hero.k);
  for (const [i, t] of plan) {
    const o = I.opps[i], y0 = padT + o.sat * rowH;
    ctx.fillStyle = wcol(o.w); ctx.fillRect(X(t), y0 + rowH * 0.16, Math.max(2, X(t + o.dur) - X(t)), rowH * 0.46);
  }
  if (hero.showOpt) {
    ctx.fillStyle = css("--c4");
    for (const [i, t] of OPT_PLAN) { const o = I.opps[i], y0 = padT + o.sat * rowH; ctx.fillRect(X(t), y0 + rowH * 0.7, Math.max(2, X(t + o.dur) - X(t)), rowH * 0.16); }
  }
  if (hero.k > 0 && hero.k <= STEPS) {
    const e = LOG[hero.k - 1];
    ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2;
    for (const tr of e.tries) {
      const o = I.opps[tr.i], y0 = padT + o.sat * rowH;
      ctx.strokeRect(X(o.ws) - 2, y0 + 2, Math.max(4, X(o.we) - X(o.ws)) + 4, rowH - 4);
      if (tr.res === "placed") { ctx.fillStyle = css("--warm"); ctx.fillRect(X(tr.start), y0 + rowH * 0.16, Math.max(3, X(tr.start + o.dur) - X(tr.start)), rowH * 0.46); }
    }
    ctx.lineWidth = 1;
  }
}

function drawZoom() {
  const cv = $("#zoom"), narrow = narrowOf(cv, 600);
  if (hero.k === 0) {
    const { ctx, w, h } = fit(cv, narrow ? 2.6 : 5.5);
    ctx.fillStyle = css("--muted"); ctx.font = "13px " + css("--font-ui"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("Press “step”: the close-up of each decision appears here.", w / 2, h / 2);
    return;
  }
  const e = LOG[hero.k - 1], ft = focusTry(e), o = I.opps[ft.i];
  const a = Math.max(0, o.ws - 240), b = o.we + 240;
  const fr = frame(cv, narrow ? 2.6 : 5.5, { x: [a, b], y: [0, 1], yticks: false, xlabel: `time on satellite ${o.sat} [s]`, pad: { t: 18 } });
  const { ctx, X, Y } = fr;
  const before = planAfter(hero.k - 1);
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, 0, fr.iw, fr.pad.t + fr.ih); ctx.clip();
  ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "bottom"; ctx.textAlign = "center";
  // tasks already on this satellite: busy (solid) and reconfiguring towards o (light)
  for (const [i, t] of before) {
    const p = I.opps[i];
    if (p.sat !== o.sat || t > b || t + p.dur + 60 < a) continue;
    ctx.fillStyle = alpha("--post", 0.22); ctx.fillRect(X(t + p.dur), Y(0.72), X(t + p.dur + G.tr(i, o.idx)) - X(t + p.dur), Y(0.42) - Y(0.72));
    ctx.fillStyle = wcol(p.w); ctx.fillRect(X(t), Y(0.72), Math.max(2, X(t + p.dur) - X(t)), Y(0.42) - Y(0.72));
    ctx.fillStyle = css("--ink-2"); ctx.fillText(`t${p.target} · ${p.w}`, X(t + p.dur / 2), Y(0.74));
  }
  // o's window and the starts the tasks already there rule out
  const L = Sat.latest(o);
  ctx.strokeStyle = css("--warm"); ctx.setLineDash([5, 3]); ctx.lineWidth = 1.5;
  ctx.strokeRect(X(o.ws), Y(0.95), X(o.we) - X(o.ws), Y(0.3) - Y(0.95)); ctx.setLineDash([]); ctx.lineWidth = 1;
  ctx.fillStyle = css("--warm"); ctx.textAlign = "left"; ctx.fillText(`window of target ${o.target}`, X(o.ws) + 3, Y(0.95) - 1);
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 3; seg(ctx, X(o.ws), Y(0.17), X(L), Y(0.17)); ctx.lineWidth = 1;
  for (const [i, t] of before) {
    const p = I.opps[i]; if (p.sat !== o.sat) continue;
    const lo = Math.max(o.ws, t - o.dur - G.tr(o.idx, i)), hi = Math.min(L, t + p.dur + G.tr(i, o.idx));
    if (hi > lo) { ctx.fillStyle = alpha("--warm", 0.55); ctx.fillRect(X(lo), Y(0.25), Math.max(1.5, X(hi) - X(lo)), Y(0.09) - Y(0.25)); }
  }
  if (ft.res === "placed") { ctx.fillStyle = css("--warm"); ctx.fillRect(X(ft.start), Y(0.72), Math.max(3, X(ft.start + o.dur) - X(ft.start)), Y(0.42) - Y(0.72)); }
  ctx.restore();
}

function drawValue() {
  const cv = $("#val"), narrow = narrowOf(cv, 600);
  const fr = frame(cv, narrow ? 2.4 : 5.2, { x: [0, STEPS], y: [0, 240], xlabel: "decisions taken (targets in greedy order)", ylabel: "collected priority", yticks: [0, 100, 193, 219] });
  const xs = [0], ys = [0]; LOG.forEach((e, q) => { xs.push(q + 1); ys.push(e.value); });
  const { ctx, X, Y } = fr;
  ctx.setLineDash([4, 4]); ctx.strokeStyle = css("--c4"); seg(ctx, X(0), Y(OPT), X(STEPS), Y(OPT)); ctx.setLineDash([]);
  plotLine2(fr, xs, ys, alpha("--post", 0.3), 1.5);
  plotLine2(fr, xs.slice(0, hero.k + 1), ys.slice(0, hero.k + 1), css("--post"), 2.2);
  ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(hero.k), Y(ys[hero.k]), 4, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = css("--c4"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ctx.fillText(`proven optimum ${OPT}`, X(2), Y(OPT) - 2);
}
function plotLine2(fr, xs, ys, color, width) { window.Kit.plotLine(fr, xs, ys, { color, width }); }

function describe(k) {
  if (k === 0) return `The empty plan. The greedy will consider the ${STEPS} targets that some satellite can see, densest first (priority per second of listening), and give each one its first opportunity that fits. Press <b>step</b> or <b>play</b>.`;
  const e = LOG[k - 1], j = e.target, wj = I.targetWeights[j], dj = I.targetDurations[j], n = G.byTarget.get(j).length;
  let s = `<b>Decision ${k} of ${STEPS}</b>: target ${j}, priority ${wj}, ${dj} s of listening, so ${fmt(density(j), 3)} per second; ${n} opportunit${n > 1 ? "ies" : "y"}. `;
  const parts = e.tries.map(x => {
    const o = I.opps[x.i], where = `satellite ${o.sat}, window ${o.ws}–${o.we} s`;
    if (x.res === "placed") return `<b>placed</b> on ${where} at <b>${x.start} s</b>${x.start === o.ws ? ", the window start" : ", the end of a task already there plus its reconfiguration time"}${x.neighbours ? ` (checked against the ${x.neighbours} task${x.neighbours > 1 ? "s" : ""} already on that satellite)` : ""}`;
    if (x.res === "budget") return `${where}: <b>budget full</b>, ${groupWords(x.group)} already holds ${x.used} of ${x.limit} s and this task needs ${o.dur} s`;
    const ts = x.blockers.map(bk => `target ${I.opps[bk.p].target} (priority ${I.opps[bk.p].w})`);
    return `${where}: <b>blocked</b>, every start from ${o.ws} to ${Sat.latest(o)} s is ruled out by ${ts.length > 1 ? "the tasks for " + ts.slice(0, -1).join(", ") + " and " + ts[ts.length - 1] : "the task for " + ts[0]}, already placed`;
  });
  s += parts.map((p, q) => (e.tries.length > 1 ? `Opportunity ${q + 1}: ` : "") + p + ".").join(" ");
  s += ` Collected so far: <b>${e.value}</b>.`;
  if (k === STEPS) s += ` That was the last target: the greedy plan has ${GPLAN.size} tasks and collects <b>${e.value}</b>, ${OPT - e.value} below the proven optimum of ${OPT} (${fmt(100 * (OPT - e.value) / OPT, 1)} %).${hero.showOpt ? " The green ticks under each row are an optimal plan found by CP-SAT (chapter 08)." : ""}`;
  if (!SAME) s += " <b>Warning: this page's greedy does not reproduce the Python plan.</b>";
  return s;
}
function drawHero() {
  drawGantt(); drawZoom(); drawValue();
  const k = hero.k, c = tallies(k);
  $("#stepV").textContent = `${k} of ${STEPS}`; $("#step").value = k;
  $("#rStep").textContent = `${k} / ${STEPS}`;
  $("#rTarget").textContent = k ? `${LOG[k - 1].target} (${I.targetWeights[LOG[k - 1].target]}, ${I.targetDurations[LOG[k - 1].target]} s)` : "–";
  $("#rDec").textContent = k ? LOG[k - 1].outcome : "–";
  $("#rVal").textContent = k ? `${LOG[k - 1].value} (optimum ${OPT})` : `0 (optimum ${OPT})`;
  $("#rTally").textContent = `${c.placed} / ${c.budget} / ${c.blocked} / ${c.mixed}`;
  caption($("#heroStatus"), describe(k));
}
function setStep(k) { hero.k = Math.max(0, Math.min(STEPS, k)); drawHero(); }
function stopPlay() { if (hero.timer) { clearInterval(hero.timer); hero.timer = null; } $("#play").textContent = "play"; $("#play").setAttribute("aria-pressed", "false"); }
$("#step").max = STEPS;
$("#step").addEventListener("input", e => { stopPlay(); setStep(+e.target.value); });
$("#back").addEventListener("click", () => { stopPlay(); setStep(hero.k - 1); });
$("#fwd").addEventListener("click", () => { stopPlay(); setStep(hero.k + 1); });
$("#reset").addEventListener("click", () => { stopPlay(); setStep(0); });
$("#end").addEventListener("click", () => { stopPlay(); setStep(STEPS); });
$("#play").addEventListener("click", () => {
  if (hero.timer) { stopPlay(); return; }
  if (hero.k >= STEPS) setStep(0);
  $("#play").textContent = "pause"; $("#play").setAttribute("aria-pressed", "true");
  hero.timer = setInterval(() => { if (hero.k >= STEPS) { stopPlay(); return; } setStep(hero.k + 1); }, 450);
});
$("#showOpt").addEventListener("change", e => { hero.showOpt = e.target.checked; drawHero(); });

/* ---------------- counterexamples ---------------- */
/* Each toy is a one-satellite instance in the shape Sat expects; all pointing directions are equal, so every
   reconfiguration time is the 10 s settle. tasks: [target, weight, duration, window start, window end, revolution]. */
const CX = {
  block: { name: "a dense task blocks two", tasks: p => [[0, p, 30, 25, 55, 0], [1, 3, 30, 0, 40, 0], [2, 3, 30, 50, 90, 0]], duty: () => 10000,
    param: { label: "priority of task A", min: 1, max: 8, step: 1, value: 4 }, span: [0, 100], labels: ["A", "B", "C"] },
  early: { name: "a budget spent too early", tasks: () => [[0, 4, 60, 0, 120, 0], [0, 4, 60, 300, 420, 1], [1, 3, 60, 130, 260, 0]], duty: p => p,
    param: { label: "power budget per revolution [s]", min: 60, max: 200, step: 10, value: 100 }, span: [0, 430], labels: ["X (pass 1)", "X (pass 2)", "Y"], boundary: 280 },
  knap: { name: "density misleads a knapsack", tasks: () => [[0, 2, 20, 0, 150, 0], [1, 9, 95, 0, 150, 0]], duty: p => p,
    param: { label: "power budget per revolution [s]", min: 20, max: 200, step: 5, value: 100 }, span: [0, 160], labels: ["S", "L"] },
};
const cx = { which: "block", p: { block: 4, early: 100, knap: 100 } };
function toyInstance(tasks, duty) {
  const opps = tasks.map(([j, w, d, ws, we, rev], k) => ({ idx: k, sat: 0, target: j, ws, we, dur: d, w, u: [1, 0, 0], rev, mem: 0 }));
  const nt = Math.max(...tasks.map(t => t[0])) + 1, W = new Array(nt).fill(0), D = new Array(nt).fill(1);
  for (const o of opps) { W[o.target] = o.w; D[o.target] = o.dur; }
  return { opps, settle: 10, slewRate: 2, dutyBudget: duty, memoryBudget: 10000, nSats: 1, nTargets: nt, targetWeights: W, targetDurations: D };
}
function permutations(a) {
  if (a.length <= 1) return [a.slice()];
  const out = [];
  a.forEach((x, q) => { for (const r of permutations(a.slice(0, q).concat(a.slice(q + 1)))) out.push([x].concat(r)); });
  return out;
}
/* Exhaustive optimum: every subset, every order, each task at its earliest start after the previous one;
   a candidate counts only if Sat.checkPlan accepts it. */
function bruteForce(inst) {
  const n = inst.opps.length; let best = { value: 0, plan: new Map() };
  for (let mask = 1; mask < (1 << n); mask++) {
    const ids = []; for (let i = 0; i < n; i++) if (mask & (1 << i)) ids.push(i);
    for (const perm of permutations(ids)) {
      const plan = new Map(); let ok = true, prev = -1, tprev = 0;
      for (const i of perm) {
        const o = inst.opps[i], t = prev < 0 ? o.ws : Math.max(o.ws, tprev + inst.opps[prev].dur + Sat.transition(inst, prev, i));
        if (t > Sat.latest(o)) { ok = false; break; }
        plan.set(i, t); prev = i; tprev = t;
      }
      if (!ok || Sat.checkPlan(inst, plan).length) continue;
      const v = Sat.planValue(inst, plan);
      if (v > best.value) best = { value: v, plan };
      break;
    }
  }
  return best;
}
function drawCx() {
  const spec = CX[cx.which], p = cx.p[cx.which];
  const inst = toyInstance(spec.tasks(p), spec.duty(p));
  const g = Sat.greedy(inst), opt = bruteForce(inst);
  const cv = $("#cxC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.5 : 3.0, { x: spec.span, y: [0, 3], yticks: false, xlabel: "time [s]", pad: { l: 70, t: 8 } });
  const { ctx, X, Y } = fr;
  ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle"; ctx.textAlign = "right"; ctx.fillStyle = css("--muted");
  ["windows", "greedy", "optimum"].forEach((lab, r) => ctx.fillText(lab, fr.pad.l - 6, Y(2.5 - r)));
  if (spec.boundary != null) {
    ctx.strokeStyle = css("--muted"); ctx.setLineDash([3, 3]); seg(ctx, X(spec.boundary), Y(3), X(spec.boundary), Y(0)); ctx.setLineDash([]);
    ctx.textAlign = "left"; ctx.fillText("revolution 1 →", X(spec.boundary) + 4, Y(2.93));
    ctx.textAlign = "right"; ctx.fillText("← revolution 0", X(spec.boundary) - 4, Y(2.93));
  }
  const nO = inst.opps.length, laneH = 0.8 / nO;
  inst.opps.forEach((o, q) => {
    const y = 2.9 - (q + 1) * laneH - 0.02;
    ctx.strokeStyle = css("--ink-2"); ctx.strokeRect(X(o.ws), Y(y + laneH * 0.85), X(o.we) - X(o.ws), Y(y) - Y(y + laneH * 0.85));
    ctx.fillStyle = css("--ink-2"); ctx.textAlign = "left"; ctx.fillText(`${spec.labels[q]}: ${o.w}, ${o.dur} s`, X(o.ws) + 3, Y(y + laneH * 0.42));
  });
  const lane = (plan, yc, col) => {
    for (const [i, t] of plan) {
      const o = inst.opps[i];
      ctx.fillStyle = col; ctx.fillRect(X(t), Y(yc + 0.3), X(t + o.dur) - X(t), Y(yc - 0.3) - Y(yc + 0.3));
      ctx.fillStyle = alpha("--ink", 0.12); ctx.fillRect(X(t + o.dur), Y(yc + 0.12), X(t + o.dur + 10) - X(t + o.dur), Y(yc - 0.12) - Y(yc + 0.12));
      ctx.fillStyle = css("--bg"); ctx.textAlign = "center"; ctx.fillText(spec.labels[i].split(" ")[0], X(t + o.dur / 2), Y(yc));
    }
  };
  lane(g.plan, 1.5, css("--warm")); lane(opt.plan, 0.5, css("--c4"));
  const lbl = spec.param.label.replace(" [s]", "");
  $("#cxPV").textContent = p + (cx.which === "block" ? "" : " s");
  $("#cxPL").textContent = spec.param.label;
  let why;
  if (cx.which === "block") why = p > 3 && p < 6 ? `Task A has the highest priority per second (${p}/30 against 3/30), so the greedy takes it first, at its only possible start, 25 s. Its 30 s of listening plus 10 s of settling on either side leave no room for B (which must end by 15 s) or C (which cannot start before 65 s, after its latest start of 60 s). B and C together are worth 6.` : p <= 3 ? `With priority ${p}, A is no denser than B and C${p === 3 ? " (a tie, broken by the target order, so A still goes first)" : ", so the greedy takes B and C first and A no longer fits"}.` : `With priority ${p}, A alone is worth at least as much as B and C together, and taking it is right.`;
  else if (cx.which === "early") why = p >= 120 ? `With ${p} s per revolution both 60 s tasks fit in revolution 0, so taking X on its first pass costs nothing.` : `X is denser (4/60 against 3/60), so the greedy handles it first and gives it its earliest opportunity, pass 1, in revolution 0. That spends 60 of the ${p} s of revolution 0's power budget, and Y, which can only be heard in revolution 0, no longer fits. X could have waited for pass 2 in revolution 1: the greedy never revisits a decision.`;
  else why = p < 95 ? `With ${p} s, only S fits (L needs 95 s), so the greedy's choice is right.` : p >= 115 ? `With ${p} s both fit (20 + 95 = 115 s): no conflict.` : `S has the better priority per second (2/20 = 0.1 against 9/95 = 0.095), so the greedy takes it first; with ${p} s per revolution, the 95 s of L no longer fit. L alone is worth 9.`;
  caption($("#cxCap"), `<b>${spec.name[0].toUpperCase() + spec.name.slice(1)}</b>, with ${lbl.toLowerCase()} ${p}${cx.which === "block" ? "" : " s"}. Top: the windows (outlined) with priority and duration. Middle: the plan of <code>Sat.greedy</code>, value <b>${g.value}</b>. Bottom: the best plan over every subset and every order, value <b>${opt.value}</b>${g.value < opt.value ? `, so the greedy loses ${opt.value - g.value}` : ", the same as the greedy"}. The grey stub after each task is its 10 s settle. ${why}`);
}
function syncCx() {
  const spec = CX[cx.which], r = $("#cxP");
  r.min = spec.param.min; r.max = spec.param.max; r.step = spec.param.step; r.value = cx.p[cx.which];
  for (const k of Object.keys(CX)) $("#cx-" + k).setAttribute("aria-pressed", String(k === cx.which));
  drawCx();
}
for (const k of Object.keys(CX)) $("#cx-" + k).addEventListener("click", () => { cx.which = k; syncCx(); });
$("#cxP").addEventListener("input", e => { cx.p[cx.which] = +e.target.value; drawCx(); });

/* ---------------- other orders and random restarts ---------------- */
const ORD = [["density", "priority per second (the package)"], ["weight", "priority alone"], ["fewest", "fewest opportunities first"], ["shortest", "shortest task first"]]
  .map(([k, lab]) => { const pl = G.run(G.orders[k]); return { k, lab, value: G.valueOf(pl), n: pl.size, ok: Sat.checkPlan(I, pl).length === 0 }; });
const rnd = { seed: 7, n: 1000, vals: [], best: null, bestPlan: null, job: 0 };
function runRandom() {
  const job = ++rnd.job, rng = Sat.rng(rnd.seed);
  rnd.vals = []; rnd.best = null; rnd.bestPlan = null;
  const chunk = () => {
    if (job !== rnd.job) return;
    for (let q = 0; q < 50 && rnd.vals.length < rnd.n; q++) {
      const pl = G.run(shuffled(G.firstSeen, rng)), v = G.valueOf(pl);
      rnd.vals.push(v);
      if (rnd.best === null || v > rnd.best) { rnd.best = v; rnd.bestPlan = pl; }
    }
    drawOrders();
    if (rnd.vals.length < rnd.n) setTimeout(chunk, 0);
  };
  chunk();
}
function drawOrders() {
  const rows = ORD.map(r => [r.lab, r.value]);
  const done = rnd.vals.length;
  if (done) rows.push([`best of ${done} random orders`, rnd.best], [`mean of ${done} random orders`, rnd.vals.reduce((s, v) => s + v, 0) / done]);
  const cv = $("#ordC"), narrow = narrowOf(cv);
  const { ctx, w, h } = fit(cv, narrow ? 1.3 : 2.6);
  const padL = narrow ? 8 : 230, padR = 46, padT = 8, padB = 22, iw = w - padL - padR, rh = (h - padT - padB) / rows.length;
  const X = v => padL + v / 240 * iw;
  ctx.font = "11px " + css("--font-mono");
  ctx.strokeStyle = css("--grid");
  for (const v of [0, 50, 100, 150, 200]) { seg(ctx, X(v) + 0.5, padT, X(v) + 0.5, h - padB); ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(String(v), X(v), h - padB + 5); }
  rows.forEach(([lab, v], r) => {
    const y = padT + r * rh;
    ctx.fillStyle = r === 0 ? css("--post") : r < ORD.length ? alpha("--post", 0.55) : alpha("--c5", 0.7);
    const by = narrow ? y + rh * 0.5 : y + rh * 0.2, bh = narrow ? rh * 0.42 : rh * 0.6;
    ctx.fillRect(X(0), by, X(v) - X(0), bh);
    ctx.fillStyle = css("--ink"); ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.fillText(Number.isInteger(v) ? String(v) : fmt(v, 1), X(v) + 4, by + bh / 2);
    ctx.fillStyle = css("--ink-2");
    if (narrow) { ctx.textAlign = "left"; ctx.fillText(lab, padL, y + rh * 0.27); }
    else { ctx.textAlign = "right"; ctx.fillText(lab, padL - 8, by + bh / 2); }
  });
  ctx.strokeStyle = css("--c4"); ctx.setLineDash([4, 3]); ctx.lineWidth = 1.5; seg(ctx, X(OPT), padT, X(OPT), h - padB); ctx.setLineDash([]); ctx.lineWidth = 1;
  ctx.fillStyle = css("--c4"); ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(String(OPT), X(OPT) + 3, padT);
  drawHist();
  const allOk = ORD.every(r => r.ok) && (!rnd.bestPlan || Sat.checkPlan(I, rnd.bestPlan).length === 0);
  caption($("#ordCap"), `The same greedy, the same earliest-feasible insertion, only the order in which targets are considered changes. Priority per second gives ${ORD[0].value}; priority alone ${ORD[1].value}; fewest opportunities first ${ORD[2].value}; shortest first ${ORD[3].value}. ${done ? `${done} random orders drawn with <code>Sat.rng(${rnd.seed})</code>: the best collects <b>${rnd.best}</b>, the mean ${fmt(rnd.vals.reduce((s, v) => s + v, 0) / done, 1)}, the worst ${Math.min(...rnd.vals)}; ${rnd.vals.filter(v => v >= ORD[0].value).length} of them reach the density order's ${ORD[0].value}.` : ""} The dashed line is the proven optimum, ${OPT}. ${allOk ? "Every plan here passes <code>Sat.checkPlan</code>." : "<b>A plan failed the checker.</b>"}`);
}
function drawHist() {
  const cv = $("#histC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.8 : 3.4, { x: [80, 225], y: [0, 160], xlabel: "value of the plan from one random order", ylabel: "number of orders", xticks: [80, 100, 120, 140, 160, 180, 193, 219] });
  const { ctx, X, Y } = fr;
  const counts = new Map(); for (const v of rnd.vals) { const b = Math.floor(v / 5) * 5; counts.set(b, (counts.get(b) || 0) + 1); }
  ctx.fillStyle = alpha("--c5", 0.7);
  for (const [b, c] of counts) ctx.fillRect(X(b) + 0.5, Y(Math.min(c, 160)), X(b + 5) - X(b) - 1, Y(0) - Y(Math.min(c, 160)));
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = css("--post"); seg(ctx, X(ORD[0].value), Y(0), X(ORD[0].value), Y(160));
  ctx.strokeStyle = css("--c4"); ctx.setLineDash([4, 3]); seg(ctx, X(OPT), Y(0), X(OPT), Y(160)); ctx.setLineDash([]); ctx.lineWidth = 1;
}
$("#rndSeed").addEventListener("change", e => { rnd.seed = Math.max(1, Math.floor(+e.target.value) || 1); e.target.value = rnd.seed; runRandom(); });
$("#rndRun").addEventListener("click", () => { rnd.seed++; $("#rndSeed").value = rnd.seed; runRandom(); });

/* ---------------- local improvement ---------------- */
/* Drop one task and refill in density order; keep it if it collects more (tried for every task). */
function dropAndRefill(plan) {
  let cur = new Map(plan), improved = true, moves = 0, tried = 0;
  while (improved) {
    improved = false;
    const v0 = G.valueOf(cur);
    for (const r of [...cur.keys()].sort((a, b) => a - b)) {
      const rest = new Map(cur); rest.delete(r); tried++;
      const cand = G.run(G.orders.density, rest);
      if (G.valueOf(cand) > v0) { cur = cand; moves++; improved = true; break; }
    }
  }
  return { plan: cur, moves, tried };
}
const DROP = dropAndRefill(GPLAN);
/* Ruin and recreate: a random satellite and stretch, its tasks removed, a random-order greedy refill;
   accepted if it collects at least as much. ch07_numbers.py's ruin_and_recreate, draw for draw. */
const ls = { win: 3600, seed: 1, iters: 2000, it: 0, v: 0, cur: null, hist: [], rng: null, job: 0 };
function runLS() {
  const job = ++ls.job;
  ls.rng = Sat.rng(ls.seed); ls.cur = new Map(GPLAN); ls.v = G.valueOf(GPLAN); ls.it = 0; ls.hist = [[0, ls.v]];
  const chunk = () => {
    if (job !== ls.job) return;
    for (let q = 0; q < 40 && ls.it < ls.iters; q++) {
      ls.it++;
      const s = Math.floor(ls.rng.next() * I.nSats), t0 = ls.rng.next() * (I.horizon - ls.win);
      const rest = new Map(); for (const [i, t] of ls.cur) if (!(I.opps[i].sat === s && t0 <= t && t < t0 + ls.win)) rest.set(i, t);
      const cand = G.run(shuffled(G.firstSeen, ls.rng), rest), vc = G.valueOf(cand);
      if (vc >= ls.v) { if (vc > ls.v) ls.hist.push([ls.it, vc]); ls.cur = cand; ls.v = vc; }
    }
    drawLS();
    if (ls.it < ls.iters) setTimeout(chunk, 0);
  };
  chunk();
}
function drawLS() {
  const cv = $("#lsC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.6 : 3.0, { x: [0, ls.iters], y: [185, 225], xlabel: "iteration", ylabel: "collected priority", yticks: [190, 193, 200, 210, 219] });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--c4"); ctx.setLineDash([4, 3]); seg(ctx, X(0), Y(OPT), X(ls.iters), Y(OPT)); ctx.setLineDash([]);
  const xs = [], ys = [];
  ls.hist.forEach(([it, v], q) => { if (q) { xs.push(it); ys.push(ls.hist[q - 1][1]); } xs.push(it); ys.push(v); });
  xs.push(ls.it); ys.push(ls.v);
  window.Kit.plotLine(fr, xs, ys, { color: css("--post"), width: 2 });
  const errs = Sat.checkPlan(I, ls.cur);
  $("#lsVal").textContent = `${ls.v} after ${ls.it}`;
  $("#lsImp").textContent = String(ls.hist.length - 1);
  $("#lsChk").textContent = errs.length ? `${errs.length} violations` : "feasible";
  $("#lsDrop").textContent = DROP.moves ? `${G.valueOf(DROP.plan)} (${DROP.moves} moves)` : `no move helps (${DROP.tried} tried)`;
  const last = ls.hist[ls.hist.length - 1];
  caption($("#lsCap"), `Ruin and recreate from the greedy plan (${G.valueOf(GPLAN)}): at each iteration a random satellite and a random stretch of ${ls.win / 60} min are drawn with <code>Sat.rng(${ls.seed})</code>, the tasks starting in it are removed, and the greedy refills the plan with the targets in a random order. The new plan is kept if it collects at least as much. After ${ls.it} iterations: <b>${ls.v}</b>${ls.hist.length > 1 ? `, last improved at iteration ${last[0]}` : ""}, ${OPT - ls.v} below the optimum (${fmt(100 * (OPT - ls.v) / OPT, 1)} %); the checker finds the plan ${errs.length ? "<b>infeasible</b>" : "feasible"}. Dropping a single task and refilling in density order, tried for all ${GPLAN.size} tasks, ${DROP.moves ? `improves the plan to ${G.valueOf(DROP.plan)}` : "never helps: the greedy plan is a local optimum for that move"}.`);
}
$("#lsWin").addEventListener("change", e => { ls.win = +e.target.value * 60; runLS(); });
$("#lsSeed").addEventListener("change", e => { ls.seed = Math.max(1, Math.floor(+e.target.value) || 1); e.target.value = ls.seed; runLS(); });
$("#lsRun").addEventListener("click", () => { ls.seed++; $("#lsSeed").value = ls.seed; runLS(); });

/* ---------------- start ---------------- */
function drawAll() { drawHero(); drawCx(); drawOrders(); if (ls.cur) drawLS(); }
syncCx(); drawHero(); drawOrders(); runRandom(); runLS();
onRedraw(drawAll);
})();
