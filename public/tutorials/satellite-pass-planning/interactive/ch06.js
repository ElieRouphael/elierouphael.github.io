/* Satellite Pass Planning from Scratch, chapter 06, Constraints and an Independent Checker: the live demos.
   (1) One satellite's pass from the course's small instance: tasks can be switched on and off and their starts
   dragged or slid, and the page runs Sat.checkPlan (the port of satplan/checker.py) on the whole plan at every
   change, with the transitions between consecutive tasks and the power and memory budgets of that satellite.
   (2) The three cases of the conflict-pair sweep (Sat.conflictPairs, the port of build_pairs): for one pair,
   both orders placed as early as possible. (3) How far the pointing direction turns inside one window, the
   approximation behind fixed transition times. Needs ../kit/kit.js, sat.js and instance-small.js. */
(() => {
"use strict";
const { $, css, fmt, caption, fit, alpha, onRedraw, narrowOf, seg, ticks } = window.Kit;
const Sat = window.Sat;
const D = window.SAT_SMALL;
const OPPS = D.opps;
const toMap = obj => new Map(Object.entries(obj).map(([k, v]) => [+k, v]));
const PLANS = { cpsat: toMap(D.python.cpsat.plan), greedy: toMap(D.python.greedy.plan) };
const GROUPS = Sat.budgetGroups(D);

/* passes: a satellite's windows sorted by start, cut wherever the next window starts more than 300 s after the
   latest end so far */
function passesOf(s) {
  const list = OPPS.filter(o => o.sat === s).sort((a, b) => a.ws - b.ws || a.idx - b.idx), out = [];
  let cur = [], end = -Infinity;
  for (const o of list) {
    if (cur.length && o.ws > end + 300) { out.push(cur); cur = []; end = -Infinity; }
    cur.push(o); end = Math.max(end, o.we);
  }
  if (cur.length) out.push(cur);
  return out;
}
const hms = t => { t = Math.round(t); const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), s = t % 60; return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`; };

/* ======================= hero: one pass, live checker ======================= */
const st = { sat: 3, pass: 2, sel: 0, plan: new Map(PLANS.cpsat), source: "cpsat" };
let geo = null, drag = null;
const pass = () => passesOf(st.sat)[st.pass];
const view = () => { const p = pass(); return [Math.min(...p.map(o => o.ws)) - 60, Math.max(...p.map(o => o.we)) + 60]; };
/* tasks of satellite s in start order, as [opp id, start] */
const tasksOf = s => [...st.plan].filter(([i]) => OPPS[i].sat === s).sort((a, b) => a[1] - b[1] || a[0] - b[0]);

function drawTimeline() {
  const cv = $("#tlC"), rows = pass(), [v0, v1] = view();
  const rowH = 16, top = 24, laneH = 26, rowsTop = top + laneH + 16, bottom = 30;
  const wCss = cv.clientWidth || cv.parentElement.clientWidth, hCss = rowsTop + rows.length * rowH + bottom;
  const { ctx, w } = fit(cv, wCss / hCss);
  const L = 62, R = 10, iw = w - L - R;
  const X = t => L + (t - v0) / (v1 - v0) * iw;
  geo = { X, Xinv: x => v0 + (x - L) / iw * (v1 - v0), rowsTop, rowH, rows, L, R, w };
  ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle";
  // grid and axis
  const tk = ticks(v0, v1, Math.max(3, Math.floor(iw / 80)));
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  for (const t of tk) seg(ctx, Math.round(X(t)) + 0.5, top, Math.round(X(t)) + 0.5, rowsTop + rows.length * rowH);
  ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of tk) ctx.fillText(String(t), Math.min(w - R - 18, Math.max(L + 18, X(t))), rowsTop + rows.length * rowH + 6);
  ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText("time [s]", w - R, rowsTop + rows.length * rowH + bottom);
  ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(`satellite ${st.sat}: the plan, then one row per window`, L, 4);
  // contacts in view, as a faint band behind everything
  ctx.fillStyle = alpha("--c4", 0.12);
  for (const [c0, c1] of D.contacts[st.sat]) if (c1 > v0 && c0 < v1) ctx.fillRect(X(Math.max(c0, v0)), top, X(Math.min(c1, v1)) - X(Math.max(c0, v0)), rowsTop + rows.length * rowH - top);
  // summary lane: the satellite's tasks in time order, with s(a, b) after each one
  const tasks = tasksOf(st.sat), y0 = top + 4;
  ctx.textBaseline = "middle"; ctx.textAlign = "right"; ctx.fillStyle = css("--ink-2"); ctx.fillText("plan", L - 6, y0 + laneH / 2 - 2);
  ctx.save(); ctx.beginPath(); ctx.rect(L, 0, iw, rowsTop + rows.length * rowH); ctx.clip();
  for (let k = 0; k < tasks.length; k++) {
    const [a, ta] = tasks[k], oa = OPPS[a], endA = ta + oa.dur;
    ctx.fillStyle = css("--post"); ctx.fillRect(X(ta), y0, Math.max(1.5, X(endA) - X(ta)), laneH - 8);
    if (k + 1 < tasks.length) {
      const [b, tb] = tasks[k + 1], s = Sat.transition(D, a, b), need = endA + s;
      ctx.fillStyle = alpha("--ink", 0.18); ctx.fillRect(X(endA), y0 + 5, X(need) - X(endA), laneH - 18);
      if (tb < need) { ctx.fillStyle = css("--warm"); ctx.fillRect(X(Math.max(tb, ta)), y0 - 3, X(need) - X(Math.max(tb, ta)), 3); ctx.fillRect(X(Math.max(tb, ta)), y0 + laneH - 8, X(need) - X(Math.max(tb, ta)), 3); }
    }
  }
  // window rows
  rows.forEach((o, r) => {
    const y = rowsTop + r * rowH;
    if (r === st.sel) { ctx.fillStyle = css("--accent-wash"); ctx.fillRect(L, y, iw, rowH); }
    ctx.fillStyle = alpha("--post", 0.16); ctx.fillRect(X(o.ws), y + 3, X(o.we) - X(o.ws), rowH - 6);
    ctx.strokeStyle = alpha("--post", 0.7); ctx.lineWidth = 1.5; seg(ctx, X(o.ws), y + rowH - 3, X(Sat.latest(o)), y + rowH - 3);
    if (st.plan.has(o.idx)) {
      const t = st.plan.get(o.idx), bad = t < o.ws || t > Sat.latest(o);
      ctx.fillStyle = css("--post"); ctx.fillRect(X(t), y + 2, Math.max(2, X(t + o.dur) - X(t)), rowH - 4);
      if (bad) { ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2; ctx.strokeRect(X(t) - 1, y + 1, X(t + o.dur) - X(t) + 2, rowH - 2); }
    }
  });
  ctx.restore();
  ctx.textAlign = "right"; ctx.font = "10px " + css("--font-mono");
  rows.forEach((o, r) => { ctx.fillStyle = r === st.sel ? css("--ink") : css("--muted"); ctx.fillText(`${o.idx} · ${o.w}`, L - 6, rowsTop + r * rowH + rowH / 2); });
}

function drawBudgets() {
  const cv = $("#budC"), s = st.sat;
  const revs = [...new Set(OPPS.filter(o => o.sat === s).map(o => o.rev))].sort((a, b) => a - b);
  const mems = [...new Set(OPPS.filter(o => o.sat === s).map(o => o.mem))].sort((a, b) => a - b);
  const items = [...revs.map(r => ({ kind: "duty", seg: r, key: s + ":" + r, label: `power, rev ${r}` })), ...mems.map(m => ({ kind: "memory", seg: m, key: s + ":" + m, label: `memory, m${m}` }))];
  const rowH = 20, hCss = items.length * rowH + 8, wCss = cv.clientWidth || cv.parentElement.clientWidth;
  const { ctx, w } = fit(cv, wCss / hCss);
  const L = 96, Rr = 92, iw = Math.max(40, w - L - Rr);
  ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle";
  const out = [];
  items.forEach((it, k) => {
    const limit = it.kind === "duty" ? D.dutyBudget : D.memoryBudget;
    const members = OPPS.filter(o => o.sat === s && (it.kind === "duty" ? o.rev : o.mem) === it.seg);
    const requested = members.reduce((a, o) => a + o.dur, 0), used = members.reduce((a, o) => a + (st.plan.has(o.idx) ? o.dur : 0), 0);
    const kept = GROUPS.some(g => g.kind === it.kind && g.key === it.key);
    const y = 4 + k * rowH, full = 1.4 * limit, Xb = v => L + Math.min(v, full) / full * iw;
    ctx.fillStyle = css("--ink-2"); ctx.textAlign = "right"; ctx.fillText(it.label, L - 6, y + rowH / 2);
    ctx.fillStyle = css("--surface-2"); ctx.fillRect(L, y + 4, iw, rowH - 8);
    ctx.fillStyle = used > limit ? css("--warm") : css("--post"); ctx.fillRect(L, y + 4, Xb(used) - L, rowH - 8);
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; seg(ctx, Xb(limit), y + 1, Xb(limit), y + rowH - 1);
    ctx.textAlign = "left"; ctx.fillStyle = used > limit ? css("--warm") : css("--ink-2");
    ctx.fillText(`${used}/${limit} s${kept ? "" : " ·"}`, L + iw + 6, y + rowH / 2);
    out.push({ ...it, used, limit, requested, kept });
  });
  return out;
}

function heroText(budgets) {
  const errs = Sat.checkPlan(D, st.plan), value = Sat.planValue(D, st.plan);
  $("#rVal").textContent = value;
  $("#rTasks").textContent = `${tasksOf(st.sat).length} (${[...st.plan.keys()].length} in all)`;
  $("#rErr").textContent = errs.length;
  const box = $("#hViol"); box.innerHTML = "";
  if (!errs.length) { const li = document.createElement("li"); li.textContent = "no violation: the checker returns an empty list"; box.appendChild(li); }
  errs.slice(0, 8).forEach(e => { const li = document.createElement("li"); li.textContent = e; box.appendChild(li); });
  if (errs.length > 8) { const li = document.createElement("li"); li.textContent = `… and ${errs.length - 8} more`; box.appendChild(li); }
  // transitions between consecutive tasks of this pass
  const ids = new Set(pass().map(o => o.idx)), tasks = tasksOf(st.sat), lines = [];
  for (let k = 0; k + 1 < tasks.length; k++) {
    const [a, ta] = tasks[k], [b, tb] = tasks[k + 1];
    if (!ids.has(a) || !ids.has(b)) continue;
    const s = Sat.transition(D, a, b), gap = tb - ta - OPPS[a].dur;
    lines.push(`${a} → ${b}: needs ${s} s, has ${fmt(gap, 0)} s${gap < s ? "  ✗" : gap === s ? "  (no slack)" : ""}`);
  }
  $("#hTrans").textContent = lines.length ? lines.join("\n") : "fewer than two tasks in this pass";
  const o = pass()[st.sel], inPlan = st.plan.has(o.idx);
  $("#hOn").checked = inPlan; $("#hStart").disabled = !inPlan;
  $("#hStartV").textContent = inPlan ? `${fmt(st.plan.get(o.idx), 0)} s (window allows ${o.ws} to ${Sat.latest(o)})` : "not in the plan";
  const over = budgets.filter(b => b.used > b.limit);
  const tight = lines.filter(l => l.includes("no slack")).length;
  const srcName = st.source === "cpsat" ? "the optimal CP-SAT plan of chapter 08" : st.source === "greedy" ? "the greedy plan of chapter 07" : "your edited plan";
  let text = `Satellite ${st.sat}, pass ${st.pass + 1} of ${passesOf(st.sat).length}: ${pass().length} opportunities between ${pass()[0].ws} and ${Math.max(...pass().map(q => q.we))} s (rows: window in light blue, allowed start range as the line beneath, task as the solid block; labels are opportunity id · weight). Plan: ${srcName}, value <b>${value}</b>. `;
  if (!errs.length) text += `<b>The checker finds no violation.</b> ${lines.length ? `Between consecutive tasks of this pass, ${tight} of ${lines.length} gaps are exactly the reconfiguration time, with no slack at all.` : ""} `;
  else text += `<b>The checker finds ${errs.length} violation${errs.length > 1 ? "s" : ""}.</b> The first: "${errs[0]}". `;
  text += `Opportunity ${o.idx} (target ${o.target}, weight ${o.w}, ${o.dur} s) is ${inPlan ? `planned at ${fmt(st.plan.get(o.idx), 0)} s` : "not planned"}. `;
  if (over.length) text += `Over budget: ${over.map(b => `${b.label} (${b.used} s of ${b.limit})`).join(", ")}. `;
  else { const fullB = budgets.filter(b => b.used === b.limit); if (fullB.length) text += `Exactly full: ${fullB.map(b => b.label).join(", ")}, so any further task in ${fullB.length > 1 ? "those segments" : "that segment"} breaks the budget. `; }
  caption($("#heroStatus"), text);
}
function drawHero() { drawTimeline(); heroText(drawBudgets()); }

/* controls */
const hSat = $("#hSat"), hPass = $("#hPass"), hTask = $("#hTask"), hOn = $("#hOn"), hStart = $("#hStart");
function fillPasses() {
  hPass.innerHTML = "";
  passesOf(st.sat).forEach((p, k) => { const op = document.createElement("option"); op.value = k; op.textContent = `pass ${k + 1}: ${p.length} windows from ${hms(p[0].ws)}`; hPass.appendChild(op); });
  hPass.value = st.pass;
}
function fillTasks() {
  hTask.innerHTML = "";
  pass().forEach((o, r) => { const op = document.createElement("option"); op.value = r; op.textContent = `opp ${o.idx} · target ${o.target} · weight ${o.w} · ${o.dur} s`; hTask.appendChild(op); });
  hTask.value = st.sel;
}
function syncSlider() {
  const o = pass()[st.sel];
  hStart.min = o.ws - 30; hStart.max = Sat.latest(o) + 30; hStart.step = 1;
  hStart.value = st.plan.has(o.idx) ? st.plan.get(o.idx) : o.ws;
}
function select(r) { st.sel = r; hTask.value = r; syncSlider(); drawHero(); }
function edited() { st.source = "edited"; ["#hCp", "#hGr"].forEach(id => $(id).setAttribute("aria-pressed", "false")); }
hSat.addEventListener("change", () => { st.sat = +hSat.value; st.pass = 0; st.sel = 0; fillPasses(); fillTasks(); syncSlider(); drawHero(); });
hPass.addEventListener("change", () => { st.pass = +hPass.value; st.sel = 0; fillTasks(); syncSlider(); drawHero(); });
hTask.addEventListener("change", () => select(+hTask.value));
hOn.addEventListener("change", () => {
  const o = pass()[st.sel];
  if (hOn.checked) st.plan.set(o.idx, o.ws); else st.plan.delete(o.idx);
  edited(); syncSlider(); drawHero();
});
hStart.addEventListener("input", () => { const o = pass()[st.sel]; if (!st.plan.has(o.idx)) return; st.plan.set(o.idx, +hStart.value); edited(); drawHero(); });
function load(src) {
  st.plan = new Map(PLANS[src]); st.source = src;
  $("#hCp").setAttribute("aria-pressed", src === "cpsat"); $("#hGr").setAttribute("aria-pressed", src === "greedy");
  syncSlider(); drawHero();
}
$("#hCp").addEventListener("click", () => load("cpsat"));
$("#hGr").addEventListener("click", () => load("greedy"));
$("#hClear").addEventListener("click", () => { for (const o of pass()) st.plan.delete(o.idx); edited(); syncSlider(); drawHero(); });

/* pointer: click a row to select it; press on a task to drag it; click inside an empty window to add a task */
const tl = $("#tlC");
tl.addEventListener("pointerdown", ev => {
  if (!geo) return;
  const rc = tl.getBoundingClientRect(), x = ev.clientX - rc.left, y = ev.clientY - rc.top;
  const r = Math.floor((y - geo.rowsTop) / geo.rowH);
  if (r < 0 || r >= geo.rows.length) return;
  const o = geo.rows[r];
  if (r !== st.sel) { st.sel = r; hTask.value = r; }
  const t = geo.Xinv(x);
  if (st.plan.has(o.idx)) {
    const t0 = st.plan.get(o.idx);
    if (t >= t0 - 3 && t <= t0 + o.dur + 3) { drag = { o, off: t - t0 }; tl.setPointerCapture(ev.pointerId); ev.preventDefault(); }
  } else if (t >= o.ws && t <= o.we) {
    st.plan.set(o.idx, Math.round(Math.min(Sat.latest(o), Math.max(o.ws, t - o.dur / 2)))); edited();
  }
  syncSlider(); drawHero();
});
tl.addEventListener("pointermove", ev => {
  if (!drag || !geo) return;
  const rc = tl.getBoundingClientRect(), t = geo.Xinv(ev.clientX - rc.left) - drag.off;
  const v = Math.round(Math.min(Sat.latest(drag.o) + 30, Math.max(drag.o.ws - 30, t)));
  if (v !== st.plan.get(drag.o.idx)) { st.plan.set(drag.o.idx, v); edited(); hStart.value = v; drawHero(); }
});
const endDrag = () => { drag = null; };
tl.addEventListener("pointerup", endDrag); tl.addEventListener("pointercancel", endDrag);

/* ======================= widget: the three cases of a conflict pair ======================= */
const PAIRS = Sat.conflictPairs(D);
const pyPairs = D.python.pairs;
const samePairs = PAIRS.length === pyPairs.length && PAIRS.every((p, k) => p.a === pyPairs[k][0] && p.b === pyPairs[k][1] && p.sab === pyPairs[k][2] && p.sba === pyPairs[k][3] && +p.aFirst === pyPairs[k][4] && +p.bFirst === pyPairs[k][5]);
const CASES = {
  both: PAIRS.filter(p => p.aFirst && p.bFirst),
  one: PAIRS.filter(p => p.aFirst !== p.bFirst),
  neither: PAIRS.filter(p => !p.aFirst && !p.bFirst),
};
const pw = { kind: "both", k: 0 };
function drawPair() {
  const list = CASES[pw.kind], p = list[Math.min(pw.k, list.length - 1)], A = OPPS[p.a], B = OPPS[p.b];
  const cv = $("#pairC"), narrow = narrowOf(cv, 560);
  const { ctx, w, h } = fit(cv, narrow ? 1.25 : 2.3);
  const v0 = Math.min(A.ws, B.ws) - 20, v1 = Math.max(A.we, B.we, A.ws + A.dur + p.sab + B.dur, B.ws + B.dur + p.sba + A.dur) + 20;
  const L = narrow ? 64 : 92, R = 10, iw = w - L - R, X = t => L + (t - v0) / (v1 - v0) * iw;
  const laneH = (h - 34) / 2, rowH = Math.min(26, laneH / 3);
  ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle";
  const tk = ticks(v0, v1, Math.max(3, Math.floor(iw / 80)));
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  for (const t of tk) seg(ctx, Math.round(X(t)) + 0.5, 4, Math.round(X(t)) + 0.5, h - 28);
  ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of tk) ctx.fillText(String(t), Math.min(w - R - 18, Math.max(L + 18, X(t))), h - 24);
  ctx.textBaseline = "middle";
  const verdicts = [];
  [[A, B, p.sab, p.aFirst, `${p.a} then ${p.b}`], [B, A, p.sba, p.bFirst, `${p.b} then ${p.a}`]].forEach(([F, S, s, ok, name], lane) => {
    const y0 = 6 + lane * laneH;
    ctx.fillStyle = css("--ink"); ctx.textAlign = "left"; ctx.font = "600 11px " + css("--font-mono");
    ctx.fillText(name + (ok ? ": fits" : ": impossible"), L, y0 + 6);
    ctx.font = "11px " + css("--font-mono");
    [F, S].forEach((o, r) => {
      const y = y0 + 14 + r * rowH;
      ctx.fillStyle = css("--muted"); ctx.textAlign = "right"; ctx.fillText(`opp ${o.idx}`, L - 6, y + rowH / 2);
      ctx.fillStyle = alpha("--post", 0.16); ctx.fillRect(X(o.ws), y + 3, X(o.we) - X(o.ws), rowH - 6);
      ctx.strokeStyle = alpha("--post", 0.8); ctx.lineWidth = 1.5; seg(ctx, X(o.ws), y + rowH - 3, X(Sat.latest(o)), y + rowH - 3);
      ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1; seg(ctx, X(Sat.latest(o)), y + 1, X(Sat.latest(o)), y + rowH - 1);
    });
    // first task at its earliest start; the second as early as the transition allows
    const t1 = F.ws, ready = t1 + F.dur + s, t2 = Math.max(S.ws, ready), fits = ready <= Sat.latest(S);
    const yF = y0 + 14, yS = y0 + 14 + rowH;
    ctx.fillStyle = css("--post"); ctx.fillRect(X(t1), yF + 4, X(t1 + F.dur) - X(t1), rowH - 8);
    ctx.fillStyle = alpha("--ink", 0.22); ctx.fillRect(X(t1 + F.dur), yF + rowH / 2 - 2, X(ready) - X(t1 + F.dur), 4);
    ctx.fillStyle = fits ? css("--post") : alpha("--warm", 0.85); ctx.fillRect(X(t2), yS + 4, X(t2 + S.dur) - X(t2), rowH - 8);
    if (!fits) { ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2; seg(ctx, X(Sat.latest(S)), yS, X(ready), yS); }
    verdicts.push({ F, S, s, t1, ready, t2, fits });
    if (ok !== fits) console.warn("pair verdict mismatch", p);
  });
  $("#pkV").textContent = `${pw.k + 1} of ${list.length}`;
  const [v1r, v2r] = verdicts;
  const say = v => `${v.F.idx} first: it starts at its earliest, ${v.t1} s, ends at ${v.t1 + v.F.dur} s, and the satellite needs [[s = ${v.s}]] s to turn, so ${v.S.idx} can start at ${v.ready} s at the earliest; its latest start is ${Sat.latest(v.S)} s, so this order ${v.fits ? "<b>fits</b>" : "<b>is impossible</b>"}`;
  let text = `Pair (${p.a}, ${p.b}) on satellite ${A.sat}: windows [${A.ws}, ${A.we}] and [${B.ws}, ${B.we}] s, durations ${A.dur} and ${B.dur} s, pointing directions ${fmt(Sat.angleDeg(A.u, B.u), 1)}° apart. ${say(v1r)}. ${say(v2r)}. `;
  if (pw.kind === "both") text += "Both orders fit, so a model must choose an order (chapter 08 adds an order literal, chapter 09 a binary variable) and enforce the matching precedence only if both tasks are selected.";
  else if (pw.kind === "one") text += "Only one order fits, so if both tasks are selected that order is forced: one precedence, no choice to make.";
  else text += "Neither order fits: the two tasks are <b>mutually exclusive</b>, and the model simply forbids selecting both.";
  if (!samePairs) text += " (The page's pair list differs from Python's: please report.)";
  caption($("#pairCap"), text);
}
const pk = $("#pk"), pKind = $("#pKind");
function syncPair() { pk.max = CASES[pw.kind].length - 1; pk.value = pw.k; }
pKind.addEventListener("change", () => { pw.kind = pKind.value; pw.k = 0; syncPair(); drawPair(); });
pk.addEventListener("input", () => { pw.k = +pk.value; drawPair(); });

/* ======================= widget: pointing inside one window ======================= */
const ORB = D.orbits.map(([a, i, r, ph]) => Sat.orbit(a, i, r, ph));
const losAt = (o, t) => { const [la, lo] = D.targets[o.target]; return Sat.unit(Sat.sub(Sat.groundECI(la, lo, t), Sat.positionECI(ORB[o.sat], t))); };
const pt = { k: 0 };
function drawPoint() {
  const o = OPPS[pt.k], ts = [], ang = [], off = [];
  for (let t = o.ws; t <= o.we + 1e-9; t += 5) {
    ts.push(t); ang.push(Sat.angleDeg(losAt(o, t), o.u));
    const [la, lo] = D.targets[o.target]; off.push(Sat.lookAngles(Sat.positionECI(ORB[o.sat], t), Sat.groundECI(la, lo, t)).offNadir);
  }
  const cv = $("#ptC"), narrow = narrowOf(cv);
  const fr = window.Kit.frame(cv, narrow ? 1.5 : 2.6, { x: [o.ws, o.we], y: [0, 70], yticks: [0, 10, 20, 30, 40, 50, 60, 70], xlabel: "time [s]", ylabel: "angle [deg]" });
  window.Kit.plotLine(fr, ts, off, { color: alpha("--muted", 0.9), width: 1.4, dash: [4, 3] });
  window.Kit.plotLine(fr, ts, ang, { color: css("--post"), width: 2 });
  const mid = 0.5 * (o.ws + o.we);
  fr.ctx.strokeStyle = css("--ink"); fr.ctx.setLineDash([2, 3]); seg(fr.ctx, fr.X(mid), fr.Y(0), fr.X(mid), fr.Y(70)); fr.ctx.setLineDash([]);
  const a0 = ang[0], a1 = ang[ang.length - 1], turn = Sat.angleDeg(losAt(o, o.ws), losAt(o, o.we)), mx = Math.max(...ang);
  $("#ptV").textContent = `opp ${o.idx} (satellite ${o.sat})`;
  caption($("#ptCap"), `Opportunity ${o.idx}, satellite ${o.sat}, target ${o.target}, window [${o.ws}, ${o.we}] s. Blue: the angle between the direction to the target at time [[t]] and the midpoint direction [[u_o]] that the model uses for every transition, in the inertial frame; it is zero at the midpoint (dotted line) by definition. Grey dashed: the target's off-nadir angle (chapter 04). The direction is ${fmt(a0, 1)}° from [[u_o]] when the window opens and ${fmt(a1, 1)}° when it closes, and turns ${fmt(turn, 1)}° from start to end: at 2°/s that is up to ${fmt(mx / 2, 0)} s of slewing that the fixed transition times do not see. Two tasks in the same pass both turn with the pass, which is why the error on a transition can go either way.`);
}
const ptK = $("#ptK");
ptK.addEventListener("input", () => { pt.k = +ptK.value; drawPoint(); });

/* ======================= start ======================= */
hSat.value = st.sat; fillPasses(); fillTasks(); syncSlider();
pKind.value = pw.kind; syncPair();
ptK.max = OPPS.length - 1; ptK.value = pt.k;
function drawAll() { drawHero(); drawPair(); drawPoint(); }
drawAll(); onRedraw(drawAll);
})();
