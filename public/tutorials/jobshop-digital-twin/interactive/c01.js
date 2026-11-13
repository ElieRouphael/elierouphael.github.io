/* Which Job Should the Machine Do Next?, chapter C1, a constraint-programming model of the shop: the demos.
   Hero: the paint machine's CP model with the reader's values (arc literals and start times); every
   constraint of the model is checked live and the objective computed. Widgets: two paint machines with
   presence literals, self-loops and one circuit each; the linearised tardiness of one order; the big-M
   disjunction of two orders and its LP relaxation. Needs ../kit/kit.js and jobshop-lib.js. */
(() => {
"use strict";
const { $, css, caption, fit, frame, alpha, onRedraw, narrowOf } = window.Kit;
const { PAINT, FAM, runSequence, bestSequence } = window.JS;

const NAMES = ["A", "B", "C", "D"];
const NODES = ["0", ...NAMES];
const ORD = Object.fromEntries(PAINT.orders.map(o => [o.name, o]));
const H = 110;                                    // horizon of the hand-written paint model (Part 6); solve_snapshot itself uses 118
const colourOf = n => (n === "0" ? PAINT.start : ORD[n].family);
const s = (f, g) => (f == null || f === g ? 0 : PAINT.setup);
const fill = n => (n === "0" ? css("--ink") : FAM[ORD[n].family]);

/* ================================================================ hero ================================ */
const st = {
  succ: { "0": "A", A: "B", B: "D", D: "C", C: "0" },
  S: { A: 0, B: 20, D: 30, C: 55 },               // the optimal order, changeovers forgotten
  note: "",
};

/* the arc literals, as five selects */
(function buildArcs() {
  const box = $("#hArcs");
  NODES.forEach(u => {
    const lab = document.createElement("label");
    lab.textContent = u === "0" ? "after 0 (blue)" : `after ${u}`;
    lab.htmlFor = "arc" + u;
    const sel = document.createElement("select");
    sel.id = "arc" + u;
    NODES.forEach(v => {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = v === "0" ? (u === "0" ? "0: idle (self-loop)" : "0: end of the tour")
        : v === u ? `${v}: self-loop (${v} absent)` : `${v} directly follows`;
      sel.appendChild(o);
    });
    sel.addEventListener("change", () => { st.succ[u] = sel.value; st.note = ""; drawHero(); });
    box.append(lab, sel);
  });
})();
NAMES.forEach(n => $("#s" + n).addEventListener("input", () => { st.S[n] = +$("#s" + n).value; st.note = ""; drawHero(); }));

/* the circuit constraint: one tour through 0 visiting every present order (all four are present) */
function checkCircuit(succ) {
  const indeg = Object.fromEntries(NODES.map(v => [v, 0]));
  NODES.forEach(u => { indeg[succ[u]]++; });
  const problems = [], badArcs = new Set();
  NAMES.forEach(v => { if (succ[v] === v) { problems.push(`${v} takes its self-loop, but ${v} is on this machine (its presence literal is 1)`); badArcs.add(v + v); } });
  if (succ["0"] === "0") { problems.push("node 0 takes its idle self-loop, but orders are present"); badArcs.add("00"); }
  NODES.forEach(v => {
    if (indeg[v] > 1) { problems.push(`${indeg[v]} chosen arcs enter ${v}`); NODES.forEach(u => { if (succ[u] === v && u !== v) badArcs.add(u + v); }); }
    if (indeg[v] === 0) problems.push(`no chosen arc enters ${v}`);
  });
  /* cycles that avoid node 0: subtours */
  const seen = new Set();
  NAMES.forEach(a => {
    if (seen.has(a)) return;
    const path = [];
    let u = a;
    while (!path.includes(u) && u !== "0" && !seen.has(u)) { path.push(u); u = succ[u]; }
    path.forEach(x => seen.add(x));
    if (u !== "0" && path.includes(u)) {
      const cyc = path.slice(path.indexOf(u));
      if (cyc.length > 1 || succ[u] !== u) {
        problems.push(`subtour ${cyc.join(" → ")} → ${u} avoids node 0`);
        cyc.forEach(x => badArcs.add(x + succ[x]));
      }
    }
  });
  let tour = null;
  if (!problems.length) { tour = []; let u = st.succ["0"]; while (u !== "0" && tour.length < 6) { tour.push(u); u = succ[u]; } }
  return { ok: !problems.length, problems, badArcs, tour };
}

function pack(withSetup = true) {
  const c = checkCircuit(st.succ);
  if (!c.ok) { st.note = "<b>Earliest starts need a valid tour.</b> Fix the arc literals first: the circuit row says what is wrong. "; return; }
  let t = 0, fam = PAINT.start;
  c.tour.forEach(n => { const start = t + (withSetup ? s(fam, ORD[n].family) : 0); st.S[n] = start; t = start + ORD[n].p; fam = ORD[n].family; });
  st.note = "";
}
function setTour(seq) {
  const chain = ["0", ...seq, "0"];
  for (let i = 0; i < chain.length - 1; i++) st.succ[chain[i]] = chain[i + 1];
  pack(true);
}
$("#bPack").addEventListener("click", () => { pack(true); drawHero(); });
$("#bForget").addEventListener("click", () => { pack(false); drawHero(); });
$("#bFifo").addEventListener("click", () => { setTour(["A", "B", "C", "D"]); drawHero(); });
$("#bOpt").addEventListener("click", () => { setTour(["A", "B", "D", "C"]); drawHero(); });

function evaluate() {
  const S = st.S, E = {}, T = {};
  NAMES.forEach(n => { E[n] = S[n] + ORD[n].p; T[n] = Math.max(0, E[n] - ORD[n].due); });
  const circ = checkCircuit(st.succ);
  const rows = [], badOrders = new Set();
  rows.push({ grp: "The circuit over nodes 0, A, B, C, D" });
  rows.push({ c: "\\mathrm{Circuit}(\\text{chosen arcs})", v: circ.ok ? "0 → " + circ.tour.join(" → ") + " → 0" : circ.problems[0] + (circ.problems.length > 1 ? ` (+${circ.problems.length - 1} more)` : ""), ok: circ.ok, txt: true });
  rows.push({ grp: "Changeover on every arc set to 1" });
  NODES.forEach(u => {
    const v = st.succ[u];
    if (v === "0" || v === u) return;
    const sv = s(colourOf(u), ORD[v].family);
    if (u === "0") {
      const ok = S[v] >= sv;
      if (!ok) badOrders.add(v);
      rows.push({ c: `z_{0${v}} = 1 \\Rightarrow S_${v} \\ge 0 + ${sv}`, v: `S_${v} = ${S[v]}`, ok });
    } else {
      const ok = S[v] >= E[u] + sv;
      if (!ok) badOrders.add(v);
      rows.push({ c: `z_{${u}${v}} = 1 \\Rightarrow S_${v} \\ge E_${u} + ${sv}`, v: `${S[v]} \\ge ${E[u]} + ${sv}`, ok });
    }
  });
  const overlaps = [];
  for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
    const a = NAMES[i], b = NAMES[j];
    if (S[a] < E[b] && S[b] < E[a]) overlaps.push([a, b]);
  }
  rows.push({ grp: "One order at a time" });
  rows.push({ c: "\\mathrm{NoOverlap}(I_A, I_B, I_C, I_D)", v: overlaps.length ? overlaps.map(p => p.join(" and ")).join(", ") + " overlap" : "no two intervals share a minute", ok: !overlaps.length, txt: true });
  const late = NAMES.filter(n => E[n] > H);
  rows.push({ grp: "Domains" });
  rows.push({ c: `0 \\le S_j,\\ E_j \\le H = ${H}`, v: late.length ? late.map(n => `E_${n} = ${E[n]}`).join(", ") : `\\max_j E_j = ${Math.max(...NAMES.map(n => E[n]))}`, ok: !late.length });
  rows.push({ grp: "Tardiness, at its smallest feasible value" });
  rows.push({ c: "T_j \\ge E_j - d_j,\\ T_j \\ge 0", v: `T = (${NAMES.map(n => T[n]).join(", ")})`, ok: true });
  const sumT = NAMES.reduce((a, n) => a + T[n], 0), sumC = NAMES.reduce((a, n) => a + E[n], 0);
  const bad = rows.filter(r => !r.grp && !r.ok).length;
  overlaps.forEach(([a, b]) => { badOrders.add(a); badOrders.add(b); });
  return { S, E, T, circ, rows, sumT, sumC, bad, overlaps, badOrders, late };
}

function drawGanttHero(ev) {
  const cv = $("#hGantt"), w = cv.clientWidth || cv.parentElement.clientWidth;
  const laneH = 34, subH = 17, top = 6, axisH = 24, h = top + laneH + 8 + 4 * subH + axisH;
  const { ctx } = fit(cv, w / h);
  const labelW = 74, t1 = 120, X = t => labelW + t / t1 * (w - labelW - 12);
  const font = css("--font-mono");
  ctx.font = "11px " + font;
  /* grid and axis */
  const yAxis = top + laneH + 8 + 4 * subH;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let t = 0; t <= t1; t += 10) {
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(Math.round(X(t)) + 0.5, top); ctx.lineTo(Math.round(X(t)) + 0.5, yAxis); ctx.stroke();
    if (t % 20 === 0 || w > 640) { ctx.fillStyle = css("--muted"); ctx.fillText(String(t), X(t), yAxis + 5); }
  }
  /* horizon */
  ctx.strokeStyle = css("--muted"); ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(X(H) + 0.5, top); ctx.lineTo(X(H) + 0.5, yAxis); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = css("--muted"); ctx.textAlign = "right"; ctx.textBaseline = "top"; ctx.fillText("H", X(H) - 3, top);
  /* labels */
  ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillStyle = css("--ink");
  ctx.fillText("machine", labelW - 8, top + laneH / 2);
  NAMES.forEach((n, i) => { ctx.fillStyle = css("--ink-2"); ctx.fillText(`${n} due ${ORD[n].due}`, labelW - 8, top + laneH + 8 + i * subH + subH / 2); });
  /* the intervals */
  NAMES.forEach(n => {
    const x0 = X(ev.S[n]), x1 = X(ev.E[n]), y = top + 4, hh = laneH - 8;
    ctx.globalAlpha = 0.92; ctx.fillStyle = FAM[ORD[n].family]; ctx.fillRect(x0, y, x1 - x0, hh); ctx.globalAlpha = 1;
    if (ev.badOrders.has(n)) { ctx.strokeStyle = css("--warm"); ctx.lineWidth = 3; ctx.strokeRect(x0 + 1.5, y + 1.5, x1 - x0 - 3, hh - 3); }
    ctx.fillStyle = "#ffffff"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.font = "600 12px " + font;
    if (x1 - x0 > 12) ctx.fillText(n, (x0 + x1) / 2, top + laneH / 2);
    ctx.font = "11px " + font;
  });
  /* changeovers required by the chosen arcs, drawn just before the start (over the predecessor if it conflicts) */
  NODES.forEach(u => {
    const v = st.succ[u];
    if (v === "0" || v === u) return;
    const sv = s(colourOf(u), ORD[v].family);
    if (!sv) return;
    const x0 = X(Math.max(0, ev.S[v] - sv)), x1 = X(ev.S[v]), y = top + 8, hh = laneH - 16;
    ctx.save(); ctx.beginPath(); ctx.rect(x0, y, x1 - x0, hh); ctx.clip();
    ctx.fillStyle = css("--surface"); ctx.fillRect(x0, y, x1 - x0, hh);
    ctx.strokeStyle = FAM.setup; ctx.lineWidth = 2;
    for (let q = x0 - laneH; q < x1; q += 6) { ctx.beginPath(); ctx.moveTo(q, y + hh); ctx.lineTo(q + hh, y); ctx.stroke(); }
    ctx.restore();
  });
  /* overlaps */
  ev.overlaps.forEach(([a, b]) => {
    const lo = Math.max(ev.S[a], ev.S[b]), hi = Math.min(ev.E[a], ev.E[b]);
    ctx.fillStyle = alpha("--warm", 0.55); ctx.fillRect(X(lo), top, X(hi) - X(lo), 4);
    ctx.fillRect(X(lo), top + laneH - 4, X(hi) - X(lo), 4);
  });
  /* due dates and tardiness */
  NAMES.forEach((n, i) => {
    const y = top + laneH + 8 + i * subH + subH / 2, d = ORD[n].due, e = ev.E[n];
    ctx.strokeStyle = css("--rule"); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(0), y); ctx.lineTo(X(t1), y); ctx.stroke();
    if (e > d) {
      ctx.strokeStyle = css("--warm"); ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(X(d), y); ctx.lineTo(X(Math.min(e, t1)), y); ctx.stroke();
      ctx.fillStyle = css("--warm"); ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(`T = ${e - d}`, Math.min(X(Math.min(e, t1)) + 5, w - 46), y);
    } else {
      ctx.fillStyle = css("--c4"); ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText("on time", Math.min(X(d) + 7, w - 52), y);
    }
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(d), y - 6); ctx.lineTo(X(d), y + 6); ctx.stroke();
  });
}

function arrow(ctx, x0, y0, x1, y1, r, colour, width, label) {
  const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
  const off = 5, nx = -uy * off, ny = ux * off;      // offset so that a -> b and b -> a do not overlap
  const ax = x0 + ux * r + nx, ay = y0 + uy * r + ny, bx = x1 - ux * (r + 2) + nx, by = y1 - uy * (r + 2) + ny;
  ctx.strokeStyle = colour; ctx.fillStyle = colour; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  const hl = 9;
  ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx - ux * hl - uy * hl * 0.5, by - uy * hl + ux * hl * 0.5); ctx.lineTo(bx - ux * hl + uy * hl * 0.5, by - uy * hl - ux * hl * 0.5); ctx.closePath(); ctx.fill();
  if (label) {
    ctx.font = "600 11px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const mx = (ax + bx) / 2 + nx * 2.2, my = (ay + by) / 2 + ny * 2.2;
    ctx.fillStyle = css("--surface"); ctx.fillRect(mx - 13, my - 7, 26, 14);
    ctx.fillStyle = colour; ctx.fillText(label, mx, my);
  }
}

function drawGraph(ev) {
  const cv = $("#hGraph"), w = cv.clientWidth || cv.parentElement.clientWidth, h = Math.min(w, 250);
  const { ctx } = fit(cv, w / h);
  const cx = w / 2, cy = h / 2 + 4, R = Math.min(w, h) / 2 - 30, r = 15;
  const pos = {};
  NODES.forEach((n, i) => { const a = -Math.PI / 2 + i * 2 * Math.PI / 5; pos[n] = [cx + R * Math.cos(a), cy + R * Math.sin(a)]; });
  NODES.forEach(u => {
    const v = st.succ[u], bad = ev.circ.badArcs.has(u + v), col = bad ? css("--warm") : ev.circ.ok ? css("--accent") : css("--muted");
    if (u === v) {
      const [x, y] = pos[u], ang = Math.atan2(y - cy, x - cx), lx = x + Math.cos(ang) * (r + 9), ly = y + Math.sin(ang) * (r + 9);
      ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.arc(lx, ly, 9, 0, 2 * Math.PI); ctx.stroke();
      return;
    }
    const sv = v === "0" ? 0 : s(colourOf(u), ORD[v].family);
    arrow(ctx, pos[u][0], pos[u][1], pos[v][0], pos[v][1], r, col, 2.2, v === "0" ? "" : "+" + sv);
  });
  NODES.forEach(n => {
    const [x, y] = pos[n];
    ctx.fillStyle = fill(n); ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = n === "0" ? css("--bg") : "#ffffff"; ctx.font = "700 12px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(n, x, y + 0.5);
  });
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillText("arcs set to 1 (+ changeover)", 2, 0);
}

function drawHero() {
  NODES.forEach(u => { $("#arc" + u).value = st.succ[u]; });
  NAMES.forEach(n => { $("#s" + n).value = st.S[n]; $("#v" + n).textContent = st.S[n]; });
  const ev = evaluate();
  /* the constraint table */
  const html = ev.rows.map(r => r.grp
    ? `<tr><td class="grp" colspan="3">${r.grp}</td></tr>`
    : `<tr class="${r.ok ? "" : "bad"}"><td>[[${r.c}]]</td><td>${r.txt ? r.v : "[[" + r.v + "]]"}</td><td class="${r.ok ? "ok" : "bad"}">${r.ok ? "✓" : "✗"}</td></tr>`).join("");
  caption($("#hCons"), "<tbody>" + html + "</tbody>");
  drawGanttHero(ev);
  drawGraph(ev);
  $("#rT").textContent = ev.sumT + " min";
  $("#rC").textContent = ev.sumC + " min";
  $("#rObj").textContent = 10 * ev.sumT + ev.sumC;
  ["#rT", "#rC", "#rObj"].forEach(id => $(id).classList.toggle("void", ev.bad > 0));
  $("#rViol").textContent = ev.bad ? `${ev.bad}: not a solution` : "0";
  $("#rViol").classList.toggle("warn", ev.bad > 0);
  const tourTxt = ev.circ.ok ? ev.circ.tour.join(", ") : null;
  let msg;
  if (!ev.bad) {
    msg = `<b>Every constraint holds:</b> a feasible schedule, order ${tourTxt}, with total tardiness [[\\sum T_j = ${ev.sumT}]] and objective [[10 \\sum T_j + \\sum C_j = ${10 * ev.sumT + ev.sumC}]]. ` +
      (ev.sumT === 40 ? "This is the optimum CP-SAT proves: no feasible assignment has a smaller total tardiness. "
        : `CP-SAT proves that the best feasible value is 40; this one is ${ev.sumT - 40} minutes worse. `) +
      "A solver searches over exactly these values, the arc literals and the start times, and keeps the best feasible one.";
  } else if (!ev.circ.ok) {
    msg = `<b>${ev.bad} constraint${ev.bad > 1 ? "s" : ""} violated.</b> The arc literals do not form one tour from 0 through A, B, C and D back to 0: ${ev.circ.problems.join("; ")}. ` +
      "The totals are struck through, since this is not a solution. Every node needs exactly one chosen arc in and one out, present orders may not take their self-loop, and a cycle avoiding node 0 is a subtour. A subtour also breaks the timing rows: along a cycle each start would have to follow its own end.";
  } else {
    const arcsBad = ev.rows.filter(r => !r.grp && !r.ok && r.c.startsWith("z_")).length;
    msg = `<b>${ev.bad} constraint${ev.bad > 1 ? "s" : ""} violated.</b> The arcs form the tour 0, ${tourTxt}, 0, but ` +
      (arcsBad ? `${arcsBad} order${arcsBad > 1 ? "s start before their" : " starts before its"} direct predecessor ends plus the changeover the arc requires (outlined; hatched blocks are the changeovers). ` : "") +
      (ev.overlaps.length ? `Intervals overlap (${ev.overlaps.map(p => p.join(" and ")).join(", ")}), which the NoOverlap row also reports. ` : "") +
      (ev.late.length ? `An interval ends after the horizon [[H = ${H}]]. ` : "") +
      "The totals are struck through: values that break a constraint are not a solution, so their objective means nothing. Press <b>Earliest starts for these arcs</b> to set each start to its smallest feasible value.";
  }
  caption($("#heroStatus"), (st.note || "") + msg);
}

/* ================================================================ two machines ======================== */
const two = { assign: [0, 0, 0, 0] };
(function buildAssign() {
  const box = $("#twoAssign");
  NAMES.forEach((n, i) => {
    const wrap = document.createElement("div");
    wrap.innerHTML = `<span class="o">${n} (${ORD[n].family})</span><span class="seg"><button class="btn" type="button" data-m="0">M0</button><button class="btn" type="button" data-m="1">M1</button></span>`;
    wrap.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { two.assign[i] = +b.dataset.m; drawTwo(); }));
    box.appendChild(wrap);
  });
})();
function machineBest(k, assign) {
  const orders = PAINT.orders.filter((o, i) => assign[i] === k);
  return bestSequence(orders);
}
function bestAssignment() {
  let best = null;
  for (let code = 0; code < 16; code++) {
    const a = [3, 2, 1, 0].map(b => (code >> b) & 1);      // A is the most significant bit, as "0101" in the numbers script
    const v = machineBest(0, a).value + machineBest(1, a).value;
    if (!best || v < best.v) best = { a, v };
  }
  return best;
}
$("#twoAll").addEventListener("click", () => { two.assign = [0, 0, 0, 0]; drawTwo(); });
$("#twoBest").addEventListener("click", () => { two.assign = bestAssignment().a.slice(); drawTwo(); });

function drawMatrix(cv, k, seq) {
  const w = cv.clientWidth || cv.parentElement.clientWidth, cell = Math.min(34, Math.floor((w - 40) / 5.6)), h = 36 + cell * 5 + 26;
  const { ctx } = fit(cv, w / h);
  const x0 = Math.max(26, (w - cell * 5) / 2), y0 = 36, font = css("--font-mono");
  const succ = {}; const chain = ["0", ...seq, "0"];
  for (let i = 0; i < chain.length - 1; i++) succ[chain[i]] = chain[i + 1];
  ctx.font = "11px " + font; ctx.fillStyle = css("--muted"); ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillText(`machine ${k}: arc literals z(from → to)`, 2, 0);
  NODES.forEach((u, i) => {
    ctx.fillStyle = css("--ink-2"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(u, x0 - 6, y0 + i * cell + cell / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(u, x0 + i * cell + cell / 2, y0 - 2);
    NODES.forEach((v, j) => {
      const x = x0 + j * cell, y = y0 + i * cell;
      let on = false, self = false;
      if (u === v) {
        self = u === "0" ? seq.length === 0 : !seq.includes(u);
        on = self;
      } else on = succ[u] === v;
      ctx.fillStyle = on ? (self ? css("--c5") : css("--accent")) : alpha("--ink", 0.05);
      ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
      if (on) { ctx.fillStyle = "#ffffff"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.font = "600 11px " + font; ctx.fillText("1", x + cell / 2, y + cell / 2); ctx.font = "11px " + font; }
    });
  });
  ctx.fillStyle = css("--muted"); ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillText("blue: tour arcs · violet: self-loops", 2, y0 + cell * 5 + 8);
}

function drawTwo() {
  $$btn();
  const b0 = machineBest(0, two.assign), b1 = machineBest(1, two.assign), seqs = [b0.seq, b1.seq];
  const blocks = [];
  seqs.forEach((seq, k) => {
    const rows = runSequence(PAINT.orders, seq);
    rows.forEach(r => {
      if (r.start > r.free) blocks.push({ lane: k, start: r.free, end: r.start, kind: "setup" });
      blocks.push({ lane: k, start: r.start, end: r.end, colour: FAM[ORD[r.name].family], label: r.name, kind: "work" });
    });
  });
  window.JS.drawGantt($("#twoGantt"), blocks, { lanes: 2, t0: 0, t1: 100, laneLabels: ["machine 0", "machine 1"], laneHeight: 32 });
  drawMatrix($("#twoM0"), 0, b0.seq);
  drawMatrix($("#twoM1"), 1, b1.seq);
  const x = NAMES.map((n, i) => `[[x_{${n}0} = ${1 - two.assign[i]},\\ x_{${n}1} = ${two.assign[i]}]]`).join("; ");
  const tour = seq => seq.length ? "0, " + seq.join(", ") + ", 0" : "empty: [[z_{00} = 1]] (idle)";
  const total = b0.value + b1.value, best = bestAssignment().v;
  caption($("#twoCap"), `Presence literals: ${x}; each pair sums to 1, the exactly-one constraint. ` +
    `Machine 0 tour: ${tour(b0.seq)}, tardiness ${b0.value}; machine 1 tour: ${tour(b1.seq)}, tardiness ${b1.value}. ` +
    `Every order not on a machine lights its diagonal cell there, the self-loop [[z_{vv} = \\neg x_{v m}]], so each circuit visits exactly the orders routed to it. ` +
    `Each machine has [[(4 + 1)^2 = 25]] arc literals, one per cell, whatever the routing. Total tardiness ${total} min` +
    (total === best ? `, the best possible with two machines (${best}).` : `; the best routing reaches ${best}.`));
}
function $$btn() {
  document.querySelectorAll("#twoAssign > div").forEach((d, i) => d.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.m === two.assign[i]))));
}

/* ================================================================ tardiness =========================== */
function drawTard() {
  const C = +$("#tC").value, T = +$("#tT").value, d = ORD.A.due, low = Math.max(0, C - d), ok = T >= low;
  $("#tCV").textContent = C; $("#tTV").textContent = T;
  const cv = $("#tardC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.25 : 2.2, { x: [0, 110], y: [0, 80], xlabel: "completion of A (min)", ylabel: "tardiness variable of A (min)", xticks: [0, 20, 30, 40, 60, 80, 100], yticks: [0, 20, 40, 60, 80] });
  const { ctx, X, Y, pad, iw, ih } = fr;
  ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t, iw, ih); ctx.clip();
  ctx.fillStyle = alpha("--accent", 0.13);
  ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(d), Y(0)); ctx.lineTo(X(110), Y(80)); ctx.lineTo(X(0), Y(80)); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(d), Y(0)); ctx.lineTo(X(110), Y(80)); ctx.stroke();
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(X(C), Y(0)); ctx.lineTo(X(C), Y(80)); ctx.stroke(); ctx.setLineDash([]);
  ctx.restore();
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  const lbl = "feasible: T ≥ C − 30 and T ≥ 0", lw = ctx.measureText(lbl).width;
  ctx.fillStyle = css("--surface"); ctx.fillRect(X(4) - 3, Y(70) - 15, lw + 6, 17);   // keeps the dashed line off the label
  ctx.fillStyle = css("--accent"); ctx.fillText(lbl, X(4), Y(70));
  /* the smallest feasible T on the vertical line, and the reader's point */
  ctx.strokeStyle = css("--c4"); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(X(C), Y(low), 6, 0, 2 * Math.PI); ctx.stroke();
  ctx.fillStyle = ok ? css("--ink") : css("--warm"); ctx.beginPath(); ctx.arc(X(C), Y(T), 5, 0, 2 * Math.PI); ctx.fill();
  caption($("#tardCap"), `Order A is due at [[d_A = ${d}]]. With [[C_A = ${C}]] the two bounds read [[T_A \\ge ${C - d}]] and [[T_A \\ge 0]], so the smallest feasible value (green ring) is [[\\max(0, ${C - d}) = ${low}]], the true tardiness. ` +
    (ok ? `Your [[T_A = ${T}]] is feasible and costs [[10 T_A = ${10 * T}]] in the objective; ` + (T > low ? `lowering it to ${low} saves ${10 * (T - low)} and breaks nothing, so no optimal solution keeps it: minimisation makes the bound exact.` : "it sits on the bound, as at every optimum.")
      : `Your [[T_A = ${T}]] is below the bound: infeasible, the model would reject it.`));
}
["#tC", "#tT"].forEach(id => $(id).addEventListener("input", drawTard));
$("#tMin").addEventListener("click", () => { const C = +$("#tC").value; $("#tT").value = Math.max(0, C - ORD.A.due); drawTard(); });

/* ================================================================ big-M =============================== */
function drawBigM() {
  const M = +$("#bM").value, y = +$("#bY").value, lo = 30 - M * (1 - y), hi = -20 + M * y;
  $("#bMV").textContent = M; $("#bYV").textContent = y.toFixed(2);
  const cv = $("#bigmC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.0 : 1.6, { x: [-6, 110], y: [-6, 110], xlabel: "start of A (min)", ylabel: "start of B (min)", xticks: [0, 20, 40, 60, 80, 100], yticks: [0, 20, 40, 60, 80, 100] });
  const { ctx, X, Y, pad, iw, ih } = fr;
  const band = (a, b, colour) => {      // the set a <= S_B - S_A <= b
    ctx.fillStyle = colour; ctx.beginPath();
    ctx.moveTo(X(-300), Y(-300 + a)); ctx.lineTo(X(400), Y(400 + a)); ctx.lineTo(X(400), Y(400 + b)); ctx.lineTo(X(-300), Y(-300 + b)); ctx.closePath(); ctx.fill();
  };
  const line = (c, colour, dash, width) => { ctx.strokeStyle = colour; ctx.setLineDash(dash); ctx.lineWidth = width; ctx.beginPath(); ctx.moveTo(X(-300), Y(-300 + c)); ctx.lineTo(X(400), Y(400 + c)); ctx.stroke(); ctx.setLineDash([]); };
  ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t, iw, ih); ctx.clip();
  band(30, 1000, alpha("--accent", 0.22));        // A before B
  band(-1000, -20, alpha("--accent", 0.22));      // B before A
  if (lo <= hi) band(lo, hi, alpha("--warm", 0.25));
  line(30, css("--accent"), [], 2); line(-20, css("--accent"), [], 2);
  line(30 - M, css("--warm"), [5, 4], 1.4); line(M - 20, css("--warm"), [5, 4], 1.4);
  ctx.restore();
  /* the LP relaxation's starts for A and B */
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(0), Y(10), 5, 0, 2 * Math.PI); ctx.fill();
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillStyle = css("--ink");
  ctx.fillText("LP relaxation: A at 0, B at 10", X(0) + 9, Y(10) - 10);
  ctx.fillStyle = css("--accent"); ctx.textAlign = "left"; ctx.fillText("A before B", X(4), Y(96));
  ctx.textAlign = "right"; ctx.fillText("B before A", X(106), Y(8));
  const inBand = 10 >= lo && 10 <= hi;
  caption($("#bigmCap"), `Blue: the real choices, [[S_B - S_A \\ge 30]] (A, then 10 min of changeover, then B) or [[S_B - S_A \\le -20]]. Shaded band: what the relaxation allows for [[y_{AB} = ${y.toFixed(2)}]], ` +
    (lo <= hi ? `[[${fmt1(lo)} \\le S_B - S_A \\le ${fmt1(hi)}]]` : "nothing (the two lines cross)") +
    `; dashed: the union over every [[y]] between 0 and 1, [[${30 - M} \\le S_B - S_A \\le ${M - 20}]]. ` +
    (y === 0 || y === 1 ? `At an integer [[y]] the relaxation is exactly one of the two blue sets. `
      : lo <= 0 && 0 <= hi ? `This [[y]] lets the band cover the diagonal [[S_B = S_A]], where A and B start together, as does every [[y]] from [[${(20 / M).toFixed(2)}]] to [[${(1 - 30 / M).toFixed(2)}]]. `
      : `This [[y]] misses the diagonal [[S_B = S_A]], but every [[y]] from [[${(20 / M).toFixed(2)}]] to [[${(1 - 30 / M).toFixed(2)}]] covers it, letting A and B start together. `) +
    `The LP relaxation's schedule (the dot) is ${inBand ? "inside" : "outside"} the shaded band and in neither blue set: it paints A and B at the same time. A larger [[M]] only widens the band.`);
}
const fmt1 = v => (Math.round(v * 10) / 10).toString();
["#bM", "#bY"].forEach(id => $(id).addEventListener("input", drawBigM));

function drawAll() { drawHero(); drawTwo(); drawTard(); drawBigM(); }
drawAll();
onRedraw(drawAll);
})();
