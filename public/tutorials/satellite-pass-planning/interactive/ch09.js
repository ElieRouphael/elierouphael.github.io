/* Satellite Pass Planning from Scratch, chapter 09, The Same Problem as a MILP: the live demos.
   (1) the hero: one real two-way pair of the small instance in the (t_a, t_b) plane, with the two big-M
       rows of satplan/milp.py moved by x_a, x_b and y, the relaxed region against the integer one;
   (2) a two-variable LP (a knapsack) to show what an LP relaxation and its bound are;
   (3) the ladder of bounds (LP relaxations computed by code/ch09_numbers.py, final solver bounds);
   (4) the benchmark of results/benchmark.json, per instance and method.
   Needs ../kit/kit.js, sat.js, instance-small.js and ch09-data.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, int, ticks, caption, frame, plotLine, alpha, onRedraw, narrowOf, fit } = window.Kit;
const Sat = window.Sat, I = window.SAT_SMALL, D = window.SAT_CH09;

/* ---------------- geometry helpers ---------------- */
/* Keep the part of a convex polygon where ax * u + ay * v >= c (one step of Sutherland-Hodgman). */
function clip(poly, ax, ay, c) {
  const out = [];
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    const fp = ax * p[0] + ay * p[1] - c, fq = ax * q[0] + ay * q[1] - c;
    if (fp >= 0) out.push(p);
    if ((fp >= 0) !== (fq >= 0)) { const t = fp / (fp - fq); out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
  }
  return out;
}
function area(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) { const p = poly[i], q = poly[(i + 1) % poly.length]; s += p[0] * q[1] - q[0] * p[1]; }
  return Math.abs(s) / 2;
}
function fillPoly(fr, poly, color, stroke) {
  if (poly.length < 3) return;
  const { ctx, X, Y } = fr;
  ctx.beginPath();
  poly.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])));
  ctx.closePath();
  if (color) { ctx.fillStyle = color; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.2; ctx.setLineDash([]); ctx.stroke(); }
}
const dot = (ctx, x, y, r, color) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); };

/* ---------------- (1) the pair in the (t_a, t_b) plane ---------------- */
/* The pair is recomputed from the shared instance with sat.js (windows, latest starts, transition times),
   and falls back to the Python export if the shared files are missing. */
const P = (() => {
  const p = Object.assign({}, D.pair);
  p.live = false;
  if (I && Sat) {
    const oa = I.opps[p.a], ob = I.opps[p.b];
    p.ea = oa.ws; p.la = Sat.latest(oa); p.eb = ob.ws; p.lb = Sat.latest(ob);
    p.da = oa.dur; p.db = ob.dur; p.wa = oa.w; p.wb = ob.w;
    p.sab = Sat.transition(I, p.a, p.b); p.sba = Sat.transition(I, p.b, p.a);
    p.gab = p.da + p.sab; p.gba = p.db + p.sba;
    p.M1 = p.gab - p.eb + p.la; p.M2 = p.gba - p.ea + p.lb;
    p.live = p.M1 === D.pair.M1 && p.M2 === D.pair.M2;
  }
  return p;
})();
const st = { xa: 0.5, xb: 0.5, y: 0.5, naive: false };
const H = I ? I.horizon : 21600;
/* Right-hand sides of milp.py's two rows for the pair:
   y = 1 (a first):  t_b - t_a >= g_ab - M1 (3 - x_a - x_b - y)
   y = 0 (b first):  t_a - t_b >= g_ba - M2 (2 - x_a - x_b + y) */
function rows() {
  const M1 = st.naive ? H : P.M1, M2 = st.naive ? H : P.M2;
  return { M1, M2, r1: P.gab - M1 * (3 - st.xa - st.xb - st.y), r2: P.gba - M2 * (2 - st.xa - st.xb + st.y) };
}
function drawHero() {
  const cv = $("#heroC"), narrow = narrowOf(cv, 560);
  const padT = 40;
  const fr = frame(cv, narrow ? 1.0 : 1.45, { x: [P.ea - padT, P.la + padT], y: [P.eb - padT, P.lb + padT],
    xlabel: `start of a, t_a [s]`, ylabel: `start of b, t_b [s]` });
  const { ctx, X, Y } = fr;
  const box = [[P.ea, P.eb], [P.la, P.eb], [P.la, P.lb], [P.ea, P.lb]];
  const boxArea = area(box);
  const R = rows();
  // the relaxed region: box and both rows
  const relaxed = clip(clip(box, -1, 1, R.r1), 1, -1, R.r2);
  // the forbidden band, where neither order is respected: -g_ba < t_b - t_a < g_ab
  const band = clip(clip(box, 1, -1, -P.gab), -1, 1, -P.gba);
  const bandAllowed = clip(clip(relaxed, 1, -1, -P.gab), -1, 1, -P.gba);
  // the integer picture when both are selected: a first (upper strip) or b first (lower strip)
  const aFirst = clip(box, -1, 1, P.gab), bFirst = clip(box, 1, -1, P.gba);
  fillPoly(fr, band, alpha("--warm", 0.16));
  fillPoly(fr, relaxed, alpha("--post", 0.22));
  fillPoly(fr, bandAllowed, alpha("--warm", 0.42));
  fillPoly(fr, box, null, css("--muted"));
  // integer strips: outlined in ink, dashed
  ctx.save(); ctx.setLineDash([5, 4]);
  for (const s of [aFirst, bFirst]) {
    if (s.length < 3) continue;
    ctx.beginPath(); s.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))); ctx.closePath();
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.4; ctx.stroke();
  }
  ctx.restore();
  // the two row boundaries: t_b = t_a + r1 and t_b = t_a - r2
  const xs = [P.ea - padT, P.la + padT];
  plotLine(fr, xs, xs.map(t => t + R.r1), { color: css("--post"), width: 2 });
  plotLine(fr, xs, xs.map(t => t - R.r2), { color: css("--c5"), width: 2 });
  // the corners that the smallest big-M lines pass through
  dot(ctx, X(P.la), Y(P.eb), 3.5, css("--ink"));
  dot(ctx, X(P.ea), Y(P.lb), 3.5, css("--ink"));
  ctx.font = "11px " + css("--font-mono"); ctx.fillStyle = css("--ink-2");
  ctx.textAlign = "right"; ctx.fillText(`(${P.la}, ${P.eb})`, X(P.la) - 6, Y(P.eb) - 6);
  ctx.textAlign = "left"; ctx.fillText(`(${P.ea}, ${P.lb})`, X(P.ea) + 6, Y(P.lb) + 14);

  const lo = P.eb - P.la, hi = P.lb - P.ea;   // range of t_b - t_a over the box
  const off1 = R.r1 <= lo + 1e-9, off2 = R.r2 <= -hi + 1e-9;   // a row is "off" when every point of the box satisfies it
  const share = band.length > 2 ? area(bandAllowed) / area(band) : 0;
  $("#rR1").textContent = `${fmt(R.r1, 1)} s` + (off1 ? " (off)" : "");
  $("#rR2").textContent = `${fmt(R.r2, 1)} s` + (off2 ? " (off)" : "");
  $("#rM").textContent = `${fmt(R.M1, 0)}, ${fmt(R.M2, 0)}`;
  $("#rArea").textContent = pct(area(relaxed) / boxArea, 0);
  $("#rBand").textContent = pct(share, 0);
  const isInt = [st.xa, st.xb, st.y].every(v => v === 0 || v === 1);
  let story;
  if (isInt && st.xa === 1 && st.xb === 1) story = `Both tasks are selected and the order literal says <b>${st.y ? "a first" : "b first"}</b>: the ${st.y ? "first" : "second"} row is the true precedence [[${st.y ? "t_b \\ge t_a + " + P.gab : "t_a \\ge t_b + " + P.gba}]] and the other row is switched off, its line ${st.naive ? "far outside the box" : "passing exactly through the corner of the box"}. The shaded region is one of the two dashed strips: the integer model is exact.`;
  else if (isInt) story = `At least one task is not selected, so both rows must be switched off, and they are: each line passes through or beyond a corner of the box, and every pair of starts is allowed, which is right, since an unselected task has no start to respect.`;
  else if (off1 && off2) story = `With these fractional values both rows are switched off: the whole box is allowed, including the orange band where the two tasks would overlap or leave less than the reconfiguration time between them. The LP relaxation sees no precedence at all for this pair.`;
  else if (share > 0.999) story = `The rows are active but too weak: they cut away ${off1 || off2 ? "a corner" : "two corners"} of the box, start pairs far apart that never needed excluding, and leave the whole orange band allowed. For this pair the relaxation still sees no precedence.`;
  else if (share > 0) story = `The rows are only partly active: the relaxed region (blue) still covers ${pct(share, 0)} of the orange band, the start pairs that no real plan can use. A fractional value has bought the LP a schedule that does not exist.`;
  else story = `The rows now exclude the orange band entirely, although the values are fractional: the relaxation is exact for this pair at this point.`;
  caption($("#heroStatus"), `Opportunities ${P.a} and ${P.b} of the small instance, both on satellite ${P.sat}: task a (priority ${P.wa}, ${P.da} s) may start in [${P.ea}, ${P.la}] s, task b (priority ${P.wb}, ${P.db} s) in [${P.eb}, ${P.lb}] s. ${P.live ? "Recomputed here with sat.js: " : ""}the reconfiguration times are ${P.sab} s (a to b) and ${P.sba} s (b to a), so the starts must be [[g_{ab} = ${P.gab}]] s apart if a goes first and [[g_{ba} = ${P.gba}]] s if b does. The grey box holds every pair of starts; the dashed strips are the starts a real plan can use when both are selected (a first above, b first below); the orange band between them is forbidden. With [[x_a = ${fmt(st.xa, 2)}]], [[x_b = ${fmt(st.xb, 2)}]], [[y = ${fmt(st.y, 2)}]]${st.naive ? " and every big-M set to the horizon, " + H + " s" : " and the smallest big-M of milp.py"}, the two rows allow the blue region. ${story}`);
}
function setHero(xa, xb, y) {
  st.xa = xa; st.xb = xb; st.y = y;
  $("#hXa").value = xa; $("#hXb").value = xb; $("#hY").value = y;
  syncHero(); drawHero();
}
function syncHero() {
  $("#hXaV").textContent = fmt(st.xa, 2); $("#hXbV").textContent = fmt(st.xb, 2); $("#hYV").textContent = fmt(st.y, 2);
  $("#hNaive").setAttribute("aria-pressed", st.naive);
}
$("#hXa").addEventListener("input", e => { st.xa = +e.target.value; syncHero(); drawHero(); });
$("#hXb").addEventListener("input", e => { st.xb = +e.target.value; syncHero(); drawHero(); });
$("#hY").addEventListener("input", e => { st.y = +e.target.value; syncHero(); drawHero(); });
$("#hNaive").addEventListener("click", () => { st.naive = !st.naive; syncHero(); drawHero(); });
$("#pA").addEventListener("click", () => setHero(1, 1, 1));
$("#pB").addEventListener("click", () => setHero(1, 1, 0));
$("#pHalf").addEventListener("click", () => setHero(0.5, 0.5, 0.5));
$("#pY").addEventListener("click", () => setHero(1, 1, 0.5));
$("#pOff").addEventListener("click", () => setHero(0, 1, 1));

/* ---------------- (2) a two-variable LP: the knapsack of one revolution ---------------- */
const kp = { B: 180, d1: 120, w1: 8, d2: 90, w2: 4 };
function drawKnap() {
  const cv = $("#knapC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.05 : 1.7, { x: [-0.08, 1.12], y: [-0.08, 1.12], xticks: [0, 0.5, 1], yticks: [0, 0.5, 1],
    xlabel: "x₁ (task 1 selected)", ylabel: "x₂ (task 2 selected)" });
  const { ctx, X, Y } = fr;
  const sq = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const feas = clip(sq, -kp.d1, -kp.d2, -kp.B);   // d1 x1 + d2 x2 <= B
  fillPoly(fr, sq, null, css("--muted"));
  fillPoly(fr, feas, alpha("--post", 0.22), css("--post"));
  // LP optimum: best vertex of the feasible polygon
  let best = null;
  for (const p of feas) { const v = kp.w1 * p[0] + kp.w2 * p[1]; if (!best || v > best.v + 1e-12) best = { p, v }; }
  // integer points
  let ibest = null;
  for (const p of sq) {
    const ok = kp.d1 * p[0] + kp.d2 * p[1] <= kp.B + 1e-9, v = kp.w1 * p[0] + kp.w2 * p[1];
    if (ok && (!ibest || v > ibest.v)) ibest = { p, v };
    ctx.lineWidth = 1.6; ctx.strokeStyle = css("--ink"); ctx.fillStyle = ok ? css("--ink") : css("--surface");
    ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }
  // objective level line through the LP optimum: w1 x1 + w2 x2 = v
  if (best) {
    const xs = [-0.08, 1.12];
    plotLine(fr, xs, xs.map(x => (best.v - kp.w1 * x) / kp.w2), { color: css("--warm"), width: 1.5, dash: [5, 4] });
    dot(ctx, X(best.p[0]), Y(best.p[1]), 6, css("--warm"));
  }
  if (ibest) { ctx.strokeStyle = css("--c4"); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(X(ibest.p[0]), Y(ibest.p[1]), 9, 0, 2 * Math.PI); ctx.stroke(); }
  $("#kBV").textContent = `${kp.B} s`;
  const frac = best && (Math.abs(best.p[0] - Math.round(best.p[0])) > 1e-9 || Math.abs(best.p[1] - Math.round(best.p[1])) > 1e-9);
  caption($("#knapCap"), `Two listening tasks in one revolution: task 1 lasts ${kp.d1} s with priority ${kp.w1}, task 2 lasts ${kp.d2} s with priority ${kp.w2}, and the power budget allows ${kp.B} s, so [[${kp.d1} x_1 + ${kp.d2} x_2 \\le ${kp.B}]]. The blue polygon is the LP's feasible set, with [[0 \\le x \\le 1]]; the four corners are the only integer choices (filled when they fit the budget). The LP optimum (orange) is at [[x = (${fmt(best.p[0], 2)}, ${fmt(best.p[1], 2)})]] with value ${fmt(best.v, 2)}, on the dashed line of equal priority; the integer optimum (green ring) is ${fmt(ibest.v, 0)}. ${frac ? `The LP fills the budget with ${fmt(best.p[0] < 1 - 1e-9 ? best.p[0] : best.p[1], 2)} of a task, which no satellite can do, so its value is an <b>upper bound</b>, ${fmt(best.v - ibest.v, 2)} above what an integer plan reaches; since priorities are integers the bound can be rounded down to ${Math.floor(best.v + 1e-9)}.` : `Here the LP optimum is already integer: the relaxation is exact and its bound is the optimum itself.`}`);
}
$("#kB").addEventListener("input", e => { kp.B = +e.target.value; drawKnap(); });

/* ---------------- (3) the ladder of bounds ---------------- */
let ladderSize = "small";
function benchRec(size, seed = 0) { return D.bench.find(r => r.size === size && r.seed === seed); }
function ladderRows(size) {
  const d = D[size], b = benchRec(size), lp = d.lp;
  const best = Math.max(b.greedy, b.cpsat, b.rolling, b.milp);
  const rowsL = [
    { k: "every reachable target collected", v: d.stats.reachable_weight, c: "--muted" },
    { k: "LP: one per target and budgets only", v: lp.bare.bound, c: "--prior" },
    { k: "LP: + exclusive pairs, no precedence rows", v: lp.noprec.bound, c: "--prior" },
    { k: "LP of milp.py (smallest big-M)", v: lp.tight.bound, c: "--post" },
    { k: "LP of milp.py with M = horizon", v: lp.naive.bound, c: "--post" },
  ];
  if (lp.cliques) rowsL.push({ k: `LP of milp.py + ${lp.cliques.count} clique rows`, v: lp.cliques.bound, c: "--c5" });
  rowsL.push({ k: `HiGHS bound after ${b.limit} s`, v: b.milp_bound, c: "--warm" });
  rowsL.push({ k: `CP-SAT bound after ${b.limit} s`, v: b.cpsat_bound, c: "--c4" });
  rowsL.push({ k: size === "small" ? "optimum (both solvers)" : "best plan found by any method", v: best, c: "--ink" });
  return { rowsL, best, b };
}
function drawLadder() {
  const cv = $("#ladC"), narrow = narrowOf(cv);
  const { rowsL, best, b } = ladderRows(ladderSize);
  const rowH = narrow ? 40 : 27;
  const w = cv.clientWidth || cv.parentElement.clientWidth;
  const h = rowsL.length * rowH + 40;
  const { ctx } = fit(cv, w / h);
  const vals = rowsL.map(r => r.v);
  const lo = Math.floor(best * 0.95 / 10) * 10, hi = Math.ceil(Math.max(...vals) * 1.02 / 10) * 10;
  const left = narrow ? 10 : Math.min(300, w * 0.46), right = w - 56;
  const X = v => left + (v - lo) / (hi - lo) * (right - left);
  ctx.font = "12px " + css("--font-ui");
  // grid
  ctx.strokeStyle = css("--grid"); ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.font = "11px " + css("--font-mono");
  for (const t of ticks(lo, hi, narrow ? 4 : 6)) { ctx.beginPath(); ctx.moveTo(X(t) + 0.5, 6); ctx.lineTo(X(t) + 0.5, h - 26); ctx.stroke(); ctx.fillText(String(t), X(t), h - 22); }
  // best-value line
  ctx.strokeStyle = css("--ink"); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(best) + 0.5, 6); ctx.lineTo(X(best) + 0.5, h - 26); ctx.stroke(); ctx.setLineDash([]);
  rowsL.forEach((r, i) => {
    const yc = 8 + i * rowH + (narrow ? 26 : rowH / 2);
    ctx.textBaseline = "middle"; ctx.font = "12px " + css("--font-ui"); ctx.fillStyle = css("--ink-2");
    if (narrow) { ctx.textAlign = "left"; ctx.fillText(r.k, 10, yc - 14); }
    else { ctx.textAlign = "right"; ctx.fillText(r.k, left - 10, yc); }
    ctx.strokeStyle = alpha(r.c, 0.6); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(Math.max(lo, best)), yc); ctx.lineTo(X(Math.min(hi, r.v)), yc); ctx.stroke(); ctx.lineWidth = 1;
    const xv = X(Math.min(hi, Math.max(lo, r.v)));
    dot(ctx, xv, yc, 5, css(r.c));
    ctx.textAlign = "left"; ctx.font = "11px " + css("--font-mono"); ctx.fillStyle = css("--ink");
    ctx.fillText(fmt(r.v, Number.isInteger(r.v) ? 0 : 2), Math.min(xv + 9, w - 50), yc);
  });
  const d = D[ladderSize], lp = d.lp.tight;
  const same = Math.abs(d.lp.naive.bound - lp.bound) < 1e-6 && Math.abs(d.lp.noprec.bound - lp.bound) < 1e-6;
  caption($("#ladCap"), `The ${ladderSize} instance, seed 0 (${d.stats.opportunities} opportunities, ${d.stats.conflict_pairs} conflict pairs). Each dot is an upper bound on the collected priority; the dashed line is the best plan known, ${best}. The LP relaxation of milp.py gives ${fmt(lp.bound, 2)}, ${same ? "and it gives <b>exactly the same number</b> with every big-M set to the horizon and with the precedence rows deleted altogether: the reconfiguration constraints, the heart of the problem, contribute nothing to the bound." : `against ${fmt(d.lp.naive.bound, 2)} with every big-M set to the horizon and ${fmt(d.lp.noprec.bound, 2)} with the precedence rows deleted.`} What the LP does see are the budgets and the one-per-target rows, which already cut the trivial bound of ${d.stats.reachable_weight} down to ${fmt(d.lp.bare.bound, 2)}. At the LP optimum ${lp.fractional} of the ${d.stats.opportunities} selection variables are fractional, and ${int(lp.pairs_off)} of the ${int(lp.pairs_prec)} precedence pairs have [[x_a + x_b \\le 1]], so their rows are switched off completely.${ladderSize === "small" ? ` Branch and bound must close the remaining distance, from ${fmt(lp.bound, 2)} to ${best}, by branching alone.` : ` After ${b.limit} s of branching HiGHS had moved its bound only to ${fmt(b.milp_bound, 0)}, and CP-SAT to ${fmt(b.cpsat_bound, 0)}.`}`);
}
$$("#ladSeg .btn").forEach(btn => btn.addEventListener("click", () => {
  ladderSize = btn.dataset.size;
  $$("#ladSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === btn));
  drawLadder();
}));

/* ---------------- (4) the benchmark ---------------- */
const METHODS = [
  { k: "greedy", t: 0, label: "greedy", c: "--prior" },
  { k: "cpsat", t: 1, label: "CP-SAT", c: "--post" },
  { k: "rolling", t: 2, label: "rolling CP-SAT", c: "--c4" },
  { k: "milp", t: 3, label: "MILP (HiGHS)", c: "--warm" },
];
let benchMetric = "value";
function drawBench() {
  const cv = $("#benchC"), narrow = narrowOf(cv, 600);
  const recs = D.bench;
  const n = recs.length;
  const isTime = benchMetric === "time";
  const tkey = { greedy: "t_greedy", cpsat: "t_cpsat", rolling: "t_rolling", milp: "t_milp" };
  const yv = (r, m) => {
    if (isTime) return Math.log10(Math.max(0.005, r[tkey[m.k]]));
    const best = Math.min(r.cpsat_bound, r.milp_bound == null ? Infinity : r.milp_bound);
    return 100 * r[m.k] / best;
  };
  const fr = frame(cv, narrow ? 1.1 : 2.3, { x: [0, n], y: isTime ? [-2.5, 2.5] : [0, 100],
    xticks: false, yticks: isTime ? [-2, -1, 0, 1, 2] : [0, 20, 40, 60, 80, 100],
    yfmt: isTime ? (v => (10 ** v >= 1 ? String(10 ** v) : (10 ** v).toFixed(-v))) : (v => v + "%"),
    ylabel: isTime ? "solve time [s], log scale" : "value / best proven bound", pad: { b: narrow ? 46 : 40 } });
  const { ctx, X, Y } = fr;
  const gw = (X(1) - X(0)) * 0.8, bw = gw / METHODS.length;
  recs.forEach((r, i) => {
    METHODS.forEach((m, j) => {
      const x0 = X(i) + (X(1) - X(0)) * 0.1 + j * bw, v = yv(r, m);
      const base = isTime ? Y(-2.5) : Y(0);
      ctx.fillStyle = css(m.c);
      ctx.fillRect(x0 + 0.5, Math.min(base, Y(v)), Math.max(1, bw - 1.5), Math.abs(base - Y(v)));
    });
    if (isTime) { ctx.strokeStyle = alpha("--ink", 0.5); ctx.setLineDash([3, 3]); const yl = Y(Math.log10(r.limit)); ctx.beginPath(); ctx.moveTo(X(i) + 2, yl); ctx.lineTo(X(i + 1) - 2, yl); ctx.stroke(); ctx.setLineDash([]); }
    ctx.fillStyle = css("--ink-2"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "top";
    const lab = narrow ? `${r.size[0].toUpperCase()}${r.seed}` : `${r.size} ${r.seed}`;
    ctx.fillText(lab, X(i + 0.5), Y(isTime ? -2.5 : 0) + 6);
    if (!narrow) { ctx.fillStyle = css("--muted"); ctx.fillText(`${int(r.pairs)} pairs`, X(i + 0.5), Y(isTime ? -2.5 : 0) + 20); }
  });
  ctx.textBaseline = "alphabetic";
  const l0 = benchRec("large"), l1 = benchRec("large", 1);
  const sm = recs.filter(r => r.size === "small"), mn = k => Math.min(...sm.map(r => r[k])), mx = k => Math.max(...sm.map(r => r[k]));
  const ratios = sm.map(r => r.t_milp / r.t_cpsat);
  const med = recs.filter(r => r.size === "medium"), belowGreedy = med.filter(r => r.milp < r.greedy);
  caption($("#benchCap"), isTime
    ? `Solve time of each method on the benchmark's eight instances (log scale; the dashed line is the time limit: 20 s small, 60 s medium, 120 s large). On the small instances both exact methods stop because they have <b>proved</b> optimality: CP-SAT in ${fmt(mn("t_cpsat"), 1)} to ${fmt(mx("t_cpsat"), 1)} s, HiGHS in ${fmt(mn("t_milp"), 1)} to ${fmt(mx("t_milp"), 1)} s, ${fmt(Math.min(...ratios), 1)} to ${fmt(Math.max(...ratios), 1)} times longer. From medium on, both run into the limit. Greedy takes a few hundredths of a second everywhere. Times from results/benchmark.json (one run, CP-SAT on 8 threads, HiGHS on 1).`
    : `Collected priority of each method as a percentage of the best upper bound proven on that instance (the smaller of the CP-SAT and HiGHS bounds), on the benchmark's eight instances${narrow ? " (S, M, L = small, medium, large, then the seed)" : ""}. On the small instances CP-SAT and the MILP reach 100%: two independent formulations prove the same optimum. On medium the MILP stays below CP-SAT${belowGreedy.length ? `, and on seed ${belowGreedy.map(r => r.seed).join(", ")} it is even below greedy (${belowGreedy.map(r => `${r.milp} against ${r.greedy}`).join("; ")})` : ""}. On large it collapses: ${l0.milp} on seed 0 and ${l1.milp} on seed 1, against ${l0.greedy} and ${l1.greedy} for greedy, which runs in a fraction of a second. Values from results/benchmark.json, cited, not rerun.`);
}
$$("#benchSeg .btn").forEach(btn => btn.addEventListener("click", () => {
  benchMetric = btn.dataset.m;
  $$("#benchSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === btn));
  drawBench();
}));

function drawAll() { drawHero(); drawKnap(); drawLadder(); drawBench(); }
syncHero();
drawAll(); onRedraw(drawAll);
})();
