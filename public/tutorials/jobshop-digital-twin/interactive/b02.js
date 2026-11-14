/* Which Job Should the Machine Do Next?, chapter B2, dispatching rules in a dynamic shop: the demos.
   Hero: one queue in front of one machine (the paint machine of notation.md, or six orders on the
   workshop's machine m2); the reader picks a rule, or the composite ATCS index with its two scaling
   parameters, and sees every priority index at every decision, the order it produces and the tardiness.
   Second widget: the earliest-expected-finish routing of the first reroute on test day 0 (frequent
   breakdowns), with the repair time the controller assumes and the work left on the other machine.
   Third widget: the four rules on the 20 test days, from the repository's results/runs.csv.
   Everything here is computed as code/b02_numbers.py computes it. Needs ../kit/kit.js and jobshop-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, fit, frame, alpha, onRedraw, narrowOf } = window.Kit;
const { WORKSHOP, FAM, drawGantt, permutations } = window.JS;

/* ---------------- the two queues (same data as code/b02_numbers.py) ---------------- */
const M2 = WORKSHOP.setup[2];
const INST = {
  paint: {
    label: "the paint machine", start: 0, famName: ["blue", "red"], colour: [FAM.blue, FAM.red],
    setup: (f, g) => (f === g ? 0 : 10),
    orders: [["A", 0, 20, 30, 1], ["B", 1, 10, 25, 2], ["C", 0, 15, 70, 3], ["D", 1, 25, 60, 4]],
  },
  six: {
    label: "six orders on machine m2", start: 2, famName: ["family 0", "family 1", "family 2", "family 3"], colour: FAM.colours,
    setup: (f, g) => M2[f][g],
    orders: [["J0", 0, 26, 60, 1], ["J1", 1, 22, 40, 2], ["J2", 2, 19, 95, 3], ["J3", 0, 8, 45, 4], ["J4", 3, 20, 70, 5], ["J5", 1, 15, 110, 6]],
  },
};
const K_VALUES = [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10, 100];
const RULE_TEXT = {
  FIFO: "First come, first served (FIFO)", SPT: "Shortest job first (SPT)", EDD: "Most urgent first (EDD)",
  SST: "Shortest setup first (SST)", ATCS: "Composite index (ATCS)",
};
const o2 = o => ({ name: o[0], f: o[1], p: o[2], d: o[3], pos: o[4] });

function averages(inst) {
  const os = inst.orders.map(o2), pbar = os.reduce((s, o) => s + o.p, 0) / os.length;
  let s = 0, n = 0;
  for (const a of os) for (const b of os) if (a !== b) { s += inst.setup(a.f, b.f); n++; }
  return { pbar, sbar: s / n };
}
/* the index of order o when the machine frees at t, set up for fam; smallest wins except ATCS */
function index(inst, rule, o, t, fam, k1, k2) {
  const s = inst.setup(fam, o.f);
  if (rule === "FIFO") return [o.pos];
  if (rule === "SPT") return [o.p, o.pos];
  if (rule === "EDD") return [o.d, o.pos];
  if (rule === "SST") return [s, o.d, o.pos];
  const { pbar, sbar } = averages(inst);
  return [(1 / o.p) * Math.exp(-Math.max(o.d - o.p - t, 0) / (k1 * pbar)) * Math.exp(-s / (k2 * sbar))];
}
function lexLess(a, b) { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i]; return false; }
function dispatch(inst, rule, k1, k2) {
  const waiting = inst.orders.map(o2), steps = [];
  let t = 0, fam = inst.start;
  while (waiting.length) {
    const keys = new Map(waiting.map(o => [o.name, index(inst, rule, o, t, fam, k1, k2)]));
    let bi = 0;
    for (let i = 1; i < waiting.length; i++) {
      const a = keys.get(waiting[i].name), b = keys.get(waiting[bi].name);
      if (rule === "ATCS" ? (a[0] > b[0] || (a[0] === b[0] && waiting[i].pos < waiting[bi].pos)) : lexLess(a, b)) bi = i;
    }
    const o = waiting.splice(bi, 1)[0];
    steps.push({ t, fam, keys, pick: o.name });
    t += inst.setup(fam, o.f) + o.p; fam = o.f;
  }
  return steps;
}
function run(inst, names) {
  const by = Object.fromEntries(inst.orders.map(o => [o[0], o2(o)]));
  let t = 0, fam = inst.start;
  return names.map(n => {
    const o = by[n], s = inst.setup(fam, o.f), row = { name: n, f: o.f, free: t, start: t + s, end: t + s + o.p, d: o.d, late: Math.max(0, t + s + o.p - o.d), s };
    t = row.end; fam = o.f;
    return row;
  });
}
const totalLate = rows => rows.reduce((s, r) => s + r.late, 0);
const bestCache = {};
function best(key) {
  if (bestCache[key]) return bestCache[key];
  const inst = INST[key];
  let bestSeq = null, bestVal = Infinity, count = 0;
  for (const p of permutations(inst.orders.map(o => o[0]))) {
    count++;
    const v = totalLate(run(inst, p));
    if (v < bestVal || (v === bestVal && p.join("") < bestSeq.join(""))) { bestSeq = p; bestVal = v; }
  }
  return (bestCache[key] = { seq: bestSeq, value: bestVal, count });
}

/* ---------------- the hero ---------------- */
const st = { inst: "paint", rule: "ATCS" };
function fmtKey(rule, k) {
  if (rule === "ATCS") return k[0].toFixed(5);
  if (rule === "SST") return `${k[0]} · ${k[1]}`;
  return String(k[0]);
}
function drawHero() {
  const inst = INST[st.inst], k1 = K_VALUES[+$("#k1").value], k2 = K_VALUES[+$("#k2").value];
  const steps = dispatch(inst, st.rule, k1, k2), seq = steps.map(s => s.pick), rows = run(inst, seq);
  const tot = totalLate(rows), b = best(st.inst), setupMin = rows.reduce((s, r) => s + r.s, 0);
  $("#k1V").textContent = k1; $("#k2V").textContent = k2;
  $$("#k1, #k2").forEach(el => { el.disabled = st.rule !== "ATCS"; });
  $$("#ruleSeg .btn").forEach(btn => btn.setAttribute("aria-pressed", String(btn.dataset.rule === st.rule)));
  $$("#instSeg .btn").forEach(btn => btn.setAttribute("aria-pressed", String(btn.dataset.inst === st.inst)));

  /* Gantt: the machine, then one row per order (in the order the rule runs them): due tick, completion tick, tardiness */
  const cv = $("#qGantt"), t1 = Math.max(...rows.map(r => r.end), ...rows.map(r => r.d)) + 14;
  const blocks = [];
  rows.forEach(r => {
    if (r.s) blocks.push({ lane: 0, start: r.free, end: r.start, kind: "setup" });
    blocks.push({ lane: 0, start: r.start, end: r.end, colour: inst.colour[r.f], label: r.name, kind: "work" });
  });
  const g = drawGantt(cv, blocks, { lanes: 1 + rows.length, t0: 0, t1, laneLabels: ["machine", ...rows.map(r => r.name)], laneHeight: rows.length > 4 ? 24 : 28, labelWidth: 70 });
  const ctx = cv.getContext("2d"), laneH = g.laneH;
  ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle";
  rows.forEach((r, i) => {
    const y = (i + 1) * laneH, xd = g.X(r.d), xc = g.X(r.end), ym = y + laneH / 2;
    if (r.late > 0) { ctx.fillStyle = alpha("--warm", 0.5); ctx.fillRect(xd, y + 6, xc - xd, laneH - 12); }
    ctx.strokeStyle = FAM.red; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xd, y + 3); ctx.lineTo(xd, y + laneH - 3); ctx.stroke();
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xc, y + 3); ctx.lineTo(xc, y + laneH - 3); ctx.stroke();
    ctx.textAlign = "left"; ctx.fillStyle = r.late > 0 ? css("--ink") : css("--muted");
    ctx.fillText(r.late > 0 ? `late ${r.late}` : "on time", Math.min(Math.max(xd, xc) + 5, g.w - 52), ym);
  });

  /* the table of indices */
  const names = inst.orders.map(o => o[0]);
  const head = `<thead><tr><th>decision</th>${names.map(n => `<th>${n}</th>`).join("")}</tr></thead>`;
  const body = steps.map(s => `<tr><td class="words">t = ${s.t}, set up for ${inst.famName[s.fam]}</td>` + names.map(n => {
    if (!s.keys.has(n)) return `<td class="done">·</td>`;
    return `<td class="${n === s.pick ? "pick" : ""}">${fmtKey(st.rule, s.keys.get(n))}</td>`;
  }).join("") + "</tr>").join("");
  $("#qTable").innerHTML = head + "<tbody>" + body + "</tbody>";

  /* the queue's data: family, processing time, due date; and the machine's changeovers */
  $("#qData").innerHTML = `<thead><tr><th>order</th><th>family</th><th>processing (min)</th><th>due at minute</th><th>FIFO rank</th></tr></thead><tbody>` +
    inst.orders.map(o2).map(o => `<tr><td>${o.name}</td><td class="words"><span class="sw" style="background:${inst.colour[o.f]}"></span>${inst.famName[o.f]}</td><td>${o.p}</td><td>${o.d}</td><td>${o.pos}</td></tr>`).join("") + "</tbody>";
  $("#qSetup").innerHTML = st.inst === "paint"
    ? `<tbody><tr><td class="words">Set up for blue at minute 0; a colour change takes 10 min, staying on a colour 0.</td></tr></tbody>`
    : `<thead><tr><th>changeover on m2 (min), from ↓ to →</th>${[0, 1, 2, 3].map(f => `<th>family ${f}</th>`).join("")}</tr></thead><tbody>` +
      M2.map((row, f) => `<tr><td class="words">family ${f}${f === inst.start ? " (set up at minute 0)" : ""}</td>${row.map(v => `<td>${v}</td>`).join("")}</tr>`).join("") + "</tbody>";

  $("#rOrder").textContent = seq.join(" ");
  $("#rTot").textContent = tot + " min";
  $("#rSetup").textContent = setupMin + " min";
  $("#rBest").textContent = `${b.value} min (${b.seq.join(" ")})`;
  $("#rBestL").textContent = `best of all ${b.count} orders`;

  const { pbar, sbar } = averages(inst);
  const what = {
    FIFO: "The index is the order's place in the queue and never changes: a <b>static</b> index.",
    SPT: "The index is the processing time [[p_j]]; it does not depend on [[t]] or on the machine's set-up: <b>static</b>.",
    EDD: "The index is the due date [[d_j]]: <b>static</b>. It ignores how long a job takes and every changeover.",
    SST: "The index is the changeover [[s_m(g_m, f(j))]] from the family [[g_m]] the machine is set up for, ties broken by due date: it changes after every order, a <b>dynamic</b> index.",
    ATCS: `The index [[I_j(t) = \\frac{1}{p_j} e^{-\\max(\\mathrm{slack}_j(t), 0)/(k_1 \\bar p)} e^{-s_m(g_m, f(j))/(k_2 \\bar s)}]], with [[\\mathrm{slack}_j(t) = d_j - p_j - t]], largest wins, with [[\\bar p = ${fmt(pbar, 2)}]], [[\\bar s = ${fmt(sbar, 2)}]], so the slack is measured in units of [[k_1 \\bar p = ${fmt(k1 * pbar, 2)}]] min and the changeover in units of [[k_2 \\bar s = ${fmt(k2 * sbar, 2)}]] min. It moves with [[t]] and with [[g_m]]: <b>dynamic</b>.`,
  }[st.rule];
  const gap = tot - b.value;
  caption($("#heroStatus"), `${RULE_TEXT[st.rule]} on ${inst.label}: ${what} ${st.rule === "ATCS" ? "Largest" : "Smallest"} index first gives <b>${seq.join(" ")}</b>, total tardiness <b>${tot} min</b> with ${setupMin} min of changeovers; the best of all ${b.count} orders gives ${b.value} min${gap ? `, so this rule is ${gap} min worse` : ", which this rule reaches"}.`);
}
$$("#ruleSeg .btn").forEach(btn => btn.addEventListener("click", () => { st.rule = btn.dataset.rule; drawHero(); }));
$$("#instSeg .btn").forEach(btn => btn.addEventListener("click", () => { st.inst = btn.dataset.inst; drawHero(); }));
["#k1", "#k2"].forEach(id => $(id).addEventListener("input", drawHero));

/* ---------------- routing: the first reroute of test day 0, frequent breakdowns, SST ----------------
   Minute 124: m4 fails with job 0 (family 0, operation 2) in its queue. m4: 13 min left on its current
   operation, 17 min committed (job 2 on its way), 7 min of processing for job 0. m1: 3 min away, its
   current operation has rem_1 = 22 min left, nothing else committed, 10 min of processing. The real repair
   ends at minute 137; the controller assumes breakdown + MTTR. */
const R0 = { t: 124, stayRem: 13, stayCommitted: 17, stayP: 7, moveTravel: 3, moveP: 10, realUp: 137 };
function drawRoute() {
  const mttr = +$("#rtM").value, R = +$("#rtR").value, t = R0.t;
  const up = Math.max(Math.round(t + mttr), t + 1);
  const stayFree = up + R0.stayRem + R0.stayCommitted, stayKey = Math.max(t, stayFree) + R0.stayP;
  const arr = t + R0.moveTravel, moveFree = t + R, moveKey = Math.max(arr, moveFree) + R0.moveP;
  const moves = moveKey < stayKey || (moveKey === stayKey && R0.moveP < R0.stayP);
  $("#rtMV").textContent = mttr; $("#rtRV").textContent = R;
  const cv = $("#routeC"), t1 = Math.max(stayKey, moveKey, 200) + 8;
  const blocks = [
    { lane: 0, start: t, end: up, kind: "down" },
    { lane: 0, start: up, end: up + R0.stayRem, colour: alpha("--ink", 0.35), label: "current", kind: "work" },
    { lane: 0, start: up + R0.stayRem, end: stayFree, colour: FAM.colours[3], label: "job 2", kind: "work" },
    { lane: 0, start: stayFree, end: stayKey, colour: FAM.colours[0], label: "job 0", kind: "work" },
    { lane: 1, start: t, end: moveFree, colour: alpha("--ink", 0.35), label: R >= 8 ? "current" : "", kind: "work" },
    { lane: 1, start: t, end: arr, colour: css("--warm"), kind: "agv" },
    { lane: 1, start: Math.max(arr, moveFree), end: moveKey, colour: FAM.colours[0], label: "job 0", kind: "work" },
  ];
  const g = drawGantt(cv, blocks, { lanes: 2, t0: t - 4, t1, laneLabels: ["stay on m4", "move to m1"], laneHeight: 36, labelWidth: 88 });
  const ctx = cv.getContext("2d");
  ctx.strokeStyle = css("--ink"); ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
  const xr = g.X(R0.realUp); ctx.beginPath(); ctx.moveTo(xr, 2); ctx.lineTo(xr, g.laneH - 2); ctx.stroke(); ctx.setLineDash([]);
  ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle";
  [[stayKey, 0], [moveKey, 1]].forEach(([k, lane]) => {
    ctx.fillStyle = css("--ink"); ctx.textAlign = "left";
    ctx.fillText(String(k), Math.min(g.X(k) + 4, g.w - 30), lane * g.laneH + g.laneH / 2);
  });
  $("#rtStay").textContent = stayKey; $("#rtMove").textContent = moveKey; $("#rtDec").textContent = moves ? "move to m1" : "stay on m4";
  const threshold = mttr + R0.stayRem + R0.stayCommitted + R0.stayP - R0.moveP;
  caption($("#routeCap"), `Job 0 waits at m4 when m4 fails at minute 124. The controller expects m4 back at [[124 + ${mttr} = ${up}]] (the shaded block; the dashed line is the real end of the repair, minute 137, which it does not know). <b>Stay</b>: [[\\max(124, ${up} + ${R0.stayRem} + ${R0.stayCommitted}) + ${R0.stayP} = ${stayKey}]]; the 17 min of job 2, still on its way, are counted ahead of job 0. <b>Move</b>: the cart needs 3 min (orange), m1 is busy for [[\\mathrm{rem}_1 = ${R}]] more min: [[\\max(127, 124 + ${R}) + ${R0.moveP} = ${moveKey}]]. Earliest expected finish: <b>${moves ? "move to m1" : "stay on m4"}</b>. With this repair estimate the job moves whenever [[\\mathrm{rem}_1 < ${threshold}]].`);
}
["#rtM", "#rtR"].forEach(id => $(id).addEventListener("input", drawRoute));

/* ---------------- the four rules on the 20 test days (results/runs.csv) ---------------- */
const DAYS = {"no breakdowns":{"FIFO":{"mean_tardiness":[6.8,19.833,3.683,9.233,2.567,1.967,17.5,97.433,32.683,7.8,21.483,4.667,23.583,2.483,5.667,2.55,3.433,41.5,3.867,2.1],"max_tardiness":[71.0,129.0,50.0,77.0,49.0,39.0,142.0,260.0,158.0,52.0,150.0,67.0,136.0,41.0,85.0,35.0,53.0,164.0,55.0,38.0],"setup_time":[1039.0,1042.0,1050.0,1012.0,1051.0,1001.0,1143.0,1100.0,1061.0,1048.0,1090.0,1147.0,1160.0,1065.0,1166.0,1163.0,1241.0,1195.0,1053.0,952.0]},"SPT":{"mean_tardiness":[7.117,11.483,4.333,5.783,1.133,1.083,10.967,55.8,26.433,6.467,15.133,2.7,15.5,2.067,1.483,3.4,3.95,19.3,3.317,2.333],"max_tardiness":[90.0,136.0,56.0,50.0,26.0,42.0,90.0,585.0,170.0,82.0,185.0,36.0,144.0,36.0,34.0,62.0,45.0,131.0,73.0,39.0],"setup_time":[1133.0,964.0,1098.0,1030.0,1004.0,949.0,966.0,974.0,1044.0,1061.0,1045.0,1126.0,1087.0,1041.0,985.0,1204.0,1139.0,1033.0,1019.0,980.0]},"EDD":{"mean_tardiness":[2.95,11.6,2.083,2.633,1.85,0.3,5.2,47.267,18.983,2.317,12.55,1.867,9.817,1.183,1.783,1.0,4.817,9.3,1.6,1.2],"max_tardiness":[34.0,90.0,34.0,33.0,31.0,9.0,85.0,161.0,90.0,49.0,101.0,37.0,100.0,32.0,40.0,34.0,59.0,60.0,45.0,31.0],"setup_time":[1013.0,1073.0,1010.0,952.0,1051.0,956.0,967.0,1010.0,979.0,958.0,1020.0,1034.0,1043.0,995.0,1057.0,1182.0,1258.0,1015.0,1007.0,928.0]},"SST":{"mean_tardiness":[6.133,9.267,2.2,6.033,4.75,0.9,10.633,51.683,21.067,9.683,10.767,3.117,10.933,3.45,3.083,3.617,2.55,23.617,2.467,1.083],"max_tardiness":[75.0,82.0,31.0,67.0,73.0,32.0,92.0,233.0,143.0,195.0,232.0,57.0,127.0,72.0,34.0,83.0,38.0,190.0,62.0,26.0],"setup_time":[974.0,848.0,1017.0,946.0,1004.0,931.0,891.0,618.0,794.0,1005.0,838.0,954.0,835.0,979.0,1042.0,1061.0,1072.0,796.0,928.0,831.0]}},"frequent breakdowns":{"FIFO":{"mean_tardiness":[32.833,71.383,40.0,56.65,22.017,14.317,153.5,169.933,99.45,78.833,34.3,32.167,89.55,32.1,31.95,18.117,47.767,144.4,19.85,27.6],"max_tardiness":[210.0,210.0,186.0,255.0,147.0,121.0,359.0,464.0,300.0,239.0,169.0,141.0,269.0,172.0,154.0,123.0,205.0,355.0,96.0,148.0],"setup_time":[1024.0,1023.0,1082.0,1116.0,1146.0,1009.0,1100.0,1014.0,1047.0,1115.0,1039.0,1175.0,1079.0,1135.0,984.0,1257.0,1302.0,1090.0,1073.0,1013.0]},"SPT":{"mean_tardiness":[37.3,41.467,48.5,34.95,15.717,14.4,104.267,127.417,69.15,41.583,38.617,22.2,71.167,21.65,26.183,11.233,27.783,114.867,20.667,19.617],"max_tardiness":[486.0,257.0,207.0,265.0,115.0,96.0,371.0,390.0,341.0,238.0,318.0,135.0,321.0,128.0,123.0,100.0,110.0,479.0,132.0,95.0],"setup_time":[1082.0,923.0,1199.0,973.0,1042.0,1005.0,946.0,913.0,1008.0,957.0,1078.0,1168.0,1002.0,1078.0,961.0,1136.0,1159.0,1118.0,1035.0,942.0]},"EDD":{"mean_tardiness":[25.533,44.117,31.25,36.883,15.833,11.283,71.35,107.483,44.817,38.95,24.7,16.333,65.367,17.583,25.967,11.717,38.8,91.567,13.217,19.867],"max_tardiness":[146.0,148.0,156.0,143.0,81.0,84.0,158.0,285.0,166.0,158.0,162.0,108.0,214.0,102.0,195.0,103.0,145.0,259.0,156.0,123.0],"setup_time":[1047.0,884.0,1056.0,1027.0,1025.0,990.0,896.0,1017.0,984.0,1063.0,1031.0,988.0,1016.0,1045.0,978.0,1216.0,1139.0,1005.0,1048.0,831.0]},"SST":{"mean_tardiness":[17.9,45.4,23.817,38.967,12.533,10.133,74.8,99.033,60.467,33.55,18.617,23.717,44.917,22.283,26.75,13.217,21.367,64.067,14.367,17.083],"max_tardiness":[130.0,244.0,142.0,283.0,182.0,92.0,467.0,463.0,299.0,187.0,233.0,198.0,217.0,143.0,150.0,108.0,127.0,283.0,129.0,169.0],"setup_time":[910.0,680.0,821.0,827.0,870.0,869.0,629.0,542.0,770.0,789.0,899.0,881.0,752.0,954.0,900.0,1059.0,938.0,607.0,944.0,753.0]}}};
const RULES4 = ["FIFO", "SPT", "EDD", "SST"];
const MEASURE = {
  mean_tardiness: { title: "mean tardiness of the day (min)", short: "mean tardiness" },
  max_tardiness: { title: "tardiness of the most tardy job of the day (min)", short: "max tardiness" },
  setup_time: { title: "changeover minutes in the day", short: "changeover minutes" },
  diff: { title: "SST minus EDD, mean tardiness, day by day (min)", short: "SST − EDD" },
};
const dst = { level: "frequent breakdowns", measure: "mean_tardiness" };
const meanOf = xs => xs.reduce((s, v) => s + v, 0) / xs.length;
function drawDays() {
  $$("#dLevel .btn").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.level === dst.level)));
  $$("#dMeasure .btn").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.measure === dst.measure)));
  const cv = $("#daysC"), narrow = narrowOf(cv), D = DAYS[dst.level];
  const ink = css("--ink"), muted = css("--muted");
  if (dst.measure === "diff") {
    const d = D.SST.mean_tardiness.map((v, i) => v - D.EDD.mean_tardiness[i]);
    const lo = Math.min(-5, ...d) - 2, hi = Math.max(5, ...d) + 2;
    const step = hi - lo > 40 ? 10 : 5, yt = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) yt.push(v);
    const fr = frame(cv, narrow ? 1.25 : 2.3, { x: [-0.5, 19.5], y: [lo, hi], xlabel: "test day (seed)", ylabel: MEASURE.diff.title, xticks: [0, 5, 10, 15, 19], yticks: yt, yfmt: v => (v > 0 ? "+" : "") + v });
    const { ctx, X, Y } = fr;
    ctx.strokeStyle = muted; ctx.beginPath(); ctx.moveTo(X(-0.5), Y(0)); ctx.lineTo(X(19.5), Y(0)); ctx.stroke();
    d.forEach((v, i) => {
      ctx.strokeStyle = v < 0 ? css("--post") : css("--warm"); ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(X(i), Y(0)); ctx.lineTo(X(i), Y(v)); ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(X(i), Y(v), 3.5, 0, 2 * Math.PI); ctx.fill();
    });
    const m = meanOf(d), better = d.filter(v => v < 0).length;
    ctx.strokeStyle = ink; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(-0.5), Y(m)); ctx.lineTo(X(19.5), Y(m)); ctx.stroke(); ctx.setLineDash([]);
    caption($("#daysCap"), `${dst.level[0].toUpperCase() + dst.level.slice(1)}: on each test day, shortest setup first minus most urgent first. Blue bars below zero are days SST wins: <b>${better} of 20</b>. The dashed line is the mean difference, <b>${m >= 0 ? "+" : ""}${fmt(m, 1)} min</b>. ${dst.level === "no breakdowns" ? "Without breakdowns most urgent first wins on most days." : "With breakdowns the sign of the mean flips, but the days split almost evenly: the paired interval of Part 4 still contains zero."}`);
    return;
  }
  const key = dst.measure, all = RULES4.flatMap(r => D[r][key]);
  const hi = Math.max(...all) * 1.08, lo = key === "setup_time" ? Math.min(...all) * 0.9 : 0;
  const fr = frame(cv, narrow ? 1.25 : 2.3, { x: [-0.5, 3.5], y: [lo, hi], ylabel: MEASURE[key].title, xticks: false });
  const { ctx, X, Y } = fr;
  const means = {};
  RULES4.forEach((r, i) => {
    const vals = D[r][key];
    means[r] = meanOf(vals);
    vals.forEach((v, s) => {
      const jitter = ((s * 7) % 20 - 9.5) / 9.5 * 0.22;
      ctx.fillStyle = css("--accent"); ctx.beginPath(); ctx.arc(X(i + jitter), Y(v), 3.6, 0, 2 * Math.PI); ctx.fill();
    });
    ctx.strokeStyle = ink; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(X(i - 0.32), Y(means[r])); ctx.lineTo(X(i + 0.32), Y(means[r])); ctx.stroke();
    ctx.fillStyle = ink; ctx.font = "600 12px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(r, X(i), Y(lo) + 6);
    ctx.font = "600 11px " + css("--font-mono"); ctx.fillStyle = ink;
    if (narrow) { ctx.textBaseline = "bottom"; ctx.fillText(fmt(means[r], key === "mean_tardiness" ? 1 : 0), X(i), Y(means[r]) - 3); }
    else { ctx.textBaseline = "middle"; ctx.textAlign = "left"; ctx.fillText(fmt(means[r], key === "mean_tardiness" ? 1 : 0), X(i + 0.34) + 3, Y(means[r])); }
  });
  const order = RULES4.slice().sort((a, b) => means[a] - means[b]);
  caption($("#daysCap"), `${dst.level[0].toUpperCase() + dst.level.slice(1)}, ${MEASURE[key].short}: one dot per test day, the bar is the mean over the 20 days. From best to worst: <b>${order.map(r => `${r} ${fmt(means[r], key === "mean_tardiness" ? 1 : 0)}`).join(", ")}</b>. ${key === "setup_time" ? "Shortest setup first saves changeovers at both levels; the saving is larger with breakdowns." : key === "max_tardiness" ? "Most urgent first protects the most tardy job at both levels, as Jackson's rule does on one machine." : "The spread between days is far wider than the gaps between the rules: compare rules day by day, not by their clouds."}`);
}
$$("#dLevel .btn").forEach(b => b.addEventListener("click", () => { dst.level = b.dataset.level; drawDays(); }));
$$("#dMeasure .btn").forEach(b => b.addEventListener("click", () => { dst.measure = b.dataset.measure; drawDays(); }));

function drawAll() { drawHero(); drawRoute(); drawDays(); }
drawAll();
onRedraw(drawAll);
})();
