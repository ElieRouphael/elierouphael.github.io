/* Which Job Should the Machine Do Next?, chapter B1, discrete-event simulation: the demos.
   The hero is a tiny event-driven simulator (one or two machines, one shared queue, blue and red
   jobs, a 10-minute colour change, breakdowns that suspend the work) that advances one event at a
   time with its event list on show; it is the simulator of code/b01_numbers.py (class Mini), line
   for line. A second widget draws exponential, uniform and constant gaps and what is left of a gap
   after s minutes; a third generates breakdowns as jobshop/scenario.py does and measures the
   availability. Needs ../kit/kit.js and jobshop-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, fit, alpha, onRedraw, narrowOf, plotLine, histogram, ticks } = window.Kit;
const { FAM, WORKSHOP, rng, expo, drawGantt } = window.JS;

const CHANGE = 10;                       // minutes for a colour change, as in the paint example
const PAINT_JOBS = [                     // the paint orders of chapters A1-A2, now arriving over time
  { name: "A", colour: "blue", p: 20, arrival: 0 },
  { name: "B", colour: "red",  p: 10, arrival: 5 },
  { name: "C", colour: "blue", p: 15, arrival: 12 },
  { name: "D", colour: "red",  p: 25, arrival: 40 },
];
const PAINT_DOWN = [[25, 33]];           // machine 0 is down from minute 25 to minute 33
const N_RANDOM = 30, P_CHOICES = [10, 15, 20, 25], MEAN_P = 17.5, MTBF = 120, MTTR = 20;

/* ---------------- the simulator ---------------- */
class Mini {
  constructor(jobs, nMachines, rule, downtime) {
    this.jobs = jobs; this.rule = rule; this.t = 0; this.seq = 0;
    this.events = []; this.cancelled = new Set(); this.queue = [];
    this.m = Array.from({ length: nMachines }, () => ({ job: null, last: "blue", down: false, remS: 0, remP: 0, seg: null, ev: null, kind: "", downSince: 0 }));
    this.done = {}; this.nEvents = 0; this.downtime = downtime; this.dtIdx = downtime.map(() => 0);
    this.blocks = []; this.history = [[0, 0]]; this.area = 0; this.note = "";
    this.inShop = 0; this.arrived = 0; this.flowSum = 0; this.evTimes = new Set();
    this.push(jobs[0].arrival, "arrival", 0);
    downtime.forEach((spans, k) => { if (spans.length) this.push(spans[0][0], "failure", k); });
  }
  push(t, kind, data) { this.seq++; this.events.push({ t, seq: this.seq, kind, data }); return this.seq; }
  pending() { return this.events.filter(e => !this.cancelled.has(e.seq)).sort((a, b) => a.t - b.t || a.seq - b.seq); }
  finished() { return Object.keys(this.done).length === this.jobs.length; }
  startSegment(k) {
    const m = this.m[k];
    if (m.remS > 0) { m.kind = "setup"; m.seg = this.t; m.ev = this.push(this.t + m.remS, "setup done", k); }
    else { m.kind = "proc"; m.seg = this.t; m.ev = this.push(this.t + m.remP, "done", k); }
  }
  closeSegment(k) {
    const m = this.m[k];
    if (m.seg != null && this.t > m.seg)
      this.blocks.push({ lane: k, start: m.seg, end: this.t, kind: m.kind === "setup" ? "setup" : "work",
                         colour: m.kind === "setup" ? null : FAM[m.job.colour], label: m.job.name });
    m.seg = null;
  }
  key(j, m) {
    if (this.rule === "SPT") return [j.p, j.arrival, j.name];
    if (this.rule === "SST") return [j.colour === m.last ? 0 : CHANGE, j.arrival, j.name];
    return [j.arrival, j.name];
  }
  tryStart(k) {
    const m = this.m[k];
    if (m.job || m.down || !this.queue.length) return null;
    const less = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i]; return false; };
    let best = this.queue[0];
    for (const j of this.queue) if (less(this.key(j, m), this.key(best, m))) best = j;
    this.queue.splice(this.queue.indexOf(best), 1);
    m.job = best; m.remS = best.colour === m.last ? 0 : CHANGE; m.remP = best.p; m.last = best.colour;
    this.startSegment(k);
    return best;
  }
  step() {
    const ev = this.pending()[0];
    if (!ev || this.finished()) return false;
    this.events.splice(this.events.indexOf(ev), 1);
    this.area += this.inShop * (ev.t - this.t);
    this.t = ev.t; this.nEvents++; this.evTimes.add(ev.t);
    const k = ev.data, m = this.m[k];
    let note = "";
    if (ev.kind === "arrival") {
      const j = this.jobs[ev.data];
      this.queue.push(j); this.inShop++; this.arrived++;
      if (ev.data + 1 < this.jobs.length) this.push(this.jobs[ev.data + 1].arrival, "arrival", ev.data + 1);
      const started = this.m.map((_, i) => this.tryStart(i));
      note = `<b>order ${j.name}</b> (${j.colour}, ${j.p} min) arrives and joins the queue` + started.map((s, i) => s ? `; machine ${i} is free and starts it${this.m[i].remS > 0 ? " after a colour change" : ""}` : "").join("")
        + (ev.data + 1 < this.jobs.length ? `. The next arrival is put on the list for minute ${this.jobs[ev.data + 1].arrival}.` : ". It was the last order of the day.");
    } else if (ev.kind === "setup done") {
      this.closeSegment(k); m.remS = 0; this.startSegment(k);
      note = `The changeover on machine ${k} is finished; <b>order ${m.job.name}</b> is processed, done at minute ${this.t + m.remP}.`;
    } else if (ev.kind === "done") {
      this.closeSegment(k);
      const j = m.job; this.done[j.name] = this.t; this.inShop--; this.flowSum += this.t - j.arrival; m.job = null;
      const nxt = this.tryStart(k);
      note = `<b>order ${j.name}</b> leaves after ${this.t - j.arrival} min in the shop. ` + (nxt ? `${this.rule} picks <b>order ${nxt.name}</b> from the queue${m.remS ? `; a colour change of ${CHANGE} min comes first` : ""}.` : (this.queue.length ? "" : "The queue is empty: machine " + k + " waits."));
    } else if (ev.kind === "failure") {
      m.down = true; m.downSince = this.t;
      if (m.job) {
        const kind = m.kind, el = this.t - m.seg; this.closeSegment(k);
        if (kind === "setup") m.remS -= el; else m.remP -= el;
        this.cancelled.add(m.ev);
        note = `Machine ${k} breaks down during the ${kind === "setup" ? "changeover" : "processing"} of <b>order ${m.job.name}</b>: its end event is struck off the list, ${kind === "setup" ? m.remS : m.remP} min of it remain.`;
      } else note = `Machine ${k} breaks down while idle.`;
      const up = this.downtime[k][this.dtIdx[k]][1];
      this.push(up, "repair", k);
      note += ` The repair is put on the list for minute ${up}.`;
    } else if (ev.kind === "repair") {
      m.down = false;
      this.blocks.push({ lane: k, start: m.downSince, end: this.t, kind: "down" });
      this.dtIdx[k]++;
      if (this.dtIdx[k] < this.downtime[k].length) this.push(this.downtime[k][this.dtIdx[k]][0], "failure", k);
      if (m.job) { this.startSegment(k); note = `Machine ${k} is repaired and resumes <b>order ${m.job.name}</b> where it stopped.`; }
      else { const s = this.tryStart(k); note = `Machine ${k} is repaired` + (s ? ` and starts <b>order ${s.name}</b>.` : "."); }
    }
    this.history.push([this.t, this.inShop]);
    this.note = `Minute ${this.t}, ${ev.kind}: ` + note;
    return !this.finished();
  }
  /* what is drawn: closed blocks plus the segments and breakdowns still open at the clock */
  drawBlocks() {
    const out = this.blocks.slice();
    this.m.forEach((m, k) => {
      if (m.job && m.seg != null && this.t > m.seg) out.push({ lane: k, start: m.seg, end: this.t, kind: m.kind === "setup" ? "setup" : "work", colour: m.kind === "setup" ? null : FAM[m.job.colour], label: m.job.name });
      if (m.down && this.t > m.downSince) out.push({ lane: k, start: m.downSince, end: this.t, kind: "down" });
    });
    return out;
  }
}

/* ---------------- the hero ---------------- */
const st = { mode: "paint", rule: "FIFO", M: 1, bd: true, rho: 0.6, seed: 1, sim: null, end: null, timer: null };

function makeScenario() {
  if (st.mode === "paint") {
    const down = Array.from({ length: st.M }, (_, k) => (st.bd && k === 0 ? PAINT_DOWN : []));
    return { jobs: PAINT_JOBS, down };
  }
  const r = rng(st.seed), gap = MEAN_P / (st.rho * st.M);
  let s = 0;
  const jobs = [];
  for (let i = 0; i < N_RANDOM; i++) {
    s += expo(r, gap);
    jobs.push({ name: String(i), colour: r() < 0.5 ? "blue" : "red", p: P_CHOICES[Math.floor(r() * 4)], arrival: Math.floor(s) });
  }
  const down = Array.from({ length: st.M }, (_, k) => {
    if (!st.bd) return [];
    const b = rng(1000 * st.seed + 17 + k), spans = [];
    let t = 0;
    while (t < 40000) {
      const d = t + Math.max(1, Math.round(expo(b, MTBF))), u = d + Math.max(1, Math.round((0.5 + b()) * MTTR));
      spans.push([d, u]); t = u;
    }
    return spans;
  });
  return { jobs, down };
}
function reset() {
  stop();
  const sc = makeScenario();
  st.sim = new Mini(sc.jobs, st.M, st.rule, sc.down);
  const full = new Mini(sc.jobs, st.M, st.rule, sc.down);
  while (full.step()) { /* run to the end once, for the axes */ }
  st.end = full;
  st.sim.note = "Minute 0, nothing has happened yet. The event list holds the first arrival" + (st.bd ? " and the first breakdown of each machine" : "") + ". Press <b>Next event</b>: the clock jumps to the top of the list.";
  drawHero();
}
function stop() { if (st.timer) { clearInterval(st.timer); st.timer = null; } $("#hPlay").textContent = "Play"; $("#hPlay").setAttribute("aria-pressed", "false"); }

function drawHero() {
  const s = st.sim, e = st.end, T1 = Math.max(10, e.t);
  /* Gantt: machines, then a lane of arrivals */
  const lanes = st.M + 1, gc = $("#hGantt");
  const labels = Array.from({ length: st.M }, (_, k) => `machine ${k}`).concat(["arrivals"]);
  const g = drawGantt(gc, s.drawBlocks(), { lanes, t0: 0, t1: T1, laneLabels: labels, laneHeight: 30, labelWidth: 78 });
  const ctx = gc.getContext("2d");
  const yA = st.M * g.laneH + g.laneH / 2;
  s.jobs.forEach((j, i) => {
    if (j.arrival > s.t || i >= s.arrived) return;
    const x = g.X(j.arrival);
    ctx.fillStyle = FAM[j.colour];
    ctx.beginPath(); ctx.moveTo(x, yA - 7); ctx.lineTo(x - 5, yA + 5); ctx.lineTo(x + 5, yA + 5); ctx.closePath(); ctx.fill();
  });
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(g.X(s.t), 0); ctx.lineTo(g.X(s.t), lanes * g.laneH); ctx.stroke(); ctx.setLineDash([]);
  /* N(t) and its running average */
  const nc = $("#hN"), narrow = narrowOf(nc);
  const maxN = Math.max(3, ...e.history.map(h => h[1]));
  const fr = frame(nc, narrow ? 2.0 : 3.6, { x: [0, T1], y: [0, maxN + 0.5], ylabel: "jobs in the shop", yticks: ticks(0, maxN, Math.min(4, maxN)).filter(v => Number.isInteger(v)), pad: { l: 78, r: 10 } });
  const xs = [], ys = [];
  s.history.forEach(([t, n], i) => { if (i) { xs.push(t); ys.push(s.history[i - 1][1]); } xs.push(t); ys.push(n); });
  xs.push(s.t); ys.push(s.inShop);
  plotLine(fr, xs, ys, { color: css("--accent"), width: 2, fill: alpha("--accent", 0.12) });
  /* running average L(t) = (1/t) * integral of N */
  const ax = [], ay = []; let area = 0;
  for (let i = 1; i < s.history.length; i++) { const [t0, n0] = s.history[i - 1], t1 = s.history[i][0]; area += n0 * (t1 - t0); if (t1 > 0) { ax.push(t1); ay.push(area / t1); } }
  if (ax.length) plotLine(fr, ax, ay, { color: css("--ink"), width: 1.5, dash: [5, 3] });
  /* the event list */
  const pend = s.pending();
  const what = ev => ev.kind === "arrival" ? `order ${s.jobs[ev.data].name}` : `machine ${ev.data}` + (["setup done", "done"].includes(ev.kind) && s.m[ev.data].job ? `, order ${s.m[ev.data].job.name}` : "");
  $("#hList").innerHTML = "<thead><tr><th>time</th><th>event</th><th>for</th></tr></thead><tbody>" +
    (pend.length && !s.finished() ? pend.slice(0, 7).map((ev, i) => `<tr${i === 0 ? ' class="next"' : ""}><td>${ev.t}</td><td>${ev.kind}</td><td>${what(ev)}</td></tr>`).join("") : '<tr><td colspan="3">empty: every order has left, the run is over</td></tr>') + "</tbody>";
  /* readouts */
  const done = Object.keys(s.done).length, W = done ? s.flowSum / done : NaN, L = s.t > 0 ? s.area / s.t : NaN, lam = s.t > 0 ? s.arrived / s.t : NaN;
  $("#rClock").textContent = s.t + " min"; $("#rEv").textContent = s.nEvents; $("#rSteps").textContent = s.nEvents ? `${s.t + 1} (${s.t + 1 - s.evTimes.size} empty)` : "–";
  $("#rN").textContent = `${s.inShop} (${done} of ${s.jobs.length} done)`;
  $("#rL").textContent = fmt(L, 3); $("#rLW").textContent = fmt(lam * W, 3);
  $("#hRhoV").textContent = fmt(st.rho, 2);
  let tail = "";
  if (s.finished()) {
    tail = ` <b>The run is over.</b> ${s.nEvents} events moved the clock through ${s.t} minutes; a loop that advanced one minute at a time would visit the ${s.t + 1} instants 0 to ${s.t}, ${s.t + 1 - s.evTimes.size} of them with nothing to do. The area under [[N(t)]] is ${s.area} job-minutes, which is also the sum of the flow times, so [[L = ${s.area}/${s.t} = ${fmt(L, 3)}]] and [[\\lambda_T W_T = (${s.arrived}/${s.t}) \\times ${fmt(W, 2)} = ${fmt(lam * W, 3)}]]: Little's law, exact because the shop started and ended empty.` + (st.mode === "paint" && st.rule === "SPT" ? " On these four orders SPT happens to choose exactly as FIFO does; SST avoids two colour changes (exercise 1)." : "");
  } else if (done && s.inShop) tail = ` Until the shop is empty again, [[L]] and [[\\lambda_T W_T]] differ: the jobs still inside have not yet contributed their flow time to [[W_T]] (Part 6).`;
  caption($("#hStatus"), s.note + tail);
  $("#hRhoF").hidden = st.mode !== "random"; $("#hSeedF").hidden = st.mode !== "random";
  $("#hStep").disabled = s.finished(); $("#hEnd").disabled = s.finished(); $("#hPlay").disabled = s.finished();
}

function segBind(id, key, conv = v => v) {
  $$(`#${id} .btn`).forEach(b => b.addEventListener("click", () => {
    st[key] = conv(b.dataset.v);
    $$(`#${id} .btn`).forEach(o => o.setAttribute("aria-pressed", String(o === b)));
    reset();
  }));
}
segBind("hMode", "mode"); segBind("hRule", "rule"); segBind("hM", "M", Number);
$("#hBd").addEventListener("change", () => { st.bd = $("#hBd").checked; reset(); });
$("#hRho").addEventListener("input", () => { st.rho = +$("#hRho").value; reset(); });
$("#hSeed").addEventListener("change", () => { st.seed = Math.max(1, Math.min(999, Math.round(+$("#hSeed").value || 1))); $("#hSeed").value = st.seed; reset(); });
$("#hStep").addEventListener("click", () => { stop(); st.sim.step(); drawHero(); });
$("#hEnd").addEventListener("click", () => { stop(); while (st.sim.step()) { /* to the end */ } drawHero(); });
$("#hReset").addEventListener("click", reset);
$("#hPlay").addEventListener("click", () => {
  if (st.timer) { stop(); return; }
  $("#hPlay").textContent = "Pause"; $("#hPlay").setAttribute("aria-pressed", "true");
  st.timer = setInterval(() => { if (!st.sim.step()) stop(); drawHero(); }, st.mode === "paint" ? 700 : 260);
});

/* ---------------- gaps and the memoryless property ---------------- */
const MEAN_GAP = WORKSHOP.meanInterarrival;          // 20.83 min
const pst = { dist: "expo", s: 20 };
const SAMPLES = (() => {
  const r = rng(7), n = 40000, out = { expo: new Float64Array(n), unif: new Float64Array(n), fixed: new Float64Array(n) };
  for (let i = 0; i < n; i++) { out.expo[i] = expo(r, MEAN_GAP); out.unif[i] = 2 * MEAN_GAP * r(); out.fixed[i] = MEAN_GAP; }
  return out;
})();
function drawPoisson() {
  const X = SAMPLES[pst.dist], s = pst.s, t = 20;
  /* a timeline of 8 hours of arrivals with these gaps */
  const lc = $("#pLine"), { ctx, w, h } = fit(lc, narrowOf(lc) ? 4.5 : 9);
  const H = 480, x = v => 10 + v / H * (w - 20);
  ctx.strokeStyle = css("--rule"); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(10, h / 2); ctx.lineTo(w - 10, h / 2); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let v = 0; v <= H; v += 60) ctx.fillText(v / 60 + " h", x(v), h / 2 + 9);
  ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2;
  let tt = 0, n = 0;
  for (let i = 100; tt + X[i] <= H; i++) { tt += X[i]; n++; ctx.beginPath(); ctx.moveTo(x(tt), h / 2 - 9); ctx.lineTo(x(tt), h / 2 + 4); ctx.stroke(); }
  /* histogram of X and of X - s given X > s */
  const hc = $("#pHist"), narrow = narrowOf(hc), lo = 0, hi = 100, bins = 40;
  const rest = []; for (const v of X) if (v > s) rest.push(v - s);
  const dens = d => d.length ? 1 / (d.length * (hi - lo) / bins) : 0;
  const binMax = d => { const c = new Float64Array(bins); for (const v of d) { const i = Math.floor((v - lo) / ((hi - lo) / bins)); if (i >= 0 && i < bins) c[i]++; } return Math.max(...c) * dens(d); };
  const ymax = pst.dist === "fixed" ? 0.3 : 1.1 * Math.max(0.05, binMax(X), rest.length ? binMax(rest) : 0);
  const fr = frame(hc, narrow ? 1.5 : 2.6, { x: [lo, hi], y: [0, ymax], xlabel: "minutes", ylabel: "density", yfmt: v => fmt(v, 2) });
  if (pst.dist !== "fixed") {
    histogram(fr, X, lo, hi, bins, dens(X), alpha("--ink", 0.18));
    if (rest.length) histogram(fr, rest, lo, hi, bins, dens(rest), alpha("--accent", 0.55));
  } else {
    const c = fr.ctx; c.fillStyle = alpha("--ink", 0.35); c.fillRect(fr.X(MEAN_GAP) - 2, fr.Y(ymax * 0.95), 4, fr.Y(0) - fr.Y(ymax * 0.95));
    if (s < MEAN_GAP) { c.fillStyle = css("--accent"); c.fillRect(fr.X(MEAN_GAP - s) - 2, fr.Y(ymax * 0.95), 4, fr.Y(0) - fr.Y(ymax * 0.95)); }
  }
  const xs = [], ys = []; for (let v = 0; v <= hi; v += 1) { xs.push(v); ys.push(Math.exp(-v / MEAN_GAP) / MEAN_GAP); }
  if (pst.dist === "expo") plotLine(fr, xs, ys, { color: css("--ink"), width: 1.5, dash: [5, 3] });
  /* the numbers */
  const pT = X.filter(v => v > t).length / X.length;
  const cond = rest.length ? rest.filter(v => v > t).length / rest.length : NaN;
  const mRest = rest.length ? rest.reduce((a, b) => a + b, 0) / rest.length : NaN;
  $("#pSV").textContent = s;
  const name = { expo: "exponential", unif: "uniform on [0, 41.7]", fixed: "constant" }[pst.dist];
  let txt = `${name[0].toUpperCase() + name.slice(1)} gaps with mean 20.83 min: ${n} arrivals in these 8 hours. `;
  if (!rest.length) txt += `No gap is longer than ${s} min, so after ${s} min of waiting the next order is overdue: the past tells you exactly when it comes.`;
  else txt += `Of the gaps that last more than [[s = ${s}]] min, the share that also lasts ${t} min more is [[P(X > s + ${t} \\mid X > s) = ${fmt(cond, 3)}]], against [[P(X > ${t}) = ${fmt(pT, 3)}]] for a fresh gap; the mean remaining wait is ${fmt(mRest, 1)} min.` +
    (pst.dist === "expo" ? ` The two agree (up to sampling noise), at every [[s]]: the exponential forgets how long you have waited, and the blue bars keep the shape of the grey ones and of the dashed density [[\\lambda e^{-\\lambda x}]].` : pst.dist === "unif" ? ` The longer you have waited, the sooner the next order comes: uniform gaps remember.` : "");
  caption($("#pCap"), txt);
}
$$("#pDist .btn").forEach(b => b.addEventListener("click", () => {
  pst.dist = b.dataset.v; $$("#pDist .btn").forEach(o => o.setAttribute("aria-pressed", String(o === b))); drawPoisson();
}));
$("#pS").addEventListener("input", () => { pst.s = +$("#pS").value; drawPoisson(); });

/* ---------------- breakdowns and availability ---------------- */
const ast = { mtbf: 120, mttr: 20, days: 20 };
const DAY = 1400;
function spansFor(seed, mtbf, mttr, horizon) {
  const b = rng(seed), out = []; let t = 0;
  while (t < horizon) { const d = t + Math.max(1, Math.round(expo(b, mtbf))), u = d + Math.max(1, Math.round((0.5 + b()) * mttr)); out.push([d, u]); t = u; }
  return out;
}
function drawAvail() {
  const { mtbf, mttr, days } = ast, A = mtbf / (mtbf + mttr);
  /* day 1 of the five machines */
  const blocks = [];
  const machines = [0, 1, 2, 3, 4].map(m => spansFor(5000 + 31 * m, mtbf, mttr, days * DAY));
  machines.forEach((sp, m) => {
    blocks.push({ lane: m, start: 0, end: DAY, kind: "agv", colour: alpha("--accent", 0.35) });
    sp.forEach(([d, u]) => { if (d < DAY) blocks.push({ lane: m, start: d, end: Math.min(u, DAY), kind: "work", colour: FAM.down }); });
  });
  drawGantt($("#aDay"), blocks, { lanes: 5, t0: 0, t1: DAY + 40, laneLabels: ["machine 0", "machine 1", "machine 2", "machine 3", "machine 4"], laneHeight: 24, labelWidth: 78 });
  /* running downtime share over the days, machine by machine (each machine-day is [0, DAY) of its own stream cut into days) */
  let down = 0, count = 0;
  const cur = [], xs = [];
  const perDay = [];
  for (let day = 0; day < days; day++) {
    let dd = 0, cc = 0;
    machines.forEach(sp => sp.forEach(([d, u]) => {
      const a = Math.max(d, day * DAY), b = Math.min(u, (day + 1) * DAY);
      if (b > a) dd += b - a;
      if (d >= day * DAY && d < (day + 1) * DAY) cc++;
    }));
    down += dd; count += cc; perDay.push(cc / 5);
    xs.push(day + 1); cur.push(down / (5 * DAY * (day + 1)));
  }
  const rc = $("#aRun"), narrow = narrowOf(rc);
  const ymax = Math.max(0.05, 1.8 * (1 - A), ...cur) * 1.05;
  const fr = frame(rc, narrow ? 1.8 : 3.2, { x: [0, Math.max(days, 2)], y: [0, ymax], xlabel: "days simulated", ylabel: "share of time down", yfmt: v => fmt(v, 2), pad: { l: 52 } });
  plotLine(fr, [0, Math.max(days, 2)], [1 - A, 1 - A], { color: css("--ink"), width: 1.5, dash: [5, 3] });
  plotLine(fr, [0, ...xs], [cur[0], ...cur], { color: FAM.down, width: 2 });
  $("#aMV").textContent = mtbf; $("#aRV").textContent = mttr; $("#aDV").textContent = days;
  caption($("#aCap"), `MTBF ${mtbf} min, MTTR ${mttr} min: [[A = ${mtbf}/(${mtbf} + ${mttr}) = ${fmt(A, 3)}]], so each machine is down [[1 - A = ${fmt(1 - A, 3)}]] of the time (dashed) and breaks down about [[T/(\\text{MTBF} + \\text{MTTR}) = ${fmt(DAY / (mtbf + mttr), 1)}]] times in a day of [[T = ${DAY}]] min. Measured on ${days} simulated day${days > 1 ? "s" : ""} of the five machines: down ${fmt(cur[cur.length - 1], 3)} of the time, ${fmt(count / (5 * days), 2)} breakdowns per machine-day, from ${Math.min(...perDay).toFixed(1)} to ${Math.max(...perDay).toFixed(1)} on single days (averaged over the five machines). The red curve wanders on a day or two and settles on the dashed line as days accumulate: the renewal-reward theorem is a statement about the long run.`);
}
$("#aM").addEventListener("input", () => { ast.mtbf = +$("#aM").value; drawAvail(); });
$("#aR").addEventListener("input", () => { ast.mttr = +$("#aR").value; drawAvail(); });
$("#aD").addEventListener("input", () => { ast.days = +$("#aD").value; drawAvail(); });

reset();
/* open on the breakdown of the worked example (minute 25, the fifth event); Reset goes back to minute 0 */
for (let i = 0; i < 5; i++) st.sim.step();
st.sim.note += " (The page opens here; <b>Reset</b> goes back to minute 0.)";
drawHero();
drawPoisson();
drawAvail();
onRedraw(() => { drawHero(); drawPoisson(); drawAvail(); });
})();
