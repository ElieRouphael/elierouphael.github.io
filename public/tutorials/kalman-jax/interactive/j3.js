/* Kalman filters with JAX, chapter J3, Automatic differentiation from scratch: the live demos.
   1. The graph of f(x1, x2) = log x1 + x1 x2 - sin x2, differentiated one node at a time in forward mode
      (tangents) or reverse mode (a forward sweep, then adjoints), computed here in float64.
   2. Finite differences of a Kalman filter's likelihood against the step size, in float32 and float64,
      from JAX (j3-data.js).
   3. The filter's negative log-likelihood over its two noise levels, computed here on chapter 02's
      measurements, with its gradient from forward-mode dual numbers implemented in this file and checked
      against jax.grad; gradient descent from any point.
   Needs ../kit/kit.js and j3-data.js. */
(() => {
"use strict";
const { $, $$, css, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const D = window.J3;
const SVGNS = "http://www.w3.org/2000/svg";

const num = (v, p = 5) => {
  if (!isFinite(v)) return String(v);
  const s = Math.abs(v) >= 1e4 ? v.toFixed(1) : String(Number(v.toPrecision(p)));
  return s.replace("-", "−");
};

/* ============================================================ 1. the graph */
const NODES = [
  { id: "vm1", name: "v₋₁", op: "x₁", x: 75, y: 85, parents: [] },
  { id: "v0", name: "v₀", op: "x₂", x: 75, y: 235, parents: [] },
  { id: "v1", name: "v₁", op: "log v₋₁", x: 250, y: 45, parents: ["vm1"] },
  { id: "v2", name: "v₂", op: "v₋₁ × v₀", x: 250, y: 160, parents: ["vm1", "v0"] },
  { id: "v3", name: "v₃", op: "sin v₀", x: 250, y: 275, parents: ["v0"] },
  { id: "v4", name: "v₄", op: "v₁ + v₂", x: 420, y: 100, parents: ["v1", "v2"] },
  { id: "v5", name: "v₅ = y", op: "v₄ − v₃", x: 585, y: 185, parents: ["v4", "v3"] },
];
const BY = Object.fromEntries(NODES.map(n => [n.id, n]));
const NM = { vm1: "v₋₁", v0: "v₀", v1: "v₁", v2: "v₂", v3: "v₃", v4: "v₄", v5: "v₅" };
const BW = 130, BH = 62;

function primals(x1, x2) {
  const v = { vm1: x1, v0: x2 };
  v.v1 = Math.log(v.vm1); v.v2 = v.vm1 * v.v0; v.v3 = Math.sin(v.v0); v.v4 = v.v1 + v.v2; v.v5 = v.v4 - v.v3;
  return v;
}
/* local partial derivatives d node / d parent */
function locals(v) {
  return {
    v1: { vm1: 1 / v.vm1 }, v2: { vm1: v.v0, v0: v.vm1 }, v3: { v0: Math.cos(v.v0) },
    v4: { v1: 1, v2: 1 }, v5: { v4: 1, v3: -1 },
  };
}

const g = { mode: "fwd1", k: 0, x1: 2, x2: 5 };
/* The list of steps for the current mode; each step says which node it touches and what it writes. */
function buildSteps() {
  const v = primals(g.x1, g.x2), L = locals(v), steps = [];
  const order = ["vm1", "v0", "v1", "v2", "v3", "v4", "v5"];
  if (g.mode !== "rev") {
    const seed = g.mode === "fwd1" ? { vm1: 1, v0: 0 } : { vm1: 0, v0: 1 };
    const t = {};
    const ft = {
      v1: () => [`v̇₁ = v̇₋₁ / v₋₁ = ${num(t.vm1)} / ${num(v.vm1)}`, t.vm1 / v.vm1],
      v2: () => [`v̇₂ = v̇₋₁·v₀ + v₋₁·v̇₀ = ${num(t.vm1)}·${num(v.v0)} + ${num(v.vm1)}·${num(t.v0)}`, t.vm1 * v.v0 + v.vm1 * t.v0],
      v3: () => [`v̇₃ = v̇₀·cos v₀ = ${num(t.v0)}·${num(Math.cos(v.v0))}`, t.v0 * Math.cos(v.v0)],
      v4: () => [`v̇₄ = v̇₁ + v̇₂ = ${num(t.v1)} + ${num(t.v2)}`, t.v1 + t.v2],
      v5: () => [`v̇₅ = v̇₄ − v̇₃ = ${num(t.v4)} − ${num(t.v3)}`, t.v4 - t.v3],
    };
    for (const id of order) {
      const n = BY[id];
      let text;
      if (!n.parents.length) { t[id] = seed[id]; text = `${n.name} = ${n.op} = ${num(v[id])}     v̇ = ${t[id]} (the seed)`; }
      else { const [s, val] = ft[id](); t[id] = val; text = `${NM[id]} = ${n.op} = ${num(v[id])}     ${s} = ${num(val)}`; }
      steps.push({ node: id, primal: true, tan: { ...t }, text, lit: n.parents.map(p => [p, id]) });
    }
    steps.result = { f: v.v5, d: t.v5 };
  } else {
    for (const id of order) {
      const n = BY[id];
      steps.push({ node: id, primal: true, text: `forward sweep: ${NM[id]} = ${n.op} = ${num(v[id])}, stored`, lit: n.parents.map(p => [p, id]), fwdOnly: true });
    }
    const a = { vm1: 0, v0: 0, v1: 0, v2: 0, v3: 0, v4: 0, v5: 1 };
    steps.push({ node: "v5", adj: { ...a }, text: "backward sweep starts: adj v₅ = ∂y/∂y = 1", lit: [], seed: true });
    for (const id of ["v5", "v4", "v3", "v2", "v1"]) {
      const parts = [];
      for (const p of BY[id].parents) {
        const before = a[p];
        a[p] += a[id] * L[id][p];
        parts.push(`adj ${NM[p]} += adj ${NM[id]} × ${num(L[id][p], 4)} → ${num(a[p])}${before ? ` (adds to ${num(before)})` : ""}`);
      }
      steps.push({ node: id, adj: { ...a }, text: `from ${NM[id]} = ${BY[id].op}:  ${parts.join(";  ")}`, lit: BY[id].parents.map(p => [p, id]), rev: true });
    }
    steps.result = { f: v.v5, d1: a.vm1, d2: a.v0 };
  }
  steps.v = v;
  return steps;
}

function drawGraph() {
  const steps = buildSteps(), v = steps.v, done = steps.slice(0, g.k), cur = done[done.length - 1];
  const svg = $("#graph");
  svg.innerHTML = "";
  const mk = (tag, attrs, parent = svg) => { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
  const defs = mk("defs", {});
  [["ah", "--rule"], ["ahf", "--accent"], ["ahr", "--warm"]].forEach(([id, c]) => {
    const m = mk("marker", { id, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" }, defs);
    mk("path", { d: "M0,0 L10,5 L0,10 z", fill: css(c) }, m);
  });
  const lit = new Set((cur ? cur.lit : []).map(([p, c]) => p + ">" + c));
  for (const n of NODES) for (const p of n.parents) {
    const a = BY[p], key = p + ">" + n.id, on = lit.has(key);
    const rev = on && cur.rev;
    const x0 = a.x + BW / 2, y0 = a.y, x1 = n.x - BW / 2, y1 = n.y;
    const cls = on ? (rev ? "edge lit-rev" : "edge lit-fwd") : "edge";
    const d = `M${x0},${y0} C${(x0 + x1) / 2},${y0} ${(x0 + x1) / 2},${y1} ${x1},${y1}`;
    mk("path", { d, class: cls, "marker-end": rev ? "" : `url(#${on ? "ahf" : "ah"})`, "marker-start": rev ? "url(#ahr)" : "" });
  }
  const seenPrimal = new Set(done.filter(s => s.primal).map(s => s.node));
  const lastTan = [...done].reverse().find(s => s.tan);
  const lastAdj = [...done].reverse().find(s => s.adj);
  for (const n of NODES) {
    const gEl = mk("g", { class: `node${cur && cur.node === n.id ? " now" : seenPrimal.has(n.id) ? " done" : ""}`, transform: `translate(${n.x - BW / 2},${n.y - BH / 2})` });
    mk("rect", { width: BW, height: BH, rx: 9 }, gEl);
    const t1 = mk("text", { x: 8, y: 16, class: "nm" }, gEl); t1.textContent = `${n.name} = ${n.op}`;
    if (n.parents.length === 0) t1.textContent = `${n.name} = ${n.op}`;
    const t2 = mk("text", { x: 8, y: 36, class: "pv" }, gEl); t2.textContent = seenPrimal.has(n.id) ? num(v[n.id]) : "·";
    let dv = "";
    if (lastTan && n.id in lastTan.tan) dv = `v̇ = ${num(lastTan.tan[n.id])}`;
    if (lastAdj) dv = `adj = ${num(lastAdj.adj[n.id])}`;
    const t3 = mk("text", { x: 8, y: 53, class: `dv ${lastAdj ? "rev" : "fwd"}` }, gEl); t3.textContent = dv;
  }

  $("#trace").textContent = cur ? cur.text : "Nothing computed yet.";
  const fin = g.k >= steps.length, r = steps.result;
  $("#rF").textContent = seenPrimal.has("v5") ? num(v.v5, 8) : "–";
  if (g.mode === "fwd1") { $("#rD1").textContent = fin ? num(r.d, 8) : "–"; $("#rD2").textContent = "needs a second sweep"; }
  else if (g.mode === "fwd2") { $("#rD1").textContent = "needs a second sweep"; $("#rD2").textContent = fin ? num(r.d, 8) : "–"; }
  else { $("#rD1").textContent = fin ? num(r.d1, 8) : "–"; $("#rD2").textContent = fin ? num(r.d2, 8) : "–"; }
  $("#rSweeps").textContent = g.mode === "rev" ? "1 forward + 1 backward" : "2 forward (one per input)";
  $("#stepBtn").disabled = fin; $("#allBtn").disabled = fin;
  $$("#modeSeg [data-mode]").forEach(b => b.setAttribute("aria-pressed", b.dataset.mode === g.mode));
  $("#x1v").textContent = num(g.x1); $("#x2v").textContent = num(g.x2);

  let msg;
  if (g.k === 0) msg = g.mode === "rev"
    ? "<b>Reverse mode.</b> First a forward sweep computes and stores every value; then adjoints flow back from the output, and each node passes its adjoint, times a local derivative, to the nodes it was made from. Press <b>Next operation</b>."
    : `<b>Forward mode, seeded on ${g.mode === "fwd1" ? "x₁" : "x₂"}.</b> Each node gets its value and, in the same step, its tangent: how fast it changes as ${g.mode === "fwd1" ? "x₁" : "x₂"} moves. Press <b>Next operation</b>.`;
  else if (fin && g.mode !== "rev") msg = `<b>One forward sweep, one partial derivative:</b> [[\\partial f/\\partial ${g.mode === "fwd1" ? "x_1" : "x_2"} = ${num(r.d, 8)}]]. The other needs a second sweep seeded on the other input. A function of a million inputs would need a million sweeps.`;
  else if (fin) msg = `<b>One backward sweep, the whole gradient:</b> [[\\partial f/\\partial x_1 = ${num(r.d1, 8)}]] and [[\\partial f/\\partial x_2 = ${num(r.d2, 8)}]]. Note that [[v_0]] received two contributions, one from [[v_2]] and one from [[v_3]], and [[v_{-1}]] two as well: that accumulation is the sum over paths of Part 2.`;
  else if (cur.fwdOnly) msg = "The forward sweep stores every value: the backward sweep will need them for the local derivatives, in reverse order.";
  else if (cur.seed) msg = "The output's adjoint is 1: how much the output changes when the output changes.";
  else if (cur.rev) msg = `${NM[cur.node]} passes its adjoint back along each of its incoming edges, multiplied by the local derivative of its operation.`;
  else msg = BY[cur.node].parents.length ? "The value and the tangent are computed together, from the parents' values and tangents." : "An input. Its tangent is the seed: 1 for the input being differentiated, 0 for the other.";
  caption($("#heroStatus"), msg);
}
$("#stepBtn").addEventListener("click", () => { g.k++; drawGraph(); });
$("#allBtn").addEventListener("click", () => { g.k = buildSteps().length; drawGraph(); });
$("#resetBtn").addEventListener("click", () => { g.k = 0; drawGraph(); });
$$("#modeSeg [data-mode]").forEach(b => b.addEventListener("click", () => { g.mode = b.dataset.mode; g.k = 0; drawGraph(); }));
$("#x1").addEventListener("input", e => { g.x1 = +e.target.value; drawGraph(); });
$("#x2").addEventListener("input", e => { g.x2 = +e.target.value; drawGraph(); });
drawGraph();
onRedraw(drawGraph);

/* ============================================================ 2. finite differences */
const REF = D.k64.grad[1], AD32 = D.k32.grad[1];
const rel = v => Math.abs(v - REF) / Math.abs(REF);
const FLOOR = -10;
const lg = v => Math.max(FLOOR, Math.log10(Math.max(v, 1e-30)));
const FD = [
  { key: "f32f", lab: "float32, forward", src: D.k32.fd.fwd, color: "--warm", dash: [5, 4] },
  { key: "f32c", lab: "float32, central", src: D.k32.fd.cen, color: "--warm" },
  { key: "f64f", lab: "float64, forward", src: D.k64.fd.fwd, color: "--accent", dash: [5, 4] },
  { key: "f64c", lab: "float64, central", src: D.k64.fd.cen, color: "--accent" },
];
$("#fdLegend").innerHTML = FD.map(c => `<span style="color:var(${c.color})"><i class="${c.dash ? "dash" : ""}" style="background:var(${c.color})"></i><span style="color:var(--ink-2)">${c.lab}</span></span>`).join("") +
  `<span><i style="background:var(--c4)"></i>float32 autodiff</span>`;
const sup = e => String(e).replace("-", "⁻").replace(/\d/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
const fd = { k: 3 };
function drawFD() {
  const cv = $("#fdC");
  const fr = frame(cv, narrowOf(cv) ? 1.15 : 2.1, {
    x: [-12.3, -0.7], y: [FLOOR, 1.2], xticks: [-12, -10, -8, -6, -4, -2], yticks: [-10, -8, -6, -4, -2, 0],
    xfmt: v => `10${sup(v)}`, yfmt: v => `10${sup(v)}`, xlabel: "step h", ylabel: "relative error",
  });
  const xs = D.h_exps.map(k => -k);
  for (const c of FD) {
    plotLine(fr, xs, c.src.map(v => lg(rel(v))), { color: css(c.color), width: 2, dash: c.dash });
    fr.ctx.fillStyle = css(c.color);
    c.src.forEach((v, i) => { fr.ctx.beginPath(); fr.ctx.arc(fr.X(xs[i]), fr.Y(lg(rel(v))), 2.6, 0, 7); fr.ctx.fill(); });
  }
  plotLine(fr, [-12.3, -0.7], [lg(rel(AD32)), lg(rel(AD32))], { color: css("--c4"), width: 2 });
  fr.ctx.strokeStyle = alpha("--muted", 0.7); fr.ctx.setLineDash([2, 3]);
  seg(fr.ctx, fr.X(-fd.k), fr.pad.t, fr.X(-fd.k), fr.Y(FLOOR)); fr.ctx.setLineDash([]);

  const i = D.h_exps.indexOf(fd.k);
  const show = (id, v) => { $(id).textContent = `${num(v, 6)}  (rel. error ${rel(v) === 0 ? "0" : rel(v).toExponential(1)})`; };
  $("#hv").textContent = `10${sup(-fd.k)}`;
  $("#fRef").textContent = num(REF, 8);
  show("#fAD", AD32); show("#f32f", D.k32.fd.fwd[i]); show("#f32c", D.k32.fd.cen[i]); show("#f64c", D.k64.fd.cen[i]);
  let msg;
  const c32 = D.k32.fd.cen[i];
  if (fd.k <= 2) msg = `With a step this large the formula itself is the problem: the derivative changes over [[h]], and both precisions agree on the same wrong answer. Central differences, with an error of order [[h^2]], do much better than forward ones, of order [[h]].`;
  else if (c32 === 0) msg = `<b>Float32 returns exactly 0.</b> [[\\theta = \\log 0.5 \\approx -0.69]], and float32 numbers near 0.69 are [[6 \\times 10^{-8}]] apart, so [[\\theta \\pm 10^{${-fd.k}}]] ${fd.k >= 8 ? "rounds back to [[\\theta]] itself" : "moves it by only two floats"}: the two evaluations of the likelihood round to the same number. Float64 still works here, until about [[10^{-10}]].`;
  else if (fd.k <= 6 && rel(c32) > 0.02) msg = `Float32 has collapsed: the two likelihood values differ in their last few bits, and dividing by [[h]] magnifies those rounding errors into the answer. Float64, with 29 more bits, is at its best around here: the central difference agrees with autodiff to ${rel(D.k64.fd.cen[i]).toExponential(0)}.`;
  else msg = `Rounding error is already visible in float32: the central difference is off by ${(100 * rel(c32)).toFixed(2)}%. The best any float32 difference does on this function is about 0.09%, at [[h = 10^{-2}]]. Autodiff in float32 is off by ${rel(AD32).toExponential(1)}.`;
  caption($("#fdCap"), msg);
}
$("#hExp").addEventListener("input", e => { fd.k = -e.target.value; drawFD(); });
drawFD();
onRedraw(drawFD);

/* ============================================================ 3. the likelihood, with forward-mode dual numbers */
const ZS = D.zs, T = ZS.length, DT = 0.1, LOG2PI = Math.log(2 * Math.PI);
const QA = DT ** 3 / 3, QB = DT ** 2 / 2, QC = DT;

/* A dual number with two tangents, one per parameter: [value, d/d theta_a, d/d theta_z]. */
const dc = v => [v, 0, 0];
const dadd = (x, y) => [x[0] + y[0], x[1] + y[1], x[2] + y[2]];
const dsub = (x, y) => [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
const dmul = (x, y) => [x[0] * y[0], x[1] * y[0] + x[0] * y[1], x[2] * y[0] + x[0] * y[2]];
const ddiv = (x, y) => { const q = x[0] / y[0]; return [q, (x[1] - q * y[1]) / y[0], (x[2] - q * y[2]) / y[0]]; };
const dscale = (x, c) => [x[0] * c, x[1] * c, x[2] * c];
const dlog = x => [Math.log(x[0]), x[1] / x[0], x[2] / x[0]];

/* The repository's filter (predict, then update with the Joseph form) and the mean negative log-likelihood. */
function nllDual(ta, tz) {
  const sa = Math.exp(ta), sz = Math.exp(tz);
  const sad = [sa, sa, 0], szd = [sz, 0, sz];
  const sa2 = dmul(sad, sad), r = dmul(szd, szd);
  const q00 = dscale(sa2, QA), q01 = dscale(sa2, QB), q11 = dscale(sa2, QC);
  let x0 = dc(0), x1 = dc(0), p00 = dc(1), p01 = dc(0), p11 = dc(1), sum = dc(0);
  for (let k = 0; k < T; k++) {
    const z = ZS[k];
    // predict: x = F x, P = (F P) F^T + Q
    const xp0 = dadd(x0, dscale(x1, DT)), xp1 = x1;
    const a00 = dadd(p00, dscale(p01, DT)), a01 = dadd(p01, dscale(p11, DT));
    const pp00 = dadd(dadd(a00, dscale(a01, DT)), q00), pp01 = dadd(a01, q01), pp11 = dadd(p11, q11);
    // update
    const v = dsub(dc(z), xp0), S = dadd(pp00, r);
    const K0 = ddiv(pp00, S), K1 = ddiv(pp01, S);
    x0 = dadd(xp0, dmul(K0, v)); x1 = dadd(xp1, dmul(K1, v));
    const a = dsub(dc(1), K0), b = dscale(K1, -1);
    const bp = dadd(dmul(b, pp00), pp01);
    p00 = dadd(dmul(dmul(a, a), pp00), dmul(r, dmul(K0, K0)));
    p01 = dadd(dmul(a, bp), dmul(r, dmul(K0, K1)));
    p11 = dadd(dadd(dmul(bp, b), dadd(dmul(b, pp01), pp11)), dmul(r, dmul(K1, K1)));
    sum = dadd(sum, dscale(dadd(dadd(dc(LOG2PI), dlog(S)), ddiv(dmul(v, v), S)), 0.5));
  }
  return dscale(sum, 1 / T);
}
/* The same with plain numbers, for drawing the surface. */
function nllPlain(ta, tz) {
  const sa2 = Math.exp(2 * ta), r = Math.exp(2 * tz);
  const q00 = sa2 * QA, q01 = sa2 * QB, q11 = sa2 * QC;
  let x0 = 0, x1 = 0, p00 = 1, p01 = 0, p11 = 1, sum = 0;
  for (let k = 0; k < T; k++) {
    const xp0 = x0 + x1 * DT, xp1 = x1;
    const a00 = p00 + p01 * DT, a01 = p01 + p11 * DT;
    const pp00 = a00 + a01 * DT + q00, pp01 = a01 + q01, pp11 = p11 + q11;
    const v = ZS[k] - xp0, S = pp00 + r, K0 = pp00 / S, K1 = pp01 / S;
    x0 = xp0 + K0 * v; x1 = xp1 + K1 * v;
    const a = 1 - K0, b = -K1, bp = b * pp00 + pp01;
    p00 = a * a * pp00 + r * K0 * K0; p01 = a * bp + r * K0 * K1; p11 = bp * b + b * pp01 + pp11 + r * K1 * K1;
    sum += 0.5 * (LOG2PI + Math.log(S) + v * v / S);
  }
  return sum / T;
}

/* checks against JAX's float64 grad */
let maxRelL = 0, maxRelG = 0;
for (const t of D.k64.tests) {
  const r = nllDual(t.theta[0], t.theta[1]);
  maxRelL = Math.max(maxRelL, Math.abs(r[0] - t.nll) / Math.abs(t.nll));
  maxRelG = Math.max(maxRelG, Math.hypot(r[1] - t.grad[0], r[2] - t.grad[1]) / Math.hypot(t.grad[0], t.grad[1]));
}
const MLE = [Math.log(0.06106381), Math.log(0.5127593)];

const XR = [Math.log(0.01), Math.log(1.5)], YR = [Math.log(0.2), Math.log(1.5)];
const NX = 64, NY = 40;
let grid = null, gmin = Infinity, gmax = -Infinity;
function buildGrid() {
  grid = new Float64Array(NX * NY);
  for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
    const ta = XR[0] + (i + 0.5) / NX * (XR[1] - XR[0]), tz = YR[0] + (j + 0.5) / NY * (YR[1] - YR[0]);
    const v = nllPlain(ta, tz);
    grid[j * NX + i] = v; gmin = Math.min(gmin, v); gmax = Math.max(gmax, v);
  }
}

const ds = { start: [0, 0], path: [[0, 0]], theta: [0, 0], k: 0, last: null, running: false, fromOrigin: true };
function evalAt() { ds.last = nllDual(ds.theta[0], ds.theta[1]); }
function stepOnce() {
  const r = ds.last || nllDual(ds.theta[0], ds.theta[1]);
  ds.theta = [ds.theta[0] - 0.5 * r[1], ds.theta[1] - 0.5 * r[2]];
  ds.theta = [Math.min(XR[1] + 2, Math.max(XR[0] - 2, ds.theta[0])), Math.min(YR[1] + 2, Math.max(YR[0] - 2, ds.theta[1]))];
  ds.k++;
  ds.path.push(ds.theta.slice());
  evalAt();
}
const sigTicks = [0.01, 0.03, 0.1, 0.3, 1];
let fr3 = null;
function drawDesc(msg) {
  if (!grid) buildGrid();
  if (!ds.last) evalAt();
  const cv = $("#descC");
  fr3 = frame(cv, narrowOf(cv) ? 1.05 : 1.75, {
    x: XR, y: YR, xticks: sigTicks.map(Math.log), yticks: [0.2, 0.3, 0.5, 0.8, 1.2].map(Math.log),
    xfmt: v => String(+Math.exp(v).toPrecision(2)), yfmt: v => String(+Math.exp(v).toPrecision(2)),
    xlabel: "process noise σa (log scale)", ylabel: "measurement noise σz", ygrid: false,
  });
  const { ctx, X, Y } = fr3;
  // the surface as one small image, stretched with smoothing: opaque pixels, so no seams between cells
  const lo = Math.log(1e-3), hi = Math.log(gmax - gmin + 1e-3);
  const hex = c => { const v = parseInt(css(c).slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; };
  const A = hex("--accent"), B = hex("--surface");
  const img = new ImageData(NX, NY);
  for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
    const t = (Math.log(grid[j * NX + i] - gmin + 1e-3) - lo) / (hi - lo);
    const w = 0.06 + 0.8 * (1 - t) ** 1.4, o = ((NY - 1 - j) * NX + i) * 4;
    for (let c = 0; c < 3; c++) img.data[o + c] = Math.round(B[c] + w * (A[c] - B[c]));
    img.data[o + 3] = 255;
  }
  const off = document.createElement("canvas"); off.width = NX; off.height = NY;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, X(XR[0]), Y(YR[1]), X(XR[1]) - X(XR[0]), Y(YR[0]) - Y(YR[1]));
  // path
  ctx.save(); ctx.beginPath(); ctx.rect(fr3.pad.l, fr3.pad.t, fr3.iw, fr3.ih); ctx.clip();
  ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2; ctx.beginPath();
  ds.path.forEach((p, i) => { const px = X(p[0]), py = Y(p[1]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
  ctx.stroke();
  // start, truth, minimum, current
  ctx.fillStyle = css("--surface"); ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(X(ds.start[0]), Y(ds.start[1]), 5, 0, 7); ctx.fill(); ctx.stroke();
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2.2;
  const tx = X(Math.log(0.1)), ty = Y(Math.log(0.5));
  seg(ctx, tx - 6, ty - 6, tx + 6, ty + 6); seg(ctx, tx - 6, ty + 6, tx + 6, ty - 6);
  ctx.strokeStyle = css("--c4"); ctx.lineWidth = 2.4;
  ctx.beginPath(); ctx.arc(X(MLE[0]), Y(MLE[1]), 7, 0, 7); ctx.stroke();
  ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(ds.theta[0]), Y(ds.theta[1]), 4.5, 0, 7); ctx.fill();
  ctx.restore();

  const r = ds.last;
  $("#dK").textContent = String(ds.k);
  $("#dA").textContent = num(Math.exp(ds.theta[0]), 4);
  $("#dZ").textContent = num(Math.exp(ds.theta[1]), 4);
  $("#dL").textContent = num(r[0], 7);
  $("#dG").textContent = `(${num(r[1], 3)}, ${num(r[2], 3)})`;
  $("#dCheck").textContent = `value ${maxRelL.toExponential(0)}, gradient ${maxRelG.toExponential(0)} (relative, at 4 points)`;
  $("#descBtn").disabled = ds.running; $("#desc10Btn").disabled = ds.running;
  caption($("#descCap"), msg || defaultDescMsg());
}
function defaultDescMsg() {
  const sa = Math.exp(ds.theta[0]), sz = Math.exp(ds.theta[1]);
  if (ds.k === 0) return `Starting at [[\\sigma_a = ${num(sa, 3)},\\ \\sigma_z = ${num(sz, 3)}]]. Each step evaluates the filter over all ${T} measurements with dual numbers, which gives the likelihood and both partial derivatives in one forward sweep, then moves half a gradient downhill. Click the map to start elsewhere. ✕ marks the noise levels that generated the data; the green ring, the likelihood's minimum.`;
  const jaxEnd = D.k64.path[D.k64.path.length - 1];
  let s = `After ${ds.k} steps: [[\\sigma_a = ${num(sa, 3)},\\ \\sigma_z = ${num(sz, 3)}]]. `;
  if (ds.fromOrigin && ds.k === 400) s += `JAX's float64 descent from the same start, with <code>jax.grad</code>, is at [[\\sigma_a = ${num(Math.exp(jaxEnd[0]), 4)},\\ \\sigma_z = ${num(Math.exp(jaxEnd[1]), 4)}]] after 400 steps; this page's dual numbers agree. `;
  s += `The descent drops quickly to the valley floor, where [[\\sigma_z]] is settled, then creeps along it towards the minimum at [[\\sigma_a = 0.061]]: the valley is about 89 times flatter along [[\\sigma_a]] than across it.`;
  return s;
}
function run(n) {
  ds.running = true;
  let left = n;
  const tick = () => {
    for (let i = 0; i < 8 && left > 0; i++, left--) stepOnce();
    drawDesc();
    if (left > 0) setTimeout(tick, 16); else { ds.running = false; drawDesc(); }
  };
  tick();
}
function resetTo(theta, fromOrigin) {
  ds.start = theta.slice(); ds.theta = theta.slice(); ds.path = [theta.slice()]; ds.k = 0; ds.last = null; ds.fromOrigin = fromOrigin;
  drawDesc();
}
$("#descBtn").addEventListener("click", () => run(400));
$("#desc10Btn").addEventListener("click", () => run(10));
$("#descReset").addEventListener("click", () => resetTo([0, 0], true));
$("#descC").addEventListener("click", e => {
  if (ds.running || !fr3) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const px = e.clientX - rect.left, py = e.clientY - rect.top;
  const ta = XR[0] + (px - fr3.pad.l) / fr3.iw * (XR[1] - XR[0]);
  const tz = YR[1] - (py - fr3.pad.t) / fr3.ih * (YR[1] - YR[0]);
  if (ta < XR[0] || ta > XR[1] || tz < YR[0] || tz > YR[1]) return;
  resetTo([ta, tz], false);
});
$("#descLegend").innerHTML = `<span><i style="background:var(--warm)"></i>gradient-descent path</span><span>✕ the generating levels (0.1, 0.5)</span><span><b style="color:var(--c4)">○</b> the minimum (0.061, 0.513)</span>`;
/* the surface takes a moment to compute: draw it after the rest of the page */
setTimeout(() => { drawDesc(); onRedraw(() => drawDesc()); }, 200);
})();
