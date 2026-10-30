/* Kalman filters with JAX, chapter J6, keys, trees and a training loop: the live demos.
   1. JAX's default random generator, Threefry-2x32 in its partitionable mode, implemented here: PRNGKey,
      split, fold_in, bits and uniform, checked bit for bit against JAX's outputs for seed 42 (j6-data.js).
   2. The tables: the operations for PRNGKey(42), the parameter tree, the optimisers' results.
   3. Four optimisers (gradient descent, momentum, two Adams, as in Optax) on J3's likelihood of chapter 02's
      noise levels, with the gradient from forward-mode dual numbers computed in this page.
   Needs ../kit/kit.js and j6-data.js. */
(() => {
"use strict";
const { $, $$, css, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const D = window.J6;
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/* ============================================================ 1. Threefry-2x32 */
function threefry2x32(k1, k2, c1, c2) {             // all values are uint32
  const ks = [k1 >>> 0, k2 >>> 0, (k1 ^ k2 ^ 0x1BD11BDA) >>> 0];
  const rot = [[13, 15, 26, 6], [17, 29, 16, 24]];
  let x0 = (c1 + ks[0]) >>> 0, x1 = (c2 + ks[1]) >>> 0;
  for (let i = 0; i < 5; i++) {                      // 5 blocks of 4 rounds
    for (const r of rot[i % 2]) {
      x0 = (x0 + x1) >>> 0;                         // add
      x1 = ((x1 << r) | (x1 >>> (32 - r))) >>> 0;   // rotate
      x1 = (x1 ^ x0) >>> 0;                         // exclusive-or
    }
    x0 = (x0 + ks[(i + 1) % 3]) >>> 0;              // inject the key, and the block number
    x1 = (x1 + ks[(i + 2) % 3] + i + 1) >>> 0;
  }
  return [x0, x1];
}
/* In JAX's default 32-bit mode the seed is first converted to 32 bits, so the high word is always 0. */
const PRNGKey = seed => [0, seed >>> 0];
const split = (key, n = 2) => Array.from({ length: n }, (_, i) => threefry2x32(key[0], key[1], 0, i));
const foldIn = (key, d) => threefry2x32(key[0], key[1], 0, d >>> 0);
const bits = (key, n) => Array.from({ length: n }, (_, i) => { const [a, b] = threefry2x32(key[0], key[1], 0, i); return (a ^ b) >>> 0; });
const uniform = (key, n) => bits(key, n).map(b => (b >>> 9) / 2 ** 23);   // float32 1.m - 1, exactly

/* checks against JAX, seed 42 */
const K = D.keys;
const flat = a => a.flat();
const eq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const k42 = PRNGKey(42);
const CHECKS = [
  ["PRNGKey(42)", eq(k42, K.key42)],
  ["split(key)", eq(flat(split(k42, 2)), K.split2)],
  ["split(key, 3)", eq(flat(split(k42, 3)), K.split3)],
  ["fold_in(key, 7)", eq(foldIn(k42, 7), K.fold7)],
  ["bits(key, (4,))", eq(bits(k42, 4), K.bits4)],
  ["uniform(key, (4,))", eq(uniform(k42, 4), K.uniform4)],
  ["split(split(key)[0])", eq(flat(split(split(k42, 2)[0], 2)), K.split_of_split)],
  ["uniform(split(key)[0], (4,))", eq(uniform(split(k42, 2)[0], 4), K.uniform_child0)],
  ["PRNGKey(2**32 + 5), which is PRNGKey(5)", eq(PRNGKey(2 ** 32 + 5), K.seed_big) && eq(PRNGKey(5), K.seed_big)],
];
const nOk = CHECKS.filter(c => c[1]).length;

const hex = w => w.toString(16).padStart(8, "0");
const kt = { nodes: [], sel: 0, seed: 42 };
function resetTree() {
  kt.nodes = [{ id: 0, path: `PRNGKey(${kt.seed})`, key: PRNGKey(kt.seed), depth: 0, draws: 0 }];
  kt.sel = 0;
}
function addChild(parent, path, key) {
  const node = { id: kt.nodes.length, path, key, depth: parent.depth + 1, draws: 0 };
  const at = kt.nodes.indexOf(parent) + 1 + kt.nodes.filter(n => n.parentId === parent.id).length;
  node.parentId = parent.id;
  kt.nodes.splice(at, 0, node);
  return node;
}
function drawTree(msg) {
  $("#kTree").innerHTML = kt.nodes.map(n => {
    const u = n.draws ? uniform(n.key, 4).map(v => v.toFixed(7)).join(", ") : "";
    return `<button type="button" class="knode" data-id="${n.id}" aria-pressed="${n.id === kt.sel}" style="margin-left:${Math.min(n.depth, 6) * 1.1}rem">` +
      `<span class="path">${esc(n.path)}</span><span class="w">[${n.key[0]}, ${n.key[1]}]</span><span class="w" style="color:var(--muted)">0x${hex(n.key[0])} 0x${hex(n.key[1])}</span>` +
      (n.draws ? `<span class="draws">uniform → [${u}]</span>` : "") +
      (n.draws > 1 ? `<span class="warn">drawn ${n.draws} times: the same four numbers every time (rule 1)</span>` : "") + `</button>`;
  }).join("");
  $$("#kTree .knode").forEach(b => b.addEventListener("click", () => { kt.sel = +b.dataset.id; drawTree(); }));
  const s = kt.nodes.find(n => n.id === kt.sel);
  $("#rKey").textContent = `[${s.key[0]}, ${s.key[1]}] = 0x${hex(s.key[0])} 0x${hex(s.key[1])}`;
  $("#rCheck").textContent = `${nOk} of ${CHECKS.length} checks match JAX bit for bit` + (nOk < CHECKS.length ? `; failing: ${CHECKS.filter(c => !c[1]).map(c => c[0]).join(", ")}` : "");
  caption($("#heroStatus"), msg || `<code>${esc(s.path)}</code> is selected. A key is just these two numbers; every random function hashes them with a counter. <b>split</b> makes two children, <b>fold_in</b> one child for a given number, and <b>uniform</b> draws four numbers without changing the key.`);
}
function sel() { return kt.nodes.find(n => n.id === kt.sel); }
$("#splitBtn").addEventListener("click", () => {
  const p = sel();
  const [a, b] = split(p.key, 2);
  const c0 = addChild(p, `${p.path} → split[0]`, a);
  addChild(p, `${p.path} → split[1]`, b);
  kt.sel = c0.id;
  drawTree(`<b>split</b>: the two children are <code>threefry(key, (0, 0))</code> and <code>threefry(key, (0, 1))</code>, hashes of the counters 0 and 1. They look unrelated to each other and to the parent, and asking for three children would give these same two first.`);
});
$("#foldIn").addEventListener("input", e => e.target.setAttribute("aria-invalid", "false"));
$("#foldBtn").addEventListener("click", () => {
  const p = sel();
  const raw = $("#foldIn").value.trim();
  if (!/^\d{1,10}$/.test(raw) || +raw > 2 ** 32 - 1) {             // fold_in takes a uint32 here
    $("#foldIn").setAttribute("aria-invalid", "true");
    caption($("#heroStatus"), "<b>fold_in needs a whole number</b> from 0 to 4 294 967 295 (a uint32), such as a step number.");
    return;
  }
  const d = +raw;
  const c = addChild(p, `${p.path} → fold_in(${d})`, foldIn(p.key, d));
  kt.sel = c.id;
  drawTree(`<b>fold_in(key, ${d})</b> hashes the number ${d} under the key: <code>threefry(key, (0, ${d}))</code>. ${d < 16 ? `It is the same key as <code>split(key, n)[${d}]</code> for any [[n > ${d}]]; split the parent and compare.` : "Use it to derive a key for step k of a loop without carrying keys around."}`);
});
$("#drawBtn").addEventListener("click", () => {
  const p = sel();
  p.draws++;
  drawTree(p.draws > 1
    ? `<b>The same key again, the same numbers.</b> Nothing in the key changed when it was used, so drawing from it again repeats the draw exactly. In a simulation, two noise sources drawn from one key are one noise source counted twice: split first.`
    : `<b>uniform(key, (4,))</b>: four counters 0 to 3, hashed; each 32-bit result's top 23 bits become the mantissa of a float32 in [1, 2), minus 1. ${p.path === `PRNGKey(42)` ? "These are JAX's numbers exactly: 0.48870957, 0.6797972, 0.6162715, 0.5610161." : ""}`);
});
$("#resetBtn").addEventListener("click", () => { resetTree(); drawTree(); });
$("#seedIn").addEventListener("input", e => {
  const v = e.target.value.trim();
  const ok = /^\d{1,15}$/.test(v) && +v < 2 ** 53;
  e.target.setAttribute("aria-invalid", ok ? "false" : "true");
  if (!ok) return;
  kt.seed = +v; resetTree();
  drawTree(kt.seed >= 2 ** 32
    ? `<b>The seed does not fit in 32 bits.</b> In JAX's default 32-bit mode it is converted to 32 bits first, so only the low word survives: <code>PRNGKey(${kt.seed})</code> is the same key as <code>PRNGKey(${kt.seed % 2 ** 32})</code>, <code>[0, ${kt.seed % 2 ** 32}]</code>, and no warning is given.`
    : undefined);
});
resetTree(); drawTree();

/* ============================================================ 2. tables */
const pairs = a => a.reduce((acc, v, i) => (i % 2 ? acc[acc.length - 1].push(v) : acc.push([v]), acc), []).map(p => `[${p.join(", ")}]`).join(", ");
$("#cSplit").textContent = pairs(K.split2);
$("#cFold").textContent = `fold_in(key, 7) = [${K.fold7.join(", ")}]`;
$("#cBits").textContent = K.bits4.join(", ");
$("#cUnif").textContent = K.uniform4.map(v => v.toFixed(7)).join(", ");
$("#cNorm").textContent = K.normal4.map(v => v.toFixed(7)).join(", ");
const T = D.trees;
$("#treeBody").innerHTML = T.shapes.map((s, i) => `<tr><td class="n">${i + 1}</td><td class="n">(${s[0].join(", ")})</td><td class="n">(${s[1].join(", ")})</td><td class="n">${s[0][0] * s[0][1] + s[1][0]}</td></tr>`).join("") +
  `<tr><td><b>Total</b></td><td colspan="2">${T.n_leaves} arrays</td><td class="n"><b>${T.n_params}</b></td></tr>`;
const OPT_NAMES = { gd: "optax.sgd(0.5): gradient descent", momentum: "optax.sgd(0.5, momentum=0.9)", adam_005: "optax.adam(0.05)", adam_01: "optax.adam(0.1)" };
$("#optBody").innerHTML = ["gd", "momentum", "adam_005", "adam_01"].map(k => {
  const r = D.training[k];
  return `<tr><td><code>${OPT_NAMES[k].split(":")[0]}</code>${OPT_NAMES[k].includes(":") ? "," + OPT_NAMES[k].split(":")[1] : ""}</td><td class="n">${r["steps_to_0.1%"]}</td><td class="n">σa ${r.end[0].toFixed(5)}, σz ${r.end[1].toFixed(5)}</td></tr>`;
}).join("");
$("#tLoop").textContent = `${(D.loop.python_ms / 1000).toFixed(1)} s`;
$("#tScan").textContent = `${(D.loop.scan_ms / 1000).toFixed(1)} s`;

/* ============================================================ 3. the optimiser race on J3's likelihood */
const ZS = D.zs, NT = ZS.length, DT = 0.1, LOG2PI = Math.log(2 * Math.PI);
const QA = DT ** 3 / 3, QB = DT ** 2 / 2, QC = DT;
const dc = v => [v, 0, 0];
const dadd = (x, y) => [x[0] + y[0], x[1] + y[1], x[2] + y[2]];
const dsub = (x, y) => [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
const dmul = (x, y) => [x[0] * y[0], x[1] * y[0] + x[0] * y[1], x[2] * y[0] + x[0] * y[2]];
const ddiv = (x, y) => { const q = x[0] / y[0]; return [q, (x[1] - q * y[1]) / y[0], (x[2] - q * y[2]) / y[0]]; };
const dscale = (x, c) => [x[0] * c, x[1] * c, x[2] * c];
const dlog = x => [Math.log(x[0]), x[1] / x[0], x[2] / x[0]];
/* J3's negative log-likelihood, with both partial derivatives by forward mode */
function nllDual(ta, tz) {
  const sa = Math.exp(ta), sz = Math.exp(tz);
  const sad = [sa, sa, 0], szd = [sz, 0, sz];
  const sa2 = dmul(sad, sad), r = dmul(szd, szd);
  const q00 = dscale(sa2, QA), q01 = dscale(sa2, QB), q11 = dscale(sa2, QC);
  let x0 = dc(0), x1 = dc(0), p00 = dc(1), p01 = dc(0), p11 = dc(1), sum = dc(0);
  for (let k = 0; k < NT; k++) {
    const xp0 = dadd(x0, dscale(x1, DT)), xp1 = x1;
    const a00 = dadd(p00, dscale(p01, DT)), a01 = dadd(p01, dscale(p11, DT));
    const pp00 = dadd(dadd(a00, dscale(a01, DT)), q00), pp01 = dadd(a01, q01), pp11 = dadd(p11, q11);
    const v = dsub(dc(ZS[k]), xp0), S = dadd(pp00, r);
    const K0 = ddiv(pp00, S), K1 = ddiv(pp01, S);
    x0 = dadd(xp0, dmul(K0, v)); x1 = dadd(xp1, dmul(K1, v));
    const a = dsub(dc(1), K0), b = dscale(K1, -1), bp = dadd(dmul(b, pp00), pp01);
    p00 = dadd(dmul(dmul(a, a), pp00), dmul(r, dmul(K0, K0)));
    p01 = dadd(dmul(a, bp), dmul(r, dmul(K0, K1)));
    p11 = dadd(dadd(dmul(bp, b), dadd(dmul(b, pp01), pp11)), dmul(r, dmul(K1, K1)));
    sum = dadd(sum, dscale(dadd(dadd(dc(LOG2PI), dlog(S)), ddiv(dmul(v, v), S)), 0.5));
  }
  return dscale(sum, 1 / NT);
}
/* The same function and gradient, written out with plain numbers: each dual quantity is three variables
   (value, d/d log sigma_a, d/d log sigma_z). About 30 times faster than allocating arrays, for the race. */
function nllFast(ta, tz) {
  const sa2 = Math.exp(2 * ta), r = Math.exp(2 * tz);
  const sa2a = 2 * sa2, rb = 2 * r;                         // d sa2/d ta, d r/d tz
  const q00 = QA * sa2, q00a = QA * sa2a, q01 = QB * sa2, q01a = QB * sa2a, q11 = QC * sa2, q11a = QC * sa2a;
  let x0 = 0, x0a = 0, x0b = 0, x1 = 0, x1a = 0, x1b = 0;
  let p00 = 1, p00a = 0, p00b = 0, p01 = 0, p01a = 0, p01b = 0, p11 = 1, p11a = 0, p11b = 0;
  let s = 0, sA = 0, sB = 0;
  for (let k = 0; k < NT; k++) {
    const xp0 = x0 + DT * x1, xp0a = x0a + DT * x1a, xp0b = x0b + DT * x1b;
    const a00 = p00 + DT * p01, a00a = p00a + DT * p01a, a00b = p00b + DT * p01b;
    const a01 = p01 + DT * p11, a01a = p01a + DT * p11a, a01b = p01b + DT * p11b;
    const pp00 = a00 + DT * a01 + q00, pp00a = a00a + DT * a01a + q00a, pp00b = a00b + DT * a01b;
    const pp01 = a01 + q01, pp01a = a01a + q01a, pp01b = a01b;
    const pp11 = p11 + q11, pp11a = p11a + q11a, pp11b = p11b;
    const v = ZS[k] - xp0, va = -xp0a, vb = -xp0b;
    const S = pp00 + r, Sa = pp00a, Sb = pp00b + rb;
    const K0 = pp00 / S, K0a = (pp00a - K0 * Sa) / S, K0b = (pp00b - K0 * Sb) / S;
    const K1 = pp01 / S, K1a = (pp01a - K1 * Sa) / S, K1b = (pp01b - K1 * Sb) / S;
    x0 = xp0 + K0 * v; x0a = xp0a + K0a * v + K0 * va; x0b = xp0b + K0b * v + K0 * vb;
    x1 = x1 + K1 * v; x1a = x1a + K1a * v + K1 * va; x1b = x1b + K1b * v + K1 * vb;
    const a = 1 - K0, aa = -K0a, ab = -K0b, b = -K1, ba = -K1a, bb = -K1b;
    const bp = b * pp00 + pp01, bpa = ba * pp00 + b * pp00a + pp01a, bpb = bb * pp00 + b * pp00b + pp01b;
    const n00 = a * a * pp00 + r * K0 * K0;
    const n00a = 2 * a * aa * pp00 + a * a * pp00a + 2 * r * K0 * K0a;
    const n00b = 2 * a * ab * pp00 + a * a * pp00b + rb * K0 * K0 + 2 * r * K0 * K0b;
    const n01 = a * bp + r * K0 * K1;
    const n01a = aa * bp + a * bpa + r * (K0a * K1 + K0 * K1a);
    const n01b = ab * bp + a * bpb + rb * K0 * K1 + r * (K0b * K1 + K0 * K1b);
    const n11 = bp * b + b * pp01 + pp11 + r * K1 * K1;
    const n11a = bpa * b + bp * ba + ba * pp01 + b * pp01a + pp11a + 2 * r * K1 * K1a;
    const n11b = bpb * b + bp * bb + bb * pp01 + b * pp01b + pp11b + rb * K1 * K1 + 2 * r * K1 * K1b;
    p00 = n00; p00a = n00a; p00b = n00b; p01 = n01; p01a = n01a; p01b = n01b; p11 = n11; p11a = n11a; p11b = n11b;
    const w = v * v / S;
    s += 0.5 * (LOG2PI + Math.log(S) + w);
    sA += 0.5 * (Sa / S + (2 * v * va - w * Sa) / S);
    sB += 0.5 * (Sb / S + (2 * v * vb - w * Sb) / S);
  }
  return [s / NT, sA / NT, sB / NT];
}
function nllPlain(ta, tz) {
  const sa2 = Math.exp(2 * ta), r = Math.exp(2 * tz);
  const q00 = sa2 * QA, q01 = sa2 * QB, q11 = sa2 * QC;
  let x0 = 0, x1 = 0, p00 = 1, p01 = 0, p11 = 1, sum = 0;
  for (let k = 0; k < NT; k++) {
    const xp0 = x0 + x1 * DT, xp1 = x1;
    const a00 = p00 + p01 * DT, a01 = p01 + p11 * DT;
    const pp00 = a00 + a01 * DT + q00, pp01 = a01 + q01, pp11 = p11 + q11;
    const v = ZS[k] - xp0, S = pp00 + r, K0 = pp00 / S, K1 = pp01 / S;
    x0 = xp0 + K0 * v; x1 = xp1 + K1 * v;
    const a = 1 - K0, b = -K1, bp = b * pp00 + pp01;
    p00 = a * a * pp00 + r * K0 * K0; p01 = a * bp + r * K0 * K1; p11 = bp * b + b * pp01 + pp11 + r * K1 * K1;
    sum += 0.5 * (LOG2PI + Math.log(S) + v * v / S);
  }
  return sum / NT;
}
const MLE = [Math.log(D.mle[0]), Math.log(D.mle[1])];
const FAST_CHECK = [[0, 0], [-2, -0.5], [-3, -1]].reduce((m, t) => { const f = nllFast(t[0], t[1]), d = nllDual(t[0], t[1]); return Math.max(m, ...[0, 1, 2].map(i => Math.abs(f[i] - d[i]) / Math.max(1e-12, Math.abs(d[i])))); }, 0);
const NLL_MIN = nllPlain(MLE[0], MLE[1]);

/* the four optimisers, as Optax defines them */
const OPTS = [
  { key: "gd", label: "gradient descent, 0.5", color: "--warm", make: () => ({ step: (th, g) => [th[0] - 0.5 * g[0], th[1] - 0.5 * g[1]] }) },
  { key: "momentum", label: "momentum 0.9, 0.5", color: "--c5", make: () => { let m = [0, 0]; return { step: (th, g) => { m = [g[0] + 0.9 * m[0], g[1] + 0.9 * m[1]]; return [th[0] - 0.5 * m[0], th[1] - 0.5 * m[1]]; } }; } },
  { key: "adam_005", label: "Adam, 0.05", color: "--accent", make: () => adam(0.05) },
  { key: "adam_01", label: "Adam, 0.1", color: "--c4", make: () => adam(0.1) },
];
function adam(lr) {
  let m = [0, 0], v = [0, 0], t = 0;
  return { step: (th, g) => {
    t++;
    m = m.map((mi, i) => 0.9 * mi + 0.1 * g[i]);
    v = v.map((vi, i) => 0.999 * vi + 0.001 * g[i] * g[i]);
    const c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
    return th.map((x, i) => x - lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8));
  } };
}
const XR = [Math.log(0.01), Math.log(1.5)], YR = [Math.log(0.2), Math.log(1.5)];
const NX = 64, NY = 40;
let grid = null, gmin = Infinity, gmax = -Infinity;
function buildGrid() {
  grid = new Float64Array(NX * NY);
  for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
    const v = nllPlain(XR[0] + (i + 0.5) / NX * (XR[1] - XR[0]), YR[0] + (j + 0.5) / NY * (YR[1] - YR[0]));
    grid[j * NX + i] = v; gmin = Math.min(gmin, v); gmax = Math.max(gmax, v);
  }
}
const N_STEPS = 1000;
const race = { start: [0, 0], runs: null, k: 0, running: false, fromOrigin: true };
function resetRace(start, fromOrigin) {
  race.start = start.slice(); race.k = 0; race.fromOrigin = fromOrigin;
  race.runs = OPTS.map(o => ({ o, opt: o.make(), th: start.slice(), path: [start.slice()], loss: [], done: null }));
}
const within = th => Math.abs(Math.exp(th[0]) / D.mle[0] - 1) < 1e-3 && Math.abs(Math.exp(th[1]) / D.mle[1] - 1) < 1e-3;
function stepAll() {
  for (const r of race.runs) {
    const d = nllFast(r.th[0], r.th[1]);
    r.loss.push(d[0]);
    r.th = r.opt.step(r.th, [d[1], d[2]]);
    r.path.push(r.th.slice());
  }
  race.k++;
}
function stepsTo(path) {
  let last = -1;
  path.forEach((th, i) => { if (!within(th)) last = i; });
  return last + 1 < path.length ? last + 1 : null;
}
let frMap = null;
function drawRace(msg) {
  if (!grid) buildGrid();
  const cv = $("#raceC");
  frMap = frame(cv, narrowOf(cv) ? 1.05 : 1.9, {
    x: XR, y: YR, xticks: [0.01, 0.03, 0.1, 0.3, 1].map(Math.log), yticks: [0.2, 0.3, 0.5, 0.8, 1.2].map(Math.log),
    xfmt: v => String(+Math.exp(v).toPrecision(2)), yfmt: v => String(+Math.exp(v).toPrecision(2)),
    xlabel: "process noise σa (log scale)", ylabel: "measurement noise σz", ygrid: false,
  });
  const { ctx, X, Y } = frMap;
  const lo = Math.log(1e-3), hi = Math.log(gmax - gmin + 1e-3);
  const hexc = c => { const v = parseInt(css(c).slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; };
  const A = hexc("--accent"), B = hexc("--surface");
  const img = new ImageData(NX, NY);
  for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
    const t = (Math.log(grid[j * NX + i] - gmin + 1e-3) - lo) / (hi - lo), w = 0.06 + 0.6 * (1 - t) ** 1.4, o = ((NY - 1 - j) * NX + i) * 4;
    for (let c = 0; c < 3; c++) img.data[o + c] = Math.round(B[c] + w * (A[c] - B[c]));
    img.data[o + 3] = 255;
  }
  const off = document.createElement("canvas"); off.width = NX; off.height = NY;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, X(XR[0]), Y(YR[1]), X(XR[1]) - X(XR[0]), Y(YR[0]) - Y(YR[1]));
  ctx.save(); ctx.beginPath(); ctx.rect(frMap.pad.l, frMap.pad.t, frMap.iw, frMap.ih); ctx.clip();
  for (const r of race.runs) {
    ctx.strokeStyle = css(r.o.color); ctx.lineWidth = 2; ctx.beginPath();
    r.path.forEach((p, i) => { i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])); });
    ctx.stroke();
    const e = r.path[r.path.length - 1];
    ctx.fillStyle = css(r.o.color); ctx.beginPath(); ctx.arc(X(e[0]), Y(e[1]), 3.5, 0, 7); ctx.fill();
  }
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(MLE[0]), Y(MLE[1]), 7, 0, 7); ctx.stroke();
  ctx.fillStyle = css("--surface"); ctx.strokeStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(race.start[0]), Y(race.start[1]), 5, 0, 7); ctx.fill(); ctx.stroke();
  ctx.restore();

  const lc = $("#lossC");
  const fl = frame(lc, narrowOf(lc) ? 1.6 : 3.4, {
    x: [0, N_STEPS], y: [-10, 0], yticks: [-10, -8, -6, -4, -2, 0], yfmt: v => `1e${v}`, xlabel: "step", ylabel: "loss above the minimum",
  });
  for (const r of race.runs) {
    if (r.loss.length < 2) continue;
    plotLine(fl, r.loss.map((_, i) => i), r.loss.map(v => Math.max(-10, Math.log10(Math.max(v - NLL_MIN, 1e-12)))), { color: css(r.o.color), width: 1.8 });
  }
  $("#raceReadouts").innerHTML = race.runs.map(r => {
    const s = stepsTo(r.path);
    const jax = race.fromOrigin ? ` (JAX: ${D.training[r.o.key]["steps_to_0.1%"]})` : "";
    return `<span style="color:var(${r.o.color})">${r.o.label}</span><span class="v">${race.k === 0 ? "–" : s === null ? `not within 0.1% after ${race.k}` : `within 0.1% from step ${s}`}${race.k >= N_STEPS ? jax : ""}</span>`;
  }).join("");
  $("#raceBtn").disabled = race.running || race.k >= N_STEPS;
  caption($("#raceCap"), msg || defaultRaceMsg());
}
function defaultRaceMsg() {
  if (race.k === 0) return `Four optimisers start together from [[\\sigma_a = ${Math.exp(race.start[0]).toFixed(3)},\\ \\sigma_z = ${Math.exp(race.start[1]).toFixed(3)}]]. Each step computes the likelihood over all 1000 measurements and its gradient with dual numbers, as in J3, then applies the optimiser's rule exactly as Optax defines it. The ring marks the minimum. Click the map to choose another start.`;
  if (race.k < N_STEPS) return `Step ${race.k}…`;
  let s = "All four reach the valley floor quickly; the difference is the crawl along it towards the ring. Gradient descent takes the smallest steps where the valley is shallow; momentum builds up speed along it; Adam rescales each coordinate so the shallow direction gets full-sized steps, and then overshoots a little before settling. ";
  if (race.fromOrigin) s += "From this start the page's step counts can be compared with Optax's, in brackets; any difference would come from float64 here against float32 in JAX.";
  s += ` (The race uses the dual-number gradient written out with plain numbers for speed; it agrees with J3's dual-number code to ${FAST_CHECK.toExponential(0)}.)`;
  return s;
}
function run() {
  race.running = true;
  const tick = () => {
    for (let i = 0; i < 25 && race.k < N_STEPS; i++) stepAll();
    drawRace();
    if (race.k < N_STEPS) setTimeout(tick, 0); else { race.running = false; drawRace(); }
  };
  tick();
}
$("#raceLegend").innerHTML = OPTS.map(o => `<span><i style="background:var(${o.color})"></i>${o.label}</span>`).join("") + `<span>○ the minimum (0.061, 0.513)</span>`;
$("#raceBtn").addEventListener("click", run);
$("#raceReset").addEventListener("click", () => { if (race.running) return; resetRace([0, 0], true); drawRace(); });
$("#raceC").addEventListener("click", e => {
  if (race.running || !frMap) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const ta = XR[0] + (e.clientX - rect.left - frMap.pad.l) / frMap.iw * (XR[1] - XR[0]);
  const tz = YR[1] - (e.clientY - rect.top - frMap.pad.t) / frMap.ih * (YR[1] - YR[0]);
  if (ta < XR[0] || ta > XR[1] || tz < YR[0] || tz > YR[1]) return;
  resetRace([ta, tz], false); drawRace();
});
resetRace([0, 0], true);
setTimeout(() => { drawRace(); onRedraw(() => drawRace()); }, 200);
})();
