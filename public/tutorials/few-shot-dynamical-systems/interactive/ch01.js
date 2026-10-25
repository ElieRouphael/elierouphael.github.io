/* Learning to Identify New Dynamical Systems, chapter 01, The system and what we can measure:
   the live demos. The true plant of the course (nonlinear oscillator, saturating delayed actuator,
   quintic term, force bias, coloured disturbance) is simulated here exactly as dynutils.simulate
   does it; the five demo members, the demo input and the one recorded experiment come from the
   notebook's own seeded run (ch01-data.js), since NumPy's random stream is not reproduced in the
   browser. The repeat-run envelope uses its own random generator. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const DT = 0.1, X0 = [0.2, 0.0];
const ETA0 = [1.20, 0.32, 0.90, 1.00];
const M = [[0.55, 0.12], [0.07, 0.11], [0.16, -0.09], [0.30, 0.45]];
const SIGMA_Y = 0.01, SIGMA_W = 0.01, RHO = 0.8;
const actuator = u => 0.3 * Math.tanh(u / 0.6);
/* Coefficients of the member with design variables c and no individual deviation. */
const etaOf = c => ETA0.map((e, i) => e + M[i][0] * c[0] + M[i][1] * c[1]);
/* A nominal member of the family at c: typical quintic coefficient, bias as the simulator's mean. */
const memberOf = c => ({ eta: etaOf(c), context: c.slice(), gamma: 0.041, bias: 0.006 * c[0] });
/* The true plant, dynutils.simulate: semi-implicit Euler, one-sample input delay. `noise`, when
   given, is a function returning standard normal draws; the disturbance is AR(1) with colour RHO.
   Returns { q: true positions, y: measurements }, both of length u.length + 1. */
function simulate(sys, u, noise) {
  const [k, d, b, a] = sys.eta;
  let q = X0[0], v = X0[1], w = 0;
  const qs = [q];
  for (let t = 0; t < u.length; t++) {
    const ut = t === 0 ? 0 : u[t - 1];
    if (noise) w = RHO * w + SIGMA_W * noise();
    const acc = -k * q - d * v + b * actuator(ut) - a * q * q * q - sys.gamma * q ** 5 + sys.bias + w;
    v += DT * acc;
    q += DT * v;
    qs.push(q);
  }
  const y = noise ? qs.map(x => x + SIGMA_Y * noise()) : qs.slice();
  return { q: qs, y };
}
/* The learner's model, dynutils.predict: the same recursion without quintic term, bias and disturbance. */
function predict(eta, u) {
  const [k, d, b, a] = eta;
  let q = X0[0], v = X0[1];
  const out = [q];
  for (let t = 0; t < u.length; t++) {
    const f = actuator(t === 0 ? 0 : u[t - 1]);
    v += DT * (-k * q - d * v + b * f - a * q * q * q);
    q += DT * v;
    out.push(q);
  }
  return out;
}
const rmse = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);
/* Small-signal gain |q/u| of the linearised member at angular frequency om: the actuator slope
   at zero is 0.3/0.6 = 0.5, and the cubic term vanishes to first order. */
const freqResp = (eta, om) => 0.5 * eta[2] / Math.hypot(eta[0] - om * om, eta[1] * om);
const natural = eta => ({ wn: Math.sqrt(eta[0]), f: Math.sqrt(eta[0]) / (2 * Math.PI), zeta: eta[1] / (2 * Math.sqrt(eta[0])) });
/* Least-squares fit y = p0 c1 + p1 c2 + p2 over samples (cs[i], ys[i]): the exercise-1 regression. */
function affineFit(cs, ys) {
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], r = [0, 0, 0];
  cs.forEach((c, i) => { const x = [c[0], c[1], 1]; for (let p = 0; p < 3; p++) { r[p] += x[p] * ys[i]; for (let q = 0; q < 3; q++) A[p][q] += x[p] * x[q]; } });
  const n = 3, Ab = A.map((row, i) => [...row, r[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let i = c + 1; i < n; i++) if (Math.abs(Ab[i][c]) > Math.abs(Ab[p][c])) p = i;
    [Ab[c], Ab[p]] = [Ab[p], Ab[c]];
    for (let i = 0; i < n; i++) { if (i === c) continue; const m = Ab[i][c] / Ab[c][c]; for (let j = c; j <= n; j++) Ab[i][j] -= m * Ab[c][j]; }
  }
  return Ab.map((row, i) => row[n] / row[i]);
}
/* Exercise 1 on a regular grid of design variables (no individual deviation). */
function familyGrid(n = 21) {
  const cs = [], wn = [], zeta = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const c = [-1 + 2 * i / (n - 1), -1 + 2 * j / (n - 1)], nat = natural(etaOf(c));
    cs.push(c); wn.push(nat.wn); zeta.push(nat.zeta);
  }
  return { cs, wn, zeta, wnFit: affineFit(cs, wn), zetaFit: affineFit(cs, zeta) };
}
/* Inputs for the hero: the notebook's five-sine signal, a unit step, or one sine of frequency fHz. */
function heroInput(kind, fHz, T = 300) {
  if (kind === "multisine") return (typeof window !== "undefined" && window.FS01) ? window.FS01.u_demo : new Array(T).fill(0);
  if (kind === "step") return new Array(T).fill(1);
  return Array.from({ length: T }, (_, t) => Math.sin(2 * Math.PI * fHz * DT * t));
}
/* xorshift128+ with a Box-Muller normal, so the envelope widget is repeatable. */
function makeRng(seed) {
  let s0 = (seed >>> 0) || 1, s1 = 0x9e3779b9 ^ s0;
  const next = () => { let x = s0, y = s1; s0 = y; x ^= x << 23; x ^= x >>> 17; x ^= y ^ (y >>> 26); s1 = x; return ((s0 + s1) >>> 0) / 4294967296; };
  let spare = null;
  const normal = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u1 = 0; while (u1 <= 1e-12) u1 = next();
    const u2 = next(), r = Math.sqrt(-2 * Math.log(u1));
    spare = r * Math.sin(2 * Math.PI * u2);
    return r * Math.cos(2 * Math.PI * u2);
  };
  return { next, normal };
}
/* Exercise 2: `runs` noisy repeats of one input on one system, with disturbance colour rho. */
function envelope(sys, u, runs, rho, seed) {
  const rng = makeRng(seed);
  const clean = simulate(sys, u).q, n = clean.length;
  const sum = new Float64Array(n), sq = new Float64Array(n), lo = new Float64Array(n).fill(Infinity), hi = new Float64Array(n).fill(-Infinity);
  for (let r = 0; r < runs; r++) {
    const q = simulateRho(sys, u, rng.normal, rho);
    for (let t = 0; t < n; t++) { sum[t] += q[t]; sq[t] += q[t] * q[t]; if (q[t] < lo[t]) lo[t] = q[t]; if (q[t] > hi[t]) hi[t] = q[t]; }
  }
  const spread = Array.from(sum, (s, t) => Math.sqrt(Math.max(0, sq[t] / runs - (s / runs) ** 2)));
  const above = spread.filter(s => s > SIGMA_Y).length;
  return { clean, lo: Array.from(lo), hi: Array.from(hi), spread, frac: above / n, meanSpread: spread.reduce((a, b) => a + b, 0) / n, first: spread.findIndex(s => s > SIGMA_Y) };
}
function simulateRho(sys, u, normal, rho) {
  const [k, d, b, a] = sys.eta;
  let q = X0[0], v = X0[1], w = 0;
  const qs = [q];
  for (let t = 0; t < u.length; t++) {
    const ut = t === 0 ? 0 : u[t - 1];
    w = rho * w + SIGMA_W * normal();
    v += DT * (-k * q - d * v + b * actuator(ut) - a * q * q * q - sys.gamma * q ** 5 + sys.bias + w);
    q += DT * v;
    qs.push(q);
  }
  return qs;
}
/* NET-END */

if (typeof module !== "undefined" && module.exports) {
  module.exports = { DT, ETA0, M, actuator, etaOf, memberOf, simulate, predict, rmse, freqResp, natural, affineFit, familyGrid, envelope, makeRng };
  return;
}

const D = window.FS01;
const C5 = ["--post", "--warm", "--c4", "--c5", "--ink"];
const timeAxis = n => Array.from({ length: n }, (_, t) => DT * t);

/* ============================================================ hero */
const hero = { input: "multisine", f: 0.17, custom: false, c: [0.5, -0.5] };
const omegas = Array.from({ length: 400 }, (_, i) => 0.05 + (3 - 0.05) * i / 399);
function drawHero() {
  const u = heroInput(hero.input, hero.f);
  const members = D.systems.map((s, i) => ({ sys: s, color: C5[i], label: `c = (${s.context[0] > 0 ? "+" : s.context[0] < 0 ? "−" : ""}${Math.abs(s.context[0])}, ${s.context[1] > 0 ? "+" : s.context[1] < 0 ? "−" : ""}${Math.abs(s.context[1])})` }));
  const cust = memberOf(hero.c);
  if (hero.custom) members.push({ sys: cust, color: "--lik", label: "yours", dash: [6, 4] });
  const traces = members.map(m => simulate(m.sys, u).q);
  const t = timeAxis(u.length + 1);
  const cvT = $("#heroT"), narrow = narrowOf(cvT, 600);
  let ymax = 0; traces.forEach(q => q.forEach(v => { ymax = Math.max(ymax, Math.abs(v)); })); ymax = Math.max(0.3, Math.ceil(ymax * 10) / 10);
  const fr = frame(cvT, narrow ? 1.6 : 2.6, { x: [0, 30], y: [-ymax, ymax], xlabel: "time [s]", ylabel: "position q", xticks: [0, 5, 10, 15, 20, 25, 30] });
  const { ctx } = fr;
  ctx.strokeStyle = css("--rule"); seg(ctx, fr.X(0), fr.Y(0) + 0.5, fr.X(30), fr.Y(0) + 0.5);
  // the input, faint, scaled into the band
  plotLine(fr, t.slice(0, u.length), u.map(v => v * ymax * 0.9), { color: alpha("--muted", 0.35), width: 1 });
  members.forEach((m, i) => plotLine(fr, t, traces[i], { color: css(m.color), width: m.dash ? 2.2 : 1.6, dash: m.dash || null }));
  // frequency response
  const cvF = $("#heroF");
  const Hs = members.map(m => omegas.map(om => freqResp(m.sys.eta, om)));
  const hmax = Math.max(...Hs.map(h => Math.max(...h)));
  const frF = frame(cvF, narrow ? 1.6 : 1.3, { x: [0, 3 / (2 * Math.PI)], y: [0, Math.ceil(hmax * 2) / 2], xlabel: "frequency [Hz]", ylabel: "|q / u| (small signal)", xticks: [0, 0.1, 0.2, 0.3, 0.4] });
  if (hero.input === "sine") { frF.ctx.strokeStyle = alpha("--warm", 0.7); frF.ctx.setLineDash([3, 3]); seg(frF.ctx, frF.X(hero.f), frF.pad.t, frF.X(hero.f), frF.Y(0)); frF.ctx.setLineDash([]); }
  members.forEach((m, i) => plotLine(frF, omegas.map(om => om / (2 * Math.PI)), Hs[i], { color: css(m.color), width: m.dash ? 2.2 : 1.6, dash: m.dash || null }));
  // legend
  $("#heroLegend").innerHTML = members.map(m => `<span><i style="background:${css(m.color)}"></i>${m.label}</span>`).join("");
  // readouts
  const nat = natural(cust.eta), peak = Math.max(...omegas.map(om => freqResp(cust.eta, om)));
  $("#rEta").textContent = `(${cust.eta.map(v => fmt(v, 3)).join(", ")})`;
  $("#rFn").textContent = `${fmt(nat.f, 3)} Hz, period ${fmt(2 * Math.PI / nat.wn, 2)} s`;
  $("#rZeta").textContent = fmt(nat.zeta, 3);
  $("#rPeak").textContent = fmt(peak, 2);
  const nats = D.systems.map(s => natural(s.eta)), fs = nats.map(n => n.f), zs = nats.map(n => n.zeta);
  const peaks = D.systems.map(s => Math.max(...omegas.map(om => freqResp(s.eta, om))));
  const amp = traces.slice(0, 5).map(q => Math.max(...q.map(Math.abs)));
  let text = `The five notebook members answer the ${hero.input === "multisine" ? "notebook's five-sine input" : hero.input === "step" ? "unit step" : `sine of ${fmt(hero.f, 2)} Hz`} (grey, scaled) with peak deflections from ${fmt(Math.min(...amp), 2)} to ${fmt(Math.max(...amp), 2)}. Their natural frequencies run from ${fmt(Math.min(...fs), 3)} to ${fmt(Math.max(...fs), 3)} Hz (periods ${fmt(1 / Math.max(...fs), 1)} to ${fmt(1 / Math.min(...fs), 1)} s), their damping ratios from ${fmt(Math.min(...zs), 3)} to ${fmt(Math.max(...zs), 3)}, and their resonance peaks from ${fmt(Math.min(...peaks), 2)} to ${fmt(Math.max(...peaks), 2)}: the softest member, [[c = (-1, -1)]], rings ${fmt(Math.max(...peaks) / Math.min(...peaks), 1)} times harder than the most damped one. `;
  if (hero.custom) text += `Your member at [[c = (${fmt(hero.c[0], 2)}, ${fmt(hero.c[1], 2)})]] (dashed) has [[\\eta = (${cust.eta.map(v => fmt(v, 3)).join(", ")})]], no individual deviation: [[f_n = ${fmt(nat.f, 3)}]] Hz and [[\\zeta = ${fmt(nat.zeta, 3)}]]. `;
  if (hero.input === "sine") text += `Move the frequency through the band 0.11 to 0.22 Hz and each member takes its turn to resonate; the saturating actuator and the hardening spring keep the true peaks below the linearised curve on the right.`;
  else if (hero.input === "step") text += `A step shows the oscillation and its decay directly: count the period, and read the damping from how fast the swings shrink.`;
  else text += `The frequency response is the linearisation about rest; the actual motion also feels the cubic spring and the saturating actuator, which is why the notebook later fits [[\\alpha]] and never a transfer function.`;
  caption($("#heroStatus"), text);
}
$$("#heroIn .btn").forEach(b => b.addEventListener("click", () => { $$("#heroIn .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); hero.input = b.dataset.in; $("#heroFreq").hidden = hero.input !== "sine"; drawHero(); }));
$("#hF").addEventListener("input", e => { hero.f = +e.target.value; $("#hFV").textContent = fmt(hero.f, 2) + " Hz"; drawHero(); });
$("#hCustom").addEventListener("click", () => { hero.custom = !hero.custom; $("#hCustom").setAttribute("aria-pressed", hero.custom); $("#heroC").hidden = !hero.custom; drawHero(); });
["hC1", "hC2"].forEach((id, i) => $("#" + id).addEventListener("input", e => { hero.c[i] = +e.target.value; $("#" + id + "V").textContent = fmt(hero.c[i], 2); drawHero(); }));
$("#hFV").textContent = fmt(hero.f, 2) + " Hz"; $("#hC1V").textContent = fmt(hero.c[0], 2); $("#hC2V").textContent = fmt(hero.c[1], 2);

/* ============================================================ actuator */
const act = { u: 1.0 };
function drawAct() {
  const cv = $("#actC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.5 : 2.4, { x: [-3, 3], y: [-0.35, 0.35], xlabel: "command u", ylabel: "force φ(u)", xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-0.3, -0.15, 0, 0.15, 0.3] });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--rule"); ctx.setLineDash([4, 4]); seg(ctx, X(-3), Y(0.3), X(3), Y(0.3)); seg(ctx, X(-3), Y(-0.3), X(3), Y(-0.3)); ctx.setLineDash([]);
  const us = Array.from({ length: 241 }, (_, i) => -3 + i / 40);
  plotLine(fr, us, us.map(u => 0.5 * u), { color: alpha("--muted", 0.6), width: 1, dash: [3, 3] });
  plotLine(fr, us, us.map(actuator), { color: css("--post"), width: 2.2 });
  const f = actuator(act.u);
  ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(act.u), Y(f), 4.5, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = alpha("--warm", 0.6); seg(ctx, X(act.u), Y(0), X(act.u), Y(f)); seg(ctx, X(0), Y(f), X(act.u), Y(f));
  caption($("#actCap"), `[[\\varphi(${fmt(act.u, 2)}) = ${fmt(f, 4)}]], which is ${fmt(100 * Math.abs(f) / 0.3, 1)}% of the force limit 0.3 and ${fmt(100 * Math.abs(f) / Math.max(1e-9, Math.abs(0.5 * act.u)), 1)}% of what the small-signal slope [[\\varphi'(0) = 0.3 / 0.6 = 0.5]] (dotted) would give. A command of 1 already delivers ${fmt(100 * actuator(1) / 0.3, 1)}% of the limit, a command of 2 delivers ${fmt(100 * actuator(2) / 0.3, 1)}%: beyond [[|u| \\approx 1]] the input is almost wasted, which is why every input in the course is scaled to peak 1 and why chapter 06 will find that experiment design is about frequency content, not amplitude.`);
}
$("#aU").addEventListener("input", e => { act.u = +e.target.value; $("#aUV").textContent = fmt(act.u, 2); drawAct(); });
$("#aUV").textContent = fmt(act.u, 2);

/* ============================================================ the family map (exercise 1) */
const fam = { colour: "c2" };
const grid = familyGrid(21);
function drawFam() {
  const cv = $("#famC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.25 : 1.8, { x: [0.65, 1.45], y: [0.08, 0.22], xlabel: "natural frequency ωₙ [rad/s]", ylabel: "damping ratio ζ", xticks: [0.7, 0.9, 1.1, 1.3], yticks: [0.1, 0.15, 0.2] });
  const { ctx, X, Y } = fr;
  const lo = css("--post"), hi = css("--warm");
  const mix = s => { const a = parseInt(lo.slice(1), 16), b = parseInt(hi.slice(1), 16); const ch = sh => Math.round(((a >> sh) & 255) * (1 - s) + ((b >> sh) & 255) * s); return `rgb(${ch(16)}, ${ch(8)}, ${ch(0)})`; };
  grid.cs.forEach((c, i) => { const s = (fam.colour === "c1" ? c[0] : c[1]) / 2 + 0.5; ctx.fillStyle = mix(s); ctx.beginPath(); ctx.arc(X(grid.wn[i]), Y(grid.zeta[i]), 2.6, 0, 2 * Math.PI); ctx.fill(); });
  // corners
  ctx.font = "11px " + css("--font-mono"); ctx.fillStyle = css("--ink"); ctx.textAlign = "center";
  [[-1, -1], [-1, 1], [1, -1], [1, 1]].forEach(c => { const n = natural(etaOf(c)); ctx.beginPath(); ctx.arc(X(n.wn), Y(n.zeta), 4, 0, 2 * Math.PI); ctx.fill(); ctx.fillText(`(${c[0] > 0 ? "+1" : "−1"}, ${c[1] > 0 ? "+1" : "−1"})`, X(n.wn), Y(n.zeta) + (c[1] > 0 ? -9 : 17)); });
  ctx.textAlign = "left";
  const wf = grid.wnFit, zf = grid.zetaFit;
  const corner = c => natural(etaOf(c));
  caption($("#famCap"), `A 21 × 21 grid of design variables, coloured by ${fam.colour === "c1" ? "[[c_1]]" : "[[c_2]]"} from blue (−1) to orange (+1). An affine fit over the grid gives [[\\omega_n \\approx ${fmt(wf[0], 3)}\\,c_1 + ${fmt(wf[1], 3)}\\,c_2 + ${fmt(wf[2], 3)}]] and [[\\zeta \\approx ${fmt(zf[0], 4)}\\,c_1 + ${fmt(zf[1], 4)}\\,c_2 + ${fmt(zf[2], 4)}]]: the frequency is set by [[c_1]], the damping ratio by [[c_2]], and the two are nearly independent. The corners: [[(+1, +1)]] is the highest-frequency member ([[\\omega_n = ${fmt(corner([1, 1]).wn, 3)}]]) and also the most heavily damped ([[\\zeta = ${fmt(corner([1, 1]).zeta, 3)}]]); the lightest damping, [[\\zeta = ${fmt(corner([-1, -1]).zeta, 3)}]], sits at [[(-1, -1)]], the slowest member. The notebook's solution attributes both to [[c_1 \\to +1]]; the numbers say otherwise.`);
}
$$("#famSeg .btn").forEach(b => b.addEventListener("click", () => { $$("#famSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); fam.colour = b.dataset.c; drawFam(); }));

/* ============================================================ the one experiment */
const exp = { y: true, truth: true, clean: true };
function drawExp() {
  const u = D.u, qt = D.q_true, y = D.y, qc = simulate(D.systems[2], u).q, t = timeAxis(u.length + 1);
  const cvU = $("#expU"), narrow = narrowOf(cvU);
  const frU = frame(cvU, narrow ? 4 : 6, { x: [0, 20], y: [-1.1, 1.1], ylabel: "input u", xticks: false, yticks: [-1, 0, 1] });
  plotLine(frU, t.slice(0, u.length).flatMap((x, i) => i ? [x, x] : [x]).concat([20]), u.flatMap((v, i) => i ? [u[i - 1], v] : [v]).concat([u[u.length - 1]]), { color: css("--muted"), width: 1.4 });
  const cv = $("#expC");
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, 20], y: [-0.35, 0.35], xlabel: "time [s]", ylabel: "position", xticks: [0, 5, 10, 15, 20], yticks: [-0.3, -0.15, 0, 0.15, 0.3] });
  const { ctx, X, Y } = fr;
  if (exp.y) { ctx.fillStyle = alpha("--muted", 0.8); y.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(t[i]), Y(v), 1.8, 0, 2 * Math.PI); ctx.fill(); }); }
  if (exp.truth) plotLine(fr, t, qt, { color: css("--ink"), width: 1.6 });
  if (exp.clean) plotLine(fr, t, qc, { color: css("--post"), width: 1.6, dash: [6, 4] });
  $("#eDist").textContent = fmt(rmse(qt, qc), 4);
  $("#eNoise").textContent = fmt(rmse(y, qt), 4);
  $("#eBoth").textContent = fmt(rmse(y, qc), 4);
  caption($("#expCap"), `The notebook's experiment on the member [[c = (0, 0)]]: a three-sine input of 20 s (top), the noisy sensor readings (dots), the true position (solid) and the response the same system would have had without its coloured disturbance (dashed, computed live). Truth and disturbance-free response differ by RMSE ${fmt(rmse(qt, qc), 4)}, the notebook's 0.0165, while the sensor noise alone contributes ${fmt(rmse(y, qt), 4)}: more of what the learner sees departing from any model is unpredictable disturbance than measurement error. Toggle the curves to see how the dots hide the gap.`);
}
$$("#expSeg .btn").forEach(b => b.addEventListener("click", () => { exp[b.dataset.k] = !exp[b.dataset.k]; b.setAttribute("aria-pressed", exp[b.dataset.k]); drawExp(); }));

/* ============================================================ repeat runs (exercise 2) */
const env = { rho: 0.8, seed: 1, runs: 50 };
function drawEnv() {
  const u = D.u, sys = D.systems[2];
  const e = envelope(sys, u, env.runs, env.rho, env.seed), t = timeAxis(u.length + 1);
  const cv = $("#envC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, 20], y: [-0.35, 0.35], xlabel: "time [s]", ylabel: "position", xticks: [0, 5, 10, 15, 20], yticks: [-0.3, -0.15, 0, 0.15, 0.3] });
  const { ctx, X, Y } = fr;
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  ctx.fillStyle = alpha("--warm", 0.22); ctx.beginPath();
  t.forEach((x, i) => i ? ctx.lineTo(X(x), Y(e.hi[i])) : ctx.moveTo(X(x), Y(e.hi[i])));
  for (let i = t.length - 1; i >= 0; i--) ctx.lineTo(X(t[i]), Y(e.lo[i]));
  ctx.closePath(); ctx.fill(); ctx.restore();
  plotLine(fr, t, e.clean, { color: css("--ink"), width: 1.6 });
  const cv2 = $("#envS");
  const fr2 = frame(cv2, narrow ? 3 : 5, { x: [0, 20], y: [0, Math.max(0.02, Math.ceil(Math.max(...e.spread) * 100) / 100)], ylabel: "spread across runs (sd)", xticks: [0, 5, 10, 15, 20] });
  fr2.ctx.strokeStyle = alpha("--muted", 0.8); fr2.ctx.setLineDash([3, 3]); seg(fr2.ctx, fr2.X(0), fr2.Y(SIGMA_Y), fr2.X(20), fr2.Y(SIGMA_Y)); fr2.ctx.setLineDash([]);
  plotLine(fr2, t, e.spread, { color: css("--warm"), width: 1.8 });
  const sd = SIGMA_W / Math.sqrt(1 - env.rho * env.rho), tc = -DT / Math.log(env.rho);
  caption($("#envCap"), `${env.runs} noisy repeats of the notebook's experiment with disturbance colour [[\\rho = ${fmt(env.rho, 2)}]]: the band is the min–max envelope, the solid line the disturbance-free response, and the lower strip the standard deviation across runs against the sensor noise [[\\sigma_y = 0.01]] (dotted). The spread exceeds the sensor noise on ${fmt(100 * e.frac, 0)}% of the samples${e.first < 0 ? "" : ` (first at [[t = ${fmt(DT * e.first, 1)}]] s)`}, mean spread ${fmt(e.meanSpread, 4)}. ${env.rho >= 0.79 && env.rho <= 0.81 ? "The notebook's own run of this exercise, on a freshly drawn member and input, reports 92%; over 20 other seeds the fraction ranges from 49% to 92%. " : ""}A coloured disturbance has stationary standard deviation [[\\sigma_w / \\sqrt{1 - \\rho^2} = ${fmt(sd, 4)}]] and correlation time [[-\\Delta t / \\ln \\rho = ${fmt(tc, 2)}]] s, so it pushes the cart the same way for several samples and the oscillator integrates it; with [[\\rho = 0]] the kicks average out and the band shrinks.`);
}
$("#vRho").addEventListener("input", e => { env.rho = +e.target.value; $("#vRhoV").textContent = fmt(env.rho, 2); drawEnv(); });
$("#vNew").addEventListener("click", () => { env.seed = (env.seed * 7 + 3) % 100003; drawEnv(); });
$("#vRhoV").textContent = fmt(env.rho, 2);

/* ============================================================ worked Euler steps */
function fillWorked() {
  const eta = ETA0, [k, d, , a] = eta;
  let q = 0.2, v = 0; const rows = [];
  for (let t = 1; t <= 3; t++) { const acc = -k * q - d * v - a * q ** 3; v += DT * acc; q += DT * v; rows.push([t, acc, v, q]); }
  $("#workT tbody").innerHTML = rows.map(r => `<tr><td>${r[0]}</td><td>${fmt(r[1], 5)}</td><td>${fmt(r[2], 5)}</td><td>${fmt(r[3], 5)}</td></tr>`).join("");
}

function drawAll() { drawHero(); drawAct(); drawFam(); drawExp(); drawEnv(); }
fillWorked();
drawAll();
onRedraw(drawAll);
})();
