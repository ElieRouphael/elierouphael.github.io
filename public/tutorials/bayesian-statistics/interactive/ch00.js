/* Chapter 00, What Bayesian statistics is: the live demos.
   The 200 patients are the notebook's (numpy seed 0, true cure rate 0.3), copied exactly;
   the posteriors are exact beta distributions computed in the browser.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, css, fmt, int, caption, frame, fit, plotLine, alpha, onRedraw, reduceMotion, narrowOf } = window.Kit;
const { pmf, pdf, makeRng, linspace } = window.Stats;
const rng = makeRng();

const NOTEBOOK = "00000001000001010010100101000001001000010011000000000000011100100000100101000000000100000001010000011001100011001001110100010101100000000000000011001110011100100010000101100000110111000101101000000000";
const TRUE_RATE = 0.3, N_MAX = 200;
const PRIORS = {
  flat: [1, 1, "every cure rate from 0 to 1 is equally plausible"],
  hopeful: [7, 3, "a hopeful guess of about 70%, worth about as much as 10 patients"],
  sceptical: [1, 9, "a sceptical guess of about 10%, worth about as much as 10 patients"],
};

/* ---------------- exact beta posteriors (integer parameters) ---------------- */
/* Beta(a, b) CDF for whole-number a and b: P(Beta(a, b) <= x) = P(Binomial(a + b - 1, x) >= a). */
function betaCdf(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  let s = 0;
  for (let k = a; k <= a + b - 1; k++) s += pmf.binomial(k, a + b - 1, x);
  return Math.min(1, s);
}
function betaQuantile(p, a, b) {
  let lo = 0, hi = 1;
  for (let i = 0; i < 50; i++) { const mid = (lo + hi) / 2; if (betaCdf(mid, a, b) < p) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}
/* the density, including its finite value at an end when a or b is 1 */
function betaPdf(x, a, b) {
  if (x <= 0) return a === 1 ? b : 0;
  if (x >= 1) return b === 1 ? a : 0;
  return pdf.beta(x, a, b);
}
function summary(a, b) {
  return { a, b, mean: a / (a + b), lo: betaQuantile(0.025, a, b), hi: betaQuantile(0.975, a, b), above: 1 - betaCdf(0.5, a, b) };
}
const fmtP = p => (p < 0.001 ? "below 0.001" : fmt(p, 3));
const fmtQ = v => fmt(v, v < 0.01 ? 3 : 2);
const GRID = linspace(0, 1, 401);

/* =====================================================================
   THE DEMO: PATIENTS ARRIVE ONE BY ONE
   ===================================================================== */
const cure = { seq: Uint8Array.from(NOTEBOOK, c => +c), n: 0, prior: "flat", running: false, single: false, lastT: 0, owed: 0 };
const improved = n => { let y = 0; for (let i = 0; i < n; i++) y += cure.seq[i]; return y; };
function newPatients() { cure.seq = Uint8Array.from({ length: N_MAX }, () => (rng.uniform() < TRUE_RATE ? 1 : 0)); }
function drawCure() {
  const [a0, b0] = PRIORS[cure.prior], y = improved(cure.n);
  const s = summary(a0 + y, b0 + cure.n - y);
  const prior = GRID.map(x => betaPdf(x, a0, b0)), post = GRID.map(x => betaPdf(x, s.a, s.b));
  const cv = $("#cureC");
  const fr = frame(cv, narrowOf(cv) ? 1.45 : 2.2, { x: [0, 1], y: [0, Math.max(...post, ...prior) * 1.12], yticks: false, xlabel: "cure rate θ", xfmt: v => fmt(v, 1) });
  plotLine(fr, GRID, prior, { color: css("--prior"), width: 1.6, fill: alpha("--prior", 0.18) });
  // the 95% credible interval, shaded under the posterior
  const band = GRID.filter(x => x >= s.lo && x <= s.hi), xs = [s.lo, ...band, s.hi];
  plotLine(fr, xs, xs.map(x => betaPdf(x, s.a, s.b)), { fill: alpha("--post", 0.3) });
  plotLine(fr, GRID, post, { color: css("--post"), width: 2.4, fill: alpha("--post", 0.1) });
  if ($("#cureTruth").checked) {
    const { ctx, X, Y } = fr;
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(X(TRUE_RATE), Y(0)); ctx.lineTo(X(TRUE_RATE), fr.pad.t); ctx.stroke(); ctx.setLineDash([]);
  }
  drawPatients();
  $("#rN").textContent = `${cure.n} of ${N_MAX}`;
  $("#rY").textContent = int(y);
  $("#rMean").textContent = fmt(s.mean, 3);
  $("#rCI").textContent = `(${fmt(s.lo, 3)}, ${fmt(s.hi, 3)})`;
  $("#rHalf").textContent = fmtP(s.above);
  return { s, y };
}
function drawPatients() {
  const cv = $("#cureP"), narrow = narrowOf(cv), cols = narrow ? 25 : 40, rows = N_MAX / cols;
  const { ctx, w } = fit(cv, cols / rows * 1.0);
  const cell = w / cols, gap = Math.max(1, cell * 0.18);
  for (let i = 0; i < N_MAX; i++) {
    const x = (i % cols) * cell, y = Math.floor(i / cols) * cell;
    if (i < cure.n) {
      ctx.fillStyle = cure.seq[i] ? css("--post") : alpha("--prior", 0.55);
      ctx.fillRect(x + gap / 2, y + gap / 2, cell - gap, cell - gap);
    } else {
      ctx.strokeStyle = css("--rule"); ctx.lineWidth = 1;
      ctx.strokeRect(x + gap / 2 + 0.5, y + gap / 2 + 0.5, cell - gap - 1, cell - gap - 1);
    }
  }
}
function status(s, y) {
  if (cure.running) return;                       // read out once the patients stop arriving
  const el = $("#cureStatus"), n = cure.n;
  if (!n) { el.innerHTML = `No patients yet, so the posterior is the prior: ${PRIORS[cure.prior][2]}. Press <b>1 patient</b> or <b>play</b>.`; return; }
  let text = `After ${n} patient${n > 1 ? "s" : ""}, <b>${y ? int(y) : "none"}</b> improved. Best guess for the cure rate: <b>${fmt(s.mean, 2)}</b>; ` +
    `with 95% probability it lies between <b>${fmtQ(s.lo)}</b> and <b>${fmtQ(s.hi)}</b>.`;
  if (cure.single) text += cure.seq[n - 1] ? " The last patient improved: the posterior moves right." : " The last patient did not improve: the posterior moves left.";
  if (cure.prior !== "flat" && n <= 10) text += " With so few patients the prior still weighs a lot: compare with the flat prior.";
  if (cure.prior !== "flat" && n >= 150) text += " With this many patients the prior hardly matters: the flat prior gives almost the same answer.";
  if (n === N_MAX && $("#cureTruth").checked) text += s.lo <= TRUE_RATE && TRUE_RATE <= s.hi ? " The true rate, 0.3, is inside the interval." : " This time the true rate, 0.3, is outside the interval: that happens about one time in twenty.";
  el.innerHTML = text;
}
function update() { const { s, y } = drawCure(); status(s, y); }
function add(k) {
  cure.n = Math.min(N_MAX, cure.n + k);
  if (cure.n >= N_MAX && cure.running) setRunning(false);
}
function setRunning(on) {
  cure.running = on;
  $("#cureRun").setAttribute("aria-pressed", on ? "true" : "false");
  $("#cureRun").textContent = on ? "pause" : "play";
  if (on) { cure.lastT = performance.now(); cure.owed = 0; requestAnimationFrame(tick); }
}
/* 5 patients a second for the first 25, then 20 a second: about 14 seconds for all 200 */
function tick(now) {
  if (!cure.running) return;
  const dt = Math.min(0.5, Math.max(0, (now - cure.lastT) / 1000));
  cure.lastT = now;
  cure.owed += (cure.n < 25 ? 5 : 20) * dt;
  const k = Math.floor(cure.owed);
  cure.owed -= k;
  if (k) add(k);
  if (!cure.running) { update(); return; }
  drawCure();
  requestAnimationFrame(tick);
}
function restart() {
  setRunning(false);
  cure.n = 0; cure.single = false;
  if ($("#cureData").value === "new") newPatients(); else cure.seq = Uint8Array.from(NOTEBOOK, c => +c);
  update();
}
$("#cure1").addEventListener("click", () => { setRunning(false); add(1); cure.single = true; update(); });
$("#cure10").addEventListener("click", () => { setRunning(false); add(10); cure.single = false; update(); });
$("#cureRun").addEventListener("click", () => {
  cure.single = false;
  if (cure.running) { setRunning(false); update(); return; }
  if (cure.n >= N_MAX) cure.n = 0;
  if (reduceMotion) { cure.n = N_MAX; update(); return; }
  setRunning(true);
});
$("#cureReset").addEventListener("click", restart);
$("#cureData").addEventListener("change", restart);
$("#curePrior").addEventListener("change", () => { cure.prior = $("#curePrior").value; cure.single = false; if (cure.running) drawCure(); else update(); });
$("#cureTruth").addEventListener("change", () => { if (cure.running) drawCure(); else update(); });

/* =====================================================================
   PART 2: THE RUNNING PROPORTION OF HEADS
   ===================================================================== */
const FLIPS = 2000, COIN_COL = ["--post", "--warm", "--c4", "--c5", "--prior"];
let coins = null;
function flipCoins() {
  coins = COIN_COL.map(() => {
    const prop = new Float64Array(FLIPS);
    let heads = 0;
    for (let i = 0; i < FLIPS; i++) { heads += rng.uniform() < TRUE_RATE ? 1 : 0; prop[i] = heads / (i + 1); }
    return prop;
  });
  drawCoins();
}
const LOGN = Array.from({ length: FLIPS }, (_, i) => Math.log10(i + 1));
function drawCoins() {
  const cv = $("#coinC");
  const fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [0, Math.log10(FLIPS)], y: [0, 1], xticks: [0, 1, 2, 3], xfmt: v => int(10 ** v), yticks: [0, 0.3, 0.5, 1], yfmt: v => fmt(v, 1), xlabel: "number of flips" });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(X(0), Y(TRUE_RATE)); ctx.lineTo(fr.pad.l + fr.iw, Y(TRUE_RATE)); ctx.stroke(); ctx.setLineDash([]);
  coins.forEach((prop, j) => plotLine(fr, LOGN, prop, { color: alpha(COIN_COL[j], 0.85), width: 1.4 }));
  const span = i => { const v = coins.map(p => p[i - 1]); return `${fmt(Math.min(...v), 2)} to ${fmt(Math.max(...v), 2)}`; };
  caption($("#coinCap"),
    `After 10 flips, the five proportions of heads range from <b>${span(10)}</b>; after 100, from <b>${span(100)}</b>; after 2,000, from <b>${span(FLIPS)}</b>. ` +
    `All five head for 0.3 (dashed), each along its own path: the frequency is only the probability in the limit.`);
}
$("#coinAgain").addEventListener("click", flipCoins);

/* =====================================================================
   PART 4: THE POSTERIOR AFTER 0, 5, 25 AND 200 OF THE NOTEBOOK'S PATIENTS
   ===================================================================== */
const SNAPS = [0, 5, 25, 200], SHADES = [0.35, 0.55, 0.75, 1];
const NB_SEQ = Uint8Array.from(NOTEBOOK, c => +c);
const snaps = SNAPS.map(n => { let y = 0; for (let i = 0; i < n; i++) y += NB_SEQ[i]; return { n, y, ...summary(1 + y, 1 + n - y) }; });
$("#snapLegend").innerHTML = snaps.map((s, i) =>
  `<span class="key"><span class="sw" style="background:var(--post);opacity:${SHADES[i]}"></span>after ${s.n} patients (${s.y} improved)</span>`).join("");
function drawSnaps() {
  const curves = snaps.map(s => GRID.map(x => betaPdf(x, s.a, s.b)));
  const cv = $("#snapC");
  const fr = frame(cv, narrowOf(cv) ? 1.45 : 2.4, { x: [0, 1], y: [0, Math.max(...curves.flat()) * 1.08], yticks: false, xlabel: "cure rate θ", xfmt: v => fmt(v, 1) });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(X(TRUE_RATE), Y(0)); ctx.lineTo(X(TRUE_RATE), fr.pad.t); ctx.stroke(); ctx.setLineDash([]);
  curves.forEach((c, i) => plotLine(fr, GRID, c, { color: alpha("--post", SHADES[i]), width: 2 }));
}
const [, s5, s25, s200] = snaps;
caption($("#snapCap"),
  `After 5 patients, none improved: the best guess is <b>${fmt(s5.mean, 2)}</b>, but anything from ${fmtQ(s5.lo)} to ${fmtQ(s5.hi)} is plausible (95%). ` +
  `After 25 (${s25.y} improved): <b>${fmt(s25.mean, 2)}</b>, from ${fmtQ(s25.lo)} to ${fmtQ(s25.hi)}. ` +
  `After 200 (${s200.y} improved): <b>${fmt(s200.mean, 2)}</b>, from ${fmtQ(s200.lo)} to ${fmtQ(s200.hi)}, around the true 0.3 (dashed).`);

/* =====================================================================
   start-up
   ===================================================================== */
update(); flipCoins(); drawSnaps();
onRedraw(() => { drawCure(); drawCoins(); drawSnaps(); });
})();
