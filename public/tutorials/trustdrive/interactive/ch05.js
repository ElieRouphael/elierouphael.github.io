/* TrustDrive chapter 05, Kalman filter and NIS: the live demos.
   simulate() is the notebook's run_filter: the same model, noise levels, steering input and
   fault, with its own seeded noise, so its numbers vary a little from the notebook's.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, int, caption, frame, plotLine, alpha, onRedraw, narrowOf } = window.Kit;
const { makeRng, pdf } = window.Stats;
const L = 2.7, TS = 0.05, V = 15, N = 500, WARM = 50, GATE = -2 * Math.log(0.01);   // chi-squared, 2 dof, 99%: 9.21
const QT = [2e-4, 1e-4], RT = [0.02 ** 2, 0.01 ** 2];                                // true noise covariances (diagonal)
function seg(ctx, x0, y0, x1, y1) { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }
const scaleOf = id => 10 ** +$(id).value;

/* ---------------- 2 x 2 helpers ---------------- */
const mul = (M, N2) => [[M[0][0] * N2[0][0] + M[0][1] * N2[1][0], M[0][0] * N2[0][1] + M[0][1] * N2[1][1]],
                        [M[1][0] * N2[0][0] + M[1][1] * N2[1][0], M[1][0] * N2[0][1] + M[1][1] * N2[1][1]]];
const mv = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
const tr = M => [[M[0][0], M[1][0]], [M[0][1], M[1][1]]];
const add = (M, N2) => [[M[0][0] + N2[0][0], M[0][1] + N2[0][1]], [M[1][0] + N2[1][0], M[1][1] + N2[1][1]]];
function inv(M) { const d = M[0][0] * M[1][1] - M[0][1] * M[1][0]; return [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]]; }
const quad = (v, M) => v[0] * (M[0][0] * v[0] + M[0][1] * v[1]) + v[1] * (M[1][0] * v[0] + M[1][1] * v[1]);
const diag = (a, b) => [[a, 0], [0, b]];

/* ---------------- the notebook's run_filter ---------------- */
function simulate({ seed = 1, onset = null, bias = [0, 0], rScale = 1, filterV = V, useInput = true } = {}) {
  const rng = makeRng(seed);
  const A = [[1, V * TS], [0, 1]], B = [0, V * TS / L];                          // the true car
  const Af = [[1, filterV * TS], [0, 1]], Bf = [0, filterV * TS / L];           // what the filter believes
  const Qf = diag(...QT), Rc = diag(RT[0] * rScale, RT[1] * rScale);
  let x = [0, 0], xh = [0, 0], P = diag(0.05, 0.02);
  const out = { t: [], x: [], y: [], xh: [], nu0: [], nu1: [], nis: [], nees: [], s0: [] };
  for (let k = 0; k < N; k++) {
    const u = 0.05 * Math.sin(2 * Math.PI * k * TS / 3);                         // known steering input
    x = [A[0][0] * x[0] + A[0][1] * x[1] + B[0] * u + rng.normal(0, Math.sqrt(QT[0])),
         A[1][0] * x[0] + A[1][1] * x[1] + B[1] * u + rng.normal(0, Math.sqrt(QT[1]))];
    const b = onset !== null && k >= onset ? bias : [0, 0];
    const y = [x[0] + b[0] + rng.normal(0, Math.sqrt(RT[0])), x[1] + b[1] + rng.normal(0, Math.sqrt(RT[1]))];
    // Kalman filter
    const xp0 = mv(Af, xh), xp = [xp0[0] + (useInput ? Bf[0] * u : 0), xp0[1] + (useInput ? Bf[1] * u : 0)];
    const Pp = add(mul(mul(Af, P), tr(Af)), Qf);
    const nu = [y[0] - xp[0], y[1] - xp[1]], S = add(Pp, Rc), Si = inv(S);
    const nis = quad(nu, Si), K = mul(Pp, Si);
    const Knu = mv(K, nu);
    xh = [xp[0] + Knu[0], xp[1] + Knu[1]];
    P = mul([[1 - K[0][0], -K[0][1]], [-K[1][0], 1 - K[1][1]]], Pp);
    const e = [x[0] - xh[0], x[1] - xh[1]];
    out.t.push((k + 1) * TS); out.x.push(x[0]); out.y.push(y[0]); out.xh.push(xh[0]);
    out.nu0.push(nu[0]); out.nu1.push(nu[1]); out.nis.push(nis); out.nees.push(quad(e, inv(P))); out.s0.push(S[0][0]);
  }
  return out;
}
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

/* =====================================================================
   THE DEMO: A FAULT THE CAMERA WILL NOT ADMIT
   ===================================================================== */
const hero = { seed: 2, run: null, onset: 250 };
function updateHero() {
  const by = +$("#kBiasY").value, bp = +$("#kBiasP").value, on = +$("#kOn").value, rs = scaleOf("#kR");
  $("#kBiasYV").textContent = `${fmt(by, 2)} m`;
  $("#kBiasPV").textContent = `${fmt(bp, 3)} rad`;
  $("#kOnV").textContent = `${fmt(on, 1)} s`;
  $("#kRV").textContent = `${fmt(rs, 2)}${Math.abs(rs - 1) < 1e-9 ? " (honest)" : rs < 1 ? " (overconfident)" : " (conservative)"}`;
  hero.onset = Math.round(on / TS);
  hero.run = simulate({ seed: hero.seed, onset: hero.onset, bias: [by, bp], rScale: rs });
  drawHero(); heroStatus();
}
function drawHero() {
  const r = hero.run, tOn = hero.onset * TS, narrow = narrowOf($("#kfC"));
  // errors relative to the truth: the simulated car is not steered, so its offset itself drifts far
  const camErr = r.y.map((y, k) => y - r.x[k]), filtErr = r.xh.map((v, k) => v - r.x[k]);
  const m = Math.max(0.1, 1.1 * Math.max(...camErr.map(Math.abs), ...filtErr.map(Math.abs)));
  const fr = frame($("#kfC"), narrow ? 1.5 : 2.6, { x: [0, 25], y: [-m, m], xlabel: "time (s)", ylabel: "error (m)", yfmt: v => fmt(v, 2) });
  fr.ctx.fillStyle = alpha("--geom", 0.07); fr.ctx.fillRect(fr.X(tOn), fr.pad.t, fr.X(25) - fr.X(tOn), fr.ih);
  fr.ctx.fillStyle = alpha("--prior", 0.7);
  r.t.forEach((t, k) => { fr.ctx.fillRect(fr.X(t) - 1, fr.Y(camErr[k]) - 1, 2, 2); });
  plotLine(fr, [0, 25], [0, 0], { color: css("--ink"), width: 1.2 });
  plotLine(fr, r.t, filtErr, { color: css("--path"), width: 2 });
  const top = Math.min(60, Math.max(14, 1.05 * Math.max(...r.nis)));
  const gr = frame($("#nisC"), narrow ? 1.7 : 3, { x: [0, 25], y: [0, top], xlabel: "time (s)", ylabel: "NIS", yfmt: v => String(Math.round(v)) });
  const { ctx, X, Y } = gr;
  ctx.fillStyle = alpha("--geom", 0.07); ctx.fillRect(X(tOn), gr.pad.t, X(25) - X(tOn), gr.ih);
  plotLine(gr, r.t, r.nis, { color: css("--warm"), width: 1.2 });
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]); seg(ctx, X(0), Y(GATE), X(25), Y(GATE)); ctx.setLineDash([]);
  ctx.fillStyle = css("--geom");
  r.t.forEach((t, k) => { if (r.nis[k] > GATE) ctx.fillRect(X(t) - 1, gr.Y(0) - 7, 2, 7); });
  ctx.font = "600 10px " + css("--font-ui"); ctx.fillStyle = css("--ink"); ctx.textAlign = "right"; ctx.fillText("9.21", X(25) - 2, Y(GATE) - 4);
  // readouts
  const before = r.nis.slice(0, hero.onset), after = r.nis.slice(hero.onset);
  $("#rB").textContent = fmt(mean(before), 2);
  $("#rA").textContent = fmt(mean(after), 2);
  $("#rDet").textContent = `${Math.round(100 * after.filter(v => v > GATE).length / after.length)}%`;
  $("#rFa").textContent = `${before.filter(v => v > GATE).length} of ${before.length}`;
}
function heroStatus() {
  const r = hero.run, before = r.nis.slice(0, hero.onset), after = r.nis.slice(hero.onset), rs = scaleOf("#kR");
  const det = after.filter(v => v > GATE).length / after.length, first = after.findIndex(v => v > GATE);
  const by = +$("#kBiasY").value, bp = +$("#kBiasP").value;
  let text;
  if (!by && !bp) text = `No fault: the NIS wanders around its expected value of 2 (mean ${fmt(mean(r.nis), 2)}), crossing the 99% line on ${fmt(100 * r.nis.filter(v => v > GATE).length / r.nis.length, 1)}% of the steps, about the 1% it should.`;
  else text = `From ${fmt(hero.onset * TS, 1)} s the camera is off by ${fmt(by, 2)} m and ${fmt(bp, 3)} rad, while still claiming its usual noise. ` +
    `The mean NIS goes from <b>${fmt(mean(before), 1)}</b> to <b>${fmt(mean(after), 1)}</b>, and <b>${Math.round(100 * det)}%</b> of the samples cross the line` +
    (first >= 0 ? `, the first ${fmt(first * TS, 2)} s after the onset. ` : ". ") +
    (det > 0.2 ? "The filter's own estimate is dragged along with the bias (the blue line), so filtering alone does not fix a lying camera; but the camera's own uncertainty never rose, and the NIS gave it away." : "A small bias hides in the noise of single samples: chapter 09's CUSUM accumulates it.");
  if (Math.abs(rs - 1) > 1e-9) text += rs < 1 ? ` The camera understates its noise ×${fmt(rs, 2)}, so even before the fault the filter cries wolf.` : ` The camera overstates its noise ×${fmt(rs, 2)}, which shrinks every NIS and helps the fault hide.`;
  caption($("#kfStatus"), text);
}
["#kBiasY", "#kBiasP", "#kOn", "#kR"].forEach(id => $(id).addEventListener("input", updateHero));
$("#kNew").addEventListener("click", () => { hero.seed++; updateHero(); });

/* =====================================================================
   PART 2: ONE MEASUREMENT UPDATE, IN ONE DIMENSION
   ===================================================================== */
const UX = Array.from({ length: 401 }, (_, i) => -1.6 + 3.4 * i / 400);
function drawUpd() {
  const P = 10 ** +$("#uP").value, R = 10 ** +$("#uR").value, y = +$("#uY").value;
  $("#uPV").textContent = fmt(P, 3); $("#uRV").textContent = fmt(R, 3); $("#uYV").textContent = fmt(y, 2);
  const K = P / (P + R), m = K * y, Pn = (1 - K) * P;
  const g = (mu, v) => UX.map(x => pdf.normal(x, mu, Math.sqrt(v)));
  const prior = g(0, P), lik = g(y, R), post = g(m, Pn), top = Math.max(...post, ...prior, ...lik) * 1.1;
  const cv = $("#updC");
  const fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [-1.6, 1.8], y: [0, top], yticks: false, xlabel: "state", xfmt: v => fmt(v, 1) });
  plotLine(fr, UX, prior, { color: css("--prior"), width: 2, fill: alpha("--prior", 0.18) });
  plotLine(fr, UX, lik, { color: css("--lik"), width: 2, dash: [6, 4] });
  plotLine(fr, UX, post, { color: css("--post"), width: 2.4, fill: alpha("--post", 0.18) });
  caption($("#updCap"),
    `[[K = \\frac{P}{P + R} = \\frac{${fmt(P, 3)}}{${fmt(P + R, 3)}} = ${fmt(K, 2)}]]: the update moves ${Math.round(100 * K)}% of the way from the prediction (0) to the measurement (${fmt(y, 2)}), landing at <b>${fmt(m, 2)}</b>, with variance [[(1 - K)P = ${fmt(Pn, 4)}]], smaller than both. ` +
    (K > 0.8 ? "The prediction is vague, so the measurement wins." : K < 0.2 ? "The measurement is noisy, so the prediction wins." : "Both carry weight."));
}
["#uP", "#uR", "#uY"].forEach(id => $(id).addEventListener("input", drawUpd));

/* =====================================================================
   PART 3: WHITENESS, AND WHAT A WRONG MODEL DOES TO IT
   ===================================================================== */
const MODELS = { ok: {}, speed: { filterV: 10 }, input: { useInput: false } };
let whtModel = "ok";
function autocorr(x, maxlag) {
  const m = mean(x), d = x.map(v => v - m), c0 = d.reduce((s, v) => s + v * v, 0);
  return Array.from({ length: maxlag + 1 }, (_, l) => { let s = 0; for (let k = 0; k + l < d.length; k++) s += d[k] * d[k + l]; return s / c0; });
}
function drawWht() {
  $$("#wht [data-model]").forEach(b => b.setAttribute("aria-pressed", b.dataset.model === whtModel ? "true" : "false"));
  const r = simulate({ seed: 1, ...MODELS[whtModel] });
  const acY = autocorr(r.nu0.slice(WARM), 25), acP = autocorr(r.nu1.slice(WARM), 25), band = 1.96 / Math.sqrt(N - WARM);
  const cv = $("#whtC");
  const fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [-0.5, 25.5], y: [-0.5, 1.05], xlabel: "lag (steps of 0.05 s)", yfmt: v => fmt(v, 1), xticks: [0, 5, 10, 15, 20, 25] });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--c4", 0.15); ctx.fillRect(X(-0.5), Y(band), X(25.5) - X(-0.5), Y(-band) - Y(band));
  ctx.strokeStyle = alpha("--ink", 0.5); ctx.lineWidth = 1; seg(ctx, X(-0.5), Y(0), X(25.5), Y(0));
  [[acY, -0.18, "--path"], [acP, 0.18, "--warm"]].forEach(([ac, dx, col]) => ac.forEach((a, l) => {
    ctx.strokeStyle = css(col); ctx.lineWidth = 1.8; seg(ctx, X(l + dx), Y(0), X(l + dx), Y(a));
    ctx.fillStyle = css(col); ctx.beginPath(); ctx.arc(X(l + dx), Y(a), 3, 0, 2 * Math.PI); ctx.fill();
  }));
  const outY = acY.slice(1).filter(a => Math.abs(a) > band).length, outP = acP.slice(1).filter(a => Math.abs(a) > band).length;
  const what = { ok: "With the correct model", speed: "With the filter believing the car goes 10 m/s instead of 15", input: "With the filter ignoring the steering it knows about" }[whtModel];
  const white = outY <= 3 && outP <= 3;
  caption($("#whtCap"),
    `${what}, the lag-1 autocorrelation is <b>${fmt(acY[1], 2)}</b> for the lateral innovation and <b>${fmt(acP[1], 2)}</b> for the heading innovation; ${outY} and ${outP} of the 25 lags fall outside the green band [[\\pm\\frac{1.96}{\\sqrt{N}}]] that white noise stays in 95% of the time. ` +
    (white ? "White: the filter has taken everything predictable." : `Not white: the innovations still carry a pattern the model failed to predict${whtModel === "input" ? ", here in the heading, which the forgotten steering turns" : ", here in the offset, which the wrong speed integrates wrongly"}. The mean NIS is ${fmt(mean(r.nis.slice(WARM)), 2)} instead of about 2.`));
}
$$("#wht [data-model]").forEach(b => b.addEventListener("click", () => { whtModel = b.dataset.model; drawWht(); }));

/* =====================================================================
   PART 4: THE CONSISTENCY TEST, AND HONEST NOISE
   ===================================================================== */
/* 95% interval for the mean of n chi-squared(2) samples (Wilson-Hilferty, chi-squared with 2n dof) */
function meanInterval(n) {
  const k = 2 * n, q = z => k * (1 - 2 / (9 * k) + z * Math.sqrt(2 / (9 * k))) ** 3 / n;
  return [q(-1.96), q(1.96)];
}
function drawCons() {
  const rs = scaleOf("#cR");
  $("#cRV").textContent = `${fmt(rs, 2)}${Math.abs(rs - 1) < 1e-9 ? " (honest)" : ""}`;
  const nis = simulate({ seed: 3, rScale: rs }).nis.slice(WARM), [lo, hi] = meanInterval(nis.length), m = mean(nis);
  const B = 30, w = 0.5, counts = new Array(B).fill(0);
  nis.forEach(v => { const i = Math.floor(v / w); if (i < B) counts[i]++; });
  const dens = counts.map(c => c / (nis.length * w)), xs = Array.from({ length: 151 }, (_, i) => 0.1 * i);
  const cv = $("#consC");
  const fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [0, 15], y: [0, Math.max(0.55, ...dens) * 1.1], xlabel: "NIS", yticks: false });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--warm", 0.55);
  dens.forEach((d, i) => { if (d) ctx.fillRect(X(i * w) + 0.5, Y(d), X((i + 1) * w) - X(i * w) - 1, Y(0) - Y(d)); });
  plotLine(fr, xs, xs.map(x => 0.5 * Math.exp(-x / 2)), { color: css("--ink"), width: 2 });
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]); seg(ctx, X(GATE), Y(0), X(GATE), fr.pad.t); ctx.setLineDash([]);
  const above = nis.filter(v => v > GATE).length / nis.length;
  const verdict = m > hi ? "above the interval: <b>overconfident</b>, the claimed noise is too small and false alarms multiply"
    : m < lo ? "below the interval: <b>conservative</b>, the claimed noise is too large, and real faults will hide" : "inside the interval: <b>consistent</b>";
  caption($("#consCap"),
    `Claiming ${fmt(rs, 2)} times the true noise covariance: the mean NIS over ${nis.length} steps is <b>${fmt(m, 2)}</b>, ${verdict}. ` +
    `The 95% interval for that mean is [${fmt(lo, 2)}, ${fmt(hi, 2)}]; ${fmt(100 * above, 1)}% of the samples cross the 9.21 line (dashed), against 1% for a consistent filter.`);
}
$("#cR").addEventListener("input", drawCons);

/* =====================================================================
   start-up
   ===================================================================== */
updateHero(); drawUpd(); drawWht(); drawCons();
onRedraw(() => { drawHero(); drawUpd(); drawWht(); drawCons(); });
})();
