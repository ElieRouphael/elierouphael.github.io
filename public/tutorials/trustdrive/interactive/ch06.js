/* TrustDrive chapter 06, Operational design domain and the integrity score: the live demos.
   scenario() is the notebook's build_scenario: the same blur and bias faults, honest noise,
   Kalman filter and measured pixel conditions, with its own seeded noise, so its numbers
   vary a little from the notebook's.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { makeRng, quantile } = window.Stats;

/* ---------------- the notebook's scores and scenario ---------------- */
const TS = 0.05, V = 15, N = 800, T_END = N * TS, CHI2 = 9.210;                    // 40 s at 20 Hz; chi-squared, 2 dof, 99%
const REF = 0.02, NIS_SCALE = 6, GAIN = 6, BASE = [0.02, 0.01], Q = 1e-4;
const ODD = { blur: 1.6, bright: 0.45 };
const sUnc = trace => Math.exp(-trace / REF);
const sNis = nis => Math.exp(-Math.max(0, nis - CHI2) / NIS_SCALE);
const sOdd = (margin, a = GAIN) => 1 / (1 + Math.exp(-a * margin));
const ramp = (t, a, b, lo, hi) => (t <= a ? lo : t >= b ? hi : lo + (hi - lo) * (t - a) / (b - a));
const noiseFactor = (blur, bright) => (1 + 2.5 * blur) * (1 + 1.5 * Math.max(0, 1 - bright));   // reported noise, x clear weather
const pixelMargin = (blur, bright) => Math.min((bright - ODD.bright) / (1 - ODD.bright), 1 - blur / ODD.blur);
const fogOf = blur => 1 - 0.4 * blur / 2.6;                                         // the scenario's fog dims with the blur

function scenario(seed) {
  const rng = makeRng(seed), a = V * TS, q = Math.sqrt(Q);
  let x0 = 0, x1 = 0, h0 = 0, h1 = 0, p00 = 0.05, p01 = 0, p11 = 0.02;            // truth, estimate, covariance
  const out = { t: [], blur: [], su: [], sn: [], so: [], nis: [], kind: [] };
  for (let k = 0; k < N; k++) {
    const t = k * TS;
    const blur = t < 14 ? ramp(t, 8, 11, 0, 2.6) : ramp(t, 16, 18, 2.6, 0), bright = fogOf(blur);
    const biasOn = t >= 24 && t < 32, f = noiseFactor(blur, bright), r0 = BASE[0] * f, r1 = BASE[1] * f;
    x0 = x0 + a * x1 + rng.normal(0, q); x1 = x1 + rng.normal(0, q);               // the true lane error (no steering)
    const y0 = x0 + (biasOn ? 0.20 : 0) + rng.normal(0, r0), y1 = x1 + (biasOn ? 0.05 : 0) + rng.normal(0, r1);
    // Kalman filter, told the honest noise R = diag(r0^2, r1^2)
    const xp0 = h0 + a * h1, xp1 = h1;
    const q00 = p00 + 2 * a * p01 + a * a * p11 + Q, q01 = p01 + a * p11, q11 = p11 + Q;
    const s00 = q00 + r0 * r0, s11 = q11 + r1 * r1, det = s00 * s11 - q01 * q01;
    const n0 = y0 - xp0, n1 = y1 - xp1, nis = (s11 * n0 * n0 - 2 * q01 * n0 * n1 + s00 * n1 * n1) / det;
    const i00 = s11 / det, i01 = -q01 / det, i11 = s00 / det;
    const k00 = q00 * i00 + q01 * i01, k01 = q00 * i01 + q01 * i11, k10 = q01 * i00 + q11 * i01, k11 = q01 * i01 + q11 * i11;
    h0 = xp0 + k00 * n0 + k01 * n1; h1 = xp1 + k10 * n0 + k11 * n1;
    p00 = q00 - (k00 * q00 + k01 * q01); p01 = q01 - (k00 * q01 + k01 * q11); p11 = q11 - (k10 * q01 + k11 * q11);
    // the pixel conditions, measured with noise, for the domain monitor
    const blurM = Math.max(0, blur + rng.normal(0, 0.15)), brightM = clamp(bright + rng.normal(0, 0.05), 0, 1);
    out.t.push(t); out.blur.push(blur); out.nis.push(nis);
    out.su.push(sUnc(r0 * r0 + r1 * r1)); out.sn.push(sNis(nis)); out.so.push(sOdd(pixelMargin(blurM, brightM)));
    out.kind.push(blur > ODD.blur ? "blur" : biasOn ? "bias" : "");                // ground truth: untrustworthy while a fault is on
  }
  return out;
}
/* Fuse the chosen scores with a rule, then smooth: I_k = (1 - beta) I_{k-1} + beta I_raw,k. */
const RULES = {
  product: s => s.reduce((p, v) => p * v, 1),
  min: s => Math.min(...s),
  average: s => s.reduce((a, v) => a + v, 0) / s.length,
};
function fuse(run, rule = "product", use = [true, true, true], beta = 0.4) {
  const keys = ["su", "sn", "so"].filter((_, i) => use[i]), raw = [], I = [];
  let level = 1;
  for (let k = 0; k < N; k++) {
    const r = keys.length ? RULES[rule](keys.map(key => run[key][k])) : 1;
    level = (1 - beta) * level + beta * r; raw.push(r); I.push(level);
  }
  return { raw, I };
}
/* ROC curve as the notebook computes it: one point per distinct score, trapezoid AUC. */
function roc(score, label) {
  const idx = score.map((_, i) => i).sort((a, b) => score[b] - score[a]);
  const P = label.filter(Boolean).length, Nn = label.length - P, fpr = [0], tpr = [0];
  let tp = 0, fp = 0;
  for (let j = 0; j < idx.length;) {
    const s = score[idx[j]];
    while (j < idx.length && score[idx[j]] === s) { if (label[idx[j]]) tp++; else fp++; j++; }
    fpr.push(fp / Nn); tpr.push(tp / P);
  }
  let auc = 0;
  for (let i = 1; i < fpr.length; i++) auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2;
  return { fpr, tpr, auc };
}
/* The fault windows (first and last sample of each), for shading. */
function windows(run) {
  const w = {};
  run.kind.forEach((k, i) => { if (k) { w[k] = w[k] || [i, i]; w[k][1] = i; } });
  return w;
}
function shadeFaults(fr, run, label = false) {
  const { ctx, X, pad, ih } = fr, w = windows(run);
  ctx.font = "600 10px " + css("--font-ui"); ctx.textAlign = "center";
  for (const [k, [a, b]] of Object.entries(w)) {
    ctx.fillStyle = alpha("--geom", 0.08); ctx.fillRect(X(run.t[a]), pad.t, X(run.t[b] + TS) - X(run.t[a]), ih);
    if (label) { ctx.fillStyle = css("--geom"); ctx.fillText(k, (X(run.t[a]) + X(run.t[b] + TS)) / 2, pad.t + 11); }
  }
}

/* =====================================================================
   THE DEMO: THREE DETECTORS, TWO FAULTS
   ===================================================================== */
const hero = { seed: 1, run: null, rule: "product", fused: null };
const DETS = [["su", "--c5", "#hUnc"], ["sn", "--warm", "#hNis"], ["so", "--c4", "#hOdd"]];
function heroUse() { return DETS.map(([, , id]) => $(id).checked); }
function updateHero(resim = false) {
  if (resim || !hero.run) hero.run = scenario(hero.seed);
  const beta = +$("#hB").value, tau = +$("#hT").value;
  $("#hBV").textContent = fmt(beta, 2); $("#hTV").textContent = fmt(tau, 2);
  $$("#demo [data-rule]").forEach(b => b.setAttribute("aria-pressed", b.dataset.rule === hero.rule ? "true" : "false"));
  hero.fused = fuse(hero.run, hero.rule, heroUse(), beta);
  drawHero(); heroStatus();
}
/* first alarm inside each fault window, relative to its onset; the alarm's share of the time outside the faults */
function alarmStats(run, I, tau) {
  const w = windows(run), out = {};
  for (const k of ["blur", "bias"]) {
    const [a, b] = w[k]; let d = null;
    for (let i = a; i <= b; i++) if (I[i] < tau) { d = (i - a) * TS; break; }
    out[k] = d;
  }
  let n = 0, fa = 0, ramps = 0, low = Infinity;
  run.kind.forEach((k, i) => {
    if (k === "bias") low = Math.min(low, I[i]);
    if (k) return;
    n++; if (I[i] < tau) { fa++; if (run.blur[i] > 0) ramps++; }
  });
  return { ...out, fa: fa / n, rampShare: fa ? ramps / fa : 0, low };
}
function drawHero() {
  const r = hero.run, use = heroUse(), tau = +$("#hT").value, narrow = narrowOf($("#scC"));
  const fr = frame($("#scC"), narrow ? 1.6 : 2.8, { x: [0, T_END], y: [0, 1.17], xlabel: "time (s)", yticks: [0, 0.5, 1], yfmt: v => fmt(v, 1) });   // headroom for the fault labels
  shadeFaults(fr, r, true);
  DETS.forEach(([key, col], i) => plotLine(fr, r.t, r[key], use[i] ? { color: css(col), width: 1.5 } : { color: alpha(col, 0.3), width: 1, dash: [4, 4] }));
  const gr = frame($("#intC"), narrow ? 1.9 : 3.2, { x: [0, T_END], y: [0, 1.05], xlabel: "time (s)", yticks: [0, 0.5, 1], yfmt: v => fmt(v, 1) });
  const { ctx, X, Y } = gr;
  shadeFaults(gr, r);
  plotLine(gr, r.t, hero.fused.I, { color: css("--path"), width: 2.2 });
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]); seg(ctx, X(0), Y(tau), X(T_END), Y(tau)); ctx.setLineDash([]);
  ctx.fillStyle = css("--geom");
  hero.fused.I.forEach((v, k) => { if (v < tau) ctx.fillRect(X(r.t[k]), Y(0) - 7, Math.max(1.5, X(TS) - X(0)), 7); });
  const st = alarmStats(r, hero.fused.I, tau), d = v => (v === null ? "missed" : `${fmt(v, 2)} s`);
  $("#rBlur").textContent = d(st.blur); $("#rBias").textContent = d(st.bias);
  $("#rFa").textContent = `${fmt(100 * st.fa, 1)}% of the time`; $("#rLow").textContent = fmt(st.low, 2);
}
function heroStatus() {
  const r = hero.run, use = heroUse(), beta = +$("#hB").value, tau = +$("#hT").value, st = alarmStats(r, hero.fused.I, tau);
  const ruleName = { product: "product", min: "weakest link", average: "average" }[hero.rule];
  const caught = ["blur", "bias"].filter(k => st[k] !== null), parts = [];
  if (!use.some(Boolean)) { caption($("#heroStatus"), "With every detector switched off, the integrity score stays at 1: nothing can be caught."); return; }
  const when = d => (d ? `after ${fmt(d, 2)} s` : "at once");
  parts.push(caught.length === 2 ? `The ${ruleName} catches <b>both faults</b>: the blur ${when(st.blur)}, the bias ${when(st.bias)}.`
    : caught.length === 1 ? `The ${ruleName} catches the ${caught[0]} only, and <b>misses the ${caught[0] === "blur" ? "bias" : "blur"}</b>.`
    : `The ${ruleName} <b>misses both faults</b>.`);
  if (st.bias === null) {
    if (!use[1]) parts.push("Without the NIS, nothing sees the bias: the conditions look perfect, and the camera claims its usual small noise.");
    else if (hero.rule === "average") parts.push(`During the bias, the uncertainty and domain scores stay near 1, so the average never falls below ${fmt(st.low, 2)}: two satisfied detectors mask the one that saw the fault.`);
    else parts.push(`The NIS crosses its gate on only part of the bias samples, and with [[\\beta = ${fmt(beta, 2)}]] the smoothing averages that flickering evidence away (lowest [[I]]: ${fmt(st.low, 2)}).`);
  }
  if (st.blur === null) {
    if (!use[0] && !use[2]) parts.push("Without the uncertainty and domain scores, the blur passes unnoticed: the filter is told the honest noise, so its NIS stays consistent.");
    else if (hero.rule === "average") parts.push("Even the blur is diluted: the NIS score stays at 1 and props the average up.");
  }
  if (st.fa > 0.005) parts.push(`The alarm also sounds ${fmt(100 * st.fa, 1)}% of the time outside the faults, ` +
    (st.rampShare > 0.5 ? "mostly while the blur builds up and clears, still inside the domain (exercise 3)." : "mostly on single NIS spikes that the light smoothing lets through."));
  else if (st.fa > 0) parts.push(`Outside the faults, the alarm sounds only ${fmt(100 * st.fa, 1)}% of the time.`);
  else if (caught.length) parts.push("No false alarm outside the faults.");
  caption($("#heroStatus"), parts.join(" "));
}
$$("#demo [data-rule]").forEach(b => b.addEventListener("click", () => { hero.rule = b.dataset.rule; updateHero(); }));
["#hUnc", "#hNis", "#hOdd"].forEach(id => $(id).addEventListener("change", () => updateHero()));
["#hB", "#hT"].forEach(id => $(id).addEventListener("input", () => updateHero()));
$("#hNew").addEventListener("click", () => { hero.seed++; updateHero(true); });

/* =====================================================================
   PART 1: MEMBERSHIP, MEASURED WITH NOISE
   ===================================================================== */
const ZB = [], ZR = [];
{ const rng = makeRng(3); for (let i = 0; i < 200; i++) { ZB.push(rng.normal(0, 1)); ZR.push(rng.normal(0, 1)); } }
function drawOdd() {
  const bl = +$("#oBl").value, b = +$("#oB").value, a = +$("#oA").value;
  $("#oBlV").textContent = `${fmt(bl, 2)} px`; $("#oBV").textContent = fmt(b, 2); $("#oAV").textContent = String(a);
  const mb = ZB.map(z => Math.max(0, bl + 0.15 * z)), mr = ZR.map(z => clamp(b + 0.05 * z, 0, 1));
  const margins = mb.map((v, i) => pixelMargin(v, mr[i])), inside = margins.map(m => m > 0);
  const soft = margins.map(m => sOdd(m, a));
  // the condition plane
  const cv = $("#oddPlane"), aspect = narrowOf(cv) ? 1.25 : 1.15;
  const fr = frame(cv, aspect, { x: [0, 2.6], y: [0.2, 1], xlabel: "blur (px)", ylabel: "illumination", xfmt: v => fmt(v, 1), yfmt: v => fmt(v, 1) });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--path", 0.1); ctx.fillRect(X(0), Y(1), X(ODD.blur) - X(0), Y(ODD.bright) - Y(1));
  ctx.strokeStyle = css("--path"); ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
  seg(ctx, X(ODD.blur), Y(1), X(ODD.blur), Y(ODD.bright)); seg(ctx, X(0), Y(ODD.bright), X(ODD.blur), Y(ODD.bright)); ctx.setLineDash([]);
  ctx.font = "600 10px " + css("--font-ui"); ctx.fillStyle = css("--path"); ctx.textAlign = "left"; ctx.fillText("domain", X(0.06), Y(ODD.bright) - 6);
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();                  // measurements beyond the axes are cut
  mb.forEach((v, i) => { ctx.fillStyle = alpha(inside[i] ? "--path" : "--geom", 0.55); ctx.beginPath(); ctx.arc(X(v), Y(mr[i]), 2.2, 0, 2 * Math.PI); ctx.fill(); });
  ctx.restore();
  ctx.fillStyle = css("--ink"); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(X(bl), Y(b), 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  // ten seconds of the same measurements
  const tv = Array.from({ length: 200 }, (_, i) => i * TS), gv = $("#oddTime");
  const gr = frame(gv, aspect, { x: [0, 10], y: [-0.05, 1.05], xlabel: "time (s)", yticks: [0, 0.5, 1], yfmt: v => fmt(v, 1) });
  const hard = inside.map(v => (v ? 1 : 0));
  const sx = [], sy = [];
  hard.forEach((v, i) => { sx.push(tv[i], tv[i] + TS); sy.push(v, v); });
  plotLine(gr, sx, sy, { color: alpha("--muted", 0.8), width: 1.2 });
  plotLine(gr, tv, soft, { color: css("--c4"), width: 2 });
  // the caption
  const mTrue = pixelMargin(bl, b), flips = hard.slice(1).filter((v, i) => v !== hard[i]).length;
  const share = inside.filter(Boolean).length / inside.length, mean = soft.reduce((s, v) => s + v, 0) / soft.length;
  const which = 1 - bl / ODD.blur < (b - ODD.bright) / (1 - ODD.bright) ? "the blur guard" : "the illumination guard";
  let text = `True conditions: ${mTrue > 0 ? `<b>inside</b> the domain by a margin of ${fmt(mTrue, 2)}` : mTrue < 0 ? `<b>outside</b> the domain, with a margin of −${fmt(-mTrue, 2)}` : "exactly on the boundary"} (${which} is the tighter). `;
  if (flips === 0) text += `Every measurement lands ${share === 1 ? "inside" : "outside"}, the hard test never flips, and the soft score stays near ${fmt(mean, 2)}. Far from the boundary, both agree.`;
  else text += `Measured with noise, ${Math.round(100 * share)}% of the measurements land inside, and the hard test <b>flips ${flips} times</b> in 10 s. The soft score moves between ${fmt(Math.min(...soft), 2)} and ${fmt(Math.max(...soft), 2)}, around <b>${fmt(mean, 2)}</b>: at every step it says how close to the edge the conditions are, rather than alternating between "inside" and "outside".`;
  if (a >= 15) text += ` With a gain of ${a}, the soft score is nearly as abrupt as the hard test.`;
  else if (a <= 2) text += ` With a gain of ${a}, the score hardly separates inside from outside: even perfect conditions score only [[\\sigma_L(${a}) = ${fmt(sOdd(1, a), 2)}]].`;
  caption($("#oddCap"), text);
}
["#oBl", "#oB", "#oA"].forEach(id => $(id).addEventListener("input", drawOdd));

/* =====================================================================
   PART 2: THE THREE SCORES
   ===================================================================== */
function drawScores() {
  const bl = +$("#sBl").value, nis = +$("#sN").value, b = fogOf(bl), f = noiseFactor(bl, b);
  const trace = (BASE[0] * f) ** 2 + (BASE[1] * f) ** 2, m = pixelMargin(bl, b);
  const su = sUnc(trace), sn = sNis(nis), so = sOdd(m);
  $("#sBlV").textContent = `${fmt(bl, 2)} px`; $("#sNV").textContent = fmt(nis, 1);
  const panel = (id, xr, fn, x, y, col, xlabel, mark, xfmt) => {
    const cv = $(id), xs = Array.from({ length: 241 }, (_, i) => xr[0] + (xr[1] - xr[0]) * i / 240);
    const fr = frame(cv, narrowOf(cv, 360) ? 2 : 1.15, { x: xr, y: [0, 1.05], xlabel, yticks: [0, 0.5, 1], yfmt: v => fmt(v, 1), xfmt, xticks: mark.ticks });
    const { ctx, X, Y } = fr;
    if (mark.at !== undefined) { ctx.strokeStyle = css("--geom"); ctx.lineWidth = 1.2; ctx.setLineDash([4, 4]); seg(ctx, X(mark.at), Y(0), X(mark.at), Y(1.05)); ctx.setLineDash([]); }
    plotLine(fr, xs, xs.map(fn), { color: css(col), width: 2.2 });
    ctx.fillStyle = css(col); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(clamp(x, xr[0], xr[1])), Y(y), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  };
  panel("#scU", [0, 0.08], sUnc, trace, su, "--c5", "trace", { ticks: [0, 0.04, 0.08] }, v => String(v));
  panel("#scN", [0, 30], sNis, nis, sn, "--warm", "NIS", { at: CHI2, ticks: [0, 10, 20, 30] }, v => String(v));
  panel("#scO", [-1, 1], m2 => sOdd(m2), m, so, "--c4", "worst margin", { at: 0, ticks: [-1, 0, 1] }, v => String(v));
  caption($("#scCap"),
    (bl > 0 ? `Blur ${fmt(bl, 2)} px, with fog dimming the image to ${fmt(b, 2)}: the perception reports noise ${fmt(f, 1)} times its clear-weather level, so [[\\mathrm{tr}\\,\\Sigma = ${fmt(trace, 4)}]] and [[s_{\\text{unc}} = ${fmt(su, 2)}]]. The worst margin is the blur's, ${fmt(m, 2)}, so [[s_{\\text{ODD}} = ${fmt(so, 2)}]]${m < 0 ? ", outside the domain" : ""}. `
      : `Clear weather: the perception reports its usual noise, [[\\mathrm{tr}\\,\\Sigma = 0.0005]], so [[s_{\\text{unc}} = ${fmt(su, 3)}]]; every margin is 1, so [[s_{\\text{ODD}} = ${fmt(so, 3)}]]. `) +
    (nis <= CHI2 ? `The NIS, ${fmt(nis, 1)}, is below its gate: [[s_{\\text{NIS}} = 1]]. ` : `The NIS, ${fmt(nis, 1)}, exceeds its gate by ${fmt(nis - CHI2, 1)}: [[s_{\\text{NIS}} = ${fmt(sn, 2)}]]. `) +
    `Their product, the raw integrity: <b>${fmt(su * sn * so, 2)}</b>.`);
}
["#sBl", "#sN"].forEach(id => $(id).addEventListener("input", drawScores));

/* =====================================================================
   PART 3: PRODUCT, WEAKEST LINK, AVERAGE
   ===================================================================== */
let fRule = "product", fuseFrame = null;
const RAMP = [[209, 73, 91], [242, 193, 78], [46, 154, 95]];                        // red, amber, green: 0, 0.5, 1
function rampColour(v) {
  const s = clamp(v, 0, 1) * 2, i = Math.min(1, Math.floor(s)), t = s - i;
  return RAMP[i].map((c, j) => Math.round(c + (RAMP[i + 1][j] - c) * t));
}
const heat = document.createElement("canvas"); heat.width = heat.height = 100;
function drawFuse() {
  const s1 = +$("#fS1").value, s2 = +$("#fS2").value, f2 = (a, b) => RULES[fRule]([a, b]);
  $("#fS1V").textContent = fmt(s1, 2); $("#fS2V").textContent = fmt(s2, 2);
  $$("#fuse [data-frule]").forEach(b => b.setAttribute("aria-pressed", b.dataset.frule === fRule ? "true" : "false"));
  const hc = heat.getContext("2d"), img = hc.createImageData(100, 100);
  for (let j = 0; j < 100; j++) for (let i = 0; i < 100; i++) {
    const [r, g, b] = rampColour(f2((i + 0.5) / 100, 1 - (j + 0.5) / 100));
    img.data.set([r, g, b, 255], 4 * (j * 100 + i));
  }
  hc.putImageData(img, 0, 0);
  const fr = fuseFrame = frame($("#fuseC"), 1, { x: [0, 1], y: [0, 1], xlabel: "s₁", ylabel: "s₂", xticks: [0, 0.5, 1], yticks: [0, 0.5, 1], ygrid: false });
  const { ctx, X, Y } = fr;
  ctx.globalAlpha = 0.85; ctx.drawImage(heat, X(0), Y(1), X(1) - X(0), Y(0) - Y(1)); ctx.globalAlpha = 1;
  // the 0.5 contour
  const line = { product: Array.from({ length: 41 }, (_, i) => { const a = 0.5 + 0.5 * i / 40; return [a, 0.5 / a]; }),
                 min: [[0.5, 1], [0.5, 0.5], [1, 0.5]], average: [[0, 1], [1, 0]] }[fRule];
  ctx.strokeStyle = "#16202a"; ctx.lineWidth = 1.8; ctx.beginPath();
  line.forEach(([a, b], i) => (i ? ctx.lineTo(X(a), Y(b)) : ctx.moveTo(X(a), Y(b)))); ctx.stroke();
  ctx.font = "600 10px " + css("--font-ui"); ctx.fillStyle = "#16202a"; ctx.textAlign = "left";
  const lp = { product: [0.72, 0.5 / 0.72], min: [0.52, 0.93], average: [0.52, 0.48] }[fRule];
  ctx.fillText("0.5", X(lp[0]) + 4, Y(lp[1]) - 4);
  ctx.fillStyle = "#ffffff"; ctx.strokeStyle = "#16202a"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(X(s1), Y(s2), 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  const p = s1 * s2, mn = Math.min(s1, s2), av = (s1 + s2) / 2, den = s1 * s2 + (1 - s1) * (1 - s2);
  const veto = v => (v < 0.5 ? "below one half" : "above one half");
  caption($("#fuseCap"),
    `At [[(s_1, s_2) = (${fmt(s1, 2)}, ${fmt(s2, 2)})]]: the product is <b>${fmt(p, 2)}</b>, the weakest link <b>${fmt(mn, 2)}</b>, the average <b>${fmt(av, 2)}</b>. ` +
    (Math.abs(s1 - s2) > 0.5 && av >= 0.5 && p < 0.5 ? "One detector is alarmed and the other is not: the product and the weakest link veto, while the average would pass. "
      : p < 0.5 && mn >= 0.5 ? "Neither detector is alarmed, yet the product is already below one half: it compounds two moderate doubts, which is right if they come from independent failures and double counting if not. "
      : `All three rules are ${veto(av)} here. `) +
    (den > 0 ? `Bayes' rule on a single "trustworthy" hypothesis, with a prior of one half, would give ${fmt(s1 * s2 / den, 2)} (the box below).` : ""));
}
$$("#fuse [data-frule]").forEach(b => b.addEventListener("click", () => { fRule = b.dataset.frule; drawFuse(); }));
["#fS1", "#fS2"].forEach(id => $(id).addEventListener("input", drawFuse));
function pickFuse(ev) {
  const fr = fuseFrame; if (!fr) return;
  const box = $("#fuseC").getBoundingClientRect(), px = ev.clientX - box.left, py = ev.clientY - box.top;
  const s1 = clamp((px - fr.pad.l) / fr.iw, 0, 1), s2 = clamp(1 - (py - fr.pad.t) / fr.ih, 0, 1);
  $("#fS1").value = s1.toFixed(2); $("#fS2").value = s2.toFixed(2); drawFuse();
}
$("#fuseC").addEventListener("pointerdown", ev => { $("#fuseC").setPointerCapture(ev.pointerId); pickFuse(ev); });
$("#fuseC").addEventListener("pointermove", ev => { if (ev.buttons) pickFuse(ev); });

/* =====================================================================
   PART 4: ROC CURVES
   ===================================================================== */
const rocRun = scenario(1), rocLabel = rocRun.kind.map(Boolean);
const rocFused = fuse(rocRun);
let rocSmooth = true;
function drawRoc() {
  $$("#roc [data-smooth]").forEach(b => b.setAttribute("aria-pressed", (b.dataset.smooth === "1") === rocSmooth ? "true" : "false"));
  const fusedScore = (rocSmooth ? rocFused.I : rocFused.raw).map(v => 1 - v);
  const curves = [["uncertainty", rocRun.su.map(v => 1 - v), "--c5"], ["NIS", rocRun.sn.map(v => 1 - v), "--warm"],
                  ["domain", rocRun.so.map(v => 1 - v), "--c4"], ["fused", fusedScore, "--path"]].map(([name, sc, col]) => ({ name, sc, col, ...roc(sc, rocLabel) }));
  const cv = $("#rocC");
  const fr = frame(cv, 1, { x: [0, 1], y: [0, 1], xlabel: "false-positive rate", ylabel: "true-positive rate", xticks: [0, 0.5, 1], yticks: [0, 0.5, 1] });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = alpha("--ink", 0.4); ctx.lineWidth = 1; ctx.setLineDash([4, 4]); seg(ctx, X(0), Y(0), X(1), Y(1)); ctx.setLineDash([]);
  curves.forEach(c => plotLine(fr, c.fpr, c.tpr, { color: css(c.col), width: c.name === "fused" ? 2.6 : 1.5 }));
  $("#rocLegend").innerHTML = curves.map(c => `<span class="key"><span class="sw" style="background:var(${c.col})"></span>${c.name}, AUC ${fmt(c.auc, 2)}</span>`).join("");
  // the fused detector at a 10% false-positive rate, fault by fault
  const fused = curves[3], neg = fused.sc.filter((_, i) => !rocLabel[i]), thr = Math.max(quantile(neg, 0.9), 1e-3);
  const rate = k => { const s = fused.sc.filter((_, i) => rocRun.kind[i] === k); return s.filter(v => v >= thr).length / s.length; };
  caption($("#rocCap"),
    `On this run: uncertainty ${fmt(curves[0].auc, 2)}, NIS ${fmt(curves[1].auc, 2)}, domain ${fmt(curves[2].auc, 2)}, fused <b>${fmt(fused.auc, 2)}</b>. ` +
    `Set to give 10% false positives, the fused detector flags ${Math.round(100 * rate("blur"))}% of the blur samples and <b>${Math.round(100 * rate("bias"))}%</b> of the bias samples. ` +
    (rocSmooth ? "Switch the smoothing off to see how much of that comes from fusion alone." : "Without the smoothing, fusion alone still covers both faults, but the flickering NIS leaves many bias samples unflagged; the smoothing accumulates them."));
}
$$("#roc [data-smooth]").forEach(b => b.addEventListener("click", () => { rocSmooth = b.dataset.smooth === "1"; drawRoc(); }));

/* =====================================================================
   start-up
   ===================================================================== */
updateHero(true); drawOdd(); drawScores(); drawFuse(); drawRoc();
onRedraw(() => { drawHero(); drawOdd(); drawScores(); drawFuse(); drawRoc(); });
})();
