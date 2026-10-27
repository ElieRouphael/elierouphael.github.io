/* Kalman filters with JAX, chapter 03, A real series, next-hour NO2: the live demos. The repository's
   Kalman-MLP forecaster (a scalar latent NO2 state, a learned residual from the state and eight sensor and
   weather inputs, a Kalman update with the next hour's reading) is re-run here on the test split with the
   parameters JAX trained, against persistence and linear baselines; the filter's noise matrices can be
   changed live. Data and trained parameters come from ch03-data.js. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
function mlp(params, inp) {
  let a = inp;
  for (let l = 0; l < params.length; l++) { const [W, b] = params[l]; const z = W.map((row, i) => row.reduce((s, w, j) => s + w * a[j], b[i])); a = l < params.length - 1 ? z.map(Math.tanh) : z; }
  return a;
}
/* The repository's forecaster for a scalar state: x_pred = A x + f(x, u), y_pred = x_pred, then a Kalman update
   with the next hour's standardised reading. Returns forecasts, filtered states and the forecast variance. */
function kalmanForecast(params, A, Q, R, x0, us, ys) {
  let x = x0, P = 1; const yPred = [], yFilt = [], Sp = [], Ks = [];
  for (let t = 0; t < us.length; t++) {
    const xp = A * x + mlp(params, [x, ...us[t]])[0], Pp = A * A * P + Q, S = Pp + R, K = Pp / S;
    x = xp + K * (ys[t] - xp); P = (1 - K) * (1 - K) * Pp + K * K * R;   // Joseph form, scalar
    yPred.push(xp); yFilt.push(x); Sp.push(S); Ks.push(K);
  }
  return { yPred, yFilt, Sp, Ks };
}
const linear = (w, feats) => feats.map(f => f.reduce((s, v, j) => s + w[j] * v, w[w.length - 1]));
const rmse = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);
const mae = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length;
/* RMSE grouped by hour of day (timestamps 'YYYY-MM-DDTHH:MM', the hour of the *current* reading). */
function byHourRmse(ts, truth, pred) {
  const se = new Array(24).fill(0), n = new Array(24).fill(0);
  ts.forEach((t, i) => { const h = +t.slice(11, 13); se[h] += (truth[i] - pred[i]) ** 2; n[h]++; });
  return se.map((v, h) => n[h] ? Math.sqrt(v / n[h]) : NaN);
}
/* Steady-state gain of the scalar filter: fixed point of P = (1-K)^2 (A^2 P + Q) + K^2 R. */
function steadyGain(A, Q, R) { let P = 1, K = 0; for (let i = 0; i < 2000; i++) { const Pp = A * A * P + Q; K = Pp / (Pp + R); P = (1 - K) * (1 - K) * Pp + K * K * R; } return K; }
/* NET-END */

if (typeof module !== "undefined" && module.exports) { module.exports = { mlp, kalmanForecast, linear, rmse, mae, byHourRmse, steadyGain }; return; }

const D = window.KJ3, T = D.test, N = T.timestamps.length, inv = v => v * D.targetStd + D.targetMean;
const truth = T.futureRaw, persist = T.currentRaw;
const linPred = linear(D.wLin, T.controls).map(inv), arPred = linear(D.wAr, T.controls.map((c, i) => [...c, T.current[i]])).map(inv);
const MODELS = Object.keys(D.models);
const st = { model: MODELS[0], Q: D.models[MODELS[0]].Q, R: D.models[MODELS[0]].R, A: D.models[MODELS[0]].A, start: 0, show: { persist: true, ar: true, kf: true } };
let cur = null;
function run() {
  const m = D.models[st.model];
  const kf = kalmanForecast(m.params, st.A, st.Q, st.R, T.current[0], T.controls, T.future);
  const pred = kf.yPred.map(inv), filt = kf.yFilt.map(inv), sd = kf.Sp.map(s => Math.sqrt(s) * D.targetStd);
  cur = { kf, pred, filt, sd, r: { kf: rmse(truth, pred), persist: rmse(truth, persist), lin: rmse(truth, linPred), ar: rmse(truth, arPred), clim: rmse(truth, truth.map(() => D.targetMean)) }, m: { kf: mae(truth, pred), persist: mae(truth, persist), ar: mae(truth, arPred) }, lagged: rmse(persist, pred), Kss: steadyGain(st.A, st.Q, st.R), cover: pred.reduce((s, p, i) => s + (Math.abs(truth[i] - p) <= 1.96 * sd[i] ? 1 : 0), 0) / N };
}
function drawHero() {
  if (!cur) run();
  const a = st.start, b = Math.min(N, a + 168), idx = Array.from({ length: b - a }, (_, i) => a + i), x = idx.map(i => i - a);
  const cv = $("#heroC"), narrow = narrowOf(cv, 600);
  const vals = idx.flatMap(i => [truth[i], cur.pred[i], persist[i]]), lo = Math.max(0, Math.min(...vals) - 10), hi = Math.max(...vals) + 10;
  const fr = frame(cv, narrow ? 1.4 : 2.4, { x: [0, 167], y: [lo, hi], xlabel: `hours from ${T.timestamps[a].replace("T", " ")}`, ylabel: "NO2 next hour [µg/m³]", xticks: [0, 24, 48, 72, 96, 120, 144] });
  if (st.show.kf) { fr.ctx.save(); fr.ctx.beginPath(); fr.ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); fr.ctx.clip(); fr.ctx.fillStyle = alpha("--post", 0.15); fr.ctx.beginPath(); idx.forEach((i, k) => { const v = cur.pred[i] + 1.96 * cur.sd[i]; k ? fr.ctx.lineTo(fr.X(x[k]), fr.Y(v)) : fr.ctx.moveTo(fr.X(x[k]), fr.Y(v)); }); for (let k = idx.length - 1; k >= 0; k--) fr.ctx.lineTo(fr.X(x[k]), fr.Y(cur.pred[idx[k]] - 1.96 * cur.sd[idx[k]])); fr.ctx.closePath(); fr.ctx.fill(); fr.ctx.restore(); }
  /* gaps in the record (more than one hour between rows) as light bands */
  fr.ctx.fillStyle = alpha("--muted", 0.18); idx.forEach((i, k) => { if (k && Date.parse(T.timestamps[i] + ":00Z") - Date.parse(T.timestamps[i - 1] + ":00Z") > 3600e3) fr.ctx.fillRect(fr.X(x[k] - 0.5) - 1, fr.pad.t, 3, fr.ih); });
  if (st.show.persist) plotLine(fr, x, idx.map(i => persist[i]), { color: alpha("--muted", 0.9), width: 1.3, dash: [4, 3] });
  if (st.show.ar) plotLine(fr, x, idx.map(i => arPred[i]), { color: css("--c4"), width: 1.4 });
  if (st.show.kf) plotLine(fr, x, idx.map(i => cur.pred[i]), { color: css("--post"), width: 1.8 });
  plotLine(fr, x, idx.map(i => truth[i]), { color: css("--ink"), width: 2 });
  const r = cur.r; $("#rKF").textContent = `${fmt(r.kf, 2)} / ${fmt(cur.m.kf, 2)}`; $("#rPer").textContent = `${fmt(r.persist, 2)} / ${fmt(cur.m.persist, 2)}`; $("#rAR").textContent = `${fmt(r.ar, 2)} / ${fmt(cur.m.ar, 2)}`; $("#rLin").textContent = fmt(r.lin, 2); $("#rK").textContent = fmt(cur.Kss, 3); $("#rCov").textContent = `${fmt(100 * cur.cover, 0)}%`;
  const m = D.models[st.model], isRepo = st.Q === m.Q && st.R === m.R && st.A === m.A;
  caption($("#heroStatus"), `One week of the test split (${T.timestamps[a].slice(0, 10)} onward; the split runs from 2005-02-16 to 2005-04-04, ${N} hours in chronological order, grey bands where the record has gaps): the measured next-hour NO2 (black), persistence (grey dashed), the linear model with the current NO2 as a ninth input (green), and the Kalman-MLP forecast (blue) with its ±1.96σ band. Over the whole split the RMSE is ${fmt(r.persist, 1)} for persistence, ${fmt(r.ar, 1)} for the linear model with the current reading, ${fmt(r.lin, 1)} for the linear model from the eight sensors alone, and ${fmt(r.kf, 1)} for the Kalman-MLP${isRepo ? ` (JAX: ${fmt(m.test, 1)})` : ` with [[A = ${st.A}]], [[Q = ${st.Q}]], [[R = ${st.R}]] (trained with [[A = ${m.A}]], [[Q = ${m.Q}]], [[R = ${m.R}]])`}, against ${fmt(r.clim, 1)} for the training mean. The learned filter loses to the simplest baseline: its forecast is closer to the current hour's value than to the next (RMSE ${fmt(cur.lagged, 1)} against the current reading)${cur.Kss < 0.7 ? `, because its state is a filtered, hence lagged, copy of the series, and the steady-state gain ${fmt(cur.Kss, 2)} only moves the state part of the way to each new reading` : `; with a gain of ${fmt(cur.Kss, 2)} the state is nearly the current reading, so what remains is the network itself, which fitted its training year rather than these test weeks`}. The band covers ${fmt(100 * cur.cover, 0)}% of the outcomes${cur.cover > 0.97 ? ", far more than the nominal 95%: the variance Q + R was never fitted to the data" : cur.cover < 0.9 ? ", fewer than the nominal 95%" : ", close to the nominal 95%"}.`);
}
function drawHours() {
  if (!cur) run();
  const hK = byHourRmse(T.timestamps, truth, cur.pred), hP = byHourRmse(T.timestamps, truth, persist), hA = byHourRmse(T.timestamps, truth, arPred);
  const cv = $("#hourC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.5 : 2.4, { x: [0, 23], y: [0, Math.max(...hK, ...hP, ...hA) * 1.1], xlabel: "hour of day of the current reading", ylabel: "RMSE of the next-hour forecast", xticks: [0, 6, 12, 18, 23] });
  const h = Array.from({ length: 24 }, (_, i) => i);
  plotLine(fr, h, hP, { color: alpha("--muted", 0.9), width: 1.6, dash: [4, 3] }); plotLine(fr, h, hA, { color: css("--c4"), width: 1.6 }); plotLine(fr, h, hK, { color: css("--post"), width: 2 });
  const cv2 = $("#diurnC"), fr2 = frame(cv2, narrow ? 1.5 : 2.4, { x: [0, 23], y: [0, Math.max(...D.byHour) * 1.15], xlabel: "hour of day", ylabel: "mean NO2 (train split) [µg/m³]", xticks: [0, 6, 12, 18, 23] });
  plotLine(fr2, h, D.byHour, { color: css("--warm"), width: 2 });
  const worstK = hK.indexOf(Math.max(...hK)), worstP = hP.indexOf(Math.max(...hP));
  const jump = D.byHour.map((v, i) => i ? v - D.byHour[i - 1] : 0), up = jump.indexOf(Math.max(...jump)), down = jump.indexOf(Math.min(...jump));
  caption($("#hourCap"), `Left: the test RMSE by hour of day for persistence (grey dashed), the linear model with the current reading (green) and the Kalman-MLP (blue). Right: the mean NO2 by hour of day on the training split, the diurnal cycle of a city street (the station is in an Italian city). All three forecasters are worst when the series moves fastest, in the morning rise (largest hourly jump at ${up}:00, +${fmt(jump[up], 0)} µg/m³) and the evening fall (${down}:00, ${fmt(jump[down], 0)}); persistence is worst at ${worstP}:00 and the Kalman-MLP at ${worstK}:00. A forecaster that knew the time of day could anticipate these moves; none of the three is given it, and the Kalman-MLP's lag hurts most exactly where the series turns.`);
}
function drawScatter() {
  if (!cur) run();
  const cv = $("#scatC"), narrow = narrowOf(cv), lim = 300;
  const fr = frame(cv, narrow ? 1 : 1.1, { x: [0, lim], y: [0, lim], xlabel: "measured next-hour NO2", ylabel: "forecast", xticks: [0, 100, 200, 300], yticks: [0, 100, 200, 300] });
  fr.ctx.strokeStyle = alpha("--ink", 0.4); seg(fr.ctx, fr.X(0), fr.Y(0), fr.X(lim), fr.Y(lim));
  fr.ctx.fillStyle = alpha("--muted", 0.45); truth.forEach((v, i) => { fr.ctx.beginPath(); fr.ctx.arc(fr.X(Math.min(lim, v)), fr.Y(Math.min(lim, persist[i])), 1.6, 0, 2 * Math.PI); fr.ctx.fill(); });
  fr.ctx.fillStyle = alpha("--post", 0.6); truth.forEach((v, i) => { fr.ctx.beginPath(); fr.ctx.arc(fr.X(Math.min(lim, v)), fr.Y(Math.min(lim, cur.pred[i])), 1.6, 0, 2 * Math.PI); fr.ctx.fill(); });
  const slope = (p) => { const mx = truth.reduce((s, v) => s + v, 0) / N, my = p.reduce((s, v) => s + v, 0) / N; let sxy = 0, sxx = 0; truth.forEach((v, i) => { sxy += (v - mx) * (p[i] - my); sxx += (v - mx) ** 2; }); return sxy / sxx; };
  caption($("#scatCap"), `Forecast against outcome on the ${N} test hours: persistence (grey) and the Kalman-MLP (blue). Persistence scatters symmetrically about the diagonal; the Kalman-MLP's cloud is tilted, with a regression slope of ${fmt(slope(cur.pred), 2)} against ${fmt(slope(persist), 2)} for persistence: it under-predicts the high values and over-predicts the low ones, the signature of a smoother that shrinks toward the recent past.`);
}
$$("#mSeg .btn").forEach(b => b.addEventListener("click", () => { $$("#mSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); st.model = b.dataset.m; const m = D.models[st.model]; st.Q = m.Q; st.R = m.R; st.A = m.A; syncSliders(); run(); drawAll(); }));
$$("#showSeg .btn").forEach(b => b.addEventListener("click", () => { st.show[b.dataset.s] = !st.show[b.dataset.s]; b.setAttribute("aria-pressed", st.show[b.dataset.s]); drawHero(); }));
$("#hQ").addEventListener("input", e => { st.Q = +(10 ** +e.target.value).toPrecision(2); $("#hQV").textContent = st.Q; run(); drawAll(); });
$("#hR").addEventListener("input", e => { st.R = +(10 ** +e.target.value).toPrecision(2); $("#hRV").textContent = st.R; run(); drawAll(); });
$("#hA").addEventListener("input", e => { st.A = +e.target.value; $("#hAV").textContent = fmt(st.A, 2); run(); drawAll(); });
$("#hWeek").addEventListener("input", e => { st.start = Math.min(N - 168, +e.target.value * 168); $("#hWeekV").textContent = T.timestamps[st.start].slice(0, 10); drawHero(); });
function syncSliders() { $("#hQ").value = Math.log10(st.Q); $("#hR").value = Math.log10(st.R); $("#hA").value = st.A; $("#hQV").textContent = st.Q; $("#hRV").textContent = st.R; $("#hAV").textContent = fmt(st.A, 2); }
syncSliders(); $("#hWeekV").textContent = T.timestamps[0].slice(0, 10); $("#hWeek").max = Math.floor((N - 168) / 168);
function fillTable() {
  const rows = [["Training mean (climatology)", D.baselines.val.climatology, D.baselines.test.climatology], ["Linear, eight sensor inputs", D.baselines.val.linear, D.baselines.test.linear], ["Persistence (next hour = this hour)", D.baselines.val.persistence, D.baselines.test.persistence], ["Linear, eight inputs + current NO2", D.baselines.val.linear_ar, D.baselines.test.linear_ar]];
  for (const k of MODELS) { const m = D.models[k]; rows.push([`Kalman-MLP, ${k} (A=${m.A}, Q=${m.Q}, R=${m.R}, ${m.steps} steps)`, [m.val, NaN], [m.test, NaN]]); }
  $("#baseT tbody").innerHTML = rows.map(r => `<tr><td>${r[0]}</td><td>${fmt(r[1][0], 1)}</td><td>${fmt(r[2][0], 1)}</td><td>${isFinite(r[2][1]) ? fmt(r[2][1], 1) : "–"}</td></tr>`).join("");
}
function drawLoss() {
  const cv = $("#lossC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, 3000], y: [0.1, 0.45], xlabel: "Adam step", ylabel: "training loss (standardised MSE)", xticks: [0, 1000, 2000, 3000] });
  MODELS.forEach((k, j) => { const m = D.models[k], step = m.steps / m.losses.length; plotLine(fr, m.losses.map((_, i) => i * step), m.losses, { color: j ? css("--warm") : css("--post"), width: 1.8 }); });
  const pm = 0.2361;   // persistence's standardised MSE on the training split, from the Python check
  fr.ctx.strokeStyle = alpha("--ink", 0.6); fr.ctx.setLineDash([4, 4]); seg(fr.ctx, fr.X(0), fr.Y(pm), fr.X(3000), fr.Y(pm)); fr.ctx.setLineDash([]);
  caption($("#lossCap"), `The one-step forecast loss during training, in standardised units, for the repository's settings (blue, 1500 steps) and the retuned noise matrices (orange, 3000 steps); the dashed line is persistence's standardised MSE on the same training split, 0.236. The repository's run ends at ${fmt(D.models[MODELS[0]].losses.slice(-1)[0], 3)}, barely below persistence after 1500 steps; the retuned run reaches ${fmt(D.models[MODELS[1]].losses.slice(-1)[0], 3)} on the training split but ${fmt(D.models[MODELS[1]].test, 1)} RMSE on the test split, still behind persistence's ${fmt(D.baselines.test.persistence[0], 1)}. The gap between training loss and test RMSE is the usual one: a network with ${D.models[MODELS[0]].params.reduce((s, [W, b]) => s + W.length * W[0].length + b.length, 0)} parameters fitted on 5173 hours of one winter and spring, tested on the following seven weeks.`);
}
function drawAll() { drawHero(); drawHours(); drawScatter(); drawLoss(); }
fillTable(); drawAll(); onRedraw(drawAll);
})();
