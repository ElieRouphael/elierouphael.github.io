/* Kalman filters with JAX, chapter 01, Tracking a vehicle: the live demos. The kalman_vehicle_tracking
   repository's twenty-line NumPy filter is reproduced exactly (textbook covariance update, scalar inverse)
   on its recorded simulation, with the noise matrices and the data adjustable. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
/* The repository's KalmanFilter class for the constant-velocity model, written out for a 2-state, 1-measurement
   filter: predict x = F x, P = F P F' + Q; update with K = P H' / S and the textbook P = P - K H P. */
function runTracker(zs, dt, Q, R, { P0 = [[1, 0], [0, 1]], x0 = [0, 0], joseph = false } = {}) {
  let x = x0.slice(), P = P0.map(r => r.slice()); const est = [], vel = [], gains = [], sds = [];
  for (const z of zs) {
    const xp = [x[0] + dt * x[1], x[1]];
    const p00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt * dt * P[1][1] + Q[0][0], p01 = P[0][1] + dt * P[1][1] + Q[0][1], p11 = P[1][1] + Q[1][1];
    const S = p00 + R, k0 = p00 / S, k1 = p01 / S, innov = z - xp[0];
    x = [xp[0] + k0 * innov, xp[1] + k1 * innov];
    if (joseph) { const a = 1 - k0; P = [[a * a * p00 + k0 * k0 * R, a * (p01 - k1 * p00) + k0 * k1 * R], [a * (p01 - k1 * p00) + k0 * k1 * R, p11 - 2 * k1 * p01 + k1 * k1 * p00 + k1 * k1 * R]]; }
    else P = [[p00 - k0 * p00, p01 - k0 * p01], [p01 - k1 * p00, p11 - k1 * p01]];   // P - K H P, as the repository
    est.push(x[0]); vel.push(x[1]); gains.push([k0, k1]); sds.push(Math.sqrt(Math.max(P[0][0], 0)));
  }
  return { est, vel, gains, sds };
}
const rmse = (a, b, i0 = 0, i1 = a.length) => { let s = 0; for (let i = i0; i < i1; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s / (i1 - i0)); };
/* The true velocity of the recorded chirp 15 sin(t^2) is 30 t cos(t^2); the CSV's "true_velocity" column is 2t. */
const chirpVel = t => 30 * t * Math.cos(t * t);
function makeRng(seed) {
  let s0 = (seed >>> 0) || 1, s1 = 0x9e3779b9 ^ s0;
  const next = () => { let x = s0, y = s1; s0 = y; x ^= x << 23; x ^= x >>> 17; x ^= y ^ (y >>> 26); s1 = x; return ((s0 + s1) >>> 0) / 4294967296; };
  let spare = null;
  const normal = () => { if (spare !== null) { const v = spare; spare = null; return v; } let u1 = 0; while (u1 <= 1e-12) u1 = next(); const u2 = next(), r = Math.sqrt(-2 * Math.log(u1)); spare = r * Math.sin(2 * Math.PI * u2); return r * Math.cos(2 * Math.PI * u2); };
  return { next, normal };
}
/* The repository's generator with the page's random numbers: 15 sin(t^2) plus N(0, sd^2) on the position. */
function generate(n, dt, sd, rng) { const time = [], truth = [], noisy = []; for (let i = 0; i < n; i++) { const t = i * dt, p = 15 * Math.sin(t * t); time.push(t); truth.push(p); noisy.push(p + sd * rng.normal()); } return { time, truth, noisy }; }
/* NET-END */

if (typeof module !== "undefined" && module.exports) { module.exports = { runTracker, rmse, chirpVel, makeRng, generate }; return; }

const D = window.KV, N = D.time.length, DT = D.dt;
const st = { source: "repo", seed: 1, qScale: 1, R: 25, joseph: false, win: 0, winLen: 300 };
let data = null, run = null;
function compute() {
  data = st.source === "repo" ? { time: D.time, truth: D.true, noisy: D.noisy } : generate(N, DT, 5, makeRng(st.seed * 131 + 11));
  const Q = [[1 * st.qScale, 0], [0, 3 * st.qScale]];
  run = runTracker(data.noisy, DT, Q, st.R, { joseph: st.joseph });
  run.r = { meas: rmse(data.noisy, data.truth), kf: rmse(run.est, data.truth), thirds: [0, 1, 2].map(k => [rmse(data.noisy, data.truth, k * 333, k === 2 ? N : (k + 1) * 333), rmse(run.est, data.truth, k * 333, k === 2 ? N : (k + 1) * 333)]) };
  run.r.vel = rmse(run.vel, data.time.map(chirpVel));
}
function drawHero() {
  if (!run) compute();
  const a = st.win, b = Math.min(N, a + st.winLen), t = data.time.slice(a, b);
  const cv = $("#heroP"), narrow = narrowOf(cv, 600);
  const vals = [...data.noisy.slice(a, b), ...data.truth.slice(a, b)];
  const fr = frame(cv, narrow ? 1.5 : 2.4, { x: [t[0], t[t.length - 1]], y: [Math.min(...vals) - 2, Math.max(...vals) + 2], xlabel: "time [s]", ylabel: "position [m]" });
  fr.ctx.fillStyle = alpha("--muted", 0.5); for (let i = a; i < b; i++) { fr.ctx.beginPath(); fr.ctx.arc(fr.X(data.time[i]), fr.Y(data.noisy[i]), 1.5, 0, 2 * Math.PI); fr.ctx.fill(); }
  plotLine(fr, t, data.truth.slice(a, b), { color: css("--ink"), width: 2 });
  plotLine(fr, t, run.est.slice(a, b), { color: css("--post"), width: 1.8 });
  const cvv = $("#heroV"), vt = t.map(chirpVel), vk = run.vel.slice(a, b), csv = D.trueVel.slice(a, b);
  const lo = Math.min(...vt, ...vk), hi = Math.max(...vt, ...vk);
  const frv = frame(cvv, narrow ? 2.4 : 4.5, { x: [t[0], t[t.length - 1]], y: [lo - 5, hi + 5], xlabel: "time [s]", ylabel: "velocity [m/s]" });
  if (st.source === "repo") plotLine(frv, t, csv, { color: alpha("--muted", 0.9), width: 1.2, dash: [4, 3] });
  plotLine(frv, t, vt, { color: css("--ink"), width: 2 }); plotLine(frv, t, vk, { color: css("--post"), width: 1.6 });
  const r = run.r;
  $("#rMeas").textContent = fmt(r.meas, 2); $("#rKF").textContent = fmt(r.kf, 2); $("#rThirds").textContent = r.thirds.map(v => fmt(v[1], 2)).join(" / "); $("#rVel").textContent = fmt(r.vel, 1); $("#rK").textContent = `(${fmt(run.gains[N - 1][0], 3)}, ${fmt(run.gains[N - 1][1], 3)})`;
  const isRepo = st.qScale === 1 && st.R === 25 && !st.joseph && st.source === "repo";
  caption($("#heroStatus"), `${st.source === "repo" ? "The repository's recorded simulation" : `A fresh simulation of the same chirp with the page's random numbers (seed ${st.seed})`}, ${fmt(t[0], 1)} to ${fmt(t[t.length - 1], 1)} s of 10: the true position [[15 \\sin t^2]] (black), the measurements with noise of sd 5 (dots) and the Kalman estimate (blue) with [[Q = ${fmt(st.qScale, 2)} \\times \\mathrm{diag}(1, 3)]], [[R = ${st.R}]]${st.joseph ? ", Joseph-form update" : ""}. Below, the velocity the filter infers (blue) against the true velocity [[30\\,t\\cos t^2]] (black)${st.source === "repo" ? " and the CSV's \"true_velocity\" column [[2t]] (grey dashed), which is the derivative of nothing in the file" : ""}. Over the whole run the RMSE is ${fmt(r.meas, 2)} m for the raw measurements and ${fmt(r.kf, 2)} m for the filter${isRepo ? " (the repository's script, reproduced to machine precision)" : ""}; by thirds of the run, ${r.thirds.map(v => fmt(v[1], 2)).join(", ")} m against ${r.thirds.map(v => fmt(v[0], 2)).join(", ")} m for the measurements. ${r.thirds[2][1] > r.thirds[2][0] ? "In the last third the filter is worse than the sensor it is filtering: the chirp oscillates faster than the constant-velocity model can follow, and the estimate lags." : "The filter beats the sensor in every third."} Its steady-state gain is [[(${fmt(run.gains[N - 1][0], 3)}, ${fmt(run.gains[N - 1][1], 3)})]].`);
}
function drawSweep() {
  const base = st.source === "repo" ? { noisy: D.noisy, truth: D.true } : data;
  const scales = [], errs = []; for (let e = -2; e <= 3; e += 0.25) { const s = 10 ** e; scales.push(e); errs.push(rmse(runTracker(base.noisy, DT, [[s, 0], [0, 3 * s]], st.R, { joseph: st.joseph }).est, base.truth)); }
  const cv = $("#sweepC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [-2, 3], y: [0, Math.max(...errs) * 1.05], xlabel: "log10 of the Q scale (repository: 0)", ylabel: "position RMSE [m]", xticks: [-2, -1, 0, 1, 2, 3] });
  const meas = rmse(base.noisy, base.truth); fr.ctx.strokeStyle = alpha("--ink", 0.5); fr.ctx.setLineDash([4, 4]); seg(fr.ctx, fr.X(-2), fr.Y(meas), fr.X(3), fr.Y(meas)); fr.ctx.setLineDash([]);
  plotLine(fr, scales, errs, { color: css("--post"), width: 2 });
  const best = errs.indexOf(Math.min(...errs));
  fr.ctx.fillStyle = css("--warm"); fr.ctx.beginPath(); fr.ctx.arc(fr.X(scales[best]), fr.Y(errs[best]), 4, 0, 2 * Math.PI); fr.ctx.fill();
  caption($("#sweepCap"), `Position RMSE of the filter as the repository's [[Q = \\mathrm{diag}(1, 3)]] is scaled by a factor from [[10^{-2}]] to [[10^{3}]], with [[R = ${st.R}]]; the dashed line is the raw measurements' RMSE of ${fmt(meas, 2)} m. The repository's choice (scale 1) gives ${fmt(errs[8], 2)} m; the minimum, ${fmt(errs[best], 2)} m, is at a scale of [[10^{${fmt(scales[best], 2)}}]] (orange). A small [[Q]] tells the filter the velocity hardly changes, so it averages over many samples and lags the chirp badly (${fmt(errs[0], 2)} m at scale 0.01); a large [[Q]] tells it to trust each new measurement, so it converges to the sensor itself (${fmt(errs[errs.length - 1], 2)} m at scale 1000). The optimum is where the model's lag and the sensor's noise balance; it is not a property of the sensor but of how wrong the constant-velocity model is for this motion.`);
}
function drawLag() {
  const base = st.source === "repo" ? { time: D.time, noisy: D.noisy, truth: D.true } : data;
  const r = runTracker(base.noisy, DT, [[st.qScale, 0], [0, 3 * st.qScale]], st.R, { joseph: st.joseph });
  const w = 50, errK = [], errM = [], tt = [];
  for (let i = 0; i + w <= N; i += 10) { tt.push(base.time[i + w / 2]); errK.push(rmse(r.est, base.truth, i, i + w)); errM.push(rmse(base.noisy, base.truth, i, i + w)); }
  const freq = tt.map(t => t / Math.PI);   // instantaneous frequency of sin(t^2): 2t rad/s = t/pi Hz
  const cv = $("#lagC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, 10], y: [0, Math.max(...errK, ...errM) * 1.1], xlabel: "time [s]  (instantaneous frequency of the chirp: t/π Hz, so 3.2 Hz at the end)", ylabel: "RMSE over 0.5 s windows [m]", xticks: [0, 2, 4, 6, 8, 10] });
  plotLine(fr, tt, errM, { color: alpha("--muted", 0.9), width: 1.4, dash: [4, 3] }); plotLine(fr, tt, errK, { color: css("--post"), width: 2 });
  const cross = tt.find((_, i) => errK[i] > errM[i] && i > 5);
  caption($("#lagCap"), `The filter's error (blue) and the measurements' error (grey dashed) over half-second windows, with [[Q]] scaled by ${fmt(st.qScale, 2)}. The chirp's frequency grows linearly with time, [[t/\\pi]] Hz, so the motion changes ever faster while the filter's model still assumes a constant velocity. ${cross ? `From about ${fmt(cross, 1)} s the filter is worse than the raw sensor: its lag costs more than its averaging saves.` : "With this much process noise the filter stays ahead of the raw sensor throughout, by trusting the measurements almost as much as the sensor does."} The remedy is a model that knows about acceleration, or a [[Q]] that admits one, which the Q-scale sweep above is a crude version of.`);
}
$("#hQ").addEventListener("input", e => { st.qScale = +(10 ** +e.target.value).toPrecision(2); $("#hQV").textContent = st.qScale; compute(); drawAll(); });
$("#hR").addEventListener("input", e => { st.R = +(10 ** +e.target.value).toPrecision(2); $("#hRV").textContent = st.R; compute(); drawAll(); });
$("#hWin").addEventListener("input", e => { st.win = +e.target.value; $("#hWinV").textContent = `${fmt(st.win * DT, 1)} to ${fmt(Math.min(N, st.win + st.winLen) * DT, 1)} s`; drawHero(); });
$("#hJoseph").addEventListener("click", () => { st.joseph = !st.joseph; $("#hJoseph").setAttribute("aria-pressed", st.joseph); compute(); drawAll(); });
$("#hRepo").addEventListener("click", () => { st.source = "repo"; sync(); compute(); drawAll(); });
$("#hNew").addEventListener("click", () => { st.source = "fresh"; st.seed++; sync(); compute(); drawAll(); });
$("#hReset").addEventListener("click", () => { st.qScale = 1; st.R = 25; st.joseph = false; $("#hQ").value = 0; $("#hR").value = Math.log10(25); $("#hQV").textContent = "1"; $("#hRV").textContent = "25"; $("#hJoseph").setAttribute("aria-pressed", "false"); compute(); drawAll(); });
function sync() { $("#hRepo").setAttribute("aria-pressed", st.source === "repo"); }
$("#hQV").textContent = "1"; $("#hRV").textContent = "25"; $("#hWinV").textContent = "0.0 to 3.0 s";
function drawAll() { drawHero(); drawSweep(); drawLag(); }
drawAll(); onRedraw(drawAll);
})();
