/* Kalman filters with JAX, chapter 05, A first-order filter in C: the live demos. The repository's
   filtrage() recursion y[k] = z0 y[k-1] + K (1 - z0) u[k-1], z0 = exp(-Te/tau), is run here on its
   recorded signal (and on a step and an impulse) with the time constant and gain adjustable, with the
   frequency response and the repository's own 16-bin histogram routine. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
/* filtrage(): the exact zero-order-hold discretisation of tau y' + y = K u, as in TD3.c (y[0] = 0, input delayed one sample). */
function filtrage(u, K, tau, Te) {
  const z0 = Math.exp(-Te / tau), y = new Array(u.length).fill(0);
  for (let k = 1; k < u.length; k++) y[k] = z0 * y[k - 1] + K * (1 - z0) * u[k - 1];
  return y;
}
/* CalculeHistogramme(): the repository's binning, index = (int)((v - min) / width), clamped to the last bin. */
function histogramme(v, nb = 16) {
  let mn = v[0], mx = v[0]; for (const x of v) { if (x < mn) mn = x; if (x > mx) mx = x; }
  const w = (mx - mn) / nb, h = new Array(nb).fill(0);
  for (const x of v) { let i = Math.trunc((x - mn) / w); if (i >= nb) i = nb - 1; if (i >= 0) h[i]++; }
  return { h, mn, mx, w };
}
/* Magnitude and phase of K / (1 + j w tau), the continuous filter, and of the discrete recursion at frequency f. */
function responseC(f, K, tau) { const wt = 2 * Math.PI * f * tau; return { mag: K / Math.sqrt(1 + wt * wt), phase: -Math.atan(wt) }; }
function responseD(f, K, tau, Te) {   // H(z) = K (1 - z0) z^-1 / (1 - z0 z^-1) at z = e^{j w Te}
  const z0 = Math.exp(-Te / tau), th = 2 * Math.PI * f * Te, c = Math.cos(th), s = Math.sin(th);
  const nr = K * (1 - z0) * c, ni = -K * (1 - z0) * s, dr = 1 - z0 * c, di = z0 * s;   // numerator K(1-z0) e^{-j th}, denominator 1 - z0 e^{-j th}
  const den = dr * dr + di * di, re = (nr * dr + ni * di) / den, im = (ni * dr - nr * di) / den;
  return { mag: Math.hypot(re, im), phase: Math.atan2(im, re) };
}
const rmse = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);
const sd = a => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
/* NET-END */

if (typeof module !== "undefined" && module.exports) { module.exports = { filtrage, histogramme, responseC, responseD, rmse, sd }; return; }

const D = window.KC, Te = D.Te, N = D.input.length, F0 = D.fitted.f;
const st = { tau: 2, K: 1, input: "record" };
const sources = { record: () => D.input, step: () => Array.from({ length: N }, (_, i) => i >= 20 ? 1 : 0), impulse: () => Array.from({ length: N }, (_, i) => i === 20 ? 1 : 0) };
function drawHero() {
  const u = sources[st.input](), y = filtrage(u, st.K, st.tau, Te), t = u.map((_, i) => i * Te), z0 = Math.exp(-Te / st.tau);
  const cv = $("#heroC"), narrow = narrowOf(cv, 600);
  const vals = [...u, ...y, ...(st.input === "record" ? D.clean : [])], lo = Math.min(...vals), hi = Math.max(...vals), pad = 0.1 * (hi - lo || 1);
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, (N - 1) * Te], y: [lo - pad, hi + pad], xlabel: "time [s]", ylabel: st.input === "record" ? "signal" : "input and output" });
  if (st.input === "record") { fr.ctx.fillStyle = alpha("--muted", 0.6); u.forEach((v, i) => { fr.ctx.beginPath(); fr.ctx.arc(fr.X(t[i]), fr.Y(v), 1.6, 0, 2 * Math.PI); fr.ctx.fill(); }); plotLine(fr, t, D.clean, { color: css("--ink"), width: 1.4, dash: [5, 4] }); }
  else plotLine(fr, t, u, { color: css("--ink"), width: 1.6 });
  plotLine(fr, t, y, { color: css("--post"), width: 2 });
  const rc = responseC(F0, st.K, st.tau), rd = responseD(F0, st.K, st.tau, Te), fc = 1 / (2 * Math.PI * st.tau);
  $("#rZ0").textContent = fmt(z0, 4); $("#rFc").textContent = `${fmt(fc, 3)} Hz`; $("#rGain").textContent = `${fmt(rd.mag, 3)} (${fmt(-rd.phase * 180 / Math.PI, 0)}° lag, ${fmt(-rd.phase / (2 * Math.PI * F0) * 1000, 0)} ms)`;
  if (st.input === "record") { const e = rmse(y, D.clean), r = sd(u.map((v, i) => v - y[i])); $("#rErr").textContent = `${fmt(e, 3)} (raw ${fmt(D.fitted.resid_sd, 3)})`; $("#rRes").textContent = fmt(r, 3); }
  else { $("#rErr").textContent = "–"; $("#rRes").textContent = "–"; }
  const settle = st.input === "step" ? y.findIndex(v => v >= 0.632 * st.K) - 20 : null;
  caption($("#heroStatus"), st.input === "record"
    ? `The repository's recording (dots: 250 samples at [[T_e = 0.01]] s, a sine at ${fmt(F0, 2)} Hz of amplitude ${fmt(D.fitted.amp, 2)} with noise of sd ${fmt(D.fitted.resid_sd, 3)}; the fitted sine is dashed) and the output of the C filter (blue) with [[\\tau = ${st.tau}]] s, [[K = ${st.K}]]: [[z_0 = e^{-T_e/\\tau} = ${fmt(z0, 4)}]], cutoff [[1/(2\\pi\\tau) = ${fmt(fc, 3)}]] Hz. At the signal's frequency the filter passes a fraction ${fmt(rd.mag, 3)} of the amplitude with a lag of ${fmt(-rd.phase * 180 / Math.PI, 0)}°. ${st.tau >= 1 ? `This is the repository's setting, and it is a mistake for this signal: the sine is above the cutoff by a factor ${fmt(F0 / fc, 0)} and comes out ${fmt(1 / rd.mag, 0)} times smaller, which the program then prints with two decimals as a column of 0.00 and 0.01. The filter has removed the signal, not the noise.` : st.tau >= 0.1 ? `The signal is attenuated and delayed by a sizeable fraction of its period: the output is a smaller, later sine, further from the clean one (RMSE ${fmt(rmse(y, D.clean), 3)}) than the raw recording is (${fmt(D.fitted.resid_sd, 3)}).` : `The sine passes nearly intact; the output is ${fmt(rmse(y, D.clean), 3)} from the clean sine against ${fmt(D.fitted.resid_sd, 3)} for the raw recording${rmse(y, D.clean) > D.fitted.resid_sd ? `: still no better, because the lag of ${fmt(-rd.phase / (2 * Math.PI * F0) * 1000, 0)} ms costs more than the smoothing saves. A causal first-order filter cannot denoise a signal whose period is only ${fmt(1 / (F0 * Te), 0)} samples without lagging it; the best time constant on this record is below one sample.` : "."}`}`
    : st.input === "step" ? `A unit step at [[t = 0.2]] s and the filter's response with [[\\tau = ${st.tau}]] s: it rises toward [[K = ${st.K}]] and reaches 63% of it after ${settle > 0 ? `${settle} samples (${fmt(settle * Te, 2)} s, the time constant up to the one-sample delay of the recursion)` : "more samples than the record holds"}. The repository's test prints an "impulse" response instead, and with [[\\tau = 2]] s over 0.5 s every printed value rounds to 0.00 or 0.01.`
    : `A unit impulse at [[t = 0.2]] s: the response is [[K(1 - z_0) z_0^{k-1}]], a geometric decay starting at ${fmt(st.K * (1 - z0), 4)}, which is what the repository's first test prints (with [[\\tau = 2]] s the first value is 0.005 and shows as 0.00 or 0.01 at two decimals).`);
}
function drawBode() {
  const cv = $("#bodeC"), narrow = narrowOf(cv), fs = [], mag = [], magD = [], ph = [];
  for (let e = -1; e <= Math.log10(50); e += 0.02) { const f = 10 ** e; fs.push(e); mag.push(20 * Math.log10(responseC(f, st.K, st.tau).mag)); magD.push(20 * Math.log10(responseD(f, st.K, st.tau, Te).mag)); ph.push(responseD(f, st.K, st.tau, Te).phase * 180 / Math.PI); }
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [-1, Math.log10(50)], y: [-60, 5], xlabel: "frequency [Hz], log scale (Nyquist 50 Hz)", ylabel: "gain [dB]", xticks: [-1, 0, 1, Math.log10(50)], xfmt: v => fmt(10 ** v, v < 0 ? 1 : 0), yticks: [-60, -40, -20, 0] });
  plotLine(fr, fs, mag, { color: alpha("--muted", 0.9), width: 1.4, dash: [4, 3] }); plotLine(fr, fs, magD, { color: css("--post"), width: 2 });
  fr.ctx.strokeStyle = css("--warm"); fr.ctx.setLineDash([3, 3]); seg(fr.ctx, fr.X(Math.log10(F0)), fr.pad.t, fr.X(Math.log10(F0)), fr.Y(-60)); seg(fr.ctx, fr.X(Math.log10(1 / (2 * Math.PI * st.tau))), fr.pad.t, fr.X(Math.log10(1 / (2 * Math.PI * st.tau))), fr.Y(-60)); fr.ctx.setLineDash([]);
  const cv2 = $("#phaseC"), fr2 = frame(cv2, narrow ? 1.5 : 2.6, { x: [-1, Math.log10(50)], y: [-180, 0], xlabel: "frequency [Hz], log scale", ylabel: "phase [°]", xticks: [-1, 0, 1, Math.log10(50)], xfmt: v => fmt(10 ** v, v < 0 ? 1 : 0), yticks: [-180, -135, -90, -45, 0] });
  plotLine(fr2, fs, ph, { color: css("--post"), width: 2 });
  fr2.ctx.strokeStyle = css("--warm"); fr2.ctx.setLineDash([3, 3]); seg(fr2.ctx, fr2.X(Math.log10(F0)), fr2.pad.t, fr2.X(Math.log10(F0)), fr2.Y(-180)); fr2.ctx.setLineDash([]);
  const rd = responseD(F0, st.K, st.tau, Te);
  caption($("#bodeCap"), `Frequency response of the filter with [[\\tau = ${st.tau}]] s: the gain of the continuous filter [[K/\\sqrt{1 + (2\\pi f\\tau)^2}]] (grey dashed) and of the C recursion evaluated on the unit circle (blue), which agree far below the Nyquist frequency of 50 Hz and differ near it, where the one-sample delay and the sampling matter; below, the recursion's phase. The orange lines mark the cutoff [[1/(2\\pi\\tau)]] and the signal's ${fmt(F0, 2)} Hz. The filter falls at 20 dB per decade above the cutoff and lags by 45° at it; at the signal's frequency it has a gain of ${fmt(20 * Math.log10(rd.mag), 1)} dB and a phase of ${fmt(rd.phase * 180 / Math.PI, 0)}°, the extra lag beyond the continuous filter's [[-\\arctan(2\\pi f\\tau)]] being the delay [[-360° f T_e = ${fmt(-360 * F0 * Te, 1)}°]] of [[u_{k-1}]].`);
}
function drawHist() {
  const u = D.input, y = filtrage(u, 1, 2, Te), yT = filtrage(u, st.K, st.tau, Te);
  const dRepo = u.map((v, i) => v - y[i]), dNow = u.map((v, i) => v - yT[i]), hr = histogramme(dRepo), hn = histogramme(dNow);
  const cv = $("#histC"), narrow = narrowOf(cv), mx = Math.max(...hr.h, ...hn.h);
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, 16], y: [0, mx * 1.15], xlabel: "bin (16 equal bins between the minimum and the maximum of u − y)", ylabel: "count", xticks: [0, 4, 8, 12, 16] });
  hr.h.forEach((c, i) => { fr.ctx.fillStyle = alpha("--muted", 0.55); fr.ctx.fillRect(fr.X(i + 0.08), fr.Y(c), fr.X(i + 0.5) - fr.X(i + 0.08), fr.Y(0) - fr.Y(c)); });
  hn.h.forEach((c, i) => { fr.ctx.fillStyle = alpha("--post", 0.8); fr.ctx.fillRect(fr.X(i + 0.5), fr.Y(c), fr.X(i + 0.92) - fr.X(i + 0.5), fr.Y(0) - fr.Y(c)); });
  caption($("#histCap"), `The repository's 16-bin histogram routine applied to the difference between input and output, [[u - y]]: with the repository's [[\\tau = 2]] s (grey; bins ${hr.h.join(", ")}, the counts the corrected program prints) and with the page's current [[\\tau = ${st.tau}]] s (blue). With [[\\tau = 2]] the output is nearly zero, so the "difference" is the input itself and its histogram has the two-humped shape of a sine, piling up near the extremes; it says nothing about the noise. With a time constant that lets the sine through (try 0.02 s) the difference is the removed part, a narrow hump of width comparable to the noise. The routine itself is sound; what it was fed was not. In the repository's own run the histogram was computed over 500 entries of arrays that hold 250, and its largest bin counted the uninitialised memory.`);
}
$("#hTau").addEventListener("input", e => { st.tau = +(10 ** +e.target.value).toPrecision(2); $("#hTauV").textContent = `${st.tau} s`; drawAll(); });
$("#hK").addEventListener("input", e => { st.K = +e.target.value; $("#hKV").textContent = fmt(st.K, 2); drawAll(); });
$$("#inSeg .btn").forEach(b => b.addEventListener("click", () => { $$("#inSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); st.input = b.dataset.i; drawHero(); }));
$$("[data-tau]").forEach(b => b.addEventListener("click", () => { st.tau = +b.dataset.tau; $("#hTau").value = Math.log10(st.tau); $("#hTauV").textContent = `${st.tau} s`; drawAll(); }));
$("#hTauV").textContent = "2 s"; $("#hKV").textContent = "1.00";
function drawAll() { drawHero(); drawBode(); drawHist(); }
drawAll(); onRedraw(drawAll);
})();
