/* Kalman filters with JAX, chapter 05, A first-order filter in C: the live demos. The repository's
   filtrage() recursion y[k] = z0 y[k-1] + K (1 - z0) u[k-1], z0 = exp(-Te/tau), is run here on its
   recorded signal (and on a step and an impulse) with the time constant and gain adjustable, beside the
   same recursion fed the current sample u[k]; then the frequency responses of both and of the continuous
   filter, and the repository's own 16-bin histogram routine. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;

/* NET-BEGIN  (self-contained: this block is also exported for Node, below, so it can be tested outside the page) */
/* filtrage(): the exact zero-order-hold discretisation of tau y' + y = K u, as in TD3.c (y[0] = 0, input delayed one sample). */
function filtrage(u, K, tau, Te) {
  const z0 = Math.exp(-Te / tau), y = new Array(u.length).fill(0);
  for (let k = 1; k < u.length; k++) y[k] = z0 * y[k - 1] + K * (1 - z0) * u[k - 1];
  return y;
}
/* The same recursion fed the current sample, y[k] = z0 y[k-1] + K (1 - z0) u[k], starting at rest: the exponential-smoothing form. */
function filtrageK(u, K, tau, Te) {
  const z0 = Math.exp(-Te / tau), b = K * (1 - z0), y = new Array(u.length).fill(0);
  let prev = 0;
  for (let k = 0; k < u.length; k++) { prev = z0 * prev + b * u[k]; y[k] = prev; }
  return y;
}
/* CalculeHistogramme(): the repository's binning, index = (int)((v - min) / width), clamped to the last bin. */
function histogramme(v, nb = 16) {
  let mn = v[0], mx = v[0]; for (const x of v) { if (x < mn) mn = x; if (x > mx) mx = x; }
  const w = (mx - mn) / nb, h = new Array(nb).fill(0);
  for (const x of v) { let i = Math.trunc((x - mn) / w); if (i >= nb) i = nb - 1; if (i >= 0) h[i]++; }
  return { h, mn, mx, w };
}
/* Magnitude and phase of K / (1 + j w tau), the continuous filter, and of the two recursions at frequency f. */
function responseC(f, K, tau) { const wt = 2 * Math.PI * f * tau; return { mag: K / Math.sqrt(1 + wt * wt), phase: -Math.atan(wt) }; }
function responseD(f, K, tau, Te, current = false) {   // H(z) = K (1 - z0) z^-1 / (1 - z0 z^-1) at z = e^{j w Te}; without the z^-1 for u_k
  const z0 = Math.exp(-Te / tau), th = 2 * Math.PI * f * Te, c = Math.cos(th), s = Math.sin(th);
  const nr = K * (1 - z0) * (current ? 1 : c), ni = current ? 0 : -K * (1 - z0) * s, dr = 1 - z0 * c, di = z0 * s;
  const den = dr * dr + di * di, re = (nr * dr + ni * di) / den, im = (ni * dr - nr * di) / den;
  let phase = Math.atan2(im, re);
  if (!current && phase > 0) phase -= 2 * Math.PI;    // the delayed form lags toward -180° at the Nyquist frequency
  return { mag: Math.hypot(re, im), phase };
}
const rmse = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);
const sd = a => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
/* NET-END */

if (typeof module !== "undefined" && module.exports) { module.exports = { filtrage, filtrageK, histogramme, responseC, responseD, rmse, sd }; return; }

const D = window.KC, Te = D.Te, N = D.input.length, F0 = D.fitted.f, RAW = rmse(D.input, D.clean), LOG_NYQ = Math.log10(0.5 / Te);
const deg = r => r * 180 / Math.PI;

/* The equivalence check: this page's filtrage() against the program's, compiled with gcc and printed with 8 decimals. */
const cMax = Math.max(...Object.entries(D.c).map(([tau, yc]) => { const y = filtrage(D.input, 1, +tau, Te); return Math.max(...yc.map((v, i) => Math.abs(v - y[i]))); }));
$("#cCheck").textContent = `Checked as the page loaded against the program's filtrage() compiled with gcc 13.2, for τ = ${Object.keys(D.c).join(", ")} s: the largest difference is ${cMax.toExponential(1)}, ${cMax <= 5.01e-9 ? "within" : "outside"} the 8 decimals the C output was printed with.`;

/* The best time constant of each form at K = 1, on a log grid of 601 values from 0.0032 to 3.2 s. */
function best(f) { let b = { tau: 0, err: Infinity }; for (let i = 0; i <= 600; i++) { const tau = 10 ** (-2.5 + i * 0.005), e = rmse(f(D.input, 1, tau, Te), D.clean); if (e < b.err) b = { tau, err: e }; } return b; }
const BEST = { zoh: best(filtrage), cur: best(filtrageK) };

const st = { tau: 2, K: 1, input: "record" };
const sources = { record: () => D.input, step: () => Array.from({ length: N }, (_, i) => i >= 20 ? 1 : 0), impulse: () => Array.from({ length: N }, (_, i) => i === 20 ? 1 : 0) };
function recordNote(rd, lag, ms, e, e2, e1) {   // the hero caption's second half, for the recording
  const d = Math.abs(e - RAW) < 0.001 ? 4 : 3, vs = `the output is ${fmt(e, d)} from the clean sine against ${fmt(RAW, d)} for the raw recording`;
  const pct = `${fmt(100 * Math.abs(1 - e2 / RAW), 0)}% ${e2 < RAW ? "below" : "above"} the recording`;
  const best = `The [[u_k]] form's best, at [[K = 1]], is ${fmt(BEST.cur.err, 3)}, at [[\\tau \\approx ${fmt(BEST.cur.tau, 3)}]] s (Part 5).`;
  if (st.tau >= 1) return `${st.tau === 2 ? (st.K === 1 ? "This is the repository's setting, and it is" : "This is the repository's time constant, and it is") : "Like the repository's 2 s, this time constant is"} a mistake for this signal: the sine is above the cutoff by a factor ${fmt(2 * Math.PI * F0 * st.tau, 0)} and comes out ${fmt(1 / rd.mag, 0)} times smaller, on top of a start-up transient of about the same size, since the filter starts at rest. At [[\\tau = 2]] s and [[K = 1]] the program, once bugs 1 and 2 are fixed, writes the output at two decimals as values between −0.05 and 0.14. The filter has removed the noise, and most of the signal with it.`;
  if (st.K !== 1) return `That gain is [[K = ${st.K}]] times its value at [[K = 1]], so ${vs} (the [[u_k]] form: ${fmt(e2, 3)}; at [[K = 1]]: ${fmt(e1, 3)}). Set [[K = 1]] to see what the time constant alone costs.`;
  if (st.tau >= 0.1) return `The signal is attenuated and delayed by a sizeable fraction of its period: the output is a smaller, later sine, further from the clean one (RMSE ${fmt(e, 3)}) than the raw recording is (${fmt(RAW, 3)}).`;
  const head = lag < 10 ? "The sine passes nearly intact" : `The sine comes out at ${fmt(rd.mag, 2)} of its amplitude and ${fmt(lag, 0)}° late`;
  if (e <= RAW) return `${head}; ${vs}: slightly closer, near the best this form can do at [[K = 1]] (${fmt(BEST.zoh.err, 4)} at [[\\tau \\approx ${fmt(BEST.zoh.tau, 4)}]] s). Fed the current sample [[u_k]], the same recursion gives ${fmt(e2, 3)}, ${pct}. ${best}`;
  if (e2 < RAW) return `${head}; ${vs}: further, because the recursion feeds on [[u_{k-1}]] and so forecasts the signal one sample ahead. Fed the current sample [[u_k]], the same recursion gives ${fmt(e2, 3)}, ${pct}. ${best}`;
  return `${head}; ${vs}: further, because averaging over about ${fmt(st.tau / Te, 0)} samples delays the sine by ${fmt(ms, 0)} ms in all, one sample of it from the recursion's [[u_{k-1}]]. Fed the current sample [[u_k]], it gives ${fmt(e2, 3)}, still ${pct}. ${best}`;
}
function drawHero() {
  const u = sources[st.input](), y = filtrage(u, st.K, st.tau, Te), t = u.map((_, i) => i * Te), z0 = Math.exp(-Te / st.tau);
  const cv = $("#heroC"), narrow = narrowOf(cv, 600);
  const vals = [...u, ...y, ...(st.input === "record" ? D.clean : [])], lo = Math.min(...vals), hi = Math.max(...vals), pad = 0.1 * (hi - lo || 1);
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, (N - 1) * Te], y: [lo - pad, hi + pad], xlabel: "time [s]", ylabel: st.input === "record" ? "signal" : "input and output" });
  if (st.input === "record") { fr.ctx.fillStyle = alpha("--muted", 0.6); u.forEach((v, i) => { fr.ctx.beginPath(); fr.ctx.arc(fr.X(t[i]), fr.Y(v), 1.6, 0, 2 * Math.PI); fr.ctx.fill(); }); plotLine(fr, t, D.clean, { color: css("--ink"), width: 1.4, dash: [5, 4] }); }
  else plotLine(fr, t, u, { color: css("--ink"), width: 1.6 });
  plotLine(fr, t, y, { color: css("--post"), width: 2 });
  $("#lgRec").hidden = st.input !== "record"; $("#lgInkT").textContent = st.input === "record" ? "fitted clean sine" : `unit ${st.input} u`;
  cv.setAttribute("aria-label", st.input === "record" ? "The recorded noisy sine as dots, the fitted clean sine dashed, and the output of the first-order filter as a line" : `A unit ${st.input} at 0.2 s and the output of the first-order filter, both as lines`);
  const rd = responseD(F0, st.K, st.tau, Te), fc = 1 / (2 * Math.PI * st.tau), lag = -deg(rd.phase), ms = lag / 360 / F0 * 1000;
  $("#rZ0").textContent = fmt(z0, 4); $("#rFc").textContent = `${fmt(fc, 3)} Hz`; $("#rGain").textContent = `${fmt(rd.mag, 3)} (${fmt(lag, 0)}° lag, ${fmt(ms, 0)} ms)`;
  let e = null, e2 = null, e1 = null;
  if (st.input === "record") { e = rmse(y, D.clean); e2 = rmse(filtrageK(u, st.K, st.tau, Te), D.clean); e1 = st.K === 1 ? e : rmse(filtrage(u, 1, st.tau, Te), D.clean); $("#rErr").textContent = `${fmt(e, 3)}, ${fmt(e2, 3)} (raw ${fmt(RAW, 3)})`; $("#rRes").textContent = fmt(sd(u.map((v, i) => v - y[i])), 3); }
  else { $("#rErr").textContent = "–"; $("#rRes").textContent = "–"; }
  const settle = st.input === "step" ? y.findIndex(v => v >= 0.632 * st.K) - 20 : null;
  caption($("#heroStatus"), st.input === "record"
    ? `The repository's recording (dots: 250 samples at [[T_e = 0.01]] s, a sine at ${fmt(F0, 2)} Hz of amplitude ${fmt(D.fitted.amp, 2)} with noise of sd ${fmt(D.fitted.resid_sd, 3)}; the fitted sine is dashed) and the output of the C filter (blue) with [[\\tau = ${st.tau}]] s, [[K = ${st.K}]]: [[z_0 = e^{-T_e/\\tau} = ${fmt(z0, 4)}]], cutoff [[1/(2\\pi\\tau) = ${fmt(fc, 3)}]] Hz. At the signal's frequency the filter's gain is ${fmt(rd.mag, 3)} and its lag ${fmt(lag, 0)}°. ${recordNote(rd, lag, ms, e, e2, e1)}`
    : st.input === "step" ? `A unit step at [[t = 0.2]] s and the filter's response with [[\\tau = ${st.tau}]] s: it rises toward [[K = ${st.K}]] and reaches 63% of it after ${settle > 0 ? `${settle} sample${settle === 1 ? "" : "s"} (${fmt(settle * Te, 2)} s): the recursion matches the continuous response [[K(1 - e^{-(t - 0.2)/\\tau})]] at every sample, so this is the time constant, rounded up to a whole number of samples` : "more samples than remain in the record"}. The repository's test feeds an impulse instead, and with [[\\tau = 2]] s every one of its 51 printed outputs rounds to 0.00.`
    : `A unit impulse at [[t = 0.2]] s: [[k]] samples after it the response is [[K(1 - z_0) z_0^{k-1}]], a geometric decay starting at ${(st.K * (1 - z0)).toPrecision(3)}, which is what the repository's self-test computes (with [[\\tau = 2]] s the first value is 0.00499, printed at two decimals as 0.00 like all the others).`);
}
function drawBode() {
  const cv = $("#bodeC"), narrow = narrowOf(cv), fs = [], mag = [], magD = [], phC = [], phD = [], phK = [];
  const es = []; for (let i = 0; -2 + i * 0.02 < LOG_NYQ; i++) es.push(-2 + i * 0.02); es.push(LOG_NYQ);
  for (const e of es) {
    const f = 10 ** e, rc = responseC(f, st.K, st.tau), rd = responseD(f, st.K, st.tau, Te), rk = responseD(f, st.K, st.tau, Te, true);
    fs.push(e); mag.push(20 * Math.log10(rc.mag)); magD.push(20 * Math.log10(rd.mag)); phC.push(deg(rc.phase)); phD.push(deg(rd.phase)); phK.push(deg(rk.phase));
  }
  const xt = [-2, -1, 0, 1, LOG_NYQ], xf = v => fmt(10 ** v, v < -1.5 ? 2 : v < 0 ? 1 : 0);
  const marks = (f, lo) => {   // the cutoff and the signal's frequency, clipped to the plot box
    f.ctx.save(); f.ctx.beginPath(); f.ctx.rect(f.pad.l, f.pad.t, f.iw, f.ih); f.ctx.clip();
    f.ctx.strokeStyle = css("--warm"); f.ctx.setLineDash([3, 3]);
    for (const v of [Math.log10(F0), Math.log10(1 / (2 * Math.PI * st.tau))]) seg(f.ctx, f.X(v), f.pad.t, f.X(v), f.Y(lo));
    f.ctx.restore();
  };
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [-2, LOG_NYQ], y: [-60, 10], xlabel: "frequency [Hz], log scale (Nyquist 50 Hz)", ylabel: "gain [dB]", xticks: xt, xfmt: xf, yticks: [-60, -40, -20, 0] });
  plotLine(fr, fs, mag, { color: alpha("--muted", 0.9), width: 1.4, dash: [4, 3] }); plotLine(fr, fs, magD, { color: css("--post"), width: 2 });
  marks(fr, -60);
  const cv2 = $("#phaseC"), fr2 = frame(cv2, narrow ? 1.5 : 2.6, { x: [-2, LOG_NYQ], y: [-180, 0], xlabel: "frequency [Hz], log scale", ylabel: "phase [°]", xticks: xt, xfmt: xf, yticks: [-180, -135, -90, -45, 0] });
  plotLine(fr2, fs, phC, { color: alpha("--muted", 0.9), width: 1.4, dash: [4, 3] }); plotLine(fr2, fs, phK, { color: css("--ink"), width: 1.3 }); plotLine(fr2, fs, phD, { color: css("--post"), width: 2 });
  marks(fr2, -180);
  const rd = responseD(F0, st.K, st.tau, Te), pC = deg(responseC(F0, st.K, st.tau).phase), pD = deg(rd.phase), pK = deg(responseD(F0, st.K, st.tau, Te, true).phase);
  caption($("#bodeCap"), `Frequency response of the filter with [[\\tau = ${st.tau}]] s. In the gain plot, the gain of the continuous filter [[K/\\sqrt{1 + (2\\pi f\\tau)^2}]] (grey dashed) and of the C recursion evaluated on the unit circle (blue): they agree far below the Nyquist frequency of 50 Hz and part near it, where the sampling matters. The recursion fed [[u_k]] has exactly the same gain. In the phase plot, the phases of the continuous filter (grey dashed), of the C recursion (blue) and of the [[u_k]] form (thin line). The orange lines mark the cutoff [[1/(2\\pi\\tau) = ${fmt(1 / (2 * Math.PI * st.tau), 3)}]] Hz and the signal's ${fmt(F0, 2)} Hz. For the continuous filter, the gain falls at 20 dB per decade above the cutoff and the lag at the cutoff is 45°. At the signal's frequency the gain is ${fmt(20 * Math.log10(rd.mag), 1).replace("-", "−")} dB; the continuous filter lags by ${fmt(-pC, 1)}°, the C recursion by ${fmt(pC - pD, 2)}° more (${fmt((pC - pD) / (360 * F0 * Te), 2)} of a sample; for a time constant of many samples this tends to half a sample, the hold's average delay of [[T_e/2]]), and the [[u_k]] form by ${fmt(pK - pC, 2)}° less, so the two discrete forms are [[360^\\circ f T_e = ${fmt(360 * F0 * Te, 1)}^\\circ]] apart. Near 50 Hz the [[u_k]] form's phase returns to 0° while the C recursion's falls to −180°: the extra half-turn is its one sample of delay.`);
}
function drawHist() {
  const u = D.input, y = filtrage(u, 1, 2, Te), yT = filtrage(u, st.K, st.tau, Te);
  const dRepo = u.map((v, i) => v - y[i]), dNow = u.map((v, i) => v - yT[i]), hr = histogramme(dRepo), hn = histogramme(dNow);
  const dens = H => H.h.map(c => c / (N * H.w)), dr = dens(hr), dn = dens(hn), sig = D.fitted.resid_sd;
  const gauss = x => Math.exp(-0.5 * (x / sig) ** 2) / (sig * Math.sqrt(2 * Math.PI));
  const lo = Math.min(hr.mn, hn.mn), hi = Math.max(hr.mx, hn.mx);
  const cv = $("#histC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [lo, hi], y: [0, Math.max(...dr, ...dn, gauss(0)) * 1.12], xlabel: "u − y [signal units]", ylabel: "density" });
  const bars = (H, d, color) => { fr.ctx.fillStyle = color; d.forEach((v, i) => { const a = fr.X(H.mn + i * H.w), b = fr.X(H.mn + (i + 1) * H.w); fr.ctx.fillRect(a + 0.5, fr.Y(v), Math.max(1, b - a - 1), fr.Y(0) - fr.Y(v)); }); };
  bars(hr, dr, alpha("--muted", 0.45)); bars(hn, dn, alpha("--post", 0.6));
  const xs = Array.from({ length: 201 }, (_, i) => lo + (hi - lo) * i / 200);
  plotLine(fr, xs, xs.map(gauss), { color: css("--ink"), width: 1.4, dash: [5, 4] });
  const tdc = st.tau === 0.02 && st.K === 1;
  caption($("#histCap"), `The repository's 16-bin histogram routine applied to the difference between input and output, [[u - y]], drawn as densities on a common axis: each count is divided by 250 times its bin's width, since the routine sizes its bins from each array's own minimum and maximum. Grey, the repository's [[\\tau = 2]] s: counts ${hr.h.join(", ")}, what the program prints with only bugs 1 and 2 fixed. Blue, the page's current [[\\tau = ${st.tau}]] s${st.K !== 1 ? ` and [[K = ${st.K}]]` : ""}: counts ${hn.h.join(", ")}${tdc ? ", what the corrected TD3.c prints" : ""}. Dashed, a normal density with the noise's sd, ${fmt(sig, 3)}. With [[\\tau = 2]] the output is nearly zero, so the "difference" is the input itself, with the two-humped shape of a sine that piles up near its extremes; it says nothing about the noise. With a time constant that lets the sine through (try 0.02 s) the difference is a single hump around zero${st.tau <= 0.025 && Math.abs(st.K - 1) <= 0.1 + 1e-9 ? `, here of sd ${fmt(sd(dNow), 3)}` : ""}, wider than the noise alone: it also holds the part of the sine that the lag shifted, and the noise that the output kept. The routine itself is sound; what it was fed was not. In the original program it was fed 500 entries of arrays in which only 250 had been written; compiled here, the 250 cells never written held zeros, and the bin that holds zero counted 264.`);
}
$("#hTau").addEventListener("input", e => { st.tau = +(10 ** +e.target.value).toPrecision(2); $("#hTauV").textContent = `${st.tau} s`; drawAll(); });
$("#hK").addEventListener("input", e => { st.K = +e.target.value; $("#hKV").textContent = fmt(st.K, 2); drawAll(); });
$$("#inSeg .btn").forEach(b => b.addEventListener("click", () => { $$("#inSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); st.input = b.dataset.i; drawHero(); }));
$$("[data-tau]").forEach(b => b.addEventListener("click", () => { st.tau = +b.dataset.tau; $("#hTau").value = Math.log10(st.tau); $("#hTauV").textContent = `${st.tau} s`; drawAll(); }));
$("#hTauV").textContent = "2 s"; $("#hKV").textContent = "1.00";
function drawAll() { drawHero(); drawBode(); drawHist(); }
drawAll(); onRedraw(drawAll);
})();
