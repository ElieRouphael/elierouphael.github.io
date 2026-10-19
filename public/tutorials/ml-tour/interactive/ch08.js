/* A Tour of Machine Learning, chapter 08, LSTM for sequences: the live demos.
   The series, the 80/20 split, the scaling and the trained LSTM(24) + Dense(1) are the
   notebook's (ch08-data.js); the cell is run here step by step, with Keras's gate order
   (input, forget, candidate, output), so the forecasts and the test MSE are the notebook's.
   The linear autoregression is fitted here by least squares on the same windows.
   Needs ../kit/kit.js and ch08-data.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp, fit } = window.Kit;
const D = window.CH08;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const sigmoid = z => 1 / (1 + Math.exp(-z));
/* One window through the cell. x: array of L standardised inputs. Returns the gates, states and the forecast (standardised). */
function lstmRun(P, x) {
  const H = P.bias.length / 4, L = x.length;
  const F = [], I = [], O = [], C = [], Hs = [];
  let h = new Float64Array(H), c = new Float64Array(H);
  for (let s = 0; s < L; s++) {
    const z = new Float64Array(4 * H);
    for (let k = 0; k < 4 * H; k++) z[k] = P.bias[k] + P.kernel[k] * x[s];
    for (let j = 0; j < H; j++) { const hj = h[j]; if (hj === 0) continue; const row = P.recurrent[j]; for (let k = 0; k < 4 * H; k++) z[k] += hj * row[k]; }
    const i = new Float64Array(H), f = new Float64Array(H), o = new Float64Array(H), cn = new Float64Array(H), hn = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      i[j] = sigmoid(z[j]); f[j] = sigmoid(z[H + j]); const ct = Math.tanh(z[2 * H + j]); o[j] = sigmoid(z[3 * H + j]);
      cn[j] = f[j] * c[j] + i[j] * ct; hn[j] = o[j] * Math.tanh(cn[j]);
    }
    F.push(f); I.push(i); O.push(o); C.push(cn); Hs.push(hn); h = hn; c = cn;
  }
  let yhat = P.dense_b; for (let j = 0; j < H; j++) yhat += P.dense_w[j] * h[j];
  return { F, I, O, C, H: Hs, yhat };
}
/* Least squares with an intercept: minimise sum (X w + b - y)^2 by the normal equations (Gaussian elimination with pivoting). */
function leastSquares(X, y) {
  const n = X.length, d = X[0].length + 1, A = Array.from({ length: d }, () => new Float64Array(d + 1));
  for (let r = 0; r < n; r++) {
    const row = X[r];
    for (let a = 0; a < d; a++) { const xa = a < d - 1 ? row[a] : 1; for (let b = a; b < d; b++) A[a][b] += xa * (b < d - 1 ? row[b] : 1); A[a][d] += xa * y[r]; }
  }
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) A[a][b] = A[b][a];
  for (let col = 0; col < d; col++) {
    let p = col; for (let r = col + 1; r < d; r++) if (Math.abs(A[r][col]) > Math.abs(A[p][col])) p = r;
    [A[col], A[p]] = [A[p], A[col]];
    for (let r = 0; r < d; r++) { if (r === col) continue; const m = A[r][col] / A[col][col]; if (!m) continue; for (let b = col; b <= d; b++) A[r][b] -= m * A[col][b]; }
  }
  const w = new Float64Array(d); for (let a = 0; a < d; a++) w[a] = A[a][d] / A[a][a];
  return w;   // the last entry is the intercept
}
const predictLinear = (w, x) => { let v = w[w.length - 1]; for (let a = 0; a < x.length; a++) v += w[a] * x[a]; return v; };
/* NET-END */

const Y = D.y, MU = D.mu, SIG = D.sigma, NTR = D.n_train, L = D.window, T = Y.length;
const signal = t => Math.sin(2 * Math.PI * t / 50) + 0.5 * Math.sin(2 * Math.PI * t / 17) + 0.02 * Math.sin(2 * Math.PI * t / 400);
const windowAt = i => Y.slice(i, i + L);                       // inputs for target index i + L
const NTEST = T - L - NTR, testIdx = Array.from({ length: NTEST }, (_, j) => NTR + j);   // window starts
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

/* the trained LSTM on every test window */
const RUNS = testIdx.map(i => lstmRun(D, windowAt(i).map(v => (v - MU) / SIG)));
const PRED = RUNS.map(r => r.yhat * SIG + MU), ACT = testIdx.map(i => Y[i + L]), NAIVE = testIdx.map(i => Y[i + L - 1]), ORACLE = testIdx.map(i => signal(i + L));
const mse = (p, a) => mean(p.map((v, j) => (v - a[j]) ** 2));
const MSE = { lstm: mse(PRED, ACT), naive: mse(NAIVE, ACT), oracle: mse(ORACLE, ACT) };
/* the linear autoregression, fitted on the training windows in original units */
const WLIN = leastSquares(Array.from({ length: NTR }, (_, i) => windowAt(i)), Array.from({ length: NTR }, (_, i) => Y[i + L]));
const LIN = testIdx.map(i => predictLinear(WLIN, windowAt(i))); MSE.lin = mse(LIN, ACT);
const FMEAN_ALL = mean(RUNS.map(r => mean(r.F.flatMap(v => Array.from(v)))));

/* =====================================================================
   Hero: one-step forecasts on the held-out stretch
   ===================================================================== */
const hs = { lin: false, oracle: false };
function drawHero() {
  const cv = $("#fcC"), narrow = narrowOf(cv), show = narrow ? 80 : 150, start = clamp(+$("#hStart").value, 0, NTEST - show);
  const fr = frame(cv, narrow ? 1.3 : 2.1, { x: [start, start + show - 1], y: [-2.1, 2.1], xlabel: "test step", yticks: [-2, -1, 0, 1, 2], xticks: undefined });
  const { ctx, X, Y: Yp } = fr, xs = Array.from({ length: show }, (_, k) => start + k), sl = a => xs.map(k => a[k]);
  ctx.strokeStyle = alpha("--ink", 0.25); seg(ctx, X(start), Yp(0), X(start + show - 1), Yp(0));
  if (hs.oracle) plotLine(fr, xs, sl(ORACLE), { color: css("--c4"), width: 1.4, dash: [2, 3] });
  plotLine(fr, xs, sl(NAIVE), { color: css("--warm"), width: 1.3, dash: [6, 4] });
  if (hs.lin) plotLine(fr, xs, sl(LIN), { color: css("--c5"), width: 1.8 });
  plotLine(fr, xs, sl(ACT), { color: css("--ink"), width: 1.8 });
  plotLine(fr, xs, sl(PRED), { color: css("--post"), width: 2.2 });
  $("#hStartV").textContent = `${start} to ${start + show - 1} of ${NTEST}`;
  const win = mse(sl(PRED), sl(ACT)), winN = mse(sl(NAIVE), sl(ACT)), winL = mse(sl(LIN), sl(ACT));
  $("#rLstm").textContent = fmt(MSE.lstm, 4); $("#rNaive").textContent = fmt(MSE.naive, 4); $("#rLin").textContent = hs.lin ? fmt(MSE.lin, 4) : "–"; $("#rFloor").textContent = hs.oracle ? fmt(MSE.oracle, 4) : "–";
  let text = `Over all ${NTEST} test windows the LSTM's mean squared error is ${fmt(MSE.lstm, 4)} against ${fmt(MSE.naive, 4)} for persistence, ${fmt(MSE.naive / MSE.lstm, 2)} times lower; on the ${show} steps shown, ${fmt(win, 4)} against ${fmt(winN, 4)}. The dashed forecast is always one step late: it repeats the last value, so it peaks after the series does. `;
  if (hs.lin) text += `The linear autoregression on the same 30 values scores ${fmt(MSE.lin, 4)} overall (${fmt(winL, 4)} here), ${MSE.lin < MSE.lstm ? "better than the LSTM" : "worse than the LSTM"}: the series is a sum of sines plus noise, which a linear rule predicts almost as well as anything can. `;
  if (hs.oracle) text += `The dotted line is the noise-free signal, the best forecast any method could make; its error, ${fmt(MSE.oracle, 4)}, is the fresh noise, and the floor the others are measured against.`;
  caption($("#heroStatus"), text);
}
$("#hStart").max = NTEST - 1; $("#hStart").addEventListener("input", drawHero);
$("#hLin").addEventListener("click", () => { hs.lin = !hs.lin; $("#hLin").setAttribute("aria-pressed", String(hs.lin)); $("#hLin").textContent = hs.lin ? "hide the linear model" : "add a linear model"; drawHero(); });
$("#hOracle").addEventListener("click", () => { hs.oracle = !hs.oracle; $("#hOracle").setAttribute("aria-pressed", String(hs.oracle)); $("#hOracle").textContent = hs.oracle ? "hide the noise floor" : "show the noise floor"; drawHero(); });

/* =====================================================================
   Part 1: the sliding window
   ===================================================================== */
function drawWin() {
  const cv = $("#winC"), narrow = narrowOf(cv), t = +$("#wT").value, lo = Math.max(0, t - 70), hi = lo + (narrow ? 90 : 110);
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [lo, hi], y: [-2.1, 2.1], xlabel: "t", yticks: [-2, -1, 0, 1, 2] });
  const { ctx, X, Y: Yp } = fr;
  ctx.fillStyle = alpha("--post", 0.14); ctx.fillRect(X(t - L), fr.pad.t, X(t - 1) - X(t - L), fr.ih);
  const xs = Array.from({ length: hi - lo + 1 }, (_, k) => lo + k);
  plotLine(fr, xs, xs.map(k => Y[k]), { color: alpha("--ink", 0.6), width: 1.3 });
  ctx.fillStyle = css("--post"); for (let k = t - L; k < t; k++) { ctx.beginPath(); ctx.arc(X(k), Yp(Y[k]), 2.2, 0, 2 * Math.PI); ctx.fill(); }
  ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(t), Yp(Y[t]), 4.5, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; ctx.fillText("window", X(t - L / 2), fr.pad.t + 12); ctx.fillText("target", X(t), Yp(Y[t]) - 10);
  $("#wTV").textContent = t;
  const w = windowAt(t - L), f3 = w.slice(0, 3).map(v => fmt(v, 2)).join(", "), l3 = w.slice(-3).map(v => fmt(v, 2)).join(", ");
  caption($("#winCap"), `Pair number ${t - L + 1} of ${T - L}: the input is the ${L} values [[y_{${t - L}}, \\dots, y_{${t - 1}}]], here (${f3}, …, ${l3}), and the target is [[y_{${t}} = ${fmt(Y[t], 3)}]]. ${t - L < NTR ? "This pair is in the training set" : "This pair is in the test set, after step " + (NTR + L - 1)}; persistence would guess [[y_{${t - 1}} = ${fmt(Y[t - 1], 3)}]], an error of ${fmt(Y[t - 1] - Y[t], 3)}.`);
}
$("#wT").min = L; $("#wT").max = T - 1; $("#wT").addEventListener("input", drawWin);

/* =====================================================================
   Part 2: a product of many factors
   ===================================================================== */
function drawVg() {
  const a = +$("#vgA").value, f = +$("#vgF").value, cv = $("#vgC"), narrow = narrowOf(cv), K = 100;
  const fr = frame(cv, narrow ? 1.4 : 2.4, { x: [0, K], y: [-6, 3], xlabel: "steps back, k", yticks: [-6, -4, -2, 0, 2], yfmt: v => v === 0 ? "1" : "10" + String(v).replace("-", "⁻").replace(/\d/g, ch => "⁰¹²³⁴⁵⁶⁷⁸⁹"[ch]) });
  const { ctx, X, Y: Yp } = fr, ks = Array.from({ length: K + 1 }, (_, k) => k);
  ctx.strokeStyle = alpha("--ink", 0.3); ctx.setLineDash([3, 3]); seg(ctx, X(0), Yp(0), X(K), Yp(0)); ctx.setLineDash([]);
  plotLine(fr, ks, ks.map(k => clamp(k * Math.log10(a), -6, 3)), { color: css("--warm"), width: 2.2 });
  plotLine(fr, ks, ks.map(k => clamp(k * Math.log10(f), -6, 3)), { color: css("--post"), width: 2.2 });
  for (const k of [29, 50]) { ctx.strokeStyle = alpha("--ink", 0.18); seg(ctx, X(k), fr.pad.t, X(k), Yp(-6)); }
  $("#vgAV").textContent = fmt(a, 2); $("#vgFV").textContent = fmt(f, 3);
  const p = k => (k * Math.log10(a) < -3 ? (a ** k).toExponential(1).replace("e", "\\times 10^{") + "}" : fmt(a ** k, 3));
  caption($("#vgCap"), `Plain RNN, factor [[w(1-h_s^2) = ${fmt(a, 2)}]] at every step: after 29 steps the gradient is multiplied by [[${p(29)}]], after 50 by [[${p(50)}]], after 100 by [[${p(100)}]]${a > 1 ? ", and it is growing: the explosion case" : ""}. LSTM memory line, forget gate [[f_s = ${fmt(f, 3)}]]: after 50 steps [[${fmt(f ** 50, 3)}]] survives, after 100 [[${fmt(f ** 100, 3)}]]. ` + (a < 1 && f > 0.98 ? "The network chooses the blue factor; the orange one is forced on it by the weights and the tanh." : a >= 1 ? "A factor at or above 1 keeps the gradient alive but, if it stays above 1, soon blows it up: there is no safe constant for a plain RNN." : "Lower the forget gate and the LSTM forgets just as fast as the RNN: remembering is a choice the gate makes, not a guarantee."));
}
$("#vgA").addEventListener("input", drawVg); $("#vgF").addEventListener("input", drawVg);

/* =====================================================================
   Part 4: one step of a one-unit cell, by hand
   ===================================================================== */
function drawCell() {
  const zf = +$("#cZf").value, zi = +$("#cZi").value, zo = +$("#cZo").value, zc = +$("#cZc").value, c0 = +$("#cC0").value;
  const f = sigmoid(zf), i = sigmoid(zi), o = sigmoid(zo), ct = Math.tanh(zc), c1 = f * c0 + i * ct, h1 = o * Math.tanh(c1);
  $("#cZfV").textContent = fmt(zf, 2); $("#cZiV").textContent = fmt(zi, 2); $("#cZoV").textContent = fmt(zo, 2); $("#cZcV").textContent = fmt(zc, 2); $("#cC0V").textContent = fmt(c0, 1);
  const cv = $("#cellC"), { ctx, w, h } = fit(cv, narrowOf(cv) ? 2.2 : 4.2);
  const items = [["old memory c", c0, "--muted"], ["kept  f·c", f * c0, "--post"], ["written  i·c̃", i * ct, "--c5"], ["new memory c", c1, "--post"], ["output  h", h1, "--warm"]];
  const pad = 18, bw = (w - 2 * pad) / items.length, mid = h * 0.52, scale = (h * 0.36) / 3;
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; ctx.fillStyle = css("--muted");
  ctx.strokeStyle = alpha("--ink", 0.3); seg(ctx, pad, mid + 0.5, w - pad, mid + 0.5);
  items.forEach(([name, v, tok], k) => {
    const x = pad + k * bw + bw * 0.25, bh = v * scale;
    ctx.fillStyle = alpha(tok, 0.85); ctx.fillRect(x, bh >= 0 ? mid - bh : mid, bw * 0.5, Math.abs(bh));
    ctx.fillStyle = css("--ink"); ctx.fillText(fmt(v, 3), x + bw * 0.25, bh >= 0 ? mid - bh - 6 : mid + Math.abs(bh) + 13);
    ctx.fillStyle = css("--muted"); ctx.fillText(name, x + bw * 0.25, h - 6);
  });
  caption($("#cellCap"), `Gates: [[f_s = \\sigma(${fmt(zf, 2)}) = ${fmt(f, 3)}]], [[i_s = \\sigma(${fmt(zi, 2)}) = ${fmt(i, 3)}]], [[o_s = \\sigma(${fmt(zo, 2)}) = ${fmt(o, 3)}]], candidate [[\\tilde c_s = \\tanh(${fmt(zc, 2)}) = ${fmt(ct, 3)}]]. Memory: [[c_s = ${fmt(f, 3)} \\times ${fmt(c0, 1)} + ${fmt(i, 3)} \\times ${fmt(ct, 3)} = ${fmt(c1, 3)}]]. Output: [[h_s = ${fmt(o, 3)} \\times \\tanh(${fmt(c1, 3)}) = ${fmt(h1, 3)}]]. Along the memory line [[\\partial c_s / \\partial c_{s-1} = f_s = ${fmt(f, 3)}]]` + (Math.abs(zf - 2.197) < 0.02 && Math.abs(zi) < 0.02 && Math.abs(zo - 1.386) < 0.02 && Math.abs(zc - 0.693) < 0.02 && c0 === 2 ? ": the notebook's worked example." : f > 0.98 ? ": the memory, and its gradient, pass almost untouched." : f < 0.3 ? ": most of the old memory is dropped this step." : "."));
}
$$("#cellW input").forEach(el => el.addEventListener("input", drawCell));
$("#cReset").addEventListener("click", () => { $("#cZf").value = 2.197; $("#cZi").value = 0; $("#cZo").value = 1.386; $("#cZc").value = 0.693; $("#cC0").value = 2; drawCell(); });

/* =====================================================================
   Part 3: the gates of the trained network on a real test window
   ===================================================================== */
function drawGates() {
  const j = +$("#gJ").value, run = RUNS[j], H = D.bias.length / 4, cv = $("#gateC"), w = cv.clientWidth || cv.parentElement.clientWidth;
  const lab = 18, cw = w / L, ch = Math.max(3, Math.min(7, Math.floor(cw * 0.75))), block = H * ch, gap = 14;
  const rows = [["forget gates f, one row per unit, one column per step (white 0, blue 1)", run.F, 0, 1, "--post"], ["input gates i", run.I, 0, 1, "--post"], ["output gates o", run.O, 0, 1, "--post"], ["cell state c (orange negative, blue positive)", run.C, -1, 1, "--post"]];
  const h = rows.length * (lab + block + gap) + 4;
  const { ctx } = fit(cv, w / h);
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left";
  let y0 = lab;
  for (const [name, M, lo, hi] of rows) {
    ctx.fillStyle = css("--muted"); ctx.fillText(name, 0, y0 - 6);
    const mx = lo < 0 ? Math.max(1e-9, ...M.flatMap(v => Array.from(v, Math.abs))) : 1;
    for (let s = 0; s < L; s++) for (let u = 0; u < H; u++) {
      const v = M[s][u];
      ctx.fillStyle = lo < 0 ? (v >= 0 ? alpha("--post", 0.08 + 0.9 * v / mx) : alpha("--warm", 0.08 + 0.9 * -v / mx)) : alpha("--post", 0.04 + 0.92 * v);
      ctx.fillRect(s * cw, y0 + u * ch, cw + 0.4, ch + 0.4);
    }
    ctx.strokeStyle = alpha("--ink", 0.25); ctx.strokeRect(0.5, y0 + 0.5, L * cw - 1, block - 1);
    y0 += block + gap + lab;
  }
  const fMean = mean(run.F.flatMap(v => Array.from(v))), keepers = Array.from({ length: H }, (_, u) => mean(run.F.map(v => v[u]))).filter(v => v > 0.9).length;
  const fLast = mean(run.F.slice(-5).flatMap(v => Array.from(v))), fFirst = mean(run.F.slice(0, 5).flatMap(v => Array.from(v)));
  $("#gJV").textContent = j;
  caption($("#gateCap"), `Test window ${j} of ${NTEST}: the network reads its 30 values left to right. The forget gates average ${fmt(fMean, 2)} (${fmt(fFirst, 2)} over the first 5 steps, ${fmt(fLast, 2)} over the last 5)${keepers ? `; ${keepers} of the 24 units keep [[f_s > 0.9]] on average, a long memory, while the others refresh their content at almost every step` : `, and no unit holds [[f_s > 0.9]] on average: over all ${NTEST} test windows the gates average ${fmt(FMEAN_ALL, 2)}. On a 30-step window of two sines this network has no use for a long memory, and rewrites most of its cells at every step; exercise 2 shows a gateless RNN doing as well`}. The forecast read from the final state is ${fmt(PRED[j], 3)}, the actual value ${fmt(ACT[j], 3)} (error ${fmt(PRED[j] - ACT[j], 3)}), persistence ${fmt(NAIVE[j], 3)} (error ${fmt(NAIVE[j] - ACT[j], 3)}).`);
}
$("#gJ").max = NTEST - 1; $("#gJ").addEventListener("input", drawGates);
$("#gPrev").addEventListener("click", () => { $("#gJ").value = (+$("#gJ").value - 1 + NTEST) % NTEST; drawGates(); });
$("#gNext").addEventListener("click", () => { $("#gJ").value = (+$("#gJ").value + 1) % NTEST; drawGates(); });

function drawAll() { drawHero(); drawWin(); drawVg(); drawCell(); drawGates(); }
drawAll();
onRedraw(drawAll);
})();
