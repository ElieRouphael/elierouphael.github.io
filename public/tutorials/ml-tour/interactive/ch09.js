/* A Tour of Machine Learning, chapter 09, Anomaly detection with an autoencoder: the live demos.
   The two series, the training-set scaling, the 24-point windows and the trained convolutional
   autoencoder are the notebook's (ch09-data.js); the network runs here on every window, so the
   scores, the threshold and the flagged windows are the notebook's. The PCA baseline is computed
   here (Jacobi eigenvalues of the 24x24 training covariance).
   Needs ../kit/kit.js and ch09-data.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp, fit } = window.Kit;
const D = window.CH09;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
/* Conv1D, padding "same", stride s, ReLU. x: array of L arrays (channels). K: [k][cin][cout]. */
function conv1d(x, K, b, s, relu) {
  const L = x.length, k = K.length, cin = K[0].length, cout = K[0][0].length, Lout = Math.ceil(L / s), padL = Math.floor(Math.max((Lout - 1) * s + k - L, 0) / 2), out = [];
  for (let t = 0; t < Lout; t++) {
    const o = b.slice();
    for (let j = 0; j < k; j++) { const p = t * s + j - padL; if (p < 0 || p >= L) continue; const xp = x[p]; for (let ci = 0; ci < cin; ci++) { const v = xp[ci]; if (v === 0) continue; const row = K[j][ci]; for (let co = 0; co < cout; co++) o[co] += v * row[co]; } }
    if (relu) for (let co = 0; co < cout; co++) if (o[co] < 0) o[co] = 0;
    out.push(o);
  }
  return out;
}
/* Conv1DTranspose, padding "same", stride s: output length L*s. K: [k][cout][cin] (Keras layout). */
function conv1dT(x, K, b, s, relu) {
  const L = x.length, k = K.length, cout = K[0].length, cin = K[0][0].length, Lout = L * s, padL = Math.floor(Math.max((L - 1) * s + k - Lout, 0) / 2);
  const out = Array.from({ length: Lout }, () => b.slice());
  for (let i = 0; i < L; i++) { const xi = x[i]; for (let j = 0; j < k; j++) { const t = i * s + j - padL; if (t < 0 || t >= Lout) continue; const o = out[t]; for (let co = 0; co < cout; co++) { const row = K[j][co]; let a = 0; for (let ci = 0; ci < cin; ci++) a += xi[ci] * row[ci]; o[co] += a; } } }
  if (relu) for (const o of out) for (let co = 0; co < cout; co++) if (o[co] < 0) o[co] = 0;
  return out;
}
/* The notebook's autoencoder on one standardised window (array of 24 numbers). Returns the reconstruction, the code and the MAE. */
function autoencode(W, names, xw) {
  let h = xw.map(v => [v]);
  const [c1, c2, t1, t2, t3] = names;
  h = conv1d(h, W[c1][0], W[c1][1], 2, true); h = conv1d(h, W[c2][0], W[c2][1], 2, true); const code = h;
  h = conv1dT(h, W[t1][0], W[t1][1], 2, true); h = conv1dT(h, W[t2][0], W[t2][1], 2, true); h = conv1dT(h, W[t3][0], W[t3][1], 1, false);
  const rec = h.map(o => o[0]); let mae = 0; for (let t = 0; t < xw.length; t++) mae += Math.abs(rec[t] - xw[t]);
  return { rec, code, mae: mae / xw.length };
}
/* Symmetric eigen-decomposition by cyclic Jacobi rotations: returns eigenvalues (descending) and eigenvectors (columns). */
function jacobiEig(A0) {
  const n = A0.length, A = A0.map(r => r.slice()), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b][b] - A[a][a]);
  return { values: order.map(i => A[i][i]), vectors: order.map(i => V.map(r => r[i])) };   // vectors[k] is the k-th eigenvector
}
/* NET-END */

const MU = D.mu, SIG = D.sigma, Wn = D.window, TR = D.train.map(v => (v - MU) / SIG), TE = D.test.map(v => (v - MU) / SIG), N = TR.length - Wn + 1, [A0, A1] = D.anomaly;
const win = (v, i) => v.slice(i, i + Wn);
const ov = i => i + Wn - 1 >= A0 && i <= A1;
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const t0 = Date.now();
const TRR = Array.from({ length: N }, (_, i) => autoencode(D.W, D.layers, win(TR, i)));
const TER = Array.from({ length: N }, (_, i) => autoencode(D.W, D.layers, win(TE, i)));
const TRM = TRR.map(r => r.mae), TEM = TER.map(r => r.mae), THR = Math.max(...TRM);
const TRM_SORTED = TRM.slice().sort((a, b) => a - b), trMean = mean(TRM), trSd = Math.sqrt(mean(TRM.map(v => (v - trMean) ** 2)));
const quantile = q => { const p = (TRM_SORTED.length - 1) * q, lo = Math.floor(p); return TRM_SORTED[lo] + (TRM_SORTED[Math.min(lo + 1, TRM_SORTED.length - 1)] - TRM_SORTED[lo]) * (p - lo); };
const RULES = { max: THR, sd3: trMean + 3 * trSd, p99: quantile(0.99) };
const label = i => { const day = Math.floor(i / 288) + 1, m = i % 288, h = Math.floor(m / 12), mi = (m % 12) * 5; return `April ${day}, ${String(h).padStart(2, "0")}:${String(mi).padStart(2, "0")}`; };
const flagsFor = tau => { const f = []; for (let i = 0; i < N; i++) if (TEM[i] > tau) f.push(i); return f; };
const summarise = tau => { const f = flagsFor(tau), hit = f.filter(ov), fp = f.filter(i => !ov(i)); return { f, hit, fp, missed: 0 }; };
const INSIDE = Array.from({ length: N }, (_, i) => i).filter(ov), FULL = INSIDE.filter(i => i >= A0 && i + Wn - 1 <= A1);

/* =====================================================================
   Hero: the test series, the score over time, a movable threshold
   ===================================================================== */
const hs = { tau: THR };
function drawHero() {
  const cv = $("#aeC"), narrow = narrowOf(cv), tau = hs.tau, S = summarise(tau), T = TE.length;
  const { ctx, w, h } = fit(cv, narrow ? 1.0 : 1.9);
  const top = { t: 10, b: h * 0.5 - 14 }, bot = { t: h * 0.5 + 12, b: h - 26 };
  const padL = 40, padR = 10, X = i => padL + i / (T - 1) * (w - padL - padR);
  const panel = (p, y0, y1, ticks) => { const Y = v => p.b - (v - y0) / (y1 - y0) * (p.b - p.t); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; for (const t of ticks) { ctx.strokeStyle = css("--grid"); seg(ctx, padL, Y(t), w - padR, Y(t)); ctx.fillStyle = css("--muted"); ctx.fillText(String(t), padL - 5, Y(t)); } ctx.textBaseline = "alphabetic"; return Y; };
  /* top: raw test series with the jump and the flagged windows */
  const Y1 = panel(top, 0, 170, [0, 50, 100, 150]);
  ctx.fillStyle = alpha("--warm", 0.12); ctx.fillRect(X(A0), top.t, X(A1) - X(A0), top.b - top.t);
  ctx.fillStyle = alpha("--post", 0.35); for (const i of S.f) ctx.fillRect(X(i), top.t, Math.max(1, X(i + Wn - 1) - X(i)), top.b - top.t);
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1; ctx.beginPath(); for (let i = 0; i < T; i++) { const px = X(i), py = Y1(D.test[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); } ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText("test series (raw value); flagged windows in blue, the injected jump in orange", padL, top.t - 1);
  /* bottom: the score per window with the threshold */
  const ymax = narrow ? 0.4 : 0.4, Y2 = panel(bot, 0, ymax, [0, 0.1, 0.2, 0.3, 0.4]);
  ctx.fillStyle = alpha("--warm", 0.12); ctx.fillRect(X(A0 - Wn + 1), bot.t, X(A1) - X(A0 - Wn + 1), bot.b - bot.t);
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 1.2; ctx.beginPath(); for (let i = 0; i < N; i++) { const px = X(i), py = Y2(Math.min(ymax, TEM[i])); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); } ctx.stroke();
  ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]); seg(ctx, padL, Y2(tau), w - padR, Y2(tau)); ctx.setLineDash([]);
  ctx.fillStyle = css("--muted"); ctx.textAlign = "left"; ctx.fillText("reconstruction MAE per window, and the threshold", padL, bot.t - 1);
  ctx.textAlign = "center"; for (const d of [1, 4, 7, 10, 13]) { const i = (d - 1) * 288; ctx.fillText("Apr " + d, X(i), h - 8); }
  /* readouts */
  $("#tauV").textContent = fmt(tau, 4);
  $("#rFlag").textContent = String(S.f.length); $("#rHit").textContent = `${S.hit.length} of ${INSIDE.length}`; $("#rFp").textContent = String(S.fp.length); $("#rMiss").textContent = `${FULL.filter(i => TEM[i] <= tau).length} of ${FULL.length}`;
  let text = `Threshold ${fmt(tau, 4)}` + (Math.abs(tau - THR) < 1e-9 ? ", the notebook's rule, the largest error on any training window" : Math.abs(tau - RULES.sd3) < 1e-9 ? ", the training mean plus three standard deviations" : Math.abs(tau - RULES.p99) < 1e-9 ? ", the 99th percentile of the training scores" : "") + `: ${S.f.length} of the ${N} test windows are flagged. ${S.hit.length} of them overlap the jump (April 11, 09:00 to 17:55, 108 readings at about 3.7 training standard deviations); ${S.fp.length} ${S.fp.length === 1 ? "is a false alarm" : "are false alarms"}` + (S.fp.length ? `, ${S.fp.length === 1 ? "at " + label(S.fp[0]) : "the first at " + label(S.fp[0])}${S.fp.length === 1 ? `, scoring ${fmt(TEM[S.fp[0]], 4)}, ${fmt(100 * (TEM[S.fp[0]] / tau - 1), 2)}% above the line` : ""}` : "") + `. `;
  const edgeFp = S.fp.filter(i => { const m = i % 288; return (m >= 100 && m <= 110) || (m >= 190 && m <= 206); }).length;
  if (S.fp.length >= 3) text += `${edgeFp} of the false alarms sit at the same two times of day, around 08:45 and 16:30, where the daily pattern steps up and down: the edges are the hardest normal windows to redraw, and a lower threshold trips on them first. `;
  const missed = FULL.filter(i => TEM[i] <= tau).length;
  text += `Of the ${FULL.length} windows that lie wholly inside the jump, ${missed} stay under the line: the flat raised plateau redraws better than the jump's two edges, which score ${fmt(Math.max(...INSIDE.map(i => TEM[i])), 3)} at worst. The detector finds the moment the level changes more surely than the fact that it stays changed.`;
  caption($("#heroStatus"), text);
}
$("#tau").addEventListener("input", () => { hs.tau = +$("#tau").value; drawHero(); });
$$("#demo [data-rule]").forEach(bt => bt.addEventListener("click", () => { hs.tau = RULES[bt.dataset.rule]; $("#tau").value = hs.tau; drawHero(); }));

/* =====================================================================
   Part 2: one window and its reconstruction
   ===================================================================== */
const rs = { set: "test", i: 2985 };
function drawRec() {
  const series = rs.set === "test" ? TE : TR, raw = rs.set === "test" ? D.test : D.train, runs = rs.set === "test" ? TER : TRR, i = rs.i, r = runs[i], x = win(series, i);
  const cv = $("#recC"), narrow = narrowOf(cv), lo = Math.min(-1.2, ...x, ...r.rec) - 0.2, hi = Math.max(2, ...x, ...r.rec) + 0.2;
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, Wn - 1], y: [lo, hi], xlabel: "position in the window (5-minute steps)", xticks: [0, 6, 12, 18, 23], ylabel: "standardised value" });
  const { ctx, X, Y } = fr;
  for (let t = 0; t < Wn; t++) { ctx.fillStyle = alpha("--warm", 0.35); const y0 = Y(x[t]), y1 = Y(r.rec[t]); ctx.fillRect(X(t) - 2, Math.min(y0, y1), 4, Math.abs(y1 - y0)); }
  plotLine(fr, [...Array(Wn).keys()], x, { color: css("--ink"), width: 2 }); plotLine(fr, [...Array(Wn).keys()], r.rec, { color: css("--post"), width: 2, dash: [5, 3] });
  $("#rIV").textContent = i; $("#rI").value = i;
  const worst = r.rec.map((v, t) => Math.abs(v - x[t])).reduce((b, v, t, a) => v > a[b] ? t : b, 0);
  caption($("#recCap"), `${rs.set === "test" ? "Test" : "Training"} window ${i}, starting ${label(i)}: raw readings ${fmt(raw[i], 1)} to ${fmt(raw[i + Wn - 1], 1)}, standardised ${fmt(x[0], 2)} to ${fmt(x[Wn - 1], 2)}. The code is 6 positions × 8 channels; from it the decoder redraws the dashed curve. The shaded gaps average to the score [[s = ${fmt(r.mae, 4)}]]` + (r.mae > THR ? `, above the threshold ${fmt(THR, 4)}: flagged.` : `, under the threshold ${fmt(THR, 4)}: normal.`) + ` The worst position is ${worst} (off by ${fmt(Math.abs(r.rec[worst] - x[worst]), 2)}). ` + (rs.set === "test" && ov(i) ? (i >= A0 && i + Wn - 1 <= A1 ? "This window lies wholly inside the jump: a plateau at 3.7 standard deviations, which the decoder redraws as a lower plateau, since it never produced values this high." : "This window straddles an edge of the jump, the hardest shape of all for a network that only knows the daily step.") : ((i % 288 >= 100 && i % 288 <= 110) || (i % 288 >= 190 && i % 288 <= 215) ? "This window contains one of the daily steps, where even normal windows redraw worst." : "A routine window: the model has seen thousands like it.")));
}
$("#rI").max = N - 1; $("#rI").addEventListener("input", () => { rs.i = +$("#rI").value; drawRec(); });
$$("#recW [data-set]").forEach(bt => bt.addEventListener("click", () => { rs.set = bt.dataset.set; $$("#recW [data-set]").forEach(b => b.setAttribute("aria-pressed", String(b === bt))); drawRec(); }));
$$("#recW [data-jump]").forEach(bt => bt.addEventListener("click", () => { rs.set = "test"; rs.i = +bt.dataset.jump; $$("#recW [data-set]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.set === "test"))); drawRec(); }));

/* =====================================================================
   Part 3: the linear autoencoder is PCA; a PCA detector with k components
   ===================================================================== */
const PCA = (() => {
  const Xtr = Array.from({ length: N }, (_, i) => win(TR, i)), m = Array.from({ length: Wn }, (_, t) => mean(Xtr.map(x => x[t])));
  const C = Array.from({ length: Wn }, () => new Float64Array(Wn));
  for (const x of Xtr) for (let a = 0; a < Wn; a++) { const da = x[a] - m[a]; for (let b = a; b < Wn; b++) C[a][b] += da * (x[b] - m[b]); }
  for (let a = 0; a < Wn; a++) for (let b = 0; b < a; b++) C[a][b] = C[b][a];
  const Cm = C.map(r => Array.from(r, v => v / (N - 1))), eig = jacobiEig(Cm), total = eig.values.reduce((s, v) => s + v, 0);
  return { m, eig, total, Xtr, Xte: Array.from({ length: N }, (_, i) => win(TE, i)) };
})();
function pcaScores(k) {
  const U = PCA.eig.vectors.slice(0, k), score = X => X.map(x => { const d = x.map((v, t) => v - PCA.m[t]); let e = 0; const z = U.map(u => u.reduce((s, v, t) => s + v * d[t], 0)); for (let t = 0; t < Wn; t++) { let r = 0; for (let j = 0; j < k; j++) r += U[j][t] * z[j]; e += Math.abs(d[t] - r); } return e / Wn; });
  return { tr: score(PCA.Xtr), te: score(PCA.Xte) };
}
function drawPca() {
  const k = +$("#pK").value, sc = pcaScores(k), th = Math.max(...sc.tr), f = sc.te.map((v, i) => v > th ? i : -1).filter(i => i >= 0), hit = f.filter(ov), fp = f.filter(i => !ov(i)), missed = FULL.filter(i => sc.te[i] <= th).length;
  const cv = $("#pcaC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.6 : 3, { x: [0, N - 1], y: [0, 0.6], xticks: [0, 1000, 2000, 3000, 4000], yticks: [0, 0.2, 0.4, 0.6], xlabel: "test window", ylabel: "PCA reconstruction MAE" }), { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--warm", 0.12); ctx.fillRect(X(A0 - Wn + 1), fr.pad.t, X(A1) - X(A0 - Wn + 1), fr.ih);
  plotLine(fr, [...Array(N).keys()], sc.te.map(v => Math.min(0.6, v)), { color: css("--c5"), width: 1.2 });
  ctx.strokeStyle = css("--warm"); ctx.setLineDash([6, 4]); seg(ctx, X(0), Y(Math.min(0.6, th)), X(N - 1), Y(Math.min(0.6, th))); ctx.setLineDash([]);
  $("#pKV").textContent = k;
  const expl = PCA.eig.values.slice(0, k).reduce((s, v) => s + v, 0) / PCA.total;
  caption($("#pcaCap"), `PCA with [[k = ${k}]] component${k > 1 ? "s" : ""} keeps ${fmt(100 * expl, 1)}% of the training windows' variance. Training MAE ${fmt(mean(sc.tr), 4)} on average, threshold (training max) ${fmt(th, 4)}: ${f.length} test windows flagged, ${hit.length} overlapping the jump, ${fp.length} false alarm${fp.length === 1 ? "" : "s"}, and ${missed} of the ${FULL.length} windows wholly inside the jump missed. ` + (k >= 4 && fp.length === 0 ? "A linear projection, fitted in a millisecond with no training loop, finds the jump's two edges as cleanly as the network, and none of the plateau between them: the first component is the window's level, so a flat stretch at any height projects almost perfectly." : k <= 2 ? "Too few components: the daily step itself is not reproduced, so normal edge windows score high and the threshold rises above part of the jump." : "") + ` The convolutional autoencoder: ${summarise(THR).f.length} flagged, ${summarise(THR).fp.length} false alarm, ${FULL.filter(i => TEM[i] <= THR).length} inside windows missed.`);
}
$("#pK").addEventListener("input", drawPca);

/* =====================================================================
   Part 4: the worked example with a movable threshold
   ===================================================================== */
const EX = [["A", 0.10, "normal"], ["B", 0.22, "normal"], ["C", 0.29, "anomaly"], ["D", 0.31, "normal"], ["E", 0.85, "anomaly"]];
function drawEx() {
  const tau = +$("#exTau").value; $("#exTauV").textContent = fmt(tau, 2);
  let fp = 0, fn = 0;
  $$("#exT tbody tr").forEach((tr, k) => {
    const [name, s, truth] = EX[k], flag = s > tau, outcome = flag && truth === "anomaly" ? "correct" : !flag && truth === "normal" ? "correct" : flag ? "false positive" : "false negative";
    if (outcome === "false positive") fp++; if (outcome === "false negative") fn++;
    tr.children[2].textContent = flag ? "yes" : "no"; tr.children[3].textContent = flag ? "anomaly" : "normal"; tr.children[5].textContent = outcome; tr.className = outcome === "correct" ? "" : "bad";
  });
  $("#exCap").textContent = `With τ = ${fmt(tau, 2)}: ${fp} false positive${fp === 1 ? "" : "s"}, ${fn} false negative${fn === 1 ? "" : "s"}. ` + (fp === 0 && fn === 0 ? "No threshold between 0.29 and 0.31 exists on this slider's grid; move it and watch one error turn into the other." : fp && fn ? "Both kinds of mistake at once: C and D sit on the wrong sides of each other." : fp ? "Lowering the line catches C but keeps the alarm on D." : "Raising the line clears D but loses C.");
}
$("#exTau").addEventListener("input", drawEx);

function drawAll() { drawHero(); drawRec(); drawPca(); drawEx(); }
$("#tau").value = THR;
drawAll();
onRedraw(drawAll);
console.log(`ch09: ${2 * N} windows scored in ${Date.now() - t0} ms; threshold ${THR.toFixed(5)}; flagged ${flagsFor(THR).length}`);
})();
