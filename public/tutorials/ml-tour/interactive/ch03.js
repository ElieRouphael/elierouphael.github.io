/* A Tour of Machine Learning, chapter 03, Support vector machines: the live demos.
   The SVMs are trained in the browser with the SMO algorithm of LIBSVM (second-order working
   set selection, Fan, Chen and Lin 2005), as scikit-learn's SVC does, with one SVM per pair of
   classes and a vote. The iris rows are the notebook's: training flowers first, in the order
   train_test_split(test_size=0.3, random_state=42, stratify=y) returns them, then the test flowers.
   Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;

/* ---------------- the data: [petal length, petal width, species, in the training set?] ---------------- */
const ROWS = [[3.0,1.1,1,1],[4.5,1.5,1,1],[1.5,0.3,0,1],[5.9,2.3,2,1],[4.1,1.3,1,1],[5.2,2.3,2,1],[1.5,0.1,0,1],[1.6,0.2,0,1],[1.4,0.2,0,1],[5.9,2.1,2,1],[5.1,2.0,2,1],[1.4,0.2,0,1],[1.6,0.2,0,1],[4.5,1.5,1,1],[3.8,1.1,1,1],[6.0,1.8,2,1],[1.6,0.4,0,1],[1.6,0.2,0,1],[5.7,2.5,2,1],[4.8,1.8,1,1],[1.5,0.4,0,1],[4.9,1.8,2,1],[6.6,2.1,2,1],[6.7,2.0,2,1],[4.2,1.2,1,1],[1.4,0.3,0,1],[1.5,0.4,0,1],[1.3,0.4,0,1],[4.9,1.5,1,1],[4.0,1.3,1,1],[1.5,0.4,0,1],[1.4,0.3,0,1],[4.1,1.0,1,1],[3.6,1.3,1,1],[1.4,0.1,0,1],[1.4,0.3,0,1],[4.4,1.4,1,1],[5.0,1.9,2,1],[6.4,2.0,2,1],[1.4,0.2,0,1],[5.6,2.1,2,1],[1.4,0.2,0,1],[5.0,1.5,2,1],[1.3,0.2,0,1],[6.0,2.5,2,1],[4.0,1.0,1,1],[1.9,0.2,0,1],[6.1,2.3,2,1],[4.0,1.3,1,1],[5.0,2.0,2,1],[3.9,1.2,1,1],[1.3,0.3,0,1],[3.7,1.0,1,1],[5.5,1.8,2,1],[4.6,1.4,1,1],[5.6,1.8,2,1],[1.5,0.2,0,1],[4.2,1.3,1,1],[1.4,0.2,0,1],[4.7,1.5,1,1],[4.8,1.4,1,1],[3.9,1.4,1,1],[6.1,2.5,2,1],[5.1,1.6,1,1],[4.3,1.3,1,1],[5.6,2.4,2,1],[4.9,2.0,2,1],[1.5,0.2,0,1],[5.7,2.3,2,1],[4.6,1.5,1,1],[4.1,1.3,1,1],[6.9,2.3,2,1],[1.5,0.1,0,1],[5.3,2.3,2,1],[5.1,1.8,2,1],[4.4,1.3,1,1],[1.6,0.2,0,1],[6.7,2.2,2,1],[5.5,2.1,2,1],[1.9,0.4,0,1],[1.1,0.1,0,1],[4.8,1.8,2,1],[5.8,1.6,2,1],[5.1,2.4,2,1],[1.7,0.4,0,1],[5.1,1.9,2,1],[3.5,1.0,1,1],[5.7,2.1,2,1],[6.1,1.9,2,1],[1.5,0.2,0,1],[4.7,1.2,1,1],[4.4,1.2,1,1],[4.7,1.4,1,1],[4.9,1.5,1,1],[4.2,1.5,1,1],[1.5,0.2,0,1],[5.4,2.1,2,1],[4.0,1.2,1,1],[5.1,1.9,2,1],[1.6,0.2,0,1],[1.4,0.1,0,1],[3.5,1.0,1,1],[1.4,0.2,0,1],[4.2,1.3,1,1],[1.5,0.2,0,1],[6.3,1.8,2,0],[4.7,1.4,1,0],[5.1,1.5,2,0],[4.7,1.6,1,0],[4.9,1.8,2,0],[5.6,2.4,2,0],[4.0,1.3,1,0],[3.9,1.1,1,0],[1.7,0.2,0,0],[5.1,2.3,2,0],[1.2,0.2,0,0],[1.3,0.2,0,0],[5.8,1.8,2,0],[5.5,1.8,2,0],[1.4,0.2,0,0],[5.4,2.3,2,0],[3.3,1.0,1,0],[1.5,0.2,0,0],[1.7,0.5,0,0],[1.7,0.3,0,0],[4.3,1.3,1,0],[1.5,0.2,0,0],[4.4,1.4,1,0],[5.8,2.2,2,0],[4.8,1.8,2,0],[4.5,1.5,1,0],[4.5,1.5,1,0],[3.3,1.0,1,0],[4.5,1.5,1,0],[1.2,0.2,0,0],[5.6,1.4,2,0],[5.6,2.2,2,0],[4.5,1.6,1,0],[1.4,0.2,0,0],[5.3,1.9,2,0],[1.3,0.3,0,0],[1.3,0.2,0,0],[1.3,0.2,0,0],[1.6,0.6,0,0],[5.0,1.7,1,0],[4.5,1.3,1,0],[1.0,0.2,0,0],[4.5,1.7,2,0],[5.2,2.0,2,0],[4.6,1.3,1,0]];
const TRAIN = ROWS.filter(r => r[3]), TEST = ROWS.filter(r => !r[3]);
const SPECIES = ["setosa", "versicolor", "virginica"], COLS = ["--post", "--warm", "--c4"];
/* gamma="scale": 1 / (n_features * variance of all entries of X_train) */
const GSCALE = (() => { const v = TRAIN.flatMap(r => [r[0], r[1]]), m = v.reduce((a, b) => a + b, 0) / v.length; return 1 / (2 * v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); })();

/* ---------------- the solver ---------------- */
const rbf = g => (a, b) => Math.exp(-g * ((a[0] - b[0]) ** 2 + ((a[1] || 0) - (b[1] || 0)) ** 2));
const linear = (a, b) => a[0] * b[0] + (a[1] || 0) * (b[1] || 0);
/* the dual of the soft-margin SVM, by SMO with LIBSVM's second-order working-set selection.
   Returns alpha and rho, with f(x) = sum_t alpha_t y_t k(x_t, x) - rho. */
function smo(X, y, C, kern, eps = 1e-3) {
  const n = X.length, K = X.map(a => X.map(b => kern(a, b))), QD = K.map((r, i) => r[i]), TAU = 1e-12;
  const alpha = new Float64Array(n), G = new Float64Array(n).fill(-1);
  const upper = t => alpha[t] >= C, lower = t => alpha[t] <= 0;
  for (let it = 0; it < 200000; it++) {
    let Gmax = -Infinity, Gmax2 = -Infinity, i = -1, j = -1, best = Infinity;
    for (let t = 0; t < n; t++) {
      if (y[t] === 1) { if (!upper(t) && -G[t] >= Gmax) { Gmax = -G[t]; i = t; } }
      else if (!lower(t) && G[t] >= Gmax) { Gmax = G[t]; i = t; }
    }
    if (i < 0) break;
    for (let t = 0; t < n; t++) {
      const yk = y[t] * K[i][t];                                           // y_i Q_it, with Q_it = y_i y_t K_it
      if (y[t] === 1) {
        if (lower(t)) continue;
        const gd = Gmax + G[t]; if (G[t] >= Gmax2) Gmax2 = G[t];
        if (gd > 0) { const qc = QD[i] + QD[t] - 2 * yk, od = -(gd * gd) / (qc > 0 ? qc : TAU); if (od <= best) { j = t; best = od; } }
      } else {
        if (upper(t)) continue;
        const gd = Gmax - G[t]; if (-G[t] >= Gmax2) Gmax2 = -G[t];
        if (gd > 0) { const qc = QD[i] + QD[t] + 2 * yk, od = -(gd * gd) / (qc > 0 ? qc : TAU); if (od <= best) { j = t; best = od; } }
      }
    }
    if (Gmax + Gmax2 < eps || j < 0) break;
    const Qij = y[i] * y[j] * K[i][j], ai = alpha[i], aj = alpha[j];
    if (y[i] !== y[j]) {
      let qc = QD[i] + QD[j] + 2 * Qij; if (qc <= 0) qc = TAU;
      const delta = (-G[i] - G[j]) / qc, diff = ai - aj;
      alpha[i] += delta; alpha[j] += delta;
      if (diff > 0) { if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = diff; } } else if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = -diff; }
      if (diff > 0) { if (alpha[i] > C) { alpha[i] = C; alpha[j] = C - diff; } } else if (alpha[j] > C) { alpha[j] = C; alpha[i] = C + diff; }
    } else {
      let qc = QD[i] + QD[j] - 2 * Qij; if (qc <= 0) qc = TAU;
      const delta = (G[i] - G[j]) / qc, sum = ai + aj;
      alpha[i] -= delta; alpha[j] += delta;
      if (sum > C) { if (alpha[i] > C) { alpha[i] = C; alpha[j] = sum - C; } } else if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = sum; }
      if (sum > C) { if (alpha[j] > C) { alpha[j] = C; alpha[i] = sum - C; } } else if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = sum; }
    }
    const di = alpha[i] - ai, dj = alpha[j] - aj;
    for (let t = 0; t < n; t++) G[t] += y[t] * (y[i] * K[i][t] * di + y[j] * K[j][t] * dj);
  }
  let ub = Infinity, lb = -Infinity, nf = 0, sf = 0;
  for (let t = 0; t < n; t++) {
    const yG = y[t] * G[t];
    if (upper(t)) { if (y[t] === -1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
    else if (lower(t)) { if (y[t] === 1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
    else { nf++; sf += yG; }
  }
  return { alpha, rho: nf > 0 ? sf / nf : (ub + lb) / 2 };
}
/* one SVM per pair of classes (class a as +1, class b as -1), and a vote, as LIBSVM */
function trainOvO(points, labels, C, kern) {
  const K = Math.max(...labels) + 1, models = [], isSV = new Array(points.length).fill(false);
  for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) {
    const idx = labels.map((l, i) => i).filter(i => labels[i] === a).concat(labels.map((l, i) => i).filter(i => labels[i] === b));
    const { alpha, rho } = smo(idx.map(i => points[i]), idx.map(i => (labels[i] === a ? 1 : -1)), C, kern);
    const sv = [];
    idx.forEach((i, t) => { if (alpha[t] > 0) { sv.push({ x: points[i], c: alpha[t] * (labels[i] === a ? 1 : -1) }); isSV[i] = true; } });
    models.push({ a, b, sv, rho });
  }
  const predict = x => {
    const votes = new Array(K).fill(0);
    for (const m of models) { let f = -m.rho; for (const s of m.sv) f += s.c * kern(s.x, x); votes[f > 0 ? m.a : m.b]++; }
    let best = 0; for (let k = 1; k < K; k++) if (votes[k] > votes[best]) best = k;
    return best;
  };
  return { models, isSV, predict };
}

/* ---------------- shared helpers ---------------- */
function dot(ctx, x, y, r, fill, stroke, lw = 2) { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); } }
function diamond(ctx, x, y, r, fill, stroke) { ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); } }
const range = (n, f) => Array.from({ length: n }, (_, i) => f(i));
const cfmt = v => (v >= 100 ? fmt(v, 0) : String(+v.toPrecision(3)));

/* =====================================================================
   THE DEMO: AN SVM ON THE IRIS PETALS
   ===================================================================== */
const hero = { kernel: "rbf", C: 1, gamma: GSCALE, showTest: true, model: null, key: "" };
function trainHero() {
  const key = `${hero.kernel}|${hero.C}|${hero.gamma}`;
  if (key === hero.key) return hero.model;
  hero.key = key;
  hero.model = trainOvO(TRAIN.map(r => [r[0], r[1]]), TRAIN.map(r => r[2]), hero.C, hero.kernel === "rbf" ? rbf(hero.gamma) : linear);
  return hero.model;
}
function drawHero() {
  const M = trainHero(), cv = $("#irisC"), narrow = narrowOf(cv);
  $$("#demo [data-kernel]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.kernel === hero.kernel)));
  $("#hGField").style.opacity = hero.kernel === "rbf" ? "1" : "0.45"; $("#hG").disabled = hero.kernel !== "rbf";
  $("#hCV").textContent = cfmt(hero.C); $("#hGV").textContent = `${cfmt(hero.gamma)} per cm²${Math.abs(hero.gamma - GSCALE) < 1e-12 ? ' ("scale")' : ""}`;
  const fr = frame(cv, narrow ? 1.0 : 1.7, { x: [0.5, 7.4], y: [-0.4, 3.0], xlabel: "petal length (cm)", ylabel: "petal width (cm)", yticks: [0, 1, 2, 3] });
  const { ctx, X, Y } = fr, cell = narrow ? 4 : 5, shades = COLS.map(c => alpha(c, 0.2));
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  for (let px = fr.pad.l; px < fr.pad.l + fr.iw; px += cell) for (let py = fr.pad.t; py < fr.pad.t + fr.ih; py += cell) {
    const x = [fr.x0 + (px + cell / 2 - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), fr.y1 - (py + cell / 2 - fr.pad.t) / fr.ih * (fr.y1 - fr.y0)];
    ctx.fillStyle = shades[M.predict(x)]; ctx.fillRect(px, py, cell, cell);
  }
  ctx.restore();
  TRAIN.forEach((r, i) => { dot(ctx, X(r[0]), Y(r[1]), 3.6, css(COLS[r[2]]), css("--surface"), 0.8); if (M.isSV[i]) dot(ctx, X(r[0]), Y(r[1]), 7, null, css("--ink"), 1.3); });
  const trainHit = TRAIN.filter(r => M.predict([r[0], r[1]]) === r[2]).length, testPred = TEST.map(r => M.predict([r[0], r[1]]));
  const testHit = testPred.filter((p, i) => p === TEST[i][2]).length;
  if (hero.showTest) TEST.forEach((r, i) => {
    diamond(ctx, X(r[0]), Y(r[1]), 4.5, css(COLS[r[2]]), css("--surface"));
  });
  if (hero.showTest) TEST.forEach((r, i) => {                                       // the errors on top, crossed with a halo
    if (testPred[i] === r[2]) return;
    const x = X(r[0]), y = Y(r[1]), d = 8;
    [[css("--surface"), 4.5], [css("--ink"), 2.2]].forEach(([c, w]) => { ctx.strokeStyle = c; ctx.lineWidth = w; seg(ctx, x - d, y - d, x + d, y + d); seg(ctx, x - d, y + d, x + d, y - d); });
  });
  const nsv = [0, 1, 2].map(k => TRAIN.filter((r, i) => r[2] === k && M.isSV[i]).length), total = nsv.reduce((a, b) => a + b, 0);
  $("#rTr").textContent = `${fmt(trainHit / TRAIN.length, 3)} (${trainHit} of 105)`; $("#rTe").textContent = `${fmt(testHit / TEST.length, 3)} (${testHit} of 45)`;
  $("#rSV").textContent = `${total} (${nsv.join(", ")})`;
  const wrong = TEST.map((r, i) => (testPred[i] !== r[2] ? `${SPECIES[r[2]]} at (${fmt(r[0], 1)}, ${fmt(r[1], 1)}) taken for ${SPECIES[testPred[i]]}` : null)).filter(Boolean);
  let text = `${hero.kernel === "rbf" ? `RBF kernel with [[\\gamma = ${cfmt(hero.gamma)}]]` : "Linear kernel"} and [[C = ${cfmt(hero.C)}]]: <b>${testHit} of 45</b> test flowers right, with ${total} of the 105 training flowers as support vectors (circled). `;
  if (hero.kernel === "rbf" && hero.gamma >= 20) text += "The bumps are so narrow that the boundary wraps islands around single flowers: the training accuracy climbs, the test accuracy does not. ";
  else if (total >= 100) text += "Violations are so cheap that almost every flower sits inside the margin, and every one of them is a support vector. ";
  else if (hero.C >= 100) text += "Violations are so expensive that the margin narrows to the few flowers near the boundary. ";
  text += wrong.length ? `Misclassified test flowers (crossed): ${wrong.join("; ")}.` : "No test flower is misclassified.";
  caption($("#heroStatus"), text);
}
$$("#demo [data-kernel]").forEach(b => b.addEventListener("click", () => { hero.kernel = b.dataset.kernel; drawHero(); }));
$("#hC").addEventListener("input", e => { hero.C = 10 ** +e.target.value; drawHero(); });
$("#hG").addEventListener("input", e => { hero.gamma = 10 ** +e.target.value; drawHero(); });
$("#hScale").addEventListener("click", () => { hero.kernel = "rbf"; hero.C = 1; hero.gamma = GSCALE; $("#hC").value = "0"; $("#hG").value = String(Math.log10(GSCALE)); drawHero(); });
$("#hTest").addEventListener("click", () => { hero.showTest = !hero.showTest; $("#hTest").setAttribute("aria-pressed", String(hero.showTest)); drawHero(); });

/* =====================================================================
   PART 2: FIND THE WIDEST STREET (the four points of the hand example)
   ===================================================================== */
const PTS = [["A", 1, 1, -1], ["B", 2, 1, -1], ["C", 3, 2, 1], ["D", 4, 3, 1]], CTR = [2.5, 1.75];
const street = { theta: 120 * Math.PI / 180, off: 0.3 };
function drawStreet() {
  const n = [Math.cos(street.theta), Math.sin(street.theta)], c = n[0] * CTR[0] + n[1] * CTR[1] + street.off;
  $("#stAV").textContent = `${Math.round(street.theta * 180 / Math.PI)}°`; $("#stOV").textContent = fmt(street.off, 2).replace("-", "−");
  // signed distances to the line n.x = c, oriented so that class +1 is on the positive side if it can be
  let d = PTS.map(p => n[0] * p[1] + n[1] * p[2] - c);
  let m = Math.min(...PTS.map((p, i) => p[3] * d[i])), mFlip = Math.min(...PTS.map((p, i) => -p[3] * d[i]));
  const flip = mFlip > m; if (flip) { d = d.map(v => -v); m = mFlip; }
  const cv = $("#streetC"), w = cv.clientWidth || 300, asp = w / (62 + 0.8 * Math.max(60, w - 54));
  const fr = frame(cv, asp, { x: [0, 5], y: [0, 4], xlabel: "x₁", ylabel: "x₂", xticks: [0, 1, 2, 3, 4, 5], yticks: [0, 1, 2, 3, 4] });
  const { ctx, X, Y } = fr;
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  const u = [-n[1], n[0]], P = s => [c * n[0] + s * u[0], c * n[1] + s * u[1]];            // points along the line
  const lineAt = off => { const a = P(-20), b = P(20); return [[a[0] + off * n[0], a[1] + off * n[1]], [b[0] + off * n[0], b[1] + off * n[1]]]; };
  if (m > 0) {
    const [a1, b1] = lineAt(m), [a2, b2] = lineAt(-m);
    ctx.fillStyle = alpha("--accent", 0.12); ctx.beginPath(); ctx.moveTo(X(a1[0]), Y(a1[1])); ctx.lineTo(X(b1[0]), Y(b1[1])); ctx.lineTo(X(b2[0]), Y(b2[1])); ctx.lineTo(X(a2[0]), Y(a2[1])); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = alpha("--ink", 0.5); ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
    seg(ctx, X(a1[0]), Y(a1[1]), X(b1[0]), Y(b1[1])); seg(ctx, X(a2[0]), Y(a2[1]), X(b2[0]), Y(b2[1])); ctx.setLineDash([]);
  }
  const [a0, b0] = lineAt(0);
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2.2; seg(ctx, X(a0[0]), Y(a0[1]), X(b0[0]), Y(b0[1]));
  ctx.restore();
  ctx.font = "700 13px " + css("--font-ui"); ctx.textAlign = "left";
  PTS.forEach((p, i) => {
    const touch = m > 0 && Math.abs(p[3] * d[i] - m) < 1e-6;
    dot(ctx, X(p[1]), Y(p[2]), 6.5, css(p[3] > 0 ? "--warm" : "--post"), touch ? css("--ink") : css("--surface"), touch ? 2.4 : 1.5);
    ctx.fillStyle = css("--ink"); ctx.fillText(p[0], X(p[1]) + 9, Y(p[2]) - 7);
  });
  const touching = PTS.filter((p, i) => m > 0 && Math.abs(p[3] * d[i] - m) < 1e-6).map(p => p[0]);
  let text;
  if (m <= 1e-9) text = `This line leaves ${PTS.filter((p, i) => p[3] * d[i] <= 1e-9).map(p => p[0]).join(" and ")} on the wrong side or on the line itself: it does not separate the classes, so it has no street at all.`;
  else {
    text = `The street is <b>${fmt(2 * m, 3)}</b> wide: ${fmt(m, 3)} on each side, up to ${touching.join(" and ")}. `;
    text += Math.abs(m - Math.SQRT1_2) < 1e-6 ? `This is the SVM's line, [[x_1 + x_2 = 4]]: no line does better, since B and C are only [[\\sqrt{2} \\approx 1.414]] apart. B and C, on its edges, are the support vectors.` : `The widest possible is 1.414, the SVM's line. Turn and slide the line to widen the street: it is pinned by whichever points touch its edges.`;
  }
  caption($("#streetCap"), text);
}
$("#stA").addEventListener("input", e => { street.theta = +e.target.value * Math.PI / 180; drawStreet(); });
$("#stO").addEventListener("input", e => { street.off = +e.target.value; drawStreet(); });
$("#stBest").addEventListener("click", () => {
  street.theta = Math.PI / 4; street.off = 4 / Math.SQRT2 - (CTR[0] + CTR[1]) / Math.SQRT2;         // n.x = 4 / sqrt(2)
  $("#stA").value = "45"; $("#stO").value = String(street.off); drawStreet();
});

/* =====================================================================
   PART 3: ONE BUMP PER SUPPORT VECTOR (a one-dimensional SVM)
   ===================================================================== */
const LINE = [[0.4, -1], [1.0, -1], [1.6, -1], [2.4, 1], [2.9, 1], [3.3, 1], [3.8, 1], [4.5, -1], [5.1, -1], [5.6, -1]];
function drawBumps() {
  const g = 10 ** +$("#bG").value, C = 10 ** +$("#bC").value, kern = rbf(g);
  $("#bGV").textContent = cfmt(g); $("#bCV").textContent = cfmt(C);
  const { alpha: al, rho } = smo(LINE.map(p => [p[0]]), LINE.map(p => p[1]), C, kern);
  const xs = range(241, i => i * 6 / 240), f = x => LINE.reduce((s, p, i) => s + al[i] * p[1] * kern([p[0]], [x]), -rho);
  const fs = xs.map(f), sv = LINE.map((p, i) => i).filter(i => al[i] > 0);
  const top = Math.max(1.5, ...fs.map(Math.abs), ...sv.map(i => al[i])) * 1.1;
  const cv = $("#bumpsC"), fr = frame(cv, narrowOf(cv) ? 1.3 : 2.4, { x: [0, 6], y: [-top, top], xlabel: "x", yticks: [-1, 0, 1] });
  const { ctx, X, Y } = fr;
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  for (let k = 0; k < xs.length - 1; k++) { ctx.fillStyle = alpha(fs[k] > 0 ? "--warm" : "--post", 0.1); ctx.fillRect(X(xs[k]), fr.pad.t, X(xs[k + 1]) - X(xs[k]) + 0.5, fr.ih); }
  ctx.restore();
  ctx.strokeStyle = alpha("--ink", 0.4); ctx.lineWidth = 1; seg(ctx, fr.pad.l, Y(0), fr.pad.l + fr.iw, Y(0));
  sv.forEach(i => plotLine(fr, xs, xs.map(x => al[i] * LINE[i][1] * kern([LINE[i][0]], [x])), { color: alpha(LINE[i][1] > 0 ? "--warm" : "--post", 0.55), width: 1.2 }));
  plotLine(fr, xs, fs, { color: css("--ink"), width: 2.4 });
  LINE.forEach((p, i) => dot(ctx, X(p[0]), Y(0), 5.5, css(p[1] > 0 ? "--warm" : "--post"), al[i] > 0 ? css("--ink") : css("--surface"), al[i] > 0 ? 2 : 1));
  const wrong = LINE.filter((p, i) => Math.sign(f(p[0])) !== p[1]).length;
  let cross = 0; for (let k = 1; k < fs.length; k++) if (Math.sign(fs[k]) !== Math.sign(fs[k - 1])) cross++;
  let text = `With [[\\gamma = ${cfmt(g)}]] (each bump reaching about ${cfmt(1 / Math.sqrt(g))}) and [[C = ${cfmt(C)}]]: <b>${sv.length} support vectors</b> (outlined) of the 10 points, each adding a bump of height [[\\alpha_i y_i]]; their sum, minus a constant, is [[f(x)]], and it crosses zero ${cross} time${cross === 1 ? "" : "s"}. `;
  if (wrong) text += `<b>${wrong} point${wrong > 1 ? "s are" : " is"} on the wrong side</b>: at this price [[C]], a violation is cheaper than bending the sum around the middle class. Raise [[C]], and it bends.`;
  else if (g > 8) text += "The bumps are so narrow that each point gets its own; between the points, f falls back to its constant, and the boundary hugs the training points.";
  else if (g < 0.1) text += "The bumps are so wide that their sum is a gentle, parabola-like curve, yet still bent enough to wrap the middle class.";
  else text += "No single threshold separates the middle class from the two outer groups, but two zero crossings of the sum do.";
  caption($("#bumpsCap"), text);
}
["#bG", "#bC"].forEach(id => $(id).addEventListener("input", drawBumps));

/* =====================================================================
   start-up
   ===================================================================== */
function drawAll() { drawHero(); drawStreet(); drawBumps(); }
$("#hG").value = String(Math.log10(GSCALE));
drawAll();
onRedraw(drawAll);
})();
