/* A Tour of Machine Learning, chapter 07, Transfer learning: the live demos.
   The source CNN is the notebook's, run here with its trained weights (ch07-data.js): conv1
   (16 kernels, 3x3, same padding, ReLU), 2x2 max pooling, conv2 (32 kernels), global average
   pooling, a dense layer of 32 ReLU features and the softmax head. The few-shot race fits the
   notebook's classifier, a multinomial logistic regression on standardised inputs with the
   scikit-learn default L2 penalty (C = 1), by L-BFGS, on 600 of the notebook's shifted digits.
   Needs ../kit/kit.js, ../kit/stats.js and ch07-data.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp, fit } = window.Kit;
const { makeRng } = window.Stats;
const D = window.CH07;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
function decodeImages(b64) {
  const s = atob(b64), n = s.length / 64, out = [];
  for (let i = 0; i < n; i++) { const im = new Float32Array(64); for (let j = 0; j < 64; j++) im[j] = s.charCodeAt(i * 64 + j) / 255; out.push(im); }
  return out;
}
/* Keras kernels are [kh][kw][Cin][Cout]; flattened so that the output channel runs fastest. */
function convW(k, b) {
  const Cin = k[0][0].length, Cout = k[0][0][0].length, w = new Float32Array(9 * Cin * Cout);
  for (let m = 0; m < 3; m++) for (let n = 0; n < 3; n++) for (let ci = 0; ci < Cin; ci++) for (let co = 0; co < Cout; co++) w[((m * 3 + n) * Cin + ci) * Cout + co] = k[m][n][ci][co];
  return { w, b: Float32Array.from(b), Cin, Cout };
}
function denseW(k, b) {
  const In = k.length, Out = k[0].length, w = new Float32Array(In * Out);
  for (let i = 0; i < In; i++) for (let j = 0; j < Out; j++) w[i * Out + j] = k[i][j];
  return { w, b: Float32Array.from(b), In, Out };
}
const makeNet = W => ({ c1: convW(W.conv1[0], W.conv1[1]), c2: convW(W.conv2[0], W.conv2[1]), fe: denseW(W.feat[0], W.feat[1]), hd: denseW(W.source_head[0], W.source_head[1]) });
/* 3x3 cross-correlation with one pixel of zero padding ("same"), bias, ReLU. x is H*W*Cin, channel last. */
function conv(x, H, Wd, L) {
  const { w, b, Cin, Cout } = L, out = new Float32Array(H * Wd * Cout);
  for (let i = 0; i < H; i++) for (let j = 0; j < Wd; j++) {
    const o = (i * Wd + j) * Cout;
    for (let co = 0; co < Cout; co++) out[o + co] = b[co];
    for (let m = 0; m < 3; m++) {
      const ii = i + m - 1; if (ii < 0 || ii >= H) continue;
      for (let n = 0; n < 3; n++) {
        const jj = j + n - 1; if (jj < 0 || jj >= Wd) continue;
        const xo = (ii * Wd + jj) * Cin, wo = (m * 3 + n) * Cin;
        for (let ci = 0; ci < Cin; ci++) { const xv = x[xo + ci]; if (xv === 0) continue; const wb = (wo + ci) * Cout; for (let co = 0; co < Cout; co++) out[o + co] += xv * w[wb + co]; }
      }
    }
    for (let co = 0; co < Cout; co++) if (out[o + co] < 0) out[o + co] = 0;
  }
  return out;
}
function maxpool(x, H, Wd, C) {
  const h = H / 2, w2 = Wd / 2, out = new Float32Array(h * w2 * C);
  for (let i = 0; i < h; i++) for (let j = 0; j < w2; j++) for (let c = 0; c < C; c++) {
    const a = x[((2 * i) * Wd + 2 * j) * C + c], b = x[((2 * i) * Wd + 2 * j + 1) * C + c], d = x[((2 * i + 1) * Wd + 2 * j) * C + c], e = x[((2 * i + 1) * Wd + 2 * j + 1) * C + c];
    out[(i * w2 + j) * C + c] = Math.max(a, b, d, e);
  }
  return out;
}
function gap(x, HW, C) { const out = new Float32Array(C); for (let p = 0; p < HW; p++) for (let c = 0; c < C; c++) out[c] += x[p * C + c]; for (let c = 0; c < C; c++) out[c] /= HW; return out; }
function dense(v, L, relu) {
  const { w, b, In, Out } = L, out = Float32Array.from(b);
  for (let i = 0; i < In; i++) { const vi = v[i]; if (vi === 0) continue; for (let j = 0; j < Out; j++) out[j] += vi * w[i * Out + j]; }
  if (relu) for (let j = 0; j < Out; j++) if (out[j] < 0) out[j] = 0;
  return out;
}
function softmax(s) { let mx = -Infinity; for (const v of s) mx = Math.max(mx, v); const e = Array.from(s, v => Math.exp(v - mx)), Z = e.reduce((a, b) => a + b, 0); return e.map(v => v / Z); }
function forwardWith(net, img) {
  const a1 = conv(img, 8, 8, net.c1), p1 = maxpool(a1, 8, 8, net.c1.Cout), a2 = conv(p1, 4, 4, net.c2), g = gap(a2, 16, net.c2.Cout), f = dense(g, net.fe, true), p = softmax(dense(f, net.hd, false));
  return { a1, p1, a2, g, f, p };
}

/* ---- multinomial logistic regression, as make_pipeline(StandardScaler(), LogisticRegression()):
   minimise sum_i -log p(y_i | x_i) + (1/2) ||W||^2 over W (d x K) and the unpenalised intercept b. ---- */
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function lbfgs(fg, x0, { maxIter = 400, m = 10, gtol = 1e-5 } = {}) {
  let x = Float64Array.from(x0), [f, g] = fg(x); const S = [], Yv = [], R = [];
  for (let it = 0; it < maxIter; it++) {
    let gmax = 0; for (let i = 0; i < g.length; i++) gmax = Math.max(gmax, Math.abs(g[i]));
    if (gmax < gtol) break;
    const q = Float64Array.from(g), al = new Array(S.length);
    for (let i = S.length - 1; i >= 0; i--) { al[i] = R[i] * dot(S[i], q); const yv = Yv[i]; for (let j = 0; j < q.length; j++) q[j] -= al[i] * yv[j]; }
    const gamma = S.length ? dot(S[S.length - 1], Yv[Yv.length - 1]) / dot(Yv[Yv.length - 1], Yv[Yv.length - 1]) : 1 / Math.max(1, gmax);
    for (let j = 0; j < q.length; j++) q[j] *= gamma;
    for (let i = 0; i < S.length; i++) { const be = R[i] * dot(Yv[i], q), s = S[i]; for (let j = 0; j < q.length; j++) q[j] += (al[i] - be) * s[j]; }
    const d = q; for (let j = 0; j < d.length; j++) d[j] = -d[j];
    let gd = dot(g, d);
    if (!(gd < 0)) { S.length = Yv.length = R.length = 0; for (let j = 0; j < d.length; j++) d[j] = -g[j]; gd = dot(g, d); }
    let t = 1, xn, fn, gn;
    for (let ls = 0; ls < 40; ls++) { xn = x.map((v, j) => v + t * d[j]); [fn, gn] = fg(xn); if (fn <= f + 1e-4 * t * gd) break; t *= 0.5; }
    const s = xn.map((v, j) => v - x[j]), yv = gn.map((v, j) => v - g[j]), sy = dot(s, yv);
    if (sy > 1e-12) { S.push(s); Yv.push(yv); R.push(1 / sy); if (S.length > m) { S.shift(); Yv.shift(); R.shift(); } }
    x = xn; f = fn; g = gn;
  }
  return { x, f };
}
/* feats: array of vectors; tr, te: index arrays; y: labels. Returns test accuracy and the objective at the optimum. */
function fitLogReg(feats, y, tr, te, K = 10) {
  const d = feats[0].length, n = tr.length, mean = new Float64Array(d), sd = new Float64Array(d);
  for (const i of tr) for (let j = 0; j < d; j++) mean[j] += feats[i][j] / n;
  for (const i of tr) for (let j = 0; j < d; j++) sd[j] += (feats[i][j] - mean[j]) ** 2 / n;
  for (let j = 0; j < d; j++) sd[j] = sd[j] > 1e-16 ? Math.sqrt(sd[j]) : 1;        // StandardScaler leaves constant columns alone
  const Xs = new Float64Array(n * d), ys = tr.map(i => y[i]);
  tr.forEach((i, r) => { for (let j = 0; j < d; j++) Xs[r * d + j] = (feats[i][j] - mean[j]) / sd[j]; });
  const lg = new Float64Array(K);
  const fg = th => {
    const grad = new Float64Array(th.length); let f = 0;
    for (let r = 0; r < n; r++) {
      const xo = r * d; let mx = -Infinity;
      for (let c = 0; c < K; c++) lg[c] = th[d * K + c];
      for (let j = 0; j < d; j++) { const xj = Xs[xo + j]; if (xj === 0) continue; const wo = j * K; for (let c = 0; c < K; c++) lg[c] += xj * th[wo + c]; }
      for (let c = 0; c < K; c++) if (lg[c] > mx) mx = lg[c];
      let Z = 0; for (let c = 0; c < K; c++) { lg[c] = Math.exp(lg[c] - mx); Z += lg[c]; }
      f += Math.log(Z) - Math.log(lg[ys[r]]);
      for (let c = 0; c < K; c++) { const rr = lg[c] / Z - (c === ys[r] ? 1 : 0); grad[d * K + c] += rr; lg[c] = rr; }
      for (let j = 0; j < d; j++) { const xj = Xs[xo + j]; if (xj === 0) continue; const wo = j * K; for (let c = 0; c < K; c++) grad[wo + c] += xj * lg[c]; }
    }
    for (let j = 0; j < d * K; j++) { f += 0.5 * th[j] * th[j]; grad[j] += th[j]; }
    return [f, grad];
  };
  const { x: th, f } = lbfgs(fg, new Float64Array(d * K + K));
  let correct = 0;
  for (const i of te) {
    let best = -1, bv = -Infinity;
    for (let c = 0; c < K; c++) { let v = th[d * K + c]; for (let j = 0; j < d; j++) v += (feats[i][j] - mean[j]) / sd[j] * th[j * K + c]; if (v > bv) { bv = v; best = c; } }
    if (best === y[i]) correct++;
  }
  return { acc: correct / te.length, obj: f };
}
/* NET-END */

const NET = makeNet(D.W);
const POOL = decodeImages(D.pool), POOL_Y = D.pool_y, CLEAN = decodeImages(D.clean), CLEAN_Y = D.clean_y;
const OUT = POOL.map(im => forwardWith(NET, im));
const FEAT = OUT.map(o => o.f), HEADPRED = OUT.map(o => o.p.indexOf(Math.max(...o.p)));
const BYCLASS = Array.from({ length: 10 }, (_, c) => POOL_Y.map((v, i) => v === c ? i : -1).filter(i => i >= 0));
const argmax = a => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const sdev = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); };

/* An untrained copy of the architecture: Keras's default Glorot-uniform kernels and zero biases, from a seeded generator. */
function randomNet(seed) {
  const rng = makeRng(seed), glorot = (fi, fo) => { const l = Math.sqrt(6 / (fi + fo)); return l * (2 * rng.uniform() - 1); };
  const ck = (Cin, Cout) => Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => Array.from({ length: Cin }, () => Array.from({ length: Cout }, () => glorot(9 * Cin, 9 * Cout)))));
  const dk = (In, Out) => Array.from({ length: In }, () => Array.from({ length: Out }, () => glorot(In, Out)));
  return makeNet({ conv1: [ck(1, 16), new Array(16).fill(0)], conv2: [ck(16, 32), new Array(32).fill(0)], feat: [dk(32, 32), new Array(32).fill(0)], source_head: D.W.source_head });
}
let RANDFEAT = null;

/* =====================================================================
   Hero: the few-shot race on the 600-digit pool
   ===================================================================== */
const KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25];
const NB = { 3: [0.161, 0.254], 5: [0.178, 0.320], 8: [0.167, 0.371], 15: [0.168, 0.416], 25: [0.184, 0.434] };
const curve = { raw: [], feat: [], rand: [], head: [], rawSd: [], featSd: [] };
let extra = null, draws = 0, randOn = false, randQueue = 0;

function splitsFor(k, seed, nTrials = 6) {
  const rng = makeRng(seed), out = [];
  for (let t = 0; t < nTrials; t++) {
    const tr = [], inTr = new Uint8Array(POOL.length);
    for (let c = 0; c < 10; c++) { const pool = BYCLASS[c].slice(); for (let q = 0; q < k; q++) { const j = q + Math.floor(rng.uniform() * (pool.length - q)); [pool[q], pool[j]] = [pool[j], pool[q]]; tr.push(pool[q]); inTr[pool[q]] = 1; } }
    const te = []; for (let i = 0; i < POOL.length; i++) if (!inTr[i]) te.push(i);
    out.push({ tr, te });
  }
  return out;
}
function evalK(k, seed, withRand) {
  const sp = splitsFor(k, seed), raw = [], feat = [], rand = [], head = [];
  for (const { tr, te } of sp) {
    raw.push(fitLogReg(POOL, POOL_Y, tr, te).acc); feat.push(fitLogReg(FEAT, POOL_Y, tr, te).acc);
    head.push(te.filter(i => HEADPRED[i] === POOL_Y[i]).length / te.length);
    if (withRand) rand.push(fitLogReg(RANDFEAT, POOL_Y, tr, te).acc);
  }
  return { raw: mean(raw), feat: mean(feat), rand: withRand ? mean(rand) : null, head: mean(head), rawSd: sdev(raw), featSd: sdev(feat) };
}
function computeNext() {
  const i = curve.raw.length; if (i >= KS.length) return;
  const r = evalK(KS[i], 100 + i, false);
  curve.raw.push(r.raw); curve.feat.push(r.feat); curve.head.push(r.head); curve.rawSd.push(r.rawSd); curve.featSd.push(r.featSd);
  drawHero(); setTimeout(computeNext, 0);
}
function computeRandNext() {
  const i = curve.rand.length; if (!randOn || i >= KS.length || i >= curve.raw.length) { if (randOn && i < KS.length) setTimeout(computeRandNext, 50); return; }
  const sp = splitsFor(KS[i], 100 + i); curve.rand.push(mean(sp.map(({ tr, te }) => fitLogReg(RANDFEAT, POOL_Y, tr, te).acc)));
  drawHero(); setTimeout(computeRandNext, 0);
}
const kNow = () => KS[+$("#hK").value];

function drawHero() {
  const k = kNow(), cv = $("#fsC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.25 : 1.75, { x: [0, 26], y: [0, 0.6], xticks: [1, 5, 10, 15, 20, 25], yticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], xlabel: "labelled examples per class", ylabel: "accuracy on the rest of the pool" });
  const { ctx, X, Y } = fr, n = curve.raw.length, ks = KS.slice(0, n);
  ctx.strokeStyle = alpha("--ink", 0.35); ctx.setLineDash([3, 3]); seg(ctx, X(0), Y(0.1), X(26), Y(0.1)); ctx.setLineDash([]);
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText("chance", X(0.4), Y(0.1) - 5);
  const band = (ys, sds, token) => { if (n < 2) return; ctx.save(); ctx.beginPath(); ks.forEach((x, i) => i ? ctx.lineTo(X(x), Y(ys[i] + sds[i])) : ctx.moveTo(X(x), Y(ys[i] + sds[i]))); for (let i = n - 1; i >= 0; i--) ctx.lineTo(X(ks[i]), Y(ys[i] - sds[i])); ctx.closePath(); ctx.fillStyle = alpha(token, 0.14); ctx.fill(); ctx.restore(); };
  band(curve.raw, curve.rawSd, "--muted"); band(curve.feat, curve.featSd, "--post");
  if (n) {
    plotLine(fr, ks, curve.head, { color: css("--warm"), width: 1.6, dash: [6, 4] });
    if (randOn && curve.rand.length) plotLine(fr, KS.slice(0, curve.rand.length), curve.rand, { color: css("--c5"), width: 2 });
    plotLine(fr, ks, curve.raw, { color: css("--muted"), width: 2.2 }); plotLine(fr, ks, curve.feat, { color: css("--post"), width: 2.4 });
  }
  const i = KS.indexOf(k), dotAt = (x, y, token, hollow) => { if (y == null) return; ctx.beginPath(); ctx.arc(X(x), Y(y), hollow ? 5 : 4.5, 0, 2 * Math.PI); if (hollow) { ctx.lineWidth = 2; ctx.strokeStyle = css(token); ctx.fillStyle = css("--surface"); ctx.fill(); ctx.stroke(); } else { ctx.fillStyle = css(token); ctx.fill(); } };
  ctx.strokeStyle = alpha("--ink", 0.2); seg(ctx, X(k), fr.pad.t, X(k), Y(0));
  if (i < n) { dotAt(k, curve.raw[i], "--muted"); dotAt(k, curve.feat[i], "--post"); if (randOn && i < curve.rand.length) dotAt(k, curve.rand[i], "--c5"); }
  if (extra && extra.k === k) { dotAt(k, extra.raw, "--muted", true); dotAt(k, extra.feat, "--post", true); if (extra.rand != null) dotAt(k, extra.rand, "--c5", true); }
  /* readouts and status */
  const src = extra && extra.k === k ? extra : i < n ? { raw: curve.raw[i], feat: curve.feat[i], rand: randOn && i < curve.rand.length ? curve.rand[i] : null, head: curve.head[i], rawSd: curve.rawSd[i], featSd: curve.featSd[i] } : null;
  $("#hKV").textContent = k; $("#hK").setAttribute("aria-valuetext", String(k));
  $("#rRaw").textContent = src ? fmt(src.raw, 3) : "…"; $("#rFeat").textContent = src ? fmt(src.feat, 3) : "…"; $("#rHead").textContent = src ? fmt(src.head, 3) : "…";
  $("#rRand").textContent = randOn ? (src && src.rand != null ? fmt(src.rand, 3) : "…") : "–";
  if (!src) { caption($("#heroStatus"), `Fitting the classifiers for ${k} labelled example${k > 1 ? "s" : ""} per class…`); return; }
  let text = `With ${k} labelled example${k > 1 ? "s" : ""} per class, ${10 * k} target labels in all${extra && extra.k === k ? ", on a fresh draw of the labels (hollow dots)" : ""}: logistic regression on the raw pixels scores ${fmt(src.raw, 3)}, on the frozen features ${fmt(src.feat, 3)}` + (src.rawSd != null ? ` (spread over the 6 draws: ±${fmt(src.rawSd, 3)} and ±${fmt(src.featSd, 3)})` : "") + `. `;
  if (src.feat <= src.head + 0.01) text += `That is no better than the source head applied unchanged, ${fmt(src.head, 3)}: the labels have not yet bought anything, the features have. `;
  else text += `The source head, applied unchanged, gets ${fmt(src.head, 3)}; the new head is ${fmt(src.feat - src.head, 3)} ahead of it, and ${fmt(src.feat / src.raw, 1)} times the raw-pixel score. `;
  if (randOn && src.rand != null) text += `The same architecture with random, untrained weights gives ${fmt(src.rand, 3)}: ${src.rand > src.raw ? "above the raw pixels, so pooling alone buys some shift tolerance" : "no better than the raw pixels"}, but ${fmt(src.feat - src.rand, 3)} below the trained features. `;
  if (NB[k]) text += `The notebook, with all 1,797 digits as the pool: ${fmt(NB[k][0], 3)} and ${fmt(NB[k][1], 3)}.`;
  caption($("#heroStatus"), text);
}
$("#hK").addEventListener("input", drawHero);
$("#hAgain").addEventListener("click", () => { const k = kNow(); const r = evalK(k, 1000 + (++draws), randOn && !!RANDFEAT); extra = Object.assign({ k }, r); drawHero(); });
$("#hRand").addEventListener("click", () => {
  randOn = !randOn; $("#hRand").setAttribute("aria-pressed", String(randOn)); $("#hRand").textContent = randOn ? "hide the untrained extractor" : "add the untrained extractor";
  if (randOn && !RANDFEAT) { const rn = randomNet(7); RANDFEAT = POOL.map(im => forwardWith(rn, im).f); }
  if (randOn && !curve.rand.length) setTimeout(computeRandNext, 0);
  if (extra && randOn && extra.rand == null) extra = null;
  drawHero();
});

/* =====================================================================
   Part 1: convolution by hand on a 5x5 image
   ===================================================================== */
const KERNELS = { v: [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], h: [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], d: [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]] };
const strokeImg = col => Array.from({ length: 5 }, () => Array.from({ length: 5 }, (_, j) => j === col ? 1 : 0));
const cs = { img: strokeImg(2), key: "v", sel: [0, 1], grids: [] };
/* The learned kernel shown: the conv1 kernel that answers the vertical stroke most strongly. */
const LEARNED = (() => {
  let best = 0, bv = -Infinity;
  for (let co = 0; co < 16; co++) { let v = NET.c1.b[co]; for (let m = 0; m < 3; m++) v += NET.c1.w[((m * 3 + 1) * 1) * 16 + co]; if (v > bv) { bv = v; best = co; } }
  return { i: best, b: NET.c1.b[best], k: Array.from({ length: 3 }, (_, m) => Array.from({ length: 3 }, (_, n) => NET.c1.w[(m * 3 + n) * 16 + best])) };
})();
const kernelNow = () => cs.key === "l" ? { k: LEARNED.k, b: LEARNED.b, learned: true } : { k: KERNELS[cs.key], b: 0, learned: false };
function convValid(img, k, b) { return Array.from({ length: 3 }, (_, r) => Array.from({ length: 3 }, (_, c) => { let v = b; for (let m = 0; m < 3; m++) for (let n = 0; n < 3; n++) v += k[m][n] * img[r + m][c + n]; return v; })); }
const num = (v, learned) => learned ? (Math.abs(v) < 0.005 ? "0" : v.toFixed(2)) : String(Math.round(v * 100) / 100);

function drawConv() {
  const cv = $("#convC"), w = cv.clientWidth || cv.parentElement.clientWidth, narrow = w < 560;
  const s = narrow ? Math.floor(Math.min(34, (w - 24) / 10)) : Math.floor(Math.min(42, (w - 90) / 14));
  const h = narrow ? 2 * 24 + 5 * s + 3 * s + 28 : 24 + 5 * s + 20;
  const { ctx } = fit(cv, w / h);
  const { k, b, learned } = kernelNow(), out = convValid(cs.img, k, b), kmax = Math.max(...k.flat().map(Math.abs)), omax = Math.max(1e-9, ...out.flat().map(Math.abs));
  ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "alphabetic";
  const grid = (name, x0, y0, cells, colour, label, fontPx) => {
    ctx.fillStyle = css("--muted"); ctx.textAlign = "left"; ctx.fillText(label, x0, y0 - 7);
    const n = cells.length, rect = { name, x0, y0, n, s };
    cells.forEach((row, i) => row.forEach((v, j) => {
      ctx.fillStyle = colour(v); ctx.fillRect(x0 + j * s, y0 + i * s, s, s);
      ctx.strokeStyle = alpha("--ink", 0.18); ctx.strokeRect(x0 + j * s + 0.5, y0 + i * s + 0.5, s - 1, s - 1);
      ctx.fillStyle = css("--ink"); ctx.font = `${fontPx}px ${css("--font-mono")}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(num(v, learned && name !== "img" && name !== "relu" ? true : (name === "relu" && learned)), x0 + j * s + s / 2, y0 + i * s + s / 2 + 0.5);
    }));
    ctx.textBaseline = "alphabetic"; ctx.font = "11px " + css("--font-mono");
    return rect;
  };
  const fnt = learned ? Math.max(8, Math.round(s * 0.3)) : Math.max(9, Math.round(s * 0.38));
  const imgCol = v => alpha("--ink", 0.08 + 0.72 * v), divCol = mx => v => v >= 0 ? alpha("--post", 0.12 + 0.78 * Math.min(1, v / mx)) : alpha("--warm", 0.12 + 0.78 * Math.min(1, -v / mx));
  const op = (txt, x, y) => { ctx.fillStyle = css("--ink"); ctx.font = `${Math.round(s * 0.45)}px ${css("--font-ui")}`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(txt, x, y); ctx.textBaseline = "alphabetic"; ctx.font = "11px " + css("--font-mono"); };
  cs.grids = [];
  if (narrow) {
    const x0 = Math.floor((w - 10 * s - 24) / 2), yA = 24, yB = yA + 5 * s + 28;
    cs.grids.push(grid("img", x0, yA, cs.img, imgCol, "image x (click a pixel)", fnt)); op("∗", x0 + 5 * s + 12, yA + 1.5 * s);
    cs.grids.push(grid("ker", x0 + 5 * s + 24, yA, k, divCol(kmax), learned ? `kernel ${LEARNED.i + 1} of conv1` : "kernel w", fnt));
    const xB = Math.floor((w - 6 * s - 24) / 2);
    cs.grids.push(grid("out", xB, yB, out, divCol(omax), "x ∗ w" + (learned ? " + b" : ""), fnt)); op("→", xB + 3 * s + 12, yB + 1.5 * s);
    cs.grids.push(grid("relu", xB + 3 * s + 24, yB, out.map(r => r.map(v => Math.max(0, v))), divCol(omax), "after the ReLU", fnt));
  } else {
    const total = 14 * s + 3 * 30, x0 = Math.floor((w - total) / 2), y0 = 24;
    cs.grids.push(grid("img", x0, y0, cs.img, imgCol, "image x (click a pixel)", fnt)); op("∗", x0 + 5 * s + 15, y0 + 1.5 * s);
    cs.grids.push(grid("ker", x0 + 5 * s + 30, y0, k, divCol(kmax), learned ? `kernel ${LEARNED.i + 1} of conv1` : "kernel w", fnt)); op("=", x0 + 8 * s + 45, y0 + 1.5 * s);
    cs.grids.push(grid("out", x0 + 8 * s + 60, y0, out, divCol(omax), "x ∗ w" + (learned ? " + b" : ""), fnt)); op("→", x0 + 11 * s + 75, y0 + 1.5 * s);
    cs.grids.push(grid("relu", x0 + 11 * s + 90, y0, out.map(r => r.map(v => Math.max(0, v))), divCol(omax), "after the ReLU", fnt));
  }
  /* highlight the selected output cell and the patch it reads */
  const [r, c] = cs.sel, gi = cs.grids[0], go = cs.grids[2], gr = cs.grids[3];
  ctx.lineWidth = 2.5; ctx.strokeStyle = css("--warm");
  ctx.strokeRect(gi.x0 + c * s + 1.25, gi.y0 + r * s + 1.25, 3 * s - 2.5, 3 * s - 2.5);
  ctx.strokeRect(go.x0 + c * s + 1.25, go.y0 + r * s + 1.25, s - 2.5, s - 2.5); ctx.strokeRect(gr.x0 + c * s + 1.25, gr.y0 + r * s + 1.25, s - 2.5, s - 2.5);
  ctx.lineWidth = 1;
  /* caption */
  const v = out[r][c], terms = [];
  for (let m = 0; m < 3; m++) for (let n = 0; n < 3; n++) { const x = cs.img[r + m][c + n]; if (x !== 0) terms.push(`${k[m][n] < 0 ? "(" + num(k[m][n], learned) + ")" : num(k[m][n], learned)}\\cdot ${x}`); }
  if (learned) terms.push(`b`);
  const sumTex = terms.length ? terms.join(" + ") : "0";
  const after = out.map(rw => rw.map(q => Math.max(0, q))), flat = after.flat(), mi = argmax(flat), all0 = flat.every(q => q <= 0);
  let text = `Output cell [[(${r}, ${c})]] reads the image rows ${r} to ${r + 2} and columns ${c} to ${c + 2} (outlined): [[${sumTex} = ${num(v, learned)}]]${learned ? ` with the kernel's bias [[b = ${num(b, true)}]]` : ""}, and after the ReLU [[\\max(0, ${num(v, learned)}) = ${num(Math.max(0, v), learned)}]]. `;
  if (cs.key === "v" && JSON.stringify(cs.img) === JSON.stringify(strokeImg(2))) text += "Each of the three rows contributes 2: the kernel's centre column sits on the stroke, its side columns on blank pixels. Every output row is (−3, 6, −3), and the map lights up over the stroke only.";
  else if (cs.key === "v" && JSON.stringify(cs.img) === JSON.stringify(strokeImg(3))) text += "The stroke moved one column right, and so did the 6: the response is the same, in the new place. That is equivariance, and max pooling would then make the pooled answer identical in both cases.";
  else if (all0) text += "Nothing in the image matches this kernel, so the map is dark everywhere.";
  else text += `The map is brightest at cell (${Math.floor(mi / 3)}, ${mi % 3}) with ${num(flat[mi], learned)}: the patch there looks most like the kernel${learned ? ", which the network learned by itself from the clean digits" : ""}.`;
  caption($("#convCap"), text);
}
$("#convC").addEventListener("click", ev => {
  const rc = ev.currentTarget.getBoundingClientRect(), x = ev.clientX - rc.left, y = ev.clientY - rc.top;
  for (const g of cs.grids) {
    const j = Math.floor((x - g.x0) / g.s), i = Math.floor((y - g.y0) / g.s);
    if (i < 0 || j < 0 || i >= g.n || j >= g.n) continue;
    if (g.name === "img") cs.img[i][j] = cs.img[i][j] ? 0 : 1; else if (g.name === "out" || g.name === "relu") cs.sel = [i, j]; else continue;
    drawConv(); return;
  }
});
$$("#convW .seg .btn").forEach(bt => bt.addEventListener("click", () => { cs.key = bt.dataset.k; $$("#convW .seg .btn").forEach(b => b.setAttribute("aria-pressed", String(b === bt))); drawConv(); }));
$("#cShift").addEventListener("click", () => { cs.img = cs.img.map(row => [0, ...row.slice(0, 4)]); drawConv(); });
$("#cReset").addEventListener("click", () => { cs.img = strokeImg(2); cs.sel = [0, 1]; drawConv(); });
$("#cClear").addEventListener("click", () => { cs.img = strokeImg(-1); drawConv(); });

/* =====================================================================
   Part 2: a digit through the whole network
   ===================================================================== */
const ns = { set: "shifted", i: 0 };
function drawNet() {
  const cv = $("#netC"), w = cv.clientWidth || cv.parentElement.clientWidth, narrow = w < 560;
  const imgs = ns.set === "shifted" ? POOL : CLEAN, ys = ns.set === "shifted" ? POOL_Y : CLEAN_Y, img = imgs[ns.i], y = ys[ns.i];
  const o = ns.set === "shifted" ? OUT[ns.i] : forwardWith(NET, img), pred = argmax(o.p);
  const perRow = narrow ? 8 : 16, gapPx = 4, mapW = (w - (perRow - 1) * gapPx) / perRow, c8 = mapW / 8, c4 = mapW / 4;
  const ci = narrow ? 14 : 18, rowsC1 = Math.ceil(16 / perRow), rowsP = rowsC1, rowsC2 = Math.ceil(32 / perRow), lab = 20;
  const yIn = lab, yC1 = yIn + 8 * ci + 32 + lab, yP = yC1 + rowsC1 * (mapW + gapPx) + 10 + lab, yC2 = yP + rowsP * (mapW + gapPx) + 10 + lab, yF = yC2 + rowsC2 * (mapW + gapPx) + 10 + lab, hF = 56, h = yF + hF + 22;
  const { ctx } = fit(cv, w / h);
  ctx.font = "11px " + css("--font-mono"); ctx.fillStyle = css("--muted"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  const label = (t, yy, x = 0) => { ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText(t, x, yy - 6); };
  /* input */
  label(`input · ${ns.set} digit ${ns.i + 1} of ${imgs.length}, labelled ${y}`, yIn);
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { ctx.fillStyle = alpha("--ink", 0.06 + 0.86 * img[i * 8 + j]); ctx.fillRect(j * ci, yIn + i * ci, ci, ci); }
  ctx.strokeStyle = alpha("--ink", 0.3); ctx.strokeRect(0.5, yIn + 0.5, 8 * ci - 1, 8 * ci - 1);
  /* head probabilities beside it */
  const px0 = 8 * ci + 24, pw = w - px0, bw = Math.min(28, (pw - 9 * 4) / 10), ph = 8 * ci - 14;
  label(`source head · p(class) · says ${pred}${pred === y ? ", right" : ", wrong"}`, yIn + 8 * ci + 26, px0); ctx.textAlign = "center";
  for (let c = 0; c < 10; c++) {
    const x = px0 + c * (bw + 4), hh = ph * o.p[c];
    ctx.fillStyle = c === y ? css("--post") : c === pred ? css("--warm") : alpha("--ink", 0.3); ctx.fillRect(x, yIn + ph - hh, bw, hh);
    ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.fillText(String(c), x + bw / 2, yIn + ph + 12);
    if (o.p[c] >= 0.1) ctx.fillText(fmt(o.p[c], 2), x + bw / 2, yIn + ph - hh - 4);
  }
  /* feature maps */
  const maps = (data, HW, C, cell, yy, perRow, mx) => {
    const side = Math.sqrt(HW);
    for (let c = 0; c < C; c++) {
      const col = c % perRow, row = Math.floor(c / perRow), x0 = col * (mapW + gapPx), y0 = yy + row * (mapW + gapPx);
      for (let p = 0; p < HW; p++) { const v = data[p * C + c] / mx; ctx.fillStyle = v > 0 ? alpha("--post", 0.1 + 0.9 * Math.min(1, v)) : alpha("--ink", 0.06); ctx.fillRect(x0 + (p % side) * cell, y0 + Math.floor(p / side) * cell, cell + 0.3, cell + 0.3); }
      ctx.strokeStyle = alpha("--ink", 0.2); ctx.strokeRect(x0 + 0.5, y0 + 0.5, mapW - 1, mapW - 1);
    }
  };
  const m1 = Math.max(1e-9, ...o.a1), m2 = Math.max(1e-9, ...o.a2);
  label(`conv1 · 16 feature maps, 8×8 (brightness relative to the layer's maximum, ${fmt(m1, 2)})`, yC1); maps(o.a1, 64, 16, c8, yC1, perRow, m1);
  label("pool1 · the same 16 maps after 2×2 max pooling, 4×4", yP); maps(o.p1, 16, 16, c4, yP, perRow, m1);
  label(`conv2 · 32 feature maps, 4×4 (maximum ${fmt(m2, 2)})`, yC2); maps(o.a2, 16, 32, c4, yC2, perRow, m2);
  /* features */
  const fmax = Math.max(1e-9, ...o.f), nz = o.f.filter(v => v > 0).length, fb = (w - 31 * 2) / 32;
  label(`feat · the 32 features after global average pooling and the dense ReLU layer, ${nz} of them non-zero`, yF);
  for (let j = 0; j < 32; j++) { const hh = hF * o.f[j] / fmax; ctx.fillStyle = o.f[j] > 0 ? css("--c5") : alpha("--ink", 0.25); ctx.fillRect(j * (fb + 2), yF + hF - Math.max(1.5, hh), fb, Math.max(1.5, hh)); }
  ctx.strokeStyle = alpha("--ink", 0.3); seg(ctx, 0, yF + hF + 0.5, w, yF + hF + 0.5);
  ctx.fillStyle = css("--muted"); ctx.textAlign = "left"; ctx.fillText("1", 0, yF + hF + 14); ctx.textAlign = "right"; ctx.fillText("32", w, yF + hF + 14);
  const top = argmax(o.f), g = o.g, gmax = Math.max(...g), c1on = (() => { let n = 0; for (let c = 0; c < 16; c++) { let any = false; for (let p = 0; p < 64; p++) if (o.a1[p * 16 + c] > 0) { any = true; break; } if (any) n++; } return n; })();
  let text = `${ns.set === "shifted" ? "A shifted, rotated" : "A clean"} ${y}. ${c1on} of the 16 conv1 maps respond somewhere; after pooling, conv2 and global average pooling the digit is 32 channel averages (the largest ${fmt(gmax, 3)}), and the dense layer turns those into 32 features, ${nz} non-zero, the strongest feature ${top + 1} at ${fmt(o.f[top], 2)}. The source head reads ${pred} with probability ${fmt(o.p[pred], 2)}` + (pred === y ? ", correctly." : `; the true class ${y} gets ${fmt(o.p[y], 2)}.`) + (ns.set === "shifted" ? " In the few-shot race, only these 32 features are shown to the new classifier; the head is thrown away." : " The clean digits are the source task, the one the network was trained on.");
  caption($("#netCap"), text);
}
$$("#netW .seg .btn").forEach(bt => bt.addEventListener("click", () => { ns.set = bt.dataset.set; ns.i = 0; $$("#netW .seg .btn").forEach(b => b.setAttribute("aria-pressed", String(b === bt))); drawNet(); }));
$("#dPrev").addEventListener("click", () => { const n = ns.set === "shifted" ? POOL.length : CLEAN.length; ns.i = (ns.i - 1 + n) % n; drawNet(); });
$("#dNext").addEventListener("click", () => { const n = ns.set === "shifted" ? POOL.length : CLEAN.length; ns.i = (ns.i + 1) % n; drawNet(); });

function drawAll() { drawHero(); drawConv(); drawNet(); }
drawAll();
onRedraw(drawAll);
setTimeout(computeNext, 0);
})();
