/* A Tour of Machine Learning, chapter 10, Transformers and self-attention: the live demos.
   Scaled dot-product attention, multi-head attention and the causal mask are the notebook's
   NumPy functions, line for line, on its 8-token sentence and its random projections
   (ch10-data.js); the default settings reproduce the notebook's numbers exactly.
   Needs ../kit/kit.js, ../kit/stats.js and ch10-data.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, alpha, onRedraw, narrowOf, seg, fit, histogram } = window.Kit;
const { makeRng } = window.Stats;
const D = window.CH10;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
function softmax(row) { const m = Math.max(...row), e = row.map(v => Math.exp(v - m)), Z = e.reduce((a, b) => a + b, 0); return e.map(v => v / Z); }
const matmulT = (A, B) => A.map(a => B.map(b => a.reduce((s, v, k) => s + v * b[k], 0)));   // A B^T
const matmul = (A, B) => A.map(a => B[0].map((_, j) => a.reduce((s, v, k) => s + v * B[k][j], 0)));
/* softmax(Q K^T / sqrt(d_k)) V, with an optional mask (mask[i][j] true = allowed). Returns the outputs and the weights. */
function attention(Q, K, V, mask) {
  const dk = Q[0].length, S = matmulT(Q, K).map(r => r.map(v => v / Math.sqrt(dk)));
  const W = S.map((r, i) => softmax(mask ? r.map((v, j) => mask[i][j] ? v : -Infinity) : r));
  return { out: matmul(W, V), weights: W, scores: S };
}
/* The notebook's multi_head_attention: one d_model x d_model projection each, columns cut into n_heads blocks. */
function multiHead(X, H, Wq, Wk, Wv, Wo) {
  const L = X.length, dm = X[0].length, dk = dm / H, Q = matmul(X, Wq), K = matmul(X, Wk), V = matmul(X, Wv);
  const split = (M, h) => M.map(r => r.slice(h * dk, (h + 1) * dk));
  const heads = Array.from({ length: H }, (_, h) => attention(split(Q, h), split(K, h), split(V, h)));
  const concat = Array.from({ length: L }, (_, i) => heads.flatMap(hd => hd.out[i]));
  return { out: matmul(concat, Wo), weights: heads.map(hd => hd.weights) };
}
/* NET-END */

const TOKENS = ["The", "animal", "didn't", "cross", "because", "it", "was", "tired"], N = 8, IT = 5, ANIMAL = 1;
const eye = n => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
function embeddings(alpha, scale) { const E = eye(N); E[IT] = E[IT].map((v, j) => ((1 - alpha) * v + alpha * (j === ANIMAL ? 1 : 0))); return E.map(r => r.map(v => v * scale)); }
const causal = n => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => j <= i));
const short = t => t.length > 5 ? t.slice(0, 4) + "." : t;

/* heat map of an n x n weight matrix with token labels; returns nothing */
function heatmap(ctx, W, x0, y0, cell, labels, vmax, narrow, token) {
  const n = W.length;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    ctx.fillStyle = alpha(token || "--post", 0.06 + 0.9 * Math.min(1, W[i][j] / vmax)); ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.3, cell + 0.3);
    if (cell >= 26) { ctx.fillStyle = W[i][j] / vmax > 0.55 ? css("--bg") : css("--ink"); ctx.font = `${Math.max(8, Math.round(cell * 0.3))}px ${css("--font-mono")}`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(W[i][j] < 0.0005 ? "0" : fmt(W[i][j], 2).replace(/^0/, ""), x0 + j * cell + cell / 2, y0 + i * cell + cell / 2); }
  }
  ctx.strokeStyle = alpha("--ink", 0.25); ctx.strokeRect(x0 + 0.5, y0 + 0.5, n * cell - 1, n * cell - 1);
  if (labels) {
    ctx.fillStyle = css("--muted"); ctx.font = `${narrow ? 9 : 11}px ${css("--font-mono")}`; ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) { ctx.textAlign = "right"; ctx.fillText(narrow ? short(labels[i]) : labels[i], x0 - 4, y0 + i * cell + cell / 2); }
    ctx.textBaseline = "alphabetic";
    for (let j = 0; j < n; j++) { ctx.save(); ctx.translate(x0 + j * cell + cell / 2, y0 + n * cell + 4); ctx.rotate(Math.PI / 4); ctx.textAlign = "left"; ctx.fillText(narrow ? short(labels[j]) : labels[j], 0, 4); ctx.restore(); }
  }
}

/* =====================================================================
   Hero: the sentence, the blend, the scale and the mask
   ===================================================================== */
const hs = { mask: false };
function drawHero() {
  const a = +$("#hAlpha").value, sc = +$("#hScale").value, E = embeddings(a, sc), res = attention(E, E, E, hs.mask ? causal(N) : null), W = res.weights;
  const cv = $("#attC"), narrow = narrowOf(cv), w = cv.clientWidth || cv.parentElement.clientWidth, labW = narrow ? 44 : 64, cell = Math.min(narrow ? 30 : 42, (w - labW - 8) / N), h = labW + N * cell + 10;
  const { ctx } = fit(cv, w / h);
  heatmap(ctx, W, labW, 6, cell, TOKENS, 1, narrow);
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  const row = W[IT], others = row.filter((_, j) => j !== IT && j !== ANIMAL), top = [...row.keys()].sort((p, q) => row[q] - row[p]).slice(0, 3);
  $("#hAlphaV").textContent = fmt(a, 2); $("#hScaleV").textContent = "×" + fmt(sc, 1);
  $("#rAnimal").textContent = fmt(row[ANIMAL], 3); $("#rIt").textContent = fmt(row[IT], 3); $("#rOther").textContent = fmt(Math.max(...others), 3);
  const nb = a === 0.6 && sc === 1 && !hs.mask;
  let text = `Row "it": ${top.map(j => `"${TOKENS[j]}" ${fmt(row[j], 3)}`).join(", ")}${nb ? ", the notebook's numbers" : ""}. `;
  if (sc === 1 && a === 0.6 && !hs.mask) text += `"animal" leads, but by a hair: a uniform row would give ${fmt(1 / N, 3)} to every token, and six of the eight tokens get ${fmt(others[0], 3)}. The scores behind the row are tiny, ${fmt(0.6 / Math.sqrt(8), 3)} for "animal" and ${fmt(0.52 / Math.sqrt(8), 3)} for "it" itself against 0 for the rest, because the embeddings have unit length and the dot products are divided by [[\\sqrt{8}]]. The coreference is in the arithmetic, faintly. `;
  else if (a === 0) text += `With no blend, "it" is a plain one-hot vector: it matches only itself (score ${fmt(sc * sc / Math.sqrt(8), 3)}) and spreads the rest evenly. `;
  else if (sc >= 4) text += `Scaling the embeddings by ${fmt(sc, 1)} multiplies every score by ${fmt(sc * sc, 1)}, and the softmax sharpens: "animal" now takes ${fmt(row[ANIMAL], 2)} of the row. The relationship did not change; its contrast did. That is what the [[\\sqrt{d_k}]] division is there to control in the other direction. `;
  else text += `Blend ${fmt(a, 2)} and scale ${fmt(sc, 1)}: "animal" gets ${fmt(row[ANIMAL], 3)}, "it" itself ${fmt(row[IT], 3)}. `;
  if (hs.mask) text += `With the causal mask, "it" (position 6) cannot see "was" or "tired"; its row is renormalised over the first six tokens, and the two columns on the right are exactly zero.`;
  caption($("#heroStatus"), text);
}
$("#hAlpha").addEventListener("input", drawHero); $("#hScale").addEventListener("input", drawHero);
$("#hMask").addEventListener("click", () => { hs.mask = !hs.mask; $("#hMask").setAttribute("aria-pressed", String(hs.mask)); $("#hMask").textContent = hs.mask ? "remove the causal mask" : "add the causal mask"; drawHero(); });
$("#hReset").addEventListener("click", () => { $("#hAlpha").value = 0.6; $("#hScale").value = 1; hs.mask = false; $("#hMask").setAttribute("aria-pressed", "false"); $("#hMask").textContent = "add the causal mask"; drawHero(); });

/* =====================================================================
   Part 3: the three-token example, by hand and by slider
   ===================================================================== */
const hand = { mask: false };
function drawHand() {
  const a = +$("#x3a").value, b = +$("#x3b").value, X = [[1, 0], [0, 1], [a, b]], res = attention(X, X, X, hand.mask ? causal(3) : null);
  $("#x3aV").textContent = fmt(a, 1); $("#x3bV").textContent = fmt(b, 1);
  const f = v => (v === -Infinity ? "−∞" : fmt(v, 3)), rows = res.weights.map((r, i) => `<tr><td>token ${i + 1}</td><td>${res.scores[i].map(f).join(", ")}</td><td>${res.scores[i].map((v, j) => hand.mask && j > i ? "0" : fmt(Math.exp(v - Math.max(...res.scores[i].filter((_, jj) => !hand.mask || jj <= i))), 3)).join(", ")}</td><td>${r.map(v => fmt(v, 3)).join(", ")}</td><td>(${res.out[i].map(v => fmt(v, 3)).join(", ")})</td></tr>`).join("");
  $("#handOut").innerHTML = `<thead><tr><th>query</th><th>scaled scores</th><th class="words">exponentials (row max subtracted)</th><th>weights</th><th>output</th></tr></thead><tbody>${rows}</tbody>`;
  const nb = a === 1 && b === 1;
  caption($("#handCap"), `[[\\mathbf x_3 = (${fmt(a, 1)}, ${fmt(b, 1)})]]${nb ? ", the notebook's example" : ""}. ` + (nb && !hand.mask ? "Token 1 gives the least weight to token 2, whose vector is perpendicular to its own (score 0), and its output (0.802, 0.599) has borrowed some of the second direction from tokens 2 and 3." : nb && hand.mask ? "With the mask, token 1 may only look at itself (weights 1, 0, 0), token 2 at tokens 1 and 2 (0.330 and 0.670), and token 3's row is unchanged because it already sees only itself and the past." : a === 0 && b === 0 ? "A zero vector scores 0 against everything, so token 3 attends uniformly, and nobody is drawn to it: the row for token 3 is the uniform (1/3, 1/3, 1/3) and its output the plain average of the three vectors." : `Token 3's row now puts ${fmt(res.weights[2][2], 3)} on itself: the longer its vector, the larger its self-score [[\\lVert \\mathbf x_3 \\rVert^2 / \\sqrt 2 = ${fmt((a * a + b * b) / Math.SQRT2, 3)}]], and the more the softmax concentrates there.`) + (hand.mask ? " Masked entries are −∞ before the softmax and exactly 0 after it." : ""));
}
$("#x3a").addEventListener("input", drawHand); $("#x3b").addEventListener("input", drawHand);
$("#handMask").addEventListener("click", () => { hand.mask = !hand.mask; $("#handMask").setAttribute("aria-pressed", String(hand.mask)); $("#handMask").textContent = hand.mask ? "remove the causal mask" : "add the causal mask"; drawHand(); });

/* =====================================================================
   Part 2: why divide by sqrt(d_k)
   ===================================================================== */
const DKS = [2, 4, 8, 16, 32, 64, 128, 256];
function drawScale() {
  const dk = DKS[+$("#scDk").value], gap = +$("#spGap").value, rng = makeRng(10), M = 4000, raw = new Float64Array(M), scaled = new Float64Array(M);
  for (let m = 0; m < M; m++) { let s = 0; for (let k = 0; k < dk; k++) s += rng.normal() * rng.normal(); raw[m] = s; scaled[m] = s / Math.sqrt(dk); }
  const sd = a => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length), sRaw = sd(raw), sSc = sd(scaled);
  const cv = $("#scC"), narrow = narrowOf(cv), lim = 20, fr = frame(cv, narrow ? 1.6 : 2.8, { x: [-lim, lim], y: [0, 0.45], xlabel: "score q · k", yticks: false, xticks: [-20, -10, 0, 10, 20] });
  histogram(fr, scaled, -lim, lim, 80, 1 / (M * (2 * lim / 80)), alpha("--post", 0.6)); histogram(fr, raw, -lim, lim, 80, 1 / (M * (2 * lim / 80)), alpha("--warm", 0.45));
  const { ctx, X, Y } = fr; ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText("blue: divided by √d_k · orange: raw", fr.pad.l + 4, fr.pad.t + 12);
  $("#scDkV").textContent = dk; $("#spGapV").textContent = fmt(gap, 1);
  const sm = softmax([0, gap]);
  caption($("#scCap"), `With [[d_k = ${dk}]], raw dot products of unit-variance queries and keys have standard deviation ${fmt(sRaw, 2)} (theory [[\\sqrt{${dk}} = ${fmt(Math.sqrt(dk), 2)}]]); divided by [[\\sqrt{d_k}]] they have ${fmt(sSc, 2)}, whatever the dimension. Why it matters: two scores ${fmt(0, 0)} and ${fmt(gap, 1)} become weights ${fmt(sm[0], gap >= 7 ? 4 : 3)} and ${fmt(sm[1], gap >= 7 ? 4 : 3)}` + (gap >= 6 ? ": the softmax has saturated, and its derivative, [[a_{ij}(\\delta_{jl} - a_{il})]], is near zero everywhere, so learning stalls." : gap <= 1 ? ": a gentle preference that gradients can still move." : "."));
}
$("#scDk").addEventListener("input", drawScale); $("#spGap").addEventListener("input", drawScale);

/* =====================================================================
   Part 4: multi-head attention with the notebook's projections
   ===================================================================== */
const mh = { H: 2 };
function drawMh() {
  const s = +$("#mhScale").value, E = embeddings(0.6, 1), scale = M => M.map(r => r.map(v => v * s)), res = multiHead(E, mh.H, scale(D.Wq), scale(D.Wk), scale(D.Wv), D.Wo);
  const cv = $("#mhC"), narrow = narrowOf(cv), w = cv.clientWidth || cv.parentElement.clientWidth, perRow = mh.H <= 2 ? mh.H : narrow ? 2 : 4, labW = narrow ? 40 : 60, gap = 12;
  const cell = Math.min(narrow ? 26 : 34, (w - perRow * (labW + gap)) / (perRow * N)), rows = Math.ceil(mh.H / perRow), blockH = labW + N * cell + 24, h = rows * blockH;
  const { ctx } = fit(cv, w / h);
  res.weights.forEach((W, hh) => {
    const x0 = (hh % perRow) * ((w - 0) / perRow) + labW, y0 = Math.floor(hh / perRow) * blockH + 18, vmax = Math.max(...W.flat());
    ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillText(`head ${hh} · d_k = ${N / mh.H} · max weight ${fmt(vmax, 2)}`, x0 - labW + 2, y0 - 6);
    heatmap(ctx, W, x0, y0, cell, TOKENS, vmax, narrow, "--warm");
  });
  const all = res.weights.flat(2), rowMax = res.weights.flat().map(r => Math.max(...r)), meanMax = rowMax.reduce((a, b) => a + b, 0) / rowMax.length, ent = res.weights.flat().map(r => -r.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0)).reduce((a, b) => a + b, 0) / rowMax.length;
  $("#mhScaleV").textContent = "×" + fmt(s, 1); $$("#mhW [data-h]").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.h === mh.H)));
  const itTop = res.weights.map(W => { const r = W[IT]; const j = r.indexOf(Math.max(...r)); return `"${TOKENS[j]}" (${fmt(r[j], 2)})`; });
  caption($("#mhCap"), `${mh.H} head${mh.H > 1 ? "s" : ""}, each with [[d_k = ${N / mh.H}]], projections scaled by ${fmt(s, 1)}${s === 1 && mh.H === 2 ? ", the notebook's setting" : ""}. Each map is coloured relative to its own maximum. Averaged over all rows, the largest weight is ${fmt(meanMax, 3)} and the entropy ${fmt(ent, 2)} nats against ${fmt(Math.log(N), 2)} for a uniform row` + (s <= 1.5 ? `: with projections of size 0.3 on unit embeddings the scores are a few hundredths, and every head is flat to within ${fmt(Math.max(...all) - Math.min(...all), 2)}. The heat maps only look patterned because each is stretched to its own maximum. ` : s >= 6 ? `. Row "it" now peaks at ${itTop.join(", ")}: strong, sharp, and meaningless, since the projections are random; the heads differ because they read different columns of the same random matrices. ` : `. Larger projections give larger scores and sharper rows. `) + (mh.H === 8 ? "With [[d_k = 1]] each head's score is a product of two numbers: one dimension cannot express much of a relationship, which is the trade-off exercise 2 is about." : ""));
}
$$("#mhW [data-h]").forEach(bt => bt.addEventListener("click", () => { mh.H = +bt.dataset.h; drawMh(); }));
$("#mhScale").addEventListener("input", drawMh);

/* =====================================================================
   Part 4: sinusoidal positional encodings
   ===================================================================== */
function drawPe() {
  const d = +$("#peD").value, L = 32, P = Array.from({ length: L }, (_, i) => Array.from({ length: d }, (_, k) => { const m = Math.floor(k / 2), ang = i / 10000 ** (2 * m / d); return k % 2 === 0 ? Math.sin(ang) : Math.cos(ang); }));
  const cv = $("#peC"), w = cv.clientWidth || cv.parentElement.clientWidth, cellW = (w - 40) / d, cellH = Math.min(9, cellW), h = L * cellH + 30;
  const { ctx } = fit(cv, w / h);
  for (let i = 0; i < L; i++) for (let k = 0; k < d; k++) { const v = P[i][k]; ctx.fillStyle = v >= 0 ? alpha("--post", 0.08 + 0.9 * v) : alpha("--warm", 0.08 - 0.9 * v); ctx.fillRect(40 + k * cellW, i * cellH, cellW + 0.3, cellH + 0.3); }
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; for (const i of [0, 8, 16, 24, 31]) ctx.fillText(String(i), 34, i * cellH + cellH / 2);
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillText("dimension 0 … " + (d - 1) + "  (rows: positions 0 … 31; blue +1, orange −1)", 40, L * cellH + 16);
  $("#peDV").textContent = d;
  const dot = (p, q) => p.reduce((s, v, k) => s + v * q[k], 0);
  caption($("#peCap"), `[[d_{\\text{model}} = ${d}]]: the left-hand dimensions oscillate fast (period [[2\\pi]] positions for the first pair), the right-hand ones slowly (period [[2\\pi \\cdot 10000^{${(d - 2) / d}} \\approx ${fmt(2 * Math.PI * 10000 ** ((d - 2) / d), 0)}]] positions for the last). Every row is distinct, and the dot product between rows depends mostly on their distance: positions 3 and 4 give ${fmt(dot(P[3], P[4]), 2)}, positions 3 and 12 give ${fmt(dot(P[3], P[12]), 2)}, out of a maximum of ${fmt(d / 2, 0)}. Added to the embeddings, these vectors let attention tell near from far.`);
}
$("#peD").addEventListener("input", drawPe);

function drawAll() { drawHero(); drawHand(); drawScale(); drawMh(); drawPe(); }
drawAll();
onRedraw(drawAll);
})();
