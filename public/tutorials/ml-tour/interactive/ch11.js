/* A Tour of Machine Learning, chapter 11, Federated learning: the live demos.
   The data, the split, the local logistic regression trained by full-batch gradient descent and
   the weighted averaging are the notebook's, line for line (ch11-data.js); the default setting,
   three IID clients, 30 local steps at rate 0.2, 15 rounds, reproduces its history exactly.
   Needs ../kit/kit.js and ch11-data.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp, fit } = window.Kit;
const D = window.CH11;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const sigmoid = z => 1 / (1 + Math.exp(-z));
/* E full-batch gradient steps of logistic regression on the rows `idx` of X, from (w, b). */
function localTrain(X, y, idx, w0, b0, epochs, lr) {
  const d = w0.length, n = idx.length; let w = w0.slice(), b = b0;
  for (let e = 0; e < epochs; e++) {
    const gw = new Float64Array(d); let gb = 0;
    for (const i of idx) { const x = X[i]; let z = b; for (let j = 0; j < d; j++) z += x[j] * w[j]; const r = sigmoid(z) - y[i]; for (let j = 0; j < d; j++) gw[j] += r * x[j]; gb += r; }
    for (let j = 0; j < d; j++) w[j] -= lr * gw[j] / n; b -= lr * gb / n;
  }
  return { w, b };
}
const accuracy = (w, b, X, y) => { let c = 0; for (let i = 0; i < X.length; i++) { let z = b; for (let j = 0; j < w.length; j++) z += X[i][j] * w[j]; if ((z > 0) === (y[i] === 1)) c++; } return c / X.length; };
/* FedAvg: broadcast, local training, size-weighted averaging; returns the history of test accuracies and the final model. */
function fedavg(X, y, clients, rounds, epochs, lr, Xte, yte) {
  const d = X[0].length; let gw = new Array(d).fill(0), gb = 0; const hist = [], models = [];
  const n = clients.reduce((s, c) => s + c.length, 0);
  for (let r = 0; r < rounds; r++) {
    const locals = clients.map(c => localTrain(X, y, c, gw, gb, epochs, lr));
    gw = new Array(d).fill(0); gb = 0;
    locals.forEach((m, k) => { const p = clients[k].length / n; for (let j = 0; j < d; j++) gw[j] += p * m.w[j]; gb += p * m.b; });
    hist.push(accuracy(gw, gb, Xte, yte)); models.push({ w: gw.slice(), b: gb });
  }
  return { hist, w: gw, b: gb, models };
}
/* NET-END */

const XTR = D.Xtr, YTR = D.ytr, XTE = D.Xte, YTE = D.yte, NTE = XTE.length, CW = D.central_w, CB = D.central_b, CACC = D.central_acc;
const majority = Math.max(YTE.reduce((s, v) => s + v, 0), YTE.length - YTE.reduce((s, v) => s + v, 0)) / NTE;
const splitInto = (order, K) => { const out = [], base = Math.floor(order.length / K), extra = order.length % K; let p = 0; for (let k = 0; k < K; k++) { const len = base + (k < extra ? 1 : 0); out.push(order.slice(p, p + len)); p += len; } return out; };
const clientsFor = (K, byAge) => splitInto(byAge ? D.age_order : D.shuffled, K);
const maxDiff = w => Math.max(...w.map((v, j) => Math.abs(v - CW[j])));

/* =====================================================================
   Hero: FedAvg, live
   ===================================================================== */
const EP = [1, 5, 30, 100];
const hs = { K: 3, byAge: false };
let cache = {};
function runHero() {
  const E = EP[+$("#hE").value], rounds = +$("#hR").value, key = `${hs.K}-${hs.byAge}-${E}-${rounds}`;
  if (!cache[key]) cache[key] = fedavg(XTR, YTR, clientsFor(hs.K, hs.byAge), rounds, E, 0.2, XTE, YTE);
  return { E, rounds, res: cache[key] };
}
function drawHero() {
  const { E, rounds, res } = runHero(), cv = $("#fedC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.3 : 2.1, { x: [0, rounds + 0.5], y: [0.6, 0.78], xlabel: "federated round", yticks: [0.6, 0.65, 0.7, 0.75], xticks: rounds <= 15 ? [1, 5, 10, 15].filter(v => v <= rounds) : [1, 10, 20, 30].filter(v => v <= rounds) });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--warm", 0.1); ctx.fillRect(X(0), Y(CACC + 1 / NTE), X(rounds + 0.5) - X(0), Y(CACC - 1 / NTE) - Y(CACC + 1 / NTE));
  ctx.strokeStyle = css("--warm"); ctx.setLineDash([6, 4]); seg(ctx, X(0), Y(CACC), X(rounds + 0.5), Y(CACC)); ctx.strokeStyle = alpha("--ink", 0.4); seg(ctx, X(0), Y(majority), X(rounds + 0.5), Y(majority)); ctx.setLineDash([]);
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText("centralised " + fmt(CACC, 3) + " (band: ±1 patient)", X(0.3), Y(CACC) - 6); ctx.fillText("majority class " + fmt(majority, 3), X(0.3), Y(majority) - 6);
  const xs = res.hist.map((_, i) => i + 1);
  plotLine(fr, xs, res.hist, { color: css("--post"), width: 2.2 });
  ctx.fillStyle = css("--post"); for (let i = 0; i < xs.length; i++) { ctx.beginPath(); ctx.arc(X(xs[i]), Y(res.hist[i]), 3, 0, 2 * Math.PI); ctx.fill(); }
  $("#hEV").textContent = E; $("#hRV").textContent = rounds;
  const final = res.hist[rounds - 1], wd = maxDiff(res.w), agree = XTE.filter((x, i) => { let z = res.b; for (let j = 0; j < 8; j++) z += x[j] * res.w[j]; let zc = CB; for (let j = 0; j < 8; j++) zc += x[j] * CW[j]; return (z > 0) === (zc > 0); }).length;
  $("#rFed").textContent = fmt(final, 3); $("#rCen").textContent = fmt(CACC, 3); $("#rDiff").textContent = fmt(wd, 3); $("#rAgree").textContent = `${agree} of ${NTE}`;
  const nb = hs.K === 3 && !hs.byAge && E === 30 && rounds === 15;
  let text = `${hs.K} clients, ${hs.byAge ? "split by age" : "split at random"}, ${E} local step${E > 1 ? "s" : ""} per round, ${rounds} rounds${nb ? ": the notebook's setting" : ""}. Test accuracy ${fmt(final, 3)} after the last round against ${fmt(CACC, 3)} for the centralised model; the largest weight difference between the two models is ${fmt(wd, 3)}, and they agree on ${agree} of the ${NTE} test patients. `;
  text += `The test set has ${NTE} patients, so one patient is ${fmt(100 / NTE, 2)} points of accuracy: the wobble between rounds is one to three patients changing side, and the two models' "exact match" at 0.714 is a tie at 110 correct. `;
  if (E === 1) text += `With one local step, each round is exactly one gradient step on the pooled data, so the curve is plain gradient descent, slowed to one step per round; ${rounds} rounds is ${rounds} steps, and the weights are still far from the optimum (difference ${fmt(wd, 2)}) even where the accuracy already looks fine.`;
  else if (hs.byAge) text += `Sorted by age the clients hold patients aged 21 to 25, 25 to 37 and 37 to 81, with diabetes rates 0.19, 0.36 and 0.51: as non-IID as this table gets. The weights drift further from the centralised ones (${fmt(wd, 3)} against 0.031 for the random split) and the accuracy is no worse; a convex model on 614 patients forgives a lot of drift.`;
  else if (hs.K === 10) text += `Ten clients of about 61 patients instead of three of 205, and the curve is the same: with full-batch steps and IID data, smaller shards give the same gradient direction with a little more noise, and the average cancels the noise.`;
  else text += `The federated model is not the centralised model; it is the model that ${rounds * E} steps of gradient descent would reach, which here predicts the same 154 test patients.`;
  caption($("#heroStatus"), text);
}
$("#hE").addEventListener("input", drawHero); $("#hR").addEventListener("input", drawHero);
$$("#demo [data-k]").forEach(bt => bt.addEventListener("click", () => { hs.K = +bt.dataset.k; $$("#demo [data-k]").forEach(b => b.setAttribute("aria-pressed", String(b === bt))); drawHero(); }));
$$("#demo [data-split]").forEach(bt => bt.addEventListener("click", () => { hs.byAge = bt.dataset.split === "age"; $$("#demo [data-split]").forEach(b => b.setAttribute("aria-pressed", String(b === bt))); drawHero(); }));

/* =====================================================================
   Part 2: the one-parameter drift example
   ===================================================================== */
function drawDrift() {
  const E = +$("#dE").value, lr = 0.1, rounds = 30, F1 = w => 0.5 * w * w, F2 = w => 1.5 * (w - 4) ** 2, g1 = w => w, g2 = w => 3 * (w - 4);
  let w = 0; const path = [0]; let a = 0, c = 0;
  for (let r = 0; r < rounds; r++) { a = w; c = w; for (let s = 0; s < E; s++) { a -= lr * g1(a); c -= lr * g2(c); } w = 0.5 * (a + c); path.push(w); }
  const cv = $("#driftC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.4, { x: [-1, 5], y: [0, 10], xlabel: "w", yticks: [0, 5, 10], xticks: [-1, 0, 1, 2, 3, 4, 5] }), { ctx, X, Y } = fr;
  const ws = Array.from({ length: 121 }, (_, i) => -1 + i * 0.05);
  plotLine(fr, ws, ws.map(F1), { color: css("--post"), width: 1.6 }); plotLine(fr, ws, ws.map(F2), { color: css("--warm"), width: 1.6 }); plotLine(fr, ws, ws.map(v => 0.5 * F1(v) + 0.5 * F2(v)), { color: css("--ink"), width: 2.2, dash: [5, 3] });
  const mark = (x, token, label, dy) => { ctx.fillStyle = css(token); ctx.beginPath(); ctx.arc(X(x), Y(0), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; ctx.fillText(label, X(x), Y(0) - 8 - dy); };
  ctx.strokeStyle = alpha("--ink", 0.3); seg(ctx, X(3), Y(0), X(3), Y(10));
  mark(a, "--post", "client 1 → " + fmt(a, 2), 0); mark(c, "--warm", "client 2 → " + fmt(c, 2), 0); mark(w, "--ink", "average " + fmt(w, 2), 14);
  $("#dEV").textContent = E;
  caption($("#driftCap"), `[[E = ${E}]] local step${E > 1 ? "s" : ""} of size 0.1 per round, 30 rounds, from [[w = 0]]. The pooled objective (dashed) has its minimum at [[w^\\star = 3]]; the server ends at [[w = ${fmt(w, 3)}]]. ` + (E === 1 ? "One local step per round is exactly one step of gradient descent on the pooled objective: the average lands on 3 (to three decimals after 30 rounds)." : E >= 30 ? "With this many local steps each client all but reaches its own minimum, 0 and 4, before the server averages: the endpoints are averaged to 2 in every round, and the steeper client 2, which should pull harder, is forgotten. This is client drift at its worst." : `In between: each client moves part of the way to its own minimum, the average lands at ${fmt(w, 2)}, short of 3 by ${fmt(3 - w, 2)}. More local steps, fewer rounds of communication, more drift.`));
}
$("#dE").addEventListener("input", drawDrift);

/* =====================================================================
   Part 3: what a single update reveals
   ===================================================================== */
const FINAL = fedavg(XTR, YTR, clientsFor(3, false), 15, 30, 0.2, XTE, YTE);
function drawLeak() {
  const i = +$("#lkI").value, x = XTR[i], y = YTR[i]; let z = FINAL.b; for (let j = 0; j < 8; j++) z += x[j] * FINAL.w[j]; const p = sigmoid(z), r = p - y;
  const rows = D.features.map((f, j) => `<tr><td>${f}</td><td>${fmt(x[j], 3)}</td><td>${fmt(r * x[j], 4)}</td><td>${fmt(r * x[j] / r, 3)}</td></tr>`).join("");
  $("#lkT").innerHTML = `<thead><tr><th>feature</th><th>standardised value <span class="m">x_j</span></th><th>gradient entry <span class="m">(p-y)\\,x_j</span></th><th>divided by <span class="m">p-y</span></th></tr></thead><tbody>${rows}</tbody>`;
  $$("#lkT .m").forEach(el => { if (!el.dataset.rendered) { katex.render(el.textContent, el, { throwOnError: false }); el.dataset.rendered = "1"; } });
  $("#lkIV").textContent = i;
  caption($("#lkCap"), `Training patient ${i} (age ${D.age_raw[i]}, label ${y}): the final model gives [[p = ${fmt(p, 3)}]], so [[p - y = ${fmt(r, 3)}]] and the bias gradient is that same number. Dividing each weight-gradient entry by it returns the patient's eight standardised measurements exactly. A client that sent the gradient of one patient would have sent the patient. Averaged over ${clientsFor(3, false)[0].length} patients and 30 steps the update blurs them together, but it is still a function of the data, which is what secure aggregation and differential privacy are for.`);
}
$("#lkI").max = XTR.length - 1; $("#lkI").addEventListener("input", drawLeak);

/* =====================================================================
   Part 4: weighted averaging by hand
   ===================================================================== */
function drawAvg() {
  const n = [+$("#n1").value, +$("#n2").value, +$("#n3").value], W = [[1, 2], [3, 0], [1, 4]], N = n[0] + n[1] + n[2], p = n.map(v => v / N);
  const wavg = [0, 1].map(j => p.reduce((s, pk, k) => s + pk * W[k][j], 0)), plain = [0, 1].map(j => W.reduce((s, w) => s + w[j], 0) / 3);
  const rows = W.map((w, k) => `<tr><td>${k + 1}</td><td>${n[k]}</td><td>${fmt(p[k], 3)}</td><td>(${w[0].toFixed(1)}, ${w[1].toFixed(1)})</td><td>(${fmt(p[k] * w[0], 3)}, ${fmt(p[k] * w[1], 3)})</td></tr>`).join("");
  $("#avgT").innerHTML = `<thead><tr><th>client</th><th>n_k</th><th>p_k</th><th>local w_k</th><th>p_k w_k</th></tr></thead><tbody>${rows}<tr><td>sum</td><td>${N}</td><td>${fmt(p.reduce((a, b) => a + b, 0), 2)}</td><td></td><td>(${fmt(wavg[0], 3)}, ${fmt(wavg[1], 3)})</td></tr></tbody>`;
  [1, 2, 3].forEach(k => { $(`#n${k}V`).textContent = n[k - 1]; });
  caption($("#avgCap"), `Weighted average [[w^{t+1} = (${fmt(wavg[0], 2)}, ${fmt(wavg[1], 2)})]]; an unweighted average would give [[(${fmt(plain[0], 2)}, ${fmt(plain[1], 2)})]]. ` + (n[0] === 200 && n[1] === 100 && n[2] === 100 ? "The notebook's numbers: client 1 holds half the data and counts for half." : n[0] === n[1] && n[1] === n[2] ? "With equal shards the two averages coincide." : "The clients with more patients pull the average toward their own weights, exactly as their patients would in the pooled data."));
}
$$("#avgW input").forEach(el => el.addEventListener("input", drawAvg));

function drawAll() { drawHero(); drawDrift(); drawLeak(); drawAvg(); }
drawAll();
onRedraw(() => { drawHero(); drawDrift(); });
})();
