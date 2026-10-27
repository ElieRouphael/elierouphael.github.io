/* Kalman filters with JAX, chapter 02, Learning what the model leaves out: the live demos. A small
   tanh network learns the nonlinear residual of a two-state plant from inside a Kalman filter, with
   the gradient of the filter's loss taken by hand-written backpropagation through the recursion and
   Adam as in optax, so training runs in the browser; the repository's rollout, its JAX-trained
   parameters and loss trace, and the generalisation study come from ch02-data.js. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const mv = (A, x) => A.map(r => r.reduce((s, v, k) => s + v * x[k], 0));
const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const tr = A => A[0].map((_, j) => A.map(r => r[j]));
const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
const sub = (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j]));
const eye = n => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
function solve(A0, B0) {
  const n = A0.length, m = B0[0].length, A = A0.map((r, i) => [...r, ...B0[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < n; r++) { if (r === c) continue; const f = A[r][c] / A[c][c]; if (!f) continue; for (let k = c; k < n + m; k++) A[r][k] -= f * A[c][k]; }
  }
  return A.map((r, i) => r.slice(n).map(v => v / r[i]));
}
/* The true plant of the repository: x+ = A x + B u + (3 sin x0 + 2 cos x1), the residual added to both components. */
const trueResidual = x => 3 * Math.sin(x[0]) + 2 * Math.cos(x[1]);
const trueStep = (A, B, x, u) => { const r = trueResidual(x); return mv(A, x).map((v, i) => v + B[i][0] * u + r); };
/* Residual network: tanh layers, params = [[W, b], ...]. Forward keeps the activations for backprop. */
function mlpForward(params, inp) {
  const acts = [inp]; let a = inp;
  for (let l = 0; l < params.length; l++) { const [W, b] = params[l]; let z = W.map((row, i) => row.reduce((s, w, j) => s + w * a[j], b[i])); a = l < params.length - 1 ? z.map(Math.tanh) : z; acts.push(a); }
  return { out: a, acts };
}
/* Backward pass: given d(loss)/d(out), returns gradients for every parameter and d(loss)/d(inp). */
function mlpBackward(params, acts, gOut, grads) {
  let g = gOut.slice();
  for (let l = params.length - 1; l >= 0; l--) {
    const [W, b] = params[l], a = acts[l], out = acts[l + 1];
    if (l < params.length - 1) g = g.map((v, i) => v * (1 - out[i] * out[i]));   // through tanh
    const [gW, gb] = grads[l];
    for (let i = 0; i < W.length; i++) { gb[i] += g[i]; for (let j = 0; j < a.length; j++) gW[i][j] += g[i] * a[j]; }
    g = a.map((_, j) => W.reduce((s, row, i) => s + row[j] * g[i], 0));
  }
  return g;
}
const zerosLike = params => params.map(([W, b]) => [W.map(r => r.map(() => 0)), b.map(() => 0)]);
/* Kalman gains and covariances do not depend on the network (A is fixed), so they are precomputed once. */
function gainSchedule(A, C, Q, R, P0, T) {
  let P = P0; const Ks = [], Ps = [];
  for (let t = 0; t < T; t++) { const Pp = add(mm(mm(A, P), tr(A)), Q); const S = add(mm(mm(C, Pp), tr(C)), R); const K = tr(solve(S, mm(C, tr(Pp)))); const IKC = sub(eye(2), mm(K, C)); P = add(mm(mm(IKC, Pp), tr(IKC)), mm(mm(K, R), tr(K))); Ks.push(K); Ps.push(P); }
  return { Ks, Ps };
}
/* One pass of the learned filter: returns filtered states, predictions and the loss of the chosen kind
   ('filtered': C x_new against y, the repository's loss_fn; 'forecast': C x_pred against y, its forecasters). */
function filterPass(params, A, B, C, us, ys, x0, sched, kind) {
  const T = us.length, xsNew = [], xsPred = [], caches = []; let x = x0, loss = 0;
  for (let t = 0; t < T; t++) {
    const inp = [x[0], x[1], us[t]], f = mlpForward(params, inp);
    const xp = mv(A, x).map((v, i) => v + B[i][0] * us[t] + f.out[i]);
    const K = sched.Ks[t], innov = ys[t].map((v, i) => v - mv(C, xp)[i]);
    const xn = xp.map((v, i) => v + mv(K, innov)[i]);
    const yp = mv(C, kind === "forecast" ? xp : xn);
    loss += (yp[0] - ys[t][0]) ** 2 + (yp[1] - ys[t][1]) ** 2;
    xsNew.push(xn); xsPred.push(xp); caches.push(f.acts); x = xn;
  }
  return { loss: loss / (2 * T), xsNew, xsPred, caches };
}
/* Gradient of the loss with respect to the parameters by reverse-mode through the recursion. */
function filterGrad(params, A, B, C, us, ys, x0, sched, kind) {
  const T = us.length, pass = filterPass(params, A, B, C, us, ys, x0, sched, kind), grads = zerosLike(params), N = 2 * T;
  let lamNew = [0, 0];                                     // d loss / d x_new_t, accumulated from the future
  for (let t = T - 1; t >= 0; t--) {
    const xp = pass.xsPred[t], xn = pass.xsNew[t], K = sched.Ks[t], IKC = sub(eye(2), mm(K, C));
    const gNew = kind === "filtered" ? [0, 1].map(i => 2 * (mv(C, xn)[i] - ys[t][i]) / N) : [0, 0];
    const gPred = kind === "forecast" ? [0, 1].map(i => 2 * (mv(C, xp)[i] - ys[t][i]) / N) : [0, 0];
    const lNew = [0, 1].map(i => lamNew[i] + gNew[i]);
    const lPred = [0, 1].map(i => gPred[i] + mv(tr(IKC), lNew)[i]);          // x_new = (I - K C) x_pred + K y
    const gIn = mlpBackward(params, pass.caches[t], lPred, grads);            // through the residual network
    lamNew = [0, 1].map(i => mv(tr(A), lPred)[i] + gIn[i]);                   // x_pred = A x_prev + B u + f(x_prev, u)
  }
  return { loss: pass.loss, grads };
}
/* Adam as optax.adam(lr): bias-corrected first and second moments. */
function makeAdam(params, lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  const m = zerosLike(params), v = zerosLike(params); let t = 0;
  return grads => {
    t++; const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
    params.forEach(([W, b], l) => {
      const [gW, gb] = grads[l], [mW, mb] = m[l], [vW, vb] = v[l];
      for (let i = 0; i < W.length; i++) { for (let j = 0; j < W[i].length; j++) { mW[i][j] = b1 * mW[i][j] + (1 - b1) * gW[i][j]; vW[i][j] = b2 * vW[i][j] + (1 - b2) * gW[i][j] ** 2; W[i][j] -= lr * (mW[i][j] / c1) / (Math.sqrt(vW[i][j] / c2) + eps); } mb[i] = b1 * mb[i] + (1 - b1) * gb[i]; vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2; b[i] -= lr * (mb[i] / c1) / (Math.sqrt(vb[i] / c2) + eps); }
    });
  };
}
const cloneParams = p => p.map(([W, b]) => [W.map(r => r.slice()), b.slice()]);
/* Evaluation on a rollout: filtered RMSE against the true state, one-step forecast RMSE against the measurements,
   the zero-residual forecast baseline, and the RMS error of the learned residual on the visited states. */
function evaluate(params, A, B, C, roll, x0, sched) {
  const T = roll.us.length, pf = filterPass(params, A, B, C, roll.us, roll.ys, x0, sched, "forecast");
  let ef = 0, ep = 0, ez = 0, ed = 0;
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < 2; i++) { ef += (pf.xsNew[t][i] - roll.xs[t][i]) ** 2; ep += (pf.xsPred[t][i] - roll.ys[t][i]) ** 2; }
    if (t > 0) { const z = mv(A, roll.ys[t - 1]).map((v, i) => v + B[i][0] * roll.us[t]); for (let i = 0; i < 2; i++) ez += (z[i] - roll.ys[t][i]) ** 2; }
    const learned = mlpForward(params, [roll.xs[t][0], roll.xs[t][1], roll.us[t]]).out, truth = trueResidual(roll.xs[t]);
    for (let i = 0; i < 2; i++) ed += (learned[i] - truth) ** 2;
  }
  return { filtered: Math.sqrt(ef / (2 * T)), forecast: Math.sqrt(ep / (2 * T)), zero: Math.sqrt(ez / (2 * (T - 1))), dyn: Math.sqrt(ed / (2 * T)) };
}
function makeRng(seed) {
  let s0 = (seed >>> 0) || 1, s1 = 0x9e3779b9 ^ s0;
  const next = () => { let x = s0, y = s1; s0 = y; x ^= x << 23; x ^= x >>> 17; x ^= y ^ (y >>> 26); s1 = x; return ((s0 + s1) >>> 0) / 4294967296; };
  let spare = null;
  const normal = () => { if (spare !== null) { const v = spare; spare = null; return v; } let u1 = 0; while (u1 <= 1e-12) u1 = next(); const u2 = next(), r = Math.sqrt(-2 * Math.log(u1)); spare = r * Math.sin(2 * Math.PI * u2); return r * Math.cos(2 * Math.PI * u2); };
  return { next, normal };
}
/* The repository's generate_data with the page's random numbers. */
function generate(A, B, T, rng, measStd = 0.05, x0 = [0, 0]) {
  const xs = [], us = [], ys = []; let x = x0;
  for (let t = 0; t < T; t++) { const u = rng.normal(); x = trueStep(A, B, x, u); xs.push(x); us.push(u); ys.push(x.map(v => v + measStd * rng.normal())); }
  return { xs, us, ys };
}
/* NET-END */

if (typeof module !== "undefined" && module.exports) {
  module.exports = { mlpForward, filterPass, filterGrad, makeAdam, cloneParams, gainSchedule, evaluate, generate, makeRng, trueResidual, mlpBackward, zerosLike };
  return;
}

const D = window.KJ2;
const A = [[1, 0.1], [0, 1]], B = [[0], [0.1]], C = eye(2), Q = [[0.1, 0], [0, 0.1]], R = [[0.5, 0], [0, 0.5]], x0 = [0, 0], P0 = eye(2);
const SCHED = gainSchedule(A, C, Q, R, P0, 50), SCHED500 = gainSchedule(A, C, Q, R, P0, 500);

/* ============================================================ hero: training in the browser */
const hero = { kind: "filtered", steps: 1000, running: false, params: cloneParams(D.params0), losses: [], step: 0, adam: null, seed: 3 };
function resetTraining() { hero.params = cloneParams(D.params0); hero.losses = []; hero.step = 0; hero.adam = makeAdam(hero.params, 1e-3); }
function runTraining() {
  if (hero.running) return; hero.running = true;
  const tick = () => {
    const t0 = performance.now();
    while (hero.step < hero.steps && performance.now() - t0 < 40) { const { loss, grads } = filterGrad(hero.params, A, B, C, D.train.us, D.train.ys, x0, SCHED, hero.kind); hero.losses.push(loss); hero.adam(grads); hero.step++; }
    $("#hStatus").textContent = hero.step < hero.steps ? `step ${hero.step} of ${hero.steps}, loss ${fmt(hero.losses[hero.losses.length - 1], 4)}` : "";
    drawHero();
    if (hero.step < hero.steps) setTimeout(tick, 0); else hero.running = false;
  };
  setTimeout(tick, 0);
}
function drawHero() {
  const pass = filterPass(hero.params, A, B, C, D.train.us, D.train.ys, x0, SCHED, "filtered");
  const t = Array.from({ length: 50 }, (_, i) => i);
  const cl = $("#heroL"), narrow = narrowOf(cl, 600);
  const ref = hero.kind === "filtered" ? D.losses : D.forecastLosses;
  const fr = frame(cl, narrow ? 1.5 : 1.3, { x: [0, 1000], y: [-3, 0.2], xlabel: "Adam step", ylabel: "log10 loss", xticks: [0, 250, 500, 750, 1000], yticks: [-3, -2, -1, 0] });
  plotLine(fr, ref.map((_, i) => i), ref.map(v => Math.log10(v)), { color: alpha("--muted", 0.8), width: 1.4 });
  if (hero.losses.length) plotLine(fr, hero.losses.map((_, i) => i), hero.losses.map(v => Math.log10(v)), { color: css("--post"), width: 2 });
  const cx = $("#heroX"), frx = frame(cx, narrow ? 1.5 : 1.3, { x: [0, 49], y: [-8, 6], xlabel: "step", ylabel: "x₀: measured, filtered, ±1.96σ", xticks: [0, 10, 20, 30, 40], yticks: [-8, -4, 0, 4] });
  const stds = SCHED.Ps.map(P => Math.sqrt(P[0][0]));
  frx.ctx.save(); frx.ctx.beginPath(); frx.ctx.rect(frx.pad.l, frx.pad.t, frx.iw, frx.ih); frx.ctx.clip(); frx.ctx.fillStyle = alpha("--post", 0.15); frx.ctx.beginPath();
  t.forEach((tt, i) => { const v = pass.xsNew[i][0] + 1.96 * stds[i]; i ? frx.ctx.lineTo(frx.X(tt), frx.Y(v)) : frx.ctx.moveTo(frx.X(tt), frx.Y(v)); }); for (let i = 49; i >= 0; i--) frx.ctx.lineTo(frx.X(t[i]), frx.Y(pass.xsNew[i][0] - 1.96 * stds[i])); frx.ctx.closePath(); frx.ctx.fill(); frx.ctx.restore();
  frx.ctx.fillStyle = css("--ink"); D.train.ys.forEach((y, i) => { frx.ctx.beginPath(); frx.ctx.arc(frx.X(t[i]), frx.Y(y[0]), 2, 0, 2 * Math.PI); frx.ctx.fill(); });
  plotLine(frx, t, pass.xsNew.map(x => x[0]), { color: css("--post"), width: 1.8 });
  const ce = $("#heroE"), fre = frame(ce, narrow ? 1.5 : 1.3, { x: [0, 49], y: [-6, 6], xlabel: "step", ylabel: "true − learned residual", xticks: [0, 10, 20, 30, 40], yticks: [-6, -3, 0, 3, 6] });
  const err = D.train.xs.map((x, i) => { const l = mlpForward(hero.params, [x[0], x[1], D.train.us[i]]).out, tr_ = trueResidual(x); return [tr_ - l[0], tr_ - l[1]]; });
  plotLine(fre, t, err.map(e => e[0]), { color: css("--warm"), width: 1.6 }); plotLine(fre, t, err.map(e => e[1]), { color: css("--c5"), width: 1.6 });
  const evTrain = evaluate(hero.params, A, B, C, D.train, x0, SCHED), rollF = D.fresh, evFresh = evaluate(hero.params, A, B, C, rollF, x0, SCHED);
  $("#rLoss").textContent = hero.losses.length ? fmt(hero.losses[hero.losses.length - 1], 4) : "–";
  $("#rTrain").textContent = `${fmt(evTrain.filtered, 3)} / ${fmt(evTrain.dyn, 3)}`; $("#rFresh").textContent = `${fmt(evFresh.filtered, 3)} / ${fmt(evFresh.dyn, 3)}`; $("#rZero").textContent = `${fmt(evFresh.forecast, 3)} / ${fmt(evFresh.zero, 3)}`;
  const done = hero.step >= hero.steps;
  caption($("#heroStatus"), `${hero.losses.length ? `Training in the browser with the ${hero.kind === "filtered" ? "repository's loss (filtered state against the measurement)" : "one-step forecast loss (predicted state against the measurement)"}: ${hero.step} Adam steps, loss ${fmt(hero.losses[hero.losses.length - 1], 4)}` : "Press Train: 1000 Adam steps on the repository's 50-step rollout, in the browser, with the gradient taken by backpropagation through the Kalman recursion"} (grey: JAX's trace for the same loss, which reaches ${fmt(ref[ref.length - 1], 4)}; 32-bit against 64-bit arithmetic lets the two drift apart after a few hundred steps). Middle: the training rollout's first state, measured (dots) and filtered (blue) with the ±1.96σ band of the filter's covariance; right: the true residual [[3\\sin x_0 + 2\\cos x_1]] minus the learned one along the rollout, for each state component. On the rollout it was trained on, the filtered estimate is within ${fmt(evTrain.filtered, 3)} of the true state and the residual is learned to ${fmt(evTrain.dyn, 2)} RMS${done ? "" : " so far"}. On a fresh rollout with different inputs the residual error is ${fmt(evFresh.dyn, 2)} and the one-step forecast misses the measurements by ${fmt(evFresh.forecast, 2)}, worse than the ${fmt(evFresh.zero, 2)} of a model with no residual at all: the network fitted the trajectory, not the function. Part 3 says why.`);
}
$$("#heroK .btn").forEach(b => b.addEventListener("click", () => { if (hero.running) return; $$("#heroK .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); hero.kind = b.dataset.k; resetTraining(); drawHero(); }));
$("#hTrain").addEventListener("click", () => { if (hero.running) return; if (hero.step >= hero.steps) resetTraining(); runTraining(); });
$("#hReset").addEventListener("click", () => { if (hero.running) return; resetTraining(); drawHero(); });
$("#hJax").addEventListener("click", () => { if (hero.running) return; hero.params = cloneParams(hero.kind === "filtered" ? D.params : D.forecastParams); hero.losses = (hero.kind === "filtered" ? D.losses : D.forecastLosses).slice(); hero.step = 1000; drawHero(); });
resetTraining();

/* ============================================================ the residual map */
const map = { which: "trained" };
function drawMap() {
  const params = map.which === "trained" ? D.params : map.which === "page" ? hero.params : D.stable.filtered.params;
  const roll = map.which === "stable" ? D.stable.train : D.train, u = 0;
  const cv = $("#mapC"), narrow = narrowOf(cv);
  const lim = 8, n = 48;
  const fr = frame(cv, narrow ? 1.1 : 1.6, { x: [-lim, lim], y: [-lim, lim], xlabel: "x₀", ylabel: "x₁", xticks: [-8, -4, 0, 4, 8], yticks: [-8, -4, 0, 4, 8] });
  const { ctx, X, Y } = fr, cw = (X(lim) - X(-lim)) / n, ch = (Y(-lim) - Y(lim)) / n;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const x = -lim + (i + 0.5) * 2 * lim / n, y = -lim + (j + 0.5) * 2 * lim / n;
    const e = Math.abs(trueResidual([x, y]) - mlpForward(params, [x, y, u]).out[0]);
    const s = Math.min(1, e / 5); ctx.fillStyle = `rgba(${Math.round(235 * s + 42 * (1 - s))}, ${Math.round(104 * s + 120 * (1 - s))}, ${Math.round(52 * s + 214 * (1 - s))}, 0.75)`;
    ctx.fillRect(X(x) - cw / 2, Y(y) - ch / 2, cw + 0.5, ch + 0.5);
  }
  ctx.fillStyle = css("--ink"); roll.xs.forEach(x => { ctx.beginPath(); ctx.arc(X(Math.max(-lim, Math.min(lim, x[0]))), Y(Math.max(-lim, Math.min(lim, x[1]))), 2.2, 0, 2 * Math.PI); ctx.fill(); });
  if (map.which !== "stable") { ctx.strokeStyle = alpha("--ink", 0.5); ctx.setLineDash([3, 3]); D.fresh.xs.forEach((x, i) => { if (i) { ctx.beginPath(); ctx.moveTo(X(Math.max(-lim, Math.min(lim, D.fresh.xs[i - 1][0]))), Y(Math.max(-lim, Math.min(lim, D.fresh.xs[i - 1][1])))); ctx.lineTo(X(Math.max(-lim, Math.min(lim, x[0]))), Y(Math.max(-lim, Math.min(lim, x[1])))); ctx.stroke(); } }); ctx.setLineDash([]); }
  const inside = roll.xs.length, fr7 = D.fresh.xs.filter(x => Math.abs(x[0]) <= 8 && Math.abs(x[1]) <= 8).length;
  caption($("#mapCap"), `Absolute error of the learned residual over the state plane at [[u = 0]], blue for zero and orange for 5 or more, with the training states as dots${map.which !== "stable" ? " and the fresh rollout of the hero demo as a dashed path" : ""}. ${map.which === "trained" ? "JAX's trained network" : map.which === "page" ? "The page's current network" : "The network trained on the bounded plant of Part 4"} is accurate near the ${inside} states it saw and wrong almost everywhere else: a tanh network cannot extrapolate a periodic function. ${map.which !== "stable" ? `The plant's state wanders, because [[A]] is a double integrator with eigenvalues at 1: the fresh rollout leaves the square [[[-8, 8]^2]] after ${fr7} of its 50 steps and reaches [[x_0 = 30]], where the network's output is whatever its last tanh layer saturated at.` : "On the bounded plant the fresh rollouts stay in the explored region, and the learned residual transfers as far as 50 samples allow."}`);
}
$$("#mapSeg .btn").forEach(b => b.addEventListener("click", () => { $$("#mapSeg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); map.which = b.dataset.w; drawMap(); }));

/* ============================================================ the study table */
function fillStudy() {
  /* Rows: plant, filter noise, T, Adam steps, final training loss, fresh-rollout forecast RMSE, zero-residual forecast RMSE, dynamics error.
     The first block is the repository's setting with more data or more steps; the second is the noise study (2000 steps). Both score the mean of fresh rollouts with keys 7, 11, 13. */
  const rows = [], S = D.study, zero7 = fmt(D.noise["original-50-0.1-0.5-filtered"].zero, 2);
  for (const key of Object.keys(S)) { const [T, steps] = key.split("-"); for (const [k, lk] of [["filtered", "loss_f"], ["forecast", "loss_p"]]) rows.push(["double integrator (repo)", `0.1 I, 0.5 I (repo)`, T, steps, k, fmt(S[key][lk], 4), fmt(S[key][k][1], 2), zero7, fmt(S[key][k][2], 2)]); }
  for (const [key, v] of Object.entries(D.noise)) { const [plant, T, Q, R, k] = key.split("-"); rows.push([plant === "stable" ? "bounded (eigenvalues 0.9)" : "double integrator (repo)", `${Q} I, ${R} I${Q === "0.1" ? " (repo)" : " (honest)"}`, T, 2000, k, fmt(v.loss, 4), fmt(v.forecast, 2), fmt(v.zero, 2), fmt(v.dyn, 2)]); }
  $("#studyT tbody").innerHTML = rows.map(r => `<tr>${r.map((c, i) => i === 6 && +c < +r[7] ? `<td><b>${c}</b></td>` : `<td>${c}</td>`).join("")}</tr>`).join("");
}

function drawAll() { drawHero(); drawMap(); }
fillStudy();
drawAll();
onRedraw(drawAll);
})();
