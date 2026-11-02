/* Advanced MPC and Learning MPC, lab B2, Learning MPC on a repeated task: the demos.
   The hero runs the whole experiment live on the canonical cart: a plain short-horizon MPC seed, then
   LMPC iterations with a chosen horizon and trajectory window, drawn in the phase plane with the
   long-horizon reference and as a learning curve. The other widgets dissect the result: safe-set hulls,
   input sequences and stored costs-to-go, and the size of the online problem with and without the window.
   Needs mpc-lib.js and mpc-lmpc.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { SYS, mv, dot, hull, area, vertices, drawPoly, legend, drawPath } = window.MPC;
const L = window.LMPC;

const T = L.makeTask({ A: SYS.A, B: SYS.B, Q: SYS.Q, R: SYS.R, xlb: SYS.xlb, xub: SYS.xub, umin: SYS.umin, umax: SYS.umax, xS: [-3.5, 2.5] });
const REF = L.solveFTOCP(T, T.xS, 30);
const sup = v => "10" + String(v).replace("-", "⁻").replace(/\d/g, ch => "⁰¹²³⁴⁵⁶⁷⁸⁹"[ch]);

/* =====================================================================
   Hero: the learning experiment, run live
   ===================================================================== */
const hs = { N: 4, window: 4, iters: 9, seedN: 2, result: null, running: false, cache: {} };
function heroCompute(done) {
  const key = `${hs.N}|${hs.window}|${hs.iters}|${hs.seedN}`; if (hs.cache[key]) { hs.result = hs.cache[key]; done(); return; }
  const seed = L.runSeed(T, hs.seedN, 400), res = { seedN: hs.seedN, N: hs.N, window: hs.window, seed, iterations: [], costs: [], store: L.makeStore(), first: null, failed: seed.failed || null };
  hs.result = res; if (seed.failed) { hs.cache[key] = res; done(); return; }
  L.addTrajectory(T, res.store, seed.X, seed.U); res.costs.push(res.store.costs[0]);
  let j = 0; hs.running = true;
  const step = () => {
    if (j >= hs.iters) { hs.running = false; hs.cache[key] = res; done(); return; }
    const t0 = performance.now(), it = L.runIteration(T, res.store, hs.N, hs.window === 0 ? null : hs.window); it.ms = performance.now() - t0;
    if (it.failed) { res.failed = `iteration ${j + 1}: ${it.failed}`; hs.running = false; hs.cache[key] = res; done(); return; }
    if (!res.first) res.first = it.first; res.iterations.push(it); res.costs.push(it.cost); j++; done(true); setTimeout(step, 10);
  };
  done(true); setTimeout(step, 10);
}
function drawHero() {
  const r = hs.result, cv = $("#phaseC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.15 : 1.5, { x: [-4.4, 4.4], y: [-3.3, 3.3], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, -1.5, 0, 1.5, 3] }), { ctx, X, Y } = fr;
  if (!r) return;
  drawPoly(fr, vertices(SYS.X), alpha("--ink", 0.03), alpha("--ink", 0.3), 1, [3, 3]);
  plotLine(fr, REF.X.map(x => x[0]), REF.X.map(x => x[1]), { color: alpha("--ink", 0.5), width: 2.2, dash: [6, 4] });
  if (r.seed.X) drawPath(fr, r.seed.X, css("--warm"), 1.8, 2.3);
  r.iterations.forEach((it, i) => { const t = r.iterations.length > 1 ? i / (r.iterations.length - 1) : 1; plotLine(fr, it.X.map(x => x[0]), it.X.map(x => x[1]), { color: alpha("--post", 0.25 + 0.75 * t), width: i === r.iterations.length - 1 ? 2.4 : 1.2 }); });
  legend(ctx, ["orange: seed MPC · blue: LMPC iterations (darker = later) · dashed: long-horizon reference"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* learning curve: the gap to the reference, log scale */
  const cv2 = $("#curveC"), costs = r.costs, nIt = Math.max(hs.iters, 1);
  const fr2 = frame(cv2, narrow ? 1.5 : 2.4, { x: [-0.5, nIt + 0.5], y: [-10, 2], xlabel: "iteration (0 = seed)", xticks: Array.from({ length: nIt + 1 }, (_, i) => i), yticks: [-10, -8, -6, -4, -2, 0, 2], yfmt: sup }), ctx2 = fr2.ctx;
  const gaps = costs.map(c => Math.log10(Math.max(c - REF.objective, 1e-10)));
  plotLine(fr2, gaps.map((_, i) => i), gaps, { color: css("--post"), width: 2 }); ctx2.fillStyle = css("--post"); gaps.forEach((v, i) => { ctx2.beginPath(); ctx2.arc(fr2.X(i), fr2.Y(v), 3, 0, 2 * Math.PI); ctx2.fill(); });
  legend(ctx2, ["gap to the long-horizon reference, J^j − J*, log scale (floor 10⁻¹⁰)"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  $("#hNV").textContent = hs.N; $("#hWV").textContent = hs.window === 0 ? "all" : String(hs.window); $("#hIV").textContent = hs.iters; $("#hSV").textContent = hs.seedN;
  const last = costs[costs.length - 1];
  $("#rSeed").textContent = r.seed.failed ? "failed" : `${r.seed.U.length} inputs, ${fmt(costs[0], 4)}`; $("#rLast").textContent = r.iterations.length ? `${r.iterations[r.iterations.length - 1].U.length} inputs, ${fmt(last, 6)}` : "–"; $("#rRef").textContent = fmt(REF.objective, 6); $("#rGap").textContent = r.iterations.length ? (last - REF.objective).toExponential(2) : "–";
  const nb = hs.N === 4 && hs.window === 4 && hs.iters === 9 && hs.seedN === 2;
  let text;
  if (r.seed.failed) text = `The seed MPC with [[N = ${hs.seedN}]] ${r.seed.failed === "infeasible" ? "hit an infeasible problem" : "never reached the target in 400 steps"}: LMPC needs one successful trajectory to start, and this controller does not provide one.`;
  else {
    text = `${nb ? "The lab's setting: " : ""}plain MPC with [[N = ${hs.seedN}]] and no terminal ingredients reaches the target in ${r.seed.U.length} inputs (the last two are the exact bridge) at a complete task cost of ${fmt(costs[0], 4)}, against ${fmt(REF.objective, 4)} for the long-horizon reference: ${fmt(100 * (costs[0] - REF.objective) / REF.objective, 1)} percent above it${hs.seedN === 2 ? ", because its first input is 0: the two-step objective sees no benefit in braking yet, and the cart overshoots to p = 2" : ""}. `;
    if (hs.running) text += `Running LMPC iteration ${r.iterations.length + 1} of ${hs.iters}… `;
    if (r.iterations.length) {
      const diffs = costs.slice(1).map((c, i) => c - costs[i]), inc = diffs.filter(d => d > 1e-7).length, lastIt = r.iterations[r.iterations.length - 1];
      text += `LMPC with [[N = ${hs.N}]] and ${hs.window === 0 ? "every stored trajectory" : "the last " + hs.window + " stored trajectories"} in the safe set: after ${r.iterations.length} iteration${r.iterations.length > 1 ? "s" : ""} the cost is ${fmt(last, 6)}, ${(last - REF.objective).toExponential(1)} above the reference (${(100 * (last - REF.objective) / REF.objective).toExponential(1)} percent), in ${lastIt.U.length} inputs. The iteration cost ${inc === 0 ? "never increased" : `increased on ${inc} of ${diffs.length} steps, by at most ${Math.max(...diffs).toExponential(1)}, the solver's tolerance, not a counter-example to the theory`}; the first iteration removed ${fmt(100 * -diffs[0] / (costs[0] - REF.objective), 1)} percent of the gap, because the stored seed already contains a whole safe path and its costs-to-go, which the ${hs.seedN}-step horizon could not see. `;
      if (diffs.length > 1 && Math.abs(diffs[diffs.length - 1]) < 1e-9 && last - REF.objective > 1e-4) text += `Then it stalls: with [[N = ${hs.N}]] the learned terminal value is too pessimistic between the stored states for the horizon to find anything cheaper than replaying the previous run (Part 6). `;
      text += `The first LMPC solve from the start applies [[u_0 = ${fmt(r.first.U[0], 4)}]] and lands its terminal state at [[(${fmt(r.first.terminal[0], 3)}, ${fmt(r.first.terminal[1], 3)})]], a convex combination of ${r.first.lam.filter(l => l > 1e-6).length} stored state${r.first.lam.filter(l => l > 1e-6).length > 1 ? "s" : ""}. Mean solve time ${fmt(r.iterations.reduce((s, it) => s + it.ms, 0) / r.iterations.reduce((s, it) => s + it.U.length, 0), 1)} ms per QP in this browser. ` + (r.failed ? `Then: ${r.failed}.` : "");
    }
  }
  caption($("#heroStatus"), text);
}
function heroStart() { heroCompute(partial => { drawHero(); if (!partial) { drawGrowth(); drawDetail(); } }); }
$("#hN").addEventListener("input", () => { hs.N = +$("#hN").value; heroStart(); });
$("#hW").addEventListener("input", () => { hs.window = +$("#hW").value; heroStart(); });
$("#hI").addEventListener("input", () => { hs.iters = +$("#hI").value; heroStart(); });
$("#hS").addEventListener("input", () => { hs.seedN = +$("#hS").value; heroStart(); });
$("#hReset").addEventListener("click", () => { $("#hN").value = 4; $("#hW").value = 4; $("#hI").value = 9; $("#hS").value = 2; hs.N = 4; hs.window = 4; hs.iters = 9; hs.seedN = 2; heroStart(); });

/* =====================================================================
   Part 4: hull growth, input sequences, stored costs-to-go
   ===================================================================== */
const det = { view: "hull" };
function drawDetail() {
  const r = hs.result; if (!r || r.seed.failed) return;
  const cv = $("#detC"), narrow = narrowOf(cv), Tr = r.store.trajectories, last = r.iterations[r.iterations.length - 1];
  if (det.view === "hull") {
    const pointsOf = list => { const P = [[0, 0]]; list.forEach(X => X.slice(0, -1).forEach(x => P.push(x))); return P; };
    const nMid = Math.min(2, Tr.length - 1), stages = [[[Tr[0]], css("--warm")], [Tr.slice(0, nMid + 1), alpha("--post", 0.5)], [Tr, css("--post")]];
    const fr = frame(cv, narrow ? 1.15 : 1.6, { x: [-4.4, 4.4], y: [-3.3, 3.3], xlabel: "p", ylabel: "v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, 0, 3] }), { ctx, X, Y } = fr;
    const all = pointsOf(Tr); ctx.fillStyle = alpha("--ink", 0.35); all.forEach(p => { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 1.8, 0, 2 * Math.PI); ctx.fill(); });
    const areas = stages.map(([list, color], i) => { const H = hull(pointsOf(list)); if (H.length >= 3) drawPoly(fr, H, null, color, i === 2 ? 2.4 : 1.6); return area(H); });
    legend(ctx, ["orange: hull after the seed · light blue: after " + nMid + " iterations · blue: after all · dots: stored states"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    caption($("#detCap"), `Convex hulls of the accumulated successful states: areas ${fmt(areas[0], 3)} after the seed, ${fmt(areas[1], 3)} after ${nMid} iteration${nMid > 1 ? "s" : ""}, ${fmt(areas[2], 3)} after all ${Tr.length - 1}. ${Math.abs(areas[2] - areas[0]) < 1e-6 ? "The hull never changes: the slow seed's overshoot already encloses every faster run, so the geometry was complete after one trajectory and everything the later iterations learned is in the stored values." : `The hull grows by a factor ${fmt(areas[2] / Math.max(areas[0], 1e-9), 2)} and then stops, while the cost keeps falling.`} The set is only half of the learning.`);
  } else if (det.view === "ctrl") {
    const seqs = [[r.seed.U, css("--warm"), []], [last ? last.U : [], css("--post"), []], [REF.U, alpha("--ink", 0.55), [6, 4]]];
    const Lm = Math.max(...seqs.map(s => s[0].length)), fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, Lm], y: [-1.2, 1.2], xlabel: "time step", ylabel: "u", yticks: [-1, 0, 1] }), { ctx, X, Y } = fr;
    ctx.strokeStyle = alpha("--ink", 0.3); ctx.setLineDash([3, 3]); seg(ctx, X(0), Y(1), X(Lm), Y(1)); seg(ctx, X(0), Y(-1), X(Lm), Y(-1)); ctx.setLineDash([]);
    seqs.forEach(([U, color, dash]) => { if (!U.length) return; const xs = [], ys = []; U.forEach((u, k) => { xs.push(k, k + 1); ys.push(u, u); }); plotLine(fr, xs, ys, { color, width: 2, dash }); });
    legend(ctx, ["orange: seed MPC · blue: last LMPC iteration · dashed: reference · dotted: |u| = 1"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    const sat = U => U.filter(u => Math.abs(u) > 1 - 1e-6).length;
    caption($("#detCap"), `Input sequences. The seed saturates on ${sat(r.seed.U)} steps (largest input ${fmt(Math.max(...r.seed.U.map(Math.abs)), 3)}) and starts with ${fmt(r.seed.U[0], 3)}; the last LMPC iteration ${last ? `starts with ${fmt(last.U[0], 3)} and saturates on ${sat(last.U)} steps` : "is not available"}, like the reference, which saturates on ${sat(REF.U)}. Braking hard at once is what the learned value taught the short horizon.`);
  } else {
    const J0 = L.costToGo(T, r.seed.X, r.seed.U), J1 = last ? L.costToGo(T, last.X, last.U) : [], top = Math.max(J0[0], 1) * 1.1;
    const fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, Math.max(J0.length, J1.length)], y: [0, top], xlabel: "time step t", ylabel: "stored J_t", yticks: [0, 10, 20, 30] }), { ctx, X, Y } = fr;
    plotLine(fr, J0.map((_, k) => k), J0, { color: css("--warm"), width: 2 }); if (J1.length) plotLine(fr, J1.map((_, k) => k), J1, { color: css("--post"), width: 2 });
    ctx.fillStyle = css("--warm"); J0.forEach((v, k) => { ctx.beginPath(); ctx.arc(X(k), Y(v), 2.2, 0, 2 * Math.PI); ctx.fill(); }); ctx.fillStyle = css("--post"); J1.forEach((v, k) => { ctx.beginPath(); ctx.arc(X(k), Y(v), 2.2, 0, 2 * Math.PI); ctx.fill(); });
    legend(ctx, ["orange: along the seed · blue: along the last LMPC iteration"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    let res = 0; if (last) for (let k = 0; k < last.U.length; k++) res = Math.max(res, Math.abs(J1[k] - L.stageCost(T, last.X[k], last.U[k]) - J1[k + 1]));
    caption($("#detCap"), `Stored costs-to-go [[J_t = q(x_t, u_t) + J_{t+1}]], [[J_T = 0]], along the seed and the last iteration: the values [[V_r]] that the terminal cost [[\\lambda^\\top V]] interpolates. [[J_0]] is the iteration cost of the learning curve, ${fmt(J0[0], 4)} for the seed${last ? ` and ${fmt(J1[0], 6)} for the last iteration` : ""}. The seed keeps half its cost for after the overshoot; the learned run pays almost all of it in the first braking steps. Backward-recursion residual on the last iteration: ${res.toExponential(1)}, round-off.`);
  }
}
$$("#detW [data-view]").forEach(b => b.addEventListener("click", () => { det.view = b.dataset.view; $$("#detW [data-view]").forEach(o => o.setAttribute("aria-pressed", String(o === b))); drawDetail(); }));

/* =====================================================================
   Part 5: database growth and the window
   ===================================================================== */
function drawGrowth() {
  const r = hs.result; if (!r || r.seed.failed) return;
  const sizes = []; let acc = 1; for (const X of r.store.trajectories) { acc += X.length - 1; sizes.push(acc); }
  const win = r.store.trajectories.map((_, j) => { const start = hs.window === 0 ? 0 : Math.max(0, j + 1 - hs.window); let s = 1; for (let i = start; i <= j; i++) s += r.store.trajectories[i].length - 1; return s; });
  const cv = $("#growC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.5 : 2.4, { x: [-0.5, sizes.length - 0.5], y: [0, Math.max(...sizes) * 1.15], xlabel: "stored trajectory index", xticks: sizes.map((_, i) => i) }), { ctx, X, Y } = fr;
  plotLine(fr, sizes.map((_, i) => i), sizes, { color: alpha("--ink", 0.5), width: 2, dash: [5, 4] }); plotLine(fr, win.map((_, i) => i), win, { color: css("--post"), width: 2.2 });
  ctx.fillStyle = css("--post"); win.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(i), Y(v), 3, 0, 2 * Math.PI); ctx.fill(); });
  legend(ctx, ["dashed: all stored safe states · blue: states in the online window"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const ms = r.iterations.map(it => it.ms / it.U.length);
  caption($("#growCap"), `Every successful iteration adds its states to the database: ${sizes[sizes.length - 1]} safe samples after ${r.store.trajectories.length} trajectories. Each sample used online is one barycentric variable [[\\lambda_r]] in the QP. With ${hs.window === 0 ? "all data active, the QP grows every iteration" : `a window of the last ${hs.window} trajectories the online problem stays at about ${win[win.length - 1]} variables`}; the mean solve time per QP in this browser went from ${fmt(ms[0], 1)} ms in the first iteration to ${fmt(ms[ms.length - 1], 1)} ms in the last. Set the window to "all" and compare: the costs agree to the solver's tolerance, the problem does not.`);
}

function drawAll() { drawHero(); drawGrowth(); drawDetail(); }
heroStart();
onRedraw(drawAll);
})();
