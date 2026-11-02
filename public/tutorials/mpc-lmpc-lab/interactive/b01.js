/* Advanced MPC and Learning MPC, lab B1, Vehicle–bike collision avoidance: the live demos.
   The two convex strategy QPs (yield, pass) of the course's collision.py are assembled here and solved by
   the shared interior-point solver (mpc-lib.js), the strategy maps over a grid of initial states are
   recomputed on every slider change, the Euler reachability bounds are evaluated analytically, and the
   occupancy-window variant is solved too. The defaults reproduce the numbers script. Needs ../kit/kit.js
   and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { zeros, mm, mv, tr, dot, solveQP, predictionMatrices, legend } = window.MPC;

const G0 = 9.81;
const conflictBounds = (dBike = 100, lCar = 4.5, wBike = 8) => [dBike - 0.5 * (lCar + wBike), dBike + 0.5 * (lCar + wBike)];   // [d_max, d_min]
const vehicleMatrices = (Ts, disc) => [[[1, Ts], [0, 1]], disc === "zoh" ? [[0.5 * Ts * Ts], [Ts]] : [[0], [Ts]]];
const DEFAULTS = { N: 4, Ts: 1, dBike: 100, lCar: 4.5, wBike: 8, conflictSteps: null, mu: 0.10, vMax: 32, vRef: 20, Qv: 3, Ru: 1, Pv: 3, yieldBoundary: "geometric", terminalStop: false, discretization: "euler" };
/* one convex QP for one logical strategy, exactly as the course's solve_collision_strategy */
function solveStrategy(x0, strategy, opts = {}) {
  const o = { ...DEFAULTS, ...opts }, N = o.N, [A, B] = vehicleMatrices(o.Ts, o.discretization), { Sx, Su } = predictionMatrices(A, B, N), [dMax, dMin] = conflictBounds(o.dBike, o.lCar, o.wBike);
  const dYield = o.yieldBoundary === "center_line" ? o.dBike : dMax, steps = o.conflictSteps || [N], amax = o.mu * G0;
  if (x0[1] < -1e-10 || x0[1] > o.vMax + 1e-10) return { strategy, success: false, J: Infinity, X: [x0], U: [] };
  const Qbar = zeros(2 * N, 2 * N); for (let k = 1; k <= N; k++) Qbar[2 * (k - 1) + 1][2 * (k - 1) + 1] = k === N ? o.Pv : o.Qv;
  const Sxx = mv(Sx, x0), base = Sxx.map((v, i) => v - (i % 2 === 1 ? o.vRef : 0)), SuTQ = mm(tr(Su), Qbar);
  const H = mm(SuTQ, Su).map((r, i) => r.map((v, j) => 2 * (v + (i === j ? o.Ru : 0)))), f = mv(SuTQ, base).map(v => 2 * v);
  const Gm = [], g = [], E = [], e = [];
  for (let k = 1; k <= N; k++) { const idx = 2 * (k - 1) + 1; Gm.push(Su[idx]); g.push(o.vMax - Sxx[idx]); Gm.push(Su[idx].map(v => -v)); g.push(Sxx[idx]); }
  for (const k of steps) { const idx = 2 * (k - 1); if (strategy === "yield") { Gm.push(Su[idx]); g.push(dYield - Sxx[idx]); } else { Gm.push(Su[idx].map(v => -v)); g.push(-dMin + Sxx[idx]); } }
  if (o.terminalStop && strategy === "yield") { const idx = 2 * (N - 1) + 1; E.push(Su[idx]); e.push(-Sxx[idx]); }
  const r = solveQP(H, f, Gm, g, E, e, new Array(N).fill(-amax), new Array(N).fill(amax), { iters: 60 });
  const U = r.z, Xp = Sxx.map((v, i) => v + dot(Su[i], U)), X = [x0]; for (let k = 0; k < N; k++) X.push([Xp[2 * k], Xp[2 * k + 1]]);
  const success = r.success, J = o.Ru * U.reduce((s, a) => s + a * a, 0) + o.Qv * X.slice(0, N).reduce((s, x) => s + (x[1] - o.vRef) ** 2, 0) + o.Pv * (X[N][1] - o.vRef) ** 2;
  return { strategy, success, J: success ? J : Infinity, X, U, dYield, dMin, dMax };
}
function chooseStrategy(x0, opts) { const sols = [solveStrategy(x0, "yield", opts), solveStrategy(x0, "pass", opts)], feas = sols.filter(s => s.success); return { best: feas.length ? feas.reduce((a, b) => b.J < a.J ? b : a) : null, sols }; }
function strategyGrid(dValues, vValues, opts) { const count = [], best = []; for (const v0 of vValues) { const rc = [], rb = []; for (const d0 of dValues) { const { best: b, sols } = chooseStrategy([d0, v0], opts); rc.push(sols.filter(s => s.success).length); rb.push(b ? (b.strategy === "yield" ? 0 : 1) : -1); } count.push(rc); best.push(rb); } return { count, best }; }
/* analytic terminal-position extrema for the Euler model when the speed bounds are inactive */
function terminalExtrema(x0, N, Ts, mu) { const lever = Ts * Ts * N * (N - 1) / 2, nom = x0[0] + N * Ts * x0[1], amax = mu * G0; return [nom - amax * lever, nom + amax * lever]; }

const DV = Array.from({ length: 20 }, (_, i) => 5 * i), VV = Array.from({ length: 33 }, (_, i) => i), RVALS = [0.1, 1, 10, 100, 1000];

/* =====================================================================
   Hero: strategy maps over the initial-state grid, and the two QPs at a chosen state
   ===================================================================== */
const hs = { mode: "count", probe: [60, 5], cache: {} };
function heroOpts() { return { N: +$("#hN").value, mu: +$("#hMu").value, Ru: RVALS[+$("#hR").value], yieldBoundary: $("#hYB").getAttribute("aria-pressed") === "true" ? "center_line" : "geometric", terminalStop: $("#hStop").getAttribute("aria-pressed") === "true", discretization: $("#hZoh").getAttribute("aria-pressed") === "true" ? "zoh" : "euler" }; }
function heroGrid(o) { const key = JSON.stringify(o); if (!hs.cache[key]) hs.cache[key] = strategyGrid(DV, VV, o); return hs.cache[key]; }
function drawHero() {
  const o = heroOpts(), grid = heroGrid(o), cv = $("#mapC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.6, { x: [-2.5, 97.5], y: [-0.5, 32.5], xlabel: "initial position d₀ [m]", ylabel: "initial speed v₀ [m/s]", xticks: [0, 20, 40, 60, 80], yticks: [0, 8, 16, 24, 32] }), { ctx, X, Y } = fr; cv._fr = fr;
  const cw = X(5) - X(0), ch = Y(0) - Y(1);
  VV.forEach((v0, i) => DV.forEach((d0, j) => { const c = grid.count[i][j], b = grid.best[i][j]; ctx.fillStyle = hs.mode === "count" ? (c === 0 ? css("--warm") : c === 1 ? alpha("--post", 0.35) : css("--post")) : (b === -1 ? css("--warm") : b === 0 ? alpha("--c5", 0.55) : alpha("--post", 0.75)); ctx.fillRect(X(d0 - 2.5), Y(v0 + 0.5), cw + 0.5, ch + 0.5); }));
  const p = hs.probe; ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; ctx.strokeRect(X(p[0] - 2.5), Y(p[1] + 0.5), cw, ch);
  legend(ctx, hs.mode === "count" ? ["orange: none feasible · light: one · blue: both"] : ["orange: none · purple: yield cheaper · blue: pass cheaper"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* the two QPs at the probe */
  const { best, sols } = chooseStrategy(p, o), [dMax, dMin] = conflictBounds(), dYield = o.yieldBoundary === "center_line" ? 100 : dMax, N = o.N;
  const cv2 = $("#trajC"), fr2 = frame(cv2, narrow ? 1.5 : 2.3, { x: [0, N], y: [0, 130], xlabel: "prediction step k", xticks: Array.from({ length: N + 1 }, (_, k) => k), yticks: [0, 40, 80, 120] }), ctx2 = fr2.ctx;
  ctx2.fillStyle = alpha("--warm", 0.18); ctx2.fillRect(fr2.pad.l, fr2.Y(dMin), fr2.iw, fr2.Y(dMax) - fr2.Y(dMin));
  ctx2.strokeStyle = alpha("--warm", 0.8); ctx2.setLineDash([5, 4]); seg(ctx2, fr2.pad.l, fr2.Y(dYield), fr2.w - fr2.pad.r, fr2.Y(dYield)); seg(ctx2, fr2.pad.l, fr2.Y(dMin), fr2.w - fr2.pad.r, fr2.Y(dMin)); ctx2.setLineDash([]);
  ctx2.fillStyle = alpha("--ink", 0.12); ctx2.fillRect(fr2.X(N) - 3, fr2.pad.t, 6, fr2.ih);
  sols.forEach(sol => { if (!sol.success) return; plotLine(fr2, sol.X.map((_, k) => k), sol.X.map(x => x[0]), { color: sol.strategy === "yield" ? css("--c5") : css("--post"), width: sol === best ? 2.6 : 1.4, dash: sol === best ? null : [4, 3] }); });
  legend(ctx2, ["purple: yield plan · blue: pass plan", "band: forbidden centre positions"], fr2.pad.l + 4, fr2.pad.t + fr2.ih - (narrow ? 44 : 18), narrow);
  $("#hNV").textContent = N; $("#hMuV").textContent = fmt(o.mu, 2); $("#hRV").textContent = o.Ru; $$("#demo [data-mode]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.mode === hs.mode)));
  const counts = { 0: 0, 1: 0, 2: 0 }, bests = { "-1": 0, 0: 0, 1: 0 }; grid.count.forEach(r => r.forEach(c => counts[c]++)); grid.best.forEach(r => r.forEach(b => bests[b]++));
  $("#rNone").textContent = `${counts[0]} of 660`; $("#rBoth").textContent = String(counts[2]); $("#rPass").textContent = String(bests[1]);
  const y = sols[0], ps = sols[1]; $("#rProbe").textContent = `yield ${y.success ? fmt(y.J, 1) : "infeasible"} · pass ${ps.success ? fmt(ps.J, 1) : "infeasible"}`;
  const nb = N === 4 && o.mu === 0.1 && o.Ru === 1 && o.yieldBoundary === "geometric" && !o.terminalStop && o.discretization === "euler", amax = o.mu * G0;
  let text = `${nb ? "The lab's setting: " : ""}[[\\mu = ${fmt(o.mu, 2)}]] gives [[|a_k| \\le ${fmt(amax, 3)}]] m/s², [[N = ${N}]], [[R = ${o.Ru}]], ${o.discretization === "zoh" ? "exact zero-order-hold" : "explicit Euler"} model, ${o.yieldBoundary === "center_line" ? "the looser centre-line yield bound [[d_N \\le 100]]" : "the geometric yield bound [[d_N \\le 93.75]]"}${o.terminalStop ? " with a full stop [[v_N = 0]] required to yield" : ""}. Over the 660 initial states of the grid, ${counts[0]} have no feasible strategy, ${counts[1]} exactly one and ${counts[2]} both; pass is the cheaper feasible choice at ${bests[1]} states and yield at ${bests[0]}. `;
  text += `At the outlined state [[(d_0, v_0) = (${fmt(p[0], 0)}, ${fmt(p[1], 0)})]]: yield is ${y.success ? `feasible with cost ${fmt(y.J, 1)} and ends at [[d_N = ${fmt(y.X[N][0], 2)}]], [[v_N = ${fmt(y.X[N][1], 2)}]]${y.X[N][0] < dYield - 1e-6 ? " (the yield bound is not even active: the car cannot reach the region, so the QP just tracks the reference speed)" : " (the yield bound is active)"}` : "infeasible"}; pass is ${ps.success ? `feasible with cost ${fmt(ps.J, 1)}, ending at [[d_N = ${fmt(ps.X[N][0], 2)}]]` : "infeasible"}. ${best ? `The lower-cost feasible strategy is <b>${best.strategy}</b>.` : "Neither modelled longitudinal manoeuvre exists from here; what that does and does not mean is Part 8."} `;
  const ex = terminalExtrema(p, N, 1, o.mu); text += o.discretization === "euler" ? `The Euler reachability bounds of Part 3 put [[d_N]] in [[[${fmt(ex[0], 2)}, ${fmt(ex[1], 2)}]]] when the speed limits stay inactive, against the yield bound ${fmt(dYield, 2)} and the pass bound ${fmt(dMin, 2)}.` : "";
  caption($("#heroStatus"), text + " Tap or click the map to move the state.");
}
["#hN", "#hMu", "#hR"].forEach(id => $(id).addEventListener("input", drawHero));
$$("#demo [data-mode]").forEach(bt => bt.addEventListener("click", () => { hs.mode = bt.dataset.mode; drawHero(); }));
["#hYB", "#hStop", "#hZoh"].forEach(id => $(id).addEventListener("click", () => { const b = $(id); b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true")); drawHero(); }));
$("#mapC").addEventListener("click", ev => { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), fr = cv._fr; if (!fr) return; const d = fr.x0 + (ev.clientX - r.left - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), v = fr.y1 - (ev.clientY - r.top - fr.pad.t) / fr.ih * (fr.y1 - fr.y0); hs.probe = [clamp(Math.round(d / 5) * 5, 0, 95), clamp(Math.round(v), 0, 32)]; drawHero(); });
$("#hReset").addEventListener("click", () => { $("#hN").value = 4; $("#hMu").value = 0.1; $("#hR").value = 1; ["#hYB", "#hStop", "#hZoh"].forEach(id => $(id).setAttribute("aria-pressed", "false")); hs.mode = "count"; hs.probe = [60, 5]; drawHero(); });

/* =====================================================================
   Part 3: analytic reachability under the Euler model
   ===================================================================== */
function drawReach() {
  const v0 = +$("#rV").value, mu = +$("#rMu").value, N = +$("#rN").value, x0 = [+$("#rD").value, v0], [lo, hi] = terminalExtrema(x0, N, 1, mu), [dMax, dMin] = conflictBounds(), muCrit = 12.5 / (G0 * N * (N - 1)), width = mu * G0 * N * (N - 1);
  const { sols } = chooseStrategy(x0, { N, mu }), cv = $("#reachC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.7 : 3, { x: [40, 140], y: [0, 1], xlabel: "terminal centre position d_N [m]", yticks: false, xticks: [40, 60, 80, 100, 120, 140] }), { ctx, X } = fr;
  ctx.fillStyle = alpha("--warm", 0.25); ctx.fillRect(X(dMax), fr.pad.t + 10, X(dMin) - X(dMax), fr.ih - 20);
  ctx.fillStyle = alpha("--post", 0.3); ctx.fillRect(X(lo), fr.pad.t + 25, X(hi) - X(lo), fr.ih - 50); ctx.strokeStyle = css("--post"); ctx.lineWidth = 2; ctx.strokeRect(X(lo), fr.pad.t + 25, X(hi) - X(lo), fr.ih - 50);
  ctx.fillStyle = css("--ink"); for (const s of sols) if (s.success) { ctx.beginPath(); ctx.arc(X(s.X[N][0]), fr.pad.t + fr.ih / 2, 5, 0, 2 * Math.PI); ctx.fill(); }
  legend(ctx, ["blue: reachable d_N · orange: forbidden gap", "dots: the two QPs' terminal positions"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#rDV").textContent = fmt(x0[0], 0); $("#rVV").textContent = fmt(v0, 0); $("#rMuV").textContent = fmt(mu, 2); $("#rNV").textContent = N;
  const nb = x0[0] === 60 && v0 === 5 && mu === 0.1 && N === 4, y = sols[0], ps = sols[1];
  caption($("#reachCap"), `From [[(d_0, v_0) = (${fmt(x0[0], 0)}, ${fmt(v0, 0)})]] with [[\\mu = ${fmt(mu, 2)}]] and [[N = ${N}]]${nb ? ", the nominal case" : ""}: [[d_N = d_0 + N T_s v_0 + T_s^2 \\sum_{j=0}^{N-2} (N - 1 - j) a_j]], so with every position-relevant acceleration at [[\\pm\\mu g]] the terminal position ranges over [[[${fmt(lo, 3)}, ${fmt(hi, 3)}]]] (the last acceleration [[a_{N-1}]] moves [[v_N]] only). The width [[\\mu g T_s^2 N(N-1) = ${fmt(width, 2)}]] m is ${width >= 12.5 ? "at least" : "less than"} the forbidden gap of 12.5 m: the critical friction for the two strategy regions to be able to overlap is [[\\mu_{\\mathrm{crit}} = \\frac{12.5}{g T_s^2 N(N-1)} = ${fmt(muCrit, 4)}]]. Here the yield bound 93.75 is ${hi < dMax ? "beyond the reachable interval, so yielding is free and passing impossible" : lo > dMax ? "below the whole interval: the car cannot stay short of the region" : "inside the interval"}${hi >= dMin ? "; the pass bound 106.25 is reachable" : "; the pass bound 106.25 is out of reach"}. The QPs agree: yield ${y.success ? "feasible, ending at " + fmt(y.X[N][0], 2) : "infeasible"}, pass ${ps.success ? "feasible, ending at " + fmt(ps.X[N][0], 2) : "infeasible"}. The formulas assume the speed bounds [[0 \\le v \\le 32]] never bind; near them the reachable interval is narrower than drawn.`);
}
["#rD", "#rV", "#rMu", "#rN"].forEach(id => $(id).addEventListener("input", drawReach));

/* =====================================================================
   Part 7: the occupancy window
   ===================================================================== */
function drawWindow() {
  const x0 = [+$("#wD").value, +$("#wV").value], N = 8, [dMax, dMin] = conflictBounds();
  const once = solveStrategy(x0, "yield", { N, conflictSteps: [4] }), win = solveStrategy(x0, "yield", { N, conflictSteps: [4, 5, 6] }), pw = solveStrategy(x0, "pass", { N, conflictSteps: [4, 5, 6] });
  const cv = $("#winC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, 8], y: [0, 170], xlabel: "time step k", xticks: [0, 2, 4, 6, 8], yticks: [0, 50, 100, 150] }), { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--ink", 0.08); ctx.fillRect(X(4), fr.pad.t, X(6) - X(4), fr.ih);
  ctx.fillStyle = alpha("--warm", 0.18); ctx.fillRect(fr.pad.l, Y(dMin), fr.iw, Y(dMax) - Y(dMin)); ctx.strokeStyle = alpha("--warm", 0.8); ctx.setLineDash([5, 4]); seg(ctx, fr.pad.l, Y(dMax), fr.w - fr.pad.r, Y(dMax)); seg(ctx, fr.pad.l, Y(dMin), fr.w - fr.pad.r, Y(dMin)); ctx.setLineDash([]);
  if (once.success) plotLine(fr, once.X.map((_, k) => k), once.X.map(x => x[0]), { color: css("--c5"), width: 1.6, dash: [4, 3] });
  if (win.success) plotLine(fr, win.X.map((_, k) => k), win.X.map(x => x[0]), { color: css("--post"), width: 2.4 });
  if (pw.success) plotLine(fr, pw.X.map((_, k) => k), pw.X.map(x => x[0]), { color: css("--ink"), width: 1.6, dash: [2, 3] });
  legend(ctx, ["grey: bicycle present · dashed: yield at k = 4 only", "blue: yield at k = 4, 5, 6 · dotted: pass"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#wDV").textContent = fmt(x0[0], 0); $("#wVV").textContent = fmt(x0[1], 0);
  const nb = x0[0] === 60 && x0[1] === 5;
  let text = `[[N = 8]], the bicycle occupies the region at [[k = 4, 5, 6]], from [[(d_0, v_0) = (${fmt(x0[0], 0)}, ${fmt(x0[1], 0)})]]${nb ? ", the nominal state" : ""}. `;
  text += once.success ? `Enforcing [[d_4 \\le d_{\\max}]] only is feasible (cost ${fmt(once.J, 1)}), with [[d_4, d_5, d_6 = ${once.X.slice(4, 7).map(x => fmt(x[0], 1)).join(", ")}]]${once.X[5][0] > dMax + 1e-6 || once.X[6][0] > dMax + 1e-6 ? ": the plan is safe at the instant the bicycle arrives and drives into the region a step later, while the bicycle is still there" : ""}. ` : "Even the single-instant yield is infeasible from here. ";
  text += win.success ? `Enforcing it at all three steps is feasible too (cost ${fmt(win.J, 1)}): the car slows so that [[d_6 = ${fmt(win.X[6][0], 2)} \\le 93.75]], then accelerates once the bicycle has gone, reaching [[d_8 = ${fmt(win.X[8][0], 1)}]]. ` : `Enforcing it at all three steps is <b>infeasible</b>: there is no admissible way to stay short of the region until [[k = 6]]${once.success ? ", although the single-instant version claimed success. Exercise 6 asks for exactly such a state" : ""}. `;
  text += pw.success ? `Passing before the window is feasible (cost ${fmt(pw.J, 1)}).` : "Passing before the window is infeasible. ";
  caption($("#winCap"), text + " One binary variable for the whole window commits to one topological manoeuvre; a binary per step could switch sides mid-window, which physically means crossing the bicycle.");
}
["#wD", "#wV"].forEach(id => $(id).addEventListener("input", drawWindow));

function drawAll() { drawHero(); drawReach(); drawWindow(); }
drawAll();
onRedraw(drawAll);
})();
