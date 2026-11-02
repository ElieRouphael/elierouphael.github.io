/* Advanced MPC and Learning MPC, chapter A13, the learned terminal value and the LMPC problem: the demos.
   The hero is the one-dimensional example: stored (x, J) pairs of the scalar integrator, their lower convex
   envelope computed by the barycentric linear program, a movable probe with its weights, the mixed input,
   the successor and the decrease residual there, and a button that adds a worse run. The second widget
   solves the LMPC problem on the canonical cart from the repeated start with the slow seed of chapter A12
   stored. Needs mpc-lib.js and mpc-lmpc.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { SYS, mv, dot, hull, vertices, drawPoly, legend, drawPath } = window.MPC;
const L = window.LMPC;

/* ---------------- the scalar example ---------------- */
const ctg = (xs, us) => { const J = new Array(xs.length).fill(0); for (let k = us.length - 1; k >= 0; k--) J[k] = xs[k] * xs[k] + 0.1 * us[k] * us[k] + J[k + 1]; return J; };
const RUN_A = { xs: [4, 3, 2, 1, 0], us: [-1, -1, -1, -1] }, RUN_B = { xs: [-3, -2, -1, 0], us: [1, 1, 1] }, RUN_W = { xs: [2, 1.6, 1.0, 0.4, 0], us: [-0.4, -0.6, -0.6, -0.4] };
const sc = { probe: 2.5, extra: false };
function scalarData() {
  const runs = [RUN_A, RUN_B].concat(sc.extra ? [RUN_W] : []), S = [[0]], V = [0], Us = [0], tag = [false];
  runs.forEach((r, i) => { const J = ctg(r.xs, r.us); r.xs.slice(0, -1).forEach((x, k) => { S.push([x]); V.push(J[k]); Us.push(r.us[k]); tag.push(i === 2); }); });
  return { S, V, Us, tag };
}
function drawHero() {
  const { S, V, Us, tag } = scalarData(), grid = Array.from({ length: 141 }, (_, i) => -3 + 7 * i / 140), Vc = grid.map(x => L.convexSafeValue([x], S, V).value);
  const cv = $("#scalarC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.3 : 2.0, { x: [-3.5, 4.5], y: [0, 33], xlabel: "x", ylabel: "cost-to-go", xticks: [-3, -2, -1, 0, 1, 2, 3, 4], yticks: [0, 10, 20, 30] }), { ctx, X, Y } = fr; cv._fr = fr;
  plotLine(fr, grid, Vc, { color: css("--post"), width: 2.2 });
  S.forEach((s, i) => { ctx.fillStyle = tag[i] ? css("--c5") : css("--warm"); ctx.beginPath(); ctx.arc(X(s[0]), Y(V[i]), 4.5, 0, 2 * Math.PI); ctx.fill(); });
  const q = L.convexSafeValue([sc.probe], S, V), act = q.lam.map((l, i) => [l, i]).filter(([l]) => l > 1e-6);
  const u = dot(q.lam, Us), xp = sc.probe + u, qp = L.convexSafeValue([xp], S, V), h = sc.probe * sc.probe + 0.1 * u * u, resid = qp.value - q.value + h;
  act.forEach(([l, i]) => { ctx.strokeStyle = alpha("--ink", 0.4); ctx.setLineDash([3, 3]); seg(ctx, X(sc.probe), Y(q.value), X(S[i][0]), Y(V[i])); ctx.setLineDash([]); });
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; seg(ctx, X(sc.probe), Y(q.value), X(xp), Y(qp.value));
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(sc.probe), Y(q.value), 5.5, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = css("--post"); ctx.beginPath(); ctx.arc(X(xp), Y(qp.value), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1; ctx.stroke();
  legend(ctx, ["orange: stored (x, J) pairs" + (sc.extra ? " · purple: the worse run's pairs" : "") + " · blue curve: V_c(x)", "black: the probe · ringed: its successor under the mixed input"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#scExtra").setAttribute("aria-pressed", String(sc.extra)); $("#hProbeV").textContent = fmt(sc.probe, 1);
  $("#rVal").textContent = fmt(q.value, 3); $("#rLam").textContent = act.map(([l, i]) => `${fmt(l, 2)}·${S[i][0]}`).join(" + "); $("#rU").textContent = fmt(u, 3); $("#rRes").textContent = resid.toExponential(2);
  let text = `Stored pairs from the two runs [[4 \\to 3 \\to 2 \\to 1 \\to 0]] and [[-3 \\to -2 \\to -1 \\to 0]] under [[x^+ = x + u]], [[q = x^2 + 0.1u^2]]${sc.extra ? ", plus a slower third run through 2, 1.6, 1.0 and 0.4" : ""}. The curve is the LP's answer for every [[x]]: the cheapest convex combination of stored pairs that represents [[x]], the lower convex envelope. At [[x = ${fmt(sc.probe, 1)}]], [[V_c = ${fmt(q.value, 3)}]] from ${act.length === 1 ? `the stored point ${S[act[0][1]][0]} alone` : act.map(([l, i]) => `[[\\lambda = ${fmt(l, 3)}]] on ${S[i][0]}`).join(" and ")}${Math.abs(sc.probe - 2.5) < 1e-9 && !sc.extra ? ", halfway between 2 and 3, so halfway between 5.2 and 14.3" : ""}. The same weights on the stored inputs give [[u = ${fmt(u, 3)}]] and the successor [[x^+ = ${fmt(xp, 2)}]] with [[V_c(x^+) = ${fmt(qp.value, 3)}]]; the decrease [[V_c(x^+) - V_c(x) + q(x, u) = ${resid.toExponential(2)}]] is ${resid <= 1e-9 ? "at most zero, as the proof of Part 3 promises" : "positive, which the proof excludes, so check the input bound"}${Math.abs(sc.probe - 2.5) < 1e-9 && !sc.extra ? ": a slack of 0.25, the convexity gap of [[x^2]] between 2.5² and the mean of 2² and 3²" : ""}. `;
  if (sc.extra) { const S0 = scalarData.base || (scalarData.base = (() => { const sv = sc.extra; sc.extra = false; const d = scalarData(); sc.extra = sv; return d; })()); const low = [0.4, 0.2, 0.7].map(x => [x, L.convexSafeValue([x], S0.S, S0.V).value, L.convexSafeValue([x], S, V).value]); text += `The slower run costs ${fmt(ctg(RUN_W.xs, RUN_W.us)[0], 3)} from [[x = 2]] against the stored 5.2, so its pairs at 2, 1.6 and 1.0 lie above the envelope: dominated, never selected. But its last pair [[(0.4, 0.176)]] lies below the old chord from [[(0, 0)]] to [[(1, 1.1)]], which was 0.44 there, so the envelope drops near the origin: [[V_c(0.4)]] falls from 0.44 to 0.176 and [[V_c(0.7)]] from 0.77 to ${fmt(low[2][2], 3)}. A worse run can still contain a better partial continuation, and the LP keeps exactly that part; worse data never raise the terminal value.`; }
  else text += "Tap or click the chart to move the probe; the button adds a worse run.";
  caption($("#heroStatus"), text);
}
$("#scalarC").addEventListener("click", ev => { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), fr = cv._fr; if (!fr) return; sc.probe = clamp(Math.round((fr.x0 + (ev.clientX - r.left - fr.pad.l) / fr.iw * (fr.x1 - fr.x0)) * 10) / 10, -3, 4); drawHero(); });
$("#scExtra").addEventListener("click", () => { sc.extra = !sc.extra; drawHero(); });
$("#hReset").addEventListener("click", () => { sc.probe = 2.5; sc.extra = false; drawHero(); });

/* ---------------- the LMPC problem on the cart, first solve from x_S with the slow seed stored ---------------- */
const T = L.makeTask({ A: SYS.A, B: SYS.B, Q: SYS.Q, R: SYS.R, xlb: SYS.xlb, xub: SYS.xub, umin: SYS.umin, umax: SYS.umax, xS: [-3.5, 2.5] });
const seed = L.runSeed(T, 2), store = L.makeStore(); L.addTrajectory(T, store, seed.X, seed.U);
const lm = { N: 3, cache: {} };
function drawLmpc() {
  const N = lm.N, key = String(N), r = lm.cache[key] || (lm.cache[key] = L.lmpcSolve(T, store, T.xS, N, null));
  const act = r.lam.map((l, i) => [l, i]).filter(([l]) => l > 1e-6), data = L.safeData(store, null);
  const uApp = dot(r.lam, r.Us), xApp = mv(T.A, r.terminal).map((v, i) => v + T.B[i][0] * uApp);
  const x1 = mv(T.A, T.xS).map((v, i) => v + T.B[i][0] * r.U[0]), r1 = L.lmpcSolve(T, store, x1, N, null);
  const q0 = L.stageCost(T, T.xS, r.U[0]);
  const cv = $("#lmpcC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.15 : 1.5, { x: [-4.4, 4.4], y: [-3.3, 3.3], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, -1.5, 0, 1.5, 3] }), { ctx, X, Y } = fr;
  drawPoly(fr, vertices(SYS.X), alpha("--ink", 0.03), alpha("--ink", 0.3), 1, [3, 3]);
  const H = hull(data.S); if (H.length >= 3) drawPoly(fr, H, alpha("--post", 0.1), alpha("--post", 0.5), 1.2);
  drawPath(fr, seed.X, alpha("--warm", 0.8), 1.4, 2.2);
  drawPath(fr, [T.xS, ...r.X], css("--ink"), 2.2, 3);
  act.forEach(([l, i]) => { ctx.fillStyle = css("--c5"); ctx.beginPath(); ctx.arc(X(data.S[i][0]), Y(data.S[i][1]), 3 + 7 * l, 0, 2 * Math.PI); ctx.fill(); });
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 2; seg(ctx, X(r.terminal[0]), Y(r.terminal[1]), X(xApp[0]), Y(xApp[1]));
  legend(ctx, ["orange: the stored seed run · shaded: its hull · black: the N-step LMPC plan", "purple: the stored states the terminal weights select · blue: the appended continuation"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#lmNV").textContent = N;
  caption($("#lmpcCap"), `[[N = ${N}]], the slow seed stored (${data.S.length} safe states). The first LMPC solve from [[x_S = (-3.5, 2.5)]] applies [[u_0 = ${fmt(r.U[0], 4)}]] where the seed applied 0, and its objective ${fmt(r.objective, 4)}, the ${N} stage costs plus [[\\lambda^\\top V]], is below the stored cost ${fmt(store.costs[0], 4)}, as the previous run is itself a feasible candidate. The terminal state [[x_N = (${fmt(r.terminal[0], 3)}, ${fmt(r.terminal[1], 3)})]] is ${act.length === 1 ? `exactly the stored state ${act[0][1]} of the seed, weight 1, with stored value ${fmt(data.V[act[0][1]], 3)}` : act.map(([l, i]) => `[[${fmt(l, 3)}]] of stored state ${i} (value ${fmt(data.V[i], 3)})`).join(" plus ")}. The shifted sequence appends [[u = \\sum_r \\lambda_r u_r = ${fmt(uApp, 4)}]], admissible, and lands at [[(${fmt(xApp[0], 4)}, ${fmt(xApp[1], 4)})]], a stored successor or a mixture of them, so the next problem is feasible: from [[x_1 = (${fmt(x1[0], 4)}, ${fmt(x1[1], 4)})]] the solver ${r1.success ? "succeeds" : "FAILS"} with objective ${fmt(r1.objective, 6)}, and [[J(x_0) - q(x_0, u_0) = ${fmt(r.objective - q0, 6)}]]: the decrease inequality holds with a slack of ${(r.objective - q0 - r1.objective).toExponential(1)}.`);
}
$("#lmN").addEventListener("input", () => { lm.N = +$("#lmN").value; drawLmpc(); });

function drawAll() { drawHero(); drawLmpc(); }
drawAll();
onRedraw(drawAll);
})();
