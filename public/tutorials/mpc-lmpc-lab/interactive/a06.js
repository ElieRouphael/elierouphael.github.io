/* Advanced MPC and Learning MPC, chapter A6, receding horizon: the demos.
   The hero runs the receding-horizon loop of the canonical double integrator in the browser, under three
   terminal designs, and shows the plan made at any step against what was applied. A second widget shows
   the horizon-one pathology on a scalar plant. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, fit, clamp } = window.Kit;
const { SYS, mv, quad, dlqr, makeMPC, closedLoopAdmissible, maximalInvariant, legend, norm } = window.MPC;

const A = SYS.A, B = SYS.B, Q = SYS.Q, R = SYS.R;
const hs = { design: "PQ", cache: {} };

/* the receding-horizon loop, keeping every plan */
function recede(mpc, x0, steps, tol) {
  let x = x0.slice(); const xs = [x.slice()], us = [], vals = [], plans = [], stages = []; let infeasibleAt = -1;
  for (let t = 0; t < steps; t++) {
    const sol = mpc.solve(x); if (!sol.feasible) { infeasibleAt = t; break; }
    vals.push(sol.J); plans.push(sol.X); const u = sol.U[0]; stages.push(quad(Q, x) + R * u * u); us.push(u);
    x = mv(A, x).map((v, i) => v + B[i][0] * u); xs.push(x.slice());
    if (norm(x) < tol) break;
  }
  const resid = vals.slice(1).map((v, t) => v - vals[t] + stages[t]);
  return { xs, us, vals, plans, stages, resid, infeasibleAt };
}
function heroRun() {
  const N = +$("#hN").value, umax = +$("#hU").value, p0 = +$("#hP0").value, v0 = +$("#hV0").value, key = `${N}|${umax}|${p0}|${v0}|${hs.design}`;
  if (hs.cache[key]) return hs.cache[key];
  let P = Q, terminal = { kind: "none" }, Xf = null;
  if (hs.design !== "PQ") {
    P = SYS.lqr.P;
    if (hs.design === "set") { Xf = maximalInvariant(SYS.lqr.Acl, closedLoopAdmissible(SYS.X, SYS.lqr.K[0], -umax, umax)).set; terminal = { kind: "set", H: Xf.H, h: Xf.h }; }
  }
  const mpc = makeMPC(A, B, Q, R, N, P, SYS.xlb, SYS.xub, -umax, umax, terminal);
  const run = recede(mpc, [p0, v0], 25, 1e-3);
  return hs.cache[key] = Object.assign(run, { N, umax, p0, v0, Xf, closedLoopCost: run.stages.reduce((s, v) => s + v, 0) });
}
function drawHero() {
  const r = heroRun(), cv = $("#mpcC"), narrow = narrowOf(cv), T = Math.max(12, r.xs.length), xs = r.xs, w = cv.clientWidth || cv.parentElement.clientWidth;
  const h = narrow ? 420 : 360, { ctx } = fit(cv, w / h), padL = 44, padR = 10, ph = (h - 40) / 3, X = t => padL + t / (T - 1) * (w - padL - padR);
  const panel = (k, y0, y1, ticks, label) => { const top = 8 + k * ph, bot = top + ph - 26, Y = v => bot - (v - y0) / (y1 - y0) * (bot - top); ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle"; ctx.textAlign = "right"; for (const t of ticks) { ctx.strokeStyle = css("--grid"); seg(ctx, padL, Y(t), w - padR, Y(t)); ctx.fillStyle = css("--muted"); ctx.fillText(String(t), padL - 5, Y(t)); } ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillText(label, padL + 4, top + 10); return { Y, top, bot }; };
  const P1 = panel(0, -5.4, 5.4, [-4, 0, 4], "position p, bound ±4"), P2 = panel(1, -4.2, 4.2, [-3, 0, 3], "velocity v, bound ±3"), P3 = panel(2, -r.umax * 1.5, r.umax * 1.5, [-r.umax, 0, r.umax].map(v => +v.toFixed(2)), "input u, bound ±" + fmt(r.umax, 2));
  const line = (pts, Y, token, width = 2) => { ctx.strokeStyle = css(token); ctx.lineWidth = width; ctx.beginPath(); pts.forEach((v, t) => t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v))); ctx.stroke(); ctx.fillStyle = css(token); pts.forEach((v, t) => { ctx.beginPath(); ctx.arc(X(t), Y(v), 2.6, 0, 2 * Math.PI); ctx.fill(); }); };
  for (const [Pn, v] of [[P1, 4], [P1, -4], [P2, 3], [P2, -3], [P3, r.umax], [P3, -r.umax]]) { ctx.strokeStyle = alpha("--warm", 0.7); ctx.setLineDash([5, 4]); seg(ctx, padL, Pn.Y(v), w - padR, Pn.Y(v)); ctx.setLineDash([]); }
  const tp = clamp(+$("#hStep").value, 0, Math.max(0, r.plans.length - 1)); $("#hStep").max = Math.max(0, r.plans.length - 1);
  line(xs.map(x => x[0]), P1.Y, "--post"); line(xs.map(x => x[1]), P2.Y, "--post");
  let planGap = 0;
  if (r.plans[tp]) { const plan = [r.xs[tp], ...r.plans[tp]]; plan.forEach((x, i) => { const act = r.xs[tp + i]; if (act) planGap = Math.max(planGap, Math.abs(x[0] - act[0]), Math.abs(x[1] - act[1])); }); ctx.strokeStyle = css("--warm"); ctx.setLineDash([3, 3]); ctx.lineWidth = 2; [[0, P1], [1, P2]].forEach(([k, Pn]) => { ctx.beginPath(); plan.forEach((x, i) => i ? ctx.lineTo(X(tp + i), Pn.Y(x[k])) : ctx.moveTo(X(tp), Pn.Y(x[k]))); ctx.stroke(); }); ctx.setLineDash([]); }
  ctx.strokeStyle = css("--c5"); ctx.lineWidth = 2; ctx.beginPath(); r.us.forEach((u, t) => { if (t === 0) ctx.moveTo(X(0), P3.Y(u)); else { ctx.lineTo(X(t), P3.Y(r.us[t - 1])); ctx.lineTo(X(t), P3.Y(u)); } ctx.lineTo(X(t + 1), P3.Y(u)); }); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; for (const t of [0, 5, 10, 15, 20, 25]) if (t < T) ctx.fillText(String(t), X(t), h - 6);
  $("#hNV").textContent = r.N; $("#hUV").textContent = fmt(r.umax, 2); $("#hP0V").textContent = fmt(r.p0, 1); $("#hV0V").textContent = fmt(r.v0, 1); $("#hStepV").textContent = tp;
  $$("#demo [data-design]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.design === hs.design)));
  const last = r.xs[r.xs.length - 1], infeasible = r.infeasibleAt >= 0, nb = r.N === 5 && r.umax === 1 && r.p0 === -3.6 && r.v0 === 2.4;
  $("#rMoves").textContent = infeasible ? (r.infeasibleAt === 0 ? "infeasible" : `${r.us.length}, then infeasible`) : String(r.us.length);
  $("#rFinal").textContent = infeasible && r.infeasibleAt === 0 ? "–" : `(${fmt(last[0], 3)}, ${fmt(last[1], 3)})`;
  $("#rCost").textContent = r.us.length ? fmt(r.closedLoopCost, 3) : "–";
  const bad = r.resid.filter(v => v > 1e-7).length; $("#rLyap").textContent = r.resid.length ? `${bad} of ${r.resid.length}` : "–";
  let text;
  if (infeasible && r.us.length === 0) text = `No admissible [[N = ${r.N}]]-step plan exists from [[(${fmt(r.p0, 1)}, ${fmt(r.v0, 1)})]]: the first problem is infeasible and the loop never starts. A state can satisfy the constraints and still have no plan that keeps satisfying them, which is what [[#properties|Part 3]] calls feasibility. `;
  else if (infeasible) text = `From [[(${fmt(r.p0, 1)}, ${fmt(r.v0, 1)})]] the problem was feasible for ${r.us.length} step${r.us.length === 1 ? "" : "s"}, with states ${r.xs.map(x => `(${fmt(x[0], 2)}, ${fmt(x[1], 2)})`).join(", ")}, and then no admissible plan existed: feasible now did not mean feasible later. This loss of recursive feasibility is what the terminal set rules out, at the price of refusing the start at once. `;
  else text = `${r.us.length} moves${nb ? ", the chapter's reference run" : ""}: the loop stops when [[\\lVert x \\rVert < 10^{-3}]], here at [[(${fmt(last[0], 4)}, ${fmt(last[1], 4)})]]. The velocity peaks at ${fmt(Math.max(...xs.map(x => Math.abs(x[1]))), 2)} and the input sits on its bound for ${r.us.filter(u => Math.abs(Math.abs(u) - r.umax) < 1e-6).length} of the moves; the closed-loop cost [[\\sum_t q(x_t, u_t)]] is ${fmt(r.closedLoopCost, 3)}. `;
  if (!infeasible || r.us.length) {
    if (r.plans[tp]) text += `The dashed orange curve is the plan made at step ${tp}: ${r.N} predicted states, of which only the first move is applied before the problem is solved again. ${planGap < 5e-4 ? "It coincides with what was then applied: the model is exact and nothing disturbs the cart, so re-solving changes nothing." : planGap < 0.05 ? `It differs from what was then applied by at most ${planGap.toFixed(3)}: with an exact model, re-solving changes the plan only where the end of the horizon moved.` : `It differs from what was then applied by up to ${planGap.toFixed(2)}: the short horizon made the plan end where the next solve, seeing one step further, chose differently.`} `;
    text += hs.design === "PQ" ? `With [[V_f = x^\\top Q x]] and no terminal set, the optimal value decreased by at least the stage cost on ${r.resid.length - bad} of ${r.resid.length} steps; no theorem is in force, and whatever convergence is seen is a property of this run, not a guarantee.` : hs.design === "P" ? `With the LQR matrix [[P]] as terminal cost but no terminal set, the value decreased by at least the stage cost on ${r.resid.length - bad} of ${r.resid.length} steps: the terminal controller may violate the input bound beyond the horizon, so the proof of chapter [[a11]] does not apply.` : `With the LQR terminal cost and the invariant terminal set [[\\mathcal X_f]] (${r.Xf.h.length} inequalities), the value [[J_N^\\star(x_t)]] decreased by at least [[q(x_t, u_t)]] on every one of the ${r.resid.length} steps, as chapter [[a11]] proves it must${bad ? ` (${bad} exceptions within solver tolerance)` : ""}.`;
  }
  caption($("#heroStatus"), text);
}
["#hN", "#hU", "#hP0", "#hV0", "#hStep"].forEach(id => $(id).addEventListener("input", drawHero));
$$("#demo [data-design]").forEach(bt => bt.addEventListener("click", () => { hs.design = bt.dataset.design; drawHero(); }));
$("#hReset").addEventListener("click", () => { $("#hN").value = 5; $("#hU").value = 1; $("#hP0").value = -3.6; $("#hV0").value = 2.4; $("#hStep").value = 0; hs.design = "PQ"; drawHero(); });
$("#hLose").addEventListener("click", () => { $("#hN").value = 1; $("#hU").value = 0.5; $("#hP0").value = -4; $("#hV0").value = 2; $("#hStep").value = 0; hs.design = "PQ"; drawHero(); });

/* Part 5: the scalar horizon-one pathology */
function drawScalar() {
  const a = +$("#sA").value, r = +$("#sR").value, L = dlqr([[a]], [[1]], [[1]], [[r]]), P = L.P[0][0], K = L.K[0][0], lam = a - K;
  const steps = 15; let x1 = 1, x2 = 1; const xs1 = [x1], xs2 = [x2]; for (let t = 0; t < steps; t++) { x1 = a * x1; xs1.push(x1); x2 = a * x2 - K * x2; xs2.push(x2); }
  const cv = $("#scalarC"), narrow = narrowOf(cv), ymax = Math.min(20, Math.max(2, Math.max(...xs1))), fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, steps], y: [-0.5, ymax], xlabel: "closed-loop step", xticks: [0, 5, 10, 15] }), { ctx } = fr;
  plotLine(fr, xs1.map((_, t) => t), xs1, { color: css("--warm"), width: 2 }); plotLine(fr, xs2.map((_, t) => t), xs2, { color: css("--post"), width: 2 });
  ctx.strokeStyle = alpha("--ink", 0.3); seg(ctx, fr.pad.l, fr.Y(0), fr.w - fr.pad.r, fr.Y(0));
  legend(ctx, ["orange: no terminal cost, u = 0", "blue: with the Riccati terminal cost"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#sAV").textContent = fmt(a, 2); $("#sRV").textContent = fmt(r, 1);
  const ref = a === 1.2 && r === 4;
  caption($("#scalarCap"), `[[x^+ = ${fmt(a, 2)}x + u]], [[q = x^2 + ${fmt(r, 1)}u^2]]${ref ? ", the chapter's numbers" : ""}. With [[N = 1]] and no terminal cost the objective [[x_0^2 + ru_0^2]] does not contain [[x_1]], so [[u_0^\\star = 0]] and the loop is [[x^+ = ${fmt(a, 2)}x]]: ${a > 1 ? `unstable, ${fmt(xs1[steps], 1)} after ${steps} steps` : a === 1 ? "marginally stable, the state never moves" : "stable only because the plant already was"}. Adding the Riccati terminal cost [[Px_1^2]] with [[P = ${fmt(P, 4)}]] gives [[u^\\star = -\\frac{Pa}{r + P}x = -${fmt(K, 4)}x]], the scalar LQR law of chapter [[a07]], and the closed-loop eigenvalue [[a - K = ${fmt(lam, 4)}]]${Math.abs(lam) < 1 ? ": stable" : ""}. One number, [[P]], carries the whole infinite future into a one-step problem. ` + (r > 15 ? "A heavy input penalty makes the repair timid: the eigenvalue creeps towards 1 but stays below it, since the DARE always yields a stabilising gain." : a >= 1.5 ? "A more unstable plant needs a larger P and a larger gain; the Riccati solution adapts automatically." : "Exercise 3 asks for the same with a = 1.1 and r = 5: the eigenvalue without terminal cost is 1.1, and with it 0.648."));
}
["#sA", "#sR"].forEach(id => $(id).addEventListener("input", drawScalar));

function drawAll() { drawHero(); drawScalar(); }
drawAll();
onRedraw(drawAll);
})();
