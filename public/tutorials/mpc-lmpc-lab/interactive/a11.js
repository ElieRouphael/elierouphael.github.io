/* Advanced MPC and Learning MPC, chapter A11, terminal ingredients and the stability theorem: the demos.
   The hero computes, for the canonical double integrator, the feasible regions of the zero-terminal and
   the terminal-set designs, the terminal set itself and the closed loop from a clickable start, with the
   full finite-horizon value on a log scale. The weaker motor |u| <= 0.5 is the default, where the horizon
   lever is visible. Other widgets: the areas against the horizon, and the LQR terminal region with its
   vertex checks. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { SYS, mv, mm, tr, add, sub, dot, quad, dlqr, makeMPC, closedLoop, closedLoopAdmissible, maximalInvariant, maximalControlInvariant, controllableSets, vertices, contains, area, zeroTarget, legend, drawPoly, clickPoint, fmtM, sci } = window.MPC;

const A = SYS.A, B = SYS.B, Q = SYS.Q, Rc = SYS.R, Xbox = SYS.X;
const RVALS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2];
const cache = {};
function sets(umax, R = Rc) {
  const key = `${umax}|${R}`; if (cache[key]) return cache[key];
  const L = R === Rc ? SYS.lqr : dlqr(A, B, Q, [[R]]), XK = closedLoopAdmissible(Xbox, L.K[0], -umax, umax), Xf = maximalInvariant(L.Acl, XK), C = maximalControlInvariant(A, B, Xbox, -umax, umax);
  return cache[key] = { L, XK, Xf: Xf.set, XfIters: Xf.history.length - 1, XfV: vertices(Xf.set), C: C.set, CV: vertices(C.set),
    zero: controllableSets(A, B, Xbox, zeroTarget(), -umax, umax, 12), term: controllableSets(A, B, Xbox, Xf.set, -umax, umax, 12) };
}

/* ---------------- hero ---------------- */
const hs = { design: "set", probe: [-3.5, -0.5], umax: 0.5, runs: {} };
function heroRun() {
  const N = +$("#hN").value, key = `${N}|${hs.umax}|${hs.design}|${hs.probe.map(v => v.toFixed(4)).join(",")}`; if (hs.runs[key]) return hs.runs[key];
  const s = sets(hs.umax), terminal = hs.design === "zero" ? { kind: "zero" } : hs.design === "set" ? { kind: "set", H: s.Xf.H, h: s.Xf.h } : { kind: "none" };
  const mpc = makeMPC(A, B, Q, Rc, N, hs.design === "zero" ? Q : s.L.P, SYS.xlb, SYS.xub, -hs.umax, hs.umax, terminal);
  return hs.runs[key] = closedLoop(mpc, hs.probe, 40, 1e-3);
}
function drawHero() {
  const N = +$("#hN").value, s = sets(hs.umax), run = heroRun(), zeroV = vertices(s.zero[N]), termV = vertices(s.term[N]);
  const cv = $("#planeC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.1 : 1.45, { x: [-4.6, 4.6], y: [-3.5, 3.5], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, -1.5, 0, 1.5, 3] }), { ctx, X, Y } = fr; cv._fr = fr;
  drawPoly(fr, vertices(Xbox), alpha("--ink", 0.04), alpha("--ink", 0.35), 1, [3, 3]);
  drawPoly(fr, s.CV, null, css("--warm"), 1.5, [5, 4]);
  drawPoly(fr, termV, alpha("--post", 0.12), css("--post"), 2);
  drawPoly(fr, zeroV, alpha("--c5", 0.18), css("--c5"), 1.8);
  drawPoly(fr, s.XfV, alpha("--post", 0.3), css("--post"), 1.2, [2, 3]);
  const p = hs.probe; ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.6; ctx.beginPath(); run.xs.forEach((x, i) => i ? ctx.lineTo(X(x[0]), Y(x[1])) : ctx.moveTo(X(x[0]), Y(x[1]))); ctx.stroke();
  ctx.fillStyle = css("--ink"); run.xs.forEach(x => { ctx.beginPath(); ctx.arc(X(x[0]), Y(x[1]), 2, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = run.infeasibleAt === 0 ? css("--warm") : css("--ink"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--bg"); ctx.lineWidth = 1.5; ctx.stroke();
  legend(ctx, ["purple: x_N = 0 region · blue: x_N ∈ X_f region", "dotted blue: X_f · dashed orange: C_∞ · black: the closed loop"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const cv2 = $("#valueC"), T = Math.max(10, run.vals.length), fr2 = frame(cv2, narrow ? 1.5 : 2.4, { x: [0, T], y: [-6, 2.2], xlabel: "closed-loop step t", yticks: [-6, -4, -2, 0, 2], yfmt: v => "10" + String(v).replace("-", "⁻").replace(/\d/g, ch => "⁰¹²³⁴⁵⁶⁷⁸⁹"[ch]) }), ctx2 = fr2.ctx;
  if (run.vals.length) { const lg = v => Math.max(-6, Math.log10(Math.max(v, 1e-300))); plotLine(fr2, run.vals.map((_, t) => t), run.vals.map(lg), { color: css("--post"), width: 2 }); ctx2.fillStyle = css("--post"); run.vals.forEach((v, t) => { ctx2.beginPath(); ctx2.arc(fr2.X(t), fr2.Y(lg(v)), 2.2, 0, 2 * Math.PI); ctx2.fill(); }); }
  legend(ctx2, ["J*_N(x(t)), the full finite-horizon value, log scale"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  $("#hNV").textContent = N; $$("#demo [data-design]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.design === hs.design))); $$("#demo [data-umax]").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.umax === hs.umax)));
  $("#rZero").textContent = fmt(area(zeroV), 2); $("#rTerm").textContent = fmt(area(termV), 2); $("#rXf").textContent = `${s.XfV.length} vertices, area ${fmt(area(s.XfV), 2)}`;
  const bad = run.resid.filter(v => v > 1e-7).length, maxRes = run.resid.length ? Math.max(...run.resid) : NaN, last = run.xs[run.xs.length - 1];
  const inZero = contains(s.zero[N], p, 1e-7), inTerm = contains(s.term[N], p, 1e-7), ref = N === 5 && hs.umax === 0.5 && Math.abs(p[0] + 3.5) < 1e-9 && Math.abs(p[1] + 0.5) < 1e-9;
  $("#rLoop").textContent = run.infeasibleAt === 0 ? "infeasible at the start" : run.infeasibleAt > 0 ? `infeasible at step ${run.infeasibleAt}` : `${run.us.length} steps, ${bad} of ${run.resid.length} residuals > 0`;
  let text = `[[|u| \\le ${fmt(hs.umax, 1)}]]${hs.umax === 0.5 ? " (the weaker motor)" : " (the full motor)"}, [[N = ${N}]], [[K = (${fmt(s.L.K[0][0], 4)}, ${fmt(s.L.K[0][1], 4)})]]${ref && hs.design === "set" ? ", the chapter's reference run" : ""}. The terminal set [[\\mathcal X_f]] (${s.XfV.length} vertices, ${s.XfIters} iterations) is the largest region where the LQR law stays within the input bound and inside the box for ever. From it, the [[N]]-step region [[x_N \\in \\mathcal X_f]] (blue) has area ${fmt(area(termV), 2)} against ${fmt(area(zeroV), 2)} for [[x_N = 0]] (purple) and ${fmt(area(s.CV), 2)} for the limit [[\\mathcal C_\\infty]]. `;
  text += `The start [[x_0 = (${fmt(p[0], 2)}, ${fmt(p[1], 2)})]] is ${inZero ? "inside" : "outside"} the zero-terminal region and ${inTerm ? "inside" : "outside"} the terminal-set region. `;
  const dname = hs.design === "zero" ? "design A, [[x_N = 0]]" : hs.design === "set" ? "design B, [[x_N \\in \\mathcal X_f]] with the Riccati cost [[P]]" : "no terminal constraint, Riccati cost [[P]] only";
  if (run.infeasibleAt === 0) text += `Under ${dname} the first problem is infeasible: no admissible ${N}-step plan meets the terminal requirement from here. The theorem says nothing from outside its feasible set.`;
  else if (run.infeasibleAt > 0) text += `Under ${dname} the loop ran ${run.infeasibleAt} steps and then found no feasible plan: feasible now did not mean feasible later. This cannot happen under designs A or B, whose shifted sequence is always a feasible plan.`;
  else text += `Under ${dname} the loop ${Math.hypot(...last) < 1e-3 ? `reaches [[\\lVert x \\rVert < 10^{-3}]] in ${run.us.length} steps` : `runs the ${run.us.length}-step cap and ends at [[(${fmt(last[0], 4)}, ${fmt(last[1], 4)})]]`}, with [[|u|]] saturated on ${run.us.filter(u => Math.abs(Math.abs(u) - hs.umax) < 1e-6).length} of them. The Lyapunov residual [[J_N^\\star(x_{t+1}) - J_N^\\star(x_t) + q(x_t, u_t)]] is at most ${sci(maxRes, 1)}: ${bad === 0 ? (hs.design === "none" ? "never positive beyond solver tolerance here, although no theorem guarantees it for this design" : "never positive beyond solver tolerance, as the theorem demands") : `positive on ${bad} of ${run.resid.length} steps, which the theorem forbids under designs A and B but does not rule out here`}. ${hs.design === "set" && bad === 0 ? "It is in fact nearly an equality: once the input leaves saturation the MPC coincides with the LQR, whose value drops by exactly [[q]] each step." : ""}`;
  caption($("#heroStatus"), text + " Tap or click the plane to move the start.");
}
$("#hN").addEventListener("input", drawHero);
$$("#demo [data-design]").forEach(bt => bt.addEventListener("click", () => { hs.design = bt.dataset.design; drawHero(); }));
$$("#demo [data-umax]").forEach(bt => bt.addEventListener("click", () => { hs.umax = +bt.dataset.umax; drawHero(); }));
$("#planeC").addEventListener("click", ev => { const p = clickPoint(ev); if (p) { hs.probe = [clamp(p[0], -4, 4), clamp(p[1], -3, 3)]; drawHero(); } });
$("#hReset").addEventListener("click", () => { $("#hN").value = 5; hs.design = "set"; hs.umax = 0.5; hs.probe = [-3.5, -0.5]; drawHero(); });

/* ---------------- Part 5: areas against the horizon ---------------- */
const ha = { umax: 0.5 };
function drawAreas() {
  const s = sets(ha.umax), zeroA = s.zero.map(S => area(vertices(S))), termA = s.term.map(S => area(vertices(S))), cA = area(s.CV);
  const cv = $("#areasC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, 13], y: [0, cA * 1.15], xlabel: "horizon N", xticks: [1, 3, 5, 7, 9, 11] }), { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--warm"); ctx.setLineDash([5, 4]); seg(ctx, fr.pad.l, Y(cA), fr.w - fr.pad.r, Y(cA)); ctx.setLineDash([]);
  const w = (X(1) - X(0)) * 0.36;
  for (let N = 0; N <= 12; N++) { ctx.fillStyle = alpha("--c5", 0.7); ctx.fillRect(X(N) - w, Y(zeroA[N]), w, Y(0) - Y(zeroA[N])); ctx.fillStyle = alpha("--post", 0.7); ctx.fillRect(X(N), Y(termA[N]), w, Y(0) - Y(termA[N])); }
  legend(ctx, ["purple: area of the x_N = 0 region · blue: x_N ∈ X_f region", "dashed: area of C_∞"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $$("#areasW [data-umax]").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.umax === ha.umax)));
  const fz = zeroA.findIndex(a => a > cA - 1e-2), ft = termA.findIndex(a => a > cA - 1e-2);
  caption($("#areasCap"), `[[|u| \\le ${fmt(ha.umax, 1)}]]: at [[N = 0]] the terminal-set design already accepts [[\\mathcal X_f]] itself (area ${fmt(termA[0], 2)}) while the zero design accepts only the origin. At [[N = 5]] the areas are ${fmt(zeroA[5], 2)} for [[x_N = 0]] and ${fmt(termA[5], 2)} for [[x_N \\in \\mathcal X_f]], ${fmt(100 * zeroA[5] / cA, 0)}% and ${fmt(100 * termA[5] / cA, 0)}% of [[\\mathcal C_\\infty]] (area ${fmt(cA, 2)}). The zero design fills [[\\mathcal C_\\infty]] at [[N = ${fz}]], the set design at [[N = ${ft}]]. ` + (ha.umax === 0.5 ? "With the weaker motor every region grows slowly in N, so the lever is visible for a dozen horizons." : "With the full motor the regions fill C_∞ within six or seven steps: on this system the horizon lever matters only for very short horizons, which is why the chapter's default is the weaker motor."));
}
$$("#areasW [data-umax]").forEach(bt => bt.addEventListener("click", () => { ha.umax = +bt.dataset.umax; drawAreas(); }));

/* ---------------- Part 4: the LQR terminal region and its vertex checks ---------------- */
const ht = { umax: 0.5 };
function drawTerminal() {
  const R = RVALS[+$("#tR").value], s = sets(ht.umax, R), L = s.L, K = L.K[0], V = s.XfV, XKV = vertices(s.XK);
  const cv = $("#termC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.15 : 1.5, { x: [-4.6, 4.6], y: [-3.5, 3.5], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, -1.5, 0, 1.5, 3] }), { ctx, X, Y } = fr;
  drawPoly(fr, vertices(Xbox), alpha("--ink", 0.04), alpha("--ink", 0.35), 1, [3, 3]); drawPoly(fr, XKV, alpha("--c5", 0.12), css("--c5"), 1.5, [5, 4]); drawPoly(fr, V, alpha("--post", 0.3), css("--post"), 2);
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; V.forEach(v => { let x = v; ctx.beginPath(); ctx.moveTo(X(x[0]), Y(x[1])); for (let k = 0; k < 3; k++) { x = mv(L.Acl, x); ctx.lineTo(X(x[0]), Y(x[1])); } ctx.stroke(); });
  legend(ctx, ["dashed purple: X_K · blue: X_f", "black: three LQR steps from each vertex"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const rows = V.map(v => { const u = -dot(K, v), vn = mv(L.Acl, v); return { v, u, inside: contains(s.Xf, vn, 1e-7), res: quad(L.P, vn) - quad(L.P, v) + quad(Q, v) + R * u * u }; });
  $("#termT").innerHTML = "<thead><tr><th>vertex</th><th>u = −Kv</th><th>next in X<sub>f</sub></th><th>V<sub>f</sub>(x⁺) − V<sub>f</sub>(x) + q</th></tr></thead><tbody>" + rows.map(r => `<tr><td>(${fmt(r.v[0], 3)}, ${fmt(r.v[1], 3)})</td><td>${fmt(r.u, 4)}</td><td class="words">${r.inside ? "yes" : "no"}</td><td>${r.res.toExponential(1)}</td></tr>`).join("") + "</tbody>";
  $("#tRV").textContent = R; $$("#termW [data-umax]").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.umax === ht.umax)));
  const rho = Math.hypot(...L.eig[0]), ref = R === 0.2, resid = Math.max(...sub(add(mm(tr(L.Acl), mm(L.P, L.Acl)), add(Q, mm(tr(L.K), mm([[R]], L.K)))), L.P).flat().map(Math.abs));
  caption($("#termCap"), `[[R = ${R}]]${ref ? ", the canonical weight" : ""}, [[|u| \\le ${fmt(ht.umax, 1)}]]: [[K = (${fmt(K[0], 4)}, ${fmt(K[1], 4)})]], [[P = ${fmtM(L.P, 3)}]], [[\\rho(A_{\\mathrm{cl}}) = ${fmt(rho, 4)}]], DARE identity residual ${sci(resid, 1)}. The LQR input is admissible on [[\\mathcal X_K]] (${XKV.length} vertices, area ${fmt(area(XKV), 2)}); the fixed point [[\\Omega_{j+1} = \\mathcal X_K \\cap A_{\\mathrm{cl}}^{-1}(\\Omega_j)]] stops after ${s.XfIters} iterations with [[\\mathcal X_f]] of ${V.length} vertices and area ${fmt(area(V), 2)}. The table checks the three conditions at every vertex: the state is in the box, the input is within the bound (the largest is ${fmt(Math.max(...rows.map(r => Math.abs(r.u))), 4)}), the next state stays in [[\\mathcal X_f]], and the terminal decrease [[V_f(A_{\\mathrm{cl}} x) - V_f(x) + q(x, -Kx)]] is zero to ${sci(Math.max(...rows.map(r => Math.abs(r.res))), 0)}: equality, as the DARE identity promises. ` + (R <= 0.05 ? "A cheap input gives an aggressive gain, which saturates sooner: the admissible strip is narrow and the terminal set small." : R >= 1 ? "An expensive input gives a gentle gain that rarely saturates: the terminal set is large, but the loop inside it converges slowly." : "Exercise 7 trades these off: smaller R, smaller terminal set; larger R, larger set but a slower local loop."));
}
$("#tR").addEventListener("input", drawTerminal);
$$("#termW [data-umax]").forEach(bt => bt.addEventListener("click", () => { ht.umax = +bt.dataset.umax; drawTerminal(); }));

function drawAll() { drawHero(); drawAreas(); drawTerminal(); }
drawAll();
onRedraw(drawAll);
})();
