/* Advanced MPC and Learning MPC, chapter C1, tube MPC for a bounded disturbance: the demos.
   Everything is computed here with the shared toolkit: the error set as a scaled Minkowski sum, the
   tightened box and input bound, the nominal MPC of chapter A11 on the tightened sets, and the
   disturbed closed loop with the tube feedback (or without it). Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { SYS, mv, dot, sub, mm, matpow, box, poly, vertices, fromVertices, minkowski, affineImage, contains, closedLoopAdmissible,
        maximalInvariant, maximalControlInvariant, controllableSets, makeMPC, area, legend, drawPoly, clickPoint, drawPath, norm } = window.MPC;

const { A, B, Q, R, X } = SYS, L = SYS.lqr, K = L.K[0], AK = L.Acl, N = 5;
const WMAXS = [0.05, 0.1, 0.15];
const supportOf = (V, row) => Math.max(...V.map(v => dot(row, v)));

/* the error set, the tightened sets and the two controllers, for a disturbance bound */
const cache = {};
function design(wmax) {
  if (cache[wmax]) return cache[wmax];
  const W = box([-wmax, -wmax], [wmax, wmax]), Fs = [W]; let F = W;
  for (let s = 2; s <= 8; s++) { F = minkowski(F, affineImage(W, matpow(AK, s - 1))); Fs.push(F); }
  const AK8 = matpow(AK, 8), al = Math.max(...vertices(W).map(v => Math.max(...mv(AK8, v).map(Math.abs)) / wmax));
  const EV = vertices(F).map(v => v.map(c => c / (1 - al))), E = fromVertices(EV);
  const hx = X.h.map((h, i) => h - supportOf(EV, X.H[i])), ku = supportOf(EV, K), umaxT = 1 - ku;
  const Xt = poly(X.H, hx), xlbT = [-hx[2], -hx[3]], xubT = [hx[0], hx[1]];
  const XKt = closedLoopAdmissible(Xt, K, -umaxT, umaxT), Xft = maximalInvariant(AK, XKt).set, Ct = maximalControlInvariant(A, B, Xt, -umaxT, umaxT).set;
  const regT = controllableSets(A, B, Xt, Xft, -umaxT, umaxT, N)[N];
  const XK = closedLoopAdmissible(X, K, -1, 1), Xf = maximalInvariant(AK, XK).set, regN = controllableSets(A, B, X, Xf, -1, 1, N)[N];
  const mpcT = makeMPC(A, B, Q, R, N, L.P, xlbT, xubT, -umaxT, umaxT, { kind: "set", H: Xft.H, h: Xft.h });
  const mpcN = makeMPC(A, B, Q, R, N, L.P, SYS.xlb, SYS.xub, -1, 1, { kind: "set", H: Xf.H, h: Xf.h });
  return cache[wmax] = { wmax, W, Fs, al, E, EV, hx, ku, umaxT, Xt, Xft, Ct, regT, regN, mpcT, mpcN };
}
/* a seeded uniform generator (mulberry32), so that a seed reproduces its disturbance sequence */
function rng(seed) { let a = (seed * 2654435761 + 1013904223) >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function runLoop(d, mode, x0, seed, steps = 40) {
  const r = rng(seed); let x = x0.slice(), z = x0.slice(); const xs = [x.slice()], zs = [z.slice()], us = []; let infeasibleAt = -1;
  for (let t = 0; t < steps; t++) {
    const sol = mode === "tube" ? d.mpcT.solve(z) : d.mpcN.solve(x);
    if (!sol.feasible) { infeasibleAt = t; break; }
    const v = sol.U[0], u = mode === "tube" ? v - dot(K, [x[0] - z[0], x[1] - z[1]]) : v;
    const w = [d.wmax * (2 * r() - 1), d.wmax * (2 * r() - 1)];
    x = mv(A, x).map((c, i) => c + B[i][0] * u + w[i]); z = mode === "tube" ? mv(A, z).map((c, i) => c + B[i][0] * v) : x.slice();
    xs.push(x.slice()); zs.push(z.slice()); us.push(u);
  }
  const viol = xs.filter(p => !contains(X, p, 1e-9)).length, errs = xs.map((p, t) => [p[0] - zs[t][0], p[1] - zs[t][1]]);
  return { xs, zs, us, infeasibleAt, viol, errs, inE: errs.every(e => contains(d.E, e, 1e-9)) };
}

const hs = { mode: "tube", x0: [-3, 2] };
function drawHero() {
  const d = design(WMAXS[+$("#hW").value]), seed = +$("#hSeed").value, run = runLoop(d, hs.mode, hs.x0, seed);
  const cv = $("#tubeC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.1 : 1.45, { x: [-4.6, 4.6], y: [-3.5, 3.5], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, 0, 3] }), { ctx, X: Xp, Y } = fr; cv._fr = fr;
  drawPoly(fr, vertices(X), alpha("--ink", 0.03), alpha("--ink", 0.35), 1, [3, 3]);
  if (hs.mode === "tube") { drawPoly(fr, vertices(d.Xt), null, css("--warm"), 1.2, [6, 4]); drawPoly(fr, vertices(d.regT), alpha("--post", 0.08), alpha("--post", 0.6), 1.2); }
  else drawPoly(fr, vertices(d.regN), alpha("--c5", 0.08), alpha("--c5", 0.6), 1.2);
  if (hs.mode === "tube") run.zs.forEach(z => drawPoly(fr, d.EV.map(e => [e[0] + z[0], e[1] + z[1]]), alpha("--post", 0.1), alpha("--post", 0.25), 0.8));
  if (hs.mode === "tube") drawPath(fr, run.zs, css("--post"), 1.4, 2);
  drawPath(fr, run.xs, css("--ink"), 1.6, 2.2);
  ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(Xp(hs.x0[0]), Y(hs.x0[1]), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--bg"); ctx.lineWidth = 1.5; ctx.stroke();
  if (run.infeasibleAt >= 0) { const p = run.xs[run.xs.length - 1]; ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2; seg(ctx, Xp(p[0]) - 6, Y(p[1]) - 6, Xp(p[0]) + 6, Y(p[1]) + 6); seg(ctx, Xp(p[0]) - 6, Y(p[1]) + 6, Xp(p[0]) + 6, Y(p[1]) - 6); }
  legend(ctx, hs.mode === "tube" ? ["dotted: the box · dashed orange: the tightened box · blue region: the tube controller's region", "blue: nominal z with the tube z ⊕ E · black: the real state x · orange: the start"]
                                 : ["dotted: the box · purple region: chapter A11's region", "black: the real state under the nominal controller · orange: the start · cross: no feasible plan"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* time plot */
  const cv2 = $("#timeC"), T = run.xs.length - 1, fr2 = frame(cv2, narrow ? 1.5 : 2.4, { x: [0, Math.max(T, 10)], y: [-4.5, 4.5], xlabel: "step t", yticks: [-4, -2, 0, 2, 4] }), c2 = fr2.ctx;
  c2.strokeStyle = alpha("--ink", 0.3); c2.setLineDash([3, 3]); [4, -4].forEach(b => seg(c2, fr2.pad.l, fr2.Y(b), fr2.w - fr2.pad.r, fr2.Y(b))); c2.strokeStyle = alpha("--warm", 0.5); [3, -3].forEach(b => seg(c2, fr2.pad.l, fr2.Y(b), fr2.w - fr2.pad.r, fr2.Y(b))); c2.setLineDash([]);
  if (hs.mode === "tube") { c2.strokeStyle = alpha("--post", 0.35); c2.setLineDash([6, 4]); [d.hx[0], -d.hx[2]].forEach(b => seg(c2, fr2.pad.l, fr2.Y(b), fr2.w - fr2.pad.r, fr2.Y(b))); c2.setLineDash([]); }
  const ts = run.xs.map((_, t) => t);
  plotLine(fr2, ts, run.xs.map(p => p[0]), { color: css("--post"), width: 2 }); plotLine(fr2, ts, run.xs.map(p => p[1]), { color: css("--warm"), width: 2 });
  if (hs.mode === "tube") plotLine(fr2, ts, run.zs.map(p => p[0]), { color: css("--post"), width: 1, dash: [4, 3] });
  legend(c2, ["blue: p (dashed: nominal) · orange: v", "dotted: the box bounds · dashed blue: the tightened position bound"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  /* readouts and caption */
  $$("#demo [data-mode]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.mode === hs.mode)));
  $("#hWV").textContent = fmt(d.wmax, 2); $("#hSeedV").textContent = seed;
  $("#rE").textContent = `${d.EV.length} vertices, area ${fmt(area(d.EV), 3)}`;
  $("#rViol").textContent = run.infeasibleAt >= 0 ? `${run.viol}, then no plan at step ${run.infeasibleAt}` : String(run.viol);
  $("#rU").textContent = run.us.length ? fmt(Math.max(...run.us.map(Math.abs)), 3) : "–";
  const last = run.xs[run.xs.length - 1]; $("#rFinal").textContent = fmt(norm(last), 3);
  const inReg = contains(d.regT, hs.x0, 1e-7), inRegN = contains(d.regN, hs.x0, 1e-7), ref = d.wmax === 0.1 && seed === 0 && hs.x0[0] === -3 && hs.x0[1] === 2;
  let text = `[[|w_i| \\le ${fmt(d.wmax, 2)}]], seed ${seed}, start [[(${fmt(hs.x0[0], 2)}, ${fmt(hs.x0[1], 2)})]]${ref && hs.mode === "tube" ? ", the chapter's run" : ""}. `;
  if (hs.mode === "tube") {
    text += `The error set [[\\mathcal E = \\mathcal F_8 / (1 - \\alpha)]] has area ${fmt(area(d.EV), 3)} ([[\\alpha = ${d.al.toExponential(1)}]]); it tightens the box to [[${fmt(d.hx[0], 3)} \\times ${fmt(d.hx[1], 3)}]] and the nominal input to [[|v| \\le ${fmt(d.umaxT, 3)}]], and the tube controller's five-step region has area ${fmt(area(vertices(d.regT)), 2)} against ${fmt(area(vertices(d.regN)), 2)} for chapter A11's. `;
    if (run.infeasibleAt === 0) text += `The start is outside that region, so the nominal problem is infeasible at once: the tube controller refuses a start it cannot keep a promise for.`;
    else text += `The nominal state (blue) is planned without disturbance and reaches the origin; the real state (black) stays inside the tube [[z_t \\oplus \\mathcal E]] at every step (${run.inE ? "checked" : "VIOLATED"}), leaves the box ${run.viol} times, uses at most [[|u| = ${fmt(Math.max(...run.us.map(Math.abs)), 3)}]] of the bound 1, and ends at [[\\lVert x_{${T}} \\rVert = ${fmt(norm(last), 3)}]], inside [[\\mathcal E]] around the origin, where the disturbance keeps it moving.`;
  } else {
    text += `Chapter A11's controller re-solved from the measured state, full box and full input, no tube. ${inRegN ? "The start is in its region" : "The start is outside its region"}${inReg ? " and in the tube controller's" : " and outside the tube controller's"}. `;
    if (run.infeasibleAt >= 0) text += `It survives ${run.infeasibleAt} steps and then finds no feasible plan${run.viol ? `, after leaving the box` : ""}: feasible now did not mean feasible later once the plant moved where the plan did not expect.`;
    else text += `It survives this sequence, leaving the box ${run.viol} times and saturating at [[|u| = ${fmt(Math.max(...run.us.map(Math.abs)), 3)}]]: inherent robustness, which holds for most starts and fails on the boundary of its region (try [[(3, 1.5)]] with seed 1).`;
  }
  caption($("#heroStatus"), text);
}
["#hW", "#hSeed"].forEach(id => $(id).addEventListener("input", drawHero));
$$("#demo [data-mode]").forEach(b => b.addEventListener("click", () => { hs.mode = b.dataset.mode; drawHero(); }));
$("#tubeC").addEventListener("click", ev => { const p = clickPoint(ev); if (p) { hs.x0 = [clamp(Math.round(p[0] * 4) / 4, -4, 4), clamp(Math.round(p[1] * 4) / 4, -3, 3)]; drawHero(); } });
$("#hReset").addEventListener("click", () => { hs.mode = "tube"; hs.x0 = [-3, 2]; $("#hW").value = 1; $("#hSeed").value = 0; drawHero(); });

/* Part 3: the partial Minkowski sums */
function drawRpi() {
  const d = design(0.1), s = +$("#rpiS").value, F = d.Fs[s - 1], V = vertices(F);
  const cv = $("#rpiC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.2 : 1.9, { x: [-0.3, 0.3], y: [-0.32, 0.32], xlabel: "e₁ (position error)", ylabel: "e₂ (velocity error)", xticks: [-0.2, 0, 0.2], yticks: [-0.2, 0, 0.2] });
  drawPoly(fr, d.EV, alpha("--post", 0.08), alpha("--post", 0.4), 1, [4, 3]);
  for (let k = 1; k < s; k++) drawPoly(fr, vertices(d.Fs[k - 1]), null, alpha("--ink", 0.15), 0.8);
  drawPoly(fr, V, alpha("--post", 0.25), css("--post"), 2);
  drawPoly(fr, vertices(d.W), alpha("--warm", 0.2), css("--warm"), 1.5);
  legend(fr.ctx, ["orange: the disturbance box W · blue: F_s · grey: F_1 … F_{s−1}", "dashed: E = F_8 / (1 − α)"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#rpiSV").textContent = s;
  const extra = s === 1 ? " is the disturbance box itself: one step from zero" : s === 2 ? ": the box plus its image under [[A_K]], a shrunk and tilted copy" : s >= 8 ? ", and adding more terms changes the area by less than [[10^{-4}]]: the sum has converged to four decimals" : "";
  caption($("#rpiCap"), `[[\\mathcal F_{${s}}]] has ${V.length} vertices and area ${fmt(area(V), 4)}${extra}. Each term [[A_K^{k} \\mathcal W]] is smaller than the last by about [[\\rho(A_K)^k]], so the sum converges geometrically; the dashed set is the outer approximation [[\\mathcal E]] used by the controller, area ${fmt(area(d.EV), 4)}.`);
}
$("#rpiS").addEventListener("input", drawRpi);

function drawAll() { drawHero(); drawRpi(); }
drawAll();
onRedraw(drawAll);
})();
