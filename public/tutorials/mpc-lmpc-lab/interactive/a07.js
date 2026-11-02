/* Advanced MPC and Learning MPC, chapter A7, Lyapunov stability and LQR: the demos.
   The hero runs the LQR closed loop of the canonical double integrator for any weights, with the value
   x'Px falling by exactly the stage cost at every step. A gain explorer solves the Lyapunov equation for a
   hand-chosen feedback, and a third widget runs the Riccati recursion from a chosen terminal weight.
   Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { SYS, mv, mm, tr, add, sub, quad, dlqr, eig2, eigsym2, solve, riccatiRecursion, legend, drawPoly, vertices, drawPath, fmtM, sci } = window.MPC;

const A = SYS.A, B = SYS.B, X0 = [-3.6, 2.4];
const maxAbs = M => Math.max(...M.flat().map(Math.abs));
const logTicks = v => "10" + String(v).replace("-", "⁻").replace(/\d/g, ch => "⁰¹²³⁴⁵⁶⁷⁸⁹"[ch]);
const RV = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2], QV = [0.05, 0.1, 0.2, 0.5, 1];

/* ---------------- hero: the LQR loop ---------------- */
function drawHero() {
  const q2 = QV[+$("#hQ").value], r = RV[+$("#hR").value], Q = [[1, 0], [0, q2]], R = [[r]], L = dlqr(A, B, Q, R);
  const resid = maxAbs(add(sub(mm(tr(L.Acl), mm(L.P, L.Acl)), L.P), add(Q, mm(tr(L.K), mm(R, L.K)))));
  let x = X0.slice(); const Vs = [], qs = [], xs = [], us = [];
  for (let k = 0; k < 14; k++) { const u = -(L.K[0][0] * x[0] + L.K[0][1] * x[1]); Vs.push(quad(L.P, x)); qs.push(quad(Q, x) + r * u * u); xs.push(x); us.push(u); x = mv(L.Acl, x); }
  const cv = $("#lqrC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.4, { x: [0, 13], y: [-12, 2], xlabel: "step k", yticks: [-12, -8, -4, 0], yfmt: logTicks }), { ctx } = fr;
  plotLine(fr, Vs.map((_, k) => k), Vs.map(v => Math.max(-12, Math.log10(Math.max(v, 1e-300)))), { color: css("--post"), width: 2.2 });
  plotLine(fr, qs.map((_, k) => k), qs.map(v => Math.max(-12, Math.log10(Math.max(v, 1e-300)))), { color: css("--warm"), width: 1.6, dash: [4, 3] });
  ctx.fillStyle = css("--post"); Vs.forEach((v, k) => { ctx.beginPath(); ctx.arc(fr.X(k), fr.Y(Math.max(-12, Math.log10(Math.max(v, 1e-300)))), 2.5, 0, 2 * Math.PI); ctx.fill(); });
  legend(ctx, ["blue: V(x_k) = x_k' P x_k, log scale · dashed orange: q(x_k, u_k)", "V(x_{k+1}) = V(x_k) − q(x_k, u_k) exactly"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const cv2 = $("#planeC"), fr2 = frame(cv2, narrow ? 1.2 : 1.6, { x: [-5.5, 5.5], y: [-4, 4], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, 0, 3] });
  drawPoly(fr2, vertices(SYS.X), alpha("--ink", 0.04), alpha("--ink", 0.35), 1, [3, 3]);
  /* level sets of V through the first three states */
  const level = (c, color) => { const pts = []; for (let i = 0; i <= 120; i++) { const th = 2 * Math.PI * i / 120, d = [Math.cos(th), Math.sin(th)], s = Math.sqrt(c / quad(L.P, d)); pts.push([s * d[0], s * d[1]]); } fr2.ctx.setLineDash([2, 3]); drawPath(fr2, pts, color, 1, 0); fr2.ctx.setLineDash([]); };
  Vs.slice(0, 3).forEach(v => level(v, alpha("--post", 0.6)));
  drawPath(fr2, xs, css("--ink"), 1.8, 2.5);
  fr2.ctx.fillStyle = css("--warm"); fr2.ctx.beginPath(); fr2.ctx.arc(fr2.X(X0[0]), fr2.Y(X0[1]), 5, 0, 2 * Math.PI); fr2.ctx.fill();
  legend(fr2.ctx, ["black: the closed loop x⁺ = (A − BK) x from (−3.6, 2.4)", "dotted: level sets of V through the first states"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  $("#hQV").textContent = fmt(q2, 2); $("#hRV").textContent = fmt(r, 2);
  const [e1] = L.eig, rho = Math.hypot(...e1), ref = q2 === 0.2 && r === 0.2;
  $("#rK").textContent = `(${fmt(L.K[0][0], 4)}, ${fmt(L.K[0][1], 4)})`; $("#rRho").textContent = fmt(rho, 4); $("#rU0").textContent = fmt(us[0], 3); $("#rRes").textContent = resid.toExponential(1);
  caption($("#heroStatus"), `[[Q = \\mathrm{diag}(1, ${fmt(q2, 2)})]], [[R = ${fmt(r, 2)}]]${ref ? ", the canonical weights" : ""}: [[K = (${fmt(L.K[0][0], 4)},\\ ${fmt(L.K[0][1], 4)})]], [[P = ${fmtM(L.P, 4)}]], closed-loop eigenvalues [[${fmt(e1[0], 4)} \\pm ${fmt(Math.abs(e1[1]), 4)}i]]${Math.abs(e1[1]) < 1e-12 ? ` and ${fmt(L.eig[1][0], 4)}` : ""}, spectral radius ${fmt(rho, 4)}. The identity [[A_{\\mathrm{cl}}^\\top P A_{\\mathrm{cl}} - P = -(Q + K^\\top R K)]] holds to ${resid.toExponential(1)}, so along the loop [[V]] falls by exactly [[q(x, -Kx)]] at every step: ${fmt(Vs[0], 3)}, ${fmt(Vs[1], 3)}, ${fmt(Vs[2], 4)}, ${fmt(Vs[3], 4)}, … The first move is [[u_0 = -Kx_0 = ${fmt(us[0], 3)}]]${Math.abs(us[0]) > 1 ? ", beyond the bound |u| ≤ 1 that the constrained chapters impose" : ", within the bound |u| ≤ 1"}. ` + (r >= 1 ? "A heavy input penalty makes the controller timid: the eigenvalues move towards 1 and the decrease slows, while the first move shrinks." : r <= 0.05 ? "A cheap input gives an aggressive controller: the eigenvalues approach 0 and the loop is nearly dead-beat, at the price of large moves from states further out." : q2 >= 0.5 ? "A heavy velocity weight brakes harder: the first move grows and the spiral tightens." : "Chapter [[a10]] asks where this unconstrained move respects the bound, and chapter [[a11]] makes that region the terminal set."));
}
["#hQ", "#hR"].forEach(id => $(id).addEventListener("input", drawHero));
$("#hReset").addEventListener("click", () => { $("#hQ").value = 2; $("#hR").value = 3; drawHero(); });

/* ---------------- Part 2: a Lyapunov function for a hand-chosen gain ---------------- */
function lyapunov2(Acl) {   // solve Acl' P Acl - P = -I for a symmetric 2x2 P, unknowns (p11, p12, p22)
  const a = Acl, M = [], rhs = [];
  for (const [i, j] of [[0, 0], [0, 1], [1, 1]]) {
    const row = [0, 0, 0];   // (Acl' P Acl)_{ij} = sum_{k,l} a_ki p_kl a_lj
    for (let k = 0; k < 2; k++) for (let l = 0; l < 2; l++) { const idx = k === l ? (k === 0 ? 0 : 2) : 1; row[idx] += a[k][i] * a[l][j]; }
    const self = i === j ? (i === 0 ? 0 : 2) : 1; row[self] -= 1;
    M.push(row); rhs.push(i === j ? -1 : 0);
  }
  const p = solve(M, rhs); return [[p[0], p[1]], [p[1], p[2]]];
}
function drawGain() {
  const k1 = +$("#gK1").value, k2 = +$("#gK2").value, Acl = sub(A, mm(B, [[k1, k2]])), e = eig2(Acl), rho = Math.max(Math.hypot(...e[0]), Math.hypot(...e[1]));
  const P = lyapunov2(Acl), lam = eigsym2(P), posdef = rho < 1 && lam[0] > 0;
  let x = X0.slice(); const Vs = []; for (let k = 0; k < 12; k++) { Vs.push(quad(P, x)); x = mv(Acl, x); }
  const cv = $("#gainC"), narrow = narrowOf(cv), top = Math.max(2, Math.ceil(Math.log10(Math.max(...Vs.map(Math.abs), 1))));
  const fr = frame(cv, narrow ? 1.4 : 2.4, { x: [0, 11], y: [-6, top], xlabel: "step k", yticks: [-6, -4, -2, 0, 2].concat(top > 2 ? [top] : []), yfmt: logTicks }), { ctx } = fr;
  plotLine(fr, Vs.map((_, k) => k), Vs.map(v => Math.max(-6, Math.min(top, Math.log10(Math.max(Math.abs(v), 1e-300))))), { color: posdef ? css("--post") : css("--warm"), width: 2.2 });
  legend(ctx, [posdef ? "|x_k' P x_k| with P from the Lyapunov equation, log scale" : "|x_k' P x_k|: no valid Lyapunov function for this gain"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#gK1V").textContent = fmt(k1, 2); $("#gK2V").textContent = fmt(k2, 2);
  const ev = Math.abs(e[0][1]) > 1e-12 ? `[[${fmt(e[0][0], 4)} \\pm ${fmt(Math.abs(e[0][1]), 4)}i]]` : `[[${fmt(e[0][0], 4)}]] and [[${fmt(e[1][0], 4)}]]`;
  caption($("#gainCap"), `[[K = (${fmt(k1, 2)}, ${fmt(k2, 2)})]]: the closed-loop eigenvalues are ${ev}, spectral radius ${fmt(rho, 4)}, ${rho < 1 ? "inside the unit circle: Schur." : "not inside the unit circle: not Schur, the loop does not converge."} The Lyapunov equation [[A_{\\mathrm{cl}}^\\top P A_{\\mathrm{cl}} - P = -I]] has the solution [[P = ${fmtM(P, 3)}]] with eigenvalues ${fmt(lam[0], 3)} and ${fmt(lam[1], 3)}: ${posdef ? "positive definite, so [[V = x^\\top P x]] falls by [[\\lVert x \\rVert^2]] at every step and certifies the stability the eigenvalues promised." : rho < 1 ? "the equation is solvable but the solution is not positive definite, which cannot happen for a Schur matrix; this is a numerical edge case." : "not positive definite: for a matrix that is not Schur no positive-definite solution exists, and the certificate fails exactly when it should."} ` + (k1 === 0.5 && k2 === 1 ? "These are the chapter's hand-chosen gains; compare with the LQR gain (0.74, 1.26) of the hero." : ""));
}
["#gK1", "#gK2"].forEach(id => $(id).addEventListener("input", drawGain));

/* ---------------- Part 5: the Riccati recursion from a terminal weight ---------------- */
function drawRiccati() {
  const L = SYS.lqr, choice = $("#pfSel").value, Pf = choice === "0" ? [[0, 0], [0, 0]] : choice === "Q" ? SYS.Q : choice === "10P" ? L.P.map(r => r.map(v => 10 * v)) : L.P;
  const { Ps, Ks } = riccatiRecursion(A, B, SYS.Q, [[SYS.R]], Pf, 25), errs = Ps.slice().reverse().map(P => maxAbs(sub(P, L.P)));   // errs[j] after j steps
  const kerr = Ks.slice().reverse().map(K => Math.max(Math.abs(K[0][0] - L.K[0][0]), Math.abs(K[0][1] - L.K[0][1])));
  const rho2 = Math.hypot(...L.eig[0]) ** 2, cv = $("#ricC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.4 : 2.4, { x: [0, 25], y: [-16, 2], xlabel: "steps j of the recursion", xticks: [0, 5, 10, 15, 20, 25], yticks: [-16, -12, -8, -4, 0], yfmt: logTicks }), { ctx } = fr;
  const lg = v => Math.max(-16, Math.log10(Math.max(v, 1e-300)));
  plotLine(fr, errs.map((_, j) => j), errs.map(lg), { color: css("--post"), width: 2.2 });
  plotLine(fr, kerr.map((_, j) => j + 1), kerr.map(lg), { color: css("--warm"), width: 1.6, dash: [4, 3] });
  if (errs[0] > 1e-14) plotLine(fr, errs.map((_, j) => j), errs.map((_, j) => lg(errs[0] * rho2 ** j)), { color: alpha("--ink", 0.5), width: 1, dash: [2, 3] });
  legend(ctx, ["blue: |P_{N−j} − P| · dashed orange: |K_{N−j} − K|", "dotted: the rate ρ(A−BK)² = " + fmt(rho2, 4) + " per step"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const first = errs.findIndex(e => e < 1e-12), mean = errs[2] > 1e-13 ? Math.pow(errs[10] / errs[2], 1 / 8) : NaN;
  caption($("#ricCap"), `Terminal weight [[P_f = ${choice === "10P" ? "10P" : choice}]]: ${choice === "P" ? "the recursion never moves, since the DARE solution is its fixed point; the error stays at round-off." : `the error [[\\lVert P_{N-j} - P \\rVert]] falls from ${sci(errs[0], 2)} to ${sci(errs[5], 2)} after five steps and below [[10^{-12}]] after ${first} steps. From step 2 to step 10 it shrinks by a factor ${fmt(mean, 3)} per step on geometric average, against the asymptotic rate [[\\rho(A - BK)^2 = ${fmt(rho2, 4)}]] of the sandwich inequality; step by step the ratio wobbles with period three, the signature of the complex closed-loop eigenvalues.`} The gain [[K_{N-j}]] converges at the same rate, which is why a long horizon with any terminal weight ends up applying the LQR law. ` + (choice === "0" ? "From P_f = 0 the iterates rise towards P: a short horizon underestimates the cost-to-go, the horizon-one pathology of chapter A6 in matrix form." : choice === "10P" ? "From above, every iterate stays above P and the bound ‖A_cl^j‖² ‖P_f − P‖ holds exactly." : choice === "Q" ? "P_f = Q is the choice chapter A6's first design makes: it reaches P to round-off in about fourteen steps." : ""));
}
$("#pfSel").addEventListener("change", drawRiccati);

function drawAll() { drawHero(); drawGain(); drawRiccati(); }
drawAll();
onRedraw(drawAll);
})();
