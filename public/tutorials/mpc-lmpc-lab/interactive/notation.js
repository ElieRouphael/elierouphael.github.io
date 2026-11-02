/* The notation page: the canonical system's sets, computed live. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, css, fmt, caption, frame, alpha, onRedraw, narrowOf } = window.Kit;
const { SYS, dlqr, vertices, box, closedLoopAdmissible, maximalInvariant, maximalControlInvariant, controllableSets, zeroTarget, area, legend, drawPoly, drawPath, mv } = window.MPC;

function draw() {
  const umax = [0.5, 1][+$("#uSel").value], L = SYS.lqr;
  const XK = closedLoopAdmissible(SYS.X, L.K[0], -umax, umax), Xf = maximalInvariant(L.Acl, XK), C = maximalControlInvariant(SYS.A, SYS.B, SYS.X, -umax, umax);
  const VXf = vertices(Xf.set), VC = vertices(C.set), VXK = vertices(XK);
  const cv = $("#sysC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.15 : 1.5, { x: [-4.6, 4.6], y: [-3.6, 3.6], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, 0, 3] });
  drawPoly(fr, vertices(SYS.X), alpha("--ink", 0.04), alpha("--ink", 0.35), 1, [3, 3]);
  drawPoly(fr, VC, alpha("--warm", 0.12), css("--warm"), 1.5, [5, 4]);
  drawPoly(fr, VXK, null, css("--c5"), 1.2, [2, 3]);
  drawPoly(fr, VXf, alpha("--post", 0.3), css("--post"), 2);
  /* the LQR loop from one start inside X_f, and the saturated loop from a corner of C_inf */
  let x = [-2.2, 0.5]; const xs = [x]; for (let k = 0; k < 8; k++) { x = mv(L.Acl, x); xs.push(x); }
  drawPath(fr, xs, css("--ink"), 1.4, 2);
  legend(fr.ctx, ["dotted: the box X · dashed orange: C_∞ · purple: X_K", "blue: X_f = O_∞ · black: the LQR loop from (−2.2, 0.5)"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#rK").textContent = `(${fmt(L.K[0][0], 4)}, ${fmt(L.K[0][1], 4)})`;
  $("#rRho").textContent = fmt(Math.hypot(...L.eig[0]), 4);
  $("#rXf").textContent = `${VXf.length} vertices, area ${fmt(area(VXf), 2)}, ${Xf.history.length - 1} iterations`;
  $("#rC").textContent = `${VC.length} vertices, area ${fmt(area(VC), 2)}, ${C.history.length - 1} iterations`;
  $("#rXK").textContent = `${VXK.length} vertices, area ${fmt(area(VXK), 2)}`;
  $("#uV").textContent = fmt(umax, 1);
  const z5 = controllableSets(SYS.A, SYS.B, SYS.X, zeroTarget(), -umax, umax, 5), f5 = controllableSets(SYS.A, SYS.B, SYS.X, Xf.set, -umax, umax, 5);
  caption($("#sysStatus"), `[[|u| \\le ${fmt(umax, 1)}]]: the LQR gain [[K = (${fmt(L.K[0][0], 4)}, ${fmt(L.K[0][1], 4)})]] with [[P = \\begin{pmatrix} ${fmt(L.P[0][0], 4)} & ${fmt(L.P[0][1], 4)} \\\\ ${fmt(L.P[1][0], 4)} & ${fmt(L.P[1][1], 4)} \\end{pmatrix}]] and closed-loop eigenvalues [[${fmt(L.eig[0][0], 4)} \\pm ${fmt(Math.abs(L.eig[0][1]), 4)}\\,i]]. The LQR input is admissible on [[\\mathcal X_K]] (${VXK.length} vertices); its largest invariant subset [[\\mathcal X_f]] has ${VXf.length} vertices and area ${fmt(area(VXf), 2)}; the largest set any admissible controller can keep safe, [[\\mathcal C_\\infty]], has ${VC.length} vertices and area ${fmt(area(VC), 2)}. At [[N = 5]] the zero-terminal region has area ${fmt(area(vertices(z5[5])), 2)} and the [[\\mathcal X_f]]-terminal region ${fmt(area(vertices(f5[5])), 2)}.`);
}
$("#uSel").addEventListener("input", draw);
draw();
onRedraw(draw);
})();
