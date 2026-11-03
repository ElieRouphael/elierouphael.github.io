/* Advanced MPC and Learning MPC, chapter C2, reference tracking and offset-free MPC: the demos.
   The hero runs plain tracking MPC and offset-free MPC on the sloping cart: the deviation problem is
   built with mpc-lib.js's makeMPC on the shifted box with a terminal set recomputed around the target,
   the observer is a current-estimator Luenberger observer whose gains come from the numbers script.
   Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { SYS, mv, dot, box, vertices, closedLoopAdmissible, maximalInvariant, makeMPC, area, legend, drawPoly } = window.MPC;

const { A, B, Q, R } = SYS, L = SYS.lqr, K = L.K[0], AK = L.Acl, N = 5;
/* observer gains for the current estimator, by pole placement in Python (code/c02_numbers.py) */
const GAINS = [{ poles: "0.3, 0.35, 0.4", L: [0.958, 0.8555, 0.273] }, { poles: "0.5, 0.55, 0.6", L: [0.835, 0.47, 0.09] }, { poles: "0.8, 0.85, 0.9", L: [0.388, 0.0605, 0.003] }];
const Aa = [[1, 1, 0.5], [0, 1, 1], [0, 0, 1]], Ba = [0.5, 1, 0];

/* the target for the cart: x_s = (r, 0), u_s = -d_hat (the input that cancels the slope) */
const target = (r, dhat) => ({ xs: [r, 0], us: -dhat });
/* the deviation problem around a target: shifted box and input bounds, the LQR's invariant set inside them */
const cache = {};
function deviationMPC(xs, us) {
  const key = `${xs[0].toFixed(4)}|${us.toFixed(4)}`; if (cache[key]) return cache[key];
  const lo = [SYS.xlb[0] - xs[0], SYS.xlb[1] - xs[1]], hi = [SYS.xub[0] - xs[0], SYS.xub[1] - xs[1]], ulo = SYS.umin - us, uhi = SYS.umax - us;
  const XK = closedLoopAdmissible(box(lo, hi), K, ulo, uhi), Xf = maximalInvariant(AK, XK).set;
  return cache[key] = { mpc: makeMPC(A, B, Q, R, N, L.P, lo, hi, ulo, uhi, { kind: "set", H: Xf.H, h: Xf.h }), Xf };
}
function simulate(rseq, d, Lobs, mode) {
  let x = [0, 0], pred = [0, 0, 0]; const xs = [x.slice()], us = [], dh = [], fails = [];
  for (let t = 0; t < rseq.length; t++) {
    const y = x[0], xh = mode === "free" ? pred.map((v, i) => v + Lobs[i] * (y - pred[0])) : [x[0], x[1], 0];
    const { xs: xS, us: uS } = target(rseq[t], mode === "free" ? xh[2] : 0), { mpc } = deviationMPC(xS, uS);
    const sol = mpc.solve([xh[0] - xS[0], xh[1] - xS[1]]);
    let u; if (!sol.feasible) { u = us.length ? us[us.length - 1] : 0; fails.push(t); } else u = uS + sol.U[0];
    x = mv(A, x).map((v, i) => v + B[i][0] * (u + d));
    pred = mv(Aa, xh).map((v, i) => v + Ba[i] * u);
    xs.push(x.slice()); us.push(u); dh.push(xh[2]);
  }
  return { xs, us, dh, fails };
}
const settled = (xs, lo, hi, r, tol = 0.01) => { for (let t = lo; t < hi; t++) { let ok = true; for (let k = t; k <= hi; k++) if (Math.abs(xs[k][0] - r) >= tol) { ok = false; break; } if (ok) return t; } return null; };

function drawHero() {
  const d = +$("#hD").value, g = GAINS[+$("#hL").value], r1 = +$("#hR1").value, r2 = +$("#hR2").value;
  const rseq = [...Array(30).fill(r1), ...Array(30).fill(r2)];
  const plain = simulate(rseq, d, g.L, "plain"), free = simulate(rseq, d, g.L, "free");
  const cv = $("#posC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, 60], y: [-4.5, 4.5], xlabel: "step t", xticks: [0, 10, 20, 30, 40, 50, 60], yticks: [-4, -2, 0, 2, 4] }), { ctx, X, Y } = fr;
  ctx.strokeStyle = alpha("--ink", 0.3); ctx.setLineDash([3, 3]); [4, -4].forEach(b => seg(ctx, fr.pad.l, Y(b), fr.w - fr.pad.r, Y(b))); ctx.setLineDash([]);
  ctx.strokeStyle = alpha("--ink", 0.5); ctx.lineWidth = 1.5; seg(ctx, X(0), Y(r1), X(30), Y(r1)); seg(ctx, X(30), Y(r1), X(30), Y(r2)); seg(ctx, X(30), Y(r2), X(60), Y(r2));
  const ts = plain.xs.map((_, t) => t);
  plotLine(fr, ts, plain.xs.map(p => p[0]), { color: css("--warm"), width: 2 });
  plotLine(fr, ts, free.xs.map(p => p[0]), { color: css("--post"), width: 2 });
  free.fails.forEach(t => { ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(t), Y(free.xs[t][0]), 3, 0, 2 * Math.PI); ctx.fill(); });
  legend(ctx, ["grey: the setpoint · orange: plain tracking MPC · blue: offset-free MPC", "dotted: the position bound ±4" + (free.fails.length ? " · orange dots: no feasible plan, last input held" : "")], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const cv2 = $("#distC"), ymax = Math.max(0.5, Math.abs(d) * 1.6, ...free.dh.map(Math.abs)) * 1.1, fr2 = frame(cv2, narrow ? 1.6 : 3, { x: [0, 60], y: [-ymax, ymax], xlabel: "step t", xticks: [0, 10, 20, 30, 40, 50, 60] }), c2 = fr2.ctx;
  c2.strokeStyle = alpha("--ink", 0.5); c2.setLineDash([5, 4]); seg(c2, fr2.pad.l, fr2.Y(d), fr2.w - fr2.pad.r, fr2.Y(d)); c2.setLineDash([]);
  plotLine(fr2, free.dh.map((_, t) => t), free.dh, { color: css("--post"), width: 2 });
  legend(c2, ["blue: the estimate d̂ · dashed: the true slope d"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  $("#hDV").textContent = fmt(d, 2); $("#hLV").textContent = g.poles; $("#hR1V").textContent = fmt(r1, 1); $("#hR2V").textContent = fmt(r2, 1);
  $("#rPlain").textContent = `${fmt(plain.xs[30][0] - r1, 3)} / ${fmt(plain.xs[60][0] - r2, 3)}`;
  $("#rFree").textContent = `${fmt(free.xs[30][0] - r1, 3)} / ${fmt(free.xs[60][0] - r2, 3)}`;
  $("#rD").textContent = fmt(free.dh[29], 4);
  const s1 = settled(free.xs, 0, 30, r1), s2 = settled(free.xs, 30, 60, r2);
  $("#rSettle").textContent = s1 === null ? "not within 30 steps" : `${s1} (plain: ${settled(plain.xs, 0, 30, r1) === null ? "never" : settled(plain.xs, 0, 30, r1)})`;
  const ref = d === -0.2 && +$("#hL").value === 1 && r1 === 2 && r2 === -1;
  const Mi = [[1 - AK[0][0], -AK[0][1]], [-AK[1][0], 1 - AK[1][1]]], det = Mi[0][0] * Mi[1][1] - Mi[0][1] * Mi[1][0], bd = [B[0][0] * d, B[1][0] * d];
  const offset = (Mi[1][1] * bd[0] - Mi[0][1] * bd[1]) / det;
  let text = `Slope [[d = ${fmt(d, 2)}]], observer eigenvalues ${g.poles}, setpoints [[r_1 = ${fmt(r1, 1)}]] then [[r_2 = ${fmt(r2, 1)}]]${ref ? ", the chapter's run" : ""}. `;
  if (d === 0) text += `With no slope the two controllers are the same controller: both reach ${fmt(plain.xs[30][0], 3)} and ${fmt(plain.xs[60][0], 3)}, and the estimate stays at zero. `;
  else text += `Plain tracking MPC settles at [[${fmt(plain.xs[30][0], 3)}]] and [[${fmt(plain.xs[60][0], 3)}]]: the offset [[(I - A_K)^{-1} B d = ${fmt(offset, 3)}]] at both setpoints, with the error supplying the input [[${fmt(-d, 2)}]] that holds the cart against the slope. `;
  if (d !== 0) text += `The offset-free controller's estimate reaches [[${fmt(free.dh[29], 3)}]] by step 30 and its position ends at [[${fmt(free.xs[30][0], 3)}]] and [[${fmt(free.xs[60][0], 3)}]]${s1 !== null ? `, within 0.01 of [[r_1]] from step ${s1}` : ", not yet within 0.01 of the first setpoint after thirty steps"}${s2 !== null ? ` and of [[r_2]] from step ${s2}, ${s2 - 30} after the change` : ""}. `;
  if (free.fails.length) text += `At ${free.fails.length} step${free.fails.length > 1 ? "s" : ""} (first at ${free.fails[0]}) the deviation problem was infeasible, because the estimate placed [[\\hat x - x_s]] outside the feasible region; the last input was held. `;
  if (Math.abs(r1) >= 3.5 || Math.abs(r2) >= 3.5) text += `A setpoint this close to the wall leaves the terminal set a sliver and the deviation box almost no room on that side.`;
  caption($("#heroStatus"), text);
}
["#hD", "#hL", "#hR1", "#hR2"].forEach(id => $(id).addEventListener("input", drawHero));
$("#hReset").addEventListener("click", () => { $("#hD").value = -0.2; $("#hL").value = 1; $("#hR1").value = 2; $("#hR2").value = -1; drawHero(); });

/* Part 2: the terminal set recomputed around the setpoint */
function drawShift() {
  const r = +$("#shiftR").value, { xs } = target(r, 0), { Xf } = deviationMPC(xs, 0), V = vertices(Xf).map(v => [v[0] + xs[0], v[1] + xs[1]]);
  const cv = $("#shiftC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.2 : 1.8, { x: [-4.6, 4.6], y: [-3.5, 3.5], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, 0, 3] }), { ctx, X, Y } = fr;
  drawPoly(fr, vertices(SYS.X), alpha("--ink", 0.03), alpha("--ink", 0.35), 1, [3, 3]);
  const V0 = vertices(deviationMPC([0, 0], 0).Xf); drawPoly(fr, V0, null, alpha("--ink", 0.25), 1, [4, 3]);
  drawPoly(fr, V, alpha("--post", 0.25), css("--post"), 2);
  ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(xs[0]), Y(0), 5, 0, 2 * Math.PI); ctx.fill();
  legend(ctx, ["blue: the LQR's invariant set around the setpoint (orange)", "grey dashed: the same set around the origin"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#shiftRV").textContent = fmt(r, 1);
  const room = 4 - Math.abs(r);
  caption($("#shiftCap"), `Setpoint [[r = ${fmt(r, 1)}]]: the deviation box leaves ${fmt(room, 1)} of room on the setpoint's side, and the LQR's invariant set inside it has ${V.length} vertices and area ${fmt(area(V), 2)} (7.31 at the origin). ${room < 1.5 ? "Near the wall the set is a sliver: the local controller would step over the bound from most of the symmetric set, so those states are excluded." : room < 2.5 ? "The corner on the setpoint's side is cut: from there the LQR's first move would leave the box." : "Away from the walls the set is the symmetric parallelogram of chapter A10, translated."}`);
}
$("#shiftR").addEventListener("input", drawShift);

function drawAll() { drawHero(); drawShift(); }
drawAll();
onRedraw(drawAll);
})();
