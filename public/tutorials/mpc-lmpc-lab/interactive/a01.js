/* Advanced MPC and Learning MPC, chapter A1, state-space models and the double integrator: the demos.
   The hero rolls the cart out under a chosen input sequence, with the zero-order-hold model and,
   optionally, the explicit-Euler model, in time and in the phase plane. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { SYS, mv, legend, drawPoly, vertices, drawPath } = window.MPC;

const PRESETS = {
  push: { U: [1, 1, 0, -1, -1], name: "push, coast, brake (1, 1, 0, −1, −1)" },
  early: { U: [1, 1, -1, -1, 0], name: "brake earlier (1, 1, −1, −1, 0)" },
  half: { U: [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5], name: "half-size inputs, eight moves" },
};
const st = { preset: "push", euler: false, scale: 1 };
const X0 = [-3, 0];

function roll(A, B, U) { const xs = [X0.slice()]; let x = X0; for (const u of U) { x = mv(A, x).map((v, i) => v + B[i][0] * u); xs.push(x); } return xs; }

function drawHero() {
  const U = PRESETS[st.preset].U.map(u => u * st.scale), Ts = 1;
  const Az = [[1, Ts], [0, 1]], Bz = [[Ts * Ts / 2], [Ts]], Be = [[0], [Ts]];
  const xz = roll(Az, Bz, U), xe = roll(Az, Be, U), T = U.length;
  const viol = xz.findIndex(x => Math.abs(x[0]) > 4 + 1e-9 || Math.abs(x[1]) > 3 + 1e-9);
  /* time plots: position and velocity as lines, the input as steps */
  const cv = $("#timeC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.3 : 2.1, { x: [0, T], y: [-5, 5], xlabel: "step k", xticks: [...Array(T + 1).keys()], yticks: [-4, -2, 0, 2, 4] }), { ctx, X, Y } = fr;
  ctx.strokeStyle = alpha("--ink", 0.25); ctx.setLineDash([3, 3]); seg(ctx, fr.pad.l, Y(4), fr.w - fr.pad.r, Y(4)); seg(ctx, fr.pad.l, Y(-4), fr.w - fr.pad.r, Y(-4)); ctx.setLineDash([]);
  /* input as a staircase */
  ctx.fillStyle = alpha("--c4", 0.25); U.forEach((u, k) => ctx.fillRect(X(k), Y(Math.max(u, 0)), X(k + 1) - X(k), Math.abs(Y(u) - Y(0))));
  const ks = xz.map((_, k) => k);
  plotLine(fr, ks, xz.map(x => x[0]), { color: css("--post"), width: 2.2 });
  plotLine(fr, ks, xz.map(x => x[1]), { color: css("--warm"), width: 2.2 });
  if (st.euler) { plotLine(fr, ks, xe.map(x => x[0]), { color: css("--post"), width: 1.4, dash: [5, 4] }); }
  ctx.fillStyle = css("--post"); xz.forEach((x, k) => { ctx.beginPath(); ctx.arc(X(k), Y(x[0]), 3, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = css("--warm"); xz.forEach((x, k) => { ctx.beginPath(); ctx.arc(X(k), Y(x[1]), 3, 0, 2 * Math.PI); ctx.fill(); });
  legend(ctx, ["blue: position p · orange: velocity v · green: input u", "dotted: the position bound ±4" + (st.euler ? " · dashed: Euler's position" : "")], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* phase plane */
  const cv2 = $("#planeC"), fr2 = frame(cv2, narrow ? 1.2 : 1.5, { x: [-5.5, 5.5], y: [-4, 4], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, 0, 3] });
  drawPoly(fr2, vertices(SYS.X), alpha("--ink", 0.04), alpha("--ink", 0.35), 1, [3, 3]);
  if (st.euler) { fr2.ctx.setLineDash([5, 4]); drawPath(fr2, xe, alpha("--ink", 0.5), 1.4, 0); fr2.ctx.setLineDash([]); }
  drawPath(fr2, xz, css("--ink"), 1.8, 2.5);
  fr2.ctx.fillStyle = css("--warm"); fr2.ctx.beginPath(); fr2.ctx.arc(fr2.X(X0[0]), fr2.Y(X0[1]), 5, 0, 2 * Math.PI); fr2.ctx.fill();
  legend(fr2.ctx, ["the state (p, v) at each step; orange: the start", "dotted box: |p| ≤ 4, |v| ≤ 3"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  /* readouts and the caption */
  $$("#demo [data-preset]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.preset === st.preset)));
  $("#hEuler").setAttribute("aria-pressed", String(st.euler)); $("#hScaleV").textContent = fmt(st.scale, 2);
  const last = xz[xz.length - 1];
  $("#rFinal").textContent = `(${fmt(last[0], 2)}, ${fmt(last[1], 2)})`;
  $("#rMaxV").textContent = fmt(Math.max(...xz.map(x => Math.abs(x[1]))), 2);
  $("#rBox").textContent = viol < 0 ? "never" : `at step ${viol}`;
  $("#rGap").textContent = st.euler ? fmt(Math.max(...xz.map((x, k) => Math.abs(x[0] - xe[k][0]))), 2) : "–";
  const scaled = st.scale !== 1 ? ` scaled by ${fmt(st.scale, 2)}` : "";
  let text = `[[x_0 = (-3, 0)]], inputs ${PRESETS[st.preset].name}${scaled}, [[T_s = 1]]. Each input is a constant acceleration held for one second: the velocity changes by [[u_k]] and the position by [[v_k + u_k / 2]], which is the zero-order-hold model [[A = \\begin{pmatrix} 1 & 1 \\\\ 0 & 1 \\end{pmatrix}]], [[B = (0.5, 1)]]. The cart ends at [[(${fmt(last[0], 2)}, ${fmt(last[1], 2)})]] after ${T} moves. `;
  text += viol < 0 ? "Every state stays inside the box: this sequence is admissible. " : `At step ${viol} the state leaves the box [[|p| \\le 4, |v| \\le 3]]: the same inputs are no longer admissible, which is what a constraint means. `;
  if (st.euler) text += `The dashed Euler model updates the position with the old velocity only, [[p_{k+1} = p_k + v_k]], so its position lags the true one by [[T_s v_k / 2]] at every step: largest gap ${fmt(Math.max(...xz.map((x, k) => Math.abs(x[0] - xe[k][0]))), 2)} here. The velocities agree exactly.`;
  else text += "Switch on the Euler model to see what a cruder discretisation changes, and scale the inputs to see when the box is left.";
  caption($("#heroStatus"), text);
}
$$("#demo [data-preset]").forEach(b => b.addEventListener("click", () => { st.preset = b.dataset.preset; drawHero(); }));
$("#hEuler").addEventListener("click", () => { st.euler = !st.euler; drawHero(); });
$("#hScale").addEventListener("input", () => { st.scale = +$("#hScale").value; drawHero(); });
$("#hReset").addEventListener("click", () => { st.preset = "push"; st.euler = false; st.scale = 1; $("#hScale").value = 1; drawHero(); });

/* Part 3: Euler against zero-order hold as the sample time shrinks, for a 5 s push-then-brake */
function drawTs() {
  const Ts = [1, 0.5, 0.25, 0.1][+$("#tsSel").value], steps = Math.round(5 / Ts), U = [];
  for (let k = 0; k < steps; k++) U.push(k < Math.floor(steps / 2) ? 1 : -1);
  const Az = [[1, Ts], [0, 1]], Bz = [[Ts * Ts / 2], [Ts]], Be = [[0], [Ts]], xz = roll(Az, Bz, U), xe = roll(Az, Be, U);
  const cv = $("#tsC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.3, { x: [0, 5], y: [-4, 4], xlabel: "time (s)", yticks: [-4, -2, 0, 2, 4] });
  const ts = xz.map((_, k) => k * Ts);
  plotLine(fr, ts, xz.map(x => x[0]), { color: css("--post"), width: 2.2 });
  plotLine(fr, ts, xe.map(x => x[0]), { color: css("--post"), width: 1.4, dash: [5, 4] });
  plotLine(fr, ts, xz.map(x => x[1]), { color: css("--warm"), width: 1.6 });
  legend(fr.ctx, ["blue: position, zero-order hold (exact) · dashed: Euler", "orange: velocity (both models agree)"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const gap = Math.max(...xz.map((x, k) => Math.abs(x[0] - xe[k][0]))), vmax = Math.max(...xz.map(x => x[1]));
  $("#tsV").textContent = Ts;
  caption($("#tsCap"), `[[T_s = ${Ts}]]: ${steps} steps of acceleration [[+1]] then [[-1]]. The largest gap between the two positions is ${fmt(gap, 3)}, which is [[T_s / 2]] times the largest velocity ${fmt(vmax, 2)}: Euler's lag is proportional to the sample time, and vanishes as [[T_s \\to 0]]. Both models reach the same final velocity, and the same final position whenever the net velocity change is zero, as it is here; what differs is every state in between, which is what a constraint is checked on.`);
}
$("#tsSel").addEventListener("input", drawTs);

function drawAll() { drawHero(); drawTs(); }
drawAll();
onRedraw(drawAll);
})();
