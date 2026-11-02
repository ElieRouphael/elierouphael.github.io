/* Advanced MPC and Learning MPC, chapter A12, sampled and convex safe sets: the demos.
   The hero stores three successful runs of the canonical cart from the repeated start (two plain
   short-horizon MPC runs and a saturated LQR run), draws their sampled safe set and its convex hull,
   and answers a clicked probe: a stored state with its continuation, a hull point with its barycentric
   weights, mixed input and successor, or a point outside the hull. Needs mpc-lib.js and mpc-lmpc.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { SYS, mv, dot, hull, area, vertices, drawPoly, clickPoint, legend, drawPath } = window.MPC;
const L = window.LMPC;

const T = L.makeTask({ A: SYS.A, B: SYS.B, Q: SYS.Q, R: SYS.R, xlb: SYS.xlb, xub: SYS.xub, umin: SYS.umin, umax: SYS.umax, xS: [-3.5, 2.5] });
const K = SYS.lqr.K[0];
const RUNS = [
  { name: "plain MPC, N = 2", token: "--warm", run: L.runSeed(T, 2) },
  { name: "plain MPC, N = 3", token: "--c4", run: L.runSeed(T, 3) },
  { name: "saturated LQR", token: "--c5", run: L.runLaw(T, x => -dot(K, x)) },
];
const store = L.makeStore();
RUNS.forEach(r => { if (!r.run.failed) L.addTrajectory(T, store, r.run.X, r.run.U); });
const hs = { probe: [-1, 1.8], shown: [true, true, true] };

function activeStore() { const s = L.makeStore(); RUNS.forEach((r, i) => { if (hs.shown[i] && !r.run.failed) L.addTrajectory(T, s, r.run.X, r.run.U); }); return s; }
function drawHero() {
  const s = activeStore(), data = L.safeData(s, null), pts = data.S, H = pts.length >= 3 ? hull(pts) : [];
  const cv = $("#safeC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.15 : 1.5, { x: [-4.4, 4.4], y: [-3.3, 3.3], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, -1.5, 0, 1.5, 3] }), { ctx, X, Y } = fr; cv._fr = fr;
  drawPoly(fr, vertices(SYS.X), alpha("--ink", 0.03), alpha("--ink", 0.3), 1, [3, 3]);
  if (H.length >= 3) drawPoly(fr, H, alpha("--post", 0.14), alpha("--post", 0.7), 1.5);
  RUNS.forEach((r, i) => { if (!hs.shown[i] || r.run.failed) return; drawPath(fr, r.run.X, css(r.token), 1.6, 2.6); });
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(0), Y(0), 4, 0, 2 * Math.PI); ctx.fill();
  const q = hs.probe;
  /* classify the probe */
  const near = pts.map((p, k) => [Math.hypot(p[0] - q[0], p[1] - q[1]), k]).sort((a, b) => a[0] - b[0])[0];
  const inside = H.length >= 3 && H.every((v, i) => { const w = H[(i + 1) % H.length]; return (w[0] - v[0]) * (q[1] - v[1]) - (w[1] - v[1]) * (q[0] - v[0]) >= -1e-9; });
  let text, succ = null;
  if (near && near[0] < 0.12) {
    const k = near[1], st = pts[k], u = data.Us[k], nx = pts[data.next[k]], src = data.from[k];
    succ = nx; ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; seg(ctx, X(st[0]), Y(st[1]), X(nx[0]), Y(nx[1]));
    if (k === 0) text = `The probe is the target itself: stored once, with value 0 and the continuation [[u = 0]], which keeps it at the origin for ever. That is the one assumption about the target the sampled safe set needs.`;
    else text = `The probe is a stored state: step ${src[1]} of the run "${RUNS[src[0]].name}", [[(${fmt(st[0], 3)}, ${fmt(st[1], 3)})]]. Its certificate is concrete: the stored input [[u = ${fmt(u, 3)}]] was applied there, in a run that respected every constraint and reached the target, and it leads to the stored successor [[(${fmt(nx[0], 3)}, ${fmt(nx[1], 3)})]], another safe state, ${data.V[k] > 0 ? `with a realised cost-to-go of ${fmt(data.V[k], 3)} from here` : ""}. Replaying the rest of that run is a known admissible way home. Nothing about the model was needed for this claim, only that the run happened.`;
  } else if (inside) {
    const r = L.convexSafeValue(q, pts, data.V), act = r.lam.map((l, i) => [l, i]).filter(([l]) => l > 1e-6);
    const u = dot(r.lam, data.Us), xdyn = mv(T.A, q).map((v, i) => v + T.B[i][0] * u), xdata = [0, 1].map(d => dot(r.lam, data.next.map(j => pts[j][d])));
    const inHull = H.every((v, i) => { const w = H[(i + 1) % H.length]; return (w[0] - v[0]) * (xdyn[1] - v[1]) - (w[1] - v[1]) * (xdyn[0] - v[0]) >= -1e-9; });
    succ = xdyn;
    act.forEach(([l, i]) => { ctx.strokeStyle = alpha("--ink", 0.35); ctx.setLineDash([3, 3]); seg(ctx, X(q[0]), Y(q[1]), X(pts[i][0]), Y(pts[i][1])); ctx.setLineDash([]); });
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; seg(ctx, X(q[0]), Y(q[1]), X(xdyn[0]), Y(xdyn[1]));
    text = `The probe [[(${fmt(q[0], 2)}, ${fmt(q[1], 2)})]] is inside the hull but on no stored run. The LP writes it as ${act.map(([l, i]) => `[[${fmt(l, 3)}]] of [[(${fmt(pts[i][0], 2)}, ${fmt(pts[i][1], 2)})]]`).join(", ")}, ${act.length} stored states (Carathéodory promises at most three). The same weights on their stored inputs give [[u = ${fmt(u, 3)}]]${Math.abs(u) <= 1 + 1e-9 ? ", admissible because the interval is convex" : ""}, and [[Ax + Bu = (${fmt(xdyn[0], 3)}, ${fmt(xdyn[1], 3)})]], which equals the same weights on the stored successors, [[(${fmt(xdata[0], 3)}, ${fmt(xdata[1], 3)})]], to ${Math.max(Math.abs(xdyn[0] - xdata[0]), Math.abs(xdyn[1] - xdata[1])).toExponential(1)}: that is linearity. The successor is ${inHull ? "inside the hull, because the hull is convex and every stored successor is in it" : "outside the hull (a rounding case at the boundary)"}. The certificate here is the model's, not the data's: it rests on linear dynamics and convex constraints.`;
  } else {
    text = `The probe [[(${fmt(q[0], 2)}, ${fmt(q[1], 2)})]] is outside the hull${near ? `; the nearest stored state is ${fmt(near[0], 2)} away` : ""}. No convex combination of stored states reaches it, so neither the data nor the convexity argument says anything about a safe continuation from here. An LMPC would refuse it as a terminal state, even if it lies inside the box and inside the control-invariant set of chapter A10: the safe set is a certificate about experience, not about the whole state space.`;
  }
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--bg"); ctx.lineWidth = 1.5; ctx.stroke();
  if (succ) { ctx.fillStyle = css("--post"); ctx.beginPath(); ctx.arc(X(succ[0]), Y(succ[1]), 4.5, 0, 2 * Math.PI); ctx.fill(); }
  legend(ctx, ["orange: plain MPC N = 2 · green: plain MPC N = 3 · purple: saturated LQR", "blue region: the convex safe set · black: the probe and its one-step successor"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $$("#demo [data-run]").forEach((b, i) => b.setAttribute("aria-pressed", String(hs.shown[i])));
  $("#rStates").textContent = String(pts.length);
  $("#rArea").textContent = H.length >= 3 ? fmt(area(H), 2) : "–";
  $("#rRuns").textContent = RUNS.map((r, i) => hs.shown[i] && !r.run.failed ? `${r.run.U.length}` : "–").join(" / ");
  $("#rCost").textContent = RUNS.map((r, i) => hs.shown[i] && !r.run.failed ? fmt(L.costToGo(T, r.run.X, r.run.U)[0], 1) : "–").join(" / ");
  caption($("#heroStatus"), text + " Tap or click the plane to move the probe.");
}
$("#safeC").addEventListener("click", ev => { const p = clickPoint(ev); if (p) { hs.probe = [clamp(p[0], -4, 4), clamp(p[1], -3, 3)]; drawHero(); } });
$$("#demo [data-run]").forEach((b, i) => b.addEventListener("click", () => { hs.shown[i] = !hs.shown[i]; if (!hs.shown.some(Boolean)) hs.shown[i] = true; drawHero(); }));
$("#hReset").addEventListener("click", () => { hs.probe = [-1, 1.8]; hs.shown = [true, true, true]; drawHero(); });

/* Part 4: the convex-combination identity on four stored states of the first run */
function drawInvariance() {
  const data = L.safeData(store, null), S = data.S, Us = data.Us, next = data.next;
  const n0 = RUNS[0].run.U.length, chosen = [1, Math.min(4, n0), Math.min(7, n0), Math.min(10, n0)].map(i => Math.min(i, n0));
  const t = +$("#ivT").value, lam = [0.10, 0.25, 0.30, 0.35].map((v, i) => i === 0 ? v * (1 - t) + t : v * (1 - t));
  const xd = [0, 1].map(d => dot(lam, chosen.map(i => S[i][d]))), ud = dot(lam, chosen.map(i => Us[i])), xDyn = mv(T.A, xd).map((v, i) => v + T.B[i][0] * ud), xData = [0, 1].map(d => dot(lam, chosen.map(i => S[next[i]][d])));
  const cv = $("#invC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.2 : 1.8, { x: [-4.4, 1.2], y: [-1.6, 3.3], xlabel: "p", ylabel: "v", xticks: [-4, -3, -2, -1, 0, 1], yticks: [-1, 0, 1, 2, 3] }), { ctx, X, Y } = fr;
  drawPath(fr, RUNS[0].run.X, alpha("--warm", 0.55), 1.4, 0);
  chosen.forEach((i, k) => { ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(S[i][0]), Y(S[i][1]), 3 + 8 * lam[k], 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = alpha("--ink", 0.3); seg(ctx, X(S[i][0]), Y(S[i][1]), X(xd[0]), Y(xd[1])); });
  ctx.fillStyle = css("--post"); ctx.beginPath(); ctx.arc(X(xd[0]), Y(xd[1]), 6, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--post"); ctx.lineWidth = 2; seg(ctx, X(xd[0]), Y(xd[1]), X(xDyn[0]), Y(xDyn[1])); ctx.beginPath(); ctx.arc(X(xDyn[0]), Y(xDyn[1]), 6, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X(xData[0]), Y(xData[1]), 9, 0, 2 * Math.PI); ctx.stroke();
  legend(ctx, ["orange: four stored states of the N = 2 run (size = weight)", "blue: their combination and its successor · ring: the combination of the stored successors"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#ivTV").textContent = fmt(t, 2);
  caption($("#invCap"), `Stored states ${chosen.join(", ")} of the first run with weights [[\\lambda = (${lam.map(v => fmt(v, 3)).join(", ")})]]: the combination [[x = \\sum_r \\lambda_r s_r = (${fmt(xd[0], 4)}, ${fmt(xd[1], 4)})]] with the combined input [[u = \\sum_r \\lambda_r u_r = ${fmt(ud, 4)}]] moves to [[Ax + Bu = (${fmt(xDyn[0], 4)}, ${fmt(xDyn[1], 4)})]], and the same weights on the stored successors give [[(${fmt(xData[0], 4)}, ${fmt(xData[1], 4)})]]: identical to ${Math.max(Math.abs(xDyn[0] - xData[0]), Math.abs(xDyn[1] - xData[1])).toExponential(1)}. That is the whole invariance proof in one line of arithmetic. The slider moves weight onto the first point.`);
}
$("#ivT").addEventListener("input", drawInvariance);

/* Part 5: a nonlinear counter-example, x+ = x^2 + u on the line */
function drawNonlinear() {
  const s1 = 0, s2 = 2, th = +$("#nlT").value, x = (1 - th) * s1 + th * s2, u = 0, fx = x * x + u, mix = (1 - th) * (s1 * s1) + th * (s2 * s2);
  const cv = $("#nlC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.5 : 2.4, { x: [-0.3, 2.3], y: [-0.3, 4.6], xlabel: "x", ylabel: "f(x, 0) = x²", xticks: [0, 0.5, 1, 1.5, 2], yticks: [0, 1, 2, 3, 4] }), { ctx, X, Y } = fr;
  const xs = Array.from({ length: 60 }, (_, i) => i * 2.2 / 59); plotLine(fr, xs, xs.map(v => v * v), { color: css("--post"), width: 2 });
  plotLine(fr, [s1, s2], [0, 4], { color: alpha("--ink", 0.5), width: 1.5, dash: [5, 4] });
  ctx.fillStyle = css("--warm"); [[s1, 0], [s2, 4]].forEach(p => { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = css("--post"); ctx.beginPath(); ctx.arc(X(x), Y(fx), 5.5, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X(x), Y(mix), 8, 0, 2 * Math.PI); ctx.stroke();
  legend(ctx, ["orange: stored states 0 and 2 with their successors 0 and 4", "blue: f at the mixed state · ring: the mixed successor"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#nlTV").textContent = fmt(th, 2);
  caption($("#nlCap"), `[[f(x, u) = x^2 + u]] on the line, stored states [[s_1 = 0]] and [[s_2 = 2]] with inputs 0 and successors 0 and 4. The mixture [[x = ${fmt(x, 2)}]] with the mixed input [[u = 0]] goes to [[f(x, 0) = ${fmt(fx, 3)}]], but the same mixture of the successors is [[${fmt(mix, 3)}]]: ${Math.abs(fx - mix) < 1e-9 ? "equal only at the endpoints" : `a gap of ${fmt(mix - fx, 3)}`}. For a nonlinear [[f]], [[f(\\sum \\lambda_r s_r, \\sum \\lambda_r u_r) \\ne \\sum \\lambda_r f(s_r, u_r)]] in general, so the one line of the proof that used linearity fails, and the hull of stored states need not be control invariant. (Here the mixed successor overshoots the true one, so this particular hull happens to be safe; the point is that nothing proves it.)`);
}
$("#nlT").addEventListener("input", drawNonlinear);

function drawAll() { drawHero(); drawInvariance(); drawNonlinear(); }
drawAll();
onRedraw(drawAll);
})();
