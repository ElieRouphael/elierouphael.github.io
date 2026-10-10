/* TrustDrive chapter 02, LQR lane keeping: the live demos.
   The Riccati solver, the closed-loop simulation and the scenarios follow the notebook
   exactly (Q = diag(1, 5), R = 8, steering limited to 0.6 rad). Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf } = window.Kit;
const L = 2.7, TS = 0.05, V0 = 15, DMAX = 0.6, Q0 = [1, 5], R0 = 8;
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const num = x => String(+x.toFixed(1));
const snapR = s => +(10 ** s).toPrecision(2);                 // 0.903 -> 8, keeps the slider's values tidy
function seg(ctx, x0, y0, x1, y1) { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }

/* ---------------- the linear model and the Riccati equation (2 states, 1 input) ---------------- */
const model = v => ({ A: [[1, v * TS], [0, 1]], b: [0, v * TS / L], e: [0, -v * TS] });
/* One Riccati step from P: returns the gain K(P) and the next P. */
function riccati(A, b, Q, r, P) {
  const Pb = [P[0][0] * b[0] + P[0][1] * b[1], P[1][0] * b[0] + P[1][1] * b[1]];
  const s = r + b[0] * Pb[0] + b[1] * Pb[1];
  const u = [A[0][0] * Pb[0] + A[1][0] * Pb[1], A[0][1] * Pb[0] + A[1][1] * Pb[1]];   // A^T P b
  const PA = [[P[0][0] * A[0][0] + P[0][1] * A[1][0], P[0][0] * A[0][1] + P[0][1] * A[1][1]],
              [P[1][0] * A[0][0] + P[1][1] * A[1][0], P[1][0] * A[0][1] + P[1][1] * A[1][1]]];
  const ATPA = [[A[0][0] * PA[0][0] + A[1][0] * PA[1][0], A[0][0] * PA[0][1] + A[1][0] * PA[1][1]],
                [A[0][1] * PA[0][0] + A[1][1] * PA[1][0], A[0][1] * PA[0][1] + A[1][1] * PA[1][1]]];
  const next = [[Q[0][0] + ATPA[0][0] - u[0] * u[0] / s, Q[0][1] + ATPA[0][1] - u[0] * u[1] / s],
                [Q[1][0] + ATPA[1][0] - u[1] * u[0] / s, Q[1][1] + ATPA[1][1] - u[1] * u[1] / s]];
  return { K: [u[0] / s, u[1] / s], next };
}
/* The notebook's dlqr: iterate to a fixed point (tolerance 1e-14), then the gain and the residual. */
function dlqr(A, b, Q, r) {
  let P = Q, iters = 0;
  for (; iters < 1000; iters++) {
    const { next } = riccati(A, b, Q, r, P);
    const diff = Math.max(...[0, 1].flatMap(i => [0, 1].map(j => Math.abs(next[i][j] - P[i][j]))));
    P = next;
    if (diff < 1e-14) break;
  }
  const { K, next } = riccati(A, b, Q, r, P);
  const residual = Math.max(...[0, 1].flatMap(i => [0, 1].map(j => Math.abs(next[i][j] - P[i][j]))));
  return { K, P, iters: iters + 1, residual };
}
const diagQ = (q1, q2) => [[q1, 0], [0, q2]];
/* eigenvalues of A - bK, as {re, im} pairs */
function poles(A, b, K) {
  const M = [[A[0][0] - b[0] * K[0], A[0][1] - b[0] * K[1]], [A[1][0] - b[1] * K[0], A[1][1] - b[1] * K[1]]];
  const tr = M[0][0] + M[1][1], det = M[0][0] * M[1][1] - M[0][1] * M[1][0], disc = tr * tr / 4 - det;
  return disc >= 0 ? [{ re: tr / 2 + Math.sqrt(disc), im: 0 }, { re: tr / 2 - Math.sqrt(disc), im: 0 }]
    : [{ re: tr / 2, im: Math.sqrt(-disc) }, { re: tr / 2, im: -Math.sqrt(-disc) }];
}
const radius = ps => Math.max(...ps.map(p => Math.hypot(p.re, p.im)));
const fmtPoles = ps => (ps[0].im ? `${fmt(ps[0].re, 3)} ± ${fmt(Math.abs(ps[0].im), 3)}j` : `${fmt(ps[0].re, 3)} and ${fmt(ps[1].re, 3)}`);
const fmtK = K => `[${fmt(K[0], 3)}, ${fmt(K[1], 3)}]`;

/* The notebook's closed loop: nonlinear plant, steering limited to 0.6 rad.
   Records the state before each step, as sim_recovery does. */
function simulate(K, { v = V0, n = 200, x0 = [0.5, 0], kappa = () => 0, ff = false } = {}) {
  let x = x0.slice();
  const out = { t: [], ey: [], epsi: [], d: [], dff: [] };
  for (let k = 0; k < n; k++) {
    const kap = kappa(k * TS), dff = ff ? Math.atan(L * kap) : 0;
    const d = clamp(dff - (K[0] * x[0] + K[1] * x[1]), -DMAX, DMAX);
    out.t.push(k * TS); out.ey.push(x[0]); out.epsi.push(x[1]); out.d.push(d); out.dff.push(dff);
    x = [x[0] + v * Math.sin(x[1]) * TS, x[1] + (v / L * Math.tan(d) - v * kap) * TS];
  }
  out.final = x;
  return out;
}
const rms = a => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);

/* =====================================================================
   THE DEMO: RECOVER, THEN TAKE A BEND
   ===================================================================== */
const BEND = [4, 11];
function hero() {
  const r = snapR(+$("#hR").value), kap = +$("#hK").value, v = +$("#hV").value, ff = $("#hFF").checked, sched = $("#hSched").checked;
  $("#hRV").textContent = `${r}${r === R0 ? " (the course's)" : ""}`;
  $("#hKV").textContent = kap ? `${fmt(kap, 3)} per m (radius ${Math.round(1 / kap)} m)` : "0 (no bend)";
  $("#hVV").textContent = `${num(v)} m/s (${Math.round(v * 3.6)} km/h)`;
  const Q = diagQ(...Q0), mDesign = model(sched ? v : V0), mTrue = model(v);
  const { K } = dlqr(mDesign.A, mDesign.b, Q, r);
  const ps = poles(mTrue.A, mTrue.b, K);
  const sim = simulate(K, { v, n: 300, kappa: t => (t >= BEND[0] && t < BEND[1] ? kap : 0), ff });
  return { r, kap, v, ff, sched, K, ps, sim };
}
let heroRun = null;
function updateHero() { heroRun = hero(); drawHero(); heroStatus(); }
function drawHero() {
  const { sim, ff } = heroRun, narrow = narrowOf($("#hOff"));
  const m = Math.max(0.6, 1.15 * Math.max(...sim.ey.map(Math.abs)));
  const fr = frame($("#hOff"), narrow ? 1.5 : 2.6, { x: [0, 15], y: [-m, m], xlabel: "time (s)", ylabel: "offset (m)", yfmt: v => fmt(v, 1) });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--muted", 0.12); ctx.fillRect(X(BEND[0]), fr.pad.t, X(BEND[1]) - X(BEND[0]), fr.ih);
  ctx.font = "600 11px " + css("--font-ui"); ctx.fillStyle = css("--muted"); ctx.textAlign = "center";
  ctx.fillText(heroRun.kap ? "bend" : "no bend", (X(BEND[0]) + X(BEND[1])) / 2, fr.pad.t + 13);
  ctx.strokeStyle = alpha("--ink", 0.4); ctx.lineWidth = 1; seg(ctx, X(0), Y(0), X(15), Y(0));
  plotLine(fr, sim.t, sim.ey, { color: css("--path"), width: 2.4 });
  const dm = Math.max(0.1, 1.15 * Math.max(...sim.d.map(Math.abs)));
  const gr = frame($("#hSteer"), narrow ? 2.2 : 4, { x: [0, 15], y: [-dm, dm], xlabel: "time (s)", ylabel: "rad", yfmt: v => fmt(v, 2) });
  gr.ctx.fillStyle = alpha("--muted", 0.12); gr.ctx.fillRect(gr.X(BEND[0]), gr.pad.t, gr.X(BEND[1]) - gr.X(BEND[0]), gr.ih);
  gr.ctx.strokeStyle = alpha("--ink", 0.4); gr.ctx.lineWidth = 1; seg(gr.ctx, gr.X(0), gr.Y(0), gr.X(15), gr.Y(0));
  if (dm > DMAX) { gr.ctx.strokeStyle = css("--geom"); gr.ctx.setLineDash([3, 3]); [DMAX, -DMAX].forEach(d => seg(gr.ctx, gr.X(0), gr.Y(d), gr.X(15), gr.Y(d))); gr.ctx.setLineDash([]); }
  plotLine(gr, sim.t, sim.d, { color: css("--path"), width: 2 });
  if (ff) plotLine(gr, sim.t, sim.dff, { color: css("--warm"), width: 1.8, dash: [6, 4] });   // on top: on the bend the two coincide
  // readouts
  const { K, ps } = heroRun, iBend = Math.round(BEND[1] / TS) - 1;
  $("#rK").textContent = fmtK(K);
  $("#rPoles").textContent = `${fmtPoles(ps)} (|·| ${fmt(radius(ps), 3)})`;
  $("#rBend").textContent = heroRun.kap ? `${fmt(sim.ey[iBend], 3)} m` : "–";
  $("#rRms").textContent = `${fmt(rms(sim.ey), 3)} m`;
  $("#rEff").textContent = `${fmt(rms(sim.d), 3)} rad`;
}
function heroStatus() {
  const { r, kap, v, ff, sched, K, ps, sim } = heroRun;
  const iBend = Math.round(BEND[0] / TS);
  let last = -1;
  for (let k = 0; k < iBend; k++) if (Math.abs(sim.ey[k]) > 0.05) last = k;
  const settle = last < iBend - 1 ? (last + 1) * TS : null;
  const bendOff = sim.ey[Math.round(BEND[1] / TS) - 1];
  let text = `With [[R = ${r}]] the gain is [[K = ${fmtK(K)}]]` +
    (!sched && v !== V0 ? `, designed for 15 m/s although the car drives at ${num(v)} m/s` : "") + ". " +
    (radius(ps) < 1 ? (settle !== null ? `The car returns from 0.5 m to within 5 cm of the centre in <b>${fmt(settle, 2)} s</b>. ` : "The car has not settled within 5 cm before the bend. ")
      : "The closed loop is unstable at this speed: the poles are outside the unit circle. ");
  if (kap) {
    const need = Math.atan(L * kap);
    text += ff
      ? `On the bend, the feed-forward supplies the ${fmt(need, 3)} rad the curve needs, and the offset stays at <b>${fmt(Math.abs(bendOff) * 100, 1)} cm</b>.`
      : `On the bend, pure feedback can only steer once an error exists: the car settles <b>${fmt(Math.abs(bendOff) * 100, 0)} cm</b> towards the outside of the curve, as predicted by [[-\\frac{\\arctan(L\\kappa)}{K_1}]].`;
  }
  caption($("#hStatus"), text);
}
["#hR", "#hK", "#hV"].forEach(id => $(id).addEventListener("input", updateHero));
["#hFF", "#hSched"].forEach(id => $(id).addEventListener("change", updateHero));

/* =====================================================================
   PART 3: THE RECURSION, THE GAIN AND THE POLES
   ===================================================================== */
const QY = [0.1, 0.2, 0.5, 1, 2, 5, 10], QP = [0.5, 1, 2, 5, 10, 20, 50], RR = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 200];
function drawRic() {
  const q1 = QY[+$("#qY").value], q2 = QP[+$("#qP").value], r = RR[+$("#qR").value];
  $("#qYV").textContent = q1; $("#qPV").textContent = q2; $("#qRV").textContent = r;
  const { A, b } = model(V0), Q = diagQ(q1, q2);
  const sol = dlqr(A, b, Q, r);
  // the backward recursion from P_N = Q, 120 steps, as in the notebook's figure
  let P = Q; const g1 = [], g2 = [];
  for (let i = 0; i < 120; i++) { const st = riccati(A, b, Q, r, P); g1.push(st.K[0]); g2.push(st.K[1]); P = st.next; }
  const within = g1.findIndex((k, i) => Math.abs(k - sol.K[0]) < 1e-3 && Math.abs(g2[i] - sol.K[1]) < 1e-3);
  const aspect = narrowOf($("#ricGain")) ? 1.3 : 1.15;
  const steps = g1.map((_, i) => i);
  const fr = frame($("#ricGain"), aspect, { x: [0, 119], y: [0, Math.max(...g1, ...g2) * 1.12], xlabel: "backward step", ylabel: "gain", yfmt: v => fmt(v, 1) });
  plotLine(fr, steps, g1, { color: css("--path"), width: 2 });
  plotLine(fr, steps, g2, { color: css("--warm"), width: 2 });
  // poles in the unit circle
  const ps = poles(A, b, sol.K);
  const gr = frame($("#ricPoles"), aspect, { x: [-1.25, 1.25], y: [-1.15, 1.15], xticks: [-1, 0, 1], yticks: [-1, 0, 1], xfmt: v => String(v), yfmt: v => String(v), xlabel: "real part" });
  const { ctx, X, Y } = gr;
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; ctx.beginPath();
  ctx.ellipse(X(0), Y(0), X(1) - X(0), Y(0) - Y(1), 0, 0, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = alpha("--muted", 0.5); seg(ctx, X(0), Y(-1.15), X(0), Y(1.15));
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(1), Y(0), 6, 0, 2 * Math.PI); ctx.stroke();   // open loop: a double pole at 1
  ctx.fillStyle = css("--geom"); ps.forEach(p => { ctx.beginPath(); ctx.arc(X(p.re), Y(p.im), 5.5, 0, 2 * Math.PI); ctx.fill(); });
  caption($("#ricCap"),
    `[[K = ${fmtK(sol.K)}]]. The recursion is within 0.001 of it after ${within} backward steps; the solver needs ${sol.iters} iterations to reach a tolerance of 1e-14, and the residual of the DARE is then ${sol.residual.toExponential(1)}. ` +
    `The closed-loop poles are ${fmtPoles(ps)}, with spectral radius <b>${fmt(radius(ps), 3)}</b>: inside the unit circle, so the loop is stable. ` +
    (r === R0 && q1 === 1 && q2 === 5 ? "These are the notebook's numbers." : r > R0 ? "Steering is dearer: smaller gains, poles closer to the circle, a slower loop." : "Steering is cheaper or errors dearer: larger gains, poles further inside, a faster loop."));
}
["#qY", "#qP", "#qR"].forEach(id => $(id).addEventListener("input", drawRic));

/* =====================================================================
   PART 4: THE COST FRONTIER (the notebook's sweep of 25 values of R)
   ===================================================================== */
const SWEEP = Array.from({ length: 25 }, (_, i) => 0.5 * (400 ** (i / 24)));
const recover = r => { const { A, b } = model(V0); const K = dlqr(A, b, diagQ(...Q0), r).K; const s = simulate(K); return { K, s, rmse: rms(s.ey), eff: rms(s.d) }; };
const frontier = SWEEP.map(recover), star = recover(R0);
function drawFront() {
  const r = snapR(+$("#fR").value), cur = recover(r);
  $("#fRV").textContent = `${r}${r === R0 ? " (the course's)" : ""}`;
  const aspect = narrowOf($("#frontC")) ? 1.25 : 1.15;
  const fr = frame($("#frontC"), aspect, { x: [0, 0.034], y: [0.055, 0.12], xlabel: "steering effort, RMS (rad)", ylabel: "RMS offset (m)", xfmt: v => fmt(v, 2), yfmt: v => fmt(v, 2), xticks: [0, 0.01, 0.02, 0.03] });
  plotLine(fr, frontier.map(p => p.eff), frontier.map(p => p.rmse), { color: css("--path"), width: 2 });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = css("--path"); frontier.forEach(p => { ctx.beginPath(); ctx.arc(X(p.eff), Y(p.rmse), 2.5, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = css("--geom"); ctx.beginPath(); ctx.arc(X(star.eff), Y(star.rmse), 6, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(cur.eff), Y(cur.rmse), 8, 0, 2 * Math.PI); ctx.stroke();
  const gr = frame($("#frontRec"), aspect, { x: [0, 10], y: [-0.15, 0.55], xlabel: "time (s)", ylabel: "offset (m)", yfmt: v => fmt(v, 1) });
  plotLine(gr, star.s.t, star.s.ey, { color: css("--geom"), width: 1.6, dash: [6, 4] });
  plotLine(gr, cur.s.t, cur.s.ey, { color: css("--path"), width: 2.2 });
  caption($("#frontCap"),
    `With [[R = ${r}]], [[K = ${fmtK(cur.K)}]]: RMS offset <b>${fmt(cur.rmse, 3)} m</b> for an RMS steering of <b>${fmt(cur.eff, 4)} rad</b> (black ring). ` +
    `The course's [[R = 8]] (red) gives ${fmt(star.rmse, 3)} m for ${fmt(star.eff, 4)} rad. ` +
    (r < R0 ? "Cheaper steering: the car snaps back faster but works the wheel harder." : r > R0 ? "Dearer steering: gentler on the wheel, slower back to the centre." : "The knee of the curve."));
}
$("#fR").addEventListener("input", drawFront);

/* =====================================================================
   PART 5: THE OFFSET ON A BEND, AND FEED-FORWARD
   ===================================================================== */
const K0 = (() => { const { A, b } = model(V0); return dlqr(A, b, diagQ(...Q0), R0).K; })();
function drawFF() {
  const kap = +$("#ffK").value;
  $("#ffKV").textContent = kap ? `${fmt(kap, 3)} per m (${kap > 0 ? "left" : "right"} bend, radius ${Math.round(Math.abs(1 / kap))} m)` : "0 (straight)";
  const opts = { n: 301, x0: [0, 0], kappa: () => kap };
  const fb = simulate(K0, opts), both = simulate(K0, { ...opts, ff: true });
  const pred = -L * kap / K0[0];                                              // the linear closed loop's steady state
  const m = Math.max(0.1, 1.2 * Math.max(...fb.ey.map(Math.abs), ...both.ey.map(Math.abs)));
  const cv = $("#ffC");
  const fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [0, 15], y: [-m, m], xlabel: "time (s)", ylabel: "offset (m)", yfmt: v => fmt(v, 2) });
  fr.ctx.strokeStyle = alpha("--ink", 0.4); fr.ctx.lineWidth = 1; seg(fr.ctx, fr.X(0), fr.Y(0), fr.X(15), fr.Y(0));
  if (kap) { fr.ctx.strokeStyle = css("--muted"); fr.ctx.setLineDash([2, 3]); seg(fr.ctx, fr.X(0), fr.Y(pred), fr.X(15), fr.Y(pred)); fr.ctx.setLineDash([]); }
  plotLine(fr, fb.t, fb.ey, { color: css("--warm"), width: 2, dash: [6, 4] });
  plotLine(fr, both.t, both.ey, { color: css("--path"), width: 2.4 });
  caption($("#ffCap"),
    !kap ? "On a straight road there is nothing to cancel: both controllers keep the car on the centre line."
    : `Feedback only settles at <b>${fmt(fb.ey[fb.ey.length - 1], 3)} m</b>, on the outside of the bend; the linear prediction [[\\big(I - (A - BK)\\big)^{-1}E\\kappa]] gives ${fmt(pred, 3)} m. ` +
      `With the feed-forward [[\\arctan(L\\kappa) = ${fmt(Math.atan(L * kap), 4)}]] rad added, the offset stays at <b>${fmt(Math.abs(both.ey[both.ey.length - 1]), 4)} m</b>: the bend is cancelled before any error appears.`);
}
$("#ffK").addEventListener("input", drawFF);

/* =====================================================================
   PART 6: GAIN SCHEDULING
   ===================================================================== */
const SPEEDS = Array.from({ length: 41 }, (_, i) => 5 + 0.5 * i);
const schedule = SPEEDS.map(v => {
  const { A, b } = model(v), K = dlqr(A, b, diagQ(...Q0), R0).K;
  return { v, K, rho: radius(poles(A, b, K)), rhoFixed: radius(poles(A, b, K0)) };
});
function drawSched() {
  const v = +$("#sV").value, cur = schedule.find(s => s.v === v);
  $("#sVV").textContent = `${num(v)} m/s (${Math.round(v * 3.6)} km/h)`;
  const aspect = narrowOf($("#schedK")) ? 1.3 : 1.15;
  const fr = frame($("#schedK"), aspect, { x: [5, 25], y: [0, 1.8], xlabel: "speed (m/s)", ylabel: "gain", yfmt: v => fmt(v, 1) });
  plotLine(fr, SPEEDS, schedule.map(s => s.K[0]), { color: css("--path"), width: 2.2 });
  plotLine(fr, SPEEDS, schedule.map(s => s.K[1]), { color: css("--warm"), width: 2.2 });
  const mark = (f, val, col) => { f.ctx.fillStyle = css(col); f.ctx.beginPath(); f.ctx.arc(f.X(v), f.Y(val), 5, 0, 2 * Math.PI); f.ctx.fill(); };
  fr.ctx.strokeStyle = alpha("--ink", 0.4); fr.ctx.lineWidth = 1; seg(fr.ctx, fr.X(v), fr.Y(0), fr.X(v), fr.pad.t);
  mark(fr, cur.K[0], "--path"); mark(fr, cur.K[1], "--warm");
  const gr = frame($("#schedRho"), aspect, { x: [5, 25], y: [0.5, 1.05], xlabel: "speed (m/s)", yfmt: v => fmt(v, 1) });
  gr.ctx.strokeStyle = css("--geom"); gr.ctx.lineWidth = 1.5; gr.ctx.setLineDash([5, 4]); seg(gr.ctx, gr.X(5), gr.Y(1), gr.X(25), gr.Y(1)); gr.ctx.setLineDash([]);
  plotLine(gr, SPEEDS, schedule.map(s => s.rhoFixed), { color: css("--warm"), width: 1.8, dash: [6, 4] });
  plotLine(gr, SPEEDS, schedule.map(s => s.rho), { color: css("--path"), width: 2.2 });
  gr.ctx.strokeStyle = alpha("--ink", 0.4); gr.ctx.lineWidth = 1; seg(gr.ctx, gr.X(v), gr.Y(0.5), gr.X(v), gr.pad.t);
  mark(gr, cur.rho, "--path");
  caption($("#schedCap"),
    `At ${num(v)} m/s the scheduled gain is [[K = ${fmtK(cur.K)}]] and the spectral radius <b>${fmt(cur.rho, 3)}</b>` +
    (v === V0 ? ": this is the course's design speed." : `; keeping the 15 m/s gain instead gives ${fmt(cur.rhoFixed, 3)}.`) +
    ` Both stay well below the stability limit of 1 at every speed.`);
}
$("#sV").addEventListener("input", drawSched);

/* =====================================================================
   start-up
   ===================================================================== */
updateHero(); drawRic(); drawFront(); drawFF(); drawSched();
onRedraw(() => { drawHero(); drawRic(); drawFront(); drawFF(); drawSched(); });
})();
