/* Advanced MPC and Learning MPC, chapter C4, Learning MPC racing: the demos.
   Everything shown here was computed in Python by code/c04_numbers.py and shipped in c04-data.js: the four
   seed laps, the twenty Learning MPC laps and the QP snapshots of one lap. The browser replays that data;
   it does not refit the model or solve the QPs. Needs ../kit/kit.js, mpc-lib.js and c04-data.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { legend } = window.MPC;
const D = window.C04_DATA, TR = D.track, L = TR.length, W = TR.halfWidth, NS = TR.s.length, DS = L / NS;

/* Frenet (s, e_y) to the plane: interpolate the centreline and its left normal between samples */
function toGlobal(s, ey) {
  s = ((s % L) + L) % L; const i = Math.min(NS - 1, Math.floor(s / DS)), j = (i + 1) % NS, t = (s - i * DS) / DS;
  const p = TR.xy[i], q = TR.xy[j], x = p[0] + t * (q[0] - p[0]), y = p[1] + t * (q[1] - p[1]);
  const tx = q[0] - p[0], ty = q[1] - p[1], n = Math.hypot(tx, ty);
  return [x - ey * ty / n, y + ey * tx / n];
}
const lapColor = (i, n) => { const t = n > 1 ? i / (n - 1) : 1; const a = [42, 120, 214], b = [235, 104, 52]; return `rgb(${a.map((v, k) => Math.round(v + t * (b[k] - v))).join(",")})`; };
const bounds = (() => { const xs = TR.outer.map(p => p[0]).concat(TR.inner.map(p => p[0])), ys = TR.outer.map(p => p[1]).concat(TR.inner.map(p => p[1])); return { x0: Math.min(...xs) - 0.3, x1: Math.max(...xs) + 0.3, y0: Math.min(...ys) - 0.3, y1: Math.max(...ys) + 0.3 }; })();

function drawTrack(fr) {
  const { ctx, X, Y } = fr;
  for (const [pts, style] of [[TR.outer, alpha("--ink", 0.5)], [TR.inner, alpha("--ink", 0.5)]]) { ctx.strokeStyle = style; ctx.lineWidth = 1.2; ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))); ctx.closePath(); ctx.stroke(); }
  ctx.strokeStyle = alpha("--ink", 0.18); ctx.setLineDash([3, 4]); ctx.beginPath(); TR.xy.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))); ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
  const f = toGlobal(0, -W), g = toGlobal(0, W); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; seg(ctx, X(f[0]), Y(f[1]), X(g[0]), Y(g[1]));
}
function drawLap(fr, X, color, width) { const { ctx } = fr; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); X.forEach((x, i) => { const p = toGlobal(x[4], x[5]); i ? ctx.lineTo(fr.X(p[0]), fr.Y(p[1])) : ctx.moveTo(fr.X(p[0]), fr.Y(p[1])); }); ctx.stroke(); }

/* ================================================================ the hero: laps, lap times, speed */
const hs = { lap: D.laps.length, all: true };
function drawHero() {
  const laps = D.laps, n = laps.length, sel = laps[hs.lap - 1];
  const cv = $("#mapC"), narrow = narrowOf(cv), span = bounds.x1 - bounds.x0, fr = frame(cv, narrow ? 1.0 : (bounds.x1 - bounds.x0) / (bounds.y1 - bounds.y0), { x: [bounds.x0, bounds.x1], y: [bounds.y0, bounds.y1], xticks: false, yticks: false, ygrid: false, pad: { l: 8, r: 8, t: 8, b: 8 } });
  drawTrack(fr);
  if (hs.all) { D.seeds.forEach(lap => drawLap(fr, lap.X, alpha("--muted", 0.45), 1)); laps.forEach((lap, i) => { if (i !== hs.lap - 1) { fr.ctx.globalAlpha = 0.35; drawLap(fr, lap.X, lapColor(i, n), 1); fr.ctx.globalAlpha = 1; } }); }
  drawLap(fr, sel.X, lapColor(hs.lap - 1, n), 2.6);
  legend(fr.ctx, ["grey: the four seed laps · blue to orange: Learning MPC laps 1 to " + n, "thick: the selected lap · black bar: the finish line"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* lap times */
  const cv2 = $("#timeC"), times = [...D.seeds.map(l => 0.1 * l.steps), ...laps.map(l => 0.1 * l.steps)], fr2 = frame(cv2, narrow ? 1.6 : 2.6, { x: [0, times.length], y: [0, 32], xlabel: "seed laps, then Learning MPC laps", yticks: [0, 10, 20, 30], xticks: false }), { ctx: c2, X: X2, Y: Y2 } = fr2;
  const bw = (X2(1) - X2(0)) * 0.7;
  times.forEach((t, i) => { const isSeed = i < D.seeds.length, k = i - D.seeds.length, lap = isSeed ? null : laps[k]; c2.fillStyle = isSeed ? alpha("--muted", 0.5) : (!lap.ok ? css("--warm") : lapColor(k, n)); c2.globalAlpha = (!isSeed && k === hs.lap - 1) ? 1 : 0.55; c2.fillRect(X2(i) + (X2(1) - X2(0) - bw) / 2, Y2(t), bw, Y2(0) - Y2(t)); c2.globalAlpha = 1; });
  c2.strokeStyle = css("--c4"); c2.setLineDash([5, 4]); seg(c2, fr2.pad.l, Y2(D.plan.time), fr2.w - fr2.pad.r, Y2(D.plan.time)); c2.setLineDash([]);
  legend(c2, ["lap time in seconds; orange bar: the lap that left the track", "dashed green: lab B3's centreline planner, " + fmt(D.plan.time, 2) + " s"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  /* speed along the lap */
  const cv3 = $("#speedC"), fr3 = frame(cv3, narrow ? 1.6 : 2.6, { x: [0, L], y: [0, 8.5], xlabel: "progress s along the lap (m)", yticks: [0, 2, 4, 6, 8] });
  plotLine(fr3, TR.s, D.plan.v, { color: css("--c4"), width: 1.4, dash: [5, 4] });
  const Xs = sel.X.slice(0, sel.steps + 1);
  plotLine(fr3, Xs.map(x => Math.min(L, x[4])), Xs.map(x => x[0]), { color: lapColor(hs.lap - 1, n), width: 2.2 });
  plotLine(fr3, Xs.map(x => Math.min(L, x[4])), Xs.map(x => 4 + 4 * x[5] / W), { color: alpha("--ink", 0.35), width: 1 });
  legend(fr3.ctx, ["coloured: v_x of the selected lap · dashed green: the planner's speed", "grey: lateral offset e_y, drawn around 4 (the track edges at 0 and 8)"], fr3.pad.l + 4, fr3.pad.t + 12, narrow);
  /* readouts and caption */
  const st = sel.stats;
  $("#hLapV").textContent = hs.lap; $("#hAll").setAttribute("aria-pressed", String(hs.all));
  $("#rTime").textContent = fmt(0.1 * sel.steps, 1) + " s"; $("#rEy").textContent = fmt(st.max_ey, 3) + " m" + (sel.ok ? "" : " (out)");
  $("#rVx").textContent = fmt(st.max_vx, 2) + " m/s"; $("#rQp").textContent = `${st.qps}, ${st.fallbacks} fallbacks`;
  const prev = hs.lap > 1 ? 0.1 * laps[hs.lap - 2].steps : 0.1 * Math.min(...D.seeds.map(l => l.steps));
  let text = `Lap ${hs.lap} of ${n}, a ${st.start} start: ${sel.steps} samples, [[${fmt(0.1 * sel.steps, 1)}]] s, ${fmt(prev - 0.1 * sel.steps, 1)} s ${prev > 0.1 * sel.steps ? "faster" : "slower"} than the lap before${hs.lap === 1 ? " (the fastest seed lap)" : ""}. The car reached ${fmt(st.max_vx, 2)} m/s and strayed at most ${fmt(st.max_ey, 3)} m from the centreline against a half-width of ${W} m`;
  text += sel.ok ? "; the lane and the input bounds were respected throughout. " : `: it LEFT the track. The QP was infeasible at ${st.fallbacks} of its ${st.qps} samples and the controller replayed the fastest stored lap's input instead, which brought it back; this lap was not added to the safe set. `;
  text += `${st.qps} QPs, about ${fmt(st.nit, 0)} interior-point iterations each` + (st.model_fallbacks ? `, and ${st.model_fallbacks} local fits with no stored transition inside the kernel bandwidth. ` : ". ");
  text += hs.lap === n ? "The learning has not stopped: every lap but one was faster than all before it, and the envelope of stored speeds keeps widening by about a tenth of a metre per second per lap." : "Move the slider to watch the line tighten and the speed rise lap by lap.";
  caption($("#heroStatus"), text);
}
$("#hLap").addEventListener("input", () => { hs.lap = +$("#hLap").value; drawHero(); });
$("#hAll").addEventListener("click", () => { hs.all = !hs.all; drawHero(); });
$("#hReset").addEventListener("click", () => { hs.lap = D.laps.length; hs.all = true; $("#hLap").value = hs.lap; drawHero(); });

/* ================================================================ one QP of one lap */
function drawQP() {
  const k = +$("#qStep").value, sn = D.snapshots[k], p0 = toGlobal(sn.x0[4], sn.x0[5]), R = 2.2;
  const cv = $("#qpC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.0 : 1.5, { x: [p0[0] - R * (narrow ? 1 : 1.5), p0[0] + R * (narrow ? 1 : 1.5)], y: [p0[1] - R, p0[1] + R], xticks: false, yticks: false, ygrid: false, pad: { l: 8, r: 8, t: 8, b: 8 } }), { ctx, X, Y } = fr;
  drawTrack(fr);
  const lap = D.laps[D.snapshotLap - 1];
  drawLap(fr, lap.X, alpha("--ink", 0.25), 1);
  /* the local safe set, shaded by cost-to-go */
  const Jmin = Math.min(...sn.J), Jmax = Math.max(...sn.J);
  sn.S.forEach((s, i) => { const p = toGlobal(s[4], s[5]), t = (sn.J[i] - Jmin) / Math.max(1e-9, Jmax - Jmin); ctx.fillStyle = `rgba(42,120,214,${0.9 - 0.7 * t})`; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), sn.lam[i] > 1e-6 ? 3 + 9 * sn.lam[i] : 2.6, 0, 2 * Math.PI); ctx.fill(); });
  /* the predicted horizon */
  ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2; ctx.beginPath(); sn.X.forEach((x, i) => { const p = toGlobal(x[4], x[5]); i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])); }); ctx.stroke();
  ctx.fillStyle = css("--warm"); sn.X.forEach(x => { const p = toGlobal(x[4], x[5]); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 2.2, 0, 2 * Math.PI); ctx.fill(); });
  const pt = toGlobal(sn.terminal[4], sn.terminal[5]); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(pt[0]), Y(pt[1]), 6, 0, 2 * Math.PI); ctx.stroke();
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(p0[0]), Y(p0[1]), 5, 0, 2 * Math.PI); ctx.fill();
  legend(ctx, ["black dot: the car · orange: the 14-step plan · circle: the terminal point", "blue dots: the 48 safe-set states, darker = fewer samples to the finish; large = active weight"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const act = sn.lam.map((l, i) => [l, i]).filter(([l]) => l > 1e-6).sort((a, b) => b[0] - a[0]);
  $("#qStepV").textContent = sn.step;
  caption($("#qpCap"), `Lap ${D.snapshotLap}, sample ${sn.step}: the car is at [[s = ${fmt(sn.x0[4], 2)}]] m, [[e_y = ${fmt(sn.x0[5], 3)}]] m, [[v_x = ${fmt(sn.x0[0], 2)}]] m/s. The QP chose the input [[(${fmt(sn.u[0], 3)}, ${fmt(sn.u[1], 3)})]] and a plan that ends at [[s_N = ${fmt(sn.terminal[4], 2)}]] m, written as ${act.length} stored states with weights ${act.map(([l]) => fmt(l, 3)).join(", ")} (Carathéodory promises at most seven in six dimensions). The learned terminal value is ${fmt(sn.value, 1)} samples to the finish line, between the ${fmt(Jmin, 0)} and ${fmt(Jmax, 0)} of the 48 candidates: the plan buys ${fmt(sn.J[act[0][1]] - sn.value, 1)} samples against the best single stored state it mixes. Everything in this figure was computed in Python; the page only draws it.`);
}
$("#qStep").max = D.snapshots.length - 1;
$("#qStep").addEventListener("input", drawQP);

function drawAll() { drawHero(); drawQP(); }
drawAll();
onRedraw(drawAll);
})();
