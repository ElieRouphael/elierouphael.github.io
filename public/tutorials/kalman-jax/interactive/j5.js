/* Kalman filters with JAX, chapter J5, loops as data: the live demos.
   1. Chapter 02's filter as a forward scan and the RTS smoother as a backward scan, computed here in float64
      on chapter 02's measurements and checked against JAX's float64 run (j5-data.js); step through either.
   2. The tables of measured costs: unrolled loop against scan, checkpointing, unroll, associative_scan.
   3. A parallel prefix over 16 steps, round by round.
   Needs ../kit/kit.js and j5-data.js. */
(() => {
"use strict";
const { $, $$, css, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const D = window.J5;
const grp = n => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
const f3 = v => (v < 0 ? "−" : "") + Math.abs(v).toFixed(3);

/* ============================================================ 1. filter and smoother */
const ZS = D.zs, T = ZS.length, DT = 0.1;
const QS = 0.1 * 0.1, Q00 = QS * DT ** 3 / 3, Q01 = QS * DT ** 2 / 2, Q11 = QS * DT, R = 0.25;
/* states as [p, v]; covariances as [p00, p01, p11] (symmetric) */
const xf = [], Pf = [], xp = [], Pp = [];
{
  let x = [0, 0], P = [1, 0, 1];
  for (let k = 0; k < T; k++) {
    const a00 = P[0] + DT * P[1], a01 = P[1] + DT * P[2];
    const pr = [x[0] + DT * x[1], x[1]];
    const PP = [a00 + DT * a01 + Q00, a01 + Q01, P[2] + Q11];
    const v = ZS[k] - pr[0], S = PP[0] + R, K0 = PP[0] / S, K1 = PP[1] / S;
    x = [pr[0] + K0 * v, pr[1] + K1 * v];
    const a = 1 - K0, b = -K1, bp = b * PP[0] + PP[1];
    P = [a * a * PP[0] + R * K0 * K0, a * bp + R * K0 * K1, bp * b + b * PP[1] + PP[2] + R * K1 * K1];
    xf.push(x); Pf.push(P); xp.push(pr); Pp.push(PP);
  }
}
const xs = new Array(T), Ps = new Array(T);
xs[T - 1] = xf[T - 1]; Ps[T - 1] = Pf[T - 1];
for (let k = T - 2; k >= 0; k--) {
  const P = Pf[k], N = Pp[k + 1];
  const FP = [P[0] + DT * P[1], P[1] + DT * P[2], P[1], P[2]];            // F P as [m00, m01, m10, m11]
  const det = N[0] * N[2] - N[1] * N[1];
  const Ni = [N[2] / det, -N[1] / det, N[0] / det];                      // inverse of the symmetric N
  // G = P F^T N^-1 = (F P)^T N^-1
  const G = [FP[0] * Ni[0] + FP[2] * Ni[1], FP[0] * Ni[1] + FP[2] * Ni[2], FP[1] * Ni[0] + FP[3] * Ni[1], FP[1] * Ni[1] + FP[3] * Ni[2]];
  const dx = [xs[k + 1][0] - xp[k + 1][0], xs[k + 1][1] - xp[k + 1][1]];
  xs[k] = [xf[k][0] + G[0] * dx[0] + G[1] * dx[1], xf[k][1] + G[2] * dx[0] + G[3] * dx[1]];
  const dP = [Ps[k + 1][0] - N[0], Ps[k + 1][1] - N[1], Ps[k + 1][2] - N[2]];
  // G dP G^T for symmetric dP
  const m00 = G[0] * dP[0] + G[1] * dP[1], m01 = G[0] * dP[1] + G[1] * dP[2];
  const m10 = G[2] * dP[0] + G[3] * dP[1], m11 = G[2] * dP[1] + G[3] * dP[2];
  Ps[k] = [P[0] + m00 * G[0] + m01 * G[1], P[1] + m00 * G[2] + m01 * G[3], P[2] + m10 * G[2] + m11 * G[3]];
}
let maxF = 0, maxS = 0;
for (let k = 0; k < T; k++) {
  maxF = Math.max(maxF, Math.abs(xf[k][0] - D.k64.filter_pos[k]));
  maxS = Math.max(maxS, Math.abs(xs[k][0] - D.k64.smooth_pos[k]));
}

const CODE = {
  fwd: `def step(<span class="c">carry</span>, <span class="i">measurement</span>):          # the repository's run_linear_kf
    <span class="c">x, P</span> = <span class="c">carry</span>
    x_pred, P_pred = linear_kf_predict(<span class="c">x, P</span>, F, Q)
    x_updated, P_updated, _, _ = linear_kf_update(x_pred, P_pred, <span class="i">measurement</span>, H, R)
    return <span class="c">(x_updated, P_updated)</span>, <span class="o">x_updated</span>

lax.scan(step, (x0, P0), measurements)       # <span class="c">carry</span> · <span class="i">input</span> · <span class="o">output</span>`,
  rev: `def bstep(<span class="c">carry</span>, <span class="i">inp</span>):                     # Rauch-Tung-Striebel, one step back
    <span class="c">xs_next, Ps_next</span> = <span class="c">carry</span>
    <span class="i">x, P, xp_next, Pp_next</span> = <span class="i">inp</span>
    G = jnp.linalg.solve(Pp_next, F @ P).T
    xs = x + G @ (<span class="c">xs_next</span> - xp_next)
    Ps = P + G @ (<span class="c">Ps_next</span> - Pp_next) @ G.T
    return <span class="c">(xs, Ps)</span>, <span class="o">(xs, Ps)</span>

lax.scan(bstep, (xf[-1], Pf[-1]), (xf[:-1], Pf[:-1], xp[1:], Pp[1:]), reverse=True)`,
};
const hs = { mode: "fwd", k: 60, timer: null };
$("#heroLegend").innerHTML = `<span><i style="background:var(--ink)"></i>true position</span><span><i class="dot" style="background:var(--muted)"></i>measurements</span>` +
  `<span><i style="background:var(--accent)"></i>filter</span><span><i class="box" style="background:var(--accent)"></i>filter ±2σ</span><span><i style="background:var(--warm)"></i>smoother</span>`;
const f4 = v => Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(1).replace("-", "−") : (v < 0 ? "−" : "") + Math.abs(v).toFixed(4);
const st = (x, P) => `x̂ = (${f3(x[0])}, ${f3(x[1])}), P = [[${f4(P[0])}, ${f4(P[1])}], [·, ${f4(P[2])}]]`;
function drawHero() {
  const k = hs.k, i = k - 1, W = 120;
  const lo = Math.max(1, Math.min(T - W + 1, k - 80)), hi = lo + W - 1;
  const idx = []; for (let t = lo; t <= hi; t++) idx.push(t);
  const ys = idx.flatMap(t => [D.true_pos[t - 1], ZS[t - 1]]);
  const ymin = Math.min(...ys), ymax = Math.max(...ys), pad = 0.08 * (ymax - ymin);
  const cv = $("#heroC");
  const fr = frame(cv, narrowOf(cv) ? 1.25 : 2.4, { x: [lo, hi], y: [ymin - pad, ymax + pad], xlabel: "step", ylabel: "position" });
  const { ctx, X, Y } = fr;
  const fEnd = hs.mode === "fwd" ? k : hi;
  const band = idx.filter(t => t <= fEnd);
  if (band.length > 1) {
    ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
    ctx.beginPath();
    band.forEach((t, j) => { const y = xf[t - 1][0] + 2 * Math.sqrt(Pf[t - 1][0]); j ? ctx.lineTo(X(t), Y(y)) : ctx.moveTo(X(t), Y(y)); });
    [...band].reverse().forEach(t => ctx.lineTo(X(t), Y(xf[t - 1][0] - 2 * Math.sqrt(Pf[t - 1][0]))));
    ctx.closePath(); ctx.fillStyle = alpha("--accent", 0.13); ctx.fill(); ctx.restore();
  }
  plotLine(fr, idx, idx.map(t => D.true_pos[t - 1]), { color: css("--ink"), width: 1.4, dash: [5, 4] });
  ctx.fillStyle = alpha("--muted", 0.75);
  idx.forEach(t => { ctx.beginPath(); ctx.arc(X(t), Y(ZS[t - 1]), 1.8, 0, 7); ctx.fill(); });
  plotLine(fr, band, band.map(t => xf[t - 1][0]), { color: hs.mode === "fwd" ? css("--accent") : alpha("--accent", 0.55), width: 2 });
  if (hs.mode === "rev") {
    const sm = idx.filter(t => t >= k);
    plotLine(fr, sm, sm.map(t => xs[t - 1][0]), { color: css("--warm"), width: 2.4 });
  }
  ctx.strokeStyle = alpha("--ink", 0.5); ctx.setLineDash([2, 3]);
  seg(ctx, X(k), fr.pad.t, X(k), Y(fr.y0)); ctx.setLineDash([]);

  $("#stepCode").innerHTML = CODE[hs.mode];
  $("#kv").textContent = String(k);
  $("#kIn").value = k;
  $$("#modeSeg [data-mode]").forEach(b => b.setAttribute("aria-pressed", b.dataset.mode === hs.mode));
  if (hs.mode === "fwd") {
    $("#lCarryIn").textContent = `carry in: (x, P) after step ${k - 1}`;
    $("#rCarryIn").textContent = k === 1 ? "x̂ = (0, 0), P = I: the initial carry (x0, P0)" : st(xf[i - 1], Pf[i - 1]);
    $("#lInput").textContent = `input: measurement z_${k}`;
    $("#rInput").textContent = f3(ZS[i]);
    $("#lCarryOut").textContent = `carry out, and output y_${k}`;
    $("#rCarryOut").textContent = st(xf[i], Pf[i]);
    $("#rStack").textContent = `${k} output${k === 1 ? "" : "s"}: an array of shape (${k}, 2) so far, (1000, 2) at the end`;
  } else {
    $("#lCarryIn").textContent = k === T ? "initial carry" : `carry in: smoothed (x, P) at step ${k + 1}`;
    $("#rCarryIn").textContent = k === T ? "the filter's last estimate, already smoothed" : st(xs[i + 1], Ps[i + 1]);
    $("#lInput").textContent = k === T ? "input" : `input: filtered (x, P) at ${k}, prediction for ${k + 1}`;
    $("#rInput").textContent = k === T ? "none: the backward scan starts here" : `x̂_${k}|${k} = (${f3(xf[i][0])}, ${f3(xf[i][1])}), x̂_${k + 1}|${k} = (${f3(xp[i + 1][0])}, ${f3(xp[i + 1][1])})`;
    $("#lCarryOut").textContent = `carry out, and output: smoothed at ${k}`;
    $("#rCarryOut").textContent = st(xs[i], Ps[i]);
    $("#rStack").textContent = k === T ? `1 smoothed step, the last; stacked in time order at the end`
      : `${T - k + 1} smoothed steps, from ${T} back to ${k}; stacked in time order at the end`;
  }
  $("#rCheck").textContent = `filter ${maxF.toExponential(0)}, smoother ${maxS.toExponential(0)} (largest position difference)`;
  let msg;
  if (hs.mode === "fwd") {
    msg = `<b>Step ${k} of the forward scan.</b> The step function receives the carry, the filter's belief after step ${k - 1}, and one input, measurement ${f3(ZS[i])}. It returns a new carry, which the next step will receive, and one output, which <code>scan</code> stacks. Only the carry crosses from step to step: six numbers, whatever the length of the sequence. The band is the filter's ±2σ, from the carry's [[P]].`;
  } else {
    const gain = Math.abs(xf[i][0] - D.true_pos[i]) - Math.abs(xs[i][0] - D.true_pos[i]);
    msg = k === T
      ? `<b>Step ${T}: the backward scan starts here</b> (<code>reverse=True</code>). There are no later measurements, so the smoothed estimate at the last step is the filter's own, and it becomes the initial carry. Step back to see the carry flow from the future. Over all ${T} steps the smoother's position error is ${D.smooth.rmse_smoother.toFixed(3)} against the filter's ${D.smooth.rmse_filter.toFixed(3)}.`
      : `<b>Step ${k} of the backward scan</b> (<code>reverse=True</code>): the carry now flows from the future. The smoother corrects the filter's estimate at ${k} using the smoothed estimate at ${k + 1}, which already contains every later measurement. Here it moves the position estimate from ${f3(xf[i][0])} to ${f3(xs[i][0])}, ${gain > 0 ? "closer to" : "away from"} the true ${f3(D.true_pos[i])}. Over all ${T} steps the smoother's position error is ${D.smooth.rmse_smoother.toFixed(3)} against the filter's ${D.smooth.rmse_filter.toFixed(3)}.`;
  }
  caption($("#heroStatus"), msg);
}
function stopPlay() { if (hs.timer) { clearTimeout(hs.timer); hs.timer = null; $("#playBtn").textContent = "Play"; } }
function move(d) { hs.k = Math.max(1, Math.min(T, hs.k + d)); drawHero(); }
$("#prevBtn").addEventListener("click", () => { stopPlay(); move(-1); });
$("#nextBtn").addEventListener("click", () => { stopPlay(); move(1); });
$("#kIn").addEventListener("input", e => { stopPlay(); hs.k = +e.target.value; drawHero(); });
$("#playBtn").addEventListener("click", () => {
  if (hs.timer) { stopPlay(); return; }
  $("#playBtn").textContent = "Pause";
  const d = hs.mode === "fwd" ? 1 : -1;
  const tick = () => {
    if ((d > 0 && hs.k >= T) || (d < 0 && hs.k <= 1)) { stopPlay(); return; }
    move(d); hs.timer = setTimeout(tick, 60);
  };
  tick();
});
$$("#modeSeg [data-mode]").forEach(b => b.addEventListener("click", () => {
  stopPlay(); hs.mode = b.dataset.mode;
  if (hs.mode === "rev" && hs.k < 200) hs.k = 160;
  drawHero();
}));
drawHero();
onRedraw(drawHero);

/* ============================================================ 2. the tables */
const ms = v => v >= 10000 ? `${(v / 1000).toFixed(0)} s` : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : v >= 10 ? `${v.toFixed(0)} ms` : v >= 1 ? `${v.toFixed(1)} ms` : `${v.toFixed(2)} ms`;
const bytes = b => b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} kB`;
const LO = Math.log10(30), HI = Math.log10(300000);
const bw = v => `${(100 * Math.max(0, Math.min(1, (Math.log10(v) - LO) / (HI - LO)))).toFixed(1)}%`;
$("#loopBody").innerHTML = D.unroll.map(r => {
  const tl = r.loop.trace_ms + r.loop.lower_ms + r.loop.compile_ms, ts = r.scan.trace_ms + r.scan.lower_ms + r.scan.compile_ms;
  return `<tr><td class="n">${grp(r.T)}</td><td class="n">${grp(r.loop.eqns)} / ${r.scan.eqns}</td><td class="n">${ms(tl)} / ${ms(ts)}</td>` +
    `<td class="n">${ms(r.loop.run_us / 1000)} / ${ms(r.scan.run_us / 1000)}</td>` +
    `<td class="barc"><div class="tb loop" style="width:${bw(tl)}"></div><div class="tb scan" style="width:${bw(ts)}"></div></td></tr>`;
}).join("");
const memT = [...new Set(D.memory.map(r => r.T))];
$("#memBody").innerHTML = memT.map(t => {
  const a = D.memory.find(r => r.T === t && !r.checkpoint), b = D.memory.find(r => r.T === t && r.checkpoint);
  return `<tr><td class="n">${grp(t)}</td><td class="n">${bytes(a.temp_bytes)}</td><td class="n">${bytes(b.temp_bytes)}</td><td class="n">${ms(a.ms)}</td><td class="n">${ms(b.ms)}</td></tr>`;
}).join("");
$("#unBody").innerHTML = D.unroll_param.map(r => `<tr><td class="n">${r.unroll}</td><td class="n">${ms(r.first_ms)}</td><td class="n">${ms(r.run_ms)}</td></tr>`).join("");
$("#asBody").innerHTML = D.assoc.timing.map(r => `<tr><td class="n">${grp(r.T)}</td><td class="n">${ms(r.scan_ms)}</td><td class="n">${ms(r.assoc_ms)}</td></tr>`).join("");
const S = D.smooth;
$("#rmseBody").innerHTML = `<tr><td>The measurements themselves</td><td class="n">${S.rmse_meas.toFixed(3)}</td><td class="n">not measured</td></tr>` +
  `<tr><td>The filter (forward scan)</td><td class="n">${S.rmse_filter.toFixed(3)}</td><td class="n">${S.rmse_filter_vel.toFixed(3)}</td></tr>` +
  `<tr><td>The smoother (forward and backward scans)</td><td class="n">${S.rmse_smoother.toFixed(3)}</td><td class="n">${S.rmse_smoother_vel.toFixed(3)}</td></tr>`;

/* ============================================================ 3. the prefix tree */
const N = 16, ROUNDS = 4;
const tr = { r: 0 };
function drawTree() {
  const svg = $("#treeSvg");
  const NS = "http://www.w3.org/2000/svg";
  svg.innerHTML = "";
  const mk = (tag, at) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); svg.appendChild(e); return e; };
  const cx = i => 10 + i * 40, cy = r => 8 + r * 44;
  let work = 0;
  for (let r = 0; r <= tr.r; r++) {
    const span = 2 ** r;                                   // after round r, element i covers max(0, i - span + 1)..i
    if (r > 0) {
      const off = 2 ** (r - 1);
      for (let i = off; i < N; i++) {
        work++;
        mk("line", { x1: cx(i - off) + 18, y1: cy(r - 1) + 26, x2: cx(i) + 18, y2: cy(r) });
      }
    }
    for (let i = 0; i < N; i++) {
      const from = Math.max(0, i - span + 1);
      mk("rect", { x: cx(i), y: cy(r), width: 36, height: 26, rx: 5, class: `cell${from === 0 ? " full" : ""}` });
      const t = mk("text", { x: cx(i) + 18, y: cy(r) + 17, "text-anchor": "middle" });
      t.textContent = from === i ? String(i + 1) : `${from + 1}–${i + 1}`;
    }
  }
  $("#tRounds").textContent = String(tr.r);
  $("#tWork").textContent = String(work);
  $("#treePrev").disabled = tr.r === 0; $("#treeNext").disabled = tr.r === ROUNDS;
  const msg = tr.r === 0
    ? "Each element starts as one step's affine map. Press <b>round ▶</b>: in each round, every element combines with the element a power of two to its left, and all of a round's combinations are independent, so they can run at once."
    : tr.r < ROUNDS
      ? `After round ${tr.r}, element [[k]] holds the composition of the last ${2 ** tr.r} maps up to [[k]]; the shaded ones already reach back to step 1, so they are final outputs.`
      : `After 4 rounds ([[\\log_2 16]]), every element holds the composition of all maps from step 1: every output is computed. That took ${work} combinations against the sequential loop's 15, but only 4 rounds against 15 steps. This simple scheme does [[T \\log_2 T]] work; JAX's does about [[2T]] in about [[2 \\log_2 T]] rounds.`;
  caption($("#treeCap"), msg);
}
$("#treeNext").addEventListener("click", () => { tr.r = Math.min(ROUNDS, tr.r + 1); drawTree(); });
$("#treePrev").addEventListener("click", () => { tr.r = Math.max(0, tr.r - 1); drawTree(); });
$("#treeReset").addEventListener("click", () => { tr.r = 0; drawTree(); });
drawTree();
})();
