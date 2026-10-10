/* TrustDrive chapter 07, Shared human-vehicle control: the live demos.
   simulate() is the notebook's sim_conflict with the course's driver model (delay, workload,
   steering noise); without noise and workload it reproduces the notebook's numbers.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { makeRng } = window.Stats;

/* ---------------- the notebook's loop ---------------- */
const L = 2.7, TS = 0.05, V = 15, A12 = V * TS, B2 = V * TS / L;               // x_{k+1} = [[1, vT],[0, 1]] x_k + [0, vT/L] delta
const KA = [0.2833571171526833, 1.5001296696414237];                         // chapter 02's LQR: Q = diag(1, 5), R = 8
const KH = [0.12, 0.9], TAU = 4, SIGMA_H = 0.02, DMAX = 0.6, N = 400;         // the course's driver: 0.2 s delay
const authority = (I, W, avail = true) => clamp(avail ? I + 0.35 * W * (I - 0.5) : Math.max(I + 0.35 * W * (I - 0.5), 0.85), 0.05, 0.95);
const vRef = I => V * (0.4 + 0.6 * I);
/* the automation's effective share of the say on the offset (Part 6) */
const share = (lam, W = 0, avail = true) => { const kh = avail ? (1 - 0.5 * W) * KH[0] : 0; return lam * KA[0] / (lam * KA[0] + (1 - lam) * kh); };

/* One conflict: kind "driver" (the driver holds a wrong offset b) or "auto" (the perception is biased by b). */
function simulate({ lam, W = 0, avail = true, kind = "driver", b = 0.6, seed = 1, noise = true, tau = TAU, kh = KH, x0 = [0, 0], n = N }) {
  const rng = makeRng(seed), buf = Array.from({ length: tau + 1 }, () => x0.slice());   // the car has been at x0
  let x = x0.slice(); const ey = [];
  const bH = kind === "driver" ? b : 0, bA = kind === "auto" ? b : 0, g = 1 - 0.5 * W;
  for (let k = 0; k < n; k++) {
    buf.push(x.slice()); buf.shift(); const xd = buf[0];                       // the driver sees x from tau steps ago
    const dA = -(KA[0] * (x[0] + bA) + KA[1] * x[1]);
    let dH = 0;
    if (avail) dH = clamp(g * -(kh[0] * (xd[0] - bH) + kh[1] * xd[1]) + (noise ? rng.normal(0, SIGMA_H * (1 + 2 * W)) : 0), -DMAX, DMAX);
    const d = clamp(lam * dA + (1 - lam) * dH, -DMAX, DMAX);
    x = [x[0] + A12 * x[1], x[1] + B2 * d];
    ey.push(x[0]);
  }
  return ey;
}
const rmsTail = ey => Math.sqrt(ey.slice(-150).reduce((s, v) => s + v * v, 0) / 150);   // the notebook's last 7.5 s

/* Spectral radius of the delayed blended loop. With the driver acting on x_{k-tau}, the
   characteristic polynomial of the augmented state is z^tau p(z), where
   p(z) = z^tau [(z-1)^2 + lam b kA2 (z-1) + lam a b kA1] + (1-lam) b [kH2 (z-1) + a kH1]. */
function polyRoots(c) {                                                      // monic, highest power first (Durand-Kerner)
  const n = c.length - 1;
  let z = Array.from({ length: n }, (_, k) => { let re = 1, im = 0; for (let j = 0; j < k; j++) [re, im] = [re * 0.4 - im * 0.9, re * 0.9 + im * 0.4]; return [re, im]; });
  const ev = x => { let re = 1, im = 0; for (let i = 1; i <= n; i++) [re, im] = [re * x[0] - im * x[1] + c[i], re * x[1] + im * x[0]]; return [re, im]; };
  for (let it = 0; it < 800; it++) {
    let step = 0;
    z = z.map((zi, i) => {
      const [pr, pi] = ev(zi); let dr = 1, di = 0;
      for (let j = 0; j < n; j++) if (j !== i) { const xr = zi[0] - z[j][0], xi = zi[1] - z[j][1]; [dr, di] = [dr * xr - di * xi, dr * xi + di * xr]; }
      const m = dr * dr + di * di, qr = (pr * dr + pi * di) / m, qi = (pi * dr - pr * di) / m;
      step = Math.max(step, Math.hypot(qr, qi));
      return [zi[0] - qr, zi[1] - qi];
    });
    if (step < 1e-15) break;
  }
  return z;
}
function spectralRadius(lam, tau, kh = KH) {
  const c1 = -2 + lam * B2 * KA[1], c0 = 1 - lam * B2 * KA[1] + lam * A12 * B2 * KA[0];
  const r1 = (1 - lam) * B2 * kh[1], r0 = (1 - lam) * B2 * (A12 * kh[0] - kh[1]);
  const c = new Array(tau + 3).fill(0); c[0] = 1; c[1] = c1; c[2] = c0;
  c[tau + 1] += r1; c[tau + 2] += r0;
  return Math.max(...polyRoots(c).map(([re, im]) => Math.hypot(re, im)));
}

/* =====================================================================
   THE DEMO: WHO STEERS?
   ===================================================================== */
const hero = { kind: "driver", seed: 1 };
function updateHero() {
  const I = +$("#hI").value, W = +$("#hW").value, avail = $("#hAv").checked, lam = authority(I, W, avail);
  $("#hIV").textContent = fmt(I, 2); $("#hWV").textContent = fmt(W, 2);
  $$("#demo [data-case]").forEach(b => b.setAttribute("aria-pressed", b.dataset.case === hero.kind ? "true" : "false"));
  const opt = { W, avail, kind: hero.kind, seed: hero.seed };
  const shared = simulate({ ...opt, lam }), auto = simulate({ ...opt, lam: 1 }), driver = simulate({ ...opt, lam: 0 });
  const t = shared.map((_, k) => (k + 1) * TS), target = hero.kind === "driver" ? 0.6 : -0.6;
  const cv = $("#heroC"), fr = frame(cv, narrowOf(cv) ? 1.4 : 2.6, { x: [0, 20], y: [-0.85, 0.85], xlabel: "time (s)", ylabel: "offset (m)", yticks: [-0.6, -0.3, 0, 0.3, 0.6], yfmt: v => fmt(v, 1) });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--geom"); ctx.lineWidth = 1.4; ctx.setLineDash([6, 4]); seg(ctx, X(0), Y(target), X(20), Y(target)); ctx.setLineDash([]);   // where the mistaken agent steers
  if (avail) plotLine(fr, t, driver, { color: alpha("--warm", 0.75), width: 1.2 });
  plotLine(fr, t, auto, { color: alpha("--c5", 0.75), width: 1.2 });
  plotLine(fr, t, shared, { color: css("--path"), width: 2.4 });
  const err = rmsTail(shared), w = share(lam, W, avail);
  $("#rLam").textContent = fmt(lam, 3); $("#rShare").textContent = `${Math.round(100 * w)}%`;
  $("#rErr").textContent = `${fmt(100 * err, 1)} cm`; $("#rV").textContent = `${fmt(vRef(I), 1)} m/s`;
  const pct = v => `${Math.round(100 * v)}%`;
  let text;
  if (!avail) text = hero.kind === "driver"
    ? `The driver is not available, so their command is zero and their mistake no longer matters: the automation, at [[\\lambda = ${fmt(lam, 2)}]], keeps the car <b>${fmt(100 * err, 1)} cm</b> from the centre.`
    : `The driver is not available, so nobody corrects the biased perception: the authority is held at 0.85 or more${I < 0.5 ? ", even with this low integrity, since there is nobody to hand over to" : ""}, and the car follows the bias, <b>${fmt(100 * err, 0)} cm</b> off.`;
  else if (hero.kind === "driver") text = lam >= 0.5
    ? `The perception is fine, and the integrity says so: [[\\lambda = ${fmt(lam, 2)}]] gives the automation ${pct(w)} of the say, so the car stays <b>${fmt(100 * err, 1)} cm</b> from the centre although the driver holds 0.6 m.`
    : `The integrity score calls the perception untrustworthy although it is fine (a false alarm), so the mistaken driver gets ${pct(1 - w)} of the say, and the car drifts <b>${fmt(100 * err, 0)} cm</b> towards the driver's wrong target.`;
  else text = lam < 0.5
    ? `The monitor caught the faulty perception: [[\\lambda = ${fmt(lam, 2)}]] leaves the automation only ${pct(w)} of the say, so the car settles <b>${fmt(100 * err, 1)} cm</b> off, instead of following the 0.6 m bias.`
    : `The monitor has been fooled: the integrity stays high while the perception is biased, so the automation keeps ${pct(w)} of the say and the car follows the bias, <b>${fmt(100 * err, 0)} cm</b> off. The arbitration is only as good as the integrity score.`;
  if (avail && W >= 0.5) text += ` The loaded driver is also noisier and weaker (workload ${fmt(W, 2)}), which shifts the effective share further to the automation.`;
  if (I < 0.7) text += ` The law also lowers the speed to ${fmt(vRef(I), 1)} m/s (this simulation keeps 15 m/s).`;
  caption($("#heroStatus"), text);
}
$$("#demo [data-case]").forEach(b => b.addEventListener("click", () => { hero.kind = b.dataset.case; updateHero(); }));
["#hI", "#hW"].forEach(id => $(id).addEventListener("input", updateHero));
$("#hAv").addEventListener("change", updateHero);
$("#hNew").addEventListener("click", () => { hero.seed++; updateHero(); });

/* =====================================================================
   PART 3: DELAY AGAINST GAIN
   ===================================================================== */
const DELAYS = Array.from({ length: 13 }, (_, i) => i);                      // 0 to 0.6 s
function drawDelay() {
  const tau = Math.round(+$("#dT").value / TS), g = +$("#dG").value, kh = [KH[0] * g, KH[1] * g];
  $("#dTV").textContent = `${fmt(tau * TS, 2)} s`; $("#dGV").textContent = `× ${fmt(g, 2)}`;
  const human = simulate({ lam: 0, kind: "none", noise: false, tau, kh, x0: [0.5, 0] }), auto = simulate({ lam: 1, kind: "none", noise: false, x0: [0.5, 0] });
  const t = human.map((_, k) => (k + 1) * TS), narrow = narrowOf($("#dlyStep"));
  const fr = frame($("#dlyStep"), narrow ? 1.25 : 1.15, { x: [0, 20], y: [-0.6, 0.6], xlabel: "time (s)", ylabel: "offset (m)", yticks: [-0.5, 0, 0.5], yfmt: v => fmt(v, 1) });
  fr.ctx.strokeStyle = alpha("--ink", 0.35); fr.ctx.lineWidth = 1; seg(fr.ctx, fr.X(0), fr.Y(0), fr.X(20), fr.Y(0));
  plotLine(fr, t, auto, { color: css("--c5"), width: 1.6, dash: [5, 4] });
  plotLine(fr, t, human, { color: css("--warm"), width: 2.2 });
  const rhos = DELAYS.map(d => spectralRadius(0, d, kh)), r = rhos[tau];
  const gr = frame($("#dlyRho"), narrow ? 1.25 : 1.15, { x: [-0.02, 0.62], y: [0.6, 1.3], xlabel: "delay (s)", xticks: [0, 0.2, 0.4, 0.6], yticks: [0.6, 0.8, 1, 1.2], yfmt: v => fmt(v, 1), xfmt: v => fmt(v, 1) });
  const { ctx, X, Y } = gr;
  ctx.fillStyle = alpha("--geom", 0.07); ctx.fillRect(X(-0.02), Y(1.3), X(0.62) - X(-0.02), Y(1) - Y(1.3));
  ctx.strokeStyle = css("--geom"); ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]); seg(ctx, X(-0.02), Y(1), X(0.62), Y(1)); ctx.setLineDash([]);
  ctx.font = "600 10px " + css("--font-ui"); ctx.fillStyle = css("--geom"); ctx.textAlign = "left"; ctx.fillText("unstable", X(0) + 2, Y(1.3) + 12);
  plotLine(gr, DELAYS.map(d => d * TS), rhos.map(v => Math.min(v, 1.3)), { color: css("--warm"), width: 1.6 });
  DELAYS.forEach((d, i) => { ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(d * TS), Y(Math.min(rhos[i], 1.3)), i === tau ? 6 : 2.5, 0, 2 * Math.PI); ctx.fill(); });
  const firstBad = rhos.findIndex(v => v > 1), stable = r < 1;
  let text = `Delay ${fmt(tau * TS, 2)} s, gain × ${fmt(g, 2)}: spectral radius <b>${fmt(r, Math.abs(r - 1) < 0.005 ? 4 : 3)}</b>, `;
  text += stable ? (r > 0.97 ? `stable, but barely: the slowest mode decays by a factor [[e]] only every ${fmt(-TS / Math.log(r), 1)} s, and the driver oscillates.` : `stable: the slowest mode decays by a factor [[e]] every ${fmt(-TS / Math.log(r), 2)} s.`)
    : `<b>unstable</b>: the oscillation grows by ${Math.round(100 * (r ** 20 - 1))}% every second.`;
  text += firstBad >= 0 ? ` With this gain, the loop goes unstable from ${fmt(firstBad * TS, 2)} s of delay.` : " With this gain, no delay up to 0.6 s destabilises the loop.";
  text += " The automation (dashed) has no delay and is back within 1 cm in 0.75 s.";
  caption($("#dlyCap"), text);
}
["#dT", "#dG"].forEach(id => $(id).addEventListener("input", drawDelay));

/* =====================================================================
   PART 5: THE AUTHORITY LAW
   ===================================================================== */
const IS = Array.from({ length: 201 }, (_, i) => i / 200);
function drawLaw() {
  const I = +$("#lI").value, W = +$("#lW").value, avail = $("#lAv").checked, lam = authority(I, W, avail), lam0 = authority(I, 0, avail);
  $("#lIV").textContent = fmt(I, 2); $("#lWV").textContent = fmt(W, 2);
  const narrow = narrowOf($("#lawLam"));
  const fr = frame($("#lawLam"), narrow ? 1.25 : 1.15, { x: [0, 1], y: [0, 1], xlabel: "integrity I", ylabel: "authority λ", xticks: [0, 0.5, 1], yticks: [0, 0.5, 1], yfmt: v => fmt(v, 1), xfmt: v => fmt(v, 1) });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--ink", 0.06); ctx.fillRect(X(0), Y(1), X(1) - X(0), Y(0.95) - Y(1)); ctx.fillRect(X(0), Y(0.05), X(1) - X(0), Y(0) - Y(0.05));
  plotLine(fr, IS, IS.map(i => authority(i, 0, avail)), { color: alpha("--muted", 0.9), width: 1.4, dash: [5, 4] });
  plotLine(fr, IS, IS.map(i => authority(i, W, avail)), { color: css("--path"), width: 2.4 });
  ctx.fillStyle = css("--path"); ctx.beginPath(); ctx.arc(X(I), Y(lam), 5.5, 0, 2 * Math.PI); ctx.fill();
  const gr = frame($("#lawV"), narrow ? 1.25 : 1.15, { x: [0, 1], y: [0, 16], xlabel: "integrity I", ylabel: "speed (m/s)", xticks: [0, 0.5, 1], yticks: [0, 5, 10, 15], xfmt: v => fmt(v, 1) });
  plotLine(gr, [0, 1], [vRef(0), vRef(1)], { color: css("--c4"), width: 2.4 });
  gr.ctx.fillStyle = css("--c4"); gr.ctx.beginPath(); gr.ctx.arc(gr.X(I), gr.Y(vRef(I)), 5.5, 0, 2 * Math.PI); gr.ctx.fill();
  const raw = I + 0.35 * W * (I - 0.5);
  let text = `At [[I = ${fmt(I, 2)}]] and workload ${fmt(W, 2)}: [[\\lambda = ${fmt(lam, 3)}]]`;
  if (!avail) text += `. The driver is not available, so the authority is held at 0.85 or more.`;
  else if (raw > 0.95 || raw < 0.05) text += `, held by the clip (the raw value is ${fmt(raw, 3)}): both agents stay in the loop.`;
  else if (!W || Math.abs(I - 0.5) < 1e-9) text += Math.abs(I - 0.5) < 1e-9 && W ? `. At exactly one half, workload has no effect.` : `.`;
  else if (I > 0.5) text += `, against ${fmt(lam0, 3)} without workload: the perception is decent, so a loaded driver hands a little more to the automation.`;
  else text += `, against ${fmt(lam0, 3)} without workload: the automation is now the less trustworthy agent, so workload shifts a little more to the driver.`;
  text += ` The reference speed is <b>${fmt(vRef(I), 1)} m/s</b>, ${Math.round(100 * (0.4 + 0.6 * I))}% of the nominal 15.`;
  caption($("#lawCap"), text);
}
["#lI", "#lW"].forEach(id => $(id).addEventListener("input", drawLaw));
$("#lAv").addEventListener("change", drawLaw);

/* =====================================================================
   PART 6: STEADY-STATE ERROR AGAINST AUTHORITY
   ===================================================================== */
const LAMS = Array.from({ length: 201 }, (_, i) => i / 200);
function drawConflict() {
  const b = +$("#cB").value, W = +$("#cW").value;
  $("#cBV").textContent = `${fmt(b, 2)} m`; $("#cWV").textContent = fmt(W, 2);
  const errDriver = l => (1 - share(l, W)) * b, errAuto = l => share(l, W) * b;   // |e_ss| for each conflict
  const cv = $("#cflC"), fr = frame(cv, narrowOf(cv) ? 1.3 : 2.4, { x: [0, 1], y: [0, 1.05], xlabel: "automation authority λ", ylabel: "steady-state error (m)", xticks: [0, 0.25, 0.5, 0.75, 1], yticks: [0, 0.25, 0.5, 0.75, 1], yfmt: v => fmt(v, 2), xfmt: v => fmt(v, 2) });
  const { ctx, X, Y } = fr;
  plotLine(fr, LAMS, LAMS.map(errDriver), { color: css("--c4"), width: 2.2 });
  plotLine(fr, LAMS, LAMS.map(errAuto), { color: css("--geom"), width: 2.2 });
  const kh = (1 - 0.5 * W) * KH[0], lc = kh / (kh + KA[0]);
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(lc), Y(b / 2), 4.5, 0, 2 * Math.PI); ctx.fill();
  ctx.font = "600 10px " + css("--font-ui"); ctx.textAlign = "left"; ctx.fillText(`cross at λ = ${fmt(lc, 2)}`, X(lc) + 7, Y(b / 2) - 6);
  const hi = authority(0.9, W), lo = authority(0.1, W);
  [[hi, errDriver(hi), "--c4", "I = 0.9"], [lo, errAuto(lo), "--geom", "I = 0.1"]].forEach(([l, e, col, lab]) => {
    ctx.strokeStyle = alpha(col, 0.6); ctx.lineWidth = 1; ctx.setLineDash([3, 3]); seg(ctx, X(l), Y(0), X(l), Y(1.05)); ctx.setLineDash([]);
    ctx.fillStyle = css(col); ctx.beginPath(); ctx.arc(X(l), Y(e), 5.5, 0, 2 * Math.PI); ctx.fill();
    ctx.textAlign = l > 0.5 ? "right" : "left"; ctx.fillText(lab, X(l) + (l > 0.5 ? -6 : 6), Y(1.05) + 12);
  });
  caption($("#cflCap"),
    `The curves cross at [[\\lambda = ${fmt(lc, 2)}]], where each agent has half the say and the car sits halfway, ${fmt(100 * b / 2, 0)} cm off. ` +
    `With integrity 0.9, the law gives [[\\lambda = ${fmt(hi, 2)}]], and a mistaken driver moves the car by <b>${fmt(100 * errDriver(hi), 1)} cm</b>. ` +
    `With integrity 0.1, [[\\lambda = ${fmt(lo, 2)}]], and a mistaken perception still moves it by <b>${fmt(100 * errAuto(lo), 1)} cm</b>: the automation keeps ${Math.round(100 * share(lo, W))}% of the say with ${Math.round(100 * lo)}% of the authority, because its gain is stiffer.` +
    (W ? ` Workload weakens the driver's gain by ${Math.round(50 * W)}%, which shifts the crossing to lower [[\\lambda]].` : ""));
}
["#cB", "#cW"].forEach(id => $(id).addEventListener("input", drawConflict));

/* =====================================================================
   start-up
   ===================================================================== */
updateHero(); drawDelay(); drawLaw(); drawConflict();
onRedraw(() => { updateHero(); drawDelay(); drawLaw(); drawConflict(); });
})();
