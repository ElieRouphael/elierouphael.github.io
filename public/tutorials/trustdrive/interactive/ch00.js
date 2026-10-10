/* TrustDrive chapter 00, Introduction and overview: the live demos.
   simulate() is the notebook's first experiment (a car on a winding lane, a Kalman predictor
   from the physics, a bias from 7.5 s), with its own seeded noise, plus a variant in which the
   network admits its error by reporting a noise that covers it.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { makeRng, mean } = window.Stats;

/* ---------------- the notebook's experiment ---------------- */
const TS = 0.05, V = 15, N = 300, ONSET = 150, CHI2 = 9.210, Q = 1e-4;
const BASE = [0.02, 0.01], BIAS = [0.22, 0.06];
const SIG_THR = Math.sqrt(0.0045);                                            // the hard handover's threshold of chapter 08, as a std
function simulate({ seed = 1, scale = 1, honest = false } = {}) {
  const rng = makeRng(seed), a = V * TS;
  let t0 = 0, t1 = 0, x0 = 0, x1 = 0, p00 = 0.05, p01 = 0, p11 = 0.02;      // true state, filter state, covariance
  const out = { nis: [], sig: [], tr: [] };
  for (let k = 0; k < N; k++) {
    const kappa = 0.012 * Math.sin(2 * Math.PI * k * TS / 6);
    [t0, t1] = [t0 + V * Math.sin(t1) * TS, t1 - V * kappa * TS];            // the car (no steering, no process noise)
    const b = k < ONSET ? [0, 0] : [BIAS[0] * scale, BIAS[1] * scale];
    const srep = honest ? [Math.hypot(BASE[0], b[0]), Math.hypot(BASE[1], b[1])] : BASE;
    const y0 = t0 + b[0] + rng.normal(0, BASE[0]), y1 = t1 + b[1] + rng.normal(0, BASE[1]);
    // Kalman predictor from the physics (steering 0), then the update with the reported noise
    x0 = x0 + a * x1; x1 = x1 - V * TS * kappa;
    const q00 = p00 + 2 * a * p01 + a * a * p11 + Q, q01 = p01 + a * p11, q11 = p11 + Q;
    const s00 = q00 + srep[0] ** 2, s11 = q11 + srep[1] ** 2, det = s00 * s11 - q01 * q01, n0 = y0 - x0, n1 = y1 - x1;
    out.nis.push((s11 * n0 * n0 - 2 * q01 * n0 * n1 + s00 * n1 * n1) / det);
    out.tr.push(srep[0] ** 2 + srep[1] ** 2); out.sig.push(Math.sqrt(srep[0] ** 2 + srep[1] ** 2));
    const i00 = s11 / det, i01 = -q01 / det, i11 = s00 / det;
    const k00 = q00 * i00 + q01 * i01, k01 = q00 * i01 + q01 * i11, k10 = q01 * i00 + q11 * i01, k11 = q01 * i01 + q11 * i11;
    x0 += k00 * n0 + k01 * n1; x1 += k10 * n0 + k11 * n1;
    p00 = q00 - (k00 * q00 + k01 * q01); p01 = q01 - (k00 * q01 + k01 * q11); p11 = q11 - (k10 * q01 + k11 * q11);
  }
  return out;
}
/* the three rules' authority: naive 1; handover on the reported uncertainty; proposed from
   I = EMA(s_unc s_NIS) with chapter 07's law at a workload of 0.2 (no domain conditions here) */
function authorities(run) {
  let I = 1;
  const proposed = run.tr.map((tr, k) => {
    I = 0.6 * I + 0.4 * Math.exp(-tr / 0.02) * Math.exp(-Math.max(0, run.nis[k] - CHI2) / 6);
    return clamp(I + 0.35 * 0.2 * (I - 0.5), 0.05, 0.95);
  });
  return { naive: run.tr.map(() => 1), handover: run.tr.map(tr => (tr > SIG_THR ** 2 ? 0 : 1)), proposed };
}
const tAxis = Array.from({ length: N }, (_, k) => k * TS);
function shadeFault(fr) { fr.ctx.fillStyle = alpha("--geom", 0.07); fr.ctx.fillRect(fr.X(ONSET * TS), fr.pad.t, fr.X(N * TS) - fr.X(ONSET * TS), fr.ih); }

/* =====================================================================
   THE DEMO: WHICH SIGNAL SEES THE FAILURE?
   ===================================================================== */
const hero = { seed: 1, mode: "confident" };
function updateHero() {
  const scale = +$("#hB").value, honest = hero.mode === "honest";
  $("#hBV").textContent = `${fmt(BIAS[0] * scale, 2)} m and ${fmt(BIAS[1] * scale, 3)} rad`;
  $$("#demo [data-mode]").forEach(b => b.setAttribute("aria-pressed", b.dataset.mode === hero.mode ? "true" : "false"));
  const r = simulate({ seed: hero.seed, scale, honest }), narrow = narrowOf($("#sigC"));
  const sTop = Math.max(0.1, 1.15 * Math.max(...r.sig));
  const fr = frame($("#sigC"), narrow ? 1.9 : 3.4, { x: [0, N * TS], y: [0, sTop], xlabel: "time (s)", yfmt: v => fmt(v, sTop > 0.3 ? 1 : 2) });
  shadeFault(fr);
  fr.ctx.strokeStyle = css("--ink"); fr.ctx.lineWidth = 1.2; fr.ctx.setLineDash([5, 4]); seg(fr.ctx, fr.X(0), fr.Y(SIG_THR), fr.X(N * TS), fr.Y(SIG_THR)); fr.ctx.setLineDash([]);
  fr.ctx.font = "600 10px " + css("--font-ui"); fr.ctx.fillStyle = css("--ink"); fr.ctx.textAlign = "left"; fr.ctx.fillText("handover threshold", fr.X(0.2), fr.Y(SIG_THR) - 5);
  plotLine(fr, tAxis, r.sig, { color: css("--c5"), width: 2.2 });
  const nTop = Math.min(80, Math.max(15, 1.05 * Math.max(...r.nis)));
  const gr = frame($("#nisC"), narrow ? 1.6 : 2.9, { x: [0, N * TS], y: [0, nTop], xlabel: "time (s)", ylabel: "NIS", yfmt: v => String(Math.round(v)) });
  shadeFault(gr);
  plotLine(gr, tAxis, r.nis.map(v => Math.min(v, nTop)), { color: css("--warm"), width: 1.3 });
  gr.ctx.strokeStyle = css("--ink"); gr.ctx.lineWidth = 1.2; gr.ctx.setLineDash([5, 4]); seg(gr.ctx, gr.X(0), gr.Y(CHI2), gr.X(N * TS), gr.Y(CHI2)); gr.ctx.setLineDash([]);
  gr.ctx.fillStyle = css("--geom"); r.nis.forEach((v, k) => { if (v > CHI2) gr.ctx.fillRect(gr.X(k * TS) - 1, gr.Y(0) - 7, 2, 7); });
  const before = r.nis.slice(0, ONSET), after = r.nis.slice(ONSET), above = after.filter(v => v > CHI2).length / after.length;
  const crossed = r.sig.some(s => s > SIG_THR);
  $("#rB").textContent = fmt(mean(before), 2); $("#rA").textContent = fmt(mean(after), 2);
  $("#rG").textContent = `${Math.round(100 * above)}%`; $("#rU").textContent = crossed ? "yes" : "no";
  let text;
  if (scale === 0) text = `No bias at all: the NIS stays around its in-distribution level (mean ${fmt(mean(r.nis), 1)}), and the reported uncertainty never moves. Nothing to detect.`;
  else if (!honest) text = `From 7.5 s the estimate is off, but the network's reported uncertainty <b>does not move</b>: a rule that watches it stays silent. The NIS, which compares each measurement with the physics, jumps from a mean of ${fmt(mean(before), 1)} to <b>${fmt(mean(after), 1)}</b>, ${Math.round(100 * above)}% of the samples crossing the 99% line.` +
      (above < 0.2 ? " This bias is small enough to hide in the noise of single samples: chapter 09 accumulates the evidence." : " The network is confidently wrong, and only the independent check sees it.");
  else text = `Now the network admits its error: its reported uncertainty jumps to ${fmt(Math.max(...r.sig), 2)}, ${crossed ? "<b>above the handover's threshold</b>" : "though not above the handover's threshold"}. The NIS, on the other hand, falls quiet (mean <b>${fmt(mean(after), 1)}</b> during the fault): the error is now within the noise the network claims. Each monitor catches a failure the other misses.`;
  caption($("#heroStatus"), text);
}
$$("#demo [data-mode]").forEach(b => b.addEventListener("click", () => { hero.mode = b.dataset.mode; updateHero(); }));
$("#hB").addEventListener("input", updateHero);
$("#hNew").addEventListener("click", () => { hero.seed++; updateHero(); drawArch(); });

/* =====================================================================
   PART 5: THREE WAYS TO USE THE SAME PERCEPTION
   ===================================================================== */
let archMode = "confident";
function drawArch() {
  $$("#arch [data-amode]").forEach(b => b.setAttribute("aria-pressed", b.dataset.amode === archMode ? "true" : "false"));
  const r = simulate({ seed: hero.seed, honest: archMode === "honest" }), L = authorities(r);
  const cv = $("#archC"), fr = frame(cv, narrowOf(cv) ? 1.5 : 2.8, { x: [0, N * TS], y: [-0.05, 1.08], xlabel: "time (s)", ylabel: "authority λ", yticks: [0, 0.5, 1], yfmt: v => fmt(v, 1) });
  shadeFault(fr);
  plotLine(fr, tAxis, L.naive, { color: css("--geom"), width: 2.6 });
  plotLine(fr, tAxis, L.handover, { color: css("--warm"), width: 2.6, dash: [9, 9] });   // dashed over naive, where they coincide
  plotLine(fr, tAxis, L.proposed, { color: css("--c4"), width: 2.4 });
  const fault = L.proposed.slice(ONSET), low = fault.filter(v => v < 0.5).length / fault.length;
  const handed = L.handover.slice(ONSET).filter(v => v === 0).length / fault.length;
  caption($("#archCap"), archMode === "confident"
    ? `With the network confidently wrong, the naive rule keeps full authority, and so does the hard handover: the reported uncertainty never crosses its threshold. Only the proposed rule reacts, through the NIS: its authority falls below one half on ${Math.round(100 * low)}% of the fault, down to ${fmt(Math.min(...fault), 2)}. It flickers, because single NIS samples cross their line only intermittently.`
    : `With the network admitting its error, the hard handover switches abruptly to the driver (${Math.round(100 * handed)}% of the fault), and the proposed rule drops its authority to ${fmt(Math.min(...fault), 2)} through the uncertainty score, smoothly. The naive rule still trusts the perception completely.`);
}
$$("#arch [data-amode]").forEach(b => b.addEventListener("click", () => { archMode = b.dataset.amode; drawArch(); }));

/* =====================================================================
   start-up
   ===================================================================== */
updateHero(); drawArch();
onRedraw(() => { updateHero(); drawArch(); });
})();
