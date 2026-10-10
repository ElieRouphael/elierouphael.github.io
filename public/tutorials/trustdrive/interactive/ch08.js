/* TrustDrive chapter 08, Capstone: the live demos.
   Everything runs on capstone.js, the course library ported step for step, with its own
   seeded noise. Needs ../kit/kit.js, ../kit/stats.js and capstone.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { makeRng, quantile } = window.Stats;
const C = window.Capstone, T_END = 30;
const COL = { naive: "--geom", handover: "--warm", proposed: "--c4" };
const NAME = { naive: "naive", handover: "hard handover", proposed: "proposed" };
const FRAMES = Object.fromEntries(C.SCENARIOS.map(s => [s, C.buildScenario(s)]));

/* shade the stretches where the scenario is faulty: outside the domain, or a fighting / absent driver */
function faultMask(sc) { return FRAMES[sc].map(f => !C.insideOdd(f) || f.conflict > 0 || !f.available); }
function shade(fr, mask) {
  const { ctx, X, pad, ih } = fr;
  ctx.fillStyle = alpha("--geom", 0.08);
  for (let k = 0; k < mask.length;) {
    if (!mask[k]) { k++; continue; }
    let j = k; while (j < mask.length && mask[j]) j++;
    ctx.fillRect(X(k * C.TS), pad.t, X(j * C.TS) - X(k * C.TS), ih); k = j;
  }
}
const tAxis = n => Array.from({ length: n }, (_, k) => k * C.TS);
const windowOf = (log, pred) => log.t.map((_, k) => pred(k));
const meanOf = (a, w) => { let s = 0, n = 0; a.forEach((v, k) => { if (w[k]) { s += v; n++; } }); return n ? s / n : NaN; };

/* =====================================================================
   THE DEMO: THREE ARCHITECTURES, FOUR SCENARIOS
   ===================================================================== */
const hero = { sc: "ood_curve", seed: 0, logs: null };
function updateHero() {
  $$("#demo [data-sc]").forEach(b => b.setAttribute("aria-pressed", b.dataset.sc === hero.sc ? "true" : "false"));
  hero.logs = Object.fromEntries(C.METHODS.map(m => [m, C.runEpisode(m, FRAMES[hero.sc], { seed: hero.seed })]));
  hero.met = Object.fromEntries(C.METHODS.map(m => [m, C.metrics(hero.logs[m])]));
  drawHero(); heroStatus();
}
function drawHero() {
  const L = hero.logs, t = tAxis(L.naive.ey.length), narrow = narrowOf($("#eyC")), mask = faultMask(hero.sc);
  const top = Math.min(1.6, Math.max(0.9, 1.1 * Math.max(...C.METHODS.map(m => hero.met[m].max))));
  const fr = frame($("#eyC"), narrow ? 1.4 : 2.6, { x: [0, T_END], y: [-top, top], xlabel: "time (s)", ylabel: "offset (m)", yfmt: v => fmt(v, 1) });
  shade(fr, mask);
  fr.ctx.strokeStyle = alpha("--geom", 0.8); fr.ctx.lineWidth = 1; fr.ctx.setLineDash([2, 4]);
  [0.7, -0.7].forEach(y => seg(fr.ctx, fr.X(0), fr.Y(y), fr.X(T_END), fr.Y(y))); fr.ctx.setLineDash([]);
  plotLine(fr, t, L.handover.ey, { color: css(COL.handover), width: 1.8 });
  plotLine(fr, t, L.naive.ey, { color: css(COL.naive), width: 1.5, dash: [5, 4] });       // dashed: it often runs under the handover
  plotLine(fr, t, L.proposed.ey, { color: css(COL.proposed), width: 2.4 });
  const p = L.proposed;
  const gr = frame($("#trustC"), narrow ? 1.9 : 3.4, { x: [0, T_END], y: [0, 1.05], xlabel: "time (s)", yticks: [0, 0.5, 1], yfmt: v => fmt(v, 1) });
  shade(gr, mask);
  plotLine(gr, t, p.v.map(v => v / C.V), { color: css("--ink"), width: 1.3, dash: [5, 4] });
  plotLine(gr, t, p.lam, { color: css("--c5"), width: 1.6 });
  plotLine(gr, t, p.I, { color: css("--path"), width: 2.2 });
  $("#heroT tbody").innerHTML = C.METHODS.map(m => { const me = hero.met[m]; return `<tr><td>${NAME[m]}</td><td>${fmt(me.rmse, 3)}</td><td>${me.exits}</td><td>${fmt(me.unsafe, 1)}%</td></tr>`; }).join("");
}
function heroStatus() {
  const L = hero.logs, M = hero.met, p = L.proposed, f = v => fmt(v, 3);
  let text;
  if (hero.sc === "nominal") text = `Inside the domain, the three methods track alike: RMSE ${f(M.naive.rmse)}, ${f(M.handover.rmse)} and ${f(M.proposed.rmse)} m, most of it the recovery from the 0.3 m start. The proposed method keeps 95% of the authority, and steers more smoothly on its filtered estimate (jerk ${fmt(M.proposed.jerk, 1)} against ${fmt(M.naive.jerk, 1)}).`;
  else if (hero.sc === "fog") {
    const h = L.handover.lam, on = h.indexOf(0), off = h.lastIndexOf(0);
    text = `As the fog thickens, the naive car follows a perception whose noise grows many-fold: RMSE <b>${f(M.naive.rmse)} m</b>, ${M.naive.exits} lane exit${M.naive.exits === 1 ? "" : "s"}. ` +
      (on >= 0 ? `The hard handover gives the wheel to the driver from ${fmt(on * C.TS, 1)} to ${fmt(off * C.TS, 1)} s, when the reported uncertainty passes its threshold: RMSE ${f(M.handover.rmse)} m. ` : "") +
      `The proposed method shifts authority gradually (down to [[\\lambda = ${fmt(Math.min(...p.lam), 2)}]]) and slows to ${fmt(Math.min(...p.v), 1)} m/s: RMSE <b>${f(M.proposed.rmse)} m</b>.`;
  } else if (hero.sc === "ood_curve") {
    const w = windowOf(p, k => !p.inside[k]), pre = windowOf(p, k => p.t[k] < 12);
    text = `On the sharp bend, the perception is biased, yet its reported uncertainty [[\\mathrm{tr}\\,\\Sigma]] rises by only ${Math.round(100 * (meanOf(p.trace, w) / meanOf(p.trace, pre) - 1))}%. The hard handover never switches, and matches naive: RMSE <b>${f(M.handover.rmse)} m</b>, peak ${fmt(M.handover.max, 2)} m, and the automation trusted outside its domain ${fmt(M.naive.unsafe, 0)}% of the time. ` +
      `The proposed method's domain monitor reads the map's curvature, and its NIS sees the inconsistency: the integrity falls to ${fmt(Math.min(...p.I), 2)}, the authority to ${fmt(Math.min(...p.lam), 2)} and the speed to ${fmt(Math.min(...p.v), 1)} m/s. RMSE <b>${f(M.proposed.rmse)} m</b>.`;
  } else {
    const w = windowOf(p, k => p.t[k] >= 14 && p.t[k] < 22), rms = log => Math.sqrt(meanOf(log.ey.map(v => v * v), w));
    text = `Naive and handover ignore the driver, so the overloaded driver fighting the wheel from 14 to 20 s changes nothing for them. The proposed method keeps listening, but the perception is fine, so the integrity stays near ${fmt(meanOf(p.I, w), 2)} and the authority at ${fmt(meanOf(p.lam, w), 2)}: during the fight, its RMS offset is ${fmt(100 * rms(p), 1)} cm, against ${fmt(100 * rms(L.naive), 1)} cm for naive. The driver is out-voted.`;
  }
  caption($("#heroStatus"), text);
}
$$("#demo [data-sc]").forEach(b => b.addEventListener("click", () => { hero.sc = b.dataset.sc; updateHero(); }));
$("#hNew").addEventListener("click", () => { hero.seed++; updateHero(); });

/* =====================================================================
   PART 4: INSIDE THE PROPOSED METHOD
   ===================================================================== */
let mechSc = "ood_curve";
function drawMech() {
  $$("#mech [data-msc]").forEach(b => b.setAttribute("aria-pressed", b.dataset.msc === mechSc ? "true" : "false"));
  const p = C.runEpisode("proposed", FRAMES[mechSc], { seed: 0 }), t = tAxis(p.ey.length), mask = faultMask(mechSc), narrow = narrowOf($("#mechE"));
  const top = Math.max(0.5, 1.1 * Math.max(...p.yey.map(Math.abs)));
  const fr = frame($("#mechE"), narrow ? 1.7 : 3.2, { x: [0, T_END], y: [-top, top], xlabel: "time (s)", ylabel: "offset (m)", yfmt: v => fmt(v, 1) });
  shade(fr, mask);
  plotLine(fr, t, p.yey, { color: alpha("--muted", 0.8), width: 1 });
  plotLine(fr, t, p.ey, { color: css("--ink"), width: 2 });
  const gr = frame($("#mechS"), narrow ? 1.5 : 2.8, { x: [0, T_END], y: [0, 1.05], xlabel: "time (s)", yticks: [0, 0.5, 1], yfmt: v => fmt(v, 1) });
  shade(gr, mask);
  [["su", "--c5"], ["sn", "--warm"], ["so", "--c4"]].forEach(([k, col]) => plotLine(gr, t, p[k], { color: css(col), width: 1.4 }));
  plotLine(gr, t, p.I, { color: css("--path"), width: 2.4 });
  const w = mask, pre = t.map(x => x < 10);
  const nisW = p.nis.filter((_, k) => w[k]), traceRatio = Math.max(...p.trace.filter((_, k) => w[k])) / meanOf(p.trace, pre);
  const lowI = p.I.filter((v, k) => w[k] && v < 0.5).length / nisW.length;
  caption($("#mechCap"), mechSc === "ood_curve"
    ? `Over the bend (shaded), the reported uncertainty [[\\mathrm{tr}\\,\\Sigma]] peaks at only ${fmt(traceRatio, 2)} times its clear-weather value, so the uncertainty score hardly moves. The perceived offset (grey) is biased and noisier than the perception admits. The NIS averages ${fmt(meanOf(p.nis, w), 1)}, peaks at ${fmt(Math.max(...nisW), 0)}, and crosses its gate on only ${Math.round(100 * nisW.filter(v => v > 9.21).length / nisW.length)}% of the samples, so its score flickers; the domain score, from the map's curvature, drops and stays down. The integrity is below one half for ${Math.round(100 * lowI)}% of the bend.`
    : `In the fog (shaded: outside the domain), the reported uncertainty [[\\mathrm{tr}\\,\\Sigma]] climbs to ${fmt(traceRatio, 0)} times its clear-weather value (the noise's standard deviation, ${fmt(Math.sqrt(traceRatio), 0)} times), and the uncertainty score falls to ${fmt(Math.min(...p.su), 2)} by itself. The NIS stays quiet (mean ${fmt(meanOf(p.nis, w), 1)}, about the 2 of a consistent filter), because the noise is reported honestly. The domain score falls too, redundantly. The integrity is below one half for ${Math.round(100 * lowI)}% of the fog.`);
}
$$("#mech [data-msc]").forEach(b => b.addEventListener("click", () => { mechSc = b.dataset.msc; drawMech(); }));

/* =====================================================================
   PART 5: 20 SEEDS AND A BOOTSTRAP
   ===================================================================== */
const ST = { done: false, rmse: { fog: {}, ood_curve: {} }, base: "naive" };
function runStats() {
  const jobs = [];
  for (const sc of ["fog", "ood_curve"]) for (const m of C.METHODS) { ST.rmse[sc][m] = []; for (let s = 0; s < 20; s++) jobs.push([sc, m, s]); }
  let i = 0;
  const slice = () => {
    const t0 = performance.now();
    while (i < jobs.length && performance.now() - t0 < 12) {
      const [sc, m, s] = jobs[i++]; ST.rmse[sc][m][s] = C.metrics(C.runEpisode(m, FRAMES[sc], { seed: 100 + s })).rmse;
    }
    if (i < jobs.length) { $("#statCap").textContent = `Running 20 seeds… ${Math.round(100 * i / jobs.length)}%`; setTimeout(slice, 0); }
    else { ST.done = true; drawStats(); }
  };
  slice();
}
function bootCi(d, n = 20000, seed = 0) {
  const rng = makeRng(seed), means = new Float64Array(n);
  for (let b = 0; b < n; b++) { let s = 0; for (let j = 0; j < d.length; j++) s += d[Math.floor(rng.uniform() * d.length)]; means[b] = s / d.length; }
  const m = d.reduce((a, v) => a + v, 0) / d.length, sorted = Array.from(means).sort((a, b) => a - b);
  return [m, quantile(sorted, 0.025), quantile(sorted, 0.975)];
}
function drawStats() {
  $$("#stw [data-base]").forEach(b => b.setAttribute("aria-pressed", b.dataset.base === ST.base ? "true" : "false"));
  if (!ST.done) return;
  const diff = sc => ST.rmse[sc].proposed.map((v, s) => v - ST.rmse[sc][ST.base][s]);
  const rows = [["fog", diff("fog")], ["sharp bend", diff("ood_curve")], ["both", [...diff("fog"), ...diff("ood_curve")]]];
  const cis = rows.map(([, d]) => bootCi(d));
  const lo = Math.min(...rows.flatMap(([, d]) => d)), xmin = Math.min(-0.05, 1.1 * lo);
  const cv = $("#statC"), fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [xmin, 0.02], y: [-0.5, 2.6], yticks: false, xlabel: "proposed minus baseline, RMSE (m)", xfmt: v => fmt(v, 2), pad: { l: 78 } });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = alpha("--ink", 0.6); ctx.lineWidth = 1.2; ctx.setLineDash([4, 4]); seg(ctx, X(0), Y(-0.5), X(0), Y(2.6)); ctx.setLineDash([]);
  const jit = makeRng(5);
  rows.forEach(([label, d], r) => {
    const y = 2 - r;
    ctx.font = "600 11px " + css("--font-ui"); ctx.fillStyle = css("--ink"); ctx.textAlign = "right"; ctx.fillText(label, X(xmin) - 8, Y(y) + 4);
    d.forEach(v => { ctx.fillStyle = alpha("--c4", 0.55); ctx.beginPath(); ctx.arc(X(v), Y(y + 0.22 * (jit.uniform() - 0.5)), 3, 0, 2 * Math.PI); ctx.fill(); });
    const [m, a, b] = cis[r];
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2.2; seg(ctx, X(a), Y(y - 0.3), X(b), Y(y - 0.3));
    seg(ctx, X(m), Y(y - 0.38), X(m), Y(y - 0.22));
  });
  const [m, a, b] = cis[2], neg = rows[2][1].filter(v => v < 0).length;
  caption($("#statCap"),
    `Against ${NAME[ST.base]}, over 20 seeds of each degraded scenario: the mean paired difference is <b>${fmt(m, 3)} m</b>, 95% bootstrap interval [${fmt(a, 3)}, ${fmt(b, 3)}] (black bars), and ${neg} of the 40 differences are negative. ` +
    `Fog: ${fmt(cis[0][0], 3)} m; sharp bend: ${fmt(cis[1][0], 3)} m. ` + (b < 0 ? "The interval excludes zero by a wide margin." : "The interval does not exclude zero."));
}
$$("#stw [data-base]").forEach(b => b.addEventListener("click", () => { ST.base = b.dataset.base; drawStats(); }));

/* =====================================================================
   PART 6: THE ABLATION
   ===================================================================== */
let ablSc = "ood_curve";
const fullCache = {};
function drawAbl() {
  $$("#abl [data-asc]").forEach(b => b.setAttribute("aria-pressed", b.dataset.asc === ablSc ? "true" : "false"));
  const opt = { useNis: $("#aNis").checked, useOdd: $("#aOdd").checked, filter: $("#aFilt").checked, speed: $("#aSpeed").checked };
  fullCache[ablSc] = fullCache[ablSc] || C.runEpisode("proposed", FRAMES[ablSc], { seed: 0 });
  const full = fullCache[ablSc], cfg = C.runEpisode("proposed", FRAMES[ablSc], { seed: 0, ...opt });
  const mf = C.metrics(full), mc = C.metrics(cfg), t = tAxis(full.ey.length);
  const top = Math.max(0.9, 1.1 * Math.max(mf.max, mc.max));
  const cv = $("#ablC"), fr = frame(cv, narrowOf(cv) ? 1.5 : 2.8, { x: [0, T_END], y: [-top, top], xlabel: "time (s)", ylabel: "offset (m)", yfmt: v => fmt(v, 1) });
  shade(fr, faultMask(ablSc));
  fr.ctx.strokeStyle = alpha("--geom", 0.8); fr.ctx.lineWidth = 1; fr.ctx.setLineDash([2, 4]);
  [0.7, -0.7].forEach(y => seg(fr.ctx, fr.X(0), fr.Y(y), fr.X(T_END), fr.Y(y))); fr.ctx.setLineDash([]);
  plotLine(fr, t, full.ey, { color: css("--c4"), width: 2.2 });
  const same = Object.values(opt).every(Boolean);
  if (!same) plotLine(fr, t, cfg.ey, { color: css("--geom"), width: 1.6 });
  const off = Object.entries({ "the NIS": opt.useNis, "the domain monitor": opt.useOdd, "the filter": opt.filter, "the speed contraction": opt.speed }).filter(([, v]) => !v).map(([k]) => k);
  const g = 9.81;
  let text = same ? `The full proposed method: RMSE <b>${fmt(mf.rmse, 3)} m</b>, unsafe usage ${fmt(mf.unsafe, 1)}%, peak lateral acceleration ${fmt(mf.ay / g, 2)} g.`
    : `Without ${off.join(" and ")}: RMSE <b>${fmt(mc.rmse, 3)} m</b> against ${fmt(mf.rmse, 3)} m for the full method, unsafe usage <b>${fmt(mc.unsafe, 1)}%</b> against ${fmt(mf.unsafe, 1)}%, peak lateral acceleration ${fmt(mc.ay / g, 2)} g against ${fmt(mf.ay / g, 2)} g.`;
  if (!same) {
    if (ablSc === "ood_curve" && !opt.useOdd && opt.useNis) text += " The NIS sees the bend, but only intermittently; on its own it cannot hold the integrity down.";
    else if (ablSc === "ood_curve" && !opt.useOdd && !opt.useNis) text += " Only the perception's self-reported uncertainty is left, and it barely moves: the method degrades to naive.";
    else if (ablSc === "ood_curve" && !opt.useNis && opt.useOdd && opt.speed && opt.filter) text += " The domain monitor alone holds the integrity down on this monitored bend.";
    else if (ablSc === "fog" && opt.speed && opt.filter) text += " In fog, the self-reported uncertainty catches the degradation on its own, so the other monitors are redundant here.";
    if (!opt.speed) text += " Without the speed contraction, the car keeps 15 m/s, and the error and the lateral acceleration grow.";
  }
  caption($("#ablCap"), text);
}
$$("#abl [data-asc]").forEach(b => b.addEventListener("click", () => { ablSc = b.dataset.asc; drawAbl(); }));
["#aNis", "#aOdd", "#aFilt", "#aSpeed"].forEach(id => $(id).addEventListener("change", drawAbl));

/* =====================================================================
   start-up
   ===================================================================== */
updateHero(); drawMech(); drawAbl(); runStats();
onRedraw(() => { drawHero(); drawMech(); drawStats(); drawAbl(); });
})();
