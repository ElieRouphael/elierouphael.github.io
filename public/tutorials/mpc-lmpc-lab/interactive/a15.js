/* Advanced MPC and Learning MPC, chapter A15, disjunctive constraints and Big-M: the demos.
   The hero draws lab B1's forbidden interval on the road, the two convex pieces of the safe set, and
   the relaxed constraint set that a Big-M encoding allows when the binary variable is relaxed to a
   fraction; the Part 4 widget enumerates the two branch QPs from a chosen state with the course's
   interior-point solver. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { predictionMatrices, solveQP, mm, mv, tr, dot, legend } = window.MPC;

/* lab B1's geometry */
const D_BIKE = 100, L_CAR = 4.5, W_BIKE = 8, G = 9.81;
const D_MAX = D_BIKE - (L_CAR + W_BIKE) / 2, D_MIN = D_BIKE + (L_CAR + W_BIKE) / 2;

/* =====================================================================
   Hero: the forbidden interval, the relaxation and the smallest valid M
   ===================================================================== */
function drawHero() {
  const s = +$("#hS").value, M = +$("#hM").value, dHigh = +$("#hE").value, dLow = 0;
  const Mmin = Math.max(dHigh - D_MAX, D_MIN - dLow), valid = M >= Mmin - 1e-9;
  const cv = $("#roadC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 2.2 : 3.6, { x: [0, 300], y: [0, 4], xlabel: "vehicle centre d (m)", yticks: false, xticks: [0, 50, 100, 150, 200, 250, 300] }), { ctx, X, Y } = fr;
  const bar = (lo, hi, y, h, fill, stroke) => { if (hi < lo) return; ctx.fillStyle = fill; ctx.fillRect(X(Math.max(lo, 0)), Y(y + h), X(Math.min(hi, 300)) - X(Math.max(lo, 0)), Y(y) - Y(y + h)); if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.2; ctx.strokeRect(X(Math.max(lo, 0)), Y(y + h), X(Math.min(hi, 300)) - X(Math.max(lo, 0)), Y(y) - Y(y + h)); } };
  /* row 1: the geometry; row 2: the two branches; row 3: the relaxed set; row 4: the envelope */
  bar(D_BIKE - W_BIKE / 2, D_BIKE + W_BIKE / 2, 3.2, 0.5, alpha("--warm", 0.5)); bar(D_MAX, D_MIN, 3.0, 0.15, css("--warm"));
  bar(0, D_MAX, 2.1, 0.5, alpha("--post", 0.25), css("--post")); bar(D_MIN, 300, 2.1, 0.5, alpha("--c5", 0.25), css("--c5"));
  const lo = D_MIN - M * s, hi = D_MAX + M * (1 - s);
  bar(lo, hi, 1.2, 0.5, alpha(valid ? "--c4" : "--warm", 0.3), valid ? css("--c4") : css("--warm"));
  bar(dLow, dHigh, 0.3, 0.4, alpha("--ink", 0.08), alpha("--ink", 0.5));
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "middle";
  ctx.fillText("bicycle's region, and the forbidden centre positions (d_max, d_min)", X(2), Y(3.45));
  ctx.fillText("yield: d ≤ d_max", X(2), Y(2.35)); ctx.textAlign = "right"; ctx.fillText("pass: d ≥ d_min", X(298), Y(2.35)); ctx.textAlign = "left";
  ctx.fillText(`relaxed Big-M set for s = ${fmt(s, 2)}: d_min − M s ≤ d ≤ d_max + M (1 − s)`, X(2), Y(1.45));
  ctx.fillText(`position envelope [${dLow}, ${dHigh}]`, X(2), Y(0.5));
  ctx.textBaseline = "alphabetic";
  $("#hSV").textContent = fmt(s, 2); $("#hMV").textContent = M; $("#hEV").textContent = dHigh;
  $("#rMmin").textContent = fmt(Mmin, 2); $("#rValid").textContent = valid ? "yes" : "no: too small"; $("#rRelax").textContent = hi < lo ? "empty" : `[${fmt(Math.max(lo, 0), 1)}, ${fmt(Math.min(hi, 300), 1)}]`;
  let text = `The car's centre must satisfy [[d \\le d_{\\max} = ${D_MAX}]] (yield, blue) or [[d \\ge d_{\\min} = ${D_MIN}]] (pass, purple): two half-lines with a gap of [[${D_MIN - D_MAX}]] m, the car's length plus the region's width. The Big-M rows [[d \\le d_{\\max} + M(1 - s)]] and [[d \\ge d_{\\min} - Ms]] with [[M = ${M}]] give, at [[s = ${fmt(s, 2)}]], the green set ${hi < lo ? "which is empty" : `[[[${fmt(Math.max(lo, 0), 1)}, ${fmt(Math.min(hi, 300), 1)}]]]`}. `;
  text += s === 1 ? "At [[s = 1]] the pass row is switched off and the set is the yield half-line, provided [[M]] reaches the far end of the envelope. " : s === 0 ? "At [[s = 0]] the yield row is switched off and the set is the pass half-line, provided [[M]] reaches the near end of the envelope. " : `At a fractional [[s]], which only the LP relaxation can choose, the set ${lo <= D_MAX && hi >= D_MIN ? "contains the whole crossing" : "still overlaps the forbidden gap"}: the relaxation's bound says nothing about avoidance, and the larger [[M]], the looser the relaxation. `;
  text += `For the envelope [[d \\in [${dLow}, ${dHigh}]]] the smallest valid constant is [[M \\ge \\max(${dHigh} - ${D_MAX}, ${D_MIN} - ${dLow}) = ${fmt(Mmin, 2)}]]; ${valid ? "the chosen M is valid" : "the chosen M is too small: at s = 1 the pass row would still exclude part of the envelope, cutting off feasible yield plans"}. Lab B1 uses a round 5000, valid and ${fmt(5000 / Mmin, 0)} times larger than needed.`;
  caption($("#heroStatus"), text);
}
["#hS", "#hM", "#hE"].forEach(id => $(id).addEventListener("input", drawHero));
$("#hReset").addEventListener("click", () => { $("#hS").value = 0.5; $("#hM").value = 206.25; $("#hE").value = 300; drawHero(); });

/* =====================================================================
   Part 4: the two branch QPs from a chosen state (lab B1's model, Euler, Ts = 1, N = 4)
   ===================================================================== */
function branch(strategy, d0, v0, mu, N, Ru) {
  const Ts = 1, A = [[1, Ts], [0, 1]], B = [[0], [Ts]], { Sx, Su } = predictionMatrices(A, B, N), vref = 20, Qv = 3, Pv = 3, vmax = 32, amax = mu * G;
  const Qbar = Array.from({ length: 2 * N }, (_, i) => Array.from({ length: 2 * N }, (_, j) => (i === j && i % 2 === 1) ? (i === 2 * N - 1 ? Pv : Qv) : 0));
  const x0 = [d0, v0], base = mv(Sx, x0).map((v, i) => v - (i % 2 === 1 ? vref : 0));
  const SuTQ = mm(tr(Su), Qbar), H = mm(SuTQ, Su).map((r, i) => r.map((v, j) => 2 * (v + (i === j ? Ru : 0)))), f = mv(SuTQ, base).map(v => 2 * v);
  const Gr = [], g = [], Sxx = mv(Sx, x0);
  for (let k = 1; k <= N; k++) { const idx = 2 * (k - 1) + 1; Gr.push(Su[idx]); g.push(vmax - Sxx[idx]); Gr.push(Su[idx].map(v => -v)); g.push(Sxx[idx]); }
  const idx = 2 * (N - 1);
  if (strategy === "yield") { Gr.push(Su[idx]); g.push(D_MAX - Sxx[idx]); } else { Gr.push(Su[idx].map(v => -v)); g.push(-D_MIN + Sxx[idx]); }
  const r = solveQP(H, f, Gr, g, [], [], new Array(N).fill(-amax), new Array(N).fill(amax));
  const Xs = Sxx.map((v, i) => v + dot(Su[i], r.z)), vs = [v0]; for (let k = 0; k < N; k++) vs.push(Xs[2 * k + 1]);
  const cost = Ru * r.z.reduce((s, a) => s + a * a, 0) + Qv * vs.slice(0, N).reduce((s, v) => s + (v - vref) ** 2, 0) + Pv * (vs[N] - vref) ** 2;
  return { feasible: r.success, cost, dN: Xs[2 * (N - 1)], vN: vs[N], U: r.z };
}
function drawBranches() {
  const d0 = +$("#bD").value, v0 = +$("#bV").value, mu = +$("#bMu").value, Ru = [0.1, 1, 10, 100, 1000][+$("#bR").value], N = 4;
  const y = branch("yield", d0, v0, mu, N, Ru), p = branch("pass", d0, v0, mu, N, Ru);
  const lo = d0 + N * v0 - mu * G * N * (N - 1) / 2, hi = d0 + N * v0 + mu * G * N * (N - 1) / 2;
  $("#bDV").textContent = d0; $("#bVV").textContent = v0; $("#bMuV").textContent = fmt(mu, 2); $("#bRV").textContent = Ru;
  $("#brT").innerHTML = "<thead><tr><th>branch</th><th>feasible</th><th>cost</th><th>d<sub>N</sub></th><th>v<sub>N</sub></th><th>inputs</th></tr></thead><tbody>" + [["yield", y], ["pass", p]].map(([n, b]) => `<tr><td>${n}</td><td class="words">${b.feasible ? "yes" : "no"}</td><td>${b.feasible ? fmt(b.cost, 2) : "–"}</td><td>${b.feasible ? fmt(b.dN, 2) : "–"}</td><td>${b.feasible ? fmt(b.vN, 2) : "–"}</td><td>${b.feasible ? b.U.map(u => fmt(u, 2)).join(", ") : "–"}</td></tr>`).join("") + "</tbody>";
  const both = y.feasible && p.feasible, none = !y.feasible && !p.feasible, chosen = none ? null : both ? (y.cost <= p.cost ? "yield" : "pass") : y.feasible ? "yield" : "pass";
  const nb = d0 === 60 && v0 === 5 && mu === 0.1 && Ru === 1;
  caption($("#brCap"), `From [[x_0 = (${d0}, ${v0})]] with [[\\mu = ${fmt(mu, 2)}]], [[N = 4]], [[R = ${Ru}]]${nb ? ", lab B1's nominal state" : ""}. The mixed-integer problem is solved exactly by its two branches: ${none ? "neither is feasible, so no modelled longitudinal manoeuvre avoids the bicycle from here." : both ? `both are feasible and the cheaper, ${chosen}, is chosen; the weights decide the preference, never the feasibility.` : `only ${chosen} is feasible, so the choice is forced.`} Under the Euler model the terminal position can only lie in [[[${fmt(lo, 1)}, ${fmt(hi, 1)}]]], whatever the inputs: ${hi < D_MIN ? "[[d_{\\min}]] is out of reach, so passing is impossible regardless of the cost" : lo > D_MAX ? "[[d_{\\max}]] is out of reach, so yielding is impossible regardless of the cost" : "both bounds are reachable"}${hi < D_MAX && y.feasible ? ", and the yield bound is not even active: the plan is pure speed tracking" : ""}.`);
}
["#bD", "#bV", "#bMu", "#bR"].forEach(id => $(id).addEventListener("input", drawBranches));
$$("#brW [data-preset]").forEach(b => b.addEventListener("click", () => { const [d, v, m, r] = b.dataset.preset.split(",").map(Number); $("#bD").value = d; $("#bV").value = v; $("#bMu").value = m; $("#bR").value = r; drawBranches(); }));

function drawAll() { drawHero(); drawBranches(); }
drawAll();
onRedraw(drawAll);
})();
