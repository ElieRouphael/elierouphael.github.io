/* Advanced MPC and Learning MPC, lab B5, MPC, Learning MPC and reinforcement learning: the demos.
   The three terminal designs on the lattice with closed loops from every feasible start (hero), the LMPC loop
   on the lattice, and tabular Q-learning with the page's own random stream against the numbers script's logs
   (b05-data.js). Needs ../kit/kit.js, mpc-lib.js, mpc-dp.js and b05-data.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { SYS, vertices, legend, drawPoly, clickPoint } = window.MPC;
const DP = window.DP, D = window.CH08_DATA;
const LQ = SYS.lqr;

const L3 = DP.buildLattice(0.5), L4 = DP.buildLattice(0.25);
const HARD = DP.hardValueIteration(L4), JSTAR4 = HARD[HARD.length - 1], FIN4 = JSTAR4.map(Number.isFinite);
const XPX4 = L4.S.map(s => s[0] * (LQ.P[0][0] * s[0] + LQ.P[0][1] * s[1]) + s[1] * (LQ.P[1][0] * s[0] + LQ.P[1][1] * s[1]));
const IN_O = L4.S.map(s => DP.inside(DP.polys.O_inf, s));
const VC = vertices(DP.polys.C_inf), VO = vertices(DP.polys.O_inf);
const DESIGNS = { A: new Array(L4.n).fill(0), B: L4.S.map((_, i) => IN_O[i] ? XPX4[i] : Infinity), C: L4.S.map((_, i) => IN_O[i] ? JSTAR4[i] : Infinity) };
const RES = {}; for (const [name, Vf] of Object.entries(DESIGNS)) RES[name] = DP.designTable(L4, Vf, JSTAR4, FIN4, 12);
const NV = FIN4.filter(Boolean).length, NO = IN_O.filter(Boolean).length;

/* =====================================================================
   Hero: three terminal designs, every feasible start
   ===================================================================== */
const hs = { design: "A", N: 1, probe: L4.S.findIndex(s => s[0] === -1 && s[1] === 2) };
function drawHero() {
  const row = RES[hs.design].rows[hs.N - 1], cv = $("#desPlane"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.1 : 1.5, { x: [-4.4, 4.4], y: [-3.3, 3.3], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, -2, -1, 0, 1, 2, 3] }), { ctx, X, Y } = fr; cv._fr = fr;
  const cell = Math.min(X(0.25) - X(0), Y(0) - Y(0.25)) * 0.8, byStart = new Map(row.rh.map(r => [r.start, r]));
  L4.S.forEach((s, i) => {
    const r = byStart.get(i); let col;
    if (!r) col = alpha("--ink", 0.1);
    else if (!r.ok) col = css("--warm");
    else if (Math.abs(r.cost - JSTAR4[i]) < 1e-9) col = alpha("--post", 0.55);
    else col = alpha("--c5", 0.75);
    ctx.fillStyle = col; ctx.fillRect(X(s[0]) - cell / 2, Y(s[1]) - cell / 2, cell, cell);
  });
  drawPoly(fr, VC, null, css("--ink"), 2); drawPoly(fr, VO, null, css("--ink"), 1.3, [5, 4]);
  /* the probe's closed loop under this design */
  const p = hs.probe, traj = [p]; let s = p, lost = false;
  for (let t = 0; t < 80 && s !== L4.origin; t++) { if (!row.F[s]) { lost = true; break; } const sn = L4.succ[s][row.act[s]]; if (sn < 0) { lost = true; break; } s = sn; traj.push(s); }
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.6; ctx.beginPath(); traj.forEach((i, t) => t ? ctx.lineTo(X(L4.S[i][0]), Y(L4.S[i][1])) : ctx.moveTo(X(L4.S[i][0]), Y(L4.S[i][1]))); ctx.stroke();
  ctx.fillStyle = css("--bg"); ctx.strokeStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(L4.S[p][0]), Y(L4.S[p][1]), 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  legend(ctx, ["blue: closed loop optimal · purple: converges, suboptimal · orange: loses feasibility", "grey: infeasible at the start · solid: C_∞ · dashed: O_∞ = X_f of B and C · ring: the probe"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* fraction optimal against N */
  const cv2 = $("#desC"), fr2 = frame(cv2, narrow ? 1.5 : 2.6, { x: [1, 12], y: [0, 1.03], xlabel: "horizon N", ylabel: "fraction of viable states", xticks: [1, 2, 3, 4, 5, 6, 8, 10, 12], yticks: [0, 0.5, 1] });
  [["A", css("--warm"), []], ["B", css("--post"), []], ["C", css("--ink"), [5, 4]]].forEach(([name, col, dash]) => plotLine(fr2, RES[name].rows.map(r => r.N), RES[name].rows.map(r => r.optimal / NV), { color: col, width: name === hs.design ? 2.8 : 1.6, dash }));
  fr2.ctx.strokeStyle = alpha("--ink", 0.3); seg(fr2.ctx, fr2.X(hs.N), fr2.pad.t, fr2.X(hs.N), fr2.Y(0));
  legend(fr2.ctx, ["orange: A (no terminal ingredients) · blue: B (x'Px on O_∞) · dashed: C (J* on O_∞) · thick: selected"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  /* readouts */
  $$("#demo [data-design]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.design === hs.design))); $("#hNV").textContent = hs.N;
  const nz = row.rh.filter(r => r.ok && r.start !== L4.origin), worst = nz.length ? Math.max(...nz.map(r => r.cost / JSTAR4[r.start])) : NaN;
  $("#rFeas").textContent = `${row.feasible} of ${NV} viable`; $("#rDoom").textContent = row.doomed; $("#rFail").textContent = row.failed; $("#rOpt").textContent = row.optimal; $("#rWorst").textContent = Number.isFinite(worst) ? fmt(worst, 4) : "–";
  const r = byStart.get(p), sp = L4.S[p], dname = hs.design === "A" ? "design A ([[V_f = 0]], no terminal set)" : hs.design === "B" ? "design B ([[V_f = x^\\top P x]] on [[\\mathcal O_\\infty]])" : "design C ([[V_f = J^\\star]] on [[\\mathcal O_\\infty]])";
  const chk = RES[hs.design], firstOpt = chk.rows.find(x => x.optimal === NV);
  let text = `Horizon [[N = ${hs.N}]], ${dname}: the MPC problem is feasible at ${row.feasible} lattice states${row.doomed ? `, ${row.doomed} of them doomed (outside [[\\mathcal C_\\infty]], bound to leave the box)` : ""}; the closed loop loses feasibility from ${row.failed} of them and is exactly optimal from ${row.optimal}${Number.isFinite(worst) ? `, the worst converging loop costing ${fmt(worst, 3)} times [[J^\\star]]` : ""}. `;
  text += `The probe at [[(${fmt(sp[0], 2)}, ${fmt(sp[1], 2)})]] is ${!r ? "infeasible at the start: no [[N]]-step plan ends in the terminal set" : lost ? `feasible at the start but loses feasibility after ${traj.length - 1} step${traj.length === 2 ? "" : "s"}: feasible now did not mean feasible later` : `driven to the origin in ${traj.length - 1} steps at ${fmt(r.cost, 3)} against [[J^\\star = ${fmt(JSTAR4[p], 3)}]]`}. `;
  text += hs.design === "A" ? `Design A carries no certificate: the three checks of the decrease condition ([[J_N \\ge J^\\star]], [[J_N]] non-increasing in [[N]], closed-loop cost at most [[J_N(x_0)]]) all fail, and the horizon ${firstOpt ? `[[N = ${firstOpt.N}]]` : "none up to 12"} from which every loop is optimal was found by running every start, not proved.`
    : hs.design === "B" ? `Design B never fails, since [[\\mathcal O_\\infty]] is control invariant on the lattice (${L4.S.filter((_, i) => IN_O[i] && L4.succ[i].some(x => x >= 0 && IN_O[x])).length} of ${NO} terminal states have a lattice input that stays inside), yet its three checks fail too: rounding the LQR input to the lattice breaks the exact decrease at ${DP.bellmanHard(L4, DESIGNS.B).TJ.filter((v, i) => IN_O[i] && v - DESIGNS.B[i] > 1e-12).length} of ${NO} terminal states. The loops are almost identical to design C's; the certificate is not.`
    : `Design C never fails and passes all three checks (${[chk.upper, chk.mono, chk.bounded].map(b => b ? "yes" : "no").join(", ")}): with the exact tail, [[T^N \\bar V_f = J^\\star]] wherever an optimal trajectory enters [[\\mathcal O_\\infty]] within [[N]] steps, so the loop is optimal from ${row.optimal} states already at this horizon.`;
  caption($("#heroStatus"), text + " Tap a state to follow its closed loop.");
}
$("#hN").addEventListener("input", () => { hs.N = +$("#hN").value; drawHero(); });
$$("#demo [data-design]").forEach(b => b.addEventListener("click", () => { hs.design = b.dataset.design; drawHero(); }));
$("#desPlane").addEventListener("click", ev => { const p = clickPoint(ev); if (!p) return; let best = 0, bd = Infinity; L4.S.forEach((s, i) => { const d = Math.hypot(s[0] - p[0], s[1] - p[1]); if (d < bd) { bd = d; best = i; } }); hs.probe = best; drawHero(); });
$("#hReset").addEventListener("click", () => { hs.design = "A"; hs.N = 1; $("#hN").value = 1; hs.probe = L4.S.findIndex(s => s[0] === -1 && s[1] === 2); drawHero(); });

/* =====================================================================
   Part 4: LMPC on the lattice learns an upper bound
   ===================================================================== */
function drawLMPC() {
  const sStart = L4.S.findIndex(s => s[0] === -4 && s[1] === 0), w = +$("#lmW").value, seed = DP.sluggishSeed(L4, sStart, w), results = {};
  for (const NL of [1, 2, 3, 4]) results[NL] = DP.lmpcLattice(L4, JSTAR4, { st: seed.st.slice(), ac: seed.ac.slice() }, NL, 9);
  const cv = $("#lmC"), narrow = narrowOf(cv), top = results[1].costs[0] * 1.08, fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, 8], y: [0, top], xlabel: "iteration j", ylabel: "closed-loop cost from x_S", xticks: [0, 2, 4, 6, 8] }), { ctx, X, Y } = fr;
  [1, 2, 3, 4].forEach((NL, c) => plotLine(fr, results[NL].costs.map((_, j) => j), results[NL].costs, { color: alpha("--post", 0.3 + 0.7 * c / 3), width: 2 }));
  ctx.strokeStyle = css("--ink"); ctx.setLineDash([5, 4]); seg(ctx, X(0), Y(JSTAR4[sStart]), X(8), Y(JSTAR4[sStart])); ctx.setLineDash([]);
  legend(ctx, ["LMPC on the lattice, N = 1, 2, 3, 4 (darker = longer) · dashed: J*(x_S) from value iteration"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#lmWV").textContent = w;
  const never = [1, 2, 3, 4].every(NL => results[NL].costs.every((c, j) => !j || c <= results[NL].costs[j - 1] + 1e-9));
  caption($("#lmCap"), `From [[x_S = (-4, 0)]], [[J^\\star(x_S) = ${fmt(JSTAR4[sStart], 3)}]]. The seed is the optimal trajectory for the sluggish cost [[${w}\\, x^\\top Q x + u^2]]: ${seed.ac.length} steps, true cost ${fmt(results[1].costs[0], 3)}, ${fmt(100 * (results[1].costs[0] / JSTAR4[sStart] - 1), 1)} percent above [[J^\\star]]. Iteration costs never increase (${never ? "verified for all four horizons" : "violated somewhere: check"}), and no stored cost-to-go lies below [[J^\\star]] (smallest difference ${Math.min(...[1, 2, 3, 4].map(NL => results[NL].minGap)).toExponential(1)}). The horizon decides how much of the data is usable: ${[1, 2, 3, 4].map(NL => `[[N = ${NL}]] ends at ${fmt(results[NL].costs[8], 3)} (${fmt(100 * (results[NL].costs[8] / JSTAR4[sStart] - 1), 1)} percent above)`).join(", ")}. With [[N = 1]] nothing beats replaying the seed; with a long enough horizon the first iteration is already optimal and its stored costs-to-go lie on the diagonal [[J_t^j = J^\\star(x^j[t])]], the principle of optimality visible in the data. Improvement is guaranteed, optimality is not; the convexified safe set of chapter A12 gives short horizons more room.`);
}
$("#lmW").addEventListener("input", drawLMPC);

/* =====================================================================
   Part 6: tabular Q-learning, run here with the page's own random stream
   ===================================================================== */
const QL = { cfg: 0, done: null };
const QCONF = [["α = 1, ε = 0.1", { alpha: 1, eps: 0.1 }, "alpha=1, eps=0.1"], ["α = 0.1, ε = 0.1", { alpha: 0.1, eps: 0.1 }, "alpha=0.1, eps=0.1"], ["α = n^−0.6, ε = 0.1", { alpha: "decay", eps: 0.1 }, "alpha=1/n^0.6, eps=0.1"], ["α = 1, ε = 0, Q₀ = 0 (optimistic)", { alpha: 1, eps: 0, Q0: 0 }, "alpha=1, eps=0, Q0=0 (optimistic)"], ["α = 1, ε = 0, Q₀ = 200 (pessimistic)", { alpha: 1, eps: 0, Q0: 200 }, "alpha=1, eps=0, Q0=200 (pessimistic)"]];
function drawQ() {
  if (!QL.done) {
    const Pb = DP.crashProblem(L3, 0.95, 100), { J: Jstar } = DP.valueIteration(Pb, new Array(L3.n + 1).fill(0)), V3 = DP.viable(L3);
    const Qsa = Pb.cost.map((r, i) => r.map((c, a) => c + 0.95 * Jstar[Pb.succ[i][a]])), optAct = Qsa.map((r, i) => r.map(v => v <= Jstar[i] + 1e-9));
    QL.done = QCONF.map(([, cfg]) => DP.qLearning(Pb, L3, V3, optAct, cfg, 200000)); QL.V3 = V3;
  }
  const cv = $("#qC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, 200000], y: [0, 1.03], xlabel: "transitions sampled", ylabel: "fraction of viable states", xticks: [0, 50000, 100000, 150000, 200000], yticks: [0, 0.5, 1], xfmt: v => (v / 1000) + "k" }), { ctx } = fr;
  const cols = [css("--post"), alpha("--post", 0.5), css("--warm"), css("--ink"), alpha("--ink", 0.45)];
  QL.done.forEach((run, c) => plotLine(fr, run.log.map(l => l[0]), run.log.map(l => l[1]), { color: cols[c], width: c === QL.cfg ? 2.6 : 1.3 }));
  const nb = D.qlogs[QCONF[QL.cfg][2]]; plotLine(fr, nb.map(l => l[0]), nb.map(l => l[1]), { color: cols[QL.cfg], width: 1.2, dash: [3, 3] });
  legend(ctx, ["greedy input of the learned Q is optimal · thick: selected configuration (this run) · dotted: the numbers script's run of it"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const run = QL.done[QL.cfg], hit = run.log.find(l => l[1] === 1), nbHit = nb.find(l => l[1] === 1), last = run.log[run.log.length - 1], nbLast = nb[nb.length - 1];
  const greedyCrash = QL.V3.reduce((s, v, i) => { if (!v) return s; let g = 0; for (let k = 1; k < L3.m; k++) if (run.Q[i][k] < run.Q[i][g]) g = k; return s + (L3.succ[i][g] < 0 ? 1 : 0); }, 0);
  $$("#qW [data-cfg]").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.cfg === QL.cfg)));
  caption($("#qCap"), `${QCONF[QL.cfg][0]}, 200 000 transitions, episodes from random states ending after 30 steps or at a crash. This run: ${hit ? `the greedy policy is optimal at every viable state from step ${hit[0].toLocaleString("en-US")}, after ${hit[2].toLocaleString("en-US")} crashes` : `the greedy policy is never optimal everywhere, final fraction ${fmt(last[1], 3)}`}; ${last[2].toLocaleString("en-US")} crashes in all; the greedy input crashes from ${greedyCrash} viable states. The numbers script's run with NumPy's stream, seed 0: ${nbHit ? `optimal from step ${nbHit[0].toLocaleString("en-US")} after ${nbHit[2].toLocaleString("en-US")} crashes` : `never, final fraction ${fmt(nbLast[1], 3)}`}, ${nbLast[2].toLocaleString("en-US")} crashes. The two streams differ, the conclusions do not: ${QL.cfg === 0 ? "with [[\\alpha = 1]], exact for deterministic transitions, and a little exploration, samples replace the model, at the price of hundreds of crashes where value iteration needed 6 sweeps (6 450 backups) and none." : QL.cfg === 1 ? "the constant [[\\alpha = 0.1]] averages targets that are exact anyway, and is still short of optimal after 200 000 transitions." : QL.cfg === 2 ? "the decaying step size satisfies the convergence conditions but learns more slowly than [[\\alpha = 1]] on a deterministic problem." : QL.cfg === 3 ? "with no exploration at all, the optimistic start [[Q_0 = 0]], below every [[Q^\\star]], makes untried inputs look attractive and still succeeds: initialisation is a form of exploration." : "the pessimistic start [[Q_0 = 200]] fails: a crash target is only its cost (at most 118), while other targets bootstrap from successors still valued near 200, so crashing looks cheap and the greedy policy crashes from viable states."} Learning is off-policy: the limit is [[Q^\\star]] whatever policy generated the data, and [[Q]] is accurate only where it matters, at the optimal input of each viable state.`);
}
$$("#qW [data-cfg]").forEach(b => b.addEventListener("click", () => { QL.cfg = +b.dataset.cfg; drawQ(); }));

function drawAll() { drawHero(); drawLMPC(); drawQ(); }
drawAll();
onRedraw(drawAll);
})();
