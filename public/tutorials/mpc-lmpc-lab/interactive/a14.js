/* Advanced MPC and Learning MPC, chapter A14, Bellman: dynamic programming, value and policy iteration: the demos.
   The lattice value iteration (hero), the integer chain, the Riccati recursion, and discounted value and policy
   iteration on the h = 0.5 lattice. Everything is tabular and exact. Needs ../kit/kit.js, mpc-lib.js and mpc-dp.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, clamp } = window.Kit;
const { SYS, mm, tr, sub, vertices, legend, drawPoly, clickPoint, fmtM } = window.MPC;
const DP = window.DP;
const LQ = SYS.lqr, Q = SYS.Q, R = SYS.R;

const L3 = DP.buildLattice(0.5), L4 = DP.buildLattice(0.25);
const HARD = DP.hardValueIteration(L4), JSTAR4 = HARD[HARD.length - 1], FIN4 = JSTAR4.map(Number.isFinite), MU4 = DP.bellmanHard(L4, JSTAR4).act;
const XPX4 = L4.S.map(s => s[0] * (LQ.P[0][0] * s[0] + LQ.P[0][1] * s[1]) + s[1] * (LQ.P[1][0] * s[0] + LQ.P[1][1] * s[1]));
const IN_O = L4.S.map(s => DP.inside(DP.polys.O_inf, s));
const VC = vertices(DP.polys.C_inf), VO = vertices(DP.polys.O_inf);
$("#hK").max = HARD.length - 1; $("#hK").value = HARD.length - 1;

/* =====================================================================
   Hero: the constrained double integrator's cost-to-go on the lattice
   ===================================================================== */
const hs = { k: HARD.length - 1, view: "value", probe: L4.S.findIndex(s => s[0] === -4 && s[1] === 0) };
function drawHero() {
  const k = hs.k, Jk = HARD[k], cv = $("#latC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.1 : 1.5, { x: [-4.4, 4.4], y: [-3.3, 3.3], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, -2, -1, 0, 1, 2, 3] }), { ctx, X, Y } = fr; cv._fr = fr;
  const cell = Math.min(X(0.25) - X(0), Y(0) - Y(0.25)) * 0.8, lo = Math.log(0.02), hi = Math.log(Math.max(...JSTAR4.filter(Number.isFinite)));
  const actK = hs.view === "policy" ? (k === HARD.length - 1 ? MU4 : DP.bellmanHard(L4, Jk).act) : null;
  L4.S.forEach((s, i) => {
    const v = Jk[i]; let col;
    if (!Number.isFinite(v)) col = alpha("--ink", 0.12);
    else if (hs.view === "value") { const t = v <= 0.02 ? 0 : (Math.log(v) - lo) / (hi - lo); col = alpha("--post", 0.15 + 0.85 * clamp(t, 0, 1)); }
    else { const u = L4.U[actK[i]]; col = u > 0 ? alpha("--post", 0.2 + 0.8 * u) : u < 0 ? alpha("--warm", 0.2 - 0.8 * u) : alpha("--ink", 0.25); }
    ctx.fillStyle = col; ctx.fillRect(X(s[0]) - cell / 2, Y(s[1]) - cell / 2, cell, cell);
  });
  drawPoly(fr, VC, null, css("--ink"), 2); drawPoly(fr, VO, null, css("--ink"), 1.3, [5, 4]);
  const p = hs.probe, traj = [p]; let s = p;
  for (let t = 0; t < 60 && s !== L4.origin && FIN4[s]; t++) { s = L4.succ[s][MU4[s]]; if (s < 0) break; traj.push(s); }
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; ctx.beginPath(); traj.forEach((i, t) => t ? ctx.lineTo(X(L4.S[i][0]), Y(L4.S[i][1])) : ctx.moveTo(X(L4.S[i][0]), Y(L4.S[i][1]))); ctx.stroke();
  ctx.fillStyle = css("--bg"); ctx.strokeStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(L4.S[p][0]), Y(L4.S[p][1]), 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  legend(ctx, [hs.view === "value" ? "squares: J_k (darker = costlier, grey = infinite) · solid: C_∞ · dashed: O_∞ · ring: the probe" : "blue: u > 0, orange: u < 0 (darker = larger), grey: u = 0 · solid: C_∞ · dashed: O_∞"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#hKV").textContent = k; $$("#demo [data-view]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.view === hs.view)));
  const ck = DP.polys.C_hist[Math.min(k, DP.polys.C_hist.length - 1)], dom = Jk.filter(Number.isFinite).length, inC = L4.S.filter(x => DP.inside(ck, x)).length;
  $("#rDom").textContent = `${dom} / ${inC}`; $("#rFin").textContent = `${FIN4.filter(Boolean).length} of ${L4.n}`;
  $("#rJ").textContent = Number.isFinite(JSTAR4[p]) ? fmt(JSTAR4[p], 3) : "∞"; $("#rU").textContent = Number.isFinite(JSTAR4[p]) ? fmt(L4.U[MU4[p]], 2) : "–";
  const sp = L4.S[p], gapO = IN_O[p] && XPX4[p] > 0 ? (JSTAR4[p] - XPX4[p]) / XPX4[p] : null, inO = IN_O.filter(Boolean).length;
  const gaps = L4.S.map((_, i) => IN_O[i] && XPX4[i] > 0 ? (JSTAR4[i] - XPX4[i]) / XPX4[i] : null).filter(v => v !== null).sort((a, b) => a - b);
  caption($("#heroStatus"), `The lattice with [[h = 0.25]]: ${L4.n} states, ${L4.m} inputs, mapped into itself exactly by [[x^+ = Ax + Bu]], so every lattice trajectory is a true trajectory of the double integrator. Undiscounted value iteration from [[J_0 = 0]] with hard constraints: at step [[k = ${k}]] the finite-cost domain has ${dom} states, exactly the lattice points of the polytope [[\\mathcal C_{${Math.min(k, DP.polys.C_hist.length - 1)}}]] of chapter A10 (${inC}); the fixed point is reached at [[k = ${HARD.length - 1}]] with ${FIN4.filter(Boolean).length} viable states, the lattice points of [[\\mathcal C_\\infty]] (solid). The grey states satisfy the constraints now but are doomed to leave the box within three steps. The probe at [[(${fmt(sp[0], 2)}, ${fmt(sp[1], 2)})]] has ${Number.isFinite(JSTAR4[p]) ? `[[J^\\star = ${fmt(JSTAR4[p], 3)}]], optimal input [[u = ${fmt(L4.U[MU4[p]], 2)}]] and reaches the origin in ${traj.length - 1} steps; the unconstrained [[x^\\top P x = ${fmt(XPX4[p], 3)}]] is a lower bound${gapO !== null ? `, and inside [[\\mathcal O_\\infty]] (dashed) the lattice value exceeds it by ${fmt(100 * gapO, 1)} percent, the price of rounding the inputs` : ""}` : "infinite cost: no admissible input sequence keeps it in the box"}. Over the ${inO} lattice states of [[\\mathcal O_\\infty]] the relative gap has median ${fmt(100 * gaps[Math.floor(gaps.length / 2)], 2)} percent and maximum ${fmt(100 * gaps[gaps.length - 1], 1)} percent. The policy is bang-bang except near the origin. Tap a state to move the probe; the slider replays the iterations.`);
}
$("#hK").addEventListener("input", () => { hs.k = +$("#hK").value; drawHero(); });
$$("#demo [data-view]").forEach(b => b.addEventListener("click", () => { hs.view = b.dataset.view; drawHero(); }));
$("#latC").addEventListener("click", ev => { const p = clickPoint(ev); if (!p) return; let best = 0, bd = Infinity; L4.S.forEach((s, i) => { const d = Math.hypot(s[0] - p[0], s[1] - p[1]); if (d < bd) { bd = d; best = i; } }); hs.probe = best; drawHero(); });
$("#hReset").addEventListener("click", () => { hs.k = HARD.length - 1; $("#hK").value = hs.k; hs.view = "value"; hs.probe = L4.S.findIndex(s => s[0] === -4 && s[1] === 0); drawHero(); });

/* =====================================================================
   Part 1: the integer chain
   ===================================================================== */
function drawChain() {
  const N = +$("#chN").value, C = DP.chainDP(N), cv = $("#chainC"), narrow = narrowOf(cv), top = Math.max(...C.J[0].filter(Number.isFinite)) * 1.05;
  const fr = frame(cv, narrow ? 1.4 : 2.2, { x: [-10.5, 10.5], y: [0, top], xlabel: "state x", ylabel: "cost-to-go J_k(x)", xticks: [-10, -5, 0, 5, 10] }), { ctx } = fr;
  for (let k = N; k >= 0; k--) { const xs = C.xs.filter((_, i) => Number.isFinite(C.J[k][i])), ys = C.J[k].filter(Number.isFinite); plotLine(fr, xs, ys, { color: alpha("--post", 0.25 + 0.75 * (N - k) / N), width: 1.6 }); }
  legend(ctx, ["one curve per stage, darker = more steps to go (k = 0 is the full problem)"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#chNV").textContent = N; const dom0 = C.xs.filter((_, i) => Number.isFinite(C.J[0][i])), i2 = C.xs.indexOf(2);
  caption($("#chainCap"), `Integer positions [[|x| \\le 10]], moves [[u \\in \\{-2, \\ldots, 2\\}]], cost [[x^2 + u^2]], horizon [[N = ${N}]], terminal constraint [[x_N = 0]]. Dynamic programming needs [[${N} \\cdot 21 \\cdot 5 = ${N * 105}]] stage-cost evaluations against [[${N} \\cdot 5^{${N}} \\cdot 21 = ${(N * 5 ** N * 21).toLocaleString("en-US")}]] for enumerating every input sequence from every state${N === 5 ? ", which the numbers script does to confirm that both agree on every state" : ""}. The domains are the controllable sets [[|x| \\le 2(N - k)]]: at [[k = 0]], ${dom0.length ? `[[|x| \\le ${Math.max(...dom0)}]]` : "nothing"}. From [[x = 2]] the optimal move is ${C.mu.map(m => m[i2]).map((u, k) => `${u === null ? "none" : u} at k = ${k}`).join(", ")}: the policy is time varying, saving the big move for the last step${N >= 2 ? ", because [[J_0(2) = " + fmt(C.J[0][i2], 0) + "]] is cheaper than any sequence that arrives early and waits at the origin paying [[u^2]] later" : ""}.`);
}
$("#chN").addEventListener("input", drawChain);

/* =====================================================================
   Part 2: the Riccati recursion
   ===================================================================== */
const norm2 = M => { const [[a, b], [c, d]] = mm(tr(M), M), t = a + d, q = Math.sqrt(Math.max(0, (a - d) ** 2 / 4 + b * c)); return Math.sqrt(Math.max(0, t / 2 + q)); };
function riccatiStep(Pn) { const r = window.MPC.riccatiRecursion(SYS.A, SYS.B, Q, [[R]], Pn, 1); return { P: r.Ps[0], K: r.Ks[0][0] }; }
function drawRiccati() {
  const choice = $("#ricPf").value, P = LQ.P, Pf = { zero: [[0, 0], [0, 0]], Q, tenP: P.map(r => r.map(v => 10 * v)), P, half: P.map(r => r.map(v => 0.5 * v)) }[choice], nBack = 25;
  let Pj = Pf.map(r => r.slice()); const errs = [norm2(sub(Pf, P))], gains = [];
  for (let j = 0; j < nBack; j++) { const r = riccatiStep(Pj); Pj = r.P; errs.push(norm2(sub(Pj, P))); gains.push(r.K); }
  const cv = $("#ricC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, nBack], y: [-17, 3], xlabel: "backward steps j = N − k", ylabel: "log10 ‖P_(N−j) − P‖", xticks: [0, 5, 10, 15, 20, 25], yticks: [-16, -12, -8, -4, 0] }), { ctx } = fr;
  plotLine(fr, errs.map((_, j) => j), errs.map(e => Math.log10(Math.max(e, 1e-17))), { color: css("--post"), width: 2 });
  let Acl = [[1, 0], [0, 1]]; const bound = []; for (let j = 0; j <= nBack; j++) { bound.push(Math.log10(Math.max(norm2(Acl) ** 2 * errs[0], 1e-17))); Acl = mm(Acl, LQ.Acl); }
  if (errs[0] > 0) plotLine(fr, bound.map((_, j) => j), bound, { color: alpha("--ink", 0.5), width: 1.4, dash: [5, 4] });
  legend(ctx, ["blue: Riccati error · dashed: bound ‖A_cl^j‖² ‖P_f − P‖"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const rho = Math.hypot(...LQ.eig[0]), rate = errs[2] > 1e-14 && errs[12] > 1e-16 ? (errs[12] / errs[2]) ** 0.1 : null;
  caption($("#ricCap"), `The canonical system, [[Q = \\operatorname{diag}(1, 0.2)]], [[R = 0.2]]: the DARE solution is [[P = ${fmtM(P, 4)}]], [[K = (${fmt(LQ.K[0][0], 4)}, ${fmt(LQ.K[0][1], 4)})]], [[\\rho(A_{\\mathrm{cl}}) = ${fmt(rho, 4)}]], [[\\rho^2 = ${fmt(rho ** 2, 4)}]]. ${choice === "P" ? "Starting at [[P_f = P]] the recursion never moves: the error is zero at every step, which is why an unconstrained MPC with terminal weight [[P]] reproduces the LQR law at every horizon." : `From this [[P_f]] the error after 25 backward steps is ${errs[nBack].toExponential(1)}, machine precision, and the gains match [[K]] to ${Math.max(...gains[nBack - 1].map((g, i) => Math.abs(g - LQ.K[0][i]))).toExponential(1)}. ${rate ? `Over [[j = 2, \\ldots, 12]] the error shrinks by a factor ${fmt(rate, 3)} per step, close to [[\\rho(A_{\\mathrm{cl}})^2 = ${fmt(rho ** 2, 3)}]]; single steps oscillate because [[A_{\\mathrm{cl}}]] has complex eigenvalues.` : ""} ${choice === "tenP" ? "Starting above [[P]], the iterates stay above it and the dashed bound [[\\|A_{\\mathrm{cl}}^j\\|_2^2 \\|P_f - P\\|_2]] holds at every step, as the sandwich inequality promises." : "Starting below [[P]] the iterates rise towards it; the bound is proved for starts above [[P]], but the asymptotic rate is the same since [[K_X \\to K]]."}`}`);
}
$("#ricPf").addEventListener("change", drawRiccati);

/* =====================================================================
   Parts 4 and 5: discounted value iteration and policy iteration on the h = 0.5 lattice
   ===================================================================== */
const vi = { gamma: 0.95, crash: 100 };
function drawVI() {
  const Pb = DP.crashProblem(L3, vi.gamma, vi.crash), { J: Jstar, iterations } = DP.valueIteration(Pb, new Array(L3.n + 1).fill(0)), V3 = DP.viable(L3), muStar = DP.bellman(Pb, Jstar).act;
  const Qsa = Pb.cost.map((r, i) => r.map((c, a) => c + vi.gamma * Jstar[Pb.succ[i][a]])), optAct = Qsa.map((r, i) => r.map(v => v <= Jstar[i] + 1e-9));
  const maxViable = Math.max(...Jstar.filter((_, i) => i < L3.n && V3[i])), minNon = Math.min(...Jstar.filter((_, i) => i < L3.n && !V3[i])), keeps = V3.every((v, i) => !v || (L3.succ[i][muStar[i]] >= 0 && V3[L3.succ[i][muStar[i]]]));
  const starts = [["J₀ = 0", () => 0], ["J₀ = 2000", () => 2000], ["J₀ random in [0, 2000]", (() => { let s = 12345; return () => { s = (s * 1103515245 + 12345) % 2147483648; return 2000 * s / 2147483648; }; })()]];
  const cv = $("#viC"), narrow = narrowOf(cv), K = 150, fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, K], y: [-16, 4], xlabel: "value-iteration step k", ylabel: "log10 ‖J_k − J*‖", xticks: [0, 50, 100, 150], yticks: [-16, -12, -8, -4, 0, 4] }), { ctx } = fr, res = [];
  starts.forEach(([name, gen], c) => {
    let J = Array.from({ length: L3.n + 1 }, gen); const e0 = Math.max(...J.map((v, i) => Math.abs(v - Jstar[i]))), errs = [e0]; let kPol = null, kExact = null;
    for (let k = 1; k <= K; k++) { const { TJ, act } = DP.bellman(Pb, J); J = TJ; const e = Math.max(...J.map((v, i) => Math.abs(v - Jstar[i]))); errs.push(e); if (kPol === null && act.every((a, i) => optAct[i][a])) kPol = k; if (kExact === null && e === 0) kExact = k; }
    const col = [css("--post"), css("--warm"), css("--ink")][c];
    plotLine(fr, errs.map((_, k) => k), errs.map(e => Math.log10(Math.max(e, 1e-16))), { color: col, width: 2 });
    plotLine(fr, errs.map((_, k) => k), errs.map((_, k) => Math.log10(Math.max(e0 * vi.gamma ** k, 1e-16))), { color: col, width: 1, dash: [4, 4] });
    res.push({ name, e0, ratio: Math.max(...errs.slice(1).map((e, k) => e / (e0 * vi.gamma ** (k + 1)))), kPol, kExact });
  });
  legend(ctx, ["solid: error · dashed: bound γ^k ‖J₀ − J*‖ · blue: J₀ = 0 · orange: J₀ = 2000 · black: random J₀"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const trace = DP.policyIteration(Pb, new Array(L3.n + 1).fill(Math.floor(L3.m / 2)), Jstar);
  $("#piT").innerHTML = "<thead><tr><th>k</th><th>‖J_μk − J*‖∞</th><th>actions changed</th></tr></thead><tbody>" + trace.map(t => `<tr><td>${t.k}</td><td>${t.err.toExponential(3)}</td><td>${t.changed}</td></tr>`).join("") + "</tbody>";
  $("#viGV").textContent = fmt(vi.gamma, 2); $("#viCV").textContent = vi.crash;
  caption($("#viCap"), `The [[h = 0.5]] lattice has ${L3.n} states and ${L3.m} inputs; leaving the box costs [[c_{\\mathrm{crash}} = ${vi.crash}]] and ends in an absorbing, cost-free crash state, with [[\\gamma = ${fmt(vi.gamma, 2)}]]. [[J^\\star]] is certified after ${iterations} iterations from zero by the stopping rule [[\\frac{\\gamma}{1 - \\gamma}\\|J_{k+1} - J_k\\| < 10^{-12}]]. ${V3.filter(Boolean).length} states are viable (some input sequence keeps them in the box for ever); there [[J^\\star \\le ${fmt(maxViable, 2)}]], elsewhere [[J^\\star \\ge ${fmt(minNon, 2)}]]${keeps ? ", and the optimal policy keeps every viable state viable: the penalty does not alter the constrained behaviour" : ", but the optimal policy crashes from some viable state: the penalty is too small to stand in for the constraint"}. ${res.map(r => `From ${r.name} (error ${fmt(r.e0, 1)}) the error stays within the bound (largest ratio ${fmt(r.ratio, 3)}), the greedy policy is optimal from step ${r.kPol ?? "never"} and the values are exact from step ${r.kExact ?? "never within 150"}`).join("; ")}. The bound is tight from the constant 2000: at the absorbing, cost-free origin [[T]] acts as [[J \\mapsto \\gamma J]]. Policies converge before values, because a constant offset in [[J]] does not change the minimiser. Policy iteration from "coast" ([[u = 0]] everywhere) takes ${trace.length - 1} improvement steps, changing ${trace.slice(0, -1).map(t => t.changed).join(", ")} actions, in a policy space of [[${L3.m}^{${L3.n}} \\approx 10^{${fmt(L3.n * Math.log10(L3.m), 0)}}]].`);
}
$("#viG").addEventListener("input", () => { vi.gamma = +$("#viG").value; drawVI(); });
$("#viCr").addEventListener("input", () => { vi.crash = +$("#viCr").value; drawVI(); });

function drawAll() { drawHero(); drawChain(); drawRiccati(); drawVI(); }
drawAll();
onRedraw(drawAll);
})();
