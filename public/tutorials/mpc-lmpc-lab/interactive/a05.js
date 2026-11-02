/* Advanced MPC and Learning MPC, chapter A5, the finite-horizon problem: the demos.
   The hero assembles the complete constrained finite-time optimal control problem of the canonical
   system as one condensed QP (box on every predicted state, input and rate bounds, an optional
   terminal equality) and solves it with the toolkit's interior-point method, multipliers included;
   a second widget softens a comfort bound and shows the exact-penalty threshold.
   Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, fit, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { SYS, mv, mm, tr, dot, zeros, quad, predictionMatrices, solveQP, legend } = window.MPC;

const kindName = { position: "position bound", velocity: "velocity bound", rate: "rate bound", input: "input bound" };
function differenceMatrix(N) { const D = zeros(N, N); for (let k = 0; k < N; k++) { D[k][k] = 1; if (k) D[k][k - 1] = -1; } return D; }

/* the condensed CFTOC on the canonical system (P = Q): returns H, f, G, g, tags, E, e and the dropped constant */
function buildCFTOC({ N = 8, x0 = [-3, 0], umax = 1, dumax = 0.5, vmax = 3, terminal = true, uPrev = 0, soft = null } = {}) {
  const A = SYS.A, B = SYS.B, Q = SYS.Q, R = SYS.R, n = 2, { Sx, Su } = predictionMatrices(A, B, N);
  const Qbar = zeros(N * n, N * n); for (let i = 0; i < N; i++) for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Qbar[i * n + r][i * n + c] = Q[r][c];
  const SuTQ = mm(tr(Su), Qbar), H0 = mm(SuTQ, Su).map((r, i) => r.map((v, j) => 2 * (v + (i === j ? R : 0)))), f0 = mv(mm(SuTQ, Sx), x0).map(v => 2 * v);
  const constant = quad(mm(tr(Sx), mm(Qbar, Sx)), x0) + quad(Q, x0), Sxx = mv(Sx, x0);
  const G = [], g = [], tags = [];
  for (let k = 0; k < N; k++) { const rp = Su[k * n], rv = Su[k * n + 1]; G.push(rp); g.push(4 - Sxx[k * n]); tags.push({ kind: "position", step: k + 1 }); G.push(rp.map(v => -v)); g.push(4 + Sxx[k * n]); tags.push({ kind: "position", step: k + 1 }); G.push(rv); g.push(vmax - Sxx[k * n + 1]); tags.push({ kind: "velocity", step: k + 1 }); G.push(rv.map(v => -v)); g.push(vmax + Sxx[k * n + 1]); tags.push({ kind: "velocity", step: k + 1 }); }
  const D = differenceMatrix(N);
  for (let k = 0; k < N; k++) { const off = k === 0 ? -uPrev : 0; G.push(D[k]); g.push(dumax - off); tags.push({ kind: "rate", step: k }); G.push(D[k].map(v => -v)); g.push(dumax + off); tags.push({ kind: "rate", step: k }); }
  for (let k = 0; k < N; k++) { const ek = new Array(N).fill(0); ek[k] = 1; G.push(ek); g.push(umax); tags.push({ kind: "input", step: k }); G.push(ek.map(v => -v)); g.push(umax); tags.push({ kind: "input", step: k }); }
  const E = terminal ? [Su[(N - 1) * n], Su[(N - 1) * n + 1]] : [], e = terminal ? [-Sxx[(N - 1) * n], -Sxx[(N - 1) * n + 1]] : [];
  if (soft === null) return { N, x0, Sx, Su, H: H0, f: f0, constant, G, g, tags, E, e, umax, dumax, vmax, terminal, nz: N };
  /* slacks on the 2N velocity rows with an l1 penalty */
  const vel = tags.map((t, i) => t.kind === "velocity" ? i : -1).filter(i => i >= 0), nz = N + vel.length;
  const H = zeros(nz, nz); for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) H[i][j] = H0[i][j];
  const f = [...f0, ...vel.map(() => soft)];
  const G2 = G.map(row => [...row, ...vel.map(() => 0)]); vel.forEach((i, j) => { G2[i][N + j] = -1; });
  const E2 = E.map(row => [...row, ...vel.map(() => 0)]);
  return { N, x0, Sx, Su, H, f, constant, G: G2, g, tags, E: E2, e, umax, dumax, vmax, terminal, nz, nslack: vel.length, H0, f0 };
}
function solveCFTOC(prob) {
  const lb = [...new Array(prob.N).fill(-Infinity), ...new Array(prob.nz - prob.N).fill(0)], ub = new Array(prob.nz).fill(Infinity);
  const r = solveQP(prob.H, prob.f, prob.G, prob.g, prob.E, prob.e, lb, ub), N = prob.N, n = 2, U = r.z.slice(0, N);
  const Xs = mv(prob.Sx, prob.x0).map((v, i) => v + dot(prob.Su[i], U)), X = []; for (let i = 0; i < N; i++) X.push([Xs[i * n], Xs[i * n + 1]]);
  const H0 = prob.H0 || prob.H, f0 = prob.f0 || prob.f, costQ = 0.5 * quad(H0, U) + dot(f0, U) + prob.constant;
  const active = prob.tags.map((t, i) => ({ ...t, lam: r.lam[i] })).filter(t => t.lam > 1e-6);
  return { U, X, slack: r.z.slice(N), cost: costQ, objective: r.obj, lam: r.lam, nu: r.nu, active, feasible: r.success, nit: r.nit };
}

/* ---------------- hero ---------------- */
const hero = { cache: {} };
function heroRun() {
  const N = +$("#hN").value, umax = +$("#hU").value, dumax = +$("#hD").value, p0 = +$("#hP0").value, v0 = +$("#hV0").value, terminal = $("#hTerm").getAttribute("aria-pressed") === "true";
  const key = [N, umax, dumax, p0, v0, terminal].join("|"); if (hero.cache[key]) return hero.cache[key];
  const prob = buildCFTOC({ N, umax, dumax, x0: [p0, v0], terminal }), sol = solveCFTOC(prob);
  return hero.cache[key] = { prob, sol, N, umax, dumax, p0, v0, terminal };
}
function drawHero() {
  const r = heroRun(), { prob, sol } = r, cv = $("#cftocC"), narrow = narrowOf(cv), w = cv.clientWidth || cv.parentElement.clientWidth, h = narrow ? 430 : 370, { ctx } = fit(cv, w / h);
  const padL = 44, padR = 10, ph = (h - 40) / 3, T = r.N, X = k => padL + k / T * (w - padL - padR);
  const panel = (k, y0, y1, ticks, label) => { const top = 8 + k * ph, bot = top + ph - 26, Y = v => bot - (v - y0) / (y1 - y0) * (bot - top); ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle"; ctx.textAlign = "right"; for (const t of ticks) { ctx.strokeStyle = css("--grid"); seg(ctx, padL, Y(t), w - padR, Y(t)); ctx.fillStyle = css("--muted"); ctx.fillText(String(t), padL - 5, Y(t)); } ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillText(label, padL + 4, top + 10); return { Y, top, bot }; };
  const P1 = panel(0, -5, 5, [-4, 0, 4], "position p, bound ±4"), P2 = panel(1, -3.6, 3.6, [-3, 0, 3], "velocity v, bound ±3"), P3 = panel(2, -r.umax * 1.5, r.umax * 1.5, [-r.umax, 0, r.umax].map(v => +v.toFixed(2)), "input u, bound ±" + fmt(r.umax, 2) + ", rate ±" + fmt(r.dumax, 2));
  for (const [P, v] of [[P1, 4], [P1, -4], [P2, 3], [P2, -3], [P3, r.umax], [P3, -r.umax]]) { ctx.strokeStyle = alpha("--warm", 0.7); ctx.setLineDash([5, 4]); seg(ctx, padL, P.Y(v), w - padR, P.Y(v)); ctx.setLineDash([]); }
  if (!sol.feasible) { ctx.fillStyle = css("--warm"); ctx.font = "13px " + css("--font-ui"); ctx.textAlign = "center"; ctx.fillText("infeasible: no input sequence satisfies all the constraints", w / 2, h / 2); }
  else {
    const xs = [prob.x0, ...sol.X];
    const line = (pts, Y, token) => { ctx.strokeStyle = css(token); ctx.lineWidth = 2; ctx.beginPath(); pts.forEach((v, t) => t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v))); ctx.stroke(); ctx.fillStyle = css(token); pts.forEach((v, t) => { ctx.beginPath(); ctx.arc(X(t), Y(v), 2.6, 0, 2 * Math.PI); ctx.fill(); }); };
    line(xs.map(x => x[0]), P1.Y, "--post"); line(xs.map(x => x[1]), P2.Y, "--post");
    ctx.strokeStyle = css("--c5"); ctx.lineWidth = 2; ctx.beginPath(); sol.U.forEach((u, t) => { if (t === 0) ctx.moveTo(X(0), P3.Y(u)); else { ctx.lineTo(X(t), P3.Y(sol.U[t - 1])); ctx.lineTo(X(t), P3.Y(u)); } if (t === sol.U.length - 1) ctx.lineTo(X(t + 1), P3.Y(u)); }); ctx.stroke();
    const maxLam = Math.max(1e-9, ...sol.active.map(a => a.lam));
    for (const a of sol.active) { const rad = 4 + 7 * Math.sqrt(a.lam / maxLam); ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.6; ctx.beginPath(); if (a.kind === "input") ctx.arc(X(a.step + 0.5), P3.Y(sol.U[a.step]), rad, 0, 2 * Math.PI); else if (a.kind === "rate") ctx.arc(X(a.step), P3.Y(sol.U[a.step]), rad, 0, 2 * Math.PI); else if (a.kind === "position") ctx.arc(X(a.step), P1.Y(xs[a.step][0]), rad, 0, 2 * Math.PI); else ctx.arc(X(a.step), P2.Y(xs[a.step][1]), rad, 0, 2 * Math.PI); ctx.stroke(); }
    if (r.terminal) { ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(T), P1.Y(0), 5, 0, 2 * Math.PI); ctx.fill(); ctx.beginPath(); ctx.arc(X(T), P2.Y(0), 5, 0, 2 * Math.PI); ctx.fill(); }
  }
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center"; for (let k = 0; k <= T; k += (T > 12 ? 4 : 2)) ctx.fillText("k = " + k, X(k), h - 6);
  $("#hNV").textContent = r.N; $("#hUV").textContent = fmt(r.umax, 2); $("#hDV").textContent = fmt(r.dumax, 2); $("#hP0V").textContent = fmt(r.p0, 1); $("#hV0V").textContent = fmt(r.v0, 1);
  const nb = r.N === 8 && r.umax === 1 && r.dumax === 0.5 && r.p0 === -3 && r.v0 === 0 && r.terminal;
  if (!sol.feasible) { $("#rCost").textContent = "–"; $("#rAct").textContent = "–"; $("#rMax").textContent = "–"; $("#rTerm").textContent = "–"; caption($("#heroStatus"), `No input sequence satisfies every constraint from [[(${fmt(r.p0, 1)}, ${fmt(r.v0, 1)})]] with [[N = ${r.N}]]${r.terminal ? " and the terminal equality" : ""}: the QP is infeasible, and the solver's failure after ${sol.nit} iterations is all it can say. ` + (r.terminal && r.N <= 4 ? "With the rate bound the input can only reach ±0.5, ±1 in the first two steps, too little to come to rest at the origin this quickly; " : "") + "a longer horizon, a looser input or rate bound, or dropping the terminal equality gives the optimiser something to work with. Chapter A4's feasibility LP would certify that nothing was missed."); return; }
  const maxU = Math.max(...sol.U.map(Math.abs)), dU = sol.U.map((u, k) => u - (k ? sol.U[k - 1] : 0)), maxdU = Math.max(...dU.map(Math.abs)), xN = sol.X[sol.X.length - 1];
  const counts = {}; sol.active.forEach(a => counts[a.kind] = (counts[a.kind] || 0) + 1); const top = sol.active.slice().sort((a, b) => b.lam - a.lam)[0];
  $("#rCost").textContent = fmt(sol.cost, 4); $("#rAct").textContent = String(sol.active.length); $("#rMax").textContent = `${fmt(maxU, 3)} / ${fmt(maxdU, 3)}`; $("#rTerm").textContent = `(${fmt(Math.abs(xN[0]) < 5e-4 ? 0 : xN[0], 3)}, ${fmt(Math.abs(xN[1]) < 5e-4 ? 0 : xN[1], 3)})`;
  let text = `${nb ? "The chapter's problem: " : ""}cost [[J = ${fmt(sol.cost, 4)}]] (the QP's own objective is ${fmt(sol.objective, 4)}; the dropped constant [[c(x_0) = ${fmt(prob.constant, 1)}]] is added back). The largest input is ${fmt(maxU, 3)} and the largest step change ${fmt(maxdU, 3)}, against bounds of ${fmt(r.umax, 2)} and ${fmt(r.dumax, 2)}. `;
  text += sol.active.length ? `${sol.active.length} inequality constraints are active (ringed; ring size follows the multiplier): ${Object.entries(counts).map(([k, v]) => `${v} ${kindName[k]}${v > 1 ? "s" : ""}`).join(", ")}. The most expensive is the ${kindName[top.kind]} at step ${top.step}, with multiplier [[\\lambda = ${fmt(top.lam, 3)}]]: relaxing that one bound by a small [[\\delta]] would lower the cost by about [[${fmt(top.lam, 3)}\\,\\delta]]. ` : "No inequality is active: the bounds do not bind and the problem is, locally, unconstrained. ";
  text += r.terminal ? `The terminal equality lands the state at [[x_N = (${fmt(Math.abs(xN[0]) < 5e-4 ? 0 : xN[0], 3)}, ${fmt(Math.abs(xN[1]) < 5e-4 ? 0 : xN[1], 3)})]] with multipliers [[\\nu = (${fmt(sol.nu[0], 3)}, ${fmt(sol.nu[1], 3)})]].` : `Without the terminal equality the horizon ends at [[x_N = (${fmt(xN[0], 3)}, ${fmt(xN[1], 3)})]], close to the origin but not on it; the terminal weight [[P = Q]] pulls, the equality would pin.`;
  caption($("#heroStatus"), text);
}
["#hN", "#hU", "#hD", "#hP0", "#hV0"].forEach(id => $(id).addEventListener("input", drawHero));
$("#hTerm").addEventListener("click", () => { const b = $("#hTerm"); b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true")); drawHero(); });
$("#hReset").addEventListener("click", () => { $("#hN").value = 8; $("#hU").value = 1; $("#hD").value = 0.5; $("#hP0").value = -3; $("#hV0").value = 0; $("#hTerm").setAttribute("aria-pressed", "true"); drawHero(); });

/* ---------------- the softened comfort bound ---------------- */
const RHOS = [0.25, 0.5, 1, 1.5, 2, 2.5145, 3, 5, 10];
function drawSoft() {
  const rho = RHOS[+$("#sRho").value], hard = solveCFTOC(buildCFTOC({ N: 8, vmax: 1 })), soft = solveCFTOC(buildCFTOC({ N: 8, vmax: 1, soft: rho }));
  const lamV = Math.max(0, ...hard.active.filter(a => a.kind === "velocity").map(a => a.lam));
  const cv = $("#softC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.3, { x: [0, 8], y: [-0.2, 1.3], xlabel: "step k", xticks: [0, 2, 4, 6, 8], yticks: [0, 0.5, 1] }), { ctx, X, Y } = fr;
  ctx.strokeStyle = alpha("--warm", 0.7); ctx.setLineDash([5, 4]); seg(ctx, X(0), Y(1), X(8), Y(1)); ctx.setLineDash([]);
  const vh = [0, ...hard.X.map(x => x[1])], vs = [0, ...soft.X.map(x => x[1])];
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.6; ctx.beginPath(); vh.forEach((v, k) => k ? ctx.lineTo(X(k), Y(v)) : ctx.moveTo(X(k), Y(v))); ctx.stroke();
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 2.2; ctx.beginPath(); vs.forEach((v, k) => k ? ctx.lineTo(X(k), Y(v)) : ctx.moveTo(X(k), Y(v))); ctx.stroke();
  ctx.fillStyle = css("--post"); vs.forEach((v, k) => { ctx.beginPath(); ctx.arc(X(k), Y(v), 2.8, 0, 2 * Math.PI); ctx.fill(); });
  legend(ctx, ["black: velocity with the hard bound |v| ≤ 1 · blue: with the soft bound", "dashed: the comfort limit"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const maxSlack = Math.max(0, ...soft.slack), peak = Math.max(...vs.map(Math.abs)), diff = Math.max(...soft.U.map((u, k) => Math.abs(u - hard.U[k])));
  $("#sRhoV").textContent = fmt(rho, 4);
  caption($("#softCap"), `The chapter's problem with a comfort limit [[|v| \\le 1]] in place of [[|v| \\le 3]]: the hard problem costs ${fmt(hard.cost, 4)}, with the velocity bound active at step 3 and multiplier [[\\lambda = ${fmt(lamV, 4)}]]. Softened with slacks [[\\epsilon_k \\ge 0]] and the penalty [[\\rho \\sum_k \\epsilon_k]], [[\\rho = ${fmt(rho, 4)}]]: ` + (rho < lamV - 1e-9 ? `the optimiser buys a violation, largest slack ${fmt(maxSlack, 4)}, peak [[|v| = ${fmt(peak, 4)}]], because exceeding the limit costs less than obeying it; the inputs differ from the hard ones by ${diff.toExponential(1)}. ` : `no slack is used (largest ${maxSlack.toExponential(1)}), the peak velocity is ${fmt(peak, 4)} and the inputs agree with the hard solution to ${diff.toExponential(1)}: the penalty is exact. `) + `The threshold is the multiplier itself: for [[\\rho \\ge \\lambda^\\star = ${fmt(lamV, 4)}]] the soft problem reproduces the hard solution whenever the hard problem is feasible, and still returns a plan, the least violating one, when it is not.`);
}
$("#sRho").addEventListener("input", drawSoft);

function drawAll() { drawHero(); drawSoft(); }
drawAll();
onRedraw(drawAll);
})();
