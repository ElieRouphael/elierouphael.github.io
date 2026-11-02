/* Advanced MPC and Learning MPC, chapter A4, solving a QP you can read: the demos.
   The hero runs the course's interior-point method on the hand QP of chapter A3 one iteration at a
   time, with the centring parameter and the tolerance adjustable, and tabulates the iterates; the
   second widget shows the multipliers the same solver returns for an MPC QP of the canonical system.
   The traced solver below is the algorithm of qp.py and of mpc-lib.js's solveQP, with the iterates
   recorded. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { SYS, mv, tr, dot, zeros, solve, predictionMatrices, solveQP, legend } = window.MPC;

/* the interior-point method with a trace: min 1/2 z'Hz + f'z, G z <= g (no equalities, no bounds: the hero problem) */
function ipmTrace(H, f, G, g, sigma, tol, maxIter = 60) {
  const n = f.length, m = g.length, Gt = tr(G), trace = [];
  let z = new Array(n).fill(0), lam = new Array(m).fill(1), s = g.map((gi, i) => Math.max(gi - dot(G[i], z), 1));
  for (let it = 0; it < maxIter; it++) {
    const rd = mv(H, z).map((v, i) => v + f[i] + dot(Gt[i], lam)), rp = G.map((row, i) => dot(row, z) + s[i] - g[i]);
    const mu = dot(s, lam) / m, rc = s.map((si, i) => sigma * mu - si * lam[i]);
    const resid = Math.max(...rd.map(Math.abs), ...rp.map(Math.abs));
    trace.push({ z: z.slice(), s: s.slice(), lam: lam.slice(), resid, mu });
    if (resid < tol && mu < tol) break;
    const D = lam.map((l, i) => l / s[i]), M = H.map(r => r.slice());
    for (let k = 0; k < m; k++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] += G[k][i] * D[k] * G[k][j];
    const t = rc.map((v, i) => (v + lam[i] * rp[i]) / s[i]), rhs = rd.map((v, i) => -v - dot(Gt[i], t));
    const dz = solve(M, rhs), ds = rp.map((v, i) => -v - dot(G[i], dz)), dl = rc.map((v, i) => (v - lam[i] * ds[i]) / s[i]);
    let a = 1; for (let i = 0; i < m; i++) { if (ds[i] < 0) a = Math.min(a, -0.99 * s[i] / ds[i]); if (dl[i] < 0) a = Math.min(a, -0.99 * lam[i] / dl[i]); }
    trace[trace.length - 1].step = a;
    z = z.map((v, i) => v + a * dz[i]); s = s.map((v, i) => v + a * ds[i]); lam = lam.map((v, i) => v + a * dl[i]);
  }
  return trace;
}
const C = [1.2, 0.8], HQ = [[2, 0], [0, 2]], FQ = [-2.4, -1.6], GQ = [[1, 1]], GG = [1];
const SIGMAS = [0.01, 0.05, 0.1, 0.3, 0.5, 0.9], TOLS = [1e-4, 1e-6, 1e-8, 1e-10];
const hs = { shown: -1 };

function drawHero() {
  const sigma = SIGMAS[+$("#hSigma").value], tol = TOLS[+$("#hTol").value], trace = ipmTrace(HQ, FQ, GQ, GG, sigma, tol);
  const last = trace[trace.length - 1], shown = hs.shown < 0 ? trace.length - 1 : Math.min(hs.shown, trace.length - 1);
  const cv = $("#pathC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.2 : 1.6, { x: [-0.3, 1.6], y: [-0.3, 1.3], xlabel: "z₁", ylabel: "z₂", xticks: [0, 0.5, 1, 1.5], yticks: [0, 0.5, 1] }), { ctx, X, Y } = fr;
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  ctx.beginPath(); ctx.moveTo(X(-1), Y(2)); ctx.lineTo(X(2), Y(-1)); ctx.lineTo(X(-1), Y(-1)); ctx.closePath(); ctx.fillStyle = alpha("--post", 0.14); ctx.fill();
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 1.5; seg(ctx, X(-1), Y(2), X(2), Y(-1));
  for (const rad of [0.25, 0.5, 0.75, 1]) { ctx.strokeStyle = alpha("--ink", 0.18); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X(C[0]), Y(C[1]), rad * fr.iw / (fr.x1 - fr.x0), 0, 2 * Math.PI); ctx.stroke(); }
  ctx.restore();
  /* the path of iterates */
  ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.6; ctx.beginPath(); trace.slice(0, shown + 1).forEach((t, i) => i ? ctx.lineTo(X(t.z[0]), Y(t.z[1])) : ctx.moveTo(X(t.z[0]), Y(t.z[1]))); ctx.stroke();
  trace.slice(0, shown + 1).forEach((t, i) => { ctx.fillStyle = i === shown ? css("--warm") : alpha("--warm", 0.6); ctx.beginPath(); ctx.arc(X(t.z[0]), Y(t.z[1]), i === shown ? 5.5 : 3, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(C[0]), Y(C[1]), 4, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText("c", X(C[0]) + 7, Y(C[1]) - 6); ctx.fillText("z⁰ = 0", X(0) + 7, Y(0) + 12);
  legend(ctx, ["shaded: z₁ + z₂ ≤ 1 · circles: level sets · orange: the iterates z¹, z², …"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* the table */
  $("#iterT").innerHTML = "<thead><tr><th>k</th><th>z₁</th><th>z₂</th><th>s</th><th>λ</th><th>residual</th><th>μ = sλ</th><th>step α</th></tr></thead><tbody>" +
    trace.map((t, k) => `<tr${k === shown ? ' style="background:var(--accent-wash)"' : ""}><td>${k}</td><td>${fmt(t.z[0], 5)}</td><td>${fmt(t.z[1], 5)}</td><td>${t.s[0].toExponential(2)}</td><td>${fmt(t.lam[0], 5)}</td><td>${t.resid.toExponential(1)}</td><td>${t.mu.toExponential(1)}</td><td>${t.step === undefined ? "–" : fmt(t.step, 3)}</td></tr>`).join("") + "</tbody>";
  $("#hSigmaV").textContent = sigma; $("#hTolV").textContent = tol.toExponential(0); $("#hStep").max = trace.length - 1; $("#hStep").value = shown; $("#hStepV").textContent = shown;
  const t = trace[shown];
  caption($("#heroStatus"), `[[\\sigma = ${sigma}]], tolerance ${tol.toExponential(0)}: ${trace.length - 1} Newton steps${trace.length >= 60 ? " (the cap)" : ""} from [[z^0 = 0]], [[s^0 = 1]], [[\\lambda^0 = 1]]. Iterate ${shown}: [[z = (${fmt(t.z[0], 5)}, ${fmt(t.z[1], 5)})]], slack [[s = ${t.s[0].toExponential(2)}]], multiplier [[\\lambda = ${fmt(t.lam[0], 5)}]]. ` + (shown === 0 ? `Before any step the stationarity residual is ${t.resid.toExponential(1)} and the complementarity [[s\\lambda = ${t.mu.toExponential(1)}]] is far from zero: the start is not on the constraint and not optimal. ` : `After the first full Newton step the linear residuals (stationarity, primal) are at round-off, ${t.resid.toExponential(1)}, because the KKT equations of a QP are linear in [[(z, \\lambda)]] once [[s\\lambda]] is fixed; what remains is the product [[s\\lambda = ${t.mu.toExponential(1)}]], driven towards zero by a factor of about [[\\sigma]] per step. `) + `The iterates approach [[z^\\star = (0.7, 0.3)]] from inside the half-plane, never touching the line [[z_1 + z_2 = 1]] until the end, which is what "interior point" means; the fraction-to-the-boundary rule keeps [[s]] and [[\\lambda]] positive, so a step of [[\\alpha = ${t.step === undefined ? "–" : fmt(t.step, 3)}]] is taken. ` + (sigma >= 0.5 ? "A large [[\\sigma]] aims at the central path rather than at the solution and needs many more steps. " : sigma <= 0.01 ? "A tiny [[\\sigma]] aims straight at the solution; on this problem it works, on harder ones it loses the centrality that keeps the Newton system well conditioned. " : "") + `The final iterate is then polished: the active set [[\\{i : \\lambda_i > s_i\\}]] is [[\\{1\\}]], and the equality-constrained QP on it returns [[(0.7, 0.3)]] and [[\\lambda = 1]] exactly.`);
}
["#hSigma", "#hTol"].forEach(id => $(id).addEventListener("input", () => { hs.shown = -1; drawHero(); }));
$("#hStep").addEventListener("input", () => { hs.shown = +$("#hStep").value; drawHero(); });
$("#hReset").addEventListener("click", () => { $("#hSigma").value = 2; $("#hTol").value = 3; hs.shown = -1; drawHero(); });

/* the multipliers of an MPC QP of the canonical system */
function mpcQP(N, x0, umax) {
  const n = 2, { Sx, Su } = predictionMatrices(SYS.A, SYS.B, N), Qbar = zeros(N * n, N * n);
  for (let i = 0; i < N; i++) for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Qbar[i * n + r][i * n + c] = SYS.Q[r][c];
  const SuTQ = window.MPC.mm(tr(Su), Qbar), H = window.MPC.mm(SuTQ, Su).map((r, i) => r.map((v, j) => 2 * (v + (i === j ? SYS.R : 0)))), f = mv(window.MPC.mm(SuTQ, Sx), x0).map(v => 2 * v);
  const Sxx = mv(Sx, x0), G = [], g = [], tags = [];
  for (let r = 0; r < N * n; r++) { const k = r % n, lim = SYS.xub[k]; G.push(Su[r]); g.push(lim - Sxx[r]); tags.push(`${k ? "v" : "p"}${Math.floor(r / n) + 1} ≤ ${lim}`); G.push(Su[r].map(v => -v)); g.push(lim + Sxx[r]); tags.push(`${k ? "v" : "p"}${Math.floor(r / n) + 1} ≥ −${lim}`); }
  return { H, f, G, g, tags, lb: new Array(N).fill(-umax), ub: new Array(N).fill(umax), Su, Sxx };
}
function drawMult() {
  const N = +$("#mN").value, umax = [0.5, 0.75, 1][+$("#mU").value], p0 = +$("#mP0").value, x0 = [p0, 0], qp = mpcQP(N, x0, umax), r = solveQP(qp.H, qp.f, qp.G, qp.g, [], [], qp.lb, qp.ub);
  const lam = [...r.lam, ...r.lamBounds], labels = [...qp.tags, ...Array.from({ length: N }, (_, k) => `u${k} ≤ ${umax}`), ...Array.from({ length: N }, (_, k) => `u${k} ≥ −${umax}`)];
  const cv = $("#multC"), narrow = narrowOf(cv), m = lam.length, ymax = Math.max(1e-3, ...lam) * 1.15;
  const fr = frame(cv, narrow ? 1.4 : 2.3, { x: [0, m], y: [0, ymax], xlabel: "constraint row (state rows, then input bounds)", xticks: false }), { ctx, X, Y } = fr;
  const w = (X(1) - X(0)) * 0.8;
  lam.forEach((l, i) => { ctx.fillStyle = i < qp.g.length ? css("--post") : css("--c4"); ctx.fillRect(X(i) + (X(1) - X(0) - w) / 2, Y(l), w, Y(0) - Y(l)); });
  ctx.strokeStyle = alpha("--ink", 0.3); seg(ctx, X(qp.g.length), fr.pad.t, X(qp.g.length), Y(0));
  legend(ctx, ["blue: multipliers of the 4N state rows · green: of the 2N input bounds", "zero bars: inactive constraints"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const act = lam.map((l, i) => [l, i]).filter(([l]) => l > 1e-9).sort((a, b) => b[0] - a[0]);
  $("#mNV").textContent = N; $("#mUV").textContent = umax; $("#mP0V").textContent = fmt(p0, 1);
  caption($("#multCap"), r.success ? `[[N = ${N}]], [[|u| \\le ${umax}]], from [[(${fmt(p0, 1)}, 0)]]: the solver converges in ${r.nit} iterations with KKT residual ${r.resid.toExponential(1)}, and returns one multiplier per row, ${m} in all. ${act.length} of them are positive: ${act.slice(0, 4).map(([l, i]) => `${labels[i]} ([[\\lambda = ${fmt(l, 3)}]])`).join(", ")}${act.length > 4 ? ", …" : ""}. Those are the active constraints, found by the solver, not guessed: the polish solved the equality-constrained problem on exactly these rows. Every zero bar is a constraint that does not bind, whose relaxation would change nothing. ` + (act.length === 0 ? "Nothing binds: the unconstrained minimiser of chapter A2 is admissible from this start. " : "") + "The largest multiplier is the bound the controller would most like to loosen." : `From [[(${fmt(p0, 1)}, 0)]] with [[N = ${N}]] the problem is infeasible: the state cannot be kept in the box. The solver reports failure (residual ${r.resid.toExponential(1)} after ${r.nit} iterations); chapter A4's feasibility LP would certify it.`);
}
["#mN", "#mU", "#mP0"].forEach(id => $(id).addEventListener("input", drawMult));

function drawAll() { drawHero(); drawMult(); }
drawAll();
onRedraw(drawAll);
})();
