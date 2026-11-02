/* Advanced MPC and Learning MPC, chapter C3, nonlinear MPC by successive linearisation: the demos.
   The unicycle's terminal-pose problem is solved by linearising the dynamics along the current plan and
   solving a QP in the step with the course's interior-point solver (mpc-lib.js), repeated to convergence;
   the hero shows the plan at every iteration and the real-time iteration in closed loop under a push.
   The algorithm is the numbers script's, line for line. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { solveQP, mv, mm, tr, zeros, eye, dot, legend } = window.MPC;

const TS = 0.2, N = 30, ALPHA = 0.05, BETA = 0.10, UMAX = 1, DUMAX = 0.2;
const Z0 = [0, 0, 0], ZF = [2, 1, 0];

/* ---- the unicycle ---- */
const step = (z, u) => [z[0] + TS * u[0] * Math.cos(z[2]), z[1] + TS * u[0] * Math.sin(z[2]), z[2] + TS * u[1]];
const jacobians = (z, u) => ({ A: [[1, 0, -TS * u[0] * Math.sin(z[2])], [0, 1, TS * u[0] * Math.cos(z[2])], [0, 0, 1]], B: [[TS * Math.cos(z[2]), 0], [TS * Math.sin(z[2]), 0], [0, TS]] });
function rollout(z0, U) { const Z = [z0.slice()]; for (const u of U) Z.push(step(Z[Z.length - 1], u)); return Z; }
const cost = U => { let c = 0; for (let k = 0; k < U.length; k++) { c += ALPHA * (U[k][0] ** 2 + U[k][1] ** 2); if (k) c += BETA * ((U[k][0] - U[k - 1][0]) ** 2 + (U[k][1] - U[k - 1][1]) ** 2); } return c; };
/* the last block row of the time-varying prediction matrix: d z_N / d U = [A_{N-1}...A_1 B_0, ..., B_{N-1}] */
function terminalMap(As, Bs) { const n = 3, Nn = As.length, M = zeros(n, 2 * Nn); let P = eye(n); for (let j = Nn - 1; j >= 0; j--) { const PB = mm(P, Bs[j]); for (let r = 0; r < n; r++) { M[r][2 * j] = PB[r][0]; M[r][2 * j + 1] = PB[r][1]; } P = mm(P, As[j]); } return M; }
function costMatrices(Nn) { const nz = 2 * Nn, D = zeros(2 * (Nn - 1), nz); for (let k = 0; k < Nn - 1; k++) { D[2 * k][2 * k] = -1; D[2 * k][2 * k + 2] = 1; D[2 * k + 1][2 * k + 1] = -1; D[2 * k + 1][2 * k + 3] = 1; } const DtD = D.length ? mm(tr(D), D) : zeros(nz, nz), H = DtD.map((r, i) => r.map((v, j) => 2 * (BETA * v + (i === j ? ALPHA : 0)))); return { H, D }; }
/* one successive-linearisation run; opts: rho (elastic terminal), trust, uPrev, maxIter, tol, log */
function sqp(z0, zf, U0, Nn, opts = {}) {
  const { H, D } = costMatrices(Nn), nz = 2 * Nn, rho = opts.rho ?? null, tol = opts.tol ?? 1e-6, maxIter = opts.maxIter ?? 50;
  const G = [...D, ...D.map(r => r.map(v => -v))], g = new Array(G.length).fill(DUMAX);
  if (opts.uPrev) { const e0 = new Array(nz).fill(0); e0[0] = 1; const e1 = new Array(nz).fill(0); e1[1] = 1; G.push(e0, e1, e0.map(v => -v), e1.map(v => -v)); g.push(DUMAX + opts.uPrev[0], DUMAX + opts.uPrev[1], DUMAX - opts.uPrev[0], DUMAX - opts.uPrev[1]); }
  let U = U0.map(u => u.slice()); const hist = [];
  let it = 0;
  for (it = 1; it <= maxIter; it++) {
    const Z = rollout(z0, U), As = [], Bs = []; for (let k = 0; k < Nn; k++) { const J = jacobians(Z[k], U[k]); As.push(J.A); Bs.push(J.B); }
    const M = terminalMap(As, Bs), defect = zf.map((v, i) => v - Z[Nn][i]), Uf = U.flat(), f = mv(H, Uf);
    const lb = Uf.map(v => Math.max(-UMAX - v, opts.trust != null ? -opts.trust : -Infinity)), ub = Uf.map(v => Math.min(UMAX - v, opts.trust != null ? opts.trust : Infinity));
    const gd = g.map((v, i) => v - dot(G[i], Uf));
    let r, dU;
    if (rho == null) { r = solveQP(H, f, G, gd, M, defect, lb, ub); dU = r.z; }
    else {
      const ext = nz + 6, Hs = zeros(ext, ext); for (let i = 0; i < nz; i++) for (let j = 0; j < nz; j++) Hs[i][j] = H[i][j];
      const fs = [...f, ...new Array(6).fill(rho)], Gs = G.map(row => [...row, 0, 0, 0, 0, 0, 0]);
      const E = M.map((row, i) => { const e = [...row, 0, 0, 0, 0, 0, 0]; e[nz + i] = 1; e[nz + 3 + i] = -1; return e; });
      r = solveQP(Hs, fs, Gs, gd, E, defect, [...lb, 0, 0, 0, 0, 0, 0], [...ub, ...new Array(6).fill(Infinity)]); dU = r.z.slice(0, nz);
    }
    if (!r.success) { hist.push({ cost: cost(U), resid: Math.max(...defect.map(Math.abs)), step: NaN, failed: true }); break; }
    U = U.map((u, k) => [u[0] + dU[2 * k], u[1] + dU[2 * k + 1]]);
    const resid = Math.max(...zf.map((v, i) => Math.abs(v - rollout(z0, U)[Nn][i]))), stepSize = Math.max(...dU.map(Math.abs));
    hist.push({ cost: cost(U), resid, step: stepSize, nit: r.nit }); if (opts.log) opts.log.push(U.map(u => u.slice()));
    if (resid < tol && stepSize < tol) break;
  }
  return { U, it: Math.min(it, maxIter), hist };
}
const GUESSES = {
  guess: { name: "the guess of chapter A5 (drive, turn, unwind)", U: Array.from({ length: N }, (_, k) => [k < 10 ? 0.8 : k < 20 ? 0.6 : 0.4, k < 10 ? 0.3 : k < 20 ? -0.3 : 0]) },
  straight: { name: "straight ahead at v = 0.5", U: Array.from({ length: N }, () => [0.5, 0]) },
  zeros: { name: "all zeros (elastic mode: the hard equality is inconsistent here)", U: Array.from({ length: N }, () => [0, 0]) },
};

/* ---- the hero ---- */
const hs = { guess: "guess", mode: "open", iter: -1, runs: {}, closed: null };
function openRun() { const key = hs.guess; if (!hs.runs[key]) { const log = [GUESSES[key].U.map(u => u.slice())]; const r = sqp(Z0, ZF, GUESSES[key].U, N, { log, rho: key === "zeros" ? 20 : null }); hs.runs[key] = { r, log }; } return hs.runs[key]; }
function closedRun() {
  const at = +$("#hPushAt").value, size = +$("#hPushSize").value, push = [0, size, 0.75 * size], key = `${at}|${size}`;
  if (hs.closed && hs.closed.key === key) return hs.closed;
  const ref = openRun().r.U; let z = Z0.slice(), uPrev = null, plan = ref.map(u => u.slice()); const Z = [z.slice()], applied = [];
  for (let t = 0; t < N; t++) { const h = N - t, r = sqp(z, ZF, plan.slice(0, h), h, { maxIter: 1, rho: 20, uPrev }); const u = r.U[0]; applied.push(u); uPrev = u; z = step(z, u); if (t === at) z = [z[0] + push[0], z[1] + push[1], z[2] + push[2]]; Z.push(z.slice()); plan = h > 1 ? [...r.U.slice(1), r.U[h - 1]] : r.U; }
  const rates = applied.slice(1).map((u, k) => Math.max(Math.abs(u[0] - applied[k][0]), Math.abs(u[1] - applied[k][1])));
  return hs.closed = { key, at, push, Z, applied, err: ZF.map((v, i) => Math.abs(v - Z[N][i])), cost: cost(applied), maxRate: Math.max(...rates), maxU: Math.max(...applied.flat().map(Math.abs)) };
}
function arrow(ctx, X, Y, z, len, color) { ctx.strokeStyle = color; ctx.lineWidth = 2; seg(ctx, X(z[0]), Y(z[1]), X(z[0] + len * Math.cos(z[2])), Y(z[1] + len * Math.sin(z[2]))); }
function drawHero() {
  const open = hs.mode === "open", run = openRun(), K = run.log.length - 1;
  if (hs.iter < 0 || hs.iter > K) hs.iter = K;
  $("#hIter").max = K; $("#hIter").value = hs.iter; $("#hIterV").textContent = hs.iter;
  const cv = $("#planeC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.2 : 1.7, { x: [-0.3, 2.7], y: [-0.35, 1.45], xlabel: "X", ylabel: "Y", xticks: [0, 0.5, 1, 1.5, 2, 2.5], yticks: [0, 0.5, 1] }), { ctx, X, Y } = fr;
  arrow(ctx, X, Y, ZF, 0.25, css("--warm")); ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(ZF[0]), Y(ZF[1]), 5, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(0), Y(0), 4, 0, 2 * Math.PI); ctx.fill();
  let text;
  if (open) {
    for (let i = 0; i < hs.iter; i++) { const Z = rollout(Z0, run.log[i]); ctx.strokeStyle = alpha("--post", 0.12 + 0.3 * i / Math.max(1, K)); ctx.lineWidth = 1; ctx.beginPath(); Z.forEach((z, k) => k ? ctx.lineTo(X(z[0]), Y(z[1])) : ctx.moveTo(X(z[0]), Y(z[1]))); ctx.stroke(); }
    const Z = rollout(Z0, run.log[hs.iter]); ctx.strokeStyle = css("--post"); ctx.lineWidth = 2.2; ctx.beginPath(); Z.forEach((z, k) => k ? ctx.lineTo(X(z[0]), Y(z[1])) : ctx.moveTo(X(z[0]), Y(z[1]))); ctx.stroke();
    Z.forEach((z, k) => { if (k % 5 === 0) arrow(ctx, X, Y, z, 0.12, alpha("--post", 0.8)); });
    arrow(ctx, X, Y, Z[N], 0.25, css("--post"));
    legend(ctx, ["blue: the plan at the chosen iteration (faint: earlier ones)", "orange: the target pose (2, 1, 0) · ticks every five samples: the heading"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    const h = hs.iter ? run.r.hist[hs.iter - 1] : null, U = run.log[hs.iter], resid = Math.max(...ZF.map((v, i) => Math.abs(v - Z[N][i])));
    const first = run.r.hist.findIndex(x => x.resid < 1e-6) + 1;
    $("#rCost").textContent = fmt(cost(U), 6); $("#rRes").textContent = resid.toExponential(1); $("#rStep").textContent = h ? h.step.toExponential(1) : "–"; $("#rIters").textContent = `${run.r.it} (${first > 0 ? first : "–"} to 10⁻⁶)`;
    text = `Open loop from ${GUESSES[hs.guess].name}. Iteration ${hs.iter} of ${K}: the plan costs [[${fmt(cost(U), 6)}]] and ends ${resid.toExponential(1)} from the pose${hs.iter === 0 ? ", before any linearisation" : ""}. `;
    if (hs.iter === 0) text += `Each iteration linearises the dynamics along this plan, solves one QP in the step [[\\delta U]] with the terminal pose as a linearised equality, and adds the step. Drag the slider.`;
    else if (hs.iter < K) text += `The step just taken moved the largest input by ${h.step.toExponential(1)}; the residual shrank from ${(hs.iter > 1 ? run.r.hist[hs.iter - 2].resid : Math.max(...ZF.map((v, i) => Math.abs(v - rollout(Z0, run.log[0])[N][i])))).toExponential(1)}, a constant factor of about six per iteration: linear convergence, because the step uses the Jacobians of the dynamics but not the curvature of the terminal map.`;
    else text += `Converged: the residual is at round-off and the last step was ${h.step.toExponential(1)}. The cost ${fmt(cost(U), 4)} is what SciPy's SLSQP finds from the same guess in 27 iterations; no input or rate bound is active at the solution (largest [[|v|]] ${fmt(Math.max(...U.map(u => Math.abs(u[0]))), 3)}, largest rate ${fmt(Math.max(...U.slice(1).map((u, k) => Math.max(Math.abs(u[0] - U[k][0]), Math.abs(u[1] - U[k][1])))), 3)}).`;
  } else {
    const c = closedRun(), ref = rollout(Z0, run.r.U);
    ctx.strokeStyle = alpha("--post", 0.35); ctx.lineWidth = 1.2; ctx.setLineDash([4, 4]); ctx.beginPath(); ref.forEach((z, k) => k ? ctx.lineTo(X(z[0]), Y(z[1])) : ctx.moveTo(X(z[0]), Y(z[1]))); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2.2; ctx.beginPath(); c.Z.forEach((z, k) => k ? ctx.lineTo(X(z[0]), Y(z[1])) : ctx.moveTo(X(z[0]), Y(z[1]))); ctx.stroke();
    c.Z.forEach((z, k) => { if (k % 5 === 0) arrow(ctx, X, Y, z, 0.12, alpha("--ink", 0.7)); });
    const zp = c.Z[c.at + 1]; ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(zp[0]), Y(zp[1]), 5, 0, 2 * Math.PI); ctx.fill();
    arrow(ctx, X, Y, c.Z[N], 0.25, css("--ink"));
    legend(ctx, ["black: the closed loop, one QP per sample · dashed: the open-loop plan", "orange dot: the state after the push"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    $("#rCost").textContent = fmt(c.cost, 4); $("#rRes").textContent = Math.max(...c.err).toExponential(1); $("#rStep").textContent = `${fmt(c.maxU, 3)} / ${fmt(c.maxRate, 3)}`; $("#rIters").textContent = `${N} QPs`;
    text = `Closed loop, the real-time iteration: at every sample one QP, linearised along the shifted plan of the previous sample, with the pose as an elastic constraint ([[\\rho = 20]]) and the first move rate-bounded against the input just applied. At sample ${c.at} the robot is pushed by [[(0, ${fmt(c.push[1], 2)}, ${fmt(c.push[2], 3)})]]. `;
    text += Math.max(...c.err) < 1e-6 ? `With ${N - c.at - 1} sample${N - c.at - 1 === 1 ? "" : "s"} left the loop recovers the pose exactly, at closed-loop cost ${fmt(c.cost, 4)} against ${fmt(cost(run.r.U), 4)} for the undisturbed plan; every input and rate bound holds (largest [[|u|]] ${fmt(c.maxU, 3)}, largest rate ${fmt(c.maxRate, 3)}).` : `With only ${N - c.at - 1} sample${N - c.at - 1 === 1 ? "" : "s"} left the pose cannot be reached within the rate and input bounds: the elastic constraint lets the QP stay feasible and the loop ends ${Math.max(...c.err).toFixed(4)} off, the least the penalty could buy; every bound still holds.`;
  }
  $$("#demo [data-guess]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.guess === hs.guess)));
  $$("#demo [data-mode]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.mode === hs.mode)));
  $("#hPushAtV").textContent = $("#hPushAt").value; $("#hPushSizeV").textContent = fmt(+$("#hPushSize").value, 2);
  $("#lblStep").textContent = open ? "largest |δU| of the last step" : "largest |u| / rate";
  caption($("#heroStatus"), text);
}
$$("#demo [data-guess]").forEach(b => b.addEventListener("click", () => { hs.guess = b.dataset.guess; hs.iter = -1; hs.closed = null; drawHero(); }));
$$("#demo [data-mode]").forEach(b => b.addEventListener("click", () => { hs.mode = b.dataset.mode; drawHero(); }));
$("#hIter").addEventListener("input", () => { hs.iter = +$("#hIter").value; hs.mode = "open"; drawHero(); });
["#hPushAt", "#hPushSize"].forEach(id => $(id).addEventListener("input", () => { hs.mode = "closed"; drawHero(); }));
$("#hReset").addEventListener("click", () => { hs.guess = "guess"; hs.mode = "open"; hs.iter = -1; $("#hPushAt").value = 10; $("#hPushSize").value = 0.15; hs.closed = null; drawHero(); });

/* ---- Part 5: convergence of the terminal residual, hard equality against the elastic mode ---- */
const cs = { cache: {} };
function drawConv() {
  const rho = [0.1, 1, 20][+$("#convRho").value], key = String(rho);
  if (!cs.hard) cs.hard = sqp(Z0, ZF, GUESSES.guess.U, N, { tol: 1e-12, maxIter: 20 }).hist;
  if (!cs.cache[key]) cs.cache[key] = sqp(Z0, ZF, GUESSES.guess.U, N, { tol: 1e-12, maxIter: 20, rho }).hist;
  const el = cs.cache[key], cv = $("#convC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.4 : 2.3, { x: [1, 20], y: [-13, 1], xlabel: "iteration", xticks: [1, 5, 10, 15, 20], yticks: [-12, -9, -6, -3, 0], yfmt: v => "10" + String(v).replace("-", "⁻").replace(/\d/g, ch => "⁰¹²³⁴⁵⁶⁷⁸⁹"[ch]) });
  const lg = h => h.map(x => Math.max(-13, Math.log10(Math.max(x.resid, 1e-300))));
  plotLine(fr, cs.hard.map((_, i) => i + 1), lg(cs.hard), { color: css("--post"), width: 2.2 });
  plotLine(fr, el.map((_, i) => i + 1), lg(el), { color: css("--warm"), width: 1.8, dash: [5, 4] });
  legend(fr.ctx, ["blue: hard linearised equality · dashed orange: elastic mode", "terminal residual against the iteration, from the guess"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#convRhoV").textContent = rho;
  const last = el[el.length - 1];
  caption($("#convCap"), `[[\\rho = ${rho}]]. The hard equality reaches [[10^{-6}]] at iteration ${cs.hard.findIndex(h => h.resid < 1e-6) + 1} and round-off by iteration 15, shrinking by a factor of about six each time. ` + (rho >= 1 ? `With [[\\rho = ${rho}]], above the largest terminal multiplier 0.261, the elastic mode is exact: its slacks are zero at the solution and it follows the same path to the same cost ${fmt(last.cost, 6)}.` : `With [[\\rho = ${rho}]], below the largest terminal multiplier 0.261, the penalty is not exact: the iteration settles on a plan that misses the pose by ${last.resid.toFixed(2)} at the cheaper cost ${fmt(last.cost, 4)}, because missing it costs less than reaching it. An exact penalty needs [[\\rho]] above the multiplier.`));
}
$("#convRho").addEventListener("input", drawConv);

function drawAll() { drawHero(); drawConv(); }
drawAll();
onRedraw(drawAll);
})();
