/* Advanced MPC and Learning MPC, chapter 05, Learning MPC on the double integrator: the live demos.
   The course's lmpc.py is reproduced here: the finite-horizon problem with an l1 state term in epigraph
   form (the seed MPC and the long-horizon reference), the safe-set database with continuation inputs and
   realised costs-to-go, the barycentric terminal value (a linear program), the LMPC quadratic program
   with x_N = S'lambda, and the repeated-iteration loop with the exact two-step bridge to the target.
   Every QP is solved by a primal-dual interior-point method with bound constraints handled on the
   diagonal, so a full nine-iteration experiment runs in a few seconds. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp, fit } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const eye = n => zeros(n, n).map((row, i) => row.map((_, j) => i === j ? 1 : 0));
const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const mv = (A, x) => A.map(r => r.reduce((s, v, k) => s + v * x[k], 0));
const tr = A => A[0].map((_, j) => A.map(r => r[j]));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
function solve(A0, b0) {
  const n = A0.length, A = A0.map((r, i) => [...r, b0[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; if (Math.abs(A[c][c]) < 1e-14) A[c][c] = 1e-14;
    const pivot = A[c];
    for (let r = c + 1; r < n; r++) { const m = A[r][c] / pivot[c]; if (!m) continue; const row = A[r]; for (let k = c; k <= n; k++) row[k] -= m * pivot[k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = A[r][n]; for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k]; x[r] = s / A[r][r]; }
  return x;
}
/* primal-dual interior point for  min 1/2 z'Hz + f'z,  G z <= g,  E z = e,  lb <= z <= ub  (bounds on the diagonal) */
function solveQP(H, f, G, g, E, e, lb, ub, iters = 300) {
  const n = f.length, m = g.length, p = e.length;
  const Ui = [], Li = []; for (let i = 0; i < n; i++) { if (ub[i] < Infinity) Ui.push(i); if (lb[i] > -Infinity) Li.push(i); }
  let z = new Array(n).fill(0); for (let i = 0; i < n; i++) { if (lb[i] > -Infinity && ub[i] < Infinity) z[i] = 0.5 * (lb[i] + ub[i]); else if (lb[i] > -Infinity) z[i] = lb[i] + 1; else if (ub[i] < Infinity) z[i] = ub[i] - 1; }
  let sg = g.map((gi, i) => Math.max(gi - dot(G[i], z), 1)), lg = new Array(m).fill(1), su = Ui.map(i => Math.max(ub[i] - z[i], 1)), lu = Ui.map(() => 1), sl = Li.map(i => Math.max(z[i] - lb[i], 1)), ll = Li.map(() => 1), nu = new Array(p).fill(0);
  const Gt = m ? tr(G) : null, Et = p ? tr(E) : null, total = m + Ui.length + Li.length;
  let result = null;
  for (let it = 0; it < iters; it++) {
    const rd = mv(H, z).map((v, i) => v + f[i] + (m ? dot(Gt[i], lg) : 0) + (p ? dot(Et[i], nu) : 0)); Ui.forEach((i, k) => rd[i] += lu[k]); Li.forEach((i, k) => rd[i] -= ll[k]);
    const rpg = G.map((row, i) => dot(row, z) + sg[i] - g[i]), rpu = Ui.map((i, k) => z[i] + su[k] - ub[i]), rpl = Li.map((i, k) => -z[i] + sl[k] + lb[i]), re = E.map((row, i) => dot(row, z) - e[i]);
    const mu = total ? (dot(sg, lg) + dot(su, lu) + dot(sl, ll)) / total : 0, sigma = 0.1;
    const rcg = sg.map((s, i) => sigma * mu - s * lg[i]), rcu = su.map((s, k) => sigma * mu - s * lu[k]), rcl = sl.map((s, k) => sigma * mu - s * ll[k]);
    const err = Math.max(...rd.map(Math.abs), ...rpg.map(Math.abs), ...rpu.map(Math.abs), ...rpl.map(Math.abs), ...re.map(Math.abs), 0);
    if (err < 1e-11 && mu < 1e-13) break;
    const M = zeros(n + p, n + p), rhs = new Array(n + p).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] = H[i][j];
    const tg = rcg.map((v, i) => (v + lg[i] * rpg[i]) / sg[i]), tu = rcu.map((v, k) => (v + lu[k] * rpu[k]) / su[k]), tl = rcl.map((v, k) => (v + ll[k] * rpl[k]) / sl[k]);
    for (let k = 0; k < m; k++) { const D = lg[k] / sg[k], row = G[k]; for (let i = 0; i < n; i++) { const a = row[i]; if (!a) continue; const Da = D * a; for (let j = 0; j < n; j++) if (row[j]) M[i][j] += Da * row[j]; } }
    Ui.forEach((i, k) => M[i][i] += lu[k] / su[k]); Li.forEach((i, k) => M[i][i] += ll[k] / sl[k]);
    for (let k = 0; k < p; k++) for (let i = 0; i < n; i++) { M[i][n + k] = E[k][i]; M[n + k][i] = E[k][i]; }
    for (let i = 0; i < n; i++) rhs[i] = -rd[i] - (m ? dot(Gt[i], tg) : 0); Ui.forEach((i, k) => rhs[i] -= tu[k]); Li.forEach((i, k) => rhs[i] += tl[k]);
    for (let k = 0; k < p; k++) rhs[n + k] = -re[k];
    const sol = solve(M, rhs), dz = sol.slice(0, n), dnu = sol.slice(n);
    const dsg = rpg.map((v, i) => -v - dot(G[i], dz)), dlg = rcg.map((v, i) => (v - lg[i] * dsg[i]) / sg[i]);
    const dsu = rpu.map((v, k) => -v - dz[Ui[k]]), dlu = rcu.map((v, k) => (v - lu[k] * dsu[k]) / su[k]);
    const dsl = rpl.map((v, k) => -v + dz[Li[k]]), dll = rcl.map((v, k) => (v - ll[k] * dsl[k]) / sl[k]);
    let a = 1; const lim = (s, ds) => { for (let i = 0; i < s.length; i++) if (ds[i] < 0) a = Math.min(a, -0.99 * s[i] / ds[i]); };
    lim(sg, dsg); lim(lg, dlg); lim(su, dsu); lim(lu, dlu); lim(sl, dsl); lim(ll, dll);
    z = z.map((v, i) => v + a * dz[i]); nu = nu.map((v, i) => v + a * dnu[i]);
    sg = sg.map((v, i) => v + a * dsg[i]); lg = lg.map((v, i) => v + a * dlg[i]); su = su.map((v, k) => v + a * dsu[k]); lu = lu.map((v, k) => v + a * dlu[k]); sl = sl.map((v, k) => v + a * dsl[k]); ll = ll.map((v, k) => v + a * dll[k]);
  }
  const viol = Math.max(...g.map((gi, i) => dot(G[i], z) - gi), ...e.map((ei, i) => Math.abs(dot(E[i], z) - ei)), ...Ui.map(i => z[i] - ub[i]), ...Li.map(i => lb[i] - z[i]), 0);
  return { z, success: viol < 1e-6, maxViolation: viol };
}
function predictionMatrices(A, B, N) { const n = 2, Sx = zeros(N * n, n), Su = zeros(N * n, N); let Ap = eye(n); const powers = [Ap]; for (let i = 1; i <= N; i++) { Ap = mm(A, Ap); powers.push(Ap); } for (let i = 0; i < N; i++) { for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Sx[i * n + r][c] = powers[i + 1][r][c]; for (let j = 0; j <= i; j++) { const blk = mm(powers[i - j], B); for (let r = 0; r < n; r++) Su[i * n + r][j] = blk[r][0]; } } return { Sx, Su }; }
/* the task */
const A = [[1, 1], [0, 1]], B = [[0], [1]], XS = [-15, 0], XLB = [-15, -15], XUB = [15, 15], ULB = -5, UUB = 5, Q = [[1, 0], [0, 1]], Q1 = [[0.01, 0], [0, 0.01]], R = 1;
const stageCost = (x, u) => x[0] * x[0] + x[1] * x[1] + R * u * u + 0.01 * (Math.abs(x[0]) + Math.abs(x[1]));
function costToGo(X, U) { const J = new Array(X.length).fill(0); for (let k = U.length - 1; k >= 0; k--) J[k] = stageCost(X[k], U[k]) + J[k + 1]; return J; }
/* common QP pieces for a horizon N from x0: cost on x_1..x_{N-1} (x_N unweighted, x_0 constant), l1 epigraph on x_1..x_{N-1} */
function buildCore(x0, N, extra) {   // extra = number of extra variables placed between U and the epigraph block (the lambdas)
  const n = 2, { Sx, Su } = predictionMatrices(A, B, N), nu = N, ne = (N - 1) * 2, nz = nu + extra + ne, Sxx = mv(Sx, x0);
  const Qbar = zeros(N * n, N * n); for (let k = 0; k < N - 1; k++) { Qbar[2 * k][2 * k] = 1; Qbar[2 * k + 1][2 * k + 1] = 1; }
  const H = zeros(nz, nz), f = new Array(nz).fill(0), SuTQ = mm(tr(Su), Qbar), Huu = mm(SuTQ, Su);
  for (let i = 0; i < nu; i++) for (let j = 0; j < nu; j++) H[i][j] = 2 * (Huu[i][j] + (i === j ? R : 0));
  const fu = mv(SuTQ, Sxx).map(v => 2 * v); for (let i = 0; i < nu; i++) f[i] = fu[i]; for (let i = 0; i < ne; i++) f[nu + extra + i] = 1;
  const G = [], g = [];
  for (let r = 0; r < N * n; r++) { const k = r % n, row = new Array(nz).fill(0); for (let j = 0; j < nu; j++) row[j] = Su[r][j]; G.push(row); g.push(XUB[k] - Sxx[r]); G.push(row.map(v => -v)); g.push(-XLB[k] + Sxx[r]); }
  for (let k = 1; k < N; k++) for (let i = 0; i < 2; i++) { const r = (k - 1) * n + i, eidx = nu + extra + (k - 1) * 2 + i; const row = new Array(nz).fill(0); for (let j = 0; j < nu; j++) row[j] = 0.01 * Su[r][j]; row[eidx] = -1; G.push(row); g.push(-0.01 * Sxx[r]); const row2 = row.map(v => -v); row2[eidx] = -1; G.push(row2); g.push(0.01 * Sxx[r]); }
  const lb = new Array(nz).fill(-Infinity), ub = new Array(nz).fill(Infinity); for (let j = 0; j < nu; j++) { lb[j] = ULB; ub[j] = UUB; } for (let i = 0; i < ne; i++) lb[nu + extra + i] = 0;
  return { Sx, Su, Sxx, nu, ne, nz, H, f, G, g, lb, ub };
}
/* the seed MPC and the long-horizon reference: no terminal ingredients */
function solveFTOCP(x0, N) {
  const c = buildCore(x0, N, 0), r = solveQP(c.H, c.f, c.G, c.g, [], [], c.lb, c.ub), U = r.z.slice(0, N), Xp = c.Sxx.map((v, i) => v + dot(c.Su[i], U)), X = [x0]; for (let k = 0; k < N; k++) X.push([Xp[2 * k], Xp[2 * k + 1]]);
  let obj = 0; for (let k = 0; k < N; k++) obj += stageCost(X[k], U[k]);
  return { success: r.success, U, X, objective: obj };
}
/* the safe-set database */
function makeStore() { return { trajectories: [], inputs: [], qfun: [], costs: [] }; }
function addTrajectory(store, X, U) { const J = costToGo(X, U); store.trajectories.push(X); store.inputs.push(U); store.qfun.push(J); store.costs.push(J[0]); }
function safeData(store, window) {
  const n = store.trajectories.length, start = window == null ? 0 : Math.max(0, n - Math.max(window, 1)), S = [[0, 0]], V = [0], Us = [0], next = [0];
  for (let it = start; it < n; it++) { const X = store.trajectories[it], U = store.inputs[it], J = store.qfun[it], T = U.length, base = S.length; for (let k = 0; k < T; k++) { S.push(X[k]); V.push(J[k]); Us.push(U[k]); next.push(k < T - 1 ? base + k + 1 : 0); } }
  return { S, V, Us, next };
}
/* the LMPC problem: variables [U (N), lambda (M), e (2(N-1))] */
function lmpcSolve(store, x0, N, window) {
  const { S, V, Us, next } = safeData(store, window), M = S.length, c = buildCore(x0, N, M), nu = c.nu;
  for (let r = 0; r < M; r++) { c.f[nu + r] = V[r]; c.lb[nu + r] = 0; c.ub[nu + r] = 1; }
  const E = [], e = []; for (let i = 0; i < 2; i++) { const row = new Array(c.nz).fill(0), r = (N - 1) * 2 + i; for (let j = 0; j < nu; j++) row[j] = c.Su[r][j]; for (let k = 0; k < M; k++) row[nu + k] = -S[k][i]; E.push(row); e.push(-c.Sxx[r]); }
  const one = new Array(c.nz).fill(0); for (let k = 0; k < M; k++) one[nu + k] = 1; E.push(one); e.push(1);
  const r = solveQP(c.H, c.f, c.G, c.g, E, e, c.lb, c.ub), U = r.z.slice(0, N), lam = r.z.slice(nu, nu + M), Xp = c.Sxx.map((v, i) => v + dot(c.Su[i], U)), X = []; for (let k = 0; k < N; k++) X.push([Xp[2 * k], Xp[2 * k + 1]]);
  let obj = 0; const Xf = [x0, ...X]; for (let k = 0; k < N; k++) obj += stageCost(Xf[k], U[k]); obj += dot(lam, V);
  return { success: r.success, U, X, lam, objective: obj, S, V, Us, next, terminal: [dot(lam, S.map(s => s[0])), dot(lam, S.map(s => s[1]))] };
}
/* exact two-step bridge to the target, as the notebook's close_double_integrator_exactly */
function closeExactly(X, U) { const [p, v] = X[X.length - 1]; if (Math.hypot(p, v) > 1e-5) throw new Error("bridge requested too far from the target"); let x = [p, v]; for (const u of [-p - 2 * v, p + v]) { x = [x[0] + x[1], x[1] + u]; U.push(u); X.push(x); } X[X.length - 1] = [0, 0]; return [X, U]; }
function runSeed(N, maxSteps = 100) { let x = XS.slice(); const X = [x], U = []; for (let t = 0; t < maxSteps; t++) { if (Math.hypot(...x) <= 1e-6) break; const s = solveFTOCP(x, N); if (!s.success) return { X, U, failed: "infeasible" }; const u = s.U[0]; x = [x[0] + x[1], x[1] + u]; U.push(u); X.push(x); } if (Math.hypot(...X[X.length - 1]) > 1e-5) return { X, U, failed: "did not reach the target" }; closeExactly(X, U); return { X, U }; }
function runIteration(store, N, window, maxSteps = 100) { let x = XS.slice(); const X = [x], U = []; let first = null; for (let t = 0; t < maxSteps; t++) { if (Math.hypot(...x) <= 1e-6) break; const s = lmpcSolve(store, x, N, window); if (!s.success) return { X, U, failed: "LMPC QP infeasible at step " + t }; if (!first) first = s; const u = s.U[0]; x = [x[0] + x[1], x[1] + u]; U.push(u); X.push(x); } if (Math.hypot(...X[X.length - 1]) > 1e-5) return { X, U, failed: "did not reach the target" }; closeExactly(X, U); addTrajectory(store, X, U); return { X, U, first, cost: store.costs[store.costs.length - 1] }; }
/* barycentric terminal value as a linear program */
function convexSafeValue(x, S, V) { const M = S.length, H = zeros(M, M), E = [...x.map((_, d) => S.map(s => s[d])), new Array(M).fill(1)], e = [...x, 1]; const r = solveQP(H, V.slice(), [], [], E, e, new Array(M).fill(0), new Array(M).fill(1)); return { value: dot(r.z, V), lam: r.z, success: r.success }; }
/* NET-END */

function legend(ctx, lines, x, y, narrow) { ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; const out = narrow ? lines.flatMap(l => l.split(" · ")) : lines; out.forEach((l, i) => ctx.fillText(l, x, y + 13 * i)); }
const REF = solveFTOCP(XS, 20);

/* =====================================================================
   Hero: the learning experiment, run live
   ===================================================================== */
const hs = { N: 3, window: 4, iters: 9, seedN: 3, result: null, running: false, cache: {} };
function heroCompute(done) {
  const key = `${hs.N}|${hs.window}|${hs.iters}|${hs.seedN}`; if (hs.cache[key]) { hs.result = hs.cache[key]; done(); return; }
  const seed = runSeed(hs.seedN, 400), res = { seedN: hs.seedN, N: hs.N, window: hs.window, seed, iterations: [], costs: [], store: makeStore(), first: null, failed: seed.failed || null };
  hs.result = res; if (seed.failed) { hs.cache[key] = res; done(); return; }
  addTrajectory(res.store, seed.X, seed.U); res.costs.push(res.store.costs[0]);
  let j = 0; hs.running = true;
  const step = () => { if (j >= hs.iters) { hs.running = false; hs.cache[key] = res; done(); return; } const t0 = performance.now(), it = runIteration(res.store, hs.N, hs.window === 0 ? null : hs.window); it.ms = performance.now() - t0; if (it.failed) { res.failed = `iteration ${j + 1}: ${it.failed}`; hs.running = false; hs.cache[key] = res; done(); return; } if (!res.first) res.first = it.first; res.iterations.push(it); res.costs.push(it.cost); j++; done(true); setTimeout(step, 10); };
  done(true); setTimeout(step, 10);
}
function drawHero(partial) {
  const r = hs.result, cv = $("#phaseC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.15 : 1.5, { x: [-16, 1.5], y: [-0.5, 6.5], xlabel: "position x₁", ylabel: "velocity x₂", xticks: [-15, -10, -5, 0], yticks: [0, 2, 4, 6] }), { ctx, X, Y } = fr;
  if (!r) return;
  plotLine(fr, REF.X.map(x => x[0]), REF.X.map(x => x[1]), { color: alpha("--ink", 0.5), width: 2.2, dash: [6, 4] });
  if (r.seed.X) { plotLine(fr, r.seed.X.map(x => x[0]), r.seed.X.map(x => x[1]), { color: css("--warm"), width: 1.8 }); ctx.fillStyle = css("--warm"); r.seed.X.forEach(x => { ctx.beginPath(); ctx.arc(X(x[0]), Y(x[1]), 2.3, 0, 2 * Math.PI); ctx.fill(); }); }
  r.iterations.forEach((it, i) => { const t = r.iterations.length > 1 ? i / (r.iterations.length - 1) : 1; plotLine(fr, it.X.map(x => x[0]), it.X.map(x => x[1]), { color: alpha("--post", 0.25 + 0.75 * t), width: i === r.iterations.length - 1 ? 2.4 : 1.2 }); });
  legend(ctx, ["orange: seed MPC · blue: LMPC iterations (darker = later) · dashed: long-horizon reference"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* learning curve */
  const cv2 = $("#curveC"), costs = r.costs, nIt = Math.max(hs.iters, 1), lo = Math.min(REF.objective, ...costs) - 0.5, hi = Math.max(...costs, REF.objective) + 2;
  const fr2 = frame(cv2, narrow ? 1.5 : 2.4, { x: [-0.5, nIt + 0.5], y: [Math.log10(1e-5), Math.log10(Math.max(30, ...costs.map(c => c - REF.objective)) * 2)], xlabel: "iteration (0 = seed)", xticks: Array.from({ length: nIt + 1 }, (_, i) => i), yticks: [-4, -2, 0, 2], yfmt: v => "10" + String(v).replace("-", "⁻").replace(/\d/g, ch => "⁰¹²³⁴⁵⁶⁷⁸⁹"[ch]) }), ctx2 = fr2.ctx;
  const gaps = costs.map(c => Math.log10(Math.max(c - REF.objective, 1e-5)));
  plotLine(fr2, gaps.map((_, i) => i), gaps, { color: css("--post"), width: 2 }); ctx2.fillStyle = css("--post"); gaps.forEach((v, i) => { ctx2.beginPath(); ctx2.arc(fr2.X(i), fr2.Y(v), 3, 0, 2 * Math.PI); ctx2.fill(); });
  legend(ctx2, ["gap to the long-horizon reference, J^j − J_ref, log scale"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  $("#hNV").textContent = hs.N; $("#hWV").textContent = hs.window === 0 ? "all" : String(hs.window); $("#hIV").textContent = hs.iters; $("#hSV").textContent = hs.seedN;
  const last = costs[costs.length - 1];
  $("#rSeed").textContent = r.seed.failed ? "failed" : `${r.seed.U.length} steps, ${fmt(costs[0], 3)}`; $("#rLast").textContent = r.iterations.length ? `${r.iterations[r.iterations.length - 1].U.length} steps, ${fmt(last, 6)}` : "–"; $("#rRef").textContent = fmt(REF.objective, 6); $("#rGap").textContent = r.iterations.length ? (last - REF.objective).toExponential(2) : "–";
  const nb = hs.N === 3 && hs.window === 4 && hs.iters === 9 && hs.seedN === 3;
  let text;
  if (r.seed.failed) text = `The seed MPC with [[N = ${hs.seedN}]] ${r.seed.failed === "infeasible" ? "hit an infeasible problem" : "never reached the target in 400 steps"}: ${hs.seedN <= 2 ? "with two steps and no terminal cost the objective contains only [[x_1]], whose position cannot be changed by [[u_0]], so the cheapest move is [[u_0 = 0]] and the cart stays at [[(-15, 0)]] forever, the horizon-one pathology of chapter 03 in another guise. LMPC needs one successful trajectory to start, and this controller never provides one." : "LMPC has nothing to learn from."}`;
  else {
    text = `${nb ? "The notebook's experiment: " : ""}seed MPC with [[N = ${hs.seedN}]] and no terminal ingredients reaches the target in ${r.seed.U.length} steps (the last two are the exact bridge) at a complete task cost of ${fmt(costs[0], 3)}, against ${fmt(REF.objective, 3)} for the long-horizon reference: a gap of ${fmt(costs[0] - REF.objective, 3)}. `;
    if (hs.running) text += `Running LMPC iteration ${r.iterations.length + 1} of ${hs.iters}… `;
    if (r.iterations.length) { const diffs = costs.slice(1).map((c, i) => c - costs[i]), inc = diffs.filter(d => d > 1e-6).length; text += `LMPC with [[N = ${hs.N}]] and ${hs.window === 0 ? "every stored trajectory" : "the last " + hs.window + " stored trajectories"} in the safe set: after ${r.iterations.length} iteration${r.iterations.length > 1 ? "s" : ""} the cost is ${fmt(last, 6)}, ${(last - REF.objective).toExponential(1)} above the reference, in ${r.iterations[r.iterations.length - 1].U.length} steps. The iteration cost ${inc === 0 ? "never increased" : `increased on ${inc} of ${diffs.length} steps, by at most ${Math.max(...diffs).toExponential(1)}, which is the solver's tolerance on a cost of 674, not a counter-example to the theory`}; the largest single improvement was the first iteration, ${fmt(-diffs[0], 3)}, because the stored seed already contains the whole safe path and its costs-to-go, which the three-step horizon could not see. The first LMPC solve from the start applies [[u_0 = ${fmt(r.first.U[0], 3)}]] and lands its terminal state at [[(${fmt(r.first.terminal[0], 3)}, ${fmt(r.first.terminal[1], 3)})]], a convex combination of ${r.first.lam.filter(l => l > 1e-6).length} stored states. Mean solve time ${fmt(r.iterations.reduce((s, it) => s + it.ms, 0) / r.iterations.reduce((s, it) => s + it.U.length, 0), 1)} ms per QP in this browser. ` + (r.failed ? `Then: ${r.failed}.` : ""); }
  }
  caption($("#heroStatus"), text);
}
function heroStart() { heroCompute(() => drawHero()); }
$("#hN").addEventListener("input", () => { hs.N = +$("#hN").value; heroStart(); });
$("#hW").addEventListener("input", () => { hs.window = +$("#hW").value; heroStart(); });
$("#hI").addEventListener("input", () => { hs.iters = +$("#hI").value; heroStart(); });
$("#hS").addEventListener("input", () => { hs.seedN = +$("#hS").value; heroStart(); });
$("#hReset").addEventListener("click", () => { $("#hN").value = 3; $("#hW").value = 4; $("#hI").value = 9; $("#hS").value = 3; hs.N = 3; hs.window = 4; hs.iters = 9; hs.seedN = 3; heroStart(); });

/* =====================================================================
   Part 5: the scalar example and its convex extension
   ===================================================================== */
const sc = { probe: 2.5, extra: false };
function drawScalar() {
  const ctg = (xs, us) => { const J = new Array(xs.length).fill(0); for (let k = us.length - 1; k >= 0; k--) J[k] = xs[k] * xs[k] + 0.1 * us[k] * us[k] + J[k + 1]; return J; };
  const xp = [4, 3, 2, 1, 0], xn = [-3, -2, -1, 0], Jp = ctg(xp, [-1, -1, -1, -1]), Jn = ctg(xn, [1, 1, 1]);
  let pts = [...xn.slice(0, -1).map((x, i) => [x, Jn[i]]), ...xp.map((x, i) => [x, Jp[i]])];
  if (sc.extra) { const xw = [2, 1.6, 1.0, 0.4, 0], uw = [-0.4, -0.6, -0.6, -0.4], Jw = ctg(xw, uw); pts = pts.concat(xw.slice(0, -1).map((x, i) => [x, Jw[i], true])); }
  pts.sort((a, b) => a[0] - b[0]);
  const base = pts.filter(p => !p[2]), Sb = base.map(p => [p[0]]), Vb = base.map(p => p[1]);
  const improving = sc.extra ? pts.filter(p => p[2] && p[1] < convexSafeValue([p[0]], Sb, Vb).value - 1e-9) : [];
  const S = pts.map(p => [p[0]]), V = pts.map(p => p[1]), grid = Array.from({ length: 141 }, (_, i) => -3 + 7 * i / 140), Vc = grid.map(x => convexSafeValue([x], S, V).value);
  const cv = $("#scalarC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.1, { x: [-3.5, 4.5], y: [0, 33], xlabel: "x", ylabel: "cost-to-go", xticks: [-3, -2, -1, 0, 1, 2, 3, 4], yticks: [0, 10, 20, 30] }), { ctx, X, Y } = fr; cv._fr = fr;
  plotLine(fr, grid, Vc, { color: css("--post"), width: 2.2 });
  pts.forEach(p => { ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 4.5, 0, 2 * Math.PI); ctx.fill(); });
  const q = convexSafeValue([sc.probe], S, V), act = q.lam.map((l, i) => [l, i]).filter(([l]) => l > 1e-6);
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(sc.probe), Y(q.value), 5.5, 0, 2 * Math.PI); ctx.fill();
  act.forEach(([l, i]) => { ctx.strokeStyle = alpha("--ink", 0.4); ctx.setLineDash([3, 3]); seg(ctx, X(sc.probe), Y(q.value), X(pts[i][0]), Y(pts[i][1])); ctx.setLineDash([]); });
  legend(ctx, ["orange: stored (x, J) pairs · blue: convex extension V_c(x) · black: the probe"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#scExtra").setAttribute("aria-pressed", String(sc.extra));
  caption($("#scalarCap"), `Two successful runs of [[x^+ = x + u]], [[4 \\to 3 \\to 2 \\to 1 \\to 0]] and [[-3 \\to -2 \\to -1 \\to 0]], with [[h = x^2 + 0.1u^2]], give the stored pairs ${pts.filter(p => !sc.extra || true).slice(0, 8).map(p => `(${p[0]}, ${fmt(p[1], 1)})`).join(", ")}${sc.extra ? " plus a third, slower run through 2, 1.6, 1.0 and 0.4" : ""}. The curve is not data: for each [[x]] it is the cheapest convex combination of stored pairs that represents [[x]], the lower convex envelope. At [[x = ${fmt(sc.probe, 2)}]], [[V_c = ${fmt(q.value, 3)}]] from ${act.length === 1 ? `the stored point ${pts[act[0][1]][0]} alone` : act.map(([l, i]) => `[[\\lambda = ${fmt(l, 3)}]] on ${fmt(pts[i][0], 1)}`).join(" and ")}${Math.abs(sc.probe - 2.5) < 1e-9 && !sc.extra ? ": the notebook's query, halfway between 2 and 3, so halfway between 5.2 and 14.3" : ""}. ${sc.extra ? `The slower run costs ${fmt(pts.find(p => p[2] && p[0] === 2)[1], 3)} from [[x = 2]] against the stored 5.2, so its pairs at 2, 1.6 and 1.0 lie above the envelope: dominated, never selected, and the curve there is unchanged. ${improving.length ? `Its pair at ${improving.map(p => `(${fmt(p[0], 1)}, ${fmt(p[1], 3)})`).join(" and ")} lies below the old chord and lowers the envelope locally: a worse run can still contain a better partial continuation, and the LP keeps exactly that part.` : "No new pair lies below the old envelope."} Worse data never raise the terminal value.` : "Tap or click the chart to move the probe; the button adds a worse run (exercise 4)."}`);
}
$("#scalarC").addEventListener("click", ev => { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), fr = cv._fr; if (!fr) return; sc.probe = clamp(Math.round((fr.x0 + (ev.clientX - r.left - fr.pad.l) / fr.iw * (fr.x1 - fr.x0)) * 10) / 10, -3, 4); drawScalar(); });
$("#scExtra").addEventListener("click", () => { sc.extra = !sc.extra; drawScalar(); });

/* =====================================================================
   Part 12: the convex-combination identity on the stored seed
   ===================================================================== */
function drawInvariance() {
  const r = hs.result; if (!r || r.seed.failed) return;
  const store = makeStore(); addTrajectory(store, r.seed.X, r.seed.U); const { S, Us, next } = safeData(store, null);
  const t = +$("#ivT").value, chosen = [2, 5, 8, 12];
  /* the slider moves weight from the notebook's (0.10, 0.25, 0.30, 0.35) towards the first point; the weights stay convex */
  const lam = [0.10, 0.25, 0.30, 0.35].map((v, i) => i === 0 ? v * (1 - t) + t : v * (1 - t));
  const xd = [dot(lam, chosen.map(i => S[i][0])), dot(lam, chosen.map(i => S[i][1]))], ud = dot(lam, chosen.map(i => Us[i])), xFromDyn = [xd[0] + xd[1], xd[1] + ud], xFromData = [dot(lam, chosen.map(i => S[next[i]][0])), dot(lam, chosen.map(i => S[next[i]][1]))];
  const cv = $("#invC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.3 : 1.9, { x: [-16, 1.5], y: [-0.5, 6.5], xlabel: "x₁", ylabel: "x₂", xticks: [-15, -10, -5, 0], yticks: [0, 2, 4, 6] }), { ctx, X, Y } = fr;
  plotLine(fr, r.seed.X.map(x => x[0]), r.seed.X.map(x => x[1]), { color: alpha("--warm", 0.6), width: 1.4 });
  chosen.forEach((i, k) => { ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(S[i][0]), Y(S[i][1]), 3 + 8 * lam[k], 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = alpha("--ink", 0.3); seg(ctx, X(S[i][0]), Y(S[i][1]), X(xd[0]), Y(xd[1])); });
  ctx.fillStyle = css("--post"); ctx.beginPath(); ctx.arc(X(xd[0]), Y(xd[1]), 6, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--post"); ctx.lineWidth = 2; seg(ctx, X(xd[0]), Y(xd[1]), X(xFromDyn[0]), Y(xFromDyn[1])); ctx.beginPath(); ctx.arc(X(xFromDyn[0]), Y(xFromDyn[1]), 6, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X(xFromData[0]), Y(xFromData[1]), 9, 0, 2 * Math.PI); ctx.stroke();
  legend(ctx, ["orange: four stored seed states (size = weight) · blue: their combination and its successor", "black ring: the same combination of the stored successors"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#ivTV").textContent = fmt(t, 2);
  caption($("#invCap"), `Stored seed states 2, 5, 8 and 12 with weights [[\\lambda = (${lam.map(v => fmt(v, 3)).join(", ")})]]${t === 0 ? ", the notebook's weights" : ""}: the combination [[x = \\sum_r \\lambda_r s_r = (${fmt(xd[0], 4)}, ${fmt(xd[1], 4)})]] with the combined input [[u = \\sum_r \\lambda_r u_r = ${fmt(ud, 4)}]] moves to [[Ax + Bu = (${fmt(xFromDyn[0], 4)}, ${fmt(xFromDyn[1], 4)})]], and the same weights on the stored successors give [[(${fmt(xFromData[0], 4)}, ${fmt(xFromData[1], 4)})]]: identical to ${Math.max(Math.abs(xFromDyn[0] - xFromData[0]), Math.abs(xFromDyn[1] - xFromData[1])).toExponential(1)}. That is the whole invariance proof in one line of arithmetic: linearity moves a convex combination of safe states, driven by the same combination of their stored inputs, to the same combination of their safe successors, which lies in the hull. The slider shifts weight onto the first point.`);
}
$("#ivT").addEventListener("input", drawInvariance);

/* =====================================================================
   Part 21: database growth and the window
   ===================================================================== */
function drawGrowth() {
  const r = hs.result; if (!r || r.seed.failed) return;
  const sizes = []; let acc = 1; for (const X of r.store.trajectories) { acc += X.length - 1; sizes.push(acc); }
  const win = r.store.trajectories.map((_, j) => { const start = hs.window === 0 ? 0 : Math.max(0, j + 1 - hs.window); let s = 1; for (let i = start; i <= j; i++) s += r.store.trajectories[i].length - 1; return s; });
  const cv = $("#growC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.5 : 2.4, { x: [-0.5, sizes.length - 0.5], y: [0, Math.max(...sizes) * 1.15], xlabel: "stored trajectory index", xticks: sizes.map((_, i) => i) }), { ctx, X, Y } = fr;
  plotLine(fr, sizes.map((_, i) => i), sizes, { color: alpha("--ink", 0.5), width: 2, dash: [5, 4] }); plotLine(fr, win.map((_, i) => i), win, { color: css("--post"), width: 2.2 });
  ctx.fillStyle = css("--post"); win.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(i), Y(v), 3, 0, 2 * Math.PI); ctx.fill(); });
  legend(ctx, ["dashed: all stored safe states · blue: states in the online window"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const ms = r.iterations.map(it => it.ms / it.U.length);
  caption($("#growCap"), `Every successful iteration adds its states to the database: ${sizes[sizes.length - 1]} safe samples after ${r.store.trajectories.length} trajectories (the notebook counts 213 after the seed and nine iterations, because its solver's runs linger longer near the target, see Part 7). Each sample used online is one barycentric variable [[\\lambda_r]] in the QP. With ${hs.window === 0 ? "all data active, the QP grows every iteration" : `a window of the last ${hs.window} trajectories the online problem stays at about ${win[win.length - 1]} variables`}; the mean solve time per QP in this browser went from ${fmt(ms[0], 1)} ms in the first iteration to ${fmt(ms[ms.length - 1], 1)} ms in the last. The notebook measures 5 to 26 ms with SciPy, growing until the window fills and then falling again as the trajectories get shorter. Exercise 9 compares the two choices: same costs to [[10^{-5}]], an eightfold difference in solve time by iteration 9.`);
}

/* =====================================================================
   Part 9: hull growth, input sequences, stored costs-to-go
   ===================================================================== */
function hull(P) { const pts = [...P].sort((a, b) => a[0] - b[0] || a[1] - b[1]); if (pts.length < 3) return pts; const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); const lower = []; for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); } const upper = []; for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); } return lower.slice(0, -1).concat(upper.slice(0, -1)); }
const det = { view: "hull" };
function drawDetail() {
  const r = hs.result; if (!r || r.seed.failed) return;
  const cv = $("#detC"), narrow = narrowOf(cv), T = r.store.trajectories, last = r.iterations[r.iterations.length - 1];
  if (det.view === "hull") {
    const pointsOf = list => { const P = [[0, 0]]; list.forEach(X => X.slice(0, -1).forEach(x => P.push(x))); return P; };
    const nMid = Math.min(3, T.length - 1);
    const stages = [[[T[0]], alpha("--warm", 0.9)], [T.slice(0, nMid + 1), alpha("--post", 0.5)], [T, css("--post")]];
    const fr = frame(cv, narrow ? 1.2 : 1.9, { x: [-16, 1.5], y: [-0.5, 6.5], xlabel: "x₁", ylabel: "x₂", xticks: [-15, -10, -5, 0], yticks: [0, 2, 4, 6] }), { ctx, X, Y } = fr;
    const all = pointsOf(T); ctx.fillStyle = alpha("--ink", 0.35); all.forEach(p => { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 1.6, 0, 2 * Math.PI); ctx.fill(); });
    const areas = stages.map(([list, color], i) => { const H = hull(pointsOf(list)); if (H.length >= 3) { ctx.strokeStyle = color; ctx.lineWidth = i === 2 ? 2.4 : 1.6; ctx.beginPath(); H.forEach((p, k) => k ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))); ctx.closePath(); ctx.stroke(); } let a = 0; for (let k = 0; k < H.length; k++) { const p = H[k], q = H[(k + 1) % H.length]; a += p[0] * q[1] - q[0] * p[1]; } return Math.abs(a) / 2; });
    legend(ctx, ["orange: hull after the seed · light blue: after " + nMid + " iterations · blue: after all · dots: stored states"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    caption($("#detCap"), `Convex hulls of the accumulated successful states, with areas ${fmt(areas[0], 2)} after the seed, ${fmt(areas[1], 2)} after ${nMid} iteration${nMid > 1 ? "s" : ""} and ${fmt(areas[2], 2)} after all ${T.length - 1}. The seed's hull is the region between one slow curve and its chord to the origin; the first learning runs push it out to the fast, saturated trajectories, a factor ${fmt(areas[2] / Math.max(areas[0], 1e-9), 2)} in area; after that it changes by ${fmt(100 * (areas[2] - areas[1]) / Math.max(areas[1], 1e-9), 2)} percent while the cost keeps falling. The geometry is only half of the learning.`);
  } else if (det.view === "ctrl") {
    const seqs = [[r.seed.U, css("--warm"), []], [last ? last.U : [], css("--post"), []], [REF.U, alpha("--ink", 0.55), [6, 4]]];
    const L = Math.max(...seqs.map(s => s[0].length)), fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, L], y: [-6, 6], xlabel: "time step", ylabel: "u", yticks: [-5, 0, 5] }), { ctx, X, Y } = fr;
    ctx.strokeStyle = alpha("--ink", 0.3); ctx.setLineDash([3, 3]); seg(ctx, X(0), Y(5), X(L), Y(5)); seg(ctx, X(0), Y(-5), X(L), Y(-5)); ctx.setLineDash([]);
    seqs.forEach(([U, color, dash]) => { if (!U.length) return; const xs = [], ys = []; U.forEach((u, k) => { xs.push(k, k + 1); ys.push(u, u); }); plotLine(fr, xs, ys, { color, width: 2, dash }); });
    legend(ctx, ["orange: seed MPC · blue: last LMPC iteration · dashed: reference · dotted: |u| = 5"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    const sat = U => U.filter(u => Math.abs(u) > 5 - 1e-6).length;
    caption($("#detCap"), `Input sequences. The seed never saturates (largest input ${fmt(Math.max(...r.seed.U.map(Math.abs)), 3)}); the last LMPC iteration ${last ? `saturates on ${sat(last.U)} steps, first accelerating at [[+5]] then braking at [[-5]]` : "is not available"}, like the reference, which saturates on ${sat(REF.U)} steps. The three-step controller learned to use the full input from data that told it what the slow approach costs.`);
  } else {
    const J0 = costToGo(r.seed.X, r.seed.U), J1 = last ? costToGo(last.X, last.U) : [], top = Math.max(J0[0], 1) * 1.1;
    const fr = frame(cv, narrow ? 1.4 : 2.2, { x: [0, Math.max(J0.length, J1.length)], y: [0, top], xlabel: "time step t", ylabel: "stored J_t", yticks: [0, 200, 400, 600] }), { ctx, X, Y } = fr;
    plotLine(fr, J0.map((_, k) => k), J0, { color: css("--warm"), width: 2 }); if (J1.length) plotLine(fr, J1.map((_, k) => k), J1, { color: css("--post"), width: 2 });
    ctx.fillStyle = css("--warm"); J0.forEach((v, k) => { ctx.beginPath(); ctx.arc(X(k), Y(v), 2.2, 0, 2 * Math.PI); ctx.fill(); }); ctx.fillStyle = css("--post"); J1.forEach((v, k) => { ctx.beginPath(); ctx.arc(X(k), Y(v), 2.2, 0, 2 * Math.PI); ctx.fill(); });
    legend(ctx, ["orange: along the seed · blue: along the last LMPC iteration"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    let res = 0; if (last) for (let k = 0; k < last.U.length; k++) res = Math.max(res, Math.abs(J1[k] - stageCost(last.X[k], last.U[k]) - J1[k + 1]));
    caption($("#detCap"), `Stored costs-to-go [[J_t = h(x_t, u_t) + J_{t+1}]], [[J_T = 0]], along the seed and the last iteration: these are the values [[V_r]] that the terminal cost [[\\lambda^\\top V]] interpolates. [[J_0]] is the iteration cost of the learning curve, ${fmt(J0[0], 3)} for the seed${last ? ` and ${fmt(J1[0], 4)} for the last iteration` : ""}. The seed spends its cost slowly, lingering near [[x_1 = -15]] at low speed; the learned run pays almost all of it in the first saturated steps. Backward-recursion residual on the last iteration: ${res.toExponential(1)}, round-off, as in the notebook.`);
  }
}
$$("#detW [data-view]").forEach(b => b.addEventListener("click", () => { det.view = b.dataset.view; $$("#detW [data-view]").forEach(o => o.setAttribute("aria-pressed", String(o === b))); drawDetail(); }));

function drawAll() { drawHero(); drawScalar(); drawInvariance(); drawGrowth(); drawDetail(); }
heroCompute(partial => { drawHero(); if (!partial) { drawInvariance(); drawGrowth(); drawDetail(); } });
drawScalar();
onRedraw(drawAll);
})();
