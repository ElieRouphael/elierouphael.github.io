/* Advanced MPC and Learning MPC: dynamic programming on the lattice, shared by chapters A14 and B5.
   Everything is tabular and exact: the integer chain, the double-integrator lattices (which the
   dynamics map into themselves), the discounted crash-state problem with value and policy iteration,
   the undiscounted hard-constrained value iteration, receding-horizon closed loops of a policy table,
   the LMPC loop on the lattice and tabular Q-learning. The polytopes of chapters A9 and A10 are computed
   live with mpc-lib.js. Loaded after mpc-lib.js, before the chapter's own script. */
(function (global) {
"use strict";
const { SYS, maximalControlInvariant, maximalInvariant, closedLoopAdmissible, contains, solve } = global.MPC;
const A = SYS.A, B = [SYS.B[0][0], SYS.B[1][0]], Q = SYS.Q, R = SYS.R;

/* ---------------- the integer chain: x+ = x + u, |x| <= xmax, u in {-umax..umax}, cost x^2 + u^2, x_N = 0 ---------------- */
function chainDP(N, umax = 2, xmax = 10) {
  const xs = []; for (let x = -xmax; x <= xmax; x++) xs.push(x);
  const us = []; for (let u = -umax; u <= umax; u++) us.push(u);
  const J = new Array(N + 1), mu = new Array(N);
  J[N] = xs.map(x => x === 0 ? 0 : Infinity);                       // the terminal constraint as an infinite cost
  for (let k = N - 1; k >= 0; k--) {
    J[k] = xs.map(x => Math.min(...us.map(u => Math.abs(x + u) <= xmax ? x * x + u * u + J[k + 1][x + u + xmax] : Infinity)));
    mu[k] = xs.map((x, i) => Number.isFinite(J[k][i]) ? us[us.map(u => Math.abs(x + u) <= xmax ? x * x + u * u + J[k + 1][x + u + xmax] : Infinity).indexOf(J[k][i])] : null);
  }
  return { xs, us, J, mu };
}

/* ---------------- the exact lattice of the double integrator ----------------
   States with v in hZ and p + v/2 in hZ, inputs in hZ: x+ = Ax + Bu maps the lattice into itself.
   succ[i][a] is the successor index, or -1 when the successor leaves |p| <= pMax, |v| <= vMax. */
function buildLattice(h, pMax = 4, vMax = 3, uMax = 1) {
  const Pm = Math.round(2 * pMax / h), Vm = Math.round(vMax / h), Am = Math.round(uMax / h), I0 = Math.floor((Pm + Vm) / 2) + 1, ij = [];
  for (let j = -Vm; j <= Vm; j++) for (let i = -I0; i <= I0; i++) if (Math.abs(2 * i - j) <= Pm) ij.push([i, j]);
  const index = new Map(); ij.forEach(([i, j], k) => index.set(i + "," + j, k));
  const a = []; for (let v = -Am; v <= Am; v++) a.push(v);
  const succ = ij.map(([i, j]) => a.map(av => { const i2 = i + j + av, j2 = j + av; return (Math.abs(j2) <= Vm && Math.abs(2 * i2 - j2) <= Pm) ? index.get(i2 + "," + j2) : -1; }));
  const S = ij.map(([i, j]) => [(2 * i - j) * h / 2, j * h]), U = a.map(v => v * h), q = S.map(s => U.map(u => s[0] * s[0] * Q[0][0] + s[1] * s[1] * Q[1][1] + R * u * u));
  return { S, U, succ, q, n: S.length, m: U.length, h, origin: S.findIndex(s => s[0] === 0 && s[1] === 0) };
}
/* the discounted problem with an absorbing, cost-free crash state at index n; leaving the box costs cCrash */
function crashProblem(L, gamma, cCrash) {
  const n = L.n, m = L.m, succ = L.succ.map(r => r.map(s => s >= 0 ? s : n)).concat([new Array(m).fill(n)]);
  const cost = L.q.map((r, i) => r.map((c, a) => c + (L.succ[i][a] < 0 ? cCrash : 0))).concat([new Array(m).fill(0)]);
  return { n, m, succ, cost, gamma };
}
/* the Bellman operator: (TJ)(x) = min_u { cost(x,u) + gamma J(f(x,u)) }, with the greedy action */
function bellman(Pb, J) {
  const TJ = new Array(Pb.n + 1), act = new Array(Pb.n + 1);
  for (let i = 0; i <= Pb.n; i++) {
    let best = Infinity, ba = 0;
    for (let a = 0; a < Pb.m; a++) { const v = Pb.cost[i][a] + Pb.gamma * J[Pb.succ[i][a]]; if (v < best) { best = v; ba = a; } }
    TJ[i] = best; act[i] = ba;
  }
  return { TJ, act };
}
/* value iteration with the a-posteriori stopping rule gamma/(1-gamma) ||J_{k+1} - J_k|| < tol */
function valueIteration(Pb, J0, tol = 1e-12, maxIter = 5000) {
  let J = J0.slice(), k;
  for (k = 1; k < maxIter; k++) {
    const { TJ } = bellman(Pb, J);
    let d = 0; for (let i = 0; i <= Pb.n; i++) d = Math.max(d, Math.abs(TJ[i] - J[i]));
    if (Pb.gamma / (1 - Pb.gamma) * d < tol) { J = TJ; break; }
    J = TJ;
  }
  return { J, iterations: k };
}
/* viable states: some input sequence keeps them in the box for ever (a Boolean fixed point) */
function viable(L) {
  let v = new Array(L.n).fill(true);
  for (;;) { const nxt = v.map((ok, i) => ok && L.succ[i].some(s => s >= 0 && v[s])); if (nxt.every((x, i) => x === v[i])) return v; v = nxt; }
}
/* policy evaluation: solve (I - gamma P_mu) J = q_mu exactly */
function evaluatePolicy(Pb, mu) {
  const N = Pb.n + 1, M = Array.from({ length: N }, () => new Array(N).fill(0)), b = new Array(N);
  for (let i = 0; i < N; i++) { M[i][i] += 1; M[i][Pb.succ[i][mu[i]]] -= Pb.gamma; b[i] = Pb.cost[i][mu[i]]; }
  return solve(M, b);
}
/* policy iteration from mu0: evaluate, improve (keeping the old action on ties), stop when nothing changes */
function policyIteration(Pb, mu0, Jstar) {
  let mu = mu0.slice(); const trace = [];
  for (let k = 0; k < 50; k++) {
    const Jmu = evaluatePolicy(Pb, mu), Qsa = Pb.cost.map((r, i) => r.map((c, a) => c + Pb.gamma * Jmu[Pb.succ[i][a]]));
    const muNew = mu.map((a, i) => { const best = Math.min(...Qsa[i]); return Qsa[i][a] <= best + 1e-10 ? a : Qsa[i].indexOf(best); });
    const changed = muNew.filter((a, i) => a !== mu[i]).length;
    trace.push({ k, err: Jstar ? Math.max(...Jmu.map((v, i) => Math.abs(v - Jstar[i]))) : NaN, changed, mu: mu.slice(), Jmu });
    if (!changed) break;
    mu = muNew;
  }
  return trace;
}
/* the undiscounted Bellman operator with hard constraints: an inadmissible input is simply skipped */
function bellmanHard(L, J) {
  const TJ = new Array(L.n), act = new Array(L.n);
  for (let i = 0; i < L.n; i++) {
    let best = Infinity, ba = 0;
    for (let a = 0; a < L.m; a++) { const s = L.succ[i][a]; if (s < 0) continue; const v = L.q[i][a] + J[s]; if (v < best - 1e-9) { best = v; ba = a; } }
    TJ[i] = best; act[i] = ba;
  }
  return { TJ, act };   /* ties within 1e-9 go to the first action, as NumPy's argmin does on exact ties */
}
/* value iteration from J_0 = 0 until the exact fixed point; returns every iterate */
function hardValueIteration(L) {
  const iters = [new Array(L.n).fill(0)];
  for (;;) { const { TJ } = bellmanHard(L, iters[iters.length - 1]); if (TJ.every((v, i) => v === iters[iters.length - 1][i])) break; iters.push(TJ); }
  return iters;
}
/* receding horizon of a policy table from every state of F: cost and success (reaches the origin, never infeasible) */
function recedingHorizon(L, kappa, F, steps = 80) {
  const out = [];
  for (let s0 = 0; s0 < L.n; s0++) {
    if (!F[s0]) continue;
    let s = s0, cost = 0, ok = true;
    for (let t = 0; t < steps; t++) { ok = ok && F[s]; const a = kappa[s]; if (ok) cost += L.q[s][a]; const sn = L.succ[s][a]; ok = ok && sn >= 0; if (sn >= 0) s = sn; }
    out.push({ start: s0, cost, ok: ok && s === L.origin });
  }
  return out;
}
/* MPC with terminal cost Vf (infinite outside X_f) for N = 1..Nmax: J_N = T^N Vf, the law is greedy, closed loops from every feasible start */
function designTable(L, Vf, Jstar, fin, Nmax = 12) {
  let Jprev = Vf.slice(); const rows = []; let upper = true, mono = true, bounded = true;
  for (let N = 1; N <= Nmax; N++) {
    const { TJ, act } = bellmanHard(L, Jprev), F = TJ.map(Number.isFinite), rh = recedingHorizon(L, act, F);
    const nz = rh.filter(r => r.ok && r.start !== L.origin), subs = nz.map(r => Math.max(r.cost - Jstar[r.start], 0) / Jstar[r.start]);
    rows.push({ N, feasible: F.filter(Boolean).length, doomed: F.filter((f, i) => f && !fin[i]).length, failed: rh.filter(r => !r.ok).length,
      optimal: rh.filter(r => r.ok && Math.abs(r.cost - Jstar[r.start]) < 1e-9).length,
      mean: subs.length ? 100 * subs.reduce((s, v) => s + v, 0) / subs.length : NaN, max: subs.length ? 100 * Math.max(...subs) : NaN, F, act, TJ, rh });
    upper = upper && TJ.every((v, i) => !F[i] || v >= Jstar[i] - 1e-9);
    mono = mono && TJ.every((v, i) => v <= Jprev[i] + 1e-9);
    bounded = bounded && rh.every(r => !r.ok || r.cost <= TJ[r.start] + 1e-9);
    Jprev = TJ;
  }
  return { rows, upper, mono, bounded };
}
/* rollout of a policy table from s0 to the origin */
function rolloutPolicy(L, pol, s0, maxSteps = 200) {
  const st = [s0], ac = [];
  while (st[st.length - 1] !== L.origin && ac.length < maxSteps) { const a = pol[st[st.length - 1]]; ac.push(a); const s = L.succ[st[st.length - 1]][a]; if (s < 0) break; st.push(s); }
  return { st, ac };
}
/* the LMPC loop on the lattice: the sampled safe set's values V^j as the terminal cost, horizon NL, iterations from a seed trajectory */
function lmpcLattice(L, Jstar, seed, NL, iterations = 9) {
  const V = new Array(L.n).fill(Infinity); V[L.origin] = 0;
  let { st, ac } = seed; const costs = [], runs = []; let minGap = Infinity;
  for (let j = 0; j < iterations; j++) {
    const ctg = new Array(ac.length).fill(0);
    for (let t = ac.length - 1; t >= 0; t--) ctg[t] = L.q[st[t]][ac[t]] + (ctg[t + 1] || 0);
    costs.push(ctg[0]); runs.push({ st: st.slice(), ctg });
    st.slice(0, -1).forEach((s, t) => { minGap = Math.min(minGap, ctg[t] - Jstar[s]); V[s] = Math.min(V[s], ctg[t]); });
    let Jt = V.slice(); for (let i = 0; i < NL - 1; i++) Jt = bellmanHard(L, Jt).TJ;
    ({ st, ac } = rolloutPolicy(L, bellmanHard(L, Jt).act, seed.st[0]));
  }
  return { costs, minGap, runs };
}
/* a sluggish seed: the optimal policy for the cost w x'Qx + u^2 (small w makes it lazy) */
function sluggishSeed(L, s0, w) {
  const xQx = L.S.map(s => s[0] * s[0] * Q[0][0] + s[1] * s[1] * Q[1][1]);
  const qLazy = L.q.map((r, i) => L.U.map((u, a) => L.succ[i][a] >= 0 ? w * xQx[i] + u * u : Infinity));
  let J = new Array(L.n).fill(0);
  for (let it = 0; it < 10000; it++) { const Jn = qLazy.map((r, i) => Math.min(...r.map((c, a) => L.succ[i][a] >= 0 ? c + J[L.succ[i][a]] : Infinity))); if (Jn.every((v, i) => v === J[i])) break; J = Jn; }
  const pol = qLazy.map((r, i) => { const vals = r.map((c, a) => L.succ[i][a] >= 0 ? c + J[L.succ[i][a]] : Infinity); return vals.indexOf(Math.min(...vals)); });
  return rolloutPolicy(L, pol, s0);
}

/* ---------------- tabular Q-learning on the crash problem, with a small deterministic random stream ---------------- */
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
function qLearning(Pb, L, V3, optAct, cfg, nSteps, every = 2000, seed = 1) {
  const r = rng(seed), n = L.n, m = L.m, Qt = Array.from({ length: n }, () => new Array(m).fill(cfg.Q0 || 0)), visits = Array.from({ length: n }, () => new Array(m).fill(0)), log = [];
  let s = Math.floor(r() * n), tEp = 0, crashes = 0;
  for (let t = 0; t < nSteps; t++) {
    const row = Qt[s]; let a;
    if (r() < cfg.eps) a = Math.floor(r() * m); else { a = 0; for (let k = 1; k < m; k++) if (row[k] < row[a]) a = k; }
    const s2 = L.succ[s][a], c = Pb.cost[s][a], target = s2 < 0 ? c : c + Pb.gamma * Math.min(...Qt[s2]);   // the temporal-difference target
    visits[s][a]++;
    const step = cfg.alpha === "decay" ? visits[s][a] ** -0.6 : cfg.alpha;
    row[a] += step * (target - row[a]);
    if (s2 < 0) crashes++;
    tEp++;
    if (s2 < 0 || tEp === 30) { s = Math.floor(r() * n); tEp = 0; } else s = s2;
    if ((t + 1) % every === 0) { let good = 0, tot = 0; for (let i = 0; i < n; i++) if (V3[i]) { tot++; let g = 0; for (let k = 1; k < m; k++) if (Qt[i][k] < Qt[i][g]) g = k; if (optAct[i][g]) good++; } log.push([t + 1, good / tot, crashes]); }
  }
  return { Q: Qt, log };
}

/* ---------------- the polytopes of chapters A9 and A10, computed live ---------------- */
const Cc = maximalControlInvariant(A, SYS.B, SYS.X, SYS.umin, SYS.umax);
const polys = { C_hist: Cc.history, C_inf: Cc.set, O_inf: maximalInvariant(SYS.lqr.Acl, closedLoopAdmissible(SYS.X, SYS.lqr.K[0], SYS.umin, SYS.umax)).set };
const inside = (poly, x, tol = 1e-9) => contains(poly, x, tol);

global.DP = { A, B, Q, R, chainDP, buildLattice, crashProblem, bellman, valueIteration, viable, evaluatePolicy, policyIteration,
  bellmanHard, hardValueIteration, recedingHorizon, designTable, rolloutPolicy, lmpcLattice, sluggishSeed, rng, qLearning, polys, inside };
})(window);
