/* Advanced MPC and Learning MPC, chapter A2, prediction matrices and condensing: the demos.
   The hero builds S_x and S_u for the canonical system at any horizon, predicts a whole trajectory
   in one product and checks it against a step-by-step rollout; a second widget draws the sparsity
   of the sparse and condensed problems. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, fit, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { SYS, mv, mm, tr, dot, zeros, predictionMatrices, rollout, eigsym2, solve, legend, drawPoly, vertices, drawPath } = window.MPC;

const UBASE = [0.8, 0.2, -0.4, -0.5, 0.3, 0.1, -0.2, 0.4];
const X0 = [-3, 0];
const st = { scale: 1 };

/* the condensed cost for Q, R and P = Q: H and f, plus the dropped constant */
function condensed(Sx, Su, N) {
  const n = 2, Qbar = zeros(N * n, N * n);
  for (let i = 0; i < N; i++) for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Qbar[i * n + r][i * n + c] = SYS.Q[r][c];
  const SuTQ = mm(tr(Su), Qbar), H = mm(SuTQ, Su).map((r, i) => r.map((v, j) => 2 * (v + (i === j ? SYS.R : 0)))), f = mv(mm(SuTQ, Sx), X0).map(v => 2 * v);
  const c = dot(X0, mv(mm(tr(Sx), mm(Qbar, Sx)), X0)) + dot(X0, mv(SYS.Q, X0));
  return { H, f, c };
}
/* symmetric eigenvalues by Jacobi rotations (small matrices only) */
function eigsym(M0) {
  const n = M0.length, M = M0.map(r => r.slice());
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += M[i][j] * M[i][j];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(M[p][q]) < 1e-300) continue;
      const th = 0.5 * Math.atan2(2 * M[p][q], M[q][q] - M[p][p]), c = Math.cos(th), s = Math.sin(th);
      for (let k = 0; k < n; k++) { const a = M[k][p], b = M[k][q]; M[k][p] = c * a - s * b; M[k][q] = s * a + c * b; }
      for (let k = 0; k < n; k++) { const a = M[p][k], b = M[q][k]; M[p][k] = c * a - s * b; M[q][k] = s * a + c * b; }
    }
  }
  return M.map((r, i) => r[i]).sort((a, b) => a - b);
}

function drawHero() {
  const N = +$("#hN").value, { Sx, Su } = predictionMatrices(SYS.A, SYS.B, N), U = UBASE.slice(0, N).map(u => u * st.scale);
  const Xc = mv(Sx, X0).map((v, r) => v + dot(Su[r], U)), Xr = rollout(SYS.A, SYS.B, X0, U).slice(1), err = Math.max(...Xc.map((v, i) => Math.abs(v - Xr.flat()[i])));
  const table = (M, name) => `<div class="mat"><span class="mlabel">${name}</span><table class="num">` + M.map(r => "<tr>" + r.map(v => `<td>${Number.isInteger(v) ? v : fmt(v, 1)}</td>`).join("") + "</tr>").join("") + "</table></div>";
  $("#pMats").innerHTML = table(Sx, `S<sub>x</sub> (${2 * N}×2)`) + table(Su, `S<sub>u</sub> (${2 * N}×${N})`);
  /* phase plane: rollout as a line, the predicted states as rings */
  const cv = $("#predC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.2 : 1.6, { x: [-5.5, 5.5], y: [-4, 4], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, 0, 3] }), { ctx, X, Y } = fr;
  drawPoly(fr, vertices(SYS.X), alpha("--ink", 0.04), alpha("--ink", 0.35), 1, [3, 3]);
  drawPath(fr, [X0, ...Xr], css("--ink"), 1.6, 0);
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 2; for (let i = 0; i < N; i++) { ctx.beginPath(); ctx.arc(X(Xc[2 * i]), Y(Xc[2 * i + 1]), 5, 0, 2 * Math.PI); ctx.stroke(); }
  ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(X0[0]), Y(X0[1]), 5, 0, 2 * Math.PI); ctx.fill();
  legend(ctx, ["black: step-by-step rollout · blue rings: S_x x_0 + S_u U", "orange: x_0 = (−3, 0)"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const { H, f, c } = condensed(Sx, Su, N), ev = eigsym(H), Ustar = solve(H, f.map(v => -v));
  $("#hNV").textContent = N; $("#hScaleV").textContent = fmt(st.scale, 2);
  $("#rErr").textContent = err.toExponential(1); $("#rCond").textContent = fmt(ev[ev.length - 1] / ev[0], 1); $("#rMin").textContent = fmt(ev[0], 4); $("#rU0").textContent = fmt(Ustar[0], 3);
  caption($("#heroStatus"), `[[N = ${N}]]: [[\\mathcal S_x]] stacks [[A, A^2, \\dots, A^{${N}}]], and block row [[i]] of [[\\mathcal S_u]] holds [[A^{i-1}B, \\dots, B]] followed by zeros, since [[u_j]] cannot influence [[x_i]] for [[j \\ge i]]. For the cart [[A^i = \\begin{pmatrix} 1 & i \\\\ 0 & 1 \\end{pmatrix}]] and [[A^i B = (i + \\tfrac12, 1)]], which is what the columns show. With [[U = (${U.map(u => fmt(u, 2)).join(", ")})]] the one-product prediction and the rollout agree to ${err.toExponential(1)}. The condensed Hessian [[H = 2(\\mathcal S_u^\\top \\bar Q \\mathcal S_u + \\bar R)]] has eigenvalues from ${fmt(ev[0], 4)} to ${fmt(ev[ev.length - 1], 1)}, condition number ${fmt(ev[ev.length - 1] / ev[0], 0)}; it is positive definite for every [[N]] because [[R > 0]]. The unconstrained minimiser [[U^\\star = -H^{-1} f]] starts with a push of ${fmt(Ustar[0], 3)}${Math.abs(Ustar[0]) > 1 ? ", beyond the bound of 1: constraints are not an afterthought" : ""}; the dropped constant is [[c(x_0) = ${fmt(c, 1)}]].`);
}
$("#hN").addEventListener("input", drawHero);
$("#hScale").addEventListener("input", () => { st.scale = +$("#hScale").value; drawHero(); });

/* sparse against condensed structure */
function drawStruct() {
  const N = +$("#sN").value, n = +$("#sn").value, m = +$("#sm").value, nz = N * n + N * m;
  const Hs = zeros(nz, nz); for (let k = 0; k < N; k++) { for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Hs[k * n + r][k * n + c] = 1; for (let r = 0; r < m; r++) for (let c = 0; c < m; c++) Hs[N * n + k * m + r][N * n + k * m + c] = 1; }
  const Es = zeros(N * n, nz); for (let k = 0; k < N; k++) for (let r = 0; r < n; r++) { for (let c = 0; c < n; c++) { Es[k * n + r][k * n + c] = 1; if (k) Es[k * n + r][(k - 1) * n + c] = 1; } for (let c = 0; c < m; c++) Es[k * n + r][N * n + k * m + c] = 1; }
  const Hc = zeros(N * m, N * m).map(r => r.map(() => 1)), Gc = zeros(N * n, N * m); for (let i = 0; i < N; i++) for (let r = 0; r < n; r++) for (let j = 0; j <= i; j++) for (let c = 0; c < m; c++) Gc[i * n + r][j * m + c] = 1;
  const count = M => M.flat().reduce((s, v) => s + v, 0);
  const draw = (id, M, title) => { const cv = $(id), w = cv.clientWidth || cv.parentElement.clientWidth, rows = M.length, cols = M[0].length, cell = Math.max(1, Math.floor(Math.min((w - 8) / cols, 160 / rows))), h = rows * cell + 22, { ctx } = fit(cv, w / h); ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText(`${title} ${rows}×${cols}`, 2, 12); const x0 = Math.max(2, (w - cols * cell) / 2); ctx.fillStyle = alpha("--ink", 0.08); ctx.fillRect(x0, 18, cols * cell, rows * cell); ctx.fillStyle = css("--post"); M.forEach((r, i) => r.forEach((v, j) => { if (v) ctx.fillRect(x0 + j * cell, 18 + i * cell, Math.max(1, cell - (cell > 3 ? 1 : 0)), Math.max(1, cell - (cell > 3 ? 1 : 0))); })); };
  draw("#sHs", Hs, "sparse H"); draw("#sEs", Es, "sparse dynamics E"); draw("#sHc", Hc, "condensed H"); draw("#sGc", Gc, "condensed H̄_x S_u");
  $("#sNV").textContent = N; $("#snV").textContent = n; $("#smV").textContent = m;
  caption($("#structCap"), `[[n = ${n}]], [[m = ${m}]], [[N = ${N}]]: the sparse problem has [[Nn + Nm = ${nz}]] decision variables with a block-diagonal Hessian (${count(Hs)} nonzeros of ${nz * nz}) and [[Nn = ${N * n}]] dynamics equalities whose matrix has only ${count(Es)} nonzeros; the condensed problem has [[Nm = ${N * m}]] variables, a dense Hessian (${count(Hc)} nonzeros, every entry) and dense lower-triangular constraint rows (${count(Gc)} of ${N * n * N * m}). The sparse matrices grow linearly with [[N]], the condensed ones quadratically. For exercise 6, [[n = 8]], [[m = 2]], [[N = 30]]: 300 sparse variables against 60 condensed; the sparse problem is five times larger and can still be faster, because a structure-exploiting solver factorises its banded KKT system in time linear in [[N]], while the dense condensed Hessian costs [[\\mathcal O((Nm)^3)]].`);
}
["#sN", "#sn", "#sm"].forEach(id => $(id).addEventListener("input", drawStruct));

function drawAll() { drawHero(); drawStruct(); }
drawAll();
onRedraw(drawAll);
})();
