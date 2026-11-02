/* Advanced MPC and Learning MPC, chapter A3, convexity, QPs and KKT: the demos.
   The hero solves the two-variable QP  min ||z - c||^2  s.t.  z1 + z2 <= b  by the KKT conditions and
   shows the multiplier as a force and as a shadow price, next to the dual function; two widgets test
   convexity of a set by segments and of a quadratic by its Hessian. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { dot, eigsym2, solveQP, legend } = window.MPC;

/* the hand QP: with a = (1, 1) and ||a||^2 = 2, lambda* = max(a'c - b, 0), z* = c - (lambda/2) a */
function kktExample(c, b) { const lam = Math.max(c[0] + c[1] - b, 0), z = [c[0] - lam / 2, c[1] - lam / 2]; return { z, lam, fstar: (z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2, objNoConst: (z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2 - dot(c, c) }; }
/* its dual d(lambda) = -1/2 (q + a lambda)' P^{-1} (q + a lambda) - b lambda, with P = 2I, q = -2c */
function dualValue(c, b, lam) { const y = [-2 * c[0] + lam, -2 * c[1] + lam]; return -0.25 * (y[0] * y[0] + y[1] * y[1]) - b * lam; }

function drawHero() {
  const c = [+$("#kC1").value, +$("#kC2").value], b = +$("#kB").value, k = kktExample(c, b);
  /* the same problem through the toolkit's solver, for the readouts */
  const r = solveQP([[2, 0], [0, 2]], [-2 * c[0], -2 * c[1]], [[1, 1]], [b]);
  const cv = $("#kktC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.2 : 1.6, { x: [-0.6, 2.6], y: [-0.6, 2.2], xlabel: "z₁", ylabel: "z₂", xticks: [0, 1, 2], yticks: [0, 1, 2] }), { ctx, X, Y } = fr;
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  ctx.beginPath(); ctx.moveTo(X(-1), Y(b + 1)); ctx.lineTo(X(b + 1), Y(-1)); ctx.lineTo(X(-1), Y(-1)); ctx.closePath(); ctx.fillStyle = alpha("--post", 0.14); ctx.fill();
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 1.5; seg(ctx, X(-1), Y(b + 1), X(b + 1), Y(-1));
  for (const rad of [0.25, 0.5, 0.75, 1, 1.25]) { ctx.strokeStyle = alpha("--ink", 0.18); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X(c[0]), Y(c[1]), rad * fr.iw / (fr.x1 - fr.x0), 0, 2 * Math.PI); ctx.stroke(); }
  if (k.fstar > 1e-9) { ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(X(c[0]), Y(c[1]), Math.sqrt(k.fstar) * fr.iw / (fr.x1 - fr.x0), 0, 2 * Math.PI); ctx.stroke(); }
  if (k.lam > 1e-9) {
    /* the gradient of f at z* (towards c, length |grad|/2 for legibility) and the constraint normal lambda a, which cancel */
    ctx.strokeStyle = css("--c4"); ctx.lineWidth = 2; seg(ctx, X(k.z[0]), Y(k.z[1]), X(k.z[0] - k.lam / 2), Y(k.z[1] - k.lam / 2));
    ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2; seg(ctx, X(k.z[0]), Y(k.z[1]), X(k.z[0] + k.lam / 2), Y(k.z[1] + k.lam / 2));
  }
  ctx.restore();
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(c[0]), Y(c[1]), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(k.z[0]), Y(k.z[1]), 5.5, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText("c", X(c[0]) + 7, Y(c[1]) - 6); ctx.fillText("z*", X(k.z[0]) + 7, Y(k.z[1]) + 12);
  legend(ctx, ["shaded: z₁ + z₂ ≤ b · circles: level sets of ‖z − c‖²", "green: ∇f(z*)/2 · orange: λa/2, they cancel"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const cv2 = $("#dualC"), fr2 = frame(cv2, narrow ? 1.5 : 2.4, { x: [0, 3], y: [-3.2, 1.2], xlabel: "λ", ylabel: "d(λ)", xticks: [0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1] }), ctx2 = fr2.ctx;
  const ls = Array.from({ length: 121 }, (_, i) => 3 * i / 120), ds = ls.map(l => dualValue(c, b, l));
  plotLine(fr2, ls, ds, { color: css("--post"), width: 2 });
  ctx2.strokeStyle = css("--warm"); ctx2.setLineDash([4, 4]); seg(ctx2, fr2.X(0), fr2.Y(k.objNoConst), fr2.X(3), fr2.Y(k.objNoConst)); ctx2.setLineDash([]);
  ctx2.fillStyle = css("--warm"); ctx2.beginPath(); ctx2.arc(fr2.X(k.lam), fr2.Y(dualValue(c, b, k.lam)), 5, 0, 2 * Math.PI); ctx2.fill();
  legend(ctx2, ["blue: the dual function d(λ) · dashed: the primal optimum p*"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  $("#kC1V").textContent = fmt(c[0], 2); $("#kC2V").textContent = fmt(c[1], 2); $("#kBV").textContent = fmt(b, 2);
  const grad = [2 * (k.z[0] - c[0]), 2 * (k.z[1] - c[1])], stat = Math.max(Math.abs(grad[0] + k.lam), Math.abs(grad[1] + k.lam)), prim = Math.max(k.z[0] + k.z[1] - b, 0), comp = Math.abs(k.lam * (k.z[0] + k.z[1] - b));
  $("#rZ").textContent = `(${fmt(k.z[0], 3)}, ${fmt(k.z[1], 3)})`; $("#rLam").textContent = fmt(k.lam, 3); $("#rF").textContent = fmt(k.fstar, 3);
  $("#rKKT").textContent = `${stat.toExponential(0)} / ${prim.toExponential(0)} / ${comp.toExponential(0)}`; $("#rSolver").textContent = `${r.nit} it., λ = ${fmt(r.lam[0], 3)}`;
  const nb = c[0] === 1.2 && c[1] === 0.8 && b === 1, act = k.lam > 1e-9;
  caption($("#heroStatus"), `[[c = (${fmt(c[0], 2)}, ${fmt(c[1], 2)})]], [[b = ${fmt(b, 2)}]]${nb ? ", the chapter's numbers" : ""}. ` + (act ? `The unconstrained minimiser [[c]] violates [[z_1 + z_2 \\le b]] (${fmt(c[0] + c[1], 2)} > ${fmt(b, 2)}), so the constraint is active: stationarity [[2(z - c) + \\lambda a = 0]] gives [[z = c - \\tfrac{\\lambda}{2} a]], and [[a^\\top z = b]] gives [[\\lambda^\\star = a^\\top c - b = ${fmt(k.lam, 3)}]], [[z^\\star = (${fmt(k.z[0], 3)}, ${fmt(k.z[1], 3)})]], [[f^\\star = \\lambda^{\\star 2}/2 = ${fmt(k.fstar, 3)}]]. At [[z^\\star]] the objective's gradient (green) points away from the feasible side and the constraint's normal, scaled by [[\\lambda^\\star]] (orange), holds it back: the two arrows cancel, which is stationarity. Relaxing [[b]] by [[\\delta]] lowers [[f^\\star]] by [[\\lambda^\\star \\delta]] to first order: the multiplier is the price of the constraint. ` : `The unconstrained minimiser [[c]] already satisfies [[z_1 + z_2 \\le b]] (${fmt(c[0] + c[1], 2)} ≤ ${fmt(b, 2)}), so [[z^\\star = c]], [[\\lambda^\\star = 0]] by complementary slackness and [[f^\\star = 0]]: the constraint is inactive and relaxing it buys nothing. `) + `On the right, the dual function [[d(\\lambda) = -\\tfrac14 \\lVert -2c + \\lambda a \\rVert^2 - b\\lambda]] is concave; its maximum ${fmt(dualValue(c, b, k.lam), 3)} at [[\\lambda = ${fmt(k.lam, 2)}]] equals the primal optimum [[p^\\star = f^\\star - c^\\top c = ${fmt(k.objNoConst, 3)}]] (the QP drops the constant [[c^\\top c]]): strong duality, zero gap. Every other [[\\lambda \\ge 0]] gives a lower bound, which is weak duality. The course's solver, run on the same problem, returns the same [[z^\\star]] and [[\\lambda^\\star]] in ${r.nit} iterations.`);
}
["#kC1", "#kC2", "#kB"].forEach(id => $(id).addEventListener("input", drawHero));
$("#hReset").addEventListener("click", () => { $("#kC1").value = 1.2; $("#kC2").value = 0.8; $("#kB").value = 1; drawHero(); });

/* convex sets: the segment test */
const cs = { pts: [[0.25, 0.30], [1.70, 0.45]], shape: "polytope", next: 0 };
const inPoly = ([x, y]) => x >= 0 && y >= 0 && x + y <= 2.4 && x <= 2 && y <= 1.8;
const inNotch = ([x, y]) => (x >= 0 && y >= 0 && x <= 2 && y <= 0.7) || (x >= 1.3 && x <= 2 && y >= 0 && y <= 1.8);
const inside = p => cs.shape === "polytope" ? inPoly(p) : inNotch(p);
function drawSet() {
  const cv = $("#setC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.3 : 1.7, { x: [-0.3, 2.6], y: [-0.3, 2.1], xlabel: "z₁", ylabel: "z₂", xticks: [0, 1, 2], yticks: [0, 1, 2] }), { ctx, X, Y } = fr; cv._fr = fr;
  const V = cs.shape === "polytope" ? [[0, 0], [2, 0], [2, 0.4], [0.6, 1.8], [0, 1.8]] : [[0, 0], [2, 0], [2, 1.8], [1.3, 1.8], [1.3, 0.7], [0, 0.7]];
  ctx.beginPath(); V.forEach((v, i) => i ? ctx.lineTo(X(v[0]), Y(v[1])) : ctx.moveTo(X(v[0]), Y(v[1]))); ctx.closePath(); ctx.fillStyle = alpha("--post", 0.16); ctx.fill(); ctx.strokeStyle = css("--post"); ctx.lineWidth = 1.5; ctx.stroke();
  const [a, b] = cs.pts, n = 200; let out = 0; const segs = [];
  for (let i = 0; i <= n; i++) { const t = i / n, p = [t * a[0] + (1 - t) * b[0], t * a[1] + (1 - t) * b[1]]; const ok = inside(p); if (!ok) out++; segs.push([p, ok]); }
  for (let i = 1; i < segs.length; i++) { ctx.strokeStyle = segs[i][1] && segs[i - 1][1] ? css("--ink") : css("--warm"); ctx.lineWidth = 2.5; seg(ctx, X(segs[i - 1][0][0]), Y(segs[i - 1][0][1]), X(segs[i][0][0]), Y(segs[i][0][1])); }
  cs.pts.forEach((p, i) => { ctx.fillStyle = inside(p) ? css("--post") : css("--warm"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 6, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText(i ? "y" : "x", X(p[0]) + 8, Y(p[1]) - 6); });
  $$("#setW [data-shape]").forEach(bt => bt.setAttribute("aria-pressed", String(bt.dataset.shape === cs.shape)));
  const bothIn = inside(a) && inside(b);
  caption($("#setCap"), (cs.shape === "polytope" ? `The polytope [[\\{z : z \\ge 0,\\ z_1 + z_2 \\le 2.4,\\ z_1 \\le 2,\\ z_2 \\le 1.8\\}]], five half-spaces intersected. ` : `An L-shaped set, the union of two boxes. A union of convex sets is in general not convex. `) + `The segment [[\\theta x + (1 - \\theta) y]] between [[x = (${fmt(a[0], 2)}, ${fmt(a[1], 2)})]] and [[y = (${fmt(b[0], 2)}, ${fmt(b[1], 2)})]] ` + (!bothIn ? "has an endpoint outside the set, so it says nothing about convexity; convexity is about segments between points of the set. " : out === 0 ? "stays inside: of 201 sample points along it, none leaves. " : `leaves the set: ${out} of its 201 sample points lie outside (drawn orange), so this set is not convex. `) + `Tap or click to move the points, alternating [[x]] and [[y]].`);
}
$("#setC").addEventListener("click", ev => { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), fr = cv._fr; if (!fr) return; cs.pts[cs.next] = [fr.x0 + (ev.clientX - r.left - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), fr.y1 - (ev.clientY - r.top - fr.pad.t) / fr.ih * (fr.y1 - fr.y0)]; cs.next = 1 - cs.next; drawSet(); });
$$("#setW [data-shape]").forEach(bt => bt.addEventListener("click", () => { cs.shape = bt.dataset.shape; drawSet(); }));

/* a quadratic's level sets and its Hessian */
function contours(fr, F, levels, g, color) {
  const { ctx, X, Y } = fr; ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.beginPath();
  const lerp = (p, q, fp, fq, lv) => { const t = (lv - fp) / (fq - fp); return [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]; };
  for (const lv of levels) for (let i = 0; i < g.length - 1; i++) for (let j = 0; j < g.length - 1; j++) {
    const c = [[g[i], g[j], F[i][j]], [g[i + 1], g[j], F[i + 1][j]], [g[i + 1], g[j + 1], F[i + 1][j + 1]], [g[i], g[j + 1], F[i][j + 1]]], pts = [];
    for (let k = 0; k < 4; k++) { const p = c[k], q = c[(k + 1) % 4]; if ((p[2] - lv) * (q[2] - lv) < 0) pts.push(lerp(p, q, p[2], q[2], lv)); }
    if (pts.length >= 2) { ctx.moveTo(X(pts[0][0]), Y(pts[0][1])); ctx.lineTo(X(pts[1][0]), Y(pts[1][1])); if (pts.length === 4) { ctx.moveTo(X(pts[2][0]), Y(pts[2][1])); ctx.lineTo(X(pts[3][0]), Y(pts[3][1])); } }
  }
  ctx.stroke();
}
const qs = { mode: "alpha" };
function drawQuad() {
  const al = +$("#qA").value, P = qs.mode === "alpha" ? [[2, al], [al, 1]] : qs.mode === "convex" ? [[3, 0.8], [0.8, 1.5]] : [[1, 0], [0, -0.6]], [e1, e2] = eigsym2(P);
  const cv = $("#quadC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.2 : 1.6, { x: [-2, 2], y: [-2, 2], xlabel: "z₁", ylabel: "z₂", xticks: [-2, -1, 0, 1, 2], yticks: [-2, -1, 0, 1, 2] }), { ctx, X, Y } = fr;
  const g = Array.from({ length: 81 }, (_, i) => -2 + 4 * i / 80), F = g.map(a => g.map(b => 0.5 * (P[0][0] * a * a + 2 * P[0][1] * a * b + P[1][1] * b * b)));
  const fmax = Math.max(...F.flat()), fmin = Math.min(...F.flat());
  contours(fr, F, Array.from({ length: 10 }, (_, k) => fmax * (k + 1) / 11), g, css("--post"));
  if (fmin < -1e-9) contours(fr, F, Array.from({ length: 6 }, (_, k) => fmin * (k + 1) / 7), g, css("--warm"));
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(0), Y(0), 3.5, 0, 2 * Math.PI); ctx.fill();
  if (e1 > 1e-9) { const vx = P[0][1] !== 0 ? [e1 - P[1][1], P[0][1]] : (P[0][0] < P[1][1] ? [1, 0] : [0, 1]), nn = Math.hypot(...vx); ctx.strokeStyle = alpha("--ink", 0.5); ctx.setLineDash([3, 3]); seg(ctx, X(-1.8 * vx[0] / nn), Y(-1.8 * vx[1] / nn), X(1.8 * vx[0] / nn), Y(1.8 * vx[1] / nn)); ctx.setLineDash([]); }
  $("#qAV").textContent = fmt(al, 2); $("#qA").disabled = qs.mode !== "alpha"; $$("#quadW [data-q]").forEach(bt => bt.setAttribute("aria-pressed", String(bt.dataset.q === qs.mode)));
  const verdict = e1 > 1e-9 ? "positive definite: strictly convex, elliptical level sets with a single minimum at the origin" : e1 > -1e-9 ? "positive semidefinite but singular: convex, with a whole line of minimisers (the dashed direction), not strictly convex" : "indefinite: a saddle; the blue levels are positive, the orange negative, and the origin is a minimum along one direction and a maximum along the other";
  caption($("#quadCap"), `[[P = \\begin{pmatrix} ${fmt(P[0][0], 2)} & ${fmt(P[0][1], 2)} \\\\ ${fmt(P[1][0], 2)} & ${fmt(P[1][1], 2)} \\end{pmatrix}]], eigenvalues ${fmt(e1, 3)} and ${fmt(e2, 3)}: ${verdict}. ` + (qs.mode === "alpha" ? `For [[P = \\begin{pmatrix} 2 & \\alpha \\\\ \\alpha & 1 \\end{pmatrix}]] the determinant is [[2 - \\alpha^2]], so [[f]] is convex exactly when [[|\\alpha| \\le \\sqrt 2 \\approx 1.414]] (exercise 1); the trace is 3, so the eigenvalues are [[\\tfrac{3 \\pm \\sqrt{1 + 4\\alpha^2}}{2}]].` : qs.mode === "convex" ? "A convex example: both eigenvalues positive." : "An indefinite example: no amount of searching finds a minimum, because there is none; the function is unbounded below along [[z_2]]."));
}
$("#qA").addEventListener("input", () => { qs.mode = "alpha"; drawQuad(); });
$$("#quadW [data-q]").forEach(bt => bt.addEventListener("click", () => { qs.mode = bt.dataset.q; drawQuad(); }));

function drawAll() { drawHero(); drawSet(); drawQuad(); }
drawAll();
onRedraw(drawAll);
})();
