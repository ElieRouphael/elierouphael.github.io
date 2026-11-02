/* Advanced MPC and Learning MPC, chapter A8, polytopes and set operations: the demos.
   The hero applies intersection, an affine image, a linear preimage and a Minkowski sum to small
   polytopes, all computed exactly in 2-D; the Part 1 widget shows one polygon in both representations
   with a membership probe. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, alpha, onRedraw, narrowOf, seg } = window.Kit;
const { mm, mv, dot, box, poly, contains, vertices, fromVertices, intersect, affineImage, preimage, minkowski, legend, drawPoly, clickPoint } = window.MPC;
const fmtVs = V => V.map(v => `(${fmt(v[0], 2)}, ${fmt(v[1], 2)})`).join(", ");

/* =====================================================================
   Hero: the four operations
   ===================================================================== */
const ops = { op: "intersect" };
function drawHero() {
  const t = +$("#oT").value, cv = $("#opsC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.7, { x: [-3.2, 3.2], y: [-2.6, 2.6], xlabel: "x₁", ylabel: "x₂", xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-2, -1, 0, 1, 2] }), { ctx } = fr;
  let text = "", rows = "";
  if (ops.op === "intersect") {
    const P1 = box([-2, -1.3], [2, 1.3]), P2 = fromVertices([[-1.5 * t, 0], [0, -2 * t], [1.5 * t, 0], [0, 2 * t]]), I = intersect(P1, P2), IV = vertices(I);
    drawPoly(fr, vertices(P1), null, css("--ink"), 1.2, [5, 4]); drawPoly(fr, vertices(P2), null, css("--c5"), 1.2, [2, 3]); drawPoly(fr, IV, alpha("--post", 0.25), css("--post"), 2.2);
    legend(ctx, ["dashed black: P₁ (box) · dotted purple: P₂ (diamond) · blue: P₁ ∩ P₂"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    rows = `rows: 4 + 4 = 8 stacked, ${I.H.length} kept`;
    text = `[[\\mathcal P_1 \\cap \\mathcal P_2]] in H-representation is the stacked system [[\\begin{pmatrix} H_1 \\\\ H_2 \\end{pmatrix} x \\le \\begin{pmatrix} h_1 \\\\ h_2 \\end{pmatrix}]]: 4 + 4 = 8 rows, of which ${I.H.length} survive redundancy removal. The intersection has ${IV.length} vertices, ${fmtVs(IV)}. The slider scales the diamond by ${fmt(t, 2)}${t === 1 ? " (the numbers script's size)" : ""}; ${t < 0.66 ? "small enough to sit inside the box, so the intersection is the diamond and all four box rows are redundant." : t > 1.6 ? "large enough to contain the box, so the intersection is the box and the diamond's rows are redundant." : "the two overlap and both contribute facets."}`;
  } else if (ops.op === "affine") {
    const Xs = box([-1, -1], [1, 1]), ang = (t - 1) * 0.6, Rm = [[Math.cos(ang), -Math.sin(ang)], [Math.sin(ang), Math.cos(ang)]], M = mm(Rm, [[1.2, 0.8], [0.2, 1.4]]), c = [0.6, -0.2], Yv = vertices(affineImage(Xs, M, c));
    drawPoly(fr, vertices(Xs), null, css("--ink"), 1.2, [5, 4]); drawPoly(fr, Yv, alpha("--post", 0.25), css("--post"), 2.2);
    vertices(Xs).forEach(v => { const y = mv(M, v).map((q, i) => q + c[i]); ctx.strokeStyle = alpha("--muted", 0.6); ctx.lineWidth = 1; seg(ctx, fr.X(v[0]), fr.Y(v[1]), fr.X(y[0]), fr.Y(y[1])); });
    legend(ctx, ["dashed: the unit square X · blue: M X + c · grey: where each vertex goes"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    rows = `|det M| = ${fmt(Math.abs(M[0][0] * M[1][1] - M[0][1] * M[1][0]), 2)}`;
    text = `The affine image [[M\\mathcal X + c]] of the unit square is the hull of the four mapped vertices [[Mv_i + c]]: ${fmtVs(Yv)}. In V-representation the operation is immediate; the H-representation of the result is then recovered from the hull's edges. With [[M = \\begin{pmatrix} ${fmt(M[0][0], 2)} & ${fmt(M[0][1], 2)} \\\\ ${fmt(M[1][0], 2)} & ${fmt(M[1][1], 2)} \\end{pmatrix}]]${t === 1 ? ", the numbers script's matrix" : ", that matrix rotated by " + fmt(ang, 2) + " rad"}, and [[c = (0.6, -0.2)]], the area is scaled by [[|\\det M| = ${fmt(Math.abs(M[0][0] * M[1][1] - M[0][1] * M[1][0]), 2)}]], rotation or not.`;
  } else if (ops.op === "preimage") {
    const S = box([-1, -0.7], [1, 0.7]), Ap = [[1, 0.7 * t], [0, 0.85]], Pre = preimage(S, Ap), PV = vertices(Pre);
    drawPoly(fr, vertices(S), alpha("--ink", 0.08), css("--ink"), 2); drawPoly(fr, PV, alpha("--post", 0.2), css("--post"), 2.2, [5, 4]);
    legend(ctx, ["grey: S · dashed blue: A⁻¹(S) = {x : A x ∈ S}"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    rows = `rows of A⁻¹(S): H_S A, ${Pre.H.length} of them`;
    text = `The preimage [[A^{-1}(\\mathcal S) = \\{x : Ax \\in \\mathcal S\\} = \\{x : (H_S A)\\, x \\le h_S\\}]]: the same right-hand sides, the rows multiplied by [[A]]. No matrix is inverted, and [[A]] need not be invertible. With [[A = \\begin{pmatrix} 1 & ${fmt(0.7 * t, 2)} \\\\ 0 & 0.85 \\end{pmatrix}]]${t === 1 ? " (the numbers script's)" : ""} the box becomes the parallelogram ${fmtVs(PV)}: the states whose one autonomous step lands in [[\\mathcal S]]. The vertical stretch by [[1/0.85]] is the second row; the shear is the coupling [[A_{12} = ${fmt(0.7 * t, 2)}]].`;
  } else {
    const As = box([-1, -0.5], [1, 0.5]), Bs = fromVertices([[-0.4 * t, 0], [0, -0.8 * t], [0.4 * t, 0], [0, 0.8 * t]]), Sm = minkowski(As, Bs), SV = vertices(Sm);
    drawPoly(fr, vertices(As), null, css("--ink"), 1.2, [5, 4]); drawPoly(fr, vertices(Bs), null, css("--c5"), 1.2, [2, 3]); drawPoly(fr, SV, alpha("--post", 0.2), css("--post"), 2.2);
    legend(ctx, ["dashed black: X · dotted purple: Y · blue: X ⊕ Y"], fr.pad.l + 4, fr.pad.t + 12, narrow);
    rows = `16 vertex sums, ${SV.length} on the hull`;
    text = `The Minkowski sum [[\\mathcal X \\oplus \\mathcal Y = \\{x + y : x \\in \\mathcal X,\\ y \\in \\mathcal Y\\}]]: all [[4 \\times 4 = 16]] pairwise vertex sums, then their hull, which keeps ${SV.length} of them: ${fmtVs(SV)}. The box is pushed out by the diamond in every direction. This is how an input set [[B\\,\\mathcal U]] enlarges a reachable set in chapter [[a09]], and how an additive disturbance inflates a tube. The slider scales [[\\mathcal Y]] by ${fmt(t, 2)}.`;
  }
  $("#oTV").textContent = fmt(t, 2); $$("#demo [data-op]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.op === ops.op)));
  $("#rOp").textContent = { intersect: "P₁ ∩ P₂", affine: "M X + c", preimage: "A⁻¹(S)", minkowski: "X ⊕ Y" }[ops.op]; $("#rRows").textContent = rows;
  caption($("#heroStatus"), text);
}
$("#oT").addEventListener("input", drawHero); $$("#demo [data-op]").forEach(bt => bt.addEventListener("click", () => { ops.op = bt.dataset.op; drawHero(); }));
$("#hReset").addEventListener("click", () => { ops.op = "intersect"; $("#oT").value = 1; drawHero(); });

/* =====================================================================
   Part 1: H- and V-representations, membership
   ===================================================================== */
const P5 = fromVertices([[-2, -1], [0.5, -1.5], [2.2, -0.2], [1.2, 1.8], [-1.4, 1.4]]), hv = { probe: [1.5, 0] };
function drawHV() {
  const cv = $("#hvC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.7, { x: [-3.2, 3.2], y: [-2.4, 2.6], xlabel: "x₁", ylabel: "x₂", xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-2, -1, 0, 1, 2] }), { ctx, X, Y } = fr; cv._fr = fr;
  const V = vertices(P5), p = hv.probe, viol = P5.H.map((row, i) => dot(row, p) - P5.h[i]);
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  P5.H.forEach((row, i) => { const d = [-row[1], row[0]], q = [row[0] * P5.h[i], row[1] * P5.h[i]]; ctx.strokeStyle = viol[i] > 1e-9 ? css("--warm") : alpha("--ink", 0.25); ctx.lineWidth = viol[i] > 1e-9 ? 1.8 : 1; ctx.setLineDash([4, 3]); seg(ctx, X(q[0] - 8 * d[0]), Y(q[1] - 8 * d[1]), X(q[0] + 8 * d[0]), Y(q[1] + 8 * d[1])); ctx.setLineDash([]); });
  ctx.restore();
  drawPoly(fr, V, alpha("--post", 0.18), css("--post"), 2);
  V.forEach((v, i) => { ctx.fillStyle = css("--post"); ctx.beginPath(); ctx.arc(X(v[0]), Y(v[1]), 4, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText(`v${i + 1}`, X(v[0]) + 6, Y(v[1]) - 5); });
  const inside = contains(P5, p); ctx.fillStyle = inside ? css("--ink") : css("--warm"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5.5, 0, 2 * Math.PI); ctx.fill();
  $("#hvT").innerHTML = "<thead><tr><th>row i</th><th>H<sub>i1</sub></th><th>H<sub>i2</sub></th><th>h<sub>i</sub></th><th>H<sub>i</sub>x</th><th></th></tr></thead><tbody>" + P5.H.map((row, i) => `<tr${viol[i] > 1e-9 ? ' style="color:var(--warm)"' : ""}><td>${i + 1}</td><td>${fmt(row[0], 4)}</td><td>${fmt(row[1], 4)}</td><td>${fmt(P5.h[i], 4)}</td><td>${fmt(dot(row, p), 4)}</td><td class="words">${viol[i] > 1e-9 ? "violated" : "ok"}</td></tr>`).join("") + "</tbody>";
  caption($("#hvCap"), `The polygon of the numbers script: five vertices ${fmtVs(V)}, or equivalently five half-spaces [[H_i^\\top x \\le h_i]] with unit normals (the dashed lines are their boundaries). Both describe the same set; a vertex is where two boundaries meet, a facet is where the set touches one boundary. The probe at [[(${fmt(p[0], 2)}, ${fmt(p[1], 2)})]] ${inside ? "satisfies all five rows, so it is inside" : `violates row${viol.filter(v => v > 1e-9).length > 1 ? "s" : ""} ${viol.map((v, i) => v > 1e-9 ? i + 1 : null).filter(v => v).join(", ")} (orange), so it is outside; one failed row is enough`}. Membership costs five inner products. Tap or click to move the probe.`);
}
$("#hvC").addEventListener("click", ev => { const p = clickPoint(ev); if (p) { hv.probe = p; drawHV(); } });

/* =====================================================================
   Part 2: a redundant row
   ===================================================================== */
function drawRedundant() {
  const b = +$("#redB").value, cv = $("#redC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.3 : 2.0, { x: [-1.6, 2.6], y: [-1.5, 1.5], xlabel: "x₁", ylabel: "x₂", xticks: [-1, 0, 1, 2], yticks: [-1, 0, 1] }), { ctx, X, Y } = fr;
  const unit = box([-1, -1], [1, 1]), P = poly([...unit.H, [1, 0]], [...unit.h, b]), kept = vertices(P);
  drawPoly(fr, kept, alpha("--post", 0.2), css("--post"), 2.2);
  ctx.strokeStyle = b >= 1 - 1e-9 ? css("--warm") : css("--c5"); ctx.lineWidth = 1.8; ctx.setLineDash([5, 4]); seg(ctx, X(b), Y(-1.5), X(b), Y(1.5)); ctx.setLineDash([]);
  const maxx = Math.max(...vertices(unit).map(v => v[0]));
  legend(ctx, ["blue: the set with the fifth row · dashed: the row x₁ ≤ b"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#redBV").textContent = fmt(b, 2);
  caption($("#redCap"), `The unit box plus a fifth row [[x_1 \\le ${fmt(b, 2)}]]. The test: maximise [[x_1]] over the other four rows, which gives ${fmt(maxx, 2)}; ${b >= maxx - 1e-9 ? `since ${fmt(maxx, 2)} \\le ${fmt(b, 2)} the fifth row never binds and is redundant: 5 rows become 4` : `since ${fmt(maxx, 2)} > ${fmt(b, 2)} the fifth row cuts the box and must be kept: the set has ${kept.length} vertices`}. One linear program per row, each over the rows still kept.`);
}
$("#redB").addEventListener("input", drawRedundant);

function drawAll() { drawHero(); drawHV(); drawRedundant(); }
drawAll();
onRedraw(drawAll);
})();
