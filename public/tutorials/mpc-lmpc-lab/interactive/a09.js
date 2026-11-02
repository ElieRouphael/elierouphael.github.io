/* Advanced MPC and Learning MPC, chapter A9, Pre, Post and controllable sets: the demos.
   The hero computes the N-step controllable sets of the canonical double integrator to a small target
   box, shades the plane by the fewest steps to the target, and reports a witness input for a clickable
   probe; the Part 2 widget shows Post and Pre of one small set. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { SYS, mv, box, contains, vertices, intersect, post, predecessor, controllableSets, maximalControlInvariant, area, witnessInterval, legend, drawPoly, clickPoint } = window.MPC;
const { A, B, X: Xbox } = SYS;
const fmtVs = V => V.map(v => `(${fmt(v[0], 2)}, ${fmt(v[1], 2)})`).join(", ");

/* =====================================================================
   Hero: controllable sets and the step map
   ===================================================================== */
const hs = { probe: [3.8, -1], cache: {} };
function heroSets() {
  const umax = +$("#hU").value, tsz = +$("#hT").value, N = +$("#hN").value, key = `${umax}|${tsz}|${N}`;
  if (hs.cache[key]) return hs.cache[key];
  const target = box([-tsz, -tsz], [tsz, tsz]), Ks = controllableSets(A, B, Xbox, target, -umax, umax, N), C = maximalControlInvariant(A, B, Xbox, -umax, umax);
  return hs.cache[key] = { umax, tsz, N, target, Ks, KV: Ks.map(K => vertices(K)), C: C.set, CV: vertices(C.set) };
}
function drawHero() {
  const r = heroSets(), cv = $("#setsC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.6, { x: [-4.6, 4.6], y: [-3.5, 3.5], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, -2, -1, 0, 1, 2, 3] }), { ctx, X, Y } = fr; cv._fr = fr;
  $("#hJ").max = r.N; const j = clamp(+$("#hJ").value, 0, r.N); $("#hJ").value = j;
  drawPoly(fr, vertices(Xbox), alpha("--ink", 0.05), alpha("--ink", 0.35), 1, [3, 3]);
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  const nx = 92, ny = 69, dx = 8 / nx, dy = 6 / ny;
  for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) { const x = [-4 + (ix + 0.5) * dx, -3 + (iy + 0.5) * dy]; let k = -1; for (let q = 0; q <= j; q++) if (contains(r.Ks[q], x, 1e-9)) { k = q; break; } if (k < 0) continue; ctx.fillStyle = alpha("--post", 0.75 - 0.6 * k / Math.max(1, r.N)); ctx.fillRect(X(x[0] - dx / 2), Y(x[1] + dy / 2), X(x[0] + dx / 2) - X(x[0] - dx / 2) + 0.5, Y(x[1] - dy / 2) - Y(x[1] + dy / 2) + 0.5); }
  ctx.restore();
  for (let q = 0; q <= j; q++) drawPoly(fr, r.KV[q], null, css("--post"), q === j ? 2 : 1);
  drawPoly(fr, r.CV, null, css("--warm"), 1.5, [5, 4]);
  /* the probe, its N_min and a witness step towards the target */
  const p = hs.probe, inX = contains(Xbox, p); let nmin = -1; for (let q = 0; q < r.Ks.length; q++) if (contains(r.Ks[q], p, 1e-7)) { nmin = q; break; }
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--bg"); ctx.lineWidth = 1.5; ctx.stroke();
  let w = null, path = null;
  if (nmin > 0) {
    /* one witness path: at each step pick the input closest to zero that reaches the next inner set */
    path = [p]; let x = p;
    for (let q = nmin; q > 0; q--) { const iv = witnessInterval(x, r.Ks[q - 1], A, B, -r.umax, r.umax); if (!iv) break; if (q === nmin) w = iv; const u = clamp(0, iv[0], iv[1]); x = mv(A, x).map((v, i) => v + B[i][0] * u); path.push(x); }
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; ctx.beginPath(); path.forEach((x, i) => i ? ctx.lineTo(X(x[0]), Y(x[1])) : ctx.moveTo(X(x[0]), Y(x[1]))); ctx.stroke();
    ctx.fillStyle = css("--ink"); path.slice(1).forEach(x => { ctx.beginPath(); ctx.arc(X(x[0]), Y(x[1]), 3, 0, 2 * Math.PI); ctx.fill(); });
  }
  legend(ctx, ["shaded: N_min(x), the fewest steps to the target", "blue outlines: K_0 … K_j · dashed orange: C_∞"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#hUV").textContent = fmt(r.umax, 1); $("#hTV").textContent = fmt(r.tsz, 2); $("#hNV").textContent = r.N; $("#hJV").textContent = j;
  $("#rK").textContent = `${r.KV[r.N].length} vertices, area ${fmt(area(r.KV[r.N]), 2)}`; $("#rKj").textContent = `${r.KV[j].length} vertices, area ${fmt(area(r.KV[j]), 2)}`;
  $("#rP").textContent = !inX ? "outside X" : nmin >= 0 ? `N_min = ${nmin}` : `not in K_${r.N}`;
  const nb = r.umax === 1 && r.tsz === 0.25 && r.N === 6;
  let text = `The target [[\\mathcal K_0 = [-${fmt(r.tsz, 2)}, ${fmt(r.tsz, 2)}]^2]] and the backward recursion [[\\mathcal K_{j+1} = \\mathcal X \\cap \\mathrm{Pre}_{\\mathcal U}(\\mathcal K_j)]] with [[|u| \\le ${fmt(r.umax, 1)}]]${nb ? ", the numbers script's setting" : ""}, drawn up to [[j = ${j}]]: ${r.KV.slice(0, j + 1).map(V => V.length).join(", ")} vertices. The shading is [[N_{\\min}(x)]], the fewest steps in which [[x]] can reach the target; the sets are nested because the target is itself control invariant. `;
  text += `[[\\mathcal K_{${r.N}}]] has area ${fmt(area(r.KV[r.N]), 2)} against ${fmt(area(r.CV), 2)} for [[\\mathcal C_\\infty]] (dashed): ${area(r.KV[r.N]) > area(r.CV) - 1e-6 ? "every state that can be kept safe for ever can also reach the target in " + r.N + " steps." : "the states between the two can be kept safe for ever but cannot reach the target in " + r.N + " steps."} `;
  text += `The probe at [[(${fmt(p[0], 2)}, ${fmt(p[1], 2)})]] is ${!inX ? "outside the state box." : nmin < 0 ? `not in [[\\mathcal K_{${r.N}}]]: no admissible ${r.N}-step input sequence reaches the target from it.` : nmin === 0 ? "already in the target." : `in [[\\mathcal K_{${nmin}}]] but not in [[\\mathcal K_{${nmin - 1}}]], so it needs exactly ${nmin} step${nmin === 1 ? "" : "s"}; the first move must send it into [[\\mathcal K_{${nmin - 1}}]], which any [[u \\in [${fmt(w[0], 2)}, ${fmt(w[1], 2)}]]] does, and the black path applies the least such input at every step.`} Tap or click to move the probe.`;
  caption($("#heroStatus"), text);
}
["#hU", "#hT", "#hN", "#hJ"].forEach(id => $(id).addEventListener("input", drawHero));
$("#setsC").addEventListener("click", ev => { const p = clickPoint(ev); if (p) { hs.probe = p; drawHero(); } });
$("#hReset").addEventListener("click", () => { $("#hU").value = 1; $("#hT").value = 0.25; $("#hN").value = 6; $("#hJ").value = 6; hs.probe = [3.8, -1]; drawHero(); });

/* =====================================================================
   Part 2: Post and Pre of one small set, with a witness input
   ===================================================================== */
const pp = { probe: [-0.65, 0.45] };
function drawPrePost() {
  const umax = +$("#pU").value, s = +$("#pS").value, S = box([-0.4 * s, -0.35 * s], [0.4 * s, 0.35 * s]), Po = post(S, A, B, -umax, umax), Pr = intersect(Xbox, predecessor(S, A, B, -umax, umax)), PoV = vertices(Po), PrV = vertices(Pr);
  const cv = $("#prepostC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.7, { x: [-3, 3], y: [-2.4, 2.4], xlabel: "position p", ylabel: "velocity v", xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-2, -1, 0, 1, 2] }), { ctx, X, Y } = fr; cv._fr = fr;
  drawPoly(fr, PoV, alpha("--c5", 0.15), css("--c5"), 1.8, [5, 4]); drawPoly(fr, PrV, alpha("--post", 0.2), css("--post"), 2.2); drawPoly(fr, vertices(S), alpha("--ink", 0.1), css("--ink"), 2);
  const p = pp.probe, w = witnessInterval(p, S, A, B, -umax, umax), inS = contains(S, p), inPre = contains(Pr, p, 1e-7);
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5.5, 0, 2 * Math.PI); ctx.fill();
  if (w) { for (const u of [w[0], w[1]]) { const n = mv(A, p).map((v, i) => v + B[i][0] * u); ctx.strokeStyle = alpha("--ink", 0.5); ctx.lineWidth = 1; seg(ctx, X(p[0]), Y(p[1]), X(n[0]), Y(n[1])); ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(n[0]), Y(n[1]), 3.5, 0, 2 * Math.PI); ctx.fill(); } }
  legend(ctx, ["grey: S · blue: Pre_U(S) · dashed purple: Post(S)"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#pUV").textContent = fmt(umax, 1); $("#pSV").textContent = fmt(s, 2);
  const nb = umax === 1 && s === 1;
  caption($("#prepostCap"), `[[\\mathcal S = [-${fmt(0.4 * s, 2)}, ${fmt(0.4 * s, 2)}] \\times [-${fmt(0.35 * s, 3)}, ${fmt(0.35 * s, 3)}]]], [[|u| \\le ${fmt(umax, 1)}]]${nb ? ", the numbers script's setting" : ""}. Forward: [[\\mathrm{Post}(\\mathcal S) = A\\mathcal S \\oplus B\\,\\mathcal U]], the hull of the ${vertices(S).length * 2} points [[Av_i + Bu]] with [[u = \\pm ${fmt(umax, 1)}]], has ${PoV.length} vertices. Backward: [[\\mathrm{Pre}_{\\mathcal U}(\\mathcal S)]] by Fourier–Motzkin elimination of [[u]] has ${PrV.length} vertices, ${fmtVs(PrV)}. They are reflections of each other through the origin here, because [[\\mathcal S]] and [[\\mathcal U]] are symmetric and the map is linear, but they answer different questions. The probe at [[(${fmt(p[0], 2)}, ${fmt(p[1], 2)})]] is ${inS ? "inside [[\\mathcal S]]" : "outside [[\\mathcal S]]"} and ${inPre ? `inside the predecessor: the witness inputs are [[u \\in [${fmt(w[0], 2)}, ${fmt(w[1], 2)}]]], and the two orange points are where [[u = ${fmt(w[0], 2)}]] and [[u = ${fmt(w[1], 2)}]] send it, both in [[\\mathcal S]].` : "outside the predecessor: no admissible input sends it into [[\\mathcal S]] in one step."} Tap or click to move the probe.`);
}
["#pU", "#pS"].forEach(id => $(id).addEventListener("input", drawPrePost));
$("#prepostC").addEventListener("click", ev => { const p = clickPoint(ev); if (p) { pp.probe = p; drawPrePost(); } });

function drawAll() { drawHero(); drawPrePost(); }
drawAll();
onRedraw(drawAll);
})();
