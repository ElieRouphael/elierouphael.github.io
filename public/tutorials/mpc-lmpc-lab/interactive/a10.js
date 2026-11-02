/* Advanced MPC and Learning MPC, chapter A10, positive and control invariance: the demos.
   The hero runs the two fixed-point iterations of the canonical double integrator, the maximal
   control-invariant set from the box and the maximal positively invariant set of the LQR loop inside
   its admissible region, with an iteration slider and a clickable probe that reports the minimum-effort
   safe input. Needs ../kit/kit.js and mpc-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { SYS, mv, dot, quad, box, contains, vertices, maximalInvariant, maximalControlInvariant, closedLoopAdmissible, area, witnessInterval, dlqr, legend, drawPoly, clickPoint } = window.MPC;
const { A, B, X: Xbox, Q } = SYS;
const fmtVs = V => V.map(v => `(${fmt(v[0], 2)}, ${fmt(v[1], 2)})`).join(", ");

/* =====================================================================
   Hero: the two fixed points
   ===================================================================== */
const hs = { mode: "C", probe: [3.8, -1], cache: {} };
const RVALS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5];
function heroSets() {
  const umax = +$("#hU").value, R = RVALS[+$("#hR").value], key = `${umax}|${R}`;
  if (hs.cache[key]) return hs.cache[key];
  const L = dlqr(A, B, Q, [[R]]), K = L.K[0], C = maximalControlInvariant(A, B, Xbox, -umax, umax), XK = closedLoopAdmissible(Xbox, K, -umax, umax), O = maximalInvariant(L.Acl, XK);
  return hs.cache[key] = { umax, R, L, K, C, XK, O, CV: vertices(C.set), OV: vertices(O.set), XKV: vertices(XK), CHV: C.history.map(h => vertices(h)), OHV: O.history.map(h => vertices(h)) };
}
function drawHero() {
  const r = heroSets(), cv = $("#setsC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.6, { x: [-4.6, 4.6], y: [-3.5, 3.5], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, -2, -1, 0, 1, 2, 3] }), { ctx, X, Y } = fr; cv._fr = fr;
  const hist = hs.mode === "C" ? r.CHV : r.OHV, last = hist.length - 1; $("#hJ").max = last; const j = clamp(+$("#hJ").value, 0, last); $("#hJ").value = j;
  drawPoly(fr, vertices(Xbox), alpha("--ink", 0.05), alpha("--ink", 0.35), 1, [3, 3]);
  if (hs.mode === "C") {
    for (let q = 0; q <= j; q++) drawPoly(fr, hist[q], q === j ? alpha("--post", 0.3) : null, css("--post"), q === j ? 2.2 : 1);
    drawPoly(fr, r.OV, null, css("--warm"), 1.5, [5, 4]);
  } else {
    drawPoly(fr, r.XKV, alpha("--c5", 0.12), css("--c5"), 1.2);
    for (let q = 0; q <= j; q++) drawPoly(fr, hist[q], q === j ? alpha("--post", 0.35) : null, css("--post"), q === j ? 2.2 : 1);
    drawPoly(fr, r.CV, null, css("--warm"), 1.5, [5, 4]);
  }
  /* the probe */
  const p = hs.probe, inX = contains(Xbox, p), inC = contains(r.C.set, p, 1e-7), inO = contains(r.O.set, p, 1e-7), inXK = contains(r.XK, p, 1e-7);
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--bg"); ctx.lineWidth = 1.5; ctx.stroke();
  const safe = inC ? witnessInterval(p, r.C.set, A, B, -r.umax, r.umax) : null, usafe = safe ? clamp(0, safe[0], safe[1]) : null;
  if (hs.mode === "C" && usafe !== null) { const nx_ = mv(A, p).map((v, i) => v + B[i][0] * usafe); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; seg(ctx, X(p[0]), Y(p[1]), X(nx_[0]), Y(nx_[1])); ctx.beginPath(); ctx.arc(X(nx_[0]), Y(nx_[1]), 3, 0, 2 * Math.PI); ctx.fillStyle = css("--ink"); ctx.fill(); }
  if (hs.mode === "O" && inX) { let x = p; ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(X(x[0]), Y(x[1])); for (let k = 0; k < 4; k++) { x = mv(r.L.Acl, x); ctx.lineTo(X(x[0]), Y(x[1])); } ctx.stroke(); }
  legend(ctx, hs.mode === "C" ? ["blue: C_j · dashed orange: O_∞ (LQR)", "black: the least safe input from the probe"] : ["purple: X_K · blue: Ω_j · dashed orange: C_∞", "black: four LQR steps from the probe"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#hUV").textContent = fmt(r.umax, 1); $("#hRV").textContent = r.R; $("#hJV").textContent = j; $("#hJlab").textContent = hs.mode === "C" ? "C" : "Ω";
  $$("#demo [data-mode]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.mode === hs.mode)));
  $("#rC").textContent = `${r.CV.length} vertices, area ${fmt(area(r.CV), 2)}, ${r.C.history.length - 1} it.`; $("#rO").textContent = `${r.OV.length} vertices, area ${fmt(area(r.OV), 2)}, ${r.O.history.length - 1} it.`;
  $("#rP").textContent = !inX ? "outside X" : (inC ? "in C_∞" : "not in C_∞") + (inO ? ", in O_∞" : "");
  const nb = r.umax === 1 && r.R === 0.2;
  let text = hs.mode === "C" ? `The fixed-point iteration [[\\mathcal C_{j+1} = \\mathcal X \\cap \\mathrm{Pre}_{\\mathcal U}(\\mathcal C_j)]] from [[\\mathcal C_0 = \\mathcal X]] with [[|u| \\le ${fmt(r.umax, 1)}]]${nb ? ", the numbers script's setting" : ""}, at [[j = ${j}]] of ${last}: ${hist.slice(0, j + 1).map(V => V.length).join(", ")} vertices, areas ${hist.slice(0, j + 1).map(V => fmt(area(V), 2)).join(", ")}. Each step removes the states from which every admissible input leaves the current candidate; the corners of the box go first, since a cart at the position limit moving outward cannot be stopped in time. ` : `The fixed LQR law [[u = -Kx]], [[K = (${fmt(r.K[0], 4)}, ${fmt(r.K[1], 4)})]] for [[R = ${r.R}]], is admissible only on [[\\mathcal X_K]] (purple, ${r.XKV.length} vertices, where [[|{-Kx}| \\le ${fmt(r.umax, 1)}]]); the iteration [[\\Omega_{j+1} = \\mathcal X_K \\cap A_{\\mathrm{cl}}^{-1}(\\Omega_j)]] at [[j = ${j}]] of ${last} has ${hist[j].length} vertices and area ${fmt(area(hist[j]), 2)}. `;
  text += `With [[|u| \\le ${fmt(r.umax, 1)}]]: [[\\mathcal C_\\infty]] has ${r.CV.length} vertices (area ${fmt(area(r.CV), 2)} of the box's 48) after ${r.C.history.length - 1} iterations, and the LQR's [[\\mathcal O_\\infty]] has ${r.OV.length} vertices (area ${fmt(area(r.OV), 2)}) after ${r.O.history.length - 1}: freedom to choose the input state by state buys a safe set ${fmt(area(r.CV) / Math.max(1e-9, area(r.OV)), 1)} times larger than committing to one law. `;
  text += `The probe at [[(${fmt(p[0], 2)}, ${fmt(p[1], 2)})]] ${!inX ? "is outside the state box." : inC ? `is in [[\\mathcal C_\\infty]]: the admissible inputs that keep it inside are [[u \\in [${fmt(safe[0], 2)}, ${fmt(safe[1], 2)}]]], the least effort is [[u = ${fmt(usafe, 3)}]]${hs.mode === "C" ? ", which the short black segment applies" : ""}.` : "satisfies the constraints now but is not in [[\\mathcal C_\\infty]]: no admissible input sequence can keep it inside them for ever."} ${inO ? "It is also in the LQR's [[\\mathcal O_\\infty]]: the fixed law keeps it safe without any choice" + (hs.mode === "O" ? ", as the black LQR steps show" : "") + "." : inXK && inX ? "It is in [[\\mathcal X_K]], so the LQR input is admissible now, but not in [[\\mathcal O_\\infty]]: " + (hs.mode === "O" ? "the black LQR steps leave the admissible region." : "the fixed law would later demand more than the motor has.") : ""} Tap or click to move the probe.`;
  caption($("#heroStatus"), text);
}
["#hU", "#hR", "#hJ"].forEach(id => $(id).addEventListener("input", drawHero));
$$("#demo [data-mode]").forEach(bt => bt.addEventListener("click", () => { hs.mode = bt.dataset.mode; $("#hJ").value = 99; drawHero(); }));
$("#setsC").addEventListener("click", ev => { const p = clickPoint(ev); if (p) { hs.probe = p; drawHero(); } });
$("#hReset").addEventListener("click", () => { $("#hU").value = 1; $("#hR").value = 2; $("#hJ").value = 99; hs.mode = "C"; hs.probe = [3.8, -1]; drawHero(); });

/* =====================================================================
   Part 4: four states, the table of memberships, recomputed live
   ===================================================================== */
function drawTable() {
  const r = heroSets();
  const states = [[3.8, -1], [3.8, 2.5], [0, 2.8], [-3.5, -2.2]];
  $("#memT").innerHTML = "<thead><tr><th>state (p, v)</th><th>in X</th><th>in C<sub>∞</sub></th><th>in O<sub>∞</sub> (LQR)</th><th>next position range</th></tr></thead><tbody>" + states.map(s => { const lo = s[0] + s[1] - 0.5 * r.umax, hi = s[0] + s[1] + 0.5 * r.umax; return `<tr><td>(${fmt(s[0], 1)}, ${fmt(s[1], 1)})</td><td class="words">${contains(Xbox, s) ? "yes" : "no"}</td><td class="words">${contains(r.C.set, s, 1e-7) ? "yes" : "no"}</td><td class="words">${contains(r.O.set, s, 1e-7) ? "yes" : "no"}</td><td>[${fmt(lo, 2)}, ${fmt(hi, 2)}]</td></tr>`; }).join("") + "</tbody>";
}

function drawAll() { drawHero(); drawTable(); }
drawAll();
onRedraw(drawAll);
["#hU", "#hR"].forEach(id => $(id).addEventListener("input", drawTable));
})();
