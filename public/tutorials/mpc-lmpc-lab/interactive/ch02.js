/* Advanced MPC and Learning MPC, chapter 02, Polytopes, reachability and invariant sets: the live demos.
   A small exact 2-D polytope toolkit, mirroring the course's polytopes.py: H- and V-representations,
   redundancy removal, intersection, affine image, linear preimage, Minkowski sum, forward reachable
   set, the controlled predecessor by Fourier–Motzkin elimination, finite-step controllable sets, and
   the maximal positively invariant and control-invariant sets by fixed-point iteration. The default
   settings reproduce the notebook's numbers. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const mv = (A, x) => A.map(r => r.reduce((s, v, k) => s + v * x[k], 0));
const tr = A => A[0].map((_, j) => A.map(r => r[j]));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const inv2 = M => { const d = M[0][0] * M[1][1] - M[0][1] * M[1][0]; return [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]]; };
const solve2 = (M, b) => mv(inv2(M), b);

/* ---- polytopes {H, h} in 2-D ---- */
const poly = (H, h) => ({ H, h });
const box = (lo, hi) => poly([[1, 0], [0, 1], [-1, 0], [0, -1]], [hi[0], hi[1], -lo[0], -lo[1]]);
const contains = (P, x, tol = 1e-8) => P.H.every((row, i) => dot(row, x) <= P.h[i] + tol);
function hull(pts) {   // Andrew's monotone chain, counter-clockwise, no collinear points
  const P = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]), cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const p of P) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 1e-12) lo.pop(); lo.push(p); }
  for (const p of P.slice().reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 1e-12) up.pop(); up.push(p); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
function vertices(P, tol = 1e-8) {   // pairwise intersection of facets, kept if feasible, hull-ordered
  const pts = [];
  for (let i = 0; i < P.H.length; i++) for (let j = i + 1; j < P.H.length; j++) {
    const det = P.H[i][0] * P.H[j][1] - P.H[i][1] * P.H[j][0]; if (Math.abs(det) < 1e-10) continue;
    const x = [(P.h[i] * P.H[j][1] - P.H[i][1] * P.h[j]) / det, (P.H[i][0] * P.h[j] - P.h[i] * P.H[j][0]) / det];
    if (contains(P, x, tol)) pts.push(x);
  }
  const uniq = []; for (const p of pts) if (!uniq.some(q => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-7)) uniq.push(p);
  return uniq.length < 3 ? uniq : hull(uniq);
}
function fromVertices(V) {   // H-representation from the hull's edges: outward normals of a counter-clockwise polygon
  const Hv = hull(V), H = [], h = [];
  for (let i = 0; i < Hv.length; i++) { const p = Hv[i], q = Hv[(i + 1) % Hv.length], n = [q[1] - p[1], -(q[0] - p[0])], len = Math.hypot(...n); H.push([n[0] / len, n[1] / len]); h.push((n[0] * p[0] + n[1] * p[1]) / len); }
  return poly(H, h);
}
function removeRedundancy(P) {   // keep a row only if some vertex touches it (and drop exact duplicates)
  const V = vertices(P); if (V.length < 3) return P;
  const H = [], h = [];
  for (let i = 0; i < P.H.length; i++) {
    const m = Math.max(...V.map(v => dot(P.H[i], v)));
    if (m > P.h[i] - 1e-7 && !H.some((row, k) => Math.abs(row[0] - P.H[i][0]) < 1e-9 && Math.abs(row[1] - P.H[i][1]) < 1e-9 && Math.abs(h[k] - P.h[i]) < 1e-9)) { H.push(P.H[i]); h.push(P.h[i]); }
  }
  return poly(H, h);
}
const intersect = (P, Q) => removeRedundancy(poly([...P.H, ...Q.H], [...P.h, ...Q.h]));
const affineImage = (P, M, c = [0, 0]) => fromVertices(vertices(P).map(v => mv(M, v).map((y, i) => y + c[i])));
const minkowski = (P, Q) => { const VP = vertices(P), VQ = vertices(Q), pts = []; for (const p of VP) for (const q of VQ) pts.push([p[0] + q[0], p[1] + q[1]]); return fromVertices(pts); };
const post = (S, A, B, umin, umax) => fromVertices(vertices(S).flatMap(x => [umin, umax].map(u => mv(A, x).map((y, i) => y + B[i][0] * u))));
const preimage = (S, A) => removeRedundancy(poly(S.H.map(row => mv(tr(A), row)), S.h.slice())); // {x : A x in S}
/* Pre_U(S) for x+ = Ax + Bu, scalar u in [umin, umax], by Fourier–Motzkin elimination of u (exactly as the course's code) */
function predecessor(S, A, B, umin, umax) {
  const ax = S.H.map(row => mv(tr(A), row)), bu = S.H.map(row => row[0] * B[0][0] + row[1] * B[1][0]), c = S.h.slice();
  ax.push([0, 0], [0, 0]); bu.push(1, -1); c.push(umax, -umin);
  const pos = [], neg = [], zer = []; bu.forEach((b, i) => (b > 1e-12 ? pos : b < -1e-12 ? neg : zer).push(i));
  const H = [], h = [];
  for (const i of zer) { H.push(ax[i]); h.push(c[i]); }
  for (const n of neg) for (const p of pos) { H.push([ax[p][0] / bu[p] - ax[n][0] / bu[n], ax[p][1] / bu[p] - ax[n][1] / bu[n]]); h.push(c[p] / bu[p] - c[n] / bu[n]); }
  return removeRedundancy(poly(H, h));
}
const approxEqual = (P, Q, tol = 1e-6) => { const V1 = vertices(P), V2 = vertices(Q); if (!V1.length || !V2.length) return V1.length === V2.length; return V1.every(v => contains(Q, v, tol)) && V2.every(v => contains(P, v, tol)); };
function maximalInvariant(A, X, maxIter = 100) { let Om = removeRedundancy(X); const history = [Om]; for (let i = 0; i < maxIter; i++) { const nxt = intersect(X, preimage(Om, A)); history.push(nxt); if (approxEqual(nxt, Om)) return { set: nxt, history }; Om = nxt; } return { set: Om, history }; }
function maximalControlInvariant(A, B, X, umin, umax, maxIter = 100) { let C = removeRedundancy(X); const history = [C]; for (let i = 0; i < maxIter; i++) { const nxt = intersect(X, predecessor(C, A, B, umin, umax)); history.push(nxt); if (approxEqual(nxt, C)) return { set: nxt, history }; C = nxt; } return { set: C, history }; }
function controllableSets(A, B, X, target, umin, umax, N) { const sets = [intersect(target, X)]; for (let i = 0; i < N; i++) sets.push(intersect(X, predecessor(sets[sets.length - 1], A, B, umin, umax))); return sets; }
const closedLoopAdmissible = (X, K, umin, umax) => intersect(X, poly([[-K[0], -K[1]], [K[0], K[1]]], [umax, -umin]));
/* the interval of inputs u in [umin, umax] with A x + B u in S, or null */
function witnessInterval(x, S, A, B, umin, umax) {
  let lo = umin, hi = umax; const Ax = mv(A, x);
  for (let i = 0; i < S.H.length; i++) { const a = S.H[i][0] * B[0][0] + S.H[i][1] * B[1][0], b = S.h[i] - dot(S.H[i], Ax); if (Math.abs(a) < 1e-12) { if (b < -1e-9) return null; } else if (a > 0) hi = Math.min(hi, b / a); else lo = Math.max(lo, b / a); }
  return lo > hi + 1e-9 ? null : [lo, hi];
}
const area = V => V.length < 3 ? 0 : Math.abs(V.reduce((s, v, i) => { const w = V[(i + 1) % V.length]; return s + v[0] * w[1] - w[0] * v[1]; }, 0)) / 2;
/* the discrete Riccati equation by iteration, u = -K x (as in chapter 00) */
function dlqr(A, B, Q, R, iters = 5000) {
  const add = (X, Y) => X.map((r, i) => r.map((v, j) => v + Y[i][j])), sub = (X, Y) => X.map((r, i) => r.map((v, j) => v - Y[i][j]));
  let P = Q;
  for (let it = 0; it < iters; it++) { const BtP = mm(tr(B), P), S = R[0][0] + mm(BtP, B)[0][0], K = mm(BtP, A).map(r => r.map(v => v / S)); const Pn = add(Q, sub(mm(tr(A), mm(P, A)), mm(tr(mm(BtP, A)), K))); const diff = Math.max(...sub(Pn, P).flat().map(Math.abs)); P = Pn; if (diff < 1e-13) break; }
  const BtP = mm(tr(B), P), S = R[0][0] + mm(BtP, B)[0][0], K = mm(BtP, A).map(r => r.map(v => v / S))[0], Acl = A.map((r, i) => r.map((v, j) => v - B[i][0] * K[j]));
  return { K, P, Acl };
}
/* NET-END */

/* ---------------- the chapter's system ---------------- */
const A = [[1, 1], [0, 1]], B = [[0.5], [1]], Xbox = box([-4, -3], [4, 3]);
const LQR = dlqr(A, B, [[1, 0], [0, 0.2]], [[0.2]]);
function legend(ctx, lines, x, y, narrow) { ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; const out = narrow ? lines.flatMap(l => l.split(" · ")) : lines; out.forEach((l, i) => ctx.fillText(l, x, y + 13 * i)); }
function drawPoly(fr, V, fill, stroke, width = 1.5, dash = null) { if (V.length < 3) return; const { ctx, X, Y } = fr; ctx.beginPath(); V.forEach((v, i) => i ? ctx.lineTo(X(v[0]), Y(v[1])) : ctx.moveTo(X(v[0]), Y(v[1]))); ctx.closePath(); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]); } }
const fmtV = V => V.map(v => `(${fmt(v[0], 2)}, ${fmt(v[1], 2)})`).join(", ");
function clickPoint(ev) { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), fr = cv._fr; if (!fr) return null; return [fr.x0 + (ev.clientX - r.left - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), fr.y1 - (ev.clientY - r.top - fr.pad.t) / fr.ih * (fr.y1 - fr.y0)]; }

/* =====================================================================
   Hero: controllable sets, the control-invariant set and the fixed-LQR invariant set
   ===================================================================== */
const hs = { mode: "K", probe: [3.8, -1], cache: {} };
function heroSets() {
  const umax = +$("#hU").value, tsz = +$("#hT").value, N = +$("#hN").value, key = `${umax}|${tsz}|${N}`;
  if (hs.cache[key]) return hs.cache[key];
  const target = box([-tsz, -tsz], [tsz, tsz]), Ks = controllableSets(A, B, Xbox, target, -umax, umax, N);
  const C = maximalControlInvariant(A, B, Xbox, -umax, umax), XK = closedLoopAdmissible(Xbox, LQR.K, -umax, umax), O = maximalInvariant(LQR.Acl, XK);
  return hs.cache[key] = { umax, tsz, N, target, Ks, C, XK, O, KV: Ks.map(K => vertices(K)), CV: vertices(C.set), OV: vertices(O.set), XKV: vertices(XK), CHV: C.history.map(h => vertices(h)), OHV: O.history.map(h => vertices(h)) };
}
function drawHero() {
  const r = heroSets(), cv = $("#setsC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.6, { x: [-4.6, 4.6], y: [-3.5, 3.5], xlabel: "position p", ylabel: "velocity v", xticks: [-4, -2, 0, 2, 4], yticks: [-3, -2, -1, 0, 1, 2, 3] }), { ctx, X, Y } = fr; cv._fr = fr;
  const j = clamp(+$("#hJ").value, 0, (hs.mode === "K" ? r.KV.length : hs.mode === "C" ? r.CHV.length : r.OHV.length) - 1); $("#hJ").max = (hs.mode === "K" ? r.KV.length : hs.mode === "C" ? r.CHV.length : r.OHV.length) - 1; $("#hJ").value = j;
  drawPoly(fr, vertices(Xbox), alpha("--ink", 0.05), alpha("--ink", 0.35), 1, [3, 3]);
  if (hs.mode === "K") {
    /* the N_min map as a background, then the nested K_j outlines */
    ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
    const nx = 92, ny = 69, dx = 8 / nx, dy = 6 / ny;
    for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) { const x = [-4 + (ix + 0.5) * dx, -3 + (iy + 0.5) * dy]; let k = -1; for (let q = 0; q <= j; q++) if (contains(r.Ks[q], x, 1e-9)) { k = q; break; } if (k < 0) continue; ctx.fillStyle = alpha("--post", 0.75 - 0.6 * k / Math.max(1, r.N)); ctx.fillRect(X(x[0] - dx / 2), Y(x[1] + dy / 2), X(x[0] + dx / 2) - X(x[0] - dx / 2) + 0.5, Y(x[1] - dy / 2) - Y(x[1] + dy / 2) + 0.5); }
    ctx.restore();
    for (let q = 0; q <= j; q++) drawPoly(fr, r.KV[q], null, css("--post"), q === j ? 2 : 1);
    drawPoly(fr, r.CV, null, css("--warm"), 1.5, [5, 4]);
  } else if (hs.mode === "C") {
    for (let q = 0; q <= j; q++) drawPoly(fr, r.CHV[q], q === j ? alpha("--post", 0.3) : null, css("--post"), q === j ? 2.2 : 1);
    drawPoly(fr, r.OV, null, css("--warm"), 1.5, [5, 4]);
  } else {
    drawPoly(fr, r.XKV, alpha("--c5", 0.12), css("--c5"), 1.2);
    for (let q = 0; q <= j; q++) drawPoly(fr, r.OHV[q], q === j ? alpha("--post", 0.35) : null, css("--post"), q === j ? 2.2 : 1);
    drawPoly(fr, r.CV, null, css("--warm"), 1.5, [5, 4]);
  }
  /* the probe */
  const p = hs.probe, inX = contains(Xbox, p), inC = contains(r.C.set, p, 1e-7), inO = contains(r.O.set, p, 1e-7); let nmin = -1; for (let q = 0; q < r.Ks.length; q++) if (contains(r.Ks[q], p, 1e-7)) { nmin = q; break; }
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--bg"); ctx.lineWidth = 1.5; ctx.stroke();
  const safe = inC ? witnessInterval(p, r.C.set, A, B, -r.umax, r.umax) : null, usafe = safe ? clamp(0, safe[0], safe[1]) : null;
  if (usafe !== null) { const nx_ = mv(A, p).map((v, i) => v + B[i][0] * usafe); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; seg(ctx, X(p[0]), Y(p[1]), X(nx_[0]), Y(nx_[1])); ctx.beginPath(); ctx.arc(X(nx_[0]), Y(nx_[1]), 3, 0, 2 * Math.PI); ctx.fillStyle = css("--ink"); ctx.fill(); }
  legend(ctx, hs.mode === "K" ? ["shaded: N_min(x) · dashed: C_∞"] : hs.mode === "C" ? ["blue: C_j · dashed: O_∞ (LQR)"] : ["purple: X_K · blue: Ω_j · dashed: C_∞"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#hUV").textContent = fmt(r.umax, 1); $("#hTV").textContent = fmt(r.tsz, 2); $("#hNV").textContent = r.N; $("#hJV").textContent = j; $("#hJlab").textContent = hs.mode === "K" ? "K" : hs.mode === "C" ? "C" : "Ω";
  $$("#demo [data-mode]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.mode === hs.mode)));
  $("#rC").textContent = `${r.CV.length} vertices, ${r.C.history.length - 1} iterations`; $("#rO").textContent = `${r.OV.length} vertices, ${r.O.history.length - 1} iterations`; $("#rK").textContent = `${r.KV[r.KV.length - 1].length} vertices, area ${fmt(area(r.KV[r.KV.length - 1]), 1)}`;
  $("#rP").textContent = !inX ? "outside X" : (nmin >= 0 ? `N_min = ${nmin}` : `not in K_${r.N}`) + (inC ? ", in C_∞" : ", not in C_∞") + (inO ? ", in O_∞" : "");
  const nb = r.umax === 1 && r.tsz === 0.25 && r.N === 6;
  let text = nb ? "The notebook's setting. " : "";
  text += hs.mode === "K" ? `The target [[\\mathcal K_0 = [-${fmt(r.tsz, 2)}, ${fmt(r.tsz, 2)}]^2]] and the backward recursion [[\\mathcal K_{j+1} = \\mathcal X \\cap \\mathrm{Pre}_{\\mathcal U}(\\mathcal K_j)]] up to [[j = ${j}]]: ${r.KV.slice(0, j + 1).map((V, q) => `${V.length}`).join(", ")} vertices. The shading is [[N_{\\min}(x)]], the fewest steps in which [[x]] can reach the target; the sets are nested because the target is itself control invariant. ` : hs.mode === "C" ? `The fixed-point iteration [[\\mathcal C_{j+1} = \\mathcal X \\cap \\mathrm{Pre}_{\\mathcal U}(\\mathcal C_j)]] from [[\\mathcal C_0 = \\mathcal X]], at [[j = ${j}]] of ${r.C.history.length - 1}: ${r.CHV.slice(0, j + 1).map(V => V.length).join(", ")} vertices. Each step removes the states from which every admissible input leaves the current candidate; the corners of the box go first, since a cart at the position limit moving outward cannot be stopped in time. ` : `The fixed LQR law [[u = -Kx]], [[K = (${fmt(LQR.K[0], 4)}, ${fmt(LQR.K[1], 4)})]], is admissible only on [[\\mathcal X_K]] (purple, where [[|{-Kx}| \\le ${fmt(r.umax, 1)}]]); the iteration [[\\Omega_{j+1} = \\mathcal X_K \\cap A_{\\mathrm{cl}}^{-1}(\\Omega_j)]] at [[j = ${j}]] of ${r.O.history.length - 1} has ${r.OHV[j].length} vertices and area ${fmt(area(r.OHV[j]), 2)}. `;
  text += `With [[|u| \\le ${fmt(r.umax, 1)}]]: [[\\mathcal C_\\infty]] has ${r.CV.length} vertices (area ${fmt(area(r.CV), 1)} of the box's 48) after ${r.C.history.length - 1} iterations, and the LQR's [[\\mathcal O_\\infty]] has ${r.OV.length} vertices (area ${fmt(area(r.OV), 2)}) after ${r.O.history.length - 1}: freedom to choose the input state by state buys a safe set ${fmt(area(r.CV) / Math.max(1e-9, area(r.OV)), 1)} times larger than committing to one law. `;
  text += `The probe at [[(${fmt(p[0], 2)}, ${fmt(p[1], 2)})]] is ${!inX ? "outside the state box." : (nmin >= 0 ? `in [[\\mathcal K_{${nmin}}]], so it can reach the target in ${nmin} step${nmin === 1 ? "" : "s"}` : `not in [[\\mathcal K_{${r.N}}]]: it cannot reach the target within ${r.N} steps`) + (inC ? `, and it is in [[\\mathcal C_\\infty]]: the minimum-effort safe input is [[u = ${fmt(usafe, 3)}]] (any [[u \\in [${fmt(safe[0], 2)}, ${fmt(safe[1], 2)}]]] keeps the next state inside), which the short black segment applies.` : ", and it is not in [[\\mathcal C_\\infty]]: it satisfies the constraints now, but no admissible input sequence can keep it inside them forever.") + (inO ? " It is also in the LQR's [[\\mathcal O_\\infty]]: the fixed law keeps it safe without any choice." : "")} Tap or click to move the probe.`;
  caption($("#heroStatus"), text);
}
["#hU", "#hT", "#hN", "#hJ"].forEach(id => $(id).addEventListener("input", drawHero));
$$("#demo [data-mode]").forEach(bt => bt.addEventListener("click", () => { hs.mode = bt.dataset.mode; $("#hJ").value = 99; drawHero(); }));
$("#setsC").addEventListener("click", ev => { const p = clickPoint(ev); if (p) { hs.probe = p; drawHero(); } });
$("#hReset").addEventListener("click", () => { $("#hU").value = 1; $("#hT").value = 0.25; $("#hN").value = 6; $("#hJ").value = 99; hs.mode = "K"; hs.probe = [3.8, -1]; drawHero(); });

/* =====================================================================
   Part 1: H- and V-representations, membership
   ===================================================================== */
const P5 = fromVertices([[-2, -1], [0.5, -1.5], [2.2, -0.2], [1.2, 1.8], [-1.4, 1.4]]), hv = { probe: [1.5, 0] };
function drawHV() {
  const cv = $("#hvC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.7, { x: [-3.2, 3.2], y: [-2.4, 2.6], xlabel: "x₁", ylabel: "x₂", xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-2, -1, 0, 1, 2] }), { ctx, X, Y } = fr; cv._fr = fr;
  const V = vertices(P5), p = hv.probe, viol = P5.H.map((row, i) => dot(row, p) - P5.h[i]);
  /* each half-space boundary line, red if violated by the probe */
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  P5.H.forEach((row, i) => { const d = [-row[1], row[0]], q = [row[0] * P5.h[i], row[1] * P5.h[i]]; ctx.strokeStyle = viol[i] > 1e-9 ? css("--warm") : alpha("--ink", 0.25); ctx.lineWidth = viol[i] > 1e-9 ? 1.8 : 1; ctx.setLineDash([4, 3]); seg(ctx, X(q[0] - 8 * d[0]), Y(q[1] - 8 * d[1]), X(q[0] + 8 * d[0]), Y(q[1] + 8 * d[1])); ctx.setLineDash([]); });
  ctx.restore();
  drawPoly(fr, V, alpha("--post", 0.18), css("--post"), 2);
  ctx.fillStyle = css("--post"); V.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(v[0]), Y(v[1]), 4, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText(`v${i + 1}`, X(v[0]) + 6, Y(v[1]) - 5); ctx.fillStyle = css("--post"); });
  const inside = contains(P5, p); ctx.fillStyle = inside ? css("--ink") : css("--warm"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5.5, 0, 2 * Math.PI); ctx.fill();
  $("#hvT").innerHTML = "<thead><tr><th>row i</th><th>H<sub>i1</sub></th><th>H<sub>i2</sub></th><th>h<sub>i</sub></th><th>H<sub>i</sub>x</th><th></th></tr></thead><tbody>" + P5.H.map((row, i) => `<tr${viol[i] > 1e-9 ? ' style="color:var(--warm)"' : ""}><td>${i + 1}</td><td>${fmt(row[0], 4)}</td><td>${fmt(row[1], 4)}</td><td>${fmt(P5.h[i], 4)}</td><td>${fmt(dot(row, p), 4)}</td><td class="words">${viol[i] > 1e-9 ? "violated" : "ok"}</td></tr>`).join("") + "</tbody>";
  caption($("#hvCap"), `The notebook's polygon: five vertices ${fmtV(V)}, or equivalently five half-spaces [[H_i^\\top x \\le h_i]] with unit normals (the dashed lines are their boundaries). Both describe the same set; a vertex is where two boundaries meet, a facet is where the set touches one boundary. The probe at [[(${fmt(p[0], 2)}, ${fmt(p[1], 2)})]] ${inside ? "satisfies all five rows, so it is inside" : `violates row${viol.filter(v => v > 1e-9).length > 1 ? "s" : ""} ${viol.map((v, i) => v > 1e-9 ? i + 1 : null).filter(v => v).join(", ")} (orange), so it is outside; one failed row is enough`}. Membership costs five inner products. Tap or click to move the probe.`);
}
$("#hvC").addEventListener("click", ev => { const p = clickPoint(ev); if (p) { hv.probe = p; drawHV(); } });

/* =====================================================================
   Part 2: set operations
   ===================================================================== */
const ops = { op: "intersect" };
function drawOps() {
  const t = +$("#oT").value, cv = $("#opsC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.7, { x: [-3.2, 3.2], y: [-2.6, 2.6], xlabel: "x₁", ylabel: "x₂", xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-2, -1, 0, 1, 2] }), { ctx } = fr;
  let text = "";
  if (ops.op === "intersect") {
    const P1 = box([-2, -1.3], [2, 1.3]), P2 = fromVertices([[-1.5 * t, 0], [0, -2 * t], [1.5 * t, 0], [0, 2 * t]]), I = intersect(P1, P2);
    drawPoly(fr, vertices(P1), null, css("--ink"), 1.2, [5, 4]); drawPoly(fr, vertices(P2), null, css("--c5"), 1.2, [2, 3]); drawPoly(fr, vertices(I), alpha("--post", 0.25), css("--post"), 2.2);
    text = `[[\\mathcal P_1 \\cap \\mathcal P_2]] in H-representation is just the stacked rows [[\\begin{pmatrix} H_1 \\\\ H_2 \\end{pmatrix} x \\le \\begin{pmatrix} h_1 \\\\ h_2 \\end{pmatrix}]], 4 + 4 = 8 rows, of which ${I.H.length} survive redundancy removal: the intersection has ${vertices(I).length} vertices, ${fmtV(vertices(I))}. The slider scales the diamond [[\\mathcal P_2]] by ${fmt(t, 2)}${t === 1 ? " (the notebook's size)" : ""}.`;
  } else if (ops.op === "affine") {
    const Xs = box([-1, -1], [1, 1]), ang = (t - 1) * 0.6, Rm = [[Math.cos(ang), -Math.sin(ang)], [Math.sin(ang), Math.cos(ang)]], M = mm(Rm, [[1.2, 0.8], [0.2, 1.4]]), c = [0.6, -0.2], Yv = vertices(affineImage(Xs, M, c));
    drawPoly(fr, vertices(Xs), null, css("--ink"), 1.2, [5, 4]); drawPoly(fr, Yv, alpha("--post", 0.25), css("--post"), 2.2);
    vertices(Xs).forEach(v => { const y = mv(M, v).map((q, i) => q + c[i]); ctx.strokeStyle = alpha("--muted", 0.6); ctx.lineWidth = 1; seg(ctx, fr.X(v[0]), fr.Y(v[1]), fr.X(y[0]), fr.Y(y[1])); });
    text = `The affine image [[M\\mathcal X + c]] of the unit square is the hull of the four mapped vertices [[Mv_i + c]]: ${fmtV(Yv)}. In V-representation the operation is immediate; the H-representation of the result is then recovered from the hull's edges. With [[M = \\begin{pmatrix} ${fmt(M[0][0], 2)} & ${fmt(M[0][1], 2)} \\\\ ${fmt(M[1][0], 2)} & ${fmt(M[1][1], 2)} \\end{pmatrix}]]${t === 1 ? ", the notebook's matrix" : ", the notebook's matrix rotated by " + fmt(ang, 2) + " rad"}, and [[c = (0.6, -0.2)]], the area is scaled by [[|\\det M| = ${fmt(Math.abs(M[0][0] * M[1][1] - M[0][1] * M[1][0]), 2)}]].`;
  } else if (ops.op === "preimage") {
    const S = box([-1, -0.7], [1, 0.7]), Ap = [[1, 0.7 * t], [0, 0.85]], Pre = preimage(S, Ap), PV = vertices(Pre);
    drawPoly(fr, vertices(S), alpha("--ink", 0.08), css("--ink"), 2); drawPoly(fr, PV, alpha("--post", 0.2), css("--post"), 2.2, [5, 4]);
    text = `The preimage [[A^{-1}(\\mathcal S) = \\{x : Ax \\in \\mathcal S\\} = \\{x : (H_S A)\\, x \\le h_S\\}]]: same right-hand sides, rows multiplied by [[A]]. No matrix is inverted. With [[A = \\begin{pmatrix} 1 & ${fmt(0.7 * t, 2)} \\\\ 0 & 0.85 \\end{pmatrix}]]${t === 1 ? " (the notebook's)" : ""} the box becomes the parallelogram ${fmtV(PV)}: states whose one autonomous step lands in [[\\mathcal S]]. The vertical stretch by [[1/0.85]] is the second row; the shear is the coupling [[A_{12}]].`;
  } else {
    const As = box([-1, -0.5], [1, 0.5]), Bs = fromVertices([[-0.4 * t, 0], [0, -0.8 * t], [0.4 * t, 0], [0, 0.8 * t]]), Sm = minkowski(As, Bs), SV = vertices(Sm);
    drawPoly(fr, vertices(As), null, css("--ink"), 1.2, [5, 4]); drawPoly(fr, vertices(Bs), null, css("--c5"), 1.2, [2, 3]); drawPoly(fr, SV, alpha("--post", 0.2), css("--post"), 2.2);
    text = `The Minkowski sum [[\\mathcal X \\oplus \\mathcal Y = \\{x + y\\}]]: all [[4 \\times 4 = 16]] pairwise vertex sums, then their hull, which keeps ${SV.length} of them: ${fmtV(SV)}. The box is pushed out by the diamond in every direction. This is how an input set [[B\\,\\mathcal U]] enlarges a reachable set, and how an additive disturbance inflates a tube. The slider scales [[\\mathcal Y]] by ${fmt(t, 2)}.`;
  }
  $("#oTV").textContent = fmt(t, 2); $$("#opsW [data-op]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.op === ops.op)));
  caption($("#opsCap"), text);
}
$("#oT").addEventListener("input", drawOps); $$("#opsW [data-op]").forEach(bt => bt.addEventListener("click", () => { ops.op = bt.dataset.op; drawOps(); }));

/* =====================================================================
   Part 3: Post and Pre, with a witness input
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
  caption($("#prepostCap"), `[[\\mathcal S = [-${fmt(0.4 * s, 2)}, ${fmt(0.4 * s, 2)}] \\times [-${fmt(0.35 * s, 3)}, ${fmt(0.35 * s, 3)}]]], [[|u| \\le ${fmt(umax, 1)}]]${nb ? ", the notebook's setting" : ""}. Forward: [[\\mathrm{Post}(\\mathcal S) = A\\mathcal S \\oplus B\\,\\mathcal U]], the hull of the ${vertices(S).length * 2} points [[Av_i + Bu]] with [[u = \\pm ${fmt(umax, 1)}]], has ${PoV.length} vertices. Backward: [[\\mathrm{Pre}_{\\mathcal U}(\\mathcal S)]] by Fourier–Motzkin elimination of [[u]] has ${PrV.length} vertices, ${fmtV(PrV)}. They are reflections of each other through the origin here, because [[\\mathcal S]] and [[\\mathcal U]] are symmetric and the map is linear, but they answer different questions. The probe at [[(${fmt(p[0], 2)}, ${fmt(p[1], 2)})]] is ${inS ? "inside [[\\mathcal S]]" : "outside [[\\mathcal S]]"} and ${inPre ? `inside the predecessor: the witness inputs are [[u \\in [${fmt(w[0], 2)}, ${fmt(w[1], 2)}]]], and the two orange points are where [[u = ${fmt(w[0], 2)}]] and [[u = ${fmt(w[1], 2)}]] send it, both in [[\\mathcal S]]${nb && Math.abs(p[0] + 0.65) < 1e-9 && Math.abs(p[1] - 0.45) < 1e-9 ? "; the notebook's LP picked " + fmt(w[1], 2) + ", the upper end" : ""}.` : "outside the predecessor: no admissible input sends it into [[\\mathcal S]] in one step."} Tap or click to move the probe.`);
}
["#pU", "#pS"].forEach(id => $(id).addEventListener("input", drawPrePost));
$("#prepostC").addEventListener("click", ev => { const p = clickPoint(ev); if (p) { pp.probe = p; drawPrePost(); } });

function drawAll() { drawHero(); drawHV(); drawOps(); drawPrePost(); }
drawAll();
onRedraw(drawAll);
})();
