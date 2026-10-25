/* Learning to Identify New Dynamical Systems, chapter 03, What many systems have in common: the
   live demos. The pretrained Family of the course (mean system, noise-whitened principal directions,
   unit-variance contexts) is rebuilt here from the notebook's 20 long-record fits and their
   covariances (ch03-data.js), with whitening, rank and the number of training systems under your
   control; the R-squared against the hidden design variables, the decoded members and their
   frequency responses are computed live. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const LOW = [0.05, 0.01, 0.05, 0.0], HIGH = [5.0, 2.0, 3.0, 5.0];
const NAMES = ["k", "d", "b", "α"];
const M_TRUE = [[0.55, 0.12], [0.07, 0.11], [0.16, -0.09], [0.30, 0.45]], EPS_SD = [0.015, 0.004, 0.008, 0.04];
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
/* Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations: { values (descending), vectors (columns) }. */
function symEig(A0) {
  const n = A0.length, A = A0.map(r => r.slice()), V = A0.map((_, i) => A0.map((__, j) => i === j ? 1 : 0));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const order = A.map((_, i) => i).sort((i, j) => A[j][j] - A[i][i]);
  return { values: order.map(i => Math.max(0, A[i][i])), vectors: V.map(row => order.map(i => row[i])) };
}
/* dynutils.Family.fit: fits = [{eta, cov}], rank, whiten. Returns the family object. Principal
   directions have an arbitrary sign; here each is oriented so that its largest entry is positive. */
function fitFamily(fits, sigmaI, { rank = 2, whiten = true } = {}) {
  const n = fits.length, etaHat = fits.map(f => f.eta);
  const mu = [0, 1, 2, 3].map(j => mean(etaHat.map(e => e[j])));
  const scale = whiten ? [0, 1, 2, 3].map(j => Math.sqrt(mean(fits.map(f => f.cov[j][j])))) : [1, 1, 1, 1];
  const centred = etaHat.map(e => e.map((v, j) => (v - mu[j]) / scale[j]));
  const C = [0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => centred.reduce((s, r) => s + r[i] * r[j], 0) / (n - 1)));
  const { values, vectors } = symEig(C);
  const total = values.reduce((s, v) => s + v, 0), explained = values.map(v => v / total);
  const dirs = [0, 1, 2, 3].map(k => { const v = vectors.map(row => row[k]); const imax = v.reduce((b, x, i) => Math.abs(x) > Math.abs(v[b]) ? i : b, 0); return v[imax] < 0 ? v.map(x => -x) : v; });
  const U = [0, 1, 2, 3].map(j => Array.from({ length: rank }, (_, k) => scale[j] * dirs[k][j] * Math.sqrt(values[k])));
  const z = centred.map(r => Array.from({ length: rank }, (_, k) => r.reduce((s, v, j) => s + v * dirs[k][j], 0) / Math.sqrt(values[k])));
  const sigma = Math.sqrt(mean(sigmaI.slice(0, n).map(s => s * s)));
  return { n, rank, whiten, mean: mu, scale, explained, values, dirs, U, z, sigma, etaHat };
}
const decode = (fam, z) => fam.mean.map((m, j) => clip(m + fam.U[j].reduce((s, u, k) => s + u * z[k], 0), LOW[j], HIGH[j]));
/* Least squares for A x = b (normal equations, tiny systems). */
function lstsq(A, b) {
  const m = A[0].length, N = Array.from({ length: m }, () => new Array(m).fill(0)), r = new Array(m).fill(0);
  A.forEach((row, i) => { for (let p = 0; p < m; p++) { r[p] += row[p] * b[i]; for (let q = 0; q < m; q++) N[p][q] += row[p] * row[q]; } });
  const Ab = N.map((row, i) => [...row, r[i]]);
  for (let c = 0; c < m; c++) {
    let p = c; for (let i = c + 1; i < m; i++) if (Math.abs(Ab[i][c]) > Math.abs(Ab[p][c])) p = i;
    [Ab[c], Ab[p]] = [Ab[p], Ab[c]];
    for (let i = 0; i < m; i++) { if (i === c) continue; const f = Ab[i][c] / Ab[c][c]; for (let j = c; j <= m; j++) Ab[i][j] -= f * Ab[c][j]; }
  }
  return Ab.map((row, i) => row[m] / row[i]);
}
/* dynutils.Family.project: best context for known coefficients, in the whitened metric. */
const project = (fam, eta) => lstsq(fam.U.map((row, j) => row.map(u => u / fam.scale[j])), eta.map((v, j) => (v - fam.mean[j]) / fam.scale[j]));
/* dynutils.r_squared: share of the variance of each column of C explained by an affine function of Z. */
function rSquared(Z, C) {
  const A = Z.map(z => [...z, 1]);
  return C[0].map((_, j) => {
    const c = C.map(r => r[j]), coef = lstsq(A, c), cm = mean(c);
    let ssr = 0, sst = 0; c.forEach((v, i) => { const p = A[i].reduce((s, a, k) => s + a * coef[k], 0); ssr += (v - p) ** 2; sst += (v - cm) ** 2; });
    return 1 - ssr / sst;
  });
}
const freqResp = (eta, om) => 0.5 * eta[2] / Math.hypot(eta[0] - om * om, eta[1] * om);
const natural = eta => ({ wn: Math.sqrt(eta[0]), f: Math.sqrt(eta[0]) / (2 * Math.PI), zeta: eta[1] / (2 * Math.sqrt(eta[0])) });
/* Genuine spread of each coefficient across the family (design variables uniform on [-1,1], so variance 1/3, plus the individual deviation). */
const trueSpread = j => Math.sqrt((M_TRUE[j][0] ** 2 + M_TRUE[j][1] ** 2) / 3 + EPS_SD[j] ** 2);
/* NET-END */

if (typeof module !== "undefined" && module.exports) {
  module.exports = { fitFamily, decode, project, rSquared, symEig, trueSpread, lstsq };
  return;
}

const D = window.FS03;
const HIDDEN = D.systems.map(s => s.context), TRUE = D.systems.map(s => s.eta);
const COLS = ["--post", "--warm", "--c4", "--c5"];
const famCache = {};
function family(n, whiten, rank = 2) { const key = `${n}-${whiten}-${rank}`; return famCache[key] || (famCache[key] = fitFamily(D.fits.slice(0, n), D.sigma_i, { rank, whiten })); }
const mix = (s, lo = "--post", hi = "--warm") => { const a = parseInt(css(lo).slice(1), 16), b = parseInt(css(hi).slice(1), 16); const ch = sh => Math.round(((a >> sh) & 255) * (1 - s) + ((b >> sh) & 255) * s); return `rgb(${ch(16)}, ${ch(8)}, ${ch(0)})`; };

/* ============================================================ hero: the learned family */
const hero = { n: 20, whiten: true, colour: 1 };
function drawHero() {
  const fam = family(hero.n, hero.whiten), Z = fam.z, H = HIDDEN.slice(0, hero.n);
  const r2 = rSquared(Z, H);
  const zAll = TRUE.map(e => project(fam, e)), r2All = rSquared(zAll, HIDDEN);
  const cv = $("#heroZ"), narrow = narrowOf(cv, 600);
  let lim = 2.6; Z.forEach(z => { lim = Math.max(lim, Math.abs(z[0]) + 0.3, Math.abs(z[1]) + 0.3); });
  const fr = frame(cv, narrow ? 1.15 : 1.35, { x: [-lim, lim], y: [-lim, lim], xlabel: "learned z₁", ylabel: "learned z₂" });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--rule"); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.ellipse(X(0), Y(0), X(2) - X(0), Y(0) - Y(2), 0, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  Z.forEach((z, i) => { ctx.fillStyle = mix((H[i][hero.colour] + 1) / 2); ctx.beginPath(); ctx.arc(X(z[0]), Y(z[1]), 5.5, 0, 2 * Math.PI); ctx.fill(); });
  const cb = $("#heroE"), frB = frame(cb, narrow ? 1.6 : 1.1, { x: [0.4, 4.6], y: [0, 1], xticks: [1, 2, 3, 4], yticks: [0, 0.5, 1], xlabel: "principal direction", ylabel: "share of variance" });
  fam.explained.forEach((e, k) => { frB.ctx.fillStyle = alpha(k < 2 ? "--post" : "--muted", 0.75); frB.ctx.fillRect(frB.X(k + 0.7), frB.Y(e), frB.X(k + 1.3) - frB.X(k + 0.7), frB.Y(0) - frB.Y(e)); });
  $("#rExp").textContent = fam.explained.map(v => fmt(100 * v, 1) + "%").join(", ");
  $("#rR2").textContent = `${fmt(r2[0], 3)}, ${fmt(r2[1], 3)}`;
  $("#rR2all").textContent = `${fmt(r2All[0], 3)}, ${fmt(r2All[1], 3)}`;
  $("#rScale").textContent = hero.whiten ? `(${fam.scale.map(v => fmt(v, 4)).join(", ")})` : "none (plain PCA)";
  $("#rSig").textContent = fmt(fam.sigma, 4);
  const isNb = hero.n === 20;
  let text = `${hero.whiten ? "Noise-whitened" : "Plain"} PCA on the first ${hero.n} training systems' estimates${isNb ? " (the notebook's family)" : ""}: the learned contexts [[z]], one point per system, coloured by the hidden design variable [[c_${hero.colour + 1}]] from blue (−1) to orange (+1), with the two-sigma circle of the unit-variance context. The directions explain ${fam.explained.slice(0, 2).map(v => fmt(100 * v, 1) + "%").join(" and ")} of the ${hero.whiten ? "noise-weighted" : "raw"} variance, ${fmt(100 * (fam.explained[0] + fam.explained[1]), 1)}% together; an affine function of [[z]] explains the hidden variables with [[R^2 = ${fmt(r2[0], 2)}]] for [[c_1]] and [[${fmt(r2[1], 2)}]] for [[c_2]]${isNb ? (hero.whiten ? ", the notebook's 0.99 and 0.96" : ", the notebook's 0.98 and 0.22") : ""}. `;
  if (!hero.whiten) text += `Without whitening the first direction is almost pure [[\\alpha]] (loading ${fmt(fam.dirs[0][3], 2)}): PCA has spent it on the estimation scatter of the one coefficient chapter 02 could not identify, and the second design variable is lost. `;
  if (hero.n < 20) text += `With ${hero.n} systems the in-sample [[R^2]] is not informative (an affine map with three parameters is fitted to ${hero.n} points); the fairer test projects all 20 true coefficient vectors through this family's subspace, which gives [[R^2 = ${fmt(r2All[0], 3)}]] and [[${fmt(r2All[1], 3)}]]. `;
  else text += `Projecting the 20 true coefficient vectors onto the learned subspace gives [[R^2 = ${fmt(r2All[0], 3)}]] and [[${fmt(r2All[1], 3)}]]: the subspace is right; what limits the in-sample figures is the scatter of the estimates themselves. `;
  text += `A colour gradient along one axis of the cloud means [[z]] is a rotated, rescaled copy of [[c]]: PCA cannot know the names of the hidden variables, and does not need to.`;
  caption($("#heroStatus"), text);
}
$("#hN").addEventListener("input", e => { hero.n = +e.target.value; $("#hNV").textContent = hero.n; drawHero(); });
$("#hWhiten").addEventListener("click", () => { hero.whiten = !hero.whiten; $("#hWhiten").setAttribute("aria-pressed", hero.whiten); $("#hWhiten").textContent = hero.whiten ? "whitening on" : "whitening off"; drawHero(); });
$$("#heroCol .btn").forEach(b => b.addEventListener("click", () => { $$("#heroCol .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); hero.colour = +b.dataset.c; drawHero(); }));
$("#hNV").textContent = hero.n;

/* ============================================================ estimated against true */
function drawEst() {
  const fam = family(20, true);
  for (let j = 0; j < 4; j++) {
    const cv = $("#estC" + j), xs = TRUE.map(e => e[j]), ys = fam.etaHat.map(e => e[j]);
    const lo = Math.min(...xs, ...ys), hi = Math.max(...xs, ...ys), pad = 0.08 * (hi - lo);
    const fr = frame(cv, 1.1, { x: [lo - pad, hi + pad], y: [lo - pad, hi + pad], xlabel: "true " + NAMES[j], ylabel: j === 0 ? "estimated" : "", pad: { l: 36 } });
    plotLine(fr, [lo - pad, hi + pad], [lo - pad, hi + pad], { color: css("--muted"), width: 1, dash: [3, 3] });
    fr.ctx.fillStyle = alpha(COLS[j], 0.8); xs.forEach((x, i) => { fr.ctx.beginPath(); fr.ctx.arc(fr.X(x), fr.Y(ys[i]), 3.5, 0, 2 * Math.PI); fr.ctx.fill(); });
  }
  const rel = [0, 1, 2, 3].map(j => { const r = TRUE.map((e, i) => Math.abs(fam.etaHat[i][j] - e[j]) / e[j]).sort((a, b) => a - b); return [0.5 * (r[9] + r[10]), r[19]]; });
  caption($("#estCap"), `The 20 training systems identified from 80 s records each, estimate against truth. Median relative errors ${NAMES.map((n, j) => `${n} ${fmt(100 * rel[j][0], 1)}%`).join(", ")}; the worst [[\\alpha]] is ${fmt(100 * rel[3][1], 0)}% off. Stiffness, damping and gain are identified well from long records; the hardening term only roughly, as chapter 02 predicted, with an average estimation standard deviation of ${fmt(fam.scale[3], 3)} against ${fmt(fam.scale[0], 4)} for [[k]]. Pretraining inherits these imperfections, and the next widget shows what that does to a naive PCA.`);
}

/* ============================================================ why whiten: the (k, alpha) plane */
const wh = { units: "physical" };
function drawWhy() {
  const fam = family(20, true), nai = family(20, false);
  const cv = $("#whyC"), narrow = narrowOf(cv);
  const phys = wh.units === "physical";
  const pts = fam.etaHat.map(e => [e[0] - fam.mean[0], e[3] - fam.mean[3]].map((v, i) => phys ? v : v / fam.scale[i ? 3 : 0]));
  const tru = TRUE.map(e => [e[0] - fam.mean[0], e[3] - fam.mean[3]].map((v, i) => phys ? v : v / fam.scale[i ? 3 : 0]));
  const lim = Math.max(...pts.flat().map(Math.abs), ...tru.flat().map(Math.abs)) * 1.15;
  const sq = !phys;
  const fr = frame(cv, narrow ? 1.1 : 1.5, { x: [-lim, lim], y: [-lim, lim], xlabel: phys ? "k̂ − mean" : "(k̂ − mean) / s_k", ylabel: phys ? "α̂ − mean" : "(α̂ − mean) / s_α" });
  const { ctx, X, Y } = fr;
  // directions: first principal direction of each family, projected onto this plane
  const dir = (f, k) => { const v = [f.U[0][k], f.U[3][k]].map((u, i) => phys ? u : u / fam.scale[i ? 3 : 0]); const nrm = Math.hypot(...v) || 1; return v.map(x => x / nrm * lim * 0.9); };
  [[nai, "--muted", "plain PCA, direction 1"], [fam, "--warm", "whitened PCA, direction 1"]].forEach(([f, col, _]) => { const d = dir(f, 0); ctx.strokeStyle = css(col); ctx.lineWidth = 2.2; seg(ctx, X(-d[0]), Y(-d[1]), X(d[0]), Y(d[1])); });
  tru.forEach((p, i) => { ctx.strokeStyle = alpha("--ink", 0.25); ctx.lineWidth = 1; seg(ctx, X(p[0]), Y(p[1]), X(pts[i][0]), Y(pts[i][1])); });
  ctx.fillStyle = alpha("--ink", 0.55); tru.forEach(p => { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 2.5, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = css("--post"); pts.forEach(p => { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 4, 0, 2 * Math.PI); ctx.fill(); });
  const snr = [0, 1, 2, 3].map(j => trueSpread(j) / fam.scale[j]);
  caption($("#whyCap"), `The 20 fits in the [[(k, \\alpha)]] plane${phys ? ", physical units" : ", each axis in units of its own estimation standard deviation"}: blue dots the estimates, small dark dots the true values, a line joining each pair, and the first principal direction of plain PCA (grey) and of whitened PCA (orange). ${phys ? `In physical units [[\\alpha]]'s scatter dwarfs everything: its estimates are off by up to ${fmt(Math.max(...TRUE.map((e, i) => Math.abs(fam.etaHat[i][3] - e[3]))), 2)}, and plain PCA's first direction follows that scatter (loading ${fmt(nai.dirs[0][3], 2)} on [[\\alpha]] against ${fmt(nai.dirs[0][0], 2)} on [[k]]).` : `Divided by their estimation standard deviations, [[k]]'s genuine spread stands ${fmt(snr[0], 0)} deviations tall and [[\\alpha]]'s about ${fmt(snr[3], 1)}: in this metric the informative direction is along [[k]], and that is the direction whitened PCA ranks first.`} The genuine spread of each coefficient across the family (design variables uniform on [[[-1, 1]]] plus the individual deviation) against its estimation standard deviation gives signal-to-noise ratios of ${NAMES.map((n, j) => `${fmt(snr[j], 1)} for ${n}`).join(", ")}: three coefficients carry the family's structure clearly, the fourth barely.`);
}
$$("#whySeg .btn").forEach(b => b.addEventListener("click", () => { $$("#whySeg .btn").forEach(x => x.setAttribute("aria-pressed", x === b)); wh.units = b.dataset.u; drawWhy(); }));

/* ============================================================ decode explorer */
const dec = { z: [1.0, -0.8] };
const omegas = Array.from({ length: 300 }, (_, i) => 0.05 + (3 - 0.05) * i / 299);
function drawDec() {
  const fam = family(20, true), eta = decode(fam, dec.z), nat = natural(eta);
  const cv = $("#decZ"), narrow = narrowOf(cv, 600);
  const fr = frame(cv, narrow ? 1.1 : 1.15, { x: [-3, 3], y: [-3, 3], xlabel: "z₁", ylabel: "z₂", xticks: [-2, 0, 2], yticks: [-2, 0, 2] });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--rule"); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.ellipse(X(0), Y(0), X(2) - X(0), Y(0) - Y(2), 0, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = alpha("--post", 0.5); fam.z.forEach(z => { ctx.beginPath(); ctx.arc(X(z[0]), Y(z[1]), 3.5, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(X(dec.z[0]), Y(dec.z[1]), 7, 0, 2 * Math.PI); ctx.fill();
  cv._fr = fr;
  const cb = $("#decB"), frB = frame(cb, narrow ? 2.2 : 1.6, { x: [-0.5, 3.5], y: [0, 2], xticks: false, yticks: [0, 0.5, 1, 1.5, 2], ylabel: "coefficient (mean system dashed)" });
  frB.ctx.font = "11px " + css("--font-mono"); frB.ctx.textAlign = "center"; frB.ctx.textBaseline = "top";
  for (let j = 0; j < 4; j++) {
    frB.ctx.fillStyle = alpha(COLS[j], 0.75); frB.ctx.fillRect(frB.X(j - 0.3), frB.Y(Math.min(2, eta[j])), frB.X(j + 0.3) - frB.X(j - 0.3), frB.Y(0) - frB.Y(Math.min(2, eta[j])));
    frB.ctx.strokeStyle = css("--ink"); frB.ctx.setLineDash([3, 3]); seg(frB.ctx, frB.X(j - 0.38), frB.Y(fam.mean[j]), frB.X(j + 0.38), frB.Y(fam.mean[j])); frB.ctx.setLineDash([]);
    frB.ctx.fillStyle = css("--ink-2"); frB.ctx.fillText(NAMES[j], frB.X(j), frB.Y(0) + 5);
  }
  const cf = $("#decF"), frF = frame(cf, narrow ? 2.2 : 1.6, { x: [0, 3 / (2 * Math.PI)], y: [0, 4.5], xlabel: "frequency [Hz]", ylabel: "|q / u|", xticks: [0, 0.1, 0.2, 0.3, 0.4] });
  plotLine(frF, omegas.map(o => o / (2 * Math.PI)), omegas.map(o => freqResp(fam.mean, o)), { color: css("--ink"), width: 1.2, dash: [4, 3] });
  plotLine(frF, omegas.map(o => o / (2 * Math.PI)), omegas.map(o => freqResp(eta, o)), { color: css("--warm"), width: 2.2 });
  const raw = fam.mean.map((m, j) => m + fam.U[j][0] * dec.z[0] + fam.U[j][1] * dec.z[1]), clipped = raw.some((v, j) => v < LOW[j] || v > HIGH[j]);
  caption($("#decCap"), `[[z = (${fmt(dec.z[0], 2)}, ${fmt(dec.z[1], 2)})]] decodes to [[\\eta = \\bar\\eta + Uz = (${eta.map(v => fmt(v, 3)).join(", ")})]]${clipped ? ", clipped to the admissible box (the raw [[\\alpha]] would be " + fmt(raw[3], 2) + ")" : ""}: natural frequency ${fmt(nat.f, 3)} Hz, damping ratio ${fmt(nat.zeta, 3)}, against ${fmt(natural(fam.mean).f, 3)} Hz and ${fmt(natural(fam.mean).zeta, 3)} for the mean system (dashed). The first direction [[u_1 = (${fam.U.map(r => fmt(r[0], 3)).join(", ")})]] moves mostly [[k]] and [[\\alpha]] together, the second [[u_2 = (${fam.U.map(r => fmt(r[1], 3)).join(", ")})]] trades damping against gain. Click or drag in the plane; the dots are the 20 training contexts. Two numbers now describe a member of the family, and from chapter 04 on they are all that a new system's data has to determine.`);
}
function decPointer(ev) { const cv = $("#decZ"), fr = cv._fr; if (!fr) return; const r = cv.getBoundingClientRect(); const px = ev.clientX - r.left, py = ev.clientY - r.top; dec.z = [clamp(fr.x0 + (px - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), -3, 3), clamp(fr.y0 + (fr.pad.t + fr.ih - py) / fr.ih * (fr.y1 - fr.y0), -3, 3)]; drawDec(); }
let decDown = false;
$("#decZ").addEventListener("pointerdown", e => { decDown = true; decPointer(e); });
$("#decZ").addEventListener("pointermove", e => { if (decDown) decPointer(e); });
window.addEventListener("pointerup", () => { decDown = false; });

/* ============================================================ worked numbers */
function fillWorked() {
  const fam = family(20, true);
  $("#snrT tbody").innerHTML = [0, 1, 2, 3].map(j => `<tr><td>${NAMES[j]}</td><td>${fmt(trueSpread(j), 4)}</td><td>${fmt(fam.scale[j], 4)}</td><td>${fmt(trueSpread(j) / fam.scale[j], 1)}</td></tr>`).join("");
  const sig = Math.hypot(0.0165, 0.01);
  $("#sigHyp").textContent = fmt(sig, 4);
}

function drawAll() { drawHero(); drawEst(); drawWhy(); drawDec(); }
fillWorked();
drawAll();
onRedraw(drawAll);
})();
