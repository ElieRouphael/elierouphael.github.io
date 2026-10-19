/* A Tour of Machine Learning, chapter 05, K-means and k-means++: the live demos.
   Lloyd's algorithm, random and (greedy) k-means++ seeding, the silhouette score and a 3-feature
   PCA view, all computed in the browser on the notebook's 200 mall customers. The random draws
   are the browser's own, so single runs differ from the notebook's; the optimum they reach is the same.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { makeRng } = window.Stats;

/* ---------------- the data: [annual income (k$), spending score, age] ---------------- */
const ROWS = [[15,39,19],[15,81,21],[16,6,20],[16,77,23],[17,40,31],[17,76,22],[18,6,35],[18,94,23],[19,3,64],[19,72,30],[19,14,67],[19,99,35],[20,15,58],[20,77,24],[20,13,37],[20,79,22],[21,35,35],[21,66,20],[23,29,52],[23,98,35],[24,35,35],[24,73,25],[25,5,46],[25,73,31],[28,14,54],[28,82,29],[28,32,45],[28,61,35],[29,31,40],[29,87,23],[30,4,60],[30,73,21],[33,4,53],[33,92,18],[33,14,49],[33,81,21],[34,17,42],[34,73,30],[37,26,36],[37,75,20],[38,35,65],[38,92,24],[39,36,48],[39,61,31],[39,28,49],[39,65,24],[40,55,50],[40,47,27],[40,42,29],[40,42,31],[42,52,49],[42,60,33],[43,54,31],[43,60,59],[43,45,50],[43,41,47],[44,50,51],[44,46,69],[46,51,27],[46,46,53],[46,56,70],[46,55,19],[47,52,67],[47,59,54],[48,51,63],[48,59,18],[48,50,43],[48,48,68],[48,59,19],[48,47,32],[49,55,70],[49,42,47],[50,49,60],[50,56,60],[54,47,59],[54,54,26],[54,53,45],[54,48,40],[54,52,23],[54,42,49],[54,51,57],[54,55,38],[54,41,67],[54,44,46],[54,57,21],[54,46,48],[57,58,55],[57,55,22],[58,60,34],[58,46,50],[59,55,68],[59,41,18],[60,49,48],[60,40,40],[60,42,32],[60,52,24],[60,47,47],[60,50,27],[61,42,48],[61,49,20],[62,41,23],[62,48,49],[62,59,67],[62,55,26],[62,56,49],[62,42,21],[63,50,66],[63,46,54],[63,43,68],[63,48,66],[63,52,65],[63,54,19],[64,42,38],[64,46,19],[65,48,18],[65,50,19],[65,43,63],[65,59,49],[67,43,51],[67,57,50],[67,56,27],[67,40,38],[69,58,40],[69,91,39],[70,29,23],[70,77,31],[71,35,43],[71,95,40],[71,11,59],[71,75,38],[71,9,47],[71,75,39],[72,34,25],[72,71,31],[73,5,20],[73,88,29],[73,7,44],[73,73,32],[74,10,19],[74,72,35],[75,5,57],[75,93,32],[76,40,28],[76,87,32],[77,12,25],[77,97,28],[77,36,48],[77,74,32],[78,22,34],[78,90,34],[78,17,43],[78,88,39],[78,20,44],[78,76,38],[78,16,47],[78,89,27],[78,1,37],[78,78,30],[78,1,34],[78,73,30],[79,35,56],[79,83,29],[81,5,19],[81,93,31],[85,26,50],[85,75,36],[86,20,42],[86,95,33],[87,27,36],[87,63,32],[87,13,40],[87,75,28],[87,10,36],[87,92,36],[88,13,52],[88,86,30],[88,15,58],[88,69,27],[93,14,59],[93,90,35],[97,32,37],[97,86,32],[98,15,46],[98,88,29],[99,39,41],[99,97,30],[101,24,54],[101,68,28],[103,17,41],[103,85,36],[103,23,34],[103,69,32],[113,8,33],[113,91,38],[120,16,47],[120,79,35],[126,28,45],[126,74,32],[137,18,32],[137,83,30]];
const X2 = ROWS.map(r => [r[0], r[1]]), N = ROWS.length;
const OPT5 = 44448.455;                                                        // the best inertia found at k = 5 (scikit-learn, 500 restarts)

/* ---------------- k-means ---------------- */
const d2 = (a, b) => { let s = 0; for (let f = 0; f < a.length; f++) s += (a[f] - b[f]) ** 2; return s; };
const nearest = (x, C) => { let b = 0, bd = Infinity; for (let j = 0; j < C.length; j++) { const d = d2(x, C[j]); if (d < bd) { bd = d; b = j; } } return [b, bd]; };
const assign = (X, C) => X.map(x => nearest(x, C)[0]);
const inertia = (X, C, lab) => X.reduce((s, x, i) => s + d2(x, C[lab[i]]), 0);
function means(X, lab, k, C) {
  const sum = Array.from({ length: k }, () => new Float64Array(X[0].length)), cnt = new Array(k).fill(0);
  X.forEach((x, i) => { cnt[lab[i]]++; x.forEach((v, f) => { sum[lab[i]][f] += v; }); });
  return sum.map((s, j) => (cnt[j] ? Array.from(s, v => v / cnt[j]) : C[j].slice()));     // an empty cluster keeps its centre
}
function initRandom(X, k, rng) { const idx = []; while (idx.length < k) { const i = Math.floor(rng.uniform() * X.length); if (!idx.includes(i)) idx.push(i); } return idx.map(i => X[i].slice()); }
/* k-means++: each new centre drawn with probability proportional to D(x)^2; greedy = scikit-learn's
   variant, which draws 2 + floor(ln k) candidates and keeps the one that lowers the inertia most */
function initPP(X, k, rng, greedy = true, trace = null) {
  const C = [X[Math.floor(rng.uniform() * X.length)].slice()], trials = greedy ? 2 + Math.floor(Math.log(k)) : 1;
  let D = X.map(x => d2(x, C[0]));
  while (C.length < k) {
    const tot = D.reduce((a, b) => a + b, 0), cands = [];
    for (let t = 0; t < trials; t++) { let u = rng.uniform() * tot, i = 0; while (i < X.length - 1 && u >= D[i]) { u -= D[i]; i++; } cands.push(i); }
    let best = null;
    for (const i of cands) { const nd = D.map((v, q) => Math.min(v, d2(X[q], X[i]))), pot = nd.reduce((a, b) => a + b, 0); if (!best || pot < best.pot) best = { i, nd, pot }; }
    if (trace) trace.push({ D: D.slice(), tot, chosen: best.i });
    C.push(X[best.i].slice()); D = best.nd;
  }
  return C;
}
function lloyd(X, C, maxIter = 300) {
  let lab = assign(X, C), it = 0;
  for (; it < maxIter; it++) { const nc = means(X, lab, C.length, C), nl = assign(X, nc); C = nc; if (nl.every((v, i) => v === lab[i])) { lab = nl; break; } lab = nl; }
  return { C, lab, J: inertia(X, C, lab), iters: it + 1 };
}
function kmeans(X, k, init, nInit, rng) {
  let best = null;
  for (let r = 0; r < nInit; r++) { const res = lloyd(X, init === "random" ? initRandom(X, k, rng) : initPP(X, k, rng)); if (!best || res.J < best.J) best = res; }
  return best;
}
function silhouette(X, lab, k) {
  const n = X.length, D = X.map(a => X.map(b => Math.sqrt(d2(a, b)))), cnt = new Array(k).fill(0); lab.forEach(l => cnt[l]++);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const sums = new Array(k).fill(0); for (let j = 0; j < n; j++) sums[lab[j]] += D[i][j];
    if (cnt[lab[i]] <= 1) continue;
    const a = sums[lab[i]] / (cnt[lab[i]] - 1); let b = Infinity; for (let c = 0; c < k; c++) if (c !== lab[i] && cnt[c]) b = Math.min(b, sums[c] / cnt[c]);
    s += (b - a) / Math.max(a, b);
  }
  return s / n;
}
const COLS = ["--post", "--warm", "--c4", "--c5", "--prior", "#c9a227", "#1a7f4b", "#b3541e"], col = j => (COLS[j].startsWith("--") ? css(COLS[j]) : COLS[j]);
function dot(ctx, x, y, r, fill, stroke, lw = 2) { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); } }
function cross(ctx, x, y, r, c) { ctx.lineCap = "round"; [[css("--surface"), 5], [c, 2.4]].forEach(([k, w]) => { ctx.strokeStyle = k; ctx.lineWidth = w; seg(ctx, x - r, y - r, x + r, y + r); seg(ctx, x - r, y + r, x + r, y - r); }); ctx.lineCap = "butt"; }
const money = v => Math.round(v).toLocaleString("en-US");
const mallFrame = (cv, aspect) => frame(cv, aspect, { x: [10, 140], y: [-2, 102], xlabel: "annual income (k$)", ylabel: "spending score", xticks: [20, 40, 60, 80, 100, 120, 140], yticks: [0, 25, 50, 75, 100] });

/* =====================================================================
   THE DEMO: LLOYD'S ALGORITHM, STEP BY STEP
   ===================================================================== */
const hero = { k: 5, init: "k-means++", seed: 1, C: null, lab: null, phase: "assign", hist: [], done: false };
function heroSeed() {
  const rng = makeRng(hero.seed * 7919 + hero.k);
  hero.C = hero.init === "random" ? initRandom(X2, hero.k, rng) : initPP(X2, hero.k, rng);
  hero.lab = assign(X2, hero.C); hero.phase = "update"; hero.hist = [inertia(X2, hero.C, hero.lab)]; hero.done = false; hero.steps = 0;
}
function heroStep() {
  if (hero.done) return;
  if (hero.phase === "update") { hero.C = means(X2, hero.lab, hero.k, hero.C); hero.phase = "assign"; }
  else { const nl = assign(X2, hero.C); hero.done = nl.every((v, i) => v === hero.lab[i]); hero.lab = nl; hero.phase = "update"; }
  hero.steps++; hero.hist.push(inertia(X2, hero.C, hero.lab));
}
function drawHero() {
  $("#hKV").textContent = String(hero.k); $$("#demo [data-init]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.init === hero.init)));
  const cv = $("#mallC"), narrow = narrowOf(cv), fr = mallFrame(cv, narrow ? 1.1 : 1.5), { ctx, X, Y } = fr;
  const soft = j => (COLS[j].startsWith("--") ? alpha(COLS[j], 0.8) : col(j));
  X2.forEach((x, i) => dot(ctx, X(x[0]), Y(x[1]), narrow ? 3 : 3.6, soft(hero.lab[i])));
  hero.C.forEach((c, j) => cross(ctx, X(c[0]), Y(c[1]), 7, col(j)));
  const J = hero.hist[hero.hist.length - 1], jc = $("#jC"), T = Math.max(10, hero.hist.length);
  const jf = frame(jc, narrow ? 1.8 : 3, { x: [0, T], y: [0, Math.max(...hero.hist) * 1.05], xlabel: "half-steps (assign, update, ...)", ylabel: "J", yfmt: v => (v >= 1000 ? `${Math.round(v / 1000)}k` : String(Math.round(v))) });
  plotLine(jf, hero.hist.map((_, t) => t), hero.hist, { color: css("--post"), width: 2 });
  hero.hist.forEach((v, t) => dot(jf.ctx, jf.X(t), jf.Y(v), 3, css("--post")));
  if (hero.k === 5) { jf.ctx.strokeStyle = alpha("--warm", 0.8); jf.ctx.setLineDash([4, 4]); seg(jf.ctx, jf.pad.l, jf.Y(OPT5), jf.pad.l + jf.iw, jf.Y(OPT5)); jf.ctx.setLineDash([]); }
  $("#rJ").textContent = money(J); $("#rStep").textContent = `${hero.steps} (${hero.done ? "converged" : `next: ${hero.phase}`})`;
  $("#rNext").textContent = hero.done ? "nothing: no point changes cluster" : hero.phase === "update" ? "move each centre to its cluster's mean" : "reassign each point to its nearest centre";
  const sizes = new Array(hero.k).fill(0); hero.lab.forEach(l => sizes[l]++);
  let text = `<b>${hero.k} clusters</b>, ${hero.init === "random" ? "random" : "k-means++"} start, after ${hero.steps} half-step${hero.steps === 1 ? "" : "s"}: [[J = ${money(J)}]], clusters of ${sizes.join(", ")} customers. `;
  if (hero.done) text += hero.k === 5 ? (J < OPT5 + 1 ? `Converged at the optimum: <b>44,448</b>, the five segments of the notebook.` : J < OPT5 + 10 ? `Converged at <b>44,454</b>, a near-twin of the optimum 44,448: the same five segments with one customer on the other side of a border.` : `Converged at a <b>local minimum</b>: J is ${pct(J / OPT5 - 1, 0)} above the optimum 44,448. Two centres share one natural group, and another has none; no single step can fix that. Press <em>new start</em>.`) : "Converged: no point changes cluster, so the centres no longer move either.";
  else if (hero.steps === 0) text += "The centres are the starting guesses; each point is coloured by its nearest one. Press <em>one step</em> to move the centres.";
  else text += hero.hist[hero.hist.length - 1] <= hero.hist[hero.hist.length - 2] + 1e-9 ? "J fell, as it must: each half-step lowers it or leaves it alone." : "";
  caption($("#heroStatus"), text);
}
$("#hK").addEventListener("input", e => { hero.k = +e.target.value; heroSeed(); drawHero(); });
$$("#demo [data-init]").forEach(b => b.addEventListener("click", () => { hero.init = b.dataset.init; heroSeed(); drawHero(); }));
$("#hStep").addEventListener("click", () => { heroStep(); drawHero(); });
$("#hRun").addEventListener("click", () => { let g = 0; while (!hero.done && g++ < 600) heroStep(); drawHero(); });
$("#hNew").addEventListener("click", () => { hero.seed++; heroSeed(); drawHero(); });

/* =====================================================================
   PART 2: SIX POINTS ON A LINE
   ===================================================================== */
const P6 = [1, 2, 3, 8, 9, 12], line = { mu: [1, 4], phase: "assign", lab: null, hist: [] };
const lineLab = mu => P6.map(x => (Math.abs(x - mu[0]) <= Math.abs(x - mu[1]) ? 0 : 1));
const lineJ = (mu, lab) => P6.reduce((s, x, i) => s + (x - mu[lab[i]]) ** 2, 0);
function lineReset() { line.mu = [1, 4]; line.lab = lineLab(line.mu); line.phase = "update"; line.hist = [lineJ(line.mu, line.lab)]; line.done = false; }
function lineStep() {
  if (line.done) return;
  if (line.phase === "update") { line.mu = [0, 1].map(j => { const m = P6.filter((_, i) => line.lab[i] === j); return m.length ? m.reduce((a, b) => a + b, 0) / m.length : line.mu[j]; }); line.phase = "assign"; }
  else { const nl = lineLab(line.mu); line.done = nl.every((v, i) => v === line.lab[i]); line.lab = nl; line.phase = "update"; }
  line.hist.push(lineJ(line.mu, line.lab));
}
function drawLine() {
  const cv = $("#lineC"), fr = frame(cv, narrowOf(cv) ? 2.2 : 4, { x: [0, 13], y: [0, 1], yticks: false, xticks: [0, 2, 4, 6, 8, 10, 12], xlabel: "x" }), { ctx, X, Y } = fr;
  P6.forEach((x, i) => { ctx.strokeStyle = alpha(COLS[line.lab[i]], 0.5); ctx.lineWidth = 1.2; seg(ctx, X(x), Y(0.5), X(line.mu[line.lab[i]]), Y(0.5)); });
  P6.forEach((x, i) => dot(ctx, X(x), Y(0.5), 7, col(line.lab[i]), css("--surface"), 1.5));
  line.mu.forEach((m, j) => cross(ctx, X(m), Y(0.5), 8, col(j)));
  ctx.fillStyle = css("--ink-2"); ctx.font = "600 11px " + css("--font-mono"); ctx.textAlign = "center";
  line.mu.forEach((m, j) => ctx.fillText(`μ${j + 1} = ${fmt(m, 2).replace(/\.?0+$/, "")}`, X(m), Y(0.5) - 16));
  const J = line.hist[line.hist.length - 1];
  caption($("#lineCap"), `Centres [[\\mu_1 = ${fmt(line.mu[0], 2).replace(/\.?0+$/, "")}]], [[\\mu_2 = ${fmt(line.mu[1], 3).replace(/\.?0+$/, "")}]]; clusters {${P6.filter((_, i) => line.lab[i] === 0).join(", ")}} and {${P6.filter((_, i) => line.lab[i] === 1).join(", ")}}; <b>J = ${fmt(J, 2).replace(/\.?0+$/, "")}</b>. So far: ${line.hist.map(v => fmt(v, 2).replace(/\.?0+$/, "")).join(" → ")}. ` + (line.done ? "No point changes cluster: converged." : line.phase === "update" ? "Next: move each centre to the mean of its points." : "Next: reassign each point to its nearest centre."));
}
$("#lineStep").addEventListener("click", () => { lineStep(); drawLine(); });
$("#lineReset").addEventListener("click", () => { lineReset(); drawLine(); });

/* =====================================================================
   PART 3: K-MEANS++ SEEDING, ONE CENTRE AT A TIME
   ===================================================================== */
const seedW = { k: 5, seed: 3, C: [], D: null, rng: null, greedy: false, last: null };
function seedReset() { seedW.rng = makeRng(seedW.seed); seedW.C = [X2[Math.floor(seedW.rng.uniform() * N)].slice()]; seedW.D = X2.map(x => d2(x, seedW.C[0])); seedW.last = null; }
function seedNext() {
  if (seedW.C.length >= seedW.k) return;
  const tot = seedW.D.reduce((a, b) => a + b, 0), trials = seedW.greedy ? 2 + Math.floor(Math.log(seedW.k)) : 1, cands = [];
  for (let t = 0; t < trials; t++) { let u = seedW.rng.uniform() * tot, i = 0; while (i < N - 1 && u >= seedW.D[i]) { u -= seedW.D[i]; i++; } cands.push(i); }
  let best = null; for (const i of cands) { const nd = seedW.D.map((v, q) => Math.min(v, d2(X2[q], X2[i]))), pot = nd.reduce((a, b) => a + b, 0); if (!best || pot < best.pot) best = { i, nd, pot }; }
  seedW.last = { i: best.i, p: seedW.D[best.i] / tot, cands }; seedW.C.push(X2[best.i].slice()); seedW.D = best.nd;
}
function drawSeed() {
  $("#sGreedy").setAttribute("aria-pressed", String(seedW.greedy));
  const cv = $("#seedC"), fr = mallFrame(cv, narrowOf(cv) ? 1.1 : 1.6), { ctx, X, Y } = fr, tot = seedW.D.reduce((a, b) => a + b, 0), maxD = Math.max(...seedW.D);
  X2.forEach((x, i) => dot(ctx, X(x[0]), Y(x[1]), 2 + 7 * Math.sqrt(seedW.D[i] / maxD), alpha("--post", 0.15 + 0.6 * seedW.D[i] / maxD)));
  seedW.C.forEach((c, j) => cross(ctx, X(c[0]), Y(c[1]), 8, j === seedW.C.length - 1 && seedW.last ? css("--warm") : css("--ink")));
  const far = X2.map((_, i) => i).sort((a, b) => seedW.D[b] - seedW.D[a]).slice(0, 10).reduce((s, i) => s + seedW.D[i], 0) / tot;
  let text = `${seedW.C.length} of ${seedW.k} centres chosen (crosses). Each point is drawn with size and shade proportional to [[D(x)^2]], its squared distance to the nearest centre so far: that is its chance of being picked next. `;
  if (seedW.last) text += `The last centre, in orange, had probability ${pct(seedW.last.p, 1)} of being drawn${seedW.greedy ? ` (the best of ${seedW.last.cands.length} candidates)` : ""}. `;
  text += seedW.C.length < seedW.k ? `Right now the ten farthest points hold ${pct(far, 0)} of the probability between them; a uniform draw would give them ${pct(10 / N, 0)}.` : `All ${seedW.k} centres are placed, one in each natural group; Lloyd's algorithm would start from here.`;
  caption($("#seedCap"), text);
}
$("#sNext").addEventListener("click", () => { seedNext(); drawSeed(); });
$("#sNew").addEventListener("click", () => { seedW.seed++; seedReset(); drawSeed(); });
$("#sGreedy").addEventListener("click", () => { seedW.greedy = !seedW.greedy; seedReset(); drawSeed(); });

/* =====================================================================
   PART 6: THE ELBOW, AND THE SILHOUETTE
   ===================================================================== */
const elbow = { J: null, S: null, show: "J" };
function elbowCompute() {
  const rng = makeRng(42); elbow.J = {}; elbow.S = {};
  for (let k = 1; k <= 10; k++) { const r = kmeans(X2, k, "k-means++", 10, rng); elbow.J[k] = r.J; if (k >= 2 && k <= 8) elbow.S[k] = silhouette(X2, r.lab, k); }
}
function drawElbow() {
  if (!elbow.J) elbowCompute();
  $$("#elbow [data-show]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.show === elbow.show)));
  const cv = $("#elbowC"), ks = Object.keys(elbow[elbow.show]).map(Number), vs = ks.map(k => elbow[elbow.show][k]);
  const fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [0.5, 10.5], y: elbow.show === "J" ? [0, 290000] : [0, 0.65], xlabel: "k", xticks: ks, yfmt: v => (elbow.show === "J" ? `${Math.round(v / 1000)}k` : fmt(v, 1)), ylabel: elbow.show === "J" ? "inertia J" : "silhouette" }), { ctx, X, Y } = fr;
  ctx.strokeStyle = alpha("--warm", 0.8); ctx.setLineDash([4, 4]); seg(ctx, X(5), fr.pad.t, X(5), Y(fr.y0)); ctx.setLineDash([]);
  plotLine(fr, ks, vs, { color: css("--post"), width: 2.2 }); ks.forEach((k, i) => dot(ctx, X(k), Y(vs[i]), 4, css("--post")));
  const J = elbow.J, drops = [2, 3, 4, 5, 6].map(k => `${k}: ${money(J[k - 1] - J[k])}`).join(", ");
  caption($("#elbowCap"), elbow.show === "J"
    ? `Best of 10 k-means++ runs for each [[k]]. The drop in [[J]] when going to [[k]] clusters, [[k = 2, \\dots, 6]]: ${drops}. The first four steps each buy tens of thousands; the fifth buys ${money(J[5] - J[6])}: the elbow is at [[k = 5]] (dashed). [[J]] keeps falling after it, as it always does, down to 0 at [[k = n]].`
    : `The silhouette score of the same clusterings: how much closer each point is to its own cluster than to the nearest other one, averaged. It peaks at [[k = 5]] (${fmt(elbow.S[5], 3)}) and falls again after, so it does not need an eye for an elbow; it costs every pairwise distance, though.`);
}
$$("#elbow [data-show]").forEach(b => b.addEventListener("click", () => { elbow.show = b.dataset.show; drawElbow(); }));

/* =====================================================================
   PART 7: RANDOM AGAINST K-MEANS++, ONE RUN EACH
   ===================================================================== */
const race = { seed: 1, n: 100, res: null };
function raceRun() {
  const rng = makeRng(race.seed * 31 + 5), R = [], P = [], ri = [], pi = [];
  for (let s = 0; s < race.n; s++) { const a = lloyd(X2, initRandom(X2, 5, rng)), b = lloyd(X2, initPP(X2, 5, rng)); R.push(a.J); P.push(b.J); ri.push(a.iters); pi.push(b.iters); }
  race.res = { R, P, ri, pi };
}
function drawRace() {
  if (!race.res) raceRun();
  const { R, P, ri, pi } = race.res, cv = $("#raceC"), lo = 40000, hi = 100000, bins = 30;
  const fr = frame(cv, narrowOf(cv) ? 1.4 : 2.4, { x: [lo, hi], y: [0, race.n], xlabel: "inertia of a single run", xticks: [40000, 50000, 60000, 70000, 80000, 90000, 100000], xfmt: v => `${v / 1000}k`, yticks: [0, 25, 50, 75, 100], ylabel: "runs" }), { ctx, X, Y } = fr;
  const w = (hi - lo) / bins, hist = arr => { const h = new Array(bins).fill(0); arr.forEach(v => { const b = Math.min(bins - 1, Math.floor((v - lo) / w)); h[b]++; }); return h; };
  const hR = hist(R), hP = hist(P);
  hR.forEach((c, b) => { if (c) { ctx.fillStyle = alpha("--warm", 0.55); ctx.fillRect(X(lo + b * w), Y(c), (X(lo + w) - X(lo)) / 2 - 1, Y(0) - Y(c)); } });
  hP.forEach((c, b) => { if (c) { ctx.fillStyle = alpha("--post", 0.55); ctx.fillRect(X(lo + b * w) + (X(lo + w) - X(lo)) / 2, Y(c), (X(lo + w) - X(lo)) / 2 - 1, Y(0) - Y(c)); } });
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length, at = (arr, lo, hi) => arr.filter(v => v >= lo && v < hi).length;
  const optR = at(R, 0, OPT5 + 1), optP = at(P, 0, OPT5 + 1), twinR = at(R, OPT5 + 1, OPT5 + 10), twinP = at(P, OPT5 + 1, OPT5 + 10);
  caption($("#raceCap"), `${race.n} seeds, one Lloyd run each, no retries. Random start (orange): mean [[J = ${money(mean(R))}]], worst ${money(Math.max(...R))}, <b>${optR} of ${race.n}</b> runs at the optimum 44,448 (and ${twinR} at its near-twin 44,454, one customer assigned differently), ${fmt(mean(ri), 1)} iterations on average. k-means++ start (blue): mean ${money(mean(P))}, worst ${money(Math.max(...P))}, <b>${optP} of ${race.n}</b> at the optimum (${twinP} at the twin), ${fmt(mean(pi), 1)} iterations. Both distributions have two humps: the optimum and its twin, and the local minima near 66k to 92k where two centres share a group. k-means++ lands in the first hump more often, and gets there in fewer iterations; it is not a guarantee.`);
}
$("#raceAgain").addEventListener("click", () => { race.seed++; raceRun(); drawRace(); });

/* =====================================================================
   PART 8: THREE FEATURES, SEEN THROUGH PCA
   ===================================================================== */
function standardise(M) { const d = M[0].length, m = new Array(d).fill(0), s = new Array(d).fill(0); M.forEach(r => r.forEach((v, f) => { m[f] += v / M.length; })); M.forEach(r => r.forEach((v, f) => { s[f] += (v - m[f]) ** 2 / M.length; })); return M.map(r => r.map((v, f) => (v - m[f]) / Math.sqrt(s[f]))); }
/* eigen-decomposition of a small symmetric matrix by Jacobi rotations; returns eigenvalues (descending) and vectors */
function eigSym(A) {
  const n = A.length, V = A.map((_, i) => A.map((__, j) => (i === j ? 1 : 0))), M = A.map(r => r.slice());
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += M[i][j] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(M[p][q]) < 1e-30) continue;
      const th = (M[q][q] - M[p][p]) / (2 * M[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const mkp = M[k][p], mkq = M[k][q]; M[k][p] = c * mkp - s * mkq; M[k][q] = s * mkp + c * mkq; }
      for (let k = 0; k < n; k++) { const mpk = M[p][k], mqk = M[q][k]; M[p][k] = c * mpk - s * mqk; M[q][k] = s * mpk + c * mqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const order = M.map((_, i) => i).sort((a, b) => M[b][b] - M[a][a]);
  return { values: order.map(i => M[i][i]), vectors: order.map(i => V.map(r => r[i])) };
}
const pca = { res: null };
function pcaCompute() {
  const Z = standardise(ROWS), d = 3, S = Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => Z.reduce((s, z) => s + z[a] * z[b], 0) / N));
  const { values, vectors: raw } = eigSym(S);
  const vectors = raw.map(v => { const big = v.reduce((b, c, f) => (Math.abs(c) > Math.abs(v[b]) ? f : b), 0); return v[big] < 0 ? v.map(c => -c) : v; });   // a sign convention: the largest loading positive
  const proj = Z.map(z => vectors.slice(0, 2).map(v => v.reduce((s, c, f) => s + c * z[f], 0)));
  const km3 = kmeans(Z, 5, "k-means++", 10, makeRng(42)), km2 = kmeans(X2, 5, "k-means++", 10, makeRng(42));
  // adjusted Rand index between the 2-feature and 3-feature clusterings
  const ct = Array.from({ length: 5 }, () => new Array(5).fill(0)); km2.lab.forEach((a, i) => ct[a][km3.lab[i]]++);
  const c2 = x => x * (x - 1) / 2, sumIJ = ct.flat().reduce((s, v) => s + c2(v), 0), a = ct.map(r => c2(r.reduce((s, v) => s + v, 0))).reduce((s, v) => s + v, 0), b = ct[0].map((_, j) => c2(ct.reduce((s, r) => s + r[j], 0))).reduce((s, v) => s + v, 0), exp = a * b / c2(N);
  pca.res = { values, vectors, proj, km3, ari: (sumIJ - exp) / ((a + b) / 2 - exp), J3: km3.J };
}
function drawPca() {
  if (!pca.res) pcaCompute();
  const { values, vectors, proj, km3, ari } = pca.res, tot = values.reduce((s, v) => s + v, 0), cv = $("#pcaC");
  const fr = frame(cv, narrowOf(cv) ? 1.1 : 1.6, { x: [-3, 3.5], y: [-3, 3], xlabel: "PC1", ylabel: "PC2" }), { ctx, X, Y } = fr;
  proj.forEach((p, i) => dot(ctx, X(p[0]), Y(p[1]), 3.6, alpha(COLS[km3.lab[i]], 0.8)));
  const feats = ["income", "spending", "age"], load = v => v.map((c, f) => `${c < 0 ? "−" : "+"}${fmt(Math.abs(c), 2)} ${feats[f]}`).join(" ");
  caption($("#pcaCap"), `Five clusters on standardised income, spending and age, drawn on the two leading principal components, which keep <b>${pct((values[0] + values[1]) / tot, 1)}</b> of the variance (${pct(values[0] / tot, 1)} and ${pct(values[1] / tot, 1)}). PC1 = ${load(vectors[0])}: a spending-against-age axis. PC2 = ${load(vectors[1])}: income. The clusters are coloured by the 3-feature k-means, not by the 2-feature one: an adjusted Rand index of ${fmt(ari, 2)} between the two labellings says they share the high-income groups and cut the middle differently.`);
}

/* =====================================================================
   start-up
   ===================================================================== */
heroSeed(); lineReset(); seedReset();
function drawAll() { drawHero(); drawLine(); drawSeed(); drawElbow(); drawRace(); drawPca(); }
drawAll();
onRedraw(drawAll);
})();
