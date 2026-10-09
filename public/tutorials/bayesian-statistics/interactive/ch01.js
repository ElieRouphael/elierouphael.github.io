/* Chapter 01, Probability and Bayes' theorem: the live demos.
   Every number in the text is from the chapter's notebook; these widgets recompute
   the same formulas in the browser. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, int, tex, fit, onRedraw } = window.Kit;

/* ratios such as likelihood ratios: 9.5, 80, 0.056, 0.20 */
const ratio = x => x >= 10 ? x.toFixed(0) : x >= 1 ? x.toFixed(1) : x.toFixed(x < 0.1 ? 3 : 2);
/* settings and tree labels: 0.1%, 1%, 8.8%, 99.9% (one decimal at most, no trailing .0) */
const pc = x => (100 * x).toFixed(1).replace(/\.0$/, "") + "%";
/* small probabilities in the factory table */
const small = v => v === 1 ? "1" : v === 0 ? "0" : v >= 1e-6 ? v.toPrecision(3) : v.toExponential(2);
/* Set a caption's HTML; [[...]] marks TeX, rendered inline with KaTeX (so fractions are stacked). */
function setCaption(el, html) {
  const parts = [];
  el.innerHTML = html.replace(/\[\[(.+?)\]\]/g, (_, t) => { parts.push(t); return '<span class="cap-tex"></span>'; });
  $$(".cap-tex", el).forEach((span, i) => tex(span, parts[i]));
}

/* =====================================================================
   THE CROWD: 10,000 people and a screening test
   ===================================================================== */
const PREVS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50];
const N = 10000;
const crowd = { prev: 0.01, sens: 0.95, spec: 0.90, onlyPos: false, random: false, c: null };

const posterior = (prev, sens, spec) => sens * prev / (sens * prev + (1 - spec) * (1 - prev));

function exactCounts() {
  const sick = Math.round(N * crowd.prev), healthy = N - sick;
  const tp = Math.round(sick * crowd.sens), fp = Math.round(healthy * (1 - crowd.spec));
  return { sick, healthy, tp, fn: sick - tp, fp, tn: healthy - fp };
}
/* Decide every person at random, as the notebook's simulation does. */
function randomCounts() {
  let tp = 0, fn = 0, fp = 0, tn = 0;
  for (let i = 0; i < N; i++) {
    if (Math.random() < crowd.prev) { if (Math.random() < crowd.sens) tp++; else fn++; }
    else if (Math.random() > crowd.spec) fp++; else tn++;
  }
  return { sick: tp + fn, healthy: fp + tn, tp, fn, fp, tn };
}
function recount() { crowd.c = crowd.random ? randomCounts() : exactCounts(); }

function drawCrowd() {
  const canvas = $("#crowd");
  const narrow = canvas.clientWidth < 520;
  const cols = narrow ? 100 : 125, rows = N / cols;
  const { ctx, w } = fit(canvas, cols / rows);
  const cell = w / cols, gap = cell >= 4 ? 1 : 0.5, sz = Math.max(0.5, cell - gap);
  const { tp, fn, fp } = crowd.c;
  const groups = [[tp, css("--sick"), 1], [fn, css("--sick-pale"), crowd.onlyPos ? 0.1 : 1],
                  [fp, css("--well"), 1], [N, css("--well-pale"), crowd.onlyPos ? 0.1 : 1]];
  let i = 0;
  for (const [count, color, alpha] of groups) {
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    const end = Math.min(N, i + count);
    for (; i < end; i++) ctx.fillRect((i % cols) * cell, Math.floor(i / cols) * cell, sz, sz);
  }
  ctx.globalAlpha = 1;
}

function crowdText() {
  const { sick, healthy, tp, fp } = crowd.c, pos = tp + fp;
  const share = pos ? tp / pos : NaN;
  if (crowd.random) {
    return `This crowd was drawn at random, person by person. <b>${int(sick)}</b> people have the condition and <b>${int(tp)}</b> of them test positive; <b>${int(fp)}</b> of the <b>${int(healthy)}</b> others test positive anyway. ` +
      `Of the <b>${int(pos)}</b> positives, <b>${int(tp)}</b> are real: <b>${pct(share)}</b>. The exact answer is ${pct(posterior(crowd.prev, crowd.sens, crowd.spec))}; draw again and the count wobbles around it.`;
  }
  return `Out of 10,000 people, <b>${int(sick)}</b> have the condition and <b>${int(tp)}</b> of them test positive. ` +
    `Of the <b>${int(healthy)}</b> who don't, <b>${int(fp)}</b> test positive anyway. ` +
    `So of the <b>${int(pos)}</b> people who test positive, only <b>${int(tp)}</b> really have it: <b>${pct(share)}</b>.`;
}

function syncCrowdUI() {
  $("#prev").value = PREVS.indexOf(crowd.prev);
  $("#sens").value = Math.round(crowd.sens * 100);
  $("#spec").value = Math.round(crowd.spec * 1000);
  $("#onlyPos").checked = crowd.onlyPos;
}
function updateCrowd() {
  $("#prevV").textContent = pc(crowd.prev);
  $("#sensV").textContent = pc(crowd.sens);
  $("#specV").textContent = pc(crowd.spec);
  $("#bigPost").textContent = pct(posterior(crowd.prev, crowd.sens, crowd.spec));
  const { sick, tp, fp } = crowd.c;
  $("#rSick").textContent = int(sick);
  $("#rPos").textContent = int(tp + fp);
  $("#rTP").textContent = `${int(tp)} (${tp + fp ? pct(tp / (tp + fp)) : "–"})`;
  $("#crowdStatus").innerHTML = crowdText();
  drawCrowd();
  drawTree();
}
function changed() { recount(); updateCrowd(); }

$("#prev").addEventListener("input", e => { crowd.prev = PREVS[+e.target.value]; changed(); });
$("#sens").addEventListener("input", e => { crowd.sens = +e.target.value / 100; changed(); });
$("#spec").addEventListener("input", e => { crowd.spec = +e.target.value / 1000; changed(); });
$("#onlyPos").addEventListener("change", e => { crowd.onlyPos = e.target.checked; updateCrowd(); });
$("#drawCrowd").addEventListener("click", () => { crowd.random = true; changed(); });
$("#exactCrowd").addEventListener("click", () => { crowd.random = false; changed(); });

const PRESETS = {
  default: { prev: 0.01, sens: 0.95, spec: 0.90 },
  rare: { prev: 0.001 },
  common: { prev: 0.30 },
  specific: { spec: 0.99 },
};
$$("[data-preset]").forEach(b => b.addEventListener("click", () => {
  Object.assign(crowd, PRESETS.default, PRESETS[b.dataset.preset], { random: false });
  syncCrowdUI();
  changed();
  $("#demo").scrollIntoView({ behavior: Kit.reduceMotion ? "auto" : "smooth", block: "start" });
}));

/* =====================================================================
   THE PROBABILITY TREE (follows the crowd's settings)
   ===================================================================== */
function drawTree() {
  const svg = $("#tree");
  const { prev: p, sens: se, spec: sp } = crowd;
  const n = x => int(N * x);
  const box = (x, y, w, h, main, sub, hl) =>
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8"${hl ? ' class="hl"' : ""}/>` +
    `<text x="${x + 10}" y="${y + 19}" style="font-size:14px;font-weight:600"${hl ? ' class="pos"' : ""}>${main}</text>` +
    `<text x="${x + 10}" y="${y + 36}" class="sub">${sub}</text>`;
  const edge = (x1, y1, x2, y2, label, dy = -5) =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>` +
    `<text x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 + dy}" text-anchor="middle" class="sub el" style="font-size:12px">${label}</text>`;
  const leaves = [
    [8, "positive", `${fmt(p, 3)} × ${fmt(se, 2)} = ${fmt(p * se, 4)}`, n(p * se), true],
    [62, "negative", `${fmt(p, 3)} × ${fmt(1 - se, 2)} = ${fmt(p * (1 - se), 4)}`, n(p * (1 - se)), false],
    [150, "positive", `${fmt(1 - p, 3)} × ${fmt(1 - sp, 3)} = ${fmt((1 - p) * (1 - sp), 4)}`, n((1 - p) * (1 - sp)), true],
    [204, "negative", `${fmt(1 - p, 3)} × ${fmt(sp, 3)} = ${fmt((1 - p) * sp, 4)}`, n((1 - p) * sp), false],
  ];
  let s = '<title id="treeTitle">Probability tree for the screening test</title>';
  s += edge(92, 125, 150, 63, pc(p));
  s += edge(92, 125, 150, 187, pc(1 - p), 16);
  s += edge(272, 63, 318, 29, pc(se));
  s += edge(272, 63, 318, 83, pc(1 - se), 14);
  s += edge(272, 187, 318, 171, pc(1 - sp));
  s += edge(272, 187, 318, 225, pc(sp), 14);
  s += box(2, 101, 90, 48, "everyone", "10,000");
  s += box(150, 39, 122, 48, "has it", `${n(p)} people`);
  s += box(150, 163, 122, 48, "doesn't", `${n(1 - p)} people`);
  for (const [y, label, sub, count, hl] of leaves) s += box(318, y, 200, 44, `${label}: ${count}`, sub, hl);
  svg.setAttribute("viewBox", "0 0 520 252");
  svg.innerHTML = s;
  const pPos = p * se + (1 - p) * (1 - sp);
  $("#treeCap").innerHTML = `Add up the two highlighted ways to test positive: <b>P(+) = ${fmt(p * se, 4)} + ${fmt((1 - p) * (1 - sp), 4)} = ${fmt(pPos, 4)}</b>, ` +
    `that is ${int(N * pPos)} of 10,000 people. The share of them on the top branch is P(D | +) = ${pct(p * se / pPos)}: that is Bayes' theorem (Part 4).`;
}

/* =====================================================================
   THE DECK OF CARDS
   ===================================================================== */
const RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
const SUITS = [["♠", "spade", false], ["♥", "heart", true], ["♦", "diamond", true], ["♣", "club", false]];
const CARDS = [];
SUITS.forEach(([sym, suit, red]) => RANKS.forEach((rank, i) => CARDS.push({ rank, sym, suit, red, value: i + 1 })));
const EVENTS = {
  kings:   { label: "a king", plural: "kings", short: "king", test: c => c.rank === "K" },
  aces:    { label: "an ace", plural: "aces", short: "ace", test: c => c.rank === "A" },
  face:    { label: "a face card (J, Q, K)", plural: "face cards", short: "face", test: c => c.value > 10 },
  hearts:  { label: "a heart", plural: "hearts", short: "heart", test: c => c.suit === "heart" },
  spades:  { label: "a spade", plural: "spades", short: "spade", test: c => c.suit === "spade" },
  red:     { label: "red (heart or diamond)", plural: "red", short: "red", test: c => c.red },
  numbers: { label: "a number card (2 to 10)", plural: "number cards", short: "number", test: c => c.value >= 2 && c.value <= 10 },
  even:    { label: "an even number (2, 4, 6, 8, 10)", plural: "even numbers", short: "even", test: c => c.value <= 10 && c.value % 2 === 0 },
};
const deck = { a: "kings", b: "hearts", view: "A" };
const grid = $("#deckGrid");
CARDS.forEach(c => {
  const el = document.createElement("span");
  el.className = "card" + (c.red ? " red" : "");
  el.textContent = c.rank + c.sym;
  grid.appendChild(el);
});
for (const sel of ["#deckA", "#deckB"]) {
  for (const [key, ev] of Object.entries(EVENTS)) {
    const o = document.createElement("option");
    o.value = key; o.textContent = ev.label;
    $(sel).appendChild(o);
  }
}

function drawDeck() {
  const A = EVENTS[deck.a], B = EVENTS[deck.b];
  const inA = CARDS.map(A.test), inB = CARDS.map(B.test);
  const nA = inA.filter(Boolean).length, nB = inB.filter(Boolean).length;
  const nAB = CARDS.filter((_, i) => inA[i] && inB[i]).length;
  const nOr = nA + nB - nAB;
  const a = `\\text{${A.short}}`, b = `\\text{${B.short}}`;
  let hit, dim = () => false, math, cap;
  const independent = Math.abs(nAB / 52 - (nA / 52) * (nB / 52)) < 1e-12;
  switch (deck.view) {
    case "A":
      hit = i => inA[i];
      math = `P(${a}) = \\frac{${nA}}{52} \\approx ${fmt(nA / 52, 3)}`;
      cap = nA === 1 ? `1 of the 52 cards is ${A.label}.` : `${nA} of the 52 cards are ${A.plural}.`;
      break;
    case "notA":
      hit = i => !inA[i];
      math = `P(${a}^{\\,c}) = 1 - \\frac{${nA}}{52} = \\frac{${52 - nA}}{52} \\approx ${fmt(1 - nA / 52, 3)}`;
      cap = `Every card that is not ${A.label}: the complement.`;
      break;
    case "and":
      hit = i => inA[i] && inB[i];
      math = `P(${a} \\text{ and } ${b}) = \\frac{${nAB}}{52} \\approx ${fmt(nAB / 52, 3)}`;
      cap = nAB === 0 ? `No card is both ${A.label} and ${B.label}: these events cannot happen together.`
        : `Compare with [[P(A)\\,P(B) = \\frac{${nA}}{52} \\times \\frac{${nB}}{52} \\approx ${fmt(nA * nB / 2704, 3)}]]: ` +
          (independent ? "the same, so the two events are independent." : "different, so the two events are not independent.");
      break;
    case "or":
      hit = i => inA[i] || inB[i];
      math = `P(${a} \\text{ or } ${b}) = \\frac{${nA} + ${nB} - ${nAB}}{52} = \\frac{${nOr}}{52} \\approx ${fmt(nOr / 52, 3)}`;
      cap = nAB ? `Adding ${nA} and ${nB} would count the ${nAB} card${nAB > 1 ? "s" : ""} in both events twice, so ${nAB > 1 ? "they are" : "it is"} taken away once.`
        : "The events don't overlap, so their probabilities simply add.";
      break;
    case "AgB":
    case "BgA": {
      const given = deck.view === "AgB";
      const [X, Y, inY, nX, nY] = given ? [A, B, inB, nA, nB] : [B, A, inA, nB, nA];
      const x = given ? a : b, y = given ? b : a;
      hit = i => inA[i] && inB[i];
      dim = i => !inY[i];
      math = `P(${x} \\mid ${y}) = \\frac{\\#(${x} \\text{ and } ${y})}{\\#${y}} = \\frac{${nAB}}{${nY}} \\approx ${fmt(nAB / nY, 3)}`;
      const pXgY = nAB / nY, pX = nX / 52;
      cap = `Knowing that the card is ${Y.label} leaves only ${nY} cards, and ${nAB} of them ${nAB === 1 ? `is ${X.label}` : `are ${X.plural}`}. ` +
        `Without that knowledge the chance was [[\\frac{${nX}}{52} \\approx ${fmt(pX, 3)}]]. ` +
        (Math.abs(pXgY - pX) < 1e-12 ? "The same: learning one tells you nothing about the other, so they are <b>independent</b>."
          : pXgY > pX ? `Higher: learning it makes ${X.label} <b>more</b> likely, so they are not independent.`
          : `Lower: learning it makes ${X.label} <b>less</b> likely, so they are not independent.`);
      break;
    }
  }
  [...grid.children].forEach((el, i) => {
    el.classList.toggle("hit", hit(i));
    el.classList.toggle("dim", dim(i));
  });
  tex($("#deckMath"), math, true);
  setCaption($("#deckCap"), cap);
  $$("#deck [data-view]").forEach(btn => btn.setAttribute("aria-pressed", btn.dataset.view === deck.view ? "true" : "false"));
}
$("#deckA").addEventListener("change", e => { deck.a = e.target.value; drawDeck(); });
$("#deckB").addEventListener("change", e => { deck.b = e.target.value; drawDeck(); });
$$("#deck [data-view]").forEach(btn => btn.addEventListener("click", () => { deck.view = btn.dataset.view; drawDeck(); }));
$$("[data-deck]").forEach(btn => btn.addEventListener("click", () => {
  [deck.a, deck.b, deck.view] = btn.dataset.deck.split(",");
  $("#deckA").value = deck.a; $("#deckB").value = deck.b;
  drawDeck();
  $("#deck").scrollIntoView({ behavior: Kit.reduceMotion ? "auto" : "smooth", block: "center" });
}));

/* =====================================================================
   REPEATED TESTS, ON THE LOG-ODDS SCALE
   ===================================================================== */
const TESTS = { 1: { sens: 0.95, spec: 0.90 }, 2: { sens: 0.80, spec: 0.99 } };
const PRIOR = 0.01, MAX_RESULTS = 12;
const seq = { items: [] };
const lr = ({ test, res }) => {
  const t = TESTS[test];
  return res === "+" ? t.sens / (1 - t.spec) : (1 - t.sens) / t.spec;
};
const toProb = lo => { const o = 10 ** lo; return o / (1 + o); };

function drawSeq() {
  const canvas = $("#seqC");
  const { ctx, w, h } = fit(canvas, canvas.clientWidth < 520 ? 1.35 : 2.3);
  const path = [Math.log10(PRIOR / (1 - PRIOR))];
  seq.items.forEach(it => path.push(path[path.length - 1] + Math.log10(lr(it))));
  const lo = Math.min(-3.3, Math.min(...path) - 0.4), hi = Math.max(3.3, Math.max(...path) + 0.4);
  const pad = { l: 52, r: 14, t: 18, b: 30 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const span = Math.max(6, seq.items.length);
  const X = i => pad.l + i / span * iw;
  const Y = v => pad.t + (hi - v) / (hi - lo) * ih;
  ctx.font = "11px " + css("--font-mono");
  ctx.textBaseline = "middle";
  for (const p of [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]) {
    const v = Math.log10(p / (1 - p));
    if (v < lo || v > hi) continue;
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(v) + 0.5); ctx.lineTo(w - pad.r, Y(v) + 0.5); ctx.stroke();
    ctx.fillStyle = css("--muted"); ctx.textAlign = "right";
    ctx.fillText(pc(p), pad.l - 6, Y(v));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
  ctx.fillStyle = css("--muted");
  ctx.fillText("start", X(0), h - 8);
  seq.items.forEach((it, i) => {
    ctx.fillStyle = it.res === "+" ? css("--lik") : css("--ink-2");
    ctx.fillText(`${it.res === "+" ? "+" : "−"}${it.test}`, X(i + 1), h - 8);
  });
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 2.2; ctx.lineJoin = "round";
  ctx.beginPath();
  path.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)));
  ctx.stroke();
  ctx.font = "600 11px " + css("--font-ui");
  path.forEach((v, i) => {
    ctx.fillStyle = css("--post"); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(i), Y(v), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    if (i === 0 || i === path.length - 1 || path.length <= 8) {
      ctx.fillStyle = css("--ink");
      ctx.fillText(pc(toProb(v)), X(i), Y(v) - 10);
    }
  });
  const odds = path.map(v => 10 ** v);
  const p = toProb(path[path.length - 1]);
  let cap;
  if (!seq.items.length) {
    cap = "Before any test: the condition affects <b>1%</b> of people, odds of 1 to 99. Add a test result.";
  } else {
    const last = seq.items[seq.items.length - 1];
    const factors = seq.items.map(it => ratio(lr(it))).join(" \\times ");
    cap = `The last result, a <b>${last.res === "+" ? "positive" : "negative"} on test ${last.test}</b>, has likelihood ratio ${ratio(lr(last))}: ` +
      `it took the odds from ${fmt(odds[odds.length - 2], 3)} to ${fmt(odds[odds.length - 1], 3)}. ` +
      `All together: [[\\frac{1}{99} \\times ${factors} = ${fmt(odds[odds.length - 1], 3)}]], a probability of <b>${pct(p)}</b>.`;
  }
  setCaption($("#seqCap"), cap);
  $$("#seq [data-test]").forEach(b => { b.disabled = seq.items.length >= MAX_RESULTS; });
}
$$("#seq [data-test]").forEach(b => b.addEventListener("click", () => {
  if (seq.items.length < MAX_RESULTS) seq.items.push({ test: +b.dataset.test, res: b.dataset.res });
  drawSeq();
}));
$("#seqUndo").addEventListener("click", () => { seq.items.pop(); drawSeq(); });
$("#seqReset").addEventListener("click", () => { seq.items = []; drawSeq(); });
$("#seqShuffle").addEventListener("click", () => {
  const a = seq.items;
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  drawSeq();
});

/* =====================================================================
   THE FACTORY: multiply, sum, divide
   ===================================================================== */
const fac = { machines: [{ name: "A", share: 50, rate: 1 }, { name: "B", share: 30, rate: 2 }, { name: "C", share: 20, rate: 5 }], bad: 0, good: 0 };
const facBody = $("#facTable tbody");
function buildFacTable() {
  facBody.innerHTML = fac.machines.map((m, i) => `<tr>
      <td>${m.name}</td>
      <td><input type="number" min="0" max="100" step="1" value="${m.share}" data-i="${i}" data-k="share" aria-label="Share of production of machine ${m.name}, in percent"> %</td>
      <td><input type="number" min="0" max="100" step="0.5" value="${m.rate}" data-i="${i}" data-k="rate" aria-label="Defect rate of machine ${m.name}, in percent"> %</td>
      <td data-c="lik"></td><td data-c="un"></td><td data-c="post"></td></tr>`).join("") +
    `<tr class="total"><td>Sum</td><td data-c="sumPrior"></td><td></td><td></td><td data-c="sumUn"></td><td>1</td></tr>`;
  $$("input", facBody).forEach(inp => {
    inp.addEventListener("input", () => {
      const v = parseFloat(inp.value);
      if (!isFinite(v) || v < 0 || v > 100) return;
      fac.machines[+inp.dataset.i][inp.dataset.k] = v;
      drawFac();
    });
  });
}
function facNumbers() {
  const total = fac.machines.reduce((s, m) => s + m.share, 0);
  const rows = fac.machines.map(m => {
    const prior = total > 0 ? m.share / total : 1 / fac.machines.length;
    const r = m.rate / 100;
    const lik = r ** fac.bad * (1 - r) ** fac.good;
    return { prior, lik, un: prior * lik };
  });
  const sum = rows.reduce((s, r) => s + r.un, 0);
  rows.forEach(r => { r.post = sum > 0 ? r.un / sum : NaN; });
  return { rows, sum, total };
}
function drawFac() {
  const { rows, sum, total } = facNumbers();
  const trs = $$("tr", facBody);
  rows.forEach((r, i) => {
    $("[data-c=lik]", trs[i]).textContent = small(r.lik);
    $("[data-c=un]", trs[i]).textContent = small(r.un);
    $("[data-c=post]", trs[i]).textContent = isFinite(r.post) ? r.post.toFixed(3) : "–";
  });
  const tot = trs[trs.length - 1];
  $("[data-c=sumPrior]", tot).textContent = Math.abs(total - 100) < 1e-9 ? "100 %" : `${fmt(total, 1)} % → 100 %`;
  $("[data-c=sumUn]", tot).textContent = small(sum);

  const { ctx, w, h } = fit($("#facC"), $("#facC").clientWidth < 520 ? 1.6 : 2.8);
  const pad = { l: 8, r: 8, t: 40, b: 26 }, ih = h - pad.t - pad.b;
  const top = Math.max(0.1, ...rows.map(r => Math.max(r.prior, isFinite(r.post) ? r.post : 0)));
  const gw = (w - pad.l - pad.r) / rows.length, bw = Math.min(54, gw * 0.3);
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center";
  rows.forEach((r, i) => {
    const cx = pad.l + gw * (i + 0.5);
    [[r.prior, css("--prior"), cx - bw / 2 - 2], [r.post, css("--post"), cx + bw / 2 + 2]].forEach(([v, color, x]) => {
      if (!isFinite(v)) return;
      const bh = v / top * ih;
      ctx.fillStyle = color;
      ctx.fillRect(x - bw / 2, pad.t + ih - bh, bw, bh);
      ctx.fillStyle = css("--ink-2");
      ctx.fillText(v.toFixed(2), x, pad.t + ih - bh - 5);
    });
    ctx.fillStyle = css("--ink");
    ctx.font = "600 12px " + css("--font-ui");
    ctx.fillText(`machine ${fac.machines[i].name}`, cx, h - 8);
    ctx.font = "11px " + css("--font-mono");
  });
  ctx.textAlign = "left";
  ctx.fillStyle = css("--prior"); ctx.fillRect(pad.l, 4, 10, 10);
  ctx.fillStyle = css("--ink-2"); ctx.fillText("prior", pad.l + 14, 13);
  ctx.fillStyle = css("--post"); ctx.fillRect(pad.l + 60, 4, 10, 10);
  ctx.fillStyle = css("--ink-2"); ctx.fillText("posterior", pad.l + 74, 13);

  let cap;
  if (!fac.bad && !fac.good) {
    cap = "No part inspected yet, so every likelihood is 1 and the posterior is just the prior. Press <b>found a defective part</b>.";
  } else if (!(sum > 0)) {
    cap = "No machine can produce what you found with these defect rates: every likelihood is 0, so there is nothing to divide by.";
  } else {
    const best = rows.reduce((bi, r, i) => r.post > rows[bi].post ? i : bi, 0);
    const parts = [fac.bad && `${fac.bad} defective`, fac.good && `${fac.good} good`].filter(Boolean).join(" and ");
    cap = `Evidence so far: ${parts} part${fac.bad + fac.good > 1 ? "s, all from the same machine" : ""}. ` +
      `<b>Multiply</b> each prior by the likelihood, <b>sum</b> (${small(sum)}), <b>divide</b>. ` +
      `Machine ${fac.machines[best].name} is now the most likely source, at <b>${pct(rows[best].post)}</b>.`;
  }
  if (Math.abs(total - 100) > 1e-9 && total > 0) cap += ` The shares add up to ${fmt(total, 1)}%, so they are rescaled to 100%.`;
  $("#facCap").innerHTML = cap;
}
$("#facBad").addEventListener("click", () => { fac.bad++; drawFac(); });
$("#facGood").addEventListener("click", () => { fac.good++; drawFac(); });
$("#facReset").addEventListener("click", () => {
  fac.machines = [{ name: "A", share: 50, rate: 1 }, { name: "B", share: 30, rate: 2 }, { name: "C", share: 20, rate: 5 }];
  fac.bad = fac.good = 0;
  buildFacTable(); drawFac();
});

/* =====================================================================
   101 HYPOTHESES FOR A CURE RATE
   ===================================================================== */
const THETA = Array.from({ length: 101 }, (_, i) => i / 100);
const cure = { cured: 0, n: 0, randomUsed: false };
function curePosterior() {
  const lp = THETA.map(t => cure.cured * Math.log(t) + (cure.n - cure.cured) * Math.log(1 - t));
  const top = Math.max(...lp.filter(isFinite));
  const un = lp.map(v => isFinite(v) ? Math.exp(v - top) : 0);
  const s = un.reduce((a, b) => a + b, 0);
  return un.map(v => v / s);
}
function drawCure() {
  const post = curePosterior();
  const { ctx, w, h } = fit($("#cureC"), $("#cureC").clientWidth < 520 ? 1.5 : 2.6);
  const pad = { l: 10, r: 10, t: 14, b: 26 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const top = Math.max(0.03, ...post) * 1.1;
  const X = t => pad.l + t * iw, Y = v => pad.t + ih - v / top * ih;
  const bw = iw / 101;
  ctx.fillStyle = css("--post");
  post.forEach((v, i) => ctx.fillRect(X(THETA[i]) - bw / 2 + 0.3, Y(v), Math.max(0.6, bw - 0.6), pad.t + ih - Y(v)));
  ctx.strokeStyle = css("--prior"); ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(X(0), Y(1 / 101)); ctx.lineTo(X(1), Y(1 / 101)); ctx.stroke();
  if (cure.randomUsed) {
    ctx.strokeStyle = css("--lik");
    ctx.beginPath(); ctx.moveTo(X(0.3), pad.t); ctx.lineTo(X(0.3), pad.t + ih); ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t + ih + 0.5); ctx.lineTo(w - pad.r, pad.t + ih + 0.5); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "center";
  for (const t of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(pct(t, 0), Math.min(w - 16, Math.max(16, X(t))), h - 8);
  ctx.textAlign = "left";
  ctx.fillText("prior", X(0.01), Y(1 / 101) - 5);
  if (cure.randomUsed) { ctx.fillStyle = css("--lik"); ctx.fillText("true rate", X(0.3) + 5, pad.t + 10); }

  let cap;
  if (!cure.n) {
    cap = "101 hypotheses for the cure rate, from 0% to 100%, all equally likely to begin with: the flat prior (dashed).";
  } else {
    const mode = THETA[post.indexOf(Math.max(...post))];
    const above = post.reduce((s, v, i) => s + (THETA[i] > 0.5 ? v : 0), 0);
    let c = 0, lo = 0, hi = 1;
    for (let i = 0; i < post.length; i++) { c += post[i]; if (c >= 0.05) { lo = THETA[i]; break; } }
    c = 0;
    for (let i = post.length - 1; i >= 0; i--) { c += post[i]; if (c >= 0.05) { hi = THETA[i]; break; } }
    cap = `<b>${cure.n}</b> patient${cure.n > 1 ? "s" : ""}, <b>${cure.cured}</b> cured. Most probable cure rate: <b>${pct(mode, 0)}</b>. ` +
      `About 90% of the probability lies between ${pct(lo, 0)} and ${pct(hi, 0)}, and the probability that the treatment cures more than half of patients is <b>${pct(above)}</b>.`;
  }
  $("#cureCap").innerHTML = cap;
}
$("#cureYes").addEventListener("click", () => { cure.cured++; cure.n++; drawCure(); });
$("#cureNo").addEventListener("click", () => { cure.n++; drawCure(); });
$("#cure10").addEventListener("click", () => {
  cure.randomUsed = true;
  for (let i = 0; i < 10; i++) { cure.n++; if (Math.random() < 0.3) cure.cured++; }
  drawCure();
});
$("#cureReset").addEventListener("click", () => { cure.cured = cure.n = 0; cure.randomUsed = false; drawCure(); });

/* =====================================================================
   start-up
   ===================================================================== */
syncCrowdUI();
recount();
updateCrowd();
$("#deckA").value = deck.a; $("#deckB").value = deck.b;
drawDeck();
drawSeq();
buildFacTable();
drawFac();
drawCure();
onRedraw(() => { drawCrowd(); drawSeq(); drawFac(); drawCure(); });
})();
