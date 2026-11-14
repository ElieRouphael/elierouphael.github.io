/* Which Job Should the Machine Do Next?, chapter D1, dispatching as reinforcement learning: the demos.
   Hero: a tabular Q-learning agent learns, in the browser, which of four rules to apply on random
   six-order instances of a paint-like machine, with its learning curve against the fixed rules.
   Widget 2: one run decision by decision, the committed tardiness and the telescoping rewards.
   Widget 3: overestimation, E[max] >= max E, with a single and a double estimator.
   Widget 4: the workshop agent's real training curves (models/training.json of the repository).
   The toy is mirrored line by line in code/d01_numbers.py, which prints the same numbers.
   Needs ../kit/kit.js and jobshop-lib.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, fit, plotLine, alpha, onRedraw, narrowOf, reduceMotion } = window.Kit;
const { rng, FAM, drawGantt } = window.JS;

const RULES = ["FIFO", "SPT", "EDD", "SST"];
const RULE_COL = () => [css("--muted"), css("--c5"), css("--c4"), css("--warm")];
const N = 6;

/* ---------------- the toy: one machine, two colours, six orders waiting at t = 0 ----------------
   An order is [idx, colour (0 blue, 1 red), painting time, due date]. Due dates are at least the
   painting time, so nothing is committed late before the first decision. */
function instance(r, spread) {
  const raw = [];
  for (let j = 0; j < N; j++) { const c = r() < 0.5 ? 1 : 0; const p = 5 + Math.floor(r() * 21); raw.push([c, p]); }
  const total = raw.reduce((s, x) => s + x[1], 0);
  return raw.map(([c, p], j) => [j, c, p, p + Math.floor(r() * spread * total)]);
}
function lexLess(a, b) { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i]; return false; }

function makeToy(S, spread) {
  const setup = (f, g) => (f === g ? 0 : S);
  const key = (a, o, f) => a === 0 ? [o[0]] : a === 1 ? [o[2], o[0]] : a === 2 ? [o[3], o[0]] : [setup(f, o[1]), o[3], o[0]];
  const pick = (a, wait, f) => { let b = wait[0]; for (let i = 1; i < wait.length; i++) if (lexLess(key(a, wait[i], f), key(a, b, f))) b = wait[i]; return b; };
  const committed = (t, doneT, wait) => doneT + wait.reduce((s, o) => s + Math.max(0, t + o[2] - o[3]), 0);
  function state(t, f, wait) {
    const n = wait.length, size = n === 2 ? 0 : n <= 4 ? 1 : 2, e = pick(2, wait, f);
    const colour = e[1] === f ? 1 : wait.some(o => o[1] === f) ? 2 : 0;
    const tight = e[3] - t - setup(f, e[1]) - e[2] < S ? 1 : 0;
    return (size * 3 + colour) * 2 + tight;
  }
  /* one run; policy(s) -> action; learn(s, a, r, s2, terminal); trace collects the decisions */
  function episode(inst, policy, learn, trace) {
    let t = 0, f = 0, doneT = 0, pending = null;
    const wait = inst.slice(), blocks = [];
    while (wait.length) {
      let o;
      if (wait.length >= 2) {
        const s = state(t, f, wait), E = committed(t, doneT, wait);
        if (pending) {
          if (learn) learn(pending[0], pending[1], -(E - pending[2]), s, false);
          if (trace) trace.push({ s: pending[0], a: pending[1], E: pending[2], t: pending[3], r: -(E - pending[2]), n: pending[4] });
        }
        const a = policy(s);
        pending = [s, a, E, t, wait.length];
        o = pick(a, wait, f);
      } else o = wait[0];
      wait.splice(wait.indexOf(o), 1);
      const st = setup(f, o[1]);
      if (trace) blocks.push({ o, free: t, start: t + st, end: t + st + o[2] });
      t += st + o[2]; f = o[1]; doneT += Math.max(0, t - o[3]);
    }
    if (learn) learn(pending[0], pending[1], -(doneT - pending[2]), null, true);
    if (trace) { trace.push({ s: pending[0], a: pending[1], E: pending[2], t: pending[3], r: -(doneT - pending[2]), n: pending[4] }); trace.blocks = blocks; trace.end = t; }
    return doneT;
  }
  function seqTardiness(seq) { let t = 0, f = 0, T = 0; for (const o of seq) { t += setup(f, o[1]) + o[2]; f = o[1]; T += Math.max(0, t - o[3]); } return T; }
  /* best of the 4^5 successions of rules (one rule per decision), in hindsight: the ceiling of a rule
     selector. Searched as a tree; two rules that pick the same order lead to the same subtree. */
  function bestSuccession(inst) {
    const rec = (t, f, wait, doneT) => {
      if (wait.length === 1) { const o = wait[0], e = t + setup(f, o[1]) + o[2]; return doneT + Math.max(0, e - o[3]); }
      let best = Infinity; const seen = [];
      for (let a = 0; a < 4; a++) {
        const o = pick(a, wait, f); if (seen.includes(o)) continue; seen.push(o);
        const e = t + setup(f, o[1]) + o[2];
        best = Math.min(best, rec(e, o[1], wait.filter(x => x !== o), doneT + Math.max(0, e - o[3])));
      }
      return best;
    };
    return rec(0, 0, inst.slice(), 0);
  }
  const rv = rng(999), rt = rng(777);
  const val = Array.from({ length: 200 }, () => instance(rv, spread));
  const test = Array.from({ length: 200 }, () => instance(rt, spread));
  const meanRule = (set, a) => set.reduce((s, i) => s + episode(i, () => a), 0) / set.length;
  return { S, spread, setup, pick, state, episode, seqTardiness, bestSuccession, val, test, meanRule };
}

const greedyOf = Q => s => { const q = Q[s]; let b = 0; for (let a = 1; a < 4; a++) if (q[a] > q[b]) b = a; return b; };
function train(toy, seed, episodes, alphaStep, gamma, every = 100) {
  const r = rng(seed), Q = Array.from({ length: 18 }, () => [0, 0, 0, 0]), visits = new Array(18).fill(0);
  let done = 0;
  const greedy = greedyOf(Q);
  const behaviour = s => { visits[s]++; const eps = Math.max(0.05, 1 - done / (0.6 * episodes)); if (r() < eps) return Math.floor(r() * 4); return greedy(s); };
  const learn = (s, a, rew, s2, terminal) => { const target = rew + (terminal ? 0 : gamma * Math.max(...Q[s2])); Q[s][a] += alphaStep * (target - Q[s][a]); };
  const curve = [];
  let best = { v: Infinity, Q: null, ep: 0 };
  for (let ep = 0; ep <= episodes; ep++) {
    if (ep % every === 0) {
      const v = toy.val.reduce((s, i) => s + toy.episode(i, greedy), 0) / toy.val.length;
      curve.push(v);
      if (v < best.v) best = { v, Q: Q.map(q => q.slice()), ep };
    }
    if (ep === episodes) break;
    toy.episode(instance(r, toy.spread), behaviour, learn);
    done++;
  }
  const gb = greedyOf(best.Q);
  const test = toy.test.reduce((s, i) => s + toy.episode(i, gb), 0) / toy.test.length;
  return { curve, every, bestVal: best.v, bestEp: best.ep, test, Q: best.Q, policy: Array.from({ length: 18 }, (_, s) => gb(s)), visits, greedy: gb };
}

/* all 720 orders of six indices, computed once */
const PERMS = (function perms(a) { if (a.length <= 1) return [a]; const out = []; a.forEach((x, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).forEach(p => out.push([x, ...p]))); return out; })([0, 1, 2, 3, 4, 5]);

/* ---------------- hero state ---------------- */
const H = { toy: null, toyKey: "", refs: null, run: null, shown: Infinity, anim: 0, sel: 11 };  // 11: 3-4 waiting, conflict, tight
function params() {
  return { S: +$("#hS").value, F: +$("#hF").value, A: +$("#hA").value, G: +$("#hG").value, E: +$("#hE").value, seed: +$("#hSeed").value };
}
function compute() {
  const p = params(), key = p.S + "|" + p.F;
  if (key !== H.toyKey) {
    H.toy = makeToy(p.S, p.F); H.toyKey = key;
    const t = H.toy, valR = [0, 1, 2, 3].map(a => t.meanRule(t.val, a)), testR = [0, 1, 2, 3].map(a => t.meanRule(t.test, a));
    let best = 0; for (let a = 1; a < 4; a++) if (valR[a] < valR[best]) best = a;
    const hind = t.test.reduce((s, i) => s + Math.min(...[0, 1, 2, 3].map(a => t.episode(i, () => a))), 0) / 200;
    const opt = t.test.reduce((s, i) => s + Math.min(...PERMS.map(pm => t.seqTardiness(pm.map(k => i[k])))), 0) / 200;
    const succ = t.test.reduce((s, i) => s + t.bestSuccession(i), 0) / 200;
    H.refs = { valR, testR, best, hind, succ, opt };
  }
  H.run = train(H.toy, p.seed, p.E, p.A, p.G);
}

function legendRow(ctx, items, x, y, narrow) {
  ctx.font = "11px " + css("--font-mono"); ctx.textBaseline = "middle"; ctx.textAlign = "left";
  let cx = x, cy = y;
  const maxX = ctx.canvas.clientWidth - 12;
  for (const [label, col, dash] of items) {
    const w = 22 + ctx.measureText(label).width + 14;
    if (cx + w > maxX && cx > x) { cx = x; cy += 15; }
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash || []);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + 16, cy); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css("--ink-2"); ctx.fillText(label, cx + 21, cy);
    cx += w;
  }
}

function drawHero() {
  const { run, refs } = H, p = params();
  const cv = $("#qlC"), narrow = narrowOf(cv);
  const nShow = Math.min(run.curve.length, H.shown);
  const all = [...run.curve, ...refs.valR, validOpt()];
  const y1 = Math.ceil(Math.max(...run.curve.slice(0, 1), ...refs.valR) / 10) * 10 + 10, y0 = Math.max(0, Math.floor(Math.min(...all) / 10) * 10 - 10);
  const fr = frame(cv, narrow ? 1.0 : 1.75, { x: [0, p.E], y: [y0, y1], xlabel: "training episodes", ylabel: "mean total tardiness of six orders, validation (min)", pad: { t: narrow ? 90 : 58 } });
  const { ctx, X, Y } = fr, cols = RULE_COL();
  refs.valR.forEach((v, a) => plotLine(fr, [0, p.E], [v, v], { color: cols[a], width: a === refs.best ? 2 : 1.2, dash: [5, 4] }));
  plotLine(fr, [0, p.E], [validOpt(), validOpt()], { color: css("--ink"), width: 1, dash: [1.5, 3] });
  const xs = run.curve.slice(0, nShow).map((_, i) => i * run.every);
  plotLine(fr, xs, run.curve.slice(0, nShow), { color: css("--accent"), width: 2.2 });
  if (nShow >= run.curve.length) {
    ctx.fillStyle = css("--accent"); ctx.beginPath(); ctx.arc(X(run.bestEp), Y(run.bestVal), 4.5, 0, 2 * Math.PI); ctx.fill();
  }
  /* the legend names each rule in its own colour (no labels on the lines, which the curve would cross) */
  legendRow(ctx, [["agent (greedy)", css("--accent")],
    ...refs.valR.map((v, a) => [`${RULES[a]} ${fmt(v, 1)}${a === refs.best ? " (best)" : ""}`, cols[a], [5, 4]]),
    [`optimum ${fmt(validOpt(), 1)}`, css("--ink"), [1.5, 3]]], fr.pad.l, 26, narrow);
  // readouts
  const b = refs.best, gain = run.test - refs.testR[b];
  $("#rBest").textContent = `${RULES[b]} · ${fmt(refs.testR[b], 1)}`;
  $("#rAgent").textContent = fmt(run.test, 1);
  $("#rGain").textContent = `${gain >= 0 ? "+" : "−"}${fmt(Math.abs(gain), 1)} (${gain >= 0 ? "+" : "−"}${fmt(Math.abs(100 * gain / refs.testR[b]), 0)} %)`;
  $("#rHind").textContent = fmt(refs.hind, 1);
  $("#rSucc").textContent = fmt(refs.succ, 1);
  $("#rOpt").textContent = fmt(refs.opt, 1);
  $("#rEp").textContent = String(run.bestEp);
  $("#hSV").textContent = p.S; $("#hFV").textContent = fmt(p.F, 1); $("#hAV").textContent = fmt(p.A, 2); $("#hGV").textContent = fmt(p.G, 2);
  $("#hEV").textContent = p.E; $("#hSeedV").textContent = p.seed;
  drawPolicy();
  caption($("#heroStatus"), `A toy, not the repository's agent: 18 situations against the workshop agent's 29 features, and a changeover of ${p.S} min (5 by default) against the 10 min of the paint example elsewhere in the course. Due-date spread ${fmt(p.F, 1)}, seed ${p.seed}. Before any training every [[Q(s,a)]] is 0 and ties go to FIFO, so the curve starts at FIFO's ${fmt(run.curve[0], 1)} min. After ${p.E} episodes of [[\\varepsilon]]-greedy Q-learning (step size [[\\alpha = ${fmt(p.A, 2)}]], [[\\gamma = ${fmt(p.G, 2)}]]) the best checkpoint on the 200 validation instances, at episode ${run.bestEp}, scores ${fmt(run.bestVal, 1)}. On 200 fresh test instances it gives ${fmt(run.test, 1)} min against ${fmt(refs.testR[b], 1)} for ${RULES[b]}, the best fixed rule on validation: ${gain < 0 ? "a state-dependent choice of rule beats every single rule here" : "no better than the best single rule"}. In hindsight, the best single rule for each instance gives ${fmt(refs.hind, 1)}, the best succession of rules (one per decision, the most a rule selector could do) ${fmt(refs.succ, 1)}, and the optimal sequence ${fmt(refs.opt, 1)}.`);
}
/* the optimum on the validation set, for the dotted line (computed lazily, cached per instance set) */
function validOpt() {
  if (H.valOptKey !== H.toyKey) {
    const t = H.toy;
    H.valOpt = t.val.reduce((s, i) => s + Math.min(...PERMS.map(pm => t.seqTardiness(pm.map(k => i[k])))), 0) / 200;
    H.valOptKey = H.toyKey;
  }
  return H.valOpt;
}

function drawPolicy() {
  const { run } = H, cols = RULE_COL();
  const head = ["", "every order needs a change", "the most urgent has the current colour", "the most urgent needs a change, another does not"];
  let html = '<table class="num pol"><thead><tr>' + head.map(h => `<th>${h}</th>`).join("") + "</tr></thead><tbody>";
  for (let size = 0; size < 3; size++) for (let tight = 0; tight < 2; tight++) {
    html += `<tr><th>${["2", "3–4", "5–6"][size]} waiting, ${tight ? "urgent order tight" : "urgent order has slack"}</th>`;
    for (let colour = 0; colour < 3; colour++) {
      const s = (size * 3 + colour) * 2 + tight, a = run.policy[s], v = run.visits[s];
      html += v ? `<td class="cell${s === H.sel ? " sel" : ""}" tabindex="0" role="button" data-s="${s}" aria-label="show the four learned values of this situation"><span class="chip" style="background:${cols[a]}">${RULES[a]}</span><span class="vis">${v}</span></td>` : '<td class="never">never seen</td>';
    }
    html += "</tr>";
  }
  $("#qlPol").innerHTML = html + "</tbody></table>";
  $$("#qlPol td.cell").forEach(td => {
    const choose = () => { H.sel = +td.dataset.s; drawPolicy(); const n = $(`#qlPol td.cell[data-s="${H.sel}"]`); if (n) n.focus(); };
    td.addEventListener("click", choose);
    td.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(); } });
  });
  const s = H.sel;
  if (!run.visits[s]) { $("#qlQ").textContent = "Click a cell to see its four learned values."; return; }
  const q = run.Q[s], g = run.policy[s];
  $("#qlQ").textContent = `${SIT_LABEL(s)}: Q(FIFO) ${fmt(q[0], 1)}, Q(SPT) ${fmt(q[1], 1)}, Q(EDD) ${fmt(q[2], 1)}, Q(SST) ${fmt(q[3], 1)}; the largest (least negative), ${RULES[g]}, is applied. Each estimates minus the tardiness still to come, in minutes${params().G < 1 ? ", discounted" : ""}, if that rule is applied now and the agent goes on as learned.`;
}
/* the situation of a state, as in the table (and in code/d01_numbers.py) */
const SIT_LABEL = s => `${["2", "3–4", "5–6"][Math.floor(s / 6)]} waiting, ${["all need a change", "urgent has the colour", "conflict"][Math.floor((s % 6) / 2)]}, ${s % 2 ? "tight" : "slack"}`;

let pending = 0;
function recompute() {
  cancelAnimationFrame(pending);
  pending = requestAnimationFrame(() => { compute(); H.shown = Infinity; drawHero(); drawTel(); });
}
["#hS", "#hF", "#hA", "#hG", "#hE", "#hSeed"].forEach(id => $(id).addEventListener("input", recompute));
$("#hWatch").addEventListener("click", () => {
  cancelAnimationFrame(H.anim);
  if (reduceMotion) { H.shown = Infinity; drawHero(); return; }
  const n = H.run.curve.length, t0 = performance.now(), dur = 2600;
  const stepA = now => { H.shown = Math.max(1, Math.ceil(n * Math.min(1, (now - t0) / dur))); drawHero(); if (H.shown < n) H.anim = requestAnimationFrame(stepA); };
  H.anim = requestAnimationFrame(stepA);
});

/* ---------------- widget 2: one run, decision by decision ---------------- */
const TEL = { policy: -1 };
$$("#telPol .btn").forEach(b => b.addEventListener("click", () => {
  TEL.policy = +b.dataset.p; $$("#telPol .btn").forEach(x => x.setAttribute("aria-pressed", String(x === b))); drawTel();
}));
$("#telI").addEventListener("input", drawTel);
$("#telG").addEventListener("input", drawTel);
function drawTel() {
  if (!H.run) return;
  const i = +$("#telI").value, g = +$("#telG").value, inst = H.toy.test[i], trace = [];
  const pol = TEL.policy < 0 ? H.run.greedy : () => TEL.policy;
  const T = H.toy.episode(inst, pol, null, trace);
  $("#telIV").textContent = i; $("#telGV").textContent = fmt(g, 2);
  const L = "ABCDEF";
  const blocks = [];
  trace.blocks.forEach(b => {
    if (b.start > b.free) blocks.push({ lane: 0, start: b.free, end: b.start, kind: "setup" });
    blocks.push({ lane: 0, start: b.start, end: b.end, colour: b.o[1] ? FAM.red : FAM.blue, label: L[b.o[0]], kind: "work" });
  });
  const tEnd = Math.max(trace.end, ...inst.map(o => o[3])) + 4;
  /* due dates on their own row below the machine, so that they stay visible on red bars */
  const gt = drawGantt($("#telG1"), blocks, { lanes: 2, t0: 0, t1: tEnd, laneLabels: ["machine", "due dates"], laneHeight: 34, labelWidth: 78 });
  const gctx = $("#telG1").getContext("2d");
  gctx.font = "600 11px " + css("--font-mono"); gctx.textAlign = "center"; gctx.textBaseline = "top";
  let lastX = -Infinity, low = false;
  inst.slice().sort((p, q) => p[3] - q[3]).forEach(o => {
    const x = gt.X(o[3]), y = gt.laneH;
    low = x - lastX < 14 ? !low : false;   // two close due dates: the second label goes lower
    lastX = x;
    gctx.strokeStyle = gctx.fillStyle = o[1] ? FAM.red : FAM.blue; gctx.lineWidth = 2;
    gctx.beginPath(); gctx.moveTo(x, y + 2); gctx.lineTo(x, y + (low ? 20 : 9)); gctx.stroke();
    gctx.fillText(L[o[0]], x, y + (low ? 21 : 10));
  });
  // committed tardiness Θ(t) at each decision, and T at the end
  const cv = $("#telE"), narrow = narrowOf(cv), yMax = Math.max(10, T * 1.15);
  const fr = frame(cv, narrow ? 1.6 : 3.2, { x: [0, tEnd], y: [0, yMax], xlabel: "time (min)", ylabel: "committed tardiness Θ(t) (min)", pad: { l: 70 } });
  const { ctx, X, Y } = fr;
  const pts = trace.map(d => [d.t, d.E]).concat([[trace.end, T]]);
  ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2; ctx.beginPath();
  pts.forEach(([t, e], k) => { if (!k) ctx.moveTo(X(t), Y(e)); else { ctx.lineTo(X(t), Y(pts[k - 1][1])); ctx.lineTo(X(t), Y(e)); } });
  ctx.stroke();
  pts.forEach(([t, e], k) => { ctx.fillStyle = k === pts.length - 1 ? css("--warm") : css("--accent"); ctx.beginPath(); ctx.arc(X(t), Y(e), 3.5, 0, 2 * Math.PI); ctx.fill(); });
  // table
  const sum = trace.reduce((s, d) => s + d.r, 0), disc = trace.reduce((s, d, k) => s + g ** k * d.r, 0);
  let html = '<table class="num"><thead><tr><th>decision k</th><th>at minute</th><th class="sit" style="text-align:left">situation (state)</th><th>rule</th><th>this Θ</th><th>reward = −(next Θ − this Θ)</th></tr></thead><tbody>';
  trace.forEach((d, k) => { html += `<tr><td>${k}</td><td>${d.t}</td><td class="sit">${SIT_LABEL(d.s)}</td><td>${RULES[d.a]}</td><td>${d.E}</td><td>${String(d.r).replace("-", "−")}</td></tr>`; });
  html += `<tr><td colspan="4">end of the run: total tardiness T (the last "next Θ")</td><td>${T}</td><td>Σ = ${String(sum).replace("-", "−")}</td></tr></tbody></table>`;
  $("#telTab").innerHTML = html;
  caption($("#telCap"), `Test instance ${i}, ${TEL.policy < 0 ? "the trained agent of the hero" : "the fixed rule " + RULES[TEL.policy]}. Orders ${inst.map(o => `${L[o[0]]} (${o[1] ? "red" : "blue"}, [[p = ${o[2]}]], [[d = ${o[3]}]])`).join(", ")}; the ticks on the lower row are the due dates. The rewards are ${trace.map(d => String(d.r).replace("-", "−")).join(", ")} and their sum is [[${sum} = -T]], minus the run's total tardiness: each reward is minus one step of the blue staircase, and the steps add up to its final height. With [[\\gamma = ${fmt(g, 2)}]] the discounted return [[\\sum_k \\gamma^k r_{k+1} = ${fmt(disc, 2)}]]${g < 1 && sum < 0 ? ", less negative than [[-T]]: late penalties are worth less the later they come" : ""}.`);
}

/* ---------------- widget 3: overestimation ---------------- */
function gauss(r) { const u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function drawOver() {
  const sigma = +$("#ovS").value, gap = +$("#ovD").value, M = 6000, r = rng(7);
  const mu = [0, gap], single = new Float64Array(M), dbl = new Float64Array(M);
  for (let k = 0; k < M; k++) {
    const A = [mu[0] + sigma * gauss(r), mu[1] + sigma * gauss(r)], B = [mu[0] + sigma * gauss(r), mu[1] + sigma * gauss(r)];
    single[k] = Math.max(A[0], A[1]);
    dbl[k] = A[0] >= A[1] ? B[0] : B[1];
  }
  const mean = xs => xs.reduce((s, v) => s + v, 0) / xs.length;
  const ms = mean(single), md = mean(dbl), lo = -3.5 * sigma + Math.min(0, gap) - 0.2, hi = gap + 3.5 * sigma + 0.2;
  const cv = $("#ovC"), narrow = narrowOf(cv), bins = 48, w = (hi - lo) / bins;
  const fr = frame(cv, narrow ? 1.3 : 2.4, { x: [lo, hi], y: [0, 0.75 / sigma], xlabel: "estimated value of the best action", yticks: false, pad: { t: narrow ? 62 : 46 } });
  const { ctx, X, Y } = fr;
  const hist = (xs, col) => {
    const c = new Float64Array(bins); for (const v of xs) { const i = Math.floor((v - lo) / w); if (i >= 0 && i < bins) c[i]++; }
    ctx.fillStyle = col; c.forEach((n, i) => { const h = n / (M * w); ctx.fillRect(X(lo + i * w) + 0.5, Y(h), Math.max(1, X(lo + (i + 1) * w) - X(lo + i * w) - 1), Y(0) - Y(h)); });
  };
  hist(dbl, alpha("--warm", 0.45)); hist(single, alpha("--accent", 0.45));
  const vline = (x, col, dash) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash || []); ctx.beginPath(); ctx.moveTo(X(x), fr.pad.t); ctx.lineTo(X(x), Y(0)); ctx.stroke(); ctx.setLineDash([]); };
  vline(ms, css("--accent")); vline(md, css("--warm"));
  /* the true best value: a dashed line drawn on top, and a triangle at the top that stays visible
     when the double estimator's mean sits on it */
  const tv = Math.max(...mu);
  vline(tv, css("--ink"), [4, 3]);
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.moveTo(X(tv), fr.pad.t + 9); ctx.lineTo(X(tv) - 6, fr.pad.t - 2); ctx.lineTo(X(tv) + 6, fr.pad.t - 2); ctx.closePath(); ctx.fill();
  legendRow(ctx, [["mean of the max of the estimates", css("--accent")], ["mean of the double estimator", css("--warm")], ["true best value (▼)", css("--ink"), [4, 3]]], fr.pad.l, 14, narrow);
  $("#ovSV").textContent = fmt(sigma, 1); $("#ovDV").textContent = fmt(gap, 1);
  caption($("#ovCap"), `Two actions with true values [[${fmt(0, 1)}]] and [[${fmt(gap, 1)}]], each estimated with noise of standard deviation [[\\sigma = ${fmt(sigma, 1)}]], ${M} trials; the histograms show the estimates, the solid vertical lines their means. Taking the larger estimate gives on average ${fmt(ms, 3)}, above the true best value ${fmt(tv, 1)} by ${fmt(ms - tv, 3)}${gap === 0 ? ` (theory: [[\\sigma/\\sqrt{\\pi} = ${fmt(sigma / Math.sqrt(Math.PI), 3)}]])` : ""}. Choosing with one estimate and reading the value from an independent one gives ${fmt(md, 3)}: no upward bias${gap > 0 ? ", and a slight downward one when the choice is sometimes wrong" : ""}. Widen the gap: when one action is clearly better, the bias vanishes, because the noise rarely changes which one is picked.`);
}
["#ovS", "#ovD"].forEach(id => $(id).addEventListener("input", drawOver));

/* ---------------- widget 4: the workshop agent's training (models/training.json) ---------------- */
/* printed by code/d01_numbers.py, section "Data embedded in interactive/d01.js" */
const TRAIN = {"rules":{"FIFO":47.54,"SPT":32.96,"EDD":26.99,"SST":24.29},"selected":3,"seeds":[{"seed":0,"best":23.27,"bestEp":1375,"val":[23.66,27.23,28.93,25.65,30.92,26.43,27.94,26.59,28.86,28.05,27.73,31.66,33.71,26.67,26.58,28.36,30.95,28.0,33.15,27.32,30.91,29.82,24.42,32.15,27.03,28.38,28.89,29.5,31.71,28.59,32.72,29.73,26.48,31.47,30.81,27.08,29.81,27.93,27.96,26.57,29.41,25.17,27.05,27.24,24.51,25.59,29.11,29.25,29.41,29.02,27.78,30.58,30.69,28.49,23.27,29.22,33.23,25.46,24.44,31.73],"train":[34.43,29.71,31.19,29.63,35.64,19.71,23.82,29.54,21.67,23.62,21.35,19.47,23.69,22.21,29.77,25.62,16.07,30.46,27.05,22.51,26.04,18.64,23.21,22.06,18.76,26.95,21.24,22.92,23.86,29.18,24.02,16.1,22.78,28.91,25.1,16.18,22.65,21.51,20.09,21.28,26.82,17.9,22.54,25.81,23.27,29.58,22.31,21.87,31.29,24.81,22.35,18.39,26.35,21.64,19.41,18.52,36.37,19.73,24.6,22.9],"eps":[0.932,0.875,0.818,0.76,0.706,0.67,0.632,0.592,0.561,0.53,0.5,0.475,0.447,0.421,0.396,0.374,0.358,0.337,0.318,0.301,0.285,0.271,0.257,0.244,0.233,0.222,0.212,0.202,0.192,0.182,0.174,0.167,0.16,0.153,0.146,0.141,0.135,0.131,0.126,0.121,0.117,0.113,0.109,0.105,0.101,0.098,0.095,0.092,0.089,0.087,0.084,0.082,0.08,0.078,0.077,0.075,0.073,0.072,0.071,0.069]},{"seed":1,"best":23.23,"bestEp":200,"val":[27.04,31.1,29.43,24.59,26.49,27.89,27.59,23.23,27.58,25.94,31.65,26.57,30.28,28.18,26.41,24.07,27.5,27.04,28.72,25.37,29.31,28.94,28.44,29.67,25.86,24.68,26.73,26.3,27.96,25.14,26.83,27.62,30.75,25.82,26.98,27.35,27.34,27.86,27.0,28.43,29.35,27.3,23.75,29.16,26.92,24.71,27.85,27.01,25.52,28.17,30.6,31.0,28.69,29.74,29.67,28.46,25.39,27.4,29.53,29.05],"train":[30.79,19.96,24.86,31.6,25.69,29.64,24.68,24.76,33.91,27.9,23.52,18.97,28.81,21.58,22.25,19.33,36.41,19.03,24.11,25.17,25.29,26.7,19.48,27.83,32.9,31.29,19.39,27.42,23.61,25.97,30.24,18.46,19.69,19.55,22.39,21.84,17.62,18.54,28.4,21.34,25.28,24.24,22.55,25.3,20.34,19.79,31.01,27.85,19.24,28.85,31.34,23.21,29.53,25.01,30.65,29.81,19.72,26.45,29.65,30.97],"eps":[0.936,0.884,0.829,0.775,0.726,0.678,0.641,0.604,0.565,0.53,0.5,0.473,0.444,0.419,0.397,0.376,0.353,0.336,0.318,0.302,0.287,0.272,0.258,0.243,0.23,0.218,0.208,0.197,0.188,0.179,0.171,0.164,0.158,0.152,0.146,0.14,0.135,0.13,0.125,0.121,0.116,0.112,0.108,0.105,0.101,0.098,0.095,0.092,0.09,0.087,0.084,0.082,0.08,0.078,0.076,0.074,0.073,0.072,0.07,0.069]},{"seed":2,"best":24.81,"bestEp":1500,"val":[25.82,31.0,27.87,29.02,27.9,28.48,30.52,27.99,30.71,29.05,29.14,28.46,30.32,32.81,32.28,29.64,30.52,29.91,32.73,27.29,35.85,26.1,27.83,30.78,31.1,30.91,28.69,29.66,34.87,37.39,31.53,31.09,32.53,26.64,26.9,34.35,26.29,30.18,27.61,33.49,24.97,29.86,26.54,29.61,29.84,27.46,29.95,29.3,26.49,29.81,30.54,28.22,28.84,34.3,29.67,24.9,30.12,31.76,25.32,24.81],"train":[28.04,25.61,25.44,28.54,22.7,22.3,35.13,30.99,22.19,32.81,33.66,24.77,31.16,26.42,35.25,33.12,21.82,27.56,28.97,33.93,22.09,28.58,15.69,25.16,38.98,27.19,42.81,21.17,25.85,30.31,23.76,24.29,18.13,21.44,36.01,25.23,28.24,22.46,21.44,30.61,26.34,33.76,26.01,19.12,23.65,23.73,28.28,20.66,25.21,21.18,33.1,17.79,35.01,22.18,19.27,21.62,34.02,22.79,18.84,22.06],"eps":[0.938,0.88,0.827,0.779,0.733,0.692,0.648,0.606,0.573,0.535,0.502,0.473,0.444,0.418,0.391,0.368,0.348,0.33,0.312,0.293,0.278,0.262,0.251,0.238,0.224,0.213,0.201,0.192,0.182,0.174,0.166,0.158,0.152,0.146,0.139,0.133,0.128,0.124,0.119,0.114,0.11,0.106,0.102,0.099,0.096,0.093,0.09,0.088,0.085,0.083,0.081,0.079,0.077,0.076,0.074,0.073,0.071,0.07,0.069,0.068]},{"seed":3,"best":21.98,"bestEp":1150,"val":[27.42,30.43,26.9,31.28,29.23,25.36,24.74,26.19,27.8,26.46,28.25,25.46,27.88,27.02,27.92,27.05,27.48,27.79,27.07,26.12,27.86,32.26,28.93,29.74,31.48,27.86,27.06,31.52,28.4,25.47,27.95,25.09,24.59,26.99,26.66,33.81,28.79,28.09,29.67,25.88,29.79,24.95,28.31,25.57,27.34,21.98,25.63,26.21,26.9,27.9,24.7,27.99,29.46,25.6,26.42,24.6,24.93,24.9,28.94,27.93],"train":[28.48,33.62,31.02,22.92,23.54,24.16,30.63,20.33,27.04,25.0,32.65,19.65,36.17,22.61,17.11,20.49,37.84,22.11,20.03,21.7,18.99,24.5,33.23,17.63,21.79,14.92,21.27,27.99,24.78,22.1,24.88,21.0,23.28,28.36,18.91,37.01,26.65,33.97,22.9,25.15,22.79,25.95,25.26,35.17,31.17,25.31,20.17,26.13,37.85,21.57,26.0,24.71,24.14,20.7,31.21,19.95,23.53,18.42,27.55,24.23],"eps":[0.937,0.879,0.822,0.771,0.722,0.685,0.639,0.602,0.565,0.532,0.5,0.475,0.446,0.422,0.399,0.378,0.354,0.333,0.317,0.303,0.289,0.274,0.258,0.245,0.234,0.225,0.214,0.204,0.193,0.183,0.175,0.167,0.16,0.153,0.147,0.14,0.134,0.128,0.124,0.119,0.114,0.11,0.106,0.102,0.099,0.095,0.093,0.09,0.087,0.085,0.082,0.081,0.079,0.077,0.075,0.074,0.072,0.071,0.069,0.068]},{"seed":4,"best":24.02,"bestEp":375,"val":[35.19,32.8,27.44,28.81,30.04,26.48,24.86,25.94,26.74,27.33,24.95,26.59,31.13,27.47,24.02,26.18,24.17,25.79,27.37,26.43,26.15,27.33,27.58,27.04,24.7,28.37,26.29,24.32,28.45,25.16,26.22,29.3,28.28,26.61,25.26,28.1,24.79,26.81,28.63,26.52,34.73,26.71,26.5,24.57,30.5,28.65,29.19,24.12,24.98,24.55,28.24,26.47,29.42,30.1,31.44,24.05,27.94,26.3,27.52,25.17],"train":[26.25,28.95,26.94,39.45,35.02,27.18,24.41,29.71,40.07,23.67,29.22,26.18,24.82,22.22,32.2,21.15,24.83,20.32,28.21,25.83,21.49,19.65,24.95,20.05,24.29,20.22,20.1,39.9,25.83,30.8,29.9,22.35,28.82,24.47,21.02,23.26,36.32,21.32,20.33,19.81,21.71,23.19,18.74,12.07,25.31,16.14,25.85,19.57,18.67,22.5,25.99,32.88,20.52,21.4,17.08,25.61,17.03,20.4,19.32,21.03],"eps":[0.934,0.876,0.825,0.766,0.717,0.673,0.634,0.594,0.555,0.526,0.493,0.466,0.44,0.415,0.391,0.37,0.351,0.332,0.313,0.296,0.282,0.268,0.255,0.243,0.231,0.22,0.21,0.198,0.189,0.18,0.17,0.163,0.155,0.149,0.143,0.137,0.131,0.126,0.122,0.118,0.114,0.11,0.107,0.104,0.101,0.098,0.095,0.092,0.09,0.088,0.085,0.083,0.081,0.079,0.078,0.076,0.074,0.073,0.072,0.07]}]};
const TR = { seed: TRAIN.selected };
$$("#trSeed .btn").forEach(b => b.addEventListener("click", () => {
  TR.seed = +b.dataset.s; $$("#trSeed .btn").forEach(x => x.setAttribute("aria-pressed", String(x === b))); drawTrain();
}));
function drawTrain() {
  const run = TRAIN.seeds.find(s => s.seed === TR.seed), cv = $("#trC"), narrow = narrowOf(cv);
  const eps = run.val.map((_, i) => 25 * (i + 1));
  const fr = frame(cv, narrow ? 1.05 : 2.0, { x: [0, 1500], y: [15, 45], xlabel: "training day (episode)", ylabel: "mean tardiness per job (min)", pad: { t: narrow ? 104 : 58 } });
  const { ctx, X, Y } = fr;
  plotLine(fr, eps, run.train, { color: alpha("--ink", 0.35), width: 1.2 });
  plotLine(fr, eps, run.eps.map(e => 15 + 30 * e), { color: css("--muted"), width: 1, dash: [2, 3] });
  [["SST", "--warm"], ["EDD", "--c4"]].forEach(([k, c]) => plotLine(fr, [0, 1500], [TRAIN.rules[k], TRAIN.rules[k]], { color: css(c), width: 1.4, dash: [5, 4] }));
  plotLine(fr, eps, run.val, { color: css("--accent"), width: 2 });
  ctx.fillStyle = css("--accent"); ctx.beginPath(); ctx.arc(X(run.bestEp), Y(run.best), 5, 0, 2 * Math.PI); ctx.fill();
  legendRow(ctx, [["validation (greedy agent)", css("--accent")], ["training days, 25-day means", alpha("--ink", 0.35)], ["ε, 0 to 1 on the axis height", css("--muted"), [2, 3]],
    ["SST on validation " + fmt(TRAIN.rules.SST, 2), css("--warm"), [5, 4]], ["EDD on validation " + fmt(TRAIN.rules.EDD, 2), css("--c4"), [5, 4]]], fr.pad.l, 26, narrow);
  const below = run.val.filter(v => v < TRAIN.rules.SST).length;
  caption($("#trCap"), `Training seed ${run.seed}${run.seed === TRAIN.selected ? " (the agent kept, best validation of the five)" : ""}: the validation score of the greedy network every 25 training days (20 runs: seeds 5000–5009 at both breakdown levels), from ${fmt(run.val[0], 2)} after day 25 to ${fmt(run.val[run.val.length - 1], 2)} after day 1500, best ${fmt(run.best, 2)} after day ${run.bestEp} (the dot). ${below} of its 60 checks ${below === 1 ? "falls" : "fall"} below SST's ${fmt(TRAIN.rules.SST, 2)}, the best rule on the same 20 runs. The grey line is what the exploring agent did on its training days; it falls mostly because [[\\varepsilon]] falls. The validation line does not trend: it wanders by a few minutes from check to check, and the checkpoint kept is the luckiest of 60.`);
}

function drawAll() { if (!H.run) compute(); drawHero(); drawTel(); drawOver(); drawTrain(); }
drawAll();
onRedraw(drawAll);
})();
