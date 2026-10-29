/* Kalman filters with JAX, chapter J2, Pure functions, tracing and jit: the live demos.
   1. The tracer: a Python function run one line at a time on stand-ins, with the jaxpr equations each
      line records. The jaxprs are JAX's own (jax.make_jaxpr), each equation tagged with its source line
      by code/j2_numbers.py; the notes are the page's.
   2. The cache: calls to a jitted Kalman step, each a hit or a miss, with the trace, lower, compile and
      run times measured in JAX.
   3. The bars of the 1000-step timing table.
   Needs ../kit/kit.js and j2-data.js. */
(() => {
"use strict";
const { $, $$, caption } = window.Kit;
const D = window.J2;
const H = D.hero;

const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const code = s => `<code>${esc(s)}</code>`;

/* ============================================================ 1. the tracer */
const tracer = s => `JitTracer(${s})`;
const FUNCS = {
  predict: {
    src: H.predict.source, header: H.predict.header, footer: H.predict.footer, eqns: H.predict.eqns,
    call: "linear_kf_predict(x0, P0, F, Q)", sees: tracer("float32[2]"),
    steps: [
      { line: 0, note: `<b>The call.</b> <code>jit</code> replaces each argument with a tracer and calls the function. <code>x</code> is now ${code(tracer("float32[2]"))}: a shape and a dtype, no numbers; <code>P</code>, <code>F</code> and <code>Q</code> are ${code(tracer("float32[2,2]"))}. The jaxpr's first line lists the four inputs, <code>a</code> to <code>d</code>, in the order <code>x</code>, <code>P</code>, <code>F</code>, <code>Q</code>. The docstring is skipped: it does nothing.` },
      { line: 2, note: `<b><code>F @ x</code></b>: Python evaluates the expression as usual, but <code>@</code> on tracers does not multiply anything. It records one equation, a <code>dot_general</code> that sums axis 1 of <code>F</code> (<code>c</code>) against axis 0 of <code>x</code> (<code>a</code>), and returns a new tracer, which Python names <code>x_pred</code>.` },
      { line: 3, note: `<b><code>F @ P @ F.T + Q</code></b> is four operations, recorded in Python's evaluation order: <code>F @ P</code>, the transpose <code>F.T</code>, the second product, and the addition of <code>Q</code>. Python's names do not reach the jaxpr: each intermediate gets a fresh letter and is assigned exactly once.` },
      { line: 4, ret: true, note: `<b><code>return</code></b>: the outputs <code>e</code> and <code>i</code> become the program's results. That is the last time this Python body runs for these shapes. The jaxpr goes to XLA, the compiled program is cached and run. Press <b>Call it again</b>.` },
    ],
    again: `<b>Second call, same shapes.</b> The cache holds a program for <code>(float32[2], float32[2,2], float32[2,2], float32[2,2])</code>, so JAX runs it directly. No line of the Python function runs: the count of Python runs stays at 1 while the compiled program's count goes up.`,
  },
  prints: {
    src: H.prints.source, header: H.prints.header, footer: H.prints.footer, eqns: H.prints.eqns,
    call: "fn_prints(x)", sees: tracer("float32[2]"),
    steps: [
      { line: 0, note: `<b>The call.</b> <code>x</code> is a tracer, ${code(tracer("float32[2]"))}.` },
      { line: 1, console: H.prints_console, note: `<b><code>print</code> runs now</b>, because it is ordinary Python and Python is running. It prints the tracer, not numbers. Nothing is recorded: <code>print</code> is not a JAX operation, so the compiled program will contain no print at all.` },
      { line: 2, note: `<b><code>2.0 * x + 1.0</code></b> records a multiplication and an addition. The Python numbers 2.0 and 1.0 enter the jaxpr as constants, typed <code>f32[]</code> to match <code>x</code> (they are weakly typed, J1 Part 4).` },
      { line: 3, ret: true, note: `<b><code>return</code></b>: the program has two operations and no print. Press <b>Call it again</b>.` },
    ],
    again: `<b>Second call: nothing is printed.</b> The <code>print</code> happened once, while tracing; the compiled program has only the multiplication and the addition. To print values on every call, use <code>jax.debug.print("{x}", x=x)</code>, which records a print operation in the program itself.`,
  },
  global: {
    src: H.global.source, header: H.global.header, footer: H.global.footer, eqns: H.global.eqns,
    pre: ["scale = 2.0", ""], preAgain: ["scale = 3.0          # changed before the second call", ""],
    call: "fn_global(x)", sees: tracer("float32[2]"),
    steps: [
      { line: 0, note: `<b>The call.</b> <code>x</code> is a tracer. <code>scale</code> is a global, set to 2.0 above the function.` },
      { line: 1, note: `<b><code>scale * x</code></b>: Python looks up <code>scale</code> now and finds 2.0, an ordinary number. The jaxpr records <code>mul 2.0:f32[] a</code>: the value, not the name. The compiled program will not know that <code>scale</code> exists.` },
      { line: 2, ret: true, note: `<b><code>return</code></b>. Now press <b>Call it again</b>: the demo first sets <code>scale = 3.0</code>.` },
    ],
    again: `<b>Second call, after <code>scale = 3.0</code>.</b> The cached program still computes <code>mul 2.0:f32[] a</code>, so the result is [[2x]], not [[3x]], and no error says so. Only a new trace, for example on a call with a new shape, would read the new value. Pass anything that can change as an argument.`,
  },
  if: {
    src: H.if_source, header: "{ lambda ; a:f32[]. let", footer: "", eqns: [{ line: 1, text: H.if_partial }],
    call: "fn_if(x)", sees: tracer("float32[]"),
    steps: [
      { line: 0, note: `<b>The call.</b> <code>x</code> is a tracer for a float32 scalar, ${code(tracer("float32[]"))}.` },
      { line: 1, error: H.if_error, note: `<b><code>x > 0</code></b> records a comparison, <code>gt</code>, whose result is another tracer, of type <code>bool[]</code>. Then Python's <code>if</code> asks it for <code>True</code> or <code>False</code>. The tracer has no value to give, so JAX raises <code>TracerBoolConversionError</code> and the trace stops: there is no program to compile. Part 5 shows three ways out; <b>jnp.where</b> is the next button.` },
    ],
  },
  where: {
    src: H.where.source, header: H.where.header, footer: H.where.footer, eqns: H.where.eqns,
    call: "fn_where(x)", sees: tracer("float32[2]"),
    steps: [
      { line: 0, note: `<b>The call.</b> <code>x</code> is a tracer, ${code(tracer("float32[2]"))}.` },
      { line: 1, note: `<b><code>jnp.where(x > 0, x, 0.0)</code></b> records the comparison, then one equation that is itself a call to a jitted function, <code>_where</code>: many <code>jnp</code> functions are jitted, and calling one while tracing records it with its own jaxpr inside. That inner program broadcasts 0.0 to the shape of <code>x</code> and uses <code>select_n</code> to pick, element by element. Both alternatives are in the program; Python never needed the answer.` },
      { line: 2, ret: true, note: `<b><code>return</code></b>: traced with no error. The comparison happens when the compiled program runs, on the actual numbers.` },
    ],
    again: `<b>Second call, same shape:</b> the compiled program runs. For <code>x = [1, −2]</code> it returns <code>[1, 0]</code>; for any other values the same program decides element by element.`,
  },
};

const hero = { fn: "predict", k: 0, error: false, again: 0 };
function heroSteps() { return FUNCS[hero.fn].steps; }
function heroDone() { return hero.error || hero.k >= heroSteps().length; }

function drawHero(note) {
  const F = FUNCS[hero.fn], steps = heroSteps();
  const ran = steps.slice(0, hero.k);
  const ranLines = new Set(ran.map(s => s.line));
  const nowLine = hero.k > 0 ? steps[hero.k - 1].line : -1;
  const pre = hero.again && F.preAgain ? F.preAgain : (F.pre || []);
  const lines = F.src.split("\n");
  let n = 0;
  const srcHtml = pre.map(t => `<span class="ln outside"><span class="no">${++n}</span>${esc(t)}</span>`).join("") +
    lines.map((t, i) => {
      const cls = hero.again ? "todo" : i === nowLine ? "now" : (ranLines.has(i) || i < nowLine) ? "done" : "todo";
      return `<span class="ln ${cls}"><span class="no">${++n}</span>${esc(t)}</span>`;
    }).join("");
  $("#srcPane").innerHTML = srcHtml;

  // the jaxpr so far
  const started = hero.k > 0;
  const recorded = F.eqns.filter(e => ranLines.has(e.line));
  const finished = steps.slice(0, hero.k).some(s => s.ret);
  let jx = "";
  if (started) {
    jx += `<span class="eq${hero.k === 1 && !hero.again ? " fresh" : ""}">${esc(F.header)}</span>`;
    recorded.forEach(e => {
      const fresh = !hero.again && e.line === nowLine;
      jx += `<span class="eq${fresh ? " fresh" : ""}">${esc(e.text.split("\n").map(l => "    " + l).join("\n"))}</span>`;
    });
    if (finished) jx += `<span class="eq${!hero.again && steps[hero.k - 1].ret ? " fresh" : ""}">${esc(F.footer)}</span>`;
    else if (!hero.error) jx += `<span class="eq dim">    …</span>`;
  }
  $("#jxPane").innerHTML = jx || `<span class="eq dim">(nothing yet: the trace has not started)</span>`;

  // console
  const out = [];
  ran.forEach(s => { if (s.console) out.push(esc(s.console)); if (s.error) out.push(`<span class="err">${esc(s.error)}</span>`); });
  for (let i = 0; i < hero.again; i++) out.push(`<span class="quiet">(call ${i + 2}: the compiled program ran; Python printed nothing)</span>`);
  $("#console").innerHTML = out.join("\n") || `<span class="quiet">(nothing printed)</span>`;

  // readouts
  $("#rSees").textContent = hero.again ? "(the Python body does not run)" : started ? F.sees : "–";
  $("#rOps").textContent = started ? String(recorded.length) : "0";
  $("#rPy").textContent = started ? "1" : "0";
  $("#rRuns").textContent = hero.error ? "0 (no program)" : finished ? String(1 + hero.again) : "0";

  // buttons
  $("#stepBtn").disabled = heroDone();
  $("#allBtn").disabled = heroDone();
  $("#againBtn").disabled = !finished || hero.error;
  $$("#fnSeg [data-fn]").forEach(b => b.setAttribute("aria-pressed", b.dataset.fn === hero.fn));

  const msg = note || (hero.k === 0
    ? `Press <b>Run one line</b> to call the jitted <code>${esc(F.call)}</code> for the first time and watch the trace, or <b>Trace it all</b>.`
    : steps[hero.k - 1].note);
  caption($("#heroStatus"), msg);
}
function heroStep() {
  if (heroDone()) return;
  const s = heroSteps()[hero.k];
  hero.k++;
  if (s.error) hero.error = true;
  drawHero();
}
$("#stepBtn").addEventListener("click", heroStep);
$("#allBtn").addEventListener("click", () => {
  while (!heroDone()) { const s = heroSteps()[hero.k]; hero.k++; if (s.error) hero.error = true; }
  drawHero();
});
$("#againBtn").addEventListener("click", () => { hero.again++; drawHero(FUNCS[hero.fn].again); });
$("#resetBtn").addEventListener("click", () => { hero.k = 0; hero.error = false; hero.again = 0; drawHero(); });
$$("#fnSeg [data-fn]").forEach(b => b.addEventListener("click", () => {
  hero.fn = b.dataset.fn; hero.k = 0; hero.error = false; hero.again = 0; drawHero();
}));
drawHero();

/* ============================================================ 2. the cache */
const T = D.timing, W = D.widget;
const CALLS = {
  chapter02: { label: "chapter 02's filter", sig: W.chapter02, st: T.stages_2 },
  newvalues: { label: "same shapes, new values", sig: W.newvalues || W.chapter02, st: T.stages_2 },
  numpy64: { label: "z as a NumPy float64 array", sig: W.numpy64 || W.chapter02, st: T.stages_2 },
  state4: { label: "a 4-state model", sig: W.state4, st: T.stages_4 },
  intz: { label: "z as an int32 array", sig: W.intz, st: T.stages_int },
};
const STAGES = [["trace", "--c5"], ["lower", "--accent"], ["compile", "--warm"], ["run", "--c4"]];
$("#stageLegend").innerHTML = STAGES.map(([n, c]) => `<span><i style="background:var(${c})"></i>${n}</span>`).join("");
const SCALE = Math.max(...Object.values(CALLS).map(c => c.st[0] + c.st[1] + c.st[2] + c.st[3] / 1000));
const cache = { map: new Map(), log: [], calls: 0, misses: 0, total: 0, comp: 0 };
const ms = v => v >= 10 ? `${v.toFixed(0)} ms` : v >= 1 ? `${v.toFixed(1)} ms` : `${v.toFixed(3)} ms`;
const sigText = sig => `(${sig.join(", ")})`;
const grp = n => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

function drawCache(msg) {
  $("#callLog").innerHTML = cache.log.map(r => {
    const w = r.parts.map((v, i) => `<i class="s-${STAGES[i][0]}" style="width:${(100 * v / SCALE).toFixed(3)}%"></i>`).join("");
    return `<div class="row"><span class="what"><b>call ${r.n}</b> · ${esc(r.label)} · <span class="${r.hit ? "hit" : "miss"}">${r.hit ? "cache hit" : "traced and compiled"}</span> · ${ms(r.total)}</span><div class="bar" role="img" aria-label="${r.hit ? "run only" : "trace, lower, compile and run"}: ${ms(r.total)}">${w}</div></div>`;
  }).join("") || `<p class="hint">No calls yet.</p>`;
  $("#cCalls").textContent = String(cache.calls);
  $("#cMiss").textContent = String(cache.misses);
  $("#cTime").textContent = ms(cache.total);
  $("#cComp").textContent = cache.total ? `${ms(cache.comp)} (${100 * cache.comp / cache.total > 99.9 ? "over 99.9" : (100 * cache.comp / cache.total).toFixed(1)}%)` : "0";
  $("#cacheBody").innerHTML = [...cache.map.entries()].map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join("") ||
    `<tr><td colspan="2">empty</td></tr>`;
  caption($("#cacheCap"), msg || "Make a call. The first call with a given signature is traced, lowered and compiled; later calls with the same signature reuse the result.");
}
$$("#callBtns [data-call]").forEach(b => b.addEventListener("click", () => {
  const name = b.dataset.call, c = CALLS[name], key = sigText(c.sig);
  const hit = cache.map.has(key);
  const run = c.st[3] / 1000;
  const parts = hit ? [0, 0, 0, run] : [c.st[0], c.st[1], c.st[2], run];
  const total = parts.reduce((a, v) => a + v, 0);
  cache.calls++; cache.total += total;
  if (!hit) { cache.misses++; cache.comp += total - run; cache.map.set(key, 0); }
  cache.map.set(key, cache.map.get(key) + 1);
  cache.log.unshift({ n: cache.calls, label: c.label, hit, parts, total });
  cache.log = cache.log.slice(0, 7);
  const firstTotal = c.st[0] + c.st[1] + c.st[2] + run;
  let msg;
  if (hit) {
    msg = `<b>A hit.</b> The signature ${code(key)} is in the cache, so the call skips Python entirely and runs the compiled program: ${ms(run)}, about ${grp(firstTotal / run)} times less than a first call. `;
    if (name === "newvalues") msg += "Only the values changed, and values are not part of the key.";
    else if (name === "numpy64") msg += "The NumPy array was float64, but JAX converts it to float32 before looking in the cache (J1, Part 4), so it matches.";
    else msg += "Same shapes and dtypes as before.";
  } else {
    msg = `<b>A miss.</b> No program for ${code(key)} yet: tracing took ${ms(c.st[0])}, lowering ${ms(c.st[1])} and XLA's compilation ${ms(c.st[2])}, then the program ran in ${ms(run)}. `;
    if (name === "state4") msg += "A new shape is a new signature. Both programs now sit in the cache side by side.";
    else if (name === "intz") msg += "A new dtype is a new signature too, even though the arithmetic inside promotes <code>z</code> to float32.";
    else if (cache.misses === 1) msg += "Every later call with this signature will skip all of that.";
    else msg += "The cache was empty for this signature, so the work is done again.";
  }
  drawCache(msg);
}));
$("#clearBtn").addEventListener("click", () => {
  cache.map.clear(); cache.log = []; cache.calls = 0; cache.misses = 0; cache.total = 0; cache.comp = 0;
  drawCache("<b>A new Python process starts with an empty cache</b>, so its first call with each signature compiles again. (JAX can also keep compiled programs on disk between processes, its persistent compilation cache, which this chapter does not cover.)");
});
drawCache();

/* ============================================================ 3. the 1000-step timing bars */
const LO = Math.log10(0.5), HI = Math.log10(1000);
$$("#runT tr[data-k]").forEach(tr => {
  const v = T[tr.dataset.k];
  if (typeof v !== "number") return;
  tr.querySelector("td.n").textContent = ms(v);
  const w = Math.max(0, Math.min(1, (Math.log10(v) - LO) / (HI - LO)));
  tr.querySelector(".tb").style.width = `${(100 * w).toFixed(1)}%`;
});
})();
