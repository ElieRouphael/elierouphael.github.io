/* Kalman filters with JAX, chapter J1, Arrays and the numbers inside them: the live demos.
   1. A float explorer: the bits of a float32 or float64, the exact decimal value they store, the error
      against what was typed, and the gap to the next float, all computed exactly with BigInt.
   2. A broadcasting checker, preloaded with the shapes kalman_jax/filters.py combines.
   3. Adding 0.1 N times in float32: a sequential loop and Kahan's compensated loop computed here with
      Math.fround (checked against JAX's lax.fori_loop results in j1-data.js), against jnp.sum and float64.
   Needs ../kit/kit.js and j1-data.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg } = window.Kit;
const D = window.J1;

/* ============================================================ exact floats */
const FMT = {
  f32: { bits: 32, expBits: 8, fracBits: 23, bias: 127, name: "float32" },
  f64: { bits: 64, expBits: 11, fracBits: 52, bias: 1023, name: "float64" },
};
const pow5 = k => 5n ** BigInt(k);
const pow10 = k => 10n ** BigInt(k);

/* The fields of v stored in format f: sign, biased exponent, fraction (BigInt), and the bit string. */
function decompose(v, f) {
  const dv = new DataView(new ArrayBuffer(8));
  let sign, exp, frac, bitstr;
  if (f.bits === 32) {
    dv.setFloat32(0, v);
    const u = dv.getUint32(0);
    sign = u >>> 31; exp = (u >>> 23) & 0xff; frac = BigInt(u & 0x7fffff);
    bitstr = u.toString(2).padStart(32, "0");
  } else {
    dv.setFloat64(0, v);
    const hi = dv.getUint32(0), lo = dv.getUint32(4);
    sign = hi >>> 31; exp = (hi >>> 20) & 0x7ff; frac = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
    bitstr = hi.toString(2).padStart(32, "0") + lo.toString(2).padStart(32, "0");
  }
  return { sign, exp, frac, bitstr };
}
/* The value as an exact decimal: { neg, digits (BigInt), p } meaning (-1)^neg * digits * 10^p; null for inf/nan. */
function exactValue(d, f) {
  const maxExp = (1 << f.expBits) - 1;
  if (d.exp === maxExp) return null;
  const m = d.exp === 0 ? d.frac : d.frac + (1n << BigInt(f.fracBits));
  const e2 = (d.exp === 0 ? 1 : d.exp) - f.bias - f.fracBits;           // value = m * 2^e2
  if (e2 >= 0) return { neg: d.sign === 1, digits: m << BigInt(e2), p: 0 };
  return { neg: d.sign === 1, digits: m * pow5(-e2), p: e2 };            // m * 2^e2 = m * 5^-e2 * 10^e2
}
/* The gap between v and the next float of larger magnitude: one unit in the last place. */
function gapOf(d, f) {
  return 2 ** ((d.exp === 0 ? 1 : d.exp) - f.bias - f.fracBits);
}
/* Parse a decimal literal exactly. */
function parseDecimal(text) {
  const t = text.replace(/[\s_]/g, "").replace(/−/g, "-");
  const mt = /^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(t);
  if (!mt || (!mt[2] && !mt[3])) return null;
  const intPart = mt[2] || "", fracPart = mt[3] || "";
  return { neg: mt[1] === "-", digits: BigInt((intPart + fracPart) || "0"), p: -fracPart.length + (mt[4] ? +mt[4] : 0) };
}
function decString(x, maxDigits = 120) {          // exact decimal string of { neg, digits, p }
  let s = x.digits.toString(), out;
  if (x.p >= 0) out = s + "0".repeat(x.p);
  else {
    const k = -x.p;
    if (s.length <= k) s = "0".repeat(k - s.length + 1) + s;
    out = (s.slice(0, s.length - k) + "." + s.slice(s.length - k)).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (out.length > maxDigits) out = out.slice(0, maxDigits) + "…";
  return (x.neg && x.digits !== 0n ? "−" : "") + out;
}
function sub(a, b) {                              // exact a - b
  const p = Math.min(a.p, b.p);
  const av = (a.neg ? -1n : 1n) * a.digits * pow10(a.p - p), bv = (b.neg ? -1n : 1n) * b.digits * pow10(b.p - p);
  const d = av - bv;
  return { neg: d < 0n, digits: d < 0n ? -d : d, p };
}
const toNum = x => (x.neg ? -1 : 1) * Number(`${x.digits}e${x.p}`);
function sci(v, d = 2) {
  if (v === 0) return "0";
  if (!isFinite(v)) return String(v);
  const e = Math.floor(Math.log10(Math.abs(v))), m = v / 10 ** e;
  if (e >= -3 && e <= 5) return String(+v.toPrecision(d + 1));
  return `${(+m.toFixed(d)).toString()} × 10${sup(e)}`;
}
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const sup = e => String(e).split("").map(c => SUP[c]).join("");

/* ============================================================ 1. the float explorer */
const fx = { text: "0.1", f: "f32" };
function drawFloat() {
  const f = FMT[fx.f], typed = parseDecimal(fx.text);
  const input = $("#fxIn");
  input.setAttribute("aria-invalid", typed ? "false" : "true");
  if (!typed) {
    ["#rStored", "#rErr", "#rGap", "#rDigits", "#rForm"].forEach(id => { $(id).textContent = "–"; });
    $("#bits").innerHTML = ""; $("#bitLegend").innerHTML = "";
    caption($("#heroStatus"), "Type a decimal number such as 0.1, 33.7, 16777217 or 1e-45.");
    return;
  }
  const v = Number(fx.text.replace(/[\s_]/g, "").replace(/−/g, "-"));
  const stored = f.bits === 32 ? Math.fround(v) : v;
  const d = decompose(stored, f), ex = exactValue(d, f);
  const s = d.bitstr, eb = f.expBits;
  const cell = (bit, cls, title) => `<span class="bit ${cls}" title="${title}">${bit}</span>`;
  $("#bits").innerHTML =
    `<span class="bgroup">${cell(s[0], "b-s", "sign")}</span>` +
    `<span class="bgroup">${s.slice(1, 1 + eb).split("").map(b => cell(b, "b-e", "exponent")).join("")}</span>` +
    `<span class="bgroup">${s.slice(1 + eb).split("").map(b => cell(b, "b-f", "fraction")).join("")}</span>`;
  const maxExp = (1 << eb) - 1, sub_ = d.exp === 0, E = (sub_ ? 1 : d.exp) - f.bias;
  const sig = Number(d.frac) / 2 ** f.fracBits + (sub_ ? 0 : 1);
  $("#bitLegend").innerHTML =
    `<span><i class="b-s"></i>sign ${d.sign}</span>` +
    `<span><i class="b-e"></i>exponent ${d.exp}${d.exp === maxExp ? " (all ones: infinity or NaN)" : sub_ ? " (all zeros: subnormal)" : ` − ${f.bias} = ${E}`}</span>` +
    `<span><i class="b-f"></i>fraction ${d.exp === maxExp ? "" : (sub_ ? "0." : "1.") + "…" + " = " + sig.toPrecision(f.bits === 32 ? 9 : 17)}</span>`;
  if (!ex) {
    $("#rStored").textContent = isNaN(stored) ? "NaN" : (stored > 0 ? "+∞" : "−∞");
    ["#rErr", "#rGap", "#rDigits"].forEach(id => { $(id).textContent = "–"; });
    $("#rForm").textContent = "–";
    caption($("#heroStatus"), `${fx.text} is beyond the largest ${f.name} (${f.bits === 32 ? "3.4 × 10³⁸" : "1.8 × 10³⁰⁸"}), so the conversion overflows to infinity: the exponent field is all ones and the fraction zero. Arithmetic that overflows does the same, and infinity minus infinity is NaN.`);
    return;
  }
  const err = sub(ex, typed), errN = toNum(err), typedN = toNum(typed), gap = gapOf(d, f);
  $("#rStored").textContent = decString(ex);
  $("#rErr").textContent = errN === 0 ? "0 (exact)" : `${errN > 0 ? "+" : "−"}${sci(Math.abs(errN))}` + (typedN !== 0 ? ` (relative ${sci(Math.abs(errN / typedN))})` : "");
  $("#rGap").textContent = sci(gap);
  const digits = stored === 0 ? null : Math.log10(Math.abs(stored) / gap);
  $("#rDigits").textContent = digits == null ? "–" : `about ${Math.max(0, Math.floor(digits))}`;
  $("#rForm").textContent = sub_ ? `(−1)^${d.sign} × ${sig.toPrecision(6)} × 2^${E}` : `(−1)^${d.sign} × ${sig.toPrecision(f.bits === 32 ? 9 : 12)} × 2^${E}`;
  const isInt = typed.p >= 0 || (typed.digits % pow10(-typed.p) === 0n);
  let msg;
  if (stored === 0 && typed.digits !== 0n) msg = `${fx.text} is smaller than half the smallest positive ${f.name}, so it rounds to zero: underflow. The smallest positive ${f.name} is ${f.bits === 32 ? "1.4 × 10⁻⁴⁵" : "4.9 × 10⁻³²⁴"}, a subnormal number with a single 1 at the end of the fraction.`;
  else if (sub_ && stored !== 0) msg = `This is a subnormal ${f.name}: the exponent field is all zeros, the leading 1 is dropped, and precision is lost gradually as numbers approach zero. Only ${Math.max(1, Math.floor(digits + 1e-9))} significant decimal digit${digits >= 2 ? "s are" : " is"} left. Error: ${errN === 0 ? "none" : sci(Math.abs(errN))}.`;
  else if (errN === 0) msg = `${fx.text} is stored exactly: it is a whole number times a power of two that fits in ${f.fracBits + 1} significant bits. Most decimal fractions are not. Try 0.1. The next ${f.name} up is ${sci(gap)} away.`;
  else if (isInt) msg = `${fx.text} is a whole number, but above ${f.bits === 32 ? "2²⁴ = 16 777 216" : "2⁵³ = 9 007 199 254 740 992"} ${f.name} has no room for the units: neighbouring ${f.name}s are ${sci(gap)} apart here, and ${fx.text} is stored as ${decString(ex)}. Integers beyond that point are rounded like any other number.`;
  else msg = `${fx.text} has no finite binary expansion, so ${f.name} stores the nearest number it can: ${decString(ex, 40)}. The error, ${sci(Math.abs(errN))}, is at most half the gap of ${sci(gap)} between neighbouring ${f.name}s here, a relative error below ${f.bits === 32 ? "2⁻²⁴ ≈ 6 × 10⁻⁸" : "2⁻⁵³ ≈ 1.1 × 10⁻¹⁶"}. That gap grows with the size of the number: it doubles at every power of two.`;
  caption($("#heroStatus"), msg);
  $$("#fxPresets [data-v]").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === fx.text));
}
$("#fxIn").addEventListener("input", e => { fx.text = e.target.value; drawFloat(); });
$$("#fxPresets [data-v]").forEach(b => b.addEventListener("click", () => { fx.text = b.dataset.v; $("#fxIn").value = fx.text; drawFloat(); }));
$$("#fmtSeg [data-f]").forEach(b => b.addEventListener("click", () => {
  fx.f = b.dataset.f; $$("#fmtSeg [data-f]").forEach(x => x.setAttribute("aria-pressed", x === b)); drawFloat();
}));

/* ============================================================ 2. broadcasting */
function parseShape(text) {
  const t = text.trim().replace(/^\(|\)$/g, "").trim();
  if (t === "") return [];
  const parts = t.split(",").map(s => s.trim()).filter(s => s !== "");
  if (!parts.every(p => /^\d+$/.test(p))) return null;
  return parts.map(Number);
}
const pyShape = s => s.length === 1 ? `(${s[0]},)` : `(${s.join(", ")})`;
function broadcast(a, b) {
  const n = Math.max(a.length, b.length), A = Array(n - a.length).fill(null).concat(a), B = Array(n - b.length).fill(null).concat(b);
  const out = [], status = [];
  for (let i = 0; i < n; i++) {
    const x = A[i] ?? 1, y = B[i] ?? 1;
    if (x === y) { out.push(x); status.push("same"); }
    else if (x === 1) { out.push(y); status.push("a"); }
    else if (y === 1) { out.push(x); status.push("b"); }
    else { out.push(null); status.push("bad"); }
  }
  return { A, B, out, status, ok: !status.includes("bad") };
}
const bc = { a: "(5, 1)", b: "(5, 2)", label: "Wm[:, None] * transformed_points" };
function drawBroadcast() {
  const a = parseShape(bc.a), b = parseShape(bc.b);
  $("#shA").setAttribute("aria-invalid", a ? "false" : "true"); $("#shB").setAttribute("aria-invalid", b ? "false" : "true");
  if (!a || !b) { caption($("#bcCap"), "Write a shape as a tuple of whole numbers, such as (5, 1), (2,) or () for a scalar."); $("#bcGrid").innerHTML = ""; return; }
  const r = broadcast(a, b), n = r.out.length;
  const cellA = (v, st) => `<td class="${v == null ? "pad" : st === "a" ? "stretch" : st === "bad" ? "bad" : ""}">${v == null ? "·" : v}</td>`;
  const cellB = (v, st) => `<td class="${v == null ? "pad" : st === "b" ? "stretch" : st === "bad" ? "bad" : ""}">${v == null ? "·" : v}</td>`;
  $("#bcGrid").innerHTML = `<table class="bcT"><tbody>
    <tr><th>left ${pyShape(a)}</th>${r.A.map((v, i) => cellA(v, r.status[i])).join("")}</tr>
    <tr><th>right ${pyShape(b)}</th>${r.B.map((v, i) => cellB(v, r.status[i])).join("")}</tr>
    <tr class="res"><th>result</th>${r.out.map(v => `<td class="${v == null ? "bad" : ""}">${v == null ? "✕" : v}</td>`).join("")}</tr>
  </tbody></table>`;
  const stretched = r.status.map((s, i) => s === "a" || s === "b" ? i : -1).filter(i => i >= 0);
  const padded = r.A.filter(v => v == null).length + r.B.filter(v => v == null).length;
  let msg;
  if (r.ok) msg = `${bc.label ? `<code>${bc.label}</code>: ` : ""}the shapes are lined up from the right${padded ? `, the shorter one padded on the left with axes of length 1 (dots)` : ""}. Along every axis the lengths are equal or one of them is 1, and a length-1 axis is stretched to match (highlighted) without copying any data. The result has shape <b>${pyShape(r.out)}</b>${stretched.length ? `: ${n === 0 ? "" : `${stretched.length} ax${stretched.length > 1 ? "es" : "is"} stretched`}.` : "."}`;
  else {
    const bad = r.status.map((s, i) => s === "bad" ? i : -1).filter(i => i >= 0);
    const many = bad.length > 1;
    msg = `${bc.label ? `<code>${bc.label}</code>: ` : ""}lined up from the right, ax${many ? "es" : "is"} ${bad.map(i => (i - n).toString().replace("-", "−")).join(" and ")} (counting from the end) ${many ? "have" : "has"} lengths ${bad.map(i => `${r.A[i]} and ${r.B[i]}`).join(", and ")}: ${many ? "on each, neither length" : "neither"} is 1, so there is no way to stretch one onto the other and JAX raises an error. ${bc.label === "Wm * transformed_points" ? "This is the line the repository does <em>not</em> write: the weights must become a column, <code>Wm[:, None]</code>, so that weight i multiplies row i." : "Add an axis of length 1 with <code>None</code> where you want stretching."}`;
  }
  caption($("#bcCap"), msg);
  $$("#bcPresets [data-a]").forEach(btn => btn.setAttribute("aria-pressed", btn.dataset.a === bc.a && btn.dataset.b === bc.b));
}
$("#shA").addEventListener("input", e => { bc.a = e.target.value; bc.label = ""; drawBroadcast(); });
$("#shB").addEventListener("input", e => { bc.b = e.target.value; bc.label = ""; drawBroadcast(); });
$$("#bcPresets [data-a]").forEach(btn => btn.addEventListener("click", () => {
  bc.a = btn.dataset.a; bc.b = btn.dataset.b; bc.label = btn.dataset.label || "";
  $("#shA").value = bc.a; $("#shB").value = bc.b; drawBroadcast();
}));

/* ============================================================ 3. adding 0.1, N times */
const fr32 = Math.fround, TENTH = fr32(0.1), NMAX = 1e7;
/* One pass to 10^7: the float32 sequential sum, Kahan's compensated sum (both rounded to float32 after every
   operation, as JAX's float32 loop does), the float64 sequential sum, and a sample of the running error. */
const S = (() => {
  let s = 0, k = 0, c = 0, d = 0;
  const at = {}, trace = [];
  let nextSample = 1;
  for (let i = 1; i <= NMAX; i++) {
    s = fr32(s + TENTH);
    const y = fr32(TENTH - c), t = fr32(k + y); c = fr32(fr32(t - k) - y); k = t;
    d += 0.1;
    if (i === nextSample || i === NMAX) { trace.push([i, s]); nextSample = Math.max(i + 1, Math.round(i * 1.02)); }
    if (Number.isInteger(Math.log10(i)) && i >= 10) at[i] = { seq: s, kahan: k, f64: d };
  }
  return { at, trace };
})();
const methods = [
  { key: "seq", label: "float32 loop", color: () => css("--warm") },
  { key: "jsum", label: "jnp.sum (JAX)", color: () => css("--accent") },
  { key: "kahan", label: "Kahan, float32", color: () => css("--c4") },
  { key: "f64", label: "float64 loop", color: () => css("--ink"), dash: [5, 4] },
];
const grp = (v, d = 7) => v.toLocaleString("en-US", { maximumFractionDigits: d }).replace(/,/g, " ");
const value = (row, key) => key === "jsum" ? row.jsum : S.at[row.n][key];
const sm = { e: 7 };
const relErr = (v, n) => Math.abs(v - n / 10) / (n / 10);
const FLOOR = -17;
function drawSum() {
  const n = 10 ** sm.e, row = D.sums.find(r => r.n === n), truth = n / 10;
  /* chart A: final relative error against N, log-log */
  const ca = $("#sumC"), narrow = narrowOf(ca);
  const fa = frame(ca, narrow ? 1.25 : 2.0, { x: [1, 7], y: [FLOOR, 0], xlabel: "N (additions)", ylabel: "relative error, log10", xticks: [1, 2, 3, 4, 5, 6, 7], xfmt: v => `10${sup(v)}`, yticks: [-17, -14, -11, -8, -5, -2], yfmt: v => v === FLOOR ? "exact" : `10${sup(v)}` });
  for (const m of methods) {
    const xs = D.sums.map(r => Math.log10(r.n)), ys = D.sums.map(r => { const e = relErr(value(r, m.key), r.n); return e === 0 ? FLOOR : Math.max(FLOOR, Math.log10(e)); });
    plotLine(fa, xs, ys, { color: m.color(), width: 2, dash: m.dash });
    fa.ctx.fillStyle = m.color();
    xs.forEach((x, i) => { fa.ctx.beginPath(); fa.ctx.arc(fa.X(x), fa.Y(ys[i]), x === sm.e ? 4.5 : 2.5, 0, 2 * Math.PI); fa.ctx.fill(); });
  }
  fa.ctx.strokeStyle = alpha("--muted", 0.6); fa.ctx.setLineDash([2, 3]); seg(fa.ctx, fa.X(sm.e), fa.pad.t, fa.X(sm.e), fa.Y(FLOOR)); fa.ctx.setLineDash([]);
  /* chart B: the float32 loop's running sum minus the exact running sum, up to N */
  const cb = $("#runC"), pts = S.trace.filter(([i]) => i <= n);
  const errs = pts.map(([i, s]) => s - i / 10), lo = Math.min(0, ...errs), hi = Math.max(0, ...errs), pad = 0.08 * (hi - lo || 1e-6);
  const fb = frame(cb, narrowOf(cb) ? 1.6 : 2.6, { x: [0, n], y: [lo - pad, hi + pad], xlabel: "additions so far", ylabel: "float32 loop minus exact", yfmt: v => Math.abs(v) >= 1000 ? `${+(v / 1000).toPrecision(3)}k` : String(+v.toPrecision(3)), xfmt: v => v >= 1e6 ? `${v / 1e6}M` : v >= 1e3 ? `${v / 1e3}k` : String(v) });
  fb.ctx.strokeStyle = css("--muted"); seg(fb.ctx, fb.pad.l, fb.Y(0), fb.w - fb.pad.r, fb.Y(0));
  plotLine(fb, pts.map(p => p[0]), errs, { color: css("--warm"), width: 2 });
  /* readouts */
  const show = (id, v) => { $(id).textContent = grp(v); };
  $("#sNv").textContent = grp(n);
  show("#sTrue", truth); show("#sSeq", value(row, "seq")); show("#sJ", row.jsum); show("#sK", value(row, "kahan")); show("#sD", value(row, "f64"));
  const okSeq = value(row, "seq") === row.seq, okK = value(row, "kahan") === row.kahan;
  $("#sCheck").textContent = okSeq && okK ? "match JAX exactly" : "differ from JAX";
  const seqV = value(row, "seq"), gapEnd = 2 ** (Math.floor(Math.log2(seqV)) - 23);
  caption($("#sumCap"), `Adding the float32 number 0.1 ${grp(n)} times; the exact answer is ${grp(truth)}. The sequential float32 loop, what a Python loop or <code>lax.fori_loop</code> does, ends at ${grp(seqV)}, ${relErr(seqV, n) < 1e-6 ? "close" : `off by ${fmt(100 * relErr(seqV, n), relErr(seqV, n) < 0.01 ? 3 : 1)}%`}. Near its final value neighbouring float32s are ${sci(gapEnd, 3)} apart, so every addition is rounded to a whole number of those gaps. ${seqV > 524288 ? "Above 524 288 the gap is 0.0625 and each 0.1 is rounded to 0.125, a quarter too much, which is why the curve below climbs steeply at the end. " : seqV > 32768 ? "Above 32 768 the gap is 2⁻⁸, then 2⁻⁷, and each 0.1 is rounded to 0.1015625, 1.6% too much, which is why the curve below climbs at the end. " : ""}<code>jnp.sum</code> does not add the numbers one after another: the compiled reduction combines partial sums, so no single running total grows large while the numbers added to it stay small, and its error stays near float32's resolution. Kahan's loop carries the rounding error of each addition in a second variable and feeds it back; in float32 it gets ${grp(value(row, "kahan"))}. The page's float32 loops ${okSeq && okK ? "match" : "do not match"} JAX's <code>lax.fori_loop</code> results bit for bit.`);
  $("#sumLegend").innerHTML = methods.map(m => `<span><i style="background:${m.color()}"></i>${m.label}</span>`).join("");
}
$("#sumN").addEventListener("input", e => { sm.e = +e.target.value; drawSum(); });

drawFloat(); drawBroadcast(); drawSum();
onRedraw(drawSum);
})();
