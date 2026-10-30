/* Satellite Pass Planning from Scratch, chapter 01, Circular Orbits from Newton: the live demos.
   The equation of motion r'' = -mu r / |r|^3 is integrated in the browser with explicit Euler, symplectic
   Euler and RK4, exactly as code/ch01_numbers.py does in NumPy, and compared with the closed-form circle of
   satplan (Sat.positionECI) or, for a non-circular start, with the Kepler ellipse. A second widget shows how
   speed and period change with altitude, a third how the error of each method shrinks with the step.
   Needs ../kit/kit.js and sat.js. */
(() => {
"use strict";
const { $, css, fmt, caption, fit, frame, plotLine, alpha, onRedraw, narrowOf } = window.Kit;
const Sat = window.Sat;
const { MU_EARTH, R_EARTH, OMEGA_EARTH } = Sat;

const ALT = 550e3;
const ORB = Sat.orbit(ALT, 0, 0, 0);          // the package's circle in the equatorial plane, phase 0
const A0 = ORB.a, T0 = ORB.period, VC = Math.sqrt(MU_EARTH / A0);
const H_LIST = [1, 2, 5, 10, 20, 30, 60, 120, 240];
const METHODS = [
  { key: "euler", name: "explicit Euler", token: "--warm", evals: 1 },
  { key: "symplectic", name: "symplectic Euler", token: "--c4", evals: 1 },
  { key: "rk4", name: "RK4", token: "--post", evals: 4 },
];

/* ---------------- the physics ---------------- */
function acc(x, y) { const d2 = x * x + y * y, d = Math.sqrt(d2), k = -MU_EARTH / (d2 * d); return [k * x, k * y]; }
const energy = s => 0.5 * (s[2] * s[2] + s[3] * s[3]) - MU_EARTH / Math.hypot(s[0], s[1]);
/* One step of each method on the state s = [x, y, vx, vy], in place (the same updates as ch01_numbers.py). */
const STEP = {
  euler(s, h) {        // both updates use the old state
    const [ax, ay] = acc(s[0], s[1]);
    s[0] += h * s[2]; s[1] += h * s[3]; s[2] += h * ax; s[3] += h * ay;
  },
  symplectic(s, h) {   // kick the velocity with the old position, then drift with the new velocity
    const [ax, ay] = acc(s[0], s[1]);
    s[2] += h * ax; s[3] += h * ay; s[0] += h * s[2]; s[1] += h * s[3];
  },
  rk4(s, h) {
    const x = s[0], y = s[1], vx = s[2], vy = s[3];
    const [a1x, a1y] = acc(x, y);
    const v2x = vx + h / 2 * a1x, v2y = vy + h / 2 * a1y, [a2x, a2y] = acc(x + h / 2 * vx, y + h / 2 * vy);
    const v3x = vx + h / 2 * a2x, v3y = vy + h / 2 * a2y, [a3x, a3y] = acc(x + h / 2 * v2x, y + h / 2 * v2y);
    const v4x = vx + h * a3x, v4y = vy + h * a3y, [a4x, a4y] = acc(x + h * v3x, y + h * v3y);
    s[0] = x + h / 6 * (vx + 2 * v2x + 2 * v3x + v4x); s[1] = y + h / 6 * (vy + 2 * v2y + 2 * v3y + v4y);
    s[2] = vx + h / 6 * (a1x + 2 * a2x + 2 * a3x + a4x); s[3] = vy + h / 6 * (a1y + 2 * a2y + 2 * a3y + a4y);
  },
};
/* The exact orbit that starts at (A0, 0) with speed k * VC along +y: the package's circle for k = 1, otherwise
   a Kepler ellipse (perigee at the start if k > 1, apogee if k < 1), solved with Newton's method. */
function reference(k) {
  const v0 = k * VC, ae = 1 / (2 / A0 - v0 * v0 / MU_EARTH), e = Math.abs(k * k - 1);
  const n = Math.sqrt(MU_EARTH / (ae * ae * ae)), peri = k >= 1, circle = e < 1e-15;
  function at(t) {
    if (circle) { const p = Sat.positionECI(ORB, t); return { r: A0, x: p[0], y: p[1] }; }
    const M = n * t + (peri ? 0 : Math.PI);
    let E = M;
    for (let i = 0; i < 50; i++) { const d = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E)); E -= d; if (Math.abs(d) < 1e-14) break; }
    const r = ae * (1 - e * Math.cos(E));
    const nu = 2 * Math.atan2(Math.sqrt(1 + e) * Math.sin(E / 2), Math.sqrt(1 - e) * Math.cos(E / 2));
    const ang = peri ? nu : nu - Math.PI;
    return { r, x: r * Math.cos(ang), y: r * Math.sin(ang) };
  }
  return { k, circle, ae, e, T: circle ? T0 : 2 * Math.PI / n, rp: circle ? A0 : ae * (1 - e), ra: circle ? A0 : ae * (1 + e), at };
}
/* Integrate `revs` periods of the reference orbit with step h. Records a decimated path and, per decimation
   bucket, the largest relative energy error and radius error (so the log plots show the envelope). */
function integrate(method, h, revs, ref, light = false) {
  const nSteps = Math.ceil(revs * ref.T / h - 1e-9), stride = Math.max(1, Math.ceil(nSteps / 1200));
  const s = [A0, 0, 0, ref.k * VC], E0 = energy(s);
  const path = [[A0, 0]], tt = [], logE = [], logR = [];
  let maxE = 0, maxR = 0, bE = 0, bR = 0, hit = null, rmax = A0, done = 0;
  for (let i = 1; i <= nSteps; i++) {
    STEP[method](s, h); done = i;
    const d = Math.hypot(s[0], s[1]);
    if (d < R_EARTH) { hit = i * h; if (!light) path.push([s[0], s[1]]); break; }
    if (d > rmax) rmax = d;
    const dE = Math.abs(energy(s) - E0) / Math.abs(E0);
    if (dE > maxE) maxE = dE;
    if (light) continue;
    const dR = Math.abs(d - ref.at(i * h).r);
    if (dR > maxR) maxR = dR;
    if (dE > bE) bE = dE;
    if (dR > bR) bR = dR;
    if (i % stride === 0 || i === nSteps) {
      path.push([s[0], s[1]]); tt.push(i * h / ref.T);
      logE.push(Math.log10(Math.max(bE, 1e-16))); logR.push(Math.log10(Math.max(bR, 1e-3)));
      bE = 0; bR = 0;
    }
  }
  const tEnd = hit != null ? hit : nSteps * h, p = ref.at(tEnd);
  return { method, h, nSteps, hit, tEnd, path, tt, logE, logR, maxE, maxR, rmax,
           dEend: (energy(s) - E0) / Math.abs(E0), posErr: Math.hypot(s[0] - p.x, s[1] - p.y),
           evals: done * (method === "rk4" ? 4 : 1) };
}

/* ---------------- formatting ---------------- */
const sci = (v, d = 1) => (v == null || !isFinite(v)) ? "–" : v === 0 ? "0" : v.toExponential(d).replace("e+", "e").replace("e-", "e−");
function len(m) {   // a distance in metres, in readable units
  const a = Math.abs(m);
  if (a >= 1e5) return Math.round(m / 1e3).toLocaleString("en-US") + " km";
  if (a >= 1e3) return fmt(m / 1e3, a >= 1e4 ? 1 : 2) + " km";
  if (a >= 1) return fmt(m, a >= 100 ? 0 : 1) + " m";
  if (a >= 1e-2) return fmt(m * 100, 1) + " cm";
  return fmt(m * 1000, 2) + " mm";
}
const LEN_LABELS = { "-3": "1 mm", "-2": "1 cm", "-1": "10 cm", "0": "1 m", "1": "10 m", "2": "100 m", "3": "1 km", "4": "10 km", "5": "100 km", "6": "1000 km", "7": "10⁴ km", "8": "10⁵ km" };
const SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
const pow10 = v => "10" + String(Math.round(v)).split("").map(c => SUP[c] || c).join("");
function intTicks(lo, hi, maxCount = 6) {
  const step = Math.max(1, Math.ceil((hi - lo) / maxCount)), out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
  return out;
}

/* ================= hero: three integrators ================= */
const st = { hIdx: 6, revs: 5, k: 1, show: { euler: true, symplectic: true, rk4: true } };
let runs = null, ref = null;
function computeHero() {
  ref = reference(st.k);
  runs = METHODS.map(m => integrate(m.key, H_LIST[st.hIdx], st.revs, ref));
}
function drawOrbit() {
  const cv = $("#orbC"), { ctx, w, h } = fit(cv, 1);
  const shown = runs.filter(r => st.show[r.method]);
  let L = ref.ra;
  for (const r of shown) L = Math.max(L, Math.min(r.rmax, 2.6 * ref.ra));
  L *= 1.06;
  const S = (Math.min(w, h) / 2 - 6) / L, cx = w / 2, cy = h / 2;
  const P = (x, y) => [cx + x * S, cy - y * S];
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip();
  // axes
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, cy + 0.5); ctx.lineTo(w, cy + 0.5); ctx.moveTo(cx + 0.5, 0); ctx.lineTo(cx + 0.5, h); ctx.stroke();
  // Earth
  ctx.beginPath(); ctx.arc(cx, cy, R_EARTH * S, 0, 2 * Math.PI);
  ctx.fillStyle = alpha("--accent", 0.13); ctx.fill(); ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1; ctx.stroke();
  // exact orbit
  ctx.beginPath();
  for (let j = 0; j <= 240; j++) { const p = ref.at(ref.T * j / 240), [px, py] = P(p.x, p.y); j ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);
  // trajectories, RK4 last so it stays on top of the exact orbit it hides
  for (const r of shown) {
    const m = METHODS.find(q => q.key === r.method);
    ctx.beginPath();
    r.path.forEach(([x, y], j) => { const [px, py] = P(x, y); j ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.strokeStyle = css(m.token); ctx.lineWidth = r.method === "rk4" ? 2 : 1.5; ctx.lineJoin = "round"; ctx.stroke();
    const [ex, ey] = P(...r.path[r.path.length - 1]);
    ctx.fillStyle = css(m.token); ctx.beginPath(); ctx.arc(ex, ey, 4, 0, 2 * Math.PI); ctx.fill();
  }
  // start point and velocity arrow
  const [sx, sy] = P(A0, 0);
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(sx, sy, 3.5, 0, 2 * Math.PI); ctx.fill();
  const ar = Math.min(w, h) * 0.12 * st.k;
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx, sy - ar); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(sx - 4, sy - ar + 6); ctx.lineTo(sx, sy - ar); ctx.lineTo(sx + 4, sy - ar + 6); ctx.stroke();
  ctx.font = "11px " + css("--font-mono"); ctx.fillStyle = css("--muted"); ctx.textAlign = "center";
  ctx.fillText("Earth", cx, cy + 4);
  ctx.textAlign = "left"; ctx.fillText("start, v₀", Math.min(sx + 6, w - 64), sy - ar + 4);
  ctx.textAlign = "right"; ctx.fillText("scale: " + Math.round(2 * L / 1e3).toLocaleString("en-US") + " km across", w - 4, h - 6);
  ctx.restore();
}
function drawErr(id, key, fmtY, floor, ylabel) {
  const cv = $(id), narrow = narrowOf(cv, 300);
  const shown = runs.filter(r => st.show[r.method] && r[key].length);
  let lo = Infinity, hi = -Infinity;
  for (const r of shown) for (const v of r[key]) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo)) { lo = floor; hi = floor + 2; }
  lo = Math.max(floor, Math.floor(lo)); hi = Math.max(lo + 2, Math.ceil(hi));
  const fr = frame(cv, narrow ? 1.5 : 1.75, { x: [0, st.revs], y: [lo, hi], yticks: intTicks(lo, hi, narrow ? 4 : 5), yfmt: fmtY,
    xlabel: "time [revolutions]", ylabel, pad: { l: 52 } });
  for (const r of shown) {
    const m = METHODS.find(q => q.key === r.method);
    plotLine(fr, r.tt, r[key], { color: css(m.token), width: 1.6 });
  }
  return fr;
}
function drawTable() {
  const rows = runs.map(r => {
    const m = METHODS.find(q => q.key === r.method);
    return `<tr><td><span class="sw" style="background:var(${m.token})"></span> ${m.name}</td><td>${sci(r.maxE)}</td><td>${len(r.maxR)}</td><td>${r.hit != null ? "hit the Earth" : len(r.posErr)}</td><td>${r.evals.toLocaleString("en-US")}</td></tr>`;
  });
  $("#hTab tbody").innerHTML = rows.join("");
}
function drawHero() {
  if (!runs) computeHero();
  drawOrbit();
  drawErr("#errE", "logE", v => pow10(v), -16, "relative energy error |ΔE/E₀|");
  drawErr("#errR", "logR", v => LEN_LABELS[String(v)] || pow10(v) + " m", -3, "radius error ||r| − r_exact|");
  drawTable();
  const h = H_LIST[st.hIdx], [eu, sy, rk] = runs, perRev = ref.T / h;
  $("#hHV").textContent = h + " s"; $("#hNV").textContent = st.revs; $("#hKV").textContent = fmt(st.k, 2);
  $("#rSteps").textContent = `${eu.nSteps.toLocaleString("en-US")} (${fmt(perRev, 1)} per rev.)`;
  $("#rAng").textContent = fmt(360 / perRev, 2) + "°";
  $("#rV0").textContent = fmt(st.k * VC, 0) + " m/s";
  $("#rRef").textContent = ref.circle ? "circle, a = " + fmt(A0 / 1e3, 1) + " km" : `${fmt((ref.rp - R_EARTH) / 1e3, 0)} × ${fmt((ref.ra - R_EARTH) / 1e3, 0)} km alt.`;
  $("#rT").textContent = fmt(ref.T / 60, 1) + " min";
  let txt;
  const refTxt = ref.circle
    ? `the package's closed-form circle [[r(t) = a(\\cos nt, \\sin nt)]] (black dashed, <code>position_eci</code> with [[i = \\Omega = 0]]), radius ${fmt(A0 / 1e3, 1)} km, period ${fmt(T0 / 60, 2)} min`
    : st.k < 1
      ? `the exact Kepler ellipse for a start ${fmt((1 - st.k) * 100, 0)} % slower than circular (black dashed): the start is now the highest point, the lowest is at ${fmt((ref.rp - R_EARTH) / 1e3, 0)} km altitude${ref.rp < R_EARTH ? ", <b>below the surface</b>" : ""}, period ${fmt(ref.T / 60, 1)} min`
      : `the exact Kepler ellipse for a start ${fmt((st.k - 1) * 100, 0)} % faster than circular (black dashed): the start is the lowest point, the highest is at ${fmt((ref.ra - R_EARTH) / 1e3, 0)} km altitude, period ${fmt(ref.T / 60, 1)} min`;
  txt = `From ${fmt(ALT / 1e3, 0)} km altitude with speed [[${fmt(st.k, 2)}\\,v_c]] = ${fmt(st.k * VC, 0)} m/s, ${st.revs} revolution${st.revs > 1 ? "s" : ""} in steps of ${h} s (${fmt(perRev, 1)} steps per revolution), against ${refTxt}. `;
  if (eu.hit != null) txt += `Explicit Euler hits the Earth after ${fmt(eu.hit / 60, 0)} min. `;
  else txt += `Explicit Euler (orange) gains energy at every step: its largest relative energy error is ${sci(eu.maxE)} and its orbit has swollen to ${len(eu.rmax - R_EARTH)} altitude. `;
  if (sy.hit != null) txt += `Symplectic Euler hits the Earth after ${fmt(sy.hit / 60, 0)} min. `;
  else txt += `Symplectic Euler (green) keeps its energy error bounded at ${sci(sy.maxE)}, but its orbit is ${sy.maxR > 1e3 ? "visibly" : "slightly"} misshapen (radius off by up to ${len(sy.maxR)}) and it runs ${len(sy.posErr)} from where the satellite should be. `;
  if (rk.hit != null) txt += `RK4 hits the Earth after ${fmt(rk.hit / 60, 0)} min${ref.rp < R_EARTH ? ", as the exact orbit does: the start is too slow to stay in orbit" : ""}. `;
  else txt += `RK4 (blue) is the accurate one: energy error ${sci(rk.maxE)}, ${len(rk.posErr)} from the exact position at the end, for four evaluations of gravity per step. `;
  if (ref.circle) txt += `The closed form gives the exact position at any time for one cosine and one sine.`;
  caption($("#heroStatus"), txt);
}
/* Slider moves are coalesced into one recomputation per turn of the event loop: at a 1 s step and 20 revolutions a
   recomputation takes most of a second, and dragging would otherwise queue one per input event. */
let heroQueued = false, convQueued = false;
function queueHero(conv) {
  convQueued = convQueued || conv;
  if (heroQueued) return;
  heroQueued = true;
  setTimeout(() => { heroQueued = false; computeHero(); drawHero(); if (convQueued) { convQueued = false; drawConv(); } }, 0);
}
$("#hH").addEventListener("input", e => { st.hIdx = +e.target.value; queueHero(true); });
$("#hN").addEventListener("input", e => { st.revs = +e.target.value; queueHero(false); });
$("#hK").addEventListener("input", e => { st.k = +(+e.target.value).toFixed(2); queueHero(false); });
METHODS.forEach(m => {
  const b = $("#show-" + m.key);
  b.addEventListener("click", () => { st.show[m.key] = !st.show[m.key]; b.setAttribute("aria-pressed", String(st.show[m.key])); drawHero(); });
});
function setHero(hIdx, revs, k) {
  st.hIdx = hIdx; st.revs = revs; st.k = k;
  $("#hH").value = hIdx; $("#hN").value = revs; $("#hK").value = k;
  computeHero(); drawHero(); drawConv();
}
$("#hReset").addEventListener("click", () => setHero(6, 5, 1));
$("#hEll").addEventListener("click", () => setHero(6, 5, 1.05));
$("#hSlow").addEventListener("click", () => setHero(6, 5, 0.95));

/* ================= altitude widget ================= */
const ALT_MIN = 200, A_GEO = Math.cbrt(MU_EARTH / (OMEGA_EARTH * OMEGA_EARTH)), ALT_GEO = (A_GEO - R_EARTH) / 1e3;
const circ = altKm => { const a = R_EARTH + altKm * 1e3; return { a, v: Math.sqrt(MU_EARTH / a), T: 2 * Math.PI * Math.sqrt(a * a * a / MU_EARTH), g: MU_EARTH / (a * a) }; };
let altKm = 550;
function drawAlt() {
  const xs = [], Ts = [], Vs = [];
  const l0 = Math.log10(ALT_MIN), l1 = Math.log10(ALT_GEO);
  for (let j = 0; j <= 200; j++) { const l = l0 + (l1 - l0) * j / 200, c = circ(10 ** l); xs.push(l); Ts.push(c.T / 3600); Vs.push(c.v / 1e3); }
  const cvT = $("#altTC"), narrow = narrowOf(cvT, 300);
  const xt = (narrow ? [200, 1000, 5000, 20000] : [200, 500, 1000, 2000, 5000, 10000, 20000]).map(Math.log10);
  const xfmt = v => { const k = Math.round(10 ** v); return k >= 1000 ? (k / 1000) + "k" : String(k); };
  const c = circ(altKm), lx = Math.log10(altKm);
  const mark = (fr, y, label) => {
    const { ctx } = fr;
    ctx.strokeStyle = alpha("--muted", 0.6); ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(fr.X(lx), fr.Y(fr.y0)); ctx.lineTo(fr.X(lx), fr.Y(y)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(fr.X(lx), fr.Y(y), 4.5, 0, 2 * Math.PI); ctx.fill();
    ctx.font = "11px " + css("--font-mono"); ctx.fillStyle = css("--ink-2");
    ctx.textAlign = fr.X(lx) > fr.pad.l + fr.iw * 0.6 ? "right" : "left";
    ctx.fillText(label, fr.X(lx) + (ctx.textAlign === "right" ? -8 : 8), fr.Y(y) - 8);
  };
  const frT = frame(cvT, narrow ? 1.3 : 1.6, { x: [l0, l1], y: [0, 25], xticks: xt, xfmt, yticks: [0, 6, 12, 18, 24], xlabel: "altitude [km], log scale", ylabel: "period [h]" });
  plotLine(frT, xs, Ts, { color: css("--post"), width: 2 });
  mark(frT, c.T / 3600, c.T < 7200 ? fmt(c.T / 60, 1) + " min" : fmt(c.T / 3600, 2) + " h");
  const cvV = $("#altVC");
  const frV = frame(cvV, narrow ? 1.3 : 1.6, { x: [l0, l1], y: [2, 8.5], xticks: xt, xfmt, yticks: [2, 4, 6, 8], xlabel: "altitude [km], log scale", ylabel: "speed [km/s]" });
  plotLine(frV, xs, Vs, { color: css("--c4"), width: 2 });
  mark(frV, c.v / 1e3, fmt(c.v / 1e3, 3) + " km/s");
  $("#altV").textContent = Math.round(altKm).toLocaleString("en-US") + " km";
  $("#aA").textContent = fmt(c.a / 1e3, 1) + " km";
  $("#aSpd").textContent = fmt(c.v, 0) + " m/s";
  $("#aT").textContent = c.T < 7200 ? fmt(c.T / 60, 2) + " min" : fmt(c.T / 3600, 3) + " h";
  $("#aRev").textContent = fmt(86400 / c.T, 2);
  $("#aG").textContent = fmt(c.g, 3) + " m/s²";
  const isGeo = Math.abs(altKm - ALT_GEO) < 1;
  caption($("#altCap"), `A circular orbit at ${Math.round(altKm).toLocaleString("en-US")} km altitude has radius [[a = ${fmt(c.a / 1e3, 1)}]] km, speed [[\\sqrt{\\mu/a} = ${fmt(c.v / 1e3, 3)}]] km/s and period [[2\\pi\\sqrt{a^3/\\mu} = ${c.T < 7200 ? fmt(c.T / 60, 1) + "\\text{ min}" : fmt(c.T / 3600, 2) + "\\text{ h}"}]]: ${fmt(86400 / c.T, 2)} revolutions per day. ${isGeo ? `This is the geostationary orbit: its period is one sidereal day, [[2\\pi/\\omega_E = ${fmt(2 * Math.PI / OMEGA_EARTH / 3600, 4)}]] h, so over the equator it would stay above the same point of the turning Earth. With the package's mean radius it sits ${Math.round(ALT_GEO).toLocaleString("en-US")} km above the surface; the textbook 35,786 km is measured from the equatorial radius, 7.1 km larger.` : altKm < 2000 ? "Low orbits all take about an hour and a half; going higher makes the orbit slower and the path longer, so the period grows faster than the radius: as [[a^{3/2}]]." : "Higher up, gravity is weaker, the orbit is slower and the path is longer, so the period grows as [[a^{3/2}]]."}`);
}
$("#altS").addEventListener("input", e => { altKm = 10 ** +e.target.value; drawAlt(); });
$("#alt550").addEventListener("click", () => { altKm = 550; $("#altS").value = Math.log10(550); drawAlt(); });
$("#altGeo").addEventListener("click", () => { altKm = ALT_GEO; $("#altS").value = Math.log10(ALT_GEO); drawAlt(); });
$("#altS").min = Math.log10(ALT_MIN); $("#altS").max = Math.log10(ALT_GEO); $("#altS").value = Math.log10(550);

/* ================= convergence widget ================= */
const cv = { revs: 5, metric: "pos" };
let convCache = null;
function computeConv() {
  const r1 = reference(1);
  convCache = { revs: cv.revs, rows: METHODS.map(m => H_LIST.map(h => integrate(m.key, h, cv.revs, r1, true))) };
}
function drawConv() {
  if (!convCache || convCache.revs !== cv.revs) computeConv();
  const vals = convCache.rows.map(row => row.map(r => cv.metric === "pos" ? Math.log10(Math.max(r.posErr, 1e-6)) : Math.log10(Math.max(r.maxE, 1e-16))));
  const flat = vals.flat(), lo = Math.floor(Math.min(...flat)), hi = Math.ceil(Math.max(...flat));
  const c = $("#convC"), narrow = narrowOf(c);
  const fr = frame(c, narrow ? 1.25 : 2.1, { x: [0, Math.log10(240)], y: [lo, hi], xticks: (narrow ? [1, 10, 60, 240] : H_LIST).map(Math.log10), xfmt: v => String(Math.round(10 ** v)),
    yticks: intTicks(lo, hi, narrow ? 5 : 7), yfmt: cv.metric === "pos" ? (v => LEN_LABELS[String(v)] || pow10(v) + " m") : pow10,
    xlabel: "step h [s], log scale", ylabel: cv.metric === "pos" ? `distance from the exact position after ${cv.revs} rev.` : `largest |ΔE/E₀| over ${cv.revs} rev.`, pad: { l: 56 } });
  const xs = H_LIST.map(Math.log10);
  METHODS.forEach((m, k) => {
    plotLine(fr, xs, vals[k], { color: css(m.token), width: 2 });
    fr.ctx.fillStyle = css(m.token);
    xs.forEach((x, j) => { fr.ctx.beginPath(); fr.ctx.arc(fr.X(x), fr.Y(vals[k][j]), 3, 0, 2 * Math.PI); fr.ctx.fill(); });
  });
  const xh = fr.X(Math.log10(H_LIST[st.hIdx]));
  fr.ctx.strokeStyle = alpha("--muted", 0.7); fr.ctx.setLineDash([3, 3]); fr.ctx.beginPath(); fr.ctx.moveTo(xh, fr.pad.t); fr.ctx.lineTo(xh, fr.Y(lo)); fr.ctx.stroke(); fr.ctx.setLineDash([]);
  const i60 = H_LIST.indexOf(60), i30 = H_LIST.indexOf(30);
  const ratio = k => cv.metric === "pos" ? convCache.rows[k][i60].posErr / convCache.rows[k][i30].posErr : convCache.rows[k][i60].maxE / convCache.rows[k][i30].maxE;
  $("#cvNV").textContent = cv.revs;
  caption($("#convCap"), `${cv.metric === "pos" ? "How far each method ends from the exact position" : "The largest relative energy error of each method"} after ${cv.revs} revolution${cv.revs > 1 ? "s" : ""}, for steps from 1 s to 240 s (both axes logarithmic; the dashed line is the hero's step). Halving the step from 60 s to 30 s divides the error by ${fmt(ratio(1), 1)} for symplectic Euler and by ${fmt(ratio(2), 1)} for RK4. ${cv.metric === "pos" ? "On a log-log plot a method of order [[p]] is a line of slope [[p]] (RK4's local error falls as [[h^5]] per step, its global error as [[h^4]]); symplectic Euler's position error is dominated by a drift along the orbit that falls as [[h^2]] here. Explicit Euler does not converge in any useful sense at these steps: its error is as large as the orbit itself at every step shown, so its curve wanders instead of falling." : "Symplectic Euler's energy error stays bounded however long the run, RK4's grows slowly with time but starts tiny. Explicit Euler's barely improves with the step: even at 1 s its energy has changed by " + fmt(100 * convCache.rows[0][0].maxE, 1) + " % after " + cv.revs + " revolution" + (cv.revs > 1 ? "s" : "") + "."}`);
}
$("#cvN").addEventListener("input", e => { cv.revs = +e.target.value; drawConv(); });
$("#cvPos").addEventListener("click", () => { cv.metric = "pos"; $("#cvPos").setAttribute("aria-pressed", "true"); $("#cvE").setAttribute("aria-pressed", "false"); drawConv(); });
$("#cvE").addEventListener("click", () => { cv.metric = "energy"; $("#cvE").setAttribute("aria-pressed", "true"); $("#cvPos").setAttribute("aria-pressed", "false"); drawConv(); });

function drawAll() { drawHero(); drawAlt(); drawConv(); }
drawAll(); onRedraw(drawAll);
})();
