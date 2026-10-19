/* A Tour of Machine Learning, chapter 01, Linear regression: the live demos.
   The notebook's 36 cars, fitted by the formulas of Part 2 (one input) or by least
   squares with Householder QR (several inputs), as scikit-learn's LinearRegression does.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, int, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { makeRng } = window.Stats;

/* ---------------- the data: data/data.csv (car, model, volume cm3, weight kg, CO2 g/km) ---------------- */
const CARS = [
  ["Toyoty", "Aygo", 1000, 790, 99], ["Mitsubishi", "Space Star", 1200, 1160, 95], ["Skoda", "Citigo", 1000, 929, 95],
  ["Fiat", "500", 900, 865, 90], ["Mini", "Cooper", 1500, 1140, 105], ["VW", "Up!", 1000, 929, 105],
  ["Skoda", "Fabia", 1400, 1109, 90], ["Mercedes", "A-Class", 1500, 1365, 92], ["Ford", "Fiesta", 1500, 1112, 98],
  ["Audi", "A1", 1600, 1150, 99], ["Hyundai", "I20", 1100, 980, 99], ["Suzuki", "Swift", 1300, 990, 101],
  ["Ford", "Fiesta", 1000, 1112, 99], ["Honda", "Civic", 1600, 1252, 94], ["Hundai", "I30", 1600, 1326, 97],
  ["Opel", "Astra", 1600, 1330, 97], ["BMW", "1", 1600, 1365, 99], ["Mazda", "3", 2200, 1280, 104],
  ["Skoda", "Rapid", 1600, 1119, 104], ["Ford", "Focus", 2000, 1328, 105], ["Ford", "Mondeo", 1600, 1584, 94],
  ["Opel", "Insignia", 2000, 1428, 99], ["Mercedes", "C-Class", 2100, 1365, 99], ["Skoda", "Octavia", 1600, 1415, 99],
  ["Volvo", "S60", 2000, 1415, 99], ["Mercedes", "CLA", 1500, 1465, 102], ["Audi", "A4", 2000, 1490, 104],
  ["Audi", "A6", 2000, 1725, 114], ["Volvo", "V70", 1600, 1523, 109], ["BMW", "5", 2000, 1705, 114],
  ["Mercedes", "E-Class", 2100, 1605, 115], ["Volvo", "XC70", 2000, 1746, 117], ["Ford", "B-Max", 1600, 1235, 104],
  ["BMW", "216", 1600, 1390, 108], ["Opel", "Zafira", 1600, 1405, 109], ["Mercedes", "SLK", 2500, 1395, 120],
];
const N = CARS.length, VOL = CARS.map(c => c[2]), WT = CARS.map(c => c[3]), CO2 = CARS.map(c => c[4]);
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

/* one input: the formulas of Part 2 */
function line(x, y) {
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  x.forEach((v, i) => { sxy += (v - mx) * (y[i] - my); sxx += (v - mx) ** 2; syy += (y[i] - my) ** 2; });
  const b1 = sxy / sxx;
  return { b0: my - b1 * mx, b1, mx, my, sxy, sxx, syy, r: sxy / Math.sqrt(sxx * syy) };
}
/* several inputs: least squares by Householder QR (needs n >= columns) */
function lstsq(X, y) {
  const n = X.length, m = X[0].length, A = X.map(r => r.slice()), b = y.slice();
  for (let j = 0; j < m; j++) {
    let norm = 0; for (let i = j; i < n; i++) norm += A[i][j] * A[i][j]; norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const a0 = A[j][j] > 0 ? -norm : norm, v = new Float64Array(n);
    for (let i = j; i < n; i++) v[i] = A[i][j]; v[j] -= a0;
    let vv = 0; for (let i = j; i < n; i++) vv += v[i] * v[i];
    if (vv === 0) continue;
    for (let c = j; c < m; c++) { let s = 0; for (let i = j; i < n; i++) s += v[i] * A[i][c]; s *= 2 / vv; for (let i = j; i < n; i++) A[i][c] -= s * v[i]; }
    let s = 0; for (let i = j; i < n; i++) s += v[i] * b[i]; s *= 2 / vv; for (let i = j; i < n; i++) b[i] -= s * v[i];
  }
  const w = new Float64Array(m);
  for (let j = m - 1; j >= 0; j--) { let s = b[j]; for (let c = j + 1; c < m; c++) s -= A[j][c] * w[c]; w[j] = s / A[j][j]; }
  return w;
}
/* fit CO2 (or y) on the given columns plus an intercept; R² as LinearRegression.score */
function fitCols(cols, y) {
  const my = mean(y), sst = y.reduce((s, v) => s + (v - my) ** 2, 0);
  if (!cols.length) return { w: [my], ssr: sst, sst, r2: 0 };
  const X = y.map((_, i) => [1, ...cols.map(c => c[i])]), w = lstsq(X, y);
  const ssr = y.reduce((s, v, i) => s + (v - X[i].reduce((a, u, k) => a + u * w[k], 0)) ** 2, 0);
  return { w, ssr, sst, r2: 1 - ssr / sst };
}
const adjR2 = (r2, n, p) => 1 - (1 - r2) * (n - 1) / (n - p - 1);
/* numbers for tables and captions: at most four decimals, a true minus sign, no "-0" */
const num = (v, d = 4) => { const s = String(+v.toFixed(d)); return s === "-0" ? "0" : s.replace("-", "−"); };
const tnum = (v, d = 4) => String(+v.toFixed(d) || 0);                       // the same, for TeX
const comma = v => Math.round(v).toLocaleString("en-US");
function dot(ctx, x, y, r, fill, stroke, lw = 2) { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); } }
/* the centre of a cloud, (mean x, mean y), as a + with a halo so it reads over points and lines */
function cross(ctx, x, y, r = 7) {
  ctx.lineCap = "round";
  [[css("--surface"), 5], [css("--ink"), 2]].forEach(([c, w]) => { ctx.strokeStyle = c; ctx.lineWidth = w; seg(ctx, x - r, y, x + r, y); seg(ctx, x, y - r, x, y + r); });
  ctx.lineCap = "butt";
}
/* the data-y under a pointer, in a frame */
const yAt = (fr, cv, ev) => { const r = cv.getBoundingClientRect(); return fr.y0 + (fr.pad.t + fr.ih - (ev.clientY - r.top)) / fr.ih * (fr.y1 - fr.y0); };
const xAt = (fr, cv, ev) => { const r = cv.getBoundingClientRect(); return fr.x0 + (ev.clientX - r.left - fr.pad.l) / fr.iw * (fr.x1 - fr.x0); };

const SIMPLE = line(WT, CO2), SST = SIMPLE.syy, SMIN = SST - SIMPLE.sxy ** 2 / SIMPLE.sxx;

/* =====================================================================
   THE DEMO: FIT THE LINE YOURSELF
   The line is CO2 = h + b (W - mean W). Then S = n (mean CO2 - h)^2 + Sxx (b - b_hat)^2 + S_min:
   the two sliders' errors add up separately.
   ===================================================================== */
const HX = [900, 1700], hero = { b: 0.03, h: 97, squares: false, revealed: false, drag: -1, fr: null };
const heroAt = x => hero.h + hero.b * (x - SIMPLE.mx);
function drawHero() {
  const cv = $("#lineC"), narrow = narrowOf(cv);
  const fr = hero.fr = frame(cv, narrow ? 1.1 : 1.9, { x: [750, 1800], y: [80, 125], xlabel: "weight (kg)", ylabel: "CO2 (g/km)", yticks: [80, 90, 100, 110, 120] });
  const { ctx, X, Y } = fr;
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  const res = CO2.map((y, i) => y - heroAt(WT[i]));
  if (hero.squares) {
    WT.forEach((x, i) => {
      const side = Math.abs(Y(CO2[i]) - Y(heroAt(x)));
      ctx.fillStyle = alpha("--c5", 0.1); ctx.strokeStyle = alpha("--c5", 0.45); ctx.lineWidth = 1;
      ctx.fillRect(X(x), Math.min(Y(CO2[i]), Y(heroAt(x))), side, side); ctx.strokeRect(X(x), Math.min(Y(CO2[i]), Y(heroAt(x))), side, side);
    });
  }
  ctx.strokeStyle = css("--c5"); ctx.lineWidth = 1.5;
  WT.forEach((x, i) => seg(ctx, X(x), Y(CO2[i]), X(x), Y(heroAt(x))));
  ctx.restore();
  if (hero.revealed) plotLine(fr, [750, 1800], [SIMPLE.b0 + SIMPLE.b1 * 750, SIMPLE.b0 + SIMPLE.b1 * 1800], { color: css("--warm"), width: 2.4, dash: [7, 5] });
  plotLine(fr, [750, 1800], [heroAt(750), heroAt(1800)], { color: css("--ink"), width: 2.2 });
  WT.forEach((x, i) => dot(ctx, X(x), Y(CO2[i]), narrow ? 3.2 : 3.8, css("--post")));
  HX.forEach(x => { const y = clamp(heroAt(x), fr.y0, fr.y1); dot(ctx, X(x), Y(y), 8, css("--surface"), css("--ink"), 2.4); });
  cross(ctx, X(SIMPLE.mx), Y(SIMPLE.my));

  const S = res.reduce((s, e) => s + e * e, 0), sum = res.reduce((s, e) => s + e, 0), b0 = hero.h - hero.b * SIMPLE.mx;
  const partH = N * (SIMPLE.my - hero.h) ** 2, partB = SIMPLE.sxx * (hero.b - SIMPLE.b1) ** 2, r2 = 1 - S / SST;
  $("#hBV").textContent = (hero.b < 0 ? "−" : "") + fmt(Math.abs(hero.b), 4); $("#hHV").textContent = fmt(hero.h, 1) + " g/km";
  $("#rLine").textContent = `your line: CO2 = ${fmt(b0, 2)} ${hero.b < 0 ? "−" : "+"} ${fmt(Math.abs(hero.b), 4)} × weight`;
  $("#rSum").textContent = num(sum, 1); $("#rS").textContent = comma(S); $("#rBestS").textContent = comma(SMIN); $("#rR2").textContent = num(r2, 3);
  let text;
  if (S - SMIN < 0.5) text = `This is the least-squares line, CO2 = ${fmt(SIMPLE.b0, 2)} + ${fmt(SIMPLE.b1, 4)} × weight: <b>S = ${comma(SMIN)}</b>, and no other line does better. It passes through the centre of the cloud (1,292 kg, 102.0 g/km), and its residuals sum to zero. Its R² is ${fmt(1 - SMIN / SST, 3)}.`;
  else if (hero.b === 0 && Math.abs(hero.h - SIMPLE.my) < 1e-9) text = `The flat line at the mean ignores the weight. Its residuals sum to zero, yet <b>S = ${comma(S)}</b>: that is the whole spread of CO2, so R² = 0. Summing the residuals cannot tell this line from a good one; summing their squares can.`;
  else {
    text = `Your line misses by <b>S = ${comma(S)}</b>, ${pct(S / SMIN - 1, 0)} more than the least-squares line (S = ${comma(SMIN)}).`;
    text += partH > partB ? ` Most of the excess comes from the line's height: move it towards the centre of the cloud, marked +.` : ` Most of the excess comes from the slope: tilt the line about the centre of the cloud, marked +.`;
    if (r2 < 0) text += ` R² is below zero: your line does worse than the flat line at the mean.`;
  }
  caption($("#heroStatus"), text);
}
function setHero(b, h, syncSliders = true) {
  hero.b = clamp(b, -0.03, 0.06); hero.h = clamp(h, 70, 135);
  if (syncSliders) { $("#hB").value = String(hero.b); $("#hH").value = String(hero.h); }
  drawHero();
}
$("#hB").addEventListener("input", e => setHero(+e.target.value, hero.h, false));
$("#hH").addEventListener("input", e => setHero(hero.b, +e.target.value, false));
$("#hBest").addEventListener("click", () => { hero.revealed = true; setHero(SIMPLE.b1, SIMPLE.my); });
$("#hFlat").addEventListener("click", () => setHero(0, SIMPLE.my));
$("#hSq").addEventListener("click", () => { hero.squares = !hero.squares; $("#hSq").setAttribute("aria-pressed", String(hero.squares)); drawHero(); });
(() => {
  const cv = $("#lineC");
  const move = ev => {
    if (hero.drag < 0 || !hero.fr) return;
    const k = hero.drag, other = HX[1 - k], yOther = heroAt(other), y = clamp(yAt(hero.fr, cv, ev), 70, 135);
    const b = (y - yOther) / (HX[k] - other);
    setHero(b, yOther + b * (SIMPLE.mx - other));
  };
  cv.addEventListener("pointerdown", ev => {
    if (!hero.fr) return;
    const x = xAt(hero.fr, cv, ev);
    hero.drag = Math.abs(x - HX[0]) < Math.abs(x - HX[1]) ? 0 : 1;
    cv.setPointerCapture(ev.pointerId); move(ev);
  });
  cv.addEventListener("pointermove", move);
  ["pointerup", "pointercancel"].forEach(t => cv.addEventListener(t, () => { hero.drag = -1; }));
})();

/* =====================================================================
   PARTS 2 AND 3: FOUR POINTS (shared by the bowl and the hand calculation)
   ===================================================================== */
const P4 = { x: [1, 2, 3, 4], y: [1, 3, 2, 4] };
const fit4 = () => line(P4.x, P4.y);

/* ---------- Part 2: the bowl S(b0, b1), seen from above ---------- */
const bowl = { b0: 2, b1: 0.1, fr: null, drag: false };
const S4 = (b0, b1) => P4.x.reduce((s, x, i) => s + (P4.y[i] - b0 - b1 * x) ** 2, 0);
function bowlView() { const f = fit4(); return { f, x: [f.b0 - 2.5, f.b0 + 2.5], y: [f.b1 - 1, f.b1 + 1] }; }
function drawBowl() {
  const v = bowlView(), f = v.f, cv = $("#bowlC");
  bowl.b0 = clamp(bowl.b0, v.x[0], v.x[1]); bowl.b1 = clamp(bowl.b1, v.y[0], v.y[1]);
  const fr = bowl.fr = frame(cv, 1.25, { x: v.x, y: v.y, xlabel: "intercept β0", ylabel: "slope β1", xfmt: t => num(t, 1), yfmt: t => num(t, 1) });
  const { ctx, X, Y } = fr, smin = f.syy - f.sxy ** 2 / f.sxx;
  // rings S = S_min + d^2: (b - b_hat)' G (b - b_hat) = d^2 with G = X'X = L L', so b = b_hat + d L^-T u
  const sx = P4.x.reduce((a, x) => a + x, 0), sxx = P4.x.reduce((a, x) => a + x * x, 0), n = P4.x.length;
  const l11 = Math.sqrt(n), l21 = sx / l11, l22 = Math.sqrt(sxx - l21 * l21);
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  for (let k = 12; k >= 1; k--) {
    const d = 0.6 * k;
    ctx.beginPath();
    for (let t = 0; t <= 96; t++) {
      const a = 2 * Math.PI * t / 96, u1 = Math.cos(a), u2 = Math.sin(a);
      const z2 = u2 / l22, z1 = (u1 - l21 * z2) / l11;                      // z = L^-T u
      const px = X(f.b0 + d * z1), py = Y(f.b1 + d * z2);
      t ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.fillStyle = alpha("--post", 0.05); ctx.fill();
    ctx.strokeStyle = alpha("--post", 0.55); ctx.lineWidth = 1; ctx.stroke();
  }
  ctx.restore();
  ctx.strokeStyle = css("--warm"); ctx.lineWidth = 2;
  seg(ctx, X(f.b0) - 6, Y(f.b1) - 6, X(f.b0) + 6, Y(f.b1) + 6); seg(ctx, X(f.b0) - 6, Y(f.b1) + 6, X(f.b0) + 6, Y(f.b1) - 6);
  dot(ctx, X(bowl.b0), Y(bowl.b1), 7, css("--surface"), css("--ink"), 2.4);

  const lf = frame($("#bowlL"), 1.25, { x: [0.5, 4.5], y: [-1, 6], xticks: [1, 2, 3, 4], yticks: [0, 2, 4, 6] });
  const L = lf.ctx;
  L.strokeStyle = css("--c5"); L.lineWidth = 1.5;
  P4.x.forEach((x, i) => seg(L, lf.X(x), lf.Y(P4.y[i]), lf.X(x), lf.Y(bowl.b0 + bowl.b1 * x)));
  plotLine(lf, [0.5, 4.5], [bowl.b0 + bowl.b1 * 0.5, bowl.b0 + bowl.b1 * 4.5], { color: css("--ink"), width: 2.2 });
  P4.x.forEach((x, i) => dot(L, lf.X(x), lf.Y(P4.y[i]), 4.5, css("--post")));

  const e = P4.x.map((x, i) => P4.y[i] - bowl.b0 - bowl.b1 * x), S = S4(bowl.b0, bowl.b1);
  const d0 = -2 * e.reduce((a, v) => a + v, 0), d1 = -2 * e.reduce((a, v, i) => a + v * P4.x[i], 0);
  const atMin = Math.abs(d0) < 1e-9 && Math.abs(d1) < 1e-9;
  let text = `The line [[\\hat y = ${tnum(bowl.b0, 2)} ${bowl.b1 < 0 ? "-" : "+"} ${tnum(Math.abs(bowl.b1), 2)}\\,x]]: <b>S = ${num(S, 2)}</b>, with [[\\frac{\\partial S}{\\partial \\beta_0} = ${tnum(d0, 2)}]] and [[\\frac{\\partial S}{\\partial \\beta_1} = ${tnum(d1, 2)}]]. `;
  if (atMin) text += `Both derivatives are zero: this is the bottom of the bowl, the least-squares line, with S = ${num(smin, 2)}.`;
  else {
    const dir = (g, nm) => (Math.abs(g) < 1e-9 ? `${nm} is already right` : `S falls if ${nm} ${g < 0 ? "rises" : "falls"}`);
    text += `Not the bottom yet: ${dir(d0, "the intercept")}, and ${dir(d1, "the slope")}. The bottom, marked ×, has S = ${num(smin, 2)}. The rings are long and thin because the intercept and the slope can partly stand in for each other.`;
  }
  caption($("#bowlCap"), text);
}
(() => {
  const cv = $("#bowlC");
  const move = ev => {
    if (!bowl.drag || !bowl.fr) return;
    const f = fit4(), fr = bowl.fr;
    let b0 = xAt(fr, cv, ev), b1 = yAt(fr, cv, ev);
    if (Math.hypot(fr.X(b0) - fr.X(f.b0), fr.Y(b1) - fr.Y(f.b1)) < 9) { b0 = f.b0; b1 = f.b1; }   // snap to the bottom
    bowl.b0 = b0; bowl.b1 = b1; drawBowl();
  };
  cv.addEventListener("pointerdown", ev => { bowl.drag = true; cv.setPointerCapture(ev.pointerId); move(ev); });
  cv.addEventListener("pointermove", move);
  ["pointerup", "pointercancel"].forEach(t => cv.addEventListener(t, () => { bowl.drag = false; }));
  $("#bowlMin").addEventListener("click", () => { const f = fit4(); bowl.b0 = f.b0; bowl.b1 = f.b1; drawBowl(); });
})();

/* ---------- Part 3: the hand calculation, live ---------- */
const hand = { fr: null, drag: -1 };
function drawHand() {
  const f = fit4(), cv = $("#handC"), narrow = narrowOf(cv);
  const fr = hand.fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0.5, 4.5], y: [-0.5, 5.5], xticks: [1, 2, 3, 4], yticks: [0, 1, 2, 3, 4, 5] });
  const { ctx, X, Y } = fr, yh = P4.x.map(x => f.b0 + f.b1 * x), e = P4.y.map((y, i) => y - yh[i]);
  ctx.strokeStyle = css("--c5"); ctx.lineWidth = 1.5;
  P4.x.forEach((x, i) => seg(ctx, X(x), Y(P4.y[i]), X(x), Y(yh[i])));
  plotLine(fr, [0.5, 4.5], [f.b0 + f.b1 * 0.5, f.b0 + f.b1 * 4.5], { color: css("--warm"), width: 2.4 });
  P4.x.forEach((x, i) => dot(ctx, X(x), Y(P4.y[i]), 7, css("--post"), css("--surface"), 2));
  cross(ctx, X(f.mx), Y(f.my));

  // the two tables of the notebook
  const dx = P4.x.map(x => x - f.mx), dy = P4.y.map(y => y - f.my), S = e.reduce((a, v) => a + v * v, 0);
  const row = cells => `<tr>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
  $("#handT1 tbody").innerHTML = P4.x.map((x, i) => row([x, num(P4.y[i]), num(dx[i]), num(dy[i]), num(dx[i] * dy[i]), num(dx[i] ** 2)])).join("") +
    `<tr class="total"><td>sum</td><td></td><td>0</td><td>0</td><td>${num(f.sxy)}</td><td>${num(f.sxx)}</td></tr>`;
  $("#handT2 tbody").innerHTML = P4.x.map((x, i) => row([x, num(P4.y[i]), num(yh[i]), num(e[i]), num(e[i] ** 2), num(x * e[i])])).join("") +
    `<tr class="total"><td>sum</td><td></td><td></td><td>${num(e.reduce((a, v) => a + v, 0))}</td><td>${num(S)}</td><td>${num(e.reduce((a, v, i) => a + v * P4.x[i], 0))}</td></tr>`;
  const b1t = f.b1 < 0 ? `(${tnum(f.b1)})` : tnum(f.b1);
  caption($("#handText"), `So [[\\hat\\beta_1 = \\frac{${tnum(f.sxy)}}{${tnum(f.sxx)}} = ${tnum(f.b1)}]] and [[\\hat\\beta_0 = ${tnum(f.my)} - ${b1t} \\times ${tnum(f.mx)} = ${tnum(f.b0)}]]. The fitted line is [[\\hat y = ${tnum(f.b0)} ${f.b1 < 0 ? "-" : "+"} ${tnum(Math.abs(f.b1))}\\,x]]. Now the predictions and residuals:`);
  const r2 = f.syy > 0 ? 1 - S / f.syy : NaN;
  const same = P4.y.join() === "1,3,2,4";
  caption($("#handCap"), `With the points at heights ${P4.y.map(v => num(v)).join(", ")}: the line [[\\hat y = ${tnum(f.b0)} ${f.b1 < 0 ? "-" : "+"} ${tnum(Math.abs(f.b1))}\\,x]], <b>S = ${num(S)}</b>, ` +
    (f.syy > 0 ? `[[SS_{\\mathrm{tot}} = ${tnum(f.syy)}]] and R² = ${num(r2, 4)}.` : `and R² is undefined: all four points are at the same height, so there is no spread to explain.`) +
    ` The residuals sum to zero and [[\\sum_i x_i e_i = 0]], wherever the points are.` + (same ? " These are the notebook's four points." : ""));
}
(() => {
  const cv = $("#handC");
  const move = ev => {
    if (hand.drag < 0 || !hand.fr) return;
    const y = clamp(Math.round(yAt(hand.fr, cv, ev) * 2) / 2, 0, 5);                // snap to halves
    if (y !== P4.y[hand.drag]) { P4.y[hand.drag] = y; drawHand(); drawBowl(); }
  };
  cv.addEventListener("pointerdown", ev => {
    if (!hand.fr) return;
    hand.drag = clamp(Math.round(xAt(hand.fr, cv, ev)), 1, 4) - 1;
    cv.setPointerCapture(ev.pointerId); move(ev);
  });
  cv.addEventListener("pointermove", move);
  ["pointerup", "pointercancel"].forEach(t => cv.addEventListener(t, () => { hand.drag = -1; }));
  $("#handReset").addEventListener("click", () => { P4.y = [1, 3, 2, 4]; drawHand(); drawBowl(); });
})();

/* =====================================================================
   PART 4: R² NEVER FALLS
   ===================================================================== */
const R2W = { base: "w", seed: 11, noise: null, curves: null };
const BASE = { none: [[], "no real input"], w: [[WT], "weight"], v: [[VOL], "volume"], wv: [[WT, VOL], "weight and volume"] };
function r2Curves() {
  if (!R2W.noise) { const r = makeRng(R2W.seed); R2W.noise = Array.from({ length: 20 }, () => Array.from({ length: N }, () => r.normal(0, 1))); }
  const cols = BASE[R2W.base][0];
  R2W.curves = Array.from({ length: 21 }, (_, k) => { const ft = fitCols([...cols, ...R2W.noise.slice(0, k)], CO2), p = cols.length + k; return { r2: ft.r2, adj: p ? adjR2(ft.r2, N, p) : 0 }; });
}
function drawR2() {
  if (!R2W.curves) r2Curves();
  $$("#r2 [data-base]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.base === R2W.base)));
  const k = +$("#uK").value, c = R2W.curves, cur = c[k], cv = $("#r2C");
  $("#uKV").textContent = String(k);
  const fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [0, 20], y: [-0.5, 1], xlabel: "useless random inputs", xticks: [0, 5, 10, 15, 20], yticks: [-0.5, 0, 0.5, 1] });
  const { ctx, X, Y } = fr, ks = c.map((_, i) => i);
  ctx.strokeStyle = alpha("--ink", 0.45); ctx.lineWidth = 1; ctx.setLineDash([3, 3]); seg(ctx, X(k), fr.pad.t, X(k), Y(-0.5)); ctx.setLineDash([]);
  plotLine(fr, ks, c.map(q => clamp(q.r2, -0.5, 1)), { color: css("--post"), width: 2.2 });
  plotLine(fr, ks, c.map(q => clamp(q.adj, -0.5, 1)), { color: css("--warm"), width: 2.2 });
  dot(ctx, X(k), Y(clamp(cur.r2, -0.5, 1)), 5, css("--post")); dot(ctx, X(k), Y(clamp(cur.adj, -0.5, 1)), 5, css("--warm"));
  $("#r2Exp").style.width = (100 * clamp(cur.r2, 0, 1)) + "%"; $("#r2Res").style.width = (100 * (1 - clamp(cur.r2, 0, 1))) + "%";
  const nm = BASE[R2W.base][1];
  let text = R2W.base === "none" && k === 0 ? `With no input the model is the flat line at the mean, so R² = 0 by definition. ` : `With ${nm}${k ? ` and ${k} useless input${k > 1 ? "s" : ""}` : ""}: <b>R² = ${fmt(cur.r2, 3)}</b>${R2W.base === "none" && k === 0 ? "" : `, adjusted R² = ${fmt(cur.adj, 3)}`}. `;
  text += `The bar splits the spread of CO2 into the part the model explains (blue, ${pct(clamp(cur.r2, 0, 1), 0)}) and the part left in the residuals. `;
  if (k > 0) text += `The ${k} random input${k > 1 ? "s" : ""}, pure noise unrelated to the cars, raised R² from ${fmt(c[0].r2, 3)} to ${fmt(cur.r2, 3)}: least squares bends to fit even noise. The adjusted R² ${cur.adj < c[0].adj ? "fell" : "rose"}, from ${fmt(c[0].adj, 3)} to ${fmt(cur.adj, 3)}; draw new random inputs, and on average it stays where it started.`;
  else text += `Slide to add random inputs, and watch R² climb.`;
  caption($("#r2Cap"), text);
}
$$("#r2 [data-base]").forEach(b => b.addEventListener("click", () => { R2W.base = b.dataset.base; R2W.curves = null; drawR2(); }));
$("#uK").addEventListener("input", drawR2);
$("#uNew").addEventListener("click", () => { R2W.seed++; R2W.noise = null; R2W.curves = null; drawR2(); });

/* =====================================================================
   PART 5: THE DATA, PAIR BY PAIR
   ===================================================================== */
const PAIRS = {
  wc: { x: WT, y: CO2, xr: [750, 1800], yr: [85, 125], xl: "weight (kg)", yl: "CO2 (g/km)", col: "--post" },
  vc: { x: VOL, y: CO2, xr: [800, 2600], yr: [85, 125], xl: "engine volume (cm³)", yl: "CO2 (g/km)", col: "--warm" },
  wv: { x: WT, y: VOL, xr: [750, 1800], yr: [800, 2600], xl: "weight (kg)", yl: "volume (cm³)", col: "--c4" },
};
const scat = { pair: "wc", pick: -1, fr: null };
function drawScat() {
  $$("#scat [data-pair]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.pair === scat.pair)));
  const P = PAIRS[scat.pair], cv = $("#scatC"), f = line(P.x, P.y);
  const fr = scat.fr = frame(cv, narrowOf(cv) ? 1.3 : 2.3, { x: P.xr, y: P.yr, xlabel: P.xl, ylabel: P.yl });
  const { ctx, X, Y } = fr;
  plotLine(fr, P.xr, P.xr.map(x => f.b0 + f.b1 * x), { color: css("--ink"), width: 1.8, dash: [6, 4] });
  P.x.forEach((x, i) => dot(ctx, X(x), Y(P.y[i]), 4, css(P.col)));
  if (scat.pick >= 0) dot(ctx, X(P.x[scat.pick]), Y(P.y[scat.pick]), 8, null, css("--ink"), 2);
  const lead = { wc: "CO2 against weight", vc: "CO2 against engine volume", wv: "Engine volume against weight" }[scat.pair];
  const tail = scat.pair === "wv"
    ? ` The two inputs are themselves strongly related: heavier cars tend to have bigger engines. When both enter one model, they will have to share the credit for CO2.`
    : ` A straight line explains r² = ${pct(f.r * f.r, 0)} of the variance of CO2; the rest is scatter that ${scat.pair === "wc" ? "weight" : "volume"} alone cannot account for.`;
  caption($("#scatCap"), `${lead}: correlation <b>r = ${fmt(f.r, 2)}</b>, and the dashed least-squares line rises ${fmt(f.b1, scat.pair === "wv" ? 2 : 4)} ${scat.pair === "wv" ? "cm³" : "g/km"} per ${scat.pair === "vc" ? "cm³" : "kg"}.` + tail);
  const c = CARS[scat.pick];
  $("#scatPick").textContent = c ? `${c[0]} ${c[1]}: ${comma(c[3])} kg, ${comma(c[2])} cm³, ${c[4]} g/km` : "Tap or hover over a point to see which car it is.";
}
(() => {
  const cv = $("#scatC");
  const pick = ev => {
    if (!scat.fr) return;
    const P = PAIRS[scat.pair], r = cv.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
    let best = -1, bd = 24;
    P.x.forEach((x, i) => { const d = Math.hypot(scat.fr.X(x) - px, scat.fr.Y(P.y[i]) - py); if (d < bd) { bd = d; best = i; } });
    if (best !== scat.pick && (best >= 0 || ev.type === "pointerdown")) { scat.pick = best; drawScat(); }
  };
  cv.addEventListener("pointerdown", pick);
  cv.addEventListener("pointermove", pick);
  $$("#scat [data-pair]").forEach(b => b.addEventListener("click", () => { scat.pair = b.dataset.pair; drawScat(); }));
})();

/* =====================================================================
   PART 7: THE PLANE, SLICED AT A FIXED VOLUME
   ===================================================================== */
const MULTI = fitCols([WT, VOL], CO2);
/* the leverage of a car (w, v): x0' (X'X)^-1 x0 = 1/n + d' C^-1 d, with d its distance from the means
   and C the centred cross-products of weight and volume (better conditioned than X'X itself) */
const LEV = (() => {
  const mw = mean(WT), mv = mean(VOL);
  let cww = 0, cvv = 0, cwv = 0;
  for (let i = 0; i < N; i++) { const a = WT[i] - mw, b = VOL[i] - mv; cww += a * a; cvv += b * b; cwv += a * b; }
  const det = cww * cvv - cwv * cwv;
  return { mw, mv, iww: cvv / det, ivv: cww / det, iwv: -cwv / det };
})();
const lev = (w, v) => { const a = w - LEV.mw, b = v - LEV.mv; return 1 / N + a * a * LEV.iww + 2 * a * b * LEV.iwv + b * b * LEV.ivv; };
const MAXLEV = Math.max(...WT.map((w, i) => lev(w, VOL[i])));
const GROUPS = [[v => v <= 1300, "--c4"], [v => v > 1300 && v <= 1600, "--post"], [v => v > 1600, "--warm"]];
const planeAt = (w, v) => MULTI.w[0] + MULTI.w[1] * w + MULTI.w[2] * v;
function drawSlice() {
  const W0 = +$("#sW").value, V0 = +$("#sV").value, cv = $("#sliceC");
  $("#sWV").textContent = comma(W0) + " kg"; $("#sVV").textContent = comma(V0) + " cm³";
  const fr = frame(cv, narrowOf(cv) ? 1.15 : 2.1, { x: [750, 1800], y: [85, 125], xlabel: "weight (kg)", ylabel: "CO2 (g/km)" });
  const { ctx, X, Y } = fr;
  plotLine(fr, [750, 1800], [SIMPLE.b0 + SIMPLE.b1 * 750, SIMPLE.b0 + SIMPLE.b1 * 1800], { color: css("--muted"), width: 2, dash: [7, 5] });
  GROUPS.forEach(([inG, col]) => {
    const idx = VOL.map((v, i) => i).filter(i => inG(VOL[i])), vm = mean(idx.map(i => VOL[i]));
    const lo = Math.min(...idx.map(i => WT[i])), hi = Math.max(...idx.map(i => WT[i]));
    plotLine(fr, [lo, hi], [planeAt(lo, vm), planeAt(hi, vm)], { color: css(col), width: 2 });
    idx.forEach(i => dot(ctx, X(WT[i]), Y(CO2[i]), 4, css(col)));
  });
  plotLine(fr, [750, 1800], [planeAt(750, V0), planeAt(1800, V0)], { color: css("--ink"), width: 2.4 });
  const pred = planeAt(W0, V0), h0 = lev(W0, V0);
  dot(ctx, X(W0), Y(clamp(pred, 85, 125)), 7, css("--surface"), css("--ink"), 2.6);
  let text = `Weight ${comma(W0)} kg, engine ${comma(V0)} cm³: predicted CO2 <b>${fmt(pred, 1)} g/km</b>. Every slice of the plane has the same slope, ${fmt(MULTI.w[1], 4)} g/km per kg, well below the simple regression's ${fmt(SIMPLE.b1, 4)}: among cars with the same engine size, extra weight goes with less extra CO2 than the dashed line suggests. The dashed line is steeper because heavier cars also carry bigger engines, and it credits weight with both.`;
  if (h0 > MAXLEV) text += ` <b>Careful:</b> no car in the table combines this weight with this engine size (leverage ${fmt(h0, 2)}, against at most ${fmt(MAXLEV, 2)} for the 36 cars), so this prediction is an extrapolation of the plane.`;
  caption($("#sliceCap"), text);
}
["#sW", "#sV"].forEach(id => $(id).addEventListener("input", drawSlice));

/* =====================================================================
   start-up
   ===================================================================== */
function drawAll() { drawHero(); drawBowl(); drawHand(); drawR2(); drawScat(); drawSlice(); }
drawAll();
onRedraw(drawAll);
})();
