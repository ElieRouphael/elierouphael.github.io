/* Satellite Pass Planning from Scratch, chapter 04, Seeing the Ground: the live demos.
   One pass over a target at an adjustable cross-track offset, with elevation and off-nadir curves, the exact
   window and the window sampled on the 10 s grid (hero); the triangle Earth centre - satellite - target drawn
   to scale; and the visibility windows of the small instance before and after the duration filter, with the
   error of the grid. Geometry from the shared port of satplan (interactive/sat.js), windows of the instance from
   interactive/ch04-data.js (written by code/ch04_numbers.py). Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, css, fmt, caption, fit, frame, plotLine, alpha, onRedraw, narrowOf, seg, histogram } = window.Kit;
const S = window.Sat, I = window.SAT_SMALL, D = window.SAT_CH04;
const R = S.R_EARTH, RAD = S.RAD, DEG = S.DEG;

/* ---------------- geometry helpers ---------------- */
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
/* Earth-fixed unit vector of an inertial position at time t (rotate by -OMEGA t about z). */
function efUnit(r, t) {
  const th = S.OMEGA_EARTH * t, c = Math.cos(th), s = Math.sin(th);
  return S.unit([r[0] * c + r[1] * s, -r[0] * s + r[1] * c, r[2]]);
}
/* The ground point at cross-track distance off_m (positive: left of the track) from the sub-satellite point at t0
   (as code/ch04_numbers.py offset_target). */
function offsetTarget(o, t0, off_m) {
  const p0 = efUnit(S.positionECI(o, t0), t0);
  let d = S.sub(efUnit(S.positionECI(o, t0 + 0.5), t0 + 0.5), efUnit(S.positionECI(o, t0 - 0.5), t0 - 0.5));
  d = S.unit(S.sub(d, S.scale(p0, S.dot(d, p0))));
  const c = cross(p0, d), b = off_m / R;
  const g = [0, 1, 2].map(k => Math.cos(b) * p0[k] + Math.sin(b) * c[k]);
  return { lat: Math.asin(g[2]) * DEG, lon: Math.atan2(g[1], g[0]) * DEG };
}
/* The exact window around t0 (1 s scan, edges bisected 50 times) and the window on the 10 s grid shifted by
   gridOff seconds (as code/ch04_numbers.py pass_window, which uses gridOff = 0). */
function passWindow(o, lat, lon, t0, maxOff, minEl, gridOff) {
  const span = 600, ts0 = Math.floor((t0 - span) / 10) * 10;
  const vis = t => S.visible(S.positionECI(o, t), S.groundECI(lat, lon, t), maxOff, minEl);
  const ts = [], m = [];
  for (let t = ts0; t <= t0 + span + 1e-9; t += 1) { ts.push(t); m.push(vis(t)); }
  const grid = [], mg = [];
  for (let t = Math.ceil((ts0 - gridOff) / 10) * 10 + gridOff; t <= t0 + span + 1e-9; t += 10) { grid.push(t); mg.push(vis(t)); }
  const rs = S.runs(m);
  if (!rs.length) return { a: null, b: null, ws: null, we: null, grid, mg };
  let best = rs[0];
  for (const r of rs) if (r[1] - r[0] > best[1] - best[0]) best = r;
  const edge = (tin, tout) => { for (let k = 0; k < 50; k++) { const tm = 0.5 * (tin + tout); if (vis(tm)) tin = tm; else tout = tm; } return 0.5 * (tin + tout); };
  const a = edge(ts[best[0]], ts[best[0]] - 1), b = edge(ts[best[1]], ts[best[1]] + 1);
  const rg = S.runs(mg).filter(r => grid[r[0]] <= b && grid[r[1]] >= a);
  return { a, b, ws: rg.length ? grid[rg[0][0]] : null, we: rg.length ? grid[rg[0][1]] : null, grid, mg };
}
/* Which limit fails just outside a window edge: "elevation" or "sensor cone". */
function bindingAt(o, lat, lon, t, maxOff, minEl) {
  const la = S.lookAngles(S.positionECI(o, t), S.groundECI(lat, lon, t));
  if (la.elevation < minEl) return "elevation";
  if (maxOff != null && la.offNadir > maxOff) return "sensor cone";
  return "–";
}

/* ---------------- hero: one pass ---------------- */
const HERO_ORBIT = S.orbit(550e3, 97.6, 0, 0), T0 = 643;
const hs = { off: 300, cone: 55, useCone: true, elev: 10, gridOff: 0 };
function drawHero() {
  const o = HERO_ORBIT, tg = offsetTarget(o, T0, hs.off * 1e3), maxOff = hs.useCone ? hs.cone : null;
  const w = passWindow(o, tg.lat, tg.lon, T0, maxOff, hs.elev, hs.gridOff);
  const half = hs.useCone ? 400 : 450, x0 = T0 - half, x1 = T0 + half;
  const ts = [], el = [], on = [];
  for (let t = x0; t <= x1 + 1e-9; t += 0.5) { const la = S.lookAngles(S.positionECI(o, t), S.groundECI(tg.lat, tg.lon, t)); ts.push(t); el.push(la.elevation); on.push(la.offNadir); }
  const cv = $("#heroC"), narrow = narrowOf(cv, 560);
  const fr = frame(cv, narrow ? 1.25 : 2.1, { x: [x0, x1], y: [-8, 90], yticks: [0, 15, 30, 45, 60, 75, 90], yfmt: v => v + "°", xlabel: "time t [s]", ylabel: "angle [deg]" });
  const { ctx, X, Y } = fr;
  // the exact window
  if (w.a != null) { ctx.fillStyle = alpha("--post", 0.13); ctx.fillRect(X(w.a), Y(90), X(w.b) - X(w.a), Y(-8) - Y(90)); }
  // limits
  ctx.save(); ctx.setLineDash([5, 4]); ctx.lineWidth = 1.3;
  ctx.strokeStyle = css("--post"); seg(ctx, X(x0), Y(hs.elev), X(x1), Y(hs.elev));
  if (hs.useCone) { ctx.strokeStyle = css("--warm"); seg(ctx, X(x0), Y(hs.cone), X(x1), Y(hs.cone)); }
  ctx.restore();
  plotLine(fr, ts, el, { color: css("--post"), width: 2 });
  plotLine(fr, ts, on, { color: css("--warm"), width: 2 });
  // the 10 s grid: a dot per sample along the bottom, filled when the target is visible
  const yDot = Y(-4);
  ctx.lineWidth = 1.2;
  w.grid.forEach((t, k) => {
    if (t < x0 || t > x1) return;
    ctx.beginPath(); ctx.arc(X(t), yDot, 2.4, 0, 2 * Math.PI);
    if (w.mg[k]) { ctx.fillStyle = css("--ink"); ctx.fill(); } else { ctx.strokeStyle = alpha("--muted", 0.8); ctx.stroke(); }
  });
  if (w.ws != null) {
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2;
    seg(ctx, X(w.ws), Y(-1), X(w.we), Y(-1)); seg(ctx, X(w.ws), Y(-1) - 4, X(w.ws), Y(-1) + 4); seg(ctx, X(w.we), Y(-1) - 4, X(w.we), Y(-1) + 4);
  }
  // readouts
  const maxEl = Math.max(...el), minOn = Math.min(...on);
  $("#rEl").textContent = fmt(maxEl, 1) + "°";
  $("#rOn").textContent = fmt(minOn, 1) + "°";
  const L = w.a != null ? w.b - w.a : 0, Ls = w.ws != null ? w.we - w.ws : null;
  $("#rExact").textContent = w.a != null ? `${fmt(L, 1)} s` : "not visible";
  $("#rSamp").textContent = Ls != null ? `${fmt(Ls, 0)} s (${fmt(w.ws, 0)} to ${fmt(w.we, 0)})` : "no sample";
  $("#rLost").textContent = Ls != null ? `${fmt(w.ws - w.a, 1)} + ${fmt(w.b - w.we, 1)} s` : "–";
  const bStart = w.a != null ? bindingAt(o, tg.lat, tg.lon, w.a - 0.01, maxOff, hs.elev) : "–", bEnd = w.a != null ? bindingAt(o, tg.lat, tg.lon, w.b + 0.01, maxOff, hs.elev) : "–";
  $("#rBind").textContent = bStart === bEnd ? bStart : `${bStart} / ${bEnd}`;
  const fits = [30, 60, 90, 120].filter(d => Ls != null && Ls >= d);
  const side = hs.off === 0 ? "directly under the ground track" : `${fmt(Math.abs(hs.off), 0)} km to the ${hs.off > 0 ? "left" : "right"} of the ground track`;
  caption($("#heroStatus"), `A satellite at 550 km and 97.6° passes over a target placed ${side}, near 40° N (the target is at ${fmt(tg.lat, 2)}°, ${fmt(tg.lon, 2)}°). Blue: the satellite's <b>elevation</b> above the target's horizon; orange: the target's <b>off-nadir angle</b> seen from the satellite. The target is visible while the blue curve is above its mask (${fmt(hs.elev, 0)}°, dashed) ${hs.useCone ? `and the orange curve below the sensor cone (${fmt(hs.cone, 0)}°, dashed)` : "(no sensor cone: the rule for ground stations)"}. ${w.a == null ? "<b>It never is</b>: the target is outside the footprint for the whole pass." : `The exact window (shaded) lasts <b>${fmt(L, 1)} s</b>; the limit that closes it is the ${bStart === bEnd ? bStart : bStart + " at the start and the " + bEnd + " at the end"}. On the package's 10 s grid${hs.gridOff ? ` (shifted by ${hs.gridOff} s)` : ""} (dots, filled when visible), the first and last visible samples give the recorded window, <b>${Ls != null ? fmt(Ls, 0) + " s" : "nothing"}</b>${Ls != null ? `, ${fmt(L - Ls, 1)} s shorter than the truth: ${fmt(w.ws - w.a, 1)} s lost at the start and ${fmt(w.b - w.we, 1)} s at the end` : ": the window falls between two samples"}. ${Ls != null ? (fits.length ? `A task of ${fits.join(", ")} s fits in it.` : "No task of 30 s or more fits: the scenario would drop this window.") : ""}`}`);
}
function heroControls() {
  const upd = () => {
    $("#hOffV").textContent = fmt(hs.off, 0) + " km"; $("#hConeV").textContent = hs.useCone ? fmt(hs.cone, 0) + "°" : "none"; $("#hElV").textContent = fmt(hs.elev, 0) + "°"; $("#hGV").textContent = fmt(hs.gridOff, 0) + " s";
    $("#hCone").disabled = !hs.useCone; $("#hNoCone").checked = !hs.useCone;
    drawHero();
  };
  $("#hOff").addEventListener("input", e => { hs.off = +e.target.value; upd(); });
  $("#hCone").addEventListener("input", e => { hs.cone = +e.target.value; upd(); });
  $("#hEl").addEventListener("input", e => { hs.elev = +e.target.value; upd(); });
  $("#hG").addEventListener("input", e => { hs.gridOff = +e.target.value; upd(); });
  $("#hNoCone").addEventListener("change", e => { hs.useCone = !e.target.checked; upd(); });
  $("#hTarget").addEventListener("click", () => { hs.useCone = true; hs.cone = 55; hs.elev = 10; $("#hCone").value = 55; $("#hEl").value = 10; upd(); });
  $("#hStation").addEventListener("click", () => { hs.useCone = false; hs.elev = 5; $("#hEl").value = 5; upd(); });
  upd();
}

/* ---------------- the triangle, to scale ---------------- */
const ts_ = { lam: 6, alt: 550 };
/* Elevation and off-nadir at the edge of a limit, from the law of sines (as code/ch04_numbers.py triangle). */
function edgeFromCone(h, eta) { const k = R / (R + h), c = Math.sin(eta * RAD) / k; if (c > 1) return null; const eps = Math.acos(c) * DEG; return { eps, eta, lam: 90 - eps - eta }; }
function edgeFromElev(h, eps) { const k = R / (R + h), eta = Math.asin(k * Math.cos(eps * RAD)) * DEG; return { eps, eta, lam: 90 - eps - eta }; }
function arcAt(ctx, x, y, r, a0, a1) { ctx.beginPath(); ctx.arc(x, y, r, Math.min(a0, a1), Math.max(a0, a1)); ctx.stroke(); }
function drawTriangle() {
  const cv = $("#triC"), narrow = narrowOf(cv);
  const { ctx, w, h } = fit(cv, narrow ? 1.0 : 1.8);
  const H = ts_.alt * 1e3, lam = ts_.lam * RAD;
  // to scale, zoomed on the part of the Earth near the satellite: the Earth's centre O is far below the picture
  const X1 = R * Math.sin(25 * RAD), xmin = -0.4 * X1, xmax = 1.08 * X1, ymin = R * Math.cos(26 * RAD), ymax = R + H + 260e3;
  const k = Math.min((w - 8) / (xmax - xmin), (h - 8) / (ymax - ymin));
  const ox = (w - (xmax - xmin) * k) / 2, oy = (h - (ymax - ymin) * k) / 2;
  const P = (x, y) => [ox + (x - xmin) * k, oy + (ymax - y) * k];
  const [Ox, Oy] = P(0, 0);
  const sat3 = [0, R + H, 0], tgt3 = [R * Math.sin(lam), R * Math.cos(lam), 0];
  const la = S.lookAngles(sat3, tgt3);   // the package's two dot products, in the triangle's plane
  const [sx, sy] = P(0, R + H), [px, py] = P(tgt3[0], tgt3[1]), [nx, ny] = P(0, R);
  // the Earth (a circle of radius R around O, mostly off the picture)
  ctx.fillStyle = css("--surface-2"); ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(Ox, Oy, R * k, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  // footprint edges for the package's limits: 55 deg cone (targets) and 5 deg mask (stations)
  const cone = edgeFromCone(H, 55), st5 = edgeFromElev(H, 5), m10 = edgeFromElev(H, 10);
  ctx.font = "10px " + css("--font-mono");
  const mark = (lamDeg, color, label, dy) => {
    const a = lamDeg * RAD, [x, y] = P(R * Math.sin(a), R * Math.cos(a));
    ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); seg(ctx, sx, sy, x, y); ctx.setLineDash([]);
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 2 * Math.PI); ctx.fill();
    ctx.textAlign = "left"; ctx.fillText(label, Math.min(x + 4, w - 64), y + dy);
  };
  if (cone) mark(cone.lam, css("--warm"), "55° cone", -6);
  mark(st5.lam, css("--muted"), "5° mask", 14);
  // the triangle: O -> S and O -> P run off the bottom of the picture toward the Earth's centre
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.moveTo(Ox, Oy); ctx.lineTo(sx, sy); ctx.lineTo(px, py); ctx.lineTo(Ox, Oy); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.textAlign = "left";
  ctx.fillText(`to the Earth's centre, ${fmt(R / 1e3, 0)} km below the surface`, Math.min(nx + 6, w - 250), h - 8);
  // lambda: the arc of the surface between the sub-satellite point and the target
  const angTo = (x0, y0, x1, y1) => Math.atan2(y1 - y0, x1 - x0);   // canvas angle (y down)
  ctx.font = "12px " + css("--font-mono"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
  if (ts_.lam > 0.05) {
    ctx.strokeStyle = css("--c4"); ctx.lineWidth = 3.5;
    ctx.beginPath(); ctx.arc(Ox, Oy, R * k, -Math.PI / 2, -Math.PI / 2 + lam); ctx.stroke();
    const am = -Math.PI / 2 + lam / 2;
    ctx.fillStyle = css("--c4"); ctx.fillText("λ", Ox + (R * k - 14) * Math.cos(am), Oy + (R * k - 14) * Math.sin(am));
  }
  // local horizontal at the target
  const ux = Math.sin(lam), uy = Math.cos(lam);   // up at the target, in picture axes (x right, y up)
  ctx.strokeStyle = alpha("--muted", 0.9); ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
  seg(ctx, px - 70 * uy, py - 70 * ux, px + 50 * uy, py + 50 * ux); ctx.setLineDash([]);
  ctx.lineWidth = 1.5;
  if (ts_.lam > 0.05) {
    // eta at the satellite, between nadir and the line of sight
    ctx.strokeStyle = css("--warm"); arcAt(ctx, sx, sy, 30, angTo(sx, sy, Ox, Oy), angTo(sx, sy, px, py));
    ctx.fillStyle = css("--warm"); { const a = 0.5 * (angTo(sx, sy, Ox, Oy) + angTo(sx, sy, px, py)); ctx.fillText("η", sx + 42 * Math.cos(a), sy + 42 * Math.sin(a)); }
    // epsilon at the target, between the horizontal (toward the satellite) and the line of sight
    const aS = angTo(px, py, sx, sy), aH = angTo(px, py, px - uy, py - ux);
    ctx.strokeStyle = css("--post"); arcAt(ctx, px, py, 34, aH, aS);
    ctx.fillStyle = css("--post"); { const a = 0.5 * (aH + aS); ctx.fillText("ε", px + 46 * Math.cos(a), py + 46 * Math.sin(a)); }
  }
  // points and labels
  for (const [x, y, c] of [[sx, sy, "--warm"], [px, py, "--post"], [nx, ny, "--ink"]]) { ctx.fillStyle = css(c); ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill(); }
  ctx.fillStyle = css("--ink"); ctx.textAlign = "right"; ctx.fillText("satellite", sx - 8, sy);
  ctx.fillText(narrow ? "nadir point" : "sub-satellite point", nx - 8, ny - 10);
  ctx.textAlign = "left"; ctx.fillText("target", px + 8, py + 14);
  ctx.textBaseline = "alphabetic";
  // readouts and caption
  const eps = la.elevation, eta = la.offNadir, rngKm = la.range / 1e3, kk = R / (R + H);
  $("#tEps").textContent = fmt(eps, 2) + "°"; $("#tEta").textContent = fmt(eta, 2) + "°"; $("#tLam").textContent = fmt(ts_.lam, 2) + "°";
  $("#tSum").textContent = fmt(eps + eta + ts_.lam, 2) + "°";
  $("#tSines").textContent = `${fmt(Math.sin(eta * RAD), 4)} = ${fmt(kk * Math.cos(eps * RAD), 4)}`;
  $("#tRng").textContent = fmt(rngKm, 0) + " km";
  $("#tGround").textContent = fmt(lam * R / 1e3, 0) + " km";
  const okT = eps >= 10 && eta <= 55, okS = eps >= 5;
  caption($("#triCap"), `To scale, at ${fmt(ts_.alt, 0)} km: the satellite, the Earth's centre and a target ${fmt(lam * R / 1e3, 0)} km from the sub-satellite point (Earth central angle [[\\lambda = ${fmt(ts_.lam, 2)}^\\circ]]). The satellite sees the target <b>${fmt(eta, 2)}°</b> off nadir and the target sees the satellite <b>${fmt(eps, 2)}°</b> above its horizon (dashed), at a slant range of ${fmt(rngKm, 0)} km; the three angles add up to [[${fmt(eps + eta + ts_.lam, 2)}^\\circ]]. With the package's limits, a target here is ${okT ? "<b>visible</b>" : "<b>not visible</b>"} (cone 55°, elevation 10°) and a ground station ${okS ? "is in contact" : "is not in contact"} (elevation 5°). ${cone && cone.eps >= 10 ? `The targets' footprint ends at [[\\lambda = ${fmt(cone.lam, 2)}^\\circ]] (${fmt(cone.lam * RAD * R / 1e3, 0)} km), where the elevation is ${fmt(cone.eps, 2)}°: the cone closes the window before the 10° mask can.` : `The targets' footprint ends at [[\\lambda = ${fmt(m10.lam, 2)}^\\circ]] (${fmt(m10.lam * RAD * R / 1e3, 0)} km), where the off-nadir angle is only ${fmt(m10.eta, 2)}°: ${cone ? "here the 10° mask closes the window before the 55° cone." : "at this altitude a 55° cone is wider than the Earth seen from the satellite, and only the elevation mask limits visibility."}`} The stations' footprint ends at [[\\lambda = ${fmt(st5.lam, 2)}^\\circ]] (${fmt(st5.lam * RAD * R / 1e3, 0)} km).`);
}
function triControls() {
  const upd = () => { $("#tLamV").textContent = fmt(ts_.lam, 2) + "°"; $("#tAltV").textContent = fmt(ts_.alt, 0) + " km"; drawTriangle(); };
  $("#tLamS").addEventListener("input", e => { ts_.lam = +e.target.value; upd(); });
  $("#tAltS").addEventListener("input", e => { ts_.alt = +e.target.value; upd(); });
  upd();
}

/* ---------------- the windows of the small instance ---------------- */
// D.windows rows: [sat, target, ws, we, duration, exact start, exact end, kept]
const W = D.windows.map(([sat, target, ws, we, dur, a, b, kept]) => ({ sat, target, ws, we, dur, a, b, kept: !!kept }));
const ws_ = { exact: false, check: null };
function drawWindows() {
  const cv = $("#winC"), narrow = narrowOf(cv);
  const len = q => ws_.exact ? q.b - q.a : q.we - q.ws;
  const kept = W.filter(q => q.kept).map(len), dropped = W.filter(q => !q.kept).map(len);
  const lo = 0, hi = 250, bins = 25;
  const count = vals => { const c = new Array(bins).fill(0); for (const v of vals) { const i = Math.min(bins - 1, Math.floor((v - lo) / ((hi - lo) / bins))); if (i >= 0) c[i]++; } return c; };
  const ck = count(kept), cd = count(dropped), top = Math.max(...ck.map((v, i) => v + cd[i]));
  const fr = frame(cv, narrow ? 1.5 : 2.6, { x: [lo, hi], y: [0, top + 2], xlabel: ws_.exact ? "exact window length [s]" : "window length on the 10 s grid [s]", ylabel: "windows" });
  const { ctx, X, Y } = fr, bw = (hi - lo) / bins;
  for (let i = 0; i < bins; i++) {
    const x0 = X(lo + i * bw) + 0.5, x1 = X(lo + (i + 1) * bw) - 0.5;
    if (ck[i]) { ctx.fillStyle = alpha("--post", 0.8); ctx.fillRect(x0, Y(ck[i]), x1 - x0, Y(0) - Y(ck[i])); }
    if (cd[i]) { ctx.fillStyle = alpha("--warm", 0.85); ctx.fillRect(x0, Y(ck[i] + cd[i]), x1 - x0, Y(ck[i]) - Y(ck[i] + cd[i])); }
  }
  ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = alpha("--ink", 0.6); ctx.lineWidth = 1;
  ctx.font = "10px " + css("--font-mono"); ctx.fillStyle = css("--muted"); ctx.textAlign = "center";
  for (const d of [30, 60, 90, 120]) { seg(ctx, X(d), Y(0), X(d), Y(top + 2)); ctx.fillText(d + " s", X(d), Y(top + 2) + 10); }
  ctx.restore();
  // the grid's error, for windows not cut by the horizon
  const inner = W.filter(q => q.ws > 0 && q.we < D.horizon);
  const err = inner.map(q => (q.ws - q.a) + (q.b - q.we));
  const cv2 = $("#errC");
  const ce = new Array(20).fill(0);
  for (const v of err) { const i = Math.floor(v); if (i >= 0 && i < 20) ce[i]++; }
  const fr2 = frame(cv2, narrow ? 2.2 : 4.2, { x: [0, 20], y: [0, Math.max(...ce) + 2], xticks: [0, 5, 10, 15, 20], xlabel: "exact length − recorded length [s]", ylabel: "windows" });
  histogram(fr2, err, 0, 20, 20, 1, alpha("--c4", 0.85));
  const mean = err.reduce((a, b) => a + b, 0) / err.length, mx = Math.max(...err);
  const marginal = W.filter(q => !q.kept && q.b - q.a >= q.dur);
  $("#wAll").textContent = String(W.length); $("#wKept").textContent = String(kept.length); $("#wDrop").textContent = String(dropped.length);
  $("#wMarg").textContent = String(marginal.length);
  $("#wErr").textContent = `${fmt(mean, 2)} s (max ${fmt(mx, 2)})`;
  const nT = new Set(W.map(q => q.target)).size, nTk = new Set(W.filter(q => q.kept).map(q => q.target)).size;
  const chk = ws_.check == null ? "The browser is recomputing every window with <code>Sat.windows</code>…" : ws_.check;
  caption($("#winCap"), `All <b>${W.length}</b> visibility windows of the small instance's 120 targets and 4 satellites over 6 h, before the duration filter, by ${ws_.exact ? "their exact length (edges bisected)" : "their length on the 10 s grid, [[t_{i_1} - t_{i_0}]]"}. Blue: the <b>${kept.length}</b> windows long enough for their target's listening time, which become the instance's opportunities; orange: the <b>${dropped.length}</b> that are too short and are dropped. The dashed lines are the four required durations. ${nT} targets have a window, ${nTk} keep at least one: ${nT - nTk} targets are lost to the filter entirely. Below, for the ${inner.length} windows not cut by the ends of the horizon, how much longer the exact window is than the recorded one: ${fmt(mean, 2)} s on average, never more than ${fmt(mx, 2)} s, so up to one grid step on each side. ${marginal.length} dropped window${marginal.length === 1 ? " was" : "s were"} long enough in truth (${marginal.map(q => `target ${q.target}: ${fmt(q.we - q.ws, 0)} s recorded, ${fmt(q.b - q.a, 1)} s exact, ${q.dur} s needed`).join("; ")}). ${chk}`);
}
function windowControls() {
  $("#wExact").addEventListener("click", () => { ws_.exact = !ws_.exact; $("#wExact").setAttribute("aria-pressed", String(ws_.exact)); drawWindows(); });
}
/* Recompute every window of the instance in the browser and compare with Python's (ch04-data.js). */
function checkWindows() {
  const cfg = I.config, mine = new Set(), theirs = new Set(W.map(q => `${q.sat}:${q.target}:${q.ws}:${q.we}`));
  I.orbits.forEach((ob, s) => {
    const o = S.orbit(ob[0], ob[1], ob[2], ob[3]);
    I.targets.forEach(([lat, lon], j) => {
      for (const [a, b] of S.windows(o, lat, lon, I.horizon, cfg.time_step_s, cfg.max_off_nadir_deg, cfg.min_elevation_deg)) mine.add(`${s}:${j}:${a}:${b}`);
    });
  });
  let same = 0; for (const k of theirs) if (mine.has(k)) same++;
  ws_.check = `Check: the browser recomputed the windows with <code>Sat.windows</code> and found ${mine.size}, of which ${same} match Python's ${theirs.size} exactly.`;
  drawWindows();
}

heroControls(); triControls(); windowControls(); drawWindows();
setTimeout(checkWindows, 60);
onRedraw(() => { drawHero(); drawTriangle(); drawWindows(); });
})();
