/* TrustDrive chapter 01, The vehicle model: the live demos.
   The constants are the course's (wheelbase 2.7 m, 15 m/s, sample time 0.05 s); the Euler
   and linearisation studies use the notebook's exact settings. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, int, caption, fit, frame, plotLine, alpha, onRedraw, narrowOf } = window.Kit;
const L = 2.7, G = 9.81, V = 15, TS = 0.05, LANE = 1.75;
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const num = x => String(+x.toFixed(1));                  // 15 rather than 15.0
const ROADS = { "0.9": "a dry road", "0.5": "a wet road", "0.2": "packed snow" };

/* One exact step of the kinematic bicycle (1): along the circle about the turning centre. */
function advance(p, delta, v, dt) {
  if (Math.abs(delta) < 1e-9) { p.X += v * Math.cos(p.psi) * dt; p.Y += v * Math.sin(p.psi) * dt; return; }
  const R = L / Math.tan(delta), dpsi = v / R * dt;
  p.X += R * (Math.sin(p.psi + dpsi) - Math.sin(p.psi));
  p.Y -= R * (Math.cos(p.psi + dpsi) - Math.cos(p.psi));
  p.psi += dpsi;
}
/* the verdict of Part 6's rule of thumb */
function verdict(ay, mu) {
  if (ay <= 0.5 * mu * G) return ["ok", "valid"];
  if (ay <= mu * G) return ["warn", "doubtful: tyres slip"];
  return ["bad", "beyond the grip"];
}
/* a line between two points in screen pixels */
function seg(ctx, x0, y0, x1, y1) { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }
/* the car seen from above: body, chassis line and the two wheels, in screen units */
function drawCar(ctx, P, psi, delta, s) {
  const c = Math.cos(psi), sn = Math.sin(psi);
  const pt = (along, across) => P(along * c - across * sn, along * sn + across * c);
  // body: from 0.85 m behind the rear axle to 0.85 m beyond the front one, 1.8 m wide
  const corners = [[-0.85, -0.9], [L + 0.85, -0.9], [L + 0.85, 0.9], [-0.85, 0.9]].map(([a, b]) => pt(a, b));
  ctx.beginPath(); corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
  ctx.fillStyle = alpha("--ink", 0.07); ctx.fill(); ctx.strokeStyle = alpha("--ink", 0.45); ctx.lineWidth = 1; ctx.stroke();
  const [rx, ry] = pt(0, 0), [fx, fy] = pt(L, 0);
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; seg(ctx, rx, ry, fx, fy);
  ctx.lineCap = "round"; ctx.lineWidth = Math.max(3.5, 0.3 * s);
  const half = 0.4;
  seg(ctx, ...pt(-half, 0), ...pt(half, 0));                                       // rear wheel, along the heading
  const fc = Math.cos(psi + delta), fs = Math.sin(psi + delta);
  seg(ctx, fx - half * fc * s, fy + half * fs * s, fx + half * fc * s, fy - half * fs * s);   // front wheel, steered
  ctx.lineCap = "butt";
  return { rear: [rx, ry], front: [fx, fy] };
}

/* =====================================================================
   THE DEMO: STEER THE BICYCLE
   ===================================================================== */
const drive = { X: 0, Y: 0, psi: Math.PI / 2, trail: [[0, 0]], running: false, lastT: 0 };
const dDelta = () => +$("#dDelta").value, dSpeed = () => +$("#dSpeed").value, dMu = () => +$("#dRoad").value;
function drawDrive() {
  const cv = $("#driveC"), { ctx, w, h } = fit(cv, narrowOf(cv) ? 1.1 : 1.55);
  const delta = dDelta(), R = Math.abs(delta) < 1e-4 ? Infinity : L / Math.tan(delta);
  const viewW = clamp(2.4 * Math.abs(R) + 8, 26, 70), s = w / viewW;
  // the camera follows the middle of the car
  const Xc = drive.X + (L / 2) * Math.cos(drive.psi), Yc = drive.Y + (L / 2) * Math.sin(drive.psi);
  const P = (dx, dy) => [w / 2 + (drive.X + dx - Xc) * s, h / 2 - (drive.Y + dy - Yc) * s];
  const W = (X, Y) => [w / 2 + (X - Xc) * s, h / 2 - (Y - Yc) * s];
  // ground grid, every 5 m
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  const halfW = viewW / 2 + 5, halfH = h / s / 2 + 5;
  for (let gx = Math.floor((Xc - halfW) / 5) * 5; gx <= Xc + halfW; gx += 5) { const [x] = W(gx, 0); seg(ctx, x, 0, x, h); }
  for (let gy = Math.floor((Yc - halfH) / 5) * 5; gy <= Yc + halfH; gy += 5) { const [, y] = W(0, gy); seg(ctx, 0, y, w, y); }
  // where the rear axle went
  if (drive.trail.length > 1) {
    ctx.strokeStyle = alpha("--path", 0.75); ctx.lineWidth = 2; ctx.beginPath();
    drive.trail.forEach(([X, Y], i) => { const [x, y] = W(X, Y); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.lineTo(...P(0, 0)); ctx.stroke();
  }
  // where it will go: the circle about the turning centre, or a straight line
  ctx.strokeStyle = css("--path"); ctx.lineWidth = 1.5; ctx.setLineDash([2, 5]);
  const cx = -R * Math.sin(drive.psi), cy = R * Math.cos(drive.psi);           // turning centre, relative to the rear axle
  if (isFinite(R)) { const [ix, iy] = P(cx, cy); ctx.beginPath(); ctx.arc(ix, iy, Math.abs(R) * s, 0, 2 * Math.PI); ctx.stroke(); }
  else seg(ctx, ...P(0, 0), ...P(200 * Math.cos(drive.psi), 200 * Math.sin(drive.psi)));
  ctx.setLineDash([]);
  const car = drawCar(ctx, P, drive.psi, delta, s);
  if ($("#dIcr").checked && isFinite(R)) {
    const [ix, iy] = P(cx, cy);
    ctx.strokeStyle = css("--geom"); ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
    seg(ctx, ix, iy, ...car.rear); seg(ctx, ix, iy, ...car.front); ctx.setLineDash([]);
    ctx.fillStyle = css("--geom"); ctx.beginPath(); ctx.arc(ix, iy, 4.5, 0, 2 * Math.PI); ctx.fill();
    if (ix > 0 && ix < w && iy > 0 && iy < h) {                                 // label below the dot, clear of the radii
      ctx.font = "600 11px " + css("--font-ui"); ctx.textAlign = "center";
      ctx.fillText("turning centre", clamp(ix, 45, w - 45), iy + 18);
    }
  }
  readouts();
}
function readouts() {
  const delta = dDelta(), v = dSpeed(), mu = dMu();
  const straight = Math.abs(delta) < 1e-4, R = L / Math.tan(Math.abs(delta)), ay = v * v * Math.abs(Math.tan(delta)) / L;
  const [cls, word] = verdict(ay, mu);
  $("#rR").textContent = straight ? "straight line" : `${fmt(R, R < 100 ? 2 : 0)} m, ${delta > 0 ? "left" : "right"}`;
  $("#rYaw").textContent = `${fmt(v * Math.tan(delta) / L, 3)} rad/s`;
  $("#rAy").textContent = `${fmt(ay, 2)} m/s² (${fmt(ay / G, 2)} g)`;
  $("#rValid").innerHTML = `<span class="verdict ${cls}">${word}</span>`;
}
function driveStatus() {
  const delta = dDelta(), v = dSpeed(), mu = dMu(), road = ROADS[$("#dRoad").value];
  $("#dDeltaV").textContent = `${fmt(delta, 3)} rad (${fmt(delta * 180 / Math.PI, 1)}°)`;
  $("#dSpeedV").textContent = `${num(v)} m/s (${int(v * 3.6)} km/h)`;
  const el = $("#driveStatus");
  if (Math.abs(delta) < 1e-4) { el.innerHTML = "Wheels straight: the turning centre is infinitely far away, the car drives a straight line, and no sideways acceleration is needed."; return; }
  const R = L / Math.tan(Math.abs(delta)), ay = v * v / R, side = delta > 0 ? "left" : "right";
  const [cls] = verdict(ay, mu);
  caption(el,
    `The turning centre is <b>${fmt(R, 2)} m</b> to the ${side} of the rear axle, so the car drives a circle of radius [[R = \\frac{L}{\\tan\\delta}]]. ` +
    `At ${num(v)} m/s that takes a sideways acceleration of <b>${fmt(ay, 2)} m/s²</b> (${fmt(ay / G, 2)} g). ` +
    (cls === "ok" ? `On ${road} that is below half the grip (${fmt(0.5 * mu * G, 2)} m/s²): the kinematic model can be trusted.`
      : cls === "warn" ? `On ${road} that is more than half the grip (${fmt(0.5 * mu * G, 2)} m/s²): the tyres start to slip, and the kinematic model starts to drift from reality.`
      : `That is more than the tyres can give on ${road} (${fmt(mu * G, 2)} m/s²): a real car would slide off this circle, but the kinematic model, which knows nothing about tyres, drives it anyway.`));
}
function setRunning(on) {
  drive.running = on;
  $("#dRun").setAttribute("aria-pressed", on ? "true" : "false");
  $("#dRun").textContent = on ? "pause" : "drive";
  if (on) { drive.lastT = performance.now(); requestAnimationFrame(tick); }
}
function tick(now) {
  if (!drive.running) return;
  const dt = Math.min(0.1, Math.max(0, (now - drive.lastT) / 1000));
  drive.lastT = now;
  advance(drive, dDelta(), dSpeed(), dt);
  const [lx, ly] = drive.trail[drive.trail.length - 1];
  if (Math.hypot(drive.X - lx, drive.Y - ly) > 0.5) { drive.trail.push([drive.X, drive.Y]); if (drive.trail.length > 900) drive.trail.shift(); }
  drawDrive();
  requestAnimationFrame(tick);
}
$("#dRun").addEventListener("click", () => setRunning(!drive.running));
$("#dReset").addEventListener("click", () => { Object.assign(drive, { X: 0, Y: 0, psi: Math.PI / 2, trail: [[0, 0]] }); drawDrive(); });
["#dDelta", "#dSpeed"].forEach(id => $(id).addEventListener("input", () => { driveStatus(); if (!drive.running) drawDrive(); }));
["#dRoad", "#dIcr"].forEach(id => $(id).addEventListener("change", () => { driveStatus(); if (!drive.running) drawDrive(); }));

/* =====================================================================
   PART 1: WHERE THE TURNING CENTRE IS
   ===================================================================== */
function drawGeo() {
  const delta = +$("#gDelta").value, R = L / Math.tan(delta);
  $("#gDeltaV").textContent = `${fmt(delta, 3)} rad (${fmt(delta * 180 / Math.PI, 1)}°)`;
  const cv = $("#geoC"), { ctx, w, h } = fit(cv, narrowOf(cv) ? 0.95 : 1.7);
  const x0 = -1.6, x1 = L + 1.8, y0 = -2.1, y1 = R + 1.0;
  const s = Math.min(w / (x1 - x0), h / (y1 - y0)), ox = (w - (x1 - x0) * s) / 2 - x0 * s, oy = h - (h - (y1 - y0) * s) / 2 + y0 * s;
  const P = (x, y) => [ox + x * s, oy - y * s];
  // the path of the rear axle, dotted
  ctx.strokeStyle = css("--path"); ctx.lineWidth = 1.5; ctx.setLineDash([2, 4]);
  const [ix, iy] = P(0, R);
  ctx.beginPath(); ctx.arc(ix, iy, R * s, Math.PI / 2 - 0.9, Math.PI / 2 + 0.45); ctx.stroke(); ctx.setLineDash([]);
  // the heading, continued past the front wheel, to show where delta is measured from
  ctx.strokeStyle = alpha("--muted", 0.6); ctx.lineWidth = 1; seg(ctx, ...P(L, 0), ...P(L + 1.6, 0));
  const car = drawCar(ctx, (dx, dy) => P(dx, dy), 0, delta, s);
  // the two radii, from the turning centre to each wheel
  ctx.strokeStyle = css("--geom"); ctx.lineWidth = 1.3; ctx.setLineDash([5, 4]);
  seg(ctx, ix, iy, ...car.rear); seg(ctx, ix, iy, ...car.front); ctx.setLineDash([]);
  ctx.fillStyle = css("--geom"); ctx.beginPath(); ctx.arc(ix, iy, 5, 0, 2 * Math.PI); ctx.fill();
  // right angles at both wheels
  const sq = Math.min(10, 0.35 * s);
  ctx.strokeStyle = css("--geom"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(car.rear[0] + sq, car.rear[1]); ctx.lineTo(car.rear[0] + sq, car.rear[1] - sq); ctx.lineTo(car.rear[0], car.rear[1] - sq); ctx.stroke();
  const u = [Math.cos(delta), -Math.sin(delta)], n = [-Math.sin(delta), -Math.cos(delta)];   // along and towards the centre, screen coordinates
  const [fx, fy] = car.front;
  ctx.beginPath(); ctx.moveTo(fx - u[0] * sq, fy - u[1] * sq); ctx.lineTo(fx - u[0] * sq + n[0] * sq, fy - u[1] * sq + n[1] * sq); ctx.lineTo(fx + n[0] * sq, fy + n[1] * sq); ctx.stroke();
  // delta, twice: between heading and wheel at the front, between the radii at the centre
  const arcR = Math.max(22, 0.9 * s);
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.arc(fx, fy, arcR, -delta, 0); ctx.stroke();
  ctx.beginPath(); ctx.arc(ix, iy, arcR * 1.4, Math.PI / 2 - delta, Math.PI / 2); ctx.stroke();
  ctx.font = "italic 600 14px " + css("--font-body"); ctx.fillStyle = css("--ink"); ctx.textAlign = "left";
  ctx.fillText("δ", fx + arcR + 4, fy - arcR * Math.sin(delta / 2) + 4);
  ctx.fillText("δ", ix + arcR * 1.4 * Math.sin(delta / 2) + 3, iy + arcR * 1.4 * Math.cos(delta / 2) + 12);
  ctx.fillText("L", ...P(L / 2 - 0.1, -0.75));
  ctx.fillStyle = css("--geom"); ctx.fillText("R", ...P(-0.75, R / 2));
  ctx.font = "600 11px " + css("--font-ui"); ctx.fillText("turning centre", ix + 9, iy - 6);
  ctx.fillStyle = css("--path"); ctx.textAlign = "left";                        // under the car, where the dotted arc is
  ctx.fillText("path of the rear axle (dotted)", 4, P(0, -0.9)[1] + 16);
  caption($("#geoCap"),
    `[[R = \\frac{L}{\\tan\\delta} = \\frac{2.7}{${fmt(Math.tan(delta), 3)}} = ${fmt(R, 2)}]] m. ` +
    `Both radii meet the wheels at right angles, so the angle between them is [[\\delta]] too, and in the right triangle [[\\tan\\delta = \\frac{L}{R}]]. ` +
    (Math.abs(delta - 0.45) < 1e-9 ? "This is the notebook's figure: for 0.45 rad, 5.59 m." : "Steer harder and the turning centre comes closer."));
}
$("#gDelta").addEventListener("input", drawGeo);

/* =====================================================================
   PART 2: A BEND, WITH AND WITHOUT FEED-FORWARD
   ===================================================================== */
const bend = { steer: "ff", sim: null };
function simulateBend() {
  const kappa = +$("#bKappa").value, ey0 = +$("#bEy").value, T = 6, h = 0.002;
  const delta = bend.steer === "ff" ? Math.atan(L * kappa) : 0;
  const car = { X: 0, Y: ey0, psi: 0 };
  let ey = ey0, epsi = 0;
  const rho = kappa ? 1 / kappa : Infinity;
  const offset = (X, Y) => (kappa ? rho - Math.sign(rho) * Math.hypot(X, Y - rho) : Y);   // exact, for a circular road
  const out = { t: [0], exact: [ey0], model: [ey0], X: [0], Y: [ey0], kappa, ey0, delta };
  const n = Math.round(T / h), every = Math.round(0.05 / h);
  for (let k = 1; k <= n; k++) {
    advance(car, delta, V, h);
    [ey, epsi] = [ey + V * Math.sin(epsi) * h, epsi + (V / L * Math.tan(delta) - V * kappa) * h];   // model (2)
    if (k % every === 0) { out.t.push(k * h); out.exact.push(offset(car.X, car.Y)); out.model.push(ey); out.X.push(car.X); out.Y.push(car.Y); }
  }
  bend.sim = out;
  drawBend();
}
function drawBend() {
  const S = bend.sim, { kappa, ey0, delta } = S;
  $("#bKappaV").textContent = `${fmt(kappa, 3)} per m` + (kappa ? ` (radius ${int(Math.abs(1 / kappa))} m)` : " (straight)");
  $("#bEyV").textContent = `${fmt(ey0, 1)} m`;
  $$("#bend [data-steer]").forEach(b => b.setAttribute("aria-pressed", b.dataset.steer === bend.steer ? "true" : "false"));
  // the road: centreline and lane edges along 100 m
  const road = (sArc, off) => {
    if (!kappa) return [sArc, off];
    const a = kappa * sArc;
    return [Math.sin(a) / kappa - off * Math.sin(a), (1 - Math.cos(a)) / kappa + off * Math.cos(a)];
  };
  const ss = Array.from({ length: 101 }, (_, i) => i);
  const edges = [-LANE, 0, LANE].map(off => ss.map(sa => road(sa, off)));
  const xsAll = [...edges.flat().map(p => p[0]), ...S.X], ysAll = [...edges.flat().map(p => p[1]), ...S.Y];
  let xlo = Math.min(...xsAll), xhi = Math.max(...xsAll), ylo = Math.min(...ysAll), yhi = Math.max(...ysAll);
  const cvT = $("#bendTop"), aspect = narrowOf(cvT) ? 1.25 : 1.15;
  const { ctx, w, h } = fit(cvT, aspect);
  // equal scales on both axes
  const pad = 10, sc = Math.min((w - 2 * pad) / Math.max(1, xhi - xlo), (h - 2 * pad) / Math.max(1, yhi - ylo));
  const ox = (w - (xhi - xlo) * sc) / 2 - xlo * sc, oy = (h + (yhi - ylo) * sc) / 2 + ylo * sc;
  const P = ([x, y]) => [ox + x * sc, oy - y * sc];
  const line = (pts, color, width, dash) => { ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.beginPath(); pts.forEach((p, i) => { const [x, y] = P(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.setLineDash([]); };
  ctx.fillStyle = alpha("--muted", 0.12);                                     // the lane, filled
  ctx.beginPath(); edges[0].forEach((p, i) => { const [x, y] = P(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  for (let i = edges[2].length - 1; i >= 0; i--) ctx.lineTo(...P(edges[2][i])); ctx.closePath(); ctx.fill();
  line(edges[0], alpha("--muted", 0.7), 1); line(edges[2], alpha("--muted", 0.7), 1); line(edges[1], alpha("--muted", 0.7), 1, [6, 5]);
  line(S.X.map((x, i) => [x, S.Y[i]]), css("--path"), 2.2);
  ctx.fillStyle = css("--path"); ctx.beginPath(); ctx.arc(...P([S.X[0], S.Y[0]]), 3.5, 0, 2 * Math.PI); ctx.fill();
  // the offset over time
  const vals = [...S.exact, ...S.model, LANE, -LANE];
  const lo = clamp(Math.min(...vals), -8, 0) - 0.3, hi = clamp(Math.max(...vals), 0, 8) + 0.3;
  const cvE = $("#bendEy");
  const fr = frame(cvE, aspect, { x: [0, 6], y: [lo, hi], xlabel: "time (s)", ylabel: "offset (m)", yfmt: v => fmt(v, Math.abs(hi - lo) < 3 ? 1 : 0) });
  fr.ctx.strokeStyle = alpha("--muted", 0.7); fr.ctx.setLineDash([3, 4]);
  [-LANE, LANE].forEach(y => seg(fr.ctx, fr.X(0), fr.Y(y), fr.X(6), fr.Y(y))); fr.ctx.setLineDash([]);
  plotLine(fr, S.t, S.exact, { color: css("--path"), width: 2.2 });
  plotLine(fr, S.t, S.model, { color: css("--warm"), width: 2, dash: [6, 4] });
  // what happened
  const end = S.exact[S.exact.length - 1], endM = S.model[S.model.length - 1];
  const leave = S.t.find((t, i) => Math.abs(S.exact[i]) > LANE);
  let text;
  if (bend.steer === "ff") {
    text = kappa
      ? `The feed-forward steering, [[\\delta = \\arctan(L\\kappa) = ${fmt(delta, 4)}]] rad, gives the car exactly the bend's radius, so model (2) predicts the offset stays at ${fmt(ey0, 1)} m. `
      : "On a straight road the feed-forward is zero: the wheels stay straight. ";
    text += !ey0 ? "Starting on the centreline, the car stays on it: the curvature, a known disturbance, is cancelled before any error appears."
      : kappa ? `The exact geometry says it ends at <b>${fmt(end, 2)} m</b> after 6 s: the car's circle has the road's radius, but its centre is shifted by the offset. That is the [[\\kappa e_y]] term that model (2) drops; it matters here because the bend is long and the car starts off-centre (exercise 3).`
      : "The offset stays where it started: nothing turns the car or the road.";
  } else {
    text = kappa
      ? `With the wheels straight, the lane turns away at [[v\\kappa = ${fmt(V * kappa, 2)}]] rad/s while the car does not. The heading error grows steadily, the offset grows like the square of time, ` +
        (leave !== undefined ? `and the car leaves the lane after <b>${fmt(leave, 2)} s</b>.` : `and after 6 s the car is ${fmt(end, 2)} m off.`) +
        ` Model (2) agrees while the car is near the lane (${fmt(endM, 1)} m against ${fmt(end, 1)} m at the end).`
      : "On a straight road, straight wheels keep the offset where it started.";
  }
  caption($("#bendCap"), text);
}
$$("#bend [data-steer]").forEach(b => b.addEventListener("click", () => { bend.steer = b.dataset.steer; simulateBend(); }));
["#bKappa", "#bEy"].forEach(id => $(id).addEventListener("input", simulateBend));

/* =====================================================================
   PART 3: FORWARD EULER IS FIRST ORDER (the notebook's settings)
   ===================================================================== */
const EU = { delta: 0.1, kappa: 0, x0: [0.2, 0.05], T: 2 };
const STEPS = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125], NB_STEPS = STEPS.slice(1);
const fEu = x => [V * Math.sin(x[1]), V / L * Math.tan(EU.delta) - V * EU.kappa];
const eulerStep = (x, hh) => { const d = fEu(x); return [x[0] + hh * d[0], x[1] + hh * d[1]]; };
function rk4Step(x, hh) {
  const add = (a, b, c) => [a[0] + c * b[0], a[1] + c * b[1]];
  const k1 = fEu(x), k2 = fEu(add(x, k1, hh / 2)), k3 = fEu(add(x, k2, hh / 2)), k4 = fEu(add(x, k3, hh));
  return [x[0] + hh / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), x[1] + hh / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])];
}
const eu = (() => {
  const ref = { t: [0], y: [EU.x0[0]] };
  let x = EU.x0.slice();
  const n = Math.round(EU.T / 1e-4);
  for (let k = 1; k <= n; k++) { x = rk4Step(x, 1e-4); if (k % 100 === 0) { ref.t.push(k * 1e-4); ref.y.push(x[0]); } }
  const exact = x;
  const runs = STEPS.map(hh => {
    let z = EU.x0.slice(); const t = [0], y = [z[0]];
    for (let k = 1; k <= Math.round(EU.T / hh); k++) { z = eulerStep(z, hh); t.push(k * hh); y.push(z[0]); }
    return { h: hh, t, y, err: Math.hypot(z[0] - exact[0], z[1] - exact[1]) };
  });
  // least-squares slope of log(error) against log(step), over the notebook's five steps
  const pts = runs.filter(r => NB_STEPS.includes(r.h)).map(r => [Math.log(r.h), Math.log(r.err)]);
  const mx = pts.reduce((a, p) => a + p[0], 0) / pts.length, my = pts.reduce((a, p) => a + p[1], 0) / pts.length;
  const slope = pts.reduce((a, p) => a + (p[0] - mx) * (p[1] - my), 0) / pts.reduce((a, p) => a + (p[0] - mx) ** 2, 0);
  return { ref, runs, slope };
})();
const lg = Math.log10;
function drawEuler() {
  const i = +$("#eStep").value, run = eu.runs[i];
  $("#eStepV").textContent = `${run.h} s (${int(EU.T / run.h)} steps)`;
  const cvP = $("#eulPath"), aspect = narrowOf(cvP) ? 1.3 : 1.15;
  const ys = [...eu.ref.y, ...run.y];
  const fr = frame(cvP, aspect, { x: [0, EU.T], y: [Math.min(...ys) - 0.5, Math.max(...ys) + 0.5], xlabel: "time (s)", ylabel: "offset (m)" });
  plotLine(fr, eu.ref.t, eu.ref.y, { color: css("--ink"), width: 1.6 });
  plotLine(fr, run.t, run.y, { color: css("--path"), width: 2 });
  fr.ctx.fillStyle = css("--path");
  if (run.t.length <= 80) run.t.forEach((t, k) => { fr.ctx.beginPath(); fr.ctx.arc(fr.X(t), fr.Y(run.y[k]), 3, 0, 2 * Math.PI); fr.ctx.fill(); });
  // error against step, log-log, the step decreasing to the right as in the notebook
  const errs = eu.runs.map(r => r.err), elo = Math.floor(lg(Math.min(...errs)) - 0.2), ehi = Math.ceil(lg(Math.max(...errs)) + 0.1);
  const gr = frame($("#eulErr"), aspect, { x: [lg(0.5), lg(0.01)], y: [elo, ehi], xticks: STEPS.filter((_, k) => k % 2 === 0).map(lg), xfmt: v => String(+(10 ** v).toPrecision(2)),
    yticks: Array.from({ length: ehi - elo + 1 }, (_, k) => elo + k), yfmt: v => String(+(10 ** v).toPrecision(1)), xlabel: "step (s)", ylabel: "error (m)" });
  plotLine(gr, STEPS.map(lg), STEPS.map(hh => lg(0.4 * eu.runs[0].err * hh / STEPS[0])), { color: alpha("--muted", 0.8), width: 1.2, dash: [5, 4] });   // slope 1, drawn below the data
  plotLine(gr, STEPS.map(lg), errs.map(lg), { color: css("--path"), width: 2 });
  eu.runs.forEach((r, k) => { gr.ctx.fillStyle = k === i ? css("--geom") : css("--path"); gr.ctx.beginPath(); gr.ctx.arc(gr.X(lg(r.h)), gr.Y(lg(r.err)), k === i ? 6 : 3.5, 0, 2 * Math.PI); gr.ctx.fill(); });
  caption($("#eulCap"),
    `With a step of ${run.h} s, Euler ends <b>${fmt(run.err, 3)} m</b> away from the exact state at 2 s (red dot). ` +
    (i > 0 ? `With twice the step it was ${fmt(eu.runs[i - 1].err, 3)} m: the ratio is ${fmt(eu.runs[i - 1].err / run.err, 2)}. ` : "") +
    `Each halving of the step about halves the error, a line of slope 1 on the log-log plot, parallel to the dashed guide. Over the notebook's five steps the fitted slope is <b>${fmt(eu.slope, 2)}</b>: forward Euler is first order.`);
}
$("#eStep").addEventListener("input", drawEuler);

/* =====================================================================
   PART 4: THE LINEAR MODEL AGAINST THE NONLINEAR ONE
   ===================================================================== */
function drawLin() {
  const e0 = +$("#lE0").value, d = +$("#lD").value, N = 150;
  $("#lE0V").textContent = `${fmt(e0, 2)} rad`;
  $("#lDV").textContent = `${fmt(d, 3)} rad`;
  let xn = [0, e0], xl = [0, e0];
  const t = [0], yn = [0], yl = [0];
  for (let k = 1; k <= N; k++) {
    xn = [xn[0] + V * Math.sin(xn[1]) * TS, xn[1] + V / L * Math.tan(d) * TS];
    xl = [xl[0] + V * TS * xl[1], xl[1] + V * TS / L * d];
    t.push(k * TS); yn.push(xn[0]); yl.push(xl[0]);
  }
  const cv = $("#linC");
  const top = Math.max(1, ...yn, ...yl);
  const fr = frame(cv, narrowOf(cv) ? 1.4 : 2.6, { x: [0, N * TS], y: [Math.min(0, ...yn, ...yl), top * 1.05], xlabel: "time (s)", ylabel: "offset (m)" });
  plotLine(fr, t, yn, { color: css("--path"), width: 2.2 });
  plotLine(fr, t, yl, { color: css("--warm"), width: 2, dash: [6, 4] });
  const at2 = Math.round(2 / TS), gap2 = Math.abs(yn[at2] - yl[at2]), gapEnd = Math.abs(yn[N] - yl[N]);
  const eEnd = xn[1], shrink = eEnd > 1e-6 ? 1 - Math.sin(eEnd) / eEnd : 0;
  caption($("#linCap"),
    !e0 && !d ? "No initial heading error and no steering: the car stays on the centreline in both models."
    : `After 2 s the two models differ by <b>${fmt(gap2, 2)} m</b>; after 7.5 s by <b>${fmt(gapEnd, 2)} m</b> (${fmt(yn[N], 2)} m against ${fmt(yl[N], 2)} m). ` +
      `By then the steering has turned the heading error to ${fmt(eEnd, 2)} rad, where [[\\sin e_\\psi]] is ${fmt(100 * shrink, 1)}% smaller than [[e_\\psi]]: the linear model overestimates how fast the car drifts sideways. ` +
      (e0 <= 0.1 ? "Even from a small initial angle the gap opens, because the constant steering keeps turning the car." : "The larger the angles, the faster the gap opens."));
}
["#lE0", "#lD"].forEach(id => $(id).addEventListener("input", drawLin));

/* =====================================================================
   PART 6: SIDEWAYS ACCELERATION AGAINST THE TWO LIMITS
   ===================================================================== */
const KAPPAS = [[0.012, "nominal road", "--path"], [0.022, "edge of the operating domain", "--c5"], [0.04, "out-of-distribution bend", "--warm"]];
$("#valLegend").innerHTML = KAPPAS.map(([k, name, col]) => `<span class="key"><span class="sw" style="background:var(${col})"></span>${name}, ${k}</span>`).join("") +
  `<span class="key"><span class="sw" style="background:var(--geom)"></span>half the grip (dashed) and the grip</span>`;
function drawVal() {
  const v = +$("#vSpeed").value, mu = +$("#vRoad").value, road = ROADS[$("#vRoad").value];
  $("#vSpeedV").textContent = `${num(v)} m/s (${int(v * 3.6)} km/h)`;
  const cv = $("#valC"), ymax = Math.max(12, mu * G * 1.5);
  const fr = frame(cv, narrowOf(cv) ? 1.35 : 2.4, { x: [5, 30], y: [0, ymax], xlabel: "speed (m/s)", ylabel: "sideways acceleration (m/s²)" });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--geom"); ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 4]); seg(ctx, X(5), Y(0.5 * mu * G), X(30), Y(0.5 * mu * G)); ctx.setLineDash([]);
  seg(ctx, X(5), Y(mu * G), X(30), Y(mu * G));
  const vs = Array.from({ length: 101 }, (_, i) => 5 + 0.25 * i);
  KAPPAS.forEach(([k, , col]) => plotLine(fr, vs, vs.map(s => s * s * k), { color: css(col), width: 2.2 }));
  ctx.strokeStyle = alpha("--ink", 0.5); ctx.lineWidth = 1; seg(ctx, X(v), Y(0), X(v), fr.pad.t);
  KAPPAS.forEach(([k, , col]) => { const ay = v * v * k; if (ay <= ymax) { ctx.fillStyle = css(col); ctx.beginPath(); ctx.arc(X(v), Y(ay), 4.5, 0, 2 * Math.PI); ctx.fill(); } });
  const SHORT = { ok: "valid", warn: "doubtful", bad: "beyond grip" };
  $("#valT tbody").innerHTML = KAPPAS.map(([k]) => {
    const ay = v * v * k, [cls] = verdict(ay, mu);
    return `<tr><td>${fmt(k, 3)}<span class="sub">radius ${fmt(1 / k, 1)} m</span></td><td>${fmt(ay, 2)} m/s²<span class="sub">${fmt(ay / G, 2)} g</span></td><td class="words"><span class="verdict ${cls}">${SHORT[cls]}</span></td></tr>`;
  }).join("");
  caption($("#valCap"),
    `On ${road}, the kinematic model holds below <b>${fmt(0.5 * mu * G, 2)} m/s²</b> (half the grip, dashed) and the tyres give up at <b>${fmt(mu * G, 2)} m/s²</b>. ` +
    `Sideways acceleration grows with the square of the speed, [[a_y = v^2\\kappa]]: to keep the model valid, the nominal road allows up to ${fmt(Math.sqrt(0.5 * mu * G / 0.012), 1)} m/s, ` +
    `the edge of the operating domain ${fmt(Math.sqrt(0.5 * mu * G / 0.022), 1)} m/s and the sharp bend only ${fmt(Math.sqrt(0.5 * mu * G / 0.04), 1)} m/s.`);
}
$("#vSpeed").addEventListener("input", drawVal);
$("#vRoad").addEventListener("change", drawVal);

/* =====================================================================
   start-up
   ===================================================================== */
driveStatus(); drawDrive(); drawGeo(); simulateBend(); drawEuler(); drawLin(); drawVal();
onRedraw(() => { drawDrive(); drawGeo(); drawBend(); drawEuler(); drawLin(); drawVal(); });
})();
