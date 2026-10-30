/* Satellite Pass Planning from Scratch, chapter 02, Placing an Orbit in Space: the live demos.
   An orthographic 3D view of the inertial frame (a sphere with a latitude/longitude grid, drawn as at t = 0,
   no coastlines), in which a circular orbit is built in three steps: the circle in its own plane, tilted by
   the inclination about x, turned by the RAAN about z, exactly the product R_z(RAAN) R_x(i) (a cos u, a sin u, 0)
   of satplan's CircularOrbit.position_eci (checked against Sat.positionECI). A second view flies the
   constellations of satplan.scenario.make_constellation (exported by code/ch02_numbers.py to ch02-data.js).
   Drag a view to rotate it. Needs ../kit/kit.js, sat.js and ch02-data.js. */
(() => {
"use strict";
const { $, css, fmt, caption, tex, fit, frame, alpha, onRedraw, narrowOf, clamp } = window.Kit;
const Sat = window.Sat;
const { R_EARTH, RAD, DEG } = Sat;
const D = window.SAT_CH02;
const ALT = 550e3, A = R_EARTH + ALT;

/* ---------------- rotations ---------------- */
const Rx = t => { const c = Math.cos(t), s = Math.sin(t); return [[1, 0, 0], [0, c, -s], [0, s, c]]; };
const Rz = t => { const c = Math.cos(t), s = Math.sin(t); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; };
const I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const mv = (M, v) => [0, 1, 2].map(r => M[r][0] * v[0] + M[r][1] * v[1] + M[r][2] * v[2]);
const mm = (P, Q) => [0, 1, 2].map(r => [0, 1, 2].map(c => P[r][0] * Q[0][c] + P[r][1] * Q[1][c] + P[r][2] * Q[2][c]));
/* Normal of the orbit plane with inclination i and RAAN (degrees): R_z(RAAN) R_x(i) applied to z. */
const normal = (incDeg, raanDeg) => { const i = incDeg * RAD, o = raanDeg * RAD; return [Math.sin(o) * Math.sin(i), -Math.cos(o) * Math.sin(i), Math.cos(i)]; };
/* Angle between two orbit planes taken as sets of points, 0..90 degrees. */
const planeAngle = (n1, n2) => Math.acos(Math.min(1, Math.abs(Sat.dot(n1, n2)))) * DEG;

/* ---------------- orthographic projection ---------------- */
/* The viewer looks at the origin from the direction d (azimuth yaw from x towards y, elevation pitch above the
   equatorial plane). Screen right is r, screen up is u. */
function basis(v) {
  const cy = Math.cos(v.yaw), sy = Math.sin(v.yaw), cp = Math.cos(v.pitch), sp = Math.sin(v.pitch);
  return { d: [cp * cy, cp * sy, sp], r: [-sy, cy, 0], u: [-sp * cy, -sp * sy, cp] };
}
function projector(canvas, view, extent, aspect) {
  const { ctx, w, h } = fit(canvas, aspect);
  const B = basis(view), S = (Math.min(w, h) / 2 - 10) / extent, cx = w / 2, cy = h / 2, rE = R_EARTH * S;
  const P = p => ({ x: cx + S * Sat.dot(p, B.r), y: cy - S * Sat.dot(p, B.u), z: Sat.dot(p, B.d) });
  /* a point above the surface is hidden when it is behind the Earth's disc */
  const hidden = q => q.z < 0 && (q.x - cx) ** 2 + (q.y - cy) ** 2 < rE * rE;
  return { ctx, w, h, B, S, cx, cy, rE, P, hidden };
}
/* A 3D polyline. Off the surface: hidden parts dashed and faint. On the surface (surface = true): only the
   hemisphere facing the viewer is drawn. */
function path3(pr, pts, { color, width = 1.5, dash = null, surface = false, faint = 0.28 } = {}) {
  const { ctx } = pr, q = pts.map(pr.P);
  const vis = q.map(p => surface ? p.z >= 0 : !pr.hidden(p));
  let start = 0;
  for (let k = 1; k <= q.length; k++) {
    if (k < q.length && vis[k] === vis[start]) continue;
    const run = q.slice(start, Math.min(k + 1, q.length));
    if (run.length > 1 && (vis[start] || !surface)) {
      ctx.beginPath(); run.forEach((p, j) => j ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round";
      ctx.setLineDash(vis[start] ? (dash || []) : [3, 3]); ctx.globalAlpha = vis[start] ? 1 : faint;
      ctx.stroke(); ctx.globalAlpha = 1; ctx.setLineDash([]);
    }
    start = k;
  }
}
function dot3(pr, p, radius, color, { ring = false } = {}) {
  const q = pr.P(p), { ctx } = pr;
  ctx.globalAlpha = pr.hidden(q) ? 0.3 : 1;
  ctx.beginPath(); ctx.arc(q.x, q.y, radius, 0, 2 * Math.PI);
  if (ring) { ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.fillStyle = css("--surface"); ctx.fill(); ctx.stroke(); }
  else { ctx.fillStyle = color; ctx.fill(); }
  ctx.globalAlpha = 1;
  return q;
}
function label3(pr, p, text, color, dx = 6, dy = -6) {
  const q = pr.P(p), { ctx } = pr;
  ctx.globalAlpha = pr.hidden(q) ? 0.35 : 1;
  ctx.font = "11px " + css("--font-mono"); ctx.fillStyle = color; ctx.textAlign = dx < 0 ? "right" : "left";
  ctx.fillText(text, q.x + dx, q.y + dy); ctx.globalAlpha = 1;
}
/* The Earth: a disc with the latitude / longitude grid on the visible hemisphere, as at t = 0 (Greenwich on x). */
function drawEarth(pr) {
  const { ctx, cx, cy, rE } = pr;
  ctx.beginPath(); ctx.arc(cx, cy, rE, 0, 2 * Math.PI);
  ctx.fillStyle = alpha("--accent", 0.1); ctx.fill(); ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1; ctx.stroke();
  const ll = (lat, lon) => [R_EARTH * Math.cos(lat * RAD) * Math.cos(lon * RAD), R_EARTH * Math.cos(lat * RAD) * Math.sin(lon * RAD), R_EARTH * Math.sin(lat * RAD)];
  for (let lat = -60; lat <= 60; lat += 30) {
    const pts = []; for (let lon = 0; lon <= 360; lon += 4) pts.push(ll(lat, lon));
    path3(pr, pts, { color: alpha("--muted", lat === 0 ? 0.95 : 0.45), width: lat === 0 ? 1.3 : 0.8, surface: true });
  }
  for (let lon = 0; lon < 360; lon += 30) {
    const pts = []; for (let lat = -90; lat <= 90; lat += 4) pts.push(ll(lat, lon));
    path3(pr, pts, { color: alpha("--muted", lon === 0 ? 0.9 : 0.45), width: lon === 0 ? 1.2 : 0.8, surface: true });
  }
}
function drawAxes(pr, len) {
  [[[1, 0, 0], "x"], [[0, 1, 0], "y"], [[0, 0, 1], "z (north)"]].forEach(([e, name]) => {
    path3(pr, [Sat.scale(e, R_EARTH), Sat.scale(e, len)], { color: css("--ink-2"), width: 1.2 });
    dot3(pr, Sat.scale(e, len), 2.5, css("--ink-2"));
    label3(pr, Sat.scale(e, len), name, css("--ink-2"));
  });
}
function circlePts(M, a, n = 180) {
  const pts = [];
  for (let j = 0; j <= n; j++) { const t = 2 * Math.PI * j / n; pts.push(mv(M, [a * Math.cos(t), a * Math.sin(t), 0])); }
  return pts;
}
function arcPts(M, a, t0, t1, n = 40) {
  const pts = [];
  for (let j = 0; j <= n; j++) { const t = t0 + (t1 - t0) * j / n; pts.push(mv(M, [a * Math.cos(t), a * Math.sin(t), 0])); }
  return pts;
}
/* Drag (mouse or touch) and arrow keys rotate a view. */
function draggable(canvas, view, redraw) {
  let last = null;
  canvas.addEventListener("pointerdown", e => { last = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = "grabbing"; });
  canvas.addEventListener("pointermove", e => {
    if (!last) return;
    const dx = e.clientX - last[0], dy = e.clientY - last[1];
    last = [e.clientX, e.clientY];
    view.yaw -= dx * 0.01; view.pitch = clamp(view.pitch + dy * 0.01, -89.9 * RAD, 89.9 * RAD);
    redraw();
  });
  const end = () => { last = null; canvas.style.cursor = "grab"; };
  canvas.addEventListener("pointerup", end); canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("keydown", e => {
    const k = { ArrowLeft: [0.08, 0], ArrowRight: [-0.08, 0], ArrowUp: [0, -0.08], ArrowDown: [0, 0.08] }[e.key];
    if (!k) return;
    e.preventDefault(); view.yaw += k[0]; view.pitch = clamp(view.pitch + k[1], -89.9 * RAD, 89.9 * RAD); redraw();
  });
}
const km = v => "(" + v.map(x => (x / 1e3).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 }).replace("-", "−")).join(", ") + ") km";
const texNum = (x, d = 4) => { const s = fmt(Math.abs(x) < 5e-5 ? 0 : x, d); return s.startsWith("-") ? "-" + s.slice(1) : s; };
const texMat = M => "\\begin{pmatrix}" + M.map(r => r.map(x => texNum(x)).join(" & ")).join(" \\\\ ") + "\\end{pmatrix}";

/* ================= hero: building position_eci in three rotations ================= */
const hs = { i: 97.6, O: 40, u: 60, stage: 3, wrong: false, view: { yaw: 35 * RAD, pitch: 22 * RAD } };
function drawHero() {
  const cv = $("#heroC"), narrow = narrowOf(cv, 560);
  const pr = projector(cv, hs.view, 1.5 * A, narrow ? 0.95 : 1.25);
  drawEarth(pr);
  drawAxes(pr, 1.42 * A);
  const i = hs.i * RAD, O = hs.O * RAD, u = hs.u * RAD;
  const MX = Rx(i), MZX = mm(Rz(O), MX), MXZ = mm(MX, Rz(O));
  const stages = [I3, MX, MZX], M = stages[hs.stage - 1];
  for (let s = 0; s < hs.stage - 1; s++) path3(pr, circlePts(stages[s], A), { color: alpha("--muted", 0.8), width: 1, dash: [2, 3] });
  const pPlane = [A * Math.cos(u), A * Math.sin(u), 0];
  if (hs.wrong) {
    path3(pr, circlePts(MXZ, A), { color: css("--warm"), width: 1.5, dash: [6, 4] });
    dot3(pr, mv(MXZ, pPlane), 5, css("--warm"), { ring: true });
    label3(pr, mv(MXZ, pPlane), "wrong order", css("--warm"));
  }
  path3(pr, circlePts(M, A), { color: css("--post"), width: 2.2 });
  // line of nodes, outside the Earth, and the two nodes
  const nodeDir = mv(M, [1, 0, 0]);
  path3(pr, [Sat.scale(nodeDir, R_EARTH), Sat.scale(nodeDir, 1.28 * A)], { color: css("--ink"), width: 1, dash: [1, 3] });
  path3(pr, [Sat.scale(nodeDir, -R_EARTH), Sat.scale(nodeDir, -1.28 * A)], { color: css("--ink"), width: 1, dash: [1, 3] });
  dot3(pr, Sat.scale(nodeDir, A), 4.5, css("--ink"));
  dot3(pr, Sat.scale(nodeDir, -A), 4.5, css("--ink"), { ring: true });
  label3(pr, Sat.scale(nodeDir, 1.28 * A), hs.stage === 1 ? "u = 0" : "ascending node", css("--ink"));
  // angle arcs: RAAN in the equatorial plane, u in the orbit plane
  if (hs.stage === 3 && hs.O > 0.5) {
    path3(pr, arcPts(I3, 1.16 * A, 0, O), { color: css("--c4"), width: 2 });
    label3(pr, mv(I3, [1.16 * A * Math.cos(O / 2), 1.16 * A * Math.sin(O / 2), 0]), "Ω", css("--c4"), 6, 12);
  }
  if (hs.u > 0.5) {
    path3(pr, arcPts(M, 1.1 * A, 0, u), { color: css("--warm"), width: 2 });
    label3(pr, mv(M, [1.1 * A * Math.cos(u / 2), 1.1 * A * Math.sin(u / 2), 0]), "u", css("--warm"), 6, -6);
  }
  // the satellite, its radius vector and its direction of motion
  const sat = mv(M, pPlane), vel = mv(M, [-Math.sin(u), Math.cos(u), 0]);
  path3(pr, [Sat.scale(Sat.unit(sat), R_EARTH), sat], { color: css("--warm"), width: 1.2 });
  path3(pr, [sat, [sat[0] + 0.3 * A * vel[0], sat[1] + 0.3 * A * vel[1], sat[2] + 0.3 * A * vel[2]]], { color: css("--warm"), width: 2 });
  dot3(pr, sat, 6, css("--warm"));
  label3(pr, sat, "satellite", css("--warm"), 9, 4);

  // readouts
  const pTilt = mv(MX, pPlane), pEci = mv(MZX, pPlane), pWrong = mv(MXZ, pPlane);
  const pkg = Sat.positionECI(Sat.orbit(ALT, hs.i, hs.O, hs.u), 0);
  const diff = Math.max(...pkg.map((x, k) => Math.abs(x - pEci[k])));
  $("#hiV").textContent = fmt(hs.i, 1) + "°"; $("#hOV").textContent = fmt(hs.O, 0) + "°"; $("#huV").textContent = fmt(hs.u, 0) + "°";
  $("#rPlane").textContent = km(pPlane); $("#rTilt").textContent = km(pTilt); $("#rECI").textContent = km(pEci);
  $("#rPkg").textContent = km(pkg); $("#rDiff").textContent = diff < 1e-6 ? "< 1 µm" : diff.toExponential(1) + " m";
  $("#rLat").textContent = fmt(Math.asin(pEci[2] / A) * DEG, 1) + "°";
  $("#rWrong").textContent = hs.wrong ? km(pWrong) : "–";
  const dir = hs.i < 89.95 ? "prograde" : hs.i > 90.05 ? "retrograde" : "polar";
  $("#rDir").textContent = dir;
  ["#st1", "#st2", "#st3"].forEach((id, k) => $(id).setAttribute("aria-pressed", String(hs.stage === k + 1)));
  $("#wrong").setAttribute("aria-pressed", String(hs.wrong));
  tex($("#matOut"), (hs.wrong ? "R_x(i)\\,R_z(\\Omega) = " + texMat(MXZ) + "\\qquad " : "") + "R_z(\\Omega)\\,R_x(i) = " + texMat(MZX), true);
  const stageTxt = [
    `Step 1: the circle in its own plane. Before any rotation the orbit lies in the equatorial plane and the satellite is at [[(a\\cos u, a\\sin u, 0)]] = ${km(pPlane)}, at angle [[u = ${fmt(hs.u, 0)}°]] from the x axis.`,
    `Step 2: tilt by the inclination. [[R_x(i)]] turns the circle about the x axis by [[i = ${fmt(hs.i, 1)}°]]; the x axis stays fixed, so it is now the line where the orbit crosses the equator, and the satellite is at ${km(pTilt)}. The faint dashed circle is step 1.`,
    `Step 3: turn by the RAAN. [[R_z(\\Omega)]] turns the tilted circle about the polar axis by [[\\Omega = ${fmt(hs.O, 0)}°]], carrying the crossing line, the line of nodes, with it. The satellite is at ${km(pEci)}, latitude ${fmt(Math.asin(pEci[2] / A) * DEG, 1)}°; <code>position_eci</code> gives the same point to ${diff < 1e-6 ? "within a micrometre" : diff.toExponential(1) + " m"}.`,
  ][hs.stage - 1];
  const incTxt = hs.i < 89.95 ? `With [[i < 90°]] the satellite moves eastward, the way the Earth turns: a prograde orbit.`
    : hs.i > 90.05 ? `With [[i = ${fmt(hs.i, 1)}° > 90°]] the satellite crosses the equator heading slightly westward, against the Earth's rotation: a retrograde orbit, ${fmt(hs.i - 90, 1)}° past polar, reaching latitude ${fmt(180 - hs.i, 1)}° at most.`
    : `With [[i = 90°]] the orbit passes over both poles.`;
  const wrongTxt = hs.wrong ? ` Orange: the same rotations in the wrong order, [[R_x(i) R_z(\\Omega)]]: the satellite lands at ${km(pWrong)}, ${Math.round(Sat.norm(Sat.sub(pWrong, pEci)) / 1e3).toLocaleString("en-US")} km from the right place. Turning the flat circle about z first only slides the satellite along it, by [[\\Omega]]; the tilt then happens about x, so the wrong orbit's ascending node stays on the x axis.` : "";
  caption($("#heroStatus"), `${stageTxt} ${incTxt}${wrongTxt}`);
}
function bindRange(id, key) { $(id).addEventListener("input", e => { hs[key] = +e.target.value; drawHero(); }); }
bindRange("#hi", "i"); bindRange("#hO", "O"); bindRange("#hu", "u");
[1, 2, 3].forEach(k => $("#st" + k).addEventListener("click", () => { hs.stage = k; drawHero(); }));
$("#wrong").addEventListener("click", () => { hs.wrong = !hs.wrong; drawHero(); });
$("#vObl").addEventListener("click", () => { hs.view.yaw = 35 * RAD; hs.view.pitch = 22 * RAD; drawHero(); });
$("#vPole").addEventListener("click", () => { hs.view.yaw = 0; hs.view.pitch = 89.9 * RAD; drawHero(); });
$("#vNode").addEventListener("click", () => { hs.view.yaw = hs.O * RAD; hs.view.pitch = 0; drawHero(); });
$("#hReset").addEventListener("click", () => {
  Object.assign(hs, { i: 97.6, O: 40, u: 60, stage: 3, wrong: false });
  hs.view.yaw = 35 * RAD; hs.view.pitch = 22 * RAD;
  $("#hi").value = 97.6; $("#hO").value = 40; $("#hu").value = 60; drawHero();
});
draggable($("#heroC"), hs.view, drawHero);

/* ================= the constellation over time ================= */
const PLANE_TOKENS = ["--post", "--warm", "--c4", "--c5"];
const cs = { size: "small", t: 0, spread360: false, playing: false, view: { yaw: 35 * RAD, pitch: 25 * RAD } };
const raanOf = (S, k) => cs.spread360 ? 2 * S.raanNom[k] + (S.orbits[k][2] - S.raanNom[k]) : S.orbits[k][2];
function constellation() {
  const S = D.sizes[cs.size];
  return { S, orbits: S.orbits.map((o, k) => Sat.orbit(o[0], o[1], raanOf(S, k), o[3])) };
}
function minPlaneSep(S, orbits) {
  let best = 180;
  for (let a = 0; a < orbits.length; a++) for (let b = 0; b < orbits.length; b++) {
    if (S.plane[a] >= S.plane[b]) continue;
    best = Math.min(best, planeAngle(normal(orbits[a].inclination_deg, orbits[a].raan_deg), normal(orbits[b].inclination_deg, orbits[b].raan_deg)));
  }
  return best;
}
function drawCon() {
  const { S, orbits } = constellation();
  const cv = $("#conC"), narrow = narrowOf(cv, 560);
  const pr = projector(cv, cs.view, 1.4 * A, narrow ? 0.95 : 1.3);
  drawEarth(pr);
  drawAxes(pr, 1.35 * A);
  const T = orbits[0].period;
  orbits.forEach((o, k) => {
    const pts = []; for (let j = 0; j <= 120; j++) pts.push(Sat.positionECI(o, T * j / 120));
    path3(pr, pts, { color: alpha(PLANE_TOKENS[S.plane[k]], 0.75), width: 1.3 });
  });
  orbits.forEach((o, k) => {
    const p = Sat.positionECI(o, cs.t);
    dot3(pr, p, 5, css(PLANE_TOKENS[S.plane[k]]));
    label3(pr, p, String(k), css("--ink"), 7, -5);
  });
  // the (RAAN, u) diagram
  const mc = $("#conMap"), mnarrow = narrowOf(mc);
  const fr = frame(mc, mnarrow ? 1.25 : 2.2, { x: [-30, 330], y: [0, 360], xticks: [0, 60, 120, 180, 240, 300], yticks: [0, 90, 180, 270, 360],
    xlabel: "right ascension of the ascending node Ω [°]", ylabel: "argument of latitude u at time t [°]" });
  const { ctx } = fr;
  ctx.strokeStyle = alpha("--muted", 0.5); ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
  for (let p = 0; p < S.nPlanes; p++) { const x = fr.X((cs.spread360 ? 360 : 180) * p / S.nPlanes); ctx.beginPath(); ctx.moveTo(x, fr.Y(0)); ctx.lineTo(x, fr.Y(360)); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.font = "11px " + css("--font-mono");
  orbits.forEach((o, k) => {
    const uu = ((o.phase_deg + 360 * cs.t / o.period) % 360 + 360) % 360, rr = ((o.raan_deg + 30) % 360 + 360) % 360 - 30;
    ctx.fillStyle = css(PLANE_TOKENS[S.plane[k]]);
    ctx.beginPath(); ctx.arc(fr.X(rr), fr.Y(uu), 5, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = css("--ink"); ctx.textAlign = "left"; ctx.fillText(String(k), fr.X(rr) + 7, fr.Y(uu) + 4);
  });
  const per = [...Array(S.nPlanes).keys()].map(p => S.plane.filter(q => q === p).length);
  const sep = minPlaneSep(S, orbits);
  $("#cTV").textContent = `${fmt(cs.t / 60, 0)} min (${fmt(cs.t / T, 2)} rev.)`;
  $("#cPlanes").textContent = `round(√${S.nSats}) = ${S.nPlanes}`;
  $("#cPer").textContent = per.join(" + ");
  $("#cSep").textContent = fmt(sep, 1) + "°";
  $("#cStep").textContent = `${fmt((cs.spread360 ? 360 : 180) / S.nPlanes, 0)}° / ${fmt(360 / S.perPlane, 0)}° / ${fmt(360 / (S.nPlanes * S.perPlane), 0)}°`;
  ["small", "medium", "large"].forEach(n => $("#c-" + n).setAttribute("aria-pressed", String(cs.size === n)));
  $("#c360").setAttribute("aria-pressed", String(cs.spread360));
  $("#cPlay").textContent = cs.playing ? "pause" : "play";
  const sizeTxt = { small: "the small instance (4 satellites, 6 h)", medium: "the medium instance (8 satellites, 12 h)", large: "the large instance (12 satellites, 12 h)" }[cs.size];
  let txt = `The constellation of ${sizeTxt}, from <code>make_constellation</code> with seed 0, at [[t = ${fmt(cs.t / 60, 0)}]] min. It has [[\\mathrm{round}(\\sqrt{${S.nSats}}) = ${S.nPlanes}]] planes (colours) with ${per.join(", ")} satellites; each plane's ascending node is [[${fmt((cs.spread360 ? 360 : 180) / S.nPlanes, 0)}°]] from the next, plus up to ±5° of jitter, so even satellites of one plane fly in slightly different planes. Within a plane the satellites are [[360°/${S.perPlane} = ${fmt(360 / S.perPlane, 0)}°]] apart, and each plane is shifted by [[${fmt(360 / (S.nPlanes * S.perPlane), 0)}°]] from the previous one (±3° jitter): in the diagram below, the satellites form a staircase that climbs as they move. `;
  if (cs.spread360) txt += S.nPlanes === 2
    ? `<b>Spread over 360°</b>, the two planes have ascending nodes about 180° apart. For a near-polar orbit that is almost the same plane flown the other way: the smallest angle between the planes is ${fmt(sep, 1)}°, so the two planes nearly coincide and half of the sky is left empty.`
    : `<b>Spread over 360°</b>, the planes are still ${fmt(sep, 1)}° apart or more as sets of points, but neighbouring planes now fly in opposite directions: a plane with its node at [[\\Omega + 180°]] is nearly the plane at [[\\Omega]] flown backwards.`;
  else txt += `The smallest angle between two planes of different groups is ${fmt(sep, 1)}°. Near-polar planes cross each other at high latitudes, near the poles.`;
  caption($("#conCap"), txt);
}
function setSize(n) {
  cs.size = n; cs.t = Math.min(cs.t, D.sizes[n].horizon);
  $("#cT").max = D.sizes[n].horizon; $("#cT").value = cs.t; drawCon();
}
["small", "medium", "large"].forEach(n => $("#c-" + n).addEventListener("click", () => setSize(n)));
$("#c360").addEventListener("click", () => { cs.spread360 = !cs.spread360; drawCon(); });
$("#cT").addEventListener("input", e => { cs.t = +e.target.value; drawCon(); });
let lastFrame = null;
function tick(now) {
  if (!cs.playing) { lastFrame = null; return; }
  if (lastFrame != null) {
    const dt = Math.min(0.1, (now - lastFrame) / 1000);   // seconds of wall time
    cs.t += dt * 1800;                                        // half an hour of orbit per second
    if (cs.t > D.sizes[cs.size].horizon) cs.t = 0;
    $("#cT").value = cs.t;
    drawCon();
  }
  lastFrame = now;
  requestAnimationFrame(tick);
}
$("#cPlay").addEventListener("click", () => { cs.playing = !cs.playing; drawCon(); if (cs.playing) requestAnimationFrame(tick); });
draggable($("#conC"), cs.view, drawCon);
$("#cT").max = D.sizes.small.horizon;

function drawAll() { drawHero(); drawCon(); }
drawAll(); onRedraw(drawAll);
})();
