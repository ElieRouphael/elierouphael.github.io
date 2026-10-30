/* Satellite Pass Planning from Scratch, chapter 03, The Earth Turns Underneath: the live demos.
   Ground tracks on an equirectangular map (hero), the same orbit seen in the inertial and in the Earth-fixed
   frame, passes per day against latitude, and the passes over each maritime area of the course's small
   instance. Every position comes from the shared port of satplan (interactive/sat.js); the instance from
   interactive/instance-small.js. Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, css, fmt, caption, fit, frame, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const S = window.Sat, I = window.SAT_SMALL;
const DAY = 86400, YEAR = 365.2422, J2 = 1.08263e-3, R_EQ = 6378137.0;
const SUN = 360 / YEAR;                         // deg/day, the Sun's apparent motion along the ecliptic
const SAT_COLORS = ["--post", "--warm", "--c4", "--c5"];

/* ---------------- shared helpers ---------------- */
const maxLatOf = inc => inc <= 90 ? inc : 180 - inc;
/* J2 nodal regression of a circular orbit, rad/s: -(3/2) J2 (R_eq / a)^2 n cos i. */
function nodalRate(o) { const n = 2 * Math.PI / o.period; return -1.5 * J2 * (R_EQ / o.a) ** 2 * n * Math.cos(o.inclination_deg * S.RAD); }
/* Sub-satellite points from t = 0 to t1 every `step` seconds. */
function track(o, t1, step) {
  const pts = [];
  for (let t = 0; t <= t1 + 1e-9; t += step) { const p = S.subPoint(S.positionECI(o, t), t); p.t = t; pts.push(p); }
  return pts;
}
/* Pieces of a track that do not jump across the map (date line, or a pass close to a pole). */
function pieces(pts) {
  const out = []; let cur = [];
  for (let k = 0; k < pts.length; k++) {
    if (k && Math.abs(pts[k].lon - pts[k - 1].lon) > 90) { out.push(cur); cur = []; }
    cur.push(pts[k]);
  }
  if (cur.length) out.push(cur);
  return out;
}
/* Ascending equator crossings of a sampled track: longitude interpolated between the two samples. */
function ascendingNodes(pts) {
  const out = [];
  for (let k = 0; k + 1 < pts.length; k++) {
    const a = pts[k], b = pts[k + 1];
    if (a.lat < 0 && b.lat >= 0 && Math.abs(b.lon - a.lon) < 90) {
      const f = -a.lat / (b.lat - a.lat);
      out.push({ lon: a.lon + f * (b.lon - a.lon), t: a.t + f * (b.t - a.t) });
    }
  }
  return out;
}

/* ---------------- the map ---------------- */
function mapFrame(cv) {
  const narrow = narrowOf(cv, 560);
  const { ctx, w, h } = fit(cv, narrow ? 1.55 : 2);
  const L = 30, R = 6, T = 6, B = 18, mw = w - L - R, mh = h - T - B;
  const P = S.mapProjection(L, T, mw, mh);
  ctx.fillStyle = css("--surface-2"); ctx.fillRect(L, T, mw, mh);
  ctx.lineWidth = 1; ctx.strokeStyle = css("--grid");
  for (let lon = -180; lon <= 180; lon += 30) seg(ctx, Math.round(P.X(lon)) + 0.5, T, Math.round(P.X(lon)) + 0.5, T + mh);
  for (let lat = -90; lat <= 90; lat += 30) seg(ctx, L, Math.round(P.Y(lat)) + 0.5, L + mw, Math.round(P.Y(lat)) + 0.5);
  ctx.strokeStyle = alpha("--muted", 0.7); seg(ctx, L, Math.round(P.Y(0)) + 0.5, L + mw, Math.round(P.Y(0)) + 0.5);
  ctx.font = "10px " + css("--font-mono"); ctx.fillStyle = css("--muted");
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const lat of [60, 30, 0, -30, -60]) ctx.fillText(lat + "°", L - 4, P.Y(lat));
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const lon of (narrow ? [-120, 0, 120] : [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])) ctx.fillText(lon + "°", P.X(lon), T + mh + 4);
  ctx.textBaseline = "alphabetic";
  // the ten maritime areas and the four ground stations of the scenario
  ctx.fillStyle = alpha("--accent", 0.22); ctx.strokeStyle = css("--accent");
  for (const [la0, la1, lo0, lo1] of I.areaBoxes) {
    const x0 = P.X(lo0), x1 = P.X(lo1), y0 = P.Y(la1), y1 = P.Y(la0);
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0); ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0, y1 - y0);
  }
  ctx.fillStyle = css("--ink");
  for (const [, lat, lon] of I.stations) {
    const x = P.X(lon), y = P.Y(lat);
    ctx.beginPath(); ctx.moveTo(x, y - 5); ctx.lineTo(x + 4.5, y + 3.5); ctx.lineTo(x - 4.5, y + 3.5); ctx.closePath(); ctx.fill();
  }
  return { ctx, w, h, P, L, T, mw, mh, narrow };
}
function drawTrack(m, pts, color, width) {
  const { ctx, P, L, T, mw, mh } = m;
  ctx.save(); ctx.beginPath(); ctx.rect(L, T, mw, mh); ctx.clip();
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round";
  for (const pc of pieces(pts)) {
    ctx.beginPath();
    pc.forEach((p, i) => { const x = P.X(p.lon), y = P.Y(p.lat); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }
  ctx.restore();
}

/* ---------------- hero: ground tracks ---------------- */
const hs = { inc: 97.6, alt: 550, hours: 6, four: true };
function heroOrbits() {
  const base = hs.four ? I.orbits : [I.orbits[0]];
  return base.map(o => S.orbit(hs.alt * 1e3, hs.inc, o[2], o[3]));
}
function drawHero() {
  const orbits = heroOrbits(), t1 = hs.hours * 3600, step = 30;
  const m = mapFrame($("#heroMap")), { ctx, P } = m;
  const ml = maxLatOf(hs.inc);
  ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.2;
  for (const s of [1, -1]) seg(ctx, m.L, P.Y(s * ml), m.L + m.mw, P.Y(s * ml));
  ctx.restore();
  const tracks = orbits.map(o => track(o, t1, step));
  tracks.forEach((pts, k) => drawTrack(m, pts, css(SAT_COLORS[k]), 1.6));
  // ascending nodes of the first satellite, numbered
  const nodes = ascendingNodes(tracks[0]);
  ctx.font = "10px " + css("--font-mono"); ctx.textAlign = "center";
  nodes.forEach((nd, k) => {
    const x = P.X(nd.lon), y = P.Y(0);
    ctx.fillStyle = css("--surface"); ctx.strokeStyle = css(SAT_COLORS[0]); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    if (!m.narrow || k < 4) { ctx.fillStyle = css("--ink"); ctx.fillText(String(k + 1), x, y + 15); }
  });
  // where each satellite is at the end of the horizon
  tracks.forEach((pts, k) => {
    const p = pts[pts.length - 1];
    ctx.fillStyle = css(SAT_COLORS[k]); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(P.X(p.lon), P.Y(p.lat), 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  });
  // readouts
  const o = orbits[0], T = o.period, shift = S.OMEGA_EARTH * T * S.DEG;
  const rate = nodalRate(o), drift = rate * t1;               // rad over the horizon
  $("#rT").textContent = fmt(T / 60, 2) + " min";
  $("#rRev").textContent = fmt(t1 / T, 2);
  $("#rShift").textContent = `${fmt(shift, 2)}° = ${fmt(shift * S.RAD * S.R_EARTH / 1e3, 0)} km`;
  $("#rMax").textContent = fmt(ml, 1) + "°";
  $("#rJ2").textContent = `${fmt(rate * S.DEG * DAY, 4)}°/day`;
  $("#rJ2d").textContent = `${fmt(drift * S.DEG, 3)}° = ${fmt(Math.abs(drift) * S.R_EARTH / 1e3, 1)} km`;
  const dirWord = hs.inc > 90 ? "retrograde: the satellite moves slightly westward as it climbs, against the Earth's rotation" : hs.inc < 90 ? "prograde: the satellite moves eastward as it climbs, with the Earth's rotation" : "exactly polar";
  const isDefault = hs.inc === 97.6 && hs.alt === 550;
  const ssGap = rate * S.DEG * DAY - SUN;
  caption($("#heroStatus"), `${hs.four ? "The four satellites of the course's small instance" : "Satellite 0 of the small instance"} at ${fmt(hs.alt, 0)} km and ${fmt(hs.inc, 1)}° (${dirWord}), over the first ${fmt(hs.hours, 1)} h: <b>${fmt(t1 / T, 2)} revolutions</b> of ${fmt(T / 60, 2)} min each. The numbered circles are the ascending equator crossings of satellite 0 (blue): each one is <b>${fmt(shift, 2)}° further west</b> than the previous, because the Earth turned that much during one revolution, [[\\omega_E T]]. No track goes beyond <b>${fmt(ml, 1)}°</b> of latitude (dashed), so the polar caps are seen only from the side, and the tracks crowd together at high latitude while they are ${fmt(shift * S.RAD * S.R_EARTH / 1e3, 0)} km apart at the equator. Shaded boxes: the ten maritime areas of the scenario; triangles: the four ground stations. ${isDefault ? "J2, which the package leaves out, would turn each orbit plane by " + fmt(rate * S.DEG * DAY, 4) + "°/day, almost exactly the Sun's " + fmt(SUN, 4) + "°/day: this is what makes 97.6° sun-synchronous at 550 km." : "J2, which the package leaves out, would turn the orbit plane by " + fmt(rate * S.DEG * DAY, 3) + "°/day here, " + (Math.abs(ssGap) < 0.05 ? "close to the Sun's " + fmt(SUN, 4) + "°/day (nearly sun-synchronous)." : "against the Sun's " + fmt(SUN, 4) + "°/day: not sun-synchronous.")}`);
}
function heroControls() {
  const upd = () => {
    $("#hIncV").textContent = fmt(hs.inc, 1) + "°"; $("#hAltV").textContent = fmt(hs.alt, 0) + " km"; $("#hHV").textContent = fmt(hs.hours, 1) + " h";
    $("#hFour").setAttribute("aria-pressed", String(hs.four)); $("#hOne").setAttribute("aria-pressed", String(!hs.four));
    drawHero();
  };
  $("#hInc").addEventListener("input", e => { hs.inc = +e.target.value; upd(); });
  $("#hAlt").addEventListener("input", e => { hs.alt = +e.target.value; upd(); });
  $("#hH").addEventListener("input", e => { hs.hours = +e.target.value; upd(); });
  $("#hFour").addEventListener("click", () => { hs.four = true; upd(); });
  $("#hOne").addEventListener("click", () => { hs.four = false; upd(); });
  $("#hReset").addEventListener("click", () => { hs.inc = 97.6; hs.alt = 550; hs.hours = 6; hs.four = true; $("#hInc").value = 97.6; $("#hAlt").value = 550; $("#hH").value = 6; upd(); });
  upd();
}

/* ---------------- two frames: inertial and Earth-fixed, seen from above the North Pole ---------------- */
const fs = { t: 3, playing: false, last: 0 };
const FR_ORBIT = S.orbit(550e3, 97.6, I.orbits[0][2], I.orbits[0][3]);
const ARCTIC = I.stations[0];               // ["GS-Arctic", 78.2, 15.4]
function drawFrame(cv, earthFixed) {
  const { ctx, w, h } = fit(cv, 1);
  const o = FR_ORBIT, t1 = fs.t * 3600, th = S.OMEGA_EARTH * t1;
  const cx = w / 2, cy = h / 2, sc = (Math.min(w, h) / 2 - 16) / o.a, Rp = S.R_EARTH * sc;
  const toPx = (x, y) => [cx + x * sc, cy - y * sc];
  // the Earth: equator (edge), parallels 30 and 60 deg N
  ctx.fillStyle = css("--surface-2"); ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, Rp, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  ctx.strokeStyle = css("--grid");
  for (const lat of [30, 60]) { ctx.beginPath(); ctx.arc(cx, cy, Rp * Math.cos(lat * S.RAD), 0, 2 * Math.PI); ctx.stroke(); }
  // Greenwich meridian: at angle th in the inertial frame, at 0 in the Earth-fixed frame
  const g = earthFixed ? 0 : th;
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2;
  seg(ctx, cx, cy, cx + Rp * Math.cos(g), cy - Rp * Math.sin(g));
  ctx.font = "10px " + css("--font-mono"); ctx.fillStyle = css("--ink");
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("Greenwich", cx + 0.62 * Rp * Math.cos(g) - 10 * Math.sin(g), cy - 0.62 * Rp * Math.sin(g) - 10 * Math.cos(g));
  // the inertial x axis (a fixed direction among the stars): at 0 inertially, at -th in the Earth-fixed frame
  const xa = earthFixed ? -th : 0, r0 = o.a * sc + 6;
  ctx.strokeStyle = alpha("--muted", 0.9); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  seg(ctx, cx + Rp * Math.cos(xa), cy - Rp * Math.sin(xa), cx + r0 * Math.cos(xa), cy - r0 * Math.sin(xa));
  ctx.setLineDash([]); ctx.fillStyle = css("--muted");
  ctx.fillText("x", clamp(cx + (r0 + 7) * Math.cos(xa), 6, w - 6), clamp(cy - (r0 + 7) * Math.sin(xa), 6, h - 6));
  // the Arctic ground station
  const st = S.groundECI(ARCTIC[1], ARCTIC[2], earthFixed ? 0 : t1);
  const [sx, sy] = toPx(st[0], st[1]);
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.moveTo(sx, sy - 5); ctx.lineTo(sx + 4.5, sy + 3.5); ctx.lineTo(sx - 4.5, sy + 3.5); ctx.closePath(); ctx.fill();
  // the trail from t = 0, in runs of the same hemisphere (north solid, south faint)
  const strong = css("--post"), faint = alpha("--post", 0.28);
  let run = [], north = null;
  const flush = () => { if (run.length > 1) { ctx.strokeStyle = north ? strong : faint; ctx.lineWidth = north ? 1.6 : 1.2; ctx.beginPath(); run.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); } };
  let last = null;
  for (let t = 0; t <= t1 + 1e-9; t += 60) {
    const r = S.positionECI(o, t);
    let x = r[0], y = r[1];
    if (earthFixed) { const a = S.OMEGA_EARTH * t, c = Math.cos(a), s = Math.sin(a); [x, y] = [x * c + y * s, -x * s + y * c]; }
    const p = toPx(x, y), n = r[2] >= 0;
    if (north === null) north = n;
    if (n !== north) { run.push(p); flush(); run = [p]; north = n; } else run.push(p);
    last = { p, r };
  }
  flush();
  // the satellite now
  const r = S.positionECI(o, t1);
  let x = r[0], y = r[1];
  if (earthFixed) { const c = Math.cos(th), s = Math.sin(th); [x, y] = [x * c + y * s, -x * s + y * c]; }
  const [px, py] = toPx(x, y);
  ctx.fillStyle = css("--warm"); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(px, py, 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  ctx.textBaseline = "alphabetic";
  return last;
}
function drawFrames() {
  drawFrame($("#frI"), false); drawFrame($("#frE"), true);
  const t1 = fs.t * 3600, revs = t1 / FR_ORBIT.period, turned = S.OMEGA_EARTH * t1 * S.DEG;
  $("#fTV").textContent = fmt(fs.t, 2) + " h";
  caption($("#frCap"), `After <b>${fmt(fs.t, 2)} h</b> the satellite has made ${fmt(revs, 2)} revolutions and the Earth has turned <b>${fmt(turned, 1)}°</b> (counter-clockwise, seen from above the North Pole). <b>Left, inertial frame:</b> the orbit plane keeps its direction among the stars, so the trail retraces one thin ellipse (a circle inclined at 97.6° seen almost edge-on, half-width [[a\\,|\\cos i| = ${fmt(FR_ORBIT.a * Math.abs(Math.cos(97.6 * S.RAD)) / 1e3, 0)}]] km), while Greenwich and the Arctic station (triangle) are carried round by the Earth. <b>Right, Earth-fixed frame:</b> the same motion with the Earth held still; now the fixed star direction [[x]] turns clockwise, and each revolution is drawn ${fmt(S.OMEGA_EARTH * FR_ORBIT.period * S.DEG, 1)}° further clockwise, that is further west, than the last. Solid trail: northern hemisphere; faint: southern. The station sits near the centre, under almost every revolution.`);
}
function frameControls() {
  const btn = $("#fPlay");
  const setBtn = () => { btn.textContent = fs.playing ? "Pause" : "Play 24 h"; btn.setAttribute("aria-pressed", String(fs.playing)); };
  const tick = ts => {
    if (!fs.playing) return;
    const dt = fs.last ? Math.min(0.1, (ts - fs.last) / 1000) : 0; fs.last = ts;
    fs.t = Math.min(24, fs.t + 1.5 * dt);           // 1.5 h of simulated time per second
    $("#fT").value = fs.t; drawFrames();
    if (fs.t >= 24) { fs.playing = false; setBtn(); return; }
    requestAnimationFrame(tick);
  };
  btn.addEventListener("click", () => {
    fs.playing = !fs.playing; setBtn();
    if (fs.playing) { if (fs.t >= 24) fs.t = 0; fs.last = 0; requestAnimationFrame(tick); }
  });
  $("#fT").addEventListener("input", e => { fs.t = +e.target.value; fs.playing = false; setBtn(); drawFrames(); });
  setBtn();
}

/* ---------------- passes per day against latitude ---------------- */
const LATS = []; for (let la = 0; la < 90; la += 5) LATS.push(la);
const LONS = []; for (let lo = -180; lo < 180; lo += 30) LONS.push(lo);
const ls = { inc: 97.6, rows: null, cache: new Map() };
/* One satellite (550 km, RAAN 0, phase 0), 24 h on a 30 s grid, the targets' limits (55 deg cone, 10 deg elevation);
   the number of visibility windows of points at each latitude, over 12 longitudes (as code/ch03_numbers.py). */
function passesByLatitude(inc) {
  if (ls.cache.has(inc)) return ls.cache.get(inc);
  const o = S.orbit(550e3, inc, 0, 0), ts = [], rs = [];
  for (let t = 0; t <= DAY + 1e-9; t += 30) { ts.push(t); rs.push(S.positionECI(o, t)); }
  const rows = LATS.map(la => {
    const counts = LONS.map(lo => S.runs(ts.map((t, k) => S.visible(rs[k], S.groundECI(la, lo, t), 55, 10))).length);
    return { lat: la, mean: counts.reduce((a, b) => a + b, 0) / counts.length, min: Math.min(...counts), max: Math.max(...counts) };
  });
  ls.cache.set(inc, rows);
  return rows;
}
function drawLat() {
  const cv = $("#latC");
  if (!ls.rows) { fit(cv, narrowOf(cv) ? 1.5 : 2.6); return; }
  const rows = ls.rows, top = Math.max(4, Math.ceil(Math.max(...rows.map(r => r.max)) + 0.5));
  const fr = frame(cv, narrowOf(cv) ? 1.5 : 2.6, { x: [-2.5, 87.5], y: [0, top], xticks: [0, 15, 30, 45, 60, 75], xfmt: v => v + "°", xlabel: "latitude of the point", ylabel: "passes per day" });
  const { ctx, X, Y } = fr;
  ctx.fillStyle = alpha("--post", 0.75);
  for (const r of rows) ctx.fillRect(X(r.lat - 2), Y(r.mean), X(r.lat + 2) - X(r.lat - 2), Y(0) - Y(r.mean));
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2;
  for (const r of rows) { seg(ctx, X(r.lat), Y(r.min), X(r.lat), Y(r.max)); seg(ctx, X(r.lat) - 3, Y(r.min), X(r.lat) + 3, Y(r.min)); seg(ctx, X(r.lat) - 3, Y(r.max), X(r.lat) + 3, Y(r.max)); }
  const ml = maxLatOf(ls.inc);
  if (ml < 87.5) { ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.4; seg(ctx, X(ml), Y(0), X(ml), Y(top)); ctx.restore(); }
  const at = la => rows.find(r => r.lat === la);
  caption($("#latCap"), `Passes per day over a point, against the point's latitude, for one satellite at 550 km and <b>${fmt(ls.inc, 1)}°</b> (bars: mean over 12 longitudes; whiskers: fewest and most). A pass is a visibility window with the targets' limits (55° cone, 10° elevation, chapter 04), found on a 30 s grid over 24 h. At the equator a point is seen <b>${fmt(at(0).mean, 2)}</b> times a day on average, and some points not at all (${at(0).min} to ${at(0).max}); at 40°, ${fmt(at(40).mean, 2)}; at 80°, <b>${fmt(at(80).mean, 2)}</b>. The dashed line is the highest latitude of the track, ${fmt(ml, 1)}°; points beyond it are still seen from the side, as long as they are within the footprint. ${ls.inc > 60 && ls.inc < 120 ? "Every revolution passes near the poles, but at the equator the tracks are " + fmt(S.OMEGA_EARTH * S.orbit(550e3, ls.inc, 0, 0).period * S.R_EARTH / 1e3, 0) + " km apart and the footprint is about 1750 km wide: the gaps between neighbouring tracks are only filled over several days." : "With a low inclination the passes concentrate at mid latitudes, near the turning points of the track, and the poles are never seen."}`);
}
function latControls() {
  const el = $("#lInc");
  const compute = () => { $("#latCap").textContent = "Computing 216 points over 24 h…"; setTimeout(() => { ls.rows = passesByLatitude(ls.inc); drawLat(); }, 20); };
  el.addEventListener("input", e => { ls.inc = +e.target.value; $("#lIncV").textContent = fmt(ls.inc, 1) + "°"; });
  el.addEventListener("change", compute);
  $("#lIncV").textContent = fmt(ls.inc, 1) + "°";
  compute();
}

/* ---------------- passes over each maritime area of the small instance ---------------- */
const rv = { area: 9 };
/* Opportunities of the area's targets, grouped into passes: a window of the same satellite that starts less than
   20 min after the pass's current end joins it (as code/ch03_numbers.py). */
function areaPasses(k) {
  const ops = I.opps.filter(o => I.targets[o.target][2] === k).slice().sort((a, b) => a.ws - b.ws);
  const passes = [];
  for (const o of ops) {
    const p = passes.find(q => q.sat === o.sat && o.ws <= q.end + 1200);
    if (p) { p.end = Math.max(p.end, o.we); p.n++; } else passes.push({ sat: o.sat, start: o.ws, end: o.we, n: 1 });
  }
  passes.sort((a, b) => a.start - b.start);
  return { ops, passes };
}
function drawRevisit() {
  const cv = $("#revC"), narrow = narrowOf(cv);
  const { ops, passes } = areaPasses(rv.area), H = I.horizon / 3600;
  const fr = frame(cv, narrow ? 1.5 : 3.2, { x: [0, H], y: [-0.6, I.nSats - 0.4], yticks: [0, 1, 2, 3], yfmt: v => "sat " + v, ygrid: false, pad: { l: 46 }, xlabel: "time [h]" });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--grid");
  for (let s = 0; s < I.nSats; s++) seg(ctx, X(0), Y(s), X(H), Y(s));
  for (const o of ops) {
    const y0 = Y(o.sat + 0.25), y1 = Y(o.sat - 0.25);
    ctx.fillStyle = alpha(SAT_COLORS[o.sat], 0.55);
    ctx.fillRect(X(o.ws / 3600), y0, Math.max(1.5, X(o.we / 3600) - X(o.ws / 3600)), y1 - y0);
  }
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.2;
  for (const p of passes) { const y0 = Y(p.sat + 0.38), y1 = Y(p.sat - 0.38), x0 = X(p.start / 3600) - 3, x1 = X(p.end / 3600) + 3; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); }
  const gaps = passes.slice(1).map((p, q) => (p.start - passes[q].end) / 60);
  $("#rvOpp").textContent = String(ops.length);
  $("#rvPass").textContent = String(passes.length);
  $("#rvGaps").textContent = gaps.length ? gaps.map(g => g < 60 ? fmt(g, 0) + " min" : fmt(g / 60, 1) + " h").join(", ") : "none";
  const fmtH = s => fmt(s / 3600, 2) + " h";
  caption($("#revCap"), `<b>${I.areas[rv.area]}</b>: the ${ops.length} opportunities of its targets in the small instance (bars, one row per satellite; overlapping windows of neighbouring targets stack), grouped into <b>${passes.length} pass${passes.length > 1 ? "es" : ""}</b> (outlined): ${passes.map(p => `satellite ${p.sat} at ${fmtH(p.start)}`).join(", ")}. ${gaps.length ? `Gaps between passes: ${$("#rvGaps").textContent}.` : "One pass in six hours, and nothing else."} ${gaps.some(g => g < 60) ? "A gap of about 43 minutes is the time between two satellites of the same plane, half a revolution apart, crossing the area while the Earth has not yet carried it away from that plane." : ""} ${gaps.some(g => g >= 60) ? "A gap of several hours is the time the Earth needs to carry the area from one orbit plane to the other, 90° of RAAN away." : ""}`);
}
function revisitControls() {
  const sel = $("#rArea");
  I.areas.forEach((name, k) => { const op = document.createElement("option"); op.value = String(k); op.textContent = name; sel.appendChild(op); });
  sel.value = String(rv.area);
  sel.addEventListener("change", e => { rv.area = +e.target.value; drawRevisit(); });
}

heroControls(); frameControls(); drawFrames(); revisitControls(); drawRevisit(); latControls();
onRedraw(() => { drawHero(); drawFrames(); drawLat(); drawRevisit(); });
})();
