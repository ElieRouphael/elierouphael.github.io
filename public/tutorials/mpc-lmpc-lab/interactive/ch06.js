/* Advanced MPC and Learning MPC, chapter 06, Frenet coordinates and minimum-time racing: the live demos.
   The course's racing.py is reproduced here: the synthetic closed track sampled uniformly in arc length
   (with the exact derivatives of the closed-form curve), the Frenet rates, the BARC dynamic bicycle model
   with its simplified Pacejka tires and its spatial right-hand side, and the closed-lap minimum-time speed
   planner in the squared-speed variable (forward acceleration pass, backward braking pass, repeated to a
   fixed point). Needs ../kit/kit.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;

/* NET-BEGIN  (this block is also run under Node by the equivalence test) */
const G = 9.81;
const curve = t => [45 * Math.cos(t) + 5 * Math.cos(2 * t), 30 * Math.sin(t) + 6 * Math.sin(3 * t)];
const curveD1 = t => [-45 * Math.sin(t) - 10 * Math.sin(2 * t), 30 * Math.cos(t) + 18 * Math.cos(3 * t)];
const curveD2 = t => [-45 * Math.cos(t) - 20 * Math.cos(2 * t), -30 * Math.sin(t) - 54 * Math.sin(3 * t)];
/* arc length on a dense parameter grid, integrated once with the trapezoidal rule (dense = 200000 as in the course) */
const DENSE = (() => { const n = 200000, t = new Float64Array(n + 1), s = new Float64Array(n + 1); let prev = 0; for (let i = 0; i <= n; i++) { t[i] = 2 * Math.PI * i / n; const d = curveD1(t[i]), sp = Math.hypot(d[0], d[1]); if (i) s[i] = s[i - 1] + 0.5 * (sp + prev) * (t[i] - t[i - 1]); prev = sp; } return { t, s, length: s[n] }; })();
function interpT(sq) { const s = DENSE.s; let lo = 0, hi = s.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (s[m] <= sq) lo = m; else hi = m; } const f = (sq - s[lo]) / (s[hi] - s[lo]); return DENSE.t[lo] + f * (DENSE.t[hi] - DENSE.t[lo]); }
function buildTrack(n, halfWidth = 3.0) {
  const length = DENSE.length, s = Array.from({ length: n }, (_, i) => i * length / n), tq = s.map(interpT), half = n >> 1;
  if (n % 2 === 0) for (let i = half + 1; i < n; i++) tq[i] = 2 * Math.PI - tq[n - i];   /* symmetric sampling, as the course */
  const xy = tq.map(curve), K = tq.map(t => { const [dx, dy] = curveD1(t), [ddx, ddy] = curveD2(t); return (dx * ddy - dy * ddx) / Math.pow(Math.hypot(dx, dy), 3); });
  if (n % 2 === 0) for (let i = half + 1; i < n; i++) K[i] = K[n - i];
  const heading = tq.map(t => { const [dx, dy] = curveD1(t); return Math.atan2(dy, dx); });
  for (let i = 1; i < n; i++) { while (heading[i] - heading[i - 1] > Math.PI) heading[i] -= 2 * Math.PI; while (heading[i] - heading[i - 1] < -Math.PI) heading[i] += 2 * Math.PI; }
  const ds = s.map((v, i) => (i + 1 < n ? s[i + 1] : length) - v);
  return { n, s, xy, heading, K, length, halfWidth, ds, normal: heading.map(h => [-Math.sin(h), Math.cos(h)]), toGlobal(ey) { return this.xy.map((p, i) => [p[0] + ey * this.normal[i][0], p[1] + ey * this.normal[i][1]]); } };
}
function frenetRates(vx, vy, r, epsi, kappa, ey) { const den = 1 - kappa * ey, tang = vx * Math.cos(epsi) - vy * Math.sin(epsi), sdot = tang / den, eydot = vx * Math.sin(epsi) + vy * Math.cos(epsi), epsidot = r - kappa * sdot; return { sdot, eydot, epsidot, dt_ds: 1 / sdot, dey_ds: eydot / sdot, depsi_ds: epsidot / sdot, denominator: den, tangential_velocity: tang }; }
const BARC = { m: 1.98, lf: 0.125, lr: 0.125, Iz: 0.024, B: 1.0, C: 1.25, peak: 0.8, get D() { return this.peak * this.m * G / 2; } };
const pacejka = (alphaRad, p = BARC) => p.D * Math.sin(p.C * Math.atan(p.B * alphaRad));
function lateralForces(vx, vy, r, delta, p = BARC) { const alpha_f = delta - Math.atan2(vy + p.lf * r, vx), alpha_r = -Math.atan2(vy - p.lr * r, vx); return { alpha_f, alpha_r, Fyf: pacejka(alpha_f, p), Fyr: pacejka(alpha_r, p) }; }
function bicycleDynamics(x, u, kappa, p = BARC) { const [vx, vy, r, epsi, , ey] = x, [delta, a] = u, f = lateralForces(vx, vy, r, delta, p), sdot = (vx * Math.cos(epsi) - vy * Math.sin(epsi)) / (1 - kappa * ey); return [a - f.Fyf * Math.sin(delta) / p.m + r * vy, (f.Fyf * Math.cos(delta) + f.Fyr) / p.m - r * vx, (p.lf * f.Fyf * Math.cos(delta) - p.lr * f.Fyr) / p.Iz, r - kappa * sdot, sdot, vx * Math.sin(epsi) + vy * Math.cos(epsi)]; }
function spatialRhs(state, ax, delta, kappa, p = BARC) { const xd = bicycleDynamics([state[0], state[1], state[2], state[3], 0, state[5]], [delta, ax], kappa, p), sdot = xd[4]; return [xd[0], xd[1], xd[2], xd[3], 1, xd[5]].map(v => v / sdot); }
const capZ = (K, mu, vmax, g = G) => K.map(k => Math.min(vmax * vmax, mu * g / Math.max(Math.abs(k), 1e-9)));
const segTime = (z0, z1, ds) => 2 * ds / (Math.sqrt(z0) + Math.sqrt(z1));
function planner(track, mu, vmax, amin, amax, tol = 1e-9, maxIter = 100) {
  const n = track.n, ds = track.ds, zmax = capZ(track.K, mu, vmax), z = zmax.slice(); let converged = false, iteration = 0;
  for (iteration = 1; iteration <= maxIter; iteration++) { const prev = z.slice(); for (let i = 0; i < n; i++) { const j = (i + 1) % n; z[j] = Math.min(z[j], z[i] + 2 * amax * ds[i]); } for (let i = n - 1; i >= 0; i--) { const j = (i + 1) % n; z[i] = Math.min(z[i], z[j] - 2 * amin * ds[i]); } let d = 0; for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(z[i] - prev[i])); if (d <= tol) { converged = true; break; } }
  const zn = z.map((_, i) => z[(i + 1) % n]), a = z.map((v, i) => (zn[i] - v) / (2 * ds[i])), st = z.map((v, i) => segTime(v, zn[i], ds[i])), feasible = z.every((v, i) => v <= zmax[i] + 1e-8 && v > 0) && a.every(v => v <= amax + 1e-8 && v >= amin - 1e-8);
  return { success: converged && feasible, iterations: iteration, time: st.reduce((p, q) => p + q, 0), v: z.map(Math.sqrt), a, z, zmax, segment_time: st };
}
const frictionUtil = (z, a, K, mu, g = G) => z.map((zi, i) => Math.hypot(a[i], zi * K[i]) / (mu * g));
/* NET-END */

function legend(ctx, lines, x, y, narrow) { ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; const out = narrow ? lines.flatMap(l => l.split(" · ")) : lines; out.forEach((l, i) => ctx.fillText(l, x, y + 13 * i)); }
const TRACKS = {}; const trackOf = n => TRACKS[n] || (TRACKS[n] = buildTrack(n, 3.0));
const T240 = trackOf(240), APEX = T240.K.reduce((b, k, i, K) => Math.abs(k) > Math.abs(K[b]) ? i : b, 0);
/* braking zone before the apex: walk back from the apex to the braking segments, then to their start */
function brakingZone(track, sol, amin) { const n = track.n, a = sol.a, near = (v, t) => Math.abs(v - t) < 1e-6, apex = track.K.reduce((b, k, i, K) => Math.abs(k) > Math.abs(K[b]) ? i : b, 0); let k = apex, guard = 0; while (!near(a[(k - 1 + n) % n], amin) && guard++ < n) k--; if (guard >= n) return null; const end = k; while (near(a[(k - 1 + n) % n], amin)) k--; return { start: ((k % n) + n) % n, end: ((end % n) + n) % n, apex, upstream: track.s[apex] - track.s[((k % n) + n) % n] }; }
/* equal-aspect map projection inside a frame */
function mapProj(fr, xr, yr) { const sc = Math.min(fr.iw / (xr[1] - xr[0]), fr.ih / (yr[1] - yr[0])), cx = (xr[0] + xr[1]) / 2, cy = (yr[0] + yr[1]) / 2; return { X: x => fr.pad.l + fr.iw / 2 + (x - cx) * sc, Y: y => fr.pad.t + fr.ih / 2 - (y - cy) * sc, sc }; }

/* =====================================================================
   Hero: the planner, live
   ===================================================================== */
const hs = { mu: 1.0, amax: 3.0, amin: -6.0, vmax: 35, n: 240, probe: APEX, sol: null };
function heroSolve() { const tr = trackOf(hs.n); hs.sol = planner(tr, hs.mu, hs.vmax, hs.amin, hs.amax); hs.sol.track = tr; }
function drawHero() {
  if (!hs.sol) heroSolve();
  const sol = hs.sol, tr = sol.track, n = tr.n, cv = $("#mapC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.6, { x: [-56, 56], y: [-40, 40], xticks: [], yticks: [] }), { ctx } = fr, P = mapProj(fr, [-56, 56], [-40, 40]); cv._fr = fr; cv._P = P;
  const inner = tr.toGlobal(-tr.halfWidth), outer = tr.toGlobal(tr.halfWidth), vlo = Math.min(...sol.v), vhi = Math.max(...sol.v);
  ctx.strokeStyle = alpha("--ink", 0.35); ctx.lineWidth = 1; [inner, outer].forEach(B => { ctx.beginPath(); B.forEach((p, i) => i ? ctx.lineTo(P.X(p[0]), P.Y(p[1])) : ctx.moveTo(P.X(p[0]), P.Y(p[1]))); ctx.closePath(); ctx.stroke(); });
  ctx.lineWidth = narrow ? 3 : 4; ctx.lineCap = "round";
  for (let i = 0; i < n; i++) { const j = (i + 1) % n, t = vhi > vlo ? (sol.v[i] - vlo) / (vhi - vlo) : 1; ctx.strokeStyle = alpha("--post", 0.15 + 0.85 * t); ctx.beginPath(); ctx.moveTo(P.X(tr.xy[i][0]), P.Y(tr.xy[i][1])); ctx.lineTo(P.X(tr.xy[j][0]), P.Y(tr.xy[j][1])); ctx.stroke(); }
  /* braking segments in orange */
  ctx.strokeStyle = css("--warm"); for (let i = 0; i < n; i++) if (Math.abs(sol.a[i] - hs.amin) < 1e-6) { const j = (i + 1) % n; ctx.beginPath(); ctx.moveTo(P.X(tr.xy[i][0]), P.Y(tr.xy[i][1])); ctx.lineTo(P.X(tr.xy[j][0]), P.Y(tr.xy[j][1])); ctx.stroke(); }
  const pr = tr.xy[hs.probe]; ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(P.X(pr[0]), P.Y(pr[1]), 5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = css("--bg"); ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText("start, s = 0 →", P.X(50) - (narrow ? 70 : 0), P.Y(0) + (narrow ? 16 : -10));
  legend(ctx, ["blue: planned speed (darker = faster) · orange: full braking · dot: the probe"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  /* profile */
  const cv2 = $("#profC"), fr2 = frame(cv2, narrow ? 1.5 : 2.6, { x: [0, tr.length], y: [0, Math.max(26, Math.ceil(Math.min(hs.vmax, Math.sqrt(Math.max(...sol.zmax))) / 5) * 5 + 2)], xlabel: "arc length s [m]", ylabel: "speed [m/s]", xticks: [0, 50, 100, 150, 200, 250], yticks: [0, 10, 20, 30] }), ctx2 = fr2.ctx; cv2._fr = fr2;
  plotLine(fr2, tr.s, sol.zmax.map(Math.sqrt).map(v => Math.min(v, 60)), { color: alpha("--ink", 0.45), width: 1.4, dash: [5, 4] });
  plotLine(fr2, tr.s, sol.v, { color: css("--post"), width: 2.2 });
  for (let i = 0; i < n; i++) if (Math.abs(sol.a[i] - hs.amin) < 1e-6) { ctx2.strokeStyle = css("--warm"); ctx2.lineWidth = 2.2; seg(ctx2, fr2.X(tr.s[i]), fr2.Y(sol.v[i]), fr2.X(tr.s[(i + 1) % n] || tr.length), fr2.Y(sol.v[(i + 1) % n])); }
  ctx2.strokeStyle = alpha("--ink", 0.5); ctx2.setLineDash([2, 3]); seg(ctx2, fr2.X(tr.s[hs.probe]), fr2.pad.t, fr2.X(tr.s[hs.probe]), fr2.pad.t + fr2.ih); ctx2.setLineDash([]);
  legend(ctx2, ["dashed: local curvature / top-speed cap · blue: minimum-time profile · orange: full braking"], fr2.pad.l + 4, fr2.pad.t + 12, narrow);
  /* readouts */
  $("#hMuV").textContent = fmt(hs.mu, 2); $("#hAmaxV").textContent = fmt(hs.amax, 1); $("#hAminV").textContent = fmt(hs.amin, 1); $("#hVmaxV").textContent = hs.vmax;
  $("#rLap").textContent = fmt(sol.time, 4) + " s"; $("#rSpeed").textContent = `${fmt(Math.min(...sol.v), 2)} to ${fmt(Math.max(...sol.v), 2)}`;
  const nAcc = sol.a.filter(v => Math.abs(v - hs.amax) < 1e-6).length, nBr = sol.a.filter(v => Math.abs(v - hs.amin) < 1e-6).length, nCap = sol.z.filter((v, i) => Math.abs(v - sol.zmax[i]) < 1e-6).length;
  $("#rActive").textContent = `${nAcc} / ${nBr} / ${nCap}`; const fu = frictionUtil(sol.z, sol.a, tr.K, hs.mu); $("#rFric").textContent = `${fmt(Math.max(...fu), 3)}, ${fu.filter(v => v > 1 + 1e-8).length} segs`;
  const i = hs.probe, k = tr.K[i], zone = brakingZone(tr, sol, hs.amin), nb = hs.mu === 1 && hs.amax === 3 && hs.amin === -6 && hs.vmax === 35 && hs.n === 240;
  const vmaxActive = sol.z.some((v, j) => Math.abs(v - hs.vmax * hs.vmax) < 1e-6);
  caption($("#heroStatus"), `${nb ? "The notebook's setting: " : ""}on the ${tr.n}-sample track (${fmt(tr.length, 1)} m) with [[\\mu = ${fmt(hs.mu, 2)}]], [[a \\in [${fmt(hs.amin, 1)}, ${fmt(hs.amax, 1)}] ]] and [[v_{\\max} = ${hs.vmax}]], the forward/backward planner converged in ${sol.iterations} sweeps${sol.success ? "" : " but the result is not feasible"}: lap time ${fmt(sol.time, 4)} s, speeds from ${fmt(Math.min(...sol.v), 3)} to ${fmt(Math.max(...sol.v), 3)} m/s. ${nAcc} segments at full acceleration, ${nBr} at full braking, ${nCap} samples on the local cap${vmaxActive ? "" : "; the top-speed limit is never active, the track is too twisty for it"}. ${zone ? `Before the tightest corner (apex at [[s = ${fmt(tr.s[zone.apex], 1)}]], radius ${fmt(1 / Math.abs(tr.K[zone.apex]), 2)} m) braking starts ${fmt(zone.upstream, 1)} m upstream, at ${fmt(sol.v[zone.start], 2)} m/s, and the corner is taken at ${fmt(sol.v[zone.apex], 3)} m/s, the curvature cap [[\\sqrt{\\mu g / |K|}]]. ` : ""}At the probe, [[s = ${fmt(tr.s[i], 1)}]]: [[K = ${fmt(k, 4)}]] (radius ${Math.abs(k) > 1e-6 ? fmt(1 / Math.abs(k), 1) + " m" : "infinite"}), cap ${fmt(Math.sqrt(sol.zmax[i]), 2)} m/s, planned ${fmt(sol.v[i], 2)} m/s, [[a_x = ${fmt(sol.a[i], 2)}]], lateral [[a_y = zK = ${fmt(sol.z[i] * k, 2)}]], friction-circle use ${fmt(fu[i], 2)}. The largest combined demand is ${fmt(Math.max(...fu), 3)} of [[\\mu g]] on ${fu.filter(v => v > 1 + 1e-8).length} segments: the decoupled planner is optimistic where it brakes or accelerates while cornering. Tap the map or the profile to move the probe.`);
}
function heroPick(ev) { const cv = ev.currentTarget, r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top, tr = hs.sol.track; let best = 0, bd = Infinity; if (cv.id === "mapC") { const P = cv._P; tr.xy.forEach((p, i) => { const d = Math.hypot(P.X(p[0]) - x, P.Y(p[1]) - y); if (d < bd) { bd = d; best = i; } }); } else { const fr = cv._fr, s = fr.x0 + (x - fr.pad.l) / fr.iw * (fr.x1 - fr.x0); tr.s.forEach((v, i) => { const d = Math.abs(v - s); if (d < bd) { bd = d; best = i; } }); } hs.probe = best; drawHero(); }
$("#mapC").addEventListener("click", heroPick); $("#profC").addEventListener("click", heroPick);
const heroInput = () => { hs.mu = +$("#hMu").value; hs.amax = +$("#hAmax").value; hs.amin = +$("#hAmin").value; hs.vmax = +$("#hVmax").value; heroSolve(); drawHero(); drawCorner(); drawFriction(); };
["hMu", "hAmax", "hAmin", "hVmax"].forEach(id => $("#" + id).addEventListener("input", heroInput));
$$("#demo [data-n]").forEach(b => b.addEventListener("click", () => { hs.n = +b.dataset.n; $$("#demo [data-n]").forEach(o => o.setAttribute("aria-pressed", String(o === b))); hs.probe = Math.round(hs.probe * hs.n / hs.sol.track.n) % hs.n; heroSolve(); drawHero(); drawCorner(); drawFriction(); }));
$("#hReset").addEventListener("click", () => { $("#hMu").value = 1; $("#hAmax").value = 3; $("#hAmin").value = -6; $("#hVmax").value = 35; hs.n = 240; $$("#demo [data-n]").forEach(o => o.setAttribute("aria-pressed", String(o.dataset.n === "240"))); hs.probe = APEX; heroInput(); });

/* =====================================================================
   Part 2: the Frenet position map and its singularity
   ===================================================================== */
const fg = { s: 99.9, ey: 1.5 };
function drawFrenet() {
  const tr = T240, n = tr.n, i = Math.round(fg.s / tr.length * n) % n, c = tr.xy[i], nv = tr.normal[i], tv = [Math.cos(tr.heading[i]), Math.sin(tr.heading[i])], K = tr.K[i], p = [c[0] + fg.ey * nv[0], c[1] + fg.ey * nv[1]];
  const cv = $("#frenetC"), narrow = narrowOf(cv), half = narrow ? 14 : 18, fr = frame(cv, narrow ? 1.2 : 1.7, { x: [c[0] - half, c[0] + half], y: [c[1] - half, c[1] + half], xlabel: "X [m]", ylabel: "Y [m]", xticks: [], yticks: [] }), { ctx } = fr, P = mapProj(fr, [c[0] - half, c[0] + half], [c[1] - half, c[1] + half]);
  const inner = tr.toGlobal(-3), outer = tr.toGlobal(3); ctx.lineWidth = 1; ctx.strokeStyle = alpha("--ink", 0.35); [inner, outer].forEach(B => { ctx.beginPath(); B.forEach((q, j) => j ? ctx.lineTo(P.X(q[0]), P.Y(q[1])) : ctx.moveTo(P.X(q[0]), P.Y(q[1]))); ctx.closePath(); ctx.stroke(); });
  ctx.strokeStyle = css("--post"); ctx.lineWidth = 2; ctx.beginPath(); tr.xy.forEach((q, j) => j ? ctx.lineTo(P.X(q[0]), P.Y(q[1])) : ctx.moveTo(P.X(q[0]), P.Y(q[1]))); ctx.closePath(); ctx.stroke();
  /* tangent, normal, the normal line and the centre of curvature */
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; seg(ctx, P.X(c[0]), P.Y(c[1]), P.X(c[0] + 4 * tv[0]), P.Y(c[1] + 4 * tv[1])); ctx.strokeStyle = css("--warm"); seg(ctx, P.X(c[0]), P.Y(c[1]), P.X(c[0] + 4 * nv[0]), P.Y(c[1] + 4 * nv[1]));
  ctx.strokeStyle = alpha("--ink", 0.35); ctx.setLineDash([3, 3]); seg(ctx, P.X(c[0] - 12 * nv[0]), P.Y(c[1] - 12 * nv[1]), P.X(c[0] + 12 * nv[0]), P.Y(c[1] + 12 * nv[1])); ctx.setLineDash([]);
  const R = Math.abs(K) > 1e-4 ? 1 / K : null; if (R && Math.abs(R) < 40) { const cc = [c[0] + R * nv[0], c[1] + R * nv[1]]; ctx.strokeStyle = alpha("--warm", 0.6); ctx.beginPath(); ctx.arc(P.X(cc[0]), P.Y(cc[1]), Math.abs(R) * P.sc, 0, 2 * Math.PI); ctx.stroke(); ctx.fillStyle = css("--warm"); ctx.beginPath(); ctx.arc(P.X(cc[0]), P.Y(cc[1]), 3.5, 0, 2 * Math.PI); ctx.fill(); }
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(P.X(p[0]), P.Y(p[1]), 5, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText("t", P.X(c[0] + 4.6 * tv[0]), P.Y(c[1] + 4.6 * tv[1])); ctx.fillText("n", P.X(c[0] + 4.6 * nv[0]), P.Y(c[1] + 4.6 * nv[1]));
  legend(ctx, ["blue: centreline · thin: track edges · black dot: p = γ(s) + e_y n(s) · orange dot: centre of curvature"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#fgSV").textContent = fmt(tr.s[i], 1); $("#fgEV").textContent = fmt(fg.ey, 2);
  const fac = 1 - K * fg.ey;
  caption($("#frenetCap"), `At [[s = ${fmt(tr.s[i], 1)}]] the centreline point is [[\\gamma = (${fmt(c[0], 2)}, ${fmt(c[1], 2)})]], tangent angle ${fmt(tr.heading[i] * 180 / Math.PI, 1)}°, left normal [[n = (${fmt(nv[0], 3)}, ${fmt(nv[1], 3)})]], curvature [[K = ${fmt(K, 4)}]] (${Math.abs(K) > 1e-4 ? `radius ${fmt(Math.abs(1 / K), 1)} m, turning ${K > 0 ? "left" : "right"}` : "practically straight"}). The offset point is [[p = (${fmt(p[0], 2)}, ${fmt(p[1], 2)})]]${Math.abs(fg.ey) > 3 ? ", outside the 3 m corridor" : ""}, and the geometric factor is [[1 - K e_y = ${fmt(fac, 3)}]]: ${fac > 0 ? `a step [[ds]] along the centreline moves this point by ${fmt(fac, 3)} ds, ${fac < 1 ? "less, on the inside of the bend" : "more, on the outside"}` : "zero or negative, the map has folded: the point has reached or passed the centre of curvature"}. The map is singular at [[e_y = 1/K${Math.abs(K) > 1e-4 ? " = " + fmt(1 / K, 1) + " m" : ""}]], the orange dot; on this track the corridor stays at least ${fmt(100 * 0.6117, 0)} percent away from it, [[\\min_s (1 - |K| W) = 0.612]].`);
}
$("#fgS").addEventListener("input", () => { fg.s = +$("#fgS").value; drawFrenet(); }); $("#fgE").addEventListener("input", () => { fg.ey = +$("#fgE").value; drawFrenet(); });

/* =====================================================================
   Part 3: Frenet rates and the spatial transformation
   ===================================================================== */
function drawRates() {
  const v = id => +$("#" + id).value, vx = v("rVx"), vy = v("rVy"), r = v("rR"), epsi = v("rE"), K = v("rK"), ey = v("rEy");
  ["rVx", "rVy", "rR", "rE", "rK", "rEy"].forEach(id => $("#" + id + "V").textContent = fmt(v(id), 2));
  const o = frenetRates(vx, vy, r, epsi, K, ey), nb = vx === 12 && vy === 0.4 && r === 0.15 && epsi === 0.05 && K === 0.02 && ey === 0.5;
  const rows = [["ṡ", o.sdot, "m/s"], ["ė_y", o.eydot, "m/s"], ["ė_ψ", o.epsidot, "rad/s"], ["dt/ds", o.dt_ds, "s/m"], ["de_y/ds", o.dey_ds, "–"], ["de_ψ/ds", o.depsi_ds, "rad/m"], ["1 − K e_y", o.denominator, "–"], ["v_t", o.tangential_velocity, "m/s"]];
  $("#ratesT").innerHTML = "<thead><tr><th>quantity</th><th>value</th><th>unit</th></tr></thead><tbody>" + rows.map(([n, val, u]) => `<tr><td>${n}</td><td>${Number.isFinite(val) ? fmt(val, 6) : "undefined"}</td><td class="words">${u}</td></tr>`).join("") + "</tbody>";
  caption($("#ratesCap"), `${nb ? "The notebook's example. " : ""}Tangential velocity [[v_t = ${fmt(o.tangential_velocity, 4)}]] divided by [[1 - K e_y = ${fmt(o.denominator, 3)}]] gives the progress rate [[\\dot s = ${fmt(o.sdot, 4)}]] m/s${o.sdot > 0 ? "" : ": not positive, so the spatial transformation is undefined here"}; the normal component [[\\dot e_y = ${fmt(o.eydot, 4)}]]; the heading error changes at [[r - K \\dot s = ${fmt(o.epsidot, 4)}]] rad/s. Dividing each by [[\\dot s]] gives the spatial rates: one metre of progress takes ${fmt(o.dt_ds, 4)} s and moves the car ${fmt(o.dey_ds, 4)} m sideways.`);
}
["rVx", "rVy", "rR", "rE", "rK", "rEy"].forEach(id => $("#" + id).addEventListener("input", drawRates));

/* =====================================================================
   Part 4: the tire curve and the bicycle model's spatial right-hand side
   ===================================================================== */
function drawTire() {
  const delta = +$("#tDelta").value; $("#tDeltaV").textContent = fmt(delta, 3);
  const f = lateralForces(10, 0.25, 0.2, delta), cv = $("#tireC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.5 : 2.3, { x: [-0.6, 0.6], y: [-9, 9], xlabel: "slip angle α [rad]", ylabel: "F_y [N]", xticks: [-0.5, -0.25, 0, 0.25, 0.5], yticks: [-8, -4, 0, 4, 8] }), { ctx, X, Y } = fr;
  const al = Array.from({ length: 121 }, (_, i) => -0.6 + 1.2 * i / 120); plotLine(fr, al, al.map(a => pacejka(a)), { color: css("--post"), width: 2.2 });
  ctx.strokeStyle = alpha("--ink", 0.4); ctx.setLineDash([4, 4]); seg(ctx, X(-0.6), Y(BARC.D), X(0.6), Y(BARC.D)); seg(ctx, X(-0.6), Y(-BARC.D), X(0.6), Y(-BARC.D)); ctx.setLineDash([]);
  [[f.alpha_f, f.Fyf, "front"], [f.alpha_r, f.Fyr, "rear"]].forEach(([a, F, lab]) => { ctx.fillStyle = lab === "front" ? css("--warm") : css("--ink"); ctx.beginPath(); ctx.arc(X(a), Y(F), 5, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left"; ctx.fillText(lab, X(a) + 7, Y(F) + 4); });
  legend(ctx, ["F_y = D sin(C atan(B α)), B = 1, C = 1.25, D = 7.77 N · dashed: ±D"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const st = [10, 0.25, 0.2, 0.04, 0, 0.3], rhs = spatialRhs(st, 1.0, delta, 0.03), nb = Math.abs(delta - 0.08) < 1e-9;
  caption($("#tireCap"), `At [[v_x = 10]], [[v_y = 0.25]], [[r = 0.2]] and steering [[\\delta = ${fmt(delta, 3)}]]: slip angles [[\\alpha_f = ${fmt(f.alpha_f, 5)}]] and [[\\alpha_r = ${fmt(f.alpha_r, 5)}]] (the formulas above), forces [[F_{y,f} = ${fmt(f.Fyf, 4)}]] N and [[F_{y,r} = ${fmt(f.Fyr, 4)}]] N${nb ? ", the notebook's values" : ""}. The curve saturates at [[D = 0.8 \\cdot m g / 2 = 7.77]] N per axle. With [[a_x = 1]], [[K = 0.03]], [[e_\\psi = 0.04]], [[e_y = 0.3]] the spatial right-hand side is [[v_x' = ${fmt(rhs[0], 4)}]], [[v_y' = ${fmt(rhs[1], 4)}]], [[r' = ${fmt(rhs[2], 4)}]], [[e_\\psi' = ${fmt(rhs[3], 4)}]], [[t' = ${fmt(rhs[4], 4)}]], [[e_y' = ${fmt(rhs[5], 4)}]]: every time derivative divided by [[\\dot s]], and [[t' = 1/\\dot s]] in s/m.`);
}
$("#tDelta").addEventListener("input", drawTire);

/* =====================================================================
   Part 7: around the tightest corner, and the braking sweep
   ===================================================================== */
const cs = { amin: -6 };
function drawCorner() {
  const tr = hs.sol.track, n = tr.n, apex = tr.K.reduce((b, k, i, K) => Math.abs(k) > Math.abs(K[b]) ? i : b, 0), win = Math.round(35 * n / 240), idx = Array.from({ length: 2 * win + 1 }, (_, j) => ((apex - win + j) % n + n) % n), mds = tr.length / n, dist = idx.map((_, j) => (j - win) * mds);
  const other = planner(tr, hs.mu, hs.vmax, cs.amin, hs.amax), sol = hs.sol;
  const cv = $("#cornerC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.4 : 2.3, { x: [dist[0], dist[dist.length - 1]], y: [0, 26], xlabel: "distance from the apex [m]", ylabel: "speed [m/s]", xticks: [-30, -20, -10, 0, 10, 20, 30], yticks: [0, 10, 20] }), { ctx, X, Y } = fr;
  plotLine(fr, dist, idx.map(i => Math.min(Math.sqrt(sol.zmax[i]), 60)), { color: alpha("--ink", 0.45), width: 1.4, dash: [5, 4] });
  plotLine(fr, dist, idx.map(i => sol.v[i]), { color: css("--post"), width: 2.2 });
  if (Math.abs(cs.amin - hs.amin) > 1e-9) plotLine(fr, dist, idx.map(i => other.v[i]), { color: css("--warm"), width: 2 });
  ctx.strokeStyle = alpha("--ink", 0.5); ctx.setLineDash([2, 3]); seg(ctx, X(0), fr.pad.t, X(0), fr.pad.t + fr.ih); ctx.setLineDash([]);
  legend(ctx, [`dashed: local cap · blue: a_min = ${fmt(hs.amin, 1)} (the demo's) · orange: a_min = ${fmt(cs.amin, 1)}`], fr.pad.l + 4, fr.pad.t + 12, narrow);
  $("#csAV").textContent = fmt(cs.amin, 1);
  const z1 = brakingZone(tr, sol, hs.amin), z2 = brakingZone(tr, other, cs.amin);
  caption($("#cornerCap"), `The tightest sampled corner is at [[s = ${fmt(tr.s[apex], 2)}]] m, [[K = ${fmt(tr.K[apex], 5)}]], radius ${fmt(1 / Math.abs(tr.K[apex]), 2)} m, where the cap is [[\\sqrt{\\mu g / |K|} = ${fmt(Math.sqrt(sol.zmax[apex]), 3)}]] m/s. ${z1 ? `With [[a_{\\min} = ${fmt(hs.amin, 1)}]] full braking runs from ${fmt(z1.upstream, 1)} m before the apex (${fmt(sol.v[z1.start], 2)} m/s) to ${fmt(tr.s[apex] - tr.s[z1.end], 1)} m before it (${fmt(sol.v[z1.end], 2)} m/s); the last metres follow the cap itself, which falls more slowly than full braking would.` : ""} ${z2 && Math.abs(cs.amin - hs.amin) > 1e-9 ? `With [[a_{\\min} = ${fmt(cs.amin, 1)}]] braking starts ${fmt(z2.upstream, 1)} m upstream instead, from ${fmt(other.v[z2.start], 2)} m/s, and the lap takes ${fmt(other.time, 4)} s against ${fmt(sol.time, 4)} s.` : "Move the slider to compare another braking limit."} The apex speed itself never changes with the braking limit: it is fixed by [[\\mu]] and [[K]]. Braking authority decides how late the braking can start, acceleration authority how fast the speed recovers after the apex.`);
}
$("#csA").addEventListener("input", () => { cs.amin = +$("#csA").value; drawCorner(); });

/* =====================================================================
   Part 9: friction, authority, grid
   ===================================================================== */
function drawSweeps() {
  const tr = T240, mus = [0.6, 0.8, 1.0, 1.2], sols = mus.map(mu => planner(tr, mu, 35, -6, 3)), cases = [["baseline", -6, 3], ["weaker braking", -3.5, 3], ["weaker acceleration", -6, 1.5]].map(([nm, lo, hi]) => [nm, planner(tr, 1, 35, lo, hi)]);
  const cv = $("#sweepC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.5 : 2.6, { x: [0, tr.length], y: [0, 26], xlabel: "arc length s [m]", ylabel: "speed [m/s]", xticks: [0, 50, 100, 150, 200, 250], yticks: [0, 10, 20] }), { ctx } = fr;
  sols.forEach((s, k) => plotLine(fr, tr.s, s.v, { color: alpha("--post", 0.3 + 0.7 * k / 3), width: 1.8 }));
  legend(ctx, ["μ = 0.6, 0.8, 1.0, 1.2 (darker = more grip), a ∈ [−6, 3]"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const grids = [60, 120, 240, 480, 960].map(n => { const t = trackOf(n); return [n, planner(t, 1, 35, -6, 3).time, Math.max(...t.K.map(Math.abs))]; });
  $("#sweepT").innerHTML = "<thead><tr><th>setting</th><th>lap time [s]</th></tr></thead><tbody>" + mus.map((mu, k) => `<tr><td>μ = ${fmt(mu, 1)}</td><td>${fmt(sols[k].time, 4)}</td></tr>`).join("") + cases.map(([nm, s]) => `<tr><td>${nm} (a ∈ [${s === cases[1][1] ? "−3.5, 3" : s === cases[2][1] ? "−6, 1.5" : "−6, 3"}])</td><td>${fmt(s.time, 4)}</td></tr>`).join("") + grids.map(([n, T, mk]) => `<tr><td>grid N = ${n}, max |K| = ${fmt(mk, 5)}</td><td>${fmt(T, 6)}</td></tr>`).join("") + "</tbody>";
  const d = sols[3].v.map((v, i) => v - sols[0].v[i]), im = d.indexOf(Math.max(...d)), il = d.indexOf(Math.min(...d));
  caption($("#sweepCap"), `All computed live. Friction: lap times ${sols.map(s => fmt(s.time, 4)).join(", ")} s for [[\\mu = 0.6, 0.8, 1.0, 1.2]]; more grip never hurts, because a larger [[\\mu]] only enlarges the feasible set. The gain is largest in the corners and their exits (${fmt(Math.max(...d), 2)} m/s at [[s = ${fmt(tr.s[im], 1)}]]) and smallest on the long sides (${fmt(Math.min(...d), 2)} m/s at [[s = ${fmt(tr.s[il], 1)}]]), where the speed is set by acceleration authority, not by grip. Authority: weaker braking costs ${fmt(cases[1][1].time - cases[0][1].time, 4)} s and moves the braking zones upstream; weaker acceleration costs ${fmt(cases[2][1].time - cases[0][1].time, 4)} s and slows every corner exit. Grid: the lap time rises with the number of samples, ${grids.map(g => fmt(g[1], 4)).join(", ")} s, because a coarse grid steps over the apex and underestimates the peak curvature (max |K| ${fmt(grids[0][2], 4)} at 60 samples against ${fmt(grids[4][2], 4)} at 960); the increments shrink, ${grids.slice(1).map((g, k) => fmt(g[1] - grids[k][1], 4)).join(", ")} s, so the 240-sample value is about ${fmt(grids[4][1] - grids[2][1], 3)} s optimistic.`);
}

/* =====================================================================
   Part 10: the friction-circle diagnostic
   ===================================================================== */
function drawFriction() {
  const sol = hs.sol, tr = sol.track, fu = frictionUtil(sol.z, sol.a, tr.K, hs.mu), cv = $("#fricC"), narrow = narrowOf(cv);
  const fr = frame(cv, narrow ? 1.3 : 2.0, { x: [-1.4, 1.4], y: [-1.4, 1.4], xlabel: "a_y / μg", ylabel: "a_x / μg", xticks: [-1, 0, 1], yticks: [-1, 0, 1] }), { ctx } = fr, P = mapProj(fr, [-1.4, 1.4], [-1.4, 1.4]);
  ctx.strokeStyle = alpha("--ink", 0.5); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.arc(P.X(0), P.Y(0), P.sc, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = alpha("--ink", 0.25); seg(ctx, P.X(-1.4), P.Y(hs.amax / (hs.mu * G)), P.X(1.4), P.Y(hs.amax / (hs.mu * G))); seg(ctx, P.X(-1.4), P.Y(hs.amin / (hs.mu * G)), P.X(1.4), P.Y(hs.amin / (hs.mu * G))); seg(ctx, P.X(-1), P.Y(-1.4), P.X(-1), P.Y(1.4)); seg(ctx, P.X(1), P.Y(-1.4), P.X(1), P.Y(1.4));
  sol.z.forEach((z, i) => { const ay = z * tr.K[i] / (hs.mu * G), ax = sol.a[i] / (hs.mu * G); ctx.fillStyle = fu[i] > 1 + 1e-8 ? css("--warm") : alpha("--post", 0.6); ctx.beginPath(); ctx.arc(P.X(ay), P.Y(ax), 2.5, 0, 2 * Math.PI); ctx.fill(); });
  legend(ctx, ["one dot per segment · dashed circle: a_x² + a_y² = (μg)² · thin box: the planner's separate limits · orange: outside the circle"], fr.pad.l + 4, fr.pad.t + 12, narrow);
  const mx = Math.max(...fu), im = fu.indexOf(mx);
  caption($("#fricCap"), `The planner bounds [[a_x]] and [[a_y = zK]] separately, so its solution lives in the box, and the box's corners stick out of the friction circle. Here ${fu.filter(v => v > 1 + 1e-8).length} of ${tr.n} segments exceed the combined limit, the worst by a factor ${fmt(mx, 4)} at [[s = ${fmt(tr.s[im], 1)}]] where the car ${sol.a[im] > 0 ? "accelerates" : "brakes"} at ${fmt(sol.a[im], 2)} m/s² while cornering at ${fmt(sol.z[im] * tr.K[im], 2)} m/s² of lateral acceleration. That is not a numerical error: it is the reduced model allowing what a tire cannot give. A full raceline formulation imposes the circle (or a tire model) instead; exercise 12 asks what that does to the two passes.`);
}

function drawAll() { drawHero(); drawFrenet(); drawRates(); drawTire(); drawCorner(); drawSweeps(); drawFriction(); }
heroSolve(); drawAll();
onRedraw(drawAll);
})();
