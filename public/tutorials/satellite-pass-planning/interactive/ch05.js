/* Satellite Pass Planning from Scratch, chapter 05, From Geometry to a Planning Problem: the live demos.
   The course's small instance (window.SAT_SMALL, exported from satplan) drawn as a planner sees it: the targets
   on a map, coloured by priority and sized by listening time, and every opportunity of the four satellites on a
   six-hour Gantt chart with the ground contacts and the revolution boundaries. A second widget shows how one
   opportunity is given its revolution and its memory segment from its window midpoint (bisect_right).
   Needs ../kit/kit.js, sat.js and instance-small.js. */
(() => {
"use strict";
const { $, css, fmt, caption, fit, frame, alpha, onRedraw, narrowOf, seg } = window.Kit;
const Sat = window.Sat;
const D = window.SAT_SMALL;
const OPPS = D.opps, P = D.period, H = D.horizon, NS = D.nSats, NT = D.nTargets;

/* ---------------- shared helpers ---------------- */
const PRIO_TOKEN = { 1: "--prior", 2: "--c4", 4: "--post", 8: "--warm" };
const radiusOf = d => 1.6 + d / 30;              // 30 s -> 2.6 px, 120 s -> 5.6 px
const hhmm = t => { const m = Math.round(t / 60); return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`; };
const byTarget = new Map();
for (const o of OPPS) { if (!byTarget.has(o.target)) byTarget.set(o.target, []); byTarget.get(o.target).push(o.idx); }
const unreachable = [];
for (let j = 0; j < NT; j++) if (!byTarget.has(j)) unreachable.push(j);
const revBounds = [];
for (let k = 1; k * P < H; k++) revBounds.push(k * P);
const contactStarts = D.contacts.map(cs => cs.map(c => c[0]));
/* bisect.bisect_right(a, x): the number of elements of the sorted list a that are <= x. */
function bisectRight(a, x) { let lo = 0, hi = a.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (x < a[mid]) hi = mid; else lo = mid + 1; } return lo; }
/* Check the export against the rules of instance.py (should print nothing). */
for (const o of OPPS) {
  const mid = 0.5 * (o.ws + o.we);
  if (Math.floor(mid / P) !== o.rev || bisectRight(contactStarts[o.sat], mid) !== o.mem) console.warn("segment mismatch", o);
}

/* ---------------- hero: the map ---------------- */
const st = { target: 11, tracks: true };
let tracks = null;
function groundTracks() {
  if (tracks) return tracks;
  tracks = D.orbits.map(([alt, inc, raan, ph]) => {
    const orb = Sat.orbit(alt, inc, raan, ph), pts = [];
    for (let t = 0; t <= H; t += 60) pts.push(Sat.subPoint(Sat.positionECI(orb, t), t));
    return Sat.splitAtDateLine(pts);
  });
  return tracks;
}
let mapGeom = null;
function drawMap() {
  const cv = $("#mapC");
  const { ctx, w, h } = fit(cv, 2);
  const pad = 4, mw = w - 2 * pad, mh = mw / 2;
  const pr = Sat.mapProjection(pad, (h - mh) / 2, mw, mh);
  mapGeom = { pr };
  ctx.fillStyle = css("--surface-2"); ctx.fillRect(pr.X(-180), pr.Y(90), mw, mh);
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  for (let lo = -180; lo <= 180; lo += 30) seg(ctx, pr.X(lo), pr.Y(90), pr.X(lo), pr.Y(-90));
  for (let la = -90; la <= 90; la += 30) seg(ctx, pr.X(-180), pr.Y(la), pr.X(180), pr.Y(la));
  ctx.strokeStyle = css("--rule"); seg(ctx, pr.X(-180), pr.Y(0), pr.X(180), pr.Y(0));
  ctx.font = "10px " + css("--font-mono"); ctx.fillStyle = css("--muted"); ctx.textBaseline = "top"; ctx.textAlign = "left";
  for (const la of [60, 30, 0, -30, -60]) ctx.fillText((la > 0 ? la + "°N" : la < 0 ? -la + "°S" : "0°"), pr.X(-180) + 3, pr.Y(la) + 2);
  if (st.tracks) {
    ctx.strokeStyle = alpha("--muted", 0.35); ctx.lineWidth = 1;
    for (const pieces of groundTracks()) for (const piece of pieces) {
      ctx.beginPath(); piece.forEach((p, k) => { const x = pr.X(p.lon), y = pr.Y(p.lat); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    }
  }
  // maritime area boxes
  ctx.strokeStyle = css("--ink-2"); ctx.lineWidth = 1; ctx.setLineDash([3, 2]);
  D.areaBoxes.forEach(([la0, la1, lo0, lo1]) => ctx.strokeRect(pr.X(lo0), pr.Y(la1), pr.X(lo1) - pr.X(lo0), pr.Y(la0) - pr.Y(la1)));
  ctx.setLineDash([]);
  // ground stations
  ctx.fillStyle = css("--ink");
  for (const [, la, lo] of D.stations) { const x = pr.X(lo), y = pr.Y(la); ctx.beginPath(); ctx.moveTo(x, y - 5); ctx.lineTo(x + 4.5, y + 3.5); ctx.lineTo(x - 4.5, y + 3.5); ctx.closePath(); ctx.fill(); }
  // targets, small ones last so they stay visible
  const order = [...Array(NT).keys()].sort((a, b) => D.targetDurations[b] - D.targetDurations[a]);
  for (const j of order) {
    const [la, lo] = D.targets[j], x = pr.X(lo), y = pr.Y(la), r = radiusOf(D.targetDurations[j]);
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
    if (byTarget.has(j)) { ctx.fillStyle = alpha(PRIO_TOKEN[D.targetWeights[j]], 0.85); ctx.fill(); }
    else { ctx.strokeStyle = css(PRIO_TOKEN[D.targetWeights[j]]); ctx.lineWidth = 1.4; ctx.stroke(); }
  }
  // the selected target
  const [la, lo] = D.targets[st.target], x = pr.X(lo), y = pr.Y(la);
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x, y, radiusOf(D.targetDurations[st.target]) + 4, 0, 2 * Math.PI); ctx.stroke();
}

/* ---------------- hero: the Gantt chart ---------------- */
function drawGantt() {
  const cv = $("#ganttC"), narrow = narrowOf(cv, 560);
  const xt = []; for (let t = 0; t <= H; t += narrow ? 7200 : 3600) xt.push(t);
  const fr = frame(cv, narrow ? 1.5 : 2.7, { x: [0, H], y: [0, NS], yticks: false, ygrid: false, pad: { l: 46, t: 18 }, xticks: xt, xfmt: t => hhmm(t), xlabel: "time since the start of the horizon [h:mm]" });
  const { ctx, X, Y } = fr;
  const top = s => Y(NS - s), bot = s => Y(NS - s - 1);
  ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillStyle = css("--ink-2");
  for (let s = 0; s < NS; s++) ctx.fillText("sat " + s, fr.pad.l - 6, (top(s) + bot(s)) / 2);
  for (let s = 0; s < NS; s++) {
    const y0 = top(s) + 3, hh = bot(s) - top(s) - 6;
    ctx.fillStyle = alpha("--c4", 0.22);
    for (const [c0, c1] of D.contacts[s]) ctx.fillRect(X(c0), y0, Math.max(1, X(c1) - X(c0)), hh);
    ctx.fillStyle = alpha("--post", 0.2);
    for (const o of OPPS) if (o.sat === s) ctx.fillRect(X(o.ws), y0 + hh * 0.15, Math.max(1.5, X(o.we) - X(o.ws)), hh * 0.7);
    ctx.strokeStyle = css("--grid"); seg(ctx, fr.pad.l, bot(s) + 0.5, fr.w - fr.pad.r, bot(s) + 0.5);
  }
  ctx.strokeStyle = alpha("--ink", 0.55); ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
  ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillStyle = css("--muted"); ctx.font = "10px " + css("--font-mono");
  revBounds.forEach((b, k) => { seg(ctx, X(b), top(0), X(b), bot(NS - 1)); ctx.fillText("rev " + (k + 1), X(b), top(0) - 2); });
  ctx.setLineDash([]);
  ctx.fillText("rev 0", X(Math.min(P, H) / 2), top(0) - 2);
  // the selected target's opportunities
  for (const i of byTarget.get(st.target) || []) {
    const o = OPPS[i], y0 = top(o.sat) + 3, hh = bot(o.sat) - top(o.sat) - 6, x0 = X(o.ws), ww = Math.max(4, X(o.we) - X(o.ws));
    ctx.fillStyle = css("--warm"); ctx.fillRect(x0, y0, ww, hh);
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; ctx.strokeRect(x0, y0, ww, hh);
  }
}

/* ---------------- hero: readouts and caption ---------------- */
function describeTarget() {
  const j = st.target, [la, lo, area] = D.targets[j], w = D.targetWeights[j], d = D.targetDurations[j], ids = byTarget.get(j) || [];
  $("#rArea").textContent = D.areas[area];
  $("#rPos").textContent = `${fmt(Math.abs(la), 2)}°${la >= 0 ? "N" : "S"}, ${fmt(Math.abs(lo), 2)}°${lo >= 0 ? "E" : "W"}`;
  $("#rPrio").textContent = `${w}  ·  ${d} s`;
  $("#rN").textContent = ids.length;
  const lines = ids.map(i => { const o = OPPS[i]; return `opp ${i}: sat ${o.sat}, window [${o.ws}, ${o.we}] s, start in [${o.ws}, ${Sat.latest(o)}], rev ${o.rev}, memory segment ${o.mem}`; });
  $("#oppList").textContent = lines.length ? lines.join("\n") : "no opportunity";
  let text = `<b>Target ${j}</b> in the ${D.areas[area]} box, priority ${w}, needs ${d} s of listening. `;
  if (ids.length) {
    const sats = [...new Set(ids.map(i => OPPS[i].sat))];
    text += `It has <b>${ids.length} opportunit${ids.length > 1 ? "ies" : "y"}</b> (orange on the chart), on satellite${sats.length > 1 ? "s" : ""} ${sats.join(" and ")}: `;
    text += ids.map(i => { const o = OPPS[i]; return `a ${o.we - o.ws} s window from ${hhmm(o.ws)}, in which a ${d} s task can start anywhere in a ${Sat.latest(o) - o.ws} s range`; }).join("; ") + ". ";
    text += `A plan may use at most one of them. On the whole instance, ${OPPS.length} opportunities serve ${byTarget.size} of the ${NT} targets; the blue bars are all of them, darker where windows of one pass overlap, and the green bands are the ground contacts that empty the memory.`;
  } else {
    text += `It has <b>no opportunity</b> (hollow circle): in these 6 h no satellite sees it inside the 55° off-nadir cone above 10° elevation for as long as ${d} s. All ${unreachable.length} unreachable targets are in the Bay of Bengal: in these 6 h the ground tracks (grey) do not pass close enough to that part of the box for long enough.`;
  }
  caption($("#heroStatus"), text);
}
function drawHero() { drawMap(); drawGantt(); describeTarget(); }

/* target selection: the select, the slider and a click on the map */
const sel = $("#tSel"), sld = $("#tSlide");
for (let j = 0; j < NT; j++) {
  const op = document.createElement("option");
  op.value = j; op.textContent = `target ${j} · ${D.areas[D.targets[j][2]]} · priority ${D.targetWeights[j]} · ${D.targetDurations[j]} s${byTarget.has(j) ? "" : " · unreachable"}`;
  sel.appendChild(op);
}
function setTarget(j) { st.target = j; sel.value = j; sld.value = j; $("#tSlideV").textContent = j; drawHero(); }
sel.addEventListener("change", () => setTarget(+sel.value));
sld.addEventListener("input", () => setTarget(+sld.value));
$("#mapC").addEventListener("click", ev => {
  if (!mapGeom) return;
  const r = ev.currentTarget.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  let best = -1, bd = 14 * 14;
  for (let j = 0; j < NT; j++) {
    const dx = mapGeom.pr.X(D.targets[j][1]) - x, dy = mapGeom.pr.Y(D.targets[j][0]) - y, dd = dx * dx + dy * dy;
    if (dd < bd) { bd = dd; best = j; }
  }
  if (best >= 0) setTarget(best);
});
$("#tUnreach").addEventListener("click", () => { const k = unreachable.indexOf(st.target); setTarget(unreachable[(k + 1) % unreachable.length]); });
$("#tMost").addEventListener("click", () => {
  const many = [...byTarget.keys()].filter(j => byTarget.get(j).length === 3).sort((a, b) => a - b);
  const k = many.indexOf(st.target); setTarget(many[(k + 1) % many.length]);
});
$("#tTracks").addEventListener("change", e => { st.tracks = e.target.checked; drawMap(); });

/* ---------------- widget: revolution and memory segment from the midpoint ---------------- */
const sg = { sat: 3, k: 0 };
const satOpps = s => OPPS.filter(o => o.sat === s).sort((a, b) => a.ws - b.ws || a.idx - b.idx);
const straddlers = OPPS.filter(o => contactStarts[o.sat].some(c => o.ws < c && c < o.we) || revBounds.some(b => o.ws < b && b < o.we)).map(o => o.idx);
function segLine(fr, s, o, x0, x1) {
  const { ctx, X, Y } = fr, yTop = Y(1), yBot = Y(0);
  ctx.fillStyle = alpha("--c4", 0.25);
  for (const [c0, c1] of D.contacts[s]) if (c1 >= x0 && c0 <= x1) ctx.fillRect(X(Math.max(c0, x0)), yTop, Math.max(1, X(Math.min(c1, x1)) - X(Math.max(c0, x0))), yBot - yTop);
  ctx.strokeStyle = css("--c4"); ctx.lineWidth = 2;
  for (const c0 of contactStarts[s]) if (c0 >= x0 && c0 <= x1) seg(ctx, X(c0), yTop, X(c0), yBot);
  ctx.strokeStyle = alpha("--ink", 0.6); ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
  for (const b of revBounds) if (b >= x0 && b <= x1) seg(ctx, X(b), yTop, X(b), yBot);
  ctx.setLineDash([]);
  ctx.fillStyle = alpha("--post", 0.22);
  for (const q of OPPS) if (q.sat === s && q.we >= x0 && q.ws <= x1) ctx.fillRect(X(Math.max(q.ws, x0)), Y(0.62), Math.max(1.2, X(Math.min(q.we, x1)) - X(Math.max(q.ws, x0))), Y(0.38) - Y(0.62));
  const mid = 0.5 * (o.ws + o.we);
  ctx.fillStyle = css("--warm"); ctx.fillRect(X(o.ws), Y(0.7), Math.max(3, X(o.we) - X(o.ws)), Y(0.3) - Y(0.7));
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; seg(ctx, X(mid), Y(0.85), X(mid), Y(0.15));
}
function drawSeg() {
  const s = sg.sat, list = satOpps(s), o = list[Math.min(sg.k, list.length - 1)], mid = 0.5 * (o.ws + o.we);
  const cs = contactStarts[s];
  // overview: the whole horizon, with the memory segment numbers between contact starts
  const cv1 = $("#segC1"), narrow = narrowOf(cv1, 560);
  const xt = []; for (let t = 0; t <= H; t += narrow ? 7200 : 3600) xt.push(t);
  const fr1 = frame(cv1, narrow ? 3.2 : 6, { x: [0, H], y: [0, 1.25], yticks: false, ygrid: false, pad: { l: 8, t: 6 }, xticks: xt, xfmt: hhmm });
  segLine(fr1, s, o, 0, H);
  const ctx = fr1.ctx; ctx.font = "10px " + css("--font-mono"); ctx.fillStyle = css("--ink-2"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const edges = [0, ...cs, H];
  for (let k = 0; k + 1 < edges.length; k++) if (fr1.X(edges[k + 1]) - fr1.X(edges[k]) > 16) ctx.fillText("m" + k, (fr1.X(edges[k]) + fr1.X(edges[k + 1])) / 2, fr1.Y(1.1));
  // zoom: 15 minutes either side of the window
  const z0 = Math.max(0, o.ws - 900), z1 = Math.min(H, o.we + 900);
  const cv2 = $("#segC2");
  const fr2 = frame(cv2, narrow ? 2.6 : 5, { x: [z0, z1], y: [0, 1.1], yticks: false, ygrid: false, pad: { l: 8, t: 6 }, xfmt: hhmm, xlabel: "zoom: 15 min either side of the window [h:mm]" });
  segLine(fr2, s, o, z0, z1);
  const k = bisectRight(cs, mid), r = Math.floor(mid / P);
  $("#sgV").textContent = `${sg.k + 1} of ${list.length} (opp ${o.idx})`;
  const crossC = cs.filter(c => o.ws < c && c < o.we), crossR = revBounds.filter(b => o.ws < b && b < o.we);
  let text = `Satellite ${s}, <b>opportunity ${o.idx}</b> (orange): window [${o.ws}, ${o.we}] s, midpoint [[m = ${fmt(mid, 0)}]] s (black tick). `;
  text += `Revolution: [[\\lfloor m / T \\rfloor = \\lfloor ${fmt(mid, 0)} / ${fmt(P, 1)} \\rfloor = ${r}]]. `;
  text += `Memory segment: the contact starts of this satellite are [${cs.join(", ")}] s (green lines), and <code>bisect_right</code> counts those at or before [[m]]: <b>${k}</b>, so the task is recorded in segment m${k} and dumped during contact ${k}${k < cs.length ? ` (from ${cs[k]} s)` : ", which is after the horizon: no contact empties it within the 6 h"}. `;
  if (crossC.length || crossR.length) {
    const B = crossC.length ? crossC[0] : crossR[0], after = mid >= B;
    const movable = after ? o.ws + o.dur <= B : Sat.latest(o) >= B;
    text += `This window <b>straddles</b> ${crossC.length ? `the start of a contact at ${crossC.join(", ")} s` : ""}${crossC.length && crossR.length ? " and " : ""}${crossR.length ? `a revolution boundary at ${crossR.map(b => fmt(b, 0)).join(", ")} s` : ""}: the midpoint is ${after ? "after" : "before"} it, so the whole window is charged to the ${after ? "later" : "earlier"} side. `;
    text += movable ? `A ${o.dur} s task could lie wholly on the other side, and its seconds would then be charged to the wrong segment.` : `No ${o.dur} s task fits wholly on the other side (it would have to ${after ? `end by ${B} s but cannot start before ${o.ws} s` : `start at ${B} s or later but must start by ${Sat.latest(o)} s`}), so here the rule only misplaces the part of a task that overlaps the boundary.`;
  }
  else text += `The window lies wholly inside one revolution and one memory segment, as ${OPPS.length - straddlers.length} of the ${OPPS.length} windows do.`;
  if (OPPS[o.idx].rev !== r || OPPS[o.idx].mem !== k) text += " (Mismatch with the Python export: please report.)";
  caption($("#segCap"), text);
}
const sgSat = $("#sgSat"), sgK = $("#sgK");
function syncSegSlider() { sgK.max = satOpps(sg.sat).length - 1; sgK.value = sg.k; }
sgSat.addEventListener("change", () => { sg.sat = +sgSat.value; sg.k = 0; syncSegSlider(); drawSeg(); });
sgK.addEventListener("input", () => { sg.k = +sgK.value; drawSeg(); });
$("#sgNext").addEventListener("click", () => {
  const cur = satOpps(sg.sat)[sg.k].idx, k = straddlers.indexOf(cur), idx = straddlers[(k + 1) % straddlers.length], o = OPPS[idx];
  sg.sat = o.sat; sgSat.value = o.sat; sg.k = satOpps(o.sat).findIndex(q => q.idx === idx); syncSegSlider(); drawSeg();
});

/* ---------------- start ---------------- */
function drawAll() { drawHero(); drawSeg(); }
sel.value = st.target; sld.value = st.target; $("#tSlideV").textContent = st.target;
sgSat.value = sg.sat; syncSegSlider();
drawAll(); onRedraw(drawAll);
})();
