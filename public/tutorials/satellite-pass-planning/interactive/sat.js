/* Satellite Pass Planning from Scratch: the shared model, ported line for line from the satplan
   package (github.com/ElieRouphael/satellite-pass-planning) so that every chapter computes what the
   Python computes. Geometry (satplan/geometry.py), transitions and conflict pairs (instance.py),
   the independent checker (checker.py) and the greedy baseline (greedy.py). Exposed as window.Sat.
   Load after ../kit/kit.js (it does not depend on it) and before the chapter's own script. */
(function (global) {
"use strict";

/* ---------------- constants (geometry.py) ---------------- */
const MU_EARTH = 3.986004418e14;   // m^3 / s^2
const R_EARTH = 6371000.0;         // m, mean radius
const OMEGA_EARTH = 7.2921159e-5;  // rad / s
const RAD = Math.PI / 180, DEG = 180 / Math.PI;

/* ---------------- small vector helpers ---------------- */
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = a => Math.sqrt(dot(a, a));
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const unit = a => scale(a, 1 / norm(a));

/* ---------------- orbits ---------------- */
/* A circular two-body orbit: altitude [m], inclination, right ascension of the ascending node (RAAN)
   and argument of latitude at t = 0 [deg]. */
function orbit(altitude_m, inclination_deg, raan_deg, phase_deg) {
  const a = R_EARTH + altitude_m;
  const period = 2 * Math.PI * Math.sqrt(a ** 3 / MU_EARTH);
  return { altitude_m, inclination_deg, raan_deg, phase_deg, a, period };
}
/* Inertial (ECI) position [m] at time t [s]: rotate the in-plane point (a cos u, a sin u, 0) by the
   inclination about x, then by the RAAN about z. */
function positionECI(o, t) {
  const u = o.phase_deg * RAD + 2 * Math.PI * t / o.period;
  const inc = o.inclination_deg * RAD, raan = o.raan_deg * RAD;
  const cu = Math.cos(u), su = Math.sin(u);
  return [
    o.a * (Math.cos(raan) * cu - Math.sin(raan) * su * Math.cos(inc)),
    o.a * (Math.sin(raan) * cu + Math.cos(raan) * su * Math.cos(inc)),
    o.a * (su * Math.sin(inc)),
  ];
}
/* Inertial position [m] of a ground point (lat, lon in deg) at time t: Greenwich is on the inertial x
   axis at t = 0 and the Earth turns about z at OMEGA_EARTH. */
function groundECI(lat_deg, lon_deg, t) {
  const lat = lat_deg * RAD, th = lon_deg * RAD + OMEGA_EARTH * t;
  return [R_EARTH * Math.cos(lat) * Math.cos(th), R_EARTH * Math.cos(lat) * Math.sin(th), R_EARTH * Math.sin(lat)];
}
/* Sub-satellite point: geocentric latitude and Earth-fixed longitude in degrees, longitude in (-180, 180]. */
function subPoint(r, t) {
  const lat = Math.asin(r[2] / norm(r)) * DEG;
  let lon = (Math.atan2(r[1], r[0]) - OMEGA_EARTH * t) * DEG;
  lon = ((lon + 180) % 360 + 360) % 360 - 180;
  return { lat, lon };
}
/* Elevation of the satellite above the target's horizon and off-nadir angle of the target seen from the
   satellite, both in degrees, and the slant range in metres (visibility_mask, for one pair). */
function lookAngles(sat, ground) {
  const los = sub(ground, sat), rng = norm(los);
  const up = unit(ground), nadir = scale(sat, -1 / norm(sat));
  const sinElev = -dot(los, up) / rng, cosOff = dot(los, nadir) / rng;
  return { elevation: Math.asin(Math.max(-1, Math.min(1, sinElev))) * DEG, offNadir: Math.acos(Math.max(-1, Math.min(1, cosOff))) * DEG, range: rng };
}
/* visibility_mask for one pair: inside the sensor cone and above the elevation mask.
   maxOffNadir = null disables the cone test (ground-station contacts). */
function visible(sat, ground, maxOffNadir, minElevation) {
  const los = sub(ground, sat), rng = norm(los);
  const up = unit(ground);
  let ok = -dot(los, up) / rng >= Math.sin(minElevation * RAD);
  if (maxOffNadir != null) ok = ok && dot(los, scale(sat, -1 / norm(sat))) / rng >= Math.cos(maxOffNadir * RAD);
  return ok;
}
/* Index ranges [i0, i1] (inclusive) of consecutive true values (boolean_runs). */
function runs(mask) {
  const out = []; let start = -1;
  for (let i = 0; i <= mask.length; i++) {
    const v = i < mask.length && mask[i];
    if (v && start < 0) start = i;
    if (!v && start >= 0) { out.push([start, i - 1]); start = -1; }
  }
  return out;
}
/* Visibility windows [start, end] in seconds of one ground point from one orbit on a time grid of `step`
   seconds over [0, horizon], as the scenario generator computes them. */
function windows(o, lat, lon, horizon, step, maxOffNadir, minElevation) {
  const times = [], mask = [];
  for (let t = 0; t <= horizon + 1e-9; t += step) { times.push(t); mask.push(visible(positionECI(o, t), groundECI(lat, lon, t), maxOffNadir, minElevation)); }
  return runs(mask).map(([i0, i1]) => [times[i0], times[i1]]);
}
/* Angle between two unit vectors, in degrees (pointing_angle_deg). */
function angleDeg(u, v) { return Math.acos(Math.max(-1, Math.min(1, dot(u, v)))) * DEG; }

/* ---------------- planning instance (instance.py) ---------------- */
/* An instance object, as exported by code/export_instance.py:
   { settle, slewRate, dutyBudget, memoryBudget, nSats, opps: [{idx, sat, target, ws, we, dur, w, u, rev, mem}], ... }.
   The latest start of an opportunity is we - dur. */
const latest = o => o.we - o.dur;
/* Reconfiguration time (integer seconds) from opportunity a to b: settle + angle / slew rate, rounded up. */
function transition(inst, a, b) {
  const oa = inst.opps[a], ob = inst.opps[b];
  return Math.ceil(inst.settle + angleDeg(oa.u, ob.u) / inst.slewRate);
}
const maxTransition = inst => Math.ceil(inst.settle + 180 / inst.slewRate);
/* The duty-cycle (per satellite and revolution) and memory (per satellite and memory segment) groups whose
   opportunities could exceed the limit if all were chosen (budget_groups). */
function budgetGroups(inst) {
  const duty = new Map(), mem = new Map();
  for (const o of inst.opps) {
    const kd = o.sat + ":" + o.rev, km = o.sat + ":" + o.mem;
    if (!duty.has(kd)) duty.set(kd, []); duty.get(kd).push(o.idx);
    if (!mem.has(km)) mem.set(km, []); mem.get(km).push(o.idx);
  }
  const groups = [];
  for (const [key, ids] of duty) groups.push({ kind: "duty", key, limit: inst.dutyBudget, ids });
  for (const [key, ids] of mem) groups.push({ kind: "memory", key, limit: inst.memoryBudget, ids });
  return groups.filter(g => g.ids.reduce((s, i) => s + inst.opps[i].dur, 0) > g.limit);
}
/* Interfering pairs on the same satellite, by a sweep over windows sorted by start (build_pairs).
   aFirst / bFirst: whether "a then b" / "b then a" can fit at all. */
function conflictPairs(inst) {
  const sMax = maxTransition(inst), bySat = new Map(), pairs = [];
  for (const o of inst.opps) { if (!bySat.has(o.sat)) bySat.set(o.sat, []); bySat.get(o.sat).push(o.idx); }
  for (const ids0 of bySat.values()) {
    const ids = ids0.slice().sort((i, j) => inst.opps[i].ws - inst.opps[j].ws);
    for (let p = 0; p < ids.length; p++) {
      const a = ids[p], oa = inst.opps[a];
      for (let q = p + 1; q < ids.length; q++) {
        const b = ids[q], ob = inst.opps[b];
        if (ob.ws >= oa.we + sMax) break;
        if (oa.target === ob.target) continue;
        const sab = transition(inst, a, b), sba = transition(inst, b, a);
        if (oa.we + sab <= ob.ws) continue;
        if (ob.we + sba <= oa.ws) continue;
        const aFirst = oa.ws + oa.dur + sab <= latest(ob), bFirst = ob.ws + ob.dur + sba <= latest(oa);
        pairs.push({ a, b, sab, sba, aFirst, bFirst });
      }
    }
  }
  return pairs;
}

/* ---------------- independent checker (checker.py) ---------------- */
/* plan: Map or object {opportunity id: start time}. Returns a list of violation messages (empty = feasible). */
function checkPlan(inst, plan) {
  const P = plan instanceof Map ? plan : new Map(Object.entries(plan).map(([k, v]) => [+k, v]));
  const TOL = 1e-6, errors = [], seen = new Map(), perSat = new Map();
  for (const [i, t] of P) {
    const o = inst.opps[i];
    if (t < o.ws - TOL || t > latest(o) + TOL) errors.push(`opportunity ${i}: start ${t} outside [${o.ws}, ${latest(o)}]`);
    if (seen.has(o.target)) errors.push(`target ${o.target} collected twice (opportunities ${seen.get(o.target)} and ${i})`);
    seen.set(o.target, i);
    if (!perSat.has(o.sat)) perSat.set(o.sat, []); perSat.get(o.sat).push(i);
  }
  for (const [s, ids] of perSat) {
    ids.sort((a, b) => P.get(a) - P.get(b));
    for (let p = 0; p < ids.length; p++) for (let q = p + 1; q < ids.length; q++) {
      const a = ids[p], b = ids[q], need = inst.opps[a].dur + transition(inst, a, b), have = P.get(b) - P.get(a);
      if (have < need - TOL) errors.push(`satellite ${s}: opportunities ${a} then ${b} need ${need} s between starts, have ${have.toFixed(1)} s`);
    }
  }
  for (const g of budgetGroups(inst)) {
    const used = g.ids.reduce((s, i) => s + (P.has(i) ? inst.opps[i].dur : 0), 0);
    if (used > g.limit + TOL) errors.push(`${g.kind} budget exceeded on satellite ${g.key.split(":")[0]}: ${used} s > ${g.limit} s`);
  }
  return errors;
}
const planValue = (inst, plan) => [...(plan instanceof Map ? plan.keys() : Object.keys(plan))].reduce((s, i) => s + inst.opps[+i].w, 0);

/* ---------------- greedy baseline (greedy.py) ---------------- */
/* Targets by value density (weight / duration, ties: fewer opportunities first), each placed on its earliest
   opportunity that fits, at the earliest feasible start among the window start and the ends-plus-transition of
   tasks already on that satellite. Returns { plan: Map, order, steps } where steps records every decision. */
function greedy(inst) {
  const byTarget = new Map();
  for (const o of inst.opps) { if (!byTarget.has(o.target)) byTarget.set(o.target, []); byTarget.get(o.target).push(o.idx); }
  const used = new Map(), limits = { duty: inst.dutyBudget, memory: inst.memoryBudget };
  const groups = o => [["duty", o.sat + ":" + o.rev], ["memory", o.sat + ":" + o.mem]];
  const schedule = new Map(), plan = new Map(), steps = [];
  for (let s = 0; s < inst.nSats; s++) schedule.set(s, []);
  const fits = (o, t) => {
    for (const [tp, p] of schedule.get(o.sat)) {
      if (t >= tp) { if (t < tp + inst.opps[p].dur + transition(inst, p, o.idx)) return false; }
      else if (t + o.dur + transition(inst, o.idx, p) > tp) return false;
    }
    return true;
  };
  const W = inst.targetWeights, Dur = inst.targetDurations;
  // Python's sorted() is stable, and dict keys keep insertion order (the order targets first appear).
  const order = [...byTarget.keys()].map((j, k) => [j, k]).sort((A, B) => {
    const da = -W[A[0]] / Dur[A[0]], db = -W[B[0]] / Dur[B[0]];
    if (da !== db) return da - db;
    const la = byTarget.get(A[0]).length, lb = byTarget.get(B[0]).length;
    return la !== lb ? la - lb : A[1] - B[1];
  }).map(x => x[0]);
  for (const j of order) {
    let placed = null, reason = "no opportunity fits";
    for (const i of byTarget.get(j).slice().sort((a, b) => inst.opps[a].ws - inst.opps[b].ws)) {
      const o = inst.opps[i];
      if (groups(o).some(([k, g]) => (used.get(k + g) || 0) + o.dur > limits[k])) { reason = "budget full"; continue; }
      const cands = [o.ws, ...schedule.get(o.sat).map(([tp, p]) => tp + inst.opps[p].dur + transition(inst, p, i))]
        .filter(c => c >= o.ws && c <= latest(o)).sort((x, y) => x - y);
      const t = cands.find(c => fits(o, c));
      if (t === undefined) { reason = "blocked by tasks already placed"; continue; }
      plan.set(i, t); schedule.get(o.sat).push([t, i]);
      for (const [k, g] of groups(o)) used.set(k + g, (used.get(k + g) || 0) + o.dur);
      placed = i; break;
    }
    steps.push({ target: j, opp: placed, start: placed == null ? null : plan.get(placed), reason: placed == null ? reason : "placed" });
  }
  return { plan, order, steps, value: planValue(inst, plan) };
}

/* ---------------- a seeded random generator, for demos ---------------- */
function rng(seed) {
  let s0 = (seed >>> 0) || 1, s1 = 0x9e3779b9 ^ s0;
  const next = () => { let x = s0, y = s1; s0 = y; x ^= x << 23; x ^= x >>> 17; x ^= y ^ (y >>> 26); s1 = x; return ((s0 + s1) >>> 0) / 4294967296; };
  return { next, uniform: (lo, hi) => lo + (hi - lo) * next() };
}

/* ---------------- drawing helpers shared by the map figures ---------------- */
/* Equirectangular map projection inside a box: lon -180..180, lat -90..90. */
function mapProjection(x0, y0, w, h) {
  return { X: lon => x0 + (lon + 180) / 360 * w, Y: lat => y0 + (90 - lat) / 180 * h };
}
/* Split a ground track into pieces at the date line so a map line does not cross the whole figure. */
function splitAtDateLine(points) {
  const out = []; let cur = [];
  for (let k = 0; k < points.length; k++) {
    if (k && Math.abs(points[k].lon - points[k - 1].lon) > 180) { out.push(cur); cur = []; }
    cur.push(points[k]);
  }
  if (cur.length) out.push(cur);
  return out;
}

global.Sat = {
  MU_EARTH, R_EARTH, OMEGA_EARTH, RAD, DEG,
  dot, norm, sub, scale, unit,
  orbit, positionECI, groundECI, subPoint, lookAngles, visible, runs, windows, angleDeg,
  latest, transition, maxTransition, budgetGroups, conflictPairs, checkPlan, planValue, greedy,
  rng, mapProjection, splitAtDateLine,
};
})(window);
