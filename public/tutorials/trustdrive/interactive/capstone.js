/* TrustDrive capstone: the course library (courselib) ported to the browser, step for step.
   Vehicle, speed-scheduled LQR, synthetic perception, Kalman monitor, operating domain,
   integrity score, shared controller, simulated driver, the four scenarios, the three
   methods and the metrics. Exposes window.Capstone. Needs ../kit/stats.js (makeRng). */
(global => {
"use strict";
const L = 2.7, TS = 0.05, V = 15, DMAX = 0.6, CHI2 = 9.210;
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const sign = x => (x > 0 ? 1 : x < 0 ? -1 : 0);

/* ---------------- LQR, scheduled on the speed ---------------- */
function dlqr(v) {                                   // Q = diag(1, 5), R = 8; Riccati iteration as the library does
  const a = v * TS, b = v * TS / L, q0 = 1, q1 = 5, r = 8;
  let p00 = q0, p01 = 0, p11 = q1, k0 = 0, k1 = 0;
  for (let it = 0; it < 600; it++) {
    // A = [[1, a],[0, 1]], B = [0, b]:  P B = [p01 b, p11 b],  B^T P B = p11 b^2
    const g = r + p11 * b * b, u0 = p01 * b, u1 = (a * p01 + p11) * b;    // A^T P B
    const n00 = q0 + p00 - u0 * u0 / g;
    const n01 = a * p00 + p01 - u0 * u1 / g;
    const n11 = q1 + a * a * p00 + 2 * a * p01 + p11 - u1 * u1 / g;
    const done = Math.max(Math.abs(n00 - p00), Math.abs(n01 - p01), Math.abs(n11 - p11)) < 1e-12;
    p00 = n00; p01 = n01; p11 = n11;
    if (done) break;
  }
  const g = r + p11 * b * b;
  k0 = (p01 * b) / g; k1 = ((a * p01 + p11) * b) / g;                       // (R + B^T P B)^-1 B^T P A
  return [k0, k1];
}
/* the library caches gains by round(v, 1), with Python's ties-to-even */
const gainCache = new Map();
function roundTenth(v) {
  const x = v * 10, f = Math.floor(x), d = x - f;
  return (Math.abs(d - 0.5) < 1e-9 ? (f % 2 === 0 ? f : f + 1) : Math.round(x)) / 10;
}
function gainAt(v) {
  const key = roundTenth(v);
  if (!gainCache.has(key)) gainCache.set(key, dlqr(key));
  return gainCache.get(key);
}

/* ---------------- scenarios ---------------- */
const pulse = (t, t0, t1, lo, hi) => (t0 <= t && t < t1 ? hi : lo);
const ramp = (t, t0, t1, lo, hi) => (t <= t0 ? lo : t >= t1 ? hi : lo + (hi - lo) * (t - t0) / (t1 - t0));
function buildScenario(name, T = 30) {
  const n = Math.round(T / TS), frames = [];
  for (let k = 0; k < n; k++) {
    const t = k * TS;
    let kappa = 0.012 * Math.sin(2 * Math.PI * t / 20), blur = 0, brightness = 1, occlusion = 0, workload = 0.2, available = true, conflict = 0;
    if (name === "fog") {
      blur = t < 17 ? ramp(t, 10, 15, 0, 3) : ramp(t, 20, 25, 3, 0);
      brightness = 1 - 0.4 * (blur / 3);
      occlusion = pulse(t, 15, 20, 0, 0.45);
    } else if (name === "ood_curve") {
      kappa += pulse(t, 12, 22, 0, 0.04); workload = 0.3;
    } else if (name === "conflict") {
      workload = pulse(t, 14, 20, 0.2, 0.8); conflict = pulse(t, 14, 20, 0, 0.8); available = !(t >= 20 && t < 22);
    }
    frames.push({ kappa, blur, brightness, occlusion, workload, available, conflict });
  }
  return frames;
}

/* ---------------- perception, domain, integrity ---------------- */
const KMAX_TRAIN = 0.02, BASE = [0.02, 0.01];
function reportedStd(f) {
  const s = (1 + 2.5 * f.blur) * (1 + 1.5 * Math.max(0, 1 - f.brightness)) * (1 + 3 * f.occlusion) * (1 + 4 * Math.max(0, Math.abs(f.kappa) - KMAX_TRAIN));
  return [BASE[0] * s, BASE[1] * s];
}
const ODD = { kappa: 0.022, brightness: 0.45, blur: 1.6, occlusion: 0.35 };
function margins(kappa, brightness, blur, occlusion) {
  return [1 - Math.abs(kappa) / ODD.kappa, (brightness - ODD.brightness) / (1 - ODD.brightness), 1 - blur / ODD.blur, 1 - occlusion / ODD.occlusion];
}
const insideOdd = f => margins(f.kappa, f.brightness, f.blur, f.occlusion).every(m => m > 0);
const authority = (I, W, avail) => clip(avail ? I + 0.35 * W * (I - 0.5) : Math.max(I + 0.35 * W * (I - 0.5), 0.85), 0.05, 0.95);
const refSpeed = I => V * (0.4 + 0.6 * I);

/* ---------------- one closed-loop episode ----------------
   method: "naive" | "handover" | "proposed".  opt: seed, x0, handoverThr, and the ablation
   switches useNis, useOdd, filter, speed (speed false really keeps 15 m/s).
   rngP and rngD may be passed in (tests); otherwise seeded from seed and seed + 1. */
function runEpisode(method, frames, opt = {}) {
  const { seed = 0, x0 = [0.3, 0.02], handoverThr = 0.0045, useNis = true, useOdd = true, filter = true, speed = true } = opt;
  const rngP = opt.rngP || global.Stats.makeRng(1000 + 2 * seed), rngD = opt.rngD || global.Stats.makeRng(1001 + 2 * seed);
  const a = V * TS, b = V * TS / L, e = -V * TS;                             // the monitor's linear model, at 15 m/s
  let x = x0.slice(), kx = x0.slice(), p00 = 0.05, p01 = 0, p11 = 0.02, I = 1, prev = 0;
  const buf = Array.from({ length: 5 }, () => [0, 0]);                       // the driver's 0.2 s delay line, starting at zero
  const log = { t: [], ey: [], yey: [], nis: [], trace: [], su: [], sn: [], so: [], I: [], lam: [], v: [], delta: [], inside: [], kappa: [] };
  frames.forEach((f, k) => {
    const tru = x.slice();
    // perception: reported noise, true noise and bias out of distribution
    const srep = reportedStd(f), ood = Math.max(0, Math.abs(f.kappa) - KMAX_TRAIN) / KMAX_TRAIN;
    const bias = [0.18 * ood * (f.kappa !== 0 ? sign(f.kappa) : 1), 0.05 * ood];
    const y = [tru[0] + bias[0] + rngP.normal(0, srep[0] * (1 + 1.2 * ood)), tru[1] + bias[1] + rngP.normal(0, srep[1] * (1 + 1.2 * ood))];
    const r0 = srep[0] ** 2, r1 = srep[1] ** 2, trace = r0 + r1;
    // Kalman monitor: predict with the applied steering and the curvature, then update
    kx = [kx[0] + a * kx[1], kx[1] + b * prev + e * f.kappa];
    const q00 = p00 + 2 * a * p01 + a * a * p11 + 1e-4, q01 = p01 + a * p11, q11 = p11 + 1e-4;
    const s00 = q00 + r0, s11 = q11 + r1, det = s00 * s11 - q01 * q01, n0 = y[0] - kx[0], n1 = y[1] - kx[1];
    const nis = (s11 * n0 * n0 - 2 * q01 * n0 * n1 + s00 * n1 * n1) / det;
    const i00 = s11 / det, i01 = -q01 / det, i11 = s00 / det;
    const k00 = q00 * i00 + q01 * i01, k01 = q00 * i01 + q01 * i11, k10 = q01 * i00 + q11 * i01, k11 = q01 * i01 + q11 * i11;
    kx = [kx[0] + k00 * n0 + k01 * n1, kx[1] + k10 * n0 + k11 * n1];
    p00 = q00 - (k00 * q00 + k01 * q01); p01 = q01 - (k00 * q01 + k01 * q11); p11 = q11 - (k10 * q01 + k11 * q11);
    // integrity: uncertainty, consistency, domain (true conditions and the map's curvature)
    const inside = insideOdd(f);
    const m = useOdd ? margins(f.kappa, f.brightness, f.blur, f.occlusion) : margins(0, 1, 0, 0);
    const su = Math.exp(-trace / 0.02), sn = Math.exp(-Math.max(0, (useNis ? nis : 0) - CHI2) / 6), so = 1 / (1 + Math.exp(-6 * Math.min(...m)));
    I = 0.6 * I + 0.4 * su * sn * so;
    // automation: feed-forward plus LQR feedback on the estimate, at the commanded speed
    const proposed = method === "proposed";
    const xh = proposed && filter ? kx : y;
    const vcmd = proposed && speed ? refSpeed(I) : V, K = gainAt(vcmd);
    const ff = Math.atan(L * f.kappa), fb = -(K[0] * xh[0] + K[1] * xh[1]);
    const dAuto = clip(ff + fb, -DMAX, DMAX);
    // the driver: delayed, noisy, weaker under workload, possibly fighting, possibly absent
    buf.push(tru.slice()); buf.shift();
    let dH = -(0.12 * buf[0][0] + 0.9 * buf[0][1]);
    dH = (1 - 0.5 * f.workload) * dH + rngD.normal(0, 0.02 * (1 + 2 * f.workload));
    if (f.conflict > 0) dH -= f.conflict * 0.4 * (dH !== 0 ? sign(dH) : 1);
    if (!f.available) dH = 0;
    dH = clip(dH, -DMAX, DMAX);
    let lam, vRef, delta;
    if (method === "naive") { lam = 1; vRef = V; delta = dAuto; }
    else if (method === "handover") { const unsafe = trace > handoverThr; lam = unsafe ? 0 : 1; vRef = V; delta = unsafe ? dH : dAuto; }
    else { lam = authority(I, f.workload, f.available); vRef = speed ? refSpeed(I) : V; delta = clip(ff + lam * fb + (1 - lam) * dH, -DMAX, DMAX); }
    // the kinematic vehicle, one Euler step at the reference speed
    const d = clip(delta, -DMAX, DMAX);
    x = [tru[0] + vRef * Math.sin(tru[1]) * TS, tru[1] + (vRef / L * Math.tan(d) - vRef * f.kappa) * TS];
    prev = delta;
    const vals = { t: k * TS, ey: tru[0], yey: y[0], nis, trace, su, sn, so, I, lam, v: vRef, delta, inside, kappa: f.kappa };
    for (const key in vals) log[key].push(vals[key]);
  });
  return log;
}
function metrics(log, thr = 0.7) {
  const e = log.ey, n = e.length;
  let exits = 0, jerk = 0, unsafe = 0, sq = 0, mx = 0, ay = 0;
  for (let k = 0; k < n; k++) {
    const over = Math.abs(e[k]) > thr;
    if (over && (k === 0 || Math.abs(e[k - 1]) <= thr)) exits++;
    if (k) jerk += Math.abs(log.delta[k] - log.delta[k - 1]);
    if (log.lam[k] > 0.5 && !log.inside[k]) unsafe++;
    sq += e[k] * e[k]; mx = Math.max(mx, Math.abs(e[k])); ay = Math.max(ay, log.v[k] ** 2 * Math.abs(log.kappa[k]));
  }
  return { rmse: Math.sqrt(sq / n), max: mx, exits, jerk, unsafe: 100 * unsafe / n, ay };
}
global.Capstone = { TS, V, METHODS: ["naive", "handover", "proposed"], SCENARIOS: ["nominal", "fog", "ood_curve", "conflict"], buildScenario, runEpisode, metrics, insideOdd, dlqr };
})(window);
