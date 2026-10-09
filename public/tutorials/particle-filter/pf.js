/* Particle-filter core for the tutorial: terrain, robot and filter.
   No DOM code here; app.js does the drawing. The algorithm mirrors
   particle_filter.py line for line. */
(function (global) {
"use strict";

/* ---------------- random numbers (seedable, so a map can be re-created) ---------------- */
function makeRng(seed) {
  let a = (seed >>> 0) || 1;
  let spare = null;
  function uniform() {                      // mulberry32: fast, good enough for a demo
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function normal(sigma = 1) {              // Box-Muller, two draws per call pair
    if (spare !== null) { const s = spare; spare = null; return s * sigma; }
    let u = 0;
    while (u === 0) u = uniform();
    const r = Math.sqrt(-2 * Math.log(u)), v = 2 * Math.PI * uniform();
    spare = r * Math.sin(v);
    return r * Math.cos(v) * sigma;
  }
  return { uniform, normal };
}

/* ---------------- the world: random hills ---------------- */
const WIDTH = 240, HEIGHT = 160;            // metres; one grid cell = 1 m

/* Smooth noise at several scales, added together, scaled to [base, base + relief]. */
function makeTerrain({ seed = 7, relief = 100, octaves = 5, falloff = 0.55 } = {}) {
  const rng = makeRng(seed * 7919 + 13);
  const z = new Float32Array(WIDTH * HEIGHT);
  for (let o = 0; o < octaves; o++) {
    const cells = 3 * 2 ** o, gw = cells + 2;
    const grid = new Float32Array(gw * gw);
    for (let i = 0; i < grid.length; i++) grid[i] = rng.uniform() * 2 - 1;
    const amp = falloff ** o;
    for (let y = 0; y < HEIGHT; y++) {
      const gy = y * cells / WIDTH, iy = Math.floor(gy);
      let fy = gy - iy; fy = fy * fy * (3 - 2 * fy);
      for (let x = 0; x < WIDTH; x++) {
        const gx = x * cells / WIDTH, ix = Math.floor(gx);
        let fx = gx - ix; fx = fx * fx * (3 - 2 * fx);
        const top = grid[iy * gw + ix] * (1 - fx) + grid[iy * gw + ix + 1] * fx;
        const bot = grid[(iy + 1) * gw + ix] * (1 - fx) + grid[(iy + 1) * gw + ix + 1] * fx;
        z[y * WIDTH + x] += amp * (top * (1 - fy) + bot * fy);
      }
    }
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of z) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const base = 50 - relief / 2;
  for (let i = 0; i < z.length; i++) z[i] = base + relief * (z[i] - lo) / (hi - lo);
  return { w: WIDTH, h: HEIGHT, z, seed, relief };
}

/* Ground height at the nearest grid cell (positions are clamped to the map). */
function heightAt(t, x, y) {
  const c = Math.min(t.w - 1, Math.max(0, Math.round(x)));
  const r = Math.min(t.h - 1, Math.max(0, Math.round(y)));
  return t.z[r * t.w + c];
}

/* Smooth (bilinear) height, used only for drawing. */
function heightSmooth(t, x, y) {
  x = Math.min(t.w - 1.0001, Math.max(0, x));
  y = Math.min(t.h - 1.0001, Math.max(0, y));
  const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy, w = t.w;
  const a = t.z[iy * w + ix], b = t.z[iy * w + ix + 1];
  const c = t.z[(iy + 1) * w + ix], d = t.z[(iy + 1) * w + ix + 1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

/* ---------------- noise levels (same names and values as the Python program) ---------------- */
const NOISE = {
  step: 0.10,       // wheels get each distance wrong by about 10 %
  turn: 0.10,       // ... and each turn by about 10 %
  drift: 0.015,     // driving wobbles the heading (radians per metre)
  compass: 0.03,    // compass error, radians (about 2 degrees)
  sensor: 1.5,      // altimeter error, metres
};

const clampX = x => Math.min(WIDTH - 1, Math.max(0, x));
const clampY = y => Math.min(HEIGHT - 1, Math.max(0, y));
const wrapAngle = a => Math.atan2(Math.sin(a), Math.cos(a));

/* ---------------- the real robot ---------------- */
class Robot {
  constructor(terrain, rng, x, y, heading) {
    this.t = terrain; this.rng = rng;
    this.x = x; this.y = y; this.heading = heading;
  }
  step(distance, turn) {
    const r = this.rng;
    this.heading = wrapAngle(this.heading + turn +
      r.normal(NOISE.turn * Math.abs(turn) + NOISE.drift * distance));
    const travelled = distance + r.normal(NOISE.step * distance);
    this.x = clampX(this.x + travelled * Math.cos(this.heading));   // a fence keeps the
    this.y = clampY(this.y + travelled * Math.sin(this.heading));   // robot on the map
  }
  readCompass() { return this.heading + this.rng.normal(NOISE.compass); }
  readAltimeter() { return heightAt(this.t, this.x, this.y) + this.rng.normal(NOISE.sensor); }
}

/* ---------------- the particle filter ---------------- */
class ParticleFilter {
  constructor(terrain, rng, opts = {}) {
    this.t = terrain; this.rng = rng;
    this.n = opts.n ?? 2000;
    this.assumedNoise = opts.assumedNoise ?? 2.5;   // a slightly worse altimeter than the real one
    this.jitter = opts.jitter ?? 0.5;               // metres of shake after resampling
    this.headingJitter = opts.headingJitter ?? 0.03;
    this.compass = opts.compass ?? true;            // false: every guess must also guess the heading
    this.explorers = opts.explorers ?? 0;           // fraction of guesses re-scattered at random
    this.reset();
  }
  reset() {
    const n = this.n, r = this.rng;
    this.px = new Float64Array(n); this.py = new Float64Array(n); this.ph = new Float64Array(n);
    this.w = new Float64Array(n).fill(1 / n);
    this.lastWeights = null;
    for (let i = 0; i < n; i++) this.scatter(i);
    this.steps = 0;
  }
  scatter(i) {
    const r = this.rng;
    this.px[i] = r.uniform() * WIDTH;
    this.py[i] = r.uniform() * HEIGHT;
    this.ph[i] = r.uniform() * 2 * Math.PI;
  }
  /* 1. Predict: move every guess the way the robot thinks it moved, plus random errors. */
  predict(distance, turn, compass) {
    const { n, px, py, ph, rng: r } = this;
    for (let i = 0; i < n; i++) {
      let heading;
      if (this.compass) {
        heading = compass + r.normal(NOISE.compass + NOISE.drift * distance);
      } else {
        ph[i] += turn + r.normal(NOISE.turn * Math.abs(turn) + NOISE.drift * distance);
        heading = ph[i];
      }
      const travelled = distance + r.normal(NOISE.step * distance);
      px[i] = clampX(px[i] + travelled * Math.cos(heading));
      py[i] = clampY(py[i] + travelled * Math.sin(heading));
    }
  }
  /* 2. Weigh: bell-curve score of how well each guess explains the altimeter reading. */
  weigh(reading) {
    const { n, px, py, w, t } = this, s = this.assumedNoise;
    let minSurprise = Infinity;
    for (let i = 0; i < n; i++) {
      const e = (reading - heightAt(t, px[i], py[i])) / s;
      w[i] = e * e;
      if (w[i] < minSurprise) minSurprise = w[i];
    }
    let sum = 0;
    for (let i = 0; i < n; i++) { w[i] = Math.exp(-0.5 * (w[i] - minSurprise)); sum += w[i]; }
    for (let i = 0; i < n; i++) w[i] /= sum;
    this.lastWeights = Float64Array.from(w);
  }
  /* 3. Resample: copy guesses in proportion to their weight (systematic, "comb" resampling). */
  resample() {
    const { n, w, rng: r } = this;
    const nx = new Float64Array(n), ny = new Float64Array(n), nh = new Float64Array(n);
    const start = r.uniform() / n;
    let i = 0, cum = w[0];
    for (let k = 0; k < n; k++) {
      const tooth = start + k / n;
      while (tooth > cum && i < n - 1) { i++; cum += w[i]; }
      nx[k] = this.px[i]; ny[k] = this.py[i]; nh[k] = this.ph[i];
    }
    this.px = nx; this.py = ny; this.ph = nh;
    this.w.fill(1 / n);
  }
  /* 4. Jitter: a small shake so copies do not sit on top of each other. */
  shake() {
    const { n, px, py, ph, rng: r } = this;
    for (let i = 0; i < n; i++) {
      px[i] += r.normal(this.jitter);
      py[i] += r.normal(this.jitter);
      ph[i] += r.normal(this.headingJitter);
    }
  }
  /* Optional: throw a few guesses to random places ("explorers"), before they are
     weighed, so that the ones that do not fit are dropped again straight away. */
  explore() {
    for (let i = 0; i < this.n; i++) if (this.rng.uniform() < this.explorers) this.scatter(i);
  }
  step(distance, turn, compass, reading) {
    if (this.explorers > 0) this.explore();
    this.predict(distance, turn, compass);
    this.weigh(reading);
    this.resample();
    this.shake();
    this.steps++;
  }
  /* Best single guess = average position; spread = how far guesses sit from it. */
  estimate() {
    const { n, px, py, ph } = this;
    let mx = 0, my = 0, sx = 0, sy = 0;
    for (let i = 0; i < n; i++) { mx += px[i]; my += py[i]; sx += Math.sin(ph[i]); sy += Math.cos(ph[i]); }
    mx /= n; my /= n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (px[i] - mx) ** 2 + (py[i] - my) ** 2;
    return { x: mx, y: my, heading: Math.atan2(sx, sy), spread: Math.sqrt(v / n) };
  }
}

global.PF = { makeRng, makeTerrain, heightAt, heightSmooth, Robot, ParticleFilter, NOISE, WIDTH, HEIGHT, wrapAngle };
})(window);
