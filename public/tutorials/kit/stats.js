/* Small statistics library for the interactive tutorial pages: densities and
   mass functions, random samplers and numerical helpers. Parameterisations follow
   the Bayesian course: gamma and exponential use the RATE, the normal its standard
   deviation. No DOM code. */
(function (global) {
"use strict";

/* ---------------- special functions ---------------- */
const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
/* log of the gamma function (Lanczos approximation, about 15 digits) */
function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
const lchoose = (n, k) => lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
/* error function (Abramowitz and Stegun 7.1.26, error below 1.5e-7) */
function erf(x) {
  const s = Math.sign(x);
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

/* ---------------- densities (continuous) ---------------- */
const SQRT2PI = Math.sqrt(2 * Math.PI);
const pdf = {
  normal: (x, mu = 0, sd = 1) => Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * SQRT2PI),
  uniform: (x, a = 0, b = 1) => (x >= a && x <= b ? 1 / (b - a) : 0),
  exponential: (x, rate = 1) => (x < 0 ? 0 : rate * Math.exp(-rate * x)),
  gamma: (x, shape, rate) => (x <= 0 ? 0 : Math.exp(shape * Math.log(rate) - lgamma(shape) + (shape - 1) * Math.log(x) - rate * x)),
  beta: (x, a, b) => (x <= 0 || x >= 1 ? 0
    : Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x))),
  t: (x, nu) => Math.exp(lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * Math.log(nu * Math.PI)
    - (nu + 1) / 2 * Math.log(1 + x * x / nu)),
};
const cdf = {
  normal: (x, mu = 0, sd = 1) => 0.5 * (1 + erf((x - mu) / (sd * Math.SQRT2))),
  exponential: (x, rate = 1) => (x < 0 ? 0 : 1 - Math.exp(-rate * x)),
  uniform: (x, a = 0, b = 1) => Math.min(1, Math.max(0, (x - a) / (b - a))),
};

/* ---------------- mass functions (discrete) ---------------- */
const pmf = {
  bernoulli: (k, p) => (k === 1 ? p : k === 0 ? 1 - p : 0),
  binomial: (k, n, p) => (k < 0 || k > n ? 0 : p === 0 ? +(k === 0) : p === 1 ? +(k === n)
    : Math.exp(lchoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p))),
  geometric: (k, p) => (k < 1 ? 0 : (1 - p) ** (k - 1) * p),          // trials until the first success
  poisson: (k, lam) => (k < 0 ? 0 : Math.exp(k * Math.log(lam) - lam - lgamma(k + 1))),
};

/* ---------------- random numbers ---------------- */
/* Seedable uniform generator (mulberry32). makeRng() without a seed uses Math.random. */
function makeRng(seed) {
  let uniform;
  if (seed == null) uniform = Math.random;
  else {
    let a = (seed >>> 0) || 1;
    uniform = () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  let spare = null;
  const r = {
    uniform,
    normal(mu = 0, sd = 1) {                         // Box-Muller, two values per pair of uniforms
      if (spare !== null) { const s = spare; spare = null; return mu + sd * s; }
      let u = 0;
      while (u === 0) u = uniform();
      const rad = Math.sqrt(-2 * Math.log(u)), ang = 2 * Math.PI * uniform();
      spare = rad * Math.sin(ang);
      return mu + sd * rad * Math.cos(ang);
    },
    exponential: (rate = 1) => -Math.log(1 - uniform()) / rate,
    gamma(shape, rate = 1) {                         // Marsaglia and Tsang
      if (shape < 1) return r.gamma(shape + 1, rate) * uniform() ** (1 / shape);
      const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) {
        let x, v;
        do { x = r.normal(); v = 1 + c * x; } while (v <= 0);
        v = v * v * v;
        const u = uniform();
        if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v / rate;
      }
    },
    beta(a, b) { const x = r.gamma(a), y = r.gamma(b); return x / (x + y); },
    bernoulli: p => (uniform() < p ? 1 : 0),
    binomial(n, p) { let k = 0; for (let i = 0; i < n; i++) if (uniform() < p) k++; return k; },
    poisson(lam) {                                   // Knuth; fine for the small rates used here
      const L = Math.exp(-lam);
      let k = 0, prod = uniform();
      while (prod > L) { k++; prod *= uniform(); }
      return k;
    },
    integer: (lo, hi) => lo + Math.floor(uniform() * (hi - lo + 1)),
  };
  return r;
}

/* ---------------- numerical helpers ---------------- */
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
/* Simpson's rule on [a, b] with n (even) intervals */
function integrate(f, a, b, n = 2000) {
  if (n % 2) n++;
  const h = (b - a) / n;
  let s = f(a) + f(b);
  for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * f(a + i * h);
  return s * h / 3;
}
/* Running integral of f on the grid xs (trapezoid rule): a numerical CDF. */
function cumulative(f, xs) {
  const out = [0];
  for (let i = 1; i < xs.length; i++) out.push(out[i - 1] + 0.5 * (f(xs[i]) + f(xs[i - 1])) * (xs[i] - xs[i - 1]));
  return out;
}
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
function variance(a) {                              // population variance, as numpy's .var()
  const m = mean(a);
  return a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length;
}

global.Stats = { lgamma, lchoose, erf, pdf, cdf, pmf, makeRng, linspace, integrate, cumulative, mean, variance };
})(window);
