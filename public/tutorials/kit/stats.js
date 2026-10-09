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
/* error function (Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7): fine for
   probabilities above about 1e-5, too coarse for far tails such as cdf.normal(-5) */
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
/* Student t: CDF and quantile, by integrating the density from 0 (it is symmetric). */
function tDensity(nu) {
  const c = Math.exp(lgamma((nu + 1) / 2) - lgamma(nu / 2)) / Math.sqrt(nu * Math.PI);
  return x => c * (1 + x * x / nu) ** (-(nu + 1) / 2);
}
function tCdf(x, nu) {
  const half = integrate(tDensity(nu), 0, Math.abs(x), 1200);
  return x >= 0 ? 0.5 + half : 0.5 - half;
}
function tQuantile(p, nu) {
  const target = Math.abs(p - 0.5), f = tDensity(nu);
  let lo = 0, hi = 400;
  for (let k = 0; k < 45; k++) { const mid = (lo + hi) / 2; if (integrate(f, 0, mid, 1200) < target) lo = mid; else hi = mid; }
  return p < 0.5 ? -lo : lo;
}
/* Quantiles of a density on [lo, hi] (normalised or not), from its numerical CDF on N grid
   points, interpolated between them. */
function densityQuantiles(dens, lo, hi, ps, N = 4001) {
  const xs = linspace(lo, hi, N), F = cumulative(dens, xs), tot = F[N - 1];
  return ps.map(p => {
    const target = p * tot, i = F.findIndex(v => v >= target);
    if (i < 0) return xs[N - 1];
    if (i === 0) return xs[0];
    return xs[i - 1] + (xs[i] - xs[i - 1]) * (target - F[i - 1]) / (F[i] - F[i - 1]);
  });
}

/* ---------------- samples ---------------- */
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
function variance(a) {                              // population variance, as numpy's .var()
  const m = mean(a);
  return a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length;
}
/* Sample quantiles as numpy computes them by default (linear interpolation).
   quantileSorted needs sorted values; quantile sorts a copy first. */
const sortedCopy = a => Float64Array.from(a).sort();
function quantileSorted(sorted, p) {
  const pos = p * (sorted.length - 1), lo = Math.floor(pos), hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
}
const quantile = (values, p) => quantileSorted(sortedCopy(values), p);

/* ---------------- Markov chain Monte Carlo diagnostics ---------------- */
/* In-place radix-2 FFT (length a power of 2), with the cos/sin tables kept per length. */
const twiddles = {};
function fft(re, im, inverse) {
  const n = re.length;
  if (!twiddles[n]) {
    const c = new Float64Array(n / 2), s = new Float64Array(n / 2);
    for (let k = 0; k < n / 2; k++) { c[k] = Math.cos(2 * Math.PI * k / n); s[k] = Math.sin(2 * Math.PI * k / n); }
    twiddles[n] = [c, s];
  }
  const [cos, sin] = twiddles[n], sign = inverse ? 1 : -1;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, stride = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const c = cos[k * stride], s = sign * sin[k * stride];
        const p = i + k, q = p + half;
        const br = re[q] * c - im[q] * s, bi = re[q] * s + im[q] * c;
        re[q] = re[p] - br; im[q] = im[p] - bi; re[p] += br; im[p] += bi;
      }
    }
  }
}
/* Effective sample size of one chain, as the Bayesian course's ess(): autocorrelations
   (computed by FFT) summed until the first negative one. */
function ess(x) {
  const n = x.length;
  if (n < 4) return n;
  let m = 0;
  for (let i = 0; i < n; i++) m += x[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - m) ** 2;
  v /= n;
  if (!(v > 0)) return 1;                       // the chain never moved
  let size = 1;
  while (size < 2 * n) size <<= 1;
  const re = new Float64Array(size), im = new Float64Array(size);
  for (let i = 0; i < n; i++) re[i] = x[i] - m;
  fft(re, im, false);
  for (let i = 0; i < size; i++) { re[i] = re[i] * re[i] + im[i] * im[i]; im[i] = 0; }
  fft(re, im, true);
  let sum = 0;
  for (let k = 1; k < n; k++) {
    const r = re[k] / size / (v * n);
    if (r < 0) break;
    sum += r;
  }
  return n / (1 + 2 * sum);
}
/* R-hat (Gelman and Rubin): between-chain against within-chain variance; chains of equal length. */
function rhat(chains) {
  const k = chains.length, n = chains[0].length;
  const means = chains.map(c => c.reduce((s, v) => s + v, 0) / n);
  const grand = means.reduce((s, v) => s + v, 0) / k;
  const B = n * means.reduce((s, v) => s + (v - grand) ** 2, 0) / (k - 1);
  const W = chains.reduce((s, c, j) => s + c.reduce((t, v) => t + (v - means[j]) ** 2, 0) / (n - 1), 0) / k;
  return Math.sqrt(((n - 1) / n * W + B / n) / W);
}

global.Stats = { lgamma, lchoose, erf, pdf, cdf, pmf, makeRng, linspace, integrate, cumulative, tCdf, tQuantile,
  densityQuantiles, mean, variance, sortedCopy, quantileSorted, quantile, ess, rhat };
})(window);
