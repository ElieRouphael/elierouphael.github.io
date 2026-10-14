/* A Tour of Machine Learning, chapter 06, A neural network from scratch: the live demos.
   The network, the loss, backpropagation and full-batch gradient descent are the notebook's,
   line for line, on the notebook's 300 training and 100 test points and its initial weights
   (default_rng(0)), so the default run reproduces the notebook exactly: loss 0.0674, test
   accuracy 0.960. Other widths use the browser's own seeded initial weights.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, pct, caption, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { makeRng } = window.Stats;

/* ---------------- the data: make_moons(400, noise=0.2, random_state=42), split 75/25 with random_state=42 ---------------- */
const XTR = [[-0.084842,0.797803],[-0.515977,1.162814],[0.570812,-0.861707],[1.62918,-0.459744],[-0.801588,0.797598],[0.059575,0.837532],[2.362384,0.363553],[0.533867,-0.613517],[1.6545,0.682001],[0.74193,0.124014],[-0.504363,0.596854],[1.930877,0.031738],[1.144658,0.743447],[0.191754,1.110474],[1.231142,-0.202874],[-0.581076,1.09252],[0.569874,-0.640534],[1.638271,-0.369449],[-0.692761,0.806866],[0.192267,-0.26437],[0.440774,0.959813],[2.085492,-0.164372],[1.850791,0.172668],[0.424206,-0.30908],[1.089258,0.227248],[0.940986,-0.111599],[0.284282,0.858624],[-0.981269,0.347699],[-0.726941,0.637682],[0.132086,1.216596],[-0.908014,0.788259],[0.802937,-0.302261],[0.198085,-0.631166],[0.311975,-0.53248],[0.805633,-0.454641],[1.208968,-0.45954],[0.01472,0.627197],[-0.705466,1.074221],[1.79792,-0.056989],[0.619766,0.066012],[0.124148,0.602481],[1.048169,-0.915139],[1.008456,0.133254],[-0.624504,0.163475],[1.731599,-0.517124],[0.354031,0.306698],[-1.205049,0.605444],[2.074257,0.428909],[0.301797,0.957388],[-0.690425,0.198639],[1.302522,-0.509157],[1.179688,-0.316215],[0.373866,0.532832],[1.35455,0.079678],[0.926052,0.100681],[2.047504,0.385741],[0.551244,1.023913],[0.830832,-0.039767],[0.805617,-0.548828],[2.166728,-0.30573],[-0.638228,0.233962],[1.559886,-0.427757],[0.041681,0.189327],[-0.52898,0.859366],[-1.006422,0.489406],[0.38012,0.913773],[-1.015835,-0.10962],[-0.925259,0.501998],[-0.081238,0.944715],[0.83377,0.215895],[-0.224016,0.996309],[0.026906,1.033578],[2.018135,0.244555],[1.747233,-0.08628],[-0.472408,1.037324],[0.75646,-0.641685],[2.021986,0.152946],[1.991736,0.03132],[1.484018,0.649693],[0.610497,-0.454671],[1.943889,-0.177644],[-1.303494,-0.17983],[0.917031,0.249814],[-0.251867,1.254181],[0.35357,-0.38793],[1.397587,-0.09494],[2.265578,0.495113],[1.073883,-0.546516],[1.586739,-0.26325],[1.749073,-0.259941],[-1.00401,0.20615],[0.594444,0.474873],[0.116481,-0.293107],[0.55355,-0.27679],[0.890244,0.755164],[0.042084,0.389413],[0.872305,-0.079043],[1.83898,-0.548454],[0.512244,0.100905],[2.025199,0.023624],[-0.770187,0.409322],[0.294473,0.189136],[0.159646,1.273566],[-0.434912,1.036029],[2.054644,0.034802],[0.277531,-0.145018],[-0.87129,0.49346],[1.879603,0.3844],[1.832761,0.213786],[-1.010227,0.383975],[0.390709,-0.00914],[0.577243,-0.251816],[1.177745,0.179],[0.415965,-0.392491],[-0.113398,-0.165883],[0.885129,-0.65634],[0.577552,0.555624],[0.158079,0.725668],[-0.521742,0.705621],[-1.294381,0.371878],[0.334956,0.596558],[1.838008,-0.035271],[0.641687,-0.533254],[0.161715,-0.039748],[1.115681,-0.005743],[0.913973,-0.598299],[1.582989,-0.111223],[2.103613,0.761597],[0.789058,0.585324],[0.850473,-0.588162],[1.052342,0.591421],[-0.289368,1.143745],[1.477722,-0.257628],[-0.360711,0.800297],[-1.061252,0.716345],[1.079836,-0.739727],[-0.272125,1.042091],[0.314441,-0.012325],[0.710161,-0.170979],[1.396347,-0.334857],[1.877043,-0.124906],[1.883709,-0.410684],[2.058342,0.369115],[1.885124,-0.025455],[0.175061,0.739233],[0.271132,-0.080113],[1.75828,-0.081564],[-0.963875,1.222082],[0.931915,-0.767015],[1.199359,0.032553],[-0.860247,0.183469],[0.801912,0.645131],[0.420433,0.786175],[0.842431,0.749021],[-0.03623,0.793532],[-1.13308,0.486949],[1.050652,0.696759],[0.298514,0.876444],[0.592801,0.818571],[1.295952,-0.374023],[1.850281,0.056454],[0.476491,1.158048],[-0.247399,1.065185],[0.464031,-0.415827],[0.248805,0.734774],[0.946526,-0.16162],[0.260253,0.200612],[1.035874,-0.843628],[-1.137935,0.81793],[1.078992,-0.603764],[1.350953,-0.426402],[2.224042,0.381133],[0.356103,0.211113],[0.236947,0.943392],[0.019231,1.165376],[0.875015,0.934791],[0.926661,0.04259],[1.035748,0.515088],[0.827756,0.764656],[0.146011,0.242608],[0.080996,0.897496],[0.787466,0.880668],[1.986059,0.48327],[0.150219,0.3321],[-1.070808,0.079228],[0.851048,0.622561],[1.286136,-0.538254],[0.408301,1.007924],[-0.51201,-0.131174],[0.182561,0.636219],[0.768414,-0.871389],[-1.130548,0.277499],[0.819197,-0.328926],[0.534369,-0.165714],[-0.8845,0.172055],[-0.290796,0.763916],[-0.03213,-0.137982],[0.861958,0.600398],[0.002178,0.535346],[-0.892,0.461095],[0.095183,0.882266],[1.266462,-0.598258],[1.516146,-0.075774],[-0.271998,0.260983],[0.239019,-0.561613],[-1.23084,-0.252966],[-1.120298,0.492299],[0.42176,-0.331942],[0.105833,1.106925],[0.775826,-0.284394],[-0.558735,0.708185],[-0.097921,1.026175],[1.653083,0.292259],[-1.214753,0.701123],[0.289562,-0.03184],[1.904589,0.255839],[1.415486,-0.51066],[1.459872,-0.222042],[0.073523,0.142428],[0.971864,0.334268],[-1.094228,0.491696],[1.272436,0.131748],[-1.143625,0.285252],[0.072845,0.184057],[0.098717,0.246392],[2.080055,0.606404],[-0.575915,0.709864],[0.179182,1.024065],[-0.245868,0.232786],[0.133288,0.171517],[0.857928,1.035303],[0.536554,-0.619687],[-0.906367,0.842951],[0.68963,-0.372643],[-1.091288,0.266633],[1.158571,0.520774],[0.347972,0.622144],[0.687466,-0.34925],[-1.205068,0.452894],[0.8999,0.037447],[-0.632109,0.942722],[0.106897,-0.085142],[1.456036,0.384641],[1.600046,0.035282],[0.366566,0.827086],[1.519376,-0.084212],[0.631897,0.612794],[-0.961671,0.670607],[-0.375178,1.056828],[0.183915,0.953026],[0.259366,0.18377],[0.140689,0.855453],[1.127693,-0.553601],[2.073911,0.121787],[0.101472,0.444071],[1.373719,0.269467],[-0.20864,1.083202],[-0.377505,0.483436],[-0.226427,0.75416],[0.025409,0.089749],[1.281917,-0.45694],[1.606792,0.131296],[1.110438,0.744948],[-0.583974,0.93447],[-0.363118,0.099554],[0.897383,-0.178485],[-0.404312,0.339688],[0.00763,0.163964],[1.15523,0.265221],[0.980673,0.179424],[1.427528,-0.434795],[1.960512,0.009157],[-0.848615,-0.005437],[1.20658,0.340877],[1.916037,0.1879],[2.328479,0.526913],[0.955822,-0.842404],[-0.403825,1.142304],[0.689593,-0.773623],[0.447248,1.059632],[0.074988,0.973244],[0.622467,0.995832],[2.002299,-0.009183],[1.340492,-0.469764],[1.617818,-0.197019],[1.194484,-0.277569],[0.273625,1.104694],[0.033226,-0.110539],[1.091801,-0.502475],[0.731212,0.648314],[1.081186,-0.497126],[-0.250061,1.197837],[-0.110142,0.593889],[0.461336,0.967211],[1.724209,-0.648807],[-0.691073,0.845914],[1.00559,0.204714],[0.796991,0.697257],[0.43954,-0.150276],[1.198193,0.354955]];
const YTR = [0,0,1,1,0,0,1,1,1,0,0,1,0,0,1,0,1,1,0,1,0,1,1,1,0,1,0,0,0,0,0,1,1,1,1,1,1,0,1,0,0,1,0,0,1,1,0,1,0,0,1,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,1,1,0,1,0,1,1,1,0,1,0,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,0,1,1,1,0,1,0,0,1,0,0,1,0,1,1,1,1,1,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,0,0,0,1,1,1,1,0,0,1,0,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,1,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,1,0,0,0,1,0];
const XTE = [[-1.064371,0.235753],[1.130754,0.265134],[0.727612,0.746457],[0.021527,0.222224],[0.368661,0.309693],[0.764275,-0.389843],[-1.006598,0.759532],[-1.084206,0.566668],[1.362242,-0.481333],[1.601322,-0.242026],[-0.429351,1.080101],[0.561247,0.681998],[0.001725,0.236516],[0.902511,-0.032384],[0.746863,-0.242457],[0.237169,0.663567],[0.182501,-0.268077],[0.696752,-0.44014],[-0.013477,-0.136155],[1.530083,-0.311801],[1.648104,0.36173],[0.927937,0.735503],[-0.303689,1.047373],[0.182881,-0.31416],[-0.09818,0.51376],[0.234695,0.902681],[1.057562,-0.911726],[-1.112753,-0.713641],[0.993518,0.258303],[-0.736891,0.712101],[0.414735,1.200192],[-0.024044,0.841323],[1.164385,0.005003],[-0.873282,0.578694],[0.680074,-0.02166],[0.64037,0.785496],[2.054132,0.02189],[0.34676,-0.210749],[1.433934,-0.80324],[-0.069282,0.686988],[1.069115,-0.481489],[1.028514,0.625018],[-0.10797,-0.011689],[2.099164,0.664785],[-0.302417,0.604363],[-0.998812,0.691953],[-0.900621,0.05576],[-0.64869,0.974097],[2.141856,0.735832],[0.409702,0.918799],[-0.733057,1.125866],[0.950752,-0.839231],[0.622528,0.508619],[1.284397,-0.391052],[1.300444,-0.676446],[-0.400548,0.838748],[-0.336861,0.638403],[-0.020682,1.096948],[2.308991,0.240654],[0.27356,0.097211],[1.086284,0.312457],[0.203777,0.489976],[2.065123,0.123949],[-1.086182,0.180157],[0.114354,0.908311],[-1.243163,-0.07035],[0.744125,-0.21838],[0.10055,0.260977],[-0.553783,0.407232],[0.578647,0.957311],[0.14418,0.802457],[1.111422,0.276692],[0.560859,0.883422],[2.042611,0.209928],[0.238172,1.156652],[1.91417,0.437265],[1.880356,0.345412],[0.652328,0.691966],[-0.332725,1.198609],[-0.010649,0.234596],[0.345189,0.194939],[0.117745,0.181096],[0.850005,-0.738008],[-0.164264,0.880922],[-0.87063,0.11572],[0.65097,1.037277],[0.280593,-0.131956],[-0.782968,0.547485],[1.715646,-0.033134],[-0.006167,0.024252],[1.07945,0.022892],[-0.004836,1.123827],[1.286142,-0.543297],[1.744595,-0.476281],[0.657825,0.653983],[1.34312,-0.421104],[0.648193,0.38877],[0.566245,-0.114033],[1.853197,-0.105329],[0.077183,-0.211001]];
const YTE = [0,0,0,1,1,1,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,0,0,0,1,0,0,1,0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,1,1,1,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,1,1,1];
/* the notebook's initial weights: default_rng(0).normal, scaled by sqrt(2 / fan-in); biases zero */
const W1_NB = [[0.1257302210933933, -0.1321048632913019, 0.6404226504432821, 0.10490011715303971, -0.535669373161111, 0.36159505490948474, 1.3040000451301372, 0.9470809631292422], [-0.7037352358069926, -1.2654214710460525, -0.6232744625373522, 0.0413259793472436, -2.3250307746388343, -0.21879166393254573, -1.2459109472530652, -0.7322673547034516]];
const W2_NB = [-0.27212949142865495, -0.15815007818457727, 0.2058152681870664, 0.5212566847213388, -0.06426733147201713, 0.6832317352748429, -0.33259733674330677, 0.17575503504650986];
const NTR = XTR.length, EPOCHS = 3000, LOG_EVERY = 20;

/* ---------------- the network, as in the notebook ---------------- */
const sigmoid = z => 1 / (1 + Math.exp(-z));
function initParams(nh) {
  if (nh === 8) return { W1: W1_NB.map(r => r.slice()), b1: new Array(8).fill(0), W2: W2_NB.slice(), b2: 0 };
  const rng = makeRng(1000 + nh);
  return { W1: [0, 1].map(() => Array.from({ length: nh }, () => rng.normal(0, 1) * Math.sqrt(2 / 2))), b1: new Array(nh).fill(0), W2: Array.from({ length: nh }, () => rng.normal(0, 1) * Math.sqrt(2 / nh)), b2: 0 };
}
/* forward pass of one point: hidden pre-activations, activations, output probability */
function forwardOne(p, x, relu) {
  const nh = p.W1[0].length, z1 = new Array(nh), a1 = new Array(nh); let z2 = p.b2;
  for (let k = 0; k < nh; k++) { z1[k] = x[0] * p.W1[0][k] + x[1] * p.W1[1][k] + p.b1[k]; a1[k] = relu ? Math.max(0, z1[k]) : z1[k]; z2 += a1[k] * p.W2[k]; }
  return { z1, a1, z2, yhat: sigmoid(z2) };
}
const EPS = 1e-9, bceOne = (y, p) => -(y * Math.log(p + EPS) + (1 - y) * Math.log(1 - p + EPS));
function lossOf(p, X, Y, relu) { let s = 0; for (let i = 0; i < X.length; i++) s += bceOne(Y[i], forwardOne(p, X[i], relu).yhat); return s / X.length; }
function accOf(p, X, Y, relu) { let s = 0; for (let i = 0; i < X.length; i++) s += ((forwardOne(p, X[i], relu).yhat > 0.5 ? 1 : 0) === Y[i]) ? 1 : 0; return s / X.length; }
/* one full-batch step: forward, loss, the four backward steps, the update; returns the loss before the step */
function gdStep(p, lr, relu) {
  const nh = p.W1[0].length, dW1 = [new Array(nh).fill(0), new Array(nh).fill(0)], db1 = new Array(nh).fill(0), dW2 = new Array(nh).fill(0); let db2 = 0, L = 0;
  for (let i = 0; i < NTR; i++) {
    const x = XTR[i], y = YTR[i], f = forwardOne(p, x, relu), dz2 = (f.yhat - y) / NTR;         // dL/dz2, the sigmoid + cross-entropy shortcut
    L += bceOne(y, f.yhat);
    db2 += dz2;
    for (let k = 0; k < nh; k++) {
      dW2[k] += f.a1[k] * dz2;
      const dz1 = dz2 * p.W2[k] * (relu ? (f.z1[k] > 0 ? 1 : 0) : 1);                             // through the ReLU mask
      dW1[0][k] += x[0] * dz1; dW1[1][k] += x[1] * dz1; db1[k] += dz1;
    }
  }
  for (let k = 0; k < nh; k++) { p.W1[0][k] -= lr * dW1[0][k]; p.W1[1][k] -= lr * dW1[1][k]; p.b1[k] -= lr * db1[k]; p.W2[k] -= lr * dW2[k]; }
  p.b2 -= lr * db2;
  return L / NTR;
}
const clone = p => ({ W1: p.W1.map(r => r.slice()), b1: p.b1.slice(), W2: p.W2.slice(), b2: p.b2 });
/* a whole run, with a snapshot every 20 epochs (as the notebook logs) and at the end */
function trainRun(lr, nh, relu) {
  const p = initParams(nh), snaps = [];
  for (let ep = 0; ep < EPOCHS; ep++) {
    if (ep % LOG_EVERY === 0) snaps.push({ ep, p: clone(p), loss: lossOf(p, XTR, YTR, relu), te: accOf(p, XTE, YTE, relu) });
    gdStep(p, lr, relu);
  }
  snaps.push({ ep: EPOCHS, p: clone(p), loss: lossOf(p, XTR, YTR, relu), te: accOf(p, XTE, YTE, relu) });
  return snaps;
}
const deadUnits = (p, relu) => { if (!relu) return 0; const nh = p.W1[0].length, alive = new Array(nh).fill(false); for (const x of XTR) { for (let k = 0; k < nh; k++) if (x[0] * p.W1[0][k] + x[1] * p.W1[1][k] + p.b1[k] > 0) alive[k] = true; } return alive.filter(a => !a).length; };
function dot(ctx, x, y, r, fill, stroke, lw = 1.2) { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); } }

/* =====================================================================
   THE DEMO: TRAIN THE NETWORK
   ===================================================================== */
const hero = { lr: 0.5, nh: 8, relu: true, ep: EPOCHS, lines: false, cache: {} };
function heroRun() { const key = `${hero.lr}|${hero.nh}|${hero.relu}`; return hero.cache[key] || (hero.cache[key] = trainRun(hero.lr, hero.nh, hero.relu)); }
const lrLabel = v => (v >= 10 ? fmt(v, 0) : v >= 1 ? fmt(v, 1) : fmt(v, 2));
function drawHero() {
  const snaps = heroRun(), si = Math.min(snaps.length - 1, Math.round(hero.ep / LOG_EVERY)), s = snaps[si], p = s.p;
  $("#hLV").textContent = lrLabel(hero.lr); $("#hEV").textContent = String(s.ep); $("#hLines").setAttribute("aria-pressed", String(hero.lines));
  $$("#demo [data-nh]").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.nh === hero.nh))); $$("#demo [data-act]").forEach(b => b.setAttribute("aria-pressed", String((b.dataset.act === "relu") === hero.relu)));
  // the boundary
  const cv = $("#bndC"), narrow = narrowOf(cv), fr = frame(cv, narrow ? 1.25 : 1.55, { x: [-1.6, 2.7], y: [-1.1, 1.7], xticks: [-1, 0, 1, 2], yticks: [-1, 0, 1] }), { ctx, X, Y } = fr, cell = narrow ? 4 : 5;
  ctx.save(); ctx.beginPath(); ctx.rect(fr.pad.l, fr.pad.t, fr.iw, fr.ih); ctx.clip();
  for (let px = fr.pad.l; px < fr.pad.l + fr.iw; px += cell) for (let py = fr.pad.t; py < fr.pad.t + fr.ih; py += cell) {
    const x = [fr.x0 + (px + cell / 2 - fr.pad.l) / fr.iw * (fr.x1 - fr.x0), fr.y1 - (py + cell / 2 - fr.pad.t) / fr.ih * (fr.y1 - fr.y0)], q = forwardOne(p, x, hero.relu).yhat;
    ctx.fillStyle = q > 0.5 ? alpha("--warm", 0.08 + 0.4 * (q - 0.5) * 2) : alpha("--post", 0.08 + 0.4 * (0.5 - q) * 2); ctx.fillRect(px, py, cell, cell);
  }
  if (hero.lines && hero.relu) {                                                               // each hidden unit's hinge: the line z1_k = 0
    ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
    for (let k = 0; k < hero.nh; k++) {
      const a = p.W1[0][k], b = p.W1[1][k], c = p.b1[k]; if (Math.abs(a) < 1e-12 && Math.abs(b) < 1e-12) continue;
      ctx.strokeStyle = alpha("--ink", deadUnitK(p, k) ? 0.15 : 0.6);
      if (Math.abs(b) > Math.abs(a)) seg(ctx, X(fr.x0), Y(-(a * fr.x0 + c) / b), X(fr.x1), Y(-(a * fr.x1 + c) / b)); else seg(ctx, X(-(b * fr.y0 + c) / a), Y(fr.y0), X(-(b * fr.y1 + c) / a), Y(fr.y1));
    }
    ctx.setLineDash([]);
  }
  ctx.restore();
  XTR.forEach((x, i) => dot(ctx, X(x[0]), Y(x[1]), narrow ? 2.6 : 3, YTR[i] ? css("--warm") : css("--post"), css("--surface"), 0.6));
  XTE.forEach((x, i) => { const r = narrow ? 3.5 : 4; ctx.beginPath(); ctx.moveTo(X(x[0]), Y(x[1]) - r); ctx.lineTo(X(x[0]) + r, Y(x[1])); ctx.lineTo(X(x[0]), Y(x[1]) + r); ctx.lineTo(X(x[0]) - r, Y(x[1])); ctx.closePath(); ctx.fillStyle = YTE[i] ? css("--warm") : css("--post"); ctx.fill(); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 0.8; ctx.stroke(); });
  // the curves
  const cc = $("#curveC"), top = Math.min(3, Math.max(1, ...snaps.map(q => q.loss)) * 1.05);
  const cf = frame(cc, narrow ? 1.6 : 2.6, { x: [0, EPOCHS], y: [0, top], xlabel: "epoch", yticks: top > 1.5 ? [0, 1, 2, 3].filter(v => v <= top) : [0, 0.25, 0.5, 0.75, 1] });
  cf.ctx.strokeStyle = alpha("--ink", 0.35); cf.ctx.setLineDash([3, 3]); seg(cf.ctx, cf.X(s.ep), cf.pad.t, cf.X(s.ep), cf.Y(0)); cf.ctx.setLineDash([]);
  plotLine(cf, snaps.map(q => q.ep), snaps.map(q => Math.min(top, q.loss)), { color: css("--post"), width: 2.2 });
  plotLine(cf, snaps.map(q => q.ep), snaps.map(q => q.te), { color: css("--warm"), width: 2.2 });
  const dead = deadUnits(p, hero.relu), trA = accOf(p, XTR, YTR, hero.relu);
  $("#rLoss").textContent = fmt(s.loss, 4); $("#rTrain").textContent = fmt(trA, 3); $("#rTest").textContent = `${fmt(s.te, 3)} (${Math.round(s.te * 100)} of 100)`; $("#rDead").textContent = hero.relu ? `${dead} of ${hero.nh}` : "no ReLU";
  const isNb = hero.lr === 0.5 && hero.nh === 8 && hero.relu;
  let text = `<b>Epoch ${s.ep}</b>, learning rate ${lrLabel(hero.lr)}, ${hero.nh} hidden units${hero.relu ? "" : " with no nonlinearity"}: training loss <b>${fmt(s.loss, 4)}</b>, test accuracy <b>${fmt(s.te, 3)}</b>. `;
  if (isNb && s.ep === EPOCHS) text += "This is the notebook's run exactly, from its own initial weights: loss 0.0674, 96 of 100 test points right. ";
  if (!hero.relu) text += "Without the ReLU the two layers collapse into one straight cut, whatever the width or the number of epochs: the boundary is a line, and it cannot follow the moons. ";
  else if (hero.lr >= 20) text += `A learning rate this large throws the weights so far that ${dead === hero.nh ? "every hidden unit" : `${dead} of the ${hero.nh} hidden units`} ended up negative on all 300 points: a dead ReLU passes nothing and gets no gradient, so it never recovers. ${dead === hero.nh ? "The output is a constant, and the loss sits at its starting value." : ""}`;
  else if (hero.lr >= 5) text += "The loss rises on about half of the steps: the step overshoots the valley, as in chapter 02, and the bounces kill hidden units along the way. ";
  else if (hero.lr <= 0.1) text += "Safe but slow: after 3000 epochs the boundary is still barely curved. ";
  else if (hero.nh <= 2) text += "Two hinges cannot bend a boundary around two crescents: the accuracy plateaus well below the 8-unit run, and the probability map shows one or two straight folds. ";
  else if (isNb) text += s.ep < 1400 && s.ep >= 400 ? "The plateau: between epochs 400 and 1400 the test accuracy sits at 0.84 while the loss creeps down, until the hidden units find the second bend. " : s.ep < 400 ? "Early on the boundary is close to a line, and a line gets about 80% of these points. " : "";
  if (hero.lines && hero.relu) text += "Dashed: each hidden unit's hinge line, where its pre-activation is zero; the boundary is bent only where it crosses one.";
  caption($("#heroStatus"), text);
}
function deadUnitK(p, k) { for (const x of XTR) if (x[0] * p.W1[0][k] + x[1] * p.W1[1][k] + p.b1[k] > 0) return false; return true; }
$("#hL").addEventListener("input", e => { hero.lr = +(10 ** +e.target.value).toPrecision(2); drawHero(); });
$("#hE").addEventListener("input", e => { hero.ep = +e.target.value; drawHero(); });
$$("#demo [data-nh]").forEach(b => b.addEventListener("click", () => { hero.nh = +b.dataset.nh; drawHero(); }));
$$("#demo [data-act]").forEach(b => b.addEventListener("click", () => { hero.relu = b.dataset.act === "relu"; drawHero(); }));
$$("#demo [data-lr]").forEach(b => b.addEventListener("click", () => { hero.lr = +b.dataset.lr; $("#hL").value = String(Math.log10(hero.lr)); drawHero(); }));
$("#hLines").addEventListener("click", () => { hero.lines = !hero.lines; drawHero(); });
$("#hReset").addEventListener("click", () => { hero.lr = 0.5; hero.nh = 8; hero.relu = true; hero.ep = EPOCHS; $("#hL").value = String(Math.log10(0.5)); $("#hE").value = String(EPOCHS); drawHero(); });

/* =====================================================================
   PART 2: THE CROSS-ENTROPY OF ONE PREDICTION
   ===================================================================== */
function drawCe() {
  const yh = +$("#ceP").value, cv = $("#ceC"), fr = frame(cv, narrowOf(cv) ? 1.6 : 2.8, { x: [0, 1], y: [0, 5], xlabel: "predicted probability ŷ", ylabel: "loss", yticks: [0, 1, 2, 3, 4, 5] }), { ctx, X, Y } = fr;
  $("#cePV").textContent = fmt(yh, 2);
  const ps = Array.from({ length: 199 }, (_, i) => 0.005 + i * 0.005);
  plotLine(fr, ps, ps.map(q => Math.min(5, -Math.log(q))), { color: css("--warm"), width: 2.2 }); plotLine(fr, ps, ps.map(q => Math.min(5, -Math.log(1 - q))), { color: css("--post"), width: 2.2 });
  dot(ctx, X(yh), Y(Math.min(5, -Math.log(yh))), 5, css("--warm")); dot(ctx, X(yh), Y(Math.min(5, -Math.log(1 - yh))), 5, css("--post"));
  caption($("#ceCap"), `If the model says [[\\hat y = ${fmt(yh, 2)}]]: when the label is 1 (orange) the loss is [[-\\log ${fmt(yh, 2)} = ${fmt(-Math.log(yh), 3)}]]; when it is 0 (blue), [[-\\log ${fmt(1 - yh, 2)} = ${fmt(-Math.log(1 - yh), 3)}]]. ` + (yh > 0.95 || yh < 0.05 ? "A confident answer costs almost nothing when right and a great deal when wrong; the loss has no ceiling." : Math.abs(yh - 0.5) < 0.03 ? "At 0.5 the model commits to nothing and pays log 2 = 0.693 either way." : "The two curves cross at 0.5; the output error ŷ − y is the slope of the loss along z₂, whichever label holds."));
}
$("#ceP").addEventListener("input", drawCe);

/* =====================================================================
   PART 4: ONE SIGMOID NEURON, ONE STEP, BY HAND
   ===================================================================== */
function drawNeuron() {
  const w1 = +$("#nW1").value, w2 = +$("#nW2").value, b = +$("#nB").value, lr = +$("#nL").value, x = [1, 2], y = 1;
  $("#nW1V").textContent = fmt(w1, 2); $("#nW2V").textContent = fmt(w2, 2); $("#nBV").textContent = fmt(b, 2); $("#nLV").textContent = fmt(lr, 2);
  const z = w1 * x[0] + w2 * x[1] + b, yh = sigmoid(z), e = yh - y, nw1 = w1 - lr * e * x[0], nw2 = w2 - lr * e * x[1], nb = b - lr * e;
  const z2 = nw1 * x[0] + nw2 * x[1] + nb, yh2 = sigmoid(z2);
  const row = (a, bb, c) => `<tr><td class="words">${a}</td><td class="words">${bb}</td><td>${c}</td></tr>`;
  $("#neuronT tbody").innerHTML = [
    row("pre-activation", `[[z = ${fmt(w1, 2)} \\cdot 1 + (${fmt(w2, 2)}) \\cdot 2 + ${fmt(b, 2)}]]`, fmt(z, 3)), row("output", "[[\\hat y = \\sigma(z)]]", fmt(yh, 3)), row("loss", `[[-\\log \\hat y]]`, fmt(-Math.log(yh), 3)),
    row("output error", "[[\\hat y - y]]", fmt(e, 3)), row("gradient for w₁", `[[(\\hat y - y)\\, x_1]]`, fmt(e * x[0], 3)), row("gradient for w₂", `[[(\\hat y - y)\\, x_2]]`, fmt(e * x[1], 3)), row("gradient for b", `[[\\hat y - y]]`, fmt(e, 3)),
    row("new w₁", `[[${fmt(w1, 2)} - ${fmt(lr, 2)} \\cdot (${fmt(e * x[0], 3)})]]`, fmt(nw1, 3)), row("new w₂", `[[${fmt(w2, 2)} - ${fmt(lr, 2)} \\cdot (${fmt(e * x[1], 3)})]]`, fmt(nw2, 3)), row("new b", `[[${fmt(b, 2)} - ${fmt(lr, 2)} \\cdot (${fmt(e, 3)})]]`, fmt(nb, 3)),
  ].join("");
  $$("#neuronT td").forEach(td => { if (td.innerHTML.includes("[[")) caption(td, td.innerHTML); });
  caption($("#neuronCap"), `After the step, [[z = ${fmt(z2, 3)}]], [[\\hat y = ${fmt(yh2, 3)}]] and the loss is [[${fmt(-Math.log(yh2), 3)}]], ${yh2 > yh ? "down" : "up"} from ${fmt(-Math.log(yh), 3)}. ` + (w1 === 0.5 && w2 === -0.25 && b === 0 && lr === 0.5 ? "These are the notebook's numbers. " : "") + `The input [[x_2 = 2]] is twice [[x_1 = 1]], so [[w_2]] moves twice as far as [[w_1]]: each weight is adjusted in proportion to the input that flowed through it.` + (lr >= 4 ? " With a step this large the neuron overshoots: it is now very confident, and a point of the other class nearby would be badly wrong." : ""));
}
["#nW1", "#nW2", "#nB", "#nL"].forEach(id => $(id).addEventListener("input", drawNeuron));
$("#nReset").addEventListener("click", () => { $("#nW1").value = "0.5"; $("#nW2").value = "-0.25"; $("#nB").value = "0"; $("#nL").value = "0.5"; drawNeuron(); });

/* =====================================================================
   PART 3: A GRADIENT CHECK OF BACKPROPAGATION
   ===================================================================== */
function backpropGrad(p, relu) {
  const nh = p.W1[0].length, g = { W1: [new Array(nh).fill(0), new Array(nh).fill(0)], b1: new Array(nh).fill(0), W2: new Array(nh).fill(0), b2: 0 };
  for (let i = 0; i < NTR; i++) {
    const x = XTR[i], f = forwardOne(p, x, relu), dz2 = (f.yhat - YTR[i]) / NTR; g.b2 += dz2;
    for (let k = 0; k < nh; k++) { g.W2[k] += f.a1[k] * dz2; const dz1 = dz2 * p.W2[k] * (relu ? (f.z1[k] > 0 ? 1 : 0) : 1); g.W1[0][k] += x[0] * dz1; g.W1[1][k] += x[1] * dz1; g.b1[k] += dz1; }
  }
  return g;
}
function drawGc() {
  const sel = $("#gcParam").value, h = 10 ** +$("#gcH").value, p = initParams(8), g = backpropGrad(p, true);
  $("#gcHV").textContent = h.toExponential(0).replace("e-", " × 10⁻").replace("e+", " × 10");
  const [name, i, j] = sel.split(":"), get = () => (name === "W1" ? p.W1[+i][+j] : name === "b1" ? p.b1[+j] : name === "W2" ? p.W2[+j] : p.b2), set = v => { if (name === "W1") p.W1[+i][+j] = v; else if (name === "b1") p.b1[+j] = v; else if (name === "W2") p.W2[+j] = v; else p.b2 = v; };
  const v0 = get(), bp = name === "W1" ? g.W1[+i][+j] : name === "b1" ? g.b1[+j] : name === "W2" ? g.W2[+j] : g.b2;
  set(v0 + h); const Lp = lossOf(p, XTR, YTR, true); set(v0 - h); const Lm = lossOf(p, XTR, YTR, true); set(v0);
  const fd = (Lp - Lm) / (2 * h), label = name === "W1" ? `W_1[${i}, ${j}]` : name === "b1" ? `b_1[${j}]` : name === "W2" ? `W_2[${j}]` : "b_2";
  caption($("#gcCap"), `At the notebook's initial weights, for [[${label}]]: backpropagation gives [[${bp.toExponential(5).replace("e", "\\times 10^{") + "}"}]]; nudging the weight by [[\\pm h]] and differencing the loss gives [[\\frac{L(\\theta + h) - L(\\theta - h)}{2h} = ${fd.toExponential(5).replace("e", "\\times 10^{") + "}"}]]. They differ by [[${Math.abs(fd - bp).toExponential(1).replace("e", "\\times 10^{") + "}"}]]. ` + (h >= 1e-2 ? "A step this large bends the loss between the two evaluations, so the difference quotient drifts from the derivative." : h <= 1e-9 ? "A step this small loses digits to rounding: the two losses agree to almost all their decimals, and their difference is mostly noise." : "Four steps of the chain rule, confirmed to many digits by arithmetic that knows nothing about calculus."));
}
["#gcParam", "#gcH"].forEach(id => $(id).addEventListener(id === "#gcH" ? "input" : "change", drawGc));

/* =====================================================================
   start-up
   ===================================================================== */
function drawAll() { drawHero(); drawCe(); drawNeuron(); drawGc(); }
drawAll();
onRedraw(drawAll);
})();
