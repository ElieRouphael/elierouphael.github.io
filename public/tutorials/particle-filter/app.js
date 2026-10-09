(() => {
"use strict";
const { makeRng, makeTerrain, heightAt, heightSmooth, Robot, ParticleFilter, NOISE, WIDTH, HEIGHT, wrapAngle } = window.PF;
const $ = (sel, root = document) => root.querySelector(sel);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t === "dark" || (t !== "light" && darkQuery.matches);
};
const DEG = Math.PI / 180;
const randomSeed = () => Math.floor(Math.random() * 1e9);

/* ---------------- math and code ---------------- */
function renderMath() {
  if (!window.katex) return;
  document.querySelectorAll(".m, .math-block").forEach(node => {
    if (node.dataset.rendered) return;
    try {
      katex.render(node.textContent, node, {
        displayMode: node.classList.contains("math-block"), throwOnError: false, strict: false,
      });
      node.dataset.rendered = "1";
    } catch (err) { /* leave the TeX source visible */ }
  });
}
renderMath();
window.addEventListener("load", renderMath);
if (window.hljs) document.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));

/* ---------------- canvas helpers ---------------- */
/* Size a canvas to its CSS box times the device pixel ratio; draw in CSS pixels. */
function fit(canvas, aspect) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = aspect ? w / aspect : canvas.clientHeight;
  if (aspect) canvas.style.height = h + "px";
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}
const hexToRgb = hex => {
  const v = parseInt(hex.replace("#", ""), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
};

/* Terrain picture: one warm-grey ramp (dark = low, light = high, by absolute height
   so that flat ground looks flat), a little hill shading and a contour line every 10 m. */
const PX_PER_M = 4;
function terrainImage(t) {
  const W = t.w * PX_PER_M, H = t.h * PX_PER_M;
  const [lo, hi] = isDark() ? [[44, 42, 38], [180, 174, 163]] : [[107, 102, 93], [245, 242, 234]];
  const heights = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    const y = (j + 0.5) / PX_PER_M - 0.5;
    for (let i = 0; i < W; i++) heights[j * W + i] = heightSmooth(t, (i + 0.5) / PX_PER_M - 0.5, y);
  }
  const canvas = document.createElement("canvas");
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(W, H);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const k = j * W + i, z = heights[k];
      const zx = heights[k + (i < W - 1 ? 1 : 0)] - heights[k - (i > 0 ? 1 : 0)];
      const zy = heights[k + (j < H - 1 ? W : 0)] - heights[k - (j > 0 ? W : 0)];
      const shade = Math.max(-0.5, Math.min(0.5, -(zx + zy) * 0.9));        // light from the top left
      const level = Math.floor(z / 10);
      const right = i < W - 1 ? Math.floor(heights[k + 1] / 10) : level;
      const down = j < H - 1 ? Math.floor(heights[k + W] / 10) : level;
      const contour = (right !== level || down !== level) ? 0.88 : 1;
      const f = Math.max(0, Math.min(1, z / 100));
      const m = (1 + 0.22 * shade) * contour;
      for (let c = 0; c < 3; c++) img.data[4 * k + c] = Math.max(0, Math.min(255, (lo[c] + (hi[c] - lo[c]) * f) * m));
      img.data[4 * k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

function drawRobot(ctx, x, y, heading, size = 1) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.strokeStyle = isDark() ? "#f4f1ea" : "#1d1b18";
  ctx.lineWidth = 2.5 * size;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + 15 * size * Math.cos(heading), y + 15 * size * Math.sin(heading));
  ctx.stroke();
  ctx.fillStyle = css("--robot");
  ctx.lineWidth = 1.6 * size;
  ctx.beginPath();
  ctx.arc(x, y, 6.5 * size, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}
function drawRing(ctx, x, y) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, 9, 0, 2 * Math.PI);
  ctx.lineWidth = 4.5; ctx.strokeStyle = "rgba(20,19,17,0.85)"; ctx.stroke();
  ctx.lineWidth = 2.2; ctx.strokeStyle = "#ffffff"; ctx.stroke();
  ctx.restore();
}
const fmt = (v, d = 1) => (v == null || !isFinite(v)) ? "–" : v.toFixed(d);

/* =====================================================================
   MAIN DEMO
   ===================================================================== */
const TERRAINS = {
  hilly: { relief: 100, octaves: 5 },
  gentle: { relief: 100, octaves: 2 },
  flat: { relief: 4, octaves: 5 },
};
const DEFAULTS = { n: 2000, terrain: "hilly", noise: 2.5, compass: true, explorers: false, show: true };
const STEP = 4, TURN = 20 * DEG, HISTORY = 120;

const demo = {
  canvas: $("#map"),
  settings: { ...DEFAULTS },
  seed: 7,
  rng: makeRng(randomSeed()),
  terrain: null, image: null, robot: null, filter: null,
  history: [], lastReading: null, autoTimer: null, interacted: false,
};

function placeRobotRandomly() {
  const r = demo.rng;
  demo.robot = new Robot(demo.terrain, r, 30 + r.uniform() * (WIDTH - 60), 25 + r.uniform() * (HEIGHT - 50), r.uniform() * 2 * Math.PI);
}
function buildTerrain() {
  demo.terrain = makeTerrain({ seed: demo.seed, ...TERRAINS[demo.settings.terrain] });
  demo.image = terrainImage(demo.terrain);
}
function buildFilter() {
  const s = demo.settings;
  demo.filter = new ParticleFilter(demo.terrain, demo.rng, {
    n: s.n, assumedNoise: s.noise, compass: s.compass, explorers: s.explorers ? 0.01 : 0,
  });
  demo.history = [];
  demo.lastReading = null;
}
function fullReset() { buildTerrain(); placeRobotRandomly(); buildFilter(); }

/* One move of the robot followed by one cycle of the filter. */
function act(distance, turn, draw = true) {
  const { robot, filter } = demo;
  robot.step(distance, turn);
  const reading = robot.readAltimeter();
  filter.step(distance, turn, robot.readCompass(), reading);
  demo.lastReading = reading;
  const e = filter.estimate();
  demo.history.push({ step: filter.steps, spread: e.spread, error: Math.hypot(e.x - robot.x, e.y - robot.y) });
  if (demo.history.length > HISTORY) demo.history.shift();
  if (draw) drawDemo();
}

/* Auto-drive: drive forward, wander a little, and turn back toward the middle near the fence. */
function autoMove() {
  const { robot, rng } = demo;
  const ax = robot.x + 22 * Math.cos(robot.heading), ay = robot.y + 22 * Math.sin(robot.heading);
  let turn;
  if (ax < 8 || ax > WIDTH - 8 || ay < 8 || ay > HEIGHT - 8) {
    const want = Math.atan2(HEIGHT / 2 - robot.y, WIDTH / 2 - robot.x);
    turn = Math.max(-0.4, Math.min(0.4, wrapAngle(want - robot.heading)));
  } else {
    turn = rng.normal(0.15);
  }
  return [STEP, turn];
}
function setAuto(on) {
  clearInterval(demo.autoTimer);
  demo.autoTimer = null;
  $("#auto").setAttribute("aria-pressed", on ? "true" : "false");
  $("#auto").textContent = on ? "❚❚ pause" : "▶ auto-drive";
  if (on) demo.autoTimer = setInterval(() => act(...autoMove()), reduceMotion ? 380 : 170);
}

function statusText() {
  const { filter, robot, history, settings } = demo;
  if (filter.steps === 0) return "The guesses are spread evenly over the map: the robot has <b>no idea</b> where it is. Drive it, or press auto-drive.";
  const e = filter.estimate();
  const err = Math.hypot(e.x - robot.x, e.y - robot.y);
  if (e.spread > 35) return "<b>Still very unsure.</b> The readings so far fit many places on the map. Keep driving.";
  if (e.spread > 8) return "<b>Narrowing down.</b> The guesses have gathered into fewer, smaller clumps. Each new reading rules some of them out.";
  if (!settings.show) return `<b>The guesses agree:</b> the robot believes it is within about ${fmt(e.spread, 0)} m of the white ring. Tick “Show the real robot” in Settings to check.`;
  if (err < 6) return `<b>Found it.</b> The guesses agree, and the best guess is ${fmt(err)} m from the real robot.`;
  return `<b>Confident, but wrong!</b> The guesses agree on a place ${fmt(err, 0)} m away from the real robot.`;
}

function drawDemo() {
  const { ctx, w, h } = fit(demo.canvas, 1.5);
  const s = w / WIDTH;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(demo.image, 0, 0, w, h);
  const f = demo.filter;
  const size = Math.max(1.6, Math.min(2.6, s * 0.75)), half = size / 2;
  ctx.fillStyle = css("--guess");
  ctx.globalAlpha = f.n > 4000 ? 0.6 : 0.85;
  for (let i = 0; i < f.n; i++) ctx.fillRect(f.px[i] * s - half, f.py[i] * s - half, size, size);
  ctx.globalAlpha = 1;
  const e = f.estimate();
  if (f.steps > 0) drawRing(ctx, e.x * s, e.y * s);
  if (demo.settings.show) drawRobot(ctx, demo.robot.x * s, demo.robot.y * s, demo.robot.heading);

  const last = demo.history[demo.history.length - 1];
  $("#rStep").textContent = f.steps;
  $("#rAlt").textContent = demo.lastReading == null ? "–" : fmt(demo.lastReading) + " m";
  $("#rSpread").textContent = fmt(e.spread) + " m";
  $("#rErr").textContent = !demo.settings.show ? "hidden" : last ? fmt(last.error) + " m" : "–";
  $("#status").innerHTML = statusText();
  drawChart();
}

/* Spread (blue) and error (orange) over the last steps, with a hover readout. */
const chart = { canvas: $("#chart"), tip: $("#chartTip"), hover: null };
function drawChart() {
  const { ctx, w, h } = fit(chart.canvas);
  ctx.clearRect(0, 0, w, h);
  const H = demo.history, showErr = demo.settings.show;
  const pad = { l: 30, r: 6, t: 6, b: 16 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  let top = 10;
  for (const p of H) top = Math.max(top, p.spread, showErr ? p.error : 0);
  top = Math.min(150, Math.ceil(top / 10) * 10);
  const span = Math.max(30, H.length) - 1;
  chart.span = span;
  const X = i => pad.l + i / span * iw;
  const Y = v => pad.t + ih - Math.min(v, top) / top * ih;
  ctx.font = "10px " + css("--font-mono");
  ctx.fillStyle = css("--muted");
  ctx.strokeStyle = css("--grid");
  ctx.lineWidth = 1;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const v of [0, top / 2, top]) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(v) + 0.5); ctx.lineTo(w - pad.r, Y(v) + 0.5); ctx.stroke();
    ctx.fillText(v + (v === top ? " m" : ""), pad.l - 4, Y(v));
  }
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  ctx.fillText(H.length ? "step " + H[0].step : "no steps yet", pad.l, h - 3);
  ctx.textAlign = "right";
  if (H.length) ctx.fillText("step " + H[H.length - 1].step, w - pad.r, h - 3);
  const line = (key, color) => {
    if (!H.length) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = "round";
    ctx.beginPath();
    H.forEach((p, i) => { const x = X(i), y = Y(p[key]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  };
  line("spread", css("--guess"));
  if (showErr) line("error", css("--robot"));
  if (chart.hover != null && H.length) {
    const i = Math.max(0, Math.min(H.length - 1, chart.hover));
    const p = H[i], x = X(i);
    ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x + 0.5, pad.t); ctx.lineTo(x + 0.5, pad.t + ih); ctx.stroke();
    const dots = [["spread", css("--guess")]].concat(showErr ? [["error", css("--robot")]] : []);
    for (const [k, c] of dots) {
      ctx.fillStyle = c; ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, Y(p[k]), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
    chart.tip.innerHTML = `step ${p.step}<br>spread ${fmt(p.spread)} m` + (showErr ? `<br>error ${fmt(p.error)} m` : "");
    chart.tip.style.left = Math.max(50, Math.min(w - 50, x)) + "px";
    chart.tip.style.top = pad.t + "px";
    chart.tip.classList.add("on");
  } else {
    chart.tip.classList.remove("on");
  }
}
chart.canvas.addEventListener("pointermove", ev => {
  const r = chart.canvas.getBoundingClientRect();
  const pad = 30, iw = r.width - pad - 6;
  chart.hover = Math.round((ev.clientX - r.left - pad) / iw * (chart.span || HISTORY - 1));
  if (chart.hover >= demo.history.length) chart.hover = demo.history.length - 1;
  drawChart();
});
chart.canvas.addEventListener("pointerleave", () => { chart.hover = null; drawChart(); });

/* ---- controls ---- */
function noteInteraction() {
  if (demo.interacted) return;
  demo.interacted = true;
  $("#mapOverlay").style.display = "none";
}
function holdButton(btn, fn) {
  let delay = null, rep = null;
  const stop = () => { clearTimeout(delay); clearInterval(rep); delay = rep = null; };
  btn.addEventListener("pointerdown", ev => {
    if (ev.button !== 0) return;
    ev.preventDefault();
    noteInteraction(); fn();
    delay = setTimeout(() => { rep = setInterval(fn, 160); }, 380);
  });
  ["pointerup", "pointerleave", "pointercancel"].forEach(t => btn.addEventListener(t, stop));
  btn.addEventListener("click", ev => { if (ev.detail === 0) { noteInteraction(); fn(); } });   // keyboard
}
holdButton($("#fwd"), () => act(STEP, 0));
holdButton($("#left"), () => act(0, -TURN));
holdButton($("#right"), () => act(0, TURN));
demo.canvas.addEventListener("keydown", ev => {
  const k = ev.key.toLowerCase();
  if (k === "arrowup" || k === "w") act(STEP, 0);
  else if (k === "arrowleft" || k === "a") act(0, -TURN);
  else if (k === "arrowright" || k === "d") act(0, TURN);
  else return;
  ev.preventDefault();
  noteInteraction();
});
demo.canvas.addEventListener("pointerdown", () => { demo.canvas.focus({ preventScroll: true }); noteInteraction(); });
$("#auto").addEventListener("click", () => { noteInteraction(); setAuto(!demo.autoTimer); });
$("#resetF").addEventListener("click", () => { buildFilter(); drawDemo(); });
$("#newMap").addEventListener("click", () => { demo.seed = randomSeed(); fullReset(); drawDemo(); });
function kidnap() {
  const r = demo.rng, old = demo.robot;
  let x, y;
  do { x = 25 + r.uniform() * (WIDTH - 50); y = 20 + r.uniform() * (HEIGHT - 40); }
  while (Math.hypot(x - old.x, y - old.y) < 70);
  old.x = x; old.y = y; old.heading = r.uniform() * 2 * Math.PI;
}
$("#kidnap").addEventListener("click", () => { kidnap(); drawDemo(); });

/* settings panel */
const ui = {
  n: $("#setN"), terrain: $("#setTerrain"), noise: $("#setNoise"),
  compass: $("#setCompass"), explorers: $("#setExplorers"), show: $("#setShow"),
};
function syncSettingsUI() {
  const s = demo.settings;
  ui.n.value = String(s.n); ui.terrain.value = s.terrain; ui.noise.value = String(s.noise);
  ui.compass.checked = s.compass; ui.explorers.checked = s.explorers; ui.show.checked = s.show;
}
ui.n.addEventListener("change", () => { demo.settings.n = +ui.n.value; buildFilter(); drawDemo(); });
ui.terrain.addEventListener("change", () => { demo.settings.terrain = ui.terrain.value; fullReset(); drawDemo(); });
ui.noise.addEventListener("change", () => { demo.settings.noise = +ui.noise.value; demo.filter.assumedNoise = demo.settings.noise; drawDemo(); });
ui.compass.addEventListener("change", () => { demo.settings.compass = ui.compass.checked; buildFilter(); drawDemo(); });
ui.explorers.addEventListener("change", () => { demo.settings.explorers = ui.explorers.checked; demo.filter.explorers = ui.explorers.checked ? 0.01 : 0; });
ui.show.addEventListener("change", () => { demo.settings.show = ui.show.checked; drawDemo(); });

/* "Try it" buttons further down the page set up an experiment and scroll back to the demo. */
const PRESETS = {
  default: {},
  few: { n: 100 },
  flat: { terrain: "flat" },
  gentle: { terrain: "gentle" },
  overconfident: { noise: 0.5 },
  nocompass: { compass: false, n: 5000 },
  hide: { show: false },
  kidnap: { explorers: false },
  explorers: { explorers: true },
};
function applyPreset(name) {
  setAuto(false);
  const wantTerrain = PRESETS[name].terrain || DEFAULTS.terrain;
  const terrainChanged = wantTerrain !== demo.settings.terrain;
  demo.settings = { ...DEFAULTS, ...PRESETS[name] };
  syncSettingsUI();
  if (terrainChanged) { buildTerrain(); placeRobotRandomly(); }
  buildFilter();
  if (name === "kidnap" || name === "explorers") {
    // let the filter find the robot first (quietly, without explorers), then kidnap it
    const explorers = demo.filter.explorers;
    demo.filter.explorers = 0;
    for (let i = 0; i < 120; i++) {
      act(...autoMove(), false);
      const e = demo.filter.estimate();
      if (i > 15 && e.spread < 5 && Math.hypot(e.x - demo.robot.x, e.y - demo.robot.y) < 5) break;
    }
    demo.filter.explorers = explorers;
    kidnap();
  }
  if (name === "hide") $("#settings").open = true;
  drawDemo();
  noteInteraction();
  $("#demo").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  demo.canvas.focus({ preventScroll: true });
  if (name !== "hide" && name !== "default") setAuto(true);
}
document.querySelectorAll("[data-preset]").forEach(b => b.addEventListener("click", () => applyPreset(b.dataset.preset)));

/* =====================================================================
   WIDGET 1: a robot on a loop track (one dimension)
   ===================================================================== */
const w1 = {
  canvas: $("#w1c"), L: 200, n: 60, move: 12, sensor: 1.5, assumed: 2.5, jitter: 1.0,
  rng: makeRng(2024), timer: null,
};
const trackHeight = s => {
  const a = 2 * Math.PI * s / 200;
  return 40 + 16 * Math.sin(2 * a + 0.3) + 9 * Math.sin(5 * a + 1.3) + 5 * Math.sin(9 * a + 2.1);
};
const wrapTrack = s => ((s % 200) + 200) % 200;
function w1Reset() {
  const r = w1.rng;
  w1.robot = 37;
  w1.p = Array.from({ length: w1.n }, (_, i) => (i + r.uniform()) * w1.L / w1.n);
  w1.w = new Array(w1.n).fill(1 / w1.n);
  w1.weighted = false;
  w1.reading = null;
  w1.phase = 0;            // next phase: 0 weigh, 1 resample, 2 move
  w1.round = 0;
  w1.caption = "60 guesses spread evenly around the loop: the robot could be anywhere. Press <b>Next step</b> to read the altimeter.";
  w1Draw();
}
function w1Next() {
  const r = w1.rng;
  if (w1.phase === 0) {
    w1.reading = trackHeight(w1.robot) + r.normal(w1.sensor);
    let sum = 0;
    w1.w = w1.p.map(s => { const e = (w1.reading - trackHeight(s)) / w1.assumed; return Math.exp(-0.5 * e * e); });
    for (const v of w1.w) sum += v;
    w1.w = w1.w.map(v => v / sum);
    w1.weighted = true;
    const strong = w1.w.filter(v => v > 0.5 / w1.n).length;
    w1.caption = `<b>Weigh.</b> The altimeter reads ${fmt(w1.reading)} m (dashed line). Guesses where the track is about that high get tall sticks; the others shrink. ${w1.round === 0 ? "The track crosses this height in several places, so several groups of guesses stay tall." : strong > w1.n / 3 ? "Most guesses still fit." : "Only the guesses that also fitted the earlier readings stay tall."}`;
  } else if (w1.phase === 1) {
    const n = w1.n, start = r.uniform() / n, next = [];
    let i = 0, cum = w1.w[0];
    for (let k = 0; k < n; k++) {
      const tooth = start + k / n;
      while (tooth > cum && i < n - 1) { i++; cum += w1.w[i]; }
      next.push(wrapTrack(w1.p[i] + r.normal(w1.jitter)));
    }
    w1.p = next;
    w1.w = new Array(n).fill(1 / n);
    w1.weighted = false;
    w1.caption = "<b>Resample and jitter.</b> Tall-stick guesses were copied, short ones dropped, and the copies were shaken a little. All sticks are equal again: the information now lives in where the guesses crowd.";
  } else {
    const r2 = w1.rng;
    w1.robot = wrapTrack(w1.robot + w1.move + r2.normal(NOISE.step * w1.move));
    w1.p = w1.p.map(s => wrapTrack(s + w1.move + r2.normal(NOISE.step * w1.move)));
    w1.reading = null;
    w1.round++;
    w1.caption = `<b>Move (predict).</b> The robot drove about ${w1.move} m to the right. Every guess moved ${w1.move} m too, each with its own small error, so each group smears out a little.`;
  }
  w1.phase = (w1.phase + 1) % 3;
  w1Draw();
}
function w1Draw() {
  const { ctx, w, h } = fit(w1.canvas, w1.canvas.clientWidth < 520 ? 1.5 : 2.3);
  ctx.clearRect(0, 0, w, h);
  const padL = 8, padR = 8, X = s => padL + s / w1.L * (w - padL - padR);
  const topY = 14, profH = h * 0.56, Y = z => topY + profH - (z - 5) / 75 * profH;
  const base = h - 22, stickMax = h - 22 - (topY + profH) - 14;
  // profile
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) { const s = i; i ? ctx.lineTo(X(s), Y(trackHeight(s))) : ctx.moveTo(X(s), Y(trackHeight(s))); }
  ctx.lineTo(X(200), topY + profH); ctx.lineTo(X(0), topY + profH); ctx.closePath();
  ctx.fillStyle = css("--surface-2"); ctx.fill();
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) { i ? ctx.lineTo(X(i), Y(trackHeight(i))) : ctx.moveTo(X(i), Y(trackHeight(i))); }
  ctx.strokeStyle = css("--ink-2"); ctx.lineWidth = 2; ctx.stroke();
  // reading line
  if (w1.reading != null) {
    ctx.save();
    ctx.setLineDash([5, 4]); ctx.strokeStyle = css("--robot"); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(0), Y(w1.reading)); ctx.lineTo(X(200), Y(w1.reading)); ctx.stroke();
    ctx.restore();
    ctx.font = "11px " + css("--font-mono"); ctx.textAlign = "left";
    const label = `reading ${fmt(w1.reading)} m`, lw = ctx.measureText(label).width;
    ctx.fillStyle = css("--surface"); ctx.globalAlpha = 0.85;
    ctx.fillRect(X(0), Y(w1.reading) - 18, lw + 8, 15);
    ctx.globalAlpha = 1; ctx.fillStyle = css("--ink-2");
    ctx.fillText(label, X(0) + 4, Y(w1.reading) - 6);
  }
  // sticks
  ctx.strokeStyle = css("--rule"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(0), base + 0.5); ctx.lineTo(X(200), base + 0.5); ctx.stroke();
  const maxW = Math.max(...w1.w);
  ctx.fillStyle = css("--guess");
  w1.p.forEach((s, i) => {
    const hgt = w1.weighted ? 3 + (stickMax - 3) * w1.w[i] / maxW : stickMax * 0.4;
    ctx.fillRect(X(s) - 1.25, base - hgt, 2.5, hgt);
  });
  // robot
  const rx = X(w1.robot), ry = Y(trackHeight(w1.robot));
  ctx.save(); ctx.setLineDash([2, 3]); ctx.strokeStyle = css("--robot"); ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(rx, base); ctx.stroke(); ctx.restore();
  ctx.fillStyle = css("--robot"); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(rx, ry, 6.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  // axis labels
  ctx.font = "10px " + css("--font-mono"); ctx.fillStyle = css("--muted");
  ctx.textAlign = "left"; ctx.fillText("0 m", X(0), h - 6);
  ctx.textAlign = "center"; ctx.fillText("position along the track", X(100), h - 6);
  ctx.textAlign = "right"; ctx.fillText("200 m", X(200), h - 6);
  ctx.textAlign = "left"; ctx.fillText("guesses (stick height = score)", X(0), base - stickMax - 2);
  ctx.textAlign = "right"; ctx.fillText("ground height along the track", X(200), topY - 2);
  // phase pills and caption
  for (let k = 0; k < 3; k++) $("#w1p" + k).classList.toggle("on", k === (w1.phase + 2) % 3 && (w1.round > 0 || w1.phase !== 0));
  $("#w1cap").innerHTML = w1.caption;
}
$("#w1next").addEventListener("click", w1Next);
$("#w1reset").addEventListener("click", () => { w1Play(false); w1Reset(); });
function w1Play(on) {
  clearInterval(w1.timer); w1.timer = null;
  $("#w1play").setAttribute("aria-pressed", on ? "true" : "false");
  $("#w1play").textContent = on ? "Pause" : "Play";
  if (on) w1.timer = setInterval(w1Next, reduceMotion ? 1600 : 1100);
}
$("#w1play").addEventListener("click", () => w1Play(!w1.timer));

/* =====================================================================
   WIDGET 2: moving makes you less sure
   ===================================================================== */
const w2 = { canvas: $("#w2c"), n: 200, Wm: 100, Hm: 45, d: 6, turn: 30 * DEG, rng: makeRng(99) };
function w2Reset() {
  w2.path = [{ x: 10, y: 22.5, h: 0 }];
  w2.p = Array.from({ length: w2.n }, () => ({ x: 10, y: 22.5, h: 0 }));
  w2Draw();
}
function w2Move(d, turn) {
  const k = +$("#w2noise").value / 10;           // 1 = the default 10 % wheel error
  const compass = $("#w2compass").checked, r = w2.rng;
  const last = w2.path[w2.path.length - 1];
  const ideal = { h: last.h + turn };
  ideal.x = last.x + d * Math.cos(ideal.h); ideal.y = last.y + d * Math.sin(ideal.h);
  w2.path.push(ideal);
  for (const q of w2.p) {
    let heading;
    if (compass) {
      q.h = ideal.h;
      heading = ideal.h + r.normal(k * (NOISE.compass + NOISE.drift * d));
    } else {
      q.h += turn + r.normal(k * (NOISE.turn * Math.abs(turn) + NOISE.drift * d));
      heading = q.h;
    }
    const travelled = d + r.normal(k * NOISE.step * d);
    q.x += travelled * Math.cos(heading); q.y += travelled * Math.sin(heading);
  }
  w2Draw();
}
function w2Draw() {
  const { ctx, w, h } = fit(w2.canvas, w2.canvas.clientWidth < 520 ? 1.6 : 2.2);
  const s = Math.min(w / w2.Wm, h / w2.Hm), ox = (w - w2.Wm * s) / 2, oy = (h - w2.Hm * s) / 2;
  const X = x => ox + x * s, Y = y => oy + y * s;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  for (let x = 0; x <= w2.Wm; x += 10) { ctx.beginPath(); ctx.moveTo(X(x) + 0.5, Y(0)); ctx.lineTo(X(x) + 0.5, Y(w2.Hm)); ctx.stroke(); }
  for (let y = 0; y <= w2.Hm; y += 10) { ctx.beginPath(); ctx.moveTo(X(0), Y(y) + 0.5); ctx.lineTo(X(w2.Wm), Y(y) + 0.5); ctx.stroke(); }
  ctx.save();
  ctx.setLineDash([5, 4]); ctx.strokeStyle = css("--ink-2"); ctx.lineWidth = 1.5;
  ctx.beginPath();
  w2.path.forEach((p, i) => i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y)));
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = css("--guess"); ctx.globalAlpha = 0.8;
  for (const q of w2.p) ctx.fillRect(X(q.x) - 1.5, Y(q.y) - 1.5, 3, 3);
  ctx.globalAlpha = 1;
  const last = w2.path[w2.path.length - 1];
  drawRobot(ctx, X(last.x), Y(last.y), last.h, 0.8);
  ctx.font = "10px " + css("--font-mono"); ctx.fillStyle = css("--muted"); ctx.textAlign = "right";
  ctx.fillText("grid: 10 m", X(w2.Wm) - 4, Y(w2.Hm) - 5);
  $("#w2noiseV").textContent = $("#w2noise").value + "%";
}
$("#w2fwd").addEventListener("click", () => w2Move(w2.d, 0));
$("#w2left").addEventListener("click", () => w2Move(0, -w2.turn));
$("#w2right").addEventListener("click", () => w2Move(0, w2.turn));
$("#w2reset").addEventListener("click", w2Reset);
$("#w2noise").addEventListener("input", () => { $("#w2noiseV").textContent = $("#w2noise").value + "%"; });
$("#w2compass").addEventListener("change", w2Reset);

/* =====================================================================
   WIDGET 3: sensing makes you more sure (the bell curve on the map)
   ===================================================================== */
const w3 = {
  map: $("#w3map"), bell: $("#w3bell"), tip: $("#w3tip"),
  terrain: makeTerrain({ seed: 7 }), rng: makeRng(5), x: 150, y: 70, hover: null,
};
w3.overlay = document.createElement("canvas");
w3.overlay.width = WIDTH; w3.overlay.height = HEIGHT;
function w3Read() {
  w3.ground = heightAt(w3.terrain, w3.x, w3.y);
  w3.reading = w3.ground + w3.rng.normal(NOISE.sensor);
  w3Draw();
}
const w3Sigma = () => +$("#w3sigma").value;
function w3Draw() {
  const sigma = w3Sigma();
  $("#w3sigmaV").textContent = sigma.toFixed(1) + " m";
  // score of every map cell
  const octx = w3.overlay.getContext("2d");
  const img = octx.createImageData(WIDTH, HEIGHT);
  const [r, g, b] = hexToRgb(css("--guess"));
  for (let i = 0; i < WIDTH * HEIGHT; i++) {
    const e = (w3.reading - w3.terrain.z[i]) / sigma, score = Math.exp(-0.5 * e * e);
    img.data[4 * i] = r; img.data[4 * i + 1] = g; img.data[4 * i + 2] = b;
    img.data[4 * i + 3] = Math.round(235 * score);
  }
  octx.putImageData(img, 0, 0);
  const { ctx, w, h } = fit(w3.map, 1.5);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(w3.image, 0, 0, w, h);
  ctx.drawImage(w3.overlay, 0, 0, w, h);
  drawRobot(ctx, w3.x * w / WIDTH, w3.y * h / HEIGHT, -Math.PI / 2, 0.9);
  $("#w3info").innerHTML = `ground under the robot: ${fmt(w3.ground)} m<br>altimeter reads: <b>${fmt(w3.reading)} m</b>`;
  w3DrawBell();
}
function w3DrawBell() {
  const { ctx, w, h } = fit(w3.bell);
  const sigma = w3Sigma(), z = w3.reading;
  const pad = { l: 40, r: 8, t: 18, b: 26 }, iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const X = v => pad.l + v / 100 * iw, Y = v => pad.t + ih - v * ih;
  ctx.clearRect(0, 0, w, h);
  ctx.font = "10px " + css("--font-mono"); ctx.lineWidth = 1;
  ctx.strokeStyle = css("--grid"); ctx.fillStyle = css("--muted");
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const v of [0, 0.5, 1]) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(v) + 0.5); ctx.lineTo(w - pad.r, Y(v) + 0.5); ctx.stroke();
    ctx.fillText(v.toFixed(1), pad.l - 4, Y(v));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
  for (const v of [0, 25, 50, 75, 100]) ctx.fillText(v, X(v), h - 13);
  ctx.fillText("ground height at a guess (m)", pad.l + iw / 2, h - 1);
  ctx.save(); ctx.translate(10, pad.t + ih / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("score", 0, 0); ctx.restore();
  const score = v => Math.exp(-0.5 * ((z - v) / sigma) ** 2);
  ctx.beginPath(); ctx.moveTo(X(0), Y(0));
  for (let v = 0; v <= 100; v += 0.25) ctx.lineTo(X(v), Y(score(v)));
  ctx.lineTo(X(100), Y(0)); ctx.closePath();
  ctx.fillStyle = css("--guess-wash"); ctx.fill();
  ctx.beginPath();
  for (let v = 0; v <= 100; v += 0.25) v ? ctx.lineTo(X(v), Y(score(v))) : ctx.moveTo(X(v), Y(score(v)));
  ctx.strokeStyle = css("--guess"); ctx.lineWidth = 2; ctx.stroke();
  ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = css("--robot"); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(X(z), Y(0)); ctx.lineTo(X(z), pad.t - 4); ctx.stroke(); ctx.restore();
  ctx.fillStyle = css("--ink-2"); ctx.textAlign = z > 70 ? "right" : "left";
  ctx.fillText(`reading ${fmt(z)} m`, X(z) + (z > 70 ? -5 : 5), pad.t - 6);
  if (w3.hover != null) {
    const v = Math.max(0, Math.min(100, w3.hover)), sc = score(v);
    ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(v) + 0.5, Y(0)); ctx.lineTo(X(v) + 0.5, Y(1)); ctx.stroke();
    ctx.fillStyle = css("--guess"); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(v), Y(sc), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    w3.tip.innerHTML = `ground ${v.toFixed(1)} m<br>score ${sc < 0.001 ? sc.toExponential(1) : sc.toFixed(3)}`;
    w3.tip.style.left = Math.max(55, Math.min(w - 55, X(v))) + "px";
    w3.tip.style.top = Y(Math.max(sc, 0.25)) + "px";
    w3.tip.classList.add("on");
  } else w3.tip.classList.remove("on");
}
function w3Place(ev) {
  const r = w3.map.getBoundingClientRect();
  w3.x = Math.max(0, Math.min(WIDTH - 1, (ev.clientX - r.left) / r.width * WIDTH));
  w3.y = Math.max(0, Math.min(HEIGHT - 1, (ev.clientY - r.top) / r.height * HEIGHT));
  w3Read();
}
let w3Dragging = false;
w3.map.addEventListener("pointerdown", ev => { w3Dragging = true; w3.map.setPointerCapture(ev.pointerId); w3Place(ev); });
w3.map.addEventListener("pointermove", ev => { if (w3Dragging) w3Place(ev); });
w3.map.addEventListener("pointerup", () => { w3Dragging = false; });
w3.bell.addEventListener("pointermove", ev => {
  const r = w3.bell.getBoundingClientRect();
  w3.hover = (ev.clientX - r.left - 40) / (r.width - 48) * 100;
  w3DrawBell();
});
w3.bell.addEventListener("pointerleave", () => { w3.hover = null; w3DrawBell(); });
$("#w3sigma").addEventListener("input", w3Draw);
$("#w3read").addEventListener("click", w3Read);

/* =====================================================================
   WIDGET 4: resampling with a comb
   ===================================================================== */
const w4 = { canvas: $("#w4c"), n: 8, rng: makeRng(31), teeth: null, hover: null };
const LETTERS = "ABCDEFGH";
function w4NewWeights() {
  const raw = Array.from({ length: w4.n }, () => 0.12 + Math.exp(0.9 * w4.rng.normal()));
  const sum = raw.reduce((a, b) => a + b, 0);
  w4.w = raw.map(v => v / sum);
  w4.teeth = null;
  w4.picks = null;
  $("#w4cap").innerHTML = "Eight guesses after weighing. Each one's segment is as long as its share of the total score. Press <b>drop the comb</b> to pick the new set of eight.";
  w4Draw();
}
function w4Resample() {
  const n = w4.n, u = w4.rng.uniform() / n;
  w4.teeth = Array.from({ length: n }, (_, k) => u + k / n);
  w4.picks = w4.teeth.map(t => { let c = 0; for (let i = 0; i < n; i++) { c += w4.w[i]; if (t <= c) return i; } return n - 1; });
  const counts = new Array(n).fill(0);
  w4.picks.forEach(i => counts[i]++);
  const best = counts.indexOf(Math.max(...counts));
  const none = LETTERS.split("").filter((_, i) => counts[i] === 0);
  $("#w4cap").innerHTML = `New set: <b>${w4.picks.map(i => LETTERS[i]).join(" ")}</b>. ` +
    `${LETTERS[best]} held ${Math.round(100 * w4.w[best])}% of the score and got ${counts[best]} of the 8 teeth. ` +
    (none.length ? `${none.join(", ")} ${none.length === 1 ? "was" : "were"} too weak to be picked.` : "Every guess was picked once.");
  w4Draw();
}
function w4Draw() {
  const { ctx, w, h } = fit(w4.canvas, w4.canvas.clientWidth < 520 ? 2.1 : 3.4);
  ctx.clearRect(0, 0, w, h);
  const padX = 10, rulerY = h * 0.42, rulerH = Math.min(46, h * 0.28), iw = w - 2 * padX;
  const X = f => padX + f * iw;
  const counts = new Array(w4.n).fill(0);
  if (w4.picks) w4.picks.forEach(i => counts[i]++);
  let c = 0;
  const shades = [css("--guess"), isDark() ? "#2a5f9e" : "#7fb0ea"];
  ctx.font = "600 13px " + css("--font-body");
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  w4.segments = [];
  w4.w.forEach((wt, i) => {
    const x0 = X(c), x1 = X(c + wt);
    w4.segments.push([x0, x1]);
    ctx.fillStyle = shades[i % 2];
    ctx.globalAlpha = w4.picks && counts[i] === 0 ? 0.3 : 1;
    ctx.beginPath();
    ctx.roundRect(x0 + 1, rulerY, Math.max(0.5, x1 - x0 - 2), rulerH, 4);
    ctx.fill();
    ctx.globalAlpha = 1;
    if (x1 - x0 > 16) { ctx.fillStyle = i % 2 ? (isDark() ? "#fff" : "#10233d") : "#fff"; ctx.fillText(LETTERS[i], (x0 + x1) / 2, rulerY + rulerH / 2); }
    ctx.fillStyle = css("--muted"); ctx.font = "10px " + css("--font-mono");
    if (x1 - x0 > 26) ctx.fillText(Math.round(100 * wt) + "%", (x0 + x1) / 2, rulerY + rulerH + 12);
    if (w4.picks && x1 - x0 > 20) { ctx.fillStyle = css("--ink"); ctx.fillText("×" + counts[i], (x0 + x1) / 2, rulerY + rulerH + 27); }
    ctx.font = "600 13px " + css("--font-body");
    c += wt;
  });
  // comb
  const combY = rulerY - 30;
  ctx.strokeStyle = css("--ink"); ctx.fillStyle = css("--ink"); ctx.lineWidth = 2;
  if (w4.teeth) {
    ctx.beginPath(); ctx.moveTo(X(w4.teeth[0]), combY); ctx.lineTo(X(w4.teeth[w4.n - 1]), combY); ctx.stroke();
    for (const t of w4.teeth) {
      ctx.beginPath(); ctx.moveTo(X(t), combY); ctx.lineTo(X(t), rulerY + rulerH * 0.5); ctx.stroke();
      ctx.beginPath(); ctx.arc(X(t), rulerY + rulerH * 0.5, 3, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.font = "10px " + css("--font-mono"); ctx.fillStyle = css("--muted"); ctx.textAlign = "left";
    ctx.fillText("the comb: 8 evenly spaced teeth, shifted at random", X(0), combY - 12);
  } else {
    ctx.font = "10px " + css("--font-mono"); ctx.fillStyle = css("--muted"); ctx.textAlign = "left";
    ctx.fillText("the ruler: each guess's share of the total score", X(0), rulerY - 10);
  }
  if (w4.hover != null) {
    const i = w4.hover, [x0, x1] = w4.segments[i];
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2;
    ctx.strokeRect(x0 + 1, rulerY - 1, Math.max(1, x1 - x0 - 2), rulerH + 2);
  }
}
w4.canvas.addEventListener("pointermove", ev => {
  const r = w4.canvas.getBoundingClientRect(), x = ev.clientX - r.left;
  const i = (w4.segments || []).findIndex(([a, b]) => x >= a && x < b);
  w4.hover = i >= 0 ? i : null;
  w4.canvas.title = i >= 0 ? `${LETTERS[i]}: ${(100 * w4.w[i]).toFixed(1)}% of the score` + (w4.picks ? `, ${w4.picks.filter(p => p === i).length} copies` : "") : "";
  w4Draw();
});
w4.canvas.addEventListener("pointerleave", () => { w4.hover = null; w4Draw(); });
$("#w4go").addEventListener("click", w4Resample);
$("#w4new").addEventListener("click", w4NewWeights);

/* =====================================================================
   start-up, resizing and theme changes
   ===================================================================== */
function drawAll() { drawDemo(); w1Draw(); w2Draw(); w3Draw(); w4Draw(); }
function rebuildImages() { demo.image = terrainImage(demo.terrain); w3.image = terrainImage(w3.terrain); }

if (window.matchMedia("(pointer: coarse)").matches) $("#mapOverlay").textContent = "drive with the buttons below";
fullReset();
w3.image = terrainImage(w3.terrain);
w1Reset();
w2Reset();
w3Read();
w4NewWeights();
drawAll();

let resizeTimer = null;
new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawAll, 60); }).observe(document.body);
const onTheme = () => { rebuildImages(); drawAll(); };
darkQuery.addEventListener("change", onTheme);
new MutationObserver(onTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
if (document.fonts) document.fonts.ready.then(drawAll);
})();
