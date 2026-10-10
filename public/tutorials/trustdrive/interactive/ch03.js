/* TrustDrive chapter 03, Perception and rendering: the live demos.
   render() is the notebook's renderer ported line by line (the same projection, rounding,
   blur and features), so without noise it reproduces the notebook's numbers exactly.
   Needs ../kit/kit.js and ../kit/stats.js. */
(() => {
"use strict";
const { $, $$, css, fmt, caption, fit, frame, plotLine, alpha, onRedraw, narrowOf, seg, clamp } = window.Kit;
const { makeRng } = window.Stats;
const H = 64, W = 64, FOCAL = 55, CAM_H = 1.25, HORIZON = 0.34 * 64, LANE = 1.75;
const ODD = { kappa: 0.022, brightness: 0.45, blur: 1.6, occlusion: 0.35 };
const pct = x => `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(0)}%`;

/* ---------------- the renderer ---------------- */
/* Python's round(): halves go to the even neighbour */
function roundHalfEven(x) {
  const f = Math.floor(x), d = x - f;
  return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1);
}
const S = Array.from({ length: 400 }, (_, i) => (i === 399 ? 45 : 3 + i * (42 / 399)));   // np.linspace(3, 45, 400)
const FADE = S.map(s => clamp(1 - (s - 3) / 42, 0.2, 1));                                   // far markings dimmer
const project = (s, y) => [W / 2 + FOCAL * y / s, HORIZON + FOCAL * CAM_H / s];
function gaussianBlur(img, sigma) {
  if (sigma <= 1e-3) return img;
  const r = Math.max(1, roundHalfEven(3 * sigma));
  const k = Array.from({ length: 2 * r + 1 }, (_, i) => Math.exp(-0.5 * ((i - r) / sigma) ** 2));
  const ks = k.reduce((a, b) => a + b, 0);
  for (let i = 0; i < k.length; i++) k[i] /= ks;
  const tmp = new Float64Array(H * W), out = new Float64Array(H * W);
  for (let c = 0; c < W; c++) for (let rr = 0; rr < H; rr++) {          // along the columns first, as np.apply_along_axis(..., 0, ...)
    let acc = 0;
    for (let t = -r; t <= r; t++) { const q = rr + t; if (q >= 0 && q < H) acc += k[t + r] * img[q * W + c]; }
    tmp[rr * W + c] = acc;
  }
  for (let rr = 0; rr < H; rr++) for (let c = 0; c < W; c++) {          // then along the rows
    let acc = 0;
    for (let t = -r; t <= r; t++) { const q = c + t; if (q >= 0 && q < W) acc += k[t + r] * tmp[rr * W + q]; }
    out[rr * W + c] = acc;
  }
  return out;
}
function render({ ey, ep, kap, b = 1, blur = 0, occ = 0, noise = 0, seed = 7 }) {
  let img = new Float64Array(H * W).fill(0.15);                          // dark asphalt
  const yc = S.map(s => ey + ep * s - 0.5 * kap * s * s);               // lane centre, to the right (chapter 01's signs)
  for (const [off, dashed] of [[-LANE, false], [LANE, false], [0, true]]) {
    for (let i = 0; i < S.length; i++) {
      if (dashed && Math.trunc(S[i]) % 3 === 0) continue;
      const [u, v] = project(S[i], yc[i] + off), rr = roundHalfEven(v), cc = roundHalfEven(u);
      if (rr >= 0 && rr < H && cc >= 0 && cc < W)
        for (const dc of [-1, 0, 1]) { const c = cc + dc; if (c >= 0 && c < W) img[rr * W + c] = Math.max(img[rr * W + c], 0.9 * FADE[i]); }
    }
  }
  for (let k = 0; k < img.length; k++) img[k] *= b;
  if (occ > 0) {
    const top = Math.trunc(HORIZON), rows = Math.trunc(occ * (H - top));
    for (let rr = top; rr < Math.min(H, top + rows); rr++) for (let c = 0; c < W; c++) img[rr * W + c] = 0.05;
  }
  img = gaussianBlur(img, blur);
  const rng = noise > 0 ? makeRng(seed) : null;
  for (let k = 0; k < img.length; k++) img[k] = clamp(img[k] + (rng ? rng.normal(0, noise) : 0), 0, 1);
  return img;
}
const brightnessFeat = img => img.reduce((a, v) => a + v, 0) / img.length;
function sharpnessFeat(img) {
  let gx = 0, gy = 0;
  for (let rr = 0; rr < H; rr++) for (let c = 0; c < W - 1; c++) gx += (img[rr * W + c + 1] - img[rr * W + c]) ** 2;
  for (let rr = 0; rr < H - 1; rr++) for (let c = 0; c < W; c++) gy += (img[(rr + 1) * W + c] - img[rr * W + c]) ** 2;
  return gx / (H * (W - 1)) + gy / ((H - 1) * W);
}
/* draw a 64 x 64 image, crisp, filling the canvas */
const off = document.createElement("canvas"); off.width = W; off.height = H;
function drawImage(canvas, img) {
  const { ctx, w, h } = fit(canvas, 1);
  const oc = off.getContext("2d"), data = oc.createImageData(W, H);
  for (let k = 0; k < img.length; k++) { const g = Math.round(255 * img[k]); data.data.set([g, g, g, 255], 4 * k); }
  oc.putImageData(data, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, w, h);
  return { ctx, w, h, px: (u, v) => [(u + 0.5) * w / W, (v + 0.5) * h / H] };    // image coordinates (pixel centres) to canvas
}

/* =====================================================================
   THE DEMO: THE CAR'S CAMERA
   ===================================================================== */
const PRESETS = {
  clear: { cK: 0.012, cEy: 0.3, cEp: 0.02, cB: 1, cBl: 0, cO: 0, cN: 0.02 },
  fog: { cK: 0.012, cEy: 0.3, cEp: 0.02, cB: 0.6, cBl: 2.6, cO: 0, cN: 0.03 },
  occ: { cK: 0.012, cEy: 0.3, cEp: 0.02, cB: 1, cBl: 0, cO: 0.45, cN: 0.02 },
  ood: { cK: 0.05, cEy: 0.3, cEp: 0.02, cB: 1, cBl: 0, cO: 0, cN: 0.02 },
};
const cam = { img: null, ref: null, p: null };
const read = () => ({ kap: +$("#cK").value, ey: +$("#cEy").value, ep: +$("#cEp").value, b: +$("#cB").value, blur: +$("#cBl").value, occ: +$("#cO").value, noise: +$("#cN").value });
function updateCam() {
  const p = cam.p = read();
  cam.img = render(p);
  cam.ref = render({ ey: p.ey, ep: p.ep, kap: 0.012, noise: p.noise });     // the same car on a clear, gentle road
  $("#cKV").textContent = `${fmt(p.kap, 3)} per m`;
  $("#cEyV").textContent = `${fmt(p.ey, 2)} m`;
  $("#cEpV").textContent = `${fmt(p.ep, 3)} rad`;
  $("#cBV").textContent = fmt(p.b, 2);
  $("#cBlV").textContent = `${fmt(p.blur, 1)} px`;
  $("#cOV").textContent = `${Math.round(p.occ * 100)}% of the road below the horizon`;
  $("#cNV").textContent = fmt(p.noise, 3);
  const match = Object.entries(PRESETS).find(([, v]) => Object.entries(v).every(([id, x]) => Math.abs(+$("#" + id).value - x) < 1e-9));
  $$("#demo [data-preset]").forEach(b => b.setAttribute("aria-pressed", match && b.dataset.preset === match[0] ? "true" : "false"));
  drawCam(); camStatus();
}
function drawCam() {
  const { ctx, w, h, px } = drawImage($("#camC"), cam.img);
  if ($("#cGuide").checked) {
    const [, hy] = px(0, HORIZON), [vx] = px(W / 2 + FOCAL * cam.p.ep, 0);   // a straight road's lines meet at u = c_x + f e_psi (exercise 1)
    ctx.strokeStyle = css("--path"); ctx.lineWidth = 1.5; seg(ctx, 0, hy, w, hy);
    ctx.fillStyle = css("--geom"); ctx.beginPath(); ctx.arc(vx, hy, 4.5, 0, 2 * Math.PI); ctx.fill();
    ctx.font = "600 11px " + css("--font-ui"); ctx.textAlign = "left"; ctx.fillStyle = css("--path");
    ctx.fillText("horizon", 4, hy - 5);
  }
  drawTop();
  const br = brightnessFeat(cam.img), sh = sharpnessFeat(cam.img), br0 = brightnessFeat(cam.ref), sh0 = sharpnessFeat(cam.ref);
  $("#rBr").textContent = fmt(br, 3);
  $("#rSh").textContent = fmt(sh, 4);
  $("#rCh").textContent = `${pct(br / br0 - 1)}, ${pct(sh / sh0 - 1)}`;
  const out = violations(cam.p);
  $("#rOdd").innerHTML = out.length ? `<span class="verdict bad">outside: ${out.join(", ")}</span>` : `<span class="verdict ok">inside</span>`;
}
function violations(p) {
  const out = [];
  if (Math.abs(p.kap) > ODD.kappa + 1e-12) out.push("curvature");
  if (p.b < ODD.brightness - 1e-12) out.push("darkness");
  if (p.blur > ODD.blur + 1e-12) out.push("blur");
  if (p.occ > ODD.occlusion + 1e-12) out.push("occlusion");
  return out;
}
/* the lane seen from above, in the car's frame: forward up, right to the right (sideways x 4) */
function drawTop() {
  const p = cam.p, cv = $("#topC"), { ctx, w, h } = fit(cv, 1);
  const latRange = 15, s0 = -5, s1 = 48;                                   // 30 m across, 53 m along: sideways x 1.8
  const X = y => w / 2 + y / latRange * (w / 2), Y = s => h - (s - s0) / (s1 - s0) * h;
  const half = (W / 2) / FOCAL;                                            // the image's edges: |y| = 32/55 s
  ctx.fillStyle = alpha("--path", 0.1); ctx.beginPath();                   // what the camera renders: 3 to 45 m, inside its field of view
  ctx.moveTo(X(-half * 3), Y(3)); ctx.lineTo(X(-half * 45), Y(45)); ctx.lineTo(X(half * 45), Y(45)); ctx.lineTo(X(half * 3), Y(3)); ctx.closePath(); ctx.fill();
  ctx.font = "600 10px " + css("--font-ui"); ctx.fillStyle = css("--muted"); ctx.textAlign = "left";
  ctx.fillText("camera view, 3 to 45 m", 4, Y(45) - 4);
  const ss = Array.from({ length: 107 }, (_, i) => i * 0.5);
  const yroad = s => p.ey + p.ep * s - 0.5 * p.kap * s * s;
  [[-LANE, []], [LANE, []], [0, [6, 6]]].forEach(([o, dash]) => {
    ctx.strokeStyle = alpha("--ink", 0.7); ctx.lineWidth = 1.6; ctx.setLineDash(dash); ctx.beginPath();
    ss.forEach((s, i) => { const x = X(yroad(s) + o), y = Y(s); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
  });
  ctx.setLineDash([]);
  // the car: the camera at s = 0 on its front, pointing up
  ctx.fillStyle = alpha("--ink", 0.12); ctx.strokeStyle = alpha("--ink", 0.6); ctx.lineWidth = 1;
  ctx.fillRect(X(-0.9), Y(0.2), X(0.9) - X(-0.9), Y(-4.2) - Y(0.2)); ctx.strokeRect(X(-0.9), Y(0.2), X(0.9) - X(-0.9), Y(-4.2) - Y(0.2));
  ctx.fillStyle = css("--geom"); ctx.beginPath(); ctx.arc(X(0), Y(0), 3.5, 0, 2 * Math.PI); ctx.fill();
}
function camStatus() {
  const p = cam.p, out = violations(p);
  const br = brightnessFeat(cam.img) / brightnessFeat(cam.ref) - 1, sh = sharpnessFeat(cam.img) / sharpnessFeat(cam.ref) - 1;
  const visible = out.filter(v => v !== "curvature");
  const change = `brightness ${pct(br)} and sharpness ${pct(sh)} compared with a clear, gentle road`;
  let text;
  if (!out.length) text = `All conditions are inside the operating domain (chapter 06's limits): this is the kind of image the perception was trained on.` +
    (Math.abs(br) > 0.05 || Math.abs(sh) > 0.05 ? ` Its features have moved, though not past the limits: ${change}.` : "");
  else if (!visible.length) text = `Outside the operating domain: this bend, [[\\kappa = ${fmt(p.kap, 3)}]], is sharper than the domain's 0.022 and than anything in training. Yet the image is clean: ${change}. <b>Nothing in the pixels says so.</b> Chapter 05's innovation test, which checks the perception against the car's dynamics, is what catches it.`;
  else text = `Outside the operating domain: ${out.join(" and ")}. ` + (out.includes("curvature") ? "The degradation shows in the image, but the bend would not on its own: " : "And the image shows it: ") + `${change}.`;
  caption($("#camStatus"), text);
}
$$("#demo [data-preset]").forEach(b => b.addEventListener("click", () => {
  Object.entries(PRESETS[b.dataset.preset]).forEach(([id, v]) => { $("#" + id).value = v; });
  updateCam();
}));
["#cK", "#cEy", "#cEp", "#cB", "#cBl", "#cO", "#cN"].forEach(id => $(id).addEventListener("input", updateCam));
$("#cGuide").addEventListener("change", drawCam);

/* =====================================================================
   PART 2: WHERE A ROAD POINT LANDS
   ===================================================================== */
function drawProj() {
  const s = 10 ** +$("#pS").value, y = +$("#pY").value;
  $("#pSV").textContent = `${fmt(s, s < 10 ? 1 : 0)} m`;
  $("#pYV").textContent = `${fmt(y, 2)} m${y ? (y > 0 ? " (right)" : " (left)") : ""}`;
  const [u, v] = project(s, y);
  const aspect = narrowOf($("#projSide")) ? 1.25 : 1.15;
  // side view: camera at height h, the ray to the road point, the image plane just in front
  const sMax = Math.max(12, s * 1.15), hx = 10;                     // heights drawn 10 times larger
  const fr = frame($("#projSide"), aspect, { x: [-0.06 * sMax, sMax], y: [-0.2 * hx, 2.2 * hx], yticks: false, xlabel: "distance ahead (m)", xfmt: v => fmt(v, 0) });
  const { ctx, X, Y } = fr, ch = CAM_H * hx, dImg = 0.05 * sMax;
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; seg(ctx, X(-0.06 * sMax), Y(0), X(sMax), Y(0));
  ctx.strokeStyle = alpha("--path", 0.7); ctx.setLineDash([5, 4]); seg(ctx, X(0), Y(ch), X(sMax), Y(ch)); ctx.setLineDash([]);
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 2; seg(ctx, X(dImg), Y(ch - 0.9 * hx), X(dImg), Y(ch + 0.6 * hx));       // the image plane
  ctx.strokeStyle = css("--geom"); ctx.lineWidth = 1.5; seg(ctx, X(0), Y(ch), X(s), Y(0));                                  // the ray
  const hitY = ch - ch * dImg / s;
  ctx.fillStyle = css("--geom"); ctx.beginPath(); ctx.arc(X(dImg), Y(hitY), 4, 0, 2 * Math.PI); ctx.fill();
  ctx.beginPath(); ctx.arc(X(s), Y(0), 4.5, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(X(0), Y(ch), 4, 0, 2 * Math.PI); ctx.fill();
  ctx.font = "600 10px " + css("--font-ui"); ctx.textAlign = "left";
  ctx.fillText("camera", X(0) + 6, Y(ch) - 8);
  ctx.fillStyle = css("--path"); ctx.textAlign = "right"; ctx.fillText("horizon", X(sMax) - 2, Y(ch) - 5);
  ctx.fillStyle = css("--muted"); ctx.textAlign = "left"; ctx.fillText("image plane", X(dImg) + 5, Y(ch + 0.6 * hx) + 4);
  ctx.fillStyle = css("--ink"); ctx.textAlign = "center"; ctx.fillText(`h = ${CAM_H} m`, X(0.03 * sMax) + 18, Y(ch / 2));
  // the image: the frame, a grid, the horizon, a straight lane, and the point
  const cv = $("#projImg"), { ctx: c2, w, h } = fit(cv, 1), P = (uu, vv) => [(uu + 0.5) * w / W, (vv + 0.5) * h / H];
  c2.fillStyle = alpha("--ink", 0.85); c2.fillRect(0, 0, w, h);
  c2.strokeStyle = alpha("--muted", 0.35); c2.lineWidth = 1;
  for (let g = 8; g < 64; g += 8) { seg(c2, g * w / W, 0, g * w / W, h); seg(c2, 0, g * h / H, w, g * h / H); }
  c2.strokeStyle = css("--path"); c2.lineWidth = 1.5; seg(c2, 0, P(0, HORIZON)[1], w, P(0, HORIZON)[1]);
  c2.strokeStyle = "rgba(235, 235, 235, 0.55)"; c2.lineWidth = 1.2;
  for (const o of [-LANE, 0, LANE]) { c2.beginPath(); [3, 4, 6, 10, 20, 45, 200].forEach((d, i) => { const [x1, y1] = P(...project(d, o)); i ? c2.lineTo(x1, y1) : c2.moveTo(x1, y1); }); c2.stroke(); }
  const inside = u >= -0.5 && u < W - 0.5 && v >= -0.5 && v < H - 0.5;
  const [px, py] = P(clamp(u, -0.5, W - 0.5), clamp(v, -0.5, H - 0.5));
  c2.fillStyle = css("--geom"); c2.beginPath(); c2.arc(px, py, inside ? 5 : 4, 0, 2 * Math.PI); c2.fill();
  caption($("#projCap"),
    `[[u = c_x + f\\,\\frac{y}{s} = 32 + 55 \\cdot \\frac{${fmt(y, 2)}}{${fmt(s, 1)}} = ${fmt(u, 1)}]] and [[v = c_y + f\\,\\frac{h}{s} = 21.76 + \\frac{68.75}{${fmt(s, 1)}} = ${fmt(v, 1)}]] (rows count down from the top). ` +
    (inside ? (s > 40 ? "Far away, every point crowds towards the vanishing point on the horizon." : "The faint lines are a straight lane, converging to the vanishing point.") : "<b>This point is outside the image</b>: too close, or too far to the side, for the camera to see."));
}
["#pS", "#pY"].forEach(id => $(id).addEventListener("input", drawProj));

/* =====================================================================
   PART 3: THE POINT-SPREAD FUNCTION
   ===================================================================== */
const SCENE = { ey: 0.3, ep: 0.02, kap: 0.012 };
const sharpScene = render(SCENE);
function drawPsf() {
  const sigma = +$("#psfS").value, r = Math.max(1, roundHalfEven(3 * sigma));
  $("#psfSV").textContent = `${fmt(sigma, 1)} px`;
  const k1 = Array.from({ length: 2 * r + 1 }, (_, i) => Math.exp(-0.5 * ((i - r) / sigma) ** 2)), mx = 1;   // k1 peaks at 1 in the middle
  const { ctx, w, h } = fit($("#psfK"), 1), n = 2 * r + 1, cell = w / n;
  ctx.fillStyle = alpha("--ink", 0.9); ctx.fillRect(0, 0, w, h);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    ctx.fillStyle = alpha("--warm", Math.min(1, (k1[i] * k1[j]) / mx)); ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
  }
  drawImage($("#psfA"), sharpScene);
  const blurred = render({ ...SCENE, blur: sigma });
  drawImage($("#psfB"), blurred);
  caption($("#psfCap"),
    `With [[\\sigma = ${fmt(sigma, 1)}]] px, the kernel is cut at [[r = 3\\sigma \\approx ${r}]] pixels, a ${n} × ${n} square. ` +
    `The sharpness feature falls from ${fmt(sharpnessFeat(sharpScene), 4)} to <b>${fmt(sharpnessFeat(blurred), 4)}</b>, while the brightness barely moves (${fmt(brightnessFeat(sharpScene), 3)} to ${fmt(brightnessFeat(blurred), 3)}): blur spreads the light, it does not remove it.`);
}
$("#psfS").addEventListener("input", drawPsf);

/* =====================================================================
   PART 4: THE TWO FEATURES (the notebook's curves)
   ===================================================================== */
const BLURS = Array.from({ length: 30 }, (_, i) => 3.5 * i / 29), GAINS = Array.from({ length: 30 }, (_, i) => 0.2 + 0.8 * i / 29);
const SH_CURVE = BLURS.map(bl => sharpnessFeat(render({ ...SCENE, blur: bl }))), BR_CURVE = GAINS.map(b => brightnessFeat(render({ ...SCENE, b })));
function drawFeat() {
  const bl = +$("#fBl").value, b = +$("#fB").value;
  $("#fBlV").textContent = `${fmt(bl, 1)} px`; $("#fBV").textContent = fmt(b, 2);
  const aspect = narrowOf($("#featSh")) ? 1.25 : 1.15, sh = sharpnessFeat(render({ ...SCENE, blur: bl })), br = brightnessFeat(render({ ...SCENE, b }));
  const limit = (f, x) => { f.ctx.strokeStyle = css("--geom"); f.ctx.lineWidth = 1.5; f.ctx.setLineDash([2, 3]); seg(f.ctx, f.X(x), f.Y(f.y0), f.X(x), f.pad.t); f.ctx.setLineDash([]); };
  const dot = (f, x, y) => { f.ctx.fillStyle = css("--ink"); f.ctx.beginPath(); f.ctx.arc(f.X(x), f.Y(y), 5, 0, 2 * Math.PI); f.ctx.fill(); };
  const fa = frame($("#featSh"), aspect, { x: [0, 3.5], y: [0, 0.027], xlabel: "blur σ (px)", yfmt: v => fmt(v, 3) });
  limit(fa, ODD.blur); plotLine(fa, BLURS, SH_CURVE, { color: css("--path"), width: 2.2 }); dot(fa, bl, sh);
  const fb = frame($("#featBr"), aspect, { x: [0.2, 1], y: [0, 0.2], xlabel: "illumination gain b", yfmt: v => fmt(v, 2) });
  limit(fb, ODD.brightness); plotLine(fb, GAINS, BR_CURVE, { color: css("--warm"), width: 2.2 }); dot(fb, b, br);
  const sh0 = SH_CURVE[0], br0 = BR_CURVE[BR_CURVE.length - 1];
  caption($("#featCap"),
    `Blur ${fmt(bl, 1)} px: sharpness <b>${fmt(sh, 4)}</b>, ${fmt(sh / sh0 * 100, 1)}% of the sharp image's${bl > ODD.blur ? ", past the domain's limit" : ""}. ` +
    `Illumination ${fmt(b, 2)}: brightness <b>${fmt(br, 3)}</b>${b < ODD.brightness ? ", below the domain's limit" : ""}. ` +
    `Brightness is exactly proportional to the gain (the image is multiplied by it); sharpness collapses quickly, to about 5% of its sharp value at the 1.6 px limit.`);
}
["#fBl", "#fB"].forEach(id => $(id).addEventListener("input", drawFeat));

/* =====================================================================
   PART 5: THE BEND THE PIXELS CANNOT SEE
   ===================================================================== */
const inOdd = render(SCENE), BR0 = brightnessFeat(inOdd), SH0 = sharpnessFeat(inOdd);
const KAPS = Array.from({ length: 31 }, (_, i) => 0.002 * i);
const KCURVE = KAPS.map(k => { const im = render({ ...SCENE, kap: k }); return [sharpnessFeat(im) / SH0 - 1, brightnessFeat(im) / BR0 - 1]; });
const BLUR_REF = sharpnessFeat(render({ ...SCENE, blur: ODD.blur })) / SH0 - 1;
function drawOod() {
  const k = +$("#oK").value, im = render({ ...SCENE, kap: k });
  $("#oKV").textContent = `${fmt(k, 3)} per m${k > ODD.kappa ? " (outside the domain)" : ""}`;
  $("#oodBTitle").textContent = `This bend: κ = ${fmt(k, 3)}`;
  drawImage($("#oodA"), inOdd); drawImage($("#oodB"), im);
  const cv = $("#oodC");
  const fr = frame(cv, narrowOf(cv) ? 1.45 : 2.6, { x: [0, 0.06], y: [-1, 0.3], xlabel: "curvature κ", yfmt: v => `${Math.round(v * 100)}%`, yticks: [-1, -0.5, 0, 0.25] });
  const { ctx, X, Y } = fr;
  ctx.strokeStyle = css("--geom"); ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]); seg(ctx, X(0), Y(BLUR_REF), X(0.06), Y(BLUR_REF));
  ctx.strokeStyle = alpha("--muted", 0.8); ctx.setLineDash([2, 3]); seg(ctx, X(ODD.kappa), Y(-1), X(ODD.kappa), fr.pad.t); ctx.setLineDash([]);
  ctx.font = "600 10px " + css("--font-ui"); ctx.fillStyle = css("--muted"); ctx.textAlign = "left"; ctx.fillText("domain's limit", X(ODD.kappa) + 4, Y(-0.85));
  plotLine(fr, KAPS, KCURVE.map(c => c[0]), { color: css("--path"), width: 2.2 });
  plotLine(fr, KAPS, KCURVE.map(c => c[1]), { color: css("--warm"), width: 2.2 });
  const dsh = sharpnessFeat(im) / SH0 - 1, dbr = brightnessFeat(im) / BR0 - 1;
  [[dsh, "--path"], [dbr, "--warm"]].forEach(([y, col]) => { ctx.fillStyle = css(col); ctx.beginPath(); ctx.arc(X(k), Y(y), 5, 0, 2 * Math.PI); ctx.fill(); });
  caption($("#oodCap"),
    `On the ordinary bend: sharpness ${fmt(SH0, 4)}, brightness ${fmt(BR0, 3)}. On this one: sharpness <b>${fmt(sharpnessFeat(im), 4)}</b> (${pct(dsh)}), brightness <b>${fmt(brightnessFeat(im), 3)}</b> (${pct(dbr)}). ` +
    `Blur at the domain's limit changes the sharpness by ${pct(BLUR_REF)} (red dashed line). ` +
    (k > ODD.kappa ? "This bend is outside the domain, and the features barely notice: if anything the image looks sharper." : "Inside the domain: an ordinary bend."));
}
$("#oK").addEventListener("input", drawOod);

/* =====================================================================
   start-up
   ===================================================================== */
updateCam(); drawProj(); drawPsf(); drawFeat(); drawOod();
onRedraw(() => { drawCam(); drawProj(); drawPsf(); drawFeat(); drawOod(); });
})();
