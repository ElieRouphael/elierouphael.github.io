"""Reproduce every number quoted on lab B4 of "Advanced MPC and Learning MPC",
"Data-driven local models for racing" (b04-data-driven-local-models-for-racing.html).

Usage:  python b04_numbers.py [path/to/b04-data.js]
Needs numpy and the course package (pip install -e . in the course folder, or PYTHONPATH=src).

The recorded laps (four training laps and one held-out lap of the nonlinear bicycle simulator under a noisy
path-following controller) come from the lab notebook and ship with the page as interactive/b04-data.js; the
script reads that file (by default next to this script, in ../interactive/) so that every regression, validation
and sweep on the page is recomputed from the same data. The local structured regression is re-implemented here
as in the notebook; the package's `local_affine_regression` is the simpler unstructured version read in the
code walk. Takes about a minute.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from advanced_mpc_lmpc.racing import local_affine_regression, nonlinear_vehicle_step

np.set_printoptions(precision=4, suppress=True)
G = 9.81
DT = 0.1
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "interactive" / "b04-data.js"


def section(title):
    print(f"\n=== {title}")


# --------------------------------------------------------------------------------------------------------------
# the data file
text = DATA.read_text(encoding="utf-8")
D = json.loads(text[text.index("{"):text.rindex("}") + 1])
TR = D["track"]; TS, TK, TXY, THEAD = np.asarray(TR["s"]), np.asarray(TR["K"]), np.asarray(TR["xy"]), np.asarray(TR["heading"])
L = float(TR["length"]); W = float(TR["W"])
LAPS = [dict(X=np.asarray(l["X"])[:-1], U=np.asarray(l["U"]), Xn=np.asarray(l["X"])[1:], Xfull=np.asarray(l["X"])) for l in D["laps"]]
VAL = dict(X=np.asarray(D["val"]["X"]), U=np.asarray(D["val"]["U"]))
NV = len(VAL["U"])


def curvature_at(s):
    q = np.mod(s, L); i = min(len(TS) - 1, int(q / L * len(TS)))
    s0, s1 = TS[i], TS[i + 1] if i + 1 < len(TS) else L
    k1 = TK[i + 1] if i + 1 < len(TS) else TK[0]
    return TK[i] + (q - s0) / (s1 - s0) * (k1 - TK[i])


# --------------------------------------------------------------------------------------------------------------
# the model: scaled-l1 neighbours per lap, Epanechnikov weights, structured weighted ridge regression,
# exact Frenet rows (the notebook's construction)
def kinematic_rows(x, dt=DT):
    vx, vy, r, epsi, s, ey = x
    kappa = curvature_at(s); den = 1 - kappa * ey; c, sn = np.cos(epsi), np.sin(epsi)
    tang = vx * c - vy * sn; dtang = -vx * sn - vy * c; sdot = tang / den
    A = np.array([[-dt * kappa * c / den, dt * kappa * sn / den, dt, 1 - dt * kappa * dtang / den, 0, -dt * kappa**2 * tang / den**2],
                  [dt * c / den, -dt * sn / den, 0, dt * dtang / den, 1, dt * kappa * tang / den**2],
                  [dt * sn, dt * c, 0, dt * tang, 0, 1]])
    f = np.array([epsi + dt * (r - kappa * sdot), s + dt * sdot, ey + dt * (vx * sn + vy * c)])
    return A, f - A @ x, kappa


def weighted_ridge(Phi, Y, w, ridge):
    N = Phi.T @ (w[:, None] * Phi) + ridge * np.eye(Phi.shape[1])
    theta = np.linalg.solve(N, Phi.T @ (w[:, None] * Y))
    ev = np.linalg.eigvalsh(N)
    return theta, ev.max() / ev.min()


def structured_fit(X, U, Xn, w, ridge, offset=True, vx_delta=False, lat_acc=False):
    one = np.ones((len(X), 1)) if offset else np.zeros((len(X), 0))
    vx_cols = [1] + ([0] if vx_delta else []); lat_cols = [0] + ([1] if lat_acc else [])
    Phiv = np.hstack([X[:, :3], U[:, vx_cols], one]); tv, cv = weighted_ridge(Phiv, Xn[:, :1], w, ridge); tv = tv[:, 0]
    Phil = np.hstack([X[:, :3], U[:, lat_cols], one]); tl, cl = weighted_ridge(Phil, Xn[:, 1:3], w, ridge)
    A = np.zeros((3, 6)); B = np.zeros((3, 2)); c = np.zeros(3)
    A[0, :3] = tv[:3]; A[1, :3] = tl[:3, 0]; A[2, :3] = tl[:3, 1]
    for i, col in enumerate(vx_cols): B[0, col] = tv[3 + i]
    for i, col in enumerate(lat_cols): B[1, col] = tl[3 + i, 0]; B[2, col] = tl[3 + i, 1]
    if offset: c[0] = tv[3 + len(vx_cols)]; c[1] = tl[3 + len(lat_cols), 0]; c[2] = tl[3 + len(lat_cols), 1]
    return A, B, c, (cv, cl)


def select(laps, x, u, scaling=(0.1, 1, 1, 1, 1), h=5.0, m=7):
    q = np.r_[x[:3], u]; sc = np.asarray(scaling)
    rows = []; sel = []
    for lap in laps:
        Z = np.hstack([lap["X"][:, :3], lap["U"]]); d = np.abs(Z - q) @ sc
        order = np.lexsort((np.arange(len(d)), d)); inside = [i for i in order if d[i] < h]
        if inside: idx = inside[:m]; hl = h; fb = False
        else: idx = list(order[:m]); hl = 1.05 * max(d[i] for i in idx); fb = True
        wl = [0.75 * (1 - (d[i] / hl) ** 2) for i in idx]
        rows += [(lap["X"][i], lap["U"][i], lap["Xn"][i], wl[k]) for k, i in enumerate(idx)]
        sel.append(dict(idx=idx, d=[d[i] for i in idx], w=wl, fallback=fb))
    X = np.array([r[0] for r in rows]); U = np.array([r[1] for r in rows]); Xn = np.array([r[2] for r in rows]); w = np.array([r[3] for r in rows])
    return X, U, Xn, w, sel


def local_fit(laps, x, u, scaling=(0.1, 1, 1, 1, 1), h=5.0, m=7, ridge=1e-8, **opts):
    X, U, Xn, w, sel = select(laps, x, u, scaling, h, m)
    A, B, c, cond = structured_fit(X, U, Xn, w, ridge, **opts)
    Ak, ck, kappa = kinematic_rows(x)
    return dict(A=np.vstack([A, Ak]), B=np.vstack([B, np.zeros((3, 2))]), c=np.r_[c, ck], cond=cond, sel=sel, kappa=kappa,
                fallback=any(s["fallback"] for s in sel))


def global_dyn(laps, ridge=1e-8, **opts):
    X = np.vstack([l["X"] for l in laps]); U = np.vstack([l["U"] for l in laps]); Xn = np.vstack([l["Xn"] for l in laps])
    return structured_fit(X, U, Xn, np.ones(len(X)), ridge, **opts)


def global_fit(gd, x):
    A, B, c, _ = gd; Ak, ck, _ = kinematic_rows(x)
    return dict(A=np.vstack([A, Ak]), B=np.vstack([B, np.zeros((3, 2))]), c=np.r_[c, ck])


apply = lambda M, x, u: M["A"] @ x + M["B"] @ u + M["c"]
def support(laps, x, u, scaling=(0.1, 1, 1, 1, 1)):
    q = np.r_[x[:3], u]; sc = np.asarray(scaling)
    return min(float((np.abs(np.hstack([l["X"][:, :3], l["U"]]) - q) @ sc).min()) for l in laps)


def validate(h=5.0, m=7, ridge=1e-8, scaling=(0.1, 1, 1, 1, 1), **opts):
    gd = global_dyn(LAPS, ridge, **opts); El, Eg, sup, cond, fb = [], [], [], [], []
    for k in range(NV):
        x, u, t = VAL["X"][k], VAL["U"][k], VAL["X"][k + 1]
        M = local_fit(LAPS, x, u, scaling, h, m, ridge, **opts)
        El.append(apply(M, x, u) - t); Eg.append(apply(global_fit(gd, x), x, u) - t)
        sup.append(support(LAPS, x, u, scaling)); cond.append(max(M["cond"])); fb.append(M["fallback"])
    El, Eg = np.array(El), np.array(Eg)
    return dict(El=El, Eg=Eg, rl=np.sqrt((El**2).mean(0)), rg=np.sqrt((Eg**2).mean(0)), sup=np.array(sup), cond=np.array(cond), fb=np.array(fb), gd=gd)


# --------------------------------------------------------------------------------------------------------------
section("Part 1: the dynamic bicycle model at the example state (v_x 10, v_y 0.25, r 0.2, delta 0.08)")
m_, lf, lr, Iz, Bp, Cp = 1.98, 0.125, 0.125, 0.024, 1.0, 1.25
Dp = 0.8 * m_ * G / 2
pacejka = lambda a: Dp * np.sin(Cp * np.arctan(Bp * a))
vx, vy, r, delta = 10.0, 0.25, 0.2, 0.08
af = delta - np.arctan2(vy + lf * r, vx); ar = -np.arctan2(vy - lr * r, vx)
Fyf, Fyr = pacejka(af), pacejka(ar)
print(f"peak force per axle D = {Dp:.2f} N; slip angles alpha_f = {af:.5f}, alpha_r = {ar:.5f}; forces F_yf = {Fyf:.4f} N, F_yr = {Fyr:.4f} N")
epsi, ey, kappa, ax = 0.04, 0.3, 0.03, 1.0
sdot = (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - kappa * ey)
xd = np.array([ax - Fyf * np.sin(delta) / m_ + r * vy, (Fyf * np.cos(delta) + Fyr) / m_ - r * vx, (lf * Fyf * np.cos(delta) - lr * Fyr) / Iz,
               r - kappa * sdot, 1.0, vx * np.sin(epsi) + vy * np.cos(epsi)])
rhs = xd / sdot; rhs[4] = 1 / sdot
print("spatial right-hand side (v_x', v_y', r', e_psi', t', e_y') =", rhs.round(4).tolist())
print(f"largest longitudinal tire term at delta = 0.5: {Dp * np.sin(0.5) / m_:.2f} m/s^2")

section("Part 2: the racing data")
print(f"track: {L:.3f} m long, half-width {W}, curvature [{TK.min():.3f}, {TK.max():.3f}] 1/m, tightest radius {1 / np.abs(TK).max():.2f} m, "
      f"Frenet margin {np.min(1 - np.abs(TK) * W):.3f}")
for i, l in enumerate(LAPS):
    print(f"seed {i}: {len(l['Xfull'])} samples, {DT * len(l['U']):.1f} s, max |e_y| {np.abs(l['Xfull'][:, 5]).max():.3f} m")
print(f"held-out lap: {len(VAL['X'])} samples, {DT * NV:.1f} s; training transitions {sum(len(l['U']) for l in LAPS)}")
Z = np.vstack([np.hstack([l["X"][:, :3], l["U"]]) for l in LAPS])
print("training feature ranges (v_x, v_y, r, delta, a_x):", [(round(float(a), 3), round(float(b), 3)) for a, b in zip(Z.min(0), Z.max(0))])
print(f"transitions with clipped steering |delta| = 0.5: {int(np.sum(np.abs(np.abs(Z[:, 3]) - 0.5) < 1e-9))}; "
      f"validation transitions faster than any training one: {int(np.sum(VAL['X'][:-1, 0] > Z[:, 0].max()))}")
print(f"scaled spread of the training features: {np.sum((Z.max(0) - Z.min(0)) * [0.1, 1, 1, 1, 1]):.2f}")

section("Hero and Parts 3-6: the local model at sample 120 of the held-out lap")
k = 120; x, u = VAL["X"][k], VAL["U"][k]
M = local_fit(LAPS, x, u)
dall = [d for s in M["sel"] for d in s["d"]]; wall = [w for s in M["sel"] for w in s["w"]]
print(f"query x = {x.round(3).tolist()}, u = {u.round(3).tolist()}, kappa = {M['kappa']:.3f}")
print(f"{len(dall)} selected transitions, all within {max(dall):.2f} of the query; weights within {100 * (1 - min(wall) / 0.75):.1f} percent of 3/4")
print("learned rows A[:3,:3] =", M["A"][:3, :3].round(3).tolist(), "B =", M["B"][:3].round(3).tolist(), "c =", M["c"][:3].round(4).tolist())
print("kinematic offsets", M["c"][3:].round(4).tolist(), "condition numbers", [f"{c:.1e}" for c in M["cond"]])

section("Part 7: one-step validation on the held-out lap")
V = validate()
print("local RMSE  (v_x, v_y, r, e_psi, s, e_y):", [f"{v:.3e}" for v in V["rl"]])
print("global RMSE (v_x, v_y, r, e_psi, s, e_y):", [f"{v:.3e}" for v in V["rg"]])
print("ratios:", (V["rg"][:3] / V["rl"][:3]).round(2).tolist(), "; fallbacks:", int(V["fb"].sum()))

section("Part 8: support, conditioning")
el = np.linalg.norm(V["El"][:, :3], axis=1)
print(f"largest support distance on the lap {V['sup'].max():.2f}; correlation(support, error) {np.corrcoef(V['sup'], el)[0, 1]:.2f}")
print(f"condition number median {np.median(V['cond']):.1e}, 95th percentile {np.quantile(V['cond'], 0.95):.1e}")

section("Part 9: the LTV horizon from sample 145")
def rollout(fitter, x0, U):
    X = [x0]; diags = []
    for uu in U:
        Mk = fitter(X[-1], uu); diags.append(Mk); X.append(apply(Mk, X[-1], uu))
    return np.array(X), diags
for H in (5, 14, 30):
    start = min(145, NV - H); U = VAL["U"][start:start + H]; T = VAL["X"][start:start + H + 1]
    Xl, dl = rollout(lambda xx, uu: local_fit(LAPS, xx, uu), VAL["X"][start], U)
    Xg, _ = rollout(lambda xx, uu: global_fit(V["gd"], xx), VAL["X"][start], U)
    rm = lambda X: np.sqrt(np.mean((X[:, :3] - T[:, :3]) ** 2))
    en = lambda X: np.linalg.norm(X[:, :3] - T[:, :3], axis=1)
    print(f"N = {H}: RMSE local {rm(Xl):.4f}, global {rm(Xg):.4f}; final-step error local {en(Xl)[-1]:.3f}, global {en(Xg)[-1]:.3f}")
    if H == 14:
        Av = max(np.linalg.norm(d["A"] - dl[0]["A"]) for d in dl); Bv = max(np.linalg.norm(d["B"] - dl[0]["B"]) for d in dl)
        sup = max(support(LAPS, Xl[i], U[i]) for i in range(H))
        print(f"   ||A_k - A_0||_F up to {Av:.2f}, ||B_k - B_0||_F up to {Bv:.2f}; support along the horizon at most {sup:.2f}")

section("Part 10: the out-of-distribution query")
xo, uo = np.array([4.0, 1.0, 4.0, 0.2, 5.0, 0.2]), np.array([0.5, 5.0])
Mo = local_fit(LAPS, xo, uo)
print(f"nearest stored transition at {support(LAPS, xo, uo):.3f}; fallback on {sum(s['fallback'] for s in Mo['sel'])} of 4 laps")

section("Exercises 2 to 9: the sweeps")
def line(name, VV): print(f"{name:45s} RMSE (v_x, v_y, r) = {VV['rl'][0]:.5f}, {VV['rl'][1]:.5f}, {VV['rl'][2]:.5f}; fallbacks {int(VV['fb'].sum())}; median cond {np.median(VV['cond']):.1e}")
line("lab setting", V)
Vu = validate(scaling=(1, 1, 1, 1, 1)); line("unscaled v_x (ex. 2)", Vu)
Xs, Us, _, _, _ = select(LAPS, x, u); Xu, Uu, _, _, _ = select(LAPS, x, u, scaling=(1, 1, 1, 1, 1))
print("   mean |deviation| of the 28 selected points, scaled:", np.abs(np.hstack([Xs[:, :3], Us]) - np.r_[x[:3], u]).mean(0).round(3).tolist(),
      "unscaled:", np.abs(np.hstack([Xu[:, :3], Uu]) - np.r_[x[:3], u]).mean(0).round(3).tolist())
for h in (0.5, 1, 2, 10): line(f"bandwidth h = {h} (ex. 3)", validate(h=h))
for mm_ in (3, 5, 15, 30, 80): line(f"{mm_} points per lap (ex. 4)", validate(m=mm_))
for lam in (0, 1e-4, 1e-2, 1, 10): line(f"ridge {lam} (ex. 5)", validate(ridge=lam))
line("no affine offset (ex. 6)", validate(offset=False))
Mq = local_fit(LAPS, x, u); print("   intercept at the query c =", Mq["c"][:3].round(3).tolist())
line("steering in the v_x row (ex. 7)", validate(vx_delta=True))
line("acceleration in the lateral rows (ex. 8)", validate(lat_acc=True))
Kv = np.array([abs(curvature_at(xx[4])) for xx in VAL["X"][:-1]]); vxs = VAL["X"][:-1, 0]
eg = np.linalg.norm(V["Eg"][:, :3], axis=1)
rms = lambda a, mask: np.sqrt(np.mean(a[mask] ** 2))
for name, mask in (("low curvature", Kv < np.median(Kv)), ("high curvature", Kv >= np.median(Kv)), ("low speed", vxs < np.median(vxs)), ("high speed", vxs >= np.median(vxs))):
    print(f"   ex. 9 {name:15s}: local {rms(el, mask):.4f}, global {rms(eg, mask):.4f}")

section("The code behind it: the package's unstructured local regression on the same data")
X_all = np.vstack([l["X"] for l in LAPS]); U_all = np.vstack([l["U"] for l in LAPS]); Y_all = np.vstack([l["Xn"] for l in LAPS])
Ap, Bp_, cp, idx = local_affine_regression(X_all, U_all, Y_all, x, u, k=28)
pred = Ap @ x + Bp_ @ u + cp
print(f"28 nearest transitions by the package's standardised Euclidean metric; one-step error (v_x, v_y, r) {np.linalg.norm(pred[:3] - VAL['X'][k + 1][:3]):.3e}"
      f" against {np.linalg.norm(apply(M, x, u)[:3] - VAL['X'][k + 1][:3]):.3e} for the structured local model")
xs = nonlinear_vehicle_step([0.0, 0.0, 15.0], [0.05, 0.5], 0.01)
print("the package's reduced kinematic step from (e_y, e_psi, v) = (0, 0, 15) under (delta, a) = (0.05, 0.5), kappa 0.01:", xs.round(4).tolist())
