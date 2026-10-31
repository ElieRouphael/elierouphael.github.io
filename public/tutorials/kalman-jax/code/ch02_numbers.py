"""Reproduce every number quoted on chapter 02 of "Kalman Filters with JAX"
(public/tutorials/kalman-jax/02-the-kalman-filter-and-its-nonlinear-cousins.html).

Usage:  python ch02_numbers.py          # print everything the page quotes
        python ch02_numbers.py data     # print the page's data file, interactive/ch02-data.js
Checked with jax 0.12.0.1, numpy 2.x and scipy 1.x on a laptop CPU. JAX runs in its default float32; the float64
numbers come from a second run of this script with jax_enable_x64, started automatically. The filter functions are
copied verbatim from the repository's kalman_jax/filters.py so that the script runs alone.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from fractions import Fraction

X64PART = len(sys.argv) > 1 and sys.argv[1] == "x64part"
if X64PART:
    import jax
    jax.config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np

DATA = len(sys.argv) > 1 and sys.argv[1] == "data"
REAL_STDOUT = sys.stdout
if DATA:
    sys.stdout = io.StringIO()
out = (lambda *a, **k: None) if DATA else print


def section(title):
    out(f"\n=== {title}")


# ---------------------------------------------------------------------------------------------------------------
# kalman_jax/filters.py, verbatim (docstrings shortened).
def _kalman_gain(P, H, innovation_cov):
    return jnp.linalg.solve(innovation_cov, H @ P.T).T


def _joseph_covariance_update(P, K, H, R):
    identity = jnp.eye(P.shape[0], dtype=P.dtype)
    residual_factor = identity - K @ H
    return residual_factor @ P @ residual_factor.T + K @ R @ K.T


def linear_kf_predict(x, P, F, Q):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def linear_kf_update(x, P, z, H, R):
    innovation = z - H @ x
    innovation_cov = H @ P @ H.T + R
    K = _kalman_gain(P, H, innovation_cov)

    x_updated = x + K @ innovation
    P_updated = _joseph_covariance_update(P, K, H, R)

    return x_updated, P_updated, innovation, innovation_cov


def extended_kf_predict(x, P, f, F_jacobian, Q):
    x_pred = f(x)
    F = F_jacobian(x)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def extended_kf_update(x, P, z, h, H_jacobian, R):
    H = H_jacobian(x)
    innovation = z - h(x)
    innovation_cov = H @ P @ H.T + R
    K = _kalman_gain(P, H, innovation_cov)

    x_updated = x + K @ innovation
    P_updated = _joseph_covariance_update(P, K, H, R)

    return x_updated, P_updated, innovation, innovation_cov


def _unscented_weights(state_dim, alpha=1.0, beta=2.0, kappa=0.0):
    lambda_ = alpha**2 * (state_dim + kappa) - state_dim
    scale = state_dim + lambda_

    Wm = jnp.full(2 * state_dim + 1, 1.0 / (2.0 * scale))
    Wc = jnp.full(2 * state_dim + 1, 1.0 / (2.0 * scale))
    Wm = Wm.at[0].set(lambda_ / scale)
    Wc = Wc.at[0].set(lambda_ / scale + (1.0 - alpha**2 + beta))
    return scale, Wm, Wc


def _sigma_points(x, P, alpha=1.0, beta=2.0, kappa=0.0, jitter=1e-9):
    state_dim = x.shape[0]
    scale, Wm, Wc = _unscented_weights(state_dim, alpha=alpha, beta=beta, kappa=kappa)
    sqrt_P = jnp.linalg.cholesky(P + jitter * jnp.eye(state_dim, dtype=P.dtype))
    offsets = jnp.sqrt(scale) * sqrt_P.T
    sigma_points = jnp.concatenate(
        [x[None, :], x[None, :] + offsets, x[None, :] - offsets],
        axis=0,
    )
    return sigma_points, Wm, Wc


def unscented_kf_predict(x, P, f, Q, alpha=1.0, beta=2.0, kappa=0.0):
    sigma_points, Wm, Wc = _sigma_points(x, P, alpha=alpha, beta=beta, kappa=kappa)
    transformed_points = jax.vmap(f)(sigma_points)

    x_pred = jnp.sum(Wm[:, None] * transformed_points, axis=0)
    diffs = transformed_points - x_pred
    cov_terms = jax.vmap(lambda diff: jnp.outer(diff, diff))(diffs)
    P_pred = jnp.sum(Wc[:, None, None] * cov_terms, axis=0) + Q

    return x_pred, P_pred


def unscented_kf_update(x, P, z, h, R, alpha=1.0, beta=2.0, kappa=0.0):
    sigma_points, Wm, Wc = _sigma_points(x, P, alpha=alpha, beta=beta, kappa=kappa)
    transformed_points = jax.vmap(h)(sigma_points)

    z_pred = jnp.sum(Wm[:, None] * transformed_points, axis=0)
    z_diffs = transformed_points - z_pred
    x_diffs = sigma_points - x

    measurement_terms = jax.vmap(lambda diff: jnp.outer(diff, diff))(z_diffs)
    cross_terms = jax.vmap(lambda x_diff, z_diff: jnp.outer(x_diff, z_diff))(x_diffs, z_diffs)

    P_zz = jnp.sum(Wc[:, None, None] * measurement_terms, axis=0) + R
    P_xz = jnp.sum(Wc[:, None, None] * cross_terms, axis=0)

    K = jnp.linalg.solve(P_zz, P_xz.T).T
    innovation = z - z_pred
    x_updated = x + K @ innovation
    P_updated = P - K @ P_zz @ K.T
    P_updated = 0.5 * (P_updated + P_updated.T)

    return x_updated, P_updated, innovation, P_zz


def generate_linear_synthetic_data(key, n_steps=100, dt=0.1, process_noise_std=0.1, measurement_noise_std=0.5,
                                   initial_state=None):
    F = jnp.array([[1.0, dt], [0.0, 1.0]])
    H = jnp.array([[1.0, 0.0]])

    Q = jnp.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]) * process_noise_std**2
    R = jnp.array([[measurement_noise_std**2]])

    x_true = jnp.array([0.0, 1.0]) if initial_state is None else initial_state

    true_states = []
    measurements = []

    for _ in range(n_steps):
        key, process_key = jax.random.split(key)
        process_noise = jax.random.multivariate_normal(process_key, jnp.zeros(2), Q)
        x_true = F @ x_true + process_noise
        true_states.append(x_true)

        key, measurement_key = jax.random.split(key)
        measurement_noise = jax.random.normal(measurement_key, (1,)) * measurement_noise_std
        measurements.append(H @ x_true + measurement_noise)

    return F, H, Q, R, jnp.array(true_states), jnp.vstack(measurements)


def run_linear_kf(measurements, F, H, Q, R, x0, P0):
    def step(carry, measurement):
        x, P = carry
        x_pred, P_pred = linear_kf_predict(x, P, F, Q)
        x_updated, P_updated, _, _ = linear_kf_update(x_pred, P_pred, measurement, H, R)
        return (x_updated, P_updated), x_updated

    (_, _), estimated_states = jax.lax.scan(step, (x0, P0), measurements)
    return estimated_states


def run_ekf(measurements, f, h, F_jacobian, H_jacobian, Q, R, x0, P0):
    def step(carry, measurement):
        x, P = carry
        x_pred, P_pred = extended_kf_predict(x, P, f, F_jacobian, Q)
        x_updated, P_updated, _, _ = extended_kf_update(x_pred, P_pred, measurement, h, H_jacobian, R)
        return (x_updated, P_updated), x_updated

    (_, _), estimated_states = jax.lax.scan(step, (x0, P0), measurements)
    return estimated_states


def run_ukf(measurements, f, h, Q, R, x0, P0, alpha=1.0, beta=2.0, kappa=0.0):
    def step(carry, measurement):
        x, P = carry
        x_pred, P_pred = unscented_kf_predict(x, P, f, Q, alpha=alpha, beta=beta, kappa=kappa)
        x_updated, P_updated, _, _ = unscented_kf_update(x_pred, P_pred, measurement, h, R, alpha=alpha, beta=beta,
                                                         kappa=kappa)
        return (x_updated, P_updated), x_updated

    (_, _), estimated_states = jax.lax.scan(step, (x0, P0), measurements)
    return estimated_states


def compute_rmse(true_states, estimated_states):
    return jnp.sqrt(jnp.mean((true_states - estimated_states) ** 2, axis=0))
# ---------------------------------------------------------------------------------------------------------------


if X64PART:                                                  # the second run: the same data, promoted to float64
    d = np.load(sys.argv[2])
    F, H, Q, R, zs = (jnp.asarray(d[k], jnp.float64) for k in ("F", "H", "Q", "R", "zs"))
    x0, P0 = jnp.zeros(2), jnp.eye(2)
    f, h = (lambda x: F @ x), (lambda x: H @ x)
    lin = run_linear_kf(zs, F, H, Q, R, x0, P0)
    print("dtype", lin.dtype)
    for a in (1e-3, 1.0):
        u = run_ukf(zs, f, h, Q, R, x0, P0, alpha=a)
        print(f"x64, jitter 1e-9, alpha {a:g}: max |UKF - linear| over all 1000 steps and both states {float(jnp.abs(u - lin).max()):.3g}")
    original = _sigma_points

    def _sigma_points(x, P, alpha=1.0, beta=2.0, kappa=0.0, jitter=1e-9):   # noqa: F811  (the jitter removed)
        return original(x, P, alpha=alpha, beta=beta, kappa=kappa, jitter=0.0)
    for a in (1e-3, 1.0):
        u = run_ukf(zs, f, h, Q, R, x0, P0, alpha=a)
        print(f"x64, no jitter,    alpha {a:g}: max |UKF - linear| {float(jnp.abs(u - lin).max()):.3g}")
    sys.exit()

out("jax", jax.__version__, "numpy", np.__version__, "default dtype", jnp.zeros(1).dtype)

key = jax.random.PRNGKey(42)
F, H, Q, R, xs, zs = generate_linear_synthetic_data(key, n_steps=1000)
x0, P0 = jnp.zeros(2), jnp.eye(2)
f, h = (lambda x: F @ x), (lambda x: H @ x)
f64 = lambda a: np.asarray(a, np.float64)
Fn, Hn, Qn, Rn = map(f64, (F, H, Q, R))


def lin_scan(Rf=R, zs_=zs, x0_=x0):
    def step(carry, z):
        x, P = carry
        xp, Pp = linear_kf_predict(x, P, F, Q)
        xn, Pn, e, S = linear_kf_update(xp, Pp, z, H, Rf)
        return (xn, Pn), (xn, Pn, e[0], S[0, 0])
    _, outs = jax.lax.scan(step, (x0_, P0), zs_)
    return outs


def ukf_scan(alpha):
    def step(carry, z):
        x, P = carry
        xp, Pp = unscented_kf_predict(x, P, f, Q, alpha=alpha)
        xn, Pn, e, S = unscented_kf_update(xp, Pp, z, h, R, alpha=alpha)
        return (xn, Pn), (xn, Pn, xp, Pp, x, P, e[0], S[0, 0], jnp.linalg.eigvalsh(Pp)[0])
    _, outs = jax.lax.scan(step, (x0, P0), zs)
    return outs


lx, lP, le, lS = lin_scan()
lin = lx

if DATA:
    def g(v, p):
        s = format(float(v), f".{p}g")
        return "0" if float(s) == 0 else s
    arr = lambda a, p: "[" + ",".join("[" + ",".join(g(v, p) for v in row) + "]" for row in np.asarray(a)) + "]"
    vec = lambda a, p: "[" + ",".join(g(v, p) for v in np.asarray(a).ravel()) + "]"
    ekf = run_ekf(zs, f, h, lambda _: F, lambda _: H, Q, R, x0, P0)
    ukf = run_ukf(zs, f, h, Q, R, x0, P0)
    rm = {k: [g(v, 6) for v in np.asarray(compute_rmse(xs, e))] for k, e in (("linear", lin), ("ekf", ekf), ("ukf", ukf))}
    bad = run_ukf(zs, f, h, Q, R, x0, P0, alpha=1e-3)
    nan_from = int(jnp.argmax(jnp.isnan(bad[:, 0])))
    P = np.eye(2)
    for _ in range(2000):
        Pp = Fn @ P @ Fn.T + Qn; S = Hn @ Pp @ Hn.T + Rn; K = Pp @ Hn.T / S
        A = np.eye(2) - K @ Hn; P = A @ Pp @ A.T + K @ Rn @ K.T
    rmse_txt = ", ".join(f'"{k}": [{v[0]}, {v[1]}]' for k, v in rm.items())
    REAL_STDOUT.write(
        "/* Kalman filters with JAX, chapter 02 data: the synthetic position-velocity rollout generated by the\n"
        "   repository's JAX random stream (1000 true states and position measurements, at 9 significant digits so that\n"
        "   Math.fround recovers JAX's float32 values exactly), its model matrices, and JAX's filter results for comparison.\n"
        "   Printed by code/ch02_numbers.py data. The filters themselves run live on the page. */\n"
        f"window.KJ1 = {{F:{arr(F, 7)},H:{arr(H, 7)},Q:{arr(Q, 7)},R:{arr(R, 7)},\n"
        f"  xs: {arr(xs, 9)},\n"
        f"  zs: {vec(zs, 9)},\n"
        f"  py: {{rmse:{{{rmse_txt}}}, ukfNanFrom:{nan_from}, lin:{arr(lin, 6)}, ukf:{arr(ukf, 6)}, "
        f"Kss:{vec(K, 6)}, Pss:{arr(P, 6)}}},\n}};\n")
    sys.exit()

section("1. The model (Part 1)")
out("Q =", np.asarray(Q).tolist(), "; Q / 1e-2 =", (f64(Q) / 1e-2).tolist())
out(f"velocity wobble per step sqrt(Q22) = {np.sqrt(Qn[1, 1]):.3f}; position sqrt(Q11) = {np.sqrt(Qn[0, 0]):.1e}; "
    f"correlation {Qn[0, 1] / np.sqrt(Qn[0, 0] * Qn[1, 1]):.4f} (sqrt(3)/2 = {np.sqrt(3) / 2:.4f})")
out(f"true position at the end {float(xs[-1, 0]):.1f}; velocity between {float(xs[:, 1].min()):.2f} and {float(xs[:, 1].max()):.2f}")

section("2. The three filters on the repository's data (hero, Part 2 callout, Part 3)")
out(f"sensor RMSE {float(jnp.sqrt(jnp.mean((zs[:, 0] - xs[:, 0]) ** 2))):.4f}")
ekf = run_ekf(zs, f, h, lambda _: F, lambda _: H, Q, R, x0, P0)
ukf1 = run_ukf(zs, f, h, Q, R, x0, P0)
for name, est in (("linear", lin), ("EKF", ekf), ("UKF alpha=1", ukf1)):
    r = np.asarray(compute_rmse(xs, est))
    out(f"{name:12s} RMSE position {r[0]:.4f}, velocity {r[1]:.4f}")
out(f"max |EKF - linear| {float(jnp.abs(ekf - lin).max()):.2g}; max |UKF(alpha=1) - linear| {float(jnp.abs(ukf1 - lin).max()):.2g}")

section("3. The worked first step (Part 2)")
P10 = Fn @ np.eye(2) @ Fn.T + Qn
S1 = P10[0, 0] + Rn[0, 0]
K1 = P10[:, 0] / S1
K1_noQ = (Fn @ Fn.T)[:, 0] / ((Fn @ Fn.T)[0, 0] + Rn[0, 0])
out(f"P_1|0 = {np.round(P10, 4).tolist()}, S_1 = {S1:.4f}, K_1 = ({K1[0]:.4f}, {K1[1]:.4f})")
out(f"K_1 with Q = 0: {K1_noQ[0]:.7f}; with Q: {K1[0]:.7f}; 1/1.25 = 0.8; Q adds {Qn[0, 0]:.1e} to P11, dt^2 P22 = {0.1 ** 2:.2f}")
out(f"posterior variance {P10[0, 0] - K1[0] ** 2 * S1:.4f}, sd {np.sqrt(P10[0, 0] - K1[0] ** 2 * S1):.3f}; "
    f"(1^-2 + 0.5^-2)^-1/2 = {(1 + 4) ** -0.5:.3f}")

section("4. Steady state (Part 5, questions 2, 3 and 8)")
P = np.eye(2); Ks = []
for i in range(2000):
    Pp = Fn @ P @ Fn.T + Qn; S = Hn @ Pp @ Hn.T + Rn; K = Pp @ Hn.T / S
    A = np.eye(2) - K @ Hn; P = A @ Pp @ A.T + K @ Rn @ K.T; Ks.append(K[:, 0])
Ks = np.array(Ks); Kss = Ks[-1]
out(f"K_inf = ({Kss[0]:.4f}, {Kss[1]:.4f}); posterior sds {np.sqrt(P[0, 0]):.4f} and {np.sqrt(P[1, 1]):.4f} "
    f"(3 decimals: {np.sqrt(P[0, 0]):.3f} and {np.sqrt(P[1, 1]):.3f})")
out(f"S_inf = {Rn[0, 0]} + {Pp[0, 0]:.4f} = {S[0, 0]:.3f}, sd {np.sqrt(S[0, 0]):.3f}")
conv = next(i for i in range(1, 2000) if abs(Ks[i, 0] - Kss[0]) < 1e-3 and abs(Ks[i, 1] - Kss[1]) < 1e-3)
out(f"the gain is within 1e-3 of K_inf after {conv} steps ({conv * 0.1:.1f} s); first-step velocity gain {Ks[0, 1]:.3f}")
out(f"achieved innovations: sd {float(jnp.std(le)):.3f} against sqrt(S_inf) {np.sqrt(S[0, 0]):.3f}, ratio {float(jnp.std(le)) / np.sqrt(S[0, 0]):.3f}")
out(f"as R -> 0 the gain tends to P H^T (H P H^T)^-1 = (1, P12/P11); for the settled predicted P, P12/P11 = {Pp[0, 1] / Pp[0, 0]:.3f}")
Pz = np.eye(2)
for _ in range(20000):
    Ppz = Fn @ Pz @ Fn.T + Qn
    Pz = np.array([[0.0, 0.0], [0.0, Ppz[1, 1] - Ppz[0, 1] ** 2 / Ppz[0, 0]]])
out(f"(a filter run with R = 0 settles at P12/P11 = {Ppz[0, 1] / Ppz[0, 0]:.2f} instead)")
for sa, sz in ((0.1, 0.5), (1.0, 0.5), (0.1, 0.05), (1.0, 0.05)):
    Qx = sa ** 2 * Qn / 0.1 ** 2
    P = np.eye(2)
    for _ in range(3000):
        Pp = Fn @ P @ Fn.T + Qx; S = Pp[0, 0] + sz ** 2; K = Pp[:, 0] / S
        A = np.eye(2) - np.outer(K, [1, 0]); P = A @ Pp @ A.T + sz ** 2 * np.outer(K, K)
    out(f"hero sliders sigma_a {sa}, sigma_z {sz}: steady-state gain ({K[0]:.4f}, {K[1]:.4f})")

section("5. The sigma-point weights (Part 4)")
for a in (1e-3, 1.0):
    lam = a ** 2 * 2 - 2; sc = 2 + lam
    W0m, W0c, Wi = lam / sc, lam / sc + (1 - a ** 2 + 2), 1 / (2 * sc)
    out(f"alpha {a:g}: lambda {lam:.6g}, n + lambda {sc:.6g}, sqrt {np.sqrt(sc):.4f}; W0m {W0m:.6f}, W0c {W0c:.6f}, Wi {Wi:.6g}; "
        f"sum Wm {W0m + 4 * Wi:.6f}, sum Wc {W0c + 4 * Wi:.6f}")
sc32, Wm32, Wc32 = _unscented_weights(2, alpha=1e-3)
out(f"as JAX stores them (float32): Wm {np.asarray(Wm32).tolist()}, Wc {np.asarray(Wc32).tolist()}; "
    f"sum Wm {float(Wm32.sum())}; W0c is exactly -999996.000001 and is stored as {float(Wc32[0]):.0f}, an error of "
    f"{abs(float(Wc32[0]) - (-999996.000001)):.0e}")

section("6. The float32 gaps (Part 4, step 1)")
for v in (1, 33, 250_000, 1_000_000, 33_000_000):
    out(f"gap above {v}: {float(np.spacing(np.float32(v))):.2g}")

section("7. The unscented filter with alpha = 1e-3 in float32 (Part 4, steps 3 to 7, exercise 2)")
ux, uP, uxp, uPp, ucx, ucP, ue, uS, umin = ukf_scan(1e-3)
mean_err = uxp - jax.vmap(lambda x: F @ x)(ucx)
k = int(jnp.argmax(umin <= 0))
out(f"exercise 2: first non-positive eigenvalue of the predicted covariance at step {k}; mean error there {np.asarray(mean_err[k])}")
nan = jnp.isnan(ux[:, 0])
out(f"first NaN estimate at step {int(jnp.argmax(nan))}; the true position there {float(xs[k, 0]):.1f}")
xk, Pk = ucx[k], ucP[k]                                   # the state carried into step k
sp, Wm, Wc = _sigma_points(xk, Pk, alpha=1e-3)
fx = jax.vmap(f)(sp)
pos = np.asarray(fx[:, 0])
terms = np.asarray(Wm * fx[:, 0])                         # float32 products, as JAX forms them
out("table 'meanT': predicted positions", [f"{v:.6f}" for v in pos], "(exact:", [repr(float(v)) for v in pos], ") weights",
    np.asarray(Wm).tolist())
out("                weighted terms (float32)", [f"{v:.8g}" for v in terms])
exact_stored = sum(Fraction(float(w)) * Fraction(float(p)) for w, p in zip(np.asarray(Wm), pos))
exact_terms = sum(Fraction(float(t)) for t in terms)
out(f"table 'sumT': exact from the stored positions and weights {float(exact_stored):.2f}; exact from the rounded terms "
    f"{float(exact_terms):.1f}; jnp.sum on its own {float(jnp.sum(jnp.asarray(terms))):.1f}; inside the compiled filter "
    f"{float(uxp[k, 0]):.3f}; correct F x in float64 {float((Fn @ f64(xk))[0]):.3f}")
absum = float(np.sum(np.abs(terms.astype(np.float64))))
out(f"sum of |terms| {absum:.2e}; amplification {absum / float((Fn @ f64(xk))[0]):.1e}; float32 bound u*amp = "
    f"{2 ** -24 * absum / float((Fn @ f64(xk))[0]):.2f}; float64 bound {2 ** -53 * absum / float((Fn @ f64(xk))[0]):.0e}")
offs = np.asarray(sp[1:3, 0] - sp[0, 0])
out(f"sigma-point offsets in position {offs} from {float(sp[0, 0]):.1f}, where float32's gap is "
    f"{float(np.spacing(np.float32(sp[0, 0]))):.1e}: relative {float(np.spacing(np.float32(sp[0, 0]))) / abs(offs[0]):.1%}")
dev = np.asarray(fx[:, 0] - uxp[k, 0])
vterms = np.asarray(Wc * (fx[:, 0] - uxp[k, 0]) ** 2)
out("table 'varT': deviations", [f"{v:.7f}" for v in dev], "Wc", np.asarray(Wc).tolist())
out("               terms", [f"{v:.2f}" for v in vterms], "(exact float32 values:", [repr(float(v)) for v in vterms], ")",
    f"; the filter's P_pred11 {float(uPp[k, 0, 0]):.3f}")
o = pos[1:].astype(np.float64) - float(pos[0])           # stored offsets of the four outer points from the centre
e_ = float(pos[0]) - float(uxp[k, 0])                      # the centre's deviation, the mean's error
Wc64 = np.asarray(Wc, np.float64)
out(f"decomposition in exact arithmetic of the stored values: e = {e_:.4f}; spread 250000 sum o^2 = {250000 * np.sum(o ** 2):.3f}; "
    f"e^2 sum Wc = {e_ ** 2 * Wc64.sum():.3f} (4 e^2 = {4 * e_ ** 2:.3f}); offsets sum to {o.sum():.2e} instead of 0, so the cross term "
    f"2 e 250000 sum o = {2 * e_ * 250000 * o.sum():.2f}; total with Q {e_ ** 2 * Wc64.sum() + 2 * e_ * 250000 * o.sum() + 250000 * np.sum(o ** 2) + Qn[0, 0]:.3f}; "
    f"symmetric points would give {250000 * np.sum(o ** 2) + 4 * e_ ** 2 + Qn[0, 0]:.2f}")
FPF = Fn @ f64(Pk) @ Fn.T + Qn
out(f"(F P F^T + Q)11 for the UKF's own carried P (P11 = {float(Pk[0, 0]):.3f}): {FPF[0, 0]:.3f}; "
    f"the linear filter's own predicted P11 is {float((Fn @ f64(lP[k - 1]) @ Fn.T + Qn)[0, 0]):.4f}")
out("Cholesky of [[-0.2, 0.04], [0.04, 0.49]]:", np.asarray(jnp.linalg.cholesky(jnp.array([[-0.2, 0.04], [0.04, 0.49]]))).tolist())

me = np.abs(np.asarray(mean_err[:k, 0])); pp = np.asarray(uxp[:k, 0]); pt = np.asarray(xs[:k, 0])
bands = [(0, 5), (5, 10), (10, 20), (20, 40)]
out("step 4: |mean error| binned by the UKF's predicted position:",
    ", ".join(f"[{lo},{hi}) {me[(pp >= lo) & (pp < hi)].mean():.2f}" for lo, hi in bands),
    f"; mean of |error|/|position| {np.mean(me / np.maximum(np.abs(pp), 1e-3)):.3f}")
out("        (binned by the true position:", ", ".join(f"{me[(pt >= lo) & (pt < hi)].mean():.2f}" for lo, hi in bands) + ")")

section("8. Was the float32 filter healthy before the NaN? (Part 4, step 7, question 7)")
sd = np.sqrt(np.asarray(uP[:, 0, 0]))
out(f"its reported position sd first exceeds the sensor's 0.5 at step {int(np.argmax(sd > 0.5))}; at step {k - 1} it is {sd[k - 1]:.2f}; "
    f"the linear filter's settles at {float(jnp.sqrt(lP[-1, 0, 0])):.3f}")
out("its posterior P11, median over steps 0-49, 50-99, 100-199, 200-299:",
    [f"{float(np.median(np.asarray(uP[a:b, 0, 0]))):.2f}" for a, b in ((0, 50), (50, 100), (100, 200), (200, 300))],
    "(single steps jump around:", [f"{float(uP[i, 0, 0]):.2f}" for i in (49, 50, 99, 100, 199, 200)], ")",
    f"; the linear filter's {float(lP[300, 0, 0]):.4f}")
nis_u, nis_l = np.asarray(ue ** 2 / uS), np.asarray(le ** 2 / lS)
out(f"mean NIS over steps 100-334: UKF {nis_u[100:k].mean():.2f}, linear filter {nis_l[100:k].mean():.2f}")
from scipy.stats import chi2                                 # noqa: E402
lo95, hi95 = chi2.ppf(0.025, 50) / 50, chi2.ppf(0.975, 50) / 50
win = lambda a: np.convolve(a, np.ones(50) / 50, mode="valid")
wu, wl = win(nis_u[:k]), win(nis_l[:k])
out(f"50-step mean NIS, 95% band {lo95:.2f} to {hi95:.2f}: the UKF's first window (steps 0-49) {wu[0]:.2f}, the linear filter's {wl[0]:.2f}; "
    f"UKF windows outside the band {np.mean((wu < lo95) | (wu > hi95)):.0%}, linear {np.mean((wl < lo95) | (wl > hi95)):.0%}")
for i in (50, 100):
    d = float(mean_err[i, 0])
    pred = float((Fn @ f64(ucP[i]) @ Fn.T + Qn)[0, 0]) + 4 * d * d
    out(f"step {i}: the filter's P_pred11 {float(uPp[i, 0, 0]):.4f} against F P F^T + Q + 4 delta^2 = {pred:.4f} (delta {d:.3f})")

section("9. Other values of alpha (Part 4, table 'alphaT', and the widget)")
for a in (1e-3, 1e-2, 1e-1, 1.0):
    u = run_ukf(zs, f, h, Q, R, x0, P0, alpha=a)
    bad = jnp.isnan(u[:, 0]); first = int(jnp.argmax(bad)) if bool(bad.any()) else None
    dist = float(jnp.where(~bad, jnp.abs(u[:, 0] - lin[:, 0]), 0.0).max())
    r = np.asarray(compute_rmse(xs, u))
    out(f"alpha {a:g}: first NaN {first}, largest distance from the linear filter {dist:.3g}, position RMSE {r[0]:.4f} "
        f"({r[0] / float(compute_rmse(xs, lin)[0]) - 1:+.1%})")
_, uP2, *_rest = ukf_scan(1e-2)
ue2, uS2 = _rest[4], _rest[5]
out(f"alpha 1e-2: mean NIS {float(jnp.mean(ue2 ** 2 / uS2)):.3f}, median reported position sd {float(jnp.median(jnp.sqrt(uP2[:, 0, 0]))):.3f}")


def ut_ref(sp, Wm, Wc, g):
    """The 'reordered' sums: the mean taken around the centre point, so no term is of size 3e7."""
    Y = jax.vmap(g)(sp); y0 = Y[0]
    m = y0 + jnp.sum(Wm[1:, None] * (Y[1:] - y0), axis=0)
    Dv = Y - m
    C = jnp.sum(Wc[:, None, None] * jax.vmap(jnp.outer)(Dv, Dv), axis=0)
    return Y, m, Dv, C


def ref_step(carry, z, a=1e-3):
    x, P = carry
    sp, Wm, Wc = _sigma_points(x, P, alpha=a)
    _, xp, _, Pp = ut_ref(sp, Wm, Wc, f); Pp = Pp + Q
    sp2, Wm, Wc = _sigma_points(xp, Pp, alpha=a)
    Y, zp, Dz, Pzz = ut_ref(sp2, Wm, Wc, h); Pzz = Pzz + R
    Dx = sp2 - xp
    Pxz = jnp.sum(Wc[:, None, None] * jax.vmap(jnp.outer)(Dx, Dz), axis=0)
    K = jnp.linalg.solve(Pzz, Pxz.T).T
    xn = xp + K @ (z - zp); Pn = Pp - K @ Pzz @ K.T; Pn = 0.5 * (Pn + Pn.T)
    return (xn, Pn), xn


_, rx = jax.lax.scan(ref_step, (x0, P0), zs)
out(f"reordered sums: any NaN {bool(jnp.isnan(rx).any())}, position RMSE {float(compute_rmse(xs, rx)[0]):.3f}")

section("10. Shifting the origin (Part 4, step 4, question 7)")
for c in (-100.0, -30.0, 0.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
    x0c, zc = jnp.array([c, 0.0]), zs + c
    linc = run_linear_kf(zc, F, H, Q, R, x0c, P0)
    u = run_ukf(zc, f, h, Q, R, x0c, P0, alpha=1e-3)
    bad = jnp.isnan(u[:, 0]); first = int(jnp.argmax(bad)) if bool(bad.any()) else None
    u1 = run_ukf(zc, f, h, Q, R, x0c, P0, alpha=1.0)
    out(f"shift {c:+7.0f}: linear filter moves by at most {float(jnp.abs(linc[:, 0] - c - lin[:, 0]).max()):.1e}; "
        f"UKF alpha=1e-3 first NaN at step {first}"
        f"{f' (|position| {abs(float(xs[first, 0]) + c):.1f})' if first is not None else ''}; "
        f"UKF alpha=1 within {float(jnp.abs(u1[:, 0] - linc[:, 0]).max()):.1e} of the linear filter")

section("11. Exercise 3: lie about R")
S_true = float(lS[-1])
for mult in (4.0, 1.0, 0.25):
    x_, P_, e_, S_ = lin_scan(R * mult)
    out(f"R x {mult:g}: position RMSE {float(compute_rmse(xs, x_)[0]):.4f}; sd(e)/sqrt(S_inf) {float(jnp.std(e_) / jnp.sqrt(S_[-1])):.2f}; "
        f"mean NIS {float(jnp.mean(e_ ** 2 / S_)):.4f}")


def gen_scan(key, n=1000, dt=0.1, ps=0.1, ms=0.5):
    """generate_linear_synthetic_data as a scan, with the same key splitting, so that it can be vmapped."""
    Fg = jnp.array([[1.0, dt], [0.0, 1.0]]); Qg = jnp.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]) * ps**2

    def step(carry, _):
        key, x = carry
        key, pk = jax.random.split(key)
        x = Fg @ x + jax.random.multivariate_normal(pk, jnp.zeros(2), Qg)
        key, mk = jax.random.split(key)
        return (key, x), (x, x[0] + jax.random.normal(mk, (1,)) * ms)
    _, (xs_, zs_) = jax.lax.scan(step, (key, jnp.array([0.0, 1.0])), None, length=n)
    return xs_, zs_


xs_g, zs_g = jax.jit(gen_scan)(jax.random.PRNGKey(42))
out(f"\nthe scan generator reproduces key 42: max |difference| {float(jnp.abs(xs_g - xs).max()):.1e} (states), "
    f"{float(jnp.abs(zs_g - zs).max()):.1e} (measurements)")


@jax.jit
def rmse_for_key(k, rmult):
    xs_k, zs_k = gen_scan(k)
    est = run_linear_kf(zs_k, F, H, Q, R * rmult, x0, P0)
    return compute_rmse(xs_k, est)[0]


keys = jax.random.split(jax.random.PRNGKey(0), 20)
for mult in (4.0, 0.25):
    rel = np.asarray(jax.vmap(lambda k_: rmse_for_key(k_, mult) / rmse_for_key(k_, 1.0))(keys))
    out(f"R x {mult:g} on 20 fresh rollouts: RMSE changes by {100 * (rel.min() - 1):+.1f}% to {100 * (rel.max() - 1):+.1f}%, "
        f"median {100 * (np.median(rel) - 1):+.1f}%")
    if mult == 0.25:
        out("   their RMSEs:", f"{float(jnp.min(jax.vmap(lambda k_: rmse_for_key(k_, mult))(keys))):.3f} to "
                             f"{float(jnp.max(jax.vmap(lambda k_: rmse_for_key(k_, mult))(keys))):.3f}")

section("12. Question 9: how much does the RMSE vary from one dataset to the next?")
r1000 = np.asarray(jax.vmap(lambda k_: rmse_for_key(k_, 1.0))(jnp.stack([jax.random.PRNGKey(i) for i in range(1000)])))
r42 = float(compute_rmse(xs, lin)[0])
out(f"1000 datasets (keys 0-999): median {np.median(r1000):.3f}, 5-95% {np.percentile(r1000, 5):.3f} to {np.percentile(r1000, 95):.3f}, "
    f"relative sd {r1000.std() / r1000.mean():.1%}; key 42's {r42:.4f} sits at the {np.mean(r1000 < r42):.0%} point")

section("13. Exercise 5: the scalar steady state")
q, r = 0.05, 0.25
p = (q + np.sqrt(q * q + 4 * q * r)) / 2
out(f"q = {q}, r = {r}: p_inf {p:.3f}, K_inf {p / (p + r):.2f}")
pp_ = 1.0
for _ in range(1000):
    Kq = pp_ / (pp_ + r); pp_ = 0.9 ** 2 * (1 - Kq) * pp_ + q
out(f"with A = 0.9: K_inf {pp_ / (pp_ + r):.2f}")

section("14. Double precision (Part 4, the 64-bit paragraph and the widget)")
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "data.npz")
    np.savez(path, F=np.asarray(F), H=np.asarray(H), Q=np.asarray(Q), R=np.asarray(R), zs=np.asarray(zs))
    proc = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "x64part", path], capture_output=True, text=True)
    out(proc.stdout.strip() or proc.stderr.strip()[-2000:])
