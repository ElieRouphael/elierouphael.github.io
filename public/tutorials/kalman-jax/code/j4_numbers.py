"""Reproduce every jaxpr, message, timing and Monte Carlo result quoted on chapter J4 of "Kalman Filters with JAX"
(public/tutorials/kalman-jax/j4-vmap-one-function-many-inputs.html).

Usage:  python j4_numbers.py          # print everything the page quotes
        python j4_numbers.py data     # print the page's data file, interactive/j4-data.js
Checked with jax 0.12.0.1, numpy 2.x, CPU. Timings depend on the machine; the Monte Carlo results do not.
The Kalman functions are copied verbatim from kalman_jax/filters.py so that the script runs alone.
"""
from __future__ import annotations

import io
import itertools
import json
import math
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

DATA = len(sys.argv) > 1 and sys.argv[1] == "data"
REAL_STDOUT = sys.stdout
if DATA:
    sys.stdout = io.StringIO()
out = (lambda *a, **k: None) if DATA else print
out("jax", jax.__version__, "numpy", np.__version__)


def section(title):
    out(f"\n=== {title}")


def error_of(thunk):
    try:
        thunk()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).splitlines()[0]}"
    return "no error"


def median_time(fn, reps=7):
    jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# ---------------------------------------------------------------- the repository's code (kalman_jax/filters.py)
def _kalman_gain(P, H, innovation_cov):
    return jnp.linalg.solve(innovation_cov, H @ P.T).T


def _joseph_covariance_update(P, K, H, R):
    identity = jnp.eye(P.shape[0], dtype=P.dtype)
    residual_factor = identity - K @ H
    return residual_factor @ P @ residual_factor.T + K @ R @ K.T


def linear_kf_predict(x, P, F, Q):
    """Pure function for the linear Kalman-filter prediction step."""
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


def run_linear_kf(measurements, F, H, Q, R, x0, P0):
    def step(carry, measurement):
        x, P = carry
        x_pred, P_pred = linear_kf_predict(x, P, F, Q)
        x_updated, P_updated, _, _ = linear_kf_update(x_pred, P_pred, measurement, H, R)
        return (x_updated, P_updated), x_updated

    (_, _), estimated_states = jax.lax.scan(step, (x0, P0), measurements)
    return estimated_states


DT = 0.1
F = jnp.array([[1.0, DT], [0.0, 1.0]])
H = jnp.array([[1.0, 0.0]])
Q_UNIT = jnp.array([[DT**3 / 3.0, DT**2 / 2.0], [DT**2 / 2.0, DT]])
SIGMA_A, SIGMA_Z = 0.1, 0.5
Q, R = Q_UNIT * SIGMA_A**2, jnp.array([[SIGMA_Z**2]])
x0, P0 = jnp.array([0.0, 1.0]), jnp.eye(2)


def simulate(key, n_steps):
    """One run of chapter 02's model, written with lax.scan so that it can be vmapped over keys.
    The initial state is drawn from the filter's prior N(x0, P0)."""
    k0, key = jax.random.split(key)
    x_start = x0 + jax.random.normal(k0, (2,))
    L = jnp.linalg.cholesky(Q)

    def step(x, k):
        kp, km = jax.random.split(k)
        x = F @ x + L @ jax.random.normal(kp, (2,))
        z = H @ x + SIGMA_Z * jax.random.normal(km, (1,))
        return x, (x, z)

    _, (xs, zs) = lax.scan(step, x_start, jax.random.split(key, n_steps))
    return xs, zs


# ---------------------------------------------------------------- 1. the idea
section("1. A function for one input, run on a batch")


def predict_one(x, W):
    return jnp.tanh(W @ x)


W = jnp.ones((3, 2)) * 0.5
xs = jnp.arange(8.0).reshape(4, 2)
loop = jnp.stack([predict_one(x, W) for x in xs])
batched = jax.vmap(predict_one, in_axes=(0, None))(xs, W)
out("loop == vmap:", bool(jnp.all(loop == batched)), batched.shape)
jx_one = str(jax.make_jaxpr(predict_one)(xs[0], W))
jx_vmap = str(jax.make_jaxpr(jax.vmap(predict_one, in_axes=(0, None)))(xs, W))
out(jx_one)
out(jx_vmap)

# ---------------------------------------------------------------- 2. the hero: linear_kf_predict under every in_axes
section("2. vmap(linear_kf_predict, in_axes=...) for a batch of 1000")
B = 1000
full = {"x": jnp.zeros((B, 2)), "P": jnp.broadcast_to(jnp.eye(2), (B, 2, 2)),
        "F": jnp.broadcast_to(F, (B, 2, 2)), "Q": jnp.broadcast_to(Q, (B, 2, 2))}
one = {"x": x0, "P": P0, "F": F, "Q": Q}
hero = {"one": str(jax.make_jaxpr(linear_kf_predict)(x0, P0, F, Q)), "cases": {}}
for axes in itertools.product([None, 0], repeat=4):
    args = [full[n] if a == 0 else one[n] for n, a in zip("xPFQ", axes)]
    key = "".join("b" if a == 0 else "-" for a in axes)
    try:
        jx = jax.make_jaxpr(jax.vmap(linear_kf_predict, in_axes=axes))(*args)
        outs = jax.eval_shape(jax.vmap(linear_kf_predict, in_axes=axes), *args)
        hero["cases"][key] = {"jaxpr": str(jx), "out": [list(o.shape) for o in outs]}
    except Exception as e:  # noqa: BLE001
        hero["cases"][key] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
for k in ("b---", "bb--", "bbbb", "----"):
    out(k, hero["cases"][k].get("jaxpr") or hero["cases"][k]["error"])

# ---------------------------------------------------------------- 3. in the unscented filter
section("3. The unscented filter's two vmaps")
diffs = jnp.arange(10.0).reshape(5, 2)
by_vmap = jax.vmap(lambda d: jnp.outer(d, d))(diffs)
by_hand = diffs[:, :, None] * diffs[:, None, :]
by_einsum = jnp.einsum("ni,nj->nij", diffs, diffs)
out("vmap(outer) == broadcasting == einsum:", bool(jnp.all(by_vmap == by_hand)), bool(jnp.all(by_vmap == by_einsum)), by_vmap.shape)
jx_outer = str(jax.make_jaxpr(jax.vmap(lambda d: jnp.outer(d, d)))(diffs))
out(jx_outer)
f_motion = lambda x: F @ x  # noqa: E731
jx_sigma = str(jax.make_jaxpr(jax.vmap(f_motion))(diffs))
out(jx_sigma)

# ---------------------------------------------------------------- 4. in_axes, out_axes and the errors
section("4. in_axes, out_axes, errors")
add = lambda a, b: a + b  # noqa: E731
axes_demo = {
    "out_axes1": list(jax.vmap(predict_one, in_axes=(0, None), out_axes=1)(xs, W).shape),
    "in_axes1": list(jax.vmap(predict_one, in_axes=(1, None))(xs.T, W).shape),
    "err_sizes": error_of(lambda: jax.vmap(add)(jnp.ones(3), jnp.ones(4))),
    "err_none": error_of(lambda: jax.vmap(add, in_axes=(None, None))(jnp.ones(3), jnp.ones(3))),
    "err_len": error_of(lambda: jax.vmap(add, in_axes=(0,))(jnp.ones(3), jnp.ones(3))),
    "err_rank": error_of(lambda: jax.vmap(add, in_axes=(1, 0))(jnp.ones(3), jnp.ones(3))),
}
for k, v in axes_demo.items():
    out(f"{k}: {v}")

seen = []


def shows(x):
    seen.append(repr(x))
    return x * 2


jax.vmap(shows)(jnp.ones((4, 2)))
axes_demo["tracer"] = seen[0]
out("inside vmap, x is", seen[0])

# ---------------------------------------------------------------- 5. control flow under vmap
section("5. lax.cond under vmap becomes a select")


def branchy(x):
    return lax.cond(x > 0, lambda v: jnp.sqrt(v), lambda v: -v, x)


jx_cond_one = str(jax.make_jaxpr(branchy)(1.0))
jx_cond_vmap = str(jax.make_jaxpr(jax.vmap(branchy))(jnp.ones(4)))
out(jx_cond_one)
out(jx_cond_vmap)
counts = []


def collatz_steps(n):
    def body(c):
        n, k = c
        return jnp.where(n % 2 == 0, n // 2, 3 * n + 1), k + 1

    return lax.while_loop(lambda c: c[0] != 1, body, (n, 0))[1]


ns = jnp.array([1, 2, 3, 6, 7, 27])
loop_counts = [int(collatz_steps(n)) for n in ns]
vmap_counts = [int(v) for v in jax.vmap(collatz_steps)(ns)]
out("while_loop per element:", loop_counts, " vmapped:", vmap_counts)
control = {"collatz_n": [int(n) for n in ns], "collatz": vmap_counts}

# ---------------------------------------------------------------- 6. composing transformations
section("6. Per-example gradients and jacfwd = vmap(jvp)")


def loss(w, x, y):
    return (jnp.dot(w, x) - y) ** 2


wv = jnp.array([1.0, -1.0])
Xs = jnp.array([[1.0, 2.0], [3.0, 1.0], [0.0, 1.0]])
Ys = jnp.array([0.0, 1.0, 2.0])
per_ex = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))(wv, Xs, Ys)
mean_grad = jax.grad(lambda w: jnp.mean(jax.vmap(loss, in_axes=(None, 0, 0))(w, Xs, Ys)))(wv)
out("per-example gradients:", per_ex, " their mean:", per_ex.mean(0), " grad of the mean:", mean_grad)
g_fn = lambda x: jnp.array([x[0] * x[1], jnp.sin(x[0]), x[1] ** 2])  # noqa: E731
xp = jnp.array([0.5, 2.0])
J_fwd = jax.jacfwd(g_fn)(xp)
J_vjvp = jax.vmap(lambda t: jax.jvp(g_fn, (xp,), (t,))[1], out_axes=1)(jnp.eye(2))
out("jacfwd == vmap(jvp):", bool(jnp.all(J_fwd == J_vjvp)))
compose = {"per_ex": per_ex.tolist(), "mean": per_ex.mean(0).tolist(), "grad_mean": mean_grad.tolist(),
           "jac_equal": bool(jnp.all(J_fwd == J_vjvp))}

# ---------------------------------------------------------------- 7. timings: many filters
section("7. Running B filters of 1000 steps: a Python loop of jitted calls against one vmapped call")
T = 1000
run_jit = jax.jit(run_linear_kf)
run_batch = jax.jit(jax.vmap(run_linear_kf, in_axes=(0, None, None, None, None, None, None)))
keys = jax.random.split(jax.random.PRNGKey(7), 10000)
_, zs_all = jax.jit(jax.vmap(simulate, in_axes=(0, None)), static_argnums=1)(keys, T)
zs_list = list(zs_all[:1000])
timing = []
for b in (1, 10, 100, 1000, 10000):
    zb = zs_all[:b]
    t_vmap = median_time(lambda: run_batch(zb, F, H, Q, R, x0, P0), reps=5)
    row = {"B": b, "vmap_ms": round(t_vmap * 1e3, 3)}
    if b <= 1000:
        zl = zs_list[:b]
        row["loop_ms"] = round(median_time(lambda: [run_jit(z, F, H, Q, R, x0, P0) for z in zl], reps=3) * 1e3, 3)
    timing.append(row)
    out(row)
same = float(jnp.max(jnp.abs(run_batch(zs_all[:100], F, H, Q, R, x0, P0)[37] - run_jit(zs_all[37], F, H, Q, R, x0, P0))))
out("batched filter 37 equals the single run to", same)

# ---------------------------------------------------------------- 8. consistency: NEES and NIS over 1000 runs
section("8. Is the filter consistent? Average NEES and NIS over 1000 simulated runs, 200 steps")
N_RUNS, T_NEES = 1000, 200
true_xs, nees_zs = jax.jit(jax.vmap(simulate, in_axes=(0, None)), static_argnums=1)(
    jax.random.split(jax.random.PRNGKey(2024), N_RUNS), T_NEES)


def nees_nis_one(xs_true, zs, fa, fz):
    """One run of the filter that assumes noise levels fa * sigma_a and fz * sigma_z."""
    Qf, Rf = Q * fa**2, R * fz**2

    def step(carry, inp):
        x, P = carry
        xt, z = inp
        xp, Pp = linear_kf_predict(x, P, F, Qf)
        xu, Pu, v, S = linear_kf_update(xp, Pp, z, H, Rf)
        e = xt - xu
        return (xu, Pu), (e @ jnp.linalg.solve(Pu, e), v @ jnp.linalg.solve(S, v))

    return lax.scan(step, (x0, P0), (xs_true, zs))[1]


nees_batch = jax.jit(jax.vmap(nees_nis_one, in_axes=(0, 0, None, None)))
FA, FZ = [0.25, 0.5, 1.0, 2.0, 4.0], [0.5, 1.0, 2.0]
nees = {}
for fa in FA:
    for fz in FZ:
        e, n = nees_batch(true_xs, nees_zs, fa, fz)
        nees[f"{fa}_{fz}"] = {"anees": [round(float(v), 4) for v in e.mean(0)],
                              "anis": [round(float(v), 4) for v in n.mean(0)],
                              "anees_all": round(float(e.mean()), 4), "anis_all": round(float(n.mean()), 4)}
        out(f"assumed sigma_a x{fa}, sigma_z x{fz}: average NEES {float(e.mean()):.3f} (expect 2), average NIS {float(n.mean()):.3f} (expect 1)")
from scipy.stats import chi2  # noqa: E402

band_nees = [chi2.ppf(0.025, 2 * N_RUNS) / N_RUNS, chi2.ppf(0.975, 2 * N_RUNS) / N_RUNS]
band_nis = [chi2.ppf(0.025, N_RUNS) / N_RUNS, chi2.ppf(0.975, N_RUNS) / N_RUNS]
out("95% band for one step's average over 1000 runs: NEES", band_nees, " NIS", band_nis)
inside = sum(band_nees[0] <= v <= band_nees[1] for v in nees["1.0_1.0"]["anees"])
out(f"correct filter: {inside} of {T_NEES} steps inside the NEES band")

# ---------------------------------------------------------------- 9. J3's question: re-estimate the noise on 1000 datasets
section("9. J3's maximum-likelihood estimate of the noise levels, repeated on the repository's generator, keys 0 to 999")


def generate_linear_synthetic_data(key, n_steps=100, dt=0.1, process_noise_std=0.1, measurement_noise_std=0.5):
    """kalman_jax/filters.py's generator with the same keys, splits and draws (its initial_state argument dropped,
    and only the states and measurements returned): a Python loop, so it cannot be vmapped efficiently."""
    F = jnp.array([[1.0, dt], [0.0, 1.0]])
    H = jnp.array([[1.0, 0.0]])
    Q = jnp.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]) * process_noise_std**2
    x_true = jnp.array([0.0, 1.0])
    true_states, measurements = [], []
    for _ in range(n_steps):
        key, process_key = jax.random.split(key)
        x_true = F @ x_true + jax.random.multivariate_normal(process_key, jnp.zeros(2), Q)
        true_states.append(x_true)
        key, measurement_key = jax.random.split(key)
        measurements.append(H @ x_true + jax.random.normal(measurement_key, (1,)) * measurement_noise_std)
    return jnp.array(true_states), jnp.vstack(measurements)


def generate_scan(key, n_steps=1000):
    """The same generator as a lax.scan: the same keys, splits and draws, so the same numbers, and vmappable."""
    def step(carry, _):
        key, x = carry
        key, process_key = jax.random.split(key)
        x = F @ x + jax.random.multivariate_normal(process_key, jnp.zeros(2), Q)
        key, measurement_key = jax.random.split(key)
        z = H @ x + jax.random.normal(measurement_key, (1,)) * SIGMA_Z
        return (key, x), (x, z)

    _, (xs, zs) = lax.scan(step, (key, jnp.array([0.0, 1.0])), None, length=n_steps)
    return xs, zs


_, zs_repo = generate_linear_synthetic_data(jax.random.PRNGKey(42), n_steps=1000)
_, zs_scan = jax.jit(generate_scan)(jax.random.PRNGKey(42))
same_gen = float(jnp.max(jnp.abs(zs_repo - zs_scan)))
out("scan replica against the repository's generator, key 42: max difference", same_gen)


def nll(theta, zs):
    """J3's negative log-likelihood: filter started at x0 = 0, P0 = I, as in the repository's script."""
    sa, sz = jnp.exp(theta[0]), jnp.exp(theta[1])
    Qt, Rt = Q_UNIT * sa**2, jnp.eye(1) * sz**2

    def step(carry, z):
        x, P = carry
        xp, Pp = linear_kf_predict(x, P, F, Qt)
        xu, Pu, v, S = linear_kf_update(xp, Pp, z, H, Rt)
        return (xu, Pu), 0.5 * (jnp.log(2 * jnp.pi) + jnp.log(S[0, 0]) + v[0] ** 2 / S[0, 0])

    return jnp.mean(lax.scan(step, (jnp.zeros(2), jnp.eye(2)), zs)[1])


def fit(zs):
    """Newton's method from the generating values, 12 steps, each clipped to at most 1 in log space."""
    g, h = jax.grad(nll), jax.hessian(nll)

    def body(theta, _):
        step = jnp.linalg.solve(h(theta, zs), g(theta, zs))
        return theta - jnp.clip(step, -1.0, 1.0), None

    theta, _ = lax.scan(body, jnp.log(jnp.array([SIGMA_A, SIGMA_Z])), None, length=12)
    return theta, jnp.max(jnp.abs(g(theta, zs)))


N_DATA = 1000
fit_keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(N_DATA))
_, fit_zs = jax.jit(jax.vmap(generate_scan))(fit_keys)
t0 = time.perf_counter()
thetas, gmax = jax.block_until_ready(jax.jit(jax.vmap(fit))(fit_zs))
t_fit = time.perf_counter() - t0
sa_hat, sz_hat = np.exp(np.asarray(thetas[:, 0])), np.exp(np.asarray(thetas[:, 1]))
ok = np.asarray(gmax) < 1e-4
out(f"{int(ok.sum())} of {N_DATA} fits converged (max |gradient| < 1e-4); {t_fit:.1f} s for all, compile included")
q = lambda a, p: float(np.quantile(a[ok], p))  # noqa: E731
fits = {"sa": [round(float(v), 5) for v in sa_hat], "sz": [round(float(v), 5) for v in sz_hat],
        "n": int(ok.sum()), "seconds": round(t_fit, 1), "same_gen": same_gen,
        "key42": [float(sa_hat[42]), float(sz_hat[42])],
        "sa_median": q(sa_hat, 0.5), "sa_q05": q(sa_hat, 0.05), "sa_q95": q(sa_hat, 0.95),
        "sz_median": q(sz_hat, 0.5), "sz_q05": q(sz_hat, 0.05), "sz_q95": q(sz_hat, 0.95),
        "sa_at_or_below_key42": int(np.sum(sa_hat[ok] <= sa_hat[42])),
        "sz_at_or_above_key42": int(np.sum(sz_hat[ok] >= sz_hat[42]))}
out({k: v for k, v in fits.items() if k not in ("sa", "sz")})

if not DATA:
    section("10. For the text and the exercises")
    lo, hi = band_nis
    for k, v in nees.items():
        out(f"assumed (sigma_a x, sigma_z x) = {k}: NIS inside its per-step band at "
            f"{sum(lo <= a <= hi for a in v['anis'])} of {len(v['anis'])} steps, NEES at "
            f"{sum(band_nees[0] <= a <= band_nees[1] for a in v['anees'])}")
    n_all = N_RUNS * T_NEES
    out("95% band for the NIS averaged over all", n_all, "innovations:",
        [1 - 1.96 * math.sqrt(2 / n_all), 1 + 1.96 * math.sqrt(2 / n_all)])
    ta = jnp.linspace(jnp.log(0.01), jnp.log(1.5), 40)
    tz = jnp.linspace(jnp.log(0.2), jnp.log(1.5), 40)
    grid = jnp.stack(jnp.meshgrid(ta, tz), axis=-1).reshape(-1, 2)
    values = jax.jit(jax.vmap(lambda t: nll(t, zs_scan)))(grid)
    out("exercise 2: grid point with the smallest likelihood (sigma_a, sigma_z):", jnp.exp(grid[jnp.argmin(values)]))
    Fn, Hn, Qu = np.asarray(F, np.float64), np.asarray(H, np.float64), np.asarray(Q_UNIT, np.float64)

    def steady(fa):
        """Steady-state predicted and updated covariance, innovation variance and gain of the filter assuming fa * sigma_a."""
        Qf, P = Qu * (SIGMA_A * fa) ** 2, np.eye(2)
        for _ in range(3000):
            Pp = Fn @ P @ Fn.T + Qf
            S = (Hn @ Pp @ Hn.T)[0, 0] + SIGMA_Z**2
            Kg = Pp @ Hn.T / S
            A = np.eye(2) - Kg @ Hn
            P = A @ Pp @ A.T + Kg @ Kg.T * SIGMA_Z**2
        return Pp, S, Kg, P

    Pp1, S1, K1, P1 = steady(1.0)
    Pp4, S4, K4, P4 = steady(4.0)
    E = np.eye(2)                                       # the x4 filter's actual error covariance under the true Q
    for _ in range(5000):
        Ep = Fn @ E @ Fn.T + Qu * SIGMA_A**2
        A = np.eye(2) - K4 @ Hn
        E = A @ Ep @ A.T + K4 @ K4.T * SIGMA_Z**2
    true_S4 = (Hn @ Ep @ Hn.T)[0, 0] + SIGMA_Z**2
    out("exercise 4: predicted position variance", Pp1[0, 0], Pp4[0, 0], " S", S1, S4,
        " actual innovation variance of the x4 filter", true_S4, " steady NIS", true_S4 / S4)
    out("exercise 4: updated covariance, x4 over x1, elementwise:", (P4 / P1).round(2).tolist())

if DATA:
    sys.stdout = REAL_STDOUT
    payload = {"jax": jax.__version__, "hero": hero, "axes": axes_demo, "control": control, "compose": compose,
               "timing": timing, "nees": nees, "FA": FA, "FZ": FZ, "band_nees": band_nees, "band_nis": band_nis,
               "fits": fits, "jaxpr": {"one": jx_one, "vmap": jx_vmap, "outer": jx_outer, "sigma": jx_sigma,
                                       "cond_one": jx_cond_one, "cond_vmap": jx_cond_vmap}}
    print("/* Kalman filters with JAX, chapter J4 data, written by code/j4_numbers.py with JAX "
          f"{jax.__version__} on a CPU: the jaxprs of vmap(linear_kf_predict) for every choice of in_axes, "
          "the timings of batched filters, average NEES and NIS over 1000 simulated runs, and the noise "
          "levels re-estimated on 1000 datasets. */")
    print("window.J4 = " + json.dumps(payload) + ";")
