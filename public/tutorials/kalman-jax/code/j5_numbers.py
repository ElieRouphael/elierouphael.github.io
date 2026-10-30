"""Reproduce every jaxpr, message, timing and filter result quoted on chapter J5 of "Kalman Filters with JAX"
(public/tutorials/kalman-jax/j5-loops-as-data-scan.html).

Usage:  python j5_numbers.py          # print everything the page quotes
        python j5_numbers.py data     # print the page's data file, interactive/j5-data.js
The float64 smoother check runs in a second process with jax_enable_x64, started automatically.
Checked with jax 0.12.0.1, numpy 2.x, CPU. Timings depend on the machine.
The Kalman functions are copied verbatim from kalman_jax/filters.py so that the script runs alone.
"""
from __future__ import annotations

import io
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

X64PART = len(sys.argv) > 1 and sys.argv[1] == "x64part"
import jax  # noqa: E402

if X64PART:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

DATA = len(sys.argv) > 1 and sys.argv[1] == "data"
REAL_STDOUT = sys.stdout
if DATA or X64PART:
    sys.stdout = io.StringIO()
out = (lambda *a, **k: None) if (DATA or X64PART) else print


def section(title):
    out(f"\n=== {title}")


def error_of(thunk):
    try:
        thunk()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).splitlines()[0]}"
    return "no error"


def error_detail(thunk):
    """The error's type, its first line and its first explanatory line."""
    try:
        thunk()
    except Exception as e:  # noqa: BLE001
        lines = [ln for ln in str(e).splitlines() if ln.strip()]
        return f"{type(e).__name__}: " + " ".join(lines[:2])
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
Q = jnp.array([[DT**3 / 3.0, DT**2 / 2.0], [DT**2 / 2.0, DT]]) * 0.1**2
R = jnp.array([[0.5**2]])
x0, P0 = jnp.zeros(2), jnp.eye(2)   # as in the repository's script


def generate_scan(key, n_steps=1000):
    """kalman_jax's generate_linear_synthetic_data as a scan; J4 showed it reproduces the original exactly."""
    def step(carry, _):
        key, x = carry
        key, process_key = jax.random.split(key)
        x = F @ x + jax.random.multivariate_normal(process_key, jnp.zeros(2), Q)
        key, measurement_key = jax.random.split(key)
        z = H @ x + jax.random.normal(measurement_key, (1,)) * 0.5
        return (key, x), (x, z)

    _, (xs, zs) = lax.scan(step, (key, jnp.array([0.0, 1.0])), None, length=n_steps)
    return xs, zs


# ---------------------------------------------------------------- the smoother: a scan in reverse
def kalman_filter_and_smoother(zs):
    """Forward scan: the repository's filter, also returning its predictions. Backward scan: Rauch-Tung-Striebel."""
    def fstep(carry, z):
        x, P = carry
        xp, Pp = linear_kf_predict(x, P, F, Q)
        xu, Pu, _, _ = linear_kf_update(xp, Pp, z, H, R)
        return (xu, Pu), (xu, Pu, xp, Pp)

    _, (xf, Pf, xp, Pp) = lax.scan(fstep, (x0, P0), zs)

    def bstep(carry, inp):
        xs_next, Ps_next = carry
        x, P, xp_next, Pp_next = inp
        G = jnp.linalg.solve(Pp_next, F @ P).T          # P F^T (P_pred)^-1; both covariances are symmetric
        xs = x + G @ (xs_next - xp_next)
        Ps = P + G @ (Ps_next - Pp_next) @ G.T
        return (xs, Ps), (xs, Ps)

    last = (xf[-1], Pf[-1])
    _, (xs_rest, Ps_rest) = lax.scan(bstep, last, (xf[:-1], Pf[:-1], xp[1:], Pp[1:]), reverse=True)
    return xf, Pf, jnp.concatenate([xs_rest, xf[-1:]]), jnp.concatenate([Ps_rest, Pf[-1:]])


def dense_posterior(zs):
    """The exact posterior mean of x_1..x_T, by solving the joint Gaussian's normal equations (float64, NumPy)."""
    zs = np.asarray(zs, dtype=np.float64)[:, 0]
    T = len(zs)
    Fn, Hn = np.asarray(F, dtype=np.float64), np.asarray(H, dtype=np.float64)
    Qi = np.linalg.inv(np.asarray(Q, dtype=np.float64))
    Ri = 1.0 / float(R[0, 0])
    P0i = np.linalg.inv(np.asarray(P0, dtype=np.float64))
    n = 2 * (T + 1)                                     # states x_0 .. x_T
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0:2, 0:2] += P0i
    b[0:2] += P0i @ np.asarray(x0, dtype=np.float64)
    for k in range(1, T + 1):
        i, j = 2 * (k - 1), 2 * k                       # x_{k-1}, x_k
        A[j:j + 2, j:j + 2] += Qi
        A[i:i + 2, i:i + 2] += Fn.T @ Qi @ Fn
        A[i:i + 2, j:j + 2] -= Fn.T @ Qi
        A[j:j + 2, i:i + 2] -= Qi @ Fn
        A[j:j + 2, j:j + 2] += np.outer(Hn[0], Hn[0]) * Ri
        b[j:j + 2] += Hn[0] * zs[k - 1] * Ri
    return np.linalg.solve(A, b).reshape(T + 1, 2)[1:]


if X64PART:
    zs64 = np.load(sys.argv[2]).astype(np.float64)
    xf, Pf, xsm, Psm = jax.jit(kalman_filter_and_smoother)(jnp.asarray(zs64)[:, None])
    dense = dense_posterior(zs64[:, None])
    r10 = lambda a: [round(float(v), 10) for v in a]  # noqa: E731
    res = {"filter_pos": r10(xf[:, 0]), "filter_vel": r10(xf[:, 1]), "filter_var": r10(Pf[:, 0, 0]),
           "smooth_pos": r10(xsm[:, 0]), "smooth_vel": r10(xsm[:, 1]), "smooth_var": r10(Psm[:, 0, 0]),
           "dense_diff": float(np.max(np.abs(np.asarray(xsm) - dense)))}
    sys.stdout = REAL_STDOUT
    print(json.dumps(res))
    sys.exit(0)

out("jax", jax.__version__, "numpy", np.__version__)
xs_true, zs = jax.jit(generate_scan)(jax.random.PRNGKey(42))

# ---------------------------------------------------------------- 1. unrolling
section("1. A Python loop under jit is unrolled; lax.scan is not")


def filter_python_loop(zs, F, H, Q, R, x0, P0):
    x, P, outs = x0, P0, []
    for k in range(zs.shape[0]):                       # a Python loop: traced once per iteration
        x, P = linear_kf_predict(x, P, F, Q)
        x, P, _, _ = linear_kf_update(x, P, zs[k], H, R)
        outs.append(x)
    return jnp.stack(outs)


def stage_times(fn, *args):
    jax.clear_caches()
    t0 = time.perf_counter()
    tr = jax.jit(fn).trace(*args)
    t1 = time.perf_counter()
    lo = tr.lower()
    t2 = time.perf_counter()
    co = lo.compile()
    t3 = time.perf_counter()
    run = median_time(lambda: co(*args), reps=7)
    return {"eqns": len(tr.jaxpr.eqns), "trace_ms": round((t1 - t0) * 1e3, 1), "lower_ms": round((t2 - t1) * 1e3, 1),
            "compile_ms": round((t3 - t2) * 1e3, 1), "run_us": round(run * 1e6, 1)}


jax.jit(lambda a: jnp.sin(a) @ a.T).trace(jnp.ones((3, 3))).lower().compile()   # warm-up
unroll_rows = []
for T in (10, 30, 100, 300, 1000):
    z = zs[:T]
    row = {"T": T, "loop": stage_times(filter_python_loop, z, F, H, Q, R, x0, P0),
           "scan": stage_times(run_linear_kf, z, F, H, Q, R, x0, P0)}
    unroll_rows.append(row)
    out(row)
same_loop = float(jnp.max(jnp.abs(jax.jit(filter_python_loop)(zs[:300], F, H, Q, R, x0, P0)
                                  - jax.jit(run_linear_kf)(zs[:300], F, H, Q, R, x0, P0))))
out("Python loop and scan agree to", same_loop)

# ---------------------------------------------------------------- 2. scan's meaning, and its jaxpr
section("2. What lax.scan means, written as Python, and its jaxpr")


def scan_reference(f, init, xs):
    carry, ys = init, []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


def cumsum_step(c, x):
    c = c + x
    return c, c


ref = scan_reference(cumsum_step, 0.0, jnp.arange(1.0, 6.0))
lib = lax.scan(cumsum_step, 0.0, jnp.arange(1.0, 6.0))
out("reference:", ref, " lax.scan:", lib)
jx_scan = str(jax.make_jaxpr(lambda zs: lax.scan(cumsum_step, 0.0, zs))(jnp.arange(1.0, 6.0)))
out(jx_scan)

# ---------------------------------------------------------------- 3. fori_loop and while_loop
section("3. fori_loop with static bounds is a scan; with traced bounds it is a while_loop")


def prim(jx):
    return sorted({e.primitive.name for e in jx.eqns})


fl_static = prim(jax.make_jaxpr(lambda x: lax.fori_loop(0, 10, lambda i, c: c * x, 1.0))(2.0).jaxpr)
fl_dynamic = prim(jax.make_jaxpr(lambda x, n: lax.fori_loop(0, n, lambda i, c: c * x, 1.0))(2.0, 10).jaxpr)
grad_static = float(jax.grad(lambda x: lax.fori_loop(0, 10, lambda i, c: c * x, 1.0))(2.0))
# under jit, n is traced, so the bounds are unknown while tracing and fori_loop becomes a while_loop
grad_dynamic = error_of(lambda: jax.jit(jax.grad(lambda x, n: lax.fori_loop(0, n, lambda i, c: c * x, 1.0)))(2.0, 10))
out("static bounds:", fl_static, " d/dx x^10 at 2 =", grad_static)
out("traced bounds:", fl_dynamic, " ", grad_dynamic)

# ---------------------------------------------------------------- 4. the carry must keep its type
section("4. The carry's shape and dtype must not change")
errs = {
    "dtype": error_detail(lambda: lax.scan(lambda c, x: (c + 0.5, x), jnp.int32(0), jnp.arange(3.0))),
    "shape": error_detail(lambda: lax.scan(lambda c, x: (jnp.append(c, x), x), jnp.zeros(1), jnp.arange(3.0))),
    "weak_int": float(lax.scan(lambda c, x: (c + 0.5, x), 0, jnp.arange(3.0))[0]),
    "py_if": error_of(lambda: lax.scan(lambda c, x: ((c + x) if x > 1 else c, x), 0.0, jnp.arange(3.0))),
}
for k, v in errs.items():
    out(f"{k}: {v}")
traced_once = []


def counting_step(c, x):
    traced_once.append(1)
    return c + x, x


lax.scan(counting_step, 0.0, jnp.arange(1000.0))
errs["body_traced"] = len(traced_once)
out("the body of a 1000-step scan ran in Python", len(traced_once), "time(s)")

# ---------------------------------------------------------------- 5. memory of reverse mode through a scan
section("5. Memory: the gradient of the filter's likelihood through T steps, with and without jax.checkpoint")


def make_nll(zs, remat):
    Q_unit = Q / 0.1**2

    def nll(theta):
        sa, sz = jnp.exp(theta[0]), jnp.exp(theta[1])
        Qt, Rt = Q_unit * sa**2, jnp.eye(1) * sz**2

        def step(carry, z):
            x, P = carry
            xp, Pp = linear_kf_predict(x, P, F, Qt)
            xu, Pu, v, S = linear_kf_update(xp, Pp, z, H, Rt)
            return (xu, Pu), 0.5 * (jnp.log(2 * jnp.pi) + jnp.log(S[0, 0]) + v[0] ** 2 / S[0, 0])

        body = jax.checkpoint(step) if remat else step
        return jnp.mean(lax.scan(body, (x0, P0), zs)[1])

    return nll


_, zs_long = jax.jit(generate_scan, static_argnums=1)(jax.random.PRNGKey(5), 100000)
mem_rows = []
theta = jnp.log(jnp.array([0.1, 0.5]))
for T in (1000, 10000, 100000):
    for remat in (False, True):
        g = jax.jit(jax.grad(make_nll(zs_long[:T], remat)))
        comp = g.trace(theta).lower().compile()
        ma = comp.memory_analysis()
        temp = int(ma.temp_size_in_bytes) if ma is not None else None
        t = median_time(lambda: comp(theta), reps=5)
        mem_rows.append({"T": T, "checkpoint": remat, "temp_bytes": temp, "ms": round(t * 1e3, 2)})
        out(mem_rows[-1])
g_plain = jax.grad(make_nll(zs[:, :], False))(theta)
g_remat = jax.grad(make_nll(zs[:, :], True))(theta)
remat_same = float(jnp.max(jnp.abs(g_plain - g_remat)))
out("gradients with and without checkpoint agree to", remat_same)

# ---------------------------------------------------------------- 6. scan's unroll parameter
section("6. unroll: how many steps each iteration of the compiled loop does")


def run_unrolled(zs, unroll):
    def step(carry, z):
        x, P = carry
        xp, Pp = linear_kf_predict(x, P, F, Q)
        xu, Pu, _, _ = linear_kf_update(xp, Pp, z, H, R)
        return (xu, Pu), xu

    return lax.scan(step, (x0, P0), zs, unroll=unroll)[1]


unroll_param = []
for u in (1, 2, 4, 8, 16, 32):
    jax.clear_caches()
    f = jax.jit(run_unrolled, static_argnums=1)
    t0 = time.perf_counter()
    jax.block_until_ready(f(zs, u))
    first = time.perf_counter() - t0
    unroll_param.append({"unroll": u, "first_ms": round(first * 1e3, 1),
                         "run_ms": round(median_time(lambda: f(zs, u), reps=15) * 1e3, 3)})
    out(unroll_param[-1])

# ---------------------------------------------------------------- 7. the smoother
section("7. A Kalman smoother in two scans, checked against the exact posterior (float64)")
xf32, Pf32, xs32, Ps32 = jax.jit(kalman_filter_and_smoother)(zs)
err = lambda a, i: float(jnp.sqrt(jnp.mean((a[:, i] - xs_true[:, i]) ** 2)))  # noqa: E731
smooth = {"rmse_filter": err(xf32, 0), "rmse_smoother": err(xs32, 0), "rmse_filter_vel": err(xf32, 1),
          "rmse_smoother_vel": err(xs32, 1), "rmse_meas": float(jnp.sqrt(jnp.mean((zs[:, 0] - xs_true[:, 0]) ** 2)))}
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "zs.npy")
    np.save(path, np.asarray(zs, dtype=np.float32)[:, 0])
    proc = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "x64part", path],
                          capture_output=True, text=True, check=True)
k64 = json.loads(proc.stdout.strip().splitlines()[-1])
smooth["dense_diff"] = k64["dense_diff"]
smooth["f32_vs_f64_smoother"] = float(np.max(np.abs(np.asarray(xs32[:, 0]) - np.array(k64["smooth_pos"]))))
out(smooth)

# ---------------------------------------------------------------- 8. associative_scan: chapter 05's filter in parallel
section("8. Chapter 05's first-order filter, y_k = z0 y_(k-1) + (1 - z0) u_k, sequentially and as a parallel prefix")
Z0 = float(np.exp(-0.005))


def lowpass_scan(u):
    def step(y, uk):
        y = Z0 * y + (1 - Z0) * uk
        return y, y
    return lax.scan(step, jnp.float32(0.0), u)[1]


def combine(left, right):
    """Compose two affine maps y -> a y + b: first left, then right."""
    a1, b1 = left
    a2, b2 = right
    return a2 * a1, a2 * b1 + b2


def lowpass_assoc(u):
    a = jnp.full_like(u, Z0)
    b = (1 - Z0) * u
    return lax.associative_scan(combine, (a, b))[1]       # y_k: the b-part of maps 1..k composed, applied to y_0 = 0


u_test = jax.random.normal(jax.random.PRNGKey(3), (1000,))
seq, par = jax.jit(lowpass_scan)(u_test), jax.jit(lowpass_assoc)(u_test)
u64 = np.asarray(u_test, dtype=np.float64)
exact = np.zeros(1000)
yv = 0.0
for k in range(1000):
    yv = Z0 * yv + (1 - Z0) * u64[k]
    exact[k] = yv
assoc = {"max_diff": float(jnp.max(jnp.abs(seq - par))), "seq_err": float(np.max(np.abs(np.asarray(seq) - exact))),
         "par_err": float(np.max(np.abs(np.asarray(par) - exact))), "y_scale": float(np.max(np.abs(exact))),
         "eqns_assoc_1024": len(jax.make_jaxpr(lowpass_assoc)(jnp.ones(1024)).eqns),
         "eqns_scan_1024": len(jax.make_jaxpr(lowpass_scan)(jnp.ones(1024)).eqns)}
out(assoc)
assoc_t = []
for T in (1000, 10000, 100000, 1000000):
    u = jax.random.normal(jax.random.PRNGKey(4), (T,))
    fs, fa = jax.jit(lowpass_scan), jax.jit(lowpass_assoc)
    assoc_t.append({"T": T, "scan_ms": round(median_time(lambda: fs(u)) * 1e3, 3),
                    "assoc_ms": round(median_time(lambda: fa(u)) * 1e3, 3)})
    out(assoc_t[-1])
assoc["timing"] = assoc_t

if DATA:
    sys.stdout = REAL_STDOUT
    r7 = lambda a: [round(float(v), 7) for v in a]  # noqa: E731
    payload = {"jax": jax.__version__, "zs": r7(np.asarray(zs)[:, 0]), "true_pos": r7(np.asarray(xs_true)[:, 0]),
               "true_vel": r7(np.asarray(xs_true)[:, 1]), "k64": k64, "smooth": smooth, "unroll": unroll_rows,
               "same_loop": same_loop, "errors": errs, "remat_same": remat_same,
               "fori": {"static": fl_static, "dynamic": fl_dynamic, "grad_static": grad_static, "grad_dynamic": grad_dynamic},
               "memory": mem_rows, "unroll_param": unroll_param, "assoc": assoc, "z0": Z0, "jaxpr": {"scan": jx_scan}}
    print("/* Kalman filters with JAX, chapter J5 data, written by code/j5_numbers.py with JAX "
          f"{jax.__version__} on a CPU: chapter 02's measurements and true states, JAX's float64 filter and smoother "
          "for checking the page's own, and the loop experiments' timings. */")
    print("window.J5 = " + json.dumps(payload) + ";")
