"""Reproduce every jaxpr, message and timing quoted on chapter J2 of "Kalman Filters with JAX"
(public/tutorials/kalman-jax/j2-pure-functions-tracing-and-jit.html).

Usage:  python j2_numbers.py          # print everything the page quotes
        python j2_numbers.py data     # print the page's data file, interactive/j2-data.js
Checked with jax 0.12.0.1, numpy 2.x, CPU. Timings depend on the machine; the ratios are what matter.
The Kalman functions below are copied verbatim from kalman_jax/filters.py so that the script runs alone.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import re
import statistics
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import source_info_util

DATA = len(sys.argv) > 1 and sys.argv[1] == "data"
REAL_STDOUT = sys.stdout
if DATA:  # silence the prints that the examples make while tracing; only the data file is written
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


def median_time(fn, reps=7, inner=1):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            r = fn()
        jax.block_until_ready(r)
        ts.append((time.perf_counter() - t0) / inner)
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


def generate_linear_synthetic_data(key, n_steps=100, dt=0.1, process_noise_std=0.1, measurement_noise_std=0.5):
    F = jnp.array([[1.0, dt], [0.0, 1.0]])
    H = jnp.array([[1.0, 0.0]])
    Q = jnp.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]) * process_noise_std**2
    R = jnp.array([[measurement_noise_std**2]])
    x_true = jnp.array([0.0, 1.0])
    true_states, measurements = [], []
    for _ in range(n_steps):
        key, process_key = jax.random.split(key)
        x_true = F @ x_true + jax.random.multivariate_normal(process_key, jnp.zeros(2), Q)
        true_states.append(x_true)
        key, measurement_key = jax.random.split(key)
        measurements.append(H @ x_true + jax.random.normal(measurement_key, (1,)) * measurement_noise_std)
    return F, H, Q, R, jnp.array(true_states), jnp.vstack(measurements)


def kf_step(x, P, z, F, H, Q, R):
    """One predict-and-update step, every input an argument."""
    x_pred, P_pred = linear_kf_predict(x, P, F, Q)
    x_new, P_new, _, _ = linear_kf_update(x_pred, P_pred, z, H, R)
    return x_new, P_new


F, H, Q, R, true_states, zs = generate_linear_synthetic_data(jax.random.PRNGKey(0), n_steps=1000)
x0, P0 = jnp.array([0.0, 1.0]), jnp.eye(2)

# ---------------------------------------------------------------- the hero: functions traced line by line
console: list[str] = []


def fn_prints(x):
    print("x is", x)
    y = 2.0 * x + 1.0
    return y


scale = 2.0


def fn_global(x):
    y = scale * x
    return y


def fn_if(x):
    if x > 0:
        return x
    return 0.0 * x


def fn_where(x):
    y = jnp.where(x > 0, x, 0.0)
    return y


def traced_lines(fn, *args):
    """make_jaxpr, split into equations, each tagged with the line of fn that produced it."""
    jaxpr = jax.make_jaxpr(fn)(*args)
    source, first = inspect.getsourcelines(fn)
    text = str(jaxpr).splitlines()
    if len(text) == 1:  # JAX prints a short jaxpr on one line; lay it out like a long one
        m = re.match(r"^(\{ lambda .*?\. let) (.*) (in \(.*\) \})$", text[0])
        text = [m.group(1)] + ["    " + e for e in m.group(2).split("; ")] + ["  " + m.group(3)]
    header, body, footer = text[0], text[1:-1], text[-1]
    eq_text, cur = [], None
    for line in body:
        if re.match(r"^    [a-z_]+:", line):
            cur = [line[4:]]
            eq_text.append(cur)
        else:
            cur.append(line[4:])
    lines = []
    for eqn, t in zip(jaxpr.eqns, eq_text):
        frame = source_info_util.user_frame(eqn.source_info.traceback)
        lines.append({"line": frame.start_line - first, "text": "\n".join(t)})
    return {"source": "".join(source).rstrip(), "header": header, "eqns": lines, "footer": footer}


hero = {}
v2 = jnp.array([1.0, -2.0])
hero["predict"] = traced_lines(linear_kf_predict, x0, P0, F, Q)
with contextlib.redirect_stdout(io.StringIO()) as printed:
    hero["prints"] = traced_lines(fn_prints, v2)
hero["prints_console"] = printed.getvalue().strip()
hero["global"] = traced_lines(fn_global, v2)
hero["where"] = traced_lines(fn_where, v2)
hero["if_partial"] = re.search(r"let (.*) in", str(jax.make_jaxpr(lambda x: x > 0)(jnp.float32(1.0)))).group(1).strip()
hero["if_source"] = inspect.getsource(fn_if).rstrip()
hero["if_error"] = error_of(lambda: jax.jit(fn_if)(jnp.float32(2.0)))
hero["prints_tracer"] = None


@jax.jit
def _f_for_print(x):
    hero["prints_tracer"] = repr(x)
    return x


_f_for_print(v2)

section("1. The jaxpr of linear_kf_predict")
out(jax.make_jaxpr(linear_kf_predict)(x0, P0, F, Q))
lowered = jax.jit(linear_kf_predict).trace(x0, P0, F, Q).lower()
section("1b. Lowered to StableHLO")
out(lowered.as_text())
section("1c. After XLA's optimisation: the transpose is gone, folded into a dot over F's second axis")
hlo = lowered.compile().as_text()
out("\n".join(l for l in hlo.splitlines() if re.search(r"= f32\[[\d,]*\]\{[\d,]*\} (dot|fusion|transpose)\(", l)))

section("2. print runs while tracing, not when the compiled code runs")


@jax.jit
def f(x):
    print("tracing f with", x)
    return 2.0 * x + 1.0


out(f(jnp.array([1.0, 2.0])))
out(f(jnp.array([3.0, 4.0])), " <- no print: the cached program ran")
out(f(jnp.array([1.0, 2.0, 3.0])), " <- new shape, traced again")


@jax.jit
def f_debug(x):
    jax.debug.print("running with x = {x}", x=x)
    return 2.0 * x + 1.0


f_debug(jnp.array([1.0, 2.0])).block_until_ready()
f_debug(jnp.array([3.0, 4.0])).block_until_ready()

section("3. A global is read once, at trace time")


@jax.jit
def g(x):
    return scale * x


out(g(jnp.float32(1.0)))
scale = 3.0
out(g(jnp.float32(1.0)), " <- still 2.0 after scale = 3.0")
out(g(jnp.ones(2)), " <- a new shape retraces and sees 3.0")
scale = 2.0

section("4. A tracer that escapes")
seen = []


@jax.jit
def h(x):
    seen.append(x)
    return x + 1


h(jnp.float32(1.0))
out(seen)
out(error_of(lambda: seen[0] + 1))

section("5. Branching on a value")
out(hero["if_error"])


@jax.jit
def relu_where(x):
    return jnp.where(x > 0, x, 0.0)


@jax.jit
def relu_cond(x):
    return lax.cond(x > 0, lambda v: v, lambda v: jnp.zeros_like(v), x)


out(relu_where(jnp.array([-1.0, 2.0])), relu_cond(jnp.float32(-3.0)), relu_cond(jnp.float32(3.0)))
out(jax.make_jaxpr(lambda x: lax.cond(x > 0, lambda v: v, lambda v: jnp.zeros_like(v), x))(jnp.float32(1.0)))


@partial(jax.jit, static_argnums=1)
def power(x, n):
    print("tracing power with n =", n)
    y = x
    for _ in range(n - 1):
        y = y * x
    return y


out(power(jnp.float32(2.0), 3), power(jnp.float32(2.0), 3), power(jnp.float32(2.0), 5))
out(jax.make_jaxpr(power.__wrapped__, static_argnums=1)(jnp.float32(2.0), 3))
out(error_of(lambda: jax.jit(power.__wrapped__)(jnp.float32(2.0), 3)))


@jax.jit
def norm_or_abs(x):
    if x.ndim == 1:  # a shape is known while tracing: a Python if on it is fine
        return jnp.sqrt(jnp.sum(x * x))
    return jnp.abs(x)


out(norm_or_abs(jnp.array([3.0, 4.0])), norm_or_abs(jnp.float32(-2.0)))
out(error_of(lambda: jax.jit(lambda x: x[x > 0])(jnp.array([-1.0, 2.0, 3.0]))))
out(jax.jit(lambda x: jnp.sum(jnp.where(x > 0, x, 0.0)))(jnp.array([-1.0, 2.0, 3.0])))
out(error_of(lambda: jax.jit(lambda f, x: f(x))(jnp.sin, 1.0)))

section("5b. NumPy randomness inside jit is drawn once, while tracing")
np.random.seed(0)


@jax.jit
def noisy(x):
    return x + np.random.normal()


out(noisy(jnp.float32(0.0)), noisy(jnp.float32(0.0)), noisy(jnp.float32(0.0)), " <- the same 'random' number three times")

section("6. What the cache is keyed on: each line says whether the call traced again")
traces = []


@jax.jit
def k(x):
    traces.append(str(jax.typeof(x)))
    return x * 2


cache_calls = [
    ("jnp.array([1.0, 2.0])", jnp.array([1.0, 2.0])),
    ("jnp.array([5.0, 6.0])", jnp.array([5.0, 6.0])),
    ("np.array([1.0, 2.0])  (NumPy float64)", np.array([1.0, 2.0])),
    ("jnp.array([1.0, 2.0, 3.0])", jnp.array([1.0, 2.0, 3.0])),
    ("jnp.array([1, 2])", jnp.array([1, 2])),
    ("1.0  (a Python float)", 1.0),
    ("7.0  (another Python float)", 7.0),
    ("jnp.float32(1.0)", jnp.float32(1.0)),
]
cache_rows = []
for label, arg in cache_calls:
    n = len(traces)
    k(arg)
    traced = len(traces) > n
    cache_rows.append({"call": label, "traced": traced, "type": traces[-1] if traced else None})
    out(f"k({label}): {'traced as ' + traces[-1] if traced else 'cache hit'}")

section("6b. The demo's calls to a jitted Kalman step: which ones trace again")
step_traces = []


@jax.jit
def kf_step_counted(x, P, z, F, H, Q, R):
    step_traces.append(tuple(str(jax.typeof(a)) for a in (x, P, z)))
    return kf_step(x, P, z, F, H, Q, R)


F4, H4, Q4 = jnp.eye(4), jnp.ones((1, 4)), 0.01 * jnp.eye(4)
widget_calls = [
    ("chapter02", (x0, P0, zs[0], F, H, Q, R)),
    ("newvalues", (x0 + 1.0, 2.0 * P0, zs[1], F, H, Q, R)),
    ("numpy64", (x0, P0, np.asarray(zs[2], dtype=np.float64), F, H, Q, R)),
    ("state4", (jnp.zeros(4), jnp.eye(4), zs[0], F4, H4, Q4, R)),
    ("intz", (x0, P0, jnp.array([3], jnp.int32), F, H, Q, R)),
    ("chapter02again", (x0, P0, zs[3], F, H, Q, R)),
]
widget_rows = {}
for name, args in widget_calls:
    n = len(step_traces)
    kf_step_counted(*args)
    widget_rows[name] = list(step_traces[-1]) if len(step_traces) > n else None
    out(name, "traced as " + ", ".join(step_traces[-1]) if len(step_traces) > n else "cache hit")

section("8. For the exercises")
counter = 0
offset = 1.0


@jax.jit
def puzzle(x):
    global counter
    counter += 1
    print("tracing")
    return x + offset


r1 = puzzle(jnp.ones(3))
offset = 10.0
r2 = puzzle(jnp.zeros(3))
r3 = puzzle(jnp.ones(4))
out("puzzle results:", r1, r2, r3, "counter =", counter)


def gate_if(z, z_pred, threshold):
    r = z - z_pred
    if jnp.abs(r) > threshold:  # an outlier: ignore it
        r = 0.0 * r
    return r


@jax.jit
def gate_where(z, z_pred, threshold):
    r = z - z_pred
    return jnp.where(jnp.abs(r) > threshold, 0.0, r)


out(error_of(lambda: jax.jit(gate_if)(jnp.array([5.0]), jnp.array([1.0]), 3.0)))
out("gate_where:", gate_where(jnp.array([5.0]), jnp.array([1.0]), 3.0), gate_where(jnp.array([2.0]), jnp.array([1.0]), 3.0))


def run_ekf(measurements, f, h, F_jacobian, H_jacobian, Q, R, x0, P0):  # kalman_jax/filters.py, shortened
    def step(carry, z):
        x, P = carry
        x_pred, F_ = f(x), F_jacobian(x)
        P_pred = F_ @ P @ F_.T + Q
        H_ = H_jacobian(x_pred)
        S = H_ @ P_pred @ H_.T + R
        K = jnp.linalg.solve(S, H_ @ P_pred.T).T
        x_new = x_pred + K @ (z - h(x_pred))
        A_ = jnp.eye(2) - K @ H_
        return (x_new, A_ @ P_pred @ A_.T + K @ R @ K.T), x_new
    return jax.lax.scan(step, (x0, P0), measurements)[1]


f_lin, h_lin = (lambda x: F @ x), (lambda x: H @ x)
Fj, Hj = jax.jacfwd(f_lin), jax.jacfwd(h_lin)
out(error_of(lambda: jax.jit(run_ekf)(zs, f_lin, h_lin, Fj, Hj, Q, R, x0, P0)))
ekf_traces = []


def run_ekf_counted(*args):
    ekf_traces.append(1)
    return run_ekf(*args)


jekf = jax.jit(run_ekf_counted, static_argnums=(1, 2, 3, 4))
e1 = jekf(zs, f_lin, h_lin, Fj, Hj, Q, R, x0, P0)
jekf(zs, f_lin, h_lin, Fj, Hj, Q, R, x0, P0)
out("same functions twice: traces =", len(ekf_traces))
jekf(zs, lambda x: F @ x, h_lin, Fj, Hj, Q, R, x0, P0)
out("a new lambda: traces =", len(ekf_traces))
out("EKF on the linear model equals the linear filter:", float(jnp.max(jnp.abs(e1 - jax.jit(run_linear_kf)(zs, F, H, Q, R, x0, P0)))))


@jax.jit
def kf_step_missing(x, P, z, F, H, Q, R):
    x_pred, P_pred = linear_kf_predict(x, P, F, Q)
    x_upd, P_upd, _, _ = linear_kf_update(x_pred, P_pred, jnp.nan_to_num(z), H, R)
    seen = ~jnp.any(jnp.isnan(z))
    return jnp.where(seen, x_upd, x_pred), jnp.where(seen, P_upd, P_pred)


xm, Pm = kf_step_missing(x0, P0, jnp.array([jnp.nan]), F, H, Q, R)
xp, Pp = linear_kf_predict(x0, P0, F, Q)
out("missing z: equals the prediction:", bool(jnp.all(xm == xp) & jnp.all(Pm == Pp)), xm, Pm[0, 0])
xs_, Ps_ = kf_step_missing(x0, P0, zs[0], F, H, Q, R)
xr, Pr = kf_step(x0, P0, zs[0], F, H, Q, R)
out("present z: equals the full step:", float(jnp.max(jnp.abs(xs_ - xr))), xs_)
naive = jax.jit(lambda x, P, z: linear_kf_update(*linear_kf_predict(x, P, F, Q), z, H, R)[0])(x0, P0, jnp.array([jnp.nan]))
out("without the guard, a NaN measurement gives", naive)

section("7. Timings (this machine)")
small = jnp.float32(1.0)
t_op = median_time(lambda: small + 1.0, reps=15, inner=2000)
npv = np.float32(1.0)
t_np_op = median_time(lambda: npv + np.float32(1.0), reps=15, inner=20000)
n_ops = len(jax.make_jaxpr(kf_step)(x0, P0, zs[0], F, H, Q, R).eqns)
z0 = zs[0]  # sliced once: slicing a JAX array is itself an operation with its own cost
t_eager_step = median_time(lambda: kf_step(x0, P0, z0, F, H, Q, R), reps=15, inner=20)


def stages(fn, *args, reps=5):
    rows = []
    for _ in range(reps):
        jax.clear_caches()
        t0 = time.perf_counter()
        traced = jax.jit(fn).trace(*args)
        t1 = time.perf_counter()
        low = traced.lower()
        t2 = time.perf_counter()
        comp = low.compile()
        t3 = time.perf_counter()
        run = median_time(lambda: comp(*args), reps=7, inner=200)
        rows.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, run * 1e6))
    return [round(statistics.median(c), 3) for c in zip(*rows)]


jax.jit(lambda a: jnp.sin(a) @ a.T).trace(jnp.ones((3, 3))).lower().compile()  # warm JAX up first
st2 = stages(kf_step, x0, P0, zs[0], F, H, Q, R)
st4 = stages(kf_step, jnp.zeros(4), jnp.eye(4), zs[0], F4, H4, Q4, R)
sti = stages(kf_step, x0, P0, jnp.array([1], jnp.int32), F, H, Q, R)
jstep = jax.jit(kf_step)  # made after the stage timings, which clear JAX's caches
jstep(x0, P0, z0, F, H, Q, R)
t_jit_step = median_time(lambda: jstep(x0, P0, z0, F, H, Q, R), reps=15, inner=200)

zs_rows = list(zs)


def loop_eager():
    x, P = x0, P0
    for z in zs_rows:
        x, P = kf_step(x, P, z, F, H, Q, R)
    return x


def loop_jitted_step():
    x, P = x0, P0
    for z in zs_rows:
        x, P = jstep(x, P, z, F, H, Q, R)
    return x


Fn, Hn, Qn, Rn, zn = (np.asarray(a) for a in (F, H, Q, R, zs))


def loop_numpy():
    x, P, eye = np.array([0.0, 1.0], np.float32), np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32)
    for z in zn:
        x, P = Fn @ x, Fn @ P @ Fn.T + Qn
        K = np.linalg.solve(Hn @ P @ Hn.T + Rn, Hn @ P.T).T
        x = x + K @ (z - Hn @ x)
        A = eye - K @ Hn
        P = A @ P @ A.T + K @ Rn @ K.T
    return x


t_loop_eager = median_time(loop_eager, reps=3)
t_loop_jit = median_time(loop_jitted_step, reps=5)
t_loop_np = median_time(loop_numpy, reps=5)
t_run_plain = median_time(lambda: run_linear_kf(zs, F, H, Q, R, x0, P0), reps=5)
jrun = jax.jit(run_linear_kf)
jax.clear_caches()
t0 = time.perf_counter()
jax.block_until_ready(jrun(zs, F, H, Q, R, x0, P0))
t_jrun_first = time.perf_counter() - t0
t_jrun = median_time(lambda: jrun(zs, F, H, Q, R, x0, P0), reps=15, inner=20)
diff = float(jnp.max(jnp.abs(loop_eager() - jrun(zs, F, H, Q, R, x0, P0)[-1])))

# does run_linear_kf compile on every call? Count XLA compilations through JAX's monitoring events.
compile_events = []
jax.monitoring.register_event_duration_secs_listener(
    lambda name, secs, **kw: compile_events.append(name) if name.endswith("backend_compile_duration") else None)


def compiles_during(thunk):
    n = len(compile_events)
    jax.block_until_ready(thunk())
    return len(compile_events) - n


run_compiles = [compiles_during(lambda: run_linear_kf(zs, F, H, Q, R, x0, P0)) for _ in range(3)]
jrun_compiles = [compiles_during(lambda: jrun(zs, F, H, Q, R, x0, P0)) for _ in range(3)]

A = jnp.ones((3000, 3000))
jax.block_until_ready(A @ A)
async_rows = []
for _ in range(5):
    t0 = time.perf_counter()
    C = A @ A
    t1 = time.perf_counter()
    C.block_until_ready()
    t2 = time.perf_counter()
    async_rows.append(((t1 - t0) * 1e3, (t2 - t0) * 1e3))
t_ret = statistics.median(r[0] for r in async_rows)
t_done = statistics.median(r[1] for r in async_rows)

timing = {
    "op_us": round(t_op * 1e6, 1), "np_op_us": round(t_np_op * 1e6, 2), "step_ops": n_ops,
    "eager_step_us": round(t_eager_step * 1e6), "jit_step_us": round(t_jit_step * 1e6, 1),
    "stages_2": st2, "stages_4": st4, "stages_int": sti,
    "loop_eager_ms": round(t_loop_eager * 1e3, 1), "loop_jit_ms": round(t_loop_jit * 1e3, 1),
    "loop_numpy_ms": round(t_loop_np * 1e3, 1), "run_plain_ms": round(t_run_plain * 1e3, 1),
    "jrun_first_ms": round(t_jrun_first * 1e3, 1), "jrun_ms": round(t_jrun * 1e3, 3),
    "run_compiles": run_compiles, "jrun_compiles": jrun_compiles,
    "async_return_ms": round(t_ret, 2), "async_done_ms": round(t_done, 1), "max_diff": diff,
}
for key, value in timing.items():
    out(f"{key}: {value}")
out("stages are [trace ms, lower ms, compile ms, run us]")

if DATA:
    sys.stdout = REAL_STDOUT
    payload = {"jax": jax.__version__, "hero": hero, "cache": cache_rows, "widget": widget_rows, "timing": timing}
    print("/* Kalman filters with JAX, chapter J2 data, written by code/j2_numbers.py with JAX "
          f"{jax.__version__} on a CPU: the jaxprs of the demo's functions, each equation tagged with the "
          "source line that produced it, the cache experiment, and the timings. */")
    print("window.J2 = " + json.dumps(payload, indent=1) + ";")
