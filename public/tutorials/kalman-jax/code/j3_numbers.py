"""Reproduce every derivative, jaxpr, message and timing quoted on chapter J3 of "Kalman Filters with JAX"
(public/tutorials/kalman-jax/j3-automatic-differentiation-from-scratch.html).

Usage:  python j3_numbers.py          # print everything the page quotes
        python j3_numbers.py data     # print the page's data file, interactive/j3-data.js
The float64 numbers come from a second run of this script with jax_enable_x64, started automatically,
on the same float32 measurements. Checked with jax 0.12.0.1, numpy 2.x, CPU. Timings depend on the machine.
The Kalman functions are copied verbatim from kalman_jax/filters.py so that the script runs alone.
"""
from __future__ import annotations

import io
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time

X64PART = len(sys.argv) > 1 and sys.argv[1] == "x64part"
X64NEWTON = len(sys.argv) > 1 and sys.argv[1] == "x64newton"
import jax  # noqa: E402

if X64PART or X64NEWTON:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

DATA = len(sys.argv) > 1 and sys.argv[1] == "data"
REAL_STDOUT = sys.stdout
if DATA or X64PART or X64NEWTON:
    sys.stdout = io.StringIO()
out = (lambda *a, **k: None) if (DATA or X64PART or X64NEWTON) else print


def section(title):
    out(f"\n=== {title}")


def error_of(thunk):
    try:
        thunk()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).splitlines()[0]}"
    return "no error"


def median_time(fn, reps=15):
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


# ---------------------------------------------------------------- the filter's negative log-likelihood
DT = 0.1


def make_nll(zs):
    """Mean negative log-likelihood per step of the measurements zs, as a function of
    theta = (log sigma_a, log sigma_z): the process-noise and measurement-noise standard deviations."""
    F = jnp.array([[1.0, DT], [0.0, 1.0]])
    H = jnp.array([[1.0, 0.0]])
    Q_unit = jnp.array([[DT**3 / 3.0, DT**2 / 2.0], [DT**2 / 2.0, DT]])
    x0, P0 = jnp.zeros(2), jnp.eye(2)

    def nll(theta):
        sigma_a, sigma_z = jnp.exp(theta[0]), jnp.exp(theta[1])
        Q, R = Q_unit * sigma_a**2, jnp.eye(1) * sigma_z**2

        def step(carry, z):
            x, P = carry
            x_pred, P_pred = linear_kf_predict(x, P, F, Q)
            x_new, P_new, v, S = linear_kf_update(x_pred, P_pred, z, H, R)
            term = 0.5 * (jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(S)[1] + v @ jnp.linalg.solve(S, v))
            return (x_new, P_new), term

        _, terms = lax.scan(step, (x0, P0), zs)
        return jnp.mean(terms)

    return nll


THETA_TRUE = [math.log(0.1), math.log(0.5)]
TEST_POINTS = [THETA_TRUE, [0.0, 0.0], [math.log(0.3), math.log(0.8)], [math.log(0.03), math.log(0.35)]]
H_EXPS = list(range(1, 13))


def kalman_experiments(zs):
    """The numbers that depend on the arithmetic's precision; run once in float32 and once in float64."""
    nll = make_nll(jnp.asarray(zs))
    vg = jax.jit(jax.value_and_grad(nll))
    nll_j = jax.jit(nll)
    theta = jnp.array(THETA_TRUE)
    g = jax.grad(nll)(theta)
    fd = {"fwd": [], "cen": []}
    for k in H_EXPS:
        e = jnp.array([0.0, 10.0**-k])
        fd["fwd"].append(float((nll_j(theta + e) - nll_j(theta)) / 10.0**-k))
        fd["cen"].append(float((nll_j(theta + e) - nll_j(theta - e)) / (2 * 10.0**-k)))
    tests = []
    for t in TEST_POINTS:
        v, gr = vg(jnp.array(t))
        tests.append({"theta": t, "nll": float(v), "grad": [float(a) for a in gr]})
    path, th = [], jnp.array([0.0, 0.0])
    for i in range(401):
        v, gr = vg(th)
        if i % 10 == 0:
            path.append([float(th[0]), float(th[1]), float(v)])
        th = th - 0.5 * gr
    t_f = median_time(lambda: nll_j(theta))
    t_g = median_time(lambda: vg(theta))
    return {"dtype": str(jnp.asarray(zs).dtype), "nll": float(nll(theta)), "grad": [float(a) for a in g], "fd": fd,
            "tests": tests, "path": path, "t_nll_ms": round(t_f * 1e3, 3), "t_grad_ms": round(t_g * 1e3, 3)}


if X64PART:
    zs64 = np.load(sys.argv[2]).astype(np.float64)
    res = kalman_experiments(zs64)
    sys.stdout = REAL_STDOUT
    print(json.dumps(res))
    sys.exit(0)

if X64NEWTON:
    # Part 7 and exercise 5: Newton's method on the likelihood in float64, its Hessian, and gradient descent's pace
    nll64 = make_nll(jnp.asarray(np.load(sys.argv[2]).astype(np.float64)))
    g64, h64 = jax.jit(jax.grad(nll64)), jax.jit(jax.hessian(nll64))
    theta, gnorms = jnp.zeros(2), []
    for _ in range(12):
        theta = theta - jnp.linalg.solve(h64(theta), g64(theta))
        gnorms.append(float(jnp.max(jnp.abs(g64(theta)))))
    n_iter = next(i + 1 for i, v in enumerate(gnorms) if v < 1e-13)
    Hm = h64(theta)
    eig = jnp.linalg.eigvalsh(Hm)
    se = jnp.sqrt(jnp.diag(jnp.linalg.inv(1000 * Hm)))          # nll is a mean over 1000 steps
    t_gd, k_gd = jnp.zeros(2), 0
    while abs(float(jnp.exp(t_gd[0] - theta[0])) - 1) >= 1e-3 and k_gd < 10000:
        t_gd, k_gd = t_gd - 0.5 * g64(t_gd), k_gd + 1
    res = {"mle": [float(v) for v in jnp.exp(theta)], "iterations": n_iter, "grad_after": gnorms[n_iter - 1],
           "eig": [float(v) for v in eig], "ratio": float(eig[1] / eig[0]), "se_log": [float(v) for v in se],
           "truth_in_se": float((np.log(0.1) - theta[0]) / se[0]), "gd_steps_sigma_a_0.1%": k_gd}
    sys.stdout = REAL_STDOUT
    print(json.dumps(res))
    sys.exit(0)

out("jax", jax.__version__, "numpy", np.__version__)

# ---------------------------------------------------------------- 1. the running example
section("1. f(x1, x2) = log(x1) + x1 * x2 - sin(x2) at (2, 5)")


def f(x1, x2):
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)


out("f =", f(2.0, 5.0), " grad =", jax.grad(f, argnums=(0, 1))(2.0, 5.0))
out("jvp with tangent (1, 0):", jax.jvp(f, (2.0, 5.0), (1.0, 0.0)))
y, f_vjp = jax.vjp(f, 2.0, 5.0)
out("vjp with cotangent 1:", f_vjp(1.0))
section("1b. jaxprs: f, its JVP, its gradient")
jx_f = str(jax.make_jaxpr(f)(2.0, 5.0))
jx_jvp = str(jax.make_jaxpr(lambda a, b, ta, tb: jax.jvp(f, (a, b), (ta, tb)))(2.0, 5.0, 1.0, 0.0))
jx_grad = str(jax.make_jaxpr(jax.grad(f, argnums=(0, 1)))(2.0, 5.0))
out(jx_f, jx_jvp, jx_grad, sep="\n")

# ---------------------------------------------------------------- 2. from scratch
section("2. Automatic differentiation from scratch, in plain Python")


class Dual:
    """A number with a tangent: forward mode."""

    def __init__(self, value, tangent=0.0):
        self.value, self.tangent = value, tangent

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.value + other.value, self.tangent + other.tangent)

    def __sub__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.value - other.value, self.tangent - other.tangent)

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.value * other.value, self.tangent * other.value + self.value * other.tangent)


def d_log(x):
    return Dual(math.log(x.value), x.tangent / x.value)


def d_sin(x):
    return Dual(math.sin(x.value), math.cos(x.value) * x.tangent)


def f_dual(x1, x2):
    return d_log(x1) + x1 * x2 - d_sin(x2)


r1 = f_dual(Dual(2.0, 1.0), Dual(5.0, 0.0))
r2 = f_dual(Dual(2.0, 0.0), Dual(5.0, 1.0))
out("forward mode, seed x1:", r1.value, r1.tangent, "  seed x2:", r2.tangent)


class Var:
    """A number that remembers how it was made: reverse mode."""

    def __init__(self, value, parents=()):
        self.value, self.parents, self.adjoint = value, parents, 0.0

    def __add__(self, other):
        return Var(self.value + other.value, ((self, 1.0), (other, 1.0)))

    def __sub__(self, other):
        return Var(self.value - other.value, ((self, 1.0), (other, -1.0)))

    def __mul__(self, other):
        return Var(self.value * other.value, ((self, other.value), (other, self.value)))


def v_log(x):
    return Var(math.log(x.value), ((x, 1.0 / x.value),))


def v_sin(x):
    return Var(math.sin(x.value), ((x, math.cos(x.value)),))


def backward(y):
    order, seen = [], set()

    def visit(v):  # topological order: every node after the nodes it was made from
        if id(v) not in seen:
            seen.add(id(v))
            for p, _ in v.parents:
                visit(p)
            order.append(v)

    visit(y)
    y.adjoint = 1.0
    for v in reversed(order):
        for p, local in v.parents:
            p.adjoint += local * v.adjoint


a, b = Var(2.0), Var(5.0)
yv = v_log(a) + a * b - v_sin(b)
backward(yv)
out("reverse mode:", yv.value, a.adjoint, b.adjoint)
scratch = {"f": yv.value, "dual": [r1.tangent, r2.tangent], "tape": [a.adjoint, b.adjoint]}

# ---------------------------------------------------------------- 3. where autodiff surprises
section("3. Edges")
edges = {
    "abs0": float(jax.grad(jnp.abs)(0.0)), "relu0": float(jax.grad(jax.nn.relu)(0.0)),
    "sqrt0": float(jax.grad(jnp.sqrt)(0.0)),
    "where_m1": float(jax.grad(lambda x: jnp.where(x > 0, jnp.sqrt(x), 0.0))(-1.0)),
    "where_4": float(jax.grad(lambda x: jnp.where(x > 0, jnp.sqrt(x), 0.0))(4.0)),
    "safe_m1": float(jax.grad(lambda x: jnp.where(x > 0, jnp.sqrt(jnp.where(x > 0, x, 1.0)), 0.0))(-1.0)),
    "norm0": [float(v) for v in jax.grad(jnp.linalg.norm)(jnp.zeros(2))],
    "stopgrad": float(jax.grad(lambda x: x * lax.stop_gradient(x))(3.0)),
    "err_int": error_of(lambda: jax.grad(lambda x: x * 2)(3)),
    "err_vec": error_of(lambda: jax.grad(lambda x: x * 2)(jnp.ones(3))),
    "err_while": error_of(lambda: jax.grad(lambda x: lax.while_loop(lambda c: c < 10, lambda c: c * 2, x))(1.0)),
    "fwd_while": [float(v) for v in jax.jvp(lambda x: lax.while_loop(lambda c: c < 10, lambda c: c * 2, x), (1.0,), (1.0,))],
}
for k, v in edges.items():
    out(f"{k}: {v}")

# ---------------------------------------------------------------- 4. forward or reverse: building a Jacobian
section("4. jacfwd against jacrev for f(x) = tanh(W x), W of shape (m, n)")
modes = []
for n, m in [(1000, 4), (4, 1000), (300, 300)]:
    W = jax.random.normal(jax.random.PRNGKey(0), (m, n)) / math.sqrt(n)
    x = jax.random.normal(jax.random.PRNGKey(1), (n,))
    g_fn = lambda x, W=W: jnp.tanh(W @ x)  # noqa: E731
    jf, jr = jax.jit(jax.jacfwd(g_fn)), jax.jit(jax.jacrev(g_fn))
    same = float(jnp.max(jnp.abs(jf(x) - jr(x))))
    row = {"n": n, "m": m, "fwd_us": round(median_time(lambda: jf(x)) * 1e6, 1),
           "rev_us": round(median_time(lambda: jr(x)) * 1e6, 1), "max_diff": same}
    modes.append(row)
    out(row)

# ---------------------------------------------------------------- 5. Kalman
section("5a. A range sensor: jacfwd against the hand-derived Jacobian")
D_SENSOR = 10.0


def h_range(x):
    return jnp.array([jnp.sqrt(x[0] ** 2 + D_SENSOR**2)])


xr = jnp.array([24.0, 1.0])
H_auto = jax.jacfwd(h_range)(xr)
H_hand = jnp.array([[xr[0] / jnp.sqrt(xr[0] ** 2 + D_SENSOR**2), 0.0]])
out("jacfwd:", H_auto, " hand:", H_hand, " jaxpr of the Jacobian:")
jx_H = str(jax.make_jaxpr(jax.jacfwd(h_range))(xr))
out(jx_H)
ekf_tie = {"x": [24.0, 1.0], "H_auto": [float(v) for v in H_auto.ravel()], "H_hand": [float(v) for v in H_hand.ravel()]}

section("5b. The filter's likelihood and its gradient, float32 and float64, on the same measurements")
F, H, Q, R, true_states, zs = generate_linear_synthetic_data(jax.random.PRNGKey(42), n_steps=1000)
zs32 = np.asarray(zs, dtype=np.float32)
k32 = kalman_experiments(zs32)
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "zs.npy")
    np.save(path, zs32)
    proc = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "x64part", path],
                          capture_output=True, text=True, check=True)
k64 = json.loads(proc.stdout.strip().splitlines()[-1])
for name, k in (("float32", k32), ("float64", k64)):
    out(f"{name}: nll {k['nll']:.10g}  grad {k['grad']}  time nll {k['t_nll_ms']} ms, value_and_grad {k['t_grad_ms']} ms")
out("finite differences of d nll / d log sigma_z; reference (float64 autodiff):", k64["grad"][1])
out(" h      f32 forward     f32 central     f64 forward     f64 central")
for i, kexp in enumerate(H_EXPS):
    out(f"1e-{kexp:<3d} {k32['fd']['fwd'][i]: .8e} {k32['fd']['cen'][i]: .8e} {k64['fd']['fwd'][i]: .8e} {k64['fd']['cen'][i]: .8e}")
out("gradient descent from sigma_a = sigma_z = 1, step 0.5, float64: every 100 steps")
for p in k64["path"][::10]:
    out(f"  sigma_a {math.exp(p[0]):.4f} sigma_z {math.exp(p[1]):.4f} nll {p[2]:.6f}")
out("float32 path ends at sigma_a %.4f sigma_z %.4f" % (math.exp(k32["path"][-1][0]), math.exp(k32["path"][-1][1])))

section("5c. J2's missing-measurement step: the gradient with respect to the measurement noise, z = NaN")
Fm, Hm = jnp.array([[1.0, DT], [0.0, 1.0]]), jnp.array([[1.0, 0.0]])


def step_missing(log_sigma_z, z, guard):
    R_ = jnp.eye(1) * jnp.exp(2 * log_sigma_z)
    x_pred, P_pred = linear_kf_predict(jnp.array([1.0, 1.0]), jnp.eye(2), Fm, Q)
    z_used = jnp.nan_to_num(z) if guard else z
    x_upd, _, _, _ = linear_kf_update(x_pred, P_pred, z_used, Hm, R_)
    return jnp.sum(jnp.where(~jnp.any(jnp.isnan(z)), x_upd, x_pred))


znan, ls = jnp.array([jnp.nan]), jnp.log(0.5)
missing = {"value_guarded": float(step_missing(ls, znan, True)), "value_unguarded": float(step_missing(ls, znan, False)),
           "grad_guarded": float(jax.grad(step_missing)(ls, znan, True)),
           "grad_unguarded": float(jax.grad(step_missing)(ls, znan, False))}
out(missing)

if not DATA:
    section("6. Newton's method and the Hessian on chapter 02's likelihood (float64), and exercise 1")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "zs.npy")
        np.save(path, zs32)
        proc = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "x64newton", path],
                              capture_output=True, text=True, check=True)
    newton = json.loads(proc.stdout.strip().splitlines()[-1])
    for k, v in newton.items():
        out(f"{k}: {v}")
    swish = lambda x: x / (1 + jnp.exp(-x))  # noqa: E731
    out("exercise 1, JAX float32: s(1.5) =", swish(1.5), " s'(1.5) =", jax.grad(swish)(1.5))
    Dual.__truediv__ = lambda a, b: Dual(a.value / b.value, (a.tangent * b.value - a.value * b.tangent) / b.value ** 2)
    xd = Dual(1.5, 1.0)
    ed = Dual(math.exp(-xd.value), -math.exp(-xd.value))       # exp(-x) and its tangent
    sd = xd / (Dual(1.0) + ed)
    out("exercise 1, dual numbers in float64: s(1.5) =", sd.value, " s'(1.5) =", sd.tangent)

if DATA:
    sys.stdout = REAL_STDOUT
    payload = {"jax": jax.__version__, "zs": [round(float(v), 9) for v in zs32.ravel()],
               "theta_true": THETA_TRUE, "h_exps": H_EXPS, "k32": k32, "k64": k64, "scratch": scratch,
               "edges": edges, "modes": modes, "ekf": ekf_tie, "missing": missing,
               "jaxpr": {"f": jx_f, "jvp": jx_jvp, "grad": jx_grad}}
    print("/* Kalman filters with JAX, chapter J3 data, written by code/j3_numbers.py with JAX "
          f"{jax.__version__} on a CPU: chapter 02's 1000 measurements (float32), the filter's likelihood and "
          "its gradient in float32 and float64, finite differences, a gradient-descent path, and the other "
          "numbers the page quotes. */")
    print("window.J3 = " + json.dumps(payload) + ";")
