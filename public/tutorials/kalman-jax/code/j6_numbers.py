"""Reproduce every key, tree, message, timing and training result quoted on chapter J6 of "Kalman Filters with JAX"
(public/tutorials/kalman-jax/j6-keys-trees-and-training.html).

Usage:  python j6_numbers.py          # print everything the page quotes
        python j6_numbers.py data     # print the page's data file, interactive/j6-data.js
Checked with jax 0.12.0.1, optax 0.2.8, numpy 2.x, CPU. Timings depend on the machine.
The Kalman and network functions are copied verbatim from the repository so that the script runs alone.
"""
from __future__ import annotations

import io
import json
import math
import statistics
import sys
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax, random

DATA = len(sys.argv) > 1 and sys.argv[1] == "data"
REAL_STDOUT = sys.stdout
if DATA:
    sys.stdout = io.StringIO()
out = (lambda *a, **k: None) if DATA else print
out("jax", jax.__version__, "optax", optax.__version__, "numpy", np.__version__,
    "threefry_partitionable", jax.config.jax_threefry_partitionable)


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


u32 = lambda a: [int(v) for v in np.asarray(a, dtype=np.uint32).ravel()]  # noqa: E731

# ---------------------------------------------------------------- 1. keys
section("1. Keys: what PRNGKey(42) holds, split, fold_in, and drawing")
key = random.PRNGKey(42)
k_split2 = random.split(key)
k_split3 = random.split(key, 3)
k_fold7 = random.fold_in(key, 7)
bits4 = random.bits(key, (4,))
unif4 = random.uniform(key, (4,))
norm4 = random.normal(key, (4,))
keys_demo = {
    "key42": u32(key), "split2": u32(k_split2), "split3": u32(k_split3), "fold7": u32(k_fold7),
    "bits4": u32(bits4), "uniform4": [float(v) for v in unif4], "normal4": [float(v) for v in norm4],
    "split_of_split": u32(random.split(k_split2[0])),
    "uniform_child0": [float(v) for v in random.uniform(k_split2[0], (4,))],
    "fold_equals_split": bool(jnp.all(random.fold_in(key, 2) == random.split(key, 5)[2])),
    "same_key_same": bool(jnp.all(random.normal(key, (1000,)) == random.normal(key, (1000,)))),
    "typed_key": str(random.key(42)), "typed_data": u32(random.key_data(random.key(42))),
    "seed_big": u32(random.PRNGKey(2**32 + 5)),
}
for k, v in keys_demo.items():
    out(f"{k}: {v}")

# reusing a key: two 'independent' noise sequences that are the same sequence
a = random.normal(key, (1000,))
b = random.normal(key, (1000,))
k1, k2 = random.split(key)
c, d = random.normal(k1, (1000,)), random.normal(k2, (1000,))
reuse = {"corr_reused": float(jnp.corrcoef(a, b)[0, 1]), "corr_split": float(jnp.corrcoef(c, d)[0, 1])}
out(reuse)

# a key under vmap and jit: pure functions of the key
draws = jax.jit(jax.vmap(lambda k: random.normal(k, ())))(random.split(key, 5))
keys_demo["vmap_draws"] = [float(v) for v in draws]

# ---------------------------------------------------------------- 2. pytrees
section("2. Pytrees: the repository's network parameters")


def init_mlp_params(key, sizes):
    """kalman_jax/learned_dynamics.py, verbatim."""
    num_layers = len(sizes) - 1
    keys = random.split(key, num_layers * 2)
    params = []
    for layer_index, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        W_key = keys[2 * layer_index]
        b_key = keys[2 * layer_index + 1]
        params.append((random.normal(W_key, (n_out, n_in)) * 0.1, random.normal(b_key, (n_out,)) * 0.1))
    return params


def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(W @ x + b)
    W, b = params[-1]
    return W @ x + b


SIZES = [3, 16, 16, 16, 16, 2]
params = init_mlp_params(random.PRNGKey(42), SIZES)
leaves, treedef = jax.tree_util.tree_flatten(params)
shapes = jax.tree_util.tree_map(lambda p: p.shape, params)
grads = jax.grad(lambda p: jnp.sum(mlp(p, jnp.ones(3)) ** 2))(params)
trees = {
    "treedef": str(treedef), "n_leaves": len(leaves), "n_params": int(sum(p.size for p in leaves)),
    "shapes": [[list(W), list(b)] for W, b in shapes],
    "grad_same_structure": jax.tree_util.tree_structure(grads) == treedef,
    "list_vs_tuple": jax.tree_util.tree_structure([1, 2]) == jax.tree_util.tree_structure((1, 2)),
    "none_leaves": len(jax.tree_util.tree_leaves({"a": None, "b": 1.0})),
}
out(trees)


class KFState(NamedTuple):
    x: jax.Array
    P: jax.Array


st = KFState(jnp.zeros(2), jnp.eye(2))
trees["kfstate_leaves"] = len(jax.tree_util.tree_leaves(st))
trees["kfstate_def"] = str(jax.tree_util.tree_structure(st))
doubled = jax.tree_util.tree_map(lambda v: 2 * v, st)
trees["kfstate_map_type"] = type(doubled).__name__
out(trees["kfstate_def"], trees["kfstate_map_type"])

# ---------------------------------------------------------------- 3. optax
section("3. Optax: a gradient transformation and its state")
opt = optax.adam(1e-3)
opt_state = opt.init(params)
upd, opt_state2 = opt.update(grads, opt_state, params)
new_params = optax.apply_updates(params, upd)
optax_demo = {
    "state_repr": str(jax.tree_util.tree_map(lambda v: getattr(v, "shape", v), opt_state))[:400],
    "state_leaves": len(jax.tree_util.tree_leaves(opt_state)),
    "first_update_maxabs": float(max(jnp.max(jnp.abs(u)) for u in jax.tree_util.tree_leaves(upd))),
    "chain_state": str(type(optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)).init(params)).__name__),
}
out(optax_demo)

# ---------------------------------------------------------------- 4. training: J3's noise estimate, by GD and by Adam
section("4. Estimating chapter 02's noise levels: gradient descent against Adam, both from sigma = 1")
DT = 0.1
F = jnp.array([[1.0, DT], [0.0, 1.0]])
H = jnp.array([[1.0, 0.0]])
Q_UNIT = jnp.array([[DT**3 / 3.0, DT**2 / 2.0], [DT**2 / 2.0, DT]])
Q_TRUE = Q_UNIT * 0.1**2


def _kalman_gain(P, H, innovation_cov):
    return jnp.linalg.solve(innovation_cov, H @ P.T).T


def _joseph_covariance_update(P, K, H, R):
    identity = jnp.eye(P.shape[0], dtype=P.dtype)
    residual_factor = identity - K @ H
    return residual_factor @ P @ residual_factor.T + K @ R @ K.T


def linear_kf_predict(x, P, F, Q):
    return F @ x, F @ P @ F.T + Q


def linear_kf_update(x, P, z, H, R):
    innovation = z - H @ x
    innovation_cov = H @ P @ H.T + R
    K = _kalman_gain(P, H, innovation_cov)
    return x + K @ innovation, _joseph_covariance_update(P, K, H, R), innovation, innovation_cov


def generate_scan(key, n_steps=1000):
    def step(carry, _):
        key, x = carry
        key, process_key = random.split(key)
        x = F @ x + random.multivariate_normal(process_key, jnp.zeros(2), Q_TRUE)
        key, measurement_key = random.split(key)
        z = H @ x + random.normal(measurement_key, (1,)) * 0.5
        return (key, x), (x, z)

    _, (xs, zs) = lax.scan(step, (key, jnp.array([0.0, 1.0])), None, length=n_steps)
    return xs, zs


_, zs = jax.jit(generate_scan)(random.PRNGKey(42))


def nll(theta):
    sa, sz = jnp.exp(theta[0]), jnp.exp(theta[1])
    Q, R = Q_UNIT * sa**2, jnp.eye(1) * sz**2

    def step(carry, z):
        x, P = carry
        xp, Pp = linear_kf_predict(x, P, F, Q)
        xu, Pu, v, S = linear_kf_update(xp, Pp, z, H, R)
        return (xu, Pu), 0.5 * (jnp.log(2 * jnp.pi) + jnp.log(S[0, 0]) + v[0] ** 2 / S[0, 0])

    return jnp.mean(lax.scan(step, (jnp.zeros(2), jnp.eye(2)), zs)[1])


MLE = (0.06106381, 0.5127593)   # J3, Newton's method in float64


def train(optimizer, steps):
    """The repository's pattern: a jitted update, called from a Python loop."""
    state = optimizer.init(jnp.zeros(2))

    @jax.jit
    def update(theta, state):
        loss, g = jax.value_and_grad(nll)(theta)
        updates, state = optimizer.update(g, state, theta)
        return optax.apply_updates(theta, updates), state, loss

    theta, path = jnp.zeros(2), []
    for _ in range(steps):
        path.append([float(theta[0]), float(theta[1])])
        theta, state, loss = update(theta, state)
    path.append([float(theta[0]), float(theta[1])])
    return path


def steps_to(path, tol=1e-3):
    for i, (ta, tz) in enumerate(path):
        rest = path[i:]
        if all(abs(math.exp(a) / MLE[0] - 1) < tol and abs(math.exp(z) / MLE[1] - 1) < tol for a, z in rest):
            return i
    return None


N_TRAIN = 1000
runs = {"gd": train(optax.sgd(0.5), N_TRAIN), "adam_005": train(optax.adam(0.05), N_TRAIN),
        "adam_01": train(optax.adam(0.1), N_TRAIN), "momentum": train(optax.sgd(0.5, momentum=0.9), N_TRAIN)}
training = {name: {"steps_to_0.1%": steps_to(p), "end": [math.exp(p[-1][0]), math.exp(p[-1][1])],
                   "path": p[::5]} for name, p in runs.items()}
for name, r in training.items():
    out(name, "steps to within 0.1% of the MLE on both:", r["steps_to_0.1%"], " end:", r["end"])

# ---------------------------------------------------------------- 5. the loop itself: Python against scan
section("5. 1000 Adam steps: a Python loop of jitted updates against one scan")
opt = optax.adam(0.05)


@jax.jit
def update(theta, state):
    loss, g = jax.value_and_grad(nll)(theta)
    updates, state = opt.update(g, state, theta)
    return optax.apply_updates(theta, updates), state, loss


def python_loop():
    theta, state = jnp.zeros(2), opt.init(jnp.zeros(2))
    for _ in range(N_TRAIN):
        theta, state, loss = update(theta, state)
    return theta


@jax.jit
def scanned():
    def body(carry, _):
        theta, state = carry
        theta, state, loss = update(theta, state)
        return (theta, state), loss
    (theta, _), losses = lax.scan(body, (jnp.zeros(2), opt.init(jnp.zeros(2))), None, length=N_TRAIN)
    return theta


t_loop = median_time(python_loop, reps=3)
t_scan = median_time(scanned, reps=3)
same_end = float(jnp.max(jnp.abs(python_loop() - scanned())))
loop_cmp = {"python_ms": round(t_loop * 1e3, 1), "scan_ms": round(t_scan * 1e3, 1), "same_end": same_end}
out(loop_cmp)

# ---------------------------------------------------------------- 6. for the exercises
section("6. For the exercises")
per_step = jax.vmap(lambda k: random.fold_in(key, k))(jnp.arange(5))
ex = {"fold_vs_split": bool(jnp.all(per_step == random.split(key, 5))),
      "clip_norm": float(optax.global_norm(optax.clip_by_global_norm(1.0).update(
          jax.tree_util.tree_map(lambda g: 100 * g, grads), optax.EmptyState())[0])),
      "grad_norm": float(optax.global_norm(grads))}
out(ex)

if DATA:
    sys.stdout = REAL_STDOUT
    payload = {"jax": jax.__version__, "optax": optax.__version__, "keys": keys_demo, "reuse": reuse,
               "trees": trees, "optax_demo": optax_demo, "training": training, "mle": MLE,
               "loop": loop_cmp, "ex": ex, "zs": [round(float(v), 7) for v in np.asarray(zs)[:, 0]]}
    print("/* Kalman filters with JAX, chapter J6 data, written by code/j6_numbers.py with JAX "
          f"{jax.__version__} and Optax {optax.__version__} on a CPU: JAX's keys and draws for checking the "
          "page's own Threefry, the parameter tree, and the training runs on chapter 02's noise levels. */")
    print("window.J6 = " + json.dumps(payload) + ";")
