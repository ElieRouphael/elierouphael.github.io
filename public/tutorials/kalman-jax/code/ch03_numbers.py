"""Reproduce every number quoted on chapter 03 of "Kalman Filters with JAX"
(public/tutorials/kalman-jax/03-learning-what-the-model-leaves-out.html).

Usage:  python ch03_numbers.py          # print everything the page quotes
        python ch03_numbers.py data     # print the page's data file, interactive/ch03-data.js
Checked with jax 0.12.0.1, optax 0.2.8 and numpy 2.x on a laptop CPU; it trains about thirty small models and takes
a few minutes. JAX runs in its default float32; the float64 numbers (the page computes in double precision) come
from a second run of this script with jax_enable_x64, started automatically. The functions of the repository's
kalman_jax/learned_dynamics.py and forecasters.py are copied verbatim (docstrings shortened) so that the script runs
alone.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile

X64PART = len(sys.argv) > 1 and sys.argv[1] == "x64part"
if X64PART:
    import jax
    jax.config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random, value_and_grad

DATA = len(sys.argv) > 1 and sys.argv[1] == "data"
REAL_STDOUT = sys.stdout
if DATA:
    sys.stdout = io.StringIO()
out = (lambda *a, **k: None) if DATA else print


def section(title):
    out(f"\n=== {title}")


# ---------------------------------------------------------------------------------------------------------------
# kalman_jax/filters.py
def _kalman_gain(P, H, innovation_cov):
    return jnp.linalg.solve(innovation_cov, H @ P.T).T


def _joseph_covariance_update(P, K, H, R):
    identity = jnp.eye(P.shape[0], dtype=P.dtype)
    residual_factor = identity - K @ H
    return residual_factor @ P @ residual_factor.T + K @ R @ K.T


# kalman_jax/learned_dynamics.py
def true_dynamics(x, u, A_true=None, B_true=None):
    A_true = jnp.array([[1.0, 0.1], [0.0, 1.0]]) if A_true is None else A_true
    B_true = jnp.array([[0.0], [0.1]]) if B_true is None else B_true
    nonlinear_residual = 3.0 * jnp.sin(x[0]) + 2.0 * jnp.cos(x[1])
    return A_true @ x + B_true @ u + nonlinear_residual


def generate_data(T=100, key=None, A_true=None, B_true=None, measurement_noise_std=0.05):
    key = random.PRNGKey(0) if key is None else key
    x = jnp.zeros((2,))
    xs, us, ys = [], [], []

    for _ in range(T):
        key, control_key, measurement_key = random.split(key, 3)
        u = random.normal(control_key, (1,))
        x = true_dynamics(x, u, A_true=A_true, B_true=B_true)
        y = x + measurement_noise_std * random.normal(measurement_key, x.shape)
        xs.append(x)
        us.append(u)
        ys.append(y)

    return jnp.stack(xs), jnp.stack(us), jnp.stack(ys)


def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(W @ x + b)
    W, b = params[-1]
    return W @ x + b


def init_mlp_params(key, sizes):
    num_layers = len(sizes) - 1
    keys = random.split(key, num_layers * 2)
    params = []
    for layer_index, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        W_key = keys[2 * layer_index]
        b_key = keys[2 * layer_index + 1]
        params.append(
            (
                random.normal(W_key, (n_out, n_in)) * 0.1,
                random.normal(b_key, (n_out,)) * 0.1,
            )
        )
    return params


def kalman_filter_step(x_prev, u, y, params, A, B, C, Q, R, P_prev):
    learned_residual = mlp(params, jnp.concatenate([x_prev, u]))
    x_pred = A @ x_prev + B @ u + learned_residual
    P_pred = A @ P_prev @ A.T + Q

    innovation_cov = C @ P_pred @ C.T + R
    K = _kalman_gain(P_pred, C, innovation_cov)
    x_new = x_pred + K @ (y - C @ x_pred)
    P_new = _joseph_covariance_update(P_pred, K, C, R)

    return x_new, P_new


def loss_fn(params, A, B, C, Q, R, x0, P0, us, ys):
    def step(carry, inputs):
        x, P = carry
        u, y = inputs
        x_new, P_new = kalman_filter_step(x, u, y, params, A, B, C, Q, R, P)
        return (x_new, P_new), C @ x_new

    (_, _), y_preds = jax.lax.scan(step, (x0, P0), (us, ys))
    return jnp.mean((ys - y_preds) ** 2)


def predict_with_kalman(us, ys, params, A, B, C, Q, R, x0, P0):
    def step(carry, inputs):
        x, P = carry
        u, y = inputs
        x_new, P_new = kalman_filter_step(x, u, y, params, A, B, C, Q, R, P)
        return (x_new, P_new), (C @ x_new, P_new)

    (_, _), (y_preds, Ps) = jax.lax.scan(step, (x0, P0), (us, ys))
    return y_preds, Ps


def train_learned_kalman(params, optimizer, *, A, B, C, Q, R, x0, P0, us, ys, num_steps=1000, log_every=None):
    opt_state = optimizer.init(params)

    @jax.jit
    def update(current_params, current_opt_state):
        loss, grads = value_and_grad(loss_fn)(current_params, A, B, C, Q, R, x0, P0, us, ys)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss

    losses = []
    for step in range(num_steps):
        params, opt_state, loss = update(params, opt_state)
        losses.append(loss)
        if log_every is not None and step % log_every == 0:
            print(f"Step {step}, Loss: {float(loss):.6f}")

    return params, opt_state, jnp.stack(losses)


def compute_dynamics_error(xs, us, params, A, B, A_true=None, B_true=None):
    real_dynamics = jax.vmap(lambda x, u: true_dynamics(x, u, A_true=A_true, B_true=B_true))(xs, us)
    learned_dynamics = jax.vmap(
        lambda x, u: A @ x + B @ u + mlp(params, jnp.concatenate([x, u]))
    )(xs, us)
    error = real_dynamics - learned_dynamics
    return error, real_dynamics, learned_dynamics


# kalman_jax/forecasters.py
def transition_predict(x_prev, u, params, A, B):
    learned_residual = mlp(params, jnp.concatenate([x_prev, u]))
    return A @ x_prev + B @ u + learned_residual


def measurement_update(x_pred, P_pred, y, C, R):
    innovation = y - C @ x_pred
    innovation_cov = C @ P_pred @ C.T + R
    K = _kalman_gain(P_pred, C, innovation_cov)

    x_new = x_pred + K @ innovation
    P_new = _joseph_covariance_update(P_pred, K, C, R)

    return x_new, P_new, innovation_cov


def forecasting_loss_fn(params, A, B, C, Q, R, x0, P0, us, ys):
    def step(carry, inputs):
        x, P = carry
        u, y = inputs

        x_pred = transition_predict(x, u, params, A, B)
        P_pred = A @ P @ A.T + Q
        y_pred = C @ x_pred

        x_new, P_new, _ = measurement_update(x_pred, P_pred, y, C, R)
        return (x_new, P_new), y_pred

    (_, _), y_preds = jax.lax.scan(step, (x0, P0), (us, ys))
    return jnp.mean((ys - y_preds) ** 2)


def forecast_with_kalman(params, A, B, C, Q, R, x0, P0, us, ys):
    def step(carry, inputs):
        x, P = carry
        u, y = inputs

        x_pred = transition_predict(x, u, params, A, B)
        P_pred = A @ P @ A.T + Q
        y_pred = C @ x_pred
        S_pred = C @ P_pred @ C.T + R

        x_new, P_new, _ = measurement_update(x_pred, P_pred, y, C, R)
        y_filtered = C @ x_new
        S_filtered = C @ P_new @ C.T + R

        outputs = (y_pred, y_filtered, S_pred, S_filtered)
        return (x_new, P_new), outputs

    (_, _), outputs = jax.lax.scan(step, (x0, P0), (us, ys))
    return outputs


def train_kalman_forecaster(params, optimizer, *, A, B, C, Q, R, x0, P0, us, ys, num_steps=1000, log_every=None):
    opt_state = optimizer.init(params)

    @jax.jit
    def update(current_params, current_opt_state):
        loss, grads = value_and_grad(forecasting_loss_fn)(current_params, A, B, C, Q, R, x0, P0, us, ys)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss

    losses = []
    for step in range(num_steps):
        params, opt_state, loss = update(params, opt_state)
        losses.append(loss)
        if log_every is not None and step % log_every == 0:
            print(f"Step {step}, Forecast Loss: {float(loss):.6f}")

    return params, opt_state, jnp.stack(losses)
# ---------------------------------------------------------------------------------------------------------------

A_ORIG = jnp.array([[1.0, 0.1], [0.0, 1.0]])
A_STAB = jnp.array([[0.9, 0.1], [0.0, 0.9]])
B = jnp.array([[0.0], [0.1]])
C = jnp.eye(2)
x0, P0 = jnp.zeros(2), jnp.eye(2)
NOISE = {"repo": (0.1, 0.5), "honest": (0.01, 0.0025)}
SIZES = [3, 16, 16, 16, 16, 2]

if X64PART:
    # The page computes in double precision. Here: JAX's own float64 run of the same algorithm, from the same
    # (float32) data and initial parameters, as the reference for the page's claims.
    d = np.load(sys.argv[2], allow_pickle=True)
    us, ys = jnp.asarray(d["us"], jnp.float64), jnp.asarray(d["ys"], jnp.float64)
    p0 = [(jnp.asarray(W, jnp.float64), jnp.asarray(b, jnp.float64)) for W, b in d["params0"]]
    Q, R = 0.1 * jnp.eye(2), 0.5 * jnp.eye(2)
    L = lambda p: loss_fn(p, A_ORIG, B, C, Q, R, x0, P0, us, ys)
    g = jax.grad(L)(p0)
    errs = []
    for layer in range(5):
        for (i, j) in ((0, 0), (1, 1), (0, 2)):   # valid in every layer (the last is 2 x 16)
            h = 1e-6
            W = p0[layer][0]
            pp = [(W.at[i, j].add(h) if l == layer else w_, b_) for l, (w_, b_) in enumerate(p0)]
            pm = [(W.at[i, j].add(-h) if l == layer else w_, b_) for l, (w_, b_) in enumerate(p0)]
            fd = (float(L(pp)) - float(L(pm))) / (2 * h)
            errs.append(abs(fd - float(g[layer][0][i, j])) / abs(float(g[layer][0][i, j])))
    print(f"x64: central differences (h = 1e-6) against jax.grad at the initial parameters, 15 weights: "
          f"median relative error {np.median(errs):.1e}, largest {max(errs):.1e}")
    params, _, losses64 = train_learned_kalman(p0, optax.adam(1e-3), A=A_ORIG, B=B, C=C, Q=Q, R=R, x0=x0, P0=P0,
                                               us=us, ys=ys, num_steps=1000)
    l32 = d["losses32"]
    rel = np.abs(np.asarray(losses64[:20]) - l32[:20]) / l32[:20]
    print(f"x64: first loss {float(losses64[0]):.6f}; the first 20 Adam steps match JAX's float32 trace to a relative "
          f"{rel.max():.1e}; after 1000 steps the float64 run ends at {float(losses64[-1]):.4f} "
          f"(float32: {float(l32[-1]):.4f}); the traces first differ by 1% at step "
          f"{int(np.argmax(np.abs(np.asarray(losses64) - l32) / l32 > 0.01))}")
    sys.exit()

out("jax", jax.__version__, "optax", optax.__version__, "numpy", np.__version__)


def scores(params, A, Q, R, xs, us, ys):
    """One rollout: filtered RMSE against the true state, one-step forecast RMSE against the measurements, the
    linear baseline A y_{t-1} + B u_t against the measurements, and the residual network's error on the visited
    states (as the page's evaluate())."""
    y_pred, y_f, *_ = forecast_with_kalman(params, A, B, C, Q, R, x0, P0, us, ys)
    filt = jnp.sqrt(jnp.mean((y_f - xs) ** 2))
    fc = jnp.sqrt(jnp.mean((y_pred - ys) ** 2))
    base = (ys[:-1] @ A.T) + us[1:] @ B.T
    lin = jnp.sqrt(jnp.mean((base - ys[1:]) ** 2))
    learned = jax.vmap(lambda x, u: mlp(params, jnp.concatenate([x, u])))(xs, us)
    truth = 3 * jnp.sin(xs[:, 0]) + 2 * jnp.cos(xs[:, 1])
    dyn = jnp.sqrt(jnp.mean((learned - truth[:, None]) ** 2))
    return filt, fc, lin, dyn


KEYS3 = (7, 11, 13)
MANY_KEYS = range(100, 1100)                                     # 1000 fresh rollouts, none of the keys above
_rollouts = {}


def many_rollouts(A):
    """generate_data(T=50, key=PRNGKey(k), A_true=A) for k = 100..1099, stacked; about 30 s per plant."""
    name = "original" if float(A[0, 0]) == 1.0 else "stable"
    if name not in _rollouts:
        rs = [generate_data(T=50, key=random.PRNGKey(k), A_true=A) for k in MANY_KEYS]
        _rollouts[name] = tuple(jnp.stack([r[i] for r in rs]) for i in range(3))
    return _rollouts[name]


def score3(params, A, Q, R):
    rows = [scores(params, A, Q, R, *generate_data(T=50, key=random.PRNGKey(k), A_true=A)) for k in KEYS3]
    return np.array([[float(v) for v in r] for r in rows])          # rows: key; columns: filtered, forecast, linear, dyn


def score_many(params, A, Q, R):
    f = jax.jit(jax.vmap(lambda xs_, us_, ys_: scores(params, A, Q, R, xs_, us_, ys_)))
    return np.array([np.asarray(v) for v in f(*many_rollouts(A))]).T     # (1000, 4)


def train(plant, T, noise, steps, kind):
    A = A_ORIG if plant == "original" else A_STAB
    q, r = NOISE[noise]
    Q, R = q * jnp.eye(2), r * jnp.eye(2)
    xs, us, ys = generate_data(T=T, A_true=A)                       # the script's rollout, key 0
    p0 = init_mlp_params(random.PRNGKey(42), SIZES)
    trainer = train_learned_kalman if kind == "filtered" else train_kalman_forecaster
    params, _, losses = trainer(p0, optax.adam(1e-3), A=A, B=B, C=C, Q=Q, R=R, x0=x0, P0=P0, us=us, ys=ys, num_steps=steps)
    return dict(params=params, losses=np.asarray(losses), A=A, Q=Q, R=R, data=(xs, us, ys))


# ---- the repository's experiment and the forecast-loss twin (the hero) ----------------------------------------
xs, us, ys = generate_data(T=50)
p0 = init_mlp_params(random.PRNGKey(42), SIZES)
Q, R = 0.1 * jnp.eye(2), 0.5 * jnp.eye(2)
M = {}
for steps in (1000, 5000):
    for T in (50, 500):
        for kind in ("filtered", "forecast"):
            M[("original", T, "repo", steps, kind)] = train("original", T, "repo", steps, kind)
for plant in ("original", "stable"):
    for T in (50, 500):
        for noise in ("repo", "honest"):
            for kind in ("filtered", "forecast"):
                M[(plant, T, noise, 2000, kind)] = train(plant, T, noise, 2000, kind)
for kind in ("filtered", "forecast"):
    M[("stable", 50, "repo", 1000, kind)] = train("stable", 50, "repo", 1000, kind)
rep, repF = M[("original", 50, "repo", 1000, "filtered")], M[("original", 50, "repo", 1000, "forecast")]
params, losses = rep["params"], rep["losses"]
xs7, us7, ys7 = generate_data(T=50, key=random.PRNGKey(7))

if DATA:
    g = lambda v, p=7: "0" if float(format(float(v), f".{p}g")) == 0 else format(float(v), f".{p}g")
    vec = lambda a, p=7: "[" + ",".join(g(v, p) for v in np.asarray(a).ravel()) + "]"
    mat = lambda a, p=7: "[" + ",".join(vec(r, p) for r in np.asarray(a)) + "]"
    prm = lambda P: "[" + ",".join(f"[{mat(W)},{vec(b)}]" for W, b in P) + "]"
    roll = lambda r: f"{{xs:{mat(r[0])},us:{vec(r[1])},ys:{mat(r[2])}}}"
    y_preds, Ps = predict_with_kalman(us, ys, params, A_ORIG, B, C, Q, R, x0, P0)
    stds = np.sqrt(np.stack([np.diag(P) for P in np.asarray(Ps)]))
    err, _, _ = compute_dynamics_error(xs, us, params, A_ORIG, B, A_true=A_ORIG, B_true=B)
    study = {}
    for T in (50, 500):
        for steps in (1000, 5000):
            m = {kind: M[("original", T, "repo", steps, kind)] for kind in ("filtered", "forecast")}
            s = {kind: score3(m[kind]["params"], A_ORIG, Q, R).mean(0) for kind in m}
            study[f"{T}-{steps}"] = {"filtered": [float(s["filtered"][i]) for i in (0, 1, 3)],
                                     "forecast": [float(s["forecast"][i]) for i in (0, 1, 3)],
                                     "loss_f": float(m["filtered"]["losses"][-1]), "loss_p": float(m["forecast"]["losses"][-1])}
    noise = {}
    for plant in ("original", "stable"):
        for T in (50, 500):
            for nz in ("repo", "honest"):
                for kind in ("filtered", "forecast"):
                    m = M[(plant, T, nz, 2000, kind)]
                    s = score3(m["params"], m["A"], m["Q"], m["R"]).mean(0)
                    q, r = NOISE[nz]
                    noise[f"{plant}-{T}-{q:g}-{r:g}-{kind}"] = {"forecast": float(g(s[1], 5)), "zero": float(g(s[2], 5)),
                                                                "dyn": float(g(s[3], 5)), "loss": float(g(m["losses"][-1], 4))}
    stab = {}
    for kind in ("filtered", "forecast"):
        m = M[("stable", 50, "repo", 1000, kind)]
        s3 = score3(m["params"], A_STAB, Q, R)
        stab[kind] = (m["params"], {"filtered": float(s3[:, 0].mean()), "forecast": float(s3[:, 1].mean()),
                                    "dyn": float(s3[:, 3].mean()), "zero": float(s3[:, 2].mean())},
                      [{"key": k, "filtered": float(r[0]), "forecast": float(r[1]), "dyn": float(r[3]), "zero": float(r[2])}
                       for k, r in zip(KEYS3, s3)])
    honest = M[("stable", 500, "honest", 2000, "filtered")]
    many = []
    for label, key in (("repository: double integrator, Q = 0.1 I, R = 0.5 I, T = 50, 1000 steps", ("original", 50, "repo", 1000, "filtered")),
                       ("double integrator, honest noise, T = 50, 2000 steps", ("original", 50, "honest", 2000, "filtered")),
                       ("bounded plant, repository's noise, T = 50, 2000 steps", ("stable", 50, "repo", 2000, "filtered")),
                       ("bounded plant, honest noise, T = 500, 2000 steps", ("stable", 500, "honest", 2000, "filtered"))):
        m = M[key]
        sm = score_many(m["params"], m["A"], m["Q"], m["R"])
        many.append({"label": label, "fc": [float(g(np.mean(sm[:, 1]), 6)), float(g(np.median(sm[:, 1]), 6))],
                     "lin": [float(g(np.mean(sm[:, 2]), 6)), float(g(np.median(sm[:, 2]), 6))],
                     "wins": int(np.sum(sm[:, 1] < sm[:, 2]))})
    tests = ",".join(f'"{k}":{roll(generate_data(T=50, key=random.PRNGKey(k), A_true=A_STAB))}' for k in KEYS3)
    pyblock = f"{{yPreds:{mat(y_preds, 6)},stds:{mat(stds, 5)},error:{mat(err, 5)}}}"
    stab_txt = ", ".join(f"{kind}:{{params:{prm(stab[kind][0])}, mean:{json.dumps(stab[kind][1])}, rows:{json.dumps(stab[kind][2])}}}"
                         for kind in ("filtered", "forecast"))
    REAL_STDOUT.write(
        "/* Kalman filters with JAX, chapter 03 data: the 50-step nonlinear rollout (states, inputs, noisy\n"
        "   measurements) from the repository's JAX stream, the initial and trained residual-network parameters,\n"
        "   JAX's training loss trace, three fresh rollouts for generalisation, the same experiment on a plant\n"
        "   whose state stays bounded, a forecast-loss model, and the main models scored on 1000 fresh rollouts.\n"
        "   Printed by code/ch03_numbers.py data. Filtering and training run live on the page. */\n"
        "window.KJ2 = {\n"
        f"  train: {roll((xs, us, ys))},\n"
        f"  params0: {prm(p0)},\n"
        f"  params: {prm(params)},\n"
        f"  losses: {vec(losses, 6)},\n"
        f"  py: {pyblock},\n"
        f"  fresh: {roll((xs7, us7, ys7))},\n"
        f"  forecastParams: {prm(repF['params'])},\n"
        f"  forecastLosses: {vec(repF['losses'], 6)},\n"
        f"  study: {json.dumps(study)},\n"
        f"  stable: {{A:[[0.9,0.1],[0,0.9]], train:{roll(generate_data(T=50, A_true=A_STAB))}, tests:{{{tests}}}, {stab_txt}, "
        f"honest:{{params:{prm(honest['params'])}, trainXs:{mat(honest['data'][0])}}}}},\n"
        f"  many: {json.dumps(many)},\n"
        "};\n"
        f"window.KJ2.noise = {json.dumps(noise, separators=(',', ':'))};\n")
    sys.exit()

section("1. The plant and the filter (Part 1)")
out(f"RMS size of the residual sqrt(9/2 + 4/2) = {np.sqrt(9 / 2 + 4 / 2):.2f}; the filter assumes sds {np.sqrt(0.1):.2f} (process) and {np.sqrt(0.5):.2f} (sensor)")


def gains(A, q, r, T):
    P = np.eye(2); Ks = []
    for _ in range(T):
        Pp = np.asarray(A) @ P @ np.asarray(A).T + q * np.eye(2); K = Pp @ np.linalg.inv(Pp + r * np.eye(2))
        I_K = np.eye(2) - K; P = I_K @ Pp @ I_K.T + r * K @ K.T; Ks.append(K)
    return np.array(Ks)


Ks = gains(A_ORIG, 0.1, 0.5, 500)
out(f"repository's noise: K_50 = {np.round(Ks[49], 4).tolist()}, so I - K_50 has diagonal {1 - Ks[49][0, 0]:.3f}, {1 - Ks[49][1, 1]:.3f}; "
    f"K_500 diagonal {Ks[-1][0, 0]:.3f}")
Kh, Khs = gains(A_ORIG, 0.01, 0.0025, 500), gains(A_STAB, 0.01, 0.0025, 500)
out(f"honest noise Q = 0.01 I, R = 0.05^2 I: K_inf diagonal {Kh[-1][0, 0]:.3f} (double integrator), {Khs[-1][0, 0]:.3f} (bounded plant); "
    f"1 - K = {1 - Kh[-1][0, 0]:.2f}; with R = 0.05^2 I and Q = 0.1 I: {gains(A_ORIG, 0.1, 0.0025, 500)[-1][0, 0]:.3f}")
lf, lp = (lambda p: float(loss_fn(p, A_ORIG, B, C, Q, R, x0, P0, us, ys))), (lambda p: float(forecasting_loss_fn(p, A_ORIG, B, C, Q, R, x0, P0, us, ys)))
out(f"final losses: repository's model {float(losses[-1]):.4f} (filtered), forecast-loss model {float(repF['losses'][-1]):.4f} (forecast), "
    f"factor {float(repF['losses'][-1]) / float(losses[-1]):.1f}")
out(f"both models on both losses, at their final parameters: filtered-loss model L_filt {lf(params):.4f}, L_pred {lp(params):.4f}; "
    f"forecast-loss model L_filt {lf(repF['params']):.4f}, L_pred {lp(repF['params']):.4f}")
ratio_w = np.mean([(1 - Ks[t][0, 0]) ** 2 for t in range(50)])
out(f"(I - K_t)^2: {(1 - Ks[0][0, 0]) ** 2:.2f} at t = 1, {(1 - Ks[49][0, 0]) ** 2:.2f} at t = 50, mean over the run {ratio_w:.2f}; "
    f"L_filt / L_pred {lf(params) / lp(params):.2f} and {lf(repF['params']) / lp(repF['params']):.2f}")
for name_, pr_ in (("filtered-loss", params), ("forecast-loss", repF["params"])):
    yp_ = np.asarray(forecast_with_kalman(pr_, A_ORIG, B, C, Q, R, x0, P0, us, ys)[0])
    e2 = np.sum((yp_ - np.asarray(ys)) ** 2, axis=1)
    out(f"  {name_} model: (I - K_t)^2 weighted by the squared forecast errors {np.sum(e2 * (1 - Ks[:50, 0, 0]) ** 2) / e2.sum():.2f}; "
        f"share of the squared forecast error in the first 5 steps {e2[:5].sum() / e2.sum():.0%}")
out(f"the forecast-loss model after 5000 steps: {float(M[('original', 50, 'repo', 5000, 'forecast')]['losses'][-1]):.4f}")

section("2. Training (Part 2, the callout and question 4)")
out(f"first loss {float(losses[0]):.6f}; final {float(losses[-1]):.4f}; falls by a factor of {float(losses[0]) / float(losses[-1]):.0f}")
# The page's forward pass is a float64 transcription of the filter. Here the same pass in NumPy float64.
Pn = []; Pc = np.eye(2); Ksn = []
for t in range(50):
    Pp = np.asarray(A_ORIG, np.float64) @ Pc @ np.asarray(A_ORIG, np.float64).T + 0.1 * np.eye(2)
    K = Pp @ np.linalg.inv(Pp + 0.5 * np.eye(2)); I_K = np.eye(2) - K
    Pc = I_K @ Pp @ I_K.T + 0.5 * K @ K.T; Pn.append(Pc); Ksn.append(K)
pw = [(np.asarray(W, np.float64), np.asarray(b, np.float64)) for W, b in params]
def mlp64(p, v):
    for W, b in p[:-1]:
        v = np.tanh(W @ v + b)
    return p[-1][0] @ v + p[-1][1]
x = np.zeros(2); fil = []
for t in range(50):
    xp = np.asarray(A_ORIG, np.float64) @ x + np.asarray(B, np.float64)[:, 0] * float(us[t, 0]) + mlp64(pw, np.r_[x, float(us[t, 0])])
    x = xp + Ksn[t] @ (np.asarray(ys[t], np.float64) - xp); fil.append(x)
y_preds, Ps = predict_with_kalman(us, ys, params, A_ORIG, B, C, Q, R, x0, P0)
out(f"float64 forward pass against JAX's float32 filtered states: max difference {np.abs(np.array(fil) - np.asarray(y_preds)).max():.1e}; "
    f"posterior sds {np.abs(np.sqrt(np.array([np.diag(P) for P in Pn])) - np.sqrt(np.array([np.diag(P) for P in np.asarray(Ps)]))).max():.1e}")
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "d.npz")
    arr0 = np.empty(len(p0), dtype=object)
    for i, (W, b) in enumerate(p0):
        arr0[i] = (np.asarray(W), np.asarray(b))
    np.savez(path, us=np.asarray(us), ys=np.asarray(ys), params0=arr0, losses32=np.asarray(losses))
    proc = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "x64part", path], capture_output=True, text=True)
    out(proc.stdout.strip() or proc.stderr.strip()[-3000:])

section("3. What the network learned (Part 3 and the hero)")
err, _, _ = compute_dynamics_error(xs, us, params, A_ORIG, B, A_true=A_ORIG, B_true=B)
truth = 3 * np.sin(np.asarray(xs[:, 0])) + 2 * np.cos(np.asarray(xs[:, 1]))
out(f"training rollout: residual RMS error {float(jnp.sqrt(jnp.mean(err ** 2))):.2f} against a residual of RMS {np.sqrt(np.mean(truth ** 2)):.2f}")
s7 = [float(v) for v in scores(params, A_ORIG, Q, R, xs7, us7, ys7)]
z7 = [float(v) for v in scores(jax.tree_util.tree_map(jnp.zeros_like, params), A_ORIG, Q, R, xs7, us7, ys7)]
out(f"key 7: filtered RMSE {s7[0]:.2f} (the same filter with a zero residual: {z7[0]:.2f}; its forecast {z7[1]:.2f}); forecast RMSE {s7[1]:.2f} "
    f"against the linear baseline A y_t-1 + B u_t {s7[2]:.2f}; residual error {s7[3]:.2f}; filtered / forecast {s7[0] / s7[1]:.2f}")
x7 = np.asarray(xs7)
inside = (np.abs(x7) <= 8).all(1)
out(f"key 7 reaches x0 = {x7[:, 0].max():.1f}; {inside.sum()} of its 50 states lie inside [-8, 8]^2; it first leaves after {int(np.argmin(inside))} steps")
s3 = score3(params, A_ORIG, Q, R)
out(f"keys 7, 11, 13: forecast {np.round(s3[:, 1], 2).tolist()} (mean {s3[:, 1].mean():.2f}) against the linear baseline "
    f"{np.round(s3[:, 2], 2).tolist()} (mean {s3[:, 2].mean():.2f}); residual error mean {s3[:, 3].mean():.2f}")
stab_r = M[("stable", 50, "repo", 1000, "filtered")]
ss = score3(stab_r["params"], A_STAB, Q, R)
out(f"bounded plant, repository's noise, 1000 steps (the map's third button): forecast mean {ss[:, 1].mean():.2f}; key 7 forecast "
    f"{ss[0, 1]:.2f} against {ss[0, 2]:.2f}")
tx = np.asarray(M[("stable", 50, "repo", 1000, "filtered")]["data"][0])
lo, hi = tx.min(0), tx.max(0)
k7 = np.asarray(generate_data(T=50, key=random.PRNGKey(7), A_true=A_STAB)[0])
out(f"bounded test key 7: {int((~((k7 >= lo) & (k7 <= hi)).all(1)).sum())} of 50 states outside the training bounding box, "
    f"{int((~(np.abs(k7) <= 8).all(1)).sum())} outside [-8, 8]^2, x0 down to {k7[:, 0].min():.2f}")
grid = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 81), np.linspace(lo[1], hi[1], 81)), -1).reshape(-1, 2)
fp = jax.vmap(lambda v: mlp(stab_r["params"], jnp.concatenate([v, jnp.zeros(1)]))[0])(jnp.asarray(grid, jnp.float32))
egrid = np.abs(np.asarray(fp) - (3 * np.sin(grid[:, 0]) + 2 * np.cos(grid[:, 1])))
near = np.min(np.linalg.norm(grid[:, None, :] - tx[None, :, :], axis=2), axis=1) < 0.5
out(f"bounded-plant network at u = 0: median |error| {np.median(egrid):.2f} over the training box ({np.mean(egrid > 2):.0%} above 2), "
    f"{np.median(egrid[near]):.2f} within 0.5 of a training state")

section("4. The study on three fresh rollouts (Part 4, the table, exercise 4)")
for key, m in M.items():
    s = score3(m["params"], m["A"], m["Q"], m["R"]).mean(0)
    out(f"{key}: final loss {float(m['losses'][-1]):.4g}; 3-key mean forecast {s[1]:.2f}, linear baseline {s[2]:.2f}, residual error {s[3]:.2f}")

section("5. The same models on 1000 fresh rollouts (generate_data, keys 100-1099)")
honest_key = ("stable", 500, "honest", 2000, "filtered")
for key in (("original", 50, "repo", 1000, "filtered"), ("original", 50, "honest", 2000, "filtered"),
            ("stable", 50, "repo", 2000, "filtered"), honest_key, ("stable", 500, "honest", 2000, "forecast")):
    m = M[key]
    sm = score_many(m["params"], m["A"], m["Q"], m["R"])
    out(f"{key}: forecast mean {sm[:, 1].mean():.2f} median {np.median(sm[:, 1]):.2f}; linear baseline mean {sm[:, 2].mean():.2f} "
        f"median {np.median(sm[:, 2]):.2f}; the learned forecast wins on {int(np.sum(sm[:, 1] < sm[:, 2]))} of 1000")
m = M[honest_key]
sm = score_many(m["params"], m["A"], m["Q"], m["R"])
train_x = np.asarray(m["data"][0]); lo, hi = train_x.min(0), train_x.max(0)
xs_many = np.asarray(many_rollouts(A_STAB)[0])
outside = (~((xs_many >= lo) & (xs_many <= hi)).all(2)).mean(1)
excess = sm[:, 1] - sm[:, 2]                                       # the learned forecast's RMSE minus the baseline's
worst = np.argsort(excess)[-20:]
lose = excess > 0
out(f"final model: its excess error over the baseline correlates {np.corrcoef(excess, outside)[0, 1]:.2f} with the share of a rollout's "
    f"states outside the training box; the 20 worst rollouts spend {outside[worst].mean():.0%} of their steps outside it; "
    f"{np.mean(outside > 0):.0%} of rollouts leave it at some point; the rollouts it loses on ({int(lose.sum())}) spend a median "
    f"{np.median(outside[lose]):.0%} of their steps outside, those it wins on {np.median(outside[~lose]):.0%}; it loses by more than 2 on {int(np.sum(excess > 2))}")
cmax = np.abs(xs_many).max(axis=(1, 2))
out(f"bounded-plant rollouts, largest coordinate: median {np.median(cmax):.2f}, {int(np.sum(cmax <= 7))} of 1000 stay inside [-7, 7]^2, "
    f"{int(np.sum(cmax >= 12.5))} reach 12.5 or more, largest {cmax.max():.2f}; "
    f"{np.mean(np.linalg.norm(xs_many, axis=2) <= 10):.0%} of all their states lie within 10 of the origin")

section("6. Exercises 2 and 3")
gj = jax.grad(lambda p: loss_fn(p, A_ORIG, B, C, Q, R, x0, P0, us, ys))(p0)
L32 = lambda p: float(loss_fn(p, A_ORIG, B, C, Q, R, x0, P0, us, ys))
rels, last = [], []
for layer in range(5):
    for (i, j) in ((0, 0), (1, 1), (0, 2)):   # valid in every layer (the last is 2 x 16)
        h = np.float32(1e-4)
        W = p0[layer][0]
        pp = [(W.at[i, j].add(h) if l == layer else w_, b_) for l, (w_, b_) in enumerate(p0)]
        pm = [(W.at[i, j].add(-h) if l == layer else w_, b_) for l, (w_, b_) in enumerate(p0)]
        fd = (L32(pp) - L32(pm)) / (2 * 1e-4)
        rel = abs(fd - float(gj[layer][0][i, j])) / abs(float(gj[layer][0][i, j]))
        rels.append(rel); last.append(rel) if layer == 4 else None
out(f"exercise 2, float32 with h = 1e-4: the difference quotient comes in multiples of ulp({float(losses[0]):.3f})/2h = "
    f"{float(np.spacing(np.float32(losses[0]))) / 2e-4:.2e}; median relative error over 15 weights {np.median(rels):.2f}; "
    f"last layer {np.round(last, 4).tolist()}; gradients of the last layer about {float(jnp.abs(gj[4][0]).mean()):.0e}")
allg = np.abs(np.concatenate([np.asarray(a).ravel() for a in jax.tree_util.tree_leaves(gj)]))
out(f"  all {allg.size} gradient entries: median |g| {np.median(allg):.1e}; {np.mean(allg < 2.98e-4):.0%} are below the step")


def loss_ekf(params, A, Q, R, us, ys, kind="filtered"):
    """Exercise 3: the covariance predicted with A + df/dx instead of A."""
    def step(carry, inp):
        x, P = carry
        u, y = inp
        f = lambda v: mlp(params, jnp.concatenate([v, u]))
        J = jax.jacfwd(f)(x)
        xp = A @ x + B @ u + f(x)
        Pp = (A + J) @ P @ (A + J).T + Q
        S = Pp + R
        K = _kalman_gain(Pp, C, S)
        xn = xp + K @ (y - xp)
        Pn = _joseph_covariance_update(Pp, K, C, R)
        return (xn, Pn), (xn, xp)
    _, (xn, xp) = jax.lax.scan(step, (x0, P0), (us, ys))
    return jnp.mean((ys - (xn if kind == "filtered" else xp)) ** 2), xp


def train_ekf(p, steps=1000):
    """The repository's training loop (Adam, 1e-3, filtered loss) on the exercise 3 filter."""
    opt = optax.adam(1e-3)
    st = opt.init(p)

    @jax.jit
    def upd(p, st):
        loss, gr = jax.value_and_grad(lambda q: loss_ekf(q, A_ORIG, Q, R, us, ys)[0])(p)
        updates, st = opt.update(gr, st, p)
        return optax.apply_updates(p, updates), st, loss
    for _ in range(steps):
        p, st, loss = upd(p, st)
    return p, loss


pe, _l = train_ekf(p0)


def ekf_scores(p, xs_, us_, ys_):
    _, xp = loss_ekf(p, A_ORIG, Q, R, us_, ys_)
    base = ys_[:-1] @ A_ORIG.T + us_[1:] @ B.T
    return jnp.sqrt(jnp.mean((xp - ys_) ** 2)), jnp.sqrt(jnp.mean((base - ys_[1:]) ** 2))


e3 = np.array([[float(v) for v in ekf_scores(pe, *generate_data(T=50, key=random.PRNGKey(k)))] for k in KEYS3])
em = np.array([np.asarray(v) for v in jax.jit(jax.vmap(lambda a, b, c: ekf_scores(pe, a, b, c)))(*many_rollouts(A_ORIG))]).T
out(f"exercise 3, covariance with A + J: final training loss {float(_l):.4f}; keys 7, 11, 13 forecast mean {e3[:, 0].mean():.2f} "
    f"(fixed gains: {s3[:, 1].mean():.2f}); 1000 rollouts mean {em[:, 0].mean():.2f} (fixed gains: "
    f"{score_many(params, A_ORIG, Q, R)[:, 1].mean():.2f}) against the baseline's {em[:, 1].mean():.2f}; it wins on {int(np.sum(em[:, 0] < em[:, 1]))} of 1000")
Jmax = max(float(np.linalg.norm(np.asarray(A_ORIG) + np.array([[3 * np.cos(a), -2 * np.sin(b)]] * 2), 2) ** 2)
           for a in np.linspace(-np.pi, np.pi, 181) for b in np.linspace(-np.pi, np.pi, 181))
out(f"for the true residual, the largest squared singular value of A + J is {Jmax:.1f}")
