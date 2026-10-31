"""Reproduce every number quoted on chapter 04 of "Kalman Filters with JAX"
(public/tutorials/kalman-jax/04-a-real-series-next-hour-no2.html).

Usage:  python ch04_numbers.py [AirQualityUCI.csv]          # print everything the page quotes
        python ch04_numbers.py data [AirQualityUCI.csv]     # print the page's data file, interactive/ch04-data.js
The data are the UCI Air Quality dataset (S. De Vito, doi:10.24432/C59K5F, CC BY 4.0); the script does not download
them. Pass the CSV's path, or put it at ../data/raw/AirQualityUCI.csv (where the course download expects it) or
data/raw/AirQualityUCI.csv. Checked with jax 0.12.0.1, optax 0.2.8, numpy 2.5 and pandas 3.1 on a laptop CPU; it
trains three models over the 5173 training hours and takes several minutes. The pipeline, the network and the
forecaster are copied from the repository (kalman_jax/air_quality.py, learned_dynamics.py, forecasters.py, filters.py;
docstrings shortened and input checks removed, computations unchanged) so that the script runs alone.
"""
from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax import random, value_and_grad

ARGS = [a for a in sys.argv[1:] if a != "data"]
DATA = "data" in sys.argv[1:]
REAL_STDOUT = sys.stdout
if DATA:
    sys.stdout = io.StringIO()
out = (lambda *a, **k: None) if DATA else print


def section(title):
    out(f"\n=== {title}")


# ---------------------------------------------------------------------------------------------------------------
# kalman_jax/air_quality.py
DEFAULT_TARGET_COLUMN = "NO2(GT)"
DEFAULT_INPUT_COLUMNS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        return (values * self.std + self.mean).astype(np.float32)


def _read_air_quality_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    parse_attempts = ({"sep": ";", "decimal": ","}, {"sep": ",", "decimal": "."})
    expected_columns = {"Date", "Time", DEFAULT_TARGET_COLUMN}
    for options in parse_attempts:
        frame = pd.read_csv(csv_path, **options)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = frame.loc[:, [column for column in frame.columns if column and not column.startswith("Unnamed")]]
        if expected_columns.issubset(frame.columns):
            return frame
    raise ValueError(f"Could not parse the air-quality CSV in {csv_path}.")


def load_air_quality_frame(csv_path):
    frame = _read_air_quality_csv(csv_path).copy()
    numeric_columns = [column for column in frame.columns if column not in {"Date", "Time"}]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[numeric_columns] = frame[numeric_columns].replace(-200, np.nan)
    date_str = frame["Date"].astype(str).str.strip()
    time_str = frame["Time"].astype(str).str.strip().str.replace(".", ":", regex=False)
    frame["timestamp"] = pd.to_datetime(date_str + " " + time_str, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame


def _fit_standardizer(values):
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return Standardizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def _time_split_bounds(n_rows, train_fraction, val_fraction):
    train_end = int(n_rows * train_fraction)
    val_end = train_end + int(n_rows * val_fraction)
    if train_end < 2 or val_end <= train_end or val_end >= n_rows:
        raise ValueError("Not enough rows to form train/validation/test time splits.")
    return train_end, val_end


def build_no2_forecasting_dataset(frame, *, input_columns=None, target_column=DEFAULT_TARGET_COLUMN, horizon=1,
                                  train_fraction=0.7, val_fraction=0.15):
    input_columns = DEFAULT_INPUT_COLUMNS if input_columns is None else list(input_columns)
    model_frame = frame.dropna(subset=input_columns + [target_column]).copy()
    model_frame["target_next"] = model_frame[target_column].shift(-horizon)
    model_frame = model_frame.dropna(subset=["target_next"]).reset_index(drop=True)

    controls_raw = model_frame[input_columns].to_numpy(dtype=np.float32)
    current_target_raw = model_frame[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)
    future_target_raw = model_frame["target_next"].to_numpy(dtype=np.float32).reshape(-1, 1)
    timestamps = model_frame["timestamp"].to_numpy()

    train_end, val_end = _time_split_bounds(len(model_frame), train_fraction, val_fraction)
    control_scaler = _fit_standardizer(controls_raw[:train_end])
    target_scaler = _fit_standardizer(current_target_raw[:train_end])
    controls = control_scaler.transform(controls_raw)
    current_target = target_scaler.transform(current_target_raw)
    future_target = target_scaler.transform(future_target_raw)

    train_effective_end = train_end - horizon
    val_effective_end = val_end - horizon
    if train_effective_end <= 0 or val_effective_end <= train_end:
        raise ValueError("Forecast horizon is too large for the chosen split sizes.")

    def make_split(start, end):
        return {"timestamps": timestamps[start:end], "controls_raw": controls_raw[start:end], "controls": controls[start:end],
                "current_target_raw": current_target_raw[start:end], "current_target": current_target[start:end],
                "future_target_raw": future_target_raw[start:end], "future_target": future_target[start:end]}

    return {"target_column": target_column, "input_columns": input_columns, "horizon": horizon,
            "control_scaler": control_scaler, "target_scaler": target_scaler,
            "splits": {"train": make_split(0, train_effective_end), "val": make_split(train_end, val_effective_end),
                       "test": make_split(val_end, len(model_frame))}}


def fit_linear_regression(features, targets):
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    design = np.concatenate([features, np.ones((len(features), 1), dtype=np.float32)], axis=1)
    weights, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    return weights.astype(np.float32)


def predict_linear_regression(weights, features):
    features = np.asarray(features, dtype=np.float32)
    design = np.concatenate([features, np.ones((len(features), 1), dtype=np.float32)], axis=1)
    return design @ weights


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))


# kalman_jax/learned_dynamics.py and forecasters.py, filters.py
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
        params.append((random.normal(keys[2 * layer_index], (n_out, n_in)) * 0.1,
                       random.normal(keys[2 * layer_index + 1], (n_out,)) * 0.1))
    return params


def _kalman_gain(P, H, innovation_cov):
    return jnp.linalg.solve(innovation_cov, H @ P.T).T


def _joseph_covariance_update(P, K, H, R):
    identity = jnp.eye(P.shape[0], dtype=P.dtype)
    residual_factor = identity - K @ H
    return residual_factor @ P @ residual_factor.T + K @ R @ K.T


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
        return (x_new, P_new), (y_pred, y_filtered, S_pred, S_filtered)

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

HERE = Path(__file__).resolve().parent
candidates = [Path(a) for a in ARGS] + [HERE.parent / "data" / "raw" / "AirQualityUCI.csv", Path("data/raw/AirQualityUCI.csv")]
CSV = next((c for c in candidates if c.exists()), None)
if CSV is None:
    sys.exit("AirQualityUCI.csv not found. Download it from https://doi.org/10.24432/C59K5F and pass its path.")

frame = load_air_quality_frame(CSV)
ds = build_no2_forecasting_dataset(frame, input_columns=DEFAULT_INPUT_COLUMNS, horizon=1)
tr, va, te = (ds["splits"][k] for k in ("train", "val", "test"))
ts_, cs_ = ds["target_scaler"], ds["control_scaler"]
inv = lambda v: ts_.inverse_transform(np.asarray(v))
mean_t, std_t = float(ts_.mean[0, 0]), float(ts_.std[0, 0])
w_lin = fit_linear_regression(tr["controls"], tr["future_target"])
ar_feats = lambda s: np.c_[s["controls"], s["current_target"]]
w_ar = fit_linear_regression(ar_feats(tr), tr["future_target"])
input_dim = tr["controls"].shape[1]
B0 = jnp.zeros((1, input_dim), dtype=jnp.float32)
C1 = jnp.array([[1.0]], dtype=jnp.float32)
P0 = jnp.eye(1, dtype=jnp.float32)


def kf_run(params, A, Q, R, split):
    A_, Q_, R_ = (jnp.array([[v]], dtype=jnp.float32) for v in (A, Q, R))
    return forecast_with_kalman(params, A_, B0, C1, Q_, R_, jnp.asarray(split["current_target"][0]), P0,
                                jnp.asarray(split["controls"]), jnp.asarray(split["future_target"]))


def kf_train(A, Q, R, steps):
    A_, Q_, R_ = (jnp.array([[v]], dtype=jnp.float32) for v in (A, Q, R))
    p0 = init_mlp_params(random.PRNGKey(42), [1 + input_dim, 32, 32, 1])
    params, _, losses = train_kalman_forecaster(p0, optax.adam(1e-3), A=A_, B=B0, C=C1, Q=Q_, R=R_,
                                                x0=jnp.asarray(tr["current_target"][0]), P0=P0,
                                                us=jnp.asarray(tr["controls"]), ys=jnp.asarray(tr["future_target"]),
                                                num_steps=steps)
    return params, np.asarray(losses)


MODELS = {"repo": dict(label="repository's settings", A=0.9, Q=0.05, R=0.25, steps=1500),
          "retuned": dict(label="retuned noise", A=1.0, Q=0.5, R=0.05, steps=3000)}
for m in MODELS.values():
    m["params"], m["losses"] = kf_train(m["A"], m["Q"], m["R"], m["steps"])
    for name, split in (("val", va), ("test", te)):
        yp, _, Sp, _ = kf_run(m["params"], m["A"], m["Q"], m["R"], split)
        m[name] = rmse(inv(split["future_target"]), inv(yp))
        if name == "test":
            m["testForecast"] = np.asarray(inv(yp)).ravel()
            m["testStd"] = np.sqrt(np.asarray(Sp)[:, 0, 0]) * std_t
truth_te = inv(te["future_target"]).ravel()        # flat: compare with flat forecasts only
cur_te = inv(te["current_target"]).ravel()
lin_te = inv(predict_linear_regression(w_lin, te["controls"])).ravel()
ar_te = inv(predict_linear_regression(w_ar, ar_feats(te))).ravel()
pm = float(np.mean((tr["future_target"] - tr["current_target"]) ** 2))
_hr = pd.DataFrame({"h": pd.to_datetime(tr["timestamps"]).hour, "v": tr["current_target_raw"].ravel()}).groupby("h")["v"]
by_hour = _hr.mean().where(_hr.size() >= 30).reindex(range(24)).to_numpy()     # 03:00 has a single training reading

if DATA:
    g = lambda v, p=7: "0" if float(format(float(v), f".{p}g")) == 0 else format(float(v), f".{p}g")
    vec = lambda a, p=7: "[" + ",".join(g(v, p) for v in np.asarray(a).ravel()) + "]"
    mat = lambda a, p=7: "[" + ",".join(vec(r, p) for r in np.asarray(a)) + "]"
    prm = lambda P: "[" + ",".join(f"[{mat(W)},{vec(b)}]" for W, b in P) + "]"

    def base(split):
        t = inv(split["future_target"])
        return {"persistence": [rmse(t, inv(split["current_target"])), mae(t, inv(split["current_target"]))],
                "linear": [rmse(t, inv(predict_linear_regression(w_lin, split["controls"]))), mae(t, inv(predict_linear_regression(w_lin, split["controls"])))],
                "linear_ar": [rmse(t, inv(predict_linear_regression(w_ar, ar_feats(split)))), mae(t, inv(predict_linear_regression(w_ar, ar_feats(split))))],
                "climatology": [rmse(t, np.full_like(t, mean_t)), mae(t, np.full_like(t, mean_t))]}
    bl = {k: {kk: [float(g(v, 6)) for v in vv] for kk, vv in base(s).items()} for k, s in (("val", va), ("test", te))}
    models_txt = ",\n".join(
        f'  "{k}": {{label:{json.dumps(m["label"])}, A:{g(m["A"])}, Q:{g(m["Q"])}, R:{g(m["R"])}, steps:{m["steps"]}, '
        f'params:{prm(m["params"])}, losses:{vec(m["losses"][::5], 5)}, val:{g(m["val"], 5)}, test:{g(m["test"], 5)}, '
        f'testForecast:{vec(m["testForecast"], 6)}, testStd:{vec(m["testStd"], 5)}}}' for k, m in MODELS.items())
    tsx = json.dumps([str(t)[:16] for t in te["timestamps"]])
    REAL_STDOUT.write(
        "/* Kalman filters with JAX, chapter 04 data: the test split of the UCI Air Quality dataset (De Vito et\n"
        "   al., CC BY 4.0) prepared by the repository's pipeline -- timestamps, the eight standardised sensor and\n"
        "   weather inputs, the current and next-hour NO2 -- with the standardisation constants, the baselines'\n"
        "   weights and metrics, and the trained Kalman-MLP forecasters. Printed by code/ch04_numbers.py data.\n"
        "   Forecasts and metrics are recomputed live. */\n"
        f"window.KJ3 = {{columns:{json.dumps(DEFAULT_INPUT_COLUMNS)}, targetMean:{g(mean_t)}, targetStd:{g(std_t, 6)}, "
        f"controlMean:{vec(cs_.mean, 7)}, controlStd:{vec(cs_.std, 7)},\n"
        f"  sizes:{{train:{len(tr['controls'])},val:{len(va['controls'])},test:{len(te['controls'])}}}, "
        f"persistTrainMSE:{g(pm, 4)}, byHour:[{','.join('null' if np.isnan(v) else g(v, 5) for v in by_hour)}],\n"
        f"  baselines:{json.dumps(bl)},\n"
        f"  wLin:{vec(w_lin)}, wAr:{vec(w_ar)},\n"
        f"  models:{{\n{models_txt}}},\n"
        f"  test:{{timestamps:{tsx}, controls:{mat(te['controls'], 6)}, current:{vec(te['current_target'], 6)}, "
        f"future:{vec(te['future_target'], 6)}, currentRaw:{vec(te['current_target_raw'])}, futureRaw:{vec(te['future_target_raw'])}}}\n"
        "};\n")
    sys.exit()

out("jax", jax.__version__, "optax", optax.__version__, "numpy", np.__version__, "pandas", pd.__version__)
out("data:", CSV.name)

section("1. The data and the pipeline (Part 1)")
n_all = len(frame)
complete = frame.dropna(subset=DEFAULT_INPUT_COLUMNS + [DEFAULT_TARGET_COLUMN])
out(f"{n_all} hourly rows, {frame['timestamp'].min()} to {frame['timestamp'].max()}; {len(complete)} complete; "
    f"splits {len(tr['controls'])} / {len(va['controls'])} / {len(te['controls'])}")
for name, s in (("train", tr), ("val", va), ("test", te)):
    out(f"{name}: {str(s['timestamps'][0])[:16]} to {str(s['timestamps'][-1])[:16]}")
out(f"training NO2 mean {mean_t:.1f}, sd {std_t:.1f}")
y = complete[DEFAULT_TARGET_COLUMN].to_numpy(dtype=float)
out(f"whole record, consecutive complete rows: lag-one correlation {np.corrcoef(y[:-1], y[1:])[0, 1]:.2f}")
tst = pd.to_datetime(te["timestamps"])
gaps = np.diff(tst.values).astype("timedelta64[h]").astype(int)
nxt = np.r_[gaps, 1]                                             # hours from each test row to the next row
out(f"test: {int(np.sum(gaps > 1))} places where consecutive rows are more than an hour apart; "
    f"the next-row target is {np.bincount(nxt)[2:4].tolist()} times two and three hours ahead")

section("2. The baselines (Part 2, the table)")
for name, s in (("val", va), ("test", te)):
    t = inv(s["future_target"])
    out(f"{name}: persistence {rmse(t, inv(s['current_target'])):.4f} / {mae(t, inv(s['current_target'])):.4f}; "
        f"linear (sensors) {rmse(t, inv(predict_linear_regression(w_lin, s['controls']))):.4f}; "
        f"linear + current NO2 {rmse(t, inv(predict_linear_regression(w_ar, ar_feats(s)))):.4f}; "
        f"climatology {rmse(t, np.full_like(t, mean_t)):.4f}")
sig_te, mu_te = truth_te.std(), truth_te.mean()
rho_te = np.corrcoef(cur_te, truth_te)[0, 1]
p_te = rmse(truth_te, cur_te)
out(f"test split: next-hour NO2 mean {mu_te:.2f}, sd {sig_te:.2f}; corrcoef(current, next) {rho_te:.3f}; "
    f"1 - (persistence/sd)^2/2 = {1 - (p_te / sig_te) ** 2 / 2:.3f} (with the training sd: {1 - (p_te / std_t) ** 2 / 2:.3f})")
out(f"climatology {rmse(truth_te, np.full_like(truth_te, mean_t)):.3f} = sqrt(sd^2 + (test mean - training mean)^2) = "
    f"{np.sqrt(sig_te ** 2 + (mu_te - mean_t) ** 2):.3f}; the shift {mu_te - mean_t:.1f} is {(mu_te - mean_t) / std_t:.2f} training sds")
out(f"persistence's standardised MSE on the training split: {pm:.4f}")

section("3. The Kalman-MLP (Part 3, the hero, the loss chart)")


def steady_gain(A, Q, R):
    P = 1.0
    for _ in range(5000):
        Pp = A * A * P + Q; K = Pp / (Pp + R); P = (1 - K) ** 2 * Pp + K * K * R
    return K


for k, m in MODELS.items():
    out(f"{k} (A={m['A']}, Q={m['Q']}, R={m['R']}, {m['steps']} steps): val {m['val']:.3f}, test {m['test']:.3f}; "
        f"final training loss {float(m['losses'][-1]):.4f}; steady-state gain {steady_gain(m['A'], m['Q'], m['R']):.3f}; "
        f"parameters {sum(int(np.size(W)) + int(np.size(b)) for W, b in m['params'])}")
rp = MODELS["repo"]
fc = rp["testForecast"]; sd = rp["testStd"]
out(f"repository's model: RMSE against the next hour {rmse(truth_te, fc):.2f}, against the current hour {rmse(cur_te, fc):.2f}; "
    f"band coverage {np.mean(np.abs(truth_te - fc) <= 1.96 * sd):.1%}")
slope = lambda p: np.polyfit(truth_te, p, 1)[0]
out(f"regression slope of forecast on outcome: Kalman-MLP {slope(fc):.2f}, persistence {slope(cur_te):.2f}")
for A, Q, R in ((1.0, 0.5, 0.05), (1.0, 10.0, 0.001)):
    yp, *_ = kf_run(rp["params"], A, Q, R, te)
    out(f"the repository's weights inside A={A}, Q={Q}, R={R} (gain {steady_gain(A, Q, R):.3f}): test RMSE {rmse(truth_te, inv(yp).ravel()):.2f}")

section("4. Where it loses (Part 4)")
hours = pd.to_datetime(te["timestamps"]).hour.to_numpy()
by = lambda p: np.array([np.sqrt(np.mean((truth_te[hours == h] - p[hours == h]) ** 2)) for h in range(24)])
hK, hP, hA = by(fc), by(cur_te), by(ar_te)
out(f"worst hour: Kalman-MLP {int(np.argmax(hK))}:00, persistence {int(np.argmax(hP))}:00, linear {int(np.argmax(hA))}:00")
out("Kalman-MLP minus persistence by hour:", " ".join(f"{h}:{hK[h] - hP[h]:.1f}" for h in range(24)))
jump = np.diff(by_hour)
out(f"training rows per hour: 03:00 has {int(_hr.size().get(3, 0))}; training diurnal cycle (03:00 left out): largest rise "
    f"{np.nanmax(jump):.0f} into {int(np.nanargmax(jump)) + 1}:00")
eve = lambda a: 12 + int(np.argmax(a[12:]))
out(f"evening worst hours: persistence {eve(hP)}:00, linear {eve(hA)}:00, Kalman-MLP {eve(hK)}:00; Kalman-MLP minus persistence at "
    f"20:00 {hK[20] - hP[20]:.1f}, 21:00 {hK[21] - hP[21]:.1f}; at 02:00 the Kalman-MLP scores {hK[2]:.1f}")
# The record's clock does not follow summer time: the morning rise moves by an hour with it.
_f = frame.dropna(subset=[DEFAULT_TARGET_COLUMN])
for name, lo_, hi_ in (("summer time 2004", "2004-03-28", "2004-10-31"), ("winter 2004", "2004-10-31", "2004-12-20"),
                       ("test weeks before 27 March 2005", "2005-02-16", "2005-03-27"), ("test weeks after", "2005-03-27", "2005-04-05")):
    d_ = _f[(_f.timestamp >= lo_) & (_f.timestamp < hi_)]
    j_ = d_.groupby(d_.timestamp.dt.hour)[DEFAULT_TARGET_COLUMN].mean().reindex(range(24)).diff().iloc[4:12]
    out(f"  {name}: steepest morning rise {j_.max():+.0f} into {int(j_.idxmax())}:00 ({len(d_)} readings)")
tmean = np.array([truth_te[hours == h].mean() for h in range(24)])
out(f"test weeks: mean next-hour NO2 rises {tmean[7] - tmean[6]:.0f} from the 06:00 rows to the 07:00 rows "
    f"(the 07:00 row's target is 08:00); rows per hour: hour 3 has {int(np.sum(hours == 3))}, the others {int(np.min(np.delete(np.bincount(hours), 3)))} to {int(np.max(np.bincount(hours)))}")
two = (nxt == 1)
out(f"persistence at 02:00: {hP[2]:.1f} over all rows, {rmse(truth_te[(hours == 2) & two], cur_te[(hours == 2) & two]):.1f} over the "
    f"{int(np.sum((hours == 2) & two))} without a gap; at 01:00 {hP[1]:.1f}, at 03:00 {hP[3]:.1f}")

section("5. Bias and spread on the test split (Part 4)")
for name, p in (("persistence", cur_te), ("linear + current NO2", ar_te), ("Kalman-MLP, repository", fc),
                ("Kalman-MLP, retuned", MODELS["retuned"]["testForecast"]), ("linear from sensors", lin_te)):
    e = truth_te - p
    out(f"{name}: mean error {e.mean():+.2f}, error sd {e.std():.2f}, RMSE {np.sqrt(np.mean(e ** 2)):.2f}; bias share of MSE {e.mean() ** 2 / np.mean(e ** 2):.0%}")
zero_params = [(jnp.zeros((1, 1 + input_dim)), jnp.zeros(1))]
yp0, *_ = kf_run(zero_params, 0.9, 0.05, 0.25, te)
out(f"the repository's filter with a zero network: mean error {np.mean(truth_te - inv(yp0).ravel()):+.2f}, RMSE {rmse(truth_te, inv(yp0).ravel()):.2f}")
xbar = np.c_[te["controls"], te["current_target"]].mean(0)
out("test split, mean standardised inputs:", dict(zip(DEFAULT_INPUT_COLUMNS + ["NO2 now"], np.round(xbar, 2).tolist())))
out(f"linear + current NO2: weight on the current NO2 {float(w_ar[8, 0]):.3f}; its share of the bias "
    f"{(1 - float(w_ar[8, 0])) * float(xbar[8]) * std_t:+.1f}, the sensor terms' {-float(np.dot(w_ar[:8, 0], xbar[:8])) * std_t - float(w_ar[9, 0]) * std_t:+.1f}")
c4 = te["controls"][:, 3]; ah = te["controls"][:, 7]
out(f"within the test weeks, PT08.S4(NO2) correlates {np.corrcoef(c4, truth_te)[0, 1]:.2f} with NO2 and {np.corrcoef(c4, ah)[0, 1]:.2f} with AH")

section("6. What would beat persistence (Part 5, exercises 2 to 4)")


def clock(split):
    h = np.array([int(str(t)[11:13]) for t in split["timestamps"]], dtype=np.float32)
    w = 2 * np.pi * h / 24
    return np.c_[np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)]


for horizon in (1, 3):
    d = build_no2_forecasting_dataset(frame, input_columns=DEFAULT_INPUT_COLUMNS, horizon=horizon)
    s_tr, s_va, s_te = (d["splits"][k] for k in ("train", "val", "test"))
    iv = d["target_scaler"].inverse_transform
    rows = {"persistence": lambda s: s["current_target"],
            "sensors + current NO2": (np.c_, lambda s: np.c_[s["controls"], s["current_target"]]),
            "current NO2 + clock": (np.c_, lambda s: np.c_[s["current_target"], clock(s)]),
            "sensors + current NO2 + clock": (np.c_, lambda s: np.c_[s["controls"], s["current_target"], clock(s)])}
    txt = []
    for name, spec in rows.items():
        if name == "persistence":
            pv, pt = spec(s_va), spec(s_te)
        else:
            f = spec[1]; w = fit_linear_regression(f(s_tr), s_tr["future_target"])
            pv, pt = predict_linear_regression(w, f(s_va)), predict_linear_regression(w, f(s_te))
        txt.append(f"{name} {rmse(iv(s_va['future_target']), iv(pv)):.1f} / {rmse(iv(s_te['future_target']), iv(pt)):.1f}")
    out(f"horizon {horizon} (val / test):", "; ".join(txt))
d1 = build_no2_forecasting_dataset(frame, input_columns=DEFAULT_INPUT_COLUMNS, horizon=1)
f3 = lambda s: np.c_[s["controls"], s["current_target"], clock(s)]
w3 = fit_linear_regression(f3(d1["splits"]["train"]), d1["splits"]["train"]["future_target"])
r3 = rmse(truth_te, inv(predict_linear_regression(w3, f3(te))).ravel())
f2 = lambda s: np.c_[s["current_target"], clock(s)]
w2 = fit_linear_regression(f2(tr), tr["future_target"])
r2 = rmse(truth_te, inv(predict_linear_regression(w2, f2(te))).ravel())
out(f"gains: sensors + NO2 + clock {r3:.2f} is {1 - r3 / rmse(truth_te, ar_te):.1%} below the linear model without the clock and "
    f"{1 - r3 / p_te:.1%} below persistence; NO2 + clock {r2:.2f} is {1 - r2 / p_te:.1%} below persistence")

# Exercise 3: Q and R learned with the network, as log Q and log R in the same Adam optimiser.
def joint_loss(theta):
    params, lq, lr = theta
    return forecasting_loss_fn(params, jnp.array([[0.9]]), B0, C1, jnp.exp(lq)[None, None], jnp.exp(lr)[None, None],
                               jnp.asarray(tr["current_target"][0]), P0, jnp.asarray(tr["controls"]), jnp.asarray(tr["future_target"]))


opt = optax.adam(1e-3)
theta = (init_mlp_params(random.PRNGKey(42), [1 + input_dim, 32, 32, 1]), jnp.log(0.05), jnp.log(0.25))
st = opt.init(theta)


@jax.jit
def joint_step(theta, st):
    loss, grads = jax.value_and_grad(joint_loss)(theta, )
    upd, st = opt.update(grads, st, theta)
    return optax.apply_updates(theta, upd), st, loss, grads


qr_trace = []
for i in range(1500):
    theta, st, loss, grads = joint_step(theta, st)
    if i in (0, 499, 999, 1499):
        qr_trace.append((i + 1, float(jnp.exp(theta[1])), float(jnp.exp(theta[2])), float(grads[1]), float(grads[2])))
ql, rl = float(jnp.exp(theta[1])), float(jnp.exp(theta[2]))
yp, *_ = kf_run(theta[0], 0.9, ql, rl, te)
ypv, *_ = kf_run(theta[0], 0.9, ql, rl, va)
out("exercise 3, (step, Q, R, dL/dlogQ, dL/dlogR):", [tuple(round(v, 5) for v in t) for t in qr_trace])
out(f"  QR {0.05 * 0.25:.5f} at the start and {ql * rl:.5f} after 1500 steps; gain {steady_gain(0.9, ql, rl):.2f}; "
    f"val {rmse(inv(va['future_target']), inv(ypv)):.1f}, test {rmse(truth_te, inv(yp).ravel()):.1f}")
for c in (0.1, 1, 10, 100):
    yp, _, Sp, _ = kf_run(rp["params"], 0.9, 0.05 * c, 0.25 * c, te)
    sdc = np.sqrt(np.asarray(Sp)[:, 0, 0]) * std_t
    out(f"  Q and R both scaled by {c}: test RMSE {rmse(truth_te, inv(yp).ravel()):.2f}, gain {steady_gain(0.9, 0.05 * c, 0.25 * c):.3f}, "
        f"coverage {np.mean(np.abs(truth_te - inv(yp).ravel()) <= 1.96 * sdc):.1%}")

section("7. Exercise 1: an honest target (gaps)")
gap_rows = np.where(nxt > 1)[0]
starts = pd.Series(pd.to_datetime(te["timestamps"][gap_rows]).hour).value_counts().to_dict()
out(f"test rows whose next row is more than an hour later: {len(gap_rows)} (two hours: {int(np.sum(nxt == 2))}, three: {int(np.sum(nxt == 3))}); "
    f"by the hour of the row: {starts}")
keep = nxt == 1
out(f"persistence on the test split's {int(keep.sum())} rows whose target is one hour ahead: {rmse(truth_te[keep], cur_te[keep]):.2f} (all rows: {p_te:.2f})")
