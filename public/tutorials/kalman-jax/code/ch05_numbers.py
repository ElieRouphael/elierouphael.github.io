"""Reproduce every number quoted on chapter 05 of "Kalman Filters with JAX"
(public/tutorials/kalman-jax/05-a-first-order-filter-in-c.html).

Usage:  python ch05_numbers.py [first-order-c]          # print everything the page quotes
        python ch05_numbers.py data [first-order-c]     # print the page's data file, interactive/ch05-data.js
first-order-c is the folder of the First-Order-Filter-using-C repository, holding the recording
enregistrement-bruit.txt, TD3_original.c and TD3.c. The course download has it next to this script's folder,
which is where the script looks when no folder is given. If gcc is on the PATH, the script also compiles and runs
the original program, a copy of it with bugs 1 and 2 fixed, and TD3.c, each in a temporary folder so that the
repository's sortie.txt is left alone. The data mode needs gcc, because the page checks its filter against the
compiled filtrage(). Checked with jax 0.12.0.1, numpy 2.5 and gcc 13.2 (MinGW) on a laptop CPU; it runs in seconds.
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)      # the C program computes in double; so does the page's JavaScript
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")       # the programs print French accents
ARGS = [a for a in sys.argv[1:] if a != "data"]
DATA = "data" in sys.argv[1:]
REAL_STDOUT = sys.stdout
if DATA:
    sys.stdout = io.StringIO()
out = (lambda *a, **k: None) if DATA else print


def section(title):
    out(f"\n=== {title}")


HERE = Path(__file__).resolve().parent
candidates = [Path(a) for a in ARGS] + [HERE.parent / "first-order-c", Path("first-order-c")]
REPO = next((c for c in candidates if (c / "enregistrement-bruit.txt").exists()), None)
if REPO is None:
    sys.exit("first-order-c/enregistrement-bruit.txt not found: pass the folder of the First-Order-Filter-using-C repository.")
GCC = shutil.which("gcc")
if DATA and GCC is None:
    sys.exit("the data mode needs gcc on the PATH: the page's equivalence check compares with the compiled filtrage().")

Te = 0.01
u_np = np.loadtxt(REPO / "enregistrement-bruit.txt")
N = len(u_np)
t = np.arange(N) * Te
u = jnp.asarray(u_np)


# ---------------------------------------------------------------------------------------------------------------
# The filter, as lax.scan. The carry (y_prev, u_prev) is TD3.c's Filtre struct and the step is filtre_step().
# current=False is the program's form, y_k = z0 y_{k-1} + K (1 - z0) u_{k-1}, with y_0 = 0;
# current=True feeds the current sample, y_k = z0 y_{k-1} + K (1 - z0) u_k, starting at rest (y_{-1} = 0).
def filtrage(u, log_tau, K=1.0, current=False):
    z0 = jnp.exp(-Te / jnp.exp(log_tau))
    b = K * (1 - z0)

    def step(carry, uk):
        y_prev, u_prev = carry
        y = z0 * y_prev + b * (uk if current else u_prev)
        return (y, uk), y

    _, ys = lax.scan(step, (jnp.zeros(()), jnp.zeros(())), u)
    return ys


def response(f, tau, K=1.0, form="zoh"):
    """Frequency response at f Hz: the continuous K / (1 + j 2 pi f tau), or a recursion on the unit circle."""
    if form == "cont":
        return K / (1 + 2j * np.pi * f * tau)
    z0, zi = np.exp(-Te / tau), np.exp(-2j * np.pi * f * Te)
    return K * (1 - z0) * (zi if form == "zoh" else 1) / (1 - z0 * zi)


def histogramme(v, nb=16):
    """CalculeHistogramme() of the repository: 16 equal bins between the minimum and the maximum, index truncated."""
    mn, mx = v.min(), v.max()
    w = (mx - mn) / nb
    h = np.zeros(nb, int)
    for x in v:
        i = int((x - mn) / w)
        h[min(i, nb - 1)] += i >= 0
    return h


rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
deg = lambda h: float(np.degrees(np.angle(h)))

# ---------------------------------------------------------------------------------------------------------------
# The clean sine: a least-squares fit of a sin + b cos + c, with the frequency chosen on a 0.001 Hz grid.
best = None
for f in np.round(np.linspace(0.95, 1.05, 101), 3):
    A = np.c_[np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones(N)]
    c = np.linalg.lstsq(A, u_np, rcond=None)[0]
    r = u_np - A @ c
    if best is None or r @ r < best[0]:
        best = (r @ r, f, c, A @ c)
_, F0, coef, clean_exact = best
amp, offset = float(np.hypot(coef[0], coef[1])), float(coef[2])
clean = np.round(clean_exact, 4)                     # the page's copy, 4 decimals like the recording
raw = rmse(u_np, clean)
resid_sd = float(np.std(u_np - clean_exact))

loss = lambda lt, current: jnp.sqrt(jnp.mean((filtrage(u, lt, current=current) - jnp.asarray(clean)) ** 2))
sweep_tau = np.logspace(-2.5, 0.5, 61)
sweep = {cur: np.asarray(jax.vmap(loss, in_axes=(0, None))(jnp.log(sweep_tau), cur)) for cur in (False, True)}
fine_tau = np.logspace(-2.5, 0.5, 3001)
fine = {cur: np.asarray(jax.vmap(loss, in_axes=(0, None))(jnp.log(fine_tau), cur)) for cur in (False, True)}


def argmin_tau(current):
    """Newton's method on log tau with jax.grad, from the best point of the fine grid."""
    g = jax.grad(loss)
    h = jax.grad(g)
    lt = jnp.log(fine_tau[np.argmin(fine[current])])
    for _ in range(20):
        lt = lt - g(lt, current) / h(lt, current)
    return float(jnp.exp(lt)), float(loss(lt, current)), float(g(lt, current))


OPT = {cur: argmin_tau(cur) for cur in (False, True)}


# ---------------------------------------------------------------------------------------------------------------
# The C programs, compiled and run in temporary folders.
def run_c(source_text, name, args=()):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / f"{name}.c").write_text(source_text, encoding="utf-8")
        shutil.copy(REPO / "enregistrement-bruit.txt", d)
        for extra in ("TD3.c",):
            shutil.copy(REPO / extra, d / f"_{extra}")
        exe = d / (name + ".exe")
        cc = subprocess.run([GCC, "-Wall", "-Wextra", "-o", str(exe), f"{name}.c", "-lm"], cwd=d, capture_output=True, text=True)
        if cc.returncode:
            raise RuntimeError(cc.stderr)
        run = subprocess.run([str(exe), *args], cwd=d, capture_output=True)
        sortie = (d / "sortie.txt").read_text() if (d / "sortie.txt").exists() else ""
        return cc.stderr.strip(), run.returncode, run.stdout.decode("utf-8", "replace").replace("\r\n", "\n"), sortie


C_TAUS = ["2", "0.2", "0.05", "0.02"]
HARNESS = r'''#include <stdlib.h>
#define main td3_main
#include "_TD3.c"
#undef main
int main(int argc, char **argv) {           /* filtrage() of TD3.c on the recording, K = 1, one tau per argument */
    static double u[MAX_VALEURS], y[MAX_VALEURS];
    int n = LireEntree("enregistrement-bruit.txt", u, MAX_VALEURS);
    for (int a = 1; a < argc; a++) {
        filtrage(u, n, y, n, 1.0, atof(argv[a]), 0.01);
        for (int i = 0; i < n; i++) printf("%.8f\n", y[i]);
    }
    return 0;
}
'''

if GCC:
    original = (REPO / "TD3_original.c").read_text(encoding="utf-8")
    fixed12 = original
    for a, b in (("filtrage(u, taille_bruit, sortie", "filtrage(bruit, taille_bruit, sortie"),
                 ("for (int i = 0; i < 500; i++) {\n        difference[i]", "for (int i = 0; i < taille_bruit; i++) {\n        difference[i]"),
                 ("CalculeHistogramme(difference, 500,", "CalculeHistogramme(difference, taille_bruit,")):
        assert fixed12.count(a) == 1, a
        fixed12 = fixed12.replace(a, b)
    W_O, RC_O, OUT_O, SORTIE_O = run_c(original, "td3_original")
    W_F, RC_F, OUT_F, SORTIE_F = run_c(fixed12, "td3_bugs12")
    W_T, RC_T, OUT_T, SORTIE_T = run_c((REPO / "TD3.c").read_text(encoding="utf-8"), "td3")
    _, _, OUT_H, _ = run_c(HARNESS, "harness", C_TAUS)
    cvals = np.array([float(v) for v in OUT_H.split()]).reshape(len(C_TAUS), N)
    C_OUT = dict(zip(C_TAUS, cvals))

# ---------------------------------------------------------------------------------------------------------------
if DATA:
    def num(v, d=None):
        v = float(v) if d is None else round(float(v), d)
        s = repr(v)
        return s[:-2] if s.endswith(".0") else s

    vec = lambda a, d=None: "[" + ",".join(num(v, d) for v in a) + "]"
    REAL_STDOUT.write(
        "/* Kalman filters with JAX, chapter 05 data: the First-Order-Filter-using-C repository's recording (enregistrement-bruit.txt,\n"
        "   250 samples at Te = 0.01 s of a 1 Hz sine with noise), the least-squares sine fitted to it, and the outputs of the repository's\n"
        "   filtrage() compiled with gcc for several time constants, which the page checks its own filter against as it loads.\n"
        "   Printed by code/ch05_numbers.py data. */\n"
        f'window.KC = {{"Te":{num(Te)},"input":{vec(u_np, 4)},"clean":{vec(clean, 4)},'
        f'"fitted":{{"f":{num(F0, 3)},"amp":{num(amp, 4)},"offset":{num(offset, 4)},"resid_sd":{num(resid_sd, 4)}}},'
        '"c":{' + ",".join(f'"{k}":{vec(C_OUT[k])}' for k in C_TAUS) + "},"
        f'"sweep":{{"tau":{vec(sweep_tau, 5)},"err":{vec(sweep[False], 5)}}}}};\n')
    sys.exit()

out("jax", jax.__version__, "numpy", np.__version__, "gcc", subprocess.run([GCC, "-dumpfullversion"], capture_output=True, text=True).stdout.strip() if GCC else "not found")
out("repository folder:", REPO)

section("1. The first-order filter (Part 1)")
z0 = np.exp(-Te / 2)
out(f"tau = 2 s: z0 = e^-0.005 = {z0:.5f}, K(1 - z0) = {1 - z0:.5f}, memory 1/(1 - z0) = {1 / (1 - z0):.0f} samples; "
    f"cutoff 1/(4 pi) = {1 / (4 * np.pi):.3f} Hz")
out(f"the recording's {F0:.3f} Hz is {F0 * 4 * np.pi:.1f} times the cutoff; gain there {abs(response(F0, 2)):.4f} "
    f"(continuous {abs(response(F0, 2, form='cont')):.4f})")
out(f"forward Euler 1 - Te/tau = {1 - Te / 2} against e^-0.005 = {z0:.7f}: they differ by {z0 - (1 - Te / 2):.2e}, in the fifth decimal")
out(f"the u_k form's pole is z0 too (gain identical, |H_zoh| = |H_uk|); backward Euler's pole tau/(tau + Te) at tau = 0.02 is "
    f"{0.02 / 0.03:.3f}, against e^-0.5 = {np.exp(-0.5):.3f}; the bilinear transform's is (2 tau - Te)/(2 tau + Te) = {0.03 / 0.05:.3f}")

section("2. What the programs print (Part 4)")
idx = [k - 1 for k in range(1, N)]
oob = [i for i in idx if i >= 51]
out(f"bug 1: filtrage(u, 250, ...) reads u[0] to u[{max(idx)}] of a 51-cell array: u[{oob[0]}] to u[{oob[-1]}], {len(oob)} cells past the end")
imp = np.asarray(filtrage(jnp.asarray(np.r_[1.0, np.zeros(50)]), np.log(2.0)))
out(f"the self-test impulse: first output 1 - z0 = {imp[1]:.7f}; printed with %.2f: {sorted(set(f'{v:.2f}' for v in imp))}")
if GCC:
    rows = re.findall(r"^(\d\.\d\d)\t\t(-?\d\.\d\d)\t\t\t(-?\d\.\d\d)$", OUT_O, re.M)
    lines_o = SORTIE_O.split()
    bins_o = [int(v) for v in re.findall(r"^Bin \d+ : (\d+)$", OUT_O, re.M)]
    out(f"original: gcc -Wall -Wextra warnings: {W_O or 'none'}; exit code {RC_O}")
    out(f"  self-test table: {len(rows)} rows, y column {sorted(set(r[2] for r in rows))}")
    out("  " + next(l for l in OUT_O.splitlines() if l.startswith("Traitement")))
    out(f"  sortie.txt: {len(lines_o)} lines, values {sorted(set(lines_o))}")
    out(f"  histogram over 500 entries (undefined behaviour: entries 250 to 499 were never written, so the counts depend on "
        f"compiler and stack): {', '.join(map(str, bins_o))}")
    # If the cells read past u held zeros, bug 1's output is the impulse response alone; if the 250 cells never
    # written held zeros too, the routine on (recording - impulse response) followed by 250 zeros gives the program's counts.
    imp250 = np.asarray(filtrage(jnp.asarray(np.r_[1.0, np.zeros(N - 1)]), np.log(2.0)))
    diff500 = np.r_[u_np - imp250, np.zeros(N)]
    guess = histogramme(diff500)
    zero_bin = int((0 - diff500.min()) / ((diff500.max() - diff500.min()) / 16))
    out(f"  the routine on (recording - impulse response) followed by 250 zeros: {', '.join(map(str, guess))} -> "
        + (f"the same: in this run the cells read past u held zeros, and so did the 250 cells never written, which all "
           f"went to bin {zero_bin}, the one that holds zero" if list(guess) == bins_o else
           "different: in this build the cells read past u, or the 250 cells never written, did not all hold zeros"))
    vals_f = np.array([float(v) for v in SORTIE_F.split()])
    bins_f = [int(v) for v in re.findall(r"^Bin \d+ : (\d+)$", OUT_F, re.M)]
    y2 = np.asarray(filtrage(u, np.log(2.0)))
    out(f"bugs 1 and 2 fixed: warnings {W_F or 'none'}; sortie.txt {len(vals_f)} lines from {vals_f.min():.2f} to {vals_f.max():.2f} "
        f"(unrounded {y2.min():.4f} to {y2.max():.4f})")
    out(f"  histogram: {', '.join(map(str, bins_f))}; the page's routine on u - y: "
        f"{', '.join(map(str, histogramme(u_np - y2)))}")
    w = 2 * np.pi * F0 * 2
    out(f"  at tau = 2 s the steady output is a sine of amplitude A/sqrt(1 + w^2 tau^2) = {amp / np.sqrt(1 + w * w):.3f}; starting at rest "
        f"adds a transient of amplitude A w tau/(1 + w^2 tau^2) = {amp * w / (1 + w * w):.3f}, decaying as e^(-t/2), "
        f"which is {np.exp(-(N - 1) * Te / 2):.2f} of its start at the record's end")
    out(f"TD3.c: warnings {W_T or 'none'}; exit code {RC_T}; its output, as the page shows it:")
    for line in OUT_T.splitlines():
        out("  | " + line)
    bins_t = [int(v) for v in re.findall(r"^Classe +\d+ : (\d+)$", OUT_T, re.M)]
    y02 = np.asarray(filtrage(u, np.log(0.02)))
    out(f"  the page's routine on u - y at tau = 0.02: {', '.join(map(str, histogramme(u_np - y02)))} "
        f"({'the same' if list(histogramme(u_np - y02)) == bins_t else 'different'})")
    out(f"  sortie.txt: {len(SORTIE_T.split())} lines, %.4f; max |TD3.c - scan| = "
        f"{np.abs(np.array([float(v) for v in SORTIE_T.split()]) - y02).max():.1e} (the 4-decimal rounding)")
    out("original program's output, abbreviated as the page shows it:")
    lo = OUT_O.splitlines()
    keep = lo[:4] + ["…"] + [l for l in lo if l.startswith("0.50")] + [next(l for l in lo if l.startswith("Valeurs lues"))] \
        + lo[lo.index(next(l for l in lo if l.startswith("Valeurs lues"))) + 1:][:2] + ["…"] + lo[[i for i, l in enumerate(lo) if l.startswith("Traitement")][0]:]
    for line in keep:
        out("  | " + line)
    for k in C_TAUS:
        out(f"harness, filtrage() of TD3.c at tau = {k}: max |C (8 decimals) - scan| = {np.abs(C_OUT[k] - np.asarray(filtrage(u, np.log(float(k))))).max():.1e}")
else:
    out("gcc not found: the programs were not compiled.")

section("3. What the filter does to the signal (Part 5, the hero and the widgets)")
out(f"fitted sine: {F0:.3f} Hz, amplitude {amp:.4f}, offset {offset:.4f}; noise sd {resid_sd:.4f}; RMSE of the recording "
    f"to the 4-decimal clean sine {raw:.5f}")
out("tau      gain   lag(ZOH)  lag(cont)  extra   lead(u_k)  delay        RMSE u_{k-1}  RMSE u_k   sd(u - y)")
for tau in (2, 0.2, 0.05, 0.02, 0.008):
    hz, hc, hk = response(F0, tau), response(F0, tau, form="cont"), response(F0, tau, form="uk")
    e1 = float(loss(np.log(tau), False))
    e2 = float(loss(np.log(tau), True))
    yd = np.asarray(filtrage(u, np.log(tau)))
    out(f"{tau:<7}  {abs(hz):.3f}  {-deg(hz):6.2f}°   {-deg(hc):6.2f}°   {deg(hz) - deg(hc):+.2f}°  {deg(hk) - deg(hc):+.2f}°    "
        f"{-deg(hz) / 360 / F0 * 1000:5.1f} ms = {-deg(hz) / 360 / F0 / Te:4.2f} samples  {e1:.4f}       {e2:.4f}     {np.std(u_np - yd):.4f}")
out(f"360° f Te = {360 * F0 * Te:.2f}°, the gap between the two discrete forms; the sine moves by up to 2 pi f Te = "
    f"{2 * np.pi * F0 * Te:.1%} of its amplitude per sample and crosses its range (half a period) in {0.5 / F0 / Te:.1f} samples")
for cur, name in ((False, "the program's form, u_{k-1}"), (True, "fed the current sample, u_k")):
    tau_s, e, g = OPT[cur]
    out(f"{name}: minimum RMSE {e:.5f} at tau = {tau_s:.5f} s ({tau_s / Te:.2f} samples, z0 = {np.exp(-Te / tau_s):.3f}), "
        f"{1 - e / raw:.1%} below the recording's {raw:.5f} (dRMSE/dlog tau there {g:.1e})")
noise = u_np - clean_exact
for cur, name in ((False, "u_{k-1}"), (True, "u_k")):
    tau_s = OPT[cur][0]
    z = np.exp(-Te / tau_s)
    yn = np.asarray(filtrage(jnp.asarray(noise), np.log(tau_s), current=cur))
    ys_ = np.asarray(filtrage(jnp.asarray(clean_exact), np.log(tau_s), current=cur))
    out(f"at the {name} form's best tau ({tau_s:.4f} s): white noise is scaled by sqrt((1 - z0)/(1 + z0)) = {np.sqrt((1 - z) / (1 + z)):.3f}; "
        f"the recording's noise comes out with sd {np.std(yn[5:]):.4f} against {np.std(noise):.4f}; error on the sine alone "
        f"{rmse(ys_, clean_exact):.4f}")
out(f"the 61-point sweep of the data file: u_{{k-1}} from {sweep[False].min():.5f} to {sweep[False].max():.5f}; "
    f"u_k from {sweep[True].min():.5f}")
out(f"Bode axis: cutoff at tau = 2 s is {1 / (4 * np.pi):.4f} Hz, at the slider's ends (tau = {10 ** 0.5:.2f} and {10 ** -2.5:.4f} s) "
    f"{1 / (2 * np.pi * 10 ** 0.5):.4f} and {1 / (2 * np.pi * 10 ** -2.5):.1f} Hz; K = 2 gives {20 * np.log10(2):.1f} dB")

section("4. Back to Kalman (Part 6)")
for tau in (2, 0.02, OPT[True][0]):
    Kinf = 1 - np.exp(-Te / tau)
    out(f"tau = {tau:.4g} s: K_inf = 1 - z0 = {Kinf:.4f}, implied q/r = K^2/(1 - K) = {Kinf ** 2 / (1 - Kinf):.3g}, "
        f"sqrt(r/q) = {np.sqrt((1 - Kinf) / Kinf ** 2):.1f} samples")
q, r, A4 = 0.05, 0.25, 0.9
p = 1.0
for _ in range(1000):
    p = A4 * A4 * p * r / (p + r) + q
K4 = p / (p + r)
out(f"chapter 04 (A = 0.9, Q = 0.05, R = 0.25): steady gain K = {K4:.3f}; (1 - K) alone gives a time constant of "
    f"{-1 / np.log(1 - K4):.1f} steps (1/K = {1 / K4:.1f}); the linear part's pole (1 - K)A = {(1 - K4) * A4:.3f}, "
    f"{-1 / np.log((1 - K4) * A4):.1f} steps")
# The random walk's prediction is the program's recursion: check it on the recording.
Kinf = 1 - np.exp(-Te / 0.02)
xu, xp, upd, pred = 0.0, 0.0, [], []
for z in u_np:
    pred.append(xp)                     # x_{k|k-1}
    xu = xp + Kinf * (z - xp)           # update
    upd.append(xu)
    xp = xu                             # random walk: x_{k+1|k} = x_{k|k}
out(f"scalar random-walk Kalman filter at K = {Kinf:.4f}, from x = 0: max |prediction - filtrage()| = "
    f"{np.abs(np.array(pred) - np.asarray(filtrage(u, np.log(0.02)))).max():.1e}, "
    f"max |update - u_k form| = {np.abs(np.array(upd) - np.asarray(filtrage(u, np.log(0.02), current=True))).max():.1e}")

section("5. Exercises")
tau = 0.05
step = jnp.asarray(np.r_[np.zeros(20), np.ones(80)])
ys = np.asarray(filtrage(step, np.log(tau)))
out(f"exercise 2, tau = 0.05 s, step at sample 20: first >= 0.632 at {np.argmax(ys >= 0.632) - 20} samples after the step "
    f"(value {ys[np.argmax(ys >= 0.632)]:.5f} = 1 - e^-1), first >= 0.993 at {np.argmax(ys >= 0.993) - 20}; "
    f"max |y - (1 - e^(-t/tau))| at the samples = {np.abs(ys - np.where(t[:100] >= 0.2, 1 - np.exp(-(t[:100] - 0.2) / tau), 0)).max():.1e}")
row = lambda cur: ", ".join(f"{float(loss(np.log(tt), cur)):.3f}" for tt in (2, 0.2, 0.05, 0.02))
out(f"exercise 3, RMSE at tau = 2, 0.2, 0.05, 0.02: u_{{k-1}} {row(False)}; u_k {row(True)}; minima "
    f"{OPT[False][1]:.3f} at {OPT[False][0]:.4f} s and {OPT[True][1]:.3f} at {OPT[True][0]:.4f} s")
