"""Reproduce every number, dtype and message quoted on chapter J1 of "Kalman Filters with JAX"
(public/tutorials/kalman-jax/j1-arrays-and-the-numbers-inside-them.html).

Usage:  python j1_numbers.py          # JAX defaults (32-bit)
        python j1_numbers.py x64      # with jax_enable_x64, for the 64-bit lines
Checked with jax 0.12.0.1, numpy 2.x, CPU.
"""
from __future__ import annotations

import sys
import warnings
from decimal import Decimal, getcontext

X64 = len(sys.argv) > 1 and sys.argv[1] == "x64"
import jax  # noqa: E402

if X64:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

getcontext().prec = 80
print("jax", jax.__version__, "numpy", np.__version__, "x64" if X64 else "default (32-bit)")


def section(title):
    print(f"\n=== {title}")


def show(label, thunk):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = thunk()
            print(f"{label}: {out!r}")
        except Exception as e:  # noqa: BLE001
            print(f"{label}: raises {type(e).__name__}: {str(e).splitlines()[0]}")
        for w in caught:
            print(f"    warning {w.category.__name__}: {str(w.message).splitlines()[0]}")


section("dtypes")
show("jnp.array(1.0).dtype", lambda: jnp.array(1.0).dtype)
show("jnp.array(1).dtype", lambda: jnp.array(1).dtype)
show("np.array(1.0).dtype", lambda: np.array(1.0).dtype)
show("jnp.asarray(np.array([0.1])).dtype", lambda: jnp.asarray(np.array([0.1])).dtype)
show("jnp.array([0.1], dtype=jnp.float64).dtype", lambda: jnp.array([0.1], dtype=jnp.float64).dtype)
if X64:
    sys.exit()

section("arrays")
x = jnp.array([1.0, 2.0, 3.0])
show("x", lambda: x)
show("x.shape, x.dtype, x.ndim, x.size", lambda: (x.shape, x.dtype, x.ndim, x.size))
show("x.devices()", lambda: x.devices())
show("type(x)", lambda: type(x))


def assign():
    x[0] = 10.0


show("x[0] = 10.0", assign)
show("x.at[0].set(10.0)", lambda: x.at[0].set(10.0))
show("x after .at", lambda: x)
show("x.at[1].add(5.0)", lambda: x.at[1].add(5.0))
show("x.at[::2].multiply(2.0)", lambda: x.at[::2].multiply(2.0))
y = jnp.arange(5)
show("jnp.arange(5)[10]", lambda: y[10])
show("jnp.arange(5)[-1]", lambda: y[-1])
show("jnp.arange(5).at[10].set(99)", lambda: y.at[10].set(99))
show("np.arange(5)[10]", lambda: np.arange(5)[10])

section("broadcasting, as in kalman_jax/filters.py")
Wm, pts = jnp.ones(5), jnp.ones((5, 2))
show("(Wm[:, None] * pts).shape", lambda: (Wm[:, None] * pts).shape)
show("(Wm * pts).shape", lambda: (Wm * pts).shape)
xs, offsets = jnp.ones(2), jnp.ones((2, 2))
show("(xs[None, :] + offsets).shape", lambda: (xs[None, :] + offsets).shape)
show("jnp.concatenate([x[None,:], x[None,:]+o, x[None,:]-o], axis=0).shape",
     lambda: jnp.concatenate([xs[None, :], xs[None, :] + offsets, xs[None, :] - offsets], axis=0).shape)
Wc, terms = jnp.ones(5), jnp.ones((5, 2, 2))
show("(Wc[:, None, None] * terms).shape", lambda: (Wc[:, None, None] * terms).shape)
show("jnp.sum(Wc[:, None, None] * terms, axis=0).shape", lambda: jnp.sum(Wc[:, None, None] * terms, axis=0).shape)
show("(jnp.ones((3, 1)) + jnp.ones(4)).shape", lambda: (jnp.ones((3, 1)) + jnp.ones(4)).shape)
show("jnp.ones((2, 3)) @ jnp.ones((3, 4))", lambda: (jnp.ones((2, 3)) @ jnp.ones((3, 4))).shape)
show("jnp.outer(jnp.ones(2), jnp.ones(2)).shape", lambda: jnp.outer(jnp.ones(2), jnp.ones(2)).shape)

section("float32 and float64 facts")
f32, f64 = np.float32, np.float64
show("exact float32(0.1)", lambda: str(Decimal(float(f32(0.1)))))
show("exact float64(0.1)", lambda: str(Decimal(0.1)))
show("float32 bits of 0.1", lambda: format(np.array(0.1, f32).view(np.uint32).item(), "032b"))
show("float32: 0.1 + 0.2 == 0.3", lambda: bool(f32(0.1) + f32(0.2) == f32(0.3)))
show("float64: 0.1 + 0.2 == 0.3", lambda: 0.1 + 0.2 == 0.3)
show("float64: 0.1 + 0.2", lambda: 0.1 + 0.2)
show("jnp float32: 0.1 + 0.2 == 0.3", lambda: bool(jnp.float32(0.1) + jnp.float32(0.2) == jnp.float32(0.3)))
show("float32(16777217)", lambda: int(f32(16777217)))
show("float32(16777216) + 1", lambda: int(f32(16777216) + f32(1)))
show("float32 eps, max, tiny(normal), smallest subnormal",
     lambda: (float(np.finfo(f32).eps), float(np.finfo(f32).max), float(np.finfo(f32).tiny), float(np.finfo(f32).smallest_subnormal)))
show("float64 eps, max, tiny", lambda: (float(np.finfo(f64).eps), float(np.finfo(f64).max), float(np.finfo(f64).tiny)))
for v in [0.1, 1.0, 33.0, 1000.0, 250000.0, 1e6, 1.6e7, 3.3e7, 1e8, 3.0e38]:
    show(f"spacing float32 at {v:g}", lambda v=v: float(np.spacing(f32(v))))
for v in [1.0, 33.0, 1e6, 3.3e7]:
    show(f"spacing float64 at {v:g}", lambda v=v: float(np.spacing(f64(v))))
show("float32 1e39", lambda: f32(1e39))
show("jnp float32 overflow 3e38*10", lambda: jnp.float32(3e38) * 10)
show("jnp 0/0, sqrt(-1)", lambda: (jnp.float32(0) / jnp.float32(0), jnp.sqrt(jnp.float32(-1))))
show("jnp 1/0", lambda: jnp.float32(1) / jnp.float32(0))
show("nan == nan", lambda: bool(jnp.nan == jnp.nan))

section("weak types and integer overflow")
show("(jnp.ones(3, jnp.float16) + 1.0).dtype", lambda: (jnp.ones(3, jnp.float16) + 1.0).dtype)
show("(jnp.ones(3, jnp.float16) + jnp.ones(3, jnp.float32)).dtype", lambda: (jnp.ones(3, jnp.float16) + jnp.ones(3, jnp.float32)).dtype)
show("(jnp.ones(3, jnp.int32) * 2.5).dtype", lambda: (jnp.ones(3, jnp.int32) * 2.5).dtype)
show("jnp.int32(2**31 - 1) + 1", lambda: jnp.int32(2**31 - 1) + 1)
show("jnp.array(2**31)", lambda: jnp.array(2**31))
show("float16 1/3 exact", lambda: str(Decimal(float(np.float16(1 / 3)))))
show("float16 eps, max", lambda: (float(np.finfo(np.float16).eps), float(np.finfo(np.float16).max)))
show("bfloat16 eps", lambda: float(jnp.finfo(jnp.bfloat16).eps))

section("rounding in arithmetic, float32")
a, b, c = f32(1e8), f32(1.0), f32(-1e8)
show("(1e8 + 1) - 1e8", lambda: float((a + b) + c))
show("(1e8 - 1e8) + 1", lambda: float((a + c) + b))
show("(1 + 1e-7) - 1 in float32", lambda: float((f32(1) + f32(1e-7)) - f32(1)))
show("relative error of that", lambda: float(((f32(1) + f32(1e-7)) - f32(1)) / 1e-7 - 1))
show("float32 1e-7 exact", lambda: str(Decimal(float(f32(1e-7)))))

section("summing 0.1, N times")
from jax import lax  # noqa: E402


@jax.jit
def naive_loop(n):
    return lax.fori_loop(0, n, lambda i, s: s + jnp.float32(0.1), jnp.float32(0.0))


@jax.jit
def kahan_loop(n):
    def body(i, carry):
        s, comp = carry
        y = jnp.float32(0.1) - comp
        t = s + y
        comp = (t - s) - y
        return t, comp
    return lax.fori_loop(0, n, body, (jnp.float32(0.0), jnp.float32(0.0)))[0]


for n in [10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]:
    seq = float(np.add.accumulate(np.full(n, 0.1, f32))[-1])
    loop = float(naive_loop(n))
    kahan = float(kahan_loop(n))
    jsum = float(jnp.sum(jnp.full(n, 0.1, jnp.float32)))
    npsum = float(np.sum(np.full(n, 0.1, f32)))
    f64sum = float(np.add.accumulate(np.full(n, 0.1, f64))[-1])
    print(f"N={n:>9}: sequential {seq!r}  fori_loop {loop!r}  kahan {kahan!r}  jnp.sum {jsum!r}  np.sum {npsum!r}  float64 sequential {f64sum!r}")

section("exercises and the Kalman code")
show("(jnp.ones((4, 1, 3)) * jnp.ones((2, 1))).shape", lambda: (jnp.ones((4, 1, 3)) * jnp.ones((2, 1))).shape)
show("(jnp.ones(3) @ jnp.ones((3, 2))).shape", lambda: (jnp.ones(3) @ jnp.ones((3, 2))).shape)
show("(jnp.ones((2, 3)) + jnp.ones((3, 2))).shape", lambda: (jnp.ones((2, 3)) + jnp.ones((3, 2))).shape)
show("(jnp.ones((5, 1)) - jnp.ones((1, 5))).shape", lambda: (jnp.ones((5, 1)) - jnp.ones((1, 5))).shape)
show("float32(16777217) == 16777216", lambda: bool(f32(16777217) == f32(16777216)))
show("int(float64(2**53 + 1))", lambda: int(f64(2**53 + 1)))
show("spacing float32 at 999999 * 33", lambda: float(np.spacing(f32(999999 * 33.0))))
show("jnp.full(5, 1.0 / (2.0 * 2e-6)).dtype", lambda: jnp.full(5, 1.0 / (2.0 * 2e-6)).dtype)
show("float32(0.5) + float32(1e-9) == 0.5", lambda: bool(f32(0.5) + f32(1e-9) == f32(0.5)))
show("float32(0.03) + float32(1e-9) == 0.03", lambda: bool(f32(0.03) + f32(1e-9) == f32(0.03)))
lost = [v for v in np.linspace(0.001, 2.0, 200_000, dtype=f32) if f32(v) + f32(1e-9) == f32(v)]
show("smallest float32 in [0.001, 2] where adding 1e-9 changes nothing", lambda: float(min(lost)))
show("largest float32 in [0.001, 2] where adding 1e-9 changes something",
     lambda: float(max(v for v in np.linspace(0.001, 2.0, 200_000, dtype=f32) if f32(v) + f32(1e-9) != f32(v))))
show("jnp.eye(2, dtype=jnp.ones(2).dtype).dtype", lambda: jnp.eye(2, dtype=jnp.ones(2).dtype).dtype)
show("float32(1) + float32(1e-7) == 1", lambda: bool(f32(1.0) + f32(1e-7) == f32(1.0)))
show("float32(1) + float32(5e-8) == 1", lambda: bool(f32(1.0) + f32(5e-8) == f32(1.0)))
show("float32(0.027) + float32(1e-9) - 0.027 (the jitter at a settled variance)", lambda: float(f32(0.027) + f32(1e-9) - f32(0.027)))
