/* Kalman filters with JAX, chapter J1 data: results computed with JAX 0.12.0.1 and NumPy 2.5 on a CPU by
   code/j1_numbers.py. For N = 10 ... 10^7: the float32 value 0.1 added N times by a sequential loop
   (lax.fori_loop; NumPy's np.add.accumulate gives the same), by Kahan's compensated loop (lax.fori_loop),
   by jnp.sum and np.sum on an array of N copies, and the float64 sequential sum. The page recomputes the
   sequential and Kahan sums itself and checks them against these values. */
window.J1 = {
  "jax": "0.12.0.1",
  "sums": [
    { "n": 10,       "seq": 1.0000001192092896, "kahan": 1.0,       "jsum": 1.0000001192092896, "npsum": 1.0,                "f64": 0.9999999999999999 },
    { "n": 100,      "seq": 10.000001907348633, "kahan": 10.0,      "jsum": 9.999999046325684,  "npsum": 10.00000286102295,  "f64": 9.99999999999998 },
    { "n": 1000,     "seq": 99.9990463256836,   "kahan": 100.0,     "jsum": 99.99996948242188,  "npsum": 100.00001525878906, "f64": 99.9999999999986 },
    { "n": 10000,    "seq": 999.9028930664062,  "kahan": 1000.0,    "jsum": 1000.0,             "npsum": 1000.0001220703125, "f64": 1000.0000000001588 },
    { "n": 100000,   "seq": 9998.556640625,     "kahan": 10000.0,   "jsum": 10000.0,            "npsum": 10000.0009765625,   "f64": 10000.000000018848 },
    { "n": 1000000,  "seq": 100958.34375,       "kahan": 100000.0,  "jsum": 100000.0,           "npsum": 100000.0078125,     "f64": 100000.00000133288 },
    { "n": 10000000, "seq": 1087937.0,          "kahan": 1000000.0, "jsum": 1000000.0625,       "npsum": 1000000.125,        "f64": 999999.9998389754 }
  ]
};
