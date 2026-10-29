/* Kalman filters with JAX, chapter J2 data, written by code/j2_numbers.py with JAX 0.12.0.1 on a CPU: the jaxprs of the demo's functions, each equation tagged with the source line that produced it, the cache experiment, and the timings. */
window.J2 = {
 "jax": "0.12.0.1",
 "hero": {
  "predict": {
   "source": "def linear_kf_predict(x, P, F, Q):\n    \"\"\"Pure function for the linear Kalman-filter prediction step.\"\"\"\n    x_pred = F @ x\n    P_pred = F @ P @ F.T + Q\n    return x_pred, P_pred",
   "header": "{ lambda ; a:f32[2] b:f32[2,2] c:f32[2,2] d:f32[2,2]. let",
   "eqns": [
    {
     "line": 2,
     "text": "e:f32[2] = dot_general[\n  dimension_numbers=(([1], [0]), ([], []))\n  preferred_element_type=float32\n] c a"
    },
    {
     "line": 3,
     "text": "f:f32[2,2] = dot_general[\n  dimension_numbers=(([1], [0]), ([], []))\n  preferred_element_type=float32\n] c b"
    },
    {
     "line": 3,
     "text": "g:f32[2,2] = transpose[permutation=(1, 0)] c"
    },
    {
     "line": 3,
     "text": "h:f32[2,2] = dot_general[\n  dimension_numbers=(([1], [0]), ([], []))\n  preferred_element_type=float32\n] f g"
    },
    {
     "line": 3,
     "text": "i:f32[2,2] = add h d"
    }
   ],
   "footer": "  in (e, i) }"
  },
  "prints": {
   "source": "def fn_prints(x):\n    print(\"x is\", x)\n    y = 2.0 * x + 1.0\n    return y",
   "header": "{ lambda ; a:f32[2]. let",
   "eqns": [
    {
     "line": 2,
     "text": "b:f32[2] = mul 2.0:f32[] a"
    },
    {
     "line": 2,
     "text": "c:f32[2] = add b 1.0:f32[]"
    }
   ],
   "footer": "  in (c,) }"
  },
  "prints_console": "x is JitTracer(float32[2])",
  "global": {
   "source": "def fn_global(x):\n    y = scale * x\n    return y",
   "header": "{ lambda ; a:f32[2]. let",
   "eqns": [
    {
     "line": 1,
     "text": "b:f32[2] = mul 2.0:f32[] a"
    }
   ],
   "footer": "  in (b,) }"
  },
  "where": {
   "source": "def fn_where(x):\n    y = jnp.where(x > 0, x, 0.0)\n    return y",
   "header": "{ lambda ; a:f32[2]. let",
   "eqns": [
    {
     "line": 1,
     "text": "b:bool[2] = gt a 0.0:f32[]"
    },
    {
     "line": 1,
     "text": "c:f32[2] = jit[\n  name=_where\n  jaxpr={ lambda ; b:bool[2] a:f32[2] d:f32[]. let\n      e:f32[] = convert_element_type[new_dtype=float32 weak_type=False] d\n      f:f32[2] = broadcast_in_dim e\n      c:f32[2] = select_n b f a\n    in (c,) }\n] b a 0.0:f32[]"
    }
   ],
   "footer": "  in (c,) }"
  },
  "if_partial": "b:bool[] = gt a 0.0:f32[]",
  "if_source": "def fn_if(x):\n    if x > 0:\n        return x\n    return 0.0 * x",
  "if_error": "TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].",
  "prints_tracer": "JitTracer(float32[2])"
 },
 "cache": [
  {
   "call": "jnp.array([1.0, 2.0])",
   "traced": true,
   "type": "float32[2]"
  },
  {
   "call": "jnp.array([5.0, 6.0])",
   "traced": false,
   "type": null
  },
  {
   "call": "np.array([1.0, 2.0])  (NumPy float64)",
   "traced": false,
   "type": null
  },
  {
   "call": "jnp.array([1.0, 2.0, 3.0])",
   "traced": true,
   "type": "float32[3]"
  },
  {
   "call": "jnp.array([1, 2])",
   "traced": true,
   "type": "int32[2]"
  },
  {
   "call": "1.0  (a Python float)",
   "traced": true,
   "type": "~float32[]"
  },
  {
   "call": "7.0  (another Python float)",
   "traced": false,
   "type": null
  },
  {
   "call": "jnp.float32(1.0)",
   "traced": true,
   "type": "float32[]"
  }
 ],
 "widget": {
  "chapter02": [
   "float32[2]",
   "float32[2,2]",
   "float32[1]"
  ],
  "newvalues": null,
  "numpy64": null,
  "state4": [
   "float32[4]",
   "float32[4,4]",
   "float32[1]"
  ],
  "intz": [
   "float32[2]",
   "float32[2,2]",
   "int32[1]"
  ],
  "chapter02again": null
 },
 "timing": {
  "op_us": 12.8,
  "np_op_us": 0.37,
  "step_ops": 31,
  "eager_step_us": 421,
  "jit_step_us": 12.6,
  "stages_2": [
   22.842,
   22.791,
   66.335,
   12.84
  ],
  "stages_4": [
   21.136,
   16.315,
   72.906,
   18.928
  ],
  "stages_int": [
   20.208,
   19.375,
   64.8,
   12.827
  ],
  "loop_eager_ms": 567.1,
  "loop_jit_ms": 13.2,
  "loop_numpy_ms": 24.3,
  "run_plain_ms": 94.1,
  "jrun_first_ms": 123.2,
  "jrun_ms": 1.172,
  "run_compiles": [
   1,
   1,
   1
  ],
  "jrun_compiles": [
   0,
   0,
   0
  ],
  "async_return_ms": 5.0,
  "async_done_ms": 161.0,
  "max_diff": 2.8014183044433594e-06
 }
};
