---
title: Notation and the canonical system
---

The course keeps one set of symbols from the first concept chapter to the capstone, and one small system on which every concept chapter computes its numbers. This page lists both. The card below is computed in your browser by the same routines the chapter demos use; the numbers script of each chapter prints the same values from the Python package.

## The canonical system {#system}

::: hero
<div class="demo" id="demo">
  <div class="stage">
    <div>
      <p class="chart-title">The cart's state box with the sets every chapter returns to</p>
      <canvas id="sysC" aria-label="The state box, the maximal control-invariant set, the LQR's invariant terminal set, and the LQR closed loop from one start" aria-describedby="sysStatus"></canvas>
    </div>
    <p class="status" id="sysStatus" role="status" aria-live="polite"></p>
  </div>
  <div class="panel">
    <div class="readouts" aria-live="off">
      <span>LQR gain <span class="m">K</span></span><span class="v" id="rK">–</span>
      <span>spectral radius <span class="m">\rho(A - BK)</span></span><span class="v" id="rRho">–</span>
      <span><span class="m">\mathcal X_f = \mathcal O_\infty</span></span><span class="v" id="rXf">–</span>
      <span><span class="m">\mathcal C_\infty</span></span><span class="v" id="rC">–</span>
      <span><span class="m">\mathcal X_K</span></span><span class="v" id="rXK">–</span>
    </div>
    <label class="field"><span>Input bound <span class="m">u_{\max}</span>: <b id="uV"></b></span>
      <span class="row"><input type="range" id="uSel" min="0" max="1" step="1" value="1"></span>
    </label>
    <p class="hint">The slider switches between the full motor <span class="m">|u| \le 1</span> and the weaker motor <span class="m">|u| \le 0.5</span> that chapters use when a horizon effect has to be visible.</p>
  </div>
</div>
:::

**Dynamics.** A cart of unit mass sampled every second under a held acceleration (chapter [[a01]]):

$$x^+ = Ax + Bu, \qquad A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \qquad B = \begin{pmatrix} 0.5 \\ 1 \end{pmatrix}, \qquad x = (p, v).$$

**Constraints.** The state box $\mathcal X = \{x : |p| \le 4,\ |v| \le 3\}$ and the input interval $\mathcal U = \{u : |u| \le 1\}$. The weaker motor is $|u| \le 0.5$.

**Cost.** The stage cost $q(x, u) = x^\top Q x + R u^2$ with $Q = \mathrm{diag}(1, 0.2)$ and $R = 0.2$; the terminal cost $V_f(x) = x^\top P x$ with $P$ from the Riccati equation (chapter [[a07]]).

::: table num
| Quantity | Value, full motor | Value, weaker motor | Where |
|---|---|---|---|
| LQR gain $K$, $u = -Kx$ | $(0.7396,\ 1.2604)$ | the same | [[a07]] |
| Riccati matrix $P$ | $\begin{pmatrix} 1.7042 & 0.5 \\ 0.5 & 0.7021 \end{pmatrix}$ | the same | [[a07]] |
| Eigenvalues of $A - BK$ | $0.1849 \pm 0.2743\,i$, $\rho = 0.3308$ | the same | [[a07]] |
| $\mathcal X_K$, where the LQR input is admissible | 6 vertices, area 12.66 | 4 vertices, area 6.35 | [[a10]] |
| $\mathcal X_f = \mathcal O_\infty$, the LQR's invariant set | 4 vertices, area 7.31, 2 iterations | 4 vertices, area 1.83 | [[a10]] |
| $\mathcal C_\infty$, the maximal control-invariant set | 10 vertices, area 39.25, 4 iterations | 14 vertices, area 30.29, 7 iterations | [[a10]] |
| Zero-terminal feasible region at $N = 5$ | area 31.75 | area 15.10 | [[a11]] |
| $\mathcal X_f$-terminal feasible region at $N = 5$ | area 39.25 (all of $\mathcal C_\infty$) | area 23.31 | [[a11]] |
:::

The vertices of $\mathcal X_f$ with the full motor are $(\pm 2.404, \mp 2.204)$ and $(\pm 2.204, \mp 0.5)$: a thin parallelogram, because the LQR law demands more than one unit of input almost everywhere else in the box.

## Symbols {#symbols}

| Symbol | Meaning | Defined in |
|---|---|---|
| $x_k \in \mathbb R^n$, $u_k \in \mathbb R^m$ | State and input at sample $k$; $x^+$ is the next state | [[a01]] |
| $A, B$; $A_k, B_k, c_k$ | LTI matrices; the matrices of a time-varying affine model | [[a01]], [[b04]] |
| $T_s$ | Sample period | [[a01]] |
| $\mathcal X, \mathcal U$ | State and input constraint sets, $\{x : H_x x \le h_x\}$ and $\{u : H_u u \le h_u\}$ | [[a01]], [[a08]] |
| $N$ | Prediction horizon | [[a02]] |
| $X, U$; $\mathcal S_x, \mathcal S_u$ | Stacked predicted states and inputs; the prediction matrices, $X = \mathcal S_x x_0 + \mathcal S_u U$ | [[a02]] |
| $H, f$ | Hessian and linear term of the condensed QP, $\tfrac12 U^\top H U + f^\top U$ | [[a02]] |
| $\lambda, \nu$ | Multipliers of the inequalities and equalities; $\lambda_i$ is the price of constraint $i$ | [[a03]] |
| $q(x, u)$ | Stage cost, $x^\top Q x + u^\top R u$ in the concept chapters | [[a05]] |
| $V_f(x)$, $\mathcal X_f$ | Terminal cost and terminal set | [[a05]], [[a11]] |
| $J_N^\star(x)$ | Optimal value of the horizon-$N$ problem at $x$, constants included | [[a05]], [[a11]] |
| $\kappa_N(x) = u_0^\star(x)$ | The receding-horizon law: the first optimised input | [[a06]] |
| $x_{k\mid t}, u_{k\mid t}$ | Prediction for time $k$ made at time $t$ | [[a06]] |
| $\mathcal X_N$ | The feasible set of the horizon-$N$ problem; the region of attraction when the theorem applies | [[a06]], [[a11]] |
| $K, P$ | LQR gain with $u = -Kx$, and the Riccati matrix; $A_{\mathrm{cl}} = A - BK$ | [[a07]] |
| $\rho(A)$ | Spectral radius; Schur means $\rho < 1$ | [[a07]] |
| $\mathrm{Pre}(\mathcal S)$, $\mathrm{Post}(\mathcal S)$ | One-step controlled predecessor and forward reachable set | [[a09]] |
| $\mathcal K_N$, $N_{\min}(x)$ | $N$-step controllable set to a target; the fewest steps to reach it | [[a09]] |
| $\mathcal O_\infty$, $\mathcal C_\infty$, $\mathcal X_K$ | Maximal positively invariant set of a fixed law; maximal control-invariant set; states where $-Kx$ is admissible | [[a10]] |
| $SS^j$, $CS^j$ | Sampled safe set after iteration $j$; its convex hull | [[a12]] |
| $\lambda_r$, $S$ | Barycentric weights; the stored safe states as the rows of $S$, $x = S^\top \lambda$ | [[a12]] |
| $J_t^i$, $V_c^j(x)$ | Realised cost-to-go of iteration $i$ from time $t$; the convex terminal value | [[a13]] |
| $J_k^\star$, $T$, $T_\mu$ | Optimal cost-to-go from stage $k$; the Bellman operator and its policy version | [[a14]] |
| $\gamma$ | Discount factor | [[a14]] |
| $Q^\star(x, u)$ | The state-input cost-to-go (the Q-function of reinforcement learning) | [[b05]] |
| $s \in \{0, 1\}$, $M$ | A binary decision variable; the Big-M constant | [[a15]] |
| $(s, e_y, e_\psi)$, $K(s)$ | Frenet progress, lateral error, heading error; the centreline curvature | [[b03]] |

## Conventions {#conventions}

- **Feedback sign.** Linear feedback is $u = -Kx$, so the closed loop is $A - BK$. A text that writes $u = Fx$ has $F = -K$ and the loop $A + BF$. The wrong sign on the course's gain gives eigenvalues outside the unit circle (chapter [[a07]]).
- **Discrete-time stability** is $\rho(A_{\mathrm{cl}}) < 1$, not negative real parts: that is the continuous-time condition.
- **Vectors and matrices.** $x^\top y$ is the inner product, $\lVert x \rVert$ the Euclidean norm, $P \succ 0$ positive definite and $P \succeq 0$ positive semidefinite. Inequalities between vectors hold componentwise.
- **The full value.** $J_N^\star$ always includes the constant $x_0^\top Q x_0$ and the terms a solver drops; the Lyapunov checks of chapter [[a11]] need it.
- **Rows.** Stored safe states are the rows of $S$, so a hull point is $S^\top \lambda$ and the code's `S.T @ lam` reads the same way.
- **Time indices.** $k$ counts prediction steps inside one optimisation, $t$ counts real closed-loop time, $j$ counts iterations of a repeated task.
