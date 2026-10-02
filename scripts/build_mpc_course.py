"""Build the "Advanced MPC and Learning MPC" course pages.

Reads the ten executed notebooks of the advanced_mpc_lmpc_lab repository
(advanced_mpc_lmpc_lab/notebooks, which ships with outputs) and writes static pages into
public/tutorials/mpc-lmpc-lab/:

    index.html                                   course landing page
    00-roadmap-and-math-primer.html ... 09-capstone-questions-and-extensions.html
    img/, notebooks/                             extracted figures, notebooks for download
    mpc-lmpc-lab-course.zip                      notebooks, the advanced_mpc_lmpc package, requirements
    cover.png                                    preview image for the tutorials card

Chapter pages, style.css and course.js come from scripts/notebook_course.py.

Usage:
    pip install markdown-it-py mdit-py-plugins numpy scipy matplotlib
    python scripts/build_mpc_course.py [path/to/advanced_mpc_lmpc_lab]
"""

from __future__ import annotations

import sys
from pathlib import Path

from notebook_course import SITE, Course, build_chapters, page_head, page_scripts, png_size, slug, write_zip

DEFAULT_REPO = SITE.parents[1] / "advanced_mpc_lmpc_lab"

# `result` lines quote the notebooks' own outputs; re-check them if the notebooks change.
CHAPTERS = [
    dict(stem="00_roadmap_and_math_primer", short="Roadmap and Mathematical Primer",
         build="States, inputs and discrete-time models, Schur stability, Lyapunov functions, the DARE, and a first constrained MPC on the double integrator.",
         result="The LQR terminal ingredients check out to machine precision (DARE identity residual 2.3e-15), and a constrained MPC brings the double integrator to the origin in 11 moves."),
    dict(stem="01_convex_optimization_and_cftoc", short="Convex Optimization and CFTOC",
         build="Convex sets and functions, KKT conditions and duality, then a constrained finite-time optimal-control problem condensed into a QP.",
         result="The KKT conditions hold to machine precision (primal-dual gap 1e-15), and the condensed CFTOC QP is solved with a largest constraint violation of 2.6e-13."),
    dict(stem="02_polytopes_reachability_invariance", short="Polytopes, Reachability and Invariant Sets",
         build="H- and V-polytopes, predecessor and reachable sets, and the fixed-point algorithms for positively invariant and control-invariant sets.",
         result="The maximal control-invariant set is found in 4 fixed-point iterations (10 vertices) and strictly contains the invariant set of the LQR closed loop (4 vertices)."),
    dict(stem="03_mpc_stability_and_terminal_sets", short="MPC Stability and Terminal Sets",
         build="The receding-horizon law, recursive feasibility by the shifted sequence, the Lyapunov argument with its three conditions, and LQR terminal ingredients.",
         result="With N = 5, an invariant terminal set gives a feasible region of area 34.5, against 16.1 with the terminal constraint x_N = 0 and 37.3 for the largest possible."),
    dict(stem="04_collision_avoidance_cftoc", short="Vehicle–Bike Collision Avoidance",
         build="A disjunctive yield-or-pass decision as two convex QPs or one mixed-integer problem, friction-limited braking, and feasibility maps over initial conditions.",
         result="From the nominal state only yielding is feasible; the critical friction coefficient is about 0.106, and halving the friction leaves 36 of 660 initial states with no safe strategy instead of 8."),
    dict(stem="05_learning_mpc_double_integrator", short="Learning MPC on a Repeated Task",
         build="Sampled and convex safe sets, a terminal cost learned from realized costs-to-go, and the guarantees of Learning MPC, iteration after iteration.",
         result="Starting from a seed trajectory 20.3 above the long-horizon optimum, Learning MPC closes the gap to 5e-5 in 9 iterations, and its cost never increases."),
    dict(stem="06_frenet_minimum_time_racing", short="Frenet Coordinates and Minimum-Time Racing",
         build="Curvilinear coordinates, a dynamic bicycle model in space rather than time, and a minimum-time speed planner with friction and acceleration limits.",
         result="The minimum-time lap of the 252 m track takes 17.09 s, with speed set by a 7.7 m-radius corner; dropping the friction coefficient from 1.0 to 0.6 costs 3.05 s."),
    dict(stem="07_data_driven_local_models_for_racing", short="Data-Driven Local Models for Racing",
         build="Kernel-weighted local regression of the vehicle dynamics from recorded laps, structured affine models and time-varying predictions for MPC.",
         result="Over a 14-step horizon, the local model's prediction error is about half the global model's (RMSE 0.021 vs 0.041), and a query far from the data is flagged on every lap."),
    dict(stem="08_dynamic_programming_to_rl", short="From Dynamic Programming to Reinforcement Learning",
         build="Bellman's principle, the Riccati recursion, contraction and value and policy iteration, where MPC and LMPC fit, and Q-learning on to DQN.",
         result="Policy iteration finds the exact optimal policy on a 215-state lattice in 4 improvement steps, out of about 10^150 possible policies."),
    dict(stem="09_capstone_questions_and_extensions", short="Capstone: Connect the Ideas",
         build="One MPC design followed end to end with numerical checks of every guarantee, Learning MPC from a poor first lap, and derivations with worked answers.",
         result="With LQR terminal ingredients, all 177 feasible starts of a 221-point grid stay feasible and converge; Learning MPC starting from a first lap 69% above optimal reaches the optimum."),
]
NOTE = "To run a chapter you also need the <code>advanced_mpc_lmpc</code> package: the full course download has it."
COURSE = Course(
    name="Advanced MPC and Learning MPC", folder="mpc-lmpc-lab", chapters=CHAPTERS,
    notes={ch["stem"]: NOTE for ch in CHAPTERS},
    license="MIT licensed.",
)


# ---------------------------------------------------------------- landing

def index_page(results: list[dict]) -> str:
    items = []
    for ch, r in zip(CHAPTERS, results):
        items.append(f"""    <li>
      <a href="{slug(ch)}.html">
        <span class="n">{ch['stem'][:2]}</span>
        <span class="body">
          <span class="t">{r['title']}</span>
          <span class="d">{ch['build']}</span>
          <span class="r"><span class="lbl">You find</span> {ch['result']}</span>
        </span>
      </a>
    </li>""")
    chapters = "\n".join(items)
    zip_name = f"{COURSE.folder}-course.zip"
    return page_head(
        "Advanced MPC and Learning MPC",
        "A ten-notebook course on constrained optimal control: convex optimization, invariant sets, MPC "
        "feasibility and stability, Learning MPC, autonomous-driving examples and the link to dynamic "
        "programming and reinforcement learning, with every guarantee checked numerically.",
    ) + f"""
<nav class="site-back col" aria-label="Site"><a href="/tutorials">&larr; All tutorials</a><a href="/">Elie Rouphael</a></nav>

<header class="hero col">
  <span class="eyebrow">Ten notebooks · NumPy, SciPy, Matplotlib</span>
  <h1>Advanced MPC and Learning MPC</h1>
  <p class="lede">Model predictive control solves an optimization problem at every sample. This course does not
  treat that optimizer as a black box: for every controller it asks what the decision variables, objective and
  constraints are, what makes the problem feasible at the next step, why the closed loop is stable, and which
  assumptions a real system would break. It goes from convex optimization and invariant sets to MPC stability,
  then to <strong>Learning MPC</strong>, which learns its terminal ingredients from its own successful
  executions, and to the connection with dynamic programming and reinforcement learning.</p>
  <p class="byline">Elie Rouphael · Assumes linear algebra, basic control (state-space models, stability) and Python</p>
  <div class="downloads">
    <a class="btn" href="{slug(CHAPTERS[0])}.html">Start with chapter 00</a>
    <a class="btn ghost" href="{zip_name}" download>Download the course (.zip)</a>
  </div>
</header>

<section class="col" aria-labelledby="chapters">
  <h2 id="chapters">Chapters</h2>
  <p class="section-lede">Chapters 00–03 build the theory: optimization, constraint geometry, and why a
  finite-horizon controller can be recursively feasible and stable. Chapters 04–07 apply it to collision
  avoidance, Learning MPC and racing. Chapter 08 places everything in the dynamic-programming picture, and
  chapter 09 runs one complete design end to end.</p>
  <ol class="chapter-list">
{chapters}
  </ol>
</section>

<section class="col prose" aria-labelledby="run">
  <h2 id="run">Running it yourself</h2>
  <p>Every chapter page shows the notebook with its outputs. To run and change the code,
  <a href="{zip_name}" download>download the course</a> (the notebooks, the small <code>advanced_mpc_lmpc</code>
  package they use, its tests and the requirements), then from its folder run:</p>
  <pre><code class="language-bash">pip install -r requirements.txt
python -m pytest            # optional: the package's tests
jupyter lab                 # then open notebooks/00_roadmap_and_math_primer.ipynb</code></pre>
  <p>No commercial solver is needed: the package uses SciPy, chosen for readability rather than speed. Each
  notebook runs in under a minute on a laptop.</p>
</section>

<section class="col prose" aria-labelledby="sources">
  <h2 id="sources">Acknowledgements</h2>
  <p>The text and code are original. The choice of topics and several examples were informed by a 2025–2026
  course on Learning MPC (Università di Napoli Federico II, Université Franco-Italienne and Grenoble INP / ENSE3);
  the course download includes the full acknowledgement and the references for each chapter.</p>
</section>

<footer class="col">
  <p>Advanced MPC and Learning MPC, by Elie Rouphael. MIT licensed. The chapter pages are rendered from the
  executed notebooks; math by KaTeX, code highlighting by highlight.js.</p>
</footer>
""" + page_scripts()


# ---------------------------------------------------------------- cover

def make_cover(path: Path, repo: Path) -> None:
    """Nested constraint, control-invariant and LQR-invariant sets with an MPC trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(repo / "src"))
    from advanced_mpc_lmpc.linear import dlqr
    from advanced_mpc_lmpc.mpc import LinearMPC, simulate_receding_horizon
    from advanced_mpc_lmpc.polytopes import (HPolytope, closed_loop_admissible_set_scalar_input,
                                             maximal_control_invariant, maximal_invariant)

    A = np.array([[1.0, 1.0], [0.0, 1.0]]); B = np.array([[0.5], [1.0]])
    Q = np.diag([1.0, 0.2]); R = np.array([[0.2]])
    X = HPolytope.box([-4.0, -3.0], [4.0, 3.0])
    K, P, _ = dlqr(A, B, Q, R)
    C_inf, _ = maximal_control_invariant(A, B, X, -1.0, 1.0)
    O_inf, _ = maximal_invariant(A - B @ K, closed_loop_admissible_set_scalar_input(X, K, -1.0, 1.0))
    mpc = LinearMPC(A, B, Q, R, 6, P=P, x_lower=[-4, -3], x_upper=[4, 3], u_lower=[-1], u_upper=[1])
    traj, _, _ = simulate_receding_horizon(mpc, [-3.6, 2.4], 25)

    ink, blue, orange = "#1d2530", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor("#fbfbf8"); ax.set_facecolor("#fbfbf8")
    for S, face, edge in ((X, "#ecebe6", "#b9bfb6"), (C_inf, "#d6e6f7", blue), (O_inf, "#f8dcc9", orange)):
        V = S.vertices_2d()
        ax.fill(V[:, 0], V[:, 1], color=face, ec=edge, lw=2)
    ax.plot(traj[:, 0], traj[:, 1], "-o", color=ink, ms=4, lw=1.6)
    ax.text(-4.3, 4.4, "Advanced MPC &\nLearning MPC", color=ink, fontsize=26, weight="bold", va="top")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-4.6, 4.6); ax.set_ylim(-3.4, 4.6)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.04)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_REPO
    nb_dir = repo / "notebooks"
    if not nb_dir.is_dir():
        sys.exit(f"Notebooks not found in {nb_dir}.")

    results = build_chapters(COURSE, nb_dir)
    out = COURSE.out
    (out / "index.html").write_text(index_page(results), encoding="utf-8")

    members = [(repo / rel, rel) for rel in ("README.md", "requirements.txt", "pyproject.toml", "LICENSE",
                                              "NOTICE.md", "CITATION.cff")]
    for folder in ("src", "tests", "docs", "scripts"):
        members += [(p, p.relative_to(repo).as_posix()) for p in sorted((repo / folder).rglob("*"))
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pdf"]
    members += [(nb_dir / f"{ch['stem']}.ipynb", f"notebooks/{ch['stem']}.ipynb") for ch in CHAPTERS]
    write_zip(COURSE, members)

    make_cover(out / "cover.png", repo)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
