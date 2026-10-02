"""Build the "Learning to Identify New Dynamical Systems from a Few Experiments" course pages.

Reads the nine executed notebooks of the course
(few-shot-dynamical-systems/course/executed) and writes static pages into
public/tutorials/few-shot-dynamical-systems/:

    index.html                                     course landing page
    01-the-system-and-what-we-can-measure.html ...
    09-summary-and-the-road-to-a-foundation-model.html
    img/, notebooks/                               extracted figures, notebooks for download
    few-shot-dynamical-systems-course.zip          notebooks, dynutils.py and requirements
    cover.png                                      preview image for the tutorials card

Chapter pages, style.css and course.js come from scripts/notebook_course.py. The
notebooks must already be executed (run build_notebooks.py in the course folder).

Unlike "A Tour of Machine Learning", this course is one sequential narrative: chapters 03-08
all rebuild the same pretrained "Family" from a fresh seed at their top (see
course/dynutils.py's build_family), which keeps every chapter independently runnable without a
shared results/ scoreboard.

Usage:
    pip install markdown-it-py mdit-py-plugins numpy matplotlib scipy
    python scripts/build_few_shot_course.py [path/to/course]
"""

from __future__ import annotations

import sys
from pathlib import Path

from notebook_course import SITE, Course, build_chapters, page_head, page_scripts, png_size, slug, write_zip

DEFAULT_COURSE = SITE.parents[1] / "few-shot-dynamical-systems" / "course"

# `result` lines quote the notebooks' own outputs; re-check them if the notebooks change.
CHAPTERS = [
    dict(stem="01_the_system_and_what_we_can_measure", short="The System, and What We Can Measure",
         build="A family of nonlinear oscillators sharing one equation but not its four coefficients, a hidden two-dimensional design space behind them, and a learner that only ever chooses an input and reads a noisy position sensor.",
         result="The gap between a system's true noise-free response and a repeat run with its coloured disturbance alone has RMSE 0.0165, already larger than the sensor noise (0.01) — most of the mismatch the learner sees is unpredictable, not measurement error."),
    dict(stem="02_identifying_one_system_from_scratch", short="Identifying One System From Scratch",
         build="Output-error least squares fit to one experiment, and what 30 repeated fits reveal about identifiability as the experiment grows from 4 to 40 seconds.",
         result="With 40 seconds of data, k, d and b settle within 6-9% of their true values, but the hardening term α still carries a 100% median relative error, no better than with only 4 seconds."),
    dict(stem="03_what_many_systems_have_in_common", short="What Many Systems Have in Common",
         build="Twenty long recordings, each identified on its own, then noise-whitened PCA to find the low-dimensional geometry the family shares: the pretrained backbone every later chapter reuses.",
         result="Two directions explain 97.8% of the noise-weighted variation across the 20 systems, and noise-whitened PCA recovers the second hidden design variable with R²=0.96, against R²=0.22 for plain PCA."),
    dict(stem="04_few_shot_identification_of_a_new_system", short="Few-Shot Identification of a New System",
         build="Freezing the pretrained backbone and estimating only a 2-number context by MAP, with a Laplace approximation for its uncertainty, from as little as two seconds of a new system's data.",
         result="From 2 seconds of data, the adapted 2-number context reaches RMSE 0.014 on a fresh 30-second validation input, 4x better than predicting with the average system (0.056) and 11x better than fitting all four coefficients from scratch, which overfits (0.158)."),
    dict(stem="05_when_is_pretraining_worth_it", short="When Is Pretraining Worth It?",
         build="A data-budget study over 40 fresh systems per budget, comparing no adaptation, from-scratch fitting, the pretrained context and the true coefficients as the experiment length grows from 1.5 to 25 seconds.",
         result="At 1.5 seconds of data, from-scratch identification's error is 5.9x the pretrained-context error; the gap narrows steadily to about 1.1x by 25 seconds, as pretraining buys data efficiency rather than a better ceiling."),
    dict(stem="06_choosing_the_best_experiment", short="Choosing the Best Experiment",
         build="Bayesian D-optimal design from the Fisher information of the pretrained model, scoring 200-300 candidate experiments before ever touching the new system, then testing the winner against a random and a worst experiment over 60 new systems per budget.",
         result="A designed 6-second experiment cuts the mean prediction error by 55% versus a random experiment of the same length (0.0096 vs 0.0216 RMSE) and wins 78% of paired trials, with a 95% bootstrap interval that excludes zero."),
    dict(stem="07_when_the_shared_knowledge_does_not_apply", short="When the Shared Knowledge Does Not Apply",
         build="Pushing a new system's coefficients away from the learned family along a direction the 2-number context cannot represent, and checking whether the adapter's own output misfit can flag it.",
         result="As the shift grows from δ=0 to δ=0.6, the pretrained model's median error rises from 0.0096 to 0.0133 while the from-scratch estimate's error falls (0.0132 to 0.0048), and the misfit test's flag rate climbs from about 1 in 10 systems to well over a third."),
    dict(stem="08_from_a_better_model_to_better_control", short="From a Better Model to Better Control",
         build="A model-based feedforward controller computed from each candidate model and applied to the true system, so tracking error becomes a direct test of model quality.",
         result="A 6-second designed experiment, adapted through the pretrained family, cuts feedforward tracking error from 0.058 (average-system model) to 0.019, close to the 0.014 achieved with the true coefficients."),
    dict(stem="09_summary_and_the_road_to_a_foundation_model", short="Summary and the Road to a Foundation Model",
         build="Mapping every piece built in this course, the backbone, the context, the design criterion, the misfit test, onto its counterpart in a real, large-scale foundation model for dynamical systems, plus the open questions it raises.",
         result="The same two numbers that explained 96% of the hidden design variables in chapter 03 are what took chapter 08's feedforward tracking error from 0.058 down to 0.019, within 40% of what the true four coefficients achieve."),
]
NOTE = "To run a chapter you also need <code>dynutils.py</code>: the full course download has it."
COURSE = Course(
    name="Learning to Identify New Dynamical Systems", folder="few-shot-dynamical-systems", chapters=CHAPTERS,
    notes={ch["stem"]: NOTE for ch in CHAPTERS},
    license=('Built from the author\'s own research tutorial repository on few-shot dynamical-system '
             'identification, MIT licensed; the companion technical report is included in the course download.'),
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
    hero = results[3]   # chapter 04's context-and-prediction figure
    return page_head(
        "Learning to Identify New Dynamical Systems from a Few Experiments",
        "A nine-chapter research tutorial: can knowledge from many previously seen dynamical systems make a "
        "brand-new one identifiable, controllable and predictable from a few seconds of data? Pretraining, "
        "few-shot Bayesian adaptation, Bayesian-optimal experiment design, out-of-family detection and "
        "model-based control, on a family of nonlinear oscillators small enough to inspect end to end.",
    ) + f"""
<nav class="site-back col" aria-label="Site"><a href="/tutorials">&larr; All tutorials</a><a href="/">Elie Rouphael</a></nav>

<header class="hero col">
  <span class="eyebrow">Nine notebooks · NumPy, SciPy, Matplotlib</span>
  <h1>Learning to identify new dynamical systems from a few experiments</h1>
  <p class="lede">Engineers rarely meet a dynamical system that is completely new: a new motor resembles motors
  characterised before, a new robot joint behaves like other joints of the same series. This course is the
  author's own research on <strong>few-shot system identification</strong>, rebuilt as nine runnable chapters on a
  family of nonlinear oscillators small enough that every step, the pretraining, the adaptation, the experiment
  design, the failure mode, can be simulated, inspected and checked against ground truth in seconds.</p>
  <p class="byline">Elie Rouphael · Assumes Python, a little linear algebra and probability</p>
  <div class="downloads">
    <a class="btn" href="{slug(CHAPTERS[0])}.html">Start with chapter 01</a>
    <a class="btn ghost" href="{zip_name}" download>Download the course (.zip)</a>
  </div>
</header>

<section class="wide" aria-label="Adapting to a new system from two seconds of data">
  <figure class="output-figure hero-figure">
    <img src="img/04-1.png" alt="Left: a scatter of training-system contexts with a small posterior ellipse around a new system's estimated context after two seconds of data, containing the true system. Right: the true response to an unseen input against three predictions -- no adaptation, from-scratch fitting, and the pretrained context -- with the pretrained context tracking the truth closely." width="{hero['hero_w']}" height="{hero['hero_h']}">
    <figcaption><b>What the course builds towards.</b> Two seconds of data are not enough to fit four coefficients from scratch: the fit overfits and its predictions drift. Interpreted through a backbone pretrained on 20 other systems, the same two seconds identify a 2-number context whose predictions track 30 unseen seconds closely, and whose posterior ellipse contains the truth. From chapter 04.</figcaption>
  </figure>
</section>

<section class="col prose" aria-labelledby="question">
  <h2 id="question">One family, nine questions</h2>
  <p>Every chapter works with the same family of carts on a nonlinear spring, and the same rule: the learner
  chooses an input and reads a noisy position sensor, nothing else. Its model is deliberately missing two of the
  true system's effects, so no method here ever reaches zero error. Chapters 01-02 set the stage and the
  from-scratch baseline. Chapter 03 pretrains a two-number "family" from 20 other systems. Chapters 04-08 each
  ask what that shared knowledge buys: identification from little data, knowing when it's worth it, choosing the
  best experiment, noticing when it stops applying, and turning it into control. Chapter 09 maps every piece onto
  its counterpart at the scale of a real foundation model for dynamical systems.</p>
  <p>Nothing here is asymptotic theory: every claim is a Monte Carlo comparison over freshly drawn systems, with
  the random baselines never selected by the criterion they're compared against, and every number quoted in a
  chapter is that chapter's own output.</p>
</section>

<section class="col" aria-labelledby="chapters">
  <h2 id="chapters">Chapters</h2>
  <p class="section-lede">Chapters 01-02 build the simulator and the from-scratch baseline every later idea has
  to beat. Chapter 03 is the hinge: it pretrains the low-dimensional family that chapters 04-08 all reuse, each
  rebuilding it deterministically from the same seed rather than depending on a saved file. Chapter 09 closes
  with a map from this toy problem to a real foundation model for dynamics.</p>
  <ol class="chapter-list">
{chapters}
  </ol>
</section>

<section class="col prose" aria-labelledby="run">
  <h2 id="run">Running it yourself</h2>
  <p>Every chapter page shows the notebook with its outputs, so you can read the whole course here. To run and
  change the code, <a href="{zip_name}" download>download the course</a> (the notebooks, the shared
  <code>dynutils.py</code> helpers, and the requirements), then from its folder run:</p>
  <pre><code class="language-bash">pip install -r requirements.txt
jupyter lab                 # then open 01_the_system_and_what_we_can_measure.ipynb</code></pre>
  <p>Chapters 03-08 each rebuild the pretrained family from scratch in well under a second, using the same seed,
  so any chapter can also be run entirely on its own. The slowest chapters (06-08) take under a minute each; the
  whole course runs in a couple of minutes on a laptop. Every chapter ends with exercises, with solutions folded
  away.</p>
</section>

<section class="col prose" aria-labelledby="data">
  <h2 id="data">Source and further reading</h2>
  <p>Every system in this course is simulated, not measured: the "data" is generated by a known nonlinear plant
  with a hidden low-dimensional family structure, so every method can be checked against ground truth. This
  course is built from the author's own research tutorial repository on few-shot dynamical-system identification
  (MIT licensed); the course download includes the companion technical report, which derives every equation used
  here in full and discusses the open questions in chapter 09 at greater length.</p>
</section>

<footer class="col">
  <p>Learning to Identify New Dynamical Systems from a Few Experiments, by Elie Rouphael. The chapter pages are
  rendered from the executed notebooks; math by KaTeX, code highlighting by highlight.js.</p>
</footer>
""" + page_scripts()


# ---------------------------------------------------------------- cover

def make_cover(path: Path, course: Path) -> None:
    """Training-system contexts with a designed-experiment posterior ellipse, drawn to read at thumbnail size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(course))
    import dynutils

    rng = np.random.default_rng(dynutils.SEED)
    family, *_ = dynutils.build_family(rng)
    new = dynutils.sample_system(rng)
    pool = np.array([dynutils.multisine(40, rng) for _ in range(200)])
    u_best = pool[np.argmax(dynutils.design_score(family, pool))]
    z_hat, cov, _ = dynutils.adapt(family, dynutils.simulate(new, u_best, rng)[1], u_best)

    ink, blue = "#1d2530", "#2a78d6"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor("#fbfbf8"); ax.set_facecolor("#fbfbf8")
    ax.scatter(*family.z_train.T, s=90, color="#9a9893", alpha=0.85, edgecolor="white", linewidth=0.6)
    dynutils.ellipse(ax, z_hat, cov, blue, n_std=2.5)
    ax.plot(*z_hat, "o", color=blue, ms=10)
    ax.plot(*family.project(new["eta"]), "*", ms=26, color=ink)
    ax.text(family.z_train[:, 0].min() - 0.3, family.z_train[:, 1].max() + 0.6,
            "Few-Shot\nDynamical Systems", color=ink, fontsize=26, weight="bold", va="top")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    pad = 0.8
    ax.set_xlim(family.z_train[:, 0].min() - pad, family.z_train[:, 0].max() + pad)
    ax.set_ylim(family.z_train[:, 1].min() - pad, family.z_train[:, 1].max() + pad * 2.4)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    course = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_COURSE
    executed = course / "executed"
    if not executed.is_dir():
        sys.exit(f"Executed notebooks not found in {executed}. Run build_notebooks.py in the course folder first.")

    results = build_chapters(COURSE, executed)
    out = COURSE.out

    w, h = png_size((out / "img" / "04-1.png").read_bytes())
    results[3].update(hero_w=w, hero_h=h)
    (out / "index.html").write_text(index_page(results), encoding="utf-8")

    members = [(course / rel, rel) for rel in ("dynutils.py", "build_notebooks.py")]
    members += [(course.parent / rel, rel) for rel in ("README.md", "requirements.txt", "LICENSE",
                                                        "report/tutorial_report.pdf")]
    members += [(course / f"{ch['stem']}.ipynb", f"{ch['stem']}.ipynb") for ch in CHAPTERS]
    members += [(executed / f"{ch['stem']}.ipynb", f"executed/{ch['stem']}.ipynb") for ch in CHAPTERS]
    write_zip(COURSE, members)

    make_cover(out / "cover.png", course)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
