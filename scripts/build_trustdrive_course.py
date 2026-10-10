"""Build the TrustDrive course pages from the executed notebooks.

Reads the ten notebooks of the TrustDrive course and writes static pages into
public/tutorials/trustdrive/:

    index.html                      course landing page
    00-overview.html ... 09-*.html  one page per notebook (markdown, code, outputs)
    img/                            figures extracted from the notebook outputs
    notebooks/*.ipynb               the notebooks, for download
    trustdrive-course.zip           notebooks + courselib + requirements, for download
    cover.png                       preview image for the tutorials card

The chapter pages, style.css and course.js come from scripts/notebook_course.py,
which is shared with the other notebook courses. The notebooks must already be
executed: outputs are copied, never recomputed.

Usage:
    pip install markdown-it-py mdit-py-plugins numpy matplotlib
    python scripts/build_trustdrive_course.py [path/to/trustdrive-course]
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

from notebook_course import SITE, Course, build_chapters, page_head, page_scripts, png_size, slug, write_zip

DEFAULT_COURSE = SITE.parents[1] / "Courses" / "trustdrive-minilab" / "trustdrive-course"

# Curated from the course README. `stem` is the notebook file name without .ipynb.
CHAPTERS = [
    dict(stem="00_overview", short="Overview",
         build="The closed-loop problem, why self-reported confidence is not enough, and the notation used throughout.",
         result="Reported uncertainty stays flat while NIS jumps from 1.3 to 13.6 out of distribution."),
    dict(stem="01_vehicle_model", short="The vehicle model",
         build="The kinematic bicycle from the no-slip constraints, Frenet error dynamics, discretisation, linearisation, controllability and observability.",
         result="Euler convergence order fits 1.00; the state is observable from the lateral offset alone."),
    dict(stem="02_lqr_control", short="LQR lane keeping",
         build="LQR by dynamic programming, the discrete Riccati equation, Bryson's rule, curvature feed-forward and gain scheduling.",
         result="DARE residual of 7e-15 and the tracking-versus-effort cost frontier."),
    dict(stem="03_perception_and_rendering", short="Perception and rendering",
         build="Pinhole projection, the ground-plane homography, blur and noise models, and a synthetic lane camera.",
         result="An out-of-distribution curve is invisible in the pixel statistics."),
    dict(stem="04_uncertainty_ensembles", short="Uncertainty and ensembles",
         build="Aleatoric versus epistemic uncertainty through the law of total variance, deep ensembles, calibration and proper scoring rules.",
         result="Ensemble NLL of -0.37 against 11.6 for a single model, and the case where uncertainty is not error."),
    dict(stem="05_kalman_and_nis", short="Kalman filter and NIS",
         build="The Kalman filter as recursive Bayes, innovation whiteness, the chi-squared NIS statistic and consistency tests.",
         result="White innovations and a mean NIS of 2.08 against the expected 2."),
    dict(stem="06_odd_and_integrity", short="ODD and integrity",
         build="The operational design domain as set membership, three detectors, their fusion into an integrity score, ROC and detection delay.",
         result="Fused AUC of 0.96 against roughly 0.70 for any single detector."),
    dict(stem="07_shared_control", short="Shared control",
         build="Haptic shared control, the McRuer crossover driver model, workload, and the law that shares authority between driver and automation.",
         result="Stability with a delayed driver in the loop and integrity-aware conflict arbitration."),
    dict(stem="08_capstone_closed_loop", short="Capstone: the whole system",
         build="Everything assembled into one closed loop and evaluated as a controlled experiment, with multi-seed statistics and a component ablation.",
         result="Proposed versus naive: -0.170 m lateral RMSE, 95% CI [-0.185, -0.157]."),
    dict(stem="09_advanced_ideas", short="Advanced ideas",
         build="CUSUM change detection and its average run length, conformal prediction with adaptive coverage, and a research roadmap.",
         result="CUSUM beats the single-sample gate on the detection-delay versus false-alarm trade-off."),
]
COURSELIB_NOTE = ("This notebook imports <code>courselib</code>, the course's reference library. "
                  "Download the full course to run it.")
COURSE = Course(
    name="TrustDrive course",
    folder="trustdrive",
    chapters=CHAPTERS,
    license="MIT licensed.",
    notes={"08_capstone_closed_loop": COURSELIB_NOTE, "09_advanced_ideas": COURSELIB_NOTE},
    # Chapters rebuilt by hand as interactive pages; the builder leaves their .html alone.
    interactive={"01_vehicle_model", "02_lqr_control", "03_perception_and_rendering"},
)



# ---------------------------------------------------------------- landing

def index_page(results: list[dict]) -> str:
    items = []
    for ch, r in zip(CHAPTERS, results):
        items.append(f"""    <li>
      <a href="{slug(ch)}.html">
        <span class="n">{ch['stem'][:2]}</span>
        <span class="body">
          <span class="t">{html.escape(r['title'])}</span>
          <span class="d">{html.escape(ch['build'])}</span>
          <span class="r"><span class="lbl">You reproduce</span> {html.escape(ch['result'])}</span>
        </span>
      </a>
    </li>""")
    chapters = "\n".join(items)
    return page_head(
        "Trustworthy AI Perception in a Control Loop",
        "A ten-chapter course that builds a small lane-keeping system which stays safe when its AI perception "
        "becomes unreliable: vehicle model, LQR, perception, ensembles, Kalman filtering and NIS, integrity "
        "monitoring and shared control. numpy and matplotlib only.",
    ) + f"""
<nav class="site-back col" aria-label="Site"><a href="/tutorials">&larr; All tutorials</a><a href="/">Elie Rouphael</a></nav>

<header class="hero col">
  <span class="eyebrow">TrustDrive course · Ten notebooks · numpy + matplotlib only</span>
  <h1>Trustworthy AI perception in a control loop</h1>
  <p class="lede">What should an autonomous vehicle do when its AI perception becomes uncertain, and how should control be shared between the automation and the driver? This course answers that by building a small lane-keeping system, one idea at a time, that stays safe when its perception fails.</p>
  <p class="byline">Elie Rouphael · Assumes basic linear algebra, probability and Python; no prior control theory or deep learning</p>
  <div class="downloads">
    <a class="btn" href="{slug(CHAPTERS[0])}.html">Start with chapter 00</a>
    <a class="btn ghost" href="trustdrive-course.zip" download>Download the course (.zip)</a>
  </div>
</header>

<section class="wide" aria-label="The central effect">
  <figure class="output-figure hero-figure">
    <img src="img/00-1.png" alt="Two stacked plots over 15 seconds. Top: the network's self-reported uncertainty is a flat line throughout. Bottom: the NIS consistency statistic stays below the chi-squared threshold of 9.21 until an out-of-distribution bias starts at 7.5 seconds, then repeatedly jumps far above it." width="{results[0].get('hero_w', 870)}" height="{results[0].get('hero_h', 410)}">
    <figcaption><b>The whole course in one experiment.</b> Halfway through, the perception becomes systematically wrong while staying just as confident. A monitor that watches the network's reported uncertainty (top) never reacts. A model-based consistency check, the normalised innovation squared (bottom), crosses its threshold straight away. The rest of the course turns that detection into safe action. From chapter 00.</figcaption>
  </figure>
</section>

<section class="col prose" aria-labelledby="idea">
  <h2 id="idea">The idea you prove to yourself</h2>
  <p>A neural network maps camera pixels to the lane offset and heading a controller needs. Inside the conditions it was trained on, it works. Outside them it can be <em>confidently wrong</em>, and in a closed loop a wrong estimate does not just sit on a screen: the controller acts on it and the car moves. The course compares three ways of using the same perception.</p>
  <div class="arch">
    <div class="arch-card"><span class="eq-name">Naive AI</span><p>The perception drives the controller directly. No monitoring.</p></div>
    <div class="arch-card"><span class="eq-name">Hard handover</span><p>Switch to the driver when the network's <em>reported</em> uncertainty crosses a threshold.</p></div>
    <div class="arch-card accent"><span class="eq-name">Integrity-aware</span><p>Fuse epistemic uncertainty, a Kalman-filter consistency test and operational-domain membership into an integrity score, then share authority smoothly and slow down.</p></div>
  </div>
  <p>On an out-of-distribution curve the network's reported uncertainty barely moves, so the threshold never triggers. Only the model-based checks catch it. The capstone shows, with a controlled and statistically tested experiment, that using perception <em>through</em> a monitoring layer is safer than either trusting it or switching away from it.</p>
</section>

<section class="col" aria-labelledby="chapters">
  <h2 id="chapters">Chapters</h2>
  <p class="section-lede">Each chapter isolates one idea, derives the mathematics, implements it from scratch in numpy, runs an experiment, and ends with exercises and references to the primary literature.</p>
  <ol class="chapter-list">
{chapters}
  </ol>
</section>

<section class="col prose" aria-labelledby="run">
  <h2 id="run">Running it yourself</h2>
  <p>Every chapter page shows the notebook with its outputs, so you can read the whole course here. To run and modify the code, <a href="trustdrive-course.zip" download>download the course</a> (notebooks, the <code>courselib</code> reference library and the requirements) and run:</p>
  <pre><code class="language-bash">pip install -r requirements.txt
jupyter notebook            # then open notebooks/00_overview.ipynb</code></pre>
  <p>No GPU, deep-learning framework or simulator is needed: the course runs on numpy and matplotlib only. Work through the notebooks in order. Chapters 00 to 07 are self-contained; the capstone and the advanced chapter import <code>courselib</code>, the tidy version of everything built before.</p>
</section>

<section class="col prose" aria-labelledby="rigour">
  <h2 id="rigour">What makes it rigorous</h2>
  <ul>
    <li><b>Derivations, not assertions.</b> The Riccati recursion, the chi-squared distribution of the NIS, the total-variance decomposition and the conformal coverage bound are each worked out.</li>
    <li><b>Every claim is run.</b> Convergence orders, filter consistency, ROC and AUC, bootstrap confidence intervals and ablations are computed, not stated.</li>
    <li><b>Primary sources.</b> About 90 references across the ten chapters, from Kalman, Bar-Shalom, Anderson and Moore and McRuer to Lakshminarayanan, Vovk, Page and the driving-specific monitoring literature.</li>
    <li><b>Honest limitations.</b> The capstone states where the demonstrator is simplified (synthetic perception, kinematic plant, heuristic thresholds), and chapter 09 turns those limits into research directions.</li>
  </ul>
</section>

<footer class="col">
  <p>TrustDrive course by Elie Rouphael. MIT licensed. The chapter pages are rendered from the executed notebooks; math by KaTeX, code highlighting by highlight.js.</p>
</footer>
""" + page_scripts()


# ---------------------------------------------------------------- cover

def make_cover(path: Path) -> None:
    """The chapter 00 teaser, redrawn as a clean 16:10 card image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Same computation as the teaser in 00_overview.
    L, Ts, V, CHI2 = 2.7, 0.05, 15.0, 9.210
    A = np.array([[1, V * Ts], [0, 1]]); E = np.array([0, -V * Ts]); Q = np.diag([1e-4, 1e-4])
    rng = np.random.default_rng(1)
    x = np.zeros(2); P = np.diag([0.05, 0.02]); true = np.zeros(2)
    srep = np.array([0.02, 0.01]); N = 300; sig, nis = [], []
    for k in range(N):
        kappa = 0.012 * np.sin(2 * np.pi * k * Ts / 6)
        true = np.array([true[0] + V * np.sin(true[1]) * Ts, true[1] - V * kappa * Ts])
        bias = np.zeros(2) if k < 150 else np.array([0.22, 0.06])
        y = true + bias + rng.normal(0, srep)
        x = A @ x + E * kappa; P = A @ P @ A.T + Q
        S = P + np.diag(srep ** 2); nu = y - x
        nis.append(float(nu @ np.linalg.solve(S, nu))); sig.append(np.sqrt((srep ** 2).sum()))
        Kg = P @ np.linalg.inv(S); x = x + Kg @ nu; P = (np.eye(2) - Kg) @ P
    t = np.arange(N) * Ts

    ink, muted, accent, violet, bug = "#16202a", "#6b7581", "#eb6834", "#4a3aa7", "#e34948"
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), dpi=160, sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 2.2], hspace=0.12))
    fig.patch.set_facecolor("#fbfbf8")
    for a in ax:
        a.set_facecolor("#fbfbf8")
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color("#b9bfb6")
        a.tick_params(colors=muted, labelsize=9, length=0)
        a.set_yticks([])
        a.axvspan(150 * Ts, N * Ts, color=bug, alpha=0.07, lw=0)
    # Drawn to be read at thumbnail size (about 240 px wide): few, large labels.
    ax[0].plot(t, sig, color=violet, lw=5); ax[0].set_ylim(0, 0.05)
    ax[0].text(0.3, 0.031, "reported uncertainty", color=violet, fontsize=26, weight="bold")
    ax[1].plot(t, nis, color=accent, lw=2.6)
    ax[1].axhline(CHI2, ls="--", color=ink, lw=2)
    ax[1].text(0.3, 38, "NIS check", color=accent, fontsize=26, weight="bold")
    ax[1].set_ylim(-2, 62)
    ax[1].set_xticks([])
    fig.subplots_adjust(left=0.04, right=0.97, top=0.95, bottom=0.05)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    course = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_COURSE
    nb_dir = course / "notebooks"
    if not nb_dir.is_dir():
        sys.exit(f"Notebooks not found in {nb_dir}. Pass the trustdrive-course folder as an argument.")

    results = build_chapters(COURSE, nb_dir)
    out = COURSE.out
    w, h = png_size((out / "img" / "00-1.png").read_bytes())
    results[0].update(hero_w=w, hero_h=h)
    (out / "index.html").write_text(index_page(results), encoding="utf-8")

    members = [(course / rel, rel) for rel in ("README.md", "LICENSE", "requirements.txt", "test_courselib.py")]
    members += [(f, f"courselib/{f.name}") for f in sorted((course / "courselib").glob("*.py"))]
    members += [(nb_dir / f"{ch['stem']}.ipynb", f"notebooks/{ch['stem']}.ipynb") for ch in CHAPTERS]
    write_zip(COURSE, members)

    make_cover(out / "cover.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
