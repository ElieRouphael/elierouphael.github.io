"""Build the Bayesian statistics course pages from the executed notebooks.

Reads the ten notebooks of the course (Courses/Bayesian Statistics/bayesian-course)
and writes static pages into public/tutorials/bayesian-statistics/:

    index.html                        course landing page
    00-overview.html ... 09-mcmc.html one page per notebook
    img/, notebooks/                  extracted figures, notebooks for download
    bayesian-statistics-course.zip    notebooks + chapter sources + requirements
    cover.png                         preview image for the tutorials card

Chapter pages, style.css and course.js come from scripts/notebook_course.py. The
notebooks must already be executed (run build_notebooks.py in the course folder).

Usage:
    pip install markdown-it-py mdit-py-plugins numpy matplotlib
    python scripts/build_bayesian_course.py [path/to/bayesian-course]
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

from notebook_course import SITE, Course, build_chapters, page_head, page_scripts, png_size, slug, write_zip

DEFAULT_COURSE = SITE.parents[1] / "Courses" / "Bayesian Statistics" / "bayesian-course"

# `result` lines quote the notebooks' own outputs; re-check them if the notebooks change.
CHAPTERS = [
    dict(stem="00_overview", short="What Bayesian statistics is",
         build="Three meanings of probability, the prior-likelihood-posterior recipe, and a first posterior watched as data arrive.",
         result="A flat prior sharpens into a narrow posterior around the true cure rate as 200 patients arrive."),
    dict(stem="01_probability_and_bayes", short="Probability and Bayes' theorem",
         build="Conditional probability, the law of total probability, Bayes' theorem and its odds form for repeated evidence.",
         result="At 1% prevalence, a positive result from a 95%-sensitive test means only an 8.8% chance of disease."),
    dict(stem="02_distributions", short="Random variables and distributions",
         build="Expectation and variance, the standard discrete and continuous families, scipy's parameterisations, and the central limit theorem.",
         result="Averages of a heavily skewed exponential look normal by n = 30."),
    dict(stem="03_likelihood_and_frequentist", short="Likelihood and frequentist inference",
         build="The likelihood function, maximum likelihood, Fisher information, and what a confidence interval does and does not promise.",
         result="A nominal 95% Wald interval covers the truth only 64% of the time at θ = 0.05, n = 20."),
    dict(stem="04_prior_to_posterior", short="From prior to posterior",
         build="Continuous Bayes by grid approximation, point estimates and credible intervals, the posterior predictive, sequential updating.",
         result="Updating one throw at a time matches the all-at-once posterior to 1e-14."),
    dict(stem="05_discrete_conjugate", short="Counts and proportions",
         build="The beta-binomial and Poisson-gamma models derived, priors as imaginary data, and closed-form predictive distributions.",
         result="An A/B test: version B converts better with posterior probability 0.96."),
    dict(stem="06_continuous_conjugate", short="Continuous data",
         build="Exponential-gamma, the normal mean by completing the square, and the normal-inverse-gamma model with its t marginal.",
         result="Exact composition sampling reproduces the analytic t posterior for a mean."),
    dict(stem="07_priors", short="Choosing priors",
         build="Eliciting a prior from an expert, improper and flat priors, Jeffreys priors, prior predictive checks and sensitivity analysis.",
         result="Jeffreys' interval covers 98% where the Wald interval covers 64%."),
    dict(stem="08_linear_regression", short="Bayesian linear regression",
         build="Reference and conjugate priors for regression, exact posterior sampling, credible and prediction bands, ridge as a posterior mean.",
         result="Credible intervals that match least squares, plus a posterior for any derived quantity."),
    dict(stem="09_mcmc", short="Monte Carlo and Metropolis",
         build="Monte Carlo error, the Metropolis algorithm from scratch, step-size tuning, effective sample size and R-hat.",
         result="Four Metropolis chains agree on a golf-putting logistic regression, R-hat 1.00."),
]
COURSE = Course(name="Bayesian statistics course", folder="bayesian-statistics", chapters=CHAPTERS,
                # Chapters rebuilt by hand as interactive pages; the builder leaves their .html alone.
                interactive={"00_overview", "01_probability_and_bayes", "02_distributions", "03_likelihood_and_frequentist",
                             "04_prior_to_posterior", "05_discrete_conjugate", "06_continuous_conjugate",
                             "07_priors", "08_linear_regression", "09_mcmc"})


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
          <span class="r"><span class="lbl">You compute</span> {html.escape(ch['result'])}</span>
        </span>
      </a>
    </li>""")
    chapters = "\n".join(items)
    zip_name = f"{COURSE.folder}-course.zip"
    return page_head(
        "Bayesian Statistics from Scratch",
        "A ten-chapter course in Bayesian statistics, from Bayes' theorem to conjugate models, priors, "
        "regression and Markov chain Monte Carlo. Every formula is derived and checked in Python.",
    ) + f"""
<nav class="site-back col" aria-label="Site"><a href="/tutorials">&larr; All tutorials</a><a href="/">Elie Rouphael</a></nav>

<header class="hero col">
  <span class="eyebrow">Bayesian statistics · Ten interactive chapters · notebooks in Python</span>
  <h1>Bayesian statistics from scratch</h1>
  <p class="lede">A course that builds Bayesian inference from one rule, Bayes' theorem, up to regression and Markov chain Monte Carlo. Each chapter derives the mathematics by hand, lets you try it in live demos, and checks every result in code: simulations against formulas, grids against closed forms, samplers against exact answers.</p>
  <p class="byline">Elie Rouphael · Assumes algebra, a little calculus and basic Python; no prior statistics course needed</p>
  <div class="downloads">
    <a class="btn" href="{slug(CHAPTERS[0])}.html">Start with chapter 00</a>
    <a class="btn ghost" href="{zip_name}" download>Download the course (.zip)</a>
  </div>
</header>

<section class="wide" aria-label="A posterior learning from data">
  <figure class="output-figure hero-figure">
    <img src="img/00-2.png" alt="Four posterior densities for a cure rate, after 0, 5, 25 and 200 patients. The first is flat; each later one is narrower, and the last is a tall narrow peak near the true value 0.3." width="{results[0]['hero_w']}" height="{results[0]['hero_h']}">
    <figcaption><b>What the course is about, in one picture.</b> The unknown fraction of patients helped by a treatment starts with a flat prior. As patients arrive, Bayes' theorem turns it into a posterior that moves towards the true value (dashed) and narrows. The answer is a whole distribution: a best guess, a range, and the probability of any statement about the cure rate. From chapter 00.</figcaption>
  </figure>
</section>

<section class="col prose" aria-labelledby="rule">
  <h2 id="rule">One rule</h2>
  <p>Bayesian statistics represents everything uncertain with a probability distribution and updates it with Bayes' theorem when data arrive:</p>
  <div class="math display">p(\\theta\\mid y) \\;\\propto\\; p(y\\mid\\theta)\\,p(\\theta)</div>
  <div class="arch">
    <div class="arch-card"><span class="eq-name">Prior · <span class="f">p(θ)</span></span><p>What you believe about the unknown θ before the data. Drawn in grey throughout the course.</p></div>
    <div class="arch-card"><span class="eq-name" style="color:#eb6834">Likelihood · <span class="f">p(y | θ)</span></span><p>How probable the observed data are for each value of θ: the model of how data arise. Drawn in orange.</p></div>
    <div class="arch-card accent"><span class="eq-name">Posterior · <span class="f">p(θ | y)</span></span><p>What you believe after seeing the data. Everything you report comes from it. Drawn in blue.</p></div>
  </div>
  <p>The course is about doing this well: choosing likelihoods and priors, computing the posterior exactly when a conjugate model allows and by sampling when it does not, and reading the result honestly, with credible intervals, predictions, model checks and a sensitivity analysis.</p>
</section>

<section class="col" aria-labelledby="chapters">
  <h2 id="chapters">Chapters</h2>
  <p class="section-lede">The first four chapters build the probability and inference toolkit and contrast the Bayesian and classical answers to the same question. The next four develop the conjugate models, priors and regression. The last replaces formulas with sampling.</p>
  <ol class="chapter-list">
{chapters}
  </ol>
</section>

<section class="col prose" aria-labelledby="run">
  <h2 id="run">Running it yourself</h2>
  <p>You can read and try the whole course here: every chapter page has live demos that run in your browser, the derivations, and the notebook's code with its printed results. To run and change the code yourself, <a href="{zip_name}" download>download the course</a> (the notebooks, their markdown sources and the requirements) and run:</p>
  <pre><code class="language-bash">pip install -r requirements.txt
jupyter notebook            # then open notebooks/00_overview.ipynb</code></pre>
  <p>Each notebook is self-contained and runs in seconds on a laptop, using only numpy, scipy and matplotlib. Chapters 01 to 09 end with exercises, each with a hint and a solution, and every chapter lists its references.</p>
</section>

<section class="col prose" aria-labelledby="sources">
  <h2 id="sources">Sources</h2>
  <p>The order of topics follows the arc of Herbert Lee's course <em>Bayesian Statistics: From Concept to Data Analysis</em> (University of California, Santa Cruz). The explanations, examples and code are written from scratch for this course. For more depth, the best next books are Hoff's <em>A First Course in Bayesian Statistical Methods</em>, McElreath's <em>Statistical Rethinking</em>, and Gelman et al.'s <em>Bayesian Data Analysis</em>; each chapter lists its own references.</p>
</section>

<footer class="col">
  <p>Bayesian statistics course by Elie Rouphael. The chapter pages are written from the executed notebooks, with demos that run in your browser; math by KaTeX, code highlighting by highlight.js.</p>
</footer>
""" + page_scripts()


# ---------------------------------------------------------------- cover

def make_cover(path: Path) -> None:
    """Prior, likelihood and posterior for the free-throw example, drawn to read at thumbnail size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.linspace(0.001, 0.999, 999)
    def beta_pdf(a, b):
        f = t ** (a - 1) * (1 - t) ** (b - 1)
        return f / (f.sum() * (t[1] - t[0]))
    prior, lik, post = beta_pdf(24, 9), beta_pdf(14, 8), beta_pdf(37, 16)   # 13 of 20 with a Beta(24, 9) prior

    grey, orange, blue = "#8a949e", "#eb6834", "#2a78d6"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor("#fbfbf8"); ax.set_facecolor("#fbfbf8")
    ax.fill_between(t, prior, color=grey, alpha=0.35, lw=0)
    ax.plot(t, prior, color=grey, lw=3)
    ax.plot(t, lik, color=orange, lw=4, ls=(0, (4, 2)))
    ax.fill_between(t, post, color=blue, alpha=0.15, lw=0)
    ax.plot(t, post, color=blue, lw=5)
    ax.text(0.83, prior.max() * 0.93, "prior", color=grey, fontsize=26, weight="bold")
    ax.text(0.20, lik.max() * 0.62, "likelihood", color=orange, fontsize=26, weight="bold")
    ax.text(0.40, post.max() * 0.92, "posterior", color=blue, fontsize=26, weight="bold", ha="right")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b9bfb6")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0.15, 1.0); ax.set_ylim(0, post.max() * 1.08)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    course = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_COURSE
    nb_dir = course / "notebooks"
    if not nb_dir.is_dir():
        sys.exit(f"Notebooks not found in {nb_dir}. Pass the bayesian-course folder as an argument.")

    results = build_chapters(COURSE, nb_dir)
    out = COURSE.out
    w, h = png_size((out / "img" / "00-2.png").read_bytes())
    results[0].update(hero_w=w, hero_h=h)
    (out / "index.html").write_text(index_page(results), encoding="utf-8")

    members = [(course / rel, rel) for rel in ("README.md", "requirements.txt", "build_notebooks.py")]
    members += [(f, f"chapters/{f.name}") for f in sorted((course / "chapters").glob("*.md"))]
    members += [(nb_dir / f"{ch['stem']}.ipynb", f"notebooks/{ch['stem']}.ipynb") for ch in CHAPTERS]
    write_zip(COURSE, members)

    make_cover(out / "cover.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
