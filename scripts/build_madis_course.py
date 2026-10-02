"""Build the "A Tour of Machine Learning" course pages from the executed notebooks.

Reads the eleven executed notebooks of the course
(MADIS Machine Learning Workshop/course/executed) and writes static pages into
public/tutorials/ml-tour/:

    index.html                                    course landing page
    01-linear-regression.html ... 11-federated-learning.html
    img/, notebooks/                              extracted figures, notebooks for download
    ml-tour-course.zip                            notebooks, mlutils.py, data and requirements
    cover.png                                     preview image for the tutorials card

Chapter pages, style.css and course.js come from scripts/notebook_course.py. The
notebooks must already be executed (run build_notebooks.py in the course folder,
using its course_venv which has tensorflow-cpu installed).

Unlike the sequential Time-Series or Bayesian courses, each chapter here is a
self-contained idea (linear regression, SVMs, an autoencoder, ...) with no shared
scoreboard between chapters.

Usage:
    pip install markdown-it-py mdit-py-plugins numpy matplotlib
    python scripts/build_madis_course.py [path/to/course]
"""

from __future__ import annotations

import sys
from pathlib import Path

from notebook_course import SITE, Course, build_chapters, page_head, page_scripts, png_size, slug, write_zip

DEFAULT_COURSE = SITE.parents[1] / "MADIS Machine Learning Workshop" / "course"

# `result` lines quote the notebooks' own outputs; re-check them if the notebooks change.
CHAPTERS = [
    dict(stem="01_linear_regression", short="Linear Regression",
         build="Simple and multiple regression on real car data, reading coefficients as rates of change, and R².",
         result="Weight and engine volume together reach R²=0.377 (up from 0.305 with weight alone); a 1300kg/1300cm³ car is predicted at 99.7 g/km of CO2."),
    dict(stem="02_minima_and_maxima", short="Finding Minima and Maxima",
         build="Critical points and the second-derivative test with sympy, then the same calculus behind a training loop, checked against gradient descent.",
         result="Gradient descent converges to w*=2.3342 after 200 steps — the exact closed-form minimiser found by solving L'(w)=0 by hand."),
    dict(stem="03_support_vector_machines", short="Support Vector Machines",
         build="An RBF-kernel SVM on real iris measurements, its decision boundary drawn directly, and a confusion matrix on held-out flowers.",
         result="93.3% test accuracy using only 34 of 105 training points as support vectors."),
    dict(stem="04_random_forest", short="Random Forest",
         build="A single decision tree vs. a 300-tree random forest on real heart-disease data, and the feature importances the forest learned.",
         result="The forest reaches 78.9% test accuracy against 72.4% for one unrestricted tree; chest-pain type is the most important feature."),
    dict(stem="05_kmeans_and_kmeans_plusplus", short="K-Means and K-Means++",
         build="Customer segmentation by the elbow method, then 30 seeds of random vs. k-means++ initialisation compared head to head, plus a PCA view.",
         result="Across 30 seeds, k-means++ averages 14% lower inertia than random init (48,140 vs 55,718) and is far more consistent (std 8,250 vs 14,855)."),
    dict(stem="06_neural_network_from_scratch", short="A Neural Network from Scratch",
         build="Forward pass, backpropagation and gradient descent implemented in plain NumPy, trained on two interleaving moons.",
         result="A hand-written 8-unit hidden layer reaches 96.0% test accuracy with a curved decision boundary no linear model could draw."),
    dict(stem="07_transfer_learning", short="Transfer Learning",
         build="A small CNN trained on clean digits, then its frozen features reused on the same digits shifted and rotated, with almost no labels.",
         result="With 8 labelled examples per class, transfer learning reaches 37.1% accuracy versus 16.7% training from scratch on the same tiny label set — more than double."),
    dict(stem="08_lstm_for_sequences", short="LSTM for Sequences",
         build="Windowing a synthetic sequence, training a small LSTM, and checking it against the naive \"tomorrow looks like today\" baseline.",
         result="The LSTM's one-step forecast error is 2.31x lower than naive persistence (MSE 0.030 vs 0.069)."),
    dict(stem="09_anomaly_detection_autoencoder", short="Anomaly Detection with an Autoencoder",
         build="A convolutional autoencoder trained only on normal NAB-style sensor data, with a reconstruction-error threshold set from training data alone.",
         result="With no anomaly labels used anywhere, 68 of the 69 windows it flags overlap the injected anomaly; the one false alarm is only 0.4% above the threshold."),
    dict(stem="10_transformers_self_attention", short="Transformers and Self-Attention",
         build="Scaled dot-product and multi-head attention implemented from scratch in NumPy, plus a causal mask for autoregressive generation.",
         result="In a hand-built toy sentence, the token \"it\" attends most strongly to \"animal\" (weight 0.147) — resolving the coreference with plain arithmetic, no training involved."),
    dict(stem="11_federated_learning", short="Federated Learning",
         build="An original FedAvg simulation: three clients train locally on real patient data and average their models, compared to a centrally-trained baseline.",
         result="After 15 rounds, the federated model matches the centrally-trained baseline exactly — 71.4% test accuracy for both — without the clients' data ever being pooled."),
]
NOTE = ("To run a chapter you also need <code>mlutils.py</code> and, for most chapters, a data file: "
        "the full course download has everything.")
COURSE = Course(
    name="A Tour of Machine Learning", folder="ml-tour", chapters=CHAPTERS,
    notes={ch["stem"]: NOTE for ch in CHAPTERS},
    license=('Built on material from the author\'s own MADIS Machine Learning Workshop. '
             'Chapter 04 loosely follows the structure of Prashant Banerjee\'s '
             '<a href="https://www.kaggle.com/code/prashant111/random-forest-classifier-tutorial">'
             'Random Forest Classifier Tutorial</a> on Kaggle; chapter 09 follows the general recipe of the '
             'Keras <a href="https://keras.io/examples/timeseries/timeseries_anomaly_detection/">'
             'Timeseries anomaly detection using an Autoencoder</a> example.'),
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
    hero = results[4]   # chapter 05's labelled cluster figure
    return page_head(
        "A Tour of Machine Learning",
        "Eleven standalone notebooks, one machine-learning idea each: linear regression, the calculus behind "
        "training, SVMs, random forests, K-means, a neural network from scratch, transfer learning, LSTMs, "
        "anomaly detection, transformers and federated learning — every chapter on real data, judged by its "
        "own honest result.",
    ) + f"""
<nav class="site-back col" aria-label="Site"><a href="/tutorials">&larr; All tutorials</a><a href="/">Elie Rouphael</a></nav>

<header class="hero col">
  <span class="eyebrow">Eleven notebooks · scikit-learn, NumPy, sympy, TensorFlow/Keras</span>
  <h1>A tour of machine learning</h1>
  <p class="lede">Eleven standalone notebooks, each covering one machine-learning idea start to finish, on real
  data: regression, the calculus that training is secretly doing, support vector machines, random forests,
  K-means, a neural network built from nothing but NumPy, transfer learning, LSTMs, an autoencoder that catches
  anomalies it was never shown, attention implemented by hand, and a from-scratch simulation of federated
  learning. No chapter depends on another — read the one you're curious about.</p>
  <p class="byline">Elie Rouphael · Assumes Python and a little numpy/pandas; no prior ML background needed</p>
  <div class="downloads">
    <a class="btn" href="{slug(CHAPTERS[0])}.html">Start with chapter 01</a>
    <a class="btn ghost" href="{zip_name}" download>Download the course (.zip)</a>
  </div>
</header>

<section class="wide" aria-label="Customer segments found by K-means">
  <figure class="output-figure hero-figure">
    <img src="img/05-3.png" alt="Mall customers coloured by five K-means clusters, plotted by annual income against spending score, with an X marking each cluster's centroid." width="{hero['hero_w']}" height="{hero['hero_h']}">
    <figcaption><b>One chapter's worth of "a tour" in one picture.</b> Five customer segments K-means finds unsupervised, from nothing but income and a spending score — budget-conscious high earners, careful low earners, big spenders regardless of income, and more. Every other chapter in this course looks completely different, and that's the point. From chapter 05.</figcaption>
  </figure>
</section>

<section class="col prose" aria-labelledby="approach">
  <h2 id="approach">One idea per chapter, not one story</h2>
  <p>Most courses on this site follow a single dataset or a single system from first principles to a final
  test. This one doesn't: it's a tour. Each chapter is a different classical or modern method, trained on a
  different real (or, where a download would be the only obstacle, honestly synthetic) dataset, judged by
  whatever result actually matters for that method — an R², an accuracy, an inertia, a reconstruction error, a
  round of federated averaging. What ties the eleven chapters together is a shared discipline: every number on
  every page is the notebook's own output, every comparison has an honest baseline, and every method is shown
  earning its result rather than assumed to work.</p>
  <p>This material reworks the author's own MADIS Machine Learning Workshop into runnable, standalone chapters.
  Two chapters are explicit reworkings of outside material, credited where they appear: chapter 04 (Random
  Forest) loosely follows a Kaggle tutorial's structure, and chapter 09 (Anomaly Detection) follows the general
  recipe of a well-known Keras example, both cited in the original workshop's own closing notes.</p>
</section>

<section class="col" aria-labelledby="chapters">
  <h2 id="chapters">Chapters</h2>
  <p class="section-lede">Chapters 01-02 are the foundations: fitting a line, and the calculus every model's
  training loop is quietly doing. Chapters 03-05 are the classical toolkit: SVMs, random forests, K-means.
  Chapters 06-10 are deep learning and modern architectures, built up from a from-scratch neural network to
  from-scratch self-attention. Chapter 11 closes with federated learning, simulated end to end from nothing
  the original workshop actually ran.</p>
  <ol class="chapter-list">
{chapters}
  </ol>
</section>

<section class="col prose" aria-labelledby="run">
  <h2 id="run">Running it yourself</h2>
  <p>Every chapter page shows the notebook with its outputs, so you can read the whole course here. To run and
  change the code, <a href="{zip_name}" download>download the course</a> (the notebooks, the shared
  <code>mlutils.py</code> helpers, the data and the requirements), then from its folder run:</p>
  <pre><code class="language-bash">pip install -r requirements.txt
jupyter lab                 # then open any chapter directly, in any order</code></pre>
  <p>Chapters 07-09 use TensorFlow/Keras and take longest (10-20 seconds each on a laptop); every other chapter
  runs in a few seconds. Every chapter ends with exercises, with solutions folded away.</p>
</section>

<section class="col prose" aria-labelledby="data">
  <h2 id="data">Data and sources</h2>
  <p>Chapter 01 uses a small car-emissions dataset; chapter 03 uses Fisher's iris measurements; chapter 04 uses
  a real cardiac-evaluation dataset; chapter 05 uses a mall customer-segmentation dataset; chapter 09 uses
  Numenta Anomaly Benchmark (NAB) style synthetic sensor data; chapter 11 uses the Pima Indians Diabetes
  dataset. Chapters 02, 06, 07, 08 and 10 use synthetic or built-in (scikit-learn digits) data by design, so
  the mechanism being taught stays the whole story. {COURSE.license}</p>
</section>

<footer class="col">
  <p>"A Tour of Machine Learning" by Elie Rouphael. The chapter pages are rendered from the executed notebooks;
  math by KaTeX, code highlighting by highlight.js.</p>
</footer>
""" + page_scripts()


# ---------------------------------------------------------------- cover

def make_cover(path: Path) -> None:
    """Five K-means customer clusters, drawn to read at thumbnail size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.cluster import KMeans

    course = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_COURSE
    mall = pd.read_csv(course / "data" / "Mall_Customers.csv")
    X = mall[["Annual Income (k$)", "Spending Score (1-100)"]].values
    km = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42).fit(X)

    ink = "#1d2530"
    palette = ["#2a78d6", "#eb6834", "#3f9142", "#8a949e", "#a24fb0"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor("#fbfbf8"); ax.set_facecolor("#fbfbf8")
    for c, color in zip(range(5), palette):
        sub = X[km.labels_ == c]
        ax.scatter(sub[:, 0], sub[:, 1], color=color, s=90, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color=ink, marker="X", s=420,
               linewidth=0, zorder=5)
    ax.text(X[:, 0].min() - 2, X[:, 1].max() + 6, "A Tour of ML", color=ink, fontsize=30, weight="bold")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(X[:, 0].min() - 5, X[:, 0].max() + 5)
    ax.set_ylim(X[:, 1].min() - 5, X[:, 1].max() + 12)
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
    for ch in CHAPTERS:                                   # downloads get the notebooks exactly as executed
        pass  # build_chapters already copies executed notebooks into out/notebooks/

    w, h = png_size((out / "img" / "05-3.png").read_bytes())
    results[4].update(hero_w=w, hero_h=h)
    (out / "index.html").write_text(index_page(results), encoding="utf-8")

    members = [(course / rel, rel) for rel in ("README.md", "requirements.txt", "mlutils.py", "build_notebooks.py")]
    members += [(course / "data" / f.name, f"data/{f.name}") for f in sorted((course / "data").glob("*.csv"))]
    members += [(course / f"{ch['stem']}.ipynb", f"{ch['stem']}.ipynb") for ch in CHAPTERS]
    members += [(executed / f"{ch['stem']}.ipynb", f"executed/{ch['stem']}.ipynb") for ch in CHAPTERS]
    write_zip(COURSE, members)

    make_cover(out / "cover.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
