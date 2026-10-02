"""Build the time-series course pages from the executed notebooks.

Reads the eleven executed notebooks of the course
(Understanding Time-Series/course/executed) and writes static pages into
public/tutorials/time-series/:

    index.html                                   course landing page
    01-time-in-pandas.html ... 11-capstone.html  one page per notebook
    img/, notebooks/                             extracted figures, notebooks for download
    time-series-course.zip                       notebooks, tsutils.py, data and requirements
    cover.png                                    preview image for the tutorials card

Chapter pages, style.css and course.js come from scripts/notebook_course.py. The
notebooks must already be executed (run build_notebooks.py in the course folder).

Two small adaptations happen in a staging copy, so the shared renderer stays unchanged:
chapter titles use "01 · Title" (the renderer expects "01 - Title"), and tables drawn
with pandas' Styler are turned into plain-text tables (the renderer shows text outputs).

Usage:
    pip install markdown-it-py mdit-py-plugins numpy pandas matplotlib
    python scripts/build_time_series_course.py [path/to/course]
"""

from __future__ import annotations

import html
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from notebook_course import SITE, Course, build_chapters, page_head, page_scripts, png_size, slug, write_zip

DEFAULT_COURSE = SITE.parents[1] / "Understanding Time-Series" / "course"

# `result` lines quote the notebooks' own outputs; re-check them if the notebooks change.
CHAPTERS = [
    dict(stem="01_time_in_pandas", short="Working with time in pandas",
         build="Timestamps from raw columns, index health, missing-data codes, gaps, resampling, trailing windows, lags, calendar features, long and wide tables.",
         result="Winter's smog shows up in the extremes: severe hours are several times more common, while the typical day barely changes."),
    dict(stem="02_anatomy_of_a_time_series", short="Anatomy of a time series",
         build="Trend, seasonality and remainder, additive vs. multiplicative, classical decomposition from scratch, STL, the periodogram and component strength.",
         result="Seasonal strength is 0.98 for CO₂ but only 0.47 for PM2.5, which is mostly weather-driven episodes."),
    dict(stem="03_stationarity", short="Stationarity",
         build="Random walks vs. mean reversion, half-lives, ADF and KPSS and their blind spots, test power by simulation, differencing vs. detrending.",
         result="Daily PM2.5 forgets a shock with a half-life of about one day."),
    dict(stem="04_baselines_and_evaluation", short="Baselines and honest evaluation",
         build="Six baselines, MAE, RMSE and MASE (and why not MAPE), the rolling-origin backtest, leakage, and prediction intervals checked against outcomes.",
         result="Naive wins for tomorrow and climatology from day 2: the bar every later model has to clear."),
    dict(stem="05_exponential_smoothing", short="Exponential smoothing",
         build="SES from scratch, fitting objectives, Holt, damped trend, Holt-Winters, and ETS with AIC and interval coverage.",
         result="Holt-Winters halves the best baseline's error on CO₂; on PM2.5 plain smoothing loses to the simple baselines."),
    dict(stem="06_autocorrelation_and_arma", short="Autocorrelation and ARMA",
         build="ACF and PACF from scratch, AR, MA and ARMA fingerprints, AIC/BIC, Ljung–Box checks and mean-reverting forecasts.",
         result="A seasonal profile plus AR(2) gives the best day-1 forecast so far, with 80% intervals that cover 78–80%."),
    dict(stem="07_arima_sarima_sarimax", short="ARIMA, SARIMA and SARIMAX",
         build="Differencing inside the model, the airline model, residual diagnostics, auto_arima, cross-correlation, and weather as inputs: oracle vs. honest.",
         result="Weather known today cuts the day-1 error to 40.9 µg/m³; perfect future weather would reach 34.2."),
    dict(stem="08_prophet", short="Prophet",
         build="Fourier seasonality, changepoints, holiday effects, extra regressors, built-in cross-validation, logistic growth and a Prophet + AR hybrid.",
         result="Spring Festival fireworks lift New Year's Day PM2.5 by about 55%."),
    dict(stem="09_machine_learning", short="Machine learning for forecasting",
         build="Forecasting as a table, honest features and training cut-offs, direct vs. recursive, ridge vs. gradient boosting, three leakage traps measured.",
         result="Ridge reaches 54.1 µg/m³ at day 2, the best so far; a leaky feature fakes 28."),
    dict(stem="10_deep_learning", short="Deep learning",
         build="Honest windows, training-only scaling, early stopping, MLP, SimpleRNN and LSTM, seed ensembles, and an hourly next-24-hours task.",
         result="Identical networks differ by several µg/m³ between random seeds; on hourly data the LSTM beats ridge from 6 hours ahead."),
    dict(stem="11_capstone", short="Capstone: the final test",
         build="Finalists fixed in advance, the locked 2015 year opened once, skill scores, a smog-alert forecast, and the same methods on temperature.",
         result="Ridge is 36% better than climatology for tomorrow, and a day-ahead alert catches 65% of heavy-pollution days."),
]
NOTE = ("To run a chapter you also need <code>tsutils.py</code> and the data file: "
        "the full course download has everything.")
COURSE = Course(name="Time-series course", folder="time-series", chapters=CHAPTERS,
                notes={ch["stem"]: NOTE for ch in CHAPTERS})


# ---------------------------------------------------------------- staging

def styler_to_text(table_html: str) -> str:
    """A pandas Styler's HTML table as a fixed-width text table."""
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)]
        rows.append(cells)
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    widths = [max(len(r[i]) for r in rows) for i in range(width)]
    lines = ["  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))).rstrip()
             for r in rows]
    return "\n".join(lines) + "\n"


def stage(executed: Path, staging: Path) -> None:
    """Copy the executed notebooks, adapting titles and Styler outputs for the renderer."""
    for ch in CHAPTERS:
        nb = json.loads((executed / f"{ch['stem']}.ipynb").read_text(encoding="utf-8"))
        first = nb["cells"][0]["source"]
        first[0] = re.sub(r"^#\s*(\d+)\s*·\s*", r"# \1 - ", first[0])
        for cell in nb["cells"]:
            for out in cell.get("outputs", []):
                data = out.get("data", {})
                plain = "".join(data.get("text/plain", ""))
                if plain.startswith("<pandas.io.formats.style.Styler") and "text/html" in data:
                    data["text/plain"] = styler_to_text("".join(data["text/html"]))
        (staging / f"{ch['stem']}.ipynb").write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")


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
          <span class="r"><span class="lbl">You find</span> {html.escape(ch['result'])}</span>
        </span>
      </a>
    </li>""")
    chapters = "\n".join(items)
    zip_name = f"{COURSE.folder}-course.zip"
    hero = results[-1]
    return page_head(
        "Understanding Time Series",
        "An eleven-chapter course in time-series analysis and forecasting on one real dataset, hourly air "
        "pollution in Beijing: from pandas and decomposition to ARIMA, Prophet, machine learning and deep "
        "learning, all judged by the same honest backtest.",
    ) + f"""
<nav class="site-back col" aria-label="Site"><a href="/tutorials">&larr; All tutorials</a><a href="/">Elie Rouphael</a></nav>

<header class="hero col">
  <span class="eyebrow">Time series · Eleven notebooks · pandas, statsmodels, Prophet, scikit-learn + Keras</span>
  <h1>Understanding time series</h1>
  <p class="lede">A practical course in time-series analysis and forecasting that follows one real dataset from start to finish: six years of hourly air-quality measurements from Beijing. Every method, from moving averages to neural networks, answers the same question on the same data, is checked against simple baselines, and is finally tested on a year that stayed locked until the last chapter.</p>
  <p class="byline">Elie Rouphael · Assumes Python and a little pandas; no prior time-series knowledge needed</p>
  <div class="downloads">
    <a class="btn" href="{slug(CHAPTERS[0])}.html">Start with chapter 01</a>
    <a class="btn ghost" href="{zip_name}" download>Download the course (.zip)</a>
  </div>
</header>

<section class="wide" aria-label="Day-ahead forecasts in the final test">
  <figure class="output-figure hero-figure">
    <img src="img/11-2.png" alt="Daily PM2.5 in Beijing from October to December 2015, with a one-day-ahead model forecast and the naive forecast. The naive line is the actual series shifted by a day; the model often catches the sharp clean-ups on the day they happen but under-forecasts the biggest peaks." width="{hero['hero_w']}" height="{hero['hero_h']}">
    <figcaption><b>What the course builds towards.</b> One-day-ahead forecasts of daily PM2.5 through the smoggy end of 2015, a year no model or decision had seen. The naive forecast ("same as today") is always a day late. The model often catches the sudden clean-ups, which the evening wind announces, but it can't see build-ups coming. From chapter 11.</figcaption>
  </figure>
</section>

<section class="col prose" aria-labelledby="question">
  <h2 id="question">One question, asked honestly</h2>
  <p>From chapter 04 on, every model answers the same question: <em>given daily PM2.5 up to today, forecast each of the next seven days.</em> They're all scored the same way, with a rolling-origin backtest that only ever lets a model see the past:</p>
  <div class="arch">
    <div class="arch-card"><span class="eq-name">2010–2013 · history</span><p>Always available for fitting. Every model learns from it.</p></div>
    <div class="arch-card"><span class="eq-name">2014 · development</span><p>Where models are compared and tuned, one forecast origin per day, on a shared scoreboard.</p></div>
    <div class="arch-card accent"><span class="eq-name">2015 · locked</span><p>Opened once, in the capstone, after every decision was made. The honest final test.</p></div>
  </div>
  <p>Along the way the course keeps asking what makes a forecast good: beating naive and seasonal baselines, avoiding leakage, checking prediction intervals against what actually happened, and knowing where the limit is. Here, with perfect weather forecasts the errors would drop sharply, and chapter 07 measures by how much.</p>
</section>

<section class="col" aria-labelledby="chapters">
  <h2 id="chapters">Chapters</h2>
  <p class="section-lede">Chapters 01–03 build the foundations: handling time in pandas, decomposing a series, and stationarity. Chapter 04 sets up the baselines and the backtest. Chapters 05–10 each add a model family: exponential smoothing, ARMA, SARIMAX with weather, Prophet, machine learning and deep learning. Chapter 11 opens the locked year.</p>
  <ol class="chapter-list">
{chapters}
  </ol>
</section>

<section class="col prose" aria-labelledby="run">
  <h2 id="run">Running it yourself</h2>
  <p>Every chapter page shows the notebook with its outputs, so you can read the whole course here. To run and change the code, <a href="{zip_name}" download>download the course</a> (the notebooks, the shared <code>tsutils.py</code> helpers, the data and the requirements), then from its folder run:</p>
  <pre><code class="language-bash">pip install -r requirements.txt
jupyter lab                 # then open 01_time_in_pandas.ipynb</code></pre>
  <p>Run the chapters in order: from chapter 04 on, each one adds its scores to a shared scoreboard that later chapters compare against. Most chapters run in a minute or two on a laptop; the Prophet, deep-learning and capstone notebooks take a few minutes. Every chapter ends with exercises, with solutions folded away.</p>
</section>

<section class="col prose" aria-labelledby="data">
  <h2 id="data">Data</h2>
  <p>Hourly PM2.5 and weather for Beijing, 2010–2015: Chen, Song (2016), <em>PM2.5 Data of Five Chinese Cities</em>, UCI Machine Learning Repository, <a href="https://doi.org/10.24432/C52K58">doi:10.24432/C52K58</a>, licensed CC BY 4.0. The Mauna Loa CO₂ record used for comparison is by Keeling &amp; Whorf (Scripps Institution of Oceanography), public domain, and ships with statsmodels. For more depth on the statistical methods, Hyndman &amp; Athanasopoulos's <em>Forecasting: Principles and Practice</em> is free online.</p>
</section>

<footer class="col">
  <p>Time-series course by Elie Rouphael. The chapter pages are rendered from the executed notebooks; math by KaTeX, code highlighting by highlight.js.</p>
</footer>
""" + page_scripts()


# ---------------------------------------------------------------- cover

def make_cover(path: Path, course: Path) -> None:
    """Daily PM2.5 through the end of 2015 against the heavy-pollution line, drawn to read at thumbnail size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    raw = pd.read_csv(course / "data" / "beijing_pm25.csv")
    stamps = pd.to_datetime(raw[["year", "month", "day", "hour"]])
    daily = raw.set_index(stamps)["PM_US Post"].resample("D").mean()["2015-09-15":"2015-12-31"]

    ink, blue, orange = "#1d2530", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor("#fbfbf8"); ax.set_facecolor("#fbfbf8")
    ax.fill_between(daily.index, daily.to_numpy(), color=blue, alpha=0.18, lw=0)
    ax.plot(daily.index, daily.to_numpy(), color=blue, lw=3)
    ax.axhline(150, color=orange, lw=3, ls=(0, (4, 2)))
    ax.text(daily.index[4], 162, "heavy pollution", color=orange, fontsize=24, weight="bold", zorder=5,
            bbox=dict(facecolor="#fbfbf8", edgecolor="none", pad=3))
    ax.text(daily.index[4], daily.max() * 0.9, "PM2.5 · Beijing", color=ink, fontsize=28, weight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b9bfb6")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, daily.max() * 1.02)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    course = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_COURSE
    executed = course / "executed"
    if not executed.is_dir():
        sys.exit(f"Executed notebooks not found in {executed}. Run build_notebooks.py in the course folder first.")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        stage(executed, staging)
        results = build_chapters(COURSE, staging)

    out = COURSE.out
    for ch in CHAPTERS:                                  # downloads get the notebooks exactly as executed
        shutil.copy2(executed / f"{ch['stem']}.ipynb", out / "notebooks" / f"{ch['stem']}.ipynb")
    w, h = png_size((out / "img" / "11-2.png").read_bytes())
    results[-1].update(hero_w=w, hero_h=h)
    (out / "index.html").write_text(index_page(results), encoding="utf-8")

    members = [(course / rel, rel) for rel in ("README.md", "requirements.txt", "tsutils.py", "build_notebooks.py",
                                                "data/README.md", "data/beijing_pm25.csv")]
    members += [(course / f"{ch['stem']}.ipynb", f"{ch['stem']}.ipynb") for ch in CHAPTERS]
    members += [(executed / f"{ch['stem']}.ipynb", f"executed/{ch['stem']}.ipynb") for ch in CHAPTERS]
    write_zip(COURSE, members)

    make_cover(out / "cover.png", course)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
