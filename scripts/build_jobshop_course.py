"""Build the course "Which Job Should the Machine Do Next?" (public/tutorials/jobshop-digital-twin/).

The chapters are Markdown in courses/jobshop-digital-twin/chapters/<id>.md, in the dialect of
scripts/course_md.py, listed in courses/jobshop-digital-twin/course.yaml. This script renders:

    <slug>.html                one page per chapter
    index.html                 the landing page (from courses/jobshop-digital-twin/index.md)
    notation.html              the notation and the course's workshop (from notation.md)
    concepts.html              every definition box and code walk, with its chapter and part
    tardiness.png              mean tardiness per method at both levels, from the repository's runs.csv
    gantt_breakdowns.png       the repository's executed CP-SAT schedule under breakdowns
    jobshop-digital-twin-course.zip   the repository (package, scripts, tests, trained agents, results)
                               and code/, one script per chapter printing every number on its page

The chapter demos are hand-written: interactive/jobshop-lib.js (shared) and interactive/<id>.js.
cover.png (the tutorials card) is a static file and is not redrawn here.

Usage:
    python scripts/build_jobshop_course.py                 # everything
    python scripts/build_jobshop_course.py --only a02      # one chapter page only (safe while others work)
Needs markdown-it-py, mdit-py-plugins, pyyaml; the charts need numpy and matplotlib.
"""
from __future__ import annotations

import csv
import html
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import course_md  # noqa: E402
import build_mpc_course as base  # noqa: E402  (its page helpers; SRC/OUT are re-pointed below)
from course_md import MD, chapter_page, load_course, mark_heading_ids, page_head, page_scripts, postprocess, render_chapter_file, resolve_refs  # noqa: E402

SITE = Path(__file__).resolve().parents[1]
SRC = SITE / "courses" / "jobshop-digital-twin"
OUT = SITE / "public" / "tutorials" / "jobshop-digital-twin"
base.SRC, base.OUT = SRC, OUT
GITHUB = "https://github.com/ElieRouphael/jobshop-digital-twin"

METHODS = ["FIFO", "SPT", "EDD", "SST", "DQN", "Greedy replan", "CP-SAT"]
PLAIN = {"FIFO": "First come, first served", "SPT": "Shortest job first", "EDD": "Most urgent first",
         "SST": "Shortest setup first", "DQN": "Self-learning agent", "Greedy replan": "Quick re-planning",
         "CP-SAT": "Optimiser re-planning"}


def build_chapter(course, ch) -> bool:
    src = SRC / "chapters" / f"{ch.id}.md"
    if not src.exists():
        return False
    r = render_chapter_file(src, course, ch)
    for key in ("lede", "result", "build", "needs"):
        if r.meta.get(key):
            setattr(ch, key, r.meta[key])
    ch.parts = r.meta.get("parts", [])
    (OUT / f"{ch.slug}.html").write_text(chapter_page(course, ch, r), encoding="utf-8")
    for want in ([f"interactive/{ch.id}.js"] if ch.script else []) + ([f"code/{ch.id}_numbers.py"] if ch.numbers else []):
        if not (OUT / want).exists():
            print(f"  warning: {ch.id} has no {want}")
    return True


def index_page(course, intro: dict) -> str:
    zip_name = f"{course.folder}-course.zip"
    parts_html = []
    for part in course.parts:
        chapters = course.in_part(part["key"])
        cards = "\n".join(base.chapter_card(ch) for ch in chapters)
        parts_html.append(f"""
<section class="col" aria-labelledby="part-{part['key']}">
  <h2 id="part-{part['key']}">Part {part['key']} · {html.escape(part['title'])}</h2>
  <p class="section-lede">{part['lede']}</p>
  <ol class="chapter-list">
{cards}
  </ol>
</section>""")
    first = course.chapters[0]
    return page_head(course.name, intro["description"]) + f"""
<nav class="site-back col" aria-label="Site"><a href="/tutorials">&larr; All tutorials</a><a href="/">{html.escape(course.author)}</a></nav>

<header class="hero col">
  <span class="eyebrow">{intro['eyebrow']}</span>
  <h1>{html.escape(course.name)}</h1>
  <p class="lede">{intro['lede']}</p>
  <p class="byline">{html.escape(course.author)} · {intro['assumes']}</p>
  <div class="downloads">
    <a class="btn" href="{first.slug}.html">Start with chapter {first.label}</a>
    <a class="btn ghost" href="{GITHUB}">The repository on GitHub</a>
    <a class="btn ghost" href="{zip_name}" download>Download the course (.zip)</a>
  </div>
</header>

<div class="col prose">
{intro['body']}
</div>
{''.join(parts_html)}

<section class="col prose" aria-labelledby="run">
  <h2 id="run">Running it yourself</h2>
  <p>Every chapter runs its demos in the browser and quotes numbers computed in Python. To reproduce them,
  <a href="{zip_name}" download>download the course</a>: the repository's <code>jobshop</code> package with its
  tests, scripts, trained agents and results, and in <code>code/</code> one script per chapter that prints every
  number quoted on that chapter's page. From the unzipped folder:</p>
  <pre><code class="language-bash">pip install -r requirements.txt
python -m pytest                      # the repository's 64 tests
python code/c01_numbers.py            # every number on chapter C1's page</code></pre>
</section>

<section class="col prose" aria-labelledby="more">
  <h2 id="more">Reference pages</h2>
  <ul>
    <li><a href="notation.html">Notation and the course's workshop</a>: every symbol, the method names, the worked example and the workshop's data.</li>
    <li><a href="concepts.html">Concept index</a>: each concept with the chapter and part that defines it, and each function of the repository a chapter reads line by line.</li>
  </ul>
</section>

<footer class="col">
  <p>{html.escape(course.name)}, by {html.escape(course.author)}. {course.license} Everything is synthetic: no data from a
  real plant. Math by KaTeX, code highlighting by highlight.js.</p>
</footer>
""" + page_scripts([])


def write_charts(repo: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    def bootstrap_ci(values, confidence=0.95, resamples=10_000, seed=0):
        """Same as jobshop/metrics.py:bootstrap_ci (percentile bootstrap of the mean), without importing SimPy."""
        values = np.asarray(values, dtype=float)
        samples = np.random.default_rng(seed).choice(values, size=(resamples, len(values)), replace=True)
        alpha = (1.0 - confidence) / 2.0
        low, high = np.quantile(samples.mean(axis=1), [alpha, 1.0 - alpha])
        return float(low), float(high)

    rows = list(csv.DictReader(open(repo / "results" / "runs.csv", encoding="utf-8")))
    ink, muted, rule, bg = "#16202a", "#6b7581", "#dcdfd8", "#ffffff"
    colours = {"DQN": "#8f6fd1", "Greedy replan": "#7fb0e6", "CP-SAT": "#2a78d6"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    fig.patch.set_facecolor(bg)
    for ax, level, title in zip(axes, ("no breakdowns", "frequent breakdowns"), ("Calm shop (no breakdowns)", "Frequent breakdowns")):
        stats = {}
        for m in METHODS:
            v = np.array([float(r["mean_tardiness"]) for r in rows if r["level"] == level and r["method"] == m])
            stats[m] = (v.mean(), *bootstrap_ci(v))
        y = np.arange(len(METHODS))[::-1]
        for yi, m in zip(y, METHODS):
            mean, lo, hi = stats[m]
            ax.barh(yi, mean, color=colours.get(m, "#a7b0ba"), height=0.62)
            ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt="none", ecolor=ink, elinewidth=1, capsize=3)
            ax.text(hi + 1, yi, f"{mean:.1f}", va="center", fontsize=9, color=ink)
        ax.set_yticks(y, [PLAIN[m] for m in METHODS], fontsize=9.5, color=ink)
        ax.set_title(title, fontsize=11, color=ink, loc="left")
        ax.set_xlabel("mean tardiness per job (min)", fontsize=9, color=muted)
        ax.set_xlim(0, max(s[2] for s in stats.values()) * 1.18)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(rule)
        ax.tick_params(colors=muted, labelsize=8.5)
        ax.grid(axis="x", color=rule, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "tardiness.png", facecolor=bg); plt.close(fig)
    shutil.copy(repo / "results" / "gantt_breakdowns.png", OUT / "gantt_breakdowns.png")


def write_zip(course, repo: Path) -> None:
    path = OUT / f"{course.folder}-course.zip"
    members = [(repo / rel, rel) for rel in ("README.md", "LICENSE", "requirements.txt", "pyproject.toml",
                                             "results/results.md", "results/runs.csv")]
    for pattern in ("jobshop/*.py", "scripts/*.py", "tests/*.py", "models/*.pt", "models/training.json"):
        members += [(p, p.relative_to(repo).as_posix()) for p in sorted(repo.glob(pattern))]
    members += [(p, f"code/{p.name}") for p in sorted((OUT / "code").glob("*.py"))]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in members:
            if src.exists():
                z.write(src, f"{course.folder}-course/{arc}")
    print(f"  {path.name}: {len(members)} members")


def main() -> None:
    args = sys.argv[1:]
    course = load_course(SRC / "course.yaml")
    course_md.ASSET_VERSION = time.strftime("%Y%m%d%H%M%S")
    (OUT / "code").mkdir(parents=True, exist_ok=True)
    (OUT / "interactive").mkdir(parents=True, exist_ok=True)
    if args[:1] == ["--only"]:
        for cid in args[1:]:
            ok = build_chapter(course, course.chapter(cid))
            print(f"  {cid}: " + ("built " + str(OUT / (course.chapter(cid).slug + ".html")) if ok else "no source yet"))
        return
    repo = course.package_root
    built = [ch.id for ch in course.chapters if build_chapter(course, ch)]
    missing = [ch.id for ch in course.chapters if ch.id not in built]
    print(f"  built {len(built)} chapters" + (f"; no source yet for {', '.join(missing)}" if missing else ""))
    intro_md = (SRC / "index.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", intro_md, flags=re.S)
    intro = yaml.safe_load(m[1])
    intro["body"] = postprocess(MD.render(mark_heading_ids(resolve_refs(intro_md[m.end():], course, course.chapters[0])), {}))
    (OUT / "index.html").write_text(index_page(course, intro), encoding="utf-8")
    if (SRC / "notation.md").exists():
        (OUT / "notation.html").write_text(base.simple_page(
            course, SRC / "notation.md", "Notation and the course's workshop",
            "Every symbol the course uses, the method names, the worked example and the data of the course's workshop.",
            [f"interactive/{course.lib}"]), encoding="utf-8")
    (OUT / "concepts.html").write_text(base.concepts_page(course), encoding="utf-8")
    if repo and repo.exists():
        write_charts(repo)
        write_zip(course, repo)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
