"""Build the generated assets of the "Rescheduling a Flexible Job Shop" tutorial.

The page itself (public/tutorials/jobshop-digital-twin/index.html) is hand-written; this script
produces the files next to it, all from the jobshop-digital-twin repository:

    jobshop-digital-twin-code.zip   the package, scripts, tests, trained DQN weights, README,
                                    LICENSE, requirements and the raw results (results.md, runs.csv)
    tardiness.png                   mean tardiness per method at both disturbance levels, with the
                                    95 % bootstrap intervals, read from results/runs.csv
    gantt_breakdowns.png            the repository's executed CP-SAT schedule under breakdowns
    cover.png                       preview image for the tutorials card

Usage:  python scripts/build_jobshop_course.py [path/to/jobshop-digital-twin]
Needs numpy and matplotlib; the default path is the author's local working copy.
"""
from __future__ import annotations

import csv
import shutil
import sys
import zipfile
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
OUT = SITE / "public" / "tutorials" / "jobshop-digital-twin"
DEFAULT_REPO = Path(r"C:/Users/rouphaee/Documents/PostDoc_v2/jobshop-digital-twin")
ZIP_NAME = "jobshop-digital-twin-code.zip"
ARCHIVE_FOLDER = "jobshop-digital-twin"

MEMBERS = [
    "README.md", "LICENSE", "requirements.txt", "pyproject.toml",
    "results/results.md", "results/runs.csv",
]
GLOBS = ["jobshop/*.py", "scripts/*.py", "tests/*.py", "models/*.pt", "models/training.json"]

METHODS = ["FIFO", "SPT", "EDD", "SST", "DQN", "Greedy replan", "CP-SAT"]
LEVELS = ["no breakdowns", "frequent breakdowns"]


def write_zip(repo: Path) -> Path:
    path = OUT / ZIP_NAME
    files = [repo / rel for rel in MEMBERS]
    for pattern in GLOBS:
        files += sorted(repo.glob(pattern))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for src in files:
            if not src.exists():
                print(f"  (skipped, not found: {src.relative_to(repo)})")
                continue
            z.write(src, f"{ARCHIVE_FOLDER}/{src.relative_to(repo).as_posix()}")
    return path


def tardiness_table(repo: Path) -> dict[tuple[str, str], tuple[float, float, float]]:
    sys.path.insert(0, str(repo))
    import numpy as np
    from jobshop.metrics import bootstrap_ci  # the interval the repository reports

    rows = list(csv.DictReader(open(repo / "results" / "runs.csv", encoding="utf-8")))
    out = {}
    for level in LEVELS:
        for m in METHODS:
            v = np.array([float(r["mean_tardiness"]) for r in rows if r["level"] == level and r["method"] == m])
            low, high = bootstrap_ci(v)
            out[(level, m)] = (float(v.mean()), low, high)
    return out


def make_chart(table, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ink, muted, rule, bg = "#16202a", "#6b7581", "#dcdfd8", "#ffffff"
    colours = {"rule": "#a7b0ba", "DQN": "#8f6fd1", "Greedy replan": "#7fb0e6", "CP-SAT": "#2a78d6"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, sharey=False)
    fig.patch.set_facecolor(bg)
    for ax, level in zip(axes, LEVELS):
        y = np.arange(len(METHODS))[::-1]
        for yi, m in zip(y, METHODS):
            mean, low, high = table[(level, m)]
            c = colours.get(m, colours["rule"])
            ax.barh(yi, mean, color=c, height=0.62)
            ax.errorbar(mean, yi, xerr=[[mean - low], [high - mean]], fmt="none", ecolor=ink, elinewidth=1, capsize=3)
            ax.text(high + 1, yi, f"{mean:.1f}", va="center", fontsize=9, color=ink)
        ax.set_yticks(y, METHODS, fontsize=9.5, color=ink)
        ax.set_title(level.capitalize(), fontsize=11, color=ink, loc="left")
        ax.set_xlabel("mean tardiness per job (min)", fontsize=9, color=muted)
        ax.set_xlim(0, max(table[(level, m)][2] for m in METHODS) * 1.18)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(rule)
        ax.tick_params(colors=muted, labelsize=8.5)
        ax.grid(axis="x", color=rule, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


def make_cover(repo: Path, table, path: Path) -> None:
    """The breakdowns level of the chart, drawn large to read at card size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ink, rule, bg = "#16202a", "#dcdfd8", "#fbfbf8"
    level = "frequent breakdowns"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    y = np.arange(len(METHODS))[::-1]
    for yi, m in zip(y, METHODS):
        mean, low, high = table[(level, m)]
        c = "#2a78d6" if m == "CP-SAT" else ("#7fb0e6" if m == "Greedy replan" else ("#8f6fd1" if m == "DQN" else "#b8c0c8"))
        ax.barh(yi, mean, color=c, height=0.66)
        ax.errorbar(mean, yi, xerr=[[mean - low], [high - mean]], fmt="none", ecolor=ink, elinewidth=1.2, capsize=4)
    ax.set_yticks(y, METHODS, fontsize=15, color=ink)
    ax.tick_params(axis="x", labelsize=11, colors="#6b7581")
    ax.set_xlabel("mean tardiness per job under breakdowns (min)", fontsize=12, color="#6b7581")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(rule)
    ax.grid(axis="x", color=rule, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


def main() -> None:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPO
    OUT.mkdir(parents=True, exist_ok=True)
    z = write_zip(repo)
    print(f"  {z.name}: {len(zipfile.ZipFile(z).namelist())} members")
    table = tardiness_table(repo)
    make_chart(table, OUT / "tardiness.png")
    make_cover(repo, table, OUT / "cover.png")
    shutil.copy(repo / "results" / "gantt_breakdowns.png", OUT / "gantt_breakdowns.png")
    for (level, m), (mean, low, high) in table.items():
        if m in ("EDD", "SST", "DQN", "Greedy replan", "CP-SAT"):
            print(f"  {level:20s} {m:14s} {mean:5.1f} [{low:.1f}, {high:.1f}]")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
