"""Build the downloadable code archive and the cover image for the "Satellite Pass Planning from Scratch" tutorial.

The chapter pages of this course are hand-built interactive HTML (public/tutorials/satellite-pass-planning/*.html
with interactive/*.js); this script only produces the two generated assets next to them:

    satellite-pass-planning-course.zip   the satplan repository's code (package, benchmark script, tests, README,
                                         requirements) and the course's scripts (code/*.py), which reproduce every
                                         number quoted on the chapter pages
    cover.png                            preview image for the tutorials card

Usage:  python scripts/build_satellite_course.py [path/to/satellite-pass-planning]
The default path is the author's local working copy of the public repository; the cover needs numpy and
matplotlib, and imports satplan from that repository.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
OUT = SITE / "public" / "tutorials" / "satellite-pass-planning"
DEFAULT_REPO = Path(r"C:/Users/rouphaee/Documents/PostDoc_v2/satellite-pass-planning")
ZIP_NAME = "satellite-pass-planning-course.zip"
ARCHIVE_FOLDER = "satellite-pass-planning-course"

MEMBERS = [
    "README.md", "LICENSE", "requirements.txt", "pyproject.toml",
    "satplan/__init__.py", "satplan/geometry.py", "satplan/scenario.py", "satplan/instance.py",
    "satplan/checker.py", "satplan/greedy.py", "satplan/cpsat.py", "satplan/milp.py", "satplan/rolling.py",
    "scripts/run_benchmark.py", "tests/test_planning.py",
    "results/results.md", "results/benchmark.json",
]


def write_zip(repo: Path) -> Path:
    """Zip the repository's code and the course scripts (no virtual environment, no git metadata)."""
    path = OUT / ZIP_NAME
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in MEMBERS:
            src = repo / rel
            if src.exists():
                z.write(src, f"{ARCHIVE_FOLDER}/{rel}")
            else:
                print(f"  (skipped, not found: {rel})")
        for src in sorted((OUT / "code").glob("*.py")):   # one script per chapter, plus the instance export
            z.write(src, f"{ARCHIVE_FOLDER}/code/{src.name}")
        # Some chapter scripts also write the page's data file to ../interactive/; give them the folder.
        z.writestr(f"{ARCHIVE_FOLDER}/interactive/", "")
    return path


def make_cover(repo: Path, path: Path) -> None:
    """Ground tracks of the small instance's four satellites over one revolution, with its targets, drawn to read at thumbnail size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(repo))
    from satplan.geometry import OMEGA_EARTH
    from satplan.scenario import MARITIME_AREAS, ScenarioConfig, make_constellation, make_targets

    cfg = ScenarioConfig(n_sats=4, n_targets=120, horizon_h=6.0, seed=0)
    orbits = make_constellation(cfg)
    lat, lon, _, weight, _ = make_targets(cfg)
    t = np.arange(0, 100 * 60 + 1, 10.0)   # about one revolution (95.5 min)

    ink, blue, warm, grey, bg = "#1d2530", "#2a78d6", "#eb6834", "#c9ccc4", "#fbfbf8"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    for g in range(-150, 181, 30):
        ax.axvline(g, color=grey, lw=0.5, zorder=0)
    for g in range(-60, 91, 30):
        ax.axhline(g, color=grey, lw=0.5, zorder=0)
    for _, la0, la1, lo0, lo1 in MARITIME_AREAS:
        ax.add_patch(plt.Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fc=blue, alpha=0.08, ec="none", zorder=1))
    for k, o in enumerate(orbits):
        r = o.position_eci(t)
        la = np.degrees(np.arcsin(r[:, 2] / np.linalg.norm(r, axis=1)))
        lo = (np.degrees(np.arctan2(r[:, 1], r[:, 0]) - OMEGA_EARTH * t) + 180) % 360 - 180
        breaks = np.flatnonzero(np.abs(np.diff(lo)) > 180) + 1
        for la_s, lo_s in zip(np.split(la, breaks), np.split(lo, breaks)):
            ax.plot(lo_s, la_s, color=blue, lw=2.2, alpha=0.85, zorder=2)
    ax.scatter(lon, lat, s=6 + 5 * weight, color=warm, alpha=0.85, edgecolor="none", zorder=3)
    ax.text(-172, 84, "Satellite Pass\nPlanning", color=ink, fontsize=27, weight="bold", va="top",
            bbox=dict(fc=bg, ec="none", alpha=0.85, pad=6), zorder=4)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_REPO
    if not (repo / "satplan").is_dir():
        sys.exit(f"Repository not found at {repo}")
    OUT.mkdir(parents=True, exist_ok=True)
    z = write_zip(repo)
    make_cover(repo, OUT / "cover.png")
    print(f"Wrote {z} ({z.stat().st_size // 1024} KB) and {OUT / 'cover.png'}")


if __name__ == "__main__":
    main()
