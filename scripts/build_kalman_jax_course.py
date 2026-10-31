"""Build the downloadable code archive and the cover image for the "Kalman Filters with JAX" tutorial.

The chapter pages of this course are hand-built interactive HTML (public/tutorials/kalman-jax/*.html with
interactive/*.js); this script only produces the two generated assets next to them:

    kalman-jax-course.zip   the JAX repository's code (package, scripts, tests, README, requirements, data notes),
                            the vehicle-tracking repository (chapter 01), the C program, original and corrected (chapter 05),
                            and Part I's scripts (code/j1_numbers.py ... j6_numbers.py, under part1/, with code/requirements.txt)
    cover.png               preview image for the tutorials card

Usage:  python scripts/build_kalman_jax_course.py [path/to/Kalman-Filter-with-JAX]
The default path is the author's local working copy of the public repository.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
OUT = SITE / "public" / "tutorials" / "kalman-jax"
WORKSHOPS = Path(r"C:/Users/elier/OneDrive/Desktop/Workshops")
DEFAULT_REPO = WORKSHOPS / "Kalman Filter with JAX"
VEHICLE_REPO = WORKSHOPS / "kalman_vehicle_tracking"            # chapter 01 (github.com/ElieRouphael/kalman_vehicle_tracking)
C_REPO = WORKSHOPS / "First-Order-Filter-using-C"               # chapter 05 (github.com/ElieRouphael/First-Order-Filter-using-C)
ZIP_NAME = "kalman-jax-course.zip"
ARCHIVE_FOLDER = "kalman-jax-course"

MEMBERS = [
    "README.md", "requirements.txt", "data/README.md",
    "Kalman Filter with JAX.py", "Kalman Filter with DL.py", "Air Quality with JAX.py",
    "kalman_jax/__init__.py", "kalman_jax/filters.py", "kalman_jax/learned_dynamics.py",
    "kalman_jax/forecasters.py", "kalman_jax/air_quality.py",
    "tests/test_filters.py", "tests/test_air_quality.py",
]
EXTRA_MEMBERS = [  # (source path, path inside the archive)
    (VEHICLE_REPO / "README.md", "vehicle/README.md"), (VEHICLE_REPO / "requirements.txt", "vehicle/requirements.txt"),
    (VEHICLE_REPO / "kalman/__init__.py", "vehicle/kalman/__init__.py"), (VEHICLE_REPO / "kalman/kalman_filter.py", "vehicle/kalman/kalman_filter.py"),
    (VEHICLE_REPO / "tracker/vehicle_tracker.py", "vehicle/tracker/vehicle_tracker.py"), (VEHICLE_REPO / "data/simulate_data.py", "vehicle/data/simulate_data.py"),
    (VEHICLE_REPO / "data/example_data.csv", "vehicle/data/example_data.csv"),
    (C_REPO / "README.md", "first-order-c/README.md"), (C_REPO / "TD3_original.c", "first-order-c/TD3_original.c"),
    (C_REPO / "TD3.c", "first-order-c/TD3.c"), (C_REPO / "enregistrement-bruit.txt", "first-order-c/enregistrement-bruit.txt"),
    (C_REPO / "sortie.txt", "first-order-c/sortie.txt"),
]


def write_zip(repo: Path) -> Path:
    """Zip the repository's code (no data CSV, no virtual environment, no git metadata)."""
    path = OUT / ZIP_NAME
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in MEMBERS:
            src = repo / rel
            if src.exists():
                z.write(src, f"{ARCHIVE_FOLDER}/{rel}")
            else:
                print(f"  (skipped, not found: {rel})")
        for src, arc in EXTRA_MEMBERS:
            if src.exists():
                z.write(src, f"{ARCHIVE_FOLDER}/{arc}")
            else:
                print(f"  (skipped, not found: {src})")
        for src in sorted((OUT / "code").glob("j*_numbers.py")):     # Part I: one script per chapter
            z.write(src, f"{ARCHIVE_FOLDER}/part1/{src.name}")
        z.write(OUT / "code" / "requirements.txt", f"{ARCHIVE_FOLDER}/part1/requirements.txt")   # the versions they were checked with
    return path


def make_cover(path: Path) -> None:
    """A constant-velocity track, its noisy measurements and the Kalman estimate, drawn to read at thumbnail size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(42)
    dt, n = 0.1, 160
    F = np.array([[1, dt], [0, 1]]); H = np.array([[1.0, 0.0]])
    Q = 0.01 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]); R = np.array([[0.25]])
    x = np.array([0.0, 1.0]); xs, zs = [], []
    L = np.linalg.cholesky(Q)
    for _ in range(n):
        x = F @ x + L @ rng.standard_normal(2); xs.append(x); zs.append(x[0] + 0.5 * rng.standard_normal())
    xs, zs = np.array(xs), np.array(zs)
    est, P, m = [], np.eye(2), np.zeros(2)
    for z in zs:
        m, P = F @ m, F @ P @ F.T + Q
        S = H @ P @ H.T + R; K = P @ H.T / S
        m = m + (K * (z - H @ m)).ravel(); IKH = np.eye(2) - K @ H; P = IKH @ P @ IKH.T + K @ R @ K.T
        est.append(m.copy())
    est = np.array(est); t = np.arange(n) * dt

    ink, blue, grey = "#1d2530", "#2a78d6", "#9a9893"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor("#fbfbf8"); ax.set_facecolor("#fbfbf8")
    ax.scatter(t, zs, s=22, color=grey, alpha=0.8, edgecolor="none")
    ax.plot(t, xs[:, 0], color=ink, lw=2.2)
    ax.plot(t, est[:, 0], color=blue, lw=2.6)
    ax.text(0.3, xs[:, 0].max() + 2.4, "Kalman Filters\nwith JAX", color=ink, fontsize=26, weight="bold", va="top")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.2, t[-1] + 0.2)
    ax.set_ylim(min(zs.min(), xs[:, 0].min()) - 1.0, xs[:, 0].max() + 2.8)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_REPO
    if not (repo / "kalman_jax").is_dir():
        sys.exit(f"Repository not found at {repo}")
    OUT.mkdir(parents=True, exist_ok=True)
    z = write_zip(repo)
    make_cover(OUT / "cover.png")
    print(f"Wrote {z} ({z.stat().st_size // 1024} KB) and {OUT / 'cover.png'}")


if __name__ == "__main__":
    main()
