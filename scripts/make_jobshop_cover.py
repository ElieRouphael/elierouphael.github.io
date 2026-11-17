"""Tutorials-card cover for the job-shop course (1280x800, the gallery's light style).

A real executed schedule from the repository's simulator: test day 1, frequent breakdowns, CP-SAT
rescheduling, minutes 240-480 (a busy window where all four product families run). One lane per
machine; blocks coloured by product family, colour changes in light grey, breakdowns hatched. The schedule is checked before drawing.

Usage: python scripts/make_jobshop_cover.py <path/to/jobshop-digital-twin> [output.png]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "public/tutorials/jobshop-digital-twin/cover.png"
sys.path.insert(0, str(REPO))
from jobshop.checker import check_schedule  # noqa: E402
from jobshop.cpsat import CPSATController  # noqa: E402
from jobshop.scenario import LEVELS, make_scenario, make_workshop  # noqa: E402
from jobshop.shop import simulate  # noqa: E402

ws = make_workshop()
scenario = make_scenario(ws, 1, n_jobs=30, mtbf=LEVELS["frequent breakdowns"])
log = simulate(ws, scenario, CPSATController())
assert check_schedule(ws, scenario, log) == []
family = {j.idx: j.family for j in scenario.jobs}

BG, INK, LANE, SETUP = "#fbfbf8", "#1d2530", "#efeee9", "#cfcdc6"
FAM = ["#2a78d6", "#ee6a33", "#9cc3ee", "#f5b08f"]   # the gallery's blue and orange, full and light
T0, T1 = 240, 480

fig = plt.figure(figsize=(8, 5), dpi=160, facecolor=BG)
ax = fig.add_axes([0.045, 0.05, 0.91, 0.72]); ax.set_facecolor(BG); ax.axis("off")
n = ws.n_machines
ax.set_xlim(T0, T1); ax.set_ylim(-0.55, n - 0.45)
ASPECT = 0.025   # keeps the corner rounding circular in data units


def block(x0, x1, y, h, fc, z=2, **kw):
    x0, x1 = max(x0, T0), min(x1, T1)
    if x1 - x0 < 4.0:   # drop slivers cut by the window edges
        return
    ax.add_patch(FancyBboxPatch((x0 + 0.35, y - h / 2), x1 - x0 - 0.7, h,
                                boxstyle=f"round,pad=0,rounding_size={min(2.0, (x1 - x0) / 3):.2f}",
                                mutation_aspect=ASPECT, fc=fc, ec="none", zorder=z, **kw))


ypos = {m: n - 1 - m for m in range(n)}   # machine 0 at the top
for m in range(n):
    ax.add_patch(FancyBboxPatch((T0, ypos[m] - 0.4), T1 - T0, 0.8, boxstyle="round,pad=0,rounding_size=2.5",
                                mutation_aspect=ASPECT, fc=LANE, ec="none", zorder=0))
    for down, up in scenario.downtime[m]:
        x0, x1 = max(down, T0), min(up, T1)
        if x1 > x0:
            ax.add_patch(Rectangle((x0, ypos[m] - 0.4), x1 - x0, 0.8, fc="none", ec=INK, lw=0,
                                   hatch="///", alpha=0.35, zorder=1))
for s in log.segments:
    block(s.start, s.end, ypos[s.machine], 0.56, SETUP if s.kind == "setup" else FAM[family[s.job]])

fig.text(0.045, 0.95, "Job-Shop Scheduling", color=INK, fontsize=26, weight="bold", va="top")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor=BG)
plt.close(fig)
print(OUT, sum(T0 < s.start < T1 for s in log.segments), "segments drawn")
