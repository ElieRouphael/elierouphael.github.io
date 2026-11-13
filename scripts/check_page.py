"""Render built course pages in headless Edge: report JavaScript errors, save screenshots.

Node.js is not installed on this machine, so this is how a chapter's demo is tested. It loads the
page from disk (file://), lets its scripts run, prints every console error or uncaught exception
the page produced, and writes a full-page screenshot in light and in dark mode.

Usage:
    python scripts/check_page.py public/tutorials/jobshop-digital-twin/a02-*.html [more pages]
    python scripts/check_page.py --height 14000 --out C:/tmp/shots page.html
Exit status 1 if any page logged an error. Screenshots: <out>/<page>-light.png and -dark.png
(default out: the system temp folder); split them into pieces to view (they are tall).
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import tempfile
from pathlib import Path

EDGE = r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
NOISE = ("chrome-extension://", "ProtocolLaunch", "Error observing HKLM", "DevTools listening")


def run(page: Path, out: Path, height: int, width: int) -> int:
    url = page.resolve().as_uri()
    base = ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=9000",
            f"--window-size={width},{height}"]
    log = subprocess.run([EDGE, *base, "--enable-logging=stderr", "--v=0", "--dump-dom", url],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    lines = [l for l in log.stderr.splitlines() if "CONSOLE" in l and not any(n in l for n in NOISE)]
    errors = [l for l in lines if any(k in l for k in ("Uncaught", "Error", "error", "TypeError", "is not defined", "failed"))]
    dom = log.stdout
    shots = []
    for mode, extra in (("light", []), ("dark", ["--blink-settings=preferredColorScheme=0"])):
        target = out / f"{page.stem}-{mode}.png"
        subprocess.run([EDGE, *base, *extra, f"--screenshot={target}", url], capture_output=True, timeout=120)
        shots.append(target)
    leftovers = []
    if "$$" in dom.split("<body", 1)[-1] or " $" in dom.split("<body", 1)[-1].replace("$ ", ""):
        pass  # math is rendered client-side by KaTeX; unrendered $ are caught by check_chapter.py
    if "katex-error" in dom:
        leftovers.append("KaTeX could not render some math (class katex-error in the DOM)")
    print(f"{page.name}: {len(lines)} console lines, {len(errors)} errors; screenshots {shots[0]} , {shots[1]}")
    for l in errors:
        print("   ! " + l.split("CONSOLE", 1)[-1][:400])
    for l in leftovers:
        print("   ! " + l)
    return 1 if errors or leftovers else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+")
    ap.add_argument("--out", default=tempfile.gettempdir())
    ap.add_argument("--height", type=int, default=12000)
    ap.add_argument("--width", type=int, default=1100)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pages = [Path(p) for pat in a.pages for p in (glob.glob(pat) or [pat])]
    return max((run(p, out, a.height, a.width) for p in pages), default=0)


if __name__ == "__main__":
    sys.exit(main())
