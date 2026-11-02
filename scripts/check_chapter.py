"""Render one chapter of the MPC course to a temporary file and report problems, without
touching public/.  For authors working on a chapter while the full build is left alone.

Usage:  python scripts/check_chapter.py a07 [a08 ...]
Reports: unresolved [[refs]], leftover container fences, parts listed in the front matter
that have no section (and the reverse), math blocks that look malformed, missing demo or
numbers script, and the word count.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course_md import chapter_page, load_course, render_chapter_file  # noqa: E402

SITE = Path(__file__).resolve().parents[1]
SRC = SITE / "courses" / "mpc-lmpc-lab"
OUT = SITE / "public" / "tutorials" / "mpc-lmpc-lab"


def check(cid: str) -> int:
    course = load_course(SRC / "course.yaml")
    ch = course.chapter(cid)
    src = SRC / "chapters" / f"{cid}.md"
    if not src.exists():
        print(f"{cid}: no source at {src}"); return 1
    r = render_chapter_file(src, course, ch)
    for key in ("lede", "result", "build", "needs"):
        if r.meta.get(key):
            setattr(ch, key, r.meta[key])
    ch.parts = r.meta.get("parts", [])
    page = chapter_page(course, ch, r)
    tmp = Path(tempfile.gettempdir()) / f"{cid}-check.html"
    tmp.write_text(page, encoding="utf-8")
    problems = []
    broken = re.findall(r'class="broken-ref">([^<]*)', page)
    if broken: problems.append(f"unresolved refs: {broken}")
    fences = re.findall(r"^\s*:{3,}.*$", re.sub(r"<pre.*?</pre>", "", page, flags=re.S), flags=re.M)
    if fences: problems.append(f"leftover container fences in the output: {fences[:5]}")
    sec_ids = [sid for sid, _, _ in r.sections]
    part_ids = [p["id"] for p in ch.parts]
    for pid in part_ids:
        if pid not in sec_ids: problems.append(f"front-matter part '{pid}' has no section")
    for sid in sec_ids:
        if sid not in part_ids: problems.append(f"section '{sid}' is not described in the front matter parts")
    if "<!--EXT:" in page: problems.append("an exercise title was not attached to a paragraph")
    if re.search(r"\$\$|\$[^$\n]+\$", re.sub(r"<pre.*?</pre>", "", page, flags=re.S)):
        problems.append("a $...$ survived into the HTML (unrendered math, often inside raw HTML)")
    for want in ([f"interactive/{cid}.js"] if ch.script else []) + ([f"code/{cid}_numbers.py"] if ch.numbers else []):
        if not (OUT / want).exists(): problems.append(f"missing {want}")
    if r.meta.get("result", ch.result) == "": problems.append("no 'result' (You find) line in the front matter")
    words = len(re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style).*?</\1>", "", page, flags=re.S)).split())
    qs = len(re.findall(r'<details class="sol"><summary>\d+\.', page))
    exs = len(re.findall(r'<ol class="exercises">', page)), len(re.findall(r'<li><div class="ex">', page))
    print(f"{cid}: {len(r.sections)} parts, {words} words, {qs} questions, {exs[1]} exercises, "
          f"{page.count('<canvas')} canvases, {page.count('code-walk')} code walks -> {tmp}")
    for p in problems:
        print(f"   ! {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(max(check(c) for c in sys.argv[1:]) if len(sys.argv) > 1 else print(__doc__))
