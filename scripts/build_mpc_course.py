"""Build the "Advanced MPC and Learning MPC" course pages.

The chapters are written in Markdown under courses/mpc-lmpc-lab/chapters/<id>.md (see
scripts/course_md.py for the dialect) and listed in courses/mpc-lmpc-lab/course.yaml. This
script renders them into public/tutorials/mpc-lmpc-lab/:

    <slug>.html                 one page per chapter
    index.html                  the course landing page (from courses/mpc-lmpc-lab/index.md)
    notation.html               the notation and the canonical system (from notation.md)
    concepts.html               every concept with the chapter and part that defines it (from concepts.md)
    notebooks/                  the lab notebooks, copied from the course repository
    mpc-lmpc-lab-course.zip     the advanced_mpc_lmpc package, its tests and docs, the
                                notebooks, and code/ (the numbers scripts)

The chapter demos are hand-written: interactive/mpc-lib.js (shared) and interactive/<id>.js.
The numbers scripts, code/<id>_numbers.py, are hand-written too and reproduce every number a
page quotes; they run against the course package.

Usage:
    python scripts/build_mpc_course.py [path/to/advanced_mpc_lmpc_lab]
Needs markdown-it-py, mdit-py-plugins and pyyaml.
"""
from __future__ import annotations

import html
import re
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import course_md  # noqa: E402
from course_md import MD, Chapter, Course, chapter_page, load_course, mark_heading_ids, page_head, page_scripts, postprocess, render_chapter_file, resolve_refs  # noqa: E402

SITE = Path(__file__).resolve().parents[1]
SRC = SITE / "courses" / "mpc-lmpc-lab"
OUT = SITE / "public" / "tutorials" / "mpc-lmpc-lab"


# ---------------------------------------------------------------- landing page

def chapter_card(ch: Chapter) -> str:
    result = f'<span class="r"><span class="lbl">You find</span> {ch.result}</span>' if ch.result else ""
    return f"""    <li>
      <a href="{ch.slug}.html">
        <span class="n">{ch.label}</span>
        <span class="body">
          <span class="t">{html.escape(ch.title)}</span>
          <span class="d">{ch.build}</span>
          {result}
        </span>
      </a>
    </li>"""


def index_page(course: Course, intro: dict) -> str:
    zip_name = f"{course.folder}-course.zip"
    parts_html = []
    for part in course.parts:
        chapters = course.in_part(part["key"])
        if not chapters:
            continue
        cards = "\n".join(chapter_card(ch) for ch in chapters)
        parts_html.append(f"""
<section class="col" aria-labelledby="part-{part['key']}">
  <h2 id="part-{part['key']}">Part {part['key']} · {html.escape(part['title'])}</h2>
  <p class="section-lede">{part['lede']}</p>
  <ol class="chapter-list">
{cards}
  </ol>
</section>""")
    first = course.chapters[0]
    first_b = next((c for c in course.chapters if c.part == "B"), first)
    return page_head(course.name, intro["description"]) + f"""
<nav class="site-back col" aria-label="Site"><a href="/tutorials">&larr; All tutorials</a><a href="/">{html.escape(course.author)}</a></nav>

<header class="hero col">
  <span class="eyebrow">{intro['eyebrow']}</span>
  <h1>{html.escape(course.name)}</h1>
  <p class="lede">{intro['lede']}</p>
  <p class="byline">{html.escape(course.author)} · {intro['assumes']}</p>
  <div class="downloads">
    <a class="btn" href="{first.slug}.html">Start with chapter {first.label}</a>
    <a class="btn" href="{first_b.slug}.html">Skip to the labs</a>
    <a class="btn ghost" href="https://github.com/ElieRouphael/advanced_mpc_lmpc_lab">The repository on GitHub</a>
    <a class="btn ghost" href="{zip_name}" download>Download the course (.zip)</a>
  </div>
</header>

<div class="col prose">
{intro['body']}
</div>
{''.join(parts_html)}

<section class="col prose" aria-labelledby="run">
  <h2 id="run">Running it yourself</h2>
  <p>Every chapter page runs its demos in the browser and quotes numbers computed in Python. To reproduce
  them, <a href="{zip_name}" download>download the course</a>: the <code>advanced_mpc_lmpc</code> package
  with its tests, the lab notebooks, and in <code>code/</code> one script per chapter that prints every
  number quoted on that chapter's page. Then, from the unzipped folder:</p>
  <pre><code class="language-bash">pip install -r requirements.txt
python -m pytest                      # the package's tests
python code/a07_numbers.py            # every number on chapter A7's page
jupyter lab                           # the lab notebooks, in notebooks/</code></pre>
  <p>No commercial solver is needed: the package's quadratic programs are solved by its own interior-point
  method, a hundred lines of NumPy that chapter A4 reads line by line. Each script runs in seconds; the lab
  notebooks in under a minute each.</p>
</section>

<section class="col prose" aria-labelledby="more">
  <h2 id="more">Reference pages</h2>
  <ul>
    <li><a href="notation.html">Notation and the canonical system</a>: every symbol the course uses, and the one double integrator every concept chapter runs on, with its numbers.</li>
    <li><a href="concepts.html">Concept index</a>: each concept with the chapter and part that defines it, the demo that shows it, and the function that computes it.</li>
  </ul>
</section>

<section class="col prose" aria-labelledby="sources">
  <h2 id="sources">Acknowledgements</h2>
  <p>The text and code are original. The choice of topics and several examples were informed by a 2025–2026
  course on Learning MPC (Università di Napoli Federico II, Université Franco-Italienne and Grenoble INP / ENSE3);
  the course download includes the full acknowledgement and the references for each chapter.</p>
</section>

<footer class="col">
  <p>{html.escape(course.name)}, by {html.escape(course.author)}. {course.license} The chapter pages are hand-built interactive
  documents whose numerics are checked against the course package; math by KaTeX, code highlighting by highlight.js.</p>
</footer>
""" + page_scripts([])


# ---------------------------------------------------------------- simple pages (notation, concepts)

def simple_page(course: Course, md_path: Path, title: str, description: str, scripts: list[str]) -> str:
    text = md_path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, flags=re.S)
    body = text[m.end():] if m else text
    dummy = course.chapters[0]
    h = postprocess(MD.render(mark_heading_ids(resolve_refs(body, course, dummy)), {"package_root": course.package_root}))
    pieces = re.split(r'(<h2[^>]*>.*?</h2>)', h, flags=re.S)
    out = [f'<div class="col prose">\n{pieces[0]}\n</div>'] if pieces[0].strip() else []
    for i in range(1, len(pieces), 2):
        head = pieces[i]
        hid = re.search(r'id="([^"]+)"', head)
        sid = hid[1] if hid else f"part-{i}"
        t = re.sub(r"<[^>]+>", "", head).strip()
        out.append(f'<section id="{sid}"><div class="col part-head"><h2>{t}</h2></div><div class="col prose">\n{pieces[i + 1]}\n</div></section>')
    return page_head(f"{title} · {course.name}", description) + f"""
<nav class="site-back col" aria-label="Site"><a href="index.html">&larr; {html.escape(course.name)}</a><a href="/">{html.escape(course.author)}</a></nav>

<header class="hero col">
  <span class="eyebrow">{html.escape(course.name)} · Reference</span>
  <h1>{html.escape(title)}</h1>
  <p class="lede">{description}</p>
</header>

{chr(10).join(out)}

<footer class="col">
  <p>Part of "{html.escape(course.name)}" by {html.escape(course.author)}. Math rendered with KaTeX, code highlighted with highlight.js. {course.license}</p>
</footer>
""" + page_scripts(scripts)


# ---------------------------------------------------------------- zip

def write_zip(course: Course, repo: Path) -> None:
    path = OUT / f"{course.folder}-course.zip"
    members = [(repo / rel, rel) for rel in ("README.md", "requirements.txt", "pyproject.toml", "LICENSE", "NOTICE.md", "CITATION.cff")]
    for folder in ("src", "tests", "docs", "scripts"):
        members += [(p, p.relative_to(repo).as_posix()) for p in sorted((repo / folder).rglob("*"))
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pdf"]
    members += [(p, f"notebooks/{p.name}") for p in sorted((OUT / "notebooks").glob("*.ipynb"))]
    members += [(p, f"code/{p.name}") for p in sorted((OUT / "code").glob("*.py"))]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in members:
            if src.exists():
                z.write(src, f"{course.folder}-course/{arc}")
    print(f"  {path.name}: {len(members)} members")


# ---------------------------------------------------------------- main

def main() -> None:
    course = load_course(SRC / "course.yaml")
    import time
    course_md.ASSET_VERSION = time.strftime("%Y%m%d%H%M%S")
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (course.package_root.parent if course.package_root else None)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "notebooks").mkdir(exist_ok=True)
    (OUT / "code").mkdir(exist_ok=True)

    built, missing = [], []
    for ch in course.chapters:
        src = SRC / "chapters" / f"{ch.id}.md"
        if not src.exists():
            missing.append(ch.id); continue
        r = render_chapter_file(src, course, ch)
        for key in ("lede", "result", "build", "needs"):
            if r.meta.get(key):
                setattr(ch, key, r.meta[key])
        ch.parts = r.meta.get("parts", [])
        (OUT / f"{ch.slug}.html").write_text(chapter_page(course, ch, r), encoding="utf-8")
        built.append(ch.id)
        if ch.notebook and repo and (repo / "notebooks" / f"{ch.notebook}.ipynb").exists():
            shutil.copy2(repo / "notebooks" / f"{ch.notebook}.ipynb", OUT / "notebooks" / f"{ch.notebook}.ipynb")
        for want in ([f"interactive/{ch.id}.js"] if ch.script else []) + ([f"code/{ch.id}_numbers.py"] if ch.numbers else []):
            if not (OUT / want).exists():
                print(f"  warning: {ch.id} has no {want}")
    print(f"  built {len(built)} chapters" + (f"; no source yet for {', '.join(missing)}" if missing else ""))

    intro_md = (SRC / "index.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", intro_md, flags=re.S)
    intro = yaml.safe_load(m[1])
    intro["body"] = postprocess(MD.render(resolve_refs(intro_md[m.end():], course, course.chapters[0]), {}))
    (OUT / "index.html").write_text(index_page(course, intro), encoding="utf-8")
    pages = (("notation", "Notation and the canonical system",
              "Every symbol the course uses, the sign conventions, and the one double integrator every concept chapter runs on, with its numbers.",
              ["interactive/mpc-lib.js", "interactive/notation.js"]),
             ("concepts", "Concept index",
              "Each concept of the course with the chapter and part that defines it, the demo that shows it and the function that computes it.", []))
    for name, title, desc, scripts in pages:
        if (SRC / f"{name}.md").exists():
            (OUT / f"{name}.html").write_text(simple_page(course, SRC / f"{name}.md", title, desc, scripts), encoding="utf-8")
    if repo and repo.exists():
        write_zip(course, repo)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
