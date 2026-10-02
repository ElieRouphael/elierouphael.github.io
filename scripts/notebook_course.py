"""Shared renderer for the notebook courses under public/tutorials/.

A course is a folder of executed Jupyter notebooks. This module turns each notebook
into a static page (markdown and math, code cells, text and figure outputs), copies
the notebooks for download, and writes the shared stylesheet and script. Each course
has its own build script that defines its chapters and landing page:

    scripts/build_trustdrive_course.py
    scripts/build_bayesian_course.py

Math is left as TeX and rendered in the browser by KaTeX; code is highlighted by
highlight.js. Outputs are copied from the notebooks, never recomputed.
"""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

SITE = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "course_assets"

KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9"
HLJS = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"
FONTS = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;"
         "12..96,800&family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;"
         "0,8..60,600;1,8..60,400&display=swap")

ANSI = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class Course:
    name: str                       # shown in the eyebrow and back link, e.g. "TrustDrive course"
    folder: str                     # public/tutorials/<folder>
    chapters: list[dict]            # each: stem (notebook file name), short, build, result
    author: str = "Elie Rouphael"
    license: str = ""               # appended to each chapter footer, e.g. "MIT licensed."
    notes: dict[str, str] = field(default_factory=dict)   # stem -> HTML note under the download buttons

    @property
    def out(self) -> Path:
        return SITE / "public" / "tutorials" / self.folder

    @property
    def last(self) -> str:
        return self.chapters[-1]["stem"][:2]


# ---------------------------------------------------------------- markdown

def make_markdown() -> MarkdownIt:
    md = MarkdownIt("commonmark").enable("table")
    md.use(dollarmath_plugin, allow_space=True, allow_digits=True, double_inline=True)
    esc = html.escape
    # KaTeX renders the TeX left in these elements on page load (course.js).
    md.add_render_rule("math_inline", lambda self, t, i, o, e: f'<span class="math inline">{esc(t[i].content)}</span>')
    md.add_render_rule("math_inline_double", lambda self, t, i, o, e: f'<span class="math display">{esc(t[i].content)}</span>')
    md.add_render_rule("math_block", lambda self, t, i, o, e: f'<div class="math display">{esc(t[i].content)}</div>\n')
    return md


MD = make_markdown()


def strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s))


def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", strip_tags(s).lower()).strip("-")
    return s or "section"


def slug(ch: dict) -> str:
    return ch["stem"][:2] + "-" + ch["stem"][3:].replace("_", "-")


def linkify_refs(s: str) -> str:
    s = re.sub(r"doi:(10\.[^\s<]+?)(?=[.,;]?(?:\s|<|$))",
               lambda m: f'<a href="https://doi.org/{m[1]}">doi:{m[1]}</a>', s)
    s = re.sub(r"arXiv:(\d{4}\.\d{4,5})",
               lambda m: f'<a href="https://arxiv.org/abs/{m[1]}">arXiv:{m[1]}</a>', s)
    return s


def png_size(data: bytes) -> tuple[int, int]:
    return struct.unpack(">II", data[16:24])


def figure_alt(code: str, n: int) -> str:
    """Use the plot titles in the code that produced a figure as its alt text."""
    titles = re.findall(r"""(?:set_title|suptitle|title)\(\s*(?:[fr]?)(["'])(.+?)\1""", code)
    words = []
    for _, t in titles[:3]:
        t = re.sub(r"\$([^$]*)\$", lambda m: re.sub(r"\\[a-zA-Z]+|[{}\\^_]", " ", m[1]), t)
        t = re.sub(r"%[-.\d]*[sdfg]|\{[^}]*\}", "…", t)
        words.append(" ".join(t.split()))
    words = [w for w in words if w.strip(" …")]
    return "Figure: " + "; ".join(words) if words else f"Figure {n} produced by the code above"


# ---------------------------------------------------------------- chapters

def render_chapter(course: Course, idx: int, nb: dict) -> dict:
    ch = course.chapters[idx]
    cells = nb["cells"]
    first = "".join(cells[0]["source"]).splitlines()
    title = re.sub(r"^#\s*\d+\s*[-·–—]\s*", "", first[0]).strip()
    intro_md = "\n".join(first[1:]).strip()

    body, toc, fig_n = [], [], 0
    used_ids = set()

    def add_ids(rendered: str) -> str:
        def repl(m):
            level, inner = m[1], m[2]
            base = slugify(inner)
            hid, k = base, 2
            while hid in used_ids:
                hid, k = f"{base}-{k}", k + 1
            used_ids.add(hid)
            if level == "2":
                toc.append((hid, inner))
            return f'<h{level} id="{hid}">{inner}</h{level}>'
        rendered = re.sub(r"<h1>(.*?)</h1>", r"<h2>\1</h2>", rendered)
        return re.sub(r"<h([23])>(.*?)</h\1>", repl, rendered)

    intro_html = add_ids(MD.render(intro_md)) if intro_md else ""

    for cell in cells[1:]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            rendered = add_ids(MD.render(src))
            head = src.lstrip().splitlines()[0] if src.strip() else ""
            cls = "md"
            if re.match(r"#+\s*References", head):
                cls += " refs"
                rendered = linkify_refs(rendered)
            elif re.match(r"#+\s*Exercises", head):
                cls += " exercises"
            body.append(f'<div class="{cls}">\n{rendered}</div>')
        elif cell["cell_type"] == "code":
            if not src.strip():
                continue
            parts = [f'<div class="cell">\n<pre class="source"><code class="language-python">{html.escape(src)}</code></pre>']
            stream = []

            def flush():
                if stream:
                    parts.append(f'<pre class="output">{html.escape(ANSI.sub("", "".join(stream)).rstrip())}</pre>')
                    stream.clear()

            for o in cell.get("outputs", []):
                t = o["output_type"]
                if t == "stream":
                    if o.get("name") == "stderr":
                        continue
                    stream.append("".join(o["text"]))
                elif t in ("display_data", "execute_result"):
                    data = o.get("data", {})
                    if "image/png" in data:
                        flush()
                        fig_n += 1
                        raw = base64.b64decode(data["image/png"])
                        name = f"{ch['stem'][:2]}-{fig_n}.png"
                        (course.out / "img" / name).write_bytes(raw)
                        w, h = png_size(raw)
                        alt = html.escape(figure_alt(src, fig_n), quote=True)
                        parts.append(f'<figure class="output-figure"><img src="img/{name}" alt="{alt}" '
                                     f'width="{w}" height="{h}" loading="lazy"></figure>')
                    elif "text/plain" in data:
                        stream.append("".join(data["text/plain"]) + "\n")
                elif t == "error":
                    flush()
                    tb = ANSI.sub("", "\n".join(o.get("traceback", [])))
                    parts.append(f'<pre class="output error">{html.escape(tb)}</pre>')
            flush()
            parts.append("</div>")
            body.append("\n".join(parts))

    return dict(title=title, intro=intro_html, body="\n\n".join(body), toc=toc, figures=fig_n)


def page_head(title: str, description: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description, quote=True)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<link rel="stylesheet" href="style.css">
</head>
<body>
"""


def page_scripts() -> str:
    return f"""
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{HLJS}"></script>
<script defer src="course.js"></script>
</body>
</html>
"""


def chapter_page(course: Course, idx: int, r: dict) -> str:
    ch = course.chapters[idx]
    num = ch["stem"][:2]
    chapters = course.chapters
    prev_ch = chapters[idx - 1] if idx > 0 else None
    next_ch = chapters[idx + 1] if idx + 1 < len(chapters) else None

    toc = "\n".join(f'    <li><a href="#{hid}">{inner}</a></li>' for hid, inner in r["toc"]
                    if hid not in ("references",))
    note = course.notes.get(ch["stem"], "")
    if note:
        note = f'<p class="dl-note">{note}</p>'

    def pager_link(c, rel):
        if not c:
            return '<span></span>'
        label = "Previous" if rel == "prev" else "Next"
        return (f'<a class="{rel}" rel="{rel}" href="{slug(c)}.html"><span class="lbl">{label} · {c["stem"][:2]}</span>'
                f'<span class="t">{html.escape(c["short"])}</span></a>')

    license_note = f" {course.license}" if course.license else ""
    return page_head(f"{num} · {r['title']} — {course.name}", ch["build"]) + f"""
<nav class="site-back col" aria-label="Site"><a href="index.html">&larr; {html.escape(course.name)}</a><a href="/">{html.escape(course.author)}</a></nav>

<header class="hero col">
  <span class="eyebrow">{html.escape(course.name)} · Chapter {num} of {course.last}</span>
  <h1>{html.escape(r['title'])}</h1>
  <div class="intro prose">
{r['intro']}
  </div>
  <div class="downloads">
    <a class="btn" href="notebooks/{ch['stem']}.ipynb" download>Download notebook (.ipynb)</a>
    <a class="btn ghost" href="{course.folder}-course.zip" download>Full course (.zip)</a>
  </div>
  {note}
</header>

<nav class="col chapter-toc" aria-label="In this chapter">
  <span class="eyebrow">In this chapter</span>
  <ol>
{toc}
  </ol>
</nav>

<main class="col notebook">
{r['body']}
</main>

<nav class="col pager" aria-label="Chapters">
  {pager_link(prev_ch, 'prev')}
  {pager_link(next_ch, 'next')}
</nav>

<footer class="col">
  <p>Chapter {num} of the {html.escape(course.name)} by {html.escape(course.author)}. Rendered from the executed Jupyter notebook; every figure and number above is the notebook's own output. Math rendered with KaTeX, code highlighted with highlight.js.{license_note}</p>
</footer>
""" + page_scripts()


def build_chapters(course: Course, nb_dir: Path) -> list[dict]:
    """Render every chapter page, extract figures, copy notebooks and shared assets."""
    out = course.out
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("img", "notebooks"):
        (out / sub).mkdir(exist_ok=True)
        for old in (out / sub).iterdir():         # clear stale outputs (file by file: OneDrive can lock folders)
            old.unlink()
    for old in out.glob("[0-9][0-9]-*.html"):
        old.unlink()
    for asset in ("style.css", "course.js"):
        shutil.copy2(ASSETS / asset, out / asset)

    results = []
    for idx, ch in enumerate(course.chapters):
        path = nb_dir / f"{ch['stem']}.ipynb"
        nb = json.loads(path.read_text(encoding="utf-8"))
        r = render_chapter(course, idx, nb)
        (out / f"{slug(ch)}.html").write_text(chapter_page(course, idx, r), encoding="utf-8")
        shutil.copy2(path, out / "notebooks" / path.name)
        results.append(r)
        print(f"  {slug(ch)}.html  {len(r['toc'])} sections, {r['figures']} figures")
    return results


def write_zip(course: Course, members: list[tuple[Path, str]]) -> Path:
    """Zip (source file, name inside the archive) pairs; names are prefixed with <folder>-course/."""
    path = course.out / f"{course.folder}-course.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in members:
            if src.exists():
                z.write(src, f"{course.folder}-course/{arc}")
    return path
