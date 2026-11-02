"""Render course chapters written in Markdown into interactive pages in the kit style.

A chapter is a Markdown file with YAML front matter, written in CommonMark plus:

    $x$ and $$x$$                 inline and display math, rendered in the browser by KaTeX
    ## Title {#anchor}            a part; the table of contents comes from these headings
    [[a07]] [[a07#lqr|text]]      a link to a chapter (or a part of it); [[#anchor|text]] within the
                                  chapter; [[ref:3]] a reference; the chapter label is used as text
    ::: define What is X?         the "In plain words / More precisely" box; list items written
                                  `Term :: meaning` become its definition list
    ::: callout [warm]            a callout
    ::: hero                      the live demo under the title (raw HTML), full width
    ::: code path[:function|:a-b] the source of a function (or a line range) of the course package,
                                  highlighted, followed by the container's own commentary
    ::: table [num]               wraps the table that follows for horizontal scrolling
    ::: steps                     the list inside becomes the numbered "chapter on one page" list
    :::: qa / ::: q Question?     check-your-understanding questions with folded answers (an outer
                                  container always uses more colons than the containers inside it)
    ::::: exercises / :::: exercise Title / ::: hint / ::: solution
    ::: refs                      a list `- [1] ...` becomes the reference list
    %narrow <tex>                 on the line after a display formula: the version used on phones

Everything else is CommonMark, including raw HTML for widgets.
"""
from __future__ import annotations

import ast
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from markdown_it import MarkdownIt
from mdit_py_plugins.attrs import attrs_plugin
from mdit_py_plugins.container import container_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin

KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9"
HLJS = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"
FONTS = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;"
         "12..96,800&family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;"
         "0,8..60,600;1,8..60,400&display=swap")


# ---------------------------------------------------------------- the course manifest

@dataclass
class Chapter:
    id: str                 # a07, b02, c01
    slug: str               # a07-lyapunov-stability-and-lqr
    title: str
    short: str
    build: str = ""         # one line for the index card
    result: str = ""        # the "You find" line
    needs: str = ""         # prerequisites, for the byline
    lede: str = ""          # filled from the front matter
    parts: list = field(default_factory=list)   # [{id, title, d}], from the front matter
    notebook: str = ""      # a lab chapter's notebook stem
    script: bool = True     # has interactive/<id>.js
    numbers: bool = True    # has code/<id>_numbers.py
    extra_scripts: list = field(default_factory=list)   # e.g. interactive/<id>-data.js

    @property
    def label(self) -> str:        # "A7", "B2"
        return self.id[0].upper() + str(int(self.id[1:]))

    @property
    def part(self) -> str:
        return self.id[0].upper()


@dataclass
class Course:
    name: str
    folder: str
    parts: list              # [{key: "A", title, lede}]
    chapters: list           # [Chapter]
    author: str = "Elie Rouphael"
    license: str = "MIT licensed."
    package_root: Path | None = None    # where `::: code` snippets are read from

    def chapter(self, cid: str) -> Chapter:
        for ch in self.chapters:
            if ch.id == cid:
                return ch
        raise KeyError(cid)

    def in_part(self, key: str) -> list:
        return [ch for ch in self.chapters if ch.part == key]


def load_course(path: Path) -> Course:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    chapters = [Chapter(**c) for c in data["chapters"]]
    return Course(name=data["name"], folder=data["folder"], parts=data["parts"], chapters=chapters,
                  author=data.get("author", "Elie Rouphael"), license=data.get("license", "MIT licensed."),
                  package_root=Path(data["package_root"]).expanduser() if data.get("package_root") else None)


# ---------------------------------------------------------------- markdown

def make_markdown() -> MarkdownIt:
    md = MarkdownIt("commonmark").enable("table")
    md.use(dollarmath_plugin, allow_space=True, allow_digits=True, double_inline=True)
    md.use(attrs_plugin)
    esc = html.escape
    md.add_render_rule("math_inline", lambda self, t, i, o, e: f'<span class="m">{esc(t[i].content)}</span>')
    md.add_render_rule("math_inline_double", lambda self, t, i, o, e: f'<div class="math-block">{esc(t[i].content)}</div>')
    md.add_render_rule("math_block", lambda self, t, i, o, e: f'<div class="math-block">{esc(t[i].content)}</div>\n')

    def inline(md_, text):
        return md_.renderInline(text)

    def container(name, open_fn, close_html):
        def render(self, tokens, idx, options, env):
            t = tokens[idx]
            if t.nesting == 1:
                info = t.info.strip()[len(name):].strip()
                return open_fn(info, env)
            return close_html
        md.use(container_plugin, name=name, render=render)

    container("define", lambda info, env: f'<div class="define"><p class="define-title">{inline(md, info)}</p>\n', "</div>\n")
    container("callout", lambda info, env: f'<div class="callout{" " + html.escape(info) if info else ""}">\n', "</div>\n")
    container("hero", lambda info, env: "<!--HERO--><div class=\"wide\">\n", "</div><!--/HERO-->\n")
    container("hint", lambda info, env: '<details class="sol"><summary>Hint</summary>\n', "</details>\n")
    container("solution", lambda info, env: '<details class="sol"><summary>Solution</summary>\n', "</details>\n")
    container("answer", lambda info, env: '<details class="sol"><summary>Answer</summary>\n', "</details>\n")
    container("exercises", lambda info, env: '<ol class="exercises">\n', "</ol>\n")
    container("exercise", lambda info, env: f'<li><div class="ex"><!--EXT:{inline(md, info)}-->\n', "</div></li>\n")
    container("qa", lambda info, env: '<div class="qa">\n', "</div>\n")
    container("q", lambda info, env: f'<details class="sol"><summary>{inline(md, info)}</summary>\n', "</details>\n")
    container("steps", lambda info, env: '<div class="steps-wrap">\n', "</div>\n")
    container("refs", lambda info, env: '<div class="refs-wrap">\n', "</div>\n")
    container("table", lambda info, env: f'<div class="table-wrap{" " + html.escape(info) if info else ""}">\n', "</div>\n")
    container("code", lambda info, env: code_open(info, env), "</div>\n")
    return md


def code_open(spec: str, env: dict) -> str:
    """`::: code advanced_mpc_lmpc/polytopes.py:predecessor_scalar_input` -> the function's source, highlighted."""
    root: Path | None = env.get("package_root")
    path, _, what = spec.partition(":")
    label = path + (f" · {what}" if what else "")
    if root is None or not (root / path).exists():
        return f'<div class="code-walk"><p class="code-src">{html.escape(label)}</p><pre><code class="language-python"># source not found: {html.escape(spec)}</code></pre>\n'
    source = (root / path).read_text(encoding="utf-8")
    lines = source.splitlines()
    if re.fullmatch(r"\d+-\d+", what):
        a, b = map(int, what.split("-"))
        snippet = "\n".join(lines[a - 1:b])
    elif what:
        tree = ast.parse(source)
        node = None
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                qual = n.name
                if qual == what or qual == what.split(".")[-1]:
                    node = n
                    if "." not in what:
                        break
        if node is None:
            snippet = f"# {what} not found in {path}"
        else:
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
            snippet = "\n".join(lines[start - 1:node.end_lineno])
            snippet = re.sub(r"\n\s*\n\s*\n", "\n\n", snippet)
    else:
        snippet = source
    return (f'<div class="code-walk"><p class="code-src">{html.escape(label)}</p>'
            f'<pre><code class="language-python">{html.escape(snippet)}</code></pre>\n')


MD = make_markdown()


# ---------------------------------------------------------------- cross references

def resolve_refs(text: str, course: Course, here: Chapter) -> str:
    """Replace [[...]] links outside code with HTML anchors."""
    def link(m):
        body = m.group(1)
        target, _, label = body.partition("|")
        target = target.strip(); label = label.strip()
        if target.startswith("ref:"):
            n = target[4:]
            return f'<a href="#ref-{n}">[{n}]</a>'
        if target.startswith("#"):
            return f'<a href="{target}">{label or target[1:]}</a>'
        cid, _, anchor = target.partition("#")
        try:
            ch = course.chapter(cid.lower())
        except KeyError:
            return f'<span class="broken-ref">{html.escape(body)}</span>'
        href = f"{ch.slug}.html" + (f"#{anchor}" if anchor else "")
        text_ = label or ch.label
        return f'<a href="{href}" title="{html.escape(ch.title, quote=True)}">{text_}</a>'

    out, in_fence = [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line); continue
        if in_fence:
            out.append(line); continue
        parts = line.split("`")
        for i in range(0, len(parts), 2):          # even segments are outside inline code
            parts[i] = re.sub(r"\[\[(.+?)\]\]", link, parts[i])
        out.append("`".join(parts))
    return "".join(out)


# ---------------------------------------------------------------- post-processing

def linkify_refs(s: str) -> str:
    s = re.sub(r"doi:(10\.[^\s<]+?)(?=[.,;]?(?:\s|<|$))",
               lambda m: f'<a href="https://doi.org/{m[1]}">doi:{m[1]}</a>', s)
    s = re.sub(r"arXiv:(\d{4}\.\d{4,5})",
               lambda m: f'<a href="https://arxiv.org/abs/{m[1]}">arXiv:{m[1]}</a>', s)
    return s


def mark_heading_ids(text: str) -> str:
    """`## Title {#anchor}` -> `## Title <!--id:anchor-->`, applied by postprocess."""
    return re.sub(r"^(#{1,4} .*?)\s*\{#([A-Za-z0-9_-]+)\}\s*$", r"\1 <!--id:\2-->", text, flags=re.M)


def postprocess(h: str) -> str:
    h = re.sub(r"<h([1-4])>(.*?)\s*<!--id:([A-Za-z0-9_-]+)-->\s*</h\1>", r'<h\1 id="\3">\2</h\1>', h, flags=re.S)
    # exercise titles join the first paragraph
    h = re.sub(r"<!--EXT:(.*?)-->\s*<p>", lambda m: f"<p><b>{m[1]}.</b> ", h, flags=re.S)
    h = re.sub(r"<!--EXT:(.*?)-->", lambda m: f"<p><b>{m[1]}.</b></p>", h, flags=re.S)
    # steps: the list becomes ol.steps with each item wrapped
    def steps(m):
        inner = re.sub(r"<li>(.*?)</li>", r"<li><div>\1</div></li>", m[1], flags=re.S)
        return f'<ol class="steps">{inner}</ol>'
    h = re.sub(r'<div class="steps-wrap">\s*<(?:ol|ul)>(.*?)</(?:ol|ul)>\s*</div>', steps, h, flags=re.S)
    # references
    def refs(m):
        inner = re.sub(r"<li>\s*\[(\d+)\]\s*(.*?)</li>",
                       lambda r: f'<li id="ref-{r[1]}"><span class="rn">[{r[1]}]</span><div>{linkify_refs(r[2])}</div></li>',
                       m[1], flags=re.S)
        return f'<ul class="refs">{inner}</ul>'
    h = re.sub(r'<div class="refs-wrap">\s*<ul>(.*?)</ul>\s*</div>', refs, h, flags=re.S)
    # definition lists inside define boxes: items "Term :: meaning"
    def dl(m):
        items = re.findall(r"<li>(.*?)</li>", m[1], flags=re.S)
        if items and all(" :: " in it for it in items):
            rows = "".join(f"<dt>{t.strip()}</dt><dd>{d.strip()}</dd>" for t, d in (it.split(" :: ", 1) for it in items))
            return f"<dl>{rows}</dl>"
        return m[0]
    h = re.sub(r"<ul>(.*?)</ul>", dl, h, flags=re.S)
    # tables: class on the wrapper moves to the table; bare tables get a wrapper
    h = re.sub(r'<div class="table-wrap num">\s*<table>', '<div class="table-wrap"><table class="num">', h)
    h = re.sub(r'(?<!<div class="table-wrap">)(?<!<div class="table-wrap">\n)<table>', '<div class="table-wrap"><table>', h)
    h = re.sub(r'<div class="table-wrap"><table>(.*?)</table>(?!\s*</div>)', r'<div class="table-wrap"><table>\1</table></div>', h, flags=re.S)
    # narrow variants of display formulas
    h = re.sub(r'<div class="math-block">(.*?)</div>\s*<p>%narrow (.*?)</p>',
               lambda m: f'<div class="math-block" data-narrow="{m[2].strip()}">{m[1]}</div>', h, flags=re.S)
    # code fences: language class for highlight.js, output blocks
    h = h.replace('<pre><code class="language-output">', '<pre class="output"><code class="nohl">')
    h = re.sub(r"<pre><code>(?!<)", '<pre><code class="nohl">', h)
    return h


# ---------------------------------------------------------------- the chapter file

@dataclass
class Rendered:
    meta: dict
    pre: str            # html before the first part (lede paragraphs, the hero demo)
    sections: list      # [(id, title, html)]


def render_chapter_file(path: Path, course: Course, ch: Chapter) -> Rendered:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, flags=re.S)
    meta = yaml.safe_load(m[1]) if m else {}
    body = text[m.end():] if m else text
    body = resolve_refs(body, course, ch)
    body = mark_heading_ids(body)
    env = {"package_root": course.package_root}
    h = postprocess(MD.render(body, env))
    # split into parts at h2
    pieces = re.split(r'(<h2[^>]*>.*?</h2>)', h, flags=re.S)
    pre = pieces[0]
    sections = []
    for i in range(1, len(pieces), 2):
        head = pieces[i]
        hid = re.search(r'id="([^"]+)"', head)
        title = re.sub(r"<[^>]+>", "", head).strip()
        sid = hid[1] if hid else re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        sections.append((sid, title, pieces[i + 1]))
    return Rendered(meta, pre, sections)


# ---------------------------------------------------------------- page assembly

def page_head(title: str, description: str, extra_css: str = "") -> str:
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
<link rel="stylesheet" href="../kit/kit.css">
<link rel="stylesheet" href="course.css{stamp()}">
{extra_css}</head>
<body>
"""


# A build stamp on the course's own assets, so that a rebuilt stylesheet or demo is not served from the cache.
ASSET_VERSION = ""


def stamp() -> str:
    return f"?v={ASSET_VERSION}" if ASSET_VERSION else ""


def page_scripts(scripts: list[str]) -> str:
    tags = "\n".join(f'<script src="{s}{stamp()}"></script>' for s in scripts)
    return f"""
<script src="{KATEX}/katex.min.js"></script>
<script src="{HLJS}"></script>
<script src="../kit/kit.js"></script>
{tags}
</body>
</html>
"""


def toc_html(ch: Chapter, sections: list) -> str:
    descs = {p["id"]: p.get("d", "") for p in ch.parts}
    items = []
    for n, (sid, title, _) in enumerate(sections, 1):
        items.append(f'    <a href="#{sid}"><span class="n">{n}</span><span class="t">{title}</span>'
                     f'<span class="d">{html.escape(descs.get(sid, ""))}</span></a>')
    return "\n".join(items)


def pager_html(course: Course, ch: Chapter) -> str:
    idx = course.chapters.index(ch)
    prev_ch = course.chapters[idx - 1] if idx > 0 else None
    next_ch = course.chapters[idx + 1] if idx + 1 < len(course.chapters) else None

    def link(c, rel):
        if not c:
            return ('<a class="prev" rel="prev" href="index.html"><span class="lbl">Course</span><span class="t">The overview</span></a>'
                    if rel == "prev" else
                    '<a class="next" href="index.html"><span class="lbl">Course</span><span class="t">Back to the overview</span></a>')
        label = "Previous" if rel == "prev" else "Next"
        return (f'<a class="{rel}" rel="{rel}" href="{c.slug}.html"><span class="lbl">{label} · {c.label}</span>'
                f'<span class="t">{html.escape(c.short)}</span></a>')
    return f'<nav class="col pager" aria-label="Chapters">\n  {link(prev_ch, "prev")}\n  {link(next_ch, "next")}\n</nav>'


def chapter_page(course: Course, ch: Chapter, r: Rendered) -> str:
    meta = r.meta
    part = next(p for p in course.parts if p["key"] == ch.part)
    n_in_part = len(course.in_part(ch.part))
    lede = meta.get("lede", ch.lede)
    needs = meta.get("needs", ch.needs)
    downloads = []
    if ch.numbers:
        downloads.append(f'<a class="btn" href="code/{ch.id}_numbers.py" download>The numbers script (.py)</a>')
    if ch.notebook:
        downloads.append(f'<a class="btn" href="notebooks/{ch.notebook}.ipynb" download>The lab notebook (.ipynb)</a>')
    downloads.append(f'<a class="btn ghost" href="{course.folder}-course.zip" download>Full course (.zip)</a>')
    concepts = meta.get("concepts", [])
    concepts_html = ""
    if concepts:
        links = ", ".join(resolve_refs(c, course, ch) for c in concepts)
        concepts_html = f'<p class="concepts"><span class="lbl">Concepts used</span>{links}</p>'
    # the pre-h2 region: the hero (wide) stays out of the column; other blocks go into a prose column
    pre_html = ""
    for chunk in re.split(r"(<!--HERO-->.*?<!--/HERO-->)", r.pre, flags=re.S):
        if not chunk.strip():
            continue
        if chunk.startswith("<!--HERO-->"):
            pre_html += chunk.replace("<!--HERO-->", "").replace("<!--/HERO-->", "") + "\n"
        else:
            pre_html += f'<div class="col prose">\n{chunk}\n</div>\n'
    sections_html = []
    for n, (sid, title, body) in enumerate(r.sections, 1):
        sections_html.append(f"""<!-- =================================================================== -->
<section id="{sid}">
  <div class="col part-head">
    <span class="part-num">Part {n}</span>
    <h2>{title}</h2>
  </div>
  <div class="col prose">
{body}
  </div>
</section>""")
    scripts = ["interactive/mpc-lib.js"] + [f"interactive/{s}" for s in ch.extra_scripts] + ([f"interactive/{ch.id}.js"] if ch.script else [])
    extra_css = f"<style>\n{meta['css']}\n</style>\n" if meta.get("css") else ""
    description = meta.get("description", ch.build)
    return page_head(f"{ch.label} · {ch.short} · {course.name}", description, extra_css) + f"""
<nav class="site-back col" aria-label="Site"><a href="index.html">&larr; {html.escape(course.name)}</a><a href="/">{html.escape(course.author)}</a></nav>

<header class="hero col">
  <span class="eyebrow">{html.escape(course.name)} · Part {ch.part}, {html.escape(part["title"])} · Chapter {ch.label} of {n_in_part}</span>
  <h1>{ch.title}</h1>
  <p class="lede">{lede}</p>
  <p class="byline">{html.escape(course.author)} · {needs} · Every number is checked in Python</p>
  {concepts_html}
  <div class="downloads">
    {chr(10).join(downloads)}
  </div>
</header>

{pre_html}
<section class="col" aria-label="Contents" style="padding-top:0">
  <nav class="toc">
{toc_html(ch, r.sections)}
  </nav>
</section>

{chr(10).join(sections_html)}

{pager_html(course, ch)}

<footer class="col">
  <p>Chapter {ch.label} of "{html.escape(course.name)}" by {html.escape(course.author)}. {meta.get("footer", "The demos run in your browser with the same algorithms as the course's code; the numbers script reproduces every number on the page.")} Math rendered with KaTeX, code highlighted with highlight.js. {course.license}</p>
</footer>
""" + page_scripts(scripts)
