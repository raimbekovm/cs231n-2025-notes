#!/usr/bin/env python3
"""Stop the head of every page from holding up the first paint.

Quarto puts a dozen `<script src>` tags in `<head>` with no `defer`. A browser
must stop parsing at each one, fetch it, and run it, before it can lay out a
single word — and none of that code does anything until `DOMContentLoaded`
anyway. Marking them `defer` changes nothing about what runs or in what order;
it only lets the text arrive first.

The same pass preloads the icon font. Nothing else: preloading the text faces
was measured and rejected. A browser cannot paint at all until the two Bootstrap
stylesheets arrive, and on a slow connection a preloaded font is competing with
them for the same pipe — one preloaded text face cost 308 ms of first paint and
all three cost 528 ms, to buy 0.017 of layout shift. The icon font is 1.4 KB
after subsetting, so it costs nothing measurable and it stops the navigation
chrome from resizing when the glyphs arrive.

One script cannot simply be deferred. Quarto initialises the lightbox from an
inline call in the body that runs while the document is still parsing, so a
deferred `glightbox.min.js` would not be there yet. That one is moved instead:
the tag travels from the head down to immediately before the call that needs
it, at the very end of the body. The order of the two is preserved exactly, so
the behaviour cannot change — the parser simply no longer stops for it while
there is still a page to lay out.

    python3 tools/unblock_head.py    # post-render
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
SITE = REPO / "_site"

# Consumed by an inline script that runs during parsing, so it is relocated
# rather than deferred — see the module docstring.
RELOCATE = "glightbox.min.js"
RELOCATE_BEFORE = "GLightbox("

# Only the icon font — see the module docstring for the measurement that ruled
# the text faces out. `crossorigin` is not optional: fonts are fetched in
# anonymous CORS mode, and a preload without it is a second, wasted download.
PRELOAD = (("site_libs/bootstrap/bootstrap-icons-subset.woff2", "font/woff2"),)

# `[^>]*` would end the tag at the first `>`, including one inside a quoted
# attribute value. Nothing Quarto emits today contains such a URL, but a tag cut
# in half is written back as broken HTML, so match quoted runs properly.
SCRIPT_RE = re.compile(r"""<script\b(?:[^>"']|"[^"]*"|'[^']*')*\bsrc=(?:[^>"']|"[^"]*"|'[^']*')*>""", re.I)
FONTS_CSS_RE = re.compile(r'<link[^>]+href="([^"]*)fonts/fonts\.css"', re.I)
# Each preload carries its own mark, so re-deriving removes exactly this pass's
# links and only those: one put in the head by Quarto or through
# `include-in-header` is none of this script's business. A mark on the tag
# rather than comments around the group means there is no pairing to get wrong
# — nothing to over-match across, and nothing to leave orphaned.
MARK = "data-unblock-head"
MARKED_RE = re.compile(r"\n<link\b[^>]*\b" + MARK + r"\b[^>]*>")


def defer_scripts(head: str) -> tuple[str, int]:
    """Mark every head script `defer` that is safe to defer."""
    count = 0

    def rewrite(match: re.Match[str]) -> str:
        nonlocal count
        tag = match.group(0)
        lowered = tag.lower()
        if any(a in lowered for a in (" defer", " async", 'type="module"')):
            return tag  # already deferred, or a module, which defers by spec
        if RELOCATE in lowered:
            return tag  # moved out of the head entirely instead
        count += 1
        return tag[:-1].rstrip() + " defer>"

    return SCRIPT_RE.sub(rewrite, head), count


def preload_links(head: str, site: pathlib.Path) -> str:
    """Font preloads, with the page's own path prefix and the required CORS mode.

    A preload for a file that is not there is worse than no preload, so anything missing is skipped — which is what
    happens when `icon_subset.py` decides to leave Quarto's full icon font in place.
    """
    prefix_match = FONTS_CSS_RE.search(head)
    if not prefix_match:
        return ""
    prefix = prefix_match.group(1)
    return "".join(
        f'\n<link rel="preload" href="{prefix}{path}" as="font" type="{mime}" crossorigin {MARK}>'
        for path, mime in PRELOAD
        if (site / path).exists()
    )


def relocate_lightbox(head: str, body: str) -> tuple[str, str]:
    """Move the lightbox library down to the inline call that constructs it.

    Both halves have to be found for the move to happen. If Quarto ever emits this differently the tag stays where it
    is, which is merely the status quo.
    """
    tag = next((m for m in SCRIPT_RE.finditer(head) if RELOCATE in m.group(0)), None)
    if tag is None:
        return head, body
    closing = head.find("</script>", tag.end())
    call = body.find(RELOCATE_BEFORE)
    consumer = body.rfind("<script", 0, call) if call != -1 else -1
    if closing == -1 or consumer == -1:
        return head, body

    element = head[tag.start() : closing + len("</script>")]
    head = head[: tag.start()] + head[closing + len("</script>") :]
    return head, body[:consumer] + element + "\n" + body[consumer:]


def process(page: pathlib.Path, site: pathlib.Path) -> bool:
    """Rewrite one page. Returns whether anything changed."""
    html = page.read_text()
    end = html.find("</head>")
    if end == -1:
        return False
    head, rest = html[:end], html[end:]

    # Quarto leaves site_libs alone on an incremental render, so a page can
    # arrive here already carrying a previous run's preload — for a file that
    # `icon_subset.py` may since have removed. Drop what the last run added and
    # derive it again from what is on disk now. The other two passes are already
    # idempotent: a script that has `defer` is skipped, and the lightbox tag is
    # only moved while it is still in the head.
    head = MARKED_RE.sub("", head)

    head, rest = relocate_lightbox(head, rest)
    head, _ = defer_scripts(head)
    links = preload_links(head, site)
    if links:
        opening = head.find(">", head.lower().find("<head")) + 1
        head = head[:opening] + links + head[opening:]
    if head + rest == html:
        return False

    page.write_text(head + rest)
    return True


def rendered_html() -> bool:
    """Report whether the render that called this actually produced HTML.

    Quarto names what it has just written. On `quarto render --to pdf` that is one PDF, and the HTML site left over from
    an earlier render is not this script's to rewrite. The list normally arrives in the environment, but
    `QUARTO_USE_FILE_FOR_PROJECT_OUTPUT_FILES` redirects it into a file — read both, or setting that flag would quietly
    restore the behaviour this exists to avoid. Outside a render neither is set, and then the caller means it.

    Deliberately a copy of the same predicate in `icon_subset.py`: the two run as separate processes and each has to
    stand on its own, and a wrong answer in one of them is a site where half the passes ran.
    """
    listed = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES")
    if listed is None:
        redirected = os.environ.get("QUARTO_USE_FILE_FOR_PROJECT_OUTPUT_FILES")
        if redirected and pathlib.Path(redirected).is_file():
            listed = pathlib.Path(redirected).read_text()
    return listed is None or any(f.endswith(".html") for f in listed.split())


def main() -> int:
    """Run the pass over every rendered page."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=pathlib.Path, default=SITE)
    args = parser.parse_args()
    if not args.site.exists() or not rendered_html():
        return 0
    changed = sum(process(page, args.site) for page in args.site.rglob("*.html"))
    print(f"unblock_head: {changed} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
