#!/usr/bin/env python3
"""Extract a lecture deck to page-addressable text, plus a one-line-per-page contents map.

A CS231n deck is 100-300 pages and several megabytes; reading the whole thing is the
single most expensive thing you can do while writing notes. This produces two files:

    slides/2025/text/lecture_1_part_1.txt    every page, wrapped in page markers
    slides/2025/text/lecture_1_part_1.map    one line per page: the page's title line

Read the .map to find the pages you need, then pull only those pages with
`tools/slides_pages.py`. Never read the .txt end to end.

    python3 tools/slides_text.py slides/2025/lecture_1_part_1.pdf
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

PAGE_BREAK = "\f"
NOISE_RE = re.compile(r"^(fei-?fei li|ehsan adeli|lecture \d+|cs ?231n|stanford|\d+|.{0,2})$", re.I)


def page_title(page: str) -> str:
    """Best guess at what a slide is about: its first substantial line."""
    for line in page.splitlines():
        line = " ".join(line.split())
        if line and not NOISE_RE.match(line):
            return line[:100]
    return "(no text — likely a full-bleed image or diagram)"


def main() -> int:
    """Run pdftotext, split on form feeds, and write the page text and contents map."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=pathlib.Path)
    ap.add_argument("--outdir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if not args.pdf.is_file():
        print(f"no such deck: {args.pdf}", file=sys.stderr)
        return 1

    outdir = args.outdir or args.pdf.parent / "text"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        raw = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(args.pdf), "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except FileNotFoundError:
        print("pdftotext not found — install poppler (brew install poppler)", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"pdftotext failed: {exc.stderr[:300]}", file=sys.stderr)
        return 1

    pages = raw.split(PAGE_BREAK)
    if pages and not pages[-1].strip():
        pages.pop()

    body, contents = [], []
    for n, page in enumerate(pages, start=1):
        text = "\n".join(line.rstrip() for line in page.splitlines()).strip()
        body.append(f"=== PAGE {n} ===\n{text}\n")
        contents.append(f"{n:>4}  {page_title(page)}")

    stem = args.pdf.stem
    txt, mp = outdir / f"{stem}.txt", outdir / f"{stem}.map"
    banner = "# Stanford course material — local working copy, never commit or redistribute.\n"
    txt.write_text(banner + "\n".join(body), encoding="utf-8")
    mp.write_text(banner + "\n".join(contents) + "\n", encoding="utf-8")

    blank = sum(1 for c in contents if "(no text" in c)
    print(f"{txt}: {len(pages)} pages")
    print(f"{mp}: {len(pages)} entries, {blank} image-only pages (render those to look at them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
