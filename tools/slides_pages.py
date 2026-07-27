#!/usr/bin/env python3
"""Print selected pages of an extracted deck, or render them to PNG to actually look at.

Text mode is the default and costs almost nothing. Use --png only for pages the
contents map flags as image-only, or when a diagram's layout is the point.

    python3 tools/slides_pages.py slides/2025/text/lecture_1_part_1.txt 12-18,44
    python3 tools/slides_pages.py slides/2025/lecture_1_part_1.pdf 44 --png --outdir /tmp/px
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

PAGE_RE = re.compile(r"^=== PAGE (\d+) ===$", re.M)


def parse_ranges(spec: str) -> list[int]:
    """Expand a '3,7,12-15' page spec into a sorted list of page numbers."""
    pages: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = (int(x) for x in chunk.split("-", 1))
            pages.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            pages.add(int(chunk))
    return sorted(pages)


def main() -> int:
    """Slice pages out of the extracted text, or rasterise them from the source PDF."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=pathlib.Path, help=".txt from slides_text.py, or the .pdf when using --png")
    ap.add_argument("pages", help="page spec, e.g. 12-18,44")
    ap.add_argument("--png", action="store_true", help="render pages as images instead of printing text")
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("."))
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    wanted = parse_ranges(args.pages)
    if not wanted:
        print("no pages requested", file=sys.stderr)
        return 1

    if args.png:
        args.outdir.mkdir(parents=True, exist_ok=True)
        for n in wanted:
            out = args.outdir / f"{args.source.stem}_p{n:03d}"
            subprocess.run(
                ["pdftoppm", "-png", "-r", str(args.dpi), "-f", str(n), "-l", str(n), str(args.source), str(out)],
                check=True,
            )
            print(f"{out}-{n}.png")
        return 0

    text = args.source.read_text(encoding="utf-8", errors="replace")
    marks = [(int(m.group(1)), m.start(), m.end()) for m in PAGE_RE.finditer(text)]
    if not marks:
        print(f"{args.source} has no page markers — run tools/slides_text.py first", file=sys.stderr)
        return 1

    bounds = {num: (end, marks[i + 1][1] if i + 1 < len(marks) else len(text)) for i, (num, _, end) in enumerate(marks)}
    for n in wanted:
        if n not in bounds:
            print(f"=== PAGE {n} === (out of range)")
            continue
        lo, hi = bounds[n]
        print(f"=== PAGE {n} ==={text[lo:hi].rstrip()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
