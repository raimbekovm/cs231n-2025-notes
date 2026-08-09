#!/usr/bin/env python3
"""Find labels that collide with each other in the generated chapter diagrams.

    tools/check_figure_labels.py                          # every figure directory
    tools/check_figure_labels.py figures/12-self-.../     # one directory
    tools/check_figure_labels.py path/to/diagram.svg      # one file

This exists because of what actually goes wrong when a `make_diagrams.py` is
written. Geometry errors — an arrow that doubles back, a node placed out of flow
order — are caught by thinking before writing coordinates. What is *not* caught
that way is text landing on other text: a caption written at the same y as an
axis label, an annotation dropped onto the curve it annotates, a note that grew
by one word and reached its neighbour. Those are invisible in `quarto render`,
invisible to the linter and invisible to `check_links.py`, and the only thing
that has ever found them is a screenshot round — which costs far more than this.

Every `<text>` element carries its own x, y, font-size and text-anchor, so the
box each one occupies is recoverable without rendering anything. Widths are
estimated rather than measured, which is why the report is advisory: the two
fonts in use are proportional, so a flagged pair may be fine and a near miss may
not be flagged. Read the hits, do not trust the silence. It is a filter that
removes the obvious cases before the contact sheet, not a replacement for
looking at the figures.

Exit status is 1 if anything overlapped, so this is safe to run before a commit.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

TEXT = re.compile(
    r'<text\s+x="(-?[\d.]+)"\s+y="(-?[\d.]+)"\s+font-size="([\d.]+)"'
    r'([^>]*?)text-anchor="(\w+)"([^>]*)>(.*?)</text>',
    re.S,
)
VIEWBOX = re.compile(r'viewBox="0 0 ([\d.]+) ([\d.]+)"')
ENTITY = re.compile(r"&[a-z]+;|&#\d+;")
TAG = re.compile(r"<[^>]*>")

# Measured against the rendered figures, a label in the body serif averages
# about 0.39 of its font-size per glyph and the monospace face about 0.6, but a
# `<text>` element does not record which face it inherited. 0.45 sits between
# them: it slightly overstates serif and understates monospace, which is the
# right way round, since a missed collision costs a screenshot round and a false
# one costs a glance.
WIDTH_PER_CHAR = 0.45

# Two labels whose boxes overlap by less than this on either axis are treated as
# adjacent rather than colliding. Real collisions in these figures overlap by
# tens of pixels, so this only has to absorb estimation noise: the ascender
# factor below is generous for glyphs that have no descender, which puts two
# deliberately stacked labels a couple of pixels into each other.
SLACK = 3.0


def glyph_count(s):
    """Visible glyphs in a label.

    Nested `<tspan>` markup — which the chapters use for subscripts — is stripped
    first, and each entity counts as one character. Counting the raw string
    instead makes a two-glyph subscripted label look six hundred pixels wide.
    """
    return len(ENTITY.sub("x", TAG.sub("", s)).strip())


def boxes(svg):
    """Every text element's estimated bounding box, as (x0, y0, x1, y1, label)."""
    out = []
    for x, y, size, before, anchor, after, body in TEXT.findall(svg):
        # A rotated label — an axis title, usually — occupies a vertical box,
        # and pretending it is horizontal reports its own figure as broken.
        # There are few enough of them that skipping is better than modelling.
        if "transform=" in before + after:
            continue
        x, y, size = float(x), float(y), float(size)
        label = body.strip()
        if not label:
            continue
        width = glyph_count(body) * size * WIDTH_PER_CHAR
        if anchor == "middle":
            x0 = x - width / 2
        elif anchor == "end":
            x0 = x - width
        else:
            x0 = x
        # y is the baseline; ascenders reach about 0.78 of the size above it and
        # descenders about 0.22 below.
        out.append((x0, y - size * 0.78, x0 + width, y + size * 0.22, label))
    return out


def check(path, fail):
    """Report labels that overlap each other or run outside the canvas."""
    svg = path.read_text()
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path

    seen = VIEWBOX.search(svg)
    if seen:
        w, h = float(seen.group(1)), float(seen.group(2))
        for x0, y0, x1, y1, label in boxes(svg):
            if x0 < -SLACK or x1 > w + SLACK or y0 < -SLACK or y1 > h + SLACK:
                fail(f'{rel}: "{label}" runs outside the {w:.0f}x{h:.0f} canvas')

    found = boxes(svg)
    for i, (ax0, ay0, ax1, ay1, a) in enumerate(found):
        for bx0, by0, bx1, by1, b in found[i + 1 :]:
            dx = min(ax1, bx1) - max(ax0, bx0)
            dy = min(ay1, by1) - max(ay0, by0)
            if dx > SLACK and dy > SLACK:
                fail(f'{rel}: "{a}" overlaps "{b}" by {dx:.0f}x{dy:.0f}px')


def targets(argv):
    """Resolve the command line to a list of SVG files."""
    if len(argv) > 1:
        found = []
        for arg in argv[1:]:
            p = pathlib.Path(arg)
            found.extend(sorted(p.glob("*.svg")) if p.is_dir() else [p])
        return found
    return sorted((ROOT / "figures").glob("*/diagram_*.svg"))


def main(argv):
    """Check every requested figure and report the total."""
    failures = []
    files = targets(argv)
    for path in files:
        check(path, failures.append)

    if failures:
        for message in failures:
            print(message)
        print(f"\n{len(failures)} overlap(s) across {len(files)} figure(s)")
        return 1
    print(f"no overlapping labels in {len(files)} figure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
