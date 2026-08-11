#!/usr/bin/env python3
"""Find what is wrong in a generated chapter diagram without rendering it.

    tools/check_figure_labels.py                          # every figure directory
    tools/check_figure_labels.py figures/12-self-.../     # one directory
    tools/check_figure_labels.py path/to/diagram.svg      # one file

This exists because of what actually goes wrong when a `make_diagrams.py` is
written. Geometry errors — an arrow that doubles back, a node placed out of flow
order — are caught by thinking before writing coordinates. Three other things are
not, and all three are invisible in `quarto render`, invisible to the linter and
invisible to `check_links.py`. Every one is recoverable from the coordinates the
generator already wrote, which is why this is cheaper than the screenshot round
that used to be the only way to find them.

**Text on text.** A caption written at the same y as an axis label, an annotation
dropped onto the curve it annotates, a note that grew by one word and reached its
neighbour.

**Text on a filled shape.** A legend that grew into a bar it has nothing to do
with. See `on_fill_edge` for why containment, not overlap, is the test — labels
belong inside filled shapes all the time, and only a label that lies across one
without sitting in it is a defect.

**Geometry off the canvas.** This is the worst of the three and the reason the
checks were extended. A curve whose peak leaves the viewBox is *silently*
truncated: the figure renders, looks plausible, passes every other check, and
quietly asserts something false. A label collision at least looks wrong.

Widths are estimated rather than measured, which is why the report is advisory:
the two fonts in use are proportional, so a flagged pair may be fine and a near
miss may not be flagged. Read the hits, do not trust the silence. It is a filter
that removes the mechanical cases before the contact sheet, not a replacement for
looking at the figures — a diagram can be perfectly composed and still say
something the subject makes impossible, and no script sees that.

Exit status is 1 if anything was reported, so this is safe to run before a commit.
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

RECT = re.compile(
    r'<rect x="(-?[\d.]+)" y="(-?[\d.]+)" width="([\d.]+)" height="([\d.]+)"'
    r' fill="([^"]*)" fill-opacity="([\d.]+)"',
)
POINTS = re.compile(r'<(polyline|polygon) points="([^"]*)"')
CIRCLE = re.compile(r'<circle cx="(-?[\d.]+)" cy="(-?[\d.]+)" r="([\d.]+)"')
LINE = re.compile(
    r'<line x1="(-?[\d.]+)" y1="(-?[\d.]+)" x2="(-?[\d.]+)" y2="(-?[\d.]+)"',
)

# A rect this solid reads as a filled area — a bar, a shaded cell — rather than
# as a tinted container. Text belongs *inside* one of those or nowhere near it;
# see `on_fill_edge` for why that distinction is the whole check. Node boxes in
# these chapters are drawn at 0.13-0.14, well under the threshold, so the
# ordinary case of a label sitting in its own box never reaches this code.
SOLID_FILL = 0.25

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

    Nested `<tspan>` markup — which the chapters use for subscripts — is stripped first, and each entity counts as one
    character. Counting the raw string instead makes a two-glyph subscripted label look six hundred pixels wide.
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
        # A label at zero opacity is not on the page, so it cannot collide with
        # anything. Chapter 6 draws a hidden copy of every layer name, and
        # counting those reported twenty collisions in a figure that has none.
        if re.search(r'fill-opacity="0(\.0+)?"', before + after):
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


def solid_rects(svg):
    """Every rect filled solidly enough that a label crossing it is a defect."""
    out = []
    for x, y, w, h, fill, op in RECT.findall(svg):
        if fill == "none" or float(op) < SOLID_FILL:
            continue
        x, y = float(x), float(y)
        out.append((x, y, x + float(w), y + float(h)))
    return out


def _contains(rect, box):
    """True if a filled rect encloses a label's box, within the slack."""
    ax0, ay0, ax1, ay1 = box
    rx0, ry0, rx1, ry1 = rect
    return ax0 >= rx0 - SLACK and ax1 <= rx1 + SLACK and ay0 >= ry0 - SLACK and ay1 <= ry1 + SLACK


def _touches(rect, box):
    """True if a label's box shares more than the slack with a filled rect on both axes."""
    ax0, ay0, ax1, ay1 = box
    rx0, ry0, rx1, ry1 = rect
    return min(ax1, rx1) - max(ax0, rx0) > SLACK and min(ay1, ry1) - max(ay0, ry0) > SLACK


def on_fill_edge(box, rects):
    """True if a label lies across filled shapes without sitting inside any of them.

    This is the discriminator that makes the check usable, and it took two passes to get right. Putting a label inside a
    filled shape is deliberate and ubiquitous — the class name inside a bar segment, the weight inside a kernel cell —
    so flagging every text-over-rect pair reports a hundred intentional labels. Flagging every label that crosses *some*
    rect's edge is not enough either: chapter 10 stacks overlapping tiles, and each tile's own centred label necessarily
    reaches over its neighbours.

    Containment in *any* one filled shape is what marks a label as deliberate. What is left over is the real defect: a
    legend that grew into a bar it has nothing to do with, an annotation that drifted onto a shaded region. Nothing
    short of a screenshot could see that before.
    """
    touched = [r for r in rects if _touches(r, box)]
    if not touched:
        return False
    return not any(_contains(r, box) for r in touched)


def drawn_extents(svg):
    """Every drawn shape's extent, as (x0, y0, x1, y1, what).

    Labels already get their own canvas check. This covers the geometry, because a curve that leaves the viewBox is
    *silently* truncated: it renders as a perfectly plausible figure and nothing tells the reader that a peak was cut
    off. That is worse than a label collision, which at least looks wrong.
    """
    out = []
    for kind, points in POINTS.findall(svg):
        xs, ys = [], []
        for pair in points.split():
            px, _, py = pair.partition(",")
            xs.append(float(px))
            ys.append(float(py))
        if xs:
            out.append((min(xs), min(ys), max(xs), max(ys), kind))
    for cx, cy, r in CIRCLE.findall(svg):
        cx, cy, r = float(cx), float(cy), float(r)
        out.append((cx - r, cy - r, cx + r, cy + r, "circle"))
    for x0, y0, x1, y1 in LINE.findall(svg):
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        out.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), "line"))
    for x, y, w, h, _fill, _op in RECT.findall(svg):
        x, y = float(x), float(y)
        out.append((x, y, x + float(w), y + float(h), "rect"))
    return out


def check(path, fail):
    """Report labels that collide, and anything that leaves the canvas."""
    svg = path.read_text()
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path

    seen = VIEWBOX.search(svg)
    if seen:
        w, h = float(seen.group(1)), float(seen.group(2))
        for x0, y0, x1, y1, label in boxes(svg):
            if x0 < -SLACK or x1 > w + SLACK or y0 < -SLACK or y1 > h + SLACK:
                fail(f'{rel}: "{label}" runs outside the {w:.0f}x{h:.0f} canvas')
        for x0, y0, x1, y1, kind in drawn_extents(svg):
            over = max(-x0, x1 - w, -y0, y1 - h)
            if over > SLACK:
                fail(f"{rel}: a {kind} leaves the {w:.0f}x{h:.0f} canvas by {over:.0f}px")

    rects = solid_rects(svg)
    for x0, y0, x1, y1, label in boxes(svg):
        if on_fill_edge((x0, y0, x1, y1), rects):
            fail(f'{rel}: "{label}" lies across a filled shape without being inside it')

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
        print(f"\n{len(failures)} problem(s) across {len(files)} figure(s)")
        return 1
    print(f"nothing to report in {len(files)} figure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
