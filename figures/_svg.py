"""Shared SVG primitives for the chapter figure scripts.

Every `figures/NN-slug/make_diagrams.py` writes hand-authored SVG by string
concatenation, and up to this point each one carried its own copy of the same
dozen emitters. This module holds the ones that are genuinely a single
function, so a chapter script is layout and nothing else.

What is deliberately *not* here, because the name is shared but the function is
not:

- `path` — chapters 3 and 4 use that name for a builder that turns a list of
  points into a `d` attribute string; chapters 6 and 7 use it for an emitter
  that returns a whole `<path>` element. Two different things.
- `op_node` — chapter 4's is a 21px disc for a computational-graph node,
  chapter 7's an 11px inline operator on a signal path.
- `cuboid` — chapter 6 added a stroke-opacity parameter whose default changes
  chapter 5's output, so unifying them would rewrite figures that are already
  published.

Conventions the emitters assume: `currentColor` for text and rules so both
colour schemes work from one file, viridis for data, a `title`/`desc` pair with
namespaced ids for screen readers, and no background rect — the site supplies a
light figure plate in both schemes.

Chapter scripts are run from the repository root, so they reach this module by
putting its directory on the path:

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from _svg import FONT, INK, TEAL, arrow, line, rect, text, wrap
"""

import math

# Viridis, sampled at five points, plus the near-black used for labels.
PURPLE = "#440154"
INDIGO = "#3b528b"
TEAL = "#21918c"
GREEN = "#5ec962"
YELLOW = "#fde725"
INK = "#1b1b1b"
FONT = "Source Serif 4, Georgia, serif"
MONO = "JetBrains Mono, SFMono-Regular, Menlo, monospace"


def wrap(width, height, ids, title, desc, body):
    """Assemble a complete SVG document."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"'
        f' width="{width}" height="{height}"\n'
        f'     role="img" aria-labelledby="{ids}Title {ids}Desc"'
        f' font-family="{FONT}">\n'
        f'  <title id="{ids}Title">{title}</title>\n'
        f'  <desc id="{ids}Desc">{desc}</desc>\n'
        f"{body}"
        "</svg>\n"
    )


def text(
    x,
    y,
    s,
    size=13,
    fill="currentColor",
    anchor="middle",
    font=None,
    style="",
    weight=None,
    op=1.0,
):
    """One text element. Every caller passes explicit coordinates."""
    f = f' font-family="{font}"' if font else ""
    st = f' font-style="{style}"' if style else ""
    w = f' font-weight="{weight}"' if weight else ""
    o = f' fill-opacity="{op}"' if op != 1.0 else ""
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"'
        f' text-anchor="{anchor}"{f}{st}{w}{o}>{s}</text>\n'
    )


def rect(
    x, y, w, h, fill="none", stroke="currentColor", sw=1.0, op=1.0, so=1.0, rx=None
):
    """One rectangle."""
    r = f' rx="{rx}"' if rx else ""
    return (
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"'
        f' fill="{fill}" fill-opacity="{op}" stroke="{stroke}"'
        f' stroke-width="{sw}" stroke-opacity="{so}"{r}/>\n'
    )


def circle(cx, cy, r, fill="none", stroke="currentColor", sw=1.0, op=1.0, so=1.0):
    """One circle."""
    return (
        f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"'
        f' fill-opacity="{op}" stroke="{stroke}" stroke-width="{sw}"'
        f' stroke-opacity="{so}"/>\n'
    )


def line(x0, y0, x1, y1, stroke="currentColor", sw=1.0, op=1.0, dash=None):
    """One line segment."""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'  <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"'
        f' stroke="{stroke}" stroke-width="{sw}" stroke-opacity="{op}"{d}/>\n'
    )


def poly(points, fill="none", stroke="currentColor", sw=1.0, op=1.0, so=1.0):
    """One closed polygon."""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'  <polygon points="{pts}" fill="{fill}" fill-opacity="{op}"'
        f' stroke="{stroke}" stroke-width="{sw}" stroke-opacity="{so}"/>\n'
    )


def polyline(points, stroke="currentColor", sw=1.6, op=1.0, dash=None):
    """One open polyline."""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'  <polyline points="{pts}" fill="none" stroke="{stroke}"'
        f' stroke-width="{sw}" stroke-opacity="{op}"{d}'
        ' stroke-linejoin="round" stroke-linecap="round"/>\n'
    )


def arrow(x0, y0, x1, y1, sw=1.4, op=0.75, head=6.0, stroke="currentColor", dash=None):
    """A plain line with a solid triangular head; no markers, no defs."""
    ang = math.atan2(y1 - y0, x1 - x0)
    bx, by = x1 - head * math.cos(ang), y1 - head * math.sin(ang)
    out = line(x0, y0, bx, by, sw=sw, op=op, stroke=stroke, dash=dash)
    tips = [
        (x1, y1),
        (bx - head * 0.45 * math.sin(ang), by + head * 0.45 * math.cos(ang)),
        (bx + head * 0.45 * math.sin(ang), by - head * 0.45 * math.cos(ang)),
    ]
    return out + poly(tips, fill=stroke, stroke="none", op=op)


def mapper(dom, rng, px, py):
    """Build a function mapping data coordinates into a pixel box."""
    d0, d1 = dom
    r0, r1 = rng
    x0, x1 = px
    y0, y1 = py

    def to_px(x, y):
        return (
            x0 + (x - d0) / (d1 - d0) * (x1 - x0),
            y1 - (y - r0) / (r1 - r0) * (y1 - y0),
        )

    return to_px
