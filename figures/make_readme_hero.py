"""Generate the README banner: the arc of the course, drawn as six motifs.

The banner this replaced was `01b-course-overview/cnn_layer_activations.jpg`,
which `SOURCES.md` marks `unidentified` — nobody has traced where that bitmap
came from, which is a poor thing to put on the front page of a repository whose
whole claim is that its diagrams are its own. Everything here is drawn from the
shared primitives, so the banner is MIT-licensed like the rest of the notes.

Two constraints come from GitHub rather than from the site:

- **No `currentColor`.** Every other figure in this project takes its ink from
  the page, which is what makes one file work in both colour schemes. A README
  image is rendered in isolation inside an `<img>`, where `currentColor` falls
  back to black and vanishes on GitHub's dark theme. So ink is set explicitly
  here, and only here.
- **An explicit background.** For the same reason: the figure sits on its own
  plate rather than inheriting one. The plate is `#eef1f5`, the same value
  `theme-dark.scss` gives `$cs-figure-plate`, so the banner and the in-page
  figures are the same object.

Run from the repository root; rewrites figures/readme-hero.svg.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _svg import (
    GREEN,
    INDIGO,
    INK,
    PURPLE,
    TEAL,
    YELLOW,
    arrow,
    circle,
    line,
    poly,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path("figures/readme-hero.svg")

W, H = 1280, 420
PLATE = "#eef1f5"
CY = 268  # vertical centre of the motif strip
MOTIF_W, GAP = 140, 46


def band(x, y, w, h, colour, op=0.55, sw=1.0):
    """A filled rectangle with explicit ink, since currentColor is unavailable."""
    return rect(x, y, w, h, fill=colour, op=op, stroke=INK, sw=sw, so=0.55)


def m_pixels(x):
    """An image: a small grid of coloured cells."""
    out = ""
    cell, n = 15, 5
    x0, y0 = x + (MOTIF_W - n * cell) / 2, CY - n * cell / 2
    vals = [
        [0.10, 0.18, 0.30, 0.22, 0.12],
        [0.20, 0.55, 0.78, 0.48, 0.20],
        [0.32, 0.80, 0.95, 0.70, 0.30],
        [0.22, 0.60, 0.75, 0.52, 0.24],
        [0.12, 0.22, 0.34, 0.26, 0.14],
    ]
    ramp = [PURPLE, INDIGO, TEAL, GREEN, YELLOW]
    for r, row in enumerate(vals):
        for c, v in enumerate(row):
            out += rect(
                x0 + c * cell,
                y0 + r * cell,
                cell,
                cell,
                fill=ramp[min(int(v * len(ramp)), len(ramp) - 1)],
                op=0.85,
                stroke=PLATE,
                sw=0.8,
            )
    return out


def m_linear(x):
    """A linear classifier: two clouds and the boundary between them."""
    out = ""
    cx = x + MOTIF_W / 2
    left = [(-40, -26), (-30, -6), (-46, 10), (-22, 24), (-36, 32), (-14, -20)]
    right = [(20, -30), (38, -12), (16, 6), (42, 18), (26, 30), (46, -28)]
    for dx, dy in left:
        out += circle(cx + dx, CY + dy, 5.0, fill=INDIGO, stroke="none", op=0.9)
    for dx, dy in right:
        out += circle(cx + dx, CY + dy, 5.0, fill=GREEN, stroke="none", op=0.9)
    out += line(cx - 4, CY - 46, cx + 8, CY + 46, stroke=PURPLE, sw=2.4, op=0.95)
    return out


def m_conv(x):
    """A convolutional stack: resolution traded for channels."""
    out = ""
    spec = [(0, 14, 92), (30, 24, 66), (72, 36, 44)]
    for dx, w, h in spec:
        out += band(x + 14 + dx, CY - h / 2, w, h, TEAL, op=0.6)
    return out


def m_residual(x):
    """A residual block: the shortcut that made depth trainable."""
    out = ""
    cx = x + MOTIF_W / 2
    out += band(cx - 34, CY - 20, 68, 40, INDIGO, op=0.5)
    out += f'  <path d="M {cx - 54:.1f} {CY:.1f} C {cx - 40:.1f} {CY - 62:.1f}, {cx + 30:.1f} {CY - 62:.1f}, {cx + 44:.1f} {CY - 8:.1f}" fill="none" stroke="{PURPLE}" stroke-width="2.4" stroke-opacity="0.95"/>\n'
    out += circle(cx + 48, CY, 10, fill=PLATE, stroke=PURPLE, sw=2.0)
    out += text(cx + 48, CY + 5, "+", size=15, fill=INK)
    return out


def m_attention(x):
    """Attention: every position consulting every other."""
    out = ""
    cell, n = 13, 6
    x0, y0 = x + (MOTIF_W - n * cell) / 2, CY - n * cell / 2
    ramp = [PURPLE, INDIGO, TEAL, GREEN, YELLOW]
    for r in range(n):
        for c in range(n):
            # A causal, decaying pattern: bright on the diagonal, empty above it.
            v = 0.0 if c > r else max(0.08, 0.95 - 0.26 * (r - c))
            out += rect(
                x0 + c * cell,
                y0 + r * cell,
                cell,
                cell,
                fill=ramp[min(int(v * len(ramp)), len(ramp) - 1)] if v else PLATE,
                op=0.9 if v else 1.0,
                stroke=PLATE,
                sw=0.8,
            )
    return out


def m_detection(x):
    """Detection and segmentation: objects, not a single label."""
    out = ""
    cx = x + MOTIF_W / 2
    out += rect(cx - 52, CY - 44, 104, 88, fill=INDIGO, op=0.14, stroke=INK, sw=1.0, so=0.5)
    out += rect(cx - 40, CY - 30, 44, 40, fill="none", stroke=GREEN, sw=2.4)
    out += rect(cx + 2, CY - 6, 40, 42, fill="none", stroke=YELLOW, sw=2.4)
    out += poly(
        [(cx - 34, CY + 6), (cx - 18, CY - 6), (cx - 4, CY + 8), (cx - 20, CY + 22)],
        fill=TEAL,
        stroke="none",
        op=0.55,
    )
    return out


MOTIFS = [
    (m_pixels, "pixels"),
    (m_linear, "a linear boundary"),
    (m_conv, "convolution"),
    (m_residual, "depth"),
    (m_attention, "attention"),
    (m_detection, "objects"),
]


def build():
    """Assemble the banner."""
    body = rect(0, 0, W, H, fill=PLATE, stroke="none")
    body += rect(0, 0, W, 5, fill=TEAL, stroke="none", op=0.9)

    body += text(W / 2, 92, "CS231n 2025 Lecture Notes", size=46, fill=INK, weight="700")
    body += text(
        W / 2,
        130,
        "Deep Learning for Computer Vision — written from the Spring 2025 lectures",
        size=19,
        fill=INK,
        op=0.72,
    )

    total = len(MOTIFS) * MOTIF_W + (len(MOTIFS) - 1) * GAP
    x = (W - total) / 2
    spots = []
    for fn, label in MOTIFS:
        body += fn(x)
        body += text(x + MOTIF_W / 2, CY + 78, label, size=14.5, fill=INK, op=0.75)
        spots.append(x)
        x += MOTIF_W + GAP

    for x0 in spots[:-1]:
        body += arrow(
            x0 + MOTIF_W + 8,
            CY,
            x0 + MOTIF_W + GAP - 8,
            CY,
            sw=1.5,
            op=0.42,
            stroke=INK,
        )

    body += text(
        W / 2,
        380,
        "Every diagram in these notes is drawn by a script committed beside it. This one included.",
        size=15,
        fill=INK,
        op=0.6,
        style="italic",
    )
    return wrap(
        W,
        H,
        "readmeHero",
        "CS231n 2025 Lecture Notes",
        "A banner showing the arc of the course as six motifs left to right: a "
        "grid of image pixels, two point clouds split by a linear boundary, a "
        "convolutional stack trading resolution for channels, a residual block "
        "with its shortcut, a causal attention matrix, and an image carrying "
        "object boxes and a segmentation mask.",
        body,
    )


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT} ({W}x{H})")
    print("no currentColor:", "currentColor" not in OUT.read_text())
