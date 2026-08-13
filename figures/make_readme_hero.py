"""Generate the README banner: the wordmark restored from noise.

The banner this replaced drew the arc of the course as six motifs and stopped
at *objects* — lecture 9. That was accurate when it was written and stopped
being accurate as the book grew, and a banner that advertises half a course on
the front page of the repository is the same kind of stale promise the README
table used to carry. Rather than extend the row with four more motifs, which
only postpones the problem to lecture 19 of some future course, this banner
says something that stays true however many chapters exist.

It is the wordmark on a coarse pixel grid — the object of lecture 1, and the
thing every later representation is a departure from — corrupted by the forward
process of lectures 13 and 14 and read left to right as the reverse process:

    x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps

`cosine_abar` is the Nichol-Dhariwal schedule with s = 0.008, evaluated rather
than chosen to look right, and the value printed under each panel is what the
panel was actually drawn with. The banner is therefore made by the thing the
book ends on, out of the thing it starts from, which is what the strapline
underneath has always claimed about every figure in these notes.

One deliberate departure from the schedule: the final panel is binarised at
abar = 1 rather than left as the continuous blend. At abar = 1 the noise term
is zero and the blend is already exactly the wordmark, so this changes no
value — it only forces the two-colour rendering that makes the letters read as
letters instead of as a bright patch, which matters at the width GitHub serves.

Two constraints come from GitHub rather than from the site, and they are why
this file does not look like the other figure scripts:

- **No `currentColor`.** Every other figure in this project takes its ink from
  the page, which is what makes one file work in both colour schemes. A README
  image is rendered in isolation inside an `<img>`, where `currentColor` falls
  back to black and vanishes on GitHub's dark theme. So ink is set explicitly
  here, and only here; `__main__` asserts the string never appears.
- **An explicit background.** For the same reason: the figure sits on its own
  plate rather than inheriting one. The plate is `#eef1f5`, the same value
  `theme-dark.scss` gives `$cs-figure-plate`, so the banner and the in-page
  figures are the same object.

Run from the repository root; rewrites figures/readme-hero.svg.
"""

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _svg import (
    INK,
    MONO,
    PURPLE,
    TEAL,
    line,
    poly,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path("figures/readme-hero.svg")

W, H = 1280, 420
PLATE = "#eef1f5"
WORD = "CS231n"
STEPS = (0.90, 0.30, 0.0)  # left to right: noisiest first, clean last
# The middle t is chosen, not arbitrary: below abar ~ 0.75 the ground's own
# variance reaches the bright end of the ramp and competes with the lit
# cells, so the wordmark stops emerging and the panel reads as more noise.
GRID_Y = 196
CELL = 7.0
SEED = 7

# Viridis, sampled at five points and interpolated. The banner needs a ramp
# rather than the five constants, because a noisy cell can land anywhere.
VIRIDIS = ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725")

# A 5x7 bitmap face, enough for the six glyphs the wordmark uses. Drawn here
# rather than measured from a font so the banner has no font dependency at all:
# every other glyph on the plate is real text, but these are pixels on purpose.
FONT5 = {
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "S": ("01111", "10000", "10000", "01110", "00001", "10001", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "n": ("00000", "00000", "10110", "11001", "10001", "10001", "10001"),
}


def viridis(t):
    """Sample the viridis ramp at t, clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    s = t * (len(VIRIDIS) - 1)
    i = min(int(s), len(VIRIDIS) - 2)
    f = s - i
    a, b = VIRIDIS[i], VIRIDIS[i + 1]
    return "#" + "".join(
        f"{round(int(a[k : k + 2], 16) + (int(b[k : k + 2], 16) - int(a[k : k + 2], 16)) * f):02x}" for k in (1, 3, 5)
    )


def cosine_abar(t):
    """Cumulative alpha at time t under the Nichol-Dhariwal cosine schedule."""
    s = 0.008

    def f(u):
        return math.cos((u + s) / (1 + s) * math.pi / 2) ** 2

    return f(t) / f(0.0)


def wordmark_bits():
    """The wordmark as a list of rows of 0/1, one column of gap per glyph."""
    cols = len(WORD) * 6 - 1
    rows = []
    for r in range(7):
        row = []
        for c in range(cols):
            glyph, within = divmod(c, 6)
            row.append(1 if within < 5 and FONT5[WORD[glyph]][r][within] == "1" else 0)
        rows.append(row)
    return rows


def noisy(bits, abar, noise):
    """One forward-process sample: sqrt(abar) x0 + sqrt(1-abar) eps, in [0, 1]."""
    out = []
    for r, row in enumerate(bits):
        out.append(
            [
                max(0.0, min(1.0, math.sqrt(abar) * v + math.sqrt(1 - abar) * (noise[r][c] * 0.5 + 0.5)))
                for c, v in enumerate(row)
            ]
        )
    return out


def ink(x, y, s, size=13, op=1.0, anchor="middle", weight=None, font=None, style=""):
    """Text with explicit ink, since currentColor is unavailable here."""
    return text(x, y, s, size=size, fill=INK, anchor=anchor, op=op, weight=weight, font=font, style=style)


def build():
    """Assemble the banner."""
    bits = wordmark_bits()
    rows, cols = len(bits), len(bits[0])
    rng = random.Random(SEED)
    noise = [[rng.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]

    panel_w = cols * CELL
    gap = (W - len(STEPS) * panel_w) / (len(STEPS) + 1)

    body = rect(0, 0, W, H, fill=PLATE, stroke="none")
    body += rect(0, 0, W, 5, fill=TEAL, stroke="none", op=0.9)
    body += ink(W / 2, 84, "CS231n 2025 Lecture Notes", size=44, weight="700")
    body += ink(
        W / 2, 120, "Deep Learning for Computer Vision — written from the Spring 2025 lectures", size=18, op=0.72
    )
    body += ink(W / 2, 172, "→  reverse process: noise to wordmark", size=13, op=0.6, font=MONO)

    mid = GRID_Y + rows * CELL / 2
    for i, t in enumerate(STEPS):
        abar = cosine_abar(t)
        x0 = gap + i * (panel_w + gap)
        field = noisy(bits, abar, noise)
        clean = abar > 0.999
        for r in range(rows):
            for c in range(cols):
                # At abar = 1 the blend is already exactly the wordmark; drawing
                # it in two colours only makes the letters legible at width.
                colour = (PURPLE if bits[r][c] else PLATE) if clean else viridis(field[r][c])
                body += rect(x0 + c * CELL, GRID_Y + r * CELL, CELL, CELL, fill=colour, stroke="none")
        body += ink(x0 + panel_w / 2, GRID_Y + rows * CELL + 26, f"t = {t:.2f}", size=13, op=0.8, font=MONO)
        body += ink(x0 + panel_w / 2, GRID_Y + rows * CELL + 44, f"ᾱ = {abar:.3f}", size=12, op=0.62, font=MONO)

        if i < len(STEPS) - 1:
            xa = x0 + panel_w + gap * 0.28
            xb = x0 + panel_w + gap * 0.72
            body += line(xa, mid, xb, mid, stroke=PURPLE, sw=1.6, op=0.75)
            body += poly([(xb, mid), (xb - 7, mid - 4), (xb - 7, mid + 4)], fill=PURPLE, stroke="none", op=0.75)

    body += ink(
        W / 2,
        396,
        "Every diagram in these notes is drawn by a script committed beside it. This one included.",
        size=15,
        op=0.6,
        style="italic",
    )
    return wrap(
        W,
        H,
        "readmeHero",
        "CS231n 2025 Lecture Notes",
        "A banner in three panels. The words CS231n are rendered as a coarse "
        "grid of coloured pixels: in the left panel they are buried in noise, "
        "in the middle panel partly visible, and in the right panel fully "
        "legible. Arrows run left to right, and the cumulative alpha of the "
        "diffusion schedule used for each panel is printed beneath it.",
        body,
    )


if __name__ == "__main__":
    svg = build()
    assert "currentColor" not in svg, "a README image cannot inherit its ink"
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT} ({W}x{H}, {len(svg) / 1024:.0f} KB)")
    print()
    print("Schedule actually drawn, evaluated from cosine_abar:")
    for t in STEPS:
        ab = cosine_abar(t)
        print(f"  t = {t:.2f}   ᾱ = {ab:.4f}   signal {math.sqrt(ab):.3f}   noise {math.sqrt(1 - ab):.3f}")
    # The endpoints are what make the banner readable as a process at all.
    assert cosine_abar(0.0) > 0.999, "the clean panel must be clean"
    assert cosine_abar(STEPS[0]) < 0.05, "the noisy panel must be mostly noise"
    print()
    print(f"  endpoints check out: ᾱ(0) = {cosine_abar(0.0):.4f}, ᾱ({STEPS[0]}) = {cosine_abar(STEPS[0]):.4f}")
