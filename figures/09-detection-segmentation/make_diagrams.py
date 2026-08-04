"""Generate the nine SVG diagrams for the lecture 9 chapter.

Everything here is structural rather than plotted: the shapes of dense-prediction
networks, the arithmetic of three upsampling operations, the three generations of
the R-CNN family, and the two ways to weight a feature map into a heatmap. The
only computed figure is the transposed-convolution example, whose output terms
are derived rather than written out by hand, and printed at the bottom so the
prose can be checked against them.

Conventions follow figures/06-cnn-architectures: currentColor for text and rules,
viridis for data, title/desc with namespaced ids, no background rect (the theme
supplies a light plate in both schemes).

Run from the repository root; rewrites all nine files.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _svg import (
    GREEN,
    INDIGO,
    MONO,
    PURPLE,
    TEAL,
    YELLOW,
    arrow,
    circle,
    line,
    poly,
    rect,
    text,
)
from _svg import wrap as _wrap

OUT = pathlib.Path("figures/09-detection-segmentation")


def wrap(width, height, ids, title, desc, body):
    """Assemble an SVG document (thin pass-through, kept for call-site brevity)."""
    return _wrap(width, height, ids, title, desc, body)


def cuboid(x, y, w, h, dx, dy, face, op=0.85, sw=0.9, so=0.6):
    """A box in oblique projection: front face at (x, y), back offset by (dx, dy)."""
    out = poly(
        [(x, y), (x + dx, y + dy), (x + w + dx, y + dy), (x + w, y)],
        fill=face,
        op=op * 0.5,
        sw=sw,
        so=so,
    )
    out += poly(
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        fill=face,
        op=op * 0.7,
        sw=sw,
        so=so,
    )
    out += rect(x, y, w, h, fill=face, op=op, sw=sw, so=so)
    return out


def grid(x, y, cols, rows, cell, sw=0.7, so=0.45, fill="none", op=1.0):
    """A plain rectangular lattice of `cols` x `rows` cells of side `cell`."""
    out = ""
    if fill != "none":
        out += rect(x, y, cols * cell, rows * cell, fill=fill, op=op, stroke="none")
    for c in range(cols + 1):
        out += line(x + c * cell, y, x + c * cell, y + rows * cell, sw=sw, op=so)
    for r in range(rows + 1):
        out += line(x, y + r * cell, x + cols * cell, y + r * cell, sw=sw, op=so)
    return out


def cells(x, y, values, cell, size=13, fill=None, op=0.9):
    """A row-major grid of labelled cells; `values` is a list of rows."""
    out = ""
    for r, row in enumerate(values):
        for c, v in enumerate(row):
            cx, cy = x + c * cell, y + r * cell
            shade = fill(v) if callable(fill) else fill
            out += rect(cx, cy, cell, cell, fill=shade or "none", op=op if shade else 1.0, so=0.5)
            out += text(cx + cell / 2, cy + cell / 2 + size * 0.36, str(v), size=size)
    return out


def pill(x, y, w, h, label, colour, size=12.5, sub=None, dash=None, op=0.16):
    """A rounded box with a centred label and an optional second line."""
    out = rect(x, y, w, h, fill=colour, op=op, stroke=colour, sw=1.3, so=0.85, rx=6)
    if dash:
        out = rect(x, y, w, h, fill="none", stroke=colour, sw=1.3, so=0.85, rx=6)
        out = out.replace("/>", f' stroke-dasharray="{dash}"/>')
    if sub:
        out += text(x + w / 2, y + h / 2 - 2, label, size=size)
        out += text(x + w / 2, y + h / 2 + 13, sub, size=size - 2.0, op=0.7)
    else:
        out += text(x + w / 2, y + h / 2 + size * 0.36, label, size=size)
    return out


# ---------------------------------------------------------------------------
# 1. Full-resolution stack against the encoder-decoder shape.
# ---------------------------------------------------------------------------


def fig_seg_shapes():
    """Two ways to build a fully convolutional network."""
    body = ""

    def block(x, w, h, ycent, colour, top, bot):
        """One feature map: height is spatial size, width is channel count."""
        out = rect(x, ycent - h / 2, w, h, fill=colour, op=0.55, sw=1.0, so=0.7)
        out += text(x + w / 2, ycent - h / 2 - 9, top, size=11, op=0.85)
        if bot:
            out += text(x + w / 2, ycent + h / 2 + 16, bot, size=11, op=0.7)
        return out

    # --- top row: constant resolution -------------------------------------
    y0 = 124
    body += text(40, 32, "Constant resolution", size=14, anchor="start", weight="600")
    body += text(
        40,
        50,
        "every layer is as wide as the image; the deepest receptive field is still small",
        size=11.5,
        anchor="start",
        op=0.7,
    )

    xs = [56, 152, 268, 384, 500, 616]
    widths = [12, 34, 34, 34, 34, 24]
    colours = [INDIGO, TEAL, TEAL, TEAL, TEAL, GREEN]
    tops = ["3", "D", "D", "D", "D", "C"]
    for x, w, colour, top in zip(xs, widths, colours, tops):
        body += block(x, w, 96, y0, colour, top, None)
    for i in range(len(xs) - 1):
        body += arrow(xs[i] + widths[i] + 4, y0, xs[i + 1] - 4, y0, sw=1.2, op=0.55)
    body += text(56 + 6, y0 + 66, "image", size=11, op=0.7)
    body += text(616 + 12, y0 + 66, "scores", size=11, op=0.7)
    body += text(
        700,
        y0 - 6,
        "cost of one",
        size=11.5,
        anchor="start",
        op=0.8,
    )
    body += text(700, y0 + 10, "layer × 1000", size=11.5, anchor="start", op=0.8)
    body += text(
        746,
        y0 + 40,
        "H × W held throughout",
        size=11,
        op=0.6,
    )

    # --- bottom row: encoder-decoder --------------------------------------
    y1 = 312
    body += text(40, 220, "Downsample, then upsample", size=14, anchor="start", weight="600")
    body += text(
        40,
        238,
        "deep layers are cheap and see far; an upsampling operation restores the output size",
        size=11.5,
        anchor="start",
        op=0.7,
    )

    spec = [
        (56, 12, 96, INDIGO, "3", "H"),
        (140, 22, 68, TEAL, "D₁", "H/2"),
        (240, 34, 46, TEAL, "D₂", "H/4"),
        (352, 50, 30, PURPLE, "D₃", "H/8"),
        (476, 34, 46, TEAL, "D₂", "H/4"),
        (580, 22, 68, TEAL, "D₁", "H/2"),
        (668, 24, 96, GREEN, "C", "H"),
    ]
    for x, w, h, colour, top, bot in spec:
        body += block(x, w, h, y1, colour, top, bot)
    for i in range(len(spec) - 1):
        x, w = spec[i][0], spec[i][1]
        body += arrow(x + w + 4, y1, spec[i + 1][0] - 4, y1, sw=1.2, op=0.55)

    body += line(150, 394, 350, 394, sw=1.0, op=0.45, dash="4 3")
    body += text(250, 410, "downsample", size=11.5, op=0.75)
    body += line(400, 394, 660, 394, sw=1.0, op=0.45, dash="4 3")
    body += text(530, 410, "upsample", size=11.5, op=0.75)
    body += text(
        760,
        y1 - 4,
        "same output",
        size=11.5,
        anchor="start",
        op=0.8,
    )
    body += text(760, y1 + 12, "shape, less work", size=11.5, anchor="start", op=0.8)

    return wrap(
        900,
        430,
        "segShapes",
        "Constant-resolution and encoder-decoder fully convolutional networks",
        "Two horizontal pipelines of feature maps drawn as rectangles whose "
        "height is spatial size and width is channel count. In the upper "
        "pipeline every map has the full height of the input. In the lower "
        "pipeline the maps shrink in height and grow in width towards a "
        "bottleneck, then reverse, ending at the input's height again.",
        body,
    )


# ---------------------------------------------------------------------------
# 2. Three parameter-free upsampling operations.
# ---------------------------------------------------------------------------

NN_OUT = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
NAILS_OUT = [[1, 0, 2, 0], [0, 0, 0, 0], [3, 0, 4, 0], [0, 0, 0, 0]]
# Positions recorded by a max-pool over the 4x4 block. Each recorded index must
# lie in the 2x2 quadrant its value was pooled from, so value 1 (the top-left
# quadrant's max) goes to a cell in rows 0-1, cols 0-1, and so on.
MAX_IDX = [(1, 1), (0, 2), (3, 0), (3, 3)]


def _max_unpool_out():
    """Place 1..4 at the recorded argmax positions; zeros elsewhere."""
    out = [[0] * 4 for _ in range(4)]
    for v, (r, c) in enumerate(MAX_IDX, start=1):
        out[r][c] = v
    return out


def fig_unpooling():
    """Nearest neighbour, bed of nails and max unpooling side by side."""
    body = ""
    cell = 26
    panels = [
        (30, "Nearest neighbour", NN_OUT, "each value copied into its whole block"),
        (322, "Bed of nails", NAILS_OUT, "one cell per block, the rest zero"),
        (614, "Max unpooling", _max_unpool_out(), "written to the recorded argmax"),
    ]

    def shade(v):
        """Fill non-zero cells; leave zeros blank."""
        return {1: INDIGO, 2: TEAL, 3: GREEN, 4: YELLOW}.get(v) if v else None

    for x0, title, out, note in panels:
        body += text(x0 + 128, 34, title, size=13.5, weight="600")
        body += cells(x0, 82, [[1, 2], [3, 4]], cell, size=13, fill=shade, op=0.55)
        body += text(x0 + cell, 82 + 2 * cell + 16, "2 × 2", size=11, op=0.7)
        body += arrow(x0 + 2 * cell + 10, 82 + cell, x0 + 2 * cell + 38, 82 + cell, sw=1.2, op=0.6)
        gx = x0 + 2 * cell + 48
        body += cells(gx, 56, out, cell, size=12.5, fill=shade, op=0.55)
        body += text(gx + 2 * cell, 56 + 4 * cell + 16, "4 × 4", size=11, op=0.7)
        body += text(x0 + 128, 238, note, size=11.5, op=0.72)

    # The recorded indices are the only thing distinguishing the third panel.
    x0 = 614
    gx = x0 + 2 * cell + 48
    for r, c in MAX_IDX:
        body += rect(gx + c * cell, 56 + r * cell, cell, cell, fill="none", stroke=PURPLE, sw=1.8, so=0.9)
    body += text(x0 + 128, 258, "outlined cells come from the paired max-pool", size=11, op=0.6)

    return wrap(
        900,
        280,
        "unpool",
        "Nearest neighbour, bed of nails and max unpooling",
        "Three panels, each showing a two by two grid of the values one to "
        "four being expanded into a four by four grid. Nearest neighbour "
        "replicates each value four times. Bed of nails writes each value to "
        "the top-left cell of its block. Max unpooling scatters the four "
        "values to irregular positions recorded by an earlier pooling layer.",
        body,
    )


# ---------------------------------------------------------------------------
# 3. Transposed convolution in one dimension.
# ---------------------------------------------------------------------------

STRIDE = 2
FILTER = ["x", "y", "z"]
INPUTS = ["a", "b"]


def transposed_1d(inputs, filt, stride):
    """Return the output terms of a 1-D transposed convolution as strings."""
    n = (len(inputs) - 1) * stride + len(filt)
    terms = [[] for _ in range(n)]
    for i, a in enumerate(inputs):
        for k, f in enumerate(filt):
            terms[i * stride + k].append(f"{a}{f}")
    return [" + ".join(t) for t in terms]


def fig_transposed_conv():
    """A stride-2 transposed convolution worked out in one dimension."""
    body = ""
    out_terms = transposed_1d(INPUTS, FILTER, STRIDE)
    cell = 58

    body += text(60, 40, "Input", size=13, anchor="start", weight="600")
    body += cells(60, 54, [INPUTS], cell, size=15, fill=lambda v: INDIGO, op=0.4)
    body += text(60 + cell, 54 + cell + 18, "2 values", size=11, op=0.7)

    body += text(228, 40, "Filter", size=13, anchor="start", weight="600")
    body += cells(228, 54, [FILTER], cell, size=15, fill=lambda v: TEAL, op=0.35)
    body += text(228 + 1.5 * cell, 54 + cell + 18, "learned, width 3", size=11, op=0.7)

    ox = 494
    body += text(ox, 40, "Output", size=13, anchor="start", weight="600")

    # One scaled copy of the filter per input, laid down `stride` cells apart.
    for i, a in enumerate(INPUTS):
        y = 54 + i * 64
        colour = INDIGO if i == 0 else GREEN
        for k, f in enumerate(FILTER):
            x = ox + (i * STRIDE + k) * cell
            body += rect(x, y, cell, cell, fill=colour, op=0.28, so=0.5)
            body += text(x + cell / 2, y + cell / 2 + 5, f"{a}{f}", size=14)
        body += text(
            ox - 14,
            y + cell / 2 + 5,
            f"{a} · filter",
            size=11.5,
            anchor="end",
            op=0.8,
        )

    # The summed output row.
    ysum = 206
    for j, term in enumerate(out_terms):
        x = ox + j * cell
        overlap = "+" in term
        body += rect(x, ysum, cell, cell, fill=PURPLE if overlap else "none", op=0.22, so=0.6, sw=1.2)
        body += text(x + cell / 2, ysum + cell / 2 + 5, term, size=12.5 if overlap else 14)
        last_row = 1 if j >= STRIDE else 0
        body += arrow(
            x + cell / 2, 54 + last_row * 64 + cell + 4, x + cell / 2, ysum - 4, sw=1.0, op=0.4
        )
    body += text(ox - 14, ysum + cell / 2 + 5, "sum", size=11.5, anchor="end", op=0.8)
    body += text(
        ox + 2.5 * cell,
        ysum + cell + 24,
        "5 values — the filter moves 2 output cells per input cell",
        size=11.5,
        op=0.72,
    )
    body += text(
        ox + 2.5 * cell,
        ysum + cell + 44,
        "only the shaded cell was reached by both copies",
        size=11,
        op=0.6,
    )

    return wrap(
        860,
        330,
        "transConv",
        "A stride-2 transposed convolution in one dimension",
        "An input of two values and a filter of three weights on the left. On "
        "the right, two scaled copies of the filter are laid out two cells "
        "apart, and beneath them the summed output of five terms, in which the "
        "middle term is the sum of contributions from both copies.",
        body,
    )


# ---------------------------------------------------------------------------
# 4. U-Net.
# ---------------------------------------------------------------------------


def fig_unet():
    """The U-Net encoder-decoder with its skip connections."""
    body = ""
    # (x, half-height, width, label) per level, left branch then right branch.
    levels = [
        (96, 46, 18, 70, "H × W"),
        (168, 34, 26, 154, "H/2"),
        (240, 24, 34, 226, "H/4"),
    ]
    bottom = (352, 17, 48, 300)

    def blk(x, hh, w, y, colour):
        """One feature map rectangle centred vertically on y."""
        return rect(x, y - hh, w, 2 * hh, fill=colour, op=0.55, sw=1.0, so=0.7)

    rights = []
    for lx, hh, w, y, label in levels:
        rx = 760 - (lx - 96) - w
        rights.append((rx, hh, w, y, label))

    body += arrow(40, 70, 90, 70, sw=1.3, op=0.7)
    body += text(40, 56, "image", size=11.5, anchor="start", op=0.8)

    prev = None
    for lx, hh, w, y, label in levels:
        body += blk(lx, hh, w, y, TEAL)
        body += text(lx + w / 2, y - hh - 9, label, size=11, op=0.8)
        if prev:
            body += arrow(prev[0] + prev[2], prev[3], lx, y, sw=1.2, op=0.5)
        prev = (lx, hh, w, y, label)

    bx, bhh, bw, by = bottom
    body += blk(bx, bhh, bw, by, PURPLE)
    body += text(bx + bw / 2, by + bhh + 18, "H/8", size=11, op=0.8)
    body += arrow(prev[0] + prev[2], prev[3], bx, by, sw=1.2, op=0.5)

    prev = (bx, bhh, bw, by)
    for rx, hh, w, y, label in reversed(rights):
        body += blk(rx, hh, w, y, GREEN)
        body += text(rx + w / 2, y - hh - 9, label, size=11, op=0.8)
        body += arrow(prev[0] + prev[2], prev[3], rx, y, sw=1.2, op=0.5)
        prev = (rx, hh, w, y, label)

    body += arrow(prev[0] + prev[2], prev[3], prev[0] + prev[2] + 50, prev[3], sw=1.3, op=0.7)
    body += text(prev[0] + prev[2] + 56, prev[3] + 4, "masks", size=11.5, anchor="start", op=0.8)

    # Skip connections: left branch to the mirrored right branch, same level.
    for (lx, hh, w, y, _), (rx, _, _, _, _) in zip(levels, rights):
        body += arrow(lx + w + 4, y, rx - 4, y, sw=1.4, op=0.85, stroke=PURPLE, dash="6 4")
        body += text(
            (lx + w + rx) / 2,
            y - 10,
            "copy + concatenate",
            size=11,
            fill=PURPLE,
        )

    body += text(150, 340, "downsampling: context grows, position blurs", size=11.5, op=0.75)
    body += text(620, 340, "upsampling: resolution restored", size=11.5, op=0.75)
    body += line(96, 322, 400, 322, sw=1.0, op=0.4, dash="4 3")
    body += line(452, 322, 778, 322, sw=1.0, op=0.4, dash="4 3")

    return wrap(
        860,
        366,
        "unet",
        "The U-Net architecture",
        "Feature maps descend to the left, shrinking in height and growing in "
        "width, reach a narrow bottleneck at the bottom, then ascend to the "
        "right in mirror image. At each of the three levels a dashed arrow "
        "runs horizontally from the left map to its counterpart on the right, "
        "labelled copy and concatenate.",
        body,
    )


# ---------------------------------------------------------------------------
# 5. Three generations of proposal-based detection.
# ---------------------------------------------------------------------------

COLS = [40, 132, 266, 400, 534, 668]
CW = [56, 118, 118, 118, 118, 150]


def fig_detector_family():
    """R-CNN, Fast R-CNN and Faster R-CNN as three pipelines."""
    body = ""
    rows = [
        (
            96,
            "R-CNN",
            "one backbone pass per region",
            [
                (0, "image", None, INDIGO, None),
                (1, "selective search", "CPU, not learned", TEAL, "5 3"),
                (2, "warp each crop", "~2000 regions", TEAL, None),
                (3, "backbone CNN", "× 2000", PURPLE, None),
                (4, "classify + regress", "per region", TEAL, None),
                (5, "class + box", None, GREEN, None),
            ],
        ),
        (
            250,
            "Fast R-CNN",
            "backbone runs once; crop its features instead",
            [
                (0, "image", None, INDIGO, None),
                (1, "backbone CNN", "× 1", TEAL, None),
                (3, "RoI pool", "crop features", TEAL, None),
                (4, "small head", "× N regions", TEAL, None),
                (5, "class + box", None, GREEN, None),
            ],
        ),
        (
            404,
            "Faster R-CNN",
            "proposals come from the same features",
            [
                (0, "image", None, INDIGO, None),
                (1, "backbone CNN", "× 1", TEAL, None),
                (2, "region proposal net", "learned", GREEN, None),
                (3, "RoI align", "crop features", TEAL, None),
                (4, "small head", "× N regions", TEAL, None),
                (5, "class + box", None, GREEN, None),
            ],
        ),
    ]

    for y, name, sub, stages in rows:
        body += text(40, y - 34, name, size=14, anchor="start", weight="600")
        body += text(40, y - 17, sub, size=11.5, anchor="start", op=0.7)
        placed = []
        for col, label, note, colour, dash in stages:
            x, w = COLS[col], CW[col]
            body += pill(x, y, w, 46, label, colour, sub=note, dash=dash)
            placed.append((x, w))
        for (x0, w0), (x1, _) in zip(placed, placed[1:]):
            body += arrow(x0 + w0 + 3, y + 23, x1 - 3, y + 23, sw=1.2, op=0.55)

    # Fast R-CNN still takes its proposals from outside the network. The box
    # sits left of the RoI pool it feeds, so the arrow runs forwards.
    y = 250
    px, pw = COLS[2], CW[2]
    body += pill(px, y - 66, pw, 40, "selective search", PURPLE, sub="CPU, not learned", dash="5 3")
    body += arrow(px + pw / 2, y - 22, COLS[3] + 22, y - 3, sw=1.2, op=0.55)

    body += text(
        250,
        488,
        "dashed — a hand-designed stage outside the network",
        size=11.5,
        op=0.68,
    )
    body += text(662, 488, "purple — the step that dominates the runtime", size=11.5, op=0.68)

    return wrap(
        900,
        510,
        "detFamily",
        "R-CNN, Fast R-CNN and Faster R-CNN",
        "Three horizontal pipelines. The first runs a full backbone once per "
        "region after cropping the image. The second runs the backbone once on "
        "the whole image and crops its features, with proposals still supplied "
        "by an external dashed box. The third replaces that box with a region "
        "proposal network reading the same features.",
        body,
    )


# ---------------------------------------------------------------------------
# 6. RoI pool against RoI align.
# ---------------------------------------------------------------------------

BOX = (2.4, 1.6, 6.7, 4.8)  # the projected proposal, in feature-grid coordinates


def fig_roi_align():
    """Snapping to grid cells against bilinear sampling."""
    body = ""
    cell = 30
    x0f, y0f, x1f, y1f = BOX

    for panel, (px, title, note) in enumerate(
        [
            (56, "RoI pool", "the box is snapped outward to whole cells"),
            (500, "RoI align", "the box is kept; samples are interpolated"),
        ]
    ):
        gy = 92
        body += text(px + 120, 44, title, size=14, weight="600")
        body += grid(px, gy, 8, 6, cell)

        def fx(u):
            """Feature-grid column to pixel x."""
            return px + u * cell

        def fy(v):
            """Feature-grid row to pixel y."""
            return gy + v * cell

        # The projected proposal, at its real-valued coordinates.
        body += rect(
            fx(x0f),
            fy(y0f),
            (x1f - x0f) * cell,
            (y1f - y0f) * cell,
            fill="none",
            stroke=GREEN,
            sw=1.8,
            so=1.0,
        ).replace("/>", ' stroke-dasharray="5 3"/>')

        if panel == 0:
            sx0, sy0, sx1, sy1 = 2, 1, 7, 5
            body += rect(
                fx(sx0),
                fy(sy0),
                (sx1 - sx0) * cell,
                (sy1 - sy0) * cell,
                fill=PURPLE,
                op=0.14,
                stroke=PURPLE,
                sw=2.0,
                so=1.0,
            )
            # 2 x 2 subdivision of the snapped region.
            body += line(fx((sx0 + sx1) / 2), fy(sy0), fx((sx0 + sx1) / 2), fy(sy1), sw=1.4, op=0.7, stroke=PURPLE)
            body += line(fx(sx0), fy((sy0 + sy1) / 2), fx(sx1), fy((sy0 + sy1) / 2), sw=1.4, op=0.7, stroke=PURPLE)
            body += arrow(fx(sx0) + 18, gy - 22, fx(sx0) + 3, fy(sy0) - 3, sw=1.2, op=0.85, stroke=PURPLE)
            body += text(fx(sx0) + 22, gy - 26, "snapped outward", size=11, fill=PURPLE, anchor="start")
        else:
            # Four sample points per subregion of a 2 x 2 division of the box.
            for sub_i in range(2):
                for sub_j in range(2):
                    u0 = x0f + sub_i * (x1f - x0f) / 2
                    v0 = y0f + sub_j * (y1f - y0f) / 2
                    for a in (0.25, 0.75):
                        for b in (0.25, 0.75):
                            u = u0 + a * (x1f - x0f) / 2
                            v = v0 + b * (y1f - y0f) / 2
                            body += circle(fx(u), fy(v), 2.6, fill=PURPLE, stroke="none", op=0.95)
            body += line(
                fx((x0f + x1f) / 2), fy(y0f), fx((x0f + x1f) / 2), fy(y1f), sw=1.2, op=0.5, stroke=GREEN, dash="3 3"
            )
            body += line(
                fx(x0f), fy((y0f + y1f) / 2), fx(x1f), fy((y0f + y1f) / 2), sw=1.2, op=0.5, stroke=GREEN, dash="3 3"
            )

            # One sample and the four grid values it interpolates between.
            su, sv = x0f + 0.25 * (x1f - x0f) / 2, y0f + 0.25 * (y1f - y0f) / 2
            for gu in (int(su), int(su) + 1):
                for gv in (int(sv), int(sv) + 1):
                    body += circle(fx(gu), fy(gv), 3.6, fill="none", stroke=INDIGO, sw=1.8)
                    body += line(fx(gu), fy(gv), fx(su), fy(sv), sw=1.1, op=0.75, stroke=INDIGO)
            body += text(fx(su) - 8, fy(sv) - 10, "bilinear", size=11, fill=INDIGO, anchor="end")

        body += text(px + 120, gy + 6 * cell + 28, note, size=11.5, op=0.75)

    body += text(176, 328, "features cropped from the wrong place", size=11, fill=PURPLE)
    body += text(620, 328, "no displacement; differentiable in the box", size=11, fill=TEAL)

    return wrap(
        900,
        352,
        "roiAlign",
        "RoI pool against RoI align",
        "Two panels, each showing an eight by six feature grid with a dashed "
        "green rectangle at real-valued coordinates. In the left panel a solid "
        "purple rectangle snapped outward to whole cells sits offset from it "
        "and is divided into four subregions. In the right panel the dashed "
        "rectangle is kept and dotted sample points fill it, with one sample "
        "joined to the four surrounding grid intersections.",
        body,
    )


# ---------------------------------------------------------------------------
# 7. The YOLO output tensor.
# ---------------------------------------------------------------------------

S, B, C_SHOWN = 7, 2, 6


def fig_yolo_grid():
    """The S x S x (5B + C) prediction tensor."""
    body = ""
    cell = 24
    gx, gy = 40, 96
    body += text(gx + S * cell / 2, 52, "S × S grid", size=13.5, weight="600")
    body += grid(gx, gy, S, S, cell, so=0.4)
    hi_r, hi_c = 3, 3
    body += rect(gx + hi_c * cell, gy + hi_r * cell, cell, cell, fill=PURPLE, op=0.4, stroke=PURPLE, sw=1.8)
    # Two base boxes anchored on the highlighted cell.
    ccx, ccy = gx + (hi_c + 0.5) * cell, gy + (hi_r + 0.5) * cell
    body += rect(ccx - 30, ccy - 20, 60, 40, fill="none", stroke=GREEN, sw=1.4, so=0.9)
    body += rect(ccx - 16, ccy - 34, 32, 68, fill="none", stroke=INDIGO, sw=1.4, so=0.9)
    body += text(gx + S * cell / 2, gy + S * cell + 20, "B base boxes per cell", size=11, op=0.75)

    # The whole prediction, as one tensor.
    tx, ty = 268, 118
    body += cuboid(tx, ty, 84, 84, 30, -26, TEAL, op=0.42)
    body += text(tx + 42, ty + 46, "S × S", size=12)
    body += text(tx + 57, ty - 40, "5B + C", size=11.5, op=0.85)
    body += text(tx + 57, 52, "one output tensor", size=13.5, weight="600")
    body += arrow(gx + S * cell + 8, gy + S * cell / 2, tx - 6, ty + 42, sw=1.3, op=0.6)

    # One cell's slice through that tensor, expanded.
    vx, vy = 452, 150
    vw = 24
    groups = [
        ([("d", INDIGO), ("d", INDIGO), ("d", INDIGO), ("d", INDIGO), ("p", PURPLE)], "box 1"),
        ([("d", INDIGO), ("d", INDIGO), ("d", INDIGO), ("d", INDIGO), ("p", PURPLE)], "box 2"),
        ([("c", GREEN)] * C_SHOWN, "class scores"),
    ]
    x = vx
    body += text(vx + (5 * B + C_SHOWN) * vw / 2, 52, "the slice at one cell", size=13.5, weight="600")
    for entries, label in groups:
        x0 = x
        for kind, colour in entries:
            body += rect(x, vy, vw, 34, fill=colour, op=0.35, so=0.6)
            body += text(x + vw / 2, vy + 22, {"d": "δ", "p": "c", "c": "s"}[kind], size=12)
            x += vw
        body += line(x0, vy + 44, x - 1, vy + 44, sw=1.2, op=0.6)
        body += text((x0 + x) / 2, vy + 60, label, size=11, op=0.8)
        x += 10
    body += text(vx, vy - 14, "4 offsets + 1 confidence per box, then C class scores", size=11, anchor="start", op=0.72)
    body += arrow(tx + 116, ty + 42, vx - 8, vy + 17, sw=1.3, op=0.6)

    body += text(
        460,
        302,
        "one forward pass; thresholding and non-maximum suppression turn this fixed tensor into a variable-length list",
        size=11.5,
        op=0.7,
    )

    return wrap(
        900,
        320,
        "yoloGrid",
        "The YOLO output tensor",
        "On the left a seven by seven grid over an image, with one cell "
        "highlighted and two base boxes of different shapes drawn on it. In "
        "the middle the whole prediction drawn as a cuboid of size S by S by "
        "five B plus C. On the right the slice belonging to one cell, expanded "
        "into two groups of five numbers for the boxes and a group of class "
        "scores.",
        body,
    )


# ---------------------------------------------------------------------------
# 8. DETR.
# ---------------------------------------------------------------------------


def fig_detr():
    """Set prediction with object queries and bipartite matching."""
    body = ""
    rail = 128
    stages = [
        (40, 62, "image", INDIGO, None),
        (128, 96, "backbone", TEAL, "CNN"),
        (256, 104, "encoder", TEAL, "self-attention"),
        (412, 104, "decoder", GREEN, "cross-attention"),
    ]
    for x, w, label, colour, sub in stages:
        body += pill(x, rail, w, 50, label, colour, sub=sub)
    for (x0, w0, *_), (x1, *_) in zip(stages, stages[1:]):
        body += arrow(x0 + w0 + 4, rail + 25, x1 - 4, rail + 25, sw=1.2, op=0.55)
    body += text(184, rail - 14, "+ positional encoding", size=11, op=0.7)

    # Object queries feed the decoder, so they sit to its left.
    qx, qy = 288, 236
    body += text(qx + 52, qy - 10, "N object queries", size=11.5, op=0.85)
    for i in range(4):
        body += rect(qx + i * 26, qy, 20, 30, fill=INDIGO, op=0.35, stroke=INDIGO, sw=1.0, so=0.9)
    body += text(qx + 52, qy + 48, "learned parameters, not image features", size=11, op=0.66)
    body += arrow(qx + 104, qy + 6, 424, rail + 54, sw=1.3, op=0.8, stroke=INDIGO)

    # N predictions, then the one-to-one match against ground truth.
    preds = [("cat", GREEN), ("dog", GREEN), ("∅", PURPLE), ("∅", PURPLE)]
    gts = [("cat", GREEN), ("dog", GREEN)]
    px, py = 566, 62
    body += text(px + 58, 44, "N predictions", size=13, weight="600")
    for i, (label, colour) in enumerate(preds):
        y = py + i * 52
        body += pill(px, y, 116, 38, label, colour, sub="class + box" if colour == GREEN else "no object")
    body += arrow(stages[-1][0] + stages[-1][1] + 4, rail + 25, px - 6, rail + 25, sw=1.2, op=0.55)

    gx, gy = 768, 62
    body += text(gx + 46, 44, "ground truth", size=13, weight="600")
    for i, (label, colour) in enumerate(gts):
        body += pill(gx, gy + i * 52, 92, 38, label, colour, op=0.1)
    body += pill(gx, gy + 2.5 * 52, 92, 38, "∅", PURPLE, sub="unmatched", op=0.1)

    # The two matched pairs, then the slots that fall through to no-object.
    for i in range(2):
        body += arrow(px + 118, py + i * 52 + 19, gx - 4, gy + i * 52 + 19, sw=1.5, op=0.9, stroke=PURPLE)
    for i in (2, 3):
        body += arrow(
            px + 118,
            py + i * 52 + 19,
            gx - 4,
            gy + 2.5 * 52 + 19,
            sw=1.1,
            op=0.5,
            dash="4 3",
        )

    body += text(
        450,
        318,
        "the one-to-one match is what forbids duplicates — a second box on the same cat can only be assigned ∅",
        size=11.5,
        op=0.72,
    )

    return wrap(
        900,
        340,
        "detr",
        "DETR: detection as set prediction",
        "A left-to-right rail runs from the image through a convolutional "
        "backbone, a transformer encoder and a transformer decoder. A row of "
        "learned object queries enters the decoder from below and to the left. "
        "On the right, four predictions are matched one-to-one against two "
        "ground-truth objects, the two unmatched predictions being assigned "
        "the no-object class.",
        body,
    )


# ---------------------------------------------------------------------------
# 9. CAM against Grad-CAM.
# ---------------------------------------------------------------------------

CAM_W = [0.9, 0.2, 0.7, 0.1]
HEAT = [
    [0.05, 0.10, 0.22, 0.18],
    [0.12, 0.62, 0.88, 0.34],
    [0.18, 0.74, 0.95, 0.40],
    [0.08, 0.20, 0.36, 0.16],
]


def _viridis(t):
    """Approximate viridis at t in [0, 1] by interpolating the five samples."""
    stops = [
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            r, g, b = (int(a + f * (b_ - a)) for a, b_ in zip(c0, c1))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#fde725"


def _panel(x0, title, weights, weight_label, note, gradient):
    """One of the two panels: K channel slices, weights, sum, heatmap."""
    body = text(x0 + 176, 42, title, size=14, weight="600")
    slice_w, gap = 44, 54
    sy = 108
    for k, w in enumerate(weights):
        x = x0 + 8 + k * gap
        body += cuboid(x, sy, slice_w, slice_w, 9, -8, TEAL, op=0.3)
        body += text(x + slice_w / 2, sy - 18, f"{weight_label}{k + 1}", size=11.5, op=0.9)
        body += text(x + slice_w / 2, sy + slice_w / 2 + 5, f"{w:.1f}", size=12, font=MONO)
    body += text(x0 + 8 + 1.5 * gap + slice_w / 2, sy + slice_w + 22, "K channel slices", size=11, op=0.7)

    sumx = x0 + 8 + 4 * gap + 6
    body += arrow(sumx - 14, sy + slice_w / 2, sumx - 2, sy + slice_w / 2, sw=1.2, op=0.55)
    body += circle(sumx + 16, sy + slice_w / 2, 14, fill="none", sw=1.3, so=0.8)
    body += text(sumx + 16, sy + slice_w / 2 + 6, "Σ", size=16)

    hx = sumx + 46
    body += arrow(sumx + 32, sy + slice_w / 2, hx - 4, sy + slice_w / 2, sw=1.2, op=0.55)
    hcell = 16
    for r, row in enumerate(HEAT):
        for c, v in enumerate(row):
            body += rect(
                hx + c * hcell,
                sy - 10 + r * hcell,
                hcell,
                hcell,
                fill=_viridis(v),
                op=0.9,
                stroke="none",
            )
    body += rect(hx, sy - 10, 4 * hcell, 4 * hcell, fill="none", sw=1.0, so=0.5)
    body += text(hx + 2 * hcell, sy - 10 + 4 * hcell + 18, "heatmap", size=11, op=0.75)

    if gradient:
        gy = sy + slice_w + 58
        body += text(hx + 2 * hcell, gy + 5, "class score", size=11.5, fill=PURPLE)
        body += arrow(hx - 10, gy, x0 + 8 + 2 * gap, gy, sw=1.3, op=0.85, stroke=PURPLE, dash="5 3")
        body += text(x0 + 8 + 0.9 * gap, gy - 10, "gradient, backwards", size=11, fill=PURPLE)

    body += text(x0 + 176, 268, note, size=11.5, op=0.72)
    return body


def fig_grad_cam():
    """CAM and Grad-CAM differ only in where the channel weights come from."""
    body = _panel(
        30,
        "CAM",
        CAM_W,
        "w",
        "weights read off the final linear layer — needs global average pooling",
        gradient=False,
    )
    body += _panel(
        470,
        "Grad-CAM",
        CAM_W,
        "α",
        "weights are spatially averaged gradients — any layer, any head",
        gradient=True,
    )
    body += line(452, 40, 452, 250, sw=1.0, op=0.25, dash="4 4")
    body += text(
        450,
        296,
        "on an architecture CAM accepts, the gradients of the right panel evaluate to the weights of the left",
        size=11.5,
        op=0.68,
    )
    return wrap(
        900,
        320,
        "gradCam",
        "CAM and Grad-CAM",
        "Two panels of identical structure. In each, four channel slices "
        "carrying weights are summed into a small viridis heatmap. The left "
        "panel labels the weights as entries of the final linear layer; the "
        "right labels them alpha and adds a dashed arrow running backwards "
        "from the class score to the slices, marked as the gradient of the "
        "score with respect to the activations.",
        body,
    )


FIGURES = {
    "diagram_seg_shapes.svg": fig_seg_shapes,
    "diagram_unpooling.svg": fig_unpooling,
    "diagram_transposed_conv.svg": fig_transposed_conv,
    "diagram_unet.svg": fig_unet,
    "diagram_detector_family.svg": fig_detector_family,
    "diagram_roi_align.svg": fig_roi_align,
    "diagram_yolo_grid.svg": fig_yolo_grid,
    "diagram_detr.svg": fig_detr,
    "diagram_grad_cam.svg": fig_grad_cam,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print(f"wrote {OUT / name}")

    # The chapter quotes these; print them so the prose can be checked.
    terms = transposed_1d(INPUTS, FILTER, STRIDE)
    print(f"transposed conv, stride {STRIDE}: ({', '.join(terms)})")
    print(f"  output length {len(terms)} from {len(INPUTS)} inputs and a width-{len(FILTER)} filter")
    print(f"  positions receiving more than one term: {[i for i, t in enumerate(terms) if '+' in t]}")
    print(f"YOLO output tensor: {S} x {S} x (5*{B} + C) = {S}x{S}x{5 * B}+C")
