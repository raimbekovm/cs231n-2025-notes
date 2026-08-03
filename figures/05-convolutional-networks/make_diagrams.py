"""Generate the seven SVG diagrams for the lecture 5 chapter.

Every number that appears in these figures is the number the arithmetic of the
chapter produces: the receptive-field fan is drawn from the actual dependency
sets, the pooling grid is the worked example of @eq-maxpool, and the Gabor
patches are evaluated rather than sketched. Conventions follow
figures/04-neural-networks-backpropagation: currentColor for text and rules,
viridis for data, title/desc with namespaced ids, no background rect (the theme
supplies a light plate in both schemes).

Run from the repository root; rewrites all seven files.
"""

import math
import pathlib

OUT = pathlib.Path("figures/05-convolutional-networks")

PURPLE = "#440154"
INDIGO = "#3b528b"
TEAL = "#21918c"
GREEN = "#5ec962"
YELLOW = "#fde725"
INK = "#1b1b1b"
FONT = "Source Serif 4, Georgia, serif"
MONO = "JetBrains Mono, SFMono-Regular, Menlo, monospace"

# The value ramp used by the filter grid. These are the viridis anchors the
# other chapters draw from, interpolated linearly between.
VIRIDIS = [
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
]


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


def text(x, y, s, size=13, fill="currentColor", anchor="middle", font=None, style=""):
    """One text element. Every caller passes explicit coordinates."""
    f = f' font-family="{font}"' if font else ""
    st = f' font-style="{style}"' if style else ""
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{f}{st}>{s}</text>\n'
    )


def rect(x, y, w, h, fill="none", stroke="currentColor", sw=1.0, op=1.0, so=1.0):
    """One rectangle."""
    return (
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"'
        f' fill="{fill}" fill-opacity="{op}" stroke="{stroke}"'
        f' stroke-width="{sw}" stroke-opacity="{so}"/>\n'
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


def arrow(x0, y0, x1, y1, sw=1.4, op=0.75, head=6.0):
    """A plain line with a solid triangular head; no markers, no defs."""
    ang = math.atan2(y1 - y0, x1 - x0)
    bx, by = x1 - head * math.cos(ang), y1 - head * math.sin(ang)
    out = line(x0, y0, bx, by, sw=sw, op=op)
    tips = [
        (x1, y1),
        (bx - head * 0.45 * math.sin(ang), by + head * 0.45 * math.cos(ang)),
        (bx + head * 0.45 * math.sin(ang), by - head * 0.45 * math.cos(ang)),
    ]
    return out + poly(tips, fill="currentColor", stroke="none", op=op)


def grid(x, y, cell, n, m=None, sw=0.7, op=0.5):
    """An n-by-m lattice of cells with its top-left corner at (x, y)."""
    m = n if m is None else m
    out = ""
    for i in range(n + 1):
        out += line(x + i * cell, y, x + i * cell, y + m * cell, sw=sw, op=op)
    for j in range(m + 1):
        out += line(x, y + j * cell, x + n * cell, y + j * cell, sw=sw, op=op)
    return out


def cuboid(x, y, w, h, dx, dy, face, top=None, side=None, op=0.85, sw=1.0):
    """A box in oblique projection: front face at (x, y), back offset by (dx, dy)."""
    top = top or face
    side = side or face
    out = poly(
        [(x, y), (x + dx, y + dy), (x + w + dx, y + dy), (x + w, y)],
        fill=top,
        op=op * 0.55,
        sw=sw,
        so=0.55,
    )
    out += poly(
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        fill=side,
        op=op * 0.75,
        sw=sw,
        so=0.55,
    )
    out += rect(x, y, w, h, fill=face, op=op, sw=sw, so=0.7)
    return out


# ---------------------------------------------------------------------------
# 1. Flattening destroys adjacency.
# ---------------------------------------------------------------------------


def fig_flatten():
    """Flattening an image destroys pixel adjacency."""
    w, h = 760, 260
    cell = 34
    gx, gy = 40, 62
    # The marked pixel is row 1, column 1 (zero-indexed), so its four
    # neighbours are at flat indices 1, 4, 6 and 9 against its own 5.
    centre = (1, 1)
    nbrs = [(0, 1), (1, 0), (1, 2), (2, 1)]

    body = text(gx + 2 * cell, 36, "the image", size=14)
    for j in range(4):
        for i in range(4):
            fill = "none"
            if (j, i) == centre:
                fill = PURPLE
            elif (j, i) in nbrs:
                fill = TEAL
            body += rect(
                gx + i * cell,
                gy + j * cell,
                cell,
                cell,
                fill=fill,
                op=0.0 if fill == "none" else 0.8,
                sw=0.9,
                so=0.55,
            )
    body += text(gx + 2 * cell, gy + 4 * cell + 26, "a pixel and its four neighbours", size=12.5)

    # The flattened strip: sixteen cells in row-major order.
    sx, sy = 300, gy + cell
    scell = 27
    body += text(sx + 8 * scell, 36, "the same pixels after flattening", size=14)
    for k in range(16):
        j, i = divmod(k, 4)
        fill = "none"
        if (j, i) == centre:
            fill = PURPLE
        elif (j, i) in nbrs:
            fill = TEAL
        body += rect(
            sx + k * scell,
            sy,
            scell,
            scell,
            fill=fill,
            op=0.0 if fill == "none" else 0.8,
            sw=0.9,
            so=0.55,
        )
        body += text(sx + k * scell + scell / 2, sy + scell + 15, str(k), size=10.5, fill=INK, font=MONO)

    # Distance braces: the two neighbours that survive, and the two that do not.
    base = sy + scell + 30
    for a, b, lab in ((4, 5, "1"), (5, 6, "1"), (1, 5, "4"), (5, 9, "4")):
        y = base + (10 if lab == "1" else 30)
        x0 = sx + min(a, b) * scell + scell / 2
        x1 = sx + max(a, b) * scell + scell / 2
        body += line(x0, y, x1, y, sw=1.0, op=0.5)
        body += line(x0, y - 3, x0, y + 3, sw=1.0, op=0.5)
        body += line(x1, y - 3, x1, y + 3, sw=1.0, op=0.5)
        body += text((x0 + x1) / 2, y - 5, lab, size=10.5, fill=INK, font=MONO)
    body += text(
        sx + 8 * scell,
        base + 58,
        "two neighbours stay adjacent; two land a whole row apart",
        size=12.5,
        fill=INK,
        style="italic",
    )
    body += arrow(gx + 4 * cell + 14, sy + scell / 2, sx - 14, sy + scell / 2)

    return wrap(
        w,
        h,
        "flattenFig",
        "Flattening an image destroys pixel adjacency",
        "A four-by-four image with one pixel and its four neighbours marked, "
        "beside the same sixteen pixels laid out as a vector in row-major "
        "order. Two of the four neighbours remain adjacent in the vector; the "
        "other two are separated by a full row.",
        body,
    )


# ---------------------------------------------------------------------------
# 2. The convolution layer: filter, activation map, output volume.
# ---------------------------------------------------------------------------


def fig_conv_layer():
    """One convolutional layer, from filter placement to output volume."""
    w, h = 900, 344
    dx, dy = 22, -16

    body = ""

    # --- Stage 1: the input volume with one filter placement highlighted.
    ix, iy, iw = 46, 108, 150
    body += cuboid(ix, iy, iw, iw, dx, dy, INDIGO, op=0.16)
    body += grid(ix, iy, iw / 8, 8, sw=0.5, op=0.28)
    fw = iw * 5 / 32
    fx, fy = ix + iw * 9 / 32, iy + iw * 7 / 32
    body += cuboid(fx, fy, fw, fw, dx, dy, TEAL, op=0.9)
    body += text(ix + iw / 2, iy - 44, "input", size=14)
    body += text(ix + iw / 2, iy + iw + 26, "3 × 32 × 32", size=12.5, font=MONO, fill=INK)
    body += text(
        ix + iw / 2,
        iy + iw + 46,
        "the filter spans all three channels",
        size=12,
        fill=INK,
        style="italic",
    )

    # --- Stage 2: one activation map, one cell lit.
    ax, ay, aw = 356, 118, 132
    body += rect(ax, ay, aw, aw, fill=INDIGO, op=0.10, sw=1.0, so=0.6)
    body += grid(ax, ay, aw / 7, 7, sw=0.5, op=0.28)
    cw = aw / 7
    body += rect(ax + 2 * cw, ay + 1 * cw, cw, cw, fill=TEAL, op=0.9, sw=0.9, so=0.6)
    body += text(ax + aw / 2, ay - 34, "activation map", size=14)
    body += text(ax + aw / 2, ay + aw + 26, "1 × 28 × 28", size=12.5, font=MONO, fill=INK)

    # The wire from the highlighted filter slab to the cell it produces.
    wire_x0 = ix + iw + dx + 10
    body += arrow(wire_x0, fy + fw / 2, ax + 2 * cw - 8, ay + 1.5 * cw, sw=1.2, op=0.7)
    label_x = (wire_x0 + ax + 2 * cw) / 2
    body += text(
        label_x,
        fy + fw / 2 - 14,
        "75-dim dot product + bias",
        size=11.5,
        fill=INK,
        style="italic",
    )
    body += text(
        label_x,
        fy + fw / 2 + 22,
        "slide to all 28 × 28 placements",
        size=11.5,
        fill=INK,
        style="italic",
    )

    # --- Stage 3: six maps stacked into the output volume.
    ox, oy, ow = 636, 132, 120
    for k in range(5, -1, -1):
        col = [PURPLE, INDIGO, TEAL, TEAL, GREEN, YELLOW][k]
        body += rect(
            ox + k * 13,
            oy - k * 10,
            ow,
            ow,
            fill=col,
            op=0.30 if k else 0.55,
            sw=0.9,
            so=0.55,
        )
    body += text(ox + ow / 2 + 32, oy - 84, "output volume", size=14)
    body += text(ox + ow / 2 + 32, oy + ow + 26, "6 × 28 × 28", size=12.5, font=MONO, fill=INK)
    body += text(
        ox + ow / 2 + 32,
        oy + ow + 46,
        "one map per filter, stacked",
        size=12,
        fill=INK,
        style="italic",
    )
    body += arrow(ax + aw + 10, ay + aw / 2, ox - 12, oy + ow / 2, sw=1.2, op=0.7)
    body += text(
        (ax + aw + ox) / 2,
        ay + aw / 2 - 12,
        "× 6 filters",
        size=11.5,
        fill=INK,
        style="italic",
    )

    return wrap(
        w,
        h,
        "convFig",
        "One convolutional layer, from filter to output volume",
        "Left: a three-channel input volume with a five-by-five filter, "
        "spanning the full depth, placed at one location. Centre: the single "
        "number that placement produces, in a twenty-eight by twenty-eight "
        "activation map filled by sliding the filter everywhere. Right: six "
        "such maps stacked into a six-channel output volume.",
        body,
    )


# ---------------------------------------------------------------------------
# 3. Gabor and centre-surround filters, evaluated.
# ---------------------------------------------------------------------------


def ramp(t):
    """Viridis at t in [0, 1], by linear interpolation between the anchors."""
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(VIRIDIS, VIRIDIS[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            r, g, b = (round(a + f * (bb - a)) for a, bb in zip(c0, c1))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#fde725"


def gabor(i, j, theta, lam, phase, sigma=1.15, gamma=0.72):
    """One sample of an oriented sinusoid under a Gaussian envelope."""
    x, y = (i - 5) * 0.5, (j - 5) * 0.5
    xr = x * math.cos(theta) + y * math.sin(theta)
    yr = -x * math.sin(theta) + y * math.cos(theta)
    env = math.exp(-(xr * xr + gamma * gamma * yr * yr) / (2 * sigma * sigma))
    return env * math.cos(2 * math.pi * xr / lam + phase)


def blob(i, j, sigma, sign, ox=0.0, oy=0.0):
    """Difference of Gaussians: a centre-surround patch."""
    x, y = (i - 5) * 0.5 - ox, (j - 5) * 0.5 - oy
    r2 = x * x + y * y
    inner = math.exp(-r2 / (2 * sigma * sigma))
    outer = math.exp(-r2 / (2 * (sigma * 1.9) ** 2))
    return sign * (inner - 0.65 * outer)


def fig_gabor():
    """Sixty-four computed 11x11 filters: oriented edges and centre-surround blobs."""
    cell, pad, n = 4, 6, 11
    tile = n * cell + pad
    w = 8 * tile - pad + 2
    h = 8 * tile - pad + 30
    body = ""

    # Six rows of oriented edges: three spatial frequencies, two phases each.
    rows = [(lam, ph) for lam in (1.3, 1.9, 2.7) for ph in (0.0, math.pi / 2)]
    for r, (lam, ph) in enumerate(rows):
        for c in range(8):
            theta = c * math.pi / 8
            vals = [[gabor(i, j, theta, lam, ph) for i in range(n)] for j in range(n)]
            body += patch(vals, 1 + c * tile, 1 + r * tile, cell, n)

    # Two rows of centre-surround blobs: four scales, both polarities, then the
    # same off-centre so the last row is not a repeat of the seventh.
    for r, off in ((6, 0.0), (7, 0.55)):
        for c in range(8):
            sigma = (0.65, 0.95, 1.25, 1.6)[c % 4]
            sign = 1.0 if c < 4 else -1.0
            vals = [[blob(i, j, sigma, sign, off, -off) for i in range(n)] for j in range(n)]
            body += patch(vals, 1 + c * tile, 1 + r * tile, cell, n)

    body += text(
        w / 2,
        h - 8,
        "sixty-four 11 × 11 templates, on the same value ramp",
        size=12,
        fill=INK,
        style="italic",
    )
    return wrap(
        w,
        h,
        "gaborFig",
        "Gabor and centre-surround filters at eleven by eleven pixels",
        "An eight-by-eight grid of computed filters. The first six rows are "
        "Gabor functions — an oriented sinusoid under a Gaussian envelope — at "
        "eight orientations, three spatial frequencies and two phases. The last "
        "two rows are centre-surround difference-of-Gaussian blobs at four "
        "scales and both polarities. These are the analytic family that trained "
        "first layers approximate, not filters from a trained network.",
        body,
    )


def patch(vals, x0, y0, cell, n):
    """One filter rendered as an n-by-n block of cells, normalised to its peak."""
    peak = max(abs(v) for row in vals for v in row) or 1.0
    out = ""
    for j, row in enumerate(vals):
        for i, v in enumerate(row):
            col = ramp((v / peak + 1.0) / 2.0)
            out += (
                f'  <rect x="{x0 + i * cell}" y="{y0 + j * cell}" width="{cell}"'
                f' height="{cell}" fill="{col}" stroke="none"/>\n'
            )
    out += rect(x0, y0, n * cell, n * cell, sw=0.6, so=0.35)
    return out


# ---------------------------------------------------------------------------
# 4. Padding and stride.
# ---------------------------------------------------------------------------


def conv_panel(x, y, cell, size, pad, stride, k=3):
    """One 7x7 input, drawn with its padding ring and its filter placements."""
    total = size + 2 * pad
    ox = x - pad * cell
    oy = y - pad * cell
    body = ""
    if pad:
        for j in range(total):
            for i in range(total):
                if 0 < i <= size and 0 < j <= size:
                    continue
                body += rect(ox + i * cell, oy + j * cell, cell, cell, fill=INK, op=0.06, sw=0.5, so=0.3)
    body += grid(x, y, cell, size, sw=0.8, op=0.55)
    if pad:
        body += grid(ox, oy, cell, total, sw=0.5, op=0.3)

    # Every valid placement, with the first drawn solid and the rest as corners.
    placements = list(range(0, total - k + 1, stride))
    for n, i in enumerate(placements):
        for m, j in enumerate(placements):
            solid = (n, m) == (0, 0)
            body += rect(
                ox + i * cell,
                oy + j * cell,
                k * cell,
                k * cell,
                fill=TEAL if solid else "none",
                op=0.35 if solid else 0.0,
                stroke=TEAL,
                sw=1.6 if solid else 0.9,
                so=1.0 if solid else 0.45,
            )
    out_n = len(placements)
    return body, out_n


def fig_padding_stride():
    """How padding and stride change the number of filter placements."""
    cell, size = 20, 7
    w, h = 760, 322
    cols = [
        (72, 0, 1, "stride 1, no padding", "7 − 3 + 1 = 5"),
        (300, 1, 1, "stride 1, padding 1", "7 − 3 + 1 + 2 = 7"),
        (548, 0, 2, "stride 2, no padding", "(7 − 3)/2 + 1 = 3"),
    ]
    body = ""
    for x, pad, stride, label, arith in cols:
        panel, out_n = conv_panel(x, 88, cell, size, pad, stride)
        body += panel
        body += text(x + size * cell / 2, 54, label, size=13)
        body += text(x + size * cell / 2, 274, arith, size=12.5, font=MONO, fill=INK)
        body += text(
            x + size * cell / 2,
            296,
            f"output {out_n} × {out_n}",
            size=12.5,
            fill=INK,
            style="italic",
        )
    body += text(
        w / 2,
        26,
        "a 3 × 3 filter on a 7 × 7 input: every valid placement outlined",
        size=14,
    )
    return wrap(
        w,
        h,
        "padFig",
        "Padding and stride change the number of filter placements",
        "Three copies of a seven-by-seven input with a three-by-three filter. "
        "Without padding at stride one there are five placements per axis; one "
        "ring of zero padding restores the lost border placements and gives "
        "seven; stride two on the unpadded input takes every other placement "
        "and gives three.",
        body,
    )


# ---------------------------------------------------------------------------
# 5. Receptive field growth over three layers.
# ---------------------------------------------------------------------------


def fig_receptive_field():
    """Receptive field growth across three stacked 3x3 convolutions."""
    w, h = 720, 404
    cell, gap = 34, 78
    layers = [7, 5, 3, 1]
    names = ["input", "layer 1", "layer 2", "layer 3"]
    top = 56
    xs = []
    for n in layers:
        xs.append([(w - n * cell) / 2 + i * cell for i in range(n)])

    body = ""
    # Dependency wires first, so the cells sit on top of them.
    for li in range(len(layers) - 1):
        y0 = top + li * gap + cell
        y1 = top + (li + 1) * gap
        for m in range(layers[li + 1]):
            for d in range(3):
                body += line(
                    xs[li][m + d] + cell / 2,
                    y0,
                    xs[li + 1][m] + cell / 2,
                    y1,
                    stroke=TEAL,
                    sw=1.0,
                    op=0.5,
                )
    for li, n in enumerate(layers):
        y = top + li * gap
        for i in range(n):
            fill = PURPLE if li == len(layers) - 1 else TEAL
            op = 0.8 if li == len(layers) - 1 else 0.20 + 0.10 * li
            body += rect(xs[li][i], y, cell, cell, fill=fill, op=op, sw=0.9, so=0.6)
        body += text(xs[li][0] - 22, y + cell / 2 + 4, names[li], size=12.5, anchor="end")
        body += text(
            xs[li][-1] + cell + 22,
            y + cell / 2 + 4,
            f"{n} wide",
            size=12,
            anchor="start",
            fill=INK,
            style="italic",
        )

    y_in = top - 16
    body += line(xs[0][0], y_in, xs[0][-1] + cell, y_in, stroke=PURPLE, sw=1.4, op=0.8)
    body += line(xs[0][0], y_in - 5, xs[0][0], y_in + 5, stroke=PURPLE, sw=1.4, op=0.8)
    body += line(xs[0][-1] + cell, y_in - 5, xs[0][-1] + cell, y_in + 5, stroke=PURPLE, sw=1.4, op=0.8)
    body += text(w / 2, 26, "receptive field of one layer-3 unit, in the input", size=14)
    body += text(
        w / 2,
        h - 26,
        "1 + 3 × (3 − 1) = 7 pixels across",
        size=12.5,
        font=MONO,
        fill=INK,
    )
    return wrap(
        w,
        h,
        "rfFig",
        "Receptive field growth across three convolutional layers",
        "Three stacked three-wide convolutions drawn in one spatial dimension. "
        "A single layer-three unit reads three layer-two units, which between "
        "them read five layer-one units, which between them read seven input "
        "pixels — two more per layer, as one plus L times K minus one gives.",
        body,
    )


# ---------------------------------------------------------------------------
# 6. Max pooling.
# ---------------------------------------------------------------------------


def fig_pooling():
    """Max pooling with a 2x2 kernel at stride 2."""
    w, h = 620, 300
    cell = 44
    vals = [[1, 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]]
    ix, iy = 56, 70
    body = text(ix + 2 * cell, 44, "one channel plane", size=14)
    for j in range(4):
        for i in range(4):
            tile_max = max(vals[2 * (j // 2) + a][2 * (i // 2) + b] for a in range(2) for b in range(2))
            hot = vals[j][i] == tile_max
            body += rect(
                ix + i * cell,
                iy + j * cell,
                cell,
                cell,
                fill=TEAL if hot else "none",
                op=0.55 if hot else 0.0,
                sw=0.7,
                so=0.4,
            )
            body += text(
                ix + i * cell + cell / 2,
                iy + j * cell + cell / 2 + 5,
                str(vals[j][i]),
                size=15,
                font=MONO,
            )
    # The 2x2 tile boundaries, drawn over the cells.
    for t in range(3):
        body += line(ix + t * 2 * cell, iy, ix + t * 2 * cell, iy + 4 * cell, sw=1.8, op=0.75)
        body += line(ix, iy + t * 2 * cell, ix + 4 * cell, iy + t * 2 * cell, sw=1.8, op=0.75)

    ox, oy = 420, 70 + cell / 2
    out = [[6, 8], [3, 4]]
    for j in range(2):
        for i in range(2):
            body += rect(ox + i * cell, oy + j * cell, cell, cell, fill=TEAL, op=0.55, sw=1.8, so=0.75)
            body += text(
                ox + i * cell + cell / 2,
                oy + j * cell + cell / 2 + 5,
                str(out[j][i]),
                size=15,
                font=MONO,
            )
    body += text(ox + cell, 44, "after pooling", size=14)
    body += arrow(ix + 4 * cell + 22, iy + 2 * cell, ox - 22, iy + 2 * cell)
    body += text(
        (ix + 4 * cell + ox) / 2,
        iy + 2 * cell - 14,
        "max, K = 2",
        size=11.5,
        fill=INK,
        style="italic",
    )
    body += text(
        (ix + 4 * cell + ox) / 2,
        iy + 2 * cell + 24,
        "stride 2",
        size=11.5,
        fill=INK,
        style="italic",
    )
    body += text(
        w / 2,
        h - 18,
        "each output is the largest value in its tile; no parameters are involved",
        size=12.5,
        fill=INK,
        style="italic",
    )
    return wrap(
        w,
        h,
        "poolFig",
        "Max pooling with a two-by-two kernel at stride two",
        "A four-by-four plane divided into four non-overlapping two-by-two "
        "tiles, with the largest value in each tile highlighted, beside the "
        "two-by-two output made of those four maxima.",
        body,
    )


# ---------------------------------------------------------------------------
# 7. The equivariance square.
# ---------------------------------------------------------------------------


def small_scene(x, y, s, shift):
    """A schematic feature map with one blob, offset horizontally by `shift`."""
    body = rect(x, y, s, s, fill=INDIGO, op=0.10, sw=1.0, so=0.55)
    cx = x + s * (0.3 + shift)
    cy = y + s * 0.42
    body += f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{s * 0.13:.1f}" fill="{TEAL}" fill-opacity="0.85"/>\n'
    body += poly(
        [(cx - s * 0.13, cy + s * 0.30), (cx, cy + s * 0.12), (cx + s * 0.13, cy + s * 0.30)],
        fill=TEAL,
        stroke="none",
        op=0.55,
    )
    return body


def fig_equivariance():
    """The commutative square: convolution commutes with translation."""
    w, h = 700, 468
    s = 124
    x0, x1 = 92, 424
    y0, y1 = 74, 250
    body = ""
    body += small_scene(x0, y0, s, 0.0)
    body += small_scene(x1, y0, s, 0.0)
    body += small_scene(x0, y1, s, 0.28)
    body += small_scene(x1, y1, s, 0.28)

    body += arrow(x0 + s + 16, y0 + s / 2, x1 - 16, y0 + s / 2)
    body += arrow(x0 + s + 16, y1 + s / 2, x1 - 16, y1 + s / 2)
    body += arrow(x0 + s / 2, y0 + s + 16, x0 + s / 2, y1 - 16)
    body += arrow(x1 + s / 2, y0 + s + 16, x1 + s / 2, y1 - 16)

    body += text((x0 + s + x1) / 2, y0 + s / 2 - 12, "conv or pool", size=12.5, fill=INK)
    body += text((x0 + s + x1) / 2, y1 + s / 2 - 12, "conv or pool", size=12.5, fill=INK)
    body += text(x0 + s / 2 - 12, (y0 + s + y1) / 2 + 4, "translate", size=12.5, anchor="end", fill=INK)
    body += text(x1 + s / 2 + 12, (y0 + s + y1) / 2 + 4, "translate", size=12.5, anchor="start", fill=INK)

    body += text(x0 + s / 2, y0 - 22, "image", size=13.5)
    body += text(x1 + s / 2, y0 - 22, "feature map", size=13.5)
    body += text(
        w / 2,
        h - 48,
        "f(T(x))  =  T(f(x))",
        size=17,
        font=MONO,
        fill=PURPLE,
    )
    body += text(
        w / 2,
        h - 20,
        "both routes reach the same feature map",
        size=12.5,
        fill=INK,
        style="italic",
    )
    return wrap(
        w,
        h,
        "equivFig",
        "Convolution commutes with translation",
        "A commutative square. Applying a convolution and then shifting the "
        "result gives the same feature map as shifting the image first and then "
        "applying the convolution.",
        body,
    )


FIGURES = {
    "diagram_flatten.svg": fig_flatten,
    "diagram_conv_layer.svg": fig_conv_layer,
    "diagram_gabor.svg": fig_gabor,
    "diagram_padding_stride.svg": fig_padding_stride,
    "diagram_receptive_field.svg": fig_receptive_field,
    "diagram_pooling.svg": fig_pooling,
    "diagram_equivariance.svg": fig_equivariance,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print(f"wrote {OUT / name}")
