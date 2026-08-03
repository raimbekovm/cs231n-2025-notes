"""Generate the five SVG diagrams for the lecture 6 chapter.

Every number plotted here is computed rather than traced. The VGG budget chart
is the architecture of configuration D evaluated layer by layer; the ILSVRC
chart is the published challenge results; the normalization panels are the
shaded regions of @eq-layernorm and @eq-batchnorm-stats drawn as geometry.
Conventions follow figures/05-convolutional-networks: currentColor for text and
rules, viridis for data, title/desc with namespaced ids, no background rect
(the theme supplies a light plate in both schemes).

Run from the repository root; rewrites all five files.
"""

import math
import pathlib

OUT = pathlib.Path("figures/06-cnn-architectures")

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


def text(x, y, s, size=13, fill="currentColor", anchor="middle", font=None, style="", weight=None, op=1.0):
    """One text element. Every caller passes explicit coordinates."""
    f = f' font-family="{font}"' if font else ""
    st = f' font-style="{style}"' if style else ""
    w = f' font-weight="{weight}"' if weight else ""
    o = f' fill-opacity="{op}"' if op != 1.0 else ""
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"'
        f' text-anchor="{anchor}"{f}{st}{w}{o}>{s}</text>\n'
    )


def rect(x, y, w, h, fill="none", stroke="currentColor", sw=1.0, op=1.0, so=1.0, rx=None):
    """One rectangle."""
    r = f' rx="{rx}"' if rx else ""
    return (
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"'
        f' fill="{fill}" fill-opacity="{op}" stroke="{stroke}"'
        f' stroke-width="{sw}" stroke-opacity="{so}"{r}/>\n'
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


def path(d, fill="none", stroke="currentColor", sw=1.0, op=1.0, dash=None):
    """One path."""
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'  <path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-opacity="{op}"{da}/>\n'


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


def cuboid(x, y, w, h, dx, dy, face, top=None, side=None, op=0.85, sw=1.0, so=0.7):
    """A box in oblique projection: front face at (x, y), back offset by (dx, dy)."""
    top = top or face
    side = side or face
    out = poly(
        [(x, y), (x + dx, y + dy), (x + w + dx, y + dy), (x + w, y)],
        fill=top,
        op=op * 0.55,
        sw=sw,
        so=so * 0.8,
    )
    out += poly(
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        fill=side,
        op=op * 0.75,
        sw=sw,
        so=so * 0.8,
    )
    out += rect(x, y, w, h, fill=face, op=op, sw=sw, so=so)
    return out


# ---------------------------------------------------------------------------
# 1. Which axes each normalization layer averages over.
# ---------------------------------------------------------------------------

BW, BH = 118.0, 90.0  # front face of the activation block
DX, DY = 32.0, -24.0  # oblique depth offset (the batch axis)
NROW = 6  # channels drawn
NSLICE = 4  # samples drawn


def _norm_panel(x0, y0, name, shade):
    """One activation block with `shade` = list of (row0, nrow, slice0, nslice)."""
    out = ""
    # Faint wireframe of the whole block.
    out += poly([(x0, y0), (x0 + DX, y0 + DY), (x0 + BW + DX, y0 + DY), (x0 + BW, y0)], fill="none", sw=0.8, so=0.35)
    out += poly(
        [(x0 + BW, y0), (x0 + BW + DX, y0 + DY), (x0 + BW + DX, y0 + BH + DY), (x0 + BW, y0 + BH)],
        fill="none",
        sw=0.8,
        so=0.35,
    )
    out += rect(x0, y0, BW, BH, fill="none", sw=0.9, so=0.55)

    # Channel rows on the front face.
    for i in range(1, NROW):
        yy = y0 + i * BH / NROW
        out += line(x0, yy, x0 + BW, yy, sw=0.55, op=0.3)
    # Sample divisions on the top face.
    for k in range(1, NSLICE):
        t = k / NSLICE
        out += line(x0 + t * DX, y0 + t * DY, x0 + BW + t * DX, y0 + t * DY, sw=0.55, op=0.3)

    # The shaded region, drawn back-to-front so the near slices overlap correctly.
    for row0, nrow, slice0, nslice in shade:
        sy = y0 + row0 * BH / NROW
        sh = nrow * BH / NROW
        near = slice0 / NSLICE
        far = (slice0 + nslice) / NSLICE
        ox, oy = near * DX, near * DY
        ddx, ddy = (far - near) * DX, (far - near) * DY
        out += cuboid(x0 + ox, sy + oy, BW, sh, ddx, ddy, TEAL, op=0.62, sw=0.9, so=0.55)

    out += text(x0 + BW / 2 + DX / 2, y0 + BH + 52, name, size=13.5)
    return out


def fig_norm_axes():
    """Four activation blocks, each shading the subset one normalization layer pools."""
    w, h = 880, 234
    body = ""
    panels = [
        # BatchNorm: one channel, every sample, every spatial position.
        ("Batch norm", [(2, 1, 0, NSLICE)]),
        # LayerNorm: one sample, every channel, every spatial position.
        ("Layer norm", [(0, NROW, 0, 1)]),
        # InstanceNorm: one sample, one channel.
        ("Instance norm", [(2, 1, 0, 1)]),
        # GroupNorm: one sample, a group of channels.
        ("Group norm", [(1, 3, 0, 1)]),
    ]
    x = 42.0
    y = 66.0
    for name, shade in panels:
        body += _norm_panel(x, y, name, shade)
        x += 210.0

    # Axis key, drawn once at the left of the first panel.
    body += arrow(30, 66 + BH, 30, 66 + 16, sw=1.0, op=0.6, head=5)
    body += text(20, 66 + BH / 2 + 4, "C", size=12, anchor="middle", style="italic", op=0.75)
    body += arrow(42, 66 + BH + 13, 42 + BW, 66 + BH + 13, sw=1.0, op=0.6, head=5)
    body += text(42 + BW / 2, 66 + BH + 26, "H × W", size=11, op=0.75)
    body += arrow(42 + BW + 6, 66 - 3, 42 + BW + 6 + DX, 66 - 3 + DY, sw=1.0, op=0.6, head=5)
    body += text(42 + BW + 30, 66 + DY - 8, "N", size=12, style="italic", op=0.75)

    body += text(w / 2, 26, "Which activations are pooled into one mean and variance", size=13.5, op=0.9)
    return wrap(
        w,
        h,
        "normAxes",
        "Normalization layers differ only in which activations they average over",
        "Four copies of the same block of convolutional activations, with axes "
        "for batch, channel and spatial position. In each copy a different "
        "subset is shaded: batch normalization takes one channel across the "
        "whole batch, layer normalization takes one sample across all channels, "
        "instance normalization takes one channel of one sample, and group "
        "normalization takes a group of channels of one sample.",
        body,
    )


# ---------------------------------------------------------------------------
# 2. ILSVRC: winning error against depth.
# ---------------------------------------------------------------------------

# Winning top-5 error, from the official challenge results; depth is the depth
# of the winning entry as reported in its own paper. 2016 is omitted: its
# winning entry is an ensemble with no single depth and no paper of its own.
ILSVRC = [
    (2010, 28.19, 0, "NEC-UIUC"),
    (2011, 25.77, 0, "XRCE"),
    (2012, 15.32, 8, "AlexNet"),
    (2013, 11.20, 8, "ZFNet"),
    (2014, 6.66, 22, "GoogLeNet"),
    (2015, 3.57, 152, "ResNet"),
    (2017, 2.25, None, "SENet"),
]
HUMAN = 5.1


def fig_depth_error():
    """ILSVRC winning top-5 error against the depth of the winning entry."""
    w, h = 780, 400
    L, R, T, B = 62.0, 706.0, 56.0, 316.0
    body = ""

    n = len(ILSVRC)
    step = (R - L) / n
    depth_max = 160.0

    def ey(e):
        """Log scale for error, 2% at the bottom to 32% at the top."""
        lo, hi = math.log(2.0), math.log(32.0)
        return B - (math.log(e) - lo) / (hi - lo) * (B - T)

    def dy(d):
        return B - d / depth_max * (B - T)

    # Error gridlines and right-hand axis.
    for e in (2, 4, 8, 16, 32):
        body += line(L, ey(e), R, ey(e), sw=0.6, op=0.18, dash="3 4")
        body += text(R + 10, ey(e) + 4, f"{e}%", size=11, anchor="start", op=0.65, font=MONO)
    # Depth axis on the left.
    for d in (0, 40, 80, 120, 160):
        body += text(L - 12, dy(d) + 4, f"{d}", size=11, anchor="end", op=0.65, font=MONO)

    body += line(L, B, R, B, sw=1.0, op=0.6)

    # Depth bars.
    bw = 26.0
    for i, (year, err, depth, name) in enumerate(ILSVRC):
        cx = L + (i + 0.5) * step
        if depth is None:
            body += text(cx, B - 12, "?", size=13, op=0.35, font=MONO)
            continue
        if depth == 0:
            body += line(cx - bw / 2, B, cx + bw / 2, B, sw=2.4, op=0.55, stroke=INDIGO)
            continue
        body += rect(cx - bw / 2, dy(depth), bw, B - dy(depth), fill=INDIGO, stroke=INDIGO, op=0.34, sw=0.9, so=0.55)
        body += text(cx, dy(depth) - 6, f"{depth}", size=10.5, op=0.7, font=MONO, fill=INDIGO)

    # Human baseline.
    body += line(L, ey(HUMAN), R, ey(HUMAN), sw=1.1, op=0.5, dash="6 4", stroke=PURPLE)
    body += text(L + 6, ey(HUMAN) - 7, "human annotator, 5.1%", size=10.5, anchor="start", op=0.75, fill=PURPLE)

    # Error line.
    pts = [(L + (i + 0.5) * step, ey(e)) for i, (_, e, _, _) in enumerate(ILSVRC)]
    body += path("M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts), stroke=TEAL, sw=2.0, op=0.9)
    for i, (x, y) in enumerate(pts):
        body += f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{TEAL}"/>\n'
        err = ILSVRC[i][1]
        body += text(x, y - 11, f"{err:.2f}", size=10.5, font=MONO, fill=TEAL, op=0.95)

    # Year and entry labels below the axis.
    for i, (year, err, depth, name) in enumerate(ILSVRC):
        cx = L + (i + 0.5) * step
        body += text(cx, B + 18, f"{year}", size=11.5, font=MONO, op=0.8)
        body += text(cx, B + 33, name, size=11, op=0.7)

    # The 2016 gap is real; say so on the chart rather than in the caption only.
    gap_x = L + 5.5 * step + step / 2
    body += text(gap_x, B + 50, "2016 omitted: its winner was an ensemble", size=10, op=0.5, style="italic")

    # Legend and titles.
    body += text(L - 12, T - 26, "layers", size=11.5, anchor="end", op=0.75)
    body += text(R + 10, T - 26, "top-5 error", size=11.5, anchor="start", op=0.75, fill=TEAL)
    body += text(w / 2, 26, "ILSVRC winners: error fell as depth rose", size=14, op=0.9)

    return wrap(
        w,
        h,
        "depthErr",
        "ILSVRC winning error against the depth of the winning entry",
        "Bars give the depth of each year's winning entry and a line gives its "
        "top-5 error on a logarithmic scale. The two pre-2012 winners are "
        "shallow feature pipelines. Error falls from 28% in 2010 to 2.25% in "
        "2017 while depth rises from zero to over a hundred layers, crossing "
        "the 5.1% human annotator baseline in 2015.",
        body,
    )


# ---------------------------------------------------------------------------
# 3. Where VGG-16 spends its memory and its parameters.
# ---------------------------------------------------------------------------


def vgg16_budget():
    """Activation count and parameter count per layer, VGG-16 configuration D."""
    layers = []
    hw, c = 224, 3
    plan = [
        ("c1_1", "conv", 64),
        ("c1_2", "conv", 64),
        ("p1", "pool", None),
        ("c2_1", "conv", 128),
        ("c2_2", "conv", 128),
        ("p2", "pool", None),
        ("c3_1", "conv", 256),
        ("c3_2", "conv", 256),
        ("c3_3", "conv", 256),
        ("p3", "pool", None),
        ("c4_1", "conv", 512),
        ("c4_2", "conv", 512),
        ("c4_3", "conv", 512),
        ("p4", "pool", None),
        ("c5_1", "conv", 512),
        ("c5_2", "conv", 512),
        ("c5_3", "conv", 512),
        ("p5", "pool", None),
    ]
    for name, kind, out_c in plan:
        if kind == "conv":
            params = 3 * 3 * c * out_c + out_c
            c = out_c
        else:
            params = 0
            hw //= 2
        layers.append((name, hw * hw * c, params))
    flat = hw * hw * c
    layers.append(("fc6", 4096, flat * 4096 + 4096))
    layers.append(("fc7", 4096, 4096 * 4096 + 4096))
    layers.append(("fc8", 1000, 4096 * 1000 + 1000))
    return layers


def fig_vgg_budget():
    """Activation memory above the axis, parameter count below, per VGG-16 layer."""
    layers = vgg16_budget()
    mem_max = max(m for _, m, _ in layers)
    par_max = max(p for _, _, p in layers)
    tot_mem = sum(m for _, m, _ in layers)
    tot_par = sum(p for _, _, p in layers)
    early = sum(m for n, m, _ in layers if n[:2] in ("c1", "c2", "p1", "p2")) / tot_mem
    fc6_share = dict((n, p) for n, _, p in layers)["fc6"] / tot_par

    w, h = 880, 470
    L, R = 74.0, 846.0
    MID = 236.0  # the shared category axis
    TOP, BOT = 74.0, 386.0
    body = ""

    n = len(layers)
    step = (R - L) / n
    bw = step * 0.62

    def mh(v):
        return (v / mem_max) * (MID - TOP)

    def ph(v):
        return (v / par_max) * (BOT - MID)

    # Gridlines: memory above, parameters below.
    for frac, lab in ((0.25, "0.8M"), (0.5, "1.6M"), (0.75, "2.4M"), (1.0, "3.2M")):
        y = MID - frac * (MID - TOP)
        body += line(L, y, R, y, sw=0.6, op=0.16, dash="3 4")
        body += text(L - 10, y + 4, lab, size=10, anchor="end", op=0.6, font=MONO)
    for frac, lab in ((0.25, "26M"), (0.5, "51M"), (0.75, "77M"), (1.0, "103M")):
        y = MID + frac * (BOT - MID)
        body += line(L, y, R, y, sw=0.6, op=0.16, dash="3 4")
        body += text(L - 10, y + 4, lab, size=10, anchor="end", op=0.6, font=MONO)

    for i, (name, mem, par) in enumerate(layers):
        cx = L + (i + 0.5) * step
        if mem:
            body += rect(cx - bw / 2, MID - mh(mem), bw, mh(mem), fill=GREEN, stroke=GREEN, op=0.55, sw=0.8, so=0.7)
        if par:
            body += rect(cx - bw / 2, MID, bw, ph(par), fill=PURPLE, stroke=PURPLE, op=0.55, sw=0.8, so=0.7)
        body += text(cx, MID + 4, name, size=9, font=MONO, op=0.0)

    body += line(L, MID, R, MID, sw=1.1, op=0.7)

    # Category labels sit on the axis line, so they go in a band beneath the
    # bars rather than on top of them.
    for i, (name, mem, par) in enumerate(layers):
        cx = L + (i + 0.5) * step
        body += text(cx, BOT + 17, name, size=9.5, font=MONO, op=0.7)

    body += text(L, TOP - 30, "activations per image", size=12.5, anchor="start", op=0.85, fill=GREEN)
    body += text(
        L,
        TOP - 14,
        f"{tot_mem / 1e6:.1f}M values in all — {early:.0%} of it in the first two blocks",
        size=10.5,
        anchor="start",
        op=0.6,
        style="italic",
    )
    body += text(L, BOT + 46, "learned parameters", size=12.5, anchor="start", op=0.85, fill=PURPLE)
    body += text(
        L,
        BOT + 58,
        f"{tot_par / 1e6:.0f}M in all — {fc6_share:.0%} of it in fc6 alone",
        size=10.5,
        anchor="start",
        op=0.6,
        style="italic",
    )
    body += text(w / 2, 28, "VGG-16: the two costs sit at opposite ends", size=14, op=0.9)

    return wrap(
        w,
        h,
        "vggBudget",
        "Activation memory and parameter count per layer of VGG-16",
        "A diverging bar chart over the twenty-one layers of VGG-16. Bars above "
        "the axis are the number of activation values each layer produces for "
        "one 224 by 224 input; bars below are the number of learned parameters. "
        "Activations peak in the first two convolutional blocks and parameters "
        "peak in the first fully connected layer, which holds about three "
        "quarters of the network's weights.",
        body,
    )


# ---------------------------------------------------------------------------
# 4. Plain block against residual block.
# ---------------------------------------------------------------------------


def _block_box(cx, y, label, w=118.0, hh=30.0, fill=TEAL, op=0.2):
    out = rect(cx - w / 2, y, w, hh, fill=fill, stroke=fill, op=op, sw=1.0, so=0.75, rx=4)
    out += text(cx, y + hh / 2 + 4.5, label, size=12)
    return out


def fig_residual_block():
    """A plain two-convolution block beside the residual block that replaced it."""
    w, h = 620, 400
    body = ""
    y0 = 62.0
    gap = 22.0
    bh = 30.0

    for cx, title, residual in ((160.0, "plain block", False), (450.0, "residual block", True)):
        body += text(cx, 34, title, size=13.5, op=0.9)
        y = y0
        body += text(cx, y, "x", size=14, style="italic")
        top_y = y
        y += 14
        for label in ("3 × 3 conv", "ReLU", "3 × 3 conv"):
            body += arrow(cx, y, cx, y + gap - 2, sw=1.2, op=0.55)
            y += gap
            body += _block_box(
                cx, y, label, fill=TEAL if "conv" in label else INDIGO, op=0.22 if "conv" in label else 0.16
            )
            y += bh

        if residual:
            # The junction where F(x) and the skip meet.
            body += arrow(cx, y, cx, y + gap - 8, sw=1.2, op=0.55)
            y += gap
            jy = y + 9
            body += f'  <circle cx="{cx:.1f}" cy="{jy:.1f}" r="11" fill="none" stroke="currentColor" stroke-width="1.1" stroke-opacity="0.75"/>\n'
            body += text(cx, jy + 5, "+", size=15)
            # The skip path: out to the right of the stack and back in.
            sx = cx + 96
            body += path(
                f"M {cx + 8:.1f} {top_y + 4:.1f} L {sx:.1f} {top_y + 4:.1f} "
                f"L {sx:.1f} {jy:.1f} L {cx + 12:.1f} {jy:.1f}",
                stroke=GREEN,
                sw=1.8,
                op=0.9,
            )
            body += poly([(cx + 11, jy), (cx + 19, jy - 4), (cx + 19, jy + 4)], fill=GREEN, stroke="none", op=0.9)
            body += text(sx + 8, (top_y + jy) / 2 + 4, "identity", size=11, anchor="start", fill=GREEN, op=0.95)
            body += text(cx - 68, (top_y + jy) / 2 + 4, "F(x)", size=12.5, anchor="end", style="italic", op=0.7)
            y = jy + 11
        else:
            body += text(cx - 68, y - bh - gap + 6, "H(x)", size=12.5, anchor="end", style="italic", op=0.7)

        body += arrow(cx, y, cx, y + gap - 2, sw=1.2, op=0.55)
        y += gap
        body += _block_box(cx, y, "ReLU", fill=INDIGO, op=0.16)
        y += bh
        body += arrow(cx, y, cx, y + 18, sw=1.2, op=0.55)
        y += 32
        body += text(cx, y, "H(x)" if not residual else "F(x) + x", size=13, style="italic")

    return wrap(
        w,
        h,
        "resBlock",
        "A plain block and a residual block",
        "Two vertical block diagrams. Both contain the same pair of three by "
        "three convolutions separated by a ReLU. In the residual block a second "
        "path carries the input around the convolutions and adds it back before "
        "the final activation, so the block computes F of x plus x rather than "
        "H of x.",
        body,
    )


# ---------------------------------------------------------------------------
# 5. Four training-curve shapes.
# ---------------------------------------------------------------------------


def _curve(x0, y0, pw, ph, f, stroke, sw=1.9, op=0.95, n=64):
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append((x0 + t * pw, y0 + ph - f(t) * ph))
    return path("M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts), stroke=stroke, sw=sw, op=op)


def fig_learning_curves():
    """Four training-curve shapes, each with the remedy it calls for."""
    w, h = 880, 268
    body = ""
    pw, ph = 168.0, 116.0
    y0 = 62.0

    def sat(t, a, k=3.4):
        """A saturating rise to `a`."""
        return a * (1 - math.exp(-k * t))

    panels = [
        (
            "still rising",
            "train longer",
            lambda t: sat(t, 0.80),
            lambda t: sat(t, 0.72),
        ),
        (
            "wide gap",
            "overfitting — regularize, or get more data",
            lambda t: sat(t, 0.97, 4.2),
            lambda t: sat(t, 0.62, 5.5) - 0.16 * max(0.0, t - 0.45) ** 1.4,
        ),
        (
            "no gap, both low",
            "underfitting — bigger model, less regularization",
            lambda t: sat(t, 0.52),
            lambda t: sat(t, 0.50),
        ),
        (
            "plateau",
            "learning rate too large for this stage",
            lambda t: min(sat(t, 0.62, 9.0), 0.44 + 0.06 * t),
            lambda t: min(sat(t, 0.58, 9.0), 0.40 + 0.05 * t),
        ),
    ]

    x = 46.0
    for name, note, ftr, fva in panels:
        body += rect(x, y0, pw, ph, fill="none", sw=0.8, so=0.35)
        for k in (0.25, 0.5, 0.75):
            body += line(x, y0 + k * ph, x + pw, y0 + k * ph, sw=0.5, op=0.13)
        body += _curve(x, y0, pw, ph, ftr, INDIGO)
        body += _curve(x, y0, pw, ph, fva, GREEN)
        body += text(x + pw / 2, y0 - 12, name, size=12.5, op=0.9)
        # Two lines of remedy, wrapped by hand so nothing runs past the panel.
        first, second = [], []
        for word in note.split():
            if not second and len(" ".join([*first, word])) <= 26:
                first.append(word)
            else:
                second.append(word)
        body += text(x + pw / 2, y0 + ph + 20, " ".join(first), size=10.5, op=0.62, style="italic")
        if second:
            body += text(x + pw / 2, y0 + ph + 34, " ".join(second), size=10.5, op=0.62, style="italic")
        x += pw + 40.0

    body += text(46, 30, "train", size=12, anchor="start", fill=INDIGO, op=0.95)
    body += text(92, 30, "validation", size=12, anchor="start", fill=GREEN, op=0.95)
    body += text(w - 46, 30, "accuracy against time", size=12, anchor="end", op=0.6, style="italic")

    return wrap(
        w,
        h,
        "learnCurves",
        "Four training-curve shapes and what each asks for",
        "Four small plots of training and validation accuracy against time. In "
        "the first both are still rising, calling for a longer run. In the "
        "second a wide and widening gap indicates overfitting. In the third the "
        "two curves coincide at a low value, indicating underfitting. In the "
        "fourth both plateau early, indicating a learning rate too large for "
        "the current stage.",
        body,
    )


FIGURES = {
    "diagram_norm_axes.svg": fig_norm_axes,
    "diagram_depth_error.svg": fig_depth_error,
    "diagram_vgg_budget.svg": fig_vgg_budget,
    "diagram_residual_block.svg": fig_residual_block,
    "diagram_learning_curves.svg": fig_learning_curves,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print(f"wrote {OUT / name}")

    # The chapter quotes these; print them so the prose can be checked.
    layers = vgg16_budget()
    tot_mem = sum(m for _, m, _ in layers)
    tot_par = sum(p for _, _, p in layers)
    first_two = sum(m for n, m, _ in layers if n[:2] in ("c1", "c2", "p1", "p2"))
    fc6 = dict((n, p) for n, _, p in layers)["fc6"]
    print(f"VGG-16 activations {tot_mem / 1e6:.2f}M values, {tot_mem * 4 / 1e6:.1f} MB at fp32")
    print(f"  first two blocks: {first_two / tot_mem:.1%}")
    print(f"VGG-16 parameters  {tot_par / 1e6:.1f}M, fc6 = {fc6 / 1e6:.1f}M = {fc6 / tot_par:.1%}")
