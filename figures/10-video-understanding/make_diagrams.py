"""Generate the eight SVG diagrams for the lecture 10 chapter.

The three bar charts replot published numbers: Sports-1M from Karpathy et al.
(CVPR 2014) with C3D from Tran et al. (ICCV 2015), UCF-101 from Simonyan and
Zisserman (NeurIPS 2014), and the Kinetics-400 progression from the papers
named beside each bar. The five schematics are geometry — where each fusion
strategy consumes the temporal axis, the filter extents that separate 2D from
3D convolution, the two properties a temporal mechanism can have, the wiring of
a non-local block, and the arithmetic that makes I3D inflation exact.

Conventions follow figures/06-cnn-architectures: currentColor for text and
rules, viridis for data, title/desc with namespaced ids, no background rect
(the theme supplies a light plate in both schemes). Yellow is used as a fill
only, never for a line or a label.

Run from the repository root; rewrites all eight files.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _svg import (
    GREEN,
    INDIGO,
    INK,
    MONO,
    PURPLE,
    TEAL,
    YELLOW,
    arrow,
    line,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path("figures/10-video-understanding")


# --------------------------------------------------------------------------
# shared: a simple vertical bar chart, used by three of the figures
# --------------------------------------------------------------------------


def bars(values, lo, hi, width, height, ylabel, colours=None, pad_left=52):
    """Emit a labelled bar chart; values is a list of (label_lines, number)."""
    top, bottom = 34, height - 46
    x0, x1 = pad_left, width - 16
    n = len(values)
    slot = (x1 - x0) / n
    bw = min(slot * 0.56, 58)
    body = ""

    # y axis: four gridlines, drawn under the bars
    steps = 4
    for i in range(steps + 1):
        v = lo + (hi - lo) * i / steps
        y = bottom - (v - lo) / (hi - lo) * (bottom - top)
        body += line(x0 - 6, y, x1, y, op=0.16)
        body += text(x0 - 11, y + 4, f"{v:.0f}", size=11, anchor="end", op=0.7)
    body += text(
        16,
        (top + bottom) / 2,
        ylabel,
        size=11,
        op=0.7,
        style="italic",
    ).replace("<text ", f'<text transform="rotate(-90 16 {(top + bottom) / 2:.1f})" ')

    for i, (label, v) in enumerate(values):
        cx = x0 + slot * (i + 0.5)
        y = bottom - (v - lo) / (hi - lo) * (bottom - top)
        fill = colours[i] if colours else TEAL
        body += rect(cx - bw / 2, y, bw, bottom - y, fill=fill, stroke="none", op=0.75)
        body += text(cx, y - 7, f"{v:.1f}", size=12, weight="600")
        for j, ln in enumerate(label):
            body += text(cx, bottom + 16 + j * 12, ln, size=10.5, op=0.85)
    body += line(x0 - 6, bottom, x1, bottom, op=0.5)
    return body


# --------------------------------------------------------------------------
# 1. where each architecture consumes the temporal axis
# --------------------------------------------------------------------------

# Each panel is a left-to-right chain of stages. A stage is (n, label, fill):
# n is the temporal extent still present at that stage, so n dropping to 1 is
# exactly the moment the architecture collapses time.
FUSION_PANELS = [
    (
        "Single-frame CNN",
        [(5, "frames", INDIGO), (5, "2D CNN", TEAL), (5, "scores", GREEN)],
        "average probabilities",
    ),
    (
        "Late fusion",
        [(5, "frames", INDIGO), (5, "2D CNN", TEAL), (1, "pool / MLP", YELLOW)],
        "fuse features, then classify",
    ),
    (
        "Early fusion",
        [(5, "frames", INDIGO), (1, "conv2D", YELLOW), (1, "2D CNN", TEAL)],
        "one layer does all of time",
    ),
    (
        "3D CNN",
        [(5, "frames", INDIGO), (3, "3D conv/pool", TEAL), (1, "3D conv", YELLOW)],
        "time shrinks gradually",
    ),
]


def _stack(x, cy, n, w, span, fill):
    """Draw n stacked slabs filling a fixed vertical span; return the SVG."""
    gap = 3.0
    h = (span - gap * (n - 1)) / n
    out = ""
    for i in range(n):
        y = cy - span / 2 + i * (h + gap)
        out += rect(x, y, w, h, fill=fill, stroke=INK, sw=0.7, op=0.28, so=0.45)
    return out


def fig_fusion():
    """Where each of the four architectures collapses the temporal axis."""
    w, h = 880, 268
    pw, cy, span, sw_ = 218, 132, 88, 38
    xs = [10, 66, 122]
    body = ""
    for p, (title, stages, foot) in enumerate(FUSION_PANELS):
        ox = 6 + p * pw
        body += text(ox + pw / 2 - 8, 26, title, size=13, weight="600")
        for i, (n, label, fill) in enumerate(stages):
            x = ox + xs[i]
            body += _stack(x, cy, n, sw_, span, fill)
            body += text(x + sw_ / 2, cy + span / 2 + 16, label, size=10, op=0.8)
            if i:
                body += arrow(ox + xs[i - 1] + sw_ + 2, cy, x - 3, cy, sw=1.1, op=0.5)
        # the class-score output, always a single box
        gx = ox + xs[2] + sw_ + 14
        body += arrow(ox + xs[2] + sw_ + 2, cy, gx - 3, cy, sw=1.1, op=0.5)
        body += rect(gx, cy - 9, 20, 18, fill=PURPLE, stroke=INK, sw=0.7, op=0.3, so=0.45)
        body += text(gx + 10, cy + span / 2 + 16, "C", size=10, op=0.8, font=MONO)
        body += text(ox + pw / 2 - 8, cy + span / 2 + 42, foot, size=10.5, op=0.62, style="italic")
        if p:
            body += line(ox - 6, 40, ox - 6, h - 34, op=0.14)
    body += text(
        w / 2,
        h - 10,
        "height of a block = frames still distinguishable at that stage",
        size=10.5,
        op=0.6,
        style="italic",
    )
    return wrap(
        w,
        h,
        "fusion",
        "Where four video architectures consume the temporal axis",
        "Four panels, each a left-to-right chain of blocks. The vertical height "
        "of a block is divided into as many slabs as there are frames still "
        "represented at that stage. The single-frame CNN keeps five slabs all "
        "the way to the class scores and averages probabilities. Late fusion "
        "keeps five through the 2D CNN and collapses to one at a pooling or MLP "
        "layer. Early fusion collapses to one at the first convolution. The 3D "
        "CNN goes from five to three to one across successive layers.",
        body,
    )


# --------------------------------------------------------------------------
# 2. 2D conv on stacked frames vs 3D conv
# --------------------------------------------------------------------------


def _clip(x0, y0, n, cw, ch, gap=4):
    """Draw a clip as n frame columns left to right; return (svg, width)."""
    out = ""
    for i in range(n):
        out += rect(x0 + i * (cw + gap), y0, cw, ch, fill=INDIGO, stroke=INK, sw=0.7, op=0.18, so=0.4)
    return out, n * (cw + gap) - gap


def fig_3d_conv():
    """Filter extent in time for early fusion against a 3D convolution."""
    w, h = 900, 348
    n, cw, ch, gap = 8, 30, 96, 4
    body = ""
    for p, (title, k) in enumerate((("2D conv (early fusion)", n), ("3D conv", 3))):
        ox = 56 + p * 440
        clip_y = 88
        body += text(ox + 134, 30, title, size=13.5, weight="600")
        svg, cwid = _clip(ox, clip_y, n, cw, ch, gap)
        body += svg
        body += text(ox + cwid / 2, clip_y - 12, f"input clip, T = {n}", size=10.5, op=0.75)
        body += text(ox - 10, clip_y + ch / 2 + 4, "H, W", size=10, op=0.7, anchor="end")

        # the filter footprint: k frames wide, part of the spatial extent tall
        fh = 34
        fy = clip_y + 26
        fw_ = k * (cw + gap) - gap
        body += rect(ox - 2, fy, fw_ + 4, fh, fill=TEAL, stroke=TEAL, sw=1.6, op=0.3, so=1.0)
        body += text(ox + fw_ / 2, clip_y + ch + 18, "filter", size=10, fill=TEAL, weight="600")
        if p:
            # a second, shifted position to show the filter slides over time
            sx = ox + 5 * (cw + gap)
            body += rect(sx - 2, fy, fw_ + 4, fh, fill="none", stroke=TEAL, sw=1.4, op=1.0, so=0.85).replace(
                "/>", ' stroke-dasharray="4 3"/>'
            )
            body += arrow(ox + fw_ + 6, fy + fh / 2, sx - 6, fy + fh / 2, sw=1.2, op=0.8, stroke=TEAL)
            body += text(sx + fw_ / 2, clip_y + ch + 18, "slides over t", size=10, fill=TEAL)

        # the output, below
        oy = 232
        if p:
            svg, owid = _clip(ox, oy, n, cw, 40, gap)
            body += svg.replace(INDIGO, PURPLE)
            body += text(ox + owid / 2, oy + 58, "output keeps T", size=11, op=0.85)
        else:
            body += rect(ox, oy, cw, 40, fill=PURPLE, stroke=INK, sw=0.7, op=0.18, so=0.4)
            body += text(ox + 70, oy + 58, "output has no t axis", size=11, op=0.85, anchor="start")
        body += arrow(ox + 15, clip_y + ch + 6, ox + 15, oy - 5, sw=1.1, op=0.45)

        note = (
            "same motion later in the clip\nneeds a different filter"
            if p == 0
            else "one filter, any moment\nin the clip"
        )
        for j, ln in enumerate(note.split("\n")):
            body += text(ox + 320, oy + 12 + j * 15, ln, size=10.5, op=0.7, style="italic")
        if p:
            body += line(ox - 32, 44, ox - 32, h - 26, op=0.14)
    body += text(56, h - 8, "time", size=10.5, op=0.7, anchor="start", style="italic")
    body += arrow(84, h - 12, 128, h - 12, sw=1.0, op=0.45)
    return wrap(
        w,
        h,
        "conv3d",
        "Temporal extent of an early-fusion filter against a 3D filter",
        "Two panels. In each the input clip is drawn as eight frame columns "
        "running left to right in time. On the left the early-fusion filter "
        "spans all eight columns and its output is a single block with no "
        "temporal axis. On the right the 3D filter spans three columns and is "
        "shown again in a shifted position, with an arrow indicating that it "
        "slides over time; its output retains all eight columns.",
        body,
    )


# --------------------------------------------------------------------------
# 3-5. the three bar charts
# --------------------------------------------------------------------------

SPORTS1M = [
    (["Single", "frame"], 77.7),
    (["Early", "fusion"], 76.8),
    (["Late", "fusion"], 78.7),
    (["3D CNN", "(slow fusion)"], 80.2),
    (["C3D", "(2015)"], 84.4),
]

UCF101 = [
    (["3D CNN"], 65.4),
    (["Spatial", "only"], 73.0),
    (["Temporal", "only"], 83.7),
    (["Two-stream", "(average)"], 86.9),
    (["Two-stream", "(SVM)"], 88.0),
]

KINETICS = [
    (["Per-frame", "CNN"], 62.2),
    (["CNN +", "LSTM"], 63.3),
    (["Two-stream", "CNN"], 65.6),
    (["I3D", "(RGB)"], 71.1),
    (["Two-stream", "I3D"], 74.2),
    (["SlowFast", "+ non-local"], 79.8),
    (["MViTv2-L", "(2022)"], 86.1),
    (["VideoMAE", "V2-g (2023)"], 90.0),
]


def fig_sports1m():
    """Video-level top-5 accuracy on Sports-1M for the four fusion strategies."""
    w, h = 660, 300
    cols = [INDIGO, GREEN, GREEN, TEAL, PURPLE]
    body = bars(SPORTS1M, 70, 86, w, h, "top-5 accuracy (%)", cols)
    body += text(w / 2, 20, "Sports-1M, video-level top-5 accuracy", size=13, weight="600")
    return wrap(
        w,
        h,
        "sports1m",
        "Top-5 accuracy on Sports-1M by fusion strategy",
        "A bar chart of five video-level top-5 accuracies on Sports-1M: single "
        "frame 77.7, early fusion 76.8, late fusion 78.7, 3D CNN 80.2, C3D 84.4.",
        body,
    )


def fig_ucf101():
    """Accuracy on UCF-101 for the two streams separately and fused."""
    w, h = 660, 300
    cols = [INDIGO, GREEN, TEAL, PURPLE, PURPLE]
    body = bars(UCF101, 60, 92, w, h, "accuracy (%)", cols)
    body += text(w / 2, 20, "UCF-101, appearance against motion", size=13, weight="600")
    return wrap(
        w,
        h,
        "ucf101",
        "Accuracy on UCF-101 for appearance, motion and both",
        "A bar chart of five accuracies on UCF-101: a 3D CNN at 65.4, the "
        "appearance stream alone at 73.0, the motion stream alone at 83.7, and "
        "the two streams fused at 86.9 by averaging and 88.0 by an SVM.",
        body,
    )


def fig_kinetics():
    """A decade of top-1 accuracy on Kinetics-400."""
    w, h = 880, 316
    cols = [INDIGO, INDIGO, GREEN, GREEN, TEAL, TEAL, PURPLE, PURPLE]
    body = bars(KINETICS, 55, 95, w, h, "top-1 accuracy (%)", cols)
    body += text(w / 2, 20, "Kinetics-400, top-1 accuracy, 2014–2023", size=13, weight="600")
    return wrap(
        w,
        h,
        "kinetics",
        "Top-1 accuracy on Kinetics-400 from 2014 to 2023",
        "A bar chart of eight top-1 accuracies on Kinetics-400: per-frame CNN "
        "62.2, CNN plus LSTM 63.3, two-stream CNN 65.6, I3D 71.1, two-stream "
        "I3D 74.2, SlowFast with non-local blocks 79.8, MViTv2-L 86.1, and "
        "VideoMAE V2-g 90.0.",
        body,
    )


# --------------------------------------------------------------------------
# 6. the two properties a temporal mechanism can have
# --------------------------------------------------------------------------


def fig_cnn_rnn():
    """The temporal-extent by structure grid and the three corners taken."""
    w, h = 600, 432
    x0, x1, y0, y1 = 96, 552, 62, 318
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    body = rect(x0, y0, x1 - x0, y1 - y0, fill="none", stroke=INK, sw=0.9, op=1.0, so=0.28)
    body += line(mx, y0, mx, y1, op=0.22, dash="5 4")
    body += line(x0, my, x1, my, op=0.22, dash="5 4")

    # axis labels
    body += text(mx, y1 + 26, "temporal extent", size=12, op=0.85, style="italic")
    body += text((x0 + mx) / 2, y1 + 44, "finite", size=11, op=0.7)
    body += text((mx + x1) / 2, y1 + 44, "unbounded", size=11, op=0.7)
    lbl = text(30, my, "state structure", size=12, op=0.85, style="italic")
    body += lbl.replace("<text ", f'<text transform="rotate(-90 30 {my:.1f})" ')
    for ty, s in (((y0 + my) / 2, "convolutional"), ((my + y1) / 2, "flat vector")):
        t = text(52, ty, s, size=11, op=0.7)
        body += t.replace("<text ", f'<text transform="rotate(-90 52 {ty:.1f})" ')

    cells = [
        ((x0 + mx) / 2, (y0 + my) / 2, "Temporal CNN", TEAL, ["local window in time,", "shared weights"]),
        ((mx + x1) / 2, (y0 + my) / 2, "Recurrent CNN", PURPLE, ["hidden state is a", "feature map"]),
        ((mx + x1) / 2, (my + y1) / 2, "RNN / LSTM", INDIGO, ["hidden state is a", "vector"]),
    ]
    for cx, cy, name, col, note in cells:
        body += rect(cx - 96, cy - 40, 192, 80, fill=col, stroke=col, sw=1.2, op=0.16, so=0.75, rx=6)
        body += text(cx, cy - 8, name, size=13, weight="600")
        for j, ln in enumerate(note):
            body += text(cx, cy + 12 + j * 14, ln, size=10.5, op=0.72)
    body += text((x0 + mx) / 2, (my + y1) / 2, "—", size=18, op=0.3)

    body += text(
        w / 2, h - 32, "The recurrent CNN takes one property from each neighbour;", size=11, op=0.72, style="italic"
    )
    body += text(
        w / 2,
        h - 16,
        "its cost is that the recurrence cannot be parallelised over time.",
        size=11,
        op=0.72,
        style="italic",
    )
    return wrap(
        w,
        h,
        "cnnrnn",
        "Temporal extent against state structure",
        "A two-by-two grid. The horizontal axis runs from finite to unbounded "
        "temporal extent, the vertical from a flat-vector state to a "
        "convolutional one. A temporal CNN occupies finite and convolutional, "
        "an RNN unbounded and flat-vector, and a recurrent CNN unbounded and "
        "convolutional. The fourth cell is empty.",
        body,
    )


# --------------------------------------------------------------------------
# 7. the non-local block
# --------------------------------------------------------------------------


def fig_nonlocal():
    """The wiring of a non-local block, input to residual sum."""
    w, h = 880, 300
    cy = 168
    body = ""

    def box(x, y, bw, bh, label, sub, col, size=11.5):
        out = rect(x, y, bw, bh, fill=col, stroke=col, sw=1.1, op=0.16, so=0.8, rx=5)
        out += text(x + bw / 2, y + bh / 2 + (0 if not sub else -5), label, size=size, weight="600")
        if sub:
            out += text(x + bw / 2, y + bh / 2 + 12, sub, size=9.5, op=0.7, font=MONO)
        return out

    # x is assigned strictly in flow order, left to right
    xin, xp, xa, xw, xo, xs = 20, 132, 300, 452, 604, 736
    body += box(xin, cy - 34, 90, 68, "features", "C×T×H×W", INDIGO)

    proj = [("queries", INDIGO, cy - 92), ("keys", TEAL, cy - 20), ("values", GREEN, cy + 52)]
    for name, col, py in proj:
        body += box(xp, py - 22, 96, 44, name, "1×1×1 conv", col, size=11)
        body += arrow(xin + 90, cy, xp - 4, py, sw=1.1, op=0.45)

    body += box(xa, cy - 92, 118, 68, "affinities", "THW × THW", TEAL)
    body += arrow(xp + 96, cy - 92, xa - 4, cy - 72, sw=1.1, op=0.5)
    body += arrow(xp + 96, cy - 20, xa - 4, cy - 44, sw=1.1, op=0.5)
    body += text(xa + 59, cy - 106, "softmax over positions", size=10, op=0.68)

    body += box(xw, cy - 34, 116, 68, "weighted", "sum of values", PURPLE)
    body += arrow(xa + 118, cy - 58, xw - 4, cy - 12, sw=1.1, op=0.5)
    body += arrow(xp + 96, cy + 52, xw - 4, cy + 14, sw=1.1, op=0.5)

    body += box(xo, cy - 34, 96, 68, "1×1×1", "back to C", YELLOW)
    body += arrow(xw + 116, cy, xo - 4, cy, sw=1.1, op=0.5)
    body += text(xo + 48, cy + 54, "zero-initialised", size=10, op=0.68, style="italic")

    body += rect(xs, cy - 17, 34, 34, fill="none", stroke=INK, sw=1.1, so=0.55, rx=17)
    body += text(xs + 17, cy + 5, "+", size=17, op=0.75)
    body += arrow(xo + 96, cy, xs - 4, cy, sw=1.1, op=0.5)
    body += arrow(xs + 34, cy, w - 22, cy, sw=1.1, op=0.5)
    body += text(w - 26, cy - 12, "out", size=11, op=0.8, anchor="end")

    # residual, routed above the block so it never crosses a projection arrow
    ry = 44
    body += line(xin + 45, cy - 34, xin + 45, ry, op=0.5, sw=1.2, dash="5 4")
    body += line(xin + 45, ry, xs + 17, ry, op=0.5, sw=1.2, dash="5 4")
    body += arrow(xs + 17, ry, xs + 17, cy - 21, sw=1.2, op=0.5, dash="5 4")
    body += text(
        (xin + 45 + xs) / 2, ry - 9, "residual: identity at initialisation", size=10.5, op=0.68, style="italic"
    )
    return wrap(
        w,
        h,
        "nonlocal",
        "The wiring of a non-local block",
        "A left-to-right diagram. The input feature map feeds three parallel "
        "one-by-one-by-one convolutions producing queries, keys and values. "
        "Queries and keys form a THW by THW affinity matrix, softmaxed over "
        "positions; this weights a sum of the values. A final "
        "one-by-one-by-one convolution restores the channel count and its "
        "output is added to the input through a residual connection drawn above "
        "the block.",
        body,
    )


# --------------------------------------------------------------------------
# 8. why the 1/Kt scaling makes inflation exact
# --------------------------------------------------------------------------


def fig_inflation():
    """The copy-and-divide construction that makes I3D initialisation exact."""
    w, h = 820, 356
    kt = 4
    body = ""
    xin, xk, xout = 40, 300, 596

    def slab(x, y, bw, bh, col, op=0.2):
        return rect(x, y, bw, bh, fill=col, stroke=INK, sw=0.7, op=op, so=0.4)

    # top row: the 2D network on a still image
    ty = 74
    body += text(xin, 40, "trained 2D layer", size=13, weight="600", anchor="start")
    body += slab(xin, ty, 62, 62, INDIGO)
    body += text(xin + 31, ty + 78, "image 3×H×W", size=10.5, op=0.8)
    body += slab(xk, ty + 14, 40, 34, TEAL, op=0.35)
    body += text(xk + 20, ty + 62, "kernel W", size=10.5, op=0.8)
    body += arrow(xin + 62 + 6, ty + 31, xk - 6, ty + 31, sw=1.1, op=0.5)
    body += slab(xout, ty, 56, 62, PURPLE)
    body += text(xout + 28, ty - 10, "output H×W", size=10.5, op=0.8)
    body += arrow(xk + 40 + 6, ty + 31, xout - 6, ty + 31, sw=1.1, op=0.5)

    # bottom row: the inflated layer on a constant video
    by = 216
    body += slab(xin, by, 62, 62, INDIGO)
    for i in range(1, kt):
        body += slab(xin + i * 9, by - i * 7, 62, 62, INDIGO, op=0.12)
    body += text(xin + 45, by + 78, f"same frame ×{kt}", size=10.5, op=0.8)
    for i in range(kt):
        body += slab(xk + i * 22, by + 14 - i * 6, 40, 34, TEAL, op=0.35)
        body += text(xk + i * 22 + 20, by + 14 - i * 6 + 22, "W/4", size=8.5, op=0.85, font=MONO)
    body += text(xk + 53, by + 74, f"kernel copied {kt}×, each divided by {kt}", size=10.5, op=0.8)
    body += arrow(xin + 62 + 6, by + 31, xk - 6, by + 31, sw=1.1, op=0.5)
    body += slab(xout, by, 56, 62, PURPLE)
    body += text(xout + 28, by + 78, "output 1×H×W", size=10.5, op=0.8)
    body += arrow(xk + kt * 22 + 24, by + 31, xout - 6, by + 31, sw=1.1, op=0.5)
    body += text(xin, 178, "inflated 3D layer", size=13, weight="600", anchor="start")

    # the equality between the two outputs
    body += line(xout + 28, ty + 62 + 4, xout + 28, by - 4, op=0.45, sw=1.2, dash="5 4")
    body += text(xout + 46, (ty + 62 + by) / 2 + 5, "identical", size=12, op=0.8, anchor="start", style="italic")
    body += text(
        w / 2,
        h - 12,
        "so the inflated network starts out reproducing the 2D network it was built from",
        size=10.5,
        op=0.65,
        style="italic",
    )
    return wrap(
        w,
        h,
        "inflation",
        "Why the one-over-Kt scaling makes inflation exact",
        "Two rows. The top row shows a trained 2D kernel applied to a still "
        "image, producing an H by W output. The bottom row shows the same frame "
        "repeated four times, a kernel copied four times along the new temporal "
        "axis with each copy divided by four, and an output of one by H by W. A "
        "dashed line marks the two outputs as identical.",
        body,
    )


FIGURES = {
    "diagram_fusion.svg": fig_fusion,
    "diagram_3d_conv.svg": fig_3d_conv,
    "diagram_sports1m.svg": fig_sports1m,
    "diagram_ucf101.svg": fig_ucf101,
    "diagram_kinetics.svg": fig_kinetics,
    "diagram_cnn_rnn.svg": fig_cnn_rnn,
    "diagram_nonlocal.svg": fig_nonlocal,
    "diagram_inflation.svg": fig_inflation,
}


def clip_bytes(t, h, w, fps, seconds, bpp=3):
    """Uncompressed size in bytes of `seconds` of video at the given geometry."""
    return round(t if t else fps * seconds) * h * w * bpp


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print(f"wrote {OUT / name}")

    # The chapter quotes these; print them so the prose can be checked.
    sd = clip_bytes(0, 480, 640, 30, 60)
    hd = clip_bytes(0, 1080, 1920, 30, 60)
    clip = clip_bytes(16, 112, 112, 5, 3.2)
    print(f"SD 640x480 @30fps, 1 min: {sd / 2**30:.2f} GiB")
    print(f"HD 1920x1080 @30fps, 1 min: {hd / 2**30:.2f} GiB")
    print(f"clip T=16, 112x112: {clip / 2**10:.0f} KiB, {16 / 5:.1f} s at 5 fps")
    print(f"C3D 39.5 GFLOP = {39.5 / 13.6:.1f}x VGG-16, {39.5 / 0.7:.0f}x AlexNet")
    for n, (t, hh, ww) in (("16x14x14", (16, 14, 14)), ("16x56x56", (16, 56, 56))):
        p = t * hh * ww
        print(f"non-local at {n}: {p} positions, {p * p / 1e9:.2f}e9 affinities")
