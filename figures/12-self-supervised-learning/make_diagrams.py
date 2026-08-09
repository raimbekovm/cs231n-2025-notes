"""Generate the ten SVG diagrams for the lecture 12 chapter.

Two of them carry numbers rather than geometry. `diagram_label_cost` replots the
annotation effort per image published by the COCO and Cityscapes papers. And
`diagram_mae_ratio` is *computed*: `block_flops` is the standard per-block
Transformer cost, and evaluating it for MAE's actual ViT-L configuration
reproduces the 3.3x FLOPs reduction and the ~9% decoder-per-token ratio that the
MAE paper reports, so the curve in the figure and the claims in the prose come
out of the same function. The `__main__` block prints both, which is how the
chapter is checked.

The rest are geometry: the two-stage pretext/downstream split, the three
transformation tasks, the split-brain channel cross-prediction, the colour
pointer that turns into a tracker, MAE's asymmetric encoder and decoder, the
InfoNCE lineup, the two places negatives can come from, and DINO's two collapse
modes with the two corrections that oppose them.

Conventions follow figures/11-distributed-training: currentColor for text and
rules, viridis for data, title/desc with namespaced ids, no background rect (the
theme supplies a light plate in both schemes). Yellow is a fill only, never a
line or a label.

Run from the repository root; rewrites all ten files.
"""

import math
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
    circle,
    line,
    mapper,
    poly,
    polyline,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path("figures/12-self-supervised-learning")


# --------------------------------------------------------------------------
# Published annotation effort, in aggregate worker-minutes per image.
#
# COCO: Lin et al., "Microsoft COCO", ECCV 2014, section 4. The paper reports
# ~20k worker hours for category labelling, ~10k for instance spotting, and
# "over 22 worker hours per 1,000 segmentations" for 2.5M instances, which is
# ~55k hours. All three are divided by the dataset's 328k images.
#
# Cityscapes: Cordts et al., CVPR 2016. Fine annotation plus quality control is
# reported as more than 1.5 h per image; the coarse tier was capped at under
# 7 min per image.
# --------------------------------------------------------------------------
COCO_IMAGES = 328_000
LABEL_COST = [
    ("COCO — which categories are present", 20_000 * 60 / COCO_IMAGES, INDIGO),
    ("COCO — where each instance is", 10_000 * 60 / COCO_IMAGES, INDIGO),
    ("COCO — outline of every instance", 55_000 * 60 / COCO_IMAGES, INDIGO),
    ("Cityscapes — coarse polygons", 7.0, PURPLE),
    ("Cityscapes — every pixel, with review", 90.0, PURPLE),
]


def fig_label_cost():
    """Annotation effort per image, from the COCO and Cityscapes papers."""
    w, h = 760, 306
    x0, x1 = 300, 706
    lo, hi = 1.0, 100.0

    def px(v):
        return x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)

    b = text(20, 30, "Human annotation effort per image", size=14, anchor="start")
    b += text(
        20,
        50,
        "aggregate worker-minutes, logarithmic scale",
        size=11.5,
        anchor="start",
        op=0.6,
    )

    for tick in (1, 3, 10, 30, 100):
        gx = px(tick)
        b += line(gx, 68, gx, 250, op=0.16, dash="3 4")
        b += text(gx, 268, f"{tick}", size=11, op=0.6, font=MONO)
    b += text((x0 + x1) / 2, 290, "minutes", size=11.5, op=0.6)

    top, step, bh = 78, 34, 20
    for i, (name, minutes, colour) in enumerate(LABEL_COST):
        y = top + i * step
        b += text(x0 - 14, y + bh - 6, name, size=12, anchor="end")
        b += rect(x0, y, px(minutes) - x0, bh, fill=colour, op=0.82, stroke="none")
        b += text(
            px(minutes) + 8, y + bh - 6, f"{minutes:.1f}", size=11.5, anchor="start", font=MONO
        )

    b += line(x0, 68, x0, 250, op=0.35)
    return wrap(
        w,
        h,
        "lc",
        "Annotation effort per image",
        "Horizontal bars on a logarithmic axis showing aggregate worker-minutes per image for three COCO annotation stages and two Cityscapes annotation tiers, spanning from under two minutes to ninety.",
        b,
    )


def fig_pretext_pipeline():
    """The two-stage pretext then downstream structure, head discarded."""
    w, h = 760, 372
    b = text(20, 28, "Stage 1 — pretext, on unlabelled data", size=13, anchor="start")

    # Top lane: x assigned strictly in flow order.
    lane = 92
    stops = [
        (60, "unlabelled\nimages", None),
        (188, "corrupt or\ntransform", TEAL),
        (316, "encoder", INDIGO),
        (444, "pretext\nhead", PURPLE),
        (572, "manufactured\ntarget", None),
    ]
    bw, bh = 100, 52
    for cx, label, colour in stops:
        if colour:
            b += rect(cx - bw / 2, lane - bh / 2, bw, bh, fill=colour, op=0.16, rx=3)
        else:
            b += rect(cx - bw / 2, lane - bh / 2, bw, bh, fill="none", op=0, sw=0.9, so=0.4, rx=3)
        for k, part in enumerate(label.split("\n")):
            b += text(cx, lane + 4 + (k - (len(label.split("\n")) - 1) / 2) * 14, part, size=12)
    for i in range(len(stops) - 1):
        b += arrow(stops[i][0] + bw / 2 + 4, lane, stops[i + 1][0] - bw / 2 - 4, lane)

    b += arrow(672, lane, 700, lane, op=0.5)
    b += text(706, lane - 6, "loss", size=11.5, anchor="start", op=0.7)
    b += text(706, lane + 10, "here", size=11.5, anchor="start", op=0.7)

    # The pretext head is thrown away.
    b += line(400, lane - 34, 488, lane + 34, stroke=PURPLE, sw=1.6, op=0.45)
    b += line(488, lane - 34, 400, lane + 34, stroke=PURPLE, sw=1.6, op=0.45)
    b += text(444, lane + 52, "discarded", size=11.5, fill=PURPLE, op=0.9)

    b += line(20, 176, 740, 176, op=0.3, dash="6 5")
    b += text(20, 200, "Stage 2 — downstream, on a small labelled set", size=13, anchor="start")

    lane2 = 268
    stops2 = [
        (60, "labelled\nimages", None),
        (188, "same\nencoder", INDIGO),
        (316, "new linear\nhead", GREEN),
        (444, "class\nlabels", None),
    ]
    for cx, label, colour in stops2:
        if colour:
            b += rect(cx - bw / 2, lane2 - bh / 2, bw, bh, fill=colour, op=0.16, rx=3)
        else:
            b += rect(cx - bw / 2, lane2 - bh / 2, bw, bh, fill="none", op=0, sw=0.9, so=0.4, rx=3)
        for k, part in enumerate(label.split("\n")):
            b += text(cx, lane2 + 4 + (k - (len(label.split("\n")) - 1) / 2) * 14, part, size=12)
    for i in range(len(stops2) - 1):
        b += arrow(stops2[i][0] + bw / 2 + 4, lane2, stops2[i + 1][0] - bw / 2 - 4, lane2)

    # The encoder is the only thing that crosses the line.
    b += arrow(316, lane + bh / 2 + 4, 188, lane2 - bh / 2 - 4, stroke=INDIGO, sw=1.6, op=0.85)
    b += text(268, 186, "the only thing that crosses", size=11.5, fill=INDIGO, op=0.95, anchor="start")

    b += text(
        20,
        344,
        "The pretext head exists only to make a gradient; what survives is the encoder.",
        size=11.5,
        anchor="start",
        op=0.65,
    )
    return wrap(
        w,
        h,
        "pp",
        "The pretext and downstream stages",
        "A two-lane diagram: the upper lane runs unlabelled images through a transformation, an encoder and a pretext head to a manufactured target, with the head crossed out; the lower lane runs labelled images through the same encoder and a new linear head, with an arrow marking the encoder as the only shared component.",
        b,
    )


# A small asymmetric glyph in unit-square coordinates. It has no rotational
# symmetry, so a reader can read the rotation off the shape rather than off the
# label — which is the whole point of the left panel.
GLYPH = [(0.22, 0.78), (0.22, 0.24), (0.44, 0.24), (0.44, 0.58), (0.76, 0.58), (0.76, 0.78)]


def _rot(points, quarter):
    """Rotate unit-square points by `quarter` right angles about (0.5, 0.5)."""
    a = quarter * math.pi / 2
    ca, sa = math.cos(a), math.sin(a)
    return [(0.5 + (x - 0.5) * ca - (y - 0.5) * sa, 0.5 + (x - 0.5) * sa + (y - 0.5) * ca) for x, y in points]


def fig_transformations():
    """Rotation, relative patch location and jigsaw as one shared shape."""
    w, h = 760, 344
    b = text(20, 28, "Rotation — four classes", size=13, anchor="start")
    b += text(430, 28, "Patch position and permutation", size=13, anchor="start")

    side = 62
    for q in range(4):
        tx, ty = 24 + q * 86, 54
        b += rect(tx, ty, side, side, fill=INDIGO, op=0.10, sw=0.9, so=0.45, rx=2)
        pts = [(tx + x * side, ty + y * side) for x, y in _rot(GLYPH, q)]
        b += poly(pts, fill=TEAL, op=0.8, stroke="none")
        b += text(tx + side / 2, ty + side + 18, f"{q * 90}°", size=12)
        b += text(tx + side / 2, ty + side + 35, f"y = {q}", size=11.5, font=MONO, op=0.65)

    b += text(
        24,
        192,
        "The label is drawn first; the image is manufactured from it.",
        size=11.5,
        anchor="start",
        op=0.65,
    )
    b += text(
        24,
        210,
        "No local statistic of a photograph reveals which way is up,",
        size=11.5,
        anchor="start",
        op=0.65,
    )
    b += text(
        24,
        228,
        "so the arrangement of the object's parts has to be encoded.",
        size=11.5,
        anchor="start",
        op=0.65,
    )

    b += line(400, 40, 400, 300, op=0.22, dash="4 5")

    # 3x3 grid: centre is the reference, the eight neighbours are the classes.
    gx, gy, cell = 434, 54, 46
    idx = 0
    for r in range(3):
        for c in range(3):
            x, y = gx + c * cell, gy + r * cell
            centre = r == 1 and c == 1
            b += rect(
                x,
                y,
                cell - 4,
                cell - 4,
                fill=YELLOW if centre else INDIGO,
                op=0.55 if centre else 0.12,
                sw=0.9,
                so=0.45,
                rx=2,
            )
            if centre:
                b += text(x + (cell - 4) / 2, y + (cell - 4) / 2 + 4, "ref", size=11)
            else:
                idx += 1
                b += text(x + (cell - 4) / 2, y + (cell - 4) / 2 + 4, str(idx), size=12, op=0.8)
    b += text(gx + 1.5 * cell - 2, gy + 3 * cell + 16, "8-way: where is this patch?", size=11.5, op=0.75)

    # Jigsaw: nine tiles in a shuffled row, predicting an index into a fixed set.
    order = [5, 2, 8, 1, 9, 4, 7, 3, 6]
    jx, jy, tw = 434, 236, 28
    for i, v in enumerate(order):
        b += rect(jx + i * (tw + 4), jy, tw, tw, fill=TEAL, op=0.16, sw=0.9, so=0.4, rx=2)
        b += text(jx + i * (tw + 4) + tw / 2, jy + tw / 2 + 4, str(v), size=11.5, op=0.8)
    b += text(
        jx,
        jy + tw + 20,
        "predict which of a fixed set of permutations was applied;",
        size=11.5,
        anchor="start",
        op=0.7,
    )
    b += text(
        jx,
        jy + tw + 38,
        "the set is chosen far apart in Hamming distance",
        size=11.5,
        anchor="start",
        op=0.7,
    )
    return wrap(
        w,
        h,
        "tr",
        "Three transformation pretext tasks",
        "On the left, one asymmetric glyph shown at four right-angle rotations with the class index below each. On the right, a three-by-three patch grid with the centre marked as reference and the eight neighbours numbered, and beneath it a shuffled row of nine tiles feeding a permutation index.",
        b,
    )


def fig_split_brain():
    """Channel split and cross-prediction, against a plain autoencoder."""
    w, h = 760, 336
    b = text(20, 26, "Autoencoder — the target is visible to the network", size=12.5, anchor="start")

    # Reference row, deliberately small: it is the thing being improved on.
    ry = 58
    b += rect(30, ry, 66, 40, fill=INK, op=0.06, sw=0.9, so=0.35, rx=3)
    b += text(63, ry + 24, "X", size=13, style="italic")
    b += arrow(100, ry + 20, 140, ry + 20)
    b += rect(144, ry, 66, 40, fill=INK, op=0.10, sw=0.9, so=0.35, rx=3)
    b += text(177, ry + 24, "F", size=13, style="italic")
    b += arrow(214, ry + 20, 254, ry + 20)
    b += rect(258, ry, 66, 40, fill=INK, op=0.06, sw=0.9, so=0.35, rx=3)
    b += text(291, ry + 24, "X̂", size=13, style="italic")
    b += text(344, ry + 24, "— nothing rules out copying", size=11.5, anchor="start", op=0.6)

    b += line(20, 122, 740, 122, op=0.25, dash="6 5")
    b += text(20, 148, "Split-brain — each half predicts what it cannot see", size=12.5, anchor="start")

    # Two lanes, both left-to-right, crossing only in what they predict.
    def lane(y, src, dst, fname, colour):
        b2 = rect(60, y, 74, 40, fill=colour, op=0.20, sw=0.9, so=0.4, rx=3)
        b2 += text(97, y + 25, src, size=13, style="italic")
        b2 += arrow(138, y + 20, 190, y + 20)
        b2 += rect(194, y, 74, 40, fill=colour, op=0.36, sw=0.9, so=0.4, rx=3)
        b2 += text(231, y + 25, fname, size=13, style="italic")
        b2 += arrow(272, y + 20, 324, y + 20)
        b2 += rect(328, y, 74, 40, fill=colour, op=0.20, sw=0.9, so=0.4, rx=3)
        b2 += text(365, y + 25, dst, size=13, style="italic")
        return b2

    b += lane(176, "X₁", "X̂₂", "F₁", TEAL)
    b += lane(252, "X₂", "X̂₁", "F₂", PURPLE)

    b += text(97, 168, "lightness L", size=11, op=0.6)
    b += text(97, 316, "colour ab", size=11, op=0.6)

    # Concatenate the two representations.
    b += arrow(240, 218, 452, 226, op=0.6)
    b += arrow(240, 250, 452, 242, op=0.6)
    b += rect(456, 208, 104, 52, fill=INDIGO, op=0.20, sw=0.9, so=0.4, rx=3)
    b += text(508, 228, "concatenate", size=12)
    b += text(508, 246, "F₁ ⊕ F₂", size=12, style="italic")
    b += arrow(564, 234, 600, 234)
    b += text(606, 230, "representation", size=12, anchor="start")
    b += text(606, 248, "of the whole tensor", size=11.5, anchor="start", op=0.65)
    return wrap(
        w,
        h,
        "sb",
        "The split-brain autoencoder",
        "A small reference row showing a plain autoencoder reconstructing its own input, and below it two parallel lanes in which one network predicts the colour channels from lightness and another predicts lightness from colour, their representations concatenated into a single feature.",
        b,
    )


def fig_video_color():
    """Colour copied from a reference frame through a learned pointer."""
    w, h = 760, 358
    b = text(20, 26, "Reference frame, in colour", size=12.5, anchor="start")
    b += text(268, 26, "Target frame, grayscale", size=12.5, anchor="start")

    cell = 30
    ref_x, ref_y = 30, 44
    tgt_x, tgt_y = 268, 44
    # One reference cell is the source of the answer; one target cell asks.
    src_r, src_c = 1, 3
    qry_r, qry_c = 2, 1
    palette = [PURPLE, INDIGO, TEAL, GREEN, YELLOW]

    for r in range(4):
        for c in range(5):
            colour = palette[(r + 2 * c) % 5]
            hit = r == src_r and c == src_c
            b += rect(
                ref_x + c * cell,
                ref_y + r * cell,
                cell - 3,
                cell - 3,
                fill=colour,
                op=0.85 if hit else 0.40,
                sw=1.6 if hit else 0.7,
                so=0.9 if hit else 0.3,
                stroke=INK if hit else "currentColor",
                rx=2,
            )
            ask = r == qry_r and c == qry_c
            b += rect(
                tgt_x + c * cell,
                tgt_y + r * cell,
                cell - 3,
                cell - 3,
                fill=INK,
                op=0.10 + 0.03 * ((r + c) % 3),
                sw=1.6 if ask else 0.7,
                so=0.9 if ask else 0.3,
                rx=2,
            )
    b += text(
        ref_x + 2.5 * cell, ref_y + 4 * cell + 20, "colours cᵢ — values only", size=11.5, op=0.7
    )
    b += text(
        tgt_x + 2.5 * cell,
        tgt_y + 4 * cell + 20,
        "no colour anywhere in the input",
        size=11.5,
        op=0.7,
    )

    # The pointer: target query cell back to the reference cell it matches.
    qx = tgt_x + qry_c * cell + (cell - 3) / 2
    qy = tgt_y + qry_r * cell + (cell - 3) / 2
    sx = ref_x + src_c * cell + (cell - 3) / 2
    sy = ref_y + src_r * cell + (cell - 3) / 2
    b += arrow(qx, qy, sx + 10, sy + 8, stroke=INDIGO, sw=1.8, op=0.9)
    b += text(
        (qx + sx) / 2,
        212,
        "the pointer Aᵢⱼ — a softmax over every reference location",
        size=11.5,
        fill=INDIGO,
        op=0.95,
    )

    # The one path colour can take.
    y = 250
    b += rect(30, y, 130, 46, fill=INDIGO, op=0.16, sw=0.9, so=0.4, rx=3)
    b += text(95, y + 20, "CNN embeddings", size=12)
    b += text(95, y + 36, "fᵢ , fⱼ", size=12, style="italic")
    b += arrow(164, y + 23, 214, y + 23)
    b += rect(218, y, 130, 46, fill=TEAL, op=0.16, sw=0.9, so=0.4, rx=3)
    b += text(283, y + 20, "similarity", size=12)
    b += text(283, y + 36, "softmax → A", size=12, style="italic")
    b += arrow(352, y + 23, 402, y + 23)
    b += rect(406, y, 150, 46, fill=GREEN, op=0.20, sw=0.9, so=0.4, rx=3)
    b += text(481, y + 20, "weighted sum", size=12)
    b += text(481, y + 36, "yⱼ = Σᵢ Aᵢⱼ cᵢ", size=12, style="italic")
    b += arrow(560, y + 23, 610, y + 23)
    b += text(616, y + 20, "predicted", size=12, anchor="start")
    b += text(616, y + 38, "colour", size=12, anchor="start")

    b += text(
        20,
        330,
        "Colour enters only at the last step, so the loss can be reduced only by fixing A.",
        size=11.5,
        anchor="start",
        op=0.65,
    )
    return wrap(
        w,
        h,
        "vc",
        "Colour copied through a learned pointer",
        "A colour reference frame grid beside a grayscale target frame grid, with an arrow from one target cell to the reference cell it matches, and below a four-stage pipeline from CNN embeddings through a softmax similarity to a weighted sum of reference colours.",
        b,
    )


# --------------------------------------------------------------------------
# MAE cost model. A Transformer block on n tokens of width d costs about
# 24*n*d^2 in the QKVO projections and the 4x feed-forward, and 4*n^2*d in the
# two attention matmuls, counting two FLOPs per multiply-accumulate.
#
# Evaluated on MAE's ViT-L configuration this reproduces two published figures:
# the decoder's ~9% cost per token relative to the encoder, and the 3.3x
# reduction in total pre-training FLOPs from keeping mask tokens out of the
# encoder. Both are printed by __main__.
# --------------------------------------------------------------------------
PATCHES = 196  # 224 / 16, squared
ENC_BLOCKS, ENC_WIDTH = 24, 1024  # ViT-L
DEC_BLOCKS, DEC_WIDTH = 8, 512  # MAE default decoder


def block_flops(n, d, blocks):
    """Approximate FLOPs for `blocks` Transformer blocks on n tokens of width d."""
    return blocks * (24 * n * d**2 + 4 * n**2 * d)


def encoder_flops(visible_fraction):
    """Encoder cost when only `visible_fraction` of the patches are kept."""
    return block_flops(round(visible_fraction * PATCHES), ENC_WIDTH, ENC_BLOCKS)


def decoder_flops():
    """Decoder cost, which is independent of the masking ratio."""
    return block_flops(PATCHES, DEC_WIDTH, DEC_BLOCKS)


def total_flops(visible_fraction):
    """Encoder plus decoder, the whole pre-training step."""
    return encoder_flops(visible_fraction) + decoder_flops()


# The two linear-probing accuracies the MAE paper states in its text: 54.6% at
# the low end of the masking sweep and 73.5% at the chosen ratio of 75%. The
# intermediate points of Figure 5 are not transcribed here, so nothing in this
# figure asserts a value the paper does not spell out in words.
PROBE_ANCHORS = [(10, 54.6), (75, 73.5)]


def fig_mae_ratio():
    """Computed pre-training cost against masking ratio, with probe anchors."""
    w, h = 760, 388
    px, py = (108, 640), (74, 264)
    to_px = mapper((0, 90), (0, 1.05), px, py)
    base = total_flops(1.0)

    b = text(20, 28, "Pre-training cost against masking ratio", size=13.5, anchor="start")
    b += text(
        20,
        46,
        "ViT-L encoder, 8-block decoder, 196 patches; cost relative to no masking",
        size=11.5,
        anchor="start",
        op=0.6,
    )

    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = to_px(0, v)[1]
        b += line(px[0], gy, px[1], gy, op=0.14, dash="3 4")
        b += text(px[0] - 10, gy + 4, f"{v:.2f}", size=11, anchor="end", font=MONO, op=0.7)
    for r in (0, 30, 60, 90):
        gx = to_px(r, 0)[0]
        b += text(gx, py[1] + 20, f"{r}%", size=11, font=MONO, op=0.7)
    b += line(px[0], py[0], px[0], py[1], op=0.35)
    b += line(px[0], py[1], px[1], py[1], op=0.35)
    b += text((px[0] + px[1]) / 2, py[1] + 42, "masking ratio", size=12, op=0.75)

    ratios = list(range(0, 91, 5))
    tot = [to_px(r, total_flops(1 - r / 100) / base) for r in ratios]
    enc = [to_px(r, encoder_flops(1 - r / 100) / base) for r in ratios]
    dec_y = to_px(0, decoder_flops() / base)[1]

    b += line(px[0], dec_y, px[1], dec_y, stroke=GREEN, sw=1.6, op=0.9)
    b += polyline(enc, stroke=TEAL, sw=1.8)
    b += polyline(tot, stroke=INDIGO, sw=2.2)

    b += text(
        px[0] + 12,
        dec_y - 9,
        "decoder — flat, always all 196 positions",
        size=11.5,
        fill=GREEN,
        anchor="start",
    )
    b += text(300, to_px(30, total_flops(0.7) / base)[1] - 12, "total", size=12, fill=INDIGO)
    b += text(300, to_px(30, encoder_flops(0.7) / base)[1] + 18, "encoder", size=12, fill=TEAL)

    # The chosen ratio, and the reduction it buys.
    cx = to_px(75, 0)[0]
    b += line(cx, py[0], cx, py[1], stroke=PURPLE, sw=1.4, op=0.7, dash="5 4")
    b += text(cx, py[0] - 28, "75% — the chosen ratio", size=11.5, fill=PURPLE)
    reduction = base / total_flops(0.25)
    b += text(cx, py[0] - 8, f"{reduction:.1f}× cheaper than no masking", size=12, fill=INDIGO)

    # Published accuracies, stated as text rather than plotted on a second axis.
    b += text(
        20, 336, "Linear probing accuracy, from the paper's text:", size=12, anchor="start", op=0.8
    )
    for i, (ratio, acc) in enumerate(PROBE_ANCHORS):
        b += text(
            318 + i * 196,
            336,
            f"{acc}% at {ratio}% masking",
            size=12,
            anchor="start",
            font=MONO,
            fill=PURPLE,
        )
    att_share = 4 * PATCHES**2 * ENC_WIDTH / (24 * PATCHES * ENC_WIDTH**2)
    b += text(
        20,
        366,
        f"The encoder curve bends because attention is quadratic, but only slightly: at 196 tokens "
        f"attention is {att_share * 100:.0f}% of encoder cost.",
        size=11.5,
        anchor="start",
        op=0.65,
    )
    return wrap(
        w,
        h,
        "mr",
        "MAE pre-training cost against masking ratio",
        "A line chart with masking ratio on the horizontal axis and relative pre-training cost on the vertical, showing a bending encoder curve, a flat decoder line and their sum, with the chosen seventy-five percent ratio marked and the resulting cost reduction annotated.",
        b,
    )


def fig_mae():
    """MAE's asymmetric encoder and decoder over a masked patch block."""
    w, h = 760, 336

    b = text(20, 26, "Masked autoencoder", size=13.5, anchor="start")

    # A 4x4 block with every fourth patch kept, so the visible quarter is a
    # property of the index rule rather than a hand-placed set of squares.
    cell, gap = 30, 4
    gx, gy = 28, 60
    visible = [i for i in range(16) if (i * 7) % 16 < 4]
    for i in range(16):
        r, c = divmod(i, 4)
        vis = i in visible
        b += rect(
            gx + c * (cell + gap),
            gy + r * (cell + gap),
            cell,
            cell,
            fill=TEAL if vis else INK,
            op=0.55 if vis else 0.07,
            sw=0.9,
            so=0.5 if vis else 0.2,
            rx=2,
        )
    b += text(gx + 2 * (cell + gap) - 2, gy + 4 * (cell + gap) + 16, "patches", size=12)
    b += text(
        gx + 2 * (cell + gap) - 2, gy + 4 * (cell + gap) + 34, "75% removed", size=11.5, op=0.7
    )

    # Everything below runs strictly left to right, so nothing doubles back.
    mid = gy + 2 * (cell + gap) - gap / 2
    b += arrow(gx + 4 * (cell + gap) + 2, mid, 190, mid)

    ey, eh = mid - 62, 124
    b += rect(194, ey, 128, eh, fill=INDIGO, op=0.22, sw=1.0, so=0.5, rx=3)
    b += text(258, ey + 48, "encoder", size=13)
    b += text(258, ey + 68, "24 blocks, 1024-d", size=11, font=MONO, op=0.75)
    b += text(258, ey + 86, "49 of 196 tokens", size=11, font=MONO, op=0.75)

    b += arrow(326, mid, 366, mid)
    b += rect(370, mid - 26, 100, 52, fill=PURPLE, op=0.22, sw=1.0, so=0.5, rx=3)
    b += text(420, mid - 6, "insert shared", size=11.5)
    b += text(420, mid + 10, "mask token", size=11.5)
    b += text(420, mid + 40, "back to 196 tokens", size=11, op=0.7)

    b += arrow(474, mid, 514, mid)
    dh = 44
    b += rect(518, mid - dh / 2, 118, dh, fill=INDIGO, op=0.22, sw=1.0, so=0.5, rx=3)
    b += text(577, mid - 3, "decoder", size=12.5)
    b += text(577, mid + 14, "8 blocks, 512-d", size=10.5, font=MONO, op=0.75)

    b += arrow(640, mid, 672, mid)
    b += rect(676, mid - 34, 62, 68, fill=GREEN, op=0.18, sw=0.9, so=0.4, rx=3)
    b += text(707, mid - 12, "MSE,", size=11.5)
    b += text(707, mid + 4, "masked", size=11.5)
    b += text(707, mid + 20, "patches", size=11.5)

    b += text(
        20,
        286,
        "The encoder is the tall box on the short sequence; the decoder is the short box on the "
        "full one.",
        size=11.5,
        anchor="start",
        op=0.7,
    )
    b += text(
        20,
        306,
        "No mask token ever reaches the encoder, so it only ever sees real image patches — which "
        "removes",
        size=11.5,
        anchor="start",
        op=0.7,
    )
    b += text(
        20,
        326,
        "the gap between pre-training and deployment as well as most of the cost.",
        size=11.5,
        anchor="start",
        op=0.7,
    )
    return wrap(
        w,
        h,
        "ma",
        "The MAE encoder and decoder",
        "A four-by-four block of patches with three quarters greyed out, feeding rightward into a "
        "tall encoder box, then a step that inserts shared mask tokens to restore the full "
        "sequence length, then a much shorter decoder box and the reconstruction loss.",
        b,
    )

def fig_infonce():
    """The InfoNCE lineup and the softmax mass it redistributes."""
    w, h = 760, 336
    b = text(20, 26, "One anchor, one positive, N − 1 negatives", size=13, anchor="start")

    ax, ay = 130, 130
    b += circle(ax, ay, 13, fill=INDIGO, op=0.9, stroke="none")
    b += text(ax, ay + 34, "anchor", size=12)
    b += text(ax, ay + 50, "f(x)", size=11.5, style="italic", op=0.7)

    pos = (206, 88)
    b += circle(pos[0], pos[1], 11, fill=GREEN, op=0.9, stroke="none")
    b += text(pos[0] + 20, pos[1] - 6, "positive", size=12, anchor="start")
    b += text(pos[0] + 20, pos[1] + 10, "another view of x", size=11, anchor="start", op=0.65)
    b += arrow(pos[0] - 6, pos[1] + 10, ax + 12, ay - 10, stroke=GREEN, sw=1.8, op=0.95)

    negs = [(268, 176), (232, 232), (144, 244), (72, 210), (60, 74)]
    for i, (nx, ny) in enumerate(negs):
        b += circle(nx, ny, 10, fill=PURPLE, op=0.75, stroke="none")
        ux, uy = nx - ax, ny - ay
        d = math.hypot(ux, uy)
        b += arrow(nx, ny, nx + 24 * ux / d, ny + 24 * uy / d, stroke=PURPLE, sw=1.4, op=0.8)
    b += text(176, 302, "negatives — pushed away", size=11.5, fill=PURPLE)

    b += line(330, 44, 330, 300, op=0.22, dash="4 5")

    # The same thing as a softmax over N candidates.
    b += text(360, 26, "which is an N-way classification", size=13, anchor="start")
    bars = [0.62, 0.11, 0.09, 0.07, 0.06, 0.05]
    bx, by, bwid, scale = 372, 250, 46, 250
    for i, p in enumerate(bars):
        x = bx + i * (bwid + 14)
        b += rect(
            x,
            by - p * scale,
            bwid,
            p * scale,
            fill=GREEN if i == 0 else PURPLE,
            op=0.75 if i == 0 else 0.45,
            stroke="none",
        )
        b += text(x + bwid / 2, by + 18, "x⁺" if i == 0 else f"x⁻{i}", size=11.5, font=MONO, op=0.8)
    b += line(bx - 10, by, bx + 6 * (bwid + 14), by, op=0.35)
    b += text(360, 286, "softmax mass; the loss is −log of the first bar", size=11.5, anchor="start", op=0.7)
    b += text(
        360,
        312,
        "A negative already far away contributes almost nothing to the gradient.",
        size=11.5,
        anchor="start",
        op=0.65,
    )
    return wrap(
        w,
        h,
        "in",
        "The InfoNCE objective",
        "On the left, an anchor point in embedding space with a positive being pulled toward it and five negatives being pushed away; on the right, the same situation as a bar chart of softmax probability mass with the positive holding most of it.",
        b,
    )


def fig_simclr_moco():
    """Where the negatives come from: the batch, or a queue."""
    w, h = 760, 366
    b = text(20, 26, "SimCLR — negatives are the batch", size=13, anchor="start")
    b += text(404, 26, "MoCo — negatives are a queue", size=13, anchor="start")
    b += line(388, 44, 388, 336, op=0.22, dash="4 5")

    # Left panel, strict flow order: images, views, encoder, negatives.
    b += rect(24, 62, 78, 44, fill=INK, op=0.06, sw=0.9, so=0.35, rx=3)
    b += text(63, 82, "batch of", size=11.5)
    b += text(63, 98, "N images", size=11.5)
    b += arrow(106, 84, 138, 84)
    b += rect(142, 54, 78, 30, fill=TEAL, op=0.20, sw=0.9, so=0.4, rx=3)
    b += text(181, 74, "view A", size=11.5)
    b += rect(142, 92, 78, 30, fill=TEAL, op=0.20, sw=0.9, so=0.4, rx=3)
    b += text(181, 112, "view B", size=11.5)
    b += arrow(224, 69, 256, 80)
    b += arrow(224, 107, 256, 92)
    b += rect(260, 62, 92, 44, fill=INDIGO, op=0.24, sw=1.0, so=0.5, rx=3)
    b += text(306, 82, "one encoder", size=11.5)
    b += text(306, 98, "gradient", size=11, font=MONO, op=0.7)
    b += arrow(306, 110, 306, 148)
    b += rect(232, 152, 148, 44, fill=PURPLE, op=0.20, sw=0.9, so=0.4, rx=3)
    b += text(306, 172, "2N − 2 negatives", size=12)
    b += text(306, 188, "from this batch only", size=11, op=0.7)
    b += text(24, 232, "Negative count and batch size are", size=11.5, anchor="start", op=0.7)
    b += text(24, 250, "the same number, so both are bounded", size=11.5, anchor="start", op=0.7)
    b += text(24, 268, "by memory. The best runs used", size=11.5, anchor="start", op=0.7)
    b += text(24, 286, "batches of 4096 and 8192.", size=11.5, anchor="start", op=0.7)

    # Right panel: query path carries gradient, key path does not.
    b += rect(408, 62, 74, 44, fill=INK, op=0.06, sw=0.9, so=0.35, rx=3)
    b += text(445, 82, "one image,", size=11.5)
    b += text(445, 98, "two views", size=11.5)
    b += arrow(486, 74, 530, 68)
    b += arrow(486, 94, 530, 122)
    b += rect(534, 48, 82, 40, fill=INDIGO, op=0.24, sw=1.0, so=0.5, rx=3)
    b += text(575, 65, "f_q  query", size=11.5, font=MONO)
    b += text(575, 81, "gradient", size=10.5, op=0.75)
    b += rect(534, 102, 82, 40, fill=TEAL, op=0.20, sw=0.9, so=0.4, rx=3, stroke=TEAL)
    b += text(575, 119, "f_k  key", size=11.5, font=MONO)
    b += text(575, 135, "no gradient", size=10.5, op=0.75)

    # The momentum update runs from the query encoder back to the key encoder.
    b += arrow(624, 74, 624, 116, stroke=INDIGO, sw=1.4, op=0.85)
    b += text(632, 98, "EMA", size=11, anchor="start", fill=INDIGO)

    # The queue starts under the encoder that fills it, so nothing runs backwards.
    qx, qy, qc = 534, 186, 6
    b += text(
        qx + qc * 32 - 4,
        qy - 14,
        "FIFO queue of keys — K = 65,536",
        size=11.5,
        anchor="end",
        op=0.8,
    )
    b += arrow(575, 148, qx + 14, qy - 2, dash="4 4", op=0.7)
    for i in range(qc):
        b += rect(
            qx + i * 32,
            qy,
            28,
            30,
            fill=PURPLE,
            op=0.20 + 0.05 * (qc - i) / qc,
            sw=0.8,
            so=0.35,
            rx=2,
        )
    b += text(qx + qc * 32 - 4, qy + 48, "evict →", size=11, anchor="end", op=0.65)
    b += arrow(qx + 3 * 32 + 14, qy + 34, qx + 3 * 32 + 14, qy + 60, op=0.6)
    b += rect(qx - 90, qy + 64, 300, 40, fill=PURPLE, op=0.20, sw=0.9, so=0.4, rx=3)
    b += text(qx + 60, qy + 82, "K negatives, from many past batches", size=11.5)
    b += text(qx + 60, qy + 98, "batch size 256", size=11, font=MONO, op=0.75)

    b += text(
        408,
        326,
        "The queue holds keys from past encoders,",
        size=11.5,
        anchor="start",
        op=0.7,
    )
    b += text(
        408,
        344,
        "which is why f_k has to move slowly.",
        size=11.5,
        anchor="start",
        op=0.7,
    )
    return wrap(
        w,
        h,
        "sm",
        "Two sources of negative samples",
        "Two panels side by side: on the left SimCLR draws its negatives from the current batch through a single encoder, and on the right MoCo maintains a first-in-first-out queue of keys produced by a separate momentum-updated encoder whose path carries no gradient.",
        b,
    )


def fig_dino():
    """Student, teacher, and the two corrections that oppose the two collapses."""
    w, h = 760, 462
    b = text(20, 26, "Student and teacher, with no negatives anywhere", size=13, anchor="start")

    b += rect(30, 50, 84, 40, fill=INK, op=0.06, sw=0.9, so=0.35, rx=3)
    b += text(72, 68, "one image,", size=11.5)
    b += text(72, 84, "two views", size=11.5)
    b += arrow(118, 60, 154, 50)
    b += arrow(118, 80, 154, 110)

    b += rect(158, 30, 106, 40, fill=INDIGO, op=0.24, sw=1.0, so=0.5, rx=3)
    b += text(211, 47, "student gθs", size=11.5)
    b += text(211, 63, "gradient", size=10.5, op=0.75)
    b += rect(158, 90, 106, 40, fill=TEAL, op=0.20, sw=0.9, so=0.4, rx=3)
    b += text(211, 107, "teacher gθt", size=11.5)
    b += text(211, 123, "stop-gradient", size=10.5, op=0.75)
    b += arrow(272, 62, 272, 100, stroke=INDIGO, sw=1.4, op=0.85)
    b += text(280, 84, "EMA", size=11, anchor="start", fill=INDIGO)

    b += arrow(288, 110, 306, 110)
    b += rect(310, 90, 108, 40, fill=PURPLE, op=0.24, sw=0.9, so=0.45, rx=3)
    b += text(364, 107, "centre, then", size=11.5)
    b += text(364, 123, "sharpen", size=11.5)
    b += arrow(422, 110, 468, 92)
    b += arrow(288, 50, 468, 70)
    b += rect(472, 56, 118, 44, fill=GREEN, op=0.20, sw=0.9, so=0.4, rx=3)
    b += text(531, 76, "cross entropy", size=11.5)
    b += text(531, 92, "student ← teacher", size=11)

    b += line(20, 156, 740, 156, op=0.25, dash="6 5")
    b += text(20, 182, "The two collapses, and the two corrections that oppose them", size=13, anchor="start")

    # Three output histograms, ordered left to right as the corrections push.
    def hist(x0, y0, heights, colour, label, sub):
        out = ""
        bwid, gap, scale = 14, 4, 62
        for i, ht in enumerate(heights):
            out += rect(
                x0 + i * (bwid + gap),
                y0 - ht * scale,
                bwid,
                ht * scale,
                fill=colour,
                op=0.7,
                stroke="none",
            )
        span = len(heights) * (bwid + gap) - gap
        out += line(x0 - 6, y0, x0 + span + 6, y0, op=0.35)
        out += text(x0 + span / 2, y0 + 22, label, size=12)
        out += text(x0 + span / 2, y0 + 40, sub, size=11, op=0.65)
        return out

    spike = [0.05, 0.03, 1.0, 0.04, 0.03, 0.05, 0.03]
    flat = [0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34]
    good = [0.22, 0.62, 0.14, 0.86, 0.18, 0.40, 0.12]

    b += hist(60, 330, spike, PURPLE, "one dimension wins", "collapse")
    b += hist(316, 330, good, GREEN, "structured output", "both together")
    b += hist(572, 330, flat, PURPLE, "uniform output", "collapse")

    # Each correction alone drives the output all the way into the *other*
    # collapse, so both arrows span the full width rather than stopping in the
    # middle. The middle histogram is where the two pushes balance.
    b += arrow(126, 216, 638, 216, stroke=INDIGO, sw=1.6, op=0.9)
    b += text(382, 208, "centring alone → uniform", size=11.5, fill=INDIGO)
    b += arrow(638, 396, 126, 396, stroke=TEAL, sw=1.6, op=0.9)
    b += text(382, 414, "sharpening alone → one dimension", size=11.5, fill=TEAL)

    b += text(
        20,
        444,
        "Neither correction is stable on its own; the structured output in the middle is where the "
        "two pushes balance.",
        size=11.5,
        anchor="start",
        op=0.65,
    )
    return wrap(
        w,
        h,
        "dn",
        "DINO's self-distillation and collapse avoidance",
        "Above, one image producing two views for a student and a momentum teacher whose output is centred and sharpened before a cross-entropy loss; below, three output histograms showing collapse to a single dimension, the structured output that is wanted, and collapse to uniform, with arrows marking which correction pushes away from which collapse.",
        b,
    )


FIGURES = {
    "diagram_label_cost.svg": fig_label_cost,
    "diagram_pretext_pipeline.svg": fig_pretext_pipeline,
    "diagram_transformations.svg": fig_transformations,
    "diagram_split_brain.svg": fig_split_brain,
    "diagram_video_color.svg": fig_video_color,
    "diagram_mae.svg": fig_mae,
    "diagram_mae_ratio.svg": fig_mae_ratio,
    "diagram_infonce.svg": fig_infonce,
    "diagram_simclr_moco.svg": fig_simclr_moco,
    "diagram_dino.svg": fig_dino,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print(f"wrote {OUT / name}")

    print()
    print("Numbers the chapter quotes, evaluated from the generator:")

    per_token_enc = block_flops(1, ENC_WIDTH, ENC_BLOCKS)
    per_token_dec = block_flops(1, DEC_WIDTH, DEC_BLOCKS)
    print(
        f"  decoder cost per token vs encoder      "
        f"{per_token_dec / per_token_enc * 100:.1f}%   (paper: 9%, 'under 10%')"
    )

    with_mask_tokens = block_flops(PATCHES, ENC_WIDTH, ENC_BLOCKS) + decoder_flops()
    print(f"  FLOPs reduction from omitting [M]      {with_mask_tokens / total_flops(0.25):.2f}x  (paper: 3.3x)")

    lin = 24 * PATCHES * ENC_WIDTH**2 * ENC_BLOCKS
    lin_q = 24 * round(0.25 * PATCHES) * ENC_WIDTH**2 * ENC_BLOCKS
    att = 4 * PATCHES**2 * ENC_WIDTH * ENC_BLOCKS
    att_q = 4 * round(0.25 * PATCHES) ** 2 * ENC_WIDTH * ENC_BLOCKS
    print(f"  encoder projections, 75% masked        {lin / lin_q:.1f}x cheaper")
    print(f"  encoder attention,   75% masked        {att / att_q:.1f}x cheaper")

    print(f"  InfoNCE ceiling at N = 256             {math.log(256):.1f} nats")

    print()
    print("Annotation effort per image (aggregate worker-minutes):")
    for name, minutes, _ in LABEL_COST:
        print(f"  {name:42s} {minutes:6.1f}")
    span = LABEL_COST[-1][1] / LABEL_COST[0][1]
    print(f"  span, first bar to last                {span:.0f}x")
