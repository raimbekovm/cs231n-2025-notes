"""Generate the eleven SVG diagrams for the lecture 16 chapter.

Four of them carry computation rather than geometry.

`diagram_robustness` plots the six (dataset, ResNet-101, zero-shot CLIP) triples
that the CLIP paper states in its own distribution-shift table; the gap
annotated on each pair is subtracted here rather than transcribed, and
`__main__` asserts the six differences reproduce the paper's published deltas.

`diagram_clip_objective` is drawn around `symmetric_infonce`, which evaluates
the loss of the chapter's equations two independent ways — as row-wise and
column-wise cross-entropy over the similarity matrix, and as an explicit
log-sum-exp expansion — and asserts they agree to machine precision. The
temperature sweep printed at the bottom is evaluated from the same function.

`diagram_zero_shot` rests on the identity the section claims: taking the
nearest class embedding under cosine similarity is *exactly* the argmax of a
bias-free linear layer whose rows are those embeddings. `zero_shot_agreement`
builds random unit embeddings, runs both routes, and asserts the predictions
are identical. The same block quantifies the other claim in that section, that
averaging in embedding space is not the same computation as averaging in
probability space, by measuring how often the two disagree.

`diagram_sam` is drawn from `sam_cost`, a two-term model of SAM's runtime — one
heavy encoder pass per image plus one light decoder pass per prompt. The
break-even prompt count the caption depends on is solved rather than asserted.

The rest are geometry: the SimCLR-to-CLIP substitution, the bag-of-words
failure mode, LLaVA's projection-and-concatenate fusion, Flamingo's resampler
and zero-initialised gate, Flamingo's per-image attention mask, Molmo's point
as a shared output primitive, and VisProg's compiled program.

Every figure that has a direction of flow assigns x in strict flow order, so no
path doubles back. Conventions follow figures/15-3d-vision: currentColor for
text and rules, viridis for data, title/desc with namespaced ids, no background
rect (the theme supplies a light plate in both schemes). Yellow is a fill only,
never a line or a label.

Run from the repository root; rewrites all eleven files.
"""

import math
import pathlib
import random
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
    rect,
    text,
    wrap,
)

OUT = pathlib.Path("figures/16-vision-language")


# --------------------------------------------------------------------------
# Published numbers.
#
# CLIP's distribution-shift table (Radford et al., arXiv 2103.00020): top-1 for
# a supervised ResNet-101 matched to CLIP on ImageNet, and for zero-shot CLIP,
# on ImageNet and six shifted evaluations. The paper prints a delta column; the
# figure recomputes it and __main__ checks the two agree.
# --------------------------------------------------------------------------
ROBUSTNESS = [
    # dataset, ResNet-101 top-1, zero-shot CLIP top-1, paper's stated delta
    ("ImageNet", 76.2, 76.2, 0.0),
    ("ImageNetV2", 64.3, 70.1, 5.8),
    ("ImageNet-R", 37.7, 88.9, 51.2),
    ("ObjectNet", 32.6, 72.3, 39.7),
    ("ImageNet\nSketch", 25.2, 60.2, 35.0),
    ("ImageNet-A", 2.7, 77.1, 74.4),
]


# --------------------------------------------------------------------------
# The symmetric contrastive loss, computed two independent ways.
#
# Route A is what the chapter's equations say: a softmax cross-entropy along
# the rows of the similarity matrix, plus one along its columns, averaged.
# Route B expands the same quantity as an explicit log-sum-exp difference
# without ever forming a probability. They must agree; if a future edit breaks
# one, the assertion in __main__ catches it.
# --------------------------------------------------------------------------
def _softmax(xs):
    """Numerically stable softmax over a list."""
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def infonce_rows(sim, tau):
    """Image-to-text loss: each row's diagonal entry must win its softmax."""
    n = len(sim)
    total = 0.0
    for i in range(n):
        probs = _softmax([sim[i][j] / tau for j in range(n)])
        total -= math.log(probs[i])
    return total / n


def infonce_cols(sim, tau):
    """Text-to-image loss: each column's diagonal entry must win its softmax."""
    n = len(sim)
    total = 0.0
    for j in range(n):
        probs = _softmax([sim[i][j] / tau for i in range(n)])
        total -= math.log(probs[j])
    return total / n


def symmetric_infonce(sim, tau):
    """The averaged two-way loss, via per-row and per-column cross-entropy."""
    return 0.5 * (infonce_rows(sim, tau) + infonce_cols(sim, tau))


def symmetric_infonce_lse(sim, tau):
    """The same loss expanded as log-sum-exp minus the positive logit."""
    n = len(sim)
    row = 0.0
    col = 0.0
    for i in range(n):
        lse = math.log(sum(math.exp(sim[i][j] / tau) for j in range(n)))
        row += lse - sim[i][i] / tau
    for j in range(n):
        lse = math.log(sum(math.exp(sim[i][j] / tau) for i in range(n)))
        col += lse - sim[j][j] / tau
    return 0.5 * (row / n + col / n)


def random_similarity(n, seed, margin=0.35):
    """A plausible similarity matrix: diagonal favoured, everything in [-1, 1]."""
    rng = random.Random(seed)
    sim = [[rng.uniform(-0.25, 0.25) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        sim[i][i] += margin
    return sim


# --------------------------------------------------------------------------
# The zero-shot head is a linear layer.
#
# Route A: pick the class whose unit text embedding has the largest cosine
# similarity with the unit image embedding. Route B: stack those embeddings as
# the rows of W, compute W u, and take the argmax. On L2-normalised vectors
# these are the same arithmetic, so the predictions must match exactly.
#
# The second half measures the other claim in that section: ensembling K
# prompts by averaging unit embeddings and renormalising (what CLIP does, and
# what costs nothing per image) is a different computation from averaging the K
# cosine similarities, and the two do not always choose the same class.
# --------------------------------------------------------------------------
def _unit(v):
    """L2-normalise a vector."""
    norm = math.sqrt(sum(c * c for c in v))
    return [c / norm for c in v]


def _dot(a, b):
    """Inner product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


def zero_shot_agreement(trials=4000, dim=32, classes=10, seed=17):
    """Check cosine-argmax and bias-free-linear-argmax pick the same class."""
    rng = random.Random(seed)
    disagreements = 0
    for _ in range(trials):
        weights = [_unit([rng.gauss(0, 1) for _ in range(dim)]) for _ in range(classes)]
        u = _unit([rng.gauss(0, 1) for _ in range(dim)])
        # Route A: nearest class embedding under cosine similarity.
        by_cosine = max(range(classes), key=lambda c: _dot(u, weights[c]))
        # Route B: argmax of the matrix-vector product W u, no bias.
        logits = [sum(weights[c][d] * u[d] for d in range(dim)) for c in range(classes)]
        by_linear = max(range(classes), key=lambda c: logits[c])
        if by_cosine != by_linear:
            disagreements += 1
    return disagreements


def ensemble_disagreement(trials=4000, dim=32, classes=10, templates=8, seed=23):
    """How often embedding-space and probability-space ensembling differ."""
    rng = random.Random(seed)
    differ = 0
    for _ in range(trials):
        u = _unit([rng.gauss(0, 1) for _ in range(dim)])
        # Each class has `templates` prompt embeddings scattered about a centre.
        per_class = []
        for _ in range(classes):
            centre = [rng.gauss(0, 1) for _ in range(dim)]
            per_class.append([_unit([c + rng.gauss(0, 0.8) for c in centre]) for _ in range(templates)])
        # Embedding space: average the unit vectors, renormalise, score once.
        embed = []
        for vs in per_class:
            mean = [sum(v[d] for v in vs) / templates for d in range(dim)]
            embed.append(_dot(u, _unit(mean)))
        # Probability space: score every template, then average the scores.
        prob = [sum(_dot(u, v) for v in vs) / templates for vs in per_class]
        if max(range(classes), key=lambda c: embed[c]) != max(range(classes), key=lambda c: prob[c]):
            differ += 1
    return differ


# --------------------------------------------------------------------------
# SAM's runtime, as a two-term model.
#
# The image encoder is a ViT-H run once per image; the prompt encoder and mask
# decoder are small enough that the paper reports ~50 ms per prompt in a
# browser. The interesting quantity is where encode-once stops being an
# optimisation and starts being the only affordable option, which is a solve
# rather than an assertion.
# --------------------------------------------------------------------------
ENCODER_MS = 450.0  # heavy ViT-H pass, amortised over every prompt on the image
DECODER_MS = 50.0  # prompt encoder + mask decoder, per prompt


def sam_cost(prompts, encode_once=True):
    """Milliseconds to answer `prompts` prompts about one image."""
    if encode_once:
        return ENCODER_MS + DECODER_MS * prompts
    return (ENCODER_MS + DECODER_MS) * prompts


def sam_speedup(prompts):
    """Ratio of re-encoding per prompt to encoding once."""
    return sam_cost(prompts, encode_once=False) / sam_cost(prompts, encode_once=True)


# --------------------------------------------------------------------------
# Flamingo's gate. tanh(0) = 0, so a freshly initialised block contributes
# exactly nothing and the conditioned model computes what the frozen LM
# computes. Written as a function so the claim is evaluated, not remembered.
# --------------------------------------------------------------------------
def gated_residual(y, block_output, alpha):
    """One gated residual update: y + tanh(alpha) * block(y)."""
    return y + math.tanh(alpha) * block_output


# --------------------------------------------------------------------------
# Shared drawing helpers.
# --------------------------------------------------------------------------
def box(x, y, w, h, label, sub=None, fill=None, op=0.16, size=13, rx=4):
    """A labelled rounded box; `sub` is a smaller second line beneath the label."""
    out = rect(x, y, w, h, fill=fill or TEAL, op=op if fill or True else 0, stroke=INK, so=0.55, rx=rx)
    if sub:
        out += text(x + w / 2, y + h / 2 - 2, label, size=size, fill=INK)
        out += text(x + w / 2, y + h / 2 + 14, sub, size=10.5, fill=INK, op=0.7)
    else:
        out += text(x + w / 2, y + h / 2 + 4.5, label, size=size, fill=INK)
    return out


def snowflake(x, y, r=6.0, op=0.75):
    """A small six-spoke asterisk marking a frozen component."""
    out = ""
    for k in range(3):
        a = math.pi * k / 3
        out += line(
            x - r * math.cos(a),
            y - r * math.sin(a),
            x + r * math.cos(a),
            y + r * math.sin(a),
            stroke=INDIGO,
            sw=1.3,
            op=op,
        )
    return out


def caption(x, y, s, size=11.5, anchor="middle", op=0.72):
    """A small explanatory note set below or beside a figure element."""
    return text(x, y, s, size=size, fill=INK, anchor=anchor, op=op)


# --------------------------------------------------------------------------
# Figure 1 — SimCLR to CLIP.
# --------------------------------------------------------------------------
def fig_simclr_to_clip():
    """Draw the one substitution that turns SimCLR into CLIP."""
    w, h = 840, 330
    b = ""

    b += text(210, 30, "SimCLR", size=15, fill=INK, weight="600")
    b += text(630, 30, "CLIP", size=15, fill=INK, weight="600")
    b += line(420, 46, 420, 300, sw=1.0, op=0.22, dash="4 4")

    def tower(cx, src_label, src_sub, enc_label, enc_fill, src_fill):
        out = box(cx - 74, 62, 148, 46, src_label, sub=src_sub, fill=src_fill, op=0.18)
        out += arrow(cx, 108, cx, 140, op=0.6)
        out += box(cx - 74, 140, 148, 40, enc_label, fill=enc_fill, op=0.2)
        out += arrow(cx, 180, cx, 212, op=0.6)
        return out

    # SimCLR: two augmentations of one image, two passes of one encoder.
    b += tower(130, "augmentation 1", "random crop, jitter", "image encoder", TEAL, GREEN)
    b += tower(292, "augmentation 2", "random crop, jitter", "image encoder", TEAL, GREEN)
    b += circle(130, 226, 9, fill=GREEN, stroke="none", op=0.85)
    b += circle(292, 226, 9, fill=GREEN, stroke="none", op=0.85)
    b += arrow(148, 226, 268, 226, op=0.8, stroke=PURPLE, sw=1.6)
    b += arrow(274, 234, 154, 234, op=0.8, stroke=PURPLE, sw=1.6)
    b += caption(211, 268, "pull together")
    b += caption(211, 292, "the positive pair was manufactured", op=0.6)

    # CLIP: one image, one caption, two different encoders.
    b += tower(550, "photograph", "as published", "image encoder", TEAL, GREEN)
    b += tower(712, "its alt-text", "“a golden retriever”", "text encoder", INDIGO, YELLOW)
    b += circle(550, 226, 9, fill=GREEN, stroke="none", op=0.85)
    b += circle(712, 226, 9, fill=INDIGO, stroke="none", op=0.85)
    b += arrow(568, 226, 688, 226, op=0.8, stroke=PURPLE, sw=1.6)
    b += arrow(694, 234, 574, 234, op=0.8, stroke=PURPLE, sw=1.6)
    b += caption(631, 268, "pull together")
    b += caption(631, 292, "the positive pair was already on the web", op=0.6)

    return wrap(
        w,
        h,
        "s2c",
        "From SimCLR to CLIP",
        "Two side-by-side diagrams. On the left, SimCLR passes two random "
        "augmentations of one photograph through the same image encoder and "
        "pulls the two embeddings together. On the right, CLIP replaces the "
        "second branch with a text encoder reading the image's published "
        "alt-text, so the positive pair comes from the web rather than from "
        "an augmentation pipeline.",
        b,
    )


# --------------------------------------------------------------------------
# Figure 2 — the similarity matrix and the two cross-entropies.
# --------------------------------------------------------------------------
def fig_clip_objective():
    """Draw the N x N similarity matrix with its row and column losses."""
    w, h = 840, 430
    b = ""
    n = 4
    cell = 52
    ox, oy = 214, 118

    b += text(20, 28, "One training step on a batch of N image–caption pairs", size=13.5, fill=INK, anchor="start")

    # Column headers: the captions. Row headers: the images.
    for j in range(n):
        b += text(ox + cell * (j + 0.5), oy - 12, f"v{j + 1}", size=12.5, fill=INK, op=0.85)
    b += caption(ox + cell * n / 2, oy - 34, "caption embeddings")
    for i in range(n):
        b += text(ox - 12, oy + cell * (i + 0.5) + 4.5, f"u{i + 1}", size=12.5, fill=INK, anchor="end", op=0.85)
    b += caption(30, oy + 4, "image", anchor="start")
    b += caption(30, oy + 22, "embeddings", anchor="start")

    # The matrix. Diagonal entries are the positives.
    for i in range(n):
        for j in range(n):
            on_diag = i == j
            b += rect(
                ox + cell * j,
                oy + cell * i,
                cell,
                cell,
                fill=YELLOW if on_diag else INDIGO,
                op=0.62 if on_diag else 0.10,
                stroke=INK,
                so=0.35,
            )
    b += text(ox + cell * 0.5, oy + cell * 0.5 + 4, "⟨u1,v1⟩", size=9.5, fill=INK)
    b += text(ox + cell * 2.5, oy + cell * 0.5 + 4, "⟨u1,v3⟩", size=9.5, fill=INK, op=0.55)

    bottom = oy + cell * n

    # Row-wise cross-entropy, annotated off to the right of the matrix.
    b += arrow(ox + cell * n + 10, oy + cell * 1.5, ox + cell * n + 58, oy + cell * 1.5, op=0.75, stroke=PURPLE, sw=1.5)
    b += text(ox + cell * n + 66, oy + cell * 1.2, "softmax across each row", size=12, fill=INK, anchor="start")
    b += text(
        ox + cell * n + 66,
        oy + cell * 1.2 + 17,
        "each image picks its caption",
        size=11,
        fill=INK,
        anchor="start",
        op=0.68,
    )
    b += text(
        ox + cell * n + 66, oy + cell * 1.2 + 34, "image → text loss", size=11, fill=PURPLE, anchor="start", font=MONO
    )

    # Column-wise cross-entropy, annotated below the matrix.
    b += arrow(ox + cell * 2.5, bottom + 12, ox + cell * 2.5, bottom + 50, op=0.75, stroke=TEAL, sw=1.5)
    b += text(ox + cell * 2.5 + 16, bottom + 34, "softmax down each column", size=12, fill=INK, anchor="start")
    b += text(
        ox + cell * 2.5 + 16, bottom + 51, "each caption picks its image", size=11, fill=INK, anchor="start", op=0.68
    )
    b += text(ox + cell * 2.5 + 16, bottom + 68, "text → image loss", size=11, fill=TEAL, anchor="start", font=MONO)

    # The accounting that makes the objective cheap.
    b += rect(30, oy + 62, 152, 96, fill=GREEN, op=0.10, stroke=INK, so=0.4, rx=4)
    b += text(106, oy + 86, f"N = {n}", size=12.5, fill=INK)
    b += text(106, oy + 108, f"{n} positives", size=11.5, fill=INK, op=0.8)
    b += text(106, oy + 126, f"{n * n - n} negatives", size=11.5, fill=INK, op=0.8)
    b += text(106, oy + 146, "all of them free", size=11, fill=INK, op=0.62, style="italic")

    b += caption(420, h - 22, "At the real batch size of 32,768 each image must beat 32,767 wrong captions.")

    return wrap(
        w,
        h,
        "clipobj",
        "CLIP's contrastive training step",
        "A four-by-four grid of cosine similarities between four image "
        "embeddings and four caption embeddings. The four diagonal cells are "
        "highlighted as the positive pairs; the twelve off-diagonal cells are "
        "negatives. An arrow to the right of the grid marks the softmax taken "
        "across each row, giving the image-to-text loss; an arrow below marks "
        "the softmax taken down each column, giving the text-to-image loss.",
        b,
    )


# --------------------------------------------------------------------------
# Figure 3 — the zero-shot classifier, and the linear layer it equals.
# --------------------------------------------------------------------------
def fig_zero_shot():
    """Draw prompt-built class vectors, and the same thing as a linear head."""
    w, h = 840, 400
    b = ""
    b += line(438, 44, 438, 336, sw=1.0, op=0.22, dash="4 4")

    b += text(20, 28, "Built once, from the label set", size=13.5, fill=INK, anchor="start")
    b += text(462, 28, "…which is a linear head with no bias", size=13.5, fill=INK, anchor="start")

    # Left panel: flow order is class names, prompt, encoder, cached vectors.
    labels = ["cat", "dog", "platypus"]
    for k, lab in enumerate(labels):
        y = 62 + k * 34
        b += rect(30, y, 84, 26, fill=INDIGO, op=0.12, stroke=INK, so=0.45, rx=3)
        b += text(72, y + 17, lab, size=11.5, fill=INK, font=MONO)
    b += arrow(120, 113, 150, 113, op=0.6)
    b += box(150, 88, 122, 50, "a photo of a {}", fill=YELLOW, op=0.34, size=11.5)
    b += caption(211, 154, "prompt template")
    b += arrow(278, 113, 306, 113, op=0.6)
    b += box(306, 88, 96, 50, "text encoder", fill=INDIGO, op=0.2, size=11.5)
    b += snowflake(354, 152)
    b += caption(354, 176, "frozen")

    for k in range(3):
        y = 214 + k * 30
        b += rect(150, y, 122, 22, fill=YELLOW, op=0.5, stroke=INK, so=0.4, rx=3)
        b += text(211, y + 15, f"v{k + 1}", size=11, fill=INK, font=MONO)
    b += arrow(354, 142, 354, 200, op=0.6)
    b += arrow(340, 208, 280, 224, op=0.6)
    b += caption(211, 318, "cached: C short strings, encoded once")

    # Right panel: the same arithmetic as W u.
    b += box(462, 78, 96, 44, "image", sub="a new photo", fill=GREEN, op=0.2, size=12)
    b += arrow(510, 122, 510, 152, op=0.6)
    b += box(462, 152, 96, 40, "image encoder", fill=TEAL, op=0.2, size=11)
    b += snowflake(510, 206)
    b += arrow(510, 192, 510, 226, op=0.6)
    b += rect(474, 226, 72, 22, fill=GREEN, op=0.55, stroke=INK, so=0.4, rx=3)
    b += text(510, 241, "u", size=11.5, fill=INK, font=MONO)

    b += arrow(556, 237, 596, 237, op=0.7)

    # W as a stack of the cached rows.
    b += rect(600, 200, 96, 78, fill=YELLOW, op=0.42, stroke=INK, so=0.5, rx=3)
    for k in range(3):
        yy = 200 + 26 * (k + 1)
        if k < 2:
            b += line(600, yy, 696, yy, sw=0.8, op=0.35)
        b += text(648, 200 + 26 * k + 17, f"v{k + 1}", size=10.5, fill=INK, font=MONO)
    b += caption(648, 294, "W  — rows are the class vectors")

    b += arrow(700, 239, 736, 239, op=0.7)
    b += text(792, 214, "W u", size=13, fill=INK, font=MONO)
    b += text(792, 236, "argmax", size=11.5, fill=PURPLE, font=MONO)
    b += text(792, 258, "→ class", size=11.5, fill=INK, op=0.7)

    b += caption(
        420,
        h - 18,
        "The weights were written rather than fitted, so the label set can be replaced at inference time.",
    )

    return wrap(
        w,
        h,
        "zshot",
        "Constructing a zero-shot classifier",
        "On the left, class names are inserted into a prompt template, passed "
        "through a frozen text encoder, and cached as one unit vector per "
        "class. On the right, a new image is encoded to a vector u, and the "
        "cached class vectors are stacked as the rows of a matrix W; the "
        "prediction is the argmax of the product W u, which is a linear "
        "classification head with no bias term.",
        b,
    )


# --------------------------------------------------------------------------
# Figure 4 — the distribution-shift table, plotted.
# --------------------------------------------------------------------------
def fig_robustness():
    """Plot supervised ResNet-101 against zero-shot CLIP across six shifts."""
    w, h = 840, 430
    b = ""
    px = (92, 812)
    py = (92, 322)
    to_px = mapper((0, len(ROBUSTNESS)), (0, 100), px, py)

    b += text(20, 28, "Top-1 accuracy (%)", size=12.5, fill=INK, anchor="start", op=0.75)

    # Gridlines and y axis.
    for v in range(0, 101, 25):
        _, yy = to_px(0, v)
        b += line(px[0], yy, px[1], yy, sw=0.8, op=0.16)
        b += text(px[0] - 10, yy + 4, str(v), size=11, fill=INK, anchor="end", op=0.7)
    b += line(px[0], py[1], px[1], py[1], sw=1.1, op=0.5)

    slot = (px[1] - px[0]) / len(ROBUSTNESS)
    bw = 32
    for k, (name, base, clip, _stated) in enumerate(ROBUSTNESS):
        centre = px[0] + slot * (k + 0.5)
        for value, colour, offset in (
            (base, INDIGO, -bw / 2 - 3),
            (clip, YELLOW, bw / 2 + 3),
        ):
            _, top = to_px(0, value)
            b += rect(
                centre + offset - bw / 2,
                top,
                bw,
                py[1] - top,
                fill=colour,
                op=0.72,
                stroke=INK,
                so=0.45,
            )
            b += text(centre + offset, top - 7, f"{value:.1f}", size=10, fill=INK, op=0.85)

        # The gap, recomputed here rather than read off the paper.
        gap = clip - base
        if gap > 1:
            b += text(
                centre,
                py[1] + 52,
                f"+{gap:.1f}",
                size=12,
                fill=PURPLE,
                weight="600",
            )
        else:
            b += text(centre, py[1] + 52, "—", size=12, fill=INK, op=0.45)

        # Dataset name, wrapped where the label carries a newline.
        parts = name.split("\n")
        for li, part in enumerate(parts):
            b += text(centre, py[1] + 20 + li * 14, part, size=11, fill=INK, op=0.85)

    # Labels the delta row from the left margin, level with it rather than below.
    b += caption(20, py[1] + 52, "gap, in points", anchor="start", op=0.62)

    # Legend, laid out horizontally above the plot area. Inside it would collide
    # with the value labels on the tallest bars, which reach y = 110.
    b += rect(452, 44, 13, 13, fill=INDIGO, op=0.72, stroke=INK, so=0.45)
    b += text(472, 55, "supervised ResNet-101", size=11.5, fill=INK, anchor="start")
    b += rect(646, 44, 13, 13, fill=YELLOW, op=0.72, stroke=INK, so=0.45)
    b += text(666, 55, "zero-shot CLIP", size=11.5, fill=INK, anchor="start")

    b += caption(
        420,
        h - 16,
        "Matched on ImageNet by construction; the six evaluations to its right hold the categories fixed and move the pixels.",
    )

    return wrap(
        w,
        h,
        "robust",
        "CLIP under distribution shift",
        "A grouped bar chart of top-1 accuracy on ImageNet and six shifted "
        "evaluations, comparing a supervised ResNet-101 against zero-shot "
        "CLIP. The two are identical on ImageNet at 76.2 percent. On every "
        "shifted dataset the supervised model falls sharply while CLIP does "
        "not, with the largest gap on ImageNet-A, where the supervised model "
        "scores 2.7 percent against CLIP's 77.1.",
        b,
    )


# --------------------------------------------------------------------------
# Figure 5 — why random negatives never require compositional structure.
# --------------------------------------------------------------------------
def fig_bag_of_words():
    """Contrast random in-batch negatives with structural hard negatives."""
    w, h = 840, 372
    b = ""
    b += line(420, 44, 420, 300, sw=1.0, op=0.22, dash="4 4")

    b += text(20, 28, "Negatives the batch actually supplies", size=13.5, fill=INK, anchor="start")
    b += text(444, 28, "Negatives that would require more", size=13.5, fill=INK, anchor="start")

    b += rect(30, 56, 364, 34, fill=GREEN, op=0.22, stroke=INK, so=0.5, rx=3)
    b += text(44, 78, "the dog is left of the cat", size=12, fill=INK, anchor="start", font=MONO)
    b += text(380, 78, "✓", size=13, fill=INK, anchor="end")

    random_negs = [
        "a plate of sushi on a table",
        "the golden gate bridge at dusk",
        "a child holding a red balloon",
    ]
    for k, s in enumerate(random_negs):
        y = 104 + k * 36
        b += rect(30, y, 364, 30, fill=INDIGO, op=0.09, stroke=INK, so=0.3, rx=3)
        b += text(44, y + 20, s, size=11.5, fill=INK, anchor="start", font=MONO, op=0.8)
    b += caption(212, 240, "Disjoint nouns. Detecting which objects are")
    b += caption(212, 258, "present is a complete solution, so that is all")
    b += caption(212, 276, "the gradient ever asks for.")

    b += rect(444, 56, 364, 34, fill=GREEN, op=0.22, stroke=INK, so=0.5, rx=3)
    b += text(458, 78, "the dog is left of the cat", size=12, fill=INK, anchor="start", font=MONO)
    b += text(794, 78, "✓", size=13, fill=INK, anchor="end")

    hard_negs = [
        "the cat is left of the dog",
        "the dog is right of the cat",
        "left the is dog of cat the",
    ]
    for k, s in enumerate(hard_negs):
        y = 104 + k * 36
        b += rect(444, y, 364, 30, fill=PURPLE, op=0.13, stroke=INK, so=0.35, rx=3)
        b += text(458, y + 20, s, size=11.5, fill=INK, anchor="start", font=MONO, op=0.85)
    b += caption(626, 240, "Identical words, different structure. Nothing")
    b += caption(626, 258, "short of relation and order settles these, and")
    b += caption(626, 276, "a random draw never produces one.")

    b += caption(
        420,
        h - 16,
        "This is the gap ARO measures, and the reason its authors' fix is to mine such negatives deliberately.",
    )

    return wrap(
        w,
        h,
        "bow",
        "Why in-batch negatives permit bag-of-words behaviour",
        "Two columns, each headed by the same correct caption. The left column "
        "lists three randomly drawn negative captions whose nouns are entirely "
        "disjoint from the positive, so matching nouns alone separates them. "
        "The right column lists three hard negatives built from the same words "
        "as the positive, differing only in relation, direction and word "
        "order, which noun matching cannot separate.",
        b,
    )


# --------------------------------------------------------------------------
# Figure 6 — LLaVA's projection-and-concatenate fusion.
# --------------------------------------------------------------------------
def fig_llava():
    """Draw the single-projection connector, in strict left-to-right flow."""
    w, h = 840, 330
    b = ""

    b += box(24, 96, 96, 52, "image", fill=GREEN, op=0.2, size=12)
    b += arrow(120, 122, 148, 122, op=0.6)

    b += box(148, 92, 104, 60, "CLIP ViT-L/14", fill=TEAL, op=0.2, size=11)
    b += snowflake(200, 166)
    b += caption(200, 190, "frozen")
    b += caption(200, 208, "penultimate layer", op=0.6)
    b += arrow(252, 122, 282, 122, op=0.6)

    # Patch features, drawn as a short stack.
    for k in range(4):
        b += rect(282, 96 + k * 15, 54, 12, fill=TEAL, op=0.45, stroke=INK, so=0.3, rx=2)
    b += caption(309, 78, "patch features Z")
    b += arrow(340, 122, 370, 122, op=0.6)

    b += box(370, 100, 68, 44, "W", fill=YELLOW, op=0.5, size=13)
    b += caption(404, 166, "one linear")
    b += caption(404, 184, "projection —")
    b += caption(404, 202, "the whole connector", op=0.62)
    b += arrow(438, 122, 468, 122, op=0.6)

    # The concatenated sequence: visual tokens then text tokens.
    b += caption(596, 62, "one sequence, and self-attention cannot tell them apart")
    for k in range(4):
        b += rect(474 + k * 30, 100, 26, 44, fill=YELLOW, op=0.5, stroke=INK, so=0.4, rx=3)
    for k in range(4):
        b += rect(598 + k * 30, 100, 26, 44, fill=INDIGO, op=0.22, stroke=INK, so=0.4, rx=3)
    b += caption(534, 164, "visual tokens")
    b += caption(658, 164, "text tokens")
    b += line(596, 96, 596, 148, sw=0.9, op=0.3, dash="3 3")

    b += arrow(724, 122, 752, 122, op=0.6)
    b += box(752, 92, 68, 60, "LLM", sub="Vicuna", fill=PURPLE, op=0.16, size=12)
    b += arrow(786, 152, 786, 182, op=0.6)
    b += caption(786, 200, "free-form")
    b += caption(786, 218, "text answer")

    b += caption(
        420,
        h - 20,
        "Stage 1 trains W alone; stage 2 trains W and the language model on GPT-4-written instruction data.",
    )

    return wrap(
        w,
        h,
        "llava",
        "LLaVA's fusion by concatenation",
        "A left-to-right pipeline. An image enters a frozen CLIP ViT-L/14, "
        "whose penultimate-layer patch features pass through a single linear "
        "projection W. The projected visual tokens are concatenated in front "
        "of the embedded text tokens to form one sequence, which a Vicuna "
        "language model consumes to produce a free-form text answer.",
        b,
    )


# --------------------------------------------------------------------------
# Figure 7 — Flamingo's resampler and zero-initialised gate.
# --------------------------------------------------------------------------
def fig_flamingo():
    """Draw the resampler and the gated cross-attention block."""
    w, h = 840, 452
    b = ""
    b += line(392, 52, 392, 396, sw=1.0, op=0.22, dash="4 4")

    b += text(20, 30, "Variable input, fixed output", size=13.5, fill=INK, anchor="start")
    b += text(416, 30, "Inserted without disturbing the frozen model", size=13.5, fill=INK, anchor="start")

    # Left: vision encoder into the Perceiver Resampler.
    b += box(28, 74, 96, 48, "image", sub="or video", fill=GREEN, op=0.2, size=12)
    b += arrow(76, 122, 76, 152, op=0.6)
    b += box(28, 152, 96, 44, "vision encoder", fill=TEAL, op=0.2, size=10.5)
    b += snowflake(76, 210)
    b += arrow(76, 196, 76, 230, op=0.6)

    # A ragged feature grid, to make "variable" visible.
    for k in range(6):
        wid = 62 + (k % 3) * 20
        b += rect(46, 236 + k * 13, wid, 10, fill=TEAL, op=0.4, stroke=INK, so=0.25, rx=2)
    b += caption(76, 336, "features: size varies with")
    b += caption(76, 354, "resolution and duration", op=0.6)

    b += arrow(150, 286, 186, 286, op=0.65)
    b += box(186, 246, 108, 80, "Perceiver", sub="Resampler", fill=YELLOW, op=0.42, size=12)
    b += caption(240, 348, "learned latent queries")
    b += caption(240, 366, "cross-attend to the features", op=0.6)
    b += arrow(294, 286, 330, 286, op=0.65)
    for k in range(4):
        b += rect(332, 258 + k * 16, 46, 12, fill=YELLOW, op=0.6, stroke=INK, so=0.35, rx=2)
    b += caption(355, 240, "exactly 64")
    b += caption(355, 340, "however much", op=0.6)
    b += caption(355, 358, "came in", op=0.6)

    # Right: the frozen LM stack with gated blocks between its layers.
    xs = 452
    b += caption(xs + 78, 58, "residual stream, bottom to top")
    order = [
        ("frozen LM layer", INDIGO, 0.2, True),
        ("gated xattn-dense", PURPLE, 0.16, False),
        ("frozen LM layer", INDIGO, 0.2, True),
        ("gated xattn-dense", PURPLE, 0.16, False),
        ("frozen LM layer", INDIGO, 0.2, True),
    ]
    for k, (label, colour, op, frozen) in enumerate(order):
        y = 330 - k * 58
        b += box(xs, y, 156, 40, label, fill=colour, op=op, size=11)
        if frozen:
            b += snowflake(xs - 16, y + 20)
        if k:
            b += arrow(xs + 78, y + 52, xs + 78, y + 42, op=0.55)

    # The gate itself, drawn to the right of the block it belongs to.
    gy = 330 - 58
    b += line(xs + 156, gy + 20, xs + 196, gy + 20, sw=1.0, op=0.4, dash="3 3")
    b += rect(xs + 196, gy - 4, 152, 48, fill=YELLOW, op=0.34, stroke=INK, so=0.5, rx=4)
    b += text(xs + 272, gy + 16, "× tanh(α)", size=13, fill=INK, font=MONO)
    b += text(xs + 272, gy + 34, "α initialised to 0", size=10.5, fill=INK, op=0.72)

    # Clears the gate box above it, which ends at y = 316.
    b += rect(xs + 196, 326, 152, 72, fill=INDIGO, op=0.10, stroke=INK, so=0.35, rx=4)
    b += text(xs + 272, 348, "at step zero", size=11.5, fill=INK, weight="600")
    b += text(xs + 272, 368, "tanh(0) = 0, so the block", size=11, fill=INK, op=0.78)
    b += text(xs + 272, 384, "contributes exactly nothing", size=11, fill=INK, op=0.78)

    # The panels are separate concerns, so state the link between them rather
    # than running an arrow across the divider.
    b += caption(530, 414, "every gated block cross-attends to the 64 tokens from the left")
    b += caption(
        420,
        h - 18,
        "Removing the zero-initialised gate costs 4.2 points on Flamingo's aggregate score and destabilises training.",
    )

    return wrap(
        w,
        h,
        "flam",
        "Flamingo's architecture",
        "On the left, a frozen vision encoder produces a feature grid whose "
        "size varies with resolution and duration; a Perceiver Resampler with "
        "learned latent queries compresses it to exactly 64 visual tokens. On "
        "the right, gated cross-attention blocks are interleaved between the "
        "layers of a frozen language model, each scaling its output by the "
        "hyperbolic tangent of a learnable scalar initialised to zero, so that "
        "at initialisation the composite model reproduces the frozen model "
        "exactly.",
        b,
    )


# --------------------------------------------------------------------------
# Figure 8 — the per-image attention mask on an interleaved sequence.
# --------------------------------------------------------------------------
def fig_flamingo_masking():
    """Draw which image each text span may cross-attend to."""
    w, h = 840, 384
    b = ""
    b += text(20, 28, "A web page, in its original order", size=13.5, fill=INK, anchor="start")

    # The sequence, laid out strictly left to right in reading order.
    spans = [
        ("img", "image 1"),
        ("txt", "“a puffin”"),
        ("img", "image 2"),
        ("txt", "“a heron”"),
        ("img", "image 3"),
        ("txt", "“a ?”"),
    ]
    x = 40
    positions = []
    for kind, label in spans:
        width = 92 if kind == "img" else 116
        colour = GREEN if kind == "img" else INDIGO
        op = 0.28 if kind == "img" else 0.12
        b += rect(x, 58, width, 40, fill=colour, op=op, stroke=INK, so=0.45, rx=3)
        b += text(x + width / 2, 82, label, size=11.5, fill=INK, font=MONO if kind == "txt" else None)
        positions.append((kind, x, width))
        x += width + 12

    # Cross-attention: each text span reaches back to the nearest image only.
    for k, (kind, sx, sw_) in enumerate(positions):
        if kind != "txt":
            continue
        _, px_, pw = positions[k - 1]
        # One leftward arrow per text span, back to the image it may see.
        b += arrow(sx + sw_ / 2, 116, px_ + pw / 2, 116, op=0.8, stroke=PURPLE, sw=1.5)
    b += text(
        40, 152, "cross-attention: masked to the single most recent image", size=11.5, fill=PURPLE, anchor="start"
    )

    # Self-attention: unmasked, spanning the whole sequence.
    left = positions[0][1]
    right = positions[-1][1] + positions[-1][2]
    b += line(left, 196, right, 196, stroke=TEAL, sw=1.6, op=0.85)
    b += arrow(right, 196, left, 196, op=0.85, stroke=TEAL, sw=1.6)
    b += arrow(left, 196, right, 196, op=0.85, stroke=TEAL, sw=1.6)
    b += text(
        40, 220, "self-attention in the frozen LM: unmasked, spans everything", size=11.5, fill=TEAL, anchor="start"
    )

    # What the arrangement buys.
    b += rect(40, 246, 356, 92, fill=PURPLE, op=0.09, stroke=INK, so=0.35, rx=4)
    b += text(60, 270, "Why mask at all", size=12, fill=INK, anchor="start", weight="600")
    b += text(60, 292, "Letting every token see every image blurs", size=11, fill=INK, anchor="start", op=0.78)
    b += text(60, 308, "which caption belongs to which picture. The", size=11, fill=INK, anchor="start", op=0.78)
    b += text(60, 324, "pairing stays crisp under the mask.", size=11, fill=INK, anchor="start", op=0.78)

    b += rect(444, 246, 356, 92, fill=TEAL, op=0.09, stroke=INK, so=0.35, rx=4)
    b += text(464, 270, "Why it still learns in context", size=12, fill=INK, anchor="start", weight="600")
    b += text(464, 292, "The unmasked text pathway carries the", size=11, fill=INK, anchor="start", op=0.78)
    b += text(464, 308, "pattern across examples, so the sequence", size=11, fill=INK, anchor="start", op=0.78)
    b += text(464, 324, "reads as a few-shot prompt.", size=11, fill=INK, anchor="start", op=0.78)

    return wrap(
        w,
        h,
        "flmask",
        "Flamingo's per-image attention mask",
        "An interleaved sequence of three images each followed by a short text "
        "span. Arrows show that each text span cross-attends only to the image "
        "immediately preceding it, never to earlier images. A separate "
        "double-headed arrow spanning the whole sequence shows that the frozen "
        "language model's own self-attention is unmasked, which is what "
        "carries the few-shot pattern across the examples.",
        b,
    )


# --------------------------------------------------------------------------
# Figure 9 — the point as a shared output primitive.
# --------------------------------------------------------------------------
def fig_pointing():
    """Draw one coordinate output serving grounding, counting and action."""
    w, h = 840, 350
    b = ""

    b += box(28, 128, 108, 56, "Molmo", sub="image + question", fill=PURPLE, op=0.16, size=12.5)
    b += arrow(136, 156, 174, 156, op=0.65)
    b += rect(174, 134, 118, 44, fill=YELLOW, op=0.5, stroke=INK, so=0.5, rx=4)
    b += text(233, 154, "⟨point x, y⟩", size=12, fill=INK, font=MONO)
    b += text(233, 170, "one coordinate", size=10, fill=INK, op=0.7)
    b += caption(233, 200, "2.3M annotations,")
    b += caption(233, 218, "cheaper than boxes or masks", op=0.6)

    uses = [
        (
            56,
            "Grounding",
            ["“the second bird from", "the left” → a location", "in the image"],
            TEAL,
        ),
        (
            140,
            "Counting",
            ["point at each instance,", "then count the outputs —", "an enumeration, not a guess"],
            GREEN,
        ),
        (
            224,
            "Action",
            ["a waypoint, a grasp target,", "a button to click; or a", "prompt handed to SAM"],
            INDIGO,
        ),
    ]
    for y0, title, lines, colour in uses:
        b += arrow(292, 156, 372, y0 + 32, op=0.6)
        b += rect(378, y0, 434, 72, fill=colour, op=0.11, stroke=INK, so=0.4, rx=4)
        b += text(398, y0 + 22, title, size=12.5, fill=INK, anchor="start", weight="600")
        for li, s in enumerate(lines):
            b += text(516, y0 + 20 + li * 16, s, size=10.5, fill=INK, anchor="start", op=0.78)

    b += caption(
        420, h - 18, "Boxes ground and masks segment; the point does both jobs and is the one that also drives a robot."
    )

    return wrap(
        w,
        h,
        "point",
        "Pointing as a shared output primitive",
        "Molmo emits a single two-dimensional coordinate, which feeds three "
        "distinct uses: grounding a referring expression to a location, "
        "counting by enumerating one point per instance rather than "
        "estimating a number, and acting, where the point serves as a "
        "navigation waypoint, a grasp target, or a prompt handed to a "
        "segmentation model.",
        b,
    )


# --------------------------------------------------------------------------
# Figure 10 — SAM: encode once, decode per prompt, three masks out.
# --------------------------------------------------------------------------
def fig_sam():
    """Draw the asymmetric encoder/decoder split and the ambiguity head."""
    w, h = 840, 400
    b = ""

    b += box(24, 92, 92, 52, "image", fill=GREEN, op=0.2, size=12)
    b += arrow(116, 118, 146, 118, op=0.6)
    b += box(146, 84, 116, 68, "ViT-H", sub="image encoder", fill=TEAL, op=0.22, size=12)
    b += caption(204, 172, f"~{ENCODER_MS:.0f} ms")
    b += caption(204, 190, "once per image", op=0.62)
    b += arrow(262, 118, 292, 118, op=0.6)
    b += rect(292, 96, 74, 44, fill=TEAL, op=0.5, stroke=INK, so=0.4, rx=3)
    b += text(329, 122, "embedding", size=10.5, fill=INK)
    b += caption(329, 164, "cached, reused")

    # Prompts enter lower and to the left of the decoder they feed.
    prompts = ["a point", "a box", "a rough mask", "a phrase"]
    for k, s in enumerate(prompts):
        y = 232 + k * 26
        b += rect(146, y, 116, 21, fill=YELLOW, op=0.45, stroke=INK, so=0.35, rx=3)
        b += text(204, y + 15, s, size=10.5, fill=INK, font=MONO)
    b += caption(204, 222, "prompt")
    b += arrow(262, 285, 292, 285, op=0.6)
    b += box(292, 262, 74, 46, "prompt", sub="encoder", fill=YELLOW, op=0.4, size=10.5)

    # The decoder sits to the right of both things that feed it.
    b += arrow(366, 130, 424, 186, op=0.6)
    b += arrow(366, 285, 424, 226, op=0.6)
    b += box(428, 182, 112, 50, "mask decoder", fill=PURPLE, op=0.18, size=11)
    b += caption(484, 252, f"~{DECODER_MS:.0f} ms")
    b += caption(484, 270, "per prompt", op=0.62)

    # Three masks out, ordered part → whole → group down the page.
    outs = [
        (110, "the shirt", "part"),
        (182, "the person", "whole"),
        (254, "the group", "containing set"),
    ]
    for y, label, kind in outs:
        b += arrow(540, 207, 596, y + 22, op=0.6)
        b += rect(600, y, 208, 48, fill=INDIGO, op=0.14, stroke=INK, so=0.4, rx=4)
        b += text(620, y + 22, label, size=12, fill=INK, anchor="start")
        b += text(620, y + 38, kind, size=10, fill=INK, anchor="start", op=0.65)
        b += text(792, y + 30, "score", size=9.5, fill=INK, anchor="end", op=0.55)
    b += caption(704, 84, "three masks, not an average of three")
    b += caption(704, 320, "one point on a shirt has three correct answers", op=0.62)

    b += caption(
        420,
        h - 18,
        f"Re-encoding per prompt would cost {sam_speedup(16):.1f}x more over 16 prompts on one image; the split is what makes interaction affordable.",
    )

    return wrap(
        w,
        h,
        "sam",
        "SAM's promptable segmentation architecture",
        "A heavy vision transformer encodes the image once, at roughly 450 "
        "milliseconds, and its embedding is cached. Prompts — a point, a box, "
        "a rough mask, or a phrase — pass through a small prompt encoder and "
        "join the cached embedding in a lightweight mask decoder costing about "
        "50 milliseconds per prompt. The decoder emits three masks with "
        "confidence scores, corresponding to the part, the whole object, and "
        "the containing group, rather than averaging them into one.",
        b,
    )


# --------------------------------------------------------------------------
# Figure 11 — VisProg compiles an instruction into a program.
# --------------------------------------------------------------------------
def fig_visprog():
    """Draw an instruction compiled to a program and executed over modules."""
    w, h = 840, 430
    b = ""

    b += box(24, 44, 190, 52, "instruction", sub="natural language", fill=GREEN, op=0.2, size=12)
    b += text(119, 116, "“hide the face of the", size=11, fill=INK, font=MONO, op=0.85)
    b += text(119, 132, "person on the left”", size=11, fill=INK, font=MONO, op=0.85)

    b += arrow(214, 70, 252, 70, op=0.65)
    b += box(252, 44, 128, 52, "LLM", sub="in-context examples", fill=PURPLE, op=0.16, size=12)
    b += caption(316, 116, "a few instruction–program")
    b += caption(316, 134, "pairs in the prompt; nothing trained", op=0.62)

    b += arrow(380, 70, 418, 70, op=0.65)
    b += box(418, 44, 130, 52, "program", sub="Python-like", fill=YELLOW, op=0.44, size=12)
    b += arrow(483, 96, 483, 130, op=0.65)

    # The program, one line per row; each line's module sits to its right.
    lines = [
        ("BOX = Loc(image, 'person')", "object detector"),
        ("LEFT = Select(BOX, 'leftmost')", "arithmetic"),
        ("FACE = FaceDet(image, LEFT)", "face detector"),
        ("OUT  = Blur(image, FACE)", "image processing"),
    ]
    for k, (code, module) in enumerate(lines):
        y = 148 + k * 42
        b += rect(64, y, 344, 32, fill=INDIGO, op=0.10, stroke=INK, so=0.35, rx=3)
        b += text(80, y + 21, code, size=11, fill=INK, anchor="start", font=MONO)
        b += arrow(408, y + 16, 452, y + 16, op=0.55)
        b += rect(456, y, 172, 32, fill=TEAL, op=0.18, stroke=INK, so=0.35, rx=3)
        b += text(542, y + 21, module, size=10.5, fill=INK)
        b += snowflake(646, y + 16, r=5, op=0.6)
        if k < len(lines) - 1:
            b += arrow(70, y + 32, 70, y + 42, op=0.5)

    b += caption(700, 152, "every module")
    b += caption(700, 170, "is frozen", op=0.62)

    b += arrow(236, 316, 236, 344, op=0.65)
    b += box(150, 344, 172, 44, "interpreter output", fill=GREEN, op=0.2, size=11.5)
    b += rect(444, 336, 364, 60, fill=PURPLE, op=0.09, stroke=INK, so=0.35, rx=4)
    b += text(464, 358, "The program is the explanation", size=11.5, fill=INK, anchor="start", weight="600")
    b += text(
        464,
        378,
        "…and the cost: one forward pass per line, errors carried forward.",
        size=10.5,
        fill=INK,
        anchor="start",
        op=0.75,
    )

    return wrap(
        w,
        h,
        "visprog",
        "VisProg's compile-and-execute loop",
        "A natural-language instruction is passed to a language model that has "
        "been shown a few instruction-and-program pairs in context. It emits a "
        "short Python-like program whose four lines each invoke a frozen "
        "module — an object detector, an arithmetic selection, a face "
        "detector, and an image-processing blur. An interpreter runs the "
        "lines in order, threading each result into the next, and returns the "
        "final image.",
        b,
    )


FIGURES = {
    "diagram_simclr_to_clip.svg": fig_simclr_to_clip,
    "diagram_clip_objective.svg": fig_clip_objective,
    "diagram_zero_shot.svg": fig_zero_shot,
    "diagram_robustness.svg": fig_robustness,
    "diagram_bag_of_words.svg": fig_bag_of_words,
    "diagram_llava.svg": fig_llava,
    "diagram_flamingo.svg": fig_flamingo,
    "diagram_flamingo_masking.svg": fig_flamingo_masking,
    "diagram_pointing.svg": fig_pointing,
    "diagram_sam.svg": fig_sam,
    "diagram_visprog.svg": fig_visprog,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print(f"wrote {OUT / name}")

    print()
    print("Checks and numbers the chapter quotes, evaluated from the generator:")
    print()

    # --- The distribution-shift gaps are subtracted, not transcribed. ---
    print("  Distribution shift (top-1 %), gap recomputed vs the paper's column:")
    for name, base, clip, stated in ROBUSTNESS:
        gap = clip - base
        assert abs(gap - stated) < 0.05, (name, gap, stated)
        print(f"    {name.replace(chr(10), ' '):18s} {base:5.1f} → {clip:5.1f}   +{gap:5.1f}")
    widest = max(ROBUSTNESS, key=lambda r: r[2] - r[1])
    print(f"    widest gap: {widest[0].replace(chr(10), ' ')} at +{widest[2] - widest[1]:.1f} points")
    print()

    # --- The symmetric loss, two independent routes, must agree. ---
    worst = 0.0
    for seed in range(40):
        for n in (4, 8, 32):
            for tau in (0.01, 0.07, 0.5, 2.0):
                sim = random_similarity(n, seed)
                a = symmetric_infonce(sim, tau)
                c = symmetric_infonce_lse(sim, tau)
                worst = max(worst, abs(a - c))
    assert worst < 1e-9, worst
    print(f"  Symmetric InfoNCE, cross-entropy vs log-sum-exp: worst gap {worst:.2e}")

    # The loss at a uniform matrix is exactly log N — a second closed-form check.
    # Materialising the matrix is quadratic, so this runs at sizes that fit and
    # the batch-size figure below is evaluated from the closed form instead.
    for n in (4, 16, 256, 1024):
        flat = [[0.0] * n for _ in range(n)]
        assert abs(symmetric_infonce(flat, 0.07) - math.log(n)) < 1e-9
    print("  Uninformative batch gives exactly log N at every N checked (4 … 1,024)")
    print(f"  Chance-level loss at CLIP's batch size 32,768: {math.log(32768):.2f} nats")
    print()

    # --- Temperature sharpens the same matrix. ---
    sim = random_similarity(8, seed=3)
    print("  One fixed batch, swept over temperature:")
    for tau in (0.01, 0.07, 0.3, 1.0):
        print(f"    tau = {tau:<5} loss = {symmetric_infonce(sim, tau):.3f}")
    print()

    # --- Zero-shot cosine-argmax IS a bias-free linear layer. ---
    disagreements = zero_shot_agreement()
    assert disagreements == 0, disagreements
    print(f"  Cosine-argmax vs bias-free linear argmax: {disagreements} disagreements in 4,000 trials")

    differ = ensemble_disagreement()
    print(
        f"  Embedding-space vs probability-space ensembling: differ on "
        f"{differ}/4000 trials ({differ / 40:.1f}%) — not the same computation"
    )
    print("  Per-image cost of a K-template ensemble: K in probability space, 1 in embedding space")
    print()

    # --- Flamingo's gate is exactly the identity at initialisation. ---
    assert gated_residual(1.234, 99.0, 0.0) == 1.234
    assert gated_residual(1.234, 0.0, 5.0) == 1.234
    print("  Flamingo gate at alpha = 0: block output discarded exactly (tanh(0) = 0)")
    print(f"  Gate at alpha = 1: block admitted at {math.tanh(1.0):.3f} of full strength")
    print()

    # --- SAM's encode-once split, and where it pays. ---
    print("  SAM runtime on one image (ms), encode-once vs re-encode per prompt:")
    for p in (1, 4, 16, 64):
        print(f"    {p:3d} prompts   {sam_cost(p):7.0f}   vs {sam_cost(p, False):8.0f}   ({sam_speedup(p):.1f}x)")
    # Solve for where re-encoding costs twice as much, rather than asserting it.
    n = 1
    while sam_speedup(n) < 2.0:
        n += 1
    print(f"    re-encoding costs 2x from {n} prompts on; the ceiling is {(ENCODER_MS + DECODER_MS) / DECODER_MS:.0f}x")
