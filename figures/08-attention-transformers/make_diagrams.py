"""Generate the eight SVG diagrams for the lecture 8 chapter.

Nothing here is traced from a slide. Seven of the eight are structural drawings
whose geometry comes from the layout constants below rather than from hand
placement; the eighth, the alignment heat map, is computed from an explicit
kernel over a stated word alignment and is a schematic of the structure a
trained model tends to produce, not the output of one.

Conventions follow figures/06-cnn-architectures and 07: currentColor for text
and rules, viridis for data, title/desc with namespaced ids, no background rect
(the theme supplies a light plate in both schemes).

Run from the repository root; rewrites all eight files and prints the two
figures the chapter quotes.
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
    rect,
    text,
    wrap,
)

OUT = pathlib.Path("figures/08-attention-transformers")

VIRIDIS = [PURPLE, INDIGO, TEAL, GREEN, YELLOW]


def op_node(cx, cy, glyph, r=13.0, colour=INK):
    """A small circled operator on a signal path."""
    out = circle(cx, cy, r, stroke=colour, sw=1.3, so=0.85)
    return out + text(cx, cy + 5.0, glyph, size=15, fill=colour, op=0.9)


def ramp(t):
    """Sample the five viridis stops at t in [0, 1], linearly interpolated."""
    t = min(max(t, 0.0), 1.0) * (len(VIRIDIS) - 1)
    i = min(int(t), len(VIRIDIS) - 2)
    f = t - i
    lo, hi = VIRIDIS[i], VIRIDIS[i + 1]
    chan = [
        round(int(lo[1 + 2 * k : 3 + 2 * k], 16) * (1 - f) + int(hi[1 + 2 * k : 3 + 2 * k], 16) * f) for k in range(3)
    ]
    return "#{:02x}{:02x}{:02x}".format(*chan)


def block(x, y, w, h, label, sub=None, colour=INK, fill=None, op=0.13, size=13):
    """A labelled rectangle, optionally with a smaller second line under it."""
    out = rect(x, y, w, h, fill=fill or colour, op=op if fill or colour else 0.0, stroke=colour, sw=1.1, so=0.65, rx=3)
    dy = h / 2 + 4.5 if sub is None else h / 2 - 2.0
    out += text(x + w / 2, y + dy, label, size=size, fill=INK)
    if sub:
        out += text(x + w / 2, y + h / 2 + 14.0, sub, size=10, fill=INK, op=0.6, font=MONO)
    return out


# ---------------------------------------------------------------------------
# 1. The encoder-decoder bottleneck.
# ---------------------------------------------------------------------------

BN_W, BN_H = 880.0, 300.0
BN_ENC_X = [60.0, 148.0, 236.0, 324.0]
BN_DEC_X = [488.0, 576.0, 664.0, 752.0]
BN_BW = 64.0
BN_C_X = 396.0
BN_STATE_Y, BN_IN_Y, BN_OUT_Y = 128.0, 210.0, 46.0
BN_BH = 40.0
BN_RAIL_Y = 288.0

SRC = ["we", "see", "the", "sky"]
TGT = ["vediamo", "il", "cielo", "&lt;stop&gt;"]


def fig_bottleneck():
    """Encoder, decoder, and the single context vector between them."""
    body = text(BN_W / 2, 24, "everything the decoder knows about the source passes through c", size=11.5, op=0.6)

    for i, x in enumerate(BN_ENC_X):
        body += block(x, BN_IN_Y, BN_BW, BN_BH, SRC[i], colour=INDIGO)
        body += arrow(x + BN_BW / 2, BN_IN_Y - 4, x + BN_BW / 2, BN_STATE_Y + BN_BH + 6)
        body += block(x, BN_STATE_Y, BN_BW, BN_BH, f"h{i + 1}", colour=INDIGO, op=0.07)
        if i:
            body += arrow(BN_ENC_X[i - 1] + BN_BW + 2, BN_STATE_Y + BN_BH / 2, x - 2, BN_STATE_Y + BN_BH / 2)

    body += arrow(BN_ENC_X[3] + BN_BW + 2, BN_STATE_Y + BN_BH / 2, BN_C_X - 2, BN_STATE_Y + BN_BH / 2)
    body += block(BN_C_X, BN_STATE_Y, BN_BW, BN_BH, "c", colour=PURPLE, op=0.3)
    body += text(BN_C_X + BN_BW / 2, BN_STATE_Y - 10, "fixed width", size=10, fill=PURPLE, op=0.85)

    # The purple rail: c is the only route from the source into every decoder step.
    cx = BN_C_X + BN_BW / 2
    body += line(cx, BN_STATE_Y + BN_BH, cx, BN_RAIL_Y, stroke=PURPLE, sw=1.6, op=0.8)
    body += line(cx, BN_RAIL_Y, BN_DEC_X[3] + BN_BW / 2, BN_RAIL_Y, stroke=PURPLE, sw=1.6, op=0.8)

    for i, x in enumerate(BN_DEC_X):
        body += arrow(x + BN_BW / 2, BN_RAIL_Y, x + BN_BW / 2, BN_STATE_Y + BN_BH + 6, stroke=PURPLE, op=0.8)
        body += block(x, BN_STATE_Y, BN_BW, BN_BH, f"s{i + 1}", colour=TEAL, op=0.07)
        body += arrow(x + BN_BW / 2, BN_STATE_Y - 4, x + BN_BW / 2, BN_OUT_Y + BN_BH + 6)
        body += block(x, BN_OUT_Y, BN_BW, BN_BH, TGT[i], colour=TEAL, size=11)
        if i:
            body += arrow(BN_DEC_X[i - 1] + BN_BW + 2, BN_STATE_Y + BN_BH / 2, x - 2, BN_STATE_Y + BN_BH / 2)

    body += text(BN_ENC_X[0], BN_IN_Y + BN_BH + 22, "encoder", size=11.5, anchor="start", op=0.6)
    body += text(BN_DEC_X[0], BN_STATE_Y + BN_BH + 22, "decoder", size=11.5, anchor="start", op=0.6)

    return wrap(
        BN_W,
        BN_H,
        "bottleneck",
        "Sequence-to-sequence with recurrent networks, bottlenecked through one context vector",
        "Four indigo input words feed a chain of four encoder states. The last "
        "encoder state produces a single purple context vector c of fixed width. "
        "A purple rail carries c to every one of the four decoder states, which "
        "chain left to right and emit the translated words above them. No other "
        "route connects the source to the output.",
        body,
    )


# ---------------------------------------------------------------------------
# 2. One decoder step of attention.
# ---------------------------------------------------------------------------

AT_W, AT_H = 800.0, 404.0
AT_ROWS = [56.0, 112.0, 168.0, 224.0]
AT_RH = 44.0
AT_HX, AT_HW = 40.0, 76.0
AT_EX, AT_AX, AT_CW = 176.0, 296.0, 60.0
AT_SUM_X, AT_SUM_Y = 440.0, 166.0
AT_S_Y = 320.0
AT_RAIL_Y = 382.0


def fig_attention_step():
    """Scores, softmax and weighted sum for one step of the decoder."""
    body = ""
    top_rail = 26.0
    for i, y in enumerate(AT_ROWS):
        body += block(AT_HX, y, AT_HW, AT_RH, f"h{i + 1}", colour=INDIGO, op=0.10)
        # the encoder states are also the things being averaged: rail over the top
        body += line(AT_HX + AT_HW - 12, y, AT_HX + AT_HW - 12, top_rail, op=0.35)
        body += arrow(AT_HX + AT_HW + 2, y + AT_RH / 2, AT_EX - 2, y + AT_RH / 2)
        body += block(AT_EX, y, AT_CW, AT_RH, f"e{i + 1}", colour=INK, op=0.0)
        body += arrow(AT_EX + AT_CW + 2, y + AT_RH / 2, AT_AX - 2, y + AT_RH / 2)
        body += block(AT_AX, y, AT_CW, AT_RH, f"a{i + 1}", colour=GREEN, op=0.18)
        body += arrow(AT_AX + AT_CW + 2, y + AT_RH / 2, AT_SUM_X - 15, AT_SUM_Y, op=0.6)

    body += line(AT_HX + AT_HW - 12, top_rail, AT_SUM_X, top_rail, op=0.35)
    body += arrow(AT_SUM_X, top_rail, AT_SUM_X, AT_SUM_Y - 15, op=0.5)
    body += text(
        AT_HX + AT_HW + 26, top_rail - 8, "the same h are the values averaged", size=10, anchor="start", op=0.5
    )

    body += text((AT_EX + AT_CW + AT_AX) / 2, AT_ROWS[0] - 14, "softmax", size=11, op=0.65)

    # The decoder state asks the question: one fan into every score.
    body += block(AT_HX, AT_S_Y, AT_HW, AT_RH, "s", colour=TEAL, op=0.14)
    body += text(AT_HX + AT_HW / 2 + 13, AT_S_Y + AT_RH / 2 + 9, "t−1", size=9, fill=INK, op=0.75)
    for y in AT_ROWS:
        body += arrow(AT_HX + AT_HW + 4, AT_S_Y + 8, AT_EX - 3, y + AT_RH - 10, op=0.4, dash="3 3")

    body += op_node(AT_SUM_X, AT_SUM_Y, "+")
    body += arrow(AT_SUM_X + 15, AT_SUM_Y, 480.0, AT_SUM_Y)
    body += block(480.0, AT_SUM_Y - AT_RH / 2, 76.0, AT_RH, "c", colour=PURPLE, op=0.28)
    body += text(480.0 + 38 + 10, AT_SUM_Y + 9, "t", size=9, fill=INK, op=0.75)
    body += arrow(556.0 + 2, AT_SUM_Y, 600.0 - 2, AT_SUM_Y)
    body += block(600.0, AT_SUM_Y - AT_RH / 2, 80.0, AT_RH, "g", colour=TEAL, op=0.10)
    body += text(600.0 + 40 + 8, AT_SUM_Y + 9, "U", size=9, fill=INK, op=0.75)
    body += arrow(680.0 + 2, AT_SUM_Y, 748.0, AT_SUM_Y)
    body += text(762.0, AT_SUM_Y + 4, "s", size=13)
    body += text(770.0, AT_SUM_Y + 9, "t", size=9, op=0.75)
    body += arrow(640.0, AT_SUM_Y - AT_RH / 2 - 2, 640.0, 96.0)
    body += block(600.0, 52.0, 80.0, AT_RH, "y", colour=TEAL, op=0.10)
    body += text(640.0 + 8, 52.0 + AT_RH / 2 + 9, "t", size=9, op=0.75)

    # s(t-1) is both the query above and an input to the recurrent unit.
    body += line(AT_HX + AT_HW / 2, AT_S_Y + AT_RH, AT_HX + AT_HW / 2, AT_RAIL_Y, op=0.45)
    body += line(AT_HX + AT_HW / 2, AT_RAIL_Y, 640.0, AT_RAIL_Y, op=0.45)
    body += arrow(640.0, AT_RAIL_Y, 640.0, AT_SUM_Y + AT_RH / 2 + 4, op=0.45)

    return wrap(
        AT_W,
        AT_H,
        "attnstep",
        "One decoder step: alignment scores, softmax, weighted sum, context vector",
        "A column of four encoder states on the left. Each produces an alignment "
        "score, and the decoder state below fans a dashed arrow into all four "
        "scores. A softmax turns the scores into four weights, which converge on "
        "a summation node together with a rail carrying the encoder states "
        "themselves. The sum is this step's purple context vector, which enters "
        "the recurrent unit alongside the previous decoder state and produces the "
        "output token.",
        body,
    )


# ---------------------------------------------------------------------------
# 3. A schematic soft alignment.
# ---------------------------------------------------------------------------

AL_SRC = ["the", "black", "cat", "sleeps", "here"]
AL_TGT = ["le", "chat", "noir", "dort", "ici"]
# Which source position each target word corresponds to. The adjective and noun
# exchange places, which is the crossing in the figure.
AL_MAP = [0, 2, 1, 3, 4]
AL_TAU = 0.62

AL_CELL = 56.0
AL_X0, AL_Y0 = 118.0, 76.0
AL_W = AL_X0 + AL_CELL * len(AL_SRC) + 30.0
AL_H = AL_Y0 + AL_CELL * len(AL_TGT) + 52.0


def alignment_weights():
    """Soft alignment rows: a Gaussian around the aligned source index."""
    rows = []
    for j in range(len(AL_TGT)):
        raw = [math.exp(-((i - AL_MAP[j]) ** 2) / (2 * AL_TAU**2)) for i in range(len(AL_SRC))]
        total = sum(raw)
        rows.append([v / total for v in raw])
    return rows


def fig_alignment():
    """Heat map of the attention weights a trained translator tends to learn."""
    rows = alignment_weights()
    body = ""
    for i, w in enumerate(AL_SRC):
        body += text(AL_X0 + AL_CELL * (i + 0.5), AL_Y0 - 14, w, size=12, op=0.8)
    for j, w in enumerate(AL_TGT):
        body += text(AL_X0 - 12, AL_Y0 + AL_CELL * (j + 0.5) + 4.5, w, size=12, anchor="end", op=0.8)

    for j, row in enumerate(rows):
        for i, v in enumerate(row):
            body += rect(
                AL_X0 + AL_CELL * i,
                AL_Y0 + AL_CELL * j,
                AL_CELL,
                AL_CELL,
                fill=ramp(v),
                op=1.0,
                stroke="#ffffff",
                sw=1.0,
                so=0.5,
            )

    body += text(AL_X0 + AL_CELL * len(AL_SRC) / 2, AL_Y0 - 36, "source", size=10.5, op=0.5)
    body += text(AL_X0 - 12, AL_Y0 - 14, "generated", size=10.5, anchor="end", op=0.5)
    body += text(
        AL_W / 2,
        AL_Y0 + AL_CELL * len(AL_TGT) + 26,
        "brighter = more weight; every row sums to one",
        size=10.5,
        op=0.55,
    )

    return wrap(
        AL_W,
        AL_H,
        "align",
        "A schematic soft alignment between an English sentence and its French translation",
        "A five by five grid of viridis-shaded cells. Source words run across the "
        "top and generated words down the left. The bright cells follow the "
        "diagonal except where the adjective and the noun exchange places, where "
        "the bright band crosses over itself. Neighbouring cells carry a little "
        "weight rather than none, so the alignment is soft rather than a hard "
        "matching.",
        body,
    )


# ---------------------------------------------------------------------------
# 4. The attention layer, with shapes.
# ---------------------------------------------------------------------------

SH_W, SH_H = 900.0, 380.0


def grid_block(x, y, w, h, nc, nr, values, colour):
    """A rectangle divided into a grid of shaded cells."""
    out = ""
    cw, ch = w / nc, h / nr
    for r in range(nr):
        for c in range(nc):
            out += rect(
                x + c * cw,
                y + r * ch,
                cw,
                ch,
                fill=colour,
                op=values(r, c),
                stroke="#ffffff",
                sw=0.8,
                so=0.35,
            )
    out += rect(x, y, w, h, stroke=INK, sw=1.1, so=0.5)
    return out


def fig_attention_shapes():
    """Queries, keys, values, scores, weights and outputs with every shape marked."""
    body = ""
    body += block(30, 56, 76, 62, "X", "NQ × D", colour=TEAL, op=0.12)
    body += text(30 + 76 / 2 + 9, 56 + 62 / 2 - 2 + 4, "Q", size=9, op=0.75)
    body += block(30, 214, 76, 88, "X", "NX × D", colour=INDIGO, op=0.12)

    body += arrow(106, 87, 194, 87)
    body += text(150, 78, "WQ", size=10.5, op=0.7, font=MONO)
    body += arrow(106, 250, 194, 205)
    body += text(148, 216, "WK", size=10.5, op=0.7, font=MONO)
    body += arrow(106, 268, 194, 300)
    body += text(148, 298, "WV", size=10.5, op=0.7, font=MONO)

    body += block(196, 56, 76, 62, "Q", "NQ × DK", colour=TEAL, op=0.16)
    body += block(196, 174, 76, 62, "K", "NX × DK", colour=INDIGO, op=0.16)
    body += block(196, 272, 76, 62, "V", "NX × DV", colour=GREEN, op=0.18)

    body += arrow(272, 87, 352, 130, op=0.6)
    body += arrow(272, 205, 352, 168, op=0.6)
    body += grid_block(356, 96, 130, 110, 5, 4, lambda r, c: 0.10 + 0.06 * ((r + c) % 3), INDIGO)
    body += text(421, 84, "E = QKᵀ / √DK", size=11.5, op=0.85, font=MONO)
    body += text(421, 224, "NQ × NX", size=10, op=0.6, font=MONO)

    body += arrow(488, 151, 556, 151)
    body += text(522, 142, "softmax", size=10.5, op=0.7)

    weights = [
        [0.06, 0.55, 0.22, 0.10, 0.07],
        [0.40, 0.12, 0.08, 0.30, 0.10],
        [0.09, 0.14, 0.60, 0.09, 0.08],
        [0.20, 0.20, 0.12, 0.13, 0.35],
    ]
    body += grid_block(558, 96, 130, 110, 5, 4, lambda r, c: 0.12 + 1.25 * weights[r][c], GREEN)
    body += text(623, 84, "A = softmax(E)", size=11.5, op=0.85, font=MONO)
    body += arrow(566, 226, 680, 226, op=0.55)
    body += text(623, 244, "each row sums to one", size=10, op=0.6)

    body += arrow(690, 151, 764, 151)
    body += line(234, 334, 234, 352, stroke=GREEN, sw=1.4, op=0.7)
    body += line(234, 352, 800, 352, stroke=GREEN, sw=1.4, op=0.7)
    body += arrow(800, 352, 800, 188, stroke=GREEN, op=0.7)
    body += block(766, 120, 76, 62, "Y", "NQ × DV", colour=YELLOW, op=0.4)

    body += text(SH_W / 2, 30, "two matrix multiplies, with a softmax between them", size=11.5, op=0.6)

    return wrap(
        SH_W,
        SH_H,
        "attnshapes",
        "The attention layer with the shape of every intermediate marked",
        "Query inputs at top left and data inputs below them. Learned projections "
        "produce Q, K and V. Q and K meet in a grid of scores of shape NQ by NX, a "
        "softmax normalizes each row of that grid into a distribution over inputs, "
        "and multiplying by V, carried along a green rail across the bottom, gives "
        "one output vector per query.",
        body,
    )


# ---------------------------------------------------------------------------
# 5. Masked self-attention.
# ---------------------------------------------------------------------------

MK_TOKENS = ["attention", "is", "very", "cool"]
MK_CELL = 54.0
MK_X0, MK_Y0 = 150.0, 92.0
MK_GAP = 92.0
MK_X1 = MK_X0 + MK_CELL * 4 + MK_GAP
MK_W, MK_H = MK_X1 + MK_CELL * 4 + 34.0, MK_Y0 + MK_CELL * 4 + 74.0


def masked_weights():
    """Row-wise softmax of a fixed score pattern under the causal mask."""
    scores = [[0.9 - 0.35 * abs(i - j) for j in range(4)] for i in range(4)]
    rows = []
    for i in range(4):
        raw = [math.exp(scores[i][j]) if j <= i else 0.0 for j in range(4)]
        total = sum(raw)
        rows.append([v / total for v in raw])
    return rows


def fig_masking():
    """The causal mask before the softmax, and the zeros it produces after."""
    rows = masked_weights()
    body = ""
    for x0, title in ((MK_X0, "scores E"), (MK_X1, "weights A")):
        body += text(x0 + MK_CELL * 2, MK_Y0 - 34, title, size=12, op=0.8, font=MONO)
        for j, w in enumerate(MK_TOKENS):
            body += text(x0 + MK_CELL * (j + 0.5), MK_Y0 - 12, w, size=9.5, op=0.55)

    for j, w in enumerate(MK_TOKENS):
        body += text(MK_X0 - 10, MK_Y0 + MK_CELL * (j + 0.5) + 4, w, size=10.5, anchor="end", op=0.75)
    body += text(MK_X0 - 10, MK_Y0 - 34, "query ↓", size=10, anchor="end", op=0.5)

    for i in range(4):
        for j in range(4):
            allowed = j <= i
            body += rect(
                MK_X0 + MK_CELL * j,
                MK_Y0 + MK_CELL * i,
                MK_CELL,
                MK_CELL,
                fill=INDIGO if allowed else INK,
                op=0.14 if allowed else 0.04,
                stroke=INK,
                sw=0.9,
                so=0.35,
            )
            body += text(
                MK_X0 + MK_CELL * (j + 0.5),
                MK_Y0 + MK_CELL * (i + 0.5) + 4.5,
                f"E{i + 1}{j + 1}" if allowed else "−∞",
                size=11 if allowed else 13,
                op=0.85 if allowed else 0.5,
                font=MONO if allowed else None,
            )
            v = rows[i][j]
            body += rect(
                MK_X1 + MK_CELL * j,
                MK_Y0 + MK_CELL * i,
                MK_CELL,
                MK_CELL,
                fill=ramp(min(1.0, v * 1.6)) if allowed else INK,
                op=1.0 if allowed else 0.04,
                stroke="#ffffff" if allowed else INK,
                sw=0.9,
                so=0.5 if allowed else 0.35,
            )
            if not allowed:
                body += text(MK_X1 + MK_CELL * (j + 0.5), MK_Y0 + MK_CELL * (i + 0.5) + 4.5, "0", size=13, op=0.45)

    mid = (MK_X0 + MK_CELL * 4 + MK_X1) / 2
    body += arrow(MK_X0 + MK_CELL * 4 + 16, MK_Y0 + MK_CELL * 2, MK_X1 - 16, MK_Y0 + MK_CELL * 2)
    body += text(mid, MK_Y0 + MK_CELL * 2 - 12, "softmax", size=11, op=0.7)
    body += text(mid, MK_Y0 + MK_CELL * 2 + 26, "per row", size=10, op=0.5)

    body += text(
        MK_W / 2, MK_Y0 + MK_CELL * 4 + 34, "each position blends only the values at or before it", size=11, op=0.6
    )

    return wrap(
        MK_W,
        MK_H,
        "mask",
        "Masked self-attention: the causal mask before the softmax and the zeros after it",
        "Two four by four grids side by side. In the left grid, rows index the "
        "querying position and columns the position being looked at; cells on and "
        "below the diagonal hold scores and cells above it hold minus infinity. A "
        "softmax turns the left grid into the right one, where the masked cells "
        "are exactly zero and the remaining weights in each row are shaded by "
        "size and sum to one.",
        body,
    )


# ---------------------------------------------------------------------------
# 6. Three ways of processing a sequence.
# ---------------------------------------------------------------------------

TW_W, TW_H = 900.0, 320.0
TW_PANEL = 296.0
TW_N = 5
TW_STEP = 52.0
TW_IN_Y, TW_MID_Y, TW_OUT_Y = 196.0, 154.0, 104.0
TW_R = 8.0

TW_NOTES = [
    (
        "Recurrent",
        ["state passed along a chain", "O(N) compute, O(N) memory", "N sequential steps: not parallel"],
    ),
    (
        "Convolution",
        ["mixes a local neighbourhood", "parallel across positions", "distance costs depth"],
    ),
    (
        "Self-attention",
        ["every output sees every input", "parallel: four matrix multiplies", "O(N²) compute"],
    ),
]


def fig_three_ways():
    """Connectivity of a recurrent layer, a convolution and self-attention."""
    body = ""
    for p in range(3):
        ox = 12.0 + p * TW_PANEL
        xs = [ox + 32.0 + i * TW_STEP for i in range(TW_N)]
        body += text(ox + TW_PANEL / 2 - 12, 40, TW_NOTES[p][0], size=13, weight="600")

        if p == 0:
            for i, x in enumerate(xs):
                body += line(x, TW_IN_Y - TW_R, x, TW_MID_Y + TW_R, op=0.45)
                body += line(x, TW_MID_Y - TW_R, x, TW_OUT_Y + TW_R, op=0.45)
                if i:
                    body += arrow(xs[i - 1] + TW_R, TW_MID_Y, x - TW_R, TW_MID_Y, sw=1.5, op=0.85, stroke=TEAL)
                body += circle(x, TW_MID_Y, TW_R, fill=TEAL, op=0.25, stroke=TEAL, so=0.7)
            body += text(
                ox + TW_PANEL / 2 - 12, TW_MID_Y - 30, "four steps from first to last", size=10, fill=TEAL, op=0.9
            )
        elif p == 1:
            for i, x in enumerate(xs):
                for j in range(max(0, i - 1), min(TW_N, i + 2)):
                    body += line(
                        xs[j], TW_IN_Y - TW_R, x, TW_OUT_Y + TW_R, op=0.5, stroke=TEAL if i == 2 else "currentColor"
                    )
        else:
            for i, x in enumerate(xs):
                for j in range(TW_N):
                    body += line(
                        xs[j],
                        TW_IN_Y - TW_R,
                        x,
                        TW_OUT_Y + TW_R,
                        op=0.55 if i == 2 else 0.16,
                        stroke=TEAL if i == 2 else "currentColor",
                    )

        for x in xs:
            body += circle(x, TW_IN_Y, TW_R, fill=INDIGO, op=0.35, stroke=INDIGO, so=0.6)
            body += circle(x, TW_OUT_Y, TW_R, fill=YELLOW, op=0.55, stroke=INK, so=0.4)

        body += text(ox + 14, TW_OUT_Y + 4, "y", size=11, anchor="end", op=0.6)
        body += text(ox + 14, TW_IN_Y + 4, "x", size=11, anchor="end", op=0.6)

        for k, note in enumerate(TW_NOTES[p][1]):
            body += text(ox + 20, 250 + k * 18, note, size=10.5, anchor="start", op=0.65)

        if p:
            body += line(ox - 6, 56, ox - 6, 290, op=0.15)

    return wrap(
        TW_W,
        TW_H,
        "threeways",
        "How information reaches a distant position under recurrence, convolution and attention",
        "Three panels, each with a row of five indigo input nodes below a row of "
        "five yellow output nodes. In the first the inputs feed a horizontal chain "
        "of teal state nodes, so reaching the last output from the first input "
        "takes four steps. In the second each output connects only to its three "
        "nearest inputs. In the third every output connects to every input, "
        "twenty-five edges in one layer.",
        body,
    )


# ---------------------------------------------------------------------------
# 7. The transformer block.
# ---------------------------------------------------------------------------

TB_W, TB_H = 528.0, 520.0
TB_L, TB_R = 96.0, 496.0
TB_COLS = [128.0, 226.0, 324.0, 422.0]
TB_BANDH = 38.0
TB_Y = {
    "in": 466.0,
    "attn": 388.0,
    "add1": 336.0,
    "norm1": 268.0,
    "mlp": 190.0,
    "add2": 138.0,
    "norm2": 70.0,
    "out": 34.0,
}


def fig_transformer_block():
    """Self-attention and MLP sublayers, each inside a residual connection."""
    body = ""
    for x in TB_COLS:
        body += text(x, TB_Y["in"], "x", size=13, op=0.85)
        body += text(x + 8, TB_Y["in"] + 5, str(TB_COLS.index(x) + 1), size=9, op=0.6)
        body += text(x, TB_Y["out"], "y", size=13, op=0.85)
        body += text(x + 8, TB_Y["out"] + 5, str(TB_COLS.index(x) + 1), size=9, op=0.6)
        body += arrow(x, TB_Y["in"] - 14, x, TB_Y["attn"] + TB_BANDH / 2 + 4, op=0.5)
        body += arrow(x, TB_Y["norm2"] - TB_BANDH / 2 - 2, x, TB_Y["out"] + 12, op=0.5)
        body += arrow(x, TB_Y["norm1"] - TB_BANDH / 2 - 2, x, TB_Y["mlp"] + 24, op=0.5)
        body += rect(x - 40, TB_Y["mlp"] - 24, 80, 48, fill=GREEN, op=0.16, stroke=GREEN, sw=1.1, so=0.6, rx=3)
        body += text(x, TB_Y["mlp"] + 5, "MLP", size=12)
        body += arrow(x, TB_Y["mlp"] - 26, x, TB_Y["add2"] + 15, op=0.5)
        body += arrow(x, TB_Y["attn"] - TB_BANDH / 2 - 2, x, TB_Y["add1"] + 15, op=0.5)
        body += arrow(x, TB_Y["add1"] - 15, x, TB_Y["norm1"] + TB_BANDH / 2 + 4, op=0.5)
        body += arrow(x, TB_Y["add2"] - 15, x, TB_Y["norm2"] + TB_BANDH / 2 + 4, op=0.5)

    for key, label, colour, op in (
        ("attn", "Multi-head self-attention", INDIGO, 0.18),
        ("norm1", "Layer normalization", TEAL, 0.12),
        ("norm2", "Layer normalization", TEAL, 0.12),
    ):
        y = TB_Y[key] - TB_BANDH / 2
        body += rect(TB_L - 32, y, TB_R - TB_L + 32, TB_BANDH, fill=colour, op=op, stroke=colour, sw=1.1, so=0.6, rx=3)
        body += text((TB_L - 32 + TB_R) / 2, y + TB_BANDH / 2 + 4.5, label, size=12)

    for key in ("add1", "add2"):
        body += op_node((TB_L - 32 + TB_R) / 2, TB_Y[key], "+")
        y = TB_Y[key]
        body += line(TB_L - 32, y, (TB_L - 32 + TB_R) / 2 - 13, y, op=0.0)

    # residual paths, up the left margin
    for src, dst in (("in", "add1"), ("norm1", "add2")):
        y0 = TB_Y[src] - 16 if src == "in" else TB_Y[src] - TB_BANDH / 2 - 2
        body += line(TB_COLS[0], y0, 44, y0, op=0.55, stroke=PURPLE, sw=1.4)
        body += line(44, y0, 44, TB_Y[dst], op=0.55, stroke=PURPLE, sw=1.4)
        body += arrow(44, TB_Y[dst], (TB_L - 32 + TB_R) / 2 - 15, TB_Y[dst], op=0.7, stroke=PURPLE)

    body += text((TB_L - 32 + TB_R) / 2, 500, "self-attention is the only place the vectors meet", size=11, op=0.6)
    body += text(52, 118, "residual", size=10, anchor="start", fill=PURPLE, op=0.85)

    return wrap(
        TB_W,
        TB_H,
        "tblock",
        "The transformer block: self-attention and a per-vector MLP, each in a residual connection",
        "Four input vectors enter at the bottom and pass through a full-width "
        "multi-head self-attention band, an addition node, and a layer "
        "normalization band. Each vector then passes through its own MLP box, a "
        "second addition and a second normalization, leaving as four output "
        "vectors. Two purple residual paths run up the left margin, one around "
        "each sublayer.",
        body,
    )


# ---------------------------------------------------------------------------
# 8. The Vision Transformer.
# ---------------------------------------------------------------------------

VT_W, VT_H = 900.0, 320.0
VT_IMG_X, VT_IMG_Y, VT_IMG = 34.0, 96.0, 132.0
VT_GRID = 4


def fig_vit():
    """An image cut into patches, projected, and fed to transformer blocks."""
    body = ""
    body += rect(VT_IMG_X, VT_IMG_Y, VT_IMG, VT_IMG, fill=INDIGO, op=0.10, stroke=INK, sw=1.1, so=0.5)
    step = VT_IMG / VT_GRID
    for k in range(1, VT_GRID):
        body += line(VT_IMG_X + k * step, VT_IMG_Y, VT_IMG_X + k * step, VT_IMG_Y + VT_IMG, op=0.35)
        body += line(VT_IMG_X, VT_IMG_Y + k * step, VT_IMG_X + VT_IMG, VT_IMG_Y + k * step, op=0.35)
    body += rect(VT_IMG_X + step, VT_IMG_Y + step, step, step, fill=YELLOW, op=0.5, stroke=INK, sw=1.4, so=0.7)
    body += text(VT_IMG_X + VT_IMG / 2, VT_IMG_Y - 14, "224 × 224 × 3", size=11, op=0.7, font=MONO)
    body += text(VT_IMG_X + VT_IMG / 2, VT_IMG_Y + VT_IMG + 22, "cut into 16 × 16 patches", size=11, op=0.6)

    body += arrow(VT_IMG_X + VT_IMG + 8, VT_IMG_Y + VT_IMG / 2, 218, VT_IMG_Y + VT_IMG / 2)
    body += rect(222, 128, 60, 60, fill=YELLOW, op=0.5, stroke=INK, sw=1.2, so=0.6)
    body += text(252, 118, "one patch", size=10.5, op=0.65)
    body += text(252, 204, "768 numbers", size=10.5, op=0.65, font=MONO)

    body += arrow(288, VT_IMG_Y + VT_IMG / 2, 344, VT_IMG_Y + VT_IMG / 2)
    body += text(316, VT_IMG_Y + VT_IMG / 2 - 12, "linear", size=10.5, op=0.7)
    body += text(316, VT_IMG_Y + VT_IMG / 2 + 22, "768 → D", size=10, op=0.55, font=MONO)

    for k in range(5):
        x = 350 + k * 26
        body += rect(x, 118, 18, 82, fill=TEAL, op=0.22 + 0.05 * k, stroke=TEAL, sw=1.0, so=0.6, rx=2)
        body += circle(x + 9, 212, 5, fill=PURPLE, op=0.55, stroke=PURPLE, so=0.6)
    body += text(410, 104, "N × D tokens", size=11, op=0.7, font=MONO)
    body += text(410, 232, "+ positional encoding", size=10.5, fill=PURPLE, op=0.9)

    body += arrow(488, 160, 536, 160)
    for k in range(3):
        body += rect(540 + k * 6, 106 + k * 6, 168, 108, fill=INDIGO, op=0.10, stroke=INDIGO, sw=1.1, so=0.55, rx=4)
    body += text(636, 166, "Transformer", size=12.5)
    body += text(636, 186, "blocks × L, unmasked", size=10.5, op=0.6)

    body += arrow(720, 160, 762, 160)
    body += circle(786, 160, 22, fill=GREEN, op=0.2, stroke=GREEN, sw=1.1, so=0.6)
    body += text(786, 164, "pool", size=10.5)
    body += arrow(810, 160, 842, 160)
    body += rect(846, 120, 22, 80, fill=YELLOW, op=0.45, stroke=INK, sw=1.0, so=0.5, rx=2)
    body += text(857, 110, "C", size=11, op=0.75, font=MONO)
    body += text(857, 216, "scores", size=10.5, op=0.6)

    return wrap(
        VT_W,
        VT_H,
        "vit",
        "The Vision Transformer: patches to tokens to transformer blocks to class scores",
        "An image on the left is divided by a grid into square patches, one of "
        "which is highlighted and enlarged. A linear map turns each flattened "
        "patch into a token vector; purple markers below the tokens stand for the "
        "positional encodings added to them. The tokens pass through a stack of "
        "transformer blocks, the outputs are pooled to a single vector, and a "
        "linear layer produces class scores.",
        body,
    )


FIGURES = {
    "diagram_bottleneck.svg": fig_bottleneck,
    "diagram_attention_step.svg": fig_attention_step,
    "diagram_alignment.svg": fig_alignment,
    "diagram_attention_shapes.svg": fig_attention_shapes,
    "diagram_masking.svg": fig_masking,
    "diagram_three_ways.svg": fig_three_ways,
    "diagram_transformer_block.svg": fig_transformer_block,
    "diagram_vit.svg": fig_vit,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print(f"wrote {OUT / name}")

    # The chapter quotes these; print them so the prose can be checked.
    n, heads, dtype_bytes = 100_000, 64, 2
    entries = heads * n * n
    print(
        f"attention weights at N={n:,}, H={heads}: {entries:.3g} entries, {entries * dtype_bytes / 1e12:.2f} TB at {dtype_bytes} bytes each"
    )

    patch, side, channels = 16, 224, 3
    print(
        f"ViT: {side}/{patch} = {side // patch} patches per side, {(side // patch) ** 2} tokens, {patch * patch * channels} numbers per patch"
    )
