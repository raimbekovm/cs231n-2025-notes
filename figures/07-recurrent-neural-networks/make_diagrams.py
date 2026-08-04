"""Generate the six SVG diagrams for the lecture 7 chapter.

Nothing here is traced from a slide. Five of the six are structural drawings —
the sequence shapes, the unrolled graph, the truncation boundary, the LSTM cell
and the captioning join — and their geometry is generated from the layout
constants below rather than placed by hand. The sixth is computed: the gradient
chart evaluates the geometric factor of @eq-jacobian-product and prints the
numbers the chapter quotes.

Conventions follow figures/06-cnn-architectures: currentColor for text and
rules, viridis for data, title/desc with namespaced ids, no background rect
(the theme supplies a light plate in both schemes).

Run from the repository root; rewrites all six files.
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
    polyline,
    rect,
    text,
    wrap,
)  # noqa: E402

OUT = pathlib.Path("figures/07-recurrent-neural-networks")


def path(d, fill="none", stroke="currentColor", sw=1.0, op=1.0, dash=None):
    """One path."""
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'  <path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-opacity="{op}"{da}/>\n'


def op_node(cx, cy, glyph, r=11.0, colour=INK):
    """A small circled operator: elementwise product or sum."""
    out = circle(cx, cy, r, fill="#ffffff", op=0.0, stroke=colour, sw=1.3, so=0.85)
    dy = 4.6 if glyph != "+" else 5.0
    return out + text(cx, cy + dy, glyph, size=14, fill=colour, op=0.9)


# ---------------------------------------------------------------------------
# 1. The five input/output shapes.
# ---------------------------------------------------------------------------

SS_W, SS_H = 900.0, 208.0
SS_PANEL = 156.0
SS_GAP = 18.0
SS_MARGIN = 24.0
SS_UNIT = 36.0  # one time step: 30 wide box plus 6 of gap
SS_BOX = 30.0
SS_YOUT, SS_YNET, SS_YIN = 34.0, 76.0, 118.0
SS_BH = 26.0
SS_NBH = 30.0

# (name, input steps, output steps, band span, example) on a four-unit grid.
SS_PANELS = [
    ("one to one", [1.5], [1.5], (1.0, 2.0), "image", "class label"),
    ("one to many", [0], [0, 1, 2, 3], (0, 3), "image", "caption"),
    ("many to one", [0, 1, 2, 3], [3], (0, 3), "video", "action class"),
    ("many to many", [0, 1], [2, 3], (0, 3), "sentence", "translation"),
    (
        "many to many, aligned",
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        (0, 3),
        "video",
        "label per frame",
    ),
]


def _ss_x(panel_x, unit):
    """Left edge of the box occupying `unit` in the panel starting at panel_x."""
    grid = 4 * SS_UNIT - (SS_UNIT - SS_BOX)  # width of a full four-unit row
    x0 = panel_x + (SS_PANEL - grid) / 2
    return x0 + unit * SS_UNIT


def _ss_panel(px, spec):
    name, ins, outs, band, ex_in, ex_out = spec
    out = text(px + SS_PANEL / 2, 18, name, size=12.5, op=0.9)

    b0 = _ss_x(px, band[0])
    b1 = _ss_x(px, band[1]) + SS_BOX
    out += rect(
        b0,
        SS_YNET,
        b1 - b0,
        SS_NBH,
        fill=INK,
        op=0.09,
        stroke=INK,
        sw=0.9,
        so=0.35,
        rx=3,
    )
    out += text((b0 + b1) / 2, SS_YNET + 20, "network", size=11, op=0.62)

    for u in ins:
        x = _ss_x(px, u)
        out += rect(
            x,
            SS_YIN,
            SS_BOX,
            SS_BH,
            fill=INDIGO,
            op=0.80,
            stroke=INDIGO,
            sw=0.9,
            so=0.9,
            rx=3,
        )
        out += arrow(
            x + SS_BOX / 2,
            SS_YIN - 3,
            x + SS_BOX / 2,
            SS_YNET + SS_NBH + 3,
            sw=1.2,
            op=0.55,
            head=5,
        )
    for u in outs:
        x = _ss_x(px, u)
        out += rect(
            x,
            SS_YOUT,
            SS_BOX,
            SS_BH,
            fill=TEAL,
            op=0.80,
            stroke=TEAL,
            sw=0.9,
            so=0.9,
            rx=3,
        )
        out += arrow(
            x + SS_BOX / 2,
            SS_YNET - 3,
            x + SS_BOX / 2,
            SS_YOUT + SS_BH + 3,
            sw=1.2,
            op=0.55,
            head=5,
        )

    cx = px + SS_PANEL / 2
    out += text(cx, 172, ex_in, size=11.5, op=0.72, style="italic")
    out += text(cx, 190, "→ " + ex_out, size=11.5, op=0.72, style="italic")
    return out


def fig_seq_shapes():
    """The five input/output shapes a sequence problem can take."""
    body = ""
    for i, spec in enumerate(SS_PANELS):
        body += _ss_panel(SS_MARGIN + i * (SS_PANEL + SS_GAP), spec)
    return wrap(
        SS_W,
        SS_H,
        "seqShapes",
        "The five input and output shapes a sequence problem can take",
        "Five panels. In each, indigo blocks below are inputs, teal blocks above "
        "are outputs, and a grey band between them is the network. One to one has "
        "a single input and output; one to many has one input and four outputs; "
        "many to one has four inputs and one output; many to many reads two "
        "inputs then writes two outputs; the aligned case pairs four inputs with "
        "four outputs.",
        body,
    )


# ---------------------------------------------------------------------------
# 2. The recurrence rolled and unrolled.
# ---------------------------------------------------------------------------

UN_W, UN_H = 900.0, 312.0
UN_STEPS = 4
UN_X0, UN_DX = 372.0, 128.0
UN_BW, UN_BH = 66.0, 46.0
UN_YBOX = 108.0
UN_YOUT = 56.0
UN_YIN = 186.0
UN_XBH = 24.0
UN_YRAIL = 254.0


def fig_unrolled():
    """The recurrence as written beside the same computation unrolled."""
    body = ""
    ybot = UN_YBOX + UN_BH

    # -- left: the loop as written -------------------------------------------
    lcx = 148.0
    body += text(lcx, 18, "as written", size=12.5, op=0.9)
    body += rect(
        lcx - UN_BW / 2,
        UN_YBOX,
        UN_BW,
        UN_BH,
        fill=PURPLE,
        op=0.16,
        stroke=PURPLE,
        sw=1.2,
        so=0.85,
        rx=4,
    )
    body += text(lcx, UN_YBOX + 29, "f", size=15, style="italic", op=0.95)
    body += text(lcx + 9, UN_YBOX + 33, "W", size=10, op=0.95)
    body += rect(
        lcx - 13,
        UN_YIN,
        26,
        22,
        fill=INDIGO,
        op=0.78,
        stroke=INDIGO,
        sw=0.9,
        so=0.9,
        rx=3,
    )
    body += text(lcx, UN_YIN + 16, "x", size=12, fill="#ffffff", style="italic")
    body += arrow(lcx, UN_YIN - 3, lcx, ybot + 3, sw=1.3, op=0.6, head=5.5)
    body += rect(
        lcx - 13, UN_YOUT, 26, 22, fill=TEAL, op=0.78, stroke=TEAL, sw=0.9, so=0.9, rx=3
    )
    body += text(lcx, UN_YOUT + 16, "y", size=12, fill="#ffffff", style="italic")
    body += arrow(lcx, UN_YBOX - 3, lcx, UN_YOUT + 22 + 3, sw=1.3, op=0.6, head=5.5)
    # the feedback loop
    body += path(
        f"M {lcx + UN_BW / 2} {UN_YBOX + 14} H {lcx + 64} V {UN_YBOX + 39} H {lcx + UN_BW / 2 + 7}",
        stroke=GREEN,
        sw=1.6,
        op=0.95,
    )
    body += arrow(
        lcx + UN_BW / 2 + 9,
        UN_YBOX + 39,
        lcx + UN_BW / 2 + 1,
        UN_YBOX + 39,
        sw=1.6,
        op=0.95,
        head=6,
        stroke=GREEN,
    )
    body += text(
        lcx + 74, UN_YBOX + 30, "h", size=12, fill=GREEN, anchor="start", style="italic"
    )

    # -- right: the same thing unrolled --------------------------------------
    cxs = [UN_X0 + i * UN_DX for i in range(UN_STEPS)]
    body += text((cxs[0] + cxs[-1]) / 2, 18, "as executed", size=12.5, op=0.9)

    body += text(
        cxs[0] - UN_BW / 2 - 34,
        UN_YBOX + 29,
        "h",
        size=12.5,
        fill=GREEN,
        style="italic",
    )
    body += text(cxs[0] - UN_BW / 2 - 27, UN_YBOX + 33, "0", size=9.5, fill=GREEN)
    body += arrow(
        cxs[0] - UN_BW / 2 - 18,
        UN_YBOX + 25,
        cxs[0] - UN_BW / 2 - 3,
        UN_YBOX + 25,
        sw=1.5,
        op=0.9,
        head=6,
        stroke=GREEN,
    )

    for i, cx in enumerate(cxs):
        body += rect(
            cx - UN_BW / 2,
            UN_YBOX,
            UN_BW,
            UN_BH,
            fill=PURPLE,
            op=0.16,
            stroke=PURPLE,
            sw=1.2,
            so=0.85,
            rx=4,
        )
        body += text(cx, UN_YBOX + 29, "f", size=15, style="italic", op=0.95)
        body += text(cx + 9, UN_YBOX + 33, "W", size=10, op=0.95)

        body += rect(
            cx - 13,
            UN_YIN,
            26,
            22,
            fill=INDIGO,
            op=0.78,
            stroke=INDIGO,
            sw=0.9,
            so=0.9,
            rx=3,
        )
        body += text(cx - 3, UN_YIN + 16, "x", size=12, fill="#ffffff", style="italic")
        body += text(cx + 6, UN_YIN + 19, str(i + 1), size=9, fill="#ffffff")
        body += arrow(cx, UN_YIN - 3, cx, ybot + 3, sw=1.3, op=0.6, head=5.5)

        body += rect(
            cx - 13,
            UN_YOUT,
            26,
            22,
            fill=TEAL,
            op=0.78,
            stroke=TEAL,
            sw=0.9,
            so=0.9,
            rx=3,
        )
        body += text(cx - 3, UN_YOUT + 16, "y", size=12, fill="#ffffff", style="italic")
        body += text(cx + 6, UN_YOUT + 19, str(i + 1), size=9, fill="#ffffff")
        body += arrow(cx, UN_YBOX - 3, cx, UN_YOUT + 25, sw=1.3, op=0.6, head=5.5)

        if i < UN_STEPS - 1:
            x0, x1 = cx + UN_BW / 2 + 3, cxs[i + 1] - UN_BW / 2 - 3
            body += arrow(
                x0, UN_YBOX + 25, x1, UN_YBOX + 25, sw=1.5, op=0.9, head=6, stroke=GREEN
            )
            body += text(
                (x0 + x1) / 2 - 4,
                UN_YBOX + 18,
                "h",
                size=11.5,
                fill=GREEN,
                style="italic",
            )
            body += text(
                (x0 + x1) / 2 + 3, UN_YBOX + 21, str(i + 1), size=8.5, fill=GREEN
            )

    # the shared-weight rail
    rail0, rail1 = cxs[0] + UN_XBH, cxs[-1] + UN_XBH
    body += line(rail0, UN_YRAIL, rail1, UN_YRAIL, stroke=YELLOW, sw=1.8, op=1.0)
    for cx in cxs:
        body += arrow(
            cx + UN_XBH,
            UN_YRAIL,
            cx + UN_XBH,
            ybot + 2,
            sw=1.3,
            op=0.95,
            head=5.5,
            stroke=YELLOW,
            dash="4 3",
        )
    body += rect(
        rail0 - 78,
        UN_YRAIL - 15,
        46,
        30,
        fill=YELLOW,
        op=0.28,
        stroke=YELLOW,
        sw=1.4,
        so=1.0,
        rx=4,
    )
    body += text(rail0 - 55, UN_YRAIL + 5, "W", size=15, op=0.95)
    body += line(rail0 - 32, UN_YRAIL, rail0, UN_YRAIL, stroke=YELLOW, sw=1.8, op=1.0)
    body += text(
        (rail0 + rail1) / 2,
        UN_YRAIL + 30,
        "one set of weights, consumed at every step",
        size=11.5,
        op=0.66,
    )

    return wrap(
        UN_W,
        UN_H,
        "unrolled",
        "The recurrence as written and the same computation unrolled over four steps",
        "On the left a single block with a feedback arrow labelled h. On the right "
        "four copies of the same block in a row, the hidden state passing from each "
        "to the next, an input entering and an output leaving at every step, and a "
        "yellow rail along the bottom connecting one weight box W to all four "
        "blocks.",
        body,
    )


# ---------------------------------------------------------------------------
# 3. Truncated backpropagation through time.
# ---------------------------------------------------------------------------

TB_W, TB_H = 900.0, 246.0
TB_N = 9
TB_CHUNK = 3
TB_X0, TB_DX = 88.0, 90.0
TB_BW, TB_BH = 42.0, 34.0
TB_YBOX = 96.0
TB_YGRAD = 172.0


def fig_truncated_bptt():
    """Chunked training: state crosses the boundary, gradient does not."""
    body = ""
    cxs = [TB_X0 + i * TB_DX for i in range(TB_N)]
    ybot = TB_YBOX + TB_BH

    # chunk brackets and boundaries
    for c in range(TB_N // TB_CHUNK):
        a, b = cxs[c * TB_CHUNK], cxs[c * TB_CHUNK + TB_CHUNK - 1]
        x0, x1 = a - TB_BW / 2, b + TB_BW / 2
        body += line(x0, 58, x1, 58, sw=1.1, op=0.5)
        body += line(x0, 58, x0, 64, sw=1.1, op=0.5)
        body += line(x1, 58, x1, 64, sw=1.1, op=0.5)
        body += text((x0 + x1) / 2, 50, f"chunk {c + 1}", size=12, op=0.85)
        body += text(
            (x0 + x1) / 2, 30, "forward, backward, one update", size=10.5, op=0.55
        )
        if c:
            bx = (cxs[c * TB_CHUNK - 1] + cxs[c * TB_CHUNK]) / 2
            body += line(bx, 66, bx, TB_YGRAD + 22, sw=1.1, op=0.45, dash="5 4")

    for i, cx in enumerate(cxs):
        body += rect(
            cx - TB_BW / 2,
            TB_YBOX,
            TB_BW,
            TB_BH,
            fill=PURPLE,
            op=0.16,
            stroke=PURPLE,
            sw=1.1,
            so=0.8,
            rx=3,
        )
        body += text(cx - 4, TB_YBOX + 22, "h", size=12.5, style="italic", op=0.95)
        body += text(cx + 4, TB_YBOX + 25, str(i + 1), size=9, op=0.95)
        if i < TB_N - 1:
            body += arrow(
                cx + TB_BW / 2 + 2,
                TB_YBOX + TB_BH / 2,
                cxs[i + 1] - TB_BW / 2 - 2,
                TB_YBOX + TB_BH / 2,
                sw=1.6,
                op=0.95,
                head=6,
                stroke=GREEN,
            )
        # the gradient, right to left, never leaving its chunk
        if i % TB_CHUNK:
            body += arrow(
                cx - 3,
                TB_YGRAD,
                cxs[i - 1] + 3,
                TB_YGRAD,
                sw=1.5,
                op=0.95,
                head=6,
                stroke=INDIGO,
                dash="5 3",
            )
            body += line(cx, ybot + 2, cx, TB_YGRAD - 8, sw=1.0, op=0.35, dash="3 3")
        else:
            body += line(cx, ybot + 2, cx, TB_YGRAD - 8, sw=1.0, op=0.2, dash="3 3")

    body += text(
        TB_X0 - TB_BW / 2 - 12, TB_YBOX + 22, "…", size=14, op=0.4, anchor="end"
    )
    body += text(
        cxs[-1] + TB_BW / 2 + 12, TB_YBOX + 22, "…", size=14, op=0.4, anchor="start"
    )

    body += line(112, 218, 140, 218, stroke=GREEN, sw=1.8, op=0.95)
    body += text(
        148,
        222,
        "hidden state, carried across every boundary",
        size=11.5,
        op=0.72,
        anchor="start",
    )
    body += line(500, 218, 528, 218, stroke=INDIGO, sw=1.6, op=0.95, dash="5 3")
    body += text(
        536,
        222,
        "gradient, stopped at the boundary",
        size=11.5,
        op=0.72,
        anchor="start",
    )

    return wrap(
        TB_W,
        TB_H,
        "truncBptt",
        "Truncated backpropagation through time",
        "Nine hidden states in a row, grouped into three chunks. Green arrows run "
        "left to right through all nine, crossing the chunk boundaries. Blue dashed "
        "gradient arrows run right to left but stop at the first state of each "
        "chunk, never crossing a boundary.",
        body,
    )


# ---------------------------------------------------------------------------
# 4. What repeated multiplication does. Computed.
# ---------------------------------------------------------------------------

GN_W, GN_H = 840.0, 386.0
GN_L, GN_R, GN_T, GN_B = 96.0, 796.0, 52.0, 300.0
GN_KMAX = 150
GN_LO, GN_HI = -8, 4  # decades of the y axis

# (largest singular value, mean tanh', label, colour, dashed, label k, label dy)
GN_CURVES = [
    (1.05, 1.0, "σ₁ = 1.05", PURPLE, False, 150, -11),
    (1.00, 1.0, "σ₁ = 1.00", TEAL, False, 150, -11),
    (0.95, 1.0, "σ₁ = 0.95", INDIGO, False, 150, -11),
    (1.05, 0.85, "σ₁ = 1.05, with tanh′ included", GREEN, True, 100, -10),
]


def gn_factor(sigma, tanhp):
    """Per-step multiplier of @eq-jacobian-product under the stated assumptions."""
    return sigma * tanhp


def _gn_x(k):
    return GN_L + (k / GN_KMAX) * (GN_R - GN_L)


def _gn_y(v):
    lg = math.log10(v)
    lg = max(GN_LO, min(GN_HI, lg))
    return GN_B - (lg - GN_LO) / (GN_HI - GN_LO) * (GN_B - GN_T)


def fig_gradient_norm():
    """Gradient norm against the number of steps, computed, on a log axis."""
    body = ""
    for d in range(GN_LO, GN_HI + 1, 2):
        y = _gn_y(10.0**d)
        body += line(GN_L, y, GN_R, y, sw=0.8, op=0.16 if d else 0.34)
        lab = "1" if d == 0 else f"10<tspan dy='-4' font-size='9'>{d}</tspan>"
        body += f'  <text x="{GN_L - 12:.1f}" y="{y + 4:.1f}" font-size="11.5" fill="currentColor" fill-opacity="0.7" text-anchor="end">{lab}</text>\n'
    for k in range(0, GN_KMAX + 1, 25):
        x = _gn_x(k)
        body += line(x, GN_B, x, GN_B + 5, sw=0.9, op=0.5)
        body += text(x, GN_B + 20, str(k), size=11.5, op=0.7)

    body += line(GN_L, GN_T, GN_L, GN_B, sw=1.1, op=0.55)
    body += line(GN_L, GN_B, GN_R, GN_B, sw=1.1, op=0.55)
    body += text(
        (GN_L + GN_R) / 2,
        GN_B + 42,
        "steps back through the sequence, k",
        size=12.5,
        op=0.8,
    )
    body += (
        f'  <text x="28" y="{(GN_T + GN_B) / 2:.1f}" font-size="12.5" fill="currentColor"'
        f' fill-opacity="0.8" text-anchor="middle"'
        f' transform="rotate(-90 28 {(GN_T + GN_B) / 2:.1f})">relative gradient norm</text>\n'
    )

    for sigma, tanhp, label, colour, dashed, lk, ldy in GN_CURVES:
        f = gn_factor(sigma, tanhp)
        pts = []
        for k in range(GN_KMAX + 1):
            v = f**k
            if v < 10.0**GN_LO:
                pts.append((_gn_x(k), _gn_y(10.0**GN_LO)))
                break
            pts.append((_gn_x(k), _gn_y(v)))
        body += polyline(
            pts, stroke=colour, sw=2.0, op=0.95, dash="6 4" if dashed else None
        )
        anchor = "start" if dashed else "end"
        dx = 8 if dashed else -4
        body += text(
            _gn_x(lk) + dx,
            _gn_y(f**lk) + ldy,
            label,
            size=11.5,
            fill=colour,
            anchor=anchor,
        )

    # the two numbers the chapter quotes
    body += text(
        GN_L,
        GN_B + 66,
        f"after 150 steps: ×{round(1.05**150)} at σ₁ = 1.05,"
        f"  ÷{round(1 / 0.95**150)} at σ₁ = 0.95,"
        f"  ÷{(1 / (1.05 * 0.85) ** 150) / 1e6:.0f} million once tanh′ is included",
        size=11.5,
        op=0.72,
        anchor="start",
    )

    return wrap(
        GN_W,
        GN_H,
        "gradNorm",
        "The norm of a gradient after k applications of the same matrix",
        "A logarithmic chart. A curve for a largest singular value of 1.05 rises "
        "steeply, one for 1.00 stays flat, one for 0.95 falls steeply, and a dashed "
        "curve including the derivative of tanh falls fastest of all.",
        body,
    )


# ---------------------------------------------------------------------------
# 5. One LSTM step.
# ---------------------------------------------------------------------------

LS_W, LS_H = 900.0, 350.0
LS_RAIL = 84.0


def fig_lstm_cell():
    """One LSTM step, with the cell state drawn as an uninterrupted path."""
    body = ""
    wx0, wx1, wy0, wy1 = 178.0, 240.0, 196.0, 282.0
    gate_x = 306.0  # where the four gate letters sit
    wire_x = 354.0  # where their outgoing wires begin, clear of the labels
    gates = [
        ("f", 206.0, "σ"),
        ("i", 232.0, "σ"),
        ("g", 258.0, "tanh"),
        ("o", 284.0, "σ"),
    ]
    # Everything on the rail is ordered left to right so nothing doubles back.
    x_mul_f, x_mul_ig, x_add = 384.0, 430.0, 496.0
    x_branch, x_tanh, x_mul_o = 566.0, 636.0, 724.0
    y_mul_ig, y_branch = 232.0, 178.0
    x_end = 800.0

    # -- the cell-state rail --------------------------------------------------
    body += text(44, LS_RAIL + 5, "c", size=14, style="italic", op=0.95, anchor="end")
    body += text(46, LS_RAIL + 9, "t−1", size=9, op=0.95, anchor="start")
    body += line(76, LS_RAIL, x_mul_f - 11, LS_RAIL, stroke=PURPLE, sw=2.4, op=0.95)
    body += line(
        x_mul_f + 11, LS_RAIL, x_add - 11, LS_RAIL, stroke=PURPLE, sw=2.4, op=0.95
    )
    body += arrow(
        x_add + 11, LS_RAIL, x_end, LS_RAIL, stroke=PURPLE, sw=2.4, op=0.95, head=7
    )
    body += text(
        x_end + 10, LS_RAIL + 5, "c", size=14, style="italic", op=0.95, anchor="start"
    )
    body += text(x_end + 20, LS_RAIL + 9, "t", size=9, op=0.95, anchor="start")
    body += text(
        (x_add + x_end) / 2,
        LS_RAIL - 14,
        "no matrix multiply, no activation on this path",
        size=11.5,
        op=0.6,
    )
    body += op_node(x_mul_f, LS_RAIL, "⊙", colour=PURPLE)
    body += op_node(x_add, LS_RAIL, "+", colour=PURPLE)

    # -- one matrix multiply, four gates --------------------------------------
    body += rect(
        wx0,
        wy0,
        wx1 - wx0,
        wy1 - wy0,
        fill=YELLOW,
        op=0.26,
        stroke=YELLOW,
        sw=1.4,
        so=1.0,
        rx=4,
    )
    body += text((wx0 + wx1) / 2, (wy0 + wy1) / 2 + 6, "W", size=17, op=0.95)
    body += text((wx0 + wx1) / 2, wy1 + 16, "4H × (H+D)", size=10.5, op=0.6)

    body += text(84, 222, "h", size=13, style="italic", op=0.9, anchor="end")
    body += text(86, 226, "t−1", size=9, op=0.9, anchor="start")
    body += text(84, 262, "x", size=13, style="italic", op=0.9, anchor="end")
    body += text(86, 266, "t", size=9, op=0.9, anchor="start")
    body += path("M 112 218 H 146 V 236", stroke=INK, sw=1.2, op=0.5)
    body += path("M 112 258 H 146 V 242", stroke=INK, sw=1.2, op=0.5)
    body += text(146, 210, "stack", size=10.5, op=0.55)
    body += arrow(146, 239, wx0 - 3, 239, sw=1.3, op=0.6, head=5.5)

    for name, gy, act in gates:
        body += line(wx1, 239, wx1 + 16, 239, sw=1.0, op=0.35)
        body += line(wx1 + 16, 239, wx1 + 16, gy, sw=1.0, op=0.35)
        body += arrow(wx1 + 16, gy, gate_x - 15, gy, sw=1.2, op=0.55, head=5)
        body += text(gate_x, gy + 4, name, size=13.5, style="italic", op=0.95)
        body += text(gate_x + 16, gy + 4, act, size=10, op=0.5, anchor="start")

    # forget gate rises into the multiply on the rail
    body += path(
        f"M {wire_x} 206 H {x_mul_f} V {LS_RAIL + 13}", stroke=TEAL, sw=1.5, op=0.9
    )
    body += arrow(
        x_mul_f,
        LS_RAIL + 24,
        x_mul_f,
        LS_RAIL + 13,
        stroke=TEAL,
        sw=1.5,
        op=0.9,
        head=6,
    )

    # input gate and candidate meet, then rise into the sum
    body += path(f"M {wire_x} 232 H {x_mul_ig - 11}", stroke=TEAL, sw=1.5, op=0.9)
    body += path(
        f"M {wire_x} 258 H {x_mul_ig} V {y_mul_ig + 11}", stroke=TEAL, sw=1.5, op=0.9
    )
    body += op_node(x_mul_ig, y_mul_ig, "⊙", colour=TEAL)
    body += path(
        f"M {x_mul_ig + 11} {y_mul_ig} H {x_add} V {LS_RAIL + 13}",
        stroke=TEAL,
        sw=1.5,
        op=0.9,
    )
    body += arrow(
        x_add, LS_RAIL + 24, x_add, LS_RAIL + 13, stroke=TEAL, sw=1.5, op=0.9, head=6
    )

    # a branch off the rail becomes the hidden state
    body += circle(x_branch, LS_RAIL, 3.2, fill=PURPLE, stroke="none", op=0.9)
    body += path(
        f"M {x_branch} {LS_RAIL} V {y_branch} H {x_tanh - 30}",
        stroke=PURPLE,
        sw=1.5,
        op=0.65,
    )
    body += rect(
        x_tanh - 30,
        y_branch - 15,
        60,
        30,
        fill=PURPLE,
        op=0.14,
        stroke=PURPLE,
        sw=1.2,
        so=0.8,
        rx=4,
    )
    body += text(x_tanh, y_branch + 5, "tanh", size=12.5, op=0.9)
    body += arrow(
        x_tanh + 30,
        y_branch,
        x_mul_o - 11,
        y_branch,
        sw=1.5,
        op=0.8,
        head=6,
        stroke=PURPLE,
    )
    body += op_node(x_mul_o, y_branch, "⊙", colour=TEAL)
    body += path(
        f"M {wire_x} 284 H {x_mul_o} V {y_branch + 11}", stroke=TEAL, sw=1.5, op=0.9
    )
    body += arrow(
        x_mul_o,
        y_branch + 22,
        x_mul_o,
        y_branch + 11,
        stroke=TEAL,
        sw=1.5,
        op=0.9,
        head=6,
    )
    body += arrow(
        x_mul_o + 11, y_branch, x_end, y_branch, sw=1.8, op=0.9, head=7, stroke=GREEN
    )
    body += text(
        x_end + 10, y_branch + 5, "h", size=14, style="italic", op=0.95, anchor="start"
    )
    body += text(x_end + 20, y_branch + 9, "t", size=9, op=0.95, anchor="start")

    body += text(
        442,
        322,
        "f  forget — how much of the cell to keep",
        size=11,
        op=0.62,
        anchor="end",
    )
    body += text(
        462, 322, "i  input — whether to write at all", size=11, op=0.62, anchor="start"
    )
    body += text(
        442, 338, "g  candidate — what would be written", size=11, op=0.62, anchor="end"
    )
    body += text(
        462,
        338,
        "o  output — how much of the cell to reveal",
        size=11,
        op=0.62,
        anchor="start",
    )

    return wrap(
        LS_W,
        LS_H,
        "lstmCell",
        "One step of a long short-term memory cell",
        "The cell state runs along the top from left to right, passing through an "
        "elementwise multiply by the forget gate and an addition of the gated "
        "candidate, with no matrix multiply on it. Below, a single weight matrix W "
        "produces four gates f, i, g and o from the stacked previous hidden state "
        "and current input. A branch off the cell state passes through tanh and the "
        "output gate to become the new hidden state.",
        body,
    )


# ---------------------------------------------------------------------------
# 6. Captioning: a CNN's penultimate vector conditioning a recurrence.
# ---------------------------------------------------------------------------

CAP_W, CAP_H = 900.0, 306.0
CAP_X0, CAP_DX = 372.0, 130.0
CAP_BW, CAP_BH = 68.0, 46.0
CAP_YBOX = 118.0
CAP_YRAIL = 250.0
CAP_WORDS = ["a", "cat", "sitting", "&lt;END&gt;"]
CAP_INPUTS = ["&lt;START&gt;", "a", "cat", "sitting"]


def fig_captioning():
    """A CNN's penultimate vector conditioning every step of a recurrence."""
    body = ""
    cxs = [CAP_X0 + i * CAP_DX for i in range(4)]
    ybot = CAP_YBOX + CAP_BH

    # the convolutional stack, tapering
    body += text(
        140, 34, "convolutional network, classifier head removed", size=11.5, op=0.7
    )
    xs = 46.0
    for i in range(4):
        h = 96 - i * 16
        w = 26 - i * 3
        y = CAP_YBOX + CAP_BH / 2 - h / 2
        body += rect(
            xs,
            y,
            w,
            h,
            fill=INDIGO,
            op=0.20 + 0.13 * i,
            stroke=INDIGO,
            sw=1.0,
            so=0.8,
            rx=2,
        )
        xs += w + 12
    body += rect(
        xs,
        CAP_YBOX - 4,
        16,
        CAP_BH + 8,
        fill=GREEN,
        op=0.5,
        stroke=GREEN,
        sw=1.2,
        so=1.0,
        rx=2,
    )
    body += text(xs + 8, CAP_YBOX + CAP_BH + 26, "v", size=14, style="italic", op=0.95)
    body += text(xs + 8, CAP_YBOX + CAP_BH + 42, "penultimate", size=10.5, op=0.6)
    body += text(xs + 8, CAP_YBOX + CAP_BH + 55, "layer", size=10.5, op=0.6)

    vx = xs + 16
    body += path(
        f"M {vx + 4} {CAP_YBOX + CAP_BH / 2} H {vx + 26} V {CAP_YRAIL} H {cxs[-1] + 30}",
        stroke=GREEN,
        sw=1.8,
        op=0.95,
    )
    body += (
        f'  <text x="{(vx + 40 + cxs[-1]) / 2:.1f}" y="{CAP_YRAIL + 22:.1f}" font-size="11.5"'
        ' fill="currentColor" fill-opacity="0.66" text-anchor="middle">'
        "the image vector enters every step, through W"
        '<tspan dy="3" font-size="8.5">ih</tspan></text>\n'
    )

    body += text(
        cxs[0] - CAP_BW / 2 - 26,
        CAP_YBOX + 29,
        "h",
        size=12.5,
        fill=TEAL,
        style="italic",
    )
    body += text(cxs[0] - CAP_BW / 2 - 19, CAP_YBOX + 33, "0", size=9.5, fill=TEAL)

    for i, cx in enumerate(cxs):
        body += rect(
            cx - CAP_BW / 2,
            CAP_YBOX,
            CAP_BW,
            CAP_BH,
            fill=PURPLE,
            op=0.16,
            stroke=PURPLE,
            sw=1.2,
            so=0.85,
            rx=4,
        )
        body += text(cx, CAP_YBOX + 29, "f", size=15, style="italic", op=0.95)
        body += text(cx + 9, CAP_YBOX + 33, "W", size=10, op=0.95)

        body += text(cx, 86, CAP_WORDS[i], size=12.5, fill=TEAL)
        body += arrow(cx, CAP_YBOX - 3, cx, 94, sw=1.3, op=0.6, head=5.5)

        body += text(cx - 6, 214, CAP_INPUTS[i], size=11, op=0.8, font=MONO)
        body += arrow(cx - 6, 202, cx - 6, ybot + 3, sw=1.2, op=0.55, head=5)

        body += arrow(
            cx + 30,
            CAP_YRAIL,
            cx + 30,
            ybot + 3,
            sw=1.3,
            op=0.95,
            head=5.5,
            stroke=GREEN,
            dash="4 3",
        )

        if i < 3:
            x0, x1 = cx + CAP_BW / 2 + 3, cxs[i + 1] - CAP_BW / 2 - 3
            body += arrow(
                x0,
                CAP_YBOX + 25,
                x1,
                CAP_YBOX + 25,
                sw=1.5,
                op=0.9,
                head=6,
                stroke=TEAL,
            )

    body += text(cxs[-1] + CAP_BW / 2 + 44, 86, "the caption ends", size=10.5, op=0.55)
    body += text(
        cxs[-1] + CAP_BW / 2 + 44,
        100,
        "when &lt;END&gt; is emitted",
        size=10.5,
        op=0.55,
    )

    return wrap(
        CAP_W,
        CAP_H,
        "captioning",
        "Captioning: a convolutional network's penultimate vector conditioning a recurrence",
        "On the left a tapering stack of convolutional layers ends in a green bar "
        "labelled v, the penultimate layer. A green rail carries v to the right and "
        "up into all four steps of the recurrence. Each step also receives the word "
        "emitted at the previous step and emits the next one, ending with an "
        "end-of-sequence token.",
        body,
    )


FIGURES = {
    "diagram_seq_shapes.svg": fig_seq_shapes,
    "diagram_unrolled.svg": fig_unrolled,
    "diagram_truncated_bptt.svg": fig_truncated_bptt,
    "diagram_gradient_norm.svg": fig_gradient_norm,
    "diagram_lstm_cell.svg": fig_lstm_cell,
    "diagram_captioning.svg": fig_captioning,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print(f"wrote {OUT / name}")

    # The chapter quotes these; print them so the prose can be checked.
    for sigma, tanhp, label, _, _, _, _ in GN_CURVES:
        f = gn_factor(sigma, tanhp)
        print(f"{label:32s} factor {f:.4f} -> after 150 steps {f**150:.4g}")
