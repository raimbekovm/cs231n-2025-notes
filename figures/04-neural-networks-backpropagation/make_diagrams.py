"""Generate the five SVG diagrams for the lecture 4 chapter.

Every curve is evaluated numerically rather than sketched, and every gradient
annotated on a graph is the number the chain rule actually produces for the
stated inputs. Conventions follow figures/03-regularization-optimization:
currentColor for text and rules, viridis for data, title/desc with namespaced
ids, no background rect (the theme supplies a light plate in both schemes).

Run from the repository root; rewrites all five files.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _svg import INDIGO, INK, MONO, PURPLE, TEAL, mapper, text, wrap  # noqa: E402

OUT = pathlib.Path("figures/04-neural-networks-backpropagation")

JADE = "#0d7a5c"
# The gradient annotations on the two graph figures are all in one colour so
# that "below the wire and teal" reads as a single signal for "backward".
GRAD = TEAL


def path(points, fmt="{:.2f}"):
    """Turn a list of (x, y) pixel pairs into an SVG path 'd' attribute."""
    head = "M " + fmt.format(points[0][0]) + " " + fmt.format(points[0][1])
    rest = "".join(" L " + fmt.format(x) + " " + fmt.format(y) for x, y in points[1:])
    return head + rest


# ---------------------------------------------------------------------------
# 1. Two fully connected networks: one hidden layer, then two.
# ---------------------------------------------------------------------------


def net(ox, layers, labels, caption_top, caption_bottom, top=64, gap=34, col=88):
    """Draw one fully connected network. `layers` is a list of unit counts."""
    tallest = max(layers)
    centres = []
    for i, n in enumerate(layers):
        x = ox + i * col
        y0 = top + (tallest - n) * gap / 2
        centres.append([(x, y0 + j * gap) for j in range(n)])

    out = ""
    # Edges first so the node discs cover their endpoints.
    for a, b in zip(centres, centres[1:]):
        for x0, y0 in a:
            for x1, y1 in b:
                out += (
                    f'  <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"'
                    f' stroke="currentColor" stroke-width="0.6" opacity="0.28"/>\n'
                )
    fills = [INDIGO, TEAL, TEAL, PURPLE]
    for i, layer in enumerate(centres):
        fill = fills[0] if i == 0 else (fills[3] if i == len(centres) - 1 else fills[1])
        for x, y in layer:
            out += f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{fill}"/>\n'

    # Layer captions sit under the deepest possible node row, not under each
    # column's own last node, so they align with one another.
    base = top + (tallest - 1) * gap + 30
    for i, lab in enumerate(labels):
        out += text(ox + i * col, base, lab, size=12.5)
    span_mid = ox + (len(layers) - 1) * col / 2
    out += text(span_mid, 34, caption_top, size=14)
    out += text(
        span_mid, base + 26, caption_bottom, size=12.5, fill=INK, style="italic"
    )
    return out


def fig_network():
    # Wide enough that the right network's four columns and their node radii
    # clear the canvas edge: 454 + 3*88 + 7 = 725.
    w, h = 780, 300
    body = net(
        106,
        [4, 5, 3],
        ["x", "h", "s"],
        "two layers, one hidden",
        "W₁, W₂",
    )
    body += (
        f'  <line x1="{w / 2:.0f}" y1="46" x2="{w / 2:.0f}" y2="256"'
        f' stroke="currentColor" stroke-width="1" opacity="0.22"/>\n'
    )
    body += net(
        454,
        [4, 5, 5, 3],
        ["x", "h₁", "h₂", "s"],
        "three layers, two hidden",
        "W₁, W₂, W₃",
    )
    return wrap(
        w,
        h,
        "netFig",
        "Fully connected networks with one and two hidden layers",
        "Left: four inputs feeding five hidden units feeding three outputs, every "
        "unit connected to every unit in the next layer, giving two weight "
        "matrices. Right: the same with a second hidden layer and three weight "
        "matrices. A network is counted by its weight matrices.",
        body,
    )


# ---------------------------------------------------------------------------
# 2. Three weighted ReLU units and the piecewise-linear function they sum to.
# ---------------------------------------------------------------------------

# Output weight and kink location for each unit. The kink of c*max(0, x - k)
# sits at x = k, so these are the corner positions in the summed curve.
UNITS = [(1.0, -2.5, INDIGO), (-2.2, 0.3, PURPLE), (2.6, 2.2, JADE)]


def unit(c, k, x):
    return c * max(0.0, x - k)


def total(x):
    return sum(unit(c, k, x) for c, k, _ in UNITS)


def fig_relu_bending():
    w, h = 660, 440
    dom = (-4.0, 4.0)
    xs = [dom[0] + i * (dom[1] - dom[0]) / 400 for i in range(401)]

    top_to = mapper(dom, (-0.5, 3.6), (78, 616), (44, 194))
    # The lower range is wider than the curves need (the tallest unit reaches
    # 6.5 and the lowest -8.14) so that headroom stays clear for the legend.
    bot_to = mapper(dom, (-9.0, 9.0), (78, 616), (262, 405))

    body = ""
    for to_px, rng in ((top_to, (-0.5, 3.6)), (bot_to, (-9.0, 9.0))):
        zx0, zy = to_px(dom[0], 0.0)
        zx1, _ = to_px(dom[1], 0.0)
        body += (
            f'  <line x1="{zx0:.1f}" y1="{zy:.1f}" x2="{zx1:.1f}" y2="{zy:.1f}"'
            f' stroke="currentColor" stroke-width="0.9" opacity="0.45"/>\n'
        )
        _ = rng

    # The corners, marked on both panels so the correspondence is visible.
    for _, k, colour in UNITS:
        x0, y0 = top_to(k, 0.0)
        body += (
            f'  <line x1="{x0:.1f}" y1="44" x2="{x0:.1f}" y2="200"'
            f' stroke="{colour}" stroke-width="0.8" stroke-dasharray="3 4"'
            f' opacity="0.45"/>\n'
            # Broken across the legend band rather than drawn through it.
            f'  <line x1="{x0:.1f}" y1="262" x2="{x0:.1f}" y2="405"'
            f' stroke="{colour}" stroke-width="0.8" stroke-dasharray="3 4"'
            f' opacity="0.45"/>\n'
        )
        _ = y0

    body += f'  <path d="{path([top_to(x, total(x)) for x in xs])}" fill="none" stroke="{INK}" stroke-width="2.4"/>\n'
    for c, k, colour in UNITS:
        pts = [bot_to(x, unit(c, k, x)) for x in xs]
        body += f'  <path d="{path(pts)}" fill="none" stroke="{colour}" stroke-width="2"/>\n'

    body += text(347, 26, "the function the network computes", size=13.5)
    body += text(347, 228, "the three units it is made of", size=13.5)
    # One legend row above the lower panel rather than inside it. Stacking the
    # three labels in the panel put the last one on the zero line; at 11.5px
    # mono each is about 152px wide, so the three fit the 538px inner width.
    for i, (c, k, colour) in enumerate(UNITS):
        body += text(
            84 + i * 184,
            250,
            f"{c:+.1f} · max(0, x − {k:.1f})".replace("− -", "+ "),
            size=11.5,
            fill=colour,
            anchor="start",
            font=MONO,
        )
    for x in (-4, -2, 0, 2, 4):
        px, _ = bot_to(x, 0.0)
        body += text(px, 429, str(x).replace("-", "−"), size=11.5, fill=INK)
    return wrap(
        w,
        h,
        "reluFig",
        "Three ReLU units summing to a piecewise-linear function",
        "Lower panel: three weighted rectified linear units, each flat until its "
        "own corner and linear after it. Upper panel: their sum, a piecewise-"
        "linear function with one corner contributed by each unit. Dashed lines "
        "mark the corner positions across both panels.",
        body,
    )


# ---------------------------------------------------------------------------
# 3. Five activation functions on shared axes.
# ---------------------------------------------------------------------------


def gelu(z):
    return z * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# Leaky ReLU is drawn with alpha = 0.1, not the 0.01 of the text: at 0.01 the
# negative slope is a third of a pixel over the whole domain and the curve is
# indistinguishable from ReLU. The caption says so. ReLU is drawn first and
# thicker so that the positive ray, where the two functions coincide exactly,
# reads as two curves rather than one.
LEAKY_ALPHA = 0.1

ACTS = [
    ("ReLU", lambda z: max(0.0, z), INK, (2.05, 2.45), 3.6),
    ("leaky ReLU", lambda z: max(LEAKY_ALPHA * z, z), INDIGO, (-2.90, -0.70), 1.9),
    # Placed by the negative dip, the one place GELU is clearly not ReLU.
    ("GELU", gelu, JADE, (-1.50, -0.48), 2.2),
    ("tanh", math.tanh, PURPLE, (2.15, 1.32), 2.2),
    ("sigmoid", sigmoid, TEAL, (2.15, 0.60), 2.2),
]


def fig_activations():
    w, h = 620, 380
    dom, rng = (-3.0, 3.0), (-1.35, 3.05)
    to_px = mapper(dom, rng, (66, 566), (40, 326))
    xs = [dom[0] + i * (dom[1] - dom[0]) / 600 for i in range(601)]

    body = ""
    for gy in (-1, 0, 1, 2, 3):
        _, py = to_px(0.0, gy)
        op = 0.5 if gy == 0 else 0.16
        body += (
            f'  <line x1="66" y1="{py:.1f}" x2="566" y2="{py:.1f}"'
            f' stroke="currentColor" stroke-width="0.8" opacity="{op}"/>\n'
        )
        body += text(
            58, py + 4, str(gy).replace("-", "−"), size=11.5, fill=INK, anchor="end"
        )
    zx, _ = to_px(0.0, 0.0)
    body += (
        f'  <line x1="{zx:.1f}" y1="40" x2="{zx:.1f}" y2="326"'
        f' stroke="currentColor" stroke-width="0.8" opacity="0.5"/>\n'
    )
    for gx in (-3, -2, -1, 1, 2, 3):
        px, _ = to_px(gx, 0.0)
        body += text(px, 345, str(gx).replace("-", "−"), size=11.5, fill=INK)

    for name, fn, colour, (lx, ly), lw in ACTS:
        pts = [to_px(x, fn(x)) for x in xs]
        body += f'  <path d="{path(pts)}" fill="none" stroke="{colour}" stroke-width="{lw}"/>\n'
        px, py = to_px(lx, ly)
        body += text(px, py, name, size=12.5, fill=colour, anchor="start")

    body += text(316, 24, "activation functions on a shared axis", size=13.5)
    body += text(316, 367, "input z", size=12, fill=INK, style="italic")
    return wrap(
        w,
        h,
        "actFig",
        "ReLU, leaky ReLU, GELU, tanh and sigmoid",
        "ReLU is zero for negative inputs and the identity for positive ones. "
        "Leaky ReLU has a shallow negative slope instead of a flat one, drawn "
        "here with alpha = 0.1 rather than the usual 0.01 so that the slope is "
        "visible; the two functions coincide exactly for positive inputs. GELU "
        "follows ReLU away from the origin but is smooth through it and dips "
        "slightly below zero. Tanh and sigmoid are bounded and flatten at both "
        "ends, which is where they destroy a gradient.",
        body,
    )


# ---------------------------------------------------------------------------
# 4. The computational graph of f = (x + y)z, annotated forwards and backwards.
# ---------------------------------------------------------------------------


def wire(x0, y0, x1, y1, value, grad, at=0.5):
    """A directed edge with its forward value above and its gradient below.

    Both offsets are measured from the wire, never from a value, so no label can land on the line it annotates however
    the endpoints move.
    """
    out = (
        f'  <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"'
        f' stroke="currentColor" stroke-width="1.4" opacity="0.55"'
        f' marker-end="url(#arrow)"/>\n'
    )
    mx = x0 + (x1 - x0) * at
    my = y0 + (y1 - y0) * at
    out += text(mx, my - 14, value, size=13, fill=INK, font=MONO)
    out += text(mx, my + 22, grad, size=12.5, fill=GRAD, font=MONO)
    return out


def op_node(x, y, symbol, r=21):
    """An operator node. Word symbols get a wider disc and smaller type."""
    word = len(symbol) > 1
    if word:
        r = max(r, 4 + 4 * len(symbol))
    return (
        f'  <circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="currentColor"'
        f' stroke-width="1.5" opacity="0.75"/>\n'
        + text(x, y + (5 if word else 7), symbol, size=12.5 if word else 19, fill=INK)
    )


def var_node(x, y, name, value):
    out = (
        f'  <rect x="{x - 26}" y="{y - 19}" width="52" height="38" rx="6"'
        f' fill="none" stroke="{INDIGO}" stroke-width="1.5"/>\n'
    )
    out += text(x, y - 2, name, size=14, fill=INDIGO, style="italic")
    out += text(x, y + 14, value, size=12, fill=INK, font=MONO)
    return out


def fig_backprop_graph():
    w, h = 720, 320
    defs = (
        "  <defs>\n"
        '    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"'
        ' markerWidth="6" markerHeight="6" orient="auto-start-reverse">\n'
        '      <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" opacity="0.55"/>\n'
        "    </marker>\n"
        "  </defs>\n"
    )
    body = defs
    body += var_node(60, 78, "x", "−2")
    body += var_node(60, 172, "y", "5")
    body += var_node(60, 262, "z", "−4")
    body += op_node(232, 125, "+")
    body += op_node(430, 196, "×")

    # The x and y wires converge, so their labels are staggered along the wires
    # rather than both sitting at the midpoint, where the four numbers stack
    # into one column and stop being attributable to a wire.
    body += wire(86, 78, 209, 116, "−2", "−4", at=0.35)
    body += wire(86, 172, 209, 136, "5", "−4", at=0.68)
    body += wire(253, 125, 407, 178, "q = 3", "−4", at=0.5)
    body += wire(86, 262, 407, 214, "−4", "3", at=0.62)
    body += wire(451, 196, 596, 196, "f = −12", "1", at=0.5)

    body += text(660, 201, "f", size=16, fill=PURPLE, style="italic")
    body += text(360, 34, "forward values", size=13, fill=INK)
    body += text(360, 54, "gradients of f", size=13, fill=GRAD)
    body += text(
        360,
        300,
        "backward pass runs right to left",
        size=12.5,
        fill=GRAD,
        style="italic",
    )
    return wrap(
        w,
        h,
        "bpFig",
        "Computational graph of f = (x + y) z with forward values and gradients",
        "Inputs x = -2, y = 5 and z = -4 feed an addition producing q = 3 and a "
        "multiplication producing f = -12. Each edge carries its forward value "
        "and, beneath it, the derivative of f with respect to that edge: -4 on "
        "x, y and q, 3 on z, and 1 on f itself.",
        body,
    )


# ---------------------------------------------------------------------------
# 5. The four gradient-flow patterns.
# ---------------------------------------------------------------------------


def pattern(ox, oy, title, symbol, ins, outs, note):
    """One small graph: inputs on the left, one operator, outputs on the right."""
    out = text(ox + 132, oy - 26, title, size=13.5, fill=INK)
    cx, cy = ox + 132, oy + 42
    out += op_node(cx, cy, symbol, r=18)
    for i, (val, grad) in enumerate(ins):
        # A lone input enters horizontally; stacking it at the first slot would
        # cross the upper output wire in the copy panel.
        y = oy + 14 + i * 56 if len(ins) > 1 else cy
        out += (
            f'  <line x1="{ox + 44}" y1="{y}" x2="{cx - 21}" y2="{cy}"'
            f' stroke="currentColor" stroke-width="1.3" opacity="0.5"'
            f' marker-end="url(#arrow)"/>\n'
        )
        out += text(ox + 34, y - 2, val, size=13, fill=INK, anchor="end", font=MONO)
        out += text(
            ox + 34, y + 16, grad, size=12.5, fill=GRAD, anchor="end", font=MONO
        )
    for i, (val, grad) in enumerate(outs):
        y = oy + 14 + i * 56 if len(outs) > 1 else cy
        out += (
            f'  <line x1="{cx + 21}" y1="{cy}" x2="{ox + 216}" y2="{y}"'
            f' stroke="currentColor" stroke-width="1.3" opacity="0.5"'
            f' marker-end="url(#arrow)"/>\n'
        )
        out += text(ox + 226, y - 2, val, size=13, fill=INK, anchor="start", font=MONO)
        out += text(
            ox + 226, y + 16, grad, size=12.5, fill=GRAD, anchor="start", font=MONO
        )
    out += text(ox + 132, oy + 132, note, size=12, fill=INK, style="italic")
    return out


def fig_gradient_patterns():
    """The four gradient-flow patterns in a two-by-two grid."""
    w, h = 720, 414
    defs = (
        "  <defs>\n"
        '    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"'
        ' markerWidth="6" markerHeight="6" orient="auto-start-reverse">\n'
        '      <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" opacity="0.5"/>\n'
        "    </marker>\n"
        "  </defs>\n"
    )
    body = defs
    body += pattern(
        26, 60, "add", "+", [("3", "2"), ("4", "2")], [("7", "2")], "sends 2 both ways"
    )
    body += pattern(
        386,
        60,
        "multiply",
        "×",
        [("2", "3·5 = 15"), ("3", "2·5 = 10")],
        [("6", "5")],
        "each input scaled by the other",
    )
    body += pattern(
        26,
        262,
        "max",
        "max",
        [("4", "0"), ("5", "9")],
        [("5", "9")],
        "all of it down the winning branch",
    )
    body += pattern(
        386,
        262,
        "copy",
        "copy",
        [("7", "4+2 = 6")],
        [("7", "4"), ("7", "2")],
        "gradients from consumers are summed",
    )
    # No dividing rules between the quadrants: the vertical one crossed the
    # multiply panel's gradient labels, and whitespace groups them well enough.
    return wrap(
        w,
        h,
        "patFig",
        "Four gradient flow patterns: add, multiply, max and copy",
        "Forward values are above each wire and gradients below. Addition passes "
        "the upstream gradient to both inputs unchanged. Multiplication scales "
        "each input's gradient by the other input. Maximum routes the whole "
        "gradient to the largest input and zero to the rest. A copy node sums "
        "the gradients arriving from its consumers.",
        body,
    )


FIGURES = {
    "diagram_network.svg": fig_network,
    "diagram_relu_bending.svg": fig_relu_bending,
    "diagram_activations.svg": fig_activations,
    "diagram_backprop_graph.svg": fig_backprop_graph,
    "diagram_gradient_patterns.svg": fig_gradient_patterns,
}

for name, build in FIGURES.items():
    (OUT / name).write_text(build())
    print(name, len((OUT / name).read_text()), "bytes")
