"""Generate the four SVG diagrams for the lecture 3 chapter.

Every curve is evaluated numerically rather than sketched, so the shapes in the
figures are the shapes the stated formulas and simulations actually produce.
Conventions follow figures/01-introduction/diagram_ilsvrc_error.svg: currentColor
for text and axes, viridis for data, title/desc with namespaced ids, no background
rect (the theme supplies a light plate in both schemes).
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _svg import INDIGO, INK, PURPLE, TEAL, mapper, wrap  # noqa: E402

OUT = pathlib.Path("figures/03-regularization-optimization")


def path(points, fmt="{:.2f}"):
    """Turn a list of (x, y) pixel pairs into an SVG path 'd' attribute."""
    head = "M " + fmt.format(points[0][0]) + " " + fmt.format(points[0][1])
    rest = "".join(" L " + fmt.format(x) + " " + fmt.format(y) for x, y in points[1:])
    return head + rest


# --------------------------------------------------------------------------
# 1. Overfitting: nine noisy points, an interpolating polynomial and a line.
# --------------------------------------------------------------------------

XS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
YS = [1.5, 1.2, 2.9, 2.2, 3.7, 4.6, 4.1, 5.6, 5.5]


def lagrange(x):
    """Degree-8 polynomial through all nine observations."""
    total = 0.0
    for i, xi in enumerate(XS):
        term = YS[i]
        for j, xj in enumerate(XS):
            if i != j:
                term *= (x - xj) / (xi - xj)
        total += term
    return total


def fig_overfitting():
    n = len(XS)
    mx = sum(XS) / n
    my = sum(YS) / n
    slope = sum((XS[i] - mx) * (YS[i] - my) for i in range(n)) / sum(
        (x - mx) ** 2 for x in XS
    )
    icpt = my - slope * mx

    to_px = mapper((-0.4, 8.4), (-2.2, 7.2), (78, 660), (48, 288))
    b = []

    b.append('  <g stroke="currentColor" opacity="0.16" stroke-width="1">\n')
    for gy in (0, 2, 4, 6):
        _, py = to_px(0, gy)
        b.append(f'    <line x1="78" y1="{py:.1f}" x2="660" y2="{py:.1f}"/>\n')
    b.append("  </g>\n")
    b.append(
        '  <g fill="currentColor" font-size="12" text-anchor="end" opacity="0.7">\n'
    )
    for gy in (0, 2, 4, 6):
        _, py = to_px(0, gy)
        b.append(f'    <text x="70" y="{py + 4:.1f}">{gy}</text>\n')
    b.append("  </g>\n")

    poly = [to_px(i / 400 * 8, lagrange(i / 400 * 8)) for i in range(401)]
    line = [to_px(x, icpt + slope * x) for x in (-0.2, 8.2)]
    b.append(
        f'  <path d="{path(poly)}" fill="none" stroke="{PURPLE}" stroke-width="2.6"/>\n'
    )
    b.append(
        f'  <path d="{path(line)}" fill="none" stroke="{TEAL}" stroke-width="2.8"/>\n'
    )

    b.append("  <g>\n")
    for x, y in zip(XS, YS):
        px, py = to_px(x, y)
        b.append(
            f'    <circle cx="{px:.1f}" cy="{py:.1f}" r="5.2" fill="#fbfbfb" stroke="{INK}" stroke-width="2.2"/>\n'
        )
    b.append("  </g>\n")

    lx, ly = to_px(1.35, -1.05)
    b.append(
        f'  <text x="{lx:.1f}" y="{ly:.1f}" fill="{PURPLE}" font-size="15" font-style="italic">f1</text>\n'
    )
    tx, ty = to_px(7.4, icpt + slope * 7.4)
    b.append(
        f'  <text x="{tx:.1f}" y="{ty + 26:.1f}" fill="{TEAL}" font-size="15" font-style="italic">f2</text>\n'
    )

    b.append(
        '  <line x1="78" y1="288" x2="660" y2="288" stroke="currentColor" stroke-width="1.2" opacity="0.5"/>\n'
    )
    b.append(
        '  <text x="369" y="313" fill="currentColor" font-size="12.5" text-anchor="middle" opacity="0.78">x</text>\n'
    )
    b.append(
        f'  <text x="78" y="{28}" fill="currentColor" font-size="12.5" opacity="0.78">y</text>\n'
    )
    b.append(
        '  <text x="660" y="313" fill="currentColor" font-size="12"'
        ' text-anchor="end" opacity="0.78">'
        "f1 has zero training error; f2 is the better predictor</text>\n"
    )

    return wrap(
        700,
        330,
        "overfit",
        "A wiggly fit and a straight fit to the same nine points",
        "Nine noisy observations of a linear relationship. A degree-eight "
        "polynomial passes exactly through every point and oscillates between "
        "them; a straight line misses all nine but follows the underlying trend.",
        "".join(b),
    )


# --------------------------------------------------------------------------
# 2. Two obstacles: a poorly conditioned valley, and a saddle in cross-section.
# --------------------------------------------------------------------------


def fig_sgd_problems():
    b = []
    b.append(
        '  <text x="290" y="34" fill="currentColor" font-size="14.5"'
        ' text-anchor="middle" font-weight="600">poor conditioning</text>\n'
    )
    b.append(
        '  <text x="790" y="34" fill="currentColor" font-size="14.5"'
        ' text-anchor="middle" font-weight="600">a saddle point</text>\n'
    )

    # -- left panel: contours of f = 0.5 (w1^2 + 20 w2^2) and a descent path
    to_px = mapper((-5.6, 5.6), (-1.75, 1.75), (70, 510), (56, 320))
    cx, cy = to_px(0, 0)
    sx = to_px(1, 0)[0] - cx
    sy = cy - to_px(0, 1)[1]

    b.append(f'  <g fill="none" stroke="{INDIGO}" stroke-width="1.3" opacity="0.42">\n')
    for c in (0.6, 2.2, 5.0, 8.8, 13.5):
        rx = math.sqrt(2 * c) * sx
        ry = math.sqrt(2 * c / 20) * sy
        b.append(
            f'    <ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}"/>\n'
        )
    b.append("  </g>\n")

    w1, w2, alpha = 4.6, 1.25, 0.09
    walk = [to_px(w1, w2)]
    for _ in range(26):
        w1, w2 = w1 - alpha * w1, w2 - alpha * 20 * w2
        walk.append(to_px(w1, w2))
    b.append(
        f'  <path d="{path(walk)}" fill="none" stroke="{PURPLE}" stroke-width="2.4"/>\n'
    )
    b.append(
        f'  <circle cx="{walk[0][0]:.1f}" cy="{walk[0][1]:.1f}" r="4.5" fill="{PURPLE}"/>\n'
    )
    b.append(
        f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="none" stroke="{INK}" stroke-width="2"/>\n'
    )
    b.append(
        f'  <text x="{walk[0][0] - 10:.1f}" y="{walk[0][1] - 12:.1f}"'
        f' fill="{PURPLE}" font-size="12.5" text-anchor="end">start</text>\n'
    )
    b.append(
        f'  <text x="{cx - 14:.1f}" y="{cy + 5:.1f}" fill="currentColor"'
        ' font-size="12.5" text-anchor="end" opacity="0.8">minimum</text>\n'
    )
    b.append(
        '  <text x="290" y="352" fill="currentColor" font-size="12.5"'
        ' text-anchor="middle" opacity="0.82">steps cross the valley faster'
        " than they travel along it</text>\n"
    )

    # -- right panel: the two principal cross-sections through one stationary point
    to_px = mapper((-3.0, 3.0), (-2.15, 2.15), (600, 1000), (56, 320))
    ox, oy = to_px(0, 0)
    span, curve = 145, 0.34  # u runs to 2.42, so 0.34 u^2 stays inside the panel
    up = [to_px(u / 60, curve * (u / 60) ** 2) for u in range(-span, span + 1)]
    dn = [to_px(u / 60, -curve * (u / 60) ** 2) for u in range(-span, span + 1)]
    b.append(
        f'  <path d="{path(up)}" fill="none" stroke="{INDIGO}" stroke-width="2.6"/>\n'
    )
    b.append(
        f'  <path d="{path(dn)}" fill="none" stroke="{PURPLE}" stroke-width="2.6"/>\n'
    )
    b.append(
        f'  <line x1="{ox - 78:.1f}" y1="{oy:.1f}" x2="{ox + 78:.1f}"'
        f' y2="{oy:.1f}" stroke="{INK}" stroke-width="1.4"'
        ' stroke-dasharray="6 5" opacity="0.75"/>\n'
    )
    b.append(f'  <circle cx="{ox:.1f}" cy="{oy:.1f}" r="5" fill="{INK}"/>\n')
    end = span / 60
    ux, uy = to_px(end, curve * end**2)
    dx, dy = to_px(end, -curve * end**2)
    b.append(
        f'  <text x="{ux - 8:.1f}" y="{uy - 12:.1f}" fill="{INDIGO}"'
        ' font-size="12.5" text-anchor="end">curves up along w1</text>\n'
    )
    b.append(
        f'  <text x="{dx - 8:.1f}" y="{dy + 20:.1f}" fill="{PURPLE}"'
        ' font-size="12.5" text-anchor="end">curves down along w2</text>\n'
    )
    b.append(
        f'  <text x="{ox - 88:.1f}" y="{oy - 14:.1f}" fill="currentColor"'
        ' font-size="12.5" text-anchor="end" opacity="0.8">'
        "gradient is zero here</text>\n"
    )
    b.append(
        '  <text x="790" y="352" fill="currentColor" font-size="12.5"'
        ' text-anchor="middle" opacity="0.82">descent stalls at a point that'
        " is not a minimum</text>\n"
    )

    return wrap(
        1040,
        372,
        "sgdProb",
        "Two obstacles for gradient descent",
        "Left: elongated elliptical level sets, with a descent path that "
        "oscillates across the narrow valley while advancing slowly along it. "
        "Right: the two principal cross-sections through a saddle point, one "
        "curving upwards and one downwards, both flat at the centre.",
        "".join(b),
    )


# --------------------------------------------------------------------------
# 3. Four learning rates on the same problem.
# --------------------------------------------------------------------------


class Lcg:
    """A small deterministic generator, so the figure is reproducible."""

    def __init__(self, seed):
        self.s = seed

    def uniform(self):
        self.s = (1103515245 * self.s + 12345) % (1 << 31)
        return self.s / (1 << 31) * 2 - 1


def simulate(alpha, epochs=50, per_epoch=6, curv=(0.25, 8.0), noise=1.8, seed=7):
    """SGD on a quadratic with additive gradient noise; returns the mean loss per epoch.

    Averaging within an epoch is what a real training curve plots, and without it the minibatch noise swamps the effect
    the figure is about. The level a run settles at grows with the step size, which is what separates the 'slightly too
    high' curve from the well-chosen one; the shallow direction (curvature 0.25) is what keeps the good run still
    descending at the right-hand edge.
    """
    rng = Lcg(seed)
    w = [5.0, 0.9]
    out = []
    for _ in range(epochs):
        acc = 0.0
        for _ in range(per_epoch):
            acc += 0.5 * sum(c * x * x for c, x in zip(curv, w))
            w = [x - alpha * (c * x + noise * rng.uniform()) for c, x in zip(curv, w)]
        out.append(acc / per_epoch)
        if max(abs(x) for x in w) > 1e4:
            out.append(float("inf"))
            break
    return out


def fig_lr_curves():
    """Loss against epoch at four step sizes, on a logarithmic loss axis.

    A linear axis cannot show this: the interesting difference is between noise floors two orders of magnitude below the
    starting loss.
    """
    # 0.252 sits just past the stability boundary 2/lambda_max = 0.25, so the
    # divergence is visible over a few epochs rather than in a single one.
    # Label anchors are given as explicit (epoch, loss) points in clear space
    # rather than derived from the curve, which put text on top of it.
    runs = [
        (0.252, PURPLE, "too high", None),
        (0.005, INDIGO, "too low", (31, 2.6)),
        (0.16, TEAL, "high: fast, then stuck", (18, 0.46)),
        (0.06, INK, "a good rate", (20, 0.040)),
    ]
    top, bottom = 1.2, -2.1  # decades of loss shown
    to_px = mapper((0, 49), (bottom, top), (82, 610), (48, 282))
    b = []

    b.append('  <g stroke="currentColor" opacity="0.16" stroke-width="1">\n')
    for gy in (1, 0, -1, -2):
        _, py = to_px(0, gy)
        b.append(f'    <line x1="82" y1="{py:.1f}" x2="610" y2="{py:.1f}"/>\n')
    b.append("  </g>\n")
    b.append(
        '  <g fill="currentColor" font-size="12" text-anchor="end" opacity="0.7">\n'
    )
    for gy, lab in ((1, "10"), (0, "1"), (-1, "0.1"), (-2, "0.01")):
        _, py = to_px(0, gy)
        b.append(f'    <text x="74" y="{py + 4:.1f}">{lab}</text>\n')
    b.append("  </g>\n")

    for alpha, colour, label, at in runs:
        losses = simulate(alpha, per_epoch=10)
        pts = []
        for t, val in enumerate(losses):
            if val == float("inf") or math.log10(max(val, 1e-9)) > top:
                pts.append(to_px(t, top))
                break
            pts.append(to_px(t, math.log10(max(val, 10**bottom))))
        width = "3.1" if colour == INK else "2.4"
        b.append(
            f'  <path d="{path(pts)}" fill="none" stroke="{colour}" stroke-width="{width}"/>\n'
        )
        if at is None:  # the diverging run: label where it leaves the top
            px, py = pts[-1][0] + 10, pts[-1][1] + 20
        else:
            px, py = to_px(at[0], math.log10(at[1]))
        b.append(
            f'  <text x="{px:.1f}" y="{py:.1f}" fill="{colour}" font-size="12.5">'
            + label
            + "</text>\n"
        )

    b.append(
        '  <line x1="82" y1="282" x2="610" y2="282" stroke="currentColor" stroke-width="1.2" opacity="0.5"/>\n'
    )
    b.append(
        '  <g fill="currentColor" font-size="12" text-anchor="middle" opacity="0.7">\n'
    )
    for t in (0, 10, 20, 30, 40):
        px, _ = to_px(t, 0)
        b.append(f'    <text x="{px:.1f}" y="302">{t}</text>\n')
    b.append("  </g>\n")
    b.append(
        '  <text x="346" y="322" fill="currentColor" font-size="12.5"'
        ' text-anchor="middle" opacity="0.78">epoch</text>\n'
    )
    b.append(
        '  <text x="82" y="28" fill="currentColor" font-size="12.5" opacity="0.78">training loss (log scale)</text>\n'
    )

    return wrap(
        700,
        338,
        "lrCurves",
        "Training loss under four learning rates",
        "Stochastic gradient descent on the same quadratic objective at four "
        "step sizes, with loss on a logarithmic axis. The largest diverges off "
        "the top; the smallest descends slowly and never arrives; an intermediate "
        "one falls fastest at first and then flattens an order of magnitude above "
        "where it could have gone; a well-chosen one falls quickly and keeps "
        "improving.",
        "".join(b),
    )


# --------------------------------------------------------------------------
# 4. Four decay schedules.
# --------------------------------------------------------------------------


def fig_lr_schedules():
    total = 100
    warm = 5

    def cosine(t):
        if t < warm:
            return t / warm
        return 0.5 * (1 + math.cos(math.pi * (t - warm) / (total - warm)))

    def linear(t):
        return 1 - t / total

    def invsqrt(t):
        return 1 / math.sqrt(max(t, 1))

    def step(t):
        return 0.1 ** (int(t) // 30)

    to_px = mapper((0, total), (0, 1.06), (78, 620), (48, 288))
    b = []

    b.append('  <g stroke="currentColor" opacity="0.16" stroke-width="1">\n')
    for gy in (0, 0.5, 1.0):
        _, py = to_px(0, gy)
        b.append(f'    <line x1="78" y1="{py:.1f}" x2="620" y2="{py:.1f}"/>\n')
    b.append("  </g>\n")
    b.append(
        '  <g fill="currentColor" font-size="12" text-anchor="end" opacity="0.7">\n'
    )
    for gy, lab in ((0, "0"), (0.5, "0.5"), (1.0, "1.0")):
        _, py = to_px(0, gy)
        b.append(f'    <text x="70" y="{py + 4:.1f}">{lab}</text>\n')
    b.append("  </g>\n")

    grid = [t / 4 for t in range(total * 4 + 1)]
    for fn, colour, label, at in (
        (step, INDIGO, "step", 14),
        (invsqrt, TEAL, "inverse sqrt", 46),
        (linear, PURPLE, "linear", 62),
        (cosine, INK, "cosine, after a linear warmup", 30),
    ):
        pts = [to_px(t, fn(t)) for t in grid]
        width = "3.1" if colour == INK else "2.4"
        b.append(
            f'  <path d="{path(pts)}" fill="none" stroke="{colour}" stroke-width="{width}"/>\n'
        )
        px, py = to_px(at, fn(at))
        b.append(
            f'  <text x="{px + 9:.1f}" y="{py - 9:.1f}" fill="{colour}" font-size="12.5">{label}</text>\n'
        )

    wx, _ = to_px(warm, 0)
    b.append(
        f'  <line x1="{wx:.1f}" y1="48" x2="{wx:.1f}" y2="288"'
        f' stroke="{INK}" stroke-width="1" stroke-dasharray="3 4"'
        ' opacity="0.45"/>\n'
    )
    b.append(
        f'  <text x="{wx + 6:.1f}" y="282" fill="currentColor" font-size="11.5" opacity="0.7">warmup ends</text>\n'
    )

    b.append(
        '  <line x1="78" y1="288" x2="620" y2="288" stroke="currentColor" stroke-width="1.2" opacity="0.5"/>\n'
    )
    b.append(
        '  <g fill="currentColor" font-size="12" text-anchor="middle" opacity="0.7">\n'
    )
    for t in (0, 25, 50, 75, 100):
        px, _ = to_px(t, 0)
        b.append(f'    <text x="{px:.1f}" y="308">{t}</text>\n')
    b.append("  </g>\n")
    b.append(
        '  <text x="349" y="328" fill="currentColor" font-size="12.5"'
        ' text-anchor="middle" opacity="0.78">epoch</text>\n'
    )
    b.append(
        '  <text x="78" y="28" fill="currentColor" font-size="12.5"'
        ' opacity="0.78">learning rate, as a fraction of its initial value</text>\n'
    )

    return wrap(
        700,
        344,
        "lrSched",
        "Four learning rate decay schedules",
        "Step decay holds the rate constant and multiplies it by one tenth "
        "every thirty epochs. Cosine decay, here preceded by a five-epoch linear "
        "warmup, sweeps smoothly to zero. Linear decay falls in a straight line "
        "to zero. Inverse square root decay falls quickly at first and then "
        "flattens without reaching zero.",
        "".join(b),
    )


for name, svg in (
    ("diagram_overfitting.svg", fig_overfitting()),
    ("diagram_sgd_problems.svg", fig_sgd_problems()),
    ("diagram_lr_curves.svg", fig_lr_curves()),
    ("diagram_lr_schedules.svg", fig_lr_schedules()),
):
    (OUT / name).write_text(svg)
    print(f"wrote {OUT / name}  ({len(svg)} bytes)")
