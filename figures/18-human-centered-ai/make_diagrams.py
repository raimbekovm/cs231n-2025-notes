"""Figures for the Human-Centered AI chapter.

This lecture carries almost no architectures and no equations, so the figures
here are not diagrams of networks. They are drawn by evaluating the arithmetic
of a single 2x2 contingency table, which is the object the whole chapter rests
on: read across groups it gives the fairness impossibility result, and read
inside one deployment at low prevalence it gives the alarm-burden problem.

Every quantity the prose quotes is computed below and printed by `__main__`,
and the load-bearing claims are checked by two independent routes that must
agree -- see the `check_*` functions. Nothing in the chapter's mathematics is
asserted from memory.
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
    mapper,
    poly,
    polyline,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent

# Published Broward County / COMPAS figures. Prevalence from Chouldechova
# (arXiv:1610.07524), which reports a two-year recidivism rate of 51% among
# Black defendants against 39% among White defendants; the error rates are the
# ProPublica ones quoted in the same paper.
COMPAS = {
    "Black": {"p": 0.51, "fnr": 0.28, "fpr": 0.45},
    "White": {"p": 0.39, "fnr": 0.48, "fpr": 0.23},
}

# Retained surgical items run at roughly one per 5,500 operations in the
# largest series, which is the prevalence any automatic instrument count would
# actually have to operate at.
RETAINED_ITEM_RATE = 1.0 / 5500.0


# --------------------------------------------------------------------------
# The arithmetic of one contingency table.
# --------------------------------------------------------------------------


def rates_from_counts(tp, fn, fp, tn):
    """Recover prevalence, PPV, FNR and FPR from the four cells of a 2x2."""
    n = tp + fn + fp + tn
    return {
        "p": (tp + fn) / n,
        "ppv": tp / (tp + fp),
        "fnr": fn / (tp + fn),
        "fpr": fp / (fp + tn),
    }


def fpr_from_identity(p, ppv, fnr):
    """False positive rate implied by prevalence, predictive value and FNR.

    This is Chouldechova's relation, arXiv:1610.07524 eq. (2.1). It is an algebraic identity on the cells, not an
    approximation and not a modelling assumption: it holds for every 2x2 table whatever generated it.
    """
    return (p / (1.0 - p)) * ((1.0 - ppv) / ppv) * (1.0 - fnr)


def ppv_from_rates(p, fnr, fpr):
    """Positive predictive value implied by prevalence and the two error rates."""
    tp = p * (1.0 - fnr)
    fp = (1.0 - p) * fpr
    return tp / (tp + fp)


def odds(p):
    """Odds of a probability."""
    return p / (1.0 - p)


def ppv_from_sens_spec(sens, spec, prev):
    """Positive predictive value from sensitivity, specificity and prevalence."""
    tp = sens * prev
    fp = (1.0 - spec) * (1.0 - prev)
    return tp / (tp + fp)


def alarms_per_true_case(sens, spec, prev):
    """Total positive alarms raised per genuine case present.

    The reciprocal of the predictive value: if one alarm in five is real, the unit reviewing them works through five to
    find the one.
    """
    return 1.0 / ppv_from_sens_spec(sens, spec, prev)


def specificity_for_ppv(target_ppv, sens, prev):
    """Specificity a detector needs to reach a target predictive value.

    Inverts `ppv_from_sens_spec`. At the prevalences the operating-room and ward applications actually run at, this is
    the number that decides whether a system is deployable, and it is far more demanding than headline accuracy.
    """
    fp_allowed = sens * prev * (1.0 - target_ppv) / target_ppv
    return 1.0 - fp_allowed / (1.0 - prev)


def impossibility_residual(p1, p2, ppv, fnr):
    """Gap in FPR forced between two groups that share PPV and FNR.

    Equal PPV across groups is the coarsened form of calibration and equal FNR is balance for the positive class. Given
    both, the FPRs are pinned by the prevalences, so this residual is what equalising the third criterion would have to
    overcome. It is zero only in the theorem's own escape cases.
    """
    return abs(fpr_from_identity(p1, ppv, fnr) - fpr_from_identity(p2, ppv, fnr))


# --------------------------------------------------------------------------
# Checks: each claim computed two ways that must agree.
# --------------------------------------------------------------------------


def check_identity_against_counts():
    """A1. The identity against direct cell ratios, over a grid of integer tables.

    Road one counts cells: FPR = FP / (FP + TN). Road two never looks at the table, only at (p, PPV, FNR). They must
    agree exactly, because the identity is a rearrangement of the cells rather than a statistical claim.
    """
    worst = 0.0
    n = 0
    for tp in range(1, 40, 3):
        for fn in range(1, 40, 3):
            for fp in range(1, 40, 3):
                for tn in range(1, 40, 3):
                    r = rates_from_counts(tp, fn, fp, tn)
                    direct = fp / (fp + tn)
                    implied = fpr_from_identity(r["p"], r["ppv"], r["fnr"])
                    worst = max(worst, abs(direct - implied))
                    n += 1
    assert worst < 1e-12, f"identity broke by {worst}"
    return n, worst


def check_impossibility():
    """A2. Equal PPV and equal FNR force the FPR ratio to be the odds ratio.

    Two roads to the same gap: evaluate each group's FPR through the identity and subtract, or predict the ratio from
    prevalence alone. If they agree, then no choice of classifier reconciles the three criteria -- the third one is
    already determined by the first two.
    """
    worst = 0.0
    smallest_offdiag = 1.0
    for p1 in [0.05 * i for i in range(1, 13)]:
        for p2 in [0.05 * j for j in range(1, 13)]:
            for ppv in (0.55, 0.7, 0.85):
                for fnr in (0.1, 0.3, 0.5):
                    f1 = fpr_from_identity(p1, ppv, fnr)
                    f2 = fpr_from_identity(p2, ppv, fnr)
                    worst = max(worst, abs(f1 / f2 - odds(p1) / odds(p2)))
                    if abs(p1 - p2) > 1e-9:
                        smallest_offdiag = min(smallest_offdiag, impossibility_residual(p1, p2, ppv, fnr))
    assert worst < 1e-12, f"ratio law broke by {worst}"
    assert smallest_offdiag > 0.0, "residual reached zero off the diagonal"
    return worst, smallest_offdiag


def check_escape_hatches():
    """A3. The residual vanishes only where the theorem says it may.

    Kleinberg, Mullainathan and Raghavan (arXiv:1609.05807, Thm. 1.1) allow exactly two escapes: equal base rates, or
    perfect prediction. In this coarsened two-group form that shows up as p1 = p2, or PPV -> 1 with no false positives.
    The degenerate FNR -> 1 branch, where the classifier identifies no true cases at all, is checked too and reported
    honestly rather than hidden.
    """
    ppv, fnr = 0.7, 0.3
    equal_rates = impossibility_residual(0.4, 0.4, ppv, fnr)
    perfect = impossibility_residual(0.51, 0.39, 1.0 - 1e-12, fnr)
    degenerate = impossibility_residual(0.51, 0.39, ppv, 1.0 - 1e-12)
    unequal = impossibility_residual(0.51, 0.39, ppv, fnr)
    assert equal_rates == 0.0
    assert perfect < 1e-9
    assert degenerate < 1e-9
    assert unequal > 0.05, "the interesting case must stay well away from zero"
    return equal_rates, perfect, degenerate, unequal


def check_compas():
    """A4. The identity predicts the published COMPAS error rates.

    Road one is ProPublica's measured FPRs. Road two never sees them: it takes each group's prevalence and FNR, assumes
    the score is exactly calibrated at the average of the two groups' predictive values, and computes what the FPRs must
    then be. Agreement to a few points is the substantive result -- the error-rate gap is not a bug in the instrument,
    it is what calibration plus unequal prevalence forces.
    """
    derived = {g: ppv_from_rates(v["p"], v["fnr"], v["fpr"]) for g, v in COMPAS.items()}
    spread = abs(derived["Black"] - derived["White"])
    assert spread < 0.05, f"COMPAS is not near-calibrated after all: {spread}"

    ppv_bar = sum(derived.values()) / len(derived)
    predicted = {g: fpr_from_identity(v["p"], ppv_bar, v["fnr"]) for g, v in COMPAS.items()}
    for g, v in COMPAS.items():
        err = abs(predicted[g] - v["fpr"])
        assert err < 0.04, f"{g}: predicted {predicted[g]:.3f} vs published {v['fpr']}"
    return derived, ppv_bar, predicted


def check_ppv_collapse():
    """A5. Bayes against an explicit cohort of integers.

    Road one is the closed form. Road two builds a synthetic population of ten million people, counts the cells, and
    divides. Rounding is the only source of disagreement.
    """
    worst = 0.0
    n = 10_000_000
    for prev in (0.001, 0.005, 0.02, 0.1, 0.3):
        for sens, spec in ((0.95, 0.95), (0.99, 0.99), (0.9, 0.999)):
            pos = round(n * prev)
            tp = round(pos * sens)
            fp = round((n - pos) * (1.0 - spec))
            worst = max(worst, abs(tp / (tp + fp) - ppv_from_sens_spec(sens, spec, prev)))
    assert worst < 1e-5, f"PPV routes disagree by {worst}"
    return worst


def check_specificity_inverse():
    """A6. The specificity solver inverts the Bayes formula it is derived from.

    Road one solves for the specificity that would deliver a target predictive value; road two feeds that specificity
    back through `ppv_from_sens_spec` and must recover the target. This one guards the retained-instrument figures the
    chapter quotes, where the answer is counter-intuitive enough that an algebra slip would pass unnoticed.
    """
    worst = 0.0
    for prev in (RETAINED_ITEM_RATE, 0.005, 0.02, 0.2):
        for sens in (0.95, 0.99):
            for target in (0.1, 0.5, 0.9):
                spec = specificity_for_ppv(target, sens, prev)
                worst = max(worst, abs(ppv_from_sens_spec(sens, spec, prev) - target))
    assert worst < 1e-9, f"specificity inverse disagrees by {worst}"
    return worst


def check_curve_extents():
    """Every plotted curve stays inside the axis box it is drawn in.

    A curve whose peak leaves the viewBox is silently truncated and the figure then asserts something false, so the
    ranges are checked rather than eyeballed.
    """
    hi = max(fpr_from_identity(p, PPV_FIX, FNR_FIX) for p in _lin(*FAIR_DOM, 200))
    assert hi <= FAIR_RNG[1], f"fairness curve peaks at {hi:.3f}, above the axis"
    for sens, spec in DETECTORS:
        lo = min(ppv_from_sens_spec(sens, spec, p) for p in _log(*BASE_DOM, 200))
        hi2 = max(ppv_from_sens_spec(sens, spec, p) for p in _log(*BASE_DOM, 200))
        assert 0.0 <= lo and hi2 <= 1.0, "PPV curve left the axis"
    return hi


# --------------------------------------------------------------------------
# Plot parameters, shared between the checks and the drawings.
# --------------------------------------------------------------------------

PPV_FIX, FNR_FIX = 0.6, 0.3
FAIR_DOM = (0.05, 0.65)
FAIR_RNG = (0.0, 0.9)
BASE_DOM = (0.001, 0.5)
DETECTORS = ((0.95, 0.95), (0.95, 0.99))


def _lin(a, b, n):
    """N evenly spaced points from a to b inclusive."""
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _log(a, b, n):
    """N logarithmically spaced points from a to b inclusive."""
    la, lb = _log10(a), _log10(b)
    return [10.0 ** (la + (lb - la) * i / (n - 1)) for i in range(n)]


def _log10(x):
    """Base-ten logarithm without importing math into the layout code."""
    import math

    return math.log10(x)


def vir(t):
    """Sample the viridis anchors used across the book at a fraction t."""
    stops = [PURPLE, INDIGO, TEAL, GREEN, YELLOW]
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    a, b = stops[i], stops[i + 1]
    ch = [round(int(a[1 + 2 * k : 3 + 2 * k], 16) * (1 - f) + int(b[1 + 2 * k : 3 + 2 * k], 16) * f) for k in range(3)]
    return "#{:02x}{:02x}{:02x}".format(*ch)


def axes(to_px, px, py, xticks, yticks, xfmt, yfmt, xlab, ylab):
    """Draw a plain L-shaped axis pair with ticks and labels outside the box."""
    out = line(px[0], py[1], px[1], py[1], op=0.5)
    out += line(px[0], py[0], px[0], py[1], op=0.5)
    for v in xticks:
        x, _ = to_px(v, 0)
        out += line(x, py[1], x, py[1] + 4, op=0.5)
        out += text(x, py[1] + 18, xfmt(v), size=11, op=0.75)
    anchor_x = xticks[0] if xticks else 0
    for v in yticks:
        _, y = to_px(anchor_x, v)
        out += line(px[0] - 4, y, px[0], y, op=0.5)
        out += text(px[0] - 9, y + 4, yfmt(v), size=11, anchor="end", op=0.75)
    out += text((px[0] + px[1]) / 2, py[1] + 40, xlab, size=12)
    out += text(px[0] - 44, py[0] - 16, ylab, size=12, anchor="start")
    return out


# --------------------------------------------------------------------------
# Figures.
# --------------------------------------------------------------------------


def fig_confusion_table():
    """The 2x2, with each metric drawn against the margin it conditions on."""
    w, h = 660, 430
    x0, y0, cw, chh = 205, 120, 150, 76
    b = ""

    b += text(x0 + cw, 44, "predicted", size=12, op=0.75)
    b += text(x0 + cw / 2, 66, "flagged", size=13, weight="600")
    b += text(x0 + cw * 1.5, 66, "not flagged", size=13, weight="600")
    b += text(80, y0 + chh / 2 - 6, "actually", size=12, anchor="start", op=0.75)
    b += text(80, y0 + chh / 2 + 12, "positive", size=13, anchor="start", weight="600")
    b += text(80, y0 + chh * 1.5 - 6, "actually", size=12, anchor="start", op=0.75)
    b += text(80, y0 + chh * 1.5 + 12, "negative", size=13, anchor="start", weight="600")

    cells = [
        (0, 0, "TP", TEAL),
        (1, 0, "FN", INDIGO),
        (0, 1, "FP", PURPLE),
        (1, 1, "TN", INK),
    ]
    for c, r, lab, col in cells:
        x, y = x0 + c * cw, y0 + r * chh
        b += rect(x, y, cw, chh, fill=col, op=0.12, stroke=col, so=0.55)
        b += text(x + cw / 2, y + chh / 2 + 6, lab, size=17, fill=col, font=MONO)

    # Three margins, each drawn on a different side so no two rules collide.
    b += line(x0 - 10, y0, x0 - 10, y0 + 2 * chh, stroke=TEAL, sw=3, op=0.9)
    b += text(x0 - 20, y0 + chh, "PPV", size=13, fill=TEAL, anchor="end", weight="600")
    b += text(x0 - 20, y0 + chh + 17, "down this column", size=11, fill=TEAL, anchor="end", op=0.8)

    b += line(x0, y0 - 8, x0 + 2 * cw, y0 - 8, stroke=INDIGO, sw=3, op=0.9)
    b += text(x0 + 2 * cw + 14, y0 - 4, "FNR", size=13, fill=INDIGO, anchor="start", weight="600")
    b += text(x0 + 2 * cw + 14, y0 + 13, "across this row", size=11, fill=INDIGO, anchor="start", op=0.8)

    b += line(x0, y0 + 2 * chh + 8, x0 + 2 * cw, y0 + 2 * chh + 8, stroke=PURPLE, sw=3, op=0.9)
    b += text(x0 + 2 * cw + 14, y0 + 2 * chh + 4, "FPR", size=13, fill=PURPLE, anchor="start", weight="600")
    b += text(x0 + 2 * cw + 14, y0 + 2 * chh + 21, "across this row", size=11, fill=PURPLE, anchor="start", op=0.8)

    b += text(
        w / 2,
        y0 + 2 * chh + 66,
        "Three criteria, three different denominators.",
        size=13,
        style="italic",
    )
    b += text(
        w / 2,
        y0 + 2 * chh + 88,
        "Fixing any two of them fixes the third — it is not free to be equalised.",
        size=12,
        op=0.8,
    )
    return wrap(
        w,
        h,
        "conf",
        "The confusion table and its three margins",
        "A two by two table of true and false positives and negatives, with "
        "positive predictive value read down the flagged column, the false "
        "negative rate read across the positive row, and the false positive "
        "rate read across the negative row.",
        b,
    )


def fig_fairness_impossible():
    """FPR against prevalence at fixed PPV and FNR, with both COMPAS groups marked."""
    w, h = 660, 420
    px, py = (110, 570), (60, 320)
    to_px = mapper(FAIR_DOM, FAIR_RNG, px, py)
    b = ""

    pts = [to_px(p, fpr_from_identity(p, PPV_FIX, FNR_FIX)) for p in _lin(*FAIR_DOM, 240)]
    b += axes(
        to_px,
        px,
        py,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.0, 0.2, 0.4, 0.6, 0.8],
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.1f}",
        "prevalence in the group",
        "false positive rate",
    )

    pw, pb = 0.39, 0.51
    fw = fpr_from_identity(pw, PPV_FIX, FNR_FIX)
    fb = fpr_from_identity(pb, PPV_FIX, FNR_FIX)
    xw, yw = to_px(pw, fw)
    xb, yb = to_px(pb, fb)

    b += poly(
        [(xw, yw), (xb, yb), (xb, to_px(pb, fw)[1])],
        fill=YELLOW,
        op=0.35,
        stroke="none",
    )
    b += polyline(pts, stroke=TEAL, sw=2.2)
    for x, y in ((xw, yw), (xb, yb)):
        b += line(x, y, x, py[1], stroke=INK, sw=1, op=0.3, dash="3 3")
    b += poly([(xw, yw - 5), (xw + 5, yw), (xw, yw + 5), (xw - 5, yw)], fill=INDIGO, stroke="none")
    b += poly([(xb, yb - 5), (xb + 5, yb), (xb, yb + 5), (xb - 5, yb)], fill=PURPLE, stroke="none")

    b += text(xw - 10, yw - 14, f"p = .39 → {fw:.2f}", size=11, fill=INDIGO, anchor="end")
    b += text(xb + 12, yb - 14, f"p = .51 → {fb:.2f}", size=11, fill=PURPLE, anchor="start")
    b += text(
        (xw + xb) / 2,
        to_px(pb, fw)[1] + 26,
        "the gap calibration forces",
        size=11,
        op=0.85,
    )
    b += text(
        px[0] + 8,
        py[0] + 16,
        f"PPV held at {PPV_FIX:.2f}, FNR held at {FNR_FIX:.2f} in both groups",
        size=12,
        anchor="start",
        op=0.85,
    )
    b += text(
        px[0] + 8,
        py[0] + 34,
        "only the base rate differs — and that is enough",
        size=12,
        anchor="start",
        op=0.7,
        style="italic",
    )
    return wrap(
        w,
        h,
        "fair",
        "False positive rate forced by prevalence",
        "A rising curve of false positive rate against prevalence at fixed "
        "predictive value and false negative rate, with the two COMPAS base "
        "rates marked and the resulting gap shaded.",
        b,
    )


def fig_compas():
    """The published COMPAS rates: predictive value nearly equal, error rates not."""
    derived, _ppv_bar, _ = check_compas()
    w, h = 660, 400
    px, py = (140, 590), (70, 300)
    to_px = mapper((0, 3), (0, 0.7), px, py)
    b = ""
    b += axes(
        to_px,
        px,
        py,
        [],
        [0.0, 0.2, 0.4, 0.6],
        lambda v: "",
        lambda v: f"{v:.0%}",
        "",
        "rate",
    )

    groups = [("Black", PURPLE), ("White", INDIGO)]
    bars = [
        ("PPV", [derived["Black"], derived["White"]]),
        ("FPR", [COMPAS["Black"]["fpr"], COMPAS["White"]["fpr"]]),
        ("FNR", [COMPAS["Black"]["fnr"], COMPAS["White"]["fnr"]]),
    ]
    bw = 46
    for i, (lab, vals) in enumerate(bars):
        cx, _ = to_px(i + 0.5, 0)
        for j, ((g, col), v) in enumerate(zip(groups, vals)):
            x = cx - bw - 6 + j * (bw + 12)
            _, y = to_px(0, v)
            b += rect(x, y, bw, py[1] - y, fill=col, op=0.75, stroke=col, so=0.9)
            b += text(x + bw / 2, y - 8, f"{v:.0%}", size=11, fill=col, weight="600")
        b += text(cx, py[1] + 20, lab, size=13, weight="600")

    b += text(
        px[0] + 6,
        py[0] - 24,
        "Roughly equal predictive value, error rates apart by a factor of two",
        size=12,
        anchor="start",
        style="italic",
    )
    for j, (g, col) in enumerate(groups):
        # Right-hand side: the PPV pair is the tallest and reaches into the
        # top-left, so a legend placed there collides with its own value labels.
        b += rect(px[1] - 140 + j * 76, py[0] + 4, 12, 12, fill=col, op=0.75, stroke=col)
        b += text(px[1] - 122 + j * 76, py[0] + 15, g, size=12, anchor="start", fill=col)
    b += text(
        w / 2,
        h - 26,
        "Both facts hold at once because the two groups' base rates differ (51% against 39%).",
        size=12,
        op=0.8,
    )
    return wrap(
        w,
        h,
        "compas",
        "COMPAS predictive value against error rates",
        "Grouped bars showing near-equal positive predictive value for two "
        "groups alongside false positive and false negative rates that differ "
        "by roughly a factor of two.",
        b,
    )


def fig_impossibility_residual():
    """The forced FPR gap over pairs of base rates: zero only on the diagonal."""
    w, h = 620, 440
    px, py = (110, 470), (70, 330)
    dom = (0.05, 0.6)
    to_px = mapper(dom, dom, px, py)
    b = ""
    n = 22
    step = (dom[1] - dom[0]) / n
    cell_w = (px[1] - px[0]) / n
    cell_h = (py[1] - py[0]) / n
    vals = []
    for i in range(n):
        for j in range(n):
            p1 = dom[0] + (i + 0.5) * step
            p2 = dom[0] + (j + 0.5) * step
            vals.append(impossibility_residual(p1, p2, PPV_FIX, FNR_FIX))
    hi = max(vals)
    k = 0
    for i in range(n):
        for j in range(n):
            x = px[0] + i * cell_w
            y = py[1] - (j + 1) * cell_h
            b += rect(x, y, cell_w + 0.6, cell_h + 0.6, fill=vir(vals[k] / hi), stroke="none", op=0.95)
            k += 1
    b += rect(px[0], py[0], px[1] - px[0], py[1] - py[0], stroke=INK, so=0.35)
    b += line(px[0], py[1], px[1], py[0], stroke=INK, sw=1.4, op=0.55, dash="4 3")

    b += axes(
        to_px,
        px,
        py,
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.1f}",
        "base rate, group 1",
        "base rate, group 2",
    )
    b += text(px[1] + 20, py[0] + 12, "forced gap", size=12, anchor="start", weight="600")
    b += text(px[1] + 20, py[0] + 30, "in FPR", size=12, anchor="start", weight="600")
    for m in range(5):
        b += rect(px[1] + 20, py[0] + 48 + m * 22, 18, 18, fill=vir(1 - m / 4), stroke="none")
        b += text(px[1] + 44, py[0] + 62 + m * 22, f"{hi * (1 - m / 4):.2f}", size=11, anchor="start", op=0.8)
    b += text(
        w / 2,
        h - 22,
        "The only solutions lie on the diagonal, where the two base rates coincide.",
        size=12,
        op=0.85,
    )
    return wrap(
        w,
        h,
        "resid",
        "The forced gap in false positive rate",
        "A heat map of the false positive rate gap forced between two groups "
        "as a function of their two base rates, vanishing only along the "
        "diagonal where the base rates are equal.",
        b,
    )


def fig_base_rate():
    """Predictive value against prevalence for two good detectors."""
    w, h = 660, 420
    px, py = (110, 560), (60, 310)
    to_px = mapper((_log10(BASE_DOM[0]), _log10(BASE_DOM[1])), (0, 1), px, py)
    b = ""
    b += axes(
        to_px,
        px,
        py,
        [-3, -2.5, -2, -1.5, -1, -0.5],
        [0.0, 0.25, 0.5, 0.75, 1.0],
        lambda v: f"{10**v:.1%}".replace(".0%", "%"),
        lambda v: f"{v:.0%}",
        "prevalence of the event (log scale)",
        "chance an alarm is real",
    )
    cols = (TEAL, INDIGO)
    for (sens, spec), col in zip(DETECTORS, cols):
        pts = [to_px(_log10(p), ppv_from_sens_spec(sens, spec, p)) for p in _log(*BASE_DOM, 240)]
        b += polyline(pts, stroke=col, sw=2.2)
        end = pts[-1]
        b += text(end[0] + 8, end[1] + 4, f"{sens:.0%} / {spec:.0%}", size=11, fill=col, anchor="start")

    lo, hi = to_px(_log10(0.002), 0)[0], to_px(_log10(0.02), 0)[0]
    b += rect(lo, py[0], hi - lo, py[1] - py[0], fill=YELLOW, op=0.22, stroke="none")
    # Inside the plot rather than above it: both curves are low on the left at
    # these prevalences, so the top of the band is the only clear space, and the
    # y-axis title already occupies the strip above the box.
    b += text((lo + hi) / 2, py[0] + 30, "typical clinical event rates", size=11, op=0.85)

    p_mark = 0.005
    for (sens, spec), col in zip(DETECTORS, cols):
        v = ppv_from_sens_spec(sens, spec, p_mark)
        x, y = to_px(_log10(p_mark), v)
        b += poly([(x, y - 4.5), (x + 4.5, y), (x, y + 4.5), (x - 4.5, y)], fill=col, stroke="none")
    # Below the box, not inside it. The bottom-left looks like clear space and is
    # not: at these prevalences both curves are near zero, so they run straight
    # through it, and the p = 0.5% markers sit there too.
    mid = (px[0] + px[1]) / 2
    b += text(
        mid,
        py[1] + 62,
        "sensitivity and specificity are held excellent throughout;",
        size=12,
        op=0.85,
    )
    b += text(
        mid,
        py[1] + 80,
        "only the rarity of the event changes.",
        size=12,
        op=0.85,
        style="italic",
    )
    return wrap(
        w,
        h,
        "base",
        "Predictive value collapses at low prevalence",
        "Two curves showing the probability that an alarm is genuine falling "
        "towards zero as the event being detected becomes rare, even for "
        "detectors with excellent sensitivity and specificity.",
        b,
    )


def fig_privacy_optics():
    """Privacy designed into the optics rather than applied to the recording."""
    w, h = 700, 330
    b = ""
    y = 96
    bw, bh = 116, 60
    xs = [40, 196, 352, 508]
    labs = [
        ("scene", "what is in\nthe room", INK),
        ("lens φ", "optimised, not\nchosen", TEAL),
        ("capture", "the only image\never stored", INDIGO),
        ("encoder", "shared\nbackbone", INK),
    ]
    for x, (t, sub, col) in zip(xs, labs):
        b += rect(x, y, bw, bh, fill=col, op=0.1, stroke=col, so=0.6, rx=6)
        b += text(x + bw / 2, y + 26, t, size=14, fill=col, weight="600")
        for i, ln in enumerate(sub.split("\n")):
            b += text(x + bw / 2, y + 44 + i * 13, ln, size=10, op=0.75)
    for i in range(3):
        b += arrow(xs[i] + bw, y + bh / 2, xs[i + 1] - 6, y + bh / 2, sw=1.5)

    hx = 640
    b += arrow(xs[3] + bw, y + bh / 2, hx - 6, y - 4, sw=1.5, stroke=TEAL)
    b += arrow(xs[3] + bw, y + bh / 2, hx - 6, y + bh + 26, sw=1.5, stroke=PURPLE)
    b += text(hx, y - 8, "action", size=13, fill=TEAL, anchor="start", weight="600")
    b += text(hx, y + 8, "maximise", size=10, fill=TEAL, anchor="start", op=0.85)
    b += text(hx, y + bh + 22, "identity", size=13, fill=PURPLE, anchor="start", weight="600")
    b += text(hx, y + bh + 38, "minimise", size=10, fill=PURPLE, anchor="start", op=0.85)

    b += arrow(xs[1] + bw / 2, y + bh + 40, xs[1] + bw / 2, y + bh + 8, sw=1.4, stroke=TEAL, dash="4 3")
    b += text(xs[1] + bw / 2, y + bh + 58, "gradients reach the glass", size=11, fill=TEAL)
    b += text(
        w / 2,
        h - 34,
        "The degradation happens in the optics, so the sensitive image is never captured —",
        size=12,
        op=0.85,
    )
    b += text(
        w / 2,
        h - 16,
        "there is no raw recording to leak, subpoena, or mishandle.",
        size=12,
        op=0.85,
        style="italic",
    )
    return wrap(
        w,
        h,
        "priv",
        "Privacy built into the lens",
        "A left to right pipeline from scene through an optimised lens to a "
        "degraded capture and a shared encoder, feeding an action head that is "
        "maximised and an identity head that is minimised, with gradients "
        "flowing back to the lens parameters.",
        b,
    )


FIGURES = {
    "confusion_table.svg": fig_confusion_table,
    "fairness_impossible.svg": fig_fairness_impossible,
    "compas.svg": fig_compas,
    "impossibility_residual.svg": fig_impossibility_residual,
    "base_rate.svg": fig_base_rate,
    "privacy_optics.svg": fig_privacy_optics,
}


if __name__ == "__main__":
    n_tables, worst_id = check_identity_against_counts()
    worst_ratio, smallest = check_impossibility()
    eq, perf, degen, unequal = check_escape_hatches()
    derived, ppv_bar, predicted = check_compas()
    worst_ppv = check_ppv_collapse()
    worst_inv = check_specificity_inverse()
    peak = check_curve_extents()

    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print(f"wrote {name}")

    print("\n--- A1  identity against direct cell ratios ---")
    print(f"integer tables checked      {n_tables}")
    print(f"worst disagreement          {worst_id:.2e}")

    print("\n--- A2  the impossibility ---")
    print(f"worst error in ratio law    {worst_ratio:.2e}")
    print(f"smallest off-diagonal gap   {smallest:.4f}  (never zero)")

    print("\n--- A3  where the residual is allowed to vanish ---")
    print(f"equal base rates            {eq:.2e}")
    print(f"perfect prediction          {perf:.2e}")
    print(f"degenerate, flags nobody    {degen:.2e}")
    print(f"COMPAS base rates, PPV .70  {unequal:.4f}  <- the real case")

    print("\n--- A4  COMPAS, predicted from prevalence alone ---")
    for g in COMPAS:
        print(
            f"{g:<6} p={COMPAS[g]['p']:.2f}  FNR={COMPAS[g]['fnr']:.2f}  "
            f"PPV derived {derived[g]:.3f}   "
            f"FPR predicted {predicted[g]:.3f} vs published {COMPAS[g]['fpr']:.2f}"
        )
    print(f"mean PPV used               {ppv_bar:.3f}")
    print(f"PPV spread between groups   {abs(derived['Black'] - derived['White']):.3f}")
    print(f"published FPR ratio         {COMPAS['Black']['fpr'] / COMPAS['White']['fpr']:.2f}x")
    print(f"odds ratio of base rates    {odds(COMPAS['Black']['p']) / odds(COMPAS['White']['p']):.2f}x")

    print("\n--- A5  predictive value at low prevalence ---")
    print(f"worst Bayes vs counts       {worst_ppv:.2e}")
    for sens, spec in DETECTORS:
        for prev in (0.2, 0.02, 0.005):
            v = ppv_from_sens_spec(sens, spec, prev)
            print(
                f"sens {sens:.0%} spec {spec:.0%}  prevalence {prev:>6.1%}  "
                f"PPV {v:.3f}   {alarms_per_true_case(sens, spec, prev):5.1f} alarms per real case"
            )

    print("\n--- A6  the operating room, at its real base rate ---")
    print(f"worst inverse disagreement  {worst_inv:.2e}")
    print(f"retained item prevalence    1 in 5500  ({RETAINED_ITEM_RATE:.5%})")
    for spec in (0.99, 0.999, 0.9999):
        v = ppv_from_sens_spec(0.99, spec, RETAINED_ITEM_RATE)
        print(
            f"sens 99% spec {spec:.2%}  PPV {v:.4f}   "
            f"{alarms_per_true_case(0.99, spec, RETAINED_ITEM_RATE):7.1f} counts per real retention"
        )
    for target in (0.5, 0.9):
        print(
            f"to reach PPV {target:.0%} needs specificity {specificity_for_ppv(target, 0.99, RETAINED_ITEM_RATE):.6%}"
        )

    print("\n--- extents ---")
    print(f"fairness curve peak         {peak:.3f}  (axis top {FAIR_RNG[1]})")
