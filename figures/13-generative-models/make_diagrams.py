"""Generate the twelve SVG diagrams for the lecture 13 chapter.

Four of them are computed rather than drawn, and they are the reason this file
exists rather than a set of hand-placed shapes:

`diagram_forward_reverse_kl` fits a single Gaussian to a bimodal target under
each direction of the KL divergence. The forward fit is moment matching, which
is what minimising D(p||q) over Gaussians reduces to; the reverse fit is found
by grid search, restricted to mu >= 0 because the target is symmetric and the
reverse-KL objective therefore has two equivalent minima.

`diagram_sampling_cost` is H*W*C, which is the whole cost model for
autoregressive sampling over raw subpixels.

`diagram_elbo_gap` and `diagram_sigma_tension` share a one-dimensional
conjugate model — p(z) = N(0,1), p(x|z) = N(z, sx^2) — in which the marginal,
the true posterior, and all three terms of the ELBO decomposition have closed
forms. That makes two claims in the prose checkable rather than asserted: that
log p(x) = ELBO + posterior KL exactly, and that maximising the ELBO over the
encoder's output lands on the true posterior, mu = x/(1+sx^2) and
sigma^2 = sx^2/(1+sx^2). The `__main__` block verifies both against a grid
search, and prints the limit as sx grows, which is posterior collapse.

`diagram_bits_per_dim` replots Table 5 of the PixelRNN paper. The values are
stated in the chapter prose as well, so nothing here is the only record of
them.

The remaining seven are geometry: the density/representation spectrum, the two
normalisation directions, the four-leaf taxonomy, the raster causal context and
the mask that enforces it, the autoencoder's missing latent distribution, the
generative and inference directions, and the two computation graphs that the
reparameterisation trick distinguishes.

Conventions follow figures/12-self-supervised-learning: currentColor for text
and rules, viridis for data, title/desc with namespaced ids, no background rect
(the theme supplies a light plate in both schemes). Yellow is a fill only,
never a line or a label.

Run from the repository root; rewrites all twelve files.
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
    arrow,
    circle,
    line,
    mapper,
    polyline,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# The one-dimensional conjugate model shared by the ELBO figures.
#
#   p(z)   = N(0, 1)
#   p(x|z) = N(z, SX^2)
#
# so the marginal is N(0, 1 + SX^2) and the posterior is Gaussian too. Every
# term below is exact; nothing is estimated.
# --------------------------------------------------------------------------
X_OBS = 1.3
SX = 0.8


def log_marginal(x=X_OBS, sx=SX):
    """Log p(x), in nats, for the conjugate model."""
    v = 1.0 + sx * sx
    return -0.5 * math.log(2 * math.pi * v) - x * x / (2 * v)


def true_posterior(x=X_OBS, sx=SX):
    """Mean and variance of p(z|x), which the ELBO optimum should reproduce."""
    return x / (1.0 + sx * sx), sx * sx / (1.0 + sx * sx)


def recon_term(mq, sq, x=X_OBS, sx=SX):
    """E_q[log p(x|z)], the ELBO's reconstruction term."""
    return -0.5 * math.log(2 * math.pi * sx * sx) - ((x - mq) ** 2 + sq * sq) / (2 * sx * sx)


def kl_to_prior(mq, sq):
    """D(q(z|x) || N(0,1)), closed form because both are diagonal Gaussians."""
    return 0.5 * (sq * sq + mq * mq - 1.0 - 2 * math.log(sq))


def kl_to_posterior(mq, sq, x=X_OBS, sx=SX):
    """D(q(z|x) || p(z|x)) — the term the ELBO discards, computable only here."""
    mp, vp = true_posterior(x, sx)
    return 0.5 * (sq * sq / vp + (mq - mp) ** 2 / vp - 1.0 - math.log(sq * sq / vp))


def elbo(mq, sq, x=X_OBS, sx=SX):
    """The evidence lower bound for one encoder output."""
    return recon_term(mq, sq, x, sx) - kl_to_prior(mq, sq)


# The three encoder settings the ELBO figure compares. The first is exact, the
# last ignores the input entirely, and the middle one is somewhere between.
_MP, _VP = true_posterior()
ELBO_CASES = [
    ("q = true posterior", _MP, math.sqrt(_VP)),
    ("q off by a little", 0.35, 0.95),
    ("q = the prior", 0.0, 1.0),
]


def gaussian(x, mu, sigma):
    """Density of N(mu, sigma^2) at x."""
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


# --------------------------------------------------------------------------
# The bimodal target for the KL-direction figure, and the two fits.
# --------------------------------------------------------------------------
MIX_SEP = 1.6
MIX_SD = 0.55

# Shared y range for both KL panels. The reverse-KL fit is narrow and therefore
# tall — it peaks near 0.77 — so a range fitted to the target alone clips it off
# the canvas, and putting the panels on different scales would hide that its
# height is the whole point.
KL_YMAX = 0.85


def mixture(x):
    """The bimodal target: an even mixture of two Gaussians."""
    return 0.5 * gaussian(x, -MIX_SEP, MIX_SD) + 0.5 * gaussian(x, MIX_SEP, MIX_SD)


def forward_kl_fit():
    """Best Gaussian under D(p||q): moment matching, which is exact here."""
    mean = 0.0
    var = MIX_SD**2 + MIX_SEP**2
    return mean, math.sqrt(var)


def reverse_kl_fit():
    """Best Gaussian under D(q||p), by grid search over mu >= 0.

    The target is symmetric, so this objective has two equivalent minima; the search is restricted to the right half so
    the figure is deterministic.
    """
    best = None
    for i in range(0, 601):
        mu = i * 0.005
        for j in range(1, 501):
            sd = j * 0.005
            # D(q||p) up to the constant entropy of q: integrate numerically.
            total = 0.0
            lo, hi, n = mu - 6 * sd, mu + 6 * sd, 900
            step = (hi - lo) / n
            for k in range(n + 1):
                z = lo + k * step
                q = gaussian(z, mu, sd)
                if q < 1e-12:
                    continue
                total += q * (math.log(q) - math.log(max(mixture(z), 1e-300))) * step
            if best is None or total < best[0]:
                best = (total, mu, sd)
    return best[1], best[2]


# Grid searching a numerical integral inside a double loop takes minutes, which
# is too slow for every build. So `reverse_kl_fit` above is the definition and
# these are the values it returned; `__main__` re-runs a coarse pass over the
# same objective every build, which lands within two grid steps of them.
REVERSE_FIT = (1.5975, 0.5215)


def reverse_kl_fit_coarse():
    """A cheap pass over the same objective, to confirm REVERSE_FIT is right."""
    best = None
    for i in range(0, 41):
        mu = i * 0.05
        for j in range(1, 41):
            sd = j * 0.025
            total = 0.0
            lo, hi, n = mu - 6 * sd, mu + 6 * sd, 400
            step = (hi - lo) / n
            for k in range(n + 1):
                z = lo + k * step
                q = gaussian(z, mu, sd)
                if q < 1e-12:
                    continue
                total += q * (math.log(q) - math.log(max(mixture(z), 1e-300))) * step
            if best is None or total < best[0]:
                best = (total, mu, sd)
    return best[1], best[2]


# --------------------------------------------------------------------------
# CIFAR-10 density, Table 5 of van den Oord et al., "Pixel Recurrent Neural
# Networks", ICML 2016. Test-set bits per dimension; lower is better. The
# uniform baseline is exactly 8 because a subpixel is one byte.
# --------------------------------------------------------------------------
BITS_PER_DIM = [
    ("uniform over 256 values", 8.00, INK),
    ("multivariate Gaussian", 4.70, PURPLE),
    ("RIDE (previous best)", 3.47, PURPLE),
    ("PixelCNN", 3.14, TEAL),
    ("Row LSTM", 3.07, TEAL),
    ("Diagonal BiLSTM", 3.00, TEAL),
]

CHANNELS = 3


def sampling_passes(side, channels=CHANNELS):
    """Sequential decoder evaluations for one autoregressive image sample."""
    return side * side * channels


def fig_spectrum():
    """The density/representation spectrum, and the second axis crossing it."""
    w, h = 760, 300
    b = text(20, 28, "Two axes, and they are independent", size=14, anchor="start")
    b += text(20, 48, "where a method sits, and what it puts in competition", size=11.5, anchor="start", op=0.6)

    ax0, ax1, ay = 60, 700, 128
    b += line(ax0, ay, ax1, ay, op=0.35)
    for frac in (0.0, 1.0):
        gx = ax0 + frac * (ax1 - ax0)
        b += line(gx, ay - 5, gx, ay + 5, op=0.35)
    b += text(ax0, ay + 24, "model the full density p(x)", size=11.5, anchor="start", op=0.65)
    b += text(ax1, ay + 24, "extract a short code z", size=11.5, anchor="end", op=0.65)

    chips = [
        (0.05, "autoregressive", TEAL),
        (0.38, "VAE", INDIGO),
        (0.62, "autoencoder", PURPLE),
        (0.80, "PCA", PURPLE),
        (0.95, "contrastive", PURPLE),
    ]
    for frac, name, colour in chips:
        cx = ax0 + frac * (ax1 - ax0)
        b += line(cx, ay - 4, cx, ay - 18, stroke=colour, op=0.55)
        b += circle(cx, ay, 4.0, fill=colour, stroke="none")
        b += text(cx, ay - 26, name, size=12, fill=colour)

    b += text(380, 200, "a separate question: what competes for probability mass", size=11.5, op=0.65)
    boxes = [
        (90, "p(y | x)", "discriminative", GREEN),
        (330, "p(x)", "generative", INDIGO),
        (570, "p(x | y)", "conditional", TEAL),
    ]
    for bx, formula, kind, colour in boxes:
        b += rect(bx, 220, 150, 48, fill=colour, op=0.14, stroke=colour, so=0.6, rx=4)
        b += text(bx + 75, 240, formula, size=13, font=MONO)
        b += text(bx + 75, 258, kind, size=11, op=0.65)

    return wrap(
        w,
        h,
        "sp",
        "Two independent axes for classifying methods",
        "A horizontal spectrum from full density estimation to short-code representation learning, with autoregressive models at the density end, autoencoders and PCA and contrastive methods at the representation end, and the variational autoencoder between them. Below it, three boxes for the discriminative, generative and conditional generative arrangements, marked as a separate question.",
        b,
    )


def fig_probability_mass():
    """The two normalisation directions: across labels, or across images."""
    w, h = 760, 330
    b = text(20, 28, "Which direction sums to one", size=14, anchor="start")
    b += text(20, 48, "the same total, apportioned along different axes", size=11.5, anchor="start", op=0.6)
    b += line(380, 70, 380, 300, op=0.14, dash="4 5")

    base, full = 250, 140

    b += text(190, 88, "discriminative  p(y | x)", size=12.5)
    columns = [
        (90, "cat photo", 0.92),
        (190, "monkey photo", 0.55),
        (290, "dog photo", 0.05),
    ]
    for cx, name, p_cat in columns:
        cat_h = p_cat * full
        dog_h = full - cat_h
        b += rect(cx - 30, base - full, 60, dog_h, fill=INDIGO, op=0.78, stroke="none")
        b += rect(cx - 30, base - cat_h, 60, cat_h, fill=GREEN, op=0.78, stroke="none")
        b += rect(cx - 30, base - full, 60, full, fill="none", so=0.3)
        if cat_h > 26:
            b += text(cx, base - cat_h / 2 + 4, "cat", size=11, fill=INK, op=0.8)
        if dog_h > 26:
            b += text(cx, base - cat_h - dog_h / 2 + 4, "dog", size=11, fill=INK, op=0.8)
        b += text(cx, base + 18, name, size=11, op=0.7)
    b += text(190, base + 42, "each column sums to 1", size=11.5, fill=INDIGO)

    b += text(570, 88, "generative  p(x)", size=12.5)
    bars = [
        (450, "cat", 0.40),
        (515, "dog", 0.38),
        (580, "monkey", 0.20),
        (645, "art", 0.02),
    ]
    scale = full / 0.40
    for cx, name, mass in bars:
        bh = mass * scale
        b += rect(cx - 22, base - bh, 44, bh, fill=PURPLE, op=0.78, stroke="none")
        b += text(cx, base + 18, name, size=11, op=0.7)
    b += line(428, base, 668, base, op=0.35)
    b += text(570, base + 42, "the four bars sum to 1", size=11.5, fill=PURPLE)

    return wrap(
        w,
        h,
        "pm",
        "Two directions of normalisation",
        "On the left, three stacked bars, one per input image, each summing to one across the cat and dog labels, so the monkey photograph must still be split between them. On the right, four bars whose heights sum to one across the images, so the abstract art can be given almost no mass.",
        b,
    )


def fig_taxonomy():
    """The four leaves, split by which operation the model supports."""
    w, h = 760, 350
    b = text(20, 28, "Generative models, by what they will compute", size=14, anchor="start")

    def node(cx, cy, bw, bh, lines, colour, sizes):
        out = rect(cx - bw / 2, cy - bh / 2, bw, bh, fill=colour, op=0.13, stroke=colour, so=0.65, rx=4)
        n = len(lines)
        for i, (s, sz) in enumerate(zip(lines, sizes)):
            out += text(cx, cy - bh / 2 + 20 + i * 19, s, size=sz)
        del n
        return out

    b += node(380, 70, 180, 38, ["generative model"], INK, [13])
    b += node(195, 168, 168, 38, ["explicit density"], INDIGO, [13])
    b += node(565, 168, 168, 38, ["implicit density"], PURPLE, [13])

    b += arrow(380, 89, 205, 147, op=0.5)
    b += arrow(380, 89, 555, 147, op=0.5)
    b += text(258, 112, "can compute p(x)", size=11, anchor="end", op=0.6)
    b += text(502, 112, "cannot, but can sample", size=11, anchor="start", op=0.6)

    leaves = [
        (105, "tractable density", "autoregressive", TEAL),
        (285, "approximate density", "VAE", INDIGO),
        (475, "direct", "GAN", PURPLE),
        (655, "iterative", "diffusion", GREEN),
    ]
    for cx, top_line, bottom_line, colour in leaves:
        b += node(cx, 272, 170, 56, [top_line, bottom_line], colour, [12, 12.5])

    b += arrow(195, 187, 115, 242, op=0.5)
    b += arrow(195, 187, 275, 242, op=0.5)
    b += arrow(565, 187, 485, 242, op=0.5)
    b += arrow(565, 187, 645, 242, op=0.5)

    for x0, x1, label in ((20, 370, "this chapter"), (390, 740, "next chapter")):
        b += polyline([(x0, 312), (x0, 320), (x1, 320), (x1, 312)], sw=1.0, op=0.3)
        b += text((x0 + x1) / 2, 340, label, size=11.5, op=0.6)

    return wrap(
        w,
        h,
        "tx",
        "Taxonomy of generative models",
        "A tree rooted at generative model, splitting into explicit density (models that can return p of x) and implicit density (models that cannot but can sample). Explicit splits into tractable, holding autoregressive models, and approximate, holding the variational autoencoder. Implicit splits into direct, holding generative adversarial networks, and iterative, holding diffusion models. Brackets mark the first two leaves as this chapter and the last two as the next.",
        b,
    )


def fig_forward_reverse_kl():
    """Best single Gaussian under each direction of the KL divergence."""
    w, h = 760, 290
    b = text(20, 28, "The same target, the two directions of KL", size=14, anchor="start")

    fm, fs = forward_kl_fit()
    rm, rs = REVERSE_FIT
    panels = [
        (50, 350, "forward KL — what likelihood does", "min D(p &#8214; q)", fm, fs, INDIGO),
        (410, 710, "reverse KL", "min D(q &#8214; p)", rm, rs, TEAL),
    ]
    for x0, x1, title, formula, mu, sd, colour in panels:
        mid = (x0 + x1) / 2
        b += text(mid, 54, title, size=12.5)
        b += text(mid, 72, formula, size=11.5, op=0.6, font=MONO)

        to_px = mapper((-4.0, 4.0), (0.0, KL_YMAX), (x0, x1), (90, 240))
        b += line(x0, 240, x1, 240, op=0.35)
        for tick in (-3, 0, 3):
            gx, _ = to_px(tick, 0)
            b += line(gx, 240, gx, 245, op=0.35)
            b += text(gx, 260, f"{tick}", size=11, op=0.6, font=MONO)

        target = [to_px(-4.0 + i * 8.0 / 240, mixture(-4.0 + i * 8.0 / 240)) for i in range(241)]
        b += polyline(target, sw=1.5, op=0.45)
        fit = [to_px(-4.0 + i * 8.0 / 240, gaussian(-4.0 + i * 8.0 / 240, mu, sd)) for i in range(241)]
        b += polyline(fit, stroke=colour, sw=2.0)

        b += line(x0 + 10, 96, x0 + 32, 96, op=0.45, sw=1.5)
        b += text(x0 + 40, 100, "the data", size=11, anchor="start", op=0.7)
        b += line(x0 + 10, 116, x0 + 32, 116, stroke=colour, sw=2.0)
        b += text(x0 + 40, 120, f"fit: &#963; = {sd:.2f}", size=11, anchor="start", fill=colour)

    b += text(380, 282, "density against x, both panels on one scale; likelihood gives the left panel", size=11.5, op=0.6)
    return wrap(
        w,
        h,
        "kl",
        "Best Gaussian fit under each direction of the KL divergence",
        "Two panels, each showing the same bimodal target density and the single Gaussian that best fits it. Under the forward divergence the fit is broad and centred in the valley between the two modes. Under the reverse divergence the fit is narrow and sits on one mode, ignoring the other.",
        b,
    )


def fig_raster_context():
    """The causal context a raster ordering gives, and the mask that enforces it."""
    w, h = 760, 340
    b = text(20, 28, "What a raster ordering makes available", size=14, anchor="start")
    b += line(372, 68, 372, 300, op=0.14, dash="4 5")

    b += text(180, 62, "raster order over the image", size=12.5)
    cell, cols, rows = 30, 7, 7
    gx0, gy0 = 75, 88
    cur_r, cur_c = 3, 3
    cur_index = cur_r * cols + cur_c
    for r in range(rows):
        for c in range(cols):
            x, y = gx0 + c * cell, gy0 + r * cell
            idx = r * cols + c
            if idx == cur_index:
                b += rect(x, y, cell, cell, fill=TEAL, op=0.85, stroke="none")
            elif idx < cur_index:
                b += rect(x, y, cell, cell, fill=INDIGO, op=0.4, stroke="none")
            b += rect(x, y, cell, cell, fill="none", so=0.25)

    legend = [(48, "generated", INDIGO, 0.4), (150, "current", TEAL, 0.85), (232, "future", None, 0.0)]
    for lx, name, colour, op in legend:
        if colour:
            b += rect(lx, 310, 12, 12, fill=colour, op=op, stroke="none")
        b += rect(lx, 310, 12, 12, fill="none", so=0.25)
        b += text(lx + 18, 320, name, size=11, anchor="start", op=0.7)

    b += text(555, 62, "masked convolution kernel", size=12.5)
    b += text(555, 80, "future weights fixed at zero", size=11, op=0.6)
    kcell, kn = 34, 5
    kx0, ky0 = 470, 100
    kcentre = (kn // 2) * kn + kn // 2
    for r in range(kn):
        for c in range(kn):
            x, y = kx0 + c * kcell, ky0 + r * kcell
            idx = r * kn + c
            if idx == kcentre:
                b += rect(x, y, kcell, kcell, fill=TEAL, op=0.85, stroke="none")
            elif idx < kcentre:
                b += rect(x, y, kcell, kcell, fill=INDIGO, op=0.3, stroke="none")
                b += text(x + kcell / 2, y + kcell / 2 + 4, "w", size=11, op=0.75, font=MONO)
            else:
                b += text(x + kcell / 2, y + kcell / 2 + 4, "0", size=11, op=0.4, font=MONO)
            b += rect(x, y, kcell, kcell, fill="none", so=0.25)
    b += text(555, 292, "the constraint is structural, not learned", size=11.5, op=0.6)

    return wrap(
        w,
        h,
        "rc",
        "Causal context under a raster ordering",
        "On the left, a seven by seven grid of pixels with everything before the current pixel in raster order shaded as available, the current pixel highlighted, and everything after it left blank. On the right, a five by five convolution kernel in which the weights covering the earlier half of the raster order are live and the rest are printed as zero.",
        b,
    )


def fig_bits_per_dim():
    """CIFAR-10 test density in bits per dimension, from the PixelRNN paper."""
    w, h = 760, 306
    x0, x1 = 300, 700
    lo, hi = 2.5, 8.5

    def px(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    b = text(20, 30, "CIFAR-10 density, bits per dimension", size=14, anchor="start")
    b += text(20, 50, "lower is better; a raw subpixel is 8 bits", size=11.5, anchor="start", op=0.6)

    for tick in (3, 4, 5, 6, 7, 8):
        gx = px(tick)
        b += line(gx, 68, gx, 262, op=0.14, dash="3 4")
        b += text(gx, 280, f"{tick}", size=11, op=0.6, font=MONO)

    top, step, bh = 78, 32, 20
    for i, (name, value, colour) in enumerate(BITS_PER_DIM):
        y = top + i * step
        b += text(x0 - 14, y + bh - 6, name, size=12, anchor="end")
        b += rect(x0, y, px(value) - x0, bh, fill=colour, op=0.28 if colour is INK else 0.82, stroke="none")
        b += text(px(value) + 8, y + bh - 6, f"{value:.2f}", size=11.5, anchor="start", font=MONO)

    b += line(x0, 68, x0, 262, op=0.35)
    b += text(500, 298, "bits per dimension", size=11.5, op=0.6)
    return wrap(
        w,
        h,
        "bd",
        "CIFAR-10 density in bits per dimension",
        "Horizontal bars for six models on CIFAR-10, from a uniform baseline at exactly eight bits per dimension down to three point zero zero for the strongest PixelRNN variant, with a multivariate Gaussian at four point seven and the previous state of the art at three point four seven in between.",
        b,
    )


def fig_sampling_cost():
    """Sequential decoder passes per sample, against image side length."""
    w, h = 760, 310
    x0, x1 = 90, 700
    ytop, ybase = 80, 250
    lx0, lx1 = math.log10(16), math.log10(2048)
    ly0, ly1 = 2.5, 7.5

    def px(side):
        return x0 + (math.log10(side) - lx0) / (lx1 - lx0) * (x1 - x0)

    def py(passes):
        return ybase - (math.log10(passes) - ly0) / (ly1 - ly0) * (ybase - ytop)

    b = text(20, 28, "Sequential decoder passes for one sample", size=14, anchor="start")
    b += text(20, 48, "three channels, one pass per subpixel; both axes logarithmic", size=11.5, anchor="start", op=0.6)

    for tick in (16, 64, 256, 1024):
        gx = px(tick)
        b += line(gx, ytop, gx, ybase, op=0.14, dash="3 4")
        b += text(gx, 268, f"{tick}", size=11, op=0.6, font=MONO)
    for value, label in ((1e3, "1,000"), (1e4, "10,000"), (1e5, "100,000"), (1e6, "1M"), (1e7, "10M")):
        gy = py(value)
        b += line(x0, gy, x1, gy, op=0.1, dash="3 4")
        b += text(x0 - 10, gy + 4, label, size=11, anchor="end", op=0.6, font=MONO)

    curve = []
    for i in range(121):
        side = 16 * (2048 / 16) ** (i / 120)
        curve.append((px(side), py(sampling_passes(side))))
    b += polyline(curve, stroke=INDIGO, sw=2.2)

    b += line(x0, ybase, x1, ybase, op=0.35)
    b += line(x0, ytop, x0, ybase, op=0.35)

    marks = [
        (32, f"32&#215;32 — {sampling_passes(32):,} passes", 20, "start"),
        (64, f"64&#215;64 — {sampling_passes(64):,}", 12, "start"),
    ]
    for side, label, dy, anchor in marks:
        mx, my = px(side), py(sampling_passes(side))
        b += circle(mx, my, 4.5, fill=INDIGO, stroke="none")
        b += text(mx + 11, my + dy, label, size=11.5, anchor=anchor, fill=INDIGO)

    mx, my = px(1024), py(sampling_passes(1024))
    b += circle(mx, my, 4.5, fill=PURPLE, stroke="none")
    b += text(
        mx - 12, my - 16, f"1024&#215;1024 — {sampling_passes(1024):,} passes", size=11.5, anchor="end", fill=PURPLE
    )

    b += text(395, 290, "image side length in pixels", size=11.5, op=0.6)
    return wrap(
        w,
        h,
        "sc",
        "Sampling cost against image resolution",
        "A straight line on log-log axes showing sequential decoder evaluations per sample rising quadratically with image side length, from about three thousand passes at thirty-two pixels square to over three million at one thousand and twenty-four pixels square.",
        b,
    )


def fig_autoencoder_gap():
    """What reconstruction constrains, and the latent distribution it does not."""
    w, h = 760, 330
    b = text(20, 28, "An autoencoder trains two maps and no distribution", size=14, anchor="start")

    lane = 108
    b += polyline([(70, lane - 22), (70, 66), (550, 66), (550, lane - 22)], sw=1.2, op=0.4, dash="4 4")
    b += text(310, 58, "L&#178; reconstruction loss", size=11.5, op=0.7)

    nodes = [
        (70, 68, "x", None),
        (190, 88, "encoder", INDIGO),
        (310, 44, "z", TEAL),
        (430, 88, "decoder", INDIGO),
        (550, 68, "x&#770;", None),
    ]
    for cx, bw, label, colour in nodes:
        fill = colour if colour else INK
        op = 0.14 if colour else 0.06
        b += rect(cx - bw / 2, lane - 20, bw, 40, fill=fill, op=op, stroke=colour or INK, so=0.5, rx=4)
        b += text(cx, lane + 5, label, size=13)
    for a, z in ((104, 146), (234, 288), (332, 386), (474, 516)):
        b += arrow(a, lane, z, lane, op=0.6)

    b += text(200, 172, "latent space", size=12.5)
    b += rect(60, 182, 280, 118, fill="none", so=0.35, rx=4)
    for i in range(44):
        t = (i % 22) / 21
        # A spine, plus a deterministic offset normal to it, so the cluster has
        # width: a region, not a curve. Nothing in the loss controls its shape.
        radius = 9.0 + 12.0 * abs(math.sin(2.3 * i))
        angle = 2.399 * i
        cx = 108 + 148 * t + 22 * math.sin(4.4 * t) + radius * math.cos(angle)
        cy = 259 - 52 * t + 12 * math.cos(6.1 * t) + radius * math.sin(angle) * 0.55
        b += circle(cx, cy, 3.4, fill=TEAL, op=0.7, stroke="none")
    b += text(196, 292, "where training put the codes", size=11, op=0.6)

    sx, sy = 306, 204
    b += line(sx - 6, sy - 6, sx + 6, sy + 6, stroke=PURPLE, sw=2.0)
    b += line(sx - 6, sy + 6, sx + 6, sy - 6, stroke=PURPLE, sw=2.0)
    b += text(300, 194, "a random z", size=11, fill=PURPLE)

    b += arrow(sx + 10, sy + 8, 452, 236, stroke=PURPLE, op=0.7, dash="4 4")
    b += rect(462, 218, 88, 40, fill=INDIGO, op=0.14, stroke=INDIGO, so=0.5, rx=4)
    b += text(506, 243, "decoder", size=13)
    b += arrow(554, 238, 594, 238, op=0.6)
    b += rect(604, 218, 96, 40, fill="none", so=0.35, rx=4)
    b += text(652, 244, "?", size=16, op=0.7)

    return wrap(
        w,
        h,
        "ag",
        "The autoencoder's missing latent distribution",
        "Above, a pipeline from an image through an encoder to a code and back through a decoder to a reconstruction, with a dashed reconstruction loss joining the two ends. Below, a box representing latent space holds a curved cluster of the codes training produced, and a cross marks a randomly drawn code well outside the cluster, whose path through the decoder ends in a question mark.",
        b,
    )


def fig_vae_directions():
    """The generative direction the model contains, and the inference direction it lacks."""
    w, h = 760, 250
    b = text(20, 28, "One model, two directions", size=14, anchor="start")

    b += rect(115, 127, 130, 46, fill=INDIGO, op=0.14, stroke=INDIGO, so=0.6, rx=4)
    b += text(180, 150, "z", size=16)
    b += text(180, 166, "latent code", size=11, op=0.65)
    b += rect(515, 127, 130, 46, fill=TEAL, op=0.14, stroke=TEAL, so=0.6, rx=4)
    b += text(580, 150, "x", size=16)
    b += text(580, 166, "image", size=11, op=0.65)

    b += text(180, 196, "prior p(z) = N(0, I)", size=11, op=0.65)

    b += arrow(250, 120, 510, 120, stroke=TEAL, op=0.85, sw=1.8)
    b += text(380, 108, "decoder  p(x | z)", size=12.5, fill=TEAL)
    b += text(380, 90, "a network we can evaluate", size=11, op=0.6)

    b += arrow(510, 182, 250, 182, stroke=PURPLE, op=0.85, sw=1.8, dash="6 4")
    b += text(380, 202, "posterior  p(z | x)", size=12.5, fill=PURPLE)
    b += text(380, 220, "needs the intractable integral", size=11, op=0.6)

    return wrap(
        w,
        h,
        "vd",
        "Generative and inference directions",
        "A box for the latent code on the left and a box for the image on the right. A solid arrow from code to image is labelled as the decoder, a network that can be evaluated. A dashed arrow from image back to code is labelled as the posterior, which needs the intractable integral.",
        b,
    )


def fig_elbo_gap():
    """Log p(x) = ELBO + posterior KL, with all three terms in closed form."""
    w, h = 760, 310
    base, top_y = 260, 80
    ymax = 3.0

    def py(v):
        return base - v / ymax * (base - top_y)

    b = text(20, 28, "The bound and the gap sum to log p(x)", size=14, anchor="start")
    b += text(
        20,
        48,
        "negated so bars are positive; nats, one-dimensional conjugate model",
        size=11.5,
        anchor="start",
        op=0.6,
    )

    neg_lp = -log_marginal()
    for tick in (0, 1, 2, 3):
        gy = py(tick)
        b += line(100, gy, 700, gy, op=0.1, dash="3 4")
        b += text(92, gy + 4, f"{tick}", size=11, anchor="end", op=0.6, font=MONO)

    for i, (name, mq, sq) in enumerate(ELBO_CASES):
        cx = 170 + i * 200
        gap = kl_to_posterior(mq, sq)
        neg_elbo = neg_lp + gap
        b += rect(cx - 45, py(neg_lp), 90, base - py(neg_lp), fill=INDIGO, op=0.8, stroke="none")
        if gap > 1e-9:
            b += rect(cx - 45, py(neg_elbo), 90, py(neg_lp) - py(neg_elbo), fill=PURPLE, op=0.8, stroke="none")
        b += text(cx, py(neg_elbo) - 10, f"{neg_elbo:.3f}", size=11.5, font=MONO)
        b += text(cx, base + 20, name, size=11.5, op=0.75)
        b += text(cx, base + 38, f"gap {gap:.3f}", size=11, op=0.6, font=MONO)

    b += line(100, py(neg_lp), 700, py(neg_lp), stroke=INDIGO, sw=1.4, dash="5 4")
    b += line(100, base, 700, base, op=0.35)
    b += line(100, top_y, 100, base, op=0.35)

    b += line(118, 96, 140, 96, stroke=PURPLE, sw=6.0)
    b += text(148, 100, "gap = D(q &#8214; p(z | x))", size=11, anchor="start", fill=PURPLE)
    b += line(118, 116, 140, 116, stroke=INDIGO, sw=6.0)
    b += text(148, 120, f"&#8722; log p(x) = {neg_lp:.3f}, fixed", size=11, anchor="start", fill=INDIGO)

    return wrap(
        w,
        h,
        "eg",
        "The evidence lower bound and its gap",
        "Three stacked bars, one per choice of encoder output. Every bar has the same lower segment, the negated log marginal likelihood, drawn against a dashed rule. The upper segment is the posterior divergence, zero when the encoder equals the true posterior and growing as it moves away, so the total height is the negated bound.",
        b,
    )


def optimal_encoder(sx=SX, x=X_OBS):
    """Grid search for the ELBO-maximising encoder output, as a check."""
    best = None
    for i in range(1, 2001):
        sq = i * 0.001
        for j in range(0, 2001):
            mq = j * 0.001
            v = elbo(mq, sq, x, sx)
            if best is None or v > best[0]:
                best = (v, mq, sq)
    return best[1], best[2]


def fig_sigma_tension():
    """The two ELBO terms against the encoder's predicted standard deviation."""
    w, h = 760, 312
    x0, x1 = 90, 700
    base, top_y = 260, 80
    smax, ymax = 2.1, 4.7
    mq, _ = true_posterior()

    def px(s):
        return x0 + s / smax * (x1 - x0)

    def py(v):
        return base - v / ymax * (base - top_y)

    def recon_cost(sq):
        """Reconstruction cost with the constant term dropped."""
        return ((X_OBS - mq) ** 2 + sq * sq) / (2 * SX * SX)

    b = text(20, 28, "The two terms of the ELBO, against &#963;", size=14, anchor="start")
    b += text(
        20,
        48,
        f"one-dimensional model, x = {X_OBS}, decoder noise &#963;&#8339; = {SX}; nats, constants dropped",
        size=11.5,
        anchor="start",
        op=0.6,
    )

    for tick in (0.0, 0.5, 1.0, 1.5, 2.0):
        gx = px(tick)
        b += line(gx, top_y, gx, base, op=0.12, dash="3 4")
        b += text(gx, 278, f"{tick:g}", size=11, op=0.6, font=MONO)

    lo, hi, n = 0.08, 2.0, 200
    recon = []
    prior = []
    total = []
    for i in range(n + 1):
        s = lo + (hi - lo) * i / n
        recon.append((px(s), py(recon_cost(s))))
        prior.append((px(s), py(kl_to_prior(mq, s))))
        total.append((px(s), py(recon_cost(s) + kl_to_prior(mq, s))))
    b += polyline(recon, stroke=TEAL, sw=1.9)
    b += polyline(prior, stroke=PURPLE, sw=1.9)
    b += polyline(total, stroke=INDIGO, sw=2.6)

    b += line(x0, base, x1, base, op=0.35)
    b += line(x0, top_y, x0, base, op=0.35)

    b += line(110, 92, 132, 92, stroke=TEAL, sw=1.9)
    b += text(140, 96, "reconstruction cost", size=11, anchor="start", fill=TEAL)
    b += line(110, 112, 132, 112, stroke=PURPLE, sw=1.9)
    b += text(140, 116, "prior KL", size=11, anchor="start", fill=PURPLE)
    b += line(110, 132, 132, 132, stroke=INDIGO, sw=2.6)
    b += text(140, 136, "their sum", size=11, anchor="start", fill=INDIGO)

    _, vp = true_posterior()
    s_star = math.sqrt(vp)
    b += line(px(s_star), py(recon_cost(s_star) + kl_to_prior(mq, s_star)) - 6, px(s_star), base, op=0.3, dash="4 4")
    b += circle(px(s_star), py(recon_cost(s_star) + kl_to_prior(mq, s_star)), 5.0, fill=INDIGO, stroke="none")
    b += text(px(s_star) + 14, 184, "the true posterior &#963;", size=11, anchor="start", fill=INDIGO)
    b += text(px(s_star) + 14, 200, f"&#963;* = {s_star:.3f}", size=11, anchor="start", fill=INDIGO, font=MONO)

    b += text(395, 300, "encoder's predicted standard deviation", size=11.5, op=0.6)
    return wrap(
        w,
        h,
        "st",
        "The two ELBO terms in tension",
        "Reconstruction cost rises steadily with the encoder's predicted standard deviation while the prior divergence falls to a minimum at one and rises again, and their sum has an interior minimum at about zero point six two five, marked, which is the standard deviation of the true posterior.",
        b,
    )


def fig_reparameterisation():
    """Two computation graphs: sampling in the path, and noise as an input."""
    w, h = 760, 290
    b = text(20, 28, "Where the randomness sits", size=14, anchor="start")
    b += line(375, 44, 375, 275, op=0.14, dash="4 5")

    b += text(195, 62, "sampling in the path", size=12.5)
    b += text(195, 80, "no gradient reaches &#956; or &#963;", size=11, op=0.6)
    for cy, label in ((115, "&#956;"), (175, "&#963;")):
        b += rect(52, cy - 15, 46, 30, fill=INDIGO, op=0.14, stroke=INDIGO, so=0.6, rx=4)
        b += text(75, cy + 5, label, size=13)
    b += rect(137, 127, 96, 36, fill=PURPLE, op=0.14, stroke=PURPLE, so=0.6, rx=4)
    b += text(185, 150, "sample", size=12.5)
    b += arrow(98, 118, 133, 138, op=0.6)
    b += arrow(98, 172, 133, 152, op=0.6)
    b += circle(300, 145, 20, fill=TEAL, op=0.2, stroke=TEAL, so=0.6)
    b += text(300, 150, "z", size=14)
    b += arrow(233, 145, 276, 145, op=0.6)

    b += arrow(292, 240, 200, 240, stroke=PURPLE, op=0.8, dash="5 4")
    b += line(178, 232, 192, 248, stroke=PURPLE, sw=2.4)
    b += line(178, 248, 192, 232, stroke=PURPLE, sw=2.4)
    b += text(160, 244, "blocked", size=11, anchor="end", fill=PURPLE)

    b += text(570, 62, "noise as an input", size=12.5)
    b += text(570, 80, "gradient reaches both", size=11, op=0.6)
    for cy, label in ((112, "&#956;"), (160, "&#963;")):
        b += rect(422, cy - 14, 46, 28, fill=INDIGO, op=0.14, stroke=INDIGO, so=0.6, rx=4)
        b += text(445, cy + 5, label, size=13)
    b += rect(414, 194, 76, 28, fill=INK, op=0.06, stroke=INK, so=0.4, rx=4)
    b += text(452, 213, "&#949; ~ N(0, I)", size=11.5)

    b += circle(545, 178, 16, fill=GREEN, op=0.25, stroke=GREEN, so=0.6)
    b += text(545, 183, "&#215;", size=14)
    b += circle(630, 130, 16, fill=GREEN, op=0.25, stroke=GREEN, so=0.6)
    b += text(630, 136, "+", size=14)
    b += circle(705, 130, 20, fill=TEAL, op=0.2, stroke=TEAL, so=0.6)
    b += text(705, 135, "z", size=14)

    b += arrow(468, 164, 530, 174, op=0.6)
    b += arrow(490, 205, 533, 188, op=0.6)
    b += arrow(556, 168, 615, 140, op=0.6)
    b += arrow(468, 114, 614, 126, op=0.6)
    b += arrow(646, 130, 683, 130, op=0.6)

    b += arrow(700, 240, 450, 240, stroke=TEAL, op=0.85, dash="5 4")
    b += text(575, 262, "gradient flows", size=11, fill=TEAL)

    return wrap(
        w,
        h,
        "rp",
        "The reparameterisation trick as two computation graphs",
        "On the left, the mean and standard deviation feed a sampling node which produces the code, and the backward path is drawn crossed out at the sampling node. On the right, the standard deviation is multiplied by an independent noise input and the mean is added, giving the same code, and the backward path runs all the way back to both parameters.",
        b,
    )


FIGURES = {
    "diagram_spectrum.svg": fig_spectrum,
    "diagram_probability_mass.svg": fig_probability_mass,
    "diagram_taxonomy.svg": fig_taxonomy,
    "diagram_forward_reverse_kl.svg": fig_forward_reverse_kl,
    "diagram_raster_context.svg": fig_raster_context,
    "diagram_bits_per_dim.svg": fig_bits_per_dim,
    "diagram_sampling_cost.svg": fig_sampling_cost,
    "diagram_autoencoder_gap.svg": fig_autoencoder_gap,
    "diagram_vae_directions.svg": fig_vae_directions,
    "diagram_elbo_gap.svg": fig_elbo_gap,
    "diagram_sigma_tension.svg": fig_sigma_tension,
    "diagram_reparameterisation.svg": fig_reparameterisation,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print(f"wrote {OUT / name}")

    print()
    print("Autoregressive sampling cost (sequential decoder passes, 3 channels):")
    for side in (32, 64, 256, 1024):
        print(f"  {side:>4d}x{side:<4d} {sampling_passes(side):>12,d}")
    print(f"  ratio 1024 to 32                {sampling_passes(1024) / sampling_passes(32):>10,.0f}x")

    print()
    print("CIFAR-10 bits per dimension (PixelRNN paper, Table 5):")
    for name, value, _ in BITS_PER_DIM:
        print(f"  {name:28s} {value:.2f}")

    print()
    print("KL direction, best single Gaussian fit to the bimodal target:")
    fm, fs = forward_kl_fit()
    print(f"  forward, moment matching        mu = {fm:.4f}  sigma = {fs:.4f}")
    print(f"  reverse, grid search            mu = {REVERSE_FIT[0]:.4f}  sigma = {REVERSE_FIT[1]:.4f}")
    cm, cs = reverse_kl_fit_coarse()
    print(f"  reverse, coarse check           mu = {cm:.4f}  sigma = {cs:.4f}")

    print()
    print("ELBO decomposition, checked against log p(x) = ELBO + posterior KL:")
    lp = log_marginal()
    print(f"  log p(x) = {lp:+.6f} nats")
    for name, mq, sq in ELBO_CASES:
        e, g = elbo(mq, sq), kl_to_posterior(mq, sq)
        ok = "exact" if abs(e + g - lp) < 1e-12 else "MISMATCH"
        print(f"  {name:20s} ELBO = {e:+.6f}  gap = {g:.6f}  sum = {e + g:+.6f}  {ok}")

    print()
    print("The optimum the two terms fight to, versus the true posterior:")
    mp, vp = true_posterior()
    gm, gs = optimal_encoder()
    print(f"  closed form   mu = {mp:.4f}  sigma = {math.sqrt(vp):.4f}")
    print(f"  grid search   mu = {gm:.4f}  sigma = {gs:.4f}")

    print()
    print("Posterior collapse: the same optimum as the decoder noise grows.")
    for sx in (0.4, 0.8, 2.0, 5.0, 20.0):
        m, v = true_posterior(X_OBS, sx)
        print(f"  sigma_x = {sx:5.1f}   mu* = {m:.4f}   sigma* = {math.sqrt(v):.4f}")
