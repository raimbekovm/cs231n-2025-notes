"""Generate the ten SVG diagrams for the lecture 14 chapter.

Eight of the ten are computed, and four of those carry assertions that would
fail loudly if the chapter's algebra were wrong.

`diagram_saturating_loss` differentiates both generator objectives with respect
to the discriminator's pre-sigmoid score, which is the variable the generator
actually influences. The ratio between the two gradients at D = 0.01 is the
number the prose quotes.

`diagram_optimal_discriminator` evaluates the GAN value function at the optimal
discriminator by quadrature and, separately, computes 2*JSD - log 4 the long
way round. The two must agree, and the sweep over the generator's mean must
bottom out at -log 4 when the distributions coincide.

`diagram_euler_error`, `diagram_generalized_diffusion` and the score check in
`__main__` share a one-dimensional Gaussian flow problem:

    p_noise = N(0, 1),  p_data = N(MU, SIGMA^2),  x_t = (1-t) x + t z

in which the optimal velocity field is affine in x_t, so the probability flow
ODE maps Gaussians to Gaussians and its solution can be propagated exactly.
That gives two independent routes to the marginal of x_t — integrate the ODE,
or write the law of the interpolant directly — and `__main__` asserts they
agree. The same model checks the score/velocity identity of the chapter, and
the Euler figure measures the discretisation error of the sampler against the
exact answer.

`diagram_regression_to_mean` uses a two-mode dataset in which the posterior over
components is closed form. Its right panel is *not* the irreducible loss, which
does not vanish at the endpoints: at t = 0 the target still contains a whole
unit of unobserved noise. What vanishes at the endpoints is the gap between the
optimal predictor and the best affine predictor of the same target, which is
exactly the lecture's claim that both ends are easy, made precise. The gap is
zero at t = 0 and t = 1 because the conditional mean is affine there, and the
generator prints those endpoint values as a check.

`diagram_guidance_tradeoff` integrates the guided ODE with both fields in closed
form, so the w = 0 case must reproduce the true conditional distribution
exactly; the generator asserts that before reporting anything about w > 0.

The two structural figures are the adversarial game and the latent diffusion
pipeline. `diagram_sequence_cost` is arithmetic on the downsampling and patchify
factors the respective models report.

Conventions follow figures/13-generative-models: currentColor for text and
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

OUT = pathlib.Path(__file__).resolve().parent


def normal_pdf(x, mu, sigma):
    """Density of N(mu, sigma^2) at x."""
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def quad(f, lo, hi, n=20000):
    """Composite Simpson quadrature of f over [lo, hi]."""
    if n % 2:
        n += 1
    h = (hi - lo) / n
    total = f(lo) + f(hi)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * f(lo + i * h)
    return total * h / 3.0


# --------------------------------------------------------------------------
# The GAN toy: a bimodal data density against a single-Gaussian generator.
# --------------------------------------------------------------------------
GAN_DATA = ((0.5, -1.2, 0.5), (0.5, 1.4, 0.7))
GAN_GEN = (0.3, 1.1)


def p_data(x, comps=GAN_DATA):
    """Bimodal data density."""
    return sum(w * normal_pdf(x, m, s) for w, m, s in comps)


def p_gen(x, mu=GAN_GEN[0], sigma=GAN_GEN[1]):
    """Generator density, a single Gaussian."""
    return normal_pdf(x, mu, sigma)


def optimal_d(x, mu=GAN_GEN[0], sigma=GAN_GEN[1]):
    """The optimal discriminator of the chapter's equation for these densities."""
    a, b = p_data(x), p_gen(x, mu, sigma)
    return a / (a + b) if a + b > 0 else 0.5


def value_at_optimum(mu=GAN_GEN[0], sigma=GAN_GEN[1], lo=-8.0, hi=8.0):
    """V(G, D*), evaluated directly as the sum of the two expectations."""

    def integrand(x):
        d = optimal_d(x, mu, sigma)
        d = min(max(d, 1e-300), 1.0 - 1e-16)
        return p_data(x) * math.log(d) + p_gen(x, mu, sigma) * math.log(1.0 - d)

    return quad(integrand, lo, hi)


def jensen_shannon(mu=GAN_GEN[0], sigma=GAN_GEN[1], lo=-8.0, hi=8.0):
    """Jensen-Shannon divergence between the data and generator densities."""

    def integrand(x):
        a, b = p_data(x), p_gen(x, mu, sigma)
        m = 0.5 * (a + b)
        out = 0.0
        if a > 1e-300:
            out += 0.5 * a * math.log(a / m)
        if b > 1e-300:
            out += 0.5 * b * math.log(b / m)
        return out

    return quad(integrand, lo, hi)


# --------------------------------------------------------------------------
# The Gaussian flow toy: p_noise = N(0,1), p_data = N(MU, SIGMA^2).
# --------------------------------------------------------------------------
MU = 1.1
SIGMA = 0.55


def interpolant_var(t, sigma=SIGMA):
    """Variance of x_t = (1-t) x + t z under the chapter's interpolant."""
    return (1.0 - t) ** 2 * sigma**2 + t**2


def field_slope(t, sigma=SIGMA):
    """The coefficient of x_t in the optimal velocity field."""
    return (t - (1.0 - t) * sigma**2) / interpolant_var(t, sigma)


def flow_field(xt, t, mu=MU, sigma=SIGMA):
    """The optimal velocity E[z - x | x_t] for the Gaussian problem."""
    return -mu + field_slope(t, sigma) * (xt - (1.0 - t) * mu)


def marginal_moments(t, mu=MU, sigma=SIGMA):
    """Closed-form mean and variance of x_t, route one."""
    return (1.0 - t) * mu, interpolant_var(t, sigma)


def integrate_moments(steps=4000, mu=MU, sigma=SIGMA):
    """Propagate the ODE's mean and variance from t=1 to t=0, route two.

    Runge-Kutta of order four, so that any disagreement with the closed form is a disagreement about the mathematics
    rather than about the step size.
    """

    def deriv(t, m, v):
        a = field_slope(t, sigma)
        b = -mu - a * (1.0 - t) * mu
        return a * m + b, 2.0 * a * v

    m, v = 0.0, 1.0
    dt = -1.0 / steps
    for k in range(steps):
        t = 1.0 + k * dt
        k1m, k1v = deriv(t, m, v)
        k2m, k2v = deriv(t + dt / 2, m + dt * k1m / 2, v + dt * k1v / 2)
        k3m, k3v = deriv(t + dt / 2, m + dt * k2m / 2, v + dt * k2v / 2)
        k4m, k4v = deriv(t + dt, m + dt * k3m, v + dt * k3v)
        m += dt * (k1m + 2 * k2m + 2 * k3m + k4m) / 6.0
        v += dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
    return m, v


def euler_sample_moments(T, mu=MU, sigma=SIGMA):
    """Mean and variance produced by the T-step Euler sampler, exactly."""
    m, v = 0.0, 1.0
    for k in range(T):
        t = 1.0 - k / T
        a = field_slope(t, sigma)
        b = -mu - a * (1.0 - t) * mu
        g = 1.0 - a / T
        m = g * m - b / T
        v = g * g * v
    return m, v


def euler_sigma_error(T):
    """Absolute error in the sampler's terminal standard deviation."""
    _, v = euler_sample_moments(T)
    return abs(math.sqrt(max(v, 0.0)) - SIGMA)


def score_two_ways(xt, t, mu=MU, sigma=SIGMA):
    """Score of the marginal, computed from the density and from the field."""
    m, v = marginal_moments(t, mu, sigma)
    direct = -(xt - m) / v
    via_field = -(xt + (1.0 - t) * flow_field(xt, t, mu, sigma)) / t
    return direct, via_field


# --------------------------------------------------------------------------
# The two-mode toy behind the optimal-prediction figure.
# --------------------------------------------------------------------------
MODES = (-1.5, 1.5)
MODE_S = 0.35


def mode_posterior(xt, t):
    """Posterior over the two mixture components given the noisy sample."""
    v = (1.0 - t) ** 2 * MODE_S**2 + t**2
    ws = [normal_pdf(xt, (1.0 - t) * m, math.sqrt(v)) for m in MODES]
    total = sum(ws)
    return [w / total for w in ws] if total > 0 else [0.5, 0.5]


def clean_expectation(xt, t):
    """E[x | x_t] for the two-mode dataset."""
    v = (1.0 - t) ** 2 * MODE_S**2 + t**2
    ws = mode_posterior(xt, t)
    return sum(w * (m + (1.0 - t) * MODE_S**2 * (xt - (1.0 - t) * m) / v) for w, m in zip(ws, MODES))


def velocity_expectation(xt, t):
    """E[z - x | x_t] for the two-mode dataset, with no division by t."""
    v = (1.0 - t) ** 2 * MODE_S**2 + t**2
    k = (t - (1.0 - t) * MODE_S**2) / v
    ws = mode_posterior(xt, t)
    return sum(w * (k * (xt - (1.0 - t) * m) - m) for w, m in zip(ws, MODES))


def marginal_xt(xt, t):
    """Density of x_t under the two-mode dataset."""
    v = (1.0 - t) ** 2 * MODE_S**2 + t**2
    return sum(0.5 * normal_pdf(xt, (1.0 - t) * m, math.sqrt(v)) for m in MODES)


def affine_gap(t, lo=-6.0, hi=6.0):
    """How much the optimal predictor beats the best affine one, at level t.

    Both predictors are of the same target v = z - x from the same input x_t, so the difference of their mean squared
    errors is Var(E[v|x_t]) minus the variance the best straight line can explain. It is zero exactly when the
    conditional mean is itself affine, which is what happens at t = 0 and t = 1.
    """
    data_var = MODE_S**2 + (MODES[0] ** 2 + MODES[1] ** 2) / 2.0
    var_xt = (1.0 - t) ** 2 * data_var + t**2
    cov = t - (1.0 - t) * data_var
    linear_explained = cov * cov / var_xt

    mean_u = quad(lambda x: marginal_xt(x, t) * velocity_expectation(x, t), lo, hi, 4000)
    second = quad(lambda x: marginal_xt(x, t) * velocity_expectation(x, t) ** 2, lo, hi, 4000)
    return (second - mean_u**2) - linear_explained


def logit_normal_pdf(t, m=0.0, s=1.0):
    """Density of the logit-normal sampling schedule at t in (0, 1)."""
    if t <= 0.0 or t >= 1.0:
        return 0.0
    lg = math.log(t / (1.0 - t))
    return normal_pdf(lg, m, s) / (t * (1.0 - t))


# --------------------------------------------------------------------------
# The guidance toy: two Gaussian classes, both fields in closed form.
# --------------------------------------------------------------------------
CLASS_MU = (-1.0, 1.0)
CLASS_S = 0.6


def conditional_field(xt, t, mu=CLASS_MU[0], sigma=CLASS_S):
    """Velocity field pointing at one class, which is a single Gaussian."""
    return flow_field(xt, t, mu, sigma)


def unconditional_field(xt, t):
    """Velocity field for the two-class mixture, which is the marginal."""
    v = (1.0 - t) ** 2 * CLASS_S**2 + t**2
    k = (t - (1.0 - t) * CLASS_S**2) / v
    ws = [normal_pdf(xt, (1.0 - t) * m, math.sqrt(v)) for m in CLASS_MU]
    total = sum(ws)
    ws = [w / total for w in ws] if total > 0 else [0.5, 0.5]
    return sum(w * (k * (xt - (1.0 - t) * m) - m) for w, m in zip(ws, CLASS_MU))


def guided_sample(z, w, steps=600):
    """Integrate the guided ODE from a noise value down to t = 0."""
    x = z
    dt = 1.0 / steps
    for k in range(steps):
        t = 1.0 - (k + 0.5) * dt
        vy = conditional_field(x, t)
        vn = unconditional_field(x, t)
        x -= dt * ((1.0 + w) * vy - w * vn)
    return x


GUIDE_NODES = [(-3.6 + 7.2 * i / 240.0) for i in range(241)]


def guided_stats(w):
    """Standard deviation and wrong-class mass of the guided sample law."""
    weights = [normal_pdf(z, 0.0, 1.0) for z in GUIDE_NODES]
    total = sum(weights)
    xs = [guided_sample(z, w) for z in GUIDE_NODES]
    mean = sum(p * x for p, x in zip(weights, xs)) / total
    var = sum(p * (x - mean) ** 2 for p, x in zip(weights, xs)) / total
    wrong = sum(p for p, x in zip(weights, xs) if x > 0.0) / total
    return math.sqrt(var), wrong


GUIDE_WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
GUIDE_STATS = [(w, *guided_stats(w)) for w in GUIDE_WEIGHTS]


# --------------------------------------------------------------------------
# Sequence lengths, from the downsampling and patchify factors.
# --------------------------------------------------------------------------
def token_count(dims, downsample, patch):
    """Tokens a transformer sees, given output dims and the two reductions."""
    n = 1
    for d, f, p in zip(dims, downsample, patch):
        n *= max(1, round(d / f / p))
    return n


SEQ_CASES = [
    ("pixel space, 1024x1024", token_count((1024, 1024), (1, 1), (2, 2))),
    ("latent image, FLUX.1", token_count((1024, 1024), (8, 8), (2, 2))),
    ("latent video, Movie Gen", token_count((264, 1024, 576), (8, 8, 8), (1, 2, 2))),
]


# --------------------------------------------------------------------------
# Figures.
# --------------------------------------------------------------------------
def fig_gan_game():
    """The adversarial game, and which network each gradient reaches."""
    w, h = 700, 300
    b = ""
    b += text(350, 22, "The two streams a discriminator sees", size=13.5, weight="600")

    b += text(50, 118, "z ~ p(z)", size=12, font=MONO)
    b += rect(95, 96, 74, 42, fill=INDIGO, op=0.16, stroke=INDIGO, sw=1.3, rx=4)
    b += text(132, 122, "G", size=15, fill=INDIGO, weight="600")
    b += arrow(169, 117, 209, 117, sw=1.3, op=0.85)
    b += rect(210, 96, 96, 42, fill=TEAL, op=0.1, stroke=TEAL, sw=1.2, rx=4)
    b += text(258, 114, "G(z)", size=12, fill=TEAL, font=MONO)
    b += text(258, 130, "fake", size=10.5, op=0.7)

    b += rect(210, 190, 96, 42, fill=PURPLE, op=0.1, stroke=PURPLE, sw=1.2, rx=4)
    b += text(258, 208, "x ~ p_data", size=11, fill=PURPLE, font=MONO)
    b += text(258, 224, "real", size=10.5, op=0.7)

    b += arrow(306, 117, 386, 140, sw=1.3, op=0.85)
    b += arrow(306, 211, 386, 188, sw=1.3, op=0.85)
    b += rect(388, 130, 66, 68, fill="none", stroke=INK, sw=1.5, rx=4)
    b += text(421, 160, "D", size=15, weight="600")
    b += text(421, 176, "0 to 1", size=9.5, op=0.65)

    b += arrow(454, 164, 496, 164, sw=1.3, op=0.85)
    b += text(600, 152, "E[ log D(x) ]", size=11.5, font=MONO, fill=PURPLE)
    b += text(600, 172, "E[ log(1 - D(G(z))) ]", size=11.5, font=MONO, fill=TEAL)
    b += text(600, 194, "D maximises both", size=10.5, op=0.7)

    b += polyline([(421, 130), (421, 62), (132, 62), (132, 96)], sw=1.3, op=0.7, dash="5 4")
    b += poly([(132, 96), (128.4, 88), (135.6, 88)], fill="currentColor", stroke="none", op=0.7)
    b += text(276, 55, "gradient to G, through the generated image", size=11, op=0.75)

    b += text(
        350,
        272,
        "The generator is trained by this path and by nothing else.",
        size=11,
        op=0.75,
    )
    return wrap(
        w,
        h,
        "ganGame",
        "The generative adversarial game",
        "A generator turns a prior sample into a fake image; a discriminator "
        "scores both that image and a real one; the generator's only gradient "
        "runs backwards from the discriminator through the generated image.",
        b,
    )



def fig_saturating_loss():
    """Both generator objectives and the gradient each one delivers."""
    w, h = 700, 356
    b = ""
    b += text(350, 22, "Two ways to say the same wish", size=13.5, weight="600")

    dpts = [0.004 + i * (0.99 - 0.004) / 400 for i in range(401)]

    px = (66, 320)
    py = (56, 250)
    m1 = mapper((0.0, 1.0), (-3.2, 5.0), px, py)
    b += line(px[0], py[1], px[1], py[1], op=0.5)
    b += line(px[0], py[0], px[0], py[1], op=0.5)
    _, yz = m1(0, 0.0)
    b += line(px[0], yz, px[1], yz, op=0.3)
    for val in (-3, -2, -1, 0, 1, 2, 3, 4, 5):
        _, y = m1(0, val)
        b += line(px[0] - 4, y, px[0], y, op=0.4)
        b += text(px[0] - 8, y + 3.5, f"{val}", size=9.5, anchor="end", op=0.7)
    for val in (0.0, 0.5, 1.0):
        x, _ = m1(val, 0)
        b += line(x, py[1], x, py[1] + 4, op=0.4)
        b += text(x, py[1] + 16, f"{val:g}", size=9.5, op=0.7)
    b += text(193, 288, "D(G(z))", size=11, op=0.8)
    b += text(193, 42, "generator loss, as plotted", size=11, op=0.8)

    sat = [m1(d, math.log(1.0 - d)) for d in dpts]
    non = [m1(d, -math.log(d)) for d in dpts]
    b += polyline([q for q in sat if py[0] <= q[1] <= py[1]], stroke=INDIGO, sw=2.0)
    b += polyline([q for q in non if py[0] <= q[1] <= py[1]], stroke=PURPLE, sw=2.0)
    b += text(108, 84, "-log D", size=10.5, fill=PURPLE, anchor="start")
    b += text(214, 214, "log(1 - D)", size=10.5, fill=INDIGO, anchor="start")

    px2 = (412, 668)
    m2 = mapper((0.0, 1.0), (0.0, 1.05), px2, py)
    b += line(px2[0], py[1], px2[1], py[1], op=0.5)
    b += line(px2[0], py[0], px2[0], py[1], op=0.5)
    for val in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, y = m2(0, val)
        b += line(px2[0] - 4, y, px2[0], y, op=0.4)
        b += text(px2[0] - 8, y + 3.5, f"{val:g}", size=9.5, anchor="end", op=0.7)
    for val in (0.0, 0.5, 1.0):
        x, _ = m2(val, 0)
        b += line(x, py[1], x, py[1] + 4, op=0.4)
        b += text(x, py[1] + 16, f"{val:g}", size=9.5, op=0.7)
    b += text(540, 288, "D(G(z))", size=11, op=0.8)
    b += text(540, 42, "| gradient reaching G |", size=11, op=0.8)

    b += polyline([m2(d, d) for d in dpts], stroke=INDIGO, sw=2.0)
    b += polyline([m2(d, 1.0 - d) for d in dpts], stroke=PURPLE, sw=2.0)
    xs, _ = m2(0.01, 0)
    b += line(xs, py[0], xs, py[1], op=0.35, dash="4 4")
    b += arrow(*m2(0.30, 0.88), *m2(0.05, 0.97), sw=1.2, op=0.7)
    b += text(*[m2(0.34, 0.92)[0], m2(0.34, 0.92)[1]], "training starts here:", size=10, anchor="start")
    b += text(*[m2(0.34, 0.84)[0], m2(0.34, 0.84)[1]], "a 99-fold difference", size=10, anchor="start")

    b += line(150, 302, 170, 302, stroke=INDIGO, sw=2.0)
    b += text(176, 306, "log(1 - D), the objective as written", size=10, fill=INDIGO, anchor="start")
    b += line(150, 322, 170, 322, stroke=PURPLE, sw=2.0)
    b += text(176, 326, "-log D, the non-saturating replacement", size=10, fill=PURPLE, anchor="start")

    b += text(
        350,
        346,
        "Left: the objectives at their true values. Right: their derivative with respect to the discriminator's score.",
        size=11,
        op=0.75,
    )
    return wrap(
        w,
        h,
        "satLoss",
        "Saturating and non-saturating generator losses",
        "Two panels comparing the two generator objectives and the gradient magnitude each delivers as a function of "
        "the discriminator's output.",
        b,
    )

def fig_optimal_discriminator():
    """The optimal discriminator, and the divergence it turns the game into."""
    w, h = 700, 320
    b = ""
    b += text(350, 22, "The discriminator reports a density ratio", size=13.5, weight="600")

    xs = [-4.0 + i * 8.0 / 400 for i in range(401)]

    px = (62, 320)
    py = (52, 244)
    m1 = mapper((-4.0, 4.0), (0.0, 0.45), px, py)
    b += line(px[0], py[1], px[1], py[1], op=0.5)
    b += line(px[0], py[0], px[0], py[1], op=0.5)
    for val in (-4, -2, 0, 2, 4):
        x, _ = m1(val, 0)
        b += line(x, py[1], x, py[1] + 4, op=0.4)
        b += text(x, py[1] + 16, f"{val}", size=9.5, op=0.7)
    b += text(191, 282, "x", size=11, op=0.8, style="italic")
    b += text(191, 40, "density", size=11, op=0.8)

    fill_data = [m1(x, p_data(x)) for x in xs]
    b += poly(
        [m1(-4.0, 0.0), *fill_data, m1(4.0, 0.0)],
        fill=YELLOW,
        stroke="none",
        op=0.32,
    )
    b += polyline(fill_data, stroke=PURPLE, sw=2.0)
    b += polyline([m1(x, p_gen(x)) for x in xs], stroke=TEAL, sw=2.0, dash="6 4")
    b += text(110, 74, "p_data", size=10.5, fill=PURPLE, font=MONO, anchor="start")
    b += text(272, 96, "p_G", size=10.5, fill=TEAL, font=MONO, anchor="end")

    px2 = (410, 668)
    m2 = mapper((-4.0, 4.0), (0.0, 1.0), px2, py)
    b += line(px2[0], py[1], px2[1], py[1], op=0.5)
    b += line(px2[0], py[0], px2[0], py[1], op=0.5)
    for val in (-4, -2, 0, 2, 4):
        x, _ = m2(val, 0)
        b += line(x, py[1], x, py[1] + 4, op=0.4)
        b += text(x, py[1] + 16, f"{val}", size=9.5, op=0.7)
    for val in (0.0, 0.5, 1.0):
        _, y = m2(0, val)
        b += line(px2[0] - 4, y, px2[0], y, op=0.4)
        b += text(px2[0] - 8, y + 3.5, f"{val:g}", size=9.5, anchor="end", op=0.7)
    b += text(539, 282, "x", size=11, op=0.8, style="italic")
    b += text(539, 40, "D*(x)", size=11, op=0.8)

    _, yhalf = m2(0, 0.5)
    b += line(px2[0], yhalf, px2[1], yhalf, op=0.35, dash="4 4")
    b += polyline([m2(x, optimal_d(x)) for x in xs], stroke=INDIGO, sw=2.2)
    b += text(px2[0] + 8, yhalf + 15, "one half", size=10, anchor="start", op=0.7)

    v = value_at_optimum()
    js = jensen_shannon()
    b += text(
        350,
        304,
        f"V(G, D*) = {v:.5f} by quadrature; 2 JSD - log 4 = {2 * js - math.log(4.0):.5f}",
        size=11,
        op=0.8,
        font=MONO,
    )
    return wrap(
        w,
        h,
        "optDisc",
        "The optimal discriminator and the Jensen-Shannon identity",
        "A bimodal data density against a Gaussian generator density, and the "
        "optimal discriminator they induce, which crosses one half where the "
        "densities cross.",
        b,
    )



def fig_interpolation_paths():
    """Two pairs whose interpolants coincide at the same time with different velocities."""
    w, h = 700, 340
    b = ""
    b += text(350, 22, "Two pairs, one point, two answers", size=13.5, weight="600")

    cross = (350, 168)
    # Chosen so that both pairs satisfy (x + z) / 2 = cross, hence both
    # interpolants sit exactly on `cross` at t = 0.5.
    pairs = [((216, 80), (484, 256)), ((172, 132), (528, 204))]
    others = [((150, 92), (470, 250)), ((236, 126), (546, 198))]

    b += circle(190, 112, 68, fill=INDIGO, op=0.09, stroke=INDIGO, sw=1.1, so=0.5)
    b += text(72, 116, "p_noise", size=11.5, fill=INDIGO, font=MONO)
    b += circle(510, 228, 66, fill=PURPLE, op=0.09, stroke=PURPLE, sw=1.1, so=0.5)
    b += text(628, 232, "p_data", size=11.5, fill=PURPLE, font=MONO)

    for (zx, zy), (dx, dy) in others:
        b += line(dx, dy, zx, zy, stroke=INK, sw=1.2, op=0.28)
        b += circle(zx, zy, 3.4, fill=INDIGO, stroke="none", op=0.45)
        b += circle(dx, dy, 3.4, fill=PURPLE, stroke="none", op=0.45)

    for (zx, zy), (dx, dy) in pairs:
        b += line(dx, dy, zx, zy, stroke=TEAL, sw=1.6, op=0.6)
        b += circle(zx, zy, 4.2, fill=INDIGO, stroke="none")
        b += circle(dx, dy, 4.2, fill=PURPLE, stroke="none")

    for i, ((zx, zy), (dx, dy)) in enumerate(pairs):
        ux, uy = zx - dx, zy - dy
        n = math.hypot(ux, uy)
        b += arrow(
            cross[0], cross[1], cross[0] + 62 * ux / n, cross[1] + 62 * uy / n,
            sw=1.8, op=0.95, stroke=TEAL,
        )
        lx, ly = cross[0] + 74 * ux / n, cross[1] + 74 * uy / n
        b += text(lx - 6, ly - 4, f"v{i + 1}", size=11.5, font=MONO, fill=TEAL, anchor="end")

    b += circle(cross[0], cross[1], 6.0, fill=TEAL, stroke="none")
    b += arrow(300, 206, cross[0] - 10, cross[1] + 10, sw=1.2, op=0.7, stroke=TEAL)
    b += text(296, 212, "x_t at t = 0.5,", size=11, font=MONO, fill=TEAL, anchor="end")
    b += text(296, 228, "from either pair", size=11, font=MONO, fill=TEAL, anchor="end")

    b += text(
        350,
        306,
        "Both bold segments pass through the marked point at the same noise level, carrying different velocities.",
        size=11,
        op=0.75,
    )
    b += text(350, 322, "The network sees one input and two targets, and squared error makes it answer with their average.", size=11, op=0.75)
    return wrap(
        w,
        h,
        "interpPaths",
        "Two interpolation paths meeting at one point",
        "Two noise-data pairs whose straight-line interpolants coincide at the halfway time, with the two different "
        "velocity vectors drawn from the shared point.",
        b,
    )

def fig_euler_error():
    """Discretisation error of the Euler sampler against the exact answer."""
    w, h = 700, 320
    b = ""
    b += text(350, 22, "Forward Euler is first order", size=13.5, weight="600")

    steps = [2, 3, 5, 8, 12, 20, 30, 50, 80, 130, 200, 320, 500, 800, 1300, 2000]
    errs = [euler_sigma_error(T) for T in steps]
    lx = (math.log10(2.0), math.log10(2000.0))
    ly = (math.log10(min(errs)) - 0.15, math.log10(max(errs)) + 0.15)

    px = (86, 620)
    py = (54, 236)
    m = mapper(lx, ly, px, py)
    b += line(px[0], py[1], px[1], py[1], op=0.5)
    b += line(px[0], py[0], px[0], py[1], op=0.5)
    for val in (2, 10, 50, 200, 1000):
        x, _ = m(math.log10(val), ly[0])
        b += line(x, py[1], x, py[1] + 4, op=0.4)
        b += text(x, py[1] + 16, f"{val}", size=9.5, op=0.7)
    for e in (-4, -3, -2, -1):
        if ly[0] <= e <= ly[1]:
            _, y = m(lx[0], e)
            b += line(px[0] - 4, y, px[0], y, op=0.4)
            b += text(px[0] - 8, y + 3.5, f"1e{e}", size=9.5, anchor="end", op=0.7)

    b += text(353, 274, "sampling steps T", size=11, op=0.8)
    b += text(353, 42, "error in the terminal standard deviation", size=11, op=0.8)

    b += polyline(
        [m(math.log10(T), math.log10(e)) for T, e in zip(steps, errs)],
        stroke=INDIGO,
        sw=2.2,
    )
    for T, e in zip(steps, errs):
        b += circle(*m(math.log10(T), math.log10(e)), 2.6, fill=INDIGO, stroke="none")

    for T, label in ((30, "T = 30"), (50, "T = 50")):
        e = euler_sigma_error(T)
        x, y = m(math.log10(T), math.log10(e))
        b += line(x, y, x, py[1], op=0.35, dash="4 4")
        b += text(x, py[1] - 8, label, size=10, op=0.85)

    b += text(
        350,
        300,
        "Slope -1 on log axes: halving the step size halves the error, and no step count removes it.",
        size=11,
        op=0.75,
    )
    return wrap(
        w,
        h,
        "eulerErr",
        "Euler discretisation error against step count",
        "A log-log plot of the error in the sampler's terminal standard "
        "deviation against the number of steps, falling with slope minus one.",
        b,
    )


def fig_regression_to_mean():
    """What the conditional mean looks like, and where it stops being affine."""
    w, h = 700, 340
    b = ""
    b += text(350, 22, "The optimal prediction averages over what it cannot see", size=13.5, weight="600")

    px = (62, 318)
    py = (56, 244)
    m1 = mapper((-3.0, 3.0), (-2.0, 2.0), px, py)
    b += line(px[0], py[1], px[1], py[1], op=0.5)
    b += line(px[0], py[0], px[0], py[1], op=0.5)
    _, yz = m1(0, 0.0)
    b += line(px[0], yz, px[1], yz, op=0.3)
    for val in (-2, 0, 2):
        x, _ = m1(val, 0)
        b += line(x, py[1], x, py[1] + 4, op=0.4)
        b += text(x, py[1] + 16, f"{val}", size=9.5, op=0.7)
    for val in (-1.5, 0.0, 1.5):
        _, y = m1(0, val)
        b += line(px[0] - 4, y, px[0], y, op=0.4)
        b += text(px[0] - 8, y + 3.5, f"{val:g}", size=9.5, anchor="end", op=0.7)
    b += text(190, 282, "noisy value x_t", size=11, op=0.8, font=MONO)
    b += text(190, 42, "E[ x | x_t ]", size=11, op=0.8, font=MONO)

    xs = [-3.0 + i * 6.0 / 300 for i in range(301)]
    for i, (t, colour, label) in enumerate(
        ((0.25, PURPLE, "t = 0.25"), (0.55, TEAL, "t = 0.55"), (0.85, INDIGO, "t = 0.85"))
    ):
        pts = [m1(x, max(-2.0, min(2.0, clean_expectation(x, t)))) for x in xs]
        b += polyline(pts, stroke=colour, sw=2.0)
        lx, ly = m1(-2.85, 1.82 - 0.34 * i)
        b += line(lx, ly - 4, lx + 16, ly - 4, stroke=colour, sw=2.0)
        b += text(lx + 22, ly, label, size=10, fill=colour, anchor="start")

    px2 = (408, 666)
    m2 = mapper((0.0, 1.0), (0.0, 1.08), px2, py)
    b += line(px2[0], py[1], px2[1], py[1], op=0.5)
    b += line(px2[0], py[0], px2[0], py[1], op=0.5)
    for val in (0.0, 0.25, 0.5, 0.75, 1.0):
        x, _ = m2(val, 0)
        b += line(x, py[1], x, py[1] + 4, op=0.4)
        b += text(x, py[1] + 16, f"{val:g}", size=9.5, op=0.7)
    b += text(537, 282, "noise level t", size=11, op=0.8)
    b += text(537, 42, "how much the network has to know", size=11, op=0.8)

    ts = [i / 120.0 for i in range(121)]
    gaps = [max(0.0, affine_gap(t)) for t in ts]
    gmax = max(gaps)
    schedule = [logit_normal_pdf(t) for t in ts]
    smax = max(schedule)
    b += poly(
        [m2(0.0, 0.0)] + [m2(t, s / smax) for t, s in zip(ts, schedule)] + [m2(1.0, 0.0)],
        fill=YELLOW,
        stroke="none",
        op=0.35,
    )
    b += polyline([m2(t, g / gmax) for t, g in zip(ts, gaps)], stroke=PURPLE, sw=2.2)
    b += text(px2[0] + 10, py[0] + 18, "gap over the best", size=10, fill=PURPLE, anchor="start")
    b += text(px2[0] + 10, py[0] + 32, "affine predictor", size=10, fill=PURPLE, anchor="start")
    b += text(px2[1] - 8, py[1] - 74, "logit-normal", size=10, anchor="end", op=0.8)
    b += text(px2[1] - 8, py[1] - 60, "schedule", size=10, anchor="end", op=0.8)

    b += text(
        350,
        312,
        "Both curves in the right panel are scaled to their own peak; only their shapes are being compared.",
        size=11,
        op=0.75,
    )
    return wrap(
        w,
        h,
        "regMean",
        "Conditional expectation and where the problem is hard",
        "Left, the implied clean prediction against the noisy value at three "
        "noise levels. Right, the advantage of the optimal predictor over the "
        "best affine one, peaking at intermediate noise.",
        b,
    )


def fig_guidance_tradeoff():
    """What guidance buys and what it spends."""
    w, h = 700, 320
    b = ""
    b += text(350, 22, "Guidance makes the model narrow, not correct", size=13.5, weight="600")

    px = (78, 328)
    py = (58, 232)
    m = mapper((0.0, 6.0), (0.25, 0.68), px, py)
    b += line(px[0], py[1], px[1], py[1], op=0.5)
    b += line(px[0], py[0], px[0], py[1], op=0.5)
    for val in (0, 2, 4, 6):
        x, _ = m(val, 0.25)
        b += line(x, py[1], x, py[1] + 4, op=0.4)
        b += text(x, py[1] + 16, f"{val}", size=9.5, op=0.7)
    for val in (0.3, 0.4, 0.5, 0.6):
        _, y = m(0, val)
        b += line(px[0] - 4, y, px[0], y, op=0.4)
        b += text(px[0] - 8, y + 3.5, f"{val:g}", size=9.5, anchor="end", op=0.7)
    b += text(203, 268, "guidance weight w", size=11, op=0.8)
    b += text(203, 44, "spread of the samples", size=11, op=0.8)

    _, ytrue = m(0, CLASS_S)
    b += line(px[0], ytrue, px[1], ytrue, op=0.4, dash="5 4")
    b += text(px[1] - 4, ytrue - 9, "true conditional sigma", size=10, anchor="end", op=0.8)

    sd_pts = [m(wt, sd) for wt, sd, _ in GUIDE_STATS]
    b += polyline(sd_pts, stroke=INDIGO, sw=2.2)
    for pt in sd_pts:
        b += circle(*pt, 2.8, fill=INDIGO, stroke="none")

    px2 = (420, 668)
    m2 = mapper((0.0, 6.0), (0.0, 0.058), px2, py)
    b += line(px2[0], py[1], px2[1], py[1], op=0.5)
    b += line(px2[0], py[0], px2[0], py[1], op=0.5)
    for val in (0, 2, 4, 6):
        x, _ = m2(val, 0.0)
        b += line(x, py[1], x, py[1] + 4, op=0.4)
        b += text(x, py[1] + 16, f"{val}", size=9.5, op=0.7)
    for val in (0.0, 0.02, 0.04):
        _, y = m2(0, val)
        b += line(px2[0] - 4, y, px2[0], y, op=0.4)
        b += text(px2[0] - 8, y + 3.5, f"{val:g}", size=9.5, anchor="end", op=0.7)
    b += text(544, 268, "guidance weight w", size=11, op=0.8)
    b += text(544, 44, "mass landing on the wrong class", size=11, op=0.8)

    wr_pts = [m2(wt, wr) for wt, _, wr in GUIDE_STATS]
    b += polyline(wr_pts, stroke=PURPLE, sw=2.2)
    for pt in wr_pts:
        b += circle(*pt, 2.8, fill=PURPLE, stroke="none")
    b += text(px2[0] + 14, wr_pts[0][1] - 12, "the true conditional's own tail", size=10, fill=PURPLE, anchor="start")

    b += text(
        350,
        300,
        "The panels have different vertical scales. At w = 0 the integrator reproduces the true conditional exactly.",
        size=11,
        op=0.75,
    )
    return wrap(
        w,
        h,
        "cfgTrade",
        "The classifier-free guidance trade-off",
        "Two panels, on separate vertical scales: the spread of the guided samples and the mass they place on the wrong class, both against the guidance weight.",
        b,
    )


def fig_latent_diffusion():
    """The two-stage pipeline, and where each network's gradients stop."""
    w, h = 700, 340
    b = ""
    b += text(350, 22, "Compress once, diffuse in the small space", size=13.5, weight="600")

    b += rect(38, 52, 178, 214, fill="none", stroke=INK, sw=1.0, so=0.35, rx=6)
    b += text(127, 72, "stage 1: autoencoder", size=11.5, weight="600", op=0.85)
    b += rect(66, 88, 122, 34, fill=PURPLE, op=0.12, stroke=PURPLE, sw=1.2, rx=4)
    b += text(127, 109, "image  1024x1024x3", size=10, font=MONO, fill=PURPLE)
    b += arrow(127, 122, 127, 146, sw=1.3, op=0.85)
    b += text(178, 138, "encoder", size=10.5, anchor="end", op=0.85)
    b += rect(66, 148, 122, 34, fill=TEAL, op=0.14, stroke=TEAL, sw=1.2, rx=4)
    b += text(127, 169, "latent  128x128x16", size=10, font=MONO, fill=TEAL)
    b += arrow(127, 182, 127, 206, sw=1.3, op=0.85)
    b += text(178, 198, "decoder", size=10.5, anchor="end", op=0.85)
    b += rect(66, 208, 122, 34, fill=PURPLE, op=0.12, stroke=PURPLE, sw=1.2, rx=4)
    b += text(127, 229, "reconstruction", size=10, font=MONO, fill=PURPLE)
    b += text(127, 258, "+ discriminator, to keep it sharp", size=10, op=0.75)

    b += arrow(220, 159, 262, 159, sw=1.4, op=0.9)
    b += text(241, 148, "freeze", size=10, op=0.8)

    b += rect(266, 52, 178, 214, fill="none", stroke=INK, sw=1.0, so=0.35, rx=6)
    b += text(355, 72, "stage 2: diffusion", size=11.5, weight="600", op=0.85)
    b += rect(292, 100, 126, 32, fill=INDIGO, op=0.14, stroke=INDIGO, sw=1.2, rx=4)
    b += text(355, 120, "noisy latent", size=10.5, font=MONO, fill=INDIGO)
    b += arrow(355, 132, 355, 158, sw=1.3, op=0.85)
    b += rect(292, 160, 126, 32, fill=INDIGO, op=0.2, stroke=INDIGO, sw=1.3, rx=4)
    b += text(355, 180, "DiT", size=12, fill=INDIGO, weight="600")
    b += text(355, 214, "trained on latents only;", size=10, op=0.8)
    b += text(355, 228, "no gradient reaches stage 1", size=10, op=0.8)

    b += rect(494, 52, 172, 214, fill="none", stroke=INK, sw=1.0, so=0.35, rx=6)
    b += text(580, 72, "sampling", size=11.5, weight="600", op=0.85)
    b += text(580, 100, "noise in latent space", size=10.5, op=0.85)
    b += arrow(580, 108, 580, 132, sw=1.3, op=0.85)
    b += text(580, 148, "50 guided Euler steps", size=10.5, op=0.85)
    b += text(580, 164, "= 100 DiT passes", size=10.5, op=0.7, font=MONO)
    b += arrow(580, 172, 580, 196, sw=1.3, op=0.85)
    b += text(580, 212, "decoder, run once", size=10.5, op=0.85)
    b += text(580, 236, "image out", size=10.5, fill=PURPLE, op=0.9)

    b += text(
        350,
        294,
        "4096 latent tokens against 262144 pixel-space tokens: a 64-fold reduction in sequence length,",
        size=11,
        op=0.75,
    )
    b += text(350, 310, "and roughly four thousand fold in attention cost.", size=11, op=0.75)
    return wrap(
        w,
        h,
        "latDiff",
        "The latent diffusion pipeline",
        "Three panels: an autoencoder trained first and frozen, a diffusion "
        "transformer trained on its latents, and the sampling path that ends "
        "with a single decoder call.",
        b,
    )


def fig_sequence_cost():
    """Sequence length and attention cost across three configurations."""
    w, h = 700, 300
    b = ""
    b += text(350, 22, "Where the cost of video generation sits", size=13.5, weight="600")

    lo, hi = 3.0, 5.8
    px = (268, 656)
    m = mapper((lo, hi), (0.0, 1.0), px, (60, 210))
    b += line(px[0], 224, px[1], 224, op=0.5)
    for val in (3, 4, 5):
        x, _ = m(val, 0)
        b += line(x, 224, x, 229, op=0.4)
        b += text(x, 242, f"1e{val}", size=9.5, op=0.7)
    b += text(462, 262, "tokens per forward pass, logarithmic", size=11, op=0.8)

    base = SEQ_CASES[1][1]
    colours = (INDIGO, TEAL, PURPLE)
    for i, ((label, n), colour) in enumerate(zip(SEQ_CASES, colours)):
        y = 76 + i * 48
        xr, _ = m(math.log10(n), 0)
        b += rect(
            px[0], y - 15, max(2.0, xr - px[0]), 30,
            fill=colour, op=0.35, stroke=colour, sw=1.2, rx=3,
        )
        b += text(256, y - 2, label, size=11, anchor="end")
        ratio = (n / base) ** 2
        b += text(
            256, y + 12, f"{n:,} tokens, {ratio:,.0f}x attention",
            size=9.5, anchor="end", op=0.75,
        )

    b += text(
        350,
        286,
        "Attention cost is relative to the text-to-image row and quadratic in the token count.",
        size=11,
        op=0.75,
    )
    return wrap(
        w,
        h,
        "seqCost",
        "Sequence length across three diffusion configurations",
        "Horizontal bars on a logarithmic axis comparing tokens per forward "
        "pass for pixel-space, latent image and latent video diffusion.",
        b,
    )


def fig_generalized_diffusion():
    """The two common interpolation schemes and the property one is named for."""
    w, h = 700, 358
    b = ""
    b += text(350, 22, "One template, two schedules", size=13.5, weight="600")

    ts = [i / 200.0 for i in range(201)]

    px = (66, 320)
    py = (56, 232)
    m1 = mapper((0.0, 1.0), (0.0, 1.06), px, py)
    b += line(px[0], py[1], px[1], py[1], op=0.5)
    b += line(px[0], py[0], px[0], py[1], op=0.5)
    for val in (0.0, 0.5, 1.0):
        x, _ = m1(val, 0)
        b += line(x, py[1], x, py[1] + 4, op=0.4)
        b += text(x, py[1] + 16, f"{val:g}", size=9.5, op=0.7)
        _, y = m1(0, val)
        b += line(px[0] - 4, y, px[0], y, op=0.4)
        b += text(px[0] - 8, y + 3.5, f"{val:g}", size=9.5, anchor="end", op=0.7)
    b += text(193, 270, "noise level t", size=11, op=0.8)
    b += text(193, 42, "coefficients: a(t) solid, b(t) dashed", size=11, op=0.8)

    b += polyline([m1(t, 1.0 - t) for t in ts], stroke=INDIGO, sw=2.0)
    b += polyline([m1(t, t) for t in ts], stroke=INDIGO, sw=2.0, dash="5 4")
    b += polyline([m1(t, math.cos(math.pi * t / 2)) for t in ts], stroke=PURPLE, sw=2.0)
    b += polyline(
        [m1(t, math.sin(math.pi * t / 2)) for t in ts], stroke=PURPLE, sw=2.0, dash="5 4"
    )

    b += line(70, 300, 90, 300, stroke=INDIGO, sw=2.0)
    b += text(96, 304, "rectified flow: a = 1 - t, b = t", size=10, fill=INDIGO, anchor="start")
    b += line(70, 322, 90, 322, stroke=PURPLE, sw=2.0)
    b += text(
        96, 326, "variance preserving: a = cos(pi t / 2), b = sin(pi t / 2)",
        size=10, fill=PURPLE, anchor="start",
    )

    px2 = (412, 666)
    m2 = mapper((0.0, 1.0), (0.6, 1.06), px2, py)
    b += line(px2[0], py[1], px2[1], py[1], op=0.5)
    b += line(px2[0], py[0], px2[0], py[1], op=0.5)
    for val in (0.0, 0.5, 1.0):
        x, _ = m2(val, 0.6)
        b += line(x, py[1], x, py[1] + 4, op=0.4)
        b += text(x, py[1] + 16, f"{val:g}", size=9.5, op=0.7)
    for val in (0.7, 0.8, 0.9, 1.0):
        _, y = m2(0, val)
        b += line(px2[0] - 4, y, px2[0], y, op=0.4)
        b += text(px2[0] - 8, y + 3.5, f"{val:g}", size=9.5, anchor="end", op=0.7)
    b += text(539, 270, "noise level t", size=11, op=0.8)
    b += text(539, 42, "standard deviation of x_t", size=11, op=0.8)

    b += polyline([m2(t, 1.0) for t in ts], stroke=PURPLE, sw=2.2)
    b += polyline(
        [m2(t, math.sqrt((1.0 - t) ** 2 + t**2)) for t in ts], stroke=INDIGO, sw=2.2
    )
    b += text(
        px2[0] + 14, m2(0, 1.0)[1] + 16, "held at one by construction",
        size=10, fill=PURPLE, anchor="start",
    )
    lowx, lowy = m2(0.5, math.sqrt(0.5))
    b += circle(lowx, lowy, 3.0, fill=INDIGO, stroke="none")
    b += text(lowx, lowy + 22, "0.707 at the midpoint", size=10, fill=INDIGO)

    b += text(
        350,
        348,
        "Both schemes are the same four functions with different values; only one keeps the input scale fixed.",
        size=11,
        op=0.75,
    )
    return wrap(
        w,
        h,
        "genDiff",
        "Rectified flow and variance-preserving schedules",
        "Left, the interpolation coefficients of the two schemes. Right, the resulting standard deviation of the "
        "noisy sample, flat for one scheme and dipping for the other.",
        b,
    )

FIGURES = {
    "diagram_gan_game.svg": fig_gan_game,
    "diagram_saturating_loss.svg": fig_saturating_loss,
    "diagram_optimal_discriminator.svg": fig_optimal_discriminator,
    "diagram_interpolation_paths.svg": fig_interpolation_paths,
    "diagram_euler_error.svg": fig_euler_error,
    "diagram_regression_to_mean.svg": fig_regression_to_mean,
    "diagram_guidance_tradeoff.svg": fig_guidance_tradeoff,
    "diagram_latent_diffusion.svg": fig_latent_diffusion,
    "diagram_sequence_cost.svg": fig_sequence_cost,
    "diagram_generalized_diffusion.svg": fig_generalized_diffusion,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print(f"wrote {OUT / name}")

    print()
    print("Generator gradient with respect to the discriminator's score:")
    for d in (0.01, 0.1, 0.5, 0.9):
        print(f"  D(G(z)) = {d:.2f}   saturating {d:.4f}   non-saturating {1 - d:.4f}   ratio {(1 - d) / d:.1f}x")

    print()
    print("The GAN value at the optimal discriminator, two ways:")
    v = value_at_optimum()
    js = jensen_shannon()
    print(f"  V(G, D*) by quadrature          {v:+.9f}")
    print(f"  2 JSD - log 4                   {2 * js - math.log(4.0):+.9f}")
    assert abs(v - (2 * js - math.log(4.0))) < 1e-7, "the JSD identity does not hold"
    print(f"  agree to                        {abs(v - (2 * js - math.log(4.0))):.2e}")
    print(f"  -log 4                          {-math.log(4.0):+.9f}")
    print("  sweeping the generator towards the data:")
    for mu, sig in ((0.3, 1.1), (0.1, 1.4), (0.1, 1.6)):
        print(f"    mu = {mu:.1f} sigma = {sig:.1f}   V = {value_at_optimum(mu, sig):+.6f}")
    same = value_at_optimum(0.0, 1.0, -9.0, 9.0)
    matched = quad(lambda x: 2.0 * normal_pdf(x, 0.0, 1.0) * math.log(0.5), -9.0, 9.0)
    print(f"  identical distributions          V = {matched:+.9f} (= -log 4)")
    assert abs(matched + math.log(4.0)) < 1e-9, "the optimum is not -log 4"
    print(f"  (Gaussian generator vs Gaussian data at the same params: {same:+.6f})")

    print()
    print("The Gaussian flow, marginal of x_t by two independent routes:")
    im, iv = integrate_moments()
    cm, cv = marginal_moments(0.0)
    print(f"  ODE integration    mean {im:.9f}  variance {iv:.9f}")
    print(f"  closed form        mean {cm:.9f}  variance {cv:.9f}")
    assert abs(im - cm) < 1e-6 and abs(iv - cv) < 1e-6, "the flow does not transport"
    print(f"  terminal sigma     {math.sqrt(iv):.9f} against {SIGMA}")

    print()
    print("Euler sampler error against step count:")
    for T in (1, 2, 5, 10, 30, 50, 200, 1000):
        print(f"  T = {T:>4d}   error {euler_sigma_error(T):.6e}")
    r = euler_sigma_error(50) / euler_sigma_error(100)
    print(f"  error ratio between T=50 and T=100: {r:.3f} (first order gives 2)")

    print()
    print("Score from the density against score from the velocity field:")
    worst = 0.0
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        for xt in (-1.0, 0.0, 0.8, 2.0):
            a, c = score_two_ways(xt, t)
            worst = max(worst, abs(a - c))
    print(f"  largest disagreement over the grid: {worst:.3e}")
    assert worst < 1e-12, "the score/velocity identity does not hold"

    print()
    print("Two-mode dataset: the optimal predictor's advantage over an affine one.")
    for t in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
        print(f"  t = {t:.2f}   gap {affine_gap(t):+.6f}")
    assert abs(affine_gap(0.0)) < 1e-6, "the gap should vanish at t = 0"
    assert abs(affine_gap(1.0)) < 1e-6, "the gap should vanish at t = 1"

    print()
    print("Guidance: sample spread and wrong-class mass.")
    print(f"  true conditional sigma {CLASS_S}")
    for wt, sd, wr in GUIDE_STATS:
        print(f"  w = {wt:.1f}   sigma {sd:.4f}   wrong-class mass {wr:.4f}")
    assert abs(GUIDE_STATS[0][1] - CLASS_S) < 2e-3, "w=0 must reproduce the conditional"

    print()
    print("Sequence lengths and attention cost:")
    base = SEQ_CASES[1][1]
    for label, n in SEQ_CASES:
        print(f"  {label:26s} {n:>9,d} tokens   {(n / base) ** 2:>10,.0f}x attention")
