"""Generate the nine SVG diagrams for the lecture 17 chapter.

Two of them carry computation rather than geometry, and both are built so that
the claim in the prose is *checked* rather than remembered.

`diagram_compounding` rests on the compounding-error argument behind DAgger. The
worst case is modelled explicitly: a policy that errs with probability epsilon
per step on the expert's state distribution, and that leaves that distribution
permanently once it errs, paying unit cost for the rest of the episode. The
expected cost is evaluated two independent ways — term-by-term summation over
the horizon, and the closed form obtained by summing the geometric series — and
`__main__` asserts they agree to machine precision at every (T, epsilon) checked.

The same block then checks the thing the section would otherwise assert from
memory. The quoted bound is O(epsilon T^2); the exact construction has leading
term epsilon T(T+1)/2, so the constant is 1/2 and there is a +1 that a
from-memory statement drops. `quadratic_ratio` measures the approach to that
limit rather than assuming it, and a second assertion pins the fact the
asymptotic statement hides entirely: the exact cost can never exceed the horizon,
so the quadratic growth is a small-(epsilon T) phenomenon and the true curve
saturates into total failure.

`diagram_multimodal` rests on the identity behind implicit and diffusion
policies. Two equally good demonstrated actions at +/- d/2, each Gaussian with
standard deviation sigma, give a mixture whose L2-optimal prediction is the
mixture mean. That is computed analytically and, independently, by numerically
minimising expected squared error over a grid; the two are asserted equal.

The second check in that figure is the one worth having. "The mean sits between
the modes, so regression fails" is the obvious claim, and it is only true above a
threshold: the mixture is bimodal if and only if d > 2 sigma. That closed-form
threshold is asserted against an empirical one found by counting local maxima of
the density on a fine grid while sweeping d/sigma, and the block then confirms
that above the threshold the L2 answer is a local *minimum* of the density
rather than merely a poor sample from it.

The rest are geometry: the open-loop function against the closed perception-
action loop, the four axes on which robot perception departs from recognition,
credit assignment along a trajectory, the experience-consumed bar chart (log
axis, every value taken from or derived from the cited paper), the three state
representations, particle message passing, and the pi-zero VLA stack.

Every figure that has a direction of flow assigns x in strict flow order, so no
path doubles back; the one return path in `diagram_loop` is the subject of that
figure rather than an accident. Conventions follow figures/16-vision-language:
currentColor for text and rules, viridis for data, title/desc with namespaced
ids, no background rect (the theme supplies a light plate in both schemes).
Yellow is a fill only, never a line or a label.

Run from the repository root; rewrites all nine files.
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
    polyline,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent
W = 840


# --------------------------------------------------------------------------
# Computation: compounding error under behaviour cloning.
# --------------------------------------------------------------------------


def bc_cost_sum(horizon, eps):
    """Expected cost of behaviour cloning, summed term by term over the horizon."""
    return sum(1.0 - (1.0 - eps) ** t for t in range(1, horizon + 1))


def bc_cost_closed(horizon, eps):
    """The same quantity from the closed form of the geometric series."""
    q = 1.0 - eps
    return horizon - q * (1.0 - q**horizon) / eps


def bc_cost_quadratic(horizon, eps):
    """The small-epsilon expansion the O(epsilon T^2) bound comes from."""
    return eps * horizon * (horizon + 1) / 2.0


def dagger_cost(horizon, eps):
    """Expected cost when training on the learner's own state distribution."""
    return eps * horizon


def quadratic_ratio(horizon, eps):
    """Ratio of the exact cost to its quadratic expansion; tends to 1 as eps falls."""
    return bc_cost_sum(horizon, eps) / bc_cost_quadratic(horizon, eps)


# --------------------------------------------------------------------------
# Computation: a two-mode action distribution.
# --------------------------------------------------------------------------


def normal_pdf(x, mu, sigma):
    """Density of a scalar Gaussian."""
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def mixture_pdf(a, sep, sigma):
    """Equal-weight two-component mixture with modes sought near +/- sep/2."""
    return 0.5 * normal_pdf(a, -sep / 2.0, sigma) + 0.5 * normal_pdf(
        a, sep / 2.0, sigma
    )


def l2_optimal_analytic(sep):
    """The prediction minimising expected squared error: the mixture mean."""
    return 0.5 * (-sep / 2.0) + 0.5 * (sep / 2.0)


def l2_optimal_numeric(sep, sigma, span=6.0, n=4001):
    """The same prediction, found by minimising expected squared error on a grid."""
    lo, hi = -span * sigma - sep, span * sigma + sep
    step = (hi - lo) / (n - 1)
    grid = [lo + i * step for i in range(n)]
    dens = [mixture_pdf(a, sep, sigma) for a in grid]

    best, best_risk = None, float("inf")
    for cand in grid:
        risk = sum(p * (a - cand) ** 2 for a, p in zip(grid, dens)) * step
        if risk < best_risk:
            best_risk, best = risk, cand
    return best


def count_modes(sep, sigma, span=6.0, n=6001):
    """Number of strict local maxima of the mixture density on a fine grid."""
    lo, hi = -span * sigma - sep, span * sigma + sep
    step = (hi - lo) / (n - 1)
    dens = [mixture_pdf(lo + i * step, sep, sigma) for i in range(n)]
    return sum(
        1 for i in range(1, n - 1) if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]
    )


def empirical_bimodal_threshold(sigma=1.0, step=0.002):
    """Smallest separation at which the density has two local maxima."""
    sep = step
    while count_modes(sep, sigma) < 2:
        sep += step
    return sep


def mode_to_mean_ratio(sep, sigma):
    """How much likelier the demonstrated action is than the L2-optimal one."""
    return mixture_pdf(sep / 2.0, sep, sigma) / mixture_pdf(0.0, sep, sigma)


# --------------------------------------------------------------------------
# Experience consumed, in hours. Every value is stated by the cited paper or
# derived from a stated one by a conversion named in `EXPERIENCE`.
# --------------------------------------------------------------------------

HOURS_PER_YEAR = 365.25 * 24.0

EXPERIENCE = [
    ("RoboCook", "20 min x 15 tools", 20.0 * 15.0 / 60.0, "real robot", TEAL),
    ("DQN, Atari", "10M frames at 60 Hz", 10e6 / 60.0 / 3600.0, "simulator", GREEN),
    ("pi-zero", "10,000 h stated", 10_000.0, "real robots", INDIGO),
    (
        "Rubik's cube",
        "~13,000 simulated years",
        13_000.0 * HOURS_PER_YEAR,
        "simulator",
        PURPLE,
    ),
]


def experience_span():
    """Orders of magnitude between the cheapest and the costliest result plotted."""
    values = [hours for _, _, hours, _, _ in EXPERIENCE]
    return math.log10(max(values) / min(values))


# --------------------------------------------------------------------------
# Figures.
# --------------------------------------------------------------------------


def fig_loop():
    """A recognition model is a function; a robot's output returns as its input."""
    h = 340
    b = ""
    b += text(
        W / 2,
        26,
        "The assumption every earlier chapter rests on, and what breaks it",
        size=14,
        weight="600",
    )

    # --- Open loop, left to right, no return. ---
    b += text(210, 58, "Recognition: a function", size=13, style="italic", op=0.75)
    xs = [96, 210, 324]
    labels = ["image", "model", "label / box / caption"]
    for x, lab in zip(xs, labels):
        fill = INDIGO if lab == "model" else "none"
        op = 0.16 if lab == "model" else 0.0
        b += rect(x - 52, 88, 104, 40, fill=fill, op=op, rx=5, so=0.6)
        b += text(x, 113, lab, size=11.5)
    b += arrow(xs[0] + 52, 108, xs[1] - 52, 108)
    b += arrow(xs[1] + 52, 108, xs[2] - 52, 108)
    b += text(210, 160, "nothing returns; the next input is", size=11, op=0.7)
    b += text(210, 176, "drawn from a fixed dataset", size=11, op=0.7)
    b += text(210, 206, "train / test split is meaningful", size=11, op=0.85, font=MONO)

    b += line(430, 60, 430, 300, op=0.22, dash="4 4")

    # --- Closed loop. Forward path left to right; the return path is the point. ---
    b += text(650, 58, "Robot learning: a loop", size=13, style="italic", op=0.75)
    b += rect(510, 88, 104, 40, fill=TEAL, op=0.16, rx=5, so=0.6)
    b += text(562, 113, "agent", size=11.5)
    b += rect(686, 88, 104, 40, fill=YELLOW, op=0.3, rx=5, so=0.6)
    b += text(738, 113, "world", size=11.5)
    b += arrow(614, 108, 686, 108)
    b += text(650, 99, "action a", size=10.5, op=0.8, font=MONO)

    # Return path, drawn below both boxes so it reads as a circuit, not a reversal.
    b += line(738, 128, 738, 170, op=0.55)
    b += line(738, 170, 562, 170, op=0.55)
    b += arrow(562, 170, 562, 130, op=0.55)
    b += text(650, 186, "state s', reward r", size=10.5, op=0.8, font=MONO)

    b += text(650, 216, "the world the agent sees next is the world", size=11, op=0.7)
    b += text(650, 232, "its own last action produced", size=11, op=0.7)
    b += text(
        650,
        262,
        "the training distribution moves with the policy",
        size=11,
        op=0.85,
        font=MONO,
    )

    b += text(
        W / 2,
        314,
        "Everything difficult in this chapter follows from the return path on the right.",
        size=11.5,
        op=0.7,
        style="italic",
    )

    return wrap(
        W,
        h,
        "loop",
        "Open-loop recognition against the closed perception-action loop",
        "On the left an image enters a model and an output leaves, with nothing returning. "
        "On the right an agent emits an action into the world and the world returns the next "
        "state and a reward, so the data the agent is trained on is produced by the policy "
        "being trained.",
        b,
    )


def fig_perception():
    """Four axes on which robot perception departs from recognition."""
    h = 372
    b = ""
    b += text(
        W / 2,
        26,
        "Four things robot perception needs that a recognition benchmark does not measure",
        size=14,
        weight="600",
    )

    cols = [
        (
            "3D and metric",
            "a box in pixels",
            "a pose in metres,",
            "in the robot's frame,",
            "with uncertainty",
            TEAL,
        ),
        (
            "Active",
            "a fixed dataset",
            "the robot chooses",
            "its own viewpoints",
            "and can act to see",
            GREEN,
        ),
        (
            "Real-time",
            "accuracy at any cost",
            "a control deadline;",
            "an accurate model",
            "that misses it is unusable",
            INDIGO,
        ),
        (
            "Task-driven",
            "one general representation",
            "correctness depends",
            "on the action next;",
            "co-designed with control",
            PURPLE,
        ),
    ]

    x0, gap, cw = 40, 12, (W - 80 - 3 * 12) / 4
    for i, (title, cv, r1, r2, r3, colour) in enumerate(cols):
        x = x0 + i * (cw + gap)
        cx = x + cw / 2
        b += text(cx, 62, title, size=12.5, weight="600", fill=colour)

        b += rect(x, 78, cw, 52, fill="none", so=0.3, rx=4)
        b += text(cx, 98, "computer vision", size=10, op=0.55, style="italic")
        b += text(cx, 116, cv, size=10.5, op=0.75)

        b += arrow(cx, 136, cx, 158, op=0.5)

        b += rect(x, 164, cw, 92, fill=colour, op=0.12, so=0.5, rx=4)
        b += text(cx, 184, "robot vision", size=10, op=0.55, style="italic")
        b += text(cx, 204, r1, size=10.5)
        b += text(cx, 220, r2, size=10.5)
        b += text(cx, 236, r3, size=10.5)

    b += line(40, 284, W - 40, 284, op=0.2)
    b += text(
        W / 2,
        308,
        "The sensors follow: proprioception is fast and never occluded, force and torque report contact,",
        size=11.5,
        op=0.72,
    )
    b += text(
        W / 2,
        326,
        "and tactile sensing is most informative exactly when the hand is occluding what it holds.",
        size=11.5,
        op=0.72,
    )
    b += text(
        W / 2,
        352,
        "No general-purpose representation is correct independently of the action about to be taken.",
        size=11.5,
        op=0.7,
        style="italic",
    )

    return wrap(
        W,
        h,
        "perception",
        "The four axes separating robot perception from recognition",
        "Four columns, each contrasting what computer vision supplies with what a robot needs: "
        "a metric 3D pose rather than a pixel box, active viewpoint selection rather than a fixed "
        "dataset, a hard control deadline, and a representation whose adequacy depends on the task.",
        b,
    )


def fig_credit():
    """A delayed scalar reward must be distributed over every decision before it."""
    h = 380
    b = ""
    b += text(W / 2, 26, "Where the learning signal attaches", size=14, weight="600")

    # --- Supervised: the loss lands on the prediction that caused it. ---
    b += text(150, 58, "Supervised", size=12.5, style="italic", op=0.75)
    b += rect(72, 76, 156, 34, fill=GREEN, op=0.15, rx=4, so=0.5)
    b += text(150, 98, "prediction", size=11)
    b += rect(72, 152, 156, 34, fill="none", so=0.5, rx=4)
    b += text(150, 174, "loss", size=11)
    b += arrow(150, 152, 150, 114, op=0.8, stroke=GREEN)
    b += text(150, 208, "one gradient, one output", size=10.5, op=0.7)
    b += text(150, 224, "responsibility is unambiguous", size=10.5, op=0.7)

    b += line(268, 60, 268, 300, op=0.22, dash="4 4")

    # --- Sequential: one scalar at the end of a trajectory. ---
    b += text(560, 58, "Sequential decision-making", size=12.5, style="italic", op=0.75)
    n = 6
    xs = [318 + i * 82 for i in range(n)]
    for i, x in enumerate(xs):
        b += circle(x, 98, 15, fill=TEAL, op=0.18, so=0.5)
        b += text(x, 102, f"s{i}", size=10.5, font=MONO)
        if i < n - 1:
            b += arrow(x + 15, 98, xs[i + 1] - 15, 98, op=0.6)
            b += text((x + xs[i + 1]) / 2, 87, f"a{i}", size=9.5, op=0.7, font=MONO)

    b += rect(xs[-1] + 26, 80, 62, 36, fill=YELLOW, op=0.35, rx=4, so=0.55)
    b += text(xs[-1] + 57, 103, "R", size=13, font=MONO)
    b += arrow(xs[-1] + 15, 98, xs[-1] + 26, 98, op=0.6)

    # One note rather than a label per gap: the per-gap labels sat exactly where
    # the credit arrows converge on the reward box.
    b += text(
        (xs[0] + xs[-1]) / 2, 128, "r = 0 at every step", size=9.5, op=0.5, font=MONO
    )

    # Credit fans back from the terminal reward to every earlier action.
    for x in xs[:-1]:
        b += arrow(
            xs[-1] + 40,
            120,
            x + 6,
            152,
            op=0.3,
            sw=1.0,
            head=5.0,
            stroke=PURPLE,
            dash="3 3",
        )
    b += text(
        560,
        176,
        "which of these earned it?",
        size=11,
        op=0.8,
        fill=PURPLE,
        style="italic",
    )

    b += text(
        560,
        208,
        "one scalar, arriving once, for every decision in the episode",
        size=10.5,
        op=0.7,
    )
    b += text(
        560,
        224,
        "a game of Go returns one bit after roughly 250 moves",
        size=10.5,
        op=0.7,
    )

    b += line(40, 258, W - 40, 258, op=0.2)
    b += text(
        W / 2,
        288,
        "The action-value function converts the backward question into a forward one:",
        size=12,
        op=0.75,
    )
    b += text(
        W / 2,
        316,
        "Q(s, a) is the return expected from here on, so each transition supplies its own target",
        size=12,
        op=0.75,
    )
    b += text(
        W / 2,
        346,
        "and credit propagates one step at a time instead of being traced through the sequence.",
        size=12,
        op=0.75,
    )

    return wrap(
        W,
        h,
        "credit",
        "Credit assignment in supervised learning against sequential decision-making",
        "On the left a loss attaches directly to the prediction that produced it. On the right a "
        "trajectory of six states returns zero reward at every step and a single scalar at the end, "
        "with dashed arrows fanning back from that scalar to every earlier action.",
        b,
    )


def fig_experience():
    """Experience consumed by four cited results, on a logarithmic axis."""
    h = 372
    b = ""
    b += text(W / 2, 26, "Experience consumed, in hours", size=14, weight="600")
    b += text(
        W / 2,
        46,
        "logarithmic axis; each value is stated by the cited paper or converted from one that is",
        size=11,
        op=0.65,
        style="italic",
    )

    lo, hi = 1.0, 1e9
    x0, x1 = 210, W - 60
    to_px = mapper((math.log10(lo), math.log10(hi)), (0, 1), (x0, x1), (0, 1))

    # Decade rules first, so the bars sit over them.
    for d in range(0, 10):
        gx = to_px(d, 0)[0]
        b += line(gx, 74, gx, 286, op=0.14)
        label = "1" if d == 0 else f"10^{d}"
        b += text(gx, 304, label, size=10, op=0.6, font=MONO)
    b += text((x0 + x1) / 2, 326, "hours of interaction", size=11, op=0.7)

    row_h, top = 46, 88
    for i, (name, note, hours, kind, colour) in enumerate(EXPERIENCE):
        y = top + i * row_h
        bx = to_px(math.log10(hours), 0)[0]
        b += rect(x0, y, bx - x0, 24, fill=colour, op=0.55, stroke="none", so=0)
        b += text(x0 - 12, y + 12, name, size=11.5, anchor="end")
        b += text(x0 - 12, y + 27, note, size=9.5, anchor="end", op=0.6, font=MONO)
        b += text(bx + 8, y + 17, kind, size=10, anchor="start", op=0.7, style="italic")

    b += line(x0, 286, x1, 286, op=0.35)
    b += line(x0, 74, x0, 286, op=0.35)

    span = experience_span()
    b += text(
        W / 2,
        352,
        f"About {span:.0f} orders of magnitude, and the cheap end is the physical manipulation.",
        size=11.5,
        op=0.72,
        style="italic",
    )

    return wrap(
        W,
        h,
        "experience",
        "Experience consumed by four robot-learning results, on a log axis",
        "Four horizontal bars on a logarithmic axis running from one hour to a billion hours: "
        "RoboCook at five hours on a real robot, DQN at forty-six simulated hours, pi-zero at ten "
        "thousand hours on real robots, and the Rubik's cube policy at roughly one hundred million "
        "simulated hours.",
        b,
    )


def _pile(cx, cy, seed, n=26, spread=34.0):
    """A deterministic scatter of particle positions, used by two figures."""
    pts = []
    state = seed
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        u = state / 2147483648.0
        state = (state * 1103515245 + 12345) % 2147483648
        v = state / 2147483648.0
        r = spread * math.sqrt(u)
        th = 2.0 * math.pi * v
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th) * 0.55))
    return pts


def fig_state_representations():
    """One scene under pixels, keypoints and particles."""
    h = 392
    b = ""
    b += text(
        W / 2,
        26,
        "What is s? Three answers, and what each one lets the dynamics model be",
        size=14,
        weight="600",
    )

    panels = [
        (
            "Pixels",
            "no annotation of any kind;",
            "the model predicts appearance,",
            "blurs over the horizon",
            TEAL,
        ),
        (
            "Keypoints",
            "compact and metric;",
            "assumes a fixed, roughly",
            "rigid set of landmarks",
            GREEN,
        ),
        (
            "Particles",
            "no fixed count, no identity;",
            "one model covers piles,",
            "dough and fluids",
            INDIGO,
        ),
    ]
    pw, gap = 240, 30
    x0 = (W - 3 * pw - 2 * gap) / 2

    for i, (title, l1, l2, l3, colour) in enumerate(panels):
        x = x0 + i * (pw + gap)
        cx = x + pw / 2
        b += text(cx, 60, title, size=13, weight="600", fill=colour)
        b += rect(x, 74, pw, 150, fill="none", so=0.3, rx=5)

        if i == 0:
            # A pixel grid over the whole panel: the model must predict all of it.
            for gx in range(12):
                for gy in range(8):
                    px, py = x + 18 + gx * 17, 92 + gy * 17
                    shade = 0.30 if (4 <= gx <= 8 and 3 <= gy <= 6) else 0.10
                    b += rect(
                        px, py, 15, 15, fill=colour, op=shade, stroke="none", so=0
                    )
        elif i == 1:
            # A rigid box with eight tracked landmarks.
            b += rect(cx - 62, 118, 124, 66, fill=colour, op=0.12, so=0.45, rx=3)
            for kx in (cx - 62, cx, cx + 62):
                for ky in (118, 151, 184):
                    if kx == cx and ky == 151:
                        continue
                    b += circle(kx, ky, 4.5, fill=colour, op=0.95, stroke="none", so=0)
        else:
            # An unordered cloud with interaction edges.
            pts = _pile(cx, 150, seed=17)
            for j, (ax, ay) in enumerate(pts):
                for bx2, by2 in pts[j + 1 :]:
                    if (ax - bx2) ** 2 + (ay - by2) ** 2 < 19.0**2:
                        b += line(ax, ay, bx2, by2, stroke=colour, op=0.3, sw=0.8)
            for px, py in pts:
                b += circle(px, py, 3.4, fill=colour, op=0.9, stroke="none", so=0)

        b += text(cx, 248, l1, size=10.5, op=0.78)
        b += text(cx, 265, l2, size=10.5, op=0.78)
        b += text(cx, 282, l3, size=10.5, op=0.78)

    b += line(40, 312, W - 40, 312, op=0.2)
    b += text(
        W / 2,
        340,
        "The planner is the same in all three; the distance in the planning objective is measured",
        size=11.5,
        op=0.72,
    )
    b += text(
        W / 2,
        358,
        "in whichever of these spaces the state lives, which is why the choice decides the result.",
        size=11.5,
        op=0.72,
    )

    return wrap(
        W,
        h,
        "staterep",
        "Pixels, keypoints and particles as three state representations",
        "Three panels of the same scene. The first is a dense pixel grid the model must predict "
        "in full. The second is a rigid box with eight tracked landmarks. The third is an "
        "unordered cloud of particles joined by short interaction edges.",
        b,
    )


def fig_particle_graph():
    """Shared relation and node functions over a particle graph."""
    h = 400
    b = ""
    b += text(
        W / 2,
        26,
        "Why a particle graph transfers when the scene changes",
        size=14,
        weight="600",
    )

    # --- Left: the local rule, drawn once. ---
    b += text(
        196,
        60,
        "The rule, applied at every edge and every node",
        size=12,
        style="italic",
        op=0.75,
    )

    nodes = {
        "i": (196, 148),
        "j1": (128, 106),
        "j2": (268, 118),
        "j3": (150, 210),
        "j4": (256, 200),
    }
    for key, (nx, ny) in nodes.items():
        if key == "i":
            continue
        b += line(nx, ny, nodes["i"][0], nodes["i"][1], stroke=INDIGO, op=0.45, sw=1.3)
    for key, (nx, ny) in nodes.items():
        r = 15 if key == "i" else 11
        b += circle(
            nx, ny, r, fill=INDIGO if key == "i" else TEAL, op=0.75, stroke="none", so=0
        )
        b += text(nx, ny + 4, key.replace("j", "j"), size=10, fill="#ffffff", font=MONO)

    b += text(
        196,
        262,
        "e(i,j) = f_rel(v_i, v_j)   for every edge",
        size=11,
        font=MONO,
        op=0.85,
    )
    b += text(
        196, 282, "v_i' = f_node(v_i, sum of e(i,j))", size=11, font=MONO, op=0.85
    )
    b += text(196, 310, "the same f_rel and the same f_node", size=11, op=0.7)
    b += text(196, 326, "everywhere in the scene", size=11, op=0.7)

    b += line(392, 60, 392, 340, op=0.22, dash="4 4")

    # --- Right: the same weights over two scenes of different size. ---
    b += text(620, 60, "Same weights, two scenes", size=12, style="italic", op=0.75)

    for k, (cx, n, spread, cap) in enumerate(
        [(516, 22, 44.0, "26 particles"), (724, 62, 62.0, "78 particles")]
    ):
        pts = _pile(cx, 168, seed=91 + k * 7, n=n, spread=spread)
        for j, (ax, ay) in enumerate(pts):
            for bx2, by2 in pts[j + 1 :]:
                if (ax - bx2) ** 2 + (ay - by2) ** 2 < 21.0**2:
                    b += line(ax, ay, bx2, by2, stroke=GREEN, op=0.3, sw=0.8)
        for px, py in pts:
            b += circle(px, py, 3.6, fill=GREEN, op=0.9, stroke="none", so=0)
        b += text(cx, 262, cap, size=10.5, op=0.7, font=MONO)

    b += text(
        620,
        292,
        "No parameter counts the particles and none names them,",
        size=11,
        op=0.72,
    )
    b += text(
        620, 308, "so a model fitted to one pile predicts the other", size=11, op=0.72
    )
    b += text(
        620,
        324,
        "and a deformable object may split without breaking it.",
        size=11,
        op=0.72,
    )

    b += line(40, 356, W - 40, 356, op=0.2)
    b += text(
        W / 2,
        380,
        "Compositional structure in the architecture rather than in the training distribution.",
        size=11.5,
        op=0.72,
        style="italic",
    )

    return wrap(
        W,
        h,
        "particlegraph",
        "Message passing on a particle graph and its transfer across scenes",
        "On the left a central particle receives edges from four neighbours, annotated with the "
        "shared relation and node functions. On the right the same functions are applied to two "
        "particle clouds of different size, twenty-six and seventy-eight particles.",
        b,
    )


def fig_compounding():
    """Expected cost against horizon under behaviour cloning and DAgger."""
    h = 430
    eps = 0.01
    t_max = 200
    b = ""
    b += text(
        W / 2,
        26,
        "Compounding error: why supervised learning on demonstrations is not supervised learning",
        size=14,
        weight="600",
    )
    b += text(
        W / 2,
        46,
        f"per-step error rate on the expert's states: {eps}",
        size=11,
        op=0.65,
        font=MONO,
    )

    px, py = (110, W - 250), (352, 76)
    y_hi = 210.0
    # `mapper` wants (top, bottom) for its pixel y range; py is stored the other
    # way round here because the axis drawing below reads py[0] as the baseline.
    to_px = mapper((0, t_max), (0, y_hi), px, (py[1], py[0]))

    # Axes and ticks.
    b += line(px[0], py[0], px[1], py[0], op=0.35)
    b += line(px[0], py[0], px[0], py[1], op=0.35)
    for t in range(0, t_max + 1, 50):
        gx, _ = to_px(t, 0)
        b += line(gx, py[0], gx, py[0] + 5, op=0.35)
        b += text(gx, py[0] + 20, str(t), size=10, op=0.65, font=MONO)
    for v in range(0, int(y_hi) + 1, 50):
        _, gy = to_px(0, v)
        b += line(px[0] - 5, gy, px[0], gy, op=0.35)
        b += text(px[0] - 10, gy + 4, str(v), size=10, anchor="end", op=0.65, font=MONO)
    b += text((px[0] + px[1]) / 2, py[0] + 42, "horizon T (steps)", size=11.5, op=0.75)
    b += text(
        px[0] - 8, py[1] - 12, "expected cost", size=11.5, op=0.75, anchor="start"
    )

    ts = list(range(0, t_max + 1, 2))

    # The quadratic expansion, drawn only where it is below the horizon ceiling.
    quad = [
        to_px(t, bc_cost_quadratic(t, eps))
        for t in ts
        if bc_cost_quadratic(t, eps) <= y_hi
    ]
    b += polyline(quad, stroke=PURPLE, sw=1.4, op=0.6, dash="5 4")

    # The ceiling: cost cannot exceed the horizon.
    b += polyline(
        [to_px(t, min(float(t), y_hi)) for t in ts],
        stroke=INK,
        sw=1.0,
        op=0.28,
        dash="2 4",
    )

    b += polyline([to_px(t, bc_cost_sum(t, eps)) for t in ts], stroke=TEAL, sw=2.2)
    b += polyline([to_px(t, dagger_cost(t, eps)) for t in ts], stroke=GREEN, sw=2.2)

    # Labels, placed to the right of the plot so none sits on a curve.
    lx = px[1] + 14
    b += text(
        lx,
        to_px(t_max, bc_cost_quadratic(t_max, eps))[1] + 4,
        "eps T(T+1)/2",
        size=10.5,
        anchor="start",
        fill=PURPLE,
        font=MONO,
    )
    # The ceiling is labelled inline, well left of the right-hand stack, because at
    # T = 200 it and the quadratic expansion land within one unit of each other.
    b += text(
        to_px(150, 0)[0] - 8,
        to_px(0, 156)[1],
        "T (total failure)",
        size=10.5,
        anchor="end",
        op=0.45,
        font=MONO,
    )
    b += text(
        lx,
        to_px(t_max, bc_cost_sum(t_max, eps))[1] + 4,
        "behaviour cloning",
        size=10.5,
        anchor="start",
        fill=TEAL,
    )
    b += text(
        lx,
        to_px(t_max, dagger_cost(t_max, eps))[1] + 4,
        "DAgger, eps T",
        size=10.5,
        anchor="start",
        fill=GREEN,
    )

    b += line(40, 390, W - 40, 390, op=0.2)
    ratio = bc_cost_sum(100, eps) / dagger_cost(100, eps)
    b += text(
        W / 2,
        414,
        f"At T = 100 the gap is {ratio:.0f}x, and the quadratic expansion is an upper bound that the exact cost leaves behind once eps T approaches 1.",
        size=11.5,
        op=0.72,
        style="italic",
    )

    return wrap(
        W,
        h,
        "compounding",
        "Expected cost against horizon for behaviour cloning and DAgger",
        "A plot with horizon on the x axis and expected cost on the y axis. Behaviour cloning "
        "grows quadratically and then bends towards the total-failure ceiling; DAgger grows "
        "linearly and stays far below it; the quadratic expansion is drawn dashed and rises above "
        "the exact curve.",
        b,
    )


def fig_multimodal():
    """Two good actions, and the action that averaging them produces."""
    h = 430
    sigma = 1.0
    b = ""
    b += text(
        W / 2,
        26,
        "Two equally good demonstrated actions, and what regression returns",
        size=14,
        weight="600",
    )

    # --- Panel (a): one separation, the three candidate answers marked. ---
    sep = 4.0
    px, py = (86, 400), (296, 76)
    a_lo, a_hi = -5.0, 5.0
    # Both panels share one density range so they can be read against each other,
    # and it is set by the tallest curve drawn in *either* — the d = 1 sigma sweep
    # in panel (b), whose peak is 0.352, not the d = 4 sigma curve in panel (a).
    d_hi = 0.38
    # `mapper` takes (top, bottom); py is stored the other way round because the
    # baseline is drawn from py[0].
    to_px = mapper((a_lo, a_hi), (0, d_hi), px, (py[1], py[0]))

    b += text(
        (px[0] + px[1]) / 2,
        58,
        f"separation d = {sep:.0f} sigma",
        size=12,
        style="italic",
        op=0.75,
    )
    b += line(px[0], py[0], px[1], py[0], op=0.35)

    curve = [
        to_px(
            a_lo + i * (a_hi - a_lo) / 400.0,
            mixture_pdf(a_lo + i * (a_hi - a_lo) / 400.0, sep, sigma),
        )
        for i in range(401)
    ]
    b += polyline(curve, stroke=TEAL, sw=2.2)

    for m in (-sep / 2.0, sep / 2.0):
        mx, my = to_px(m, mixture_pdf(m, sep, sigma))
        b += line(mx, py[0], mx, my, stroke=GREEN, op=0.55, sw=1.4, dash="4 3")
        b += circle(mx, my, 4.5, fill=GREEN, stroke="none", so=0)
    b += text((px[0] + px[1]) / 2, py[0] + 20, "action a", size=11, op=0.7)
    b += text(
        to_px(-sep / 2.0, 0)[0], py[0] + 38, "go left", size=10.5, fill=GREEN, font=MONO
    )
    b += text(
        to_px(sep / 2.0, 0)[0], py[0] + 38, "go right", size=10.5, fill=GREEN, font=MONO
    )

    zx, zy = to_px(0.0, mixture_pdf(0.0, sep, sigma))
    b += line(zx, py[0], zx, zy, stroke=PURPLE, op=0.6, sw=1.4)
    b += circle(zx, zy, 5.0, fill=PURPLE, stroke="none", so=0)
    # The annotation goes in the gap between the two peaks, which is empty from the
    # top of the panel down to the trough the arrow points at.
    b += text(zx, py[1] + 34, "the L2-optimal action", size=11, fill=PURPLE)
    b += text(zx, py[1] + 50, "is a local minimum of p(a)", size=11, fill=PURPLE)
    b += arrow(zx, py[1] + 60, zx, zy - 10, stroke=PURPLE, op=0.6, sw=1.2)

    ratio = mode_to_mean_ratio(sep, sigma)
    b += text(
        (px[0] + px[1]) / 2,
        py[0] + 62,
        f"a demonstrated action is {ratio:.1f}x likelier than the average of the two",
        size=11,
        op=0.72,
        style="italic",
    )

    # --- Panel (b): the sweep, and the threshold. ---
    px2, py2 = (474, W - 60), (296, 76)
    to_px2 = mapper((a_lo, a_hi), (0, d_hi), px2, (py2[1], py2[0]))
    b += text(
        (px2[0] + px2[1]) / 2,
        58,
        "the failure has a threshold",
        size=12,
        style="italic",
        op=0.75,
    )
    b += line(px2[0], py2[0], px2[1], py2[0], op=0.35)

    for s, colour, dash in ((1.0, INDIGO, "5 4"), (2.0, INK, "2 3"), (3.0, TEAL, None)):
        pts = [
            to_px2(
                a_lo + i * (a_hi - a_lo) / 400.0,
                mixture_pdf(a_lo + i * (a_hi - a_lo) / 400.0, s, sigma),
            )
            for i in range(401)
        ]
        b += polyline(
            pts, stroke=colour, sw=1.8, op=0.85 if dash is None else 0.7, dash=dash
        )

    # The right tail is the only corner all three curves have vacated by the top of
    # the panel, so the key sits there rather than over the d = 1 sigma peak.
    for i, (lab, colour, op) in enumerate(
        (
            ("d = 1 sigma", INDIGO, 1.0),
            ("d = 2 sigma", INK, 0.55),
            ("d = 3 sigma", TEAL, 1.0),
        )
    ):
        b += text(
            px2[1],
            py2[1] + 22 + i * 16,
            lab,
            size=10.5,
            anchor="end",
            fill=colour,
            op=op,
            font=MONO,
        )

    b += text((px2[0] + px2[1]) / 2, py2[0] + 20, "action a", size=11, op=0.7)
    b += text(
        (px2[0] + px2[1]) / 2,
        py2[0] + 44,
        "below d = 2 sigma the density has one peak, the mean",
        size=11,
        op=0.72,
    )
    b += text(
        (px2[0] + px2[1]) / 2,
        py2[0] + 62,
        "lies on it, and regression is the right answer after all.",
        size=11,
        op=0.72,
    )

    b += line(40, 390, W - 40, 390, op=0.2)
    b += text(
        W / 2,
        414,
        "An energy or a diffusion policy represents the whole distribution, so it commits to one mode instead of averaging them.",
        size=11.5,
        op=0.72,
        style="italic",
    )

    return wrap(
        W,
        h,
        "multimodal",
        "A two-mode action distribution and the prediction squared error selects",
        "On the left the density of an equal-weight two-mode action distribution separated by four "
        "standard deviations, with the two modes marked and the L2-optimal prediction marked at the "
        "trough between them. On the right the same density at separations of one, two and three "
        "standard deviations, showing the transition from unimodal to bimodal at exactly two.",
        b,
    )


def fig_vla():
    """The pi-zero stack: a pretrained VLM, an action expert, a chunk of actions."""
    h = 400
    b = ""
    b += text(
        W / 2, 26, "A vision-language-action model: pi-zero", size=14, weight="600"
    )
    b += text(
        W / 2,
        46,
        "no explicit state, no transition function — observation and goal in, actions out",
        size=11,
        op=0.65,
        style="italic",
    )

    # Strict flow order left to right: inputs, backbone, expert, chunk, robot.
    b += rect(46, 108, 118, 40, fill="none", so=0.5, rx=5)
    b += text(105, 132, "camera images", size=11)
    b += rect(46, 158, 118, 40, fill="none", so=0.5, rx=5)
    b += text(105, 182, "instruction", size=11)
    b += rect(46, 208, 118, 40, fill="none", so=0.5, rx=5)
    b += text(105, 232, "joint states", size=11)

    b += rect(216, 108, 132, 90, fill=TEAL, op=0.18, rx=5, so=0.55)
    b += text(282, 145, "VLM backbone", size=11.5)
    b += text(282, 163, "PaliGemma, 3B", size=10, op=0.7, font=MONO)
    b += text(282, 180, "web pretrained", size=10, op=0.7, style="italic")

    b += rect(400, 108, 132, 90, fill=INDIGO, op=0.18, rx=5, so=0.55)
    b += text(466, 145, "action expert", size=11.5)
    b += text(466, 163, "300M", size=10, op=0.7, font=MONO)
    b += text(466, 180, "flow matching", size=10, op=0.7, style="italic")

    for y in (128, 178, 228):
        b += arrow(164, y, 216, min(max(y, 128), 178), op=0.55)
    b += arrow(348, 153, 400, 153, op=0.6)

    # The chunk: H future actions emitted in one pass.
    b += text(650, 96, "one action chunk per pass", size=11, op=0.75, style="italic")
    for i in range(8):
        cx = 578 + i * 22
        b += rect(cx, 122, 18, 30, fill=YELLOW, op=0.45, stroke="none", so=0)
    b += arrow(532, 153, 574, 137, op=0.6)
    b += text(666, 172, "up to 50 Hz", size=10.5, op=0.75, font=MONO)
    b += text(666, 190, "continuous actions, not tokens", size=10.5, op=0.75)

    b += rect(600, 214, 132, 40, fill=GREEN, op=0.2, rx=5, so=0.55)
    b += text(666, 238, "robot", size=11.5)
    b += arrow(666, 200, 666, 214, op=0.6)

    b += line(40, 288, W - 40, 288, op=0.2)
    b += text(
        W / 2,
        314,
        "Pretraining: about 10,000 hours across 7 robot configurations and 68 tasks, plus open cross-embodiment data.",
        size=11.5,
        op=0.72,
    )
    b += text(
        W / 2,
        336,
        "The backbone supplies semantics the robot data could never contain; the expert supplies a rate the backbone could never reach.",
        size=11.5,
        op=0.72,
    )
    b += text(
        W / 2,
        366,
        "Nothing here models the environment. The generated trajectory is always fluent, which is not the same as always correct.",
        size=11.5,
        op=0.72,
        style="italic",
    )

    return wrap(
        W,
        h,
        "vla",
        "The pi-zero vision-language-action stack",
        "Camera images, a language instruction and joint states enter a pretrained vision-language "
        "backbone, whose representation feeds a smaller action expert that emits a chunk of eight "
        "continuous future actions by flow matching, which are executed on the robot at up to fifty "
        "hertz.",
        b,
    )


FIGURES = {
    "diagram_loop.svg": fig_loop,
    "diagram_perception.svg": fig_perception,
    "diagram_credit.svg": fig_credit,
    "diagram_experience.svg": fig_experience,
    "diagram_state_representations.svg": fig_state_representations,
    "diagram_particle_graph.svg": fig_particle_graph,
    "diagram_compounding.svg": fig_compounding,
    "diagram_multimodal.svg": fig_multimodal,
    "diagram_vla.svg": fig_vla,
}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print(f"wrote {name}")
    print()

    # --- Compounding error: two routes to the same expected cost. ---
    for horizon in (1, 5, 25, 100, 400):
        for eps in (0.001, 0.01, 0.05, 0.2):
            a, c = bc_cost_sum(horizon, eps), bc_cost_closed(horizon, eps)
            assert abs(a - c) < 1e-9, (horizon, eps, a, c)
    print(
        "  Behaviour-cloning cost: summation and closed form agree to 1e-9 at 20 (T, eps) pairs"
    )

    # The cost of an episode can never exceed its length. The asymptotic
    # statement hides this, and it is what makes the growth non-quadratic late.
    for horizon in (10, 100, 1000, 10_000):
        for eps in (0.001, 0.01, 0.1, 0.5):
            assert bc_cost_sum(horizon, eps) <= horizon + 1e-9, (horizon, eps)
    print(
        "  Exact cost never exceeds the horizon, so the quadratic regime is bounded by eps T ~ 1"
    )
    print()

    print("  The bound is O(eps T^2); the construction's leading term is eps T(T+1)/2:")
    for eps in (0.1, 0.01, 0.001, 0.0001):
        print(
            f"    eps = {eps:<8} exact / [eps T(T+1)/2] at T = 50: {quadratic_ratio(50, eps):.4f}"
        )
    assert quadratic_ratio(50, 1e-6) > 0.9999, quadratic_ratio(50, 1e-6)
    print(
        "    -> ratio tends to 1, so the constant is 1/2 and the +1 is real, not a rounding of eps T^2"
    )
    print()

    print("  Behaviour cloning against DAgger at eps = 0.01:")
    for horizon in (10, 50, 100, 200, 400):
        bc, dag = bc_cost_sum(horizon, 0.01), dagger_cost(horizon, 0.01)
        print(
            f"    T = {horizon:<5} BC {bc:8.2f}   DAgger {dag:6.2f}   ratio {bc / dag:6.2f}x"
        )
    print()

    # --- Multimodality: the L2-optimal action, two ways. ---
    for sep in (1.0, 2.0, 4.0, 8.0):
        analytic = l2_optimal_analytic(sep)
        numeric = l2_optimal_numeric(sep, 1.0)
        assert abs(analytic - numeric) < 5e-3, (sep, analytic, numeric)
    print(
        "  L2-optimal action: analytic mixture mean and grid argmin agree to 5e-3 at four separations"
    )

    # The check worth having: "the mean falls between the modes" is only a
    # failure above a threshold, and the threshold is exactly two sigma.
    empirical = empirical_bimodal_threshold(sigma=1.0)
    assert abs(empirical - 2.0) < 0.01, empirical
    print(
        f"  Bimodality threshold: closed form 2.000 sigma, counted from the density {empirical:.3f} sigma"
    )
    assert count_modes(1.99, 1.0) == 1 and count_modes(2.01, 1.0) == 2
    print("    d = 1.99 sigma -> 1 mode; d = 2.01 sigma -> 2 modes")
    print()

    print("  Below the threshold the mean IS the mode, so regression is correct:")
    for sep in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        modes = count_modes(sep, 1.0)
        ratio = mode_to_mean_ratio(sep, 1.0)
        verdict = (
            "regression fine" if modes == 1 else "regression averages into the gap"
        )
        print(
            f"    d = {sep:<4} sigma  modes {modes}   p(mode)/p(mean) {ratio:6.2f}   {verdict}"
        )
    print()

    # --- Numbers the experience figure quotes. ---
    print("  Experience consumed, hours:")
    for name, note, hours, kind, _ in EXPERIENCE:
        print(f"    {name:<14} {hours:>14,.1f} h   ({note}, {kind})")
    print(f"    span: {experience_span():.2f} orders of magnitude")
