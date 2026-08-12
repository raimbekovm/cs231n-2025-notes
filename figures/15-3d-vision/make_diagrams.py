"""Generate the thirteen SVG diagrams for the lecture 15 chapter.

Nine of the thirteen are computed rather than laid out by hand, and three of
those carry assertions that would fail loudly if the chapter's algebra were
wrong.

`diagram_bezier` evaluates a cubic Bézier twice — once through the Bernstein
polynomial form of the chapter's @eq-bezier, once through de Casteljau's
recursion — and `__main__` asserts the two agree over a dense grid. The
right panel draws the recursion's intermediate polygons at one value of t.

`diagram_sdf_union` is the check the prose leans on. The lecture states the
CSG identities (union = min, intersection = max) without qualification. The
generator computes a brute-force signed distance to the boundary of the union
of two overlapping disks and compares it against min(d1, d2) on a grid. The
two agree to sampling accuracy *outside* the union and diverge inside it, and
`__main__` prints the worst discrepancy on each side plus the closed-form
answer at the centre, where min reports -0.5 against a true -sqrt(3)/2.

`diagram_volume_rendering` evaluates NeRF's discrete compositing sum along one
ray. Two things are asserted: that the weights and the surviving transmittance
partition unity, sum(w_i) + T_{N+1} = 1, which is the telescoping identity of
@eq-weight-partition; and that for a slab of constant density the discrete sum
equals the analytic 1 - exp(-sigma L). The second holds exactly at every sample
count, because the transmittance product telescopes into one exponential, which
localises the discretisation error to density varying between samples.

`diagram_voxel_cost` counts cells two ways for a sphere at each resolution: a
dense grid is n^3, and the octree count comes from an actual recursive
subdivision that descends only into cells the surface crosses. Nothing is read
off a published chart.

`diagram_chamfer_emd` computes both matchings for the same pair of point sets —
independent nearest neighbours for Chamfer, an exact optimal assignment by
brute-force permutation search for the Earth Mover's distance. The figure is
drawn from whatever those return, so a Chamfer collision or an unmatched target
appears because the metric produced it.

`diagram_data_scale` and `diagram_splat_tradeoff` plot published numbers stated
in sentences or tables of the primary sources, never values recovered from a
figure's text layer. Their sources are recorded in figures/SOURCES.md.

The four structural figures are the taxonomy grid, the multi-view pipeline,
PointNet and the AtlasNet decoder, plus the NeRF/splat sampling contrast.

Conventions follow figures/14-generative-models-2: currentColor for text and
rules, viridis for data, title/desc with namespaced ids, no background rect
(the theme supplies a light plate in both schemes). Yellow is a fill only,
never a line or a label.

Run from the repository root; rewrites all thirteen files.
"""

import itertools
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

OUT = pathlib.Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# Bézier: two evaluation routes that must agree.
# ----------------------------------------------------------------------------

CONTROL = [(0.06, 0.18), (0.28, 0.92), (0.74, 0.04), (0.96, 0.66)]


def bernstein(t, pts):
    """Evaluate a Bézier curve through the Bernstein polynomial form."""
    n = len(pts) - 1
    x = y = 0.0
    for i, (px, py) in enumerate(pts):
        w = math.comb(n, i) * (1 - t) ** (n - i) * t**i
        x += w * px
        y += w * py
    return x, y


def de_casteljau(t, pts):
    """Evaluate a Bézier curve by repeated linear interpolation."""
    cur = list(pts)
    while len(cur) > 1:
        cur = [((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1]) for a, b in zip(cur, cur[1:])]
    return cur[0]


def de_casteljau_levels(t, pts):
    """Return every intermediate control polygon of the recursion."""
    levels = [list(pts)]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1]) for a, b in zip(cur, cur[1:])])
    return levels


def bezier_disagreement(steps=2001):
    """Largest gap between the two evaluation routes over a dense grid."""
    worst = 0.0
    for k in range(steps):
        t = k / (steps - 1)
        ax, ay = bernstein(t, CONTROL)
        bx, by = de_casteljau(t, CONTROL)
        worst = max(worst, math.hypot(ax - bx, ay - by))
    return worst


# ----------------------------------------------------------------------------
# Signed distance fields: min(d1, d2) against a true distance to the union.
# ----------------------------------------------------------------------------

DISKS = [(-0.5, 0.0, 1.0), (0.5, 0.0, 1.0)]


def disk_sdf(px, py, disk):
    """Signed distance to one circle; negative inside."""
    cx, cy, r = disk
    return math.hypot(px - cx, py - cy) - r


def composed_sdf(px, py):
    """The CSG union rule: the pointwise minimum of the two fields."""
    return min(disk_sdf(px, py, d) for d in DISKS)


def true_union_sdf(px, py, samples=4000):
    """Brute-force signed distance to the boundary of the union.

    The boundary is the part of each circle that is not strictly inside the other disk, so the arcs swallowed by the
    overlap are excluded — which is exactly what the pointwise minimum fails to notice.
    """
    best = float("inf")
    for i, (cx, cy, r) in enumerate(DISKS):
        other = DISKS[1 - i]
        for k in range(samples):
            a = 2 * math.pi * k / samples
            qx, qy = cx + r * math.cos(a), cy + r * math.sin(a)
            if disk_sdf(qx, qy, other) < -1e-12:
                continue  # buried inside the other disk: not on the boundary
            best = min(best, math.hypot(px - qx, py - qy))
    inside = composed_sdf(px, py) < 0
    return -best if inside else best


def sdf_union_errors(n=61, half=2.6):
    """Worst gap between the composed and true fields, inside and outside."""
    worst_in = worst_out = 0.0
    for i in range(n):
        for j in range(n):
            px = -half + 2 * half * i / (n - 1)
            py = -half + 2 * half * j / (n - 1)
            gap = abs(composed_sdf(px, py) - true_union_sdf(px, py, samples=1200))
            if composed_sdf(px, py) < 0:
                worst_in = max(worst_in, gap)
            else:
                worst_out = max(worst_out, gap)
    return worst_in, worst_out


# ----------------------------------------------------------------------------
# Voxel and octree cell counts for the same sphere.
# ----------------------------------------------------------------------------


def octree_cells(depth, radius=0.42):
    """Count nodes an octree keeps when it subdivides only across a surface.

    A cell is subdivided when the sphere's surface passes through it, which is tested by comparing the cell's distance
    to the surface against its own half-diagonal. Leaves that are wholly inside or wholly outside stop.
    """
    total = 0
    frontier = [(0.5, 0.5, 0.5, 1.0)]  # centre and side length of the root
    for _ in range(depth):
        nxt = []
        for cx, cy, cz, s in frontier:
            for dx, dy, dz in itertools.product((-0.25, 0.25), repeat=3):
                nx, ny, nz = cx + dx * s, cy + dy * s, cz + dz * s
                ns = s / 2
                total += 1
                d = abs(math.dist((nx, ny, nz), (0.5, 0.5, 0.5)) - radius)
                if d < ns * math.sqrt(3) / 2:
                    nxt.append((nx, ny, nz, ns))
        frontier = nxt
    return total


def surface_cells(depth, radius=0.42):
    """Count the dense-grid cells the sphere's surface actually passes through.

    This is the occupancy the prose quotes: the fraction of a dense grid that carries any information at all. It is
    counted cell by cell, not estimated from the surface area.
    """
    n = 2**depth
    s = 1.0 / n
    half = s * math.sqrt(3) / 2
    inner, outer = (radius - half) ** 2, (radius + half) ** 2

    def span(a, b):
        """How many cell centres have dz in the open interval (a, b)."""
        ka = math.ceil((a + 0.5) * n - 0.5)
        kb = math.floor((b + 0.5) * n - 0.5)
        return max(0, min(kb, n - 1) - max(ka, 0) + 1)

    hit = 0
    for i in range(n):
        dx = (i + 0.5) * s - 0.5
        for j in range(n):
            dy = (j + 0.5) * s - 0.5
            rho = dx * dx + dy * dy
            if rho >= outer:
                continue  # no k in this column can reach the shell
            hi = math.sqrt(outer - rho)
            lo = math.sqrt(max(0.0, inner - rho))
            hit += span(lo, hi) + (span(-hi, -lo) if lo > 0 else 0)
    return hit


def voxel_cost_table(depths=(4, 5, 6, 7, 8, 9)):
    """Dense cells, octree nodes and occupied cells per resolution."""
    return [(2**d, (2**d) ** 3, octree_cells(d), surface_cells(d)) for d in depths]


# ----------------------------------------------------------------------------
# Chamfer and Earth Mover's matchings on one pair of point sets.
# ----------------------------------------------------------------------------

PRED = [(0.18, 0.62), (0.30, 0.74), (0.40, 0.55), (0.55, 0.70), (0.66, 0.44)]
TARG = [(0.14, 0.48), (0.34, 0.84), (0.46, 0.40), (0.62, 0.78), (0.90, 0.66)]


def chamfer_matching(a, b):
    """Nearest neighbour in b for every point of a, independently."""
    return [min(range(len(b)), key=lambda j: math.dist(p, b[j])) for p in a]


def chamfer_distance(a, b):
    """The symmetric squared nearest-neighbour sum of @eq-chamfer."""
    fwd = sum(min(math.dist(p, q) ** 2 for q in b) for p in a)
    bwd = sum(min(math.dist(p, q) ** 2 for p in a) for q in b)
    return fwd + bwd


def emd_matching(a, b):
    """Exact optimal bijection by brute-force search over permutations."""
    best, cost = None, float("inf")
    for perm in itertools.permutations(range(len(b))):
        c = sum(math.dist(a[i], b[perm[i]]) for i in range(len(a)))
        if c < cost:
            best, cost = perm, c
    return list(best), cost


# ----------------------------------------------------------------------------
# NeRF's discrete volume rendering sum.
# ----------------------------------------------------------------------------


def density_profile(t):
    """A soft-edged surface: a Gaussian bump of density around t = 0.55."""
    return 26.0 * math.exp(-(((t - 0.55) / 0.075) ** 2))


def composite(sigmas, deltas):
    """Return the alphas, transmittances and weights of @eq-volume-rendering."""
    alphas = [1.0 - math.exp(-s * d) for s, d in zip(sigmas, deltas)]
    trans, acc = [], 1.0
    for a in alphas:
        trans.append(acc)
        acc *= 1.0 - a
    weights = [t * a for t, a in zip(trans, alphas)]
    return alphas, trans, weights, acc


def ray_samples(n=140, near=0.0, far=1.0):
    """Uniform samples along one ray, with the density profile evaluated."""
    d = (far - near) / n
    ts = [near + (i + 0.5) * d for i in range(n)]
    return ts, [density_profile(t) for t in ts], [d] * n


def partition_residual(n=140):
    """How far sum(w_i) + T_{N+1} strays from one. Should be machine noise."""
    _, sig, dlt = ray_samples(n)
    _, _, w, tail = composite(sig, dlt)
    return abs(sum(w) + tail - 1.0)


def slab_convergence(sigma=2.5, length=1.0, counts=(1, 2, 4, 16, 64, 256, 4096)):
    """Discrete opacity of a constant-density slab against its closed form."""
    exact = 1.0 - math.exp(-sigma * length)
    rows = []
    for n in counts:
        d = length / n
        _, _, w, _ = composite([sigma] * n, [d] * n)
        rows.append((n, sum(w), abs(sum(w) - exact)))
    return exact, rows


# ----------------------------------------------------------------------------
# Published numbers. Stated in sentences or tables of the primary sources.
# ----------------------------------------------------------------------------

# (label, year, number of shapes or scenes)
CORPORA = [
    ("Princeton SB", 2004, 1_814),
    ("ShapeNetCore", 2015, 51_300),
    ("ScanNet", 2017, 1_513),
    ("PartNet", 2019, 26_671),
    ("CO3D", 2021, 18_619),
    ("Objaverse", 2022, 818_000),
    ("Objaverse-XL", 2023, 10_200_000),
]
IMAGENET = ("ImageNet", 2009, 14_197_122)

# (label, frames per second, PSNR, training time) from the 3DGS teaser figure,
# which states all four for every method.
SPLAT_METHODS = [
    ("Mip-NeRF360", 0.071, 24.3, "48 h"),
    ("Plenoxels", 8.2, 21.9, "26 min"),
    ("InstantNGP", 9.2, 22.1, "7 min"),
    ("3DGS, 7K", 135.0, 23.6, "6 min"),
    ("3DGS, 30K", 93.0, 25.2, "51 min"),
]

# MVCNN on ModelNet40: classification accuracy and retrieval mAP.
MVCNN_ROWS = [
    ("3D ShapeNets (voxels)", 77.3, 49.2),
    ("MVCNN, 12 views", 89.9, 70.1),
    ("MVCNN, 80 views", 90.1, 70.4),
]


# ----------------------------------------------------------------------------
# Figures.
# ----------------------------------------------------------------------------


def fig_bezier():
    """Bernstein weights beside the de Casteljau construction."""
    w, h = 760, 330
    b = ""
    # Left panel: the four Bernstein basis functions.
    lx = mapper((0, 1), (0, 1), (66, 340), (60, 250))
    b += line(66, 250, 340, 250, op=0.4)
    b += line(66, 250, 66, 60, op=0.4)
    for v, lab in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
        px, py = lx(v, 0)
        b += line(px, py, px, py + 4, op=0.4)
        b += text(px, py + 17, lab, size=10, op=0.55)
    b += text(203, 285, "t", size=11, style="italic", op=0.7)
    b += text(50, 155, "weight", size=10, op=0.55, anchor="middle")
    cols = [PURPLE, INDIGO, TEAL, GREEN]
    for i in range(4):
        pts = []
        for k in range(121):
            t = k / 120
            wt = math.comb(3, i) * (1 - t) ** (3 - i) * t**i
            pts.append(lx(t, wt))
        b += polyline(pts, stroke=cols[i], sw=1.8)
        tx, _ = lx(0.5, math.comb(3, i) * 0.5**3)
        b += text(tx + (i - 1.5) * 46, 46, f"B{'₀₁₂₃'[i]}", size=11, fill=cols[i], weight=600)
    b += text(203, 306, "the Bernstein basis, degree three", size=11, op=0.7)

    # Right panel: the curve, the control polygon, and the recursion at t.
    t0 = 0.62
    rx = mapper((0, 1), (-0.02, 1.0), (430, 720), (70, 250))
    curve = [rx(*bernstein(k / 200, CONTROL)) for k in range(201)]
    b += polyline(curve, stroke=INK, sw=2.2, op=0.85)
    levels = de_casteljau_levels(t0, CONTROL)
    shades = [0.30, 0.55, 0.85]
    for depth, lv in enumerate(levels[:-1]):
        px = [rx(*p) for p in lv]
        b += polyline(px, stroke=INDIGO, sw=1.2, op=shades[depth], dash="4 3")
        for q in px:
            b += circle(q[0], q[1], 3.0, fill=INDIGO, stroke="none", op=shades[depth])
    fx, fy = rx(*levels[-1][0])
    b += circle(fx, fy, 5.2, fill=PURPLE, stroke="none")
    b += text(fx, fy - 13, f"γ({t0})", size=11, fill=PURPLE, weight=600)
    for i, p in enumerate(CONTROL):
        qx, qy = rx(*p)
        b += text(qx, qy + (-12 if i % 2 == 0 else 20), f"b{'₀₁₂₃'[i]}", size=10, op=0.6)
    b += text(575, 306, "de Casteljau at t = 0.62", size=11, op=0.7)
    return wrap(
        w,
        h,
        "bez",
        "Bernstein weights and the de Casteljau construction",
        "Left, the four cubic Bernstein basis functions over t in [0,1]. Right, a "
        "cubic Bezier curve with its control polygon and the nested linear "
        "interpolations that reduce four control points to one point on the curve.",
        b,
    )


def fig_two_queries():
    """The sampling / inside-outside asymmetry, one panel per representation."""
    w, h = 760, 330
    b = ""
    for k, (cx, title, sub) in enumerate(
        (
            (196, "Explicit", "γ(u, v) → point"),
            (564, "Implicit", "f(x, y, z) = 0"),
        )
    ):
        b += text(cx, 34, title, size=14, weight=600)
        b += text(cx, 53, sub, size=11, font=MONO, op=0.65)
        b += circle(cx, 150, 62, stroke=INK, sw=2.0, op=0.85)
    # Left: sampling is a direct evaluation.
    for i in range(9):
        a = 2 * math.pi * (i / 9) + 0.2
        b += circle(196 + 62 * math.cos(a), 150 + 62 * math.sin(a), 4.2, fill=TEAL, stroke="none")
    _sx = 196 + 62 * math.cos(math.pi * 0.86)
    _sy = 150 + 62 * math.sin(math.pi * 0.86)
    b += arrow(106, 80, _sx - 5, _sy - 7, stroke=TEAL, sw=1.5, op=0.9)
    b += text(84, 68, "sample: evaluate", size=11, fill=TEAL, anchor="start")
    qx, qy = 196 + 34, 150 - 22
    b += circle(qx, qy, 4.4, fill=PURPLE, stroke="none")
    b += arrow(qx, qy, qx + 96, qy - 46, stroke=PURPLE, sw=1.3, op=0.8, dash="4 3")
    b += arrow(qx, qy, qx + 96, qy + 6, stroke=PURPLE, sw=1.3, op=0.8, dash="4 3")
    b += text(214, 268, "inside? cast a ray, count crossings", size=11, fill=PURPLE)
    # Right: containment is a direct evaluation.
    b += circle(564 + 30, 150 - 20, 4.4, fill=PURPLE, stroke="none")
    b += text(660, 118, "f &lt; 0 → inside", size=11, fill=PURPLE, anchor="start")
    b += arrow(596, 128, 652, 116, stroke=PURPLE, sw=1.5, op=0.9)
    for i in range(7):
        a = 2 * math.pi * (i / 7) + 0.5
        px, py = 564 + 62 * math.cos(a), 150 + 62 * math.sin(a)
        b += circle(px, py, 4.2, fill="none", stroke=TEAL, sw=1.4, op=0.8)
    b += text(564, 268, "on the surface? solve f = 0", size=11, fill=TEAL)
    b += line(380, 44, 380, 292, op=0.25, dash="5 5")
    b += text(380, 312, "each representation is the other's complement", size=11, op=0.6)
    return wrap(
        w,
        h,
        "twoq",
        "Sampling and containment across explicit and implicit surfaces",
        "Two panels showing the same circular object. On the explicit side "
        "sampling points is one evaluation and testing containment needs a ray "
        "cast; on the implicit side the two costs are exchanged.",
        b,
    )


def fig_sdf_union():
    """Distance-field contours for two disks and their union."""
    w, h = 760, 360
    b = ""
    scale = 52.0

    def to_px(cx, x, y):
        return cx + x * scale, 190 - y * scale

    for cx, title, use_true in (
        (200, "min(d₁, d₂)", False),
        (556, "true distance to ∂(A ∪ B)", True),
    ):
        b += text(cx, 34, title, size=13, weight=600)
        # Level sets, drawn by marching a coarse grid and joining sign changes.
        for lvl, op in ((0.4, 0.30), (0.8, 0.22), (-0.3, 0.30)):
            for disk in DISKS if not use_true else [None]:
                pass
            pts = []
            n = 240
            for k in range(n + 1):
                a = 2 * math.pi * k / n
                # Ray-march outward from the origin for this level set.
                lo, hi = 0.0, 3.2
                fn = (lambda x, y: true_union_sdf(x, y, samples=720)) if use_true else composed_sdf
                for _ in range(28):
                    mid = 0.5 * (lo + hi)
                    if fn(mid * math.cos(a), mid * math.sin(a)) < lvl:
                        lo = mid
                    else:
                        hi = mid
                r = 0.5 * (lo + hi)
                pts.append(to_px(cx, r * math.cos(a), r * math.sin(a)))
            col = YELLOW if lvl < 0 else INDIGO
            if lvl < 0:
                b += poly(pts, fill=YELLOW, stroke="none", op=0.22)
            else:
                b += polyline(pts, stroke=col, sw=1.1, op=op, dash="4 3")
        # The surface itself.
        pts = []
        n = 320
        for k in range(n + 1):
            a = 2 * math.pi * k / n
            lo, hi = 0.0, 3.2
            fn = (lambda x, y: true_union_sdf(x, y, samples=720)) if use_true else composed_sdf
            for _ in range(28):
                mid = 0.5 * (lo + hi)
                if fn(mid * math.cos(a), mid * math.sin(a)) < 0.0:
                    lo = mid
                else:
                    hi = mid
            r = 0.5 * (lo + hi)
            pts.append(to_px(cx, r * math.cos(a), r * math.sin(a)))
        b += polyline(pts, stroke=INK, sw=2.0, op=0.9)
        ox, oy = to_px(cx, 0, 0)
        b += circle(ox, oy, 3.6, fill=PURPLE, stroke="none")

    # The disagreement at the centre, annotated once, between the panels.
    lx, ly = to_px(200, 0, 0)
    b += line(lx, ly, lx, ly - 0.5 * scale, stroke=PURPLE, sw=2.0, op=0.9)
    b += text(lx + 8, ly - 16, "0.500", size=11, fill=PURPLE, anchor="start")
    rxp, ryp = to_px(556, 0, 0)
    b += line(rxp, ryp, rxp, ryp - (math.sqrt(3) / 2) * scale, stroke=PURPLE, sw=2.0)
    b += text(rxp + 8, ryp - 28, "0.866", size=11, fill=PURPLE, anchor="start")
    b += text(
        378,
        336,
        "the two fields agree outside and differ inside, most at the centre",
        size=11,
        op=0.62,
    )
    return wrap(
        w,
        h,
        "sdfu",
        "Composed and true signed distance fields for a union of two disks",
        "Two panels of the same union of two overlapping disks. The composed "
        "field takes the pointwise minimum of the two disk fields; the true "
        "field measures distance to the union's actual boundary. They coincide "
        "outside and differ inside, by the largest amount at the centre.",
        b,
    )


def fig_grid():
    """The explicit/implicit by non-parametric/parametric taxonomy."""
    w, h = 760, 400
    b = ""
    cx = [258, 566]
    cy = [140, 268]
    b += text(258, 44, "Explicit", size=14, weight=600)
    b += text(566, 44, "Implicit", size=14, weight=600)
    b += text(96, 145, "Non-parametric", size=12, op=0.8)
    b += text(96, 273, "Parametric", size=12, op=0.8)
    b += line(412, 58, 412, 330, op=0.25)
    b += line(140, 204, 700, 204, op=0.25)
    cells = [
        (0, 0, ["Points", "Meshes"]),
        (1, 0, ["Voxels", "Level sets"]),
        (0, 1, ["Splines", "Subdivision"]),
        (1, 1, ["Algebraic", "CSG"]),
    ]
    for col, row, names in cells:
        for k, nm in enumerate(names):
            b += text(cx[col] + (k - 0.5) * 132, cy[row], nm, size=13)
    # Each arrow runs from the cell it leaves to the cell it lands in. The
    # last one is a long diagonal on purpose: splats are explicit and
    # non-parametric, so the tour returns to the cell beside where it started.
    moves = [
        (498, 112, 210, 112, "voxels → points", TEAL, (354, 100)),
        (258, 154, 258, 250, "points → surfaces", INDIGO, (176, 192)),
        (368, 300, 566, 300, "surfaces → fields", PURPLE, (467, 318)),
        (622, 250, 226, 130, "fields → splats", GREEN, (470, 232)),
    ]
    for x0, y0, x1, y1, lab, col, (lx, ly) in moves:
        dash = "6 4" if lab.startswith("fields") else None
        b += arrow(x0, y0, x1, y1, stroke=col, sw=1.6, op=0.9, head=7, dash=dash)
        b += text(lx, ly, lab, size=11, fill=col, weight=600)
    b += text(
        380,
        366,
        "four moves, four different dominant costs; the tour ends beside where it began",
        size=11,
        op=0.62,
    )
    return wrap(
        w,
        h,
        "grid",
        "Taxonomy of geometry representations with the deep-learning tour",
        "A two-by-two grid whose columns are explicit and implicit and whose "
        "rows are non-parametric and parametric. Four labelled arrows trace the "
        "moves the deep-learning literature made between the cells.",
        b,
    )


def fig_data_scale():
    """3D corpus sizes on a log axis against ImageNet."""
    w, h = 760, 400
    b = ""
    to_px = mapper((2002.5, 2025.0), (3.0, 7.4), (92, 700), (60, 300))
    for e in range(3, 8):
        _, py = to_px(2002.5, e)
        b += line(92, py, 700, py, op=0.14)
        lab = ["1K", "10K", "100K", "1M", "10M"][e - 3]
        b += text(80, py + 4, lab, size=10, op=0.55, anchor="end")
    for yr in (2005, 2010, 2015, 2020, 2025):
        px, _ = to_px(yr, 3.0)
        b += line(px, 300, px, 305, op=0.4)
        b += text(px, 322, str(yr), size=10, op=0.55)
    b += line(92, 300, 700, 300, op=0.4)
    b += text(396, 344, "year of release", size=11, op=0.65)
    b += text(48, 172, "instances", size=10, op=0.55)

    _, iy = to_px(IMAGENET[1], math.log10(IMAGENET[2]))
    b += line(92, iy, 700, iy, stroke=INK, sw=1.3, op=0.5, dash="6 4")
    b += text(696, iy - 9, "ImageNet, 14.2M images", size=11, op=0.62, anchor="end")

    synth = [c for c in CORPORA if c[0] in ("Princeton SB", "ShapeNetCore", "Objaverse", "Objaverse-XL")]
    pts = [to_px(y, math.log10(n)) for _, y, n in synth]
    b += polyline(pts, stroke=INDIGO, sw=2.0, op=0.55)
    offsets = {
        "Princeton SB": (0, -18),
        "ShapeNetCore": (0, -18),
        "ScanNet": (4, 22),
        "PartNet": (0, 22),
        "CO3D": (0, -18),
        "Objaverse": (-6, -18),
        "Objaverse-XL": (-16, 22),
    }
    for name, yr, n in CORPORA:
        px, py = to_px(yr, math.log10(n))
        scanned = name in ("ScanNet", "CO3D")
        col = TEAL if scanned else INDIGO
        b += circle(px, py, 5.4, fill=col, stroke="none")
        dx, dy = offsets[name]
        b += text(px + dx, py + dy, name, size=11, fill=col, weight=600)
    b += circle(112, 372, 5.4, fill=INDIGO, stroke="none")
    b += text(124, 376, "synthetic assets", size=11, op=0.7, anchor="start")
    b += circle(266, 372, 5.4, fill=TEAL, stroke="none")
    b += text(278, 376, "real scans (scenes, videos)", size=11, op=0.7, anchor="start")
    return wrap(
        w,
        h,
        "dscale",
        "Sizes of 3D datasets by year against ImageNet",
        "A logarithmic scatter of 3D corpus sizes from 2004 to 2023, "
        "distinguishing synthetic asset collections from real scans, with "
        "ImageNet drawn as a horizontal reference line above all of them.",
        b,
    )


def fig_multiview():
    """The multi-view CNN pipeline, ordered left to right along the flow."""
    w, h = 760, 340
    b = ""
    b += text(74, 44, "rendered views", size=11, op=0.7)
    for i in range(4):
        y = 74 + i * 52
        b += rect(46, y, 56, 40, fill=INDIGO, op=0.14, stroke=INDIGO, so=0.5, rx=3)
        b += rect(140, y, 62, 40, fill="none", stroke=INK, so=0.6, rx=3)
        b += text(171, y + 25, "CNN₁", size=11, op=0.85)
        b += arrow(106, y + 20, 136, y + 20, sw=1.2, op=0.6)
        b += arrow(206, y + 20, 268, y + 20, sw=1.2, op=0.6)
    b += text(171, 62, "shared weights", size=10, op=0.55)
    b += rect(272, 74, 62, 196, fill=YELLOW, op=0.28, stroke=TEAL, so=0.7, rx=3)
    b += text(303, 165, "max", size=12, fill=TEAL, weight=600)
    b += text(303, 182, "over views", size=10, fill=TEAL)
    b += text(303, 296, "view pooling", size=11, fill=TEAL, weight=600)
    b += arrow(338, 172, 400, 172, sw=1.4, op=0.75)
    b += rect(404, 152, 68, 40, fill="none", stroke=INK, so=0.6, rx=3)
    b += text(438, 177, "CNN₂", size=11, op=0.85)
    b += arrow(476, 172, 530, 172, sw=1.4, op=0.75)
    b += rect(534, 152, 76, 40, fill=PURPLE, op=0.14, stroke=PURPLE, so=0.5, rx=3)
    b += text(572, 177, "softmax", size=11, fill=PURPLE)
    b += text(438, 132, "one descriptor per shape", size=10, op=0.6)

    b += text(680, 96, "ModelNet40", size=11, weight=600, op=0.8, anchor="end")
    for k, (lab, acc, _) in enumerate(MVCNN_ROWS):
        col = PURPLE if k == 0 else TEAL
        b += text(624, 122 + k * 30, lab, size=10, fill=col, anchor="start")
        b += text(716, 122 + k * 30 + 14, f"{acc:.1f}%", size=13, fill=col, weight=600, anchor="end")
    b += text(380, 322, "the maximum is what makes the descriptor order-independent", size=11, op=0.62)
    return wrap(
        w,
        h,
        "mvcnn",
        "The multi-view CNN pipeline",
        "Four rendered views each pass through the same convolutional network, "
        "an element-wise maximum pools them into one descriptor, and a second "
        "network classifies it. ModelNet40 accuracies are listed beside it.",
        b,
    )


def fig_voxel_cost():
    """Dense against octree cell counts for the same sphere."""
    w, h = 760, 360
    b = ""
    rows = voxel_cost_table()
    to_px = mapper((3.6, 9.4), (2.6, 8.4), (100, 470), (58, 262))
    for e in range(3, 9):
        _, py = to_px(3.6, e)
        b += line(100, py, 470, py, op=0.14)
        b += text(88, py + 4, f"10{'⁰¹²³⁴⁵⁶⁷⁸'[e]}", size=10, op=0.55, anchor="end")
    for d, n in ((4, 16), (6, 64), (8, 256), (9, 512)):
        px, _ = to_px(d, 1.6)
        b += line(px, 262, px, 267, op=0.4)
        b += text(px, 284, f"{n}³", size=10, op=0.55)
    b += line(100, 262, 470, 262, op=0.4)
    b += text(285, 306, "output resolution", size=11, op=0.65)
    b += text(58, 160, "cells", size=10, op=0.55)

    dense = [to_px(math.log2(n), math.log10(d)) for n, d, _, _ in rows]
    tree = [to_px(math.log2(n), math.log10(t)) for n, _, t, _ in rows]
    b += polyline(dense, stroke=PURPLE, sw=2.2)
    b += polyline(tree, stroke=TEAL, sw=2.2)
    for p in dense:
        b += circle(p[0], p[1], 4.0, fill=PURPLE, stroke="none")
    for p in tree:
        b += circle(p[0], p[1], 4.0, fill=TEAL, stroke="none")
    b += text(dense[-1][0] - 6, dense[-1][1] - 14, "dense, O(n³)", size=11, fill=PURPLE, weight=600, anchor="end")
    b += text(tree[3][0] + 4, tree[3][1] + 26, "octree, O(n²)", size=11, fill=TEAL, weight=600, anchor="start")

    b += text(624, 58, "occupied fraction", size=11, weight=600, op=0.8)
    for k, (n, dn, _, occ) in enumerate(rows):
        y = 84 + k * 24
        b += text(556, y, f"{n}³", size=11, op=0.7, anchor="start")
        b += text(700, y, f"{100 * occ / dn:.2f}%", size=12, fill=INDIGO, anchor="end")
    b += text(628, 262, "a surface touches ever less", size=10, op=0.6)
    b += text(628, 278, "of the volume it sits in", size=10, op=0.6)
    return wrap(
        w,
        h,
        "vcost",
        "Dense grid and octree cell counts against resolution",
        "A logarithmic plot of how many cells a dense voxel grid and a "
        "surface-following octree need for the same sphere at resolutions from "
        "16 cubed to 256 cubed, with the occupied fraction tabulated beside it.",
        b,
    )


def fig_pointnet():
    """Per-point MLP, symmetric pooling, global descriptor."""
    w, h = 760, 330
    b = ""
    b += text(80, 46, "unordered points", size=11, op=0.7)
    coords = ["(1, 2, 3)", "(1, 1, 1)", "(2, 3, 2)", "(2, 3, 4)"]
    for i, c in enumerate(coords):
        y = 78 + i * 46
        b += text(84, y + 16, c, size=11, font=MONO, op=0.8)
        b += rect(150, y, 58, 30, fill="none", stroke=INK, so=0.6, rx=3)
        b += text(179, y + 20, "h", size=12, style="italic", op=0.85)
        b += arrow(128, y + 15, 146, y + 15, sw=1.2, op=0.55)
        b += arrow(212, y + 15, 268, y + 15, sw=1.2, op=0.6)
    b += text(179, 62, "shared MLP", size=10, op=0.55)
    b += rect(272, 78, 58, 168, fill=YELLOW, op=0.30, stroke=TEAL, so=0.7, rx=3)
    b += text(301, 158, "max", size=13, fill=TEAL, weight=600)
    b += text(301, 176, "per dim", size=10, fill=TEAL)
    b += text(301, 272, "symmetric", size=11, fill=TEAL, weight=600)
    b += arrow(334, 162, 392, 162, sw=1.4, op=0.75)
    b += rect(396, 142, 58, 40, fill="none", stroke=INK, so=0.6, rx=3)
    b += text(425, 167, "γ", size=13, style="italic", op=0.85)
    b += arrow(458, 162, 516, 162, sw=1.4, op=0.75)
    b += rect(520, 142, 92, 40, fill=PURPLE, op=0.14, stroke=PURPLE, so=0.5, rx=3)
    b += text(566, 167, "class scores", size=11, fill=PURPLE)
    b += text(646, 116, "any order in,", size=11, op=0.65, anchor="start")
    b += text(646, 134, "same answer out", size=11, op=0.65, anchor="start")
    b += text(
        380, 308, "the lift comes first: a maximum over raw coordinates returns only a bounding box", size=11, op=0.62
    )
    return wrap(
        w,
        h,
        "pnet",
        "The PointNet architecture",
        "Four unordered points each pass through the same multilayer "
        "perceptron, an element-wise maximum reduces the resulting features to "
        "one global descriptor, and a final perceptron produces class scores.",
        b,
    )


def fig_chamfer_emd():
    """Chamfer's nearest neighbours beside the Earth Mover's bijection."""
    w, h = 760, 340
    b = ""
    cm = chamfer_matching(PRED, TARG)
    perm, _ = emd_matching(PRED, TARG)
    claimed = set(perm)
    for panel, (cx, title, match) in enumerate(
        ((58, "Chamfer: independent nearest neighbours", cm), (416, "Earth Mover's: an optimal bijection", perm))
    ):
        to_px = mapper((0.05, 1.0), (0.32, 0.92), (cx + 24, cx + 296), (72, 250))
        b += text(cx + 160, 42, title, size=12, weight=600)
        used = [match.count(j) for j in range(len(TARG))]
        for i, p in enumerate(PRED):
            px, py = to_px(*p)
            qx, qy = to_px(*TARG[match[i]])
            b += arrow(px, py, qx, qy, stroke=INDIGO, sw=1.3, op=0.75, head=5.5)
        for j, q in enumerate(TARG):
            qx, qy = to_px(*q)
            hit = used[j] > 0 if panel == 0 else j in claimed
            if panel == 0 and used[j] > 1:
                b += circle(qx, qy, 10.5, fill="none", stroke=PURPLE, sw=1.4, so=0.85)
            b += circle(qx, qy, 6.0, fill=TEAL if hit else "none", stroke=TEAL, sw=1.6)
        for p in PRED:
            px, py = to_px(*p)
            b += circle(px, py, 5.0, fill=PURPLE, stroke="none")
        if panel == 0:
            dup = [j for j in range(len(TARG)) if used[j] > 1]
            miss = [j for j in range(len(TARG)) if used[j] == 0]
            note = []
            if dup:
                note.append(f"{len(dup)} target claimed twice")
            if miss:
                note.append(f"{len(miss)} target unmatched")
            b += text(cx + 160, 282, "; ".join(note), size=11, fill=PURPLE, op=0.85)
        else:
            b += text(cx + 160, 282, "every target claimed exactly once", size=11, fill=TEAL, op=0.85)
    b += circle(214, 314, 5.0, fill=PURPLE, stroke="none")
    b += text(226, 318, "prediction", size=11, op=0.7, anchor="start")
    b += circle(340, 314, 6.0, fill="none", stroke=TEAL, sw=1.6)
    b += text(354, 318, "target", size=11, op=0.7, anchor="start")
    b += line(380, 56, 380, 268, op=0.22, dash="5 5")
    return wrap(
        w,
        h,
        "cemd",
        "Chamfer and Earth Mover's matchings on the same point sets",
        "Two panels showing the same five predicted and five target points. "
        "Chamfer's arrows are independent nearest-neighbour lookups and can "
        "collide; the Earth Mover's arrows form a bijection.",
        b,
    )


def fig_atlas():
    """The unit square, the learned map, and the surface patch it lands on."""
    w, h = 760, 320
    b = ""
    b += text(140, 42, "parameter domain", size=12, weight=600)
    for k, col in enumerate((INDIGO, TEAL, PURPLE)):
        ox, oy = 74 + k * 6, 96 + k * 6
        b += rect(ox, oy, 110, 110, fill=col, op=0.10, stroke=col, so=0.55, rx=2)
    for i in range(1, 5):
        b += line(86 + i * 22, 108, 86 + i * 22, 218, op=0.18)
        b += line(86, 108 + i * 22, 196, 108 + i * 22, op=0.18)
    b += rect(86, 108, 110, 110, fill="none", stroke=PURPLE, so=0.8, sw=1.5)
    b += text(141, 244, "(u, v) ∈ [0,1]²", size=11, font=MONO, op=0.75)
    b += text(141, 264, "K squares, one per patch", size=10, op=0.6)

    b += arrow(214, 162, 292, 162, sw=1.5, op=0.8)
    b += rect(296, 128, 104, 68, fill="none", stroke=INK, so=0.6, rx=3)
    b += text(348, 156, "MLP", size=13, op=0.85)
    b += text(348, 178, "(z, u, v)", size=11, font=MONO, op=0.7)
    b += text(348, 96, "shape code z", size=10, op=0.6)
    b += arrow(348, 104, 348, 124, sw=1.2, op=0.55)
    b += arrow(404, 162, 480, 162, sw=1.5, op=0.8)

    b += text(596, 42, "image in ℝ³", size=12, weight=600)

    # A curved patch: the grid carried through a smooth map.
    def bend(u, v):
        x = 500 + 172 * u + 26 * v
        y = 210 - 96 * v - 34 * math.sin(math.pi * u) - 10 * v * v
        return x, y

    for i in range(6):
        b += polyline([bend(i / 5, k / 40) for k in range(41)], stroke=PURPLE, sw=1.0, op=0.35)
        b += polyline([bend(k / 40, i / 5) for k in range(41)], stroke=PURPLE, sw=1.0, op=0.35)
    b += polyline(
        [bend(k / 40, 0) for k in range(41)]
        + [bend(1, k / 40) for k in range(41)]
        + [bend(1 - k / 40, 1) for k in range(41)]
        + [bend(0, 1 - k / 40) for k in range(41)],
        stroke=PURPLE,
        sw=1.8,
        op=0.9,
    )
    b += text(596, 264, "adjacency survives the map, so this is a mesh", size=11, op=0.65)
    b += text(380, 300, "sample as densely at inference as you like: the map has no resolution", size=11, op=0.62)
    return wrap(
        w,
        h,
        "atlas",
        "The AtlasNet parametric decoder",
        "A gridded unit square on the left, a multilayer perceptron taking a "
        "shape code and a square coordinate in the middle, and the curved image "
        "of that grid in three dimensions on the right.",
        b,
    )


def fig_volume_rendering():
    """Density, transmittance and compositing weights along one ray."""
    w, h = 760, 380
    b = ""
    ts, sig, dlt = ray_samples(140)
    _, trans, weights, tail = composite(sig, dlt)

    to_px = mapper((0.0, 1.0), (0.0, 1.05), (100, 636), (62, 268))
    b += line(100, 268, 660, 268, op=0.4)
    b += line(100, 268, 100, 56, op=0.4)
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        px, py = to_px(v, 0)
        b += line(px, py, px, py + 4, op=0.4)
        b += text(px, py + 18, f"{v:g}", size=10, op=0.55)
    for v in (0.0, 0.5, 1.0):
        _, py = to_px(0, v)
        b += text(90, py + 4, f"{v:g}", size=10, op=0.55, anchor="end")
    b += text(368, 306, "distance along the ray, t", size=11, op=0.65)

    smax = max(sig)
    b += polyline([to_px(t, s / smax) for t, s in zip(ts, sig)], stroke=INDIGO, sw=1.6, op=0.55, dash="5 3")
    b += polyline([to_px(t, v) for t, v in zip(ts, trans)], stroke=TEAL, sw=2.2)
    wmax = max(weights)
    b += polyline([to_px(t, v / wmax) for t, v in zip(ts, weights)], stroke=PURPLE, sw=2.2)

    dpx, dpy = to_px(0.55, 1.0)
    b += text(dpx + 84, dpy + 6, "density σ (rescaled)", size=11, fill=INDIGO, weight=600)
    tpx, tpy = to_px(0.40, 0.98)
    b += text(tpx - 66, tpy + 26, "transmittance T", size=11, fill=TEAL, weight=600)
    wpx, wpy = to_px(0.50, 1.0)
    b += text(wpx - 92, wpy - 8, "weights w = T·α", size=11, fill=PURPLE, weight=600)

    px, py = to_px(1.0, tail)
    b += circle(px, py, 4.4, fill=TEAL, stroke="none")
    b += text(px + 8, py + 4, f"T꜀ = {tail:.4f}", size=11, fill=TEAL, anchor="start")

    b += rect(148, 326, 466, 34, fill=YELLOW, op=0.22, stroke="none", rx=4)
    b += text(
        381,
        348,
        f"Σ wᵢ + T꜀ = {sum(weights):.6f} + {tail:.6f} = {sum(weights) + tail:.9f}",
        size=12,
        font=MONO,
        op=0.85,
    )
    return wrap(
        w,
        h,
        "volr",
        "Volume rendering quantities along one ray",
        "Three curves against distance along a ray: a rescaled density profile "
        "peaking at a soft surface, a transmittance that decays monotonically "
        "through it, and the compositing weights that peak where the two "
        "cross. The weights and the surviving transmittance sum to one.",
        b,
    )


def fig_nerf_vs_splat():
    """Dense sampling along a ray against sparse projected primitives."""
    w, h = 760, 340
    b = ""
    for panel, (cx, title, sub) in enumerate(
        (
            (56, "NeRF: dense everywhere", "one network call per sample"),
            (416, "Splatting: sparse where it matters", "one projection per primitive"),
        )
    ):
        b += text(cx + 152, 42, title, size=13, weight=600)
        b += text(cx + 152, 62, sub, size=11, op=0.6)
        camx, camy = cx + 36, 168
        b += poly(
            [(camx - 14, camy - 12), (camx + 8, camy - 20), (camx + 8, camy + 20), (camx - 14, camy + 12)],
            fill=INK,
            stroke="none",
            op=0.55,
        )
        b += text(camx - 4, camy + 42, "camera", size=10, op=0.55)
        # The object: a soft blob to the right of centre.
        blob = [(cx + 214, 150, 30, 20), (cx + 246, 186, 24, 26), (cx + 190, 196, 20, 16)]
        for bx, by, brx, bry in blob:
            b += (
                f'  <ellipse cx="{bx:.1f}" cy="{by:.1f}" rx="{brx:.1f}" ry="{bry:.1f}"'
                f' fill="{YELLOW}" fill-opacity="0.30" stroke="{GREEN}"'
                f' stroke-width="1.3" stroke-opacity="0.75"/>\n'
            )
        for k in range(3):
            ay = 128 + k * 40
            b += line(camx + 10, camy, cx + 316, ay, op=0.3)
            if panel == 0:
                for s in range(20):
                    f = 0.10 + 0.88 * s / 19
                    sx = camx + 10 + f * (cx + 316 - camx - 10)
                    sy = camy + f * (ay - camy)
                    inside = any(((sx - bx) / brx) ** 2 + ((sy - by) / bry) ** 2 < 1.0 for bx, by, brx, bry in blob)
                    b += circle(
                        sx, sy, 2.4, fill=PURPLE if inside else INDIGO, stroke="none", op=0.95 if inside else 0.34
                    )
        if panel == 1:
            for bx, by, brx, bry in blob:
                b += circle(bx, by, 2.8, fill=PURPLE, stroke="none")
                b += arrow(bx - brx - 2, by, camx + 26, camy + (by - camy) * 0.42, stroke=GREEN, sw=1.1, op=0.55, head=5)
        b += text(
            cx + 152,
            286,
            "empty samples: most of them" if panel == 0 else "empty space costs nothing",
            size=11,
            fill=INDIGO if panel == 0 else GREEN,
            op=0.9,
        )
    b += line(380, 56, 380, 274, op=0.22, dash="5 5")
    b += text(380, 318, "the same argument that took voxels to octrees, applied to compute", size=11, op=0.62)
    return wrap(
        w,
        h,
        "nvs",
        "Dense ray sampling against sparse primitive projection",
        "Two panels with the same camera and the same object. On the left, "
        "samples are spread evenly along each ray and most fall in empty space; "
        "on the right, only the primitives themselves are projected.",
        b,
    )


def fig_splat_tradeoff():
    """Rendering speed against reconstruction quality, with training times."""
    w, h = 760, 380
    b = ""
    to_px = mapper((-1.35, 2.4), (21.3, 25.8), (110, 664), (62, 282))
    for e, lab in ((-1, "0.1"), (0, "1"), (1, "10"), (2, "100")):
        px, _ = to_px(e, 21.3)
        b += line(px, 282, px, 287, op=0.4)
        b += text(px, 304, lab, size=10, op=0.55)
    b += line(110, 282, 664, 282, op=0.4)
    for q in (22, 23, 24, 25):
        _, py = to_px(-1.35, q)
        b += line(110, py, 664, py, op=0.14)
        b += text(98, py + 4, str(q), size=10, op=0.55, anchor="end")
    b += text(387, 326, "frames per second, logarithmic", size=11, op=0.65)
    b += text(58, 172, "PSNR", size=10, op=0.55)
    b += text(58, 188, "(dB)", size=10, op=0.55)

    offsets = {
        "Mip-NeRF360": (0, -20),
        "Plenoxels": (-52, -20),
        "InstantNGP": (44, -20),
        "3DGS, 7K": (0, 24),
        "3DGS, 30K": (0, -20),
    }
    for name, fps, psnr, train in SPLAT_METHODS:
        px, py = to_px(math.log10(fps), psnr)
        ours = name.startswith("3DGS")
        col = PURPLE if ours else INDIGO
        b += circle(px, py, 6.4 if ours else 5.2, fill=col, stroke="none")
        dx, dy = offsets[name]
        b += text(px + dx, py + dy, name, size=11, fill=col, weight=600 if ours else None)
        b += text(px + dx, py + dy + (-15 if dy < 0 else 15), train, size=10, fill=col, op=0.7)

    ax, ay = to_px(math.log10(0.071), 24.3)
    bx, by = to_px(math.log10(93.0), 25.2)
    b += arrow(ax + 14, ay - 2, bx - 16, by + 6, stroke=TEAL, sw=1.5, op=0.8, head=7)
    b += text((ax + bx) / 2 + 30, (ay + by) / 2 - 14, "×1,310 faster, +0.9 dB", size=12, fill=TEAL, weight=600)
    b += text(387, 356, "the quality axis does not start at zero; the speed axis spans four decades", size=11, op=0.62)
    return wrap(
        w,
        h,
        "strade",
        "Rendering speed against quality for radiance-field methods",
        "A scatter of five methods with rendering speed on a logarithmic axis "
        "and PSNR on a truncated linear axis, each labelled with its training "
        "time. An arrow marks the gap between the slowest and the fastest.",
        b,
    )


FIGURES = {
    "diagram_bezier.svg": fig_bezier,
    "diagram_two_queries.svg": fig_two_queries,
    "diagram_sdf_union.svg": fig_sdf_union,
    "diagram_grid.svg": fig_grid,
    "diagram_data_scale.svg": fig_data_scale,
    "diagram_multiview.svg": fig_multiview,
    "diagram_voxel_cost.svg": fig_voxel_cost,
    "diagram_pointnet.svg": fig_pointnet,
    "diagram_chamfer_emd.svg": fig_chamfer_emd,
    "diagram_atlas.svg": fig_atlas,
    "diagram_volume_rendering.svg": fig_volume_rendering,
    "diagram_nerf_vs_splat.svg": fig_nerf_vs_splat,
    "diagram_splat_tradeoff.svg": fig_splat_tradeoff,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print(f"wrote {OUT / name}")

    print()
    print("Bézier: Bernstein form against de Casteljau's recursion.")
    gap = bezier_disagreement()
    print(f"  largest disagreement over 2001 values of t: {gap:.3e}")
    assert gap < 1e-12, "the two Bézier evaluations disagree"

    print()
    print("CSG union: min(d1, d2) against the true distance to the union boundary.")
    wi, wo = sdf_union_errors()
    print(f"  worst gap outside the union: {wo:.4f}")
    print(f"  worst gap inside the union:  {wi:.4f}")
    print(
        f"  at the centre: composed {composed_sdf(0, 0):.4f}"
        f"  true {true_union_sdf(0, 0, samples=20000):.4f}"
        f"  (-sqrt(3)/2 = {-math.sqrt(3) / 2:.4f})"
    )
    assert wo < 5e-3, "min should reproduce the true distance outside the union"
    assert wi > 0.3, "min should understate the depth inside the union"

    print()
    print("Voxel and octree cell counts for the same sphere:")
    vt = voxel_cost_table()
    for n, dense, tree, occ in vt:
        print(
            f"  {n:>4d}³   dense {dense:>10,d}   octree {tree:>9,d}   surface cells {occ:>7,d}"
            f"   occupied {100 * occ / dense:6.3f}%"
        )
    fr = [occ / dense for _, dense, _, occ in vt]
    print("  occupancy ratio as n doubles: " + ", ".join(f"{fr[i] / fr[i + 1]:.2f}" for i in range(len(fr) - 1)))
    assert all(fr[i] > fr[i + 1] for i in range(len(fr) - 1)), "occupancy must fall with n"

    print()
    print("Chamfer and Earth Mover's on the chapter's point sets:")
    cm = chamfer_matching(PRED, TARG)
    perm, cost = emd_matching(PRED, TARG)
    print(f"  Chamfer matching  {cm}   distance {chamfer_distance(PRED, TARG):.4f}")
    print(f"  EMD    matching  {perm}   distance {cost:.4f}")
    print(f"  Chamfer leaves {len([j for j in range(len(TARG)) if j not in cm])} target(s) unmatched")
    assert sorted(perm) == list(range(len(TARG))), "EMD must be a bijection"
    assert len(set(cm)) < len(TARG), "the Chamfer example should show a collision"

    print()
    print("Volume rendering: the weight partition of @eq-weight-partition.")
    for n in (20, 60, 140, 400):
        print(f"  N = {n:>4d}   |Σw + T꜀ - 1| = {partition_residual(n):.3e}")
    assert partition_residual(140) < 1e-12, "the weights do not partition unity"

    print()
    print("Constant-density slab: the discrete sum against 1 - exp(-σL).")
    exact, rows = slab_convergence()
    print(f"  closed form  {exact:.9f}")
    for n, got, err in rows:
        print(f"  N = {n:>5d}   sum {got:.9f}   error {err:.3e}")
    assert rows[-1][2] < 1e-6, "the discrete sum does not converge to the analytic value"

    print()
    print("Numbers the prose quotes:")
    print(f"  3DGS speedup over Mip-NeRF360: {93.0 / 0.071:,.0f}x  (+{25.2 - 24.3:.1f} dB)")
    print(f"  Objaverse-XL against ImageNet: {10_200_000 / IMAGENET[2]:.2f}x")
    print(f"  NeRF weights 5 MB against 15 GB of voxel grid: {15 * 1024 / 5:,.0f}x")
