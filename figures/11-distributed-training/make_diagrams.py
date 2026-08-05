"""Generate the nine SVG diagrams for the lecture 11 chapter.

Two of them replot published numbers: the bandwidth hierarchy from the Llama 3
paper's network description, and the BF16 MFU figures from its Table 4. Two are
computed — the checkpointing trade-off curve is evaluated from the recurrence in
`checkpoint_cost`, and the pipeline schedule is generated from the GPipe timing
rule in `gpipe_schedule` rather than drawn cell by cell, so the bubble fraction
the chapter quotes falls out of the same code that draws the figure. The rest
are geometry: the four axes a transformer offers, the all-reduce that makes data
parallelism work, the three-lane overlap of an FSDP backward pass, the two-axis
grid of hybrid sharding, and the column-then-row block multiply of tensor
parallelism.

Conventions follow figures/10-video-understanding: currentColor for text and
rules, viridis for data, title/desc with namespaced ids, no background rect (the
theme supplies a light plate in both schemes). Yellow is a fill only, never a
line or a label.

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
    line,
    polyline,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path("figures/11-distributed-training")

# One colour per device index, used wherever a figure has to distinguish four
# GPUs or four microbatches. Yellow is absent on purpose: these are read as
# labels as often as fills.
DEVICE = [PURPLE, INDIGO, TEAL, GREEN]


def cuboid(x, y, w, h, dx, dy, stroke=INK, fill="none", op=0.10):
    """A box drawn in oblique projection; (dx, dy) is the depth offset."""
    body = rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.2, op=op)
    top = (
        f'  <polygon points="{x:.1f},{y:.1f} {x + dx:.1f},{y + dy:.1f}'
        f' {x + w + dx:.1f},{y + dy:.1f} {x + w:.1f},{y:.1f}"'
        f' fill="{stroke}" fill-opacity="{op * 0.6:.3f}" stroke="{stroke}"'
        ' stroke-width="1.2"/>\n'
    )
    side = (
        f'  <polygon points="{x + w:.1f},{y:.1f} {x + w + dx:.1f},{y + dy:.1f}'
        f' {x + w + dx:.1f},{y + h + dy:.1f} {x + w:.1f},{y + h:.1f}"'
        f' fill="{stroke}" fill-opacity="{op * 0.3:.3f}" stroke="{stroke}"'
        ' stroke-width="1.2"/>\n'
    )
    return body + top + side


def box(x, y, w, h, label, sub=None, colour=INK, op=0.10, size=13):
    """A rounded labelled block, optionally with a smaller second line."""
    out = rect(x, y, w, h, fill=colour, stroke=colour, sw=1.2, op=op, rx=4)
    if sub:
        out += text(x + w / 2, y + h / 2 - 1, label, size=size)
        out += text(x + w / 2, y + h / 2 + 14, sub, size=size - 3, op=0.72)
    else:
        out += text(x + w / 2, y + h / 2 + 4.5, label, size=size)
    return out


# --------------------------------------------------------------------------
# 1. the bandwidth hierarchy
# --------------------------------------------------------------------------

# Bandwidths in GB/sec and the memory reachable at each level, for the Llama 3
# cluster. The cross-pod figure is not published as a number; the paper gives a
# 1:7 oversubscription ratio at the aggregation layer, so it is derived here
# from the pod figure rather than asserted.
POD_BW = 50.0
OVERSUBSCRIPTION = 7
LEVELS = [
    ("Within one GPU", "HBM to compute cores", 3352.0, "80 GB", TEAL),
    ("Within one server", "NVLink, 8 GPUs", 900.0, "640 GB", GREEN),
    ("Within one pod", "192 racks, 3,072 GPUs", POD_BW, "246 TB", INDIGO),
    ("Across pods", "8 pods, 24,576 GPUs", POD_BW / OVERSUBSCRIPTION, "1.9 PB", PURPLE),
]


def fig_memory_hierarchy():
    """Bandwidth and reachable capacity at each level of the hierarchy."""
    w, h = 780, 300
    x0, x1 = 200, 600
    lo, hi = 1.0, 10000.0

    def px(v):
        return x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)

    body = text(20, 26, "Bandwidth between the two ends of each level", size=13.5, anchor="start", weight="600")
    for tick in (1, 10, 100, 1000, 10000):
        body += line(px(tick), 52, px(tick), 236, op=0.16, dash="2 4")
        body += text(px(tick), 258, f"{tick:,}", size=10.5, op=0.6)
    body += text((x0 + x1) / 2, 278, "GB / sec  (log scale)", size=11.5, op=0.7)

    for i, (name, sub, bw, cap, colour) in enumerate(LEVELS):
        y = 62 + i * 44
        body += text(190, y + 12, name, size=12.5, anchor="end")
        body += text(190, y + 26, sub, size=10, anchor="end", op=0.62)
        body += rect(x0, y, px(bw) - x0, 22, fill=colour, stroke=colour, sw=1.1, op=0.42, rx=2)
        shown = f"{bw:,.0f}" if bw >= 10 else f"≈{bw:.0f}"
        body += text(px(bw) + 8, y + 16, shown, size=11.5, anchor="start", font=MONO)
        body += text(w - 16, y + 16, cap, size=11.5, anchor="end", op=0.72)

    body += text(w - 16, 40, "memory reachable", size=10.5, anchor="end", op=0.6)
    return wrap(
        w,
        h,
        "memhier",
        "Bandwidth at four levels of a GPU cluster's memory hierarchy",
        "Horizontal bars on a logarithmic axis. Within one GPU, 3352 gigabytes "
        "per second reaching 80 gigabytes. Within one server of eight GPUs over "
        "NVLink, 900 gigabytes per second reaching 640 gigabytes. Within one pod "
        "of 3072 GPUs, 50 gigabytes per second reaching 246 terabytes. Across the "
        "eight pods of the full cluster, about 7 gigabytes per second reaching "
        "1.9 petabytes. Each step outward multiplies capacity and divides speed.",
        body,
    )


# --------------------------------------------------------------------------
# 2. the four axes
# --------------------------------------------------------------------------


def fig_axes():
    """The four indices of a transformer's computation and their splits."""
    w, h = 840, 330
    body = text(20, 28, "A stack of L layers", size=12.5, anchor="start", op=0.75)

    # The layer stack, drawn bottom to top so that the dashed cut is legible.
    n_layers = 7
    sx, sw_, sh = 46, 130, 20
    for i in range(n_layers):
        y = 250 - i * 30
        body += rect(sx, y, sw_, sh, fill=INK, stroke=INK, sw=1.1, op=0.08, rx=3)
        body += text(sx + sw_ / 2, y + 14.5, f"layer {n_layers - i}", size=10.5, op=0.8)
    body += line(sx - 18, 155, sx + sw_ + 18, 155, stroke=PURPLE, sw=1.6, dash="5 4")
    # Beside the stack, not over it: the gaps between plates are 10px tall.
    body += text(sx + sw_ + 26, 159, "PP — split L", size=11.5, anchor="start", fill=PURPLE, weight="600")

    # Kept low and short so it clears the sequence labels on the cuboid's left edge.
    body += arrow(sx + sw_ + 26, 212, 370, 212, sw=1.3, op=0.6)
    body += text((sx + sw_ + 26 + 370) / 2, 202, "each layer acts on", size=10.5, op=0.7)

    # The (B, S, D) activation tensor.
    bx, by, bw_, bh, dx, dy = 380, 96, 268, 168, 54, -46
    body += cuboid(bx, by, bw_, bh, dx, dy, stroke=INK, fill=INK, op=0.09)

    # D runs along the front face's width -> a vertical cut.
    for frac in (0.5,):
        cx = bx + bw_ * frac
        body += line(cx, by, cx + dx, by + dy, stroke=TEAL, sw=1.6, dash="5 4")
        body += line(cx, by, cx, by + bh, stroke=TEAL, sw=1.6, dash="5 4")
    body += text(bx + bw_ / 2, by + bh + 22, "D — feature", size=11, op=0.75)
    body += text(bx + bw_ / 2 + 4, by + bh + 40, "TP — split D", size=11.5, fill=TEAL, weight="600")

    # S runs down the front face -> a horizontal cut.
    cy = by + bh * 0.5
    body += line(bx, cy, bx + bw_, cy, stroke=GREEN, sw=1.6, dash="5 4")
    body += line(bx + bw_, cy, bx + bw_ + dx, cy + dy, stroke=GREEN, sw=1.6, dash="5 4")
    body += text(bx - 12, by + 44, "S — sequence", size=11, anchor="end", op=0.75)
    body += text(bx - 12, by + 60, "CP — split S", size=11.5, anchor="end", fill=GREEN, weight="600")

    # B runs into the page -> a cut across the depth.
    mx, my = bx + bw_ + dx * 0.5, by + dy * 0.5
    body += line(mx, my, mx, my + bh, stroke=INDIGO, sw=1.6, dash="5 4")
    body += text(bx + bw_ + dx / 2 + 24, by + dy - 26, "B — batch", size=11, op=0.75, anchor="start")
    body += text(
        bx + bw_ + dx / 2 + 24, by + dy - 10, "DP — split B", size=11.5, fill=INDIGO, weight="600", anchor="start"
    )

    body += text(20, h - 14, "Four indices, four ways to cut them across devices.", size=11, anchor="start", op=0.7)
    return wrap(
        w,
        h,
        "axes",
        "The four axes of a transformer's computation",
        "At left a stack of seven layers with a dashed cut between them labelled "
        "pipeline parallelism splits L. At right the activation tensor drawn as a "
        "cuboid with three dashed cutting planes: along the feature dimension D "
        "for tensor parallelism, along the sequence dimension S for context "
        "parallelism, and across the batch depth B for data parallelism.",
        body,
    )


# --------------------------------------------------------------------------
# 3. data parallelism
# --------------------------------------------------------------------------


def fig_data_parallel():
    """Three replicas, local forward and backward, one all-reduce."""
    w, h = 820, 300
    rows = 3
    top, gap = 62, 68
    # x increases strictly along the direction of the computation.
    cx_data, cx_fwd, cx_bwd, cx_grad = 56, 116, 246, 380
    ar_x, ar_w = 476, 62
    cx_avg, cx_upd = 566, 676

    body = ""
    for c, label in (
        (cx_data, "data"),
        (cx_fwd + 55, "forward"),
        (cx_bwd + 55, "backward"),
        (cx_grad + 38, "local ∂L/∂W"),
        (ar_x + ar_w / 2, "all-reduce"),
        (cx_avg + 38, "averaged"),
        (cx_upd + 55, "update"),
    ):
        body += text(c + 14 if label == "data" else c, 34, label, size=11, op=0.68)

    for i in range(rows):
        y = top + i * gap
        colour = DEVICE[i]
        body += text(24, y + 26, f"GPU {i + 1}", size=11, anchor="start", op=0.8)
        body += rect(cx_data, y + 8, 30, 30, fill=colour, stroke=colour, sw=1.1, op=0.34, rx=3)
        body += text(cx_data + 15, y + 28, f"x{i + 1}", size=10.5, font=MONO)
        body += arrow(cx_data + 34, y + 23, cx_fwd - 6, y + 23, sw=1.2, op=0.55)
        body += box(cx_fwd, y + 4, 110, 38, "forward", colour=colour, op=0.14)
        body += arrow(cx_fwd + 114, y + 23, cx_bwd - 6, y + 23, sw=1.2, op=0.55)
        body += box(cx_bwd, y + 4, 110, 38, "backward", colour=colour, op=0.14)
        body += arrow(cx_bwd + 114, y + 23, cx_grad - 6, y + 23, sw=1.2, op=0.55)
        body += rect(cx_grad, y + 8, 76, 30, fill=colour, stroke=colour, sw=1.1, op=0.34, rx=3)
        body += text(cx_grad + 38, y + 28, "∂L/∂W", size=10.5, font=MONO)
        body += arrow(cx_grad + 80, y + 23, ar_x - 4, y + 23, sw=1.2, op=0.55)
        # out of the all-reduce and into the update
        body += arrow(ar_x + ar_w + 4, y + 23, cx_avg - 6, y + 23, sw=1.2, op=0.55)
        body += rect(cx_avg, y + 8, 76, 30, fill=INK, stroke=INK, sw=1.1, op=0.12, rx=3)
        body += text(cx_avg + 38, y + 28, "∂L/∂W", size=10.5, font=MONO)
        body += arrow(cx_avg + 80, y + 23, cx_upd - 6, y + 23, sw=1.2, op=0.55)
        body += box(cx_upd, y + 4, 110, 38, "W ← W − η∂L/∂W", colour=INK, op=0.08, size=10.5)

    ar_top, ar_bot = top - 2, top + (rows - 1) * gap + 48
    body += rect(ar_x, ar_top, ar_w, ar_bot - ar_top, fill=YELLOW, stroke=INDIGO, sw=1.4, op=0.28, rx=5)
    body += text(ar_x + ar_w / 2, ar_bot + 18, "the only", size=10.5, fill=INDIGO)
    body += text(ar_x + ar_w / 2, ar_bot + 32, "communication", size=10.5, fill=INDIGO)

    body += text(
        24,
        h - 12,
        "Every replica ends the step with identical weights, so no drift accumulates.",
        size=11,
        anchor="start",
        op=0.7,
    )
    return wrap(
        w,
        h,
        "dp",
        "Data parallelism across three GPUs",
        "Three rows, one per GPU. Each row runs left to right: its own minibatch, "
        "a forward pass, a backward pass, and its own local gradient. All three "
        "rows then pass through a single shaded all-reduce column, after which "
        "each row holds the same averaged gradient and applies the same weight "
        "update.",
        body,
    )


# --------------------------------------------------------------------------
# 4. the FSDP backward pass
# --------------------------------------------------------------------------

N_FSDP_LAYERS = 6
FSDP_LANES = [
    ("all-gather weights", "interconnect", -1, INDIGO),
    ("backward pass", "tensor cores", 0, TEAL),
    ("reduce-scatter + step", "interconnect", +1, PURPLE),
]


def fig_fsdp():
    """The three-lane steady state of an FSDP backward pass."""
    w, h = 812, 300
    cols = N_FSDP_LAYERS + 1
    x0, cw = 168, 88
    lane_h, lane_gap = 44, 14
    top = 70
    highlight = 2

    body = text(20, 28, "time", size=11.5, anchor="start", op=0.7)
    body += arrow(56, 24, 150, 24, sw=1.2, op=0.5)
    body += text(x0 + cols * cw / 2, 28, "backward pass sweeps from layer 6 down to layer 1", size=11, op=0.7)

    hx = x0 + highlight * cw
    body += rect(
        hx - 3, top - 12, cw + 6, 3 * lane_h + 2 * lane_gap + 24, fill=YELLOW, stroke=INDIGO, sw=1.3, op=0.24, rx=5
    )

    for li, (name, where, shift, colour) in enumerate(FSDP_LANES):
        y = top + li * (lane_h + lane_gap)
        body += text(156, y + lane_h / 2 - 2, name, size=11.5, anchor="end")
        body += text(156, y + lane_h / 2 + 13, where, size=9.5, anchor="end", op=0.6)
        for t in range(cols):
            layer = N_FSDP_LAYERS - t - shift
            if not 1 <= layer <= N_FSDP_LAYERS:
                continue
            bx = x0 + t * cw
            body += rect(bx + 3, y, cw - 6, lane_h, fill=colour, stroke=colour, sw=1.2, op=0.32, rx=3)
            body += text(bx + cw / 2, y + lane_h / 2 + 4.5, f"L{layer}", size=12, font=MONO)

    body += text(
        hx + cw / 2,
        top + 3 * lane_h + 2 * lane_gap + 30,
        "three consecutive layers, three resources, one instant",
        size=11,
        fill=INDIGO,
    )
    body += text(
        20,
        h - 12,
        "The gather runs one layer ahead of the arithmetic and the reduction one layer behind it.",
        size=11,
        anchor="start",
        op=0.7,
    )
    return wrap(
        w,
        h,
        "fsdp",
        "Overlapped communication and computation in an FSDP backward pass",
        "Three horizontal lanes against time. The top lane, on the interconnect, "
        "all-gathers the weights for the next layer down. The middle lane, on the "
        "tensor cores, computes the backward pass for the current layer. The "
        "bottom lane, on the interconnect, reduce-scatters the gradients of the "
        "previous layer and applies its optimizer step. A highlighted column shows "
        "layers three, four and five in flight at the same instant.",
        body,
    )


# --------------------------------------------------------------------------
# 5. hybrid sharding
# --------------------------------------------------------------------------

HSDP_SHARDS = 4
HSDP_REPLICAS = 2


def fig_hsdp():
    """Sharding inside the fast domain, replication across the slow one."""
    w, h = 800, 362
    gw, gh = 118, 62
    x0, y0 = 176, 84
    xgap, ygap = 138, 138

    body = text(20, 28, "shard the model this way  →", size=11.5, anchor="start", fill=TEAL, weight="600")
    body += text(20, 48, "replicate it this way  ↓", size=11.5, anchor="start", fill=PURPLE, weight="600")

    for r in range(HSDP_REPLICAS):
        ytop = y0 + r * ygap
        body += rect(
            x0 - 18,
            ytop - 20,
            (HSDP_SHARDS - 1) * xgap + gw + 36,
            gh + 40,
            fill=TEAL,
            stroke=TEAL,
            sw=1.3,
            op=0.07,
            rx=8,
        )
        body += text(x0 - 26, ytop + gh / 2 + 4, f"server {r}", size=11.5, anchor="end")
        for s in range(HSDP_SHARDS):
            x = x0 + s * xgap
            body += box(
                x, ytop, gw, gh, f"GPU {r * HSDP_SHARDS + s}", sub=f"owns W{s + 1}", colour=DEVICE[s], op=0.22, size=12
            )
            if s:
                body += arrow(x - xgap + gw + 4, ytop + gh / 2, x - 6, ytop + gh / 2, sw=1.2, op=0.5, stroke=TEAL)
                body += arrow(x - 6, ytop + gh + 12, x - xgap + gw + 4, ytop + gh + 12, sw=1.2, op=0.5, stroke=TEAL)
        # Both rows are the same arrangement, so the caption is drawn once. A
        # second copy would land where the DP caption belongs.
        if r == 0:
            body += text(
                x0 + ((HSDP_SHARDS - 1) * xgap + gw) / 2,
                ytop + gh + 34,
                "FSDP inside the server — ≈3× model size on NVLink, 900 GB/s",
                size=10.5,
                fill=TEAL,
            )

    for s in range(HSDP_SHARDS):
        x = x0 + s * xgap + gw / 2
        body += arrow(x, y0 + gh + 44, x, y0 + ygap - 24, sw=1.3, op=0.55, stroke=PURPLE)
    body += text(
        x0 + ((HSDP_SHARDS - 1) * xgap + gw) / 2,
        y0 + ygap + gh + 34,
        "DP across servers — all-reduce gradients only, 1× model size, 50 GB/s",
        size=10.5,
        fill=PURPLE,
    )
    body += text(
        20,
        h - 12,
        "The expensive collective is confined to the fastest link in the machine.",
        size=11,
        anchor="start",
        op=0.7,
    )
    return wrap(
        w,
        h,
        "hsdp",
        "Hybrid sharded data parallelism as a two-dimensional grid",
        "Two rows of four GPUs. Each row is one server, boxed, with arrows running "
        "left and right between its GPUs marked as fully sharded data parallelism "
        "costing about three model sizes of traffic on NVLink at 900 gigabytes per "
        "second. Vertical arrows join the two rows, marked as plain data "
        "parallelism costing one model size of gradient traffic at 50 gigabytes "
        "per second.",
        body,
    )


# --------------------------------------------------------------------------
# 6. activation checkpointing
# --------------------------------------------------------------------------

N_CKPT = 126  # Llama 3 405B's depth, so the figure is about a real network


def checkpoint_cost(n, c):
    """Compute and peak memory for n layers with c naive checkpoints.

    Compute is counted in single-layer steps: n forward, n backward, and, for each backward step, a rerun of its
    segment's prefix from the nearest checkpoint — on average half a segment. Memory is the c retained
    activations. This is the accounting in which recomputation is redone per
    layer rather than once per segment; `SEGMENT_SCHEME` below is the version real implementations use.
    """
    seg = n / c
    return 2 * n + n * max(seg - 1, 0.0) / 2, float(c)


# What torch.utils.checkpoint actually does: recompute each segment once, so the
# whole forward pass is repeated exactly once and the segment stays resident.
def segment_scheme(n, seg):
    """Compute and peak memory when each segment is recomputed once."""
    return 3 * n, n / seg + seg


def fig_checkpointing():
    """The compute-memory trade-off for activation checkpointing."""
    w, h = 760, 380
    x0, x1, y0, y1 = 92, 700, 300, 52
    mlo, mhi = 1.0, 200.0
    clo, chi = 200.0, 12000.0

    def px(m, c):
        return (
            x0 + (math.log10(m) - math.log10(mlo)) / (math.log10(mhi) - math.log10(mlo)) * (x1 - x0),
            y0 - (math.log10(c) - math.log10(clo)) / (math.log10(chi) - math.log10(clo)) * (y0 - y1),
        )

    body = ""
    for m in (1, 10, 100):
        gx, _ = px(m, clo)
        body += line(gx, y1, gx, y0, op=0.14, dash="2 4")
        body += text(gx, y0 + 20, f"{m}", size=10.5, op=0.62)
    for c in (200, 1000, 10000):
        _, gy = px(mlo, c)
        body += line(x0, gy, x1, gy, op=0.14, dash="2 4")
        body += text(x0 - 10, gy + 4, f"{c:,}", size=10.5, anchor="end", op=0.62)
    body += text(
        (x0 + x1) / 2, y0 + 40, "peak activations held  (units of one layer's activations, log)", size=11.5, op=0.75
    )
    body += text(20, 30, "compute", size=11.5, anchor="start", op=0.75)
    body += text(20, 46, "(layer-steps, log)", size=10, anchor="start", op=0.6)

    pts = [px(mem, comp) for comp, mem in (checkpoint_cost(N_CKPT, c) for c in range(1, N_CKPT + 1))]
    body += polyline(pts, stroke=INDIGO, sw=2.0, op=0.85)

    marks = [
        (checkpoint_cost(N_CKPT, 1), "recompute everything", "C = 1", PURPLE, 12, -14),
        (
            checkpoint_cost(N_CKPT, round(math.sqrt(N_CKPT))),
            "√N checkpoints",
            f"C = {round(math.sqrt(N_CKPT))}",
            INDIGO,
            14,
            -6,
        ),
        # "store everything" is labelled above rather than to the left, so its
        # text cannot run back into the segment-scheme label further up the plot.
        (checkpoint_cost(N_CKPT, N_CKPT), "store everything", "C = N", TEAL, 0, -34),
    ]
    for (comp, mem), name, sub, colour, ox, oy in marks:
        cx, cy = px(mem, comp)
        body += rect(cx - 5, cy - 5, 10, 10, fill=colour, stroke=colour, sw=1.4, op=0.85, rx=2)
        anchor = "start" if ox > 0 else "end" if ox < 0 else "middle"
        body += text(cx + ox, cy + oy, name, size=11.5, anchor=anchor, fill=colour, weight="600")
        body += text(cx + ox, cy + oy + 14, sub, size=10, anchor=anchor, op=0.65, font=MONO)

    scomp, smem = segment_scheme(N_CKPT, math.sqrt(N_CKPT))
    sx, sy = px(smem, scomp)
    body += rect(sx - 5, sy - 5, 10, 10, fill=GREEN, stroke=GREEN, sw=1.4, op=0.9, rx=2)
    body += text(sx - 14, sy - 10, "recompute each segment once", size=11.5, anchor="end", fill=GREEN, weight="600")
    body += text(sx - 14, sy + 4, "3N compute at 2√N memory", size=10, anchor="end", op=0.65, font=MONO)

    body += text(
        20,
        h - 12,
        f"N = {N_CKPT} layers, the depth of Llama 3 405B. Down and to the left is better.",
        size=11,
        anchor="start",
        op=0.7,
    )
    return wrap(
        w,
        h,
        "ckpt",
        "The compute-memory trade-off of activation checkpointing",
        "A log-log plot with peak activation memory on the horizontal axis and "
        "compute on the vertical. A curve traces the family of naive checkpointing "
        "schemes for a 126-layer network, running from recompute-everything at the "
        "top left through the square-root-N point to store-everything at the bottom "
        "right. A separate marked point below the curve is the segment-recompute "
        "scheme, which reaches square-root memory at three times the compute.",
        body,
    )


# --------------------------------------------------------------------------
# 7. model FLOPs utilisation
# --------------------------------------------------------------------------

# Table 4 of the Llama 3 paper: GPUs, CP degree, sequence length, BF16 MFU.
MFU_STAGES = [
    ("8,192 GPUs", "S = 8,192, DP = 64", 43),
    ("16,384 GPUs", "S = 8,192, DP = 128", 41),
    ("16,384 GPUs", "S = 131,072, CP = 16", 38),
]


def fig_mfu():
    """BF16 MFU for the three stages of Llama 3 405B pre-training."""
    w, h = 640, 320
    y0, y1 = 250, 56
    bw_, gap = 96, 62
    x_start = 110

    body = text(20, 28, "BF16 MFU", size=12, anchor="start", op=0.75)
    for v in (0, 10, 20, 30, 40, 50):
        gy = y0 - v / 50 * (y0 - y1)
        body += line(x_start - 24, gy, w - 24, gy, op=0.14, dash="2 4")
        body += text(x_start - 34, gy + 4, f"{v}%", size=10.5, anchor="end", op=0.62)

    for i, (label, sub, v) in enumerate(MFU_STAGES):
        x = x_start + i * (bw_ + gap)
        top = y0 - v / 50 * (y0 - y1)
        body += rect(x, top, bw_, y0 - top, fill=DEVICE[i], stroke=DEVICE[i], sw=1.2, op=0.5, rx=3)
        body += text(x + bw_ / 2, top - 10, f"{v}%", size=13, weight="600")
        body += text(x + bw_ / 2, y0 + 20, label, size=11)
        body += text(x + bw_ / 2, y0 + 35, sub, size=9.5, op=0.65, font=MONO)

    band_lo = y0 - 30 / 50 * (y0 - y1)
    body += line(x_start - 24, band_lo, w - 24, band_lo, stroke=INDIGO, sw=1.5, dash="6 4")
    body += text(w - 24, band_lo - 8, "30% — below this, something is wrong", size=10.5, anchor="end", fill=INDIGO)
    body += text(
        20,
        h - 12,
        "Two months of a 16,384-GPU cluster, spending three-fifths of its peak on something other than the model.",
        size=11,
        anchor="start",
        op=0.7,
    )
    return wrap(
        w,
        h,
        "mfu",
        "Model FLOPs utilisation for the three stages of Llama 3 405B pre-training",
        "Three bars on a percentage axis: 43 percent on 8192 GPUs with a sequence "
        "length of 8192, 41 percent on 16384 GPUs at the same sequence length with "
        "the data-parallel degree doubled, and 38 percent on 16384 GPUs in the "
        "long-context stage at sequence length 131072 with 16-way context "
        "parallelism. A dashed line marks the 30 percent level.",
        body,
    )


# --------------------------------------------------------------------------
# 8. tensor parallelism
# --------------------------------------------------------------------------

TP_WAYS = 4


def fig_tensor_parallel():
    """Column-parallel feeding row-parallel, with one all-reduce per pair."""
    w, h = 840, 400
    body = ""

    def matrix(x, y, mw, mh, splits, axis, label, shape, strong=True):
        """Draw a block matrix split along 'axis' ('col' or 'row')."""
        out = ""
        for i in range(splits):
            if axis == "col":
                bx, by = x + i * mw / splits, y
                bw2, bh2 = mw / splits, mh
            else:
                bx, by = x, y + i * mh / splits
                bw2, bh2 = mw, mh / splits
            out += rect(bx, by, bw2, bh2, fill=DEVICE[i], stroke=DEVICE[i], sw=1.2, op=0.40 if strong else 0.18)
            out += text(
                bx + bw2 / 2, by + bh2 / 2 + 4.5, f"{label}<tspan font-size='9' dy='3'>{i + 1}</tspan>", size=12
            )
        out += rect(x, y, mw, mh, fill="none", stroke=INK, sw=1.3)
        out += text(x + mw / 2, y + mh + 17, shape, size=10, op=0.68, font=MONO)
        return out

    def plain(x, y, mw, mh, label, shape):
        out = rect(x, y, mw, mh, fill=INK, stroke=INK, sw=1.3, op=0.09)
        out += text(x + mw / 2, y + mh / 2 + 5, label, size=13)
        out += text(x + mw / 2, y + mh + 17, shape, size=10, op=0.68, font=MONO)
        return out

    # Row 1: column-parallel. No communication.
    y = 56
    body += text(20, 34, "layer 1 — split W into column blocks", size=12, anchor="start", weight="600")
    body += plain(58, y, 62, 84, "X", "N × D")
    body += text(134, y + 48, "×", size=16, op=0.7)
    body += matrix(150, y, 208, 84, TP_WAYS, "col", "W", "D × D")
    body += text(374, y + 48, "=", size=16, op=0.7)
    body += matrix(392, y, 208, 84, TP_WAYS, "col", "Y", "N × D")
    body += text(628, y + 34, "each GPU already holds X,", size=11, anchor="start", op=0.8)
    body += text(628, y + 50, "so it computes X W", size=11, anchor="start", op=0.8)
    body += text(628, y + 66, "with no communication.", size=11, anchor="start", fill=TEAL, weight="600")

    # Row 2: row-parallel. One all-reduce.
    y2 = 228
    body += text(20, 206, "layer 2 — split U into row blocks", size=12, anchor="start", weight="600")
    body += matrix(58, y2, 168, 84, TP_WAYS, "col", "Y", "N × D", strong=False)
    body += text(240, y2 + 48, "×", size=16, op=0.7)
    body += matrix(258, y2, 96, 84, TP_WAYS, "row", "U", "D × D")
    body += text(370, y2 + 48, "=", size=16, op=0.7)
    for i in range(TP_WAYS):
        bx = 388 + i * 46
        body += rect(bx, y2, 40, 84, fill=DEVICE[i], stroke=DEVICE[i], sw=1.2, op=0.40)
        body += text(
            bx + 20,
            y2 + 46,
            f"Y<tspan font-size='9' dy='3'>{i + 1}</tspan>"
            f"<tspan font-size='12' dy='-3'>U</tspan>"
            f"<tspan font-size='9' dy='3'>{i + 1}</tspan>",
            size=12,
        )
        if i:
            body += text(bx - 6, y2 + 48, "+", size=13, op=0.7)
    body += text(468, y2 + 101, "four full N × D partial sums", size=10, op=0.68, font=MONO)

    ax = 584
    body += arrow(ax, y2 + 42, ax + 34, y2 + 42, sw=1.4, op=0.7, stroke=INDIGO)
    body += rect(ax + 40, y2 + 14, 74, 56, fill=YELLOW, stroke=INDIGO, sw=1.4, op=0.28, rx=5)
    body += text(ax + 77, y2 + 38, "all-", size=11, fill=INDIGO)
    body += text(ax + 77, y2 + 52, "reduce", size=11, fill=INDIGO)
    body += arrow(ax + 118, y2 + 42, ax + 150, y2 + 42, sw=1.4, op=0.7, stroke=INDIGO)
    body += plain(ax + 156, y2 + 8, 62, 68, "Z", "N × D")

    body += text(
        20,
        h - 14,
        "Two layers, one collective — which is exactly the shape of a transformer's feed-forward block.",
        size=11,
        anchor="start",
        op=0.7,
    )
    return wrap(
        w,
        h,
        "tp",
        "Column-parallel and row-parallel block matrix multiplication",
        "The top row shows X times W equals Y, with W cut into four column blocks "
        "in four colours and Y inheriting the same four column blocks, requiring no "
        "communication. The bottom row shows those Y blocks multiplied by U cut "
        "into four row blocks of matching colour, producing four full-sized partial "
        "sums that are combined by a single all-reduce into Z.",
        body,
    )


# --------------------------------------------------------------------------
# 9. the pipeline bubble
# --------------------------------------------------------------------------

PP_STAGES = 4
PP_MICROBATCHES = 4


def gpipe_schedule(p, m):
    """Return {(stage, slot): (microbatch, 'F' | 'B')} for a GPipe schedule.

    Forward for microbatch j reaches stage g at slot j + g, so the last forward lands at slot m + p - 2. The backward
    sweep runs in the opposite direction: stage g takes microbatch j at slot (m + p - 1) + j + (p - 1 - g). Total length
    is 2(m + p - 1) slots.
    """
    cells = {}
    for g in range(p):
        for j in range(m):
            cells[(g, j + g)] = (j, "F")
            cells[(g, (m + p - 1) + j + (p - 1 - g))] = (j, "B")
    return cells


def fig_pipeline_bubble():
    """A four-stage pipeline with four microbatches, bubble shaded."""
    p, m = PP_STAGES, PP_MICROBATCHES
    cells = gpipe_schedule(p, m)
    slots = 2 * (m + p - 1)
    cw, ch = 50, 44
    x0, y0 = 96, 74
    w, h = x0 + slots * cw + 24, y0 + p * ch + 76

    body = text(20, 32, "time", size=11.5, anchor="start", op=0.7)
    body += arrow(56, 28, 140, 28, sw=1.2, op=0.5)
    body += text(x0 + slots * cw / 2, 32, "forward sweep, then backward sweep", size=11, op=0.7)

    for g in range(p):
        y = y0 + g * ch
        body += text(x0 - 12, y + ch / 2 + 4, f"GPU {g + 1}", size=11, anchor="end")
        for t in range(slots):
            x = x0 + t * cw
            hit = cells.get((g, t))
            if hit is None:
                body += rect(x, y, cw, ch, fill=INK, stroke=INK, sw=0.8, op=0.06, so=0.25)
                continue
            j, kind = hit
            body += rect(x, y, cw, ch, fill=DEVICE[j], stroke=DEVICE[j], sw=1.1, op=0.44 if kind == "F" else 0.24)
            body += text(x + cw / 2, y + ch / 2 + 4.5, f"{kind}{j + 1}", size=11.5, font=MONO)

    filled = len(cells)
    total = p * slots
    frac = filled / total
    ybot = y0 + p * ch
    body += text(
        x0, ybot + 26, f"{filled} busy cells of {total} — utilisation {frac:.1%}", size=12, anchor="start", weight="600"
    )
    body += text(
        x0,
        ybot + 44,
        f"m / (m + P − 1) = {m} / {m + p - 1};  the pale cells are the bubble",
        size=11,
        anchor="start",
        op=0.7,
    )
    body += text(
        20,
        h - 12,
        "Raising the microbatch count shrinks the bubble and raises the activation memory that must be held.",
        size=11,
        anchor="start",
        op=0.7,
    )
    return wrap(
        w,
        h,
        "pp",
        "A four-stage GPipe pipeline with four microbatches",
        "A grid of four GPU rows against fourteen time slots. Coloured cells mark "
        "forward and backward work on each of four microbatches, arranged as a "
        "descending diagonal during the forward sweep and an ascending one during "
        "the backward sweep. The pale cells at the start and end are the bubble; "
        "32 of the 56 cells are busy, a utilisation of 57.1 percent.",
        body,
    )


FIGURES = {
    "diagram_memory_hierarchy.svg": fig_memory_hierarchy,
    "diagram_axes.svg": fig_axes,
    "diagram_data_parallel.svg": fig_data_parallel,
    "diagram_fsdp.svg": fig_fsdp,
    "diagram_hsdp.svg": fig_hsdp,
    "diagram_checkpointing.svg": fig_checkpointing,
    "diagram_mfu.svg": fig_mfu,
    "diagram_tensor_parallel.svg": fig_tensor_parallel,
    "diagram_pipeline_bubble.svg": fig_pipeline_bubble,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print(f"wrote {OUT / name}")

    print()
    # The chapter quotes these; print them so the prose can be checked.
    params, tokens = 405e9, 15.6e12
    print(f"6ND = {6 * params * tokens:.3g} FLOPs (paper: 3.8e25)")
    gpus, per_gpu = 16384, 400e12
    secs = 6 * params * tokens / (gpus * per_gpu)
    print(f"at {gpus:,} GPUs x {per_gpu:.0e} FLOP/s: {secs:.2g} s = {secs / 86400:.0f} d")

    sm_fp32, sm_tc = 128 * 2, 16 * 4 * 8 * 2 * 4
    print(f"per SM per cycle: FP32 {sm_fp32}, tensor {sm_tc}, ratio {sm_tc // sm_fp32}x")
    print(f"cluster: {24576 * 132 * 128:,} FP32 cores, {24576 * 132 * 4:,} tensor cores")
    print(f"cluster peak: {24576 * 989e12 / 1e18:.1f} EFLOP/s, {24576 * 80 / 2**20:.3f} PiB HBM")
    print(f"cross-pod bandwidth: {POD_BW}/{OVERSUBSCRIPTION} = {POD_BW / OVERSUBSCRIPTION:.1f} GB/s")

    layers, dim, seq = 126, 16384, 8192
    act = layers * seq * dim * 2
    print(f"one tensor per layer, one sequence: {act / 1e9:.1f} GB = {act / 80e9:.0%} of an 80 GB H100")

    c_all = checkpoint_cost(N_CKPT, N_CKPT)
    c_root = checkpoint_cost(N_CKPT, round(math.sqrt(N_CKPT)))
    c_one = checkpoint_cost(N_CKPT, 1)
    print(
        f"N={N_CKPT}: store all {c_all[0]:.0f} compute / {c_all[1]:.0f} mem; "
        f"sqrt {c_root[0]:.0f} / {c_root[1]:.0f}; "
        f"full recompute {c_one[0]:.0f} / {c_one[1]:.0f}"
    )
    seg = segment_scheme(N_CKPT, math.sqrt(N_CKPT))
    print(f"segment scheme: {seg[0]:.0f} compute / {seg[1]:.1f} mem")

    for p, mm in ((4, 1), (8, 1), (4, 4), (4, 16)):
        print(f"PP P={p} m={mm}: utilisation {mm / (mm + p - 1):.1%}, bubble {(p - 1) / (mm + p - 1):.1%}")
    print(f"Llama 3 4D: 8 x 16 x 16 x 8 = {8 * 16 * 16 * 8:,} GPUs")
