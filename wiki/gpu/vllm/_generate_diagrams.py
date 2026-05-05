"""Append new scenes to the Phase 1 Excalidraw file.

Generates 7 additional diagrams with consistent styling.
Run once; does not mutate scenes already present.
"""
import json
from pathlib import Path

PATH = Path(__file__).parent / "vllm-phase1.excalidraw"

# Counters for unique IDs
_seed = [1000]
def next_seed():
    _seed[0] += 1
    return _seed[0]

def rect(x, y, w, h, fill="#ffffff", stroke="#1e1e1e", fill_style="solid",
         opacity=100, dashed=False, rounded=False):
    return {
        "id": f"r{next_seed()}",
        "type": "rectangle",
        "x": x, "y": y, "width": w, "height": h, "angle": 0,
        "strokeColor": stroke, "backgroundColor": fill, "fillStyle": fill_style,
        "strokeWidth": 2, "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 1, "opacity": opacity, "groupIds": [], "frameId": None,
        "roundness": {"type": 3} if rounded else None,
        "seed": next_seed(), "version": 1, "versionNonce": next_seed(),
        "isDeleted": False, "boundElements": None, "updated": 1,
        "link": None, "locked": False,
    }

def text(x, y, t, size=14, color="#1e1e1e", width=None, family=1):
    if width is None:
        width = max(200, len(t) * size * 0.6)
    return {
        "id": f"t{next_seed()}",
        "type": "text",
        "x": x, "y": y, "width": width, "height": size + 8, "angle": 0,
        "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": next_seed(), "version": 1,
        "versionNonce": next_seed(), "isDeleted": False,
        "boundElements": None, "updated": 1, "link": None, "locked": False,
        "text": t, "fontSize": size, "fontFamily": family,
        "textAlign": "left", "verticalAlign": "top",
        "baseline": size - 2, "containerId": None, "originalText": t,
    }

def arrow(x1, y1, x2, y2, color="#1e1e1e"):
    return {
        "id": f"a{next_seed()}",
        "type": "arrow",
        "x": x1, "y": y1, "width": abs(x2 - x1), "height": abs(y2 - y1),
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": {"type": 2}, "seed": next_seed(), "version": 1,
        "versionNonce": next_seed(), "isDeleted": False,
        "boundElements": None, "updated": 1, "link": None, "locked": False,
        "points": [[0, 0], [x2 - x1, y2 - y1]],
        "lastCommittedPoint": None,
        "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": "arrow",
    }

def frame(x, y, w, h, label, color="#495057", bg="#f8f9fa"):
    elements = [rect(x, y, w, h, fill=bg, stroke=color, opacity=30, rounded=True)]
    elements.append(text(x + w - 180, y + 8, label, size=14, color=color, width=170))
    return elements


BLUE = "#a5d8ff"
GREEN = "#b2f2bb"
RED = "#ffc9c9"
YELLOW = "#fff3bf"
PURPLE = "#d0bfff"
GRAY = "#e9ecef"
DARK_BLUE = "#1971c2"
DARK_GREEN = "#2b8a3e"
DARK_RED = "#c92a2a"
DARK_YELLOW = "#e67700"
DARK_PURPLE = "#6741d9"
DARK_GRAY = "#495057"


new_elements = []

# ============================================================================
# Scene 5: Lazy allocation over time
# ============================================================================
y0 = 1450
new_elements.extend(frame(70, y0, 1400, 320, "LAZY ALLOCATION (fix #1)", DARK_BLUE, "#e7f5ff"))
new_elements.append(text(90, y0 + 10, "Sequence's block table grows one entry at a time, as decode crosses each 16-token boundary",
                         size=18, color="#0b7285", width=1200))

# 4 snapshots: after 10, 20, 50, 100 tokens
snap_x = [100, 450, 800, 1150]
token_counts = [10, 20, 50, 100]
for i, (x, n) in enumerate(zip(snap_x, token_counts)):
    blocks_used = (n + 15) // 16
    new_elements.append(text(x, y0 + 50, f"After {n} tokens generated",
                             size=13, color=DARK_GRAY, width=200))
    new_elements.append(text(x, y0 + 72, f"→ owns {blocks_used} block(s)",
                             size=13, color=DARK_BLUE, width=200))
    # Draw blocks as small squares in a row
    for b in range(blocks_used):
        bx = x + b * 35
        new_elements.append(rect(bx, y0 + 100, 30, 30, fill=BLUE, stroke=DARK_BLUE))
        new_elements.append(text(bx + 7, y0 + 107, str(b), size=12, color=DARK_BLUE, width=25))
    # Table representation below
    tbl_entries = [f"{b}" for b in range(blocks_used)]
    tbl_text = f"block_table = [{', '.join(tbl_entries)}]"
    new_elements.append(text(x, y0 + 150, tbl_text, size=12, color=DARK_GRAY, width=300, family=3))
    # Used vs reserved
    unused = blocks_used * 16 - n
    new_elements.append(text(x, y0 + 180, f"tokens: {n} used / {blocks_used*16} slots",
                             size=12, color=DARK_GRAY, width=300))
    new_elements.append(text(x, y0 + 200, f"reserved-but-empty: {unused} (max 15)",
                             size=12, color=DARK_GRAY, width=300))

new_elements.append(text(100, y0 + 250, "Key: reserved-but-empty is capped at block_size - 1 = 15 tokens, regardless of how far the seq gets.",
                         size=14, color=DARK_BLUE, width=1300))
new_elements.append(text(100, y0 + 275, "Compare naive (fixed slab at max_seq_len=2048): ~1948 tokens reserved-but-empty at the 100-token mark.",
                         size=14, color=DARK_RED, width=1300))


# ============================================================================
# Scene 6: Continuous vs Static Batching Timeline
# ============================================================================
y0 = 1800
new_elements.extend(frame(70, y0, 1400, 450, "CONTINUOUS vs STATIC BATCHING", DARK_PURPLE, "#f3f0ff"))
new_elements.append(text(90, y0 + 10, "Same 5 requests, same arrivals, same completion points. Two scheduling strategies.",
                         size=18, color=DARK_PURPLE, width=1200))

# Requests: (name, arrival_step, num_tokens)
#  A: arrives step 0, 3 tokens
#  B: arrives step 0, 8 tokens  (the long one — dictates static batch length)
#  C: arrives step 0, 2 tokens
#  D: arrives step 0, 5 tokens
#  E: arrives step 0, 4 tokens
# We'll show 10 steps.
reqs_static = [
    ("A", 0, 3),
    ("B", 0, 8),
    ("C", 0, 2),
    ("D", 0, 5),
    ("E", 0, 4),
]
# For continuous, as each finishes a new one (F, G, H) enters
reqs_continuous = [
    ("A", 0, 3),
    ("B", 0, 8),
    ("C", 0, 2),
    ("D", 0, 5),
    ("E", 0, 4),
    ("F", 3, 6),  # replaces A
    ("G", 3, 4),  # replaces C (C finished at 2)
    ("H", 5, 3),  # replaces E
    ("I", 6, 2),  # replaces D
]

steps = 10
col_w = 90
row_h = 30
chart_x = 200
# ---------- static ----------
sy = y0 + 60
new_elements.append(text(100, sy + 5, "STATIC", size=16, color=DARK_GRAY, width=100))
new_elements.append(text(100, sy + 25, "batching", size=13, color=DARK_GRAY, width=100))
# step headers
for s in range(steps):
    new_elements.append(text(chart_x + s * col_w + 30, sy - 25, f"step {s+1}",
                             size=11, color=DARK_GRAY, width=60))
# rows for each slot (5 slots in static)
for slot_idx, (name, arr, n) in enumerate(reqs_static):
    ry = sy + slot_idx * row_h
    for s in range(steps):
        active = arr <= s < arr + n
        # static batch waits for the longest (B, 8 steps) before releasing
        # so all slots are "locked" until step 8 even if the seq finished earlier
        if active:
            color = BLUE
            lbl = name
        elif s < 8:  # locked, waiting
            color = GRAY
            lbl = ""
        else:  # batch released
            color = "#ffffff"
            lbl = ""
        new_elements.append(rect(chart_x + s * col_w, ry, col_w - 2, row_h - 2,
                                 fill=color, stroke=DARK_GRAY if color != "#ffffff" else "#ced4da"))
        if lbl:
            new_elements.append(text(chart_x + s * col_w + col_w // 2 - 5, ry + 6,
                                     lbl, size=14, color=DARK_BLUE, width=30))

new_elements.append(text(100, sy + 160, "Slots stay locked\nuntil B (longest)\nfinishes. Big gaps\n= wasted capacity.",
                         size=11, color=DARK_RED, width=130))

# ---------- continuous ----------
cy = sy + 230
new_elements.append(text(100, cy + 5, "CONTINUOUS", size=16, color=DARK_GRAY, width=120))
new_elements.append(text(100, cy + 25, "batching", size=13, color=DARK_GRAY, width=120))
for s in range(steps):
    new_elements.append(text(chart_x + s * col_w + 30, cy - 25, f"step {s+1}",
                             size=11, color=DARK_GRAY, width=60))

# Figure out which request is in which slot at each step (simple greedy fill)
slot_occupancy = [None] * 5  # slot → (name, end_step)
queue = sorted(reqs_continuous, key=lambda r: r[1])
# We'll simulate step by step
schedule = {s: [None]*5 for s in range(steps)}
for s in range(steps):
    # free slots where seq ended
    for i, occ in enumerate(slot_occupancy):
        if occ is not None and s >= occ[1]:
            slot_occupancy[i] = None
    # admit from queue
    for req in queue[:]:
        name, arr, n = req
        if arr <= s:
            for i in range(5):
                if slot_occupancy[i] is None:
                    slot_occupancy[i] = (name, s + n)
                    queue.remove(req)
                    break
    for i, occ in enumerate(slot_occupancy):
        schedule[s][i] = occ[0] if occ else None

# Render
for slot_idx in range(5):
    ry = cy + slot_idx * row_h
    for s in range(steps):
        name = schedule[s][slot_idx]
        color = GREEN if name else "#ffffff"
        new_elements.append(rect(chart_x + s * col_w, ry, col_w - 2, row_h - 2,
                                 fill=color, stroke=DARK_GREEN if name else "#ced4da"))
        if name:
            new_elements.append(text(chart_x + s * col_w + col_w // 2 - 5, ry + 6,
                                     name, size=14, color=DARK_GREEN, width=30))

new_elements.append(text(100, cy + 160, "Slots refill as\nseqs finish.\nBatch stays full.",
                         size=11, color=DARK_GREEN, width=130))

new_elements.append(text(100, cy + 220,
                         "Static: 5 seqs in 8 steps.  Continuous: 9 seqs in 10 steps.  ~80% more throughput, same GPU.",
                         size=15, color=DARK_GREEN, width=1300))


# ============================================================================
# Scene 7: Prefill vs Decode shapes
# ============================================================================
y0 = 2280
new_elements.extend(frame(70, y0, 1400, 360, "PREFILL vs DECODE — two different workload shapes", DARK_YELLOW, "#fff9db"))
new_elements.append(text(90, y0 + 10, "Both are forward passes through the same model. They differ in input shape and which resource they saturate.",
                         size=17, color=DARK_YELLOW, width=1300))

# PREFILL side
px = 150
py = y0 + 60
new_elements.append(text(px, py, "PREFILL (once per request)", size=18, color=DARK_YELLOW, width=400))
new_elements.append(text(px, py + 35, "Input: full prompt, many tokens", size=14, color=DARK_GRAY, width=400))
# Draw input as a row of blocks
for i in range(15):
    new_elements.append(rect(px + i * 28, py + 65, 25, 40, fill=BLUE, stroke=DARK_BLUE))
new_elements.append(text(px, py + 115, "shape: (1, N=15, D)   [N many tokens]",
                         size=13, color=DARK_GRAY, width=400, family=3))
new_elements.append(text(px, py + 145, "FLOPs ∝ N × 2·P  (big)", size=14, color=DARK_GRAY, width=400))
new_elements.append(text(px, py + 170, "HBM bytes ∝ 2·P   (same as decode)",
                         size=14, color=DARK_GRAY, width=400))
new_elements.append(text(px, py + 195, "Intensity ∝ N  →  compute-bound for N ≳ 330",
                         size=15, color=DARK_GREEN, width=500))
new_elements.append(rect(px - 10, py + 225, 550, 40, fill=YELLOW, stroke=DARK_YELLOW, opacity=50, rounded=True))
new_elements.append(text(px, py + 235, "→ saturates TENSOR CORES. Fast per-token.",
                         size=14, color=DARK_YELLOW, width=500))

# DECODE side
dx = 800
new_elements.append(text(dx, py, "DECODE (many times per request)", size=18, color=DARK_YELLOW, width=400))
new_elements.append(text(dx, py + 35, "Input: one token (the previous output)", size=14, color=DARK_GRAY, width=400))
# Draw input as a single block
new_elements.append(rect(dx, py + 65, 25, 40, fill=GREEN, stroke=DARK_GREEN))
new_elements.append(text(dx + 35, py + 75, "← just 1 token", size=13, color=DARK_GRAY, width=300))
new_elements.append(text(dx, py + 115, "shape: (1, 1, D)",
                         size=13, color=DARK_GRAY, width=400, family=3))
new_elements.append(text(dx, py + 145, "FLOPs ∝ 2·P  (small)", size=14, color=DARK_GRAY, width=400))
new_elements.append(text(dx, py + 170, "HBM bytes ∝ 2·P   (same weights to read)",
                         size=14, color=DARK_GRAY, width=400))
new_elements.append(text(dx, py + 195, "Intensity = 1  →  memory-bound, 330× idle",
                         size=15, color=DARK_RED, width=500))
new_elements.append(rect(dx - 10, py + 225, 550, 40, fill=RED, stroke=DARK_RED, opacity=50, rounded=True))
new_elements.append(text(dx, py + 235, "→ waiting on HBM. Slow per-token.",
                         size=14, color=DARK_RED, width=500))


# ============================================================================
# Scene 8: Prefix sharing across sequences
# ============================================================================
y0 = 2680
new_elements.extend(frame(70, y0, 1400, 360, "PREFIX SHARING (read-only)", DARK_GREEN, "#ebfbee"))
new_elements.append(text(90, y0 + 10,
                         "3 users sharing a common system prompt — 3 block tables, 1 physical copy of the prompt's KV.",
                         size=17, color=DARK_GREEN, width=1300))

# Physical pool row
pool_y = y0 + 200
new_elements.append(text(100, pool_y - 30, "Physical block pool", size=14, color=DARK_GRAY, width=200))
pool_colors = [PURPLE, PURPLE, PURPLE, BLUE, GREEN, YELLOW, GRAY, GRAY]
pool_labels = ["P0", "P1", "P2", "A3", "B3", "C3", "—", "—"]
for i, (c, lbl) in enumerate(zip(pool_colors, pool_labels)):
    x = 100 + i * 80
    new_elements.append(rect(x, pool_y, 75, 50, fill=c, stroke=DARK_GRAY))
    new_elements.append(text(x + 30, pool_y + 15, lbl, size=14, color=DARK_GRAY, width=40))
new_elements.append(text(740, pool_y + 60, "P0..P2 = shared prompt (3 blocks) — 1 copy",
                         size=13, color=DARK_PURPLE, width=500))
new_elements.append(text(740, pool_y + 80, "A3, B3, C3 = per-user continuations",
                         size=13, color=DARK_GRAY, width=500))

# Three block tables above
tbl_y = y0 + 60
seqs = [("Seq A", DARK_BLUE, BLUE, "A3"),
        ("Seq B", DARK_GREEN, GREEN, "B3"),
        ("Seq C", DARK_YELLOW, YELLOW, "C3")]
for i, (name, dark, light, own) in enumerate(seqs):
    x = 100 + i * 450
    new_elements.append(text(x, tbl_y, name + " block_table", size=15, color=dark, width=200))
    # 4 entries: P0, P1, P2, own
    entries = ["P0", "P1", "P2", own]
    entry_colors = [PURPLE, PURPLE, PURPLE, light]
    for j, (e, ec) in enumerate(zip(entries, entry_colors)):
        bx = x + j * 60
        new_elements.append(rect(bx, tbl_y + 25, 55, 40, fill=ec, stroke=DARK_GRAY))
        new_elements.append(text(bx + 20, tbl_y + 35, e, size=13, color=DARK_GRAY, width=30))


# ============================================================================
# Scene 9: Copy-on-write
# ============================================================================
y0 = 3080
new_elements.extend(frame(70, y0, 1400, 420, "COPY-ON-WRITE (branching within one request)", DARK_GRAY, "#f8f9fa"))
new_elements.append(text(90, y0 + 10,
                         "Beam search / parallel sampling: n branches share prompt blocks, each generates different tokens.",
                         size=17, color=DARK_GRAY, width=1300))

panel_w = 430
panel_y = y0 + 60
panel_h = 320

for panel_i, (title, subtitle, bg) in enumerate([
    ("1. Shared (reading only)", "Both branches point at P0", BLUE),
    ("2. Branch A wants to write", "P0 is shared. Copy-on-write.", YELLOW),
    ("3. After CoW", "Branches now have independent tail blocks", GREEN),
]):
    px = 100 + panel_i * panel_w
    new_elements.append(rect(px, panel_y, panel_w - 20, panel_h, fill="#ffffff",
                             stroke=DARK_GRAY, rounded=True))
    new_elements.append(text(px + 15, panel_y + 15, title, size=15, color=DARK_GRAY, width=panel_w - 40))
    new_elements.append(text(px + 15, panel_y + 38, subtitle, size=12, color=DARK_GRAY, width=panel_w - 40))

    # Common layout: two block tables + shared pool
    tbl_y = panel_y + 70
    # Branch A
    new_elements.append(text(px + 15, tbl_y, "Branch A:", size=13, color=DARK_BLUE, width=100))
    # Branch B
    new_elements.append(text(px + 15, tbl_y + 45, "Branch B:", size=13, color=DARK_GREEN, width=100))

    pool_y2 = tbl_y + 120
    new_elements.append(text(px + 15, pool_y2 - 20, "Pool:", size=13, color=DARK_GRAY, width=100))

    if panel_i == 0:
        # Both point at P0
        a_ids = ["P0"]
        b_ids = ["P0"]
        pool = [("P0", BLUE, 2)]  # (id, color, refcount)
    elif panel_i == 1:
        # A about to write into P0. Marked red/alert.
        a_ids = ["P0*"]   # * = pending write
        b_ids = ["P0"]
        pool = [("P0", YELLOW, 2)]
    else:
        # CoW done: A now has P0' (copy), B still has P0
        a_ids = ["P0'"]
        b_ids = ["P0"]
        pool = [("P0", GREEN, 1), ("P0'", GREEN, 1)]

    for j, e in enumerate(a_ids):
        new_elements.append(rect(px + 110 + j * 60, tbl_y - 5, 55, 30, fill=BLUE, stroke=DARK_BLUE))
        new_elements.append(text(px + 110 + j * 60 + 15, tbl_y + 3, e, size=12, color=DARK_BLUE, width=40))
    for j, e in enumerate(b_ids):
        new_elements.append(rect(px + 110 + j * 60, tbl_y + 40, 55, 30, fill=GREEN, stroke=DARK_GREEN))
        new_elements.append(text(px + 110 + j * 60 + 15, tbl_y + 48, e, size=12, color=DARK_GREEN, width=40))
    for j, (pid, pcol, refc) in enumerate(pool):
        new_elements.append(rect(px + 75 + j * 80, pool_y2, 70, 40, fill=pcol, stroke=DARK_GRAY))
        new_elements.append(text(px + 85 + j * 80, pool_y2 + 8, pid, size=12, color=DARK_GRAY, width=50))
        new_elements.append(text(px + 85 + j * 80, pool_y2 + 24, f"rc={refc}", size=10, color=DARK_GRAY, width=50))

    # Note below
    notes = [
        "One physical copy, two readers. refcount=2.",
        "Allocator sees refcount>1 → triggers CoW.",
        "Alloc new block, copy data, decrement old refcount.",
    ]
    new_elements.append(text(px + 15, panel_y + panel_h - 60, notes[panel_i], size=12, color=DARK_GRAY, width=panel_w - 40))


# ============================================================================
# Scene 10: Fragmentation over time (extends the earlier external-frag)
# ============================================================================
y0 = 3540
new_elements.extend(frame(70, y0, 1400, 420, "FRAGMENTATION OVER TIME — naive vs paged", DARK_RED, "#fff5f5"))
new_elements.append(text(90, y0 + 10,
                         "Same churn pattern (admit/free sequences) plays out differently in the two designs.",
                         size=17, color=DARK_RED, width=1300))

# NAIVE timeline (4 snapshots)
frames = [
    ("t=0: 10 seqs admitted", ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"], [1]*10),
    ("t=5: S3, S7 done", ["S1","S2","—","S4","S5","S6","—","S8","S9","S10"], [1,1,0,1,1,1,0,1,1,1]),
    ("t=8: S2, S6 done too", ["S1","—","—","S4","S5","—","—","S8","S9","S10"], [1,0,0,1,1,0,0,1,1,1]),
    ("t=10: want admit seq needing 2.5 'slots'", ["S1","?","?","S4","S5","?","?","S8","S9","S10"], [1,0,0,1,1,0,0,1,1,1]),
]
# naive
sy = y0 + 55
new_elements.append(text(100, sy + 5, "NAIVE (contiguous, fixed-size 1 GB slabs)",
                         size=15, color=DARK_RED, width=500))
for fi, (title, labels, busy) in enumerate(frames):
    fy = sy + 30 + fi * 75
    new_elements.append(text(100, fy + 5, title, size=12, color=DARK_GRAY, width=300))
    for si in range(10):
        x = 400 + si * 45
        color = BLUE if busy[si] else "#ffffff"
        stroke = DARK_BLUE if busy[si] else DARK_RED
        new_elements.append(rect(x, fy, 40, 50, fill=color, stroke=stroke,
                                 dashed=not busy[si]))
        new_elements.append(text(x + 5, fy + 15, labels[si], size=11,
                                 color=DARK_BLUE if busy[si] else DARK_RED, width=35))
    if fi == 3:
        new_elements.append(text(870, fy + 15,
                                 "New seq needs 2.5 consecutive slabs. Gaps are size-1 each. Doesn't fit.",
                                 size=12, color=DARK_RED, width=500))


# paged side-by-side — show 1 frame emphasizing "16-block pool, scattered allocation is fine"
# Actually place PAGED below the naive summary
py = sy + 340
new_elements.append(text(100, py, "PAGED (16 uniform blocks, scattered allocation)",
                         size=15, color=DARK_GREEN, width=500))
# A simple flat row of 16 blocks showing "same churn" → no fragmentation issue
pool_y2 = py + 30
new_elements.append(text(100, pool_y2 + 15, "After same churn:", size=12, color=DARK_GRAY, width=200))
busy_paged = [1,1,0,0,1,1,1,1,0,0,1,1,1,0,1,0]
labels_paged = ["","",None,None,"","","","",None,None,"","","",None,"",None]
for i in range(16):
    x = 400 + i * 40
    color = GREEN if busy_paged[i] else "#ffffff"
    stroke = DARK_GREEN if busy_paged[i] else "#ced4da"
    new_elements.append(rect(x, pool_y2, 35, 40, fill=color, stroke=stroke, dashed=not busy_paged[i]))

new_elements.append(text(100, pool_y2 + 60, "New seq needs 5 blocks → grab any 5 free blocks. Total free ≥ need ⇒ always fits.",
                         size=13, color=DARK_GREEN, width=1200))


# ============================================================================
# Scene 11: Architecture overview (scheduler + pool + sequences)
# ============================================================================
y0 = 4020
new_elements.extend(frame(70, y0, 1400, 440, "vLLM ARCHITECTURE — putting it all together", "#1e1e1e", "#ffffff"))
new_elements.append(text(90, y0 + 10, "Request queue → Scheduler → Model forward pass. Block pool supplies KV cache memory.",
                         size=17, color="#1e1e1e", width=1300))

# Request queue (top-left)
rq_x, rq_y = 120, y0 + 60
new_elements.append(rect(rq_x, rq_y, 240, 140, fill=YELLOW, stroke=DARK_YELLOW, rounded=True))
new_elements.append(text(rq_x + 15, rq_y + 10, "Request queue", size=16, color=DARK_YELLOW, width=220))
new_elements.append(text(rq_x + 15, rq_y + 40, "→ req 4 (waiting)", size=13, color=DARK_GRAY, width=220))
new_elements.append(text(rq_x + 15, rq_y + 60, "→ req 5 (waiting)", size=13, color=DARK_GRAY, width=220))
new_elements.append(text(rq_x + 15, rq_y + 80, "→ req 6 (waiting)", size=13, color=DARK_GRAY, width=220))
new_elements.append(text(rq_x + 15, rq_y + 110, "new request from user", size=11, color=DARK_GRAY, width=220))

# Scheduler (center-top)
sch_x, sch_y = 500, y0 + 60
new_elements.append(rect(sch_x, sch_y, 280, 140, fill=PURPLE, stroke=DARK_PURPLE, rounded=True))
new_elements.append(text(sch_x + 15, sch_y + 10, "Scheduler", size=16, color=DARK_PURPLE, width=260))
new_elements.append(text(sch_x + 15, sch_y + 40, "each step:", size=13, color=DARK_GRAY, width=260))
new_elements.append(text(sch_x + 15, sch_y + 60, "• admit if blocks available", size=12, color=DARK_GRAY, width=260))
new_elements.append(text(sch_x + 15, sch_y + 80, "• build batch of active seqs", size=12, color=DARK_GRAY, width=260))
new_elements.append(text(sch_x + 15, sch_y + 100, "• call model.forward(batch)", size=12, color=DARK_GRAY, width=260))
new_elements.append(text(sch_x + 15, sch_y + 120, "• free blocks of finished seqs", size=12, color=DARK_GRAY, width=260))

# Arrow request → scheduler
new_elements.append(arrow(rq_x + 240, rq_y + 70, sch_x, sch_y + 70, DARK_GRAY))

# Model forward (right-top)
mf_x, mf_y = 920, y0 + 60
new_elements.append(rect(mf_x, mf_y, 440, 140, fill=BLUE, stroke=DARK_BLUE, rounded=True))
new_elements.append(text(mf_x + 15, mf_y + 10, "Model forward pass", size=16, color=DARK_BLUE, width=400))
new_elements.append(text(mf_x + 15, mf_y + 40, "for each sequence in batch:", size=13, color=DARK_GRAY, width=400))
new_elements.append(text(mf_x + 15, mf_y + 60, "  attention reads K, V via its block_table", size=12, color=DARK_GRAY, width=400))
new_elements.append(text(mf_x + 15, mf_y + 80, "  writes new K, V to allocated block", size=12, color=DARK_GRAY, width=400))
new_elements.append(text(mf_x + 15, mf_y + 110, "(Triton PagedAttention kernel, Phase 3)", size=11, color=DARK_GRAY, width=400))

# Arrow scheduler → model
new_elements.append(arrow(sch_x + 280, sch_y + 70, mf_x, mf_y + 70, DARK_GRAY))

# Active sequences row
as_y = y0 + 230
new_elements.append(text(120, as_y, "Active sequences (their block tables)", size=15, color=DARK_GRAY, width=500))
active = [
    ("req 1", BLUE, DARK_BLUE, ["P4","P7","P2"]),
    ("req 2", GREEN, DARK_GREEN, ["P0","P9"]),
    ("req 3", YELLOW, DARK_YELLOW, ["P5","P1","P8","P11"]),
]
for i, (name, c, dc, blocks) in enumerate(active):
    x = 120 + i * 440
    y = as_y + 30
    new_elements.append(rect(x, y, 100, 40, fill=c, stroke=dc, rounded=True))
    new_elements.append(text(x + 15, y + 10, name, size=14, color=dc, width=80))
    # block table entries
    for j, b in enumerate(blocks):
        bx = x + 110 + j * 50
        new_elements.append(rect(bx, y + 5, 45, 30, fill=c, stroke=dc))
        new_elements.append(text(bx + 7, y + 12, b, size=12, color=dc, width=35))

# Block pool at bottom
bp_y = y0 + 340
new_elements.append(text(120, bp_y - 5, "BlockPool (physical blocks) — shared by all sequences",
                         size=15, color=DARK_GRAY, width=600))
for i in range(16):
    x = 120 + i * 80
    owned_by = None
    for name, c, dc, blocks in active:
        if f"P{i}" in blocks:
            owned_by = (c, dc, name)
            break
    if owned_by:
        color, dark, owner = owned_by
    else:
        color, dark, owner = GRAY, DARK_GRAY, ""
    new_elements.append(rect(x, bp_y + 20, 75, 50, fill=color, stroke=dark))
    new_elements.append(text(x + 25, bp_y + 28, f"P{i}", size=12, color=dark, width=40))
    if owner:
        new_elements.append(text(x + 20, bp_y + 48, owner, size=10, color=dark, width=50))
    else:
        new_elements.append(text(x + 25, bp_y + 48, "free", size=10, color=DARK_GRAY, width=50))


# ============================================================================
# Load, append, write
# ============================================================================
with open(PATH) as f:
    doc = json.load(f)

doc["elements"].extend(new_elements)

with open(PATH, "w") as f:
    json.dump(doc, f, indent=2)

print(f"Appended {len(new_elements)} elements. Total now: {len(doc['elements'])}")
