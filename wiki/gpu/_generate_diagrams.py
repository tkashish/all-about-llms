"""Generate the GPU fundamentals Excalidraw file.

Rewrites gpu-fundamentals.excalidraw from scratch. Add new scenes by
appending a new `scene_*` function and calling it in main().

Run: uv run python wiki/gpu/_generate_diagrams.py
"""
import json
from pathlib import Path

PATH = Path(__file__).parent / "gpu-fundamentals.excalidraw"

# ============================================================================
# Element helpers (mirror vllm-phase1 generator for consistency)
# ============================================================================
_seed = [1000]
def next_seed():
    _seed[0] += 1
    return _seed[0]

def rect(x, y, w, h, fill="#ffffff", stroke="#1e1e1e", fill_style="solid",
         opacity=100, dashed=False, rounded=False):
    return {
        "id": f"r{next_seed()}", "type": "rectangle",
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
        "id": f"t{next_seed()}", "type": "text",
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
        "id": f"a{next_seed()}", "type": "arrow",
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
    els = [rect(x, y, w, h, fill=bg, stroke=color, opacity=30, rounded=True)]
    els.append(text(x + w - 360, y + 8, label, size=14, color=color, width=350))
    return els

# Color palette
BLUE, DARK_BLUE       = "#a5d8ff", "#1971c2"
GREEN, DARK_GREEN     = "#b2f2bb", "#2b8a3e"
RED, DARK_RED         = "#ffc9c9", "#c92a2a"
YELLOW, DARK_YELLOW   = "#fff3bf", "#e67700"
PURPLE, DARK_PURPLE   = "#d0bfff", "#6741d9"
GRAY, DARK_GRAY       = "#e9ecef", "#495057"
ORANGE, DARK_ORANGE   = "#ffd8a8", "#d9480f"


# ============================================================================
# Scene 1 — GPU hardware hierarchy (L4 example)
# ============================================================================
def scene_hardware_hierarchy(y0):
    els = []
    els.extend(frame(60, y0, 1500, 460,
                     "GPU HARDWARE HIERARCHY — L4 example (60 SMs, 24 GB HBM)",
                     DARK_BLUE, "#e7f5ff"))
    els.append(text(80, y0 + 10,
                    "Scope: what the GPU physically IS. Static structure. "
                    "Memory is a hierarchy; compute is a grid of SMs.",
                    size=14, color=DARK_BLUE, width=1400))

    # HBM (top)
    els.append(rect(120, y0 + 50, 1380, 50, fill=ORANGE, stroke=DARK_ORANGE, rounded=True))
    els.append(text(140, y0 + 60, "HBM — 24 GB, ~300 GB/s  (all model weights + KV cache live here)",
                    size=16, color=DARK_ORANGE, width=1300))

    # L2 cache
    els.append(rect(220, y0 + 120, 1180, 40, fill=YELLOW, stroke=DARK_YELLOW, rounded=True))
    els.append(text(240, y0 + 128, "L2 cache — ~48 MB, hardware-managed (automatic)",
                    size=14, color=DARK_YELLOW, width=1100))

    # SMs
    els.append(text(120, y0 + 180, "60 × SM (Streaming Multiprocessor):",
                    size=14, color=DARK_GRAY, width=400))
    for i in range(12):
        x = 120 + i * 115
        els.append(rect(x, y0 + 210, 105, 40, fill=BLUE, stroke=DARK_BLUE, rounded=True))
        els.append(text(x + 30, y0 + 220, f"SM {i}", size=13, color=DARK_BLUE, width=60))
    els.append(text(120 + 12 * 115, y0 + 218, "...", size=18, color=DARK_GRAY, width=50))

    # Zoom-in of one SM
    els.append(text(120, y0 + 275, "Zoom into one SM:",
                    size=15, color=DARK_GRAY, width=300))

    zx, zy = 120, y0 + 310
    els.append(rect(zx, zy, 1380, 130, fill="#ffffff", stroke=DARK_BLUE, rounded=True))
    els.append(text(zx + 15, zy + 10, "SM internals", size=14, color=DARK_BLUE, width=300))
    # Four sub-boxes
    sub_w = 320
    subs = [
        ("128 CUDA cores", "general-purpose arithmetic\n(add, mul, compare, shift, ...)", GREEN, DARK_GREEN),
        ("4 Tensor cores", "matmul-only, fp16/bf16/int8/fp8\n(~50x matmul throughput)", PURPLE, DARK_PURPLE),
        ("Registers (~64 KB)", "per-thread, fastest\n(effectively free to read/write)", RED, DARK_RED),
        ("SRAM / shared mem (~128 KB)", "per-SM, manually-managed\n(what FlashAttention uses)", YELLOW, DARK_YELLOW),
    ]
    for i, (title, body, c, dc) in enumerate(subs):
        x = zx + 20 + i * (sub_w + 15)
        els.append(rect(x, zy + 40, sub_w, 75, fill=c, stroke=dc, rounded=True))
        els.append(text(x + 10, zy + 48, title, size=13, color=dc, width=sub_w - 20))
        els.append(text(x + 10, zy + 70, body, size=11, color=DARK_GRAY, width=sub_w - 20))

    # Arrows for memory hierarchy
    els.append(text(30, y0 + 70, "fastest\n↑\n\n\n\n\n\nslowest",
                    size=11, color=DARK_GRAY, width=80))

    return els


# ============================================================================
# Scene 2 — Execution model (grid → block → warp → thread)
# ============================================================================
def scene_execution_model(y0):
    els = []
    els.extend(frame(60, y0, 1500, 430,
                     "EXECUTION MODEL — how your kernel runs on the hardware",
                     DARK_GREEN, "#ebfbee"))
    els.append(text(80, y0 + 10,
                    "Scope: logical, per-kernel. You pick grid and block size; GPU handles the rest.",
                    size=14, color=DARK_GREEN, width=1400))

    # Level 1: Kernel launch
    els.append(rect(100, y0 + 50, 1400, 70, fill=YELLOW, stroke=DARK_YELLOW, rounded=True))
    els.append(text(120, y0 + 58, "1. You launch:  kernel[grid_size](args, BLOCK_SIZE=...)",
                    size=16, color=DARK_YELLOW, width=900, family=3))
    els.append(text(120, y0 + 85, "e.g. grid_size=(3907,), BLOCK_SIZE=256  →  3907 blocks, 256 threads each, 1M total threads",
                    size=13, color=DARK_GRAY, width=1300))

    els.append(arrow(780, y0 + 125, 780, y0 + 145, DARK_GRAY))

    # Level 2: GPU scheduler → SMs
    els.append(rect(100, y0 + 150, 1400, 70, fill=BLUE, stroke=DARK_BLUE, rounded=True))
    els.append(text(120, y0 + 158, "2. GPU scheduler distributes blocks across SMs (dynamic, hardware):",
                    size=15, color=DARK_BLUE, width=900))
    els.append(text(120, y0 + 185, "each SM holds ~16 blocks resident; queues the rest; refills as blocks finish. "
                                   "Block stays on one SM for its lifetime.",
                    size=13, color=DARK_GRAY, width=1300))

    els.append(arrow(780, y0 + 225, 780, y0 + 245, DARK_GRAY))

    # Level 3: Inside an SM — warps → cores
    els.append(rect(100, y0 + 250, 1400, 160, fill=PURPLE, stroke=DARK_PURPLE, rounded=True))
    els.append(text(120, y0 + 260, "3. Inside an SM — per cycle:",
                    size=15, color=DARK_PURPLE, width=400))
    els.append(text(120, y0 + 285, "• Block's 256 threads → grouped into 8 warps of 32",
                    size=13, color=DARK_GRAY, width=800))
    els.append(text(120, y0 + 305, "• 4 warp schedulers → pick 4 warps to run this cycle",
                    size=13, color=DARK_GRAY, width=800))
    els.append(text(120, y0 + 325, "• Each active warp's 32 threads → use 32 CUDA cores simultaneously",
                    size=13, color=DARK_GRAY, width=800))
    els.append(text(120, y0 + 345, "• 4 warps × 32 cores = 128 CUDA cores busy per cycle (all of them)",
                    size=13, color=DARK_GRAY, width=800))
    els.append(text(120, y0 + 370, "Stalled warps (waiting on memory) → scheduler swaps in another resident warp.\n"
                                   "This is how the GPU hides memory latency — with occupancy.",
                    size=12, color=DARK_PURPLE, width=1200))

    # Side panel: what you vs GPU control
    bx = 1060
    by = y0 + 250
    els.append(rect(bx, by, 440, 160, fill="#ffffff", stroke=DARK_GRAY, rounded=True))
    els.append(text(bx + 15, by + 10, "Programmer vs GPU", size=14, color=DARK_GRAY, width=400))
    els.append(text(bx + 15, by + 35, "YOU control:", size=12, color=DARK_BLUE, width=200))
    els.append(text(bx + 15, by + 55, "  • grid size", size=12, color=DARK_GRAY, width=200))
    els.append(text(bx + 15, by + 72, "  • block size", size=12, color=DARK_GRAY, width=200))
    els.append(text(bx + 15, by + 89, "  • what each thread touches", size=12, color=DARK_GRAY, width=250))
    els.append(text(bx + 15, by + 110, "GPU handles:", size=12, color=DARK_GREEN, width=200))
    els.append(text(bx + 15, by + 130, "  • blocks → SMs, warps → cores", size=12, color=DARK_GRAY, width=300))
    els.append(text(bx + 15, by + 147, "  • latency hiding, scheduling", size=12, color=DARK_GRAY, width=300))

    return els


# ============================================================================
# Scene 3 — Cache lines and memory coalescing
# ============================================================================
def scene_coalescing(y0):
    els = []
    els.extend(frame(60, y0, 1500, 490,
                     "MEMORY COALESCING — cache lines decide everything",
                     DARK_RED, "#fff5f5"))
    els.append(text(80, y0 + 10,
                    "HBM delivers memory in 128-byte cache lines. The # of cache lines your warp touches "
                    "= # of memory transactions = your effective bandwidth.",
                    size=14, color=DARK_RED, width=1400))

    # Illustrate: a cache line as a band of 32 slots (each slot = 4 bytes)
    def draw_cache_line(x, y, w, label, color=BLUE, highlight=None):
        # Highlight: list of slot indices to mark as "requested"
        result = []
        slots = 32
        slot_w = w / slots
        for i in range(slots):
            fill = DARK_RED if (highlight and i in highlight) else color
            result.append(rect(x + i * slot_w, y, slot_w - 0.5, 30,
                               fill=fill, stroke=DARK_GRAY))
        result.append(text(x, y - 18, label, size=12, color=DARK_GRAY, width=400))
        return result

    # Scenario 1: contiguous (100%)
    y = y0 + 50
    els.append(text(80, y, "Scenario 1 — CONTIGUOUS (stride 1)",
                    size=16, color=DARK_GREEN, width=600))
    els.append(text(80, y + 22, "Warp wants 32 floats at addresses 0..124. Fits in 1 cache line.",
                    size=12, color=DARK_GRAY, width=700))
    els.extend(draw_cache_line(80, y + 60, 900, "Cache line (128 B)",
                               color=GREEN, highlight=list(range(32))))
    els.append(text(1000, y + 60, "Fetched: 128 B\nUsed: 128 B\n→ 100% efficient, 1 transaction",
                    size=13, color=DARK_GREEN, width=400))

    # Scenario 2: stride 2 (50%)
    y = y0 + 160
    els.append(text(80, y, "Scenario 2 — STRIDE 2 (every other element)",
                    size=16, color=DARK_YELLOW, width=600))
    els.append(text(80, y + 22, "Warp wants 32 floats at addresses 0, 8, 16, ..., 248. Spans 2 cache lines.",
                    size=12, color=DARK_GRAY, width=700))
    # Two cache lines stacked. In each, 16 slots highlighted (every other).
    hi = list(range(0, 32, 2))
    els.extend(draw_cache_line(80, y + 60, 430, "Line 1 (0..124)", color=YELLOW, highlight=hi))
    els.extend(draw_cache_line(560, y + 60, 430, "Line 2 (128..252)", color=YELLOW, highlight=hi))
    els.append(text(1000, y + 60, "Fetched: 256 B\nUsed: 128 B\n→ 50% efficient, 2 transactions",
                    size=13, color=DARK_YELLOW, width=400))

    # Scenario 3: scattered (3%)
    y = y0 + 280
    els.append(text(80, y, "Scenario 3 — SCATTERED (stride 1000+ — e.g. wrong axis of 2D matrix)",
                    size=16, color=DARK_RED, width=700))
    els.append(text(80, y + 22, "Each thread's element is in a different cache line. 32 separate transactions.",
                    size=12, color=DARK_GRAY, width=700))
    # Show 4 cache lines side by side, each with 1 highlighted slot + "..."
    for i in range(4):
        x = 80 + i * 220
        els.extend(draw_cache_line(x, y + 60, 200, f"Line {i+1}", color=RED, highlight=[i * 2]))
    els.append(text(970, y + 70, "…  (32 total lines)", size=14, color=DARK_GRAY, width=200))
    els.append(text(1200, y + 60, "Fetched: 4096 B\nUsed: 128 B\n→ ~3% efficient",
                    size=13, color=DARK_RED, width=300))

    # Summary
    y = y0 + 410
    els.append(rect(80, y, 1420, 60, fill="#ffffff", stroke=DARK_GRAY, rounded=True))
    els.append(text(95, y + 10,
                    "RULE: Threads 0..31 of a warp should touch CONTIGUOUS memory slots.",
                    size=15, color=DARK_GRAY, width=1200))
    els.append(text(95, y + 34,
                    "In Triton: offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  →  coalesced by construction.",
                    size=13, color=DARK_GREEN, width=1300, family=3))
    return els


# ============================================================================
# Scene 4 — Triton vector_add walkthrough
# ============================================================================
def scene_vector_add(y0):
    els = []
    els.extend(frame(60, y0, 1500, 460,
                     "TRITON VECTOR ADD — your first kernel",
                     DARK_PURPLE, "#f3f0ff"))
    els.append(text(80, y0 + 10,
                    "Mental model: each 'program' (= block) processes BLOCK_SIZE contiguous elements as a vector.",
                    size=14, color=DARK_PURPLE, width=1400))

    # LEFT: caller side
    cx = 80
    cy = y0 + 50
    els.append(rect(cx, cy, 660, 400, fill="#ffffff", stroke=DARK_BLUE, rounded=True))
    els.append(text(cx + 15, cy + 10, "CALLER (Python, CPU side)",
                    size=15, color=DARK_BLUE, width=500))
    caller_code = [
        "x = torch.randn(1000, device='cuda')",
        "y = torch.randn(1000, device='cuda')",
        "out = torch.empty_like(x)",
        "N = x.numel()    # 1000",
        "",
        "grid = (triton.cdiv(N, 256),)",
        "# = (4,) — 4 blocks for 1000 elements",
        "",
        "vector_add[grid](x, y, out, N,",
        "                 BLOCK_SIZE=256)",
    ]
    for i, line in enumerate(caller_code):
        els.append(text(cx + 20, cy + 45 + i * 22, line,
                        size=13, color=DARK_GRAY, width=600, family=3))

    # Notes under caller
    els.append(text(cx + 20, cy + 290,
                    "• PyTorch tensor → GPU memory pointer (automatic)",
                    size=12, color=DARK_GRAY, width=600))
    els.append(text(cx + 20, cy + 310,
                    "• grid = (4,) means 4 program instances will run",
                    size=12, color=DARK_GRAY, width=600))
    els.append(text(cx + 20, cy + 330,
                    "• BLOCK_SIZE is tl.constexpr — compile-time constant",
                    size=12, color=DARK_GRAY, width=600))
    els.append(text(cx + 20, cy + 350,
                    "• triton.cdiv = ceil divide: handles N not divisible by block size",
                    size=12, color=DARK_GRAY, width=600))

    # RIGHT: kernel side
    kx = cx + 680
    ky = cy
    els.append(rect(kx, ky, 720, 400, fill="#ffffff", stroke=DARK_PURPLE, rounded=True))
    els.append(text(kx + 15, ky + 10, "KERNEL (on GPU, runs per block)",
                    size=15, color=DARK_PURPLE, width=600))
    kernel_code = [
        "@triton.jit",
        "def vector_add(x_ptr, y_ptr, out_ptr, n,",
        "               BLOCK_SIZE: tl.constexpr):",
        "    pid = tl.program_id(0)",
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "    mask = offsets < n",
        "",
        "    x = tl.load(x_ptr + offsets, mask=mask)",
        "    y = tl.load(y_ptr + offsets, mask=mask)",
        "    out = x + y",
        "    tl.store(out_ptr + offsets, out, mask=mask)",
    ]
    for i, line in enumerate(kernel_code):
        els.append(text(kx + 20, ky + 45 + i * 22, line,
                        size=13, color=DARK_GRAY, width=680, family=3))

    # Block processing picture below kernel
    els.append(text(kx + 20, ky + 305, "How the 4 blocks work:",
                    size=13, color=DARK_PURPLE, width=400))
    block_info = [
        ("Block pid=0", "offsets [0..255]", GREEN),
        ("Block pid=1", "offsets [256..511]", GREEN),
        ("Block pid=2", "offsets [512..767]", GREEN),
        ("Block pid=3", "offsets [768..1023] — last 24 masked out", YELLOW),
    ]
    for i, (name, rng, c) in enumerate(block_info):
        x = kx + 20
        y = ky + 330 + i * 18
        els.append(rect(x, y, 12, 12, fill=c, stroke=DARK_GRAY))
        els.append(text(x + 18, y - 2, f"{name}: {rng}",
                        size=11, color=DARK_GRAY, width=600, family=3))

    return els


# ============================================================================
# Scene 5 — SM structure: schedulers × warps × cores
# ============================================================================
def scene_sm_structure(y0):
    els = []
    els.extend(frame(60, y0, 1500, 720,
                     "SM STRUCTURE — schedulers × warps × cores",
                     DARK_BLUE, "#e7f5ff"))
    els.append(text(80, y0 + 10,
                    "Inside one SM: 4 warp schedulers, each with 12 warps, each warp has 32 threads that map to 32 CUDA cores.",
                    size=14, color=DARK_BLUE, width=1400))

    # Key statement box
    ksy = y0 + 45
    els.append(rect(80, ksy, 1420, 42, fill=YELLOW, stroke=DARK_YELLOW, rounded=True))
    els.append(text(95, ksy + 5,
                    "Each cycle, each scheduler picks ONE ready warp and issues ONE instruction.",
                    size=13, color=DARK_YELLOW, width=1400))
    els.append(text(95, ksy + 23,
                    "That instruction is EITHER a memory op (load/store to HBM/SRAM) OR a compute op (CUDA or Tensor core).",
                    size=13, color=DARK_YELLOW, width=1400))

    # ---- The 4 scheduler columns ----
    panel_y = ksy + 60
    panel_w = 345
    panel_h = 550
    colors = [
        (BLUE, DARK_BLUE),
        (GREEN, DARK_GREEN),
        (YELLOW, DARK_YELLOW),
        (PURPLE, DARK_PURPLE),
    ]
    for s in range(4):
        x = 80 + s * (panel_w + 15)
        c, dc = colors[s]

        # Panel frame
        els.append(rect(x, panel_y, panel_w, panel_h, fill="#ffffff", stroke=dc, rounded=True))

        # Header
        els.append(text(x + 12, panel_y + 10, f"Warp Scheduler {s}",
                        size=16, color=dc, width=300))

        # Warp pool (3 rows × 4 columns of warps, 12 total)
        els.append(text(x + 12, panel_y + 38, "Warp pool (12 resident):",
                        size=12, color=DARK_GRAY, width=300))
        for i in range(12):
            wi = s + i * 4
            col = i % 4
            row = i // 4
            wx = x + 12 + col * 82
            wy = panel_y + 60 + row * 32
            # Make W0 the "currently issuing" warp on scheduler 0 for illustration
            is_current = (s == 0 and i == 0)
            fill = ORANGE if is_current else c
            stroke = DARK_ORANGE if is_current else dc
            els.append(rect(wx, wy, 78, 26, fill=fill, stroke=stroke))
            els.append(text(wx + 5, wy + 5, f"W{wi}", size=11, color=stroke, width=60))
            if is_current:
                els.append(text(wx + 42, wy + 5, "← picked", size=9, color=DARK_ORANGE, width=50))

        # Arrow from pool down to "what it does"
        arrow_x = x + panel_w // 2
        els.append(arrow(arrow_x, panel_y + 170, arrow_x, panel_y + 195, DARK_GRAY))

        # "Current instruction" box
        inst_y = panel_y + 200
        els.append(rect(x + 20, inst_y, panel_w - 40, 70, fill=c, stroke=dc, rounded=True))
        els.append(text(x + 30, inst_y + 8,
                        f"W{s*4} issues 1 instruction" if s == 0 else
                        f"W{s} issues 1 instruction",
                        size=12, color=dc, width=280))
        if s == 0:
            els.append(text(x + 30, inst_y + 28, "(e.g. HBM load)",
                            size=11, color=DARK_RED, width=280))
            els.append(text(x + 30, inst_y + 45, "memory op → stalls 600–800 cyc",
                            size=10, color=DARK_RED, width=280))
        else:
            els.append(text(x + 30, inst_y + 28, "(e.g. fp32 add)",
                            size=11, color=DARK_GREEN, width=280))
            els.append(text(x + 30, inst_y + 45, "compute op → ~4 cyc",
                            size=10, color=DARK_GREEN, width=280))

        # Arrow down to cores
        els.append(arrow(arrow_x, inst_y + 75, arrow_x, inst_y + 100, DARK_GRAY))

        # 32 CUDA cores as 4x8 grid
        cores_y = inst_y + 110
        els.append(text(x + 20, cores_y, "32 CUDA cores (this scheduler's lane):",
                        size=11, color=DARK_GRAY, width=280))
        for i in range(32):
            col = i % 8
            row = i // 8
            cx = x + 20 + col * 38
            cy = cores_y + 20 + row * 24
            # If scheduler 0, show cores idle (memory op in flight, nothing to compute with yet)
            # Others: cores active
            if s == 0:
                els.append(rect(cx, cy, 34, 20, fill="#ffffff", stroke=DARK_RED, dashed=True))
            else:
                els.append(rect(cx, cy, 34, 20, fill=c, stroke=dc))

        # Annotation
        ann_y = cores_y + 125
        if s == 0:
            els.append(text(x + 20, ann_y,
                            "Memory op → no compute this cycle.\n"
                            "Cores in this lane are idle UNTIL\n"
                            "another ready warp replaces W0\n"
                            "next cycle.",
                            size=10, color=DARK_RED, width=300))
        else:
            els.append(text(x + 20, ann_y,
                            "Compute op → 32 threads use\n"
                            "the scheduler's 32 cores this cycle.\n"
                            "One thread per core.",
                            size=10, color=DARK_GRAY, width=300))

    # Summary box at the bottom
    sum_y = panel_y + panel_h + 15
    els.append(rect(80, sum_y, 1420, 70, fill="#ffffff", stroke=DARK_BLUE, rounded=True))
    els.append(text(95, sum_y + 8,
                    "PER SM, PER CYCLE — max throughput:",
                    size=14, color=DARK_BLUE, width=900))
    els.append(text(95, sum_y + 30,
                    "4 schedulers × 1 instruction each = 4 instructions.  "
                    "Compute instructions: 4 warps × 32 threads = 128 CUDA cores busy.",
                    size=13, color=DARK_GRAY, width=1300))
    els.append(text(95, sum_y + 50,
                    "Memory instructions: don't use cores — they issue a request and wait. Scheduler picks different warp next cycle.",
                    size=13, color=DARK_GRAY, width=1300))

    return els


# ============================================================================
# Scene 6 — Warp scheduling timeline (was Scene 5 before)
# ============================================================================
def scene_warp_scheduling(y0):
    els = []
    els.extend(frame(60, y0, 1500, 720,
                     "WARP SCHEDULING — occupancy and latency hiding",
                     DARK_ORANGE, "#fff4e6"))
    els.append(text(80, y0 + 10,
                    "Inside one SM: 4 schedulers, up to 48 resident warps, "
                    "~600–800 cycle HBM latency. Occupancy = how well you hide stalls.",
                    size=14, color=DARK_ORANGE, width=1400))

    # ---- Top: one SM, 4 schedulers, each with a warp pool ----
    sm_y = y0 + 55
    els.append(text(80, sm_y, "One SM — 4 warp schedulers, each manages up to 12 warps (warp_id % 4):",
                    size=14, color=DARK_GRAY, width=900))
    sm_y += 30
    pool_w = 340
    pool_h = 180
    for s in range(4):
        x = 80 + s * (pool_w + 15)
        c = [BLUE, GREEN, YELLOW, PURPLE][s]
        dc = [DARK_BLUE, DARK_GREEN, DARK_YELLOW, DARK_PURPLE][s]
        els.append(rect(x, sm_y, pool_w, pool_h, fill=c, stroke=dc, rounded=True))
        els.append(text(x + 10, sm_y + 10,
                        f"Scheduler {s}", size=14, color=dc, width=150))
        els.append(text(x + 10, sm_y + 32,
                        f"Drives 32 CUDA cores", size=11, color=DARK_GRAY, width=200))
        els.append(text(x + 10, sm_y + 52,
                        "Warp pool (12 resident):", size=11, color=DARK_GRAY, width=200))
        # 12 warp slots, 3x4 grid
        for i in range(12):
            wi = s + i * 4  # warp_id
            wx = x + 10 + (i % 4) * 80
            wy = sm_y + 75 + (i // 4) * 32
            # Mix of ready (bright) and stalled (grey)
            stalled = (i == 0 and s == 0) or (i == 3)
            fill = "#ffffff" if stalled else c
            els.append(rect(wx, wy, 75, 26, fill=fill, stroke=dc,
                            dashed=stalled))
            label = f"W{wi}"
            if stalled:
                label += " (wait)"
            els.append(text(wx + 6, wy + 5, label, size=10,
                            color=DARK_RED if stalled else dc, width=70))

    # Legend
    lg_y = sm_y + pool_h + 10
    els.append(rect(80, lg_y, 18, 14, fill=BLUE, stroke=DARK_BLUE))
    els.append(text(105, lg_y, "= ready warp (can run this cycle)",
                    size=11, color=DARK_GRAY, width=400))
    els.append(rect(400, lg_y, 18, 14, fill="#ffffff", stroke=DARK_RED, dashed=True))
    els.append(text(425, lg_y, "= stalled warp (waiting on memory)",
                    size=11, color=DARK_GRAY, width=400))

    # ---- Middle: per-cycle timeline ----
    tl_y = lg_y + 40
    els.append(text(80, tl_y, "Per-cycle timeline — each scheduler picks ONE ready warp to issue:",
                    size=14, color=DARK_GRAY, width=900))
    tl_y += 30
    cycles = 8
    col_w = 160
    row_h = 34
    # Header: cycle labels
    for c in range(cycles):
        els.append(text(180 + c * col_w, tl_y, f"cycle {c+1}",
                        size=12, color=DARK_GRAY, width=100))
    # 4 rows: one per scheduler
    for s in range(4):
        ry = tl_y + 25 + s * row_h
        els.append(text(80, ry + 6, f"Sched {s}",
                        size=13, color=DARK_GRAY, width=100))
        # Timeline: show what runs each cycle
        # Scheduler 0 starts with W0 (HBM load, stalls), then runs W4, W8, W4, W0 returns, etc.
        timelines = {
            0: ["W0→HBM", "W4 math", "W8 math", "W4 math", "W8 math", "W4 math", "(idle)", "W0 done"],
            1: ["W1 math", "W5 math", "W1 math", "W9 math", "W5 math", "W1 math", "W9 math", "W5 math"],
            2: ["W2 math", "W6 math", "W10 math", "W2 math", "W6 math", "W10 math", "W2 math", "W6 math"],
            3: ["W3 math", "W7 math", "W11 math", "W3 math", "W7 math", "W11 math", "W3 math", "W7 math"],
        }
        for c in range(cycles):
            x = 170 + c * col_w
            label = timelines[s][c]
            if "idle" in label:
                fill, stroke = "#ffffff", DARK_RED
            elif "HBM" in label:
                fill, stroke = ORANGE, DARK_ORANGE
            elif "done" in label:
                fill, stroke = GREEN, DARK_GREEN
            else:
                fill, stroke = [BLUE, GREEN, YELLOW, PURPLE][s], \
                               [DARK_BLUE, DARK_GREEN, DARK_YELLOW, DARK_PURPLE][s]
            els.append(rect(x, ry, col_w - 5, row_h - 4, fill=fill, stroke=stroke))
            els.append(text(x + 5, ry + 8, label, size=11,
                            color=stroke, width=col_w - 10))

    # Annotation below timeline
    note_y = tl_y + 25 + 4 * row_h + 15
    els.append(text(80, note_y,
                    "Scheduler 0 issued W0's HBM load at cycle 1. W0 is stalled ~700 cycles. "
                    "Sched 0 keeps its 32 cores busy by running W4 and W8 instead.",
                    size=12, color=DARK_GRAY, width=1400))
    els.append(text(80, note_y + 20,
                    "Cycle 7: all of sched 0's pool happens to be stalled (unusual — needs more warps) → "
                    "32 cores IDLE that cycle. This is the occupancy cost.",
                    size=12, color=DARK_RED, width=1400))

    # ---- Bottom: high vs low occupancy ----
    comp_y = note_y + 60
    els.append(text(80, comp_y, "High vs low occupancy:",
                    size=14, color=DARK_GRAY, width=300))
    # Two side-by-side panels
    for panel_i, (title, desc, note, color) in enumerate([
        ("HIGH OCCUPANCY",
         "Each scheduler has ~12 resident warps.\n"
         "When one stalls, another runs.\n"
         "128 cores busy almost every cycle.",
         "→ memory latency fully hidden",
         DARK_GREEN),
        ("LOW OCCUPANCY",
         "Each scheduler has only 1–2 warps.\n"
         "When one stalls, scheduler idles.\n"
         "Big gaps of 32+ idle cores.",
         "→ memory-bound, GPU underutilized",
         DARK_RED),
    ]):
        x = 80 + panel_i * 720
        els.append(rect(x, comp_y + 30, 700, 90,
                        fill="#ffffff", stroke=color, rounded=True))
        els.append(text(x + 15, comp_y + 40, title, size=15, color=color, width=300))
        els.append(text(x + 15, comp_y + 63, desc, size=12, color=DARK_GRAY, width=670))
        els.append(text(x + 15, comp_y + 105, note, size=12, color=color, width=670))

    return els


# ============================================================================
# Scene 7 — Vector add full trace: launch → SM → warps → cycles
# ============================================================================
def scene_vector_add_trace(y0):
    els = []
    els.extend(frame(60, y0, 1500, 850,
                     "VECTOR_ADD FULL TRACE — 4 blocks × 256 threads, on A10G (80 SMs)",
                     DARK_ORANGE, "#fff4e6"))
    els.append(text(80, y0 + 10,
                    "Kernel: out[i] = x[i] + y[i]. N=1000, BLOCK_SIZE=256. "
                    "grid=(4,), so 4 programs launched.",
                    size=14, color=DARK_ORANGE, width=1400))

    # ---- Row 1: Launch distribution (4 blocks → 4 SMs out of 80) ----
    r1_y = y0 + 50
    els.append(text(80, r1_y, "1. GRID LAUNCH — 4 blocks distributed to 4 SMs (76 SMs idle):",
                    size=14, color=DARK_GRAY, width=800))
    # Draw 80 SMs as a 10x8 grid, first 4 highlighted
    sm_size = 28
    sm_pad = 4
    grid_x = 80
    grid_y = r1_y + 30
    for i in range(80):
        col = i % 10
        row = i // 10
        x = grid_x + col * (sm_size + sm_pad)
        y = grid_y + row * (sm_size + sm_pad)
        if i < 4:
            els.append(rect(x, y, sm_size, sm_size, fill=ORANGE, stroke=DARK_ORANGE))
            els.append(text(x + 4, y + 7, f"SM{i}", size=9, color=DARK_ORANGE, width=24))
        else:
            els.append(rect(x, y, sm_size, sm_size, fill="#ffffff", stroke="#adb5bd",
                            dashed=True))

    # Annotation next to SM grid
    ann_x = grid_x + 10 * (sm_size + sm_pad) + 30
    els.append(rect(ann_x, r1_y + 30, 700, 80, fill="#ffffff", stroke=DARK_ORANGE, rounded=True))
    els.append(text(ann_x + 10, r1_y + 40,
                    "Orange = running one of our 4 blocks",
                    size=12, color=DARK_ORANGE, width=600))
    els.append(text(ann_x + 10, r1_y + 60,
                    "Dashed = idle (we have only 4 blocks for 80 SMs)",
                    size=12, color=DARK_GRAY, width=600))
    els.append(text(ann_x + 10, r1_y + 85,
                    "→ KERNEL UNDER-UTILIZES GPU. A real workload would have thousands of blocks.",
                    size=12, color=DARK_RED, width=650))

    # ---- Row 2: Inside one block (SM 0 running block 0) ----
    r2_y = r1_y + 340
    els.append(text(80, r2_y,
                    "2. INSIDE SM 0 — block 0 has 8 warps (256 threads), split across 4 schedulers:",
                    size=14, color=DARK_GRAY, width=1000))
    # 4 scheduler columns
    for s in range(4):
        x = 80 + s * 360
        y = r2_y + 30
        c = [BLUE, GREEN, YELLOW, PURPLE][s]
        dc = [DARK_BLUE, DARK_GREEN, DARK_YELLOW, DARK_PURPLE][s]
        els.append(rect(x, y, 340, 120, fill=c, stroke=dc, rounded=True))
        els.append(text(x + 15, y + 10, f"Scheduler {s}", size=14, color=dc, width=200))
        els.append(text(x + 15, y + 35,
                        f"Has 2 warps: W{s}, W{s+4}",
                        size=12, color=DARK_GRAY, width=250))
        # Warp boxes
        for wi, w in enumerate([s, s + 4]):
            wx = x + 15 + wi * 150
            wy = y + 60
            els.append(rect(wx, wy, 140, 40, fill="#ffffff", stroke=dc))
            els.append(text(wx + 10, wy + 6, f"W{w}", size=13, color=dc, width=100))
            els.append(text(wx + 10, wy + 22, "32 threads", size=10, color=DARK_GRAY, width=100))

    # ---- Row 3: Cycle-by-cycle trace for scheduler 0 ----
    r3_y = r2_y + 190
    els.append(text(80, r3_y,
                    "3. SCHEDULER 0 CYCLE TRACE — both W0 and W4 stall on LOAD, scheduler idles:",
                    size=14, color=DARK_GRAY, width=1200))

    # Table-style trace
    table_y = r3_y + 30
    cycles = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "...", "705"]
    # Rows: W0, W4, scheduler issues
    row_names = ["W0 status", "W4 status", "Sched 0 picks"]

    col_w_trace = 105
    row_h_trace = 30
    # Header
    for ci, c in enumerate(cycles):
        els.append(text(300 + ci * col_w_trace, table_y, f"cyc {c}",
                        size=11, color=DARK_GRAY, width=70))
    # Rows
    W0 = ["pid", "MUL", "ADD", "CMP", "LOADx", "stall", "stall", "stall", "stall", "stall", "...", "x back"]
    W4 = ["-",  "pid", "MUL", "ADD", "CMP",   "LOADx", "stall", "stall", "stall", "stall", "...", "stall"]
    pick = ["W0",  "W4", "W4", "W4", "W0",    "W4",    "IDLE",  "IDLE",  "IDLE",  "IDLE",  "...", "W0"]

    def row_color(entry):
        if entry == "IDLE":
            return "#ffffff", DARK_RED
        if "stall" in entry:
            return "#ffffff", DARK_RED
        if "LOAD" in entry:
            return ORANGE, DARK_ORANGE
        if "back" in entry:
            return GREEN, DARK_GREEN
        if "W" in entry:
            return YELLOW, DARK_YELLOW
        return BLUE, DARK_BLUE

    for ri, (name, row_data) in enumerate(zip(row_names, [W0, W4, pick])):
        ry = table_y + 25 + ri * row_h_trace
        els.append(text(80, ry + 6, name, size=13, color=DARK_GRAY, width=200))
        for ci, entry in enumerate(row_data):
            fill, stroke = row_color(entry)
            x = 290 + ci * col_w_trace
            els.append(rect(x, ry, col_w_trace - 5, row_h_trace - 4,
                            fill=fill, stroke=stroke,
                            dashed=("stall" in entry or entry == "IDLE" or entry == "-")))
            els.append(text(x + 4, ry + 7, entry, size=10, color=stroke, width=col_w_trace - 10))

    # Annotation
    note_y = table_y + 25 + 3 * row_h_trace + 15
    els.append(text(80, note_y,
                    "Cycles 7–10+: scheduler 0's 2 warps both stalled on HBM. 32 CUDA cores idle, ~700 cycles.",
                    size=13, color=DARK_RED, width=1400))
    els.append(text(80, note_y + 20,
                    "This is LOW-OCCUPANCY memory-bound behavior. Fix: more warps per scheduler.",
                    size=13, color=DARK_GRAY, width=1400))

    # ---- Row 4: Hardware unit utilization ----
    r4_y = note_y + 70
    els.append(text(80, r4_y, "4. HARDWARE UNIT UTILIZATION across the kernel's runtime:",
                    size=14, color=DARK_GRAY, width=800))
    # Horizontal bar chart
    units = [
        ("CUDA cores (128 per SM)", 2, DARK_GREEN, "~1% — trivial compute (just ADD)"),
        ("Tensor cores (4 per SM)", 0, DARK_PURPLE, "0% — no matmul"),
        ("DIV / SFU units", 0, DARK_PURPLE, "0% — no DIV/EXP"),
        ("Load/Store units", 90, DARK_ORANGE, "~100% — constantly issuing + waiting"),
        ("HBM bandwidth", 95, DARK_RED, "saturated (~3 GB read + 1 GB write)"),
    ]
    for i, (name, pct, color, desc) in enumerate(units):
        y = r4_y + 30 + i * 34
        els.append(text(80, y + 6, name, size=13, color=DARK_GRAY, width=280))
        # Full bar background
        els.append(rect(380, y, 400, 20, fill="#f1f3f5", stroke="#ced4da"))
        # Filled bar
        if pct > 0:
            els.append(rect(380, y, pct * 4, 20, fill=color, stroke=color))
        els.append(text(790, y + 5, f"{pct}%", size=13, color=color, width=50))
        els.append(text(840, y + 5, desc, size=12, color=DARK_GRAY, width=650))

    return els


# ============================================================================
# Main
# ============================================================================
def main():
    elements = []
    y = 100
    GAP = 80

    elements.extend(scene_hardware_hierarchy(y)); y += 460 + GAP
    elements.extend(scene_execution_model(y));   y += 430 + GAP
    elements.extend(scene_sm_structure(y));      y += 720 + GAP
    elements.extend(scene_coalescing(y));        y += 490 + GAP
    elements.extend(scene_warp_scheduling(y));   y += 720 + GAP
    elements.extend(scene_vector_add(y));        y += 460 + GAP
    elements.extend(scene_vector_add_trace(y));  y += 850 + GAP

    doc = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": "#ffffff",
        },
        "files": {},
    }
    with open(PATH, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote {len(elements)} elements to {PATH}")


if __name__ == "__main__":
    main()
