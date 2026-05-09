# GPU execution model — how kernels map onto the hardware

Model A (see `02-hardware-anatomy.md`) described what the GPU *is*:
SMs, cores, registers, memory hierarchy. Static structure.

Model B (this note) describes how your kernel *runs on it*:
grid → block → warp → thread, and how each level maps to the
hardware.

## What it is, plainly

When you launch a kernel, you tell the GPU: "run this function in
parallel, this many times, like this." The execution model is the
set of rules for *how* that parallelism is organized.

## What problem it solves

A GPU has thousands of cores, but your function is one function.
You need a way to say "each core does a different piece of the
work." The execution model provides the scaffolding: every thread
runs the *same* code, but each thread knows its own **identity**,
and uses that identity to pick which slice of data to process.

## The three logical levels

```
       Grid                  (a single kernel launch)
        │
   ┌────┼────┐
   │    │    │
 Block Block Block ...       (you choose how many)
   │
 ┌─┼─┐
 │ │ │
Thread Thread Thread ...     (you choose how many per block)
```

- **Grid** = the whole kernel launch. One grid per
  `kernel[grid_size](...)` call.
- **Block** = a bundle of threads that share an SM and can
  communicate via SRAM + synchronization. Configurable size
  (often 128, 256, 512, 1024).
- **Thread** = one independent execution of the kernel function.

**Key property:** every thread runs the same code, but with a
different (block_id, thread_id). It uses those to decide "which
slice of data am I responsible for?"

## Two knobs, one command

You decide just two numbers before launching:

```python
# Triton-flavored syntax
vector_add[(3907,)](x, y, out, N, BLOCK_SIZE=256)
#         ^^^^^^^^                  ^^^^^^^^^^^^^
#        grid size                 block size
```

- **Grid size** = how many blocks total.
- **Block size** = threads per block.

Example — add two arrays of 1 million elements, 256 threads per
block:

- `ceil(1,000,000 / 256) = 3907 blocks`.
- Grid = 3907 blocks; each block has 256 threads; total =
  1,000,192 threads (slight overshoot handled with a mask).

Everything else is up to the GPU.

## How the levels map onto the hardware

Three levels of scheduling happen in sequence, none of which the
programmer controls:

| Level | What it schedules | Who does it |
|---|---|---|
| 1 | Blocks → SMs | GPU scheduler (hardware) |
| 2 | Block threads → warps of 32 | Automatic |
| 3 | Warps → 32 CUDA cores per cycle | SM's warp schedulers |

### Level 1 — Blocks to SMs

The GPU's global scheduler distributes blocks across the 60 SMs
(on L4). Each SM holds multiple blocks *resident* (e.g. 16 per SM),
limited by registers + SRAM + max-blocks-per-SM constants. When a
block finishes, the SM grabs another from the global queue.

**Dynamic + automatic.** Happens in hardware. Invisible to the
programmer.

**Rule:** a block stays on its SM for its entire lifetime. Threads
within a block can share SRAM and synchronize, so they can't migrate.

### Level 2 — Threads to warps

A block's threads are automatically grouped into **warps of 32**.
A 256-thread block → 8 warps. Threads in a warp execute in
lockstep (SIMT — same instruction, different data).

### Level 3 — Warps to cores per cycle

Each SM has 4 warp schedulers. Per cycle, each scheduler picks one
warp to issue. That warp's 32 threads each use one of 32 CUDA
cores (or tensor cores, for matmul instructions).

- 4 warps active per cycle × 32 threads each = 128 CUDA cores busy
  per cycle. Matches the 128 CUDA cores per SM on L4.
- Many more warps are resident but waiting. When an active warp
  stalls on memory, the scheduler swaps it out and runs another —
  this is how the GPU hides memory latency.

## Full flow diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  1. YOU (Python side)                                            │
│                                                                  │
│  kernel[grid_size](args, BLOCK_SIZE=256)                         │
│     ↓                                                            │
│  "Launch 3907 blocks, 256 threads/block"                         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. GPU SCHEDULER (hardware)                                     │
│                                                                  │
│  Distributes 3907 blocks across 60 SMs:                          │
│  - Assigns resident blocks (e.g. 16 per SM initially)            │
│  - Queues the rest                                               │
│  - Refills each SM when blocks finish                            │
└──────┬─────────┬─────────┬─────────┬──────────────────────┬──────┘
       │         │         │         │                      │
       ▼         ▼         ▼         ▼                      ▼
    ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐              ┌─────┐
    │SM 0 │  │SM 1 │  │SM 2 │  │SM 3 │      ...     │SM 59│
    └──┬──┘  └─────┘  └─────┘  └─────┘              └─────┘
       │
       ▼  (zoom into SM 0)
┌─────────────────────────────────────────────────────────────────┐
│  3. INSIDE ONE SM (per-cycle)                                    │
│                                                                  │
│  Block's 256 threads  →  grouped into 8 warps of 32              │
│                                                                  │
│  4 warp schedulers pick 4 warps to run THIS cycle:               │
│                                                                  │
│    Warp 0 ─────►  CUDA cores  0─31   (32 cores)                  │
│    Warp 1 ─────►  CUDA cores 32─63   (32 cores)                  │
│    Warp 2 ─────►  CUDA cores 64─95   (32 cores)                  │
│    Warp 3 ─────►  CUDA cores 96─127  (32 cores)                  │
│                                                                  │
│  Other warps (from this & other resident blocks) wait.           │
│  Scheduler rotates them in when an active warp stalls.           │
└─────────────────────────────────────────────────────────────────┘
```

## What the programmer controls vs doesn't

| You control | GPU handles |
|---|---|
| Grid size | Blocks-to-SMs assignment |
| Block size | Threads-to-warps grouping |
| What each thread does (via its ID) | Warps-to-cores per cycle |
| Which data each thread touches | Memory coalescing, latency hiding |

**Your job as a kernel author:** pick good grid and block sizes,
and write correct code that uses the thread ID to index into data.
The rest is the GPU's problem.

## Why this matters for kernel performance

Two kinds of underutilization to watch for:

1. **Too few blocks.** If your grid has fewer blocks than SMs, some
   SMs sit idle. For L4, you want at least 60 blocks just to cover
   every SM. Usually you want many more so the scheduler has a
   queue.
2. **Blocks too small.** A 32-thread block → 1 warp → uses only 32
   of 128 cores per SM. You'd need 4 such blocks resident per SM to
   fill the cores. Better: use 128+ thread blocks so each block
   alone contributes multiple warps.

Typical rule of thumb: block sizes of 128, 256, or 512. Grid sizes
in the thousands. Triton autotunes both within a range, so you
often just need reasonable defaults.

## What's next

The next note walks through the actual Triton syntax — `@triton.jit`,
`tl.load`, `tl.store`, `tl.arange`, masks — and builds a trivial
vector-add kernel end to end.
