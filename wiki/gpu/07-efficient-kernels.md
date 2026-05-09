# Writing efficient GPU kernels

Principles that decide whether a kernel uses the GPU well. Notes
build up as each concept is covered. Companion to
`02-hardware-anatomy.md` (Model A) and `06-execution-model.md` (Model B).

## 1. Memory coalescing

### Prerequisite: cache lines

**Cache line, plainly:** the smallest unit of memory transfer on the
GPU. Always **128 bytes** on modern NVIDIA GPUs. You never read one
float from HBM — you read a 128-byte chunk, whether you need all of
it or not.

**Why they exist:** HBM is slow to address but fast to stream. Setting
up a memory transaction has fixed overhead; once started, pulling 128
bytes costs little more than pulling 4 bytes. Hardware batches
requests into cache-line-sized chunks to amortize the overhead.

**Why they govern coalescing:** a warp is 32 threads × 4 bytes = 128
bytes of total request. If those 128 bytes all land in **one cache
line**, the hardware serves the whole warp with **one transaction**.
If scattered across 32 different cache lines, 32 transactions. ~30×
bandwidth penalty.

### What coalescing is

When the 32 threads in a warp read memory addresses that are
**right next to each other**, the GPU services all 32 reads with
**one memory transaction**. When they're scattered, it's 32
separate transactions. ~30× difference in effective memory bandwidth.

### What problem it solves

HBM (the GPU's main memory) is slow-ish (~300 GB/s on L4) but
**wide** — each memory transaction fetches a big chunk (typically
128 bytes) at once, whether you use all of it or not. Coalescing
is about using every byte of that chunk.

### Concrete picture

A warp has 32 threads. Each thread wants one 4-byte float = 128
bytes total for the warp.

```
Good (coalesced):
  threads 0..31 read addresses 0, 4, 8, ..., 124   ← one 128-byte transaction. ✓
  [████████████████████████████████]

Bad (scattered):
  thread 0  reads address 0
  thread 1  reads address 4096
  thread 2  reads address 8192
  ...                                              ← 32 separate transactions. ✗
```

### The rule that matters

When designing a kernel, **threads `0..31` of a warp should touch
contiguous slots** of the data they're reading. This single rule
drives 90% of layout decisions:

- Which axis of a multi-dim tensor does each thread index?
- What does `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`
  imply about which thread hits which address? (Answer: thread `i`
  hits offset `pid*BLOCK + i`, i.e. contiguous — coalesced by
  construction.)

### Common anti-patterns

TODO — cover once we start writing kernels:
- Row-major vs column-major access in matmul.
- Strided reads (stride > 1 between consecutive threads).
- Reading from a `(batch, feature)` tensor along the batch axis.

## 2. Occupancy and latency hiding

### What it is, plainly

**Occupancy** = how many warps are resident on each SM. **Latency
hiding** = the GPU's ability to run other warps while one waits on
memory. These are two sides of the same concept: you need enough
resident warps that the warp scheduler always has something to run
during memory stalls.

### The numbers (L4 / Ada Lovelace)

- Max threads per SM: 1536 → max **48 resident warps** per SM.
- 4 warp schedulers per SM, assigned statically by `warp_id % 4`
  → **each scheduler manages up to 12 warps**.
- Per cycle: each scheduler picks **one ready warp** from its pool
  and issues one instruction. That instruction uses its 32 CUDA cores.

### Resident vs ready

- **Resident** = loaded on the SM with registers + shared memory
  allocated. Known to the scheduler.
- **Ready** = can run *this cycle*. All previous-instruction
  dependencies resolved (in particular, no outstanding HBM load
  waiting to return).

A warp stalled on memory is resident but not ready.

### Memory latency in cycles

| Memory tier | Typical latency (cycles) |
|---|---|
| Register | ~0 |
| SRAM / L1 hit | 20–30 |
| L2 hit | ~200 |
| **HBM load** | **600–800** |

Simple arithmetic (add, FMA) is ~4 cycles. So one HBM load is
roughly **150 math instructions** of idle time, if nothing else can
run.

### Why occupancy matters

Each cycle, each scheduler needs ≥1 ready warp. When one warp stalls
on a 700-cycle HBM load, you need ~700 cycles of other ready work
queued across that scheduler's pool to avoid idling.

**With 12 warps per scheduler, each having dozens of math
instructions between memory ops, latency is fully hidden.**
128 cores busy every cycle.

**With only 1–2 warps per scheduler (low occupancy):** when that
warp stalls, the scheduler has nothing. 32 cores idle for ~700
cycles. This is memory-bound underutilization — a very common
kernel performance bug.

### What kills occupancy

Two main causes, both things your kernel controls:

- **Register pressure.** Registers per SM are fixed (~64 KB). Each
  resident thread consumes registers. If each thread uses 128
  registers, you fit fewer warps. Tune kernel to use fewer registers
  per thread (smaller local variables, less intermediate state).
- **Shared memory per block.** SRAM per SM is fixed (~128 KB). If
  each block allocates 64 KB of shared memory, only 2 blocks fit
  per SM, dramatically limiting resident warps.

The CUDA Occupancy Calculator and `ncu` both report achieved
occupancy directly.

### Rule of thumb

> If your kernel is memory-bound, **increase occupancy first.** It's
> the cheapest optimization and often the biggest.

Only after occupancy is maxed should you worry about tiling, shared
memory reuse, or other advanced tricks.

### How to pick grid size, block size, and num_warps

**Hardware facts (A10G / Ada Lovelace):**

| Limit | Value |
|---|---|
| Warp size (always) | 32 threads |
| Max threads per block | 1024 |
| Max blocks resident per SM | 16 |
| Max warps resident per SM | 48 |
| Max threads resident per SM | 1536 |
| Registers per SM | 65,536 (64 KB) |
| SRAM per SM | ~128 KB |

**Your two knobs when launching a Triton kernel:**
- `grid_size` — number of blocks to launch.
- `num_warps` — warps per block (determines threads per block).
  Note: in Triton, `BLOCK_SIZE` is vector width per program, not
  thread count. Threads per block = `num_warps × 32`.

Each Triton thread handles `BLOCK_SIZE / (num_warps × 32)` elements
via a small inner loop inside the kernel body.

**Per-block thread count:**

```
num_warps=1   →   32 threads/block   (tiny)
num_warps=4   →  128 threads/block   (Triton default)
num_warps=8   →  256 threads/block
num_warps=16  →  512 threads/block
num_warps=32  → 1024 threads/block   (hardware max)
```

**Per-SM occupancy, warp-budget only (ignoring registers/SRAM for now):**

```
max_blocks_per_SM = min(
    16,                                    # hard limit
    48 / num_warps_per_block,              # warp limit
    registers_per_SM / (regs_per_thread × threads_per_block),
    sram_per_SM / sram_per_block,
)
```

| `num_warps` | threads/block | max blocks/SM | resident warps | Occupancy |
|---|---|---|---|---|
| 1 | 32 | 16 (hits block cap) | 16 | 33% |
| 2 | 64 | 16 (hits block cap) | 32 | 67% |
| 4 | 128 | 12 | 48 | **100%** |
| 8 | 256 | 6 | 48 | **100%** |
| 16 | 512 | 3 | 48 | **100%** |
| 32 | 1024 | 1 | 32 | 67% |

**Sweet spot: `num_warps` between 4 and 16.** You hit max occupancy
(48 warps/SM = 1536 threads, full utilization) with room for
multiple blocks to share the SM.

**Extremes:**

- **Too small** (`num_warps=1`): tiny blocks hit the 16-block cap →
  only 16 × 32 = 512 threads resident. 33% occupancy. Blocks are
  too small to fill the SM.
- **Too big** (`num_warps=32`): only 1 block fits per SM → 32 warps.
  67% occupancy. Worse: no second block means no inter-block
  latency hiding when this block stalls.

**Per-scheduler pool (inside one SM):**

Each SM has 4 warp schedulers. Warps distribute by `warp_id % 4`.

- `num_warps=4, blocks_per_SM=12`: 48 warps / 4 = **12 warps per
  scheduler's pool** (best for latency hiding).
- `num_warps=32, blocks_per_SM=1`: 32 warps / 4 = **8 warps per
  scheduler's pool** (still OK, but less).

**Grid-size math — how many blocks to launch:**

Minimum to fill the GPU once:
```
min_grid = num_SMs × max_blocks_per_SM
         = 80 × 12 = 960   (for num_warps=4 on A10G)
```

In practice, launch **many times** this — often 10× — so the
scheduler always has work queued and SMs are never idle waiting
for the next block.

**Concrete recipe for a memory-bound vector kernel:**

```python
BLOCK_SIZE = 1024          # vector width per program
num_warps = 4              # 128 threads/block; each thread handles 8 elements

grid = (triton.cdiv(N, BLOCK_SIZE),)
# For N=1M: grid = (977,). 977 blocks / 80 SMs ≈ 12 blocks per SM.
# Full occupancy: 12 × 4 = 48 warps per SM. 
```

**Summary — sizing decisions in order:**

1. **`num_warps`** sets block size and register pressure. Start with 4.
2. **`BLOCK_SIZE`** (Triton vector width) sets how much data per
   program. Larger = more work amortizes launch + loop overhead,
   up to register pressure limits.
3. **`grid_size`** = `ceil(N / BLOCK_SIZE)`. Should be many thousands
   for a real workload. Well above `num_SMs × blocks_per_SM`.
4. Occupancy maxes at 48 warps per SM (A10G). Going much below =
   losing latency hiding.

## 3. SRAM tiling

TODO — covered when we write the paged-attention kernel.

## 4. Branch divergence

TODO — as it comes up.

## 5. Register pressure

TODO — as it comes up.
