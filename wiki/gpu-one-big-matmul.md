# Why one big matmul beats many small ones

Recurring GPU optimization principle. You will see this rule again and again:

> **On GPUs: fuse small ops into big ops.**

Same total FLOPs, very different wall-clock time.

## Three reasons

### 1. Kernel launch overhead

Every GPU op costs ~5–20 μs of fixed overhead (serialize args, send to
device, schedule, synchronize). That's tiny for one launch, but multiplied
across H=64 heads or thousands of small ops it dominates.

For a matmul whose actual compute is ~100 μs, launch overhead can be **half
the total time**.

### 2. Tensor core tile utilization

GPUs compute matmuls in fixed-size tiles (e.g. 16×16 on H100 tensor cores).
If your matmul is smaller than the tile, the GPU runs the tile anyway —
filling empty spots with zeros. Wasted silicon.

- One big `(768, 768)` matmul → 48×48 tiles → full utilization.
- 12 small `(64, 64)` matmuls → 4×4 tiles → 3/4 of each tile wasted.

Bigger matmul → higher fraction of peak FLOPs achieved.

### 3. Memory bandwidth — reads dominate

Reading `x` from HBM costs the same whether you use it for 1 matmul or 12.
Twelve small matmuls re-read `x` from HBM 12 times. One big matmul reads
`x` once.

HBM bandwidth is the bottleneck on modern GPUs. H100 needs arithmetic
intensity > 295 FLOPs/byte to be compute-bound. Small matmuls plus repeated
reads = memory-bound = slow.

## Mental model

GPUs are **throughput** machines, not **latency** machines. They want one
big chunk of work. Twelve small chunks with overhead between each is like
asking a freight train to deliver 12 individual letters — the train works,
but you're wasting it.

## Where you see this rule in practice

- **Multi-head attention**: one `(D, D)` matmul producing Q for all H
  heads, then reshape → split heads. Not H separate `(D, D_head)` matmuls.
- **QKV fusion**: some implementations go further — one `(D, 3D)` matmul
  produces Q, K, V in a single pass. Three matmuls become one.
- **FlashAttention**: fuses scores + mask + softmax + matmul-with-V into
  one kernel. Multiple ops become one.
- **Triton kernel writing**: the whole point is fusing memory-
  bound ops into single kernels.
- **Batching** at inference: 32 requests → 32× batch dim → one matmul
  instead of 32.

## Caveat

There's a limit. Very large matmuls can exceed available registers / SMEM,
causing register spills or un-tiled execution. Standard tile-aware
libraries (cuBLAS, Triton autotune) handle the sweet spot for you.

## Takeaway

Whenever you see a loop of small tensor ops, ask: **can I collapse this
into one big op via reshape or concat?** Usually yes, usually 3–10× faster.
