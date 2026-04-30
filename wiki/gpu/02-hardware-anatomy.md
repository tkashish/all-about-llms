# GPU hardware anatomy — SMs and the memory hierarchy

## What an SM is

**SM = Streaming Multiprocessor.** One "compute unit" on the GPU.
A GPU is really a collection of SMs running in parallel.

Concretely, an H100 has **132 SMs**. Each SM has:
- Its own math units (including tensor cores for matmul)
- Its own registers
- Its own small scratchpad memory (shared memory / SRAM)

Analogy: if a GPU were a factory, SMs are the individual workstations.
Each workstation has its own tools (math units) and its own workbench
(registers + scratchpad). They all pull raw materials from the same
warehouse (HBM).

## The memory hierarchy

Smaller = faster = closer to the math units. Numbers are H100, bf16:

| Layer            | Size                 | Bandwidth   | Scope    |
|------------------|----------------------|-------------|----------|
| HBM              | 80 GB                | ~3 TB/s     | Shared   |
| L2 cache         | ~50 MB               | ~12 TB/s    | Shared   |
| SRAM / shared    | ~228 KB per SM       | ~20 TB/s    | Per-SM   |
| Registers        | ~64 KB per SM        | effectively free | Per-SM |

Total register storage across the whole GPU: 132 × 64 KB ≈ 8 MB.

## The structural shape

```
        HBM (shared across all 132 SMs)
              │
         L2 cache (shared)
              │
   ┌──────────┼──────────┐
   │          │          │
  SM 0      SM 1       ... SM 131
  ├ SRAM   ├ SRAM       ├ SRAM      (per-SM)
  └ regs   └ regs       └ regs      (per-SM)
```

## Consequence for code

- **Cross-SM communication goes through HBM or L2.** Slow.
- **Within an SM, data in SRAM/registers is basically free to reuse.**

A well-written kernel loads a chunk from HBM into SRAM *once*, then
has the SM chew on it many times before writing the result back.
This is the core trick behind FlashAttention and most fast kernels:
**maximize reuse inside the SM, minimize round-trips to HBM.**

When we say "memory-bound," we almost always mean **HBM-bound** —
waiting on the big slow outer layer. Inner layers (L2, SRAM, registers)
are fast enough to keep the math units fed.

## What's next

The next note introduces **arithmetic intensity** — FLOPs per byte
loaded from HBM — the single number that tells you whether a given
kernel is doing enough on-SM work to keep the math units busy.
