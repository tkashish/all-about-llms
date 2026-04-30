# Arithmetic intensity and the roofline

## What it is

A single number that describes a workload:

> **Arithmetic intensity = FLOPs performed / bytes moved across HBM.**

Units: FLOPs per byte.

## Why it exists

We need a way to compare a workload against the hardware. The hardware
has a fixed ratio — how many FLOPs it can do per byte of HBM bandwidth.
If the workload's ratio is below the hardware's ratio, you're
memory-bound. Above it, compute-bound.

## The hardware ratio (H100, bf16)

```
compute throughput / memory bandwidth
= 1000 TFLOP/s / 3 TB/s
≈ 330 FLOPs per byte
```

This number is called the **ridge point** of the roofline.

Meaning: to keep the math units fully fed, every byte pulled from HBM
must participate in ~330 FLOPs before the next byte arrives. Less than
that → math finishes first → math units idle → memory-bound.

Hold onto this: **~330 FLOPs/byte on H100 in bf16.**

## The workload ratio — worked example

Matmul of two N×N bf16 matrices: `C = A @ B`.

- **FLOPs:** each output element is a dot product of length N
  (N multiplies + N adds = 2N FLOPs). There are N² outputs.
  **Total: 2N³ FLOPs.**
- **HBM bytes moved:** read A (2N²) + read B (2N²) + write C (2N²).
  **Total: 6N² bytes.**

  Note: C is written, not loaded. But writes cost bandwidth too —
  every byte of C has to travel over the HBM bus once, from the SM
  out to HBM. The bus doesn't care about direction.

- **Intensity:** 2N³ / 6N² = **N/3 FLOPs/byte.**

## Where this lands for real sizes

| N     | Intensity (N/3) | Regime (H100, ridge≈330) |
|-------|-----------------|--------------------------|
| 128   | ~43             | memory-bound             |
| 1024  | ~340            | right at the ridge       |
| 4096  | ~1365           | compute-bound            |

**Punchline:** big matmuls are compute-bound, small ones are
memory-bound. Same operation, different regime based on size.

## Why the output matrix can't just stay in SRAM

A natural question: why do we write C to HBM at all? Why not keep it
in SRAM?

Because C doesn't fit. SRAM is ~228 KB per SM. A 4096×4096 bf16
matrix is 32 MB — ~140× bigger than one SM's SRAM. Across all 132 SMs
there's ~30 MB total — barely room for C once, with nothing left for
A or B.

**Pattern for fast kernels:**
- Inputs live in HBM (too big for SRAM).
- Outputs live in HBM (too big for SRAM).
- **Intermediates stay on-SM** — that's where reuse wins come from.

Partial sums accumulate in registers; only the final tile of C is
written to HBM once. So 2N² bytes for C is the correct count.

## The roofline picture

```
    achievable FLOP/s
        ▲
compute ┤        ┌──────────── (compute-bound ceiling)
ceiling │       ╱
        │      ╱
        │     ╱   ← memory-bound (slope = bandwidth)
        │    ╱
        │   ╱
        └──┴──────────────────► arithmetic intensity
           ridge
          (~330)
```

Any workload plots as a point under this curve. The ridge tells you
where memory-bound stops and compute-bound starts.
