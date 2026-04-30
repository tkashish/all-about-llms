# Compute vs memory — the core tension

## What it is

Every GPU engineering decision comes down to one question:

> Is my code waiting on math, or is it waiting on memory?

Kernel fusion, batching, quantization, FlashAttention, KV caches — all
of them are moves in one of those two fights.

## Why the tension exists

A GPU does two things, at wildly different speeds:

- **Do math** (multiply, add). An H100 does ~1000 trillion float ops
  per second (1000 TFLOPS in bf16).
- **Move numbers** between its big memory (HBM) and the tiny scratchpad
  next to the math units. ~3 TB/s on an H100.

Numbers to hold in your head (H100, bf16):

| Thing             | Rate          |
|-------------------|---------------|
| Math (tensor core)| ~1000 TFLOP/s |
| HBM bandwidth     | ~3 TB/s       |

Each bf16 number is 2 bytes, so 3 TB/s = ~1.5 trillion numbers/s moved.
Math throughput is ~1000 trillion ops/s. Ratio: **~600× more math per
second than numbers-moved per second.**

## What this means for code

If a kernel reads a number, does one multiply, and throws it away:
you spent ~600 units of time fetching 1 number to do 1 unit of math.
The math units sat idle 99.8% of the time. Memory-bound.

If a kernel reads a number and does 1000 multiplies on it before
discarding: math units stay busy. Compute-bound.

## The takeaway

"Fast" on a GPU means one of:
1. You're doing enough math per byte loaded that the math units stay
   fed (compute-bound → limited by FLOPS).
2. You're moving memory as fast as the bus allows and can't do
   anything about it (memory-bound → limited by bandwidth).

"Slow" means neither — you're bandwidth-bound but not saturating
bandwidth, usually because of bad access patterns or launch overhead.

The next note introduces **arithmetic intensity** — the single number
that tells you which regime a given workload is in.
