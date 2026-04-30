# Llama-2-7B decode — full arithmetic intensity walkthrough

Running example to make "decode is memory-bound" concrete with real
numbers. Built up step by step; each section added as we work through
it in conversation.

## Model spec

| Quantity          | Value                 |
|-------------------|-----------------------|
| Layers (L)        | 32                    |
| Hidden dim (D)    | 4096                  |
| Attention heads   | 32 (D_head = 128)     |
| MLP hidden (D_ff) | 11008 (≈ 2.7·D)       |
| Vocab (V)         | 32000                 |
| Total params      | ~6.7B (call it 7B)    |
| Dtype             | bf16 (2 bytes/param)  |

Target hardware: H100, 3 TB/s HBM bandwidth, ~1000 TFLOP/s bf16.

## Step 1 — total bytes to read from HBM per forward pass

### 1a. Parameter count, from scratch

**Per transformer block:**

Attention has 4 matrices of shape (D, D): W_Q, W_K, W_V, W_O.
→ 4D² params.

MLP (SwiGLU) has 3 matrices of shape (D, D_ff): W_gate, W_up, W_down.
With D_ff ≈ 2.7·D → 3·D·D_ff ≈ 8D² params.

**Total per block: 4D² + 8D² = 12D².**
(Handwavy — ignoring biases, norms, and exact D_ff ratio.)

Embedding table: V × D = 32000 × 4096 ≈ 0.13B params. In Llama the
output head is tied, so no extra params for it.

**Grand total:**
- 32 blocks × 12·4096² = 32 × 201M ≈ **6.4B**
- Embedding: **0.13B**
- **Sum ≈ 6.5B**, rounds to the advertised 7B.

### 1b. Bytes

bf16 → 2 bytes/param.

**7 × 10⁹ params × 2 B/param = 14 × 10⁹ B = 14 GB.**

So a single forward pass requires reading 14 GB of weights from HBM.
That's also why 7B barely fits on a 24 GB GPU in bf16.

## Step 2 — time to read the weights

HBM bandwidth on H100 is **3 TB/s = 3 × 10¹² B/s**.

```
time = bytes / bandwidth
     = 14 × 10⁹ B / 3 × 10¹² B/s
     = 14 / 3000 s
     ≈ 4.67 ms
```

**~4.7 ms just to read the weights once.**

### Why this is a floor, not the actual time

4.7 ms is the **minimum possible** forward-pass time, assuming math
were free. The weights physically have to travel from HBM → SM
regardless of how fast the math units are.

Real execution is not simply `memory_time + compute_time`. GPUs
**overlap** the two: while the math units chew on bytes that already
arrived, the memory system is already fetching the next bytes. So:

```
total time ≈ max(memory_time, compute_time)
```

- If memory_time > compute_time → compute hides behind memory →
  **memory-bound** → total ≈ memory_time.
- If compute_time > memory_time → memory hides behind compute →
  **compute-bound** → total ≈ compute_time.

To prove decode is memory-bound, we need to compute the compute time
and show it's less than 4.7 ms. That's step 3.

### Token/sec ceiling

If one token takes at least 4.7 ms:

```
1 s / 4.7 ms per token ≈ 213 tokens/s
```

That's the theoretical ceiling for batch=1 Llama-2-7B decode on a
single H100. Real systems hit ~100 tok/s — close to the ceiling,
confirming bandwidth is genuinely the bottleneck in practice.

## Step 3 — FLOPs per forward pass

Rule of thumb:

> **Forward-pass FLOPs ≈ 2 × params × tokens_processed**

The 2 comes from each parameter contributing one multiply + one add
per token it touches.

For one decode token, tokens_processed = 1:
```
FLOPs ≈ 2 × 7 × 10⁹ × 1 = 1.4 × 10¹⁰ FLOPs ≈ 14 GFLOPs
```

(This ignores attention's Q·K^T and scores·V, which scale with
context length, and the LM head. Small fraction for short contexts.)

## Step 4 — compute time

H100 bf16 throughput ≈ 1000 TFLOP/s = 10¹⁵ FLOP/s.

```
time = 1.4 × 10¹⁰ / 10¹⁵ s = 1.4 × 10⁻⁵ s ≈ 0.014 ms ≈ 14 µs
```

## Step 5 — verdict

| Phase           | Time      |
|-----------------|-----------|
| Memory (weights)| 4.7 ms    |
| Compute (math)  | 0.014 ms  |

**Memory time is ~330× larger than compute time.**

That ratio is not a coincidence — it equals (hardware ridge /
workload intensity) = 330 / 1.

**Verdict: decode at batch=1 is brutally memory-bound.** The math
units sit idle for ~4.69 ms of every 4.7 ms → **~99.7% idle.**

## Step 6 — what batching does to these numbers

For a batch of B sequences, each decoding one token:

- **Weights read from HBM:** unchanged. 14 GB. W is shared across
  all B sequences — read once, reused B times. Memory time stays
  ~4.7 ms.
- **FLOPs:** scale linearly with B. Each sequence does its own
  14 GFLOPs → total 14·B GFLOPs. Compute time = 0.014·B ms.

| B     | Memory time | Compute time | Bottleneck   |
|-------|-------------|--------------|--------------|
| 1     | 4.7 ms      | 0.014 ms     | memory       |
| 10    | 4.7 ms      | 0.14 ms      | memory       |
| 100   | 4.7 ms      | 1.4 ms       | memory       |
| 330   | 4.7 ms      | 4.7 ms       | **balanced** |
| 1000  | 4.7 ms      | 14 ms        | compute      |

At B ≈ 330 we hit the H100 ridge — memory and compute take equal
time. Math units fully fed, bandwidth fully used, nothing wasted.

### Throughput consequence

At B=1: 1 token per 4.7 ms → **~210 tok/s** (single user).
At B=330: 330 tokens per 4.7 ms → **~70,000 tok/s** aggregate.

**Same hardware. 330× more output.**

### Practical consequences

- **Single-user chat is wasteful.** A dedicated GPU with one user is
  ~99.7% idle. APIs amortize this across many users; self-hosted
  single-stream inference pays the full memory cost per token.
- **Latency vs throughput.** Batching doesn't speed up one token —
  still 4.7 ms. It lets many users share that 4.7 ms. Per-user
  latency unchanged; aggregate throughput up 330×.
- **Most inference optimizations target this exact problem.**
  - Quantization (int8, int4): shrink the 14 GB → less bytes to move.
  - FlashAttention: reuse SRAM, reduce HBM round-trips for attention.
  - Speculative decoding: spend idle compute to reduce total decode
    steps (fewer memory passes overall).
  - KV cache: avoid recomputing K, V for past tokens every step.

### The catch: memory capacity, not just bandwidth

The 330× headroom assumes the batch fits in GPU memory. It often
doesn't. **KV cache** grows with batch × context length and can
exceed HBM capacity long before B reaches 330. That's the specific
problem **PagedAttention / vLLM** solves.

## Summary

| Quantity              | Value        |
|-----------------------|--------------|
| Model size (bf16)     | 14 GB        |
| FLOPs per decode token| 14 GFLOPs    |
| Memory time @ B=1     | 4.7 ms       |
| Compute time @ B=1    | 0.014 ms     |
| Intensity @ B=1       | 1 FLOP/byte  |
| H100 ridge            | 330 FLOP/byte|
| B to hit ridge        | ~330         |
| Batch=1 ceiling       | ~210 tok/s   |
| Batch=330 ceiling     | ~70,000 tok/s|

The whole field of LLM inference optimization is a reaction to this
table.

## Step 7 — how large can B actually be?

Two different limits:

### Limit A — compute efficiency (the ridge)

From the table, B ≈ 330 saturates the math units on H100. Beyond
that, we're compute-bound; larger B just queues work without
increasing per-token throughput.

**Compute-efficient ideal: B ≈ 330.**

### Limit B — memory capacity

But can 330 sequences actually fit? Each sequence has its own KV
cache. For Llama-2-7B:

Per-token KV cache bytes:
```
  2 (K + V) × L layers × H heads × D_head × 2 bytes
= 2 × 32 × 32 × 128 × 2
= 524 KB per token
```

For a 2048-token context per sequence:
```
524 KB × 2048 = ~1 GB per sequence
```

H100 (80 GB) budget after weights:
```
80 GB − 14 GB (weights) = 66 GB free
66 GB / 1 GB per seq = ~66 sequences max at 2048 context
```

**Memory-capacity ideal: B ≈ 60–70** — well below the compute ridge.

### The gap is the vLLM problem

The gap between "330 would be ideal for compute" and "only 66 fit
in memory" is exactly what **PagedAttention** attacks. Naive
allocation reserves max-context memory per sequence even when the
actual context is short, wasting huge amounts of HBM. PagedAttention
allocates in small fixed-size pages on demand, so a batch where most
sequences are short can pack in where naive allocation couldn't.

**Summary of batch-size ideals for Llama-2-7B on H100:**

| Criterion                     | Ideal B |
|-------------------------------|---------|
| Compute efficiency (ridge)    | ~330    |
| Memory capacity, naive, 2k ctx| ~66     |
| With PagedAttention           | approaches the compute ridge |

This is why vLLM exists: to close the gap between compute-efficient B
and memory-feasible B.
