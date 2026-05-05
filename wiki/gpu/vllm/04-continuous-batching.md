# Continuous batching — the scheduler idea

Covers why the batch must be *dynamic* to convert paged memory into
throughput, and how the scheduler juggles prefill and decode work.

## Continuous batching in plain English

Instead of a fixed batch of sequences that all start together and all
finish together, the server keeps the batch **dynamic**. New
sequences enter the batch as old ones leave — every decode step,
possibly a slightly different set of sequences.

## What problem does it solve?

From earlier: saturating an H100 at Llama-2-7B decode wants
B ≈ 330 concurrent sequences. But sequences finish at very different
times — one user's 50-token reply next to another user's 2000-token
essay.

**Static batching** (the naive approach): lock the batch at start.
Everyone waits for the longest sequence. A 50-token seq's slot is
wasted for the 1950 steps the long one is still running. Average
utilization collapses — often 30% or less.

**Continuous batching:** the moment a seq finishes, its slot frees.
A waiting request joins on the next step. Batch stays full.

That's the whole idea. The scheduler's job: keep the batch near-max
on every step.

## Why it needs PagedAttention

Continuous batching *wants* churn — sequences entering and leaving
constantly. With the naive contiguous-slab cache, churn is expensive:

- Seq A finishes → slab at offset X freed.
- Seq C arrives → needs a contiguous slab somewhere.
- Over time, differently-sized/lived sequences fragment the memory.
  Eventually a new seq can't fit even though total free > seq size.
  (External fragmentation — see `01-motivation.md`.)

Static batching dodges this by refusing churn. Continuous batching
needs a memory manager that handles churn *gracefully*. PagedAttention
does — freed blocks are uniform and slot into the free list. No
fragmentation regardless of admission order.

The two ideas are inseparable:

- **PagedAttention alone** → fixes fragmentation but static batches
  still waste slots.
- **Continuous batching alone** → constantly fighting fragmentation,
  practical limits well below theory.
- **Both together** → vLLM's actual contribution. 2–4× throughput
  vs naive serving on the same hardware.

## Prefill vs decode — two workload shapes

The scheduler must handle two completely different regimes:

**Prefill** (once per request, at start):
- Input: full prompt, N tokens.
- Work: run all N tokens through the model in parallel, populate KV
  cache for positions 0..N-1, produce first output token.
- Cost: proportional to prompt length. Big matmul, compute-bound.

**Decode** (N_output times per request, after prefill):
- Input: the last generated token only (1 token).
- Work: one forward pass, produce one token, append one KV cache slot.
- Cost: fixed per step, memory-bound
  (see `wiki/gpu/04-why-decode-is-memory-bound.md`).

Prefill is rare and expensive; decode is frequent and cheap per step.

## Simplest scheduler — prefill-priority, non-mixing

Early vLLM approach: prefill and decode don't share a step.

```
Each step, pick ONE of:
  (a) if any request is waiting AND free blocks >= prompt_len / block_size:
        run prefill for that request
  (b) else:
        run a decode step for all in-progress sequences
```

Timeline with 3 users arriving over time:

```
step 1:  [prefill A, 500 tokens]     ← A admitted
step 2:  [decode A]                   ← A produces token 1
step 3:  [decode A]                   ← token 2
step 4:  [prefill B, 200 tokens]     ← B admitted
step 5:  [decode A, decode B]
step 6:  [decode A, decode B]
...
step 20: [decode B]                   ← A finished; slot freed
step 21: [prefill C, 1000 tokens]    ← C admitted
step 22: [decode B, decode C]
```

Every step does something. Batch fills as requests arrive, drains
as they finish.

## The uniform-shape problem — why attention can't just `Q @ K.T`

A subtle consequence of continuous batching + paged memory that's
easy to miss. In naive batched inference you'd write:

```
Q: (B, 1, H, D)
K: (B, T, H, D)     ← same T for every seq (via right-padding)
V: (B, T, H, D)
scores = Q @ K.T    ← one batched matmul
```

This works for **static** batching with padding. Under continuous
batching + paging, **neither assumption holds:**

1. Every sequence in the decode batch has a different number of prior
   tokens (`T_seq` varies wildly — 50 for one user, 1500 for another).
2. Each sequence's K/V isn't contiguous in memory — it's scattered
   across the block pool via that seq's block table.

There is no single `(B, T, H, D)` tensor to multiply against. You
**cannot** express the attention step as one `Q @ K.T`.

### What replaces the single matmul

Conceptually:

```python
for seq in batch:
    K_seq = gather_from_blocks(seq)       # (T_seq, H, D)
    V_seq = gather_from_blocks(seq)       # (T_seq, H, D)
    scores = Q[seq.idx] @ K_seq.T         # (1, T_seq)
    out[seq.idx] = softmax(scores) @ V_seq
```

B different small matmuls, one per sequence, each with its own T.
Not one big batched matmul.

In practice these B matmuls are **fused into one kernel launch** — a
custom Triton/CUDA kernel where each GPU thread block handles one
sequence: walks that seq's block table, reads K/V from the pool,
does the per-seq matmul, writes the output. All B of them run in
parallel on different SMs.

### What this means for the rest of the model

The split runs deep into the code structure:

| Layer                  | Shape-uniform? | Uses normal matmul? |
|------------------------|----------------|----------------------|
| Embedding              | yes, `(B, 1, D)` | yes |
| MLP / feed-forward     | yes, `(B, 1, D)` | yes |
| QKV projections        | yes, `(B, 1, D)` | yes |
| Output projection      | yes, `(B, 1, D)` | yes |
| LM head                | yes, `(B, 1, D)` | yes |
| **Attention core**     | **no — per-seq** | **no — custom kernel** |

Only attention needs the specialized path. Everything else stays on
standard batched ops.

### Why this is Phase 3 work

The `gather_from_blocks(seq)` loop is trivial in PyTorch (a `for` plus
`torch.cat`) but slow — Python overhead and a materialized tensor per
step per sequence. Phase 2 (correctness) ships with this. Phase 3
rewrites it as a Triton kernel that reads straight from the block pool
using the block table as an index — no Python loop, no `cat`, no
intermediate tensor. That's the actual "PagedAttention kernel" from
the paper.

The off-the-shelf options (`F.scaled_dot_product_attention`, stock
FlashAttention) assume a single uniform K tensor and don't apply —
paged attention needs its own kernel by construction.

### Trade-off: the prefill stall

A prefill step blocks decode. If user A is mid-stream and user C's
1000-token prefill takes 50 ms, A waits 50 ms for its next token.
That's a visible latency hiccup to A.

Modern vLLM mitigates this with **chunked prefill** — slice a big
prefill into pieces and interleave it with decode in the same batch.
See dangling-threads for the full discussion.

## Dangling threads for later

- **Chunked prefill** — mechanics of slicing a prefill and
  interleaving with decode in one step. How the kernel handles
  mixed query lengths (1 for decode tokens, M for prefill chunks)
  in the same batch.
- **Admission policies** — when to *reject* a request vs queue it.
  vLLM can preempt decoding sequences when memory pressure gets too
  high ("swap out" to CPU or "recompute"). Important for real
  workloads.
- **Scheduling fairness** — FCFS vs priority vs throughput-maximizing
  policies.

## What this note unlocks

With continuous batching + PagedAttention we have the core
inference loop. Next note covers the *memory-sharing* ideas — prefix
caching, copy-on-write for beam search — which are extensions
of the block-table indirection.
