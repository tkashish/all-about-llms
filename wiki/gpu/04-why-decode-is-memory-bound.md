# Why LM decode is memory-bound — and how batching fixes it

## The setup

One decode step in a transformer LM = generate one token for one
sequence. The dominant cost is the weight matmuls inside each block:
`x @ W` where W is a (D, D) weight matrix.

Shapes for **one decode step, batch=1, seq=1**:
- `x`: (1, 1, D) — a single vector
- `W`: (D, D)

## The intensity calculation

- **FLOPs:** 2 · 1 · D · D = **2D² FLOPs**
- **HBM bytes:** read x (~2D) + read W (2D²) + write out (~2D).
  Dominated by W → **~2D² bytes**
- **Intensity:** 2D² / 2D² = **1 FLOP/byte**

Compare to H100 ridge ~330 FLOPs/byte: decode is **330× below the
ridge.** Math units sit idle ~99.7% of the time. The bottleneck is
pulling W from HBM.

The ugly part: we pay the full 2D² bytes to read W **every single
token.** Generate 100 tokens → read every weight matrix 100 times
from HBM.

## The fix: batch across sequences

Decode B sequences at once instead of 1:
- `x`: (B, 1, D) — B vectors stacked
- `W`: (D, D) — **same matrix, read once, reused B times**

Recompute:
- **FLOPs:** 2 · B · D² (B× more work)
- **HBM bytes:** still ~2D² (W dominates, still read once)
- **Intensity:** 2BD² / 2D² = **B FLOPs/byte**

Intensity scales linearly with batch size. To hit the H100 ridge
(~330), need **B ≈ 330**.

## Why inference servers batch aggressively

You pay the HBM cost to read W regardless of how many users are in
the batch. Amortizing that read over many users is free throughput.
A single user's decode wastes the machine.

**Continuous batching** (vLLM's trick) keeps the batch full by
swapping in new requests as others finish, so the server stays near
the ridge instead of dipping memory-bound every time a request
completes.

## The caveat: attention doesn't benefit the same way

Batching helps the weight matmuls (W is shared across the batch).
It does **not** fix attention in the same way, because each
sequence's K and V are its own — there's no shared matrix to
amortize the read over.

This is exactly the problem the **KV cache** attacks: instead of
recomputing K and V from scratch for every prior token on every
step, compute them once and stash them. See
`wiki/transformer-primer/18-kv-cache.md`.

And it's the problem **PagedAttention** (vLLM) attacks at the
memory-management layer — how to pack many sequences' KV caches
efficiently so batches can stay large without OOM.

## The mental model to carry forward

- **Prefill** (processing the prompt, many tokens at once): behaves
  like a normal matmul on a (prompt_len, D) activation. Usually
  compute-bound for reasonable prompt lengths.
- **Decode** (generating one token at a time): matmul against a
  (B, 1, D) activation. Memory-bound unless B is large.

Essentially every inference optimization — batching, KV cache,
quantization, speculative decoding, FlashAttention — is aimed at
the decode problem.
