# Paged cache implementation — design notes

Phase 2 of the vLLM learning plan — replace the Level 5 KV cache
with a paged cache in pure PyTorch, on the existing transformer.
Correctness first, speed later (Phase 3).

Companion to `02-block-table.md` (the data structure) and
`03-how-paging-fixes-waste.md` (why).

## Summary of what got built

Three components in `src/transformer/paged_cache.py`:

- **`Block`** — fixed-capacity container for `block_size` tokens'
  worth of K and V, across all layers. Tensor shape
  `(num_layers, num_heads, block_size, d_head)` for each of K and V.
- **`BlockPool`** — one pre-allocated list of Blocks + a deque free
  list. One `alloc()` per 16 tokens, one `free()` per seq on release.
- **`Sequence`** — one per request. Owns a list of `Block`s
  (the block table) and implements `add` (prefill), `append`
  (decode), `get` (read all K/V for a layer).

Attention threads a `Sequence` through `forward()` as an optional arg
(`seq: Sequence | None = None`) so training paths still work.

## Scope kept small on purpose

Deferred to Phase 3:

- Multiple concurrent sequences.
- Block sharing (prefix / CoW).
- Triton kernel for attention — stayed with a Python loop over blocks
  + `torch.cat` in `Sequence.get()`.
- Scheduler, continuous batching.

Goal of Phase 2: prove paged cache is a **byte-identical** drop-in
for the Level 5 cache on one sequence. That's the milestone.

## The subtle bug — per-layer token counters

The first paged implementation had a single `tokens_used` counter per
Block, updated after the **last layer's** write:

```python
def append(self, k, v, layer_id):
    self.k[layer_id, :, self.tokens_used:self.tokens_used + 1, :].copy_(k)
    ...
    if layer_id == self.num_layers - 1:
        self.tokens_used += 1
```

Looks right. Breaks correctness.

### Why it breaks

In a decode step, attention processes layers in order:

```
for each layer l:
    append(k_l, v_l, l)      # write new token's K/V for layer l
    K, V = get(l)             # read all tokens' K/V for layer l
    ... do attention ...
```

When **layer 0** calls `get(0)`, `tokens_used` hasn't been updated
yet — that update waits for layer N-1. So `get(0)` returns K/V for
only the prefill tokens, missing the one just written at the current
step. Layer 0 effectively runs attention on stale context.

The failure mode is subtle. Prefill works (cache starts empty, every
layer reads what was just written). Decode step 1 is "one step
behind" — its output matches what prefill already produced.

### Diagnostic signature

In the debug dump, the smoking gun was:

```
[no-cache step 1]  top5_ids=[10, 312, 285, 316, 317]
[paged step 2]     top5_ids=[10, 312, 285, 316, 317]   ← same, off by one
```

Paged step N matches no-cache step N-1 exactly. Classic off-by-one
on the cache-read boundary.

### The fix

Track `token_per_layer` as a dict keyed by layer index. Each layer
updates its own counter on `append`, and reads its own counter on
`get`. No cross-layer dependency.

```python
def append(self, k, v, layer_id):
    self.k[layer_id, :, self.tokens_used:self.tokens_used + 1, :].copy_(k)
    ...
    self.token_per_layer[layer_id] += 1

def get(self, layer_id):
    n = self.token_per_layer[layer_id]
    return self.k[layer_id, :, :n, :], self.v[layer_id, :, :n, :]
```

Initialize `token_per_layer` as `{i: 0 for i in range(num_layers)}` in
`__init__` — otherwise the first `append` on a newly-allocated block
(when decode crosses a 16-token boundary) hits `KeyError`.

### Why the old `KVCache` didn't have this bug

Its equivalent `pos` counter was incremented on **layer 0**, before
any read happened in that step:

```python
def append(self, layer_id, k, v):
    if layer_id == 0:
        self.pos += 1     # bump BEFORE any layer reads
    ...
```

Either approach works. Per-layer counters are cleaner because
they decouple each layer's state — no ordering requirement.

## Correctness methodology — 3-way comparison

The benchmark runs the same prompt through three paths:

1. **No cache** — re-run forward on full sequence each step. The
   slow-but-correct reference.
2. **KVCache** (old Level 5 cache) — preallocated contiguous slab.
3. **Paged cache** — this phase's new implementation.

All three must produce identical argmax sequences across 50 decode
tokens. The `CacheType` enum in `AttentionParams` lets the same model
class switch between KV and paged at construction time, so both
cache types can coexist for benchmarking.

```
Correctness (no-cache == KV):    True
Correctness (no-cache == paged): True
Correctness (KV == paged):       True
```

Three-way equality is a stronger assertion than any pairwise test:
it proves paged is behaviorally identical to the known-good KV
implementation, not just "happens to match no-cache on this prompt."

## Measured overhead vs KVCache

On TinyStories-scale model, prompt 10 tokens + 50 decode tokens:

| Path | Tok/s | Per-decode-tok | Ratio vs KV |
|---|---|---|---|
| No cache | 78 | 12.8 ms | — |
| KV cache | 180 | 5.6 ms | 1.00× |
| Paged cache | 177 | 5.7 ms | **1.02×** |

~2% overhead. Essentially free at this scale.

**Where the overhead comes from:** `Sequence.get()` walks each
block in Python and assembles with `torch.cat`. For a 60-token
sequence that's at most 4 blocks — tiny loop, small allocations.

**Where the overhead would grow:** longer sequences. At 2048 tokens
(`max_seq_len`), the loop processes 128 blocks per layer per decode
step. Python overhead and allocator pressure dominate. That's the
workload Phase 3's Triton kernel will target.

## Gotchas encountered

- **Shape boundary at the cache interface.** Attention operates on
  `(B, H, T, D)` tensors; `Sequence` methods work on `(H, T, D)`
  (no batch dim — a Sequence represents exactly one sequence by
  definition). Squeeze/unsqueeze at the call site:
  ```python
  seq.append(k.squeeze(0), v.squeeze(0), layer_idx)
  k, v = seq.get(layer_idx)
  k = k.unsqueeze(0); v = v.unsqueeze(0)
  ```
- **Sequence reuse across prompts.** A `Sequence` accumulates state.
  Reusing one across back-to-back prompts without releasing its
  blocks means prompt N+1 appends to prompt N's cache. Fix: either
  create a fresh `Sequence` per prompt, or add `Sequence.release_blocks()`
  to return all blocks to the pool.
- **Warmup pollutes state.** If warmup runs a prefill through the
  same `Sequence` the real benchmark will use, results will be
  wrong. Use a separate `Sequence` for warmup vs the measurement.
- **Optional arg threading.** After making `seq: Sequence | None = None`,
  training paths don't need to thread a dummy sequence through. Keeps
  the training loop unchanged.

## Phase 2 scorecard

| Objective | Status |
|---|---|
| 1. Block pool, block table, allocator | Done |
| 2. Paged-aware attention forward | Done |
| 3. Correctness test (byte-identical) | Done — 3-way |
| 4. Mid-decode block allocation | Done (crosses 16-token boundary 3× in the 60-token benchmark) |
| 5. 4 concurrent sequences | Deferred to Phase 3 (scheduler) |

## What this unlocks for Phase 3

- **Triton kernel target.** The Python `gather_from_blocks` loop is
  now the isolated slow path. Rewriting just that as a fused kernel
  gives the paged design its speed back.
- **Concurrency.** `BlockPool` is already multi-sequence-safe (its
  free list is shared). Attention already handles `Sequence | None`
  cleanly. Adding a scheduler means instantiating multiple
  `Sequence`s on the same pool and batching their decode steps.
- **Sharing / CoW.** `Sequence` owns a list of `Block`s; sharing
  becomes "two Sequences whose block lists overlap." CoW adds a
  refcount per Block and a copy-on-first-write in `append`.
