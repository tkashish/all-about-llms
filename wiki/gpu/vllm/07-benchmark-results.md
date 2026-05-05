# Phase 2 benchmark results

Numbers from `src/transformer/benchmark_inference.py` comparing three
inference paths side-by-side on the same prompt and model weights.

## Setup

- **Device:** Apple Silicon MPS (local machine; not a rented GPU).
- **Model:** TinyStories-trained decoder-only transformer from
  `src/transformer/`, loaded from `data/model/model.pt`.
- **Prompt:** `"Once upon a time, there was a little girl"` (10
  tokens).
- **Workload:** generate 50 new tokens with argmax sampling.
- **Block size:** 16 tokens per block (so the 60-token sequence
  crosses 3 block boundaries during decode).

## Three paths compared

| Path | Cache | Forward shape |
|---|---|---|
| No cache | none | Re-runs full sequence every step; O(N²) |
| KV cache | Level 5 contiguous slab | `KVCache` from `kv_cache.py` |
| Paged cache | Fixed-size blocks | `Sequence`/`Block`/`BlockPool` from `paged_cache.py` |

All three walk through the identical `Attention.forward` with
`CacheType` branching to either `self.kv_cache` or the `seq: Sequence`
argument.

## Representative run

```
Prompt: 'Once upon a time, there was a little girl'
Prompt tokens: 10
Generating 50 new tokens (argmax sampling)

============================================================
RESULTS
============================================================

No cache:
  Total:    0.64 s
  Per tok:  12.8 ms
  Tok/s:    78.3
  First step: 76.0 ms
  Last step:  9.6 ms

KV cache:
  Total:    0.28 s
    Prefill:  2.2 ms
    Decode:   0.28 s
  Per decode tok: 5.7 ms
  Tok/s (incl. prefill): 177.0
  First decode step: 15.7 ms
  Last decode step:  6.6 ms

Paged cache:
  Total:    0.35 s
    Prefill:  2.1 ms
    Decode:   0.35 s
  Per decode tok: 7.1 ms
  Tok/s (incl. prefill): 142.6
  First decode step: 8.4 ms
  Last decode step:  5.4 ms

Speedup KV vs no-cache:    2.3x
Speedup paged vs no-cache: 1.8x
Paged vs KV (ratio):       1.24x  (>1 = paged slower)

Correctness (no-cache == KV):    True
Correctness (no-cache == paged): True
Correctness (KV == paged):       True
```

## Run-to-run variance

Four back-to-back runs on the same machine:

| Run | KV tok/s | Paged tok/s | Paged/KV ratio |
|---|---|---|---|
| 1 | 177.0 | 142.6 | 1.24× |
| 2 | 151.1 | 148.3 | 1.02× |
| 3 | 163.0 | 148.6 | 1.10× |
| 4 | 171.5 | 152.7 | 1.12× |

Median overhead around **10%**. Variance is high because this is a
tiny model on MPS — Python / kernel-launch overhead and thermal
noise dominate at 5–7 ms per decode step. Numbers should be read
as "paged is in the same ballpark as KV, slightly slower."

## Correctness — the real takeaway

Every run: **all three paths produce byte-identical argmax
sequences** across 50 decode tokens. Including the strong
KV-vs-paged assertion, which proves paged is a true drop-in for the
old cache — not just "happens to agree with no-cache by accident."

This is the main Phase 2 deliverable. Overhead numbers are
informational; correctness is the milestone.

## Where the paged overhead comes from

`Sequence.get()` walks each block in Python and concatenates:

```python
def get(self, layer_id):
    k_chunks, v_chunks = [], []
    for block in self.blocks:
        k, v = block.get(layer_id)
        k_chunks.append(k); v_chunks.append(v)
    return torch.cat(k_chunks), torch.cat(v_chunks)
```

For this 60-token run that's at most 4 blocks. Tiny loop, small
allocations — hence only ~10% overhead. At longer sequences
(e.g. `max_seq_len=2048`, 128 blocks) the Python loop + repeated
allocator pressure would dominate, widening the gap meaningfully.

## What Phase 3 changes

Swap the Python loop for a **single fused Triton kernel** that reads
directly from `BlockPool.blocks` using the block table as an index.
No Python loop, no `cat`, no materialized intermediate K/V tensor.
Should close the gap with KVCache (or exceed it) while preserving
all the memory-management advantages of paging.

## Reproducing

```sh
uv run python -m transformer.benchmark_inference
```

Tweak `NUM_NEW_TOKENS` or `PROMPT` at the top of the script to
explore longer-workload behavior. Be aware that `max_seq_len` on
the checkpoint bounds the total context length.
