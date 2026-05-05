# Session checkpoint — 2026-05-04 evening (Phase 2 complete)

Load with: `please load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md`

## State

- Repo: `/Users/katayal/Documents/llm/AllAboutLLMs`, branch `main`,
  remote `github.com/tkashish/all-about-llms`.
- Phase 2 of the vLLM learning plan: **done and committed**.
  Paged cache produces byte-identical outputs to both the no-cache
  reference and the old KVCache implementation.
- 6 wiki notes under `wiki/gpu/vllm/` covering motivation through
  Phase 2 implementation.

## User

- Kashish Tayal, self-studying LLM internals.
- **Teaching style is mandatory**:
  `/Users/katayal/Documents/llm/cs336/TEACHING-STYLE.md`. Plain
  English → problem → mechanics, one idea per message, ~15 lines max
  before check-in, concrete numbers over abstract math. User writes
  the code; Kiro explains / reviews / debugs.

## What got done 2026-05-04

Phase 2 — replaced the Level 5 KV cache with a paged cache in pure
PyTorch, kept both implementations alive behind a `CacheType` enum
for A/B testing, proved 3-way byte-identical correctness.

**Files created/modified:**

- `src/transformer/paged_cache.py` — `Block`, `BlockPool`, `Sequence`.
  Per-layer token counters (`token_per_layer` dict) so each layer's
  `get()` returns exactly the slots that layer has appended, no
  cross-layer ordering dependency.
- `src/transformer/attention.py` — added `CacheType` enum (KV or
  PAGED). Attention branches on `cache_type` and either talks to
  `self.kv_cache` (old) or the `seq: Sequence` arg (paged).
  Squeeze/unsqueeze at the Sequence boundary to bridge
  `(B, H, T, D)` ↔ `(H, T, D)`.
- `src/transformer/model.py` — `HyperParams.cache_type` with
  `PAGED` default. `Model.__init__` builds a KVCache only when
  `cache_type == KV`.
- `src/transformer/transformer.py`, `inference.py` — thread
  `seq: Sequence | None = None` through `forward()`.
- `src/transformer/benchmark_inference.py` — 3-way benchmark:
  no-cache, KV cache, paged cache. Same-model argmax equality check
  across all pairs.
- `wiki/gpu/vllm/06-paged-cache-impl.md` — Phase 2 implementation
  notes: design, the subtle off-by-one bug and why per-layer
  counters fix it, 3-way correctness methodology, measured
  overhead, gotchas, Phase 3 unlocks.
- `wiki/gpu/vllm/05-sharing-and-cow.md` — updated closing to link
  forward to 06 instead of the learning plan.

## Key findings locked in

### The off-by-one bug

Naive impl tried to update a single `tokens_used` counter on the
last layer's write. But each decode step processes layers in order:
each layer writes then reads. Layer 0's read happens **before**
layer N-1's write, so layer 0 sees stale `tokens_used` from the
previous step. Symptom: paged step N outputs match no-cache step N-1.

Fix: per-layer counter dict. Each layer's append/get uses its own
counter. No cross-layer ordering.

### Overhead measurement

Paged ~2% slower than KVCache on a 60-token workload. Overhead is
the Python loop + `torch.cat` in `Sequence.get()`. Will grow with
longer sequences; exactly the workload the Phase 3 Triton kernel
will target.

### Shape boundary

Attention: `(B, H, T, D)`. Sequence: `(H, T, D)` — a Sequence
represents one sequence, no batch dim. Squeeze/unsqueeze at the
three call sites in `Attention.forward()` bridge the two.

## Where we paused

Phase 2 complete. Optional remaining Phase 2 item — objective #5,
"4 concurrent sequences with different lengths" — was deferred.
It overlaps heavily with the Phase 3 scheduler, so we'll tackle it
there.

Next session can start Phase 3:

1. Learn Triton basics. Trivial vector-add kernel first.
2. Write a PagedAttention decode kernel. Start naive.
3. Benchmark Triton paged vs pure-Py paged vs KVCache.
4. Build a minimal scheduler. Continuous batching over multiple
   concurrent sequences. Free blocks on completion.
5. Measure aggregate tokens/sec, watch it approach the compute-ridge
   ceiling.

See `wiki/gpu/vllm-learning-plan.md` for full Phase 3 details.

## Dangling threads

1. **GPU rental** — still open. Needed for Phase 3 Triton work.
   Cloud Desktop on MPS/Metal won't cut it.
2. **Chunked prefill** — noted in continuous-batching doc, not yet
   explored.
3. **Sequence.release_blocks()** exists but is not hooked into
   request lifecycle anywhere. Will matter once a scheduler is
   preempting sequences.
4. **Warmup pollution** — `benchmark_inference.py` handles this for
   paged by rebuilding `Sequence` between warmup and real runs.
   The KVCache path handles it by rebuilding the whole model,
   which is wasteful; could add a reset method.
5. **`generate_with_kv_cache` and `generate_with_paged_cache`** are
   nearly identical. Good candidate for refactoring into one
   function that takes a cache object once a common interface
   exists.
6. **Excalidraw line-height quirk** — from 2026-05-01 session,
   still open, not blocking.
7. **Earlier dangling threads** (RoPE impl notes, pytorch wiki
   holes, SwiGLU walkthrough, train a bigger model) — still open.

## Resume prompt

> "Load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md,
>  then kick off Phase 3 of the vLLM learning plan (Triton)."
