# vLLM learning plan

Goal: understand vLLM deeply enough to **reason about it** and
**reimplement its key ideas** on top of the existing `AllAboutLLMs`
transformer + KV cache.

## Guiding principles

- Build on what's already here. Extend `src/transformer/` rather than
  starting a new repo — the point is to feel PagedAttention replace
  the current Level 5 cache on a model you wrote.
- Python + PyTorch first (correctness), Triton second (speed). Complexity
  arrives in order: ideas → kernels.
- Hardware: single RTX 4090 or A100 rented hourly is plenty. Cheap
  options: Vast.ai, RunPod, Lambda Labs, Modal.
- Teaching style: one idea per message, ~15-line cap, plain English
  → problem → mechanics. User writes the code; Kiro explains / reviews
  / debugs. See `/Users/katayal/Documents/llm/cs336/TEACHING-STYLE.md`.

## Prerequisites already in place

- Working decoder-only transformer with RMSNorm, RoPE, SwiGLU.
- KV cache Level 5 (preallocated, caller-managed, per-layer 5D tensor).
- Benchmark harness that compares with-cache vs no-cache paths and
  asserts output equivalence.
- Conceptual foundation: compute-vs-memory, arithmetic intensity,
  why decode is memory-bound, why batching helps, why memory capacity
  caps batch size below the compute ridge. See `wiki/gpu/01-05`.

## Phase 1 — Foundations (no code)

Understand vLLM from the outside. Goal: be able to draw the request
lifecycle on a whiteboard and name every component.

**Objectives:**

1. Read the PagedAttention paper (Kwon et al. 2023) end-to-end.
2. Understand the three memory-waste categories naive KV cache
   suffers from: internal fragmentation, reservation waste, external
   fragmentation. Be able to give a concrete example of each.
3. Understand block tables: how a logical (sequence, position) pair
   maps to a physical (block, slot). Why indirection enables packing.
4. Understand continuous batching: the scheduler-level idea. How
   prefill and decode interleave. Why it needs PagedAttention to
   work at scale.
5. Understand sharing: prefix caching, copy-on-write, beam search
   sharing. One concrete scenario each.

**Deliverables:**

- `wiki/gpu/vllm/01-motivation.md` — the 66-vs-330 gap, three
  fragmentation types, why naive allocation fails.
- `wiki/gpu/vllm/02-block-table.md` — diagrams + worked example
  of logical↔physical mapping.
- `wiki/gpu/vllm/03-continuous-batching.md` — request lifecycle.
- `wiki/gpu/vllm/04-sharing-and-cow.md` — prefix caching, CoW.

No code in this phase. Pure reading + notes.

## Phase 2 — PagedAttention in pure PyTorch

Replace the Level 5 cache with a paged cache on the existing model.
Correctness over speed; we want outputs byte-identical to the current
implementation.

**Objectives:**

1. Define the paged KV cache data structure:
   - Global block pool: tensor `(num_blocks, 2, L, H, block_size, D_head)`
     (or similar — user decides layout).
   - Per-sequence block table: list of block indices.
   - Free list + block allocator.
2. Rewrite the attention forward pass to read K/V via the block table.
   Pure PyTorch `gather` or indexing. No kernel work yet.
3. Rewrite the benchmark to compare Level 5 vs paged, assert identical
   argmax outputs. (Same correctness test that caught the RoPE bug.)
4. Support mid-decode block allocation — when a sequence crosses a
   block boundary, allocate a new block and extend its block table.
5. Add a simple scenario: run 4 sequences concurrently with different
   lengths, watch block-pool utilization.

**Deliverables:**

- `src/transformer/paged_cache.py` — BlockPool, BlockTable, allocator.
- Modified attention forward supporting paged reads.
- `src/transformer/benchmark_paged.py` — correctness + memory
  utilization comparison.
- `wiki/gpu/vllm/05-paged-cache-impl.md` — design notes, gotchas.

## Phase 3 — Kernelize + scheduler

Make it actually fast, and add the scheduler that makes it useful.

**Objectives:**

1. Learn Triton basics: tile, pid, mask, load/store, auto-tuning.
   Write a trivial vector-add kernel first.
2. Write a Triton PagedAttention kernel for decode (single-token query
   against a paged K/V). Start naive; optimize later.
3. Benchmark against the pure-PyTorch paged version and the original
   Level 5 cache. Expect Triton ≈ Level 5, pure-Py paged << both.
4. Build a minimal scheduler:
   - Request queue with prefill and decode phases.
   - Each step: pick a batch (continuous batching), run one decode
     step across mixed sequences, free blocks on completion.
   - Simulate many concurrent requests with varying lengths.
5. Measure aggregate tokens/sec vs single-stream, watch it approach
   the compute-ridge ceiling.

**Deliverables:**

- `src/inference/paged_attention_kernel.py` — Triton kernel.
- `src/inference/scheduler.py` — continuous batching.
- `src/inference/server.py` — tiny stdin-driven request driver.
- `wiki/gpu/vllm/06-triton-kernel.md` — Triton tour, kernel walkthrough.
- `wiki/gpu/vllm/07-scheduler.md` — continuous batching, lifecycle.
- `wiki/gpu/vllm/08-final-benchmarks.md` — end-to-end numbers.

## Reading list

- **Paper:** Kwon et al., "Efficient Memory Management for Large
  Language Model Serving with PagedAttention" (SOSP 2023).
- **Source:** vllm-project/vllm on GitHub. Key dirs to read *after*
  Phase 1: `vllm/core/` (scheduler, block manager),
  `vllm/attention/` (kernels), `vllm/engine/`.
- **Triton:** openai/triton docs + tutorials (matmul, attention).
- **FlashAttention** (Dao et al.) — useful context for why the
  attention kernel is organized the way it is; optional side reading.

## Open questions to resolve as we go

- Do we implement prefix caching / copy-on-write (Phase 2.5 stretch)?
- Do we bother with speculative decoding on top (probably no — out
  of scope).
- Triton version compatibility — pin a version once the 4090 is up.

## Resume prompt

> "Load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md,
>  then start Phase 1 of the vLLM learning plan."
