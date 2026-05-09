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

## Phase 3 — Build a minimal vLLM-style inference engine

Make paged attention fast (Triton kernel) and useful (a scheduler).
Build the mental model for *why* specific designs are fast — so
every efficiency choice feels intuitive, not memorized.

**Goal:** show a clear throughput gap between naive single-request
serving and continuous-batched paged serving, and **explain the gap
with profiler evidence.** Not a hard "close the 2–4× gap" target —
the exact gap depends on model size, GPU, batch size, sequence
lengths, kernel quality, memory bandwidth, and scheduler policy.

Every objective has a **success condition.** Don't move on until
you can meet it.

### 1. GPU mental model — hardware + execution

**Learn:** Model A (SMs, CUDA vs Tensor cores, memory hierarchy)
and Model B (grid → block → warp → thread, three levels of
hardware scheduling).

**Success condition:** draw both diagrams from memory; explain
what the programmer controls vs what the hardware does; explain
what a "warp" actually is and why it's 32 threads wide.

### 2. Learn Triton basics

**Learn:** `@triton.jit`, `tl.load`, `tl.store`, `tl.arange`,
`tl.program_id`, masks, autotuning. The mental shift: a Triton
program handles a **block** of data, not a single element.

**Build** — progressively harder warm-up kernels, each verified
against a PyTorch reference:

- `vector_add_kernel` — launch, pid, masks, load/store.
- `copy_kernel` / `scale_kernel` — single-pointer variants.
- `row_sum_kernel` — first reduction across an axis.
- `softmax_kernel` — reductions + numerical stability (max-subtract
  trick). Direct lead-in to attention.

**Success condition:** given "write a kernel that does X across a
1D/2D tensor," you can sketch the offsets, mask, and load/store
without looking up syntax.

### 3. Kernel performance mental model — hands-on, not abstract

**Learn:** the concepts that govern whether a kernel uses the GPU well.

- **Memory coalescing** — threads in a warp touching contiguous
  memory so one load services all 32 threads.
- **Occupancy and latency hiding** — many resident warps per SM so
  the scheduler can always find work while others stall on memory.
- **Warp divergence** — what happens when threads in a warp take
  different `if` branches.
- **Register pressure** — how per-thread register use caps how many
  warps fit per SM.
- **Arithmetic intensity** — FLOPs per byte loaded; decides
  memory-bound vs compute-bound.

**Build — "intentionally-bad kernels" exercise.** Take `vector_add`
and produce versions that are:

- Non-coalesced (threads stride > 1 apart).
- Block size way too small (e.g. 8 threads).
- Block size way too large.
- With unnecessary branching on `pid`.
- With redundant memory reads.

Benchmark each and compare to the coalesced baseline. **This
hands-on exercise is the single best way to internalize the concepts.**

**Success condition:** given a slow kernel, you can hypothesize
*which* category explains it (memory-bound? bad coalescing?
launch-overhead-dominated? register-pressure-limited?) and check
with a profiler.

### 4. Correctness + profiling harness — before optimizing anything

Build this **before** any attention kernel. Every subsequent kernel
gets verified + profiled through it.

**Correctness progression:**

```
torch reference attention
        ↓
naive Triton attention
        ↓
paged Triton attention
        ↓
optimized paged Triton attention
```

For each version, measure: `max_abs_error`, `max_relative_error`,
kernel time, tokens/sec, GPU memory used. Use fixed random seeds.

**Test matrix for attention specifically:**
- sequence length ∈ {1, 17, 128, 1024, not-divisible-by-block}
- batch size ∈ {1, 2, 4} with mixed sequence lengths

**Profiling tools** (learn early, use continuously):

- `torch.cuda.Event` — simple wall-clock between points.
- `torch.profiler` — which ops are slow, where CUDA syncs happen.
- **Nsight Systems (`nsys`)** — timeline view, kernel launches,
  host-device overlap.
- **Nsight Compute (`ncu`)** — per-kernel deep dive: memory
  throughput, occupancy, warp stall reasons. The most actionable
  tool.

Reference: NVIDIA CUDA Best Practices Guide — profiler-driven
optimization workflow.

**Success condition:** you trust your benchmark numbers. No "I
think it works." Every kernel has a correctness test you run
before trusting any performance number.

### 5. Contiguous KV decode attention — Triton baseline

Before paged, build normal decode attention in Triton.

**Build:** Triton kernel for the decode case:
```
q:       [num_heads, head_dim]              (one new token)
k_cache: [seq_len, num_heads, head_dim]     (contiguous)
v_cache: [seq_len, num_heads, head_dim]
output:  [num_heads, head_dim]
```

Keep the first version naive and correct: load Q, loop K/V blocks,
compute QK, softmax, multiply by V, store. Covers:

- Q @ K for one new token vs all prior.
- Softmax + scaling (safe, max-subtract).
- Softmax @ V.

**Success condition:** replace PyTorch decode attention with this
kernel; outputs match bit-for-bit (within fp16 tolerance).

### 6. Online softmax — the core algorithmic trick

Dedicated stage. This is the most important algorithmic idea in
Phase 3.

**The problem:** naive softmax wants the full attention-score
vector in memory. Long contexts + paged K/V make this infeasible.

**The fix:** maintain three running values as you stream through
K/V blocks:
- `m` — running max of scores seen so far.
- `l` — running sum of `exp(score - m)`.
- `acc` — running weighted-V accumulator.

Each new block: compute its local max, rescale old `l` and `acc`
by `exp(old_m - new_m)` if the new max is larger, add the new
block's contribution. At the end, divide `acc / l`.

**Learn:**
- Log-sum-exp stability.
- Why we keep `m` *and* `l` (not just one).
- Why rescaling is needed when a new larger max appears.
- Why this lets you process blocks independently without
  materializing the full score vector.

**Success condition:** you can explain, without notes, exactly
why the "rescale when new max appears" step is necessary and
what goes wrong if you skip it. This is what FlashAttention is
built on.

### 7. KV block manager — runtime memory system

Phase 2 built `BlockPool` / `Sequence` as a data-structure
exercise. Phase 3 uses it as a runtime component both the kernel
and the scheduler talk to.

**Build:** `KVBlockManager` supporting:
- Fixed-size KV blocks (e.g. `block_size=16`).
- Free list + pool.
- `request_id → block_table` mapping.
- `append_token(request_id, k, v)` — allocate a new block if the
  last one is full.
- `free(request_id)` on request completion.
- Hook for reference counts (CoW prep, not implemented yet).

**Success condition:** given a `request_id` and a `logical_token_index`,
you can find the `(physical_block_id, offset_in_block)` and the
K/V vectors for that token. This mapping must feel automatic.

### 8. Paged KV cache layout

**Learn:** memory layout decisions matter for coalescing. For
learning, start with the simplest:

```
K_cache: [num_blocks, block_size, num_heads, head_dim]
V_cache: [num_blocks, block_size, num_heads, head_dim]
```

Alternative layouts (e.g. `[num_blocks, num_heads, block_size, head_dim]`)
can improve coalescing for certain access patterns. Worth
experimenting with after the kernel works.

**Build:** layout-aware accessors:
- `write_token_to_kv_cache(request_id, logical_idx, k, v)`
- `read_logical_token(request_id, logical_idx)`
- `debug_dump_request_cache(request_id)`

**Success condition:** layout decision is explicit and documented;
you know which access pattern it optimizes for.

### 9. Paged decode attention kernel — the main event

Build in stages, each verified via the correctness harness before
moving on.

- **9A.** Single request, single head, small `head_dim` (e.g. 64).
  Replace contiguous K/V with block pool + block table. Keep
  everything else identical to Obj 5.
- **9B.** Single request, multiple heads.
- **9C.** Multiple requests, each with its own block table.
- **9D.** Variable sequence lengths in the same batch (the real
  continuous-batching use case).
- **9E.** GQA / MQA (`num_query_heads > num_kv_heads`) — modern
  LLM requirement; add after basic multi-head works.
- **9F.** Optimize memory layout + SRAM tiling. Use `ncu` to
  confirm each change helps.

**Success condition:** model runs end-to-end with paged Triton
attention replacing the Phase 2 Python `Sequence.get()` loop,
and outputs match bit-for-bit.

### 10. 3-way kernel benchmark

Under the correctness + profiling harness:

- Pure-PyTorch KVCache (Phase 2 baseline).
- Contiguous Triton KV (Obj 5).
- Paged Triton KV (Obj 9).

**Expect:** contiguous Triton ≈ KVCache; paged Triton slightly
slower on a single seq due to block-table indirection. That's
fine. Paged wins when **many** varied-length seqs run concurrently
(see Obj 12).

**Quantify with profiler evidence, not just tokens/sec.** Use `ncu`
to compare memory throughput and occupancy.

**Success condition:** you can explain *why* paged is slower than
contiguous on a single seq and *why* that reverses under concurrency.

### 11. Simple continuous-batching scheduler

Move from:
```
finish A → then B → then C
```
to:
```
decode(A, B, C) together, per step.
remove finished. admit new.
```

**Build:** scheduler with `waiting_queue`, `running_requests`,
`finished_requests`, the KV block manager, a decode step, and
on-completion cleanup. Each request tracks prompt, generated
tokens, `max_new_tokens`, current length, block table, status.

First version: prefill each request completely, then place it in
the decode batch. No chunked prefill yet.

```
while requests are active:
    pick active requests
    run one batched decode step
    sample next token
    append token to each request
    update KV cache
    free completed requests
    admit new requests
```

**Success condition:** many concurrent fake requests generate
outputs without waiting for each to finish. Multi-seq outputs
match single-seq outputs when replayed individually.

### 12. Aggregate throughput benchmark — where Phase 3 pays off

Measure on multiple workload distributions:
- Short prompts, short outputs.
- Long prompts, short outputs.
- Short prompts, long outputs.
- Mixed random lengths.

**Metrics** (richer than Phase 2's tokens/sec):
- **TTFT** — time to first token.
- **ITL / TPOT** — inter-token latency / time per output token.
- **End-to-end latency** per request.
- **p50 / p95 / p99** latency distributions.
- **GPU memory used** over time.
- **KV cache fragmentation / waste**.
- **Throughput vs concurrency** — sweep N concurrent seqs.
- **Throughput vs context length** — sweep seq length.
- **Max concurrent requests before OOM.**

Compare: naive single-stream vs continuous batching vs paged +
continuous batching.

**Success condition:** you can produce a table that shows a clear
throughput gap and explain *why* via profiler evidence and the
memory-efficiency argument. The exact numbers don't matter; the
explanation does.

### 13. Chunked prefill — avoid prefill starvation

Only after Obj 11 works.

**Problem:** request A has an 8,000-token prompt. Requests B, C, D
are decoding. If prefill monopolizes a step, B/C/D see terrible
inter-token latency.

**Fix:** per-step token budget, mixing prefill chunks with decode
tokens.

```
max_tokens_per_step = 2048
step = [decode_tokens_from_B_C_D...] + [prefill_chunk_of_A...]
```

**Success condition:** a long prompt arriving mid-flight no longer
visibly degrades latency for active decode requests.

### 14. Unified token-budget scheduler (vLLM V1 style)

Beyond chunked prefill: treat prefill and decode tokens uniformly.
Allocate a per-step token budget across all requests. This is how
modern vLLM actually schedules.

**Success condition:** scheduler logic doesn't special-case prefill
vs decode — it's all "tokens to process this step."

### 15. Optional Phase 3.5 — advanced techniques

Only after the core system works:

- **Prefix caching** end-to-end. Block hashing, refcounts, CoW.
- **CUDA graphs** — capture a full decode step as a graph to cut
  launch overhead. Depends on repeated shapes, which decode has.
- **Speculative decoding prep** — design the draft/target split,
  don't implement yet.

## What NOT to do in Phase 3

Keep focus. These belong to Phase 4+:

- Tensor parallelism / multi-GPU.
- Distributed serving.
- Training-grade FlashAttention kernel (forward+backward).
- Quantization (fp8/int4/AWQ/GPTQ).
- Full speculative decoding implementation.
- OpenAI-compatible HTTP server.
- Production-grade API (auth, rate limiting, etc.).

Phase 3 is about: **KV memory, decode kernel, scheduler, throughput.**
Everything else waits.

## Suggested repo structure

```
src/inference/
  01_triton_basics/
    vector_add.py
    row_sum.py
    softmax.py
    benchmark.py
  02_kernel_harness/
    correctness.py
    timing.py
    profiler_notes.md
  03_contiguous_decode_attention/
    torch_reference.py
    triton_decode_attention.py
    tests.py
    benchmark.py
  04_kv_block_manager/
    block_manager.py
    tests.py
  05_paged_attention/
    paged_kv_cache.py
    paged_decode_attention.py
    tests.py
    benchmark.py
  06_scheduler/
    request.py
    scheduler.py
    continuous_batching.py
    benchmark.py
  07_aggregate_benchmarks/
    workloads.py
    run_benchmark.py
    results.md
```

Phase 2's `src/transformer/paged_cache.py` evolves into
`src/inference/04_kv_block_manager/` in the new layout. Phase 2's
`benchmark_inference.py` graduates to `07_aggregate_benchmarks/`.

## Phase 3 deliverables

Code:
- `src/inference/paged_attention_kernel.py` — Triton kernel.
- `src/inference/scheduler.py` — continuous batching + chunked prefill.
- `src/inference/server.py` — tiny stdin-driven request driver.
- `src/inference/profiling_harness.py` — correctness + profiling harness.

Wiki:
- `wiki/gpu/06-execution-model.md` — grid/block/thread mapping (done).
- `wiki/gpu/07-efficient-kernels.md` — coalescing, occupancy,
  tiling, divergence, register pressure.
- `wiki/gpu/08-profiling.md` — Event, profiler, nsys, ncu walkthrough.
- `wiki/gpu/vllm/08-triton-kernel.md` — Triton tour + paged-attention
  walkthrough (including online softmax).
- `wiki/gpu/vllm/09-scheduler.md` — continuous batching, chunked
  prefill, token-budget scheduler.
- `wiki/gpu/vllm/10-final-benchmarks.md` — end-to-end numbers.

## Phase 3 final deliverable

A mini inference server that can:
- Accept multiple concurrent requests.
- Prefill prompts.
- Allocate and free KV blocks.
- Run paged decode attention.
- Continuously batch active requests.
- Report throughput and latency.

And the ability to explain:
- Why KV cache dominates inference memory.
- Why decode is memory-bound.
- Why paged KV helps serving but not single-request latency.
- Why one-request benchmarks understate paged's value.
- Why continuous batching matters.
- Why scheduler design affects GPU utilization.

**The explanations matter more than the numbers.**

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
