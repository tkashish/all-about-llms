# KV cache — inference optimization for autoregressive generation

## What problem does it solve?

Generating text with an LLM is autoregressive: predict token t+1 from
tokens 0..t, then predict token t+2 from tokens 0..t+1, and so on.

The naive approach reruns the entire forward pass on the whole
growing sequence at every step. That's O(N²) work to generate N
tokens — wasteful, because most of the computation was done in the
previous step and hasn't changed.

**KV cache = store K and V tensors from earlier positions and reuse
them.** Brings generation from O(N²) to O(N).

## Prefill vs decode — two phases, different code paths

Inference splits into two phases with different shapes and different
bottlenecks.

### Prefill: processing the prompt

User gives you a prompt of P tokens. One forward pass over all P
tokens at once. You only need the logits at the LAST position to
sample the first generated token, but you still compute K and V at
every position (to save for decode) and you still compute full Q
through most of the network (see "Why Q for all positions?" below).

- **Input shape:** `(B, P, D)`
- **Attention shape inside:** `Q, K, V ∈ (B, H, P, D_head)`
- **Output you keep:** logits at position P-1 only
- **Side effect:** K and V at every position, every layer → saved to cache

### Decode: generating one token at a time

For each new token:

1. Compute Q, K, V for the ONE new position.
2. Look up cached K, V for all earlier positions.
3. Append new K, V to the cache.
4. Run attention with Q = just-new-token, K,V = full cache.
5. Sample next token from logits.

- **Input shape:** `(B, 1, D)` — one token
- **Q shape inside:** `(B, H, 1, D_head)` — one token's query
- **K, V shape inside:** `(B, H, t+1, D_head)` — full history
- **Output:** logits at position t, used for sampling

Prefill is compute-bound (big matmuls). Decode is memory-bound (tiny
matmuls, lots of memory reads for the cache).

## Why only K and V, not Q?

Every token produces Q, K, V at every layer. The difference is how
they're USED.

At attention for output position `i`:

```
out_i = Σ_j  softmax(Q_i · K_j / √D) · V_j
```

- **Q_i** — used only to compute output for position i. After that, never touched again.
- **K_j, V_j** — used by every future position that attends to j. Re-read at every generation step.

At generation step `t`:
- `Q_t` fresh computation, one-shot use → no value in caching
- `K_0..K_{t-1}` reused for step t's attention, and for step t+1, t+2, ...
- Same for V

So:

| vector | lifetime | cache? |
|---|---|---|
| Q | 1 step | no — wasted memory |
| K | unbounded | yes |
| V | unbounded | yes |

Hence "KV cache" — only K and V.

## Why compute Q at positions 0..P-2 during prefill?

Sharp question: during prefill we only want `logits[P-1]`. So do we
need Q for earlier positions?

**Final layer:** No. `logits[P-1]` only needs `Q_{P-1}` at the final
layer's attention. Q for positions 0..P-2 at the final layer is
unused.

**Earlier layers:** Yes. Here's why.

To compute `K_j` or `V_j` at layer L, you need `x_j` at layer L
(the input to that layer at position j). That input is the OUTPUT of
layer L-1 at position j, which was produced by attention at layer L-1
using `Q_j` at layer L-1.

So Q at positions 0..P-2 is needed at layers 0 through L-1 to produce
the intermediate hidden states, which become the inputs to layer L's
K and V projections.

**Only the very last layer can skip Q for positions 0..P-2.**

| position range | Q at layers 0..L-1? | Q at layer L? |
|---|---|---|
| 0 to P-2 | yes (feeds K, V of next layer) | no (logits thrown away) |
| P-1 | yes | yes |

**In practice:** most codebases compute full Q at every layer including
the last, because the saving at the last layer is tiny (~3% of prefill
compute) and special-casing it adds code complexity. Production
inference engines like vLLM do apply this last-layer optimization;
research codebases usually don't.

## Level 1 — Naive (no cache)

The starting point. Re-run the full forward pass on the entire
growing sequence every step.

```python
tokens = prompt
for step in range(max_new):
    logits = model(tokens)              # forward on ALL tokens every step
    next_id = sample(logits[:, -1])
    tokens = torch.cat([tokens, next_id], dim=1)
```

Every step recomputes attention, MLP, norms for positions whose
values haven't changed since last step.

**Cost to generate N tokens from prompt of length P:** each step
runs forward on a sequence of length `P + step`, which is O((P+step)²)
attention. Summed over N steps, total is roughly O(N³).

**Why it's wrong:** K and V at old positions don't change once
computed. They're the same data you computed last step. Re-doing all
that work is the problem the KV cache solves.

This is what the simplest scripts (including the one originally in
`scratch/transformer/inference.py`) do. Fine for debugging, unusable
for serving.

## Level 2 — Basic per-layer cache

**Idea:** every attention layer keeps K and V from earlier positions.
At each new step, compute fresh K, V only for the NEW token(s) and
append to the layer's cached K, V. Then run attention.

### The per-step shape game

For one attention layer, with `T_cached` positions already processed
and `T_new` new tokens arriving this step:

| tensor | shape | notes |
|---|---|---|
| `Q_new` | `(B, H, T_new, D_head)` | freshly computed from new tokens |
| `K_new` | `(B, H, T_new, D_head)` | freshly computed from new tokens |
| `V_new` | `(B, H, T_new, D_head)` | freshly computed from new tokens |
| `K_cached` | `(B, H, T_cached, D_head)` | from previous steps |
| `V_cached` | `(B, H, T_cached, D_head)` | from previous steps |
| `K_full` | `(B, H, T_cached + T_new, D_head)` | `cat([K_cached, K_new], dim=-2)` |
| `V_full` | `(B, H, T_cached + T_new, D_head)` | `cat([V_cached, V_new], dim=-2)` |

Then standard attention: `Q_new @ K_full.transpose → softmax → @ V_full`.

Save `(K_full, V_full)` for next step.

### Phases

- **Prefill:** `T_cached = 0`, `T_new = P`. After prefill, `T_cached = P`.
- **Decode step:** `T_cached = t`, `T_new = 1`. After step, `T_cached = t + 1`.

### What it buys you

| | naive | basic cache |
|---|---|---|
| per-step cost | O((P+t)²) | O(P+t) |
| total N-token cost | O(N³) | O(N · (P+N)) |
| speedup for P=100, N=1000 | baseline | ~1000× faster |

90% of the win of "KV cache" comes from this one transition.

### Missing features

1. No causal mask needed during decode (only the one new query, only
   past K/V present — no future to mask).
2. Doesn't handle batched generation where sequences finish at different
   times cleanly.
3. **`torch.cat` every step is slow on GPU** — see next subsection.

### Why `torch.cat` every step is slow on GPU

`torch.cat([K_cached, K_new])` doesn't modify `K_cached` in place. It:

1. Allocates a NEW tensor in HBM of size `K_cached.size + K_new.size`.
2. Copies all of `K_cached` into the first portion.
3. Copies all of `K_new` into the second portion.
4. Old `K_cached` gets freed (eventually, via refcount).

Two costs compound:

**Cost A — allocation overhead.** Every `torch.cat` calls PyTorch's
caching allocator. Each call needs to find a free block, possibly
fragment a bigger one, occasionally sync with the CUDA driver
(milliseconds). Across 10 layers, every decode step, this adds up to
0.5-2 ms of pure overhead per step.

**Cost B — copy bandwidth.** Growing from size N to N+1 copies N
elements of existing data. Summed over N decode steps:
`1 + 2 + 3 + ... + N = O(N²)` memory traffic. Ironically reintroduces
the O(N²) cost the cache was supposed to eliminate — just in memory
bandwidth instead of compute.

For a 50M model generating 1000 tokens on MPS/CUDA, this overhead
can eat 30-60% of decode time.

**The fix (later levels):** preallocate the cache up front, never resize.
Insert new K, V into pre-reserved slots. Zero allocation per step.

## Level 3 — Batched inference

Production inference doesn't serve one user at a time — you batch B
requests together so the GPU is fully utilized.

You run `model(tokens_batch)` where `tokens_batch` has shape
`(B, T, ...)`: B sequences processed in parallel. This multiplies your
throughput by up to B times with almost no extra latency.

### The catch — sequences have different lengths

User 1's prompt: 12 tokens.
User 2's prompt: 47 tokens.
User 3's prompt: 3 tokens.

You can't pack three 1D tensors of different lengths into one 2D
batch. Something has to give.

### Fix A — padding + attention mask

Pad all prompts to the longest one (47 tokens in the example). Fill
with a special pad token. Use an attention mask to tell attention
"ignore these padded positions — don't let them affect anyone else's
attention scores."

- Cache shape stays `(B, H, T_cached, D_head)` where `T_cached` = longest sequence.
- Short sequences have wasted cache slots (padding).
- Simple to implement; common in research code.

### Fix B — packed representation

Concatenate all prompts end-to-end into a single 1D tensor:
`[user1_tokens, user2_tokens, user3_tokens]`. Use segment IDs to say
which positions belong to which user. Attention kernels only look
within each segment.

- Cache is one long buffer with segment IDs.
- No wasted slots.
- Requires special kernels that respect segment boundaries (FlashInfer, xformers, vLLM).

Fix A is naive-but-simple. Fix B is production.

## Level 4 — Dynamic-length caches (sequences finish at different times)

Batched generation has a second problem: **not all sequences finish
at the same step**.

User 1 emits `<|endoftext|>` at step 50.
User 2 emits `<|endoftext|>` at step 200.
User 3 runs to max length, step 1000.

What do you do with User 1's cache after step 50?

### The naive answer — keep generating wasted tokens

Leave User 1 in the batch, keep computing their outputs, ignore the
results. Wastes compute, wastes cache memory, and worst of all the
whole batch waits for the slowest sequence.

### The fix — dynamic batching

Maintain a pool of active sequences. Each decode step:

1. Run decode step for all active sequences.
2. Check each for EOS or max length.
3. **Remove finished sequences** — free their cache slots.
4. **Add new sequences** waiting in the queue — give them cache slots.
5. Next step proceeds with the updated batch.

The batch composition changes every step. Some sequences are on decode
step 5, others on step 500.

### What this does to the cache

Padding-based (Fix A from Level 3):
- Cache shape `(B, H, T_cached, D_head)` where `T_cached` = longest
  active sequence.
- Finished sequences → shrink `B` by 1, or replace slot with a new
  prompt.
- **Weakness:** a single long-tail request keeps `T_cached` high for
  the whole batch, wasting cache memory on short sequences.

Packed (Fix B):
- Cache is a flat buffer; each sequence's positions can be anywhere.
- A finished sequence's slots get freed and reused.
- No "longest sequence sets the pace" problem.

### The fragmentation problem this exposes

Even the packed approach has a weakness: after thousands of requests
have joined and left the batch, the cache buffer has holes of all
sizes. Allocating a new sequence's cache needs a contiguous chunk big
enough — possibly impossible even when total free memory is plenty.

Classic memory fragmentation, exactly what virtual memory solved in
operating systems 50 years ago. That's the motivation for Level 6
(paged KV cache).

## Level 5 — Preallocated caches (fix the torch.cat problem)

**Idea:** allocate the full cache once at startup. Never resize. Write
new K, V into pre-reserved slots using in-place indexing.

### The setup

At inference start, pick a `max_context` length (e.g., 2048) and
allocate:

```python
K_cache = torch.zeros(B, H, max_context, D_head, device=..., dtype=...)
V_cache = torch.zeros(B, H, max_context, D_head, device=..., dtype=...)
pos = 0  # how many slots are filled
```

### The per-step update

For each new step with `T_new` new tokens:

```python
K_cache[:, :, pos:pos+T_new] = K_new    # in-place write to reserved slots
V_cache[:, :, pos:pos+T_new] = V_new
pos += T_new

# For attention, slice to just the filled portion:
K_used = K_cache[:, :, :pos]             # a view, no copy
V_used = V_cache[:, :, :pos]
scores = Q_new @ K_used.transpose(-2, -1) / sqrt(D_head)
```

### What this fixes

- No allocation per step (reserved once at startup).
- No copying of existing data (writes go directly to reserved slots).
- Cache management is O(1) per step in memory ops.
- Attention access via slicing is a view, no copy.

Decode is now truly O(T) per step — no hidden O(T) allocation cost.

### The cost — over-reservation

If `max_context = 2048` but your sequence only reaches 100 tokens,
you've reserved 20× more memory than needed. For a single user this
is usually fine. For serving many users with varying needs, this is
wasteful — which motivates Level 6.

For B=32, max_context=2048, 10 layers, 8 heads, D_head=64, bf16:
`32 × 10 × 8 × 2048 × 64 × 2 bytes ≈ 670 MB` reserved up front,
whether used or not.

### What production libraries do here

- **HuggingFace `generate`:** preallocated with user-supplied
  `max_length`. If user asks for 500 tokens, allocate for 500.
- **llama.cpp:** preallocated contiguous cache — simple, targets
  single-user local inference.

## Level 6 — Paged KV cache (vLLM)

The 2023 vLLM paper's main contribution. Made multi-user serving
~24× faster than HuggingFace at the time.

### Problem it solves

Serving many users with unknown future lengths:

- User A wants 10 tokens (you don't know).
- User B wants 2000 tokens (you don't know).
- User C wants 50 tokens (you don't know).

With preallocated caches, you must reserve `max_context` per user.
If `max_context = 4096`:

- User A: 10/4096 → 99.75% waste
- User B: 2000/4096 → 51% waste
- User C: 50/4096 → 98.78% waste

With hundreds of concurrent users, most of your GPU memory sits
unused. You serve far fewer users than your hardware should allow.

### The idea — virtual memory for KV

Borrowed from operating systems. Instead of giving each sequence a
contiguous cache region, split the cache into fixed-size **blocks**
(e.g., 16 tokens per block) and track per-sequence **block tables**
mapping logical positions to physical blocks.

1. Reserve a big GPU memory pool (e.g., 20 GB) at startup.
2. Divide it into small blocks — 16 tokens per block per layer per head.
3. Each sequence has a block table: "positions 0-15 → block 42,
   positions 16-31 → block 7, positions 32-47 → block 183."
4. Sequence grows past current blocks? Allocate a new block from the pool.
5. Sequence finishes? Return its blocks to the pool.

### Why this wins

- **No over-reservation:** use exactly as many blocks as needed.
- **No fragmentation:** all blocks are the same size, so any free
  block fits any need.
- **High utilization:** vLLM typically runs at 90-95% of theoretical
  max batch size; HuggingFace generate runs at 30-50%.

### The cost — custom attention kernel

A sequence's K is now scattered across physical blocks in arbitrary
locations. Naive attention kernels (which assume contiguous K) won't
work.

vLLM wrote **PagedAttention**, a custom CUDA kernel that:
1. Takes Q and a block table as input.
2. For each Q, looks up which physical blocks hold its relevant K's.
3. Runs attention across the non-contiguous access pattern.

Per-step compute is slower than contiguous attention (non-contiguous
memory reads are slower). But the memory savings enable 5-10× bigger
batches. Net win: much higher throughput.

### Which libraries use paged

| library | approach |
|---|---|
| vLLM | paged (the original) |
| TensorRT-LLM | paged + kernel fusion |
| SGLang | paged |
| HuggingFace generate | preallocated per request |
| llama.cpp | contiguous, single-batch |

## Practical recommendation

For a learning project like this repo, **Level 5 (preallocated)** is
the right target.

- Level 2 (basic cat-based cache) teaches the concept but runs slowly.
- Level 5 fixes the allocation problem cleanly, stays pure PyTorch, and
  is 95% of what production single-user inference does.
- Level 6 (paged) requires a custom CUDA kernel. Overkill for a
  learning project, and you can't drop it into existing PyTorch code
  without a lot of extra machinery.

### The path to implement Level 5 in this codebase

1. Add `kv_cache` argument to `Attention.forward` — optional
   `(K_cache, V_cache, pos)` tuple.
2. If present, write new K/V into `K_cache[:, :, pos:pos+T_new]` and
   slice to `[:pos+T_new]` for attention.
3. Return new `pos` (updated).
4. Propagate cache through all layers in `Model.forward`.
5. Write a `generate()` method at the top level that:
   - Allocates L `(K_cache, V_cache)` tensors once.
   - Runs prefill (big T_new).
   - Loops decode (T_new=1) until EOS or max length.

Then compare decode speed against the naive re-run-forward approach —
you should see 5-10× speedup on typical generation lengths.

## Verifying correctness — a must-do

A KV cache implementation can look like it works (reasonable speedup,
plausible-looking output) while silently computing wrong math. The
bug signature: "model generates text, but it's different from what
the model without the cache would have generated."

The correctness test is simple — but you need to set it up right.

### The protocol

1. **Use argmax sampling, not temperature.** Argmax is deterministic.
   If two code paths compute the same logits, they produce the same
   token stream. Any divergence proves a bug.
2. **Run two generation paths on the same prompt:**
   - Naive path (no cache): re-run forward on the growing sequence
     each step.
   - Cache path: prefill once, decode one token at a time using cache.
3. **Compare the generated token IDs token-by-token.** If they differ
   at any step, the cache is wrong. Report the first divergence.

```python
tokens_no_cache = generate_no_cache(...)
tokens_with_cache = generate_with_cache(...)

assert torch.equal(tokens_no_cache, tokens_with_cache)
```

### Why argmax, not temperature

Temperature + multinomial sampling hides bugs. Two paths with
slightly different logits will produce different tokens via RNG and
you'd just say "random variation." Argmax flips the output on any
real logit difference, exposing the bug.

After the cache is verified correct, switch back to temperature
sampling for actual generation.

### Common bugs this test catches

- **Off-by-one in cache indexing.** Writing to slot `pos+1` instead
  of `pos` (or vice versa). Cache gets corrupted, logits wrong.
- **RoPE applied with wrong position.** Passing `pos=0` during
  prefill rotates all tokens by the same zero angle, instead of
  per-position angles. Produces wrong K and Q.
- **Broadcasting bug on cache write.** `K_cache[:, :, :pos+1, :] =
  K_new` where `K_new` has T=1 will broadcast and overwrite all
  slots 0..pos with the new token's K, corrupting history.
- **Cache not being read back correctly.** Attention math uses the
  just-computed K, not the cached K, which misses earlier positions.
- **Causal mask applied wrong during decode.** The mask is for prefill/
  training (when Q and K have the same length). During decode, Q has
  length 1 and K has length pos+1 — the mask shape doesn't match.

### What a correct output looks like

With argmax sampling, both paths should produce **byte-identical**
token streams. Not "close" — identical. The cache optimization doesn't
change the math, so there's no room for numerical precision to matter.

If outputs match exactly, your cache is correct. Any mismatch, no
matter how small, is a bug.

### Diagnosing a mismatch

When two paths diverge at step N:

1. Print the top-1 logit at step N-1 from each path. If they differ,
   the divergence started earlier. Back up.
2. Print the K and V tensors at each layer going into attention. Find
   the first layer / position where they differ. That's your bug.
3. Print the output of attention itself. If inputs match but output
   differs, the bug is in attention math — probably the mask or the
   score shape.

### What the benchmark should also measure

Beyond "match / no-match":

- **Per-step time.** With cache, should be roughly constant across
  decode steps. Without cache, should grow linearly.
- **Speedup.** For a trained model on reasonable lengths, expect 5-10×.
  Smaller speedups on tiny models (overhead per step dominates).
- **Prefill time vs decode time.** Both are useful separately — prefill
  dominates for short generations from long prompts, decode dominates
  for long generations from short prompts.

### Summary table

| level | approach | decode speed | memory use | complexity |
|---|---|---|---|---|
| 1 | no cache | O(N²) — bad | zero cache | trivial |
| 2 | per-layer cat | O(N²) memory | grows each step | easy |
| 3 | batched (padded/packed) | O(N) | varies | medium |
| 4 | + dynamic batching | O(N) | efficient | medium |
| 5 | preallocated | O(N) | over-reserves | easy-medium |
| 6 | paged | O(N) | optimal | needs custom kernel |

## Appendix — roadmap to implementing Level 6 yourself

Not just "write a CUDA kernel" — a whole system. Five layers of work:

1. **GPU kernel programming.** The paged attention kernel itself,
   usually in CUDA C++ or Triton. Need: thread blocks, warps, shared
   memory, tensor cores, memory coalescing, occupancy. vLLM's
   PagedAttention is ~500 lines of CUDA plus heavy tuning.
2. **Custom ops plumbing.** Register the kernel as a PyTorch op so
   you can call it from Python. `torch.library`, TORCH_LIBRARY macros,
   or newer custom-op APIs.
3. **Block table management.** Python / C++ code maintaining
   per-sequence block tables, free-block pools, allocation, and
   (optionally) reference counting for prefix sharing. Small memory
   allocator.
4. **Model integration.** Thread the block table through every
   attention call. Rest of the model (MLP, norms) is unchanged.
5. **Testing and tuning.** Correctness (paged output matches
   contiguous on same inputs) and performance (block size, launch
   params, tail-block edge cases).

### Stepped learning path

1. Write a **contiguous Triton attention** kernel that matches
   PyTorch's baseline. Teaches Triton, tile-based thinking. ~1-2 weeks.
2. Read **FlashAttention / FlashAttention-2** papers + reference code.
   ~3-5 days.
3. Implement **FlashAttention-2 forward in Triton** (online softmax +
   tiling). Classic learning project.
4. Build **block-table data structures** in Python (CPU-side allocator).
5. Write **PagedAttention in Triton** — FlashAttention tiling with
   block-indirection for K, V reads.
6. **Wire it up and benchmark** end-to-end.

Total focused effort: 2-3 months from scratch. Excellent portfolio
project; requires sustained attention but builds skills that transfer
everywhere in systems ML.
