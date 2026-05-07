# Session checkpoint — 2026-05-04 evening (Phase 2 complete, Phase 3 queued)

Load with: `please load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md`

## State

- Repo: `/Users/katayal/Documents/llm/AllAboutLLMs`, branch `main`,
  remote `github.com/tkashish/all-about-llms`.
- **Three unpushed commits** on local main (Phase 1 wiki, Phase 2
  code, Phase 2 benchmark results). `git -P log --oneline -3` to
  see them; push when ready.
- Phase 2 of the vLLM learning plan: **done**. Paged cache produces
  byte-identical outputs to both the no-cache reference and the old
  KVCache implementation. 3-way correctness test passes.
- 7 wiki notes under `wiki/gpu/vllm/` covering motivation through
  Phase 2 benchmark results.
- Phase 3 kickoff is blocked on GPU provisioning (see below).

## User

- Kashish Tayal, self-studying LLM internals.
- **Teaching style is mandatory**:
  `/Users/katayal/Documents/llm/cs336/TEACHING-STYLE.md`. Plain
  English → problem → mechanics, one idea per message, ~15 lines max
  before check-in, concrete numbers over abstract math. User writes
  the code; Kiro explains / reviews / debugs.

## What got done 2026-05-04

**Phase 2 — replaced the Level 5 KV cache with a paged cache in
pure PyTorch.** Kept both implementations alive behind a `CacheType`
enum for A/B testing. Proved 3-way byte-identical correctness
(no-cache == KVCache == paged). Measured ~10% median overhead of
paged vs KVCache on a 60-token workload; overhead is isolated to
the Python loop + `torch.cat` in `Sequence.get()` — exactly what
Phase 3's Triton kernel will replace.

**Files created/modified:**

- `src/transformer/paged_cache.py` — `Block`, `BlockPool`,
  `Sequence`. Per-layer token counters (`token_per_layer` dict) so
  each layer's `get()` returns exactly the slots that layer has
  appended, no cross-layer ordering dependency.
- `src/transformer/attention.py` — added `CacheType` enum (KV or
  PAGED). Attention branches on `cache_type` and either talks to
  `self.kv_cache` or the `seq: Sequence` arg. Squeeze/unsqueeze at
  the Sequence boundary to bridge `(B, H, T, D)` ↔ `(H, T, D)`.
- `src/transformer/model.py` — `HyperParams.cache_type` with
  `PAGED` default. `Model.__init__` builds a KVCache only when
  `cache_type == KV`.
- `src/transformer/transformer.py`, `inference.py`,
  `benchmark_inference.py` — thread `seq: Sequence | None = None`
  through `forward()`; 3-way benchmark in main().
- `wiki/gpu/vllm/06-paged-cache-impl.md` — Phase 2 implementation
  notes: design, the subtle off-by-one bug and why per-layer
  counters fix it, 3-way correctness methodology, measured
  overhead, gotchas, Phase 3 unlocks.
- `wiki/gpu/vllm/07-benchmark-results.md` — benchmark numbers,
  variance table, and takeaways.
- `wiki/gpu/vllm/05-sharing-and-cow.md` — updated closing to link
  forward to 06 instead of the learning plan.

## Key findings locked in

**The off-by-one bug.** Naive impl tried to update a single
`tokens_used` counter on the last layer's write. But each decode
step processes layers in order: each layer writes then reads. Layer
0's read happens **before** layer N-1's write, so layer 0 sees
stale `tokens_used` from the previous step. Symptom: paged step N
outputs match no-cache step N-1. Fix: per-layer counter dict.

**Overhead measurement.** Paged ~10% slower than KVCache on a
60-token workload. Overhead is the Python loop + `torch.cat` in
`Sequence.get()`. Will grow with longer sequences; exactly the
workload the Phase 3 Triton kernel will target.

**Shape boundary.** Attention: `(B, H, T, D)`. Sequence:
`(H, T, D)` — a Sequence represents one sequence, no batch dim.
Squeeze/unsqueeze at the three call sites in `Attention.forward()`
bridge the two.

## Phase 3 kickoff — prep done, provisioning pending

**Instance plan (picked tonight, not launched):**

- **Type:** `g5.2xlarge` (1× A10G 24GB, ~$1.20/hr). Revised down
  from g5.12xlarge — only 1 GPU needed for Phase 3, and g5.12xlarge
  hit "Insufficient capacity" in the target AZ anyway.
- **AMI:** Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10
  (Amazon Linux 2023), x86 — `ami-01a5fd1331e9628eb`. Ships with
  PyTorch 2.10, CUDA/cuDNN, NVIDIA drivers, Triton, SSM agent.
- **Storage:** 100 GiB gp3 encrypted root. Optionally mount the
  ~3.8 TB ephemeral NVMe instance store at `/mnt/nvme` for scratch.
- **Key pair:** none. Connect via SSM Session Manager.
- **IAM role:** `InstanceRole` (created tonight) with
  `AmazonSSMManagedInstanceCore` + `CloudWatchAgentServerPolicy`.
- **Security group:** no inbound rules needed (SSM only).
- **Shutdown behavior:** Stop (not Terminate).

**Provisioning decision:** an Amazon internal security warning
flagged creating an AWS-managed (non-Midway-signed) SSH key pair
as a HIGH risk. We're using SSM Session Manager instead — no SSH
keys at all.

**Launch attempt today failed** with "Insufficient capacity" on
g5.12xlarge. Plan for tomorrow: use g5.2xlarge; if that also fails,
try a different AZ or the `us-west-2` / `us-east-2` regions.

## Day-1 Phase 3 checklist (tomorrow morning)

1. Launch the instance per the plan above.
2. Connect: `aws ssm start-session --target i-<instance-id>`.
3. Sanity-check: `sudo su - ec2-user`, then
   `nvidia-smi` (should show 1× A10G), then
   `python -c "import torch, triton; print(torch.cuda.is_available(), triton.__version__)"`.
4. Clone the repo: `git clone https://github.com/tkashish/all-about-llms.git`, `cd all-about-llms`, `uv sync`.
5. Begin Phase 3 Objective 1 — trivial Triton vector-add kernel to
   get feel for `@triton.jit`, `tl.load`, `tl.store`, `pid`, masks.
6. Rebuild the 3-way benchmark on the new hardware as baseline
   before touching the kernel.

## Phase 3 roadmap (from `wiki/gpu/vllm-learning-plan.md`)

**Half 1 — Triton kernel (make it fast):**

1. Learn Triton basics with a vector-add kernel.
2. Write a PagedAttention decode kernel reading K/V straight from
   `BlockPool.blocks` using the block table as an index. Replaces
   `Sequence.get()` entirely.
3. Benchmark Triton paged vs pure-Py paged vs KVCache. Expect
   Triton ≈ KV.

**Half 2 — Scheduler (make it useful):**

4. Minimal request queue with prefill + decode phases, continuous
   batching across multiple concurrent sequences.
5. Simulate many concurrent requests with varying lengths. Measure
   aggregate throughput, watch it climb toward the compute-ridge
   ceiling.

**Deliverables:**

- `src/inference/paged_attention_kernel.py`
- `src/inference/scheduler.py`
- `src/inference/server.py` (stdin-driven request driver)
- `wiki/gpu/vllm/08-triton-kernel.md`
- `wiki/gpu/vllm/09-scheduler.md`
- `wiki/gpu/vllm/10-final-benchmarks.md`

## Phase 4+ wishlist (from user, not yet scoped)

User wants to eventually cover:

- **FlashAttention** — Triton kernel for prefill that avoids
  materializing the T×T attention matrix (tiling + online softmax
  + stay-in-SRAM). Natural extension of Phase 3's Triton work —
  the primitives are identical. Recommended first Phase 4 topic.
- **Prefix caching / CoW** — extends Phase 2 code; small / high
  learning density. Good second.
- **Speculative decoding** — clever algorithmic idea; builds on the
  base model, doesn't need new serving infra.
- **OpenAI-compatible HTTP server** — wrap scheduler in FastAPI with
  OpenAI chat/completions schema. Mostly engineering.
- **Quantization (fp8, int4, AWQ, GPTQ)** — huge topic on its own,
  deserves its own multi-phase plan.
- **Multi-GPU tensor parallelism** — NCCL, collectives, comms
  patterns. Requires multi-GPU instance.
- **More model architectures** — mostly boilerplate.
- **Production details** — catch-all; best learned by reading real
  vLLM source with reimplementation as context.

Formalize a Phase 4 plan only once Phase 3 ships.

## Dangling threads

1. **Three unpushed commits.** Local main is ahead of origin/main
   by 3 commits. Push when comfortable.
2. **Chunked prefill** — noted in continuous-batching doc, not yet
   explored.
3. **`Sequence.release_blocks()`** exists but is not hooked into
   request lifecycle anywhere. Will matter once a scheduler is
   preempting sequences.
4. **Warmup pollution** — `benchmark_inference.py` handles this for
   paged by rebuilding `Sequence` between warmup and real runs.
   The KVCache path handles it by rebuilding the whole model,
   which is wasteful; could add a reset method.
5. **`generate_with_kv_cache` and `generate_with_paged_cache`** are
   nearly identical. Good refactoring candidate once a common cache
   interface exists.
6. **Excalidraw line-height quirk** — from 2026-05-01 session,
   still open, not blocking.
7. **Earlier dangling threads** (RoPE impl notes, pytorch wiki
   holes, SwiGLU walkthrough, train a bigger model) — still open.
8. **Single-sequence benchmark understates paged's benefits.** A
   multi-sequence memory-utilization comparison was discussed but
   deferred — the payoff shows up in Phase 3 scheduler work anyway.

## Resume prompt

> "Load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md.
>  The g5.2xlarge instance should be up — help me through Phase 3
>  Day 1 (Triton vector-add to warm up, then start on the paged
>  attention kernel)."
