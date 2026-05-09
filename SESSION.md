# Session checkpoint — 2026-05-06 evening (Phase 3 kickoff — GPU foundations)

Load with: `please load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md`

## State

- Repo: `/Users/katayal/Documents/llm/AllAboutLLMs`, branch `main`,
  remote `github.com/tkashish/all-about-llms`. **All Phase 2 commits
  are on origin.**
- Phase 2 (paged cache + 3-way correctness test) complete and
  committed, see prior SESSION history if needed.
- Phase 3 kickoff: GPU provisioned, environment verified, and a full
  GPU mental-model session done. No kernel code yet. Paused right
  before "memory coalescing" — the first efficient-kernel principle.

## User

- Kashish Tayal, self-studying LLM internals.
- **Teaching style is mandatory**:
  `/Users/katayal/Documents/llm/cs336/TEACHING-STYLE.md`. Plain
  English → problem → mechanics, one idea per message, ~15 lines max
  before check-in, concrete numbers over abstract math. User writes
  the code; Kiro explains / reviews / debugs.

## What got done 2026-05-06

Phase 3 kickoff — GPU foundations and execution-model mental model.
Zero kernel code written yet. Today was about building intuition
*before* any Triton syntax, so efficiency reasoning becomes natural.

### Infra provisioned (AWS)

- **Instance:** `g6.xlarge` (1× NVIDIA L4 24GB, ~$0.80/hr).
  Revised down from g5.2xlarge (A10G) — L4 has lower memory
  bandwidth but same VRAM, same Triton support, 33% cheaper.
- **Instance ID:** `i-0c8cf119364f6acec` (us-east-1).
- **AMI:** Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10
  (Amazon Linux 2023), x86 — `ami-0b246ca76fc968679`.
- **Storage:** 100 GiB gp3 root.
- **IAM role:** `InstanceRole` (with `AmazonSSMManagedInstanceCore`
  + `CloudWatchAgentServerPolicy`).
- **No key pair; no inbound SG rules.** Access via SSM only.

### Connect + environment verified

```bash
# From Mac
aws ssm start-session --target i-0c8cf119364f6acec --region us-east-1

# Inside the instance
sudo su - ec2-user
source activate pytorch               # activates the DL AMI env
nvidia-smi                            # shows 1× L4 24GB

# Project setup
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~
git clone https://github.com/tkashish/all-about-llms.git
cd all-about-llms
uv sync

# Verify GPU access through uv env
uv run python -c "import torch, triton; print(torch.cuda.is_available(), triton.__version__)"
# → True 3.6.0
```

### Conceptual material covered

Walked through two separate mental models, each with its own
hierarchy:

**Model A — Hardware (physical, static):**
- GPU has many SMs (60 on L4, 132 on H100).
- Each SM has 128 CUDA cores + 4 Tensor cores + registers (~64 KB)
  + SRAM (~128 KB).
- Memory hierarchy: HBM (24 GB, ~300 GB/s on L4) → L2 cache (~48 MB,
  hardware-managed) → SRAM (per-SM, manually-managed) → registers.
- CUDA cores = general math (softmax, elementwise). Tensor cores =
  matmul specialists (fp16/bf16/int8/fp8).
- "CUDA" is overloaded: the *platform* (language + runtime) vs
  *CUDA cores* (hardware units inside an SM).

**Model B — Execution (logical, per-kernel):**
- Kernel launch: grid of blocks, block of threads. Programmer picks
  grid size and block size; everything else is automatic.
- Three levels of scheduling, all hardware:
  1. Blocks → SMs (global scheduler, dynamic).
  2. Threads → warps of 32 (automatic).
  3. Warps → cores per cycle (SM's 4 warp schedulers).
- One block stays on its SM for its lifetime (threads share SRAM,
  can't migrate).
- A warp's 32 threads use 32 CUDA cores simultaneously. 4 warps ×
  32 = 128 cores busy per SM per cycle.
- Scheduler hides memory latency by swapping stalled warps for
  active ones — so you want many resident warps per SM.

### Wiki changes (uncommitted on Mac)

- `wiki/gpu/02-hardware-anatomy.md` — added:
  - **"Why GPUs exist (for ML)"** — CPU vs GPU framing.
  - **'"CUDA" is overloaded — SM vs CUDA cores'** — naming
    disambiguation + hierarchy diagram + L4/H100 core counts.
  - **"Two kinds of cores inside an SM"** — CUDA vs Tensor cores,
    with pizza-oven analogy.
  - **"L4 at a glance"** — consolidated hardware picture specific
    to the instance we're using.
- `wiki/gpu/06-execution-model.md` — **NEW**. Full execution model
  top-to-bottom with grid/block/warp/thread diagram, three
  scheduling levels table, what-the-programmer-controls table,
  and two common underutilization patterns.
- `wiki/gpu/vllm-learning-plan.md` — updated Phase 3 objectives:
  - New Obj 1: GPU mental model.
  - New Obj 2: Efficient-kernel principles (coalescing,
    occupancy, tiling, divergence, register pressure).
  - Reframed existing Triton/kernel/benchmark/scheduler objectives.
  - Added `wiki/gpu/07-efficient-kernels.md` to deliverables.

## Where we paused

End-of-GPU-foundations, right before starting efficient-kernel
principles. User asked for the efficient-kernels arc in this order:
coalescing > occupancy > (tiling later, with the paged kernel).

**Next concept to cover on resume: memory coalescing.** Why threads
in a warp should access contiguous memory, what happens when they
don't (multiple memory transactions instead of one), and how to
structure thread-to-data mapping to coalesce by default. Belongs in
the planned `wiki/gpu/07-efficient-kernels.md` note (not yet
created).

After that: **occupancy + latency hiding**, then **vector-add
kernel in Triton** to make the mental model concrete. Then start
on the paged attention kernel.

## Day-2 checklist (tomorrow)

1. Reconnect: `aws ssm start-session --target i-0c8cf119364f6acec
   --region us-east-1`.
2. If instance was stopped, start it first: `aws ec2
   start-instances --instance-ids i-0c8cf119364f6acec`.
3. `source activate pytorch && cd ~/all-about-llms && git pull`.
4. Resume with "memory coalescing" concept.

## Running-cost note

The instance costs **$0.80/hr on-demand**. If left running, that's
~$19/day. Stop when not in use:

```bash
aws ec2 stop-instances --instance-ids i-0c8cf119364f6acec --region us-east-1
```

EBS persists; uv env + cloned repo survive a stop/start. Only the
ephemeral instance-store (if used) would be wiped.

## Dangling threads (carried over)

1. **Uncommitted wiki changes** on Mac — 3 files modified / 1 new.
   Not committed tonight; review and commit tomorrow before doing
   more wiki work.
2. **Chunked prefill** — noted in continuous-batching doc.
3. **`Sequence.release_blocks()`** not wired into request lifecycle.
4. **Warmup pollution** handling in `benchmark_inference.py` is ad
   hoc for the paged path; could be generalized once a common cache
   interface exists.
5. **Excalidraw line-height quirk** — from 2026-05-01 session.
6. **Earlier dangling threads** (RoPE impl notes, pytorch wiki
   holes, SwiGLU walkthrough, train a bigger model) — still open.
7. **Single-sequence benchmark understates paged's benefits.** The
   multi-sequence comparison naturally happens as part of Phase 3
   Objective 7 (aggregate throughput).

## Phase 4+ wishlist (reminder, scope after Phase 3 ships)

- **FlashAttention** — Triton kernel for prefill, natural extension
  of Phase 3. Recommended first Phase 4 topic.
- Prefix caching / CoW, speculative decoding, OpenAI HTTP server,
  quantization, multi-GPU tensor parallelism, more model
  architectures, production details.

## Resume prompt

> "Load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md. The
>  g6.xlarge should be stopped — start it, reconnect via SSM, and
>  pick up where we left off: memory coalescing (first efficient-
>  kernel principle)."
