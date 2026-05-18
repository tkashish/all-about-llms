# Session checkpoint — 2026-05-18 (Phase 3 GPU foundations deep-dive complete)

Load with: `please load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md`

## State

- Repo: `/Users/katayal/Documents/llm/AllAboutLLMs`, branch `main`,
  remote `github.com/tkashish/all-about-llms`. Phase 2 commits are
  pushed; **Phase 3 wiki/diagram work is uncommitted**.
- Phase 3 status: Obj 1 (GPU mental model) **deeply explored** with
  many follow-up clarifications and diagrams. Plan revised to 15
  objectives. **No kernel code written yet** — about to start the
  vector_add kernel as the entry to Triton.

## User

- Kashish Tayal, self-studying LLM internals.
- **Teaching style is mandatory**:
  `/Users/katayal/Documents/llm/cs336/TEACHING-STYLE.md`. Plain
  English → problem → mechanics, one idea per message, ~15 lines
  max before check-in, concrete numbers over abstract math. User
  writes the code; Kiro explains / reviews / debugs.
- User explicitly asked: "as we go through phase 3, please show
  me how to write efficient kernels. Cover internals so efficiency
  reasoning becomes intuitive."

## Infra

- **Instance:** `i-0c8cf119364f6acec` in us-east-1, type
  **`g5.xlarge`** (1× NVIDIA A10G 24GB, ~$1.00/hr). Switched from
  g6.xlarge after AWS hit "InsufficientInstanceCapacity"; instance
  type was modified while stopped. Not g6 / L4.
- **AMI:** Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10
  (Amazon Linux 2023). PyTorch 2.x + CUDA + Triton 3.6.0
  pre-installed.
- **IAM role:** `InstanceRole` with `AmazonSSMManagedInstanceCore`
  + `CloudWatchAgentServerPolicy`.
- **Connect:** `aws ssm start-session --target i-0c8cf119364f6acec
  --region us-east-1`. Activate venv with
  `source /opt/pytorch/bin/activate` (NOT `source activate pytorch`
  — no conda).

## Phase 3 plan revision

After my draft and the user's feedback (a separate detailed plan
they fed in), the merged Phase 3 plan now has **15 objectives**
in `wiki/gpu/vllm-learning-plan.md`:

1. GPU mental model (hardware + execution).
2. Triton basics — `vector_add`, `copy`, `scale`, `row_sum`,
   `softmax` warm-ups.
3. Efficient-kernel principles (coalescing, occupancy, tiling,
   divergence, register pressure) with hands-on **"intentionally
   bad kernels"** exercise.
4. Correctness + profiling harness — built **before** any
   attention kernel. Tools: `torch.cuda.Event`, `torch.profiler`,
   `nsys`, `ncu`.
5. Contiguous KV decode attention (Triton baseline).
6. **Online softmax** (its own dedicated stage with explicit
   learning questions: why keep m and l? why rescale?).
7. KV block manager (Phase 2 code becomes a runtime component).
8. Paged KV cache layout (deliberate choice).
9. Paged decode attention kernel (staged 9A–9F:
   single-req → multi-head → multi-req → variable-length →
   GQA/MQA → optimize layout/tiling).
10. 3-way kernel benchmark (PyTorch KV / contiguous Triton /
    paged Triton).
11. Simple continuous-batching scheduler.
12. Aggregate throughput benchmark with rich metrics: TTFT, ITL,
    p50/p95/p99, GPU mem, KV waste, throughput vs concurrency
    and context length.
13. Chunked prefill.
14. Unified token-budget scheduler (vLLM V1 style).
15. Optional Phase 3.5 (prefix caching, CUDA graphs, spec decode prep).

**Goal soft-framed:** show a clear throughput gap between naive
single-request and continuous-batched paged serving, **explain
the gap with profiler evidence.** Not a hard "close 2–4× gap"
target.

## What got covered 2026-05-08 (and adjacent days)

Deep dive on the GPU mental model, hardware-level. No code yet.
Sections cover what was learned in conversation **plus** the
clarifications when the user pushed back on hand-waving.

### Hardware mental model

- **GPU = many SMs + shared HBM + shared L2.** A10G has 80 SMs
  (Ampere, GA10x). L4 has 60 SMs (Ada). Same per-SM structure on
  both: 128 CUDA cores + 4 Tensor cores + 4 warp schedulers + 64 KB
  registers + 128 KB SRAM.
- **Memory hierarchy:** HBM (~600 GB/s, slow) → L2 (~48 MB,
  hardware-managed cache) → SRAM (per-SM, manually-managed by
  kernel code) → registers (per-thread, effectively free).
- **Variation table** in wiki for Volta/Turing/Ampere/Ada/Hopper.

### Execution units inside an SM (key clarification)

CUDA cores are **not** the only math hardware. An SM contains
several **separate physical circuits** that coexist:

| Unit | Count per SM | Op | Throughput |
|---|---|---|---|
| CUDA cores | 128 | ADD, MUL, FMA, int | 128/cycle |
| Tensor cores | 4 | matmul tiles | ~256 MAC/cycle |
| DIV/SQRT | ~4 | division, sqrt | 4/cycle |
| SFU | ~4 | exp, log, sin, cos, rcp, rsqrt | 4/cycle |
| Load/Store | ~4 | memory ops | 32 req/cycle issue |

CUDA cores don't "become" DIV units. They're **separate silicon**.
"Warp → 32 threads → 32 cores → 1 cycle" only holds for ops that
hit a unit with ≥32 lanes (ADD/MUL/FMA). For DIV/EXP, the warp's
32 threads serialize through 4 lanes → ~8 cycles per warp, with
all 128 CUDA cores idle during that time.

### Execution model (Model B)

- Programmer's two knobs: **`grid_size` and `num_warps`** (and
  `BLOCK_SIZE` in Triton, which is vector width per program, not
  thread count).
- Three levels of scheduling, all hardware:
  1. Blocks → SMs (global scheduler, dynamic, ~16 blocks resident
     per SM).
  2. Block's threads → warps of 32 (automatic).
  3. Per cycle: each scheduler picks one **ready** warp from its
     pool of up to 12 warps and issues one instruction.
- One block stays on its SM for its lifetime.

### Warp scheduling and occupancy (the hard part)

- **Resident vs ready.** Resident = loaded with registers/SRAM
  allocated. Ready = next instruction's dependencies are
  satisfied (no outstanding HBM load it depends on, no pending
  result it needs).
- **Pipelining.** CUDA cores have a ~4-stage pipeline. A single
  instruction has ~4-cycle latency but throughput is 1/cycle —
  the core accepts a new instruction every cycle, with up to 4
  instructions in flight at different stages. Result available
  4 cycles after issue.
- **Latency hiding.** When a warp stalls on a ~700-cycle HBM
  load, the scheduler picks a different ready warp. Cores stay
  busy as long as **some** warp in the scheduler's pool is ready.
- **Occupancy = resident warps / max (48 on A10G).** Low occupancy
  = scheduler runs out of warps to swap when one stalls = idle
  cores = memory-bound underutilization. The single most common
  performance bug.

### Memory coalescing — cache lines

- **Cache line = 128 bytes.** Smallest unit of HBM transfer.
- **Coalesced** = a warp's 32 threads request data that all fits
  in one cache line → **1 transaction, 100% efficient**.
- **Strided** = stride 2 spans 2 lines → 50% efficient (half the
  bytes wasted).
- **Scattered** = each thread in different line → up to 32
  transactions, ~3% efficient. ~30× bandwidth penalty.
- **Rule:** threads `0..31` of a warp should touch contiguous
  slots. `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`
  is coalesced by construction.
- **Strides matter for non-contiguous tensors.** Triton kernels
  for 2D+ ops should take stride args explicitly, not assume
  row-major contiguous.

### Kernel sizing math

A10G fixed limits:
- Warp = 32 threads (always).
- Max threads/block = 1024.
- Max blocks/SM = 16.
- Max warps/SM = 48 (= 1536 threads/SM).
- 65,536 registers + 128 KB SRAM per SM.

Occupancy table:

| `num_warps` | threads/block | blocks/SM | warps/SM | Occupancy |
|---|---|---|---|---|
| 1 | 32 | 16 (block cap) | 16 | 33% |
| 4 | 128 | 12 | 48 | **100%** |
| 8 | 256 | 6 | 48 | **100%** |
| 16 | 512 | 3 | 48 | **100%** |
| 32 | 1024 | 1 | 32 | 67% |

Sweet spot: `num_warps=4..16`. Triton default is 4. Concrete
recipe for a memory-bound kernel:
```python
BLOCK_SIZE = 1024     # vector width
num_warps = 4         # 128 threads/block; each handles 8 elems
grid = (triton.cdiv(N, BLOCK_SIZE),)
# For N=1M: ~977 blocks ≈ 12 per SM = full occupancy.
```

### vector_add walkthrough (no code yet, just conceptual)

End-to-end trace for `vector_add[(4,)](x, y, out, N=1000,
BLOCK_SIZE=256)` covered. Key insights:

- 4 blocks → 4 SMs out of 80 → kernel under-utilizes the GPU.
- Per-block: 8 warps split across 4 schedulers → 2 warps each.
- Per warp instruction stream: pid → MUL → ADD → CMP → LOAD x
  (stall ~700 cyc) → LOAD y (stall) → ADD → STORE.
- With 2 warps per scheduler, both stall on LOAD → **scheduler
  IDLE** for hundreds of cycles. Classic low-occupancy memory-
  bound pattern.
- **CUDA cores ~1% utilized; HBM bandwidth saturated.** Real
  fix is more blocks / smaller blocks / larger N, not bigger
  blocks (which hits hardware limits).

## Wiki / diagram changes (uncommitted)

All under `wiki/gpu/`:

- **`02-hardware-anatomy.md`** — added sections for "CUDA is
  overloaded", "Two kinds of cores", "Other execution units"
  (DIV/SFU breakdown), "Variation across GPU architectures".
- **`06-execution-model.md`** (new earlier) — grid → block → warp
  → thread mapping with 3 scheduling levels.
- **`07-efficient-kernels.md`** — filled in coalescing
  (with cache lines + efficiency table), occupancy, kernel sizing
  math.
- **`08-triton-basics.md`** (new) — Triton mental shift,
  vector_add code annotated. Mostly stub for further filling.
- **`_generate_diagrams.py`** — generator script (~700 lines).
- **`gpu-fundamentals.excalidraw`** — regenerated with **7 scenes**:
  hardware hierarchy, execution model, SM structure
  (schedulers × warps × cores), coalescing, warp scheduling
  timeline, vector_add caller+kernel, vector_add full trace.
  ~1100 elements.
- **`vllm-learning-plan.md`** — Phase 3 fully revised to 15
  objectives.

## Where we paused

End of conceptual GPU mental model. Last topic discussed was
pipeline depth (~4 stages for ADD/MUL/FMA, fixed in hardware).

**Next step:** actually write `vector_add` in Triton on the
instance, run it, verify against `torch.add`. This is Phase 3
Obj 2 in the revised plan.

## Day-N checklist (when resuming)

1. Verify instance: `aws ec2 describe-instances --instance-ids
   i-0c8cf119364f6acec --region us-east-1 --query
   'Reservations[].Instances[].State.Name' --output text`. If
   `stopped`, start it. If `terminated`, re-provision.
2. Connect via SSM, activate `/opt/pytorch/bin/activate`.
3. `cd ~/all-about-llms && git pull` to grab any uncommitted
   wiki content (after committing locally first).
4. Resume with **vector_add kernel implementation**:
   - Create `src/inference/01_triton_basics/vector_add.py`
     (per the suggested repo structure in `vllm-learning-plan.md`).
   - User writes the kernel; Kiro reviews and helps debug.
   - Verify against `torch.add` on a 1M-element vector.
   - Profile with `torch.cuda.Event` for first timing.

## Dangling threads (carried over)

1. **Uncommitted wiki/diagram changes on Mac** — substantial work
   from this week. Should be committed before resuming.
2. **Excalidraw load failed** in user's last attempt with the
   same error. Existing `vllm-phase1.excalidraw` (Phase 1) loaded
   fine. Might be excalidraw-version compatibility, browser cache,
   or a subtle invalid element. Worth diagnosing with a minimal
   test file.
3. **Chunked prefill** — covered in plan, not implemented.
4. **`Sequence.release_blocks()`** unhooked from request lifecycle.
5. **Warmup pollution handling** in benchmark could be generalized.
6. **Earlier dangling threads** (RoPE impl notes, pytorch wiki
   holes, SwiGLU walkthrough, train a bigger model, blog title
   undecided) — still open.

## Phase 4+ wishlist (reminder, scope after Phase 3 ships)

- **FlashAttention** — Triton kernel for prefill, recommended
  first Phase 4 topic.
- Prefix caching / CoW.
- Speculative decoding.
- OpenAI-compatible HTTP server.
- Quantization (fp8, int4, AWQ, GPTQ).
- Multi-GPU tensor parallelism.
- More model architectures.
- Production details.

## Resume prompt

> "Load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md.
>  GPU mental model is done; the wiki + excalidraw is up to date.
>  Time to actually write the vector_add Triton kernel — Phase 3
>  Obj 2."
