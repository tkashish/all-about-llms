# Session checkpoint — 2026-04-29 afternoon

Load with: `please load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md`

## State

- Repo: `/Users/katayal/Documents/llm/AllAboutLLMs`, branch `main`,
  remote `github.com/tkashish/all-about-llms`. Wiki updates from today
  are uncommitted.
- Transformer build: complete. KV cache Level 5 implemented and verified
  (~2.4× speedup on smoke-test model). Smoke-test weights at
  `data/model/model.pt`.
- No code changes today — pure study session on GPU internals.

## User

- Kashish Tayal, self-studying LLM internals.
- Teaching style is captured in `/Users/katayal/Documents/llm/cs336/TEACHING-STYLE.md`
  and MUST be followed: plain English → problem → mechanics, one idea
  per message, ~15 lines max before check-in, concrete numbers over
  abstract math, save notes after concepts land. User writes model code
  himself; Kiro explains / reviews / debugs.

## What got done today

Built out a new wiki section `wiki/gpu/` covering GPU fundamentals
through the lens of "is this compute-bound or memory-bound?"

1. **`01-compute-vs-memory.md`** — the core tension. H100 does math
   ~600× faster than it moves bytes across HBM, so every engineering
   decision is a move in one of those two fights.
2. **`02-hardware-anatomy.md`** — SMs (132 on H100), memory hierarchy
   (HBM / L2 shared, SRAM / registers per-SM), consequence for code
   (maximize reuse in SRAM, minimize HBM round-trips).
3. **`03-arithmetic-intensity.md`** — FLOPs/byte, the ridge point
   (~330 on H100 bf16), roofline picture, worked example for square
   matmul (N/3 intensity).
4. **`04-why-decode-is-memory-bound.md`** — conceptual summary showing
   decode at batch=1 has intensity 1, 330× below the ridge.
5. **`05-llama-7b-decode-walkthrough.md`** — full numerical walkthrough
   for Llama-2-7B on H100, built step-by-step with the user doing the
   arithmetic. Covers:
   - Param count from first principles (12D² per block + vocab·D)
   - 14 GB weights, 4.7 ms floor to read them
   - 14 GFLOPs per token, 0.014 ms compute → 99.7% idle at B=1
   - Overlap model (total ≈ max(mem, compute), not sum)
   - Batching: B ≈ 330 hits the compute ridge → ~70k tok/s aggregate
   - Memory-capacity limit: KV cache ~1 GB/seq at 2k ctx → only ~66
     seqs fit on 80 GB H100 after weights
   - The gap between 330 (compute ideal) and 66 (memory-feasible) is
     exactly the vLLM / PagedAttention problem

## Key takeaways the user now owns

- Arithmetic intensity = FLOPs / HBM bytes moved. Compare to hardware
  ridge to determine regime.
- H100 ridge ≈ 330 FLOPs/byte (bf16).
- Llama-2-7B decode at B=1: intensity = 1, 330× below ridge, ~210 tok/s
  ceiling, 99.7% math idle.
- Batching amortizes the weight-read cost because W is shared across
  the batch — this is why inference APIs are cheap and self-hosted
  single-stream is expensive per token.
- Real-world B is capped by KV-cache memory capacity, not compute ridge.
  That gap motivates PagedAttention.

## Dangling threads (carried forward)

1. **vLLM study** — user's stated next interest. We've now built the
   motivation from first principles, so the paper will land harder.
2. **Train a bigger model** — still outstanding. 50M-param config,
   30k-step run to get coherent output.
3. **Level 6 (paged KV cache)** — roadmap in wiki appendix, after vLLM.
4. **SwiGLU walkthrough** — never finished past the motivation.
5. **RoPE implementations wiki** at
   `wiki/transformer-primer/14a-rope-implementations.md` — only intro
   filled.
6. **pytorch wiki holes** — 02 (broadcasting), 03 (modules/params/
   buffers), 04 (contiguous/strided memory).
7. **GPU wiki branches not yet taken:**
   - Training memory (params + grads + optimizer + activations),
     parallelism (DP/TP/PP/FSDP).
   - Inference depth: prefill vs decode, quantization, speculative
     decoding, FlashAttention.
   - Kernel-level: how matmul actually executes on SMs, tiling.

## Pickup options for next session

- **vLLM deep dive** — natural next step. Read the paper, trace source,
  understand PagedAttention and continuous batching. The 66-vs-330 gap
  is the hook.
- **Training memory branch** — `wiki/gpu/06-training-memory.md` onward.
  Why a 7B model takes ~112 GB to train (not 14 GB), and how FSDP/ZeRO
  split it. Ties to `wiki/foundations/20-training-memory-four-components.md`.
- **Kernel-level branch** — go one layer deeper into how SMs actually
  execute a matmul (tiling, warps, tensor cores). Needed to truly
  understand FlashAttention.
- **Actually train something bigger** — or any of the pre-existing
  dangling threads.

## Resume prompt

> "Load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md"

**Next session: start Phase 1 of the vLLM learning plan**
(`wiki/gpu/vllm-learning-plan.md`). User has committed to path 2
(PyTorch + Triton), will rent a GPU (4090 on Vast.ai or similar),
and wants to build PagedAttention on top of the existing transformer
rather than starting a new repo. Phase 1 is read-the-paper + note-taking,
no code.
