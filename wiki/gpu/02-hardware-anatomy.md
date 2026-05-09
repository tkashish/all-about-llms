# GPU hardware anatomy — SMs and the memory hierarchy

## Why GPUs exist (for ML)

**CPUs** optimize for running a few things fast and flexibly — ~10–50
cores, each doing its own thing. Great for interactive programs,
compilers, databases.

**GPUs** optimize for running **thousands of identical calculations
in parallel**. Consumer cards have ~5000 tiny cores; data-center
cards like A100 have ~7000, L4 has 7680.

Neural networks are giant matrix multiplications repeated many times.
That's a workload made for the parallel-cores model — hence the
GPU takeover of ML.

## What an SM is

**SM = Streaming Multiprocessor.** One "compute unit" on the GPU.
A GPU is really a collection of SMs running in parallel.

Concretely, an H100 has **132 SMs**. Each SM has:
- Its own math units (including tensor cores for matmul)
- Its own registers
- Its own small scratchpad memory (shared memory / SRAM)

Analogy: if a GPU were a factory, SMs are the individual workstations.
Each workstation has its own tools (math units) and its own workbench
(registers + scratchpad). They all pull raw materials from the same
warehouse (HBM).

## "CUDA" is overloaded — SM vs CUDA cores

Two different things share the name "CUDA":

1. **CUDA cores** — a type of *hardware unit* inside an SM. A specific
   kind of arithmetic circuit.
2. **CUDA** (the platform) — NVIDIA's *programming language/toolkit*
   for GPUs (C-like syntax, `nvcc` compiler, CUDA runtime API).

Naming overlap only. CUDA-the-platform programs CUDA-the-cores, among
other things, but they're different nouns.

### The hierarchy

```
GPU
├── SM 0
│   ├── 128 CUDA cores        (general-purpose arithmetic)
│   ├── 4 Tensor cores        (matmul specialists)
│   ├── Registers             (~64 KB)
│   └── Shared memory / L1    (~128 KB)
├── SM 1
│   ├── 128 CUDA cores
│   ├── 4 Tensor cores
│   ...
└── ...  (60 SMs on L4, 132 on H100)
```

- **SM = compute unit** (container).
- **CUDA core = a general-purpose math unit inside an SM.**
- **Tensor core = a specialized matmul unit inside an SM.**

Total core counts come from multiplication:

- L4: 60 SMs × 128 CUDA cores = **7680 CUDA cores**; 60 × 4 = **240 tensor cores**.
- H100: 132 SMs × 128 CUDA cores = **16,896 CUDA cores**; 132 × 4 = **528 tensor cores**.

When you launch a Triton kernel, you schedule **blocks of threads
onto SMs.** Inside each SM, the CUDA cores and Tensor cores do the
actual math. The programming abstraction starts at the SM level —
you rarely address individual cores directly.

## Two kinds of "cores" inside an SM

**CUDA cores — general-purpose.** One scalar arithmetic op per core
per cycle: add, multiply, compare, shift, divide. Work with any
dtype. Handle everything that isn't matmul: softmax, layer norm,
element-wise ops, reductions, control flow.

**Tensor cores — one trick, extremely well.** Specialized silicon
that computes `D = A × B + C` on small matrix tiles (e.g. 16×16
fp16) in a single cycle. Orders of magnitude more matmul throughput
than CUDA cores, but only matmul, only specific precisions (fp16,
bf16, int8, fp8).

Why both exist: matmul dominates neural networks — 80%+ of compute.
Dedicated silicon is a huge win. But softmax, normalizations,
activations aren't matmul and still need CUDA cores.

Analogy: CUDA cores are a full kitchen — can cook anything, one pan
at a time. Tensor cores are a professional pizza oven — only pizza,
but 50 pizzas at once.

In practice, PyTorch and Triton both invoke tensor cores implicitly
when you do fp16/bf16 matmul; you rarely program them directly.

## The memory hierarchy

Smaller = faster = closer to the math units. Numbers are H100, bf16:

| Layer            | Size                 | Bandwidth   | Scope    |
|------------------|----------------------|-------------|----------|
| HBM              | 80 GB                | ~3 TB/s     | Shared   |
| L2 cache         | ~50 MB               | ~12 TB/s    | Shared   |
| SRAM / shared    | ~228 KB per SM       | ~20 TB/s    | Per-SM   |
| Registers        | ~64 KB per SM        | effectively free | Per-SM |

Total register storage across the whole GPU: 132 × 64 KB ≈ 8 MB.

## The structural shape

```
        HBM (shared across all 132 SMs)
              │
         L2 cache (shared)
              │
   ┌──────────┼──────────┐
   │          │          │
  SM 0      SM 1       ... SM 131
  ├ SRAM   ├ SRAM       ├ SRAM      (per-SM)
  └ regs   └ regs       └ regs      (per-SM)
```

## L4 at a glance — the whole hardware picture

The consolidated mental model for the GPU we're working on in
Phase 3:

```
┌─────────────────────────────────────────────────────────────┐
│                        GPU (L4)                              │
│                                                              │
│    ┌──────────────────────────────────────────────────┐     │
│    │             HBM — 24 GB (off-chip)                │     │  ← big, shared,
│    │ (weights, activations, KV cache all live here)    │     │    ~300 GB/s
│    └──────────────────────┬───────────────────────────┘     │
│                           │                                  │
│    ┌──────────────────────┴───────────────────────────┐     │
│    │              L2 cache — ~48 MB                    │     │  ← shared, auto
│    └──────────────────────┬───────────────────────────┘     │
│                           │                                  │
│    ┌──────┐ ┌──────┐ ┌──────┐           ┌──────┐            │
│    │ SM 0 │ │ SM 1 │ │ SM 2 │   ...     │SM 59 │            │  ← 60 SMs on L4
│    └──────┘ └──────┘ └──────┘           └──────┘            │
└─────────────────────────────────────────────────────────────┘

Zoom into one SM:

┌──────────────────────────────────────────┐
│                   SM                      │
│                                           │
│   128 CUDA cores      4 Tensor cores      │  ← math units
│                                           │
│   4 warp schedulers                       │  ← rotation logic
│                                           │
│   ~64 KB registers                        │  ← fastest, per-thread
│   ~128 KB SRAM (shared mem / L1)          │  ← fast, manually managed
└──────────────────────────────────────────┘
```

Everything you need to know about the GPU hardware:
- 60 SMs + shared HBM + shared L2.
- Each SM has 128 CUDA cores + 4 Tensor cores + its own fast local
  memory + 4 warp schedulers.

The execution model (how kernels map work onto this hardware — grids,
blocks, warps, threads) is a separate layer on top, covered in the
next note.

## Other execution units inside the SM

CUDA cores and Tensor cores are the two most prominent, but an SM
actually contains **several different kinds of specialized hardware**
that coexist. Each instruction routes to the unit that can execute
it. The units run independently — when one is busy, the others can
be active or idle.

**Full list per SM (Ampere/Ada rough counts):**

| Unit | How many | What it does | Throughput per cycle |
|---|---|---|---|
| CUDA cores | 128 | ADD, MUL, FMA, int, logic | 128 |
| Tensor cores | 4 | matmul tiles (fp16/bf16/int8/fp8) | ~256 MAC/cycle |
| DIV / SQRT units | ~4 | floating-point division, sqrt | 4 |
| SFU (Special Function Units) | ~4 | exp, log, sin, cos, rcp, rsqrt | 4 |
| Load/Store units | ~4 | memory ops to HBM/SRAM | handles 32 requests/cycle |

**Key points:**

- **These are all separate physical circuits** on the SM die.
  "128 CUDA cores" means 128 literal ADD/MUL circuits. "4 DIV units"
  means 4 separate division circuits. CUDA cores don't "become" DIV
  units for divide instructions — DIV units are their own hardware.
- When a warp issues an instruction, the scheduler routes it to the
  matching unit. Other units that aren't servicing that instruction
  are idle or busy with other warps' instructions.
- Asymmetry exists because circuits are different sizes. ADD/MUL are
  cheap → 128 per SM. DIV is complex (iterative algorithm) → only
  ~4 per SM. Transcendentals go through an even smaller SFU.

### How warp throughput depends on the unit

A warp has 32 threads. A single warp instruction services all 32
threads, but **the per-cycle throughput depends on how many of the
targeted hardware unit there are.**

| Warp issues | Hardware used | Lanes | Time for 32 threads |
|---|---|---|---|
| ADD / MUL / FMA | CUDA cores | 32 (of 128) | 1 cycle (4-cycle latency) |
| DIV | DIV unit | 4 | **8 cycles** |
| EXP / LOG / SIN | SFU | 4 | 8 cycles |
| matmul tile | Tensor core | whole-tile op | 1 cycle (special) |
| HBM load | Load/Store | 32 req/cycle | 1 cyc to issue; ~700 cyc to return |

**Corrected mental model:**

> "Warp → 32 threads → 32 cores → 1 cycle" only holds for
> instructions that hit a unit with ≥32 lanes (ADD/MUL/FMA).
>
> For DIV (4 lanes), the 32 threads serialize through the 4 lanes,
> taking ~8 cycles. The CUDA cores sit idle during this time — a
> different unit is handling the DIV.

### What this means for kernel performance

- **ADD/MUL/FMA-heavy code** is fast — 128 CUDA cores at max throughput.
- **DIV-heavy code** bottlenecks on ~4 DIV lanes per SM. ~32× slower
  than ADD. Avoid DIV; multiply by precomputed reciprocals where
  possible.
- **Softmax's `exp`** uses the SFU (4 lanes). Real cost in long-context
  attention. Fast-math / reduced-precision `exp` variants help.
- **fp64** on consumer chips is catastrophic — tiny fp64 unit, ~1/32
  the throughput of fp32. Avoid unless absolutely necessary.
- **Matmul** uses tensor cores — dedicated silicon, enormous
  throughput. One reason ML loves the GPU.

## Variation across GPU architectures

The *shape* of the mental model (SM → schedulers → warp pools →
cores) is identical across all modern NVIDIA GPUs. Only the counts
differ.

**Fixed across all NVIDIA GPUs:**
- Warp size: **32 threads** (since 2006; unchanged).

**Per-GPU numbers:**

| GPU | Arch | SMs | CUDA cores/SM | Tensor cores/SM | Warp schedulers/SM | Max resident warps/SM |
|---|---|---|---|---|---|---|
| Tesla V100 | Volta | 80 | 64 | 8 (1st gen) | 4 | 64 |
| T4 | Turing | 40 | 64 | 8 (2nd gen) | 4 | 32 |
| RTX 3090 | Ampere | 82 | 128 | 4 (3rd gen) | 4 | 48 |
| A100 | Ampere | 108 | 64 | 4 (3rd gen) | 4 | 64 |
| A10G | Ampere | 80 | 128 | 4 (3rd gen) | 4 | 48 |
| **L4** | **Ada** | **60** | **128** | **4 (4th gen)** | **4** | **48** |
| RTX 4090 | Ada | 128 | 128 | 4 (4th gen) | 4 | 48 |
| H100 | Hopper | 132 | 128 | 4 (4th gen) | 4 | 64 |

The A10G (Phase 3 instance) and L4 (our primary running example)
are both 4-scheduler, 128-CUDA-core, 4-tensor-core per SM — the
diagrams work for both. A10G just has 80 SMs to L4's 60.

**Non-obvious detail — consumer vs datacenter split:**
Ampere gaming chips (RTX 30xx) advertise "128 cores per SM" but
only 64 can do int32 at a time — the other 64 are fp32-only.
Datacenter Ampere (A100) is cleaner: 64 "pure" cores, each does
fp32 or int32. Rarely matters for ML code; PyTorch and Triton
handle it transparently.

**Querying the running GPU at runtime:**

```python
p = torch.cuda.get_device_properties(0)
print(p.multi_processor_count)      # SM count
print(p.major, p.minor)              # Compute capability → arch generation
```

`multi_processor_count` is the single most important number for
kernel design — it sets your grid-size targets. CUDA cores per SM
isn't directly queryable; derive from compute capability or the
table above.

## Consequence for code

- **Cross-SM communication goes through HBM or L2.** Slow.
- **Within an SM, data in SRAM/registers is basically free to reuse.**

A well-written kernel loads a chunk from HBM into SRAM *once*, then
has the SM chew on it many times before writing the result back.
This is the core trick behind FlashAttention and most fast kernels:
**maximize reuse inside the SM, minimize round-trips to HBM.**

When we say "memory-bound," we almost always mean **HBM-bound** —
waiting on the big slow outer layer. Inner layers (L2, SRAM, registers)
are fast enough to keep the math units fed.

## What's next

The next note introduces **arithmetic intensity** — FLOPs per byte
loaded from HBM — the single number that tells you whether a given
kernel is doing enough on-SM work to keep the math units busy.
