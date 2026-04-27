# Training memory — the 4 components

During training, the GPU holds four kinds of numbers simultaneously:

1. **Parameters** — the weights themselves
2. **Gradients** — one per parameter, from `loss.backward()`
3. **Optimizer state** — running averages the optimizer needs
4. **Activations** — intermediate layer outputs, kept for backward

## Per-parameter costs

| component | precision | bytes per parameter |
|-----------|-----------|---------------------|
| parameters | bf16 | **2** |
| gradients | bf16 | **2** |
| optimizer state (AdamW) | fp32 | **8** (m + v = 2 tensors × 4 bytes) |
| optimizer state (AdaGrad, SGD+momentum, RMSProp) | fp32 | **4** (1 tensor × 4 bytes) |
| optimizer state (plain SGD) | — | **0** |

### Why optimizer state is fp32 but gradients are bf16

Gradients are computed fresh every step and used immediately — no
accumulation error.
Optimizer state (`m`, `v`, `g²`) updates incrementally every step;
tiny numerical errors compound over thousands of iterations. fp32 keeps
the running averages stable.

## Activations — the different one

Unlike the other three, activations scale with **batch size** and
**sequence length**, not just parameter count:

```
activation_numbers ≈ B · T · D · L   (one vector per layer, per token)
activation_memory  ≈ 2 · B · T · D · L     (bf16)
```

- `B` = batch size, `T` = sequence length, `D` = hidden dim, `L` = layers
- For a non-transformer toy `B · D · L` (no T dimension)

Activations are cached during forward because **backward needs them**:
```
∂loss/∂W = h_in.T @ ∂loss/∂h_out
```
Layer i's weight gradient needs that layer's input `h_{i-1}`, which was
produced during forward.

### Why "activations"?

Legacy neuroscience term — original neural nets were modeled on neurons
that "activate" (fire) when stimulated. Modern usage just means
"intermediate tensors between layers."

## Memory budget for a 7B model with AdamW

| component | formula | size |
|-----------|---------|-----:|
| parameters | `2 · N` | 14 GB |
| gradients | `2 · N` | 14 GB |
| optimizer state | `8 · N` | 56 GB |
| activations | `2 · B · T · D · L` | 20–40 GB (depends on B, T) |
| **total** | | **~100–130 GB** |

A single H100 has **80 GB**. You cannot train a 7B model on one GPU
without tricks.

## Consequences — why each technique exists

| technique | what it saves | how |
|-----------|---------------|-----|
| **Mixed precision (bf16)** | everything × 2 | store in bf16 instead of fp32 |
| **Activation checkpointing** | activations | don't cache all; re-compute in backward |
| **Gradient accumulation** | activations | use small micro-batches, accumulate grads |
| **ZeRO-1** | optimizer state | shard fp32 state across GPUs |
| **ZeRO-2** | + gradients | shard grads across GPUs too |
| **ZeRO-3 / FSDP** | + parameters | shard params too — only one GPU holds each shard |
| **CPU / NVMe offload** | any | move cold tensors to system RAM or disk |

## Scaling observations

- **Optimizer state** (`8N` with AdamW) is usually the single biggest eater
  of memory — bigger than params + grads combined. Top target for FSDP.
- **Activations** are the only component that scales with batch/seq.
  Longer context → activations dominate. Fixed by checkpointing.
- **Parameters + gradients** are fixed by model size. Only shardable
  via FSDP.

## Toy example: DeepNetwork (D=8, L=3, B=2) with AdaGrad

```
N = D² · L = 192
parameter_memory      = 2 · 192  = 384 B
gradient_memory       = 2 · 192  = 384 B
optimizer_state_memory = 4 · 192 = 768 B   (AdaGrad: 1 fp32 per param)
activation_memory     = 2 · B · D · L = 2·2·8·3 = 96 B
total                                              = 1632 B  ≈ 1.6 KB
```

This toy exists so every number can be traced by hand. The same formulas
scale up directly to GPT-sized models — just with N ~ 10¹⁰ and T ~ 10³.

## Compute vs memory — the B and N framing

```
compute ≈ 6 · B · N                         FLOPs/step
memory  ≈ C_param · N   +   2 · B · N       bytes
          └──── fixed ───┘   └─ scales with batch ─┘
```
- `B` = tokens per step (batch × sequence length)
- `N` = total parameters
- `C_param` depends on the optimizer (see table below)

### Parameter-side memory per optimizer

Parameter-side = params + gradients + optimizer state (no activations).

| optimizer | params (bf16) | grads (bf16) | opt state (fp32) | **total bytes / param** |
|-----------|:-------------:|:------------:|:----------------:|:-----------------------:|
| SGD (plain) | 2 | 2 | 0 | **4·N** |
| SGD + momentum | 2 | 2 | 4 (v) | **8·N** |
| AdaGrad | 2 | 2 | 4 (g²) | **8·N** |
| RMSProp | 2 | 2 | 4 (v) | **8·N** |
| **AdamW** | 2 | 2 | 8 (m + v) | **12·N** |

Activation-side (all optimizers):
```
activations ≈ 2 · B · N bytes   (bf16, rough constant)
```

### Consequences

- **Compute scales with `B · N`** — both levers cost more.
- **Parameter-side memory is fixed in `N`** — doesn't care about batch.
- **Activations scale with `B · N`** — only component that grows with batch.
- If you hit OOM: shrink `B` to cut activations. Then use **gradient
  accumulation** to get the same effective batch without the memory.
