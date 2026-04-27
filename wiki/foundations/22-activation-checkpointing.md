# Activation checkpointing

## Problem

Activations (intermediate layer outputs stored during forward for backward)
scale as `2 · B · T · D · L`. For deep models, they're often the biggest
memory cost.

## Idea

**Don't store every activation. Re-compute missing ones during backward.**

Trade compute for memory.

## How it works

Normal forward stores every layer's output:
```
h_1 → h_2 → h_3 → h_4 → h_5 → h_6 → loss
 ✓     ✓     ✓     ✓     ✓     ✓
```
Checkpointed forward stores only every k-th layer:
```
h_1 → h_2 → h_3 → h_4 → h_5 → h_6 → loss
 ✓                 ✓                       (checkpoints only)
```
During backward, when a missing activation is needed:
1. Re-run forward from the nearest saved checkpoint to regenerate it.
2. Do the backward step.
3. Discard the recomputed activation.

Like saving keyframes in a video and replaying forward from the nearest
keyframe to reach any intermediate frame.

## Costs

- **Memory:** falls by ~k (if checkpoints are every k-th layer). L saved
  activations → L/k saved.
- **Compute:** extra forward passes during backward. Typical total
  training-step overhead: **~1.3× FLOPs** (one extra forward per
  checkpoint interval).

Substantial memory savings at moderate compute cost. Standard practice
for large models.

## PyTorch API

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.block1, x)   # don't cache inside block1
    x = checkpoint(self.block2, x)
    x = checkpoint(self.block3, x)
    return x
```

For a transformer: wrap each transformer block in `checkpoint(...)`.
Typically 3–5 lines of change for large memory savings.

## Contrast with gradient accumulation

Both cut activation memory, different levers:

| technique | saves | costs | how |
|---|---|---|---|
| gradient accumulation | peak activation by K | K× wall-clock | split batch into K chunks |
| activation checkpointing | activation by ~k | ~1.3× compute | recompute instead of store |

Often used **together** in serious training setups.

## The full memory-saving toolkit

Single-GPU:
1. **Mixed precision (bf16)** — halve bytes per number.
2. **Gradient accumulation** — process batch in chunks.
3. **Activation checkpointing** — recompute instead of store.

Multi-GPU:
4. **FSDP / ZeRO** — shard parameters, gradients, optimizer state
   across devices.

## What's checkpointable

Any **deterministic, differentiable** chunk of computation. Not
transformer-specific — `checkpoint(...)` just means "don't cache
activations for this region, recompute them."

Two gotchas:

1. **Dropout / random ops.** Stochastic forward = different recompute.
   PyTorch's `checkpoint` handles this by default: it preserves RNG
   state so the recompute uses the same drop mask. Free, just know
   it's happening.
2. **In-place ops** on tensors that were inputs to the checkpointed
   region can cause wrong recomputes. Rare footgun.

Granularity is your choice — coarser chunk = more memory saved, more
recompute. Common scopes:
- whole transformer block (most common)
- MLP sub-block only
- attention sub-block only
- any arbitrary region you define
