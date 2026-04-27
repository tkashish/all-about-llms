# Gradient accumulation

## Problem

Activation memory scales linearly with batch size:
```
activation_memory ≈ 2 · B · D · L   (bf16, per-layer size × layers)
```
- Bigger `B` → more stable gradients, better GPU utilization → we want big `B`.
- Bigger `B` → bigger activations → eventually OOM.

Classic tension: want big batch for training quality, can't fit it in memory.

## Solution

Split the big batch into `K` micro-batches. Run forward+backward on each
one sequentially, **without zeroing gradients in between**. After K
iterations, call `optimizer.step()` once.

```python
optimizer.zero_grad()
for micro in micro_batches:          # K iterations
    loss = loss_fn(model(micro))
    loss.backward()                  # .grad accumulates each time
optimizer.step()                     # one update with the summed grad
```

## Why this works

`loss.backward()` **adds** to `.grad` instead of overwriting. After K calls:
```
param.grad = grad_from_micro_1 + grad_from_micro_2 + ... + grad_from_micro_K
```
Gradient of a sum = sum of gradients (linear op). So the accumulated gradient
equals what one big backward would produce.

## Why it saves activation memory

Activations live only between a forward and its backward. Once backward
finishes on a micro-batch, PyTorch frees its graph; that memory is reused
for the next micro-batch's forward.

```
peak activation memory = one micro-batch's worth, not full batch's worth
```

## Tradeoff

|  | one big batch | gradient accumulation (K micro-batches) |
|---|---|---|
| effective batch | B | B (= K × b_micro) |
| peak activation memory | B × unit | b_micro × unit |
| wall-clock | T | ~K · T (sequential) |
| gradient signal | same | same |

You trade **time** for **memory**. No parallelism win — pure memory savings.

## The loss-averaging subtlety

If your loss averages over the batch (like `F.mse_loss`, `F.cross_entropy`),
each micro-batch's loss is already divided by its local size. Summing K
such gradients gives you **K times** the "per-example average" — too big.

Fix: scale down each micro-batch loss before backward:
```python
loss = loss_fn(model(micro)) / K
loss.backward()
```
Then the accumulated gradient matches what a single big-batch average would
have produced.

## Where you'll see it

Every modern LLM training config has a `gradient_accumulation_steps`
hyperparameter. Effective global batch size =
`num_gpus · per_gpu_batch · gradient_accumulation_steps`.
