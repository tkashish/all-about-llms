# Learning rate schedules — warmup + cosine decay

## What is an LR schedule?

A rule that changes the learning rate over the course of training,
instead of keeping it fixed. Most production LLM training uses some
form of **warmup + decay**.

## What problem does it solve?

Two distinct problems, solved by two distinct phases.

### Problem 1: the first step is risky

At step 0 the model is random. A big first update can push the model
into a bad basin it never recovers from — this is especially common in
deep transformers with many layers. The fix: **warmup.**

### Problem 2: late in training, big steps overshoot

After many steps the model is close to a good minimum. A full-magnitude
LR step bounces you around the minimum instead of settling in it. The
fix: **decay.** Shrink the learning rate as you get close.

## The two phases

```
LR
 │
 │       ╱─────╮
 │      ╱       ╲
 │     ╱         ╲
 │    ╱           ╲_____
 │   ╱
 │  ╱  warmup    cosine decay   (plateau at min_lr)
 └──────────────────────────── step
    0    W                    max_steps
```

- **Steps 0 to W (warmup):** linear ramp from 0 up to `max_lr`.
- **Steps W to max_steps (decay):** cosine curve down to `min_lr`.

Typical values:

| parameter | common value |
|-----------|--------------|
| `max_lr` | `3e-4` to `1e-3` (for transformer LMs) |
| `min_lr` | `max_lr / 10` |
| `W` (warmup steps) | 1-2% of `max_steps` (e.g., 500 of 30000) |

## The formula

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps          # linear ramp
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

Walking through cosine decay:

- `progress = 0` (start of decay): `cos(0) = 1` → LR = `max_lr` ✓
- `progress = 0.5` (midway): `cos(π/2) = 0` → LR = `(max_lr + min_lr) / 2`
- `progress = 1` (end): `cos(π) = -1` → LR = `min_lr` ✓

The `0.5 * (1 + cos)` remaps cosine's `[-1, 1]` range into `[0, 1]`, then
scales between `min_lr` and `max_lr`.

## Why cosine, specifically?

Options (all work):

- **Linear decay** — straight line down. Simple. A bit aggressive early.
- **Step decay** — drop LR by 10x at fixed step counts. Classic for
  ResNets. Too discrete for transformers.
- **Cosine decay** — smooth, decays slowly at first (when model is
  still improving fast), then faster, then slowly again near the end.
  Empirically best for LLM training.

No deep theoretical reason cosine wins — it was found empirically,
then everyone copied. It's also what Chinchilla, Llama, GPT-3, and
most recipes use.

## Applying the schedule inside the training loop

```python
optimizer = AdamW(model.parameters(), lr=get_lr(0))

for step in range(num_steps):
    # update LR for this step
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # ... forward, backward, step ...
```

The `for pg in optimizer.param_groups` loop sets LR on all groups. For
a single-group optimizer this is one assignment; for multi-group it
scales all groups by the same factor (which is usually what you want).

## When can you skip the schedule?

Constant LR is fine when:

- Model is small (< 10M params)
- Training is short (< few thousand steps)
- You're doing a smoke test / debugging run, not a production train

For anything else — especially 10M+ params or 10k+ steps — add warmup
and decay. They add ~10 lines and prevent failure modes that would
otherwise waste hours of compute.

## Common gotchas

1. **Past `max_steps`:** the cosine formula extrapolates past the
   minimum if `step > max_steps`. Fix with `progress = min(progress, 1)`.
2. **Starting at LR=0 for step 0:** my formula above gives LR=0 at
   step 0. Some implementations start at 1e-7 instead to keep the
   optimizer's first moment non-zero. Minor.
3. **Forgetting `optimizer.param_groups`:** changing `optimizer.lr`
   directly doesn't work — AdamW reads `lr` per param group. Always
   update via the `param_groups` list.

## Key takeaways

- Warmup prevents bad first-step updates from wrecking init.
- Decay prevents overshooting the minimum late in training.
- Cosine is the empirical winner for transformer LMs.
- `max_lr` around 3e-4, `min_lr = max_lr / 10`, warmup 1-2% of total
  steps is the standard recipe.
- Constant LR is fine for tiny models or debugging — add the schedule
  before production runs.
