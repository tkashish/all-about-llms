# The four training-loop calls

Every PyTorch training step is built from four calls. Each does one
specific job; skipping any one breaks training. This note pins down
what each does, what it reads, what it writes.

## The whole step

```python
for step in range(num_steps):
    # forward
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

    # the four calls
    optimizer.zero_grad()                            # 1
    loss.backward()                                   # 2
    torch.nn.utils.clip_grad_norm_(                   # 3
        model.parameters(), max_norm=1.0
    )
    optimizer.step()                                  # 4
```

## 1. `optimizer.zero_grad()` — reset gradients to zero

**Plain English:** wipe last step's gradients before computing new ones.

**Why:** `loss.backward()` *accumulates* into `.grad` (adds, not
overwrites). Without zeroing first, this step's gradients would be
summed with last step's. Effective learning rate would grow every step
→ explosion.

**Reads:** nothing of importance.
**Writes:** `param.grad = 0` (or `None` with `set_to_none=True`) for
every parameter.

**Gotcha — accumulation is actually useful:** if you want gradient
accumulation (simulate bigger batch on limited memory), you skip
`zero_grad` for N micro-batches, call `backward` N times so `.grad`
sums, then `step` + `zero_grad` once. That's the whole mechanism.

## 2. `loss.backward()` — compute gradients

**Plain English:** for every parameter, fill in `.grad` with
"how much should this parameter change to decrease the loss?"

**Mechanics:** PyTorch walks the computation graph backward (the DAG
built during forward), applies chain rule at each op, and
**accumulates** the result into each parameter's `.grad`.

Each op has a pre-registered backward function written in C++. The
graph walk composes them. See
[17-pytorch-autograd-mechanics.md](../foundations/17-pytorch-autograd-mechanics.md)
for how autograd builds and walks that graph.

**Reads:** the computation graph + intermediate tensors saved during forward.
**Writes:** `param.grad` for every `nn.Parameter` in the model that
contributed to `loss`.

**What it does NOT do:** does not update any weights. Parameters still
have their old values after `backward()`. Only `.grad` has changed.

**Common mistakes:**
- Calling `.backward()` on a **non-scalar** (e.g., a vector loss) —
  errors unless you pass `gradient=...`. Fix: reduce to scalar with
  `.sum()` or `.mean()` first (though cross-entropy already returns a
  scalar).
- Calling `.backward()` twice on the same loss — the graph is freed
  after the first call. Fix: use `retain_graph=True` (rarely needed)
  or run forward again.

## 3. `torch.nn.utils.clip_grad_norm_(params, max_norm)` — cap gradient magnitude

**Plain English:** if the overall gradient is too big, shrink it.
Keep its direction, cap its magnitude.

**Why:** unusually large gradients (from a weird batch, numerical
instability, or just bad luck) can send weights somewhere
unrecoverable. Clipping caps the damage.

**Mechanics:**
1. Flatten all params' `.grad` into one big vector.
2. Compute L2 norm: `g_norm = sqrt(sum(g² for every gradient element))`.
3. If `g_norm > max_norm`, scale every gradient by `max_norm / g_norm`.
4. If `g_norm <= max_norm`, do nothing.

Concrete: with `max_norm=1.0`:
- Total gradient norm is 0.5 → do nothing.
- Total gradient norm is 5.0 → multiply all gradients by `1.0/5.0 = 0.2`.
  After scaling, new norm is exactly 1.0.

**Direction preserved, magnitude capped.** Every parameter's gradient
is scaled by the same factor, so the *ratio* between any two gradients
is unchanged.

**Reads:** `param.grad` for every parameter.
**Writes:** `param.grad` for every parameter (scaled down if triggered).

**Note on `_` in the name:** PyTorch convention — function names
ending in `_` mutate inputs in place. `clip_grad_norm_` modifies
`.grad` tensors directly; the return value (total norm before
clipping) is usually discarded.

**Typical `max_norm`:** 1.0 is standard for transformer training.
Higher (10.0) is permissive; lower (0.5) stricter. Without clipping,
transformer training often diverges mid-run on a bad batch.

## 4. `optimizer.step()` — actually update the weights

**Plain English:** use each parameter's gradient to update it.

**Mechanics (AdamW specifically):** for each parameter `p` with `g = p.grad`:

```
m ← β₁·m + (1 − β₁)·g                         # first moment
v ← β₂·v + (1 − β₂)·g²                        # second moment
m̂ ← m / (1 − β₁ᵗ)                            # bias correction
v̂ ← v / (1 − β₂ᵗ)
p.data -= lr · m̂ / (√v̂ + ε)                 # main update
p.data -= lr · λ · p.data                     # decoupled weight decay
```

`m`, `v`, `t` are kept in `optimizer.state[p]` across steps — this is
how AdamW gets its momentum. See
[18-pytorch-optimizer-api.md](../foundations/18-pytorch-optimizer-api.md)
for the full optimizer API.

**Reads:** `param.grad`, `optimizer.state[p]` (m, v, step count).
**Writes:** `param.data` (the actual weights), `optimizer.state[p]`.

After `step()`, your weights have new values and the model has
"learned" a tiny bit from this batch.

## Summary flow

```
call                      reads                         writes
─────────────────────────────────────────────────────────────────────
zero_grad()               —                             param.grad = 0
backward()                computation graph             param.grad += dL/dp
clip_grad_norm_(...)      param.grad (all params)       param.grad (scaled)
step()                    param.grad, optimizer state   param.data, optimizer state
```

## What breaks if you skip each one

| skip | what breaks |
|---|---|
| `zero_grad()` | gradients accumulate indefinitely → effective LR grows each step → divergence |
| `backward()` | `.grad` stays None → `step()` does nothing → loss stays constant |
| `clip_grad_norm_` | occasional huge gradient → weight explosion → loss → nan |
| `step()` | weights never update → loss stays constant |

## Order matters

The canonical order is:

```
zero_grad → forward → loss → backward → clip → step
```

The only flexible one is `zero_grad`: it can go at the start of the
step (before forward) or at the end of the previous step (after
`step`). Both equivalent; start-of-step is more common because it
pairs the "reset and run" logically.

`clip_grad_norm_` MUST go after `backward()` (needs `.grad` to exist)
and before `step()` (needs to clip before weights are updated).

## Key takeaways

- Every training step is four PyTorch calls, each doing one job.
- `zero_grad` wipes, `backward` fills, `clip` scales, `step` applies.
- `.grad` is the shared conduit — backward writes it, clip modifies
  it, step reads it.
- The full cycle: `data → forward → loss → backward → clip → step →
  zero_grad → next batch`.
