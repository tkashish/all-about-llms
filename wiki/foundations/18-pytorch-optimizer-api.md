# The PyTorch optimizer API

## Big picture

An optimizer takes `(parameters, gradients)` and produces updated parameters.
PyTorch hides the loop behind a small API:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for batch in loader:
    optimizer.zero_grad()     # reset .grad
    loss = compute_loss(...)
    loss.backward()            # populate .grad
    optimizer.step()           # apply update rule
```

Three calls: `__init__`, `zero_grad`, `step`.

## Subclassing `torch.optim.Optimizer`

Skeleton to fill in when writing your own optimizer:

```python
class MyOpt(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                # ... your update rule ...
                p.data -= lr * grad
        return loss
```

## Three concepts to know

### `self.param_groups`

List of dicts. Each dict = one bundle of parameters + their hyperparameters
(`lr`, `weight_decay`, etc.). Lets you use different LRs for different parts
of a model:
```python
optim = AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-3},
    {"params": model.decoder.parameters(), "lr": 1e-4},
])
```
For simple cases: one group with one learning rate.

### `self.state[p]` — per-parameter backpack

A dict-of-dicts keyed by parameter tensor. Persists across `step()` calls.
Used to store running state the update rule needs:

| optimizer | per-param state |
|-----------|-----------------|
| SGD (plain) | none |
| SGD with decaying LR | `t` (step count) |
| AdaGrad | `g²` (running sum of squared grads) |
| Adam / AdamW | `m`, `v` (1st + 2nd moments), `t` (step count) |

Read/write pattern:
```python
state = self.state[p]           # fetch (auto-creates empty dict)
m = state.get("m", torch.zeros_like(p))
state["m"] = m                   # stash for next time
```

### `p.data` — bypass autograd for the update

Parameters track gradients. When you write to `p` directly, PyTorch wants to
track that as an op too. Use `p.data` to edit the underlying tensor without
autograd tracking:
```python
p.data -= lr * grad           # in-place, no graph
# or equivalently:
with torch.no_grad():
    p -= lr * grad
```

## The update rules

### SGD with decaying LR — baseline

```
W_{t+1} = W_t − (lr / √(t + 1)) · g_t
```
State needed per param: `t` only.

### AdamW — what real LLMs use

```
m ← β₁·m + (1 − β₁)·g
v ← β₂·v + (1 − β₂)·g²
m̂ ← m / (1 − β₁ᵗ)           # bias correction
v̂ ← v / (1 − β₂ᵗ)           # bias correction
W ← W − lr · m̂ / (√v̂ + ε)
W ← W − lr · λ · W           # decoupled weight decay (the "W" in AdamW)
```
- `β₁, β₂`: large-LM convention is `(0.9, 0.95)`; PyTorch's default is `(0.9, 0.999)`
- `ε`: small constant, `1e-8` typically
- `λ`: weight decay coefficient (e.g. 0.1)

State needed per param: `m`, `v`, `t`.

### Why "AdamW" not "Adam"

Original Adam folded weight decay into the gradient (`g ← g + λW`), which
interacts weirdly with the moment estimates. AdamW applies weight decay
**directly to W** (decoupled), giving cleaner and empirically better
regularization. It's a 2-line change but matters.

## Memory footprint of optimizer state

Per parameter (float32 state):
- SGD plain: **0 bytes**
- SGD + momentum / AdaGrad / RMSProp: **4 bytes** (1 tensor)
- Adam / AdamW: **8 bytes** (2 tensors; ignore the tiny scalar `t`)

For a 7B-param model with AdamW in fp32 optimizer state:
`7e9 × 8 = 56 GB` for optimizer state alone. That's why FSDP / ZeRO
shard optimizer state across GPUs.

## Implementation checklist

When writing your own AdamW / SGD:

- [ ] Use `self.state[p]` for running state (moments, step counter).
- [ ] Use `p.data -= ...` (or `torch.no_grad()`) for the update — avoid touching autograd.
- [ ] Handle `p.grad is None` (some params may not have gradients on a given step).
- [ ] Order matters in AdamW: the original paper does decoupled decay AFTER the Adam step.
