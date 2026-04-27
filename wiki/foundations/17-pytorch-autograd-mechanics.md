# PyTorch autograd — how backward actually runs in code

The math is done (see 14, 15). Here's how PyTorch turns it into mechanics.

## `requires_grad=True`

Flag on a tensor meaning "track me; I'm a leaf for backward to populate."

```python
x = torch.tensor([1., 2, 3])                        # data  — no grad
w = torch.tensor([1., 1, 1], requires_grad=True)    # weight — grad
```

`nn.Parameter(...)` sets this automatically. That's one of its two jobs
(the other: register itself with the containing `nn.Module`).

## The computation graph (autograd)

Every op on a `requires_grad=True` tensor is **recorded** as a node in a
DAG. The graph stores:
- inputs
- the function applied
- how to compute local gradients

For `loss = 0.5 * (x @ w - 5)**2`:
```
w ─┐
   ├── @ ──→ pred_y ──sub(5)──→ ──pow(2)──→ ──mul(0.5)──→ loss
x ─┘
```

No forward code explicitly wrote "backward logic." PyTorch derives it from
the operations you chose.

## `loss.backward()`

Starts at the scalar `loss`, walks the graph right-to-left, applies chain
rule at each node, and **accumulates** results into every leaf's `.grad`.

```python
loss.backward()
print(w.grad)   # tensor([1., 2., 3.])
```

### Rule: `loss` must be a scalar

`backward()` from a non-scalar needs an explicit gradient arg. Easiest: use
`.sum()` or `.mean()` to reduce to a scalar first.

## `.grad` accumulates — but the graph gets freed

Calling `loss.backward()` a second time on the same `loss` **fails**:
```
RuntimeError: Trying to backward through the graph a second time ...
```
The computation graph is freed after one backward pass to save memory.
To keep it: `loss.backward(retain_graph=True)`.

The real accumulation pattern (used by gradient accumulation) is:
**fresh forward → fresh graph → backward → repeat**, letting `.grad`
build up across iterations before a single `optimizer.step()`:
```python
for micro_batch in batch:
    loss = model(micro_batch)   # new graph each time
    loss.backward()             # .grad accumulates across iterations
optimizer.step()
optimizer.zero_grad()
```

Standard per-step dance (no accumulation):
```python
optimizer.zero_grad()   # reset .grad on every parameter
loss.backward()
optimizer.step()
```

## Dtype rule: gradients need floats

```python
torch.tensor([1, 2, 3], requires_grad=True)
# RuntimeError: Only Tensors of floating point and complex dtype can require gradients
torch.tensor([1., 2., 3.], requires_grad=True)   # ✓
```
Gradients are derivatives (real-valued nudges). Integers can't be nudged
fractionally, so PyTorch refuses.

## `torch.Tensor` vs `torch.tensor`

- `torch.tensor(data, ...)` — **factory function**, accepts `requires_grad`,
  `dtype`, `device`. Use this.
- `torch.Tensor(...)` — legacy class constructor, no kwargs like
  `requires_grad`. Avoid.

## `grad_fn` — how to tell if a tensor is in the graph

Intermediate tensors show a `grad_fn`:
```
pred_y  → tensor(6., grad_fn=<DotBackward0>)
loss    → tensor(0.5, grad_fn=<MulBackward0>)
```
Leaves (inputs with `requires_grad=True`) don't have `grad_fn`; they just
have `.grad` populated after `backward()`.

## The full per-step dance

```python
optimizer.zero_grad()       # 1. clear old grads
y = model(x)                # 2. forward
loss = criterion(y, target) # 3. compute scalar loss
loss.backward()             # 4. backward — populate .grad everywhere
optimizer.step()            # 5. apply the update rule
```

## Hand-check the math

```python
x = [1, 2, 3], w = [1, 1, 1]
pred_y = x @ w       = 6
loss   = 0.5(6-5)²   = 0.5

dloss/d(pred_y) = pred_y - 5 = 1
dpred_y/dw_i    = x_i
dloss/dw_i      = 1 · x_i    = x_i

w.grad → [1, 2, 3]          ✓ matches PyTorch
```

## Debugging aids

- `tensor.retain_grad()` — by default intermediate (non-leaf) tensors
  don't keep their gradients. `retain_grad()` forces them to. Useful for
  inspecting `h1.grad`, `h2.grad` mid-network.
- `torch.autograd.grad(outputs, inputs)` — one-off gradient computation
  without populating `.grad` attributes.
- `with torch.no_grad():` — run forward ops but **don't build the graph**.
  Used at inference time — saves memory and compute since no backward is
  needed.

## What nn.Parameter does, reprised

1. Sets `requires_grad=True`.
2. Registers itself as a parameter of the containing `nn.Module`.

Plain `torch.randn(...)` fails on both: no grad tracking, not registered.
`nn.Parameter(torch.randn(...))` gets both.
