# AdaGrad `step()` — line-by-line walkthrough

This is the cleanest way to see how `torch.optim.Optimizer`
subclasses work. AdaGrad is simple enough to trace end-to-end, but hits
a real failure mode (LR → 0 plateau) that motivates RMSProp and Adam.

## Why AdaGrad exists

SGD uses one learning rate for every parameter:
```
W ← W − lr · g
```
Some parameters want big updates, others want small ones. AdaGrad gives
each parameter its **own** effective learning rate based on how big its
gradients have been historically:

- big past gradients → shrink LR
- small past gradients → normal LR

To do that, the optimizer needs to **remember** gradient history per
parameter. That memory is `g²` — a running sum of squared past gradients.

Squared so +1 and −1 count the same (magnitude, not sign).

## The full `step()`

```python
def step(self):
    for group in self.param_groups:
        lr = group["lr"]
        for p in group["params"]:
            state = self.state[p]
            grad  = p.grad.data

            g2 = state.get("g2", torch.zeros_like(grad))
            g2 += torch.square(grad)
            state["g2"] = g2

            p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```

## Line-by-line

### The two loops
```python
for group in self.param_groups:
    for p in group["params"]:
```
Visit every parameter tensor. Usually one group; loop exists so PyTorch
can support per-group learning rates.

### Fetch what we need for this parameter
```python
state = self.state[p]       # dict-of-dicts, persists across step() calls
grad  = p.grad.data         # gradient populated by loss.backward()
```
`.data` hands back the raw tensor without autograd tracking.

### Read/update the accumulator
```python
g2 = state.get("g2", torch.zeros_like(grad))   # fetch or init to zeros
g2 += torch.square(grad)                       # add this step's g²
state["g2"] = g2                               # stash for next time
```
First call: no `"g2"` key yet, start from zeros. Later calls: read the
growing accumulator and add to it.

### The update
```python
p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```
As math:
```
p ← p − lr · (grad / √g²)
```
- still gradient descent (subtract LR × direction)
- direction is `grad / √g²` — "normalized by historical gradient size"
- `+ 1e-5` avoids divide-by-zero for parameters with zero history

## Tensor-math gotchas (encountered while implementing)

- `torch.square(x)` and `x ** 2` — equivalent on tensors. Pick a style.
- `math.sqrt(tensor)` — fails. Use `torch.sqrt(tensor)` for elementwise.
  Rule: `math.X` is for Python scalars, `torch.X` is for tensors.

## Classic one-letter bug

```python
g2 += torch.square(g2)    # ❌ squaring the accumulator
g2 += torch.square(grad)  # ✅ squaring the current gradient
```
With the bug, `g2` stays 0 → `√g² = √ε` → the first step's update is
enormous → network dies. Symptom: loss jumps then freezes at a constant.

## Full training loop that exercises it

```python
model = DeepNetwork(dim=8, num_layers=2).to('mps')
optimizer = AdaGrad(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
x      = torch.randn(1, 8, device='mps')
target = torch.ones(1, 8, device='mps')

for i in range(200):
    optimizer.zero_grad()
    y = model(x)
    loss = criterion(y, target)
    loss.backward()
    optimizer.step()
    print(f"step {i}: loss = {loss.item():.4f}")
```

Expected: loss monotonically decreases.

## Debugging failures seen in practice

- **"backward through the graph a second time"**
  → forward pass is outside the loop, so `loss` is the same stale tensor
     every iteration. Move `y = model(x)` and `loss = ...` inside.
- **loss jumps then freezes (e.g. 0.8 → 4.0 → 1.147 constant)**
  → dying ReLU. After a too-big first step, every activation becomes ≤ 0;
     ReLU gradient is 0; no more updates. Fix: lower `lr`, fewer layers,
     or a non-zero target.

## The plateau — an AdaGrad-specific failure mode

Observed curve (200 steps, lr=0.01):
```
step   0: 0.7058
step  30: 0.4282
step 100: 0.3793
step 200: 0.3751   ← frozen
```
Two causes:
1. Half the neurons are dead (ReLU killed them); the survivors can only
   approximate the target, not match it.
2. `g²` accumulates **without decay**, so `lr / √g²` shrinks toward zero
   over time. AdaGrad eventually stops learning. This is the exact bug
   that motivated RMSProp (exponential average of `g²`) and Adam
   (RMSProp + momentum).

## Takeaways

- Optimizer subclass = `__init__` + `step()`, nothing more.
- `self.state[p]` is per-parameter persistent storage.
- `p.data` bypasses autograd for the weight update.
- Adaptive optimizers work by dividing gradients by some measure of
  gradient history (raw sum for AdaGrad, EMA for RMSProp/Adam).
