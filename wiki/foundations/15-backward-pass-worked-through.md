# Backward pass with scalars — worked through

A walk from `y = 3x` up to a 3-layer net, to earn the claim
"**backward FLOPs ≈ 2 × forward FLOPs**."

## 1-layer network

```
y    = W · x
loss = (y - t)²
```
- `x, t` are fixed (data + target). `W` is the learnable weight.
- Chain: `W ──→ y ──→ loss`.

### Two local rates

```
dy/dW      = x                  (since y = W · x)
dloss/dy   = 2(y - t)            (d/dy of (y - t)²)
```

### Chain rule

```
dloss/dW  =  dloss/dy · dy/dW
          =  2(y - t) · x
```

### Numerical trace: `x = 2, t = 3, W = 4`

Forward:
- `y = 4 · 2 = 8`
- `loss = (8 - 3)² = 25`

Backward:
- `dloss/dy  = 2(8 - 3) = 10`
- `dy/dW     = 2`
- `dloss/dW  = 10 · 2 = 20`

Meaning: nudging `W` up by 1 increases loss by ~20. To decrease loss, subtract.

Step (η = 0.01):
```
W_new = 4 − 0.01 · 20 = 3.8
y_new = 3.8 · 2 = 7.6
loss_new = (7.6 − 3)² = 21.16   ✓ dropped from 25
```

---

## 3-layer network — where the pattern shows up

```
h_1  = W_1 · x
h_2  = W_2 · h_1
y    = W_3 · h_2
loss = (y - t)²
```

Need three gradients: `dloss/dW_1, dloss/dW_2, dloss/dW_3`.

### Naive chain-rule expansion

```
dloss/dW_3  =  dloss/dy · h_2
dloss/dW_2  =  dloss/dy · W_3 · h_1
dloss/dW_1  =  dloss/dy · W_3 · W_2 · x
```

Earlier layers have more factors → redundant work if computed independently.

### Smart approach: pass a running "gradient-at-output" leftward

Define `g_i = dloss/d(output of layer i)`:

```
g_3 = dloss/dy              ← from the loss function
g_2 = dloss/dh_2
g_1 = dloss/dh_1
```

Backward, layer-by-layer (right → left):

| layer | receives | compute weight grad | compute grad to pass back |
|------:|:--------:|:-------------------:|:--------------------------:|
| 3     | `g_3`    | `dloss/dW_3 = g_3 · h_2` | `g_2 = g_3 · W_3`       |
| 2     | `g_2`    | `dloss/dW_2 = g_2 · h_1` | `g_1 = g_2 · W_2`       |
| 1     | `g_1`    | `dloss/dW_1 = g_1 · x`   | (no earlier layer)       |

**Every middle layer produces two outputs:**
1. A **weight gradient** → saved for the optimizer.
2. A **gradient w.r.t. its input** → handed left so backward can continue.

That's what "two gradients per layer" means.

---

## Why backward ≈ 2× forward FLOPs

In real layers `h` and `W` are tensors and the "multiplications" above are
matmuls. For a linear layer with `h_in: (B, D_in)`, `W: (D_in, D_out)`:

**Forward** — 1 matmul:
```
h_out = h_in @ W
FLOPs = 2 · B · D_in · D_out
```

**Backward** — 2 matmuls:
```
weight grad:  dloss/dW   = h_in.T @ g        # (D_in, B) × (B, D_out)
input grad:   g_new      = g @ W.T           # (B, D_out) × (D_out, D_in)
```
Both matmuls have the same FLOP count as the forward matmul.

```
backward FLOPs  =  2 × forward FLOPs
step FLOPs      =  forward + backward  =  3 × forward
```

## The 6ND rule

Forward ≈ `2 · tokens · N`.
Training step ≈ 3 × forward = `6 · tokens · N`.

Training a model with **N** parameters on **D** tokens:
```
Total training FLOPs  ≈  6 · N · D
```
This is the single most-used formula in scaling-law / compute-budget work.
