# Weight initialization — the cheat sheet

Different kinds of weights need different init scales. The scale depends
on **how the weight will be used downstream**.

## Rule 1 — Linear layers (matmul weights): `1/√fan_in`

```python
W = nn.Parameter(torch.randn(d_in, d_out) / math.sqrt(d_in))
```

### Why

`y = x @ W`. If `x` has unit variance and each entry of `W` has unit
variance, each output `y[i]` is a sum of `d_in` products. Variance of a
sum of `d_in` independent terms ≈ `d_in`. So output variance explodes with
layer count.

Scaling `W` by `1/√d_in` → `Var(W) = 1/d_in` → output variance stays ~1.
Activations neither blow up nor vanish through the stack.

Variants (same idea, minor tweaks):
- **Xavier / Glorot**: `1/√(d_in + d_out)` — slightly different denom.
- **Kaiming / He**: `√(2/d_in)` — adjusted for ReLU (since ReLU kills
  half the outputs, variance needs to be 2× higher going in).

## Rule 2 — Embedding tables: small constant, e.g. `0.02`

```python
embedding_table = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
```

### Why

Embedding is **lookup, not matmul**. No summation over `d_in` happens.
There's nothing to blow up or vanish. We just need the rows to start
small and distinguishable.

`0.02` is the GPT-2 convention — empirically works well, no deeper reason.

## Rule 3 — Output projection: often smaller still

Some implementations scale the final projection weight by
`0.02 / √(2·L)` (L = number of transformer layers). This dampens the
residual signal's growth through a deep stack. Convention from
GPT-2 / LLaMA.

## Cheat sheet

| weight used as... | init scale | reason |
|---|---|---|
| linear (matmul) `y = x @ W` | `1/√fan_in` | keep activation variance = 1 |
| linear before ReLU | `√(2/fan_in)` | compensate for ReLU killing half |
| embedding (lookup) | `0.02` (constant) | no matmul → no variance issue |
| output projection | `0.02 / √(2·L)` | stabilize deep residual stacks |
| bias | 0 | biases don't need variance |

## Deeper principle

Init is about **signal propagation at step 0**. If activations explode or
vanish through layers before training starts, the first few gradient
updates will be garbage and training may never recover. The various
`1/√n` factors are all ways of saying "scale weights so activations
maintain ~unit variance through the whole forward pass."

Once training starts, weights adapt — the exact init matters less. But
bad init can kill a run before it ever gets going.
