# 10 — einops: rearrange, reduce, einsum

Why: pure PyTorch tensor ops (`view`, `permute`, `transpose`, `sum(dim=...)`)
rely on remembering which axis is which. einops names the axes, making code
self-documenting and less bug-prone. Same underlying operations (views where
possible, copies only when needed) — just better syntax.

## The three functions

1. **`rearrange`** — move / merge / split axes. No numeric change.
2. **`reduce`** — sum/mean/max over named axes.
3. **`einsum`** — multiply tensors and reduce along matching axis names.

## 1. `rearrange`

```python
from einops import rearrange

# transpose
y = rearrange(x, 'a b -> b a')       # swap axes

# merge (flatten) axes
y = rearrange(x, 'b h w c -> b (h w c)')    # (B,H,W,C) → (B, H·W·C)

# split (unflatten) an axis — provide the size of one of the new axes
y = rearrange(x, 'b (h w) c -> b h w c', h=H)
```

Behavior: returns a view if strides can describe it, otherwise copies via
`.contiguous()` internally. Same rules as [04-memory-layout](./04-memory-layout.md).

### Parentheses — merge and split axes

- On the **input** side: parentheses mean "this single axis is really
  multiple axes multiplied together". Split them apart.
- On the **output** side: parentheses mean "merge these axes into one".
- You must pass the missing size as a kwarg when einops can't infer it.

```python
# split the last dim 8 into (head, hidden1) with head=2 → hidden1 inferred as 4
x = torch.ones(3, 8)
rearrange(x, '... (head hidden1) -> ... head hidden1', head=2)
# shape (3, 8) → (3, 2, 4)
```

This is **the** transformer multi-head split: take a hidden vector of size
`D` and view it as `head × D/head`.

### `...` (ellipsis) — "any leading axes"

`...` matches zero or more leading axes whose exact count you don't care about.
Great for writing operations that work on any-rank tensor.

```python
# works for (D,), (B, D), (B, T, D), etc.
rearrange(x, '... d -> ... d')
```

## 2. `reduce`

General shape: `reduce(tensor, 'input -> output', 'op')` where `op` is
`'sum' | 'mean' | 'max' | 'min' | 'prod'`.

**Rule**: any axis in the input but missing from the output is reduced.

Example: `x` shape `(2, 3)`, axes `r c`.

```python
reduce(x, 'r c -> c', 'sum')   # column sums, shape (3,)
reduce(x, 'r c -> r', 'sum')   # row sums,    shape (2,)
reduce(x, 'r c -> ',  'sum')   # grand total, scalar
```

ML example: activation shape `(B, T, D)`:

```python
reduce(x, 'b t d -> b d', 'mean')     # mean-pool over tokens per example
reduce(x, 'b t d -> d',   'mean')     # mean per feature across batch+tokens
```

## 3. `einsum`

**The single rule**: for each output position, multiply the corresponding input
entries and **sum over every axis name that doesn't appear in the output**.

Formula:
```
out[indices] = Σ over (axes missing from output) of (product of indexed inputs)
```

### Canonical examples

| op | pattern |
|---|---|
| dot product of vectors | `'i, i -> '` |
| outer product | `'i, j -> i j'` |
| matmul (2D) | `'i j, j k -> i k'` |
| batched matmul | `'b i j, b j k -> b i k'` |
| elementwise multiply | `'i j, i j -> i j'` |

### Axis roles

| axis appears in | role |
|---|---|
| both inputs & output | "zipped" — computed independently per value (like batch dim) |
| both inputs, not output | contracted — multiplied & summed (like matmul's shared dim) |
| one input, output | passed through |
| one input, not output | that input is reduced over this axis |

### Worked non-obvious example

```python
x = torch.rand(2, 3)   # axes (i, j)
y = torch.rand(3, 2)   # axes (j, k)
einsum(x, y, 'i j, j k -> j')
```

Output axes = `j` only. Missing from output: `i`, `k`. Rule:
```
out[j] = Σᵢ Σₖ  x[i, j] · y[j, k]
       = (Σᵢ x[i, j]) · (Σₖ y[j, k])       # i,k independent here
       = (column-j sum of x) · (row-j sum of y)
```
Perfectly valid einsum, but **not** a matmul. Matmul would be `'i j, j k -> i k'`.

## Why use einops instead of raw PyTorch

1. **Self-documenting**: `'b h w c -> b (h w c)'` says the intent.
2. **Shape-checked**: mismatches raise clear errors instead of silently producing
   wrong results.
3. **Uniform syntax**: one mental model for reshape, permute, merge, split, matmul.

Same ops underneath — just harder to misuse.

## Key takeaways

- `rearrange` = move/merge/split (no numerical change).
- `reduce` = drop axes via sum/mean/max/…; axes missing from output are reduced.
- `einsum` = multiply + reduce; one rule covers dot, outer, matmul, batched matmul.
- **Rule of einsum**: axes missing from output are summed.
- einops is syntactic sugar on top of PyTorch — same memory behavior.

## Questions I still have

-
