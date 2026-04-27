# Attention phase 2, step 3 — Weighted sum of V + the full formula

## What is the weighted sum in simple English?

For each token, take a weighted average of all the V vectors, using
the softmax weights from step 2. The result is that token's new
context-aware vector.

## What problem does it solve?

Phase 1 gave us Q, K, V. Scores + softmax gave us weights. Now we
actually **use** those weights to pull content from the V vectors —
this is the mixing step that finally contextualizes each token.

## Formula (one token)

```
output_i = Σ_j weight_{i,j} · V_j
```

Multiply each V_j by its weight, sum them up.

## Concrete example (D=3)

Weights for token "bank":
```
deposit: 0.40, at: 0.05, the: 0.05, bank: 0.50
```
V vectors:
```
V_deposit = [0.5, 1.0, 0.0]
V_at      = [0.1, 0.1, 0.1]
V_the     = [0.1, 0.1, 0.1]
V_bank    = [0.2, 0.0, 0.9]
```
```
output_bank = 0.40·V_deposit + 0.05·V_at + 0.05·V_the + 0.50·V_bank
            = [0.31, 0.41, 0.46]
```

## Whole sequence — one matmul

```
weights: (T, T)
V:       (T, D)

output = weights @ V       shape: (T, T) × (T, D) = (T, D)
```

T weighted sums done in parallel.

## Putting phase 2 together

```
scores  = Q @ K.T                # (T, T)
weights = softmax(scores, -1)    # (T, T)
output  = weights @ V            # (T, D)
```

## The scaled part

Divide scores by `√D` before softmax:

```
Attention(Q, K, V) = softmax(Q @ K.T / √D) @ V
```

### Why divide by `√D`?

Dot products `Q·K` are sums of D products → variance scales with D →
scores grow ~`√D` in magnitude as D gets big. Large scores push softmax
into a near-one-hot distribution (one weight ≈ 1, rest ≈ 0), which
kills gradients during training.

Dividing by `√D` keeps scores roughly unit-variance regardless of D,
so softmax stays healthy and gradients flow.

Same idea as the `1/√fan_in` trick in weight init — it's a "neutralize
fan-in" move.

## The complete scaled dot-product attention, in code

```python
import math
import torch

def attention(Q, K, V):
    # Q, K, V all shape (T, D)
    D = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(D)   # (T, T)
    weights = torch.softmax(scores, dim=-1)           # (T, T)
    output  = weights @ V                             # (T, D)
    return output
```

Three lines. That's scaled dot-product attention. In practice you wrap it as an nn.Module with batching:
with batching).

## Recap of the whole attention layer

```
input vectors (T, D)
     │
     ├─ @ W_Q → Q ─┐
     ├─ @ W_K → K ─┤── scores = Q@K.T/√D
     │             │
     │             ├── weights = softmax(scores)
     │             │
     └─ @ W_V → V ─┴── output = weights @ V  →  (T, D)
```

Same input/output shape `(T, D)`. But output vectors now carry
context from all the other tokens, weighted by learned relevance.
