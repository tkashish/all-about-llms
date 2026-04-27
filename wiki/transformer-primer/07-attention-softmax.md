# Attention phase 2, step 2 — Softmax

## What is softmax in simple English?

Softmax takes a row of numbers and squashes them into **proper weights**
— all between 0 and 1, all summing to 1. Like turning raw votes into
percentages.

## What problem does it solve?

Raw scores from `Q · K` can be anything — positive, negative, huge,
tiny. You can't use those as weights in a weighted sum because:
- Some are negative (can't have negative weight in an average).
- They don't sum to anything consistent.
- Big scores should dominate smoothly, not linearly.

Softmax fixes all of this in one step.

## The formula

For a row of numbers `[s_1, s_2, ..., s_T]`:

```
softmax(s_i) = exp(s_i) / Σ_j exp(s_j)
```

Two steps:
1. Apply `exp` to every score → all positive, big scores get
   exponentially bigger.
2. Divide each by the total → they sum to 1.

## Concrete example

Scores: `[0.74, 0.00, -0.3, 0.90]`

After exp: `[2.10, 1.00, 0.74, 2.46]`, sum = 6.30

After dividing: `[0.333, 0.159, 0.117, 0.391]`. All ≥ 0, sum to 1.

## Why exp specifically?

- Makes negative scores positive (since `e^x > 0` for all real x).
- Big scores win disproportionately — doubling score ~triples weight.
  This is the behavior attention wants: confident matches dominate.

## Applied to the attention matrix

Scores matrix shape: `(T, T)`.
- rows = queries (one per token as a query)
- columns = keys (one per token as a key)

For token i, its row `scores[i, :]` is its scores against every key.
Softmax normalizes across that row.

```
weights = torch.softmax(scores, dim=-1)
```

Result: same shape `(T, T)`, every row sums to 1.

## `dim=1` vs `dim=-1` — which to use?

For `(T, T)`, both do the same thing:
```python
torch.softmax(scores, dim=1)   # softmax across columns
torch.softmax(scores, dim=-1)  # last axis = same axis here
```

**Prefer `dim=-1`** because it stays correct as tensor shapes grow:

| shape | axis of keys | `dim=1`? | `dim=-1`? |
|---|---|---|---|
| `(T, T)` | axis 1 | ✓ | ✓ |
| `(B, T, T)` batched | axis 2 | ✗ | ✓ |
| `(B, H, T, T)` multi-head | axis 3 | ✗ | ✓ |

`dim=-1` always means "the last axis" — which is always the key/softmax
axis. Future-proof.

## What's next

Now every row of the score matrix is a probability distribution over
keys. Step 3 uses these weights to take a weighted sum of V vectors,
giving each token its new context-aware output.
