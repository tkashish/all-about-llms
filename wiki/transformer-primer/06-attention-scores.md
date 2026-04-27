# Attention phase 2, step 1 — Scores

## What are scores in simple English?

A number for each **pair** of tokens, saying "how much should token i
pay attention to token j?" Higher = more relevant.

## What problem does this solve?

We have Q and K vectors — we need to turn them into a single number per
pair that measures "match strength." A similarity metric.

## The answer: dot product

```
score(i, j) = Q_i · K_j
```

### Why dot product measures similarity

For two vectors:
- High positive → point in same direction → similar
- ~0 → perpendicular → unrelated
- Negative → point opposite → dissimilar

So `Q_i · K_j` = "how aligned is what token i is looking for
with what token j advertises?"

### Concrete example (D=3)

```
Q_bank    = [0.9, 0.1, 0.0]    "looking for deposit-like stuff"
K_deposit = [0.8, 0.2, 0.0]    "I'm deposit-like"
K_at      = [0.0, 0.0, 1.0]    "I'm a preposition"
```
```
score(bank, deposit) = 0.9·0.8 + 0.1·0.2 + 0.0·0.0 = 0.74
score(bank, at)      = 0.9·0.0 + 0.1·0.0 + 0.0·1.0 = 0.00
```
Big score for the relevant pair, zero for the irrelevant pair.

## Scaling to the whole sequence — one matmul

For T tokens, we have T×T scores. Stack Qs and Ks:
```
Q: shape (T, D)     — rows are Q_1, Q_2, ..., Q_T
K: shape (T, D)     — rows are K_1, K_2, ..., K_T
```
Then the entire score matrix is:
```
scores = Q @ K.T              shape: (T, D) × (D, T) = (T, T)
```

One matmul produces all T² scores at once. This is why attention is
efficient on GPUs — it's just matmuls.

### Reading the score matrix

```
          K_deposit  K_at  K_the  K_bank
Q_deposit   0.74     0.01   0.02   0.25
Q_at        0.01     0.85   0.10   0.03
Q_the       0.02     0.10   0.82   0.05
Q_bank      0.25     0.03   0.05   0.90
```
Row i, column j = how much token i attends to token j.

## Why transpose K?

Q has shape (T, D), K has shape (T, D). You need a `(D, T)` matrix on
the right to do `Q @ X → (T, T)`. So transpose K.

## What's next

Raw scores can be any size — positive, negative, huge. We need to turn
them into **proper weights** (non-negative, sum to 1) before using them
to average values. That's **softmax**, next step.
