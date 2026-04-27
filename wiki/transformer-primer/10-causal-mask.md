# Causal masking (step 10)

## What it is in simple English

A rule: token `i` can only attend to tokens `0 … i`. Never to `i+1,
i+2, …`. Each token's attention is blocked from looking at the future.

## What problem does it solve

Language models predict the **next** token given the previous ones.
During training, the model processes a full sentence in one forward
pass — but when computing token `i`'s representation, it must not see
tokens `i+1, i+2, ...`. Otherwise it would "cheat" by copying the
answer during training, and fail at inference time when the future
genuinely doesn't exist yet.

So: for query Q_i, we must ignore K_j and V_j for all `j > i`.

## Where in the pipeline to block

Pipeline: compute Q/K/V → **scores = Q @ K.T** → scale → **softmax** →
weights @ V.

Rule: **block before softmax**, not after.

- If you zero weights *after* softmax, the remaining weights no longer
  sum to 1 → broken attention.
- If you zero *scores* before softmax: `exp(0) = 1` → future positions
  still get non-zero weight. Doesn't actually block.

Correct move: set blocked scores to `-∞`. `exp(-∞) = 0` → blocked
weights become 0; allowed weights still sum to 1. Clean.

## The mask matrix

Shape `(T, T)`. Lower triangle (incl. diagonal) = 0. Strict upper
triangle = -∞.

Build once in `__init__` with `max_seq_len`, then slice `[:T, :T]` in
`forward`:

```python
mask = torch.zeros(max_seq_len, max_seq_len)
mask.masked_fill_(
    torch.triu(torch.ones_like(mask), diagonal=1).bool(),
    float('-inf'),
)
self.register_buffer('mask', mask)
```

## Applying the mask

```python
scaled_score = scaled_score + self.mask[:t, :t]
```

- Addition broadcasts `(T, T)` → `(B, H, T, T)` for free (trailing axes
  match; leading dims auto-expanded).
- Boolean indexing does **not** broadcast — one of the footguns hit
  during implementation.

## `register_buffer` gotcha

`model.to('mps')` moves **`nn.Parameters`** and **registered buffers**,
but NOT plain `self.mask = tensor` attributes.

Rule of thumb:
- Learnable tensor → `nn.Parameter`
- Non-learnable tensor that must ride along with `.to(device)` /
  `state_dict()` → `register_buffer(...)`
- Throwaway local → regular variable

## Verified (T=5, unit-scale input)

```
row 0: [1.00,  0,    0,    0,    0   ]   ← attends only to itself
row 1: [0.07,  0.93, 0,    0,    0   ]
row 2: [0.40,  0.30, 0.30, 0,    0   ]
row 3: [...,   ...,  ...,  0.89, 0   ]
row 4: [...,   ...,  ...,  ...,  ... ]   ← attends to all
```

Triangular zeros in the upper-right. Rows sum to 1. Causal mask
working.
