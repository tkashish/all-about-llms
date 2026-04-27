# Full LM: stack of blocks + output head (step 16)

## What a full transformer LM looks like

```
tokens (B, T) integers
  │
  ↓  embedding lookup               (B, T, D)
  │
  ↓  block 1 ─┐
  │          │
  ↓  block 2 │
  │          │  stack of L blocks
  ...        │
  │          │
  ↓  block L ┘
  │
  ↓  final RMSNorm                  (B, T, D)
  │
  ↓  output projection @ W_out      (B, T, vocab_size)   ← logits

```

Shape at the end: one score per possible next-token ID, for every
position in every sequence.

## Why a final norm

The last block's output can have drifted magnitudes. A final
`RMSNorm` right before the output projection stabilizes the logits'
scale. Standard in all modern LMs.

## Output projection — producing logits

```python
logits = x @ W_out       # (B, T, D) @ (D, vocab_size) → (B, T, vocab_size)
```

For every token position, produce **vocab_size** logits. One scalar per
possible next-token ID. Positive = likely, negative = unlikely.

No softmax yet — the loss function (step 17) applies softmax inside
cross-entropy for numerical stability.

## Weight tying (optional)

`embedding_table` has shape `(vocab_size, D)`.
`W_out` has shape `(D, vocab_size)`. They're transposes.

Many LMs **share** these weights — use `embedding_table.T` as `W_out`
instead of allocating a new matrix. Saves `vocab_size · D` parameters.

Example: 50k vocab × D=768 ≈ 38M parameters saved. Significant for
small models; less impactful for large ones.

GPT-2 tied; Llama-3-70B doesn't. Either works.

## Python-list-of-blocks trap

```python
self.blocks = [TransformerBlock(...) for _ in range(L)]   # ❌
```

`nn.Module` does **not** recurse into plain Python lists to find
submodules. `.to(device)` won't move them; `.parameters()` won't find
them.

Fix:
```python
self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(L)])
```

Same iteration syntax (`for b in self.blocks`), but PyTorch now knows
about the children.

## In-place residual trap

```python
x += self.attention(self.norm1(x))   # ❌ in-place
```

`+=` mutates `x` in memory. But autograd saved references to the old
`x` for the backward pass. Overwriting it can either:
- raise `RuntimeError: modified by an inplace operation`, or
- silently produce wrong gradients.

Fix:
```python
x = x + self.attention(self.norm1(x))   # ✅ new tensor
```

## Step 16 complete — our scratch LM

```
Model
├── TokenEmbedding (vocab_size, D)
├── nn.ModuleList[TransformerBlock × L]
└── OutputHead (final RMSNorm + tied W_out)
```

Output shape `(B, T, vocab_size)` = logits. Ready for training
(cross-entropy loss over next-token prediction).
