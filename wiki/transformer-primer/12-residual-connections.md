# Residual connections (step 12)

## What they are in simple English

Instead of replacing the input at each layer, **add the input back to
the layer's output**:

```
before:  x → layer → out
after:   x → layer → layer(x);  then  out = x + layer(x)
```

Each block now outputs `x + sublayer(x)` instead of just `sublayer(x)`.

## What problem does it solve — vanishing gradients

Deep networks (L=12, 80, 100+ layers) used to be **unstable to train**.
As gradients flowed backward through many layers, they tended to either:
- **vanish** (shrink toward 0) — most common, more harmful
- **explode** (grow toward ∞)

### Concrete example — without residuals

5 layers, each with `∂layer/∂x = 0.1`:
```
gradient reaching x = 0.1 × 0.1 × 0.1 × 0.1 × 0.1 = 0.00001
```
Vanished. The early layer can't learn.

### With residuals

Same 5 layers, each `∂/∂x = 1 + 0.1 = 1.1`:
```
gradient reaching x = 1.1 × 1.1 × 1.1 × 1.1 × 1.1 ≈ 1.61
```
Survives fine.

## Why the `+1` is the whole trick

For `out = x + layer(x)`:
```
∂out/∂x = 1 + ∂layer/∂x
         └┬┘   └────┬────┘
      direct path  through layer
```

The `1` means gradient can flow **straight through** the layer,
unchanged. Even if `∂layer/∂x` is tiny or noisy, the `1` guarantees
the gradient reaches the earlier layer.

A "gradient highway" alongside every layer.

## Conceptual shift — layers learn corrections, not replacements

Without residuals, each layer must produce the full answer from
scratch. With residuals, each layer learns:

> "what should I **add** to x to make it slightly better?"

If the layer has nothing useful to contribute, it can learn to output
~0 → the identity is passed through. Much easier than having to
re-learn the identity from scratch.

## Explosion isn't fixed by residuals

Residuals help vanishing but can **make explosion slightly worse**.

5 layers with `∂layer/∂x = 2`:
- Without residuals: `2⁵ = 32`
- With residuals: `(1+2)⁵ = 243` ← worse

The fix for explosion comes from **normalization** (LayerNorm / RMSNorm
— step 13). Norm keeps activation magnitudes bounded so the exploding
case rarely happens in practice.

> **Residuals + Norm together** = the two pillars that let deep
> transformers train reliably.

## Where they go in the transformer block

Residuals wrap **each sub-layer** (attention and MLP) separately:

```
x → x + attention(x) → x + mlp(x)
     └─────┬──────┘   └────┬────┘
       first residual   second residual
```

In code:
```python
x = x + self.attention(x)
x = x + self.mlp(x)
```

Two residuals per block, one per sub-layer.
