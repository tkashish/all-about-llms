# Normalization (step 13)

## What it is in simple English

Rescale each token's vector so its values have a **consistent
magnitude** (roughly unit variance). Think of it as resetting the
volume knob to a normal level at each layer.

## What problem does it solve

As data flows through many layers (residuals keep adding), magnitudes
drift — some features get huge, others tiny. Two problems follow:

1. **Training instability** — gradients explode or vanish with the
   magnitudes.
2. **Slow learning** — later layers can't adapt when their input scale
   keeps shifting as earlier layers train.

Norm **resets the scale** every layer so every layer sees inputs at
the same magnitude, regardless of what earlier layers have done.

## LayerNorm — the original

For each token's vector `x` of length `D`:

```
μ  = mean(x)              # scalar
σ  = std(x)               # scalar
x' = (x - μ) / σ          # mean 0, variance 1
out = γ · x' + β          # γ, β are learned, shape (D,)
```

Normalization is **per token, independently**. No mixing across the
batch or sequence dim.

## RMSNorm — what modern LLMs use

Same idea, simpler:

```
RMS = sqrt(mean(x**2))    # scalar
x'  = x / RMS             # magnitude ≈ 1
out = γ · x'              # γ learned, shape (D,); no β
```

Two things removed vs LayerNorm:
- No mean subtraction (`- μ`)
- No bias (`β`)

Empirically, LLMs train just as well without them and RMSNorm is
faster. Llama, Mistral, GPT-NeoX — all RMSNorm.

## Effect on magnitude — small and large collapse to the same thing

D=4 example:

```
small: x = [0.01, 0.02, -0.01, 0.03]   → RMS ≈ 0.019
       x / RMS ≈ [0.52, 1.05, -0.52, 1.58]

large: x = [10, 20, -10, 30]           → RMS ≈ 19
       x / RMS ≈ [0.52, 1.05, -0.52, 1.58]   ← same output!
```

**Magnitude is thrown away; the pattern is preserved.** That's the whole
point.

### Why throwing away magnitude is OK

1. **Pattern matters more than scale.** Which features are strong and
   how they relate to each other is the signal. Whether the overall
   vector was scaled by 0.01 or 100 rarely adds meaning.

2. **Norm sits inside the residual, not replacing it.** With pre-norm:
   ```
   x = x + attention(norm(x))
   ```
   The raw `x` still flows through the residual path with its original
   magnitude. Norm only affects the input to the sublayer.

3. **`γ` can restore magnitude if needed.** `γ` is learned per-feature;
   if the model decides feature 3 really should be large, `γ[3]` can
   learn to be 10 — reintroducing magnitude in a controlled way.

## Where norm goes — pre-norm vs post-norm

**Post-norm** (original 2017 transformer):
```
x = norm(x + attention(x))
x = norm(x + mlp(x))
```

**Pre-norm** (modern, GPT / Llama):
```
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

Pre-norm puts the norm **inside** the residual's sublayer branch, so
the `x + ...` highway stays pure. More stable for deep networks —
that's why it won.

## Full block with residuals + pre-norm + RMSNorm

```python
x = x + self.attention(self.norm1(x))
x = x + self.mlp(self.norm2(x))
```

Two separate norm instances, one per sublayer. Each has its own `γ`
(shape `(D,)`).

## Summary

- **Residuals**: gradient highway, cure vanishing, can worsen explosion.
- **Norm**: magnitude reset, cures explosion.
- **Together**: deep transformers train reliably.
- **Pre-norm + RMSNorm** is the modern recipe.
