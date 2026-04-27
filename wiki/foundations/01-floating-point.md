# 01 — Floating Point & fp32

## Why this matters

Every number in a neural network — weights, activations, gradients, optimizer state —
is stored as a floating-point number. The **choice of format** determines memory per
number and hardware speed. fp32 is the default; half-precision formats are the
interesting tradeoffs. Before we can talk about those, we need to know what fp32 is.

## What "floating point" means

Scientific notation, generalized. A number is stored as:

```
value = sign × mantissa × base^exponent
```

In decimal, `123.45 = 1.2345 × 10²`. The decimal point "floats" to wherever the
exponent puts it. Computers use base 2 instead of base 10:

```
5.5 (decimal) = 101.1 (binary) = 1.011 × 2²
```

Two parts: the digits (mantissa) and the exponent.

## fp32 bit layout (IEEE 754 single precision, 1985)

```
 1 bit  │    8 bits   │      23 bits
 sign   │  exponent   │   mantissa (fraction)
```

- **sign**: 0 = positive, 1 = negative.
- **exponent**: unsigned 0–255, subtract bias of 127 → actual exponent −126 to +127.
  Gives range ~10⁻³⁸ to ~10³⁸.
- **mantissa**: 23 bits after an implied leading `1.` (normalized form), so you
  effectively get 24 bits of precision ≈ 7 decimal digits.

## Worked example: encode 6.75

1. Convert to binary: `6.75 = 110.11₂` (see [02-binary-fractions](./02-binary-fractions.md))
2. Normalize: `110.11 = 1.1011 × 2²`
3. Fill fields:
   - sign = 0
   - exponent = 2 + 127 = 129 = `10000001`
   - mantissa = bits after the leading 1 → `1011`, padded to 23 bits:
     `10110000000000000000000`

Full fp32 bits: `0 10000001 10110000000000000000000`

## Dtype comparison

| format | total bits | exponent | mantissa | range | precision |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~10⁻³⁸ to 10³⁸ | ~7 decimal digits |
| fp16 | 16 | 5 | 10 | ~6×10⁻⁵ to 6×10⁴ | ~3 decimal digits |
| bf16 | 16 | 8 | 7 | same as fp32 | ~2 decimal digits |

fp16 saves memory but has narrow range → overflow risk during training.
bf16 keeps fp32's range but sacrifices precision → the ML-favored 16-bit format.

## Why fp32 is the default

1. **Range**: ~10⁻³⁸ to 10³⁸ covers anything you'll hit in training without over/underflow.
2. **Precision**: 7 digits means rounding errors accumulated over millions of ops stay tiny.
3. **Hardware**: every CPU and GPU supports fp32 natively.

## The cost

4 bytes per number. For a 70B-param model:
```
70e9 × 4 bytes = 280 GB   ← just for parameters
```
Training also needs gradients (+280 GB) and AdamW optimizer state (+560 GB). That's
why half-precision exists.

## Key takeaways

- fp32 = 1 sign + 8 exponent + 23 mantissa bits = 4 bytes / number.
- Range and precision are separate — controlled by exponent and mantissa respectively.
- The "safe default": no overflow risk, but expensive at scale.

## Questions I still have

-
