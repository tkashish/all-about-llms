# 05 — Range vs precision (slow build)

## Two different things

- **Range** = how big or small a number can get.
- **Precision** = how finely you can tell two numbers apart.

Separate ideas. Don't mix them up.

## Ruler analogy

- Ruler A: 1 m long, marks every 1 cm.
  - Range: 1 to 100 cm.
  - Precision: 1 cm.
- Ruler B: 1 km long, marks every 10 cm.
  - Range: 10 to 100,000 cm (much wider).
  - Precision: 10 cm (coarser).

More range ↔ less precision. Can't have both for free in a fixed size.

## The two bit-groups

A float has:
- **exponent bits** → range
- **mantissa bits** → precision

More exponent bits = wider range. More mantissa bits = finer precision.

## The three formats

| format | exponent | mantissa |
|---|---|---|
| fp32 | 8 | 23 |
| fp16 | 5 | 10 |
| bf16 | 8 | 7 |

- fp16 cut **both** vs fp32 → less range AND less precision.
- bf16 cut **only the mantissa** → same range as fp32, less precision.

## Hitting the range limit

- Number bigger than max → **overflow** → becomes ∞.
- Number smaller than min → **underflow** → becomes 0.
- Either way, the number is lost.

## Hitting the precision limit

A float format can only represent a **finite set** of values. Like a ruler: if
the ruler is marked every 1 cm, you can't show 1.3 cm — it rounds to 1 or 2.

Floats do the same. If two real numbers fall between the same two "marks", they
get snapped to the same stored value — their difference is lost.

## The twist: marks are NOT evenly spaced

A float is stored as `mantissa × 2^exponent`. One "step" of the mantissa means
a much bigger absolute gap when the exponent is large:

```
Near magnitude 10⁰: 1.234 → 1.235   gap = 0.001
Near magnitude 10³: 1234  → 1235    gap = 1
Near magnitude 10⁶: 1234000 → 1235000  gap = 1000
```

Same fractional step, exploding absolute gap. This is **relative precision**:
you always get ~the same number of significant digits, but the absolute gap grows
with magnitude.

Consequence:
- Near 1, fp32 can tell 1.0 from 1.0000001.
- Near 1,000,000, fp32 **cannot** tell 1,000,000 from 1,000,000.01.

## The mantissa concretely

Every normalized binary number is written as `1.xxxx × 2ⁿ`. The leading `1.` is
implied (not stored, saves a bit) — so the mantissa bits are just the `xxxx`.

**6.75** = `1.1011 × 2²`  → mantissa bits (after the 1.) = `1011`

Fits in 4 bits. Both fp32 (23 bits) and bf16 (7 bits) have more than enough room,
so both store 6.75 **exactly**.

**1.1** = `1.00011001100110011… × 2⁰` — *repeating forever* (like 1/3 in decimal).

- fp32 keeps 23 bits: close, tiny error.
- bf16 keeps 7 bits: same number, much more chopped off.

Verified in PyTorch:
```
You asked for:  1.1
fp32 stores:    1.10000002384185791016     (error ~2e-8)
bf16 stores:    1.10156250000000000000     (error ~1.6e-3)
```
bf16 is ~100,000× less accurate here.

## Range numbers

| format | max positive | smallest positive |
|---|---|---|
| fp32 | ~3.4 × 10³⁸ | ~1.2 × 10⁻³⁸ |
| bf16 | ~3.4 × 10³⁸ | ~1.2 × 10⁻³⁸ |
| fp16 | ~6.5 × 10⁴ | ~6.1 × 10⁻⁵ |

fp32 and bf16 → **same range** (both have 8 exponent bits).
fp16 → much narrower range (only 5 exponent bits).

This is why ML prefers bf16 over fp16: keep the range of fp32, lose only precision.

## Key takeaways

- Range ≠ precision. Exponent bits ≠ mantissa bits.
- Overflow/underflow = range failure. Rounding = precision failure.
- Float marks aren't evenly spaced — relative precision stays the same, absolute
  gap grows with magnitude.
- bf16 = fp32's range with coarser precision → ML's training default.

## Questions I still have

-
