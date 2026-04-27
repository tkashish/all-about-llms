# 02 — Binary fractions: how 6.75 = 110.11₂

## The idea

A decimal number is split by the decimal point:
- Left of point → powers of 10 going **up** (1, 10, 100, …)
- Right of point → powers of 10 going **down** (1/10, 1/100, …)

Binary is the same, but with powers of 2.

## The integer side: 6 → 110

```
position:   4   2   1
power:     2²  2¹  2⁰
bit:        1   1   0
value:      4 + 2 + 0  =  6
```

So `6 = 110₂`.

## The fractional side: 0.75 → 0.11

After the binary point, powers of 2 go **negative**:

```
position:   1/2   1/4   1/8   1/16
power:      2⁻¹   2⁻²   2⁻³   2⁻⁴
bit:         1     1     0     0
value:     0.5 + 0.25 + 0 + 0  =  0.75
```

So `0.75 = 0.11₂`.

## Putting them together

```
 6 . 75
110. 11
```

`6.75 = 110.11₂` ✓

## Algorithm for arbitrary fractions

**Repeatedly multiply by 2, record the integer part, keep the fraction:**

Example with 0.75:
```
0.75 × 2 = 1.5    →  bit = 1, remainder = 0.5
0.5  × 2 = 1.0    →  bit = 1, remainder = 0   (done)
```
Top-to-bottom: `0.11₂` ✓

## What happens with 0.1

```
0.1 × 2 = 0.2  →  0
0.2 × 2 = 0.4  →  0
0.4 × 2 = 0.8  →  0
0.8 × 2 = 1.6  →  1, remainder 0.6
0.6 × 2 = 1.2  →  1, remainder 0.2
...repeats forever
```

`0.1 = 0.0001100110011…₂` — a repeating fraction.

This is why `0.1 + 0.2 != 0.3` in every IEEE-754 language. 0.1 simply **cannot be
represented exactly** in any finite-precision binary float.

## Key takeaways

- Binary fractions work exactly like decimal fractions, but with halves/quarters/eighths
  instead of tenths/hundredths.
- A decimal is exact in binary iff it's a finite sum of negative powers of 2
  (i.e., `k / 2ⁿ` for some integers).
- This is **the** source of floating-point weirdness in ML and elsewhere.

## Questions I still have

-
