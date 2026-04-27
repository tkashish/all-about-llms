# 07 — Why no bf24?

Natural question: if bf16 is useful and fp32 is safe, why not an in-between
format like `bf24` (8 exponent + 15 mantissa + 1 sign)?

It doesn't exist, not because of math but because of three practical reasons.

## 1. Hardware word sizes

Computers are built around power-of-2 sizes: 8, 16, 32, 64, 128 bits. Memory
buses, registers, cache lines, SIMD lanes — all assume these sizes.

A 24-bit float doesn't fit any standard word. Packing them tightly would force
every load/store to do extra shifting/masking. Wasted silicon and time.

Cheap: 8, 16, 32, 64 bits.
Expensive: anything else.

## 2. The whole point is halving

Each generation of dtype doubles throughput and halves memory:

| step | memory | speedup |
|---|---|---|
| fp32 → bf16 | 2× smaller | 2× faster |
| bf16 → fp8  | 2× smaller (4× vs fp32) | 2× faster |
| fp8 → fp4   | 2× smaller (8× vs fp32) | 2× faster |

bf24 would be only 1.33× smaller than fp32. Not enough savings to justify
new silicon designs, new tensor core variants, new compiler/library support.

## 3. bf16 is empirically good enough

Models trained in bf16 (with fp32 master weights and fp32 accumulators) converge
nearly identically to fp32-only training. ML is inherently noisy — the precision
loss from bf16 is absorbed.

So there's no demand for a safer middle ground. When labs want more savings,
they skip straight to fp8 (with scaling factors) or fp4 (with block formats).

## Key takeaways

- No math reason it can't exist.
- Hardware hates non-power-of-2 sizes.
- Only 1.33× memory savings — not worth the complexity.
- bf16 works well enough that nobody needs bf24.
