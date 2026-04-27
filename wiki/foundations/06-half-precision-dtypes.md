# 06 — Half precision and below: fp16, bf16, fp8, fp4

## The three axes of every float format

Every new dtype picks a point in this tradeoff:

1. **Total bits** → memory, bandwidth, speed
2. **Exponent bits** → range (overflow/underflow limits)
3. **Mantissa bits** → precision (how finely nearby numbers differ)

Fewer bits = cheaper, but something has to give.

## fp32 — the baseline (IEEE 754, 1985)

```
1 sign | 8 exp | 23 mant    → 4 bytes, range ~10±³⁸, ~7 decimal digits
```

Safe, universal, expensive. 70B params × 4 B = 280 GB.

## fp16 — first half-precision (used in ML ~2017)

```
1 sign | 5 exp | 10 mant    → 2 bytes, range ~6×10±⁴, ~3 decimal digits
```

**Problem**: narrow range (5 exp bits). Gradients < ~6×10⁻⁵ **underflow to zero**.

**Hack: loss scaling.** Multiply loss by a constant (e.g. 2¹⁵) so gradients are
large enough to represent in fp16, then divide gradients by that constant in fp32
before the optimizer step.

## bf16 — the ML-native format (Google Brain ~2018)

```
1 sign | 8 exp | 7 mant     → 2 bytes, range ~10±³⁸, ~2 decimal digits
```

**Insight**: in ML, range > precision. Gradients span many orders of magnitude;
extra decimal digits rarely matter.

Keep fp32's 8 exponent bits, trade precision. **Drop-in replacement for fp32 in
most training. No loss scaling needed.** This is what you actually use today —
Llama, GPT, Claude, DeepSeek all trained in bf16.

**Downside**: only ~2 decimal digits of precision. Reductions (loss sum, softmax,
LayerNorm) still need fp32 accumulators or errors drift.

## fp8 — current frontier (H100+, 2022+)

Two variants, 1 byte each:

```
E4M3: 1 sign | 4 exp | 3 mant   → narrower range, more precision (weights, activations)
E5M2: 1 sign | 5 exp | 2 mant   → wider range, less precision (gradients)
```

Range is so tight you need **per-tensor (or per-block) scaling factors** —
every tensor carries its own exponent shift. DeepSeek-V3 (2025) trained the
whole model in fp8 end-to-end — most efficient frontier-scale training so far.

## fp4 / nvfp4 / mxfp4 — experimental (Blackwell, 2024+)

4 bits per number. 8× smaller than fp32. Needs fine-grained block scaling
(e.g. MX formats: 32-element blocks, shared scale factor).

Mostly **inference** today (quantize a trained model). Training from scratch
in fp4 is an active research area, not yet standard.

## What's used for what (as of 2026)

| context | format | why |
|---|---|---|
| **Training (default)** | **bf16** + fp32 master weights | standard; no loss scaling; widely supported |
| Training (bleeding edge) | fp8 (E4M3/E5M2) | DeepSeek-V3 proved it works at scale |
| Inference / serving | bf16 or fp16, increasingly fp8/int8/int4 | latency + memory dominated |
| Inference (extreme) | fp4, int4 | 8× smaller weights, some quality cost |
| Reductions (loss, softmax) | **fp32 accumulators** | precision critical here |

## The real picture: mixed precision

You don't pick one format for everything. A typical training step:

1. Master weights → stored in **fp32**.
2. Cast down to **bf16** for the forward pass.
3. Activations and matmul inputs/outputs → **bf16**.
4. Matmul accumulators → **fp32** (hardware does this inside tensor cores).
5. Loss and reductions → **fp32**.
6. Gradients → **bf16**.
7. Optimizer (AdamW) → reads **fp32** master weights, updates them in **fp32**.

PyTorch's `torch.autocast` automates picking per-op.

## Issues that come with lower precision

- **Underflow / overflow** — small gradients → 0 (fp16), big values → ∞. Worse as
  bits drop. Fixed by bf16's range, reappears in fp8/fp4.
- **Precision loss in sums** — `1e-3 + 1e-3` ten thousand times ≠ 10 in bf16.
  Running sum dominates, small adds get swallowed. Fix: fp32 accumulator.
- **Non-determinism** — lower precision amplifies tiny differences across GPUs.
  Reduces bit-exact reproducibility.
- **Hardware lock-in** — fp8 requires H100+; fp4 requires B200+. You can't just
  "use fp8" on an A100; it silently upcasts or errors.

## Key takeaways

- **fp32** = default, safe, expensive. Still used for master weights & accumulators.
- **bf16** = what you actually train in. Same range as fp32, half the memory.
- **fp8** = frontier labs, needs scaling tricks, proven by DeepSeek-V3.
- **fp4 / int4** = mostly inference, still experimental for training.
- Training is **mixed precision**: different dtypes for different parts of the step.
- The cost of low precision: range issues, precision loss in reductions,
  non-determinism, hardware constraints.

## Questions I still have

-
