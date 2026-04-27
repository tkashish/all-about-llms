# The 6ND rule

## Formula

```
Total training FLOPs  ≈  6 · N · D
```
- **N** = number of parameters
- **D** = number of training tokens
- **6** = constant from forward + backward math

## Where the 6 comes from

It's really `3 × 2`:

| step | FLOPs / token | notes |
|-----:|:-------------:|:------|
| forward  | `2 · N`   | matmul's "mul + add" × all params |
| backward | `4 · N`   | backward = 2× forward (two matmuls per layer) |
| **one step** | **`6 · N`** | sum |
| **D tokens** | **`6 · N · D`** | total training cost |

The original `2` comes from matmul FLOPs: 1 multiply + 1 add per element pair.
Forward uses it once per layer, backward uses it twice (weight grad + input grad).

## Example: 7B model on 2T tokens

```
FLOPs = 6 · 7e9 · 2e12 = 8.4 × 10²²
```

- 1 H100 at ~40% MFU (~4 × 10¹⁴ FLOP/s actual): ≈ 6.7 years
- 1000 H100s at 40% MFU: ≈ 2.5 days

## Uses

Every compute-planning question starts here:
- Budget X FLOPs → tokens you can afford: `D = X / (6N)`
- Given a finished run (e.g. Llama-3-70B on 15T): `6 · 70e9 · 15e12 ≈ 6.3 × 10²⁴`
- Scaling laws say optimal `D ≈ 20·N` → for 7B model, ~140B tokens

## Caveats

- Assumes the model is mostly matmuls (true for standard transformers at
  normal context length; 92/8/0 split between matmul / attention / elementwise).
- Ignores optimizer FLOPs (tiny).
- Under-counts at very long context where attention's T² term grows.
- Accurate to ~10% at normal sizes.
