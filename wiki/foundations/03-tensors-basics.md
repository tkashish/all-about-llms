# 03 — Tensors: the three things that define one

## What a tensor is

**An n-dimensional array of numbers.** Generalization of scalar → vector → matrix:

```
0-d (scalar):   5
1-d (vector):   [1, 2, 3, 4]
2-d (matrix):   [[1, 2, 3], [4, 5, 6]]
3-d:            [[[1,2],[3,4]], [[5,6],[7,8]]]
```

The number of dimensions = **rank** = **number of axes** = `x.ndim`.

## Three things define every tensor

### 1. Shape

Tuple of sizes per axis. `x.shape`, `x.numel()` = product of the shape.

```python
x = torch.zeros(2, 3)
x.shape    # torch.Size([2, 3])
x.ndim     # 2
x.numel()  # 6
```

In transformers you'll see `(B, T, D)` = (batch, sequence length, feature dim).
E.g. `(32, 1024, 768)` holds ~25M numbers.

### 2. Dtype (data type)

Single dtype for **all** elements. Sets bytes per element:

| dtype | bytes |
|---|---|
| fp32 (float32) | 4 |
| fp16 / bf16 | 2 |
| fp8 | 1 |
| int64 | 8 |

```python
x = torch.zeros(2, 3)                       # default float32
y = torch.zeros(2, 3, dtype=torch.bfloat16)
```

**Memory** = `numel × bytes_per_element`.

Example: `(32, 1024, 768)` fp32 tensor →
```
32 × 1024 × 768 × 4 = 100,663,296 bytes ≈ 96 MB
```
That's **one** activation tensor. Imagine many of those × many layers × training
(need grads and optimizer state) → quickly into the hundreds of GB.

### 3. Device

Where the tensor physically lives: CPU RAM or a specific GPU.

```python
x = torch.zeros(2, 3)                       # CPU
y = x.to("cuda")                            # copy to GPU
z = torch.zeros(2, 3, device="cuda")        # born on GPU
```

Operations happen on the device of the tensors. Two tensors must share a device
to operate together.

## Tensors are NOT Python lists

`a + b` is **not** a Python loop. It's a single call into optimized C++/CUDA that
runs in parallel across all elements (SIMD / thousands of GPU threads). That's
why ML works.

## Key takeaways

- Three things: **shape, dtype, device**.
- Memory = `numel × bytes(dtype)`.
- Ops are vectorized: fast on tensors, glacial on Python lists.

## Questions I still have

-
