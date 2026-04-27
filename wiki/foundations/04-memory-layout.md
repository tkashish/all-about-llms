# 04 — Memory layout: strides, views, and the no-copy trick

## 1. Memory is a line

RAM is a one-dimensional array of bytes: address 0, 1, 2, … There are no rows
or columns in hardware. Just a line.

## 2. Tensors look multi-dim, but must be flattened

A matrix:

```
[[1, 2, 3],
 [4, 5, 6]]         shape (2, 3)
```

has to be stored as a straight line. Two natural orderings:

- **row-major**: `[1, 2, 3, 4, 5, 6]`   ← PyTorch, NumPy, C
- **column-major**: `[1, 4, 2, 5, 3, 6]`   ← Fortran, MATLAB

Row-major is the default. Also called **C-contiguous**.

## 3. From (row, col) to memory index

Need a formula that finds `x[row, col]` in the flat line.

For shape `(2, 3)` row-major: memory = `[1, 2, 3, 4, 5, 6]`.

To reach row `r`, skip `r × (# columns)` elements (one full row per step).
Then add `col` to step within the row.

```
memory_index = row × 3 + col × 1
```

Verify:
- (0, 0) → 0   ✓   (value 1)
- (1, 0) → 3   ✓   (value 4)
- (1, 2) → 5   ✓   (value 6)

## 4. Strides

The multipliers `3` and `1` have a name: **strides**. One per axis.

```python
x.shape    # (2, 3)
x.stride() # (3, 1)
```

Read as: "axis 0 step = 3 memory slots; axis 1 step = 1 slot."

General formula:
```
memory_index = sum(index[i] × stride[i])
```

For shape `(d₀, d₁, …, dₙ₋₁)` C-contiguous, stride[i] = product of all dim sizes
to the right of axis i.

**Check**: shape `(4, 5)` C-contiguous → strides `(5, 1)`. ✓

## 5. The twist: shape and strides are just metadata

The actual numbers live in memory. Shape and strides are a small **description**
telling PyTorch how to read that memory.

That means: if you change the shape/strides but keep the same memory, you get a
**different-looking tensor for free — no data copied**.

## 6. Transpose, for free

```
x = [[1, 2, 3],
     [4, 5, 6]]          shape (2, 3), stride (3, 1)
memory: [1, 2, 3, 4, 5, 6]
```

Transpose:

```
[[1, 4],
 [2, 5],
 [3, 6]]                 shape (3, 2), stride (1, 3)
```

Memory is **the same bytes**. Only the metadata changed:
- shape `(2, 3) → (3, 2)` (swap)
- stride `(3, 1) → (1, 3)` (swap)

Verify using `index = r*stride[0] + c*stride[1]`:
- `(0,0) → 0*1 + 0*3 = 0` → `memory[0] = 1` ✓
- `(0,1) → 0*1 + 1*3 = 3` → `memory[3] = 4` ✓
- `(2,1) → 2*1 + 1*3 = 5` → `memory[5] = 6` ✓

`x.T` is instant in PyTorch, regardless of tensor size. Same trick generalizes
to arbitrary axis permutations via `x.permute(...)`.

## 7. Slicing, also for free

From `x = [[1,2,3],[4,5,6]]`, take the second column → want `[2, 5]`.

Three pieces of metadata do it without copying:

1. **offset** = where in memory the view starts. Element 2 is at memory[1] → offset 1.
2. **shape** = `(2,)`
3. **stride** = `(3,)` (to go from 2 to 5 in memory, skip 3 slots)

Reading:
- element 0: memory[1 + 0*3] = memory[1] = 2 ✓
- element 1: memory[1 + 1*3] = memory[4] = 5 ✓

## 8. Shared memory → gotchas

Views share the underlying buffer with the original tensor. Three consequences
(writes propagate, reference counting keeps the original alive, tiny views can
pin huge buffers) → see [08-views-memory-gotchas](./08-views-memory-gotchas.md).

## Key takeaways

- Memory is 1-D; shape + strides are **metadata** describing how to read it.
- Transpose, permute, slice → change metadata, **no data copy**.
- Views share memory with the original → refcount keeps original alive,
  writes are seen by both, tiny views of huge tensors leak memory.
- `.clone()` forces an independent copy when you need isolation or to free a buffer.

## Questions I still have

-
