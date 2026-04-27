# 08 — Views and memory: shared-buffer gotchas

Views from slicing/transposing share memory with the original tensor. That's the
whole no-copy speedup — but it creates three surprises you need to know about.

## 1. Reference counting keeps the original alive

```python
x = torch.zeros(1_000_000)     # buffer refcount = 1
y = x[:10]                     # view → refcount = 2
del x                          # refcount = 1 (y still references)
# buffer still alive, y still works
del y                          # refcount = 0 → now freed
```

Python/PyTorch won't GC a memory buffer while any tensor still points at it.

## 2. Writes are shared both ways

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
y = x[:, 1]                    # view of column 1
y[0] = 99
# x is now:
# [[1, 99, 3],
#  [4,  5, 6]]
```

Both `x` and `y` point at the same bytes. Mutating one mutates the other.

If you need independence:

```python
y = x[:, 1].clone()            # forces a real copy
```

## 3. Tiny views of huge tensors leak memory

This is the classic PyTorch memory leak:

```python
huge = torch.zeros(1_000_000_000)   # 4 GB
small = huge[:10]                    # 10-element view
del huge
# 4 GB is STILL in memory — small holds the whole buffer alive
```

A 10-element tensor is keeping 4 GB alive because it's a view into that buffer.

Fix — break the view by cloning:

```python
small = huge[:10].clone()            # now independent
del huge                             # 4 GB freed
```

Common in training loops: if you keep references to slices of activations,
batches, or model outputs across iterations, the underlying tensors never get
freed. Always `.clone()` or `.detach().clone()` before storing.

## How to tell: `.data_ptr()`

```python
x = torch.arange(6).reshape(2, 3)
y = x.T                        # view
x.data_ptr() == y.data_ptr()   # True  ← same buffer

z = x.clone()
x.data_ptr() == z.data_ptr()   # False ← independent buffer
```

Two tensors share storage iff they have the same `data_ptr()`.

## Key takeaways

- Views share memory → the original is kept alive by references.
- Writing to a view writes to the original (and vice versa).
- Tiny views can pin massive buffers in memory → `.clone()` to break the link.
- `.data_ptr()` tells you whether two tensors share storage.
