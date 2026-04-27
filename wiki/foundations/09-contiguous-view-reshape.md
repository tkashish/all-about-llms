# 09 — Contiguous memory, `.contiguous()`, view vs reshape

Sequel to [04-memory-layout](./04-memory-layout.md). Once you understand that
strides can describe transposes/slices for free, the question is: *when can they
NOT?* That's when a copy happens.

## 1. Contiguous = elements laid out in natural row-major order, no gaps

For shape `(d₀, d₁, …, dₙ₋₁)` C-contiguous means strides =
`(d₁·d₂·…·dₙ₋₁, …, dₙ₋₁, 1)`.

```
[[1, 2, 3], [4, 5, 6]]   shape (2, 3)  stride (3, 1)  → contiguous ✓
```

## 2. Transpose breaks contiguity

```
y = x.T                  shape (3, 2)  stride (1, 3)
```

If it were contiguous, stride for `(3, 2)` would be `(2, 1)`. Our stride is `(1, 3)`.

Logical order when reading row-by-row: `1, 4, 2, 5, 3, 6`.
Memory order:                           `1, 2, 3, 4, 5, 6`.

Different → **non-contiguous**.

## 3. Why it matters

Many CUDA kernels and downstream ops assume contiguous memory. Hand them a
non-contiguous tensor and either:

- PyTorch calls `.contiguous()` implicitly (silent copy), or
- It errors: "expected contiguous tensor" → you must copy manually.

## 4. `.contiguous()` — forces a real copy

```python
x = torch.arange(6).reshape(2, 3)   # [[0,1,2],[3,4,5]]
x.is_contiguous()    # True
x.stride()           # (3, 1)

y = x.T              # view, shape (3, 2)
y.is_contiguous()    # False
y.stride()           # (1, 3)

z = y.contiguous()   # allocates a fresh buffer and copies
z.is_contiguous()    # True
z.stride()           # (2, 1)     ← proper contiguous strides
```

After `.contiguous()`, `z` has its own buffer laid out as `[0, 3, 1, 4, 2, 5]` —
the transposed order stored linearly.

## 5. Memory implication

`z = y.contiguous()` → independent buffer. You can free the original:

```python
x = ...
y = x.T
z = y.contiguous()
del x, y             # their buffer is now unreferenced → GC'd
# z lives on its own
```

Common pattern when you want to drop a huge tensor but keep a transformed copy.

## 6. `view` vs `reshape`

| method | behavior |
|---|---|
| `x.view(new_shape)` | **Never copies.** Errors if impossible. |
| `x.reshape(new_shape)` | Returns a view if possible, **silently copies** otherwise. |

```python
x = torch.arange(6).reshape(2, 3)   # contiguous
x.view(3, 2)      # ✓ works, no copy

y = x.T                              # non-contiguous
y.view(6)                            # ❌ RuntimeError
y.reshape(6)                         # ✓ works, but copies internally
```

Use `.view()` when you want loud failure if a copy would be needed.
Use `.reshape()` when you don't care.

## Key takeaways

- Contiguous = stride matches "pure row-major, no gaps".
- Transpose (and fancy slices, like `x[::2]`) break contiguity.
- `.contiguous()` forces a real copy into a fresh row-major buffer.
- After `.contiguous()` the result is independent — originals can be GC'd.
- `.view()` = strict no-copy, `.reshape()` = best-effort no-copy.

## Questions I still have

-
