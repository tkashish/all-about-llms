# PyTorch ops and vectorization

## What "op" means

An **op** (operation) is any function PyTorch provides that takes tensors
in, does math, and returns tensors out. The word is jargon for "PyTorch
function or method."

Three flavors, same thing underneath:

```python
torch.cos(x)   # module-level function
x.cos()        # tensor method — equivalent
x + y          # operator — dispatches to torch.add
```

Most ops have all three forms. `torch.cos(x)`, `x.cos()`, and the `+`
operator all call the same C++/CUDA code.

## Why ops, not Python loops

Ops apply to **every element** of a tensor in parallel, running in
optimized C++/CUDA. A Python loop runs one interpreter step per cell —
thousands of times slower.

```python
# Python loop — slow, not vectorized
for i in range(len(x)):
    y[i] = math.cos(x[i])

# PyTorch op — fast, one call into C++
y = torch.cos(x)
```

For a million-element tensor, the loop is ~1000× slower. On GPU,
~10000× slower (loop can't use SIMD lanes or tensor cores).

**Rule of thumb:** if you see a Python `for` inside a forward pass,
you've fallen off the fast path. Find the op that does it in one call.

## What counts as a "PyTorch op"

- `torch.cos(x)` where `x` is a tensor → PyTorch op ✓
- `x.cos()` → PyTorch op ✓
- `x + y` with two tensors → PyTorch op ✓
- `math.cos(x)` with a Python float → Python stdlib, not PyTorch ✗
- `np.cos(x)` → NumPy, not PyTorch ✗

If either side of the operation is a tensor, you're in PyTorch land.
If both sides are Python scalars, you're using Python's built-in
number ops.

## Scalar + tensor works automatically

PyTorch **broadcasts** scalars to match any tensor:

```python
x = torch.tensor([0.0, 1.0, 2.0, 3.0])

x + 1         # [1.0, 2.0, 3.0, 4.0]
x * 2         # [0.0, 2.0, 4.0, 6.0]
x ** 2        # [0.0, 1.0, 4.0, 9.0]
10000 ** x    # [1, 10000, 1e8, 1e12]
1 / x         # [inf, 1.0, 0.5, 0.333]
```

Every scalar in the expression gets applied elementwise to the tensor.
No loop needed. See [broadcasting](02-broadcasting.md) for the full rules.

## Chaining ops — the "no loops" idiom

When a formula depends on indices, build an index tensor and apply ops
to it, instead of looping over indices.

**Goal:** compute `f_k = 1 / 10000^(2k / D)` for `k = 0, 1, ..., D/2 - 1`.

Naive loop:
```python
f = []
for k in range(D // 2):
    f.append(1 / 10000 ** (2 * k / D))
f = torch.tensor(f)                 # slow
```

Vectorized — one expression, no loop:
```python
k = torch.arange(D // 2, dtype=torch.float32)
f = 1 / 10000 ** (2 * k / D)        # fast — elementwise on k
```

Each op (`*`, `/`, `**`) runs across the whole tensor at once.

## Common dtype gotcha

`torch.arange(n)` returns an **integer** tensor by default:

```python
torch.arange(4).dtype        # torch.int64
```

Integer tensors can overflow or produce wrong results in math that
expects floats:

```python
10000 ** torch.arange(4)     # int tensor — overflows at ~10000^9
```

Fix: cast at creation time.

```python
torch.arange(4, dtype=torch.float32)
```

Always think about dtype when you build tensors from scratch.

## Key takeaways

- Ops run elementwise over entire tensors in native code.
- Any formula that works on a scalar works on a tensor of the same ops.
- Replace Python loops with chained ops + broadcasting.
- Mind dtype — `arange` defaults to int, cast to float when doing float math.
