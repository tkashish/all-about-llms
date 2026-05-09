# Triton basics — writing GPU kernels in Python

Triton is a Python-based DSL for writing GPU kernels. See
`02-hardware-anatomy.md` and `06-execution-model.md` for the GPU
fundamentals this note builds on.

## The key mental shift

You don't write code that runs on *one thread*. You write code
that runs on **one block of threads** at a time. Each invocation
of your `@triton.jit` function processes a **chunk** of data, not
a single element.

**Contrast with CUDA:**

- **CUDA (traditional):** you write code for one thread. The runtime
  launches thousands of threads; each thread runs your scalar code.
- **Triton:** you write code for one block. The runtime launches
  many blocks; your code operates on whole vectors. `tl.load` reads
  many elements at once, `tl.store` writes many at once.

## Concrete picture — vector add

```
Input arrays:    [────────────────── N = 1,000,000 ──────────────────]
                 split into blocks of BLOCK_SIZE = 256

Your kernel runs once per block:
  Block 0:    process elements [0..255]       ← one Triton program instance
  Block 1:    process elements [256..511]     ← another program instance
  Block 2:    process elements [512..767]
  ...
  Block 3906: process elements [999,936..999,999 + padding]
```

Each block's work is a **vector operation**: load 256 elements at
once, do the math on all 256 at once, store 256 elements.

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # vector of 256 offsets
x = tl.load(x_ptr + offsets)                            # vector of 256 floats
y = tl.load(y_ptr + offsets)
out = x + y                                             # vector add on 256 elements
tl.store(out_ptr + offsets, out)
```

## Takeaway to keep re-reading

> When you read Triton code, think "I'm writing operations on whole
> vectors of size `BLOCK_SIZE`, not on scalars."

That reframing unlocks everything.

## What comes next in this note

TODO — to be filled as we go:

- `tl.arange`, `tl.load`, `tl.store`, and how masks handle edge cases.
- `tl.program_id` and grid launch semantics.
- Writing the full vector_add kernel.
- Progression: copy → scale → row_sum → softmax.
- Autotuning with `@triton.autotune`.
