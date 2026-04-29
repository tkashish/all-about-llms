# Debugging tools in PyTorch

When a model produces wrong output, wrong shapes, or diverges during
training — these are the tools for finding out why. Ordered roughly by
which you reach for first.

## Comparing tensors

### `torch.equal(a, b)` — exact bit-identical check

Returns True if both tensors have the same shape and every element
matches byte-for-byte.

```python
torch.equal(tokens_nc, tokens_c)    # True iff identical
```

Strict — a single bit difference returns False. Good for argmax outputs
and integer tensors.

### `torch.allclose(a, b, atol=1e-5, rtol=1e-3)` — approximate check

Same but allows small numerical tolerance. Used for float comparisons
where tiny numerical noise is acceptable.

```python
torch.allclose(logits_a, logits_b, atol=1e-4, rtol=1e-3)
```

Choose `atol` based on what "similar" means for your use case. For
bf16 intermediate, `1e-2` is lenient; for fp32, `1e-5` is strict. MPS
tends to produce slightly larger drift than CUDA.

### `.abs().max()` — quantify the biggest difference

Shows the worst-case element-wise difference.

```python
diff = (a - b).abs()
print(f"max: {diff.max().item()}, mean: {diff.mean().item()}")
print(f"at index: {diff.argmax().item()}")
```

More diagnostic than a boolean — tells you whether the mismatch is
1e-7 (precision) or 1e-2 (real bug).

## Running through the model

### `breakpoint()` — interactive debugger

Insert anywhere in a forward pass:

```python
def forward(self, x, pos):
    q = ...
    breakpoint()          # drops into pdb REPL
    ...
```

At the REPL: `p q.shape`, `p q[0,0,:4].cpu()`, `p self.is_training`.
Commands: `n` (next line), `c` (continue), `q` (quit). `l` (list code).

On MPS specifically: tensors are async — call `.cpu()` when you want to
see values. `q.cpu()[0, 0, 0, :4]` to force synchronization.

### `register_forward_hook` — inspect outputs without editing code

Register a function that runs after a module's forward:

```python
def hook(module, inputs, output):
    print(f"{module.__class__.__name__}: output shape={output.shape}, "
          f"mean={output.mean():.4f}, max={output.abs().max():.4f}")

# attach to every attention layer
for block in model.transformers:
    block.attention.register_forward_hook(hook)

model(inputs)    # hooks fire automatically
```

No changes to the module code. Remove by calling
`handle.remove()` where `handle` is what `register_forward_hook` returned.

Great for finding "where does my model first produce garbage?" Log the
output mean/max at each layer; the first layer where it explodes or
NaNs is your bug location.

### `torch.autograd.set_detect_anomaly(True)` — find NaNs in backward

If your loss goes NaN during training:

```python
torch.autograd.set_detect_anomaly(True)    # expensive, debug only
```

When backward produces a NaN, it raises with a stack trace pointing to
the op that caused it. Slow — use only while debugging, then turn off.

## Checking correctness between two paths

When you have two code paths that should produce identical output
(e.g., with-cache vs no-cache inference, or `torch.compile` vs eager):

```python
out_a = path_a(inputs)
out_b = path_b(inputs)

# tokens must match exactly
assert torch.equal(out_a, out_b)

# or: logits are close
assert torch.allclose(logits_a, logits_b, atol=1e-4)
```

If they don't match, localize with hooks:

```python
# record outputs from each layer in both models
outputs_a, outputs_b = [], []
for block in model_a.blocks: block.register_forward_hook(
    lambda m, i, o: outputs_a.append(o.detach().clone()))
for block in model_b.blocks: block.register_forward_hook(
    lambda m, i, o: outputs_b.append(o.detach().clone()))

path_a(inputs); path_b(inputs)

for i, (a, b) in enumerate(zip(outputs_a, outputs_b)):
    diff = (a - b).abs().max().item()
    print(f"layer {i}: max diff = {diff}")
```

First layer where `diff > threshold` is where the divergence starts.
Bug is in that layer's code.

## Disabling autograd for inspection

```python
with torch.no_grad():
    out = model(x)
```

Skips building the autograd graph. Faster, uses less memory. Required
for inference. Don't mix with training.

For single tensors: `x.detach()` returns a version with no grad history.

## Shape and dtype assertions

Catch bugs early:

```python
def forward(self, x):
    assert x.shape[-1] == self.d_model, f"expected d_model={self.d_model}, got {x.shape[-1]}"
    assert x.dtype == torch.float32, f"expected float32, got {x.dtype}"
    ...
```

Beats debugging a cryptic broadcasting error 10 layers deeper. Cheap
to run, catches the common class of "I passed the wrong tensor in."

## Determinism (optional)

When you need reproducible runs for debugging:

```python
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)    # errors on non-deterministic ops
```

Some ops don't have deterministic kernels and will raise if you enable
this. MPS in particular has more nondeterminism than CUDA. Use only
for reproducing a specific bug, turn off afterwards.

## Running on CPU to rule out device issues

Sometimes MPS or CUDA produce slightly different results than CPU due
to float reduction order. To check:

```python
model_cpu = Model(...)
model_cpu.load_state_dict(state_dict)
# model runs on CPU — slower, fully deterministic
```

If the bug reproduces on CPU, it's in your code. If it only happens on
GPU, it's hardware-specific (usually non-determinism). Rare for a
real bug to be MPS-only — your code is the likely culprit ~99% of
the time.

## Profiling (separate use case)

Not for correctness debugging, but:

```python
with torch.profiler.profile(activities=[
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,   # or MPS
]) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

Use for finding slow ops after your code is correct.

## Standard debugging playbook

When a PyTorch model produces wrong output:

1. **Print shapes at every layer.** Add temporary `print(x.shape)` or
   assertions. Shape bugs are the most common source of "wrong but
   silent" behavior.
2. **Print tensor norms** (`x.abs().mean()`) to find layers where
   values explode or vanish.
3. **Compare two paths** with hooks — find the first layer where they
   disagree.
4. **Use `breakpoint()`** to poke at intermediates at the offending
   layer.
5. **Simplify the input** — can you trigger the bug with a 2-token,
   batch-size-1 sequence? Smaller inputs make debugging faster.
6. **Check dtype and device** — mismatches produce silent wrong math
   or rare crashes, not loud errors.

## Key takeaways

- `torch.equal` for integer / argmax comparisons.
- `torch.allclose` for float comparisons; always specify `atol`.
- `(a-b).abs().max()` to quantify how wrong.
- `register_forward_hook` to inspect layer outputs without editing.
- `breakpoint()` for interactive poking.
- `torch.no_grad()` for inference.
- Log layer outputs to bisect the location of a bug — first layer
  where something goes wrong is where the bug lives.
