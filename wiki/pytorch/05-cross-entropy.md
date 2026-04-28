# Cross-entropy in PyTorch

The loss function for every classifier and every language model. This
note builds the intuition first (one position, actual numbers), then
scales it up to full-batch LM training, then covers PyTorch's API.

## Part 1 — the simplest case: one position, one sample

Forget batches for now. Imagine **one** token in a sequence, with a
tiny vocabulary of 5 tokens so numbers fit on the page.

**Model output:** a vector of 5 scores, one per possible next token.
```
logits = [2.1, 0.3, 5.0, -0.7, 1.2]
```
These are raw scores (called "logits"), not yet probabilities.

**Target:** the correct next-token ID. One integer.
```
target = 2
```
Meaning "token with ID 2 is the right answer."

## Step 1 — softmax turns logits into probabilities

```
exp each:     [ 8.17,  1.35, 148.40, 0.50, 3.32 ]    sum = 161.74
divide by sum: [0.051, 0.008,  0.918, 0.003, 0.020]    sum = 1.000
```

Now it's a probability distribution. Class 2 has 91.8% probability —
the model is quite confident token 2 is next. `exp` is monotonic, so
the biggest logit becomes the biggest probability. Softmax keeps the
ordering and forces everything to sum to 1.

## Step 2 — loss = `-log(probability of the correct class)`

The correct class is `target = 2`. Its probability is `0.918`.

```
loss = -log(0.918) = 0.086
```

Small number → small loss → model was right and confident.

### What if the model was wrong?

Same logits, but suppose `target = 0` instead. The probability softmax
assigned to class 0 was only `0.051`:

```
loss = -log(0.051) = 2.97
```

Big number → big loss → model was wrong.

## The one big table

Cross-entropy has one variable: the probability your model assigned to
the correct class.

| prob of target class | loss = -log(prob) |
|---|---|
| 99% | 0.01 |
| 90% | 0.11 |
| 50% | 0.69 |
| 10% | 2.30 |
| 1% | 4.61 |
| 0.01% | 9.21 |

**Nothing else enters the formula.** The other 4 probabilities in the
5-class example don't appear directly. But they matter **indirectly**:
softmax makes everything sum to 1, so pushing up the correct class's
logit automatically pushes down the others. That's how gradient
descent via cross-entropy teaches the model to distinguish classes.

## Part 2 — scaling up: multiple positions

Now bring back the `B, T` dimensions.

### Why every position is its own training example

Language-model training predicts the next token at **every** position.
For a sequence like `["The", "cat", "sat", "on"]`, the model learns:

```
position 0 (given "The")               → predict "cat"
position 1 (given "The cat")           → predict "sat"
position 2 (given "The cat sat")       → predict "on"
position 3 (given "The cat sat on")    → predict (whatever comes next)
```

Four training examples from one four-token sequence. Each position is
independent as far as the loss is concerned.

### Shapes you see in a forward pass

For a batch of `B=4` sequences of length `T=512` with vocab `V=10000`:

| tensor | shape | meaning |
|---|---|---|
| `logits` | `(4, 512, 10000)` | a 10000-score vector for every position of every batch |
| `targets` | `(4, 512)` | one correct token ID per position of every batch |

Total training examples in this batch: `4 * 512 = 2048`. Each is
exactly the single-position case from Part 1 — 10000 logits + 1
target integer.

### What cross-entropy does with them

1. For every one of the 2048 positions, apply softmax to its 10000
   logits.
2. For every position, take `-log(prob[target])`.
3. Average the 2048 losses into one scalar.

The scalar is what you `.backward()` on. Gradient descent pushes the
correct class's logit up (and others down, via softmax) at every
position simultaneously.

## Part 3 — calling it in PyTorch

```python
import torch.nn.functional as F

loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),   # (B*T, V)
    targets.reshape(-1),               # (B*T,)
)
```

Returns a scalar — the mean cross-entropy across all `B*T` positions.

### What goes in

| arg | shape | dtype | contents |
|---|---|---|---|
| `logits` | `(N, V)` | float32 / bf16 | raw scores over V classes. **NOT softmaxed.** |
| `targets` | `(N,)` | int64 | correct class index per example, in `[0, V)` |

`N` is the number of examples. For LM training, `N = B * T`.

### Why reshape? PyTorch's class-axis convention

`F.cross_entropy` accepts higher-rank inputs, but the **class axis
must be at position 1**, not the last axis:

```
PyTorch convention: (N, C, ...)     C is the class axis
Your tensor:         (B, T, V)       V (the class axis) is at -1
```

Two ways to reconcile:

**Option A — flatten (standard):**
```python
F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
```

**Option B — transpose:**
```python
F.cross_entropy(logits.transpose(1, 2), targets)   # (B, V, T) vs (B, T)
```

Both give the same result. **Flattening is more common** — slightly
more efficient (no transpose copy), and it matches the mental model
that every position is an independent example.

### The `.reshape(-1, V)` idiom

`-1` means "figure this out from the other dims":

```python
logits.shape               # (4, 512, 10000)
logits.reshape(-1, 10000)  # (2048, 10000) — PyTorch computes 4*512
```

Idiom: "flatten leading dims, keep the last."

## Part 4 — why you must NOT softmax first

`F.cross_entropy` does the softmax internally. Doubling up is a bug.

Wrong:
```python
probs = F.softmax(logits, dim=-1)
loss = F.cross_entropy(probs, targets)    # double softmax — garbage
```

Right:
```python
loss = F.cross_entropy(logits, targets)   # raw logits in, scalar out
```

The inputs to `F.cross_entropy` should be whatever came out of your
last Linear layer — raw unnormalized scores.

## Part 5 — numerical stability (why the fused op matters)

Doing softmax + log manually is unstable. For large logits:

```python
probs = torch.softmax(logits, dim=-1)     # exp(20) = 485M; exp(100) = inf
loss = -torch.log(probs)                  # log(0) = -inf, log(nan) = nan
```

`F.cross_entropy` uses the **log-sum-exp trick** internally:

```
log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
```

Subtract the max first → every `exp` is of a number `≤ 0` → no
overflow, no nan. Same math, numerically stable. Always use the fused
`F.cross_entropy`; never write softmax + log manually.

## Part 6 — useful arguments

```python
F.cross_entropy(
    logits, targets,
    ignore_index=-100,        # skip positions where target == -100
    reduction="mean",         # "mean" (default), "sum", "none"
    label_smoothing=0.0,      # soften targets; 0.1 common for big models
)
```

**`ignore_index`**: positions with target `-100` contribute 0 to the
loss. Standard for padding — set pad-token positions to `-100` in
targets and they won't be trained on.

**`reduction="none"`** returns per-example loss instead of the mean.
Useful for weighted loss or per-example logging.

**`label_smoothing`** replaces the one-hot target with a mix of
one-hot and uniform. Helps overconfidence; 0.1 is common for large
models (GPT-3, Llama).

## Part 7 — gotchas

1. **`targets` must be `int64` (`long`), not `int32` or `float`.**
   Cast with `.long()` if loading from numpy `uint16`.
2. **`targets` must be in `[0, V)`.** Anything outside either errors or
   silently returns nan.
3. **Don't softmax first.** Common beginner bug.
4. **In 3D+ mode, class axis is position 1, not -1.** Hence the reshape.

## Key takeaways

- Cross-entropy = `-log(prob_of_correct_class)`, averaged over examples.
- Every `(b, t)` position in an LM is one independent example.
- Flatten `(B, T, V)` → `(B*T, V)` with `reshape(-1, V)` to fit
  PyTorch's 2D API.
- Feed raw logits, not probabilities — softmax is inside.
- Targets are `int64` indices in `[0, V)`.
- Returns a scalar (mean by default). `reduction="none"` for per-example.
