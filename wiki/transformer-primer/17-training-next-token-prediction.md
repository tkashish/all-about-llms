# Training — cross-entropy over next-token prediction (step 17)

## What we're teaching the model

Predict the **next token** at every position, given everything before it.

## What the model outputs

For input of shape `(B, T)`, the model outputs logits of shape
`(B, T, vocab_size)`.

For every token position, one score per possible next-token ID:

```
position 0: [2.1, -0.5, 0.3, 1.0, ...]   ← scores over vocab
position 1: [0.1,  3.2, -1.0, 0.8, ...]
...
```

Higher score = model thinks that token is more likely next.

## Training vs inference: why use every position

During **training**, one sentence gives T training examples for free
(one per position):

```
sentence = ["The", "cat", "sat", "on", "mat"]

pos 0 target: "cat"
pos 1 target: "sat"
pos 2 target: "on"
pos 3 target: "mat"
```

Train the model at all positions at once. Causal mask makes this safe
— position `i` cannot see positions `> i`, so it must genuinely
predict.

During **inference** (generation) we only use the **last** position's
logits — sample the next token, append it, rerun.

## Inputs and targets via shift-by-1

From a batch of shape `(B, T+1)`:

```python
inputs  = batch[:, :-1]    # (B, T) — drop last
targets = batch[:,  1:]    # (B, T) — drop first
```

Concrete (one sequence):
```
batch   = [5, 42, 19,  7, 88]       # token IDs, length 5

inputs  = [5, 42, 19,  7]            # drop last
targets = [42, 19,  7, 88]           # drop first

pos 0: input=5  → target=42
pos 1: input=42 → target=19
pos 2: input=19 → target=7
pos 3: input=7  → target=88
```

Position `i` of `inputs` aligns with position `i` of `targets` —
target is exactly "the next token after inputs[i]."

## Cross-entropy — per position

For one position:

```
1. softmax(logits)           → probability distribution over vocab
2. take prob at the target   → prob[target]
3. loss = -log(prob[target])
```

- Correct token had high probability → loss small.
- Correct token had low probability → loss large.

Formula only touches the target's probability. Other vocab entries
matter **indirectly** via softmax: raising target's logit
automatically lowers everyone else's probability (must sum to 1).
Backprop pushes up on target, implicitly down on the rest.

## Shapes entering the loss

| tensor | shape |
|---|---|
| `logits` | (B, T, vocab_size) |
| `targets` | (B, T) |

PyTorch's `F.cross_entropy` wants `(N, C)` and `(N,)`, so flatten
batch+time:

```python
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),   # (B·T, vocab_size)
    targets.reshape(-1),              # (B·T,)
)
```

`F.cross_entropy` fuses `log_softmax + NLL` for numerical stability —
don't manually softmax then log.

## Full training loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(num_steps):
    batch = get_batch(...)                    # (B, T+1)
    inputs  = batch[:, :-1]                   # (B, T)
    targets = batch[:,  1:]                   # (B, T)

    logits = model(inputs)                    # (B, T, vocab_size)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

That's one training step. Run many thousands of them and the model
learns to predict next tokens.

## Perplexity — a human-readable metric

```
perplexity = exp(cross_entropy_loss)
```

Intuition: the "effective number of tokens the model is choosing
between." perplexity = 1 means perfect prediction. perplexity =
vocab_size means uniform random guessing.

- TinyStories-level tiny model: perplexity ~3-10 after training.
- GPT-3 on diverse web text: ~15-25.

## Connecting back to the whole primer

You now have every piece:

1. Tokens → embeddings (step 3)
2. Attention (steps 4–10) — mixes tokens, causal, multi-head
3. MLP (step 11) — per-token non-linear transform
4. Residuals + norm (steps 12–13) — deep stacks stay stable
5. Position (step 14) — RoPE inside attention
6. Block = attention + MLP + 2 residuals + 2 norms (step 15)
7. Stack + output head = full LM (step 16)
8. **Train with cross-entropy next-token loss (this step)**

This is a complete GPT-style language model, ready for real training.
