# Attention — Q, K, V explained

## What are they in simple English?

Three different views of each token's vector:

- **Query (Q)** = what this token is asking about / looking for
- **Key (K)** = what this token offers for others to match against
- **Value (V)** = what this token actually contributes if matched

## What problem does this solve?

To compute relevance between two tokens, we need a way to compare them.
But the same vector can't play all three roles — asking vs being
compared vs being pulled in need different shapes of information.

Solution: **three separate learned projections** of the original
embedding vector. Same token, three specialized views.

## Library analogy

You walk into a library:
- Your **question** = Query ("I want something about WW2 tanks")
- Book **titles/blurbs** = Keys (what each book advertises)
- Book **contents** = Values (what you actually read)

You compare your question (Q) to every title (K). Good match → open the
book and read (V). Weighted sum of contents = what you end up
remembering.

## Mechanism

Take each token's embedding vector and multiply it by **three different
learnable weight matrices**:

```
v     ← original vector, shape (D,)
Q_v   = v @ W_Q     shape (D,)  ← query view
K_v   = v @ W_K     shape (D,)  ← key view
V_v   = v @ W_V     shape (D,)  ← value view
```

`W_Q`, `W_K`, `W_V` are each shape `(D, D)`, all learnable parameters of
the attention layer. Total parameters so far: `3 · D²`.

## Concrete trace

4 tokens, D=3:

```
v_deposit  = [0.5, 1.0, 0.0]
v_at       = [0.1, 0.1, 0.1]
v_the      = [0.1, 0.1, 0.1]
v_bank     = [0.2, 0.0, 0.9]
```

Apply three matmuls to every token:

```
v_deposit ──┬→ @ W_Q → Q_deposit
             ├→ @ W_K → K_deposit
             └→ @ W_V → V_deposit

v_bank    ──┬→ @ W_Q → Q_bank
             ├→ @ W_K → K_bank
             └→ @ W_V → V_bank
  ... and for every other token
```

After: for each of 4 tokens we have 3 vectors (Q, K, V). 12 new vectors
from 4 starting ones.

**Same `W_Q, W_K, W_V`** is used for every token. These matrices are the
shared learnable parameters; they see every position in every sequence.

## What each matrix learns

- `W_Q` — how to turn a token into "what am I looking for?"
- `W_K` — how to turn a token into "what do I advertise?"
- `W_V` — how to turn a token into "what do I contribute when matched?"

All three are trained by backprop to make the final next-token
prediction as accurate as possible. There's no separate loss for "good
queries" vs "good keys" — the whole thing is end-to-end learned.

## What's next

With Q, K, V for each token, we compute:

1. **Attention scores:** how strongly does each query match each key?
   → dot product between Q and K.
2. **Attention weights:** normalize scores into probabilities
   → softmax.
3. **Output:** weighted average of values using the weights
   → the new contextualized vector.

That's the scaled dot-product attention formula, coming next.

## Clarification: Q, K, V are NOT derived from each other

A common confusion: "is V derived from Q and K?" **No.**

All three come **directly from the embedding, in parallel**, via three
separate weight matrices. None depends on the others.

```
embedding ──┬→ @ W_Q → Q
            ├→ @ W_K → K
            └→ @ W_V → V
```

Three independent matmuls. Happens in parallel.

### Two distinct phases

**Phase 1 — Compute Q, K, V (parallel, independent):**
```
Q = embedding @ W_Q
K = embedding @ W_K
V = embedding @ W_V
```

**Phase 2 — Use Q, K, V together to produce the output:**
```
scores  = Q vs K              # how well each query matches each key
weights = softmax(scores)     # normalize to probabilities
output  = weights @ V         # weighted sum of V vectors
```

In phase 2, Q and K decide **how much of each V** to include — but
they don't create V. V is its own thing, computed once in phase 1.
