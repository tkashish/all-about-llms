# Multi-head attention

## What it is in simple English

Instead of running attention once, run it **H times in parallel** on the
same input. Each copy ("head") looks at the sentence in its own way. Then
combine the H outputs into one.

## What problem does it solve

Single attention has one set of `W_Q`, `W_K`, `W_V` → **one way** of
deciding what's relevant to what.

But in a real sentence, many kinds of relationships matter at once:
- "the dog bit **its** tail" → *its* needs to find *dog* (pronoun ↔ noun)
- "the dog **bit** its tail" → *bit* needs to find *dog* and *tail* (verb ↔ subject/object)
- Every word cares about its **neighbors** for local grammar

One attention with one Q/K/V can't do all of these well — it has to
compromise. Multi-head lets several parallel views each specialize.

## The key insight — identical mechanism, different weights

Every head runs the **exact same math** (scaled-dot-product attention).
The only thing that differs is the **learned weights** inside each
head's `W_Q`, `W_K`, `W_V`.

Analogy: two readers of the same sentence. Same eyes, same brain
structure, but different focus. Grammarian notices subject-verb
agreement; poet notices sound patterns.

Specialization **emerges from training**, not from explicit design.
Different random init → different initial state → gradient descent
pushes each head toward whatever pattern most reduces loss, given what
the other heads are already doing.

## The trick: one big matmul, then reshape

You don't literally run H separate matmuls for Q. You run **one** (D, D)
matmul and reshape its output into H heads of size `D_head = D / H`.

Why: see `wiki/gpu-one-big-matmul.md`. One big matmul beats many
small ones on GPU.

### Block-matrix intuition (D=4, H=2, so D_head=2)

Two tiny per-head Q matrices, each (D, D_head) = (4, 2):
```
W_Q_0 = [[1, 2],        W_Q_1 = [[5, 6],
         [3, 4],                 [7, 8],
         [5, 6],                 [1, 2],
         [7, 8]]                 [3, 4]]
```

Glue them side-by-side:
```
W_Q_big = [[1, 2, 5, 6],
           [3, 4, 7, 8],
           [5, 6, 1, 2],
           [7, 8, 3, 4]]       shape (4, 4) = (D, D)
```

For x = [1, 0, 0, 0]:
```
x @ W_Q_0    = [1, 2]                 # head 0's Q
x @ W_Q_1    = [5, 6]                 # head 1's Q
x @ W_Q_big  = [1, 2, 5, 6]           # both heads, concatenated!
```

One matmul produced both heads' queries, sitting next to each other in
the last axis. Then reshape `(4,) → (H=2, D_head=2)` to split them.

## The full pipeline (D=8, H=2, D_head=4, B=2, T=5)

| stage | shape | op |
|---|---|---|
| input `x` | (B, T, D) = (2, 5, 8) | token embeddings |
| Q, K, V | (B, T, D) = (2, 5, 8) | one matmul per: `x @ W_Q`, etc. |
| split into heads | (B, T, H, D_head) = (2, 5, 2, 4) | `rearrange('b t (h d) -> b t h d', h=H)` |
| move H outer | (B, H, T, D_head) = (2, 2, 5, 4) | `rearrange('b t h d -> b h t d')` |
| scores | (B, H, T, T) = (2, 2, 5, 5) | `Q @ K.T / √D_head` |
| weights | (B, H, T, T) | `softmax(scores, dim=-1)` |
| out per head | (B, H, T, D_head) | `weights @ V` |
| merge heads | (B, T, D) = (2, 5, 8) | `rearrange('b h t d -> b t (h d)')` |
| project | (B, T, D) | `out @ W_O` |

### Notes on the pipeline

- Scaling uses `√D_head`, **not** `√D`. Each head works in its own
  D_head-dim space.
- Moving H next to B (step "move H outer") makes the attention math
  treat `(B, H)` as a joint batch — H independent attentions run in
  parallel.
- Output shape `(B, T, D)` matches input → multi-head attention is a
  drop-in replacement for single-head, stackable as layers later.

## Why `W_O` (the output projection)

After merging heads, the result is `(B, T, D)` with heads laid out
side-by-side inside D:
```
out[b, t] = [ head_0's D_head outputs | head_1's D_head outputs | ... ]
```

**Each head wrote into its own slice** of D. Heads never touch each
other's numbers. Concatenation puts them next to each other but doesn't
mix them.

If downstream layers want to combine facts from different heads
("animate" from head 0 + "subject" from head 1 → "person"), something
needs to mix across head slices. That's `W_O`.

`W_O` is a (D, D) matmul. Each output feature = weighted sum over all D
inputs → features from any head can contribute to any output.

### Why not just add the two halves?

`out[:D_head] + out[D_head:]` only lets same-position features combine.
Too rigid.

| approach | expressiveness |
|---|---|
| `r[:D_head] + r[D_head:]` | same-position only, 0 params |
| elementwise `r * w` | scale per feature, no mixing |
| matmul `r @ W_O` | **any output depends on any input** |

Full matmul is the most general linear combination → what's needed.

## Parameter count

| matrix | shape | role |
|---|---|---|
| `W_Q` | (D, D) | queries (all heads, stacked) |
| `W_K` | (D, D) | keys |
| `W_V` | (D, D) | values |
| `W_O` | (D, D) | mix heads after merge |

Total: **4·D²** per attention layer. H does **not** appear — it's a
reshape hyperparameter, not a parameter count lever.

## Heads vs layers — don't confuse them

- **Heads**: multiple attention computations inside **one** layer,
  running in parallel. H heads.
- **Layers**: transformer **stacks** L attention blocks on top of each
  other. Sequential — output of block i feeds block i+1.

Llama-3-70B: L=80 layers × H=64 heads per layer.
GPT-2 small: L=12 × H=12.

## How H is chosen

`D_head` is usually 64 or 128 (tensor-core friendly). Then H = D / D_head.

- D=768 → H=12 heads of D_head=64 (GPT-2)
- D=8192 → H=64 heads of D_head=128 (Llama-3-70B)

## How this differs from Mixture of Experts

- **Multi-head**: all H heads run for **every** token. Dense.
- **MoE**: N experts exist but each token routes to only **top-k** (usually 1 or 2). Sparse.

MoE replaces the **FFN block** (not attention). It scales total params
without scaling compute per token.

| | Multi-head attention | MoE |
|---|---|---|
| What's multiple? | Attention views | FFN experts |
| All active per token? | Yes (all H) | No (top-k of N) |
| Why? | Different relationship types | Scale params without scaling compute |

## Sanity test (done)

With unit-scale input (std=1), a 2-head attention produced clearly
different attention patterns per head — e.g. head 1 row 3:
`[0.05, 0.07, 0.05, 0.11, 0.73]` — strongly peaked on token 4. Proves
the multi-head math is correct even when the embedding-scale run shows
uniform weights (tiny-init artifact, not a bug).

## Key takeaways

- Multi-head = H parallel attentions, same math, different learned
  weights → specialize during training.
- One big (D, D) matmul produces all H heads' Q (or K, V) at once;
  reshape to split.
- Scale by `√D_head`, not `√D`.
- `W_O` is a separate (D, D) projection after merging heads — it's the
  only place heads actually **mix**.
- 4·D² params per attention layer, regardless of H.
- Drop-in replacement for single-head: same (B, T, D) in/out.
