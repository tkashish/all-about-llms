# Position embeddings (step 14, in progress)

## What it is in simple English

A way to tell the model **where** each token sits in the sequence
(position 0, 1, 2, ...). Without this, the transformer has no sense
of order.

## What problem does it solve

### Concrete example — why position matters

Two sentences, same 4 tokens: `["the", "dog", "bit", "cat"]`.

- "**the dog bit the cat**" → dog is the biter, cat is the victim
- "**the cat bit the dog**" → cat is the biter, dog is the victim

Same bag of tokens, **opposite meaning**. Position tells you which word
plays which role.

### Why attention can't tell them apart

Each token's Q, K, V are computed from its embedding vector. Attention
then does dot products and softmax over those vectors.

The attention math **never looks at the index `i`**. It only operates
on vector contents.

So if "dog" is at position 1 or position 4, its token embedding is
identical (same lookup row) → its Q, K, V are identical → the
attention output is identical.

The model literally cannot distinguish "dog bit cat" from "cat bit dog."

### The fix

Make "dog at position 1" produce a **different vector** than "dog at
position 4." Then they flow through attention differently, and the
model can learn that order matters.

Position embeddings do exactly that: inject position information into
each token's vector before attention sees it.

## The simplest approach — learned absolute position embeddings

Just like the token embedding table, create a **second lookup table**
keyed by position:

```
token_embedding_table:    (vocab_size,    D)   one row per token ID
position_embedding_table: (max_seq_len,   D)   one row per position
```

For each token in the input, add its position row to its token row:

```
final[t] = token_table[token_id_at_t] + position_table[t]
```

### Tiny example (max_seq_len=4, D=3)

```
position_table (learnable, shape (4, 3)):
  position 0 → [ 0.1, -0.2,  0.3]
  position 1 → [ 0.4,  0.1, -0.5]
  position 2 → [-0.3,  0.6,  0.2]
  position 3 → [ 0.7, -0.4,  0.1]
```

Input tokens `["the", "dog", "ran", "."]` at positions 0..3:

```
final[0] = token_table["the"] + position_table[0]
final[1] = token_table["dog"] + position_table[1]
...
```

"dog" at position 1 and "dog" at position 2 now yield **different**
final vectors — because the position half differs.

## Problems with learned absolute positions

1. **Fixed max length.** Table has only `max_seq_len` rows. Sequences
   longer than that break — no row to look up.
2. **No sharing between positions.** Position 3 and position 4 are
   trained as **independent** vectors. If position 500 is rarely seen,
   it stays poorly trained — nothing is learned from its neighbors.
3. **No built-in relative distance.** The model has to learn "position
   5 and position 6 are adjacent" from data; it isn't free.

What we want: a scheme where adjacent positions naturally look similar,
and where any length works. That's what **sinusoidal** (original
transformer) and **RoPE** (modern) give.

## Sinusoidal positional encoding

### Plain English

Instead of **learning** a position table, **compute** it from a fixed
math formula using `sin` and `cos`. No learnable parameters — every
position is fully determined by its index.

### Why it helps vs learned absolute

1. **Unlimited length.** Formula works for any `t`; no table-size cap.
2. **Smooth structure.** Nearby positions are automatically similar,
   far positions naturally differ. No independent-training problem.

### Quick refresher on sinusoids

`sin(x)` and `cos(x)` are the familiar wavy curves, oscillating between
−1 and +1 forever:

```
sin(x):        _    _    _
              / \  / \  / \
         ____/   \/   \/   \____
                  time →
```

**Frequency** = how fast the wave oscillates.
- Low frequency → long wave → slow change.
- High frequency → short wave → fast change.

### The encoding idea

For each position `t`, compute `sin(t · f)` and `cos(t · f)` at
**several different frequencies** `f₁, f₂, …`, and pack the values
into a vector.

Tiny example, D=4, 2 frequencies (f₁ fast, f₂ slow):

```
pos 0: [sin(0·f₁), cos(0·f₁), sin(0·f₂), cos(0·f₂)] = [0, 1, 0, 1]
pos 1: [sin(1·f₁), cos(1·f₁), sin(1·f₂), cos(1·f₂)]
pos 2: [sin(2·f₁), cos(2·f₁), sin(2·f₂), cos(2·f₂)]
```

Each position gets its own 4-dim vector.

### Why close positions produce similar vectors

A slow wave moves little between neighbors:

```
sin(0 · 0.1) = 0.00
sin(1 · 0.1) = 0.10    ← tiny change
sin(2 · 0.1) = 0.20
```

Positions 0 and 1 differ only a little → their encoding vectors differ
only a little. Smooth by construction.

And far positions on the same slow wave differ a lot:

```
sin(0 · 0.1)  = 0.00
sin(50 · 0.1) = sin(5.0) ≈ -0.96
```

So "distance between two encodings" grows naturally with "distance
between their positions."

### Why use multiple frequencies

A single wave has a **period** — after one period, it wraps and
repeats. Two positions differing by exactly one period look identical
on that wave.

Fast wave (period 10), same value at positions 0, 10, 20, ...:

```
sin(0·2π/10)  = 0
sin(10·2π/10) = 0   ← same!
sin(20·2π/10) = 0   ← same!
```

Fast wave alone doesn't tell position 0 apart from position 10.

**Slow wave disambiguates:** position 0 vs 10 fall at clearly different
points on a slow wave (period 1000).

Together — fast + slow at once — positions look identical on **all**
waves only if they differ by the least-common-multiple of all periods,
which is astronomically large.

> Fast waves: pin down position within a small window.
> Slow waves: pin down position globally.
> Together: unique fingerprint per position over a huge range.

**Analogy:** a clock has hour, minute, and second hands (three
frequencies). Any one hand alone leaves ambiguity; together they give
a unique time.

### Used before RoPE

The original 2017 transformer used sinusoidal encoding. It's still a
valid choice; most 2023+ models moved to RoPE because RoPE bakes
relative-distance into attention directly (details next).

## RoPE — Rotary Position Embedding (modern)

### Plain English

Instead of **adding** a position vector to the token embedding, **rotate**
each Q and K vector by an angle that depends on its position. Position
info gets injected **inside attention** (on Q and K), not at the
embedding step.

V is **not** rotated.

### Why rotation

Attention ultimately cares about the **relative distance** between a
query and a key — "how far apart are token i and token j?" — not each
one's absolute position.

The key algebraic fact: when you dot-product two rotated vectors, the
result depends only on the **difference** of the two rotation angles.
So if the angles are `t · f`, the dot product ends up depending on
`(i − j) · f` — purely relative.

### Step-by-step derivation

#### Step 1 — Rotation on a 2D pair

A point `v = (x, y)` rotated by angle `θ`:

```
v' = (x·cos θ − y·sin θ,  x·sin θ + y·cos θ)
```

Pure 2D geometry.

#### Step 2 — Dot product between two rotated 2D vectors

`Q = (q_x, q_y)` rotated by `α` → `Q'`.
`K = (k_x, k_y)` rotated by `β` → `K'`.

```
Q' · K'  =  (Q · K) · cos(α − β)  +  (something) · sin(α − β)
```

Both terms depend on `(α − β)` only — not on `α` and `β` separately.

#### Step 3 — Pick angles based on position

Set `α = i · f`, `β = j · f` for positions `i` and `j` with frequency
`f`. Then:

```
α − β = (i − j) · f          ← relative distance
```

So `Q' · K'` depends only on `(i − j)`:

- Position 5 and 3 → distance 2 → `cos(2·f), sin(2·f)`
- Position 105 and 103 → distance 2 → `cos(2·f), sin(2·f)` (same!)

**Same signal regardless of absolute positions.** The model learns
patterns in terms of relative distance automatically.

#### Step 4 — Extend to D-dim vectors

Real Q/K vectors are D-dim (say `D_head = 64`). Split into `D_head / 2`
pairs and rotate each pair with **its own frequency**:

```
[q_0, q_1, q_2, q_3, ..., q_62, q_63]
 └─┬─┘  └─┬─┘       └────┬────┘
 pair 0  pair 1        pair 31
 rotate  rotate         rotate
 by f_0  by f_1          by f_31
```

Standard choice (RoPE paper / Llama):

```
f_k = 1 / 10000^(2k / D_head)     for k = 0, 1, ..., D_head/2 - 1
```

- Low `k` → high frequency → fast rotation (fine-grained position).
- High `k` → low frequency → slow rotation (coarse-grained position).

Same multi-frequency logic as sinusoidal — avoids wrap-around ambiguity.

#### Step 5 — Where RoPE sits in attention

```
x
├─ @ W_Q → Q → RoPE(Q, positions) → Q_rot
├─ @ W_K → K → RoPE(K, positions) → K_rot
└─ @ W_V → V     (no rotation)

scores = Q_rot · K_rot    ← relative-distance baked in
weights = softmax(scores)
out = weights @ V
```

Two things to remember:
- RoPE is **not** a separate layer before the transformer — it lives
  inside attention, on Q and K.
- V is **untouched** — only the query-key comparison needs position,
  not the content being pulled.

### Feature = dimension = one number

In this discussion "feature" and "dimension" mean the same thing: one
number slot in the vector. `D_head = 4` means 4 numbers per token per
head → 2 pairs for RoPE.

## Summary table

| | Learned absolute | Sinusoidal | RoPE |
|---|---|---|---|
| Parameters | yes (table) | no | no |
| Unlimited length | ❌ | ✅ | ✅ |
| Relative distance encoded | ❌ | partial | ✅ native |
| Applied at | embedding | embedding | inside attention (Q, K) |
| V affected | yes | yes | no |

Used by Llama, Mistral, GPT-NeoX, Qwen, DeepSeek — and by the model in this repo.
