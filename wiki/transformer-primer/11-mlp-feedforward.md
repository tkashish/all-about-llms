# MLP / feed-forward block (step 11, in progress)

## What is an MLP?

**Multi-Layer Perceptron** — fancy name for the simplest neural net: a
stack of linear layers with nonlinearities in between.

In the transformer, "the MLP" specifically means a **2-layer** version:

```
input → Linear → nonlinearity → Linear → output
```

## What it does in the transformer

After attention mixes information *across* tokens, the MLP transforms
each token's vector **independently**. Same 2-layer net applied to
every token, no mixing between tokens.

## What problem does it solve

Attention is a **weighted sum** — a linear operation. Stacking many
linear ops collapses into one big linear op. Linear-only models can't
learn curves, thresholds, or "if X and Y then Z" patterns.

The MLP injects a **nonlinearity** — the ability to learn complex
functions of each token's features.

## Where it sits in the architecture

The MLP is **not** part of the attention layer. It's a separate block.

A **transformer block** = attention + MLP, stacked:

```
x → attention → MLP → output
```

A full transformer stacks L such blocks:

```
tokens → embedding → [attn + MLP] → [attn + MLP] → ... → output
                     └─ block 1 ─┘   └─ block 2 ─┘
```

- GPT-2 small: L = 12 blocks
- Llama-3-70B: L = 80 blocks

## Mechanics — the shapes

Input from attention: `(B, T, D)`.

```
(B, T, D) → Linear → (B, T, D_ff) → nonlinearity → Linear → (B, T, D)
```

Two weight matrices:

| matrix | shape | role |
|---|---|---|
| `W_1` | (D, D_ff) | expand |
| `W_2` | (D_ff, D) | contract back |

Forward:

```python
h   = x @ W_1            # (B, T, D_ff)
h   = nonlinearity(h)    # same shape
out = h @ W_2            # (B, T, D)
```

Two things to notice:
1. **Same in/out shape** `(B, T, D)` → drop-in, stackable.
2. **Middle is wider** — `D_ff`, typically `4·D`.

## Why wider in the middle

The nonlinearity is the **only** place the network can learn non-linear
patterns. More room there = more patterns it can represent.

At `D_ff = 4·D` the nonlinearity has 4× more "slots" to fire in
different ways. Each slot can act like a mini-detector ("is this a
verb?", "is this sentiment negative?", ...).

So the MLP is: **expand → transform (nonlinear) → compress back**. The
wide middle is where the actual thinking happens; the two linear layers
are wrappers that get in and out.

## Attention vs MLP — two different jobs

| | Attention | MLP |
|---|---|---|
| Across or within tokens? | **across** (mixes tokens) | **within** (per-token) |
| Question it answers | *who should I listen to?* | *given what I've heard, what does this mean?* |
| Operation | weighted sum of other tokens' V | expand → nonlinear → contract |

Example: after attention, the "bank" vector has absorbed "river" info →
roughly `[word:bank, context:water]`. The MLP turns that into
`[concept:shore, not:financial]` — a refined representation the next
layer can use.

## Nonlinearity — why it's needed

Plain English: a function that **bends** the output — not a straight line.

Without it, two stacked linear layers collapse into one:
```
(x @ W_1) @ W_2  =  x @ (W_1 @ W_2)
```
No matter how many you stack, you only get one linear function total.
The nonlinearity is the **only** thing that prevents this collapse.

## ReLU — the simplest

```
ReLU(x) = max(0, x)
```
- x > 0: pass through
- x ≤ 0: output 0

Nonlinear because of the bend at zero. Works, stacks correctly.

### Problem: dying neurons

If a neuron's pre-activation goes negative, ReLU outputs 0 → gradient = 0
→ the neuron can't update → it stays stuck forever. "Dead."

Observed in the AdaGrad walkthrough (see `wiki/foundations/19-adagrad-walkthrough.md`)
— the loss plateau was partly dying ReLUs.

## GeLU — the smooth alternative

"ReLU with a smooth corner."

- x >> 0: looks like ReLU, passes through
- x << 0: output slowly drops toward 0 (asymptote, never exactly 0)
- around 0: smooth S-curve, not a sharp corner
- small dip below 0 around x ≈ -0.5 (a tiny valley)

Formula: `GeLU(x) = x · Φ(x)` where Φ is the Gaussian CDF.

### Why it fixes dying neurons

ReLU at large negative: output = 0 **exactly**, gradient = 0 **exactly**.
GeLU at large negative: output and gradient both → 0 **asymptotically**,
never *exactly* zero.

That tiny nonzero gradient is what lets the neuron recover. It idles
slowly instead of dying.

### Why not just `exp(x)`?

Tempting ("always positive, no zero problem"), but:
1. No "turn off" region — network can only scale up, not suppress.
2. `exp` explodes fast — `exp(10) ≈ 22000` → unstable training.

What you want: smooth, bounded-ish, still has a "mostly off" region.
GeLU has all three.

## SwiGLU — what Llama / Mistral use (and what we use in this repo)

### Problem it solves

In ReLU/GeLU, the nonlinearity is **fixed** — same rule applied to
every number. The network can't *learn* how much of each feature to
let through; the suppression is hardcoded.

SwiGLU adds a **learnable gate**: the network decides per-feature how
much signal to keep.

Analogy: ReLU = light switch (on/off based on sign). SwiGLU = dimmer
(learned brightness per feature).

### Mechanics — three matrices, not two

| matrix | shape | role |
|---|---|---|
| `W_1` | (D, D_ff) | signal projection |
| `W_gate` | (D, D_ff) | gate projection |
| `W_2` | (D_ff, D) | contract back |

### Forward

```python
signal = x @ W_1              # (B, T, D_ff)
gate   = x @ W_gate           # (B, T, D_ff)
h      = SiLU(signal) * gate  # element-wise multiply
out    = h @ W_2              # (B, T, D)
```

- `SiLU(x) = x · sigmoid(x)` — smooth activation, very close to GeLU.
- `* gate` is **element-wise** → each of the `D_ff` hidden features
  gets its own learned multiplier.

### Why two projections of the same input

Because `W_gate` is learnable, backprop can push it to produce:
- **large** gate values for features the model wants to keep
- **small** gate values for features to suppress

The gate is learned per-feature, per-token. Same input x, but the
"which features to amplify right now" question is answered by a
trained matrix rather than a hardcoded rule.

### Param count — SwiGLU vs ReLU/GeLU

SwiGLU has 3 matrices; ReLU MLP has 2. To keep total params the same,
SwiGLU uses a smaller hidden dim — typically `D_ff = 8·D / 3` instead
of `4·D`. Three matrices of size `(D, 8D/3)` ≈ 8·D² params, matching
two matrices of `(D, 4D)` = 8·D² from a ReLU MLP.

## Acronyms

- **MLP** — Multi-Layer Perceptron
- **ReLU** — Rectified Linear Unit
- **GeLU** — Gaussian Error Linear Unit
- **SiLU** — Sigmoid Linear Unit (also called **Swish**: `x · sigmoid(x)`)
- **GLU** — Gated Linear Unit (the "two-matrices-multiplied" pattern)
- **SwiGLU** — **Swi**sh + **GLU** → SiLU as the activation combined with
  GLU-style gating

## Init gotcha — divide by √fan_in, not √fan_out

The `1/√fan_in` rule uses the **dim you contract over** (the dim that
gets summed in the matmul).

| matrix | shape | contracted dim | divide by |
|---|---|---|---|
| `W_signal` | (D, D_ff) | D | `√D` |
| `W_gate` | (D, D_ff) | D | `√D` |
| `W_2` | (D_ff, D) | D_ff | `√D_ff` |

Reason: `h = x @ W`. If W is `(D, D_ff)`, each `h[i]` sums over `D`
products → variance scales with `D` → divide by `√D` to keep it ~1.
Same rule, applied per matrix by looking at its *left* dim.

Common mistake: using `√D_ff` for all three because the hidden dim
"feels like the big one." Only matters when `D ≠ D_ff`, but matters
meaningfully when they differ.
