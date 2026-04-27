# Backward pass — intuition first

## Training, in one loop

```
forward  →  loss  →  backward  →  optimizer step  →  repeat
```
- **Forward:** push `x` through the model, get prediction `y`, compare to target → loss (a single scalar).
- **Backward:** for every weight `W_ij`, compute `∂loss/∂W_ij` — "how much does loss change if I nudge this weight."
- **Optimizer:** `W ← W − η · ∂loss/∂W`. Learning rate `η` is step size; the gradient is direction + steepness.

The gradient is a property of the loss landscape. The learning rate is *your* stride length.

## The chain rule — why backward works

If `y = 3x` then `dy/dx = 3`: nudge x by 1 → y moves by 3.

If `y = 3x` and `z = 5y`, then a nudge in `x` causes a nudge in `y` causes a nudge in `z`:
```
dz/dx  =  dz/dy · dy/dx  =  5 · 3  =  15
```
Rates multiply when effects chain.

## Applied to a network

```
W_1 → h_1 → h_2 → loss
```
```
dloss/dW_1 = dloss/dh_2 · dh_2/dh_1 · dh_1/dW_1
```
Each factor is a **local** rate — only needs one layer's math to compute.

## Why it's called "backward"

The leftmost factor `dloss/dh_2` lives at the **end** of the network. We compute
it first, then multiply in earlier factors as we walk back through layers.

```
Forward:   x ──→ h_1 ──→ h_2 ──→ loss
Backward:  x ←── h_1 ←── h_2 ←── loss
```

At each layer, given the gradient flowing in from the right, we compute:
1. **`∂loss/∂W`** — this layer's weight gradient (hand to optimizer).
2. **`∂loss/∂(input)`** — pass to the previous layer so it can keep going.

Two gradients per layer. That's the key fact for the "backward ≈ 2× forward"
FLOP count, coming next.

## Notation note

`∂y/∂x` (partial) vs `dy/dx` (ordinary) — same meaning in this context:
"change in y per unit change in x." Use `∂` when there are multiple inputs
and you want to vary one at a time.
