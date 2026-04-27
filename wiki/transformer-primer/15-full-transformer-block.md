# Full transformer block (step 15)

One block = the atomic unit we stack to build a transformer.

## The complete forward pass

```python
x = x + attention(norm1(x))   # pre-norm + residual
x = x + mlp(norm2(x))         # pre-norm + residual
return x
```

Same `(B, T, D)` shape in and out — stackable.

## What each piece contributes

| piece | purpose |
|---|---|
| `norm1` (RMSNorm) | stabilize magnitudes going into attention |
| `attention` (multi-head, causal, RoPE on Q/K) | mix information **across** tokens |
| `x +` (residual 1) | gradient highway; learn a **correction** to x |
| `norm2` (RMSNorm) | stabilize magnitudes going into MLP |
| `mlp` (SwiGLU) | transform **within** each token non-linearly |
| `x +` (residual 2) | second gradient highway |

## Two residuals, two norms

- Each sublayer (attention, MLP) gets its **own** `RMSNorm`.
- Each sublayer is wrapped in its **own** residual.

Not one norm, not one residual — two of each.

## Why pre-norm

Putting the norm **inside** the residual's sublayer path keeps the
`x + ...` highway numerically clean. More stable for deep stacks —
why modern LLMs (GPT, Llama) use it.

## What's learnable in one block

| component | parameters |
|---|---|
| W_Q, W_K, W_V, W_O (attention) | 4·D² |
| W_signal, W_gate, W_2 (SwiGLU MLP, D_ff = 8D/3) | ≈ 8·D² |
| γ for norm1, γ for norm2 | 2·D |
| **total per block** | **≈ 12·D²** |

RoPE is **not learnable** (fixed cos/sin tables).
Causal mask is **not learnable**.

## Scaling up

Real LMs stack L copies of this block. L is the "depth" hyperparameter:

- GPT-2 small: L=12
- Llama-3-8B: L=32
- Llama-3-70B: L=80

Output of block `i` → input to block `i+1`. Same D throughout.
