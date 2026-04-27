# Wiki

Personal reference notes on language-model internals, built while writing the
code in this repo. Each topic is a short standalone note: plain English first,
then the problem it solves, then the mechanics.

## Reading order for a first pass

1. **Tensors & FLOPs** — [`foundations/01`](foundations/01-floating-point.md)
   through [`foundations/16`](foundations/16-6nd-rule.md).
   Floats, dtypes, strides, einops, FLOPs counting, backward-pass math,
   the 6ND rule.
2. **Autograd & optimizers** — [`foundations/17`](foundations/17-pytorch-autograd-mechanics.md)
   through [`foundations/19`](foundations/19-adagrad-walkthrough.md).
   How PyTorch's graph works, how to subclass Optimizer, AdaGrad walkthrough.
3. **Training memory** — [`foundations/20`](foundations/20-training-memory-four-components.md)
   through [`foundations/22`](foundations/22-activation-checkpointing.md).
   The four components that eat GPU memory, plus gradient accumulation and
   activation checkpointing.
4. **The transformer** — [`transformer-primer/03`](transformer-primer/03-weight-initialization.md)
   through [`transformer-primer/17`](transformer-primer/17-training-next-token-prediction.md).
   Builds a decoder-only LM from weight init through attention, MLP,
   residuals, RMSNorm, RoPE, the full block, the full stack, and the
   cross-entropy training loop.

## Layout

```
wiki/
├── foundations/           PyTorch internals, autograd, training memory
├── transformer-primer/    decoder-only transformer, built up step by step
└── gpu-one-big-matmul.md  standalone cheatsheet
```

## Index

### `foundations/`

| # | topic |
|---|---|
| 01 | [floating-point](foundations/01-floating-point.md) |
| 02 | [binary fractions (why 0.1 is weird)](foundations/02-binary-fractions.md) |
| 03 | [tensors: shape, dtype, device](foundations/03-tensors-basics.md) |
| 04 | [memory layout: strides, views](foundations/04-memory-layout.md) |
| 05 | [range vs precision](foundations/05-range-vs-precision.md) |
| 06 | [half-precision dtypes (fp16, bf16, fp8, fp4)](foundations/06-half-precision-dtypes.md) |
| 07 | [why there's no bf24](foundations/07-why-no-bf24.md) |
| 08 | [views: shared-buffer gotchas](foundations/08-views-memory-gotchas.md) |
| 09 | [contiguous, view vs reshape](foundations/09-contiguous-view-reshape.md) |
| 10 | [einops: rearrange, reduce, einsum](foundations/10-einops.md) |
| 14 | [backward pass — intuition](foundations/14-backward-pass-intro.md) |
| 15 | [backward pass — worked through](foundations/15-backward-pass-worked-through.md) |
| 16 | [the 6ND rule](foundations/16-6nd-rule.md) |
| 17 | [PyTorch autograd mechanics](foundations/17-pytorch-autograd-mechanics.md) |
| 18 | [PyTorch optimizer API](foundations/18-pytorch-optimizer-api.md) |
| 19 | [AdaGrad step() — walkthrough](foundations/19-adagrad-walkthrough.md) |
| 20 | [training memory: four components](foundations/20-training-memory-four-components.md) |
| 21 | [gradient accumulation](foundations/21-gradient-accumulation.md) |
| 22 | [activation checkpointing](foundations/22-activation-checkpointing.md) |

### `transformer-primer/`

| # | topic |
|---|---|
| 03 | [weight initialization](transformer-primer/03-weight-initialization.md) |
| 04 | [attention: motivation](transformer-primer/04-attention-motivation.md) |
| 05 | [attention: Q, K, V](transformer-primer/05-attention-qkv.md) |
| 06 | [attention scores (Q·K)](transformer-primer/06-attention-scores.md) |
| 07 | [softmax](transformer-primer/07-attention-softmax.md) |
| 08 | [weighted sum of V + scaled dot-product formula](transformer-primer/08-attention-output-and-scaled-formula.md) |
| 09 | [multi-head attention](transformer-primer/09-multi-head-attention.md) |
| 10 | [causal masking](transformer-primer/10-causal-mask.md) |
| 11 | [MLP / feed-forward (SwiGLU)](transformer-primer/11-mlp-feedforward.md) |
| 12 | [residual connections](transformer-primer/12-residual-connections.md) |
| 13 | [normalization (RMSNorm, pre-norm)](transformer-primer/13-normalization.md) |
| 14 | [position embeddings (sinusoidal, RoPE)](transformer-primer/14-position-embeddings.md) |
| 15 | [the full transformer block](transformer-primer/15-full-transformer-block.md) |
| 16 | [the full LM: stack + output head](transformer-primer/16-full-lm-stack.md) |
| 17 | [training: cross-entropy on next-token prediction](transformer-primer/17-training-next-token-prediction.md) |

### Standalone

- [GPU: one big matmul beats many small ones](gpu-one-big-matmul.md) —
  kernel launch overhead, tile utilization, memory bandwidth.

## Style

Each note tries to follow the same shape:

1. What it is, in plain English.
2. What problem it solves.
3. The mechanics / math / code.

Concrete numbers and small worked examples over abstract explanations.
