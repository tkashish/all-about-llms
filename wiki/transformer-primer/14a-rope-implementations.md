# RoPE — implementation approaches

Three mathematically-equivalent ways to implement RoPE. Same output for
the same input, different code shapes, very different speed.

Assumes you already understand what RoPE is (see
`wiki/transformer-primer/14-position-embeddings.md`). This note is about
**how to write the code**, not what it does.

## The setup

You have Q (or K) of shape `(B, H, T, D_head)`. You need to rotate each
2D pair inside `D_head` by an angle `t · f_k` where `t` is the token's
position and `f_k` is the pair's frequency.

The frequencies are fixed (not learned):
```
f_k = 1 / 10000^(2k / D_head)     for k = 0, 1, ..., D_head/2 - 1
```

So for each `(t, k)` pair you need `cos(t · f_k)` and `sin(t · f_k)`.
You precompute these into two lookup tables, shape `(max_T, D_head/2)`:
```
cos_table[t, k] = cos(t · f_k)
sin_table[t, k] = sin(t · f_k)
```

The only question is: **given Q and the cos/sin tables, how do you
actually apply the rotation?**

That's where the three approaches differ.

## Approach 1 — pairwise explicit

(to fill in)

## Approach 2 — split halves (Meta / Llama style)

(to fill in)

## Approach 3 — complex-number rotation

(to fill in)

## Comparison

(to fill in)

## Recommendation

(to fill in)
