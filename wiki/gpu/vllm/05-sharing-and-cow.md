# Memory sharing — prefix caching and copy-on-write

The block-table indirection unlocks one more capability beyond fixing
fragmentation: **two sequences can share physical blocks.** Both
their block tables point at the same block IDs, one copy of K/V in
HBM, multiple readers.

## Why causality makes this legal

A subtle-but-critical point. Many people expect sharing to be
"dangerous" because every layer mixes information across tokens.
It isn't dangerous in causal (decoder-only) attention.

**The causal mask:** position `p` in layer L reads K/V from
positions `0..p`, and no further. Position 3's attention output at
every layer depends only on tokens 0..3. Tokens 4..N exist but are
masked out.

Consequence: if two sequences A and B agree on tokens 0..p, then at
**every** layer, their activations and KV at positions 0..p are
byte-identical. Divergence starts at position p+1 (the first
differing token) and only propagates **forward**, never back into
the shared prefix.

One physical block holds KV for all layers of 16 consecutive tokens
(tensor shape `(block_size, 2, L, H, D_head)`). Sharing that block
shares KV for every layer of those 16 tokens. Which is correct, by
the causality argument above.

## Why sharing is worth the engineering

Two distinct situations, handled by the same mechanism.

### Case 1 — Prefix sharing across independent users

Many serving workloads pin a **fixed system prompt** onto every
request. A chat API often starts every conversation with something
like:

> "You are a helpful assistant, answer in markdown, here are 50
> few-shot examples..."

That's often ~1000 tokens, identical for every user.

Without sharing: 100 concurrent users = 100 copies of those 1000
tokens' KV = ~50 GB duplicated on a Llama-2-7B server.

With sharing: 1 copy, all 100 users' block tables point at the same
~63 blocks (1000/16). ~500 MB total. **~100× savings on the prefix.**

Read-only: no branch ever modifies the shared prefix. Simple.

### Case 2 — Branching within one request (beam search / parallel sampling)

A single prompt that produces **multiple** continuations.

- **Parallel sampling** (`n=4`): four independent completions from
  the same prompt. Exposed as `n=` in the OpenAI API,
  `num_return_sequences=` in HuggingFace `generate()`, etc.
- **Beam search** (`num_beams=4`): at each step, keep the top-K most-
  probable continuations, branch, repeat.

Triggered by the API caller, not the server — normal requests
(`n=1`) don't hit this path at all.

Prompt is identical across all branches; only the generated tokens
differ.

Example. Prompt: "Write a haiku about the sea." n=3 samples.
- Branch 1: "Waves crash on the shore..."
- Branch 2: "Endless blue expanse..."
- Branch 3: "Salt wind on my face..."

All 3 share the prompt's KV. They diverge at the first generated
token.

Without sharing: 3× prompt KV. With sharing: 1× prompt KV + 3× small
per-branch tails.

Read-write: branches *will* write different continuations. Sharing
must handle "at what point do they stop reading the same blocks and
start writing their own?" That's copy-on-write.

## Copy-on-write

Problem: two seqs share block B. A has so far filled slots 0..15 of B
(the last token of the shared prefix). Now A wants to generate token
16, which would write to slot 16 of what's now **A's next block**.
B's next block might be different (B generates a different 17th
token) or identical. We don't know yet.

**Solution (borrowed from OS memory):** on the first write into a
shared block, make a copy.

```python
def append_token(seq, k, v):
    slot = seq.num_tokens % BLOCK_SIZE
    if slot == 0:
        # new block needed — just allocate, nothing shared
        seq.block_table.append(pool.alloc())
    else:
        # writing into existing tail block — check sharing
        block_id = seq.block_table[-1]
        if pool.refcount[block_id] > 1:
            # shared — copy-on-write
            new_id = pool.alloc()
            pool.blocks[new_id].copy_(pool.blocks[block_id])
            pool.refcount[block_id] -= 1
            seq.block_table[-1] = new_id
            block_id = new_id
    pool.blocks[block_id, slot] = (k, v)
    seq.num_tokens += 1
```

Key additions to the BlockPool:

- **Refcount per block.** `refcount[block_id]` is the number of
  sequences whose block table references this block.
- **`free()` decrements, not releases.** "Free the block" means
  "decrement refcount; if it hits 0, return to the free list."

This is the exact same algorithm as OS `fork()`. Same data structure,
same name.

## Summary

| Sharing type          | Used for                          | Needs CoW? |
|-----------------------|-----------------------------------|------------|
| Prefix across users   | System prompts, few-shot examples | No (read-only) |
| Prefix within request | Beam search, parallel sampling    | Yes        |

## Closing thought on Phase 1

Five notes now cover what vLLM is and why it works:

1. **Motivation** — the 66-vs-330 gap and three waste types.
2. **Block table** — the data structure, write/read paths.
3. **How paging fixes waste** — each of the three fixed, with numbers.
4. **Continuous batching** — the scheduler that turns freed memory
   into throughput; prefill vs decode.
5. **Sharing and CoW** — the bonus capability the block table unlocks.

Phase 2 built the paged cache in pure PyTorch on top of the existing
transformer. See `06-paged-cache-impl.md` for the implementation
notes (Block / BlockPool / Sequence, the per-layer-counter gotcha,
the 3-way correctness test, measured overhead vs KVCache).
