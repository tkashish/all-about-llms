# How PagedAttention fixes the three waste types

Companion note to `01-motivation.md` (the problems) and
`02-block-table.md` (the mechanism). Here we line up each problem
against the specific paged-design feature that eliminates it.

Assumes `block_size = 16`, Llama-2-7B numbers (~524 KB KV per token,
so ~8 MB per block), `max_seq_len = 2048` tokens.

## Fix #1 — Reservation waste

**Problem recap:** naive cache reserves `max_seq_len` tokens per seq
up front. A seq that has generated 100 of a possible 2048 tokens has
~1020 MB locked and empty "just in case."

**Paged fix:** allocation is **lazy**. A block is grabbed only when
the seq crosses into a new `block_size`-token boundary.

**Numbers:**
- Seq has generated 100 tokens → owns `ceil(100/16) = 7 blocks` →
  space for 112 tokens total.
- Reserved-but-unused: **at most 15 tokens' worth** (partial tail of
  the last block) → ~8 MB.
- Compared to naive: ~1020 MB vs ~8 MB → **~125× reduction.**

The bigger point isn't just the ratio. Every byte the seq doesn't own
is immediately available to another seq. In the naive design the 1020
MB was locked. Here, the moment the seq doesn't need it, the pool
has it.

## Fix #2 — Internal fragmentation

**Problem recap:** seq finishes at token 100, never fills its reserved
2048-token slab, ~1020 MB permanently wasted until next allocation.

**Paged fix:** on seq release, every block it owned returns to the
pool. Reusable by a new seq immediately.

**Numbers:**
- Seq ends at token 100 = slot 3 of block index 6. Slots 4–15 of
  block 6 are unused → ~6 MB wasted while the seq lives.
- When the seq is released, even those 6 MB come back.

**Worst-case internal fragmentation per live seq: `block_size - 1`
tokens.** For `block_size = 16`, at most 15 tokens' worth → ~8 MB.
A rounding error on a 2048-token slot.

## Fix #3 — External fragmentation

**Problem recap:** naive cache has 2 GB free but can't fit a 1.5 GB
request because free space is chopped into two 1 GB gaps.

**Paged fix:** every block is identical in size. There is no "this
hole is too small." If N blocks are free, you can allocate exactly N
blocks — anywhere in the pool, in any order. Sequence's block table
records whichever IDs `alloc()` returned.

**Numbers:**
- Free list has 132 blocks. New seq prompt is 800 tokens →
  needs `ceil(800/16) = 50 blocks`. Grab any 50. Done.
- Their physical IDs are scattered; the block table records them.
  Kernel reads via the table; scattering is invisible above.

**External fragmentation is eliminated by construction, not managed.**
This is the clean part of the design. Malloc has to fight for this
with buddy allocators and best-fit heuristics; paging just sidesteps
the problem.

## Summary

| Waste type            | Naive (2048-token slab)   | Paged (block_size=16) |
|-----------------------|---------------------------|-----------------------|
| Reservation           | up to 1948 tokens (~1 GB) | up to 15 tokens (~8 MB) |
| Internal frag (done)  | up to 1948 tokens (~1 GB) | up to 15 tokens (~8 MB) |
| External frag (holes) | unbounded                 | zero                   |

Three problems. One mechanism. That's the whole pitch of
PagedAttention.

## What this unlocks

Remember the 66-vs-330 gap from
`wiki/gpu/05-llama-7b-decode-walkthrough.md`:
H100 compute could feed ~330 concurrent sequences but naive cache
only fit ~66. The paged design closes most of this gap. Sequences
now claim ~exactly what they use; freed memory is immediately
reusable. In practice vLLM gets 2–4× throughput vs naive serving
on the same hardware.

But getting from "memory is efficient" to "throughput is high"
requires one more piece: the **scheduler** — the component that
decides which sequences run in each decode step and when to admit
new ones. That's the next note.
