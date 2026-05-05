# vLLM — motivation

## What vLLM is, plainly

vLLM is an inference server for LLMs. Point it at a model, it opens a
port, many users send prompts concurrently and get responses. That's
the whole shape of the thing.

## The problem it solves

From our earlier analysis (see `wiki/gpu/05-llama-7b-decode-walkthrough.md`):

- **Compute ridge on an H100:** ~330 concurrent sequences would saturate
  the math units.
- **Memory-feasible on an H100:** ~66 sequences fit after weights, with
  a naive KV cache.

That ~5× gap between what the compute wants and what the memory allows
is where GPU money gets burned. vLLM closes it by managing KV cache
memory smarter — same hardware, multiple times the throughput.

## Vocabulary: "sequence"

One user's conversation with the model, as a stream of tokens. Prompt
plus generated output. Each sequence has its **own** KV cache — caches
are per-sequence, not shared. Holding 100 concurrent users = holding
100 separate KV caches in GPU memory.

## The core technique: PagedAttention

Named by analogy to OS paging. Operating systems don't hand programs
one big contiguous chunk of RAM — they hand out fixed-size **pages**
(~4 KB) and maintain a table mapping each program's logical addresses
to physical pages. This lets the OS pack many programs into RAM
without fragmentation.

vLLM does the same for KV cache. Instead of one big contiguous slab
per sequence, break the cache into small fixed-size **blocks**
(typically 16 tokens' worth) and maintain a table mapping each
sequence's logical positions to physical blocks.

That indirection — one extra lookup — is the trick. Everything else
in the paper is engineering around that one idea.

## Why the naive approach wastes memory

The naive design (what our Level 5 cache does today) reserves
`max_seq_len` tokens of KV cache per sequence up front, in a
contiguous chunk. With `max_seq_len = 2048` for Llama-2-7B, that's
1 GB reserved per sequence, contiguous.

This wastes memory three distinct ways. The paper names all three:

### 1. Reservation waste

Memory reserved for tokens not yet generated, which may or may not
be generated. If a sequence has produced 100 of a possible 2048
tokens so far, 1948 tokens' worth of slot is locked and empty "just
in case." Dominant waste for sequences still in-progress.

### 2. Internal fragmentation

A sequence finishes *before* filling its reserved slot. E.g. reserved
2048 tokens, generated 100, hit end-of-sequence — the 1948 unused
tokens' worth of memory was never going to be used.

Difference from reservation waste: "definitely wasted" vs "might
still get used." Functionally identical in terms of locked memory.

Paper's observation: real outputs average ~100–300 tokens while
`max_seq_len` is typically 2048. So **~80–90% of each sequence's
reserved slot ends up as internal fragmentation** by completion.

### 3. External fragmentation

Because each sequence's cache is one **contiguous** slab, freed slots
leave gaps in memory. New requests must fit entirely in one gap.

Example: 10 GB of cache holds 10 sequences. Sequences 3 and 7 finish,
leaving two 1 GB gaps. A new request needing 1.5 GB doesn't fit in
either, even though 2 GB total is free.

Same problem malloc/free faces. OSes solved it with paging — fixed-size
blocks and a lookup table. PagedAttention borrows the same fix.

See `vllm-phase1.excalidraw` (fragmentation section) for diagrams
of all three.

## The takeaway

The naive contiguous + max-reserve design is simple but wasteful.
PagedAttention replaces "contiguous slab per sequence" with
"pool of fixed-size blocks + per-sequence block table," eliminating
reservation waste and external fragmentation by construction.

Next note: how the block table actually maps logical positions to
physical blocks.
