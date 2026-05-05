# Block tables — PagedAttention's core data structure

Covers how PagedAttention actually represents KV cache memory: the
block pool, the per-sequence block table, and the write/read paths.

## The block

A **block** is a fixed-size chunk of KV cache memory. Always the same
size, for every sequence, for every layer.

Real vLLM uses `block_size = 16` tokens. For Llama-2-7B, each token's
KV cache is ~524 KB (2 × L × H × D_head × 2 bytes across K and V for
all layers and heads), so one block holds 16 × 524 KB ≈ **8 MB** of
KV cache. At startup the GPU memory budget for KV cache is chopped
into thousands of these 8 MB blocks.

### Where does 8 MB per block come from?

The per-token KV cost, worked out for Llama-2-7B:

| Dimension | Value |
|---|---|
| Layers (L) | 32 |
| Heads per layer (H) | 32 |
| Dim per head (D_head) | 128 |
| Bytes per value (fp16) | 2 |
| K **and** V | × 2 |

```
bytes_per_token = L × H × D_head × bytes × 2
                = 32 × 32 × 128 × 2 × 2
                = 524,288 bytes  (≈ 512 KB)

bytes_per_block = 16 × bytes_per_token
                = 8,388,608 bytes  (≈ 8 MB)
```

Every token stores K and V at every layer and every head — that's
where the factors come from. One attention call needs **all** of these
values, for every prior token in the sequence.

### How big is the block pool?

vLLM sizes the pool to eat almost all of HBM left after weights.
For an 80 GB H100 running Llama-2-7B in fp16:

| Component | Size |
|---|---|
| Model weights | ~14 GB |
| CUDA / framework overhead | ~2 GB |
| **Block pool** | **~60+ GB** |

At 8 MB per block, 60 GB ≈ **~7,500 blocks** = ~120,000 tokens worth
of KV cache, sharable across any number of concurrent sequences.

One allocation at startup, never grows. Every spare byte of HBM goes
to the pool because KV cache is the only thing that scales with
concurrent users — weights are fixed.

### Why all layers in one block?

A natural alternative: give each layer its own blocks. One token would
then occupy L = 32 separate pieces of memory instead of one.

Reason the bundled design wins: **a sequence's KV grows at the same
rate for every layer.** One new token = one new K/V slot at every
layer, same instant, same lifetime. Bundling lets one `alloc()` per
16 tokens serve all layers; splitting would mean L allocations and L
free-list updates per 16 tokens. Block-table lookups also jump from
1 to L per attention call.

Cost of bundling: larger blocks (~8 MB vs ~256 KB). Acceptable — total
memory is the same, just coarser granularity.

### What problem does fixed-size solve?

It kills **external fragmentation** by construction. If every block
is the same size, any freed block can serve any future allocation —
no "too small" or "too big" gaps. Same reason OS memory is managed
in 4 KB pages: uniform chunks, no irregular holes.

## The block pool

A single global array of blocks shared across all sequences, plus a
free-list:

```python
class BlockPool:
    blocks: Tensor          # (num_blocks, block_size, 2, L, H, D_head)
                            # the 2 dim is K/V; everything else is standard
    free_list: list[int]    # IDs of blocks currently unused

    def alloc(self) -> int:
        return self.free_list.pop()

    def free(self, block_id: int):
        self.free_list.append(block_id)
```

Preallocated once at startup. The tensor lives in HBM; the free list
lives in CPU memory (it's just integers).

## The block table (per sequence)

A small array whose entries are block IDs:

```
block_table = [block_id_0, block_id_1, block_id_2, ...]
```

Entry `i` says: "my tokens `[i·block_size, i·block_size + block_size - 1]`
live in the physical block at `block_table[i]`."

### What problem does it solve?

Without indirection, a sequence's KV cache would need one contiguous
slab (the malloc problem). With this table, a sequence's blocks can
live anywhere in the pool — scattered, not contiguous. The table
glues them back in order when attention reads.

Cost of indirection: one integer lookup per block per attention call.
Cheap.

## Worked example

Setup: `block_size = 4` (tiny, for the picture), 8-block pool.

```
Sequence A (6 tokens generated):
  block_table_A = [3, 1]
  logical block 0 (tokens 0–3) → physical block 3
  logical block 1 (tokens 4–5, slots 2–3 empty) → physical block 1

Sequence B (5 tokens generated):
  block_table_B = [5, 2]
  logical block 0 (tokens 0–3) → physical block 5
  logical block 1 (token 4 only, slots 1–3 empty) → physical block 2

Physical pool:
  [ 0:free,  1:A-logblock1,  2:B-logblock1,  3:A-logblock0,
    4:free,  5:B-logblock0,  6:free,         7:free ]
```

A's blocks (3, 1) are **not contiguous**. B's blocks (5, 2) are also
not contiguous. That's fine — the block table glues each sequence
back together during attention.

See `vllm-phase1.excalidraw` → SOLUTION section for the diagram.

## Write path: appending a token

```python
class Sequence:
    block_table: list[int]
    num_tokens: int

    def append_token(self, pool: BlockPool, k, v):
        slot_in_block = self.num_tokens % BLOCK_SIZE
        if slot_in_block == 0:
            # crossed a boundary — allocate a new block
            new_block = pool.alloc()
            self.block_table.append(new_block)

        block_id = self.block_table[-1]
        pool.blocks[block_id, slot_in_block] = (k, v)
        self.num_tokens += 1

    def release(self, pool: BlockPool):
        # seq finished; hand blocks back
        for bid in self.block_table:
            pool.free(bid)
        self.block_table.clear()
```

Key properties:
- **Lazy allocation.** A block is grabbed only when the seq crosses into
  a new `block_size`-token boundary. No up-front max-length reservation.
- **Free is O(blocks in this seq).** Just return each to the pool.
- **Block IDs are whatever `alloc()` hands out.** Non-contiguous by nature.

## Read path: attention uses the block table

During decode, attention needs K and V for every prior token:

```python
def gather_kv(seq: Sequence, pool: BlockPool):
    k_chunks, v_chunks = [], []
    for logical_block_idx, physical_block_id in enumerate(seq.block_table):
        block = pool.blocks[physical_block_id]
        # shape (block_size, 2, L, H, D_head)

        if logical_block_idx == len(seq.block_table) - 1:
            # last block may be partially filled
            valid = seq.num_tokens - logical_block_idx * BLOCK_SIZE
            block = block[:valid]

        k_chunks.append(block[:, 0])
        v_chunks.append(block[:, 1])

    return torch.cat(k_chunks), torch.cat(v_chunks)

# Standard attention from here on:
K, V = gather_kv(seq, pool)
scores = Q @ K.transpose(-2, -1) / sqrt(d_head)
probs = softmax(scores + causal_mask)
out = probs @ V
```

Attention doesn't care that blocks are scattered in physical memory.
The `block_table` loop walks them in logical order, so the assembled
K and V look exactly like a contiguous cache would.

### Cost this introduces

Every attention call does a Python loop over blocks plus a `torch.cat`.
For pure-PyTorch (Phase 2), fine — correctness first.

For Phase 3 we replace the gather with a **single fused Triton kernel**
that reads directly from `pool.blocks` using the block table as an
index. No Python loop, no concat, no materialized K/V tensor. That's
the actual "PagedAttention kernel" — the algorithm above rewritten
to happen inside one kernel launch.

## Up next

With this machinery in place, the three waste types from
`01-motivation.md` are all fixed. The next note walks through each:
reservation → lazy allocation eliminates it; internal fragmentation
→ bounded to at most one partial block per seq (tiny); external
fragmentation → impossible by construction.
