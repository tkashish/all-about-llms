# Tokenizer

Byte-level Byte Pair Encoding (BPE) in pure Python.

## Files

- `tokenizer.py` — `Tokenizer` (encode / decode) and the per-word encoder
- `pre_tokenizer.py` — GPT-2 regex pre-tokenization + parallel chunk processing
- `train.py` — BPE trainer with incremental pair-frequency updates

## Training

```sh
uv run python -m tokenizer.train \
    --input data/corpus/TinyStoriesV2-GPT4-train.txt \
    --vocab-size 10000 \
    --out-prefix data/tokenizer/tinystories
```

Produces:

- `data/tokenizer/tinystories_vocab.pkl` — `dict[int, bytes]`
- `data/tokenizer/tinystories_merges.pkl` — `list[tuple[bytes, bytes]]`

Rough timing on TinyStories (2.1 GB, vocab 10k, 8 CPU cores): ~8 minutes.

## Usage

```python
import pickle
from tokenizer.tokenizer import Tokenizer

with open("data/tokenizer/tinystories_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("data/tokenizer/tinystories_merges.pkl", "rb") as f:
    merges = pickle.load(f)

tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

ids = tok.encode("Once upon a time, there was a little girl.")
text = tok.decode(ids)
```

## Design notes

- **Byte-level.** Vocab starts with all 256 byte values → every UTF-8 string
  can be tokenized without UNK / OOV. Merges operate on `bytes`, not `str`.
- **GPT-2 pre-tokenization regex.** The same `PAT` splits text into
  pre-tokens; BPE only merges **within** a pre-token, never across.
  ```
  '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
  ```
- **Training: incremental pair counts.** Naive BPE rescans the corpus after
  each merge. This trainer maintains a pair-frequency counter and a
  reverse map (pair → set of pre-tokens containing it), and only updates
  affected pre-tokens when a merge is applied.
- **Encoding: merge-rank dict.** Each training merge is numbered by its
  learning order; encoding applies the lowest-ranked (earliest-learned)
  applicable pair at each step. This matches what training would produce.
- **Special tokens.** During training, hard boundaries (split + drop).
  During encoding, preserved as single-ID atomic tokens.
