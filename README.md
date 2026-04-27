# All About LLMs

A byte-level BPE tokenizer and (soon) a transformer language model, built
from scratch in PyTorch as a self-study project on language-model internals.

## Layout

```
.
├── tokenizer/        byte-level BPE — see tokenizer/README.md
└── data/             corpora and trained artifacts (gitignored)
    ├── corpus/           raw source text
    └── tokenizer/        trained vocab / merges pickles
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Data

The repo doesn't ship training data. For the TinyStories corpus:

```sh
mkdir -p data/corpus
cd data/corpus
curl -L -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```
