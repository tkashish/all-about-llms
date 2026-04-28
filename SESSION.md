# Session checkpoint — 2026-04-28

Load with: `please load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md`

## State

- Repo: `/Users/katayal/Documents/llm/AllAboutLLMs` (git: `main`, remote `github.com/tkashish/all-about-llms`, 2 commits pushed).
- Layout: src-layout with hatchling. Packages: `src/tokenizer/` (done, tested) and `src/transformer/` (user re-coding from scratch).
- Deps in pyproject: torch, numpy, regex, einops. Dev: pytest.
- Data (gitignored): `data/corpus/TinyStoriesV2-GPT4-{train,valid}.txt`, `data/tokenizer/tinystories_{vocab,merges}.pkl`.
- Wiki: `wiki/foundations/` (19 notes), `wiki/transformer-primer/` (15 notes + skeleton `14a-rope-implementations.md`), `wiki/gpu-one-big-matmul.md`.

## User

- Kashish Tayal, self-studying LLM internals. Not a CS336 student; honor code does NOT apply here — but user prefers to **write transformer code themselves** for learning retention. Kiro explains, reviews, debugs. Does NOT write full modules for them.
- Teaching style: one idea per message, ~15 lines max before check-in, plain English → problem → mechanics, concrete numbers > abstract math. Push-back if responses get long.

## Where we are in the re-code

User is rewriting the transformer in `src/transformer/`. Started with `embedding_table.py` + `model.py`. Has asked about:
- Embedding lookup shape `(B,T)` indices + `(vocab_size,D)` table → `(B,T,D)` via fancy indexing.
- Why `/sqrt(d_model)` init for Linear but `*0.02` for Embedding.
- Why residual/norm around every sublayer (not whole block).
- Why no norm after embedding (answered: redundant with `norm1` of block 1, some models like Gemma do add one).

## Dangling threads

1. **SwiGLU walkthrough** — started "gate vs value", stopped after motivating why a learned gate beats hardcoded ReLU. Resume at step 2: introduce the three matrices (W_signal, W_gate, W_2) and the elementwise `SiLU(signal) * gate` step. One idea per message, check-in after each.
2. **RoPE implementations wiki** — skeleton written at `wiki/transformer-primer/14a-rope-implementations.md`, 3 approaches stubbed (pairwise explicit, split halves, complex). User wants to understand trade-offs + recommended approach. Next step: fill in Approach 1 (pairwise explicit) with a worked example.

## Pickup prompt for next session

> "Load /Users/katayal/Documents/llm/AllAboutLLMs/SESSION.md. Continue the SwiGLU explanation at step 2 (the three matrices), OR fill in Approach 1 of the RoPE implementations wiki. Ask me which."
